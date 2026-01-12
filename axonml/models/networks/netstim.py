import heapq
from typing import Iterable, Optional

import torch

from ..parametric import PositiveParam
from ..slice import Sliceable


def _ste_gate(x, tau):
    s_soft = torch.sigmoid(x / tau)  # smooth [0,1]
    hard = (x >= 0).to(s_soft.dtype)  # hard 0/1
    gate = hard + (s_soft - s_soft.detach())  # forward==hard, backward like soft
    return gate, hard


class NetStim(torch.nn.Module, Sliceable):
    """
    Differentiable spike generator modeled after NEURON's NetStim.

    Each generator produces a renewal process with mean interval ``interval``.
    A ``noise`` parameter blends deterministic intervals with exponential
    variability:

    .. math::

        T_{k+1} = T_k + (1 - \\eta)\\,\\Delta + \\eta\\,\\Delta\\,E,

    where ``Delta = interval``, ``eta = noise`` in ``[0,1]``, and ``E`` is
    unit-rate exponential. When ``eta=0`` firing is periodic; when ``eta=1``
    firing is Poisson with rate ``1/interval``. Initial spikes occur at
    ``start``. A straight-through estimator gate provides a hard spike mask for
    simulation while preserving gradients through the soft sigmoid surface,
    enabling differentiation w.r.t. interval parameters.

    Parameters
    ----------
    N : int, optional
        Number of independent generators. Default is 1.
    interval : float or Iterable[float], optional
        Mean inter-spike interval (ms). Can be per-generator. Default is 10.0.
    start : float or Iterable[float], optional
        Start time (ms) after which spikes may occur. Default is 0.0.
    noise : float or Iterable[float], optional
        Randomness in [0, 1]; 0=deterministic, 1=Poisson. Default is 0.0.
    max_spikes : int or Iterable[int], optional
        Maximum spikes per generator. Default is 1e9.
    tau : float, optional
        Sigmoid temperature for the straight-through gate (ms). Default is 0.1.
    seed : int, optional
        Seed for reproducible randomness. Default is None.
    """

    __constants__ = ["seed"]

    def __init__(
        self,
        N: int = 1,
        interval: float | Iterable[float] = 10.0,
        start: float | Iterable[float] = 0.0,
        noise: float | Iterable[float] = 0.0,
        max_spikes: int | Iterable[int] = 1e9,
        tau: float = 0.1,
        seed: Optional[int] = None,
    ):
        super().__init__()
        Sliceable.__init__(self)
        self._validate_parameters(N, interval, start, noise, max_spikes)
        self.name = "netstim"

        self.N = N
        self.tau = tau
        self.shape = (N,)

        self.register_buffer("noise", torch.as_tensor(noise).float())
        self.register_buffer("start", torch.as_tensor(start).float())
        self.register_buffer("start_cache", self.start.clone())
        self.register_buffer(
            "max_spikes", torch.as_tensor(max_spikes, dtype=torch.long)
        )
        self.interval = PositiveParam(interval, threshold=20.0)
        self.noise.clamp_(0.0, 1.0)

        self.seed: Optional[int] = seed
        self._seeder = torch.Generator()
        self._rng = torch.Generator().manual_seed(self._seeder.seed())

        # ——— split clocks: stochastic vs scheduled ———
        self.register_buffer(
            "next_stoch_time", torch.zeros(self.N)
        )  # stochastic process
        self.register_buffer(
            "next_sched_time", torch.full((self.N,), float("inf"))
        )  # next explicit spike or +inf

        self.register_buffer("spike_counts", torch.zeros(self.N, dtype=torch.long))
        self.register_buffer("spikes", torch.zeros(self.N, dtype=torch.bool))
        self.register_buffer(
            "t_last", torch.tensor(-float("inf"))
        )  # last time seen in forward()

        self.spike_gate = torch.zeros(self.N)  # differentiable spike gate

        # Per-generator min-heaps of future scheduled times (CPU-side metadata)
        self._sched_heaps: list[list[float]] = [[] for _ in range(self.N)]

    # ───────────────────────── misc API ─────────────────────────
    def device(self):
        """
        Device on which the generator state lives.

        Returns
        -------
        torch.device
            Device of internal buffers.
        """
        return self.next_stoch_time.device

    def dtype(self):
        """
        Data type used by generator state.

        Returns
        -------
        torch.dtype
            Dtype of internal buffers.
        """
        return self.next_stoch_time.dtype

    def init_rng(self):
        """
        Initialize or reseed the internal RNG on the correct device.

        If ``seed`` was provided on initialization, the RNG is made deterministic;
        otherwise a device-local generator is created with nondeterministic seeding.
        """
        if self._rng.device != self.device():
            self._rng = torch.Generator(device=self.device()).manual_seed(
                self._seeder.seed()
            )
        if self.seed is not None:
            self._rng.manual_seed(self.seed)

    def _prep_start_for_steady_state(self, tstop):
        self.start.fill_(-tstop + self.start_cache)

    def _reset_start_times(self):
        self.start.detach().copy_(self.start_cache)

    def _validate_parameters(
        self,
        N: int,
        interval: float | Iterable[float],
        start: float | Iterable[float],
        noise: float | Iterable[float],
        max_spikes: int | Iterable[int],
    ):
        if N <= 0:
            raise ValueError(
                "Number of independent event generators (N) must be positive."
            )
        if isinstance(interval, Iterable) and len(interval) != N:
            raise ValueError(
                f"Interval must be a single value or a list of length {N}."
            )
        if isinstance(start, Iterable) and len(start) != N:
            raise ValueError(f"Start must be a single value or a list of length {N}.")
        if isinstance(noise, Iterable) and len(noise) != N:
            raise ValueError(f"Noise must be a single value or a list of length {N}.")
        if isinstance(max_spikes, Iterable) and len(max_spikes) != N:
            raise ValueError(
                f"Max spikes must be a single value or a list of length {N}."
            )

    # ───────────────────────── init/reset ─────────────────────────
    def initialize(self):
        """
        Reset generator state while preserving scheduled spikes.

        Notes
        -----
        - ``next_stoch_time`` is set to ``start`` plus an exponential offset
          when ``noise>0`` so the first interval is drawn from the same renewal
          distribution used during stepping.
        - Scheduled spikes already enqueued via :meth:`schedule` are retained.
        """
        self.init_rng()
        device, dtype = self.device(), self.dtype()

        with torch.no_grad():
            # start time baseline
            self.next_stoch_time.copy_(self.start)

            if torch.any(self.noise > 0):
                U = torch.rand(
                    (self.N,), generator=self._rng, device=device, dtype=dtype
                )
                # Use a detached interval here; we don't want grads through this random kick
                interval0 = self.interval(cache=(not self.training)).detach()
                init_offsets = -(self.noise * interval0) * torch.log(U)
                self.next_stoch_time.add_(init_offsets)

            # recompute next_sched_time from heaps
            self._refresh_next_sched_time_tensor()

            # spike_counts: how many spikes each synapse has emitted
            self.spike_counts.zero_()
            self.spikes.zero_()
            self.spike_gate = self.spike_gate.detach().zero_().to(dtype=dtype)
            self.t_last.fill_(-float("inf"))

        return self

    # ───────────────────────── scheduling ─────────────────────────
    def _refresh_next_sched_time_tensor(self):
        # Pull heap tops into device tensor (O(N))
        vals = []
        for h in self._sched_heaps:
            vals.append(h[0] if h else float("inf"))
        self.next_sched_time = torch.as_tensor(
            vals, device=self.device(), dtype=self.dtype()
        )

    @torch.no_grad()
    def schedule(self, indices, times):
        """
        Add explicit spike times per generator without overwriting existing ones.
        Allows duplicates and multiple times per generator, e.g. indices=[0,0,0], times=[1.0,2.0,3.0].

        - Ignores times <= t_last (already in the past given monotonic t).
        - Complexity: O(k log m) inserts total, where m is current # scheduled per generator.

        Parameters
        ----------
        indices : int or Iterable[int]
            Generator indices to receive scheduled spikes.
        times : float or Iterable[float]
            Spike times (ms) aligned with ``indices``. Past times are ignored.
        """
        # normalize to python lists on CPU for heapq
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().tolist()
        if isinstance(times, torch.Tensor):
            times = times.detach().cpu().tolist()
        if not isinstance(indices, Iterable):
            indices = [indices]
        if not isinstance(times, Iterable):
            times = [times]
        if len(indices) != len(times):
            raise ValueError("indices and times must have the same length")

        t_cut = float(self.t_last.item())  # don't schedule into the past
        for i, t in zip(indices, times):
            if not (0 <= i < self.N):
                raise IndexError(f"Generator index {i} out of range [0,{self.N})")
            if not (t > t_cut):  # skip retroactive times
                continue
            heapq.heappush(self._sched_heaps[i], float(t))

        # update device-side next_sched_time only for changed rows
        # (simple full refresh; micro-opt: touch only rows that received inserts)
        self._refresh_next_sched_time_tensor()

    @torch.no_grad()
    def clear_schedule(self, indices: Optional[Iterable[int]] = None):
        """
        Remove all scheduled spikes for selected generators (or all if None).

        Parameters
        ----------
        indices : Iterable[int], optional
            Generators whose schedules should be cleared. Default None (all).
        """
        if indices is None:
            indices = range(self.N)
        for i in indices:
            self._sched_heaps[i].clear()
        self._refresh_next_sched_time_tensor()

    @torch.no_grad()
    @torch._dynamo.disable()  # keep everything here out of Dynamo/Inductor
    def _consume_scheduled_tensor(self, s_sched: torch.Tensor):
        fired_idx = torch.nonzero(s_sched, as_tuple=True)[0]
        # fired_idx is a 1D Long tensor of generator indices (may be empty)
        if fired_idx is None or fired_idx.numel() == 0:
            return
        idx_list = fired_idx.detach().cpu().tolist()
        for i in idx_list:
            if self._sched_heaps[i]:
                heapq.heappop(self._sched_heaps[i])
            head = self._sched_heaps[i][0] if self._sched_heaps[i] else float("inf")
            # write back to device tensor
            self.next_sched_time[i] = head

    def forward(self, t, *, bptt: bool = False):
        """
        Advance generator clocks to time ``t`` and emit spike mask.

        The spike decision uses a straight-through estimator:

        - Soft gate: ``sigmoid((t - t_next)/tau)`` (provides gradients).
        - Hard mask: ``t >= t_next`` (used for simulation).
        - Combined with ``max_spikes`` constraint.

        For stochastic intervals, the next arrival is sampled as

        .. math::

            \\Delta t = (1-\\eta)\\,\\Delta + \\eta\\,\\Delta\\,E, \\quad E\\sim \\text{Exp}(1)

        so gradients flow to ``interval`` through the linear mixing term.

        Parameters
        ----------
        t : float or torch.Tensor
            Current simulation time (ms).
        bptt : bool, optional
            If True, keep graph connections across steps (no detach) to enable
            backprop-through-time. Default is False.

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape ``(N,)`` indicating which generators fired.
        """
        device, dtype = self.device(), self.dtype()
        t = torch.as_tensor(t, device=device, dtype=dtype)

        # SNAPSHOT (avoid using live buffers directly in the graph)
        stoch_snap = (
            self.next_stoch_time if bptt else self.next_stoch_time.detach()
        ).clone()
        sched_snap = (
            self.next_sched_time.detach().clone()
        )  # treat schedule as constant for grads

        # 1) differentiable gating (keep this under grad!)
        next_combined = torch.minimum(stoch_snap, sched_snap)
        x = t - next_combined
        gate, s_hard = _ste_gate(x, tau=self.tau)

        can_spike = self.spike_counts < self.max_spikes
        s_hard = torch.logical_and(s_hard, can_spike)
        self.spikes = s_hard  # bool view; fine to keep as is
        self.spike_gate = gate * can_spike.to(gate.dtype)  # keep grad if used in loss

        # 2) stochastic interval draw (grad will flow to interval via this)
        eps = torch.finfo(dtype).tiny
        U = torch.rand(
            (self.N,), generator=self._rng, device=device, dtype=dtype
        ).clamp_min(eps)
        exp_rand = -torch.log(U)
        interval = self.interval(cache=(not self.training))
        next_interval = interval * (1 - self.noise) + interval * self.noise * exp_rand

        # Split origin of spikes (scheduled vs stochastic)
        from_sched = torch.logical_and(
            sched_snap <= stoch_snap, torch.isfinite(sched_snap)
        )
        s_sched = torch.logical_and(s_hard, from_sched)
        # s_stoch = s_hard & (~from_sched)

        # 3) advance clocks
        #    a) stochastic: choose whether to backprop-through-time
        delta = (
            next_interval
            * self.spike_gate
            * torch.logical_not(from_sched).to(gate.dtype)
        )

        new_stoch = stoch_snap + delta  # <- no in-place on the buffer used in 'minimum'

        # COMMIT state safely (no version-bump hazards)
        if bptt:
            # keep graph across steps: replace the buffer with the new tensor
            self.next_stoch_time = new_stoch
        else:
            # no BPTT: commit numerically but don't grow graph
            with torch.no_grad():
                self.next_stoch_time.copy_(new_stoch)  # copy into original storage

        # b) scheduled: heap pops are side-effects; keep them out of the graph
        torch._dynamo.graph_break()
        self._consume_scheduled_tensor(s_sched)

        # 4) counters (not part of the computational graph)
        with torch.no_grad():
            self.spike_counts.add_(s_hard.to(torch.long))
            self.t_last.copy_(t)

        return self.spikes

    def numel(self):
        """
        Total number of generators.

        Returns
        -------
        int
            Number of generators (N).
        """
        return self.N

    def batch(self, n):
        """
        Replicate this NetStim ``n`` times (independent copies).

        Parameters
        ----------
        n : int
            Batch size multiplier (must be positive).

        Returns
        -------
        NetStim
            Self, with state expanded to ``N*n`` generators.
        """
        if n <= 0:
            raise ValueError("Batch size n must be positive.")
        device, dtype = self.device(), self.dtype()

        self.N *= n
        self.shape = (self.N,)

        self.noise = self.noise.repeat(n)
        self.start = self.start.repeat(n)
        self.max_spikes = self.max_spikes.repeat(n)
        self.interval = self.interval.repeat(n)

        # expand state tensors
        self.next_stoch_time = torch.zeros(self.N, device=device, dtype=dtype)
        self.next_sched_time = torch.full(
            (self.N,), float("inf"), device=device, dtype=dtype
        )
        self.spike_counts = torch.zeros(self.N, device=device, dtype=torch.long)
        self.spikes = torch.zeros(self.N, device=device, dtype=torch.bool)
        self.spike_gate = torch.zeros(self.N, device=device, dtype=dtype)

        # expand heaps
        self._sched_heaps = [list(h) for h in self._sched_heaps for _ in range(n)]
        return self

    def detach(self):
        """
        Detach state tensors from any computation graph.

        Returns
        -------
        NetStim
            Self, detached in-place.
        """
        with torch.no_grad():
            self.next_stoch_time = self.next_stoch_time.detach()
            self.next_sched_time = self.next_sched_time.detach()
            self.spike_counts = self.spike_counts.detach()
            self.spikes = self.spikes.detach()
            self.spike_gate = self.spike_gate.detach()
            self.t_last = self.t_last.detach()
        return self

    def detach_(self):
        """In-place variant of :meth:`detach` that returns ``None``."""
        self.detach()

    # ───────────────────────── checkpointing ─────────────────────────
    def state_dict_for_checkpoint(self):
        """Return a replay-safe snapshot of NetStim *dynamic* state.

        This is intended for activation checkpointing in long unrolled runs.
        Key points:

        - We MUST include the per-instance torch.Generator RNG state, because
          torch.utils.checkpoint can preserve global RNG state, but not custom
          Generator objects passed via `generator=...`.
        - Scheduled spikes live in Python-side heaps. We store them as an
          immutable tuple-of-tuples so the returned state object is not
          accidentally mutated by subsequent simulation steps.

        Returns
        -------
        dict
            A lightweight state snapshot that can be passed through
            torch.utils.checkpoint and used to deterministically restore the
            NetStim internal state.
        """
        # Store scheduled heaps as immutable data (avoid aliasing/mutation).
        sched_heaps = tuple(tuple(h) for h in self._sched_heaps)

        return {
            # Differentiable state (needed for BPTT across chunks)
            "next_stoch_time": self.next_stoch_time,
            # Non-differentiable state (counters/logical guards)
            "spike_counts": self.spike_counts,
            "t_last": self.t_last,
            # Scheduled-spike state
            "sched_heaps": sched_heaps,
            # RNG state (critical for deterministic checkpoint replay)
            "rng_state": self._rng.get_state(),
            "seeder_state": self._seeder.get_state(),
        }

    def restore_dict_from_checkpoint(self, state_dict):
        """Restore NetStim dynamic state from :meth:`state_dict_for_checkpoint`.

        Important details for activation checkpointing:

        - We rebind/clone any buffers that are mutated in-place during forward
          (e.g., spike_counts, t_last) to avoid mutating checkpoint inputs.
        - We rebind next_stoch_time directly in training mode to preserve its
          autograd history (BPTT across chunks).
        - We restore the torch.Generator state so stochastic renewal sampling
          is replay-identical during backward recomputation.

        Parameters
        ----------
        state_dict:
            A dict produced by :meth:`state_dict_for_checkpoint`.
        """
        device, dtype = self.device(), self.dtype()
        # 1) Restore RNGs.
        #    torch.utils.checkpoint can preserve *global* RNG state, but it does
        #    not handle custom per-module Generators. We must restore it here.
        if "seeder_state" in state_dict:
            self._seeder.set_state(state_dict["seeder_state"])
        # Ensure the sampling generator exists on the correct device.
        # (Module `.to(...)` does not automatically move torch.Generator.)
        if getattr(self._rng, "device", torch.device("cpu")) != device:
            self._rng = torch.Generator(device=device)
        if "rng_state" in state_dict:
            self._rng.set_state(state_dict["rng_state"])
        else:
            # Best-effort fallback.
            self.init_rng()

        # 2) Restore scheduled spikes (Python-side heaps) and rebuild
        #    the derived device tensor ``next_sched_time``.
        heaps_in = state_dict.get("sched_heaps", None)
        if heaps_in is not None:
            self._sched_heaps = [list(h) for h in heaps_in]
            for h in self._sched_heaps:
                heapq.heapify(h)
            self._refresh_next_sched_time_tensor()
        elif "next_sched_time" in state_dict:
            # Backward-compatible path if an older checkpoint stored the tensor.
            self.next_sched_time = (
                state_dict["next_sched_time"]
                .detach()
                .to(device=device, dtype=dtype)
                .clone()
            )

        # 3) Restore differentiable stochastic state.
        #    Preserve gradient connectivity across chunks in training mode.
        ns = state_dict["next_stoch_time"]
        if ns.device != device or ns.dtype != dtype:
            ns = ns.to(device=device, dtype=dtype)

        if self.training:
            # Rebind so downstream steps see the exact tensor with its
            # autograd history.
            self.next_stoch_time = ns
        else:
            # Eval path: keep the existing buffer object and copy values in.
            with torch.no_grad():
                self.next_stoch_time.copy_(ns)

        # 4) Restore non-differentiable counters/buffers.
        #    These are mutated in-place during forward, so we rebind to clones
        #    to avoid aliasing checkpoint inputs.
        sc = state_dict.get("spike_counts", None)
        if sc is not None:
            self.spike_counts = sc.detach().to(device=device, dtype=torch.long).clone()

        tl = state_dict.get("t_last", None)
        if tl is not None:
            self.t_last = tl.detach().to(device=device, dtype=dtype).clone()

        # 5) Reset per-step outputs (they will be recomputed on the next call).
        self.spikes = torch.zeros(self.N, device=device, dtype=torch.bool)
        self.spike_gate = torch.zeros(self.N, device=device, dtype=dtype)
