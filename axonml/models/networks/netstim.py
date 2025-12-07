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
    A PyTorch implementation of NEURON's NetStim-like spike generator.

    This class generates spike events according to a stochastic process with
    configurable timing parameters, similar to NEURON's NetStim mechanism.
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
        """
        Initialize the NetStim spike generator.

        Parameters
        ----------
        N : int, optional
            Number of independent event generators. Default is 1.
        interval : float | list[float], optional
            Mean inter-spike interval in ms. Default is 10.0.
        start : float | list[float], optional
            Start time (ms) after which synapses can begin spiking. Default is 0.0.
        noise : float | list[float], optional
            Controls randomness of intervals, between 0 and 1. Default is 0.0.
        max_spikes : int | list[int], optional
            Maximum number of spikes each synapse can deliver. Default is 1e9.
        seed : int, optional
            Seed for reproducible random number generation. If None,
            uses non-deterministic seeding. Default is None.
        """
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
        return self.next_stoch_time.device

    def dtype(self):
        return self.next_stoch_time.dtype

    def init_rng(self):
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
        Initializes next_stoch_time; scheduled heaps are preserved.
        If noise>0: first stochastic spike = start + Exp(mean=noise*interval).
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
        """Remove all scheduled spikes for selected generators (or all if None)."""
        if indices is None:
            indices = range(self.N)
        for i in indices:
            self._sched_heaps[i].clear()
        self._refresh_next_sched_time_tensor()

    @torch.no_grad()
    @torch._dynamo.disable()  # keep everything here out of Dynamo/Inductor
    def _consume_scheduled_tensor(self, fired_idx: torch.Tensor):
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
        U = torch.rand((self.N,), generator=self._rng, device=device, dtype=dtype)
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
            self._buffers["next_stoch_time"] = new_stoch
        else:
            # no BPTT: commit numerically but don't grow graph
            with torch.no_grad():
                self.next_stoch_time.copy_(new_stoch)  # copy into original storage

        #    b) scheduled: heap pops are side-effects; keep them out of the graph
        fired_idx = torch.nonzero(s_sched, as_tuple=True)[0]
        torch._dynamo.graph_break()
        self._consume_scheduled_tensor(fired_idx)

        # 4) counters (not part of the computational graph)
        with torch.no_grad():
            self.spike_counts.add_(s_hard.to(torch.long))
            self.t_last.copy_(t)

        return self.spikes

    def numel(self):
        """
        Returns the total number of elements in the spike generator.
        """
        return self.N

    def batch(self, n):
        """
        Replicate this NetStim n times (independent copies).
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
        with torch.no_grad():
            self.next_stoch_time = self.next_stoch_time.detach()
            self.next_sched_time = self.next_sched_time.detach()
            self.spike_counts = self.spike_counts.detach()
            self.spikes = self.spikes.detach()
            self.spike_gate = self.spike_gate.detach()
            self.t_last = self.t_last.detach()
        return self

    def detach_(self):
        self.detach()
