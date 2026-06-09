import heapq
from collections.abc import Iterable
from numbers import Integral
from typing import Optional

import torch

from ..modular import DNModule
from ..parametric import PositiveParam
from ..slice import Sliceable


def _ste_gate(x, tau):
    s_soft = torch.sigmoid(x / tau)  # smooth [0,1]
    hard = (x >= 0).to(s_soft.dtype)  # hard 0/1
    gate = hard + (s_soft - s_soft.detach())  # forward==hard, backward like soft
    return gate, hard


class NetStim(DNModule, Sliceable):
    r"""
    Differentiable spike generator modeled after NEURON's NetStim.

    The last tensor dimension indexes the ``N`` independent NetStim generators.
    Any leading dimensions are treated as batch dimensions. For example, an
    unbatched instance with ``N == i`` has state/parameter shape ``(i,)``;
    after ``batch(n)`` the shape is ``(n, i)``. Calling ``batch`` again prepends
    another leading batch dimension rather than flattening generators.

    Each generator produces a renewal process with mean interval ``interval``.
    A ``noise`` parameter blends deterministic intervals with exponential
    variability:

    .. math::

        T_{k+1} = T_k + (1 - \eta)\,\Delta + \eta\,\Delta\,E,

    where ``Delta = interval``, ``eta = noise`` in ``[0,1]``, and ``E`` is
    unit-rate exponential. When ``eta=0`` firing is periodic; when ``eta=1``
    firing is Poisson with rate ``1/interval``. Initial spikes occur at
    ``start``. A straight-through estimator gate provides a hard spike mask for
    simulation while preserving gradients through the soft sigmoid surface,
    enabling differentiation w.r.t. interval parameters.

    Parameters
    ----------
    N : int, optional
        Number of independent generators in the final dimension. Default is 1.
    interval : float or Iterable[float] or torch.Tensor, optional
        Mean inter-spike interval (ms). Can be scalar, per-generator with shape
        ``(N,)``, or broadcastable to a shape ending in ``N``. Default is 10.0.
    start : float or Iterable[float] or torch.Tensor, optional
        Start time (ms) after which spikes may occur. Can be scalar,
        per-generator, or batched/broadcastable. Default is 0.0.
    noise : float or Iterable[float] or torch.Tensor, optional
        Randomness in [0, 1]; 0=deterministic, 1=Poisson. Can be scalar,
        per-generator, or batched/broadcastable. Default is 0.0.
    max_spikes : int or Iterable[int] or torch.Tensor, optional
        Maximum spikes per generator. Can be scalar, per-generator, or
        batched/broadcastable. Default is 1e9.
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
        interval_t, start_t, noise_t, max_spikes_t, shape = (
            self._canonicalize_parameters(
                N=N,
                interval=interval,
                start=start,
                noise=noise,
                max_spikes=max_spikes,
            )
        )
        self.name = "netstim"

        self.N = int(shape[-1])
        self.tau = tau
        self.shape = tuple(shape)

        self.register_buffer("noise", noise_t)
        self.register_buffer("start", start_t)
        self.register_buffer("start_cache", self.start.clone())
        self.register_buffer("max_spikes", max_spikes_t)
        self.interval = PositiveParam(interval_t, threshold=20.0)
        self.noise.clamp_(0.0, 1.0)

        self.seed: Optional[int] = seed
        self._seeder = torch.Generator()
        self._rng = torch.Generator().manual_seed(self._seeder.seed())

        # ——— split clocks: stochastic vs scheduled ———
        self.register_buffer(
            "next_stoch_time", torch.zeros(self.shape)
        )  # stochastic process
        self.register_buffer(
            "next_sched_time", torch.full(self.shape, float("inf"))
        )  # next explicit spike or +inf

        self.register_buffer("spike_counts", torch.zeros(self.shape, dtype=torch.long))
        self.register_buffer("spikes", torch.zeros(self.shape, dtype=torch.bool))
        self.register_buffer(
            "t_last", torch.full(self.shape, -float("inf"))
        )  # last time seen in forward(), per state element

        self.spike_gate = torch.zeros(self.shape)  # differentiable spike gate

        # Per-state-element min-heaps of future scheduled times (CPU metadata).
        # The flat order is row-major over ``self.shape``; the last dimension is
        # the generator index, all earlier dimensions are batch coordinates.
        self._sched_heaps: list[list[float]] = [
            [] for _ in range(self._flat_numel_from_shape(self.shape))
        ]

    # ───────────────────────── shape helpers ─────────────────────────
    @staticmethod
    def _flat_numel_from_shape(shape: tuple[int, ...]) -> int:
        n = 1
        for dim in shape:
            n *= int(dim)
        return int(n)

    @staticmethod
    def _canonicalize_parameters(
        *,
        N: int,
        interval,
        start,
        noise,
        max_spikes,
    ):
        if N <= 0:
            raise ValueError(
                "Number of independent event generators (N) must be positive."
            )

        interval_t = torch.as_tensor(interval, dtype=torch.float32)
        start_t = torch.as_tensor(start, dtype=torch.float32)
        noise_t = torch.as_tensor(noise, dtype=torch.float32)
        max_spikes_t = torch.as_tensor(max_spikes, dtype=torch.long)

        try:
            shape = torch.broadcast_shapes(
                torch.Size((int(N),)),
                interval_t.shape,
                start_t.shape,
                noise_t.shape,
                max_spikes_t.shape,
            )
        except RuntimeError as exc:
            raise ValueError(
                "interval, start, noise, and max_spikes must be scalar, have "
                f"shape ({N},), or be mutually broadcastable to a shape ending "
                f"in {N}."
            ) from exc

        shape = tuple(int(d) for d in shape)
        if len(shape) == 0 or shape[-1] != int(N):
            raise ValueError(
                "NetStim parameter tensors must broadcast to a non-empty shape "
                f"whose final dimension is N={N}. Got broadcast shape {shape}."
            )

        interval_t = interval_t.expand(shape).clone()
        start_t = start_t.expand(shape).clone()
        noise_t = noise_t.expand(shape).clone()
        max_spikes_t = max_spikes_t.expand(shape).clone()

        return interval_t, start_t, noise_t, max_spikes_t, shape

    @staticmethod
    def _normalize_batch_dims(n) -> tuple[int, ...]:
        if isinstance(n, torch.Tensor):
            if n.ndim == 0:
                dims = (int(n.item()),)
            else:
                dims = tuple(int(v) for v in n.detach().cpu().reshape(-1).tolist())
        elif isinstance(n, Integral):
            dims = (int(n),)
        elif isinstance(n, Iterable) and not isinstance(n, (str, bytes)):
            dims = tuple(int(v) for v in n)
        else:
            raise TypeError("Batch size must be an int or an iterable of ints.")

        if len(dims) == 0:
            raise ValueError("At least one batch dimension is required.")
        if any(dim <= 0 for dim in dims):
            raise ValueError("All batch dimensions must be positive.")
        return dims

    @staticmethod
    def _prepend_batch_dims_to_tensor(x: torch.Tensor, dims: tuple[int, ...]):
        old_shape = tuple(x.shape)
        return x.reshape((1,) * len(dims) + old_shape).expand(dims + old_shape).clone()

    def _flat_numel(self) -> int:
        return self._flat_numel_from_shape(self.shape)

    def _batch_numel(self) -> int:
        return self._flat_numel() // self.N

    def _set_shape_metadata(self, shape: tuple[int, ...]):
        if len(shape) == 0:
            raise ValueError("NetStim state shape must be non-empty.")
        if any(int(dim) <= 0 for dim in shape):
            raise ValueError(f"Invalid NetStim state shape {shape}.")
        self.shape = tuple(int(dim) for dim in shape)
        self.N = int(self.shape[-1])

    def _broadcast_to_state(
        self,
        value,
        *,
        dtype: torch.dtype,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        value = torch.as_tensor(value, device=device, dtype=dtype)
        try:
            return torch.broadcast_to(value, self.shape)
        except RuntimeError as direct_exc:
            if value.ndim > 0:
                try:
                    return torch.broadcast_to(value.unsqueeze(-1), self.shape)
                except RuntimeError:
                    pass
            raise ValueError(
                f"{name} with shape {tuple(value.shape)} cannot be broadcast "
                f"to NetStim state shape {self.shape}."
            ) from direct_exc

    def _broadcast_time_to_state(self, t) -> torch.Tensor:
        device, dtype = self.device(), self.dtype()
        t = torch.as_tensor(t, device=device, dtype=dtype)

        if tuple(t.shape) == self.shape:
            return t
        # A time tensor with exactly the leading batch shape is interpreted as
        # one time per batch element and is broadcast over generators.
        if len(self.shape) > 1 and tuple(t.shape) == self.shape[:-1]:
            return t.unsqueeze(-1)

        try:
            return torch.broadcast_to(t, self.shape)
        except RuntimeError as direct_exc:
            if t.ndim > 0:
                try:
                    return torch.broadcast_to(t.unsqueeze(-1), self.shape)
                except RuntimeError:
                    pass
            raise ValueError(
                f"t with shape {tuple(t.shape)} cannot be broadcast to "
                f"NetStim state shape {self.shape}. Use a scalar, shape "
                f"{self.shape[:-1]} for per-batch times, or shape {self.shape}."
            ) from direct_exc

    def _positive_param_from_value(self, value: torch.Tensor):
        threshold = getattr(self.interval, "threshold", None)
        if threshold is None:
            threshold = getattr(self.interval, "_threshold", 20.0)

        requires = [p.requires_grad for p in self.interval.parameters()]
        requires_grad = any(requires) if requires else True

        try:
            param = PositiveParam(value.detach().clone(), threshold=threshold)
        except TypeError:
            # Fallback for PositiveParam implementations that do not expose the
            # threshold keyword in their constructor.
            param = PositiveParam(value.detach().clone())
        param.to(device=value.device)
        for p in param.parameters():
            p.requires_grad_(requires_grad)
        return param

    # ───────────────────────── parameter control ─────────────────────────
    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        return self

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad_(True)
        return self

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
        with torch.no_grad():
            tstop = self._broadcast_time_to_state(tstop)
            self.start.copy_(self.start_cache - torch.broadcast_to(tstop, self.shape))

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
        # Kept for external callers/backward compatibility. Construction uses
        # _canonicalize_parameters so that scalar, per-generator, and batched
        # broadcastable values are all handled uniformly.
        self._canonicalize_parameters(
            N=N,
            interval=interval,
            start=start,
            noise=noise,
            max_spikes=max_spikes,
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
            noise = self._broadcast_to_state(
                self.noise, dtype=dtype, device=device, name="noise"
            )
            start = self._broadcast_to_state(
                self.start, dtype=dtype, device=device, name="start"
            )

            # start time baseline
            self.next_stoch_time.copy_(start)

            if torch.any(noise > 0):
                eps = torch.finfo(dtype).tiny
                U = torch.rand(
                    self.shape, generator=self._rng, device=device, dtype=dtype
                ).clamp_min(eps)
                # Use a detached interval here; we don't want grads through this random kick.
                interval0 = self._broadcast_to_state(
                    self.interval().detach(),
                    dtype=dtype,
                    device=device,
                    name="interval",
                )
                init_offsets = -(noise * interval0) * torch.log(U)
                self.next_stoch_time.add_(init_offsets)

            # recompute next_sched_time from heaps
            self._refresh_next_sched_time_tensor()

            # spike_counts: how many spikes each state element has emitted
            self.spike_counts.zero_()
            self.spikes.zero_()
            self.spike_gate = torch.zeros(self.shape, device=device, dtype=dtype)
            self.t_last.fill_(-float("inf"))

        return self

    # ───────────────────────── scheduling ─────────────────────────
    def _refresh_next_sched_time_tensor(self):
        # Pull heap tops into a device tensor (O(prod(shape))).
        expected = self._flat_numel()
        if len(self._sched_heaps) != expected:
            raise RuntimeError(
                f"Scheduled heap count ({len(self._sched_heaps)}) does not match "
                f"NetStim state shape {self.shape} ({expected} elements)."
            )
        vals = [h[0] if h else float("inf") for h in self._sched_heaps]
        self.next_sched_time = torch.as_tensor(
            vals, device=self.device(), dtype=self.dtype()
        ).reshape(self.shape)

    @staticmethod
    def _normalize_dim_index(i: int, size: int, label: str) -> int:
        if i < 0:
            i += size
        if not (0 <= i < size):
            raise IndexError(f"{label} index {i} out of range [0,{size})")
        return i

    def _ravel_index(self, index: tuple[int, ...]) -> int:
        if len(index) != len(self.shape):
            raise IndexError(
                f"Expected an index with {len(self.shape)} dimensions for "
                f"NetStim shape {self.shape}; got {index}."
            )
        flat = 0
        for dim, size in zip(index, self.shape):
            dim = self._normalize_dim_index(int(dim), int(size), "State")
            flat = flat * int(size) + dim
        return int(flat)

    def _normalize_index_specs(self, indices) -> list[int | tuple[int, ...]]:
        if isinstance(indices, torch.Tensor):
            idx = indices.detach().cpu()
            if idx.ndim == 0:
                return [int(idx.item())]
            if idx.ndim == 1:
                return [int(v) for v in idx.tolist()]
            if idx.ndim == 2 and idx.shape[1] == len(self.shape):
                return [tuple(int(v) for v in row) for row in idx.tolist()]
            raise IndexError(
                "Tensor indices must be scalar, 1D generator indices, or a "
                f"2D tensor with trailing dimension {len(self.shape)}."
            )

        if isinstance(indices, Integral):
            return [int(indices)]

        if isinstance(indices, tuple) and all(isinstance(v, Integral) for v in indices):
            if len(indices) == len(self.shape):
                return [tuple(int(v) for v in indices)]
            if len(indices) == 1:
                return [int(indices[0])]
            raise IndexError(
                f"Tuple index {indices} must have length 1 or {len(self.shape)} "
                f"for NetStim shape {self.shape}."
            )

        if isinstance(indices, Iterable) and not isinstance(indices, (str, bytes)):
            specs: list[int | tuple[int, ...]] = []
            for item in indices:
                if isinstance(item, torch.Tensor):
                    item = item.detach().cpu()
                    if item.ndim == 0:
                        specs.append(int(item.item()))
                    elif item.ndim == 1 and item.numel() == len(self.shape):
                        specs.append(tuple(int(v) for v in item.tolist()))
                    elif item.ndim == 1 and item.numel() == 1:
                        specs.append(int(item.item()))
                    else:
                        raise IndexError(
                            f"Invalid tensor index item with shape {tuple(item.shape)}."
                        )
                elif isinstance(item, Integral):
                    specs.append(int(item))
                elif isinstance(item, (tuple, list)) and all(
                    isinstance(v, Integral) for v in item
                ):
                    if len(item) == len(self.shape):
                        specs.append(tuple(int(v) for v in item))
                    elif len(item) == 1:
                        specs.append(int(item[0]))
                    else:
                        raise IndexError(
                            f"Index {item} must have length 1 or {len(self.shape)} "
                            f"for NetStim shape {self.shape}."
                        )
                else:
                    raise TypeError(f"Unsupported schedule index {item!r}.")
            return specs

        raise TypeError(f"Unsupported schedule indices {indices!r}.")

    def _flat_indices_for_spec(self, spec: int | tuple[int, ...]) -> list[int]:
        if isinstance(spec, Integral):
            # A bare integer always denotes a generator index in the final
            # dimension. In batched mode it applies to every leading batch item.
            gen = self._normalize_dim_index(int(spec), self.N, "Generator")
            return [
                batch_flat * self.N + gen for batch_flat in range(self._batch_numel())
            ]
        return [self._ravel_index(tuple(int(v) for v in spec))]

    @staticmethod
    def _normalize_schedule_times(times, num_specs: int):
        if isinstance(times, torch.Tensor):
            if times.ndim == 0:
                out = [float(times.detach().cpu().item())]
            else:
                out = [float(v) for v in times.detach().cpu().reshape(-1).tolist()]
        elif isinstance(times, Iterable) and not isinstance(times, (str, bytes)):
            out = [float(v) for v in times]
        else:
            out = [float(times)]

        if len(out) == 0:
            raise ValueError("At least one scheduled time is required.")
        if len(out) == 1 and num_specs > 1:
            out = out * num_specs
        return out

    @torch.no_grad()
    def schedule(self, indices, times):
        """
        Add explicit spike times without overwriting existing ones.

        Unbatched usage is backward-compatible: integer indices refer to
        generator indices in ``[0, N)``. In batched mode, a bare integer still
        refers to a generator in the final dimension and is applied to every
        leading batch element. To schedule a single batched element, pass a full
        coordinate tuple matching ``self.shape``, e.g. ``(batch_index, gen)`` for
        shape ``(batch, N)``.

        Allows duplicates and multiple times per generator. Past times are
        ignored per state element using that element's ``t_last``.

        Parameters
        ----------
        indices : int, tuple[int, ...], or Iterable
            Generator indices or full state coordinates to receive scheduled
            spikes.
        times : float or Iterable[float]
            Spike times (ms) aligned with ``indices``. A scalar time broadcasts
            across multiple indices; multiple times with one index schedule all
            times onto that index.
        """
        specs = self._normalize_index_specs(indices)
        times_list = self._normalize_schedule_times(times, len(specs))

        if len(specs) == 1 and len(times_list) > 1:
            specs = specs * len(times_list)
        elif len(specs) != len(times_list):
            raise ValueError(
                "indices and times must have the same length, unless one of them "
                "is scalar/broadcastable."
            )

        t_last_flat = self.t_last.detach().reshape(-1).cpu()
        for spec, t in zip(specs, times_list):
            for flat_i in self._flat_indices_for_spec(spec):
                t_cut = float(t_last_flat[flat_i].item())
                if not (float(t) > t_cut):  # skip retroactive times
                    continue
                heapq.heappush(self._sched_heaps[flat_i], float(t))

        self._refresh_next_sched_time_tensor()

    @torch.no_grad()
    def clear_schedule(self, indices: Optional[Iterable[int]] = None):
        """
        Remove scheduled spikes for selected state elements, or all if ``None``.

        In batched mode, bare integer indices clear a generator across all
        leading batch elements. Full coordinate tuples clear one state element.
        """
        if indices is None:
            flat_indices = range(self._flat_numel())
        else:
            specs = self._normalize_index_specs(indices)
            flat_indices = []
            for spec in specs:
                flat_indices.extend(self._flat_indices_for_spec(spec))

        for flat_i in sorted(set(int(i) for i in flat_indices)):
            self._sched_heaps[flat_i].clear()
        self._refresh_next_sched_time_tensor()

    @torch.no_grad()
    @torch._dynamo.disable()  # keep everything here out of Dynamo/Inductor
    def _consume_scheduled_tensor(self, s_sched: torch.Tensor):
        fired_idx = torch.nonzero(s_sched.reshape(-1), as_tuple=True)[0]
        if fired_idx is None or fired_idx.numel() == 0:
            return

        idx_list = fired_idx.detach().cpu().tolist()
        next_sched_flat = self.next_sched_time.reshape(-1)
        for flat_i in idx_list:
            if self._sched_heaps[flat_i]:
                heapq.heappop(self._sched_heaps[flat_i])
            head = (
                self._sched_heaps[flat_i][0]
                if self._sched_heaps[flat_i]
                else float("inf")
            )
            next_sched_flat[flat_i] = head

    def forward(self, t, *, bptt: bool = False):
        r"""
        Advance generator clocks to time ``t`` and emit a spike mask.

        ``t`` can be a scalar, a tensor broadcastable to ``self.shape``, or a
        tensor with shape equal to the leading batch dimensions ``self.shape[:-1]``
        to provide one time per batch element.

        The spike decision uses a straight-through estimator:

        - Soft gate: ``sigmoid((t - t_next)/tau)`` (provides gradients).
        - Hard mask: ``t >= t_next`` (used for simulation).
        - Combined with ``max_spikes`` constraint.

        For stochastic intervals, the next arrival is sampled as

        .. math::

            \Delta t = (1-\eta)\,\Delta + \eta\,\Delta\,E, \quad E\sim \text{Exp}(1)

        so gradients flow to ``interval`` through the linear mixing term.

        Parameters
        ----------
        t : float or torch.Tensor
            Current simulation time (ms). Scalar, per-batch, or per-state.
        bptt : bool, optional
            If True, keep graph connections across steps (no detach) to enable
            backprop-through-time. Default is False.

        Returns
        -------
        torch.Tensor
            Boolean tensor with shape ``self.shape`` indicating which generators
            fired for each batch element.
        """
        device, dtype = self.device(), self.dtype()
        t = self._broadcast_time_to_state(t)

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

        max_spikes = self._broadcast_to_state(
            self.max_spikes, dtype=torch.long, device=device, name="max_spikes"
        )
        can_spike = self.spike_counts < max_spikes
        s_hard = torch.logical_and(s_hard, can_spike)
        self.spikes = s_hard  # bool view; fine to keep as is
        self.spike_gate = gate * can_spike.to(gate.dtype)  # keep grad if used in loss

        # 2) stochastic interval draw (grad will flow to interval via this)
        eps = torch.finfo(dtype).tiny
        U = torch.rand(
            self.shape, generator=self._rng, device=device, dtype=dtype
        ).clamp_min(eps)
        exp_rand = -torch.log(U)
        interval = self._broadcast_to_state(
            self.interval(), dtype=dtype, device=device, name="interval"
        )
        noise = self._broadcast_to_state(
            self.noise, dtype=dtype, device=device, name="noise"
        )
        next_interval = interval * (1 - noise) + interval * noise * exp_rand

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
            t_full = torch.broadcast_to(t.detach(), self.shape)
            if tuple(self.t_last.shape) == self.shape:
                self.t_last.copy_(t_full)
            else:
                self.t_last = t_full.clone()

        return self.spikes

    def numel(self):
        """
        Total number of NetStim state elements, including batch dimensions.

        Returns
        -------
        int
            ``prod(self.shape)``.
        """
        return self._flat_numel()

    def batch(self, n):
        """
        Prepend one or more batch dimensions to this NetStim in-place.

        Unlike the previous flattening behavior, ``batch(n)`` transforms an
        unbatched shape ``(N,)`` into ``(n, N)``. Existing leading dimensions are
        preserved, so another ``batch(m)`` gives ``(m, n, N)``. Each replicated
        state element receives its own stochastic draw and schedule heap.

        Parameters
        ----------
        n : int or Iterable[int]
            Batch dimension(s) to prepend. All must be positive.

        Returns
        -------
        NetStim
            Self, with parameters and state expanded over the new leading batch
            dimensions.
        """
        dims = self._normalize_batch_dims(n)
        old_shape = self.shape
        new_shape = dims + old_shape

        # Preserve the current device/dtype and replicate current values. This
        # makes batching safe before or after initialize(), schedule(), or BPTT
        # state construction.
        for name in (
            "noise",
            "start",
            "start_cache",
            "max_spikes",
            "next_stoch_time",
            "spike_counts",
            "spikes",
            "t_last",
        ):
            setattr(
                self,
                name,
                self._prepend_batch_dims_to_tensor(getattr(self, name), dims),
            )

        self.interval = self.interval.batch(n)

        old_heaps = self._sched_heaps
        repeat_count = self._flat_numel_from_shape(dims)
        self._sched_heaps = [list(h) for _ in range(repeat_count) for h in old_heaps]

        self._set_shape_metadata(new_shape)
        self.spike_gate = self._prepend_batch_dims_to_tensor(self.spike_gate, dims).to(
            device=self.device(), dtype=self.dtype()
        )
        self._refresh_next_sched_time_tensor()
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
        - ``shape`` is stored so batched checkpoint restore can rebuild the
          flat schedule heap layout.

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
            # Shape metadata (needed because leading batch dims are meaningful)
            "shape": self.shape,
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

        if "shape" in state_dict:
            self._set_shape_metadata(tuple(int(d) for d in state_dict["shape"]))

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
            if len(heaps_in) != self._flat_numel():
                raise ValueError(
                    f"Checkpoint has {len(heaps_in)} schedule heaps, but current "
                    f"NetStim shape {self.shape} requires {self._flat_numel()}."
                )
            self._sched_heaps = [list(h) for h in heaps_in]
            for h in self._sched_heaps:
                heapq.heapify(h)
            self._refresh_next_sched_time_tensor()
        elif "next_sched_time" in state_dict:
            # Backward-compatible path if an older checkpoint stored the tensor.
            self.next_sched_time = self._broadcast_to_state(
                state_dict["next_sched_time"].detach(),
                dtype=dtype,
                device=device,
                name="next_sched_time",
            ).clone()

        # 3) Restore differentiable stochastic state.
        #    Preserve gradient connectivity across chunks in training mode.
        ns = state_dict["next_stoch_time"]
        ns_input_shape = tuple(ns.shape)
        if ns.device != device or ns.dtype != dtype:
            ns = ns.to(device=device, dtype=dtype)
        ns = self._broadcast_to_state(
            ns, dtype=dtype, device=device, name="next_stoch_time"
        )

        if self.training:
            # Rebind so downstream steps see the tensor with autograd history.
            # Materialize broadcasted views so later non-BPTT copy_ commits are safe.
            self.next_stoch_time = ns if ns_input_shape == self.shape else ns.clone()
        else:
            # Eval path: keep the existing buffer object when possible.
            with torch.no_grad():
                if tuple(self.next_stoch_time.shape) == self.shape:
                    self.next_stoch_time.copy_(ns)
                else:
                    self.next_stoch_time = ns.detach().clone()

        # 4) Restore non-differentiable counters/buffers.
        #    These are mutated in-place during forward, so we rebind to clones
        #    to avoid aliasing checkpoint inputs.
        sc = state_dict.get("spike_counts", None)
        if sc is not None:
            self.spike_counts = self._broadcast_to_state(
                sc.detach(), dtype=torch.long, device=device, name="spike_counts"
            ).clone()

        tl = state_dict.get("t_last", None)
        if tl is not None:
            self.t_last = self._broadcast_to_state(
                tl.detach(), dtype=dtype, device=device, name="t_last"
            ).clone()

        # 5) Reset per-step outputs (they will be recomputed on the next call).
        self.spikes = torch.zeros(self.shape, device=device, dtype=torch.bool)
        self.spike_gate = torch.zeros(self.shape, device=device, dtype=dtype)
