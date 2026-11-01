import copy
import heapq
import math
from collections.abc import Iterable
from numbers import Integral, Real
from typing import Optional

import torch

from dendra.helpers import _normalize_dtype_value, current_device, current_dtype

from ..modular import DNModule
from ..parametric import PositiveParam
from ..slice import Sliceable
from ._heap_calendar import (
    heap_calendar_consume,
    heap_calendar_dtype,
    heap_calendar_head,
    pack_heap_calendar,
    validate_heap_calendar,
)
from ._netstim_parameters import (
    broadcast_netstim_parameter,
    evaluate_netstim_interval,
    materialize_netstim_parameters,
)
from ._netstim_renewal import (
    NetStimState,
    broadcast_netstim_time,
    netstim_initial_state,
    netstim_renewal,
    sample_netstim_exponential,
)
from ._tensor_schedules import causal_step_index as _causal_step_index
from ._tensor_schedules import netstim_tensor_schedule, resolve_schedule_values


def _clone_checkpoint_state(value, memo=None):
    """Clone nested NetStim runtime state while preserving tensor history."""
    if memo is None:
        memo = {}
    if torch.is_tensor(value):
        value_id = id(value)
        if value_id not in memo:
            memo[value_id] = value.clone()
        return memo[value_id]
    if isinstance(value, dict):
        return {key: _clone_checkpoint_state(item, memo) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_checkpoint_state(item, memo) for item in value)
    if isinstance(value, list):
        return [_clone_checkpoint_state(item, memo) for item in value]
    return copy.deepcopy(value)


def _matching_tensor_schedule_configuration(previous, current):
    """Cold comparison used to preserve plans across matching carry restores."""
    for name in ("diff_scheduled_times", "sched_width"):
        if previous[name] != current[name]:
            return False
    for name in (
        "sched_flat_idx",
        "sched_abs_step",
        "sched_time_ms",
        "sched_weight",
        "sched_weight_idx",
        "sched_time_idx",
        "schedule_dt",
    ):
        left, right = previous[name], current[name]
        if (
            left.shape != right.shape
            or left.dtype != right.dtype
            or left.device != right.device
        ):
            return False
        if left.is_floating_point():
            if not bool(torch.all((left == right) | (left.isnan() & right.isnan()))):
                return False
        elif not torch.equal(left, right):
            return False
    return True


def _matching_heap_configuration(previous, current):
    """Runtime cursor movement does not change the immutable calendar."""
    if previous["shape"] != current["shape"]:
        return False
    if "heap_times" in previous and "heap_times" in current:
        return all(
            previous[name].shape == current[name].shape
            and previous[name].dtype == current[name].dtype
            and previous[name].device == current[name].device
            and torch.equal(previous[name], current[name])
            for name in ("heap_times", "heap_row_offsets")
        )
    if ("heap_times" in previous) != ("heap_times" in current):
        return False
    for left, right in zip(previous["sched_heaps"], current["sched_heaps"]):
        if len(left) != len(right) or any(
            a != b and not (math.isnan(a) and math.isnan(b))
            for a, b in zip(left, right)
        ):
            return False
    return True


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
    firing is Poisson with rate ``1/interval``. The initial clock is
    ``start + noise * interval * E``, or ``start`` when ``noise=0``; a sampled
    call emits the spike at or after that clock. A straight-through estimator
    gate provides a hard spike mask for
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
        device=None,
        dtype=None,
    ):
        super().__init__()
        Sliceable.__init__(self)
        init_device = (
            current_device(torch.device("cpu"))
            if device is None
            else torch.device(device)
        )
        init_dtype = (
            current_dtype(torch.float32)
            if dtype is None
            else _normalize_dtype_value(dtype)
        )
        interval_t, start_t, noise_t, max_spikes_t, shape = (
            self._canonicalize_parameters(
                N=N,
                interval=interval,
                start=start,
                noise=noise,
                max_spikes=max_spikes,
                device=init_device,
                dtype=init_dtype,
            )
        )
        self.name = "netstim"
        self._renewal_initialized = False
        self._renewal_generation = 0
        # Canonical configuration mutations invalidate frozen lowering plans.
        # Runtime clock/RNG restores leave this revision alone when unscheduled.
        self._renewal_config_generation = 0

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
            "next_stoch_time",
            torch.zeros(self.shape, device=init_device, dtype=init_dtype),
        )  # stochastic process
        self.register_buffer(
            "next_sched_time",
            torch.full(self.shape, float("inf"), device=init_device, dtype=init_dtype),
        )  # next explicit spike or +inf

        self.register_buffer(
            "spike_counts",
            torch.zeros(self.shape, device=init_device, dtype=torch.long),
        )
        self.register_buffer(
            "spikes", torch.zeros(self.shape, device=init_device, dtype=torch.bool)
        )
        self.register_buffer(
            "t_last",
            torch.full(self.shape, -float("inf"), device=init_device, dtype=init_dtype),
        )  # last time seen in forward(), per state element

        self.spike_gate = torch.zeros(
            self.shape, device=init_device, dtype=init_dtype
        )  # differentiable spike gate

        # A cached compact calendar is shared with functional execution. Cold
        # heap layouts retain original precision when MPS needs lower precision.
        rows = self._flat_numel_from_shape(self.shape)
        self.register_buffer(
            "heap_times",
            torch.full(
                (1,),
                float("inf"),
                device=init_device,
                dtype=heap_calendar_dtype(init_device, init_dtype),
            ),
        )
        self.register_buffer(
            "heap_row_offsets",
            torch.zeros(rows + 1, device=init_device, dtype=torch.long),
        )
        self.register_buffer(
            "heap_cursor", torch.zeros(rows, device=init_device, dtype=torch.long)
        )
        self.register_buffer(
            "heap_head_override",
            torch.full(self.shape, float("inf"), device=init_device, dtype=init_dtype),
        )
        self.register_buffer(
            "heap_override_active",
            torch.zeros(self.shape, device=init_device, dtype=torch.bool),
        )
        self.heap_depth = 0
        # Cold-only original layout preserves later heapq NaN ordering; the
        # numerical runtime reads only tensor calendar buffers and cursors.
        self._heap_legacy_rows = [[] for _ in range(rows)]
        # heapq has insertion-dependent NaN ordering. Keep that exceptional
        # legacy path native-only instead of silently sorting away barriers.
        self._heap_nan_heaps = None

        # Differentiable / weighted scheduled events remain separate from the
        # consuming exact unit-amplitude calendar above. Tensor schedules let
        # gradients flow
        # through scheduled times and amplitudes when NetCon consumes
        # ``NetStim.spike_gate`` with ``diff_spiking=True``.
        self.diff_scheduled_times = True
        self.sched_width = 1.0  # triangular-kernel half-width in steps
        self.register_buffer(
            "schedule_dt", torch.tensor(1.0, device=init_device, dtype=init_dtype)
        )
        self._sched_w_source = None
        self._sched_t_source = None
        self.register_buffer(
            "sched_flat_idx", torch.empty(0, device=init_device, dtype=torch.long)
        )
        self.register_buffer(
            "sched_abs_step", torch.empty(0, device=init_device, dtype=torch.long)
        )
        self.register_buffer(
            "sched_weight_idx", torch.empty(0, device=init_device, dtype=torch.long)
        )
        self.register_buffer(
            "sched_time_idx", torch.empty(0, device=init_device, dtype=torch.long)
        )
        # Value-mode schedule tensors are not buffers; this preserves autograd
        # links to tensors supplied by the caller, matching NetCon scheduling.
        self.sched_time_ms = torch.empty(0, device=init_device, dtype=init_dtype)
        self.sched_weight = torch.empty(0, device=init_device, dtype=init_dtype)

    def __deepcopy__(self, memo):
        """Clone configuration/runtime values without retaining a live BPTT graph.

        PyTorch's default module deepcopy rejects non-leaf tensors. NetStim can
        legitimately own those after differentiable stochastic or scheduled
        execution, both in registered buffers and in ordinary tensor
        attributes such as ``spike_gate``. Seed the deepcopy memo with detached
        value copies, matching Population's structural-clone contract.
        """
        existing = memo.get(id(self))
        if existing is not None:
            return existing

        for module in self.modules():
            for value in module._buffers.values():
                if torch.is_tensor(value) and not value.is_leaf:
                    memo.setdefault(id(value), value.detach().clone())
            for value in vars(module).values():
                if torch.is_tensor(value) and not value.is_leaf:
                    memo.setdefault(id(value), value.detach().clone())

        result = self.__class__.__new__(self.__class__)
        memo[id(self)] = result
        # PyTorch installs a serialization-only __getstate__ blocker on
        # generated parametrization classes; structural clones retain their
        # registered originals and transforms through Module's ordinary state.
        state = copy.deepcopy(torch.nn.Module.__getstate__(self), memo)
        result.__setstate__(state)
        return result

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
        device=None,
        dtype=torch.float32,
    ):
        if N <= 0:
            raise ValueError(
                "Number of independent event generators (N) must be positive."
            )

        interval_t = torch.as_tensor(interval, device=device, dtype=dtype)
        start_t = torch.as_tensor(start, device=device, dtype=dtype)
        noise_t = torch.as_tensor(noise, device=device, dtype=dtype)
        max_spikes_t = torch.as_tensor(max_spikes, device=device, dtype=torch.long)

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
        return broadcast_netstim_parameter(
            value, self.shape, device=device, dtype=dtype, name=name
        )

    def renewal_parameters(self):
        """Derive renewal inputs from their current registered raw sources."""
        return materialize_netstim_parameters(
            evaluate_netstim_interval(self.interval),
            self.start,
            self.noise,
            self.tau,
            self.shape,
            device=self.device(),
            dtype=self.dtype(),
        )

    def _broadcast_time_to_state(self, t) -> torch.Tensor:
        return broadcast_netstim_time(
            t, self.shape, device=self.device(), dtype=self.dtype()
        )

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

    def _align_schedule_tensors(self):
        """Keep differentiable schedule metadata on the NetStim device."""
        device, dtype = self.device(), self.dtype()
        for name in (
            "sched_flat_idx",
            "sched_abs_step",
            "sched_weight_idx",
            "sched_time_idx",
        ):
            if hasattr(self, name):
                buf = getattr(self, name)
                if buf.device != device or buf.dtype != torch.long:
                    setattr(self, name, buf.to(device=device, dtype=torch.long))
        if hasattr(self, "sched_time_ms"):
            self.sched_time_ms = self.sched_time_ms.to(device=device, dtype=dtype)
        if hasattr(self, "sched_weight"):
            self.sched_weight = self.sched_weight.to(device=device, dtype=dtype)
        return self

    def to(self, *args, **kwargs):  # type: ignore[override]
        super().to(*args, **kwargs)
        self._align_schedule_tensors()
        return self

    def _apply(self, fn, recurse=True):
        # Parent Network.to()/float()/half() recurse through _apply, bypassing
        # NetStim.to. Rebuild from original calendar precision on these paths,
        # including moves to/from MPS, which cannot allocate Float64 tensors.
        times = self._buffers.pop("heap_times", None)
        try:
            result = super()._apply(fn, recurse=recurse)
            # Value-mode schedule tensors are ordinary attributes so their
            # graphs still reach caller-owned leaves. Parent module moves and
            # float()/half() need the same alignment as direct NetStim.to().
            self._align_schedule_tensors()
        finally:
            if times is not None:
                rows = (
                    self._heap_legacy_rows
                    if self._heap_nan_heaps is None
                    else [[] for _ in self._heap_legacy_rows]
                )
                self._buffers["heap_times"] = pack_heap_calendar(
                    rows,
                    device=self.next_stoch_time.device,
                    simulation_dtype=self.dtype(),
                )[0]
                # Reconstruct the derived cache from original event precision.
                # Casting an already-rounded head through float().double()
                # would otherwise disagree with functional calendar lookup.
                # Explicit legacy overrides retain their ordinary buffer casts.
                self._update_heap_head()
        return result

    def set_dt(self, dt):
        """Set the timestep used to interpret differentiable scheduled times."""
        self.schedule_dt = torch.as_tensor(
            dt, device=self.device(), dtype=self.dtype()
        ).reshape(())
        self._renewal_config_generation += 1
        return self

    def set_diff_config(
        self,
        *,
        diff_scheduled_times: bool = True,
        sched_width: float = 1.0,
        **_,
    ):
        """Configure differentiable tensor-backed scheduled events.

        This mirrors the scheduled-time portion of ``NetCon.set_diff_config``.
        Tensor-backed schedules use ``spike_gate`` for differentiable amplitudes;
        downstream NetCons must use ``diff_spiking=True`` to consume that gate.
        """
        self.diff_scheduled_times = bool(diff_scheduled_times)
        self.sched_width = float(sched_width)
        self._renewal_config_generation += 1
        return self

    def bind_weight_source(self, source: torch.Tensor):
        """Bind a tensor that provides scheduled-event amplitudes by index."""
        if source.device != self.device():
            raise ValueError("weight source must live on the NetStim device")
        self._sched_w_source = source
        self._renewal_config_generation += 1
        return self

    def bind_time_source(self, source: torch.Tensor):
        """Bind a tensor that provides scheduled-event times, in ms, by index."""
        if source.device != self.device():
            raise ValueError("time source must live on the NetStim device")
        self._sched_t_source = source
        self._renewal_config_generation += 1
        return self

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
        self._renewal_initialized = False
        self.init_rng()
        device, dtype = self.device(), self.dtype()
        initialization_draws = vars(self).get("_native_initialization_draws")

        with torch.no_grad():
            noise = self._broadcast_to_state(
                self.noise, dtype=dtype, device=device, name="noise"
            )
            start = self._broadcast_to_state(
                self.start, dtype=dtype, device=device, name="start"
            )

            if torch.any(noise > 0):
                if initialization_draws is None:
                    eps = torch.finfo(dtype).tiny
                    U = torch.rand(
                        self.shape, generator=self._rng, device=device, dtype=dtype
                    ).clamp_min(eps)
                else:
                    exp_rand = initialization_draws.netstim_draw(self, noise)
                # Use a detached interval here; we don't want grads through this random kick.
                interval0 = self._broadcast_to_state(
                    evaluate_netstim_interval(self.interval).detach(),
                    dtype=dtype,
                    device=device,
                    name="interval",
                )
                if initialization_draws is None:
                    exp_rand = -torch.log(U)
            else:
                # Deterministic initialization must not consume an RNG draw.
                interval0 = torch.zeros_like(start)
                exp_rand = (
                    torch.zeros_like(start)
                    if initialization_draws is None
                    else initialization_draws.netstim_draw(self, noise)
                )

            state = netstim_initial_state(interval0, start, noise, exp_rand)
            self.next_stoch_time = self.next_stoch_time.detach()
            self.next_stoch_time.copy_(state.next_stoch_time)

            # recompute next_sched_time from heaps
            self._refresh_next_sched_time_tensor()

            # spike_counts: how many spikes each state element has emitted
            self.spike_counts.copy_(state.spike_counts)
            self.spikes.zero_()
            self.spike_gate = torch.zeros(self.shape, device=device, dtype=dtype)
            self.t_last.copy_(state.t_last)

        if initialization_draws is not None:
            initialization_draws.complete_netstim(self)

        self._renewal_generation += 1
        self._renewal_initialized = True
        return self

    # ───────────────────────── scheduling ─────────────────────────
    @property
    def _sched_heaps(self):
        """Cold compatibility view of remaining legacy heaps (independent lists)."""
        return self._materialize_heap_rows()

    def _materialize_heap_rows(self):
        if self._heap_nan_heaps is not None:
            return [list(row) for row in self._heap_nan_heaps]
        starts = self.heap_row_offsets[:-1].detach().cpu().tolist()
        cursors = self.heap_cursor.detach().cpu().tolist()
        rows = [list(row) for row in self._heap_legacy_rows]
        # Retaining the original heap layout matters when NaN is inserted
        # later: heapq's non-total ordering depends on that exact layout.
        for row, start, cursor in zip(rows, starts, cursors):
            for _ in range(cursor - start):
                heapq.heappop(row)
        return rows

    def _replace_heap_rows(self, rows):
        """Rebuild after a cold insertion, clear, batch, or legacy import."""
        rows = [[float(value) for value in row] for row in rows]
        if len(rows) != self._flat_numel():
            raise ValueError(
                f"Checkpoint has {len(rows)} schedule heaps, but current "
                f"NetStim shape {self.shape} requires {self._flat_numel()}."
            )
        nan_rows = any(math.isnan(float(value)) for row in rows for value in row)
        self._heap_legacy_rows = [list(row) for row in rows]
        self._heap_nan_heaps = [list(row) for row in rows] if nan_rows else None
        packed = [[] for _ in rows] if nan_rows else rows
        self.heap_times, self.heap_row_offsets, self.heap_cursor, self.heap_depth = (
            pack_heap_calendar(
                packed, device=self.device(), simulation_dtype=self.dtype()
            )
        )
        self._refresh_next_sched_time_tensor()

    def _update_heap_head(self):
        if self._heap_nan_heaps is not None:
            values = [row[0] if row else float("inf") for row in self._heap_nan_heaps]
            head = torch.tensor(values, device=self.device(), dtype=self.dtype())
        else:
            head = heap_calendar_head(
                self.heap_times,
                self.heap_row_offsets[1:],
                self.heap_cursor,
                dtype=self.dtype(),
            )
        self.next_sched_time = torch.where(
            self.heap_override_active,
            self.heap_head_override,
            head.reshape(self.shape),
        )

    def _refresh_next_sched_time_tensor(self):
        # Cold refresh intentionally drops old head-only checkpoint overrides.
        self.heap_override_active = torch.zeros(
            self.shape, device=self.device(), dtype=torch.bool
        )
        self.heap_head_override = torch.full(
            self.shape, float("inf"), device=self.device(), dtype=self.dtype()
        )
        self._update_heap_head()

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

    def _normalize_schedule_tensor(
        self, value, num_specs: int, *, name: str, dtype=None
    ):
        """Return a 1-D device tensor that is scalar or aligned to specs."""
        device = self.device()
        dtype = self.dtype() if dtype is None else dtype
        if torch.is_tensor(value):
            out = value.to(device=device, dtype=dtype).reshape(-1)
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            out = torch.as_tensor(list(value), device=device, dtype=dtype).reshape(-1)
        else:
            out = torch.as_tensor([value], device=device, dtype=dtype).reshape(-1)
        if out.numel() == 0:
            raise ValueError(f"At least one scheduled {name} is required.")
        if out.numel() == 1 and num_specs > 1:
            out = out.expand(num_specs)
        return out

    def _expand_schedule_specs_to_flat_tensors(
        self, specs, values: dict[str, torch.Tensor]
    ):
        """Expand generator specs to flat indices and repeat values accordingly."""
        if len(specs) == 0:
            raise ValueError("At least one scheduled index is required.")
        n_specs = len(specs)
        target_len = max([n_specs] + [int(v.numel()) for v in values.values()])
        if n_specs == 1 and target_len > 1:
            specs = specs * target_len
            n_specs = target_len

        aligned = {}
        for name, value in values.items():
            if value.numel() == 1 and n_specs > 1:
                aligned[name] = value.expand(n_specs)
            elif value.numel() == n_specs:
                aligned[name] = value
            else:
                raise ValueError(
                    f"{name} must be scalar or have length {n_specs}; got {value.numel()}."
                )

        flat_chunks = []
        value_chunks = {name: [] for name in aligned}
        for i, spec in enumerate(specs):
            flat_i = torch.as_tensor(
                self._flat_indices_for_spec(spec),
                device=self.device(),
                dtype=torch.long,
            )
            if flat_i.numel() == 0:
                continue
            flat_chunks.append(flat_i)
            for name, value in aligned.items():
                value_chunks[name].append(value[i].expand(flat_i.numel()))

        if not flat_chunks:
            flat = torch.empty(0, device=self.device(), dtype=torch.long)
            return flat, {
                name: torch.empty(0, device=self.device(), dtype=value.dtype)
                for name, value in aligned.items()
            }
        return torch.cat(flat_chunks, dim=0), {
            name: torch.cat(chunks, dim=0) for name, chunks in value_chunks.items()
        }

    def _should_use_tensor_schedule(self, times, weight, differentiable):
        if differentiable is not None:
            return bool(differentiable)
        if torch.is_tensor(times) and bool(times.requires_grad):
            return True
        if torch.is_tensor(weight):
            return True
        if isinstance(weight, Iterable) and not isinstance(weight, (str, bytes)):
            return True
        try:
            return float(weight) != 1.0
        except Exception:
            return True

    def _append_tensor_schedule(
        self,
        flat_idx: torch.Tensor,
        *,
        times_ms: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
        time_idx: torch.Tensor | None = None,
        weight_idx: torch.Tensor | None = None,
        allow_past: bool = False,
    ):
        """Append differentiable/value-mode scheduled events."""
        device, dtype = self.device(), self.dtype()
        flat_idx = flat_idx.to(device=device, dtype=torch.long).reshape(-1)
        n_evt = int(flat_idx.numel())
        if n_evt == 0:
            return self
        if (flat_idx < 0).any() or (flat_idx >= self._flat_numel()).any():
            raise IndexError("NetStim scheduled index out of range")

        if times_ms is None:
            times_ms = torch.zeros(n_evt, device=device, dtype=dtype)
        else:
            times_ms = times_ms.to(device=device, dtype=dtype).reshape(-1)
        if weight is None:
            weight = torch.ones(n_evt, device=device, dtype=dtype)
        else:
            weight = weight.to(device=device, dtype=dtype).reshape(-1)
        if time_idx is None:
            time_idx = torch.full((n_evt,), -1, device=device, dtype=torch.long)
        else:
            time_idx = time_idx.to(device=device, dtype=torch.long).reshape(-1)
        if weight_idx is None:
            weight_idx = torch.full((n_evt,), -1, device=device, dtype=torch.long)
        else:
            weight_idx = weight_idx.to(device=device, dtype=torch.long).reshape(-1)

        for name, value in (
            ("times_ms", times_ms),
            ("weight", weight),
            ("time_idx", time_idx),
            ("weight_idx", weight_idx),
        ):
            if value.numel() == 1 and n_evt > 1:
                value = value.expand(n_evt)
            elif value.numel() != n_evt:
                raise ValueError(f"{name} must be scalar or length {n_evt}")
            if name == "times_ms":
                times_ms = value
            elif name == "weight":
                weight = value
            elif name == "time_idx":
                time_idx = value
            else:
                weight_idx = value

        if (time_idx < -1).any():
            raise IndexError("time_idx must be -1 or a valid bound-source index")
        if (weight_idx < -1).any():
            raise IndexError("weight_idx must be -1 or a valid bound-source index")
        if (time_idx >= 0).any():
            if self._sched_t_source is None:
                raise RuntimeError("time_idx provided but no time source is bound")
            if (time_idx >= self._sched_t_source.numel()).any():
                raise IndexError("time_idx out of range for bound time source")
        if (weight_idx >= 0).any():
            if self._sched_w_source is None:
                raise RuntimeError("weight_idx provided but no weight source is bound")
            if (weight_idx >= self._sched_w_source.numel()).any():
                raise IndexError("weight_idx out of range for bound weight source")

        # Filter currently-past events only at scheduling time, matching NetCon.
        t_eval = times_ms
        if self._sched_t_source is not None and bool((time_idx >= 0).any()):
            idx = torch.clamp(time_idx, min=0)
            src = self._sched_t_source.index_select(0, idx).to(
                device=device, dtype=dtype
            )
            t_eval = t_eval + src * (time_idx >= 0).to(dtype)
        steps = _causal_step_index(t_eval, self._dt_scalar(dtype))
        if not allow_past:
            t_last = self.t_last.detach().reshape(-1).to(device=device, dtype=dtype)
            keep = t_eval > t_last.index_select(0, flat_idx)
            if not bool(keep.any()):
                return self
            flat_idx = flat_idx[keep]
            times_ms = times_ms[keep]
            weight = weight[keep]
            time_idx = time_idx[keep]
            weight_idx = weight_idx[keep]
            steps = steps[keep]

        self._renewal_config_generation += 1
        self.sched_flat_idx = torch.cat([self.sched_flat_idx, flat_idx], dim=0)
        self.sched_abs_step = torch.cat([self.sched_abs_step, steps], dim=0)
        self.sched_time_ms = torch.cat([self.sched_time_ms, times_ms], dim=0)
        self.sched_weight = torch.cat([self.sched_weight, weight], dim=0)
        self.sched_time_idx = torch.cat([self.sched_time_idx, time_idx], dim=0)
        self.sched_weight_idx = torch.cat([self.sched_weight_idx, weight_idx], dim=0)
        return self

    def _dt_scalar(self, dtype=None, dt=None):
        dtype = self.dtype() if dtype is None else dtype
        if dt is None:
            dt = getattr(self, "schedule_dt", 1.0)
        return torch.as_tensor(dt, device=self.device(), dtype=dtype).reshape(())

    def _scheduled_tensor_values(self):
        """Resolve value/ref schedules with the canonical NetStim source policy."""
        device, dtype = self.device(), self.dtype()
        time_source = self._sched_t_source
        weight_source = self._sched_w_source
        if time_source is not None:
            time_source = (
                time_source if bool((self.sched_time_idx >= 0).any()) else None
            )
        if weight_source is not None:
            weight_source = (
                weight_source if bool((self.sched_weight_idx >= 0).any()) else None
            )
        return resolve_schedule_values(
            self.sched_time_ms.to(device=device, dtype=dtype),
            self.sched_weight.to(device=device, dtype=dtype),
            self.sched_time_idx,
            self.sched_weight_idx,
            time_source=time_source,
            weight_source=weight_source,
            time_dtype=dtype,
            weight_dtype=dtype,
            device=device,
        )

    def _scheduled_tensor_gate(self, t: torch.Tensor, *, dt=None):
        """Aggregate tensor schedules through the shared pure evaluator."""
        device, dtype = self.device(), self.dtype()
        times, weights = self._scheduled_tensor_values()
        return netstim_tensor_schedule(
            self.sched_flat_idx.to(device=device, dtype=torch.long),
            times,
            weights,
            t.to(device=device, dtype=dtype),
            self._dt_scalar(dtype, dt=dt),
            shape=self.shape,
            differentiable_times=bool(getattr(self, "diff_scheduled_times", True)),
            width=float(getattr(self, "sched_width", 1.0)),
        )

    def _schedule_tensor_events(
        self,
        specs,
        *,
        times=None,
        weight=1.0,
        time_idx=None,
        weight_idx=None,
        allow_past=False,
        api="schedule",
    ):
        """Expand one declaration with independently selected value/ref fields."""
        fields = {}
        time_reference = time_idx is not None or api == "schedule_time_ref"
        weight_reference = weight_idx is not None or api == "schedule_ref"
        if not time_reference:
            fields["times"] = self._normalize_schedule_tensor(
                times, len(specs), name="time"
            )
        else:
            fields["time_idx"] = self._normalize_schedule_tensor(
                time_idx, len(specs), name="time_idx", dtype=torch.long
            )
        if not weight_reference:
            fields["weight"] = self._normalize_schedule_tensor(
                weight, len(specs), name="weight"
            )
        else:
            fields["weight_idx"] = self._normalize_schedule_tensor(
                weight_idx, len(specs), name="weight_idx", dtype=torch.long
            )
        flat_idx, fields = self._expand_schedule_specs_to_flat_tensors(specs, fields)
        if api == "schedule":
            for name in ("time_idx", "weight_idx"):
                if name in fields and bool((fields[name] < 0).any()):
                    raise IndexError(f"{name} must contain nonnegative indices")
        zeros = None
        if api == "schedule_ref":
            zeros = torch.zeros_like(fields["times"], dtype=self.dtype())
        elif time_reference or weight_reference:
            zeros = torch.zeros(
                flat_idx.numel(), device=self.device(), dtype=self.dtype()
            )
        return self._append_tensor_schedule(
            flat_idx,
            times_ms=fields.get("times", zeros),
            weight=fields.get("weight", zeros),
            time_idx=fields.get("time_idx"),
            weight_idx=fields.get("weight_idx"),
            allow_past=allow_past,
        )

    def schedule(
        self,
        indices,
        times=None,
        weight=1.0,
        *,
        times_ms=None,
        allow_past: bool = False,
        differentiable=None,
        time_idx=None,
        weight_idx=None,
    ):
        """Add explicit scheduled spikes, optionally with differentiable weights/times.

        ``differentiable=True`` selects tensor scheduling; ``False`` selects the
        consuming unit-amplitude calendar. When omitted, times requiring
        gradients, tensor/list weights, or non-unit scalar weights select
        tensor scheduling. Other unit-amplitude events use a cached compact
        calendar retaining Python-float precision. Calendar events are consumed
        at the first sampled time at or beyond their head, coalescing overdue
        events into one spike. Tensor schedules reevaluate on each call and
        assign hard events causally to the first fixed-step grid point at or
        after their absolute time.

        Tensor schedules contribute their amplitude to ``spike_gate``; downstream
        NetCons must use ``diff_spiking=True`` to propagate gradients into the
        scheduled times or weights.

        ``time_idx`` reads live times from ``bind_time_source`` instead of
        ``times``/``times_ms``. ``weight_idx`` reads live amplitudes from
        ``bind_weight_source`` instead of ``weight``. Either or both fields may
        be indexed. Referenced fields have no additive value offset: the default
        unit ``weight`` (or ``None``) is omitted when ``weight_idx`` is supplied.
        Indices must be nonnegative. Indexed events use tensor storage;
        ``differentiable=False`` selects the incompatible consuming calendar.
        """
        if times is None:
            times = times_ms
        elif times_ms is not None:
            raise ValueError("Provide only one of times or times_ms")
        if time_idx is not None and times is not None:
            raise ValueError("Provide only one of times/times_ms or time_idx")
        if weight_idx is not None and not (
            weight is None or (isinstance(weight, Real) and weight == 1.0)
        ):
            raise ValueError("Provide only one of weight or weight_idx")
        indexed = time_idx is not None or weight_idx is not None
        if indexed and differentiable is not None and not bool(differentiable):
            raise ValueError(
                "Indexed schedules require tensor storage; differentiable=False "
                "selects the consuming calendar."
            )
        if time_idx is not None and self._sched_t_source is None:
            raise RuntimeError("call bind_time_source(...) before schedule(...)")
        if weight_idx is not None and self._sched_w_source is None:
            raise RuntimeError("call bind_weight_source(...) before schedule(...)")
        if times is None and time_idx is None:
            raise ValueError("scheduled times are required")

        specs = self._normalize_index_specs(indices)

        if indexed or self._should_use_tensor_schedule(times, weight, differentiable):
            return self._schedule_tensor_events(
                specs,
                times=times,
                weight=weight,
                time_idx=time_idx,
                weight_idx=weight_idx,
                allow_past=allow_past,
            )

        # Legacy exact unit-amplitude schedule.
        times_list = self._normalize_schedule_times(times, len(specs))
        if len(specs) == 1 and len(times_list) > 1:
            specs = specs * len(times_list)
        elif len(specs) != len(times_list):
            raise ValueError(
                "indices and times must have the same length, unless one of them "
                "is scalar/broadcastable."
            )

        with torch.no_grad():
            t_last_flat = self.t_last.detach().reshape(-1).cpu()
            heaps = self._materialize_heap_rows()
            changed = False
            for spec, t_i in zip(specs, times_list):
                for flat_i in self._flat_indices_for_spec(spec):
                    t_cut = float(t_last_flat[flat_i].item())
                    if (not allow_past) and not (float(t_i) > t_cut):
                        continue
                    if not changed:
                        self._renewal_config_generation += 1
                        changed = True
                    heapq.heappush(heaps[flat_i], float(t_i))
            if changed:
                self._replace_heap_rows(heaps)
            else:
                self._refresh_next_sched_time_tensor()
        return self

    def schedule_ref(
        self,
        indices,
        times=None,
        weight_idx=None,
        *,
        times_ms=None,
        allow_past: bool = False,
    ):
        """Schedule events whose amplitudes are read from a bound weight source."""
        if self._sched_w_source is None:
            raise RuntimeError("call bind_weight_source(...) before schedule_ref(...)")
        if times is None:
            times = times_ms
        elif times_ms is not None:
            raise ValueError("Provide only one of times or times_ms")
        if times is None or weight_idx is None:
            raise ValueError("times/times_ms and weight_idx are required")
        specs = self._normalize_index_specs(indices)
        return self._schedule_tensor_events(
            specs,
            times=times,
            weight_idx=weight_idx,
            allow_past=allow_past,
            api="schedule_ref",
        )

    def schedule_time_ref(
        self, indices, time_idx, weight=1.0, *, allow_past: bool = False
    ):
        """Schedule events whose times are read from a bound time source."""
        if self._sched_t_source is None:
            raise RuntimeError(
                "call bind_time_source(...) before schedule_time_ref(...)"
            )
        specs = self._normalize_index_specs(indices)
        return self._schedule_tensor_events(
            specs,
            time_idx=time_idx,
            weight=weight,
            allow_past=allow_past,
            api="schedule_time_ref",
        )

    @torch.no_grad()
    def clear_schedule(self, indices: Optional[Iterable[int]] = None):
        """Remove scheduled spikes for selected state elements, or all if ``None``."""
        if indices is None:
            flat_indices = list(range(self._flat_numel()))
            clear_all = True
        else:
            specs = self._normalize_index_specs(indices)
            flat_indices = []
            for spec in specs:
                flat_indices.extend(self._flat_indices_for_spec(spec))
            flat_indices = sorted(set(int(i) for i in flat_indices))
            clear_all = False

        self._renewal_config_generation += 1
        heaps = self._materialize_heap_rows()
        for flat_i in flat_indices:
            heaps[flat_i].clear()
        self._replace_heap_rows(heaps)

        device, dtype = self.device(), self.dtype()
        if clear_all:
            self.sched_flat_idx = torch.empty(0, device=device, dtype=torch.long)
            self.sched_abs_step = torch.empty(0, device=device, dtype=torch.long)
            self.sched_weight_idx = torch.empty(0, device=device, dtype=torch.long)
            self.sched_time_idx = torch.empty(0, device=device, dtype=torch.long)
            self.sched_time_ms = torch.empty(0, device=device, dtype=dtype)
            self.sched_weight = torch.empty(0, device=device, dtype=dtype)
            return self

        if self.sched_flat_idx.numel() > 0:
            flat_filter = torch.as_tensor(flat_indices, device=device, dtype=torch.long)
            remove = torch.isin(self.sched_flat_idx, flat_filter)
            keep = ~remove
            self.sched_flat_idx = self.sched_flat_idx[keep]
            self.sched_abs_step = self.sched_abs_step[keep]
            self.sched_weight_idx = self.sched_weight_idx[keep]
            self.sched_time_idx = self.sched_time_idx[keep]
            self.sched_time_ms = self.sched_time_ms[keep]
            self.sched_weight = self.sched_weight[keep]
        return self

    @torch.no_grad()
    def _consume_scheduled_heaps(self, s_sched: torch.Tensor, t: torch.Tensor):
        if self._heap_nan_heaps is None:
            self.heap_cursor = heap_calendar_consume(
                self.heap_times,
                self.heap_row_offsets[1:],
                self.heap_cursor,
                torch.broadcast_to(t.detach(), self.shape).reshape(-1),
                s_sched.reshape(-1),
                depth=self.heap_depth,
                dtype=self.dtype(),
            )
            self.heap_override_active = self.heap_override_active & ~s_sched
            self._update_heap_head()
            return
        self._consume_nan_heaps(s_sched, t)

    @torch.no_grad()
    @torch._dynamo.disable()
    def _consume_nan_heaps(self, s_sched, t):
        """Compatibility path for heapq's non-total ordering around NaN."""
        fired_idx = torch.nonzero(s_sched.reshape(-1), as_tuple=True)[0]
        if fired_idx is None or fired_idx.numel() == 0:
            return

        idx_list = fired_idx.detach().cpu().tolist()
        t_flat = torch.broadcast_to(t.detach(), self.shape).reshape(-1)
        for flat_i in idx_list:
            cutoff = t_flat[flat_i]
            heap = self._heap_nan_heaps[flat_i]
            # A boolean NetStim source can emit at most one spike per sampled
            # time. Consume every heap event that is already due so duplicate or
            # skipped-over events coalesce now instead of leaking into later
            # timesteps as spurious spikes.
            while (
                heap
                and torch.as_tensor(heap[0], device=cutoff.device, dtype=cutoff.dtype)
                <= cutoff
            ):
                heapq.heappop(heap)
        self.heap_override_active = self.heap_override_active & ~s_sched
        self._update_heap_head()

    def forward(self, t, *, bptt: bool = False, dt=None, exp_rand=None):
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
        exp_rand : torch.Tensor, optional
            Prevalidated full-shaped unit-rate exponential base draw. Its values
            are detached before use. Omitting it samples the native generator;
            supplying it leaves generator state unchanged. The enclosing runtime
            draw adapter publishes the packet's authoritative RNG frame.

        Returns
        -------
        torch.Tensor
            Boolean tensor with shape ``self.shape`` indicating which generators
            fired for each batch element.
        """
        device, dtype = self.device(), self.dtype()
        t = self._broadcast_time_to_state(t)

        max_spikes = self._broadcast_to_state(
            self.max_spikes, dtype=torch.long, device=device, name="max_spikes"
        )
        sched_gate, sched_hard = self._scheduled_tensor_gate(t, dt=dt)

        if exp_rand is None:
            # Native sampling consumes one full-shaped draw, including
            # deterministic, exhausted, or not-yet-due generators.
            exp_rand = sample_netstim_exponential(
                self._rng, self.shape, device=device, dtype=dtype
            )
        else:
            exp_rand = exp_rand.detach()
        parameters = self.renewal_parameters()
        transition = netstim_renewal(
            NetStimState(self.next_stoch_time, self.spike_counts, self.t_last),
            parameters["interval"],
            parameters["noise"],
            max_spikes,
            t,
            exp_rand,
            tau=parameters["tau"],
            bptt=bptt,
            next_sched_time=self.next_sched_time,
            scheduled_gate=sched_gate,
            scheduled_hard=sched_hard,
        )
        self.spikes = transition.spikes
        self.spike_gate = transition.spike_gate

        if bptt:
            self.next_stoch_time = transition.state.next_stoch_time
        else:
            # A caller may switch from a BPTT step to a detached step. no_grad
            # copy_ alone would retain the previous buffer's stale grad_fn.
            self.next_stoch_time = self.next_stoch_time.detach()
            with torch.no_grad():
                self.next_stoch_time.copy_(transition.state.next_stoch_time)

        # The normal consuming calendar is a shared pure tensor transition.
        self._consume_scheduled_heaps(transition.heap_fires, t)

        with torch.no_grad():
            self.spike_counts.copy_(transition.state.spike_counts)
            if tuple(self.t_last.shape) == self.shape:
                self.t_last.copy_(transition.state.t_last)
            else:
                self.t_last = transition.state.t_last

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

        # Parametric.batch accepts one integer dimension at a time. Apply
        # dimensions inside-out so ``batch((2, 3))`` prepends ``(2, 3)``.
        for dim in reversed(dims):
            self.interval.batch(dim)

        old_heaps = self._sched_heaps
        repeat_count = self._flat_numel_from_shape(dims)
        old_flat_numel = self._flat_numel_from_shape(old_shape)
        batched_heaps = [list(h) for _ in range(repeat_count) for h in old_heaps]

        # Replicate tensor-backed schedules into the new leading batch copies.
        if getattr(self, "sched_flat_idx", torch.empty(0)).numel() > 0:
            offsets = torch.arange(
                repeat_count, device=self.device(), dtype=torch.long
            ) * int(old_flat_numel)
            self.sched_flat_idx = (
                self.sched_flat_idx.reshape(1, -1) + offsets.reshape(-1, 1)
            ).reshape(-1)
            self.sched_abs_step = self.sched_abs_step.repeat(repeat_count)
            self.sched_weight_idx = self.sched_weight_idx.repeat(repeat_count)
            self.sched_time_idx = self.sched_time_idx.repeat(repeat_count)
            self.sched_time_ms = self.sched_time_ms.repeat(repeat_count)
            self.sched_weight = self.sched_weight.repeat(repeat_count)

        self._set_shape_metadata(new_shape)
        self.spike_gate = self._prepend_batch_dims_to_tensor(self.spike_gate, dims).to(
            device=self.device(), dtype=self.dtype()
        )
        self._replace_heap_rows(batched_heaps)
        self._renewal_generation += 1
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
            self._detach_runtime()
        return self

    def _detach_runtime(self, memo=None, *, _promote_inference=False):
        from ._runtime_detach import (
            _detach_runtime_tree,
            _promote_inference_runtime_tree,
        )

        memo = {} if memo is None else memo
        transform = (
            _promote_inference_runtime_tree
            if _promote_inference
            else _detach_runtime_tree
        )
        for name in (
            "next_stoch_time",
            "next_sched_time",
            "spike_counts",
            "spikes",
            "spike_gate",
            "t_last",
            "heap_cursor",
            "heap_override_active",
            "heap_head_override",
        ):
            value = self._buffers.get(name, self.__dict__.get(name))
            if value is not None:
                setattr(self, name, transform(value, memo))

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
        - Scheduled spikes use a fixed compact calendar and explicit cursor.
          Legacy remaining-heap records and full original heap layouts retain
          compatibility with old checkpoints and later NaN insertions.
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

        checkpoint = {
            # Shape metadata (needed because leading batch dims are meaningful)
            "shape": self.shape,
            # Differentiable state (needed for BPTT across chunks)
            "next_stoch_time": self.next_stoch_time,
            # Non-differentiable state (counters/logical guards)
            "spike_counts": self.spike_counts,
            "t_last": self.t_last,
            # Scheduled-spike state
            "sched_heaps": sched_heaps,
            "sched_flat_idx": self.sched_flat_idx,
            "sched_abs_step": self.sched_abs_step,
            "sched_time_ms": self.sched_time_ms,
            "sched_weight": self.sched_weight,
            "sched_weight_idx": self.sched_weight_idx,
            "sched_time_idx": self.sched_time_idx,
            "diff_scheduled_times": bool(self.diff_scheduled_times),
            "sched_width": float(self.sched_width),
            "schedule_dt": self.schedule_dt,
            # RNG state (critical for deterministic checkpoint replay)
            "rng_state": self._rng.get_state(),
            "seeder_state": self._seeder.get_state(),
        }
        if self._heap_nan_heaps is None:
            checkpoint.update(
                heap_times=self.heap_times,
                heap_row_offsets=self.heap_row_offsets,
                heap_cursor=self.heap_cursor,
                heap_head_override=self.heap_head_override,
                heap_override_active=self.heap_override_active,
                heap_layout=tuple(tuple(row) for row in self._heap_legacy_rows),
            )
        return checkpoint

    def _restore_dict_from_checkpoint_unchecked(
        self, state_dict, *, preserve_heap_layout=False
    ):
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

        # 2) Modern calendars preserve topology and the consumed prefix. Old
        # heap snapshots contain only remaining events and are packed cold.
        heaps_in = state_dict.get("sched_heaps", None)
        if heaps_in is not None and len(heaps_in) != self._flat_numel():
            raise ValueError(
                f"Checkpoint has {len(heaps_in)} schedule heaps, but current "
                f"NetStim shape {self.shape} requires {self._flat_numel()}."
            )
        calendar_keys = ("heap_times", "heap_row_offsets", "heap_cursor")
        if any(name in state_dict for name in calendar_keys):
            if not all(name in state_dict for name in calendar_keys):
                raise ValueError("Checkpoint has an incomplete NetStim heap calendar.")
            values = tuple(state_dict[name] for name in calendar_keys)
            if not all(torch.is_tensor(value) for value in values):
                raise ValueError("Checkpoint heap calendar values must be tensors.")
            # Validate the saved representation cold before moving it. MPS
            # cannot receive a Float64 tensor, and a lower-precision MPS checkpoint
            # needs its original heap layout to recover precision on CPU/CUDA.
            times, offsets, cursor = (value.to(device="cpu") for value in values)
            validate_heap_calendar(
                times,
                offsets,
                cursor,
                rows=self._flat_numel(),
                device="cpu",
                dtype=times.dtype,
            )
            layout = state_dict.get("heap_layout")
            if layout is None:
                raw = times.detach().cpu().tolist()
                bounds = offsets.detach().cpu().tolist()
                layout = [
                    raw[left:right] for left, right in zip(bounds[:-1], bounds[1:])
                ]
            if len(layout) != self._flat_numel():
                raise ValueError("Checkpoint heap layout has the wrong row count.")
            layout = [[float(value) for value in row] for row in layout]
            if any(
                row[(index - 1) // 2] > value
                for row in layout
                for index, value in enumerate(row)
                if index
            ):
                raise ValueError(
                    "Checkpoint heap layout must satisfy min-heap ordering."
                )
            layout_times, layout_offsets, _, _ = pack_heap_calendar(
                layout, device="cpu", dtype=times.dtype
            )
            if not torch.equal(layout_times, times) or not torch.equal(
                layout_offsets, offsets
            ):
                raise ValueError("Checkpoint heap layout does not match its calendar.")
            self.heap_times, self.heap_row_offsets, _, self.heap_depth = (
                pack_heap_calendar(layout, device=device, simulation_dtype=dtype)
            )
            self.heap_cursor = cursor.to(device=device).clone()
            self._heap_nan_heaps = None
            self._heap_legacy_rows = [list(row) for row in layout]
            self.heap_head_override = (
                self._broadcast_to_state(
                    state_dict.get(
                        "heap_head_override",
                        torch.full(
                            self.shape, float("inf"), device=device, dtype=dtype
                        ),
                    ),
                    dtype=dtype,
                    device=device,
                    name="heap_head_override",
                )
                .detach()
                .clone()
            )
            self.heap_override_active = (
                self._broadcast_to_state(
                    state_dict.get(
                        "heap_override_active",
                        torch.zeros(self.shape, device=device, dtype=torch.bool),
                    ),
                    dtype=torch.bool,
                    device=device,
                    name="heap_override_active",
                )
                .detach()
                .clone()
            )
            self._update_heap_head()
        elif heaps_in is not None:
            heaps = [list(h) for h in heaps_in]
            if not preserve_heap_layout:
                for h in heaps:
                    heapq.heapify(h)
            self._replace_heap_rows(heaps)
        if heaps_in is None and "next_sched_time" in state_dict:
            # Backward-compatible path if an older checkpoint stored the tensor.
            # It overrides, rather than merges with, any existing calendar head.
            self.heap_head_override = self._broadcast_to_state(
                state_dict["next_sched_time"].detach(),
                dtype=dtype,
                device=device,
                name="next_sched_time",
            ).clone()
            self.heap_override_active = torch.ones(
                self.shape, device=device, dtype=torch.bool
            )
            self._update_heap_head()

        # Restore tensor-backed scheduled events. Bound value sources are
        # intentionally not serialized here; callers should keep them attached
        # to the module / training object as with NetCon reference schedules.
        self.diff_scheduled_times = bool(
            state_dict.get(
                "diff_scheduled_times", getattr(self, "diff_scheduled_times", True)
            )
        )
        self.sched_width = float(
            state_dict.get("sched_width", getattr(self, "sched_width", 1.0))
        )
        if "schedule_dt" in state_dict:
            self.schedule_dt = torch.as_tensor(
                state_dict["schedule_dt"], device=device, dtype=dtype
            ).reshape(())
        for name in (
            "sched_flat_idx",
            "sched_abs_step",
            "sched_weight_idx",
            "sched_time_idx",
        ):
            if name in state_dict:
                setattr(
                    self,
                    name,
                    state_dict[name]
                    .detach()
                    .to(device=device, dtype=torch.long)
                    .clone(),
                )
            else:
                setattr(self, name, torch.empty(0, device=device, dtype=torch.long))
        if "sched_time_ms" in state_dict:
            stm = state_dict["sched_time_ms"].to(device=device, dtype=dtype)
            self.sched_time_ms = stm.clone() if not stm.requires_grad else stm
        else:
            self.sched_time_ms = torch.empty(0, device=device, dtype=dtype)
        if "sched_weight" in state_dict:
            sw = state_dict["sched_weight"].to(device=device, dtype=dtype)
            self.sched_weight = sw.clone() if not sw.requires_grad else sw
        else:
            self.sched_weight = torch.empty(0, device=device, dtype=dtype)

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
            # Eval path: keep the existing buffer storage when possible.
            # A preceding training step may have left a differentiable clock.
            # A no_grad copy alone would retain that unrelated history.
            self.next_stoch_time = self.next_stoch_time.detach()
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

    def restore_dict_from_checkpoint(self, state_dict):
        """Restore a NetStim runtime checkpoint atomically."""
        previous = _clone_checkpoint_state(self.state_dict_for_checkpoint())
        previous_outputs = self.spikes, self.spike_gate
        try:
            result = self._restore_dict_from_checkpoint_unchecked(state_dict)
            current = self.state_dict_for_checkpoint()
            tensor_schedule_present = (
                previous["sched_flat_idx"].numel() > 0
                or current["sched_flat_idx"].numel() > 0
            )
            if not _matching_heap_configuration(previous, current) or (
                tensor_schedule_present
                and not _matching_tensor_schedule_configuration(previous, current)
            ):
                self._renewal_config_generation += 1
            self._renewal_initialized = True
            return result
        except Exception:
            try:
                self._restore_dict_from_checkpoint_unchecked(
                    previous, preserve_heap_layout=True
                )
            except Exception as rollback_error:
                raise RuntimeError(
                    "NetStim checkpoint restore failed and rollback could not "
                    "recover the previous runtime state."
                ) from rollback_error
            finally:
                # Successful restore resets outputs for the next call, whereas
                # a failed restore must retain the caller's current outputs and
                # their autograd history as well as the checkpointed carry.
                self.spikes, self.spike_gate = previous_outputs
            raise
