import hashlib
import math
import warnings
from typing import Any, Dict, Literal

import torch

from ...helpers import current_native_extension_policy
from ...utils.tensor_ops import _logical_tensor_bytes
from ..parametric import Referency
from .netstim import NetStim, _causal_step_index
from .spiking import update_active, update_active_diff
from .utils import make_getattr


def _handle_native_bitpack_failure(operation: str, exc: BaseException | None) -> bool:
    """Apply the configured policy to an eligible CUDA native-path failure."""
    policy = current_native_extension_policy()
    cause = exc or RuntimeError("native NetCon bitpack extension is unavailable")
    detail = (
        f"NetCon native CUDA bitpack {operation} failed. Cause: {cause!r}. "
        "Run `dendra doctor` for environment diagnostics."
    )
    if policy == "require":
        raise RuntimeError(
            f"{detail} Pure-PyTorch fallback is disabled by "
            "NATIVE_EXTENSION_POLICY='require'."
        ) from cause
    if policy == "warn":
        warnings.warn(
            f"{detail} Falling back to the pure-PyTorch implementation.",
            RuntimeWarning,
            stacklevel=3,
        )
    return False


def _cache_dt_value(dt) -> float:
    """Return a Python float from a scalar dt tensor or number."""
    if torch.is_tensor(dt):
        return float(dt.detach().cpu().item())
    return float(dt)


def _cache_current_slot(step) -> int:
    """Return a Python int from a scalar current_time_step tensor or number."""
    if torch.is_tensor(step):
        return int(step.detach().cpu().reshape(-1)[0].item())
    return int(step)


def _round_half_up_time_indices(
    count: int,
    numerator_dt: float,
    denominator_dt: float,
    *,
    device,
) -> torch.Tensor:
    """Return robust nearest-step indices without device float64 requirements.

    Cache translation is a cold path, so calculate the floating-point positions
    on CPU.  This keeps the operation compatible with devices such as MPS and a
    small scale-aware tolerance makes documented half-up ties stable when a
    decimal ratio (for example 0.3 / 0.2) lands one ULP below its ideal value.
    """
    count = int(count)
    if count <= 0:
        return torch.empty(0, device=device, dtype=torch.long)
    positions = torch.arange(count, dtype=torch.float64) * (
        float(numerator_dt) / float(denominator_dt)
    )
    tolerance = (
        torch.maximum(positions.abs(), torch.ones_like(positions))
        * torch.finfo(torch.float64).eps
        * 8.0
    )
    indices = torch.floor(positions + 0.5 + tolerance).to(torch.long)
    return indices.to(device=device)


def _round_half_up_time_index(
    index: int, numerator_dt: float, denominator_dt: float
) -> int:
    """Scalar counterpart of :func:`_round_half_up_time_indices`."""
    value = float(index) * float(numerator_dt) / float(denominator_dt)
    tolerance = max(abs(value), 1.0) * torch.finfo(torch.float64).eps * 8.0
    return int(math.floor(value + 0.5 + tolerance))


def _dilate_time_rows(
    rows: torch.Tensor,
    old_dt: float,
    new_dt: float,
    *,
    n_limit: int,
) -> torch.Tensor:
    """Re-bin a [T, E] time-ring tensor into a new dt/depth.

    Row 0 is interpreted as the next/current due row, row k as k timesteps in
    the future/history depending on the caller.  Values that collide after
    re-binning are summed, matching Network.dilate() semantics.
    """
    if rows.ndim != 2:
        raise ValueError("cached time rows must be 2D")
    n_limit = int(n_limit)
    T, E = rows.shape
    if n_limit <= 0:
        return rows.new_zeros((0, E))
    if T == 0:
        return rows.new_zeros((n_limit, E))
    if float(old_dt) == float(new_dt) and T == n_limit:
        return rows.detach().clone()

    idx = _round_half_up_time_indices(
        T,
        old_dt,
        new_dt,
        device=rows.device,
    )
    idx.clamp_(0, n_limit - 1)
    out = torch.zeros((n_limit, E), device=rows.device, dtype=rows.dtype)
    out.index_add_(0, idx, rows)
    return out


def _resample_continuous_time_rows(
    rows: torch.Tensor,
    old_dt: float,
    new_dt: float,
    *,
    n_limit: int,
) -> torch.Tensor:
    """Nearest-neighbor resample a future continuous-signal delay line.

    Continuous projections represent one analog sample at each time row.  Unlike
    event payloads, two old samples that map to one coarser row must not be
    summed, and a finer grid must not insert zero-valued gaps.  The connection
    delays themselves use nearest-step quantization, so nearest-neighbor sampling
    gives the matching deterministic cache-translation rule.
    """
    if rows.ndim != 2:
        raise ValueError("cached continuous time rows must be 2D")
    n_limit = int(n_limit)
    T, E = rows.shape
    if n_limit <= 0:
        return rows.new_zeros((0, E))
    if T == 0:
        return rows.new_zeros((n_limit, E))
    if float(old_dt) == float(new_dt) and T == n_limit:
        return rows.detach().clone()

    old_idx = _round_half_up_time_indices(
        n_limit,
        new_dt,
        old_dt,
        device=rows.device,
    )
    # Endpoint hold is the only stable reconstruction when timestep rounding
    # makes the new physical horizon slightly longer than the cached one.
    old_idx.clamp_(0, T - 1)
    return rows.index_select(0, old_idx).detach().clone()


def _dilate_packed_time_rows(
    rows: torch.Tensor,
    old_dt: float,
    new_dt: float,
    *,
    n_limit: int,
) -> torch.Tensor:
    """Re-bin packed int64 spike-history rows using bitwise OR collisions."""
    if rows.ndim != 2:
        raise ValueError("cached packed history must be 2D")
    n_limit = int(n_limit)
    T, W = rows.shape
    if n_limit <= 0:
        return rows.new_zeros((0, W))
    if T == 0:
        return rows.new_zeros((n_limit, W))
    if float(old_dt) == float(new_dt) and T == n_limit:
        return rows.detach().clone()

    idx = _round_half_up_time_indices(
        T,
        old_dt,
        new_dt,
        device=rows.device,
    )
    idx.clamp_(0, n_limit - 1)
    out = torch.zeros((n_limit, W), device=rows.device, dtype=rows.dtype)
    # n_limit is delay depth, not connection count; this loop is out of the hot path.
    for old_i, new_i in enumerate(idx.detach().cpu().tolist()):
        out[int(new_i)].bitwise_or_(rows[old_i])
    return out


def _ring_rows_to_age_rows(rows: torch.Tensor, cur_slot: int) -> torch.Tensor:
    """Return age-ordered rows from a circular history ring.

    Row 0 in the returned tensor is the current/next-write slot.  Row ``a`` is
    the row that was written ``a`` timesteps before the current slot.  Restore
    code generally ignores age 0 because it corresponds to the next row that
    will be overwritten, not to a past emission.
    """
    if rows.ndim != 2:
        raise ValueError("history rows must be 2D")
    T = int(rows.shape[0])
    if T == 0:
        return rows.detach().clone()
    cur_slot = int(cur_slot) % T
    ages = torch.arange(T, device=rows.device, dtype=torch.long)
    idx = (cur_slot - ages).remainder(T)
    return rows.index_select(0, idx).detach().clone()


def _age_rows_to_ring_rows(age_rows: torch.Tensor, *, depth: int) -> torch.Tensor:
    """Convert age-ordered history rows back to a current_slot==0 ring."""
    if age_rows.ndim != 2:
        raise ValueError("age rows must be 2D")
    depth = int(depth)
    if depth <= 0:
        return age_rows.new_zeros((0, age_rows.shape[1]))
    out = age_rows.new_zeros((depth, age_rows.shape[1]))
    n = min(int(age_rows.shape[0]), depth)
    if n == 0:
        return out
    ages = torch.arange(n, device=age_rows.device, dtype=torch.long)
    idx = (-ages).remainder(depth)
    out.index_copy_(0, idx, age_rows[:n])
    return out


def _dilate_history_age_rows(
    rows: torch.Tensor,
    old_dt: float,
    new_dt: float,
    *,
    n_limit: int,
) -> torch.Tensor:
    """Re-bin age-ordered numeric history rows across dt changes.

    Row ``a`` is interpreted as a sample/event emitted ``a * old_dt`` ms before
    the cached state.  Collisions are summed, matching the event-delivery cache
    convention used elsewhere in Network.
    """
    if rows.ndim != 2:
        raise ValueError("cached age rows must be 2D")
    n_limit = int(n_limit)
    T, E = rows.shape
    if n_limit <= 0:
        return rows.new_zeros((0, E))
    if T == 0:
        return rows.new_zeros((n_limit, E))
    if float(old_dt) == float(new_dt) and T == n_limit:
        return rows.detach().clone()

    idx = _round_half_up_time_indices(
        T,
        old_dt,
        new_dt,
        device=rows.device,
    )
    idx.clamp_(0, n_limit - 1)
    out = torch.zeros((n_limit, E), device=rows.device, dtype=rows.dtype)
    out.index_add_(0, idx, rows)
    return out


def _resample_continuous_history_age_rows(
    rows: torch.Tensor,
    old_dt: float,
    new_dt: float,
    *,
    n_limit: int,
) -> torch.Tensor:
    """Nearest-neighbor resample age-ordered continuous source values.

    Age row 0 is the next-write ring slot and is not a historical sample.  For
    finer grids, ages younger than the newest cached sample therefore hold that
    newest sample.  Requests just outside the cached physical horizon hold the
    oldest endpoint instead of introducing a discontinuous zero.
    """
    if rows.ndim != 2:
        raise ValueError("cached continuous age rows must be 2D")
    n_limit = int(n_limit)
    T, E = rows.shape
    if n_limit <= 0:
        return rows.new_zeros((0, E))
    if T <= 1:
        return rows.new_zeros((n_limit, E))
    if float(old_dt) == float(new_dt) and T == n_limit:
        return rows.detach().clone()

    out = rows.new_zeros((n_limit, E))
    if n_limit <= 1:
        return out
    old_ages = _round_half_up_time_indices(
        n_limit,
        new_dt,
        old_dt,
        device=rows.device,
    )[1:]
    # Age zero is not a sample.  Hold the newest/oldest cached endpoint when a
    # finer grid or quantized horizon asks just outside the available interval.
    old_ages.clamp_(1, T - 1)
    target_ages = torch.arange(1, n_limit, device=rows.device, dtype=torch.long)
    out.index_copy_(0, target_ages, rows.index_select(0, old_ages))
    return out


def _dilate_packed_history_age_rows(
    rows: torch.Tensor,
    old_dt: float,
    new_dt: float,
    *,
    n_limit: int,
) -> torch.Tensor:
    """Re-bin age-ordered packed int64 history rows using bitwise OR."""
    if rows.ndim != 2:
        raise ValueError("cached packed age rows must be 2D")
    n_limit = int(n_limit)
    T, W = rows.shape
    if n_limit <= 0:
        return rows.new_zeros((0, W))
    if T == 0:
        return rows.new_zeros((n_limit, W))
    if float(old_dt) == float(new_dt) and T == n_limit:
        return rows.detach().clone()

    idx = _round_half_up_time_indices(
        T,
        old_dt,
        new_dt,
        device=rows.device,
    )
    idx.clamp_(0, n_limit - 1)
    out = torch.zeros((n_limit, W), device=rows.device, dtype=rows.dtype)
    for old_i, new_i in enumerate(idx.detach().cpu().tolist()):
        out[int(new_i)].bitwise_or_(rows[old_i])
    return out


def _clone_calendar_chunks(calendar: dict, *, cur_slot: int, depth: int):
    """Normalize Python sparse-calendar buckets so slot 0 is next/current."""
    out = {}
    depth = max(1, int(depth))
    cur_slot = int(cur_slot)
    for slot, chunks in calendar.items():
        rel_slot = (int(slot) - cur_slot) % depth
        copied = []
        for idx, val in chunks:
            copied.append((idx.detach().clone(), val.detach().clone()))
        if copied:
            out.setdefault(rel_slot, []).extend(copied)
    return out


def _restore_calendar_chunks(
    cached_calendar: dict,
    *,
    old_dt: float,
    new_dt: float,
    depth: int,
    idx_device,
    idx_dtype,
    val_device,
    val_dtype,
):
    """Restore normalized sparse-calendar chunks onto a current_slot==0 ring."""
    out = {}
    depth = max(1, int(depth))
    for rel_slot, chunks in cached_calendar.items():
        new_slot = _round_half_up_time_index(rel_slot, old_dt, new_dt)
        new_slot = max(0, min(depth - 1, new_slot))
        restored = []
        for idx, val in chunks:
            restored.append(
                (
                    idx.detach().to(device=idx_device, dtype=idx_dtype).clone(),
                    val.detach().to(device=val_device, dtype=val_dtype).clone(),
                )
            )
        if restored:
            out.setdefault(new_slot, []).extend(restored)
    return out


def _checkpoint_calendar_chunks(calendar: dict):
    """Copy an exact runtime calendar without normalizing its ring slots."""
    return {
        int(slot): [(idx.clone(), value.clone()) for idx, value in chunks]
        for slot, chunks in calendar.items()
        if chunks
    }


def _restore_checkpoint_calendar_chunks(
    calendar: dict,
    *,
    idx_device,
    idx_dtype,
    value_device,
    value_dtype,
):
    """Rebuild a checkpoint calendar on the receiving component's devices."""
    if not isinstance(calendar, dict):
        raise TypeError("NetCon checkpoint calendar must be a dict.")
    restored = {}
    for slot, chunks in calendar.items():
        if not isinstance(slot, int):
            raise TypeError("NetCon checkpoint calendar slots must be integers.")
        copied = []
        for chunk in chunks:
            if not isinstance(chunk, (tuple, list)) or len(chunk) != 2:
                raise TypeError(
                    "NetCon checkpoint calendar chunks must be (index, value) pairs."
                )
            idx, value = chunk
            if not torch.is_tensor(idx) or not torch.is_tensor(value):
                raise TypeError("NetCon checkpoint calendar payloads must be tensors.")
            if idx.ndim != 1 or value.ndim != 1 or idx.numel() != value.numel():
                raise ValueError(
                    "NetCon checkpoint calendar indices and values must be equally "
                    "sized one-dimensional tensors."
                )
            copied.append(
                (
                    idx.detach().to(device=idx_device, dtype=idx_dtype).clone(),
                    value.to(device=value_device, dtype=value_dtype).clone(),
                )
            )
        if copied:
            restored[int(slot)] = copied
    return restored


def _topology_tensor_digest(*tensors: torch.Tensor) -> str:
    """Return a stable digest for small/static connection-topology tensors."""
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(_logical_tensor_bytes(value).numpy().tobytes())
    return digest.hexdigest()


def _require_checkpoint_tensor(state_dict, name, *, shape, dtype=None):
    """Validate a required runtime checkpoint tensor before rebinding state."""
    if name not in state_dict:
        raise KeyError(f"NetCon checkpoint is missing {name!r}.")
    value = state_dict[name]
    if not torch.is_tensor(value):
        raise TypeError(f"NetCon checkpoint {name!r} must be a tensor.")
    if tuple(value.shape) != tuple(shape):
        raise ValueError(
            f"NetCon checkpoint {name!r} has shape {tuple(value.shape)}, "
            f"expected {tuple(shape)}."
        )
    if dtype is not None and value.dtype != dtype:
        raise TypeError(
            f"NetCon checkpoint {name!r} has dtype {value.dtype}, expected {dtype}."
        )
    return value


class ContinuousCon(Referency):
    """Continuous analog projection between a presynaptic variable and a postsynaptic mechanism.

    ``ContinuousCon`` is the analog counterpart of :class:`NetCon`: it gathers a
    presynaptic variable, applies optional weighting and integer delay, scatters
    the resulting values into the target synapse's local shape, and calls
    ``syn.continuous_receive(...)``.  It performs no thresholding and no event
    detection.

    The hot path is specialized at construction / delay-rebuild time.  Static
    delay masks, selected post indices, flat delay-buffer offsets, and dispatch
    mode are precomputed once so ``advance()`` does not repeatedly build masks or
    check for empty delayed subsets.
    """

    def __init__(
        self,
        pre,
        pre_idx,
        post,
        post_idx,
        post_syn,
        weight,
        delay,
        dt,
        pre_var=None,
        input=None,
        reduce="sum",
        max_delay=None,
        transform=None,
        reset_inputs=False,
        device=None,
        dtype=None,
        pre_device=None,
        pre_dtype=None,
    ):
        super().__init__()
        self.weight = weight
        self.delay_ms = delay
        self.syn = post_syn
        self.pre = pre
        self.post = post
        self.input = input
        self.reduce = reduce
        self.transform = transform
        self.max_delay = max_delay

        # reset_inputs is accepted for compatibility with older internal
        # builders, but input reset is now owned by Network.step so every
        # ContinuousSynapse target is reset exactly once before all analog
        # deliveries.  ContinuousCon therefore has no per-step reset branch.
        self._reset_inputs = False

        self._refresh_peer_devices(
            device=device, dtype=dtype, pre_device=pre_device, pre_dtype=pre_dtype
        )
        self.weight = self.weight.to(device=self.device, dtype=self.dtype)
        self.delay_ms = self.delay_ms.to(device=self.device, dtype=self.dtype)
        if isinstance(self.transform, torch.nn.Module):
            self.transform = self.transform.to(device=self.device, dtype=self.dtype)

        self.dt = torch.tensor(dt, device=self.device, dtype=self.dtype)
        if pre_var is None:
            pre_var = "v"
        self.get_pre_var = make_getattr(pre_var)
        self.pre_var = pre_var
        self._has_transform = transform is not None

        self.register_buffer(
            "pre_idx", pre_idx.flatten().to(self.pre_device, dtype=torch.long)
        )
        self.register_buffer(
            "post_idx", post_idx.flatten().to(self.device, dtype=torch.long)
        )
        self.register_buffer(
            "syn_numel",
            torch.prod(torch.tensor(self.syn.shape_f, device=self.device)).to(
                self.device, dtype=torch.long
            ),
        )
        self._syn_numel = int(self.syn_numel.item())
        self.register_buffer(
            "n",
            torch.tensor(self.pre_idx.numel(), device=self.device, dtype=torch.long),
        )
        self._n_conn = int(self.pre_idx.numel())
        self.register_buffer(
            "current_time_step", torch.tensor([0], device=self.device, dtype=torch.long)
        )
        self.register_buffer(
            "global_step", torch.tensor([0], device=self.device, dtype=torch.long)
        )

        delay_values = self.delay_ms.init().w
        delay_steps = (delay_values / dt).round().long()
        self.register_buffer("delay_steps", delay_steps.flatten().to(self.device))
        self.max_delay_steps = self._compute_max_delay_steps()
        self._delivery_numel = int(self.max_delay_steps * self._syn_numel)
        self.register_buffer(
            "delivery_buffer",
            torch.zeros(
                (self.max_delay_steps, self._syn_numel),
                device=self.device,
                dtype=self.dtype,
            ),
        )
        self.register_buffer(
            "delivery_mask",
            torch.zeros(
                (self.max_delay_steps, self._syn_numel),
                device=self.device,
                dtype=torch.bool,
            ),
        )
        self.register_buffer(
            "con_range",
            torch.arange(self._n_conn, device=self.device, dtype=torch.long),
        )

        # Delay/layout metadata buffers are registered in _rebuild_delay_metadata().
        self._rebuild_delay_metadata()
        self._align_buffer_devices()
        self._configure_advance_impl()

    # ------------------------------------------------------------------
    # Device / dtype bookkeeping
    # ------------------------------------------------------------------

    def _refresh_peer_devices(
        self, *, device=None, dtype=None, pre_device=None, pre_dtype=None
    ):
        actual_pre_device = self.pre.device()
        actual_pre_dtype = self.pre.dtype()
        post_device = self.post.device()
        post_dtype = self.post.dtype()
        syn_device = self.syn.device() if hasattr(self.syn, "device") else post_device

        self.pre_device = (
            torch.device(pre_device) if pre_device is not None else actual_pre_device
        )
        self.pre_dtype = pre_dtype if pre_dtype is not None else actual_pre_dtype
        self.post_device = post_device
        self.device = torch.device(device) if device is not None else syn_device
        self.dtype = dtype if dtype is not None else post_dtype

    def _move_buffer(self, name, device, dtype=None):
        if not hasattr(self, name):
            return
        buf = getattr(self, name)
        target_dtype = dtype if dtype is not None else buf.dtype
        if buf.device != device or buf.dtype != target_dtype:
            setattr(self, name, buf.to(device=device, dtype=target_dtype))

    def _set_buffer(self, name, value):
        if name in self._buffers:
            setattr(self, name, value)
        else:
            self.register_buffer(name, value)

    def _align_buffer_devices(self):
        self._move_buffer("pre_idx", self.pre_device)
        for name in (
            "post_idx",
            "syn_numel",
            "n",
            "delay_steps",
            "current_time_step",
            "global_step",
            "con_range",
            # Precomputed continuous-delay metadata.
            "zero_con_idx",
            "nonzero_con_idx",
            "post_idx_zero",
            "post_idx_nz",
            "delay_steps_nz",
            "flat_delay_offsets_nz",
            "post_delivery_mask",
            "post_zero_delivery_mask",
            "state_cache_pre_value_history",
        ):
            self._move_buffer(name, self.device)
        self._move_buffer("delivery_buffer", self.device, self.dtype)
        self._move_buffer("delivery_mask", self.device, torch.bool)
        self.dt = self.dt.to(device=self.device, dtype=self.dtype)
        self.weight = self.weight.to(device=self.device, dtype=self.dtype)
        self.delay_ms = self.delay_ms.to(device=self.device, dtype=self.dtype)

    def to(self, *args, **kwargs):  # type: ignore[override]
        super().to(*args, **kwargs)
        self._refresh_peer_devices()
        self._align_buffer_devices()
        if isinstance(self.transform, torch.nn.Module):
            self.transform = self.transform.to(device=self.device, dtype=self.dtype)
        self._configure_advance_impl()
        return self

    # ------------------------------------------------------------------
    # Delay metadata / specialization
    # ------------------------------------------------------------------

    def _compute_max_delay_steps(self):
        if self.max_delay is not None:
            return max(1, int(self.max_delay / self.dt.item()) + 1)
        if self.delay_steps.numel() == 0:
            return 1
        return max(1, int(self.delay_steps.max().item()) + 1)

    def _rebuild_delay_metadata(self):
        """Precompute masks/indices/offsets used by the continuous hot path."""
        delay_steps = self.delay_steps.flatten().to(self.device, dtype=torch.long)
        n_conn = int(delay_steps.numel())
        self._n_conn = n_conn
        self._has_connections = n_conn > 0

        if n_conn == 0:
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            self._set_buffer("zero_con_idx", empty)
            self._set_buffer("nonzero_con_idx", empty)
            self._set_buffer("post_idx_zero", empty)
            self._set_buffer("post_idx_nz", empty)
            self._set_buffer("delay_steps_nz", empty)
            self._set_buffer("flat_delay_offsets_nz", empty)
            self._set_buffer(
                "post_delivery_mask",
                torch.zeros(self._syn_numel, device=self.device, dtype=torch.bool),
            )
            self._set_buffer(
                "post_zero_delivery_mask",
                torch.zeros(self._syn_numel, device=self.device, dtype=torch.bool),
            )
            self._has_zero_delay = False
            self._has_nonzero_delay = False
            self._all_zero_delay = False
            self._all_nonzero_delay = False
            self._nonzero_delay_uniform = False
            self._direct_all = False
            return

        zero_mask = delay_steps == 0
        nonzero_mask = ~zero_mask
        zero_idx = torch.nonzero(zero_mask, as_tuple=False).flatten().to(self.device)
        nonzero_idx = (
            torch.nonzero(nonzero_mask, as_tuple=False).flatten().to(self.device)
        )

        self._has_zero_delay = bool(zero_idx.numel() > 0)
        self._has_nonzero_delay = bool(nonzero_idx.numel() > 0)
        self._all_zero_delay = bool(
            self._has_zero_delay and not self._has_nonzero_delay
        )
        self._all_nonzero_delay = bool(
            self._has_nonzero_delay and not self._has_zero_delay
        )

        post_idx_zero = (
            self.post_idx.index_select(0, zero_idx) if zero_idx.numel() else zero_idx
        )
        post_idx_nz = (
            self.post_idx.index_select(0, nonzero_idx)
            if nonzero_idx.numel()
            else nonzero_idx
        )
        delay_steps_nz = (
            delay_steps.index_select(0, nonzero_idx)
            if nonzero_idx.numel()
            else nonzero_idx
        )
        flat_offsets = delay_steps_nz * int(self._syn_numel) + post_idx_nz

        self._set_buffer("zero_con_idx", zero_idx)
        self._set_buffer("nonzero_con_idx", nonzero_idx)
        self._set_buffer("post_idx_zero", post_idx_zero)
        self._set_buffer("post_idx_nz", post_idx_nz)
        self._set_buffer("delay_steps_nz", delay_steps_nz)
        self._set_buffer("flat_delay_offsets_nz", flat_offsets)
        post_delivery_mask = torch.zeros(
            self._syn_numel, device=self.device, dtype=torch.bool
        )
        post_delivery_mask.index_fill_(0, self.post_idx.to(self.device), True)
        post_zero_delivery_mask = torch.zeros_like(post_delivery_mask)
        if post_idx_zero.numel() > 0:
            post_zero_delivery_mask.index_fill_(0, post_idx_zero, True)
        self._set_buffer("post_delivery_mask", post_delivery_mask)
        self._set_buffer("post_zero_delivery_mask", post_zero_delivery_mask)

        if delay_steps_nz.numel() > 0:
            first = delay_steps_nz[0]
            self._nonzero_delay_uniform = bool(
                torch.all(delay_steps_nz == first).item()
            )
            self._uniform_delay_step = (
                int(first.item()) if self._nonzero_delay_uniform else None
            )
        else:
            self._nonzero_delay_uniform = False
            self._uniform_delay_step = None

        # Fast path for dense one-to-one all-immediate projections: avoid scatter.
        if n_conn == self._syn_numel:
            expected = torch.arange(
                self._syn_numel, device=self.device, dtype=torch.long
            )
            self._direct_all = bool(
                torch.equal(self.post_idx.to(self.device), expected)
            )
        else:
            self._direct_all = False

    def _configure_advance_impl(self):
        if not getattr(self, "_has_connections", False):
            self.advance = self._advance_empty
        elif self._all_zero_delay:
            self.advance = self._advance_all_immediate
        elif self._all_nonzero_delay:
            if self._nonzero_delay_uniform:
                self.advance = self._advance_all_delayed_uniform
            else:
                self.advance = self._advance_all_delayed_mixed
        else:
            if self._nonzero_delay_uniform:
                self.advance = self._advance_mixed_uniform
            else:
                self.advance = self._advance_mixed

    def _rebuild_delay_buffers(self):
        with torch.no_grad():
            delay_values = self.delay_ms()
            delay_steps = (delay_values / self.dt.to(self.dtype)).round().long()
            self.delay_steps = delay_steps.flatten().to(self.device)
            self.max_delay_steps = self._compute_max_delay_steps()
            self._delivery_numel = int(self.max_delay_steps * self._syn_numel)
            self.delivery_buffer = torch.zeros(
                (self.max_delay_steps, self._syn_numel),
                device=self.device,
                dtype=self.dtype,
            )
            self.delivery_mask = torch.zeros(
                (self.max_delay_steps, self._syn_numel),
                device=self.device,
                dtype=torch.bool,
            )
            self._rebuild_delay_metadata()
            self._align_buffer_devices()
            self._configure_advance_impl()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def zero(self, clear_deliveries=True):
        self.current_time_step.fill_(0)
        self.global_step.fill_(0)
        if clear_deliveries:
            self.delivery_buffer.zero_()
            self.delivery_mask.zero_()

    def detach(self):
        for n, b in self.named_buffers():
            setattr(self, n, b.detach())

    def initialize(
        self, reinit_weights=True, reinit_delays=True, clear_deliveries=True
    ):
        self.zero(clear_deliveries=clear_deliveries)
        self.weight.init(reinit=reinit_weights)
        self.delay_ms.init(reinit=reinit_delays)
        if reinit_delays:
            self._rebuild_delay_buffers()
        if hasattr(self, "t"):
            step = torch.round(
                self.t.to(self.device, dtype=self.dtype) / self.dt.to(self.dtype)
            ).long()
            self.global_step.copy_(step.reshape_as(self.global_step))
        self.detach()
        self._configure_advance_impl()
        return self

    def enable_state_cache_recording(self, *, horizon_steps: int | None = None):
        """Record raw presynaptic values for parameter-invariant steady-state cache.

        The recorded history is used by ``state_cache()`` to rebuild pending
        continuous deliveries with the *current* transform, weights, and delays
        during a later ``initialize_from_state_cache()`` call.  Transforms used
        with this cache path must be deterministic, memoryless functions of one
        source-value vector.  The recorder is only intended for
        steady-state/cache runs, not the hot simulation path.
        """
        steps = (
            int(horizon_steps)
            if horizon_steps is not None
            else int(self.max_delay_steps)
        )
        steps = max(1, steps)
        shape = (steps, int(self._n_conn))
        if (
            "state_cache_pre_value_history" not in self._buffers
            or tuple(self.state_cache_pre_value_history.shape) != shape
            or self.state_cache_pre_value_history.device != self.device
            or self.state_cache_pre_value_history.dtype != self.dtype
        ):
            self._set_buffer(
                "state_cache_pre_value_history",
                torch.zeros(shape, device=self.device, dtype=self.dtype),
            )
        else:
            self.state_cache_pre_value_history.zero_()
        self._state_cache_recording_enabled = True
        return self

    def disable_state_cache_recording(self, *, release: bool = False):
        self._state_cache_recording_enabled = False
        if release and "state_cache_pre_value_history" in self._buffers:
            self._set_buffer(
                "state_cache_pre_value_history",
                torch.empty((0, 0), device=self.device, dtype=self.dtype),
            )
        return self

    def _record_pre_value_for_state_cache(self, x: torch.Tensor):
        if not getattr(self, "_state_cache_recording_enabled", False):
            return
        if (
            "state_cache_pre_value_history" not in self._buffers
            or self.state_cache_pre_value_history.numel() == 0
        ):
            return
        slot = int(self.current_time_step.detach().cpu().reshape(-1)[0].item())
        slot %= int(self.state_cache_pre_value_history.shape[0])
        self.state_cache_pre_value_history[slot].copy_(
            x.detach().to(self.device, self.dtype)
        )

    def _materialize_delivery_from_pre_value_history(
        self, age_history: torch.Tensor, *, apply_transform: bool = True
    ):
        """Rebuild the dense continuous delay buffer from unweighted value history."""
        age_history = age_history.detach().to(device=self.device, dtype=self.dtype)
        if age_history.ndim != 2 or age_history.shape[1] != self._n_conn:
            raise ValueError(
                "Cached ContinuousCon pre-value history has incompatible shape: "
                f"{tuple(age_history.shape)} vs (*, {self._n_conn})."
            )
        self.delivery_buffer.zero_()
        self.delivery_mask.zero_()
        if self._n_conn == 0 or age_history.shape[0] <= 1:
            return
        flat = self.delivery_buffer.view(-1)
        weight = self.weight().detach().to(device=self.device, dtype=self.dtype)
        delay_steps = self.delay_steps.to(device=self.device, dtype=torch.long)
        post_idx = self.post_idx.to(device=self.device, dtype=torch.long)
        max_age = min(int(age_history.shape[0]), int(self.max_delay_steps))
        for age in range(1, max_age):
            remaining = delay_steps - int(age)
            keep = remaining >= 0
            if not bool(keep.any()):
                continue
            row = age_history[age]
            if apply_transform and self._has_transform:
                # Cache reconstruction initializes detached runtime state. Apply
                # the current memoryless transform to the complete connection
                # row before selecting the edges whose deliveries remain pending.
                row = self.transform(row).detach()
            vals = row.index_select(0, torch.nonzero(keep, as_tuple=False).flatten())
            if vals.numel() == 0:
                continue
            con_idx = torch.nonzero(keep, as_tuple=False).flatten()
            vals = vals * weight.index_select(0, con_idx)
            flat_idx = remaining.index_select(
                0, con_idx
            ) * self._syn_numel + post_idx.index_select(0, con_idx)
            flat.index_add_(0, flat_idx.reshape(-1), vals.reshape(-1))
            self.delivery_mask.view(-1).index_fill_(0, flat_idx.reshape(-1), True)

    def n_connections(self):
        return int(self.n.item())

    def state_cache(self):
        """Package runtime state for Network.cache_state()/steady_state().

        Unlike state_dict_for_checkpoint(), this cache is detached and normalized
        so it can be restored by Network.initialize(...): row 0 of the cached
        delivery buffer is the next row due after the cached state.
        """
        cur_slot = _cache_current_slot(self.current_time_step)
        cache = {
            "kind": "continuous",
            "version": 4,
            "dt": _cache_dt_value(self.dt),
            "max_delay_steps": int(self.max_delay_steps),
        }
        if (
            hasattr(self, "state_cache_pre_value_history")
            and self.state_cache_pre_value_history.numel() > 0
            and getattr(self, "_state_cache_recording_enabled", False)
        ):
            cache["pre_value_history"] = _ring_rows_to_age_rows(
                self.state_cache_pre_value_history.detach(), cur_slot
            )
            cache["history_layout"] = "age"
            cache["value_layout"] = "raw"
            cache["param_invariant"] = True
        else:
            # Backward-compatible fallback: this is tied to the weights/delays
            # that produced the already-weighted future delivery rows.
            cache["delivery_buffer"] = torch.roll(
                self.delivery_buffer.detach(), -cur_slot, dims=0
            ).clone()
            cache["delivery_mask"] = torch.roll(
                self.delivery_mask.detach(), -cur_slot, dims=0
            ).clone()
            cache["param_invariant"] = False
        return cache

    def initialize_from_state_cache(
        self, state_cache, *, dt=None, rebuild_delays: bool = True
    ):
        """Restore a cache produced by state_cache() after initialize().

        Network calls this after ``initialize(...)`` has already refreshed
        ``WeightExpander.w`` and, when requested, rebuilt delay metadata.
        ``rebuild_delays`` remains True for direct calls so the method is
        self-contained; Network passes False to avoid a redundant second rebuild.
        """
        if not isinstance(state_cache, dict):
            raise TypeError("ContinuousCon state cache must be a dict")
        # Cached histories are replayed through the current expanded weights and
        # current integer delays.  If Network.initialize(...) just rebuilt delay
        # storage, only clear the buffer; otherwise rebuild from delay_ms.w now.
        if rebuild_delays:
            self._rebuild_delay_buffers()
        else:
            self._align_buffer_devices()
            self.delivery_buffer.zero_()
            self.delivery_mask.zero_()
        old_dt = float(state_cache.get("dt", _cache_dt_value(self.dt)))
        new_dt = float(_cache_dt_value(self.dt) if dt is None else dt)
        if "pre_value_history" in state_cache:
            hist = (
                state_cache["pre_value_history"]
                .detach()
                .to(device=self.device, dtype=self.dtype)
            )
            hist = _resample_continuous_history_age_rows(
                hist,
                old_dt,
                new_dt,
                n_limit=int(self.max_delay_steps),
            )
            # Version 2 caches recorded already-transformed values.  Keep those
            # loadable without applying the transform twice; version 3 stores raw
            # source values and deliberately replays the current transform.
            raw_values = (
                state_cache.get("value_layout") == "raw"
                or int(state_cache.get("version", 2)) >= 3
            )
            self._materialize_delivery_from_pre_value_history(
                hist, apply_transform=raw_values
            )
        else:
            cached = (
                state_cache["delivery_buffer"]
                .detach()
                .to(device=self.device, dtype=self.dtype)
            )
            if cached.ndim != 2 or cached.shape[1] != self._syn_numel:
                raise ValueError(
                    "Cached ContinuousCon delivery_buffer has incompatible shape: "
                    f"{tuple(cached.shape)} vs (*, {self._syn_numel})."
                )
            restored = _resample_continuous_time_rows(
                cached,
                old_dt,
                new_dt,
                n_limit=int(self.delivery_buffer.shape[0]),
            )
            if tuple(self.delivery_buffer.shape) != tuple(restored.shape):
                self._set_buffer(
                    "delivery_buffer",
                    torch.zeros_like(restored, device=self.device, dtype=self.dtype),
                )
            self.delivery_buffer.copy_(restored)
            cached_mask = state_cache.get("delivery_mask", None)
            if cached_mask is None:
                # Version 3 and older caches did not retain delivery presence.
                # A nonzero payload is the only recoverable evidence; genuine
                # scheduled zeros necessarily remain a best-effort limitation.
                cached_mask = cached != 0
            elif not torch.is_tensor(cached_mask):
                raise TypeError("Cached ContinuousCon delivery_mask must be a tensor")
            else:
                cached_mask = cached_mask.detach().to(
                    device=self.device, dtype=torch.bool
                )
                if tuple(cached_mask.shape) != tuple(cached.shape):
                    raise ValueError(
                        "Cached ContinuousCon delivery_mask has incompatible shape: "
                        f"{tuple(cached_mask.shape)} vs {tuple(cached.shape)}."
                    )
            restored_mask = _resample_continuous_time_rows(
                cached_mask,
                old_dt,
                new_dt,
                n_limit=int(self.delivery_mask.shape[0]),
            )
            if tuple(self.delivery_mask.shape) != tuple(restored_mask.shape):
                self._set_buffer(
                    "delivery_mask",
                    torch.zeros_like(
                        restored_mask, device=self.device, dtype=torch.bool
                    ),
                )
            self.delivery_mask.copy_(restored_mask)
        self.current_time_step.zero_()
        if hasattr(self, "t"):
            step = torch.round(
                self.t.to(self.device, dtype=self.dtype) / self.dt.to(self.dtype)
            ).long()
            self.global_step.copy_(step.reshape_as(self.global_step))
        else:
            self.global_step.zero_()
        self.detach()
        self._configure_advance_impl()
        return self

    def state_dict_for_checkpoint(self):
        return {
            "delivery_buffer": self.delivery_buffer,
            "delivery_mask": self.delivery_mask,
            "current_time_step": self.current_time_step,
            "global_step": self.global_step,
        }

    def restore_dict_from_checkpoint(self, state_dict):
        if not isinstance(state_dict, dict):
            raise TypeError("ContinuousCon checkpoint state must be a dict.")
        delivery = _require_checkpoint_tensor(
            state_dict,
            "delivery_buffer",
            shape=self.delivery_buffer.shape,
            dtype=self.dtype,
        )
        current = _require_checkpoint_tensor(
            state_dict,
            "current_time_step",
            shape=self.current_time_step.shape,
            dtype=torch.long,
        )
        global_step = _require_checkpoint_tensor(
            state_dict,
            "global_step",
            shape=self.global_step.shape,
            dtype=torch.long,
        )
        delivery_mask = state_dict.get("delivery_mask", None)
        if delivery_mask is None:
            # Older checkpoints cannot distinguish a scheduled zero from an
            # empty ring slot. Preserve loadability and recover every presence
            # bit that is inferable from the payload itself.
            delivery_mask = delivery != 0
        else:
            if not torch.is_tensor(delivery_mask):
                raise TypeError(
                    "ContinuousCon checkpoint 'delivery_mask' must be a tensor."
                )
            if tuple(delivery_mask.shape) != tuple(self.delivery_mask.shape):
                raise ValueError(
                    "ContinuousCon checkpoint 'delivery_mask' has shape "
                    f"{tuple(delivery_mask.shape)}, expected "
                    f"{tuple(self.delivery_mask.shape)}."
                )
            if delivery_mask.dtype != torch.bool:
                raise TypeError(
                    "ContinuousCon checkpoint 'delivery_mask' has dtype "
                    f"{delivery_mask.dtype}, expected torch.bool."
                )
        current_value = int(current.detach().cpu().reshape(-1)[0].item())
        if current_value < 0 or current_value >= int(self.max_delay_steps):
            raise ValueError(
                "ContinuousCon checkpoint current_time_step is outside the delay "
                f"ring: {current_value} not in [0, {int(self.max_delay_steps)})."
            )
        self.delivery_buffer = delivery.to(device=self.device)
        self.delivery_mask = delivery_mask.to(device=self.device).clone()
        self.current_time_step = current.to(device=self.device)
        self.global_step = global_step.to(device=self.device)
        return self

    def checkpoint_topology_signature(self):
        """Return immutable structure needed to validate fresh-object replay."""
        return {
            "kind": "continuous",
            "n_connections": int(self._n_conn),
            "synapse_numel": int(self._syn_numel),
            "max_delay_steps": int(self.max_delay_steps),
            "topology": _topology_tensor_digest(self.pre_idx, self.post_idx),
        }

    # ------------------------------------------------------------------
    # Hot-path helpers
    # ------------------------------------------------------------------

    def _maybe_reset_inputs(self):
        # Deprecated compatibility hook.  Continuous input resets are performed
        # by Network before advancing continuous projections.
        return None

    def _advance_counters(self):
        self.current_time_step.add_(1).remainder_(self.max_delay_steps)
        self.global_step.add_(1)

    def _pre_value(self):
        # The presynaptic variable should already live on the source device; only
        # move if necessary.  This avoids redundant no-op .to(...) dispatches in
        # the common single-device path.
        x_full = self.get_pre_var(self.pre)
        if x_full.device != self.pre_device or x_full.dtype != self.pre_dtype:
            x_full = x_full.to(device=self.pre_device, dtype=self.pre_dtype)
        x = x_full.reshape(-1).index_select(0, self.pre_idx)
        if x.device != self.device or x.dtype != self.dtype:
            x = x.to(device=self.device, dtype=self.dtype)
        # Record raw selected values. Restore reapplies the current memoryless
        # transform so parameter updates cannot leave stale transformed traffic.
        self._record_pre_value_for_state_cache(x)
        if self._has_transform:
            x = self.transform(x)
        return x

    def _weighted_pre_value(self):
        return self.weight() * self._pre_value()

    def _zeros_delivery(self):
        return torch.zeros(self._syn_numel, device=self.device, dtype=self.dtype)

    def _scatter_all(self, values):
        if self._direct_all:
            return values
        out = self._zeros_delivery()
        out.index_add_(0, self.post_idx, values)
        return out

    def _scatter_zero_subset(self, values):
        out = self._zeros_delivery()
        vals = values.index_select(0, self.zero_con_idx)
        out.index_add_(0, self.post_idx_zero, vals)
        return out

    def _read_and_clear_current_row(self):
        cur_idx = self.current_time_step
        delayed_delivery = self.delivery_buffer.index_select(0, cur_idx).squeeze(0)
        delayed_mask = self.delivery_mask.index_select(0, cur_idx).squeeze(0)
        self.delivery_buffer.index_fill_(0, cur_idx, 0.0)
        self.delivery_mask.index_fill_(0, cur_idx, False)
        return cur_idx, delayed_delivery, delayed_mask

    def _deliver(self, flat_delivery, flat_mask):
        self.syn.continuous_receive(
            flat_delivery.view(*self.syn.shape_f),
            self,
            input=self.input,
            reduce=self.reduce,
            mask=flat_mask.view(*self.syn.shape_f),
        )

    def _schedule_all_delayed_uniform(self, cur_idx, weighted):
        future = (cur_idx + int(self._uniform_delay_step)).remainder(
            self.max_delay_steps
        )
        flat = future * self._syn_numel + self.post_idx
        self.delivery_buffer.view(-1).index_add_(0, flat.reshape(-1), weighted)
        self.delivery_mask.view(-1).index_fill_(0, flat.reshape(-1), True)

    def _schedule_selected_delayed_uniform(self, cur_idx, weighted):
        vals = weighted.index_select(0, self.nonzero_con_idx)
        future = (cur_idx + int(self._uniform_delay_step)).remainder(
            self.max_delay_steps
        )
        flat = future * self._syn_numel + self.post_idx_nz
        self.delivery_buffer.view(-1).index_add_(0, flat.reshape(-1), vals)
        self.delivery_mask.view(-1).index_fill_(0, flat.reshape(-1), True)

    def _schedule_all_delayed_mixed(self, cur_idx, weighted):
        base = cur_idx * self._syn_numel
        flat = (base + self.flat_delay_offsets_nz).remainder(self._delivery_numel)
        self.delivery_buffer.view(-1).index_add_(0, flat.reshape(-1), weighted)
        self.delivery_mask.view(-1).index_fill_(0, flat.reshape(-1), True)

    def _schedule_selected_delayed_mixed(self, cur_idx, weighted):
        vals = weighted.index_select(0, self.nonzero_con_idx)
        base = cur_idx * self._syn_numel
        flat = (base + self.flat_delay_offsets_nz).remainder(self._delivery_numel)
        self.delivery_buffer.view(-1).index_add_(0, flat.reshape(-1), vals)
        self.delivery_mask.view(-1).index_fill_(0, flat.reshape(-1), True)

    # ------------------------------------------------------------------
    # Specialized advance paths
    # ------------------------------------------------------------------

    def _advance_empty(self):
        self._advance_counters()

    def _advance_all_immediate(self):
        weighted = self._weighted_pre_value()
        self._deliver(self._scatter_all(weighted), self.post_delivery_mask)
        # All delays are zero, so current_time_step remains zero modulo one.
        self.global_step.add_(1)

    def _advance_all_delayed_uniform(self):
        cur_idx, delayed_delivery, delayed_mask = self._read_and_clear_current_row()
        self._deliver(delayed_delivery, delayed_mask)
        weighted = self._weighted_pre_value()
        self._schedule_all_delayed_uniform(cur_idx, weighted)
        self._advance_counters()

    def _advance_all_delayed_mixed(self):
        cur_idx, delayed_delivery, delayed_mask = self._read_and_clear_current_row()
        self._deliver(delayed_delivery, delayed_mask)
        weighted = self._weighted_pre_value()
        self._schedule_all_delayed_mixed(cur_idx, weighted)
        self._advance_counters()

    def _advance_mixed_uniform(self):
        cur_idx, delayed_delivery, delayed_mask = self._read_and_clear_current_row()
        weighted = self._weighted_pre_value()
        immediate = self._scatter_zero_subset(weighted)
        self._deliver(
            delayed_delivery + immediate,
            delayed_mask | self.post_zero_delivery_mask,
        )
        self._schedule_selected_delayed_uniform(cur_idx, weighted)
        self._advance_counters()

    def _advance_mixed(self):
        cur_idx, delayed_delivery, delayed_mask = self._read_and_clear_current_row()
        weighted = self._weighted_pre_value()
        immediate = self._scatter_zero_subset(weighted)
        self._deliver(
            delayed_delivery + immediate,
            delayed_mask | self.post_zero_delivery_mask,
        )
        self._schedule_selected_delayed_mixed(cur_idx, weighted)
        self._advance_counters()


class NetCon(Referency):
    """
    Event-based connectivity wrapper between a pre-synaptic source and a
    post-synaptic synapse mechanism.

    Overview
    --------
    A ``NetCon`` instance mediates communication between:

    * a **pre-synaptic source** (population module or :class:`NetStim`),
    * a **post-synaptic synapse mechanism** (module exposing ``net_receive``), and
    * an implicit **post population** that owns the synapse state.

    Conceptually, each time step performs three operations:

    1. **Event detection**:

       - Observe a pre-synaptic variable (by default ``pre.v`` or an explicit
         gate on a :class:`NetStim`).
       - Detect intrinsic events (hard or surrogate spiking) via threshold
         crossings, or directly consume externally scheduled events.

    2. **Weighting and delay**:

       - Combine intrinsic and scheduled events into a scalar gate per
         connection.
       - Multiply by a per-connection weight (optionally differentiable).
       - Write the resulting payload into a per-synapse delay line
         (delivery buffer) according to per-connection delays
         (integer or differentiable).

    3. **Delivery**:

       - At each step, read the current row of the delay buffer.
       - Call ``syn.net_receive(payload, netcon=self)`` on the synapse
         mechanism, with payload reshaped to ``syn.shape_f``.

    The class is designed to:

    * Support heterogeneous device placement for pre, post, and synapse modules.
    * Provide differentiable pathways for:

      - weights,
      - delays,
      - spiking / event detection,
      - and optionally scheduled event times.

    * Provide explicit *scheduled* events that can be trained by reference.

    Construction and lifecycle
    --------------------------
    ``NetCon`` objects are normally constructed by
    :class:`dendra.models.networks.Network` during ``Network.build()``.
    Initialization and stepping are managed by the network's simulation loop.

    1. **Construction** (usually internal):

       .. code-block:: python

           nc = NetCon(
               pre=pre_pop,
               pre_idx=pre_indices,
               thresholds=thresholds,
               post=post_pop,
               post_idx=post_indices,
               post_syn=synapse_module,
               weight=weight_param,
               delay=delay_param,
               dt=dt,
               pre_var="v",         # optional
               max_delay=None,      # optional
           )

    2. **Differentiable configuration (optional but required for training)**:

       Call :meth:`set_diff_config` once before the first differentiable
       simulation step to configure which components are treated as
       differentiable:

       .. code-block:: python

           nc.set_diff_config(
               diff_weights=True,
               diff_delays=True,
               diff_spiking=True,
               diff_scheduled_times=True,
               # taps, sigma, tau, sched_width as appropriate
           )

    3. **Initialization**:

       Call :meth:`initialize` before stepping:

       .. code-block:: python

           nc.train()         # if using standard PyTorch training mode
           nc.set_diff_config(...)  # must be called before first advance_diff
           nc.initialize(
               reinit_weights=True,
               reinit_delays=True,
               clear_deliveries=True,
           )

       ``initialize`` will:

       * select ``nc.advance`` to point to either :meth:`advance_diff` or
         :meth:`advance_non_diff` depending on ``nc.training``,
       * (re)initialize weight and delay parameter modules,
       * rebuild internal delay buffers if delays changed,
       * reset the internal timing state, and
       * detach buffers from any pre-existing computation graph
         (fresh forward pass).

    4. **Simulation loop**:

       .. code-block:: python

           for step in range(T):
               nc.advance()  # calls advance_diff or advance_non_diff

       At each call, ``NetCon`` delivers events to ``post_syn`` via
       ``post_syn.net_receive(...)``.

    Scheduling: value- vs reference-mode
    ------------------------------------
    In addition to intrinsic spiking, ``NetCon`` supports *scheduled* events.
    These are particularly useful for:

    * injecting known spike trains or stimulus patterns,
    * training event **weights** and/or **times** as separate learnable tensors,
    * conditioning the synaptic drive on external controllers or task structure.

    There are two orthogonal axes:

    1. **Value mode** (direct tensors):

       - You provide explicit event weights and/or times (as numbers or Tensors).
       - Gradients flow into those tensors if they require gradients and are
         wired into your optimization loop.

       Value-mode APIs:

       * :meth:`schedule`
       * :meth:`schedule_time_ref` with ``weight`` specified as a Tensor.

    2. **Reference mode** (indices into bound sources):

       - You bind a *source* tensor once via :meth:`bind_weight_source` or
         :meth:`bind_time_source`.
       - You then schedule events by storing integer indices into the source;
         the values are looked up on-the-fly during the simulation.
       - Gradients flow into the bound source tensors, not the indices.

       Reference-mode APIs:

       * :meth:`schedule_ref` for reference-mode weights,
       * :meth:`schedule_time_ref` for reference-mode times.

    Mixed use is allowed: you can combine intrinsic spiking, value-mode
    scheduled events, and reference-mode scheduled events in the same ``NetCon``.

    Order-of-operations for training scheduled **weights**
    ------------------------------------------------------
    To make scheduled event weights trainable by reference:

    1. **Create a trainable weight source**:

       .. code-block:: python

           # Example: one scalar weight per planned event
           sched_w = torch.nn.Parameter(
               torch.zeros(num_events, device=device, dtype=dtype)
           )

    2. **Bind the weight source once**:

       .. code-block:: python

           nc.bind_weight_source(sched_w)

       This does not copy data; it stores a reference to ``sched_w``. The
       tensor must live on the same device as the synapse.

    3. **Schedule events by index** using :meth:`schedule_ref`:

       .. code-block:: python

           # con_indices or pre_indices must map to concrete connections
           nc.schedule_ref(
               con_indices=con_idx,   # or pre_indices=...
               times_ms=t_ms,         # float times in ms
               weight_idx=w_idx,      # Long tensor of indices into sched_w
               allow_past=False,
           )

       On each time step:

       * ``NetCon`` will look up per-event weights as
         ``sched_w[weight_idx]``.
       * These contribute to the per-connection gates, and hence to the
         synaptic deliveries.
       * Gradients from the loss will flow back into ``sched_w``.

    Order-of-operations for training scheduled **times**
    ----------------------------------------------------
    To make scheduled event **times** trainable by reference:

    1. **Create a trainable time source**:

       .. code-block:: python

           # Example: one scalar time per event, in ms
           sched_t = torch.nn.Parameter(
               torch.zeros(num_events, device=device, dtype=dtype)
           )

    2. **Bind the time source once**:

       .. code-block:: python

           nc.bind_time_source(sched_t)

    3. **Schedule events by time index** using :meth:`schedule_time_ref`:

       .. code-block:: python

           nc.schedule_time_ref(
               con_indices=con_idx,   # or pre_indices=...
               time_idx=t_idx,        # Long tensor of indices into sched_t
               weight=1.0,            # scalar or tensor (value mode)
               allow_past=False,
           )

       On each time step:

       * ``NetCon`` will read the *current* values from ``sched_t[time_idx]``.
       * The resulting times are interpreted in one of two ways:

         - If ``diff_scheduled_times=True`` in :meth:`set_diff_config`,
           events are smeared with a differentiable triangular kernel over
           time steps, so gradients can move event times.
         - If ``diff_scheduled_times=False``, each time is assigned to the first
           integer step at or after the event and treated as exact.

       In both cases, gradients flow into ``sched_t`` when using the
       differentiable configuration.

    Interactions with global time
    -----------------------------
    Many scheduling methods accept an ``allow_past`` flag:

    * When ``allow_past=False`` (default), events whose scheduled steps are
      **earlier** than the current internal ``global_step`` are silently
      dropped.
    * When ``allow_past=True``, events are kept regardless of their scheduled
      step index.

    In practice, this means that you usually want to:

    * Call :meth:`initialize` (which also sets ``global_step``) before
      scheduling events, or
    * Explicitly pass ``allow_past=True`` when scheduling relative to a
      fixed origin (e.g., step 0) while reusing a ``NetCon``.

    Notes
    -----
    * ``NetCon`` automatically tracks and realigns its internal buffers when
      moved across devices via ``.to(...)`` or when the peer modules move.
    * The class is intentionally conservative about in-loop memory allocation:
      buffers are pre-allocated where possible and reused across steps.
    """

    def __init__(
        self,
        pre,
        pre_idx,
        thresholds,
        post,
        post_idx,
        post_syn,
        weight,
        delay,
        dt,
        pre_var=None,
        max_delay=None,
        track_events: bool = False,
        delay_backend: Literal[
            "dense", "sparse_calendar", "bitpacked_history"
        ] = "dense",
        train_delay_backend: Literal["dense", "source_history", "auto"] = "auto",
        device=None,
        dtype=None,
        pre_device=None,
        pre_dtype=None,
    ):
        """
        Parameters
        ----------
        pre : :class:`~dendra.models.core.Population` or :class:`~dendra.models.networks.NetStim`
            Pre-synaptic population module or :class:`NetStim`. The module must
            implement ``device()`` and ``dtype()``; when ``pre`` is not a
            :class:`NetStim`, it is also expected to expose the attribute named
            by ``pre_var`` (e.g., ``v``) which holds the pre-synaptic state
            used for spiking.
        pre_idx : torch.Tensor
            1-D Long tensor of indices into the flattened pre-synaptic variable
            (e.g., ``pre.v.view(-1)[pre_idx]``). Length defines the number of
            managed connections.
        thresholds : torch.Tensor
            Per-connection spiking thresholds in the same shape as ``pre_idx``.
            Entries may be NaN to indicate:

            * all-NaN: skip thresholding and treat the selected pre variable as
              a continuous gate;
            * mixed: apply thresholding for finite entries, and use the raw
              pre variable wherever the threshold is NaN.

        post : :class:`~dendra.models.core.Population`
            Post-synaptic population module that owns the synapse mechanism.
            Must implement ``device()`` and ``dtype()`` interfaces and is used
            mainly for device alignment.
        post_idx : torch.Tensor
            1-D Long tensor of indices into the flattened synapse output space;
            used to scatter connection events into the synapse's delivery buffer.
        post_syn : :class:`dendra.models.mechanisms.Synapse`
            Synapse mechanism module. Must expose:

            * ``shape_f``: final tensor shape of per-synapse payloads, and
            * ``net_receive(payload, netcon)``: function invoked each step with
              delivery buffer slice reshaped to ``shape_f``.

        weight : torch.nn.Module
            Parameter-like module representing per-connection weights. Must support:

            * ``weight.init(reinit: bool)``,
            * ``weight()`` returning the current weight tensor, and
            * ``weight.w`` exposing the underlying tensor.

        delay : torch.nn.Module
            Parameter-like module representing per-connection delays in ms. Must
            support:

            * ``delay.init(reinit: bool)`` and
            * ``delay()`` returning the current delays tensor.

        dt : float
            Simulation time step in milliseconds.
        pre_var : str, optional
            Name of the attribute on ``pre`` to use as the spiking variable when
            ``pre`` is not a :class:`NetStim`. Defaults to ``"v"``.
        max_delay : float, optional
            Optional maximum delay (ms). If provided, the internal delay buffer
            depth is set to ``int(max_delay / dt) + 1``; otherwise it is inferred
            from the maximum delay in ``delay``.
        track_events : bool, optional
            If True, allocate and maintain the historical ``event_queue`` used
            for per-connection delivery introspection. If False (default), skip
            the large ``[max_delay_steps, n_connections]`` int32 queue and keep
            only the lightweight current-step ``events`` buffer, which is zeroed
            on each non-differentiable step.
        delay_backend : {"dense", "sparse_calendar", "bitpacked_history"}, optional
            Delay-line implementation for non-differentiable inference. The
            default ``"dense"`` keeps the existing ``[max_delay_steps,
            syn_numel]`` ring buffer, but uses precomputed integer-delay routing
            metadata and specialized uniform-/mixed-delay hot paths. The
            ``"sparse_calendar"`` backend stores pending nonzero inference
            deliveries in Python-side calendar buckets and uses only a reusable
            one-row dense scratch buffer before calling ``syn.net_receive(...)``.
            ``"bitpacked_history"`` is an inference-only source-spike history
            backend for large SNNs: it stores only packed source spikes over the
            delay horizon and reconstructs the dense postsynaptic receive payload
            with a lazy C++/CUDA extension on eligible CUDA paths, falling back
            to PyTorch otherwise. The separate Triton implementation is an
            experimental/reference path and is not selected automatically.
        train_delay_backend : {"dense", "source_history", "auto"}, optional
            Differentiable training delay backend. ``"dense"`` preserves the
            fully general dense differentiable delay buffer. ``"source_history"``
            uses source-level spike/gate history when the NetCon has exact
            source-level event semantics. With ``diff_spiking=False`` this uses
            packed boolean source history and still differentiates weights and
            delays for realized events. With ``diff_spiking=True`` this uses a
            floating source-gate history so surrogate gradients can flow through
            source-level thresholding. ``"auto"`` selects source history when
            exact and otherwise falls back to dense. Default is ``"auto"`` so
            source-level projections use the compact training backend when it is
            safe, while per-connection/scheduled cases retain dense semantics.

        Notes
        -----
        This constructor is typically invoked by
        :class:`dendra.models.networks.Network` during ``Network.build()``; user
        code usually interacts with :class:`NetCon` via high-level network
        building APIs rather than constructing it directly.
        """
        super().__init__()

        self.weight = weight
        self.syn = post_syn
        self.pre = pre
        self.post = post

        # Track peer devices/dtypes and keep NetCon buffers aligned with them.
        self._refresh_peer_devices(
            device=device, dtype=dtype, pre_device=pre_device, pre_dtype=pre_dtype
        )

        # Ensure parameter modules live on the delivery (post/synapse) device.
        self.weight = self.weight.to(device=self.device, dtype=self.dtype)
        delay = delay.to(device=self.device, dtype=self.dtype)

        self.dt = torch.tensor(dt, device=self.device, dtype=self.dtype)
        self.max_delay = max_delay
        self.track_events = bool(track_events)
        if delay_backend not in ("dense", "sparse_calendar", "bitpacked_history"):
            raise ValueError(
                "delay_backend must be one of 'dense', 'sparse_calendar', or 'bitpacked_history'."
            )
        self.delay_backend = delay_backend
        if train_delay_backend not in ("dense", "source_history", "auto"):
            raise ValueError(
                "train_delay_backend must be one of 'dense', 'source_history', or 'auto'."
            )
        self.train_delay_backend = train_delay_backend
        self._calendar_compact_threshold = 32
        self._sparse_calendar: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._sparse_event_calendar: dict[
            int, list[tuple[torch.Tensor, torch.Tensor]]
        ] = {}
        self._bitpack_bits_per_word = 63
        self._bitpack_can_use = False
        self._bitpack_mode = "disabled"
        self._bitpack_ineligible_reason = None

        self.delay_ms = delay
        delay = delay.init().w

        if pre_var is None:
            pre_var = "v"

        self.pre_var = pre_var
        self.get_pre_var = make_getattr(pre_var)

        if isinstance(pre, NetStim):
            self.determine_spiking = self.determine_spiking_ns
        else:
            self.determine_spiking = self.determine_spiking_var

        self.register_buffer(
            "pre_idx", pre_idx.flatten().to(self.pre_device, dtype=torch.long)
        )
        self.register_buffer(
            "post_idx", post_idx.flatten().to(self.device, dtype=torch.long)
        )
        self.register_buffer(
            "threshold",
            torch.nan_to_num(thresholds)
            .flatten()
            .to(self.pre_device, dtype=self.pre_dtype),
        )

        nan_thresh = torch.isnan(thresholds).to(self.pre_device)
        self.register_buffer("thresh_is_nan", nan_thresh)

        self.skip_thresholding = bool(nan_thresh.all())
        self.apply_masking = bool(nan_thresh.any()) and not self.skip_thresholding

        self.register_buffer(
            "syn_numel",
            torch.prod(torch.tensor(self.syn.shape_f, device=self.device)).to(
                self.device, dtype=torch.long
            ),
        )
        self._syn_numel = int(self.syn_numel.item())  # cache for performance

        self.register_buffer(
            "n",
            torch.tensor(self.pre_idx.numel()).to(self.device, dtype=torch.long),
        )
        self.register_buffer(
            "pre_range",
            torch.arange(self.n.item(), device=self.pre_device, dtype=torch.long),
        )
        n_pre = self.n.item()
        assert len(self.threshold) == n_pre, (
            "Thresholds must match the number of pre synaptic locations."
        )
        assert delay.shape[0] == self.pre_idx.shape[0], (
            "Delay tensor shape must match pre_idx."
        )

        # Spike-state buffers are allocated by ``_ensure_spike_state_storage``.
        # The bitpacked inference backend intentionally avoids persistent
        # per-connection spike-state buffers.
        self.register_buffer(
            "has_spiked", torch.empty(0, device=self.pre_device, dtype=torch.bool)
        )
        self.register_buffer(
            "is_spiking",
            torch.empty(0, device=self.pre_device, dtype=self.pre_dtype),
        )

        # --- Delay Handling Logic (Compiler-Friendly) ---
        delay_steps = (delay / dt).round().long()
        self.register_buffer("delay_steps", delay_steps.flatten().to(self.device))

        self.max_delay_steps = self._compute_max_delay_steps()

        # Build source-level metadata before choosing the inference storage layout.
        # The bitpacked backend is exact only when intrinsic spiking can be reduced
        # to one binary event stream per unique presynaptic source.
        self._rebuild_source_csr_metadata()
        self._configure_bitpacked_history_metadata()
        if self.delay_backend == "bitpacked_history" and not self._bitpack_can_use:
            raise ValueError(
                "NetCon delay_backend='bitpacked_history' is not exact for this "
                f"connection: {self._bitpack_ineligible_reason}. "
                "Use delay_backend='dense' or provide a pre-side binary event "
                "pre_var / source-consistent threshold."
            )

        # In sparse-calendar or bitpacked-history inference mode, ``delivery_buffer``
        # is intentionally kept as a one-row dense scratch buffer so existing
        # synaptic mechanisms can continue to receive
        # ``syn.net_receive(dense_payload, netcon)``.
        buffer_depth = (
            1
            if self.delay_backend in ("sparse_calendar", "bitpacked_history")
            else self.max_delay_steps
        )
        buffer_shape = (buffer_depth, self.syn_numel.item())
        self.register_buffer(
            "delivery_buffer",
            torch.zeros(buffer_shape, device=self.device, dtype=self.dtype),
        )
        if self.track_events and self.delay_backend == "dense":
            self.register_buffer(
                "event_queue",
                torch.zeros(
                    (self.max_delay_steps, n_pre),
                    device=self.device,
                    dtype=torch.int32,
                ),
            )
        event_len = (
            0
            if self.delay_backend == "bitpacked_history" and not self.track_events
            else n_pre
        )
        self.register_buffer(
            "events", torch.zeros(event_len, device=self.device, dtype=torch.int32)
        )

        self.register_buffer(
            "current_time_step", torch.tensor([0], device=self.device, dtype=torch.long)
        )

        # --- Pre-computed tensor for masking, avoids creating tensors in the loop ---
        self.register_buffer(
            "time_indices", torch.arange(self.max_delay_steps, device=self.device)
        )

        self.train_flags = None
        # A dense-to-dense mode switch can preserve the live delay ring. Compact
        # inference/source-history layouts cannot be migrated merely by toggling
        # nn.Module.training; Network execution rejects them until initialize().
        self._mode_requires_initialize = False

        # scheduled events
        # Python-side calendar: abs_step -> list of (pre_idx:int, weight:float)
        self._sched_pre = {}

        # flat calendar (initially empty)
        self.register_buffer(
            "sched_pre_idx", torch.empty(0, device=self.device, dtype=torch.long)
        )
        self.register_buffer(
            "sched_abs_step", torch.empty(0, device=self.device, dtype=torch.long)
        )

        # ---- Value mode (snapshot) ----
        # store the actual per-event weights; NOT a Parameter, NOT a buffer (to keep graph)
        self.sched_weight = torch.empty(0, device=self.device, dtype=self.dtype)

        self.register_buffer(
            "global_step", torch.tensor([0], device=self.device, dtype=torch.long)
        )
        self._n_conn: int = int(self.pre_idx.numel())  # Python int
        con_range = (
            torch.empty(0, device=self.device, dtype=torch.long)
            if self.delay_backend == "bitpacked_history" and not self.track_events
            else torch.arange(
                self.pre_idx.numel(), device=self.device, dtype=torch.long
            )
        )
        self.register_buffer("con_range", con_range)

        # Dense inference routing metadata is static between delay rebuilds and
        # avoids recomputing future slots/flat indices every step.
        self._rebuild_dense_delay_metadata()

        # connection-schedule storage (fixed-length vectors; all per-connection)
        self.register_buffer(
            "sched_con_idx", torch.empty(0, device=self.device, dtype=torch.long)
        )
        self.register_buffer(
            "sched_abs_step", torch.empty(0, device=self.device, dtype=torch.long)
        )
        # value-mode per-event weights (kept as a Tensor to preserve autograd to producers)
        self.sched_weight = torch.empty(0, device=self.device, dtype=self.dtype)
        # reference-mode source + indices
        self._sched_w_source = None
        self.register_buffer(
            "sched_weight_idx", torch.empty(0, device=self.device, dtype=torch.long)
        )

        self.sched_time_ms = torch.empty(
            0, device=self.device, dtype=self.dtype
        )  # NOT a buffer

        # optional reference-mode source + indices for times
        self._sched_t_source = None
        self.register_buffer(
            "sched_time_idx", torch.empty(0, device=self.device, dtype=torch.long)
        )

        # optional introspection (per-connection)
        self.register_buffer(
            "sched_wsum",
            torch.zeros(self._n_conn, device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "sched_counts",
            torch.zeros(self._n_conn, device=self.device, dtype=torch.int32),
        )

        # Final sanity alignment (important if builder later calls .to()).
        self._align_buffer_devices()

    def _move_buffer(self, name: str, device: torch.device, dtype=None):
        if not hasattr(self, name):
            return
        buf = getattr(self, name)
        if buf is None:
            return
        target_dtype = dtype if dtype is not None else buf.dtype
        if buf.device != device or buf.dtype != target_dtype:
            setattr(
                self,
                name,
                buf.to(device=device, dtype=target_dtype),
            )

    def _set_buffer(self, name: str, value: torch.Tensor):
        """Register or replace a non-parameter tensor buffer."""
        if name in self._buffers:
            setattr(self, name, value)
        else:
            self.register_buffer(name, value)

    def _empty_buffer(self, *, device=None, dtype=torch.long, shape=(0,)):
        """Create an empty tensor on the requested device/dtype.

        This small helper is used during low-memory mode switches to eagerly
        release large stale buffers before any new connection-sized temporary is
        materialized.  Rebinding a registered buffer to an empty tensor returns
        the old allocation to PyTorch's caching allocator, making it immediately
        reusable by subsequent PyTorch allocations in the same process.
        """
        return torch.empty(
            shape,
            device=device if device is not None else self.device,
            dtype=dtype,
        )

    def _empty_long(self, device=None):
        return torch.empty(
            0, device=device if device is not None else self.device, dtype=torch.long
        )

    def _empty_bool(self, device=None):
        return torch.empty(
            0, device=device if device is not None else self.device, dtype=torch.bool
        )

    def _empty_dtype(self, device=None):
        return torch.empty(
            0, device=device if device is not None else self.device, dtype=self.dtype
        )

    def _rebuild_source_csr_metadata(self):
        """Build source→connection CSR metadata shared by scheduling and bitpacking."""
        with torch.no_grad():
            pre_idx = self.pre_idx.to(device=self.pre_device, dtype=torch.long)
            if pre_idx.numel() == 0:
                empty = torch.empty(0, device=self.pre_device, dtype=torch.long)
                self._csr_pre_ids = empty
                self._csr_starts = empty
                self._csr_counts = empty
                self._csr_conidx_sorted = empty
                return

            vals, order = torch.sort(pre_idx)
            uniq, counts = torch.unique_consecutive(vals, return_counts=True)
            starts = torch.cat(
                [
                    torch.zeros(1, device=self.pre_device, dtype=torch.long),
                    counts.cumsum(0)[:-1],
                ],
                dim=0,
            )
            self._csr_pre_ids = uniq
            self._csr_starts = starts
            self._csr_counts = counts
            self._csr_conidx_sorted = order.to(device=self.pre_device, dtype=torch.long)

    def _set_empty_bitpack_buffers(self):
        """Install empty bitpack buffers so device movement/state inspection is stable."""
        empty_pre_long = torch.empty(0, device=self.pre_device, dtype=torch.long)
        empty_dev_long = torch.empty(0, device=self.device, dtype=torch.long)
        empty_dev_i64 = torch.empty(0, device=self.device, dtype=torch.int64)
        empty_pre_bool = torch.empty(0, device=self.pre_device, dtype=torch.bool)
        empty_pre_dtype = torch.empty(0, device=self.pre_device, dtype=self.pre_dtype)
        self._set_buffer("bitpack_source_pre_idx", empty_pre_long)
        self._set_buffer("bitpack_source_word_idx", empty_dev_long)
        self._set_buffer("bitpack_source_bit_mask", empty_dev_i64)
        self._set_buffer("bitpack_conn_source_pos", empty_dev_long)
        self._set_buffer("bitpack_conn_word_idx", empty_dev_long)
        self._set_buffer("bitpack_conn_bit_mask", empty_dev_i64)
        self._set_buffer("bitpack_source_threshold", empty_pre_dtype)
        self._set_buffer("bitpack_source_has_spiked", empty_pre_bool)
        self._set_buffer("source_history_conn_source_pos", empty_dev_long)
        self._set_buffer(
            "spike_history_packed",
            torch.empty((0, 0), device=self.device, dtype=torch.int64),
        )
        self._set_buffer(
            "source_gate_history",
            torch.empty((0, 0), device=self.device, dtype=self.dtype),
        )

    def _resize_bitpacked_spike_history(self, *, clear: bool = True):
        """Resize/clear bitpacked runtime history without rebuilding topology metadata.

        Delay reinitialization changes ``delay_steps`` and can change
        ``max_delay_steps``, but it does not change the source→connection layout.
        Rebuilding the full bitpack metadata would allocate several
        connection-sized tensors.  This helper touches only the small
        ``[max_delay_steps, n_source_words]`` history and threshold state.
        """
        if not getattr(self, "_bitpack_can_use", False):
            return

        bits = int(self._bitpack_bits_per_word)
        n_source = int(
            getattr(
                self, "bitpack_source_pre_idx", self._empty_long(self.pre_device)
            ).numel()
        )
        n_words = max(1, (n_source + bits - 1) // bits)
        desired_shape = (int(self.max_delay_steps), n_words)

        if (
            "spike_history_packed" not in self._buffers
            or tuple(self.spike_history_packed.shape) != desired_shape
            or self.spike_history_packed.device != self.device
            or self.spike_history_packed.dtype != torch.int64
        ):
            self._set_buffer(
                "spike_history_packed",
                torch.zeros(desired_shape, device=self.device, dtype=torch.int64),
            )
        elif clear:
            self.spike_history_packed.zero_()

        if (
            clear
            and hasattr(self, "bitpack_source_has_spiked")
            and self.bitpack_source_has_spiked.numel() > 0
        ):
            self.bitpack_source_has_spiked.zero_()

    def _configure_bitpacked_history_metadata(self, *, for_training: bool = False):
        """Validate and allocate metadata for source-level bitpacked/source-history backends.

        The source-level backends are exact only when each unique presynaptic
        source has one event/gate stream that can be reused by all outgoing
        connections.  This is true for NetStim sources, explicit pre-side
        event/gate variables supplied via ``pre_var`` with NaN thresholds, and
        thresholded variables whose thresholds are source-consistent across all
        outgoing connections.  Packed inference/hard training interpret the
        stream as binary; floating source-history training stores the gate value.
        """
        self._bitpack_can_use = False
        self._bitpack_mode = "disabled"
        self._bitpack_ineligible_reason = None
        self._set_empty_bitpack_buffers()

        needs_source_level_history = self.delay_backend == "bitpacked_history" or bool(
            for_training
        )
        if not needs_source_level_history:
            return

        if self.track_events:
            self._bitpack_ineligible_reason = (
                "track_events=True needs per-connection delivery history"
            )
            return
        if self.pre_device != self.device:
            self._bitpack_ineligible_reason = "pre and post/synapse devices differ"
            return
        if self.apply_masking:
            self._bitpack_ineligible_reason = (
                "mixed finite/NaN thresholds require per-connection gates"
            )
            return
        mode = None
        source_threshold = torch.empty(0, device=self.pre_device, dtype=self.pre_dtype)

        if isinstance(self.pre, NetStim):
            mode = "netstim"
        elif self.skip_thresholding:
            # ``threshold=None``/NaN means the selected pre variable is already a
            # gate.  Bitpacking deliberately treats this as a binary pre-side
            # event stream, so we only allow explicit non-voltage pre variables.
            if self.pre_var in (None, "v"):
                self._bitpack_ineligible_reason = "NaN thresholds on the default voltage variable are continuous gates, not binary source spikes"
                return
            mode = "pre_var"
        else:
            # Thresholded mode is source-level only if all edges from the same
            # presynaptic source share the same threshold.  This accepts global
            # uniform thresholds and the slightly more general per-source uniform case.
            if bool(self.thresh_is_nan.any().item()):
                self._bitpack_ineligible_reason = "NaN thresholds are present"
                return
            if self._csr_pre_ids.numel() == 0:
                mode = "threshold"
                source_threshold = torch.empty(
                    0, device=self.pre_device, dtype=self.pre_dtype
                )
            else:
                order = self._csr_conidx_sorted.to(
                    device=self.pre_device, dtype=torch.long
                )
                thresh_sorted = self.threshold.index_select(0, order).to(self.pre_dtype)
                starts = self._csr_starts.to(device=self.pre_device, dtype=torch.long)
                counts = self._csr_counts.to(device=self.pre_device, dtype=torch.long)
                source_threshold = thresh_sorted.index_select(0, starts)
                expanded = torch.repeat_interleave(source_threshold, counts)
                if not bool(torch.all(thresh_sorted == expanded).item()):
                    self._bitpack_ineligible_reason = "threshold differs across outgoing edges from at least one source"
                    return
                mode = "threshold"

        bits = int(self._bitpack_bits_per_word)
        source_pre_idx = self._csr_pre_ids.to(device=self.pre_device, dtype=torch.long)
        n_source = int(source_pre_idx.numel())
        n_words = max(1, (n_source + bits - 1) // bits)
        source_pos = torch.arange(n_source, device=self.device, dtype=torch.long)
        source_word_idx = source_pos.div(bits, rounding_mode="floor")
        source_bit_mask = torch.ones_like(source_pos, dtype=torch.int64) << (
            source_pos % bits
        ).to(torch.int64)

        if n_source == 0:
            conn_source_pos = torch.empty(0, device=self.device, dtype=torch.long)
        else:
            conn_source_pos = torch.searchsorted(
                source_pre_idx.to(device=self.device),
                self.pre_idx.to(device=self.device, dtype=torch.long),
            ).to(device=self.device, dtype=torch.long)
        conn_word_idx = conn_source_pos.div(bits, rounding_mode="floor")
        conn_bit_mask = torch.ones_like(conn_source_pos, dtype=torch.int64) << (
            conn_source_pos % bits
        ).to(torch.int64)

        self._set_buffer("bitpack_source_pre_idx", source_pre_idx)
        self._set_buffer("bitpack_source_word_idx", source_word_idx)
        self._set_buffer("bitpack_source_bit_mask", source_bit_mask)
        # ``conn_source_pos`` is only an initialization intermediate.  Persisting
        # it costs another n_conn int64 buffer and is not used by the hot path.
        self._set_buffer(
            "bitpack_conn_source_pos", self._empty_buffer(dtype=torch.long)
        )
        self._set_buffer("bitpack_conn_word_idx", conn_word_idx)
        self._set_buffer("bitpack_conn_bit_mask", conn_bit_mask)
        del conn_source_pos
        self._set_buffer(
            "bitpack_source_threshold",
            source_threshold.to(device=self.pre_device, dtype=self.pre_dtype),
        )
        self._set_buffer(
            "bitpack_source_has_spiked",
            torch.zeros(
                n_source if mode == "threshold" else 0,
                device=self.pre_device,
                dtype=torch.bool,
            ),
        )
        self._set_buffer(
            "spike_history_packed",
            torch.zeros(
                (self.max_delay_steps, n_words), device=self.device, dtype=torch.int64
            ),
        )
        self._bitpack_can_use = True
        self._bitpack_mode = mode
        self._bitpack_ineligible_reason = None

    def _use_bitpacked_history_runtime(self) -> bool:
        """Return True when the initialized inference runtime should use bitpacked history."""
        return (
            (not self.training)
            and self.delay_backend == "bitpacked_history"
            and bool(getattr(self, "_bitpack_can_use", False))
        )

    def _source_history_train_mode_from_flags(self) -> str:
        """Return ``"float"`` when surrogate source gates are required.

        ``set_diff_config`` is the authoritative source of training semantics.
        If it has not been called yet, we avoid pre-allocating the larger float
        history under ``train_delay_backend="auto"`` and keep initialization on
        the dense path until the user supplies explicit differentiability flags.
        """
        if self.train_flags is None:
            return "unconfigured"
        return "float" if bool(self.train_flags[2]) else "packed"

    def _source_history_training_requested(self) -> bool:
        return (
            bool(self.training)
            and self.train_flags is not None
            and self.train_delay_backend in ("source_history", "auto")
        )

    def _validate_strict_source_history_policy(self):
        """Validate a requested strict compact training policy without switching mode."""
        if self.train_flags is None or self.train_delay_backend != "source_history":
            return
        if self._has_scheduled_events():
            raise RuntimeError(
                "NetCon train_delay_backend='source_history' currently supports "
                "intrinsic source-level events only. Use train_delay_backend='dense' "
                "or 'auto' for scheduled-event training."
            )
        self._ensure_source_level_training_metadata()
        if not bool(getattr(self, "_bitpack_can_use", False)):
            raise ValueError(
                "NetCon train_delay_backend='source_history' is not exact for this "
                f"connection: {self._bitpack_ineligible_reason}. Use "
                "train_delay_backend='dense' or source-level thresholds/pre_var."
            )

    def _validate_train_mode_transition(self, mode: bool):
        """Preflight mode changes that could otherwise fail after partial propagation."""
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        if mode and not self._mode_requires_initialize:
            self._validate_strict_source_history_policy()

    def _ensure_source_level_training_metadata(self):
        """Build source-level metadata lazily for source-history training.

        ``netcon_train_backend="auto"`` is intended to be cheap for users who
        never train a given network.  Therefore constructor-time metadata is
        built eagerly only for the bitpacked inference backend; training-only
        source-history metadata is built here, once the module is actually in
        training mode and ``set_diff_config`` has supplied semantics.
        """
        if bool(getattr(self, "_bitpack_can_use", False)):
            return
        self._configure_bitpacked_history_metadata(for_training=True)

    def _use_source_history_training_runtime(self) -> bool:
        """Return True when training should use source-level history instead of dense buffers.

        ``source_history`` is strict: if the source-level specialization cannot
        preserve exact semantics, it raises. ``auto`` is opportunistic: it uses
        source history only for exact intrinsic source-level traffic and falls
        back to the fully general dense differentiable backend for scheduled
        events, per-connection thresholds, mixed finite/NaN thresholds,
        cross-device projections, or event-introspection configurations.
        """
        if not self._source_history_training_requested():
            return False

        self._validate_strict_source_history_policy()

        self._ensure_source_level_training_metadata()

        if self._has_scheduled_events():
            return False

        if bool(getattr(self, "_bitpack_can_use", False)):
            return True

        return False

    def _source_history_diff_spiking_enabled(self) -> bool:
        return self._source_history_train_mode_from_flags() == "float"

    def _refresh_training_advance_after_diff_config(
        self, *, clear_histories: bool = False
    ):
        """Re-select training storage after ``set_diff_config`` changes.

        This lets users call ``set_diff_config`` before or after ``train()`` and,
        for interactive workflows, even after a preliminary ``initialize``.  The
        next call to ``advance`` then sees storage that matches the requested
        ``diff_spiking`` mode: packed source spikes for hard events or floating
        source gates for surrogate spiking.
        """
        if not bool(getattr(self, "training", False)):
            return
        if self._use_source_history_training_runtime():
            self._ensure_source_history_training_storage(clear=clear_histories)
            self._shrink_connection_spike_buffers_for_source_history_training()
            self.advance = self.advance_diff_source_history
        else:
            self._ensure_connection_spike_buffers()
            self._ensure_delivery_storage_for_current_mode(clear=clear_histories)
            self.advance = self.advance_diff

    def _ensure_connection_spike_buffers(self):
        """Ensure dense/training paths have only the spike state they require."""
        # An all-NaN threshold marks ``pre_var`` as an already-computed event or
        # gate signal. ``determine_spiking_var`` mirrors that signal directly,
        # so allocating one threshold-history boolean per connection is pure
        # memory overhead and can misleadingly suggest that another crossing
        # detector is active.
        if self.skip_thresholding:
            self.has_spiked = torch.empty(0, device=self.pre_device, dtype=torch.bool)
        elif self.has_spiked.numel() != self._n_conn:
            self.has_spiked = torch.zeros(
                self._n_conn, device=self.pre_device, dtype=torch.bool
            )
        else:
            self.has_spiked = self.has_spiked.to(
                device=self.pre_device, dtype=torch.bool
            )
        if self.is_spiking.numel() != self._n_conn:
            self.is_spiking = torch.zeros(
                self._n_conn, device=self.pre_device, dtype=self.pre_dtype
            )
        else:
            self.is_spiking = self.is_spiking.to(
                device=self.pre_device, dtype=self.pre_dtype
            )
        if self.con_range.numel() != self._n_conn:
            self.con_range = torch.arange(
                self._n_conn, device=self.device, dtype=torch.long
            )
        if self.events.numel() != self._n_conn:
            self.events = torch.zeros(
                self._n_conn, device=self.device, dtype=torch.int32
            )

    def _shrink_connection_spike_buffers_for_bitpack(self):
        """Drop per-connection debug/spike-state buffers in bitpacked inference mode."""
        if not self._use_bitpacked_history_runtime():
            return
        self.has_spiked = torch.empty(0, device=self.pre_device, dtype=torch.bool)
        self.is_spiking = torch.empty(0, device=self.pre_device, dtype=self.pre_dtype)
        self.con_range = torch.empty(0, device=self.device, dtype=torch.long)
        if not self.track_events:
            self.events = torch.empty(0, device=self.device, dtype=torch.int32)

    def _refresh_peer_devices(
        self, *, device=None, dtype=None, pre_device=None, pre_dtype=None
    ):
        """
        Inspect the pre/post/synapse modules and record their devices and dtypes.

        Constructor-supplied ``device``/``dtype`` are used only to avoid
        constructing large NetCon buffers on an intermediate device. Later
        calls from ``to(...)`` omit these overrides and follow the peer modules.
        """
        actual_pre_device = self.pre.device()
        actual_pre_dtype = self.pre.dtype()
        post_device = self.post.device()
        post_dtype = self.post.dtype()
        syn_device = self.syn.device() if hasattr(self.syn, "device") else post_device

        target_pre_device = (
            torch.device(pre_device) if pre_device is not None else actual_pre_device
        )
        target_pre_dtype = pre_dtype if pre_dtype is not None else actual_pre_dtype
        target_device = torch.device(device) if device is not None else syn_device
        target_dtype = dtype if dtype is not None else post_dtype

        pre_changed = (
            getattr(self, "pre_device", None) != target_pre_device
            or getattr(self, "pre_dtype", None) != target_pre_dtype
        )
        post_changed = (
            getattr(self, "device", None) != target_device
            or getattr(self, "dtype", None) != target_dtype
        )

        self.pre_device = target_pre_device
        self.pre_dtype = target_pre_dtype
        self.post_device = post_device
        self.device = target_device
        self.dtype = target_dtype

        return pre_changed, post_changed

    def _align_buffer_devices(self):
        """
        Move internal buffers to the appropriate devices/dtypes.

        This is called at construction time, at the end of ``__init__``, and
        after any external ``.to(...)`` call via :meth:`to`. The method keeps:

        * all buffers that interact with pre-synaptic state on ``pre_device``,
        * all delivery-path buffers (delay lines, queues, scheduling metadata)
          on ``self.device`` (the synapse/post device),
        * parameter modules (weights and delays) and ``dt`` on ``self.device``.

        It also ensures that auxiliary CSR structures used to expand pre indices
        to connection indices are kept on the pre-synaptic device.
        """
        # Keep buffers that interact with pre-synaptic state on the pre device/dtype.
        self._move_buffer("pre_idx", self.pre_device)
        self._move_buffer("threshold", self.pre_device, self.pre_dtype)
        self._move_buffer("thresh_is_nan", self.pre_device)
        self._move_buffer("has_spiked", self.pre_device)
        self._move_buffer("is_spiking", self.pre_device, self.pre_dtype)
        self._move_buffer("pre_range", self.pre_device)
        self._move_buffer("bitpack_source_pre_idx", self.pre_device)
        self._move_buffer("bitpack_source_threshold", self.pre_device, self.pre_dtype)
        self._move_buffer("bitpack_source_has_spiked", self.pre_device)

        # Buffers that feed the post-synaptic delivery path live with the synapse.
        for name in (
            "post_idx",
            "syn_numel",
            "n",
            "delay_steps",
            "inference_delay_steps",
            "flat_delay_offsets",
            "flat_event_offsets",
            "bitpack_conn_source_pos",
            "bitpack_conn_word_idx",
            "bitpack_conn_bit_mask",
            "source_history_conn_source_pos",
            "bitpack_source_word_idx",
            "bitpack_source_bit_mask",
            "spike_history_packed",
            "source_gate_history",
            "event_queue",
            "events",
            "current_time_step",
            "time_indices",
            "sched_pre_idx",
            "sched_abs_step",
            "sched_con_idx",
            "sched_weight_idx",
            "sched_time_idx",
            "sched_counts",
            "global_step",
            "con_range",
            "state_cache_gate_history",
        ):
            self._move_buffer(name, self.device)

        self._move_buffer("delivery_buffer", self.device, self.dtype)
        self._move_buffer("sched_wsum", self.device, self.dtype)
        self.sched_weight = self.sched_weight.to(device=self.device, dtype=self.dtype)
        self.sched_time_ms = self.sched_time_ms.to(device=self.device, dtype=self.dtype)

        # Keep helper caches with the pre buffers.
        if hasattr(self, "_csr_pre_ids"):
            self._csr_pre_ids = self._csr_pre_ids.to(device=self.pre_device)
            self._csr_starts = self._csr_starts.to(device=self.pre_device)
            self._csr_counts = self._csr_counts.to(device=self.pre_device)
            self._csr_conidx_sorted = self._csr_conidx_sorted.to(device=self.pre_device)

        # dt and parameter modules should follow the synapse device.
        self.dt = self.dt.to(device=self.device, dtype=self.dtype)
        self.weight = self.weight.to(device=self.device, dtype=self.dtype)
        self.delay_ms = self.delay_ms.to(device=self.device, dtype=self.dtype)
        self._move_sparse_calendar_to_device()

    def to(self, *args, **kwargs):  # type: ignore[override]
        """
        Move this ``NetCon`` and its internal state to a new device/dtype.

        This is a thin wrapper around ``super().to(...)`` which also:

        * refreshes peer devices/dtypes via :meth:`_refresh_peer_devices`, and
        * realigns internal buffers via :meth:`_align_buffer_devices`.

        All scheduling metadata and delay buffers are preserved, but they are
        moved to the target device/dtype as appropriate.

        Returns
        -------
        NetCon
            The same instance, for chaining.
        """
        super().to(*args, **kwargs)
        self._refresh_peer_devices()
        self._align_buffer_devices()
        return self

    def _compute_max_delay_steps(self):
        """
        Compute the depth (in steps) of the internal delay line.

        If ``max_delay`` is provided at construction, this method simply
        computes ``int(max_delay / dt) + 1``. Otherwise it uses the maximum
        integer delay in ``self.delay_steps`` if present, defaulting to 1
        when no delays exist.

        Returns
        -------
        int
            Maximum number of steps in the delay buffer.
        """
        if self.max_delay is not None:
            return int(self.max_delay / self.dt.item()) + 1
        else:
            return (
                int(self.delay_steps.max().item()) + 1
                if len(self.delay_steps) > 0
                else 1
            )

    def _rebuild_dense_delay_metadata(self):
        """Precompute integer-delay routing metadata for inference.

        NetCon delays are positive-valued by construction, but very small
        positive delays can round to zero integer steps.  In non-differentiable
        inference, the earliest valid delivery is the next simulation step, so
        ``inference_delay_steps`` is clamped to at least one step.

        Dense inference additionally needs connection-sized flat routing offsets.
        Sparse-calendar and bitpacked-history inference do not: they only need
        ``inference_delay_steps``.  Keeping those modes free of ``flat_*`` buffers
        is important for large SNN reinitialization, because a single unnecessary
        ``flat_delay_offsets`` tensor can be hundreds of MiB and can trigger OOM
        even when the steady-state model fits.
        """
        syn_numel = int(self._syn_numel)
        n_conn = int(self._n_conn)

        needs_dense_flat_offsets = self.delay_backend == "dense"
        needs_event_flat_offsets = (
            bool(getattr(self, "track_events", False)) and needs_dense_flat_offsets
        )

        # If we are in a non-dense backend, eagerly drop stale dense-only buffers
        # before constructing/reusing any connection-sized metadata.  This matters
        # when reinitializing a model that was previously built with an older
        # implementation that retained ``flat_delay_offsets`` in bitpacked mode.
        if not needs_dense_flat_offsets:
            empty_long = self._empty_buffer(dtype=torch.long)
            if (
                "flat_delay_offsets" in self._buffers
                and self.flat_delay_offsets.numel() != 0
            ):
                self._set_buffer("flat_delay_offsets", empty_long)
            if (
                "flat_event_offsets" in self._buffers
                and self.flat_event_offsets.numel() != 0
            ):
                self._set_buffer("flat_event_offsets", empty_long)

        # Reuse the persistent inference-delay buffer during delay rebuilds rather
        # than allocating another n_conn LongTensor.
        delay_steps = self.delay_steps
        if delay_steps.device != self.device or delay_steps.dtype != torch.long:
            delay_steps = delay_steps.to(device=self.device, dtype=torch.long)

        if (
            "inference_delay_steps" in self._buffers
            and tuple(self.inference_delay_steps.shape) == tuple(delay_steps.shape)
            and self.inference_delay_steps.device == self.device
            and self.inference_delay_steps.dtype == torch.long
        ):
            inference_delay_steps = self.inference_delay_steps
            inference_delay_steps.copy_(delay_steps)
            if inference_delay_steps.numel() > 0:
                inference_delay_steps.clamp_min_(1)
        else:
            inference_delay_steps = delay_steps.clone()
            if inference_delay_steps.numel() > 0:
                inference_delay_steps.clamp_min_(1)
            self._set_buffer("inference_delay_steps", inference_delay_steps)

        self._delivery_numel = int(max(1, self.max_delay_steps) * syn_numel)
        self._event_queue_numel = int(max(1, self.max_delay_steps) * max(n_conn, 1))

        if inference_delay_steps.numel() > 0:
            # Avoid ``torch.all(inference_delay_steps == first)``, which creates a
            # connection-sized temporary bool tensor.  min/max reductions give the
            # same uniform-delay test with much lower peak memory.
            d_min = int(inference_delay_steps.min().item())
            d_max = int(inference_delay_steps.max().item())
            self._dense_delay_uniform = d_min == d_max
            self._dense_uniform_delay_step = d_min if self._dense_delay_uniform else 1
        else:
            self._dense_delay_uniform = True
            self._dense_uniform_delay_step = 1

        if needs_dense_flat_offsets:
            post_idx = self.post_idx.to(self.device, dtype=torch.long)
            if (
                "flat_delay_offsets" in self._buffers
                and tuple(self.flat_delay_offsets.shape)
                == tuple(inference_delay_steps.shape)
                and self.flat_delay_offsets.device == self.device
                and self.flat_delay_offsets.dtype == torch.long
            ):
                flat_delay_offsets = self.flat_delay_offsets
                flat_delay_offsets.copy_(inference_delay_steps)
            else:
                flat_delay_offsets = inference_delay_steps.clone()
            flat_delay_offsets.mul_(syn_numel).add_(post_idx)
            self._set_buffer("flat_delay_offsets", flat_delay_offsets)
        else:
            self._set_buffer("flat_delay_offsets", self._empty_buffer(dtype=torch.long))

        if needs_event_flat_offsets:
            if self.con_range.numel() == n_conn:
                con_range = self.con_range.to(self.device, dtype=torch.long)
            else:
                con_range = torch.arange(n_conn, device=self.device, dtype=torch.long)
            if (
                "flat_event_offsets" in self._buffers
                and tuple(self.flat_event_offsets.shape)
                == tuple(inference_delay_steps.shape)
                and self.flat_event_offsets.device == self.device
                and self.flat_event_offsets.dtype == torch.long
            ):
                flat_event_offsets = self.flat_event_offsets
                flat_event_offsets.copy_(inference_delay_steps)
            else:
                flat_event_offsets = inference_delay_steps.clone()
            flat_event_offsets.mul_(max(n_conn, 1)).add_(con_range)
            self._set_buffer("flat_event_offsets", flat_event_offsets)
        else:
            self._set_buffer("flat_event_offsets", self._empty_buffer(dtype=torch.long))

    def _has_scheduled_events(self) -> bool:
        """Cheap shape-only check used to skip scheduled-event aggregation."""
        return self.sched_con_idx.numel() > 0

    def _dense_advance_target(self):
        """Return the specialized dense inference advance method for current delays."""
        if getattr(self, "_dense_delay_uniform", False):
            return self.advance_non_diff_dense_uniform
        return self.advance_non_diff_dense_mixed

    def _use_sparse_calendar_runtime(self) -> bool:
        """Return True when the current initialized mode should use sparse inference."""
        return (not self.training) and self.delay_backend == "sparse_calendar"

    def _desired_delivery_buffer_depth(self) -> int:
        """Depth of ``delivery_buffer`` for the current mode.

        Dense training/inference needs one row per delay slot. Sparse-calendar
        and bitpacked-history inference keep only one dense row, used as a
        scratch receive payload.
        """
        return (
            1
            if (
                self._use_sparse_calendar_runtime()
                or self._use_bitpacked_history_runtime()
                or self._use_source_history_training_runtime()
            )
            else self.max_delay_steps
        )

    def _ensure_delivery_storage_for_current_mode(self, *, clear: bool = False):
        """Allocate dense delay storage or sparse scratch storage as needed."""
        desired_depth = self._desired_delivery_buffer_depth()
        desired_shape = (desired_depth, self._syn_numel)

        if tuple(self.delivery_buffer.shape) != desired_shape:
            self._set_buffer(
                "delivery_buffer",
                torch.empty((0, self._syn_numel), device=self.device, dtype=self.dtype),
            )
            self._set_buffer(
                "delivery_buffer",
                torch.zeros(desired_shape, device=self.device, dtype=self.dtype),
            )
        elif clear:
            self.delivery_buffer.zero_()

        if (
            self._use_sparse_calendar_runtime()
            or self._use_bitpacked_history_runtime()
            or self._use_source_history_training_runtime()
        ):
            if clear:
                self._clear_sparse_calendar()
                if (
                    self._use_bitpacked_history_runtime()
                    or self._use_source_history_training_runtime()
                ):
                    if (
                        hasattr(self, "spike_history_packed")
                        and self.spike_history_packed.numel() > 0
                    ):
                        self.spike_history_packed.zero_()
                    if (
                        hasattr(self, "source_gate_history")
                        and self.source_gate_history.numel() > 0
                    ):
                        self.source_gate_history = self.source_gate_history.detach()
                        self.source_gate_history.zero_()
                    if self.bitpack_source_has_spiked.numel() > 0:
                        self.bitpack_source_has_spiked.zero_()
            # Sparse/source-history modes do not allocate the dense historical event queue.
            if "event_queue" in self._buffers:
                del self._buffers["event_queue"]
        elif self.track_events and "event_queue" not in self._buffers:
            self.register_buffer(
                "event_queue",
                torch.zeros(
                    (self.max_delay_steps, self.n.item()),
                    device=self.device,
                    dtype=torch.int32,
                ),
            )
        elif not self.track_events and "event_queue" in self._buffers:
            del self._buffers["event_queue"]

    def enable_state_cache_recording(self, *, horizon_steps: int | None = None):
        """Record unweighted connection gates for parameter-invariant cache.

        Bitpacked-history inference already maintains a compact source-spike
        history, so it does not need an additional connection-sized recorder.
        Dense and sparse-calendar backends use this recorder during
        ``Network.steady_state`` so future pending deliveries can later be
        reconstructed with updated weights/delays and/or a new dt.
        """
        self._state_cache_recording_enabled = True
        if self._use_bitpacked_history_runtime():
            return self
        steps = (
            int(horizon_steps)
            if horizon_steps is not None
            else int(self.max_delay_steps)
        )
        steps = max(1, steps)
        shape = (steps, int(self._n_conn))
        if (
            "state_cache_gate_history" not in self._buffers
            or tuple(self.state_cache_gate_history.shape) != shape
            or self.state_cache_gate_history.device != self.device
            or self.state_cache_gate_history.dtype != self.dtype
        ):
            self._set_buffer(
                "state_cache_gate_history",
                torch.zeros(shape, device=self.device, dtype=self.dtype),
            )
        else:
            self.state_cache_gate_history.zero_()
        return self

    def disable_state_cache_recording(self, *, release: bool = False):
        self._state_cache_recording_enabled = False
        if release and "state_cache_gate_history" in self._buffers:
            self._set_buffer(
                "state_cache_gate_history",
                torch.empty((0, 0), device=self.device, dtype=self.dtype),
            )
        return self

    def _record_gate_for_state_cache(self, gate: torch.Tensor):
        if not getattr(self, "_state_cache_recording_enabled", False):
            return
        if self._use_bitpacked_history_runtime():
            return
        if (
            "state_cache_gate_history" not in self._buffers
            or self.state_cache_gate_history.numel() == 0
        ):
            return
        slot = int(self.current_time_step.detach().cpu().reshape(-1)[0].item())
        slot %= int(self.state_cache_gate_history.shape[0])
        self.state_cache_gate_history[slot].copy_(
            gate.detach().to(device=self.device, dtype=self.dtype)
        )

    def _materialize_delivery_from_gate_history(self, age_history: torch.Tensor):
        """Rebuild pending deliveries from unweighted per-connection gates."""
        age_history = age_history.detach().to(device=self.device, dtype=self.dtype)
        if age_history.ndim != 2 or age_history.shape[1] != self._n_conn:
            raise ValueError(
                "Cached NetCon gate history has incompatible shape: "
                f"{tuple(age_history.shape)} vs (*, {self._n_conn})."
            )
        if self._use_source_history_training_runtime():
            self._restore_source_history_training_from_gate_history(age_history)
            return

        self.delivery_buffer.zero_()
        self._clear_sparse_calendar()
        if self._n_conn == 0 or age_history.shape[0] <= 1:
            return

        weight = self.weight().detach().to(device=self.device, dtype=self.dtype)
        delay_steps = self.inference_delay_steps.to(
            device=self.device, dtype=torch.long
        )
        post_idx = self.post_idx.to(device=self.device, dtype=torch.long)
        max_age = min(int(age_history.shape[0]), int(self.max_delay_steps))

        if self._use_sparse_calendar_runtime():
            for age in range(1, max_age):
                remaining = delay_steps - int(age)
                gate = age_history[age]
                keep = (remaining >= 0) & (gate != 0)
                if not bool(keep.any()):
                    continue
                con_idx = torch.nonzero(keep, as_tuple=False).flatten()
                values = weight.index_select(0, con_idx) * gate.index_select(0, con_idx)
                future_slots = remaining.index_select(0, con_idx).remainder(
                    self.max_delay_steps
                )
                self._append_sparse_payloads(
                    future_slots,
                    post_idx.index_select(0, con_idx),
                    values,
                )
            return

        # Dense runtime, including training with delay_backend='bitpacked_history'.
        flat = self.delivery_buffer.view(-1)
        for age in range(1, max_age):
            remaining = delay_steps - int(age)
            gate = age_history[age]
            keep = (remaining >= 0) & (gate != 0)
            if not bool(keep.any()):
                continue
            con_idx = torch.nonzero(keep, as_tuple=False).flatten()
            values = weight.index_select(0, con_idx) * gate.index_select(0, con_idx)
            flat_idx = remaining.index_select(
                0, con_idx
            ) * self._syn_numel + post_idx.index_select(0, con_idx)
            flat.index_add_(0, flat_idx.reshape(-1), values.reshape(-1))

    def _materialize_delivery_from_packed_source_history(
        self, age_history: torch.Tensor
    ):
        """Rebuild dense/sparse pending deliveries from bitpacked source history.

        This is used when a bitpacked steady-state cache is restored into a
        dense runtime, for example when the network is switched to training mode.
        Source-history training uses one-row receive scratch storage rather than
        a future delivery ring, so that runtime must restore the compact source
        history itself instead of materializing future dense deliveries.
        """
        age_history = age_history.detach().to(device=self.device, dtype=torch.int64)
        expected_words = (
            int(self.spike_history_packed.shape[1])
            if hasattr(self, "spike_history_packed")
            and self.spike_history_packed.numel() > 0
            else max(
                1,
                (
                    int(
                        getattr(
                            self, "bitpack_source_pre_idx", self._empty_long()
                        ).numel()
                    )
                    + int(self._bitpack_bits_per_word)
                    - 1
                )
                // int(self._bitpack_bits_per_word),
            )
        )
        if age_history.ndim != 2 or age_history.shape[1] != expected_words:
            raise ValueError(
                "Cached bitpacked source history has incompatible word count: "
                f"{tuple(age_history.shape)} vs (*, {expected_words})."
            )

        if self._use_source_history_training_runtime():
            self._restore_source_history_training_from_packed_age_history(age_history)
            return

        self.delivery_buffer.zero_()
        self._clear_sparse_calendar()
        if self._n_conn == 0 or age_history.shape[0] <= 1:
            return

        weight = self.weight().detach().to(device=self.device, dtype=self.dtype)
        delay_steps = self.inference_delay_steps.to(
            device=self.device, dtype=torch.long
        )
        post_idx = self.post_idx.to(device=self.device, dtype=torch.long)
        word_idx = self.bitpack_conn_word_idx.to(device=self.device, dtype=torch.long)
        bit_mask = self.bitpack_conn_bit_mask.to(device=self.device, dtype=torch.int64)
        max_age = min(int(age_history.shape[0]), int(self.max_delay_steps))

        if self._use_sparse_calendar_runtime():
            for age in range(1, max_age):
                words = age_history[age].index_select(0, word_idx)
                active = torch.bitwise_and(words, bit_mask) != 0
                remaining = delay_steps - int(age)
                keep = active & (remaining >= 0)
                if not bool(keep.any()):
                    continue
                con_idx = torch.nonzero(keep, as_tuple=False).flatten()
                values = weight.index_select(0, con_idx)
                future_slots = remaining.index_select(0, con_idx).remainder(
                    self.max_delay_steps
                )
                self._append_sparse_payloads(
                    future_slots,
                    post_idx.index_select(0, con_idx),
                    values,
                )
            return

        flat = self.delivery_buffer.view(-1)
        for age in range(1, max_age):
            words = age_history[age].index_select(0, word_idx)
            active = torch.bitwise_and(words, bit_mask) != 0
            remaining = delay_steps - int(age)
            keep = active & (remaining >= 0)
            if not bool(keep.any()):
                continue
            con_idx = torch.nonzero(keep, as_tuple=False).flatten()
            values = weight.index_select(0, con_idx)
            flat_idx = remaining.index_select(
                0, con_idx
            ) * self._syn_numel + post_idx.index_select(0, con_idx)
            flat.index_add_(0, flat_idx.reshape(-1), values.reshape(-1))

    def _first_connection_per_source(self) -> torch.Tensor:
        """Return representative connection indices for each unique source.

        Source-history modes are exact only when every outgoing edge from a
        unique presynaptic source shares the same gate.  During cache restore we
        may receive a connection-level gate history from a dense steady-state
        run; this helper lets us collapse it back to source-level history by
        selecting one representative connection per source.
        """
        n_source = int(
            getattr(self, "bitpack_source_pre_idx", self._empty_long()).numel()
        )
        if n_source == 0 or self._n_conn == 0:
            return torch.empty(0, device=self.device, dtype=torch.long)
        source_pos = self._ensure_source_history_conn_source_pos().to(
            device=self.device, dtype=torch.long
        )
        positions = torch.arange(
            source_pos.numel(), device=self.device, dtype=torch.long
        )
        first = torch.full(
            (n_source,),
            source_pos.numel(),
            device=self.device,
            dtype=torch.long,
        )
        first.scatter_reduce_(
            0, source_pos, positions, reduce="amin", include_self=True
        )
        return first

    def _source_age_history_from_connection_gate_history(
        self, age_history: torch.Tensor
    ) -> torch.Tensor:
        """Collapse connection-level age history to source-level age history."""
        age_history = age_history.detach().to(device=self.device, dtype=self.dtype)
        n_source = int(
            getattr(self, "bitpack_source_pre_idx", self._empty_long()).numel()
        )
        out = torch.zeros(
            (int(age_history.shape[0]), n_source),
            device=self.device,
            dtype=self.dtype,
        )
        if n_source == 0 or age_history.numel() == 0:
            return out
        first = self._first_connection_per_source()
        valid = first < int(self._n_conn)
        if bool(valid.any()):
            src = torch.nonzero(valid, as_tuple=False).flatten()
            con = first.index_select(0, src)
            out.index_copy_(1, src, age_history.index_select(1, con))
        return out

    def _unpack_source_age_history(self, age_history: torch.Tensor) -> torch.Tensor:
        """Unpack bitpacked age rows to floating source-gate age rows."""
        age_history = age_history.detach().to(device=self.device, dtype=torch.int64)
        n_source = int(
            getattr(self, "bitpack_source_pre_idx", self._empty_long()).numel()
        )
        out = torch.zeros(
            (int(age_history.shape[0]), n_source),
            device=self.device,
            dtype=self.dtype,
        )
        if n_source == 0 or age_history.numel() == 0:
            return out
        word_idx = self.bitpack_source_word_idx.to(device=self.device, dtype=torch.long)
        bit_mask = self.bitpack_source_bit_mask.to(
            device=self.device, dtype=torch.int64
        )
        words = age_history[:, word_idx]
        active = torch.bitwise_and(words, bit_mask.view(1, -1)) != 0
        out.copy_(active.to(dtype=self.dtype))
        return out

    def _pack_source_age_history_to_ring(
        self, source_age_history: torch.Tensor
    ) -> torch.Tensor:
        """Pack source-level age rows into a current_slot==0 history ring."""
        self._resize_bitpacked_spike_history(clear=True)
        depth = int(self.spike_history_packed.shape[0])
        n_words = int(self.spike_history_packed.shape[1])
        ring = torch.zeros((depth, n_words), device=self.device, dtype=torch.int64)
        if depth <= 0 or source_age_history.numel() == 0:
            return ring
        n = min(int(source_age_history.shape[0]), depth)
        word_idx = self.bitpack_source_word_idx.to(device=self.device, dtype=torch.long)
        bit_mask = self.bitpack_source_bit_mask.to(
            device=self.device, dtype=torch.int64
        )
        for age in range(n):
            active = source_age_history[age].to(device=self.device) != 0
            if not bool(active.any()):
                continue
            row_idx = (-int(age)) % depth
            vals = bit_mask * active.to(torch.int64)
            ring[row_idx].index_add_(0, word_idx, vals)
        return ring

    def _restore_source_history_training_from_source_age_history(
        self, source_age_history: torch.Tensor
    ):
        """Restore compact source-history training state from age rows."""
        self.delivery_buffer.zero_()
        self._clear_sparse_calendar()
        if self._source_history_diff_spiking_enabled():
            self._resize_source_gate_history(clear=True)
            source_age_history = source_age_history.detach().to(
                device=self.device, dtype=self.dtype
            )
            if (
                source_age_history.ndim != 2
                or source_age_history.shape[1] != self.source_gate_history.shape[1]
            ):
                raise ValueError(
                    "Cached source gate history has incompatible shape: "
                    f"{tuple(source_age_history.shape)} vs (*, {self.source_gate_history.shape[1]})."
                )
            ring = _age_rows_to_ring_rows(
                source_age_history, depth=int(self.source_gate_history.shape[0])
            )
            self.source_gate_history.copy_(ring)
        else:
            ring = self._pack_source_age_history_to_ring(source_age_history)
            self.spike_history_packed.copy_(ring)

    def _restore_source_history_training_from_gate_history(
        self, age_history: torch.Tensor
    ):
        """Restore source-history training state from connection gate history."""
        source_age = self._source_age_history_from_connection_gate_history(age_history)
        self._restore_source_history_training_from_source_age_history(source_age)

    def _restore_source_history_training_from_packed_age_history(
        self, age_history: torch.Tensor
    ):
        """Restore source-history training state from packed source spike history."""
        if self._source_history_diff_spiking_enabled():
            source_age = self._unpack_source_age_history(age_history)
            self._restore_source_history_training_from_source_age_history(source_age)
        else:
            self.delivery_buffer.zero_()
            self._clear_sparse_calendar()
            self._resize_bitpacked_spike_history(clear=True)
            ring = _age_rows_to_ring_rows(
                age_history.detach().to(device=self.device, dtype=torch.int64),
                depth=int(self.spike_history_packed.shape[0]),
            )
            self.spike_history_packed.copy_(ring)

    def _restore_source_has_spiked_from_connection_cache(
        self, has_spiked: torch.Tensor
    ):
        """Restore source-level threshold state from a connection-level cache."""
        if (
            not self._use_source_history_training_runtime()
            or not hasattr(self, "bitpack_source_has_spiked")
            or self.bitpack_source_has_spiked.numel() == 0
            or has_spiked.numel() != self._n_conn
        ):
            return
        first = self._first_connection_per_source()
        valid = first < int(self._n_conn)
        src_state = torch.zeros_like(self.bitpack_source_has_spiked)
        if bool(valid.any()):
            src = torch.nonzero(valid, as_tuple=False).flatten()
            con = first.index_select(0, src)
            src_state.index_copy_(
                0,
                src.to(device=self.pre_device),
                has_spiked.to(device=self.pre_device, dtype=torch.bool).index_select(
                    0, con.to(self.pre_device)
                ),
            )
        self.bitpack_source_has_spiked.copy_(src_state)

    def _clear_sparse_calendar(self):
        """Drop all pending sparse-calendar deliveries and event counts."""
        self._sparse_calendar.clear()
        self._sparse_event_calendar.clear()

    def _move_sparse_calendar_to_device(self):
        """Move Python-side calendar chunks after ``.to(...)`` or device realignment."""
        if not hasattr(self, "_sparse_calendar"):
            return

        def move_payload(chunks):
            return [
                (
                    idx.to(device=self.device, dtype=torch.long),
                    val.to(device=self.device, dtype=self.dtype),
                )
                for idx, val in chunks
            ]

        def move_events(chunks):
            return [
                (
                    idx.to(device=self.device, dtype=torch.long),
                    cnt.to(device=self.device, dtype=torch.int32),
                )
                for idx, cnt in chunks
            ]

        self._sparse_calendar = {
            int(slot): move_payload(chunks)
            for slot, chunks in self._sparse_calendar.items()
            if len(chunks) > 0
        }
        self._sparse_event_calendar = {
            int(slot): move_events(chunks)
            for slot, chunks in self._sparse_event_calendar.items()
            if len(chunks) > 0
        }

    def _compact_sparse_payload_slot(self, slot: int):
        chunks = self._sparse_calendar.get(slot)
        if not chunks or len(chunks) <= 1:
            return
        self._sparse_calendar[slot] = [
            (
                torch.cat([idx for idx, _ in chunks], dim=0),
                torch.cat([val for _, val in chunks], dim=0),
            )
        ]

    def _compact_sparse_event_slot(self, slot: int):
        chunks = self._sparse_event_calendar.get(slot)
        if not chunks or len(chunks) <= 1:
            return
        self._sparse_event_calendar[slot] = [
            (
                torch.cat([idx for idx, _ in chunks], dim=0),
                torch.cat([cnt for _, cnt in chunks], dim=0),
            )
        ]

    def _append_sparse_payloads(
        self,
        future_slots: torch.Tensor,
        post_idx: torch.Tensor,
        values: torch.Tensor,
    ):
        """Append nonzero dense-payload contributions to sparse calendar buckets."""
        if values.numel() == 0:
            return

        for slot in torch.unique(future_slots.detach()).detach().cpu().tolist():
            slot_i = int(slot)
            mask = future_slots == slot_i
            self._sparse_calendar.setdefault(slot_i, []).append(
                (
                    post_idx[mask].detach(),
                    values[mask].detach(),
                )
            )
            if len(self._sparse_calendar[slot_i]) > self._calendar_compact_threshold:
                self._compact_sparse_payload_slot(slot_i)

    def _append_sparse_events(
        self,
        future_slots: torch.Tensor,
        con_idx: torch.Tensor,
        counts: torch.Tensor,
    ):
        """Append per-connection event counts for optional sparse introspection."""
        if counts.numel() == 0:
            return

        for slot in torch.unique(future_slots.detach()).detach().cpu().tolist():
            slot_i = int(slot)
            mask = future_slots == slot_i
            self._sparse_event_calendar.setdefault(slot_i, []).append(
                (
                    con_idx[mask].detach(),
                    counts[mask].detach(),
                )
            )
            if (
                len(self._sparse_event_calendar[slot_i])
                > self._calendar_compact_threshold
            ):
                self._compact_sparse_event_slot(slot_i)

    def _pop_sparse_calendar_to_dense(self, slot: int) -> torch.Tensor:
        """Reduce due sparse payload chunks into the reusable dense scratch row."""
        scratch = self.delivery_buffer.squeeze(0)
        scratch.zero_()
        chunks = self._sparse_calendar.pop(slot, None)
        if chunks:
            for post_idx, values in chunks:
                if values.numel() > 0:
                    scratch.index_add_(0, post_idx, values)
        return scratch

    def _pop_sparse_events(self, slot: int):
        """Populate ``self.events`` from sparse event-count chunks due now."""
        self.events.zero_()
        if not self.track_events:
            return
        chunks = self._sparse_event_calendar.pop(slot, None)
        if chunks:
            for con_idx, counts in chunks:
                if counts.numel() > 0:
                    self.events.index_add_(0, con_idx, counts)

    def _add_sparse_calendar_to_dense(self, slot: int, scratch: torch.Tensor):
        """Add due sparse side-calendar payloads into an existing dense scratch row."""
        chunks = self._sparse_calendar.pop(slot, None)
        if chunks:
            for post_idx, values in chunks:
                if values.numel() > 0:
                    scratch.index_add_(0, post_idx, values)

    def _ensure_source_history_conn_source_pos(self):
        """Ensure connection→unique-source positions exist for float source history."""
        if self.source_history_conn_source_pos.numel() == self._n_conn:
            return self.source_history_conn_source_pos
        if self.bitpack_source_pre_idx.numel() == 0:
            pos = torch.empty(0, device=self.device, dtype=torch.long)
        else:
            pos = torch.searchsorted(
                self.bitpack_source_pre_idx.to(device=self.device, dtype=torch.long),
                self.pre_idx.to(device=self.device, dtype=torch.long),
            ).to(device=self.device, dtype=torch.long)
        self._set_buffer("source_history_conn_source_pos", pos)
        return pos

    def _resize_source_gate_history(self, *, clear: bool = True):
        """Resize/clear differentiable floating source-gate history."""
        if not getattr(self, "_bitpack_can_use", False):
            return
        n_source = int(self.bitpack_source_pre_idx.numel())
        desired_shape = (int(self.max_delay_steps), n_source)
        if (
            "source_gate_history" not in self._buffers
            or tuple(self.source_gate_history.shape) != desired_shape
            or self.source_gate_history.device != self.device
            or self.source_gate_history.dtype != self.dtype
        ):
            self._set_buffer(
                "source_gate_history",
                torch.zeros(desired_shape, device=self.device, dtype=self.dtype),
            )
        elif clear:
            self.source_gate_history = self.source_gate_history.detach()
            self.source_gate_history.zero_()
        self._ensure_source_history_conn_source_pos()

    def _ensure_source_history_training_storage(self, *, clear: bool = False):
        """Install the minimal storage for the selected source-history training mode."""
        if not self._use_source_history_training_runtime():
            return
        if self._has_scheduled_events():
            raise RuntimeError(
                "train_delay_backend='source_history' currently supports intrinsic "
                "source-level events only. Use train_delay_backend='dense' for "
                "scheduled-event training."
            )
        self._ensure_delivery_storage_for_current_mode(clear=clear)
        if self._source_history_diff_spiking_enabled():
            self._resize_source_gate_history(clear=clear)
            # In float mode the packed bool history is not part of the training hot path.
            if (
                "spike_history_packed" in self._buffers
                and self.spike_history_packed.numel() > 0
            ):
                self._set_buffer(
                    "spike_history_packed",
                    torch.empty((0, 0), device=self.device, dtype=torch.int64),
                )
        else:
            self._resize_bitpacked_spike_history(clear=clear)
            if (
                "source_gate_history" in self._buffers
                and self.source_gate_history.numel() > 0
            ):
                self._set_buffer(
                    "source_gate_history",
                    torch.empty((0, 0), device=self.device, dtype=self.dtype),
                )
            if (
                "source_history_conn_source_pos" in self._buffers
                and self.source_history_conn_source_pos.numel() > 0
            ):
                self._set_buffer(
                    "source_history_conn_source_pos",
                    self._empty_buffer(dtype=torch.long),
                )

    def _shrink_connection_spike_buffers_for_source_history_training(self):
        """Drop per-connection spike buffers when source-level history is exact."""
        if not self._use_source_history_training_runtime():
            return
        self.has_spiked = torch.empty(0, device=self.pre_device, dtype=torch.bool)
        self.is_spiking = torch.empty(0, device=self.pre_device, dtype=self.pre_dtype)
        if not self.track_events:
            self.events = torch.empty(0, device=self.device, dtype=torch.int32)

    def _source_history_source_gate_this_step(
        self, *, diff_spiking: bool, tau: float
    ) -> torch.Tensor:
        """Return one source-level gate value per unique presynaptic source."""
        if isinstance(self.pre, NetStim):
            attr = self.pre.spike_gate if diff_spiking else self.pre.spikes
            gate = (
                attr.to(self.pre_device)
                .reshape(-1)
                .index_select(0, self.bitpack_source_pre_idx)
            )
            return gate.to(device=self.device, dtype=self.dtype)

        x_full = self.get_pre_var(self.pre)
        if x_full.device != self.pre_device:
            x_full = x_full.to(device=self.pre_device)
        x = x_full.reshape(-1).index_select(0, self.bitpack_source_pre_idx)

        if self._bitpack_mode == "pre_var":
            if x.dtype == torch.bool:
                gate = x.to(self.pre_dtype)
            else:
                gate = x.to(self.pre_dtype)
                if not diff_spiking:
                    gate = (gate > 0).to(self.pre_dtype)
            return gate.to(device=self.device, dtype=self.dtype)

        if self._bitpack_mode == "threshold":
            x = x.to(self.pre_dtype)
            if diff_spiking:
                ge_hard, _ge_gate, spk_gate = update_active_diff(
                    self.bitpack_source_has_spiked,
                    x,
                    self.bitpack_source_threshold,
                    tau,
                )
                # ``has_spiked`` is the hard above-threshold state used for the next
                # rising-edge decision; keep it out of the differentiable graph.
                self.bitpack_source_has_spiked = ge_hard.detach()
                return spk_gate.to(device=self.device, dtype=self.dtype)
            self.bitpack_source_has_spiked, spikes = update_active(
                self.bitpack_source_has_spiked,
                x,
                self.bitpack_source_threshold,
            )
            return spikes.to(device=self.device, dtype=self.dtype)

        raise RuntimeError("source-history training backend is not configured")

    def _source_history_read_packed_connection_gates(
        self, rows: torch.Tensor
    ) -> torch.Tensor:
        """Read hard source-spike gates from packed history for connection rows."""
        if rows.ndim == 1:
            words = self.spike_history_packed[
                rows.reshape(-1), self.bitpack_conn_word_idx
            ]
            gate = torch.bitwise_and(words, self.bitpack_conn_bit_mask) != 0
            return gate.to(device=self.device, dtype=self.dtype)
        word_idx = self.bitpack_conn_word_idx.unsqueeze(-1).expand_as(rows)
        mask = self.bitpack_conn_bit_mask.unsqueeze(-1).expand_as(rows)
        words = self.spike_history_packed[
            rows.reshape(-1), word_idx.reshape(-1)
        ].view_as(rows)
        gate = torch.bitwise_and(words, mask) != 0
        return gate.to(device=self.device, dtype=self.dtype)

    def _source_history_read_float_connection_gates(
        self, rows: torch.Tensor
    ) -> torch.Tensor:
        """Read differentiable source gates for connection rows."""
        source_pos = self._ensure_source_history_conn_source_pos()
        if rows.ndim == 1:
            return self.source_gate_history[rows.reshape(-1), source_pos]
        src = source_pos.unsqueeze(-1).expand_as(rows)
        return self.source_gate_history[rows.reshape(-1), src.reshape(-1)].view_as(rows)

    def _source_history_build_delivery_from_history(
        self,
        cur_idx: torch.Tensor,
        *,
        diff_weights: bool,
        diff_delays: bool,
        taps: int,
        sigma: float,
    ) -> torch.Tensor:
        """Construct the current dense payload from compact source history."""
        out = torch.zeros(self._syn_numel, device=self.device, dtype=self.dtype)
        if self._n_conn == 0:
            return out

        wvals = self.weight()
        if not diff_weights:
            wvals = wvals.detach()

        read_gates = (
            self._source_history_read_float_connection_gates
            if self._source_history_diff_spiking_enabled()
            else self._source_history_read_packed_connection_gates
        )

        if diff_delays:
            d_ms = self.delay_ms().to(self.dtype)
            lam = (d_ms / self.dt.to(self.dtype)).clamp_min(1.0)
            k = torch.floor(lam)
            kL = k.to(torch.long)

            if taps == 2:
                alpha = (lam - k).to(self.dtype)
                row0 = (cur_idx - kL).remainder(self.max_delay_steps)
                row1 = (row0 - 1).remainder(self.max_delay_steps)
                gate0 = read_gates(row0.reshape(-1))
                gate1 = read_gates(row1.reshape(-1))
                vals = wvals * (gate0 * (1.0 - alpha) + gate1 * alpha)
                out.index_add_(0, self.post_idx, vals)
            else:
                offs = torch.stack([kL - 1, kL, kL + 1], dim=-1)
                centers = offs.to(self.dtype)
                lam_e = lam.unsqueeze(-1)
                weights = torch.softmax(
                    -0.5 * ((lam_e - centers) / (sigma + 1e-6)) ** 2,
                    dim=-1,
                )
                # Future-write offset ``o`` corresponds at delivery time to source
                # history row ``cur_idx - o``.
                rows = (cur_idx - offs).remainder(self.max_delay_steps)
                gates = read_gates(rows)
                vals = wvals * (gates * weights).sum(dim=-1)
                out.index_add_(0, self.post_idx, vals)
        else:
            delay_steps = self.inference_delay_steps
            rows = (cur_idx - delay_steps).remainder(self.max_delay_steps)
            vals = wvals * read_gates(rows.reshape(-1))
            out.index_add_(0, self.post_idx, vals)
        return out

    def _source_history_record_current_gate(
        self,
        cur_idx: torch.Tensor,
        source_gate: torch.Tensor,
        *,
        diff_spiking: bool,
    ):
        """Append the current source-level event/gate into compact history."""
        if diff_spiking:
            hist_next = self.source_gate_history.clone()
            hist_next.index_copy_(
                0,
                cur_idx.reshape(-1),
                source_gate.to(device=self.device, dtype=self.dtype).reshape(1, -1),
            )
            self.source_gate_history = hist_next
            return
        self._bitpack_pack_source_spikes(source_gate.to(torch.bool), cur_idx)

    def _record_source_gate_for_state_cache_if_needed(self, source_gate: torch.Tensor):
        if not getattr(self, "_state_cache_recording_enabled", False):
            return
        if (
            "state_cache_gate_history" not in self._buffers
            or self.state_cache_gate_history.numel() == 0
        ):
            return
        source_pos = self._ensure_source_history_conn_source_pos()
        gate_conn = source_gate.to(device=self.device, dtype=self.dtype).index_select(
            0, source_pos
        )
        self._record_gate_for_state_cache(gate_conn)

    def _bitpack_source_spikes_this_step(self) -> torch.Tensor:
        """Return one binary spike/event value per unique presynaptic source."""
        if self._bitpack_mode == "netstim":
            spikes = (
                self.pre.spikes.to(self.pre_device)
                .reshape(-1)
                .index_select(0, self.bitpack_source_pre_idx)
            )
            return spikes.to(torch.bool)

        x_full = self.get_pre_var(self.pre)
        if x_full.device != self.pre_device:
            x_full = x_full.to(device=self.pre_device)
        x = x_full.reshape(-1).index_select(0, self.bitpack_source_pre_idx)

        if self._bitpack_mode == "pre_var":
            # Explicit pre-side event variables are interpreted as binary gates.
            # bool tensors are used directly; numeric tensors fire when > 0.
            if x.dtype == torch.bool:
                return x
            return x.to(self.pre_dtype) > 0

        if self._bitpack_mode == "threshold":
            x = x.to(self.pre_dtype)
            self.bitpack_source_has_spiked, spikes = update_active(
                self.bitpack_source_has_spiked,
                x,
                self.bitpack_source_threshold,
            )
            return spikes.to(torch.bool)

        raise RuntimeError("bitpacked_history backend is not configured")

    def _try_bitpack_pack_kernel(
        self, source_spikes: torch.Tensor, cur_idx: torch.Tensor
    ) -> bool:
        if not source_spikes.is_cuda or self.spike_history_packed.dtype != torch.int64:
            return False
        from . import netcon_bitpack_ops as bitpack_ops

        try:
            available = bitpack_ops.is_available()
        except Exception as exc:
            return _handle_native_bitpack_failure("source-spike packing", exc)
        if not available:
            return _handle_native_bitpack_failure(
                "source-spike packing", bitpack_ops.last_error()
            )
        try:
            bitpack_ops.pack_source_spikes(
                source_spikes, self.spike_history_packed, cur_idx
            )
            return True
        except Exception as exc:
            return _handle_native_bitpack_failure("source-spike packing", exc)

    def _try_bitpack_delivery_kernel(
        self, cur_idx: torch.Tensor, scratch: torch.Tensor
    ) -> bool:
        if (
            (not self.spike_history_packed.is_cuda)
            or (not scratch.is_cuda)
            or (not torch.is_floating_point(scratch))
        ):
            return False
        from . import netcon_bitpack_ops as bitpack_ops

        try:
            available = bitpack_ops.is_available()
        except Exception as exc:
            return _handle_native_bitpack_failure("delivery construction", exc)
        if not available:
            return _handle_native_bitpack_failure(
                "delivery construction", bitpack_ops.last_error()
            )
        try:
            if getattr(self, "_dense_delay_uniform", False) and hasattr(
                bitpack_ops, "build_delivery_uniform"
            ):
                bitpack_ops.build_delivery_uniform(
                    self.spike_history_packed,
                    cur_idx,
                    int(getattr(self, "_dense_uniform_delay_step", 1)),
                    self.bitpack_conn_word_idx,
                    self.bitpack_conn_bit_mask,
                    self.post_idx,
                    self.weight(),
                    scratch,
                )
            else:
                bitpack_ops.build_delivery(
                    self.spike_history_packed,
                    cur_idx,
                    self.inference_delay_steps,
                    self.bitpack_conn_word_idx,
                    self.bitpack_conn_bit_mask,
                    self.post_idx,
                    self.weight(),
                    scratch,
                )
            return True
        except Exception as exc:
            return _handle_native_bitpack_failure("delivery construction", exc)

    def _bitpack_pack_source_spikes(
        self, source_spikes: torch.Tensor, cur_idx: torch.Tensor
    ):
        """Pack current source spikes into the current circular history row."""
        if source_spikes.numel() == 0:
            slot = int(cur_idx.item())
            self.spike_history_packed[slot].zero_()
            return
        if self._try_bitpack_pack_kernel(source_spikes, cur_idx):
            return

        # Pure-PyTorch fallback.  This is correct and memory-light in source
        # space; the lazy C++/CUDA extension above is the normal CUDA hot path.
        slot = int(cur_idx.item())
        row = self.spike_history_packed[slot]
        row.zero_()
        vals = self.bitpack_source_bit_mask * source_spikes.to(torch.int64)
        row.index_add_(0, self.bitpack_source_word_idx, vals)

    def _bitpack_build_delivery_from_history(
        self, cur_idx: torch.Tensor, scratch: torch.Tensor
    ):
        """Construct today's dense delivery payload from packed source history."""
        scratch.zero_()
        if self._n_conn == 0:
            return scratch
        if self._try_bitpack_delivery_kernel(cur_idx, scratch):
            return scratch

        # Pure-PyTorch fallback.  It allocates connection-sized gate/values, so it
        # is mainly for CPU, tests, and installations where the C++/CUDA extension
        # is unavailable or cannot handle the current launch. A failed native
        # launch may have written a partial result before reporting failure, so
        # reset the output again before constructing the complete fallback result.
        scratch.zero_()
        rows = (cur_idx - self.inference_delay_steps).remainder(self.max_delay_steps)
        words = self.spike_history_packed[rows.reshape(-1), self.bitpack_conn_word_idx]
        gate = torch.bitwise_and(words, self.bitpack_conn_bit_mask) != 0
        values = self.weight() * gate.to(dtype=self.dtype)
        scratch.index_add_(0, self.post_idx, values)
        return scratch

    def _bitpack_schedule_due_scheduled_events(self, cur_idx: torch.Tensor):
        """Route explicit scheduled per-connection events through sparse side buckets."""
        if not self._has_scheduled_events():
            return

        sched_wsum_conn, sched_counts_conn = self._scheduled_gate_this_step(
            self.global_step, use_tri_kernel=False
        )
        self.sched_wsum.copy_(sched_wsum_conn)
        self.sched_counts.copy_(sched_counts_conn)

        active = torch.nonzero(sched_wsum_conn != 0, as_tuple=False).flatten()
        if active.numel() == 0:
            return
        values = self.weight().index_select(0, active) * sched_wsum_conn.index_select(
            0, active
        )
        keep = values != 0
        if not bool(keep.any()):
            return
        active = active.index_select(0, torch.nonzero(keep, as_tuple=False).flatten())
        values = values[keep]
        delay_steps = self.inference_delay_steps.index_select(0, active)
        future_slots = (cur_idx + delay_steps).remainder(self.max_delay_steps)
        post_payload = self.post_idx.index_select(0, active)
        self._append_sparse_payloads(future_slots, post_payload, values)

    def _rebuild_delay_buffers(self):
        """
        Rebuild delay-related buffers from the current delay parameters.

        This is called by :meth:`initialize` when ``reinit_delays=True``.
        It:

        * recomputes integer delay steps from ``delay_ms()`` and ``dt``,
        * recomputes ``max_delay_steps``,
        * reallocates the delivery buffer and, when event tracking is enabled,
          the event queue to match the new depth, and
        * re-aligns all relevant buffers to the correct devices via
          :meth:`_align_buffer_devices`.
        """
        with torch.no_grad():
            delay_steps = (self.delay_ms() / self.dt.to(self.dtype)).round().long()
            self.delay_steps.copy_(delay_steps.flatten().to(self.device))

            self.max_delay_steps = self._compute_max_delay_steps()
            self._rebuild_dense_delay_metadata()

            needs_source_history = (
                self.delay_backend == "bitpacked_history"
                or self._source_history_training_requested()
            )
            if needs_source_history:
                # Delay changes do not alter source-level topology. Avoid rebuilding
                # conn_word_idx/conn_bit_mask on every reinitialization; resize only
                # the time-history rows needed by the active runtime.
                if not getattr(self, "_bitpack_can_use", False):
                    self._configure_bitpacked_history_metadata(
                        for_training=self._source_history_training_requested()
                    )
                if (
                    self.delay_backend == "bitpacked_history"
                    and not self._bitpack_can_use
                ):
                    raise ValueError(
                        "NetCon delay_backend='bitpacked_history' is not exact after delay rebuild: "
                        f"{self._bitpack_ineligible_reason}."
                    )
                if self._bitpack_can_use:
                    if (
                        self._use_source_history_training_runtime()
                        and self._source_history_diff_spiking_enabled()
                    ):
                        self._resize_source_gate_history(clear=True)
                    else:
                        self._resize_bitpacked_spike_history(clear=True)
            elif getattr(self, "_bitpack_can_use", False):
                # If source-history/bitpacked backends were switched off at runtime,
                # drop source-history-only state before allocating dense delay buffers.
                self._set_empty_bitpack_buffers()
                self._bitpack_can_use = False

            buffer_depth = self._desired_delivery_buffer_depth()
            buffer_shape = (buffer_depth, self.syn_numel.item())
            if tuple(self.delivery_buffer.shape) == buffer_shape:
                self.delivery_buffer.zero_()
            else:
                # Release the old buffer before allocating the replacement.  This
                # prevents reinit from briefly needing old+new delivery storage.
                self._set_buffer(
                    "delivery_buffer",
                    torch.empty(
                        (0, self._syn_numel), device=self.device, dtype=self.dtype
                    ),
                )
                self._set_buffer(
                    "delivery_buffer",
                    torch.zeros(buffer_shape, device=self.device, dtype=self.dtype),
                )
            if (
                self._use_sparse_calendar_runtime()
                or self._use_bitpacked_history_runtime()
                or self._use_source_history_training_runtime()
            ):
                self._clear_sparse_calendar()
                if (
                    self._use_bitpacked_history_runtime()
                    or self._use_source_history_training_runtime()
                ):
                    if (
                        self._use_source_history_training_runtime()
                        and self._source_history_diff_spiking_enabled()
                    ):
                        self._resize_source_gate_history(clear=True)
                    else:
                        self._resize_bitpacked_spike_history(clear=True)
            if self.track_events and not (
                self._use_sparse_calendar_runtime()
                or self._use_bitpacked_history_runtime()
                or self._use_source_history_training_runtime()
            ):
                event_queue = torch.zeros(
                    (self.max_delay_steps, self.n.item()),
                    device=self.device,
                    dtype=torch.int32,
                )
                if "event_queue" in self._buffers:
                    self.event_queue = event_queue
                else:
                    self.register_buffer("event_queue", event_queue)
            elif "event_queue" in self._buffers:
                del self._buffers["event_queue"]
            self.time_indices = torch.arange(
                self.max_delay_steps, device=self.device, dtype=torch.long
            )
            self._align_buffer_devices()

    def set_diff_config(
        self,
        diff_weights: bool = True,
        diff_delays: bool = True,
        diff_spiking: bool = True,
        taps: int = 2,
        sigma: float = 0.35,  # used for taps=3 (in steps),
        tau: float = 0.1,  # temperature for surrogate spiking
        diff_scheduled_times: bool = True,
        sched_width: float = 1.0,  # kernel half-width in *steps*
        train_delay_backend: Literal["dense", "source_history", "auto"] | None = None,
    ):
        """
        Configure differentiable behavior for training.

        This method must be called before the first call to :meth:`advance_diff`
        (via ``NetCon.advance`` in training mode). It controls which aspects of
        the connection are treated as differentiable and how.

        Parameters
        ----------
        diff_weights : bool, optional
            If True (default), synaptic weights are treated as differentiable
            parameters. If False, the weight tensor returned by ``self.weight()``
            is detached before being used to compute deliveries, freezing its
            contribution during training.
        diff_delays : bool, optional
            If True (default), delays are interpreted as continuous values in
            milliseconds and used with a soft, differentiable interpolation
            scheme over time steps (2-tap linear or 3-tap Gaussian, see
            ``taps`` and ``sigma``). If False, delays are rounded to integer
            steps and used as hard delay lines.
        diff_spiking : bool, optional
            If True (default), pre-synaptic spiking is computed via a surrogate
            gradient function (see :func:`update_active_diff`), allowing
            gradients to flow through threshold crossings. If False, hard
            thresholding via :func:`update_active` is used, yielding non-
            differentiable (boolean) spikes.
        taps : int, optional
            Number of interpolation taps to use when ``diff_delays=True``.

                * ``2`` (default): 2-tap linear interpolation between adjacent steps.
                * ``3``: 3-tap Gaussian-like interpolation weighted by ``sigma``.
        sigma : float, optional
            Standard deviation (in steps) for the 3-tap Gaussian interpolation
            when ``taps=3``. Ignored for ``taps=2``.
        tau : float, optional
            Temperature parameter for surrogate spiking, passed through to
            :func:`update_active_diff` as ``tau``. Lower values yield a steeper
            surrogate; higher values produce smoother gates.
        diff_scheduled_times : bool, optional
            If True (default), scheduled event times (including those provided
            via :meth:`schedule_time_ref`) are handled via a differentiable
            triangular kernel over time steps. This allows gradients to move
            event times when they come from value- or reference-mode tensors.
            If False, scheduled times are assigned to the first integer step at
            or after the event, and timing behavior is non-differentiable.
        sched_width : float, optional
            Half-width of the triangular kernel (in steps) when
            ``diff_scheduled_times=True``. A value of 1.0 yields contributions
            spread over approximately 2 steps around the nominal event time.
        train_delay_backend : {"dense", "source_history", "auto"}, optional
            Optional per-NetCon training backend override. ``"source_history"``
            requires source-level event semantics; ``"auto"`` uses source
            history when exact and otherwise keeps the dense differentiable path.

        Notes
        -----
        * ``set_diff_config`` does not itself change ``training`` mode; it only
          records configuration flags. ``NetCon.advance`` will dispatch to
          :meth:`advance_diff` when ``self.training = True`` and to
          :meth:`advance_non_diff` otherwise.
        * ``track_events=True`` is supported in training only when delays,
          spiking, and scheduled times all use hard semantics. Surrogate or
          fractional deliveries do not define unambiguous integer event counts.
        """
        if train_delay_backend is not None:
            if train_delay_backend not in ("dense", "source_history", "auto"):
                raise ValueError(
                    "train_delay_backend must be one of 'dense', 'source_history', or 'auto'."
                )
        if self.track_events and (diff_delays or diff_spiking or diff_scheduled_times):
            raise ValueError(
                "track_events=True in training requires hard event semantics: "
                "set diff_delays=False, diff_spiking=False, and "
                "diff_scheduled_times=False. Fractional surrogate deliveries "
                "do not have unambiguous integer event counts."
            )
        previous_backend = self.train_delay_backend
        previous_flags = self.train_flags
        if train_delay_backend is not None:
            self.train_delay_backend = train_delay_backend
        self.train_flags = (
            diff_weights,
            diff_delays,
            diff_spiking,
            taps,
            sigma,
            tau,
            diff_scheduled_times,
            float(sched_width),
        )
        try:
            self._validate_strict_source_history_policy()
            if self._mode_requires_initialize:
                # Validate strict source-history policy now, but do not replace
                # the fail-closed advance sentinel or reshape live pending
                # traffic during an incompatible mode transition.
                if self.training:
                    self._use_source_history_training_runtime()
                return
            # Re-select storage immediately for interactive/training-loop workflows.
            # This is especially important for ``diff_spiking=True`` source-history
            # training, which needs floating source-gate history rather than packed
            # hard spikes.
            self._refresh_training_advance_after_diff_config(clear_histories=True)
        except Exception:
            self.train_delay_backend = previous_backend
            self.train_flags = previous_flags
            raise

    @property
    def w(self):
        """
        Raw synaptic weight tensor for this ``NetCon``.

        Returns
        -------
        torch.Tensor
            The underlying weight tensor managed by the ``weight`` parameter
            module (i.e., ``self.weight.w``). This tensor has one entry per
            connection and is typically used for:

            * inspection and diagnostics,
            * direct regularization or constraints,
            * manual initialization or export.

        Notes
        -----
        To obtain the *current* weight values used during simulation steps
        (which may include functional transformations), use ``self.weight()``
        instead of accessing ``w`` directly.
        """
        return self.weight.w

    def _expand_pre_to_con(self, pre_indices: torch.Tensor | list[int]) -> torch.Tensor:
        """
        Expand pre-synaptic indices to connection indices using the CSR map.

        This helper is used by the scheduling APIs (e.g. :meth:`schedule`,
        :meth:`schedule_ref`, :meth:`schedule_time_ref`) when the caller
        specifies pre-synaptic indices instead of explicit connection
        indices.

        Parameters
        ----------
        pre_indices : array-like of int
            Pre-synaptic indices into the original population. These must be
            compatible with the pre index space used to construct ``pre_idx``.

        Returns
        -------
        torch.Tensor
            1-D Long tensor of connection indices containing all connections
            whose ``pre_idx`` entry matches one of the requested ``pre_indices``.
            The tensor may be empty if no connections originate from the
            specified pre indices.
        """
        pres = torch.as_tensor(
            pre_indices, device=self.pre_device, dtype=torch.long
        ).view(-1)
        if pres.numel() == 0:
            return pres

        # Find each pre id in CSR unique list with searchsorted
        if self._csr_pre_ids.numel() == 0:
            return torch.empty(0, device=self.pre_device, dtype=torch.long)
        pos = torch.searchsorted(self._csr_pre_ids, pres)
        valid = pos < self._csr_pre_ids.numel()
        # ``searchsorted`` returns len(ids) for values above the largest id.
        # Clamp only for the lookup, then retain the explicit bounds mask.
        safe_pos = pos.clamp_max(self._csr_pre_ids.numel() - 1)
        valid = valid & (self._csr_pre_ids.index_select(0, safe_pos) == pres)
        if not bool(valid.any()):
            return torch.empty(0, device=self.pre_device, dtype=torch.long)

        pos = pos[valid]
        starts = self._csr_starts.index_select(0, pos)
        lens = self._csr_counts.index_select(0, pos)

        # Gather slices of the sorted connection-index array and concat
        con_chunks = []
        for s, L in zip(starts.tolist(), lens.tolist()):
            if L > 0:
                con_chunks.append(self._csr_conidx_sorted.narrow(0, s, L))
        if len(con_chunks) == 0:
            return torch.empty(0, device=self.pre_device, dtype=torch.long)
        return torch.cat(con_chunks, dim=0)  # [n_con_from_these_pres]

    def schedule(
        self,
        *,
        con_indices=None,
        pre_indices=None,
        times_ms=None,
        weight=1.0,
        allow_past: bool = False,
    ):
        """
        Schedule value-mode events on specific connections or pre indices.

        This API attaches *value-mode* scheduled events, where weights and
        times are stored directly as tensors owned by the ``NetCon``. It is
        suitable when you want event-specific weights/times that are not
        shared by reference with other structures, or when you are fine
        managing these tensors directly.

        Exactly one of ``con_indices`` or ``pre_indices`` must be provided.

        Parameters
        ----------
        con_indices : Sequence[int] or torch.Tensor, optional
            Explicit connection indices at which to schedule events.
            Must be in ``[0, num_connections)``. Mutually exclusive with
            ``pre_indices``.
        pre_indices : Sequence[int] or torch.Tensor, optional
            Pre-synaptic indices. These are expanded internally to a set of
            connection indices using the pre→connection CSR map (see
            :meth:`_expand_pre_to_con`). Mutually exclusive with
            ``con_indices``.
        times_ms : Sequence[float] or torch.Tensor
            Event times in milliseconds. Must be broadcast-compatible with
            the selected connections *after* filtering out past events (see
            ``allow_past``). Times are converted to steps via
            the first grid step at or after the event for non-differentiable
            scheduling, and interpreted as continuous values when
            ``diff_scheduled_times=True``.
        weight : float or torch.Tensor, optional
            Dimensionless scalar or per-event weight *multiplier* applied on
            top of the base connection weights returned by ``self.weight()``.
            It is not another value in the target synapse's weight unit.
            Accepted shapes:

            * scalar (single value): broadcast to all scheduled events;
            * tensor of length ``E_before`` (events before past-event filtering):
              will be sliced by the filter; or
            * tensor of length ``E_after`` (events after filtering): used as is.

            When provided as a tensor requiring gradients, and when
            differentiable scheduling is enabled, gradients will flow into
            this tensor.
        allow_past : bool, optional
            If False (default), events whose scheduled integer step is smaller
            than the current ``global_step`` are dropped. If True, no such
            filtering is applied.

        Notes
        -----
        * This is a *value-mode* scheduling API: weights and times are stored
          directly in ``self.sched_weight`` and ``self.sched_time_ms``.
        * For *reference-mode* weights or times (where event metadata lives in
          a separate trainable tensor), use :meth:`schedule_ref` and
          :meth:`schedule_time_ref` after binding sources with
          :meth:`bind_weight_source` and :meth:`bind_time_source`.
        * This method can be called multiple times; new events are appended.
        """
        if (con_indices is None) == (pre_indices is None):
            raise ValueError("Provide exactly one of con_indices or pre_indices")
        if times_ms is None:
            raise ValueError("times_ms is required")

        device, dtype = self.device, self.dtype

        # → connection indices (raw, before any filtering)
        if pre_indices is not None:
            con_idx_raw = self._expand_pre_to_con(pre_indices).to(device)
        else:
            con_idx_raw = torch.as_tensor(
                con_indices, device=device, dtype=torch.long
            ).view(-1)

        tms_raw = torch.as_tensor(times_ms, device=device, dtype=dtype).view(-1)

        if con_idx_raw.numel() != tms_raw.numel():
            raise ValueError("con_indices/pre_indices and times_ms must match length")
        if (con_idx_raw < 0).any() or (con_idx_raw >= self.pre_idx.numel()).any():
            raise IndexError("connection index out of range")

        # legacy step field (used when diff_scheduled_times=False)
        steps_raw = _causal_step_index(tms_raw, self.dt.to(tms_raw.dtype))

        # optional filtering of past events
        if not allow_past:
            cur = self.global_step.view(())
            keep = steps_raw >= cur
            if not bool(keep.any()):
                return
            con_idx = con_idx_raw[keep]
            tms = tms_raw[keep]
            steps = steps_raw[keep]
        else:
            con_idx = con_idx_raw
            tms = tms_raw
            steps = steps_raw

        E_before = con_idx_raw.numel()
        E_after = con_idx.numel()

        # Normalize weight (supports scalar, pre-filter length, or post-filter length)
        if torch.is_tensor(weight):
            w_raw = weight.to(device=device, dtype=dtype).view(-1)
            if w_raw.numel() == 1:
                w = w_raw.expand(E_after)  # scalar → broadcast
            elif w_raw.numel() == E_before:
                w = (
                    w_raw[keep] if not allow_past else w_raw
                )  # per-event (pre-filter) → slice if filtered
            elif w_raw.numel() == E_after:
                w = w_raw  # per-event (post-filter) already aligned
            else:
                raise ValueError(
                    f"weight tensor length must be 1, {E_before} (pre-filter), or {E_after} (post-filter)"
                )
        else:
            w = torch.full((E_after,), float(weight), device=device, dtype=dtype)

        # append (outside compiled loop: changing E is fine)
        self.sched_con_idx = torch.cat([self.sched_con_idx, con_idx], dim=0)
        self.sched_abs_step = torch.cat([self.sched_abs_step, steps], dim=0)
        self.sched_time_ms = torch.cat([self.sched_time_ms, tms], dim=0)
        self.sched_weight = torch.cat([self.sched_weight, w], dim=0)

        # keep ref-index arrays aligned (sentinel -1)
        filler = torch.full((E_after,), -1, device=device, dtype=torch.long)
        self.sched_weight_idx = torch.cat([self.sched_weight_idx, filler], dim=0)
        self.sched_time_idx = torch.cat([self.sched_time_idx, filler], dim=0)

    def bind_weight_source(self, source: torch.Tensor):
        """
        Bind a tensor providing per-event weights for reference-mode scheduling.

        This method enables *reference-mode* scheduled weights used by
        :meth:`schedule_ref`. Instead of storing per-event weights directly
        inside ``NetCon``, you bind a tensor (usually a parameter) and then
        refer to entries in that tensor by integer indices.

        Parameters
        ----------
        source : torch.Tensor
            Tensor containing candidate dimensionless multipliers for scheduled
            events. Each value multiplies the target-unit base connection
            weight; it is not itself another target-unit weight. It must:

            * reside on the same device as the synapse (``self.device``), and
            * have a size at least as large as any ``weight_idx`` used in
              :meth:`schedule_ref`.

            Typically this is a ``torch.nn.Parameter`` or another tensor that
            participates in the training loop.

        Notes
        -----
        Order of operations:

            1. Construct the weight source tensor (e.g. ``nn.Parameter``).
            2. Call ``netcon.bind_weight_source(weight_source)`` exactly once (or
               whenever you want to replace the source).
            3. Use :meth:`schedule_ref` with ``weight_idx`` values indexing into
               the bound tensor.

        * Calling this replaces any previously bound weight source.
        * The tensor is not copied; ``NetCon`` keeps a reference and reads from
          it during simulation.
        * Gradients flow into the bound tensor whenever
          ``diff_weights=True`` and the scheduling path is differentiable.
        """
        if source.device != self.device:
            raise ValueError("weight source must be on same device")
        self._sched_w_source = source

    def bind_time_source(self, source: torch.Tensor):
        """
        Bind a tensor providing per-event times (ms) for reference-mode scheduling.

        This method enables *reference-mode* scheduled times used by
        :meth:`schedule_time_ref`. Instead of storing times directly in
        ``NetCon``, you bind a tensor (usually a parameter) and then reference
        entries in that tensor by integer indices.

        Parameters
        ----------
        source : torch.Tensor
            Tensor containing candidate event times in milliseconds. It must:

            * reside on the same device as the synapse (``self.device``), and
            * have a size at least as large as any ``time_idx`` used in
              :meth:`schedule_time_ref`.

            Typically this is a ``torch.nn.Parameter`` or another tensor that
            participates in the training loop.

        Notes
        -----
        Order of operations:

            1. Construct the time source tensor (e.g. ``nn.Parameter`` in ms).
            2. Call ``netcon.bind_time_source(time_source)``.
            3. Use :meth:`schedule_time_ref` with ``time_idx`` values indexing into
               the bound tensor.

        * Calling this replaces any previously bound time source.
        * The tensor is not copied; ``NetCon`` reads from it at runtime.
        * When used with :meth:`set_diff_config(diff_scheduled_times=True)`,
          and when the bound tensor requires gradients, event times become
          differentiable via the triangular timing kernel.
        """
        if source.device != self.device:
            raise ValueError("time source must be on same device")
        self._sched_t_source = source

    def schedule_ref(
        self,
        *,
        con_indices=None,
        pre_indices=None,
        times_ms=None,
        weight_idx=None,
        allow_past: bool = False,
    ):
        """
        Schedule reference-mode events with weights drawn from a bound source.

        This API is the reference-mode counterpart to :meth:`schedule`. Instead
        of passing explicit weights, you pass integer indices into a tensor
        that has previously been bound via :meth:`bind_weight_source`.

        Exactly one of ``con_indices`` or ``pre_indices`` must be provided,
        and :meth:`bind_weight_source` must have been called beforehand.

        Parameters
        ----------
        con_indices : Sequence[int] or torch.Tensor, optional
            Explicit connection indices at which to schedule events.
            Mutually exclusive with ``pre_indices``.
        pre_indices : Sequence[int] or torch.Tensor, optional
            Pre-synaptic indices to expand to connection indices via the CSR
            map (see :meth:`_expand_pre_to_con`). Mutually exclusive with
            ``con_indices``.
        times_ms : Sequence[float] or torch.Tensor
            Event times in milliseconds, one per event. Converted to integer
            steps via rounding for non-differentiable behavior; used as
            continuous times when ``diff_scheduled_times=True``.
        weight_idx : Sequence[int] or torch.Tensor
            Integer indices into the tensor bound by :meth:`bind_weight_source`.
            Must have the same length as ``times_ms`` and ``con_indices`` /
            expanded connections.
        allow_past : bool, optional
            If False (default), events scheduled before the current
            ``global_step`` are discarded. If True, no filtering is applied.

        Notes
        -----
        *Reference-mode* scheduling is designed for scenarios where you want
        scheduled weights to be:

        * shared across multiple events or :class:`NetCon` instances,
        * trained by a separate module or optimizer, or
        * constrained/regularized jointly (e.g., via an external loss).

        Because the weight values are read from a shared tensor at runtime,
        gradients from all events that reference a given index accumulate into
        that entry.

        **Order of operations**:

            1. Bind a weight source tensor:

               .. code-block:: python

                   nc.bind_weight_source(weight_source)

            2. Schedule reference-mode events:

               .. code-block:: python

                    nc.schedule_ref(
                        con_indices=...,
                        times_ms=...,
                        weight_idx=...,
                    )

            3. Run simulation (``nc.advance()`` in a loop).

        * If :meth:`bind_weight_source` has not been called, this method raises
          a ``RuntimeError``.
        * The value-mode slots for weights are still allocated but set to zero;
          only reference-mode contributions are active for these events.
        """
        if self._sched_w_source is None:
            raise RuntimeError("call bind_weight_source(...) before schedule_ref(...)")
        if (con_indices is None) == (pre_indices is None):
            raise ValueError("Provide exactly one of con_indices or pre_indices")
        if times_ms is None or weight_idx is None:
            raise ValueError("times_ms and weight_idx are required")

        device, dtype = self.device, self.dtype
        if pre_indices is not None:
            con_idx = self._expand_pre_to_con(pre_indices).to(device)
        else:
            con_idx = torch.as_tensor(
                con_indices, device=device, dtype=torch.long
            ).view(-1)

        tms = torch.as_tensor(times_ms, device=device, dtype=dtype).view(-1)
        widx = torch.as_tensor(weight_idx, device=device, dtype=torch.long).view(-1)
        if not (con_idx.numel() == tms.numel() == widx.numel()):
            raise ValueError("indices, times_ms, and weight_idx must have same length")

        if (con_idx < 0).any() or (con_idx >= self.pre_idx.numel()).any():
            raise IndexError("connection index out of range")
        if (widx < 0).any() or (widx >= self._sched_w_source.numel()).any():
            raise IndexError("weight_idx out of range for bound weight source")

        steps = _causal_step_index(tms, self.dt.to(tms.dtype))
        if not allow_past:
            cur = self.global_step.view(())
            keep = steps >= cur
            con_idx, tms, steps, widx = (
                con_idx[keep],
                tms[keep],
                steps[keep],
                widx[keep],
            )
            if con_idx.numel() == 0:
                return

        self.sched_con_idx = torch.cat([self.sched_con_idx, con_idx], dim=0)
        self.sched_abs_step = torch.cat([self.sched_abs_step, steps], dim=0)
        self.sched_time_ms = torch.cat([self.sched_time_ms, tms.to(dtype=dtype)], dim=0)
        self.sched_weight = torch.cat(
            [
                self.sched_weight,
                torch.zeros(con_idx.numel(), device=device, dtype=dtype),
            ],
            dim=0,
        )
        self.sched_weight_idx = torch.cat([self.sched_weight_idx, widx], dim=0)
        self.sched_time_idx = torch.cat(
            [
                self.sched_time_idx,
                torch.full((con_idx.numel(),), -1, device=device, dtype=torch.long),
            ],
            dim=0,
        )

    def schedule_time_ref(
        self,
        *,
        con_indices=None,
        pre_indices=None,
        time_idx=None,
        weight=1.0,
        allow_past: bool = False,
    ):
        """
        Schedule events whose times come by reference from a bound tensor.

        This API is the time-reference analogue of :meth:`schedule`. Instead
        of providing explicit event times, you provide integer indices into a
        tensor bound via :meth:`bind_time_source`. Weights are still supplied
        in value mode.

        Exactly one of ``con_indices`` or ``pre_indices`` must be provided, and
        :meth:`bind_time_source` must have been called beforehand.

        Parameters
        ----------
        con_indices : Sequence[int] or torch.Tensor, optional
            Explicit connection indices for the events.
        pre_indices : Sequence[int] or torch.Tensor, optional
            Pre-synaptic indices, expanded to connection indices via the CSR map.
        time_idx : Sequence[int] or torch.Tensor
            Integer indices into the tensor bound via :meth:`bind_time_source`.
            Must match the number of events after expanding ``pre_indices`` if
            used.
        weight : float or torch.Tensor, optional
            Dimensionless value-mode weight multiplier, as in
            :meth:`schedule`. May be a scalar or a per-event tensor of length
            equal to the number of scheduled events. Gradients can flow into
            this tensor when differentiable scheduling is enabled.
        allow_past : bool, optional
            If False (default), events whose current time (from the bound
            source) would be in the past relative to ``global_step`` are
            dropped. If True, keep all events.

        Notes
        -----
        Time-reference scheduling is designed for scenarios where you want to:

        * train event times explicitly as parameters,
        * share timing parameters across multiple events,
        * or couple event times to other model components (e.g., another network).

        When used with ``diff_scheduled_times=True`` in :meth:`set_diff_config`,
        gradients flow into the bound time tensor and can move events in time.

        **Order of operations**:

            1. Bind a time source tensor:

                .. code-block:: python

                    nc.bind_time_source(time_source)

            2. Schedule events by time index:

                .. code-block:: python

                    nc.schedule_time_ref(
                        con_indices=...,
                        time_idx=...,
                        weight=...,
                    )

            3. Run simulation (``nc.advance()`` in a loop).

        * If :meth:`bind_time_source` has not been called, this method raises a
          ``RuntimeError``.
        * The value-mode time slot in ``sched_time_ms`` is populated with zeros;
          the actual times are obtained at runtime from the bound tensor via
          ``sched_time_idx``.
        """
        if self._sched_t_source is None:
            raise RuntimeError(
                "call bind_time_source(...) before schedule_time_ref(...)"
            )
        if (con_indices is None) == (pre_indices is None):
            raise ValueError("Provide exactly one of con_indices or pre_indices")
        if time_idx is None:
            raise ValueError("time_idx is required")

        device, dtype = self.device, self.dtype

        # -> connection indices
        if pre_indices is not None:
            con_idx = self._expand_pre_to_con(pre_indices).to(device)
        else:
            con_idx = torch.as_tensor(
                con_indices, device=device, dtype=torch.long
            ).view(-1)

        tidx = torch.as_tensor(time_idx, device=device, dtype=torch.long).view(-1)
        if con_idx.numel() != tidx.numel():
            raise ValueError("con_indices/pre_indices and time_idx must match length")

        if (con_idx < 0).any() or (con_idx >= self.pre_idx.numel()).any():
            raise IndexError("connection index out of range")
        if (tidx < 0).any() or (tidx >= self._sched_t_source.numel()).any():
            raise IndexError("time_idx out of range for bound time source")

        # Use a zeros value slot for times and record sched_time_idx. Keep the
        # legacy hard-step field populated from the current reference values.
        tms_now = self._sched_t_source.index_select(0, tidx).to(dtype=dtype)
        steps = _causal_step_index(tms_now, self.dt.to(dtype))
        E_before = con_idx.numel()
        if not allow_past:
            cur = self.global_step.view(())
            keep = steps >= cur
            con_idx, steps, tidx, tms_now = (
                con_idx[keep],
                steps[keep],
                tidx[keep],
                tms_now[keep],
            )
            if con_idx.numel() == 0:
                return
        E_after = con_idx.numel()

        # weight value-mode (optional)
        if torch.is_tensor(weight):
            w_raw = weight.to(device=device, dtype=dtype).view(-1)
            if w_raw.numel() == 1:
                w = w_raw.expand(E_after)
            elif w_raw.numel() == E_before:
                w = w_raw[keep] if not allow_past else w_raw
            elif w_raw.numel() == E_after:
                w = w_raw
            else:
                raise ValueError(
                    f"weight must be scalar or have length {E_before} "
                    f"(pre-filter) or {E_after} (post-filter)"
                )
        else:
            w = torch.full((E_after,), float(weight), device=device, dtype=dtype)

        self.sched_con_idx = torch.cat([self.sched_con_idx, con_idx], dim=0)
        self.sched_abs_step = torch.cat([self.sched_abs_step, steps], dim=0)
        self.sched_time_ms = torch.cat(
            [
                self.sched_time_ms,
                torch.zeros(E_after, device=device, dtype=dtype),
            ],
            dim=0,
        )  # value slot 0
        self.sched_time_idx = torch.cat([self.sched_time_idx, tidx], dim=0)
        self.sched_weight = torch.cat([self.sched_weight, w], dim=0)

        # keep weight ref index aligned (no weight-ref here, so -1)
        self.sched_weight_idx = torch.cat(
            [
                self.sched_weight_idx,
                torch.full((con_idx.numel(),), -1, device=device, dtype=torch.long),
            ],
            dim=0,
        )

    def clear_schedule(self):
        """
        Remove all scheduled events and reset scheduling-related buffers.

        This clears both value-mode and reference-mode scheduled events:

        * connection indices (``sched_con_idx``),
        * absolute step indices (``sched_abs_step``),
        * value-mode weights and times (``sched_weight``, ``sched_time_ms``),
        * reference-mode indices (``sched_weight_idx``, ``sched_time_idx``),
        * per-connection introspection statistics (``sched_wsum``,
          ``sched_counts``).

        The bound sources set via :meth:`bind_weight_source` and
        :meth:`bind_time_source` are **not** modified.

        Notes
        -----
        * This is typically called between episodes or trials when you want
          to reuse the same ``NetCon`` instance but with a fresh schedule.
        """
        device, dtype = self.device, self.dtype
        self.sched_con_idx = torch.empty(0, device=device, dtype=torch.long)
        self.sched_abs_step = torch.empty(0, device=device, dtype=torch.long)
        self.sched_weight = torch.empty(0, device=device, dtype=dtype)
        self.sched_weight_idx = torch.empty(0, device=device, dtype=torch.long)
        self.sched_time_ms = torch.empty(0, device=device, dtype=dtype)
        self.sched_time_idx = torch.empty(0, device=device, dtype=torch.long)
        self.sched_wsum.zero_()
        self.sched_counts.zero_()

    def _scheduled_gate_this_step(self, gs_long: torch.Tensor, *, use_tri_kernel: bool):
        """
        Aggregate scheduled contributions for the current absolute step.

        This internal helper constructs:

        * a per-connection scheduled gate amplitude (sum of per-event weights),
        * a per-connection count of active scheduled events (for bookkeeping),

        given the current global step and the configuration flags in
        ``self.train_flags``.

        Parameters
        ----------
        gs_long : torch.Tensor
            Scalar Long tensor with the current global step index.
        use_tri_kernel : bool
            If True, use a differentiable triangular kernel in step space for
            scheduled times (used when ``diff_scheduled_times=True``). If False,
            scheduled events are active only when their first causal grid step
            equals ``gs_long``.

        Returns
        -------
        sched_wsum_conn : torch.Tensor
            1-D tensor of shape ``[n_conn]`` containing the scheduled gate
            amplitudes per connection for this step.
        sched_counts_conn : torch.Tensor
            1-D int32 tensor of shape ``[n_conn]`` containing the number of
            scheduled events contributing to each connection at this step.
        """
        n_conn = self._n_conn
        device, dtype = self.device, self.dtype

        if self.sched_con_idx.numel() == 0:
            return (
                torch.zeros(n_conn, device=device, dtype=dtype),
                torch.zeros(n_conn, device=device, dtype=torch.int32),
            )

        # --- scheduled weights (value + ref) ---
        w_val = self.sched_weight
        if self._sched_w_source is not None:
            idxw = torch.clamp(self.sched_weight_idx, min=0)
            w_src = self._sched_w_source.index_select(0, idxw)
            w_ref_mask = (self.sched_weight_idx >= 0).to(dtype)
            w_val = w_val + w_src * w_ref_mask  # combine if both present

        # --- scheduled times (value + ref) -> float ms ---
        t_val = self.sched_time_ms  # may be zeros if using time-by-ref only
        if self._sched_t_source is not None:
            idxt = torch.clamp(self.sched_time_idx, min=0)
            t_src = self._sched_t_source.index_select(0, idxt).to(dtype=dtype)
            t_ref_mask = (self.sched_time_idx >= 0).to(t_src.dtype)
            t_val = t_val + t_src * t_ref_mask

        if use_tri_kernel:
            # Differentiable triangular kernel in *steps*
            lam = t_val.to(dtype) / self.dt.to(dtype)  # [E] float
            # gs_long is int64; cast to float
            x = lam - gs_long.to(dtype).view(())
            # width in steps (half-width). <=0 disables contribution
            _, _, _, _, _, _, _, sched_width = self.train_flags
            tri = (1.0 - (x.abs() / (sched_width + 1e-6))).clamp(
                min=0.0, max=1.0
            )  # [E]
            amp_evt = w_val * tri  # [E]
            # counts (bookkeeping): event "active" if tri>0
            cnt_evt = (tri > 0).to(torch.int32)  # [E]
        else:
            # Step-exact, causal firing: first grid step >= event time.
            abs_step = _causal_step_index(t_val.to(dtype), self.dt.to(dtype))
            now_mask = abs_step == gs_long.view(())
            amp_evt = w_val * now_mask.to(dtype)  # [E]
            cnt_evt = now_mask.to(torch.int32)  # [E]

        # Aggregate to connections
        sched_wsum_conn = torch.zeros(n_conn, device=device, dtype=dtype)
        sched_counts_conn = torch.zeros(n_conn, device=device, dtype=torch.int32)
        sched_wsum_conn.index_add_(0, self.sched_con_idx, amp_evt)
        sched_counts_conn.index_add_(0, self.sched_con_idx, cnt_evt)
        return sched_wsum_conn, sched_counts_conn

    def advance_diff_source_history(self):
        """Differentiable source-history training path.

        This backend replaces the dense ``[max_delay_steps, syn_numel]`` training
        delivery ring with source-level event history.  With ``diff_spiking=False``
        the source history is bit-packed and treated as a hard event tape; weight
        and delay gradients still flow for realized events.  With
        ``diff_spiking=True`` the source history is floating and preserves
        surrogate source-gate gradients.
        """
        if self.train_flags is None:
            raise RuntimeError(
                "NetCon.set_diff_config(...) must be called before "
                "advance_diff_source_history()."
            )
        if self._has_scheduled_events():
            raise RuntimeError(
                "train_delay_backend='source_history' currently supports intrinsic "
                "source-level events only. Use train_delay_backend='dense' for "
                "scheduled-event training."
            )
        (
            diff_weights,
            diff_delays,
            diff_spiking,
            taps,
            sigma,
            tau,
            diff_sched_times,
            _,
        ) = self.train_flags
        if diff_sched_times and self._has_scheduled_events():
            raise RuntimeError(
                "Differentiable scheduled times are not supported by source-history training."
            )

        cur_idx = self.current_time_step.detach()

        todays = self._source_history_build_delivery_from_history(
            cur_idx,
            diff_weights=bool(diff_weights),
            diff_delays=bool(diff_delays),
            taps=int(taps),
            sigma=float(sigma),
        )
        self.syn.net_receive(todays.view(*self.syn.shape_f), self)

        source_gate = self._source_history_source_gate_this_step(
            diff_spiking=bool(diff_spiking),
            tau=float(tau),
        )
        self._record_source_gate_for_state_cache_if_needed(source_gate)
        self._source_history_record_current_gate(
            cur_idx,
            source_gate,
            diff_spiking=bool(diff_spiking),
        )

        with torch.no_grad():
            self.current_time_step = (
                (self.current_time_step + 1).remainder(self.max_delay_steps).detach()
            )
            self.global_step = (self.global_step + 1).detach()

    def advance_diff(self):
        """
        Advance the connection state by one time step (differentiable path).

        This method implements the differentiable update kernel used when
        ``self.training`` is True and :meth:`set_diff_config` has been
        called. It performs:

        1. Delivery of the current delay-buffer row to ``syn.net_receive``.
        2. Determination of intrinsic spiking via ``determine_spiking`` with
           surrogate gradients if configured.
        3. Construction of the per-connection gate combining intrinsic and
           scheduled contributions (with optional differentiable timing).
        4. Application of weights (optionally differentiable) and delays
           (optionally differentiable via interpolation) to write into future
           delay-buffer rows.
        5. Update of ``current_time_step`` and ``global_step``.

        Notes
        -----
        * Users typically do not call this directly; instead, call
          ``netcon.initialize(...)`` and then invoke ``netcon.advance()``
          inside a simulation loop.
        * A ``RuntimeError`` is raised if :meth:`set_diff_config` has not been
          called (``train_flags is None``).
        """
        if self.train_flags is None:
            raise RuntimeError(
                "NetCon.train(...) must be called before advance_diff()."
            )
        (
            diff_weights,
            diff_delays,
            diff_spiking,
            taps,
            sigma,
            tau,
            diff_sched_times,
            _,
        ) = self.train_flags
        device, dtype = self.device, self.dtype
        if self.track_events and (diff_delays or diff_spiking or diff_sched_times):
            raise RuntimeError(
                "Tracked training events require hard delay, spiking, and "
                "scheduled-time semantics. Reconfigure this NetCon before "
                "advancing it."
            )

        # Snapshot indices for this step (avoid version bumps).
        # clone() is not necessary here; detach is enough because we never mutate
        # these tensors in-place and we treat them as non-differentiable counters.
        cur_idx = self.current_time_step.detach()  # [1], long
        gs = self.global_step.detach()  # [1], long

        # 1) deliver today's payload and expose the matching hard-event counts.
        # Ambiguous surrogate/fractional event tracking is rejected by
        # set_diff_config(), so track_events here always has integer semantics.
        todays = self.delivery_buffer.index_select(0, cur_idx).squeeze(0)  # [n_syn]
        if self.track_events:
            self.events.copy_(self.event_queue.index_select(0, cur_idx).squeeze(0))
        else:
            self.events.zero_()
        self.syn.net_receive(todays.view(*self.syn.shape_f), self)

        # 2) intrinsic spiking
        self.determine_spiking(
            self.pre, diff_spiking, tau
        )  # self.is_spiking -> [n_conn] bool
        intrinsic_gate = self.is_spiking.to(device=device, dtype=dtype)  # [n_conn]

        # scheduled additions per connection (constant shapes)
        sched_wsum_conn, sched_counts_conn = self._scheduled_gate_this_step(
            gs, use_tri_kernel=bool(diff_sched_times)
        )

        # expose (non-diff)
        with torch.no_grad():
            self.sched_wsum.copy_(sched_wsum_conn)
            self.sched_counts.copy_(sched_counts_conn)

        # combine gates
        gate = intrinsic_gate + sched_wsum_conn  # [n_conn]
        self._record_gate_for_state_cache(gate)
        event_counts = None
        if self.track_events:
            event_counts = (intrinsic_gate > 0).to(torch.int32) + sched_counts_conn

        # weights (optionally detach)
        wvals = self.weight()
        if not diff_weights:
            wvals = wvals.detach()
        weighted_spikes = wvals * gate  # [n_conn]

        # ------------------------------------------------------------------
        # Build next delivery buffer WITHOUT allocating a full-size "contrib"
        # buffer and WITHOUT constructing a full-size dense mask.
        #
        # We clone once (this is the output buffer), clear the delivered row,
        # then scatter-add into it.
        #
        # This reduces per-step peak memory substantially compared to:
        #   contrib=zeros_like(buf_flat) + mask2d + (buf_flat*mask2d + contrib)
        # ------------------------------------------------------------------
        buf_next = self.delivery_buffer.clone()
        buf_next.index_fill_(0, cur_idx, 0.0)  # clear the row we just delivered
        buf_flat = buf_next.view(-1)
        event_queue_next = None
        if self.track_events:
            event_queue_next = self.event_queue.clone()
            event_queue_next.index_fill_(0, cur_idx, 0)

        if diff_delays:
            d_ms = self.delay_ms().to(dtype)  # [n_conn]
            # Detection happens after the current receive slot has already been
            # delivered.  Consequently, even a sub-timestep physical delay can
            # arrive no earlier than the next simulation step.  Keep the
            # differentiable dense path aligned with inference and the compact
            # source-history training backend; otherwise a rounded-zero delay
            # is written into the cleared current slot and is not seen again
            # until the circular buffer wraps.
            lam = (d_ms / self.dt.to(dtype)).clamp_min(1.0)
            k = torch.floor(lam)
            kL = k.to(torch.long)

            if taps == 2:
                alpha = (lam - k).to(dtype)
                idx0 = (cur_idx + kL).remainder(self.max_delay_steps)
                idx1 = (idx0 + 1).remainder(self.max_delay_steps)
                flat0 = idx0 * self.syn_numel + self.post_idx
                flat1 = idx1 * self.syn_numel + self.post_idx
                # Vectorized 2-tap write
                buf_flat.index_add_(0, flat0, weighted_spikes * (1.0 - alpha))
                buf_flat.index_add_(0, flat1, weighted_spikes * alpha)
            else:
                offs = torch.stack([kL - 1, kL, kL + 1], dim=-1)
                centers = offs.to(dtype)
                lam_e = lam.unsqueeze(-1)
                w = torch.softmax(
                    -0.5 * ((lam_e - centers) / (sigma + 1e-6)) ** 2, dim=-1
                )

                idxs = (cur_idx + offs).remainder(self.max_delay_steps)
                # Vectorized 3-tap write
                flat_idx = idxs * self.syn_numel + self.post_idx.unsqueeze(
                    -1
                )  # [n_conn,3]
                vals = weighted_spikes.unsqueeze(-1) * w  # [n_conn,3]
                buf_flat.index_add_(0, flat_idx.reshape(-1), vals.reshape(-1))
        else:
            future_steps = (cur_idx + self.inference_delay_steps).remainder(
                self.max_delay_steps
            )
            flat = future_steps * self.syn_numel + self.post_idx
            # Integer delay: single destination per connection
            buf_flat.index_add_(0, flat, weighted_spikes)
            if event_queue_next is not None:
                flat_events = future_steps * self._n_conn + self.con_range
                event_queue_next.view(-1).index_add_(0, flat_events, event_counts)

        # Commit next buffer (already cleared + updated)
        self.delivery_buffer = buf_next
        if event_queue_next is not None:
            self.event_queue = event_queue_next

        # advance counters (no grad)
        with torch.no_grad():
            # Rebind buffers (out-of-place) to avoid version bumps on saved tensors
            self.current_time_step = (
                (self.current_time_step + 1).remainder(self.max_delay_steps).detach()
            )
            self.global_step = (self.global_step + 1).detach()

    def _dense_current_delivery(self):
        """Return the dense delivery row due at the current slot."""
        cur_idx = self.current_time_step
        todays_delivery = self.delivery_buffer.index_select(0, cur_idx).squeeze(0)
        return cur_idx, todays_delivery

    def _dense_current_events(self, cur_idx):
        """Expose per-connection event counts for the current slot if enabled."""
        if self.track_events:
            self.events.copy_(self.event_queue.index_select(0, cur_idx).squeeze(0))
        else:
            self.events.zero_()

    def _dense_clear_current_slot(self, cur_idx):
        """Clear the dense delivery/event row after it has been delivered."""
        self.delivery_buffer.index_fill_(0, cur_idx, 0.0)
        if self.track_events:
            self.event_queue.index_fill_(0, cur_idx, 0)

    def _dense_gate_and_event_counts(self):
        """Compute this step's connection gate and optional event counts."""
        self.determine_spiking(self.pre, diff_spiking=False)
        intrinsic_gate = self.is_spiking.to(device=self.device, dtype=self.dtype)

        if self._has_scheduled_events():
            sched_wsum_conn, sched_counts_conn = self._scheduled_gate_this_step(
                self.global_step, use_tri_kernel=False
            )
            with torch.no_grad():
                self.sched_wsum.copy_(sched_wsum_conn)
                self.sched_counts.copy_(sched_counts_conn)
            gate = intrinsic_gate + sched_wsum_conn
            if self.track_events:
                event_counts = (intrinsic_gate > 0).to(torch.int32) + sched_counts_conn
            else:
                event_counts = None
        else:
            gate = intrinsic_gate
            if self.track_events:
                event_counts = (intrinsic_gate > 0).to(torch.int32)
            else:
                event_counts = None

        self._record_gate_for_state_cache(gate)
        return gate, event_counts

    def _dense_schedule_uniform(self, cur_idx, weighted_spikes, event_counts=None):
        """Schedule all connections into one future dense delay row."""
        future = (cur_idx + int(self._dense_uniform_delay_step)).remainder(
            self.max_delay_steps
        )
        flat = future * self._syn_numel + self.post_idx
        self.delivery_buffer.view(-1).index_add_(0, flat.reshape(-1), weighted_spikes)

        if self.track_events and event_counts is not None:
            flat_e = future * self._n_conn + self.con_range
            self.event_queue.view(-1).index_add_(0, flat_e.reshape(-1), event_counts)

    def _dense_schedule_mixed(self, cur_idx, weighted_spikes, event_counts=None):
        """Schedule all connections using precomputed per-connection offsets."""
        base = cur_idx * self._syn_numel
        flat = (base + self.flat_delay_offsets).remainder(self._delivery_numel)
        self.delivery_buffer.view(-1).index_add_(0, flat.reshape(-1), weighted_spikes)

        if self.track_events and event_counts is not None:
            base_e = cur_idx * self._n_conn
            flat_e = (base_e + self.flat_event_offsets).remainder(
                self._event_queue_numel
            )
            self.event_queue.view(-1).index_add_(0, flat_e.reshape(-1), event_counts)

    def _advance_dense_counters(self):
        self.current_time_step.add_(1).remainder_(self.max_delay_steps)
        self.global_step.add_(1)

    def advance_non_diff_dense_uniform(self):
        """Dense inference hot path for NetCons whose integer delays are uniform."""
        cur_idx, todays_delivery = self._dense_current_delivery()
        self._dense_current_events(cur_idx)
        self.syn.net_receive(todays_delivery.view(*self.syn.shape_f), self)
        self._dense_clear_current_slot(cur_idx)

        gate, event_counts = self._dense_gate_and_event_counts()
        weighted_spikes = self.weight() * gate
        self._dense_schedule_uniform(cur_idx, weighted_spikes, event_counts)
        self._advance_dense_counters()

    def advance_non_diff_dense_mixed(self):
        """Dense inference hot path for NetCons with mixed integer delays."""
        cur_idx, todays_delivery = self._dense_current_delivery()
        self._dense_current_events(cur_idx)
        self.syn.net_receive(todays_delivery.view(*self.syn.shape_f), self)
        self._dense_clear_current_slot(cur_idx)

        gate, event_counts = self._dense_gate_and_event_counts()
        weighted_spikes = self.weight() * gate
        self._dense_schedule_mixed(cur_idx, weighted_spikes, event_counts)
        self._advance_dense_counters()

    def advance_non_diff(self):
        """
        Advance the connection state by one non-differentiable timestep.

        This compatibility entry point dispatches to the dense inference hot path
        selected from precomputed delay metadata.  ``initialize()`` normally
        binds ``self.advance`` directly to the selected method, so this wrapper
        is only used by direct callers.
        """
        if self._use_bitpacked_history_runtime():
            return self.advance_non_diff_bitpacked_history()
        if self._use_sparse_calendar_runtime():
            return self.advance_non_diff_sparse_calendar()
        return self._dense_advance_target()()

    @torch.no_grad()
    def advance_non_diff_sparse_calendar(self):
        """
        Advance one inference step using sparse calendar buckets.

        This path preserves the existing ``syn.net_receive(payload, self)``
        contract by reducing due sparse deliveries into a reusable dense scratch
        row before delivery.  Unlike :meth:`advance_non_diff`, it does not keep
        a dense ``[max_delay_steps, syn_numel]`` pending-delivery ring.  Pending
        future deliveries are stored only as nonzero ``(post_idx, value)``
        chunks in ``self._sparse_calendar``.
        """
        cur_idx = self.current_time_step
        cur_slot = int(cur_idx.item())

        # 1) Deliver events whose ring slot is due now.  This produces the same
        # dense payload shape expected by existing synaptic mechanisms.
        todays_delivery = self._pop_sparse_calendar_to_dense(cur_slot)
        self._pop_sparse_events(cur_slot)
        self.syn.net_receive(todays_delivery.view(*self.syn.shape_f), self)

        # 2) Intrinsic spikes / gates (hard, inference-only).
        self.determine_spiking(self.pre, diff_spiking=False)
        intrinsic_gate = self.is_spiking.to(
            device=self.device, dtype=self.dtype
        )  # [n_conn]

        # 3) Scheduled events still use the existing exact-step aggregator.  A
        # later optimization can replace this with its own source-event calendar.
        gs = self.global_step
        sched_wsum_conn, sched_counts_conn = self._scheduled_gate_this_step(
            gs, use_tri_kernel=False
        )
        self.sched_wsum.copy_(sched_wsum_conn)
        self.sched_counts.copy_(sched_counts_conn)

        # 4) Schedule only nonzero payloads into future calendar slots.
        gate = intrinsic_gate + sched_wsum_conn
        self._record_gate_for_state_cache(gate)
        active = torch.nonzero(gate != 0, as_tuple=False).flatten()
        if active.numel() > 0:
            gate_a = gate.index_select(0, active)
            values = self.weight().index_select(0, active) * gate_a
            keep_payload = values != 0
            if bool(keep_payload.any()):
                con_payload = active.index_select(
                    0, torch.nonzero(keep_payload, as_tuple=False).flatten()
                )
                values = values[keep_payload]
                # Since delays are positive-valued by construction, the soonest
                # discrete delivery is the next timestep.  Clamp protects the
                # sparse backend from round-to-zero delays.
                delay_steps = self.inference_delay_steps.index_select(0, con_payload)
                future_slots = (cur_idx + delay_steps).remainder(self.max_delay_steps)
                post_payload = self.post_idx.index_select(0, con_payload)
                self._append_sparse_payloads(future_slots, post_payload, values)

        # 5) Optional per-connection event introspection, without allocating the
        # dense ``[max_delay_steps, n_conn]`` debug queue.
        if self.track_events:
            event_counts = (intrinsic_gate > 0).to(torch.int32) + sched_counts_conn
            active_events = torch.nonzero(event_counts != 0, as_tuple=False).flatten()
            if active_events.numel() > 0:
                delay_steps = self.inference_delay_steps.index_select(0, active_events)
                future_slots = (cur_idx + delay_steps).remainder(self.max_delay_steps)
                counts = event_counts.index_select(0, active_events)
                self._append_sparse_events(future_slots, active_events, counts)

        self.current_time_step.add_(1).remainder_(self.max_delay_steps)
        self.global_step.add_(1)

    @torch.no_grad()
    def advance_non_diff_bitpacked_history(self):
        """Advance one inference step using bitpacked source-spike history.

        This backend stores one binary spike history per unique presynaptic
        source, packed into int64 words, rather than storing a dense future
        delivery row for every delay slot.  Today's dense ``net_receive`` payload
        is reconstructed from the packed history and the static edge list.
        """
        cur_idx = self.current_time_step
        scratch = self.delivery_buffer.squeeze(0)

        # 1) Reconstruct and deliver all intrinsic delayed source spikes due now.
        self._bitpack_build_delivery_from_history(cur_idx, scratch)

        # 2) Optional explicit scheduled per-connection events are handled by a
        # sparse side calendar so they do not force a dense delay buffer.
        if self._sparse_calendar:
            self._add_sparse_calendar_to_dense(int(cur_idx.item()), scratch)
        if self.events.numel() > 0:
            self.events.zero_()
        self.syn.net_receive(scratch.view(*self.syn.shape_f), self)

        # 3) Detect current source-level spikes and pack them into the current
        # history row.  They can be delivered no earlier than the next timestep
        # because inference_delay_steps is clamped to >= 1.
        source_spikes = self._bitpack_source_spikes_this_step()
        self._bitpack_pack_source_spikes(source_spikes, cur_idx)

        # 4) Scheduled events, if present, are evaluated exactly for this global
        # step and inserted into sparse side buckets after applying NetCon delay.
        self._bitpack_schedule_due_scheduled_events(cur_idx)

        self.current_time_step.add_(1).remainder_(self.max_delay_steps)
        self.global_step.add_(1)

    def determine_spiking_ns(self, pre: NetStim, diff_spiking: bool = True, tau=None):
        """
        Determine spiking when the pre-synaptic source is a :class:`NetStim`.

        For :class:`NetStim` sources, the pre-synaptic event signal is already
        represented as ``spikes`` (hard) or ``spike_gate`` (soft/surrogate).
        This method:

        * extracts the relevant entries using ``pre_idx``,
        * selects hard or soft gates depending on ``diff_spiking``, and
        * stores the result in ``self.is_spiking`` as a float tensor.

        Parameters
        ----------
        pre : NetStim
            Pre-synaptic :class:`NetStim` providing ``spikes`` and
            ``spike_gate`` attributes.
        diff_spiking : bool, optional
            If True, use ``pre.spike_gate`` directly (assumed differentiable).
            If False, use ``pre.spikes`` and cast to float gates {0, 1}.
        tau : float, optional
            Unused for :class:`NetStim` sources; present for API symmetry with
            :meth:`determine_spiking_var`.
        """
        # If the pre-synaptic source is a NetStim, we can directly use its spikes
        # (or spikes_gate for differentiable spiking)
        if diff_spiking:
            # surrogate / soft gate, already float
            gate = (
                pre.spike_gate.to(self.pre_device)
                .view(-1)
                .index_select(0, self.pre_idx)
            )
        else:
            # hard spikes; usually bool → cast to float gate {0, 1}
            gate = (
                pre.spikes.to(self.pre_device)
                .view(-1)
                .index_select(0, self.pre_idx)
                .to(self.pre_dtype)
            )

        # Ensure dtype is always self.dtype
        self.is_spiking = gate.to(device=self.pre_device, dtype=self.pre_dtype)

    def determine_spiking_var(self, pre, diff_spiking: bool = True, tau=0.1):
        """
        Determine spiking from a generic pre-synaptic variable (non-NetStim).

        This method is used when the pre-synaptic source is not a
        :class:`NetStim`. It:

        1. Extracts the relevant entries from the configured pre variable
           (e.g., ``pre.v``) using ``pre_idx``.
        2. Applies either surrogate-gradient or hard thresholding depending on
           ``diff_spiking`` and the presence of finite thresholds.
        3. Handles NaN thresholds by optionally using the raw variable as a
           continuous gate for those connections.

        Parameters
        ----------
        pre :
            Pre-synaptic module exposing the attribute named by
            ``self.get_pre_var`` (usually ``"v"``).
        diff_spiking : bool, optional
            If True, use :func:`update_active_diff` with a surrogate gradient
            defined by ``tau``. If False, use :func:`update_active` with hard
            thresholding.
        tau : float, optional
            Temperature parameter passed through to :func:`update_active_diff`.
            Smaller values yield steeper surrogate activation functions.

        Notes
        -----
        * If all thresholds are NaN (``skip_thresholding=True``), the method
          simply mirrors the pre variable as a continuous gate.
        * If some thresholds are NaN and others are finite, the method applies
          thresholding where finite and uses raw pre values where thresholds
          are NaN.
        """
        # Always work in the module's float dtype
        v_selected = (
            self.get_pre_var(pre)
            .to(self.pre_device)
            .view(-1)
            .index_select(0, self.pre_idx)
            .to(self.pre_dtype)
        )

        if self.skip_thresholding:
            # Just mirror v as a continuous "gate"
            self.is_spiking = v_selected.to(
                device=self.pre_device, dtype=self.pre_dtype
            )
            return

        if diff_spiking:
            # Surrogate / differentiable spiking
            self.has_spiked, _, gate = update_active_diff(
                self.has_spiked, v_selected, self.threshold, tau=tau
            )
            # gate is typically float already, but enforce dtype
            gate = gate.to(self.pre_dtype)
        else:
            # Hard threshold spiking; usually returns Bool spikes
            self.has_spiked, spikes = update_active(
                self.has_spiked, v_selected, self.threshold
            )
            # Turn {False,True} into {0.0,1.0} float gate
            gate = spikes.to(self.pre_dtype)

        if self.apply_masking:
            # thresh_is_nan: bool mask; choose between v_selected and gate (both float)
            gate = torch.where(
                self.thresh_is_nan,
                v_selected,
                gate,
            )

        self.is_spiking = gate.to(device=self.pre_device, dtype=self.pre_dtype)

    def _advance_requires_mode_initialize(self):
        raise RuntimeError(
            "NetCon train/eval mode changed across incompatible runtime layouts. "
            "Call initialize() before advancing this connection."
        )

    def train(self, mode: bool = True):  # type: ignore[override]
        previous_mode = bool(self.training)
        previous_advance = getattr(getattr(self, "advance", None), "__name__", "")
        self._validate_train_mode_transition(mode)
        super().train(mode)
        if self._mode_requires_initialize:
            return self
        if mode:
            wants_compact_training = self._use_source_history_training_runtime()
            if previous_mode != bool(mode) and (
                self.delay_backend != "dense" or wants_compact_training
            ):
                # Preserve the existing runtime verbatim. initialize() owns the
                # conversion to compact/dense storage and clears this guard only
                # after every dependent buffer has been rebuilt coherently.
                self._mode_requires_initialize = True
                self.advance = self._advance_requires_mode_initialize
                return self
            # Defer dense per-connection buffer allocation when the compact
            # source-history backend may be selected.  If diff flags have not
            # been configured yet, ``initialize``/``set_diff_config`` will choose
            # the concrete storage later; this keeps plain ``train()`` cheap for
            # very large SNNs.
            if self.train_flags is not None or self.train_delay_backend == "dense":
                self._refresh_training_advance_after_diff_config(clear_histories=False)
        else:
            # Dense training and inference share the same delay-ring layout, so
            # switching back to evaluation can select the hard-event runtime
            # immediately without clearing queued deliveries or moving the
            # current ring slot.  Compact source-history/sparse runtimes may
            # require an explicit initialize() to reshape their storage; their
            # existing initialization path remains authoritative.
            incompatible_compact_runtime = (
                previous_advance == ("advance_diff_source_history")
                or self.delay_backend != "dense"
            )
            if previous_mode != bool(mode) and incompatible_compact_runtime:
                self._mode_requires_initialize = True
                self.advance = self._advance_requires_mode_initialize
                return self
            if previous_advance == "advance_diff":
                self._ensure_connection_spike_buffers()
                self.advance = self._dense_advance_target()
            # ``eval`` is the intended mode for the bitpacked backend.  Reclaim
            # per-connection debug/spike-state storage as soon as the module enters
            # inference mode.
            self._shrink_connection_spike_buffers_for_bitpack()
        self._mode_requires_initialize = False
        return self

    def eval(self):  # type: ignore[override]
        return self.train(False)

    def zero(self, clear_delivery_buffers=True):
        """
        Reset the per-step state of the connection.

        This method:

        * sets ``current_time_step`` to 0,
        * optionally clears the delivery buffer and, when event tracking is
          enabled, the event queue,
        * resets spike-history buffers (``has_spiked``, ``is_spiking``).

        Parameters
        ----------
        clear_delivery_buffers : bool, optional
            If True (default), zero the delivery buffer and reset spike
            histories. If False, keep existing contents of the delay line and
            optional event queue, but always reset ``current_time_step`` to 0.

        Notes
        -----
        * This does **not** clear schedules; use :meth:`clear_schedule` for
          that.
        * ``initialize`` calls this internally, so explicit calls are only
          needed when manually resetting state mid-simulation.
        """
        self.current_time_step.fill_(0)
        if clear_delivery_buffers:
            self.delivery_buffer.zero_()
            self._clear_sparse_calendar()
            if (
                hasattr(self, "spike_history_packed")
                and self.spike_history_packed.numel() > 0
            ):
                self.spike_history_packed.zero_()
            if (
                hasattr(self, "source_gate_history")
                and self.source_gate_history.numel() > 0
            ):
                self.source_gate_history = self.source_gate_history.detach()
                self.source_gate_history.zero_()
            if (
                hasattr(self, "bitpack_source_has_spiked")
                and self.bitpack_source_has_spiked.numel() > 0
            ):
                self.bitpack_source_has_spiked.zero_()
            if self.track_events and hasattr(self, "event_queue"):
                self.event_queue.zero_()
            self.events.zero_()
            self.has_spiked.fill_(False)
            self.is_spiking.zero_()  # float buffer, reset to 0.0

    def detach(self):
        """
        Detach internal buffers from the current computation graph.

        This simply calls ``.detach()`` on all registered buffers and rebinds
        them. It is useful when:

        * starting a fresh forward pass between episodes,
        * freezing the internal state for inference,
        * or avoiding backpropagation through previous simulation windows.

        Notes
        -----
        * This does not change the values of buffers; it only affects their
          ``grad_fn`` and autograd history.
        """
        for n, b in self.named_buffers():
            setattr(self, n, b.detach())

    def initialize(
        self, reinit_weights=True, reinit_delays=True, clear_deliveries=True
    ):
        """
        Prepare ``NetCon`` for simulation by initializing buffers and parameters.

        This method must be called before the first call to ``advance`` in a
        new simulation episode. It performs the following steps:

        1. **Select advance kernel**:

           - If ``self.training`` is True, set ``self.advance = self.advance_diff``.
           - Otherwise, set ``self.advance = self.advance_non_diff``.

        2. **Reset per-step state** by calling :meth:`zero` with
           ``clear_deliveries``.

        3. **(Re)initialize parameters**:

           - Call ``self.weight.init(reinit=reinit_weights)``.
           - Call ``self.delay_ms.init(reinit=reinit_delays)``.

        4. **Rebuild delay buffers** if delays were reinitialized via
           :meth:`_rebuild_delay_buffers`.

        5. **Align internal time**:

           - Set ``global_step`` from the external time attribute ``self.t``
             and ``self.dt`` (assumes ``self.t`` is managed by the broader
             simulation).

        6. **Detach state** by calling :meth:`detach`.

        Parameters
        ----------
        reinit_weights : bool, optional
            If True (default), reinitialize weights via the underlying
            parameter module. If False, leave existing weights unchanged.
        reinit_delays : bool, optional
            If True (default), reinitialize delays via the underlying
            parameter module and rebuild delay buffers. If False, keep
            existing delays and the current delay buffers.
        clear_deliveries : bool, optional
            If True (default), zero delivery buffers and spike histories via
            :meth:`zero`. If False, keep existing delivery-buffer contents
            while still resetting ``current_time_step`` to 0.

        Notes
        -----
        * For differentiable training, call :meth:`set_diff_config` before
          calling ``initialize`` so that :meth:`advance_diff` can use the
          correct configuration.
        * ``initialize`` does not clear scheduled events; call
          :meth:`clear_schedule` if you need a fresh schedule.
        * This method is invoked automatically by higher-level simulation
          loops in Dendra (e.g., :class:`dendra.models.networks.Network.initialize`).
        """
        if self.training:
            if self._use_source_history_training_runtime():
                self._ensure_source_history_training_storage(clear=False)
                self._shrink_connection_spike_buffers_for_source_history_training()
                self.advance = self.advance_diff_source_history
            else:
                self._ensure_connection_spike_buffers()
                self.advance = self.advance_diff
        elif self.delay_backend == "bitpacked_history":
            if not self._bitpack_can_use:
                raise ValueError(
                    "NetCon delay_backend='bitpacked_history' is not exact for this "
                    f"connection: {self._bitpack_ineligible_reason}."
                )
            self.advance = self.advance_non_diff_bitpacked_history
            self._shrink_connection_spike_buffers_for_bitpack()
        elif self.delay_backend == "sparse_calendar":
            self._ensure_connection_spike_buffers()
            self.advance = self.advance_non_diff_sparse_calendar
        else:
            self._ensure_connection_spike_buffers()
            self.advance = self._dense_advance_target()
        self._ensure_delivery_storage_for_current_mode(clear=False)
        self.zero(clear_delivery_buffers=clear_deliveries)
        self.weight.init(reinit=reinit_weights)
        self.delay_ms.init(reinit=reinit_delays)
        if reinit_delays:
            self._rebuild_delay_buffers()
            if self.training:
                if self._use_source_history_training_runtime():
                    self._ensure_source_history_training_storage(clear=True)
                    self._shrink_connection_spike_buffers_for_source_history_training()
                    self.advance = self.advance_diff_source_history
                else:
                    self._ensure_connection_spike_buffers()
                    self.advance = self.advance_diff
            elif self.delay_backend == "bitpacked_history":
                self.advance = self.advance_non_diff_bitpacked_history
                self._shrink_connection_spike_buffers_for_bitpack()
            elif self.delay_backend == "sparse_calendar":
                self._ensure_connection_spike_buffers()
                self.advance = self.advance_non_diff_sparse_calendar
            else:
                self._ensure_connection_spike_buffers()
                self.advance = self._dense_advance_target()
        with torch.no_grad():
            self.global_step.fill_(int(round(float(self.t) / float(self.dt))))
        self.detach()
        self._mode_requires_initialize = False

    def numel(self):
        """
        Return the number of synaptic connections managed by this ``NetCon``.

        Returns
        -------
        int
            Number of connections, equal to ``len(pre_idx)``. This is useful
            for:

            * sanity-checking connectivity sizes,
            * allocating auxiliary tensors,
            * reporting statistics about network scale.
        """
        return int(self.n.item())

    def state_cache(self):
        """Package runtime state for Network.cache_state()/steady_state().

        This cache is detached and normalized to a zero current_time_step.  It is
        intentionally distinct from state_dict_for_checkpoint(), which is used
        for BPTT/checkpoint replay and may need to preserve autograd history.
        """
        cur_slot = _cache_current_slot(self.current_time_step)
        cache: Dict[str, Any] = {
            "kind": "netcon",
            "version": 1,
            "backend": self.delay_backend,
            "training": bool(self.training),
            "dt": _cache_dt_value(self.dt),
            "max_delay_steps": int(self.max_delay_steps),
        }

        if not self.skip_thresholding and self.has_spiked.numel() > 0:
            cache["has_spiked"] = self.has_spiked.detach().clone()
        if self.is_spiking.numel() > 0:
            cache["is_spiking"] = self.is_spiking.detach().clone()

        if self._use_bitpacked_history_runtime():
            cache["backend_state"] = {
                "source_spike_history_packed": _ring_rows_to_age_rows(
                    self.spike_history_packed.detach(), cur_slot
                ),
                "history_layout": "age",
                "bitpack_source_has_spiked": self.bitpack_source_has_spiked.detach().clone(),
                # Explicit scheduled events that have already been converted into
                # delayed payloads live in the sparse side-calendar.  These are
                # already weighted, so they remain a compatibility fallback for
                # unusual explicit schedules; intrinsic spikes are restored from
                # the parameter-invariant source history above.
                "sparse_calendar": _clone_calendar_chunks(
                    self._sparse_calendar,
                    cur_slot=cur_slot,
                    depth=self.max_delay_steps,
                ),
            }
            cache["param_invariant"] = True
        elif (
            hasattr(self, "state_cache_gate_history")
            and self.state_cache_gate_history.numel() > 0
            and getattr(self, "_state_cache_recording_enabled", False)
        ):
            cache["backend_state"] = {
                "gate_history": _ring_rows_to_age_rows(
                    self.state_cache_gate_history.detach(), cur_slot
                ),
                "history_layout": "age",
            }
            cache["param_invariant"] = True
        elif self._use_sparse_calendar_runtime():
            state = {
                "sparse_calendar": _clone_calendar_chunks(
                    self._sparse_calendar,
                    cur_slot=cur_slot,
                    depth=self.max_delay_steps,
                )
            }
            if self.track_events:
                state["sparse_event_calendar"] = _clone_calendar_chunks(
                    self._sparse_event_calendar,
                    cur_slot=cur_slot,
                    depth=self.max_delay_steps,
                )
            cache["backend_state"] = state
            cache["param_invariant"] = False
        else:
            state = {
                "delivery_buffer": torch.roll(
                    self.delivery_buffer.detach(), -cur_slot, dims=0
                ).clone()
            }
            if self.track_events and hasattr(self, "event_queue"):
                state["event_queue"] = torch.roll(
                    self.event_queue.detach(), -cur_slot, dims=0
                ).clone()
            cache["backend_state"] = state
            cache["param_invariant"] = False

        return cache

    def initialize_from_state_cache(
        self, state_cache, *, dt=None, rebuild_delays: bool = True
    ):
        """Restore a cache produced by state_cache() after initialize().

        Network calls this after ``NetCon.initialize(...)`` has already refreshed
        ``WeightExpander.w`` and, when requested, rebuilt integer delay metadata.
        ``rebuild_delays`` remains True for direct calls so the method is
        self-contained; Network passes False to avoid a redundant second rebuild
        and peak-memory spike.
        """
        # Backward compatibility with the legacy Network._syn_cache tuple:
        # (old_dt, has_spiked, is_spiking, rolled_delivery_buffer).
        if isinstance(state_cache, tuple):
            old_dt, has_spiked, is_spiking, delivery_buffer = state_cache
            state_cache = {
                "kind": "netcon",
                "version": 0,
                "backend": "dense",
                "dt": float(old_dt),
                "has_spiked": has_spiked,
                "is_spiking": is_spiking,
                "backend_state": {"delivery_buffer": delivery_buffer},
            }

        if not isinstance(state_cache, dict):
            raise TypeError("NetCon state cache must be a dict or legacy tuple")
        cache_backend = state_cache.get("backend", "dense")
        if cache_backend != self.delay_backend:
            raise ValueError(
                "Cannot restore NetCon state cache from backend "
                f"{cache_backend!r} into backend {self.delay_backend!r}."
            )

        # Delay parameters may have changed since steady_state() was cached, for
        # example after an optimizer step.  The expanded delay buffer
        # ``delay_ms.w`` must already reflect the desired current parameters.
        # Direct callers can leave rebuild_delays=True; Network.initialize()
        # passes False because syn.initialize(reinit_delays=True) has already
        # refreshed metadata.
        if rebuild_delays:
            self._rebuild_delay_buffers()
        self._ensure_delivery_storage_for_current_mode(clear=True)

        old_dt = float(state_cache.get("dt", _cache_dt_value(self.dt)))
        new_dt = float(_cache_dt_value(self.dt) if dt is None else dt)
        backend_state = state_cache.get("backend_state", {})

        if "is_spiking" in state_cache and self.is_spiking.numel() > 0:
            isp = (
                state_cache["is_spiking"]
                .detach()
                .to(device=self.pre_device, dtype=self.pre_dtype)
            )
            if isp.numel() == self.is_spiking.numel():
                self.is_spiking.copy_(isp.reshape_as(self.is_spiking))
            else:
                self.is_spiking = isp.clone()

        # Preferred parameter-invariant cache formats.  These store the
        # presynaptic event/gate history, not already-weighted future payloads,
        # so pending deliveries are rebuilt using the *current* weights, delays,
        # and dt.
        if (
            "source_spike_history_packed" in backend_state
            or "spike_history_packed" in backend_state
        ):
            hist = backend_state.get(
                "source_spike_history_packed",
                backend_state.get("spike_history_packed", None),
            )
            if hist is None:
                raise KeyError("Bitpacked NetCon cache is missing source spike history")
            hist = hist.detach().to(device=self.device, dtype=torch.int64)
            layout = backend_state.get("history_layout", "ring_zero")
            if layout == "age":
                age_hist = _dilate_packed_history_age_rows(
                    hist,
                    old_dt,
                    new_dt,
                    n_limit=int(self.max_delay_steps),
                )
            else:
                # Backward-compatible best effort for caches produced by the
                # first steady-state refactor, where row 0 was the current ring
                # slot rather than explicit age 0.
                ring_hist = _dilate_packed_time_rows(
                    hist,
                    old_dt,
                    new_dt,
                    n_limit=int(self.max_delay_steps),
                )
                age_hist = _ring_rows_to_age_rows(ring_hist, 0)

            if (
                hasattr(self, "spike_history_packed")
                and self.spike_history_packed.numel() > 0
            ):
                if age_hist.shape[1] != self.spike_history_packed.shape[1]:
                    raise ValueError(
                        "Cached bitpacked spike history has incompatible word count: "
                        f"{age_hist.shape[1]} vs {self.spike_history_packed.shape[1]}."
                    )
                ring_hist = _age_rows_to_ring_rows(
                    age_hist,
                    depth=int(self.spike_history_packed.shape[0]),
                )
                self.spike_history_packed.copy_(ring_hist)

            if self._use_bitpacked_history_runtime():
                self.delivery_buffer.zero_()
                self._sparse_calendar = _restore_calendar_chunks(
                    backend_state.get("sparse_calendar", {}),
                    old_dt=old_dt,
                    new_dt=new_dt,
                    depth=self.max_delay_steps,
                    idx_device=self.device,
                    idx_dtype=torch.long,
                    val_device=self.device,
                    val_dtype=self.dtype,
                )
                self._sparse_event_calendar.clear()
            elif self._use_source_history_training_runtime():
                # Source-history training has only a one-row receive scratch.
                # Restore compact source history rather than materializing future
                # deliveries into a dense delay ring.
                self._restore_source_history_training_from_packed_age_history(age_hist)
            else:
                # Training/dense runtime: materialize an ordinary future delivery
                # buffer from the source history and current weights/delays.
                self._materialize_delivery_from_packed_source_history(age_hist)

        elif "gate_history" in backend_state:
            gate_hist = (
                backend_state["gate_history"]
                .detach()
                .to(device=self.device, dtype=self.dtype)
            )
            gate_hist = _dilate_history_age_rows(
                gate_hist,
                old_dt,
                new_dt,
                n_limit=int(self.max_delay_steps),
            )
            if self._use_source_history_training_runtime():
                self._restore_source_history_training_from_gate_history(gate_hist)
            else:
                self._materialize_delivery_from_gate_history(gate_hist)

        elif self._use_sparse_calendar_runtime():
            self.delivery_buffer.zero_()
            self._sparse_calendar = _restore_calendar_chunks(
                backend_state.get("sparse_calendar", {}),
                old_dt=old_dt,
                new_dt=new_dt,
                depth=self.max_delay_steps,
                idx_device=self.device,
                idx_dtype=torch.long,
                val_device=self.device,
                val_dtype=self.dtype,
            )
            if self.track_events:
                self._sparse_event_calendar = _restore_calendar_chunks(
                    backend_state.get("sparse_event_calendar", {}),
                    old_dt=old_dt,
                    new_dt=new_dt,
                    depth=self.max_delay_steps,
                    idx_device=self.device,
                    idx_dtype=torch.long,
                    val_device=self.device,
                    val_dtype=torch.int32,
                )
            else:
                self._sparse_event_calendar.clear()
        else:
            # Legacy/compatibility path: restore already-weighted future payloads.
            # This is dt-adjustable, but it is tied to the weights/delays that
            # produced the cache.
            cached = backend_state.get("delivery_buffer", None)
            if cached is None:
                raise KeyError("Dense NetCon cache is missing delivery_buffer")
            cached = cached.detach().to(device=self.device, dtype=self.dtype)
            if cached.ndim != 2 or cached.shape[1] != self._syn_numel:
                raise ValueError(
                    "Cached NetCon delivery_buffer has incompatible shape: "
                    f"{tuple(cached.shape)} vs (*, {self._syn_numel})."
                )
            restored = _dilate_time_rows(
                cached,
                old_dt,
                new_dt,
                n_limit=int(self.delivery_buffer.shape[0]),
            )
            self.delivery_buffer.copy_(restored)
            if self.track_events and hasattr(self, "event_queue"):
                ev = backend_state.get("event_queue", None)
                if ev is not None:
                    ev = ev.detach().to(device=self.device, dtype=torch.int32)
                    ev_restored = _dilate_time_rows(
                        ev,
                        old_dt,
                        new_dt,
                        n_limit=int(self.event_queue.shape[0]),
                    )
                    self.event_queue.copy_(ev_restored)

        # Reapply threshold latches only after reconstructing runtime history.
        # Compact-history allocation clears these latches as part of its normal
        # fresh-episode setup; restoring them earlier would spuriously re-arm a
        # source that was already above threshold at the cache boundary.
        if "has_spiked" in state_cache:
            hs = (
                state_cache["has_spiked"]
                .detach()
                .to(device=self.pre_device, dtype=torch.bool)
            )
            self._restore_source_has_spiked_from_connection_cache(hs)
            if self.has_spiked.numel() > 0:
                if hs.numel() == self.has_spiked.numel():
                    self.has_spiked.copy_(hs.reshape_as(self.has_spiked))
                else:
                    self.has_spiked = hs.clone()

        if (
            "bitpack_source_has_spiked" in backend_state
            and hasattr(self, "bitpack_source_has_spiked")
            and self.bitpack_source_has_spiked.numel() > 0
        ):
            src_hs = (
                backend_state["bitpack_source_has_spiked"]
                .detach()
                .to(device=self.pre_device, dtype=torch.bool)
            )
            self.bitpack_source_has_spiked.copy_(
                src_hs.reshape_as(self.bitpack_source_has_spiked)
            )
            if self.has_spiked.numel() == self._n_conn and src_hs.numel() > 0:
                source_pos = torch.searchsorted(
                    self.bitpack_source_pre_idx.to(
                        device=self.pre_device, dtype=torch.long
                    ),
                    self.pre_idx.to(device=self.pre_device, dtype=torch.long),
                )
                self.has_spiked.copy_(src_hs.index_select(0, source_pos))

        self.current_time_step.zero_()
        if hasattr(self, "t"):
            self.global_step.fill_(int(round(float(self.t) / float(self.dt))))
        else:
            self.global_step.zero_()
        self.detach()
        return self

    def state_dict_for_checkpoint(self):
        """
        Return a state dictionary suitable for gradient checkpointing / long-run chunking.

        This should include ONLY mutable tensors that affect future dynamics and are
        mutated by advance_*(). The returned structure must be stable across chunks
        (same keys; same shapes).

        Notes
        -----
        - This is *not* a full PyTorch state_dict. It is the *minimal mutable*
          runtime state needed to resume stepping identically mid-simulation.
        - Do NOT include weights/delays/modules here; those are model params
          and remain constant across a forward.
        - Tracked event counts are public runtime state. ``events`` records the
          most recently delivered counts, while the dense ``event_queue`` (or a
          compact backend's event calendar) determines future observations; both
          are retained so checkpoint replay preserves event introspection.
        """

        if self._mode_requires_initialize:
            raise RuntimeError(
                "NetCon train/eval mode changed across incompatible runtime "
                "layouts. Call initialize() before checkpointing this connection."
            )

        runtime = self._checkpoint_runtime_kind()
        compact_runtime = runtime != "dense"
        sd: Dict[str, Any] = {
            "delivery_buffer": (
                self.delivery_buffer.clone()
                if compact_runtime
                else self.delivery_buffer
            ),
            "current_time_step": (
                self.current_time_step.clone()
                if compact_runtime
                else self.current_time_step
            ),
            "global_step": (
                self.global_step.clone() if compact_runtime else self.global_step
            ),
        }

        # Threshold-crossing history matters for spike detection when thresholds are used.
        if not self.skip_thresholding and self.has_spiked.numel() > 0:
            sd["has_spiked"] = (
                self.has_spiked.clone() if compact_runtime else self.has_spiked
            )

        if self.track_events:
            sd["events"] = self.events.clone()
            if hasattr(self, "event_queue"):
                sd["event_queue"] = self.event_queue.clone()

        if compact_runtime:
            backend_state: Dict[str, Any] = {
                "kind": runtime,
                "sparse_calendar": _checkpoint_calendar_chunks(self._sparse_calendar),
                "sparse_event_calendar": _checkpoint_calendar_chunks(
                    self._sparse_event_calendar
                ),
            }
            if self.spike_history_packed.numel() > 0:
                backend_state["spike_history_packed"] = (
                    self.spike_history_packed.clone()
                )
            if self.source_gate_history.numel() > 0:
                backend_state["source_gate_history"] = self.source_gate_history.clone()
            if self.bitpack_source_has_spiked.numel() > 0:
                backend_state["bitpack_source_has_spiked"] = (
                    self.bitpack_source_has_spiked.clone()
                )
            sd["backend_state"] = backend_state

        return sd

    def restore_dict_from_checkpoint(self, state_dict):
        """
        Restore the state of this NetCon from a checkpoint dictionary created by
        state_dict_for_checkpoint().

        Important checkpointing detail:
        - delivery_buffer must be rebound directly (to preserve gradient flow).
        - non-differentiable tensors that NetCon mutates in-place should be cloned
        here to avoid in-place mutation of checkpoint *inputs* inside a chunk.

        This matches the expectation in longrun_checkpointed that restore rebinds
        mutable tensors.
        """
        if not isinstance(state_dict, dict):
            raise TypeError("NetCon checkpoint state must be a dict.")

        runtime = self._checkpoint_runtime_kind()
        backend_state = state_dict.get("backend_state", None)
        if backend_state is not None and not isinstance(backend_state, dict):
            raise TypeError("NetCon checkpoint backend_state must be a dict.")
        checkpoint_runtime = (
            "dense" if backend_state is None else backend_state.get("kind", None)
        )
        if checkpoint_runtime != runtime:
            raise ValueError(
                "NetCon checkpoint runtime does not match the active runtime: "
                f"checkpoint={checkpoint_runtime!r}, active={runtime!r}."
            )

        delivery = _require_checkpoint_tensor(
            state_dict,
            "delivery_buffer",
            shape=self.delivery_buffer.shape,
            dtype=self.dtype,
        )
        current = _require_checkpoint_tensor(
            state_dict,
            "current_time_step",
            shape=self.current_time_step.shape,
            dtype=torch.long,
        )
        global_step = _require_checkpoint_tensor(
            state_dict,
            "global_step",
            shape=self.global_step.shape,
            dtype=torch.long,
        )
        current_value = int(current.detach().cpu().reshape(-1)[0].item())
        if current_value < 0 or current_value >= int(self.max_delay_steps):
            raise ValueError(
                "NetCon checkpoint current_time_step is outside the delay ring: "
                f"{current_value} not in [0, {int(self.max_delay_steps)})."
            )

        has_spiked = None
        if not self.skip_thresholding and self.has_spiked.numel() > 0:
            has_spiked = _require_checkpoint_tensor(
                state_dict,
                "has_spiked",
                shape=self.has_spiked.shape,
                dtype=torch.bool,
            )

        events = None
        event_queue = None
        if self.track_events:
            events = _require_checkpoint_tensor(
                state_dict,
                "events",
                shape=self.events.shape,
                dtype=torch.int32,
            )
            if hasattr(self, "event_queue"):
                event_queue = _require_checkpoint_tensor(
                    state_dict,
                    "event_queue",
                    shape=self.event_queue.shape,
                    dtype=torch.int32,
                )

        sparse_calendar = None
        sparse_event_calendar = None
        spike_history = None
        source_gate_history = None
        source_has_spiked = None
        if backend_state is not None:
            sparse_calendar = _restore_checkpoint_calendar_chunks(
                backend_state.get("sparse_calendar", {}),
                idx_device=self.device,
                idx_dtype=torch.long,
                value_device=self.device,
                value_dtype=self.dtype,
            )
            sparse_event_calendar = _restore_checkpoint_calendar_chunks(
                backend_state.get("sparse_event_calendar", {}),
                idx_device=self.device,
                idx_dtype=torch.long,
                value_device=self.device,
                value_dtype=torch.int32,
            )
            invalid_slots = {
                slot
                for slot in (*sparse_calendar, *sparse_event_calendar)
                if slot < 0 or slot >= int(self.max_delay_steps)
            }
            if invalid_slots:
                raise ValueError(
                    "NetCon checkpoint calendar contains slots outside the delay "
                    f"ring: {sorted(invalid_slots)}."
                )
            if self.spike_history_packed.numel() > 0:
                spike_history = _require_checkpoint_tensor(
                    backend_state,
                    "spike_history_packed",
                    shape=self.spike_history_packed.shape,
                    dtype=torch.int64,
                )
            if self.source_gate_history.numel() > 0:
                source_gate_history = _require_checkpoint_tensor(
                    backend_state,
                    "source_gate_history",
                    shape=self.source_gate_history.shape,
                    dtype=self.dtype,
                )
            if self.bitpack_source_has_spiked.numel() > 0:
                source_has_spiked = _require_checkpoint_tensor(
                    backend_state,
                    "bitpack_source_has_spiked",
                    shape=self.bitpack_source_has_spiked.shape,
                    dtype=torch.bool,
                )

        # Restore by rebinding differentiable tensors so graph connectivity is
        # preserved across checkpoint chunks. Integer/bool history is cloned
        # because the next step mutates it in-place.
        self.delivery_buffer = delivery.to(device=self.device)
        self.current_time_step = current.to(device=self.device)
        self.global_step = global_step.to(device=self.device)
        if has_spiked is not None:
            self.has_spiked = has_spiked.to(device=self.pre_device)
        if events is not None:
            self.events = events.to(device=self.device).clone()
        if event_queue is not None:
            self.event_queue = event_queue.to(device=self.device).clone()
        if backend_state is not None:
            self._sparse_calendar = sparse_calendar
            self._sparse_event_calendar = sparse_event_calendar
            if spike_history is not None:
                self.spike_history_packed = spike_history.to(device=self.device).clone()
            if source_gate_history is not None:
                self.source_gate_history = source_gate_history.to(device=self.device)
            if source_has_spiked is not None:
                self.bitpack_source_has_spiked = source_has_spiked.to(
                    device=self.pre_device
                ).clone()
        return self

    def _checkpoint_runtime_kind(self) -> str:
        if self._use_sparse_calendar_runtime():
            return "sparse_calendar"
        if self._use_bitpacked_history_runtime():
            return "bitpacked_history"
        if self._use_source_history_training_runtime():
            return "source_history"
        return "dense"

    def checkpoint_topology_signature(self):
        """Return immutable structure needed to validate fresh-object replay."""
        train_flags = None
        if self.train_flags is not None:
            train_flags = tuple(self.train_flags)
        return {
            "kind": "event",
            "n_connections": int(self._n_conn),
            "synapse_numel": int(self._syn_numel),
            "max_delay_steps": int(self.max_delay_steps),
            "delay_backend": self.delay_backend,
            "train_delay_backend": self.train_delay_backend,
            "runtime": self._checkpoint_runtime_kind(),
            "training": bool(self.training),
            "track_events": bool(self.track_events),
            "train_flags": train_flags,
            "topology": _topology_tensor_digest(
                self.pre_idx,
                self.post_idx,
                self.threshold,
                self.thresh_is_nan,
            ),
        }
