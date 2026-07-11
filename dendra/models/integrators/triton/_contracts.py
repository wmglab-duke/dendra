"""Synchronization-free contracts for public Triton solver wrappers."""

from __future__ import annotations

from collections.abc import Iterable

import torch

_FLOAT_DTYPES = (torch.float32, torch.float64)
_INDEX_DTYPES = (torch.int32, torch.int64)
_WARP_SIZE = 32


def _require_tensors(named_tensors: Iterable[tuple[str, object]]) -> None:
    for name, tensor in named_tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")


def _validate_float_group(named_tensors: tuple[tuple[str, torch.Tensor], ...]) -> None:
    first_name, first = named_tensors[0]
    if first.device.type != "cuda":
        raise ValueError("Triton solver inputs must be CUDA tensors")
    if first.dtype not in _FLOAT_DTYPES:
        raise TypeError("Triton solvers support only float32 and float64")
    for name, tensor in named_tensors[1:]:
        if tensor.device != first.device:
            raise ValueError(
                f"{name} must be on the same device as {first_name}; "
                f"got {tensor.device} and {first.device}"
            )
        if tensor.dtype != first.dtype:
            raise ValueError(
                f"{name} must have the same dtype as {first_name}; "
                f"got {tensor.dtype} and {first.dtype}"
            )


def _validate_index_group(
    named_tensors: tuple[tuple[str, torch.Tensor], ...],
    *,
    device: torch.device,
) -> None:
    for name, tensor in named_tensors:
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be 1-D")
        if tensor.dtype not in _INDEX_DTYPES:
            raise TypeError(f"{name} must have dtype int32 or int64")
        if tensor.device != device:
            raise ValueError(f"{name} must be on device {device}; got {tensor.device}")


def validate_threads(threads: int) -> int:
    """Validate the sub-warp lane count assumed by DHS kernels."""
    if isinstance(threads, bool) or not isinstance(threads, int):
        raise TypeError("threads must be a positive integer that divides 32")
    if threads <= 0 or threads > _WARP_SIZE:
        raise ValueError("threads must be in [1, 32]")
    if _WARP_SIZE % threads != 0:
        raise ValueError("threads must divide 32 (warp size)")
    return threads


def validate_tridiagonal(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> tuple[int, int]:
    """Validate a batched scalar tridiagonal solve."""
    named = (("a", a), ("b", b), ("c", c), ("d", d))
    _require_tensors(named)
    if b.ndim != 2:
        raise ValueError(f"b must have shape (B, K), got {tuple(b.shape)}")
    batch, size = map(int, b.shape)
    if batch < 1 or size < 1:
        raise ValueError("B and K must both be >= 1")
    expected_band = (batch, size - 1)
    if a.shape != expected_band:
        raise ValueError(f"a must have shape {expected_band}, got {tuple(a.shape)}")
    if c.shape != expected_band:
        raise ValueError(f"c must have shape {expected_band}, got {tuple(c.shape)}")
    if d.shape != (batch, size):
        raise ValueError(f"d must have shape {(batch, size)}, got {tuple(d.shape)}")
    _validate_float_group(named)
    return batch, size


def validate_block_tridiagonal(
    lower: torch.Tensor,
    main: torch.Tensor,
    upper: torch.Tensor,
    rhs: torch.Tensor,
) -> tuple[int, int]:
    """Validate a batched tridiagonal system with 3x3 diagonal blocks."""
    named = (
        ("lower", lower),
        ("main", main),
        ("upper", upper),
        ("rhs", rhs),
    )
    _require_tensors(named)
    if rhs.ndim != 3 or rhs.shape[-1] != 3:
        raise ValueError(f"rhs must have shape (B, K, 3), got {tuple(rhs.shape)}")
    batch, size, _ = map(int, rhs.shape)
    if batch < 1 or size < 1:
        raise ValueError("B and K must both be >= 1")
    expected_band = (batch, size - 1, 3)
    if lower.shape != expected_band:
        raise ValueError(
            f"lower must have shape {expected_band}, got {tuple(lower.shape)}"
        )
    if upper.shape != expected_band:
        raise ValueError(
            f"upper must have shape {expected_band}, got {tuple(upper.shape)}"
        )
    expected_main = (batch, size, 3, 3)
    if main.shape != expected_main:
        raise ValueError(
            f"main must have shape {expected_main}, got {tuple(main.shape)}"
        )
    _validate_float_group(named)
    return batch, size


def validate_tree(
    d_mem: torch.Tensor,
    a_geom: torch.Tensor,
    b: torch.Tensor,
    parent_idx: torch.Tensor,
    order: torch.Tensor,
    layer_ptr: torch.Tensor,
    threads: int,
) -> tuple[int, int, int]:
    """Validate a scalar tree solve without reading topology values."""
    data = (("d_mem", d_mem), ("a_geom", a_geom), ("b", b))
    topology = (
        ("parent_idx", parent_idx),
        ("order", order),
        ("layer_ptr", layer_ptr),
    )
    _require_tensors(data + topology)
    if d_mem.ndim != 2:
        raise ValueError(f"d_mem must have shape (B, K), got {tuple(d_mem.shape)}")
    batch, size = map(int, d_mem.shape)
    if batch < 1 or size < 1:
        raise ValueError("B and K must both be >= 1")
    if a_geom.shape != (batch, size) or b.shape != (batch, size):
        raise ValueError("d_mem, a_geom, and b must have identical (B, K) shapes")
    _validate_float_group(data)
    _validate_index_group(topology, device=d_mem.device)
    if parent_idx.numel() != size or order.numel() != size:
        raise ValueError("parent_idx and order must each contain K entries")
    if layer_ptr.numel() < 2:
        raise ValueError("layer_ptr must contain at least two entries")
    return batch, size, validate_threads(threads)


def validate_tree_block(
    main: torch.Tensor,
    edge: torch.Tensor,
    rhs: torch.Tensor,
    parent_idx: torch.Tensor,
    order: torch.Tensor,
    layer_ptr: torch.Tensor,
    threads: int,
) -> tuple[int, int, int]:
    """Validate a 3-component tree solve without reading topology values."""
    data = (("main", main), ("edge", edge), ("rhs", rhs))
    topology = (
        ("parent_idx", parent_idx),
        ("order", order),
        ("layer_ptr", layer_ptr),
    )
    _require_tensors(data + topology)
    if rhs.ndim != 3 or rhs.shape[-1] != 3:
        raise ValueError(f"rhs must have shape (B, K, 3), got {tuple(rhs.shape)}")
    batch, size, _ = map(int, rhs.shape)
    if batch < 1 or size < 1:
        raise ValueError("B and K must both be >= 1")
    if edge.shape != (batch, size, 3):
        raise ValueError(
            f"edge must have shape {(batch, size, 3)}, got {tuple(edge.shape)}"
        )
    if main.shape != (batch, size, 3, 3):
        raise ValueError(
            f"main must have shape {(batch, size, 3, 3)}, got {tuple(main.shape)}"
        )
    _validate_float_group(data)
    _validate_index_group(topology, device=rhs.device)
    if parent_idx.numel() != size or order.numel() != size:
        raise ValueError("parent_idx and order must each contain K entries")
    if layer_ptr.numel() < 2:
        raise ValueError("layer_ptr must contain at least two entries")
    return batch, size, validate_threads(threads)


def validate_tree_multi(
    d_mem: torch.Tensor,
    a_geom: torch.Tensor,
    b: torch.Tensor,
    p_cat: torch.Tensor,
    order_cat: torch.Tensor,
    layer_ptr_cat: torch.Tensor,
    warp_p_off: torch.Tensor,
    warp_order_off: torch.Tensor,
    warp_lptr_off: torch.Tensor,
    warp_l: torch.Tensor,
    warp_row_base: torch.Tensor,
    warp_row_count: torch.Tensor,
    k_stride: int,
    l_max: int,
    threads: int,
    grid_x: int | None,
) -> tuple[int, int]:
    """Validate the row-padded, multi-morphology DHS launch plan."""
    data = (("d_mem", d_mem), ("a_geom", a_geom), ("b", b))
    plan = (
        ("P_cat", p_cat),
        ("ORDER_cat", order_cat),
        ("LAYER_PTR_cat", layer_ptr_cat),
        ("WARP_P_OFF", warp_p_off),
        ("WARP_ORDER_OFF", warp_order_off),
        ("WARP_LPTR_OFF", warp_lptr_off),
        ("WARP_L", warp_l),
        ("WARP_ROW_BASE", warp_row_base),
        ("WARP_ROW_COUNT", warp_row_count),
    )
    _require_tensors(data + plan)
    if d_mem.ndim != 2:
        raise ValueError(
            f"d_mem must have shape (B, K_stride), got {tuple(d_mem.shape)}"
        )
    if a_geom.shape != d_mem.shape or b.shape != d_mem.shape:
        raise ValueError("d_mem, a_geom, and b must have identical shapes")
    batch, width = map(int, d_mem.shape)
    if batch < 1 or width < 1:
        raise ValueError("B and K_stride must both be >= 1")
    if isinstance(k_stride, bool) or not isinstance(k_stride, int):
        raise TypeError("K_stride must be an integer")
    if k_stride != width:
        raise ValueError(f"K_stride must equal the padded row width {width}")
    if isinstance(l_max, bool) or not isinstance(l_max, int):
        raise TypeError("L_max must be an integer")
    if l_max < 1:
        raise ValueError("L_max must be >= 1")

    _validate_float_group(data)
    _validate_index_group(plan, device=d_mem.device)
    if p_cat.numel() != order_cat.numel() or p_cat.numel() < 1:
        raise ValueError("P_cat and ORDER_cat must have the same non-zero length")
    if layer_ptr_cat.numel() < 2:
        raise ValueError("LAYER_PTR_cat must contain at least two entries")

    warp_count = int(warp_row_base.numel())
    if warp_count < 1:
        raise ValueError("the launch plan must contain at least one warp")
    for name, tensor in plan[3:]:
        if tensor.numel() != warp_count:
            raise ValueError(f"{name} must contain one entry per warp ({warp_count})")
    if grid_x is not None:
        if isinstance(grid_x, bool) or not isinstance(grid_x, int):
            raise TypeError("grid_x must be an integer or None")
        if grid_x != warp_count:
            raise ValueError(f"grid_x must equal the plan warp count {warp_count}")

    validate_threads(threads)
    return batch, warp_count


def adjoint_main_blocks(main: torch.Tensor) -> torch.Tensor:
    """Return contiguous block transposes for an adjoint tree solve."""
    return main.transpose(-1, -2).contiguous()


def copy_rhs_workspace(rhs: torch.Tensor) -> torch.Tensor:
    """Return an independent contiguous RHS for an in-place kernel."""
    return rhs.clone(memory_format=torch.contiguous_format)
