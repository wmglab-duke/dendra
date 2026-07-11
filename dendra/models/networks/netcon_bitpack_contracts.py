"""Structural contracts shared by NetCon accelerator backends.

These checks intentionally avoid reading tensor values.  They therefore run
without synchronizing a CUDA stream and are cheap enough to keep at the public
kernel boundary.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

_BITS_PER_WORD = 63
_DELIVERY_FLOAT_DTYPES = (torch.float32, torch.float64)


def _require_tensors(named_tensors: Iterable[tuple[str, object]]) -> None:
    for name, tensor in named_tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")


def _require_same_device(named_tensors: tuple[tuple[str, torch.Tensor], ...]) -> None:
    first_name, first = named_tensors[0]
    for name, tensor in named_tensors[1:]:
        if tensor.device != first.device:
            raise ValueError(
                f"{name} must be on the same device as {first_name}; "
                f"got {tensor.device} and {first.device}"
            )


def validate_pack_structure(
    source_spikes: torch.Tensor,
    packed_history: torch.Tensor,
    current_time_step: torch.Tensor,
) -> tuple[int, int]:
    """Validate packing metadata without launching or synchronizing a device."""
    named = (
        ("source_spikes", source_spikes),
        ("packed_history", packed_history),
        ("current_time_step", current_time_step),
    )
    _require_tensors(named)
    if source_spikes.dtype != torch.bool:
        raise TypeError("source_spikes must be torch.bool")
    if packed_history.dtype != torch.int64:
        raise TypeError("packed_history must be torch.int64")
    if current_time_step.dtype != torch.int64 or current_time_step.numel() != 1:
        raise TypeError("current_time_step must be a scalar torch.long tensor")
    if source_spikes.ndim != 1:
        raise ValueError("source_spikes must be 1-D")
    if packed_history.ndim != 2:
        raise ValueError("packed_history must be 2-D")

    n_source = int(source_spikes.numel())
    n_words = int(packed_history.shape[1])
    expected_words = (n_source + _BITS_PER_WORD - 1) // _BITS_PER_WORD
    if n_words != expected_words:
        raise ValueError(
            "packed_history has the wrong word width: "
            f"expected {expected_words}, got {n_words}"
        )
    if n_words > 0 and packed_history.shape[0] < 1:
        raise ValueError("packed_history must contain at least one history row")

    _require_same_device(named)
    return n_source, n_words


def validate_delivery_structure(
    packed_history: torch.Tensor,
    current_time_step: torch.Tensor,
    delay_steps: torch.Tensor,
    conn_word_idx: torch.Tensor,
    conn_bit_mask: torch.Tensor,
    post_idx: torch.Tensor,
    weight: torch.Tensor,
    delivery_out: torch.Tensor,
) -> tuple[int, int, int]:
    """Validate delivery tensor structure without inspecting device values."""
    named = (
        ("packed_history", packed_history),
        ("current_time_step", current_time_step),
        ("delay_steps", delay_steps),
        ("conn_word_idx", conn_word_idx),
        ("conn_bit_mask", conn_bit_mask),
        ("post_idx", post_idx),
        ("weight", weight),
        ("delivery_out", delivery_out),
    )
    _require_tensors(named)
    n_conn = int(delay_steps.numel())
    return _validate_delivery_common(named, n_conn)


def validate_delivery_uniform_structure(
    packed_history: torch.Tensor,
    current_time_step: torch.Tensor,
    conn_word_idx: torch.Tensor,
    conn_bit_mask: torch.Tensor,
    post_idx: torch.Tensor,
    weight: torch.Tensor,
    delivery_out: torch.Tensor,
) -> tuple[int, int, int]:
    """Validate the uniform-delay delivery variant without a device sync."""
    named = (
        ("packed_history", packed_history),
        ("current_time_step", current_time_step),
        ("conn_word_idx", conn_word_idx),
        ("conn_bit_mask", conn_bit_mask),
        ("post_idx", post_idx),
        ("weight", weight),
        ("delivery_out", delivery_out),
    )
    _require_tensors(named)
    n_conn = int(conn_word_idx.numel())
    return _validate_delivery_common(named, n_conn)


def _validate_delivery_common(
    named: tuple[tuple[str, torch.Tensor], ...], n_conn: int
) -> tuple[int, int, int]:
    packed_history = named[0][1]
    current_time_step = named[1][1]
    integer_metadata = named[2:-2]
    weight = named[-2][1]
    delivery_out = named[-1][1]

    if packed_history.dtype != torch.int64:
        raise TypeError("packed_history must be torch.int64")
    if current_time_step.dtype != torch.int64 or current_time_step.numel() != 1:
        raise TypeError("current_time_step must be a scalar torch.long tensor")
    for name, tensor in integer_metadata:
        if tensor.dtype != torch.int64:
            raise TypeError(f"{name} must be torch.long")
    if weight.dtype != delivery_out.dtype:
        raise TypeError("weight and delivery_out must have the same dtype")
    if weight.dtype not in _DELIVERY_FLOAT_DTYPES:
        raise TypeError("weight and delivery_out must be float32 or float64")
    if packed_history.ndim != 2:
        raise ValueError("packed_history must be 2-D")

    for name, tensor in named[2:-1]:
        if tensor.numel() != n_conn:
            raise ValueError(
                f"{name} must contain {n_conn} values to match delay_steps; "
                f"got {tensor.numel()}"
            )

    max_delay_steps, n_words = map(int, packed_history.shape)
    if n_conn > 0 and (max_delay_steps < 1 or n_words < 1):
        raise ValueError(
            "packed_history must have non-empty row and word dimensions "
            "when connections are present"
        )

    _require_same_device(named)
    return n_conn, max_delay_steps, n_words
