"""Experimental Triton kernels for NetCon bitpacked source-spike history.

Normal ``NetCon`` operation does not dispatch to this module. Eligible CUDA
workloads first try the lazy C++/CUDA extension in ``netcon_bitpack_ops`` and
fall back to PyTorch when it is unavailable or a launch fails. This module is
retained as an opt-in alternative and reference implementation; callers must
invoke it directly. Importing Dendra does not require Triton, and these kernels
are defined lazily on first direct use.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .netcon_bitpack_contracts import (
    validate_delivery_structure,
    validate_pack_structure,
)

_BITS_PER_WORD = 63
_PACK_BLOCK_BITS = 64
_DELIVERY_BLOCK = 256

_triton_mod = None
_tl_mod = None
_kernels: Optional[Tuple[object, object]] = None
_last_error: Optional[BaseException] = None
_disabled = False


def _set_error(exc: BaseException) -> None:
    global _last_error
    _last_error = exc


def _disable(exc: BaseException) -> None:
    global _disabled, _last_error
    _disabled = True
    _last_error = exc


def _define_kernels():
    """Import Triton and define JIT kernels on demand."""
    global _triton_mod, _tl_mod, _kernels

    if _kernels is not None:
        return _kernels
    if _disabled:
        return None
    if not torch.cuda.is_available():
        _set_error(RuntimeError("CUDA is not available"))
        return None

    try:
        import triton
        import triton.language as tl
    except BaseException as exc:  # pragma: no cover - depends on install
        _set_error(exc)
        return None

    @triton.jit
    def _pack_source_spikes_kernel(
        source_spikes,
        packed_history,
        current_time_step,
        n_source: tl.constexpr,
        n_words: tl.constexpr,
        BITS_PER_WORD: tl.constexpr,
        BLOCK_BITS: tl.constexpr,
    ):
        # One Triton program owns one packed word.  This avoids atomicOr and
        # writes exactly one int64 word for the current history slot.
        word = tl.program_id(0)
        lane = tl.arange(0, BLOCK_BITS)
        src = word * BITS_PER_WORD + lane
        valid = (lane < BITS_PER_WORD) & (src < n_source)

        # Avoid forming 1 << 63.  The sign bit is intentionally unused, so lane
        # 63 is masked out and shifted as lane 62 only to keep the expression in
        # range for signed int64 lowering.
        safe_lane = tl.minimum(lane, BITS_PER_WORD - 1)
        one = tl.full((BLOCK_BITS,), 1, tl.int64)
        masks = one << safe_lane

        spikes = tl.load(source_spikes + src, mask=valid, other=0).to(tl.int64)
        values = tl.where((spikes != 0) & valid, masks, 0)
        packed = tl.sum(values, axis=0)

        slot = tl.load(current_time_step)
        tl.store(packed_history + slot * n_words + word, packed)

    @triton.jit
    def _build_delivery_kernel(
        packed_history,
        current_time_step,
        delay_steps,
        conn_word_idx,
        conn_bit_mask,
        post_idx,
        weight,
        delivery_out,
        n_conn: tl.constexpr,
        max_delay_steps: tl.constexpr,
        n_words: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # One program handles a block of connections.  For each edge, read the
        # relevant delayed source bit and atomically accumulate weight to the
        # dense postsynaptic receive scratch.
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_conn

        cur = tl.load(current_time_step)
        delay = tl.load(delay_steps + offsets, mask=mask, other=0)
        row = cur - delay
        # delay_steps are clamped to [1, max_delay_steps - 1] before this kernel,
        # so row is at worst -(max_delay_steps - 1).  One add is enough.
        row += tl.where(row < 0, max_delay_steps, 0)

        word = tl.load(conn_word_idx + offsets, mask=mask, other=0)
        bit_mask = tl.load(conn_bit_mask + offsets, mask=mask, other=0).to(tl.int64)
        packed = tl.load(
            packed_history + row * n_words + word,
            mask=mask,
            other=0,
        ).to(tl.int64)
        active = (packed & bit_mask) != 0

        dst = tl.load(post_idx + offsets, mask=mask, other=0)
        val = tl.load(weight + offsets, mask=mask, other=0.0)
        tl.atomic_add(delivery_out + dst, val, sem="relaxed", mask=mask & active)

    _triton_mod = triton
    _tl_mod = tl
    _kernels = (_pack_source_spikes_kernel, _build_delivery_kernel)
    return _kernels


def is_available() -> bool:
    """Return True when Triton kernels can be attempted in this process."""
    return _define_kernels() is not None


def last_error() -> Optional[BaseException]:
    """Return the most recent import/launch error, if any."""
    return _last_error


def _require_cuda_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise RuntimeError(f"{name} must be contiguous")


def pack_source_spikes(
    source_spikes: torch.Tensor,
    packed_history: torch.Tensor,
    current_time_step: torch.Tensor,
) -> None:
    """Pack one bool source-spike vector into the current int64 history row.

    Parameters
    ----------
    source_spikes:
        CUDA bool tensor of shape ``[n_source]``.
    packed_history:
        CUDA int64 tensor of shape ``[max_delay_steps, ceil(n_source / 63)]``.
    current_time_step:
        CUDA int64 scalar/length-one tensor containing the ring slot to write.
    """
    n_source, n_words = validate_pack_structure(
        source_spikes, packed_history, current_time_step
    )
    for name, tensor in (
        ("source_spikes", source_spikes),
        ("packed_history", packed_history),
        ("current_time_step", current_time_step),
    ):
        _require_cuda_contiguous(name, tensor)
    if n_words == 0:
        return

    kernels = _define_kernels()
    if kernels is None:
        raise RuntimeError("NetCon bitpack Triton kernels are not available")

    pack_kernel, _ = kernels
    try:
        pack_kernel[(n_words,)](
            source_spikes,
            packed_history,
            current_time_step,
            n_source,
            n_words,
            BITS_PER_WORD=_BITS_PER_WORD,
            BLOCK_BITS=_PACK_BLOCK_BITS,
        )
    except BaseException as exc:  # pragma: no cover - depends on GPU/Triton
        _disable(exc)
        raise


def build_delivery(
    packed_history: torch.Tensor,
    current_time_step: torch.Tensor,
    delay_steps: torch.Tensor,
    conn_word_idx: torch.Tensor,
    conn_bit_mask: torch.Tensor,
    post_idx: torch.Tensor,
    weight: torch.Tensor,
    delivery_out: torch.Tensor,
) -> None:
    """Build dense delivery scratch from bitpacked source-spike history.

    The caller is responsible for clearing ``delivery_out`` before this launch.
    Duplicate posts are reduced with GPU atomics, preserving the same additive
    semantics as ``index_add_``/the dense NetCon backend.
    """
    n_conn, max_delay_steps, n_words = validate_delivery_structure(
        packed_history,
        current_time_step,
        delay_steps,
        conn_word_idx,
        conn_bit_mask,
        post_idx,
        weight,
        delivery_out,
    )
    tensors = (
        packed_history,
        current_time_step,
        delay_steps,
        conn_word_idx,
        conn_bit_mask,
        post_idx,
        weight,
        delivery_out,
    )
    if not all(t.is_cuda for t in tensors):
        raise RuntimeError("NetCon bitpack Triton kernels require CUDA tensors")
    for name, tensor in zip(
        (
            "packed_history",
            "current_time_step",
            "delay_steps",
            "conn_word_idx",
            "conn_bit_mask",
            "post_idx",
            "weight",
            "delivery_out",
        ),
        tensors,
    ):
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")
    if n_conn == 0:
        return

    kernels = _define_kernels()
    if kernels is None:
        raise RuntimeError("NetCon bitpack Triton kernels are not available")

    _, delivery_kernel = kernels
    grid = ((n_conn + _DELIVERY_BLOCK - 1) // _DELIVERY_BLOCK,)
    try:
        delivery_kernel[grid](
            packed_history,
            current_time_step,
            delay_steps,
            conn_word_idx,
            conn_bit_mask,
            post_idx,
            weight,
            delivery_out,
            n_conn,
            max_delay_steps,
            n_words,
            BLOCK=_DELIVERY_BLOCK,
        )
    except BaseException as exc:  # pragma: no cover - depends on GPU/Triton
        _disable(exc)
        raise
