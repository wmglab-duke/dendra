"""Optional CUDA extension for NetCon bitpacked source-spike history.

The Python NetCon backend falls back to pure PyTorch when this extension is not
available, so importing Dendra does not require compiling kernels.  The extension
is compiled lazily on first use in a CUDA process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

_ext = None
_ext_error: Optional[BaseException] = None


def _load_extension():
    global _ext, _ext_error
    if _ext is not None:
        return None if _ext is False else _ext
    if not torch.cuda.is_available():
        _ext = False
        return None

    try:
        from torch.utils.cpp_extension import load

        here = Path(__file__).resolve().parent
        sources = [
            str(here / "netcon_bitpack_kernel.cpp"),
            str(here / "netcon_bitpack_kernel.cu"),
        ]
        _ext = load(
            name="dendra_netcon_bitpack_ops",
            sources=sources,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        return _ext
    except BaseException as exc:  # pragma: no cover - depends on local CUDA toolchain
        _ext_error = exc
        _ext = False
        return None


def is_available() -> bool:
    return _load_extension() is not None


def last_error() -> Optional[BaseException]:
    return _ext_error


def pack_source_spikes(
    source_spikes: torch.Tensor,
    packed_history: torch.Tensor,
    current_time_step: torch.Tensor,
) -> None:
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("NetCon bitpack CUDA extension is not available")
    ext.pack_source_spikes(source_spikes, packed_history, current_time_step)


def build_delivery(
    packed_history: torch.Tensor,
    current_time_step: torch.Tensor,
    delay_steps: torch.Tensor,
    conn_source_pos: torch.Tensor,
    post_idx: torch.Tensor,
    weight: torch.Tensor,
    delivery_out: torch.Tensor,
) -> None:
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("NetCon bitpack CUDA extension is not available")
    ext.build_delivery(
        packed_history,
        current_time_step,
        delay_steps,
        conn_source_pos,
        post_idx,
        weight,
        delivery_out,
    )
