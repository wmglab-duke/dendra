"""Lazy build of the optional CPU event enqueue operator."""

import os
from pathlib import Path
from threading import Lock

import torch

_build_lock = Lock()
_operator = None
_build_error = None


def load_cpu_two_tap_enqueue():
    """Return the registered operator, or raise if the toolchain is unavailable."""
    global _operator, _build_error

    if _operator is not None:
        return _operator
    with _build_lock:
        if _operator is not None:
            return _operator
        if _build_error is None:
            try:
                from torch.utils.cpp_extension import load

                source = Path(__file__).with_name("_two_tap_enqueue_cpu.cpp")
                # Contraction changes the rounding of a multiply followed by
                # an addition into the queue. Retain the tensor path's order.
                flags = (
                    ["/O2", "/fp:strict"]
                    if os.name == "nt"
                    else ["-O3", "-ffp-contract=off"]
                )
                load(
                    name="dendra_two_tap_enqueue_cpu",
                    sources=[str(source)],
                    extra_cflags=flags,
                    with_cuda=False,
                    is_python_module=False,
                )
                _operator = torch.ops.dendra_native.ordered_two_tap_enqueue.default
            except Exception as error:
                _build_error = f"{type(error).__name__}: {error}"
        if _build_error is not None:
            raise RuntimeError(
                f"Could not load the Dendra CPU two-tap enqueue operator: {_build_error}"
            )
        return _operator
