"""Forward-only warm-up around the existing host runner lifecycle."""

from __future__ import annotations

import sys
from threading import Event, Thread
from time import perf_counter

import torch

from ._recording import _discard_recording_publication
from ._runner_policy import _validate_host_call

_WARM_PROGRESS_INTERVAL = 10.0


def _progress(message):
    # Optional console diagnostics must not replace an execution exception.
    try:
        print(message, file=sys.stderr, flush=True)
    except Exception:
        pass


def _warm_forward(program, state, call_kwargs):
    with _discard_recording_publication():
        program(state, **call_kwargs)
        device = program.functional.device
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()


def _warm_runner(program, state=None, *, verbose=False, **call_kwargs):
    """Execute the actual forward call, discard its results and return its runner.

    Preserve example owners, views and gradient flags, as well as the runner's
    configured initialization, callbacks and scheduling. Numerical callback
    execution remains enabled; host Recorder publication is discarded. Training
    backward is not executed and no result graph is retained by this helper.
    """
    if type(verbose) is not bool:
        raise TypeError("verbose must be a bool")
    _validate_host_call()
    if not verbose:
        _warm_forward(program, state, call_kwargs)
        return program

    began = perf_counter()
    stopped = Event()

    def heartbeat():
        while not stopped.wait(_WARM_PROGRESS_INTERVAL):
            _progress(f"Warming forward... {perf_counter() - began:.1f} s elapsed")

    _progress("Warming forward...")
    reporter = Thread(target=heartbeat, name="dendra-forward-warmup", daemon=True)
    reporter.start()
    failed = False
    try:
        _warm_forward(program, state, call_kwargs)
    except BaseException:
        failed = True
        raise
    finally:
        stopped.set()
        reporter.join()
        elapsed = perf_counter() - began
        if failed:
            _progress(f"Forward warm-up failed after {elapsed:.2f} s")
        else:
            _progress(f"Forward ready in {elapsed:.2f} s")
    return program


__all__ = []
