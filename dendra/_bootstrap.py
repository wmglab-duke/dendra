# dendra/_bootstrap.py

from __future__ import annotations

import atexit
import getpass
import os
import shutil
import sys
import tempfile
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path

_DISABLED_INDUCTOR_CACHE_POLICIES = {
    "0",
    "false",
    "none",
    "off",
    "disable",
    "disabled",
}
_PROCESS_INDUCTOR_CACHE_POLICIES = {
    "process",
    "isolated",
    "per_process",
    "per-process",
}
_SHARED_INDUCTOR_CACHE_POLICIES = {"shared", "global"}


@contextmanager
def torch_compiler_warning_context():
    """Filter only known PyTorch compiler deprecations in strict-warning runs."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*torch\.jit\.script_method.*deprecated.*",
            category=DeprecationWarning,
            module=r"torch\.jit\._script",
        )
        warnings.filterwarnings(
            "ignore",
            message=(
                r".*torch\.autograd\.function\.Function.*should not be "
                r"instantiated\..*Methods on autograd functions.*"
            ),
            category=DeprecationWarning,
        )
        yield


def reset_torch_compiler(*, prefer_public: bool = True) -> None:
    """Reset Torch compiler state without leaking PyTorch-internal warnings.

    PyTorch 2.13 can emit deprecations for ``torch.jit.script_method`` while
    lazily initializing Inductor and for internal ``autograd.Function``
    handling while tracing custom functions. Dendra still treats every other
    warning as actionable; only those upstream deprecations are filtered around
    compiler operations that trigger them.
    """
    import torch

    with torch_compiler_warning_context():
        if not prefer_public and hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        elif hasattr(torch, "compiler") and hasattr(torch.compiler, "reset"):
            torch.compiler.reset()
        elif hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "reset"):
            torch._dynamo.reset()


def _truthy(x: str | None) -> bool:
    return str(x).lower() in {"1", "true", "yes", "on"}


def _should_configure_torchinductor_cache_on_import() -> bool:
    """Return whether importing Dendra should configure TorchInductor's cache."""
    return _truthy(os.environ.get("DENDRA_CONFIGURE_TORCHINDUCTOR_CACHE", "0"))


def _should_cache_cpu_isa_for_dendra() -> bool:
    """Return whether the selected policy enables Dendra's CPU ISA cache."""
    policy = os.environ.get("DENDRA_INDUCTOR_CACHE_POLICY", "process").lower()
    return policy in (
        _PROCESS_INDUCTOR_CACHE_POLICIES | _SHARED_INDUCTOR_CACHE_POLICIES
    )


def _get_notebook_or_process_id() -> str:
    """
    Stable-enough process identifier for interactive notebooks and scripts.

    We include PID to avoid cross-process cache sharing.
    UUID avoids stale collisions if a PID is reused.
    """
    user = "unknown"
    try:
        user = getpass.getuser()
    except Exception:
        pass

    return f"{user}_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def configure_torchinductor_cache_for_dendra() -> None:
    """
    Configure TorchInductor cache behavior before importing torch.

    Environment variables
    ---------------------
    DENDRA_CONFIGURE_TORCHINDUCTOR_CACHE:
        "1" opts ``import dendra`` into Dendra-managed TorchInductor cache
        behavior, including this function and the CPU ISA cache. Default: "0".
        The flag does not affect explicit calls to this function.

    DENDRA_INDUCTOR_CACHE_POLICY:
        "process"  -> per-process cache directory, safest for notebooks
        "shared"   -> leave TORCHINDUCTOR_CACHE_DIR alone unless user set it
        "off"      -> do nothing

    DENDRA_INDUCTOR_CACHE_ROOT:
        Parent directory for Dendra-owned process caches.
        Default: tempfile.gettempdir()/dendra_torchinductor

    DENDRA_INDUCTOR_DISABLE_PCH:
        "1" disables C++ precompiled header caching where supported.

    DENDRA_INDUCTOR_COMPILE_THREADS:
        Sets TORCHINDUCTOR_COMPILE_THREADS. Default "1" in process mode.

    DENDRA_INDUCTOR_CLEANUP:
        "1" removes the Dendra-owned per-process cache at interpreter exit.
    """
    policy = os.environ.get("DENDRA_INDUCTOR_CACHE_POLICY", "process").lower()

    if policy in _DISABLED_INDUCTOR_CACHE_POLICIES:
        return

    # This is the important limitation. If torch is already imported, some
    # Inductor config/cache state may already be initialized.
    if "torch" in sys.modules:
        warnings.warn(
            "Dendra: torch was imported before Dendra configured TorchInductor. "
            "Dendra cannot reliably isolate TORCHINDUCTOR_CACHE_DIR after torch import. "
            "For concurrent notebook use, import dendra before torch or set "
            "TORCHINDUCTOR_CACHE_DIR before starting Python.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    if policy in _PROCESS_INDUCTOR_CACHE_POLICIES:
        root = Path(
            os.environ.get(
                "DENDRA_INDUCTOR_CACHE_ROOT",
                str(Path(tempfile.gettempdir()) / "dendra_torchinductor"),
            )
        ).expanduser()

        session_id = os.environ.get("DENDRA_INDUCTOR_SESSION_ID")
        if not session_id:
            session_id = _get_notebook_or_process_id()

        cache_dir = root / session_id
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Only set if user did not explicitly choose a cache directory.
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir))

        # This is the key workaround for the malformed .h/.h.pch failure mode.
        # It is an internal/implementation-dependent knob, so setdefault keeps
        # user choice authoritative.
        if _truthy(os.environ.get("DENDRA_INDUCTOR_DISABLE_PCH", "1")):
            os.environ.setdefault("TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS", "0")

        # Reduces within-process async compile races and makes failures easier
        # to reproduce/debug. Users can override.
        os.environ.setdefault(
            "TORCHINDUCTOR_COMPILE_THREADS",
            os.environ.get("DENDRA_INDUCTOR_COMPILE_THREADS", "1"),
        )

        if _truthy(os.environ.get("DENDRA_INDUCTOR_CLEANUP", "0")):

            def _cleanup_cache_dir(path: Path = cache_dir) -> None:
                shutil.rmtree(path, ignore_errors=True)

            atexit.register(_cleanup_cache_dir)

    elif policy in _SHARED_INDUCTOR_CACHE_POLICIES:
        # Respect user's chosen shared cache. Do not clean it automatically.
        if _truthy(os.environ.get("DENDRA_INDUCTOR_DISABLE_PCH", "1")):
            os.environ.setdefault("TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS", "0")

    else:
        warnings.warn(
            f"Dendra: unknown DENDRA_INDUCTOR_CACHE_POLICY={policy!r}; "
            "expected 'process', 'shared', or 'off'.",
            RuntimeWarning,
            stacklevel=2,
        )
