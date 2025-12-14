from __future__ import annotations

import contextlib
import functools
import importlib
import logging
import os
import re
import time
from typing import Any, Callable, ClassVar, Optional, TypeVar

import numpy as np
import torch


@functools.lru_cache(maxsize=None)
def getenv(key: str, default=0):
    return type(default)(os.getenv(key, default))


class classproperty(object):
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


class ctx(contextlib.ContextDecorator):
    """
    Context manager for temporarily setting AxonML ContextVar values.

    These flags control compilation and runtime behaviors across AxonML:

    - ``BACKEND`` (str): torch.compile backend (e.g., ``\"inductor\"``).
    - ``FULLGRAPH`` (int/bool): request full-graph compilation.
    - ``DYNAMIC`` (int/bool): enable dynamic shape compilation.
    - ``JIT`` (int/bool): enable/disable torch.compile wrapping.
    - ``COMPILE_MODE`` (str): torch.compile mode (e.g., ``\"default\"``).
    - ``DEBUG`` (int/bool): increase logging verbosity for mechanism/state
      compilation (symbolic transforms, conductance differentiation).
    - ``IMEM`` (int/bool): whether integrators compute/store ``i_membrane``
      in populations (required for LFP calculations).
    - ``USETABLES`` (int/bool): toggle lookup tables declared via ``TABLE`` on
      State/Mechanism.
    - ``TF32`` is available but typically not toggled here.

    Example
    -------
    >>> with ctx(DEBUG=1, JIT=0):
    ...     net.build(...)

    Parameters
    ----------
    **kwargs
        Mapping from ContextVar key to temporary value. Restored on exit.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        self.old_context: dict[str, int] = {
            k: v.value for k, v in ContextVar._cache.items()
        }
        for k, v in self.kwargs.items():
            ContextVar._cache[k].value = v

    def __exit__(self, *args):
        for k, v in self.old_context.items():
            ContextVar._cache[k].value = v


class ContextVar:
    _cache: ClassVar[dict[str, "ContextVar"]] = {}
    value: int
    key: str

    def __init__(self, key, default_value):
        if key in ContextVar._cache:
            raise RuntimeError(f"attempt to recreate ContextVar {key}")
        ContextVar._cache[key] = self
        self.value, self.key = getenv(key, default_value), key

    def __bool__(self):
        return bool(self.value)

    def __ge__(self, x):
        return self.value >= x

    def __gt__(self, x):
        return self.value > x

    def __lt__(self, x):
        return self.value < x

    def __le__(self, x):
        return self.value <= x


DEBUG = ContextVar("DEBUG", 0)
TF32 = ContextVar("TF32", 0)
IMEM = ContextVar("IMEM", 0)
CUDA = ContextVar("CUDA", int(torch.cuda.is_available()))
PADE = ContextVar("PADE", -1)
REQUIRE_GRAD = ContextVar("REQUIRE_GRAD", 0)
USETABLES = ContextVar("USETABLES", 1)

BACKEND = ContextVar("BACKEND", "inductor")
FULLGRAPH = ContextVar("FULLGRAPH", 0)
DYNAMIC = ContextVar("DYNAMIC", 0)
JIT = ContextVar("JIT", 1)
COMPILE_MODE = ContextVar("COMPILE_MODE", "default")


def set_jit_enabled(enable=True):
    """
    Enable or disable JIT compilation globally for AxonML models.

    Parameters
    ----------
    enable : bool
        If True, enable JIT compilation. If False, disable it. Default is True.
    """
    global JIT
    JIT.value = int(enable)
    return


def numpify(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


# --- function decorator to check if package available ---
def requires_packages(*pkgs: str):
    missing = [p for p in pkgs if importlib.util.find_spec(p) is None]

    def decorator(func):
        if not missing:
            return func

        @functools.wraps(func)
        def _missing(*args, **kwargs):
            names = "', '".join(missing)
            raise ImportError(f"{func.__name__} requires '{names}'.")

        return _missing

    return decorator


# --- pytorch functions ---


def detach_vars(obj, vars: list[str]):
    for v in vars:
        setattr(obj, v, getattr(obj, v).detach())


def allow_tf32(allow: bool = True) -> None:
    """
    Toggle TF32 usage for matmul (cuBLAS) and cuDNN (conv/RNN).
    - PyTorch >= 2.9: use the new fp32_precision API.
    - 2.7.0 <= PyTorch < 2.9.0: use the legacy allow_tf32 flags.

    On non-CUDA builds this is a no-op.
    """
    # Parse X.Y.Z from versions like "2.9.0+cu121"
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)", torch.__version__)
    ver = tuple(map(int, m.groups())) if m else (0, 0, 0)
    use_new = ver >= (2, 9, 0)

    if not hasattr(torch.backends, "cuda"):  # CPU/MPS build
        return

    if use_new:
        mode = "tf32" if allow else "ieee"
        # New fine-grained switches (don’t mix with old ones)
        try:
            torch.backends.cuda.matmul.fp32_precision = mode
        except Exception:
            pass
        try:
            torch.backends.cudnn.conv.fp32_precision = mode
        except Exception:
            pass
        try:
            torch.backends.cudnn.rnn.fp32_precision = mode
        except Exception:
            pass
    else:
        # Legacy flags for 2.7–2.8
        torch.backends.cuda.matmul.allow_tf32 = bool(allow)
        torch.backends.cudnn.allow_tf32 = bool(allow)


def ve_from_s_t(space, time, n, device, multicontact=False):
    ve_s = torch.as_tensor(space, device=device)
    ve_t = torch.as_tensor(time, device=device)

    if multicontact:
        ve_s = ve_s.expand(-1, n, -1)
        ve_t = ve_t.expand(-1, n, -1)
        einsum = op_mc
    else:
        ve_s = ve_s.expand(n, -1)
        ve_t = ve_t.expand(n, -1)
        einsum = op_sc

    return einsum(ve_s, ve_t)


def op_mc(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.einsum("can,cat->tan", s, t).contiguous()


def op_sc(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.einsum("an,at->tan", s, t).contiguous()


F = TypeVar("F", bound=Callable[..., Any])


@functools.wraps(torch._dynamo.disable)
def nojit(fn: Optional[F] = None, recursive: bool = True) -> F:  # type: ignore[misc]
    """Disable TorchDynamo compilation for a function.

    This is a convenience wrapper around :func:`torch._dynamo.disable`.
    """
    return torch._dynamo.disable(fn=fn, recursive=recursive)  # type: ignore[return-value]


# -- timing utilities --

TIME_STACK = []

# -- define base logger and formatting options --
logger = logging.getLogger("axonml")
logFormatter = logging.Formatter(
    "%(asctime)s %(name)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
logger.setLevel(logging.INFO)


def tic(message=None, log=True):
    """
    Start a wall-clock timer.

    Parameters
    ----------
    message : str, optional
        Optional message to log when starting the timer.
    log : bool, optional
        If True (default), log the message via the module logger.
    """
    TIME_STACK.append(time.time())
    if message and log:
        logger.info(str(message))


def toc(message=None, log=True):
    """
    Stop the most recent timer and return elapsed seconds.

    Parameters
    ----------
    message : str, optional
        Optional message to prefix the elapsed time.
    log : bool, optional
        If True (default), log the elapsed time via the module logger.

    Returns
    -------
    float
        Elapsed time in seconds.

    Notes
    -----
    Raises a log error if called without a matching :func:`tic`.
    """
    try:
        t = time.time() - TIME_STACK.pop()
        output = f"Elapsed: {t:.3f}s"
        if message:
            output = f"{message}:: {output}"
        if log:
            logger.info(output)
        return t
    except IndexError:
        logger.error("You have to tic() before you toc()")
