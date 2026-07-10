from __future__ import annotations

import contextlib
import functools
import importlib
import json
import logging
import os
import re
import time
from collections.abc import Mapping
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
    Context manager for temporarily setting Dendra ContextVar values.

    These flags control compilation and runtime behaviors across Dendra:

    - ``BACKEND`` (str): torch.compile backend (e.g., ``\"inductor\"``).
    - ``FULLGRAPH`` (int/bool): request full-graph compilation.
    - ``DYNAMIC`` (int/bool): enable dynamic shape compilation.
    - ``JIT`` (int/bool): enable standalone population/integrator JIT and
      opt into aggressive network-side compilation where supported.
    - ``JIT_NETWORK_SOLVES`` (int/bool): enable population/integrator JIT only
      for populations stepped by a Network. This replaces the legacy
      ``JIT_IN_NETWORK`` name.
    - ``JIT_NETWORK_OPS`` (int/bool): enable network-side/event/synapse JIT
      where a component exposes a compile-safe kernel.
    - ``COMPILE_MODE`` (str): torch.compile mode (e.g., ``\"default\"``).
    - ``COMPILE_OPTIONS`` (dict/str/None): optional ``torch.compile`` options
      dictionary forwarded to supported compiled kernels.
    - ``DEBUG`` (int/bool): increase logging verbosity for mechanism/state
      compilation (symbolic transforms, conductance differentiation).
    - ``DEVICE`` (str/torch.device/None): default device for newly constructed
      Dendra modules that honor context placement.
    - ``DTYPE`` (str/torch.dtype/None): optional default floating dtype for newly
      constructed Dendra modules that honor context placement.
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
        self.old_context: dict[str, Any] = {
            k: v.value for k, v in ContextVar._cache.items()
        }
        for k, v in self.kwargs.items():
            key = CONTEXT_ALIASES.get(k, k)
            if key not in ContextVar._cache:
                valid = ", ".join(sorted(ContextVar._cache))
                aliases = ", ".join(sorted(CONTEXT_ALIASES))
                raise KeyError(
                    f"Unknown Dendra context variable {k!r}. "
                    f"Valid keys: {valid}. Legacy aliases: {aliases}."
                )
            ContextVar._cache[key].value = v

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


_DEVICE_DEFAULT_SENTINELS = {"", "none", "null", "default"}
_DTYPE_DEFAULT_SENTINELS = {"", "none", "null", "default"}


def _normalize_device_value(value, default=None):
    """Normalize a Dendra device context value to ``torch.device`` or default."""
    if value is None:
        return default
    if isinstance(value, torch.device):
        return value
    if isinstance(value, str):
        v = value.strip()
        if v.lower() in _DEVICE_DEFAULT_SENTINELS:
            return default
        return torch.device(v)
    return torch.device(value)


def _normalize_dtype_value(value, default=None):
    """Normalize a Dendra dtype context value to ``torch.dtype`` or default."""
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        v = value.strip()
        if v.lower() in _DTYPE_DEFAULT_SENTINELS:
            return default
        v = v.removeprefix("torch.")
        if not hasattr(torch, v):
            raise ValueError(
                f"Unknown torch dtype {value!r}. Expected e.g. 'float32', "
                "'float64', 'bfloat16', or a torch.dtype object."
            )
        dtype = getattr(torch, v)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"torch.{v} is not a dtype")
        return dtype
    raise TypeError(f"Unsupported dtype context value {value!r}")


_COMPILE_OPTIONS_NONE_SENTINELS = {"", "none", "null", "default", "{}"}


def _freeze_compile_option_value(value):
    """Return a hashable representation of a torch.compile option value."""
    if isinstance(value, Mapping):
        return tuple(
            (str(k), _freeze_compile_option_value(v))
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_compile_option_value(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_compile_option_value(v) for v in value))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def normalize_compile_options(options=None):
    """Normalize a Dendra compile-options value for ``torch.compile``.

    Parameters
    ----------
    options : dict, Mapping, str, or None
        Optional backend-specific options to forward to ``torch.compile``.
        ``None``, an empty string, ``"none"``, ``"null"``, ``"default"``, and
        ``"{}"`` all mean no options. Strings that are not sentinels must be a
        JSON object.

    Returns
    -------
    dict or None
        A shallow ``dict`` copy suitable for passing as ``options=...`` to
        ``torch.compile``.
    """
    if options is None:
        return None
    if isinstance(options, str):
        text = options.strip()
        if text.lower() in _COMPILE_OPTIONS_NONE_SENTINELS:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "COMPILE_OPTIONS strings must be JSON objects, e.g. "
                "'{\"triton.cudagraphs\": true}', or one of: "
                f"{sorted(_COMPILE_OPTIONS_NONE_SENTINELS)}."
            ) from exc
        options = parsed
    if not isinstance(options, Mapping):
        raise TypeError(
            "COMPILE_OPTIONS must be a mapping, JSON object string, or None; "
            f"got {type(options)!r}."
        )
    out = dict(options)
    return out or None


def compile_options_key(options=None):
    """Return a stable hashable key for a compile-options value."""
    normalized = normalize_compile_options(options)
    if normalized is None:
        return None
    return tuple(
        (str(k), _freeze_compile_option_value(v))
        for k, v in sorted(normalized.items(), key=lambda item: str(item[0]))
    )


def current_compile_options(default=None):
    """Return the active normalized ``torch.compile`` options dictionary."""
    options = normalize_compile_options(COMPILE_OPTIONS.value)
    if options is None:
        return default
    return options


def current_device(default=None):
    """Return the active Dendra default device, or ``default`` when unset."""
    return _normalize_device_value(DEVICE.value, default=default)


def current_dtype(default=None):
    """Return the active Dendra default dtype, or ``default`` when unset."""
    return _normalize_dtype_value(DTYPE.value, default=default)


# Backward-compatible context-key aliases. The exported variable
# ``JIT_IN_NETWORK`` below points at ``JIT_NETWORK_SOLVES`` as well, but ctx()
# needs a key-level alias so ``with dendra.ctx(JIT_IN_NETWORK=0): ...`` keeps
# working.
CONTEXT_ALIASES = {
    "JIT_IN_NETWORK": "JIT_NETWORK_SOLVES",
}


DEBUG = ContextVar("DEBUG", 0)
DEVICE = ContextVar("DEVICE", "")
DTYPE = ContextVar("DTYPE", "")
TF32 = ContextVar("TF32", 0)
IMEM = ContextVar("IMEM", 0)
CUDA = ContextVar("CUDA", int(torch.cuda.is_available()))
PADE = ContextVar("PADE", -1)
REQUIRE_GRAD = ContextVar("REQUIRE_GRAD", 0)
USETABLES = ContextVar("USETABLES", 1)

BACKEND = ContextVar("BACKEND", "inductor")
FULLGRAPH = ContextVar("FULLGRAPH", 0)
DYNAMIC = ContextVar("DYNAMIC", 0)

# Compilation policy:
#   JIT                 -> standalone population/integrator kernels and aggressive
#                          network-side opt-in where supported.
#   JIT_NETWORK_SOLVES  -> population/integrator kernels only when the population
#                          is stepped by a Network.
#   JIT_NETWORK_OPS     -> network event/synapse plumbing where compile-safe
#                          kernels are explicitly exposed.
#
# For compatibility, the old environment variable JIT_IN_NETWORK can still seed
# JIT_NETWORK_SOLVES, and the exported JIT_IN_NETWORK object is an alias.
JIT = ContextVar("JIT", 0)
JIT_NETWORK_SOLVES = ContextVar("JIT_NETWORK_SOLVES", getenv("JIT_IN_NETWORK", 0))
JIT_NETWORK_OPS = ContextVar("JIT_NETWORK_OPS", 0)
JIT_IN_NETWORK = JIT_NETWORK_SOLVES
COMPILE_MODE = ContextVar("COMPILE_MODE", "default")
COMPILE_OPTIONS = ContextVar("COMPILE_OPTIONS", "")


def set_compile_options(options=None, **kwargs):
    """Set global ``torch.compile(options=...)`` for Dendra JIT call sites.

    Examples
    --------
    >>> set_compile_options({"triton.cudagraphs": True})
    >>> set_compile_options(None)  # clear options
    >>> set_compile_options(**{"trace.enabled": True})

    Parameters
    ----------
    options : Mapping, JSON str, or None
        Options to normalize and store. ``None`` with no keyword arguments
        clears the global options.
    **kwargs
        Convenience option entries merged into ``options``. Keyword entries
        override entries from ``options``.

    Returns
    -------
    dict or None
        The normalized options value that was stored.
    """
    normalized = normalize_compile_options(options)
    if kwargs:
        merged = {} if normalized is None else dict(normalized)
        merged.update(kwargs)
        normalized = normalize_compile_options(merged)
    COMPILE_OPTIONS.value = normalized
    return normalized


def set_jit_enabled(enable=True):
    """
    Enable or disable JIT compilation globally for Dendra models (Populations and Networks).

    Parameters
    ----------
    enable : bool
        If True, enable JIT compilation. If False, disable it. Default is True.
    """
    global JIT
    JIT.value = int(enable)
    return


def set_jit_network_solves_enabled(enable=True):
    """
    Enable or disable population/integrator kernel compilation for populations
    stepped by a Network.

    This is the preferred spelling for the legacy
    :func:`set_jit_in_network_enabled` helper.
    """
    JIT_NETWORK_SOLVES.value = int(enable)
    return


def set_jit_network_ops_enabled(enable=True):
    """
    Enable or disable network-side/event/synapse compilation where supported.

    This flag is intentionally separate from population solve compilation,
    because NetCon/NetStim/event delivery has much more Python-side state.
    """
    JIT_NETWORK_OPS.value = int(enable)
    return


def set_jit_in_network_enabled(enable=True):
    """Deprecated alias for :func:`set_jit_network_solves_enabled`."""
    return set_jit_network_solves_enabled(enable)


def jit_enabled_for_scope(scope: str, owner=None) -> bool:
    """Return the effective JIT decision for a Dendra execution scope.

    Scopes
    ------
    ``"population"``
        Standalone Population.run/longrun/steady_state. Only ``JIT`` enables
        compilation.
    ``"network_population"``
        Population/integrator kernels while stepped by a Network. Enabled by
        ``JIT`` or ``JIT_NETWORK_SOLVES``.
    ``"network_ops"``
        Network-side event/synapse/NetStim plumbing. Enabled by ``JIT`` or
        ``JIT_NETWORK_OPS``.
    """
    if owner is None:
        jit = bool(JIT)
        jit_network_solves = bool(JIT_NETWORK_SOLVES)
        jit_network_ops = bool(JIT_NETWORK_OPS)
    else:
        jit = bool(getattr(owner, "jit", bool(JIT)))
        jit_network_solves = bool(
            getattr(
                owner,
                "jit_network_solves",
                getattr(owner, "jit_in_network", bool(JIT_NETWORK_SOLVES)),
            )
        )
        jit_network_ops = bool(getattr(owner, "jit_network_ops", bool(JIT_NETWORK_OPS)))

    if scope == "population":
        return jit
    if scope == "network_population":
        return jit or jit_network_solves
    if scope == "network_ops":
        return jit or jit_network_ops
    raise ValueError(
        "Unknown Dendra JIT scope "
        f"{scope!r}; expected 'population', 'network_population', or 'network_ops'."
    )


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
logger = logging.getLogger("dendra")
logFormatter = logging.Formatter(
    "%(asctime)s %(name)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
logger.setLevel(logging.DEBUG)


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
