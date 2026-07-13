import inspect
import re
import textwrap
from typing import Any, List, Optional

try:
    from IPython import get_ipython
    from IPython.core.oinspect import Inspector

    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False


class SourceUnavailableError(RuntimeError):
    """Raised when none of the supported source-recovery strategies succeeds."""


# ---------------------------------------------------------------------------
# tiny helpers for the inspector fall-back
# ---------------------------------------------------------------------------
def _ipython_source(obj) -> Optional[str]:
    """Return source text via IPython Inspector, or None if unavailable."""
    try:
        insp = Inspector()
    except Exception:
        # Inspector's constructor has changed across IPython releases.  Source
        # recovery must still reach the ordinary method-inspection fallback when
        # an optional, installed IPython version is incompatible.
        return None

    # Inspector APIs have also changed names over time.  Some releases expose
    # more than one of these methods, so try each independently instead of
    # assuming that the newest-looking attribute is usable.
    for method_name in ("getsourcelines", "findsourcelines", "getsource"):
        try:
            method = getattr(insp, method_name, None)
        except Exception:
            continue
        if method is None:
            continue
        try:
            result = method(obj)
        except Exception:
            continue

        if method_name in ("getsourcelines", "findsourcelines"):
            try:
                lines, _ = result
                result = "".join(lines)
            except Exception:
                continue
        if isinstance(result, str) and result:
            return result
    return None


def _ipython_history_source(obj, *, strip_decorators: bool) -> Optional[str]:
    """Return an object's defining IPython cell, or None when history is unusable."""
    try:
        ip = get_ipython()
        if ip is None or not hasattr(obj, "__code__"):
            return None

        filename = obj.__code__.co_filename  # e.g. <ipython-input-17-abc123>
        match = re.match(r"<ipython-input-(\d+)-", filename)
        if match is None:
            return None

        cell_num = int(match.group(1))
        src = ip.user_ns["In"][cell_num]
        if not isinstance(src, str) or not src:
            return None
    except Exception:
        # An inactive shell, truncated history, or a nonstandard user namespace
        # must not prevent the final class-method reconstruction strategy.
        return None

    if strip_decorators:
        src = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith("@")
        )
    return src


def safe_source(obj: Any, *, strip_decorators: bool = False) -> str:
    """
    Robustly fetch the source of `obj` (class or function) in a Jupyter
    notebook, trying several strategies before giving up.
    """
    # 1. Plain inspect -------------------------------------------------------
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        pass

    if IPYTHON_AVAILABLE:
        # 2. IPython Inspector -----------------------------------------------
        ip_src = _ipython_source(obj)
        if ip_src:
            return textwrap.dedent(ip_src)

        # 3. Pull the defining cell from history -----------------------------
        history_src = _ipython_history_source(obj, strip_decorators=strip_decorators)
        if history_src:
            return history_src

    # 4. Re-assemble a class from its methods (last resort) ------------------
    if inspect.isclass(obj):
        pieces: List[str] = []
        for name, member in obj.__dict__.items():
            if inspect.isfunction(member):
                try:
                    pieces.append(textwrap.dedent(inspect.getsource(member)))
                except (OSError, TypeError):
                    continue
        if pieces:
            body = "\n\n".join(pieces)
            return f"class {obj.__name__}:\n" + textwrap.indent(body, "    ")

    raise SourceUnavailableError(f"Could not retrieve source for {obj!r}")
