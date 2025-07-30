import inspect
import re
import textwrap
from typing import Any, List, Optional, Tuple

from IPython import get_ipython
from IPython.core.oinspect import Inspector


# ---------------------------------------------------------------------------
# tiny helpers for the inspector fall-back
# ---------------------------------------------------------------------------
def _ipython_source(obj) -> Optional[str]:
    """Return source text via IPython Inspector, or None if unavailable."""
    insp = Inspector()
    # Newer IPython (≥ 7.17) ----------------------------
    if hasattr(insp, "getsourcelines"):  # returns (lines, lineno)
        try:
            lines, _ = insp.getsourcelines(obj)
            return "".join(lines)
        except Exception:
            pass
    # Older IPython (< 7.17) ----------------------------
    elif hasattr(insp, "findsourcelines"):
        try:
            lines, _ = insp.findsourcelines(obj)
            return "".join(lines)
        except Exception:
            pass
    # Very old (rare): 'getsource'
    elif hasattr(insp, "getsource"):
        try:
            return insp.getsource(obj)
        except Exception:
            pass
    return None


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

    # 2. IPython Inspector ---------------------------------------------------
    ip_src = _ipython_source(obj)
    if ip_src:
        return textwrap.dedent(ip_src)

    # 3. Pull the defining cell from history --------------------------------
    ip = get_ipython()
    if ip and hasattr(obj, "__code__"):
        fn = obj.__code__.co_filename  # e.g. "<ipython-input-17-abc123>"
        if fn.startswith("<ipython-input-"):
            m = re.match(r"<ipython-input-(\d+)-", fn)
            if m:
                cell_num = int(m.group(1))
                src = ip.user_ns["In"][cell_num]
                if strip_decorators:
                    src = "\n".join(
                        l for l in src.splitlines() if not l.lstrip().startswith("@")
                    )
                return src

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

    raise RuntimeError(f"Could not retrieve source for {obj!r}")
