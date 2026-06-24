import builtins
import hashlib
import linecache
from collections.abc import Mapping
from typing import Any


def _numbered_source(source: str) -> str:
    return "\n".join(
        f"{lineno:4d}: {line}"
        for lineno, line in enumerate(source.splitlines(), start=1)
    )


def _register_linecache(filename: str, source: str) -> None:
    # Makes tracebacks from generated code display useful source lines.
    if not source.endswith("\n"):
        source = source + "\n"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )


def compile_generated_function(
    source: str,
    *,
    func_name: str,
    filename_prefix: str,
    global_ns: Mapping[str, Any] | None = None,
    extra_ns: Mapping[str, Any] | None = None,
):
    """
    Compile `source`, execute it in an explicit module-like namespace,
    and return the named function.

    This is compatible with Python 3.13+ because it never relies on
    implicit frame locals or `locals()` after `exec()`.
    """
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    filename = f"<{filename_prefix}:{digest}>"

    ns: dict[str, Any] = dict(global_ns if global_ns is not None else globals())
    ns.setdefault("__builtins__", builtins.__dict__)

    if extra_ns is not None:
        ns.update(extra_ns)

    _register_linecache(filename, source)

    try:
        code = compile(source, filename, "exec", dont_inherit=True)
        exec(code, ns, ns)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to compile/execute generated function {func_name!r} "
            f"from {filename}.\n\nGenerated source:\n{_numbered_source(source)}"
        ) from exc

    try:
        fn = ns[func_name]
    except KeyError as exc:
        raise RuntimeError(
            f"Generated source did not define {func_name!r}.\n\n"
            f"Generated source:\n{_numbered_source(source)}"
        ) from exc

    if not callable(fn):
        raise TypeError(
            f"Generated object {func_name!r} is not callable: {type(fn).__name__}"
        )

    # Very useful when debugging generated mechanisms interactively.
    try:
        fn.__source__ = source
        fn.__generated_filename__ = filename
    except Exception:
        pass

    return fn
