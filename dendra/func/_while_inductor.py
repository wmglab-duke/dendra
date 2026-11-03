"""Ordinary Torch compilation of Dendra's bounded inference while loops."""

from __future__ import annotations

from dendra._bootstrap import compile_owned, isolate_compilation_callable


def _compile_while(function, compile_options, *, device=None):
    """Keep compiler choices while outlining qualified large CPU while bodies.

    Unknown contracts use ordinary compilation. The immutable device hint is
    internal and is never forwarded as a Torch compiler option.
    """
    options = dict(compile_options)
    if options.get("options") is not None:
        options["options"] = dict(options["options"])
    if getattr(device, "type", None) == "cpu":
        from ._while_outlining import _while_outlining_options

        options = _while_outlining_options(options, device=device)
    return compile_owned(isolate_compilation_callable(function), **options)


_STANDARD_COMPILER = _compile_while
