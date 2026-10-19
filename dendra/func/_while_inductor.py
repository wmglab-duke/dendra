"""Ordinary Torch compilation of Dendra's bounded inference while loops."""

from __future__ import annotations

import torch

from dendra._bootstrap import isolate_compilation_callable


def _compile_while(function, compile_options):
    """Preserve compiler selection/options without private scheduling hooks.

    The owned root keeps an independent specialization budget. All options,
    including nested dictionaries, pass to ordinary ``torch.compile`` without
    mutation; unsupported compiler contracts raise from that selected route.
    """
    options = dict(compile_options)
    if options.get("options") is not None:
        options["options"] = dict(options["options"])
    return torch.compile(isolate_compilation_callable(function), **options)
