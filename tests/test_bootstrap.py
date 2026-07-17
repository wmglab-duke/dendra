"""Regression coverage for process-wide Dendra bootstrap helpers."""

from __future__ import annotations

import warnings

import pytest
import torch

from dendra._bootstrap import (
    reset_torch_compiler,
    torch_compiler_warning_context,
)


def test_compiler_warning_context_filters_only_known_deprecations():
    autograd_message = (
        "<class 'torch.autograd.function.Function'> should not be instantiated. "
        "Methods on autograd functions are all static."
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with torch_compiler_warning_context():
            warnings.warn_explicit(
                "torch.jit.script_method is deprecated",
                DeprecationWarning,
                filename="torch/jit/_script.py",
                lineno=365,
                module="torch.jit._script",
            )
            warnings.warn(autograd_message, DeprecationWarning, stacklevel=1)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(DeprecationWarning, match="unrelated deprecation"):
            with torch_compiler_warning_context():
                warnings.warn("unrelated deprecation", DeprecationWarning, stacklevel=1)


def test_compiler_reset_filters_only_pytorch_script_method_deprecation(monkeypatch):
    calls = []

    def reset_with_known_warning():
        calls.append("known")
        warnings.warn_explicit(
            "torch.jit.script_method is deprecated",
            DeprecationWarning,
            filename="torch/jit/_script.py",
            lineno=365,
            module="torch.jit._script",
        )

    monkeypatch.setattr(torch.compiler, "reset", reset_with_known_warning)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reset_torch_compiler()
    assert calls == ["known"]

    def dynamo_reset_with_known_warning():
        calls.append("dynamo")
        warnings.warn_explicit(
            "torch.jit.script_method is deprecated",
            DeprecationWarning,
            filename="torch/jit/_script.py",
            lineno=365,
            module="torch.jit._script",
        )

    monkeypatch.setattr(torch._dynamo, "reset", dynamo_reset_with_known_warning)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reset_torch_compiler(prefer_public=False)
    assert calls == ["known", "dynamo"]

    def reset_with_unrelated_warning():
        calls.append("unrelated")
        warnings.warn("unrelated compiler warning", RuntimeWarning, stacklevel=1)

    monkeypatch.setattr(torch.compiler, "reset", reset_with_unrelated_warning)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeWarning, match="unrelated compiler warning"):
            reset_torch_compiler()
    assert calls == ["known", "dynamo", "unrelated"]
