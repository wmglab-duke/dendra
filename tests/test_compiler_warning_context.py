"""The optimized compiler filter must retain the previous warning boundaries."""

from __future__ import annotations

import warnings

import pytest

import dendra._bootstrap as bootstrap


@pytest.fixture(params=[False, True], ids=["public-fallback", "precompiled"])
def compiler_context(request, monkeypatch):
    if request.param and not bootstrap._USE_PRECOMPILED_WARNING_FILTERS:
        pytest.skip("direct warning-filter update is unavailable on this runtime")
    monkeypatch.setattr(bootstrap, "_USE_PRECOMPILED_WARNING_FILTERS", request.param)
    return bootstrap.torch_compiler_warning_context


_KNOWN_WARNINGS = (
    ("torch.jit.script_method is deprecated", DeprecationWarning, "torch.jit._script"),
    ("torch.jit.script is deprecated", DeprecationWarning, "torch.jit._script"),
    ("torch.jit.script_method is deprecated", FutureWarning, "torch.jit._script"),
    ("torch.jit.script is deprecated", FutureWarning, "torch.jit._script"),
    (
        "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed.",
        UserWarning,
        "torch._dynamo.variables.tensor",
    ),
    (
        "<class 'torch.autograd.function.Function'> should not be instantiated. "
        "Methods on autograd functions are all static.",
        DeprecationWarning,
        "torch.autograd.function",
    ),
)


def _emit(message, category, module):
    warnings.warn_explicit(
        message,
        category,
        filename="compiler_warning_probe.py",
        lineno=1,
        module=module,
        registry={},
    )


@pytest.mark.parametrize("message, category, module", _KNOWN_WARNINGS)
def test_known_warning_is_scoped_and_repeated_calls_still_filter(
    compiler_context, message, category, module
):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # A cached compiled callable may start recompiling on any later call.
        for _ in range(3):
            with compiler_context():
                _emit(message, category, module)
            # Equivalent warnings emitted by callbacks outside the compiler
            # boundary must not inherit its suppression.
            with pytest.raises(category):
                _emit(message, category, module)


@pytest.mark.parametrize(
    "message, category, module",
    (
        ("unrelated runtime warning", RuntimeWarning, "torch._dynamo"),
        ("unrelated user warning", UserWarning, "torch._dynamo"),
        ("unrelated deprecation", DeprecationWarning, "torch.jit._script"),
        ("unrelated future warning", FutureWarning, "torch.jit._script"),
        ("torch.jit.script is deprecated", DeprecationWarning, "user_callbacks"),
        ("torch.jit.script is deprecated", FutureWarning, "user_callbacks"),
        ("torch.jit.script is deprecated", UserWarning, "torch.jit._script"),
    ),
)
def test_unrelated_warnings_remain_visible(compiler_context, message, category, module):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(category):
            with compiler_context():
                _emit(message, category, module)


@pytest.mark.parametrize("duplicate_count", [0, 1, 2])
def test_filter_precedence_and_dedup_match_public_api(
    compiler_context, duplicate_count
):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filters.extend(
            list(bootstrap._TORCH_COMPILER_WARNING_FILTERS) * duplicate_count
        )
        inherited = list(warnings.filters)
        with warnings.catch_warnings():
            for message, category, module in bootstrap._TORCH_COMPILER_WARNING_SPECS:
                warnings.filterwarnings(
                    "ignore", message=message, category=category, module=module
                )
            expected = list(warnings.filters)
        with compiler_context():
            assert warnings.filters == expected
        assert warnings.filters == inherited


def test_nested_filter_changes_and_exception_restore_exact_outer_state(
    compiler_context,
):
    class ExpectedFailure(Exception):
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        original = warnings.filters
        original_values = list(original)
        original_showwarning = warnings.showwarning
        with compiler_context():
            outer = warnings.filters
            outer_values = list(outer)
            with pytest.raises(ExpectedFailure):
                with compiler_context():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    warnings.showwarning = lambda *args, **kwargs: None
                    raise ExpectedFailure
            assert warnings.filters is outer
            assert warnings.filters == outer_values
            assert warnings.showwarning is original_showwarning
            with pytest.raises(RuntimeWarning):
                warnings.warn("unrelated after inner exit", RuntimeWarning)
        assert warnings.filters is original
        assert warnings.filters == original_values
        assert warnings.showwarning is original_showwarning


def test_warning_capture_and_registry_invalidation_match_previous_calls(
    compiler_context,
):
    registry = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        for _ in range(3):
            with compiler_context():
                warnings.warn_explicit(
                    "unrelated repeated warning",
                    RuntimeWarning,
                    filename="compiler_warning_probe.py",
                    lineno=1,
                    module="probe",
                    registry=registry,
                )
        # Like the old filterwarnings path, each entry invalidates registries.
        assert len(caught) == 3
        assert all(str(item.message) == "unrelated repeated warning" for item in caught)


def test_precompiled_path_does_not_call_filterwarnings(monkeypatch):
    if not bootstrap._USE_PRECOMPILED_WARNING_FILTERS:
        pytest.skip("direct warning-filter update is unavailable on this runtime")

    def unexpected_filter_call(*args, **kwargs):
        raise AssertionError("hot path rebuilt warning filters")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        monkeypatch.setattr(warnings, "filterwarnings", unexpected_filter_call)
        with bootstrap.torch_compiler_warning_context():
            for message, category, module in _KNOWN_WARNINGS:
                _emit(message, category, module)
