"""Cross-version robustness tests for mechanism source recovery."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dendra.models.mechanisms.compilers import source as source_compiler


class _SourceFixture:
    def i(self, v):
        return self.g * (v - self.e)


def _dynamic_fixture(name="DynamicSourceFixture"):
    return type(
        name,
        (),
        {
            "__module__": __name__,
            "i": _SourceFixture.i,
        },
    )


def _force_class_inspection_failure(monkeypatch, dynamic_class):
    real_getsource = source_compiler.inspect.getsource

    def getsource_without_dynamic_class(obj):
        if obj is dynamic_class:
            raise OSError("dynamic class has no class-body source")
        return real_getsource(obj)

    monkeypatch.setattr(
        source_compiler.inspect, "getsource", getsource_without_dynamic_class
    )


@pytest.mark.parametrize("successful_method", range(3))
def test_ipython_source_tries_each_inspector_api_independently(
    monkeypatch, successful_method
):
    calls = []
    source = "class Recovered:\n    pass\n"

    class FakeInspector:
        def _result(self, index, name):
            calls.append(name)
            if index != successful_method:
                raise RuntimeError(f"{name} is unavailable")
            if name == "getsource":
                return source
            return source.splitlines(keepends=True), 1

        def getsourcelines(self, obj):
            return self._result(0, "getsourcelines")

        def findsourcelines(self, obj):
            return self._result(1, "findsourcelines")

        def getsource(self, obj):
            return self._result(2, "getsource")

    monkeypatch.setattr(source_compiler, "Inspector", FakeInspector)

    assert source_compiler._ipython_source(object()) == source
    assert (
        calls
        == [
            "getsourcelines",
            "findsourcelines",
            "getsource",
        ][: successful_method + 1]
    )


def test_ipython_source_skips_malformed_api_results(monkeypatch):
    source = "class Recovered:\n    pass\n"

    class MalformedThenWorkingInspector:
        def getsourcelines(self, obj):
            return 123

        def findsourcelines(self, obj):
            return source.splitlines(keepends=True), 1

    monkeypatch.setattr(source_compiler, "Inspector", MalformedThenWorkingInspector)

    assert source_compiler._ipython_source(object()) == source


def test_inspector_constructor_failure_reaches_class_method_reconstruction(monkeypatch):
    dynamic_class = _dynamic_fixture("ConstructorFailureFixture")
    _force_class_inspection_failure(monkeypatch, dynamic_class)
    monkeypatch.setattr(source_compiler, "IPYTHON_AVAILABLE", True)

    def incompatible_inspector():
        raise TypeError("this IPython release requires different constructor arguments")

    monkeypatch.setattr(source_compiler, "Inspector", incompatible_inspector)
    monkeypatch.setattr(source_compiler, "get_ipython", lambda: None)

    recovered = source_compiler.safe_source(dynamic_class)

    assert "class ConstructorFailureFixture" in recovered
    assert "def i(self, v)" in recovered


def test_valid_ipython_history_is_still_recovered(monkeypatch):
    dynamic_class = _dynamic_fixture("ValidHistoryFixture")
    dynamic_class.__code__ = _SourceFixture.i.__code__.replace(
        co_filename="<ipython-input-7-audit>"
    )
    _force_class_inspection_failure(monkeypatch, dynamic_class)
    monkeypatch.setattr(source_compiler, "IPYTHON_AVAILABLE", True)
    monkeypatch.setattr(source_compiler, "Inspector", _FailingInspectorAPIs)
    cell_source = "class FromHistory:\n    pass\n"
    monkeypatch.setattr(
        source_compiler,
        "get_ipython",
        lambda: SimpleNamespace(user_ns={"In": {7: cell_source}}),
    )

    assert source_compiler.safe_source(dynamic_class) == cell_source


class _FailingInspectorAPIs:
    def getsourcelines(self, obj):
        raise RuntimeError("getsourcelines failed")

    def findsourcelines(self, obj):
        raise RuntimeError("findsourcelines failed")

    def getsource(self, obj):
        raise RuntimeError("getsource failed")


@pytest.mark.parametrize(
    "shell_factory",
    [
        pytest.param(lambda: None, id="inactive"),
        pytest.param(
            lambda: (_ for _ in ()).throw(RuntimeError("shell lookup failed")),
            id="lookup-error",
        ),
        pytest.param(lambda: SimpleNamespace(), id="missing-user-namespace"),
        pytest.param(lambda: SimpleNamespace(user_ns={}), id="missing-history"),
        pytest.param(
            lambda: SimpleNamespace(user_ns={"In": []}), id="truncated-history"
        ),
        pytest.param(
            lambda: SimpleNamespace(user_ns={"In": {7: None}}),
            id="non-string-cell",
        ),
    ],
)
def test_malformed_ipython_history_reaches_class_method_reconstruction(
    monkeypatch, shell_factory
):
    dynamic_class = _dynamic_fixture("MalformedHistoryFixture")
    # Classes normally have no code object.  Supplying a notebook-style one
    # exercises history recovery before the class-method fallback.
    dynamic_class.__code__ = _SourceFixture.i.__code__.replace(
        co_filename="<ipython-input-7-audit>"
    )
    _force_class_inspection_failure(monkeypatch, dynamic_class)
    monkeypatch.setattr(source_compiler, "IPYTHON_AVAILABLE", True)
    monkeypatch.setattr(source_compiler, "Inspector", _FailingInspectorAPIs)
    monkeypatch.setattr(source_compiler, "get_ipython", shell_factory)

    recovered = source_compiler.safe_source(dynamic_class)

    assert "class MalformedHistoryFixture" in recovered
    assert "def i(self, v)" in recovered
