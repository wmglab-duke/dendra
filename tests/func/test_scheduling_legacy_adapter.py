"""Legacy callable metadata stays outside canonical binding execution."""

import pytest
from test_runners import DT, _assert_tree_close, _model

import dendra as dn

pytestmark = pytest.mark.cpu


def test_legacy_bound_metadata_is_inspected_before_source_validation(monkeypatch):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    other, _ = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    bound = functional.bind(tensors.parameters, prepared)
    expected = dn.func.run(bound, tensors.state, steps=0)

    # Explicit bindings carry their own owner. Legacy callable scheduling still
    # interprets wrapper metadata, including metadata attached to a binding.
    object.__setattr__(bound, "__wrapped__", other.step)
    actual = dn.func.run(bound, tensors.state, steps=0)
    _assert_tree_close(actual, expected, rtol=0, atol=0)

    def unexpected_source_check():
        raise AssertionError("legacy callable plan checks must precede source checks")

    monkeypatch.setattr(functional, "_validate_source", unexpected_source_check)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="step belongs to a different FunctionalPopulation plan",
    ):
        dn.func.run(functional, bound, tensors.state, tstop=0.0)
