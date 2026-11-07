"""Canonical bindings do not acquire legacy callable-adapter semantics."""

import pytest
import torch
from test_runners import DT, _assert_tree_close, _model

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("compiled", (False, True))
def test_explicit_binding_ignores_legacy_wrapper_metadata(monkeypatch, compiled):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    kernel = functional.compile_rollout_chunk(2, backend="eager") if compiled else None
    bound = (functional if kernel is None else kernel).bind(
        tensors.parameters, prepared
    )

    def legacy_metadata(_binding):
        raise AssertionError("An explicit binding must not inspect wrapper metadata")

    # Authored wrapper attributes cannot redirect an already explicit owner.
    # Ordinary callable compatibility forms retain their separate adapter rules.
    for name in ("__wrapped__", "_torchdynamo_orig_callable"):
        monkeypatch.setattr(
            dn.func.BoundPopulation, name, property(legacy_metadata), raising=False
        )

    with torch.no_grad(), torch_compiler_warning_context():
        expected = functional.rollout(
            tensors.parameters, prepared, tensors.state, steps=3
        )
        actual = dn.func.run(bound, tensors.state, steps=3, host_span_steps=2)
    _assert_tree_close(actual, expected, rtol=2e-10, atol=2e-11)
