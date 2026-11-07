"""Canonical geometry survives refresh while trainable workspaces renew graphs."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_native_cable import DT, _model

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"
            ),
        ):
            yield
    finally:
        torch.set_num_threads(threads)


def test_canonical_refresh_preserves_geometry_and_renews_consumed_parameter_graphs():
    model = _model()
    callbacks = dn.func.make_callbacks(model, {"trace": dn.Recorder(("v", "t"))}, dt=DT)
    fixed = {
        name: (getattr(model, name), getattr(model, name)._version)
        for name in dn.Cable._CANONICAL_GEOMETRY_TENSORS
    }
    previous = model.mech.pas.g
    raw = model.mech.pas.g_param
    torch.autograd.grad(previous.square().sum(), raw)
    model._deterministically_rematerialize_initialization_parameters(
        preserve_state=True
    )
    assert model.mech.pas.g is not previous
    for name, (value, version) in fixed.items():
        assert getattr(model, name) is value and value._version == version
    result = model.run(steps=1, dt=DT, callbacks=callbacks)
    assert result["trace"]["t"].tolist() == [0.0, DT]
    gradient = torch.autograd.grad(result["trace"]["v"].square().mean(), raw)[0]
    assert gradient.isfinite().all() and gradient.abs().sum() > 0
    for name, (value, version) in fixed.items():
        assert getattr(model, name) is value and value._version == version


def test_changed_canonical_rhoa_still_rejects_public_reinitialization():
    model = _model()
    with torch.no_grad():
        model.rhoa_param.rho.add_(0.5)
    with pytest.raises(
        RuntimeError, match="compiled geometry buffer 'rhoa' was modified"
    ):
        model.initialize()
    assert not model.initialized
