"""Portable continuation schemas cover geometry and mechanism-owned sources."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE
from tests.func.test_continuation import _close
from tests.func.test_continuation_checkpoints import _disk, _frames, _tensors
from tests.func.test_mechanism_delay_execution import _MultiCircularLeak
from tests.func.test_mechanism_injection_execution import DT as MECHANISM_DT
from tests.func.test_mechanism_injection_execution import (
    _NamedInjectedLeak,
)
from tests.func.test_mechanism_injection_execution import (
    _population as _injected_population,
)
from tests.func.test_mechanism_injection_execution import (
    _SelectedWaveform,
)
from tests.func.test_multi_population import HH_GNABAR, _native_cable_multi
from tests.func.test_native_cable import DT
from tests.func.test_native_cable import _model as _cable
from tests.func.test_scalar_tree import _model as _tree

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
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
        torch.set_num_threads(previous)


def _restore(functional, receiver, current, path, monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("portable restoration must use the lowered tensor schema")

    packet = _disk(dn.func.export_continuation(functional, current), path)
    with monkeypatch.context() as guarded:
        for name in ("extract", "prepare", "initialize", "sample_runtime_draws"):
            guarded.setattr(receiver, name, reject)
        restored = dn.func.restore_continuation(receiver, packet)
    _close(_frames(restored), _frames(current))
    copies = {}
    for original, value in zip(
        _tensors(_frames(current)), _tensors(_frames(restored)), strict=True
    ):
        assert value is not original
        assert type(value) is torch.Tensor and value.is_leaf and value.grad_fn is None
        assert value.requires_grad is original.requires_grad
        if id(original) in copies:
            assert value is copies[id(original)]
        else:
            assert all(value is not previous for previous in copies.values())
            copies[id(original)] = value
    return restored


_SOLVER = pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE, reason="this topology requires dendra-solvers"
)


@pytest.mark.parametrize(
    ("build", "conductance"),
    (
        pytest.param(_cable, "integrator.mech.mechanisms.pas.g_param", id="cable"),
        pytest.param(
            _tree,
            "integrator.mech.mechanisms.pas.g_param",
            id="tree",
            marks=_SOLVER,
        ),
        pytest.param(_native_cable_multi, HH_GNABAR, id="packed-cable", marks=_SOLVER),
    ),
)
def test_geometry_continuation_restores_against_lowered_metadata_and_rebinds(
    build, conductance, tmp_path, monkeypatch
):
    functional, tensors = dn.func.make_functional(build(), dt=DT)
    receiver, _ = dn.func.make_functional(build(), dt=DT)
    parameters = tensors.independent_parameters((conductance,))
    bound = functional.bind(parameters, constants=tensors.constants)
    _, prefix = dn.func.run(bound, tensors.state, steps=2)
    current = prefix["continuation"]
    restored = _restore(
        functional, receiver, current, tmp_path / "geometry.pt", monkeypatch
    )
    target = restored.parameters[conductance]
    assert target.requires_grad
    expected, _ = dn.func.run(bound, current, steps=3)
    receiving_bound = receiver.bind(restored.parameters, constants=restored.constants)
    actual, suffix = dn.func.run(receiving_bound, restored, steps=3)
    _close(actual, expected)
    derivative = torch.autograd.grad(actual["integrator"]["v"].square().sum(), target)[
        0
    ]
    assert torch.isfinite(derivative).all() and derivative.abs().sum() > 0
    assert suffix["continuation"].parameters[conductance] is target


def _injected_delayed_population():
    population = _injected_population(_NamedInjectedLeak, waveform=_SelectedWaveform())
    population.insert(_MultiCircularLeak.rename("delays"))
    population.initialize()
    population.integrator._initialize(
        population, population.t.new_tensor(MECHANISM_DT), force=True
    )
    population.detach()
    return population


def test_mechanism_delay_and_waveform_constants_restore_with_owned_trainable_sources(
    tmp_path, monkeypatch
):
    functional, tensors = dn.func.make_functional(
        _injected_delayed_population(), dt=MECHANISM_DT
    )
    receiver, _ = dn.func.make_functional(
        _injected_delayed_population(), dt=MECHANISM_DT
    )
    parameters = tensors.independent_parameters(("*injected_waveforms.*.amp",))
    constants = dict(tensors.constants)
    spatial_name = next(name for name in constants if name.endswith(".spatial"))
    constants[spatial_name] = constants[spatial_name].detach().clone().requires_grad_()
    delay_name = next(
        name for name in constants if name.endswith(".signal_delay_steps")
    )
    assert constants[delay_name].dtype is torch.int64
    torch.testing.assert_close(constants[delay_name], torch.tensor((0, 1, 3)))
    bound = functional.bind(parameters, constants=constants)
    _, prefix = dn.func.run(bound, tensors.state, steps=2)
    current = prefix["continuation"]
    restored = _restore(
        functional, receiver, current, tmp_path / "mechanisms.pt", monkeypatch
    )
    amplitude_name = next(name for name in parameters if name.endswith(".amp"))
    amplitude = restored.parameters[amplitude_name]
    spatial = restored.constants[spatial_name]
    assert amplitude.requires_grad and spatial.requires_grad
    torch.testing.assert_close(restored.constants[delay_name], constants[delay_name])
    expected, _ = dn.func.run(bound, current, steps=3)
    receiving_bound = receiver.bind(restored.parameters, constants=restored.constants)
    actual, _ = dn.func.run(receiving_bound, restored, steps=3)
    _close(actual, expected)
    gradients = torch.autograd.grad(
        actual["integrator"]["v"].square().sum(), (amplitude, spatial)
    )
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
    )
