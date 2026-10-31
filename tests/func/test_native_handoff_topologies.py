"""Native solver workspaces remain coherent after continuation publication."""

from __future__ import annotations

import math

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_extcell_multi_population import _model as _block_multi
from tests.func.test_multi_population import _native_cable_multi as _scalar_multi
from tests.func.test_native_cable import _model as _native_cable
from tests.func.test_scalar_tree import _model as _scalar_tree

DT = 0.01
PREFIX, SUFFIX = 2, 2

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
    ):
        yield


def _handoff_and_continue(
    source, target, *, geometry_key=None, geometry_owner=None, geometry_name="dx"
):
    functional, tensors = dn.func.make_functional(source, dt=DT)
    parameters = tensors.independent_parameters(trainable="*")
    constants = _snapshot(tensors.constants)
    changed = tuple(
        name
        for name in parameters
        if name.endswith(("cm_param.rho", "rhoa_scale_param.rho"))
    )
    assert changed
    originals = dict(target.named_parameters(remove_duplicate=False))
    old_values = {name: value.detach().clone() for name, value in originals.items()}
    versions = {name: value._version for name, value in originals.items()}
    flags = {name: value.requires_grad for name, value in originals.items()}
    geometry_owner = target if geometry_owner is None else geometry_owner
    geometry = getattr(geometry_owner, geometry_name)
    original_geometry = geometry
    geometry_version = geometry._version
    with torch.no_grad():
        for index, name in enumerate(changed):
            parameters[name].add_(0.07 + 0.02 * index)
        if geometry_key is not None:
            constants[geometry_key].mul_(1.1)
        initial = dn.func.Continuation(tensors.state, parameters, constants)
        _, prefix = dn.func.run(functional, initial, steps=PREFIX)
        continuation = prefix["continuation"]
        expected, _ = dn.func.run(functional, continuation, steps=SUFFIX)

        assert dn.func.commit_continuation_(functional, target, continuation) is target
        _close(functional.extract(target).state, continuation.state)
        assert target.integrator.initialized
        # Equal raw sources retain their versions; changed sources preserve
        # identity and leave the newly installed workspace coherent.
        assert getattr(geometry_owner, geometry_name) is original_geometry
        if geometry_key is None:
            assert geometry._version == geometry_version
        else:
            assert geometry._version > geometry_version
        current = dict(target.named_parameters(remove_duplicate=False))
        assert current.keys() == originals.keys()
        for name, value in current.items():
            assert value is originals[name]
            assert value.requires_grad == flags[name]
        for name in changed:
            torch.testing.assert_close(current[name], parameters[name])
            assert not torch.equal(current[name], old_values[name])
            assert current[name]._version > versions[name]
        if geometry_key is not None:
            torch.testing.assert_close(geometry, constants[geometry_key])

        # Public inference execution validates cached Cable/Multi workspaces;
        # it cannot use training's deferred rebuild to hide a stale signature.
        target.run(tstop=SUFFIX * DT, dt=DT)
        _close(functional.extract(target).state, expected)
    return target


@pytest.mark.parametrize("family", ("cable", "tree"))
def test_native_exact_geometry_handoff_rebases_solver_inputs_for_inference(family):
    if family == "tree" and not DENDRA_SOLVERS_AVAILABLE:
        pytest.skip("Tree CPU execution requires dendra-solvers")
    factory = _native_cable if family == "cable" else _scalar_tree
    _handoff_and_continue(factory(), factory())


def _assert_writeback(model):
    offset = 0
    for component in model.populations.values():
        width = math.prod(component.core_shape())
        selected = model.v[..., offset : offset + width].reshape(component.shape)
        torch.testing.assert_close(component.v, selected)
        if isinstance(getattr(component, "vc", None), torch.Tensor):
            layers = component.vc.shape[-1]
            selected_circuit = model.vc[..., offset : offset + width, :].reshape(
                *component.shape, layers
            )
            torch.testing.assert_close(component.vc, selected_circuit)
        offset += width
    assert offset == model.v.shape[-1]


@pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE, reason="packed CPU execution requires dendra-solvers"
)
@pytest.mark.parametrize("block", (False, True), ids=("scalar", "block"))
def test_packed_handoff_preserves_component_sources_and_voltage_writeback(block):
    factory = _block_multi if block else _scalar_multi
    source, target = factory(), factory()
    if block:
        key = "populations.axon.xc"
        geometry_owner = target.populations["axon"]
        geometry_name = "xc"
    else:
        key = "populations.point.dx"
        geometry_owner = target.populations["point"]
        geometry_name = "dx"
    _handoff_and_continue(
        source,
        target,
        geometry_key=key,
        geometry_owner=geometry_owner,
        geometry_name=geometry_name,
    )
    _assert_writeback(target)
