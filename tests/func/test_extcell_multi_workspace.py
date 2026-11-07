"""Packed block workspaces cannot outlive changed component physical fields."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.extcell import gather_extcell
from dendra.models.integrators import dhs_bt
from dendra.models.integrators.tree_bt import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mod import pas
from dendra.models.tree import gather_morphology
from tests.func.test_extcell_multi_population import (
    DT,
    _drives,
    _graph,
    _model,
    _seed_circuit,
)

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        not DENDRA_SOLVERS_AVAILABLE,
        reason="packed ExtCell CPU execution requires dendra-solvers",
    ),
]


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("inference_tensors", [False, True])
@pytest.mark.parametrize(
    "field", ("cm", "rhoa", "cm_scale", "rhoa_scale", "area_scale", "diam", "dx")
)
def test_changed_legacy_extcell_axon_physics_requires_packed_initialize(
    threads, inference_tensors, field
):
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        with (
            dn.ctx(JIT=0, REQUIRE_GRAD=0, IMEM=False),
            torch.inference_mode(inference_tensors),
        ):
            model = _model()
            model.requires_grad_(False)
            model.eval()
            drive, _ = _drives(model, steps=1)
            model.run(ve=drive, dt=DT)
            before = model._integrator_workspace_contract_signature()
            getattr(model.populations["axon"], field).mul_(1.1)
            if not inference_tensors:
                assert model._integrator_workspace_contract_signature() != before
            with pytest.raises(RuntimeError, match="initialize"):
                model.run(ve=drive, dt=DT)
            model.initialize()
            model.run(ve=drive, dt=DT)
            assert torch.isfinite(model.vc).all()
    finally:
        torch.set_num_threads(previous)


class _ScaleDiameter(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.register_buffer("scale", reference.new_ones(()))

    def forward(self, value):
        return value * self.scale


def test_parametrized_child_physics_tracks_stable_inputs_not_derived_view_ids():
    with dn.ctx(JIT=0, REQUIRE_GRAD=0, IMEM=False):
        model = _model()
        model.requires_grad_(False)
        model.eval()
        component = model.populations["axon"]
        transform = _ScaleDiameter(component.diam)
        torch.nn.utils.parametrize.register_parametrization(
            component, "diam", transform
        )
        model.initialize()
        drive, _ = _drives(model, steps=1)
        model.run(ve=drive, dt=DT)
        signature = model._integrator_workspace_contract_signature()
        assert model._integrator_workspace_contract_signature() == signature
        model.run(ve=drive, dt=DT)
        transform.scale.mul_(1.1)
        with pytest.raises(RuntimeError, match="initialize"):
            model.run(ve=drive, dt=DT)
        model.initialize()
        model.run(ve=drive, dt=DT)
        assert torch.isfinite(model.vc).all()


@pytest.mark.parametrize("threads", [1, 4])
def test_low_level_extcell_tree_pack_preserves_graph_fallback_and_gradient(threads):
    """Imperative packs retain the standalone low-level graph constructor."""
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        results = []
        with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64):
            for packed in (False, True):
                # Root 3 precedes neither child in storage; there are two core
                # replicas plus an explicit leading batch of two simulations.
                graph = _graph()
                tree = dn.ExtCellTree(
                    2,
                    len(graph),
                    graph,
                    celsius=6.3,
                    v_init=torch.tensor([-64.0, -61.0, -65.0, -62.0]),
                    dtype=torch.float64,
                    integrator=dhs_bt(threads=2, imem=False),
                )
                tree.load_morphology(gather_morphology(graph))
                tree.load_extcell(gather_extcell(graph))
                tree.rhoa_scale_param.set(1.15)
                tree.insert(pas)
                assert tree.compartment_graph is None
                model = (
                    dn.concat_models({"tree": tree}, threads=2, celsius=6.3)
                    if packed
                    else tree
                )
                model.batch(2)
                model.initialize()
                model.train()
                _seed_circuit(model)
                ve, _ = _drives(model)
                recorder = dn.Recorder(("v", "vc"))
                model.run(ve=ve, dt=DT, callbacks=[recorder])
                recorded = {
                    field: recorder.stack(field).reshape(
                        ve.shape[0] + 1, 2, 2, len(graph), *tail
                    )
                    for field, tail in (("v", ()), ("vc", (3,)))
                }
                loss = (
                    recorded["v"].square().mean()
                    + 0.03 * recorded["vc"].square().mean()
                )
                (gradient,) = torch.autograd.grad(loss, tree.rhoa_scale_param.rho)
                assert torch.isfinite(gradient).all()
                assert torch.count_nonzero(gradient)
                results.append((recorded, gradient))
        expected, actual = results
        for field in ("v", "vc"):
            torch.testing.assert_close(
                actual[0][field], expected[0][field], rtol=3.0e-8, atol=3.0e-9
            )
        torch.testing.assert_close(actual[1], expected[1], rtol=2.0e-7, atol=1.0e-10)
    finally:
        torch.set_num_threads(previous)
