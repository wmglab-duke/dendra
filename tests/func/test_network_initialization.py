"""Native-ordered pure Network initialization and parameter transactions."""

import gc
import weakref

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mod import pas
from tests.func.test_network import DT, DTYPE, _clone, _close, _mixed_network

pytestmark = pytest.mark.cpu

PARAMETER = "integrator.mech.materials.pool._initial_sources.amount"


class _Add(torch.nn.Module):
    def forward(self, value, amount):
        return (value + amount,)


class _Multiply(torch.nn.Module):
    def forward(self, value, factor):
        return (value * factor,)


class _RecordParameter(torch.nn.Module):
    def forward(self, voltage, parameter):
        return (voltage + 100 * parameter,)


def _shared_network():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE, DEVICE="cpu"):
        cells = {}
        shared = torch.nn.Parameter(torch.tensor([0.001], dtype=DTYPE))
        for name in ("left", "right"):
            cell = dn.SingleCompartment(
                N=1, C=1, integrator=dn.bwd_euler_sc(imem=False), v_init=-65.0
            )
            cell.insert(pas, g=0.001, e=-65.0)
            cell.material(
                "pool",
                fields={"amount": shared},
                min_values={"amount": 0.0},
                conserved={"amount": False},
            )
            cell.build()
            cells[name] = cell
        for name, operation, inputs in (
            ("left", _Add(), {"amount": torch.tensor(0.0001, dtype=DTYPE)}),
            ("right", _Multiply(), {"factor": torch.tensor(2.0, dtype=DTYPE)}),
        ):
            cells[name].register_pre_initialize_transform(
                "adjust",
                operation,
                reads=(f"parameters.{PARAMETER}",),
                writes=(f"parameters.{PARAMETER}",),
                inputs=inputs,
            )
            cells[name].register_post_initialize_transform(
                "observe",
                _RecordParameter(),
                reads=("state.integrator.v", f"parameters.{PARAMETER}"),
                writes=("state.integrator.v",),
            )
        model = dn.Network(cells)
        model.train()
        model.initialize(DT)
    return model


@pytest.mark.parametrize("batch", (None, 2))
@pytest.mark.parametrize("training", (False, True))
def test_mixed_initialization_resets_complete_boundary_and_matches_native(
    batch, training
):
    model, native = _mixed_network(batch=batch), _mixed_network(batch=batch)
    if not training:
        for network in (model, native):
            network.eval()
            network.initialize(DT, reinit_weights=False, reinit_delays=False)
    for _ in range(4):
        model.step()
    model._set_duration_remainder(0.07)
    functional, tensors = dn.func.make_functional_network(model)
    before = _clone(tensors, detached=True)
    initialized = functional.initialize(
        *tensors,
        functional.initialization_inputs._replace(
            t=functional.initialization_inputs.t.new_tensor(-0.2)
        ),
    )
    native.initialize(DT, reinit_weights=False, reinit_delays=False, t=-0.2)
    _close(initialized.state, functional.extract(native).state)
    _close(tensors, before)
    assert all(
        value["delivery_buffer"].eq(0).all()
        for value in initialized.state["synapses"].values()
    )
    actual, _ = functional.prepare_and_step(*initialized)
    native.step()
    _close(actual, functional.extract(native).state)


def test_population_order_publishes_each_parameter_write_to_every_alias():
    model, native = _shared_network(), _shared_network()
    functional, tensors = dn.func.make_functional_network(model)
    initial = tensors.parameters["populations"]["left"][PARAMETER]
    assert initial is tensors.parameters["populations"]["right"][PARAMETER]
    before = initial.detach().clone()
    initialized = functional.initialize(*tensors, functional.initialization_inputs)
    left = initialized.parameters["populations"]["left"][PARAMETER]
    right = initialized.parameters["populations"]["right"][PARAMETER]
    assert left is right
    torch.testing.assert_close(left, (before + 0.0001) * 2)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    reference = functional.extract(native)
    _close(initialized.state, reference.state)
    _close(initialized.parameters, reference.parameters)
    torch.testing.assert_close(initial, before)


def test_shared_parameter_replacements_keep_history_and_initialization_gradients():
    functional, tensors = dn.func.make_functional_network(_shared_network())
    defaults = functional.initialization_inputs
    source = tensors.parameters["populations"]["left"][PARAMETER]
    amount = torch.tensor(0.0003, dtype=DTYPE, requires_grad=True)
    factor = torch.tensor(1.5, dtype=DTYPE, requires_grad=True)
    populations = dict(defaults.populations)
    populations["left"] = populations["left"]._replace(
        transforms={"pre.adjust.amount": amount}
    )
    populations["right"] = populations["right"]._replace(
        transforms={"pre.adjust.factor": factor}
    )
    initialized = functional.initialize(
        *tensors, defaults._replace(populations=populations)
    )
    effective = initialized.parameters["populations"]["left"][PARAMETER]
    assert effective is initialized.parameters["populations"]["right"][PARAMETER]
    loss = effective.sum() + sum(
        value["integrator"]["v"].sum()
        for value in initialized.state["populations"].values()
    )
    gradient = torch.autograd.grad(loss, (source, amount, factor), create_graph=True)
    torch.testing.assert_close(
        gradient[0], torch.ones_like(source) * (100 + 101 * factor)
    )
    torch.testing.assert_close(gradient[1], amount.new_tensor(100) + 101 * factor)
    torch.testing.assert_close(gradient[2], 101 * (source + amount).sum())
    cross = torch.autograd.grad(gradient[1], factor)[0]
    torch.testing.assert_close(cross, factor.new_tensor(101))


def test_initialization_defaults_are_independent_and_do_not_retain_result_graphs():
    functional, tensors = dn.func.make_functional_network(_shared_network())
    defaults = functional.initialization_inputs
    snapshots = _clone(defaults, detached=True)
    original = _clone(tensors, detached=True)
    first = functional.initialize(*tensors, defaults)
    second = functional.initialize(*tensors, defaults)
    _close(first, second)
    _close(defaults, snapshots)
    _close(tensors, original)
    references = [
        weakref.ref(value)
        for value in pytree.tree_leaves((first, second))
        if torch.is_tensor(value) and value.grad_fn is not None
    ]
    assert references
    del first, second
    gc.collect()
    assert all(reference() is None for reference in references)
    for value in pytree.tree_leaves(defaults):
        if torch.is_tensor(value):
            assert value.grad_fn is None


def test_shared_runtime_parameter_writes_before_first_reader_match_native():
    from tests.func.test_network_fresh_compile import _ScaleParameter

    def model():
        with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE, DEVICE="cpu"):
            populations = {}
            for name in ("left", "right"):
                cell = dn.SingleCompartment(
                    N=1, C=1, integrator=dn.bwd_euler_sc(imem=False), v_init=-60.0
                )
                cell.insert(pas, g=0.001, e=-65.0)
                cell.build()
                populations[name] = cell
            populations["right"].cm_scale_param.rho = populations[
                "left"
            ].cm_scale_param.rho
            network = dn.Network(populations)
            network.train()
            network.initialize(DT)
            populations["left"].register_pre_initialize_transform(
                "scale",
                _ScaleParameter(),
                reads=("parameters.cm_scale_param.rho",),
                writes=("parameters.cm_scale_param.rho",),
                inputs={"factor": torch.tensor(1.5, dtype=DTYPE)},
            )
        return network

    source, native = model(), model()
    functional, tensors = dn.func.make_functional_network(source)
    initialized = functional.initialize(*tensors, functional.initialization_inputs)
    assert (
        initialized.parameters["populations"]["left"]["cm_scale_param.rho"]
        is initialized.parameters["populations"]["right"]["cm_scale_param.rho"]
    )
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    _close(initialized.state, functional.extract(native).state)
    state, _ = functional.prepare_and_step(*initialized)
    native.step()
    _close(state, functional.extract(native).state)


@pytest.mark.parametrize("packet", ("parameters", "constants", "state", "inputs"))
def test_initialization_normalizes_mapping_access_before_source_validation(packet):
    from tests.func.test_network_mapping_boundaries import _OnItems

    model = _shared_network()
    functional, tensors = dn.func.make_functional_network(model)
    packets = dict(
        parameters=tensors.parameters,
        constants=tensors.constants,
        state=tensors.state,
        inputs=functional.initialization_inputs,
    )
    if packet == "inputs":
        packets[packet] = packets[packet]._replace(
            populations=_OnItems(packets[packet].populations, model.eval)
        )
    else:
        packets[packet] = _OnItems(packets[packet], model.eval)
    with pytest.raises(dn.func.FunctionalizationError):
        functional.initialize(**packets)
