"""SPD cable solves retain native state, sharing, initialization, and replay."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators import implicit
from dendra.models.mod import graded_syn, pas
from dendra.units import nA
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _assert_tensor_snapshot,
    _module_snapshot,
    _tensor_snapshot,
)
from tests.func.test_network import _RAW_WEIGHT
from tests.func.test_network_cache_initialization import _warm_cache

DT = 0.005
DTYPE = torch.float64
PAS_G = "integrator.mech.mechanisms.pas.g_param"
pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        implicit._FUNCTIONAL_SCALAR_SPD is None
        or implicit._FUNCTIONAL_BLOCK_SPD is None,
        reason="dendra-solvers pure scalar and block SPD facades are required",
    ),
]


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=3e-10, atol=3e-11, equal_nan=True)


def _release(voltage):
    return torch.sigmoid(0.03 * (voltage + 65.0))


def _network_snapshot(network):
    return (
        {
            name: _module_snapshot(population)
            for name, population in network.populations.items()
        },
        {
            name: _tensor_snapshot(value)
            for name, value in network.named_parameters(remove_duplicate=False)
        },
        {
            name: _tensor_snapshot(value)
            for name, value in network.named_buffers(remove_duplicate=False)
        },
    )


def _assert_network_snapshot(network, snapshot):
    populations, parameters, buffers = snapshot
    for name, population in network.populations.items():
        _assert_module_snapshot(population, populations[name])
    for name, value in network.named_parameters(remove_duplicate=False):
        _assert_tensor_snapshot(value, parameters[name])
    for name, value in network.named_buffers(remove_duplicate=False):
        _assert_tensor_snapshot(value, buffers[name])


def _population(kind, *, count=3, batch=None, shared_g=None):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True, DTYPE=DTYPE, DEVICE="cpu"):
        initial = torch.linspace(-66.0, -62.0, count, dtype=DTYPE)
        if kind == "scalar":
            population = dn.Unmyelinated(
                [2.0],
                L=10.0 * count,
                dx=10.0,
                v_init=initial,
                integrator=dn.bwd_euler_ub(method="spd"),
                dtype=DTYPE,
            )
        else:
            population = dn.ExtCellAxon(
                [5.0],
                n_comp=count,
                v_init=initial,
                integrator=dn.bwd_euler_bt(method="spd"),
                dtype=DTYPE,
            )
            population.xraxial.fill_(2.0)
            population.xc.fill_(0.12)
            population.xg.fill_(0.0002)
        population.insert(pas, g=0.001 if shared_g is None else shared_g, e=-70.0)
        if batch is not None:
            population.batch(batch)
        population.train()
        population.initialize()
    return population


def _network(*, training=False, cached=False):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True, DTYPE=DTYPE, DEVICE="cpu"):
        shared_g = torch.nn.Parameter(torch.tensor(0.001, dtype=DTYPE))
        cable = _population("scalar", shared_g=shared_g)
        shell = _population("block", shared_g=shared_g)
        shell.insert(graded_syn.rename("syn"), e=0.0)
        cable[:, 1].inject(dn.mono_rect(amp=0.1 * nA, pw=0.03))
        network = dn.Network({"cable": cable, "shell": shell})
        network.connect_continuous_one_to_one(
            cable[:],
            shell[:],
            shell.mech.syn,
            pre_var="v",
            input="g_pre",
            weight=0.0001,
            delay=DT,
            transform=_release,
        )
        network.train(training)
        network.initialize(DT)
        if cached:
            _warm_cache(network)
            network.initialize(DT)
    assert cable.mech.pas.g_param is shell.mech.pas.g_param
    return network


def _state_loss(state):
    return sum(
        population["integrator"]["v"].square().mean()
        + (
            0.01 * population["integrator"]["vc"].square().mean()
            if "vc" in population["integrator"]
            else 0.0
        )
        for population in state["populations"].values()
    )


def _native_loss(network):
    return (
        network.cable.v.square().mean()
        + network.shell.v.square().mean()
        + 0.01 * network.shell.vc.square().mean()
    )


@pytest.mark.parametrize("kind", ["scalar", "block"])
@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("batch", [None, 2])
def test_spd_population_steps_preserve_all_native_leaves_and_input_ownership(
    kind, count, batch
):
    source = _population(kind, count=count, batch=batch)
    native = _population(kind, count=count, batch=batch)
    before = _module_snapshot(source)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    state = tensors.state
    dt = tensors.constants["dt"]
    native.integrator._initialize(native, dt, force=True, compile_scope="population")
    for index in range(3):
        intra = source.v.new_full(source.shape, (index + 1) * 1e-9)
        incoming = _clone(state)
        actual, _ = functional.step(
            tensors.parameters, prepared, state, dn.func.StepInput(intra=intra)
        )
        _close(state, incoming)
        native.integrator.step(native, dt, intra=intra)
        native.t = native.t + dt
        _close(actual, functional.extract(native).state)
        state = actual
    assert not any("chol" in key for key in source.state_dict())
    _assert_module_snapshot(source, before)


@pytest.mark.parametrize(
    "execution,initialization,cached,backend",
    [
        ("scan", "snapshot", False, "aot_eager"),
        ("while", "snapshot", False, "aot_eager"),
        ("scan", "fresh", False, "aot_eager"),
        ("while", "fresh", False, "aot_eager"),
        ("scan", "fresh", True, "aot_eager"),
        ("while", "fresh", True, "aot_eager"),
        ("scan", "snapshot", False, "inductor"),
        ("while", "snapshot", False, "inductor"),
    ],
)
def test_spd_network_recording_initialization_restart_and_native_checkpoint(
    execution, initialization, cached, backend
):
    source = _network(cached=cached)
    native = _network(cached=cached)
    before = _network_snapshot(source)
    runner = dn.compile(
        source,
        dt=DT,
        steps=3,
        initialization=initialization,
        inference=True,
        execution=execution,
        backend=backend,
        callbacks={"trace": dn.func.Recorder(("cable.v", "shell.v", "t"))},
    )
    frozen = _clone(runner.initial_state)
    if initialization == "fresh":
        native.initialize(DT)
    expected = {
        "cable.v": [native.cable.v.detach().clone()],
        "shell.v": [native.shell.v.detach().clone()],
        "t": [native.t.clone()],
    }
    with torch.no_grad():
        for _ in range(6):
            native.step()
            for name, value in (
                ("cable.v", native.cable.v),
                ("shell.v", native.shell.v),
                ("t", native.t),
            ):
                expected[name].append(value.detach().clone())
    prefix, first = runner()
    parameters = first.get("parameters", runner.parameters)
    constants = first.get("constants", runner.constants)
    repeated, repeated_aux = runner()
    _close(prefix, repeated)
    _close(first["callbacks"].outputs, repeated_aux["callbacks"].outputs)
    final, second = runner(
        prefix,
        parameters=parameters,
        constants=constants,
        callback_state=first["callbacks"].state,
    )
    _close(final, runner.functional.extract(native).state)
    for name, frames in expected.items():
        trace = torch.cat(
            (first["callbacks"]["trace"][name], second["callbacks"]["trace"][name])
        )
        _close(trace, torch.stack(frames))
    receiver = _network(cached=cached)
    runner.functional.commit_state_(receiver, final)
    checkpoint = receiver.state_dict_for_checkpoint()
    resumed = _network(cached=cached)
    resumed.restore_dict_from_checkpoint(checkpoint)
    with torch.no_grad():
        resumed.step()
        continued, _ = runner.functional.prepare_and_step(parameters, constants, final)
    _close(continued, runner.functional.extract(resumed).state)
    _close(runner.initial_state, frozen)
    _assert_network_snapshot(source, before)


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
@pytest.mark.parametrize("checkpoint_every", [None, 2])
def test_spd_network_shared_parameter_and_physical_gradients_match_native(
    backend, checkpoint_every
):
    source, native = _network(training=True), _network(training=True)
    before = _network_snapshot(source)
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        inference=False,
        backend=backend,
        checkpoint_every=checkpoint_every,
        trainable=(
            "*pas.g_param",
            "*cm_param*",
            "*rhoa_param*",
            f"continuous_synapses.*.{_RAW_WEIGHT}",
        ),
    )
    actual, _ = runner()
    for _ in range(4):
        native.step()
    _close(actual, runner.functional.extract(native).state)
    connection = next(iter(source.continuous_synapses))
    names = (
        f"populations.cable.{PAS_G}",
        "populations.cable.cm_param",
        "populations.cable.rhoa_param",
        "populations.shell.cm_param",
        "populations.shell.rhoa_param",
        f"continuous_synapses.{connection}.{_RAW_WEIGHT}",
    )
    targets = tuple(runner.get_parameter(name) for name in names)
    assert targets[0] is runner.get_parameter(f"populations.shell.{PAS_G}")
    native_targets = (
        native.cable.mech.pas.g_param,
        native.cable.cm_param,
        native.cable.rhoa_param,
        native.shell.cm_param.rho,
        native.shell.rhoa_param.rho,
        dict(native.continuous_synapses[connection].named_parameters())[_RAW_WEIGHT],
    )
    expected_gradients = torch.autograd.grad(_native_loss(native), native_targets)
    gradients = torch.autograd.grad(_state_loss(actual), targets)
    _close(gradients, expected_gradients)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
    )
    _assert_network_snapshot(source, before)


@pytest.mark.parametrize("kind", ["scalar", "block"])
def test_spd_population_owned_cache_reinitializes_at_changed_timestep(kind):
    source = _population(kind)
    dt = source.v.new_tensor(DT)
    source.integrator._initialize(source, dt, force=True, compile_scope="population")
    for _ in range(2):
        source.integrator.step(source, dt)
        source.t = source.t + dt
    source.cache("_steady_state")
    native = copy.deepcopy(source)
    functional, tensors = dn.func.make_functional(source, dt=2 * DT)
    initialized = functional.initialize(
        tensors.parameters, tensors.constants, tensors.initialization
    )
    native.initialize()
    _close(initialized.state, functional.extract(native).state)
    actual, _ = functional.prepare_and_step(
        initialized.parameters, initialized.constants, initialized.state
    )
    dt = initialized.constants["dt"]
    native.integrator._initialize(native, dt, force=True, compile_scope="population")
    native.integrator.step(native, dt)
    native.t = native.t + dt
    _close(actual, functional.extract(native).state)
