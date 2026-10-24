"""Mixed native topologies share compiled lifecycle, recording, and gradients."""

from __future__ import annotations

import networkx as nx
import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators import dhs_bt
from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mod import expsyn, graded_syn, pas
from dendra.units import nA
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _assert_tensor_snapshot,
    _module_snapshot,
    _tensor_snapshot,
)
from tests.func.test_network import _RAW_WEIGHT, _clone, _close
from tests.func.test_network_cache_initialization import _warm_cache

DT = 0.005
DTYPE = torch.float64
STEPS = 4
GAIN = "integrator.mech.mechanisms.pas.g_param"
pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        not DENDRA_SOLVERS_AVAILABLE, reason="mixed topologies require dendra-solvers"
    ),
]


class _Constructed:
    """A construction mixin exercises inherited geometry and execution hooks."""

    def describe(self):
        return type(self).__name__


class _Cell(_Constructed, dn.SingleCompartment):
    pass


class _Cable(_Constructed, dn.Cable):
    pass


class _Tree(_Constructed, dn.Tree):
    pass


class _Shell(_Constructed, dn.ExtCellAxon):
    pass


class _ShellTree(_Constructed, dn.ExtCellTree):
    pass


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True, DTYPE=DTYPE, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _graph(*, shell=False):
    graph = nx.DiGraph()
    for node in range(3):
        attributes = dict(
            name=f"branch.{node}",
            L=10.0 + node,
            diam=2.0 - 0.3 * node,
            Ra=90.0 + 10 * node,
            cm=0.8 + 0.2 * node,
            area=100.0 + 30 * node,
        )
        if shell:
            attributes.update(xraxial=[1.5, 2.2], xc=[0.08, 0.12], xg=[2e-4, 3e-4])
        graph.add_node(node, **attributes)
    graph.add_edge(1, 0, R_ohm=1.1e8)
    graph.add_edge(1, 2, R_ohm=1.7e8)
    return graph


def _morphology():
    morphology = dn.Morphology(rhoa=90.0, cm=1.0)
    morphology.section("shaft", L=30.0, diam=2.0, nseg=3)
    return morphology


def _release(voltage):
    return torch.sigmoid(0.03 * (voltage + 65.0))


def _network(kind, *, training=False, cache=None, lean=False):
    shared = torch.nn.Parameter(torch.tensor(0.0004, dtype=DTYPE))
    point = _Cell(N=1, C=1, v_init=-63.0, integrator=dn.rk4())
    if kind == "scalar":
        cable = _Cable.from_morphology(
            _morphology(), N=1, v_init=-65.0, integrator=dn.bwd_euler_ub(method="pcr")
        )
        populations = {"point": point, "cable": cable}
        if not lean:
            populations["tree"] = _Tree.from_graph(
                _graph(), N=1, v_init=-67.0, integrator=dn.dhs(threads=1)
            )
    elif kind == "block":
        point = _Cell(N=1, C=1, v_init=-63.0, integrator=dn.euler())
        shell = _Shell([5.0], n_comp=3, v_init=-65.0, integrator=dn.bwd_euler_bt())
        shell.xraxial.fill_(1.8)
        shell.xc.fill_(0.1)
        shell.xg.fill_(2e-4)
        tree = _ShellTree.from_graph(
            _graph(shell=True), N=1, v_init=-67.0, integrator=dhs_bt(threads=1)
        )
        populations = {"point": point, "shell": shell, "tree": tree}
    else:
        point = _Cell(N=1, C=3, v_init=-63.0, integrator=dn.dfh(conv=True, beta=0.8))
        head = _Cell(N=1, C=1, v_init=-65.0)
        shaft = dn.Unmyelinated([2.0], L=30.0, dx=10.0, v_init=-67.0)
        for child in (head, shaft):
            child.insert(pas, g=shared, e=-71.0)
            child[:, 0].label("recorded")
        packed = dn.concat_models({"head": head, "shaft": shaft})
        populations = {"point": point, "packed": packed}
    for name, population in populations.items():
        if name != "packed":
            population.insert(pas, g=shared, e=-71.0)
        population[:, 0].label("recorded")
    receiver = next(reversed(populations.values()))
    receiver.insert(graded_syn.rename("analog"), e=-10.0)
    if not lean:
        receiver.insert(expsyn.rename("events"), tau=0.03, e=0.0)
    point[:, 0].inject(dn.sin(amp=0.05 * nA, freq=0.7, phase=0.3))
    network = dn.Network(populations)
    network.connect_continuous_one_to_one(
        point[:, 0],
        receiver[:, -1],
        receiver.mech.analog,
        input="g_pre",
        transform=_release,
        weight=0.0002,
        delay=DT,
    )
    if not lean:
        network.connect_one_to_one(
            point[:, 0],
            receiver[:, 0],
            receiver.mech.events,
            threshold=0.0,
            weight=0.0001,
            delay=3 * DT,
        )
    network.train(training)
    network.initialize(DT)
    network.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=False,
        diff_spiking=False,
        diff_scheduled_times=False,
        train_delay_backend="dense",
    )
    if not lean:
        next(iter(network.synapses.values())).schedule(
            con_indices=[0, 0], times_ms=[0.0, 3 * DT], weight=1.0
        )
    if cache == "owned":
        # The first owner restores a cache while its different-topology peer
        # follows fresh initialization in the same ordered transaction.
        for _ in range(2):
            with torch.no_grad():
                network.step()
        point.cache("_steady_state")
        network.initialize(DT)
    elif cache == "network":
        _warm_cache(network)
        network.initialize(DT)
    return network


def _snapshots(network):
    return (
        {name: _module_snapshot(pop) for name, pop in network.populations.items()},
        {
            name: _tensor_snapshot(value)
            for name, value in network.named_parameters(remove_duplicate=False)
        },
        {
            name: _tensor_snapshot(value)
            for name, value in network.named_buffers(remove_duplicate=False)
        },
    )


def _assert_ownership(network, snapshots):
    populations, parameters, buffers = snapshots
    for name, population in network.populations.items():
        _assert_module_snapshot(population, populations[name])
    for name, value in network.named_parameters(remove_duplicate=False):
        _assert_tensor_snapshot(value, parameters[name])
    for name, value in network.named_buffers(remove_duplicate=False):
        _assert_tensor_snapshot(value, buffers[name])


def _callbacks(network):
    names = tuple(network.populations)
    callbacks = {
        "trace": dn.func.Recorder((*[f"{name}.v" for name in names], "t")),
        "soma": dn.func.Recorder(
            f"{names[-1]}.v", region=network.populations[names[-1]].recorded
        ),
    }
    if names[-1] == "packed":
        callbacks["component"] = dn.func.Recorder(
            "packed.v", region=network.packed.shaft.recorded
        )
    return callbacks


@pytest.mark.parametrize(
    "kind,initialization,execution,cache,backend",
    (
        ("scalar", "snapshot", "scan", None, "aot_eager"),
        ("scalar", "fresh", "while", None, "aot_eager"),
        ("scalar", "fresh", "scan", "owned", "aot_eager"),
        ("block", "snapshot", "while", None, "aot_eager"),
        ("block", "fresh", "scan", None, "aot_eager"),
        ("block", "fresh", "while", "network", "aot_eager"),
        ("packed", "snapshot", "scan", None, "aot_eager"),
        ("packed", "fresh", "while", "owned", "aot_eager"),
        ("packed", "fresh", "scan", "network", "aot_eager"),
        ("scalar", "snapshot", "while", None, "inductor"),
    ),
)
def test_mixed_network_lifecycle_recording_and_native_checkpoint(
    kind, initialization, execution, cache, backend
):
    def make():
        return _network(kind, cache=cache, lean=backend == "inductor")

    source, native = make(), make()
    before = _snapshots(source)
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        initialization=initialization,
        inference=True,
        execution=execution,
        backend=backend,
        callbacks=_callbacks(source),
    )
    frozen = _clone(runner.initial_state, detached=True)
    source_packet = _clone(runner.functional.extract(source), detached=True)
    if initialization == "fresh":
        native.initialize(DT)
    frames = {f"{name}.v": [pop.v.clone()] for name, pop in native.populations.items()}
    frames["t"] = [native.t.clone()]
    with torch.no_grad():
        for index in range(2 * STEPS):
            native.step()
            for name, pop in native.populations.items():
                frames[f"{name}.v"].append(pop.v.clone())
            frames["t"].append(native.t.clone())
            if index == STEPS - 1:
                middle = _clone(runner.functional.extract(native).state, detached=True)
    prefix, first = runner()
    _close(prefix, middle)
    if backend != "inductor" and cache is None:
        assert any(
            bool(torch.count_nonzero(connection["delivery_buffer"]))
            for connection in prefix["synapses"].values()
        ), "checkpoint boundary must retain scheduled traffic for its native suffix"
    assert runner.execution_report().strategy == (
        "compiled_while_loop" if execution == "while" else "compiled_scan"
    )
    repeated, repeated_aux = runner()
    _close(repeated, prefix)
    _close(first["callbacks"].outputs, repeated_aux["callbacks"].outputs)
    parameters = first.get("parameters", runner.parameters)
    constants = first.get("constants", runner.constants)
    final, second = runner(
        prefix,
        parameters=parameters,
        constants=constants,
        callback_state=first["callbacks"].state,
    )
    _close(final, runner.functional.extract(native).state)
    for name, expected in frames.items():
        trace = torch.cat(
            (first["callbacks"]["trace"][name], second["callbacks"]["trace"][name])
        )
        _close(trace, torch.stack(expected))
    last = next(reversed(source.populations))
    for outputs in (first["callbacks"], second["callbacks"]):
        full = outputs["trace"][f"{last}.v"]
        _close(
            outputs["soma"][f"{last}.v"].reshape(len(full), -1),
            full.reshape(len(full), -1)[:, :1],
        )
        if kind == "packed":
            # The child remains an addressable physical region of its packed
            # owner while a separate explicit point Population also advances.
            _close(
                outputs["component"]["packed.v"].reshape(len(full), -1),
                full.reshape(len(full), -1)[:, 1:2],
            )
    receiver, resumed = make(), make()
    runner.functional.commit_state_(receiver, prefix)
    resumed.restore_dict_from_checkpoint(receiver.state_dict_for_checkpoint())
    with torch.no_grad():
        for _ in range(STEPS):
            resumed.step()
    _close(final, runner.functional.extract(resumed).state)
    _close(runner.initial_state, frozen)
    _close(runner.functional.extract(source), source_packet)
    _assert_ownership(source, before)


def _loss(state):
    return sum(
        pop["integrator"]["v"].square().mean()
        + (
            0.01 * pop["integrator"]["vc"].square().mean()
            if "vc" in pop["integrator"]
            else 0.0
        )
        for pop in state["populations"].values()
    )


def _native_loss(network):
    return sum(
        pop.v.square().mean()
        + (0.01 * pop.vc.square().mean() if hasattr(pop, "vc") else 0.0)
        for pop in network.populations.values()
    )


@pytest.mark.parametrize(
    "kind,backend",
    (
        ("scalar", "aot_eager"),
        ("block", "aot_eager"),
        ("packed", "aot_eager"),
        ("scalar", "inductor"),
    ),
)
def test_mixed_network_checkpoint_replay_shared_parameter_and_connection_vjps(
    kind, backend
):
    source = _network(kind, training=True, lean=backend == "inductor")
    native = _network(kind, training=True, lean=backend == "inductor")
    before = _snapshots(source)
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=False,
        backend=backend,
        checkpoint_every=2,
        trainable=(
            "*pas.g*",
            "*cm_param*",
            "*stimulation.intra*.amp",
            f"continuous_synapses.*.{_RAW_WEIGHT}",
        ),
    )
    source_packet = _clone(runner.functional.extract(source), detached=True)
    actual, _ = runner()
    native.initialize(DT)
    for _ in range(STEPS):
        native.step()
    _close(actual, runner.functional.extract(native).state)
    assert runner.execution_report().strategy == "compiled_scan"
    connection = next(iter(source.continuous_synapses))
    amplitude = next(
        name
        for name in runner.parameters["populations"]["point"]
        if name.startswith("stimulation.intra.") and name.endswith(".amp")
    )
    names = (
        f"populations.point.{GAIN}",
        "populations.point.cm_param",
        f"populations.point.{amplitude}",
        f"continuous_synapses.{connection}.{_RAW_WEIGHT}",
    )
    targets = tuple(runner.get_parameter(name) for name in names)
    # Packed mechanisms retain an unused declaration-default g_param beside
    # the actual component bindings. Check the bindings that share in native
    # execution rather than treating every pas field as one parameter.
    shared_aliases = [
        runner.get_parameter(
            f"populations.{name}.integrator.mech.mechanisms.pas."
            + ("g_head" if name == "packed" else "g_param")
        )
        for name in source.populations
    ]
    assert len(shared_aliases) >= 2
    assert all(value is targets[0] for value in shared_aliases)
    if kind == "packed":
        assert native.packed.mech.pas.g_head is native.point.mech.pas.g_param
        assert native.packed.mech.pas.g_shaft is native.point.mech.pas.g_param
    native_targets = (
        native.point.mech.pas.g_param,
        native.point.cm_param.rho,
        native.point.intra.stims[0].amp,
        dict(native.continuous_synapses[connection].named_parameters())[_RAW_WEIGHT],
    )
    actual_gradients = torch.autograd.grad(_loss(actual), targets)
    expected_gradients = torch.autograd.grad(_native_loss(native), native_targets)
    _close(actual_gradients, expected_gradients)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0
        for value in actual_gradients
    )
    _assert_ownership(source, before)
    _close(runner.functional.extract(source), source_packet)
