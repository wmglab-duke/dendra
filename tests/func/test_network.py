"""Standard Networks lower by composing their existing pure component plans."""

from __future__ import annotations

import inspect

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mechanisms import Mechanism
from dendra.models.mod import expsyn, graded_syn, pas
from tests.networks.test_functional_event_delivery import ThresholdRelay

pytestmark = pytest.mark.cpu
DT = 0.125
DTYPE = torch.float64
_RAW_WEIGHT = "weight.weights.0.param.rho"


class _Release(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.7, dtype=DTYPE))
        self.bias = torch.nn.Parameter(torch.tensor(-0.2, dtype=DTYPE))
        self.register_buffer("offset", torch.tensor(0.03, dtype=DTYPE))

    def forward(self, value):
        return torch.sigmoid(self.scale * value + self.bias) + self.offset


def _mixed_network(*, batch=None, explicit_input=True):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        cell = dn.SingleCompartment(
            N=2, C=1, integrator=dn.bwd_euler_sc(imem=False), v_init=-65.0
        )
        cell.insert(pas, g=0.001, e=-65.0)
        cell.insert(ThresholdRelay.rename("relay"))
        cell.insert(graded_syn.rename("analog"), e=-10.0)
        network = dn.Network(
            {"cell": cell},
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        for field, threshold, weight in (("level", 0.0, 0.2), ("mirror", 1.5, 0.3)):
            network.connect_one_to_one(
                cell[:],
                cell[:],
                cell.mech.relay,
                pre_var=f"mech.relay.{field}",
                threshold=threshold,
                weight=weight,
                delay=DT,
                allow_autapses=True,
            )
        for field, delay, weight, transform in (
            ("level", [0.0, 2 * DT], [0.002, 0.003], _Release()),
            ("mirror", [DT, 3 * DT], [0.001, 0.004], torch.sigmoid),
        ):
            network.connect_continuous_one_to_one(
                cell[:],
                cell[:],
                cell.mech.analog,
                pre_var=f"mech.relay.{field}",
                input="g_pre" if explicit_input else None,
                weight=torch.tensor(weight, dtype=DTYPE),
                delay=torch.tensor(delay, dtype=DTYPE),
                transform=transform,
                allow_autapses=True,
            )
        if batch is not None:
            network.batch(batch)
        network.train()
        network.initialize(DT)
        network.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=True,
            diff_scheduled_times=False,
            tau=0.23,
            train_delay_backend="dense",
        )
    return network


def _stim_network():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        cell = dn.SingleCompartment(N=1, C=1, integrator=dn.bwd_euler_sc(imem=False))
        cell.insert(pas, g=0.001, e=-65.0)
        cell.insert(expsyn.rename("syn"), tau=1.2, e=0.0)
        stimulus = dn.NetStim(
            N=1, interval=0.5, start=0.0, noise=0.6, max_spikes=20, seed=812
        )
        network = dn.Network(
            {"cell": cell},
            netstim=stimulus,
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        network.connect_one_to_one(
            stimulus[:], cell[:], cell.mech.syn, weight=0.003, delay=DT
        )
        network.train()
        network.initialize(DT)
        network.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=True,
            diff_scheduled_times=False,
            train_delay_backend="dense",
        )
    return network


def _clone(tree, *, detached=False):
    return pytree.tree_map_only(
        torch.Tensor,
        lambda value: value.detach().clone() if detached else value.clone(),
        tree,
    )


def _close(actual, expected):
    left, spec = pytree.tree_flatten(actual)
    right, other = pytree.tree_flatten(expected)
    assert spec == other
    for value, reference in zip(left, right, strict=True):
        if torch.is_tensor(value):
            torch.testing.assert_close(value, reference, rtol=2e-8, atol=2e-9)
        else:
            assert value == reference


def _execution(functional, tensors, compiled):
    if compiled:
        # Opaque Population preparation stays inside the compiled region.
        return (
            torch.compile(
                functional.prepare_and_step, backend="aot_eager", fullgraph=True
            ),
            tensors.constants,
        )
    return functional.step, functional.prepare(tensors.parameters, tensors.constants)


def _loss(state):
    cell = state["populations"]["cell"]
    relay = cell["mechanism_buffers"]["relay"]
    return (
        cell["integrator"]["v"].square().sum()
        + relay["level"].square().sum()
        + 0.01 * relay["ledger"].square().sum()
        + sum(
            queue["delivery_buffer"].square().sum()
            for queue in state["continuous_synapses"].values()
        )
    )


def _native_loss(network):
    return (
        network.cell.v.square().sum()
        + network.cell.mech.relay.level.square().sum()
        + 0.01 * network.cell.mech.relay.ledger.square().sum()
        + sum(
            connection.delivery_buffer.square().sum()
            for connection in network.continuous_synapses.values()
        )
    )


@pytest.mark.parametrize("compiled", (False, True))
@pytest.mark.parametrize("batch", (None, 2))
def test_mixed_recurrent_network_native_order_gradients_and_no_source_mutation(
    compiled, batch
):
    model, native = _mixed_network(batch=batch), _mixed_network(batch=batch)
    model._reset_runtime_clock(3.25)
    native._reset_runtime_clock(3.25)
    model._set_duration_remainder(0.4 * DT)
    native._set_duration_remainder(0.4 * DT)
    functional, tensors = dn.func.make_functional_network(model)
    frozen = _clone(tensors.state, detached=True)
    state = _clone(tensors.state)
    relay = state["populations"]["cell"]["mechanism_buffers"]["relay"]
    initial_level = relay["level"].detach().clone().requires_grad_()
    relay["level"] = initial_level
    native_level = initial_level.detach().clone().requires_grad_()
    native.cell.mech.relay.level = native_level
    execute, prepared = _execution(functional, tensors, compiled)
    for index in range(7):
        incoming = state
        before = _clone(incoming, detached=True)
        state, _ = execute(tensors.parameters, prepared, incoming)
        native.step()
        _close(incoming, before)
        _close(state, functional.extract(native).state)
        if index == 0:
            # Event sources must be observed after each receiver transaction.
            observed = state["populations"]["cell"]["mechanism_buffers"]["relay"]
            torch.testing.assert_close(
                observed["level"], torch.full_like(initial_level, 1.5)
            )
            assert all(
                queue["has_spiked"].all() for queue in state["synapses"].values()
            )
    event_weights = tuple(
        values[_RAW_WEIGHT] for values in tensors.parameters["synapses"].values()
    )
    continuous_weights = tuple(
        values[_RAW_WEIGHT]
        for values in tensors.parameters["continuous_synapses"].values()
    )
    transform_parameters = tuple(
        value
        for values in tensors.parameters["continuous_synapses"].values()
        for value in (values["transform_parameters"] or {}).values()
        if value.requires_grad
    )
    actual = torch.autograd.grad(
        _loss(state),
        (initial_level, *event_weights, *continuous_weights, *transform_parameters),
    )
    native_inputs = (
        native_level,
        *(
            connection.weight.weights[0].param.rho
            for connection in native.synapses.values()
        ),
        *(
            connection.weight.weights[0].param.rho
            for connection in native.continuous_synapses.values()
        ),
        *(
            value
            for connection in native.continuous_synapses.values()
            if isinstance(connection.transform, torch.nn.Module)
            for value in connection.transform.parameters()
        ),
    )
    expected = torch.autograd.grad(_native_loss(native), native_inputs)
    _close(actual, expected)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in actual
    )
    _close(tensors.state, frozen)
    _close(functional.extract().state, frozen)


def test_split_checkpoint_preserves_prefix_gradients_and_commits_a_complete_boundary():
    model, native, resumed = _mixed_network(), _mixed_network(), _mixed_network()
    functional, tensors = dn.func.make_functional_network(model)
    state = tensors.state
    saved = None
    trajectory = []
    for index in range(7):
        state, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, state
        )
        native.step()
        trajectory.append(_clone(state))
        if index == 2:
            saved = _clone(state)
    assert saved is not None
    replay = saved
    for index in range(3, 7):
        replay, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, replay
        )
        _close(replay, trajectory[index])
    weights = tuple(
        values[_RAW_WEIGHT] for values in tensors.parameters["synapses"].values()
    )
    gradients = torch.autograd.grad(_loss(state), weights, retain_graph=True)
    replay_gradients = torch.autograd.grad(_loss(replay), weights)
    _close(gradients, replay_gradients)
    assert functional.commit_state_(resumed, saved) is resumed
    _close(functional.extract(resumed).state, saved)
    for index in range(3, 7):
        resumed.step()
        _close(functional.extract(resumed).state, trajectory[index])
    _close(functional.extract(native).state, state)


def test_netstim_draws_and_rng_checkpoint_continuation_match_native_network():
    model, native, resumed = _stim_network(), _stim_network(), _stim_network()
    for network in (native, resumed):
        network.netstim.restore_dict_from_checkpoint(
            _clone(model.netstim.state_dict_for_checkpoint(), detached=True)
        )
    functional, tensors = dn.func.make_functional_network(model)
    frozen = _clone(tensors.state, detached=True)
    state = tensors.state
    saved = None
    for index in range(12):
        draw = functional.sample_step_draw(state)
        state, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, state, netstim_draw=draw
        )
        native.step()
        _close(state, functional.extract(native).state)
        if index == 4:
            saved = _clone(state)
    assert saved is not None
    replay = saved
    for _ in range(5, 12):
        draw = functional.sample_step_draw(replay)
        replay, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, replay, netstim_draw=draw
        )
    _close(replay, state)
    functional.commit_state_(resumed, saved)
    for _ in range(5, 12):
        resumed.step()
    _close(functional.extract(resumed).state, state)
    _close(tensors.state, frozen)
    _close(functional.extract().state, frozen)


def test_shared_connection_parameters_keep_one_explicit_gradient_leaf():
    model = _mixed_network()
    connections = tuple(model.synapses.values())
    connections[1].weight.weights[0].param = connections[0].weight.weights[0].param
    model.initialize(DT)
    functional, tensors = dn.func.make_functional_network(model)
    weights = tuple(
        values[_RAW_WEIGHT] for values in tensors.parameters["synapses"].values()
    )
    assert weights[0] is weights[1]
    assert (
        weights[0].data_ptr() != connections[0].weight.weights[0].param.rho.data_ptr()
    )
    state = tensors.state
    for _ in range(5):
        state, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, state
        )
    (gradient,) = torch.autograd.grad(_loss(state), (weights[0],))
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize("change", ("wiring", "mode", "population_order"))
def test_changed_native_network_configuration_invalidates_existing_plan(change):
    model = _mixed_network()
    functional, tensors = dn.func.make_functional_network(model)
    if change == "wiring":
        model.connect_one_to_one(
            model.cell[:],
            model.cell[:],
            model.cell.mech.relay,
            pre_var="mech.relay.ledger",
            threshold=None,
            weight=0.1,
            delay=DT,
            allow_autapses=True,
        )
    elif change == "mode":
        model.eval()
    else:
        model._population_step_items = ()
    with pytest.raises(dn.func.FunctionalizationError):
        functional.prepare_and_step(
            tensors.parameters, tensors.constants, tensors.state
        )


def test_invalid_component_commit_is_atomic_for_the_native_network():
    model, target = _mixed_network(), _mixed_network()
    functional, tensors = dn.func.make_functional_network(model)
    state, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state
    )
    frozen = functional.extract(target).state
    bad = _clone(state)
    second = tuple(bad["synapses"])[1]
    bad["synapses"][second]["delivery_buffer"] = torch.empty(0, dtype=DTYPE)
    with pytest.raises((ValueError, dn.func.FunctionalizationError)):
        functional.commit_state_(target, bad)
    _close(functional.extract(target).state, frozen)


def test_unconnected_continuous_inputs_are_not_reset_by_network_step():
    def make():
        with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
            cell = dn.SingleCompartment(N=1, C=1, integrator=dn.scnv(imem=False))
            cell.insert(pas, g=0.001, e=-65.0)
            cell.insert(graded_syn.rename("orphan"), e=-10.0)
            network = dn.Network({"cell": cell})
            network.train()
            network.initialize(DT)
            cell.mech.orphan.g_pre = torch.full_like(cell.mech.orphan.g_pre, 0.002)
            cell.mech.orphan.g_pre_old = torch.full_like(
                cell.mech.orphan.g_pre_old, 0.004
            )
        return network

    model, native = make(), make()
    functional, tensors = dn.func.make_functional_network(model)
    before = _clone(tensors.state, detached=True)
    state, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state
    )
    native.step()
    orphan = state["populations"]["cell"]["mechanism_buffers"]["orphan"]
    previous = before["populations"]["cell"]["mechanism_buffers"]["orphan"]
    _close(
        {name: orphan[name] for name in ("g_pre", "g_pre_old")},
        {name: previous[name] for name in ("g_pre", "g_pre_old")},
    )
    _close(state, functional.extract(native).state)


@pytest.mark.parametrize("compiled", (False, True))
def test_netstim_surrogate_gradients_reach_connected_population(compiled):
    model, native = _stim_network(), _stim_network()
    for network in (model, native):
        network.netstim.noise.requires_grad_()
        network.netstim.tau = torch.tensor(0.17, dtype=DTYPE, requires_grad=True)
    functional, tensors = dn.func.make_functional_network(model)
    execute, prepared = _execution(functional, tensors, compiled)
    state = tensors.state
    functional_loss = torch.zeros((), dtype=DTYPE)
    native_loss = torch.zeros((), dtype=DTYPE)
    for index in range(12):
        state, _ = execute(
            tensors.parameters,
            prepared,
            state,
            netstim_draw=functional.sample_step_draw(state),
        )
        native.step()
        functional_loss = (
            functional_loss
            + (index + 1)
            * state["populations"]["cell"]["integrator"]["v"].square().sum()
        )
        native_loss = native_loss + (index + 1) * native.cell.v.square().sum()
    parameters = tensors.parameters["netstim"]["parameters"]
    weights = next(iter(tensors.parameters["synapses"].values()))[_RAW_WEIGHT]
    actual = torch.autograd.grad(
        functional_loss,
        (parameters["interval.rho"], parameters["noise"], parameters["tau"], weights),
    )
    expected = torch.autograd.grad(
        native_loss,
        (
            native.netstim.interval.rho,
            native.netstim.noise,
            native.netstim.tau,
            next(iter(native.synapses.values())).weight.weights[0].param.rho,
        ),
    )
    _close(actual, expected)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in actual
    )


@pytest.mark.parametrize("compiled", (False, True))
def test_multiple_population_source_bindings_use_explicit_state_and_keep_gradients(
    compiled,
):
    from examples.networks.functional_continuous_connectivity import make_network

    model, native = make_network(), make_network()
    functional, tensors = dn.func.make_functional_network(model)
    state = _clone(tensors.state)
    source_values, native_values = [], []
    for index, name in enumerate(("pre_a", "pre_b")):
        value = torch.tensor(
            [[0.2 + index], [-0.3 + index]], dtype=DTYPE, requires_grad=True
        )
        reference = value.detach().clone().requires_grad_()
        state["populations"][name]["mechanism_buffers"]["source"]["gate"] = value
        native.populations[name].mech.source.gate = reference
        source_values.append(value)
        native_values.append(reference)
    execute, prepared = _execution(functional, tensors, compiled)
    for _ in range(6):
        state, _ = execute(tensors.parameters, prepared, state)
        native.step()
        _close(state, functional.extract(native).state)
    loss = state["populations"]["post"]["integrator"]["v"].square().sum()
    gradients = torch.autograd.grad(loss, source_values)
    references = torch.autograd.grad(native.post.v.square().sum(), native_values)
    _close(gradients, references)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
    )


def test_default_continuous_input_is_resolved_like_the_native_connection():
    model, native = (
        _mixed_network(explicit_input=False),
        _mixed_network(explicit_input=False),
    )
    functional, tensors = dn.func.make_functional_network(model)
    state = tensors.state
    for _ in range(5):
        state, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, state
        )
        native.step()
        _close(state, functional.extract(native).state)


def test_incoherent_population_checkpoint_clock_rejects_atomically():
    model, target = _mixed_network(), _mixed_network()
    functional, tensors = dn.func.make_functional_network(model)
    state, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state
    )
    frozen = functional.extract(target).state
    state = _clone(state)
    state["populations"]["cell"]["clock"]["t"] += DT
    with pytest.raises(ValueError, match="clock"):
        functional.commit_state_(target, state)
    _close(functional.extract(target).state, frozen)


@pytest.mark.parametrize("compiled", (False, True))
def test_changed_raw_network_method_rejects_before_authored_descriptor_execution(
    monkeypatch, compiled
):
    model = _mixed_network()
    functional, tensors = dn.func.make_functional_network(model)
    execute, prepared = _execution(functional, tensors, compiled)
    state, _ = execute(tensors.parameters, prepared, tensors.state)
    kind = type(model)
    method = inspect.getattr_static(kind, "step")
    calls = []

    class ReplacedStep:
        def __get__(self, instance, owner=None):
            calls.append(instance)
            return method if instance is None else method.__get__(instance, owner)

    monkeypatch.setattr(kind, "step", ReplacedStep())
    error = (dn.func.FunctionalizationError, torch._dynamo.exc.Unsupported)
    with pytest.raises(
        error, match="adapters|method|configuration|canonical|Observed exception"
    ):
        execute(tensors.parameters, prepared, state)
    assert calls == []


def test_heap_and_tensor_schedules_flow_through_network_checkpoint_continuation():
    model, native, resumed = _stim_network(), _stim_network(), _stim_network()
    for network in (model, native, resumed):
        stimulus = network.netstim
        stimulus.set_diff_config(diff_scheduled_times=True, sched_width=0.8)
        stimulus.schedule([0, 0], [0.1, 0.6])
        stimulus.schedule(
            [0, 0],
            torch.tensor([0.31, 0.82], dtype=DTYPE, requires_grad=True),
            weight=torch.tensor([0.4, -0.2], dtype=DTYPE, requires_grad=True),
        )
        next(iter(network.synapses.values())).schedule(
            con_indices=[0, 0],
            times_ms=torch.tensor([0.25, 0.75], dtype=DTYPE, requires_grad=True),
            weight=torch.tensor([0.35, -0.1], dtype=DTYPE, requires_grad=True),
        )
    for network in (native, resumed):
        network.netstim.restore_dict_from_checkpoint(
            _clone(model.netstim.state_dict_for_checkpoint(), detached=True)
        )
    # The detached provenance restore also restores schedule storage; enable
    # the native oracle's explicitly trainable schedule leaves again.
    for network in (native, resumed):
        network.netstim.sched_weight.requires_grad_()
        network.netstim.sched_time_ms.requires_grad_()
    functional, tensors = dn.func.make_functional_network(model)
    stimulus_schedule = tensors.parameters["netstim"]["schedule"]
    connection_schedule = next(iter(tensors.parameters["synapses"].values()))[
        "schedule"
    ]
    assert stimulus_schedule is not None and connection_schedule is not None
    initial = _clone(tensors.state, detached=True)
    state = tensors.state
    checkpoint = None
    for index in range(10):
        state, _ = functional.prepare_and_step(
            tensors.parameters,
            tensors.constants,
            state,
            netstim_draw=functional.sample_step_draw(state),
        )
        native.step()
        _close(state, functional.extract(native).state)
        if index == 3:
            checkpoint = _clone(state)
    assert checkpoint is not None
    assert (
        state["netstim"]["heap_cursor"].sum() > initial["netstim"]["heap_cursor"].sum()
    )
    replay = checkpoint
    functional.commit_state_(resumed, checkpoint)
    for _ in range(4, 10):
        replay, _ = functional.prepare_and_step(
            tensors.parameters,
            tensors.constants,
            replay,
            netstim_draw=functional.sample_step_draw(replay),
        )
        resumed.step()
        _close(replay, functional.extract(resumed).state)
    _close(replay, state)
    loss = state["populations"]["cell"]["integrator"]["v"].square().sum()
    gradients = torch.autograd.grad(
        loss, (stimulus_schedule.amplitudes, connection_schedule.amplitudes)
    )
    references = torch.autograd.grad(
        native.cell.v.square().sum(),
        (
            native.netstim.sched_weight,
            next(iter(native.synapses.values())).sched_weight,
        ),
    )
    _close(gradients, references)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
    )
    _close(tensors.state, initial)


class _BooleanGate(Mechanism):
    Mechanism.CARRY("gate", dtype=torch.bool)


class _IntegerGate(Mechanism):
    Mechanism.CARRY("gate", dtype=torch.long)


@pytest.mark.parametrize(
    "kind,dtype,values",
    (
        (_BooleanGate, torch.bool, [True, False]),
        (_IntegerGate, torch.long, [2, -1]),
    ),
)
def test_typed_carry_sources_use_native_floating_cast_for_both_projection_kinds(
    kind, dtype, values
):
    def make():
        with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
            pre = dn.SingleCompartment(N=2, C=1, integrator=dn.bwd_euler_sc(imem=False))
            pre.insert(kind.rename("source"))
            post = dn.SingleCompartment(
                N=2, C=1, integrator=dn.bwd_euler_sc(imem=False)
            )
            post.insert(pas, g=0.001, e=-65.0)
            post.insert(expsyn.rename("event"), tau=1.2, e=0.0)
            post.insert(graded_syn.rename("analog"), e=-10.0)
            network = dn.Network(
                {"pre": pre, "post": post},
                netcon_delay_backend="dense",
                netcon_train_backend="dense",
            )
            network.connect_one_to_one(
                pre[:],
                post[:],
                post.mech.event,
                pre_var="mech.source.gate",
                threshold=None,
                weight=torch.tensor([0.002, 0.003]),
                delay=DT,
            )
            network.connect_continuous_one_to_one(
                pre[:],
                post[:],
                post.mech.analog,
                pre_var="mech.source.gate",
                weight=torch.tensor([0.001, 0.002]),
                delay=0.0,
                transform=torch.sigmoid,
            )
            network.train()
            network.initialize(DT)
            network.set_synaptic_diff_config(
                diff_weights=True,
                diff_delays=False,
                diff_spiking=True,
                train_delay_backend="dense",
            )
            pre.mech.source.gate = torch.tensor(values, dtype=dtype).reshape(2, 1)
        return network

    model, native = make(), make()
    functional, tensors = dn.func.make_functional_network(model)
    state = tensors.state
    for _ in range(5):
        state, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, state
        )
        native.step()
        _close(state, functional.extract(native).state)
        assert (
            state["populations"]["pre"]["mechanism_buffers"]["source"]["gate"].dtype
            == dtype
        )
    weights = (
        next(iter(tensors.parameters["synapses"].values()))[_RAW_WEIGHT],
        next(iter(tensors.parameters["continuous_synapses"].values()))[_RAW_WEIGHT],
    )
    actual = torch.autograd.grad(
        state["populations"]["post"]["integrator"]["v"].square().sum(), weights
    )
    expected = torch.autograd.grad(
        native.post.v.square().sum(),
        (
            next(iter(native.synapses.values())).weight.weights[0].param.rho,
            next(iter(native.continuous_synapses.values())).weight.weights[0].param.rho,
        ),
    )
    _close(actual, expected)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in actual
    )
