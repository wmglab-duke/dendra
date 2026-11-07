"""Prepared observations follow compatible native receivers and explicit carry."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_continuation_checkpoints import _disk
from tests.func.test_imperative_checkpoint_callbacks import EnergyReducer
from tests.func.test_native_handoff import _capture, _unchanged
from tests.func.test_network_population_callbacks import CONDUCTANCE, DT, _network
from tests.func.test_recorder_regions import _point_pack
from tests.func.test_stimulation_rng_sharing import DT as STOCHASTIC_DT
from tests.func.test_stimulation_rng_sharing import _lower as _lower_stochastic
from tests.func.test_stimulation_rng_sharing import _network as _stochastic_network
from tests.func.test_stimulation_rng_sharing import _one_stream

pytestmark = pytest.mark.cpu
_METHODS = ("run", "longrun", "longrun_checkpointed", "step")


@pytest.fixture(autouse=True)
def _environment():
    threads = torch.get_num_threads()
    rng = torch.get_rng_state().clone()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0,
                REQUIRE_GRAD=1,
                DEVICE="cpu",
                DTYPE=torch.float64,
                IMEM=True,
                CHECK_PURITY=False,
            ),
        ):
            yield
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


def _case(network):
    if network:
        return _network()
    model = dn.SingleCompartment(
        N=2,
        C=3,
        v_init=-63.0,
        integrator=dn.bwd_euler_sc(imem=True),
    )
    model.insert(pas, g=0.001, e=-60.0)
    model.train()
    model.initialize()
    return model


def _population(model, network):
    return model.left if network else model


def _parameter(model, network):
    return _population(model, network).mech.pas.g_param


def _scope(callback, network):
    return dn.callbacks.PopulationCallback("left", callback) if network else callback


def _specifications(field, network, *, portable=False):
    class PortableEnergy(EnergyReducer):
        def checkpoint_configuration(self):
            return {"kind": "tests.rebinding.energy", "version": 1, "fields": {}}

    return {
        "trace": dn.Recorder(
            ("left.v", "right.v", "t") if network else ("v", "t"), dt=2 * DT
        ),
        "energy": _scope(PortableEnergy() if portable else EnergyReducer(), network),
        "count": _scope(
            dn.callbacks.APCount(threshold=-100, node_check=[0, -1]), network
        ),
        "lfp": _scope(dn.callbacks.LFP([field, field]), network),
    }


def _prepare(source, preparation, specifications):
    if preparation == "program":
        program = dn.compile(
            source,
            dt=DT,
            steps=3,
            inference=False,
            trainable="*g_param",
            callbacks=specifications,
            backend="eager",
            disable=True,
        )
        return (
            program,
            program.functional,
            program.callbacks,
            dn.func.Continuation(
                program.initial_state, program.parameters, program.constants
            ),
        )
    if preparation == "native":
        callbacks = dn.func.make_callbacks(source, specifications, dt=DT)
        functional = callbacks._functional
        tensors = functional.extract()
    else:
        functional, tensors = dn.func.make_functional(source, dt=DT)
        callbacks = functional.make_callbacks(specifications)
    return (
        functional,
        functional,
        callbacks,
        dn.func.Continuation(
            tensors.state,
            tensors.independent_parameters(trainable="*g_param"),
            tensors.constants,
        ),
    )


def _native(model, network, method, callbacks, *, state=None, steps=2):
    options = {"callbacks": callbacks}
    if state is not None:
        options["callback_state"] = state
    if method == "step":
        return model.step(**options) if network else model.step(DT, **options)
    options["tstop"] = steps * DT
    if not network:
        options["dt"] = DT
    if method != "run":
        options["chunklength"] = 2
    if method == "longrun_checkpointed":
        options.update(safe_checkpoint=True, restore_state_after_backward=True)
    result = getattr(model, method)(**options)
    if method == "longrun_checkpointed":
        assert result[0] is None
        return result[1]
    return result


def _objective(result, network):
    trace = result["trace"]["left.v" if network else "v"]
    return (
        result["energy"]["energy"].sum()
        + trace.square().mean()
        + (result["lfp"] * 1e6).square().mean()
    )


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("preparation", ("plan", "program", "native"))
def test_functional_handoff_resumes_the_same_callbacks_on_an_independent_native_receiver(
    network, method, preparation
):
    source, receiver = _case(network), _case(network)
    field = torch.ones_like(_population(source, network).v, requires_grad=True)
    specifications = _specifications(field, network)
    subject, functional, callbacks, initial = _prepare(
        source, preparation, specifications
    )
    selected = (
        initial.parameters["populations"]["left"][CONDUCTANCE]
        if network
        else initial.parameters[CONDUCTANCE]
    )
    with torch.no_grad():
        selected.mul_(1.4)
    source_before = _capture(source)
    _, prefix = dn.func.run(functional, initial, steps=3, callbacks=callbacks)
    packet = prefix["continuation"]
    callback_state = callbacks.detach_state(packet.callback_state)
    assert callback_state.plan_token is packet.callback_state.plan_token
    plans = callbacks._plans
    assert dn.func.commit_continuation_(subject, receiver, packet) is receiver
    _close(callback_state.carries, packet.callback_state.carries)
    steps = 1 if method == "step" else 2
    expected_state, expected = dn.func.run(
        functional,
        dn.func.Continuation(
            _snapshot(packet.state), packet.parameters, packet.constants, callback_state
        ),
        steps=steps,
        callbacks=callbacks,
    )
    actual = _native(receiver, network, method, callbacks, state=callback_state)
    assert callbacks._plans is plans and callbacks._functional is functional
    assert actual.state.plan_token is callback_state.plan_token
    _close(actual.outputs, expected["callbacks"].outputs)
    _close(functional.extract(receiver).state, expected_state)
    assert actual["trace"]["t"].shape == (1,)
    assert actual["energy"]["steps"].item() == 3 + steps
    assert torch.equal(actual["count"], prefix["callbacks"]["count"])
    native_gradient = torch.autograd.grad(
        _objective(actual, network),
        (_parameter(receiver, network), field, _parameter(source, network)),
        allow_unused=True,
    )
    expected_gradient = torch.autograd.grad(
        _objective(expected["callbacks"], network), (selected, field)
    )
    _close(native_gradient[:2], expected_gradient)
    assert native_gradient[2] is None
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0
        for value in native_gradient[:2]
    )
    _unchanged(source, source_before)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_fresh_zero_step_receivers_are_independent_and_resumed_carry_stays_explicit(
    network,
):
    source, left, right = _case(network), _case(network), _case(network)
    callbacks = dn.func.make_callbacks(source, {"trace": dn.Recorder("t")}, dt=DT)
    _native(left, network, "run", None, steps=1)
    _native(right, network, "run", None, steps=2)
    source_before = _capture(source)
    first = _native(left, network, "run", callbacks, steps=0)
    second = _native(right, network, "longrun", callbacks, steps=0)
    _close(first["trace"]["t"], left.t.reshape(1))
    _close(second["trace"]["t"], right.t.reshape(1))
    assert first.state.plan_token is second.state.plan_token
    resumed = _native(right, network, "run", callbacks, state=first.state, steps=0)
    assert resumed["trace"]["t"].shape == (0,)
    _close(resumed.state.carries, first.state.carries)
    _unchanged(source, source_before)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_native_rebinding_keeps_functional_owner_and_original_source_validation(
    network,
):
    source, receiver = _case(network), _case(network)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    callbacks = functional.make_callbacks({"trace": dn.Recorder("t")})
    other, other_tensors = dn.func.make_functional(receiver, dt=DT)
    with pytest.raises(
        dn.func.FunctionalizationError, match="different functional execution plan"
    ):
        dn.func.run(
            other,
            dn.func.Continuation(
                other_tensors.state, other_tensors.parameters, other_tensors.constants
            ),
            steps=0,
            callbacks=callbacks,
        )
    _native(receiver, network, "run", callbacks, steps=0)
    source.eval()
    source.initialize(DT) if network else source.initialize()
    receiver_before = _capture(receiver)
    with pytest.raises(dn.func.FunctionalizationError):
        _native(receiver, network, "run", callbacks, steps=0)
    with pytest.raises(dn.func.FunctionalizationError):
        dn.func.run(
            functional,
            dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
            steps=0,
            callbacks=callbacks,
        )
    _unchanged(receiver, receiver_before)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("incompatibility", ("mode", "dtype", "topology", "dt"))
def test_incompatible_receiver_rejection_precedes_observations(
    network, incompatibility
):
    source, receiver = _case(network), _case(network)
    callbacks = dn.func.make_callbacks(
        source, {"trace": dn.Recorder("t", dt=2 * DT)}, dt=DT
    )
    if incompatibility == "mode":
        receiver.eval()
        receiver.initialize(DT) if network else receiver.initialize()
    elif incompatibility == "dtype":
        receiver.to(dtype=torch.float32)
        receiver.initialize(DT) if network else receiver.initialize()
    elif incompatibility == "topology":
        receiver.batch(3)
        receiver.initialize(DT) if network else receiver.initialize()
    elif network:
        receiver.initialize(2 * DT)
    before = _capture(receiver)
    source_before = _capture(source)
    with pytest.raises(dn.func.FunctionalizationError):
        if incompatibility == "dt" and not network:
            receiver.run(tstop=0, dt=2 * DT, callbacks=callbacks)
        else:
            _native(receiver, network, "run", callbacks, steps=0)
    _unchanged(receiver, before)
    _unchanged(source, source_before)


def test_network_rebinding_rejects_different_population_scopes():
    source = _case(True)
    unbatched = _network(batch=None)
    receiver = dn.Network({"other": unbatched.left, "right": unbatched.right})
    receiver.batch(2)
    receiver.train()
    receiver.initialize(DT)
    callbacks = dn.func.make_callbacks(
        source,
        {"trace": dn.callbacks.PopulationCallback("left", dn.Recorder("v"))},
        dt=DT,
    )
    before = _capture(receiver)
    source_before = _capture(source)
    with pytest.raises(
        dn.func.FunctionalizationError, match="scope|topology|structure"
    ):
        _native(receiver, True, "run", callbacks, steps=0)
    _unchanged(receiver, before)
    _unchanged(source, source_before)


@pytest.mark.parametrize(
    "packed", (False, True), ids=("physical_slice", "packed_named_region")
)
def test_rebound_population_recorders_reuse_resolved_physical_regions(packed):
    if packed:
        source, functional, _ = _point_pack()
        receiver = _point_pack()[0]
        region = source.cell_a.soma
    else:
        source, receiver = _case(False), _case(False)
        functional, _ = dn.func.make_functional(source, dt=DT)
        region = source[..., [2, 0, 2]]
    callbacks = functional.make_callbacks(
        {
            "full": dn.Recorder(("v", "t")),
            "selected": dn.Recorder(("v", "t"), region=region),
        }
    )
    with torch.no_grad():
        receiver.v.add_(2.0)
    source_before = _capture(source)
    first = _native(receiver, False, "run", callbacks, steps=1)
    last = _native(receiver, False, "step", callbacks, state=first.state)
    full = torch.cat((first["full"]["v"], last["full"]["v"]))
    selected = torch.cat((first["selected"]["v"], last["selected"]["v"]))
    expected = torch.stack([frame[region.index] for frame in full])
    _close(selected, expected)
    _close(last["selected"]["t"], last["full"]["t"])
    assert selected.shape == (3, *region.shape)
    assert not torch.equal(full[0], source.v)
    assert last.state.plan_token is first.state.plan_token
    _unchanged(source, source_before)


@pytest.mark.parametrize("method", ("run", "longrun_checkpointed", "step"))
def test_stochastic_network_rebinding_reads_receiver_streams_without_touching_source(
    method,
):
    source, source_pulses = _stochastic_network(seed=601)
    receiver, receiver_pulses = _stochastic_network(seed=991)
    receiver.step()
    functional, tensors = _lower_stochastic(source)
    callbacks = functional.make_callbacks(
        {
            "trace": dn.Recorder(("left.v", "right.v", "t")),
            "energy": dn.callbacks.PopulationCallback("left", EnergyReducer()),
        }
    )
    source_before = _capture(source)
    source_rng = source_pulses[0].generator.get_state().clone()
    _, prefix = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        steps=2,
        callbacks=callbacks,
    )
    packet = prefix["continuation"]
    state = callbacks.detach_state(packet.callback_state)
    dn.func.commit_continuation_(functional, receiver, packet)
    steps = 1 if method == "step" else 3
    expected, auxiliary = dn.func.run(
        functional, packet, steps=steps, callbacks=callbacks
    )
    options = {"callbacks": callbacks, "callback_state": state}
    if method == "step":
        result = receiver.step(**options)
    else:
        options["tstop"] = steps * STOCHASTIC_DT
        if method == "longrun_checkpointed":
            options.update(
                chunklength=2, safe_checkpoint=True, restore_state_after_backward=True
            )
        result = getattr(receiver, method)(**options)
        if method == "longrun_checkpointed":
            assert result[0] is None
            result = result[1]
    _close(result.outputs, auxiliary["callbacks"].outputs)
    _close(functional.extract(receiver).state, expected)
    _close(receiver_pulses[0].generator.get_state(), _one_stream(expected))
    _close(source_pulses[0].generator.get_state(), source_rng)
    _unchanged(source, source_before)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_rebound_checkpoint_carry_remains_portable_for_reconstructed_callbacks(
    tmp_path, network
):
    source, receiver = _case(network), _case(network)
    field = torch.ones_like(_population(source, network).v)
    callbacks = dn.func.make_callbacks(
        source, _specifications(field, network, portable=True), dt=DT
    )
    prefix = _native(receiver, network, "longrun_checkpointed", callbacks, steps=3)
    native_state = _snapshot(receiver.state_dict_for_checkpoint())
    saved = _disk(callbacks.export_state(prefix.state), tmp_path / "callbacks.pt")
    first = _native(
        receiver, network, "longrun", callbacks, state=prefix.state, steps=2
    )
    restored = _case(network)
    restored.restore_dict_from_checkpoint(native_state)
    receiving = dn.func.make_callbacks(
        restored, _specifications(field, network, portable=True), dt=DT
    )
    carry = receiving.restore_state(saved)
    assert carry.plan_token is not prefix.state.plan_token
    second = _native(restored, network, "run", receiving, state=carry, steps=2)
    _close(second.outputs, first.outputs)
    _assert_packet_equal(
        restored.state_dict_for_checkpoint(), receiver.state_dict_for_checkpoint()
    )
