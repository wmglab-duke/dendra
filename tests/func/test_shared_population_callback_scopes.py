"""One Population scope configures native and pure Network observations."""

from __future__ import annotations

import os
from contextlib import nullcontext

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_continuation_checkpoints import _disk
from tests.func.test_lfp import _model as _mps_population
from tests.func.test_network_population_callbacks import (
    CONDUCTANCE,
    DT,
    _chunk,
    _network,
    _PopulationEnergy,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=True),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _initial(tensors):
    return dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants)


def _configurations(network):
    result = {}
    for name, population in network.populations.items():
        options = dict(
            threshold=-62.95,
            node_check=[0, 0, -1],
            t_start_check=DT,
            t_end_check=5 * DT,
            dt=DT,
        )
        result[name] = {
            "trace": dn.Recorder(("v", "t"), dt=2 * DT),
            "lfp": dn.callbacks.LFP([torch.ones_like(population.v)]),
            "count": dn.callbacks.APCount(**options),
            "active": dn.callbacks.Active(**options),
            "sites": dn.callbacks.ActiveAL(**options, at_least=2),
            "total": dn.callbacks.ActiveALCount(**options, at_least=3),
        }
    return result


def _scoped(configurations, scope=dn.callbacks.PopulationCallback):
    return {
        f"{population}_{name}": scope(population, callback)
        for population, callbacks in configurations.items()
        for name, callback in callbacks.items()
    }


@pytest.mark.parametrize("execution", ("run", "longrun"))
def test_shared_scopes_match_native_network_callbacks_for_unequal_populations(
    execution,
):
    network = _network()
    functional, tensors = dn.func.make_functional_network(network)
    native = _configurations(network)
    configured = _configurations(network)
    callbacks = functional.make_callbacks(_scoped(configured))
    legacy = functional.make_callbacks(_scoped(configured, dn.func.PopulationCallback))
    with torch.no_grad():
        _, actual = dn.func.run(
            functional, _initial(tensors), steps=6, callbacks=callbacks
        )
        _, compatible = dn.func.run(
            functional, _initial(tensors), steps=6, callbacks=legacy
        )
        kwargs = {"chunklength": 4} if execution == "longrun" else {}
        getattr(network, execution)(
            6 * DT, callbacks=list(_scoped(native).values()), **kwargs
        )
    _close(actual["callbacks"].outputs, compatible["callbacks"].outputs)
    for population, specifications in native.items():
        for name, callback in specifications.items():
            output = actual["callbacks"][f"{population}_{name}"]
            if name == "trace":
                for field, values in output.items():
                    _close(values, callback.stack(field))
                assert output["v"].shape == (4, *network.populations[population].shape)
            elif name == "lfp":
                _close(output, callback.lfp)
            elif name == "count":
                _close(output, callback.record)
                assert output.dtype == torch.int32
            else:
                _close(output, callback.is_active())
                assert output.dtype == torch.bool
        source = configured[population]
        assert source["trace"].rec["v"] == []
        assert source["lfp"]._lfp == source["lfp"]._t == []
        for name in ("count", "active", "sites", "total"):
            assert source[name].record is source[name].state_cache is None
            assert source[name].i == 0
    assert actual["callbacks"]["left_count"].any()
    assert not actual["callbacks"]["right_count"].any()


def test_shared_scoped_recorder_selects_a_named_region_inside_a_concatenated_population():
    network = _network(packed=True)
    functional, tensors = dn.func.make_functional_network(network)
    region = network.packed.left.soma
    configured = dn.Recorder(("v", "i_membrane", "t"), region=region)
    callbacks = functional.make_callbacks(
        {"selected": dn.callbacks.PopulationCallback("packed", configured)}
    )
    native = dn.Recorder(("v", "i_membrane", "t"), region=region)
    with torch.no_grad():
        _, auxiliary = dn.func.run(
            functional, _initial(tensors), steps=4, callbacks=callbacks
        )
        network.longrun(
            4 * DT,
            chunklength=3,
            callbacks=[dn.callbacks.PopulationCallback("packed", native)],
        )
    for field, values in auxiliary["callbacks"]["selected"].items():
        _close(values, native.stack(field))
    assert auxiliary["callbacks"]["selected"]["v"].shape == (
        5,
        *network.packed.v[region.index].shape,
    )


def _compiled_case(network, execution, tmp_path):
    functional, tensors = dn.func.make_functional_network(network)
    population = network.populations["left"]
    field = torch.ones_like(population.v, requires_grad=execution == "scan")
    configured = dn.callbacks.LFP([field, field])
    source = _snapshot(functional.extract().state)
    public = configured.v_unit
    histories = configured._lfp, configured._t

    def specifications(scope):
        return {
            "lfp": scope("left", configured),
            "trace": scope("right", dn.Recorder(("v", "t"), dt=2 * DT)),
            "count": scope(
                "left", dn.callbacks.APCount(threshold=-62.9, node_check=[0, -1], dt=DT)
            ),
        }

    callbacks = functional.make_callbacks(
        specifications(dn.callbacks.PopulationCallback)
    )
    legacy = functional.make_callbacks(specifications(dn.func.PopulationCallback))
    kernel = _chunk(functional, execution)
    inputs = {
        name: dn.func.Inputs(
            intra=torch.linspace(
                -2e-7,
                3e-7,
                8 * owner.v.numel(),
                dtype=owner.v.dtype,
                device=owner.v.device,
            )
            .reshape(8, *owner.shape)
            .requires_grad_(execution == "scan")
        )
        for name, owner in network.populations.items()
    }

    def sliced(start, stop):
        return {
            name: dn.func.Inputs(intra=packet.intra[start:stop])
            for name, packet in inputs.items()
        }

    checkpoint = {"checkpoint_every": 2} if execution == "scan" else {}
    with nullcontext() if execution == "scan" else torch.no_grad():
        expected, whole = dn.func.run(
            functional, _initial(tensors), steps=8, inputs=inputs, callbacks=legacy
        )
        _, first = dn.func.run(
            kernel,
            _initial(tensors),
            steps=4,
            inputs=sliced(0, 4),
            callbacks=callbacks,
            **checkpoint,
        )
        assert kernel.execution_report().strategy == (
            "compiled_scan" if execution == "scan" else "compiled_while_loop"
        )
        _, empty = dn.func.run(
            kernel, first["continuation"], steps=0, callbacks=callbacks, **checkpoint
        )
        actual, last = dn.func.run(
            kernel,
            empty["continuation"],
            steps=4,
            inputs=sliced(4, 8),
            callbacks=callbacks,
            **checkpoint,
        )
    _close(actual, expected)
    assert empty["callbacks"]["lfp"].shape[0] == 0
    joined = torch.cat((first["callbacks"]["lfp"], last["callbacks"]["lfp"]))
    _close(joined * 1e6, whole["callbacks"]["lfp"] * 1e6)
    for key in ("v", "t"):
        _close(
            torch.cat(
                (first["callbacks"]["trace"][key], last["callbacks"]["trace"][key])
            ),
            whole["callbacks"]["trace"][key],
        )
    _close(last["callbacks"]["count"], whole["callbacks"]["count"])
    if execution == "scan":
        targets = (
            field,
            inputs["left"].intra,
            tensors.parameters["populations"]["left"][CONDUCTANCE],
        )
        actual_grad = torch.autograd.grad(
            (joined * 1e6).square().mean(), targets, retain_graph=True
        )
        expected_grad = torch.autograd.grad(
            (whole["callbacks"]["lfp"] * 1e6).square().mean(), targets
        )
        _close(actual_grad, expected_grad)
        assert all(
            torch.isfinite(value).all() and value.abs().sum() > 0
            for value in actual_grad
        )
    saved = _disk(
        dn.func.export_continuation(
            functional, first["continuation"], callbacks=callbacks
        ),
        tmp_path / "scoped.pt",
    )
    restored = dn.func.restore_continuation(functional, saved, callbacks=legacy)
    _assert_packet_equal(
        saved,
        dn.func.export_continuation(functional, restored, callbacks=legacy),
    )
    with torch.no_grad():
        resumed, tail = dn.func.run(
            kernel, restored, steps=4, inputs=sliced(4, 8), callbacks=legacy
        )
    _close(resumed, expected)
    _close(tail["callbacks"].outputs, last["callbacks"].outputs)
    with pytest.raises(dn.func.FunctionalizationError, match="different callback plan"):
        dn.func.run(kernel, first["continuation"], steps=0, callbacks=legacy)
    assert configured.v_unit is public
    assert configured._lfp is histories[0] and configured._t is histories[1]
    assert histories[0] == histories[1] == []
    _close(functional.extract().state, source)
    assert network.t.item() == 0
    return actual


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_shared_scopes_keep_gradients_live_continuation_and_portable_legacy_restore(
    execution, tmp_path
):
    _compiled_case(_network(), execution, tmp_path)


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_shared_scopes_accept_custom_pure_callbacks_with_population_local_frames(
    execution,
):
    network = _network()
    functional, tensors = dn.func.make_functional_network(network)
    callbacks = functional.make_callbacks(
        {"energy": dn.callbacks.PopulationCallback("right", _PopulationEnergy())}
    )
    legacy = functional.make_callbacks(
        {"energy": dn.func.PopulationCallback("right", _PopulationEnergy())}
    )
    with nullcontext() if execution == "scan" else torch.no_grad():
        expected, oracle = dn.func.run(
            functional, _initial(tensors), steps=4, callbacks=legacy
        )
        actual, result = dn.func.run(
            _chunk(functional, execution),
            _initial(tensors),
            steps=4,
            callbacks=callbacks,
        )
    _close(actual, expected)
    _close(result["callbacks"].outputs, oracle["callbacks"].outputs)


class _NativeVoltageLoss(dn.callbacks.Callback):
    def post_step_hook(self, model):
        return (1 + model.t) * model.v.square().mean()


def test_native_checkpointed_scoped_loss_matches_eager_and_restores_state_after_backward():
    eager = _network(batch=None)
    reference = eager.left.mech.pas.g_param
    loss = torch.zeros((), dtype=torch.float64)
    callback = _NativeVoltageLoss()
    for _ in range(5):
        eager.step()
        loss = loss + callback.post_step_hook(eager.left)
    expected_gradient = torch.autograd.grad(loss, reference)[0]
    checkpointed = _network(batch=None)
    target = checkpointed.left.mech.pas.g_param
    actual, final = checkpointed.longrun_checkpointed(
        5 * DT,
        chunklength=3,
        callbacks=[dn.callbacks.PopulationCallback("left", _NativeVoltageLoss())],
        safe_checkpoint=True,
        restore_state_after_backward=True,
        return_final_state=True,
    )
    frozen = _snapshot(final)
    _close(actual, loss)
    actual.backward()
    _close(target.grad, expected_gradient)
    assert torch.isfinite(target.grad).all() and target.grad.abs().sum() > 0
    _assert_packet_equal(checkpointed.state_dict_for_checkpoint(), frozen)
    _close(checkpointed.left.v, eager.left.v)
    _close(checkpointed.right.v, eager.right.v)


def test_functional_scope_binding_captures_the_selected_population_and_child_configuration():
    network = _network()
    functional, tensors = dn.func.make_functional_network(network)
    original = dn.Recorder("v")
    scope = dn.callbacks.PopulationCallback("left", original)
    callbacks = functional.make_callbacks({"trace": scope})
    replacement = dn.Recorder(("v", "t"))
    scope.population, scope.callback = "right", replacement
    rebound = functional.make_callbacks({"trace": scope})
    _, before = dn.func.run(functional, _initial(tensors), steps=0, callbacks=callbacks)
    _, after = dn.func.run(functional, _initial(tensors), steps=0, callbacks=rebound)
    assert set(before["callbacks"]["trace"]) == {"v"}
    assert set(after["callbacks"]["trace"]) == {"v", "t"}
    _close(before["callbacks"]["trace"]["v"], network.left.v.unsqueeze(0))
    _close(after["callbacks"]["trace"]["v"], network.right.v.unsqueeze(0))
    assert original.rec["v"] == replacement.rec["v"] == []


def test_functional_scope_diagnostics_match_legacy_for_names_and_standalone_populations():
    network = _network()
    functional, _ = dn.func.make_functional_network(network)
    population, _ = dn.func.make_functional(network.left, dt=DT)
    messages = []
    for scope in (dn.callbacks.PopulationCallback, dn.func.PopulationCallback):
        with pytest.raises(
            dn.func.FunctionalizationError, match="Unknown Network Population"
        ) as error:
            functional.make_callbacks({"trace": scope("missing", dn.Recorder("v"))})
        messages.append(str(error.value))
        with pytest.raises(
            dn.func.FunctionalizationError,
            match="must be bound to a functional Network",
        ):
            population.make_callbacks({"trace": scope("left", dn.Recorder("v"))})
    assert messages[0] == messages[1]


@pytest.mark.parametrize("customization", ("subclass", "hook", "native_child"))
def test_functional_shared_scope_rejects_custom_native_behavior(customization):
    class CustomScope(dn.callbacks.PopulationCallback):
        pass

    functional, _ = dn.func.make_functional_network(_network())
    scope = dn.callbacks.PopulationCallback("left", dn.Recorder("v"))
    if customization == "subclass":
        scope = CustomScope("left", dn.Recorder("v"))
    elif customization == "hook":
        scope.post_step_hook = lambda model: model.v
    else:
        scope.callback = _NativeVoltageLoss()
    error_type = (
        dn.func.FunctionalizationError if customization == "hook" else TypeError
    )
    with pytest.raises(error_type, match="FunctionalCallback"):
        functional.make_callbacks({"trace": scope})


@pytest.mark.mps
def test_mps_shared_scope_while_and_portable_continuation_use_native_configurations(
    tmp_path,
):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DTYPE=torch.float32, DEVICE="mps"):
        network = dn.Network(
            {
                "left": _mps_population(device="mps", dtype=torch.float32),
                "right": _mps_population(device="mps", dtype=torch.float32),
            }
        )
        network.train()
        network.initialize(DT)
        result = _compiled_case(network, "while", tmp_path)
        assert result["populations"]["left"]["integrator"]["v"].device.type == "mps"
