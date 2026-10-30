"""Recorder and LFP prepare their public pure implementations without copying sources."""

from __future__ import annotations

import os
from contextlib import nullcontext

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._callback_carry import _CallbackCarryPartition
from dendra.func._callbacks import _prepare_callback
from tests.func.test_callback_checkpoints import _assert_packet_equal, _model
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_continuation_checkpoints import _disk
from tests.func.test_lowering_api import DT, _voltage
from tests.func.test_network_recorder_regions import DT as NETWORK_DT
from tests.func.test_network_recorder_regions import _network as _region_network

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


def _case():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    return functional, tensors._replace(
        parameters=tensors.independent_parameters(trainable="*g_param")
    )


def _initial(tensors):
    return dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants)


def _immutable(prepared):
    with pytest.raises((AttributeError, TypeError)):
        prepared.new_attribute = None
    name = "states" if type(prepared) is dn.func.Recorder else "fields"
    with pytest.raises((AttributeError, TypeError)):
        setattr(prepared, name, ())
    with pytest.raises((AttributeError, TypeError)):
        delattr(prepared, name)


def _drive(functional, steps, *, trainable=False):
    return (
        torch.linspace(
            -1e-6,
            2e-6,
            steps * torch.Size(functional.shape).numel(),
            dtype=functional.dtype,
            device=functional.device,
        )
        .reshape(steps, *functional.shape)
        .requires_grad_(trainable)
    )


def _join(first, last, name):
    return {
        field: torch.cat((first["callbacks"][name][field], values))
        for field, values in last["callbacks"][name].items()
    }


@pytest.mark.parametrize("selection", ("full", "indices", "maximum", "cadence"))
def test_recorder_prepares_a_canonical_immutable_snapshot_and_keeps_cadence_adapter(
    selection,
):
    functional, tensors = _case()
    indices, partition = torch.tensor([2, 0, 2]), [2, 2]
    states = ["v", "t"] if selection in ("full", "cadence") else ["v"]
    options = {
        "indices": {"node_indices": indices},
        "maximum": {"max_only": True, "partition": partition},
        "cadence": {"dt": 2.5 * DT},
        "full": {},
    }[selection]
    authored = dn.func.Recorder(states, **options)
    original_states = authored.states
    with torch.inference_mode():
        bound = authored._bind(functional)
    prepared = bound.recorder if selection == "cadence" else bound
    assert type(prepared) is dn.func.Recorder and prepared is not authored
    assert prepared.states is original_states
    assert prepared.paths == tuple(
        ("clock", "t") if name == "t" else ("integrator", name) for name in states
    )
    assert prepared._bind(functional) is prepared
    if selection == "indices":
        assert authored.node_indices is indices and prepared.node_indices is not indices
        assert not prepared.node_indices.is_inference()
    if selection == "maximum":
        assert authored.partition is partition and prepared.partition == (2, 2)
    if selection == "cadence":
        assert (
            bound.recorder is prepared
            and bound.save_every == 2
            and bound.callback_dt == DT
        )
    authored.states = ("t",)
    authored.max_only = not authored.max_only
    indices.fill_(1)
    partition[:] = [1, 3]
    assert prepared.states == tuple(states)
    carry, observed = prepared.initialize(tensors.state, None)
    assert carry == ()
    voltage = tensors.state["integrator"]["v"]
    expected = (
        voltage.index_select(-1, torch.tensor([2, 0, 2]))
        if selection == "indices"
        else torch.stack([part.amax(-1) for part in voltage.split((2, 2), -1)], -1)
        if selection == "maximum"
        else voltage
    )
    _close(observed["v"], expected)
    if selection in ("full", "cadence"):
        assert observed["v"] is voltage and observed["t"] is tensors.state["clock"]["t"]
    collection = functional.make_callbacks({"trace": bound})
    assert (
        collection._plans[0].callback is bound
        and not collection._plans[0].validate_updates
    )
    _immutable(prepared)


def test_lfp_binding_retains_live_field_aliases_without_stacking_or_conversion():
    functional, _ = _case()
    field = torch.ones(functional.shape, dtype=torch.float32, requires_grad=True)
    authored = dn.func.LFP([field, field])

    class NoMaterialization(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert func not in (
                torch.ops.aten.stack.default,
                torch.ops.aten._to_copy.default,
            )
            return func(*args, **(kwargs or {}))

    with torch.inference_mode(), NoMaterialization():
        prepared = authored._bind(functional)
    assert type(prepared) is dn.func.LFP and prepared is not authored
    assert prepared.fields is authored.v_unit
    assert prepared.fields[0] is prepared.fields[1] is field
    assert prepared.dtype == functional.dtype and prepared.device == functional.device
    assert prepared.rule is None and prepared.padding == 0
    assert prepared._bind(functional) is prepared
    authored.v_unit = (torch.zeros_like(field),)
    authored.rule = "invalid"
    assert prepared.fields == (field, field) and prepared.rule is None
    _immutable(prepared)


@pytest.mark.parametrize("family", ("recorder", "lfp"))
def test_authored_observers_remain_mutable_guarded_and_unsupported_checkpoint_descriptions(
    family,
):
    functional, tensors = _case()
    authored = (
        dn.func.Recorder("v")
        if family == "recorder"
        else dn.func.LFP([torch.ones(functional.shape)])
    )
    for execute in (
        lambda: authored.initialize(tensors.state, None),
        lambda: authored.update(None, tensors.state, None),
        lambda: authored.finalize(None, None),
    ):
        with pytest.raises(dn.func.FunctionalizationError, match="make_callbacks"):
            execute()
    with pytest.raises(
        dn.func.FunctionalizationError, match="custom callbacks|unsupported"
    ):
        authored.checkpoint_configuration()
    authored.new_attribute = "mutable"
    assert authored.new_attribute == "mutable"


def test_recorder_subclass_binding_hooks_keep_order_and_default_canonical_execution():
    functional, _ = _case()
    calls = []

    class Authored(dn.func.Recorder):
        def _resolve_paths(self, plan):
            calls.append("paths")
            return super()._resolve_paths(plan)

        def _bind_region(self, plan, paths):
            calls.append("region")
            return super()._bind_region(plan, paths)

        def _bind_paths(self, plan, paths, region):
            calls.append("prepare")
            return super()._bind_paths(plan, paths, region)

        def initialize(self, state, auxiliary):
            raise AssertionError("default binding must use the canonical Recorder")

    callbacks = functional.make_callbacks({"trace": Authored("v")})
    assert calls == ["paths", "region", "prepare"]
    assert type(callbacks._plans[0].callback) is dn.func.Recorder
    assert not callbacks._plans[0].validate_updates

    class AuthoredLFP(dn.func.LFP):
        def initialize(self, state, auxiliary):
            raise AssertionError("default binding must use the canonical LFP")

    callbacks = functional.make_callbacks(
        {"field": AuthoredLFP([torch.ones(functional.shape)])}
    )
    assert type(callbacks._plans[0].callback) is dn.func.LFP
    assert not callbacks._plans[0].validate_updates


@pytest.mark.parametrize("family", ("recorder", "lfp"))
def test_custom_bind_authority_and_prepared_subclasses_keep_custom_validation(family):
    functional, tensors = _case()
    base = dn.func.Recorder if family == "recorder" else dn.func.LFP
    authored = (
        base("v") if family == "recorder" else base([torch.ones(functional.shape)])
    )
    prepared = authored._bind(functional)

    class Derived(base):
        pass

    derived = _prepare_callback(
        Derived, {name: getattr(prepared, name) for name in prepared._checkpoint_fields}
    )
    # A custom prepared implementation keeps its own type when deliberately
    # supplied; inheriting the default authored binding still canonicalizes it.
    collection = functional.make_callbacks({"custom": derived})
    assert (
        collection._plans[0].callback is derived
        and collection._plans[0].validate_updates
    )
    partition = _CallbackCarryPartition.from_plans(collection._plans)
    assert partition.invariant_names == ()
    with pytest.raises(
        dn.func.FunctionalizationError, match="custom callbacks|unsupported"
    ):
        derived.checkpoint_configuration()

    calls = []

    class Override(base):
        def _bind(self, plan):
            calls.append(plan)
            return derived

    override = (
        Override("not_a_state")
        if family == "recorder"
        else Override([torch.zeros(999)])
    )
    collection = functional.make_callbacks({"custom": override})
    assert calls == [functional] and collection._plans[0].callback is derived
    if family == "lfp":
        exact = functional.make_callbacks({"field": prepared, "custom": derived})
        assert not exact._plans[0].validate_updates and exact._plans[1].validate_updates
        assert _CallbackCarryPartition.from_plans(exact._plans).invariant_names == (
            "field",
        )
    carry, _ = collection._initialize(tensors.state)
    assert carry.plan_token is collection._state_token


@pytest.mark.parametrize(
    "packed", (False, True), ids=("network", "concatenated_network")
)
def test_prepared_network_regional_recorder_reuse_skips_authored_resolution_and_restores_carry(
    packed, monkeypatch, tmp_path
):
    model = _region_network(packed=packed, batch=None)
    functional, tensors = dn.func.make_functional(model, dt=NETWORK_DT)
    name = "packed" if packed else "left"
    population = model.populations[name]
    region = population.left.soma if packed else population.soma
    authored = dn.func.Recorder((f"{name}.v", f"{name}.i_membrane", "t"), region=region)
    callbacks = functional.make_callbacks({"trace": authored})
    prepared = callbacks._plans[0].callback
    assert type(prepared) is dn.func.Recorder and prepared.region is not region

    def reject(*args, **kwargs):
        raise AssertionError(
            "prepared Network Recorder must bypass authored resolution"
        )

    import dendra.func._network_callbacks as network_callbacks

    with monkeypatch.context() as guarded:
        guarded.setattr(network_callbacks, "_NetworkRecorderLayout", reject)
        for method in ("_resolve_paths", "_bind_region", "_bind_paths"):
            guarded.setattr(dn.func.Recorder, method, reject)
        receiving = functional.make_callbacks({"trace": prepared})
        custom = None
        if packed:

            class RegionalSubclass(dn.func.Recorder):
                def _observe(self, state):
                    return {
                        name: value if name == "t" else value * 2
                        for name, value in super()._observe(state).items()
                    }

            custom_recorder = _prepare_callback(
                RegionalSubclass,
                {
                    field: getattr(prepared, field)
                    for field in prepared._checkpoint_fields
                },
            )
            custom = functional.make_callbacks({"trace": custom_recorder})
            assert custom._plans[0].callback is custom_recorder
            assert custom._plans[0].validate_updates
            with pytest.raises(
                dn.func.FunctionalizationError, match="custom callbacks|unsupported"
            ):
                custom_recorder.checkpoint_configuration()
    assert receiving._plans[0].callback is prepared
    initial = _initial(tensors)
    expected, whole = dn.func.run(functional, initial, steps=5, callbacks=callbacks)
    prefix, first = dn.func.run(functional, initial, steps=2, callbacks=callbacks)
    restored = receiving.restore_state(
        _disk(
            callbacks.export_state(first["callbacks"].state), tmp_path / "regional.pt"
        )
    )
    actual, last = dn.func.run(
        functional.bind(initial),
        prefix,
        steps=3,
        callbacks=receiving,
        callback_state=restored,
    )
    _close(actual, expected)
    _close(_join(first, last, "trace"), whole["callbacks"]["trace"])
    assert last["callbacks"]["trace"][f"{name}.v"].shape[0] == 3
    if custom is not None:
        _, custom_auxiliary = dn.func.run(
            functional, initial, steps=5, callbacks=custom
        )
        for field, values in whole["callbacks"]["trace"].items():
            _close(
                custom_auxiliary["callbacks"]["trace"][field],
                values if field == "t" else values * 2,
            )


def test_prepared_cadence_recorder_continues_phase_without_duplicate_boundaries():
    functional, tensors = _case()
    outer = dn.func.Recorder(("v", "t"), dt=2.5 * DT)._bind(functional)
    assert type(outer.recorder) is dn.func.Recorder
    assert outer.recorder._bind(functional) is outer.recorder
    callbacks = functional.make_callbacks({"trace": outer})
    dense = functional.make_callbacks({"trace": dn.func.Recorder(("v", "t"))})
    initial = _initial(tensors)
    expected, full = dn.func.run(functional, initial, steps=7, callbacks=dense)
    _, first = dn.func.run(functional, initial, steps=3, callbacks=callbacks)
    _, zero = dn.func.run(
        functional, first["continuation"], steps=0, callbacks=callbacks
    )
    assert zero["callbacks"]["trace"]["v"].shape[0] == 0
    actual, last = dn.func.run(
        functional, zero["continuation"], steps=4, callbacks=callbacks
    )
    _close(actual, expected)
    selected = {
        name: value[[0, 1, 3, 5, 7]]
        for name, value in full["callbacks"]["trace"].items()
    }
    _close(_join(first, last, "trace"), selected)
    assert last["callbacks"].state["trace"].item() == 7


def test_inference_prepared_lfp_keeps_repeated_backward_graphs_and_live_field_alias_updates():
    functional, tensors = _case()
    field = (
        torch.linspace(
            0.5, 1.5, torch.Size(functional.shape).numel(), dtype=torch.float32
        )
        .reshape(functional.shape)
        .requires_grad_()
    )
    with torch.inference_mode():
        prepared = dn.func.LFP([field, field])._bind(functional)
        callbacks = functional.make_callbacks(
            {"field": prepared, "current": dn.func.Recorder("i_membrane")}
        )
    previous = None
    expected = None
    for iteration in range(3):
        if iteration == 2:
            with torch.no_grad():
                field.add_(0.25)
        drive = _drive(functional, 3, trainable=True)
        _, auxiliary = dn.func.run(
            functional,
            _initial(tensors),
            inputs=dn.func.Inputs(intra=drive),
            steps=3,
            callbacks=callbacks,
            checkpoint_every=2,
        )
        current = auxiliary["callbacks"]["current"]["i_membrane"]
        projection = (current * field.to(dtype=functional.dtype)).sum((-1, -2))
        _close(
            auxiliary["callbacks"]["field"], torch.stack((projection, projection), -1)
        )
        carry = auxiliary["callbacks"].state["field"]
        assert (
            carry is not previous and carry.requires_grad and carry.grad_fn is not None
        )
        previous = carry
        gradients = torch.autograd.grad(
            (auxiliary["callbacks"]["field"] * 1e6).square().mean(), (field, drive)
        )
        assert gradients[0].dtype == torch.float32
        assert all(
            torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
        )
        observed = auxiliary["callbacks"]["field"].detach().clone()
        if iteration == 0:
            expected = observed, _snapshot(gradients)
        elif iteration == 1:
            _close((observed, gradients), expected)
        else:
            assert not torch.equal(observed, expected[0])
    assert prepared.fields[0] is prepared.fields[1] is field


def _descriptor(tensor):
    return {
        "tensor": tensor.detach().cpu().clone(),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


@pytest.mark.parametrize("kind", ("recorder", "cadence", "lfp"))
def test_prepared_observers_keep_weights_only_v1_checkpoint_encoding_and_invariants(
    kind, tmp_path
):
    functional, tensors = _case()
    if kind == "lfp":
        field = torch.ones(functional.shape, dtype=torch.float32, requires_grad=True)
        authored = dn.func.LFP([field])
        expected = {
            "kind": "LFP",
            "fields": {
                "fields": [_descriptor(field)],
                "rule": None,
                "padding": 0,
                "device": str(functional.device),
                "dtype": str(functional.dtype),
            },
        }
    else:
        states = ("v",) if kind == "recorder" else ("v", "t")
        indices = torch.tensor([2, 0, 2]) if kind == "recorder" else None
        authored = dn.func.Recorder(
            states, node_indices=indices, dt=2.5 * DT if kind == "cadence" else None
        )
        recorder = {
            "kind": "Recorder",
            "fields": {
                "states": list(states),
                "paths": [
                    ["clock", "t"] if name == "t" else ["integrator", name]
                    for name in states
                ],
                "max_only": False,
                "node_indices": None if indices is None else _descriptor(indices),
                "partition": None,
                "region": None,
            },
        }
        expected = (
            recorder
            if kind == "recorder"
            else {
                "kind": "CadencedRecorder",
                "fields": {"recorder": recorder, "save_every": 2, "callback_dt": DT},
            }
        )
    callbacks = functional.make_callbacks({"observation": authored})
    receiving = functional.make_callbacks({"observation": authored})
    _, auxiliary = dn.func.run(
        functional, _initial(tensors), steps=2, callbacks=callbacks
    )
    saved = callbacks.export_state(auxiliary["callbacks"].state)
    assert saved["format"] == "dendra.func.callback_state" and saved["version"] == 1
    _assert_packet_equal(saved["configuration"]["callbacks"][0]["callback"], expected)
    restored = receiving.restore_state(_disk(saved, tmp_path / "observer.pt"))
    assert (
        restored.plan_token is receiving._state_token
        and restored.plan_token is not callbacks._state_token
    )
    _close(restored.carries, auxiliary["callbacks"].state.carries)
    if kind == "lfp":
        assert (
            not restored["observation"].requires_grad
            and restored["observation"].grad_fn is None
        )
        assert _CallbackCarryPartition.from_plans(receiving._plans).invariant_names == (
            "observation",
        )
        with torch.no_grad():
            field.add_(0.5)
        with pytest.raises(
            dn.func.FunctionalizationError, match="lead.field|LFP|configuration"
        ):
            callbacks.export_state(auxiliary["callbacks"].state)


def _compiled_observers(execution):
    functional, tensors = _case()
    trainable = execution == "scan"
    field = torch.ones(
        functional.shape,
        dtype=torch.float32,
        device=functional.device,
        requires_grad=trainable,
    )
    drive = _drive(functional, 9, trainable=trainable)
    callbacks = functional.make_callbacks(
        {
            "field": dn.func.LFP([field]),
            "dense": dn.func.Recorder(("v", "i_membrane", "t")),
            "slow": dn.func.Recorder(("v", "t"), dt=2.5 * DT),
        }
    )
    assert all(not plan.validate_updates for plan in callbacks._plans)
    assert _CallbackCarryPartition.from_plans(callbacks._plans).invariant_names == (
        "field",
    )
    kernel = functional.compile_rollout_chunk(
        4, execution=execution, backend="aot_eager"
    )
    options = {"checkpoint_every": 3} if trainable else {}
    with nullcontext() if trainable else torch.no_grad():
        expected, whole = dn.func.run(
            functional,
            _initial(tensors),
            inputs=dn.func.Inputs(intra=drive),
            steps=9,
            callbacks=callbacks,
        )
        _, first = dn.func.run(
            kernel,
            _initial(tensors),
            inputs=dn.func.Inputs(intra=drive[:5]),
            steps=5,
            callbacks=callbacks,
            **options,
        )
        actual, last = dn.func.run(
            kernel,
            first["continuation"],
            inputs=dn.func.Inputs(intra=drive[5:]),
            steps=4,
            callbacks=callbacks,
            **options,
        )
    _close(actual, expected)
    for name in ("dense", "slow"):
        _close(_join(first, last, name), whole["callbacks"][name])
    observed = torch.cat((first["callbacks"]["field"], last["callbacks"]["field"]))
    _close(observed, whole["callbacks"]["field"])
    if trainable:
        gradients = torch.autograd.grad(
            (observed * 1e6).square().mean(), (field, drive)
        )
        reference = torch.autograd.grad(
            (whole["callbacks"]["field"] * 1e6).square().mean(), (field, drive)
        )
        _close(gradients, reference)
        assert all(
            torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
        )
    assert kernel.execution_report().strategy == (
        "compiled_scan" if trainable else "compiled_while_loop"
    )
    return actual


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_prepared_observers_share_compiled_execution_gradients_and_cadence_continuation(
    execution,
):
    _compiled_observers(execution)


@pytest.mark.mps
def test_mps_prepared_observers_keep_lfp_invariant_and_cadence_while_execution():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DTYPE=torch.float32, DEVICE="mps"):
        actual = _compiled_observers("while")
        assert _voltage(actual, False).device.type == "mps"
