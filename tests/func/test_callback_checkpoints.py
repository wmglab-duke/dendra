"""Portable callback carry resumes equivalent plans without weakening ownership."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas

DT = 0.025
STEPS = 8
HALF = STEPS // 2
pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=True),
    ):
        yield


def _model():
    model = dn.Population(
        N=1,
        C=4,
        v_init=-60.0,
        integrator=dn.bwd_euler_sc(imem=True),
    )
    model.insert(pas, g=0.001, e=-45.0)
    model[:, [0, 2]].label("even")
    model.train()
    model.initialize(DT)
    return model


def _case(*, dt=DT):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=dt)
    return model, functional, tensors


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _assert_packet_equal(actual, expected):
    leaves, spec = pytree.tree_flatten(actual)
    reference, reference_spec = pytree.tree_flatten(expected)
    assert spec == reference_spec
    for value, other in zip(leaves, reference, strict=True):
        if value is other:
            continue
        assert type(value) is type(other)
        if torch.is_tensor(value):
            torch.testing.assert_close(value, other, rtol=0, atol=0, equal_nan=True)
        elif type(value) is float:
            assert value == other or (math.isnan(value) and math.isnan(other))
        else:
            assert value == other


def _disk(packet, tmp_path, filename="checkpoint.pt"):
    path = tmp_path / filename
    torch.save(packet, path)
    return torch.load(path, weights_only=True)


def _plain(value):
    if torch.is_tensor(value):
        assert type(value) is torch.Tensor
        assert not value.requires_grad and value.grad_fn is None
        assert not torch.is_inference(value)
    elif type(value) is dict:
        for key, item in value.items():
            assert type(key) in (str, int)
            _plain(item)
    elif type(value) in (tuple, list):
        for item in value:
            _plain(item)
    else:
        assert type(value) in (str, int, float, bool, type(None)), type(value)


def _specifications(model, *, fields=None):
    if fields is None:
        fields = [torch.linspace(0.5, 1.0, model.v.numel()).reshape(model.shape)]
    threshold = dict(
        threshold=-59.0,
        node_check=[0, 3],
        t_start_check=DT,
        t_end_check=7 * DT,
    )
    return {
        "trace": dn.func.Recorder(("v", "i_membrane", "t")),
        "maximum": dn.func.Recorder(("v",), max_only=True, partition=[2, 2]),
        "lfp": dn.func.LFP(fields),
        "raster": dn.func.Raster(**threshold),
        "count": dn.func.APCount(**threshold),
        "active": dn.func.Active(**threshold),
        "sites": dn.func.ActiveAL(**threshold, at_least=2),
        "total": dn.func.ActiveALCount(**threshold, at_least=2),
        "anomaly": dn.func.AnomalyDetector(),
    }


def _bound(functional, tensors, execution):
    if execution == "eager":
        bound = functional.bind(tensors.parameters, constants=tensors.constants)
    else:
        kernel = functional.compile_rollout_chunk(
            HALF,
            execution="default" if execution == "while" else "scan",
            backend="aot_eager",
        )
        bound = kernel.bind(tensors.parameters, constants=tensors.constants)
    return bound


def _execute(
    functional, tensors, state, callbacks, steps, execution, carry=None, *, bound=None
):
    if bound is None:
        bound = _bound(functional, tensors, execution)
    return dn.func.run(
        bound, state, steps=steps, callbacks=callbacks, callback_state=carry
    )


def _joined(first, second, expected):
    emitters = {"trace", "maximum", "lfp", "raster"}
    assert set(first) == set(second) == set(expected)
    for name, reference in expected.items():
        actual = second[name]
        if name in emitters:
            actual = pytree.tree_map(
                lambda prefix, suffix: torch.cat((prefix, suffix)), first[name], actual
            )
        torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("execution", ("eager", "scan", "while"))
def test_disk_checkpoint_resumes_all_population_builtins_and_recording_phase(
    tmp_path, execution
):
    model, functional, tensors = _case()
    callbacks = functional.make_callbacks(_specifications(model))
    receiver_model, receiver, received = _case()
    receiver_callbacks = receiver.make_callbacks(_specifications(receiver_model))
    bound = _bound(functional, tensors, execution)
    receiver_bound = _bound(receiver, received, execution)
    with torch.no_grad():
        expected, whole = _execute(
            functional, tensors, tensors.state, callbacks, STEPS, execution, bound=bound
        )
        prefix, first = _execute(
            functional, tensors, tensors.state, callbacks, HALF, execution, bound=bound
        )
        saved = {
            "model": _clone(prefix),
            "callbacks": callbacks.export_state(first["callbacks"].state),
        }
        _plain(saved)
        loaded = _disk(saved, tmp_path)
        restored = receiver_callbacks.restore_state(loaded["callbacks"])
        assert restored.plan_token is not first["callbacks"].state.plan_token
        with pytest.raises(
            dn.func.FunctionalizationError, match="different callback plan"
        ):
            _execute(
                receiver,
                received,
                loaded["model"],
                receiver_callbacks,
                0,
                execution,
                first["callbacks"].state,
                bound=receiver_bound,
            )
        empty, middle = _execute(
            receiver,
            received,
            loaded["model"],
            receiver_callbacks,
            0,
            execution,
            restored,
            bound=receiver_bound,
        )
        final, second = _execute(
            receiver,
            received,
            empty,
            receiver_callbacks,
            HALF,
            execution,
            middle["callbacks"].state,
            bound=receiver_bound,
        )
    torch.testing.assert_close(final, expected)
    _joined(first["callbacks"], second["callbacks"], whole["callbacks"])
    assert first["callbacks"]["trace"]["v"].shape[0] == HALF + 1
    assert second["callbacks"]["trace"]["v"].shape[0] == HALF
    assert middle["callbacks"]["trace"]["v"].shape[0] == 0
    assert middle["callbacks"]["raster"].shape[0] == 0
    assert whole["callbacks"]["count"].sum() > 0
    assert restored.emission_schemas[3] is not None


def test_export_and_restore_own_detached_ordinary_tensors_and_leave_carry_intact(
    tmp_path,
):
    model, functional, tensors = _case()
    field = torch.ones(model.shape, requires_grad=True)
    callbacks = functional.make_callbacks({"lfp": dn.func.LFP([field])})
    _, auxiliary = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=2,
        callbacks=callbacks,
    )
    original = auxiliary["callbacks"].state
    assert original["lfp"].requires_grad and original["lfp"].grad_fn is not None
    frozen = _clone(original.carries)
    with torch.inference_mode():
        packet = callbacks.export_state(original)
        restored = callbacks.restore_state(_disk(packet, tmp_path))
    _plain(packet)
    _plain(dict(restored.carries))
    torch.testing.assert_close(restored.carries, frozen)
    assert packet["carries"]["lfp"].data_ptr() != original["lfp"].data_ptr()
    assert restored["lfp"].data_ptr() != packet["carries"]["lfp"].data_ptr()
    restored["lfp"].add_(2)
    torch.testing.assert_close(packet["carries"], frozen)
    packet["carries"]["lfp"].add_(1)
    torch.testing.assert_close(original.carries, frozen)
    gradient = torch.autograd.grad(original["lfp"].sum(), field)[0]
    torch.testing.assert_close(gradient, torch.ones_like(field))


def test_population_runner_exposes_readonly_callbacks_for_reconstructed_continuation(
    tmp_path,
):
    options = dict(
        dt=DT,
        steps=HALF,
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        backend="eager",
    )
    runner = dn.compile(_model(), **options)
    receiver = dn.compile(_model(), **options)
    assert isinstance(runner.callbacks, dn.func.FunctionalCallbacks)
    with pytest.raises(AttributeError):
        runner.callbacks = None
    prefix, first = runner()
    packet = _disk(
        {
            "model": _clone(prefix),
            "callbacks": runner.callbacks.export_state(first["callbacks"].state),
        },
        tmp_path,
    )
    final, second = receiver(
        packet["model"],
        callback_state=receiver.callbacks.restore_state(packet["callbacks"]),
    )
    expected, whole = dn.compile(_model(), **{**options, "steps": STEPS})()
    torch.testing.assert_close(final, expected)
    for field in ("v", "t"):
        torch.testing.assert_close(
            torch.cat(
                (
                    first["callbacks"]["trace"][field],
                    second["callbacks"]["trace"][field],
                )
            ),
            whole["callbacks"]["trace"][field],
        )
    plain = dn.compile(_model(), dt=DT, steps=HALF, backend="eager")
    assert plain.callbacks is None


def test_restored_population_carry_keeps_suffix_gradients_without_retaining_prefix(
    tmp_path,
):
    _, functional, tensors = _case()
    source_callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("v",))})
    prefix, first = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=HALF,
        callbacks=source_callbacks,
    )
    saved = _disk(
        {
            "model": _clone(prefix),
            "callbacks": source_callbacks.export_state(first["callbacks"].state),
        },
        tmp_path,
    )
    _, receiver, received = _case()
    parameters = received.independent_parameters(trainable="*g_param")
    target = next(
        value for name, value in parameters.items() if name.endswith("g_param")
    )
    callbacks = receiver.make_callbacks({"trace": dn.func.Recorder(("v",))})
    restored = callbacks.restore_state(saved["callbacks"])
    actual, auxiliary = receiver.compile_rollout_chunk(
        HALF, execution="scan", backend="aot_eager"
    ).bind(parameters, constants=received.constants)(
        saved["model"], callbacks=callbacks, callback_state=restored
    )
    observed = torch.autograd.grad(
        auxiliary["callbacks"]["trace"]["v"].square().mean(), target
    )[0]
    expected, reference = receiver.prepare_and_rollout(
        parameters,
        received.constants,
        _clone(saved["model"]),
        steps=HALF,
        callbacks=callbacks,
        callback_state=callbacks.restore_state(saved["callbacks"]),
    )
    expected_gradient = torch.autograd.grad(
        reference["callbacks"]["trace"]["v"].square().mean(), target
    )[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(observed, expected_gradient)
    assert torch.isfinite(observed).all() and observed.abs().sum() > 0


def _configuration_pair(kind):
    model, functional, tensors = _case()
    other = functional
    field = torch.ones(model.shape)
    threshold = dict(
        node_check=[0, 3], threshold=-59.0, t_start_check=DT, t_end_check=7 * DT
    )
    if kind == "name":
        left = {"before": dn.func.APCount(**threshold)}
        right = {"after": dn.func.APCount(**threshold)}
    elif kind == "order":
        left = {"first": dn.func.AnomalyDetector(), "second": dn.func.AnomalyDetector()}
        right = dict(reversed(tuple(left.items())))
    else:
        a = b = dn.func.APCount(**threshold)
        if kind == "type":
            b = dn.func.Active(**threshold)
        elif kind in {"threshold", "start", "end", "nodes"}:
            field_name, value = {
                "threshold": ("threshold", -58.5),
                "start": ("t_start_check", 0.0),
                "end": ("t_end_check", 6 * DT),
                "nodes": ("node_check", [1, 2]),
            }[kind]
            b = dn.func.APCount(**{**threshold, field_name: value})
        elif kind == "dt":
            _, other, _ = _case(dt=2 * DT)
        elif kind == "path":
            a, b = dn.func.Recorder(("v",)), dn.func.Recorder(("i_membrane",))
        elif kind == "region":
            a = dn.func.Recorder(("v",), region=model[:, [0, 2]])
            b = dn.func.Recorder(("v",), region=model[:, [1, 3]])
        elif kind == "maximum":
            a = dn.func.Recorder(("v",), max_only=True)
            b = dn.func.Recorder(("v",), node_indices=[0])
        elif kind == "partition":
            a = dn.func.Recorder(("v",), max_only=True, partition=[1, 3])
            b = dn.func.Recorder(("v",), max_only=True, partition=[2, 2])
        elif kind == "lead_fields":
            a, b = dn.func.LFP([field]), dn.func.LFP([field * 2])
        elif kind == "lfp_rule":
            a, b = dn.func.LFP([field]), dn.func.LFP([field], rule="nc,knc->k")
        elif kind in {"at_least", "inv", "activity_partition"}:
            a = dn.func.ActiveAL(**threshold, at_least=1)
            options = {"at_least": 1}
            options.update(
                {"at_least": 2}
                if kind == "at_least"
                else {"inv": True}
                if kind == "inv"
                else {"partition": [1, 1]}
            )
            b = dn.func.ActiveAL(**threshold, **options)
        else:
            raise AssertionError(kind)
        left, right = {"analysis": a}, {"analysis": b}
    source = functional.make_callbacks(left)
    receiver = other.make_callbacks(right)
    _, auxiliary = functional.prepare_and_rollout(
        tensors.parameters, tensors.constants, tensors.state, steps=0, callbacks=source
    )
    return source.export_state(auxiliary["callbacks"].state), receiver


@pytest.mark.parametrize(
    "kind",
    (
        "name",
        "order",
        "type",
        "threshold",
        "start",
        "end",
        "nodes",
        "dt",
        "path",
        "region",
        "maximum",
        "partition",
        "lead_fields",
        "lfp_rule",
        "at_least",
        "inv",
        "activity_partition",
    ),
)
def test_restore_rejects_semantically_different_callback_configuration(kind):
    packet, receiver = _configuration_pair(kind)
    frozen = _clone(packet)
    with pytest.raises(
        dn.func.FunctionalizationError, match="configuration|compatible|checkpoint"
    ):
        receiver.restore_state(packet)
    _assert_packet_equal(packet, frozen)


def _packet():
    model, functional, tensors = _case()
    callbacks = functional.make_callbacks(
        {
            "count": dn.func.APCount(node_check=[0, 3]),
            "lfp": dn.func.LFP([torch.ones(model.shape)]),
        }
    )
    _, auxiliary = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=0,
        callbacks=callbacks,
    )
    return callbacks, callbacks.export_state(auxiliary["callbacks"].state)


@pytest.mark.parametrize(
    "kind",
    ("type", "format", "missing", "extra", "version", "boolean_version", "schemas"),
)
def test_restore_rejects_malformed_packet_envelopes(kind):
    callbacks, packet = _packet()
    if kind == "type":
        packet = []
    elif kind == "format":
        packet["format"] = "other"
    elif kind == "missing":
        del packet["configuration"]
    elif kind == "extra":
        packet["unknown"] = None
    elif kind == "version":
        packet["version"] = 999
    elif kind == "boolean_version":
        packet["version"] = True
    else:
        packet["emission_schemas"] = []
    with pytest.raises((TypeError, ValueError, dn.func.FunctionalizationError)):
        callbacks.restore_state(packet)


@pytest.mark.parametrize(
    "kind", ("keys", "structure", "leaf", "shape", "dtype", "device", "values")
)
def test_restore_rejects_invalid_tensor_carry_without_mutating_collection(kind):
    callbacks, packet = _packet()
    valid = _clone(packet)
    if kind == "keys":
        del packet["carries"]["count"]
    elif kind == "structure":
        packet["carries"]["count"] = list(packet["carries"]["count"])
    elif kind == "leaf":
        packet["carries"]["lfp"] = 1
    elif kind == "shape":
        packet["carries"]["lfp"] = packet["carries"]["lfp"][..., :2]
    elif kind == "dtype":
        packet["carries"]["lfp"] = packet["carries"]["lfp"].float()
    elif kind == "device":
        packet["carries"]["lfp"] = torch.empty_like(
            packet["carries"]["lfp"], device="meta"
        )
    else:
        packet["carries"]["lfp"].add_(1)
    with pytest.raises((TypeError, ValueError, dn.func.FunctionalizationError)):
        callbacks.restore_state(packet)
    restored = callbacks.restore_state(valid)
    torch.testing.assert_close(dict(restored.carries), valid["carries"])


def test_export_rejects_old_lfp_carry_after_bound_fields_change():
    model, functional, tensors = _case()
    field = torch.ones(model.shape, requires_grad=True)
    callbacks = functional.make_callbacks({"lfp": dn.func.LFP([field])})
    _, auxiliary = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=0,
        callbacks=callbacks,
    )
    with torch.no_grad():
        field.add_(0.5)
    with pytest.raises(
        dn.func.FunctionalizationError, match="LFP|lead.field|configuration"
    ):
        callbacks.export_state(auxiliary["callbacks"].state)


def test_raster_checkpoint_preserves_undiscovered_schema_until_first_step(tmp_path):
    _, functional, tensors = _case()
    callback = {"raster": dn.func.Raster(node_check=[0, 3])}
    callbacks = functional.make_callbacks(callback)
    receiver = functional.make_callbacks(callback)
    _, initial = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=0,
        callbacks=callbacks,
    )
    packet = _disk(callbacks.export_state(initial["callbacks"].state), tmp_path)
    assert packet["emission_schemas"] == [None]
    restored = receiver.restore_state(packet)
    actual, auxiliary = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=1,
        callbacks=receiver,
        callback_state=restored,
    )
    expected, reference = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=1,
        callbacks=callbacks,
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        dict(auxiliary["callbacks"]), dict(reference["callbacks"])
    )
    assert auxiliary["callbacks"].state.emission_schemas[0] is not None


def test_restore_rejects_well_formed_but_incompatible_discovered_emission_schema():
    _, functional, tensors = _case()
    callbacks = functional.make_callbacks(
        {"trace": dn.func.Recorder(("v",)), "raster": dn.func.Raster(node_check=[0, 3])}
    )
    _, auxiliary = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=1,
        callbacks=callbacks,
    )
    packet = callbacks.export_state(auxiliary["callbacks"].state)
    assert packet["emission_schemas"][1] is not None
    packet["emission_schemas"][1] = packet["emission_schemas"][0]
    with pytest.raises(dn.func.FunctionalizationError, match="emission|schema"):
        callbacks.restore_state(packet)


class _CustomSum(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        del auxiliary
        return state["integrator"]["v"].sum(), None

    def update(self, carry, state, auxiliary):
        del auxiliary
        return carry + state["integrator"]["v"].sum(), None

    def finalize(self, carry, emissions):
        assert emissions is None
        return carry


class _AuthoredAnomaly(dn.func.AnomalyDetector):
    pass


@pytest.mark.parametrize("configuration", (_CustomSum(), _AuthoredAnomaly()))
def test_custom_callback_execution_remains_supported_but_checkpointing_is_explicitly_rejected(
    configuration,
):
    _, functional, tensors = _case()
    callbacks = functional.make_callbacks({"custom": configuration})
    _, auxiliary = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=1,
        callbacks=callbacks,
    )
    assert auxiliary["callbacks"]["custom"] is not None
    with pytest.raises(
        dn.func.FunctionalizationError, match="custom callbacks|unsupported"
    ):
        callbacks.export_state(auxiliary["callbacks"].state)
    supported = functional.make_callbacks({"custom": dn.func.Recorder(("v",))})
    _, ordinary = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=0,
        callbacks=supported,
    )
    with pytest.raises(
        dn.func.FunctionalizationError, match="custom callbacks|unsupported"
    ):
        callbacks.restore_state(supported.export_state(ordinary["callbacks"].state))


def test_export_keeps_strict_in_memory_plan_identity_checks():
    model, functional, tensors = _case()
    first = functional.make_callbacks(_specifications(model))
    second = functional.make_callbacks(_specifications(model))
    _, auxiliary = functional.prepare_and_rollout(
        tensors.parameters, tensors.constants, tensors.state, steps=0, callbacks=first
    )
    with pytest.raises(dn.func.FunctionalizationError, match="different callback plan"):
        second.export_state(auxiliary["callbacks"].state)


@pytest.mark.parametrize(
    ("left", "right", "equal"),
    (
        (complex(float("nan"), 1), complex(float("nan"), 1), True),
        (complex(float("nan"), 1), complex(float("nan"), 2), False),
        (complex(1, float("nan")), complex(1, float("nan")), True),
        (complex(1, float("nan")), complex(2, float("nan")), False),
    ),
)
def test_checkpoint_configuration_compares_complex_nan_components_independently(
    left, right, equal
):
    from dendra.func._callback_checkpoints import _same

    actual = _same(
        {"field": torch.tensor([left], dtype=torch.complex128)},
        {"field": torch.tensor([right], dtype=torch.complex128)},
    )
    assert actual is equal


def test_numpy_integral_region_selectors_export_as_portable_plain_data(tmp_path):
    model, functional, tensors = _case()
    callbacks = functional.make_callbacks(
        {
            "trace": dn.func.Recorder(
                ("v",), region=model[:, slice(np.int64(0), np.int64(4), np.int64(2))]
            )
        }
    )
    receiver = functional.make_callbacks(
        {"trace": dn.func.Recorder(("v",), region=model[:, 0:4:2])}
    )
    _, auxiliary = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=0,
        callbacks=callbacks,
    )
    packet = callbacks.export_state(auxiliary["callbacks"].state)
    _plain(packet)
    restored = receiver.restore_state(_disk(packet, tmp_path))
    assert restored["trace"] == ()


class _String(str):
    pass


@pytest.mark.parametrize("string", (np.str_, _String))
def test_accepted_string_subclasses_export_plain_names_states_and_lfp_rules(
    tmp_path, string
):
    model, functional, tensors = _case()
    field = torch.ones(model.shape)
    callbacks = functional.make_callbacks(
        {
            string("trace"): dn.func.Recorder((string("v"), string("t"))),
            string("lfp"): dn.func.LFP([field], rule=string("nc,knc->k")),
        }
    )
    receiver = functional.make_callbacks(
        {
            "trace": dn.func.Recorder(("v", "t")),
            "lfp": dn.func.LFP([field], rule="nc,knc->k"),
        }
    )
    with torch.no_grad():
        prefix, first = functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            steps=2,
            callbacks=callbacks,
        )
        packet = callbacks.export_state(first["callbacks"].state)
        _plain(packet)
        restored = receiver.restore_state(_disk(packet, tmp_path))
        final, second = functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            prefix,
            steps=2,
            callbacks=receiver,
            callback_state=restored,
        )
        expected, whole = functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            steps=4,
            callbacks=receiver,
        )
    torch.testing.assert_close(final, expected)
    for field in ("v", "t"):
        torch.testing.assert_close(
            torch.cat(
                (
                    first["callbacks"]["trace"][field],
                    second["callbacks"]["trace"][field],
                )
            ),
            whole["callbacks"]["trace"][field],
        )
    torch.testing.assert_close(
        torch.cat((first["callbacks"]["lfp"], second["callbacks"]["lfp"])),
        whole["callbacks"]["lfp"],
    )
