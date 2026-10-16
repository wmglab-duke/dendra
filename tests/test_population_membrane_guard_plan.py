"""Segment-local membrane structure proofs must not cache numerical policy.

The counting compiler avoids new Inductor compilation. Full native/C++ traces
and all eight gradient targets are additionally checked by the CPU campaign.
"""

from __future__ import annotations

import copy
import pickle
import sys
from contextlib import contextmanager

import pytest
import torch

import dendra as dn
from dendra.models.callbacks import Callback
from dendra.models.integrators.core import Integrator
from dendra.models.mechanisms._handler import MechanismHandler
from tests.test_membrane_assembly_context import InconsistentAffine
from tests.test_membrane_frame_mechanisms import (
    Affine,
    MixedAssigned,
    Nonlinear,
    Saved,
    handler,
)
from tests.test_population_eligibility_cache import (
    _assert_traces_equal,
    _execute_full,
)
from tests.test_population_fused_intra import _case, _counting_compiler, _NoopPreStep

pytestmark = pytest.mark.cpu


class _Owner:
    jit = False
    jit_network_solves = False
    backend = "eager"
    fullgraph = True
    dynamic = False
    compile_mode = None
    compile_options = None

    def device(self):
        return torch.device("cpu")


def _integrator(mechanisms, *, imem=False):
    return Integrator(_Owner(), mechanisms, imem=imem)


@contextmanager
def _observe_membrane(model, active=()):
    """Observe canonical methods without invalidating executable provenance."""
    methods = {
        MechanismHandler._can_evaluate_direct_membrane_frame.__code__: "proof",
        MechanismHandler._evaluate_membrane_frame.__code__: "frame",
        MechanismHandler._direct_membrane_dtype_compatible.__code__: "dtype",
    }
    events = {
        "proof": [],
        "frame": [],
        "dtype": [],
        "proof_results": [],
        "direct_results": [],
    }
    previous = sys.getprofile()

    def profile(frame, event, arg):
        name = methods.get(frame.f_code)
        if frame.f_locals.get("self") is model.mech and name is not None:
            if event == "call":
                events[name].append(
                    {
                        "compiled": bool(active),
                        "prevalidated": frame.f_locals.get(
                            "_structure_prevalidated", False
                        ),
                    }
                )
            elif event == "return" and name == "proof":
                events["proof_results"].append(arg)
            elif event == "return" and name == "frame" and arg is not None:
                events["direct_results"].append(arg[2] is None)
        if callable(previous):
            previous(frame, event, arg)

    sys.setprofile(profile)
    try:
        yield events
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize("entrypoint, segments", [("run", 1), ("longrun", 8)])
def test_full_hh_recording_proves_structure_once_per_host_segment(
    monkeypatch, entrypoint, segments
):
    model, _ = _case()
    model.integrator.imem = False
    _, active = _counting_compiler(monkeypatch)
    with _observe_membrane(model, active) as events:
        actual = _execute_full(model, entrypoint, chunklength=1)
    assert len(events["proof"]) == segments
    assert not any(event["compiled"] for event in events["proof"])
    assert events["proof_results"] == [True] * segments
    assert len(events["frame"]) == len(events["dtype"]) == 8
    assert all(event["prevalidated"] for event in events["frame"])
    assert events["direct_results"] == [True] * 8
    assert model.integrator._direct_membrane_structure_prevalidated is False

    reference, _ = _case()
    reference.integrator.imem = False
    expected = _execute_full(
        reference, entrypoint, callbacks=[_NoopPreStep()], chunklength=1
    )
    _assert_traces_equal(actual, expected)


@pytest.mark.parametrize("mutation", ["callback", "sampler", "getter"])
def test_authored_execution_retains_per_step_structural_checks(monkeypatch, mutation):
    model, _ = _case()
    model.integrator.imem = False
    _counting_compiler(monkeypatch)
    callbacks, reads = [], []
    if mutation == "callback":
        callbacks.append(_NoopPreStep())
    elif mutation == "sampler":
        original = model.mech.sample_runtime_noises_

        def sample(*args, **kwargs):
            reads.append("sample")
            return original(*args, **kwargs)

        model.mech.sample_runtime_noises_ = sample
    else:
        model.setreference(
            "unused_authored_getter", lambda: reads.append("getter") or model.v
        )

    with _observe_membrane(model) as events:
        _execute_full(model, "run", callbacks=callbacks)
    assert len(events["proof"]) == 8
    assert not any(event["prevalidated"] for event in events["frame"])
    assert model.integrator._direct_membrane_structure_prevalidated is False
    assert reads == (["sample"] * 8 if mutation == "sampler" else [])


def test_nested_and_exceptional_scopes_restore_the_enclosing_proof():
    model, _ = _case()
    integrator = model.integrator
    integrator.imem = False
    assert integrator._direct_membrane_structure_prevalidated is False
    with pytest.raises(RuntimeError, match="scope failure"):
        with integrator._population_step_scope(stable=True):
            assert integrator._direct_membrane_structure_prevalidated is True
            with integrator._population_step_scope():
                assert integrator._direct_membrane_structure_prevalidated is False
            assert integrator._direct_membrane_structure_prevalidated is True
            with integrator._population_step_scope(stable=True):
                assert integrator._direct_membrane_structure_prevalidated is True
            raise RuntimeError("scope failure")
    assert integrator._direct_membrane_structure_prevalidated is False


def test_serialization_never_exports_an_active_structure_proof():
    # A minimal owner avoids unrelated dynamically generated HH proxy methods.
    integrator = Integrator.__new__(Integrator)
    torch.nn.Module.__init__(integrator)
    integrator._direct_membrane_structure_prevalidated = True
    integrator._compiled_kernels = {"unpickleable": lambda: None}
    state = integrator.__getstate__()
    assert state["_direct_membrane_structure_prevalidated"] is False
    assert state["_compiled_kernels"] == {}
    for restored in (copy.deepcopy(integrator), pickle.loads(pickle.dumps(integrator))):
        assert restored._direct_membrane_structure_prevalidated is False
        assert restored._compiled_kernels == {}
    assert integrator._direct_membrane_structure_prevalidated is True
    assert integrator._compiled_kernels


@pytest.mark.parametrize("mechanism", [Saved, Nonlinear])
def test_saved_or_nonaffine_currents_cannot_acquire_structure_proof(mechanism):
    obj, voltage = handler(mechanism)
    integrator = _integrator(obj)
    with integrator._population_step_scope(stable=True):
        assert integrator._direct_membrane_structure_prevalidated is False
        assert integrator._membrane_frame(voltage, call_local=True)[2] is not None


def test_context_and_membrane_current_requirements_stay_live_inside_scope():
    obj, voltage = handler()
    integrator = _integrator(obj)
    with integrator._population_step_scope(stable=True):
        assert integrator._direct_membrane_structure_prevalidated is True
        assert integrator._membrane_frame(voltage, call_local=True)[2] is None
        assert (
            integrator._membrane_frame(voltage, call_local=True, require_current=True)[
                2
            ]
            is not None
        )
        with dn.ctx(MEMBRANE_ASSEMBLY="off"):
            assert integrator._membrane_frame(voltage, call_local=True)[2] is not None
        assert integrator._membrane_frame(voltage, call_local=True)[2] is None
    with (
        dn.ctx(MEMBRANE_ASSEMBLY="off"),
        integrator._population_step_scope(stable=True),
    ):
        assert integrator._direct_membrane_structure_prevalidated is False
    integrator.imem = True
    with integrator._population_step_scope(stable=True):
        assert integrator._direct_membrane_structure_prevalidated is False
        assert integrator._membrane_frame(voltage, call_local=True)[2] is not None


@pytest.mark.parametrize("mixed", ["parameter", "assigned_result"])
def test_structure_proof_does_not_skip_dtype_or_result_precision_checks(mixed):
    obj, voltage = handler(
        Affine if mixed == "parameter" else MixedAssigned,
        dtype=torch.float64 if mixed == "parameter" else torch.float32,
    )
    if mixed == "parameter":
        obj.m.g = torch.ones_like(obj.m.g, dtype=torch.float32)
    else:
        # Authored ASSIGNED outputs must match their workspace's own schema;
        # that workspace may legally differ from voltage precision.
        obj.m._buffers["effective_g"] = obj.m.effective_g.double()
    integrator = _integrator(obj)
    with integrator._population_step_scope(stable=True):
        assert integrator._direct_membrane_structure_prevalidated is True
        actual = integrator._membrane_frame(voltage, call_local=True)
        assert actual[2] is not None
        assert actual[0].dtype == actual[1].dtype == actual[2].dtype == voltage.dtype


def test_validation_policy_is_not_bypassed_by_structure_proof():
    obj, voltage = handler(InconsistentAffine)
    integrator = _integrator(obj)
    with integrator._population_step_scope(stable=True):
        assert integrator._direct_membrane_structure_prevalidated is True
        with dn.ctx(MEMBRANE_ASSEMBLY="validate"):
            with pytest.raises(RuntimeError, match="Invalid AFFINE"):
                integrator._membrane_frame(voltage, call_local=True)
    assert integrator._direct_membrane_structure_prevalidated is False


def test_direct_functional_style_handler_calls_revalidate_by_default():
    obj, voltage = handler()
    integrator = _integrator(obj)
    proxy = type("OwnerProxy", (), {"mech": obj})()
    with _observe_membrane(proxy) as events:
        for _ in range(2):
            assert obj._evaluate_membrane_frame(voltage, call_local=True)[2] is None
    assert len(events["proof"]) == 2
    assert not any(event["prevalidated"] for event in events["frame"])
    assert integrator._direct_membrane_structure_prevalidated is False


def test_new_host_call_rechecks_current_observer_metadata(monkeypatch):
    model, _ = _case()
    model.integrator.imem = False
    _counting_compiler(monkeypatch)
    with _observe_membrane(model) as first:
        _execute_full(model, "run")
    assert first["proof_results"] == [True]
    # A declared observer invalidates the structure even if HH's authored
    # current code itself has not changed. This must not reuse the last call.
    model.mech.hh._save = {"i"}
    with _observe_membrane(model) as second:
        _execute_full(model, "run")
    assert second["proof_results"] == [False] * 9  # scope, then every step
    assert not any(event["prevalidated"] for event in second["frame"])
    assert second["direct_results"] == [False] * 8
    assert model.integrator._direct_membrane_structure_prevalidated is False


def test_chunk_hook_mutation_is_visible_before_next_transition(monkeypatch):
    model, _ = _case()
    model.integrator.imem = False
    _counting_compiler(monkeypatch)

    class ObserveCurrentAfterFirstChunk(Callback):
        def __init__(self):
            super().__init__()
            self.chunks = 0

        def pre_chunk_hook(self, population, timepoints=None):
            self.chunks += 1
            if self.chunks == 2:
                population.mech.hh._save = {"i"}

    callback = ObserveCurrentAfterFirstChunk()
    with _observe_membrane(model) as events:
        _execute_full(model, "longrun", callbacks=[callback], chunklength=3)
    # Unknown hooks retain the ordinary per-step checks, including after a
    # metadata mutation; an earlier positive result cannot authorize the rest.
    assert callback.chunks == 3
    assert events["proof_results"] == [True] * 3 + [False] * 5
    assert events["direct_results"] == [True] * 3 + [False] * 5
    assert not any(event["prevalidated"] for event in events["frame"])
    assert model.integrator._direct_membrane_structure_prevalidated is False


def test_custom_handler_frame_keeps_its_old_keyword_contract():
    obj, voltage = handler()
    original = obj._evaluate_membrane_frame
    calls = []

    def old_signature(value, *, call_local=False, require_current=False):
        calls.append((call_local, require_current))
        return original(value, call_local=call_local, require_current=require_current)

    obj._evaluate_membrane_frame = old_signature
    integrator = _integrator(obj)
    with integrator._population_step_scope(stable=True):
        assert integrator._direct_membrane_structure_prevalidated is False
        integrator._membrane_frame(voltage, call_local=True)
    assert calls == [(True, False)]


def test_legacy_unpickle_without_instance_marker_defaults_to_unvalidated():
    # Python's unpickler restores nn.Module state through __setstate__. Model
    # files created before this marker existed have no such dictionary entry.
    original = Integrator.__new__(Integrator)
    torch.nn.Module.__init__(original)
    original._compiled_kernels = {}
    original.imem = False
    legacy_state = original.__getstate__()
    legacy_state.pop("_direct_membrane_structure_prevalidated")
    restored = Integrator.__new__(Integrator)
    restored.__setstate__(legacy_state)
    assert "_direct_membrane_structure_prevalidated" not in restored.__dict__
    assert restored._direct_membrane_structure_prevalidated is False

    obj, voltage = handler()
    restored.mech = obj
    proxy = type("LegacyOwnerProxy", (), {"mech": obj})()
    with _observe_membrane(proxy) as ordinary:
        assert restored._membrane_frame(voltage, call_local=True)[2] is None
    assert len(ordinary["proof"]) == 1
    assert ordinary["frame"][0]["prevalidated"] is False
    with restored._population_step_scope(stable=True):
        assert restored._direct_membrane_structure_prevalidated is True
        assert restored._membrane_frame(voltage, call_local=True)[2] is None
    assert restored._direct_membrane_structure_prevalidated is False


@pytest.mark.parametrize(
    "replacement",
    [
        "currents_dict",
        "current_names_list",
        "save_set",
        "truth_flag",
        "owners_tuple",
        "registry_dict",
    ],
)
def test_authored_structural_metadata_fails_closed_without_executing_hooks(replacement):
    obj, _ = handler()
    integrator = _integrator(obj)
    authored = []

    class AuthoredDict(dict):
        def items(self):
            authored.append("dict.items")
            return super().items()

        def __iter__(self):
            authored.append("dict.iter")
            return super().__iter__()

    class AuthoredList(list):
        def __iter__(self):
            authored.append("list.iter")
            return super().__iter__()

    class AuthoredSet(set):
        def __iter__(self):
            authored.append("set.iter")
            return super().__iter__()

        def __bool__(self):
            authored.append("set.bool")
            return bool(len(self))

    class AuthoredTuple(tuple):
        def __iter__(self):
            authored.append("tuple.iter")
            return super().__iter__()

    class AuthoredTruth:
        def __bool__(self):
            authored.append("flag.bool")
            return True

    if replacement == "currents_dict":
        obj.currents = AuthoredDict(obj.currents)
    elif replacement == "current_names_list":
        obj.currents = {
            category: {name: AuthoredList(names) for name, names in mapping.items()}
            for category, mapping in obj.currents.items()
        }
    elif replacement == "save_set":
        obj.m._save = AuthoredSet()
    elif replacement == "truth_flag":
        obj._membrane_map_supported = AuthoredTruth()
    elif replacement == "owners_tuple":
        obj._membrane_owners = AuthoredTuple(obj._membrane_owners)
    else:
        obj.mechanisms._modules = AuthoredDict(obj.mechanisms._modules)
    authored.clear()

    proxy = type("MetadataOwnerProxy", (), {"mech": obj})()
    with _observe_membrane(proxy) as observed:
        with integrator._population_step_scope(stable=True):
            assert integrator._direct_membrane_structure_prevalidated is False
    assert not observed["proof"], (
        "Reject executable metadata before the structural validator walks it"
    )
    assert not authored, (
        "A fail-closed absence proof must not invoke authored iteration or truth"
    )
    assert integrator._direct_membrane_structure_prevalidated is False


@pytest.mark.parametrize("mode_kind", ["torch_function", "torch_dispatch"])
def test_authored_torch_modes_disable_structural_prevalidation(mode_kind):
    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import TorchDispatchMode

    obj, voltage = handler()
    integrator = _integrator(obj)
    dispatched = []

    class AuthoredFunctionMode(TorchFunctionMode):
        def __torch_function__(self, function, types, args=(), kwargs=None):
            dispatched.append("function")
            return function(*args, **(kwargs or {}))

    class AuthoredDispatchMode(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            dispatched.append("dispatch")
            return function(*args, **(kwargs or {}))

    mode = (
        AuthoredFunctionMode()
        if mode_kind == "torch_function"
        else AuthoredDispatchMode()
    )
    expected = integrator._membrane_frame(voltage, call_local=True)
    with mode, integrator._population_step_scope(stable=True):
        assert integrator._direct_membrane_structure_prevalidated is False
        actual = integrator._membrane_frame(voltage, call_local=True)
    assert dispatched, "The authored mode must remain installed and execute normally"
    for actual_value, expected_value in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
    assert integrator._direct_membrane_structure_prevalidated is False


def test_saved_tensor_pack_unpack_hooks_disable_structural_prevalidation():
    obj, _ = handler()
    integrator = _integrator(obj)
    calls = []

    def pack(value):
        calls.append("pack")
        return value.detach()

    def unpack(value):
        calls.append("unpack")
        return value

    value = torch.tensor([2.0], requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        with integrator._population_step_scope(stable=True):
            assert integrator._direct_membrane_structure_prevalidated is False
            loss = value.square().sum()
            (gradient,) = torch.autograd.grad(loss, value)
    torch.testing.assert_close(gradient, torch.tensor([4.0]), rtol=0, atol=0)
    assert calls == ["pack", "unpack"]
    assert integrator._direct_membrane_structure_prevalidated is False


def test_canonical_device_context_keeps_structure_prevalidation_available():
    obj, voltage = handler()
    integrator = _integrator(obj)
    expected = integrator._membrane_frame(voltage, call_local=True)
    with torch.device("cpu"), integrator._population_step_scope(stable=True):
        assert integrator._direct_membrane_structure_prevalidated is True
        actual = integrator._membrane_frame(voltage, call_local=True)
    for actual_value, expected_value in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
    assert integrator._direct_membrane_structure_prevalidated is False


@pytest.mark.parametrize("failure", ["missing", "raises"])
@pytest.mark.parametrize(
    "owner_name, api_name",
    [
        ("C", "_len_torch_dispatch_stack"),
        ("C", "_len_torch_function_stack"),
        ("C", "_get_function_stack_at"),
        ("autograd", "_top_saved_tensors_default_hooks"),
        ("ops", "_len_torch_dispatch_stack_pre_dispatch"),
    ],
)
def test_missing_or_failing_mode_introspection_falls_back_without_numerical_change(
    monkeypatch, owner_name, api_name, failure
):
    import torch._ops

    obj, voltage = handler()
    integrator = _integrator(obj)
    expected = integrator._membrane_frame(voltage, call_local=True)
    owner = {"C": torch._C, "autograd": torch._C._autograd, "ops": torch._ops}[
        owner_name
    ]

    def unavailable(*args, **kwargs):
        raise RuntimeError("mode introspection unavailable")

    # An ordinary device scope also exercises function-stack entry inspection;
    # with an empty stack, a broken _get_function_stack_at need not be called.
    with torch.device("cpu"), monkeypatch.context() as patched:
        if failure == "missing":
            patched.delattr(owner, api_name)
        else:
            patched.setattr(owner, api_name, unavailable)
        with integrator._population_step_scope(stable=True):
            assert integrator._direct_membrane_structure_prevalidated is False
            # Restore Torch's introspection API before ordinary numerical calls.
            # The already-established conservative scope must remain false.
            patched.undo()
            actual = integrator._membrane_frame(voltage, call_local=True)
    for actual_value, expected_value in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
    assert integrator._direct_membrane_structure_prevalidated is False


@pytest.mark.parametrize("malformed", [None, -1, True])
def test_unrecognized_mode_stack_introspection_fails_closed(monkeypatch, malformed):
    obj, _ = handler()
    integrator = _integrator(obj)
    monkeypatch.setattr(torch._C, "_len_torch_dispatch_stack", lambda: malformed)
    with integrator._population_step_scope(stable=True):
        assert integrator._direct_membrane_structure_prevalidated is False


@pytest.mark.parametrize("ambient", ["torch_function", "torch_dispatch", "saved_hooks"])
def test_ambient_torch_execution_disables_central_segment_cache(monkeypatch, ambient):
    from collections import Counter

    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import TorchDispatchMode

    from tests.test_population_fused_intra import _count_handler_noise_calls

    class PassthroughFunctionMode(TorchFunctionMode):
        def __torch_function__(self, function, types, args=(), kwargs=None):
            return function(*args, **(kwargs or {}))

    class PassthroughDispatchMode(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            return function(*args, **(kwargs or {}))

    contexts = {
        "torch_function": PassthroughFunctionMode,
        "torch_dispatch": PassthroughDispatchMode,
        "saved_hooks": lambda: torch.autograd.graph.saved_tensors_hooks(
            lambda value: value.detach(), lambda value: value
        ),
    }
    model, _ = _case()
    model.delete_injections()
    model.integrator.imem = False
    compiled, _ = _counting_compiler(monkeypatch)
    # The no-injection path makes skipped empty-noise sampling observable:
    # unknown ambient execution must retain all eight ordinary sampler calls.
    with contexts[ambient](), _observe_membrane(model) as observed:
        with _count_handler_noise_calls() as sampled:
            actual = _execute_full(model, "run")
    assert len(sampled) == 8
    assert Counter(compiled) == {"_step": 8}
    assert len(observed["proof"]) == 8
    assert not any(event["prevalidated"] for event in observed["frame"])
    assert model.integrator._direct_membrane_structure_prevalidated is False

    reference, _ = _case()
    reference.delete_injections()
    reference.integrator.imem = False
    expected = _execute_full(reference, "run")
    _assert_traces_equal(actual, expected)
