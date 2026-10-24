"""Class restoration remains observational at physical and receiver audits."""

import random

import numpy as np
import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func import _population as population_api
from dendra.models.mod import pas
from tests.func.test_continuous_receive import _model as _continuous_model
from tests.func.test_continuous_receive import _Remember
from tests.func.test_event_receive import _model as _event_model
from tests.func.test_event_receive import _OrderedEvent
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _module_snapshot,
)

pytestmark = pytest.mark.cpu
DT = 0.05


def _rng_snapshot():
    return torch.random.get_rng_state(), random.getstate(), np.random.get_state()


def _restore_rng(snapshot):
    torch.random.set_rng_state(snapshot[0])
    random.setstate(snapshot[1])
    np.random.set_state(snapshot[2])


def _assert_rng(snapshot):
    current = _rng_snapshot()
    torch.testing.assert_close(current[0], snapshot[0], rtol=0, atol=0)
    assert current[1] == snapshot[1]
    assert current[2][0] == snapshot[2][0]
    np.testing.assert_array_equal(current[2][1], snapshot[2][1])
    assert current[2][2:] == snapshot[2][2:]


def _consume_rng():
    torch.rand(())
    random.random()
    np.random.random()


@pytest.fixture(autouse=True)
def _environment():
    rng, threads = _rng_snapshot(), torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0,
                REQUIRE_GRAD=1,
                IMEM=False,
                DTYPE=torch.float64,
                DEVICE="cpu",
                CHECK_PURITY=False,
            ),
        ):
            yield
    finally:
        _restore_rng(rng)
        torch.set_num_threads(threads)


def _class_values(owner, monkeypatch):
    offset = torch.arange(10.0, dtype=torch.float64)[2:8].reshape(2, 3)
    offset.requires_grad_(True)
    expanded = torch.arange(3.0, dtype=torch.float64).reshape(1, 3).expand(2, 3)
    array = np.arange(6.0, dtype=np.float64).reshape(2, 3)
    for name, value in (
        ("audit_offset", offset),
        ("audit_alias", offset),
        ("audit_expanded", expanded),
        ("audit_array", array),
    ):
        monkeypatch.setattr(owner, name, value, raising=False)
    return {
        "offset": offset,
        "expanded": expanded,
        "array": array,
        "offset_value": offset.detach().clone(),
        "expanded_value": expanded.clone(),
        "array_value": array.copy(),
        "offset_storage": offset.untyped_storage().data_ptr(),
        "expanded_storage": expanded.untyped_storage().data_ptr(),
    }


def _mutate_class(owner):
    with torch.no_grad():
        owner.audit_offset.data = torch.zeros((3, 2), dtype=torch.float32)
        owner.audit_offset.requires_grad_(False)
        if owner.audit_expanded._base is not None:
            owner.audit_expanded._base.zero_()
    owner.audit_alias = torch.ones(1)
    owner.audit_expanded = torch.ones((2, 3))
    owner.audit_array.resize((3, 2), refcheck=False)
    owner.audit_array.fill(-1)


def _assert_class(owner, saved):
    assert owner.audit_offset is saved["offset"]
    assert owner.audit_alias is saved["offset"]
    assert owner.audit_expanded is saved["expanded"]
    assert owner.audit_array is saved["array"]
    assert owner.audit_offset.dtype == torch.float64
    assert owner.audit_offset.requires_grad is True
    assert owner.audit_offset.shape == (2, 3)
    assert owner.audit_offset.stride() == (3, 1)
    assert owner.audit_offset.storage_offset() == 2
    assert owner.audit_offset.untyped_storage().data_ptr() == saved["offset_storage"]
    assert owner.audit_expanded.stride() == (0, 1)
    assert (
        owner.audit_expanded.untyped_storage().data_ptr() == saved["expanded_storage"]
    )
    torch.testing.assert_close(owner.audit_offset, saved["offset_value"])
    torch.testing.assert_close(owner.audit_expanded, saved["expanded_value"])
    np.testing.assert_array_equal(owner.audit_array, saved["array_value"])


class _PhysicalTransform(torch.nn.Module):
    def forward(self, value):
        return value


def _physical_model():
    source = dn.Unmyelinated([1.5], L=3.0, dx=1.0, dtype=torch.float64)
    source.insert(pas)
    source.register_parametrization_in_graph("rhoa", _PhysicalTransform())
    source.initialize()
    source.detach()
    return source


def _case(kind, monkeypatch, *, body_failure):
    if kind == "packed":
        source = _physical_model()
        owner, original = _PhysicalTransform, _PhysicalTransform.forward

        def authored(self, value):
            _mutate_class(owner)
            _consume_rng()
            if body_failure is not None:
                raise body_failure
            return original(self, value)

        monkeypatch.setattr(owner, "forward", authored)
        candidate = population_api._deepcopy_execution_module(source)

        def execute():
            with dn.ctx(CHECK_PURITY=True):
                return population_api._ComponentPhysicalPreparation(candidate)

        return source, owner, execute

    continuous = kind == "continuous"
    owner = _Remember if continuous else _OrderedEvent
    source = (
        _continuous_model(owner, batch=None)
        if continuous
        else _event_model(owner, batch=None)
    )
    functional, tensors = dn.func.make_functional(source, dt=DT)
    target = "analog" if continuous else "event"
    method = "continuous_receive_values" if continuous else "net_receive_values"
    owner = next(
        cls
        for cls in type(source.mech.mechanisms[target]).__mro__
        if method in vars(cls)
    )
    original = getattr(owner, method)

    def authored(self, *args, **kwargs):
        _mutate_class(owner)
        _consume_rng()
        if body_failure is not None:
            raise body_failure
        return original(self, *args, **kwargs)

    monkeypatch.setattr(owner, method, authored)

    def execute():
        return functional._audit_receivers(tensors, continuous=continuous)

    return source, owner, execute


@pytest.mark.parametrize("kind", ("packed", "event", "continuous"))
@pytest.mark.parametrize("raises", (False, True))
def test_audit_restores_class_metadata_aliases_rng_and_existing_failure(
    kind, raises, monkeypatch
):
    failure = RuntimeError("authored diagnostic body failure") if raises else None
    source, owner, execute = _case(kind, monkeypatch, body_failure=failure)
    saved = _class_values(owner, monkeypatch)
    source_before, rng_before = _module_snapshot(source), _rng_snapshot()
    error = (
        RuntimeError if raises and kind == "packed" else dn.func.FunctionalizationError
    )
    with pytest.raises(error) as caught:
        execute()
    if raises:
        if kind == "packed":
            assert caught.value is failure
        else:
            target = "analog" if kind == "continuous" else "event"
            assert str(caught.value) == (
                f"Synapse {target!r} pure receive audit failed: {failure}"
            )
            assert caught.value.__cause__ is failure
    elif kind == "packed":
        assert str(caught.value).startswith(
            "Packed component physical preparation observed authored mutation of shared Python class state "
        )
    else:
        assert str(caught.value).startswith(
            "pure receiver audit observed class mutation or RNG consumption: classes="
        )
    _assert_class(owner, saved)
    _assert_rng(rng_before)
    _assert_module_snapshot(source, source_before)


@pytest.mark.parametrize("kind", ("packed", "event", "continuous"))
def test_class_rollback_failure_overrides_body_failure_but_restores_rng(
    kind, monkeypatch
):
    body_failure = RuntimeError("authored diagnostic body failure")
    rollback_failure = RuntimeError("deliberate class rollback failure")
    source, owner, execute = _case(kind, monkeypatch, body_failure=body_failure)
    saved = _class_values(owner, monkeypatch)
    original_restore = population_api._restore_class_state

    def failing_restore(snapshot):
        original_restore(snapshot)
        _consume_rng()
        raise rollback_failure

    monkeypatch.setattr(population_api, "_restore_class_state", failing_restore)
    rng_before = _rng_snapshot()
    with pytest.raises(RuntimeError) as caught:
        execute()
    assert caught.value is rollback_failure
    _assert_class(owner, saved)
    _assert_rng(rng_before)


def test_class_snapshot_failure_restores_outer_rng_transaction(monkeypatch):
    failure = RuntimeError("deliberate class snapshot failure")
    source, owner, execute = _case("packed", monkeypatch, body_failure=None)
    saved = _class_values(owner, monkeypatch)
    source_before, rng_before = _module_snapshot(source), _rng_snapshot()

    def failing_snapshot(classes):
        _consume_rng()
        raise failure

    monkeypatch.setattr(population_api, "_class_state_snapshot", failing_snapshot)
    with pytest.raises(RuntimeError) as caught:
        execute()
    assert caught.value is failure
    _assert_class(owner, saved)
    _assert_rng(rng_before)
    _assert_module_snapshot(source, source_before)


@pytest.mark.parametrize("kind", ("packed", "event", "continuous"))
def test_disabled_audits_skip_class_and_rng_scopes(kind, monkeypatch):
    if kind == "packed":
        source = _physical_model()
        candidate = population_api._deepcopy_execution_module(source)
    else:
        source = (
            _continuous_model(_Remember, batch=None)
            if kind == "continuous"
            else _event_model(_OrderedEvent, batch=None)
        )
        with dn.ctx(CHECK_PURITY=False):
            functional, tensors = dn.func.make_functional(source, dt=DT)
    source_before, rng_before = _module_snapshot(source), _rng_snapshot()

    def unexpected_audit(*args, **kwargs):
        raise AssertionError("disabled diagnostic was entered")

    monkeypatch.setattr(population_api, "_class_state_snapshot", unexpected_audit)
    monkeypatch.setattr(population_api, "_rng_diagnostic_scope", unexpected_audit)
    monkeypatch.setattr(population_api, "_authored_workspace_classes", unexpected_audit)
    if kind == "packed":
        with dn.ctx(CHECK_PURITY=False):
            prepared = population_api._ComponentPhysicalPreparation(candidate)
        assert prepared.audit_builder_inputs is False
    else:
        # Lowering captures the policy; changing the surrounding context does
        # not enable receiver diagnostics on an existing functional plan.
        with dn.ctx(CHECK_PURITY=True):
            functional._audit_event_receivers(tensors)
    _assert_rng(rng_before)
    _assert_module_snapshot(source, source_before)
