"""Invocation capability probes retain their restore-only transaction policy."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import dendra as dn
from dendra.func import _poisson_invocations as poisson
from dendra.func import _population as population
from dendra.func import _rng_diagnostics as rng
from dendra.func import _stimulation_rng as stimulation

pytestmark = pytest.mark.cpu
_CPU = torch.device("cpu")
_SCENARIOS = (
    "disabled",
    "rng-setup",
    "class-setup",
    "body",
    "class-restore",
    "rng-restore",
    "late-device",
)


@pytest.fixture(autouse=True)
def _environment():
    before = rng._global_rng_snapshot(_CPU)
    try:
        with dn.ctx(JIT=0, REQUIRE_GRAD=True, DEVICE="cpu", CHECK_PURITY=False):
            yield
    finally:
        rng._restore_global_rng(before, _CPU)


class _FalsySnapshot(dict):
    def __bool__(self):
        return False


def _case(monkeypatch, endpoint, scenario):
    events = []
    real_rng_snapshot = population._global_rng_snapshot
    real_class_snapshot = population._class_state_snapshot
    real_class_restore = population._restore_class_state
    real_rng_restore = population._restore_global_rng
    real_rng_changes = rng._global_rng_changes
    before = real_rng_snapshot(_CPU)
    original = torch.tensor([2.0, 3.0], requires_grad=True)
    value = original.detach().clone()

    class Authored:
        tensor = original
        alias = original

    case = SimpleNamespace(
        events=events,
        before=before,
        errors={name: RuntimeError(name) for name in _SCENARIOS if name != "disabled"},
        body_error=RuntimeError("probe body"),
        rng_snapshot=None,
        class_snapshot=None,
        initial_device=torch.device("cpu", 0),
        cleanup_device=torch.device("cpu", 1),
        rng_changes=lambda: real_rng_changes(before, _CPU),
        class_restored=lambda: Authored.tensor is original
        and Authored.alias is original
        and torch.equal(original, value),
    )
    transition = torch.nn.Module()
    entry = SimpleNamespace(
        kind="mechanism", key="poisson", mapping="population.waveform"
    )
    sde = SimpleNamespace(key="sde", path="state", name="x")
    poisson_plan = SimpleNamespace(
        entries=(entry,), _entries_for_phase=lambda phase: (entry,)
    )

    class Functional:
        _check_purity = scenario != "disabled"
        _poisson_plan = poisson_plan
        _noise_plan = SimpleNamespace(sde=SimpleNamespace(entries=(sde,)))
        _transition = transition

        def __init__(self):
            self.selected_device = case.initial_device

        @property
        def device(self):
            events.append(f"device:{self.selected_device.index}")
            return self.selected_device

        def _prepare_values(self, parameters, constants):
            events.append("body")
            assert parameters == constants == {}
            assert not torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
            mutate()
            if scenario == "late-device":
                self.selected_device = case.cleanup_device
            raise case.body_error

    functional = Functional()

    def mutate():
        torch.rand(())
        random.random()
        np.random.random()
        with torch.no_grad():
            Authored.tensor.add_(10)
        Authored.alias = torch.zeros_like(original)

    def snapshot_rng(device):
        events.append("rng-snapshot")
        assert device is case.initial_device
        if scenario == "rng-setup":
            mutate()
            raise case.errors[scenario]
        saved = real_rng_snapshot(device)
        case.rng_snapshot = _FalsySnapshot(saved) if scenario == "body" else saved
        return case.rng_snapshot

    def authored_classes(module):
        events.append("classes")
        assert module is transition
        return (Authored,)

    def snapshot_classes(classes):
        events.append("class-snapshot")
        assert classes == (Authored,)
        if scenario == "class-setup":
            mutate()
            raise case.errors[scenario]
        saved = real_class_snapshot(classes)
        case.class_snapshot = _FalsySnapshot(saved) if scenario == "body" else saved
        return case.class_snapshot

    def restore_classes(saved):
        events.append("class-restore")
        assert saved is case.class_snapshot
        if scenario in ("class-restore", "rng-restore"):
            raise case.errors["class-restore"]
        return real_class_restore(saved)

    def restore_rng(saved, device):
        events.append("rng-restore")
        assert saved is case.rng_snapshot
        assert device is (
            case.cleanup_device if scenario == "late-device" else case.initial_device
        )
        if scenario == "rng-restore":
            raise case.errors[scenario]
        return real_rng_restore(saved, device)

    def forbidden_findings(*args, **kwargs):
        pytest.fail("restore-only invocation probes must not compare RNG findings")

    monkeypatch.setattr(population, "_global_rng_snapshot", snapshot_rng)
    monkeypatch.setattr(population, "_authored_workspace_classes", authored_classes)
    monkeypatch.setattr(population, "_class_state_snapshot", snapshot_classes)
    monkeypatch.setattr(population, "_restore_class_state", restore_classes)
    monkeypatch.setattr(population, "_restore_global_rng", restore_rng)
    monkeypatch.setattr(rng, "_global_rng_changes", forbidden_findings)
    tensors = SimpleNamespace(parameters={}, constants={}, state={})

    def invoke():
        # Capability probing follows the policy captured by lowering even when
        # the ambient context requests the opposite diagnostic behavior.
        with dn.ctx(CHECK_PURITY=not functional._check_purity):
            if endpoint == "poisson":
                return poisson.validate_poisson_invocations(functional, tensors)
            return stimulation._kernel_requests(functional, tensors)

    case.invoke = invoke
    case.expected_error = (
        case.errors[scenario]
        if scenario in ("rng-setup", "class-setup", "class-restore", "rng-restore")
        else case.body_error
    )
    case.expected_events = [
        "device:0",
        "rng-snapshot",
        "classes",
        "class-snapshot",
        "body",
        "class-restore",
        f"device:{1 if scenario == 'late-device' else 0}",
        "rng-restore",
    ]
    if scenario == "disabled":
        case.expected_events = ["body"]
    elif scenario == "rng-setup":
        case.expected_events = case.expected_events[:2]
    elif scenario == "class-setup":
        case.expected_events = case.expected_events[:4]
    return case


@pytest.mark.parametrize("endpoint", ("poisson", "shared-kernel"))
@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_invocation_probe_preserves_snapshot_gates_restore_order_and_error_identity(
    monkeypatch, endpoint, scenario
):
    case = _case(monkeypatch, endpoint, scenario)
    with pytest.raises(RuntimeError) as caught:
        case.invoke()
    assert caught.value is case.expected_error
    assert case.events == case.expected_events
    if scenario in ("body", "class-restore", "late-device"):
        assert case.rng_changes() == set()
    else:
        assert case.rng_changes() == {"global PyTorch", "global Python", "global NumPy"}
    assert case.class_restored() is (scenario in ("body", "late-device"))
    if scenario == "body":
        assert not case.rng_snapshot and not case.class_snapshot


def test_empty_and_single_family_capability_paths_do_not_enter_diagnostics(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a capability fast path entered diagnostic setup")

    for name in (
        "_global_rng_snapshot",
        "_authored_workspace_classes",
        "_class_state_snapshot",
        "_restore_class_state",
        "_restore_global_rng",
    ):
        monkeypatch.setattr(population, name, forbidden)
    assert (
        poisson.validate_poisson_invocations(SimpleNamespace(_poisson_plan=None), None)
        is None
    )
    empty = SimpleNamespace(entries=(SimpleNamespace(kind="intra"),))
    assert (
        poisson.validate_poisson_invocations(SimpleNamespace(_poisson_plan=empty), None)
        is None
    )
    entry = object()
    poisson_only = SimpleNamespace(
        _noise_plan=None,
        _poisson_plan=SimpleNamespace(_entries_for_phase=lambda phase: (entry,)),
    )
    sde_only = SimpleNamespace(
        _poisson_plan=None,
        _noise_plan=SimpleNamespace(sde=SimpleNamespace(entries=(entry,))),
    )
    assert stimulation._kernel_requests(poisson_only, None) == (("poisson", entry),)
    assert stimulation._kernel_requests(sde_only, None) == (("sde", entry),)
