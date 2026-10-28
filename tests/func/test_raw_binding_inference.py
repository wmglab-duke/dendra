"""Raw Population bindings retain only safe, freshness-checked inference work."""

from __future__ import annotations

import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

import dendra as dn
from tests.func.test_raw_bindings import _case, _close
from tests.func.test_raw_bindings import _environment as _environment

pytestmark = pytest.mark.cpu


def test_raw_inference_binding_refreshes_after_default_dtype_context_changes(
    monkeypatch,
):
    _, functional, parameters, constants, state, _ = _case()
    bound = functional.bind(parameters, constants=constants)
    count = 0
    prepare = functional._prepare_source_validated

    def observe(*arguments):
        nonlocal count
        count += 1
        return prepare(*arguments)

    monkeypatch.setattr(functional, "_prepare_source_validated", observe)
    original = torch.get_default_dtype()
    changed = torch.float64 if original != torch.float64 else torch.float32
    try:
        with torch.no_grad():
            normal, _ = bound(state)
            first_cache = bound._inference_prepared
            assert first_cache is not None and count == 1
            torch.set_default_dtype(changed)
            for _ in range(2):
                actual, _ = bound(state)
                expected, _ = functional.prepare_and_rollout(
                    parameters, constants, state, steps=1
                )
                _close(actual, expected)
                assert count == 2
                assert bound._inference_prepared is not first_cache
            torch.set_default_dtype(original)
            resumed, _ = bound(state)
            _close(resumed, normal)
            assert count == 3
    finally:
        torch.set_default_dtype(original)


def test_raw_inference_binding_bypasses_autocast_without_replacing_normal_cache(
    monkeypatch,
):
    _, functional, parameters, constants, state, _ = _case()
    bound = functional.bind(parameters, constants=constants)
    count = 0
    prepare = functional._prepare_source_validated

    def observe(*arguments):
        nonlocal count
        count += 1
        return prepare(*arguments)

    monkeypatch.setattr(functional, "_prepare_source_validated", observe)
    with torch.no_grad():
        normal, _ = bound(state)
        cached = bound._inference_prepared
        assert cached is not None and count == 1
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assert torch.is_autocast_enabled("cpu")
            for iteration in range(2):
                actual, _ = bound(state)
                expected, _ = functional.prepare_and_rollout(
                    parameters, constants, state, steps=1
                )
                _close(actual, expected)
                assert count == iteration + 2
                assert bound._inference_prepared is cached
        resumed, _ = bound(state)
        _close(resumed, normal)
        assert count == 3
        assert bound._inference_prepared is cached


@pytest.mark.parametrize("kind", ("function", "dispatch"))
def test_raw_binding_preserves_authored_modes_without_reusing_or_replacing_normal_cache(
    monkeypatch, kind
):
    _, functional, parameters, constants, state, _ = _case()
    bound = functional.bind(parameters, constants=constants)
    count = 0
    prepare = functional._prepare_source_validated
    events = []

    def observe(*arguments):
        nonlocal count
        count += 1
        return prepare(*arguments)

    def invoke(function, args, kwargs):
        result = function(*args, **(kwargs or {}))
        reciprocal = (
            torch.Tensor.reciprocal
            if kind == "function"
            else torch.ops.aten.reciprocal.default
        )
        if function is reciprocal:
            # Geometry preparation derives solver coefficients using
            # reciprocals. Authored dispatch participates in that arithmetic.
            events.append("reciprocal")
            return 1.2 * result
        return result

    if kind == "function":

        class Mode(TorchFunctionMode):
            def __torch_function__(self, function, types, args=(), kwargs=None):
                return invoke(function, args, kwargs)
    else:

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                return invoke(function, args, kwargs)

    monkeypatch.setattr(functional, "_prepare_source_validated", observe)
    with torch.no_grad():
        normal, _ = bound(state)
        cached = bound._inference_prepared
        assert cached is not None and count == 1
        with Mode():
            for iteration in range(2):
                previous = len(events)
                actual, _ = bound(state)
                assert len(events) > previous
                expected, _ = functional.prepare_and_rollout(
                    parameters, constants, state, steps=1
                )
                _close(actual, expected)
                assert not torch.equal(
                    actual["integrator"]["v"], normal["integrator"]["v"]
                )
                assert count == iteration + 2
                assert bound._inference_prepared is cached
        resumed, _ = bound(state)
        _close(resumed, normal)
        assert count == 3
        assert bound._inference_prepared is cached


@pytest.mark.parametrize("mode", (torch.no_grad, torch.inference_mode))
def test_raw_inference_binding_reuses_preparation_and_refreshes_live_edits(
    monkeypatch, mode
):
    _, functional, parameters, constants, state, name = _case()
    bound = functional.bind(parameters, constants=constants)
    frames = []
    prepare = functional._prepare_source_validated

    def observe(*arguments):
        result = prepare(*arguments)
        frames.append(result)
        return result

    monkeypatch.setattr(functional, "_prepare_source_validated", observe)
    with mode():
        first, _ = bound(state)
        repeated, _ = bound(state)
        _close(first, repeated)
        assert len(frames) == 1
        assert not torch.is_inference(frames[0].values["integrator"]["diag_base"])
        for index, changed in enumerate((parameters[name], constants["diam"]), start=2):
            changed.mul_(1.1)
            actual, _ = bound(state)
            expected, _ = functional.prepare_and_rollout(
                parameters, constants, state, steps=1
            )
            _close(actual, expected)
            assert len(frames) == index
    assert bound.prepared is None


@pytest.mark.parametrize("damage", ("workspace", "foreign"))
def test_raw_inference_binding_does_not_hide_corrupt_or_foreign_preparation(damage):
    _, functional, parameters, constants, state, _ = _case()
    bound = functional.bind(parameters, constants=constants)
    with torch.no_grad():
        bound(state)
        if damage == "workspace":
            bound._inference_prepared.values["integrator"]["diag_base"].add_(1.0)
            message = "prepared workspaces changed after prepare"
        else:
            _, other, other_parameters, other_constants, _, _ = _case()
            foreign = other.prepare(other_parameters, other_constants)
            object.__setattr__(bound, "_inference_prepared", foreign)
            message = "different functional Population plan"
        with pytest.raises(dn.func.FunctionalizationError, match=message):
            bound(state)


def test_raw_binding_training_ignores_a_prior_inference_preparation():
    _, functional, parameters, constants, state, name = _case()
    bound = functional.bind(parameters, constants=constants)
    with torch.no_grad():
        bound(state)
    cached = bound._inference_prepared
    final, _ = bound(state)
    actual = torch.autograd.grad(
        final["integrator"]["v"].square().mean(), parameters[name]
    )[0]
    expected, _ = functional.prepare_and_rollout(parameters, constants, state, steps=1)
    reference = torch.autograd.grad(
        expected["integrator"]["v"].square().mean(), parameters[name]
    )[0]
    _close(actual, reference)
    assert bound._inference_prepared is cached
    assert not cached.grad_enabled_at_prepare


def test_raw_binding_keeps_inference_tensor_source_diagnostics():
    _, functional, parameters, constants, state, _ = _case()
    with torch.inference_mode():
        constants["diam"] = constants["diam"].clone()
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="inference-tensor parameters or constants.*clone them outside inference mode",
    ):
        functional.bind(parameters, constants=constants)(state)


def test_raw_binding_does_not_cache_forward_ad_preparation(monkeypatch):
    _, functional, parameters, constants, state, _ = _case()
    prepared_count = 0
    prepare = functional._prepare_source_validated

    def observe(*arguments):
        nonlocal prepared_count
        prepared_count += 1
        return prepare(*arguments)

    monkeypatch.setattr(functional, "_prepare_source_validated", observe)
    with torch.no_grad(), torch.autograd.forward_ad.dual_level():
        constants["diam"] = torch.autograd.forward_ad.make_dual(
            constants["diam"], torch.ones_like(constants["diam"])
        )
        bound = functional.bind(parameters, constants=constants)
        for _ in range(2):
            actual, _ = bound(state)
            expected, _ = functional.prepare_and_rollout(
                parameters, constants, state, steps=1
            )
            _close(actual, expected)
            tangent = torch.autograd.forward_ad.unpack_dual(
                actual["integrator"]["v"]
            ).tangent
            reference = torch.autograd.forward_ad.unpack_dual(
                expected["integrator"]["v"]
            ).tangent
            assert tangent is not None and tangent.abs().sum() > 0
            _close(tangent, reference)
        assert prepared_count == 2
        assert bound._inference_prepared is None


@pytest.mark.parametrize("kind", ("subclass", "nonleaf"))
def test_raw_binding_keeps_tensor_dispatch_and_nonleaf_input_lifetimes(
    monkeypatch, kind
):
    _, functional, parameters, constants, state, _ = _case()

    class AuthoredTensor(torch.Tensor):
        pass

    if kind == "subclass":
        constants["diam"] = constants["diam"].as_subclass(AuthoredTensor)
    else:
        constants["diam"] = constants["diam"].requires_grad_() * 1.0
        assert constants["diam"].grad_fn is not None
    bound = functional.bind(parameters, constants=constants)
    assert bound.constants["diam"] is constants["diam"]
    prepared_count = 0
    prepare = functional._prepare_source_validated

    def observe(*arguments):
        nonlocal prepared_count
        prepared_count += 1
        return prepare(*arguments)

    monkeypatch.setattr(functional, "_prepare_source_validated", observe)
    with torch.no_grad():
        for _ in range(2):
            actual, _ = bound(state)
            expected, _ = functional.prepare_and_rollout(
                parameters, constants, state, steps=1
            )
            _close(actual, expected)
    assert prepared_count == 2
    assert bound._inference_prepared is None
