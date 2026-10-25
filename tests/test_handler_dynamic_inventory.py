"""Dynamic handler bindings retain native metadata admission without bulk copies."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func import _population as func_api
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.mechanisms._noise_workspaces import noise_workspace_noise_values
from dendra.models.rng import RNGModule
from tests.func.test_material_stochastic_diffusion_checkpoint import (
    _population as material_population,
)
from tests.test_declared_state_inventory import _assert_unchanged, _registered_snapshot
from tests.test_handler_state_inventory import (
    _assert_bindings,
    _forbidden,
    _legacy_bindings,
)
from tests.test_mechanism_delays_pure import _resized_deferred_history
from tests.test_noise_workspace_lifecycle import _population as noise_population
from tests.test_noise_workspace_lifecycle import (
    _StateNoiseWorkspace,
    _Workspace,
)
from tests.test_runtime_noise_and_state_sde import _delayed_handler

pytestmark = pytest.mark.cpu
_KINDS = (
    "random",
    "sde",
    "single",
    "multi",
    "carry",
    "workspace",
    "state_workspace",
    "material",
)


class _RandomState(State):
    State.STATE("x")
    State.DERIVATIVE("x' = 0.0 * x")
    State.RANGERAND("fixed", seed=43, rng_name="shared_rng")
    State.RANGENOISE("eta", seed=43, rng_name="shared_rng")


class _RandomMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_RandomState)
    Mechanism.CARRY("scratch")
    Mechanism.RANGERAND("fixed", seed=43, rng_name="shared_rng")
    Mechanism.RANGENOISE("eta", seed=43, rng_name="shared_rng")


class _SDEState(State):
    State.STATE("x")
    State.DERIVATIVE("x' = 0.0 * x")
    State.DIFFUSION("x = 1.0")
    State.METHOD("euler_maruyama")


class _SDEMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_SDEState)


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(
            JIT=0,
            REQUIRE_GRAD=True,
            DEVICE="cpu",
            DTYPE=torch.float64,
            CHECK_PURITY=False,
        ),
    ):
        yield


def _case(kind, monkeypatch):
    model = old = None
    if kind in ("random", "sde"):
        shape = (1, 3)
        celsius = torch.full(shape, 36.0, dtype=torch.float64)
        diameters = torch.ones_like(celsius)
        module = (_RandomMechanism if kind == "random" else _SDEMechanism)(
            "probe", celsius, diameters, shape, shape
        )
        handler = MechanismHandler(celsius, diameters, {"probe": module})
        if kind == "random":
            handler.populate(random_generation=object())
            module.DE["_RandomState"].shared_rng = module.shared_rng
        else:
            handler.init_rng()
            module.x = torch.zeros_like(diameters)
    elif kind in ("single", "multi"):
        handler, module = _delayed_handler(batched=kind == "multi")
    elif kind == "carry":
        module, handler, old = _resized_deferred_history()
    elif kind in ("workspace", "state_workspace"):
        model, module = noise_population(
            _Workspace if kind == "workspace" else _StateNoiseWorkspace
        )
        handler = model.mech
        handler.set_dt(0.1)
    else:
        assert kind == "material"
        model = material_population("path", monkeypatch)
        handler = model.mech
        module = next(iter(handler.material_processes.values()))
        assert module._spatial_configured
    return SimpleNamespace(handler=handler, module=module, model=model, old=old)


def _rng_snapshot(handler):
    result = []
    for owner in handler.modules():
        if isinstance(owner, RNGModule):
            handles = (owner._cpu_gen, *owner._device_gens.values())
            result.append(
                (
                    owner,
                    owner._base_seed,
                    handles,
                    [handle.get_state().clone() for handle in handles],
                )
            )
    return result


def _assert_rng_unchanged(snapshot):
    for owner, seed, handles, frames in snapshot:
        current = (owner._cpu_gen, *owner._device_gens.values())
        assert type(owner._base_seed) is type(seed)
        assert owner._base_seed == seed
        assert len(current) == len(handles)
        for actual, original, frame in zip(current, handles, frames, strict=True):
            assert actual is original
            assert torch.equal(actual.get_state(), frame)


@pytest.mark.parametrize("kind", _KINDS)
def test_dynamic_bindings_match_native_admission_without_snapshot_adapters_or_bulk_clones(
    kind, monkeypatch
):
    case = _case(kind, monkeypatch)
    handler = case.handler
    expected = _legacy_bindings(handler)
    keys = tuple(func_api._canonical_handler_keys(handler).items())
    before = _registered_snapshot(handler)
    rng = _rng_snapshot(handler)
    global_rng = torch.get_rng_state().clone()
    bulk = [
        value for _owner, _name, value, _copy in before if value.is_floating_point()
    ]
    tensor_clone, torch_clone = torch.Tensor.clone, torch.clone

    def check(value):
        if value.is_floating_point() and any(
            torch._C._is_alias_of(value, source) for source in bulk
        ):
            raise AssertionError(
                "dynamic inventory cloned live floating runtime storage"
            )

    def clone(value, *args, **kwargs):
        check(value)
        return tensor_clone(value, *args, **kwargs)

    def clone_function(value, *args, **kwargs):
        check(value)
        return torch_clone(value, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(handler, "mutable_state_dict", _forbidden)
        guard.setattr(handler, "_preflight_mutable_state_dict", _forbidden)
        guard.setattr(torch.Tensor, "clone", clone)
        guard.setattr(torch, "clone", clone_function)
        guard.setattr(RNGModule, "rand", _forbidden)
        guard.setattr(RNGModule, "randn", _forbidden)
        actual = handler._mutable_state_bindings()
        actual_keys = tuple(func_api._canonical_handler_keys(handler).items())
    _assert_bindings(actual, expected)
    assert actual_keys == keys
    _assert_unchanged(before)
    _assert_rng_unchanged(rng)
    assert torch.equal(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize(
    "invalid", ("pointer", "steps", "provenance", "rng", "active", "dt")
)
def test_live_dynamic_metadata_errors_match_native_preflight_before_mutation(
    invalid, monkeypatch
):
    kind = {
        "pointer": "single",
        "steps": "multi",
        "provenance": "random",
        "rng": "random",
        "active": "workspace",
        "dt": "workspace",
    }[invalid]
    case = _case(kind, monkeypatch)
    module, handler = case.module, case.handler
    if invalid == "pointer":
        module.signal_delay_ptr.fill_(2)
    elif invalid == "steps":
        module.signal_delay_steps = torch.tensor([0, 1], dtype=torch.long)
    elif invalid == "provenance":
        module._random_parameter_initialized["fixed"] = "invalid"
    elif invalid == "rng":
        module.shared_rng._base_seed = True
    elif invalid == "active":
        module._buffers["_noise_workspace_derived_active"] = torch.tensor(
            1.0, dtype=torch.float64
        )
    else:
        module._buffers["_noise_workspace_timestep_dt"] = torch.tensor(
            float("nan"), dtype=torch.float64
        )
    before = _registered_snapshot(handler)
    rng = _rng_snapshot(handler)
    global_rng = torch.get_rng_state().clone()
    with pytest.raises((ValueError, TypeError, RuntimeError)) as legacy:
        _legacy_bindings(handler)
    with pytest.raises(type(legacy.value)) as actual:
        handler._mutable_state_bindings()
    assert str(actual.value) == str(legacy.value)
    for owner, name, original, expected in before:
        assert owner._buffers[name] is original
        torch.testing.assert_close(original, expected, rtol=0, atol=0, equal_nan=True)
    _assert_rng_unchanged(rng)
    assert torch.equal(torch.get_rng_state(), global_rng)


def test_uncommon_tensor_clone_failure_precedes_rng_payload_admission(monkeypatch):
    case = _case("random", monkeypatch)
    module, handler = case.module, case.handler
    before = _registered_snapshot(handler)
    error = RuntimeError("uncommon carry clone failed")

    class CloneFailure(torch.Tensor):
        @staticmethod
        def __new__(cls, value):
            return torch.Tensor._make_subclass(cls, value, False)

        def clone(self, *args, **kwargs):
            raise error

    special = CloneFailure(torch.zeros_like(module.diam))
    module.scratch = special
    module.shared_rng._base_seed = True
    rng = _rng_snapshot(handler)
    global_rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError) as legacy:
        _legacy_bindings(handler)
    assert legacy.value is error
    with pytest.raises(RuntimeError) as actual:
        handler._mutable_state_bindings()
    assert actual.value is error
    assert module.scratch is special
    _assert_unchanged(
        [
            entry
            for entry in before
            if not (entry[0] is module and entry[1] == "scratch")
        ]
    )
    _assert_rng_unchanged(rng)
    assert torch.equal(torch.get_rng_state(), global_rng)


def test_carry_history_restores_capacity_and_replays_differentiable_queue(monkeypatch):
    case = _case("carry", monkeypatch)
    module, handler = case.module, case.handler
    handler.restore_mutable_state_dict(case.old)
    assert module._carry_resolved_shapes["signal_delay_buffer"] == tuple(
        case.old["delay.signal_delay_buffer"].shape
    )
    signal = torch.full_like(module.diam, 3.0, requires_grad=True)
    module.delayed_state("signal", signal, mode="shift")
    checkpoint = handler.mutable_state_dict()
    expected = module.delayed_state("signal", torch.zeros_like(signal), mode="shift")
    handler.restore_mutable_state_dict(checkpoint)
    actual = module.delayed_state("signal", torch.zeros_like(signal), mode="shift")
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, signal)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum() + expected.sum(), signal)[0],
        torch.full_like(signal, 2.0),
    )


def test_held_noise_workspace_checkpoint_replays_runtime_draws_and_parameter_gradients(
    monkeypatch,
):
    case = _case("workspace", monkeypatch)
    module, handler = case.module, case.handler
    held = noise_workspace_noise_values(module, "derived")["eta"].clone()
    checkpoint = handler.mutable_state_dict()
    handler.sample_runtime_noises_(dt=0.1, phase="pre_state")
    expected = module.eta.clone()
    handler.restore_mutable_state_dict(checkpoint)
    handler.sample_runtime_noises_(dt=0.1, phase="pre_state")
    torch.testing.assert_close(module.eta, expected)
    gradient = torch.autograd.grad(
        (module.static + module.coefficient).sum(), module.scale_param
    )[0]
    torch.testing.assert_close(gradient.sum(), ((1.0 + held) * 1.1).sum())
