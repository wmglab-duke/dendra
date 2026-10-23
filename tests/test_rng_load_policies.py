"""Model loads and explicit runtime checkpoints own different RNG policies."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.utils import _pytree

import dendra as dn
from dendra.models.mechanisms import Mechanism
from dendra.models.rng import RNGModule, _device_seed, should_restore_rng

DT = 0.025
DTYPE = torch.float64
pytestmark = pytest.mark.cpu


class _StochasticPopulation(dn.SingleCompartment):
    dn.SingleCompartment.GLOBALRAND(
        "trial", mu=0.2, sigma=0.01, seed=43, rng_name="stream"
    )
    dn.SingleCompartment.GLOBALNOISE("initial_noise", seed=44, rng_name="stream")


class _StochasticLeak(Mechanism):
    Mechanism.GLOBALRAND("g", mu=0.001, sigma=0.00001, seed=45, rng_name="stream")
    Mechanism.RANGENOISE("e", mu=-65.0, sigma=0.1, seed=46, rng_name="stream")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e.clone()


@pytest.fixture(autouse=True)
def _environment():
    with dn.ctx(JIT=0, REQUIRE_GRAD=0, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        yield


def _close(actual, expected):
    left, spec = _pytree.tree_flatten(actual)
    right, other = _pytree.tree_flatten(expected)
    assert spec == other
    for value, reference in zip(left, right, strict=True):
        if torch.is_tensor(value):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        else:
            assert value == reference


def _population():
    cell = _StochasticPopulation(N=1, C=2, integrator=dn.bwd_euler_sc(imem=False))
    cell.insert(_StochasticLeak.rename("probe"))
    cell.initialize()
    cell.mech.probe.stream = cell.stream
    cell.initialize()
    return cell


def _network():
    cells = {name: _population() for name in ("left", "right")}
    stream = cells["left"].stream
    for cell in cells.values():
        cell.stream = stream
        cell.mech.probe.stream = stream
    model = dn.Network(cells)
    model.initialize(DT)
    return model


def _legacy(checkpoint, kind):
    if kind == "network":
        for child in checkpoint["populations"].values():
            _legacy(child, "population")
    elif kind == "population":
        checkpoint["stochastic"]["rng_states"] = {
            name: payload["rng_state"]
            for name, payload in checkpoint["stochastic"]["rng_states"].items()
        }
        _legacy(checkpoint["mech"], "handler")
    else:
        for name, payload in tuple(checkpoint.items()):
            if name.endswith("rng.stream"):
                checkpoint[name] = payload["rng_state"]
    return checkpoint


@pytest.mark.parametrize("policy", ("cache", "checkpoint"))
@pytest.mark.parametrize("ignore", (False, True))
def test_rng_restore_policy_selects_saved_stream(policy, ignore):
    assert should_restore_rng(policy=policy, ignore_rng_on_load=ignore) == (
        policy == "checkpoint" or not ignore
    )


def test_rng_restore_rejects_unknown_policy_even_when_loads_are_ignored():
    stream = RNGModule(11, (), ())
    stream.ignore_rng_on_load()
    with pytest.raises(ValueError, match="restoration policy"):
        stream.restore_rng_checkpoint(None, policy="unknown")


def test_ignored_model_load_skips_malformed_payload_but_runtime_restore_validates_it():
    stream = RNGModule(11, (), ())
    stream.rand((3,))
    stream.ignore_rng_on_load()
    before = copy.deepcopy(stream.get_extra_state())
    stream.load_state_dict({"_extra_state": None})
    stream.restore_rng_checkpoint({"malformed": object()}, policy="cache")
    _close(stream.get_extra_state(), before)
    with pytest.raises(TypeError, match="payload must be a mapping"):
        stream.restore_rng_checkpoint(None, policy="checkpoint")
    _close(stream.get_extra_state(), before)
    assert stream._ignore_rng_on_load is True


@pytest.mark.parametrize("legacy", (False, True))
def test_authoritative_rng_restore_preserves_suffix_seed_and_reset(legacy):
    source = RNGModule(-43, (), ())
    source.randn((5,))
    payload = copy.deepcopy(source.rng_state() if legacy else source.get_extra_state())
    expected_suffix = (source.rand((7,)), source.randn((3,)))
    target = RNGModule(991, (), ())
    target.ignore_rng_on_load()
    target.restore_rng_checkpoint(payload, policy="checkpoint", allow_legacy=legacy)
    _close((target.rand((7,)), target.randn((3,))), expected_suffix)
    assert target._base_seed == (991 if legacy else -43)
    target.reset()
    reset = RNGModule(target._base_seed, (), ())
    _close(target.rand((9,)), reset.rand((9,)))
    assert target._ignore_rng_on_load is True


@pytest.mark.parametrize("kind", ("population", "handler", "network"))
@pytest.mark.parametrize("legacy", (False, True))
def test_native_runtime_checkpoint_restores_ignored_root_and_mechanism_streams(
    kind, legacy
):
    model = _network() if kind == "network" else _population()
    cell = model.left if kind == "network" else model
    if kind == "network":
        model.step()
    else:
        model.step(DT)
    owner = cell.mech if kind == "handler" else model
    checkpoint = copy.deepcopy(
        owner.mutable_state_dict()
        if kind == "handler"
        else owner.state_dict_for_checkpoint()
    )
    saved_rng = copy.deepcopy(cell.stream.get_extra_state())
    expected = copy.deepcopy(model)
    cell.stream.reseed(997)
    cell.stream.rand((13,))
    cell.stream.ignore_rng_on_load()
    if legacy:
        checkpoint = _legacy(checkpoint, kind)
    if kind == "handler":
        owner.restore_mutable_state_dict(checkpoint)
    else:
        owner.restore_dict_from_checkpoint(checkpoint)
    assert cell.stream is cell.mech.probe.stream
    if kind == "network":
        assert model.right.stream is cell.stream
        assert model.right.mech.probe.stream is cell.stream
    assert cell.stream._base_seed == (997 if legacy else saved_rng["base_seed"])
    _close(cell.stream.rng_state(), saved_rng["rng_state"])
    if kind == "network":
        model.step()
        expected.step()
        for name in model.populations:
            _close(model.populations[name].v, expected.populations[name].v)
    elif kind == "population":
        model.step(DT)
        expected.step(DT)
        _close(model.v, expected.v)
    assert cell.stream._ignore_rng_on_load is True


def test_authoritative_rng_restore_defers_unavailable_device_suffix(monkeypatch):
    real_generator = torch.Generator
    delayed = real_generator(device="cpu").manual_seed(314)
    delayed_state = delayed.get_state().clone()
    expected = torch.rand(6, generator=delayed)

    def unavailable(device="cpu"):
        if torch.device(device).type == "cuda":
            raise RuntimeError("CUDA unavailable in this test")
        return real_generator(device=device)

    monkeypatch.setattr(torch, "Generator", unavailable)
    source = RNGModule(71, (), ())
    source.rand((3,))
    payload = source.get_extra_state()
    payload["rng_state"]["cuda:7"] = delayed_state
    target = RNGModule(11, (), ())
    target.ignore_rng_on_load()
    target.restore_rng_checkpoint(payload)
    assert target._base_seed == 71
    _close(target._pending_device_states["cuda:7"], delayed_state)
    _close(target.get_extra_state(), payload)
    monkeypatch.setattr(
        torch, "Generator", lambda device="cpu": real_generator(device="cpu")
    )
    _close(torch.rand(6, generator=target._rng("cuda:7")), expected)
    assert "cuda:7" not in target._pending_device_states


def test_authoritative_rng_apply_failure_restores_original_seed_suffix_and_active_generator(
    monkeypatch,
):
    target = RNGModule(17, (), ())
    target.rand((3,))
    target.ignore_rng_on_load()
    original_active = target.rng
    original_cpu = target._cpu_gen
    before = copy.deepcopy(target.get_extra_state())
    replacement = RNGModule(901, (), ()).get_extra_state()
    apply = target._apply_validated_rng_state
    calls = 0

    def fail_once(state):
        nonlocal calls
        calls += 1
        if calls == 1:
            target._cpu_gen.manual_seed(888)
            target.rng = torch.Generator(device="cpu")
            target._device_gens[torch.device("cuda:7")] = target.rng
            raise RuntimeError("injected RNG restore failure")
        apply(state)

    monkeypatch.setattr(target, "_apply_validated_rng_state", fail_once)
    with pytest.raises(RuntimeError, match="injected RNG restore failure"):
        target.restore_rng_checkpoint(replacement)
    assert calls == 2
    assert target.rng is original_active
    assert target._cpu_gen is original_cpu
    assert target._device_gens == {}
    _close(target.get_extra_state(), before)
    assert target._ignore_rng_on_load is True


def test_network_late_invalid_checkpoint_rolls_back_shared_ignored_stream():
    model = _network()
    model.step()
    model.left.stream.ignore_rng_on_load()
    before = copy.deepcopy(model.state_dict_for_checkpoint())
    invalid = copy.deepcopy(before)
    replacement = RNGModule(719, (), ()).get_extra_state()
    invalid["populations"]["left"]["stochastic"]["rng_states"]["stream"] = replacement
    right = invalid["populations"]["right"]["stochastic"]["rng_states"]["stream"]
    right["rng_state"]["cpu"].zero_()
    with pytest.raises(RuntimeError, match="not a valid generator state"):
        model.restore_dict_from_checkpoint(invalid)
    _close(model.state_dict_for_checkpoint(), before)
    assert model.left.stream is model.right.stream
    assert model.left.stream._ignore_rng_on_load is True


def test_complete_rng_frame_evicts_absent_generators_and_recreates_from_seed(
    monkeypatch,
):
    real_generator = torch.Generator
    target = RNGModule(11, (), ())
    retained_device = torch.device("cpu:1")
    evicted_device = torch.device("cuda:7")
    retained = real_generator(device="cpu").manual_seed(231)
    evicted = real_generator(device="cpu").manual_seed(511)
    target._device_gens = {retained_device: retained, evicted_device: evicted}
    target.rng = evicted
    target._pending_device_states["cuda:8"] = evicted.get_state().clone()
    source = RNGModule(71, (), ())
    source.rand((3,))
    payload = source.get_extra_state()
    payload["rng_state"][str(retained_device)] = retained.get_state().clone()
    target.ignore_rng_on_load()
    target.restore_rng_checkpoint(payload)
    assert target._device_gens == {retained_device: retained}
    assert target.rng is None
    assert target._pending_device_states == {}
    _close(target.get_extra_state(), payload)

    # A CPU stand-in exercises the lifecycle without accelerator hardware.
    monkeypatch.setattr(
        torch, "Generator", lambda device="cpu": real_generator(device="cpu")
    )
    recreated = target._rng(evicted_device)
    assert recreated is not evicted
    expected = real_generator(device="cpu").manual_seed(
        _device_seed(71, evicted_device)
    )
    _close(torch.rand(7, generator=recreated), torch.rand(7, generator=expected))


@pytest.mark.parametrize("method", ("set_rng_state", "restore_rng_checkpoint"))
def test_rng_frame_eviction_rolls_back_active_generator_after_late_apply_failure(
    monkeypatch, method
):
    target = RNGModule(17, (), ())
    device = torch.device("cpu:1")
    retained = torch.Generator(device="cpu").manual_seed(119)
    target._device_gens[device] = retained
    target.rng = retained
    before = copy.deepcopy(target.get_extra_state())
    replacement = RNGModule(901, (), ()).get_extra_state()
    apply = target._apply_validated_rng_state
    calls = 0

    def fail_once(state):
        nonlocal calls
        calls += 1
        apply(state)
        if calls == 1:
            assert target._device_gens == {}
            assert target.rng is None
            raise RuntimeError("injected late RNG restore failure")

    monkeypatch.setattr(target, "_apply_validated_rng_state", fail_once)
    with pytest.raises(RuntimeError, match="injected late RNG restore failure"):
        if method == "set_rng_state":
            target.set_rng_state(replacement["rng_state"])
        else:
            target.restore_rng_checkpoint(replacement)
    assert calls == 2
    assert target._device_gens == {device: retained}
    assert target.rng is retained
    _close(target.get_extra_state(), before)


def _activate_accelerator_standin(stream, monkeypatch):
    device = torch.device("cuda:7")
    real_generator = torch.Generator
    generator = real_generator(device="cpu").manual_seed(381)
    stream._device_gens[device] = generator
    stream.rng = generator
    reference = real_generator(device="cpu")
    reference.set_state(generator.get_state())
    expected = torch.rand((7,), generator=reference, dtype=DTYPE)

    def unavailable(device="cpu"):
        if torch.device(device) == torch.device("cuda:7"):
            raise RuntimeError("CUDA device unavailable in this stand-in test")
        return real_generator(device=device)

    monkeypatch.setattr(torch, "Generator", unavailable)
    return device, generator, expected


def _evict_saved_accelerator_frames(checkpoint, *, legacy):
    payloads = [
        *checkpoint["stochastic"]["rng_states"].values(),
        *(
            payload
            for name, payload in checkpoint["mech"].items()
            if name.endswith("rng.stream")
        ),
    ]
    for payload in payloads:
        frame = payload if legacy else payload["rng_state"]
        frame.pop("cuda:7")
        if not legacy:
            payload["base_seed"] = 8101


@pytest.mark.parametrize("legacy", (False, True))
@pytest.mark.parametrize("ignore", (False, True))
def test_network_late_invalid_checkpoint_restores_evicted_generator_lifecycle(
    monkeypatch, legacy, ignore
):
    model = _network()
    model.step()
    stream = model.left.stream
    stream.ignore_rng_on_load(ignore)
    cpu_generator = stream._cpu_gen
    device, generator, expected = _activate_accelerator_standin(stream, monkeypatch)
    before = copy.deepcopy(model.state_dict_for_checkpoint())
    invalid = copy.deepcopy(before)
    if legacy:
        invalid = _legacy(invalid, "network")
    _evict_saved_accelerator_frames(invalid["populations"]["left"], legacy=legacy)
    right = invalid["populations"]["right"]["stochastic"]["rng_states"]["stream"]
    (right if legacy else right["rng_state"])["cpu"].zero_()
    with pytest.raises(RuntimeError, match="not a valid generator state"):
        model.restore_dict_from_checkpoint(invalid)
    _close(model.state_dict_for_checkpoint(), before)
    assert model.left.stream is model.right.stream
    assert model.left.mech.probe.stream is stream
    assert model.right.mech.probe.stream is stream
    assert stream._cpu_gen is cpu_generator
    assert stream._device_gens == {device: generator}
    assert stream.rng is generator
    assert stream._ignore_rng_on_load is ignore
    _close(stream.rand((7,), dtype=DTYPE), expected)


@pytest.mark.parametrize("legacy", (False, True))
def test_population_late_integrator_failure_restores_evicted_generator_lifecycle(
    monkeypatch, legacy
):
    population = _population()
    population.step(DT)
    stream = population.stream
    stream.ignore_rng_on_load()
    cpu_generator = stream._cpu_gen
    device, generator, expected = _activate_accelerator_standin(stream, monkeypatch)
    before = copy.deepcopy(population.state_dict_for_checkpoint())
    replacement = copy.deepcopy(before)
    if legacy:
        replacement = _legacy(replacement, "population")
    _evict_saved_accelerator_frames(replacement, legacy=legacy)
    restore_integrator = population.integrator.restore_mutable_state_dict
    calls = 0

    def fail_once(model, state):
        nonlocal calls
        calls += 1
        restore_integrator(model, state)
        if calls == 1:
            assert stream._device_gens == {}
            assert stream.rng is None
            raise RuntimeError("injected late integrator restore failure")

    monkeypatch.setattr(population.integrator, "restore_mutable_state_dict", fail_once)
    with pytest.raises(RuntimeError, match="injected late integrator restore failure"):
        population.restore_dict_from_checkpoint(replacement)
    assert calls == 2
    _close(population.state_dict_for_checkpoint(), before)
    assert population.stream is population.mech.probe.stream
    assert stream._cpu_gen is cpu_generator
    assert stream._device_gens == {device: generator}
    assert stream.rng is generator
    assert stream._ignore_rng_on_load is True
    _close(stream.rand((7,), dtype=DTYPE), expected)
