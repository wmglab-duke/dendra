"""Explicit RNG frames preserve native sampler ownership and continuation."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._network_rng import NetworkRNGPlan
from dendra.func._random_parameters import _rng_extra, _rng_values
from dendra.models.rng import RNGModule
from tests.func.test_network import _clone, _close
from tests.func.test_network_rng_sharing import _network
from tests.func.test_stimulation_rng_sharing import _lower, _population

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
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


@pytest.mark.parametrize("seed", (-(1 << 63), -1, 0, 1 << 63, (1 << 64) - 1))
def test_pooled_samplers_preserve_seed_frame_encoding_and_native_continuation(seed):
    source = RNGModule(seed, shape_p=(), shape_f=())
    source.randn((7,), dtype=torch.float64)
    checkpoint = source.get_extra_state()
    original = _clone(checkpoint)
    frame = _rng_values(source, torch.device("cpu"))
    _close(frame, _rng_values(checkpoint, torch.device("cpu")))
    assert tuple(frame) == ("seed", "negative", "cpu")
    assert frame["seed"].dtype == frame["cpu"].dtype == torch.uint8
    assert frame["seed"].shape == (8,)
    assert frame["seed"].tolist() == list((seed % (1 << 64)).to_bytes(8, "little"))
    assert frame["negative"].dtype == torch.bool
    assert frame["negative"].shape == () and bool(frame["negative"]) is (seed < 0)
    assert all(
        value.device.type == "cpu" and not value.requires_grad
        for value in frame.values()
    )
    assert not torch._C._is_alias_of(frame["cpu"], checkpoint["rng_state"]["cpu"])
    converted = _rng_extra(frame)
    assert converted["base_seed"] == seed
    assert not torch._C._is_alias_of(converted["rng_state"]["cpu"], frame["cpu"])
    frozen = _clone(frame)
    global_rng = torch.get_rng_state().clone()
    samplers = NetworkRNGPlan._samplers({"later": frame, "earlier": frame})
    assert tuple(samplers) == ("later", "earlier")
    assert samplers["later"] is not samplers["earlier"]
    assert samplers["later"]._cpu_gen is not samplers["earlier"]._cpu_gen
    assert samplers["later"]._cpu_gen is not source._cpu_gen
    oracle = RNGModule.from_checkpoint(checkpoint)
    expected = oracle.randn((5,), dtype=torch.float64)
    for sampler in samplers.values():
        assert sampler.get_extra_state()["base_seed"] == seed
        torch.testing.assert_close(sampler.randn((5,), dtype=torch.float64), expected)
        _close(sampler.get_extra_state(), oracle.get_extra_state())
    second = _clone(samplers["earlier"].get_extra_state())
    samplers["later"].randn((4,), dtype=torch.float64)
    _close(samplers["earlier"].get_extra_state(), second)
    _close(source.get_extra_state(), original)
    _close(frame, frozen)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)


def test_pooled_sampler_keeps_unavailable_device_states_deferred_and_owned(monkeypatch):
    source = RNGModule(-31, shape_p=(), shape_f=())
    checkpoint = source.get_extra_state()
    checkpoint["rng_state"]["cuda:7"] = torch.tensor([4, 9, 2], dtype=torch.uint8)
    frame = _rng_values(checkpoint, torch.device("cpu"))
    frozen = _clone(frame)
    original = torch.Generator

    def available_cpu_only(device="cpu", **kwargs):
        if torch.device(device).type != "cpu":
            raise RuntimeError("test device is unavailable")
        return original(device=device, **kwargs)

    monkeypatch.setattr(torch, "Generator", available_cpu_only)
    sampler = NetworkRNGPlan._samplers({"stream": frame})["stream"]
    restored = sampler.get_extra_state()
    assert tuple(frame) == ("seed", "negative", "cpu", "cuda:7")
    assert tuple(restored["rng_state"]) == ("cpu", "cuda:7")
    assert "cuda:7" in sampler._pending_device_states
    assert not sampler._device_gens
    torch.testing.assert_close(restored["rng_state"]["cuda:7"], frame["cuda:7"])
    assert not torch._C._is_alias_of(restored["rng_state"]["cuda:7"], frame["cuda:7"])
    sampler.randn((3,), dtype=torch.float64)
    _close(frame, frozen)


class _ObservedFrame(dict):
    def __init__(self, values, events, label):
        super().__init__(values)
        self.events, self.label = events, label

    def __getitem__(self, key):
        self.events.append((self.label, key))
        return super().__getitem__(key)

    def items(self):
        self.events.append((self.label, "items"))
        return super().items()


@pytest.mark.parametrize("invalid", ("seed", "negative", "cpu"))
def test_pool_restoration_keeps_frame_lookup_and_native_validation_error_order(
    invalid, monkeypatch
):
    frame = _rng_values(RNGModule(71, shape_p=(), shape_f=()), torch.device("cpu"))
    bad = _clone(frame)
    if invalid == "cpu":
        bad["cpu"] = torch.zeros(3, dtype=torch.uint8)
    else:
        bad.pop(invalid)
    events = []
    original = RNGModule.from_checkpoint.__func__

    def observed(cls, payload, **kwargs):
        events.append(("restore", payload["base_seed"]))
        return original(cls, payload, **kwargs)

    monkeypatch.setattr(RNGModule, "from_checkpoint", classmethod(observed))
    pool = {
        "first": _ObservedFrame(frame, events, "first"),
        "bad": _ObservedFrame(bad, events, "bad"),
        "unvisited": _ObservedFrame(frame, events, "unvisited"),
    }
    with pytest.raises(RuntimeError if invalid == "cpu" else KeyError) as caught:
        NetworkRNGPlan._samplers(pool)
    if invalid == "cpu":
        assert (
            str(caught.value)
            == "RNG state for device 'cpu' is not a valid generator state."
        )
    else:
        assert caught.value.args == (invalid,)
    expected = [
        ("first", "seed"),
        ("first", "negative"),
        ("first", "items"),
        ("restore", 71),
        ("bad", "seed"),
    ]
    if invalid != "seed":
        expected.append(("bad", "negative"))
    if invalid == "cpu":
        expected.extend((("bad", "items"), ("restore", 71)))
    assert events == expected


def test_network_pool_aliases_reuse_samplers_without_changing_frames_or_source():
    source = _network()
    functional, tensors = dn.func.make_functional_network(source)
    plan = functional._rng_plan
    state_before = _clone(tensors.state)
    source_before = _clone(source.state_dict_for_checkpoint())
    pool = plan._samplers(tensors.state["rng"])
    locals_by_population = {
        name: plan._local_samplers(name, pool) for name in functional.populations
    }
    for name, bindings in plan.local_streams.items():
        for local, canonical in bindings.items():
            assert locals_by_population[name][local] is pool[canonical]
    assert len(pool) == 2
    assert (
        len(
            {
                id(sampler)
                for local in locals_by_population.values()
                for sampler in local.values()
            }
        )
        == 2
    )
    next(iter(locals_by_population["left"].values())).randn((3,), dtype=torch.float64)
    _close(tensors.state, state_before)
    _close(source.state_dict_for_checkpoint(), source_before)


@pytest.mark.parametrize("supplied", (False, True))
def test_joint_stimulation_keeps_supplied_sampler_identity_and_frame_lookup_policy(
    supplied,
):
    source, extra, _pulse = _population()
    functional, tensors = _lower(source, extra)
    plan = functional._stimulation_rng_plan
    assert plan.enabled and plan.borrowed
    samplers = NetworkRNGPlan._samplers(tensors.state["random"]["rng"])
    poisson = {name: torch.Generator().manual_seed(29) for name in plan.poisson.streams}
    before = _clone(tensors.state)
    state = _clone(tensors.state)
    first_stream = next(iter(state["random"]["rng"]))
    state["random"]["rng"][first_stream].pop("seed")
    if not supplied:
        with pytest.raises(KeyError) as caught:
            plan.begin_packets(state, poisson_samplers=poisson)
        assert caught.value.args == ("seed",)
    else:
        context = plan.begin_packets(state, samplers=samplers, poisson_samplers=poisson)
        assert context["random_samplers"] is samplers
        for stream, (owner, device) in plan.borrowed.items():
            assert context["poisson_samplers"][stream] is samplers[
                owner
            ]._generator_for(device)
            assert context["poisson_samplers"][stream] is not poisson[stream]
    _close(functional.extract().state, before)


def test_zero_step_network_tapes_keep_empty_frame_schema_and_source_suffix():
    source = _network()
    functional, tensors = dn.func.make_functional_network(source)
    frozen = _clone(tensors.state)
    packets = functional.sample_population_noise_tapes(tensors.state, 0)
    for name, packet in packets.items():
        assert packet is not None
        for local, canonical in functional._rng_plan.local_streams[name].items():
            frame = tensors.state["rng"][canonical]
            assert tuple(packet.rng[local]) == tuple(frame)
            for key, value in packet.rng[local].items():
                assert value.shape == (0, *frame[key].shape)
                assert (
                    value.dtype == frame[key].dtype
                    and value.device == frame[key].device
                )
    _close(tensors.state, frozen)
    _close(functional.extract().state, frozen)


def test_provided_population_frame_fails_before_later_population_sampling(monkeypatch):
    source = _network()
    functional, tensors = dn.func.make_functional_network(source)
    packet = functional.sample_population_noise_tapes(tensors.state, 1)["left"]
    local = next(iter(packet.rng))
    broken = packet._replace(
        rng={
            **packet.rng,
            local: {
                key: value
                for key, value in packet.rng[local].items()
                if key != "negative"
            },
        }
    )
    frozen = _clone(tensors.state)

    def unexpected(*args, **kwargs):
        raise AssertionError(
            "later Population sampled before provided frame restoration"
        )

    monkeypatch.setattr(
        functional.populations["right"]._noise_plan, "sample", unexpected
    )
    with pytest.raises(KeyError) as caught:
        functional._rng_plan.sample_noise_tapes(
            tensors.state, 1, provided={"left": broken}
        )
    assert caught.value.args == ("negative",)
    _close(tensors.state, frozen)
    _close(functional.extract().state, frozen)
