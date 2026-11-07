"""Zero-step samplers retain schema ownership and caller-specific boundaries."""

from types import SimpleNamespace

import pytest
import torch

from dendra.func import _network_rng, _rng_frames, _runtime_noise, _stimulation_rng
from dendra.func._types import RuntimeNoiseDraw, StimulationDraw
from dendra.models.rng import RNGModule

pytestmark = pytest.mark.cpu

_ENDPOINTS = ("population", "network", "joint")


class _Template:
    def __init__(self, tensor, events, label):
        self.tensor, self.events, self.label = tensor, events, label
        self.allocations = []

    @property
    def shape(self):
        return self.tensor.shape

    def new_empty(self, shape):
        self.events.append(self.label)
        result = self.tensor.new_empty(shape)
        self.allocations.append(result)
        return result


class _RngLookup(dict):
    def __init__(self, rng, events, failure):
        super().__init__(rng=rng)
        self.events, self.failure = events, failure

    def __getitem__(self, key):
        assert key == "rng"
        self.events.append("rng-source")
        if self.failure is not None:
            raise self.failure
        return super().__getitem__(key)


class _FinishedPacket:
    def __init__(self, rng, events, failure):
        self._lookup = _RngLookup(rng, events, failure)

    @property
    def rng(self):
        return self._lookup["rng"]


def _noise(entries):
    plan = object.__new__(_runtime_noise.RuntimeNoisePlan)
    plan.runtime = tuple(entries)
    plan.population = SimpleNamespace(device=lambda: torch.device("cpu"))
    return plan


def _entry(names, leaves):
    return SimpleNamespace(draw_names=tuple(names), leaves=tuple(leaves))


def _endpoint(kind, noise, rng, events, monkeypatch, *, failure=None):
    """Exercise real zero-step branches without source qualification or sampling."""
    module = {
        "population": _runtime_noise,
        "network": _network_rng,
        "joint": _stimulation_rng,
    }[kind]

    def packet(draws, frames):
        events.append("packet")
        return RuntimeNoiseDraw(draws, frames)

    monkeypatch.setattr(module, "RuntimeNoiseDraw", packet)
    if kind == "population":
        state = {"random": _RngLookup(rng, events, failure)}
        return lambda: noise.sample(state, steps=0)
    if kind == "network":
        plan = object.__new__(_network_rng.NetworkRNGPlan)
        plan.enabled = True
        plan.populations = {
            "cell": SimpleNamespace(_noise_plan=noise),
            "quiet": SimpleNamespace(_noise_plan=None),
        }

        def samplers(pool):
            assert pool is rng
            events.append("samplers")
            return {}

        def expand(name, state, pool):
            assert name == "cell" and pool is rng
            events.append("expand")
            return {"random": _RngLookup(rng, events, failure)}

        plan._samplers, plan.expand_population = samplers, expand
        state = {"rng": rng, "populations": {"cell": {}}}

        def invoke():
            packets = plan.sample_noise_tapes(state, 0)
            assert tuple(packets) == ("cell", "quiet")
            assert packets["quiet"] is None
            return packets["cell"]

        return invoke
    plan = object.__new__(_stimulation_rng.StimulationRNGPlan)
    plan.noise, plan.poisson = noise, None
    context = object()

    def begin(state, **kwargs):
        events.append("begin")
        return context

    def finish(actual):
        assert actual is context
        events.append("finish")
        return _FinishedPacket(rng, events, failure), None

    plan.begin_packets, plan.finish_packets = begin, finish

    def invoke():
        packet, stimulation = plan.sample_runtime({}, steps=0)
        assert stimulation is None
        return packet

    return invoke


def _prefix(kind):
    return {
        "population": ["rng-source"],
        "network": ["samplers", "expand"],
        "joint": ["begin", "finish"],
    }[kind]


@pytest.mark.parametrize("kind", _ENDPOINTS)
def test_empty_tapes_preserve_order_metadata_alias_allocations_and_sources(
    kind, monkeypatch
):
    events = []
    shared = _Template(
        torch.ones(2, 3, dtype=torch.float64, requires_grad=True), events, "shared"
    )
    meta = _Template(torch.empty(3, dtype=torch.int32, device="meta"), events, "meta")
    discarded = _Template(torch.tensor(True), events, "discarded")
    replacement = _Template(
        torch.ones(4, dtype=torch.float32, requires_grad=True), events, "replacement"
    )
    noise = _noise(
        (
            _entry(("first", "alias_a", "alias_b"), (shared, shared, shared)),
            _entry(("meta", "replace"), (meta, discarded)),
            _entry(("replace",), (replacement,)),
        )
    )
    generator = torch.Generator().manual_seed(107)
    frame = _Template(generator.get_state(), events, "rng")
    rng = {"first": {"cpu": frame}, "alias": {"cpu": frame}}
    templates = (shared, meta, discarded, replacement, frame)
    versions = tuple(value.tensor._version for value in templates)
    source_values = tuple(value.tensor.detach().clone() for value in templates)
    source_rng = generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    invoke = _endpoint(kind, noise, rng, events, monkeypatch)
    packets = (invoke(), invoke())
    expected_events = _prefix(kind) + [
        "shared",
        "shared",
        "shared",
        "meta",
        "discarded",
        "replacement",
    ]
    if kind != "population":
        expected_events.append("rng-source")
    expected_events += ["rng", "rng", "packet"]
    assert events == expected_events * 2
    for packet in packets:
        assert isinstance(packet, RuntimeNoiseDraw)
        assert tuple(packet.draws) == ("first", "alias_a", "alias_b", "meta", "replace")
        assert tuple(packet.rng) == ("first", "alias")
        assert packet.draws["first"].shape == (0, 2, 3)
        assert (
            packet.draws["alias_a"].shape == packet.draws["alias_b"].shape == (0, 2, 3)
        )
        assert packet.draws["first"].dtype == torch.float64
        assert packet.draws["meta"].shape == (0, 3)
        assert packet.draws["meta"].dtype == torch.int32
        assert packet.draws["meta"].device == torch.device("meta")
        assert packet.draws["replace"].shape == (0, 4)
        assert packet.draws["replace"].dtype == torch.float32
        assert packet.draws["replace"].device == torch.device("cpu")
        assert packet.rng["first"]["cpu"].shape == (0, *source_rng.shape)
        assert packet.rng["first"]["cpu"].dtype == torch.uint8
        assert all(not value.requires_grad for value in packet.draws.values())
    allocated = [value for template in templates for value in template.allocations]
    assert len(allocated) == 16  # Includes the overwritten duplicate-key allocation.
    storage_ids = {value.untyped_storage()._cdata for value in allocated}
    assert len(storage_ids) == len(allocated)
    assert tuple(value.tensor._version for value in templates) == versions
    for template, saved in zip(templates, source_values, strict=True):
        if saved.device.type != "meta":
            torch.testing.assert_close(template.tensor, saved, rtol=0, atol=0)
    assert shared.tensor.requires_grad and replacement.tensor.requires_grad
    assert shared.tensor.grad is None and replacement.tensor.grad is None
    torch.testing.assert_close(generator.get_state(), source_rng, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)


@pytest.mark.parametrize("kind", _ENDPOINTS)
@pytest.mark.parametrize("names, count", ((("first",), 2), (("first", "missing"), 1)))
def test_malformed_draw_schema_fails_before_later_rng_access_or_packet_construction(
    kind, names, count, monkeypatch
):
    events = []
    leaf = _Template(torch.tensor(1.0), events, "draw")
    noise = _noise((_entry(names, (leaf,) * count),))
    invoke = _endpoint(kind, noise, {}, events, monkeypatch)
    with pytest.raises(ValueError, match="zip\\(\\) argument"):
        invoke()
    assert events == _prefix(kind) + ["draw"]


@pytest.mark.parametrize("kind", _ENDPOINTS)
def test_rng_lookup_failure_preserves_each_consumers_draw_allocation_boundary(
    kind, monkeypatch
):
    events = []
    leaf = _Template(torch.tensor(1.0), events, "draw")
    failure = RuntimeError("unavailable caller RNG frame")
    invoke = _endpoint(
        kind,
        _noise((_entry(("draw",), (leaf,)),)),
        {},
        events,
        monkeypatch,
        failure=failure,
    )
    with pytest.raises(RuntimeError) as caught:
        invoke()
    assert caught.value is failure
    assert events == (
        ["rng-source"]
        if kind == "population"
        else _prefix(kind) + ["draw", "rng-source"]
    )


def test_explicit_sampler_frames_are_read_before_draw_allocation_without_state_lookup(
    monkeypatch,
):
    events = []
    leaf = _Template(torch.tensor(1.0, requires_grad=True), events, "draw")
    noise = _noise((_entry(("draw",), (leaf,)),))
    sampler = RNGModule(109, shape_p=(), shape_f=())
    before = sampler._cpu_gen.get_state().clone()
    original = _rng_frames._rng_values

    def frame(source, device):
        assert source is sampler
        events.append("sampler-frame")
        return original(source, device)

    monkeypatch.setattr(_rng_frames, "_rng_values", frame)
    packet = noise.sample({}, steps=0, samplers={"first": sampler, "alias": sampler})
    assert events == ["sampler-frame", "sampler-frame", "draw"]
    assert tuple(packet.rng) == ("first", "alias")
    assert all(
        value.shape[0] == 0 for frame in packet.rng.values() for value in frame.values()
    )
    torch.testing.assert_close(sampler._cpu_gen.get_state(), before, rtol=0, atol=0)


def test_joint_packets_finish_noise_before_allocating_poisson_draws(monkeypatch):
    events = []
    noise_draw = _Template(torch.tensor(1.0), events, "noise-draw")
    generator = torch.Generator().manual_seed(113)
    saved_rng, global_rng = generator.get_state().clone(), torch.get_rng_state().clone()
    noise_rng = {"stream": _Template(saved_rng, events, "noise-frame")}
    stimulation_rng = {"stream": _Template(saved_rng, events, "stimulation-frame")}
    plan = object.__new__(_stimulation_rng.StimulationRNGPlan)
    plan.noise = _noise((_entry(("noise",), (noise_draw,)),))
    entry = SimpleNamespace(key="poisson", stream="stream")

    def shape(actual):
        assert actual is entry
        events.append("poisson-shape")
        return (2, 3)

    plan.poisson = SimpleNamespace(
        entries=(entry,),
        _draw_shape=shape,
        draw_dtype=torch.float32,
        streams={"stream": generator},
    )
    context = object()

    def begin(state, **kwargs):
        events.append("begin")
        return context

    def finish(actual):
        assert actual is context
        events.append("finish")
        return (
            _FinishedPacket(noise_rng, events, None),
            _FinishedPacket(stimulation_rng, events, None),
        )

    def noise_packet(draws, rng):
        events.append("noise-packet")
        return RuntimeNoiseDraw(draws, rng)

    def stimulation_packet(draws, rng):
        events.append("stimulation-packet")
        return StimulationDraw(draws, rng)

    original_empty = torch.empty

    def empty(*args, **kwargs):
        events.append("poisson-allocation")
        return original_empty(*args, **kwargs)

    plan.begin_packets, plan.finish_packets = begin, finish
    monkeypatch.setattr(_stimulation_rng, "RuntimeNoiseDraw", noise_packet)
    monkeypatch.setattr(_stimulation_rng, "StimulationDraw", stimulation_packet)
    monkeypatch.setattr(torch, "empty", empty)
    noise, stimulation = plan.sample_runtime({}, steps=0)
    assert events == [
        "begin",
        "finish",
        "noise-draw",
        "rng-source",
        "noise-frame",
        "noise-packet",
        "poisson-shape",
        "poisson-allocation",
        "rng-source",
        "stimulation-frame",
        "stimulation-packet",
    ]
    assert isinstance(noise, RuntimeNoiseDraw)
    assert isinstance(stimulation, StimulationDraw)
    assert stimulation.draws["poisson"].shape == (0, 2, 3)
    assert stimulation.draws["poisson"].dtype == torch.float32
    torch.testing.assert_close(generator.get_state(), saved_rng, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
