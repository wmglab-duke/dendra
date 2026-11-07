"""Network host endpoints preserve borrowed and isolated Poisson samplers."""

from types import SimpleNamespace

import pytest
import torch

from dendra.func._network_rng import NetworkStimulationRNGPlan

pytestmark = pytest.mark.cpu

_ENDPOINTS = ("initialization", "runtime")


def _endpoint(kind, streams, borrowed, model_samplers, pool, events):
    """Keep packet consumers observable without executing a model transition."""
    plan = object.__new__(NetworkStimulationRNGPlan)
    plan.enabled = True
    plan.borrowed = borrowed
    model_rng, population_state = {}, {}
    captured = []
    supplied_noise, supplied_stimulation = object(), object()
    noise_result, stimulation_result = object(), object()

    def models(actual):
        assert actual is model_rng
        events.append("models")
        return model_samplers

    def local_models(name, actual):
        assert name == "cell" and actual is model_samplers
        events.append("local-models")
        return model_samplers

    def expand_pool(actual_random, actual_stimulation):
        assert actual_random is model_rng and actual_stimulation is pool
        events.append("expand-pool")
        return pool

    def expand_population(name, state, actual_random, actual_stimulation):
        assert name == "cell" and state is population_state
        assert actual_random is model_rng and actual_stimulation is pool
        events.append("expand-population")
        return population_state

    def provided(noise, stimulation, steps):
        assert noise is supplied_noise and stimulation is supplied_stimulation
        assert steps == 0
        events.append("provided")
        return {}, {}

    def consume(state, *, samplers, poisson_samplers, **kwargs):
        assert state is population_state and samplers is model_samplers
        assert kwargs == ({} if kind == "initialization" else {"steps": 0})
        events.append("consume")
        captured.append(poisson_samplers)
        return noise_result, stimulation_result

    def validate(noise, stimulation):
        assert noise == {"cell": noise_result}
        assert stimulation == {"cell": stimulation_result}
        events.append("validate")

    plan.random = SimpleNamespace(_samplers=models, _local_samplers=local_models)
    plan.stimulation = SimpleNamespace(
        streams=streams, local={"cell": {key: key for key in streams}}
    )
    plan.functional = SimpleNamespace(
        populations={
            "cell": SimpleNamespace(
                _stimulation_rng_plan=SimpleNamespace(
                    sample_initialization=consume, sample_runtime=consume
                )
            )
        }
    )
    plan.expand_pool, plan.expand_population = expand_pool, expand_population
    plan._provided, plan.validate_packets = provided, validate

    def invoke():
        if kind == "initialization":
            result = plan.sample_population_initialization(
                "cell", population_state, model_rng, pool
            )
            assert result == (noise_result, stimulation_result)
        else:
            state = {
                "rng": model_rng,
                "stimulation_rng": pool,
                "populations": {"cell": population_state},
            }
            result = plan.sample_tapes(
                state, 0, noise=supplied_noise, stimulation=supplied_stimulation
            )
            assert result == ({"cell": noise_result}, {"cell": stimulation_result})

    return SimpleNamespace(
        plan=plan,
        invoke=invoke,
        captured=captured,
        model_rng=model_rng,
        population_state=population_state,
    )


def _prefix(kind):
    return (["provided"] if kind == "runtime" else []) + ["models", "expand-pool"]


@pytest.mark.parametrize("kind", _ENDPOINTS)
def test_mixed_streams_preserve_borrowed_aliases_and_fresh_owned_continuation(
    kind, monkeypatch
):
    events = []
    owned_a = torch.Generator().manual_seed(131)
    owned_b = torch.Generator().manual_seed(137)
    borrowed = torch.Generator().manual_seed(139)
    torch.rand(5, generator=owned_a)
    torch.rand(3, generator=borrowed)
    saved_owned, saved_other = owned_a.get_state().clone(), owned_b.get_state().clone()
    saved_borrowed = borrowed.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    model = SimpleNamespace()

    def generator_for(device):
        assert device == "cpu"
        events.append("borrow")
        return borrowed

    model._generator_for = generator_for
    endpoint = _endpoint(
        kind,
        {
            "owned-a": owned_a,
            "borrowed-a": borrowed,
            "owned-b": owned_b,
            "borrowed-b": borrowed,
        },
        {"borrowed-a": ("owner", "cpu"), "borrowed-b": ("owner", "cpu")},
        {"owner": model},
        {"owned-a": saved_owned, "owned-b": saved_owned},
        events,
    )
    endpoint.plan.stimulation.local["cell"]["owned-alias"] = "owned-a"
    real_generator = torch.Generator

    def construct(*, device):
        assert device == torch.device("cpu")
        events.append("construct")
        return real_generator(device=device)

    monkeypatch.setattr(torch, "Generator", construct)
    endpoint.invoke()
    endpoint.invoke()
    expected = _prefix(kind) + [
        "construct",
        "borrow",
        "construct",
        "borrow",
        "expand-population",
        "local-models",
        "consume",
    ]
    if kind == "runtime":
        expected.append("validate")
    assert events == expected * 2
    first, second = endpoint.captured
    assert (
        tuple(first)
        == tuple(second)
        == ("owned-a", "borrowed-a", "owned-b", "borrowed-b", "owned-alias")
    )
    assert first["owned-alias"] is first["owned-a"]
    assert second["owned-alias"] is second["owned-a"]
    assert first["borrowed-a"] is first["borrowed-b"] is borrowed
    assert second["borrowed-a"] is second["borrowed-b"] is borrowed
    owned = [
        packet[key] for packet in (first, second) for key in ("owned-a", "owned-b")
    ]
    assert len({id(generator) for generator in owned}) == 4
    assert all(
        generator is not owned_a and generator is not owned_b for generator in owned
    )
    for generator in owned:
        torch.testing.assert_close(generator.get_state(), saved_owned, rtol=0, atol=0)
    torch.testing.assert_close(owned_a.get_state(), saved_owned, rtol=0, atol=0)
    torch.testing.assert_close(owned_b.get_state(), saved_other, rtol=0, atol=0)
    torch.testing.assert_close(borrowed.get_state(), saved_borrowed, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
    oracle = real_generator().set_state(saved_owned)
    expected_draw = torch.rand(7, generator=oracle)
    for generator in owned:
        torch.testing.assert_close(
            torch.rand(7, generator=generator), expected_draw, rtol=0, atol=0
        )
    torch.testing.assert_close(owned_a.get_state(), saved_owned, rtol=0, atol=0)
    torch.testing.assert_close(owned_b.get_state(), saved_other, rtol=0, atol=0)
    torch.testing.assert_close(borrowed.get_state(), saved_borrowed, rtol=0, atol=0)


@pytest.mark.parametrize("kind", _ENDPOINTS)
@pytest.mark.parametrize(
    "failure_phase", ("device", "construct", "frame", "detach", "cpu", "set-state")
)
def test_owned_restoration_preserves_construction_lookup_and_conversion_error_order(
    kind, failure_phase, monkeypatch
):
    events = []
    failure = RuntimeError(f"{failure_phase} failure")

    def observe(phase):
        events.append(phase)
        if phase == failure_phase:
            raise failure

    class Source:
        @property
        def device(self):
            observe("device")
            return torch.device("cpu")

    class Frame:
        def detach(self):
            observe("detach")
            return self

        def cpu(self):
            observe("cpu")
            return self

    class Pool(dict):
        def __getitem__(self, key):
            assert key == "owned"
            observe("frame")
            return Frame()

    class Generator:
        def set_state(self, frame):
            assert isinstance(frame, Frame)
            observe("set-state")

    def construct(*, device):
        assert device == torch.device("cpu")
        observe("construct")
        return Generator()

    endpoint = _endpoint(kind, {"owned": Source()}, {}, {}, Pool(), events)
    monkeypatch.setattr(torch, "Generator", construct)
    with pytest.raises(RuntimeError) as caught:
        endpoint.invoke()
    assert caught.value is failure
    phases = ["device", "construct", "frame", "detach", "cpu", "set-state"]
    assert events == _prefix(kind) + phases[: phases.index(failure_phase) + 1]
    assert endpoint.captured == []


@pytest.mark.parametrize(
    "kind, phase",
    (
        ("runtime", "provided"),
        ("runtime", "models"),
        ("runtime", "expand-pool"),
        ("initialization", "models"),
        ("initialization", "expand-pool"),
    ),
)
def test_caller_failures_precede_any_poisson_generator_construction(
    kind, phase, monkeypatch
):
    events = []
    source = torch.Generator().manual_seed(151)
    endpoint = _endpoint(
        kind, {"owned": source}, {}, {}, {"owned": source.get_state()}, events
    )
    failure = RuntimeError(f"{phase} failure")

    def fail(*args, **kwargs):
        events.append(phase)
        raise failure

    def unexpected(*args, **kwargs):
        pytest.fail("Poisson generator constructed after an earlier caller failure")

    if phase == "provided":
        endpoint.plan._provided = fail
    elif phase == "models":
        endpoint.plan.random._samplers = fail
    else:
        endpoint.plan.expand_pool = fail
    monkeypatch.setattr(torch, "Generator", unexpected)
    with pytest.raises(RuntimeError) as caught:
        endpoint.invoke()
    assert caught.value is failure
    prefix = _prefix(kind)
    assert events == prefix[: prefix.index(phase) + 1]


@pytest.mark.parametrize("kind", _ENDPOINTS)
def test_borrowed_frames_are_eagerly_expanded_before_owned_generator_construction(
    kind, monkeypatch
):
    events = []
    endpoint = _endpoint(
        kind,
        {"owned-first": SimpleNamespace(device="cpu"), "borrowed": object()},
        {"borrowed": ("owner", "cpu")},
        {},
        {},
        events,
    )

    class MissingDevice(dict):
        def __getitem__(self, key):
            events.append("borrowed-frame")
            raise KeyError(key)

    endpoint.model_rng["owner"] = MissingDevice()
    original_expand = NetworkStimulationRNGPlan.expand_pool

    def expand(random, stimulation):
        events.append("expand-pool")
        return original_expand(endpoint.plan, random, stimulation)

    def unexpected(*args, **kwargs):
        pytest.fail("Owned generator constructed before borrowed frame expansion")

    endpoint.plan.expand_pool = expand
    monkeypatch.setattr(torch, "Generator", unexpected)
    with pytest.raises(KeyError) as caught:
        endpoint.invoke()
    assert caught.value.args == ("cpu",)
    assert events == _prefix(kind) + ["borrowed-frame"]
    assert endpoint.captured == []


@pytest.mark.parametrize("kind", _ENDPOINTS)
def test_invalid_owned_bytes_keep_native_restoration_error_and_sources_unchanged(kind):
    events = []
    source = torch.Generator().manual_seed(149)
    before, global_rng = source.get_state().clone(), torch.get_rng_state().clone()
    invalid = torch.zeros(3, dtype=torch.uint8)
    endpoint = _endpoint(kind, {"owned": source}, {}, {}, {"owned": invalid}, events)
    with pytest.raises(RuntimeError) as reference:
        torch.Generator().set_state(invalid)
    with pytest.raises(RuntimeError) as caught:
        endpoint.invoke()
    assert str(caught.value) == str(reference.value)
    assert events == _prefix(kind)
    assert endpoint.captured == []
    torch.testing.assert_close(invalid, torch.zeros(3, dtype=torch.uint8))
    torch.testing.assert_close(source.get_state(), before, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
