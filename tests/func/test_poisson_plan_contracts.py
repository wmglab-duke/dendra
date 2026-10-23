"""Poisson draw contracts retain invocation paths and physical stream ownership."""

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.func._poisson_stimulation import randomized_nodes
from dendra.models.stim.waveform._poisson import poisson_generator_state
from dendra.units import nA
from tests.func.test_poisson_stimulation import (
    DT,
    DTYPE,
    _network,
    _population,
    _waveform,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    state = torch.get_rng_state().clone()
    dtype = torch.get_default_dtype()
    try:
        with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=DTYPE, IMEM=False):
            yield
    finally:
        torch.set_default_dtype(dtype)
        torch.set_rng_state(state)


def _clone(value):
    return pytree.tree_map_only(
        torch.Tensor, lambda tensor: tensor.detach().clone(), value
    )


def _equal(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _wrapped(kind):
    waveform = _waveform(amp=0.2 * nA)
    if kind == "sum":
        return waveform + dn.constant(value=0.0)
    if kind == "product":
        return waveform * dn.constant(value=2.0)
    return waveform.repeat(freq=0.4, delay=0.0)


@pytest.mark.parametrize("wrapper", ("sum", "product", "repeat"))
def test_wrapped_intracellular_node_extract_commit_and_checkpoint_resolve_the_child(
    wrapper,
):
    source, _, _ = _population(waveform=_wrapped(wrapper))
    target, target_waveform, _ = _population(waveform=_wrapped(wrapper))
    functional, tensors = dn.func.make_functional(source, dt=DT)
    plan = functional._poisson_plan
    draw = plan.sample(tensors.state)
    stimulation = plan.advance(tensors.state["stimulation"], draw)
    state = {**tensors.state, "stimulation": stimulation}
    functional.commit_state_(target, state)
    node = randomized_nodes(target_waveform)[0][1]
    _equal(poisson_generator_state(node), draw.rng[plan.entries[0].stream])
    assert not hasattr(target_waveform, "_spike_times")
    _equal(functional.extract(target).state, state)
    saved = _clone(target.state_dict_for_checkpoint())
    target.step(DT)
    target.restore_dict_from_checkpoint(saved)
    _equal(functional.extract(target).state, state)


def test_source_generator_rebinding_is_rejected_without_consuming_rng():
    source, waveform, _ = _population()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    previous_generator = waveform.generator
    previous_state = previous_generator.get_state().clone()
    waveform.generator = torch.Generator().manual_seed(77)
    current_state = waveform.generator.get_state().clone()
    with pytest.raises(dn.func.FunctionalizationError, match="generator bindings"):
        functional._poisson_plan.sample(tensors.state)
    _equal(previous_generator.get_state(), previous_state)
    _equal(waveform.generator.get_state(), current_state)


def test_registered_private_rng_cannot_alias_poisson_carry():
    from dendra.models.rng import RNGModule

    source, waveform, _ = _population()
    source.poisson_rng = RNGModule(seed=71, shape_p=(), shape_f=())
    waveform.generator = source.poisson_rng._cpu_gen
    before = waveform.generator.get_state().clone()
    with pytest.raises(dn.func.FunctionalizationError, match="registered RNGModule"):
        dn.func.make_functional(source, dt=DT)
    _equal(waveform.generator.get_state(), before)


def test_network_private_rng_cannot_alias_population_poisson_carry():
    source = _network()
    source.populations["left"].mech.owner.injected_waveforms[
        0
    ].generator = source._cpu_gen
    before = source._cpu_gen.get_state().clone()
    with pytest.raises(dn.func.FunctionalizationError, match="private RNG"):
        dn.func.make_functional_network(source)
    _equal(source._cpu_gen.get_state(), before)


def test_compatible_network_rejects_changed_cross_population_generator_aliases():
    source = _network(sharing="generator")
    functional, _ = dn.func.make_functional_network(source)
    with pytest.raises(dn.func.FunctionalizationError, match="sharing"):
        functional._stimulation_plan.validate_compatible_network(
            _network(sharing="independent")
        )


def test_shared_generator_requires_all_explicit_consumers_and_keeps_tapes_exact():
    source = _network(sharing="generator")
    functional, tensors = dn.func.make_functional_network(source)
    before = _clone(tensors.state)
    tapes = functional.sample_population_stimulation_tapes(tensors.state, 3)
    with pytest.raises(ValueError, match="every consuming Population"):
        functional.sample_population_stimulation_tapes(
            tensors.state, 3, {"left": tapes["left"]}
        )
    result = functional.sample_population_stimulation_tapes(tensors.state, 3, tapes)
    assert result["left"] is tapes["left"]
    assert result["right"] is tapes["right"]
    _equal(tensors.state, before)


def test_shared_explicit_tapes_reject_inconsistent_valid_rng_end_frames():
    source = _network(sharing="generator")
    functional, tensors = dn.func.make_functional_network(source)
    tapes = functional.sample_population_stimulation_tapes(tensors.state, 2)
    right = _clone(tapes["right"])
    key = next(iter(right.rng))
    right.rng[key][0] = torch.Generator().manual_seed(888).get_state()
    with pytest.raises(ValueError, match="disagree"):
        functional.sample_population_stimulation_tapes(
            tensors.state, 2, {"left": tapes["left"], "right": right}
        )


def test_independent_generators_allow_one_explicit_population_tape():
    source = _network(sharing="independent")
    functional, tensors = dn.func.make_functional_network(source)
    tapes = functional.sample_population_stimulation_tapes(tensors.state, 2)
    mixed = functional.sample_population_stimulation_tapes(
        tensors.state, 2, {"left": tapes["left"]}
    )
    assert mixed["left"] is tapes["left"]
    _equal(mixed["right"], tapes["right"])


@pytest.mark.parametrize(
    "fault",
    ("nan", "negative", "one", "shape", "dtype", "missing", "rng-shape", "rng-bytes"),
)
def test_malformed_step_packets_fail_without_changing_source_or_input_carry(fault):
    source, waveform, _ = _population()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    packet = _clone(functional.sample_stimulation_draw(tensors.state))
    key = next(iter(packet.draws))
    rng_key = next(iter(packet.rng))
    if fault in {"nan", "negative", "one"}:
        packet.draws[key][0] = {"nan": torch.nan, "negative": -0.1, "one": 1.0}[fault]
    elif fault == "shape":
        packet.draws[key] = packet.draws[key][:-1]
    elif fault == "dtype":
        packet.draws[key] = packet.draws[key].to(
            torch.float32 if packet.draws[key].dtype == torch.float64 else torch.float64
        )
    elif fault == "missing":
        del packet.draws[key]
    elif fault == "rng-shape":
        packet.rng[rng_key] = packet.rng[rng_key][:-1]
    else:
        packet.rng[rng_key] = torch.zeros_like(packet.rng[rng_key])
    before = _clone(tensors.state)
    generator_state = waveform.generator.get_state().clone()
    with pytest.raises(ValueError):
        functional.step(
            tensors.parameters,
            prepared,
            tensors.state,
            dn.func.StepInput(stimulation=packet),
        )
    _equal(tensors.state, before)
    _equal(waveform.generator.get_state(), generator_state)


@pytest.mark.parametrize("draw_dtype", (torch.float32, torch.float64))
def test_finite_off_dynamic_schedule_extracts_to_fixed_shape_and_checkpoints_in_model_dtype(
    draw_dtype,
):
    torch.set_default_dtype(draw_dtype)
    waveform = dn.mono_rect(amp=0.2 * nA, pw=0.03).poisson(
        interval=0.1,
        n=17,
        start=-0.02,
        noise=0.9,
        off=0.08,
        randomize_every_call=True,
        generator=torch.Generator().manual_seed(43),
    )
    source, waveform, _ = _population(waveform=waveform)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    for _ in range(4):
        source.step(DT)
        extracted = functional.extract(source)
        schedule = next(iter(extracted.state["stimulation"]["schedules"].values()))
        assert schedule.shape == (17,)
        assert schedule.dtype == DTYPE
        _equal(
            schedule[: waveform._spike_times.numel()], waveform._spike_times.reshape(-1)
        )
        saved = _clone(source.state_dict_for_checkpoint())
        expected = waveform._spike_times.clone()
        torch.set_default_dtype(
            torch.float64 if draw_dtype == torch.float32 else torch.float32
        )
        source.step(DT)
        source.restore_dict_from_checkpoint(saved)
        _equal(waveform._spike_times, expected)
        assert waveform._spike_times.dtype == DTYPE
        torch.set_default_dtype(draw_dtype)
    assert tensors.state["stimulation"]["rng"]


def test_latest_schedule_is_canonicalized_across_shared_waveform_population_aliases():
    source = _network(sharing="waveform")
    functional, tensors = dn.func.make_functional_network(source)
    tapes = functional.sample_population_stimulation_tapes(tensors.state, 1)
    states = {}
    for name, plan in functional._stimulation_plan.plans.items():
        state = functional._expand_population_state(
            name, tensors.state["populations"][name], tensors.state
        )
        packet = pytree.tree_map_only(torch.Tensor, lambda value: value[0], tapes[name])
        states[name] = {
            **state,
            "stimulation": plan.advance(state["stimulation"], packet),
        }
    latest = next(iter(states["right"]["stimulation"]["schedules"].values()))
    canonical = functional._stimulation_plan.canonicalize(states)
    for state in canonical.values():
        for schedule in state["stimulation"]["schedules"].values():
            _equal(schedule, latest)


def test_repeated_intracellular_registration_retains_distinct_calls_and_shared_parameters():
    def build():
        waveform = _waveform(amp=0.2 * nA)
        population, _, _ = _population(waveform=waveform)
        population[:, 2].inject(waveform)
        population.initialize()
        population.integrator._initialize(
            population, population.t.new_tensor(DT), force=True
        )
        population.detach()
        return population

    source, native = build(), build()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    assert len(functional._poisson_plan.entries) == 2
    assert len({id(entry.source) for entry in functional._poisson_plan.entries}) == 1
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    state = tensors.state
    for _ in range(3):
        draw = functional.sample_stimulation_draw(state)
        state, _ = functional.step(
            tensors.parameters, prepared, state, dn.func.StepInput(stimulation=draw)
        )
        native.step(DT)
        torch.testing.assert_close(
            state["integrator"]["v"], native.v, rtol=2e-10, atol=2e-11
        )
    _equal(state, functional.extract(native).state)


def test_network_commit_rejects_conflicting_shared_latest_schedules_atomically():
    source = _network(sharing="waveform")
    functional, tensors = dn.func.make_functional_network(source)
    state = _clone(tensors.state)
    next(iter(state["populations"]["left"]["stimulation"]["schedules"].values()))[
        0
    ] -= 0.001
    target = _network(sharing="waveform")
    before = _clone(functional.extract(target).state)
    with pytest.raises(ValueError, match="latest schedules across Populations"):
        functional.commit_state_(target, state)
    _equal(functional.extract(target).state, before)
