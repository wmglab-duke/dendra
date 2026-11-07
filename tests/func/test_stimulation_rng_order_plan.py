"""Mixed draw recipes follow observed native calls and own explicit suffixes."""

from types import MethodType

import pytest
import torch
from torch.utils import _pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._stimulation_rng import _assert_shared_frame
from dendra.models.rng import RNGModule
from tests.func.test_poisson_stimulation import DTYPE, _clone, _close
from tests.func.test_stimulation_rng_sharing import (
    _initialize_native,
    _lower,
    _population,
    _step_native,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def test_request_order_matches_independently_observed_native_random_boundaries(
    monkeypatch,
):
    source, extra, _pulse = _population()
    native, native_extra, native_pulse = _population()
    functional, tensors = _lower(source, extra)
    plan = functional._stimulation_rng_plan
    events = []
    owned_fn = "fn" in vars(native_pulse)
    original_poisson = native_pulse.fn
    original_normal = RNGModule.randn

    def poisson(self, times):
        del self
        events.append("poisson")
        return original_poisson(times)

    def normal(self, *args, **kwargs):
        events.append("normal")
        return original_normal(self, *args, **kwargs)

    with monkeypatch.context() as observed:
        observed.setattr(native_pulse, "fn", MethodType(poisson, native_pulse))
        observed.setattr(RNGModule, "randn", normal)
        _step_native(native, native_extra)
    # Older pytest restores an inherited method as an instance shadow; newer
    # pytest removes the temporary attribute itself during undo.
    if not owned_fn and "fn" in vars(native_pulse):
        delattr(native_pulse, "fn")
    requests = plan.runtime_requests()
    assert [
        "poisson" if kind == "poisson" else "normal" for kind, _ in requests
    ] == events
    assert any(kind == "sde" for kind, _entry in requests)
    assert events.index("poisson") < events.index("normal", 2) < len(events) - 1
    noise, stimulation = plan.sample_runtime(tensors.state)
    actual, _ = functional.prepare_and_step(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        dn.func.StepInput(noise=noise, stimulation=stimulation),
    )
    _close(actual, functional.extract(native).state)


def test_packet_builders_share_one_sampler_and_zero_tapes_keep_source_unchanged():
    source, extra, pulse = _population()
    functional, tensors = _lower(source, extra)
    plan = functional._stimulation_rng_plan
    before = _clone(tensors)
    source_rng = pulse.generator.get_state().clone()
    context = plan.begin_packets(tensors.state)
    for stream, (owner, device) in plan.borrowed.items():
        assert context["poisson_samplers"][stream] is context["random_samplers"][
            owner
        ]._generator_for(device)
    noise, stimulation = plan.sample_runtime(tensors.state, steps=0)
    for leaf in _pytree.tree_leaves((noise, stimulation)):
        assert leaf.shape[0] == 0
    _close(tensors, before)
    _close(pulse.generator.get_state(), source_rng)


def test_explicit_tapes_reject_an_earlier_conflicting_suffix_even_when_last_agrees():
    source, extra, pulse = _population()
    functional, tensors = _lower(source, extra)
    plan = functional._stimulation_rng_plan
    noise, stimulation = plan.sample_runtime(tensors.state, steps=2)
    changed = _clone(stimulation)
    stream = next(iter(plan.borrowed))
    changed.rng[stream][0] = torch.Generator().manual_seed(877).get_state()
    before, before_rng = _clone(tensors), pulse.generator.get_state().clone()
    with pytest.raises(ValueError, match="different end frames"):
        plan.validate_packets(noise, changed)
    plan.validate_packets(noise, stimulation)
    _close(tensors, before)
    _close(pulse.generator.get_state(), before_rng)


def test_shared_frame_assertion_is_retained_in_a_fullgraph_program():
    expected = torch.stack([torch.Generator().manual_seed(31).get_state()] * 2)

    def check(actual, reference):
        _assert_shared_frame(actual, reference)
        return reference.clone()

    compiled = torch.compile(check, backend="aot_eager", fullgraph=True)
    _close(compiled(expected, expected), expected)
    changed = expected.clone()
    changed[0] = torch.Generator().manual_seed(47).get_state()
    with pytest.raises(RuntimeError, match="different end frames"):
        compiled(changed, expected)


@pytest.mark.parametrize("reseed", (False, True))
def test_initialization_packet_suffix_follows_rand_noise_then_poisson(reseed):
    source, extra, pulse = _population(reseed=reseed)
    native, _, native_pulse = _population(reseed=reseed)
    functional, tensors = _lower(source, extra)
    plan = functional._stimulation_rng_plan
    source_rng = pulse.generator.get_state().clone()
    random, stimulation = plan.sample_initialization(tensors.state)
    _initialize_native(native)
    for stream, (owner, device) in plan.borrowed.items():
        _close(stimulation.rng[stream], random.rng[owner][device])
        _close(stimulation.rng[stream], native_pulse.generator.get_state())
    _close(pulse.generator.get_state(), source_rng)
