"""Compatible native Populations retain cross-domain generator ownership."""

import pytest
import torch
from torch.utils import _pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.rng import RNGModule
from tests.func.test_poisson_stimulation import DTYPE, _clone, _close
from tests.func.test_stimulation_rng_sharing import (
    _draw_input,
    _functional_gradient_leaves,
    _lower,
    _population,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0,
                REQUIRE_GRAD=1,
                IMEM=False,
                DTYPE=DTYPE,
                DEVICE="cpu",
                CHECK_PURITY=False,
            ),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _case(borrowed):
    cell, _, pulse = _population()
    if not borrowed:
        independent = torch.Generator(device=pulse.generator.device)
        independent.set_state(pulse.generator.get_state())
        pulse.generator = independent
    return cell, pulse


def _unchanged_checkpoint(actual, expected):
    actual_leaves, actual_spec = _pytree.tree_flatten(actual)
    expected_leaves, expected_spec = _pytree.tree_flatten(expected)
    assert actual_spec == expected_spec
    for left, right in zip(actual_leaves, expected_leaves, strict=True):
        if torch.is_tensor(left):
            _close(left, right)
        else:
            assert left == right


@pytest.mark.parametrize("source_borrowed", (False, True))
@pytest.mark.parametrize("target_borrowed", (False, True))
def test_population_extraction_and_commit_preserve_cross_domain_rng_aliases(
    source_borrowed, target_borrowed
):
    source, _ = _case(source_borrowed)
    target, pulse = _case(target_borrowed)
    functional, tensors = _lower(source)
    before = _clone(target.state_dict_for_checkpoint())
    generator = pulse.generator.get_state().clone()
    if source_borrowed != target_borrowed:
        for operation in (
            lambda: functional.extract(target),
            lambda: functional.commit_state_(target, tensors.state),
        ):
            with pytest.raises(dn.func.FunctionalizationError, match="RNG sharing"):
                operation()
            _unchanged_checkpoint(target.state_dict_for_checkpoint(), before)
            _close(pulse.generator.get_state(), generator)
    else:
        functional.extract(target)
        functional.commit_state_(target, tensors.state)
        _close(functional.extract(target).state, tensors.state)


def test_source_validation_rejects_new_borrowing_of_a_captured_poisson_generator():
    source, pulse = _case(False)
    functional, tensors = _lower(source)
    assert not functional._stimulation_rng_plan.enabled
    source.mech.probe.stream._cpu_gen = pulse.generator
    before = pulse.generator.get_state().clone()
    with pytest.raises(dn.func.FunctionalizationError, match="RNG sharing"):
        functional.prepare(tensors.parameters, tensors.constants)
    _close(pulse.generator.get_state(), before)


def test_compatible_population_rejects_new_alias_to_an_untracked_registered_rng():
    source, _ = _case(False)
    target, pulse = _case(False)
    for cell in (source, target):
        cell.untracked = RNGModule(811, shape_p=(), shape_f=())
    functional, tensors = _lower(source)
    pulse.generator = target.untracked._generator_for("cpu")
    before = _clone(target.state_dict_for_checkpoint())
    generator = pulse.generator.get_state().clone()
    for operation in (
        lambda: functional.extract(target),
        lambda: functional.commit_state_(target, tensors.state),
    ):
        with pytest.raises(
            dn.func.FunctionalizationError, match="untracked registered"
        ):
            operation()
        _unchanged_checkpoint(target.state_dict_for_checkpoint(), before)
        _close(pulse.generator.get_state(), generator)


@pytest.mark.parametrize("canonical", (False, True))
def test_public_borrowed_step_preserves_fullgraph_and_functional_transforms(canonical):
    source, pulse = _case(True)
    functional, tensors = _lower(source)
    inputs = None if canonical else _draw_input(functional, tensors.state)
    options = (
        {"draws": functional.sample_runtime_draws(tensors.state)} if canonical else {}
    )
    before = _clone(tensors)
    generator = pulse.generator.get_state().clone()

    def step(parameters):
        return functional.prepare_and_step(
            parameters, tensors.constants, tensors.state, inputs, **options
        )

    expected = step(tensors.parameters)
    compiled = torch.compile(step, backend="aot_eager", fullgraph=True)
    _close(compiled(tensors.parameters), expected)

    def loss(parameters):
        return step(parameters)[0]["integrator"]["v"].square().sum()

    actual_gradients = torch.func.grad(loss)(tensors.parameters)
    expected_gradients = torch.autograd.grad(
        loss(tensors.parameters), _functional_gradient_leaves(tensors)
    )
    for actual, expected_gradient in zip(
        _functional_gradient_leaves(tensors._replace(parameters=actual_gradients)),
        expected_gradients,
        strict=True,
    ):
        _close(actual, expected_gradient)

    batched_parameters = _pytree.tree_map_only(
        torch.Tensor, lambda value: torch.stack((value, value)), tensors.parameters
    )
    expected_batch = _pytree.tree_map_only(
        torch.Tensor, lambda value: torch.stack((value, value)), expected
    )
    _close(torch.vmap(step)(batched_parameters), expected_batch)
    _close(tensors, before)
    _close(pulse.generator.get_state(), generator)
