"""Native analog receivers share pure reset and reduction algebra."""

from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import ContinuousSynapse
from dendra.models.mechanisms._continuous_receive import (
    reduce_continuous_values,
    reset_continuous_values,
)
from dendra.models.mod import graded_syn

pytestmark = pytest.mark.cpu
DTYPE = torch.float64


class _InheritedInput(graded_syn):
    pass


class _TwoInputs(ContinuousSynapse):
    ContinuousSynapse.INPUT("x")
    ContinuousSynapse.INPUT("y", keep_old=False)


def _synapse(kind=graded_syn):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Population(N=1, C=3, dtype=DTYPE)
        model.insert(kind.rename("analog"))
        model.initialize()
    return model.mech.analog


@pytest.mark.parametrize("kind", (graded_syn, _InheritedInput))
def test_inherited_and_renamed_inputs_are_unique_and_reset_old_value_once(kind):
    synapse = _synapse(kind)
    assert synapse._continuous_inputs == ("g_pre",)
    current = torch.tensor([[1.0, -2.0, 0.0]], dtype=DTYPE, requires_grad=True)
    synapse.g_pre = current
    synapse.reset_continuous_inputs()
    torch.testing.assert_close(synapse.g_pre_old, current)
    assert not torch.count_nonzero(synapse.g_pre)
    assert not synapse._continuous_received_masks["g_pre"].any()
    synapse.continuous_receive(torch.zeros_like(current))
    assert synapse._continuous_reset_count == 1
    torch.testing.assert_close(synapse.g_pre_old, current)
    torch.testing.assert_close(
        torch.autograd.grad(synapse.g_pre_old.sum(), current)[0],
        torch.ones_like(current),
    )


@pytest.mark.parametrize(
    "reduction", ("sum", "add", "set", "replace", "last", "min", "max")
)
def test_native_and_pure_reductions_preserve_real_zero_and_absent_slots(reduction):
    synapse = _synapse()
    initial = torch.tensor([[3.0, 2.0, 1.0]], dtype=DTYPE, requires_grad=True)
    synapse.g_pre = initial
    updates, scratch = reset_continuous_values(
        {"g_pre": initial}, ("g_pre",), {"g_pre": "g_pre_old"}
    )
    synapse.reset_continuous_inputs()
    current = updates["g_pre"]
    received = scratch["g_pre"]
    values = (
        torch.tensor([[0.0, -3.0, 99.0]], dtype=DTYPE, requires_grad=True),
        torch.tensor([[2.0, -1.0, -4.0]], dtype=DTYPE, requires_grad=True),
    )
    masks = (torch.tensor([[True, True, False]]), torch.tensor([[True, False, True]]))
    for value, mask in zip(values, masks):
        current = reduce_continuous_values(current, value, received, mask, reduction)
        received = received | mask
        synapse.continuous_receive(
            value, con=SimpleNamespace(reduce=reduction), mask=mask
        )
        torch.testing.assert_close(synapse.g_pre, current)
        torch.testing.assert_close(
            synapse._continuous_received_masks["g_pre"], received
        )
    if reduction == "min":
        assert current.tolist() == [[0.0, -3.0, -4.0]]
    elif reduction == "max":
        assert current.tolist() == [[2.0, -3.0, -4.0]]
    actual = torch.autograd.grad(
        synapse.g_pre.square().sum(), values, retain_graph=True
    )
    expected = torch.autograd.grad(current.square().sum(), values)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)
    assert not actual[0][0, 2]  # An absent scatter slot carries no gradient.


def test_multiple_input_reset_retains_only_declared_old_values():
    synapse = _synapse(_TwoInputs)
    synapse.x = torch.full_like(synapse.x, 4)
    synapse.y = torch.full_like(synapse.y, 7)
    synapse.reset_continuous_inputs()
    torch.testing.assert_close(synapse.x_old, torch.full_like(synapse.x_old, 4))
    assert not hasattr(synapse, "y_old")
    assert not torch.count_nonzero(synapse.x)
    assert not torch.count_nonzero(synapse.y)
    with pytest.raises(ValueError, match="requires `input=`"):
        synapse.continuous_receive(torch.ones_like(synapse.x))


def test_nonbroadcastable_native_mask_raises_contextual_value_error_without_mutation():
    synapse = _synapse()
    synapse.reset_continuous_inputs()
    synapse.continuous_receive(torch.ones_like(synapse.g_pre), reduce="max")
    current, previous = synapse.g_pre, synapse.g_pre_old
    received = synapse._continuous_received_masks["g_pre"]
    values = [value.clone() for value in (current, previous, received)]
    with pytest.raises(
        ValueError,
        match=r"mask with shape \(2, 2\).*input 'g_pre' with shape \(1, 3\)",
    ):
        synapse.continuous_receive(
            torch.full_like(current, 9), mask=torch.ones(2, 2, dtype=torch.bool)
        )
    assert synapse.g_pre is current
    assert synapse.g_pre_old is previous
    assert synapse._continuous_received_masks["g_pre"] is received
    for value, original in zip((current, previous, received), values):
        torch.testing.assert_close(value, original)


@pytest.mark.parametrize("reduction", ("sum", "set", "min", "max"))
@pytest.mark.parametrize("lanes", (0, 3))
def test_pure_reduction_maps_masks_and_values_including_empty(reduction, lanes):
    current = torch.tensor([2.0, -1.0], dtype=DTYPE)
    values = torch.full((lanes, 2), -3.0, dtype=DTYPE, requires_grad=True)
    masks = torch.ones(lanes, 2, dtype=torch.bool)
    received = torch.tensor([False, True])

    def evaluate(value, mask):
        return reduce_continuous_values(current, value, received, mask, reduction)

    result = torch.vmap(evaluate)(values, masks)
    expected = (
        torch.stack([evaluate(value, mask) for value, mask in zip(values, masks)])
        if lanes
        else values.new_empty(0, 2)
    )
    torch.testing.assert_close(result, expected)
    gradient = torch.autograd.grad(result.sum(), values)[0]
    assert gradient.shape == values.shape
