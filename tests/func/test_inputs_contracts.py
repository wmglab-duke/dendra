"""Deterministic packets preserve public layouts and live tensor ownership."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.func._draws import _with_runtime_draws
from dendra.func._inputs import (
    _index_runtime_inputs,
    _legacy_step_input,
    _population_inputs,
    _slice_runtime_inputs,
)

pytestmark = pytest.mark.cpu


def test_deterministic_inputs_preserve_tensor_aliases_and_existing_public_tuples():
    values = torch.arange(12.0, dtype=torch.float64).reshape(3, 4).requires_grad_()
    inputs = dn.func.Inputs(values, values)
    assert inputs._fields == ("ve", "intra")
    assert dn.func.StepInput._fields == ("ve", "intra", "noise", "stimulation")
    assert dn.func.RolloutInput._fields == ("ve", "intra", "noise", "stimulation")
    assert dn.func.NetworkRolloutInput._fields == ("populations", "netstim")
    leaves, spec = pytree.tree_flatten(inputs)
    rebuilt = pytree.tree_unflatten(leaves, spec)
    assert rebuilt.ve is rebuilt.intra is values
    frame = _population_inputs(rebuilt)
    assert frame.ve is frame.intra is values
    assert frame.draws is frame.noise is frame.stimulation is None
    selected = _slice_runtime_inputs(frame, 1, 3)
    step = _index_runtime_inputs(
        selected.ve,
        selected.intra,
        selected.noise,
        selected.stimulation,
        0,
        draws=selected.draws,
    )
    assert step.ve.untyped_storage().data_ptr() == values.untyped_storage().data_ptr()
    assert (
        step.intra.untyped_storage().data_ptr() == values.untyped_storage().data_ptr()
    )
    observed = torch.autograd.grad(step.ve.sum() + 2 * step.intra.sum(), values)[0]
    reference = torch.zeros_like(values)
    reference[1] = 3
    torch.testing.assert_close(observed, reference, rtol=0, atol=0)


@pytest.mark.parametrize("legacy", (False, True))
def test_canonical_draws_have_their_own_operand_and_keep_rng_views(legacy):
    values = torch.arange(12.0).reshape(3, 4)
    rng = torch.arange(24, dtype=torch.uint8).reshape(3, 8)
    draws = dn.func.RuntimeDraws({"probe": values}, {"stream": rng})
    if legacy:
        frame = _population_inputs(dn.func.RolloutInput(ve=values, noise=draws))
    else:
        frame = _with_runtime_draws(dn.func.Inputs(ve=values), draws)
    assert frame.draws is draws
    assert frame.noise is frame.stimulation is None
    selected = _index_runtime_inputs(
        frame.ve, frame.intra, frame.noise, frame.stimulation, 1, draws=frame.draws
    )
    assert (
        selected.draws.rng["stream"].untyped_storage().data_ptr()
        == rng.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(selected.draws.draws["probe"], values[1])
    public = _legacy_step_input(selected)
    assert type(public) is dn.func.StepInput
    assert public.noise is selected.draws


@pytest.mark.parametrize("step", (False, True))
def test_draw_conflicts_keep_legacy_diagnostics(step):
    values = torch.ones(2)
    packet = dn.func.RuntimeDraws({}, {})
    legacy = dn.func.StepInput if step else dn.func.RolloutInput
    for inputs in (legacy(noise=object()), legacy(stimulation=object())):
        with pytest.raises(
            ValueError,
            match="draws cannot be combined with legacy noise or stimulation packets",
        ):
            _with_runtime_draws(inputs, packet, step=step)
    with pytest.raises(TypeError, match="draws must be a RuntimeDraws packet"):
        _with_runtime_draws(dn.func.Inputs(values), object(), step=step)
