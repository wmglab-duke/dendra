"""Host and lowered loops retain per-step input views and tensor transforms."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils import _pytree as pytree

from dendra.func._inputs import _population_inputs, _RuntimeInputs
from dendra.func._lowered import (
    _LoweredCallbackPopulationChunk,
    _LoweredPopulationChunk,
)
from dendra.func._runners import _execute_steps, _execute_steps_with_callbacks
from dendra.func._types import (
    RolloutInput,
    RuntimeNoiseDraw,
    StimulationDraw,
)

pytestmark = pytest.mark.cpu


def _inputs():
    base = torch.arange(24, dtype=torch.float64).reshape(3, 8).requires_grad_()
    frames = torch.arange(15, dtype=torch.uint8).reshape(5, 3).t()
    return base, _population_inputs(
        RolloutInput(
            ve=base[:, ::4],
            intra=base[:, 1::4],
            noise=RuntimeNoiseDraw(
                {"eta": base[:, 2::4]}, {"stream": {"bytes": frames}}
            ),
            stimulation=StimulationDraw({"u": base[:, 3::4]}, {"generator": frames}),
        )
    )


def _transition(state, inputs):
    voltage = (
        state["integrator"]["v"]
        + inputs.ve
        + 2 * inputs.intra
        + 3 * inputs.noise.draws["eta"]
        + 4 * inputs.stimulation.draws["u"]
    )
    return {"integrator": {"v": voltage}}, (
        inputs.noise.rng,
        inputs.stimulation.rng,
    )


def _executor(entry, step):
    """Use the real loops with a small pure transition and callback counter."""
    functional = SimpleNamespace(
        _validate_source=lambda: None,
        _step_values=lambda parameters, prepared, state, inputs: step(state, inputs),
    )
    callback = SimpleNamespace(update=lambda carry, state, auxiliary: (carry + 1, None))
    if entry == "host":
        return lambda state, inputs: _execute_steps(step, state, inputs, 3)
    if entry == "host_callbacks":
        callbacks = SimpleNamespace(_update=callback.update, _stack=lambda values: {})
        return lambda state, inputs: _execute_steps_with_callbacks(
            step, state, inputs, 3, callbacks, torch.tensor(0)
        )
    if entry == "lowered":
        kernel = _LoweredPopulationChunk(functional, steps=3, stepwise_stimulation=True)
        carries = None
    else:
        assert entry == "lowered_callbacks"
        plan = SimpleNamespace(
            name="count",
            callback=callback,
            emission_schema=None,
            validate_updates=False,
        )
        kernel = _LoweredCallbackPopulationChunk(functional, steps=3, plans=(plan,))
        carries = {"count": torch.tensor(0)}
    return lambda state, inputs: kernel(
        {},
        {},
        state,
        inputs.ve,
        inputs.intra,
        carries,
        inputs.noise,
        inputs.stimulation,
    )


@pytest.mark.parametrize(
    "entry", ("host", "host_callbacks", "lowered", "lowered_callbacks")
)
def test_all_indexed_loops_pass_borrowed_rows_and_retain_gradients(entry):
    base, inputs = _inputs()
    frozen = pytree.tree_map_only(torch.Tensor, torch.clone, inputs)
    received = []

    def observe(state, row):
        assert type(row) is _RuntimeInputs
        assert row.draws is None
        received.append(row)
        return _transition(state, row)

    initial = {"integrator": {"v": torch.zeros(2, dtype=torch.float64)}}
    result = _executor(entry, observe)(initial, inputs)
    voltage = result[0]["integrator"]["v"]
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64).repeat(2)
    torch.testing.assert_close(
        voltage, (base.reshape(3, 2, 4) * weights[:4]).sum((0, 2))
    )
    torch.testing.assert_close(
        torch.autograd.grad(voltage.sum(), base)[0], weights.expand_as(base)
    )
    assert len(received) == 3
    for index, row in enumerate(received):
        for actual, tape in zip(
            pytree.tree_leaves(row), pytree.tree_leaves(inputs), strict=True
        ):
            if actual is None:
                assert tape is None
                continue
            expected = tape[index]
            assert (
                actual.untyped_storage().data_ptr() == tape.untyped_storage().data_ptr()
            )
            assert actual.storage_offset() == expected.storage_offset()
            assert actual.stride() == expected.stride()
            torch.testing.assert_close(actual, expected)
        assert row.noise.rng["stream"]["bytes"] is not row.stimulation.rng["generator"]
        assert (
            row.noise.rng["stream"]["bytes"].data_ptr()
            == row.stimulation.rng["generator"].data_ptr()
        )
    if entry.endswith("callbacks"):
        carry = result[2] if entry == "host_callbacks" else result[2]["count"]
        assert carry.item() == 3
        assert result[3] == ({} if entry == "host_callbacks" else {"count": None})
    pytree.tree_map(torch.testing.assert_close, inputs, frozen)


@pytest.mark.parametrize("entry", ("lowered", "lowered_callbacks"))
def test_indexed_tensor_loops_compile_fullgraph_with_all_input_columns(entry):
    base, inputs = _inputs()
    execute = _executor(entry, _transition)
    initial = {"integrator": {"v": torch.zeros(2, dtype=torch.float64)}}
    expected = execute(initial, inputs)
    compiled = torch.compile(execute, backend="aot_eager", fullgraph=True)
    actual = compiled(initial, inputs)
    pytree.tree_map(torch.testing.assert_close, actual, expected)
    expected_gradient = torch.autograd.grad(
        expected[0]["integrator"]["v"].square().sum(), base
    )[0]
    actual_gradient = torch.autograd.grad(
        actual[0]["integrator"]["v"].square().sum(), base
    )[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
