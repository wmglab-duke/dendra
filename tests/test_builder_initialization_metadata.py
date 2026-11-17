"""Regression for native State builders with large population index metadata."""

import pytest
import torch

import dendra  # noqa: F401 - initialize Dendra before mechanism imports
from dendra.models.mechanisms import Mechanism, State


@pytest.mark.parametrize(
    ("container", "operation"),
    ((list, "read"), (tuple, "read"), (list, "edit"), (list, "rebind")),
)
def test_native_derived_builder_isolates_large_integer_metadata(container, operation):
    class IndexedState(State):
        State.STATE("x")
        State.DERIVATIVE("x' = -x")
        State.DERIVED_BUFFER("workspace")

        def derive_buffers(self):
            # State receives the original Python selector, while Mechanism
            # owns the corresponding registered Tensor selector.
            assert self.key is self.selector_alias
            scale = self.key[-1] / len(self.key)
            if self.operation == "edit":
                self.key[-1] -= 1
            elif self.operation == "rebind":
                self.key = self.key.copy()
            return {"workspace": scale * self.diam}

    class IndexedMechanism(Mechanism):
        Mechanism.STATE_BUNDLE(IndexedState)

    n_compartments = 8_192
    shape = (1, n_compartments)
    key = container(range(n_compartments))
    diameter = torch.ones(shape, dtype=torch.float64, requires_grad=True)
    mechanism = IndexedMechanism(
        "indexed",
        torch.tensor(37.0, dtype=torch.float64),
        diameter,
        shape,
        shape,
        key=key,
    )
    state = mechanism.DE.IndexedState
    state.selector_alias = key
    state.operation = operation
    voltage = torch.zeros_like(diameter)

    if operation == "read":
        mechanism._init_buffers_s(voltage)
        scale = (n_compartments - 1) / n_compartments
        expected = torch.full_like(diameter, scale)
        torch.testing.assert_close(state.workspace, expected, rtol=0.0, atol=0.0)
        (grad,) = torch.autograd.grad(state.workspace.sum(), diameter)
        torch.testing.assert_close(grad, expected, rtol=0.0, atol=0.0)
    else:
        with pytest.raises(RuntimeError, match="must not mutate Python instance state"):
            mechanism._init_buffers_s(voltage)
        assert torch.count_nonzero(state.workspace) == 0

    assert state.key is state.selector_alias is key
    assert key == container(range(n_compartments))
