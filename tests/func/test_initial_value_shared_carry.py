"""Shared-field and CARRY acceptance for pure initial values."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mod import pas

DT = 0.0125
DTYPE = torch.float64

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


class _SharedSeedSodiumState(State):
    State.STATE("nai")
    State.DERIVATIVE("nai' = 0.0 * nai")


class _SharedSeedMaterialState(State):
    State.STATE("amount")
    State.DERIVATIVE("amount' = 0.0 * amount")


class _RegionalSodiumWriter(Mechanism):
    Mechanism.STATE_BUNDLE(_SharedSeedSodiumState)
    Mechanism.USEION("na", write=["nai"])


class _RegionalMaterialWriter(Mechanism):
    Mechanism.STATE_BUNDLE(_SharedSeedMaterialState)
    Mechanism.USEMATERIAL("pool", write=["amount"])


class _SharedWritableInitialValue(Mechanism):
    Mechanism.GLOBAL(shift=0.25)
    Mechanism.CARRY("carry", "sample")
    Mechanism.USEION("na", write=["nai"])
    Mechanism.USEMATERIAL("pool", write=["amount"])

    def initial_values(self, v, values):
        del v
        nai = values["nai"] + self.shift
        amount = values["amount"] + 2.0 * self.shift
        sample = nai + amount
        return {
            "nai": nai,
            "amount": amount,
            "carry": sample - self.shift,
            "sample": sample,
        }


def _assert_tree_close(actual, expected, *, rtol=0.0, atol=0.0):
    actual_values, actual_spec = torch.utils._pytree.tree_flatten(actual)
    expected_values, expected_spec = torch.utils._pytree.tree_flatten(expected)
    assert actual_spec == expected_spec
    for actual_value, expected_value in zip(
        actual_values,
        expected_values,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=rtol,
            atol=atol,
        )


def _parameter_name(parameters, suffix):
    matches = [name for name in parameters if name.endswith(suffix)]
    assert len(matches) == 1, (suffix, tuple(parameters))
    return matches[0]


def _initialize(functional, tensors, replacements=None):
    parameters = dict(tensors.parameters)
    if replacements:
        parameters.update(replacements)
    return functional.initialize(
        parameters,
        tensors.constants,
        dn.func.InitializationInput(
            v_init=tensors.initialization.v_init,
            states=tensors.initialization.states,
        ),
    )


def _regional_seed_model():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            torch.ones((2, 5), dtype=DTYPE),
            L=4.0,
            dx=1.0,
            v_init=-65.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.concentrations(
            nai0=torch.nn.Parameter(torch.arange(11.0, 16.0, dtype=DTYPE))
        )
        model.material(
            "pool",
            fields={"amount": torch.nn.Parameter(torch.arange(2.0, 7.0, dtype=DTYPE))},
            min_values={"amount": 0.0},
            conserved={"amount": False},
        )
        model.insert(pas)
        model[:, 1:4].insert(_RegionalSodiumWriter)
        model[:, 1:4].insert(_RegionalMaterialWriter)
        model.train()
        model.initialize()
    return model


def test_regional_state_without_defaults_uses_differentiable_shared_initial_seed():
    model = _regional_seed_model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    initialized = _initialize(functional, tensors)
    _assert_tree_close(initialized.state, functional.extract(model).state)

    sodium = initialized.state["mechanisms"]["_RegionalSodiumWriter"]["nai"]
    amount = initialized.state["mechanisms"]["_RegionalMaterialWriter"]["amount"]
    expected_sodium = sodium.new_tensor([[12.0, 13.0, 14.0]]).expand(2, -1)
    expected_amount = amount.new_tensor([[3.0, 4.0, 5.0]]).expand(2, -1)
    torch.testing.assert_close(sodium, expected_sodium)
    torch.testing.assert_close(amount, expected_amount)

    nai_name = _parameter_name(tensors.parameters, "ions.na.i_init")
    amount_name = _parameter_name(
        tensors.parameters,
        "materials.pool._initial_sources.amount",
    )
    base_nai = tensors.parameters[nai_name]
    base_amount = tensors.parameters[amount_name]

    def projected(nai_source, amount_source):
        state = _initialize(
            functional,
            tensors,
            {
                nai_name: nai_source,
                amount_name: amount_source,
            },
        ).state
        return torch.cat(
            (
                state["mechanisms"]["_RegionalSodiumWriter"]["nai"].flatten(),
                state["mechanisms"]["_RegionalMaterialWriter"]["amount"].flatten(),
            )
        )

    reverse = torch.func.jacrev(projected, argnums=(0, 1))(
        base_nai,
        base_amount,
    )
    forward = torch.func.jacfwd(projected, argnums=(0, 1))(
        base_nai,
        base_amount,
    )
    _assert_tree_close(reverse, forward)
    assert torch.count_nonzero(reverse[0]) == 6
    assert torch.count_nonzero(reverse[1]) == 6
    assert torch.count_nonzero(reverse[0][6:]) == 0
    assert torch.count_nonzero(reverse[1][:6]) == 0

    nai_lanes = torch.stack((base_nai - 0.5, base_nai + 0.5))
    amount_lanes = torch.stack((base_amount - 0.25, base_amount + 0.25))
    actual = torch.vmap(projected)(nai_lanes, amount_lanes)
    expected = torch.stack(
        tuple(projected(nai_lanes[index], amount_lanes[index]) for index in range(2))
    )
    torch.testing.assert_close(actual, expected)
    empty = torch.vmap(projected)(nai_lanes[:0], amount_lanes[:0])
    assert empty.shape == (0, 12)


def test_mechanism_initial_values_can_initialize_carry_and_writable_shared_locals():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.SingleCompartment(
            N=1,
            C=3,
            v_init=-65.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.concentrations(nai0=torch.nn.Parameter(torch.tensor(9.25, dtype=DTYPE)))
        model.material(
            "pool",
            fields={"amount": torch.nn.Parameter(torch.tensor(3.5, dtype=DTYPE))},
            min_values={"amount": 0.0},
            conserved={"amount": False},
        )
        model.insert(_SharedWritableInitialValue)
        model.train()
        model.initialize()

    functional, tensors = dn.func.make_functional(model, dt=DT)
    initialized = _initialize(functional, tensors)
    _assert_tree_close(initialized.state, functional.extract(model).state)

    state = initialized.state
    mechanism_name = "_SharedWritableInitialValue"
    expected_nai = state["ions"]["na"]["nai"].new_full(model.shape, 9.5)
    expected_amount = state["materials"]["pool"]["amount"].new_full(
        model.shape,
        4.0,
    )
    expected_sample = expected_nai + expected_amount
    torch.testing.assert_close(state["ions"]["na"]["nai"], expected_nai)
    torch.testing.assert_close(
        state["materials"]["pool"]["amount"],
        expected_amount,
    )
    torch.testing.assert_close(
        state["ion_write_buffers"][mechanism_name]["nai"],
        expected_nai,
    )
    torch.testing.assert_close(
        state["material_write_buffers"][mechanism_name]["amount"],
        expected_amount,
    )
    torch.testing.assert_close(
        state["mechanism_buffers"][mechanism_name]["sample"],
        expected_sample,
    )
    torch.testing.assert_close(
        state["mechanism_buffers"][mechanism_name]["carry"],
        expected_sample - 0.25,
    )

    mechanism = model.mech.mechanisms[mechanism_name]
    (gradient,) = torch.autograd.grad(
        mechanism.sample.sum(),
        mechanism.shift_param,
    )
    torch.testing.assert_close(gradient, gradient.new_tensor(9.0))
