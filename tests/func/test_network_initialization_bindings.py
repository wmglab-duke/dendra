"""Fresh Network admission preserves parameter and constant binding contracts."""

import pytest
import torch

import dendra as dn
from dendra.models.mod import pas
from tests.func.test_network import DT, DTYPE, _close

pytestmark = pytest.mark.cpu


class _ScaleByArgument(torch.nn.Module):
    def forward(self, value, factor):
        return value * factor


class _Add(torch.nn.Module):
    def forward(self, value, amount):
        return (value + amount,)


def _cell():
    cell = dn.SingleCompartment(
        N=1, C=1, integrator=dn.bwd_euler_sc(imem=False), v_init=-60.0
    )
    cell.insert(pas, g=0.001, e=-65.0)
    return cell


def _cross_population_binding(*, shared=True):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE, DEVICE="cpu"):
        parameter = torch.nn.Parameter(
            torch.tensor([1.0], dtype=DTYPE), requires_grad=False
        )
        left, right = _cell(), _cell()
        left.material(
            "pool",
            fields={"amount": parameter},
            min_values={"amount": 0.0},
            conserved={"amount": False},
        )
        right.register_buffer(
            "factor", parameter if shared else parameter.detach().clone()
        )
        right.register_parametrization_in_graph(
            "cm", _ScaleByArgument(), args=("factor",)
        )
        model = dn.Network({"left": left, "right": right})
        model.train()
        model.initialize(DT)
        key = next(
            name for name, value in left.named_parameters() if value is parameter
        )
        # Leave native materializations coherent at lowering. The new action
        # changes the alias only when fresh initialization is requested.
        left.register_pre_initialize_transform(
            "adjust",
            _Add(),
            reads=(f"parameters.{key}",),
            writes=(f"parameters.{key}",),
            inputs={"amount": parameter.new_tensor(1.0)},
        )
    return model, key


@pytest.mark.parametrize("execution", ("pure", "runner"))
def test_fresh_parameter_writes_shared_with_other_population_constants_fail(execution):
    model, key = _cross_population_binding()
    functional, tensors = dn.func.make_functional_network(model)
    parameter = tensors.parameters["populations"]["left"][key]
    constant = tensors.constants["populations"]["right"]["parametrizations.factor"]
    assert parameter is constant
    with pytest.raises(dn.func.FunctionalizationError, match="shared.*constants"):
        if execution == "pure":
            functional.initialize(*tensors, functional.initialization_inputs)
        else:
            dn.compile(
                model,
                dt=DT,
                steps=1,
                initialization="fresh",
                inference=False,
                disable=True,
            )()
    torch.testing.assert_close(parameter, parameter.new_tensor([1.0]))
    torch.testing.assert_close(model.right.factor, parameter.new_tensor([1.0]))


def test_cross_population_constant_alias_does_not_block_snapshot_or_continuation():
    model, _ = _cross_population_binding()
    snapshot = dn.compile(
        model,
        dt=DT,
        steps=1,
        initialization="snapshot",
        inference=False,
        disable=True,
    )
    fresh = dn.compile(
        model,
        dt=DT,
        steps=1,
        initialization="fresh",
        inference=False,
        disable=True,
    )
    expected, _ = snapshot()
    actual, _ = fresh(fresh.initial_state, parameters=fresh.parameters)
    _close(actual, expected)
    assert actual["clock"]["step"].item() == 1


def test_equal_unshared_constants_allow_fresh_initialization_and_native_parity():
    model, _ = _cross_population_binding(shared=False)
    native, _ = _cross_population_binding(shared=False)
    functional, tensors = dn.func.make_functional_network(model)
    assert not functional._parameter_constant_aliases
    initialized = functional.initialize(*tensors, functional.initialization_inputs)
    actual, _ = functional.prepare_and_step(*initialized)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    native.step()
    _close(actual, functional.extract(native).state)


def test_local_parameter_buffer_alias_uses_one_parameter_binding_during_initialization():
    def model():
        with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE, DEVICE="cpu"):
            cell = _cell()
            parameter = torch.nn.Parameter(torch.tensor(1.0, dtype=DTYPE))
            cell.register_parameter("shared_factor", parameter)
            cell.register_buffer("factor", parameter)
            cell.register_parametrization_in_graph(
                "cm", _ScaleByArgument(), args=("factor",)
            )
            network = dn.Network({"cell": cell})
            network.train()
            network.initialize(DT)
            cell.register_pre_initialize_transform(
                "adjust",
                _Add(),
                reads=("parameters.shared_factor",),
                writes=("parameters.shared_factor",),
                inputs={"amount": parameter.new_tensor(1.0)},
            )
        return network

    source, native = model(), model()
    functional, tensors = dn.func.make_functional_network(source)
    assert "shared_factor" in tensors.parameters["populations"]["cell"]
    assert "parametrizations.factor" not in tensors.constants["populations"]["cell"]
    assert not functional._parameter_constant_aliases
    initialized = functional.initialize(*tensors, functional.initialization_inputs)
    torch.testing.assert_close(
        initialized.parameters["populations"]["cell"]["shared_factor"],
        torch.tensor(2.0, dtype=DTYPE),
    )
    actual, _ = functional.prepare_and_step(*initialized)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    native.step()
    _close(actual, functional.extract(native).state)
