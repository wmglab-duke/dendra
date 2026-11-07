"""Fresh initialization parity for myelinated, finite-shell, and packed models.

Keep recurrence lengths short. Directional finite differences re-run actual
imperative initialization, whose voltage reset intentionally detaches Vinit;
imperative autograd alone would therefore be the wrong Vinit-gradient oracle.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mod import hh
from tests.func.test_functional_initialization import (
    AM1,
    _assert_module_snapshot,
    _assert_tensor_snapshot,
    _assert_tree_close,
    _module_snapshot,
    _tensor_snapshot,
)
from tests.func.test_membrane_current import _model as _membrane_model

DT = 0.01
STEPS = 3
KINDS = (
    "myelinated",
    "cable",
    "block_path",
    "block_tree",
    "multi_point",
    "multi_scalar",
    "multi_block",
)
CHANNELS = ("gnabar", "gkbar", "gl", "ena", "ek", "el")
PROBE = "_InitializationProbe"
IC = f"mechanisms.{PROBE}.fixed"

pytestmark = pytest.mark.cpu


class _ProbeState(State):
    State.STATE("inferred", "fixed")
    State.DERIVATIVE("inferred' = -0.01 * inferred", "fixed' = 0.0 * fixed")

    def state_defaults(self, v, values):
        del values
        return {"inferred": torch.sigmoid(0.02 * v)}


class _InitializationProbe(Mechanism):
    """Generic regional initialization reads explicit geometry and shared Ion."""

    Mechanism.STATE_BUNDLE(_ProbeState)
    Mechanism.GLOBAL(gain=0.2, g=2.0e-5, e=-61.0)
    Mechanism.CARRY("seed")
    Mechanism.USEION("na", read=["nai"])
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def initial_values(self, v, values):
        del v
        inferred = values["inferred"] + self.gain * (
            0.002 * values["diam"] + 0.001 * values["celsius"] + 0.003 * values["nai"]
        )
        return {"inferred": inferred, "seed": inferred + values["fixed"]}

    def i(self, v):
        return self.g * (self.inferred + 0.1 * self.fixed) * (v - self.e)


@pytest.fixture(params=(1, 4), ids=("threads1", "threads4"))
def cpu_threads(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    try:
        yield request.param
    finally:
        torch.set_num_threads(previous)


def _model(
    kind,
    *,
    device="cpu",
    dtype=torch.float64,
    batch=(2,),
    imem=True,
    shell_resistance_scale=1.0,
):
    model = _membrane_model(kind, device=device, dtype=dtype, imem=imem, batch=None)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=imem, DEVICE=device, DTYPE=dtype):
        components = (
            tuple(model.populations.values()) if kind.startswith("multi_") else (model,)
        )
        for component in components:
            if hasattr(component, "xraxial"):
                component.xraxial.mul_(shell_resistance_scale)
        model.concentrations(nai0=torch.nn.Parameter(model.v.new_tensor(9.25)))
        # A packed model has heterogeneous mechanism support: the extra probe
        # belongs to one component only, while HH spans the consolidated pack.
        target = (
            getattr(model, next(iter(model.populations)))
            if kind.startswith("multi_")
            else model[:]
        )
        target[..., ::2].insert(_InitializationProbe, ic={"fixed": 0.45})
        for size in batch:
            model.batch(size)
        model.initialize()
        model.train()
    model._initialization_parity_kind = kind
    model._initialization_parity_batch = tuple(batch)
    model._initialization_parity_shell_scale = shell_resistance_scale
    return model


def _voltage(model):
    return torch.linspace(
        -72.0, -52.0, model.v.numel(), device=model.device(), dtype=model.dtype()
    ).reshape(model.shape)


def _inputs(model, steps=STEPS):
    shape = (steps, *model.shape)
    options = {"device": model.device(), "dtype": model.dtype()}
    return dn.func.RolloutInput(
        ve=torch.linspace(-0.3, 0.4, steps * model.v.numel(), **options).reshape(shape),
        intra=torch.linspace(
            -2.0e-9, 3.0e-9, steps * model.v.numel(), **options
        ).reshape(shape),
    )


def _initialize(
    functional, tensors, voltage, parameters=None, constants=None, states=None
):
    return functional.initialize(
        tensors.parameters if parameters is None else parameters,
        tensors.constants if constants is None else constants,
        tensors.initialization._replace(
            v_init=voltage,
            states=tensors.initialization.states if states is None else states,
        ),
    )


def _live_state(functional, model):
    state = {}
    for leaf in functional._state_layout:
        owner = model if not leaf.module_path else model.get_submodule(leaf.module_path)
        cursor = state
        for part in leaf.public_path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[leaf.public_path[-1]] = getattr(owner, leaf.buffer_name).clone()
    return state


def _constant_target(model, key):
    owner_path, _, name = key.rpartition(".")
    owner = model if not owner_path else model.get_submodule(owner_path)
    if name == "diam_original":
        return owner.parametrizations.diam.original
    if name == "canonical_area_cm2" and isinstance(owner, (dn.Tree, dn.ExtCellTree)):
        return owner.area
    if name == "canonical_edge_resistance_ohm" and isinstance(
        owner, (dn.Tree, dn.ExtCellTree)
    ):
        return owner.edge_resistance_ohm
    if name.startswith("canonical_"):
        name = "_" + name
    return getattr(owner, name)


def _imperative_initialized(
    functional,
    model,
    voltage,
    parameters=None,
    constants=None,
    states=None,
    *,
    reference=None,
):
    """Re-run the real initializer on a separate model, never the lowered source."""
    if reference is None:
        reference = _model(
            model._initialization_parity_kind,
            device=model.device(),
            dtype=model.dtype(),
            batch=model._initialization_parity_batch,
            imem=model.integrator.imem,
            shell_resistance_scale=model._initialization_parity_shell_scale,
        )
    with torch.no_grad():
        if parameters is not None:
            raw = dict(reference.named_parameters())
            for name, value in parameters.items():
                raw[name].copy_(value)
        if constants is not None:
            for name, value in constants.items():
                if name != "dt" and not name.startswith("parametrizations."):
                    target = _constant_target(reference, name)
                    if not torch.equal(target, value):
                        target.copy_(value)
        if states is not None:
            for name, value in states.items():
                _, mechanism_name, state_name = name.split(".")
                reference.mech.mechanisms[mechanism_name]._init_params[state_name] = (
                    value.detach().clone()
                )
        reference.set_v_init(voltage)
        reference.initialize()
    return reference, _live_state(functional, reference)


def _stack(states):
    return pytree.tree_map(lambda *values: torch.stack(values), *states)


def _functional_history(functional, initialized, inputs):
    prepared = functional.prepare(initialized.parameters, initialized.constants)
    state = initialized.state
    states = [state]
    for index in range(inputs.intra.shape[0]):
        state, _ = functional.step(
            initialized.parameters,
            prepared,
            state,
            dn.func.StepInput(inputs.ve[index], inputs.intra[index]),
        )
        states.append(state)
    return _stack(states)


def _imperative_history(functional, model, inputs):
    timestep = model.v.new_tensor(DT)
    model.integrator._initialize(
        model, timestep, force=True, compile_scope="population"
    )
    states = [_live_state(functional, model)]
    for index in range(inputs.intra.shape[0]):
        model.integrator.step(model, timestep, inputs.ve[index], inputs.intra[index])
        model.t = model.t + timestep
        states.append(_live_state(functional, model))
    return _stack(states)


def _loss(history):
    integrator = history["integrator"]
    loss = 0.001 * integrator["v"].square().mean()
    if "i_membrane" in integrator:
        loss = loss + 1.0e-4 * (integrator["i_membrane"] * 1.0e6).square().mean()
    if "vc" in integrator:
        loss = loss + 0.0001 * integrator["vc"].square().mean()
    loss = loss + 0.1 * history["mechanisms"]["hh"]["m"].square().mean()
    loss = loss + 0.02 * history["mechanisms"][PROBE]["inferred"].square().mean()
    loss = loss + 0.01 * history["mechanisms"][PROBE]["fixed"].square().mean()
    return loss + 0.01 * history["mechanism_buffers"][PROBE]["seed"].square().mean()


def _arguments(model, tensors):
    parameters = {
        name: value.detach().clone() for name, value in tensors.parameters.items()
    }
    targets = {
        name: parameters[f"integrator.mech.mechanisms.hh.{name}_param"].requires_grad_()
        for name in CHANNELS
    }
    targets["rate"] = parameters[AM1].requires_grad_()
    voltage = _voltage(model).requires_grad_()
    targets["initial_voltage"] = voltage
    amplitude = model.v.new_tensor(1.0, requires_grad=True)
    targets["stimulus"] = amplitude
    inputs = _inputs(model)
    inputs = inputs._replace(intra=inputs.intra * amplitude)
    return parameters, voltage, inputs, targets


def _close_history(actual, expected, *, dtype=torch.float64):
    actual_paths, actual_spec = pytree.tree_flatten_with_path(actual)
    expected_paths, expected_spec = pytree.tree_flatten_with_path(expected)
    assert actual_spec == expected_spec
    for (path, value), (reference_path, reference) in zip(
        actual_paths, expected_paths, strict=True
    ):
        assert path == reference_path
        assert torch.isfinite(value).all() and torch.isfinite(reference).all(), path
        if tuple(getattr(part, "key", None) for part in path) in {
            ("integrator", "i_membrane"),
            ("i_membrane",),
        }:
            value, reference = value * 1.0e6, reference * 1.0e6
        torch.testing.assert_close(
            value,
            reference,
            rtol=2.0e-4 if dtype == torch.float32 else 3.0e-9,
            atol=2.0e-5 if dtype == torch.float32 else 3.0e-10,
            msg=lambda message: f"{path}: {message}",
        )


@pytest.mark.parametrize("kind", KINDS)
def test_fresh_family_initialization_matches_all_imperative_leaves_and_full_traces(
    kind, cpu_threads
):
    model = _model(kind)
    model.t.fill_(3.0)
    model._duration_remainder.fill_(0.004)
    model.i_membrane.fill_(0.25)
    model.mech.mechanisms[PROBE].seed.fill_(91.0)
    if hasattr(model, "vc"):
        model.vc[..., 1:].fill_(2.0)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    voltage = _voltage(model)
    parameters = dict(tensors.parameters)
    parameters[AM1] = parameters[AM1] * 1.13
    states = {
        name: value + 0.07 for name, value in tensors.initialization.states.items()
    }
    snapshot = _module_snapshot(model)
    actual = _initialize(functional, tensors, voltage, parameters, states=states)
    reference, expected = _imperative_initialized(
        functional, model, voltage, parameters, states=states
    )
    _assert_tree_close(actual.state, expected)
    assert torch.count_nonzero(actual.state["integrator"]["i_membrane"]) == 0
    assert (
        actual.state["clock"]["t"].item()
        == actual.state["control"]["duration_remainder"].item()
        == 0
    )
    if "vc" in actual.state["integrator"]:
        torch.testing.assert_close(actual.state["integrator"]["vc"][..., 0], voltage)
        assert torch.count_nonzero(actual.state["integrator"]["vc"][..., 1:]) == 0
    _close_history(
        _functional_history(functional, actual, _inputs(model)),
        _imperative_history(functional, reference, _inputs(model)),
    )
    _assert_module_snapshot(model, snapshot)


@pytest.mark.parametrize("kind", ("myelinated", "multi_block"))
def test_float32_fresh_family_traces_match_imperative(kind, cpu_threads):
    model = _model(kind, dtype=torch.float32, shell_resistance_scale=100.0)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    voltage = _voltage(model)
    actual = _initialize(functional, tensors, voltage)
    reference, expected = _imperative_initialized(functional, model, voltage)
    _close_history(actual.state, expected, dtype=torch.float32)
    _close_history(
        _functional_history(functional, actual, _inputs(model)),
        _imperative_history(functional, reference, _inputs(model)),
        dtype=torch.float32,
    )


@pytest.mark.parametrize("kind", KINDS)
def test_fresh_family_all_eight_and_rate_gradients_match_imperative_differences(
    kind, cpu_threads
):
    model = _model(kind)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters, voltage, inputs, targets = _arguments(model, tensors)
    initialized = _initialize(functional, tensors, voltage, parameters)
    history = _functional_history(functional, initialized, inputs)
    gradients = torch.autograd.grad(_loss(history), tuple(targets.values()))
    assert len(gradients) == 9
    reference, _ = _imperative_initialized(functional, model, voltage, parameters)
    for name, target, gradient in zip(
        targets, targets.values(), gradients, strict=True
    ):
        assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0, (
            name
        )
        direction = (
            torch.linspace(
                0.5, 1.5, target.numel(), device=target.device, dtype=target.dtype
            ).reshape_as(target)
            if target.ndim
            else torch.ones_like(target)
        )
        scale = max(float(target.detach().abs().max()), 1.0e-3)
        epsilon = 1.0e-5 * scale
        losses = []
        for sign in (-1.0, 1.0):
            candidate_parameters = dict(parameters)
            candidate_voltage = voltage.detach()
            candidate_inputs = _inputs(model)
            changed = target.detach() + sign * epsilon * direction
            if name == "initial_voltage":
                candidate_voltage = changed
            elif name == "stimulus":
                candidate_inputs = candidate_inputs._replace(
                    intra=candidate_inputs.intra * changed
                )
            else:
                candidate_parameters[
                    AM1
                    if name == "rate"
                    else f"integrator.mech.mechanisms.hh.{name}_param"
                ] = changed
            with torch.no_grad():
                reference, _ = _imperative_initialized(
                    functional,
                    model,
                    candidate_voltage,
                    candidate_parameters,
                    reference=reference,
                )
                losses.append(
                    _loss(_imperative_history(functional, reference, candidate_inputs))
                )
        numerical = (losses[1] - losses[0]) / (2.0 * epsilon)
        analytic = (gradient * direction).sum()
        torch.testing.assert_close(
            analytic,
            numerical,
            rtol=4.0e-4,
            atol=2.0e-7,
            msg=lambda message: f"{kind}/{name}: {message}",
        )


def _parameter_name(parameters, suffix):
    names = [name for name in parameters if name.endswith(suffix)]
    assert len(names) == 1, (suffix, names)
    return names[0]


@pytest.mark.parametrize(
    "kind", ("myelinated", "block_path", "multi_scalar", "multi_block")
)
def test_generic_initialization_ic_shared_field_and_geometry_gradients(
    kind, cpu_threads
):
    model = _model(kind)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters = {
        name: value.detach().clone() for name, value in tensors.parameters.items()
    }
    constants = {
        name: value.detach().clone() for name, value in tensors.constants.items()
    }
    targets = {}
    for label, suffix in (
        ("gain", f"{PROBE}.gain_param"),
        ("temperature", "celsius_param"),
        ("nai", "ions.na.i_init"),
    ):
        # The packed public temperature is the parent scalar, never a private
        # child parameter retained only by its imperative source container.
        name = (
            "celsius_param"
            if label == "temperature"
            else _parameter_name(parameters, suffix)
        )
        targets[label] = parameters[name].requires_grad_()
    if kind == "myelinated":
        geometry = _parameter_name(parameters, "noded2")
        targets["geometry"] = parameters[geometry].requires_grad_()
    else:
        geometry = {
            "block_path": "diam",
            "multi_scalar": "populations.point.diam",
            "multi_block": "populations.block_path.diam",
        }[kind]
        targets["geometry"] = constants[geometry].requires_grad_()
    voltage = _voltage(model).requires_grad_()
    targets["voltage"] = voltage
    states = {
        name: value.detach().clone()
        for name, value in tensors.initialization.states.items()
    }
    states[IC] = (states[IC] + 0.08).requires_grad_()
    targets["ic"] = states[IC]
    source_snapshot = _module_snapshot(model)
    snapshots = {name: _tensor_snapshot(value) for name, value in targets.items()}
    initialized = _initialize(
        functional, tensors, voltage, parameters, constants, states
    )
    actual = initialized.state["mechanisms"][PROBE]
    carry = initialized.state["mechanism_buffers"][PROBE]["seed"]

    # An independent declaration-level reference, not another call to the new
    # initializer. Preparation supplies differentiated model-family geometry;
    # the public support map supplies the probe's sparse local coordinates.
    mechanism = model.mech.mechanisms[PROBE]
    prepared = functional.prepare(parameters, constants)
    local_v = mechanism.get(voltage)
    diameter = prepared.values["population"].get(
        "diam", prepared.values["geometry"].get("diam")
    )
    assert diameter is not None
    local_diam = mechanism.get(diameter)
    expected = torch.sigmoid(0.02 * local_v) + targets["gain"] * (
        0.002 * local_diam + 0.001 * targets["temperature"] + 0.003 * targets["nai"]
    )
    expected_carry = expected + states[IC]
    torch.testing.assert_close(actual["inferred"], expected, rtol=2.0e-12, atol=2.0e-12)
    torch.testing.assert_close(actual["fixed"], states[IC])
    torch.testing.assert_close(carry, expected_carry, rtol=2.0e-12, atol=2.0e-12)
    actual_loss = actual["inferred"].square().mean() + carry.square().mean()
    reference_loss = expected.square().mean() + expected_carry.square().mean()
    gradients = torch.autograd.grad(actual_loss, tuple(targets.values()))
    reference_gradients = torch.autograd.grad(reference_loss, tuple(targets.values()))
    for name, actual_gradient, reference in zip(
        targets, gradients, reference_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, reference, rtol=2.0e-10, atol=2.0e-11
        )
        assert (
            torch.isfinite(actual_gradient).all()
            and torch.count_nonzero(actual_gradient) > 0
        ), name
    _assert_module_snapshot(model, source_snapshot)
    for name, value in targets.items():
        _assert_tensor_snapshot(value, snapshots[name])


@pytest.mark.parametrize("kind", ("myelinated", "multi_block"))
def test_new_initializers_preserve_vmap_empty_lanes_and_higher_order_ad(
    kind, cpu_threads
):
    model = _model(kind)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    voltage = _voltage(model)
    rate = tensors.parameters[AM1]

    def gates(offset, am1):
        parameters = {**tensors.parameters, AM1: am1}
        # PyTorch's zero-lane vmap cannot broadcast a visibly scalar mapped
        # offset against an unmapped matrix. Give the test-authored offset its
        # visible rank before entering the functional initialization API.
        shifted = voltage + offset.reshape((1,) * voltage.ndim)
        state = _initialize(functional, tensors, shifted, parameters).state
        return torch.cat(
            (
                state["mechanisms"]["hh"]["m"].flatten(),
                state["mechanisms"][PROBE]["inferred"].flatten(),
            )
        )

    offsets = voltage.new_tensor([-1.0, 0.5])
    rates = torch.stack((rate * 0.8, rate * 1.2))
    actual = torch.vmap(gates)(offsets, rates)
    expected = torch.stack(
        [
            gates(offset, local_rate)
            for offset, local_rate in zip(offsets, rates, strict=True)
        ]
    )
    torch.testing.assert_close(actual, expected)
    assert torch.vmap(gates)(offsets[:0], rates[:0]).shape == (0, actual.shape[-1])
    reverse = torch.func.jacrev(gates, argnums=(0, 1))(offsets[0], rate)
    forward = torch.func.jacfwd(gates, argnums=(0, 1))(offsets[0], rate)
    _assert_tree_close(reverse, forward, rtol=2.0e-10, atol=2.0e-11)
    hessian = torch.func.hessian(lambda offset: gates(offset, rate).square().mean())(
        offsets[0]
    )
    assert torch.isfinite(hessian) and hessian.abs() > 0
    # Initial gate dependence is real, not merely the direct identity Vinit
    # contribution from returning a fresh voltage tensor.
    initial_gate_gradient = torch.func.grad(
        lambda offset: gates(offset, rate).square().mean()
    )(offsets[0])
    assert torch.isfinite(initial_gate_gradient) and initial_gate_gradient.abs() > 0


class _PreLayerShift(torch.nn.Module):
    def forward(self, voltage, layers, shift):
        delta = shift.expand_as(voltage)
        return voltage + delta, layers + torch.stack(
            (2.0 * delta, 0.5 * delta, -0.25 * delta), dim=-1
        )


class _PostLayerShift(torch.nn.Module):
    def forward(self, voltage, layers, m, shift):
        delta = shift.expand_as(voltage)
        return voltage + delta * m, layers + torch.stack(
            (0.1 * delta, 0.2 * delta, 0.3 * delta), dim=-1
        )


def _register_layer_transforms(model):
    voltage, layers = "state.integrator.v", "state.integrator.vc"
    model.register_pre_initialize_transform(
        "layer_pre",
        _PreLayerShift(),
        reads=(voltage, layers),
        writes=(voltage, layers),
        inputs={"shift": model.v.new_tensor(0.2)},
    )
    model.register_post_initialize_transform(
        "layer_post",
        _PostLayerShift(),
        reads=(voltage, layers, "state.mechanisms.hh.m"),
        writes=(voltage, layers),
        inputs={"shift": model.v.new_tensor(0.3)},
    )


@pytest.mark.parametrize("kind", ("block_path", "multi_block"))
def test_block_initializer_pre_and_post_transforms_receive_fresh_layer_frame(
    kind, cpu_threads
):
    model = _model(kind)
    _register_layer_transforms(model)
    model.initialize()
    model.vc[..., 1:].fill_(8.0)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    voltage = _voltage(model).requires_grad_()
    transforms = {
        name: value.detach().clone().requires_grad_()
        for name, value in tensors.initialization.transforms.items()
    }
    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization._replace(v_init=voltage, transforms=transforms),
    )
    reference = _model(kind)
    _register_layer_transforms(reference)
    _, expected = _imperative_initialized(
        functional, model, voltage, reference=reference
    )
    _assert_tree_close(initialized.state, expected, rtol=2.0e-12, atol=2.0e-12)
    shift, post = transforms["pre.layer_pre.shift"], transforms["post.layer_post.shift"]
    zero = torch.zeros_like(voltage)
    expected_layers = torch.stack((voltage, zero, zero), -1) + torch.stack(
        (
            2.0 * shift.expand_as(voltage),
            0.5 * shift.expand_as(voltage),
            -0.25 * shift.expand_as(voltage),
        ),
        -1,
    )
    expected_layers = expected_layers + torch.stack(
        (
            0.1 * post.expand_as(voltage),
            0.2 * post.expand_as(voltage),
            0.3 * post.expand_as(voltage),
        ),
        -1,
    )
    torch.testing.assert_close(initialized.state["integrator"]["vc"], expected_layers)
    torch.testing.assert_close(
        initialized.state["integrator"]["v"],
        voltage + shift + post * initialized.state["mechanisms"]["hh"]["m"],
    )
    gradients = torch.autograd.grad(
        _loss(_stack([initialized.state])), (voltage, *transforms.values())
    )
    assert all(
        torch.isfinite(value).all() and torch.count_nonzero(value) > 0
        for value in gradients
    )


def _unexpected_child_hook(population):
    del population
    raise AssertionError("parent initialization must not replay component hooks")


@pytest.mark.parametrize("kind", ("multi_scalar", "multi_block"))
def test_packed_parent_initializer_does_not_execute_or_mutate_children(
    kind, cpu_threads
):
    model = _model(kind)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    for component in model.populations.values():
        component.register_pre_initialize_hook(_unexpected_child_hook)
        component.register_post_initialize_hook(_unexpected_child_hook)
        component.t.fill_(7.0)
    snapshots = {
        name: _module_snapshot(child) for name, child in model.populations.items()
    }
    initialized = _initialize(functional, tensors, _voltage(model))
    reference = _model(kind)
    for component in reference.populations.values():
        component.register_pre_initialize_hook(_unexpected_child_hook)
        component.register_post_initialize_hook(_unexpected_child_hook)
    _, expected = _imperative_initialized(
        functional, model, _voltage(model), reference=reference
    )
    _assert_tree_close(initialized.state, expected)
    for name, component in model.populations.items():
        _assert_module_snapshot(component, snapshots[name])


@pytest.mark.parametrize("block", (False, True), ids=("scalar", "block"))
def test_packed_default_initialization_retains_heterogeneous_component_voltages(block):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:

        def packed():
            kinds = ("block_path", "block_tree") if block else ("point", "path")
            components = {
                name: _membrane_model(kind, batch=None)
                for name, kind in zip(("first", "second"), kinds, strict=True)
            }
            first, second = components.values()
            first.set_v_init(-73.0)
            # An explicit core tensor distinguishes both cells and individual
            # compartments. Establish both defaults before concatenation.
            per_cell = torch.linspace(
                -65.0,
                -54.0,
                second.v.numel(),
                device=second.device(),
                dtype=second.dtype(),
            ).reshape(second.shape)
            second.set_v_init(per_cell)
            expected_core = torch.cat(
                (first.v.new_full((first.v.numel(),), -73.0), per_cell.flatten())
            ).unsqueeze(0)
            with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True, DTYPE=torch.float64):
                model = dn.concat_models(components, celsius=6.3, threads=2)
                model.batch(2)
                model.initialize()
                model.train()
            return model, expected_core.unsqueeze(0).expand(model.shape)

        model, expected_voltage = packed()
        with torch.no_grad():
            # A runtime snapshot must not replace the configured defaults.
            model.v.add_(2.0)
            if block:
                model.vc[..., 0].add_(2.0)
            model.t.fill_(4.0)
            model.i_membrane.fill_(0.25)
        functional, tensors = dn.func.make_functional(model, dt=DT)
        torch.testing.assert_close(
            tensors.initialization.v_init, expected_voltage, rtol=0.0, atol=0.0
        )
        source = _module_snapshot(model)
        children = {
            name: _module_snapshot(child) for name, child in model.populations.items()
        }
        initial_voltage = _tensor_snapshot(tensors.initialization.v_init)
        initialized = functional.initialize(
            tensors.parameters, tensors.constants, tensors.initialization
        )
        reference, _ = packed()
        _assert_tree_close(initialized.state, _live_state(functional, reference))
        torch.testing.assert_close(
            initialized.state["integrator"]["v"], expected_voltage, rtol=0.0, atol=0.0
        )
        _assert_module_snapshot(model, source)
        for name, child in model.populations.items():
            _assert_module_snapshot(child, children[name])
        _assert_tensor_snapshot(tensors.initialization.v_init, initial_voltage)
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("kind", ("myelinated", "multi_block"))
def test_new_initializer_fails_closed_after_parent_hook_changes(kind):
    model = _model(kind)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    model.register_pre_initialize_hook(_unexpected_child_hook)
    with pytest.raises(
        dn.func.FunctionalizationError, match="initialization|fresh-initialization"
    ):
        _initialize(functional, tensors, _voltage(model))


def test_packed_initializer_rejects_private_component_parameter_transform():
    model = _model("multi_point")
    path = "parameters.populations.point.celsius_param"
    model.register_pre_initialize_transform(
        "private_temperature",
        _AddScalar(),
        reads=(path,),
        writes=(path,),
        inputs={"shift": model.v.new_tensor(0.2)},
    )
    functional, tensors = dn.func.make_functional(model, dt=DT)
    with pytest.raises(
        dn.func.FunctionalizationError, match="canonical|public.*mapping"
    ):
        _initialize(functional, tensors, _voltage(model))


class _AddScalar(torch.nn.Module):
    def forward(self, value, shift):
        return (value + shift,)


def test_scalar_initializer_rejects_nonexistent_layer_reference():
    model = _model("multi_point")
    with pytest.raises((ValueError, AttributeError), match="vc|integrator|voltage"):
        model.register_pre_initialize_transform(
            "missing_layers",
            _AddScalar(),
            reads=("state.integrator.vc",),
            writes=("state.integrator.vc",),
            inputs={"shift": model.v.new_tensor(0.2)},
        )


def test_unbuilt_extcell_default_integrator_accepts_public_layer_transforms():
    pytest.importorskip("dendra_solvers")
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True, DTYPE=torch.float64):
        model = dn.ExtCellAxon([5.0], n_comp=3, celsius=6.3, dtype=torch.float64)
        model.insert(hh)
        assert not model.is_built
        _register_layer_transforms(model)
        assert not model.is_built  # Reference registration must not build it.
        model.initialize()
        model.train()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    initialized = functional.initialize(
        tensors.parameters, tensors.constants, tensors.initialization
    )
    _assert_tree_close(initialized.state, _live_state(functional, model))
    assert torch.count_nonzero(initialized.state["integrator"]["vc"][..., 1:]) > 0


@pytest.mark.parametrize("kind", ("myelinated", "multi_block"))
def test_new_initializers_without_membrane_output_keep_their_original_state_schema(
    kind,
):
    model = _model(kind, imem=False)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    voltage = _voltage(model)
    initialized = _initialize(functional, tensors, voltage)
    _, expected = _imperative_initialized(functional, model, voltage)
    assert "i_membrane" not in initialized.state["integrator"]
    _assert_tree_close(initialized.state, expected)


def _record_paths(functional):
    paths = {}
    for leaf in functional._state_layout:
        if leaf.public_path[0] == "integrator":
            paths[leaf.public_path[-1]] = leaf.public_path
        elif leaf.public_path == ("clock", "t"):
            paths["t"] = leaf.public_path
        elif leaf.checkpoint_key is not None:
            paths[leaf.checkpoint_key] = leaf.public_path
    return paths


def _read_path(tree, path):
    for part in path:
        tree = tree[part]
    return tree


@pytest.mark.slow
@pytest.mark.parametrize("kind", ("myelinated", "multi_block"))
def test_fresh_initialization_composes_with_cpp_scan_and_all_eight_gradients(
    kind, cpu_threads
):
    # Keep both CPU thread counts, but only two tiny model graphs: neither
    # initialization nor scan needs a long unrolled compiler matrix.
    previous = torch.get_num_threads()
    torch.set_num_threads(cpu_threads)
    try:
        model = _model(kind)
        functional, tensors = dn.func.make_functional(model, dt=DT)
        parameters, voltage, inputs, targets = _arguments(model, tensors)
        eager_initialized = _initialize(functional, tensors, voltage, parameters)
        expected = _functional_history(functional, eager_initialized, inputs)

        def initialize(v_init, raw_parameters, constants):
            return _initialize(
                functional, tensors, v_init, raw_parameters, constants
            ).state

        compiled_initialize = torch.compile(
            initialize,
            backend="inductor",
            fullgraph=True,
            options={"cpp_wrapper": True, "compile_threads": 1},
        )
        with torch_compiler_warning_context():
            compiled_state = compiled_initialize(voltage, parameters, tensors.constants)
        _assert_tree_close(
            compiled_state, eager_initialized.state, rtol=2.0e-12, atol=2.0e-12
        )
        # These fixtures have no parameter-writing hooks, so the public raw
        # parameter mapping is unchanged. Only fresh state crosses this first
        # real compiled boundary before being consumed by the compiled scan.
        initialized = eager_initialized._replace(state=compiled_state)
        paths = _record_paths(functional)
        callbacks = functional.make_callbacks({"trace": dn.func.Recorder(tuple(paths))})
        kernel = functional.compile_rollout_chunk(
            STEPS,
            execution="scan",
            backend="inductor",
            fullgraph=True,
            options={"cpp_wrapper": True, "compile_threads": 1},
        )
        with torch_compiler_warning_context():
            _, auxiliary = dn.func.run(
                kernel.bind(
                    initialized.parameters,
                    functional.prepare(initialized.parameters, initialized.constants),
                ),
                initialized.state,
                steps=STEPS,
                inputs=inputs,
                callbacks=callbacks,
            )
        actual = auxiliary["callbacks"]["trace"]
        expected_records = {
            name: _read_path(expected, path) for name, path in paths.items()
        }
        _close_history(dict(actual), expected_records)
        # Read each recorded leaf back into the same objective tree. Control
        # carry is not an observable and contributes nothing to this loss.
        actual_history = {}
        for name, path in paths.items():
            cursor = actual_history
            for part in path[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[path[-1]] = actual[name]
        reference_gradients = torch.autograd.grad(
            _loss(expected), tuple(targets.values()), retain_graph=True
        )
        gradients = torch.autograd.grad(_loss(actual_history), tuple(targets.values()))
        for name, gradient, reference in zip(
            targets, gradients, reference_gradients, strict=True
        ):
            torch.testing.assert_close(
                gradient,
                reference,
                rtol=2.0e-8,
                atol=2.0e-10,
                msg=lambda message: f"{kind}/{name}: {message}",
            )
            assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
        assert kernel.execution_report().strategy == "compiled_scan"
    finally:
        torch.set_num_threads(previous)
