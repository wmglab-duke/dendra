"""Fresh initialization and its rollout gradients on a real CUDA device.

Run later with ``pytest tests/func/test_cuda_initialization_parity.py -m cuda``.
The ordinary matrix checks public imperative initialization and every subsequent
state leaf; orthogonal cases check a CPU float64 reference, finite differences,
and first-order compiled scan. No unrolled compiler matrix is needed.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_functional_initialization import (
    AM1,
    GNABAR,
    _assert_module_snapshot,
    _assert_tensor_snapshot,
    _module_snapshot,
    _tensor_snapshot,
)
from tests.func.test_initialization_parity import (
    DT,
    _imperative_initialized,
    _initialize,
    _inputs,
    _model,
    _voltage,
)

STEPS = 3
KINDS = ("myelinated", "block_path", "block_tree", "multi_scalar", "multi_block")
CHANNELS = ("gnabar", "gkbar", "gl", "ena", "ek", "el")
pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
    pytest.mark.filterwarnings(r"ignore:`torch\.jit\.script` is deprecated.*"),
    pytest.mark.filterwarnings(r"ignore:`torch\.jit\.script_method` is deprecated.*"),
]


def _clone(tree):
    return pytree.tree_map(lambda value: value.detach().clone(), tree)


def _conditioned_model(kind, *, device="cuda", dtype=torch.float32):
    # Finite but less stiff shell resistance gives float32 gradients a useful
    # cross-precision gate instead of testing near-singular common-mode solves.
    model = _model(
        kind, device=device, dtype=dtype, batch=(2,), shell_resistance_scale=100.0
    )
    model.train()
    with torch.no_grad():
        # Lower an already-used-looking source, not a construction-fresh zero
        # circuit. Functional fresh initialization must ignore resumed carry.
        model.v = model.v.detach().clone() + 1.5
        if hasattr(model, "vc"):
            inner = torch.full_like(model.v, 0.4)
            outer = torch.full_like(model.v, -0.25)
            model.vc = torch.stack((model.v + inner, inner, outer), dim=-1)
        model.t.fill_(3.0)
        model._duration_remainder.fill_(0.004)
    return model


def _arguments(model, tensors, kind):
    parameters, constants = _clone((tensors.parameters, tensors.constants))
    voltage = _voltage(model).detach().clone().requires_grad_()
    parameters[AM1] = (parameters[AM1] * 1.17).requires_grad_()
    parameters[GNABAR] = parameters[GNABAR] * 1.08
    targets = {
        name: parameters[f"integrator.mech.mechanisms.hh.{name}_param"].requires_grad_()
        for name in CHANNELS
    }
    if kind == "myelinated":
        geometry_group, geometry_name = parameters, "noded2"
    elif kind == "block_path":
        geometry_group, geometry_name = constants, "dx"
    elif kind == "block_tree":
        geometry_group = constants
        matches = [name for name in constants if name.endswith("canonical_area_cm2")]
        assert len(matches) == 1
        (geometry_name,) = matches
    else:
        # Packed canonical Tree geometry is immutable source metadata. Use the
        # packed path's genuine mutable length input for the independent public
        # imperative finite-difference oracle, not a prepared workspace leaf.
        component = "path" if kind == "multi_scalar" else "block_path"
        geometry_group, geometry_name = constants, f"populations.{component}.dx"
    if kind != "block_tree":
        geometry_group[geometry_name] = geometry_group[geometry_name] * 1.04
    geometry_group[geometry_name].requires_grad_()
    targets.update(
        {
            "initial_voltage": voltage,
            "initial_rate": parameters[AM1],
            "stimulus": model.v.new_tensor(1.0, requires_grad=True),
            "geometry": geometry_group[geometry_name],
        }
    )
    return parameters, constants, voltage, targets, geometry_name


def _trajectory(
    functional, model, tensors, parameters, constants, voltage, amplitude=None
):
    initialized = _initialize(
        functional, tensors, voltage, parameters=parameters, constants=constants
    )
    prepared = functional.prepare(initialized.parameters, initialized.constants)
    state = initialized.state
    states = [state]
    inputs = _inputs(model, steps=STEPS)
    intra = inputs.intra if amplitude is None else inputs.intra * amplitude
    for bath, current in zip(inputs.ve, intra, strict=True):
        state, _ = functional.step(
            initialized.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=bath, intra=current),
        )
        states.append(state)
    return states


def _imperative_trajectory(
    functional, model, voltage, parameters, constants, amplitude=None
):
    # Canonical Tree tensors are reconstructed from immutable source graph
    # metadata. They are unchanged in these public imperative oracle cases.
    imperative_constants = {
        name: value
        for name, value in constants.items()
        if not name.rsplit(".", 1)[-1].startswith("canonical_")
    }
    reference, initial = _imperative_initialized(
        functional,
        model,
        voltage,
        parameters=parameters,
        constants=imperative_constants,
    )
    states = [initial]
    dt = reference.v.new_tensor(DT)
    reference.integrator._initialize(
        reference, dt, force=True, compile_scope="population"
    )
    inputs = _inputs(reference, steps=STEPS)
    intra = inputs.intra if amplitude is None else inputs.intra * amplitude
    for bath, current in zip(inputs.ve, intra, strict=True):
        reference.integrator.step(reference, dt, bath, current)
        reference.t = reference.t + dt
        states.append(functional.extract(reference).state)
    return states


def _assert_close(actual, expected, *, dtype, cross_device=False):
    left, left_spec = pytree.tree_flatten_with_path(actual)
    right, right_spec = pytree.tree_flatten_with_path(expected)
    assert left_spec == right_spec
    for (path, value), (other_path, reference) in zip(left, right, strict=True):
        assert path == other_path
        assert value.shape == reference.shape
        assert torch.isfinite(value).all() and torch.isfinite(reference).all()
        if cross_device:
            value, reference = (
                value.to(device="cpu", dtype=torch.float64),
                reference.to(device="cpu", dtype=torch.float64),
            )
        if any(getattr(key, "key", None) == "i_membrane" for key in path):
            value, reference = value * 1.0e6, reference * 1.0e6
        torch.testing.assert_close(
            value,
            reference,
            rtol=5e-4 if dtype == torch.float32 else 2e-9,
            atol=2e-5 if dtype == torch.float32 else 2e-9,
            msg=lambda message: f"{path}: {message}",
        )


def _objective(states):
    voltage = torch.stack([state["integrator"]["v"] for state in states])
    weights = torch.linspace(
        0.6, 1.4, voltage.shape[-1], device=voltage.device, dtype=voltage.dtype
    )
    value = 0.001 * (weights * voltage.square()).mean()
    for state in states:
        # The shared fixture includes a non-HH regional INITIAL mechanism with
        # ion/diameter dependence, so this also exercises generic state setup.
        for leaf in pytree.tree_leaves(state["mechanisms"]):
            value = value + 0.01 * leaf.square().mean() / len(states)
        for leaf in pytree.tree_leaves(state.get("mechanism_buffers", {})):
            value = value + 0.01 * leaf.square().mean() / len(states)
        if "vc" in state["integrator"]:
            value = value + 0.0003 * state["integrator"]["vc"].square().mean()
        if "i_membrane" in state["integrator"]:
            current = state["integrator"]["i_membrane"] * 1.0e6
            value = value + 0.01 * current.square().mean()
    return value


def _gradients(states, targets):
    values = torch.autograd.grad(_objective(states), tuple(targets.values()))
    for name, value in zip(targets, values, strict=True):
        assert torch.isfinite(value).all(), name
        assert torch.count_nonzero(value), name
    return dict(zip(targets, values, strict=True))


def _value_at(state, path):
    for key in path:
        state = state[key]
    return state


def _recording_fields(functional):
    fields = {}
    for leaf in functional._state_layout:
        if leaf.public_path[0] == "integrator":
            name = leaf.public_path[-1]
        elif leaf.public_path == ("clock", "t"):
            name = "t"
        else:
            name = leaf.checkpoint_key
        if name is not None:
            assert name not in fields
            fields[name] = leaf.public_path
        else:
            assert leaf.public_path == ("control", "duration_remainder")
    return fields


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_cuda_fresh_initialization_matches_imperative_full_state_traces(kind, dtype):
    model = _conditioned_model(kind, dtype=dtype)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters, constants, voltage, targets, _ = _arguments(model, tensors, kind)
    before = _module_snapshot(model)
    inputs = [*parameters.values(), *constants.values(), voltage]
    snapshots = [_tensor_snapshot(value) for value in inputs]
    actual = _trajectory(
        functional, model, tensors, parameters, constants, voltage, targets["stimulus"]
    )
    expected = _imperative_trajectory(
        functional, model, voltage, parameters, constants, targets["stimulus"]
    )
    _assert_close(actual, expected, dtype=dtype)
    torch.testing.assert_close(actual[0]["integrator"]["v"], voltage, rtol=0, atol=0)
    if "vc" in actual[0]["integrator"]:
        circuit = actual[0]["integrator"]["vc"]
        torch.testing.assert_close(circuit[..., 0], voltage, rtol=0, atol=0)
        assert torch.count_nonzero(circuit[..., 1:]) == 0
    assert torch.count_nonzero(actual[0]["integrator"]["i_membrane"]) == 0
    for state in actual:
        for leaf in functional._state_layout:
            value = state
            for key in leaf.public_path:
                value = value[key]
            # In particular, duration_remainder is intentionally host float64,
            # not a physical CUDA/model-dtype state variable.
            assert value.device == leaf.device
            assert value.dtype == leaf.dtype
    assert not torch.equal(
        actual[0]["mechanisms"]["hh"]["m"], tensors.state["mechanisms"]["hh"]["m"]
    )
    _assert_module_snapshot(model, before)
    for value, snapshot in zip(inputs, snapshots, strict=True):
        _assert_tensor_snapshot(value, snapshot)


@pytest.mark.parametrize("kind", ("myelinated", "multi_block"))
def test_cuda_fresh_rollout_gradients_match_float64_cpu(kind):
    pytest.importorskip("dendra_solvers")
    results = []
    for device, dtype in (("cpu", torch.float64), ("cuda", torch.float32)):
        model = _conditioned_model(kind, device=device, dtype=dtype)
        functional, tensors = dn.func.make_functional(model, dt=DT)
        parameters, constants, voltage, targets, _ = _arguments(model, tensors, kind)
        states = _trajectory(
            functional,
            model,
            tensors,
            parameters,
            constants,
            voltage,
            targets["stimulus"],
        )
        results.append((states, _gradients(states, targets)))
    expected, actual = results
    _assert_close(actual[0], expected[0], dtype=torch.float32, cross_device=True)
    for name, value in actual[1].items():
        reference = expected[1][name]
        torch.testing.assert_close(
            value.to(device="cpu", dtype=torch.float64),
            reference,
            rtol=5e-3,
            atol=2e-6,
            msg=lambda message: f"{name}: {message}",
        )


@pytest.mark.parametrize("kind", ("myelinated", "multi_block"))
def test_cuda_fresh_rollout_gradient_matches_public_imperative_finite_difference(kind):
    model = _conditioned_model(kind, dtype=torch.float64)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters, constants, voltage, targets, geometry_name = _arguments(
        model, tensors, kind
    )
    states = _trajectory(
        functional, model, tensors, parameters, constants, voltage, targets["stimulus"]
    )
    gradients = _gradients(states, targets)
    epsilon = 1e-4
    for name, leaf in targets.items():
        # A relative direction preserves zero-area junctions and positivity of
        # geometry while independently perturbing each user-facing input.
        direction = leaf.detach()
        expected = (gradients[name] * direction).sum()
        losses = []
        for sign in (-1, 1):
            p, c = _clone((parameters, constants))
            v = voltage.detach()
            amplitude = targets["stimulus"].detach()
            changed = leaf.detach() + sign * epsilon * direction
            if name == "initial_voltage":
                v = changed
            elif name == "initial_rate":
                p[AM1] = changed
            elif name in CHANNELS:
                p[f"integrator.mech.mechanisms.hh.{name}_param"] = changed
            elif name == "stimulus":
                amplitude = changed
            elif geometry_name in p:
                p[geometry_name] = changed
            else:
                c[geometry_name] = changed
            reference = _imperative_trajectory(functional, model, v, p, c, amplitude)
            losses.append(_objective(reference))
        actual = (losses[1] - losses[0]) / (2 * epsilon)
        assert torch.isfinite(actual) and torch.count_nonzero(actual)
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=2e-7)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ("multi_scalar", "multi_block"))
def test_cuda_fresh_initialization_composes_with_compiled_scan_first_order(kind):
    model = _conditioned_model(kind)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters, constants, voltage, targets, _ = _arguments(model, tensors, kind)
    expected = _trajectory(
        functional, model, tensors, parameters, constants, voltage, targets["stimulus"]
    )
    expected_gradients = _gradients(expected, targets)
    parameters, constants, voltage, targets, _ = _arguments(model, tensors, kind)

    def initialize(v, p, c):
        return _initialize(functional, tensors, v, parameters=p, constants=c).state

    compiled_initialize = torch.compile(
        initialize, backend="inductor", fullgraph=True, dynamic=False
    )
    with torch_compiler_warning_context():
        state = compiled_initialize(voltage, parameters, constants)
        prepared = functional.prepare(parameters, constants)
        chunk = functional.compile_rollout_chunk(
            STEPS,
            execution="scan",
            backend="inductor",
            fullgraph=True,
            options={"cpp_wrapper": False, "compile_threads": 1},
        )
        fields = _recording_fields(functional)
        callbacks = functional.make_callbacks(
            {"trace": dn.func.Recorder(tuple(fields))}
        )
        inputs = _inputs(model, steps=STEPS)
        final, auxiliary = dn.func.run(
            chunk.bind(parameters, prepared),
            state,
            steps=STEPS,
            inputs=inputs._replace(intra=inputs.intra * targets["stimulus"]),
            callbacks=callbacks,
        )
        record = auxiliary["callbacks"]["trace"]
        expected_record = {
            name: torch.stack([_value_at(step, path) for step in expected])
            for name, path in fields.items()
        }
        _assert_close(dict(record), expected_record, dtype=torch.float32)
        actual = []
        for index in range(STEPS + 1):
            # Reconstruct only from the independently recorded scan outputs;
            # do not smuggle unrecorded expected Ion/CARRY values into the AD
            # objective or into the intermediate history comparisons.
            step = pytree.tree_map(lambda value: None, state)
            step["control"]["duration_remainder"] = state["control"][
                "duration_remainder"
            ]
            for name, path in fields.items():
                cursor = step
                for key in path[:-1]:
                    cursor = cursor.setdefault(key, {})
                cursor[path[-1]] = record[name][index]
            actual.append(step)
        gradients = _gradients(actual, targets)
    _assert_close(state, expected[0], dtype=torch.float32)
    _assert_close(final, expected[-1], dtype=torch.float32)
    _assert_close(actual, expected, dtype=torch.float32)
    _assert_close(gradients, expected_gradients, dtype=torch.float32)
    report = chunk.execution_report()
    assert report.strategy == "compiled_scan"
    assert report.grad_enabled
