"""CUDA functional Population acceptance contracts.

These tests are intentionally skipped on hosts without CUDA. They are
correctness tests, not benchmarks. The ordinary lane checks tiny recurrences,
all public carry leaves, reverse AD, explicit batches, and torch.func. The
``slow`` lane additionally builds real Inductor scan/while executors.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import hh, pas
from tests.func.test_extcell_axon import _model as _block_cable
from tests.func.test_extcell_tree import _model as _block_tree
from tests.func.test_multi_population import _mixed_multi
from tests.func.test_scalar_tree import _model as _scalar_tree
from tests.func.test_single_compartment import _model as _single_compartment

DT = 0.01
STEPS = 4
CHANNELS = ("gnabar", "gkbar", "gl", "ena", "ek", "el")
KINDS = (
    "sc",
    "ub_thomas",
    "ub_pcr",
    "myelinated",
    "tree",
    "ext_axon",
    "ext_tree",
    "multi",
)
pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


def _clone(value):
    return pytree.tree_map(lambda tensor: tensor.detach().clone(), value)


def _seed_circuit_state(model):
    if not hasattr(model, "vc"):
        return
    inner = torch.linspace(
        -1.0, 0.5, model.v.numel(), device=model.device(), dtype=model.dtype()
    ).reshape(model.shape)
    outer = -0.5 * inner
    model.vc = torch.stack((model.v + inner, inner, outer), dim=-1)


def _model(kind, *, device="cuda", dtype=torch.float64, batch_calls=()):
    # Reuse the independently tested CPU fixtures where possible. The outer
    # context determines device; their local contexts do not override it.
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DEVICE=device, DTYPE=dtype):
        if kind == "sc":
            model = _single_compartment(dtype=dtype, batch_calls=batch_calls)
        elif kind.startswith("ub_"):
            # Ten-micron segments keep this float32 geometry-gradient oracle
            # well conditioned (axial matrix condition about 52--65). The old
            # one-micron fixture reached 5,100--6,400: raw/facade coefficient
            # gradients were bitwise equal, but geometry-chain cancellation
            # obscured that agreement. Keep the same five compartments, all
            # gradient targets, and strict tolerances rather than loosen them.
            model = dn.Unmyelinated(
                [2.0, 2.5],
                L=40.0,
                dx=10.0,
                v_init=torch.tensor(
                    [-64.0, -60.0, -56.0, -59.0, -63.0],
                    device=device,
                    dtype=dtype,
                ),
                dtype=dtype,
                integrator=dn.bwd_euler_ub(method=kind.removeprefix("ub_"), imem=False),
            )
            model.insert(hh)
            for size in batch_calls:
                model.batch(size)
        elif kind == "myelinated":
            model = dn.Myelinated(
                [8.0, 12.0],
                n_node=3,
                celsius=6.3,
                v_init=-65.0,
                dtype=dtype,
                integrator=dn.bwd_euler_ub(method="pcr", imem=False),
            )
            model.insert(hh)
            for size in batch_calls:
                model.batch(size)
        elif kind == "tree":
            model = _scalar_tree(batch_calls=batch_calls)
        elif kind == "ext_axon":
            model = _block_cable(batch_calls=batch_calls)
        elif kind == "ext_tree":
            model = _block_tree(batch_calls=batch_calls)
        elif kind == "multi":
            model = _mixed_multi(batch_calls=batch_calls)
        else:  # pragma: no cover - test parameter contract
            raise AssertionError(kind)
        if kind in {"tree", "ext_axon", "ext_tree"}:
            model.delete(pas)
            model.insert(hh)
        model.to(device=device, dtype=dtype)
        model.initialize()
        model.train()
        _seed_circuit_state(model)
        assert model.integrator.imem is False
        assert model.v.device.type == torch.device(device).type
        assert all(
            parameter.device == model.device() for parameter in model.parameters()
        )
        if hasattr(model, "vc"):
            assert model.vc.shape == (*model.shape, 3)
    return model


def _drives(model, steps=STEPS, *, pulse=False):
    count = steps * model.v.numel()
    ve = torch.linspace(
        -1.5, 2.0, count, device=model.device(), dtype=model.dtype()
    ).reshape(steps, *model.shape)
    intra = torch.linspace(
        -1.0e-9, 2.0e-9, count, device=model.device(), dtype=model.dtype()
    ).reshape(steps, *model.shape)
    if pulse:
        intra = torch.zeros_like(intra)
        intra[..., 0] = 0.7 * dn.units.nA
    return ve, intra


def _close(actual, expected, *, dtype, cross_device=False):
    def canonical(value):
        if isinstance(value, dict):
            return {name: canonical(value[name]) for name in sorted(value)}
        if isinstance(value, list):
            return [canonical(item) for item in value]
        if isinstance(value, tuple):
            return tuple(canonical(item) for item in value)
        return value

    # Mapping insertion order is not a recording/state contract. Preserve
    # complete path/key coverage, but compare equal mappings in canonical order.
    actual, expected = canonical(actual), canonical(expected)
    left, left_spec = pytree.tree_flatten_with_path(actual)
    right, right_spec = pytree.tree_flatten_with_path(expected)
    assert left_spec == right_spec
    rtol = 5.0e-4 if dtype == torch.float32 else 3.0e-8
    atol = 3.0e-4 if dtype == torch.float32 else 3.0e-9
    for (path, value), (other_path, reference) in zip(left, right, strict=True):
        assert path == other_path
        assert value.shape == reference.shape and value.dtype == reference.dtype
        assert torch.isfinite(value).all() and torch.isfinite(reference).all()
        if cross_device:
            value, reference = value.cpu(), reference.cpu()
        else:
            assert value.device == reference.device
        clock = bool(path) and getattr(path[-1], "key", None) in {
            "t",
            "duration_remainder",
        }
        torch.testing.assert_close(
            value,
            reference,
            rtol=0 if clock else rtol,
            atol=0 if clock else atol,
            msg=lambda message: f"state leaf {path}: {message}",
        )


def _close_gradients(actual, expected, *, dtype, cross_device=False):
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        reference = expected[name]
        assert value.shape == reference.shape and value.dtype == reference.dtype
        assert torch.isfinite(value).all() and torch.isfinite(reference).all(), name
        if cross_device:
            value, reference = value.cpu(), reference.cpu()
        # Relative-norm accuracy is meaningful for very small current/geometry
        # derivatives; a large absolute tolerance would hide their loss.
        error = torch.linalg.vector_norm((value - reference).double())
        norm = torch.linalg.vector_norm(reference.double())
        assert norm > 0, name
        relative = error / norm.clamp_min(1.0e-12)
        assert relative <= (2.0e-3 if dtype == torch.float32 else 2.0e-7), (
            name,
            relative.item(),
        )


def _channel_names(parameters):
    names = []
    for channel in CHANNELS:
        # MultiPopulation also owns non-authoritative child mechanism modules.
        # The packed integrator's HH is the numerical channel owner.
        name = f"integrator.mech.mechanisms.hh.{channel}_param"
        assert name in parameters, name
        names.append(name)
    return tuple(names)


def _geometry(kind, model):
    """Return one differentiable physical-geometry source and its public key."""
    if kind in {"tree", "ext_tree"}:
        return "parameter", "area_scale_param.rho", None, None
    if kind == "multi":
        return "constant", "populations.cable.dx", model.populations["cable"], "dx"
    return "constant", "dx", model, "dx"


def _voltage_state(state, voltage):
    integrator = {**state["integrator"], "v": voltage}
    if "vc" in integrator:
        shell = integrator["vc"][..., 1:]
        # Membrane v is redundant in a block model. Connect the genuine circuit
        # carry to the V0 leaf instead of accidentally testing a zero gradient.
        integrator["vc"] = torch.cat(
            (voltage.unsqueeze(-1) + shell[..., :1], shell), dim=-1
        )
    return {**state, "integrator": integrator}


def _functional_arguments(kind, model, tensors, *, steps=STEPS, pulse=False):
    parameters, constants, state = (
        _clone(tensors.parameters),
        _clone(tensors.constants),
        _clone(tensors.state),
    )
    targets = {}
    for channel, name in zip(CHANNELS, _channel_names(parameters), strict=True):
        parameters[name].requires_grad_()
        targets[channel] = parameters[name]
    category, name, _, _ = _geometry(kind, model)
    geometry = parameters[name] if category == "parameter" else constants[name]
    geometry.requires_grad_()
    targets["geometry"] = geometry
    initial_voltage = state["integrator"]["v"].requires_grad_()
    state = _voltage_state(state, initial_voltage)
    targets["initial_voltage"] = initial_voltage
    ve, drive = _drives(model, steps, pulse=pulse)
    amplitude = torch.ones(
        (), dtype=model.dtype(), device=model.device(), requires_grad=True
    )
    targets["stimulus"] = amplitude
    return parameters, constants, state, ve, drive * amplitude, targets


def _functional_trajectory(functional, parameters, constants, state, ve, intra):
    prepared = functional.prepare(parameters, constants)
    states = [state]
    for index in range(ve.shape[0]):
        state, _ = functional.step(
            parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        states.append(state)
    return states


def _loss(states):
    voltage = torch.stack([state["integrator"]["v"] for state in states])
    return voltage.square().mean()


def _gradients(loss, targets):
    values = torch.autograd.grad(loss, tuple(targets.values()))
    return dict(zip(targets, values, strict=True))


def _functional_result(kind, model, *, steps=STEPS, pulse=False, dt=DT):
    functional, tensors = dn.func.make_functional(model, dt=dt)
    assert tensors.state["integrator"]["v"].device == model.device()
    assert tensors.state["control"]["duration_remainder"].device.type == "cpu"
    assert tensors.state["control"]["duration_remainder"].dtype == torch.float64
    for gate in ("m", "h", "n"):
        assert tensors.state["mechanisms"]["hh"][gate].device == model.device()
    parameters, constants, state, ve, intra, targets = _functional_arguments(
        kind, model, tensors, steps=steps, pulse=pulse
    )
    states = _functional_trajectory(functional, parameters, constants, state, ve, intra)
    return functional, tensors, states, _gradients(_loss(states), targets)


def _imperative_result(kind, model, functional, *, steps=STEPS, pulse=False, dt=DT):
    targets = {}
    parameters = dict(model.named_parameters())
    for channel, name in zip(CHANNELS, _channel_names(parameters), strict=True):
        targets[channel] = parameters[name]
        targets[channel].requires_grad_(True)
    category, name, owner, attribute = _geometry(kind, model)
    if category == "parameter":
        geometry = parameters[name]
        geometry.requires_grad_(True)
    else:
        geometry = getattr(owner, attribute).detach().clone().requires_grad_(True)
        setattr(owner, attribute, geometry)
    # Functional prepare always rederives effective physical fields from the
    # explicit geometry. Do the same for the imperative oracle after replacing
    # a leaf: Myelinated.rhoa depends on dx, and MultiPopulation additionally
    # owns packed copies of its components' physical planes. Merely rebuilding
    # the integrator would leave those preexisting fields' derivatives stale.
    model.populate_parameter_buffers()
    targets["geometry"] = geometry
    voltage = model.v.detach().clone().requires_grad_(True)
    model.v = voltage
    if hasattr(model, "vc"):
        shells = model.vc[..., 1:].detach().clone()
        model.vc = torch.cat((voltage.unsqueeze(-1) + shells[..., :1], shells), dim=-1)
    for gate in ("m", "h", "n"):
        setattr(model.mech.hh, gate, getattr(model.mech.hh, gate).detach())
    targets["initial_voltage"] = voltage
    ve, drive = _drives(model, steps, pulse=pulse)
    amplitude = torch.ones(
        (), dtype=model.dtype(), device=model.device(), requires_grad=True
    )
    targets["stimulus"] = amplitude
    intra = drive * amplitude
    timestep = torch.as_tensor(dt, dtype=model.dtype(), device=model.device())
    model.integrator._initialize(
        model, timestep, force=True, compile_scope="population"
    )
    states = [functional.extract(model).state]
    voltage_trace = [model.v]
    for index in range(steps):
        model.integrator.step(model, timestep, ve[index], intra[index])
        model.t = model.t + timestep
        states.append(functional.extract(model).state)
        voltage_trace.append(model.v)
    loss = torch.stack(voltage_trace).square().mean()
    return states, _gradients(loss, targets)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_full_trajectory_and_all_source_gradients_match_imperative(kind, dtype):
    source, reference = _model(kind, dtype=dtype), _model(kind, dtype=dtype)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cuda", DTYPE=dtype):
        functional, _, states, gradients = _functional_result(kind, source)
        expected_states, expected_gradients = _imperative_result(
            kind, reference, functional
        )
    assert len(states) == STEPS + 1
    assert len(gradients) == 9
    _close(states, expected_states, dtype=dtype)
    _close_gradients(gradients, expected_gradients, dtype=dtype)


@pytest.mark.parametrize("kind", KINDS)
def test_cuda_explicit_population_batch_preserves_every_carry_leaf(kind):
    source = _model(kind, batch_calls=(2,))
    reference = _model(kind, batch_calls=(2,))
    assert source.shape == (2, *source.core_shape())
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cuda", DTYPE=torch.float64):
        functional, _, states, gradients = _functional_result(kind, source)
        expected, expected_gradients = _imperative_result(kind, reference, functional)
    _close(states, expected, dtype=torch.float64)
    _close_gradients(gradients, expected_gradients, dtype=torch.float64)


@pytest.mark.parametrize("kind", KINDS)
def test_cuda_and_cpu_oracles_match_states_and_normalized_gradients(kind):
    if kind in {"tree", "ext_axon", "ext_tree", "multi", "ub_thomas"}:
        pytest.importorskip("dendra_solvers")
    cpu = _model(kind, device="cpu", dtype=torch.float64)
    gpu = _model(kind, device="cuda", dtype=torch.float64)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64):
        _, _, expected, expected_gradients = _functional_result(kind, cpu)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cuda", DTYPE=torch.float64):
        _, _, actual, gradients = _functional_result(kind, gpu)
    _close(actual, expected, dtype=torch.float64, cross_device=True)
    _close_gradients(
        gradients, expected_gradients, dtype=torch.float64, cross_device=True
    )


@pytest.mark.parametrize("kind", KINDS)
def test_cuda_torch_func_grad_vmap_empty_vmap_and_jvp(kind):
    model = _model(kind, batch_calls=(2,))
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, steps=1)
    channel = _channel_names(tensors.parameters)[0]
    category, geometry_name, _, _ = _geometry(kind, model)
    geometry = (
        tensors.parameters[geometry_name]
        if category == "parameter"
        else tensors.constants[geometry_name]
    )

    def response(gnabar, local_geometry):
        parameters = {**tensors.parameters, channel: gnabar}
        constants = dict(tensors.constants)
        if category == "parameter":
            parameters[geometry_name] = local_geometry
        else:
            constants[geometry_name] = local_geometry
        state, _ = functional.prepare_and_step(
            parameters,
            constants,
            tensors.state,
            dn.func.StepInput(ve=ve[0], intra=intra[0]),
        )
        return state["integrator"]["v"]

    gnabar = tensors.parameters[channel]
    arguments = (gnabar, geometry)
    tangents = tuple(torch.ones_like(value) * 0.01 for value in arguments)
    with torch_compiler_warning_context():
        output, tangent = torch.func.jvp(response, arguments, tangents)
        reverse_output, vjp = torch.func.vjp(response, *arguments)
    _close(output, reverse_output, dtype=model.dtype())
    weights = torch.linspace(
        0.2, 1.1, output.numel(), dtype=output.dtype, device=output.device
    ).reshape_as(output)
    gradients = vjp(weights)
    forward_pairing = (tangent * weights).sum()
    reverse_pairing = sum(
        (gradient * direction).sum()
        for gradient, direction in zip(gradients, tangents, strict=True)
    )
    torch.testing.assert_close(
        forward_pairing, reverse_pairing, rtol=3.0e-7, atol=3.0e-9
    )
    assert torch.isfinite(tangent).all() and torch.count_nonzero(tangent)

    def loss(local_g):
        return response(local_g, geometry).square().mean()

    ordinary = gnabar.detach().clone().requires_grad_(True)
    expected_gradient = torch.autograd.grad(loss(ordinary), ordinary)[0]
    transformed_gradient = torch.func.grad(loss)(gnabar)
    _close_gradients(
        {"gnabar": transformed_gradient},
        {"gnabar": expected_gradient},
        dtype=model.dtype(),
    )
    lanes = torch.stack((gnabar * 0.9, gnabar, gnabar * 1.1))
    actual = torch.vmap(response, in_dims=(0, None))(lanes, geometry)
    expected = torch.stack([response(lane, geometry) for lane in lanes])
    _close(actual, expected, dtype=model.dtype())
    empty = torch.vmap(response, in_dims=(0, None))(lanes[:0], geometry)
    assert empty.shape == (0, *model.shape)
    assert empty.dtype == model.dtype() and empty.device == model.device()


def _record_fields(state):
    fields = ["v", "hh.m", "hh.h", "hh.n", "t"]
    if "vc" in state["integrator"]:
        fields.append("vc")
    return tuple(fields)


def _recorded_states(states):
    recorded = {
        "v": torch.stack([state["integrator"]["v"] for state in states]),
        "t": torch.stack([state["clock"]["t"] for state in states]),
    }
    for gate in ("m", "h", "n"):
        recorded["hh." + gate] = torch.stack(
            [state["mechanisms"]["hh"][gate] for state in states]
        )
    if "vc" in states[0]["integrator"]:
        recorded["vc"] = torch.stack([state["integrator"]["vc"] for state in states])
    return recorded


@pytest.mark.slow
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("execution", ["scan", "while"])
def test_cuda_compiled_inference_loops_preserve_full_recordings(kind, execution):
    model = _model(kind, dtype=torch.float32)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    fields = _record_fields(tensors.state)
    ve, intra = _drives(model)
    with torch.no_grad():
        expected = _functional_trajectory(
            functional, tensors.parameters, tensors.constants, tensors.state, ve, intra
        )
    runner = dn.func.compile(
        model,
        dt=DT,
        steps=STEPS,
        execution=execution,
        inference=True,
        callbacks={"trace": dn.func.Recorder(fields)},
        backend="inductor",
    )
    options = runner.kernel._compile_options.get("options", {})
    assert not options.get("cpp_wrapper", False)
    with torch_compiler_warning_context():
        state, auxiliary = runner(inputs=dn.func.RolloutInput(ve=ve, intra=intra))
    _close(state, expected[-1], dtype=model.dtype())
    _close(
        dict(auxiliary["callbacks"]["trace"]),
        _recorded_states(expected),
        dtype=model.dtype(),
    )
    assert runner.execution_report().strategy == (
        "compiled_scan" if execution == "scan" else "compiled_while_loop"
    )


@pytest.mark.slow
@pytest.mark.parametrize("kind", KINDS)
def test_cuda_compiled_scan_backward_preserves_all_nine_gradient_targets(kind):
    model = _model(kind, dtype=torch.float32)
    functional, tensors = dn.func.make_functional(model, dt=DT)

    def evaluate(compiled):
        parameters, constants, state, ve, intra, targets = _functional_arguments(
            kind, model, tensors
        )
        if not compiled:
            states = _functional_trajectory(
                functional, parameters, constants, state, ve, intra
            )
            return (
                states[-1],
                _recorded_states(states),
                _gradients(_loss(states), targets),
            )
        prepared = functional.prepare(parameters, constants)
        kernel = functional.compile_rollout_chunk(
            STEPS, execution="scan", backend="inductor", fullgraph=True
        )
        callbacks = functional.make_callbacks(
            {"trace": dn.func.Recorder(_record_fields(state))}
        )
        final, auxiliary = dn.func.run(
            kernel.bind(parameters, prepared),
            state,
            inputs=dn.func.RolloutInput(ve=ve, intra=intra),
            steps=STEPS,
            callbacks=callbacks,
        )
        recorded = dict(auxiliary["callbacks"]["trace"])
        gradients = _gradients(recorded["v"].square().mean(), targets)
        assert kernel.execution_report().strategy == "compiled_scan"
        return final, recorded, gradients

    expected = evaluate(False)
    with torch_compiler_warning_context():
        actual = evaluate(True)
    _close(actual[:2], expected[:2], dtype=model.dtype())
    _close_gradients(actual[2], expected[2], dtype=model.dtype())


def test_cuda_strong_hh_recurrence_preserves_full_traces_and_gradients():
    source = _model("ub_pcr", dtype=torch.float32)
    reference = _model("ub_pcr", dtype=torch.float32)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cuda", DTYPE=torch.float32):
        functional, _, states, gradients = _functional_result(
            "ub_pcr", source, steps=40, pulse=True, dt=0.025
        )
        expected, expected_gradients = _imperative_result(
            "ub_pcr", reference, functional, steps=40, pulse=True, dt=0.025
        )
    _close(states, expected, dtype=source.dtype())
    _close_gradients(gradients, expected_gradients, dtype=source.dtype())
    assert (
        states[-1]["mechanisms"]["hh"]["m"] - states[0]["mechanisms"]["hh"]["m"]
    ).abs().amax() > 1.0e-3


@pytest.mark.parametrize(
    "model_name, diameter",
    [("bigMRG", 8.0), ("exactMRG", 5.7), ("smolMRG", 2.0)],
)
def test_optional_real_mrg_models_preserve_every_gpu_carry_leaf(model_name, diameter):
    peripheral = pytest.importorskip("dendra_models.models.cells.peripheral")

    def model():
        with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DEVICE="cuda"):
            result = getattr(peripheral, model_name)([diameter], n_node=2)
            result.to(device="cuda", dtype=torch.float64)
            result.initialize()
            result.train()
        return result

    source, reference = model(), model()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    assert "vc" in tensors.state["integrator"]
    assert tensors.state["integrator"]["vc"].shape == (*source.shape, 3)
    ve, intra = _drives(source)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    timestep = torch.as_tensor(DT, device="cuda", dtype=torch.float64)
    reference.integrator._initialize(reference, timestep, force=True)
    state = tensors.state
    for index in range(STEPS):
        state, _ = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        reference.integrator.step(reference, timestep, ve[index], intra[index])
        reference.t = reference.t + timestep
        _close(state, functional.extract(reference).state, dtype=torch.float64)
