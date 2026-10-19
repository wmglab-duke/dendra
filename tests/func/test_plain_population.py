"""Functional acceptance of exact plain Population independent-point models.

Plain Population does not imply axial coupling: every slot follows the same
independent-point equations as the SingleCompartment convenience class. Keep
the CI recurrences short; real scan/while compilation is an orthogonal smoke
slice, not a combinatorial compiler matrix.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._population import _functional_topology_kind
from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mod import hh
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _module_snapshot,
)

DT = 0.01
STEPS = 4
CHANNELS = ("gnabar", "gkbar", "gl", "ena", "ek", "el")
RECORD = ("v", "hh.m", "hh.h", "hh.n", "t")
CM = "cm_param.rho"
AM1 = "integrator.mech.mechanisms.hh.DE.mhn.am1_param"

pytestmark = pytest.mark.cpu


@pytest.fixture(params=(1, 4), ids=("threads1", "threads4"))
def cpu_threads(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    try:
        yield request.param
    finally:
        torch.set_num_threads(previous)


def _model(
    *, model_class=dn.Population, device="cpu", dtype=torch.float64, batch_calls=()
):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DEVICE=device, DTYPE=dtype):
        model = model_class(
            N=2,
            C=3,
            celsius=6.3,
            cm=1.1,
            v_init=torch.tensor([-66.0, -61.0, -57.0], device=device, dtype=dtype),
            dtype=dtype,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.diam.copy_(
            torch.linspace(2.0, 4.0, 6, device=device, dtype=dtype).reshape(2, 3)
        )
        model.dx.copy_(
            torch.linspace(8.0, 13.0, 6, device=device, dtype=dtype).reshape(2, 3)
        )
        model.insert(hh)
        for size in batch_calls:
            model.batch(size)
        model.initialize()
        model.train()
    return model


def _clone(value):
    return pytree.tree_map(lambda tensor: tensor.detach().clone(), value)


def _drives(model, steps=STEPS):
    count = steps * model.v.numel()
    options = {"device": model.device(), "dtype": model.dtype()}
    ve = torch.linspace(-0.5, 0.8, count, **options).reshape(steps, *model.shape)
    intra = torch.linspace(-2.0e-7, 3.0e-7, count, **options).reshape(
        steps, *model.shape
    )
    return ve, intra


def _recorded(states):
    result = {
        "v": torch.stack([state["integrator"]["v"] for state in states]),
        "t": torch.stack([state["clock"]["t"] for state in states]),
    }
    for gate in ("m", "h", "n"):
        result["hh." + gate] = torch.stack(
            [state["mechanisms"]["hh"][gate] for state in states]
        )
    return result


def _live_state(model):
    return {
        "integrator": {"v": model.v},
        "clock": {"t": model.t},
        "mechanisms": {
            "hh": {gate: getattr(model.mech.hh, gate) for gate in ("m", "h", "n")}
        },
    }


def _trajectory(functional, parameters, constants, state, ve, intra):
    prepared = functional.prepare(parameters, constants)
    states = [state]
    for index in range(intra.shape[0]):
        state, _ = functional.step(
            parameters,
            prepared,
            state,
            dn.func.StepInput(
                ve=ve[index] if ve is not None else None, intra=intra[index]
            ),
        )
        states.append(state)
    return states


def _arguments(model, tensors, *, steps=STEPS):
    parameters, constants, state = map(
        _clone, (tensors.parameters, tensors.constants, tensors.state)
    )
    targets = {}
    for channel in CHANNELS:
        key = f"integrator.mech.mechanisms.hh.{channel}_param"
        targets[channel] = parameters[key].requires_grad_()
    targets["cm"] = parameters[CM].requires_grad_()
    for name in ("diam", "dx"):
        targets[name] = constants[name].requires_grad_()
    targets["initial_voltage"] = state["integrator"]["v"].requires_grad_()
    ve, drive = _drives(model, steps=steps)
    amplitude = model.v.new_ones((), requires_grad=True)
    targets["stimulus"] = amplitude
    return parameters, constants, state, ve, drive * amplitude, targets


def _loss(recorded):
    weights = torch.linspace(
        0.6,
        1.4,
        recorded["v"].shape[-1],
        device=recorded["v"].device,
        dtype=recorded["v"].dtype,
    )
    return (recorded["v"].square() * weights).mean() + 0.1 * recorded["hh.m"].mean()


def _gradients(recorded, targets):
    values = torch.autograd.grad(_loss(recorded), tuple(targets.values()))
    return dict(zip(targets, values, strict=True))


def _functional_result(model, *, compiled=False, options=None, steps=STEPS):
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters, constants, state, ve, intra, targets = _arguments(
        model, tensors, steps=steps
    )
    if compiled:
        kernel = functional.compile_rollout_chunk(
            steps, execution="scan", backend="inductor", fullgraph=True, options=options
        )
        callbacks = functional.make_callbacks({"trace": dn.func.Recorder(RECORD)})
        _, auxiliary = dn.func.run(
            kernel.bind(parameters, functional.prepare(parameters, constants)),
            state,
            inputs=dn.func.RolloutInput(ve=ve, intra=intra),
            steps=steps,
            callbacks=callbacks,
        )
        assert kernel.execution_report().strategy == "compiled_scan"
        recorded = dict(auxiliary["callbacks"]["trace"])
    else:
        recorded = _recorded(
            _trajectory(functional, parameters, constants, state, ve, intra)
        )
    return recorded, _gradients(recorded, targets)


def _imperative_result(model, *, steps=STEPS):
    parameters = dict(model.named_parameters())
    targets = {}
    for channel in CHANNELS:
        key = f"integrator.mech.mechanisms.hh.{channel}_param"
        targets[channel] = parameters[key].requires_grad_(True)
    targets["cm"] = parameters[CM].requires_grad_(True)
    for name in ("diam", "dx"):
        value = getattr(model, name).detach().clone().requires_grad_()
        setattr(model, name, value)
        targets[name] = value
    model.populate_parameter_buffers()
    model.v = model.v.detach().clone().requires_grad_()
    targets["initial_voltage"] = model.v
    for gate in ("m", "h", "n"):
        setattr(model.mech.hh, gate, getattr(model.mech.hh, gate).detach().clone())
    ve, drive = _drives(model, steps=steps)
    amplitude = model.v.new_ones((), requires_grad=True)
    targets["stimulus"] = amplitude
    intra = drive * amplitude
    timestep = torch.as_tensor(DT, device=model.device(), dtype=model.dtype())
    model.integrator._initialize(
        model, timestep, force=True, compile_scope="population"
    )
    states = [_live_state(model)]
    for index in range(steps):
        model.integrator.step(model, timestep, ve[index], intra[index])
        model.t = model.t + timestep
        states.append(_live_state(model))
    recorded = _recorded(states)
    return recorded, _gradients(recorded, targets)


def _close(actual, expected, *, dtype=torch.float64, cross_device=False):
    def canonical(value):
        if isinstance(value, dict):
            return {name: canonical(value[name]) for name in sorted(value)}
        if isinstance(value, list):
            return [canonical(item) for item in value]
        if isinstance(value, tuple):
            return tuple(canonical(item) for item in value)
        return value

    left, left_spec = pytree.tree_flatten_with_path(canonical(actual))
    right, right_spec = pytree.tree_flatten_with_path(canonical(expected))
    assert left_spec == right_spec
    for (path, value), (other_path, reference) in zip(left, right, strict=True):
        assert path == other_path
        assert value.shape == reference.shape
        assert torch.isfinite(value).all() and torch.isfinite(reference).all(), path
        if cross_device:
            value, reference = value.cpu().double(), reference.cpu().double()
        torch.testing.assert_close(
            value,
            reference,
            rtol=3.0e-5 if dtype == torch.float32 else 2.0e-11,
            atol=3.0e-6 if dtype == torch.float32 else 2.0e-11,
            msg=lambda message: f"{path}: {message}",
        )


def _close_gradients(actual, expected, *, dtype=torch.float64, cross_device=False):
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        reference = expected[name]
        assert value.shape == reference.shape
        assert torch.isfinite(value).all() and torch.isfinite(reference).all(), name
        if cross_device:
            value, reference = value.cpu(), reference.cpu()
        norm = torch.linalg.vector_norm(reference.double())
        assert norm > 0, name
        relative = torch.linalg.vector_norm(value.double() - reference.double()) / norm
        assert relative <= (2.0e-3 if dtype == torch.float32 else 2.0e-8), (
            name,
            relative.item(),
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_calls", [(), (2,), (2, 2)])
def test_plain_population_full_traces_and_all_eight_plus_physics_gradients(
    cpu_threads, dtype, batch_calls
):
    actual = _functional_result(_model(dtype=dtype, batch_calls=batch_calls))
    for model_class in (dn.Population, dn.SingleCompartment):
        reference = _model(
            model_class=model_class, dtype=dtype, batch_calls=batch_calls
        )
        expected = _imperative_result(reference)
        assert len(actual[1]) == 11
        _close(actual[0], expected[0], dtype=dtype)
        _close_gradients(actual[1], expected[1], dtype=dtype)


def test_plain_population_extracellular_drive_does_not_create_axial_coupling(
    cpu_threads,
):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model)
    expected = _trajectory(
        functional, tensors.parameters, tensors.constants, tensors.state, None, intra
    )
    actual = _trajectory(
        functional, tensors.parameters, tensors.constants, tensors.state, ve, intra
    )
    _close(actual, expected)
    localized = torch.zeros_like(intra)
    localized[..., 0, 1] = intra[..., 0, 1]
    perturbed = _trajectory(
        functional,
        tensors.parameters,
        tensors.constants,
        tensors.state,
        None,
        localized,
    )
    quiet = _trajectory(
        functional,
        tensors.parameters,
        tensors.constants,
        tensors.state,
        None,
        torch.zeros_like(intra),
    )
    difference = perturbed[-1]["integrator"]["v"] - quiet[-1]["integrator"]["v"]
    assert difference[0, 1] != 0
    difference = difference.clone()
    difference[0, 1] = 0
    assert torch.count_nonzero(difference) == 0


def test_plain_population_functional_work_does_not_mutate_source_and_commit_resumes(
    cpu_threads,
):
    model = _model(batch_calls=(2,))
    snapshot = _module_snapshot(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model)
    states = _trajectory(
        functional, tensors.parameters, tensors.constants, tensors.state, ve, intra
    )
    _assert_module_snapshot(model, snapshot)
    target = _model(batch_calls=(2,))
    functional.commit_state_(target, states[2])
    _close(functional.extract(target).state, states[2])
    timestep = target.v.new_tensor(DT)
    target.integrator._initialize(
        target, timestep, force=True, compile_scope="population"
    )
    for index in range(2, STEPS):
        target.integrator.step(target, timestep, ve[index], intra[index])
        target.t = target.t + timestep
    _close(functional.extract(target).state, states[-1])
    with pytest.raises(dn.func.FunctionalizationError):
        functional.commit_state_(
            _model(model_class=dn.SingleCompartment, batch_calls=(2,)), states[2]
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_calls", [(), (2,)])
def test_plain_population_fresh_initialization_matches_imperative_and_stays_pure(
    cpu_threads, dtype, batch_calls
):
    model = _model(dtype=dtype, batch_calls=batch_calls)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    voltage = (
        torch.linspace(
            -71.0, -53.0, model.v.numel(), device=model.device(), dtype=dtype
        )
        .reshape(model.shape)
        .requires_grad_()
    )
    parameters = _clone(tensors.parameters)
    am1 = parameters[AM1].requires_grad_()
    source_before = _module_snapshot(model)
    initialized = functional.initialize(
        parameters, tensors.constants, dn.func.InitializationInput(v_init=voltage)
    )
    _assert_module_snapshot(model, source_before)
    reference = _model(dtype=dtype, batch_calls=batch_calls)
    reference.set_v_init(voltage)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        reference.initialize()
    _close(initialized.state, functional.extract(reference).state, dtype=dtype)
    assert initialized.state["integrator"]["v"] is not voltage
    assert torch.count_nonzero(initialized.state["clock"]["t"]) == 0
    assert torch.count_nonzero(initialized.state["control"]["duration_remainder"]) == 0
    loss = sum(
        initialized.state["mechanisms"]["hh"][gate].square().mean()
        for gate in ("m", "h", "n")
    )
    gradients = torch.autograd.grad(loss, (voltage, am1))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient)
    # Imperative init_v intentionally detaches the initial voltage. The
    # established pure SingleCompartment initializer is the differentiable
    # cross-container oracle, while imperative initialization checks values.
    single = _model(
        model_class=dn.SingleCompartment, dtype=dtype, batch_calls=batch_calls
    )
    single_functional, single_tensors = dn.func.make_functional(single, dt=DT)
    single_parameters = _clone(single_tensors.parameters)
    single_am1 = single_parameters[AM1].requires_grad_()
    single_voltage = voltage.detach().clone().requires_grad_()
    single_initialized = single_functional.initialize(
        single_parameters,
        single_tensors.constants,
        dn.func.InitializationInput(v_init=single_voltage),
    )
    _close(initialized.state, single_initialized.state, dtype=dtype)
    single_loss = sum(
        single_initialized.state["mechanisms"]["hh"][gate].square().mean()
        for gate in ("m", "h", "n")
    )
    expected_gradients = torch.autograd.grad(single_loss, (single_voltage, single_am1))
    _close_gradients(
        dict(zip(("v_init", "am1"), gradients, strict=True)),
        dict(zip(("v_init", "am1"), expected_gradients, strict=True)),
        dtype=dtype,
    )


@pytest.mark.parametrize("kind", ["points", "mixed"])
def test_plain_population_stays_supported_inside_scalar_multipopulation(
    cpu_threads, kind
):
    if kind == "mixed" and not DENDRA_SOLVERS_AVAILABLE:
        pytest.skip(
            "mixed scalar MultiPopulation CPU execution requires dendra-solvers"
        )

    def build():
        with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64):
            plain = _model()
            # concat_models requires unbatched models; the packed owner carries
            # the explicit batch axis, not separately advancing children.
            other = (
                dn.SingleCompartment(N=1, C=2, celsius=6.3, dtype=torch.float64)
                if kind == "points"
                else dn.Unmyelinated(
                    [2.0], L=30.0, dx=10.0, celsius=6.3, dtype=torch.float64
                )
            )
            other.insert(hh)
            model = dn.concat_models(
                {"plain": plain, "other": other}, celsius=6.3, threads=2
            )
            model.batch(2)
            model.initialize()
            model.train()
        return model

    source, reference = build(), build()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    ve, intra = _drives(source)
    actual = _recorded(
        _trajectory(
            functional, tensors.parameters, tensors.constants, tensors.state, ve, intra
        )
    )
    timestep = reference.v.new_tensor(DT)
    reference.integrator._initialize(
        reference, timestep, force=True, compile_scope="population"
    )
    states = [_live_state(reference)]
    for index in range(STEPS):
        reference.integrator.step(reference, timestep, ve[index], intra[index])
        reference.t = reference.t + timestep
        states.append(_live_state(reference))
    _close(actual, _recorded(states))


class _CustomPopulation(dn.Population):
    pass


def test_plain_population_support_does_not_admit_custom_subclasses_or_generic_axon(
    cpu_threads,
):
    with pytest.raises(dn.func.FunctionalizationError):
        dn.func.make_functional(_model(model_class=_CustomPopulation), dt=DT)
    # Base Axon expects a family-owned _x() implementation. Classify its exact
    # type without constructing geometry or broadening the generic Axon API.
    assert _functional_topology_kind(object.__new__(dn.Axon)) is None


@pytest.mark.slow
def test_plain_population_cpp_scan_full_recording_and_all_gradients(cpu_threads):
    expected = _imperative_result(_model())
    with torch_compiler_warning_context():
        actual = _functional_result(
            _model(), compiled=True, options={"cpp_wrapper": True}
        )
    _close(actual[0], expected[0])
    _close_gradients(actual[1], expected[1])


@pytest.mark.slow
def test_plain_population_cpp_while_full_recording(cpu_threads):
    model = _model(batch_calls=(2,))
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model)
    with torch.no_grad():
        states = _trajectory(
            functional, tensors.parameters, tensors.constants, tensors.state, ve, intra
        )
    runner = dn.func.compile(
        model,
        dt=DT,
        steps=STEPS,
        execution="while",
        inference=True,
        callbacks={"trace": dn.func.Recorder(RECORD)},
    )
    with torch_compiler_warning_context():
        final, auxiliary = runner(inputs=dn.func.RolloutInput(ve=ve, intra=intra))
    assert runner.execution_report().strategy == "compiled_while_loop"
    assert runner.kernel._compile_options["options"]["cpp_wrapper"] is True
    _close(final, states[-1])
    _close(dict(auxiliary["callbacks"]["trace"]), _recorded(states))
