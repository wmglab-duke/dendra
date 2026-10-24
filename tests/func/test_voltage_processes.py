"""Pure reset/refractory dynamics retain full imperative traces and gradients.

The point model additionally has an independent passive-membrane recurrence
and the original hard/STE fire equations as an oracle. Keep the compiler matrix
small: broad coverage uses the eager scan backend; only two representative
graphs exercise native C++ scan/checkpointing and inference while-loop.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import fire_d, pas
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _assert_tree_close,
    _module_snapshot,
)
from tests.func.test_initialization_parity import _live_state, _read_path, _record_paths
from tests.func.test_membrane_current import _model as _membrane_model
from tests.test_voltage_process_pure import DT, FIRE_TYPES, STEPS, _reference_fire

pytestmark = pytest.mark.cpu
KINDS = (
    "point",
    "path",
    "myelinated",
    "tree",
    "block_path",
    "block_tree",
    "multi_scalar",
    "multi_block",
)


class _SecondReset(fire_d):
    """A distinct ordered voltage process with overlapping support."""


@pytest.fixture(params=(1, 4), ids=("threads1", "threads4"))
def cpu_threads(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    try:
        yield request.param
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _model(
    variant="fire_r_d",
    *,
    kind="point",
    device="cpu",
    dtype=torch.float64,
    batch=2,
    regional=False,
    ordered=False,
):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True, DEVICE=device, DTYPE=dtype):
        if kind == "point":
            model = dn.Population(
                N=2, C=3, dtype=dtype, integrator=dn.bwd_euler_sc(imem=True)
            )
            model.insert(pas, g=0.001, e=-54.0)
        else:
            model = _membrane_model(
                kind, device=device, dtype=dtype, batch=None, imem=True
            )
        parameters = {"threshold": -50.0, "rest": -53.0}
        if variant in {"fire_r", "fire_r_d"}:
            parameters["refractory"] = 0.12
        if variant in {"fire_d", "fire_r_d"}:
            parameters.update(tau_gate=0.8, ste_scale=0.75)
        target = model[..., ::2] if regional else model
        target.insert(FIRE_TYPES[variant], **parameters)
        if ordered:
            model[..., 1:].insert(
                _SecondReset, threshold=-52.0, rest=-54.0, tau_gate=0.6, ste_scale=0.8
            )
        if batch is not None:
            model.batch(batch)
        voltage = torch.linspace(
            -51.2, -49.2, model.v.numel(), device=device, dtype=dtype
        ).reshape(model.shape)
        model.set_v_init(voltage)
        model.train()
        model.initialize()
    model._voltage_test_options = dict(
        variant=variant,
        kind=kind,
        device=device,
        dtype=dtype,
        batch=batch,
        regional=regional,
        ordered=ordered,
    )
    return model


def _runner(
    model,
    *,
    backend="eager",
    execution="scan",
    initialization="fresh",
    checkpoint_every=None,
    steps=STEPS,
):
    functional, _ = dn.func.make_functional(model, dt=DT)
    options = {}
    if backend == "inductor":
        options = {
            "fullgraph": True,
            "options": {
                "cpp_wrapper": model.device().type == "cpu",
                "compile_threads": 1,
            },
        }
    return dn.compile(
        model,
        dt=DT,
        steps=steps,
        initialization=initialization,
        execution=execution,
        inference=execution == "while",
        checkpoint_every=checkpoint_every,
        callbacks={"trace": dn.func.Recorder(tuple(_record_paths(functional)))},
        backend=backend,
        **options,
    )


def _key(mapping, suffix):
    matches = [name for name in mapping if name.endswith(suffix)]
    assert len(matches) == 1, (suffix, matches)
    return matches[0]


def _arguments(runner, model, *, steps=STEPS):
    parameters = _clone(dict(runner.parameters))
    targets = {
        name: value.requires_grad_()
        for name, value in parameters.items()
        if ".mechanisms." in name and value.is_floating_point()
    }
    initialization = _clone(runner.initialization_inputs)
    targets["initial_voltage"] = initialization.v_init.requires_grad_()
    targets["stimulus"] = model.v.new_ones((), requires_grad=True)
    shape = (steps, *model.shape)
    drive = torch.linspace(
        -1e-9,
        2e-9,
        steps * model.v.numel(),
        device=model.device(),
        dtype=model.dtype(),
    ).reshape(shape)
    field = torch.linspace(
        -0.03,
        0.04,
        drive.numel(),
        device=model.device(),
        dtype=model.dtype(),
    ).reshape(shape)
    return (
        parameters,
        initialization,
        dn.func.RolloutInput(field, drive * targets["stimulus"]),
        targets,
    )


def _loss(recorded):
    voltage = recorded["v"]
    weights = torch.linspace(
        0.7, 1.3, voltage.shape[-1], device=voltage.device, dtype=voltage.dtype
    )
    result = 0.001 * (weights * voltage.square()).mean()
    for name, value in recorded.items():
        if name.endswith(("reset_gate", "spike_gate", "time_refractory")):
            result = result + 0.1 * value.square().mean()
        elif name in {"hh.m", "hh.h", "hh.n"}:
            result = result + 0.05 * value.square().mean()
        elif name == "vc":
            result = result + 0.001 * value.square().mean()
    return result


def _gradients(recorded, targets):
    gradients = torch.autograd.grad(
        _loss(recorded), tuple(targets.values()), allow_unused=True
    )
    result = {
        name: torch.zeros_like(target) if gradient is None else gradient
        for (name, target), gradient in zip(targets.items(), gradients, strict=True)
    }
    for name, value in result.items():
        assert torch.isfinite(value).all(), name
    return result


def _close(actual, expected, *, dtype=torch.float64):
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        reference = expected[name]
        if value.dtype == torch.bool:
            assert torch.equal(value, reference), name
            continue
        # Numerical cancellation in the accepted-step current is relative to
        # nA, not mA. Never let a public-unit tolerance hide a wrong trace.
        if name == "i_membrane":
            value, reference = value * 1e6, reference * 1e6
        torch.testing.assert_close(
            value,
            reference,
            rtol=5e-4 if dtype == torch.float32 else 2e-8,
            atol=3e-5 if dtype == torch.float32 else 2e-9,
            msg=name,
        )


def _imperative_reference(model, runner, parameters, initialization, inputs):
    reference = _model(**model._voltage_test_options)
    raw = dict(reference.named_parameters())
    with torch.no_grad():
        for name, value in parameters.items():
            raw[name].copy_(value)
        reference.set_v_init(initialization.v_init.detach().clone())
        reference.initialize()
    reference.populate_parameter_buffers()
    targets = {
        name: raw[name].requires_grad_()
        for name, value in parameters.items()
        if ".mechanisms." in name and value.is_floating_point()
    }
    # Imperative initialization deliberately detaches v_init. Install an
    # independent leaf after running the real initialization, including fresh
    # refractory/gate state, to check the initial-voltage transition derivative.
    reference.v = reference.v.detach().clone().requires_grad_()
    targets["initial_voltage"] = reference.v
    if hasattr(reference, "vc"):
        reference.vc = torch.cat(
            (reference.v[..., None] + reference.vc[..., 1:2], reference.vc[..., 1:]),
            dim=-1,
        )
    # Run the actual canonical carry initialization and parameter preparation
    # on the independent model before stepping its imperative integrator.
    reference.mech.initialize(reference.v, reference.celsius, reference.diam)
    for mechanism in reference.mech.mechanisms.values():
        if mechanism.DE:
            # _init_buffers_s intentionally detaches inferred State defaults.
            # Reinstall those same defaults without detach for the requested
            # fresh-initialization AD oracle (the transition still runs the
            # independently constructed imperative model).
            defaults = mechanism._derive_initial_state_values(
                mechanism.get(reference.v)
            )
            for name, value in defaults.items():
                mechanism._buffers[name] = value
    dt = runner.constants["dt"].detach().clone()
    amplitude = reference.v.new_ones((), requires_grad=True)
    targets["stimulus"] = amplitude
    inputs = _clone(inputs)._replace(intra=inputs.intra.detach() * amplitude)
    reference.integrator._initialize(
        reference, dt, force=True, compile_scope="population"
    )
    paths = _record_paths(runner.functional)

    def observe():
        state = _live_state(runner.functional, reference)
        return {name: _read_path(state, path) for name, path in paths.items()}

    frames = [observe()]
    for index in range(inputs.intra.shape[0]):
        reference.integrator.step(reference, dt, inputs.ve[index], inputs.intra[index])
        reference.t = reference.t + dt
        frames.append(observe())
    recorded = {name: torch.stack([frame[name] for frame in frames]) for name in paths}
    return recorded, _gradients(recorded, targets)


def _point_recurrence(model, parameters, initialization, inputs, dt):
    """Independent reset-first passive implicit-Euler equations, without Dendra."""
    kind = model._voltage_test_options["variant"]
    voltage = initialization.v_init
    zero = torch.zeros_like(voltage)
    state = {}
    if kind in {"fire_r", "fire_r_d"}:
        state.update(is_refractory=zero.bool(), time_refractory=zero)
    if kind in {"fire_d", "fire_r_d"}:
        state["reset_gate"] = zero
    if kind == "fire_r_d":
        state["spike_gate"] = zero
    fire_parameters = {
        name: parameters[_key(parameters, f".{kind}.{name}_param")]
        for name in ("threshold", "rest", "refractory", "tau_gate", "ste_scale")
        if any(key.endswith(f".{kind}.{name}_param") for key in parameters)
    }
    g = parameters[_key(parameters, ".pas.g_param")]
    e = parameters[_key(parameters, ".pas.e_param")]
    cmdt = 1e-3 * model.cm.detach() * model.cm_scale.detach() / dt
    area = model.area.detach() * model.area_scale.detach()
    frames = [
        {"v": voltage, **{f"{kind}.{key}": value for key, value in state.items()}}
    ]
    for drive in inputs.intra:
        reset, state = _reference_fire(kind, voltage, dt, fire_parameters, state)
        voltage = reset + (drive / area - g * (reset - e)) / (cmdt + g)
        frames.append(
            {"v": voltage, **{f"{kind}.{key}": value for key, value in state.items()}}
        )
    return {name: torch.stack([frame[name] for frame in frames]) for name in frames[0]}


def _exercise(
    variant="fire_r_d",
    *,
    backend="eager",
    execution="scan",
    checkpoint_every=None,
    **options,
):
    model = _model(variant, **options)
    runner = _runner(
        model, backend=backend, execution=execution, checkpoint_every=checkpoint_every
    )
    snapshot = _module_snapshot(model)
    parameters, initialization, inputs, targets = _arguments(runner, model)
    actual = runner(parameters=parameters, initialization=initialization, inputs=inputs)
    recorded = dict(actual[1]["callbacks"]["trace"])
    gradients = _gradients(recorded, targets) if execution == "scan" else {}
    reference, expected_gradients = _imperative_reference(
        model, runner, parameters, initialization, inputs
    )
    _close(recorded, reference, dtype=model.dtype())
    if gradients:
        _close(gradients, expected_gradients, dtype=model.dtype())
        assert torch.count_nonzero(gradients["stimulus"])
        assert torch.count_nonzero(gradients["initial_voltage"])
    if options.get("kind", "point") == "point" and not options.get("regional"):
        rp, ri, rx, rt = _arguments(runner, model)
        independent = _point_recurrence(model, rp, ri, rx, runner.constants["dt"])
        _close(
            {name: recorded[name] for name in independent},
            independent,
            dtype=model.dtype(),
        )
        if gradients:
            _close(gradients, _gradients(independent, rt), dtype=model.dtype())
    _assert_module_snapshot(model, snapshot)
    return recorded, gradients


@pytest.mark.parametrize("variant", tuple(FIRE_TYPES))
def test_fire_full_recording_and_gradient_oracles(variant, cpu_threads):
    _exercise(variant)


@pytest.mark.parametrize("variant", tuple(FIRE_TYPES))
def test_float32_fire_full_traces_and_gradients(variant, cpu_threads):
    _exercise(variant, dtype=torch.float32)


@pytest.mark.parametrize("kind", KINDS[1:])
@pytest.mark.parametrize("variant", tuple(FIRE_TYPES))
def test_fire_geometries_full_traces_and_gradients(kind, variant, cpu_threads):
    _exercise(variant, kind=kind)


@pytest.mark.parametrize("variant", tuple(FIRE_TYPES))
def test_voltage_process_precedes_hh_gating_and_voltage_solve(variant, cpu_threads):
    model = _model(variant, kind="single")
    runner = _runner(model, steps=1)
    recorded = runner()[1]["callbacks"]["trace"]
    voltage = recorded["v"][0]
    carry = {
        name.removeprefix(f"{variant}."): value[0]
        for name, value in recorded.items()
        if name.startswith(f"{variant}.")
    }
    process = model.mech.mechanisms[variant]
    parameters = {
        name: getattr(process, name)
        for name in ("threshold", "rest", "refractory", "tau_gate", "ste_scale")
        if hasattr(process, name)
    }
    reset_voltage, _ = _reference_fire(variant, voltage, DT, parameters, carry)
    assert not torch.equal(reset_voltage, voltage)
    # Standard HH rate equations at these nonsingular voltages, independently
    # of Dendra's assigned-value/CNEXP implementation. This catches a reset
    # applied after the gate update, or accidentally after the voltage solve.
    rates = {
        "m": (
            0.1 * (-(reset_voltage + 40)) / torch.expm1(-(reset_voltage + 40) / 10),
            4 * torch.exp(-(reset_voltage + 65) / 18),
        ),
        "h": (
            0.07 * torch.exp(-(reset_voltage + 65) / 20),
            1 / (torch.exp(-(reset_voltage + 35) / 10) + 1),
        ),
        "n": (
            0.01 * (-(reset_voltage + 55)) / torch.expm1(-(reset_voltage + 55) / 10),
            0.125 * torch.exp(-(reset_voltage + 65) / 80),
        ),
    }
    temperature_scale = 3 ** ((model.celsius - 6.3) / 10)
    for gate, (alpha, beta) in rates.items():
        equilibrium = alpha / (alpha + beta)
        expected = equilibrium + (recorded[f"hh.{gate}"][0] - equilibrium) * torch.exp(
            -DT * temperature_scale * (alpha + beta)
        )
        torch.testing.assert_close(
            recorded[f"hh.{gate}"][1], expected, rtol=2e-10, atol=2e-11
        )


@pytest.mark.parametrize("variant", tuple(FIRE_TYPES))
def test_regional_fire_leaves_other_compartments_untouched(variant, cpu_threads):
    _exercise(variant, regional=True)


def test_overlapping_voltage_processes_keep_declared_order(cpu_threads):
    _exercise("fire_r_d", regional=True, ordered=True)


@pytest.mark.parametrize("variant", tuple(FIRE_TYPES))
def test_fire_vmap_empty_vmap_and_forward_reverse_derivatives(variant, cpu_threads):
    model = _model(variant, batch=None)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters = _clone(tensors.parameters)
    initialization = _clone(tensors.initialization)
    threshold_key = _key(parameters, f".{variant}.threshold_param")
    callbacks = functional.make_callbacks(
        {"trace": dn.func.Recorder(tuple(_record_paths(functional)))}
    )

    def rollout(threshold, voltage, drive):
        initialized = functional.initialize(
            {**parameters, threshold_key: threshold},
            tensors.constants,
            initialization._replace(v_init=voltage),
        )
        _, auxiliary = functional.prepare_and_rollout(
            initialized.parameters,
            initialized.constants,
            initialized.state,
            inputs=dn.func.RolloutInput(None, drive),
            steps=STEPS,
            callbacks=callbacks,
        )
        return dict(auxiliary["callbacks"]["trace"])

    def objective(threshold, voltage, drive):
        return _loss(rollout(threshold, voltage, drive))

    values = (
        parameters[threshold_key],
        initialization.v_init,
        model.v.new_full((STEPS, *model.shape), 1e-10),
    )
    lanes = tuple(torch.stack((value * 0.99, value * 1.01)) for value in values)

    def compare(evaluate):
        expected = pytree.tree_map(
            lambda *leaves: torch.stack(leaves),
            *[evaluate(*(value[index] for value in lanes)) for index in range(2)],
        )
        _assert_tree_close(
            torch.func.vmap(evaluate)(*lanes), expected, rtol=2e-8, atol=2e-9
        )

    compare(rollout)
    derivative = torch.func.grad(objective, argnums=(0, 1, 2))
    compare(derivative)
    empty = torch.func.vmap(rollout)(*(value[:0] for value in lanes))
    assert all(value.shape[0] == 0 for value in empty.values())
    directions = tuple(
        torch.full_like(value, 0.01 if index < 2 else 1e-10)
        for index, value in enumerate(values)
    )
    _, tangent = torch.func.jvp(objective, values, directions)
    expected = sum(
        (gradient * direction).sum()
        for gradient, direction in zip(derivative(*values), directions, strict=True)
    )
    torch.testing.assert_close(tangent, expected, rtol=2e-8, atol=2e-9)


def _exercise_continuation(variant, *, device="cpu", dtype=torch.float64):
    model = _model(variant, device=device, dtype=dtype)
    fresh = _runner(model, steps=STEPS // 2)
    first = fresh()
    repeated = fresh()
    _assert_tree_close(first[0], repeated[0])
    _close(
        dict(first[1]["callbacks"]["trace"]),
        dict(repeated[1]["callbacks"]["trace"]),
    )
    # A source with active refractory state is a legitimate snapshot restart;
    # fresh initialization must instead reset timers/gates without a t=0 step.
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        dt = model.v.new_tensor(DT)
        model.integrator._initialize(model, dt, force=True, compile_scope="population")
        model.integrator.step(model, dt)
        model.t = model.t + dt
    snapshot = _runner(model, initialization="snapshot", steps=STEPS // 2)
    reset = _runner(model, initialization="fresh", steps=STEPS // 2)
    result = snapshot()
    paths = _record_paths(snapshot.functional)
    for name, path in paths.items():
        torch.testing.assert_close(
            result[1]["callbacks"]["trace"][name][0],
            _read_path(snapshot.initial_state, path),
            rtol=0,
            atol=0,
        )
    if variant != "fire":
        fresh_trace = reset()[1]["callbacks"]["trace"]
        for name in paths:
            if name.startswith(f"{variant}."):
                assert torch.count_nonzero(fresh_trace[name][0]) == 0
    continuation = fresh(first[0], callback_state=first[1]["callbacks"].state)
    whole = _runner(_model(variant, device=device, dtype=dtype), steps=STEPS)()
    _assert_tree_close(
        continuation[0],
        whole[0],
        rtol=5e-4 if dtype == torch.float32 else 2e-9,
        atol=3e-5 if dtype == torch.float32 else 2e-10,
    )
    joined = {
        name: torch.cat((first[1]["callbacks"]["trace"][name], value))
        for name, value in continuation[1]["callbacks"]["trace"].items()
    }
    _close(joined, dict(whole[1]["callbacks"]["trace"]), dtype=dtype)


@pytest.mark.parametrize("variant", tuple(FIRE_TYPES))
def test_fresh_snapshot_and_explicit_state_continuation(variant, cpu_threads):
    _exercise_continuation(variant)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("execution", "checkpoint_every"), (("scan", 2), ("while", None))
)
def test_cpp_fire_scan_checkpoint_and_inference_while(
    execution, checkpoint_every, cpu_threads
):
    recorded, gradients = _exercise(
        "fire_r_d",
        kind="tree" if execution == "scan" else "point",
        backend="inductor",
        execution=execution,
        checkpoint_every=checkpoint_every,
    )
    assert recorded["fire_r_d.is_refractory"].dtype == torch.bool
    if execution == "while":
        assert not gradients and not recorded["v"].requires_grad
