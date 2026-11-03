"""MPS clock staging retains waveform, grid and chunked-training gradients."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.core import _time_grid_from_step_count
from dendra.models.mod import pas
from dendra.models.stim.waveform import Waveform
from dendra.units import nA

DT = 0.025
STEPS = 4
pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS is unavailable"
    ),
]


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(
            DEVICE="mps",
            DTYPE=torch.float32,
            JIT=0,
            REQUIRE_GRAD=1,
            CHECK_PURITY=False,
            IMEM=False,
        ),
    ):
        yield


class _Linear(Waveform):
    FUNCTIONAL_PURE = True
    Waveform.PARAMETER(amp=1.25, slope=0.75)

    def fn(self, t):
        return self.amp + self.slope * t


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _close(actual, expected):
    left, spec = pytree.tree_flatten(actual)
    right, other = pytree.tree_flatten(expected)
    assert spec == other
    for value, reference in zip(left, right, strict=True):
        if torch.is_tensor(value):
            floating = value.is_floating_point() or value.is_complex()
            torch.testing.assert_close(
                value,
                reference,
                rtol=3e-4 if floating else 0,
                atol=3e-5 if floating else 0,
            )
        else:
            assert value == reference


def _parameter(mapping, suffix):
    matches = [value for name, value in mapping.items() if name.endswith(suffix)]
    assert len(matches) == 1
    return matches[0]


def _point():
    model = dn.SingleCompartment(
        N=1,
        C=1,
        v_init=-65.0,
        dtype=torch.float32,
        device="mps",
        integrator=dn.bwd_euler_sc(imem=False),
    )
    model.insert(pas, g=0.001, e=-65.0)
    return model


@pytest.mark.parametrize("backend", ("eager", "aot_eager"))
@pytest.mark.parametrize("kind", ("intra", "extra"))
def test_cpu_binary64_stimulation_times_keep_linear_time_and_parameter_vjps(
    kind, backend
):
    model, waveform = _point(), _Linear()
    if kind == "intra":
        model[:, 0].inject(waveform)
    model.train()
    model.initialize(DT)
    field = torch.full(model.shape, 2.0, device="mps", dtype=torch.float32)
    plan = (
        dn.func.FunctionalIntra(model)
        if kind == "intra"
        else dn.func.FunctionalExtra(model, (field, waveform))
    )
    tensors = plan.extract()
    parameters = {
        name: value.detach().clone().requires_grad_()
        for name, value in tensors.parameters.items()
    }
    amp, slope = _parameter(parameters, ".amp"), _parameter(parameters, ".slope")
    times = torch.tensor(
        [0.25, 0.5, 0.75, 1.0], device="cpu", dtype=torch.float64, requires_grad=True
    )
    frozen = _clone((parameters, tensors.constants, times))
    versions = [
        value._version
        for value in pytree.tree_leaves((parameters, tensors.constants, times))
    ]

    def assemble(raw, clock):
        # Source/schema validation belongs outside capture; the conversion is
        # the same one used by the public deterministic assembly boundary.
        return plan.assemble_values(raw, tensors.constants, plan._times(clock))

    execute = (
        assemble
        if backend == "eager"
        else torch.compile(assemble, backend=backend, fullgraph=True)
    )
    actual = execute(parameters, times)
    _close(actual, plan.assemble(parameters, tensors.constants, times))
    coefficient = 1.0 if kind == "intra" else 2.0
    expected = coefficient * (
        amp + slope * times.to(dtype=torch.float32).to(device="mps")
    )
    _close(actual, expected.reshape(STEPS, 1, 1))
    weights = torch.arange(1, STEPS + 1, device="mps", dtype=torch.float32)
    gradients = torch.autograd.grad(
        (actual[:, 0, 0] * weights).sum(), (amp, slope, times)
    )
    wanted = (
        torch.ones_like(amp) * coefficient * weights.sum(),
        torch.ones_like(slope)
        * coefficient
        * (weights * times.detach().float().to("mps")).sum(),
        (coefficient * slope.detach() * weights).to("cpu").to(torch.float64),
    )
    _close(gradients, wanted)
    assert all(torch.count_nonzero(value) > 0 for value in gradients)
    assert gradients[-1].device.type == "cpu" and gradients[-1].dtype == torch.float64
    _close((parameters, tensors.constants, times), frozen)
    assert [
        value._version
        for value in pytree.tree_leaves((parameters, tensors.constants, times))
    ] == versions


@pytest.mark.parametrize("backend", ("eager", "aot_eager"))
def test_mps_time_grid_keeps_binary64_cpu_staging_and_start_dt_vjps(backend):
    start = torch.tensor(1.25, device="mps", dtype=torch.float32, requires_grad=True)
    dt = torch.tensor(0.125, device="cpu", dtype=torch.float64, requires_grad=True)
    frozen = _clone((start, dt))

    def grid(initial, timestep):
        return _time_grid_from_step_count(
            initial, STEPS, timestep, device=torch.device("mps")
        )

    execute = (
        grid
        if backend == "eager"
        else torch.compile(grid, backend=backend, fullgraph=True)
    )
    actual = execute(start, dt)
    offsets = torch.arange(STEPS, device="cpu", dtype=torch.float64)
    expected = (start.detach().cpu().double() + offsets * dt.detach()).float().to("mps")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    weights = torch.arange(1, STEPS + 1, device="mps", dtype=torch.float32)
    actual_start, actual_dt = torch.autograd.grad((actual * weights).sum(), (start, dt))
    _close(actual_start, weights.sum())
    _close(actual_dt, (weights.cpu().double() * offsets).sum())
    assert actual_start.device.type == "mps" and actual_start.dtype == torch.float32
    assert actual_dt.device.type == "cpu" and actual_dt.dtype == torch.float64
    assert actual_start > 0 and actual_dt > 0
    _close((start, dt), frozen)


def _model(network):
    cell = _point()
    waveform = _Linear(amp=0.2 * nA, slope=0.3 * nA)
    cell[:, 0].inject(waveform)
    model = dn.Network({"cell": cell}) if network else cell
    model.train()
    model.initialize(DT)
    return model, cell, waveform


def _voltage(state, network):
    return (state["populations"]["cell"] if network else state)["integrator"]["v"]


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("checkpointed", (False, True), ids=("scan", "checkpointed"))
def test_mps_waveform_scan_chunks_keep_native_gradients_controls_and_rng(
    network, checkpointed
):
    source, _, _ = _model(network)
    native, native_cell, native_waveform = _model(network)
    trace_name = "cell.v" if network else "v"
    program = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        chunk_steps=STEPS,
        execution="scan",
        inference=False,
        initialization="snapshot",
        trainable=("*amp", "*slope"),
        checkpoint_every=STEPS if checkpointed else None,
        backend="aot_eager",
        callbacks={"trace": dn.func.Recorder((trace_name, "t"))},
    )
    initial = _clone(program.initial_state)
    initial_voltage = _voltage(initial, network).requires_grad_()
    native_voltage = native_cell.v.detach().clone().requires_grad_()
    native_cell.v = native_voltage
    frozen = _clone((initial, program.parameters, program.constants))
    versions = [
        value._version
        for value in pytree.tree_leaves(initial)
        if torch.is_tensor(value)
    ]
    cpu_rng, mps_rng = torch.get_rng_state().clone(), torch.mps.get_rng_state().clone()
    first, prefix = program(initial)
    final, suffix = program(prefix["continuation"])
    recorded = [native_cell.v.clone()]
    for _ in range(2 * STEPS):
        native.step() if network else native.step(DT)
        recorded.append(native_cell.v.clone())
    _close(final, program.functional.extract(native).state)
    trace = torch.cat(
        (
            prefix["callbacks"]["trace"][trace_name],
            suffix["callbacks"]["trace"][trace_name],
        )
    )
    _close(trace, torch.stack(recorded))
    amp = _parameter(program.named_parameters, ".amp")
    slope = _parameter(program.named_parameters, ".slope")
    actual_gradients = torch.autograd.grad(
        _voltage(final, network).square().mean(), (amp, slope, initial_voltage)
    )
    expected_gradients = torch.autograd.grad(
        native_cell.v.square().mean(),
        (native_waveform.amp, native_waveform.slope, native_voltage),
    )
    _close(actual_gradients, expected_gradients)
    assert all(
        torch.isfinite(value).all() and torch.count_nonzero(value) > 0
        for value in actual_gradients
    )
    _close((initial, program.parameters, program.constants), frozen)
    assert [
        value._version
        for value in pytree.tree_leaves(initial)
        if torch.is_tensor(value)
    ] == versions
    _close(torch.get_rng_state(), cpu_rng)
    _close(torch.mps.get_rng_state(), mps_rng)
    assert program.execution_report().strategy == "compiled_scan"
    assert _voltage(first, network).requires_grad


def _poisson_case():
    generator = torch.Generator(device="cpu").manual_seed(317)
    pulse = dn.mono_rect(amp=100.0 * nA, pw=0.045).poisson(
        interval=0.055,
        n=3,
        start=-0.035,
        noise=0.7,
        randomize_every_call=True,
        generator=generator,
    )
    model = dn.SingleCompartment(
        N=2,
        C=1,
        v_init=-60.0,
        integrator=dn.bwd_euler_sc(imem=False),
    )
    model.insert(pas.rename("leak"), g=0.001, e=-45.0)
    model[:, 0].inject(pulse)
    model.train()
    model.initialize()
    model.integrator._initialize(model, model.t.new_tensor(DT), force=True)
    model.detach()
    return model, pulse, generator


def _native(model, steps, *, gradients=False):
    expected = [model.v.clone()]
    with torch.set_grad_enabled(gradients):
        for _ in range(steps):
            model.step(DT)
            expected.append(model.v.clone())
    return torch.stack(expected)


def _rng_matches(state, native_generator):
    values = state["stimulation"]["rng"]
    assert len(values) == 1
    torch.testing.assert_close(
        next(iter(values.values())), native_generator.get_state(), rtol=0, atol=0
    )


def test_mps_float32_poisson_aot_scan_amplitude_gradient_and_detached_draws():
    source, _, generator = _poisson_case()
    native, native_pulse, native_generator = _poisson_case()
    before = generator.get_state().clone()
    program = dn.compile(
        source,
        dt=DT,
        steps=2 * STEPS,
        execution="scan",
        inference=False,
        initialization="snapshot",
        trainable="*waveform.amp",
        backend="aot_eager",
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
    )
    packet = program.functional.sample_runtime_draws(
        program.initial_state, steps=2 * STEPS
    )
    packet = packet._replace(
        draws=pytree.tree_map_only(
            torch.Tensor,
            lambda value: value.detach()
            .clone()
            .requires_grad_(value.is_floating_point()),
            packet.draws,
        )
    )
    frozen = _clone(packet)
    actual, auxiliary = program(draws=packet)
    expected_trace = _native(native, 2 * STEPS, gradients=True)
    _close(actual, program.functional.extract(native).state)
    _close(auxiliary["callbacks"]["trace"]["v"], expected_trace)
    amplitude = next(
        value
        for name, value in program.named_parameters.items()
        if name.endswith("waveform.amp")
    )
    base = tuple(
        value
        for value in pytree.tree_leaves(packet.draws)
        if torch.is_tensor(value) and value.requires_grad
    )
    gradients = torch.autograd.grad(
        actual["integrator"]["v"].square().mean(),
        (amplitude, *base),
        allow_unused=True,
    )
    reference = torch.autograd.grad(
        native.v.square().mean(), native_pulse.waveform.amp
    )[0]
    assert torch.count_nonzero(reference) > 0
    assert gradients[0] is not None
    _close(gradients[0], reference)
    assert all(
        value is None or torch.count_nonzero(value) == 0 for value in gradients[1:]
    )
    _rng_matches(actual, native_generator)
    _close(packet, frozen)
    torch.testing.assert_close(generator.get_state(), before, rtol=0, atol=0)
    assert program.execution_report().strategy == "compiled_scan"
