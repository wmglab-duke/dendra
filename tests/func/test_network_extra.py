"""Bound Network extracellular stimulation retains Population semantics."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators.implicit import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mod import pas
from dendra.models.stim.waveform import Waveform
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _module_snapshot,
)
from tests.func.test_network import DTYPE, _clone, _close
from tests.func.test_stimulation import (
    _assert_module_unchanged,
)
from tests.func.test_stimulation import _module_snapshot as _waveform_snapshot

DT = 0.01
START = 0.175
PREFIX = "stimulation.extra."
CONDUCTANCE = "integrator.mech.mechanisms.pas.g_param"
pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _compiler_environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch_compiler_warning_context():
            yield
    finally:
        torch.set_num_threads(previous)


class _BufferedSine(Waveform):
    FUNCTIONAL_PURE = True
    Waveform.PARAMETER(amp=0.18, freq=0.9, phase=0.2)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("offset", self.amp.new_tensor(0.03))

    def fn(self, t):
        return (
            self.amp * torch.sin(2 * torch.pi * self.freq * t + self.phase)
            + self.offset
        )


def _network(kind="path", *, dtype=DTYPE, batch=2):
    if kind == "finite" and not DENDRA_SOLVERS_AVAILABLE:
        pytest.skip("finite extracellular cable execution requires dendra-solvers")
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True, DTYPE=dtype, DEVICE="cpu"):
        shared = torch.nn.Parameter(torch.tensor(0.001, dtype=dtype))
        populations = {}
        for index, name in enumerate(("left", "right")):
            options = dict(v_init=-65.0 - index, dtype=dtype)
            if kind == "finite":
                population = dn.ExtCellAxon(
                    [5.0, 8.0],
                    n_comp=3,
                    integrator=dn.bwd_euler_bt(method="thomas", imem=True),
                    **options,
                )
                for attribute, low, high in (
                    ("xraxial", 1.5, 3.0),
                    ("xc", 0.08, 0.18),
                    ("xg", 1e-4, 3e-4),
                ):
                    value = getattr(population, attribute)
                    value.copy_(
                        torch.linspace(
                            low, high, value.numel(), dtype=dtype
                        ).reshape_as(value)
                    )
            else:
                population = dn.Unmyelinated(
                    [2.0, 2.5],
                    L=4.0,
                    dx=1.0,
                    integrator=dn.bwd_euler_ub(method="pcr", imem=True),
                    **options,
                )
            population.insert(pas, g=shared, e=-63.0 - 4 * index)
            populations[name] = population
        network = dn.Network(populations)
        if batch is not None:
            network.batch(batch)
        network.train()
        network.initialize(DT)
        network._reset_runtime_clock(START)
        for index, population in enumerate(network.populations.values()):
            population.v = torch.linspace(
                -66.0 - index, -60.0 - index, population.v.numel(), dtype=dtype
            ).reshape(population.shape)
            if hasattr(population, "vc"):
                inner = torch.linspace(
                    -0.4, 0.6, population.v.numel(), dtype=dtype
                ).reshape(population.shape)
                population.vc = torch.stack(
                    (population.v + inner, inner, -0.3 * inner), dim=-1
                )
    return network


def _extra(network, *, shared=False):
    first = next(iter(network.populations.values()))
    field = torch.linspace(-1.25, 1.75, first.v.numel(), dtype=first.dtype()).reshape(
        first.shape
    )
    with dn.ctx(REQUIRE_GRAD=1, DTYPE=first.dtype(), DEVICE="cpu"):
        waveform = _BufferedSine()
        return {
            name: (
                field if shared or index == 0 else field.flip(-1) * -0.7,
                (
                    waveform
                    if shared or index == 0
                    else _BufferedSine(amp=-0.11, freq=0.55, phase=-0.15)
                ),
            )
            for index, name in enumerate(network.populations)
        }


def _recorders(network):
    fields = ["t"]
    for name, population in network.populations.items():
        fields.extend((f"{name}.v", f"{name}.i_membrane", f"{name}.t"))
        if hasattr(population, "vc"):
            fields.append(f"{name}.vc")
    return {"trace": dn.func.Recorder(tuple(fields))}


def _observe(network):
    result = {"t": network.t.clone()}
    for name, population in network.populations.items():
        for field in ("v", "i_membrane", "t"):
            result[f"{name}.{field}"] = getattr(population, field).clone()
        if hasattr(population, "vc"):
            result[f"{name}.vc"] = population.vc.clone()
    return result


def _native(network, extra, steps):
    frames = [_observe(network)]
    for _ in range(steps):
        network.step(extra=extra)
        frames.append(_observe(network))
    return {name: torch.stack([frame[name] for frame in frames]) for name in frames[0]}


def _explicit_inputs(parameters, state, steps, *, intra=None):
    """Assemble the authored sine directly, independently of FunctionalExtra."""
    clock = state["clock"]
    times = (
        clock["origin"]
        + (clock["step"] + torch.arange(steps, device=clock["step"].device)).to(
            clock["origin"].dtype
        )
        * DT
    )
    populations = {}
    for name, values in parameters["populations"].items():
        field = values[PREFIX + "contacts.0.field"]
        time = times.to(field)
        wave = PREFIX + "waveforms.0."
        amplitude = (
            values[wave + "amp"]
            * torch.sin(
                2 * torch.pi * values[wave + "freq"] * time + values[wave + "phase"]
            )
            + values[wave + "offset"]
        )
        ve = amplitude.reshape(steps, *((1,) * field.ndim)) * field
        populations[name] = dn.func.RolloutInput(
            ve=ve, intra=None if intra is None else intra.populations[name].intra
        )
    return dn.func.NetworkRolloutInput(populations=populations)


def _ordinary_parameters(parameters, plain):
    return {
        **parameters,
        "populations": {
            name: {
                key: parameters["populations"][name][key]
                for key in plan._parameter_names
            }
            for name, plan in plain.populations.items()
        },
    }


def _plain_reference(network, parameters, constants, state, steps, *, intra=None):
    plain, plain_tensors = dn.func.make_functional_network(network)
    del constants
    return plain.prepare_and_rollout(
        _ordinary_parameters(parameters, plain),
        plain_tensors.constants,
        state,
        _explicit_inputs(parameters, state, steps, intra=intra),
        steps=steps,
        callbacks=plain.make_callbacks(_recorders(network)),
    )


@pytest.mark.parametrize("kind", ("path", "finite"))
@pytest.mark.parametrize("execution", ("eager", "scan", "while"))
def test_bound_network_extra_matches_native_pre_step_sampling_and_explicit_voltage_tapes(
    kind, execution
):
    model, native = _network(kind), _network(kind)
    extra, native_extra = _extra(model), _extra(native)
    source = {
        name: _module_snapshot(population)
        for name, population in model.populations.items()
    }
    waveforms = {name: _waveform_snapshot(pair[1]) for name, pair in extra.items()}
    functional, tensors = dn.func.make_functional_network(model, extra=extra)
    frozen = _clone(tensors, detached=True)
    callbacks = functional.make_callbacks(_recorders(model))
    chunk = (
        None
        if execution == "eager"
        else functional.compile_rollout_chunk(
            4, execution=execution, backend="aot_eager"
        )
    )
    with torch.no_grad():
        reference, explicit = _plain_reference(
            model, tensors.parameters, tensors.constants, tensors.state, 4
        )
        actual, auxiliary = functional.run(
            *tensors, steps=4, compiled_chunk=chunk, callbacks=callbacks
        )
        recorded = _native(native, native_extra, 4)
        unstimulated, _ = dn.func.make_functional_network(_network(kind))
        baseline, _ = unstimulated.prepare_and_rollout(*unstimulated.extract(), steps=4)
    _close(actual, reference)
    _close(actual, functional.extract(native).state)
    _close(auxiliary["callbacks"]["trace"], explicit["callbacks"]["trace"])
    _close(auxiliary["callbacks"]["trace"], recorded)
    for name in model.populations:
        assert not torch.equal(
            actual["populations"][name]["integrator"]["v"],
            baseline["populations"][name]["integrator"]["v"],
        )
        _assert_module_snapshot(model.populations[name], source[name])
        _assert_module_unchanged(extra[name][1], waveforms[name])
    _close(tensors, frozen)
    if chunk is not None:
        assert chunk.execution_report().strategy == (
            "compiled_while_loop" if execution == "while" else "compiled_scan"
        )


def _arguments(runner, network, steps):
    state = _clone(runner.initial_state, detached=True)
    initial, inputs = [], {}
    for index, (name, population) in enumerate(network.populations.items()):
        integrator = state["populations"][name]["integrator"]
        voltage = integrator["v"].requires_grad_()
        initial.append(voltage)
        if "vc" in integrator:
            vc = integrator["vc"]
            integrator["vc"] = torch.cat(
                (voltage[..., None] + vc[..., 1:2], vc[..., 1:]), dim=-1
            )
        drive = (
            torch.linspace(
                -2e-9,
                3e-9 + index * 1e-9,
                steps * population.v.numel(),
                dtype=population.dtype(),
            )
            .reshape(steps, *population.shape)
            .requires_grad_()
        )
        inputs[name] = dn.func.RolloutInput(intra=drive)
    return state, dn.func.NetworkRolloutInput(populations=inputs), tuple(initial)


def _loss(auxiliary):
    trace = auxiliary["callbacks"]["trace"]
    loss = trace["left.v"].new_zeros(())
    for name in ("left", "right"):
        loss = loss + 0.001 * trace[f"{name}.v"].square().mean()
        loss = loss + (trace[f"{name}.i_membrane"] * 1e6).square().mean()
        if f"{name}.vc" in trace:
            shell = trace[f"{name}.vc"][..., 1:]
            weights = torch.linspace(0.7, 1.3, shell.shape[-2], dtype=shell.dtype)
            loss = loss + 0.1 * (shell * weights[:, None]).mean()
    return loss


@pytest.mark.parametrize("kind", ("path", "finite"))
def test_bound_shared_network_extra_scan_checkpoint_preserves_all_stimulation_and_drive_gradients(
    kind,
):
    network = _network(kind)
    extra = _extra(network, shared=True)
    runner = dn.compile(
        network,
        dt=DT,
        steps=6,
        extra=extra,
        execution="scan",
        inference=False,
        trainable=("populations.*.stimulation.extra.*", f"populations.*.{CONDUCTANCE}"),
        checkpoint_every=4,
        callbacks=_recorders(network),
        backend="aot_eager",
    )
    shared_names = (
        PREFIX + "contacts.0.field",
        *(
            PREFIX + "waveforms.0." + name
            for name in ("amp", "freq", "phase", "offset")
        ),
        CONDUCTANCE,
    )
    left, right = (runner.parameters["populations"][name] for name in ("left", "right"))
    for name in shared_names:
        assert left[name] is right[name]
        assert left[name].is_leaf and left[name].requires_grad
    fields_before = extra["left"][0].clone()
    waveform_before = _waveform_snapshot(extra["left"][1])

    def evaluate(*, reference):
        state, inputs, initial = _arguments(runner, network, 6)
        targets = (
            *initial,
            *(value.intra for value in inputs.populations.values()),
            *(left[name] for name in shared_names),
        )
        if reference:
            result = _plain_reference(
                network, runner.parameters, runner.constants, state, 6, intra=inputs
            )
        else:
            result = runner(state, inputs=inputs)
        gradients = torch.autograd.grad(_loss(result[1]), targets)
        assert all(
            torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
        )
        return result, gradients

    oracle, expected = evaluate(reference=True)
    for _ in range(2):
        actual, gradients = evaluate(reference=False)
        _close(actual[0], oracle[0])
        _close(actual[1]["callbacks"]["trace"], oracle[1]["callbacks"]["trace"])
        for gradient, wanted in zip(gradients, expected, strict=True):
            torch.testing.assert_close(gradient, wanted, rtol=2e-8, atol=1e-12)
    torch.testing.assert_close(extra["left"][0], fields_before)
    _assert_module_unchanged(extra["left"][1], waveform_before)
    assert (
        runner.parameters["populations"]["left"][shared_names[0]].data_ptr()
        != extra["left"][0].data_ptr()
    )


@pytest.mark.parametrize("initialization", ("snapshot", "fresh"))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_network_extra_runner_restart_and_explicit_continuation_match_native(
    initialization, execution
):
    model, native = _network(), _network()
    extra, native_extra = _extra(model, shared=True), _extra(native, shared=True)
    runner = dn.compile(
        model,
        dt=DT,
        steps=4,
        extra=extra,
        initialization=initialization,
        execution=execution,
        inference=execution == "while",
        callbacks=_recorders(model),
        backend="aot_eager",
    )
    frozen = _clone((runner._parameters, runner.initial_state), detached=True)
    if initialization == "fresh":
        native.initialize(DT)
    with torch.no_grad():
        recorded = _native(native, native_extra, 8)
    prefix, first = runner()
    repeated, repeated_aux = runner()
    final, last = runner(prefix, callback_state=first["callbacks"].state)
    _close(prefix, repeated)
    _close(first["callbacks"]["trace"], repeated_aux["callbacks"]["trace"])
    joined = {
        name: torch.cat((first["callbacks"]["trace"][name], value))
        for name, value in last["callbacks"]["trace"].items()
    }
    _close(joined, recorded)
    _close(final, runner.functional.extract(native).state)
    assert joined["t"].shape == (9,)
    assert joined["t"][0] == (0.0 if initialization == "fresh" else START)
    _close((runner._parameters, runner.initial_state), frozen)


def test_network_extra_explicit_parameter_overrides_keep_shared_field_generator_gradients():
    network = _network()
    runner = dn.compile(
        network,
        dt=DT,
        steps=4,
        extra=_extra(network, shared=True),
        inference=False,
        checkpoint_every=2,
        callbacks=_recorders(network),
        backend="aot_eager",
    )

    def evaluate(*, reference):
        template = runner.parameters["populations"]["left"][PREFIX + "contacts.0.field"]
        coordinate = template.new_tensor(0.2, requires_grad=True)
        raw_amplitude = template.new_tensor(0.3, requires_grad=True)
        position = torch.linspace(
            -1.0, 1.0, template.numel(), dtype=template.dtype
        ).reshape_as(template)
        field = torch.exp(-(position - coordinate).square())
        amplitude = raw_amplitude.sigmoid() * 0.3
        parameters = {
            **runner.parameters,
            "populations": {
                name: {
                    **values,
                    PREFIX + "contacts.0.field": field,
                    PREFIX + "waveforms.0.amp": amplitude,
                }
                for name, values in runner.parameters["populations"].items()
            },
        }
        result = (
            _plain_reference(
                network, parameters, runner.constants, runner.initial_state, 4
            )
            if reference
            else runner(parameters=parameters)
        )
        gradients = torch.autograd.grad(_loss(result[1]), (coordinate, raw_amplitude))
        return result, gradients

    actual, gradients = evaluate(reference=False)
    oracle, expected = evaluate(reference=True)
    _close(actual[0], oracle[0])
    for gradient, wanted in zip(gradients, expected, strict=True):
        torch.testing.assert_close(gradient, wanted, rtol=2e-8, atol=1e-12)
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_bound_network_extra_commit_and_native_checkpoint_round_trip_preserve_continuation():
    model, committed, restored = (_network() for _ in range(3))
    functional, tensors = dn.func.make_functional_network(
        model, extra=_extra(model, shared=True)
    )
    parameters = tensors.independent_parameters()
    values = parameters["populations"]["left"]
    # Extra is external to native runtime checkpoints. Align those numerical
    # sources explicitly when publishing carry from independently edited leaves.
    with torch.no_grad():
        values[PREFIX + "contacts.0.field"].mul_(1.4)
        values[PREFIX + "waveforms.0.amp"].mul_(0.7)
    with dn.ctx(REQUIRE_GRAD=0, DTYPE=DTYPE, DEVICE="cpu"):
        waveform = _BufferedSine(
            amp=values[PREFIX + "waveforms.0.amp"].detach().clone(),
            freq=values[PREFIX + "waveforms.0.freq"].detach().clone(),
            phase=values[PREFIX + "waveforms.0.phase"].detach().clone(),
        )
    waveform.offset.copy_(values[PREFIX + "waveforms.0.offset"])
    native_extra = {
        name: (values[PREFIX + "contacts.0.field"].detach().clone(), waveform)
        for name in model.populations
    }
    callbacks = functional.make_callbacks(_recorders(model))
    with torch.no_grad():
        prefix, _ = functional.prepare_and_rollout(
            parameters, tensors.constants, tensors.state, steps=3
        )
        assert functional.commit_state_(committed, prefix) is committed
        checkpoint = _clone(committed.state_dict_for_checkpoint(), detached=True)
        restored.restore_dict_from_checkpoint(checkpoint)
        _close(functional.extract(restored).state, prefix)
        final, auxiliary = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            prefix,
            steps=4,
            callbacks=callbacks,
        )
        expected = _native(restored, native_extra, 4)
    _close(final, functional.extract(restored).state)
    _close(auxiliary["callbacks"]["trace"], expected)
    torch.testing.assert_close(
        final["clock"]["origin"], tensors.state["clock"]["origin"]
    )
    assert final["clock"]["step"].item() == 7
    _close(functional.extract().state, tensors.state)


@pytest.mark.parametrize("leaf", ("contacts.0.field", "waveforms.0.offset"))
def test_bound_network_extra_checkpoint_backward_rejects_changed_promoted_leaves(leaf):
    network = _network()
    runner = dn.compile(
        network,
        dt=DT,
        steps=4,
        extra=_extra(network, shared=True),
        inference=False,
        trainable=f"populations.*.{PREFIX}{leaf}",
        checkpoint_every=2,
        callbacks=_recorders(network),
        backend="aot_eager",
    )
    result = runner()
    target = runner.parameters["populations"]["left"][PREFIX + leaf]
    with torch.no_grad():
        target.add_(0.1)
    with pytest.raises(
        (dn.func.FunctionalizationError, RuntimeError),
        match="changed|modified|version",
    ):
        torch.autograd.grad(_loss(result[1]), target)


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_bound_network_extra_float32_pulse_samples_exact_network_clock_across_continuation(
    execution,
):
    model, native, unstimulated = (
        _network(dtype=torch.float32, batch=None) for _ in range(3)
    )
    for network in (model, native, unstimulated):
        network._reset_runtime_clock(1000.0)
    field = torch.linspace(
        -1.25, 1.75, model.left.v.numel(), dtype=torch.float32
    ).reshape(model.left.shape)
    with dn.ctx(REQUIRE_GRAD=1, DTYPE=torch.float32, DEVICE="cpu"):
        pulse = dn.mono_rect(
            amp=1.0, delay=float(torch.tensor(1000.07, dtype=torch.float32)), pw=0.001
        )
    extra = {"left": (field, pulse)}
    runner = dn.compile(
        model,
        dt=DT,
        steps=4,
        extra=extra,
        execution=execution,
        inference=execution == "while",
        checkpoint_every=2 if execution == "scan" else None,
        callbacks=_recorders(model),
        backend="aot_eager",
    )
    prefix, first = runner()
    final, tail = runner(prefix, callback_state=first["callbacks"].state)
    trace = {
        name: torch.cat((first["callbacks"]["trace"][name], value))
        for name, value in tail["callbacks"]["trace"].items()
    }
    with torch.no_grad():
        expected = _native(native, extra, 8)
        baseline = _native(unstimulated, None, 8)
    for name in trace:
        torch.testing.assert_close(trace[name], expected[name], rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(trace["t"], expected["t"], rtol=0, atol=0)
    assert trace["t"].shape == (9,)
    assert not torch.equal(trace["left.v"][-1], baseline["left.v"][-1])
    torch.testing.assert_close(
        trace["right.v"], baseline["right.v"], rtol=3e-5, atol=3e-6
    )
    assert final["clock"]["step"].item() == 8


@pytest.mark.slow
def test_cpp_while_bound_network_extra_matches_eager_explicit_voltage_tapes():
    network = _network(batch=1)
    runner = dn.compile(
        network,
        dt=DT,
        steps=5,
        extra=_extra(network, shared=True),
        execution="while",
        inference=True,
        callbacks=_recorders(network),
        backend="inductor",
        options={"cpp_wrapper": True, "compile_threads": 1},
    )
    with torch.no_grad():
        expected, oracle = _plain_reference(
            network, runner.parameters, runner.constants, runner.initial_state, 5
        )
        plain, tensors = dn.func.make_functional_network(network)
        baseline, _ = plain.prepare_and_rollout(*tensors, steps=5)
    actual, auxiliary = runner()
    _close(actual, expected)
    _close(auxiliary["callbacks"]["trace"], oracle["callbacks"]["trace"])
    for name in network.populations:
        assert not torch.equal(
            actual["populations"][name]["integrator"]["v"],
            baseline["populations"][name]["integrator"]["v"],
        )
    assert runner.execution_report().strategy == "compiled_while_loop"
