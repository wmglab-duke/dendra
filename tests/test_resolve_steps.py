"""Read-only step budgets size deterministic tapes at explicit boundaries."""

from __future__ import annotations

import os

import pytest
import torch

import dendra as dn
from dendra.models.backend import Backend
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_lowering_api import DT, _network, _population, _target, _voltage
from tests.func.test_native_handoff import _capture, _unchanged
from tests.test_native_runtime_draws import _environment as _environment
from tests.test_runtime_draw_sampling_horizons import (
    _native_run,
    _pending,
    _with_pending,
)
from tests.test_runtime_draw_sampling_targets import _ExtraNoiseStep
from tests.test_runtime_draw_sampling_targets import _Network as _SamplingNetwork
from tests.test_runtime_draw_sampling_targets import _NoisyLeak

pytestmark = pytest.mark.cpu


_TARGETS = [
    (network, kind)
    for network in (False, True)
    for kind in ("native", "plan", "raw", "prepared", "kernel", "runner")
    if not (network and kind == "prepared")
]


def _model(network):
    return _network() if network else _population()


def _inputs(state, network, steps):
    voltage = (
        state["populations"]["left"]["integrator"]["v"]
        if network
        else state["integrator"]["v"]
    )
    current = (
        torch.linspace(
            0.2e-9,
            0.7e-9,
            steps,
            device=voltage.device,
            dtype=voltage.dtype,
        )
        .reshape(steps, *(1,) * voltage.ndim)
        .requires_grad_()
    )
    inputs = dn.func.Inputs(intra=current)
    return current, {"left": inputs, "right": inputs} if network else inputs


@pytest.mark.parametrize("network,kind", _TARGETS)
def test_resolve_steps_reads_each_supported_boundary_without_executing_hooks(
    network, kind, monkeypatch
):
    model = _model(network)
    _native_run(model, DT, duration=0.6 * DT)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    if kind == "native":
        target, arguments, options = model, (), {"dt": DT}
    else:
        if kind == "plan":
            target = functional
        elif kind == "raw":
            target = functional.bind(tensors)
        elif kind == "prepared":
            target = functional.bind(
                tensors.parameters,
                functional.prepare(tensors.parameters, tensors.constants),
            )
        elif kind == "kernel":
            target = functional.compile_rollout_chunk(2, backend="eager", disable=True)
        else:
            target = dn.compile(model, dt=DT, steps=7, backend="eager", disable=True)
            functional = target.functional
            tensors = functional.extract()
        state = (
            tensors.state
            if kind == "plan"
            else dn.func.Continuation(
                tensors.state, tensors.parameters, tensors.constants
            )
        )
        arguments, options = (state,), {}

    native = _capture(model)
    frames = _snapshot((tensors.parameters, tensors.constants, tensors.state))
    ambient = torch.get_rng_state().clone()
    import dendra.func._runtime_sampling as sampling

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Step resolution executed preparation, extraction or sampling"
        )

    with monkeypatch.context() as guard:
        guard.setattr(functional, "prepare", forbidden)
        guard.setattr(functional, "extract", forbidden)
        guard.setattr(dn.func, "make_functional", forbidden)
        guard.setattr(dn.func, "sample_runtime_draws", forbidden)
        guard.setattr(sampling, "sample_native_runtime_draws", forbidden)
        guard.setattr(type(functional._runtime_draw_plan), "sample_runtime", forbidden)
        count = dn.func.resolve_steps(target, *arguments, duration=0.6 * DT, **options)
        assert type(count) is int and count == 1
        assert dn.func.resolve_steps(target, *arguments, duration=0.0, **options) == 0
        assert dn.func.resolve_steps(target, *arguments, steps=0, **options) == 0
        assert dn.func.resolve_steps(target, *arguments, steps=5, **options) == 5
    _close((tensors.parameters, tensors.constants, tensors.state), frames)
    _close(torch.get_rng_state(), ambient)
    _unchanged(model, native)


@pytest.mark.parametrize(
    "functional_target", (False, True), ids=("native", "functional")
)
@pytest.mark.parametrize(
    "pending,duration,expected",
    ((0.15, 0.15, 1), (0.15, 0.0, 0), (0.15, 0.1, 0), (0.6, 0.0, 2)),
)
def test_step_resolution_uses_decimal_and_retained_duration_without_spending_it(
    functional_target, pending, duration, expected
):
    model = _population()
    model._set_duration_remainder(pending)
    functional, tensors = dn.func.make_functional(model, dt=0.3)
    target, arguments, options = (
        (functional, (tensors.state,), {})
        if functional_target
        else (model, (), {"dt": 0.3})
    )
    before, native = _snapshot(tensors.state), _capture(model)
    for _ in range(2):
        assert (
            dn.func.resolve_steps(target, *arguments, duration=duration, **options)
            == expected
        )
        assert dn.func.resolve_steps(target, *arguments, steps=1, **options) == 1
    _close(tensors.state, before)
    _unchanged(model, native)


@pytest.mark.parametrize("kind", ("plan", "raw", "prepared", "runner"))
def test_population_step_resolution_honors_effective_timestep_ownership(kind):
    program = dn.compile(_population(), dt=DT, steps=5, backend="eager", disable=True)
    functional = program.functional
    constants = {**program.constants, "dt": program.constants["dt"].new_tensor(2 * DT)}
    state = _with_pending(program.initial_state, False, 0.4 * DT)
    packet = dn.func.Continuation(state, program.parameters, constants)
    target = (
        functional
        if kind == "plan"
        else program
        if kind == "runner"
        else functional.bind(program.parameters, constants=constants)
        if kind == "raw"
        else functional.bind(
            program.parameters, functional.prepare(program.parameters, constants)
        )
    )
    assert dn.func.resolve_steps(target, packet, duration=1.8 * DT) == 1
    assert dn.func.resolve_steps(target, packet, dt=2 * DT, duration=1.8 * DT) == 1
    with pytest.raises(ValueError, match="timestep"):
        dn.func.resolve_steps(target, packet, dt=DT, duration=1.8 * DT)
    # A plain-state plan cannot infer constants owned by a different binding.
    assert dn.func.resolve_steps(functional, state, duration=1.8 * DT) == 2


def test_plain_runner_state_uses_live_default_timestep_without_refreshing_preparation(
    monkeypatch,
):
    program = dn.compile(
        _population(), dt=DT, steps=3, inference=True, backend="eager", disable=True
    )
    program(steps=0)
    prepared, report = program._prepared, program.execution_report()
    assert prepared is not None
    with torch.no_grad():
        program.constants["dt"].mul_(2)
    state = _with_pending(program.initial_state, False, 0.4 * DT)

    def forbidden(*args, **kwargs):
        raise AssertionError("Step resolution refreshed a runner preparation cache")

    with monkeypatch.context() as guard:
        guard.setattr(program, "_prepare", forbidden)
        guard.setattr(program.functional, "prepare", forbidden)
        assert dn.func.resolve_steps(program, state, duration=1.8 * DT) == 1
    assert program._prepared is prepared and program.execution_report() is report


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_resolved_input_tapes_preserve_native_functional_vjps_and_checkpoint_suffix(
    network, tmp_path
):
    model = _model(network)
    _native_run(model, DT, duration=0.6 * DT)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters = tensors.independent_parameters(trainable="*g_param")
    initial = dn.func.Continuation(tensors.state, parameters, tensors.constants)
    count = dn.func.resolve_steps(functional, initial, duration=2.6 * DT)
    assert count == dn.func.resolve_steps(model, dt=DT, duration=2.6 * DT) == 3
    current, inputs = _inputs(initial.state, network, count)
    expected, auxiliary = dn.func.run(
        functional, initial, duration=2.6 * DT, inputs=inputs, checkpoint_every=2
    )
    wanted = torch.autograd.grad(
        _voltage(expected, network).square().mean(),
        (current, _target(parameters, network)),
    )
    _native_run(
        model, DT, duration=2.6 * DT, inputs=inputs, method="longrun_checkpointed"
    )
    native_parameter = (
        model.left.mech.leak.g_param if network else model.mech.leak.g_param
    )
    native_voltage = (
        torch.cat([cell.v.flatten() for cell in model.populations.values()])
        if network
        else model.v
    )
    observed = torch.autograd.grad(
        native_voltage.square().mean(),
        (current, native_parameter),
    )
    _close(observed, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in observed)
    _close(functional.extract(model).state, expected)
    assert _pending(expected, network).item() == pytest.approx(0.2 * DT)

    path = tmp_path / "continuation.pt"
    torch.save(dn.func.export_continuation(functional, auxiliary["continuation"]), path)
    restored = dn.func.restore_continuation(
        functional, torch.load(path, weights_only=True)
    )
    receiver = _model(network)
    receiver.restore_dict_from_checkpoint(_snapshot(model.state_dict_for_checkpoint()))
    suffix = dn.func.resolve_steps(functional, restored, duration=0.8 * DT)
    assert suffix == dn.func.resolve_steps(receiver, dt=DT, duration=0.8 * DT) == 1
    _, next_inputs = _inputs(restored.state, network, suffix)
    final, _ = dn.func.run(functional, restored, duration=0.8 * DT, inputs=next_inputs)
    _native_run(receiver, DT, duration=0.8 * DT, inputs=next_inputs)
    _close(functional.extract(receiver).state, final)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_step_queries_require_one_explicit_horizon_and_validate_boundaries(network):
    model = _model(network)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    initial = dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants)
    native = _capture(model)
    for target, arguments, options in (
        (model, (), {"dt": DT}),
        (functional, (initial,), {}),
    ):
        for request in (
            {},
            {"steps": 1, "duration": DT},
            {"steps": True},
            {"steps": -1},
            {"steps": 1.5},
            {"duration": True},
            {"duration": -DT},
            {"duration": float("nan")},
            {"duration": float("inf")},
        ):
            with pytest.raises((TypeError, ValueError)):
                dn.func.resolve_steps(target, *arguments, **options, **request)
    with pytest.raises(TypeError, match="state"):
        dn.func.resolve_steps(model, initial, steps=1, dt=DT)
    with pytest.raises(TypeError, match="state"):
        dn.func.resolve_steps(functional, steps=1)
    with pytest.raises(ValueError, match="dt|timestep"):
        dn.func.resolve_steps(functional, initial, duration=DT, dt=2 * DT)
    invalid = _with_pending(tensors.state, network, -DT)
    with pytest.raises(ValueError, match="non-negative"):
        dn.func.resolve_steps(functional, invalid, duration=DT)
    invalid = _with_pending(tensors.state, network, 0.0)
    _pending(invalid, network).requires_grad_()
    with pytest.raises(dn.func.FunctionalizationError, match="gradients"):
        dn.func.resolve_steps(functional, invalid, duration=DT)
    _unchanged(model, native)


@pytest.mark.parametrize("context", ("compile", "grad", "forward_ad"))
def test_resolve_steps_rejects_function_transforms_before_sampling(
    context, monkeypatch
):
    functional, tensors = dn.func.make_functional(_population(), dt=DT)

    def work(value):
        dn.func.resolve_steps(functional, tensors.state, duration=DT)
        return value.square()

    with pytest.raises(dn.func.FunctionalizationError, match="resolve_steps.*outside"):
        if context == "compile":
            monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
            work(torch.tensor(0.5))
        elif context == "grad":
            torch.func.grad(work)(torch.tensor(0.5))
        else:
            with torch.autograd.forward_ad.dual_level():
                work(torch.tensor(0.5))


def test_native_backend_default_and_tiny_duration_match_the_explicit_budget(
    monkeypatch,
):
    model = _population()
    monkeypatch.setattr(Backend, "dt", 0.3)
    model._set_duration_remainder(0.15)
    before = _capture(model)
    assert dn.func.resolve_steps(model, duration=0.15) == 1
    assert dn.func.resolve_steps(model, duration=1e-300) == 0
    _unchanged(model, before)
    model._set_duration_remainder(0.0)
    assert dn.func.resolve_steps(model, dt=1e-300, duration=1e-300) == 1


def test_custom_network_budgets_are_not_invoked_by_step_queries(monkeypatch):
    model = _SamplingNetwork()
    model.initialize(DT)
    native = _capture(model)

    def forbidden(*args, **kwargs):
        raise AssertionError("Step query invoked an authored duration budget")

    monkeypatch.setattr(type(model), "_duration_budget", property(forbidden))
    assert dn.func.resolve_steps(model, steps=2) == 2
    with pytest.raises(dn.func.FunctionalizationError, match="canonical.*duration"):
        dn.func.resolve_steps(model, duration=DT)
    _unchanged(model, native)


def test_native_resolution_does_not_require_a_functional_integrator_or_sampler(
    monkeypatch,
):
    import dendra.func._runtime_sampling as sampling

    model = dn.SingleCompartment(N=1, C=1, integrator=_ExtraNoiseStep)
    model.insert(_NoisyLeak.rename("leak"))
    model.train()
    model.initialize()
    before = _capture(model)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Step query entered functional integration or RNG sampling"
        )

    with monkeypatch.context() as guard:
        guard.setattr(_ExtraNoiseStep, "_functional_spec", forbidden)
        guard.setattr(dn.func, "make_functional", forbidden)
        guard.setattr(dn.func, "sample_runtime_draws", forbidden)
        guard.setattr(sampling, "sample_native_runtime_draws", forbidden)
        assert dn.func.resolve_steps(model, dt=DT, duration=1.5 * DT) == 1
        assert dn.func.resolve_steps(model, dt=DT, steps=2) == 2
    _unchanged(model, before)


@pytest.mark.mps
def test_mps_read_only_resolution_sizes_while_inputs_and_portable_continuation(
    tmp_path,
):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DEVICE="mps", DTYPE=torch.float32):
        model = _population()
        _native_run(model, DT, duration=0.6 * DT)
        program = dn.compile(
            model, dt=DT, steps=4, chunk_steps=4, execution="while", backend="aot_eager"
        )
        initial = dn.func.Continuation(
            program.initial_state, program.parameters, program.constants
        )
        before, ambient = _capture(model), torch.mps.get_rng_state().clone()
        with torch.inference_mode():
            count = dn.func.resolve_steps(program, initial, duration=3.6 * DT)
            assert type(count) is int and count == 4
        _unchanged(model, before)
        _close(torch.mps.get_rng_state(), ambient)
        _, inputs = _inputs(initial.state, False, count)
        state, auxiliary = program(initial, duration=3.6 * DT, inputs=inputs)
        assert state["integrator"]["v"].device.type == "mps"
        assert state["clock"]["t"].item() == pytest.approx(4 * DT)
        path = tmp_path / "while.pt"
        torch.save(program.export_continuation(auxiliary["continuation"]), path)
        restored = program.restore_continuation(torch.load(path, weights_only=True))
        assert dn.func.resolve_steps(program, restored, duration=0.8 * DT) == 1
        _, tail = _inputs(restored.state, False, 1)
        final, _ = program(restored, duration=0.8 * DT, inputs=tail)
        assert final["clock"]["t"].item() == pytest.approx(5 * DT)
