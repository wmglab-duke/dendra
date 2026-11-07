"""Runtime draw tapes share execution's duration and fractional-time budget."""

from __future__ import annotations

import os

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.backend import Backend
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _plain, _snapshot
from tests.func.test_native_handoff import _capture, _unchanged
from tests.func.test_state_sde_execution import DT as SDE_DT
from tests.func.test_state_sde_execution import _HeunLeak
from tests.func.test_state_sde_execution import _population as _sde_population
from tests.func.test_state_sde_execution import _SharedLeak
from tests.test_native_runtime_draw_tapes import (
    _functional_sources,
    _loss,
    _sources,
    _state_loss,
)
from tests.test_native_runtime_draws import _environment as _environment
from tests.test_runtime_draw_sampling_targets import DT as TARGET_DT
from tests.test_runtime_draw_sampling_targets import _Network as _SamplingNetwork
from tests.test_shared_runtime_draw_sampling import (
    _case,
    _generators,
    _selected,
)

pytestmark = pytest.mark.cpu


def _native_run(model, dt, *, extra=None, method="run", **options):
    if not isinstance(model, dn.Network):
        options["dt"] = dt
    if method == "longrun_checkpointed":
        options.update(chunklength=2, safe_checkpoint=True)
    return getattr(model, method)(extra=extra, **options)


def _pending(state, network):
    return state["clock" if network else "control"]["duration_remainder"]


def _with_pending(state, network, value):
    result = _plain(state)
    owner = result["clock" if network else "control"]
    owner["duration_remainder"] = owner["duration_remainder"].new_tensor(value)
    return result


def _frame(packet):
    return pytree.tree_map_only(torch.Tensor, lambda value: value[0], packet)


_TARGETS = [
    (network, kind)
    for network in (False, True)
    for kind in ("native", "plan", "raw", "prepared", "kernel", "runner")
    if not (network and kind == "prepared")
]


@pytest.mark.parametrize("network,kind", _TARGETS)
def test_duration_sampling_preserves_frame_defaults_and_all_target_boundaries(
    network, kind
):
    model, _, dt = _case("netstim" if network else "noise")
    _native_run(model, dt, duration=0.6 * dt)
    functional, tensors = dn.func.make_functional(model, dt=dt)
    if kind == "native":
        target, arguments, options = model, (), {"dt": dt}
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
            target = dn.compile(model, dt=dt, steps=7, backend="eager", disable=True)
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
    before = _snapshot((tensors.parameters, tensors.constants, tensors.state))
    native = _capture(model)
    expected = dn.func.sample_runtime_draws(target, *arguments, steps=1, **options)
    default = dn.func.sample_runtime_draws(target, *arguments, **options)
    duration = dn.func.sample_runtime_draws(
        target, *arguments, duration=0.6 * dt, **options
    )
    _assert_packet_equal(default, _frame(expected))
    _assert_packet_equal(duration, expected)
    _assert_packet_equal(
        dn.func.sample_runtime_draws(target, *arguments, duration=0.6 * dt, **options),
        duration,
    )
    _close((tensors.parameters, tensors.constants, tensors.state), before)
    assert _pending(tensors.state, network).item() == pytest.approx(0.6 * dt)
    _unchanged(model, native)


@pytest.mark.parametrize("kind", ("noise", "netstim", "poisson", "mixed"))
def test_duration_draw_replay_preserves_stochastic_gradients_and_checkpoint_suffix(
    kind, tmp_path
):
    source, source_extra, dt = _case(kind)
    native, extra, _ = _case(kind)
    network = isinstance(native, dn.Network)
    if network and native.netstim is not None:
        # NetStim's entropy-seeded reseeding generator is independent of its
        # configured runtime seed. Align the complete checkpoint boundary.
        native.netstim.restore_dict_from_checkpoint(
            _snapshot(source.netstim.state_dict_for_checkpoint())
        )
    for model, binding in ((source, source_extra), (native, extra)):
        _native_run(model, dt, extra=binding, duration=0.6 * dt)
    functional, tensors = dn.func.make_functional(source, dt=dt, extra=source_extra)
    boundary = dn.func.Continuation(
        tensors.state, tensors.parameters, tensors.constants
    )
    saved = _capture(native)
    generators = _generators(native, extra)
    generator_states = [value.get_state().clone() for value in generators]
    duration = 2.6 * dt
    draws = dn.func.sample_runtime_draws(native, dt=dt, extra=extra, duration=duration)
    _assert_packet_equal(
        draws, dn.func.sample_runtime_draws(functional, boundary, duration=duration)
    )
    _assert_packet_equal(
        draws, dn.func.sample_runtime_draws(native, dt=dt, extra=extra, steps=3)
    )
    _unchanged(native, saved)
    for generator, state in zip(generators, generator_states, strict=True):
        _close(generator.get_state(), state)
    expected, auxiliary = dn.func.run(
        functional, boundary, duration=duration, draws=draws
    )
    automatic, _ = dn.func.run(functional, boundary, duration=duration)
    _close(automatic, expected)
    _native_run(
        native,
        dt,
        extra=extra,
        method="longrun_checkpointed",
        duration=duration,
        draws=draws,
    )
    actual = dn.func.make_functional(native, dt=dt, extra=extra)[1].state
    _close(actual, expected)
    assert _pending(expected, network).item() == pytest.approx(0.2 * dt)
    selected = _selected if kind in ("poisson", "mixed") else _sources
    wanted = torch.autograd.grad(
        _state_loss(expected, network),
        _functional_sources(functional, tensors, selected(source)),
    )
    observed = torch.autograd.grad(_loss(native), selected(native))
    _close(observed, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in observed)

    # Both portable and native checkpoint restoration retain the same incoming
    # fractional time and logical RNG boundary for the next draw tape.
    path = tmp_path / "functional.pt"
    torch.save(dn.func.export_continuation(functional, auxiliary["continuation"]), path)
    restored = dn.func.restore_continuation(
        functional, torch.load(path, weights_only=True)
    )
    receiver, receiving_extra, _ = _case(kind)
    receiver.restore_dict_from_checkpoint(_snapshot(native.state_dict_for_checkpoint()))
    following = dn.func.sample_runtime_draws(
        native, dt=dt, extra=extra, duration=0.8 * dt
    )
    _assert_packet_equal(
        following,
        dn.func.sample_runtime_draws(
            receiver, dt=dt, extra=receiving_extra, duration=0.8 * dt
        ),
    )
    _assert_packet_equal(
        following, dn.func.sample_runtime_draws(functional, restored, duration=0.8 * dt)
    )


@pytest.mark.parametrize(
    "functional_target", (False, True), ids=("native", "functional")
)
@pytest.mark.parametrize(
    "pending,duration,steps",
    ((0.15, 0.15, 1), (0.15, 0.0, 0), (0.15, 0.1, 0), (0.6, 0.0, 2)),
)
def test_decimal_duration_budget_and_empty_tapes_are_side_effect_free(
    functional_target, pending, duration, steps
):
    model = _sde_population(_SharedLeak)
    model._set_duration_remainder(pending)
    dt = 0.3
    functional, tensors = dn.func.make_functional(model, dt=dt)
    target, arguments, options = (
        (functional, (tensors.state,), {})
        if functional_target
        else (model, (), {"dt": dt})
    )
    native = _capture(model)
    before = _snapshot(tensors.state)
    expected = dn.func.sample_runtime_draws(target, *arguments, steps=steps, **options)
    actual = dn.func.sample_runtime_draws(
        target, *arguments, duration=duration, **options
    )
    _assert_packet_equal(actual, expected)
    repeated = dn.func.sample_runtime_draws(
        target, *arguments, duration=duration, **options
    )
    _assert_packet_equal(repeated, actual)
    if steps == 0:
        for value, other in zip(
            pytree.tree_leaves(actual), pytree.tree_leaves(repeated), strict=True
        ):
            if isinstance(value, torch.Tensor):
                assert value is not other and value.shape[0] == 0
    _assert_packet_equal(tensors.state, before)
    _unchanged(model, native)


def test_tiny_duration_and_backend_default_timestep_follow_native_execution(
    monkeypatch,
):
    model = _sde_population(_SharedLeak)
    monkeypatch.setattr(Backend, "dt", 0.3)
    model._set_duration_remainder(0.15)
    empty = dn.func.sample_runtime_draws(model, duration=1e-300)
    _assert_packet_equal(empty, dn.func.sample_runtime_draws(model, steps=0))
    # A tiny positive request is not rounded to zero when it completes a tiny
    # timestep; explicit Population dt overrides the backend default.
    model._set_duration_remainder(0.0)
    tiny = dn.func.sample_runtime_draws(model, dt=1e-300, duration=1e-300)
    _assert_packet_equal(tiny, dn.func.sample_runtime_draws(model, dt=1e-300, steps=1))
    model._set_duration_remainder(0.15)
    actual = dn.func.sample_runtime_draws(model, duration=0.15)
    _assert_packet_equal(actual, dn.func.sample_runtime_draws(model, dt=0.3, steps=1))


@pytest.mark.parametrize("kind", ("plan", "runner", "raw", "prepared"))
def test_population_duration_sampling_uses_effective_source_timestep_without_preparing(
    kind, monkeypatch
):
    model = _sde_population(_SharedLeak)
    program = dn.compile(model, dt=SDE_DT, steps=3, backend="eager", disable=True)
    functional = program.functional
    tensors = functional.extract()
    constants = {
        **tensors.constants,
        "dt": tensors.constants["dt"].new_tensor(2 * SDE_DT),
    }
    state = _with_pending(tensors.state, False, 0.4 * SDE_DT)
    packet = dn.func.Continuation(state, tensors.parameters, constants)
    target = (
        functional
        if kind == "plan"
        else program
        if kind == "runner"
        else functional.bind(tensors.parameters, constants=constants)
        if kind == "raw"
        else functional.bind(
            tensors.parameters, functional.prepare(tensors.parameters, constants)
        )
    )
    expected = dn.func.sample_runtime_draws(target, packet, steps=1)

    def forbidden(*args, **kwargs):
        raise AssertionError("Duration draw budgeting entered numerical preparation")

    with monkeypatch.context() as guard:
        guard.setattr(functional, "prepare", forbidden)
        guard.setattr(functional, "extract", forbidden)
        actual = dn.func.sample_runtime_draws(target, packet, duration=1.8 * SDE_DT)
    _assert_packet_equal(actual, expected)
    with torch.no_grad():
        result, _ = dn.func.run(
            functional,
            packet,
            duration=1.8 * SDE_DT,
            draws=actual,
        )
    assert result["clock"]["t"].item() == pytest.approx(
        state["clock"]["t"].item() + 2 * SDE_DT
    )
    assert _pending(result, False).item() == pytest.approx(0.2 * SDE_DT)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_invalid_duration_requests_fail_without_touching_native_or_explicit_boundaries(
    network,
):
    model, _, dt = _case("netstim" if network else "noise")
    functional, tensors = dn.func.make_functional(model, dt=dt)
    packet = dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants)
    before = _snapshot((packet.state, packet.parameters, packet.constants))
    native = _capture(model)
    for target, args, options in ((model, (), {"dt": dt}), (functional, (packet,), {})):
        for request in (
            {"duration": -dt},
            {"duration": float("nan")},
            {"duration": float("inf")},
            {"duration": True},
            {"duration": object()},
            {"steps": 1, "duration": dt},
        ):
            with pytest.raises((TypeError, ValueError)):
                dn.func.sample_runtime_draws(target, *args, **options, **request)
    _close((packet.state, packet.parameters, packet.constants), before)
    _unchanged(model, native)


def test_runner_plain_state_duration_reads_live_timestep_without_refreshing_cache(
    monkeypatch,
):
    program = dn.compile(
        _sde_population(_SharedLeak),
        dt=SDE_DT,
        steps=3,
        inference=True,
        backend="eager",
        disable=True,
    )
    program(steps=0)
    cached = program._prepared
    assert cached is not None
    with torch.no_grad():
        program.constants["dt"].mul_(2)
    state = _with_pending(program.initial_state, False, 0.4 * SDE_DT)
    expected = dn.func.sample_runtime_draws(program, state, steps=1)

    def forbidden(*args, **kwargs):
        raise AssertionError("Duration sampling refreshed the runner cache")

    with monkeypatch.context() as guard:
        guard.setattr(program.functional, "prepare", forbidden)
        guard.setattr(program, "_prepare", forbidden)
        actual = dn.func.sample_runtime_draws(program, state, duration=1.8 * SDE_DT)
    _assert_packet_equal(actual, expected)
    assert program._prepared is cached
    result, _ = program(state, duration=1.8 * SDE_DT, draws=actual)
    assert result["clock"]["t"].item() == pytest.approx(2 * SDE_DT)
    assert _pending(result, False).item() == pytest.approx(0.2 * SDE_DT)


@pytest.mark.mps
def test_mps_duration_sampling_retains_fractional_carry_and_checkpointed_gradients():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DEVICE="mps", DTYPE=torch.float32):
        model = _sde_population(_HeunLeak)
        _native_run(model, SDE_DT, duration=0.6 * SDE_DT)
        functional, tensors = dn.func.make_functional(model, dt=SDE_DT)
        before, ambient = _capture(model), torch.mps.get_rng_state().clone()
        with torch.inference_mode():
            packet = dn.func.sample_runtime_draws(
                model, dt=SDE_DT, duration=2.6 * SDE_DT
            )
        _assert_packet_equal(
            packet,
            dn.func.sample_runtime_draws(
                functional, tensors.state, duration=2.6 * SDE_DT
            ),
        )
        _unchanged(model, before)
        _close(torch.mps.get_rng_state(), ambient)
        assert all(
            not value.is_inference() and not value.requires_grad
            for value in pytree.tree_leaves(packet)
            if isinstance(value, torch.Tensor)
        )
        _native_run(
            model,
            SDE_DT,
            method="longrun_checkpointed",
            duration=2.6 * SDE_DT,
            draws=packet,
        )
        gate = next(iter(model.mech.sde.DE.values()))
        gradient = torch.autograd.grad(model.v.square().mean(), gate.sigma_param)[0]
        assert (
            gradient.device.type == "mps"
            and gradient.isfinite().all()
            and gradient.abs().sum() > 0
        )
        assert model._duration_remainder.item() == pytest.approx(0.2 * SDE_DT)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("invalid", ("negative", "nonfinite", "gradient"))
def test_duration_sampling_validates_fractional_carry_before_rng_requests(
    network, invalid, monkeypatch
):
    model, _, dt = _case("netstim" if network else "noise")
    functional, tensors = dn.func.make_functional(model, dt=dt)
    state = _with_pending(
        tensors.state,
        network,
        -dt
        if invalid == "negative"
        else float("nan")
        if invalid == "nonfinite"
        else 0.0,
    )
    if invalid == "gradient":
        _pending(state, network).requires_grad_()

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid duration carry reached RNG sampling")

    monkeypatch.setattr(
        type(functional._runtime_draw_plan), "sample_runtime", forbidden
    )
    error = dn.func.FunctionalizationError if invalid == "gradient" else ValueError
    message = (
        "must not require gradients"
        if invalid == "gradient"
        else "finite and non-negative"
    )
    with pytest.raises(error, match=message):
        dn.func.sample_runtime_draws(functional, state, duration=dt)


@pytest.mark.parametrize("replacement", ("method", "descriptor", "instance"))
def test_network_duration_sampling_rejects_custom_budget_without_calling_it(
    replacement, monkeypatch
):
    model = _SamplingNetwork()
    model.initialize(TARGET_DT)
    before = _capture(model)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(replacement)
        raise AssertionError("Sampler invoked an authored duration budget")

    if replacement == "instance":
        monkeypatch.setattr(model, "_duration_budget", forbidden)
    else:
        monkeypatch.setattr(
            type(model),
            "_duration_budget",
            property(forbidden) if replacement == "descriptor" else forbidden,
        )
    # The duration-only guard leaves existing frame and step admission intact.
    assert isinstance(dn.func.sample_runtime_draws(model), dn.func.RuntimeDraws)
    assert isinstance(
        dn.func.sample_runtime_draws(model, steps=0), dn.func.RuntimeDraws
    )
    with pytest.raises(dn.func.FunctionalizationError, match="canonical.*duration"):
        dn.func.sample_runtime_draws(model, duration=TARGET_DT)
    assert calls == []
    _unchanged(model, before)
