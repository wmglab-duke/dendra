"""Native run methods replay accepted-step draws without resampling banks."""

from __future__ import annotations

import os

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.stim.waveform import Waveform
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_native_handoff import _capture, _unchanged
from tests.func.test_network import _stim_network
from tests.func.test_network_runtime_noise import DT as NETWORK_DT
from tests.func.test_network_runtime_noise import _network as _noise_network
from tests.func.test_poisson_invocation_execution import _native_state
from tests.func.test_poisson_stimulation import DT as POISSON_DT
from tests.func.test_poisson_stimulation import _waveform
from tests.func.test_state_sde_execution import DT as STATE_DT
from tests.func.test_state_sde_execution import _HeunLeak
from tests.func.test_state_sde_execution import _population as _state_population
from tests.func.test_state_sde_execution import _SharedLeak
from tests.networks.test_native_network_poisson_replay import _case as _network_banks
from tests.networks.test_native_network_poisson_replay import (
    _changed_packet,
    _gradient_sources,
)
from tests.test_native_poisson_replay import _amplitudes
from tests.test_native_poisson_replay import _case as _population_banks
from tests.test_native_poisson_replay import _changed_uniforms, _root
from tests.test_native_runtime_draws import (
    _base_draws,
    _changed_draws,
)
from tests.test_native_runtime_draws import _environment as _environment
from tests.test_native_runtime_draws import (
    _forbid_owned_sampling,
    _parameters,
    _same_parameters,
)
from tests.test_native_runtime_inputs import _HookLog
from tests.test_native_runtime_inputs import _model as _drive_population

pytestmark = pytest.mark.cpu
STEPS = 5
METHODS = ("run", "longrun", "longrun_checkpointed")


@pytest.fixture(autouse=True)
def _deterministic_stream(request):
    request.getfixturevalue("_environment")
    torch.manual_seed(17)


def _slice(value, start, stop):
    return pytree.tree_map_only(torch.Tensor, lambda leaf: leaf[start:stop], value)


def _at(value, index):
    return pytree.tree_map_only(torch.Tensor, lambda leaf: leaf[index], value)


def _cells(model):
    return model.populations if isinstance(model, dn.Network) else {"cell": model}


def _case(network):
    return (
        (_noise_network(stimulus=True), NETWORK_DT)
        if network
        else (_state_population(_SharedLeak), STATE_DT)
    )


def _drives(model, steps):
    values = {}
    for index, (name, cell) in enumerate(_cells(model).items()):
        current = torch.linspace(
            (index + 1) * 0.1e-9,
            (index + 2) * 0.5e-9,
            steps,
            dtype=cell.dtype(),
            device=cell.device(),
        ).reshape(steps, *(1,) * cell.v.ndim)
        values[name] = dn.func.Inputs(intra=current.requires_grad_())
    return values if isinstance(model, dn.Network) else values["cell"]


def _sources(model):
    result = tuple(
        value for name, value in model.named_parameters() if name.endswith("g_param")
    )
    if not isinstance(model, dn.Network):
        result += (next(iter(model.mech.sde.DE.values())).sigma_param,)
    return result


def _functional_sources(functional, tensors, originals):
    wanted, found = {id(value) for value in originals}, {}
    plans = (
        functional.populations.items()
        if isinstance(functional, dn.func.FunctionalNetwork)
        else ((None, functional),)
    )
    for name, plan in plans:
        explicit = (
            tensors.parameters
            if name is None
            else tensors.parameters["populations"][name]
        )
        authored = dict(plan._population.named_parameters(remove_duplicate=False))
        authored.update(plan._intra._tensor_sources())
        authored.update(plan._extra._tensor_sources())
        for key, source in authored.items():
            if id(source) in wanted and key in explicit:
                previous = found.setdefault(id(source), explicit[key])
                assert previous is explicit[key]
    assert found.keys() == wanted
    return tuple(found[id(value)] for value in originals)


def _recording(model, dt):
    paths = (
        tuple(f"{name}.v" for name in model.populations)
        if isinstance(model, dn.Network)
        else ("v",)
    )
    return {"trace": dn.Recorder((*paths, "t"), dt=2 * dt)}


def _run(model, method, dt, *, steps=STEPS, **options):
    options.setdefault("steps", steps)
    if not isinstance(model, dn.Network):
        options["dt"] = dt
    if method != "run":
        options.setdefault("chunklength", 3)
    if method == "longrun_checkpointed":
        options.setdefault("safe_checkpoint", True)
    result = getattr(model, method)(**options)
    if (
        method == "longrun_checkpointed"
        and options.get("functional_callbacks") is not None
    ):
        assert isinstance(result, tuple) and len(result) == 2 and result[0] is None
        return result[1]
    return result


def _loss(model, results=None):
    value = sum(cell.v.square().mean() for cell in _cells(model).values())
    if results is not None:
        value = value + sum(
            field.square().mean()
            for name, field in results["trace"].items()
            if name != "t" and field.numel()
        )
    return value


def _state_loss(state, network, results=None):
    cells = state["populations"].values() if network else (state,)
    value = sum(cell["integrator"]["v"].square().mean() for cell in cells)
    if results is not None:
        value = value + sum(
            field.square().mean()
            for name, field in results["trace"].items()
            if name != "t" and field.numel()
        )
    return value


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("method", METHODS)
def test_draw_tapes_match_functional_gradients_recording_and_automatic_checkpoint_suffix(
    network, method, monkeypatch
):
    source, dt = _case(network)
    native, _ = _case(network)
    native.restore_dict_from_checkpoint(_snapshot(source.state_dict_for_checkpoint()))
    functional, tensors = dn.func.make_functional(source, dt=dt)
    packet = _changed_draws(functional.sample_runtime_draws(tensors.state, steps=STEPS))
    inputs = _drives(native, STEPS)
    saved_packet, parameters = _snapshot(packet), _parameters(native)
    source_state = _snapshot(source.state_dict_for_checkpoint())
    configurations = _recording(native, dt)
    callbacks = dn.func.make_callbacks(native, configurations, dt=dt)
    expected_callbacks = functional.make_callbacks(configurations)
    expected, auxiliary = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        steps=STEPS,
        inputs=inputs,
        draws=packet,
        callbacks=expected_callbacks,
    )
    generator = () if not network else (native.netstim._rng,)
    ambient = torch.get_rng_state().clone()
    with monkeypatch.context() as guard:
        _forbid_owned_sampling(guard, native, *generator)
        result = _run(
            native,
            method,
            dt,
            inputs=inputs,
            draws=packet,
            functional_callbacks=callbacks,
        )
    _close(functional.extract(native).state, expected)
    _close(result.outputs, auxiliary["callbacks"].outputs)
    _close(result.state.carries, auxiliary["callbacks"].state.carries)
    frozen = _snapshot(native.state_dict_for_checkpoint())
    originals = _sources(source)
    native_parameters = _sources(native)
    source_parameters = _functional_sources(functional, tensors, originals)
    input_leaves = tuple(
        value for value in pytree.tree_leaves(inputs) if torch.is_tensor(value)
    )
    actual = torch.autograd.grad(
        _loss(native, result),
        (*native_parameters, *input_leaves, *_base_draws(packet)),
        allow_unused=True,
    )
    wanted = torch.autograd.grad(
        _state_loss(expected, network, auxiliary["callbacks"]),
        (*source_parameters, *input_leaves),
        allow_unused=True,
    )
    _close(actual[: len(wanted)], wanted)
    assert wanted[0].isfinite().all() and wanted[0].abs().sum() > 0
    assert all(
        value is not None and value.abs().sum() > 0
        for value in wanted[len(native_parameters) :]
    )
    assert all(
        value is None or value.count_nonzero() == 0 for value in actual[len(wanted) :]
    )
    _assert_packet_equal(native.state_dict_for_checkpoint(), frozen)
    _assert_packet_equal(packet, saved_packet)
    _assert_packet_equal(source.state_dict_for_checkpoint(), source_state)
    _same_parameters(native, parameters)
    _close(torch.get_rng_state(), ambient)

    receiver, _ = _case(network)
    receiver.restore_dict_from_checkpoint(frozen)
    frame = _drives(receiver, 1)
    following, _ = dn.func.run(
        functional,
        dn.func.Continuation(
            expected,
            auxiliary["continuation"].parameters,
            auxiliary["continuation"].constants,
        ),
        steps=1,
        inputs=frame,
    )
    if network:
        receiver.step(inputs=_at(frame, 0))
    else:
        receiver.step(dt, inputs=_at(frame, 0))
    _close(functional.extract(receiver).state, following)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("method", METHODS)
def test_explicit_draw_tapes_keep_native_loop_chunk_and_step_hook_order(
    network, method
):
    actual, dt = _case(network)
    baseline, _ = _case(network)
    baseline.restore_dict_from_checkpoint(_snapshot(actual.state_dict_for_checkpoint()))
    functional, tensors = dn.func.make_functional(actual, dt=dt)
    packet = functional.sample_runtime_draws(tensors.state, steps=STEPS)
    observed, expected = _HookLog(), _HookLog()
    _run(actual, method, dt, draws=packet, callbacks=[observed])
    _run(baseline, method, dt, callbacks=[expected])
    assert observed.events == expected.events


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("method", METHODS)
def test_late_invalid_rng_frames_reject_the_entire_tape_before_execution(
    network, method, monkeypatch
):
    model, dt = _case(network)
    functional, tensors = dn.func.make_functional(model, dt=dt)
    packet = _snapshot(functional.sample_runtime_draws(tensors.state, steps=STEPS))
    streams = packet.rng["model"]
    first = next(iter(streams.values()))
    boundary = next(iter(first["cpu"].values()))
    boundary[-1].fill_(255)
    saved = _capture(model)
    with monkeypatch.context() as guard:
        _forbid_owned_sampling(guard, model)
        with pytest.raises((ValueError, RuntimeError, dn.func.FunctionalizationError)):
            _run(model, method, dt, draws=packet)
    _unchanged(model, saved)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("invalid", ("frame", "length", "dtype"))
def test_draw_tapes_require_time_leading_schema_matching_the_resolved_horizon(
    network, invalid
):
    model, dt = _case(network)
    functional, tensors = dn.func.make_functional(model, dt=dt)
    packet = functional.sample_runtime_draws(
        tensors.state, **({} if invalid == "frame" else {"steps": STEPS})
    )
    if invalid == "length":
        packet = _slice(packet, 0, STEPS - 1)
    elif invalid == "dtype":
        packet = packet._replace(
            draws=pytree.tree_map_only(
                torch.Tensor, lambda value: value.float(), packet.draws
            )
        )
    saved = _capture(model)
    with pytest.raises(
        (TypeError, ValueError, RuntimeError, dn.func.FunctionalizationError)
    ):
        _run(model, "run", dt, draws=packet)
    _unchanged(model, saved)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("restore_state_after_backward", (False, True))
def test_checkpoint_training_rejects_inference_draw_tapes_before_live_mutation(
    network, restore_state_after_backward
):
    model, dt = _case(network)
    functional, tensors = dn.func.make_functional(model, dt=dt)
    sampled = functional.sample_runtime_draws(tensors.state, steps=STEPS)
    with torch.inference_mode():
        packet = pytree.tree_map_only(
            torch.Tensor, lambda value: value.clone(), sampled
        )
        empty = _slice(packet, 0, 0)
    assert all(
        torch.is_inference(value)
        for value in pytree.tree_leaves(packet)
        if torch.is_tensor(value)
    )
    saved = _capture(model)
    options = {"restore_state_after_backward": restore_state_after_backward}
    with pytest.raises(ValueError, match="inference|mutation|checkpoint"):
        _run(model, "longrun_checkpointed", dt, draws=packet, **options)
    _unchanged(model, saved)

    _run(model, "longrun_checkpointed", dt, steps=0, draws=empty, **options)
    _assert_packet_equal(model.state_dict_for_checkpoint(), saved[1])
    _close(torch.get_rng_state(), saved[2])

    ordinary = _snapshot(packet)
    assert not any(
        torch.is_inference(value)
        for value in pytree.tree_leaves(ordinary)
        if torch.is_tensor(value)
    )
    expected, _ = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        steps=STEPS,
        draws=ordinary,
    )
    _run(model, "longrun_checkpointed", dt, draws=ordinary, **options)
    _close(functional.extract(model).state, expected)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize(
    ("method", "inference"),
    (("run", False), ("longrun", False), ("longrun_checkpointed", True)),
)
def test_inference_draw_tapes_remain_supported_without_checkpoint_training(
    network, method, inference
):
    model, dt = _case(network)
    baseline, _ = _case(network)
    baseline.restore_dict_from_checkpoint(_snapshot(model.state_dict_for_checkpoint()))
    functional, tensors = dn.func.make_functional(model, dt=dt)
    sampled = functional.sample_runtime_draws(tensors.state, steps=STEPS)
    with torch.inference_mode():
        packet = pytree.tree_map_only(
            torch.Tensor, lambda value: value.clone(), sampled
        )
    with torch.inference_mode(inference):
        _run(model, method, dt, draws=packet)
        _run(baseline, method, dt, draws=sampled)
    _assert_packet_equal(
        model.state_dict_for_checkpoint(), baseline.state_dict_for_checkpoint()
    )


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_checkpoint_backward_rejects_changed_draw_tape_without_moving_live_state(
    network,
):
    model, dt = _case(network)
    functional, tensors = dn.func.make_functional(model, dt=dt)
    packet = _changed_draws(functional.sample_runtime_draws(tensors.state, steps=STEPS))
    callbacks = dn.func.make_callbacks(model, _recording(model, dt), dt=dt)
    result = _run(
        model, "longrun_checkpointed", dt, draws=packet, functional_callbacks=callbacks
    )
    frozen = _snapshot(model.state_dict_for_checkpoint())
    leaf = _base_draws(packet)[0]
    with torch.no_grad():
        leaf[-1].add_(0.25)
    with pytest.raises((ValueError, RuntimeError, dn.func.FunctionalizationError)):
        _loss(model, result).backward()
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)


@pytest.mark.parametrize("method", METHODS)
def test_netstim_tape_keeps_renewal_parameter_gradients_and_detached_draw_bases(method):
    source, model = _stim_network(), _stim_network()
    with torch.no_grad():
        source.netstim.next_stoch_time.zero_()
    model.restore_dict_from_checkpoint(_snapshot(source.state_dict_for_checkpoint()))
    functional, tensors = dn.func.make_functional_network(source)
    packet = _changed_draws(functional.sample_runtime_draws(tensors.state, steps=STEPS))
    packet.draws["netstim"] = torch.full_like(
        packet.draws["netstim"], -0.25
    ).requires_grad_()
    expected, _ = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        steps=STEPS,
        draws=packet,
    )
    _run(model, method, NETWORK_DT, draws=packet)
    _close(functional.extract(model).state, expected)
    frozen = _snapshot(model.state_dict_for_checkpoint())
    actual = torch.autograd.grad(
        model.netstim.next_stoch_time.sum(),
        (model.netstim.interval.rho, *_base_draws(packet)),
        allow_unused=True,
    )
    wanted = torch.autograd.grad(
        expected["netstim"]["next_stoch_time"].sum(),
        tensors.parameters["netstim"]["parameters"]["interval.rho"],
    )[0]
    _close(actual[0], wanted)
    assert wanted.isfinite().all() and wanted.abs().sum() > 0
    assert all(value is None or value.count_nonzero() == 0 for value in actual[1:])
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_partitioned_duration_uses_only_accepted_draw_frames_and_keeps_input_gradients(
    network,
):
    model, dt = _case(network)
    functional, tensors = dn.func.make_functional(model, dt=dt)
    packet = functional.sample_runtime_draws(tensors.state, steps=STEPS)
    inputs = _drives(model, STEPS)
    expected, _ = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        steps=STEPS,
        inputs=inputs,
        draws=packet,
    )
    for method, duration, start, stop in (
        ("longrun", 2.4 * dt, 0, 2),
        ("run", 0.3 * dt, 2, 2),
        ("longrun_checkpointed", 2.3 * dt, 2, 5),
    ):
        options = dict(
            duration=duration,
            inputs=_slice(inputs, start, stop),
            draws=_slice(packet, start, stop),
        )
        if not network:
            options["dt"] = dt
        if method != "run":
            options["chunklength"] = 2
        if method == "longrun_checkpointed":
            options["safe_checkpoint"] = True
        getattr(model, method)(**options)
    _close(functional.extract(model).state, expected)
    assert model._duration_remainder.item() == pytest.approx(0.0, abs=1e-12)
    leaves = tuple(
        value for value in pytree.tree_leaves(inputs) if torch.is_tensor(value)
    )
    wanted = torch.autograd.grad(_state_loss(expected, network), leaves)
    frozen = _snapshot(model.state_dict_for_checkpoint())
    _close(torch.autograd.grad(_loss(model), leaves), wanted)
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)


def _bank_case(network):
    if network:
        case = _network_banks(cross_banks=True)
        return case.model, case.extra, case.generator, _gradient_sources(case)
    model, extra, _pulse, generator = _population_banks("custom", "extra")
    extra[0].requires_grad_()
    return model, extra, generator, (*_amplitudes(_root(model, extra)), extra[0])


class _MeanGrid(Waveform):
    FUNCTIONAL_PURE = True

    def __init__(self, amp):
        super().__init__()
        self.amp = torch.nn.Parameter(torch.as_tensor(amp, dtype=torch.float64))

    def fn(self, time):
        return self.amp * (1.0 + time.mean()).expand_as(time)


def _mean_grid_case(network, *, poisson=False):
    populations, extra, waveforms = {}, {}, {}
    for name in ("left", "right") if network else ("cell",):
        cell, _ = _drive_population()
        cell.insert(_SharedLeak.rename("sde"))
        intra, temporal = _MeanGrid(0.2e-9), _MeanGrid(0.7)
        cell[:, 1].inject(intra)
        if poisson and name == "left":
            cell[:, 1].inject(
                _waveform(generator=torch.Generator().manual_seed(619), amp=0.1e-9)
            )
        cell.initialize()
        cell.integrator._initialize(cell, cell.t.new_tensor(POISSON_DT), force=True)
        field = torch.linspace(-0.2, 0.3, cell.v.numel(), dtype=cell.dtype()).reshape(
            cell.shape
        )
        populations[name], extra[name], waveforms[name] = (
            cell,
            (field, temporal),
            (intra, temporal),
        )
    model = dn.Network(populations) if network else populations["cell"]
    if network:
        model.train()
        model.initialize(POISSON_DT)
    return model, extra if network else extra["cell"], waveforms


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("method", METHODS)
def test_noise_only_tapes_preserve_deterministic_waveform_grid_semantics(
    network, method, monkeypatch
):
    source, source_extra, _ = _mean_grid_case(network)
    model, extra, waveforms = _mean_grid_case(network)
    baseline, baseline_extra, baseline_waveforms = _mean_grid_case(network)
    saved = _snapshot(source.state_dict_for_checkpoint())
    model.restore_dict_from_checkpoint(saved)
    baseline.restore_dict_from_checkpoint(saved)
    functional, tensors = dn.func.make_functional(
        source, dt=POISSON_DT, extra=source_extra
    )
    packet = functional.sample_runtime_draws(tensors.state, steps=STEPS)
    owned = {id(waveform) for pair in waveforms.values() for waveform in pair}
    observed = []
    original = _MeanGrid.fn

    def observe(self, time):
        if id(self) in owned:
            observed.append(time.numel())
        return original(self, time)

    with monkeypatch.context() as guard:
        guard.setattr(_MeanGrid, "fn", observe)
        _run(model, method, POISSON_DT, extra=extra, draws=packet)
    _run(baseline, method, POISSON_DT, extra=baseline_extra)
    _assert_packet_equal(
        model.state_dict_for_checkpoint(), baseline.state_dict_for_checkpoint()
    )
    counts = (
        [STEPS] * (2 * len(waveforms))
        if method == "run"
        else [3] * (2 * len(waveforms)) + [2] * (2 * len(waveforms))
    )
    assert sorted(observed) == sorted(counts)
    targets = tuple(waveform.amp for pair in waveforms.values() for waveform in pair)
    references = tuple(
        waveform.amp for pair in baseline_waveforms.values() for waveform in pair
    )
    frozen = _snapshot(model.state_dict_for_checkpoint())
    wanted = torch.autograd.grad(_loss(baseline), references)
    _close(torch.autograd.grad(_loss(model), targets), wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in wanted)
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)


def test_network_tape_preserves_deterministic_banks_per_population(monkeypatch):
    source, source_extra, _ = _mean_grid_case(True, poisson=True)
    model, extra, waveforms = _mean_grid_case(True, poisson=True)
    baseline, baseline_extra, _ = _mean_grid_case(True, poisson=True)
    oracle, oracle_extra, _ = _mean_grid_case(True, poisson=True)
    saved = _snapshot(source.state_dict_for_checkpoint())
    for target in (model, baseline, oracle):
        target.restore_dict_from_checkpoint(saved)
    functional, tensors = dn.func.make_functional_network(source, extra=source_extra)
    packet = functional.sample_runtime_draws(tensors.state, steps=STEPS)
    names = {
        id(waveform): name for name, pair in waveforms.items() for waveform in pair
    }
    observed = {name: [] for name in waveforms}
    original = _MeanGrid.fn

    def observe(self, time):
        if id(self) in names:
            observed[names[id(self)]].append(time.numel())
        return original(self, time)

    with monkeypatch.context() as guard:
        guard.setattr(_MeanGrid, "fn", observe)
        model.run(steps=STEPS, extra=extra, draws=packet)
    baseline.run(steps=STEPS, extra=baseline_extra)
    for index in range(STEPS):
        oracle.step(extra=oracle_extra, draws=_at(packet, index))
    assert observed["left"] == [1] * (2 * STEPS)
    assert observed["right"] == [STEPS, STEPS]
    _close(model.left.v, oracle.left.v)
    _close(model.right.v, baseline.right.v)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("method", METHODS)
def test_poisson_tapes_replay_every_accepted_step_independently_of_native_chunk_width(
    network, method, monkeypatch
):
    source, source_extra, _source_generator, references = _bank_case(network)
    model, extra, generator, targets = _bank_case(network)
    oracle, oracle_extra, _oracle_generator, _ = _bank_case(network)
    functional, tensors = dn.func.make_functional(
        source, dt=POISSON_DT, extra=source_extra
    )
    sampled = functional.sample_runtime_draws(tensors.state, steps=STEPS)
    packet = _changed_packet(sampled) if network else _changed_uniforms(sampled)
    callbacks = dn.func.make_callbacks(
        model, _recording(model, POISSON_DT), dt=POISSON_DT
    )
    expected, auxiliary = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        steps=STEPS,
        draws=packet,
        callbacks=functional.make_callbacks(_recording(source, POISSON_DT)),
    )
    before = _snapshot(packet)
    with monkeypatch.context() as guard:
        _forbid_owned_sampling(guard, model, generator)
        result = _run(
            model,
            method,
            POISSON_DT,
            draws=packet,
            extra=extra,
            functional_callbacks=callbacks,
        )
    for index in range(STEPS):
        options = dict(extra=oracle_extra, draws=_at(packet, index))
        if not network:
            options["dt"] = POISSON_DT
        oracle.step(**options)
    _close(_native_state(model, extra), expected)
    _close(_native_state(oracle, oracle_extra), expected)
    _close(result.outputs, auxiliary["callbacks"].outputs)
    frozen = _snapshot(model.state_dict_for_checkpoint())
    wanted = torch.autograd.grad(
        _state_loss(expected, network, auxiliary["callbacks"]),
        _functional_sources(functional, tensors, references),
    )
    actual = torch.autograd.grad(
        _loss(model, result), (*targets, *_base_draws(packet)), allow_unused=True
    )
    _close(actual[: len(wanted)], wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in wanted)
    assert all(
        value is None or value.count_nonzero() == 0 for value in actual[len(wanted) :]
    )
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
    _assert_packet_equal(packet, before)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("method", METHODS)
def test_empty_explicit_poisson_tapes_do_not_invoke_banks_or_consume_pending_rng(
    network, method, monkeypatch
):
    model, extra, generator, _targets = _bank_case(network)
    functional, tensors = dn.func.make_functional(model, dt=POISSON_DT, extra=extra)
    packet = functional.sample_runtime_draws(tensors.state, steps=0)
    saved = _snapshot(model.state_dict_for_checkpoint())
    global_rng, stream = torch.get_rng_state().clone(), generator.get_state().clone()
    with monkeypatch.context() as guard:
        _forbid_owned_sampling(guard, model, generator)
        _run(model, method, POISSON_DT, steps=0, extra=extra, draws=packet)
    _assert_packet_equal(model.state_dict_for_checkpoint(), saved)
    _close(generator.get_state(), stream)
    _close(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize("method", METHODS)
def test_late_invalid_bank_uniform_rejects_before_rebuilding_lazy_intra(method):
    model, extra, pulse, generator = _population_banks("repeated", "intra")
    functional, tensors = dn.func.make_functional(model, dt=POISSON_DT, extra=extra)
    packet = _snapshot(functional.sample_runtime_draws(tensors.state, steps=STEPS))
    next(iter(packet.draws["stimulation"].values()))[-1].fill_(torch.nan)
    model[:, 0].inject(dn.constant(value=0.01e-9))
    assert model.intra is None
    saved, before = _capture(model), generator.get_state().clone()
    with pytest.raises((ValueError, RuntimeError, dn.func.FunctionalizationError)):
        _run(model, method, POISSON_DT, extra=extra, draws=packet)
    assert model.intra is None
    _unchanged(model, saved)
    _close(generator.get_state(), before)
    assert pulse.generator is generator


@pytest.mark.parametrize("method", METHODS)
def test_omitted_draws_retain_existing_chunkwise_poisson_invocation_cadence(
    method, monkeypatch
):
    model, extra, pulse, _generator = _population_banks("repeated", "intra")
    original = type(pulse).fn
    observed = []

    def count(self, time):
        if self is pulse:
            observed.append(time.shape[-1])
        return original(self, time)

    with monkeypatch.context() as guard:
        guard.setattr(type(pulse), "fn", count)
        _run(model, method, POISSON_DT, extra=extra)
    expected = [STEPS, STEPS] if method == "run" else [3, 3, 2, 2]
    assert observed == expected


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_actual_mps_heun_draw_tape_matches_functional_values_and_checkpoint_gradients():
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DEVICE="mps", DTYPE=torch.float32):
        source, model = _state_population(_HeunLeak), _state_population(_HeunLeak)
        streams = {
            id(stream): stream
            for state in source.mech.sde.DE.values()
            for name in state._sde_rng_names
            for stream in (getattr(state, name),)
        }
        source_rng = tuple(
            (
                stream,
                _snapshot(stream.get_extra_state()),
                stream._cpu_gen,
                dict(stream._device_gens),
            )
            for stream in streams.values()
        )
        # Initialize the ambient MPS stream before checking that lowering and
        # sampling only materialize their private indexed device route.
        torch.mps.get_rng_state()
        cpu_rng = torch.get_rng_state().clone()
        mps_rng = torch.mps.get_rng_state().clone()
        functional, tensors = dn.func.make_functional(source, dt=STATE_DT)
        packet = functional.sample_runtime_draws(tensors.state, steps=STEPS)
        for stream, saved, cpu_generator, device_generators in source_rng:
            _assert_packet_equal(stream.get_extra_state(), saved)
            assert stream._cpu_gen is cpu_generator
            assert stream._device_gens.keys() == device_generators.keys()
            assert all(
                stream._device_gens[device] is generator
                for device, generator in device_generators.items()
            )
        _close(torch.get_rng_state(), cpu_rng)
        _close(torch.mps.get_rng_state(), mps_rng)
        expected, _ = dn.func.run(
            functional,
            dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
            steps=STEPS,
            draws=packet,
        )
        _run(model, "longrun_checkpointed", STATE_DT, draws=packet, chunklength=2)
        torch.testing.assert_close(
            model.v, expected["integrator"]["v"], rtol=2e-5, atol=2e-6
        )
        sigma = next(iter(model.mech.sde.DE.values())).sigma_param
        wanted = torch.autograd.grad(
            expected["integrator"]["v"].square().mean(),
            _functional_sources(
                functional,
                tensors,
                (next(iter(source.mech.sde.DE.values())).sigma_param,),
            ),
        )[0]
        actual = torch.autograd.grad(model.v.square().mean(), sigma)[0]
        torch.testing.assert_close(actual, wanted, rtol=2e-5, atol=2e-6)
        assert actual.isfinite().all() and actual.abs().sum() > 0
