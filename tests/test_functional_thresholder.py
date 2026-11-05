"""Compiled amplitude trials share the native threshold-search contract."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.instruments import Thresholder
from dendra.models.mod import pas
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _module_snapshot,
)

DT = 0.01
STEPS = 8
DURATION = DT * STEPS
REST = -65.0
CUTOFF = -64.0
ATOL = 1.0e-4

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False
            ),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _model(*, batch=()):
    model = dn.Unmyelinated(
        [2.0, 2.0],
        L=4.0,
        dx=1.0,
        v_init=REST,
        dtype=torch.float64,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    model.insert(pas, g=1.0e-4, e=REST)
    for size in batch:
        model.batch(size)
    model.initialize()
    model.train()
    return model


def _field(model):
    profile = torch.linspace(-10.0, 10.0, model.nc, dtype=model.dtype())
    lanes = torch.arange(
        1,
        model.v.numel() // model.nc + 1,
        dtype=model.dtype(),
    ).reshape(*model.shape[:-1], 1)
    return lanes * profile


def _waveform():
    return dn.mono_rect(amp=1.0, delay=0.0, pw=1.0)


def _active():
    return dn.callbacks.Active(threshold=CUTOFF, node_check=[0, -1])


def _runner(
    model,
    *,
    callbacks=None,
    extra=True,
    execution="scan",
    initialization="snapshot",
    backend="aot_eager",
):
    return dn.compile(
        model,
        dt=DT,
        duration=DURATION,
        extra=dn.VarInput() if extra else None,
        chunk_steps=STEPS,
        callbacks=(
            {"activation": _active(), "trace": dn.func.Recorder(("v", "t"))}
            if callbacks is None
            else callbacks
        ),
        initialization=initialization,
        execution=execution,
        backend=backend,
        inference=True,
    )


def _clone(value):
    return pytree.tree_map_only(torch.Tensor, lambda x: x.detach().clone(), value)


def _close(actual, expected):
    leaves, spec = pytree.tree_flatten(actual)
    wanted, other = pytree.tree_flatten(expected)
    assert spec == other
    for value, reference in zip(leaves, wanted, strict=True):
        if torch.is_tensor(value):
            torch.testing.assert_close(value, reference, rtol=2e-10, atol=2e-11)
        else:
            assert value == reference


def _frames(program):
    return dict(program.parameters), dict(program.constants), program.initial_state


def _gain_from_trace(result):
    voltage = result[1]["callbacks"]["trace"]["v"]
    selected = voltage.index_select(-1, torch.tensor([0, voltage.shape[-1] - 1]))
    gain = selected.amax(dim=(0, -1)) - REST
    assert torch.isfinite(gain).all() and (gain > 0).all()
    return gain


def _assert_bracket(bounds, expected):
    upper, lower = bounds
    assert upper.device.type == lower.device.type == "cpu"
    assert tuple(upper.shape) == tuple(lower.shape) == tuple(expected.shape)
    assert torch.isfinite(upper).all() and torch.isfinite(lower).all()
    assert torch.all(upper >= expected - 1e-10)
    assert torch.all(lower <= expected + 1e-10)
    assert torch.all(upper - lower < ATOL)


@pytest.mark.parametrize("execution", ("scan", "while"))
@pytest.mark.parametrize("batch", ((), (2,)))
def test_compiled_search_matches_linear_response_and_native_independent_lanes(
    execution, batch
):
    model = _model(batch=batch)
    field, waveform, criterion = _field(model), _waveform(), _active()
    program = _runner(
        model,
        execution=execution,
        callbacks={
            "activation": criterion,
            "trace": dn.func.Recorder(("v", "t")),
        },
    )
    source = _module_snapshot(model)
    frames = _clone(_frames(program))
    field_before = field.clone()
    rng = torch.get_rng_state().clone()
    # The passive cable is linear in extracellular amplitude. This independent
    # unit-amplitude trace determines the exact crossing amplitude per lane.
    expected = (CUTOFF - REST) / _gain_from_trace(program(extra=(field, waveform)))
    search = Thresholder(
        program,
        active="activation",
        space=field,
        time=waveform,
        ub=2 * expected,
        atol=ATOL,
    )
    actual = search.calculate_thresholds()
    _assert_bracket(actual, expected)
    _close(search.calculate_thresholds(), actual)
    # A previous active amplitude must not contaminate the lower-bound trial,
    # neural state or callback carry in either repeated search.
    assert criterion.record is criterion.state_cache is None and criterion.i == 0
    assert criterion.node_check == [0, -1]
    _assert_module_snapshot(model, source)
    _close(_frames(program), frames)
    torch.testing.assert_close(field, field_before, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    native_model = _model(batch=batch)
    native = Thresholder(
        native_model,
        active=_active(),
        space=field,
        time=_waveform(),
        ub=2 * expected,
        atol=ATOL,
    )
    _close(native.calculate_thresholds(tstop=DURATION, dt=DT), actual)


def test_amplitude_trials_reuse_the_warmed_compiler_capture():
    from torch._dynamo.backends.registry import lookup_backend

    captures = []
    aot_eager = lookup_backend("aot_eager")

    def backend(graph, examples):
        captures.append(graph)
        return aot_eager(graph, examples)

    model = _model()
    field, waveform = _field(model), _waveform()
    program = _runner(model, backend=backend)
    expected = (CUTOFF - REST) / _gain_from_trace(program(extra=(field, waveform)))
    before = len(captures)
    assert before > 0
    search = Thresholder(
        program,
        active="activation",
        space=field,
        time=waveform,
        ub=2 * expected,
        atol=ATOL,
    )
    _assert_bracket(search.calculate_thresholds(), expected)
    _assert_bracket(search.calculate_thresholds(reset_bounds=False), expected)
    assert len(captures) == before


def test_compiled_bases_use_their_time_axis_and_preserve_nan_lane_handling():
    model = _model()
    model.t.fill_(0.175)
    program = _runner(model, extra=False)
    bases = _field(model).unsqueeze(0).expand(6, *model.shape).clone()
    expected = (CUTOFF - REST) / _gain_from_trace(
        program(inputs=dn.func.Inputs(ve=bases), steps=len(bases))
    )
    search = Thresholder(
        program, active="activation", bases=bases, ub=2 * expected, atol=ATOL
    )
    # Bases retain the native contract: their own time axis supplies the span;
    # an unrelated absolute stop does not truncate or pad them.
    for unrelated_stop in (100.0, 0.1, -1.0):
        _assert_bracket(
            search.calculate_thresholds(tstop=unrelated_stop, dt=DT), expected
        )
    with_nan = bases.clone()
    with_nan[1, 0, 0] = torch.nan
    ignored = Thresholder(
        program, active="activation", bases=with_nan, ub=2 * expected, atol=ATOL
    )
    upper, lower = ignored.calculate_thresholds()
    assert torch.isnan(upper[0]) and torch.isnan(lower[0])
    assert torch.isfinite(upper[1]) and torch.isfinite(lower[1])
    assert bool(ignored.ignore[0]) and not bool(ignored.ignore[1])
    assert torch.isnan(with_nan[1, 0, 0])


def test_compiled_absolute_stop_override_uses_the_snapshot_clock():
    model = _model()
    model.t.fill_(0.175)
    program = _runner(model)
    field, waveform = _field(model), _waveform()
    start = float(program.initial_state["clock"]["t"])
    stop = start + 4 * DT
    expected = (CUTOFF - REST) / _gain_from_trace(
        program(extra=(field, waveform), duration=stop - start)
    )
    search = Thresholder(
        program,
        active="activation",
        space=field,
        time=waveform,
        ub=2 * expected,
        atol=ATOL,
    )
    _assert_bracket(search.calculate_thresholds(tstop=stop, dt=DT), expected)
    with pytest.raises(ValueError, match="dt|timestep"):
        search.calculate_thresholds(dt=2 * DT)
    with pytest.raises(ValueError, match="stop|start|time"):
        search.calculate_thresholds(tstop=start - DT)


def test_compiled_fresh_policy_ignores_the_edited_restart_snapshot():
    model = _model()
    field, waveform = _field(model), _waveform()
    programs = {
        policy: _runner(model, initialization=policy)
        for policy in ("snapshot", "fresh")
    }
    expected = (CUTOFF - REST) / _gain_from_trace(
        programs["fresh"](extra=(field, waveform))
    )
    for program in programs.values():
        program.initial_state["integrator"]["v"].fill_(CUTOFF + 1)
    searches = {
        policy: Thresholder(
            program,
            active="activation",
            space=field,
            time=waveform,
            ub=2 * expected,
            atol=ATOL,
        )
        for policy, program in programs.items()
    }
    with pytest.raises(RuntimeError, match="lower bounds are active"):
        searches["snapshot"].calculate_thresholds()
    _assert_bracket(searches["fresh"].calculate_thresholds(), expected)


@dataclass(frozen=True)
class _BandActivity(dn.func.FunctionalCallback):
    lower: float = CUTOFF
    upper: float = CUTOFF + 1

    def initialize(self, state, auxiliary):
        del auxiliary
        return state["integrator"]["v"].amax(dim=-1), None

    def update(self, carry, state, auxiliary):
        del auxiliary
        return torch.maximum(carry, state["integrator"]["v"].amax(dim=-1)), None

    def finalize(self, carry, emissions):
        assert emissions is None
        return (carry >= self.lower) & (carry < self.upper)


class _PeakVoltage(_BandActivity):
    """Retain one floating peak per lane without emitting a trajectory."""

    def finalize(self, carry, emissions):
        assert emissions is None
        return carry


@pytest.mark.parametrize("peak_kind", ("recorder", "reduction"))
def test_custom_boolean_result_and_peak_metric_support_block_bound_repair(peak_kind):
    model = _model()
    field, waveform = _field(model), _waveform()
    program = _runner(
        model,
        callbacks={
            "activation": _BandActivity(),
            "peak_voltage": (
                dn.func.Recorder(("v",), max_only=True)
                if peak_kind == "recorder"
                else _PeakVoltage()
            ),
            "trace": dn.func.Recorder(("v", "t")),
        },
    )
    expected = (CUTOFF - REST) / _gain_from_trace(program(extra=(field, waveform)))
    search = Thresholder(
        program,
        active="activation",
        peak="peak_voltage",
        block_threshold=CUTOFF,
        space=field,
        time=waveform,
        # The initial upper bound is above the custom activity band. The block
        # diagnostic must shrink it, then recover a valid active bracket.
        ub=4 * expected,
        atol=ATOL,
        max_tries_bound_fix=10,
    )
    _assert_bracket(search.calculate_thresholds(block_possible=True), expected)


def test_snapshot_trials_restart_bound_poisson_rng_and_custom_activity():
    from tests.func.test_poisson_stimulation import _population

    class ObservedRunner(dn.func.CompiledRunner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.observations = []

        def __call__(self, *args, **kwargs):
            # Observation happens after the ordinary public execution boundary;
            # the simulation, callback and sampling inputs remain unchanged.
            state, auxiliary = super().__call__(*args, **kwargs)
            self.observations.append(
                _clone(
                    (
                        state,
                        dict(auxiliary["callbacks"]["trace"]),
                        auxiliary["callbacks"]["signal"],
                    )
                )
            )
            return state, auxiliary

    model, pulse, _ = _population("intra")
    field, waveform = _field(model), _waveform()
    program = ObservedRunner(
        model,
        dt=DT,
        duration=DURATION,
        extra=dn.VarInput(),
        chunk_steps=STEPS,
        inference=True,
        initialization="snapshot",
        backend="aot_eager",
        callbacks={
            "signal": _BandActivity(),
            "trace": dn.func.Recorder(("v", "t")),
        },
    )
    before = _clone(_frames(program))
    source = _module_snapshot(model)
    source_rng = pulse.generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    low, high = model.v.new_zeros(model.shape[:-1]), model.v.new_ones(model.shape[:-1])
    for amplitude in (low, high):
        program(extra=(field * amplitude.unsqueeze(-1), waveform))
    reference_low, reference_high = program.observations
    program.observations = []
    search = Thresholder(
        program,
        active="signal",
        space=field,
        time=waveform,
        ub=1.0,
        atol=ATOL,
    )
    first = search.check_active(DURATION, DT, low)
    other = search.check_active(DURATION, DT, high)
    repeated = search.check_active(DURATION, DT, low)
    _close(
        (first, other, repeated),
        (reference_low[2], reference_high[2], reference_low[2]),
    )
    assert len(program.observations) == 3
    for actual, expected in zip(
        program.observations,
        (reference_low, reference_high, reference_low),
        strict=True,
    ):
        _close(actual, expected)
    assert not torch.equal(reference_low[1]["v"], reference_high[1]["v"])
    initial_rng = before[2]["stimulation"]["rng"]
    final_rng = reference_low[0]["stimulation"]["rng"]
    assert initial_rng and initial_rng.keys() == final_rng.keys()
    assert any(
        not torch.equal(frame, initial_rng[name]) for name, frame in final_rng.items()
    )
    # Replaying an amplitude restarts both neural state and the entire owned
    # draw stream; neither the source generator nor the restart is consumed.
    _close(_frames(program), before)
    _assert_module_snapshot(model, source)
    torch.testing.assert_close(pulse.generator.get_state(), source_rng, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)


def test_compiled_activity_validation_and_missing_block_diagnostics():
    model = _model()
    field, waveform = _field(model), _waveform()
    program = _runner(model)
    options = dict(space=field, time=waveform, ub=2.0, atol=ATOL)
    with pytest.raises(TypeError, match="name|named|string"):
        Thresholder(program, active=_active(), **options)
    with pytest.raises((KeyError, ValueError), match="missing"):
        Thresholder(program, active="missing", **options)
    with pytest.raises(ValueError, match="ub|upper"):
        Thresholder(program, active="activation", space=field, time=waveform, atol=ATOL)
    search = Thresholder(program, active="activation", **options)
    with pytest.raises(ValueError, match="peak|block"):
        search.calculate_thresholds(block_possible=True)

    counts = _runner(
        model,
        callbacks={"activation": dn.func.APCount(threshold=CUTOFF, node_check=[0])},
    )
    with pytest.raises(TypeError, match="[Bb]oolean"):
        Thresholder(counts, active="activation", **options).calculate_thresholds()


def test_compiled_stimulation_contract_and_dtype_changes_fail_before_mutation():
    model = _model()
    field, waveform = _field(model), _waveform()
    runtime = _runner(model)
    fixed = _runner(model, extra=False)
    options = dict(active="activation", ub=2.0, atol=ATOL)
    with pytest.raises((TypeError, ValueError), match="VarInput|runtime"):
        Thresholder(fixed, space=field, time=waveform, **options)
    with pytest.raises((TypeError, ValueError), match="bases|extra|VarInput"):
        Thresholder(
            runtime, bases=field.unsqueeze(0).expand(STEPS, *model.shape), **options
        )
    search = Thresholder(runtime, space=field, time=waveform, **options)
    before = _clone(_frames(runtime))
    assert search.double() is search
    with pytest.raises(TypeError, match="dtype|compile|precision"):
        search.float()
    _close(_frames(runtime), before)
    assert search.ub.dtype == torch.float64 and search.space.dtype == torch.float64


def test_execution_options_and_peak_cadence_are_validated_before_trials():
    model = _model()
    field, waveform = _field(model), _waveform()
    options = dict(active="activation", space=field, time=waveform, ub=2.0, atol=ATOL)
    with pytest.raises((TypeError, ValueError), match="chunk_steps|chunklength"):
        Thresholder(_runner(model), chunklength=4, **options)
    cadenced = _runner(
        model,
        callbacks={
            "activation": _active(),
            "peak_voltage": dn.func.Recorder(("v",), max_only=True, dt=2 * DT),
        },
    )
    with pytest.raises(ValueError, match="cadence|dt|solver|stride|every"):
        Thresholder(cadenced, peak="peak_voltage", **options)
    native = dict(options)
    native["active"] = _active()
    with pytest.raises((TypeError, ValueError), match="peak|compiled"):
        Thresholder(model, peak="peak_voltage", **native)
    with pytest.raises((TypeError, ValueError), match="block_threshold|compiled"):
        Thresholder(model, block_threshold=CUTOFF, **native)


@dataclass(frozen=True)
class _CompartmentActivity(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        del auxiliary
        return state["integrator"]["v"], None

    def update(self, carry, state, auxiliary):
        del carry, auxiliary
        return state["integrator"]["v"], None

    def finalize(self, carry, emissions):
        assert emissions is None
        return carry >= CUTOFF


def test_custom_activity_must_return_exactly_one_boolean_per_search_lane():
    model = _model()
    program = _runner(model, callbacks={"activation": _CompartmentActivity()})
    search = Thresholder(
        program,
        active="activation",
        space=_field(model),
        time=_waveform(),
        ub=2.0,
        atol=ATOL,
    )
    with pytest.raises(ValueError, match="shape"):
        search.calculate_thresholds()


def test_compiled_partitions_must_match_prepared_activity_and_peak_configuration():
    model = _model()
    field, waveform = _field(model), _waveform()
    callbacks = {
        "activation": dn.func.Active(
            threshold=CUTOFF, node_check=[0, -1], partition=(1, 1)
        ),
        "peak_voltage": dn.func.Recorder(
            ("v",), max_only=True, partition=(2, model.nc - 2)
        ),
    }
    program = _runner(model, callbacks=callbacks)
    search = Thresholder(
        program,
        active="activation",
        peak="peak_voltage",
        space=field,
        time=waveform,
        ub=2.0,
        atol=ATOL,
    )
    before = search.ub.clone()
    with pytest.raises(ValueError, match="partition"):
        search.set_partition((2, model.nc - 2), active_partition=(2, 0))
    torch.testing.assert_close(search.ub, before, rtol=0, atol=0)
    search.set_partition((2, model.nc - 2), active_partition=(1, 1))
    assert search.model_partition == (2, model.nc - 2)
    assert search.active_partition == (1, 1)
    assert search.ub.shape == (*model.shape[:-1], 2)

    unpartitioned = Thresholder(
        _runner(model),
        active="activation",
        space=field,
        time=waveform,
        ub=2.0,
        atol=ATOL,
    )
    before = unpartitioned.ub.clone()
    with pytest.raises(ValueError, match="partition"):
        unpartitioned.set_partition((2, model.nc - 2), active_partition=(1, 1))
    torch.testing.assert_close(unpartitioned.ub, before, rtol=0, atol=0)


def test_compiled_network_requires_an_explicit_network_threshold_design():
    network = dn.Network({"cell": _model()})
    network.initialize(DT)
    program = dn.compile(
        network, dt=DT, duration=DURATION, backend="aot_eager", inference=True
    )
    with pytest.raises(TypeError, match="Network|Population"):
        Thresholder(
            program,
            active="activation",
            bases=torch.ones(STEPS, 2, 5),
            ub=2.0,
            atol=ATOL,
        )
