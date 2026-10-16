"""Run/longrun dispatch shortcuts preserve observable mutable execution."""

import functools
from collections import Counter
from types import MethodType

import pytest
import torch

import dendra as dn
import dendra.models.core as population_core
import dendra.models.integrators.tree as tree_module
from dendra.models.callbacks import Callback
from dendra.models.integrators.core import Integrator, _DeferredIntra, _NoiseFreeStep
from dendra.models.mechanisms import Mechanism
from dendra.models.mechanisms._state import State
from tests.test_population_fused_intra import (
    DT,
    _case,
    _count_handler_noise_calls,
    _counting_compiler,
    _NoopPreStep,
    _run,
)

pytestmark = pytest.mark.cpu


def _execute(model, entrypoint, *, steps=8, callbacks=(), chunklength=3):
    recorder = dn.Recorder(["v"])
    with dn.ctx(
        JIT=1,
        BACKEND="eager",
        FULLGRAPH=1,
        REQUIRE_GRAD=int(model.training),
        DEVICE="cpu",
        DTYPE=model.dtype(),
    ):
        kwargs = dict(tstop=steps * DT, dt=DT, callbacks=[recorder, *callbacks])
        if entrypoint == "longrun":
            kwargs["chunklength"] = chunklength
        getattr(model, entrypoint)(**kwargs)
    return recorder.stack("v")


def _without_intra(**kwargs):
    model, amplitude = _case(**kwargs)
    model.delete_injections()
    return model, amplitude


def _assert_no_payload(model):
    for module in model.modules():
        assert not any(
            isinstance(value, (_DeferredIntra, _NoiseFreeStep))
            for value in vars(module).values()
        ), f"Transient execution payload retained by {type(module).__name__}"


@pytest.mark.parametrize("training", [False, True])
def test_longrun_fuses_every_step_and_matches_run_trace_and_gradient(
    monkeypatch, training
):
    candidate, amplitude = _case(training=training)
    reference, reference_amplitude = _case(training=training)
    calls, _ = _counting_compiler(monkeypatch)
    with _count_handler_noise_calls() as noise_calls:
        actual = _execute(candidate, "longrun")
    assert Counter(calls) == {"_step_with_intra": 8}
    assert not noise_calls
    calls.clear()
    expected = _run(reference, callbacks=[_NoopPreStep()])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name in ("m", "h", "n"):
        torch.testing.assert_close(
            getattr(candidate.mech.hh, name),
            getattr(reference.mech.hh, name),
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(candidate.t, reference.t, rtol=0, atol=0)
    if training:
        actual_gradient = torch.autograd.grad(actual.square().mean(), amplitude)[0]
        expected_gradient = torch.autograd.grad(
            expected.square().mean(), reference_amplitude
        )[0]
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
        assert torch.all(actual_gradient != 0)
    _assert_no_payload(candidate)


@pytest.mark.parametrize("entrypoint", ["run", "longrun"])
def test_no_stimulus_run_skips_only_empty_noise_walk(monkeypatch, entrypoint):
    candidate, _ = _without_intra()
    reference, _ = _without_intra()
    calls, _ = _counting_compiler(monkeypatch)
    with _count_handler_noise_calls() as fast_calls:
        actual = _execute(candidate, entrypoint)
    assert not fast_calls
    assert Counter(calls) == {"_step": 8}
    with _count_handler_noise_calls() as ordinary_calls:
        expected = _execute(reference, entrypoint, callbacks=[_NoopPreStep()])
    assert len(ordinary_calls) == 8
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _assert_no_payload(candidate)
    # The optimization belongs to this execution call, not to the integrator.
    with _count_handler_noise_calls() as later_step_calls:
        candidate.step(dt=DT)
    assert len(later_step_calls) == 1
    _assert_no_payload(candidate)


@pytest.mark.skipif(
    not callable(getattr(tree_module.dendra_solvers, "DHSPlan", None)),
    reason="requires the owning-plan native solver candidate",
)
def test_compiled_no_stimulus_dhs_trace_and_voltage_gradient():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1871)
            candidate, _ = _without_intra(
                batch=2,
                dtype=torch.float64,
                training=True,
                tree=True,
            )
            torch.manual_seed(1871)
            reference, _ = _without_intra(
                batch=2,
                dtype=torch.float64,
                training=True,
                tree=True,
            )
        candidate.v = candidate.v.detach().requires_grad_(True)
        reference.v = reference.v.detach().requires_grad_(True)
        candidate_initial_v = candidate.v
        reference_initial_v = reference.v
        with _count_handler_noise_calls() as candidate_calls:
            actual = _run(candidate, backend="inductor")
        assert not candidate_calls
        with _count_handler_noise_calls() as reference_calls:
            expected = _run(reference, backend="inductor", callbacks=[_NoopPreStep()])
        assert len(reference_calls) == 8
        assert actual.shape == expected.shape == (9, *candidate.shape)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for name in ("m", "h", "n"):
            torch.testing.assert_close(
                getattr(candidate.mech.hh, name),
                getattr(reference.mech.hh, name),
                rtol=0,
                atol=0,
            )
        actual_gradient = torch.autograd.grad(
            actual.square().mean(), candidate_initial_v
        )[0]
        expected_gradient = torch.autograd.grad(
            expected.square().mean(), reference_initial_v
        )[0]
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
        assert torch.isfinite(actual_gradient).all()
        assert torch.any(actual_gradient != 0)
        for model in (candidate, reference):
            assert isinstance(
                model.integrator._cpu_topology_plan, tree_module.dendra_solvers.DHSPlan
            )
            assert all(
                not torch.is_inference(getattr(model.integrator, name))
                for name in ("parent_idx", "order", "layer_ptr")
            )
            _assert_no_payload(model)
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("entrypoint", ["run", "longrun"])
@pytest.mark.parametrize("level", ["handler", "mechanism", "state"])
def test_no_stimulus_custom_sampler_is_never_skipped(monkeypatch, entrypoint, level):
    candidate, _ = _without_intra()
    reference, _ = _without_intra()
    _counting_compiler(monkeypatch)
    mechanism = candidate.mech.hh
    owner = {
        "handler": candidate.mech,
        "mechanism": mechanism,
        "state": next(iter(mechanism.DE.values())),
    }[level]
    original = owner.sample_runtime_noises_
    sampled = []

    def sample(*args, **kwargs):
        sampled.append(candidate.t.detach().clone())
        return original(*args, **kwargs)

    owner.sample_runtime_noises_ = sample
    actual = _execute(candidate, entrypoint)
    expected = _execute(reference, entrypoint, callbacks=[_NoopPreStep()])
    assert len(sampled) == 8
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _assert_no_payload(candidate)


@pytest.mark.parametrize("entrypoint", ["run", "longrun"])
def test_no_stimulus_custom_step_receives_original_none_payload(
    monkeypatch, entrypoint
):
    model, _ = _without_intra()
    _counting_compiler(monkeypatch)
    original_step = model.integrator.step
    observed = []

    def custom_step(population, dt, ve=None, intra=None):
        assert intra is None, "Framework marker escaped into a custom step"
        observed.append(population.t.detach().clone())
        return original_step(population, dt, ve, intra)

    model.integrator.step = custom_step
    with _count_handler_noise_calls() as noise_calls:
        _execute(model, entrypoint)
    assert len(observed) == len(noise_calls) == 8
    _assert_no_payload(model)


@pytest.mark.parametrize("entrypoint", ["run", "longrun"])
def test_population_voltage_setter_cannot_cache_empty_noise_proof(
    monkeypatch, entrypoint
):
    candidate, _ = _without_intra()
    reference, _ = _without_intra()
    _counting_compiler(monkeypatch)
    original_setattr = type(candidate).__setattr__
    original_sampler = candidate.mech.sample_runtime_noises_
    installed = []
    sampled = []

    def sample(*args, **kwargs):
        sampled.append(None)
        return original_sampler(*args, **kwargs)

    def setattr_(owner, name, value):
        result = original_setattr(owner, name, value)
        if owner is candidate and name == "v" and not installed:
            installed.append(True)
            candidate.mech.sample_runtime_noises_ = sample
        return result

    monkeypatch.setattr(type(candidate), "__setattr__", setattr_)
    with _count_handler_noise_calls() as handler_calls:
        actual = _execute(candidate, entrypoint)
    assert installed == [True]
    assert len(handler_calls) == 8
    assert len(sampled) == 7
    expected = _execute(reference, entrypoint, callbacks=[_NoopPreStep()])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _assert_no_payload(candidate)


@pytest.mark.parametrize("entrypoint", ["run", "longrun"])
@pytest.mark.parametrize("scope", ["instance", "class"])
def test_custom_tensor_step_cannot_cache_empty_noise_proof(
    monkeypatch, entrypoint, scope
):
    candidate, _ = _without_intra()
    reference, _ = _without_intra()
    _counting_compiler(monkeypatch)
    integrator = candidate.integrator
    original_step = type(integrator)._step
    original_sampler = candidate.mech.sample_runtime_noises_
    advanced = []
    sampled = []

    def sample(*args, **kwargs):
        sampled.append(None)
        return original_sampler(*args, **kwargs)

    def tensor_step(owner, *args, **kwargs):
        assert args[-1] is None, "Framework marker escaped into a custom tensor step"
        if owner is integrator:
            if not advanced:
                candidate.mech.sample_runtime_noises_ = sample
            advanced.append(None)
        return original_step(owner, *args, **kwargs)

    if scope == "class":
        monkeypatch.setattr(type(integrator), "_step", tensor_step)
    else:
        monkeypatch.setattr(integrator, "_step", MethodType(tensor_step, integrator))
    with _count_handler_noise_calls() as handler_calls:
        actual = _execute(candidate, entrypoint)
    assert len(advanced) == len(handler_calls) == 8
    assert len(sampled) == 7
    expected = _execute(reference, entrypoint, callbacks=[_NoopPreStep()])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _assert_no_payload(candidate)


@pytest.mark.parametrize("entrypoint", ["run", "longrun"])
def test_custom_voltage_helper_cannot_cache_empty_noise_proof(monkeypatch, entrypoint):
    candidate, _ = _without_intra()
    reference, _ = _without_intra()
    _counting_compiler(monkeypatch)
    original_update = candidate.integrator._voltage_update
    reference_update = reference.integrator._voltage_update
    original_sampler = candidate.mech.sample_runtime_noises_
    outer_step = candidate.integrator.step.__func__
    tensor_step = candidate.integrator._step.__func__
    advanced = []
    sampled = []

    def sample(*args, **kwargs):
        sampled.append(None)
        return original_sampler(*args, **kwargs)

    def update(*args, **kwargs):
        result = original_update(*args, **kwargs)
        if not advanced:
            candidate.mech.sample_runtime_noises_ = sample
        advanced.append(None)
        return result

    candidate.integrator._voltage_update = update
    # Both controls use the authored-helper current/RHS contract, so this
    # comparison isolates noise dispatch rather than affine assembly rounding.
    reference.integrator._voltage_update = lambda *args, **kwargs: reference_update(
        *args, **kwargs
    )
    assert candidate.integrator.step.__func__ is outer_step
    assert candidate.integrator._step.__func__ is tensor_step
    with _count_handler_noise_calls() as handler_calls:
        actual = _execute(candidate, entrypoint)
    assert len(advanced) == len(handler_calls) == 8
    assert len(sampled) == 7
    expected = _execute(reference, entrypoint, callbacks=[_NoopPreStep()])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _assert_no_payload(candidate)


@pytest.mark.parametrize("entrypoint", ["run", "longrun"])
@pytest.mark.parametrize("scope", ["instance", "class"])
def test_custom_fused_tensor_helper_keeps_split_dispatch(
    monkeypatch, entrypoint, scope
):
    candidate, _ = _case()
    calls, _ = _counting_compiler(monkeypatch)
    integrator = candidate.integrator
    original = Integrator._step_with_intra
    fused_calls = []

    def fused(owner, *args, **kwargs):
        fused_calls.append(None)
        return original(owner, *args, **kwargs)

    if scope == "class":
        monkeypatch.setattr(Integrator, "_step_with_intra", fused)
    else:
        monkeypatch.setattr(
            integrator, "_step_with_intra", MethodType(fused, integrator)
        )
    with _count_handler_noise_calls() as sampled:
        _execute(candidate, entrypoint)
    assert not fused_calls
    assert Counter(calls) == {"make_intra": 8, "_step": 8}
    assert len(sampled) == 8
    _assert_no_payload(candidate)


@pytest.mark.parametrize("entrypoint", ["run", "longrun"])
def test_authored_staticproperty_getter_cannot_cache_empty_noise_proof(
    monkeypatch, entrypoint
):
    candidate, _ = _without_intra()
    reference, _ = _without_intra()
    _, active = _counting_compiler(monkeypatch)
    mechanism = candidate.mech.hh
    gnabar = mechanism.gnabar
    original_sampler = candidate.mech.sample_runtime_noises_
    reads = []
    sampled = []
    installed = []

    def sample(*args, **kwargs):
        sampled.append(None)
        return original_sampler(*args, **kwargs)

    def getter():
        reads.append(None)
        # Arm the replacement only inside actual transition execution, not
        # preparatory reads. The proof itself must never evaluate this getter.
        if active and not installed:
            installed.append(True)
            candidate.mech.sample_runtime_noises_ = sample
        return gnabar

    mechanism.setreference("gnabar", getter)
    assert not population_core._runtime_execution_stable(candidate.mech)
    assert not reads
    with _count_handler_noise_calls() as handler_calls:
        actual = _execute(candidate, entrypoint)
    assert installed == [True]
    assert len(handler_calls) == 8
    assert len(sampled) == 7
    expected = _execute(reference, entrypoint, callbacks=[_NoopPreStep()])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _assert_no_payload(candidate)


@pytest.mark.parametrize(
    "phase, fused_steps",
    [("pre_loop", 0), ("pre_chunk", 3), ("post_chunk", 3), ("post_step", 1)],
)
@pytest.mark.parametrize("replacement", ["sampler", "pre_step"])
def test_longrun_hook_mutation_refreshes_dispatch(
    monkeypatch, phase, fused_steps, replacement
):
    candidate, _ = _case()
    reference, _ = _case()
    calls, _ = _counting_compiler(monkeypatch)

    class Replace(Callback):
        def __init__(self):
            super().__init__()
            self.changed = False
            self.chunk_count = 0
            self.observed = []

        def replace(self, model):
            if self.changed:
                return
            self.changed = True
            if replacement == "sampler":
                original = model.mech.sample_runtime_noises_

                def sample(*args, **kwargs):
                    assert calls[-1] == "make_intra"
                    self.observed.append(model.t.detach().clone())
                    return original(*args, **kwargs)

                model.mech.sample_runtime_noises_ = sample
            else:

                def before(callback, population):
                    assert calls[-1] == "make_intra"
                    callback.observed.append(population.t.detach().clone())

                self.pre_step_hook = MethodType(before, self)

        def pre_loop_hook(self, model):
            if phase == "pre_loop":
                self.replace(model)

        def pre_chunk_hook(self, model, timepoints=None):
            self.chunk_count += 1
            if phase == "pre_chunk" and self.chunk_count == 2:
                self.replace(model)

        def post_chunk_hook(self, model, timepoints=None):
            if phase == "post_chunk":
                self.replace(model)

        def post_step_hook(self, model):
            if phase == "post_step":
                self.replace(model)

    changes = Replace()
    actual = _execute(candidate, "longrun", callbacks=[changes])
    expected_calls = {"make_intra": 8 - fused_steps, "_step": 8 - fused_steps}
    if fused_steps:
        expected_calls["_step_with_intra"] = fused_steps
    assert Counter(calls) == expected_calls
    assert len(changes.observed) == 8 - fused_steps
    calls.clear()
    reference_changes = Replace()
    expected = _execute(
        reference, "longrun", callbacks=[reference_changes, _NoopPreStep()]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.stack(changes.observed),
        torch.stack(reference_changes.observed),
        rtol=0,
        atol=0,
    )
    _assert_no_payload(candidate)


def test_longrun_preserves_scatter_before_mutating_pre_step_hook(monkeypatch):
    candidate, _ = _case()
    reference, _ = _case()
    calls, _ = _counting_compiler(monkeypatch)

    class MutateIndices(Callback):
        def pre_step_hook(self, model):
            assert calls[-1] == "make_intra"
            # The scatter for this step already used the previous destination.
            model.intra.indices[0][-1].fill_(1)

    actual = _execute(candidate, "longrun", callbacks=[MutateIndices()])
    assert Counter(calls) == {"make_intra": 8, "_step": 8}
    expected = _run(reference, callbacks=[MutateIndices(), _NoopPreStep()])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("entrypoint", ["run", "longrun"])
def test_no_stimulus_runtime_noise_retains_seeded_samples(monkeypatch, entrypoint):
    candidate, _ = _without_intra(noise=True)
    reference, _ = _without_intra(noise=True)
    _counting_compiler(monkeypatch)
    # Runtime NOISE mutates its buffer in-place. Indexed recording creates
    # independent samples; unindexed Recorder would retain the same live alias.
    nodes = torch.arange(candidate.nc)
    actual_noise = dn.Recorder(["_RuntimeNoise.eta"], node_indices=nodes)
    expected_noise = dn.Recorder(["_RuntimeNoise.eta"], node_indices=nodes)
    with _count_handler_noise_calls() as sampled:
        actual = _execute(candidate, entrypoint, callbacks=[actual_noise])
    assert len(sampled) == 8
    expected = _execute(
        reference, entrypoint, callbacks=[expected_noise, _NoopPreStep()]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_samples = actual_noise.stack("_RuntimeNoise.eta")
    torch.testing.assert_close(
        actual_samples, expected_noise.stack("_RuntimeNoise.eta"), rtol=0, atol=0
    )
    assert not torch.equal(actual_samples[0], actual_samples[-1])


class _DiffusiveState(State):
    State.STATE("x")
    State.DERIVATIVE("x' = 0.0")
    State.DIFFUSION("x = 0.2")
    State.METHOD("euler_maruyama")


class _ReplacingSamplerState(State):
    State.STATE("x")
    State.DERIVATIVE("x' = 0.0 * x")

    def advance(self, v, dt, values):
        if not getattr(self, "replacement_installed", False):
            self.replacement_installed = True
            self.sample_runtime_noises_ = self._replacement_sampler
        return {"x": values["x"] + dt}

    def _replacement_sampler(self, *args, **kwargs):
        self.replacement_calls = getattr(self, "replacement_calls", 0) + 1
        return False


class _ReplacingSamplerMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_ReplacingSamplerState)


@pytest.mark.parametrize("entrypoint", ["run", "longrun"])
def test_custom_advance_cannot_make_empty_noise_proof_stale(monkeypatch, entrypoint):
    model, _ = _without_intra()
    with dn.ctx(JIT=0, REQUIRE_GRAD=0, DEVICE="cpu", DTYPE=torch.float64):
        model.insert(_ReplacingSamplerMechanism)
        model.initialize()
    _counting_compiler(monkeypatch)
    with _count_handler_noise_calls() as sampled:
        _execute(model, entrypoint)
    state = next(iter(model.mech._ReplacingSamplerMechanism.DE.values()))
    assert len(sampled) == 8
    assert state.replacement_calls == 7
    torch.testing.assert_close(
        model.mech._ReplacingSamplerMechanism.x,
        torch.full_like(model.v, 8 * DT),
        rtol=1e-15,
        atol=1e-15,
    )
    _assert_no_payload(model)


class _DiffusiveCurrent(Mechanism):
    Mechanism.STATE_BUNDLE(_DiffusiveState)
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return 1.0e-4 * self.x


@pytest.mark.parametrize("entrypoint", ["run", "longrun"])
def test_state_sde_rng_and_trace_survive_empty_runtime_noise_optimization(
    monkeypatch, entrypoint
):
    def case():
        model, _ = _without_intra()
        with dn.ctx(JIT=0, REQUIRE_GRAD=0, DEVICE="cpu", DTYPE=torch.float64):
            model.insert(_DiffusiveCurrent)
            model.initialize()
        state = next(iter(model.mech._DiffusiveCurrent.DE.values()))
        state.x_dW_rng.reseed(1415)
        return model, state

    candidate, candidate_state = case()
    reference, reference_state = case()
    _counting_compiler(monkeypatch)
    actual_states = dn.Recorder(["_DiffusiveCurrent.x"])
    expected_states = dn.Recorder(["_DiffusiveCurrent.x"])
    actual = _execute(candidate, entrypoint, callbacks=[actual_states])
    expected = _execute(
        reference, entrypoint, callbacks=[expected_states, _NoopPreStep()]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    state_trace = actual_states.stack("_DiffusiveCurrent.x")
    torch.testing.assert_close(
        state_trace, expected_states.stack("_DiffusiveCurrent.x"), rtol=0, atol=0
    )
    assert not torch.equal(state_trace[0], state_trace[-1])
    torch.testing.assert_close(
        candidate_state.x_dW_rng.rng_state()["cpu"],
        reference_state.x_dW_rng.rng_state()["cpu"],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("entrypoint", ["run", "longrun"])
def test_empty_noise_payload_does_not_leak_after_kernel_exception(
    monkeypatch, entrypoint
):
    model, _ = _without_intra()
    _counting_compiler(monkeypatch)
    counting_compile = torch.compile
    fail_once = [True]

    def compile_(function, **kwargs):
        compiled = counting_compile(function, **kwargs)

        @functools.wraps(compiled)
        def invoke(*args, **kw):
            if function.__name__ == "_step" and fail_once[0]:
                fail_once[0] = False
                raise RuntimeError("intentional kernel failure")
            return compiled(*args, **kw)

        return invoke

    monkeypatch.setattr(torch, "compile", compile_)
    with pytest.raises(RuntimeError, match="intentional kernel failure"):
        _execute(model, entrypoint, steps=1)
    _assert_no_payload(model)
    assert model.t.item() == 0.0
    # A later custom sampler must execute normally, not inherit a skip flag.
    sampled = []
    original = model.mech.sample_runtime_noises_

    def sample(*args, **kwargs):
        sampled.append(None)
        return original(*args, **kwargs)

    model.mech.sample_runtime_noises_ = sample
    _execute(model, entrypoint, steps=1)
    assert len(sampled) == 1
    _assert_no_payload(model)


@pytest.mark.parametrize("entrypoint", ["run", "longrun"])
def test_empty_noise_payload_is_local_to_nested_execution(monkeypatch, entrypoint):
    outer, _ = _without_intra()
    inner, _ = _without_intra()
    _counting_compiler(monkeypatch)
    counting_compile = torch.compile
    nested = []
    sampled = []
    original_sample = inner.mech.sample_runtime_noises_

    def sample(*args, **kwargs):
        sampled.append(None)
        return original_sample(*args, **kwargs)

    inner.mech.sample_runtime_noises_ = sample

    def compile_(function, **kwargs):
        compiled = counting_compile(function, **kwargs)

        @functools.wraps(compiled)
        def invoke(*args, **kw):
            if (
                function.__name__ == "_step"
                and function.__self__ is outer.integrator
                and not nested
            ):
                nested.append(True)
                _execute(inner, entrypoint, steps=1)
            return compiled(*args, **kw)

        return invoke

    monkeypatch.setattr(torch, "compile", compile_)
    with _count_handler_noise_calls() as handler_calls:
        _execute(outer, entrypoint, steps=2)
    assert nested == [True]
    assert sampled == [None]
    assert len(handler_calls) == 1  # Inner only; the outer handler remains empty.
    _assert_no_payload(outer)
    _assert_no_payload(inner)
