"""Compiled runners expose fresh initialization without beginning execution."""

from __future__ import annotations

import os
from contextlib import nullcontext

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_continuation import (
    _assign,
    _callbacks,
    _close,
    _plain,
    _snapshot,
)
from tests.func.test_continuation_initialization import _Scale
from tests.func.test_explicit_initialization import _frames
from tests.func.test_lowering_api import DT, G, _network, _population, _target, _voltage
from tests.func.test_membrane_current import DT as CACHE_DT
from tests.func.test_network import DT as NETWORK_DT
from tests.func.test_network_cache_initialization import _cached_network
from tests.func.test_network_initialization_bindings import _cross_population_binding
from tests.func.test_network_mapping_boundaries import _OnItems
from tests.func.test_population_owned_cache import _cached as _cached_population
from tests.func.test_stimulation_rng_sharing import DT as SHARED_DT
from tests.func.test_stimulation_rng_sharing import _network as _shared_network
from tests.func.test_stimulation_rng_sharing import _population as _shared_population

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


def _program(network, *, scaled=False, **options):
    model = _network() if network else _population()
    if scaled:
        owner = model.left if network else model
        owner.register_pre_initialize_transform(
            "scale",
            _Scale(),
            reads=(f"parameters.{G}",),
            writes=(f"parameters.{G}",),
            inputs={"factor": owner.v.new_tensor(1.5)},
        )
    return dn.compile(
        model,
        dt=DT,
        steps=2,
        trainable="*g_param",
        backend=options.pop("backend", "eager"),
        disable=options.pop("disable", True),
        **options,
    )


def _reference(program, *, parameters=None, constants=None, initialization=None):
    tensors = program.functional.extract()._replace(
        parameters=program.parameters if parameters is None else parameters,
        constants=program.constants if constants is None else constants,
        state=program.initial_state,
    )
    return dn.func.initialize(
        program.functional,
        tensors,
        initialization=(
            program.initialization_inputs if initialization is None else initialization
        ),
    )


def _inputs(program, network, *, voltage=None, factor=None):
    defaults = program.initialization_inputs
    selected = defaults.populations["left"] if network else defaults
    updates = {}
    if voltage is not None:
        updates["v_init"] = voltage
    if factor is not None:
        updates["transforms"] = {
            **selected.transforms,
            "pre.scale.factor": factor,
        }
    selected = selected._replace(**updates)
    return (
        defaults._replace(populations={**defaults.populations, "left": selected})
        if network
        else selected
    )


class _OnTraversal(_OnItems):
    def __iter__(self):
        if not self.accessed:
            self.accessed = True
            self.action()
        return super().__iter__()


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("policy", ("snapshot", "fresh"))
def test_runner_initializer_is_fresh_without_starting_callbacks_or_preparing_a_run(
    monkeypatch, network, policy
):
    captures = []

    def backend(graph, _examples):
        captures.append(graph)
        return graph.forward

    program = _program(
        network,
        initialization=policy,
        callbacks=_callbacks(network),
        backend=backend,
        disable=False,
    )
    original, _ = program()
    if network:
        program.initialization_inputs.populations["left"].v_init.add_(3.0)
        program.initialization_inputs.t.fill_(0.75)
    else:
        program.initialization_inputs.v_init.add_(3.0)
    expected = _reference(program)
    defaults = _snapshot(
        (
            program.parameters,
            program.constants,
            program.initial_state,
            program.initialization_inputs,
        )
    )
    source = _snapshot(program.functional.extract().state)
    report, count = program.execution_report(), len(captures)
    prepared = None if network else program._prepared

    def reject(*args, **kwargs):
        raise AssertionError("initialization must not enter rollout or callback setup")

    with monkeypatch.context() as guarded:
        guarded.setattr(program.callbacks, "_initialize", reject)
        guarded.setattr(program.kernel, "bind", reject)
        guarded.setattr(program.functional, "prepare", reject)
        guarded.setattr(program.functional, "sample_runtime_draws", reject)
        guarded.setattr(program.functional, "extract", reject)
        if not network:
            guarded.setattr(program, "_prepare", reject)
        actual = dn.func.initialize(program)
    assert type(actual) is dn.func.Continuation and actual.callback_state is None
    _close(_frames(actual), _frames(expected))
    assert program.execution_report() is report and len(captures) == count
    if not network:
        assert program._prepared is prepared
    _close(
        (
            program.parameters,
            program.constants,
            program.initial_state,
            program.initialization_inputs,
        ),
        defaults,
    )
    _close(program.functional.extract().state, source)
    repeated, _ = program()
    if policy == "snapshot":
        _close(repeated, original)
    else:
        continued, _ = program(actual)
        _close(repeated, continued)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_runner_initialization_preserves_effective_aliases_overrides_and_checkpoint_vjps(
    network,
):
    program = _program(
        network,
        scaled=True,
        inference=False,
        callbacks=_callbacks(network),
        checkpoint_every=2,
        backend="aot_eager",
        disable=False,
    )
    parameters, constants = _plain(program.parameters), _plain(program.constants)
    upstream = _target(parameters, network).detach().clone().requires_grad_()
    replacement = upstream * 1.2
    _assign(parameters, network, replacement)
    selected_constants = constants["populations"]["left"] if network else constants
    timestep = selected_constants["dt"].clone()
    selected_constants["dt"] = timestep
    factor = upstream.new_tensor(1.7, requires_grad=True)
    inputs = _inputs(program, network, factor=factor)
    before = _snapshot((parameters, constants, inputs, program.initial_state))
    expected = _reference(
        program, parameters=parameters, constants=constants, initialization=inputs
    )
    actual = dn.func.initialize(
        program, parameters=parameters, constants=constants, initialization=inputs
    )
    _close(_frames(actual), _frames(expected))
    effective = _target(actual.parameters, network)
    assert effective is not replacement and effective.grad_fn is not None
    torch.testing.assert_close(effective, replacement * factor)
    if network:
        assert effective is actual.parameters["populations"]["right"][G]
    selected = actual.constants["populations"]["left"] if network else actual.constants
    assert selected["dt"] is timestep
    _close((parameters, constants, inputs, program.initial_state), before)

    reference, whole = program.functional.prepare_and_rollout(
        expected.parameters,
        expected.constants,
        expected.state,
        steps=5,
        callbacks=program.callbacks,
    )
    _, first = program(actual, steps=2)
    final, last = program(first["continuation"], steps=3)
    _close(final, reference)
    key = "left.v" if network else "v"
    joined = torch.cat(
        (first["callbacks"]["trace"][key], last["callbacks"]["trace"][key])
    )
    _close(joined, whole["callbacks"]["trace"][key])
    wanted = torch.autograd.grad(
        _voltage(reference, network).square().sum(),
        (upstream, factor),
        retain_graph=True,
    )
    observed = torch.autograd.grad(
        _voltage(final, network).square().sum(), (upstream, factor)
    )
    _close(observed, wanted)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed
    )
    assert _target(last["continuation"].parameters, network) is effective


@pytest.mark.parametrize("reader_first", (False, True))
def test_runner_initializer_preserves_parameter_owned_constant_readers(reader_first):
    model, key = _cross_population_binding(reader_first=reader_first)
    program = dn.compile(
        model,
        dt=NETWORK_DT,
        steps=2,
        inference=False,
        trainable=f"populations.left.{key}",
        backend="eager",
        disable=True,
    )
    result = dn.func.initialize(program)
    selected = result.parameters["populations"]["left"][key]
    assert (
        selected is result.constants["populations"]["right"]["parametrizations.factor"]
    )
    assert selected.requires_grad
    state, _ = program(result)
    gradient = torch.autograd.grad(
        _voltage(state, True).square().sum(),
        program.get_parameter(f"populations.left.{key}"),
    )[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("mode", ("grad", "no_grad", "inference"))
def test_explicit_runner_initialization_obeys_caller_grad_mode_and_owns_versioned_tensors(
    network, mode
):
    program = _program(network, scaled=True, initialization="fresh", inference=True)
    parameters = _plain(program.parameters)
    upstream = _target(parameters, network).detach().clone().requires_grad_()
    _assign(parameters, network, upstream * 1.2)
    context = (
        torch.no_grad()
        if mode == "no_grad"
        else torch.inference_mode()
        if mode == "inference"
        else nullcontext()
    )
    with context:
        result = dn.func.initialize(program, parameters=parameters)
        assert torch.is_grad_enabled() == (mode == "grad")
        assert torch.is_inference_mode_enabled() == (mode == "inference")
    effective = _target(result.parameters, network)
    assert effective.requires_grad == (mode == "grad")
    assert (effective.grad_fn is not None) == (mode == "grad")
    assert all(
        not value.is_inference()
        for value in pytree.tree_leaves(_frames(result))
        if isinstance(value, torch.Tensor)
    )
    if mode == "grad":
        derivative = torch.autograd.grad(effective.sum(), upstream)[0]
        torch.testing.assert_close(derivative, torch.full_like(upstream, 1.8))
        _, automatic = program(parameters=parameters, steps=0)
        assert not _target(automatic["continuation"].parameters, network).requires_grad


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_runner_initialization_replays_shared_rng_without_consuming_native_streams(
    monkeypatch, network
):
    if network:
        model, pulses = _shared_network()
    else:
        model, _, pulse = _shared_population()
        pulses = [pulse]
    program = dn.compile(model, dt=SHARED_DT, steps=2, backend="eager", disable=True)
    functional = program.functional
    source, global_rng = _snapshot(functional.extract()), torch.get_rng_state().clone()
    native_rng = [pulse.generator.get_state().clone() for pulse in pulses]
    expected = _reference(program)
    actual = dn.func.initialize(program)
    repeated = dn.func.initialize(program)
    _close(_frames(actual), _frames(expected))
    _close(_frames(repeated), _frames(actual))
    defaults = program.initialization_inputs
    if network:
        random = functional.sample_population_initialization_draws(
            program.parameters, program.initial_state
        )
        stimulation = functional.sample_population_stimulation_initialization_draws(
            program.parameters, program.initial_state
        )
        inputs = defaults._replace(
            populations={
                name: value._replace(
                    random_draw=random[name], stimulation=stimulation[name]
                )
                for name, value in defaults.populations.items()
            }
        )
    else:
        inputs = defaults._replace(
            random_draw=functional.sample_initialization_draw(
                program.parameters, program.initial_state
            ),
            stimulation=functional.sample_stimulation_initialization_draw(
                program.initial_state
            ),
        )

    def reject(*args, **kwargs):
        raise AssertionError("explicit initialization draws must be replayed")

    with monkeypatch.context() as guarded:
        for name in (
            "sample_initialization_draw",
            "sample_stimulation_initialization_draw",
            "sample_population_initialization_draws",
            "sample_population_stimulation_initialization_draws",
        ):
            if hasattr(functional, name):
                guarded.setattr(functional, name, reject)
        replayed = dn.func.initialize(program, initialization=inputs)
    _close(_frames(replayed), _frames(actual))
    _close(functional.extract(), source)
    torch.testing.assert_close(torch.get_rng_state(), global_rng)
    for pulse, state in zip(pulses, native_rng, strict=True):
        torch.testing.assert_close(pulse.generator.get_state(), state)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_runner_initialization_restores_owned_and_network_caches_without_editing_defaults(
    network,
):
    model = (
        _cached_network(training=False, stimulus=True)
        if network
        else _cached_population("point")
    )
    program = dn.compile(
        model,
        dt=NETWORK_DT if network else CACHE_DT,
        steps=2,
        backend="eager",
        disable=True,
    )
    before = _snapshot(
        (
            program.parameters,
            program.constants,
            program.initial_state,
            program.initialization_inputs,
        )
    )
    expected = _reference(program)
    result = dn.func.initialize(program)
    _close(_frames(result), _frames(expected))
    constants = result.constants["populations"]["cell"] if network else result.constants
    assert bool(constants["cache.active"])
    _close(
        (
            program.parameters,
            program.constants,
            program.initial_state,
            program.initialization_inputs,
        ),
        before,
    )


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_initialized_runner_checkpoint_starts_callbacks_once_and_resumes_recorded_carry(
    network, monkeypatch, tmp_path
):
    program = _program(
        network, inference=False, callbacks=_callbacks(network), checkpoint_every=1
    )
    calls = []
    initialize = program.callbacks._initialize

    def counted(*args, **kwargs):
        calls.append(None)
        return initialize(*args, **kwargs)

    monkeypatch.setattr(program.callbacks, "_initialize", counted)
    initial = dn.func.initialize(program)
    assert calls == [] and initial.callback_state is None
    saved = dn.func.export_continuation(program.functional, initial)
    torch.save(saved, tmp_path / "initial.pt")
    restored = dn.func.restore_continuation(
        program.functional, torch.load(tmp_path / "initial.pt", weights_only=True)
    )
    assert restored.callback_state is None and calls == []
    if network:
        assert (
            _target(restored.parameters, True)
            is restored.parameters["populations"]["right"][G]
        )
    _, first = program(restored, steps=2)
    assert calls == [None]
    saved = program.export_continuation(first["continuation"])
    torch.save(saved, tmp_path / "recorded.pt")
    resumed = program.restore_continuation(
        torch.load(tmp_path / "recorded.pt", weights_only=True)
    )
    final, last = program(resumed, steps=1)
    assert calls == [None]
    reference, outputs = program.functional.prepare_and_rollout(
        resumed.parameters,
        resumed.constants,
        resumed.state,
        steps=1,
        callbacks=program.callbacks,
        callback_state=resumed.callback_state,
    )
    _close(final, reference)
    _close(last["callbacks"], outputs["callbacks"])
    key = "left.v" if network else "v"
    assert first["callbacks"]["trace"][key].shape[0] == 3
    assert last["callbacks"]["trace"][key].shape[0] == 1
    targets = dn.func.inspect_sources(resumed).trainable_leaves
    assert targets
    wanted = torch.autograd.grad(
        _voltage(reference, network).square().sum(), targets, allow_unused=True
    )
    observed = torch.autograd.grad(
        _voltage(final, network).square().sum(), targets, allow_unused=True
    )
    assert tuple(value is None for value in observed) == tuple(
        value is None for value in wanted
    )
    for actual_gradient, reference_gradient in zip(observed, wanted, strict=True):
        if actual_gradient is not None:
            torch.testing.assert_close(actual_gradient, reference_gradient)
            assert torch.isfinite(actual_gradient).all()
    assert any(value is not None and value.abs().sum() > 0 for value in observed)
    if network:
        # Portable restoration makes active retained values independent leaves;
        # raw sources can consequently have zero or unused suffix gradients.
        raw = _target(resumed.parameters, True)
        raw_gradient = observed[
            next(i for i, value in enumerate(targets) if value is raw)
        ]
        assert raw_gradient is None or not torch.count_nonzero(raw_gradient)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_runner_initializer_rejects_conflicting_forms_and_preserves_defaults(network):
    program = _program(network)
    before = _snapshot((program.parameters, program.constants, program.initial_state))
    bundle = program.functional.extract()
    for positional in (None, bundle, object()):
        with pytest.raises(TypeError):
            dn.func.initialize(program, positional)
    for keyword in ("parameters", "constants"):
        for value in (None, object()):
            with pytest.raises(
                TypeError, match=f"unexpected keyword argument '{keyword}'"
            ):
                dn.func.initialize(program.functional, bundle, **{keyword: value})
    for keywords in (
        {"parameters": object()},
        {"constants": object()},
        {"initialization": object()},
    ):
        with pytest.raises((TypeError, ValueError)):
            dn.func.initialize(program, **keywords)

    class Derived(type(program)):
        pass

    with pytest.raises(TypeError):
        dn.func.initialize(object.__new__(Derived))
    _close((program.parameters, program.constants, program.initial_state), before)
    assert program.execution_report() is None


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_runner_initializer_traverses_selected_mappings_before_source_validation(
    network,
):
    for domain in ("parameters", "constants", "initialization"):
        program = _program(network)
        source = (
            program.functional._source if network else program.functional._population
        )

        def invalidate():
            if network:
                source._step = object()
            else:
                source._flag_rebuild = True

        if domain == "initialization":
            defaults = program.initialization_inputs
            wrapped = _OnTraversal(
                defaults.populations if network else defaults.transforms, invalidate
            )
            value = defaults._replace(
                **{"populations" if network else "transforms": wrapped}
            )
        else:
            wrapped = value = _OnTraversal(_plain(getattr(program, domain)), invalidate)
        with pytest.raises(dn.func.FunctionalizationError):
            dn.func.initialize(program, **{domain: value})
        assert wrapped.accessed


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_runner_initializer_does_not_traverse_unused_default_sources(network):
    program = _program(network)
    parameters, constants = _plain(program.parameters), _plain(program.constants)
    defaults = program.initialization_inputs
    expected = _reference(
        program, parameters=parameters, constants=constants, initialization=defaults
    )

    def reject():
        raise AssertionError("overrides must not traverse unused runner defaults")

    program._parameters = _OnTraversal(parameters, reject)
    program._constants = _OnTraversal(constants, reject)
    wrapped = _OnTraversal(
        defaults.populations if network else defaults.transforms, reject
    )
    program._initialization_inputs = defaults._replace(
        **{"populations" if network else "transforms": wrapped}
    )
    actual = dn.func.initialize(
        program, parameters=parameters, constants=constants, initialization=defaults
    )
    _close(_frames(actual), _frames(expected))
    assert not any(
        value.accessed for value in (program._parameters, program._constants, wrapped)
    )


@pytest.mark.parametrize("context", ("compile", "grad", "forward_ad"))
def test_runner_initializer_rejects_function_transforms_before_host_initialization(
    monkeypatch, context
):
    program = _program(False)

    def reject(*args, **kwargs):
        raise AssertionError("transform rejection must precede host initialization")

    monkeypatch.setattr(program.functional, "initialize_host", reject)

    def work(value):
        dn.func.initialize(program)
        return value.square()

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="initialize must run outside torch.compile",
    ):
        if context == "compile":
            monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
            work(torch.tensor(0.5))
        elif context == "grad":
            torch.func.grad(work)(torch.tensor(0.5))
        else:
            with torch.autograd.forward_ad.dual_level():
                work(torch.tensor(0.5))


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_runner_initializer_does_not_require_runtime_extra_bindings(network):
    program = _program(network, extra=dn.VarInput())
    result = dn.func.initialize(program)
    _close(_frames(result), _frames(_reference(program)))
    assert result.callback_state is None and program.execution_report() is None
    with pytest.raises(TypeError, match="extra"):
        program(result)


@pytest.mark.mps
def test_mps_runner_initialization_preserves_caller_gradients_and_while_continuation():
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DTYPE=torch.float32, DEVICE="mps"):
        program = dn.compile(
            _population(),
            dt=DT,
            steps=4,
            chunk_steps=4,
            execution="while",
            callbacks={"trace": dn.func.Recorder(("v", "t"))},
            backend="aot_eager",
        )
        voltage = program.initialization_inputs.v_init.detach().clone().requires_grad_()
        inputs = _inputs(program, False, voltage=voltage)
        expected = _reference(program, initialization=inputs)
        actual = dn.func.initialize(program, initialization=inputs)
        _close(_frames(actual), _frames(expected))
        assert actual.state["integrator"]["v"].device.type == "mps"
        assert actual.state["integrator"]["v"].dtype == torch.float32
        wanted = torch.autograd.grad(
            expected.state["integrator"]["v"].sum(), voltage, retain_graph=True
        )[0]
        observed = torch.autograd.grad(actual.state["integrator"]["v"].sum(), voltage)[
            0
        ]
        torch.testing.assert_close(observed, wanted)
        with torch.no_grad():
            reference, outputs = program.functional.prepare_and_rollout(
                expected.parameters,
                expected.constants,
                expected.state,
                steps=4,
                callbacks=program.callbacks,
            )
        final, auxiliary = program(actual)
        _close(final, reference)
        _close(auxiliary["callbacks"], outputs["callbacks"])
