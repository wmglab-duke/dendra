"""Population-owned steady-state snapshots are explicit initialization inputs."""

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.random_parameters import make_random_parameter_spec
from tests.func.test_membrane_current import DT, _model
from tests.func.test_network import _clone, _close
from tests.func.test_population_initialization_transforms import _Add
from tests.func.test_stimulation_sources import _SourceWaveform

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _cached(kind="path"):
    source = _model(kind, batch=None)
    source.t.fill_(1.25)
    source._duration_remainder.fill_(0.5 * DT)
    source.cache("_steady_state")
    source.initialize()
    source.integrator._initialize(source, source.v.new_tensor(DT), force=True)
    return source


def _step(source):
    source.integrator._initialize(source, source.v.new_tensor(DT), force=True)
    source._step(source.integrator, source, source.v.new_tensor(DT), None, None)
    source.t = source.t + DT


@pytest.mark.parametrize(
    "kind",
    (
        "point",
        "path",
        "cable",
        "tree",
        "block_path",
        "block_tree",
        "multi_point",
        "multi_scalar",
        "multi_block",
    ),
)
def test_owned_cache_initialize_and_step_match_native_across_topologies(kind):
    source, native = _cached(kind), _cached(kind)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    assert tensors.initialization.cache is not None
    assert not tensors.constants["cache.active"]
    initialized = functional.initialize(
        tensors.parameters, tensors.constants, tensors.initialization
    )
    native.initialize()
    _close(initialized.state, functional.extract(native).state)
    assert initialized.constants["cache.active"]
    assert initialized.state["clock"]["t"] == 0
    assert initialized.state["control"]["duration_remainder"] == 0
    actual, _ = functional.prepare_and_step(
        initialized.parameters, initialized.constants, initialized.state
    )
    _step(native)
    _close(actual, functional.extract(native).state)


def test_owned_cache_defaults_are_independent_and_explicit_overrides_restore_effective_values():
    source, native = _cached(), _cached()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    original = tensors.initialization.cache["v"].clone()
    source._caches["_steady_state"]["v"].add_(99)
    refreshed = functional.extract()
    torch.testing.assert_close(refreshed.initialization.cache["v"], original)
    cache = _clone(tensors.initialization.cache, detached=True)
    cache["cm_param"].fill_(3)
    cache["cm"].fill_(2)
    native._caches["_steady_state"]["cm_param"].fill_(3)
    native._caches["_steady_state"]["cm"].fill_(2)
    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization._replace(cache=cache),
    )
    native.initialize()
    torch.testing.assert_close(initialized.parameters["cm_param"], native.cm_param)
    torch.testing.assert_close(initialized.constants["cache.values.cm"], native.cm)
    actual, _ = functional.prepare_and_step(
        initialized.parameters, initialized.constants, initialized.state
    )
    _step(native)
    _close(actual, functional.extract(native).state)
    gradient = torch.autograd.grad(
        actual["integrator"]["v"].sum(),
        tensors.parameters["cm_param"],
        allow_unused=True,
    )[0]
    assert gradient is None or torch.count_nonzero(gradient) == 0


@pytest.mark.parametrize("kind", ("multi_point", "multi_scalar"))
def test_owned_cache_partial_child_voltage_edits_preserve_native_write_order(kind):
    source, native = _cached(kind), _cached(kind)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    cache = _clone(tensors.initialization.cache, detached=True)
    cache["populations.point.v"].add_(4)
    native._caches["_steady_state"]["populations.point.v"].add_(4)
    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization._replace(cache=cache),
    )
    native.initialize()
    _close(initialized.state, functional.extract(native).state)


def test_owned_cache_initialize_compiles_and_supports_jvp_and_vmap():
    functional, tensors = dn.func.make_functional(_cached(), dt=DT)

    def initialize(raw):
        packet = functional.initialize(
            {**tensors.parameters, "cm_param": raw},
            tensors.constants,
            tensors.initialization,
        )
        state, _ = functional.prepare_and_step(
            packet.parameters, packet.constants, packet.state
        )
        return packet.parameters["cm_param"], state["integrator"]["v"]

    raw = tensors.parameters["cm_param"]
    expected = initialize(raw)
    _close(
        torch.compile(initialize, backend="aot_eager", fullgraph=True)(raw), expected
    )
    primal, tangent = torch.func.jvp(initialize, (raw,), (torch.ones_like(raw),))
    _close(primal, expected)
    torch.testing.assert_close(tangent[0], torch.ones_like(raw))
    torch.testing.assert_close(tangent[1], torch.zeros_like(tangent[1]))
    for lanes in (0, 2):
        actual = torch.vmap(initialize)(raw.expand(lanes, *raw.shape))
        _close(actual, tuple(value.expand(lanes, *value.shape) for value in expected))


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_compiled_owned_cache_restart_and_explicit_constants_continuation(execution):
    source, native = _cached(), _cached()
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        initialization="fresh",
        execution=execution,
        inference=execution == "while",
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        backend="aot_eager",
    )
    first, auxiliary = runner()
    repeated, _ = runner()
    _close(first, repeated)
    with pytest.raises(ValueError, match="parameters="):
        runner(first)
    with pytest.raises(ValueError, match="constants="):
        runner(first, parameters=auxiliary["parameters"])
    final, _ = runner(
        first, parameters=auxiliary["parameters"], constants=auxiliary["constants"]
    )
    native.initialize()
    with torch.no_grad():
        for _ in range(8):
            _step(native)
    _close(final, runner.functional.extract(native).state)


def test_owned_cache_presence_and_tensor_schema_are_frozen():
    source = _cached()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    with pytest.raises(dn.func.FunctionalizationError, match="schema"):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization._replace(cache=None),
        )
    source.clear_steady_state()
    with pytest.raises(dn.func.FunctionalizationError, match="presence"):
        functional.initialize(
            tensors.parameters, tensors.constants, tensors.initialization
        )
    functional.prepare_and_step(tensors.parameters, tensors.constants, tensors.state)


def test_owned_cache_skips_pre_hooks_and_structured_post_transforms():
    source = _cached()
    source.register_pre_initialize_hook(
        lambda _population: (_ for _ in ()).throw(AssertionError("pre hook ran"))
    )
    source.register_post_initialize_transform(
        "shift",
        _Add(),
        reads=("state.integrator.v",),
        writes=("state.integrator.v",),
        inputs={"increment": source.v.new_tensor(10)},
    )
    source.cache("_steady_state")
    source.initialize()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    initialized = functional.initialize(
        tensors.parameters, tensors.constants, tensors.initialization
    )
    torch.testing.assert_close(initialized.state["integrator"]["v"], source.v)


def test_owned_cache_legacy_post_hook_rejects_fresh_but_preserves_snapshot():
    source = _cached()
    source.register_post_initialize_hook(lambda _population: None)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    with pytest.raises(dn.func.FunctionalizationError, match="legacy post"):
        functional.initialize(
            tensors.parameters, tensors.constants, tensors.initialization
        )
    functional.prepare_and_step(tensors.parameters, tensors.constants, tensors.state)


def test_owned_cache_rejects_a_source_policy_that_invalidates_it_before_rng_work():
    source = _cached()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    # Random declarations are mutable source configuration. Native initialize
    # discards a steady-state cache when any declaration requests a redraw.
    source.random_parameters["drive"] = make_random_parameter_spec(
        "drive",
        scope="global",
        distribution="normal",
        seed=19,
        resample_on_initialize=True,
    )
    assert source._has_random_parameters_resampled_on_initialize()
    rng = torch.get_rng_state().clone()
    with pytest.raises(
        dn.func.FunctionalizationError, match="random parameter resampling"
    ):
        functional.initialize(
            tensors.parameters, tensors.constants, tensors.initialization
        )
    torch.testing.assert_close(torch.get_rng_state(), rng)
    assert "_steady_state" in source._caches


@pytest.mark.parametrize("parent_cached", (False, True))
def test_packed_owner_initialization_ignores_component_owned_caches(parent_cached):
    source, native = (
        (_cached("multi_point"), _cached("multi_point"))
        if parent_cached
        else (_model("multi_point", batch=None), _model("multi_point", batch=None))
    )
    for owner in (source, native):
        for component in owner.populations.values():
            component.cache("_steady_state")
            component._caches["_steady_state"]["v"].add_(99)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    assert functional._owned_cache_bound == parent_cached
    initialized = functional.initialize(
        tensors.parameters, tensors.constants, tensors.initialization
    )
    native.initialize()
    _close(initialized.state, functional.extract(native).state)


def test_owned_cache_replacing_duration_keeps_outside_borrowed_field_unchanged():
    source = _cached()
    borrowed = source._duration_remainder
    functional, tensors = dn.func.make_functional(
        source, dt=DT, extra=(borrowed.expand_as(source.v), _SourceWaveform())
    )
    initialized = functional.initialize(
        tensors.parameters, tensors.constants, tensors.initialization
    )
    source.initialize()
    torch.testing.assert_close(borrowed, torch.zeros_like(borrowed))
    torch.testing.assert_close(
        initialized.parameters["stimulation.extra.contacts.0.field"],
        tensors.parameters["stimulation.extra.contacts.0.field"],
    )
