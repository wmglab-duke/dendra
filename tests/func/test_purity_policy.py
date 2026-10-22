"""Purity diagnostics are opt-in and independent of execution schemas."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func import _population as population_api
from tests.func.test_custom_material_processes import DT, DTYPE, _network, _population
from tests.func.test_functional_initialization import _assert_tree_close

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE, DEVICE="cpu", CHECK_PURITY=False):
        yield


def test_context_defaults_normalization_nested_restoration_and_atomic_failure():
    assert dn.current_check_purity() is False
    with dn.ctx(CHECK_PURITY=True):
        assert dn.current_check_purity() is True
        with dn.ctx(CHECK_PURITY="0"):
            assert dn.current_check_purity() is False
        assert dn.current_check_purity() is True
        with pytest.raises(ValueError, match="CHECK_PURITY"):
            with dn.ctx(CHECK_PURITY="maybe"):
                pass
        assert dn.current_check_purity() is True
    assert dn.current_check_purity() is False
    assert dn.normalize_check_purity(" true ") is True
    assert dn.normalize_check_purity(0) is False


def test_default_population_lowering_skips_purity_probes_and_global_snapshots(
    monkeypatch,
):
    model = _population(physiology=False)

    def unexpected(*args, **kwargs):
        pytest.fail("Optional purity diagnostics ran in default trust mode")

    for name in (
        "_audit_read_only_workspaces",
        "_audit_event_receivers",
        "_audit_initialization",
    ):
        monkeypatch.setattr(population_api.FunctionalPopulation, name, unexpected)
    monkeypatch.setattr(population_api, "_global_rng_snapshot", unexpected)
    monkeypatch.setattr(population_api, "_class_state_snapshot", unexpected)
    monkeypatch.setattr(
        population_api, "_initialization_callable_dependency_signature", unexpected
    )
    functional, tensors = dn.func.make_functional(model, dt=DT)
    assert functional._check_purity is False
    final, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state
    )
    assert torch.isfinite(final["integrator"]["v"]).all()


def test_default_in_graph_transforms_skip_deep_dependency_inventory(monkeypatch):
    import copy

    from tests.func.test_parameter_materialization import _scaled_unmyelinated

    model, transform = _scaled_unmyelinated(1.25)

    def unexpected(*args, **kwargs):
        pytest.fail("Trusted transforms triggered a deep authored dependency inventory")

    for name in (
        "_initialization_callable_dependency_signature",
        "_transform_structure_signature",
        "_transform_callable_structure_signature",
    ):
        monkeypatch.setattr(population_api, name, unexpected)
    functional, inputs = dn.func.make_functional(model, dt=DT)
    with dn.ctx(CHECK_PURITY=True):
        extracted = functional.extract(copy.deepcopy(model))
        snapshot, _ = functional.prepare_and_step(
            inputs.parameters, inputs.constants, inputs.state
        )
        initialized = functional.initialize(
            extracted.parameters, extracted.constants, extracted.initialization
        )
        fresh, _ = functional.prepare_and_step(
            initialized.parameters, initialized.constants, initialized.state
        )
    assert torch.isfinite(snapshot["integrator"]["v"]).all()
    assert torch.isfinite(fresh["integrator"]["v"]).all()
    transform.factor = 2.0
    with pytest.raises(dn.func.FunctionalizationError, match="structure changed"):
        functional.prepare(inputs.parameters, inputs.constants)


@pytest.mark.parametrize(
    "kind", ("table", "injection", "voltage", "event", "continuous", "transform")
)
def test_default_authored_hook_contracts_skip_recursive_dependency_inspection(
    kind, monkeypatch
):
    if kind == "table":
        from tests.func.test_table_plan import _population as make_model

        model = make_model()
        model.mech.mechanisms["table"].usetables(False)
    elif kind == "injection":
        from tests.func.test_mechanism_injection_plan import _population as make_model

        model = make_model()
    elif kind == "voltage":
        from tests.func.test_voltage_process_guards import _model as make_model

        model = make_model()
    elif kind == "event":
        from tests.func.test_event_receive import _model as make_model

        model = make_model(batch=None)
    elif kind == "continuous":
        from tests.func.test_continuous_receive import _model as make_model

        model = make_model(batch=None)
    else:
        from tests.func.test_population_initialization_transforms import (
            _Add,
        )
        from tests.func.test_population_initialization_transforms import (
            _model as make_model,
        )

        model = make_model(
            pre=(
                (
                    ("shift", _Add()),
                    {
                        "reads": ("state.integrator.v",),
                        "writes": ("state.integrator.v",),
                        "inputs": {"increment": torch.tensor(0.5, dtype=DTYPE)},
                    },
                ),
            )
        )

    def unexpected(*args, **kwargs):
        pytest.fail("Trusted hooks triggered recursive purity dependency inspection")

    monkeypatch.setattr(
        population_api, "_initialization_callable_dependency_signature", unexpected
    )
    functional, inputs = dn.func.make_functional(model, dt=DT)
    with dn.ctx(CHECK_PURITY=True):
        # Retained trust applies to extraction, fresh setup, and transitions.
        extracted = functional.extract(model)
        initialized = functional.initialize(
            extracted.parameters, extracted.constants, extracted.initialization
        )
        final, _ = functional.prepare_and_step(
            initialized.parameters, initialized.constants, initialized.state
        )
    assert torch.isfinite(final["integrator"]["v"]).all()


def test_enabled_population_diagnostics_are_captured_and_preserve_results():
    model = _population(physiology=False)
    trusted, inputs = dn.func.make_functional(model, dt=DT)
    with dn.ctx(CHECK_PURITY=True):
        checked, checked_inputs = dn.func.make_functional(model, dt=DT)
    assert checked._check_purity is True
    assert trusted._check_purity is False
    actual, _ = checked.prepare_and_step(
        checked_inputs.parameters, checked_inputs.constants, checked_inputs.state
    )
    expected, _ = trusted.prepare_and_step(
        inputs.parameters, inputs.constants, inputs.state
    )
    _assert_tree_close(actual, expected)
    with dn.ctx(CHECK_PURITY=True):
        # A pre-existing trusted plan retains its policy as well.
        repeated, _ = trusted.prepare_and_step(
            inputs.parameters, inputs.constants, inputs.state
        )
    _assert_tree_close(repeated, expected)


def test_default_network_compilation_skips_population_diagnostic_probes(monkeypatch):
    network = _network()

    def unexpected(*args, **kwargs):
        pytest.fail("Network compilation ran optional purity probes")

    for name in (
        "_audit_read_only_workspaces",
        "_audit_event_receivers",
        "_audit_initialization",
    ):
        monkeypatch.setattr(population_api.FunctionalPopulation, name, unexpected)
    monkeypatch.setattr(
        population_api, "_initialization_callable_dependency_signature", unexpected
    )
    runner = dn.compile(
        network, dt=DT, steps=1, initialization="fresh", backend="aot_eager"
    )
    assert all(
        not plan._check_purity for plan in runner.functional.populations.values()
    )
    final, _ = runner()
    assert torch.isfinite(final["populations"]["left"]["integrator"]["v"]).all()


def test_required_parameter_schema_is_checked_in_trusted_mode():
    functional, inputs = dn.func.make_functional(_population(physiology=False), dt=DT)
    parameters = dict(inputs.parameters)
    parameters.pop(next(iter(parameters)))
    with pytest.raises(KeyError, match="parameter tree mismatch"):
        functional.prepare(parameters, inputs.constants)


@pytest.mark.parametrize("checked", (False, True))
def test_fresh_initialization_retains_diagnostics_outside_creation_context(
    checked, monkeypatch
):
    model = _population(physiology=False)
    with dn.ctx(CHECK_PURITY=checked):
        functional, inputs = dn.func.make_functional(model, dt=DT)
    calls = []
    original = population_api._initialization_callable_dependency_signature

    def observe(*args, **kwargs):
        calls.append(dn.current_check_purity())
        return original(*args, **kwargs)

    monkeypatch.setattr(
        population_api, "_initialization_callable_dependency_signature", observe
    )
    with dn.ctx(CHECK_PURITY=not checked):
        state = functional.initialize(
            inputs.parameters, inputs.constants, inputs.initialization
        )
        final, _ = functional.prepare_and_step(
            state.parameters, state.constants, state.state
        )
        functional.extract(model)
        assert dn.current_check_purity() is (not checked)
    assert bool(calls) is checked
    assert all(calls)
    assert torch.isfinite(final["integrator"]["v"]).all()


@pytest.mark.parametrize("checked", (False, True))
def test_compiled_fresh_initialization_retains_creation_policy(checked):
    model = _population(physiology=False)
    with dn.ctx(CHECK_PURITY=checked):
        functional, inputs = dn.func.make_functional(model, dt=DT)

    def initialize(parameters, constants):
        fresh = functional.initialize(parameters, constants, inputs.initialization)
        return functional.prepare_and_step(
            fresh.parameters, fresh.constants, fresh.state
        )[0]

    with dn.ctx(CHECK_PURITY=not checked), torch_compiler_warning_context():
        compiled = torch.compile(initialize, fullgraph=True, backend="aot_eager")
        result = compiled(inputs.parameters, inputs.constants)
    _assert_tree_close(result, initialize(inputs.parameters, inputs.constants))
