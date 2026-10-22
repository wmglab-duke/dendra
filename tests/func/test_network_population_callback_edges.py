"""Population-scoped callbacks preserve binding, validation, and ownership."""

from __future__ import annotations

import gc
import weakref
from dataclasses import FrozenInstanceError, dataclass

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._callback_carry import _CallbackCarryPartition
from tests.func.test_network import _close
from tests.func.test_network_recorder_regions import DT, _network
from tests.func.test_network_while_edges import _backend, _has_while

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _case():
    return dn.func.make_functional_network(_network(batch=None))


def _scoped(population, callback):
    return dn.func.PopulationCallback(population, callback)


@dataclass(frozen=True)
class _ProjectionProbe(dn.func.FunctionalCallback):
    """Require the selected Population's actual state and auxiliary mapping."""

    def _bind(self, functional):
        assert isinstance(functional, dn.func.FunctionalPopulation)
        assert functional.shape[-1] == 2
        return self

    def initialize(self, state, auxiliary):
        assert auxiliary is None
        assert "populations" not in state and "netstim" not in state
        voltage = state["integrator"]["v"]
        assert voltage.shape[-1] == 2
        return {"sum": torch.zeros_like(voltage), "count": voltage.new_zeros(())}, None

    def update(self, carry, state, auxiliary):
        assert "populations" not in auxiliary and "netstim" not in auxiliary
        return {
            "sum": carry["sum"] + auxiliary["custom"] + state["integrator"]["v"],
            "count": carry["count"] + 1,
        }, None

    def finalize(self, carry, emissions):
        assert emissions is None
        return {"selected": (carry["sum"], carry["count"])}


@dataclass(frozen=True)
class _ChangesCarry(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        del auxiliary
        return state["integrator"]["v"].new_zeros(()), None

    def update(self, carry, state, auxiliary):
        del state, auxiliary
        return carry.to(torch.int64), None

    def finalize(self, carry, emissions):
        assert emissions is None
        return carry


class _CustomAnomaly(dn.func.AnomalyDetector):
    """A subclass remains user-authored even when its numerical rule agrees."""

    def update(self, carry, state, auxiliary):
        return super().update(carry, state, auxiliary)


def test_population_callback_configuration_is_frozen():
    recorder = dn.func.Recorder("v")
    callback = _scoped("left", recorder)
    assert callback.population == "left" and callback.callback is recorder
    with pytest.raises(FrozenInstanceError):
        callback.population = "right"


@pytest.mark.parametrize("population", ("", None, 1, True))
def test_population_callback_rejects_invalid_population_names(population):
    with pytest.raises((TypeError, ValueError), match="population|Population"):
        _scoped(population, dn.func.Recorder("v"))


@pytest.mark.parametrize("callback", (None, {}, object()))
def test_population_callback_requires_a_functional_callback(callback):
    with pytest.raises(TypeError, match="FunctionalCallback|callback"):
        _scoped("left", callback)


def test_population_callback_rejects_nested_scopes():
    with pytest.raises(
        (ValueError, dn.func.FunctionalizationError), match="nested|scope"
    ):
        _scoped("left", _scoped("right", dn.func.Recorder("v")))


@pytest.mark.parametrize("population", ("missing", "left.soma", "populations.left"))
def test_population_scope_selects_exact_top_level_network_keys(population):
    functional, _ = _case()
    with pytest.raises(dn.func.FunctionalizationError, match="Population|population"):
        functional.make_callbacks({"trace": _scoped(population, dn.func.Recorder("v"))})


def test_population_scope_is_rejected_by_a_standalone_population_plan():
    functional, _ = _case()
    with pytest.raises(dn.func.FunctionalizationError, match="Network|network"):
        functional.populations["left"].make_callbacks(
            {"trace": _scoped("left", dn.func.Recorder("v"))}
        )


def test_scoped_callback_receives_substate_subauxiliary_and_none_initial_auxiliary():
    functional, tensors = _case()
    callbacks = functional.make_callbacks(
        {"summary": _scoped("right", _ProjectionProbe())}
    )
    carry, initial = callbacks._initialize(tensors.state)
    assert initial is None
    selected = tensors.state["populations"]["right"]["integrator"]["v"]
    custom = torch.full_like(selected, 7)
    next_carry, emitted = callbacks._update(
        carry,
        tensors.state,
        {
            "populations": {"left": {"custom": None}, "right": {"custom": custom}},
            "netstim": object(),
        },
    )
    assert emitted is None
    results = callbacks._finalize(next_carry, callbacks._stack([]))
    assert tuple(results) == ("summary",)
    total, count = results["summary"]["selected"]
    torch.testing.assert_close(total, selected + custom)
    assert count.item() == 1
    assert not callbacks._plans[0].initial_emits
    assert callbacks._plans[0].validate_updates


def test_scoped_custom_callback_retains_update_schema_validation():
    functional, tensors = _case()
    callbacks = functional.make_callbacks({"broken": _scoped("left", _ChangesCarry())})
    carry, _ = callbacks._initialize(tensors.state)
    with pytest.raises(dn.func.FunctionalizationError, match="must use"):
        callbacks._update(carry, tensors.state, {"populations": {"left": {}}})


def test_scoped_recorder_keeps_relative_names_and_separate_outer_result_keys():
    functional, tensors = _case()
    callbacks = functional.make_callbacks(
        {
            "left_trace": _scoped("left", dn.func.Recorder(("v", "t"))),
            "right_trace": _scoped("right", dn.func.Recorder(("v", "t"))),
            "qualified": dn.func.Recorder(("left.v", "right.v", "t")),
        }
    )
    carry, emitted = callbacks._initialize(tensors.state)
    results = callbacks._finalize(carry, callbacks._stack([emitted]))
    assert tuple(results) == ("left_trace", "right_trace", "qualified")
    for name in ("left", "right"):
        assert tuple(results[f"{name}_trace"]) == ("v", "t")
        torch.testing.assert_close(
            results[f"{name}_trace"]["v"], results["qualified"][f"{name}.v"]
        )
        assert results[f"{name}_trace"]["v"].shape[-1] == (3 if name == "left" else 2)


def test_scoped_recorder_cannot_reach_a_foreign_population_or_region():
    functional, _ = _case()
    for recorder in (
        dn.func.Recorder("right.v"),
        dn.func.Recorder("v", region=functional._source.populations["right"][:]),
    ):
        with pytest.raises(
            dn.func.FunctionalizationError, match="selection|Population"
        ):
            functional.make_callbacks({"trace": _scoped("left", recorder)})


def test_scoped_threshold_configuration_uses_the_selected_compartment_count():
    functional, _ = _case()
    functional.make_callbacks(
        {"count": _scoped("left", dn.func.APCount(node_check=[2]))}
    )
    with pytest.raises(IndexError, match="compartment"):
        functional.make_callbacks(
            {"count": _scoped("right", dn.func.APCount(node_check=[2]))}
        )


def _builtins(functional):
    source = functional._source.populations["left"]
    return {
        "trace": dn.func.Recorder("v"),
        "field": dn.func.LFP([torch.ones(source.shape, dtype=source.dtype())]),
        "anomaly": dn.func.AnomalyDetector(),
        "raster": dn.func.Raster(node_check=[0]),
        "count": dn.func.APCount(node_check=[0]),
        "active": dn.func.Active(node_check=[0]),
        "sites": dn.func.ActiveAL(node_check=[0]),
        "events": dn.func.ActiveALCount(node_check=[0]),
    }


def test_scoped_builtin_classification_keeps_lfp_invariant_and_fast_updates():
    functional, tensors = _case()
    specifications = _builtins(functional)
    callbacks = functional.make_callbacks(
        {name: _scoped("left", callback) for name, callback in specifications.items()}
    )
    assert all(not plan.validate_updates for plan in callbacks._plans)
    partition = _CallbackCarryPartition.from_plans(callbacks._plans)
    assert partition.invariant_names == ("field",)
    carry, _ = callbacks._initialize(tensors.state)
    recurrent, invariant = partition.split(carry.carries)
    assert tuple(invariant) == ("field",)
    assert "field" not in recurrent
    merged = partition.merge(recurrent, invariant)
    assert tuple(merged) == tuple(specifications)
    assert all(merged[name] is carry[name] for name in merged)


@pytest.mark.parametrize("steps", (1, 4))
@pytest.mark.parametrize("disable", (False, True))
def test_scoped_custom_callbacks_reject_while_before_short_or_disabled_fallback(
    steps, disable
):
    functional, tensors = _case()
    callbacks = functional.make_callbacks(
        {"summary": _scoped("right", _ProjectionProbe())}
    )
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    chunk = functional.compile_rollout_chunk(
        steps, execution="while", backend="eager", disable=disable
    )
    with torch.no_grad(), pytest.raises(dn.func.FunctionalizationError, match="custom"):
        chunk(tensors.parameters, prepared, tensors.state, callbacks=callbacks)
    assert chunk.execution_report() is None


def test_scoped_builtin_subclass_keeps_custom_validation_and_while_policy():
    functional, _ = _case()
    callbacks = functional.make_callbacks(
        {"anomaly": _scoped("left", _CustomAnomaly())}
    )
    assert callbacks._plans[0].validate_updates
    chunk = functional.compile_rollout_chunk(4, execution="while", backend="eager")
    with pytest.raises(dn.func.FunctionalizationError, match="custom"):
        chunk._validate_while_callbacks(callbacks)


@pytest.mark.parametrize(
    "kind", ("Raster", "APCount", "Active", "ActiveAL", "ActiveALCount")
)
def test_scoped_threshold_callbacks_keep_both_runtime_dt_checks(kind):
    functional, _ = _case()
    callback = getattr(dn.func, kind)(node_check=[0], dt=DT)
    callbacks = functional.make_callbacks({"count": _scoped("left", callback)})
    callbacks._validate_runtime_dt(DT)
    callbacks._validate_runtime_dt_tensor(torch.tensor(DT, dtype=torch.float64))
    with pytest.raises(dn.func.FunctionalizationError, match="timestep"):
        callbacks._validate_runtime_dt(2 * DT)
    with pytest.raises(dn.func.FunctionalizationError, match="timestep"):
        callbacks._validate_runtime_dt_tensor(torch.tensor(2 * DT, dtype=torch.float64))
    with pytest.raises(dn.func.FunctionalizationError, match="dt|timestep"):
        functional.make_callbacks(
            {
                "count": _scoped(
                    "left", getattr(dn.func, kind)(node_check=[0], dt=2 * DT)
                )
            }
        )


def test_scoped_callback_continuation_tokens_still_identify_the_exact_collection():
    functional, tensors = _case()
    specifications = {"count": _scoped("left", dn.func.APCount(node_check=[0]))}
    first = functional.make_callbacks(specifications)
    second = functional.make_callbacks(specifications)
    carry, _ = first._initialize(tensors.state)
    assert first._validate_state(carry) is carry
    with pytest.raises(dn.func.FunctionalizationError, match="different callback plan"):
        second._validate_state(carry)
    other, _ = _case()
    with pytest.raises(dn.func.FunctionalizationError, match="different functional"):
        first._validate_owner(other)


def test_scoped_recorder_binding_does_not_retain_public_scope_or_configuration():
    functional, tensors = _case()

    def bind():
        recorder = dn.func.Recorder("v")
        scope = _scoped("left", recorder)
        callbacks = functional.make_callbacks({"trace": scope})
        return callbacks, weakref.ref(scope), weakref.ref(recorder)

    callbacks, scope_ref, recorder_ref = bind()
    gc.collect()
    assert scope_ref() is None and recorder_ref() is None
    _, emitted = callbacks._initialize(tensors.state)
    torch.testing.assert_close(
        emitted["trace"]["v"], tensors.state["populations"]["left"]["integrator"]["v"]
    )


def test_bare_population_builtins_still_require_an_explicit_network_scope():
    functional, _ = _case()
    for name, callback in _builtins(functional).items():
        if name == "trace":
            continue
        with pytest.raises(
            dn.func.FunctionalizationError, match="Population|population"
        ):
            functional.make_callbacks({name: callback})


def test_scoped_anomaly_ignores_initial_nonfinite_then_stays_local_and_sticky():
    functional, tensors = _case()
    healthy = tensors.state
    selected = healthy["populations"]["right"]
    voltage = selected["integrator"]["v"]
    unhealthy = {
        **healthy,
        "populations": {
            **healthy["populations"],
            "right": {
                **selected,
                "integrator": {
                    **selected["integrator"],
                    "v": voltage.new_tensor([[float("nan"), float("inf")]]),
                },
            },
        },
    }
    callbacks = functional.make_callbacks(
        {
            name: _scoped(name, dn.func.AnomalyDetector())
            for name in functional.populations
        }
    )
    carry, initial = callbacks._initialize(unhealthy)
    assert initial is None
    assert not carry["left"].any() and not carry["right"].any()
    auxiliary = {"populations": {"left": {}, "right": {}}}
    detected, emitted = callbacks._update(carry, unhealthy, auxiliary)
    assert emitted is None
    assert not detected["left"].any() and detected["right"].all()
    resumed = callbacks._validate_state(detected)
    recovered, emitted = callbacks._update(resumed, healthy, auxiliary)
    assert emitted is None
    results = callbacks._finalize(recovered, callbacks._stack([]))
    assert not results["left"].any() and results["right"].all()


def test_scoped_reducers_alone_use_a_real_while_without_emission_storage():
    functional, tensors = _case()
    callbacks = functional.make_callbacks(
        {
            "count": _scoped(
                "left", dn.func.APCount(threshold=-100, node_check=[0, -1])
            ),
            "anomaly": _scoped("right", dn.func.AnomalyDetector()),
        }
    )
    initial, emitted = callbacks._initialize(tensors.state)
    assert emitted is None
    assert all(plan.emission_schema is None for plan in callbacks._plans)
    graphs = []
    chunk = functional.compile_rollout_chunk(
        4, execution="while", backend=_backend(graphs)
    )
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    with torch.no_grad():
        first, auxiliary = chunk(
            tensors.parameters, prepared, tensors.state, callbacks=callbacks
        )
        final, resumed = chunk(
            tensors.parameters,
            prepared,
            first,
            callbacks=callbacks,
            callback_state=auxiliary["callbacks"].state,
        )
        expected, reference = functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            steps=8,
            callbacks=callbacks,
        )
    assert chunk.execution_report().strategy == "compiled_while_loop"
    assert any(_has_while(graph) for graph in graphs)
    _close(final, expected)
    _close(resumed["callbacks"].outputs, reference["callbacks"].outputs)
    assert (resumed["callbacks"]["count"] == 1).all()
    assert not resumed["callbacks"]["anomaly"].any()
    initial_leaves = torch.utils._pytree.tree_leaves(initial.carries)
    final_leaves = torch.utils._pytree.tree_leaves(resumed["callbacks"].state.carries)
    assert [(value.shape, value.dtype) for value in final_leaves] == [
        (value.shape, value.dtype) for value in initial_leaves
    ]
    assert resumed["callbacks"].state.emission_schemas == (None, None)
