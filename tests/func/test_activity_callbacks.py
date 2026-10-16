"""Hard activity reducers retain the imperative threshold-family semantics."""

import types

import pytest
import torch
from test_callbacks import DT, _functional_case, _model, _threshold_sequence_run

import dendra as dn
from dendra.models.callbacks import Active as ImperativeActive
from dendra.models.callbacks import ActiveAL as ImperativeActiveAL
from dendra.models.callbacks import ActiveALCount as ImperativeActiveALCount

FAMILIES = [
    (dn.func.Active, ImperativeActive, {}),
    (dn.func.ActiveAL, ImperativeActiveAL, {"at_least": 2}),
    (dn.func.ActiveALCount, ImperativeActiveALCount, {"at_least": 3}),
]


def _voltage(shape):
    values = torch.full((6, *shape), -1.0, dtype=torch.float64)
    first = values.new_tensor([1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    last = values.new_tensor([-1.0, -1.0, -1.0, 0.0, 1.0, 1.0])
    view = (6, *([1] * (values.ndim - 2)))
    values[..., 0] = first.reshape(view)
    values[..., -1] = last.reshape(view)
    return values


def _imperative(callback, voltage, *, partition=None):
    model = types.SimpleNamespace(
        v=voltage[0],
        shape=voltage.shape[1:],
        nc=voltage.shape[-1],
        device=lambda: voltage.device,
    )
    callback.pre_loop_hook(model)
    for frame in voltage:
        model.v = frame
        callback.post_step_hook(model)
    return callback.is_active(partition), callback.record


@pytest.mark.parametrize("runner", ["run", "longrun", "longrun_checkpointed"])
@pytest.mark.parametrize("functional_type,imperative_type,options", FAMILIES)
@pytest.mark.parametrize("inv", [False, True])
def test_activity_matches_imperative_crossings(
    runner, functional_type, imperative_type, options, inv
):
    source, functional, tensors = _functional_case()
    voltage = _voltage(source.shape)
    configuration = {"node_check": [0, -1], "threshold": 0.0, "inv": inv, **options}
    callbacks = functional.make_callbacks({"active": functional_type(**configuration)})
    _final, auxiliary = _threshold_sequence_run(
        runner, functional, tensors, callbacks, voltage
    )
    expected, counts = _imperative(imperative_type(**configuration, dt=DT), voltage)
    actual = auxiliary["callbacks"]["active"]
    assert actual.dtype == torch.bool
    assert actual.shape == source.shape[:-1]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        auxiliary["callbacks"].state["active"][1], counts.to(torch.int64)
    )


def test_activity_distinguishes_distinct_slots_from_total_spike_count():
    source, functional, tensors = _functional_case()
    voltage = _voltage(source.shape)
    voltage[..., -1] = -1.0
    callbacks = functional.make_callbacks(
        {
            "any": dn.func.Active(node_check=[0, -1]),
            "sites": dn.func.ActiveAL(node_check=[0, -1], at_least=2),
            "count": dn.func.ActiveALCount(node_check=[0, -1], at_least=3),
            "count_too_high": dn.func.ActiveALCount(node_check=[0, -1], at_least=4),
        }
    )
    _final, auxiliary = _threshold_sequence_run(
        "run", functional, tensors, callbacks, voltage
    )
    results = auxiliary["callbacks"]
    assert results["any"].all()
    assert not results["sites"].any()
    assert results["count"].all()
    assert not results["count_too_high"].any()


@pytest.mark.parametrize("functional_type,imperative_type,options", FAMILIES)
def test_activity_window_first_checked_above_and_continuation(
    functional_type, imperative_type, options
):
    source, functional, tensors = _functional_case()
    voltage = _voltage(source.shape)
    # Both sites are already above threshold when checking starts. The cache
    # remains armed outside the check window, including the unobserved initial
    # frame, so each site contributes a crossing at i=2.
    voltage[:3, ..., 0] = 1.0
    voltage[:3, ..., -1] = 1.0
    configuration = {
        "node_check": [0, -1],
        "t_start_check": 2 * DT,
        "t_end_check": 5 * DT,
        **options,
    }
    callbacks = functional.make_callbacks({"active": functional_type(**configuration)})
    first_state, first_auxiliary = _threshold_sequence_run(
        "longrun", functional, tensors, callbacks, voltage[:3]
    )
    _final, second_auxiliary = _threshold_sequence_run(
        "longrun_checkpointed",
        functional,
        tensors,
        callbacks,
        voltage[3:],
        state=first_state,
        callback_state=first_auxiliary["callbacks"].state,
    )
    _final, complete_auxiliary = _threshold_sequence_run(
        "run", functional, tensors, callbacks, voltage
    )
    expected, _counts = _imperative(imperative_type(**configuration, dt=DT), voltage)
    torch.testing.assert_close(second_auxiliary["callbacks"]["active"], expected)
    torch.testing.assert_close(
        second_auxiliary["callbacks"]["active"],
        complete_auxiliary["callbacks"]["active"],
    )
    for actual, expected_carry in zip(
        second_auxiliary["callbacks"].state["active"],
        complete_auxiliary["callbacks"].state["active"],
        strict=True,
    ):
        torch.testing.assert_close(actual, expected_carry)
    assert second_auxiliary["callbacks"].state["active"][2].item() == 6


def test_activity_does_not_check_the_initial_voltage():
    source, functional, tensors = _functional_case()
    state = dict(tensors.state)
    state["integrator"] = dict(state["integrator"])
    state["integrator"]["v"] = torch.ones_like(state["integrator"]["v"])
    callbacks = functional.make_callbacks({"active": dn.func.Active(node_check=[0])})
    voltage = torch.full((1, *source.shape), -1.0, dtype=torch.float64)
    _final, auxiliary = _threshold_sequence_run(
        "run", functional, tensors, callbacks, voltage, state=state
    )
    assert not auxiliary["callbacks"]["active"].any()


@pytest.mark.parametrize("functional_type,imperative_type,options", FAMILIES)
@pytest.mark.parametrize("inv", [False, True])
def test_activity_partitions_duplicates_and_repeated_batch_axes(
    functional_type, imperative_type, options, inv
):
    model = _model()
    model.batch(2)
    model.batch(3)
    model.initialize()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    voltage = _voltage(model.shape)
    voltage[:, 0, 0, ..., :] = -1.0
    # Duplicate slots intentionally contribute independently, matching the
    # imperative slot-based contract even though they select the same nodes.
    configuration = {"node_check": [0, 0, -1, 0, 0, -1], "inv": inv, **options}
    partition = [3, 3]
    callbacks = functional.make_callbacks(
        {"active": functional_type(**configuration, partition=partition)}
    )
    partition[:] = [1, 5]  # Binding snapshots immutable static grouping.
    _final, auxiliary = _threshold_sequence_run(
        "run", functional, tensors, callbacks, voltage
    )
    expected, _counts = _imperative(
        imperative_type(**configuration, dt=DT), voltage, partition=[3, 3]
    )
    actual = auxiliary["callbacks"]["active"]
    assert actual.shape == (*model.shape[:-1], 2)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("functional_type", [dn.func.ActiveAL, dn.func.ActiveALCount])
@pytest.mark.parametrize("target", [0, -1])
def test_activity_rejects_nonpositive_targets(functional_type, target):
    with pytest.raises(ValueError, match="positive integer"):
        functional_type(at_least=target)


@pytest.mark.parametrize("functional_type", [dn.func.ActiveAL, dn.func.ActiveALCount])
@pytest.mark.parametrize("target", [True, 1.5, "2"])
def test_activity_rejects_nonintegral_targets(functional_type, target):
    with pytest.raises(TypeError, match="positive integer"):
        functional_type(at_least=target)


@pytest.mark.parametrize("functional_type", [dn.func.ActiveAL, dn.func.ActiveALCount])
@pytest.mark.parametrize(
    "partition,message",
    [([[2, 2]], "1D"), ([], "non-empty"), ([2, 1], "sum"), ([1, 3], "at least")],
)
def test_activity_rejects_invalid_partitions(functional_type, partition, message):
    _source, functional, _tensors = _functional_case()
    with pytest.raises(ValueError, match=message):
        functional.make_callbacks(
            {
                "active": functional_type(
                    node_check=[0, 0, -1, -1], at_least=2, partition=partition
                )
            }
        )


@pytest.mark.parametrize("functional_type,imperative_type,options", FAMILIES)
def test_activity_empty_selection_and_out_of_range_nodes(
    functional_type, imperative_type, options
):
    source, functional, tensors = _functional_case()
    callbacks = functional.make_callbacks(
        {
            "active": functional_type(node_check=[], **options),
            "inactive": functional_type(node_check=[], inv=True, **options),
        }
    )
    _final, auxiliary = _threshold_sequence_run(
        "run", functional, tensors, callbacks, _voltage(source.shape)
    )
    assert not auxiliary["callbacks"]["active"].any()
    assert auxiliary["callbacks"]["inactive"].all()
    for nodes in ([source.nc], [-source.nc - 1]):
        with pytest.raises(IndexError, match="valid compartment indices"):
            functional.make_callbacks({"invalid": functional_type(node_check=nodes)})


@pytest.mark.parametrize("functional_type,imperative_type,options", FAMILIES)
def test_activity_update_and_finalize_are_fullgraph_and_vmap_safe(
    functional_type, imperative_type, options
):
    source, functional, tensors = _functional_case()
    callback = functional_type(node_check=[0, -1], **options)._bind(functional)

    def execute(voltage):
        state = {"integrator": {"v": voltage[0]}}
        carry, _ = callback.initialize(state, None)
        for frame in voltage:
            state = {"integrator": {"v": frame}}
            carry, _ = callback.update(carry, state, None)
        return callback.finalize(carry, None)

    voltage = _voltage(source.shape)
    expected = execute(voltage)
    compiled = torch.compile(execute, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(voltage), expected)
    batched = torch.stack([voltage, torch.full_like(voltage, -1.0)])
    torch.testing.assert_close(
        torch.vmap(execute)(batched),
        torch.stack([expected, torch.zeros_like(expected)]),
    )
