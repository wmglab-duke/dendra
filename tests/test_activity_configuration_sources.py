"""Shared activity target validation and native configuration projection."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dendra.models.callbacks import (
    Active,
    ActiveAL,
    ActiveALCount,
    _bind_callback_dt,
)

pytestmark = pytest.mark.cpu
_FAMILIES = ((Active, False), (ActiveAL, False), (ActiveALCount, True))
_INVALID_TARGETS = (
    pytest.param(0, ValueError, id="zero"),
    pytest.param(-2, ValueError, id="negative"),
    pytest.param(False, TypeError, id="false"),
    pytest.param(True, TypeError, id="true"),
    pytest.param(1.0, TypeError, id="integral-float"),
    pytest.param(1.5, TypeError, id="fractional-float"),
    pytest.param("2", TypeError, id="string"),
    pytest.param(None, TypeError, id="none"),
    pytest.param(np.float64(2.0), TypeError, id="numpy-float"),
    pytest.param(np.bool_(True), TypeError, id="numpy-bool"),
    pytest.param(torch.tensor(2), TypeError, id="tensor"),
    pytest.param(torch.nn.Parameter(torch.tensor(2.0)), TypeError, id="parameter"),
)


def _callback(family, *, at_least=2, inv=False, **configuration):
    if family is Active:
        callback = family(inv=inv, **configuration)
        # Active inherits the same mutable target, without a constructor argument.
        callback.at_least = at_least
        return callback
    return family(at_least=at_least, inv=inv, **configuration)


def _counts():
    return torch.tensor([[2, 0, 0, 1], [0, 0, 3, 0]], dtype=torch.int32)


def _model():
    voltage = torch.zeros((2, 3), dtype=torch.float64)
    return SimpleNamespace(
        v=voltage,
        shape=tuple(voltage.shape),
        nc=voltage.shape[-1],
        device=lambda: voltage.device,
    )


@pytest.mark.parametrize("family", (ActiveAL, ActiveALCount))
@pytest.mark.parametrize("at_least,error", _INVALID_TARGETS)
def test_activity_constructors_reject_nonpositive_or_noninteger_targets(
    family, at_least, error
):
    with pytest.raises(error, match="^at_least must be a positive integer$"):
        family(at_least=at_least)


@pytest.mark.parametrize("family,total", _FAMILIES)
@pytest.mark.parametrize("at_least,error", _INVALID_TARGETS)
def test_edited_activity_targets_are_validated_before_query_partition(
    family, total, at_least, error
):
    callback = _callback(family, node_check=[0, 0, -1, -1])
    callback.at_least = at_least
    callback.record = _counts()
    record = callback.record
    before = record.clone()
    # The invalid target takes precedence over the invalid partition shape.
    with pytest.raises(error, match="^at_least must be a positive integer$"):
        callback.is_active([[2, 2]])
    with pytest.raises(error, match="^at_least must be a positive integer$"):
        callback.numpy()
    assert callback.record is record
    torch.testing.assert_close(record, before)


@pytest.mark.parametrize("family", (ActiveAL, ActiveALCount))
@pytest.mark.parametrize("at_least", (1, 2, np.int64(2), np.uint64(3)))
def test_activity_constructors_normalize_positive_integral_targets(family, at_least):
    callback = family(at_least=at_least)
    configuration = callback._functional_activity_configuration()
    assert type(callback.at_least) is int
    assert callback.at_least == int(at_least)
    assert configuration["at_least"] is callback.at_least
    assert configuration["inv"] is False


@pytest.mark.parametrize("family,total", _FAMILIES)
@pytest.mark.parametrize("at_least", (1, 2, 4, np.int64(2)))
def test_positive_integral_queries_preserve_reduction_and_native_history(
    family, total, at_least
):
    callback = _callback(family, at_least=at_least, node_check=[0, 0, -1, -1])
    callback.record = _counts()
    counts = callback.record
    measure = counts.sum(-1) if total else torch.count_nonzero(counts, dim=-1)
    expected = measure >= int(at_least)
    actual = callback.is_active()
    assert actual.dtype == torch.bool and not actual.requires_grad
    torch.testing.assert_close(actual, expected)
    assert callback.numpy().tolist() == expected.tolist()
    assert callback.record is counts
    torch.testing.assert_close(counts, _counts())


@pytest.mark.parametrize("family,total", _FAMILIES)
@pytest.mark.parametrize("inv", (False, True))
def test_projection_leaves_posthoc_partition_queries_and_numpy_unchanged(
    family, total, inv
):
    callback = _callback(family, inv=inv, node_check=[0, 0, -1, -1])
    callback.record = _counts()
    record = callback.record
    before = record.clone()
    configuration = callback._functional_activity_configuration()
    assert "partition" not in configuration
    expected = (
        torch.tensor([[True, False], [False, True]])
        if total
        else torch.zeros((2, 2), dtype=torch.bool)
    )
    if inv:
        expected = ~expected
    torch.testing.assert_close(callback.is_active([2, 2]), expected)
    assert callback.numpy([2, 2]).tolist() == expected.tolist()
    assert callback.record is record
    torch.testing.assert_close(record, before)


@pytest.mark.parametrize("family,total", _FAMILIES)
@pytest.mark.parametrize("configured_dt", (None, 0.125))
def test_projection_preserves_authored_threshold_settings_and_native_history(
    family, total, configured_dt
):
    nodes = [0, 0, -1, -1]
    callback = _callback(
        family,
        threshold=-20.0,
        t_start_check=0.1,
        t_end_check=0.8,
        node_check=nodes,
        dt=configured_dt,
    )
    _bind_callback_dt(callback, 0.2)
    callback.pre_loop_hook(_model())
    callback.record.copy_(_counts())
    callback.i = 7
    record, latch, published = (
        callback.record,
        callback.state_cache,
        callback.node_check,
    )
    frozen = tuple(value.clone() for value in (record, latch, published))
    configuration = callback._functional_activity_configuration()
    assert tuple(configuration) == (
        "threshold",
        "t_start_check",
        "t_end_check",
        "node_check",
        "dt",
        "at_least",
        "inv",
    )
    assert configuration["threshold"] == -20.0
    assert configuration["t_start_check"] == 0.1
    assert configuration["t_end_check"] == 0.8
    assert configuration["node_check"] is nodes
    assert configuration["dt"] == configured_dt
    assert configuration["at_least"] == 2
    assert callback.dt == 0.2 and callback.i == 7
    assert callback.record is record and callback.state_cache is latch
    assert callback.node_check is published
    torch.testing.assert_close((record, latch, published), frozen)


@pytest.mark.parametrize("family,total", _FAMILIES)
def test_projection_preserves_raw_inversion_parameter_without_new_aliases(
    family, total
):
    inv = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    callback = _callback(family, inv=inv, node_check=[0, -1])
    parameters = dict(callback.named_parameters())
    buffers = dict(callback.named_buffers())
    keys = tuple(callback.state_dict())
    configuration = callback._functional_activity_configuration()
    assert configuration["at_least"] == 2
    assert configuration["inv"] is inv
    current_parameters = dict(callback.named_parameters())
    current_buffers = dict(callback.named_buffers())
    assert tuple(current_parameters) == tuple(parameters)
    assert all(current_parameters[name] is value for name, value in parameters.items())
    assert tuple(current_buffers) == tuple(buffers)
    assert all(current_buffers[name] is value for name, value in buffers.items())
    assert tuple(callback.state_dict()) == keys == ("inv",)
    callback.record = torch.zeros((2, 2), dtype=torch.int32)
    torch.testing.assert_close(callback.is_active(), torch.tensor([False, False]))
    with torch.no_grad():
        inv.fill_(1.0)
    assert callback._functional_activity_configuration()["inv"] is inv
    torch.testing.assert_close(callback.is_active(), torch.tensor([True, True]))


def test_projection_reads_current_target_without_queries_or_truth_conversion():
    class Truth:
        def __bool__(self):
            raise AssertionError("source projection must not evaluate inversion")

    class CustomQuery(ActiveAL):
        def is_active(self, partition=None):
            raise AssertionError("source projection must not query native history")

        def numpy(self, partition=None):
            raise AssertionError("source projection must not query native history")

    inv = Truth()
    callback = CustomQuery(at_least=1, inv=inv, node_check=[0])
    callback.at_least = 2
    configuration = callback._functional_activity_configuration()
    assert configuration["at_least"] == 2 and configuration["inv"] is inv
    assert callback.record is None and callback.state_cache is None and callback.i == 0
