"""Adversarial contracts for delayed-state checkpoint metadata."""

from __future__ import annotations

import copy

import pytest
import torch

from dendra.models.mechanisms import Mechanism
from dendra.models.mechanisms._handler import (
    MechanismHandler,
    _validate_delayed_state_specs,
)


def _base(shape):
    return Mechanism(
        "delayed",
        torch.full(shape, 34.0, dtype=torch.float64),
        torch.ones(shape, dtype=torch.float64),
        shape,
        shape,
    )


def _single():
    mechanism = _base((2,))
    mechanism.register_delayed_state(
        "signal", torch.zeros(2, dtype=torch.float64), 2, mode="shift"
    )
    return mechanism


def _multi(delays=(0, 2)):
    mechanism = _base((2, 3))
    mechanism.register_delayed_states(
        "streams",
        torch.zeros(2, 3, dtype=torch.float64),
        delays,
        mode="shift",
        stream_axis=0,
    )
    return mechanism


def _payload(mechanism):
    return copy.deepcopy(mechanism._delayed_state_specs)


@pytest.mark.parametrize(
    "case,error,message",
    (
        ("payload_type", TypeError, "must be a mapping"),
        ("names", ValueError, "names do not match"),
        ("item_type", TypeError, "metadata must be a mapping"),
        ("fields", ValueError, "metadata fields do not match"),
        ("buffer_type", TypeError, "field 'buffer' must be a string"),
        ("buffer_name", ValueError, "must remain"),
        ("pointer_type", TypeError, "field 'pointer' must be a string"),
        ("steps_bool", TypeError, "steps must be an integer"),
        ("steps_negative", ValueError, "steps must be at least 0"),
        ("depth_zero", ValueError, "depth must be at least 1"),
        ("depth_inconsistent", ValueError, "depth 4 is inconsistent"),
        ("shape_type", TypeError, "value_shape must be a tuple or list"),
        ("shape_size_type", TypeError, r"value_shape\[0\] must be an integer"),
        ("shape_negative", ValueError, r"value_shape\[0\] must be at least 0"),
        ("shape_mismatch", ValueError, "does not match the live payload shape"),
        ("axis_type", TypeError, "axis must be an integer"),
        ("axis_bounds", ValueError, "axis 2 is invalid"),
        ("axis_mismatch", ValueError, "does not match the live axis"),
        ("mode_type", TypeError, "mode must be a string"),
        ("mode_invalid", ValueError, "invalid mode"),
        ("mode_mismatch", ValueError, "does not match the live mode"),
    ),
)
def test_single_delayed_state_metadata_rejects_every_malformed_field(
    case, error, message
):
    mechanism = _single()
    payload = _payload(mechanism)
    saved = payload["signal"]

    if case == "payload_type":
        payload = []
    elif case == "names":
        payload = {}
    elif case == "item_type":
        payload["signal"] = []
    elif case == "fields":
        saved.pop("depth")
    elif case == "buffer_type":
        saved["buffer"] = 1
    elif case == "buffer_name":
        saved["buffer"] = "other"
    elif case == "pointer_type":
        saved["pointer"] = None
    elif case == "steps_bool":
        saved["steps"] = True
    elif case == "steps_negative":
        saved["steps"] = -1
    elif case == "depth_zero":
        saved["depth"] = 0
    elif case == "depth_inconsistent":
        saved["depth"] = 4
    elif case == "shape_type":
        saved["value_shape"] = 2
    elif case == "shape_size_type":
        saved["value_shape"] = ("2",)
    elif case == "shape_negative":
        saved["value_shape"] = (-1,)
    elif case == "shape_mismatch":
        saved["value_shape"] = (3,)
    elif case == "axis_type":
        saved["axis"] = True
    elif case == "axis_bounds":
        saved["axis"] = 2
    elif case == "axis_mismatch":
        saved["axis"] = 0
    elif case == "mode_type":
        saved["mode"] = 1
    elif case == "mode_invalid":
        saved["mode"] = "unknown"
    elif case == "mode_mismatch":
        saved["mode"] = "circular"

    with pytest.raises(error, match=message):
        _validate_delayed_state_specs(mechanism, payload, label="mechanism")


@pytest.mark.parametrize(
    "case,error,message",
    (
        ("batched_type", TypeError, "field 'batched' must be a bool"),
        ("batched_kind", ValueError, "changed its batched registration kind"),
        ("steps_buffer_type", TypeError, "field 'steps_buffer' must be a string"),
        ("steps_buffer_name", ValueError, "must remain"),
        ("n_streams_zero", ValueError, "n_streams must be at least 1"),
        ("n_streams_mismatch", ValueError, "does not match the live stream count"),
        ("stream_axis_type", TypeError, "stream_axis must be an integer"),
        ("stream_axis_mismatch", ValueError, "does not match the live value"),
        ("value_stream_axis_mismatch", ValueError, "does not match the live value"),
        ("zero_flag_type", TypeError, "field 'has_zero_delay' must be a bool"),
        ("all_flag_type", TypeError, "field 'all_zero_delay' must be a bool"),
    ),
)
def test_batched_delayed_state_metadata_rejects_incompatible_registration(
    case, error, message
):
    mechanism = _multi()
    payload = _payload(mechanism)
    saved = payload["streams"]

    if case == "batched_type":
        saved["batched"] = 1
    elif case == "batched_kind":
        saved["batched"] = False
    elif case == "steps_buffer_type":
        saved["steps_buffer"] = 1
    elif case == "steps_buffer_name":
        saved["steps_buffer"] = "other"
    elif case == "n_streams_zero":
        saved["n_streams"] = 0
    elif case == "n_streams_mismatch":
        saved["n_streams"] = 3
    elif case == "stream_axis_type":
        saved["stream_axis"] = True
    elif case == "stream_axis_mismatch":
        saved["stream_axis"] = 0
    elif case == "value_stream_axis_mismatch":
        saved["value_stream_axis"] = 1
    elif case == "zero_flag_type":
        saved["has_zero_delay"] = 1
    elif case == "all_flag_type":
        saved["all_zero_delay"] = 0

    with pytest.raises(error, match=message):
        _validate_delayed_state_specs(mechanism, payload, label="mechanism")


def _assert_nested_equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "case,error,message",
    (
        ("pointer_float", TypeError, "must have integer dtype"),
        ("pointer_range", ValueError, "must be between"),
        ("steps_shape", ValueError, "has shape"),
        ("steps_float", TypeError, "must have integer dtype"),
        ("steps_negative", ValueError, "non-negative delays"),
        ("steps_max", ValueError, "maximum 1"),
        ("zero_flag", ValueError, "has_zero_delay is inconsistent"),
    ),
)
def test_handler_delayed_state_preflight_is_atomic(case, error, message):
    mechanism = _multi()
    handler = MechanismHandler(
        torch.full((2, 3), 34.0, dtype=torch.float64),
        torch.ones((2, 3), dtype=torch.float64),
        {"multi": mechanism},
    )
    valid = handler.mutable_state_dict()
    corrupt = copy.deepcopy(valid)
    pointer_key = "multi.streams_delay_ptr"
    steps_key = "multi.streams_delay_steps"

    if case == "pointer_float":
        corrupt[pointer_key] = corrupt[pointer_key].to(torch.float32)
    elif case == "pointer_range":
        corrupt[pointer_key] = torch.tensor(3, dtype=torch.long)
    elif case == "steps_shape":
        corrupt[steps_key] = torch.tensor([0, 1, 2], dtype=torch.long)
    elif case == "steps_float":
        corrupt[steps_key] = corrupt[steps_key].to(torch.float32)
    elif case == "steps_negative":
        corrupt[steps_key] = torch.tensor([-1, 2], dtype=torch.long)
    elif case == "steps_max":
        corrupt[steps_key] = torch.tensor([0, 1], dtype=torch.long)
    elif case == "zero_flag":
        corrupt[steps_key] = torch.tensor([1, 2], dtype=torch.long)

    before = handler.mutable_state_dict()
    with pytest.raises(error, match=message):
        handler.restore_mutable_state_dict(corrupt)
    _assert_nested_equal(handler.mutable_state_dict(), before)
