"""Native threshold execution keeps authored sources separate from workspaces."""

from __future__ import annotations

import pickle
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from dendra.models._threshold_callbacks import _COUNT_DTYPE, _restore_count
from dendra.models.backend import Backend
from dendra.models.callbacks import (
    AnomalyDetector,
    APCount,
    Callback,
    Raster,
    ThresholdCallback,
    _bind_callback_dt,
)

pytestmark = pytest.mark.cpu


def _model(compartments=4):
    return SimpleNamespace(
        v=torch.zeros((2, compartments), dtype=torch.float64),
        shape=(2, compartments),
        nc=compartments,
        device=lambda: torch.device("cpu"),
    )


@pytest.mark.parametrize("family", (APCount, Raster))
def test_native_normalization_retains_authored_negative_selectors_and_history(family):
    nodes = [0, -1, -1]
    callback = family(node_check=nodes, threshold=1.0, dt=0.1)
    callback.pre_loop_hook(_model(4))
    torch.testing.assert_close(callback.node_check, torch.tensor([0, 3, 3]))
    record, cache = callback.record, callback.state_cache
    callback.i = 17
    values = callback._functional_threshold_configuration()
    assert values["node_check"] is nodes
    assert values["threshold"] == 1.0 and values["dt"] == 0.1
    assert values["t_end_check"] == 1e9
    assert callback.record is record and callback.state_cache is cache
    assert callback.i == 17
    # Native reuse retains its published indices, just as ordinary callbacks
    # do. A functional binding can separately resolve the authored -1 anew.
    callback.pre_loop_hook(_model(6))
    torch.testing.assert_close(callback.node_check, torch.tensor([0, 3, 3]))
    assert callback._functional_threshold_configuration()["node_check"] is nodes


def test_source_projection_observes_raw_aliases_and_public_reassignments():
    nodes = torch.tensor([0, -1])
    callback = APCount(node_check=nodes)
    assert callback._functional_threshold_configuration()["node_check"] is nodes
    callback.pre_loop_hook(_model())
    nodes[0] = 1
    assert callback._functional_threshold_configuration()["node_check"] is nodes
    torch.testing.assert_close(
        callback._functional_threshold_configuration()["node_check"],
        torch.tensor([1, -1]),
    )
    assert callback.node_check.tolist() == [0, 3]
    replacement = [2, -1]
    callback.node_check = replacement
    assert callback._functional_threshold_configuration()["node_check"] is replacement
    replacement[0] = 1
    callback.pre_loop_hook(_model())
    assert callback.node_check.tolist() == [1, 3]
    assert callback._functional_threshold_configuration()["node_check"] is replacement


@pytest.mark.parametrize("inference", (False, True))
def test_edits_to_published_indices_become_authoritative_configuration(inference):
    callback = APCount(node_check=[0, -1])
    with torch.inference_mode(inference):
        callback.pre_loop_hook(_model())
        published = callback.node_check
        published[0] = 2
    assert callback._functional_threshold_configuration()["node_check"] is published
    callback.pre_loop_hook(_model(5))
    assert callback._functional_threshold_configuration()["node_check"] is published
    assert callback.node_check.tolist() == [2, 3]


@pytest.mark.parametrize("copy_kind", ("deepcopy", "pickle"))
def test_copy_preserves_authored_selectors_after_native_publication(copy_kind):
    def copy(value):
        return (
            deepcopy(value)
            if copy_kind == "deepcopy"
            else pickle.loads(pickle.dumps(value))
        )

    nodes = [0, -1]
    callback = APCount(node_check=nodes)
    callback.pre_loop_hook(_model())
    copied = copy(callback)
    assert copied.node_check is not callback.node_check
    assert copied._functional_threshold_configuration()["node_check"] == nodes
    copied.node_check[0] = 2
    assert (
        copied._functional_threshold_configuration()["node_check"] is copied.node_check
    )
    assert callback._functional_threshold_configuration()["node_check"] is nodes
    callback.node_check[0] = 1
    edited = copy(callback)
    assert (
        edited._functional_threshold_configuration()["node_check"] is edited.node_check
    )
    assert edited.node_check.tolist() == [1, 3]


@pytest.mark.parametrize("family", (APCount, Raster))
@pytest.mark.parametrize("copy_kind", ("deepcopy", "pickle"))
def test_legacy_callback_state_uses_effective_sources_when_authored_metadata_is_absent(
    family, copy_kind
):
    callback = family(node_check=[0, -1], t_start_check=0.2, t_end_check=0.5, dt=0.1)
    callback.pre_loop_hook(_model())
    callback.i = 17
    for name in (
        "_configured_dt",
        "_configured_node_check",
        "_normalized_node_check",
        "_normalized_node_check_version",
        "_normalized_node_check_snapshot",
        "_binding_node_check",
    ):
        callback.__dict__.pop(name, None)
    restored = (
        deepcopy(callback)
        if copy_kind == "deepcopy"
        else pickle.loads(pickle.dumps(callback))
    )
    assert "_configured_dt" not in restored.__dict__
    indices, record, cache = restored.node_check, restored.record, restored.state_cache
    configuration = restored._functional_threshold_configuration()
    assert configuration["dt"] == 0.1
    assert configuration["node_check"] is indices
    assert indices.tolist() == [0, 3]
    _bind_callback_dt(restored, 0.05)
    assert restored.dt == 0.05 and (restored.ind_start, restored.ind_end) == (4, 10)
    assert restored._functional_threshold_configuration()["dt"] == 0.1
    restored.pre_loop_hook(_model(6))
    assert restored.node_check.tolist() == [0, 3]
    assert restored._functional_threshold_configuration()["node_check"] is indices
    assert (
        restored.record is record and restored.state_cache is cache and restored.i == 17
    )


def test_tensor_subclass_indices_keep_dispatch_and_aliases():
    class Nodes(torch.Tensor):
        pass

    nodes = torch.tensor([0, -1]).as_subclass(Nodes)
    callback = APCount(node_check=nodes)
    callback.pre_loop_hook(_model())
    assert type(callback.node_check) is Nodes
    assert callback._functional_threshold_configuration()["node_check"] is nodes
    published = callback.node_check
    published[0] = 2
    assert callback._functional_threshold_configuration()["node_check"] is published
    assert type(callback.node_check) is Nodes


def test_authored_timestep_and_runtime_binding_have_distinct_ownership(monkeypatch):
    with monkeypatch.context() as context:
        context.setattr(Backend, "dt", 0.25)
        callback = APCount(node_check=[0], t_start_check=0.2, t_end_check=0.5)
    assert callback.dt == 0.25
    assert callback._functional_threshold_configuration()["dt"] is None
    _bind_callback_dt(callback, 0.1)
    assert callback.dt == 0.1 and (callback.ind_start, callback.ind_end) == (2, 5)
    assert callback._functional_threshold_configuration()["dt"] is None
    callback.dt = 0.05
    assert callback._functional_threshold_configuration()["dt"] == 0.05
    _bind_callback_dt(callback, 0.025)
    assert callback.dt == 0.025 and (callback.ind_start, callback.ind_end) == (8, 20)
    assert callback._functional_threshold_configuration()["dt"] == 0.05


def test_runtime_binding_preserves_public_setter_authority_and_source_on_failure():
    calls = []

    class Custom(ThresholdCallback):
        @ThresholdCallback.dt.setter
        def dt(self, value):
            calls.append(value)
            ThresholdCallback.dt.fset(self, value)
            if value == 0.05:
                raise RuntimeError("custom setter")

    callback = Custom(node_check=[0], dt=0.1)
    with pytest.raises(RuntimeError, match="custom setter"):
        _bind_callback_dt(callback, 0.05)
    assert calls == [0.05] and callback.dt == 0.05
    assert callback._functional_threshold_configuration()["dt"] == 0.1

    class Plain(Callback):
        @property
        def dt(self):
            return self._dt

        @dt.setter
        def dt(self, value):
            calls.append(value)
            self._dt = value

    plain = Plain()
    _bind_callback_dt(plain, 0.2)
    assert plain.dt == 0.2 and calls == [0.05, 0.2]


def test_source_metadata_adds_no_registered_parameters_or_buffers():
    nodes = torch.tensor([0, -1])
    callback = APCount(node_check=nodes)
    assert dict(callback.named_parameters()) == dict(callback.named_buffers()) == {}
    assert callback.state_dict() == {}
    callback.pre_loop_hook(_model())
    callback.to(dtype=torch.float32)
    assert callback._functional_threshold_configuration()["node_check"] is nodes
    assert callback.state_dict() == {}
    parameter = torch.nn.Parameter(torch.tensor([0, -1]), requires_grad=False)
    registered = APCount(node_check=parameter)
    assert tuple(registered.named_parameters()) == (("node_check", parameter),)
    assert tuple(registered.state_dict()) == ("node_check",)


def test_public_node_setter_is_not_bypassed_by_native_normalization():
    writes = []

    class Custom(ThresholdCallback):
        def __setattr__(self, name, value):
            if name == "node_check":
                writes.append(value)
            super().__setattr__(name, value)

    callback = Custom(node_check=[0, -1])
    callback.pre_loop_hook(_model())
    assert len(writes) == 3
    assert callback.node_check is writes[-1]
    assert callback._functional_threshold_configuration()["node_check"] is writes[0]


@pytest.mark.parametrize("shape", ((1, 1), (2, 1), (2, 1, 1), (2, 1, 3), (2, 3, 2, 1)))
def test_native_anomaly_flags_preserve_population_and_batch_axes(shape):
    voltage = torch.zeros(shape, dtype=torch.float64)
    model = SimpleNamespace(v=voltage)
    callback = AnomalyDetector()
    callback.pre_loop_hook(model)
    assert callback.rec.shape == shape[:-1] and callback.rec.dtype == torch.bool
    assert not callback.rec.any()
    voltage.reshape(-1)[0] = float("nan")
    callback.post_step_hook(model)
    expected = (~torch.isfinite(voltage)).any(dim=-1)
    torch.testing.assert_close(callback.rec, expected)
    voltage.fill_(0.0)
    callback.post_step_hook(model)
    torch.testing.assert_close(callback.rec, expected)
    voltage.reshape(-1)[-1] = float("inf")
    callback.post_step_hook(model)
    expected = expected | (~torch.isfinite(voltage)).any(dim=-1)
    torch.testing.assert_close(callback.rec, expected)
    callback.reset()
    assert callback.rec.shape == shape[:-1] and callback.rec.dtype == torch.bool
    assert not callback.rec.any()


def test_native_anomaly_initialization_does_not_observe_the_initial_voltage():
    model = SimpleNamespace(v=torch.full((2, 1, 1), float("nan")))
    callback = AnomalyDetector()
    callback.pre_loop_hook(model)
    assert callback.rec.shape == (2, 1) and not callback.rec.any()
    callback.post_step_hook(model)
    assert callback.rec.all()


@pytest.mark.parametrize("shape", ((2, 4), (1, 1), (2, 3, 1, 4)))
@pytest.mark.parametrize("steps", (0, 3), ids=("zero_step", "outside_window"))
def test_native_raster_empty_window_has_bound_boolean_shape(shape, steps):
    voltage = torch.ones(shape, dtype=torch.float64)
    model = SimpleNamespace(
        v=voltage,
        shape=shape,
        nc=shape[-1],
        device=lambda: voltage.device,
    )
    callback = Raster(node_check=[0, -1], t_start_check=1.0, t_end_check=2.0, dt=0.1)
    callback.pre_loop_hook(model)
    record = callback.record
    for _ in range(steps):
        callback.post_step_hook(model)
    observed = callback.stack()
    assert observed.shape == (0, *shape[:-1], 2)
    assert observed.dtype == torch.bool and observed.device == voltage.device
    assert callback.record is record and record == []
    assert callback.numpy().shape == tuple(observed.shape)
    assert callback.i == steps


def test_native_raster_keeps_nonempty_stacking_and_unbound_errors():
    callback = Raster(node_check=[0])
    with pytest.raises((RuntimeError, TypeError)):
        callback.stack()
    records = [
        torch.tensor([[True], [False]]),
        torch.tensor([[False], [True]]),
    ]
    callback.record = records
    observed = callback.stack()
    torch.testing.assert_close(observed, torch.stack(records))
    assert callback.record is records


def test_native_apcount_uses_int32_and_n_owns_its_result():
    model = _model()
    callback = APCount(node_check=[0], threshold=0.0)
    callback.pre_loop_hook(model)
    assert callback.record.dtype == _COUNT_DTYPE == torch.int32
    model.v.fill_(1.0)
    callback.post_step_hook(model)
    expected = torch.ones((2, 1), dtype=torch.int32)
    torch.testing.assert_close(callback.record, expected)
    observed = callback.n
    assert observed.dtype == torch.int32 and observed is not callback.record
    observed.fill_(0)
    torch.testing.assert_close(callback.record, expected)
    identity = callback.record
    callback.pre_loop_hook(model)
    assert callback.record is identity


@pytest.mark.parametrize("dtype", (torch.float32, torch.int64))
def test_native_preloop_migrates_legacy_counts_without_resetting_other_carry(dtype):
    callback = APCount(node_check=[0, -1], threshold=0.0)
    callback.pre_loop_hook(_model())
    previous = torch.tensor([[2, 3], [4, 5]], dtype=dtype)
    callback.record = previous
    cache = callback.state_cache
    callback.i = 13
    callback.pre_loop_hook(_model())
    assert callback.record.dtype == torch.int32 and callback.record is not previous
    torch.testing.assert_close(callback.record, previous.to(torch.int32))
    assert callback.state_cache is cache and callback.i == 13
    assert previous.dtype == dtype


@pytest.mark.parametrize("dtype", (torch.float32, torch.int64))
def test_legacy_count_restoration_preserves_shape_device_values_and_empty_frames(dtype):
    source = torch.tensor([0, 2, 17, 1024], dtype=dtype).reshape(2, 2).transpose(0, 1)
    restored = _restore_count(source, label="legacy count")
    assert restored.shape == source.shape and restored.device == source.device
    assert restored.dtype == torch.int32 and restored is not source
    torch.testing.assert_close(restored, source.to(torch.int32), rtol=0, atol=0)
    empty = source[:0]
    converted = _restore_count(empty, label="empty count")
    assert converted.shape == empty.shape and converted.dtype == torch.int32
    assert _restore_count(restored, label="current count") is restored


def test_legacy_count_restoration_checks_representable_boundaries_before_casting():
    integer = torch.tensor([2**31 - 1], dtype=torch.int64)
    torch.testing.assert_close(
        _restore_count(integer, label="integer boundary"), integer.to(torch.int32)
    )
    floating = torch.tensor([2**31 - 128], dtype=torch.float32)
    torch.testing.assert_close(
        _restore_count(floating, label="floating boundary"), floating.to(torch.int32)
    )
    for unsafe in (
        torch.tensor([2**31], dtype=torch.int64),
        torch.tensor([2**31], dtype=torch.float32),
        # float32 cannot represent INT32_MAX and rounds this input upward.
        torch.tensor([2**31 - 1], dtype=torch.float32),
        torch.tensor([-1], dtype=torch.int64),
        torch.tensor([-1.0], dtype=torch.float32),
        torch.tensor([0.5], dtype=torch.float32),
        torch.tensor([float("nan")], dtype=torch.float32),
        torch.tensor([float("inf")], dtype=torch.float32),
        torch.tensor([-float("inf")], dtype=torch.float32),
    ):
        with pytest.raises(ValueError, match="boundary count.*non-negative integral"):
            _restore_count(unsafe, label="boundary count")


def test_legacy_count_restoration_rejects_unsupported_payloads_and_keeps_native_frame_on_failure():
    for unsupported in (
        None,
        [1, 2],
        torch.ones(2, dtype=torch.float64),
        torch.ones(2, dtype=torch.bool),
    ):
        with pytest.raises(TypeError, match="count payload"):
            _restore_count(unsupported, label="count payload")
    callback = APCount(node_check=[0])
    source = torch.tensor([[1.5], [2.0]], dtype=torch.float32)
    callback.record = source
    with pytest.raises(ValueError, match="APCount record"):
        callback.pre_loop_hook(_model())
    assert callback.record is source and callback.state_cache is None
