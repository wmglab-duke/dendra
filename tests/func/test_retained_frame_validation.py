"""Retained validation keeps caller boundaries and ordered metadata reads."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_initialization_execution import _assert_snapshot, _snapshot
from tests.func.test_network_initialization_ordering import DTYPE, _network
from tests.func.test_noise_workspaces import DT as NOISE_DT
from tests.func.test_noise_workspaces import _population as _noise_population
from tests.func.test_retained_materialization_inventory import _ORDINARY_PATHS

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(
            JIT=0,
            REQUIRE_GRAD=True,
            IMEM=False,
            DEVICE="cpu",
            DTYPE=DTYPE,
            CHECK_PURITY=False,
        ),
    ):
        yield


class _Observed(Mapping):
    def __init__(self, entries, events, label, *, retained_only=False):
        self.entries, self.events, self.label = entries, events, label
        self.retained_only = retained_only

    def __iter__(self):
        self.events.append(f"{self.label}:iter")
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, name):
        if (
            not self.retained_only
            or name.startswith("materializations.")
            or name == "dt"
        ):
            self.events.append(f"{self.label}:get:{name}")
        return self.entries[name]


class _Dimension:
    def __init__(self, value, events, label):
        self.value, self.events, self.label = value, events, label

    def __int__(self):
        self.events.append(self.label)
        return self.value

    def __eq__(self, other):
        raise AssertionError("retained dimensions must be normalized with int()")


class _MetadataTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls, value, events, label, *, failure=None, shape=None, dtype=None, device=None
    ):
        result = torch.Tensor._make_subclass(cls, value, value.requires_grad)
        result._events, result._label, result._failure = events, label, failure
        result._shape = tuple(value.shape) if shape is None else shape
        result._dtype = value.dtype if dtype is None else dtype
        result._device = value.device if device is None else device
        return result

    def _read(self, name):
        self._events.append(f"{self._label}:{name}")
        if self._failure is not None and self._failure[0] == name:
            raise self._failure[1]

    @property
    def shape(self):
        self._read("shape")
        return tuple(
            _Dimension(value, self._events, f"{self._label}:int:{index}")
            for index, value in enumerate(self._shape)
        )

    @property
    def dtype(self):
        self._read("dtype")
        return self._dtype

    @property
    def device(self):
        self._read("device")
        return self._device


def _metadata_reads(label, shape, *, through="device"):
    result = [f"{label}:shape", *(f"{label}:int:{i}" for i in range(len(shape)))]
    if through in ("dtype", "device"):
        result.append(f"{label}:dtype")
    if through == "device":
        result.append(f"{label}:device")
    return result


def _constant_prefix():
    # The preceding NOISE validator scans the full packet, including non-NOISE
    # entries. It must complete before retained metadata validation starts.
    return [
        "constants:iter",
        "owner:device",
        "owner:device",
        "constants:iter",
        "constants:get:dt",
        *(
            f"constants:get:materializations.{kind}.{path}"
            for kind in ("values", "active", "baseline")
            for path in _ORDINARY_PATHS
        ),
    ]


def _case():
    functional, tensors = dn.func.make_functional_network(_network())
    plan = functional.populations["left"]
    parameters = tensors.parameters["populations"]["left"]
    constants = tensors.constants["populations"]["left"]
    prepared = plan.prepare(parameters, constants)
    return SimpleNamespace(
        plan=plan, parameters=parameters, constants=constants, prepared=prepared
    )


def _observe_device(monkeypatch, case, events):
    original = type(case.plan).device.fget

    def device(owner):
        if owner is case.plan:
            events.append("owner:device")
        return original(owner)

    monkeypatch.setattr(type(case.plan), "device", property(device))


def _prepared_boundary(monkeypatch, case, retained, events):
    values = dict(case.prepared.values)
    values["materializations"] = retained

    class Values(_Observed):
        def __iter__(self):
            return iter(self.entries)

        def __getitem__(self, name):
            if name == "materializations":
                self.events.append("values:materializations")
            return self.entries[name]

    frame = Values(values, events, "values")

    # Admit a supplied frame at the existing freshness/schema boundary. A
    # separate test below exercises actual opaque-plan freshness precedence.
    def admitted(prepared, parameters):
        assert prepared is case.prepared and parameters is case.parameters
        events.append("freshness")
        return frame

    monkeypatch.setattr(case.plan, "_validate_prepared_freshness", admitted)
    return frame


def test_constant_fields_keep_lazy_owner_device_reads_and_int_shape(monkeypatch):
    case, events = _case(), []
    snapshots = _snapshot((case.parameters, case.constants, case.prepared.values))
    entries = dict(case.constants)
    probes = {}
    for kind in ("values", "baseline", "active"):
        key = f"materializations.{kind}.cm"
        probes[key] = _MetadataTensor(entries[key], events, kind)
        entries[key] = probes[key]
    observed = _Observed(entries, events, "constants", retained_only=True)
    _observe_device(monkeypatch, case, events)
    assert case.plan._validate_constants(observed) is None
    expected = _constant_prefix()
    for path in _ORDINARY_PATHS:
        for kind in ("values", "baseline", "active"):
            key = f"materializations.{kind}.{path}"
            expected.append(f"constants:get:{key}")
            if path == "cm":
                expected.extend(
                    _metadata_reads(kind, (1, 1) if kind != "active" else ())
                )
        expected.append("owner:device")
    expected.extend(("constants:get:dt", "owner:device"))
    assert events == expected
    assert all(entries[key] is value for key, value in probes.items())
    _assert_snapshot((case.parameters, case.constants, case.prepared.values), snapshots)


@pytest.mark.parametrize("property_name", ("shape", "dtype", "device"))
def test_constant_metadata_exceptions_stop_before_later_fields(
    monkeypatch, property_name
):
    case, events = _case(), []
    snapshots = _snapshot((case.parameters, case.constants))
    error = RuntimeError(f"retained {property_name} failed")
    key = "materializations.values.celsius"
    entries = dict(case.constants)
    entries[key] = _MetadataTensor(
        entries[key], events, "value", failure=(property_name, error)
    )
    observed = _Observed(entries, events, "constants", retained_only=True)
    _observe_device(monkeypatch, case, events)
    with pytest.raises(RuntimeError) as caught:
        case.plan._validate_constants(observed)
    assert caught.value is error
    expected = [*_constant_prefix(), f"constants:get:{key}"]
    expected.extend(_metadata_reads("value", (), through=property_name))
    assert events == expected
    _assert_snapshot((case.parameters, case.constants), snapshots)


def test_prepared_fields_keep_key_then_row_order_and_int_shape(monkeypatch):
    case, events = _case(), []
    snapshots = _snapshot((case.parameters, case.constants, case.prepared.values))
    entries = dict(case.prepared.values["materializations"])
    probe = entries["cm"] = _MetadataTensor(entries["cm"], events, "cm")
    retained = _Observed(entries, events, "retained")
    frame = _prepared_boundary(monkeypatch, case, retained, events)
    result = case.plan._validate_prepared(case.prepared, case.parameters)
    expected = ["freshness", "values:materializations", "retained:iter"]
    for path in _ORDINARY_PATHS:
        expected.append(f"retained:get:{path}")
        if path == "cm":
            expected.extend(_metadata_reads("cm", (1, 1)))
    assert events == expected
    assert result is frame and entries["cm"] is probe
    _assert_snapshot((case.parameters, case.constants, case.prepared.values), snapshots)


@pytest.mark.parametrize(
    "fault",
    ("not_mapping", "missing", "extra", "not_tensor", "shape", "dtype", "device"),
)
def test_prepared_schema_failures_keep_complete_access_prefix(monkeypatch, fault):
    case, events = _case(), []
    snapshots = _snapshot((case.parameters, case.constants, case.prepared.values))
    entries = dict(case.prepared.values["materializations"])
    # A later bad row must never replace an earlier structural/schema error.
    entries["rhoa_scale"] = object()
    if fault == "missing":
        del entries["celsius"]
    elif fault == "extra":
        entries["unexpected"] = object()
    elif fault == "not_tensor":
        entries["celsius"] = object()
    elif fault in ("shape", "dtype", "device"):
        options = (
            {"shape": (1,)}
            if fault == "shape"
            else {fault: torch.float32 if fault == "dtype" else torch.device("meta")}
        )
        entries["celsius"] = _MetadataTensor(
            entries["celsius"], events, "value", **options
        )
    retained = (
        None if fault == "not_mapping" else _Observed(entries, events, "retained")
    )
    _prepared_boundary(monkeypatch, case, retained, events)
    with pytest.raises(dn.func.FunctionalizationError) as caught:
        case.plan._validate_prepared(case.prepared, case.parameters)
    expected = ["freshness", "values:materializations"]
    if fault != "not_mapping":
        expected.append("retained:iter")
    if fault in ("not_mapping", "missing", "extra"):
        message = "prepared materialization schema is invalid"
    else:
        message = "prepared retained materialization 'celsius' has invalid schema"
        expected.append("retained:get:celsius")
        if fault != "not_tensor":
            expected.extend(
                _metadata_reads(
                    "value", (1,) if fault == "shape" else (), through=fault
                )
            )
    assert str(caught.value) == message
    assert events == expected
    _assert_snapshot((case.parameters, case.constants, case.prepared.values), snapshots)


def test_actual_freshness_rejection_precedes_retained_schema():
    case = _case()
    case.prepared.values["materializations"]["celsius"] = torch.ones(2, dtype=DTYPE)
    snapshots = _snapshot((case.parameters, case.constants, case.prepared.values))
    with pytest.raises(dn.func.FunctionalizationError) as caught:
        case.plan._validate_prepared(case.prepared, case.parameters)
    assert (
        str(caught.value)
        == "prepared workspaces changed after prepare(); call prepare() again"
    )
    _assert_snapshot((case.parameters, case.constants, case.prepared.values), snapshots)


def test_earlier_prepared_geometry_failure_precedes_retained_lookup(monkeypatch):
    case, events = _case(), []
    frame = _prepared_boundary(monkeypatch, case, None, events)
    frame.entries["geometry"] = None
    with pytest.raises(dn.func.FunctionalizationError) as caught:
        case.plan._validate_prepared(case.prepared, case.parameters)
    assert str(caught.value) == "prepared geometry tensor schema is invalid"
    assert events == ["freshness"]


def test_empty_retained_layout_preserves_prepared_identity_without_section_lookup(
    monkeypatch,
):
    plan, tensors = dn.func.make_functional(_noise_population(), dt=NOISE_DT)
    prepared = plan.prepare(tensors.parameters, tensors.constants)
    assert "materializations" not in prepared.values
    snapshots = _snapshot((tensors.parameters, tensors.constants, prepared.values))

    class Values(dict):
        def __getitem__(self, name):
            assert name != "materializations", "empty layout read retained section"
            return super().__getitem__(name)

    values = Values(prepared.values)
    monkeypatch.setattr(plan, "_validate_prepared_freshness", lambda *_: values)
    assert plan._validate_prepared(prepared, tensors.parameters) is values
    _assert_snapshot(
        (tensors.parameters, tensors.constants, prepared.values), snapshots
    )
