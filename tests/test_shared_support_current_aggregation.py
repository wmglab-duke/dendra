"""Regression tests for gather-once/scatter-once current aggregation."""

from __future__ import annotations

import pytest
import torch

import dendra as dn  # noqa: F401 - configure Dendra before compiling mechanisms
from dendra.models.mechanisms import Mechanism, PointProcess, State, VoltageProcess
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.mechanisms._support_registry import SupportEntry

FULL_SHAPE = (2, 4)
DTYPE = torch.float64


class _TensorAffine(Mechanism):
    Mechanism.RANGE(g=0.5, bias=-0.75)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i(self, v):
        return self.g * v + self.bias

    def i_with_conductance(self, v):
        return self.i(v), self.g.expand_as(v)


class _ScalarConstant(Mechanism):
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i(self, v):
        del v
        return 1.25

    def i_with_conductance(self, v):
        del v
        return 1.25, 0.0


class _ReducedConstant(Mechanism):
    Mechanism.RANGE(scale=0.5)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i(self, v):
        del v
        return self.scale.sum()

    def i_with_conductance(self, v):
        value = self.scale.sum()
        return value, torch.zeros_like(v)


class _PointAffine(PointProcess):
    PointProcess.RANGE(g=2.0, e=-10.0)
    PointProcess.NONSPECIFIC_CURRENT("i")
    PointProcess.AFFINE("i")

    def i(self, v):
        return self.g * (v - self.e)

    def i_with_conductance(self, v):
        return self.i(v), self.g.expand_as(v)


class _HalfLarge(Mechanism):
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i(self, v):
        return torch.tensor(2048.0, device=v.device, dtype=torch.float16)

    def i_with_conductance(self, v):
        return self.i(v), torch.tensor(0.0, device=v.device, dtype=torch.float16)


class _HalfOne(_HalfLarge):
    def i(self, v):
        return torch.tensor(1.0, device=v.device, dtype=torch.float16)


class _SavedAffine(Mechanism):
    Mechanism.RANGE(g=0.5, bias=-0.75)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.SAVE_CURRENT("i")
    Mechanism.AFFINE("i")

    def i(self, v):
        return self.g * v + self.bias

    def i_with_conductance(self, v):
        return self.i(v), self.g.expand_as(v)


class _SavedNonlinear(Mechanism):
    Mechanism.RANGE(scale=0.1)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.SAVE_CURRENT("i")
    Mechanism.NUMERICAL("i")

    def i(self, v):
        return self.scale * v.square()


class _RelaxState(State):
    State.STATE("x")
    State.RANGE(rate=0.1)
    State.DERIVATIVE("x' = -rate * x")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.full_like(v, -65.0)}


class _StateOnly(Mechanism):
    Mechanism.STATE_BUNDLE(_RelaxState)


class _OffsetVoltage(VoltageProcess):
    VoltageProcess.RANGE(offset=2.0)

    def update_v(self, v):
        return v + self.offset


class _CallLocalCurrent(torch.nn.Module):
    """Expose the handler's pure current frame through ``forward`` for transforms."""

    def __init__(self, handler):
        super().__init__()
        self.handler = handler

    def forward(self, voltage):
        current, conductance, _current_frame, _conductance_frame = (
            self.handler._evaluate_current_frame(voltage)
        )
        return current, conductance


def _make_mechanism(
    cls,
    name,
    *,
    key=None,
    dtype=DTYPE,
    is_composable=False,
    **parameters,
):
    celsius = torch.full(FULL_SHAPE, 34.0, dtype=dtype)
    diameters = torch.ones(FULL_SHAPE, dtype=dtype)
    if key is None:
        local_shape = FULL_SHAPE
    elif is_composable:
        local_shape = tuple(celsius[key].shape)
    else:
        local_shape = (torch.as_tensor(key).numel(),)
    return cls(
        name,
        celsius,
        diameters,
        local_shape,
        local_shape,
        key=key,
        is_composable=is_composable,
        **parameters,
    ).to(dtype=dtype)


def _make_handler(mechanisms, *, area=None, dtype=DTYPE):
    if area is None:
        area = torch.ones(FULL_SHAPE, dtype=dtype)
    celsius = torch.full(FULL_SHAPE, 34.0, dtype=dtype)
    handler = MechanismHandler(
        celsius,
        area,
        mechanisms,
        currents={"nonspecific": {name: ["i"] for name in mechanisms}},
    ).to(dtype=dtype)
    handler.make_maps()
    handler.init_i_g_bufs(torch.zeros(FULL_SHAPE, dtype=dtype))
    return handler


def _scatter_oracle(handler, voltage):
    expected_i = torch.zeros_like(voltage)
    expected_g = torch.zeros_like(voltage)
    for _current_index, mechanism, fn, scaler, _factorable in handler._map:
        local_i, local_g = scaler(*getattr(mechanism, fn)(mechanism.get(voltage)))
        mechanism.add_(expected_i, local_i)
        mechanism.add_(expected_g, local_g)
    return expected_i, expected_g


REGIONAL_CURRENT_CASES = (
    pytest.param(None, False, id="dense"),
    pytest.param((slice(None), slice(1, 3)), True, id="rectangular"),
    pytest.param(torch.tensor([1, 3, 5, 7]), False, id="shared_columns"),
    pytest.param(torch.tensor([0, 3, 5]), False, id="packed"),
    pytest.param(torch.tensor([1, 1, 6]), False, id="duplicate_packed"),
)


@pytest.mark.parametrize("key,is_composable", REGIONAL_CURRENT_CASES)
def test_call_local_current_frame_matches_imperative_for_every_support(
    key,
    is_composable,
):
    handler = _make_handler(
        {
            "probe": _make_mechanism(
                _TensorAffine,
                "probe",
                key=key,
                is_composable=is_composable,
                g=0.375,
                bias=-0.25,
            )
        }
    )
    voltage = torch.linspace(-2.0, 5.0, 8, dtype=DTYPE).reshape(FULL_SHAPE)
    expected = handler.i(voltage)
    scratch_before = tuple(
        buffer.clone()
        for buffers in (handler._buf_i, handler._buf_g)
        for buffer in buffers
    )

    actual = handler._evaluate_current_frame(voltage)[:2]

    torch.testing.assert_close(actual, expected)
    scratch_after = tuple(
        buffer for buffers in (handler._buf_i, handler._buf_g) for buffer in buffers
    )
    for before, after in zip(scratch_before, scratch_after):
        torch.testing.assert_close(after, before)


@pytest.mark.parametrize("key,is_composable", REGIONAL_CURRENT_CASES)
@pytest.mark.parametrize("lane_count", (3, 0), ids=("nonempty", "empty"))
def test_call_local_regional_current_supports_parameter_only_vmap(
    key,
    is_composable,
    lane_count,
):
    handler = _make_handler(
        {
            "probe": _make_mechanism(
                _TensorAffine,
                "probe",
                key=key,
                is_composable=is_composable,
                g=0.375,
                bias=-0.25,
            )
        }
    )
    module = _CallLocalCurrent(handler)
    voltage = torch.linspace(-2.0, 5.0, 8, dtype=DTYPE).reshape(FULL_SHAPE)
    conductance_name, base_conductance = next(
        (name, buffer)
        for name, buffer in module.named_buffers()
        if name.endswith("mechanisms.probe.g")
    )
    conductances = torch.randn(
        (lane_count,) + tuple(base_conductance.shape),
        dtype=DTYPE,
        requires_grad=True,
    )

    def evaluate(conductance):
        return torch.func.functional_call(
            module,
            {conductance_name: conductance},
            (voltage,),
            strict=False,
        )

    current, conductance = torch.vmap(evaluate)(conductances)
    assert current.shape == conductance.shape == (lane_count,) + FULL_SHAPE
    gradient = torch.autograd.grad(
        current.square().sum() + conductance.square().sum(),
        conductances,
    )[0]
    assert gradient.shape == conductances.shape
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("key,is_composable", REGIONAL_CURRENT_CASES)
@pytest.mark.parametrize("lane_count", (3, 0), ids=("nonempty", "empty"))
def test_call_local_regional_current_supports_voltage_vmap(
    key,
    is_composable,
    lane_count,
):
    handler = _make_handler(
        {
            "probe": _make_mechanism(
                _TensorAffine,
                "probe",
                key=key,
                is_composable=is_composable,
                g=0.375,
                bias=-0.25,
            )
        }
    )
    module = _CallLocalCurrent(handler)
    voltages = torch.randn(
        (lane_count,) + FULL_SHAPE,
        dtype=DTYPE,
        requires_grad=True,
    )

    current, conductance = torch.vmap(module)(voltages)
    assert current.shape == conductance.shape == voltages.shape
    gradient = torch.autograd.grad(
        current.square().sum() + conductance.square().sum(),
        voltages,
    )[0]
    assert gradient.shape == voltages.shape
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize(
    "key,is_composable",
    REGIONAL_CURRENT_CASES[1:4],
)
@pytest.mark.parametrize("lane_count", (3, 0), ids=("nonempty", "empty"))
def test_call_local_group_broadcasts_each_transformed_scalar_before_reduction(
    key,
    is_composable,
    lane_count,
):
    handler = _make_handler(
        {
            "reduced": _make_mechanism(
                _ReducedConstant,
                "reduced",
                key=key,
                is_composable=is_composable,
            ),
            "spatial": _make_mechanism(
                _TensorAffine,
                "spatial",
                key=key,
                is_composable=is_composable,
                g=0.375,
                bias=-0.25,
            ),
        }
    )
    assert [len(entries) for *_, entries in handler._map_grouped] == [2]
    module = _CallLocalCurrent(handler)
    voltage = torch.linspace(-2.0, 5.0, 8, dtype=DTYPE).reshape(FULL_SHAPE)
    scale_name, base_scale = next(
        (name, buffer)
        for name, buffer in module.named_buffers()
        if name.endswith("mechanisms.reduced.scale")
    )
    scales = torch.randn(
        (lane_count,) + tuple(base_scale.shape),
        dtype=DTYPE,
        requires_grad=True,
    )

    def evaluate(scale):
        return torch.func.functional_call(
            module,
            {scale_name: scale},
            (voltage,),
            strict=False,
        )

    vmapped = torch.vmap(evaluate)
    expected = vmapped(scales)
    torch.compiler.reset()
    compiled = torch.compile(vmapped, backend="eager", fullgraph=True)
    actual = compiled(scales)

    assert actual[0].shape == actual[1].shape == (lane_count,) + FULL_SHAPE
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(
        expected[0].square().sum() + expected[1].square().sum(),
        scales,
    )[0]
    assert gradient.shape == scales.shape


@pytest.mark.parametrize("key,is_composable", REGIONAL_CURRENT_CASES)
def test_compiled_vmapped_call_local_regional_current_is_fullgraph(
    key,
    is_composable,
):
    handler = _make_handler(
        {
            "probe": _make_mechanism(
                _TensorAffine,
                "probe",
                key=key,
                is_composable=is_composable,
                g=0.375,
                bias=-0.25,
            )
        }
    )
    module = _CallLocalCurrent(handler)
    voltage = torch.linspace(-2.0, 5.0, 8, dtype=DTYPE).reshape(FULL_SHAPE)
    conductance_name, base_conductance = next(
        (name, buffer)
        for name, buffer in module.named_buffers()
        if name.endswith("mechanisms.probe.g")
    )
    conductances = torch.randn(
        (2,) + tuple(base_conductance.shape),
        dtype=DTYPE,
    )

    def evaluate(conductance):
        return torch.func.functional_call(
            module,
            {conductance_name: conductance},
            (voltage,),
            strict=False,
        )

    vmapped = torch.vmap(evaluate)
    torch.compiler.reset()
    compiled = torch.compile(vmapped, backend="eager", fullgraph=True)
    expected = vmapped(conductances)
    actual = compiled(conductances)

    torch.testing.assert_close(actual, expected)


def test_direct_handler_itot_before_map_construction_keeps_safe_defaults():
    mechanism = _make_mechanism(
        _TensorAffine,
        "direct",
        key=torch.tensor([0, 3, 5, 7]),
        g=0.25,
        bias=-1.0,
    )
    handler = MechanismHandler(
        torch.full(FULL_SHAPE, 34.0, dtype=DTYPE),
        torch.ones(FULL_SHAPE, dtype=DTYPE),
        {"direct": mechanism},
        currents={"nonspecific": {"direct": ["i"]}},
    )
    voltage = torch.zeros(FULL_SHAPE, dtype=DTYPE)
    handler.init_i_g_bufs(voltage)

    torch.testing.assert_close(handler.itot(voltage), torch.zeros_like(voltage))


def test_exact_ordered_fancy_supports_share_but_reordered_support_does_not():
    shared = torch.tensor([0, 3, 5, 7])
    reordered = torch.tensor([7, 5, 3, 0])
    handler = _make_handler(
        {
            "first": _make_mechanism(
                _TensorAffine, "first", key=shared, g=0.25, bias=-1.0
            ),
            "second": _make_mechanism(
                _TensorAffine, "second", key=shared.clone(), g=-0.5, bias=2.0
            ),
            "reordered": _make_mechanism(
                _TensorAffine, "reordered", key=reordered, g=0.75, bias=0.5
            ),
        }
    )

    assert len(handler._current_support_representatives) == 2
    assert [len(entries) for *_, entries in handler._map_grouped] == [2, 1]
    # IDs are authored mechanism ordinals, so the distinct third mechanism
    # retains ID 2 even though the first two mechanisms share ID 0.
    assert handler._current_assigned_plan == ()

    voltage = torch.linspace(-3.0, 4.0, 8, dtype=DTYPE).reshape(FULL_SHAPE)
    expected_i, expected_g = _scatter_oracle(handler, voltage)
    actual_i, actual_g = handler.i(voltage)
    torch.testing.assert_close(actual_i, expected_i)
    torch.testing.assert_close(actual_g, expected_g)


def test_equal_supports_separated_in_authored_order_remain_distinct_runs():
    shared = torch.tensor([0, 3, 5, 7])
    intervening = torch.tensor([1, 2, 4, 6])
    handler = _make_handler(
        {
            "first": _make_mechanism(_TensorAffine, "first", key=shared, g=0.25),
            "middle": _make_mechanism(_TensorAffine, "middle", key=intervening, g=-0.5),
            "last": _make_mechanism(_TensorAffine, "last", key=shared.clone(), g=0.75),
        }
    )

    assert handler._current_assigned_plan == ()
    assert [len(entries) for *_, entries in handler._map_grouped] == [1, 1, 1]

    voltage = torch.linspace(-3.0, 4.0, 8, dtype=DTYPE).reshape(FULL_SHAPE)
    expected = _scatter_oracle(handler, voltage)
    actual = handler.i(voltage)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_handler_current_paths_use_support_entries_not_mechanism_mappers(monkeypatch):
    support = torch.tensor([0, 3, 5, 7])
    handler = _make_handler(
        {
            "first": _make_mechanism(
                _TensorAffine, "first", key=support, g=0.25, bias=-1.0
            ),
            "second": _make_mechanism(
                _TensorAffine,
                "second",
                key=support.clone(),
                g=-0.5,
                bias=2.0,
            ),
        }
    )
    voltage = torch.linspace(-3.0, 4.0, 8, dtype=DTYPE).reshape(FULL_SHAPE)
    previous = voltage - 5.0
    expected_i = handler.i(voltage)
    expected_iexp = handler.iexp(voltage)
    expected_idf = handler.idf(voltage, previous)
    expected_itot = handler.itot(voltage)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("handler execution re-entered a Mechanism mapper")

    for mechanism in handler.mechanisms.values():
        for name in ("get", "add_", "add", "put"):
            monkeypatch.setattr(mechanism, name, forbidden)

    actual_i = handler.i(voltage)
    actual_iexp = handler.iexp(voltage)
    actual_idf = handler.idf(voltage, previous)
    actual_itot = handler.itot(voltage)
    for actual, expected in ((actual_i, expected_i), (actual_idf, expected_idf)):
        for actual_field, expected_field in zip(actual, expected):
            torch.testing.assert_close(actual_field, expected_field)
    torch.testing.assert_close(actual_iexp, expected_iexp)
    torch.testing.assert_close(actual_itot, expected_itot)


def test_restricted_voltage_process_uses_canonical_support_access(monkeypatch):
    support = torch.tensor([0, 3, 5, 7])
    process = _make_mechanism(_OffsetVoltage, "offset", key=support, offset=2.5)
    handler = MechanismHandler(
        torch.full(FULL_SHAPE, 34.0, dtype=DTYPE),
        torch.ones(FULL_SHAPE, dtype=DTYPE),
        {"offset": process},
    )
    handler.make_maps()

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("handler execution re-entered a Mechanism mapper")

    monkeypatch.setattr(process, "get", forbidden)
    monkeypatch.setattr(process, "put", forbidden)
    voltage = torch.arange(8, dtype=DTYPE).reshape(FULL_SHAPE)
    expected = voltage.clone()
    expected.reshape(-1)[support] += 2.5

    torch.testing.assert_close(handler.update_v(voltage), expected)


def test_state_initialization_and_advance_gather_once_per_exact_support(monkeypatch):
    shared = torch.tensor([0, 3, 5, 7])
    other = torch.tensor([1, 2, 4, 6])
    mechanisms = {
        "first": _make_mechanism(_StateOnly, "first", key=shared),
        "second": _make_mechanism(_StateOnly, "second", key=shared.clone()),
        "other": _make_mechanism(_StateOnly, "other", key=other),
    }
    field = torch.linspace(-72.0, -58.0, 8, dtype=DTYPE).reshape(FULL_SHAPE)
    handler = MechanismHandler(
        torch.full(FULL_SHAPE, 34.0, dtype=DTYPE),
        torch.ones(FULL_SHAPE, dtype=DTYPE),
        mechanisms,
    )
    handler.make_maps()

    assert len(handler._state_support_representatives) == 2
    assert [support for _, support in handler._state_advance_plan] == [0, 0, 2]

    calls = [0, 0]
    support_positions = {
        entry.support_id: index
        for index, entry in enumerate(handler._state_support_entries)
    }
    original_gather = SupportEntry.gather

    def counted_gather(entry, tensor):
        calls[support_positions[entry.support_id]] += 1
        return original_gather(entry, tensor)

    monkeypatch.setattr(SupportEntry, "gather", counted_gather)

    handler.compute_initial_conditions(field)
    assert calls == [1, 1]

    calls[:] = [0, 0]
    handler.advance(field, torch.tensor(0.025, dtype=DTYPE), 34.0)
    assert calls == [1, 1]
    for mechanism in mechanisms.values():
        assert torch.isfinite(mechanism.x).all()


def test_duplicate_fancy_support_shares_gather_but_keeps_separate_scatters(
    monkeypatch,
):
    duplicate_support = torch.tensor([1, 1, 6])
    handler = _make_handler(
        {
            "first": _make_mechanism(
                _TensorAffine, "first", key=duplicate_support, g=0.25
            ),
            "second": _make_mechanism(
                _TensorAffine,
                "second",
                key=duplicate_support.clone(),
                g=-0.75,
                bias=1.5,
            ),
        }
    )

    assert len(handler._current_support_representatives) == 1
    assert len(handler._map_grouped) == 2
    assert all(len(entries) == 1 for *_, entries in handler._map_grouped)

    voltage = torch.linspace(-2.0, 5.0, 8, dtype=DTYPE).reshape(FULL_SHAPE)
    expected = _scatter_oracle(handler, voltage)
    scatter_calls = 0
    original_scatter_add = SupportEntry.scatter_add_

    def counted_scatter_add(entry, destination, local):
        nonlocal scatter_calls
        scatter_calls += 1
        return original_scatter_add(entry, destination, local)

    monkeypatch.setattr(SupportEntry, "scatter_add_", counted_scatter_add)
    actual = handler.i(voltage)
    # Two mechanisms remain separate runs, each scattering current and
    # conductance independently.
    assert scatter_calls == 4
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_call_local_duplicate_scatter_preserves_authored_float_association():
    support = torch.tensor([0, 0])
    handler = _make_handler(
        {
            "large_then_small": _make_mechanism(
                _TensorAffine,
                "large_then_small",
                key=support,
                dtype=torch.float32,
                g=torch.zeros(2, dtype=torch.float32),
                bias=torch.tensor([1.0e20, 1.0], dtype=torch.float32),
            ),
            "cancel_then_small": _make_mechanism(
                _TensorAffine,
                "cancel_then_small",
                key=support.clone(),
                dtype=torch.float32,
                g=torch.zeros(2, dtype=torch.float32),
                bias=torch.tensor([-1.0e20, 1.0], dtype=torch.float32),
            ),
        },
        dtype=torch.float32,
    )
    voltage = torch.zeros(FULL_SHAPE, dtype=torch.float32)

    expected = handler.i(voltage)
    actual = handler._evaluate_current_frame(voltage)[:2]

    assert expected[0][0, 0].item() == 1.0
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_identical_composable_slices_share_one_gather_and_scatter():
    support = (slice(None), slice(1, 3))
    handler = _make_handler(
        {
            "first": _make_mechanism(
                _TensorAffine,
                "first",
                key=support,
                is_composable=True,
                g=0.25,
            ),
            "second": _make_mechanism(
                _TensorAffine,
                "second",
                key=(slice(None), slice(1, 3)),
                is_composable=True,
                g=-0.75,
                bias=1.5,
            ),
        }
    )

    assert len(handler._current_support_representatives) == 1
    assert [len(entries) for *_, entries in handler._map_grouped] == [2]

    voltage = torch.linspace(-2.0, 5.0, 8, dtype=DTYPE).reshape(FULL_SHAPE)
    expected = _scatter_oracle(handler, voltage)
    actual = handler.i(voltage)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_scalar_and_tensor_reduction_preserves_eager_autograd_and_compile_eager():
    handler = _make_handler(
        {
            "scalar": _make_mechanism(_ScalarConstant, "scalar"),
            "tensor": _make_mechanism(_TensorAffine, "tensor", g=0.375, bias=-0.25),
        }
    )
    assert len(handler._current_support_representatives) == 1
    assert [len(entries) for *_, entries in handler._map_grouped] == [2]

    eager_voltage = (
        torch.linspace(-1.0, 2.5, 8, dtype=DTYPE).reshape(FULL_SHAPE).requires_grad_()
    )
    eager_i, eager_g = handler.i(eager_voltage)
    eager_gradient = torch.autograd.grad(
        eager_i.square().sum() + eager_g.square().sum(), eager_voltage
    )[0]

    # Integrator initialization detaches ephemeral current scratch before a
    # compiled run. Mirror that lifecycle here so Dynamo does not capture the
    # completed eager autograd graph through the handler's reusable buffers.
    handler.detach_i_g_bufs()

    compiled_i_g = torch.compile(handler.i, backend="eager", fullgraph=True)
    compiled_voltage = eager_voltage.detach().clone().requires_grad_()
    compiled_i, compiled_g = compiled_i_g(compiled_voltage)
    compiled_gradient = torch.autograd.grad(
        compiled_i.square().sum() + compiled_g.square().sum(), compiled_voltage
    )[0]

    torch.testing.assert_close(compiled_i, eager_i)
    torch.testing.assert_close(compiled_g, eager_g)
    torch.testing.assert_close(compiled_gradient, eager_gradient)


def test_point_scaling_happens_before_local_reduction_and_maps_rebuild_on_dtype():
    support = torch.tensor([1, 4, 6])
    area = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 8.0, 10.0, 16.0]], dtype=DTYPE)
    handler = _make_handler(
        {
            "density": _make_mechanism(
                _TensorAffine, "density", key=support, g=0.125, bias=0.75
            ),
            "point": _make_mechanism(
                _PointAffine, "point", key=support.clone(), g=3.0, e=-8.0
            ),
        },
        area=area,
    )

    assert len(handler._current_support_representatives) == 1
    assert [len(entries) for *_, entries in handler._map_grouped] == [2]
    old_point_scaler = handler._map[1][3]
    voltage = torch.linspace(-5.0, 2.0, 8, dtype=DTYPE).reshape(FULL_SHAPE)
    expected = _scatter_oracle(handler, voltage)
    actual = handler.i(voltage)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])

    handler.float()
    assert handler._map[1][3] is not old_point_scaler
    assert all(buffer.dtype == torch.float32 for buffer in handler._buf_i)
    assert all(buffer.dtype == torch.float32 for buffer in handler._buf_g)
    assert len(handler._current_support_representatives) == 1
    assert handler._current_support_representatives[0] is handler.density

    float_voltage = voltage.float()
    expected_float = _scatter_oracle(handler, float_voltage)
    actual_float = handler.i(float_voltage)
    torch.testing.assert_close(actual_float[0], expected_float[0])
    torch.testing.assert_close(actual_float[1], expected_float[1])


def test_group_plans_reject_state_dict_for_different_ordered_supports():
    shared = torch.tensor([0, 1])
    target = _make_handler(
        {
            "first": _make_mechanism(_TensorAffine, "first", key=shared, g=1.0),
            "second": _make_mechanism(
                _TensorAffine, "second", key=shared.clone(), g=2.0
            ),
        }
    )
    source = _make_handler(
        {
            "first": _make_mechanism(
                _TensorAffine, "first", key=torch.tensor([0, 1]), g=1.0
            ),
            "second": _make_mechanism(
                _TensorAffine, "second", key=torch.tensor([2, 3]), g=2.0
            ),
        }
    )

    assert len(target._current_support_representatives) == 1
    expected_key = target.second.key.clone()
    with pytest.raises(ValueError, match="selector keys encode mechanism placement"):
        target.load_state_dict(source.state_dict())
    assert len(target._current_support_representatives) == 1
    assert torch.equal(target.second.key, expected_key)

    voltage = torch.arange(8, dtype=DTYPE).reshape(FULL_SHAPE)
    expected = _scatter_oracle(target, voltage)
    actual = target.i(voltage)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_each_contribution_casts_before_local_reduction():
    """Match legacy destination-dtype accumulation under mixed precision."""

    handler = _make_handler(
        {
            "large": _make_mechanism(_HalfLarge, "large", dtype=torch.float32),
            "one": _make_mechanism(_HalfOne, "one", dtype=torch.float32),
        },
        dtype=torch.float32,
    )
    assert [len(entries) for *_, entries in handler._map_grouped] == [2]

    current = handler.iexp(torch.zeros(FULL_SHAPE, dtype=torch.float32))
    torch.testing.assert_close(current, torch.full(FULL_SHAPE, 2049.0))


def test_grouped_idf_preserves_factorability_and_save_mirrors():
    support = torch.tensor([0, 3, 5, 7])
    handler = _make_handler(
        {
            "affine": _make_mechanism(
                _SavedAffine,
                "affine",
                key=support,
                g=0.25,
                bias=-1.0,
            ),
            "nonlinear": _make_mechanism(
                _SavedNonlinear,
                "nonlinear",
                key=support.clone(),
                scale=0.05,
            ),
        }
    )
    assert [entry[-1] for entry in handler._map] == [True, False]
    assert [len(entries) for *_, entries in handler._map_grouped] == [2]

    voltage = torch.linspace(-3.0, 4.0, 8, dtype=DTYPE).reshape(FULL_SHAPE)
    previous = torch.linspace(-8.0, -1.0, 8, dtype=DTYPE).reshape(FULL_SHAPE)
    local_voltage = handler.affine.get(voltage)
    local_half = handler.affine.get(0.5 * previous)
    expected_affine = handler.affine.i(local_half)
    expected_nonlinear = handler.nonlinear.i(local_voltage)
    expected_i = torch.zeros_like(voltage)
    expected_g = torch.zeros_like(voltage)
    handler.affine.add_(expected_i, expected_affine + expected_nonlinear)
    handler.affine.add_(expected_g, handler.affine.g)

    actual_i, actual_g = handler.idf(voltage, previous)
    torch.testing.assert_close(actual_i, expected_i)
    torch.testing.assert_close(actual_g, expected_g)
    torch.testing.assert_close(handler.affine.i_, expected_affine)
    torch.testing.assert_close(handler.nonlinear.i_, expected_nonlinear)
