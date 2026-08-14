"""Regression tests for gather-once/scatter-once current aggregation."""

from __future__ import annotations

import torch

import dendra as dn  # noqa: F401 - configure Dendra before compiling mechanisms
from dendra.models.mechanisms import Mechanism, PointProcess
from dendra.models.mechanisms._handler import MechanismHandler

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
    Mechanism.SAVE("i")
    Mechanism.AFFINE("i")

    def i(self, v):
        return self.g * v + self.bias

    def i_with_conductance(self, v):
        return self.i(v), self.g.expand_as(v)


class _SavedNonlinear(Mechanism):
    Mechanism.RANGE(scale=0.1)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.SAVE("i")
    Mechanism.NUMERICAL("i")

    def i(self, v):
        return self.scale * v.square()


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
    assert [support for _, support in handler._current_breakpoint_plan] == [0, 0, 1]

    voltage = torch.linspace(-3.0, 4.0, 8, dtype=DTYPE).reshape(FULL_SHAPE)
    expected_i, expected_g = _scatter_oracle(handler, voltage)
    actual_i, actual_g = handler.i(voltage)
    torch.testing.assert_close(actual_i, expected_i)
    torch.testing.assert_close(actual_g, expected_g)


def test_duplicate_fancy_support_shares_gather_but_keeps_separate_scatters():
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
    actual = handler.i(voltage)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


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


def test_group_plans_rebuild_when_loading_different_ordered_supports():
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
    target.load_state_dict(source.state_dict())
    assert len(target._current_support_representatives) == 2

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
