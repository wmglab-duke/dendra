"""Shared membrane coefficient contracts, observers, supports and transforms."""

from copy import deepcopy
from types import MethodType

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism, PointProcess
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.mechanisms.compilers.ast import affine_coefficients_in_v
from dendra.models.mod import hh


class Affine(Mechanism):
    Mechanism.RANGE(g=0.25, e=-40.0)
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.g * (v - self.e)


class Constant(Mechanism):
    Mechanism.GLOBAL(offset=0.125)
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.offset


class Saved(Affine):
    Mechanism.SAVE_CURRENT("i")


class Explicit(Affine):
    Mechanism.EXPLICIT("i")


class Analytic(Affine):
    def i_with_conductance(self, v):
        return self.i(v), self.g


class Nonlinear(Mechanism):
    Mechanism.RANGE(g=0.25)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.NUMERICAL("i")

    def i(self, v):
        return self.g * v.square()


class PointAffine(PointProcess):
    PointProcess.RANGE(g=0.25, e=-40.0)
    PointProcess.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.g * (v - self.e)


class Assigned(Mechanism):
    Mechanism.RANGE(g=0.25, e=-40.0)
    Mechanism.ASSIGNED("effective_g")
    Mechanism.NONSPECIFIC_CURRENT("i")

    def assigned_values(self, v, values):
        return {"effective_g": self.g * (1.0 + v / 100.0)}

    def i(self, v):
        return self.effective_g * (v - self.e)


class MixedAssigned(Assigned):
    def assigned_values(self, v, values):
        return {"effective_g": (self.g * (1.0 + v / 100.0)).double()}


class OverriddenHH(hh):
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        g, j = super().i_affine_terms(v)
        return g + self.gl, j + self.gl * self.el


class CancelledDenominator(Affine):
    def i(self, v):
        return self.g * (v - self.e) / self.g


class CancelledNegativePower(Affine):
    def i(self, v):
        return self.g * (v - self.e) * self.g**-1


class FixedDenominator(Affine):
    def i(self, v):
        return self.g * (v - self.e) / 2.0


class FrameModule(torch.nn.Module):
    def __init__(self, handler):
        super().__init__()
        self.handler = handler

    def forward(self, v):
        g, j, current, _, _ = self.handler._evaluate_membrane_frame(v, call_local=True)
        assert current is None
        return g, j


def mechanism(cls, name="m", key=None, dtype=torch.float64):
    shape = (2, 4)
    local = (
        shape if key is None else ((2, 2) if isinstance(key, tuple) else (key.numel(),))
    )
    return cls(
        name,
        torch.full(shape, 23.0, dtype=dtype),
        torch.ones(shape, dtype=dtype),
        local,
        local,
        key=key,
        is_composable=isinstance(key, tuple),
    ).to(dtype=dtype)


def handler(cls=Affine, *, key=None, dtype=torch.float64, others=None):
    mechs = {"m": mechanism(cls, key=key, dtype=dtype)}
    mechs.update(others or {})
    currents = {
        "nonspecific": {
            name: list(m._currents["nonspecific"]) for name, m in mechs.items()
        }
    }
    voltage = torch.linspace(-65.0, -45.0, 8, dtype=dtype).reshape(2, 4)
    obj = MechanismHandler(
        torch.full_like(voltage, 23.0),
        torch.full_like(voltage, 2e-6),
        mechs,
        currents=currents,
    ).to(dtype=dtype)
    obj.make_maps()
    obj.init_i_g_bufs(voltage)
    return obj, voltage


@pytest.fixture(autouse=True)
def eager_cpu():
    with dn.ctx(JIT=0, DEVICE="cpu", REQUIRE_GRAD=1):
        yield


@pytest.mark.parametrize("cls", (Affine, Constant, PointAffine))
@pytest.mark.parametrize(
    "key",
    (
        None,
        (slice(None), slice(1, 3)),
        torch.tensor([0, 3, 5]),
        torch.tensor([], dtype=torch.long),
    ),
)
@pytest.mark.parametrize("call_local", (False, True))
def test_direct_matches_authored_density_and_zero_conductance(cls, key, call_local):
    obj, v = handler(cls, key=key)
    current, conductance = obj.i(v)
    g, j, actual, ion_i, ion_g = obj._evaluate_membrane_frame(v, call_local=call_local)
    assert actual is None and ion_i == ion_g == ()
    torch.testing.assert_close(g, conductance, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(j, conductance * v - current, rtol=1e-13, atol=1e-13)
    if cls is Constant:
        assert affine_coefficients_in_v(Constant, method="i")[0] == "0"
        assert torch.isfinite(j).all()


@pytest.mark.parametrize("cls", (Saved, Explicit, Analytic, Nonlinear))
@pytest.mark.parametrize("call_local", (False, True))
def test_unsupported_or_observed_currents_retain_actual_frame(cls, call_local):
    obj, v = handler(cls)
    expected_i, expected_g = obj.i(v)
    g, j, current, ion_i, ion_g = obj._evaluate_membrane_frame(v, call_local=call_local)
    assert current is not None and ion_i == ion_g == ()
    torch.testing.assert_close(current, expected_i, rtol=0, atol=0)
    torch.testing.assert_close(g, expected_g, rtol=0, atol=0)
    torch.testing.assert_close(j, g * v - current, rtol=0, atol=0)
    if cls is Saved:
        torch.testing.assert_close(obj.m.i_, obj.m.i(v))
    if cls is Explicit:
        assert not torch.count_nonzero(g)


def test_duplicate_support_falls_back_without_reassociation():
    obj, v = handler(key=torch.tensor([1, 1, 6]))
    assert obj._evaluate_membrane_frame(v)[2] is not None


@pytest.mark.parametrize("call_local", (False, True))
def test_require_current_keeps_legacy_scratch_and_captured_frame(call_local):
    obj, v = handler()
    obj.i(v)
    old_i, old_g = obj._buf_i[0], obj._buf_g[0]
    old_values = old_i.clone(), old_g.clone()
    frame = obj._evaluate_membrane_frame(
        v + 1.0, call_local=call_local, require_current=True
    )
    assert frame[2] is not None
    torch.testing.assert_close(old_i, old_values[0], rtol=0, atol=0)
    torch.testing.assert_close(old_g, old_values[1], rtol=0, atol=0)
    if call_local:
        assert obj._buf_i[0] is old_i and obj._buf_g[0] is old_g
    else:
        assert obj._buf_i[0] is not old_i and obj._buf_g[0] is not old_g


@pytest.mark.parametrize(
    "override",
    ("current_bound", "current_plain", "pair", "coefficient", "analytic_pair"),
)
def test_validation_detects_late_inconsistent_method_overrides(override):
    obj, v = handler()
    assert obj._evaluate_membrane_frame(v)[2] is None
    if override == "current_bound":
        obj.m.i = MethodType(lambda self, value: self.g * (value - self.e) + 1.0, obj.m)
    elif override == "current_plain":
        obj.m.i = lambda value: obj.m.g * (value - obj.m.e) + 1.0
    elif override == "pair":
        obj.m.i_with_g = lambda value: (torch.ones_like(value), torch.ones_like(value))
    elif override == "coefficient":
        obj.m._affine_i = lambda value: (torch.ones_like(value), torch.ones_like(value))
    else:
        obj.m.i_with_conductance = lambda value: (obj.m.i(value), obj.m.g)
        # Installing an analytic method after construction does not change the
        # already bound ordinary pair. Identical behavior remains valid.
        with dn.ctx(MEMBRANE_ASSEMBLY="validate"):
            assert obj._evaluate_membrane_frame(v)[2] is None
        return
    with dn.ctx(MEMBRANE_ASSEMBLY="validate"):
        with pytest.raises(RuntimeError, match="Invalid AFFINE"):
            obj._evaluate_membrane_frame(v)


@pytest.mark.parametrize("cls", (Assigned, MixedAssigned))
@pytest.mark.parametrize("call_local", (False, True))
def test_assigned_evaluated_once_before_dtype_fallback(cls, call_local):
    obj, v = handler(cls, dtype=torch.float32)
    if cls is MixedAssigned:
        # Mixed registered workspace dtype is legal; its authored output must
        # still match that workspace's own runtime schema.
        obj.m._buffers["effective_g"] = obj.m.effective_g.double()
    calls = []
    original = obj.m._evaluate_assigned
    obj.m._evaluate_assigned = lambda value: (
        calls.append("assigned"),
        original(value),
    )[1]
    frame = obj._evaluate_membrane_frame(v, call_local=call_local)
    assert calls == ["assigned"]
    assert (frame[2] is None) is (cls is Assigned)
    expected = obj.m.i(v).to(v.dtype)
    torch.testing.assert_close(frame[0] * v - frame[1], expected, rtol=3e-6, atol=3e-6)


def test_assigned_and_symbolic_coefficients_retain_second_derivatives():
    obj, v = handler(Assigned)
    v = v.requires_grad_()
    g, j, actual, _, _ = obj._evaluate_membrane_frame(v, call_local=True)
    assert actual is None
    loss = (g * v - j).square().sum()
    first = torch.autograd.grad(loss, v, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), v)[0]
    obj.m._evaluate_assigned(v)
    expected = obj.m.i(v).square().sum()
    first_expected = torch.autograd.grad(expected, v, create_graph=True)[0]
    second_expected = torch.autograd.grad(first_expected.sum(), v)[0]
    torch.testing.assert_close(first, first_expected)
    torch.testing.assert_close(second, second_expected)


@pytest.mark.parametrize("lanes", (0, 3))
def test_parameter_only_vmap_expands_before_coefficient_arithmetic(lanes):
    obj, v = handler(Assigned, key=torch.tensor([0, 3, 5]))
    module = FrameModule(obj)
    parameters = dict(module.named_parameters())
    name = next(key for key in parameters if key.endswith("g_param"))
    target = parameters[name]
    values = target.detach().expand(lanes, *target.shape).clone()

    def evaluate(value):
        return torch.func.functional_call(module, {**parameters, name: value}, (v,))

    g, j = torch.vmap(evaluate)(values)
    assert g.shape == j.shape == (lanes, *v.shape)


def test_clone_load_and_dtype_move_rebuild_live_plan_ownership():
    with torch.no_grad():
        obj, v = handler()
    copied = deepcopy(obj)
    assert copied._evaluate_membrane_frame(v)[2] is None
    for _, _, entry, terms in copied._membrane_map_grouped:
        assert entry.representative is copied.m
        assert all(term[0] is copied.m for term in terms)
    copied.load_state_dict(obj.state_dict())
    copied.float()
    assert copied._evaluate_membrane_frame(v.float())[2] is None
    copied.m.g = torch.ones_like(copied.m.g, dtype=torch.float64)
    assert copied._evaluate_membrane_frame(v.float())[2] is not None


def test_custom_current_frame_signature_does_not_receive_new_keyword():
    obj, v = handler()
    calls = []
    original = obj._evaluate_current_frame

    def old_signature(value, *, scratch=None):
        calls.append(True)
        return original(value, scratch=scratch)

    obj._evaluate_current_frame = old_signature
    assert obj._evaluate_membrane_frame(v, call_local=True)[2] is not None
    assert calls == [True]


def test_overlapping_supports_keep_authored_coefficient_addition_order():
    other = mechanism(Affine, name="other", key=torch.tensor([1, 3, 5]))
    obj, v = handler(key=torch.tensor([0, 3, 5]), others={"other": other})
    current, conductance = obj.i(v)
    g, j, actual, _, _ = obj._evaluate_membrane_frame(v, call_local=True)
    assert actual is None
    torch.testing.assert_close(g, conductance, rtol=0, atol=0)
    torch.testing.assert_close(j, conductance * v - current)


def test_single_hh_current_uses_public_coefficient_hook():
    obj, v = handler(hh)
    obj.m.m = torch.full_like(v, 0.2)
    obj.m.h = torch.full_like(v, 0.6)
    obj.m.n = torch.full_like(v, 0.3)
    assert len(obj._membrane_map_grouped[0][3]) == 1
    assert obj._membrane_map_grouped[0][3][0][1] == "i_affine_terms"
    assert obj._evaluate_membrane_frame(v)[2] is None
    obj.m.i_affine_terms = lambda value: (value, value)
    with dn.ctx(MEMBRANE_ASSEMBLY="validate"):
        with pytest.raises(RuntimeError, match="Invalid AFFINE"):
            obj._evaluate_membrane_frame(v)


def test_affine_hook_override_is_the_combined_current_authority():
    obj, v = handler(OverriddenHH)
    obj.m.m = torch.full_like(v, 0.2)
    obj.m.h = torch.full_like(v, 0.6)
    obj.m.n = torch.full_like(v, 0.3)
    assert obj._membrane_map_grouped[0][3][0][1] == "i_affine_terms"
    g, j, current, _, _ = obj._evaluate_membrane_frame(v)
    assert current is None
    i, expected_g = obj.i(v)
    torch.testing.assert_close(g, expected_g)
    torch.testing.assert_close(j, expected_g * v - i)


@pytest.mark.parametrize("call_local", (False, True))
def test_ion_frames_fallback_remain_unpublished_until_solver_commit(call_local):
    obj, v = handler()
    ion = torch.nn.Module()
    ion.register_buffer("ina", torch.zeros_like(v))
    obj.ions["na"] = ion
    obj.currents = {"ina": {"m": ["i"]}}
    obj.make_maps()
    obj.init_i_g_bufs(v)
    published_before = ion.ina
    first = obj._evaluate_membrane_frame(v, call_local=call_local)
    saved = first[3][0].clone()
    second = obj._evaluate_membrane_frame(v + 1.0, call_local=call_local)
    assert first[2] is not None and second[2] is not None
    assert ion.ina is published_before
    torch.testing.assert_close(first[3][0], saved, rtol=0, atol=0)
    torch.testing.assert_close(first[3][0], first[2], rtol=0, atol=0)
    torch.testing.assert_close(first[4][0], first[0], rtol=0, atol=0)
    obj._publish_ion_current_frame(first[3])
    assert ion.ina is first[3][0]


@pytest.mark.parametrize(
    "collection", ("materials", "voltage_processes", "material_processes")
)
def test_material_and_voltage_collections_conservatively_disable_direct(collection):
    obj, v = handler()
    getattr(obj, collection)["observer"] = torch.nn.Module()
    assert obj._evaluate_membrane_frame(v)[2] is not None


def test_late_current_declaration_change_disables_cached_coefficient_plan():
    obj, v = handler()
    assert obj._evaluate_membrane_frame(v)[2] is None
    obj.m._currents = {"nonspecific": ["i", "new_current"]}
    assert obj._evaluate_membrane_frame(v)[2] is not None


def test_current_category_reordering_and_replaced_owners_disable_stale_plan():
    obj, v = handler()
    obj.currents = {"new_category": {"m": ["i"]}}
    assert obj._evaluate_membrane_frame(v)[2] is not None
    obj.make_maps()
    assert obj._evaluate_membrane_frame(v)[2] is None
    obj.mechanisms["m"] = mechanism(Affine)
    assert not obj._can_evaluate_direct_membrane_frame()


def test_same_count_renamed_owner_disables_stale_plan_without_key_error():
    obj, _ = handler()
    assert obj._can_evaluate_direct_membrane_frame()
    original = obj.mechanisms.pop("m")
    obj.mechanisms["renamed"] = original
    assert not obj._can_evaluate_direct_membrane_frame()


@pytest.mark.parametrize("cls", (CancelledDenominator, CancelledNegativePower))
def test_cancelled_parameter_poles_keep_authored_invalid_current(cls):
    obj, v = handler(cls)
    obj.m.g = torch.zeros_like(obj.m.g)
    g, j, current, _, _ = obj._evaluate_membrane_frame(v)
    assert current is not None
    assert torch.isnan(current).all() and torch.isnan(j).all()
    assert torch.isfinite(g).all()


def test_fixed_numeric_denominator_keeps_direct_capability():
    obj, v = handler(FixedDenominator)
    assert obj._evaluate_membrane_frame(v)[2] is None


def test_direct_domain_proof_shares_legacy_source_analysis(monkeypatch):
    from dendra.models.mechanisms.compilers import ast as compiler

    calls = []
    original = compiler._runtime_function_source

    def read_once(function, method):
        calls.append((function, method))
        return original(function, method)

    compiler.affine_coefficients_in_v.cache_clear()
    monkeypatch.setattr(compiler, "_runtime_function_source", read_once)
    assert compiler.linear_conductance_in_v(CancelledDenominator) == "1"
    with pytest.raises(ValueError, match="denominators"):
        compiler.affine_coefficients_in_v(CancelledDenominator)
    assert compiler.linear_conductance_in_v(CancelledDenominator) == "1"
    assert len(calls) == 1
