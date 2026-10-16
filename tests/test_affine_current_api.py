"""Public affine coefficient authoring, inheritance, and diagnostic contracts."""

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism


class CoefficientsOnly(Mechanism):
    Mechanism.RANGE(g=0.25, e=-40.0)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        g = torch.broadcast_to(self.g, v.shape)
        e = torch.broadcast_to(self.e, v.shape)
        return g, g * e


class SavedCoefficients(CoefficientsOnly):
    Mechanism.SAVE_CURRENT("i")


class ExplicitCoefficients(CoefficientsOnly):
    Mechanism.EXPLICIT("i")


class CoefficientsWithoutAssertion(Mechanism):
    Mechanism.RANGE(g=0.25, e=-40.0)
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i_affine_terms(self, v):
        return self.g, self.g * self.e


class InheritedCoefficients(CoefficientsOnly):
    pass


class OverrideCoefficientsWithoutAssertion(CoefficientsOnly):
    def i_affine_terms(self, v):
        return self.g, self.g * self.e + v.square()


class OverrideCurrent(CoefficientsOnly):
    def i(self, v):
        return 2.0 * self.g * (v - self.e)


class OverrideCurrentAsserted(CoefficientsOnly):
    Mechanism.AFFINE("i")

    def i(self, v):
        return 2.0 * self.g * (v - self.e)


class OpaqueAnalytic(Mechanism):
    Mechanism.RANGE(g=0.25, e=-40.0)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i(self, v):
        return torch.add(v, -self.e) * self.g

    def i_with_conductance(self, v):
        return self.i(v), self.g


class OpaqueNumerical(Mechanism):
    Mechanism.RANGE(g=0.25, e=-40.0)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")
    Mechanism.NUMERICAL("i")

    def i(self, v):
        return torch.add(v, -self.e) * self.g


class ConflictingAnalytic(CoefficientsOnly):
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        return self.g, self.g * self.e

    def i_with_conductance(self, v):
        return self.i(v) + 1.0, 2.0 * self.g


class ConflictingAuthoredCurrent(CoefficientsOnly):
    Mechanism.AFFINE("i")

    def i(self, v):
        return self.g * (v - self.e) + 1.0

    def i_affine_terms(self, v):
        return self.g, self.g * self.e


class InvalidVoltageDependentG(CoefficientsOnly):
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        return self.g * v, self.g * self.e


class InvalidVoltageDependentJ(CoefficientsOnly):
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        return self.g, self.g * self.e + v


class ScalarCoefficients(CoefficientsOnly):
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        return 0.0, 3.0


class InvalidCoupledCoefficients(CoefficientsOnly):
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        # Uniform shifts leave J unchanged and therefore cannot detect this
        # coupled current. A nonuniform frozen-voltage probe is necessary.
        return 2.0, v - v.mean()


class FrozenAssigned(CoefficientsOnly):
    Mechanism.AFFINE("i")
    Mechanism.ASSIGNED("effective_g")

    def assigned_values(self, v, values):
        raise AssertionError("affine diagnostics must not evaluate ASSIGNED")

    def i_affine_terms(self, v):
        return self.effective_g, self.effective_g * self.e


def mechanism(cls, dtype=torch.float64):
    shape = (2, 3)
    return cls(
        cls.__name__,
        torch.full(shape, 23.0, dtype=dtype),
        torch.ones(shape, dtype=dtype),
        shape,
        shape,
        dtype=dtype,
    )


def coefficients(obj, voltage):
    return getattr(obj, obj._affine_evaluators["i"])(voltage)


@pytest.fixture(autouse=True)
def eager_cpu():
    with dn.ctx(JIT=0, DEVICE="cpu", REQUIRE_GRAD=1):
        yield


@pytest.mark.parametrize(
    "cls", (CoefficientsOnly, InheritedCoefficients, SavedCoefficients)
)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_hook_only_supplies_all_current_interfaces(cls, dtype):
    obj = mechanism(cls, dtype)
    v = torch.linspace(-70.0, 20.0, 6, dtype=dtype).reshape(2, 3)
    g, j = coefficients(obj, v)
    current, pair_g = obj.i_with_g(v)
    torch.testing.assert_close(obj.i(v), obj.g * (v - obj.e))
    torch.testing.assert_close(current, obj.i(v))
    torch.testing.assert_close(pair_g, g)
    torch.testing.assert_close(j, obj.g * obj.e)
    assert obj._affine_evaluators == {"i": "i_affine_terms"}
    assert obj._affine_coefficient_fields == {"i": None}
    assert obj._current_factorable == {"i": True}
    if cls is SavedCoefficients:
        torch.testing.assert_close(obj.i_, current)


def test_explicit_overrides_hook_affinity_but_keeps_current():
    obj = mechanism(ExplicitCoefficients)
    v = torch.full((2, 3), -65.0, dtype=torch.float64)
    current, g = obj.i_with_g(v)
    torch.testing.assert_close(current, obj.i(v))
    assert g == 0.0
    assert obj._affine_evaluators == {}
    assert obj._current_factorable == {"i": False}


@pytest.mark.parametrize(
    "cls", (CoefficientsWithoutAssertion, OverrideCoefficientsWithoutAssertion)
)
def test_unasserted_hook_only_definition_does_not_authorize_fast_path(cls):
    obj = mechanism(cls)
    assert obj._affine_evaluators == {}
    assert obj._current_factorable == {"i": False}


@pytest.mark.parametrize("cls", (OverrideCurrent, OverrideCurrentAsserted))
def test_current_override_does_not_inherit_stale_coefficients(cls):
    obj = mechanism(cls)
    v = torch.full((2, 3), -65.0, dtype=torch.float64)
    assert obj._affine_evaluators == {"i": "_affine_i"}
    if cls is OverrideCurrent:
        assert "i" not in obj._affine
    else:
        assert "i" in obj._affine
    g, j = coefficients(obj, v)
    torch.testing.assert_close(g * v - j, 2.0 * obj.g * (v - obj.e))


@pytest.mark.parametrize("cls", (OpaqueAnalytic, OpaqueNumerical))
def test_asserted_existing_pair_needs_no_coefficient_hook(cls):
    obj = mechanism(cls)
    v = torch.linspace(-70.0, 20.0, 6, dtype=torch.float64).reshape(2, 3)
    g, j = coefficients(obj, v)
    current, pair_g = obj.i_with_g(v)
    torch.testing.assert_close(g * v - j, current)
    torch.testing.assert_close(g, pair_g)
    assert obj._affine_coefficient_fields == {"i": None}
    obj._validate_affine_current("i", v, g, j)


def test_existing_analytic_pair_retains_ordinary_path_precedence():
    obj = mechanism(ConflictingAnalytic)
    v = torch.full((2, 3), -65.0, dtype=torch.float64)
    current, g = obj.i_with_g(v)
    torch.testing.assert_close(current, obj.i(v) + 1.0)
    torch.testing.assert_close(g, 2.0 * obj.g)
    assert obj._current_conductance_mode == {"i": "analytic"}
    with pytest.raises(RuntimeError, match="Invalid AFFINE current"):
        obj._validate_affine_current("i", v, *coefficients(obj, v))


def test_authored_current_retains_ordinary_path_precedence():
    obj = mechanism(ConflictingAuthoredCurrent)
    v = torch.full((2, 3), -65.0, dtype=torch.float64)
    torch.testing.assert_close(obj.i_with_g(v)[0], obj.i(v))
    with pytest.raises(RuntimeError, match="disagree with authored current"):
        obj._validate_affine_current("i", v, *coefficients(obj, v))


@pytest.mark.parametrize("cls", (InvalidVoltageDependentG, InvalidVoltageDependentJ))
def test_validation_rejects_voltage_dependent_coefficients(cls):
    obj = mechanism(cls)
    v = torch.full((2, 3), -65.0, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="depends on voltage"):
        obj._validate_affine_current("i", v, *coefficients(obj, v))


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_coefficient_gradients_match_ordinary_current(dtype):
    obj = mechanism(CoefficientsOnly, dtype)
    obj.g.requires_grad_()
    obj.e.requires_grad_()
    v = torch.linspace(-70.0, 20.0, 6, dtype=dtype).reshape(2, 3).requires_grad_()
    g, j = coefficients(obj, v)
    actual = torch.autograd.grad((g * v - j).square().sum(), (v, obj.g, obj.e))
    expected = torch.autograd.grad(obj.i(v).square().sum(), (v, obj.g, obj.e))
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right)


def test_scalar_zero_conductance_hook_and_validation():
    obj = mechanism(ScalarCoefficients)
    v = torch.full((2, 3), -65.0, dtype=torch.float64)
    torch.testing.assert_close(obj.i(v), torch.full_like(v, -3.0))
    obj._validate_affine_current("i", v, *coefficients(obj, v))


def test_validation_detects_coupling_missed_by_uniform_voltage_shifts():
    obj = mechanism(InvalidCoupledCoefficients)
    v = torch.full((2, 3), -65.0, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="J depends on voltage"):
        obj._validate_affine_current("i", v, *coefficients(obj, v))


def test_validation_never_reevaluates_assigned_values():
    obj = mechanism(FrozenAssigned)
    obj.effective_g = 3.0 * obj.g
    v = torch.full((2, 3), -65.0, dtype=torch.float64)
    obj._validate_affine_current("i", v, *coefficients(obj, v))


@pytest.mark.parametrize("invalid", ("dtype", "device", "shape", "value_type"))
def test_validation_rejects_invalid_coefficient_metadata(invalid):
    obj = mechanism(CoefficientsOnly)
    v = torch.full((2, 3), -65.0, dtype=torch.float64)
    g, j = coefficients(obj, v)
    if invalid == "dtype":
        g = g.float()
    elif invalid == "device":
        g = torch.empty_like(g, device="meta")
    elif invalid == "shape":
        g = torch.ones((2, 2), dtype=v.dtype)
    else:
        g = [1.0]
    with pytest.raises(RuntimeError, match="Invalid AFFINE current.*coefficient"):
        obj._validate_affine_current("i", v, g, j)


@pytest.mark.parametrize("cls", (CoefficientsOnly, InvalidVoltageDependentJ))
def test_validation_compiles_with_runtime_assertions(cls):
    obj = mechanism(cls)

    def evaluate(v):
        g, j = coefficients(obj, v)
        obj._validate_affine_current("i", v, g, j)
        return g * v - j

    compiled = torch.compile(evaluate, fullgraph=True, backend="eager")
    v = torch.full((2, 3), -65.0, dtype=torch.float64)
    if cls is CoefficientsOnly:
        torch.testing.assert_close(compiled(v), obj.i(v))
    else:
        with pytest.raises(RuntimeError, match="Invalid AFFINE current"):
            compiled(v)


def test_validation_vmap_is_explicitly_unsupported_not_silently_skipped():
    obj = mechanism(CoefficientsOnly)

    def evaluate(v):
        obj._validate_affine_current("i", v, *coefficients(obj, v))
        return obj.i(v)

    with pytest.raises(RuntimeError, match="validate.*does not support vmap"):
        torch.vmap(evaluate)(torch.full((2, 2, 3), -65.0, dtype=torch.float64))


@pytest.mark.parametrize("batch", (0, 2))
def test_hook_and_generated_pair_support_empty_and_mapped_voltage(batch):
    obj = mechanism(CoefficientsOnly)
    v = torch.full((batch, 2, 3), -65.0, dtype=torch.float64)
    actual_i, actual_g = torch.vmap(obj.i_with_g)(v)
    expected_i = torch.vmap(obj.i)(v)
    torch.testing.assert_close(actual_i, expected_i)
    assert actual_g.shape == v.shape


def test_renamed_hook_only_mechanism_preserves_contract_and_adapters():
    renamed = CoefficientsOnly.rename("renamed_coefficients_only")
    obj = mechanism(renamed)
    v = torch.full((2, 3), -65.0, dtype=torch.float64)
    assert obj._affine_evaluators == {"i": "i_affine_terms"}
    torch.testing.assert_close(obj.i(v), obj.g * (v - obj.e))
