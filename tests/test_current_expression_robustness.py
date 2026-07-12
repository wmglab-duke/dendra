"""Cross-platform robustness contracts for mechanism current compilation."""

from __future__ import annotations

import inspect

import pytest
import torch

from dendra.models.mechanisms import (
    Mechanism,
    UnsafeAutomaticNumericalFallbackError,
)
from dendra.models.mechanisms import _symbolic as symbolic
from dendra.models.mechanisms.compilers import ast as ast_compiler
from dendra.models.mechanisms.compilers import source as source_compiler
from dendra.models.mechanisms.compilers.ast import (
    factorize_linear_in_v,
    linear_conductance_in_v,
)
from dendra.models.mechanisms.compilers.source import SourceUnavailableError
from dendra.models.mod import (
    alphasynapse,
    alphasynapse_d,
    exp2syn,
    expsyn,
    graded_syn,
    hh,
    pas,
)

FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


class _DirectAffine(Mechanism):
    Mechanism.RANGE(g=0.375, e=-48.0)
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.g * (v - self.e)


class _LocalTemporariesAffine(Mechanism):
    Mechanism.RANGE(gain=0.625, gate=0.75, scale=1.25, e=-37.0)
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        conductance = self.gain * self.gate**2 / self.scale
        driving_force = v - self.e
        current = conductance * driving_force
        return current


class _MultilineStateDifferenceAffine(Mechanism):
    Mechanism.RANGE(A=0.125, B=0.5, e=-37.0)
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        conductance = self.A - self.B
        i = conductance * (v - self.e)
        return i


class _InheritedMultilineStateDifferenceAffine(_MultilineStateDifferenceAffine):
    pass


class _ExpandedAffine(Mechanism):
    Mechanism.RANGE(g=0.3125, e=-61.0)
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.g * v - self.g * self.e


class _SavedMultiCurrent(Mechanism):
    Mechanism.RANGE(gl=0.125, el=-54.0, gna=0.75, ena=42.0, gate=0.5)
    Mechanism.SAVE("ina")
    Mechanism.NONSPECIFIC_CURRENT("il", "ina")

    def il(self, v):
        return self.gl * (v - self.el)

    def ina(self, v):
        conductance = self.gna * self.gate**3
        return conductance * (v - self.ena)


class _AnalyticNonlinearPair(Mechanism):
    Mechanism.RANGE(a=0.03125, b=0.375, c=-1.25)
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.a * v**2 + self.b * v + self.c

    def i_with_conductance(self, v):
        return self.i(v), 2 * self.a * v + self.b


class _CancellingAffineTerms(Mechanism):
    Mechanism.RANGE(g1=1.0, e1=-70.0, g2=-1.0, e2=-50.0)
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.g1 * (v - self.e1) + self.g2 * (v - self.e2)


class _IonicAffine(Mechanism):
    Mechanism.RANGE(g=0.375, e=48.0)
    Mechanism.USEION("na", write=["ina"])

    def ina(self, v):
        return self.g * (v - self.e)


class _MultiClassCellTarget(Mechanism):
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return 3.0 * v


def _replacement_current(self, v):
    return 4.0 * (v - self.e)


def _mechanism(cls, dtype=torch.float64, device="cpu"):
    shape = (1, 3)
    mechanism = cls(
        cls.__name__,
        torch.full(shape, 34.0, dtype=dtype, device=device),
        torch.ones(shape, dtype=dtype, device=device),
        shape,
        shape,
        dtype=dtype,
        device=device,
    )
    return mechanism.to(device=device, dtype=dtype)


def _voltage(dtype):
    return torch.tensor([[-73.25, -41.5, 12.75]], dtype=dtype)


def _assert_close(actual, expected):
    actual, expected = torch.broadcast_tensors(actual, expected)
    rtol = {
        torch.float16: 3.0e-3,
        torch.bfloat16: 2.0e-2,
        torch.float32: 2.0e-6,
        torch.float64: 2.0e-12,
    }[actual.dtype]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=0.0)


def _autograd_conductance(mechanism, current_name, voltage):
    probe = voltage.detach().clone().requires_grad_(True)
    current = getattr(mechanism, current_name)(probe)
    return torch.autograd.grad(current.sum(), probe)[0]


def _assert_symbolic_path(mechanism, current_name):
    assert mechanism._current_conductance_mode[current_name] == "symbolic"
    assert mechanism._current_conductance_fallback_reason[current_name] is None
    assert mechanism._current_factorable[current_name] is True


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize(
    "mechanism_cls,expected_name",
    [
        (_DirectAffine, "direct"),
        (_LocalTemporariesAffine, "temporaries"),
        (_ExpandedAffine, "expanded"),
    ],
)
def test_symbolic_affine_expression_forms_match_independent_oracles(
    mechanism_cls, expected_name, dtype
):
    mechanism = _mechanism(mechanism_cls, dtype)
    voltage = _voltage(dtype)

    current, conductance = mechanism.i_with_g(voltage)
    authored_current = mechanism.i(voltage)
    autograd_conductance = _autograd_conductance(mechanism, "i", voltage)

    if expected_name in {"direct", "expanded"}:
        analytic_conductance = mechanism.g.expand_as(voltage)
    else:
        analytic_conductance = (
            mechanism.gain * mechanism.gate**2 / mechanism.scale
        ).expand_as(voltage)

    _assert_close(current, authored_current)
    _assert_close(conductance, analytic_conductance)
    _assert_close(conductance, autograd_conductance)
    _assert_symbolic_path(mechanism, "i")


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_shipped_hh_multi_current_conductances_match_autograd(dtype):
    mechanism = _mechanism(hh, dtype)
    voltage = _voltage(dtype)
    with torch.no_grad():
        mechanism.m.fill_(0.25)
        mechanism.h.fill_(0.625)
        mechanism.n.fill_(0.375)

    expected = {
        "il": mechanism.gl,
        "ina": mechanism.gnabar * mechanism.m**3 * mechanism.h,
        "ik": mechanism.gkbar * mechanism.n**4,
    }
    for current_name, analytic_conductance in expected.items():
        current, conductance = getattr(mechanism, f"{current_name}_with_g")(voltage)
        _assert_close(current, getattr(mechanism, current_name)(voltage))
        _assert_close(conductance, analytic_conductance.expand_as(voltage))
        _assert_close(
            conductance,
            _autograd_conductance(mechanism, current_name, voltage),
        )
        _assert_symbolic_path(mechanism, current_name)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize(
    "mechanism_cls,expected_mode",
    [
        (pas, "symbolic"),
        (expsyn, "symbolic"),
        (alphasynapse, "symbolic"),
        (alphasynapse_d, "symbolic"),
        (graded_syn, "symbolic"),
        (exp2syn, "analytic"),
    ],
)
def test_shipped_single_current_inventory_matches_autograd(
    mechanism_cls, expected_mode, dtype
):
    mechanism = _mechanism(mechanism_cls, dtype)
    voltage = _voltage(dtype)

    with torch.no_grad():
        if mechanism_cls in (pas, expsyn):
            mechanism.g.fill_(0.25)
        elif mechanism_cls in (alphasynapse, alphasynapse_d):
            mechanism.gmax.fill_(0.25)
            mechanism.onset.zero_()
            mechanism.tau.fill_(1.0)
            mechanism.t = torch.ones_like(voltage)
        elif mechanism_cls is graded_syn:
            mechanism.g_scale.fill_(0.5)
            mechanism.g_pre.fill_(0.5)
        else:
            mechanism.A.fill_(0.125)
            mechanism.B.fill_(0.5)

    current, conductance = mechanism.i_with_g(voltage)

    _assert_close(current, mechanism.i(voltage))
    _assert_close(conductance, _autograd_conductance(mechanism, "i", voltage))
    assert mechanism._current_conductance_mode == {"i": expected_mode}
    assert mechanism._current_conductance_fallback_reason == {"i": None}


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_ionic_current_uses_the_same_exact_symbolic_contract(dtype):
    mechanism = _mechanism(_IonicAffine, dtype)
    voltage = _voltage(dtype)

    current, conductance = mechanism.ina_with_g(voltage)

    _assert_close(current, mechanism.ina(voltage))
    _assert_close(conductance, mechanism.g.expand_as(voltage))
    _assert_close(conductance, _autograd_conductance(mechanism, "ina", voltage))
    assert mechanism._current_conductance_mode == {"ina": "symbolic"}
    assert mechanism._current_conductance_fallback_reason == {"ina": None}


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_saved_current_assignment_and_named_multi_current_paths(dtype):
    mechanism = _mechanism(_SavedMultiCurrent, dtype)
    voltage = _voltage(dtype)

    leak_current, leak_conductance = mechanism.il_with_g(voltage)
    sodium_current, sodium_conductance = mechanism.ina_with_g(voltage)

    _assert_close(leak_current, mechanism.il(voltage))
    _assert_close(leak_conductance, mechanism.gl.expand_as(voltage))
    _assert_close(sodium_current, mechanism.ina(voltage))
    _assert_close(
        sodium_conductance,
        (mechanism.gna * mechanism.gate**3).expand_as(voltage),
    )
    _assert_close(mechanism.ina_, sodium_current)
    _assert_symbolic_path(mechanism, "il")
    _assert_symbolic_path(mechanism, "ina")


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_headless_renamed_synapse_recovers_symbolic_source(monkeypatch, dtype):
    alias = expsyn.rename("headless_current_expression_matrix")
    real_getsource = inspect.getsource

    def getsource_without_dynamic_class(obj):
        if obj is alias:
            raise OSError("dynamic class has no class-body source")
        return real_getsource(obj)

    monkeypatch.setattr(source_compiler, "IPYTHON_AVAILABLE", False)
    monkeypatch.setattr(
        source_compiler.inspect, "getsource", getsource_without_dynamic_class
    )
    factorize_linear_in_v.cache_clear()

    mechanism = _mechanism(alias, dtype)
    voltage = _voltage(dtype)
    with torch.no_grad():
        mechanism.g.copy_(torch.tensor([[0.125, 0.5, 0.75]], dtype=dtype))

    current, conductance = mechanism.i_with_g(voltage)

    _assert_close(current, mechanism.i(voltage))
    _assert_close(conductance, mechanism.g)
    _assert_close(conductance, _autograd_conductance(mechanism, "i", voltage))
    _assert_symbolic_path(mechanism, "i")


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_headless_renamed_multiline_current_stays_exactly_symbolic(monkeypatch, dtype):
    alias = _MultilineStateDifferenceAffine.rename("headless_renamed_multiline_current")
    assert alias._dendra_symbolic_source_class is _MultilineStateDifferenceAffine
    real_getsource = inspect.getsource
    real_safe_source = ast_compiler.safe_source
    source_owners = []

    def getsource_without_dynamic_class(obj):
        if obj is alias:
            raise OSError("dynamic class has no class-body source")
        return real_getsource(obj)

    def safe_source_from_stable_owner(obj):
        source_owners.append(obj)
        if obj is alias:
            raise AssertionError("symbolic analysis inspected the dynamic alias")
        if obj is _MultilineStateDifferenceAffine.i:
            raise SourceUnavailableError("force stable class-source fallback")
        return real_safe_source(obj)

    monkeypatch.setattr(source_compiler, "IPYTHON_AVAILABLE", False)
    monkeypatch.setattr(
        source_compiler.inspect, "getsource", getsource_without_dynamic_class
    )
    monkeypatch.setattr(ast_compiler, "safe_source", safe_source_from_stable_owner)
    factorize_linear_in_v.cache_clear()

    mechanism = _mechanism(alias, dtype)
    voltage = _voltage(dtype)
    current, conductance = mechanism.i_with_g(voltage)
    expected_conductance = mechanism.A - mechanism.B

    _assert_close(current, mechanism.i(voltage))
    _assert_close(conductance, expected_conductance)
    _assert_close(conductance, _autograd_conductance(mechanism, "i", voltage))
    _assert_symbolic_path(mechanism, "i")
    assert source_owners == [
        _MultilineStateDifferenceAffine.i,
        _MultilineStateDifferenceAffine,
    ]


def test_nested_and_inherited_renames_preserve_multiline_symbolic_analysis():
    nested_alias = _MultilineStateDifferenceAffine.rename(
        "multiline_first_alias"
    ).rename("multiline_nested_alias")
    inherited_alias = _InheritedMultilineStateDifferenceAffine.rename(
        "multiline_inherited_alias"
    )
    cases = (
        (nested_alias, _MultilineStateDifferenceAffine),
        (inherited_alias, _InheritedMultilineStateDifferenceAffine),
    )
    voltage = _voltage(torch.float64)

    for alias, expected_provenance in cases:
        assert alias._dendra_symbolic_source_class is expected_provenance
        factorize_linear_in_v.cache_clear()
        mechanism = _mechanism(alias)
        current, conductance = mechanism.i_with_g(voltage)

        _assert_close(current, mechanism.i(voltage))
        _assert_close(conductance, mechanism.A - mechanism.B)
        _assert_symbolic_path(mechanism, "i")


def test_runtime_method_selection_uses_class_identity_in_multiclass_source(
    monkeypatch,
):
    cell_source = """
    class _MultiClassCellTarget:
        def i(self, v):
            return 3.0 * v

    class _MultiClassCellWrong:
        def i(self, v):
            return 2.0 * v
    """
    real_safe_source = ast_compiler.safe_source

    def multiclass_source(obj):
        if obj is _MultiClassCellTarget.i:
            return cell_source
        return real_safe_source(obj)

    monkeypatch.setattr(ast_compiler, "safe_source", multiclass_source)
    factorize_linear_in_v.cache_clear()

    assert float(linear_conductance_in_v(_MultiClassCellTarget)) == 3.0


def test_runtime_current_replacement_invalidates_symbolic_analysis_cache():
    alias = _DirectAffine.rename("runtime_replaced_current")
    factorize_linear_in_v.cache_clear()
    original = _mechanism(alias)
    voltage = _voltage(torch.float64)
    _, original_conductance = original.i_with_g(voltage)
    _assert_close(original_conductance, original.g)

    alias.i = _replacement_current
    replaced = _mechanism(alias)
    current, conductance = replaced.i_with_g(voltage)

    _assert_close(current, replaced.i(voltage))
    assert conductance == pytest.approx(4.0)
    _assert_close(
        torch.full_like(voltage, conductance),
        _autograd_conductance(replaced, "i", voltage),
    )
    _assert_symbolic_path(replaced, "i")


def test_source_less_runtime_replacement_requires_explicit_safe_path(monkeypatch):
    alias = _DirectAffine.rename("source_less_runtime_replaced_current")
    alias.i = _replacement_current
    real_safe_source = ast_compiler.safe_source

    def source_without_replacement(obj):
        if obj is _replacement_current:
            raise SourceUnavailableError("replacement source unavailable")
        return real_safe_source(obj)

    monkeypatch.setattr(ast_compiler, "safe_source", source_without_replacement)
    factorize_linear_in_v.cache_clear()

    with pytest.raises(UnsafeAutomaticNumericalFallbackError) as exc_info:
        _mechanism(alias)

    message = str(exc_info.value)
    assert "source_less_runtime_replaced_current" in message
    assert "current 'i'" in message
    assert "SourceUnavailableError: replacement source unavailable" in message
    assert "deterministic, side-effect-free, pointwise" in message
    assert "i_with_conductance(self, v)" in message
    assert "Mechanism.NUMERICAL('i')" in message
    assert isinstance(exc_info.value.__cause__, SourceUnavailableError)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_analytic_pair_is_source_independent_and_low_precision_safe(monkeypatch, dtype):
    alias = _AnalyticNonlinearPair.rename("source_independent_analytic_pair")

    def unavailable_factorization(*args, **kwargs):
        raise SourceUnavailableError("analytic current source unavailable")

    monkeypatch.setattr(symbolic, "linear_conductance_in_v", unavailable_factorization)
    mechanism = _mechanism(alias, dtype)
    voltage = _voltage(dtype)

    current, conductance = mechanism.i_with_g(voltage)

    assert mechanism.i_with_g.__func__ is alias.i_with_conductance
    # The exact pair remains source-independent for ordinary implicit solvers,
    # but unavailable source cannot prove the stronger affine contract needed
    # by Dufort--Frankel. It therefore takes that solver's explicit path.
    assert mechanism._current_factorable == {"i": False}
    _assert_close(current, mechanism.i(voltage))
    _assert_close(conductance, 2 * mechanism.a * voltage + mechanism.b)
    _assert_close(conductance, _autograd_conductance(mechanism, "i", voltage))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "mechanism_cls,path",
    [
        (_LocalTemporariesAffine, "symbolic"),
        (_AnalyticNonlinearPair, "analytic"),
    ],
)
def test_exact_current_paths_match_between_cpu_and_cuda(mechanism_cls, path):
    cpu_mechanism = _mechanism(mechanism_cls, torch.float32, "cpu")
    cuda_mechanism = _mechanism(mechanism_cls, torch.float32, "cuda")
    cpu_voltage = _voltage(torch.float32)
    cuda_voltage = cpu_voltage.cuda()

    cpu_current, cpu_conductance = cpu_mechanism.i_with_g(cpu_voltage)
    cuda_current, cuda_conductance = cuda_mechanism.i_with_g(cuda_voltage)

    torch.testing.assert_close(cuda_current.cpu(), cpu_current, rtol=2.0e-6, atol=0.0)
    torch.testing.assert_close(
        cuda_conductance.cpu(), cpu_conductance, rtol=2.0e-6, atol=0.0
    )
    _assert_close(
        cuda_conductance,
        _autograd_conductance(cuda_mechanism, "i", cuda_voltage),
    )
    if path == "symbolic":
        _assert_symbolic_path(cuda_mechanism, "i")
    else:
        assert cuda_mechanism.i_with_g.__func__ is mechanism_cls.i_with_conductance


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_missing_source_never_silently_falls_back_to_numerical_derivative(
    monkeypatch, dtype
):
    alias = _DirectAffine.rename("missing_source_numerical_fallback")
    real_safe_source = ast_compiler.safe_source

    def unavailable_source(obj):
        if obj in (alias, _DirectAffine, _DirectAffine.i):
            raise SourceUnavailableError("source unavailable")
        return real_safe_source(obj)

    monkeypatch.setattr(ast_compiler, "safe_source", unavailable_source)
    factorize_linear_in_v.cache_clear()

    with pytest.raises(
        UnsafeAutomaticNumericalFallbackError,
        match=(
            "missing_source_numerical_fallback.*"
            "SourceUnavailableError: source unavailable.*analytic.*"
            "Mechanism.NUMERICAL\\('i'\\)"
        ),
    ):
        _mechanism(alias, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_missing_source_is_rejected_before_dtype_can_mask_the_root_cause(
    monkeypatch, dtype
):
    alias = _DirectAffine.rename("missing_source_low_precision")

    def unavailable_source(obj):
        if obj in (alias, _DirectAffine, _DirectAffine.i):
            raise SourceUnavailableError("source unavailable")
        return source_compiler.safe_source(obj)

    monkeypatch.setattr(ast_compiler, "safe_source", unavailable_source)
    factorize_linear_in_v.cache_clear()

    with pytest.raises(
        UnsafeAutomaticNumericalFallbackError,
        match="SourceUnavailableError: source unavailable",
    ):
        _mechanism(alias, dtype)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cancelling_affine_terms_keep_finite_current_and_zero_conductance(dtype):
    mechanism = _mechanism(_CancellingAffineTerms, dtype)
    voltage = _voltage(dtype)

    current, conductance = mechanism.i_with_g(voltage)

    _assert_close(current, mechanism.i(voltage))
    _assert_close(current, torch.full_like(voltage, 20.0))
    _assert_close(conductance, torch.zeros_like(voltage))
    _assert_close(conductance, _autograd_conductance(mechanism, "i", voltage))
    assert torch.isfinite(current).all()
    _assert_symbolic_path(mechanism, "i")
