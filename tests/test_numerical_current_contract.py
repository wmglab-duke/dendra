"""Correctness contracts for numerically differentiated mechanism currents."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import (
    Mechanism,
    NumericalCurrentContractError,
    UnsafeAutomaticNumericalFallbackError,
)
from dendra.models.mechanisms._handler import MechanismHandler


class _PointwiseNonlinearNumerical(Mechanism):
    Mechanism.RANGE(a=0.03125, b=-0.25, offset=3.0)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.NUMERICAL("i")

    def i(self, v):
        return self.a * v**3 + self.b * v**2 + self.offset


class _CoupledCurrent(Mechanism):
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return v + v.mean()


class _SavedPointwiseNonlinearNumerical(_PointwiseNonlinearNumerical):
    Mechanism.SAVE("i")


class _StatefulCurrent(Mechanism):
    Mechanism.BUFFER("calls")
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        self.calls.add_(1.0)
        return v + self.calls


class _NondeterministicCurrent(Mechanism):
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return v + torch.rand_like(v)


class _DeclaredCoupledCurrent(Mechanism):
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.NUMERICAL("i")

    def i(self, v):
        return v + v.mean()


class _DeclaredStatefulCurrent(Mechanism):
    Mechanism.BUFFER("calls")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.NUMERICAL("i")

    def i(self, v):
        self.calls.add_(1.0)
        return v + self.calls


class _DeclaredNondeterministicCurrent(Mechanism):
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.NUMERICAL("i")

    def i(self, v):
        return v + torch.rand_like(v)


class _DeclaredBadShapeCurrent(Mechanism):
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.NUMERICAL("i")

    def i(self, v):
        return v[..., :1]


class _DeclaredBadDtypeCurrent(Mechanism):
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.NUMERICAL("i")

    def i(self, v):
        return v.to(torch.float32)


class _DeclaredMutatingRandomCurrent(Mechanism):
    Mechanism.BUFFER("calls")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.NUMERICAL("i")

    def i(self, v):
        self.calls.add_(1.0)
        return v + torch.rand_like(v)


class _DeclaredVoltageMutatingCurrent(Mechanism):
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.NUMERICAL("i")

    def i(self, v):
        self.observed_voltage = v
        v.transpose_(0, 1).add_(17.0)
        raise ValueError("failure after mutating voltage")


class _DeclaredZeroingVoltageCurrent(Mechanism):
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.NUMERICAL("i")

    def i(self, v):
        return v.zero_()


class _DeclaredResizingStateCurrent(Mechanism):
    Mechanism.BUFFER("scratch")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.NUMERICAL("i")

    def i(self, v):
        self.scratch.resize_(3, 2).fill_(101.0)
        return v


def _mechanism(cls, dtype=torch.float64):
    shape = (2, 3)
    return cls(
        cls.__name__,
        torch.full(shape, 34.0, dtype=dtype),
        torch.ones(shape, dtype=dtype),
        shape,
        shape,
        dtype=dtype,
    )


def _jacobian_matrix(function, voltage):
    jacobian = torch.autograd.functional.jacobian(function, voltage)
    return jacobian.reshape(voltage.numel(), voltage.numel())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_declared_pointwise_numerical_current_matches_full_jacobian_diagonal(dtype):
    mechanism = _mechanism(_PointwiseNonlinearNumerical, dtype)
    voltage = torch.tensor([[-70.0, -2.0, -0.25], [0.0, 0.5, 4.0]], dtype=dtype)

    current, conductance = mechanism.i_with_g(voltage)
    jacobian = _jacobian_matrix(mechanism.i, voltage)
    diagonal = jacobian.diagonal().reshape_as(voltage)
    off_diagonal = jacobian - torch.diag_embed(jacobian.diagonal())

    tolerance = 5.0e-5 if dtype == torch.float32 else 2.0e-9
    torch.testing.assert_close(current, mechanism.i(voltage))
    torch.testing.assert_close(
        conductance,
        diagonal,
        rtol=tolerance,
        atol=tolerance,
    )
    torch.testing.assert_close(off_diagonal, torch.zeros_like(off_diagonal))
    assert mechanism._current_conductance_mode == {"i": "numerical-declared"}
    assert mechanism._current_conductance_fallback_reason == {"i": None}


def test_declared_numerical_conductance_remains_differentiable_to_outer_autograd():
    mechanism = _mechanism(_PointwiseNonlinearNumerical)
    voltage = torch.tensor(
        [[-0.75, -0.5, -0.25], [0.0, 0.25, 0.75]],
        dtype=torch.float64,
        requires_grad=True,
    )

    _, conductance = mechanism.i_with_g(voltage)
    outer_gradient = torch.autograd.grad(conductance.sum(), voltage)[0]
    expected = 6.0 * mechanism.a * voltage + 2.0 * mechanism.b

    torch.testing.assert_close(outer_gradient, expected, rtol=2.0e-9, atol=2.0e-9)
    assert voltage.grad is None


def test_declared_numerical_save_current_updates_its_mirror_buffer():
    mechanism = _mechanism(_SavedPointwiseNonlinearNumerical)
    voltage = torch.tensor([[-2.0, -0.5, 0.0], [0.25, 1.0, 4.0]], dtype=torch.float64)

    current, _ = mechanism.i_with_g(voltage)

    torch.testing.assert_close(mechanism.i_, current)
    assert mechanism._current_conductance_mode == {"i": "numerical-declared"}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_declared_numerical_low_precision_error_directs_to_exact_pair(dtype):
    mechanism = _mechanism(_PointwiseNonlinearNumerical, dtype)

    with pytest.raises(TypeError) as exc_info:
        mechanism.i_with_g(torch.zeros((2, 3), dtype=dtype))

    message = str(exc_info.value)
    assert "_PointwiseNonlinearNumerical" in message
    assert "current 'i'" in message
    assert "supports only torch.float32 and torch.float64" in message
    assert "analytic current/conductance pair" in message


def test_declared_numerical_current_is_consistent_through_handler_i_and_idf():
    mechanism = _mechanism(_PointwiseNonlinearNumerical)
    shape = (2, 3)
    handler = MechanismHandler(
        torch.full(shape, 34.0, dtype=torch.float64),
        torch.ones(shape, dtype=torch.float64),
        {"nonlinear": mechanism},
        currents={"nonspecific": {"nonlinear": ["i"]}},
    ).to(torch.float64)
    handler.make_maps()
    handler.init_i_g_bufs(torch.zeros(shape, dtype=torch.float64))
    voltage = torch.tensor([[-2.0, -0.5, 0.0], [0.25, 1.0, 4.0]], dtype=torch.float64)
    previous_voltage = torch.tensor(
        [[-3.0, -0.75, 0.5], [1.5, 3.0, 6.0]], dtype=torch.float64
    )

    current, conductance = handler.i(voltage)
    idf_current, idf_conductance = handler.idf(voltage, previous_voltage)

    torch.testing.assert_close(current, mechanism.i(voltage))
    torch.testing.assert_close(
        conductance,
        3.0 * mechanism.a * voltage**2 + 2.0 * mechanism.b * voltage,
        rtol=2.0e-9,
        atol=2.0e-9,
    )
    # A valid numerical derivative does not prove affinity. Dufort--Frankel
    # therefore treats this nonlinear current explicitly at the current voltage.
    torch.testing.assert_close(idf_current, mechanism.i(voltage))
    torch.testing.assert_close(idf_conductance, torch.zeros_like(voltage))


def test_declared_numerical_contract_rejects_demonstrable_tensor_coupling():
    mechanism = _mechanism(_DeclaredCoupledCurrent)
    voltage = torch.tensor(
        [[-70.0, -60.0, -50.0], [-40.0, -30.0, -20.0]],
        dtype=torch.float64,
    )

    with pytest.raises(NumericalCurrentContractError) as exc_info:
        mechanism.i_with_g(voltage)

    message = str(exc_info.value)
    assert "_DeclaredCoupledCurrent" in message
    assert "Mechanism.NUMERICAL('i')" in message
    assert "demonstrating tensor coupling" in message
    assert "Genuinely coupled voltage dependence must be modeled outside" in message
    assert "not proof for unprobed or state-dependent behavior" in message


def test_declared_numerical_contract_rejects_and_restores_buffer_mutation():
    mechanism = _mechanism(_DeclaredStatefulCurrent)
    voltage = torch.full((2, 3), -65.0, dtype=torch.float64)
    original_calls = mechanism.calls.clone()
    original_object = mechanism.calls

    with pytest.raises(NumericalCurrentContractError) as exc_info:
        mechanism.i_with_g(voltage)

    message = str(exc_info.value)
    assert "registered buffer 'calls' was modified in place" in message
    assert mechanism.calls is original_object
    torch.testing.assert_close(mechanism.calls, original_calls)


def test_declared_numerical_contract_rejects_nondeterminism_and_restores_rng():
    mechanism = _mechanism(_DeclaredNondeterministicCurrent)
    voltage = torch.full((2, 3), -65.0, dtype=torch.float64)
    torch.manual_seed(90210)
    original_rng_state = torch.random.get_rng_state().clone()

    with pytest.raises(NumericalCurrentContractError) as exc_info:
        mechanism.i_with_g(voltage)

    assert "demonstrating nondeterminism" in str(exc_info.value)
    assert torch.equal(torch.random.get_rng_state(), original_rng_state)


def test_declared_numerical_contract_requires_exact_current_shape():
    mechanism = _mechanism(_DeclaredBadShapeCurrent)
    voltage = torch.full((2, 3), -65.0, dtype=torch.float64)

    with pytest.raises(NumericalCurrentContractError) as exc_info:
        mechanism.i_with_g(voltage)

    message = str(exc_info.value)
    assert "returned shape (2, 1)" in message
    assert "voltage shape is (2, 3)" in message


def test_declared_numerical_contract_requires_exact_current_dtype():
    mechanism = _mechanism(_DeclaredBadDtypeCurrent)
    voltage = torch.full((2, 3), -65.0, dtype=torch.float64)

    with pytest.raises(NumericalCurrentContractError) as exc_info:
        mechanism.i_with_g(voltage)

    message = str(exc_info.value)
    assert "returned dtype torch.float32" in message
    assert "voltage has dtype torch.float64" in message


def test_declared_numerical_failure_restores_mutated_buffer_and_rng_together():
    mechanism = _mechanism(_DeclaredMutatingRandomCurrent)
    voltage = torch.full((2, 3), -65.0, dtype=torch.float64)
    original_calls = mechanism.calls.clone()
    original_object = mechanism.calls
    torch.manual_seed(1729)
    original_rng_state = torch.random.get_rng_state().clone()

    with pytest.raises(NumericalCurrentContractError, match="modified in place"):
        mechanism.i_with_g(voltage)

    assert mechanism.calls is original_object
    torch.testing.assert_close(mechanism.calls, original_calls)
    assert torch.equal(torch.random.get_rng_state(), original_rng_state)


def test_declared_numerical_contract_rejects_and_restores_voltage_input_mutation():
    mechanism = _mechanism(_DeclaredVoltageMutatingCurrent)
    voltage = torch.tensor(
        [[-70.0, -60.0, -50.0], [-40.0, -30.0, -20.0]],
        dtype=torch.float64,
    )
    original_voltage = voltage.clone()
    original_stride = voltage.stride()

    with pytest.raises(NumericalCurrentContractError) as exc_info:
        mechanism.i_with_g(voltage)

    message = str(exc_info.value)
    assert "voltage input was modified in place" in message
    assert "shape changed from (2, 3) to (3, 2)" in message
    assert "stride changed" in message
    assert isinstance(exc_info.value.__cause__, ValueError)
    # The public input is isolated from probes, and even a retained reference to
    # the actual probe is restored transactionally before the error escapes.
    torch.testing.assert_close(voltage, original_voltage)
    assert voltage.stride() == original_stride
    assert mechanism.observed_voltage.shape == original_voltage.shape
    assert mechanism.observed_voltage.stride() == original_stride
    torch.testing.assert_close(mechanism.observed_voltage, original_voltage)


def test_declared_numerical_contract_rejects_nonraising_voltage_zeroing():
    mechanism = _mechanism(_DeclaredZeroingVoltageCurrent)
    voltage = torch.tensor(
        [[-70.0, -60.0, -50.0], [-40.0, -30.0, -20.0]],
        dtype=torch.float64,
    )
    original_voltage = voltage.clone()

    with pytest.raises(
        NumericalCurrentContractError, match="voltage input was modified in place"
    ):
        mechanism.i_with_g(voltage)

    torch.testing.assert_close(voltage, original_voltage)


def test_declared_numerical_contract_restores_resized_registered_tensor_metadata():
    mechanism = _mechanism(_DeclaredResizingStateCurrent)
    voltage = torch.full((2, 3), -65.0, dtype=torch.float64)
    noncontiguous = torch.arange(6.0, dtype=torch.float64).reshape(3, 2).t()
    mechanism.scratch = noncontiguous
    original_object = mechanism.scratch
    original_value = original_object.clone()
    original_shape = original_object.shape
    original_stride = original_object.stride()
    original_storage = original_object.untyped_storage()

    with pytest.raises(NumericalCurrentContractError) as exc_info:
        mechanism.i_with_g(voltage)

    message = str(exc_info.value)
    assert "registered buffer 'scratch' was modified in place" in message
    assert "shape changed from (2, 3) to (3, 2)" in message
    assert "stride changed from (1, 2) to (2, 1)" in message
    assert mechanism.scratch is original_object
    assert mechanism.scratch.shape == original_shape
    assert mechanism.scratch.stride() == original_stride
    assert mechanism.scratch.untyped_storage() is original_storage
    torch.testing.assert_close(mechanism.scratch, original_value)


def test_population_initialize_enforces_declared_numerical_contract():
    population = dn.Population(N=1, C=6, v_init=-65.0, dtype=torch.float64)
    population.insert(_DeclaredCoupledCurrent)

    with pytest.raises(
        NumericalCurrentContractError, match="demonstrating tensor coupling"
    ):
        population.initialize()

    assert population.initialized is False


def test_declared_numerical_current_compiles_after_eager_validation():
    mechanism = _mechanism(_PointwiseNonlinearNumerical)
    voltage = torch.tensor([[-2.0, -0.5, 0.0], [0.25, 1.0, 4.0]], dtype=torch.float64)
    expected = mechanism.i_with_g(voltage)

    def evaluate(v):
        return mechanism.i_with_g(v)

    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    actual = compiled(voltage)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_coupled_current_is_rejected_instead_of_returning_directional_derivative():
    voltage = torch.zeros((2, 2), dtype=torch.float64)

    def coupled(v):
        return v + v.mean()

    jacobian = _jacobian_matrix(coupled, voltage)
    required_diagonal = jacobian.diagonal().reshape_as(voltage)
    step = torch.full_like(voltage, torch.finfo(voltage.dtype).eps ** (1.0 / 3.0))
    old_simultaneous_result = (coupled(voltage + step) - coupled(voltage - step)) / (
        2.0 * step
    )

    torch.testing.assert_close(required_diagonal, torch.full_like(voltage, 1.25))
    torch.testing.assert_close(old_simultaneous_result, torch.full_like(voltage, 2.0))
    assert not torch.allclose(old_simultaneous_result, required_diagonal)

    with pytest.raises(UnsafeAutomaticNumericalFallbackError) as exc_info:
        _mechanism(_CoupledCurrent)

    message = str(exc_info.value)
    assert "_CoupledCurrent" in message
    assert "current 'i'" in message
    assert "directional derivative" in message
    assert "Jacobian diagonal" in message
    assert "model genuinely coupled voltage dependence outside" in message
    assert "exact analytic i_with_conductance(self, v)" in message
    assert "Mechanism.NUMERICAL('i')" in message
    assert isinstance(exc_info.value.__cause__, NotImplementedError)


@pytest.mark.parametrize(
    "mechanism_cls,root_cause",
    [
        (_StatefulCurrent, "Only straight-line code is supported"),
        (_NondeterministicCurrent, "Unsupported AST node"),
    ],
)
def test_automatic_numerical_path_rejects_unproven_impure_currents(
    mechanism_cls, root_cause
):
    with pytest.raises(UnsafeAutomaticNumericalFallbackError) as exc_info:
        _mechanism(mechanism_cls)

    message = str(exc_info.value)
    assert mechanism_cls.__name__ in message
    assert root_cause in message
    assert "deterministic, side-effect-free, pointwise" in message
    assert "For a pointwise current, rewrite it" in message
    assert "provide an exact analytic i_with_conductance(self, v)" in message
    assert "Mechanism.NUMERICAL('i')" in message
