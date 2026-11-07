"""Runtime policy changes must select fresh compiled behavior, not stale graphs."""

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism


class InconsistentAffine(Mechanism):
    """Deliberately inconsistent coefficients make policy selection observable."""

    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")
    Mechanism.GLOBAL(g=0.02, e=-40.0)

    def i(self, v):
        return self.g * (v - self.e)

    def i_with_conductance(self, v):
        return self.i(v), self.g

    def i_affine_terms(self, v):
        return 2.0 * self.g, 2.0 * self.g * self.e


def model():
    with dn.ctx(JIT=0, IMEM=0, REQUIRE_GRAD=0):
        result = dn.SingleCompartment(
            N=1,
            C=2,
            dtype=torch.float64,
            v_init=-65.0,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        result.insert(InconsistentAffine)
        result.initialize()
    return result


def test_policy_normalization_nested_restoration_and_atomic_rejection():
    original = dn.current_membrane_assembly()
    with dn.ctx(MEMBRANE_ASSEMBLY=" OFF "):
        assert dn.current_membrane_assembly() == "off"
        with dn.ctx(MEMBRANE_ASSEMBLY="validate"):
            assert dn.current_membrane_assembly() == "validate"
        assert dn.current_membrane_assembly() == "off"
        for bad in (True, False, None, "fast", "legacy", 1):
            with pytest.raises(ValueError, match="MEMBRANE_ASSEMBLY"):
                with dn.ctx(MEMBRANE_ASSEMBLY=bad):
                    pass
            assert dn.current_membrane_assembly() == "off"
    assert dn.current_membrane_assembly() == original


@pytest.mark.parametrize("backend", ["eager", "aot_eager", "inductor"])
@pytest.mark.parametrize("execution", ["unrolled", "default"])
def test_policy_changes_on_existing_compiled_functional_kernel(backend, execution):
    source = model()
    functional, tensors = dn.func.make_functional(source, dt=0.01)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    kernel = functional.compile_rollout_chunk(4, execution=execution, backend=backend)
    bound = kernel.bind(tensors.parameters, prepared)
    values = []
    for policy in ("auto", "off", "auto"):
        with dn.ctx(MEMBRANE_ASSEMBLY=policy), torch.no_grad():
            state, _ = dn.func.run(bound, tensors.state, steps=4)
            values.append(state["integrator"]["v"])
    torch.testing.assert_close(values[0], values[2], rtol=0, atol=0)
    assert not torch.allclose(values[0], values[1])
    with dn.ctx(MEMBRANE_ASSEMBLY="validate"), torch.no_grad():
        with pytest.raises(RuntimeError, match="Invalid AFFINE"):
            dn.func.run(bound, tensors.state, steps=4)


@pytest.mark.parametrize("jit", [0, 1])
def test_policy_changes_on_existing_population(jit):
    source = model()
    values = []
    for policy in ("auto", "off", "auto"):
        with (
            dn.ctx(MEMBRANE_ASSEMBLY=policy, JIT=jit, BACKEND="eager"),
            torch.no_grad(),
        ):
            source.initialize()
            source.run(tstop=0.04, dt=0.01)
            values.append(source.v.clone())
    torch.testing.assert_close(values[0], values[2], rtol=0, atol=0)
    assert not torch.allclose(values[0], values[1])
    with (
        dn.ctx(MEMBRANE_ASSEMBLY="validate", JIT=jit, BACKEND="eager"),
        torch.no_grad(),
    ):
        source.initialize()
        with pytest.raises(RuntimeError, match="Invalid AFFINE"):
            source.run(tstop=0.04, dt=0.01)
