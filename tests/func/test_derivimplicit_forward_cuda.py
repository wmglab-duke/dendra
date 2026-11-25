"""CUDA Inductor qualification for the default coupled Newton forward.

Run this file directly on CUDA workers without pytest installed.
"""

import torch
from torch._inductor import config

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mechanisms._derivimplicit_forward import _PROGRAMS

if __name__ != "__main__":
    import pytest

    pytestmark = [
        pytest.mark.cuda,
        pytest.mark.slow,
        pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
    ]


class _CoupledDecayState(State):
    State.STATE("u", "w")
    State.GLOBAL(rate=2.0)
    State.DERIVATIVE("u' = -rate*u*u", "w' = rate*u*u")
    State.METHOD("derivimplicit")

    def state_defaults(self, v, values):
        return {"u": torch.full_like(v, 0.5), "w": torch.full_like(v, 0.3)}


class _CoupledDecay(Mechanism):
    Mechanism.STATE_BUNDLE(_CoupledDecayState)


def _reference(u, w, rate, dt, steps, weights):
    inputs = tuple(
        value.detach().cpu().double().requires_grad_() for value in (u, w, rate)
    )
    u, w, rate = inputs
    total = u + w
    # Match the representable timestep used by the float32 simulation.
    dt = float(torch.tensor(dt, dtype=torch.float32, device="cpu"))
    for _ in range(steps):
        u = 2 * u / (1 + (1 + 4 * dt * rate * u).sqrt())
    final = (u, total - u)
    gradients = torch.autograd.grad(
        final, inputs, tuple(weight.cpu().double() for weight in weights)
    )
    return final, gradients


def test_cuda_inductor_uses_separate_newton_by_default_and_recovers_after_failure():
    dt, steps = 0.1, 4
    dtype = torch.float32
    name = _CoupledDecay.__name__
    with (
        torch_compiler_warning_context(),
        config.patch({"triton.cudagraphs": False}),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=dtype, DEVICE="cuda", IMEM=False),
    ):
        model = dn.SingleCompartment(N=2, C=1, dtype=dtype).train()
        model.insert(_CoupledDecay)
        model.initialize()
        previous_programs = set(_PROGRAMS)
        runner = dn.compile(
            model,
            dt=dt,
            steps=steps,
            execution="scan",
            inference=False,
            backend="inductor",
            trainable="*rate_param",
        )
        initial = runner.initial_state
        u = torch.tensor(
            [[0.5], [1.25]], dtype=dtype, device="cuda", requires_grad=True
        )
        w = torch.tensor([[0.3], [0.6]], dtype=dtype, device="cuda", requires_grad=True)
        (rate,) = runner.trainable_parameters.values()
        weights = (u.new_tensor([[0.3], [1.3]]), w.new_tensor([[0.7], [-0.2]]))

        def incoming(u):
            return {
                **initial,
                "mechanisms": {**initial["mechanisms"], name: {"u": u, "w": w}},
            }

        def values_and_gradients(state):
            final, _ = runner(state)
            values = tuple(final["mechanisms"][name][key] for key in ("u", "w"))
            gradients = torch.autograd.grad(values, (u, w, rate), weights)
            return tuple(
                tuple(x.detach() for x in group) for group in (values, gradients)
            )

        valid = incoming(u)
        actual = values_and_gradients(valid)
        assert runner.execution_report().strategy == "compiled_scan"
        programs = set(_PROGRAMS) - previous_programs
        assert len(programs) == 1
        expected = _reference(u, w, rate, dt, steps, weights)
        cpu_values, cpu_gradients = (
            tuple(value.cpu().double() for value in group) for group in actual
        )
        torch.testing.assert_close(cpu_values, expected[0], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(cpu_gradients, expected[1], rtol=3e-4, atol=1e-5)

        # The second lane starts with a real backward-Euler root. That root
        # lies below -1.25, so the second step has no real root.
        failing_u = u.detach().clone()
        failing_u[1, 0] = -1.0
        frozen = failing_u.clone()
        try:
            runner(incoming(failing_u.requires_grad_()))
        except RuntimeError as error:
            message = str(error)
            assert f"derivimplicit failed to converge in {name}." in message
            assert "chunk step 1, batch index (1, 0)" in message
        else:
            raise AssertionError(
                "Expected a no-root failure in the second batch member"
            )
        torch.testing.assert_close(failing_u, frozen, rtol=0, atol=0)

        replay = values_and_gradients(valid)
        torch.testing.assert_close(replay, actual, rtol=0, atol=0)
        assert set(_PROGRAMS) - previous_programs == programs

        with torch.no_grad():
            final, _ = runner(valid)
        values = tuple(final["mechanisms"][name][key] for key in ("u", "w"))
        assert all(
            not value.requires_grad and value.grad_fn is None for value in values
        )
        torch.testing.assert_close(
            tuple(value.cpu().double() for value in values),
            expected[0],
            rtol=2e-5,
            atol=2e-6,
        )
        assert set(_PROGRAMS) - previous_programs == programs


if __name__ == "__main__":
    test_cuda_inductor_uses_separate_newton_by_default_and_recovers_after_failure()
    print(
        "CUDA Inductor Newton forward, gradients, failure recovery, and no-grad passed."
    )
