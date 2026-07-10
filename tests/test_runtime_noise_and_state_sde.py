import torch

import dendra as dn  # noqa: F401 - ensure top-level API exports are available
from dendra.models.mechanisms._state import State, valid_integration_methods
from dendra.models.parametric import Parameterized


def test_runtime_noise_buffer_resamples_detached_in_place():
    class P(Parameterized):
        Parameterized.RANGENOISE(
            "eta", distribution="normal", mu=0.0, sigma=1.0, seed=123
        )

    p = P(shape=(1, 8), shape_f=(1, 8), dtype=torch.float32)
    p.populate_parameter_buffers()

    first = p.eta.clone()
    buffer_id = id(p.eta)
    p.sample_runtime_noises_(dt=torch.tensor(0.1), phase="pre_state")

    assert id(p.eta) == buffer_id
    assert not torch.equal(first, p.eta)
    assert not p.eta.requires_grad
    assert hasattr(p, "eta_mu")
    assert hasattr(p, "eta_sigma")


def test_truncated_normal_and_distribution_registration_api_are_exported():
    assert "truncated_normal" in dn.available_random_distributions()
    spec = dn.get_random_distribution("truncated_normal")
    assert tuple(spec.parameter_defaults) == ("mu", "sigma", "low", "high")

    def constant_sampler(parameters, rng, shape, *, device, dtype):
        return (
            torch.empty(shape, device=device, dtype=dtype).fill_(1.0)
            * parameters["value"]
        )

    dn.register_random_distribution(
        "constant_runtime_test",
        {"value": 2.5},
        sampler=constant_sampler,
        overwrite=True,
    )

    class P(Parameterized):
        Parameterized.RANGENOISE("eta", distribution="constant_runtime_test", value=3.0)

    p = P(shape=(1, 5), shape_f=(1, 5), dtype=torch.float32)
    p.populate_parameter_buffers()
    assert torch.allclose(p.eta, torch.full_like(p.eta, 3.0))


def test_state_euler_maruyama_is_registered_and_runs():
    assert "euler_maruyama" in valid_integration_methods()

    class OU(State):
        State.STATE("x")
        State.RANGE(mu=0.0, tau=10.0, sigma=1.0)
        State.ASSIGNED("drift_mu", "drift_tau", "diff_sigma")
        State.DERIVATIVE("x' = (drift_mu - x) / drift_tau")
        State.DIFFUSION("x = diff_sigma")
        State.METHOD("euler_maruyama")

        def breakpoint(self, v, states):
            return {
                "drift_mu": self.mu,
                "drift_tau": self.tau,
                "diff_sigma": self.sigma,
            }

    state = OU(
        torch.tensor(36.0),
        torch.ones(1, 6),
        torch.arange(6),
        shape=(1, 6),
        shape_f=(1, 6),
    )
    state.populate_parameter_buffers()
    state.init_rng()

    out = state.advance(
        torch.zeros(1, 6),
        torch.tensor(0.1),
        {"x": torch.zeros(1, 6)},
    )

    assert set(out) == {"x"}
    assert out["x"].shape == (1, 6)
    assert not torch.allclose(out["x"], torch.zeros_like(out["x"]))


def test_state_diffusion_requires_euler_maruyama():
    class Bad(State):
        State.STATE("x")
        State.DERIVATIVE("x' = -x")
        State.DIFFUSION("x = 1.0")
        State.METHOD("cnexp")

    try:
        Bad(
            torch.tensor(36.0),
            torch.ones(1, 1),
            torch.arange(1),
            shape=(1, 1),
            shape_f=(1, 1),
        )
    except ValueError as exc:
        assert "DIFFUSION" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("State.DIFFUSION should require euler_maruyama in v1")


def test_mechanism_handler_exposes_runtime_noise_sampling():
    from dendra.models.mechanisms._handler import MechanismHandler
    from dendra.models.mechanisms._mechanism import Mechanism

    class NoiseMech(Mechanism):
        Mechanism.RANGENOISE("eta", distribution="normal", mu=0.0, sigma=1.0, seed=321)

    mech = NoiseMech(
        "noise",
        torch.tensor(36.0),
        torch.ones(1, 4),
        shape=(1, 4),
        shape_f=(1, 4),
        key=torch.arange(4),
    )
    handler = MechanismHandler(
        torch.tensor(36.0),
        torch.ones(1, 4),
        {"noise": mech},
        currents={},
    )
    handler.populate()
    first = handler.noise.eta.clone()

    sampled = handler.sample_runtime_noises_(dt=torch.tensor(0.1), phase="pre_state")

    assert sampled is True
    assert not torch.equal(first, handler.noise.eta)


def test_state_euler_heun_is_registered_and_recomputes_diffusion_at_predictor():
    assert "euler_heun" in valid_integration_methods()

    class Geometric(State):
        State.STATE("x")
        State.RANGE(sigma=2.0)
        State.ASSIGNED("diff_sigma")
        State.DERIVATIVE("x' = 0.0")
        State.DIFFUSION("x = diff_sigma * x")
        State.METHOD("euler_heun")

        def breakpoint(self, v, states):
            return {"diff_sigma": self.sigma}

    state = Geometric(
        torch.tensor(36.0),
        torch.ones(1, 1),
        torch.arange(1),
        shape=(1, 1),
        shape_f=(1, 1),
    )
    state.populate_parameter_buffers()
    state.init_rng()

    def one_like(_shape, *, device=None, dtype=None):
        return torch.ones(_shape, device=device, dtype=dtype)

    state.x_dW_rng.randn = one_like
    x0 = torch.ones(1, 1)
    dt = torch.tensor(0.25)
    out = state.advance(torch.zeros(1, 1), dt, {"x": x0})["x"]

    dW = torch.sqrt(dt)
    diff_old = 2.0 * x0
    pred = x0 + diff_old * dW
    diff_pred = 2.0 * pred
    expected = x0 + 0.5 * (diff_old + diff_pred) * dW
    assert torch.allclose(out, expected)


def test_state_euler_heun_average_drift_option():
    class LinearDrift(State):
        State.STATE("x")
        State.DERIVATIVE("x' = x")
        State.DIFFUSION("x = 0.0")
        State.METHOD("euler_heun", average_drift=True)

    state = LinearDrift(
        torch.tensor(36.0),
        torch.ones(1, 1),
        torch.arange(1),
        shape=(1, 1),
        shape_f=(1, 1),
    )
    state.populate_parameter_buffers()
    state.init_rng()

    def zero_like(_shape, *, device=None, dtype=None):
        return torch.zeros(_shape, device=device, dtype=dtype)

    state.x_dW_rng.randn = zero_like
    x0 = torch.ones(1, 1)
    dt = torch.tensor(0.1)
    out = state.advance(torch.zeros(1, 1), dt, {"x": x0})["x"]
    pred = x0 + x0 * dt
    expected = x0 + 0.5 * (x0 + pred) * dt
    assert torch.allclose(out, expected)
