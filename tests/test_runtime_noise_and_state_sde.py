import copy

import pytest
import torch

import dendra as dn  # noqa: F401 - ensure top-level API exports are available
from dendra.models.mechanisms import MaterialProcess, Mechanism
from dendra.models.mechanisms._handler import MechanismHandler
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


def test_handler_checkpoint_replays_runtime_noise_and_random_parameter_streams():
    class StochasticMechanism(Mechanism):
        Mechanism.RANGERAND(
            "quenched", distribution="normal", mu=0.0, sigma=1.0, seed=123
        )
        Mechanism.RANGENOISE("eta", distribution="normal", mu=0.0, sigma=1.0, seed=456)

    shape = (1, 5)
    mech = StochasticMechanism(
        "stochastic",
        torch.tensor(36.0),
        torch.ones(shape),
        shape,
        shape,
        key=torch.arange(shape[-1]),
    )
    handler = MechanismHandler(
        torch.tensor(36.0), torch.ones(shape), {"stochastic": mech}
    )
    handler.populate(random_generation=object())
    snapshot = handler.mutable_state_dict()
    boundary_quenched = mech.quenched.clone()
    boundary_eta = mech.eta.clone()

    handler.resample_random_parameters("quenched")
    handler.sample_runtime_noises_(dt=torch.tensor(0.1), phase="pre_state")
    expected_quenched = mech.quenched.clone()
    expected_eta = mech.eta.clone()
    mech.quenched_rng.reseed(1001)
    mech.eta_rng.reseed(1002)

    handler.restore_mutable_state_dict(snapshot)
    torch.testing.assert_close(mech.quenched, boundary_quenched)
    torch.testing.assert_close(mech.eta, boundary_eta)
    assert mech.quenched_rng._base_seed == 123
    assert mech.eta_rng._base_seed == 456
    handler.resample_random_parameters("quenched")
    handler.sample_runtime_noises_(dt=torch.tensor(0.1), phase="pre_state")

    torch.testing.assert_close(mech.quenched, expected_quenched)
    torch.testing.assert_close(mech.eta, expected_eta)


def test_handler_checkpoint_replays_nested_state_sde_rng():
    class DiffusiveState(State):
        State.STATE("x")
        State.DERIVATIVE("x' = 0.0")
        State.DIFFUSION("x = 1.0")
        State.METHOD("euler_maruyama")

    class DiffusiveMechanism(Mechanism):
        Mechanism.STATE(DiffusiveState)

    shape = (1, 4)
    mech = DiffusiveMechanism(
        "diffusive",
        torch.tensor(36.0),
        torch.ones(shape),
        shape,
        shape,
        key=torch.arange(shape[-1]),
    )
    handler = MechanismHandler(
        torch.tensor(36.0), torch.ones(shape), {"diffusive": mech}
    )
    handler.init_rng()
    mech.x = torch.zeros(shape)
    state = next(iter(mech.DE.values()))
    voltage = torch.zeros(shape)
    dt = torch.tensor(0.1)
    snapshot = handler.mutable_state_dict()

    def advance_twice():
        outputs = []
        for _ in range(2):
            mech.x = state.advance(voltage, dt, {"x": mech.x})["x"]
            outputs.append(mech.x.clone())
        return outputs

    expected = advance_twice()
    handler.restore_mutable_state_dict(snapshot)
    actual = advance_twice()

    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference)


def test_handler_checkpoint_restore_is_atomic_on_corrupt_rng_state():
    class ScratchMechanism(Mechanism):
        Mechanism.BUFFER("scratch")

    class NoiseMechanism(Mechanism):
        Mechanism.RANGENOISE("eta", distribution="normal", mu=0.0, sigma=1.0, seed=789)

    shape = (1, 3)
    celsius = torch.tensor(36.0)
    diameters = torch.ones(shape)
    scratch = ScratchMechanism(
        "scratch", celsius, diameters, shape, shape, key=torch.arange(shape[-1])
    )
    noise = NoiseMechanism(
        "noise", celsius, diameters, shape, shape, key=torch.arange(shape[-1])
    )
    handler = MechanismHandler(
        celsius, torch.ones(shape), {"scratch": scratch, "noise": noise}
    )
    scratch.scratch = torch.ones(shape)
    target = handler.mutable_state_dict()

    scratch.scratch = torch.full(shape, 9.0)
    handler.sample_runtime_noises_(dt=torch.tensor(0.1), phase="pre_state")
    before_scratch = scratch.scratch.clone()
    before_eta = noise.eta.clone()
    before_scratch_ref = scratch.scratch
    before_eta_ref = noise.eta
    before_rng = noise.eta_rng.rng_state()["cpu"].clone()

    corrupt = copy.deepcopy(target)
    rng_key = next(
        key
        for key in corrupt
        if key.endswith(".__dendra_stochastic_state__.rng.eta_rng")
    )
    corrupt[rng_key]["rng_state"]["cpu"] = torch.zeros(1, dtype=torch.uint8)

    with pytest.raises(RuntimeError):
        handler.restore_mutable_state_dict(corrupt)

    torch.testing.assert_close(scratch.scratch, before_scratch)
    torch.testing.assert_close(noise.eta, before_eta)
    assert scratch.scratch is before_scratch_ref
    assert noise.eta is before_eta_ref
    assert torch.equal(noise.eta_rng.rng_state()["cpu"], before_rng)


def _delayed_handler(*, batched=False):
    shape = (2, 2) if batched else (2,)
    celsius = torch.full(shape, 36.0, dtype=torch.float64)
    mech = Mechanism("base", celsius, torch.ones(shape), shape, shape)
    if batched:
        mech.register_delayed_states(
            "signal", torch.zeros(shape, dtype=torch.float64), [0, 2], stream_axis=0
        )
    else:
        mech.register_delayed_state(
            "signal", torch.zeros(shape, dtype=torch.float64), 1, mode="shift"
        )
    handler = MechanismHandler(celsius, torch.ones(shape), {"base": mech})
    return handler, mech


def test_handler_restore_preflights_shape_and_casts_to_live_dtype():
    class ScratchMechanism(Mechanism):
        Mechanism.BUFFER("scratch")

    shape = (1, 3)
    celsius = torch.full(shape, 36.0, dtype=torch.float64)
    mech = ScratchMechanism("scratch", celsius, torch.ones(shape), shape, shape)
    handler = MechanismHandler(celsius, torch.ones(shape), {"scratch": mech})
    mech.scratch = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    snapshot = handler.mutable_state_dict()
    snapshot["scratch.scratch"] = snapshot["scratch.scratch"].float()

    handler.restore_mutable_state_dict(snapshot)

    assert mech.scratch.dtype == torch.float64
    torch.testing.assert_close(
        mech.scratch, torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    )

    before_ref = mech.scratch
    before_value = mech.scratch.clone()
    corrupt = dict(snapshot)
    corrupt["scratch.scratch"] = torch.zeros(3, dtype=torch.float32)
    with pytest.raises(ValueError, match="shape"):
        handler.restore_mutable_state_dict(corrupt)

    assert mech.scratch is before_ref
    torch.testing.assert_close(mech.scratch, before_value)


@pytest.mark.parametrize(
    "malformed",
    [
        {},
        {
            "signal": {
                "buffer": "wrong_buffer",
                "pointer": "signal_delay_ptr",
                "steps": 1,
                "depth": 2,
                "axis": 1,
                "value_shape": (2,),
                "mode": "shift",
            }
        },
    ],
)
def test_handler_restore_rejects_malformed_delay_registration_atomically(malformed):
    handler, mech = _delayed_handler()
    snapshot = handler.mutable_state_dict()
    metadata_key = "base.__dendra_delayed_state_specs__"
    corrupt = dict(snapshot)
    corrupt[metadata_key] = malformed
    buffer_ref = mech.signal_delay_buffer
    pointer_ref = mech.signal_delay_ptr
    specs_ref = mech._delayed_state_specs
    before_buffer = buffer_ref.clone()

    with pytest.raises((TypeError, ValueError, KeyError)):
        handler.restore_mutable_state_dict(corrupt)

    assert mech.signal_delay_buffer is buffer_ref
    assert mech.signal_delay_ptr is pointer_ref
    assert mech._delayed_state_specs is specs_ref
    torch.testing.assert_close(mech.signal_delay_buffer, before_buffer)
    assert set(mech._delayed_state_specs) == {"signal"}


def test_handler_restore_validates_queue_shape_from_saved_delay_metadata():
    handler, mech = _delayed_handler()
    snapshot = handler.mutable_state_dict()
    mech.delayed_state(
        "signal", torch.ones(2, dtype=torch.float64), delay_steps=3, mode="shift"
    )
    assert mech.signal_delay_buffer.shape == (2, 4)
    live_buffer = mech.signal_delay_buffer
    live_pointer = mech.signal_delay_ptr
    live_specs = mech._delayed_state_specs

    corrupt = dict(snapshot)
    # This matches the resized live queue but not the saved one-step metadata.
    corrupt["base.signal_delay_buffer"] = torch.zeros_like(live_buffer)
    with pytest.raises(ValueError, match=r"expected \(2, 2\)"):
        handler.restore_mutable_state_dict(corrupt)

    assert mech.signal_delay_buffer is live_buffer
    assert mech.signal_delay_ptr is live_pointer
    assert mech._delayed_state_specs is live_specs
    assert mech._delayed_state_specs["signal"]["steps"] == 3


@pytest.mark.parametrize("field", ["steps", "has_zero_delay", "all_zero_delay"])
def test_handler_restore_cross_checks_batched_delay_metadata(field):
    handler, mech = _delayed_handler(batched=True)
    snapshot = handler.mutable_state_dict()
    metadata_key = "base.__dendra_delayed_state_specs__"
    corrupt = copy.deepcopy(snapshot)
    spec = corrupt[metadata_key]["signal"]
    if field == "steps":
        corrupt["base.signal_delay_steps"] = torch.tensor([0, 1])
    else:
        spec[field] = not spec[field]
    buffer_ref = mech.signal_delay_buffer
    specs_ref = mech._delayed_state_specs

    with pytest.raises(ValueError, match="inconsistent|maximum"):
        handler.restore_mutable_state_dict(corrupt)

    assert mech.signal_delay_buffer is buffer_ref
    assert mech._delayed_state_specs is specs_ref


@pytest.mark.parametrize("legacy_state_name", ["alpha", "omega"])
def test_nested_state_rng_uses_canonical_key_and_restores_legacy_fallback(
    legacy_state_name,
):
    class RandomState(State):
        State.STATE("omega", "alpha")
        State.DERIVATIVE("omega' = 0.0", "alpha' = 0.0")
        State.RNG(stream=123)

    class RandomMechanism(Mechanism):
        Mechanism.STATE(RandomState)

    shape = (1, 4)
    celsius = torch.full(shape, 36.0)
    mech = RandomMechanism(
        "random", celsius, torch.ones(shape), shape, shape, key=torch.arange(4)
    )
    handler = MechanismHandler(celsius, torch.ones(shape), {"random": mech})
    state = mech.DE["RandomState"]
    canonical_key = "random.DE.RandomState.__dendra_stochastic_state__.rng.stream"
    snapshot = handler.mutable_state_dict()

    assert canonical_key in snapshot
    assert "random.alpha.stream" not in snapshot
    assert "random.omega.stream" not in snapshot
    expected = state.stream.rand((5,))

    legacy = dict(snapshot)
    payload = legacy.pop(canonical_key)
    legacy[f"random.{legacy_state_name}.stream"] = payload["rng_state"]
    handler.restore_mutable_state_dict(legacy)
    actual = state.stream.rand((5,))

    torch.testing.assert_close(actual, expected)


def test_material_process_restore_has_the_same_atomic_shape_contract():
    class ScratchProcess(MaterialProcess):
        MaterialProcess.BUFFER("scratch")

    shape = (1, 3)
    celsius = torch.full(shape, 36.0)
    process = ScratchProcess("process", celsius, torch.ones(shape), shape, shape)
    handler = MechanismHandler(celsius, torch.ones(shape), {"process": process})
    process.scratch = torch.tensor([[2.0, 3.0, 5.0]])
    snapshot = handler.mutable_state_dict()
    before_ref = process.scratch
    before_value = process.scratch.clone()
    snapshot["process.scratch"] = torch.zeros(3)

    with pytest.raises(ValueError, match="shape"):
        handler.restore_mutable_state_dict(snapshot)

    assert process.scratch is before_ref
    torch.testing.assert_close(process.scratch, before_value)
