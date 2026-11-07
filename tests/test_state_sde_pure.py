"""State SDE solvers share explicit draws with Native sampling adapters."""

import copy
from types import MethodType, SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mechanisms._euler_heun import build_euler_heun
from dendra.models.mechanisms._euler_maruyama import build_euler_maruyama
from dendra.models.mechanisms._sde import sample_state_sde_draw

METHODS = [
    ("euler_maruyama", False),
    ("euler_heun", False),
    ("euler_heun", True),
]


def _make_state(method, average_drift=False, *, shape=(2, 3), dtype=torch.float64):
    class Geometric(State):
        State.STATE("x")
        State.INPUT("mu", "sigma")
        State.ASSIGNED("drift", "diffusion")
        State.DERIVATIVE("x' = drift")
        State.DIFFUSION("x = diffusion")
        if method == "euler_heun":
            State.METHOD(method, average_drift=average_drift)
        else:
            State.METHOD(method)

        def assigned_values(self, v, values):
            del v
            return {
                "drift": values["mu"] * values["x"],
                "diffusion": values["sigma"] * values["x"],
            }

    state = Geometric(
        torch.tensor(36.0, dtype=dtype),
        torch.ones(shape, dtype=dtype),
        torch.arange(shape[-1]),
        shape=shape,
        shape_f=shape,
    )
    state.populate_parameter_buffers()
    state.init_rng()
    state.x_dW_rng.reseed(123)
    return state


def _frame(shape=(2, 3), dtype=torch.float64, requires_grad=False):
    return {
        name: torch.full(shape, value, dtype=dtype, requires_grad=requires_grad)
        for name, value in (("x", 0.75), ("mu", 0.2), ("sigma", 0.4))
    }


def _reference(method, average_drift, x, mu, sigma, dt, draw):
    increment = dt.sqrt() * draw.detach()
    drift = mu * x
    diffusion = sigma * x
    if method == "euler_maruyama":
        return x + drift * dt + diffusion * increment
    predicted = x + drift * dt + diffusion * increment
    corrected_drift = 0.5 * (drift + mu * predicted) if average_drift else drift
    return x + corrected_drift * dt + 0.5 * (diffusion + sigma * predicted) * increment


@pytest.mark.parametrize("method,average_drift", METHODS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_explicit_sde_draws_match_ito_and_stratonovich_stages_and_gradients(
    method, average_drift, dtype
):
    state = _make_state(method, average_drift, dtype=dtype)
    values = _frame(dtype=dtype, requires_grad=True)
    dt = torch.tensor(0.125, dtype=dtype, requires_grad=True)
    draw = torch.linspace(-1.0, 1.0, 6, dtype=dtype).reshape(2, 3).requires_grad_()
    global_rng = torch.random.get_rng_state().clone()
    own_rng = state.x_dW_rng.rng_state()["cpu"].clone()
    originals = {name: value.detach().clone() for name, value in values.items()}
    draw_before = draw.detach().clone()
    actual = state.advance(
        torch.zeros_like(values["x"]), dt, values, draws={"x": draw}
    )["x"]
    expected = _reference(method, average_drift, *values.values(), dt, draw)
    tolerance = 3e-7 if dtype == torch.float32 else 3e-15
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    arguments = (*values.values(), dt, draw)
    gradients = torch.autograd.grad(actual.sum(), arguments, allow_unused=True)
    reference_gradients = torch.autograd.grad(
        expected.sum(), arguments, allow_unused=True
    )
    assert gradients[-1] is None
    assert bool(gradients[2].abs().sum() > 0) and bool(gradients[3].abs().sum() > 0)
    for value, reference in zip(gradients[:-1], reference_gradients[:-1], strict=True):
        torch.testing.assert_close(value, reference, rtol=tolerance, atol=tolerance)
    for name, value in values.items():
        torch.testing.assert_close(value, originals[name], rtol=0, atol=0)
    torch.testing.assert_close(draw, draw_before, rtol=0, atol=0)
    assert torch.equal(global_rng, torch.random.get_rng_state())
    assert torch.equal(own_rng, state.x_dW_rng.rng_state()["cpu"])


@pytest.mark.parametrize("method,average_drift", METHODS)
def test_native_sampling_matches_explicit_draws_rng_and_checkpoint_continuation(
    method, average_drift
):
    native = _make_state(method, average_drift)
    explicit = copy.deepcopy(native)
    values = _frame()
    voltage = torch.zeros_like(values["x"])
    dt = voltage.new_tensor(0.025)
    checkpoint = copy.deepcopy(native.state_dict())
    boundary = copy.deepcopy(values)
    expected = []
    for _ in range(4):
        draw = explicit._sde_randn_like("x", values["x"])
        first = native.advance(voltage, dt, values)["x"]
        second = explicit.advance(voltage, dt, values, draws={"x": draw})["x"]
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert torch.equal(
            native.x_dW_rng.rng_state()["cpu"], explicit.x_dW_rng.rng_state()["cpu"]
        )
        values = {**values, "x": first}
        expected.append(first)
    native.load_state_dict(checkpoint)
    values = boundary
    for reference in expected:
        actual = native.advance(voltage, dt, values)["x"]
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        values = {**values, "x": actual}


@pytest.mark.parametrize("method,average_drift", METHODS)
@pytest.mark.parametrize("dt_shape", [(3,), (2, 1), (2, 3)])
def test_explicit_sde_nonscalar_timesteps_preserve_broadcasting_and_gradients(
    method, average_drift, dt_shape
):
    state = _make_state(method, average_drift)
    values = _frame()
    dt = torch.full(dt_shape, 0.125, dtype=torch.float64, requires_grad=True)
    draw = torch.linspace(-1.0, 1.0, 6, dtype=dt.dtype).reshape(2, 3)
    actual = state.advance(
        torch.zeros_like(values["x"]), dt, values, draws={"x": draw}
    )["x"]
    expected = _reference(method, average_drift, *values.values(), dt, draw)
    torch.testing.assert_close(actual, expected, rtol=3e-15, atol=3e-15)
    actual_gradient = torch.autograd.grad(actual.sum(), dt)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), dt)[0]
    assert actual.shape == (2, 3) and actual_gradient.shape == dt_shape
    torch.testing.assert_close(
        actual_gradient, expected_gradient, rtol=3e-15, atol=3e-15
    )


@pytest.mark.parametrize("method", ["euler_maruyama", "euler_heun"])
def test_native_assigned_and_draw_order_includes_zero_diffusion_states(method):
    events = []

    class Ordered(State):
        State.STATE("z", "x")
        State.ASSIGNED("coefficient")
        State.DERIVATIVE("z' = 0.0", "x' = 0.0")
        State.DIFFUSION("z = 0.0", "x = coefficient")
        State.METHOD(method)

        def assigned_values(self, v, values):
            del v
            events.append(("assigned", float(values["x"][0, 0])))
            return {"coefficient": values["x"]}

    state = Ordered(
        torch.tensor(36.0),
        torch.ones(1, 2),
        torch.arange(2),
        shape=(1, 2),
        shape_f=(1, 2),
    )
    state.populate_parameter_buffers()
    state.init_rng()

    def draw(self, name, like):
        del self
        events.append(("draw", name))
        return torch.ones_like(like)

    state._sde_randn_like = MethodType(draw, state)
    values = {"z": torch.zeros(1, 2), "x": torch.ones(1, 2)}
    state.advance(torch.zeros(1, 2), torch.tensor(0.25), values)
    assert state._sde_draw_state_names == ("z", "x")
    assert state._sde_rng_names == ("z_dW_rng", "x_dW_rng")
    expected = [("assigned", 1.0), ("draw", "z"), ("draw", "x")]
    if method == "euler_heun":
        expected.append(("assigned", 1.5))
    assert events == expected


@pytest.mark.parametrize("method", ["euler_maruyama", "euler_heun"])
def test_generated_solver_excludes_eliminated_state_draws_and_never_samples(method):
    kwargs = dict(
        states=["eliminated", "z", "x"],
        assigned=(),
        derivative=["z' = 0.0", "x' = -x"],
        diffusion=["z = 0.0", "x = 0.25"],
        eliminate={"eliminated": "1 - x"},
    )
    solve = (build_euler_maruyama if method == "euler_maruyama" else build_euler_heun)(
        **kwargs
    )
    owner = SimpleNamespace(_derive_assigned_values=lambda v, values: {})
    values = {
        name: torch.full((1, 2), value)
        for name, value in (("eliminated", 0.25), ("z", 0.0), ("x", 0.75))
    }
    draws = {"z": torch.full((1, 2), 9.0), "x": torch.ones(1, 2)}
    dt = torch.tensor(0.25)
    if method == "euler_maruyama":
        actual = solve(owner, dt, **values, __draws=draws)
    else:
        actual = solve(owner, torch.zeros(1, 2), dt, values, draws, {})
    torch.testing.assert_close(actual["eliminated"], 1 - values["x"], rtol=0, atol=0)
    torch.testing.assert_close(actual["z"], values["z"], rtol=0, atol=0)
    assert set(actual) == set(values)


@pytest.mark.parametrize("method", ["euler_maruyama", "euler_heun"])
def test_kinetic_elimination_preserves_rng_modules_but_does_not_consume_their_draws(
    method,
):
    class Conserved(State):
        State.STATE("closed", "open")
        State.GLOBAL(alpha=0.1, beta=0.2)
        State.KINETIC("~ closed <-> open (alpha, beta)", "CONSERVE closed + open = 1")
        State.DIFFUSION("closed = 0.1", "open = 0.1")
        State.METHOD(method)

    state = Conserved(
        torch.tensor(36.0),
        torch.ones(1, 2),
        torch.arange(2),
        shape=(1, 2),
        shape_f=(1, 2),
    )
    state.populate_parameter_buffers()
    state.init_rng()
    assert state._sde_draw_state_names == ("closed",)
    assert state._sde_rng_names == ("closed_dW_rng", "open_dW_rng")
    before = {
        name: getattr(state, name).rng_state()["cpu"].clone()
        for name in state._sde_rng_names
    }
    values = {"closed": torch.full((1, 2), 0.75), "open": torch.full((1, 2), 0.25)}
    generator = torch.Generator().set_state(before["closed_dW_rng"])
    draw = torch.randn((1, 2), generator=generator)
    dt = torch.tensor(0.1)
    actual = state.advance(torch.zeros(1, 2), dt, values)
    drift = state.beta * values["open"] - state.alpha * values["closed"]
    expected = values["closed"] + drift * dt + 0.1 * dt.sqrt() * draw
    torch.testing.assert_close(actual["closed"], expected)
    torch.testing.assert_close(actual["open"], 1 - values["closed"], rtol=0, atol=0)
    assert torch.equal(generator.get_state(), state.closed_dW_rng.rng_state()["cpu"])
    assert torch.equal(before["open_dW_rng"], state.open_dW_rng.rng_state()["cpu"])
    boundary = state.closed_dW_rng.rng_state()["cpu"].clone()
    zero_draw = state.advance(
        torch.zeros(1, 2), dt, values, draws={"closed": torch.zeros(1, 2)}
    )
    torch.testing.assert_close(zero_draw["closed"], values["closed"] + drift * dt)
    assert torch.equal(boundary, state.closed_dW_rng.rng_state()["cpu"])


@pytest.mark.parametrize("method,average_drift", METHODS)
def test_explicit_sde_evaluation_composes_with_vmap_jvp_and_fullgraph(
    method, average_drift
):
    state = _make_state(method, average_drift, shape=(3,))
    x = torch.tensor([[0.25, 0.5, 0.75], [0.3, 0.6, 0.9]], dtype=torch.float64)
    mu = torch.full_like(x, 0.2)
    sigma = torch.full_like(x, 0.4)
    draw = torch.linspace(-1.0, 1.0, 6, dtype=x.dtype).reshape_as(x)
    dt = x.new_tensor(0.125)

    def response(x, mu, sigma, dt, draw):
        return state.advance(
            torch.zeros_like(x),
            dt,
            {"x": x, "mu": mu, "sigma": sigma},
            draws={"x": draw},
        )["x"]

    expected = _reference(method, average_drift, x, mu, sigma, dt, draw)
    actual = torch.vmap(response, in_dims=(0, 0, 0, None, 0))(x, mu, sigma, dt, draw)
    torch.testing.assert_close(actual, expected)
    empty = torch.vmap(response)(x[:0], mu[:0], sigma[:0], dt.expand(0), draw[:0])
    assert empty.shape == (0, 3) and empty.dtype == x.dtype
    arguments = (x, mu, sigma, dt, draw)
    directions = tuple(torch.ones_like(value) for value in arguments)
    _, actual_tangent = torch.func.jvp(response, arguments, directions)
    _, reference_tangent = torch.func.jvp(
        lambda *args: _reference(method, average_drift, *args), arguments, directions
    )
    torch.testing.assert_close(actual_tangent, reference_tangent)
    compiled = torch.compile(response, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(*arguments), expected)


@pytest.mark.parametrize(
    "malformed",
    [
        None,
        {},
        {"other": torch.zeros(2, 3)},
        {"x": torch.zeros(1, 3)},
        {"x": torch.zeros(2, 3, dtype=torch.float32)},
        {"x": 0.0},
    ],
)
def test_explicit_sde_draw_schema_rejects_malformed_packets_without_rng(malformed):
    state = _make_state("euler_maruyama")
    values = _frame()
    before = state.x_dW_rng.rng_state()["cpu"].clone()
    if malformed is None:
        malformed = []
    with pytest.raises((TypeError, ValueError), match="State SDE draw"):
        state.advance(
            torch.zeros_like(values["x"]), torch.tensor(0.1), values, draws=malformed
        )
    assert torch.equal(before, state.x_dW_rng.rng_state()["cpu"])


def test_shared_host_sampler_detaches_even_a_custom_existing_tensor():
    value = torch.ones(2, 3, requires_grad=True)
    rng = SimpleNamespace(randn=lambda shape, *, device, dtype: value)
    actual = sample_state_sde_draw(
        rng, value.shape, device=value.device, dtype=value.dtype
    )
    assert not actual.requires_grad
    torch.testing.assert_close(actual, value, rtol=0, atol=0)


def test_legacy_custom_advance_override_keeps_original_three_argument_signature():
    class Custom(State):
        State.STATE("x")
        State.DERIVATIVE("x' = 0.0")

        def advance(self, v, dt, values):
            del v
            return {"x": values["x"] + dt}

    state = Custom(
        torch.tensor(36.0),
        torch.ones(1, 2),
        torch.arange(2),
        shape=(1, 2),
        shape_f=(1, 2),
    )
    voltage = torch.zeros(1, 2)
    actual = state._derive_advance_values(voltage, 0.25, {"x": voltage})
    torch.testing.assert_close(actual["x"], voltage + 0.25, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["euler_maruyama", "euler_heun"])
def test_native_mechanism_uses_shared_state_draw_adapter(method):
    class Additive(State):
        State.STATE("x")
        State.DERIVATIVE("x' = 0.0")
        State.DIFFUSION("x = 1.0")
        State.METHOD(method)

    class Diffusive(Mechanism):
        Mechanism.STATE_BUNDLE(Additive)

    with torch.no_grad(), dn.ctx(JIT=0, REQUIRE_GRAD=0):
        model = dn.SingleCompartment(N=1, C=2, dtype=torch.float64)
        model.insert(Diffusive.rename("diffusive"))
        model.initialize()
        state = model.mech.diffusive.DE["Additive"]
        state.x_dW_rng.randn = lambda shape, *, device, dtype: torch.ones(
            shape, device=device, dtype=dtype
        )
        before = model.mech.diffusive.x.clone()
        model.step(0.25)
    torch.testing.assert_close(model.mech.diffusive.x, before + 0.5, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["euler_maruyama", "euler_heun"])
def test_native_jit_preserves_sde_trajectory_rng_and_checkpoint_continuation(method):
    from torch.utils import _pytree as pytree

    from tests.func.test_state_sde_execution import (
        DT,
        _HeunLeak,
        _OULeak,
        _population,
    )

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=0, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"
            ),
        ):
            mechanism = _OULeak if method == "euler_maruyama" else _HeunLeak
            reference = _population(mechanism).eval()
            candidate = _population(mechanism).eval()
            candidate.restore_dict_from_checkpoint(
                reference.state_dict_for_checkpoint()
            )

            for _ in range(2):
                recordings = []
                for population, jit in ((reference, 0), (candidate, 1)):
                    recorder = dn.Recorder(["v", "sde.x", "t"])
                    # Native generators are sampled outside captured graph
                    # segments; FULLGRAPH=0 is the default Native JIT contract.
                    with dn.ctx(JIT=jit, BACKEND="eager", FULLGRAPH=0):
                        population.run(tstop=3 * DT, dt=DT, callbacks=[recorder])
                    assert bool(population.integrator._compiled_kernels) == bool(jit)
                    recordings.append(recorder)
                for name in ("v", "sde.x", "t"):
                    torch.testing.assert_close(
                        recordings[1].stack(name),
                        recordings[0].stack(name),
                        rtol=0,
                        atol=0,
                    )
                expected, expected_schema = pytree.tree_flatten(
                    reference.mech.mutable_state_dict()
                )
                actual, actual_schema = pytree.tree_flatten(
                    candidate.mech.mutable_state_dict()
                )
                assert actual_schema == expected_schema
                for value, target in zip(actual, expected, strict=True):
                    if torch.is_tensor(target):
                        torch.testing.assert_close(value, target, rtol=0, atol=0)
                    else:
                        assert value == target
                # Continue after restoring the compiled path's boundary into a
                # fresh object, including every State RNG stream position.
                restored = _population(mechanism).eval()
                restored.restore_dict_from_checkpoint(
                    candidate.state_dict_for_checkpoint()
                )
                candidate = restored
    finally:
        torch.set_num_threads(previous_threads)
