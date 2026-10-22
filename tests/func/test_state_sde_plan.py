"""Ordered SDE tapes share NOISE streams and contain detached Gaussian bases."""

import copy

import pytest
import torch
from torch.utils import _pytree

import dendra as dn
from dendra.func._random_parameters import RandomParameterPlan
from dendra.func._runtime_noise import RuntimeNoisePlan, stream_inventory
from dendra.func._state_sde import state_sde_failures
from dendra.models.mechanisms import Mechanism, State
from dendra.models.rng import RNGModule


class _SharedDrive(State):
    State.STATE("x", "y")
    State.RANGENOISE("eta", rng_name="x_dW_rng", seed=31)
    State.DERIVATIVE("x' = eta - x", "y' = -y")
    State.DIFFUSION("x = 0.2", "y = 0.0")
    State.METHOD("euler_maruyama")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.zeros_like(v), "y": torch.ones_like(v)}


class _SDEOnly(State):
    State.STATE("x")
    State.DERIVATIVE("x' = -x")
    State.DIFFUSION("x = 0.2")
    State.METHOD("euler_maruyama")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.ones_like(v)}


class _Conserved(State):
    State.STATE("closed", "open")
    State.RANGE(alpha=0.2, beta=0.1)
    State.KINETIC("~ closed <-> open (alpha, beta)", "CONSERVE closed + open = 1")
    State.DIFFUSION("closed = 0.1", "open = 0.2")
    State.METHOD("euler_maruyama")

    def state_defaults(self, v, values):
        del values
        return {"closed": torch.ones_like(v), "open": torch.zeros_like(v)}


def _population(state_type=_SharedDrive):
    class ModelMechanism(Mechanism):
        Mechanism.STATE_BUNDLE(state_type)
        Mechanism.RANGE(g=0.001, e=-65.0)
        Mechanism.NONSPECIFIC_CURRENT("i")
        Mechanism.AFFINE("i")

        def i_affine_terms(self, v):
            del v
            return self.g, self.g * self.e

    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        population = dn.SingleCompartment(C=3, dtype=torch.float64)
        population.insert(ModelMechanism.rename("drive"))
        population.initialize()
    state = next(iter(population.mech.mechanisms["drive"].DE.values()))
    for index, name in enumerate(state._sde_rng_names):
        if name != "x_dW_rng" or state_type is not _SharedDrive:
            getattr(state, name).reseed(101 + index)
    return population, state


def _plans(population):
    noise = RuntimeNoisePlan(population)
    random = RandomParameterPlan(population)
    random.noise = noise
    _, state = random.extract()
    return noise, random, {"random": state}


def _close(actual, expected):
    left, spec = _pytree.tree_flatten(actual)
    right, other = _pytree.tree_flatten(expected)
    assert spec == other
    for value, reference in zip(left, right, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


def test_combined_noise_sde_packet_preserves_native_draw_order_and_shared_rng():
    population, state = _population()
    noise, random, carry = _plans(population)
    saved = copy.deepcopy(carry)
    native = {
        key: RNGModule.from_checkpoint(
            getattr(population.get_submodule(path), name).get_extra_state()
        )
        for key, (path, name) in random.streams.items()
    }
    expected = {}
    for entry in noise.noise_runtime:
        expected[entry.draw_names[0]] = native[entry.stream].randn(
            entry.shape, device=entry.device, dtype=entry.dtype
        )
    for entry in noise.sde.entries:
        expected[entry.key] = native[entry.stream].randn(
            entry.shape, device=entry.device, dtype=entry.dtype
        )
    packet = noise.sample(carry)
    noise.validate(packet)
    _close(packet.draws, expected)
    _close(carry, saved)
    assert len(random.streams) == 2
    assert noise.noise_runtime[0].stream == noise.sde.entries[0].stream
    assert tuple(entry.name for entry in noise.sde.entries) == ("x", "y")
    assert state._sde_draw_state_names == ("x", "y")
    assert all(not value.requires_grad for value in packet.draws.values())
    initialization = random.sample(carry)
    assert all(not name.startswith("sde.draws.") for name in initialization.draws)
    assert set(initialization.rng) == set(packet.rng)


def test_sde_only_and_zero_length_tapes_retain_rng_without_initialization_draws():
    population, _state = _population(_SDEOnly)
    noise, random, carry = _plans(population)
    assert not noise.entries and noise.runtime
    assert not random.sample(carry).draws
    packet = noise.sample(carry, steps=0)
    noise.validate(packet, steps=0)
    assert all(value.shape[0] == 0 for value in packet.draws.values())
    assert all(value.shape[0] == 0 for value in _pytree.tree_leaves(packet.rng))
    first = noise.sample(carry, steps=2)
    second = noise.sample(carry)
    _close(_pytree.tree_map(lambda value: value[0], first.draws), second.draws)


def test_conserved_sde_fields_keep_eliminated_rng_but_do_not_consume_it():
    population, state = _population(_Conserved)
    noise, random, carry = _plans(population)
    assert state._sde_draw_state_names == ("closed",)
    assert len(random.streams) == 2
    packet = noise.sample(carry)
    used = noise.sde.entries[0].stream
    for stream in set(packet.rng) - {used}:
        _close(packet.rng[stream], carry["random"]["rng"][stream])
    assert len(packet.draws) == 1


@pytest.mark.parametrize("bad", ("names", "shape", "dtype"))
def test_sde_packet_schema_rejects_incompatible_explicit_draws(bad):
    population, _state = _population(_SDEOnly)
    noise, _random, carry = _plans(population)
    packet = noise.sample(carry)
    name = next(iter(packet.draws))
    if bad == "names":
        draws = {}
    elif bad == "shape":
        draws = {name: packet.draws[name][..., :1]}
    else:
        draws = {name: packet.draws[name].float()}
    with pytest.raises(ValueError, match="names|schema"):
        noise.validate(packet._replace(draws=draws))


def test_sde_source_adapter_and_stream_alias_changes_are_rejected():
    population, state = _population()
    noise, _random, _carry = _plans(population)
    state.x_dW_rng = RNGModule(28, shape_f=state.shape_f, shape_p=state.shape_p)
    with pytest.raises(dn.func.FunctionalizationError, match="sharing|binding"):
        noise.validate_source()
    state.advance = lambda v, dt, values: values
    assert any("advance" in failure for failure in state_sde_failures(state))


def test_sde_transient_draws_are_detached_without_detaching_solver_coefficients():
    population, _state = _population(_SDEOnly)
    noise, _random, carry = _plans(population)
    packet = noise.sample(carry)
    packet = packet._replace(
        draws={name: value.requires_grad_() for name, value in packet.draws.items()}
    )
    assert all(not value.requires_grad for value in noise.sde_values(packet).values())
    from dendra.func._population import _clone_execution_population

    owned = _clone_execution_population(population)
    noise.install(owned)
    field = next(iter(owned.mech.mechanisms["drive"].DE.values()))
    assert tuple(field._functional_sde_slots) == (("x", "_functional_sde_draw_x"),)
    assert field._buffers["_functional_sde_draw_x"].eq(1).all()
    assert not hasattr(
        next(iter(population.mech.mechanisms["drive"].DE.values())),
        "_functional_sde_slots",
    )


def test_sde_stream_inventory_is_stable_for_shared_noise_aliases():
    population, _state = _population()
    streams, identities = stream_inventory(population)
    assert len(streams) == len(identities) == 2


def test_shared_sde_stream_follows_pre_and_post_current_phase_order():
    class Reader(Mechanism):
        Mechanism.STATE_BUNDLE(_SDEOnly)
        Mechanism.USEION("na", read=["ina"])

    class Producer(Mechanism):
        Mechanism.STATE_BUNDLE(_SDEOnly)
        Mechanism.RANGE(g=0.001, ena=-60.0)
        Mechanism.USEION("na", write=["ina"])
        Mechanism.AFFINE("ina")

        def ina(self, v):
            return self.g * (v - self.ena)

    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        population = dn.SingleCompartment(C=2, dtype=torch.float64)
        population.insert(Reader.rename("post"))
        population.insert(Producer.rename("pre"))
        population.initialize()
    pre = next(iter(population.mech.mechanisms["pre"].DE.values()))
    post = next(iter(population.mech.mechanisms["post"].DE.values()))
    pre.x_dW_rng.reseed(51)
    post.x_dW_rng = pre.x_dW_rng
    noise, _random, carry = _plans(population)
    assert len(carry["random"]["rng"]) == 1
    assert ".mechanisms.pre." in noise.sde.entries[0].path
    assert ".mechanisms.post." in noise.sde.entries[1].path
    packet = noise.sample(carry)
    rng = RNGModule.from_checkpoint(pre.x_dW_rng.get_extra_state())
    for entry in noise.sde.entries:
        expected = rng.randn(entry.shape, dtype=entry.dtype, device=entry.device)
        torch.testing.assert_close(packet.draws[entry.key], expected, rtol=0, atol=0)


@pytest.mark.parametrize("helper", ("state_sde_draw", "state_sde_dt"))
def test_sde_generated_solver_rejects_replaced_shared_pure_helper(helper):
    population, state = _population(_SDEOnly)
    from types import FunctionType, MethodType

    original = state._solve.__func__
    replacement = FunctionType(
        original.__code__,
        {**original.__globals__, helper: lambda *args: args[-1]},
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    replacement.__dict__.update(original.__dict__)
    state._solve = MethodType(replacement, state)
    assert any(
        "canonical generated" in failure for failure in state_sde_failures(state)
    )


@pytest.mark.parametrize("method", ("euler_maruyama", "euler_heun"))
def test_sde_generated_equations_cannot_mutate_ephemeral_state_inputs(method):
    class MutatingInput(State):
        State.STATE("x")
        State.INPUT("drive", default=0.5)
        State.DERIVATIVE("x' = bump(drive) - x")
        State.DIFFUSION("x = 0.1")
        State.METHOD(method)

        def bump(self, value):
            return value.add_(1.0)

        def state_defaults(self, v, values):
            del values
            return {"x": torch.zeros_like(v)}

    population, _state = _population(MutatingInput)
    with pytest.raises(dn.func.FunctionalizationError, match="read-only INPUT"):
        dn.func.make_functional(population, dt=0.125)


@pytest.mark.parametrize("method", ("euler_maruyama", "euler_heun"))
def test_read_only_sde_inputs_remain_available_to_generated_drift_and_diffusion(method):
    class DrivenInput(State):
        State.STATE("x")
        State.INPUT("drive", default=0.5)
        State.DERIVATIVE("x' = drive - x")
        State.DIFFUSION("x = 0.1 * drive")
        State.METHOD(method)

        def state_defaults(self, v, values):
            del values
            return {"x": torch.zeros_like(v)}

    population, _state = _population(DrivenInput)
    functional, tensors = dn.func.make_functional(population, dt=0.125)
    packet = functional.sample_noise_draw(tensors.state)
    packet = packet._replace(
        draws={
            name: torch.full_like(value, 2.0) for name, value in packet.draws.items()
        }
    )
    final, _auxiliary = functional.prepare_and_step(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        dn.func.StepInput(noise=packet),
    )
    value = final["mechanisms"]["drive"]["x"]
    expected = torch.full_like(value, 0.5 * 0.125 + 0.05 * 0.125**0.5 * 2.0)
    torch.testing.assert_close(value, expected)
    assert population.mech.mechanisms["drive"].x.eq(0).all()
