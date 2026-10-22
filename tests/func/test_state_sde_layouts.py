"""State SDE packets follow native batched, regional, and packed field layouts."""

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from dendra.models.multi import MultiPopulation
from tests.func.test_state_sde_execution import (
    DT,
    DTYPE,
    _HeunLeak,
    _OULeak,
    _population,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _clone(value):
    return pytree.tree_map_only(
        torch.Tensor, lambda tensor: tensor.detach().clone(), value
    )


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)


def _build(layout, mechanism):
    if layout == "batch":
        population = _population(mechanism)
        population.batch(3)
    elif layout == "regional":
        population = dn.SingleCompartment(
            N=2, C=3, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        population.insert(pas)
        population[:, :2].insert(mechanism.rename("sde"))
    else:
        population = dn.concat_models(
            {"left": _population(mechanism), "right": _population(mechanism, seed=271)},
            threads=1,
        )
    population.train()
    population.initialize()
    offset = 0
    for parent in population.mech.mechanisms.values():
        for state in parent.DE.values():
            for name in state._sde_rng_names:
                getattr(state, name).reseed(171 + offset)
                offset += 1
    population.integrator._initialize(
        population, population.t.new_tensor(DT), force=True
    )
    return population


@pytest.mark.parametrize("mechanism", (_OULeak, _HeunLeak))
@pytest.mark.parametrize("layout", ("batch", "regional", "packed"))
def test_sde_layouts_match_native_step_scan_rng_and_zero_length_packets(
    layout, mechanism
):
    source, native = _build(layout, mechanism), _build(layout, mechanism)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    before = _clone((tensors, functional.extract()))
    global_rng = torch.get_rng_state().clone()
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    tape = functional.sample_noise_draw_tape(tensors.state, 3)
    saved_tape = _clone(tape)
    expected_shapes = sorted(
        tuple(parent._buffers[name].shape)
        for parent in native.mech.mechanisms.values()
        for state in parent.DE.values()
        for name in state._sde_draw_state_names
    )
    assert (
        sorted(tuple(value.shape[1:]) for value in tape.draws.values())
        == expected_shapes
    )
    if layout == "batch":
        assert all(shape[0] == 3 for shape in expected_shapes)
    elif layout == "regional":
        assert expected_shapes == [(2, 2)]
        assert source.v.shape == (2, 3)
    else:
        assert isinstance(source, MultiPopulation)
        assert set(source.populations) == {"left", "right"}

    empty = functional.sample_noise_draw_tape(tensors.state, 0)
    unchanged, _ = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(noise=empty),
        steps=0,
    )
    _close(unchanged, tensors.state)

    first = pytree.tree_map_only(torch.Tensor, lambda value: value[0], tape)
    stepped, _ = functional.step(
        tensors.parameters, prepared, tensors.state, dn.func.StepInput(noise=first)
    )
    with torch.no_grad():
        native.step(DT)
    _close(stepped, functional.extract(native).state)

    executable = functional.compile_rollout_chunk(3, execution="scan", backend="eager")
    final, _ = dn.func.run(
        executable.bind(tensors.parameters, prepared),
        tensors.state,
        steps=3,
        inputs=dn.func.RolloutInput(noise=tape),
    )
    with torch.no_grad():
        for _ in range(2):
            native.step(DT)
    _close(final, functional.extract(native).state)
    _close((tensors, functional.extract()), before)
    _close(tape, saved_tape)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
