from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import hh

DT = 0.01
GNABAR = "integrator.mech.mechanisms.hh.gnabar_param"

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


def _model(*, method="pcr"):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [2.0, 2.5],
            L=4.0,
            dx=1.0,
            v_init=torch.tensor([-64.0, -60.0, -56.0, -59.0, -63.0]),
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method=method, imem=False),
        )
        model.insert(hh)
        model.initialize()
        model.train()
    return model


def _drives(model, steps):
    count = steps * model.v.numel()
    ve = torch.linspace(
        -2.0,
        2.0,
        count,
        dtype=model.dtype(),
        device=model.device(),
    ).reshape(steps, *model.shape)
    intra = torch.linspace(
        -1.0e-9,
        1.0e-9,
        count,
        dtype=model.dtype(),
        device=model.device(),
    ).reshape(steps, *model.shape)
    return ve, intra


def _clone_mapping(values: Mapping[str, torch.Tensor]):
    return {name: value.detach().clone() for name, value in values.items()}


def _clone_state(state):
    return torch.utils._pytree.tree_map(lambda value: value.detach().clone(), state)


def _differentiable_case(tensors, ve, intra):
    parameters = _clone_mapping(tensors.parameters)
    parameters[GNABAR].requires_grad_()

    constants = _clone_mapping(tensors.constants)
    constants["diam"].requires_grad_()
    constants["dx"].requires_grad_()

    state = _clone_state(tensors.state)
    state["integrator"]["v"].requires_grad_()
    ve = ve.detach().clone().requires_grad_()
    intra = intra.detach().clone().requires_grad_()

    targets = {
        "parameter": parameters[GNABAR],
        "diam": constants["diam"],
        "dx": constants["dx"],
        "initial_voltage": state["integrator"]["v"],
        "ve": ve,
        "intra": intra,
    }
    return parameters, constants, state, ve, intra, targets


def _build_schedule(functional, *, steps, chunk_steps, backend):
    full_chunks, tail_steps = divmod(steps, chunk_steps)
    chunk = functional.compile_rollout_chunk(chunk_steps, backend=backend)
    tail = (
        functional.compile_rollout_chunk(tail_steps, backend=backend)
        if tail_steps
        else None
    )
    return chunk, tail, full_chunks


def _run_schedule(
    schedule,
    parameters,
    prepared,
    state,
    ve,
    intra,
    *,
    checkpointed,
):
    chunk, tail, full_chunks = schedule

    def advance(active_chunk, current_state, start, stop):
        chunk_ve = ve[start:stop]
        chunk_intra = intra[start:stop]
        if not checkpointed:
            return active_chunk(
                parameters,
                prepared,
                current_state,
                dn.func.RolloutInput(ve=chunk_ve, intra=chunk_intra),
            )[0]

        def checkpoint_body(replay_state, replay_ve, replay_intra):
            return active_chunk(
                parameters,
                prepared,
                replay_state,
                dn.func.RolloutInput(ve=replay_ve, intra=replay_intra),
            )[0]

        return checkpoint(
            checkpoint_body,
            current_state,
            chunk_ve,
            chunk_intra,
            use_reentrant=False,
        )

    cursor = 0
    for _ in range(full_chunks):
        stop = cursor + chunk.steps
        state = advance(chunk, state, cursor, stop)
        cursor = stop
    if tail is not None:
        state = advance(tail, state, cursor, ve.shape[0])
    return state


def _loss(state):
    hh_state = state["mechanisms"]["hh"]
    return (
        state["integrator"]["v"].square().mean()
        + 0.1 * hh_state["m"].square().mean()
        + 0.01 * hh_state["h"].square().mean()
        + 0.001 * hh_state["n"].square().mean()
    )


def _execute(
    functional,
    tensors,
    schedule,
    ve,
    intra,
    *,
    checkpointed,
):
    parameters, constants, state, ve, intra, targets = _differentiable_case(
        tensors,
        ve,
        intra,
    )
    prepared = functional.prepare(parameters, constants)
    final = _run_schedule(
        schedule,
        parameters,
        prepared,
        state,
        ve,
        intra,
        checkpointed=checkpointed,
    )
    loss = _loss(final)
    gradients = dict(
        zip(
            targets,
            torch.autograd.grad(loss, tuple(targets.values())),
            strict=True,
        )
    )
    return final, loss, gradients


def _assert_tree_close(actual, expected, *, rtol=0.0, atol=0.0):
    actual_with_paths, actual_spec = torch.utils._pytree.tree_flatten_with_path(actual)
    expected_with_paths, expected_spec = torch.utils._pytree.tree_flatten_with_path(
        expected
    )
    assert actual_spec == expected_spec
    for (actual_path, actual_leaf), (expected_path, expected_leaf) in zip(
        actual_with_paths,
        expected_with_paths,
        strict=True,
    ):
        assert actual_path == expected_path
        torch.testing.assert_close(
            actual_leaf,
            expected_leaf,
            rtol=rtol,
            atol=atol,
            msg=lambda message: f"state leaf {actual_path}: {message}",
        )


@pytest.mark.parametrize("method", ["thomas", "pcr"])
@pytest.mark.parametrize(("steps", "chunk_steps"), [(4, 2), (5, 2)])
def test_user_checkpointed_compiled_chunks_match_ordinary_bptt(
    method,
    steps,
    chunk_steps,
    monkeypatch,
):
    model = _model(method=method)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, steps)
    schedule = _build_schedule(
        functional,
        steps=steps,
        chunk_steps=chunk_steps,
        backend="aot_eager",
    )

    preparation_calls = 0
    original_prepare_values = functional._prepare_values

    def counted_prepare_values(parameters, constants):
        nonlocal preparation_calls
        preparation_calls += 1
        return original_prepare_values(parameters, constants)

    monkeypatch.setattr(functional, "_prepare_values", counted_prepare_values)
    with torch_compiler_warning_context():
        ordinary = _execute(
            functional,
            tensors,
            schedule,
            ve,
            intra,
            checkpointed=False,
        )
        checkpointed = _execute(
            functional,
            tensors,
            schedule,
            ve,
            intra,
            checkpointed=True,
        )

    ordinary_final, ordinary_loss, ordinary_gradients = ordinary
    checkpointed_final, checkpointed_loss, checkpointed_gradients = checkpointed

    # Each forward prepares exactly once. Backward checkpoint replay reuses the
    # captured tensor workspace instead of rebuilding geometry/mechanism data.
    assert preparation_calls == 2
    _assert_tree_close(
        checkpointed_final,
        ordinary_final,
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    torch.testing.assert_close(
        checkpointed_loss,
        ordinary_loss,
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    assert checkpointed_gradients.keys() == ordinary_gradients.keys()
    for name in checkpointed_gradients:
        actual = checkpointed_gradients[name]
        expected = ordinary_gradients[name]
        assert torch.isfinite(actual).all(), name
        assert torch.count_nonzero(actual) > 0, name
        torch.testing.assert_close(
            actual,
            expected,
            rtol=2.0e-8,
            atol=2.0e-10,
            msg=lambda message: f"gradient {name}: {message}",
        )


def _saved_tensor_summary(fn):
    count = 0
    total_bytes = 0

    def pack(tensor):
        nonlocal count, total_bytes
        count += 1
        total_bytes += tensor.numel() * tensor.element_size()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        fn()
    return count, total_bytes


def test_user_checkpointing_meaningfully_reduces_saved_tensor_storage():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 12)
    schedule = _build_schedule(
        functional,
        steps=12,
        chunk_steps=3,
        backend="eager",
    )

    def run(*, checkpointed):
        return _execute(
            functional,
            tensors,
            schedule,
            ve,
            intra,
            checkpointed=checkpointed,
        )

    with torch_compiler_warning_context():
        ordinary_count, ordinary_bytes = _saved_tensor_summary(
            lambda: run(checkpointed=False)
        )
        checkpointed_count, checkpointed_bytes = _saved_tensor_summary(
            lambda: run(checkpointed=True)
        )

    assert checkpointed_count < ordinary_count / 2
    assert checkpointed_bytes < ordinary_bytes / 2
