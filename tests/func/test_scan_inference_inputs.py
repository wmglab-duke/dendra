"""Blocked Population inference owns recurrent leaves before backend guards."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_lowering_api import DT, _close, _population
from tests.func.test_poisson_invocation_execution import _case
from tests.func.test_poisson_stimulation import _close as _poisson_close

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
    ):
        yield


def _clone(tree):
    return pytree.tree_map_only(torch.Tensor, lambda value: value.clone(), tree)


def _leaves(tree):
    return [value for value in pytree.tree_leaves(tree) if torch.is_tensor(value)]


def _offset(value):
    storage = value.new_empty((2 * value.numel() + 3,))
    selected = storage[1 : 1 + 2 * value.numel() : 2].view(value.shape)
    selected.copy_(value)
    return selected


def test_blocked_inference_backend_reuses_owned_state_across_aliases_and_views():
    graphs = []

    def backend(graph, _examples):
        graphs.append(graph)
        return graph.forward

    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    parameters = tensors.independent_parameters(trainable=())
    kernel = functional.compile_rollout_chunk(
        4, execution="scan", backend=backend, fullgraph=True
    )
    with torch.no_grad():
        prepared = functional.prepare(parameters, tensors.constants)
        previous = []
        count = None
        for kind in ("ordinary", "aliased", "offset", "inference"):
            if kind == "inference":
                with torch.inference_mode():
                    state = _clone(tensors.state)
            else:
                state = _clone(tensors.state)
                if kind == "aliased":
                    state["control"]["duration_remainder"] = state["clock"]["t"]
                    assert state["control"]["duration_remainder"] is state["clock"]["t"]
                elif kind == "offset":
                    state = pytree.tree_map_only(torch.Tensor, _offset, state)
            frozen = _clone(state)
            expected, _ = functional.prepare_and_rollout(
                parameters, tensors.constants, state, steps=4
            )
            actual, _ = kernel(parameters, prepared, state)
            _close(actual, expected)
            _close(state, frozen)
            for value in _leaves(actual):
                assert all(
                    not torch._C._is_alias_of(value, original)
                    for original in _leaves((state, previous))
                    if value.numel() and original.numel()
                )
            previous.append(actual)
            if count is None:
                count = len(graphs)
                assert count > 0
            assert len(graphs) == count, kind
    assert len(kernel._compiled._kernels) == 1
    assert kernel.execution_report().strategy == "compiled_scan"


def test_poisson_inference_backend_reuses_raw_cloned_and_restored_continuations(
    tmp_path,
):
    graphs = []

    def backend(graph, _examples):
        graphs.append(graph)
        return graph.forward

    source, _, _, generator = _case("repeated", "mechanism", n=4)
    native, _, _, _ = _case("repeated", "mechanism", n=4)
    source_rng = generator.get_state().clone()
    program = dn.compile(
        source,
        dt=DT,
        steps=4,
        initialization="fresh",
        execution="scan",
        inference=True,
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        backend=backend,
    )
    with torch.no_grad():
        prefix, first = program()
        count = len(graphs)
        assert count > 0
        raw, second = program(prefix, callback_state=first["callbacks"].state)
        assert len(graphs) == count
        cloned, cloned_auxiliary = program(
            _clone(prefix), callback_state=first["callbacks"].state
        )
        _poisson_close(raw, cloned)
        _poisson_close(
            second["callbacks"]["trace"], cloned_auxiliary["callbacks"]["trace"]
        )
        assert len(graphs) == count
        path = tmp_path / "scan-continuation.pt"
        torch.save(program.export_continuation(first["continuation"]), path)
        restored = program.restore_continuation(torch.load(path, weights_only=True))
        portable, portable_auxiliary = program(restored)
        _poisson_close(raw, portable)
        _poisson_close(
            second["callbacks"]["trace"], portable_auxiliary["callbacks"]["trace"]
        )
        assert len(graphs) == count
        final, third = program(second["continuation"])
        assert len(graphs) == count

        native.initialize()
        expected = [native.v.clone()]
        for index in range(12):
            native.step(DT)
            expected.append(native.v.clone())
            if index == 3:
                _poisson_close(prefix, program.functional.extract(native).state)
            elif index == 7:
                _poisson_close(raw, program.functional.extract(native).state)
        _poisson_close(final, program.functional.extract(native).state)
        observed = torch.cat(
            (
                first["callbacks"]["trace"]["v"],
                second["callbacks"]["trace"]["v"],
                third["callbacks"]["trace"]["v"],
            )
        )
        _poisson_close(observed, torch.stack(expected))
        _poisson_close(generator.get_state(), source_rng)
    assert program.execution_report().strategy == "compiled_scan"
