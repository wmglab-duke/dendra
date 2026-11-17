"""The private scan carry crosses host chunks without crossing public boundaries."""

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_network import _RAW_WEIGHT, DT, _clone, _close, _stim_network

pytestmark = pytest.mark.cpu


def _voltage(state):
    return state["populations"]["cell"]["integrator"]["v"]


@pytest.mark.parametrize("checkpointed", (False, True))
def test_host_scan_retains_packed_carry_and_publishes_owned_resume(
    monkeypatch, checkpointed
):
    model = _stim_network()
    functional, tensors = dn.func.make_functional_network(model)
    steps, split, width = 13, 9, 3
    state = _clone(tensors.state, detached=True)
    initial_v = _voltage(state).requires_grad_()
    origin = state["clock"]["origin"].requires_grad_()
    input_current = (
        torch.linspace(0, 1e-7, steps * initial_v.numel(), dtype=initial_v.dtype)
        .reshape(steps, *initial_v.shape)
        .requires_grad_()
    )
    draws = functional.sample_runtime_draws(state, steps=steps)
    callbacks = functional.make_callbacks(
        {"trace": dn.func.Recorder(("cell.v", "t"), dt=2 * DT)}
    )
    before = _clone(state, detached=True)
    before_draws = _clone(draws, detached=True)
    (projection_name,) = tensors.parameters["synapses"]
    weight = tensors.parameters["synapses"][projection_name][_RAW_WEIGHT]
    targets = (initial_v, origin, weight, input_current)
    versions = tuple(target._version for target in targets)

    def packets(start, stop):
        return (
            dn.func.NetworkRolloutInput(
                populations={
                    "cell": dn.func.RolloutInput(intra=input_current[start:stop])
                },
            ),
            pytree.tree_map_only(torch.Tensor, lambda value: value[start:stop], draws),
        )

    chunk = functional.compile_rollout_chunk(
        width,
        execution="scan",
        backend="inductor",
        options={"compile_threads": 1, "cpp.threads": 1},
    )

    def run(start, stop, initial, continuation=None):
        inputs, packet = packets(start, stop)
        return functional.run(
            tensors.parameters,
            tensors.constants,
            initial,
            inputs,
            steps=stop - start,
            draws=packet,
            chunklength=width,
            checkpoint_every=width if checkpointed else None,
            callbacks=callbacks,
            callback_state=continuation,
            compiled_chunk=chunk,
        )

    transitions = []

    def observe(execute):
        def wrapped(parameters, prepared, initial, invariants, samples):
            packed, emissions = execute(
                parameters, prepared, initial, invariants, samples
            )
            transitions.append((initial, packed))
            return packed, emissions

        return wrapped

    with torch_compiler_warning_context():
        # Monitor live inputs and output storage at the actual compiled boundary
        # after capture. The probe does not replace clone/pack operations.
        run(0, split, state)
        with monkeypatch.context() as patch:
            kernels = {**chunk._legacy_callback_kernels, **chunk._callback_kernels}
            engine = kernels[callbacks]
            assert engine._kernels
            for signature, (
                execute,
                layout,
                isolate_views,
            ) in engine._kernels.copy().items():
                patch.setitem(
                    engine._kernels,
                    signature,
                    (observe(execute), layout, isolate_views),
                )
            prefix, prefix_aux = run(0, split, state)
        assert len(transitions) == split // width
        for (_, previous), (incoming, _) in zip(
            transitions[:-1], transitions[1:], strict=True
        ):
            reused = any(
                torch._C._is_alias_of(source, target)
                for source, target in zip(previous, incoming, strict=True)
            )
            assert reused is (not checkpointed)
        prefix_callback = prefix_aux["callbacks"]
        public_leaves = pytree.tree_leaves((prefix, prefix_aux))
        assert not any(
            torch._C._is_alias_of(value, pack)
            for value in public_leaves
            if torch.is_tensor(value)
            for pack in transitions[-1][1]
        )
        final, auxiliary = run(split, steps, prefix, prefix_callback.state)

    inputs, packet = packets(0, steps)
    expected, expected_aux = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        state,
        inputs,
        steps=steps,
        draws=packet,
        callbacks=callbacks,
    )
    _close(final, expected)
    _close(auxiliary["callbacks"].state, expected_aux["callbacks"].state)
    actual_trace = {
        name: torch.cat(
            (prefix_callback["trace"][name], auxiliary["callbacks"]["trace"][name])
        )
        for name in ("cell.v", "t")
    }
    _close(actual_trace, expected_aux["callbacks"]["trace"])

    def loss(end, trace):
        # Use both an emitted frame and the final carry in the VJP.
        return (
            trace["cell.v"].square().mean()
            + _voltage(end).sum()
            + end["clock"]["origin"]
        )

    expected_grads = torch.autograd.grad(
        loss(expected, expected_aux["callbacks"]["trace"]), targets
    )
    actual_grads = torch.autograd.grad(loss(final, actual_trace), targets)
    _close(actual_grads, expected_grads)
    assert all(gradient.abs().sum() > 0 for gradient in actual_grads)
    assert final["clock"]["origin"].requires_grad
    assert not final["clock"]["t"].requires_grad
    assert not torch._C._is_alias_of(_voltage(final), _voltage(prefix))
    assert not torch._C._is_alias_of(_voltage(final), initial_v)
    assert not torch._C._is_alias_of(
        _voltage(final), auxiliary["populations"]["cell"]["v"]
    )
    assert not torch._C._is_alias_of(
        _voltage(final), auxiliary["callbacks"]["trace"]["cell.v"]
    )
    assert not torch._C._is_alias_of(
        final["synapses"][projection_name]["delivery_buffer"],
        prefix["synapses"][projection_name]["delivery_buffer"],
    )
    assert tuple(target._version for target in targets) == versions
    _close(state, before)
    _close(draws, before_draws)
