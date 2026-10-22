"""Population scan and recorded while loops consume explicit NOISE tapes."""

import pytest
import torch
from torch.utils import _pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism
from tests.func.test_network import _clone, _close

pytestmark = pytest.mark.cpu
DT = 0.125


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"),
    ):
        yield


class _NoisyLeak(Mechanism):
    Mechanism.RANGE(g=0.001, e=-65.0)
    Mechanism.RANGENOISE("eta", mu=0.0, sigma=0.001, seed=31)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e + self.eta


def _case():
    population = dn.SingleCompartment(
        N=2, C=2, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    population.insert(_NoisyLeak)
    population.initialize()
    population.integrator._initialize(
        population, population.t.new_tensor(DT), force=True
    )
    functional, tensors = dn.func.make_functional(population, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    return population, functional, tensors, prepared


def _run(
    functional, tensors, prepared, tape, *, kernel=None, callbacks=None, state=None
):
    executable = functional if kernel is None else kernel
    return dn.func.run(
        executable.bind(tensors.parameters, prepared),
        tensors.state if state is None else state,
        inputs=dn.func.RolloutInput(noise=tape),
        steps=next(iter(tape.draws.values())).shape[0],
        callbacks=callbacks,
    )


@pytest.mark.parametrize("record", (False, True))
def test_while_noise_tape_preserves_state_rng_and_continuation(record):
    population, functional, tensors, prepared = _case()
    tape = functional.sample_noise_draw_tape(tensors.state, 8)
    callbacks = (
        functional.make_callbacks({"trace": dn.func.Recorder(["v", "t"])})
        if record
        else None
    )
    graphs = []

    def backend(graph, examples):
        del examples
        graphs.append(graph)
        return graph.forward

    kernel = functional.compile_rollout_chunk(4, backend=backend)
    original = _clone((tensors, tape), detached=True)
    global_rng = torch.random.get_rng_state().clone()
    with torch.no_grad():
        expected = _run(functional, tensors, prepared, tape, callbacks=callbacks)
        actual = _run(
            functional, tensors, prepared, tape, kernel=kernel, callbacks=callbacks
        )
        repeated = _run(
            functional, tensors, prepared, tape, kernel=kernel, callbacks=callbacks
        )
        prefix = _pytree.tree_map(lambda value: value[:4], tape)
        first = _run(
            functional, tensors, prepared, prefix, kernel=kernel, callbacks=callbacks
        )
        suffix = functional.sample_noise_draw_tape(first[0], 4)
        _close(suffix, _pytree.tree_map(lambda value: value[4:], tape))
        final = _run(
            functional,
            tensors,
            prepared,
            suffix,
            kernel=kernel,
            callbacks=callbacks,
            state=first[0],
        )
    for result in (actual, repeated):
        _close(result[0], expected[0])
        if record:
            _close(dict(result[1]["callbacks"]), dict(expected[1]["callbacks"]))
    _close(final[0], expected[0])
    _close(
        actual[0]["random"]["rng"], _pytree.tree_map(lambda value: value[-1], tape.rng)
    )
    _close((tensors, tape), original)
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
    assert population.t.item() == 0.0
    assert kernel.execution_report().strategy == "compiled_while_loop"
    assert graphs and any(
        node.op == "call_function" and node.target is torch.ops.higher_order.while_loop
        for graph in graphs
        for node in graph.graph.nodes
    )


@pytest.mark.parametrize("record", (False, True))
def test_scan_noise_tape_preserves_trainable_dynamics_and_detaches_noise(record):
    _population, functional, tensors, prepared = _case()
    tape = functional.sample_noise_draw_tape(tensors.state, 8)
    tape = tape._replace(
        draws={name: value.requires_grad_() for name, value in tape.draws.items()}
    )
    callbacks = (
        functional.make_callbacks({"trace": dn.func.Recorder(["v"])})
        if record
        else None
    )
    kernel = functional.compile_rollout_chunk(4, execution="scan", backend="eager")
    expected = _run(functional, tensors, prepared, tape, callbacks=callbacks)
    actual = _run(
        functional, tensors, prepared, tape, kernel=kernel, callbacks=callbacks
    )
    _close(actual[0], expected[0])
    assert all(not value.requires_grad for value in actual[0]["runtime_noise"].values())
    if record:
        _close(dict(actual[1]["callbacks"]), dict(expected[1]["callbacks"]))
    g_name = next(name for name in tensors.parameters if name.endswith("g_param"))
    mu_name = next(name for name in tensors.parameters if name.endswith("eta_mu_param"))
    operands = (
        tensors.parameters[g_name],
        tensors.parameters[mu_name],
        *tape.draws.values(),
    )
    expected_gradients = torch.autograd.grad(
        expected[0]["integrator"]["v"].sum(), operands, allow_unused=True
    )
    actual_gradients = torch.autograd.grad(
        actual[0]["integrator"]["v"].sum(), operands, allow_unused=True
    )
    torch.testing.assert_close(actual_gradients[0], expected_gradients[0])
    assert actual_gradients[0].abs().sum().item() > 0
    assert expected_gradients[1:] == (None,) * (len(operands) - 1)
    for value in actual_gradients[1:]:
        assert value is None or torch.count_nonzero(value).item() == 0
    assert kernel.execution_report().strategy == "compiled_scan"
