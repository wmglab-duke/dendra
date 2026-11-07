"""Model lifecycle operations cannot evict unrelated compiled programs."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas

DT = 0.025
pytestmark = [
    pytest.mark.cpu,
    pytest.mark.filterwarnings(r"ignore:`torch\.jit\.script` is deprecated.*"),
    pytest.mark.filterwarnings(r"ignore:`torch\.jit\.script_method` is deprecated.*"),
]


@pytest.fixture(autouse=True)
def isolated_compiler_cache():
    # Explicit test isolation is independent of ordinary library lifecycle.
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _population():
    model = dn.SingleCompartment(
        N=1,
        C=2,
        v_init=-62.0,
        device="cpu",
        dtype=torch.float32,
        integrator=dn.bwd_euler_sc(imem=False),
    )
    model.insert(pas, g=0.001, e=-70.0)
    model.requires_grad_(False)
    model.eval()
    model.initialize()
    return model


def _unrelated_program():
    compilations = []

    def backend(graph, inputs):
        compilations.append(graph)
        return graph.forward

    def numerical(value):
        return value.sin() + value.square()

    compiled = torch.compile(numerical, backend=backend, fullgraph=True)
    value = torch.linspace(-0.5, 0.5, 5, dtype=torch.float32)

    def check():
        with torch_compiler_warning_context():
            actual = compiled(value)
        torch.testing.assert_close(actual, numerical(value), rtol=0, atol=0)

    check()
    assert len(compilations) == 1
    return check, compilations


@pytest.mark.parametrize(
    "operation",
    [
        "population_construct",
        "axon_construct",
        "population_initialize",
        "population_force_rebuild",
        "network_construct",
        "network_initialize",
        "network_force_rebuild",
        "network_dt_change",
    ],
)
def test_population_and_network_lifecycle_preserves_unrelated_graph(operation):
    with (
        dn.ctx(JIT=0, REQUIRE_GRAD=0, DEVICE="cpu", DTYPE=torch.float32),
        torch.no_grad(),
    ):
        model = _population()
        network = dn.Network({"cell": model})
        network.initialize(DT)
        check, compilations = _unrelated_program()

        if operation == "population_construct":
            _population()
        elif operation == "axon_construct":
            dn.Unmyelinated([2.0], L=40.0, dx=10.0, dtype=torch.float32)
        elif operation == "population_initialize":
            model.initialize()
        elif operation == "population_force_rebuild":
            model.initialize(force_rebuild=True)
        elif operation == "network_construct":
            dn.Network({"other": _population()})
        elif operation == "network_initialize":
            network.initialize(DT)
        elif operation == "network_force_rebuild":
            network.initialize(DT, force_rebuild=True)
        else:
            network.initialize(DT * 2)

        check()
        assert len(compilations) == 1


@pytest.mark.parametrize("owner", ["population", "integrator", "network"])
@pytest.mark.parametrize("reset_global", [False, True])
def test_global_compiler_reset_remains_an_explicit_pickle_option(owner, reset_global):
    with (
        dn.ctx(JIT=0, REQUIRE_GRAD=0, DEVICE="cpu", DTYPE=torch.float32),
        torch.no_grad(),
    ):
        model = _population()
        network = dn.Network({"cell": model})
        target = {
            "population": model,
            "integrator": model.integrator,
            "network": network,
        }[owner]
        check, compilations = _unrelated_program()
        assert target.pickleable(reset_global_compiler=reset_global) is target
        check()
        assert len(compilations) == (2 if reset_global else 1)


class _ShiftVoltage(torch.nn.Module):
    def forward(self, voltage, offset):
        return (voltage + offset,)


def test_initialization_transform_refreshes_state_without_discarding_valid_kernel():
    compilations = []

    def backend(graph, inputs):
        compilations.append(graph)
        return graph.forward

    with (
        dn.ctx(
            JIT=1,
            BACKEND=backend,
            FULLGRAPH=1,
            REQUIRE_GRAD=0,
            DEVICE="cpu",
            DTYPE=torch.float32,
        ),
        torch.no_grad(),
    ):
        model = _population()
        action = model.register_pre_initialize_transform(
            "shift",
            _ShiftVoltage(),
            reads=("state.integrator.v",),
            writes=("state.integrator.v",),
            inputs={"offset": torch.tensor(1.0)},
        )
        model.initialize()
        with torch_compiler_warning_context():
            model.run(tstop=DT, dt=DT)
        old_kernels = dict(model.integrator._compiled_kernels)
        compiled_count = len(compilations)
        assert old_kernels and compiled_count > 0

        action.input_values()[0].fill_(3.0)
        model.initialize()
        torch.testing.assert_close(model.v, torch.full_like(model.v, -59.0))
        with torch_compiler_warning_context():
            model.run(tstop=DT, dt=DT)
        assert model.integrator._compiled_kernels == old_kernels
        assert len(compilations) == compiled_count

        reference = _population()
        reference.set_v_init(-59.0)
        reference.initialize()
        with dn.ctx(JIT=0):
            reference.run(tstop=DT, dt=DT)
        torch.testing.assert_close(model.v, reference.v, rtol=0, atol=0)
