"""State/workspace resets preserve compiled code and refresh numerical inputs."""

from types import MethodType

import pytest
import torch

import dendra as dn
from dendra.models.integrators.core import Integrator
from dendra.models.integrators.implicit import _bwd_euler_sc
from dendra.models.mod import pas

pytestmark = pytest.mark.cpu


class _CountingCompiler:
    def __init__(self, monkeypatch):
        self.graphs = []
        self.wrappers = []
        compile_ = torch.compile

        def compile_counted(function, **kwargs):
            self.wrappers.append(function)
            return compile_(function, **kwargs)

        monkeypatch.setattr(torch, "compile", compile_counted)

    def backend(self, graph, _example_inputs):
        self.graphs.append(graph)
        return graph.forward

    def integrator_wrapper_count(self, integrator):
        return sum(
            getattr(function, "__self__", None) is integrator
            for function in self.wrappers
        )


@pytest.fixture
def compiler(monkeypatch):
    # Each test owns its compiler observations, independently of suite order.
    torch._dynamo.reset()
    value = _CountingCompiler(monkeypatch)
    yield value
    torch._dynamo.reset()


def _population():
    with dn.ctx(JIT=0, REQUIRE_GRAD=0, DEVICE="cpu", DTYPE=torch.float32):
        model = dn.Population(
            N=2,
            C=3,
            dtype=torch.float32,
            v_init=-62.0,
            integrator=_bwd_euler_sc,
        )
        model.insert(pas, g=0.0002, e=-71.0)
        model.requires_grad_(False)
        model.eval()
        model.initialize()
    return model


def test_population_reinitialize_reuses_compiled_kernel_and_repeats_voltage(compiler):
    model = _population()
    with dn.ctx(
        JIT=1,
        BACKEND=compiler.backend,
        FULLGRAPH=True,
        DEVICE="cpu",
        DTYPE=torch.float32,
    ):
        model.run(tstop=0.05, dt=0.025)
        expected = model.v.detach().clone()
        assert len(compiler.graphs) == 1
        kernels = dict(model.integrator._compiled_kernels)
        wrappers = len(compiler.wrappers)

        for _ in range(2):
            model.initialize()
            assert not model.integrator.initialized
            model.run(tstop=0.05, dt=0.025)
            torch.testing.assert_close(model.v, expected, rtol=0, atol=0)
            assert model.v.dtype == torch.float32
            assert model.integrator._compiled_kernels == kernels
            assert len(compiler.wrappers) == wrappers
            assert len(compiler.graphs) == 1


def test_population_reinitialize_updates_dt_and_capacitance_without_new_wrappers(
    compiler,
):
    model, reference = _population(), _population()
    kernels = None
    wrappers = None
    previous_voltage = None
    for dt, capacitance in (
        (0.025, 1.0),
        (0.05, 1.0),
        (0.05, 2.0),
    ):
        for owner in (model, reference):
            with torch.no_grad():
                owner.cm.fill_(capacitance)
            owner.initialize(populate_parameter_buffers=False)
        with dn.ctx(
            JIT=1,
            BACKEND=compiler.backend,
            FULLGRAPH=True,
            DEVICE="cpu",
            DTYPE=torch.float32,
        ):
            model.run(tstop=2 * dt, dt=dt)
        with dn.ctx(JIT=0, DEVICE="cpu", DTYPE=torch.float32):
            reference.run(tstop=2 * dt, dt=dt)
        torch.testing.assert_close(model.v, reference.v, rtol=0, atol=0)
        assert model.v.dtype == torch.float32
        assert len(compiler.graphs) == 1
        if previous_voltage is not None:
            assert not torch.equal(model.v, previous_voltage)
        previous_voltage = model.v.detach().clone()
        if kernels is None:
            kernels = dict(model.integrator._compiled_kernels)
            wrappers = compiler.integrator_wrapper_count(model.integrator)
        else:
            assert model.integrator._compiled_kernels == kernels
            assert compiler.integrator_wrapper_count(model.integrator) == wrappers


class _WorkspaceModel(torch.nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.register_buffer("v", torch.arange(1, 7, dtype=torch.float32).reshape(2, 3))
        self.gain = torch.nn.Parameter(torch.tensor(2.0))
        self.shape = tuple(self.v.shape)
        self.jit = True
        self.jit_network_solves = False
        self.backend = backend
        self.fullgraph = True
        self.dynamic = False
        self.compile_mode = None
        self.compile_options = None
        self.fail = False

    def device(self):
        return self.v.device


class _WorkspaceIntegrator(Integrator):
    def __init__(self, model):
        handler = torch.nn.Module()
        handler.mechanisms = torch.nn.ModuleDict()
        super().__init__(model, handler)
        self.register_buffer("coefficient", torch.zeros_like(model.gain))

    def initialize(self, model, dt):
        self.coefficient = model.gain * dt
        if model.fail:
            raise RuntimeError("incomplete workspace")

    def _advance(self, value):
        return value + self.coefficient * value

    def _double(self, value):
        return value * 2

    def step(self, model, dt, ve=None, intra=None):
        return self._call_kernel("_advance", model.v)


def test_rebuilt_workspaces_read_changed_dt_parameters_and_new_autograd_graphs(
    compiler,
):
    model = _WorkspaceModel(compiler.backend)
    integrator = _WorkspaceIntegrator(model)
    kernels = None
    for index, (dt, gain) in enumerate(((0.025, 2.0), (0.05, 2.0), (0.05, 3.0))):
        with torch.no_grad():
            model.gain.fill_(gain)
        model.gain.grad = None
        integrator._initialize(model, dt, force=True)
        actual = integrator.step(model, dt)
        torch.testing.assert_close(actual, model.v * (1.0 + gain * dt))
        actual.sum().backward()
        torch.testing.assert_close(model.gain.grad, model.v.sum() * dt)
        assert actual.dtype == torch.float32
        assert len(compiler.graphs) == 1
        assert len(compiler.wrappers) == 1
        if index == 0:
            kernels = dict(integrator._compiled_kernels)
        else:
            assert integrator._compiled_kernels == kernels


@pytest.mark.parametrize("change", ["shape", "dtype"])
def test_changed_tensor_contract_specializes_retained_wrapper(compiler, change):
    model = _WorkspaceModel(compiler.backend)
    integrator = _WorkspaceIntegrator(model)
    integrator._initialize(model, 0.025)
    integrator.step(model, 0.025)
    assert len(compiler.graphs) == 1
    kernels = dict(integrator._compiled_kernels)

    if change == "shape":
        model.v = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)
        model.shape = tuple(model.v.shape)
    else:
        model.to(dtype=torch.float64)
    integrator._initialize(model, 0.05, force=True)
    actual = integrator.step(model, 0.05)
    torch.testing.assert_close(actual, model.v * 1.1)
    assert actual.dtype == model.v.dtype
    assert integrator._compiled_kernels == kernels
    assert len(compiler.wrappers) == 1
    assert len(compiler.graphs) == 2


def test_failed_workspace_initialization_clears_local_kernels_and_retries(compiler):
    model = _WorkspaceModel(compiler.backend)
    integrator = _WorkspaceIntegrator(model)
    integrator._initialize(model, 0.025)
    integrator.step(model, 0.025)
    assert integrator._compiled_kernels
    model.fail = True
    with pytest.raises(RuntimeError, match="incomplete workspace"):
        integrator._initialize(model, 0.05, force=True)
    assert not integrator.initialized
    assert not integrator._compiled_kernels
    assert not integrator._compiled_kernel_sources
    assert integrator.dt == 0.025

    model.fail = False
    integrator._initialize(model, 0.05)
    torch.testing.assert_close(integrator.step(model, 0.05), model.v * 1.1)
    assert integrator.initialized
    assert len(compiler.wrappers) == 2


def test_changed_compiler_configuration_invalidates_local_wrappers(compiler):
    model = _WorkspaceModel(compiler.backend)
    integrator = _WorkspaceIntegrator(model)
    integrator._initialize(model, 0.025)
    integrator.step(model, 0.025)
    assert integrator._compiled_kernels

    model.fullgraph = False
    integrator.configure_jit(model)
    assert not integrator._compiled_kernels
    assert not integrator._compiled_kernel_sources
    torch.testing.assert_close(integrator.step(model, 0.025), model.v * 1.05)
    assert len(compiler.wrappers) == 2


def test_independent_integrators_have_separate_specialization_budgets(compiler):
    owners = []
    for columns in range(3, 15):
        model = _WorkspaceModel(compiler.backend)
        model.v = torch.arange(1, 2 * columns + 1, dtype=torch.float32).reshape(
            2, columns
        )
        model.shape = tuple(model.v.shape)
        integrator = _WorkspaceIntegrator(model)
        integrator._initialize(model, 0.025)
        torch.testing.assert_close(integrator.step(model, 0.025), model.v * 1.05)
        owners.append((model, integrator))
    assert len(compiler.graphs) == len(owners) == 12
    assert len(compiler.wrappers) == 12

    # Compiling more than the default shared-code budget must leave an earlier
    # instance warm, including its rebuilt, differentiable numerical workspace.
    model, integrator = owners[0]
    kernels = dict(integrator._compiled_kernels)
    integrator._initialize(model, 0.025, force=True)
    actual = integrator.step(model, 0.025)
    actual.sum().backward()
    torch.testing.assert_close(model.gain.grad, model.v.sum() * 0.025)
    assert integrator._compiled_kernels == kernels
    assert len(compiler.graphs) == len(compiler.wrappers) == 12


@pytest.mark.parametrize("replacement", ["method", "code"])
def test_replacing_kernel_source_replaces_only_its_cached_wrapper(
    compiler, monkeypatch, replacement
):
    model = _WorkspaceModel(compiler.backend)
    integrator = _WorkspaceIntegrator(model)
    integrator._initialize(model, 0.025)
    integrator.step(model, 0.025)
    integrator._call_kernel("_double", model.v)
    previous = dict(integrator._compiled_kernels)

    def subtract(self, value):
        return value - self.coefficient * value

    if replacement == "method":
        integrator._advance = MethodType(subtract, integrator)
    else:
        monkeypatch.setattr(
            _WorkspaceIntegrator._advance, "__code__", subtract.__code__
        )
    integrator._initialize(model, 0.025, force=True)
    torch.testing.assert_close(integrator.step(model, 0.025), model.v * 0.95)
    assert len(compiler.graphs) == len(compiler.wrappers) == 3
    for key, old in previous.items():
        if key[0] == "_advance":
            assert integrator._compiled_kernels[key] is not old
        else:
            assert integrator._compiled_kernels[key] is old
    assert len(integrator._compiled_kernels) == len(previous)
    integrator.step(model, 0.025)
    assert len(compiler.graphs) == len(compiler.wrappers) == 3


class _StaticWorkspaceIntegrator(_WorkspaceIntegrator):
    @staticmethod
    def _advance(value, offset=1.0, *, scale=2.0):
        return (value + offset) * scale


@pytest.mark.parametrize("change", ["code", "defaults", "kwdefaults"])
def test_static_kernel_source_and_defaults_remain_live(compiler, monkeypatch, change):
    model = _WorkspaceModel(compiler.backend)
    integrator = _StaticWorkspaceIntegrator(model)
    integrator._initialize(model, 0.025)
    torch.testing.assert_close(integrator.step(model, 0.025), (model.v + 1) * 2)
    function = _StaticWorkspaceIntegrator._advance

    if change == "code":

        def subtract(value, offset=1.0, *, scale=2.0):
            return (value - offset) * scale

        monkeypatch.setattr(function, "__code__", subtract.__code__)
        base, scale = model.v - 1, 2
    elif change == "defaults":
        monkeypatch.setattr(function, "__defaults__", (3.0,))
        base, scale = model.v + 3, 2
    else:
        monkeypatch.setattr(function, "__kwdefaults__", {"scale": 4.0})
        base, scale = model.v + 1, 4
    torch.testing.assert_close(integrator.step(model, 0.025), base * scale)
    assert len(compiler.graphs) == len(compiler.wrappers) == 2

    # In-place edits share the same mapping with the isolated function and
    # should be guarded by Dynamo without recreating the owning wrapper.
    monkeypatch.setitem(function.__kwdefaults__, "scale", 5.0)
    torch.testing.assert_close(integrator.step(model, 0.025), base * 5)
    assert len(compiler.wrappers) == 2
    assert len(compiler.graphs) == 3


def test_kernel_identity_metadata_is_disposable_and_not_serialized(compiler):
    model = _WorkspaceModel(compiler.backend)
    integrator = _WorkspaceIntegrator(model)
    integrator._initialize(model, 0.025)
    integrator.step(model, 0.025)
    assert integrator._compiled_kernel_sources
    state = integrator.__getstate__()
    assert state["_compiled_kernels"] == {}
    assert state["_compiled_kernel_sources"] == {}
    assert integrator._compiled_kernel_sources
    integrator.clear_jit_cache()
    assert not integrator._compiled_kernels
    assert not integrator._compiled_kernel_sources
