"""Prepared Network sources share canonical bindings without owning all state."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._network_bindings import SourceBinding
from tests.func.test_network import _clone, _close
from tests.func.test_network_bindings import DT, _network

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"),
    ):
        yield


def test_prepared_projection_matches_registered_native_fields_and_legacy_bindings():
    source = _network()
    functional, tensors = dn.func.make_functional(source.cell, dt=DT)
    prepared = functional._prepare_values(tensors.parameters, tensors.constants)
    frozen = _clone((tensors, prepared), detached=True)
    checked = []
    for reference, native in source.cell.named_buffers(remove_duplicate=False):
        try:
            binding = SourceBinding(functional, source.cell, reference)
        except dn.func.FunctionalizationError:
            continue
        if binding._kind != "mapping":
            continue
        expected = binding.resolve(tensors.parameters, prepared, tensors.state)
        actual = binding.resolve(
            tensors.parameters, prepared, tensors.state, _prepared_only=True
        )
        _close(actual, expected)
        _close(actual, native)
        assert actual.shape == native.shape
        checked.append(reference)
    assert len(checked) >= 3
    assert any(name.endswith("analog.e") for name in checked)
    _close((tensors, prepared), frozen)


def test_prepared_source_uses_each_fresh_frame_and_preserves_raw_parameter_vjps():
    source = _network()
    functional, tensors = dn.func.make_functional(source.cell, dt=DT)
    binding = SourceBinding(functional, source.cell, "mech.analog.e")
    name = next(name for name in tensors.parameters if name.endswith("analog.e_param"))
    results = []
    for shift in (0.0, 0.7):
        raw = (tensors.parameters[name].detach() + shift).requires_grad_()
        parameters = {**tensors.parameters, name: raw}
        prepared = functional._prepare_values(parameters, tensors.constants)
        actual = binding.resolve(
            parameters, prepared, tensors.state, _prepared_only=True
        )
        expected = binding.resolve(parameters, prepared, tensors.state)
        _close(actual, expected)
        observed = torch.autograd.grad(actual.square().sum(), raw, retain_graph=True)
        reference = torch.autograd.grad(expected.square().sum(), raw)
        _close(observed, reference)
        assert torch.count_nonzero(observed[0]) > 0
        results.append(actual.detach())
    assert not torch.equal(*results)


def test_admitted_prepared_read_omits_unused_state_clone_graphs():
    source = _network()
    functional, tensors = dn.func.make_functional(source.cell, dt=DT)
    prepared = functional._prepare_values(tensors.parameters, tensors.constants)
    binding = SourceBinding(functional, source.cell, "mech.analog.e")

    def read(parameters, values, state):
        return binding.resolve(parameters, values, state, _prepared_only=True)

    arguments = tensors.parameters, prepared, tensors.state
    graph = make_fx(read)(*arguments)
    _close(graph(*arguments), source.cell.mech.analog.e)
    assert not any(
        node.op == "call_function" and node.target is torch.ops.aten.clone.default
        for node in graph.graph.nodes
    )


@pytest.mark.parametrize("prepared_only", (False, True))
def test_matching_prepared_source_shape_needs_no_extra_tensor_metadata(prepared_only):
    class Value(torch.Tensor):
        @property
        def ndim(self):
            raise AssertionError("matching source shape does not need ndim")

    value = torch.ones(2).as_subclass(Value)

    def mapping(*args, **kwargs):
        return ({"stored": value},)

    binding = object.__new__(SourceBinding)
    binding._kind, binding.slot, binding.shape = "mapping", "stored", (2,)
    binding._plan = SimpleNamespace(
        _mapping=mapping, _read_only_mapping=mapping, _poisson_plan=None
    )
    assert binding.resolve({}, {}, {}, _prepared_only=prepared_only) is value


@pytest.mark.parametrize("prepared_only", (False, True))
def test_global_source_removes_only_execution_padding_and_preserves_transform_lanes(
    prepared_only,
):
    source = _network()
    functional, tensors = dn.func.make_functional(source.cell, dt=DT)
    prepared = functional._prepare_values(tensors.parameters, tensors.constants)
    binding = SourceBinding(functional, source.cell, "mech.relay.scale")
    assert binding.shape == () and binding._kind == "mapping"
    entry = next(
        entry
        for entry in functional._prepared_buffers
        if f"population.{entry.module_path}.{entry.buffer_name}" == binding.slot
    )

    def read(value):
        values = {
            **prepared,
            "mechanisms": {**prepared["mechanisms"], entry.key: value},
        }
        return binding.resolve(
            tensors.parameters, values, tensors.state, _prepared_only=prepared_only
        )

    value = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    actual = read(value)
    assert actual.shape == ()
    _close(actual, value)
    _close(torch.autograd.grad(actual, value), (torch.ones_like(value),))
    _, tangent = torch.func.jvp(read, (value,), (torch.full_like(value, 0.3),))
    _close(tangent, value.new_tensor(0.3))
    for lanes in (0, 3):
        batch = torch.arange(lanes, dtype=value.dtype)
        _close(torch.func.vmap(read)(batch), batch)


def _prepared_network():
    network = _network()
    network.connect_continuous_one_to_one(
        network.cell[:],
        network.cell[:],
        network.cell.mech.analog,
        pre_var="mech.analog.e",
        input="g_pre",
        transform=torch.sigmoid,
        weight=0.003,
        delay=DT,
        allow_autapses=True,
    )
    network.initialize(DT)
    return network


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_compiled_network_prepared_source_records_resumes_and_matches_native(execution):
    source, native = _prepared_network(), _prepared_network()
    training = execution == "scan"
    program = dn.compile(
        source,
        dt=DT,
        steps=4,
        execution=execution,
        inference=not training,
        initialization="snapshot",
        backend="aot_eager",
        trainable="*analog.e_param" if training else (),
        callbacks={"trace": dn.func.Recorder(("cell.v", "t"))},
    )
    with torch.set_grad_enabled(training):
        _, first = program()
        actual, second = program(first["continuation"])
        traces = [native.cell.v.clone()]
        for _ in range(8):
            native.step()
            traces.append(native.cell.v.clone())
        _close(actual, program.functional.extract(native).state)
        observed = torch.cat(
            (
                first["callbacks"]["trace"]["cell.v"],
                second["callbacks"]["trace"]["cell.v"],
            )
        )
        _close(observed, torch.stack(traces))
        if training:
            parameter = next(
                value
                for name, value in program.named_parameters.items()
                if name.endswith("analog.e_param")
            )
            gradient = torch.autograd.grad(
                actual["populations"]["cell"]["integrator"]["v"].square().sum(),
                parameter,
            )
            expected = torch.autograd.grad(
                native.cell.v.square().sum(), native.cell.mech.analog.e_param
            )
            _close(gradient, expected)
            assert torch.count_nonzero(gradient[0]) > 0
