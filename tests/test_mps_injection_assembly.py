"""MPS current assembly retains gradients and native checkpoint execution."""

from __future__ import annotations

import io

import pytest
import torch
from torch._dynamo.backends.registry import lookup_backend

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.core import make_intra
from dendra.models.mod import pas
from dendra.models.stim.intra import Intra
from tests.test_injection_assembly import _ordered_reference, _scatter_count

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS is unavailable"
    ),
]
DT = 0.025


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(DEVICE="mps", DTYPE=torch.float32, REQUIRE_GRAD=1, JIT=0),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _model(*, overlap=False):
    model = dn.Population(
        N=2,
        C=6,
        dtype=torch.float32,
        device="mps",
        integrator=dn.bwd_euler_sc(imem=False),
    )
    model.insert(pas, g=0.001, e=-65.0)
    targets = ([0, 0], 2, 0, 4) if overlap else (0, 2, 4)
    for index, target in enumerate(targets):
        waveform = dn.constant(value=(index + 1) * 0.125 * dn.units.nA)
        model[:, target].inject(waveform)
    model.train()
    model.initialize()
    return model


def _amplitudes(model):
    return tuple(waveform.value for waveform, _, _ in model.injections)


def _equal(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("backend_name", ("aot_eager", "inductor"))
def test_mps_compiled_native_groups_refresh_and_preserve_waveform_gradients(
    backend_name,
):
    model = _model()
    intra = Intra(model, model.injections)
    values, indices = intra.init(torch.tensor([0.0], device="mps", dtype=torch.float32))
    samples = [value[..., 0] for value in values]
    scatter_counts = []

    def backend(graph, inputs):
        scatter_counts.append(_scatter_count(graph))
        return lookup_backend(backend_name)(graph, inputs)

    model.make_intra = torch.compile(make_intra, backend=backend, fullgraph=True)
    initial = model._call_make_intra(intra, samples, indices)
    reference = _ordered_reference(intra, samples)
    _equal(initial, reference)
    assert scatter_counts == [1]
    assert [count for count, _ in intra._prepare_assembly(samples, indices).groups] == [
        3
    ]
    weights = torch.arange(1, 13, device="mps", dtype=torch.float32).reshape(2, 6)
    actual_gradients = torch.autograd.grad(
        (initial * weights).sum(), _amplitudes(model), retain_graph=True
    )
    expected_gradients = torch.autograd.grad(
        (reference * weights).sum(), _amplitudes(model), retain_graph=True
    )
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        _equal(actual, expected)
        assert expected.abs().item() > 0

    indices[1][-1].fill_(0)
    changed = model._call_make_intra(intra, samples, indices)
    _equal(changed, _ordered_reference(intra, samples))
    assert scatter_counts[-1] == 2
    replacement = [tuple(item.clone() for item in index) for index in indices]
    replacement[2][-1].fill_(0)
    changed = model._call_make_intra(intra, samples, replacement)
    _equal(changed, _ordered_reference(intra, samples, replacement))
    assert scatter_counts[-1] == 3


@pytest.mark.parametrize("execution", ("aot_eager", "inductor", "vmap"))
def test_mps_functional_grouped_assembly_and_overlap_keep_gradients(execution):
    model = _model(overlap=True)
    plan = dn.func.FunctionalIntra(model)
    tensors = plan.extract()
    assert any(count > 1 for count, _ in plan._current_groups)
    times = torch.tensor([0.0, DT, 2 * DT], device="mps", dtype=torch.float32)

    def assemble(*values):
        return plan.assemble_values(
            dict(zip(tensors.parameters, values, strict=True)), tensors.constants, times
        )

    parameters = tuple(tensors.parameters.values())
    if execution in ("aot_eager", "inductor"):
        actual = torch.compile(assemble, backend=execution, fullgraph=True)(*parameters)
    else:
        actual = torch.vmap(assemble)(
            *(torch.stack((value, 2 * value)) for value in parameters)
        )[0]
    intra = Intra(model, model.injections)
    values, indices = intra.init(times)
    reference = torch.stack(
        [
            _ordered_reference(intra, [value[..., step] for value in values], indices)
            for step in range(len(times))
        ]
    )
    _equal(actual, reference)
    weights = torch.arange(1, 13, device="mps", dtype=torch.float32).reshape(1, 2, 6)
    actual_gradients = torch.autograd.grad(
        (actual * weights).sum(), parameters, retain_graph=True
    )
    expected_gradients = torch.autograd.grad(
        (reference * weights).sum(), _amplitudes(model)
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        _equal(actual_gradient, expected_gradient)
        assert expected_gradient.abs().item() > 0


def _run(model, steps, *, checkpointed=False):
    recorder = dn.Recorder(("v",))
    with dn.ctx(JIT=1, BACKEND="aot_eager", FULLGRAPH=True):
        if checkpointed:
            model.longrun_checkpointed(
                tstop=steps * DT,
                dt=DT,
                chunklength=3,
                safe_checkpoint=True,
            )
        else:
            model.run(tstop=steps * DT, dt=DT, callbacks=[recorder])
    return model.v.unsqueeze(0) if checkpointed else recorder.stack("v")


def test_mps_multi_injection_rollouts_preserve_checkpoint_gradients_and_continuation():
    direct = _model(overlap=True)
    checkpointed = _model(overlap=True)
    expected = _run(direct, 8)
    actual = _run(checkpointed, 8, checkpointed=True)
    torch.testing.assert_close(actual, expected[-1:], rtol=1e-5, atol=1e-5)
    actual_gradients = torch.autograd.grad(
        checkpointed.v.square().mean(), _amplitudes(checkpointed)
    )
    expected_gradients = torch.autograd.grad(
        direct.v.square().mean(), _amplitudes(direct)
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=3e-5, atol=1e-7
        )
        assert expected_gradient.abs().item() > 0

    prefix = _model(overlap=True)
    with torch.no_grad():
        _run(prefix, 3)
        checkpoint = prefix.state_dict_for_checkpoint()
        stream = io.BytesIO()
        torch.save(checkpoint, stream)
        stream.seek(0)
        restored = _model(overlap=True)
        restored.restore_dict_from_checkpoint(torch.load(stream, weights_only=False))
        expected_suffix = _run(prefix, 5)
        actual_suffix = _run(restored, 5)
    _equal(actual_suffix, expected_suffix)
    _equal(restored.v, prefix.v)
    _equal(restored.t, prefix.t)
