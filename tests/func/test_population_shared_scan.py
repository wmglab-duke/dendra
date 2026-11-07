"""Population training scans preserve their public transition boundaries."""

from __future__ import annotations

import gc
import weakref
from dataclasses import replace

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from tests.func.test_lfp import _VoltageEnergy
from tests.func.test_poisson_stimulation import DT as NOISE_DT
from tests.func.test_poisson_stimulation import (
    _NoisyPoissonLeak,
)
from tests.func.test_poisson_stimulation import _population as poisson_population
from tests.func.test_poisson_stimulation import (
    _slice,
)
from tests.func.test_runner_stimulation_clock import STEPS as PULSE_STEPS
from tests.func.test_runner_stimulation_clock import _case as pulse_case

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]
DT = 0.01
STEPS = 3
_BACKENDS = ("aot_eager", pytest.param("inductor", marks=pytest.mark.slow))


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0,
                REQUIRE_GRAD=True,
                DEVICE="cpu",
                DTYPE=torch.float64,
                CHECK_PURITY=False,
            ),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _tensor_leaves(tree):
    return [value for value in pytree.tree_leaves(tree) if torch.is_tensor(value)]


def _close(actual, expected, *, rtol=2e-10, atol=2e-11):
    values, structure = pytree.tree_flatten(actual)
    references, reference_structure = pytree.tree_flatten(expected)
    assert structure == reference_structure
    for value, reference in zip(values, references, strict=True):
        torch.testing.assert_close(value, reference, rtol=rtol, atol=atol)


def _gradients(loss, targets, *, retain_graph=False):
    return dict(
        zip(
            targets,
            torch.autograd.grad(
                loss,
                tuple(targets.values()),
                retain_graph=retain_graph,
                allow_unused=True,
            ),
            strict=True,
        )
    )


def _close_gradients(actual, expected, *, rtol=2e-8, atol=2e-10):
    assert actual.keys() == expected.keys()
    for name, reference in expected.items():
        value = actual[name]
        if reference is None or not torch.count_nonzero(reference):
            assert value is None or not torch.count_nonzero(value), name
        else:
            assert value is not None and torch.isfinite(value).all(), name
            scale = reference.abs().amax()
            torch.testing.assert_close(
                value / scale, reference / scale, rtol=rtol, atol=atol, msg=name
            )


def _model(*, imem=False):
    with dn.ctx(IMEM=imem):
        if imem:
            model = dn.Population(
                N=2,
                C=3,
                v_init=[-65.0, -62.0, -59.0],
                dtype=torch.float64,
                integrator=dn.bwd_euler_sc(imem=True),
            )
        else:
            model = dn.Unmyelinated(
                [2.0, 2.5],
                L=2.0,
                dx=1.0,
                v_init=[-65.0, -62.0, -59.0],
                dtype=torch.float64,
                integrator=dn.bwd_euler_ub(method="pcr", imem=False),
            )
        model.insert(pas, g=0.001, e=-65.0)
        model.initialize()
        model.train()
    return model


def _operands(functional, tensors, presence, *, offset=0.0):
    parameters, constants, state = _clone(
        (tensors.parameters, tensors.constants, tensors.state)
    )
    g_name = next(name for name in parameters if name.endswith(".g_param"))
    parameters[g_name] = (parameters[g_name] + offset).requires_grad_()
    constants["diam"] = (constants["diam"] + offset).requires_grad_()
    voltage = state["integrator"]["v"].requires_grad_()
    targets = {"g": parameters[g_name], "diam": constants["diam"], "v": voltage}
    shape = (STEPS, *functional.shape)
    ve = intra = None
    if presence[0]:
        raw = (
            torch.linspace(-0.03, 0.05, (STEPS + 2) * shape[-1], dtype=torch.float64)
            .reshape(STEPS + 2, 1, shape[-1])
            .requires_grad_()
        )
        ve = raw[1:-1].expand(shape)
        targets["ve"] = raw
        assert 0 in ve.stride() and ve.storage_offset() > 0
    if presence[1]:
        raw = (
            torch.linspace(
                -2e-9,
                3e-9,
                (STEPS + 2) * shape[-2] * shape[-1] * 2,
                dtype=torch.float64,
            )
            .reshape(STEPS + 2, shape[-2], shape[-1] * 2)
            .requires_grad_()
        )
        intra = raw[1:-1, :, ::2]
        targets["intra"] = raw
        assert not intra.is_contiguous() and intra.storage_offset() > 0
    return (
        parameters,
        constants,
        state,
        dn.func.RolloutInput(ve=ve, intra=intra),
        targets,
    )


def _plain_pair(functional, tensors, chunk, presence, *, offset=0.0):
    expected = _operands(functional, tensors, presence, offset=offset)
    actual = _operands(functional, tensors, presence, offset=offset)
    parameters, constants, state, inputs, targets = actual
    prepared = functional.prepare(parameters, constants)
    references = {
        "prepared": weakref.ref(prepared),
        "g": weakref.ref(targets["g"]),
        "diam": weakref.ref(constants["diam"]),
        "voltage": weakref.ref(state["integrator"]["v"]),
    }
    workspace = next(
        value
        for value in pytree.tree_leaves(prepared.values)
        if torch.is_tensor(value) and value.requires_grad
    )
    references["workspace"] = weakref.ref(workspace)
    for name in ("ve", "intra"):
        if name in targets:
            references[name] = weakref.ref(targets[name])
    before = _clone((parameters, constants, state, inputs))
    versions = [
        value._version
        for value in _tensor_leaves((parameters, constants, state, inputs))
    ]
    wanted, wanted_aux = functional.rollout(
        expected[0],
        functional.prepare(expected[0], expected[1]),
        expected[2],
        expected[3],
        steps=STEPS,
    )
    final, auxiliary = chunk(parameters, prepared, state, inputs)
    _close((final, auxiliary), (wanted, wanted_aux))
    wanted_gradients = _gradients(
        wanted["integrator"]["v"].square().mean(), expected[4]
    )
    gradients = _gradients(final["integrator"]["v"].square().mean(), targets)
    _close_gradients(gradients, wanted_gradients)
    assert not final["control"]["duration_remainder"].requires_grad
    for source in _tensor_leaves((parameters, constants, state, inputs)):
        assert not torch._C._is_alias_of(final["integrator"]["v"], source)
    _close((parameters, constants, state, inputs), before, rtol=0, atol=0)
    assert versions == [
        value._version
        for value in _tensor_leaves((parameters, constants, state, inputs))
    ]
    assert chunk.execution_report().strategy == "compiled_scan"
    return references, _clone((final, auxiliary))


@pytest.mark.parametrize(
    "presence", ((False, False), (True, False), (False, True), (True, True))
)
def test_population_scan_optional_drive_views_preserve_forward_and_all_source_gradients(
    presence,
):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    chunk = functional.compile_rollout_chunk(
        STEPS, execution="scan", backend="aot_eager"
    )
    _plain_pair(functional, tensors, chunk, presence)


@pytest.mark.parametrize("kind", ("intra", "extra"))
def test_population_scan_keeps_distinct_atomic_and_recurrent_bound_waveform_clocks(
    kind,
):
    from tests.func.test_runner_stimulation_clock import DT as pulse_dt
    from tests.func.test_runner_stimulation_clock import _source_case

    functional, tensors, parameters, prepared, amplitude = pulse_case(kind)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("v", "t"))})
    chunk = functional.compile_rollout_chunk(
        PULSE_STEPS, execution="scan", backend="aot_eager"
    )
    wanted_atomic, wanted_aux = functional.rollout(
        parameters, prepared, tensors.state, steps=PULSE_STEPS
    )
    atomic, auxiliary = chunk(parameters, prepared, tensors.state)
    _close((atomic, auxiliary), (wanted_atomic, wanted_aux), rtol=2e-5, atol=2e-5)
    atomic_reference_gradient = torch.autograd.grad(
        wanted_atomic["integrator"]["v"].square().mean(), amplitude, retain_graph=True
    )[0]
    atomic_gradient = torch.autograd.grad(
        atomic["integrator"]["v"].square().mean(), amplitude, retain_graph=True
    )[0]
    torch.testing.assert_close(atomic_gradient, atomic_reference_gradient)
    expected, expected_aux = dn.func.run(
        functional.bind(parameters, prepared),
        tensors.state,
        steps=PULSE_STEPS,
        callbacks=callbacks,
    )
    actual, actual_aux = dn.func.run(
        chunk.bind(parameters, prepared),
        tensors.state,
        steps=PULSE_STEPS,
        callbacks=callbacks,
    )
    _close((actual, actual_aux), (expected, expected_aux), rtol=2e-5, atol=2e-5)
    native, native_extra, native_waveform = _source_case(kind)
    recorder = dn.Recorder(("v", "t"))
    native.run(
        steps=PULSE_STEPS,
        dt=pulse_dt,
        extra=native_extra,
        callbacks=[recorder],
    )
    native_trace = {name: recorder.stack(name) for name in ("v", "t")}
    _close(actual, functional.extract(native).state, rtol=2e-5, atol=2e-5)
    _close(dict(actual_aux["callbacks"]["trace"]), native_trace, rtol=2e-5, atol=2e-5)
    _close(wanted_atomic, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        actual_aux["callbacks"]["trace"]["t"],
        expected_aux["callbacks"]["trace"]["t"],
        rtol=0,
        atol=0,
    )
    reference_gradient = torch.autograd.grad(
        expected_aux["callbacks"]["trace"]["v"].square().mean(),
        amplitude,
        retain_graph=True,
    )[0]
    gradient = torch.autograd.grad(
        actual_aux["callbacks"]["trace"]["v"].square().mean(), amplitude
    )[0]
    native_gradient = torch.autograd.grad(
        native_trace["v"].square().mean(), native_waveform.amp
    )[0]
    assert torch.count_nonzero(reference_gradient) == 0
    torch.testing.assert_close(gradient, reference_gradient, rtol=2e-5, atol=1e-8)
    torch.testing.assert_close(gradient, native_gradient, rtol=2e-5, atol=1e-8)
    accepted = tensors.state
    frames = [accepted["integrator"]["v"]]
    for _ in range(PULSE_STEPS):
        accepted, _ = functional.step(parameters, prepared, accepted)
        frames.append(accepted["integrator"]["v"])
    manual_gradient = torch.autograd.grad(
        torch.stack(frames).square().mean(), amplitude
    )[0]
    assert not torch.equal(expected["integrator"]["v"], accepted["integrator"]["v"])
    assert torch.isfinite(manual_gradient).all()
    assert torch.count_nonzero(manual_gradient) > 0


def test_population_scan_explicit_noise_and_poisson_tapes_replay_and_detach_draws():
    source, waveform, _extra = poisson_population(
        "mechanism", mechanism=_NoisyPoissonLeak
    )
    functional, tensors = dn.func.make_functional(source, dt=NOISE_DT)
    parameters = _clone(tensors.parameters)
    amp_name = next(name for name in parameters if name.endswith(".amp"))
    amplitude = parameters[amp_name].requires_grad_()
    prepared = functional.prepare(parameters, tensors.constants)
    noise = functional.sample_noise_draw_tape(tensors.state, 2 * STEPS)
    stimulation = functional.sample_stimulation_draw_tape(tensors.state, 2 * STEPS)
    noise = noise._replace(
        draws={key: value.requires_grad_() for key, value in noise.draws.items()}
    )
    stimulation = stimulation._replace(
        draws={key: value.requires_grad_() for key, value in stimulation.draws.items()}
    )
    inputs = dn.func.RolloutInput(noise=noise, stimulation=stimulation)
    before, source_rng, global_rng = (
        _clone(inputs),
        waveform.generator.get_state().clone(),
        torch.get_rng_state().clone(),
    )
    expected, _ = functional.rollout(
        parameters, prepared, tensors.state, inputs, steps=2 * STEPS
    )
    chunk = functional.compile_rollout_chunk(
        STEPS, execution="scan", backend="aot_eager"
    )
    prefix, _ = chunk(parameters, prepared, tensors.state, _slice(inputs, 0, STEPS))
    actual, _ = chunk(parameters, prepared, prefix, _slice(inputs, STEPS, 2 * STEPS))
    replay, _ = chunk(parameters, prepared, tensors.state, _slice(inputs, 0, STEPS))
    _close(prefix, replay)
    _close(actual, expected)
    for group in ("runtime_noise", "stimulation"):
        assert all(not value.requires_grad for value in _tensor_leaves(actual[group]))
    targets = {
        "amp": amplitude,
        **{f"noise:{key}": value for key, value in noise.draws.items()},
        **{f"poisson:{key}": value for key, value in stimulation.draws.items()},
    }
    reference = _gradients(
        expected["integrator"]["v"].square().mean(), targets, retain_graph=True
    )
    gradients = _gradients(actual["integrator"]["v"].square().mean(), targets)
    _close_gradients(gradients, reference)
    assert reference["amp"].abs().sum() > 0
    assert all(
        value is None or not torch.count_nonzero(value)
        for name, value in gradients.items()
        if name != "amp"
    )
    _close(inputs, before, rtol=0, atol=0)
    assert torch.equal(waveform.generator.get_state(), source_rng)
    assert torch.equal(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_population_scan_resumes_recurrent_and_strided_invariant_callbacks(backend):
    model = _model(imem=True)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters = _clone(tensors.parameters)
    g_name = next(name for name in parameters if name.endswith(".g_param"))
    parameters[g_name].requires_grad_()
    fields = [
        torch.linspace(1.0 + index, 2.0 + index, 6, dtype=torch.float64)
        .reshape(3, 2)
        .T.detach()
        .requires_grad_()
        for index in (0, 1)
    ]
    drive = (
        torch.linspace(-2e-7, 3e-7, 5 * model.v.numel(), dtype=torch.float64)
        .reshape(5, *model.shape)
        .requires_grad_()
    )
    callbacks = functional.make_callbacks(
        {
            "trace": dn.func.Recorder(("v", "t", "i_membrane")),
            "lfp": dn.func.LFP(fields),
            "energy": _VoltageEnergy(),
        }
    )
    prepared = functional.prepare(parameters, tensors.constants)
    expected, expected_aux = dn.func.run(
        functional.bind(parameters, prepared),
        tensors.state,
        inputs=dn.func.RolloutInput(intra=drive),
        steps=5,
        callbacks=callbacks,
    )
    options = (
        {"cpp_wrapper": True, "compile_threads": 1} if backend == "inductor" else None
    )
    chunk = functional.compile_rollout_chunk(
        STEPS,
        execution="scan",
        backend=backend,
        **({"options": options} if options else {}),
    )
    prefix, first = dn.func.run(
        chunk.bind(parameters, prepared),
        tensors.state,
        inputs=dn.func.RolloutInput(intra=drive[:2]),
        steps=2,
        callbacks=callbacks,
    )
    carry = first["callbacks"].state
    invariant = carry.carries["lfp"]
    strided = invariant.unsqueeze(-1).expand(*invariant.shape, 2).clone()[..., 0]
    assert not strided.is_contiguous()
    resumed = replace(carry, carries={**carry.carries, "lfp": strided})
    actual, second = dn.func.run(
        chunk.bind(parameters, prepared),
        prefix,
        inputs=dn.func.RolloutInput(intra=drive[2:]),
        steps=3,
        callbacks=callbacks,
        callback_state=resumed,
    )
    recorded = {
        name: torch.cat(
            (first["callbacks"]["trace"][name], second["callbacks"]["trace"][name])
        )
        for name in ("v", "t", "i_membrane")
    }
    projection = torch.cat((first["callbacks"]["lfp"], second["callbacks"]["lfp"]))
    _close(actual, expected)
    _close(recorded, expected_aux["callbacks"]["trace"])
    _close(projection, expected_aux["callbacks"]["lfp"], atol=1e-18)
    _close(second["callbacks"]["energy"], expected_aux["callbacks"]["energy"])
    assert recorded["v"].shape[0] == 6
    assert second["callbacks"]["energy"] > first["callbacks"]["energy"] > 0
    returned = second["callbacks"].state.carries["lfp"]
    torch.testing.assert_close(returned, strided)
    assert not torch._C._is_alias_of(returned, strided)
    targets = {
        "g": parameters[g_name],
        "field0": fields[0],
        "field1": fields[1],
        "drive": drive,
    }
    reference_loss = (
        expected_aux["callbacks"]["lfp"] * 1e6
    ).square().mean() + 1e-3 * expected_aux["callbacks"]["energy"]
    loss = (projection * 1e6).square().mean() + 1e-3 * second["callbacks"]["energy"]
    reference = _gradients(reference_loss, targets, retain_graph=True)
    _close_gradients(_gradients(loss, targets), reference)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_population_scan_cache_reuses_fresh_view_operands_without_retaining_examples(
    backend,
):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    captures = []
    if backend == "aot_eager":
        from torch._dynamo.backends.registry import lookup_backend

        compiler = lookup_backend(backend)

        def counted(graph, examples):
            captures.append(None)
            return compiler(graph, examples)

        selected = counted
    else:
        selected = backend
    options = (
        {"cpp_wrapper": True, "compile_threads": 1} if backend == "inductor" else None
    )
    chunk = functional.compile_rollout_chunk(
        STEPS,
        execution="scan",
        backend=selected,
        **({"options": options} if options else {}),
    )
    references, saved = _plain_pair(functional, tensors, chunk, (True, True))
    warm_captures = len(captures)
    later_references, changed = _plain_pair(
        functional, tensors, chunk, (True, True), offset=0.0002
    )
    assert not torch.equal(saved[0]["integrator"]["v"], changed[0]["integrator"]["v"])
    if backend == "aot_eager":
        assert warm_captures > 0 and len(captures) == warm_captures
    gc.collect()
    assert all(
        reference() is None
        for reference in (*references.values(), *later_references.values())
    )
