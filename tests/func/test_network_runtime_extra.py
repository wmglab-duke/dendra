"""Call-local Network extracellular recipes become bounded explicit ve inputs."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_network import DTYPE, _clone, _close
from tests.func.test_network_extra import (
    DT,
    _BufferedSine,
    _native,
    _network,
    _observe,
    _recorders,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True, DTYPE=DTYPE, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _field(network, *, requires_grad=False):
    return (
        torch.linspace(-1.25, 1.75, network.left.v.numel(), dtype=network.left.dtype())
        .reshape(network.left.shape)
        .requires_grad_(requires_grad)
    )


def _runner(network, *, steps=5, **options):
    return dn.compile(
        network,
        dt=DT,
        steps=steps,
        extra=dn.VarInput(),
        chunk_steps=2,
        callbacks=_recorders(network),
        backend="aot_eager",
        **options,
    )


def _loss(state):
    return sum(
        values["integrator"]["v"].square().mean()
        for values in state["populations"].values()
    )


def _native_contacts(network, extra, steps):
    """Sample each authored contact independently before native Network steps."""
    frames = [_observe(network)]
    unit = dn.constant(value=1.0)
    for _ in range(steps):
        selected = {}
        for name, specification in ({} if extra is None else extra).items():
            contacts = (
                (specification,) if isinstance(specification, tuple) else specification
            )
            field = sum(spatial * waveform(network.t) for spatial, waveform in contacts)
            selected[name] = (field, unit)
        network.step(extra=selected)
        frames.append(_observe(network))
    return {name: torch.stack([frame[name] for frame in frames]) for name in frames[0]}


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_runtime_network_recipes_change_between_calls_and_omitted_slots_are_zero(
    execution,
):
    network = _network(batch=None)
    runner = _runner(network, execution=execution)
    snapshot = _clone(runner.initial_state, detached=True)
    field = _field(network)
    first = {"left": (field, _BufferedSine())}
    second = {
        "left": [
            (field * -0.7, dn.mono_rect(amp=0.3, delay=0.18, pw=0.02)),
            (field.flip(-1), _BufferedSine(amp=-0.11)),
        ],
        "right": (field, _BufferedSine(freq=0.3)),
    }
    for extra in (first, second, {}, None):
        native = _network(batch=None)
        with torch.no_grad():
            expected_trace = _native_contacts(native, extra, 5)
        actual, auxiliary = runner(extra=extra)
        _close(actual, runner.functional.extract(native).state)
        _close(auxiliary["callbacks"]["trace"], expected_trace)
        _close(runner.initial_state, snapshot)
    assert runner.kernel.steps == 2
    assert not any("stimulation.extra" in name for name in runner.named_parameters)


def test_runtime_network_mapping_can_mix_bound_and_call_local_populations():
    network, native = _network(batch=None), _network(batch=None)
    field = _field(network)
    fixed = (field, _BufferedSine(amp=-0.1))
    supplied = (field.flip(-1), _BufferedSine(amp=0.2))
    runner = dn.compile(
        network,
        dt=DT,
        steps=3,
        extra={"left": dn.VarInput(), "right": fixed},
        chunk_steps=2,
        backend="aot_eager",
    )
    actual, _ = runner(extra={"left": supplied})
    with torch.no_grad():
        _native(native, {"left": supplied, "right": fixed}, 3)
    _close(actual, runner.functional.extract(native).state)
    assert runner.functional.populations["right"].extra.enabled
    assert not runner.functional.populations["left"].extra.enabled
    with pytest.raises(ValueError, match="without runtime slots"):
        runner(extra={"right": supplied})


def test_runtime_network_slots_require_explicit_extra_and_reject_explicit_ve():
    network = _network(batch=None)
    runner = _runner(network, steps=2)
    with pytest.raises(TypeError, match="extra is required"):
        runner()
    with pytest.raises(TypeError, match="extra must map"):
        runner(extra=())
    with pytest.raises(ValueError, match="without runtime slots"):
        runner(extra={"missing": None})
    ve = torch.zeros((2, *network.left.shape), dtype=DTYPE)
    inputs = dn.func.NetworkRolloutInput(
        populations={"left": dn.func.RolloutInput(ve=ve)}
    )
    for extra in ({}, None, {"left": (_field(network), _BufferedSine())}):
        with pytest.raises(ValueError, match="explicit ve.*runtime extra"):
            runner(inputs=inputs, extra=extra)
    bound = dn.compile(network, dt=DT, steps=2, backend="aot_eager")
    with pytest.raises(TypeError, match="requires slots"):
        bound(extra={})


@pytest.mark.parametrize("bound_extra", (False, True))
def test_nonruntime_network_preserves_the_public_run_boundary(monkeypatch, bound_extra):
    network = _network(batch=None)
    runner = dn.compile(
        network,
        dt=DT,
        steps=2,
        extra={"left": (_field(network), _BufferedSine())} if bound_extra else None,
        backend="aot_eager",
    )
    calls = []

    def public_run(*args, **kwargs):
        calls.append((args, kwargs))
        return args[2], {}

    monkeypatch.setattr(runner.functional, "run", public_run)
    final, auxiliary = runner()
    assert final is runner.initial_state
    assert set(auxiliary) == {"continuation"}
    assert isinstance(auxiliary["continuation"], dn.func.Continuation)
    _close(auxiliary["continuation"].state, final)
    assert len(calls) == 1
    assert calls[0][1]["steps"] == 2
    assert "_runtime_extra" not in calls[0][1]


@pytest.mark.parametrize("checkpoint_every", (None, 3))
def test_network_runtime_extra_shared_field_and_waveform_gradients_match_explicit_ve(
    checkpoint_every,
):
    network = _network(batch=None)
    runner = _runner(
        network, steps=5, inference=False, checkpoint_every=checkpoint_every
    )

    def evaluate(reference):
        coordinate = torch.tensor(0.2, dtype=DTYPE, requires_grad=True)
        positions = _field(network)
        field = torch.exp(-(positions - coordinate).square())
        waveform = _BufferedSine()
        extra = {name: (field, waveform) for name in network.populations}
        if reference:
            clock = runner.initial_state["clock"]
            times = (
                clock["origin"]
                + (clock["step"] + torch.arange(5)).to(clock["origin"].dtype) * DT
            )
            amplitude = (
                waveform.amp
                * torch.sin(2 * torch.pi * waveform.freq * times + waveform.phase)
                + waveform.offset
            )
            ve = amplitude.reshape(5, *((1,) * field.ndim)) * field
            inputs = dn.func.NetworkRolloutInput(
                populations={
                    name: dn.func.RolloutInput(ve=ve) for name in network.populations
                }
            )
            final, _ = runner.functional.prepare_and_rollout(
                runner.parameters,
                runner.constants,
                runner.initial_state,
                inputs,
                steps=5,
            )
        else:
            final, _ = runner(extra=extra)
        gradients = torch.autograd.grad(
            _loss(final), (coordinate, waveform.amp, waveform.freq)
        )
        return final, gradients

    expected, expected_gradients = evaluate(True)
    for _ in range(2):
        final, gradients = evaluate(False)
        _close(final, expected)
        _close(gradients, expected_gradients)
        assert all(
            torch.isfinite(value).all() and value.abs().sum() for value in gradients
        )


@pytest.mark.parametrize("leaf", ("field", "waveform", "waveform_rebind"))
def test_network_runtime_extra_checkpoint_rejects_mutated_source_tensors(leaf):
    network = _network(batch=None)
    runner = _runner(network, inference=False, checkpoint_every=3)
    field, waveform = _field(network, requires_grad=True), _BufferedSine()
    final, _ = runner(extra={"left": (field, waveform)})
    with torch.no_grad():
        if leaf == "waveform_rebind":
            waveform.amp = torch.nn.Parameter(waveform.amp.detach().clone() + 0.01)
        else:
            (field if leaf == "field" else waveform.amp).add_(0.01)
    if leaf == "waveform":
        # Checkpoint may inspect saved operands before calling our replay guard.
        error = RuntimeError
        match = "runtime extra.*changed|modified by an inplace operation"
    else:
        error = dn.func.FunctionalizationError
        match = "runtime extra.*changed"
    with pytest.raises(error, match=match):
        _loss(final).backward()


def test_network_runtime_extra_assembly_is_bounded_and_replays_inside_checkpoints(
    monkeypatch,
):
    import dendra.func._network_rollout as module

    network = _network(batch=None)
    runner = _runner(network, steps=7, inference=False, checkpoint_every=5)
    original = module._assemble_extra_chunk
    widths = []

    def assemble(*args, **kwargs):
        widths.append(args[5])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_assemble_extra_chunk", assemble)
    waveform = _BufferedSine()
    final, _ = runner(extra={"left": (_field(network), waveform)})
    forward_calls = len(widths)
    assert widths and max(widths) <= 2
    _loss(final).backward()
    assert len(widths) > forward_calls
    assert max(widths) <= 2
    default = dn.compile(
        network, dt=DT, steps=130, extra=dn.VarInput(), backend="aot_eager"
    )
    assert default.kernel.steps == 64


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_runtime_network_float32_pulse_uses_canonical_clock_across_continuation(
    execution,
):
    network, native = (_network(dtype=torch.float32, batch=None) for _ in range(2))
    for value in (network, native):
        value._reset_runtime_clock(1000.0)
    with dn.ctx(DTYPE=torch.float32):
        pulse = dn.mono_rect(
            amp=1.0,
            delay=float(torch.tensor(1000.07, dtype=torch.float32)),
            pw=0.001,
        )
    extra = {"left": (_field(network), pulse)}
    runner = _runner(network, steps=4, execution=execution)
    prefix, first = runner(extra=extra)
    final, second = runner(prefix, extra=extra, callback_state=first["callbacks"].state)
    trace = {
        name: torch.cat((first["callbacks"]["trace"][name], value))
        for name, value in second["callbacks"]["trace"].items()
    }
    with torch.no_grad():
        expected = _native(native, extra, 8)
    for name, value in trace.items():
        torch.testing.assert_close(value, expected[name], rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(trace["t"], expected["t"], rtol=0, atol=0)
    assert final["clock"]["step"].item() == 8
