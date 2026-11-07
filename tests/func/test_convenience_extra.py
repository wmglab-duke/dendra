"""Convenience runners bind extracellular contacts without hiding trainable leaves."""

from __future__ import annotations

import inspect
import math

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _module_snapshot,
)
from tests.func.test_membrane_current import (
    CHANNELS,
    _close,
)
from tests.func.test_membrane_current import _loss as _membrane_loss
from tests.func.test_membrane_current import _model as _membrane_model
from tests.func.test_membrane_current import (
    _observable,
)
from tests.func.test_stimulation import (
    _assert_module_unchanged,
)
from tests.func.test_stimulation import _module_snapshot as _waveform_snapshot
from tests.func.test_stimulation import (
    _MutatingPureWaveform,
    _StochasticPureWaveform,
    _UnmarkedCustomWaveform,
)

DT = 0.01
STEPS = 4
PREFIX = "stimulation.extra."
pytestmark = pytest.mark.cpu


@pytest.fixture(params=(1, 4), ids=("threads1", "threads4"))
def cpu_threads(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    try:
        yield request.param
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _clone(value):
    return pytree.tree_map(lambda tensor: tensor.detach().clone(), value)


def _model(
    kind="path",
    *,
    device="cpu",
    dtype=torch.float64,
    batch=(2,),
    shell_resistance_scale=1.0,
):
    model = _membrane_model(kind, device=device, dtype=dtype, batch=None)
    if batch:
        model.batch(*batch)
    if shell_resistance_scale != 1.0:
        components = (
            model.populations.values() if kind.startswith("multi_") else (model,)
        )
        for component in components:
            if hasattr(component, "xraxial"):
                component.xraxial.mul_(shell_resistance_scale)
    model.initialize()
    model.train()
    model.t.fill_(0.175)
    return model


def _extra(model):
    options = {"device": model.device(), "dtype": model.dtype()}
    count = model.v.numel()
    first = torch.linspace(-1.25, 1.75, count, **options).reshape(model.shape)
    second = torch.linspace(0.8, -0.6, count, **options).reshape(model.shape)
    with dn.ctx(REQUIRE_GRAD=1, DEVICE=model.device(), DTYPE=model.dtype()):
        sine = dn.sin(amp=0.18, freq=0.9, phase=0.2)
        cosine = dn.cos(amp=-0.11, freq=0.55, phase=-0.15)
    return [(first, sine), (second, cosine)]


def _selected(name):
    return name.startswith(PREFIX) and name.endswith(
        (".field", ".amp", ".freq", ".phase")
    )


def _runner(
    model,
    extra=None,
    *,
    backend="eager",
    execution="scan",
    checkpoint_every=None,
    steps=STEPS,
):
    # Resolve the same exact names as independent_parameters, rather than
    # assuming that a shared Waveform is copied once per contact.
    _, tensors = dn.func.make_functional(model, dt=DT, extra=extra)
    trainable = tuple(name for name in tensors.parameters if _selected(name)) + tuple(
        f"integrator.mech.mechanisms.hh.{name}_param" for name in CHANNELS
    )
    keywords = {}
    if backend == "inductor":
        keywords = {
            "fullgraph": True,
            "options": {
                "cpp_wrapper": model.device().type == "cpu",
                "compile_threads": 1,
            },
        }
    return dn.compile(
        model,
        dt=DT,
        steps=steps,
        extra=extra,
        execution=execution,
        callbacks={"trace": dn.func.Recorder(tuple(_observable(model)) + ("t",))},
        inference=execution == "while",
        checkpoint_every=checkpoint_every,
        trainable=trainable,
        backend=backend,
        **keywords,
    )


def _arguments(runner):
    parameters = _clone(dict(runner.parameters))
    targets = {}
    for name in CHANNELS:
        key = f"integrator.mech.mechanisms.hh.{name}_param"
        targets[name] = parameters[key].requires_grad_()
    for name, value in parameters.items():
        if _selected(name):
            targets[name] = value.requires_grad_()
    state = _clone(runner.initial_state)
    voltage = state["integrator"]["v"].requires_grad_()
    targets["initial_voltage"] = voltage
    if "vc" in state["integrator"]:
        vc = state["integrator"]["vc"]
        state["integrator"]["vc"] = torch.cat(
            (voltage[..., None] + vc[..., 1:2], vc[..., 1:]), dim=-1
        )
    amplitude = voltage.new_ones((), requires_grad=True)
    targets["stimulus"] = amplitude
    drive = torch.linspace(
        -2.0e-9,
        3.0e-9,
        runner._steps * voltage.numel(),
        device=voltage.device,
        dtype=voltage.dtype,
    ).reshape(runner._steps, *voltage.shape)
    return parameters, state, dn.func.RolloutInput(intra=amplitude * drive), targets


def _recorded(result):
    return dict(result[1]["callbacks"]["trace"])


def _loss(recorded):
    loss = _membrane_loss(recorded)
    if "vc" in recorded:
        # Finite extracellular stimulation is directly observable in the shell
        # potentials, not only through a potentially cancelling membrane loss.
        # Nonuniform spatial weights expose order and field-shape derivatives;
        # a linear term retains signal even near the initially zero shells.
        shells = recorded["vc"][..., 1:]
        weights = torch.linspace(
            0.7, 1.3, shells.shape[-2], device=shells.device, dtype=shells.dtype
        )
        loss = loss + 0.1 * (weights.unsqueeze(-1) * shells).mean()
    return loss


def _run_bound(runner, parameters, state, inputs):
    return _recorded(runner(state, parameters=parameters, inputs=inputs))


def _assembled_ve(parameters, state, steps):
    # Deliberately independent of FunctionalExtra and its waveform bank. These
    # two built-in waveforms have delay=0 and infinite off times; all sampling
    # times are positive, away from their hard timing gates.
    time = state["clock"]["t"]
    template = parameters[f"{PREFIX}contacts.0.field"]
    dt = template.new_tensor(DT).to(time)
    fields = []
    for _ in range(steps):
        sampled_time = time.to(device=template.device, dtype=template.dtype)
        value = None
        for index, function in enumerate((torch.sin, torch.cos)):
            prefix = f"{PREFIX}waveforms.{index}."
            carrier = function(
                2.0 * math.pi * parameters[prefix + "freq"] * sampled_time
                + parameters[prefix + "phase"]
            )
            contribution = (
                parameters[f"{PREFIX}contacts.{index}.field"]
                * parameters[prefix + "amp"]
                * carrier
            )
            value = contribution if value is None else value + contribution
        fields.append(value)
        time = time + dt
    return torch.stack(fields)


def _explicit_reference(model, runner, parameters, state, inputs):
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ordinary_parameters = {name: parameters[name] for name in tensors.parameters}
    prepared = functional.prepare(ordinary_parameters, tensors.constants)
    callbacks = functional.make_callbacks(
        {"trace": dn.func.Recorder(tuple(_observable(model)) + ("t",))}
    )
    return _recorded(
        dn.func.run(
            functional.bind(ordinary_parameters, prepared),
            state,
            steps=runner._steps,
            inputs=dn.func.RolloutInput(
                ve=_assembled_ve(parameters, state, runner._steps), intra=inputs.intra
            ),
            callbacks=callbacks,
        )
    )


def _bound_reference(runner, parameters, state, inputs):
    functional = runner.functional
    prepared = functional.prepare(parameters, runner.constants)
    return _recorded(
        dn.func.run(
            functional.bind(parameters, prepared),
            state,
            steps=runner._steps,
            inputs=inputs,
            callbacks=runner._callbacks,
        )
    )


def _imperative_reference(model, extra, *, steps=STEPS):
    history = [{**_observable(model), "t": model.t.clone()}]
    for _ in range(steps):
        model.step(dt=DT, extra=extra)
        history.append({**_observable(model), "t": model.t.clone()})
    return {
        name: torch.stack([frame[name] for frame in history]) for name in history[0]
    }


def _gradients(recorded, targets):
    gradients = torch.autograd.grad(_loss(recorded), tuple(targets.values()))
    result = dict(zip(targets, gradients, strict=True))
    for name, value in result.items():
        assert torch.isfinite(value).all(), name
        assert torch.count_nonzero(value) > 0, name
    return result


def _exercise(
    kind="path",
    *,
    device="cpu",
    dtype=torch.float64,
    backend="eager",
    execution="scan",
    checkpoint_every=None,
    shell_resistance_scale=1.0,
):
    model = _model(
        kind, device=device, dtype=dtype, shell_resistance_scale=shell_resistance_scale
    )
    extra = _extra(model)
    source = _module_snapshot(model)
    waveforms = [_waveform_snapshot(waveform) for _, waveform in extra]
    fields = [field.clone() for field, _ in extra]
    runner = _runner(
        model,
        extra,
        backend=backend,
        execution=execution,
        checkpoint_every=checkpoint_every,
    )
    parameters, state, inputs, targets = _arguments(runner)
    actual = _run_bound(runner, parameters, state, inputs)
    gradients = _gradients(actual, targets) if execution == "scan" else {}
    reference_parameters, reference_state, reference_inputs, reference_targets = (
        _arguments(runner)
    )
    expected = _explicit_reference(
        model, runner, reference_parameters, reference_state, reference_inputs
    )
    _close(actual, expected, dtype=dtype)
    if execution == "scan":
        _close(gradients, _gradients(expected, reference_targets), dtype=dtype)
    else:
        assert not actual["v"].requires_grad
    ordinary_parameters, ordinary_state, ordinary_inputs, _ = _arguments(runner)
    _close(
        actual,
        _bound_reference(runner, ordinary_parameters, ordinary_state, ordinary_inputs),
        dtype=dtype,
    )
    assert actual["v"].shape == (STEPS + 1, *model.shape)
    assert actual["i_membrane"].shape == actual["v"].shape
    if hasattr(model, "vc"):
        assert actual["vc"].shape == (*actual["v"].shape, 3)
    _assert_module_snapshot(model, source)
    for (field, waveform), saved_field, saved_waveform in zip(
        extra, fields, waveforms, strict=True
    ):
        torch.testing.assert_close(field, saved_field, rtol=0.0, atol=0.0)
        assert field.grad is None
        _assert_module_unchanged(waveform, saved_waveform)
        assert all(parameter.grad is None for parameter in waveform.parameters())
    return actual, gradients


@pytest.mark.parametrize("kind", ("path", "block_path", "multi_scalar", "multi_block"))
@pytest.mark.parametrize(
    ("execution", "checkpoint_every"), (("scan", None), ("scan", 2), ("while", None))
)
def test_extra_full_traces_and_all_model_and_contact_gradients(
    kind, execution, checkpoint_every, cpu_threads
):
    _exercise(kind, execution=execution, checkpoint_every=checkpoint_every)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ("path", "multi_block"))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_real_cpp_extra_scan_and_while(kind, execution, cpu_threads):
    _exercise(
        kind,
        backend="inductor",
        execution=execution,
        checkpoint_every=2 if execution == "scan" else None,
    )


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_extra_matches_actual_imperative_steps(execution, cpu_threads):
    model = _model()
    runner = _runner(model, _extra(model), execution=execution)
    reference = _model()
    with torch.no_grad():
        expected = _imperative_reference(reference, _extra(reference))
    _close(_recorded(runner()), expected)


def test_extra_is_explicit_and_none_preserves_default_behavior():
    for factory in (dn.compile, dn.func.CompiledRunner):
        assert inspect.signature(factory).parameters["extra"].default is None
    model = _model()
    keywords = {
        "dt": DT,
        "steps": STEPS,
        "backend": "eager",
        "callbacks": {"trace": dn.func.Recorder("v")},
    }
    omitted = dn.compile(model, **keywords)()
    explicit = dn.compile(model, extra=None, **keywords)()
    torch.testing.assert_close(
        _recorded(omitted)["v"], _recorded(explicit)["v"], rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_bound_extra_conflicts_with_explicit_ve(execution):
    model = _model()
    runner = _runner(model, tuple(_extra(model)[0]), execution=execution)
    inputs = dn.func.RolloutInput(ve=model.v.new_zeros((STEPS, *model.shape)))
    with pytest.raises(ValueError, match="ve"):
        runner(inputs=inputs)


def test_runner_extra_leaves_are_independent_optimizer_parameters(cpu_threads):
    model = _model()
    extra = _extra(model)
    for field, _ in extra:
        field.requires_grad_()
    source = _module_snapshot(model)
    waveforms = [_waveform_snapshot(waveform) for _, waveform in extra]
    runner = _runner(model, extra, checkpoint_every=2)
    selected = {
        name: value for name, value in runner.parameters.items() if _selected(name)
    }
    assert len(selected) == 8  # Two fields plus amp/freq/phase per waveform.
    assert all(
        isinstance(value, torch.nn.Parameter) and value.is_leaf
        for value in selected.values()
    )
    for index, (field, waveform) in enumerate(extra):
        assert (
            runner.parameters[f"{PREFIX}contacts.{index}.field"].data_ptr()
            != field.data_ptr()
        )
        for leaf in ("amp", "freq", "phase"):
            assert (
                runner.parameters[f"{PREFIX}waveforms.{index}.{leaf}"].data_ptr()
                != getattr(waveform, leaf).data_ptr()
            )
    optimizer = torch.optim.SGD(selected.values(), lr=1.0e-3)
    first = None
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        recorded = _recorded(runner())
        if first is None:
            first = recorded["v"].detach().clone()
        _loss(recorded).backward()
        assert all(
            value.grad is not None and torch.isfinite(value.grad).all()
            for value in selected.values()
        )
        optimizer.step()
    assert not torch.equal(_recorded(runner())["v"], first)
    _assert_module_snapshot(model, source)
    for (field, waveform), saved in zip(extra, waveforms, strict=True):
        assert field.grad is None
        _assert_module_unchanged(waveform, saved)


@pytest.mark.parametrize("leaf", ("contacts.0.field", "waveforms.0.amp"))
def test_checkpoint_replay_rejects_mutated_bound_extra_parameters(
    monkeypatch, leaf, cpu_threads
):
    import dendra.func._runners as runners

    refreshes = []
    original_refresh = runners._refresh_checkpoint_execution

    def refresh(execution, state):
        refreshes.append(None)
        return original_refresh(execution, state)

    monkeypatch.setattr(runners, "_refresh_checkpoint_execution", refresh)
    model = _model()
    runner = _runner(model, _extra(model), backend="aot_eager", checkpoint_every=2)
    loss = _loss(_recorded(runner()))
    forward_refreshes = len(refreshes)
    assert forward_refreshes == 2
    target = runner.parameters[PREFIX + leaf]
    with torch.no_grad():
        target.add_(0.1)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="parameters changed after checkpoint forward",
    ):
        torch.autograd.grad(loss, target)
    # AOT's saved-tensor backward requests replay. Raw eager scan may retain
    # its own autograd context and would not exercise this guard reliably.
    assert len(refreshes) > forward_refreshes


def test_checkpointed_extra_optimizer_updates_after_backward_allow_new_forward():
    model = _model()
    runner = _runner(model, _extra(model), backend="aot_eager", checkpoint_every=2)
    selected = {
        name: value for name, value in runner.parameters.items() if _selected(name)
    }
    optimizer = torch.optim.SGD(selected.values(), lr=1.0e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        _loss(_recorded(runner())).backward()
        before = {name: value.detach().clone() for name, value in selected.items()}
        assert all(
            value.grad is not None and torch.isfinite(value.grad).all()
            for value in selected.values()
        )
        optimizer.step()
        assert any(
            not torch.equal(value, before[name]) for name, value in selected.items()
        )


def test_parameter_overrides_preserve_upstream_field_generator_gradients(cpu_threads):
    model = _model()
    runner = _runner(model, _extra(model), checkpoint_every=2)

    def evaluate(*, reference):
        parameters, state, inputs, _ = _arguments(runner)
        template = parameters[f"{PREFIX}contacts.0.field"]
        coordinate = template.new_tensor(0.2, requires_grad=True)
        scale = template.new_tensor(-0.1, requires_grad=True)
        raw_amplitude = template.new_tensor(0.3, requires_grad=True)
        position = torch.linspace(
            -1.0, 1.0, template.numel(), device=template.device, dtype=template.dtype
        ).reshape_as(template)
        parameters[f"{PREFIX}contacts.0.field"] = scale.exp() * torch.exp(
            -(position - coordinate).square()
        )
        parameters[f"{PREFIX}waveforms.0.amp"] = raw_amplitude.sigmoid() * 0.3
        assert not parameters[f"{PREFIX}contacts.0.field"].is_leaf
        assert not parameters[f"{PREFIX}waveforms.0.amp"].is_leaf
        recorded = (
            _explicit_reference(model, runner, parameters, state, inputs)
            if reference
            else _run_bound(runner, parameters, state, inputs)
        )
        targets = {
            "coordinate": coordinate,
            "scale": scale,
            "raw_amplitude": raw_amplitude,
        }
        return recorded, _gradients(recorded, targets)

    actual, gradients = evaluate(reference=False)
    expected, expected_gradients = evaluate(reference=True)
    _close(actual, expected)
    _close(gradients, expected_gradients)


def test_inference_parameter_edits_refresh_bound_extra_without_repreparing():
    model = _model()
    runner = _runner(model, _extra(model), execution="while")
    first = _recorded(runner())["v"]
    prepared = runner._prepared
    with torch.no_grad():
        runner.parameters[f"{PREFIX}contacts.0.field"].mul_(1.4)
        runner.parameters[f"{PREFIX}waveforms.0.amp"].mul_(1.3)
    second = _recorded(runner())["v"]
    assert runner._prepared is prepared
    assert not torch.equal(first, second)


@pytest.mark.parametrize("target", ("field", "amp", "freq", "phase", "initial_voltage"))
def test_contact_and_initial_voltage_directional_finite_differences(target):
    model = _model()
    runner = _runner(model, _extra(model), checkpoint_every=2)
    parameters, state, inputs, targets = _arguments(runner)
    key = (
        "initial_voltage"
        if target == "initial_voltage"
        else next(
            name
            for name in targets
            if name.startswith(PREFIX) and name.endswith("." + target)
        )
    )
    leaf = targets[key]
    direction = torch.linspace(
        0.7, 1.3, leaf.numel(), dtype=leaf.dtype, device=leaf.device
    ).reshape_as(leaf)
    gradient = _gradients(_run_bound(runner, parameters, state, inputs), targets)[key]
    analytic = (gradient * direction).sum()
    epsilon = 1.0e-4
    values = []
    for sign in (-1.0, 1.0):
        changed_parameters, changed_state, changed_inputs, _ = _arguments(runner)
        replacement = leaf.detach() + sign * epsilon * direction
        if key == "initial_voltage":
            changed_state["integrator"]["v"] = replacement
        else:
            changed_parameters[key] = replacement
        values.append(
            _loss(
                _explicit_reference(
                    model, runner, changed_parameters, changed_state, changed_inputs
                )
            ).detach()
        )
    numerical = (values[1] - values[0]) / (2.0 * epsilon)
    torch.testing.assert_close(analytic, numerical, rtol=2.0e-5, atol=2.0e-8)


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_continuation_samples_pulse_at_recurrent_clock_boundary(execution, cpu_threads):
    # start + arange*dt misses the final pulse, but the repeated accepted clock
    # used by imperative stepping reaches it (regression from runner clocks).
    model = _model(dtype=torch.float32, batch=())
    model.t.fill_(1000.0)
    field = torch.linspace(-1.0, 1.0, model.v.numel()).reshape(model.shape)
    with dn.ctx(REQUIRE_GRAD=1, DTYPE=torch.float32):
        pulse = dn.mono_rect(amp=1.0, delay=1000.070068359375, pw=0.001)
    extra = (field, pulse)
    runner = dn.compile(
        model,
        dt=DT,
        steps=4,
        extra=extra,
        execution=execution,
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        inference=execution == "while",
        checkpoint_every=2 if execution == "scan" else None,
        backend="eager",
    )
    first = runner()
    second = runner(first[0], callback_state=first[1]["callbacks"].state)
    joined = {
        name: torch.cat((_recorded(first)[name], _recorded(second)[name]))
        for name in ("v", "t")
    }
    reference = _model(dtype=torch.float32, batch=())
    reference.t.fill_(1000.0)
    with torch.no_grad():
        expected = _imperative_reference(reference, (field, pulse), steps=8)
        unstimulated = _model(dtype=torch.float32, batch=())
        unstimulated.t.fill_(1000.0)
        baseline = _imperative_reference(unstimulated, None, steps=8)
    _close(joined, {name: expected[name] for name in joined}, dtype=torch.float32)
    assert joined["v"].shape[0] == 9
    assert not torch.equal(joined["v"][-1], baseline["v"][-1])
    if execution == "scan":
        amplitude = runner.parameters[f"{PREFIX}waveforms.0.amp"]
        # trainable defaults are empty: an explicit live leaf enables pulse AD.
        amplitude.requires_grad_()
        repeated = runner()
        resumed = runner(repeated[0], callback_state=repeated[1]["callbacks"].state)
        (gradient,) = torch.autograd.grad(
            _recorded(resumed)["v"].square().mean(), amplitude
        )
        assert torch.isfinite(gradient).all() and gradient.abs().item() > 0.0


@pytest.mark.parametrize(
    "waveform_type",
    (_UnmarkedCustomWaveform, _MutatingPureWaveform, _StochasticPureWaveform),
)
def test_extra_rejects_undeclared_mutating_and_stochastic_waveforms(waveform_type):
    model = _model()
    with (
        dn.ctx(CHECK_PURITY=True),
        pytest.raises(dn.func.FunctionalizationError),
    ):
        dn.compile(
            model,
            dt=DT,
            steps=4,
            extra=(torch.ones_like(model.v), waveform_type()),
            backend="eager",
        )


def test_bound_tensor_temporal_extra_still_requires_explicit_runtime_input():
    model = _model()
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="Tensor-temporal extra.*pass it at runtime",
    ):
        dn.compile(
            model,
            dt=DT,
            steps=4,
            extra=(torch.ones_like(model.v), model.v.new_ones(4)),
            backend="eager",
        )


def test_bound_extra_cannot_share_a_waveform_with_registered_intra():
    model = _model()
    field, waveform = _extra(model)[0]
    model[..., 1].inject(waveform)
    with pytest.raises(
        dn.func.FunctionalizationError, match="same Waveform object cannot be shared"
    ):
        dn.compile(model, dt=DT, steps=4, extra=(field, waveform), backend="eager")


def test_shared_waveform_contacts_keep_one_canonical_trainable_parameter():
    model = _model()
    extra = _extra(model)
    waveform = extra[0][1]
    runner = _runner(model, [(extra[0][0], waveform), (extra[1][0], waveform)])
    amplitudes = [
        name
        for name in runner.parameters
        if name.startswith(PREFIX) and name.endswith(".amp")
    ]
    assert len(amplitudes) == 1
    assert (
        sum(
            name.startswith(PREFIX) and name.endswith(".field")
            for name in runner.parameters
        )
        == 2
    )
    recorded = _recorded(runner())
    (gradient,) = torch.autograd.grad(_loss(recorded), runner.parameters[amplitudes[0]])
    assert torch.isfinite(gradient).all() and gradient.abs().item() > 0.0
