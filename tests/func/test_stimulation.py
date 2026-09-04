"""Functional stimulation contracts shared with imperative Population execution."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from dendra.models.stim.waveform import Waveform
from dendra.units import nA

DTYPE = torch.float64
DT = 0.025
START = 0.175
STEPS = 4
INTRA_PREFIX = "stimulation.intra."
EXTRA_PREFIX = "stimulation.extra."


class _ModeAwarePureWaveform(Waveform):
    FUNCTIONAL_PURE = True
    Waveform.PARAMETERP(value=0.2)

    def fn(self, t):
        scale = 2.0 if self.training else 3.0
        return torch.ones_like(t) * scale * self.value()


class _MutatingPureWaveform(Waveform):
    FUNCTIONAL_PURE = True

    def __init__(self):
        super().__init__()
        self.register_buffer("counter", torch.zeros(()))

    def fn(self, t):
        self.counter.add_(1)
        return torch.ones_like(t)


class _StochasticPureWaveform(Waveform):
    FUNCTIONAL_PURE = True

    def fn(self, t):
        return torch.rand_like(t)


class _UnmarkedCustomWaveform(Waveform):
    def fn(self, t):
        return torch.ones_like(t)


class _CallableStatePureWaveform(Waveform):
    FUNCTIONAL_PURE = True

    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def fn(self, t):
        return self.operation(t)


def _model(diameters=(2.0, 2.5)):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE):
        model = dn.Unmyelinated(
            list(diameters),
            L=4.0,
            dx=1.0,
            v_init=-65.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(pas, g=1.0e-4, e=-65.0)
        model.initialize()
        model.t.fill_(START)
        model.train()
    return model


def _injected_model(style="single"):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE):
        model = dn.Unmyelinated(
            [2.0, 2.5],
            L=4.0,
            dx=1.0,
            v_init=-65.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(pas, g=1.0e-4, e=-65.0)

        if style == "single":
            waveforms = [
                dn.sin(
                    amp=0.2 * nA,
                    freq=0.7,
                    phase=0.3,
                    delay=0.0,
                )
            ]
            model[:, 1].inject(waveforms[0])
        elif style == "duplicate_index":
            waveforms = [dn.constant(value=0.2 * nA)]
            model[:, [1, 1]].inject(waveforms[0])
        elif style == "double_amplitude":
            waveforms = [dn.constant(value=0.4 * nA)]
            model[:, 1].inject(waveforms[0])
        elif style == "two_injections":
            waveforms = [
                dn.constant(value=0.2 * nA),
                dn.constant(value=0.2 * nA),
            ]
            model[:, 1].inject(waveforms[0])
            model[:, 1].inject(waveforms[1])
        else:  # pragma: no cover - test-helper guard
            raise ValueError(style)

        model.initialize()
        model.t.fill_(START)
        model.train()
    return model, waveforms


def _extra(shape, *, requires_grad=False):
    count = torch.tensor(shape).prod().item()
    field_a = torch.linspace(-1.25, 1.75, count, dtype=DTYPE).reshape(shape)
    field_b = torch.linspace(0.8, -0.6, count, dtype=DTYPE).reshape(shape)
    field_a.requires_grad_(requires_grad)
    field_b.requires_grad_(requires_grad)
    with dn.ctx(REQUIRE_GRAD=1, DTYPE=DTYPE):
        waveform_a = dn.sin(amp=0.18, freq=0.9, phase=0.2)
        waveform_b = dn.cos(amp=-0.11, freq=0.55, phase=-0.15)
    return [(field_a, waveform_a), (field_b, waveform_b)]


def _module_snapshot(module):
    def snapshot(values):
        return {
            name: (
                id(value),
                value.untyped_storage().data_ptr(),
                value._version,
                value.requires_grad,
                value.device,
                value.dtype,
                tuple(value.shape),
                value.detach().clone(),
            )
            for name, value in values
        }

    return {
        "parameters": snapshot(module.named_parameters(remove_duplicate=False)),
        "buffers": snapshot(module.named_buffers(remove_duplicate=False)),
        "training": tuple(
            (name, child.training) for name, child in module.named_modules()
        ),
        "reshaped_for_intra": getattr(module, "_reshaped_for_intra", None),
    }


def _assert_module_unchanged(module, expected):
    def assert_values(values, snapshots):
        assert set(values) == set(snapshots)
        for name, value in values.items():
            (
                identity,
                storage,
                version,
                requires_grad,
                device,
                dtype,
                shape,
                data,
            ) = snapshots[name]
            assert id(value) == identity
            assert value.untyped_storage().data_ptr() == storage
            assert value._version == version
            assert value.requires_grad == requires_grad
            assert value.device == device
            assert value.dtype == dtype
            assert tuple(value.shape) == shape
            assert torch.equal(value, data)

    assert_values(
        dict(module.named_parameters(remove_duplicate=False)),
        expected["parameters"],
    )
    assert_values(
        dict(module.named_buffers(remove_duplicate=False)),
        expected["buffers"],
    )
    assert (
        tuple((name, child.training) for name, child in module.named_modules())
        == expected["training"]
    )
    assert (
        getattr(module, "_reshaped_for_intra", None) == expected["reshaped_for_intra"]
    )


def _assert_state_matches_model(state, model, *, rtol=1.0e-12, atol=1.0e-12):
    torch.testing.assert_close(
        state["integrator"]["v"],
        model.v,
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(state["clock"]["t"], model.t, rtol=0.0, atol=0.0)


def _functional_state(model, *, steps=STEPS):
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    state, _auxiliary = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        steps=steps,
    )
    return state


def _bound_step(functional, tensors):
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    def step(state, inputs):
        return functional.step(tensors.parameters, prepared, state, inputs)

    return step


def _clone_mapping(values: Mapping[str, torch.Tensor]):
    return {name: value.detach().clone() for name, value in values.items()}


def test_make_functional_exposes_registered_waveforms_without_mutating_them():
    model, (waveform,) = _injected_model()
    snapshot = _module_snapshot(waveform)

    functional, tensors = dn.func.make_functional(model, dt=DT)

    assert functional.intra.enabled
    intra_names = {
        name.removeprefix(INTRA_PREFIX)
        for name in tensors.parameters
        if name.startswith(INTRA_PREFIX)
    }
    assert intra_names
    assert tensors.intra_parameters == {
        name: value
        for name, value in tensors.parameters.items()
        if name.startswith(INTRA_PREFIX)
    }
    assert not tensors.extra_parameters
    assert tensors.parameter_name("amp", within="intra").endswith(".amp")
    for leaf_name, expected in waveform.named_parameters():
        matches = [
            value
            for name, value in tensors.parameters.items()
            if name.startswith(INTRA_PREFIX) and name.endswith(f".{leaf_name}")
        ]
        assert len(matches) == 1
        torch.testing.assert_close(matches[0], expected)
    _assert_module_unchanged(waveform, snapshot)


def test_parameter_search_scopes_disambiguate_bound_intra_and_extra_leaves():
    model, _waveforms = _injected_model()
    functional, tensors = dn.func.make_functional(
        model,
        dt=DT,
        extra=_extra(model.shape),
    )

    intra_amp = tensors.parameter_name("waveforms.0.amp", within="intra")
    extra_amp = tensors.parameter_name("waveforms.0.amp", within="extra")
    assert intra_amp.startswith(INTRA_PREFIX)
    assert extra_amp.startswith(EXTRA_PREFIX)
    assert functional.intra.enabled
    assert functional.extra.enabled
    assert any(name.endswith(".field") for name in tensors.extra_parameters)
    with pytest.raises(KeyError, match="ambiguous"):
        tensors.parameter_name("waveforms.0.amp")


@pytest.mark.parametrize("entrypoint", ["step", "rollout"])
def test_registered_intra_is_applied_automatically_at_nonzero_time(entrypoint):
    functional_model, _ = _injected_model()
    imperative_model, _ = _injected_model()
    functional, tensors = dn.func.make_functional(functional_model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    if entrypoint == "step":
        state = tensors.state
        for _ in range(STEPS):
            state, _auxiliary = functional.step(
                tensors.parameters,
                prepared,
                state,
            )
    else:
        state, _auxiliary = functional.rollout(
            tensors.parameters,
            prepared,
            tensors.state,
            steps=STEPS,
        )

    for _ in range(STEPS):
        imperative_model.step(dt=DT)

    _assert_state_matches_model(state, imperative_model)


def test_registered_intra_accumulates_overlaps_and_duplicate_indices():
    states = {}
    for style in ("duplicate_index", "double_amplitude", "two_injections"):
        model, _waveforms = _injected_model(style)
        states[style] = _functional_state(model)

    expected = states["double_amplitude"]["integrator"]["v"]
    torch.testing.assert_close(
        states["duplicate_index"]["integrator"]["v"],
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        states["two_injections"]["integrator"]["v"],
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_registered_intra_amplitude_composes_with_jacrev_jacfwd_and_vmap():
    model, _waveforms = _injected_model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters = _clone_mapping(tensors.parameters)
    constants = _clone_mapping(tensors.constants)
    amp_name = next(
        name
        for name in parameters
        if name.startswith(INTRA_PREFIX) and name.endswith(".amp")
    )
    amplitude = parameters[amp_name]

    def response(amp):
        local_parameters = dict(parameters)
        local_parameters[amp_name] = amp
        state, _auxiliary = functional.prepare_and_rollout(
            local_parameters,
            constants,
            tensors.state,
            steps=2,
        )
        return state["integrator"]["v"]

    jacobian = torch.func.jacrev(response)(amplitude)
    assert jacobian.shape == model.shape
    assert torch.isfinite(jacobian).all()
    assert torch.count_nonzero(jacobian) > 0
    with torch_compiler_warning_context():
        forward_jacobian = torch.func.jacfwd(response)(amplitude)
    torch.testing.assert_close(
        forward_jacobian,
        jacobian,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    amplitudes = torch.stack((0.75 * amplitude, amplitude, 1.25 * amplitude))
    batched = torch.vmap(response)(amplitudes)
    assert batched.shape == (3, *model.shape)
    torch.testing.assert_close(
        batched[1], response(amplitude), rtol=1.0e-12, atol=1.0e-12
    )

    with torch_compiler_warning_context():
        compiled_jacobian = torch.compile(
            torch.func.jacrev(response),
            backend="aot_eager",
            fullgraph=True,
        )(amplitude)
    torch.testing.assert_close(compiled_jacobian, jacobian)


@pytest.mark.parametrize("entrypoint", ["step", "rollout"])
def test_raw_multicontact_extra_matches_imperative_at_nonzero_time(entrypoint):
    functional_model = _model()
    imperative_model = _model()
    functional_extra = _extra(functional_model.shape)
    imperative_extra = _extra(imperative_model.shape)
    snapshots = [_module_snapshot(waveform) for _field, waveform in functional_extra]
    functional, tensors = dn.func.make_functional(functional_model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    if entrypoint == "step":
        state = tensors.state
        for _ in range(STEPS):
            state, _auxiliary = functional.step(
                tensors.parameters,
                prepared,
                state,
                extra=functional_extra,
            )
    else:
        state, _auxiliary = functional.rollout(
            tensors.parameters,
            prepared,
            tensors.state,
            steps=STEPS,
            extra=functional_extra,
        )

    for _ in range(STEPS):
        imperative_model.step(dt=DT, extra=imperative_extra)

    _assert_state_matches_model(state, imperative_model)
    for (_field, waveform), snapshot in zip(functional_extra, snapshots, strict=True):
        _assert_module_unchanged(waveform, snapshot)


def test_raw_extra_preserves_field_and_waveform_gradients():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    extra = _extra(model.shape, requires_grad=True)
    field, waveform = extra[0]

    state, _auxiliary = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        steps=STEPS,
        extra=extra,
    )
    loss = state["integrator"]["v"].square().mean()
    field_gradient, amplitude_gradient = torch.autograd.grad(
        loss,
        (field, waveform.amp),
    )

    assert torch.isfinite(field_gradient).all()
    assert torch.count_nonzero(field_gradient) > 0
    assert torch.isfinite(amplitude_gradient).all()
    assert torch.count_nonzero(amplitude_gradient) > 0


def test_raw_tensor_temporal_extra_supports_step_and_make_extra_rollout():
    step_model = _model()
    step_reference = _model()
    field = torch.linspace(
        -1.0,
        1.0,
        step_model.v.numel(),
        dtype=DTYPE,
    ).reshape(step_model.shape)
    temporal = torch.linspace(0.1, 0.4, STEPS, dtype=DTYPE)
    functional, tensors = dn.func.make_functional(step_model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    stepped, _ = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
        extra=(field, temporal[:1]),
    )
    step_reference.step(dt=DT, extra=(field, temporal[:1]))
    _assert_state_matches_model(stepped, step_reference)

    rollout_model = _model()
    rollout_reference = _model()
    functional, tensors = dn.func.make_functional(rollout_model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    extra_plan = functional.make_extra((field, temporal))
    rolled_out, _ = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        steps=STEPS,
        extra=extra_plan,
    )
    rollout_reference.run(tstop=STEPS * DT, dt=DT, extra=(field, temporal))
    _assert_state_matches_model(rolled_out, rollout_reference)


@pytest.mark.parametrize("runner", ["run", "longrun"])
def test_host_runners_accept_the_same_raw_extra_spec_as_population(runner):
    functional_model = _model()
    imperative_model = _model()
    functional_extra = _extra(functional_model.shape)
    imperative_extra = _extra(imperative_model.shape)
    functional, tensors = dn.func.make_functional(functional_model, dt=DT)
    step = _bound_step(functional, tensors)
    duration = STEPS * DT

    if runner == "run":
        state, _auxiliary = dn.func.run(
            functional,
            step,
            tensors.state,
            tstop=duration,
            extra=functional_extra,
        )
        imperative_model.run(tstop=duration, dt=DT, extra=imperative_extra)
    else:
        state, _auxiliary = dn.func.longrun(
            functional,
            step,
            tensors.state,
            duration,
            2,
            extra=functional_extra,
        )
        imperative_model.longrun(
            tstop=duration,
            dt=DT,
            chunklength=2,
            extra=imperative_extra,
        )

    _assert_state_matches_model(state, imperative_model)


@pytest.mark.parametrize("runner", ["run", "longrun_checkpointed"])
def test_host_runners_accept_raw_tensor_temporal_extra(runner):
    functional_model = _model()
    imperative_model = _model()
    field = torch.linspace(
        -1.0,
        1.0,
        functional_model.v.numel(),
        dtype=DTYPE,
    ).reshape(functional_model.shape)
    temporal = torch.linspace(0.1, 0.4, STEPS, dtype=DTYPE)
    functional, tensors = dn.func.make_functional(functional_model, dt=DT)
    step = _bound_step(functional, tensors)

    if runner == "run":
        state, _ = dn.func.run(
            functional,
            step,
            tensors.state,
            tstop=STEPS * DT,
            extra=(field, temporal),
        )
    else:
        state, _ = dn.func.longrun_checkpointed(
            functional,
            step,
            tensors.state,
            STEPS * DT,
            2,
            extra=(field, temporal),
        )
    imperative_model.run(
        tstop=STEPS * DT,
        dt=DT,
        extra=(field, temporal),
    )

    _assert_state_matches_model(state, imperative_model)


def test_stimulation_rejects_ambiguous_explicit_tensor_drives():
    injected_model, _waveforms = _injected_model()
    functional, tensors = dn.func.make_functional(injected_model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    zeros = torch.zeros(injected_model.shape, dtype=DTYPE)

    with pytest.raises(ValueError, match="intra"):
        functional.step(
            tensors.parameters,
            prepared,
            tensors.state,
            dn.func.StepInput(intra=zeros),
        )
    with pytest.raises(ValueError, match="intra"):
        functional.rollout(
            tensors.parameters,
            prepared,
            tensors.state,
            dn.func.RolloutInput(intra=zeros.unsqueeze(0)),
            steps=1,
        )

    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    extra = _extra(model.shape)
    with pytest.raises(ValueError, match="ve.*extra|extra.*ve"):
        functional.step(
            tensors.parameters,
            prepared,
            tensors.state,
            dn.func.StepInput(ve=zeros),
            extra=extra,
        )
    with pytest.raises(ValueError, match="ve.*extra|extra.*ve"):
        functional.rollout(
            tensors.parameters,
            prepared,
            tensors.state,
            dn.func.RolloutInput(ve=zeros.unsqueeze(0)),
            steps=1,
            extra=extra,
        )

    step = _bound_step(functional, tensors)
    with pytest.raises(ValueError, match="ve.*extra|extra.*ve"):
        dn.func.run(
            functional,
            step,
            tensors.state,
            dn.func.RolloutInput(ve=zeros.unsqueeze(0)),
            extra=extra,
        )


def test_make_functional_can_bind_an_existing_extra_spec():
    functional_model = _model()
    imperative_model = _model()
    functional_extra = _extra(functional_model.shape)
    imperative_extra = _extra(imperative_model.shape)

    functional, tensors = dn.func.make_functional(
        functional_model,
        dt=DT,
        extra=functional_extra,
    )
    assert functional.extra.enabled
    assert any(name.startswith(EXTRA_PREFIX) for name in tensors.parameters)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    state, _auxiliary = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        steps=STEPS,
    )
    for _ in range(STEPS):
        imperative_model.step(dt=DT, extra=imperative_extra)

    _assert_state_matches_model(state, imperative_model)
    with pytest.raises(ValueError, match="ve"):
        functional.step(
            tensors.parameters,
            prepared,
            tensors.state,
            dn.func.StepInput(ve=torch.zeros(functional_model.shape, dtype=DTYPE)),
        )


@pytest.mark.parametrize("samples", [1, STEPS])
def test_make_functional_rejects_bound_tensor_temporal_extra(samples):
    model = _model()
    field = torch.ones(model.shape, dtype=DTYPE)
    temporal = torch.linspace(0.1, 0.4, samples, dtype=DTYPE)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="Tensor-temporal extra.*pass it at runtime",
    ):
        dn.func.make_functional(
            model,
            dt=DT,
            extra=(field, temporal),
        )


@pytest.mark.parametrize("entrypoint", ["rollout", "run"])
def test_incompatible_functional_extra_plan_is_rejected(entrypoint):
    model = _model()
    other_model = _model((2.0,))
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    extra_plan = dn.func.FunctionalExtra(
        other_model,
        (
            torch.ones(other_model.shape, dtype=DTYPE),
            torch.linspace(0.1, 0.4, STEPS, dtype=DTYPE),
        ),
    )

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="FunctionalExtra shape, device, or dtype",
    ):
        if entrypoint == "rollout":
            functional.rollout(
                tensors.parameters,
                prepared,
                tensors.state,
                steps=STEPS,
                extra=extra_plan,
            )
        else:
            dn.func.run(
                functional,
                _bound_step(functional, tensors),
                tensors.state,
                tstop=STEPS * DT,
                extra=extra_plan,
            )


def test_stimulus_parameters_can_change_without_repreparing_and_compile_fullgraph():
    model, _waveforms = _injected_model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    amp_name = next(
        name
        for name in tensors.parameters
        if name.startswith(INTRA_PREFIX) and name.endswith(".amp")
    )
    changed = dict(tensors.parameters)
    changed[amp_name] = 1.25 * changed[amp_name]

    expected, _ = functional.rollout(
        changed,
        prepared,
        tensors.state,
        steps=2,
    )
    compiled = functional.compile_rollout_chunk(2, backend="aot_eager")
    with torch_compiler_warning_context():
        actual, _ = compiled(changed, prepared, tensors.state)

    torch.testing.assert_close(actual["integrator"]["v"], expected["integrator"]["v"])
    baseline, _ = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        steps=2,
    )
    assert not torch.equal(actual["integrator"]["v"], baseline["integrator"]["v"])


@pytest.mark.parametrize(
    ("corruption", "error_type", "error_match"),
    [
        ("unexpected", KeyError, "unexpected"),
        ("wrong_shape", ValueError, "has shape"),
        ("wrong_dtype", ValueError, "must use"),
    ],
)
def test_cached_compiled_chunk_revalidates_stimulation_parameter_schema(
    corruption,
    error_type,
    error_match,
):
    model, _waveforms = _injected_model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    compiled = functional.compile_rollout_chunk(1, backend="eager")

    # Populate CompiledPopulationChunk's weak prepared-plan cache first. The
    # next call must retain the complete public schema checks while using the
    # cheaper dynamics-only preparation freshness path.
    with torch_compiler_warning_context():
        compiled(tensors.parameters, prepared, tensors.state)

    invalid = dict(tensors.parameters)
    amp_name = next(
        name
        for name in invalid
        if name.startswith(INTRA_PREFIX) and name.endswith(".amp")
    )
    if corruption == "unexpected":
        invalid["unexpected.stimulation.leaf"] = torch.zeros((), dtype=DTYPE)
    elif corruption == "wrong_shape":
        invalid[amp_name] = invalid[amp_name].reshape(1)
    else:
        invalid[amp_name] = invalid[amp_name].to(torch.float32)

    with torch_compiler_warning_context(), pytest.raises(error_type, match=error_match):
        compiled(invalid, prepared, tensors.state)


def test_bound_intra_prewarm_and_no_grad_compiled_inference_use_drive_signature():
    model, _waveforms = _injected_model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    assert functional.prewarm_structured_rollout()
    assert (False, True) in functional._structured_step_graphs
    compiled = functional.compile_rollout_chunk(3, backend="aot_eager")

    with torch.no_grad(), torch_compiler_warning_context():
        expected, _ = functional.rollout(
            tensors.parameters,
            prepared,
            tensors.state,
            steps=3,
        )
        actual, _ = compiled(tensors.parameters, prepared, tensors.state)

    torch.testing.assert_close(actual["integrator"]["v"], expected["integrator"]["v"])


def test_bound_intra_atomic_prepare_and_rollout_is_fullgraph_compilable():
    model, _waveforms = _injected_model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    functional.prewarm_structured_rollout()

    def atomic(parameters, constants, state):
        return functional.prepare_and_rollout(
            parameters,
            constants,
            state,
            steps=2,
        )[0]["integrator"]["v"]

    compiled = torch.compile(atomic, backend="aot_eager", fullgraph=True)
    expected = atomic(tensors.parameters, tensors.constants, tensors.state)
    with torch_compiler_warning_context():
        actual = compiled(tensors.parameters, tensors.constants, tensors.state)
    torch.testing.assert_close(actual, expected)


def test_bound_extra_is_explicit_replaceable_and_fullgraph_compilable():
    model = _model()
    extra = _extra(model.shape, requires_grad=True)
    functional, tensors = dn.func.make_functional(model, dt=DT, extra=extra)
    field_name = next(
        name
        for name in tensors.parameters
        if name.startswith(EXTRA_PREFIX) and name.endswith(".field")
    )
    changed = dict(tensors.parameters)
    changed[field_name] = 1.1 * changed[field_name]

    def atomic(parameters, constants, state):
        return functional.prepare_and_rollout(
            parameters,
            constants,
            state,
            steps=2,
        )[0]["integrator"]["v"]

    compiled = torch.compile(atomic, backend="aot_eager", fullgraph=True)
    expected = atomic(changed, tensors.constants, tensors.state)
    with torch_compiler_warning_context():
        actual = compiled(changed, tensors.constants, tensors.state)
    torch.testing.assert_close(actual, expected)

    (field_gradient,) = torch.autograd.grad(actual.square().mean(), (extra[0][0],))
    assert torch.isfinite(field_gradient).all()
    assert torch.count_nonzero(field_gradient) > 0


def test_shared_and_eval_mode_waveforms_remain_explicit_and_pure():
    model = _model()
    waveform = dn.sin(amp=0.2 * nA, freq=0.7).eval()
    model[:, 1].inject(waveform)
    model[:, 2].inject(waveform)
    snapshot = _module_snapshot(waveform)

    functional, tensors = dn.func.make_functional(model, dt=DT)
    amp_names = [
        name
        for name in tensors.parameters
        if name.startswith(INTRA_PREFIX) and name.endswith(".amp")
    ]
    assert len(amp_names) == 1
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    baseline, _ = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        steps=2,
    )
    changed = dict(tensors.parameters)
    changed[amp_names[0]] = 1.5 * changed[amp_names[0]]
    modified, _ = functional.rollout(
        changed,
        prepared,
        tensors.state,
        steps=2,
    )

    assert not torch.equal(baseline["integrator"]["v"], modified["integrator"]["v"])
    _assert_module_unchanged(waveform, snapshot)


def test_custom_eval_mode_is_preserved_while_bounded_cache_is_bypassed():
    model = _model()
    waveform = _ModeAwarePureWaveform(value=0.2 * nA).eval()
    model[:, 1].inject(waveform)
    functional, _tensors = dn.func.make_functional(model, dt=DT)
    stimulation = functional.intra.extract()
    times = torch.arange(STEPS, dtype=DTYPE) * DT + START

    assembled = functional.intra.assemble_tensors(stimulation, times)
    expected = waveform(times)
    torch.testing.assert_close(assembled[:, :, 1], expected[:, None].expand(-1, 2))

    rho_name = next(name for name in stimulation.parameters if name.endswith(".rho"))
    changed = functional.intra.replace(
        stimulation,
        parameters={rho_name: stimulation.parameters[rho_name] + 0.25},
    )
    modified = functional.intra.assemble_tensors(changed, times)
    assert not torch.equal(modified, assembled)
    assert not waveform.training


@pytest.mark.parametrize(
    ("waveform", "message"),
    [
        (_UnmarkedCustomWaveform(), "FUNCTIONAL_PURE"),
        (_MutatingPureWaveform(), "mutated registered tensor"),
        (_StochasticPureWaveform(), "RNG|deterministic"),
        (_CallableStatePureWaveform(torch.sin), "callable"),
    ],
)
def test_custom_waveforms_fail_closed_without_an_auditable_pure_contract(
    waveform, message
):
    model = _model()
    model[:, 1].inject(waveform)

    with pytest.raises(dn.func.FunctionalizationError, match=message):
        dn.func.make_functional(model, dt=DT)


def test_lowering_late_off_dtype_injection_does_not_move_the_source_waveform():
    model = _model()
    waveform = dn.sin(
        amp=torch.tensor(0.2 * nA, dtype=torch.float32),
        freq=torch.tensor(0.7, dtype=torch.float32),
    )
    model[:, 1].inject(waveform)
    snapshot = _module_snapshot(waveform)

    functional, tensors = dn.func.make_functional(model, dt=DT)

    assert functional.intra.enabled
    assert all(
        value.dtype == DTYPE
        for name, value in tensors.parameters.items()
        if name.startswith(INTRA_PREFIX)
    )
    _assert_module_unchanged(waveform, snapshot)


def test_randomized_poisson_fails_closed_but_a_fixed_schedule_is_supported():
    randomized = _model()
    randomized[:, 1].inject(
        dn.mono_rect(amp=0.2 * nA, pw=0.05).poisson(
            interval=0.1,
            n=3,
            randomize_every_call=True,
            generator=torch.Generator().manual_seed(12),
        )
    )
    with pytest.raises(dn.func.FunctionalizationError, match="randomizes"):
        dn.func.make_functional(randomized, dt=DT)

    fixed = _model()
    fixed[:, 1].inject(
        dn.mono_rect(amp=0.2 * nA, pw=0.05).poisson(
            interval=0.1,
            n=3,
            randomize_every_call=False,
            generator=torch.Generator().manual_seed(12),
        )
    )
    functional, tensors = dn.func.make_functional(fixed, dt=DT)
    assert any(name.endswith("._spike_times") for name in tensors.parameters)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    state, _ = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        steps=STEPS,
    )
    assert torch.isfinite(state["integrator"]["v"]).all()


def test_longrun_streams_tensor_temporal_extra_with_global_chunk_slices():
    functional_model = _model()
    imperative_model = _model()
    field = torch.linspace(-1.0, 1.0, functional_model.v.numel(), dtype=DTYPE).reshape(
        functional_model.shape
    )
    temporal = torch.linspace(0.1, 0.4, STEPS, dtype=DTYPE, requires_grad=True)
    functional, tensors = dn.func.make_functional(functional_model, dt=DT)
    step = _bound_step(functional, tensors)

    state, _ = dn.func.longrun(
        functional,
        step,
        tensors.state,
        STEPS * DT,
        3,
        extra=(field, temporal),
    )
    imperative_model.longrun(
        tstop=STEPS * DT,
        dt=DT,
        chunklength=3,
        extra=(field.detach(), temporal.detach()),
    )

    _assert_state_matches_model(state, imperative_model)
    (temporal_gradient,) = torch.autograd.grad(
        state["integrator"]["v"].square().mean(),
        (temporal,),
    )
    assert torch.isfinite(temporal_gradient).all()
    assert torch.count_nonzero(temporal_gradient) > 0


@pytest.mark.parametrize("runner", ["run", "longrun", "longrun_checkpointed"])
def test_zero_step_host_runner_validates_tensor_temporal_extra_horizon(runner):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    step = _bound_step(functional, tensors)
    field = torch.ones(model.shape, dtype=DTYPE)
    temporal = torch.linspace(0.1, 0.4, STEPS, dtype=DTYPE)

    with pytest.raises(ValueError, match="samples; expected total_steps=0"):
        if runner == "run":
            dn.func.run(
                functional,
                step,
                tensors.state,
                tstop=0.0,
                extra=(field, temporal),
            )
        elif runner == "longrun":
            dn.func.longrun(
                functional,
                step,
                tensors.state,
                0.0,
                2,
                extra=(field, temporal),
            )
        else:
            dn.func.longrun_checkpointed(
                functional,
                step,
                tensors.state,
                0.0,
                2,
                extra=(field, temporal),
            )


def test_zero_step_run_accepts_empty_tensor_temporal_extra():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    state, auxiliary = dn.func.run(
        functional,
        _bound_step(functional, tensors),
        tensors.state,
        tstop=0.0,
        extra=(
            torch.ones(model.shape, dtype=DTYPE),
            torch.empty(0, dtype=DTYPE),
        ),
    )

    assert auxiliary is None
    torch.testing.assert_close(
        state["integrator"]["v"],
        tensors.state["integrator"]["v"],
    )


def test_checkpointed_runtime_extra_replays_waveforms_and_preserves_gradients():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    step = _bound_step(functional, tensors)
    extra = _extra(model.shape, requires_grad=True)

    ordinary, _ = dn.func.longrun(
        functional,
        step,
        tensors.state,
        STEPS * DT,
        2,
        extra=extra,
    )
    checkpointed, _ = dn.func.longrun_checkpointed(
        functional,
        step,
        tensors.state,
        STEPS * DT,
        2,
        extra=extra,
    )
    torch.testing.assert_close(
        checkpointed["integrator"]["v"],
        ordinary["integrator"]["v"],
    )

    field, waveform = extra[0]
    field_gradient, amplitude_gradient = torch.autograd.grad(
        checkpointed["integrator"]["v"].square().mean(),
        (field, waveform.amp),
    )
    assert torch.isfinite(field_gradient).all()
    assert torch.count_nonzero(field_gradient) > 0
    assert torch.isfinite(amplitude_gradient).all()
    assert torch.count_nonzero(amplitude_gradient) > 0


def test_stimulus_tensors_are_plan_bound_pytree_inputs_for_jacrev():
    model = _model()
    functional, _tensors = dn.func.make_functional(model, dt=DT)
    field = torch.linspace(-0.5, 1.0, model.v.numel(), dtype=DTYPE).reshape(model.shape)
    with dn.ctx(REQUIRE_GRAD=1, DTYPE=DTYPE):
        waveform = dn.constant(value=0.2)
    extra = functional.make_extra((field, waveform))
    bundle = extra.extract()

    leaves, tree_spec = torch.utils._pytree.tree_flatten(bundle)
    assert leaves
    assert all(torch.is_tensor(leaf) for leaf in leaves)
    restored = torch.utils._pytree.tree_unflatten(leaves, tree_spec)
    assert isinstance(restored, dn.func.StimulusTensors)
    assert restored.plan_token is bundle.plan_token

    times = START + DT * torch.arange(STEPS, dtype=DTYPE)

    def response(stimulus_tensors):
        return extra.assemble_tensors(stimulus_tensors, times).square().sum()

    jacobian = torch.func.jacrev(response)(bundle)
    assert isinstance(jacobian, dn.func.StimulusTensors)
    assert jacobian.plan_token is bundle.plan_token
    field_name = next(name for name in jacobian.constants if name.endswith(".field"))
    value_name = next(name for name in jacobian.parameters if name.endswith(".value"))
    assert torch.isfinite(jacobian.constants[field_name]).all()
    assert torch.count_nonzero(jacobian.constants[field_name]) > 0
    assert torch.isfinite(jacobian.parameters[value_name]).all()
    assert torch.count_nonzero(jacobian.parameters[value_name]) > 0


@pytest.mark.parametrize("transform", ["jacrev", "compile"])
def test_raw_extra_lowering_fails_clearly_inside_tensor_transforms(transform):
    model = _model()
    functional, _tensors = dn.func.make_functional(model, dt=DT)
    with dn.ctx(REQUIRE_GRAD=1, DTYPE=DTYPE):
        raw_extra = (
            torch.ones(model.shape, dtype=DTYPE),
            dn.constant(value=0.2),
        )

    def lower_inside_transform(value):
        functional.make_extra(raw_extra)
        return value.square()

    message = "Raw extra specifications cannot be lowered.*make_functional"
    if transform == "jacrev":
        with pytest.raises(dn.func.FunctionalizationError, match=message):
            torch.func.jacrev(lower_inside_transform)(torch.ones((), dtype=DTYPE))
    else:
        compiled = torch.compile(
            lower_inside_transform,
            backend="eager",
            fullgraph=True,
        )
        # Dynamo reports an exception deliberately observed while tracing as
        # Unsupported, while preserving our FunctionalizationError and its
        # remediation in the diagnostic's developer context.
        with (
            torch_compiler_warning_context(),
            pytest.raises(torch._dynamo.exc.Unsupported, match=message),
        ):
            compiled(torch.ones((), dtype=DTYPE))


def test_bound_extra_rejects_waveform_object_shared_with_registered_intra():
    model, (waveform,) = _injected_model()
    extra = (torch.ones(model.shape, dtype=DTYPE), waveform)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="same Waveform object cannot be shared",
    ):
        dn.func.make_functional(model, dt=DT, extra=extra)
