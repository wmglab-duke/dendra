"""Native LFP configurations share pure projections without native history."""

from __future__ import annotations

import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_continuation_checkpoints import _disk
from tests.func.test_lfp import DT, _model

pytestmark = pytest.mark.cpu
STEPS = 7


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=True),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _initial(tensors):
    return dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants)


def _case(network=False, *, batch=(), device="cpu", dtype=torch.float64):
    population = _model(batch=batch, device=device, dtype=dtype)
    owner = population
    if network:
        owner = dn.Network(
            {
                "observed": population,
                "other": _model(batch=batch, device=device, dtype=dtype),
            }
        )
        owner.train()
        owner.initialize(DT)
    functional, tensors = dn.func.make_functional(owner, dt=DT)
    return owner, population, functional, tensors


def _scoped(callback, network):
    return dn.func.PopulationCallback("observed", callback) if network else callback


def _execute(callback, currents):
    carry, initial = callback.initialize(
        {"integrator": {"i_membrane": currents[0]}}, None
    )
    emissions = [initial]
    for current in currents[1:]:
        carry, emitted = callback.update(
            carry, {"integrator": {"i_membrane": current}}, None
        )
        emissions.append(emitted)
    return callback.finalize(carry, torch.stack(emissions))


@pytest.mark.parametrize(
    "batch,field_shape", (((), (2, 3)), ((2,), ()), ((2, 3), (3,)))
)
@pytest.mark.parametrize("rule", (None, "...,n...->n"))
def test_native_and_functional_projection_match_for_batches_broadcasts_and_explicit_rules(
    batch, field_shape, rule
):
    model = _model(batch=batch)
    functional, _ = dn.func.make_functional(model, dt=DT)
    field = torch.full(field_shape, 1.25, dtype=torch.float32, requires_grad=True)
    fields = [field, field * 2]
    configured = dn.callbacks.LFP(fields, rule=rule)
    collection = functional.make_callbacks({"lfp": configured})
    callback = collection._plans[0].callback
    assert type(callback) is dn.func.LFP and not collection._plans[0].validate_updates
    assert callback.rule == rule
    currents = torch.arange(1, 1 + 4 * model.v.numel(), dtype=torch.float64).reshape(
        4, *model.shape
    )
    observed = _execute(callback, currents)
    native = dn.callbacks.LFP(fields, rule=rule)
    native_model = SimpleNamespace(
        i_membrane=currents[0],
        integrator=SimpleNamespace(imem=True),
        device=lambda: currents.device,
    )
    native.pre_loop_hook(native_model)
    for current in currents[1:]:
        native_model.i_membrane = current
        native.post_step_hook(native_model)
    if rule is None:
        expected = torch.stack(
            [(currents * value.double()).sum((-1, -2)) for value in fields], dim=-1
        )
        assert observed.shape == (4, *model.shape[:-2], len(fields))
    else:
        expected = torch.stack(
            [
                torch.einsum(rule, current, torch.stack(fields).double())
                for current in currents
            ]
        )
        assert observed.shape == (4, len(fields))
    _close(observed, expected)
    _close(native.lfp, expected)
    assert observed.dtype == native.lfp.dtype == torch.float64
    assert configured._lfp == configured._t == []
    assert configured.v_unit.dtype == torch.float32
    actual_gradient = torch.autograd.grad(
        observed.square().sum(), field, retain_graph=True
    )[0]
    expected_gradient = torch.autograd.grad(expected.square().sum(), field)[0]
    _close(actual_gradient, expected_gradient)
    assert actual_gradient.dtype == torch.float32 and actual_gradient.abs().sum() > 0


def test_native_configuration_retains_live_field_aliases_across_fresh_training_runs():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    field = torch.ones(model.shape, dtype=torch.float32, requires_grad=True)
    configured = dn.callbacks.LFP([field, field])
    configured._lfp = [torch.tensor([123.0])]
    configured._t = [123.0]
    history, times, public = configured._lfp, configured._t, configured.v_unit
    frozen = _snapshot((history, times, public))
    callbacks = functional.make_callbacks(
        {"lfp": configured, "trace": dn.Recorder(("i_membrane",))}
    )
    callback = callbacks._plans[0].callback
    assert callback.fields[0] is callback.fields[1] is field
    previous = None
    for _ in range(2):
        drive = (
            torch.linspace(-2e-7, 3e-7, 3 * model.v.numel(), dtype=torch.float64)
            .reshape(3, *model.shape)
            .requires_grad_()
        )
        _, auxiliary = dn.func.run(
            functional,
            _initial(tensors),
            steps=3,
            inputs=dn.func.Inputs(intra=drive),
            callbacks=callbacks,
        )
        result = auxiliary["callbacks"]
        current = result["trace"]["i_membrane"]
        expected = (current * field.double()).sum((-1, -2))
        _close(result["lfp"][..., 0], expected)
        _close(result["lfp"][..., 1], expected)
        gradients = torch.autograd.grad(
            (result["lfp"] * 1e6).square().mean(), (field, drive)
        )
        assert all(
            torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
        )
        if previous is not None:
            _close(gradients, previous)
        previous = _snapshot(gradients)
    assert configured._lfp is history and configured._t is times
    assert configured.v_unit is public
    _close((history, times, public), frozen)
    with torch.no_grad():
        field.add_(0.5)
    rebound = functional.make_callbacks({"lfp": configured})._plans[0].callback
    assert rebound.fields[0] is field
    currents = torch.ones((2, *model.shape), dtype=torch.float64)
    expected = currents.sum((-1, -2)).unsqueeze(-1).expand(2, 2) * 1.5
    _close(_execute(callback, currents), expected)
    _close(_execute(rebound, currents), expected)


def test_public_native_buffer_edits_are_live_and_replacement_requires_rebinding():
    model = _model(batch=(2,))
    functional, _ = dn.func.make_functional(model, dt=DT)
    original = torch.ones((2, 3))
    configured = dn.callbacks.LFP([original])
    configured.v_unit.add_(2)
    first = functional.make_callbacks({"lfp": configured})._plans[0].callback
    currents = torch.ones((2, *model.shape), dtype=torch.float64)
    expected = currents.sum((-1, -2)).unsqueeze(-1)
    _close(_execute(first, currents), expected * 3)
    original.add_(100)
    configured.v_unit.mul_(2)
    _close(_execute(first, currents), expected * 6)
    configured.v_unit = torch.full_like(configured.v_unit, 9)
    rebound = functional.make_callbacks({"lfp": configured})._plans[0].callback
    _close(_execute(first, currents), expected * 6)
    _close(_execute(rebound, currents), expected * 9)
    configured.rule = "...,n...->n"
    explicit = functional.make_callbacks({"lfp": configured})._plans[0].callback
    assert explicit.rule == "...,n...->n"
    _close(_execute(explicit, currents), torch.full((2, 1), 108.0, dtype=torch.float64))


def test_inference_binding_of_a_trainable_public_buffer_preserves_fresh_gradients():
    model = _model()
    functional, _ = dn.func.make_functional(model, dt=DT)
    configured = dn.callbacks.LFP([torch.ones(model.shape)])
    public = (
        torch.linspace(0.25, 1.25, 2 * model.v.numel(), dtype=torch.float64)
        .reshape(2, *model.shape)
        .requires_grad_()
    )
    configured.v_unit = public
    configured._lfp = [torch.tensor([123.0])]
    configured._t = [123.0]
    history, times = configured._lfp, configured._t
    frozen_history = _snapshot((history, times))
    initial_public = public.detach().clone()
    with torch.inference_mode():
        callbacks = functional.make_callbacks({"lfp": configured})
    callback = callbacks._plans[0].callback
    for epoch in range(2):
        currents = (
            torch.arange(1, 1 + 3 * model.v.numel(), dtype=torch.float64)
            .reshape(3, *model.shape)
            .requires_grad_()
        )
        observed = _execute(callback, currents)
        expected = torch.stack(
            [(currents * contact).sum((-1, -2)) for contact in public], dim=-1
        )
        _close(observed, expected)
        gradients = torch.autograd.grad(observed.square().sum(), (public, currents))
        reference = torch.autograd.grad(expected.square().sum(), (public, currents))
        _close(gradients, reference)
        assert all(
            torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
        )
        if epoch == 0:
            with torch.no_grad():
                public.add_(0.25)
    assert configured.v_unit is public
    assert dict(configured.named_buffers())["v_unit"] is public
    assert configured._lfp is history and configured._t is times
    _close((history, times), frozen_history)
    _close(public, initial_public + 0.25)


def test_native_lfp_requires_current_and_valid_default_broadcast():
    for imem, fields, message in (
        (False, [torch.ones(2, 3)], "membrane-current"),
        (True, [torch.ones(4)], "broadcast"),
    ):
        functional, _ = dn.func.make_functional(_model(imem=imem), dt=DT)
        configured = dn.callbacks.LFP(fields)
        with pytest.raises(dn.func.FunctionalizationError, match=message):
            functional.make_callbacks({"lfp": configured})
        assert configured._lfp == configured._t == []


def test_native_subclasses_and_imperative_hook_overrides_are_not_silently_lowered():
    model = _model()
    functional, _ = dn.func.make_functional(model, dt=DT)

    class Derived(dn.callbacks.LFP):
        pass

    for configured in (
        Derived([torch.ones(model.shape)]),
        dn.callbacks.LFP([torch.ones(model.shape)]),
    ):
        if type(configured) is dn.callbacks.LFP:
            configured.post_step_hook = lambda model: None
        with pytest.raises(
            (TypeError, dn.func.FunctionalizationError),
            match="FunctionalCallback|custom|hook|configuration",
        ):
            functional.make_callbacks({"lfp": configured})


def test_explicit_functional_binding_retains_authority_for_a_native_lfp_subclass():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    calls = []

    class Pure(dn.callbacks.LFP, dn.func.FunctionalCallback):
        def _bind(self, plan):
            calls.append(plan)
            return self

        def initialize(self, state, auxiliary):
            return (), state["integrator"]["v"].sum()

        def update(self, carry, state, auxiliary):
            return carry, state["integrator"]["v"].sum()

        def finalize(self, carry, emissions):
            return emissions

    configured = Pure([torch.ones(model.shape)])
    callbacks = functional.make_callbacks({"custom": configured})
    assert calls == [functional] and callbacks._plans[0].callback is configured
    assert callbacks._plans[0].validate_updates
    _, auxiliary = dn.func.run(
        functional, _initial(tensors), steps=2, callbacks=callbacks
    )
    assert auxiliary["callbacks"]["custom"].shape == (3,)
    assert configured._lfp == configured._t == []


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_compile_accepts_native_lfp_configuration_without_importing_history(network):
    owner, population, _, _ = _case(network)
    configured = dn.callbacks.LFP([torch.ones(population.shape)])
    configured._lfp = [torch.tensor([123.0])]
    history = configured._lfp
    program = dn.compile(
        owner,
        dt=DT,
        steps=3,
        callbacks={"lfp": _scoped(configured, network)},
        backend="eager",
        disable=True,
    )
    _, auxiliary = program()
    assert auxiliary["callbacks"]["lfp"].shape == (4, 1)
    assert configured._lfp is history and configured._lfp[0].item() == 123.0
    assert configured._t == []


def test_native_lfp_requires_an_explicit_network_population_scope():
    _, population, functional, _ = _case(True)
    configured = dn.callbacks.LFP([torch.ones(population.shape)])
    with pytest.raises(
        dn.func.FunctionalizationError, match="Population-specific|PopulationCallback"
    ):
        functional.make_callbacks({"lfp": configured})
    callbacks = functional.make_callbacks({"lfp": _scoped(configured, True)})
    assert callbacks._plans[0].callback.population == "observed"


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_native_and_functional_lfp_keep_identical_checkpoint_configuration(
    network, tmp_path
):
    _, population, functional, tensors = _case(network)
    field = torch.ones(population.shape, dtype=torch.float32, requires_grad=True)
    configurations = (dn.callbacks.LFP([field, field]), dn.func.LFP([field, field]))
    collections = [
        functional.make_callbacks({"lfp": _scoped(value, network)})
        for value in configurations
    ]
    packets = []
    for callbacks in collections:
        _, auxiliary = dn.func.run(
            functional, _initial(tensors), steps=0, callbacks=callbacks
        )
        packets.append(callbacks.export_state(auxiliary["callbacks"].state))
    _assert_packet_equal(*packets)
    assert (
        packets[0]["format"] == "dendra.func.callback_state"
        and packets[0]["version"] == 1
    )
    restored = collections[1].restore_state(
        _disk(packets[0], tmp_path / "callbacks.pt")
    )
    _close(restored["lfp"], packets[0]["carries"]["lfp"])
    assert not restored["lfp"].requires_grad


def _compiled_case(network, execution, tmp_path, *, device="cpu", dtype=torch.float64):
    owner, population, functional, tensors = _case(network, device=device, dtype=dtype)
    field = torch.ones(
        (2, 3), device=device, dtype=dtype, requires_grad=execution == "scan"
    )
    configured = dn.callbacks.LFP([field, field])
    configured._lfp = [torch.tensor([123.0])]
    configured._t = [123.0]
    history, times, public = configured._lfp, configured._t, configured.v_unit
    frozen = _snapshot((history, times, public))
    prefix = "observed." if network else ""
    trace = (prefix + "i_membrane", "t")

    def specifications(callback):
        return {"lfp": _scoped(callback, network), "trace": dn.Recorder(trace)}

    callbacks = functional.make_callbacks(specifications(configured))
    equivalent = functional.make_callbacks(specifications(dn.func.LFP([field, field])))
    initial = _initial(tensors)
    source = _snapshot(functional.extract().state)
    kernel = functional.compile_rollout_chunk(
        4, execution=execution, backend="aot_eager"
    )
    drive = (
        torch.linspace(
            -2e-7, 3e-7, STEPS * population.v.numel(), device=device, dtype=dtype
        )
        .reshape(STEPS, *population.shape)
        .requires_grad_(execution == "scan")
    )

    def inputs(values):
        values = dn.func.Inputs(intra=values)
        return {"observed": values} if network else values

    checkpoint = {"checkpoint_every": 2} if execution == "scan" else {}
    with nullcontext() if execution == "scan" else torch.no_grad():
        expected, whole = dn.func.run(
            functional, initial, steps=STEPS, inputs=inputs(drive), callbacks=equivalent
        )
        _, first = dn.func.run(
            kernel,
            initial,
            steps=4,
            inputs=inputs(drive[:4]),
            callbacks=callbacks,
            **checkpoint,
        )
        assert kernel.execution_report().strategy == (
            "compiled_scan" if execution == "scan" else "compiled_while_loop"
        )
        _, empty = dn.func.run(
            kernel, first["continuation"], steps=0, callbacks=callbacks, **checkpoint
        )
        actual, last = dn.func.run(
            kernel,
            empty["continuation"],
            steps=3,
            inputs=inputs(drive[4:]),
            callbacks=callbacks,
            **checkpoint,
        )
    _close(actual, expected)
    joined = torch.cat(
        (first["callbacks"]["lfp"], empty["callbacks"]["lfp"], last["callbacks"]["lfp"])
    )
    _close(joined * 1e6, whole["callbacks"]["lfp"] * 1e6)
    assert first["callbacks"]["lfp"].shape == (5, 2)
    assert empty["callbacks"]["lfp"].shape == (0, 2)
    assert last["callbacks"]["lfp"].shape == (3, 2)
    if execution == "scan":
        observed_gradient = torch.autograd.grad(
            (joined * 1e6).square().mean(), (field, drive)
        )
        reference_gradient = torch.autograd.grad(
            (whole["callbacks"]["lfp"] * 1e6).square().mean(), (field, drive)
        )
        _close(observed_gradient, reference_gradient)
        assert all(
            torch.isfinite(value).all() and value.abs().sum() > 0
            for value in observed_gradient
        )
    assert kernel.execution_report().strategy == (
        "compiled_scan" if execution == "scan" else "compiled_unrolled"
    )
    saved = _disk(
        dn.func.export_continuation(
            functional, first["continuation"], callbacks=callbacks
        ),
        tmp_path / "continuation.pt",
    )
    receiver = functional.make_callbacks(specifications(dn.func.LFP([field, field])))
    restored = dn.func.restore_continuation(functional, saved, callbacks=receiver)
    with torch.no_grad():
        resumed, tail = dn.func.run(
            kernel, restored, steps=3, inputs=inputs(drive[4:]), callbacks=receiver
        )
    _close(resumed, expected)
    _close(tail["callbacks"]["lfp"] * 1e6, last["callbacks"]["lfp"] * 1e6)
    assert tail["callbacks"]["trace"]["t"].shape == (3,)
    assert (
        configured._lfp is history
        and configured._t is times
        and configured.v_unit is public
    )
    _close((history, times, public), frozen)
    _close(functional.extract().state, source)
    assert owner.t.item() == 0
    return actual


@pytest.mark.parametrize("network,execution", ((False, "scan"), (True, "while")))
def test_compiled_native_lfp_preserves_gradients_and_portable_continuation(
    network, execution, tmp_path
):
    _compiled_case(network, execution, tmp_path)


@pytest.mark.mps
def test_mps_native_lfp_supports_compiled_while_and_portable_continuation(tmp_path):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DTYPE=torch.float32, DEVICE="mps"):
        result = _compiled_case(
            False, "while", tmp_path, device="mps", dtype=torch.float32
        )
        assert result["integrator"]["v"].device.type == "mps"
