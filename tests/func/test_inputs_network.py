"""Shared deterministic inputs retain Network routing, gradients and RNG."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_network import DT, DTYPE, _clone, _close
from tests.func.test_network_rng_sharing import _network as _noise_network
from tests.func.test_network_rollout import _driven_network, _native_driven_step

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"),
    ):
        yield


def _drives(functional, steps):
    shape = functional.populations["left"].shape
    size = steps * torch.Size(shape).numel()
    ve = torch.linspace(-0.3, 0.7, size, dtype=DTYPE).reshape(steps, *shape)
    intra = torch.linspace(0.01, 0.04, size, dtype=DTYPE).reshape(steps, *shape)
    return ve.requires_grad_(), intra.requires_grad_()


def _inputs(ve, intra):
    # Both populations observe the same live voltage field. Intracellular
    # stimulation has separate signed routes to exercise name resolution.
    return {
        "left": dn.func.Inputs(ve=ve, intra=intra),
        "right": dn.func.Inputs(ve=ve, intra=-0.5 * intra),
    }


def _legacy(inputs):
    return dn.func.NetworkRolloutInput(
        {
            name: dn.func.RolloutInput(value.ve, value.intra)
            for name, value in inputs.items()
        }
    )


def _loss(state):
    return sum(
        (
            population["integrator"]["v"]
            * torch.linspace(
                0.7, 1.3, population["integrator"]["v"].shape[-1], dtype=DTYPE
            )
        )
        .square()
        .mean()
        for population in state["populations"].values()
    )


@pytest.mark.parametrize("execution", ("step", "rollout", "chunk", "host"))
def test_network_shared_inputs_match_native_legacy_and_drive_gradients(execution):
    model, native = _driven_network(), _driven_network()
    functional, tensors = dn.func.make_functional_network(model)
    parameters = tensors.independent_parameters()
    state = _clone(tensors.state, detached=True)
    ve, intra = _drives(functional, 5)
    inputs = _inputs(ve, intra)
    reference, _ = functional.prepare_and_rollout(
        parameters, tensors.constants, state, _legacy(_inputs(ve, intra)), steps=5
    )
    reference_gradients = torch.autograd.grad(_loss(reference), (ve, intra))
    if execution == "step":
        actual = state
        for index in range(5):
            selected = pytree.tree_map_only(
                torch.Tensor, lambda value: value[index], inputs
            )
            actual, _ = functional.prepare_and_step(
                parameters, tensors.constants, actual, inputs=selected
            )
    elif execution == "rollout":
        actual, _ = functional.prepare_and_rollout(
            parameters, tensors.constants, state, inputs, steps=5
        )
    elif execution == "chunk":
        prepared = functional.prepare(parameters, tensors.constants)
        chunk = functional.compile_rollout_chunk(5, backend="aot_eager")
        actual, _ = chunk(parameters, prepared, state, inputs)
    else:
        bound = functional.compile_rollout_chunk(3, backend="aot_eager").bind(
            parameters, constants=tensors.constants
        )
        actual, _ = dn.func.run(bound, state, inputs=inputs, checkpoint_every=2)
    actual_gradients = torch.autograd.grad(_loss(actual), (ve, intra))
    native_ve = ve.detach().clone().requires_grad_()
    native_intra = intra.detach().clone().requires_grad_()
    native_inputs = _inputs(native_ve, native_intra)
    for index in range(5):
        _native_driven_step(
            native,
            pytree.tree_map_only(
                torch.Tensor, lambda value: value[index], native_inputs
            ),
        )
    native_state = functional.extract(native).state
    native_gradients = torch.autograd.grad(
        _loss(
            {
                "populations": {
                    name: {"integrator": {"v": cell.v}}
                    for name, cell in native.populations.items()
                }
            }
        ),
        (native_ve, native_intra),
    )
    _close(actual, reference)
    _close(actual, native_state)
    _close(actual_gradients, reference_gradients)
    _close(actual_gradients, native_gradients)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0
        for value in actual_gradients
    )


def test_network_internal_inputs_preserve_live_aliases_and_global_draw_authority():
    from dendra.func._inputs import _RuntimeInputs
    from dendra.func._network_rollout import _resolve_inputs

    functional, tensors = dn.func.make_functional_network(_driven_network())
    ve, intra = _drives(functional, 3)
    draws = functional.sample_runtime_draws(tensors.state, steps=3)
    tape, count = _resolve_inputs(functional, _inputs(ve, intra), None, draws=draws)
    assert count == 3
    assert tape["draws"] is draws
    left, right = (tape["populations"][name] for name in ("left", "right"))
    assert isinstance(left, _RuntimeInputs) and isinstance(right, _RuntimeInputs)
    assert left.ve.untyped_storage().data_ptr() == ve.untyped_storage().data_ptr()
    assert right.ve.untyped_storage().data_ptr() == ve.untyped_storage().data_ptr()
    assert left.intra.untyped_storage().data_ptr() == intra.untyped_storage().data_ptr()
    _close(left.ve, ve)
    _close(right.ve, ve)
    _close(left.intra, intra)
    assert left.ve.stride() == right.ve.stride() == ve.stride()
    _close(
        torch.autograd.grad(left.ve.sum() + right.ve.sum(), ve)[0],
        torch.full_like(ve, 2),
    )
    assert left.noise is right.noise is None
    assert left.draws is right.draws is None


@pytest.mark.parametrize("execution", ("host", "bound", "compiled"))
def test_legacy_network_local_canonical_packets_remain_provided_without_resampling(
    monkeypatch, execution
):
    functional, tensors = dn.func.make_functional_network(_noise_network(shared=False))
    assert not functional._rng_plan.enabled
    assert not functional._stimulation_rng_plan.enabled
    parameters = tensors.independent_parameters()
    state = _clone(tensors.state, detached=True)
    intra = _drives(functional, 3)[1]
    packets = {
        name: plan.sample_runtime_draws(state["populations"][name], steps=3)
        for name, plan in functional.populations.items()
    }
    saved = _clone(packets, detached=True)
    inputs = dn.func.NetworkRolloutInput(
        {
            name: dn.func.RolloutInput(intra=intra, noise=packets[name])
            for name in functional.populations
        }
    )
    expected, _ = functional.prepare_and_rollout(
        parameters, tensors.constants, state, inputs, steps=3
    )
    reference = torch.autograd.grad(_loss(expected), (intra,))

    def unexpected_sampling(*_arguments, **_keywords):
        pytest.fail("Provided local canonical packets must not be sampled again")

    for plan in functional.populations.values():
        monkeypatch.setattr(plan, "sample_noise_draw_tape", unexpected_sampling)
    if execution == "host":
        actual, _ = functional.run(
            parameters, tensors.constants, state, inputs, steps=3, chunklength=2
        )
    else:
        owner = (
            functional.compile_rollout_chunk(2, backend="aot_eager")
            if execution == "compiled"
            else functional
        )
        bound = owner.bind(parameters, constants=tensors.constants)
        actual, _ = dn.func.run(bound, state, inputs=inputs, steps=3, host_span_steps=2)
    _close(actual, expected)
    _close(torch.autograd.grad(_loss(actual), (intra,)), reference)
    assert all(value.abs().sum() > 0 for value in reference)
    _close(packets, saved)


def test_legacy_local_poisson_carrier_preserves_validation_and_sampler_order(
    monkeypatch,
):
    from tests.func.test_runtime_draws_dormant_rng import _network

    model, _ = _network("poisson")
    functional, tensors = dn.func.make_functional_network(model)
    assert not functional._rng_plan.enabled
    assert not functional._stimulation_rng_plan.enabled
    receiver = functional.populations["receiver"]
    packet = functional._runtime_draw_plan.local(
        "receiver", functional.sample_runtime_draws(tensors.state, steps=2)
    )
    inputs = dn.func.NetworkRolloutInput(
        {"receiver": dn.func.RolloutInput(noise=packet)}
    )
    original_noise = functional._rng_plan.sample_noise_tapes
    original_donor = functional.populations["donor"].sample_noise_draw_tape
    original_stimulation = functional.sample_population_stimulation_tapes
    events = []

    def observe_noise(*arguments, **keywords):
        events.append("noise")
        return original_noise(*arguments, **keywords)

    def observe_donor(*arguments, **keywords):
        events.append("donor_noise")
        return original_donor(*arguments, **keywords)

    def observe_stimulation(*arguments, **keywords):
        events.append("stimulation")
        return original_stimulation(*arguments, **keywords)

    def unexpected_receiver(*_arguments, **_keywords):
        pytest.fail("The supplied local canonical packet is already complete")

    monkeypatch.setattr(functional._rng_plan, "sample_noise_tapes", observe_noise)
    monkeypatch.setattr(
        functional.populations["donor"], "sample_noise_draw_tape", observe_donor
    )
    monkeypatch.setattr(receiver, "sample_noise_draw_tape", unexpected_receiver)
    monkeypatch.setattr(
        functional, "sample_population_stimulation_tapes", observe_stimulation
    )
    with pytest.raises(ValueError, match="explicit StimulationDraw"):
        functional.prepare_and_rollout(
            tensors.parameters, tensors.constants, tensors.state, inputs, steps=2
        )
    assert events == []
    # The legacy local carrier does not satisfy the Network's separate Poisson
    # family. Host execution retains its sampling phase and mixed-packet error.
    with pytest.raises(ValueError, match="draws cannot be combined"):
        functional.run(
            tensors.parameters, tensors.constants, tensors.state, inputs, steps=2
        )
    assert events == ["noise", "donor_noise", "stimulation"]


@pytest.mark.parametrize("compiled", (False, True))
def test_direct_network_binding_uses_frame_for_eager_and_tape_for_compiled(compiled):
    functional, tensors = dn.func.make_functional_network(_driven_network())
    parameters = tensors.independent_parameters()
    ve, intra = _drives(functional, 1)
    tape = _inputs(ve, intra)
    frame = pytree.tree_map_only(torch.Tensor, lambda value: value[0], tape)
    expected, _ = functional.prepare_and_step(
        parameters, tensors.constants, tensors.state, inputs=frame
    )
    owner = (
        functional.compile_rollout_chunk(1, backend="aot_eager")
        if compiled
        else functional
    )
    bound = owner.bind(parameters, constants=tensors.constants)
    actual, _ = bound(tensors.state, tape if compiled else frame)
    _close(actual, expected)


def test_direct_eager_network_binding_expands_explicit_draw_frame_and_keeps_legacy_tape():
    functional, tensors = dn.func.make_functional_network(_noise_network())
    parameters = tensors.independent_parameters()
    state = _clone(tensors.state, detached=True)
    shape = functional.populations["left"].shape
    current = torch.full(shape, 0.02, dtype=DTYPE, requires_grad=True)
    inputs = {name: dn.func.Inputs(intra=current) for name in functional.populations}
    packet = functional.sample_runtime_draws(state)
    expected, _ = functional.prepare_and_step(
        parameters, tensors.constants, state, inputs=inputs, draws=packet
    )
    bound = functional.bind(parameters, constants=tensors.constants)
    actual, _ = bound(state, inputs, draws=packet)
    _close(actual, expected)
    legacy = dn.func.NetworkRolloutInput(
        {
            name: dn.func.RolloutInput(intra=current.unsqueeze(0))
            for name in functional.populations
        }
    )
    tape = pytree.tree_map_only(torch.Tensor, lambda value: value.unsqueeze(0), packet)
    compatible, _ = bound(state, legacy, draws=tape)
    _close(compatible, expected)


def test_compiled_network_runner_accepts_shared_inputs_with_recording_and_gradients():
    recorder = {"trace": dn.func.Recorder(("left.v", "right.v", "t"))}
    runner = dn.compile(
        _driven_network(),
        dt=DT,
        steps=3,
        inference=False,
        checkpoint_every=2,
        callbacks=recorder,
        backend="aot_eager",
    )
    ve, intra = _drives(runner.functional, 3)
    bound = runner.kernel.bind(runner.parameters, constants=runner.constants)
    callbacks = runner.functional.make_callbacks(recorder)
    expected, expected_aux = dn.func.run(
        bound,
        runner.initial_state,
        inputs=_legacy(_inputs(ve, intra)),
        steps=3,
        callbacks=callbacks,
    )
    gradients = torch.autograd.grad(_loss(expected), (ve, intra))
    actual, auxiliary = runner(inputs=_inputs(ve, intra))
    _close(actual, expected)
    _close(auxiliary["callbacks"]["trace"], expected_aux["callbacks"]["trace"])
    observed = torch.autograd.grad(_loss(actual), (ve, intra))
    _close(observed, gradients)
    assert all(value.abs().sum() > 0 for value in observed)


@pytest.mark.parametrize("automatic", (False, True))
def test_network_shared_inputs_preserve_shared_noise_order_and_checkpoint_continuation(
    automatic,
):
    model, native = _noise_network(), _noise_network()
    functional, tensors = dn.func.make_functional_network(model)
    state = _clone(tensors.state, detached=True)
    shape = functional.populations["left"].shape
    current = torch.linspace(0.01, 0.04, 4, dtype=DTYPE).reshape(4, 1, 1)
    current = current.expand(4, *shape).contiguous().requires_grad_()
    inputs = {name: dn.func.Inputs(intra=current) for name in functional.populations}
    packet = functional.sample_runtime_draws(state, steps=4)
    saved = _clone(packet, detached=True)
    noise = functional.sample_population_noise_tapes(state, 4)
    legacy = dn.func.NetworkRolloutInput(
        {
            name: dn.func.RolloutInput(intra=current, noise=noise[name])
            for name in functional.populations
        }
    )
    expected, _ = functional.prepare_and_rollout(
        tensors.parameters, tensors.constants, state, legacy, steps=4
    )
    bound = functional.compile_rollout_chunk(3, backend="aot_eager").bind(
        tensors.parameters, constants=tensors.constants
    )
    callbacks = functional.make_callbacks(
        {"trace": dn.func.Recorder(("left.v", "right.v", "t"))}
    )
    whole, whole_aux = dn.func.run(
        bound,
        state,
        inputs=inputs,
        draws=None if automatic else packet,
        steps=4,
        checkpoint_every=2,
        callbacks=callbacks,
    )
    actual_gradients = torch.autograd.grad(_loss(whole), current)[0]
    expected_gradients = torch.autograd.grad(_loss(expected), current)[0]
    _close(actual_gradients, expected_gradients)
    assert torch.isfinite(actual_gradients).all() and actual_gradients.abs().sum() > 0
    prefix_inputs = pytree.tree_map_only(torch.Tensor, lambda value: value[:2], inputs)
    suffix_inputs = pytree.tree_map_only(torch.Tensor, lambda value: value[2:], inputs)
    first, first_aux = dn.func.run(
        bound, state, inputs=prefix_inputs, callbacks=callbacks, checkpoint_every=2
    )
    zero, zero_aux = dn.func.run(
        bound,
        first,
        steps=0,
        callbacks=callbacks,
        callback_state=first_aux["callbacks"].state,
    )
    final, final_aux = dn.func.run(
        bound,
        zero,
        inputs=suffix_inputs,
        callbacks=callbacks,
        callback_state=zero_aux["callbacks"].state,
        checkpoint_every=2,
    )
    for index in range(4):
        _native_driven_step(
            native,
            {
                name: dn.func.Inputs(intra=current[index])
                for name in functional.populations
            },
        )
    _close(whole, expected)
    _close(final, whole)
    _close(final, functional.extract(native).state)
    _close(packet, saved)
    for key, value in whole_aux["callbacks"]["trace"].items():
        continued = torch.cat(
            (first_aux["callbacks"]["trace"][key], final_aux["callbacks"]["trace"][key])
        )
        _close(continued, value)
        assert continued.shape[0] == 5


@pytest.mark.parametrize("method", ("step", "prepare_and_step"))
def test_network_shared_inputs_requalify_source_after_authored_mapping(method):
    model = _driven_network()
    functional, tensors = dn.func.make_functional_network(model)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    class MutatingMapping(dict):
        def items(self):
            model.eval()
            return super().items()

    inputs = MutatingMapping({"left": dn.func.Inputs()})
    with pytest.raises(dn.func.FunctionalizationError, match="mode|training|changed"):
        getattr(functional, method)(
            tensors.parameters,
            prepared if method == "step" else tensors.constants,
            tensors.state,
            inputs=inputs,
        )


@pytest.mark.parametrize("mutation", ("write", "replace"))
def test_network_shared_input_checkpoint_rejects_late_mutation(mutation):
    functional, tensors = dn.func.make_functional_network(_driven_network())
    ve, intra = _drives(functional, 3)
    inputs = _inputs(ve, intra)
    bound = functional.compile_rollout_chunk(2, backend="aot_eager").bind(
        tensors.parameters, constants=tensors.constants
    )
    final, _ = dn.func.run(bound, tensors.state, inputs=inputs, checkpoint_every=2)
    if mutation == "write":
        with torch.no_grad():
            ve.add_(0.1)
    else:
        inputs["left"] = dn.func.Inputs(ve=ve.clone(), intra=intra)
    error = RuntimeError if mutation == "write" else dn.func.FunctionalizationError
    message = (
        "input tapes changed|modified by an inplace operation"
        if mutation == "write"
        else "input tapes changed"
    )
    with pytest.raises(error, match=message):
        _loss(final).backward()


def test_network_shared_input_horizons_and_population_names_remain_strict():
    functional, tensors = dn.func.make_functional_network(_driven_network())
    bound = functional.bind(tensors.parameters, constants=tensors.constants)
    ve, intra = _drives(functional, 3)
    with pytest.raises(ValueError, match="known Network names"):
        dn.func.run(
            bound, tensors.state, inputs={"unknown": dn.func.Inputs(intra=intra)}
        )
    with pytest.raises(ValueError, match="exactly steps samples"):
        dn.func.run(bound, tensors.state, inputs=_inputs(ve, intra), steps=2)
