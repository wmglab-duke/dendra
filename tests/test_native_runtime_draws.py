"""Native single steps consume canonical draws without sampling replacements."""

from __future__ import annotations

import os

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.rng import RNGModule
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_material_stochastic_sampling import (
    _population as _material_population,
)
from tests.func.test_state_sde_execution import DT, _HeunLeak, _OULeak
from tests.func.test_state_sde_execution import _population as _sde_population
from tests.func.test_state_sde_execution import _SharedLeak
from tests.func.test_stimulation_rng_sharing import DT as POISSON_DT
from tests.func.test_stimulation_rng_sharing import _population as _poisson_population

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    threads, rng = torch.get_num_threads(), torch.get_rng_state().clone()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0,
                REQUIRE_GRAD=1,
                IMEM=False,
                DTYPE=torch.float64,
                DEVICE="cpu",
                CHECK_PURITY=False,
            ),
        ):
            yield
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


def _changed_draws(packet):
    return dn.func.RuntimeDraws(
        pytree.tree_map_only(
            torch.Tensor,
            lambda value: (0.2 + 0.4 * value.detach()).clone().requires_grad_(),
            packet.draws,
        ),
        _snapshot(packet.rng),
    )


def _base_draws(packet):
    return tuple(
        value for value in pytree.tree_leaves(packet.draws) if torch.is_tensor(value)
    )


def _parameters(model):
    return dict(model.named_parameters(remove_duplicate=False))


def _same_parameters(model, saved):
    current = _parameters(model)
    assert current.keys() == saved.keys()
    assert all(current[name] is value for name, value in saved.items())


def _forbid_owned_sampling(monkeypatch, model, *generators):
    owned = {id(generator) for generator in generators}
    for module in model.modules():
        if isinstance(module, RNGModule):
            owned.add(id(module._cpu_gen))
            owned.update(id(generator) for generator in module._device_gens.values())
    for name in ("rand", "randn"):
        original = getattr(torch, name)

        def guarded(*args, _original=original, **kwargs):
            if id(kwargs.get("generator")) in owned:
                raise AssertionError(
                    "Explicit native draws sampled a model-owned generator"
                )
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, guarded)


@pytest.mark.parametrize(
    "mechanism", (_SharedLeak, _HeunLeak), ids=("noise_sde", "heun")
)
def test_population_explicit_draw_values_gradients_and_native_parameter_ownership(
    mechanism, monkeypatch
):
    source, native, automatic = (_sde_population(mechanism) for _ in range(3))
    functional, tensors = dn.func.make_functional(source, dt=DT)
    packet = _changed_draws(functional.sample_runtime_draws(tensors.state))
    saved_packet, saved_source = (
        _snapshot(packet),
        _snapshot(source.state_dict_for_checkpoint()),
    )
    parameters, global_rng = _parameters(native), torch.get_rng_state().clone()
    expected, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state, draws=packet
    )
    callbacks = dn.func.make_callbacks(
        native, {"trace": dn.Recorder(("v", "sde.x", "t"))}, dt=DT
    )
    with monkeypatch.context() as guard:
        _forbid_owned_sampling(guard, native)
        result = dn.step(native, dt=DT, draws=packet, callbacks=callbacks)
    automatic.step(DT)
    _close(functional.extract(native).state, expected)
    assert not torch.equal(native.mech.sde.x, automatic.mech.sde.x)
    _close(result["trace"]["v"][-1], native.v)
    assert result["trace"]["t"].shape == (2,)
    source_state = next(iter(source.mech.sde.DE.values()))
    native_state = next(iter(native.mech.sde.DE.values()))
    targets = (native.mech.sde.g_param, native_state.sigma_param)
    references = (source.mech.sde.g_param, source_state.sigma_param)
    if mechanism is _SharedLeak:
        targets += (native_state.eta_mu_param,)
        references += (source_state.eta_mu_param,)
        assert not native_state.eta.requires_grad and native_state.eta.grad_fn is None
    actual = torch.autograd.grad(
        native.v.square().mean(), (*targets, *_base_draws(packet)), allow_unused=True
    )
    wanted = torch.autograd.grad(
        expected["integrator"]["v"].square().mean(),
        (*references, *_base_draws(packet)),
        allow_unused=True,
    )
    _close(actual[:2], wanted[:2])
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in actual[:2])
    assert all(value is None or value.count_nonzero() == 0 for value in actual[2:])
    _same_parameters(native, parameters)
    _assert_packet_equal(packet, saved_packet)
    _assert_packet_equal(source.state_dict_for_checkpoint(), saved_source)
    _close(torch.get_rng_state(), global_rng)


def _material_sde_population():
    model = _material_population()
    model.insert(_OULeak.rename("sde"))
    model.initialize()
    gate = next(iter(model.mech.sde.DE.values()))
    gate.x_dW_rng = model.mech.current.stream
    model.mech.material_processes["random"].stream = model.mech.current.stream
    model.initialize()
    model.integrator._initialize(model, model.t.new_tensor(DT), force=True)
    return model


def test_mixed_material_noise_and_state_sde_share_the_replayed_native_stream():
    source, native = _material_sde_population(), _material_sde_population()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    packet = _changed_draws(functional.sample_runtime_draws(tensors.state))
    saved_packet, parameters = _snapshot(packet), _parameters(native)
    gate = next(iter(native.mech.sde.DE.values()))
    stream, generator = native.mech.current.stream, native.mech.current.stream.rng
    assert gate.x_dW_rng is native.mech.material_processes["random"].stream is stream
    expected, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state, draws=packet
    )
    assert dn.step_population(native, DT, draws=packet) is native
    _close(functional.extract(native).state, expected)
    assert gate.x_dW_rng is native.mech.material_processes["random"].stream is stream
    assert stream.rng is generator
    assert not native.mech.material_processes["random"].drive.requires_grad
    _same_parameters(native, parameters)
    _assert_packet_equal(packet, saved_packet)


def test_explicit_then_automatic_steps_continue_through_a_native_checkpoint():
    source, native, receiver = (_sde_population(_SharedLeak) for _ in range(3))
    functional, tensors = dn.func.make_functional(source, dt=DT)
    packet = _changed_draws(functional.sample_runtime_draws(tensors.state))
    expected, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state, draws=packet
    )
    native.step(DT, draws=packet)
    for index in range(2):
        expected, auxiliary = dn.func.run(
            functional,
            dn.func.Continuation(expected, tensors.parameters, tensors.constants),
            steps=1,
        )
        if index == 0:
            native.step(DT)
            receiver.restore_dict_from_checkpoint(
                _snapshot(native.state_dict_for_checkpoint())
            )
            _close(functional.extract(receiver).state, expected)
        else:
            dn.step(receiver, DT)
            _close(functional.extract(receiver).state, expected)
    assert receiver.t.item() == pytest.approx(3 * DT)
    assert auxiliary["continuation"].callback_state is None


@pytest.mark.parametrize("invalid", ("type", "noise", "dtype", "rng", "tape"))
def test_population_bad_runtime_packets_reject_before_sampling_or_advancing(
    invalid, monkeypatch
):
    model = _sde_population(_SharedLeak)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    packet = functional.sample_runtime_draws(tensors.state)
    if invalid == "type":
        packet = {}
    elif invalid == "noise":
        packet = packet._replace(draws={**packet.draws, "noise": {}})
    elif invalid == "dtype":
        packet = packet._replace(
            draws=pytree.tree_map_only(
                torch.Tensor, lambda value: value.float(), packet.draws
            )
        )
    elif invalid == "rng":
        packet = packet._replace(rng={**packet.rng, "model": {}})
    else:
        packet = functional.sample_runtime_draws(tensors.state, steps=1)
    saved, parameters, global_rng = (
        _snapshot(model.state_dict_for_checkpoint()),
        _parameters(model),
        torch.get_rng_state().clone(),
    )
    with monkeypatch.context() as guard:
        _forbid_owned_sampling(guard, model)
        with pytest.raises((TypeError, ValueError, dn.func.FunctionalizationError)):
            model.step(DT, draws=packet)
    _assert_packet_equal(model.state_dict_for_checkpoint(), saved)
    _same_parameters(model, parameters)
    _close(torch.get_rng_state(), global_rng)


def test_randomized_poisson_runtime_draws_reject_before_waveform_or_generator_use(
    monkeypatch,
):
    model, _, pulse = _poisson_population()
    functional, tensors = dn.func.make_functional(model, dt=POISSON_DT)
    packet = functional.sample_runtime_draws(tensors.state)
    saved, rng = (
        _snapshot(model.state_dict_for_checkpoint()),
        pulse.generator.get_state().clone(),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported replay evaluated a randomized waveform")

    monkeypatch.setattr(pulse, "forward", forbidden)
    with pytest.raises(
        (ValueError, dn.func.FunctionalizationError), match="Poisson|poisson"
    ):
        model.step(POISSON_DT, draws=packet)
    _assert_packet_equal(model.state_dict_for_checkpoint(), saved)
    _close(pulse.generator.get_state(), rng)


def test_explicit_state_draws_reach_an_actual_compiled_native_kernel():
    source = _sde_population(_HeunLeak)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    packet = _changed_draws(functional.sample_runtime_draws(tensors.state))
    expected, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state, draws=packet
    )
    second = _changed_draws(functional.sample_runtime_draws(expected))
    assert any(
        not torch.equal(first, following)
        for first, following in zip(
            _base_draws(packet), _base_draws(second), strict=True
        )
    )
    final, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, expected, draws=second
    )
    automatic, _ = dn.func.run(
        functional,
        dn.func.Continuation(final, tensors.parameters, tensors.constants),
        steps=1,
    )
    with dn.ctx(JIT=1, BACKEND="aot_eager", FULLGRAPH=True):
        native = _sde_population(_HeunLeak)
        native.step(DT, draws=packet)
        kernels = dict(native.integrator._compiled_kernels)
        assert kernels
        assert all(
            "_native_sde_draws" not in vars(module)
            and "_native_runtime_noise_draws" not in vars(module)
            for module in native.modules()
        )
        _close(functional.extract(native).state, expected)
        native.step(DT, draws=second)
        assert native.integrator._compiled_kernels.keys() == kernels.keys()
        assert all(
            native.integrator._compiled_kernels[name] is kernel
            for name, kernel in kernels.items()
        )
        assert all(
            "_native_sde_draws" not in vars(module)
            and "_native_runtime_noise_draws" not in vars(module)
            for module in native.modules()
        )
        _close(functional.extract(native).state, final)
        assert native.integrator.backend == "aot_eager"
    # Automatic State SDE sampling retains the native backend's existing
    # eager requirement; explicit replay above exercises fullgraph capture.
    native.step(DT)
    assert all(
        "_native_sde_draws" not in vars(module)
        and "_native_runtime_noise_draws" not in vars(module)
        for module in native.modules()
    )
    _close(functional.extract(native).state, automatic)
    sigma = next(iter(native.mech.sde.DE.values())).sigma_param
    reference_sigma = next(iter(source.mech.sde.DE.values())).sigma_param
    gradients = torch.autograd.grad(
        native.v.square().mean(), (native.mech.sde.g_param, sigma)
    )
    wanted = torch.autograd.grad(
        automatic["integrator"]["v"].square().mean(),
        (source.mech.sde.g_param, reference_sigma),
    )
    _close(gradients, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in gradients)


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_native_runtime_draws_preserve_float32_values_and_gradients():
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DEVICE="mps", DTYPE=torch.float32):
        source, native = _sde_population(_SharedLeak), _sde_population(_SharedLeak)
        functional, tensors = dn.func.make_functional(source, dt=DT)
        packet = _changed_draws(functional.sample_runtime_draws(tensors.state))
        saved = _snapshot(packet)
        expected, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, tensors.state, draws=packet
        )
        dn.step_population(native, DT, draws=packet)
        torch.testing.assert_close(
            native.v, expected["integrator"]["v"], rtol=2e-5, atol=2e-6
        )
        sigma = next(iter(native.mech.sde.DE.values())).sigma_param
        reference_sigma = next(iter(source.mech.sde.DE.values())).sigma_param
        actual = torch.autograd.grad(native.v.square().mean(), sigma)[0]
        wanted = torch.autograd.grad(
            expected["integrator"]["v"].square().mean(), reference_sigma
        )[0]
        torch.testing.assert_close(actual, wanted, rtol=2e-5, atol=2e-6)
        assert actual.isfinite().all() and actual.abs().sum() > 0
        _assert_packet_equal(packet, saved)
