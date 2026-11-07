"""Custom explicit-draw recipes compose with native Network initialization."""

from __future__ import annotations

import pytest
import torch
from torch.nn.utils.parametrize import register_parametrization

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._network_initialization import initialize_host
from dendra.models.distributions import Distribution, Normal
from dendra.models.mod import graded_syn, pas
from tests.func.test_network import DT, DTYPE, _clone, _close, _loss, _native_loss
from tests.func.test_network_connection_sampling import (
    _check_native,
    _materialized_loss,
    _native_materialized_loss,
)
from tests.networks.test_functional_event_delivery import ThresholdRelay

pytestmark = pytest.mark.cpu
_COLLECTIONS = ("synapses", "continuous_synapses")


class _TwoUniforms(Distribution):
    """A scalar recipe whose independent draw has an extra two-lane axis."""

    def __init__(self, location, scale):
        super().__init__()
        self.location = torch.nn.Parameter(torch.tensor(location, dtype=DTYPE))
        self.scale = torch.nn.Parameter(torch.tensor(scale, dtype=DTYPE))
        self.register_buffer("offset", torch.tensor(0.0, dtype=DTYPE))

    def sample_draw_schema(self, sample_shape=torch.Size()):
        return (
            torch.Size(sample_shape) + (2,),
            self.location.dtype,
            self.location.device,
        )

    def sample_draw(self, sample_shape=torch.Size(), *, generator=None):
        shape, dtype, device = self.sample_draw_schema(sample_shape)
        return torch.rand(shape, dtype=dtype, device=device, generator=generator)

    def sample_from_draw(self, draw):
        return (
            self.location
            + self.scale * (draw[..., 0] + 2 * draw[..., 1] - 1.5)
            + self.offset
        )


class _AmbientTwoUniforms(_TwoUniforms):
    def sample_draw_schema(self, sample_shape=torch.Size()):
        return (
            torch.Size(sample_shape) + (2,),
            torch.get_default_dtype(),
            self.location.device,
        )


class _ScaledLocation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.factor = torch.nn.Parameter(torch.tensor(1.1, dtype=DTYPE))

    def forward(self, value):
        return value * self.factor


@pytest.fixture(autouse=True)
def _environment():
    previous_threads = torch.get_num_threads()
    previous_rng = torch.get_rng_state()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_rng_state(previous_rng)
        torch.set_num_threads(previous_threads)


def _network(*, parametrized=False, interpolated=False, cache=None, ambient_draw=False):
    torch.manual_seed(4378)
    distribution = _AmbientTwoUniforms if ambient_draw else _TwoUniforms
    weights = distribution(0.003, 0.0002)
    delays = distribution(1.1 * DT, 0.02 * DT)
    if parametrized:
        register_parametrization(weights, "location", _ScaledLocation())
    builtin_weights = Normal(mean=0.002, std=0.0001)
    builtin_delays = Normal(mean=1.1 * DT, std=0.01 * DT)
    cell = dn.SingleCompartment(
        N=2, C=1, integrator=dn.bwd_euler_sc(imem=False), v_init=-65.0
    )
    cell.insert(pas, g=0.001, e=-65.0)
    cell.insert(ThresholdRelay.rename("relay"))
    cell.insert(graded_syn.rename("analog"), e=-10.0)
    model = dn.Network(
        {"cell": cell}, netcon_delay_backend="dense", netcon_train_backend="dense"
    )
    for weight, delay in ((weights, delays), (builtin_weights, builtin_delays)):
        model.connect_one_to_one(
            cell[:],
            cell[:],
            cell.mech.relay,
            pre_var="mech.relay.level",
            weight=weight,
            delay=delay,
            allow_autapses=True,
        )
        model.connect_continuous_one_to_one(
            cell[:],
            cell[:],
            cell.mech.analog,
            pre_var="mech.relay.level",
            input="g_pre",
            weight=weight,
            delay=delay,
            transform=torch.sigmoid,
            allow_autapses=True,
        )
    model.train()
    model.initialize(DT)
    model.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=interpolated,
        diff_spiking=True,
        diff_scheduled_times=False,
        tau=0.23,
        taps=3,
        sigma=0.43,
        train_delay_backend="dense",
    )
    model.initialize(DT)
    if cache is not None:
        if cache == "network":
            model._enable_synapse_state_cache_recording()
        with torch.no_grad():
            for _ in range(2):
                model.step()
            if cache == "network":
                model.cache_state()
            else:
                model.cell.cache("_steady_state")
        if cache == "network":
            model._disable_synapse_state_cache_recording(release=True)
    return model, weights, delays


def _sources(packet):
    return next(iter(packet.parameters["synapses"].values()))


def _targets(packet, parametrized):
    values = _sources(packet)
    names = (
        ("parametrizations.location.original", "parametrizations.location.0.factor")
        if parametrized
        else ("location",)
    )
    return (
        *(values[f"weight.weights.0.param.{name}"] for name in names),
        values["weight.weights.0.param.scale"],
        values["delay_ms.weights.0.param.location"],
        values["delay_ms.weights.0.param.scale"],
    )


def _native_targets(weights, delays, parametrized):
    locations = (
        (
            weights.parametrizations.location.original,
            weights.parametrizations.location[0].factor,
        )
        if parametrized
        else (weights.location,)
    )
    return (*locations, weights.scale, delays.location, delays.scale)


@pytest.mark.parametrize("parametrized", (False, True))
@pytest.mark.parametrize("check_purity", (False, True))
def test_custom_snapshot_keeps_shared_raw_recipes_multilane_draws_and_gradients(
    parametrized, check_purity
):
    model, weights, delays = _network(parametrized=parametrized)
    before = _clone(model.state_dict(), detached=True)
    rng = torch.get_rng_state().clone()
    with dn.ctx(CHECK_PURITY=check_purity):
        functional, packet = dn.func.make_functional_network(model)
    _check_native(functional, packet, model)
    event = _sources(packet)
    analog = next(iter(packet.parameters["continuous_synapses"].values()))
    assert "weight.w" not in event
    for key in event:
        if key.startswith(("weight.weights.", "delay_ms.weights.")):
            assert event[key] is analog[key]
    for collection in _COLLECTIONS:
        for constants in packet.constants[collection].values():
            assert constants["draw.weights.0"].shape == (2, 2)
            assert constants["draw.delays.0"].shape == (2, 2)
            assert constants["draw.weights.1"].shape == (2,)
            assert constants["draw_valid.weights"]
            assert constants["draw_valid.delays"]
    observed = torch.autograd.grad(
        _materialized_loss(functional, packet), _targets(packet, parametrized)
    )
    expected = torch.autograd.grad(
        _native_materialized_loss(model), _native_targets(weights, delays, parametrized)
    )
    _close(observed, expected)
    assert all(
        gradient.isfinite().all() and gradient.abs().sum() > 0 for gradient in observed
    )
    _close(model.state_dict(), before)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize("reinit_weights", (False, True))
@pytest.mark.parametrize("reinit_delays", (False, True))
def test_custom_refresh_and_retention_follow_native_mixed_block_rng_order(
    reinit_weights, reinit_delays
):
    model, *_ = _network(parametrized=True)
    native, native_weights, native_delays = _network(parametrized=True)
    functional, packet = dn.func.make_functional_network(model)
    before = _clone(packet, detached=True)
    rng = torch.get_rng_state().clone()
    flags = dict(reinit_weights=reinit_weights, reinit_delays=reinit_delays)
    draw = functional.sample_connection_initialization_draw(*packet, **flags)
    repeated = functional.sample_connection_initialization_draw(*packet, **flags)
    _close(draw, repeated)
    initialized = functional.initialize(
        *packet,
        functional.initialization_inputs._replace(connection_draw=draw, **flags),
    )
    _close(packet, before)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    native.initialize(DT, **flags)
    _check_native(functional, initialized, native)
    torch.testing.assert_close(
        initialized.state["connection_sampling"]["rng"]["cpu"],
        torch.get_rng_state(),
        rtol=0,
        atol=0,
    )
    for collection in _COLLECTIONS:
        for roles in draw.values[collection].values():
            assert bool(roles["weights"]) is reinit_weights
            assert bool(roles["delays"]) is reinit_delays
            for enabled, role in (
                (reinit_weights, "weights"),
                (reinit_delays, "delays"),
            ):
                if enabled:
                    assert roles[role]["0"].shape == (2, 2)
                    assert roles[role]["1"].shape == (2,)
    if reinit_weights and reinit_delays:
        observed = torch.autograd.grad(
            _materialized_loss(functional, initialized), _targets(initialized, True)
        )
        expected = torch.autograd.grad(
            _native_materialized_loss(native),
            _native_targets(native_weights, native_delays, True),
        )
        _close(observed, expected)
    actual, _ = functional.prepare_and_step(*initialized)
    native.step()
    _close(actual, functional.extract(native).state)


@pytest.mark.parametrize("cache", ("population", "network"))
def test_custom_sampling_composes_with_owned_initialization_caches(cache):
    model, *_ = _network(parametrized=True, cache=cache)
    native, native_weights, native_delays = _network(parametrized=True, cache=cache)
    functional, packet = dn.func.make_functional_network(model)
    rng = torch.get_rng_state().clone()
    initialized = initialize_host(functional, *packet, functional.initialization_inputs)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    native.initialize(DT)
    _check_native(functional, initialized, native)
    _close(initialized.state, functional.extract(native).state)
    observed = torch.autograd.grad(
        _materialized_loss(functional, initialized), _targets(initialized, True)
    )
    expected = torch.autograd.grad(
        _native_materialized_loss(native),
        _native_targets(native_weights, native_delays, True),
    )
    _close(observed, expected)
    assert all(
        gradient.isfinite().all() and gradient.abs().sum() > 0 for gradient in observed
    )
    actual, _ = functional.prepare_and_step(*initialized)
    native.step()
    _close(actual, functional.extract(native).state)


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_custom_recipes_compile_record_continue_and_restore_native_checkpoints(
    execution,
):
    model, *_ = _network(parametrized=True, interpolated=execution == "scan")
    native, native_weights, native_delays = _network(
        parametrized=True, interpolated=execution == "scan"
    )
    name = next(iter(model.synapses))
    keys = (
        "weight.weights.0.param.parametrizations.location.original",
        "weight.weights.0.param.parametrizations.location.0.factor",
        "weight.weights.0.param.scale",
        "delay_ms.weights.0.param.location",
        "delay_ms.weights.0.param.scale",
    )
    runner = dn.compile(
        model,
        dt=DT,
        steps=4,
        inference=execution == "while",
        execution=execution,
        initialization="fresh",
        trainable=tuple(f"synapses.{name}.{key}" for key in keys),
        callbacks={"trace": dn.func.Recorder(("cell.v", "t"))},
        backend="aot_eager",
        **({"checkpoint_every": 2} if execution == "scan" else {}),
    )
    frozen = _clone(runner.initial_state, detached=True)
    rng = torch.get_rng_state().clone()
    prefix, first = runner()
    repeated, _ = runner()
    _close(prefix, repeated)
    parameters, constants = first["parameters"], first["constants"]
    final, second = runner(
        prefix,
        parameters=parameters,
        constants=constants,
        callback_state=first["callbacks"].state,
    )
    _close(runner.initial_state, frozen)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    native.initialize(DT)
    frames = [native.cell.v.detach().clone()]
    for _ in range(8):
        native.step()
        frames.append(native.cell.v.detach().clone())
    _close(final, runner.functional.extract(native).state)
    trace = torch.cat(
        (first["callbacks"]["trace"]["cell.v"], second["callbacks"]["trace"]["cell.v"])
    )
    torch.testing.assert_close(trace, torch.stack(frames))
    if execution == "scan":
        targets = tuple(parameters["synapses"][name][key] for key in keys)
        observed = torch.autograd.grad(_loss(final), targets)
        expected = torch.autograd.grad(
            _native_loss(native), _native_targets(native_weights, native_delays, True)
        )
        _close(observed, expected)
        assert all(
            gradient.isfinite().all() and gradient.abs().sum() > 0
            for gradient in observed
        )
    else:
        assert runner.execution_report().strategy == "compiled_while_loop"
    # Model-state provenance and explicit runtime RNG both survive publication.
    target, *_ = _network(parametrized=True, interpolated=execution == "scan")
    target.load_state_dict(native.state_dict())
    target.initialize(DT, reinit_weights=False, reinit_delays=False)
    runner.functional.commit_state_(target, final)
    checkpoint = _clone(target.state_dict_for_checkpoint(), detached=True)
    resumed, *_ = _network(parametrized=True, interpolated=execution == "scan")
    resumed.load_state_dict(target.state_dict())
    resumed.initialize(DT, reinit_weights=False, reinit_delays=False)
    resumed.restore_dict_from_checkpoint(checkpoint)
    restored = runner.functional.extract(resumed)
    _close(restored.state["connection_sampling"], final["connection_sampling"])
    _check_native(runner.functional, restored, resumed)
    continued, _ = runner.functional.prepare_and_step(parameters, constants, final)
    resumed.step()
    _close(continued, runner.functional.extract(resumed).state)


@pytest.mark.parametrize(
    "captured_dtype,fresh_dtype",
    ((torch.float64, torch.float32), (torch.float32, torch.float64)),
)
def test_checkpoint_keeps_custom_historical_draw_dtype_and_raw_gradients(
    captured_dtype, fresh_dtype
):
    # All model tensors stay float64. Only the independent sampler's ambient
    # dtype changes between saving and restoring the native checkpoint.
    source, *_ = _network(parametrized=True, ambient_draw=True)
    target, *_ = _network(parametrized=True, ambient_draw=True)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(captured_dtype)
        source.initialize(DT)
        original, packet = dn.func.make_functional_network(source)
        expected = torch.autograd.grad(
            _materialized_loss(original, packet), _targets(packet, True)
        )
        original.commit_state_(source, packet.state)
        model_state = _clone(source.state_dict(), detached=True)
        checkpoint = _clone(source.state_dict_for_checkpoint(), detached=True)
        torch.set_default_dtype(fresh_dtype)
        rng = torch.get_rng_state().clone()
        target.load_state_dict(model_state)
        target.initialize(DT, reinit_weights=False, reinit_delays=False)
        target.restore_dict_from_checkpoint(checkpoint)
        assert all(value.dtype is DTYPE for value in target.parameters())
        for collection in _COLLECTIONS:
            for name, connection in getattr(target, collection).items():
                saved = getattr(source, collection)[name]
                for role in ("weight", "delay_ms"):
                    restored_draw = getattr(connection, role).sampling_draws["0"]
                    saved_draw = getattr(saved, role).sampling_draws["0"]
                    assert restored_draw.dtype is captured_dtype
                    torch.testing.assert_close(
                        restored_draw, saved_draw, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        getattr(connection, role).derive_values(
                            getattr(connection, role).sampling_draws
                        ),
                        getattr(connection, role).w,
                        rtol=0,
                        atol=0,
                    )
        restored, resumed = dn.func.make_functional_network(target)
        _check_native(restored, resumed, target)
        _close(
            resumed.state["connection_sampling"], packet.state["connection_sampling"]
        )
        observed = torch.autograd.grad(
            _materialized_loss(restored, resumed), _targets(resumed, True)
        )
        torch.testing.assert_close(observed, expected, rtol=1e-13, atol=1e-15)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        actual, _ = restored.prepare_and_step(*resumed)
        target.step()
        _close(actual, restored.extract(target).state)
        # A subsequent refresh uses the new dtype and continues from the exact
        # saved RNG frame, without changing the retained checkpoint's draw lane.
        draw = restored.sample_connection_initialization_draw(*resumed)
        for collection in _COLLECTIONS:
            for roles in draw.values[collection].values():
                assert roles["weights"]["0"].dtype is fresh_dtype
                assert roles["delays"]["0"].dtype is fresh_dtype
        initialized = restored.initialize(
            *resumed, restored.initialization_inputs._replace(connection_draw=draw)
        )
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        target.initialize(DT)
        _check_native(restored, initialized, target)
        torch.testing.assert_close(
            initialized.state["connection_sampling"]["rng"]["cpu"],
            torch.get_rng_state(),
            rtol=0,
            atol=0,
        )
    finally:
        torch.set_default_dtype(previous_dtype)
