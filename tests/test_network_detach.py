"""Native connected training truncates runtime history without resetting it."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.distributions import Normal
from dendra.models.mod import graded_syn, pas
from dendra.models.networks.net import make_weight
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_continuation_checkpoints import _disk
from tests.func.test_network import DT, _mixed_network, _stim_network
from tests.func.test_network_pending_calendar import _network as _pending_network
from tests.func.test_network_pending_calendar import _pending_count
from tests.func.test_network_source_history import _stim_history
from tests.func.test_stimulation_rng_sharing import DT as STOCHASTIC_DT
from tests.func.test_stimulation_rng_sharing import _network as _stochastic_network

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64, IMEM=False
            ),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


class _Release(torch.nn.Module):
    def __init__(self, *, dtype=torch.float64, device="cpu"):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.02, dtype=dtype, device=device))

    def forward(self, voltage):
        return torch.sigmoid(self.scale * voltage)


def _continuous(*, stochastic=False, dtype=torch.float64, device="cpu"):
    with dn.ctx(DTYPE=dtype, DEVICE=device):
        shared_g = torch.nn.Parameter(torch.tensor(0.001, dtype=dtype, device=device))
        populations = {}
        for name, voltage, reversal in (("pre", -63.0, -55.0), ("post", -67.0, -70.0)):
            cell = dn.SingleCompartment(
                N=1, C=1, cm=1.2, v_init=voltage, integrator=dn.bwd_euler_sc(imem=False)
            )
            cell.insert(pas, g=shared_g, e=reversal)
            if name == "post":
                cell.insert(graded_syn.rename("analog"), e=-10.0)
            populations[name] = cell
        model = dn.Network(populations)
        source = (
            Normal(mean=0.002, std=0.0001).to(dtype=dtype, device=device)
            if stochastic
            else 0.002
        )
        model.connect_continuous_one_to_one(
            model.pre[:],
            model.post[:],
            model.post.mech.analog,
            input="g_pre",
            transform=_Release(dtype=dtype, device=device),
            weight=source,
            delay=0.0,
        )
        model.train()
        model.initialize(DT)
        return model, next(iter(model.continuous_synapses.values()))


def _continuous_oracle(model, connection, before, *, steps, weight=None):
    """Direct backward Euler, observing the old presynaptic value first."""
    pre, post = before
    g = model.pre.mech.pas.g_param
    cm_pre, cm_post = model.pre.cm_param(), model.post.cm_param()
    weight = (
        connection.weight.derive_values(connection.weight.sampling_draws)
        if weight is None
        else weight
    )
    for _ in range(steps):
        release = weight.reshape_as(pre) * torch.sigmoid(
            connection.transform.scale * pre
        )
        post = (cm_post * post + 1000 * DT * (-70.0 * g - 10.0 * release)) / (
            cm_post + 1000 * DT * (g + release)
        )
        pre = (cm_pre * pre - 1000 * DT * 55.0 * g) / (cm_pre + 1000 * DT * g)
    return pre, post


def _sources(model):
    return dict(model.named_parameters(remove_duplicate=False))


def _assert_sources(model, saved):
    current = _sources(model)
    assert current.keys() == saved.keys()
    assert all(current[name] is value for name, value in saved.items())


def _assert_clocks(model):
    for collection in (model.synapses, model.continuous_synapses):
        for connection in collection.values():
            assert connection.t is model.t
    for population in model.populations.values():
        _close(population.t, model.t.to(population.t))
        for mechanism in population.mech.mechanisms.values():
            assert mechanism.t is population.t


@contextmanager
def _forbid_initialization(model):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Network.detach continuation must not initialize or sample"
        )

    bindings = [(model, "initialize")]
    for population in model.populations.values():
        bindings.extend(
            ((population, "initialize"), (population, "populate_parameter_buffers"))
        )
    for collection in (model.synapses, model.continuous_synapses):
        for connection in collection.values():
            bindings.extend(
                (
                    (connection, "initialize"),
                    (connection.weight, "init"),
                    (connection.delay_ms, "init"),
                    (connection.weight, "sample_draws"),
                    (connection.delay_ms, "sample_draws"),
                )
            )
    saved = []
    try:
        for owner, name in bindings:
            saved.append((owner, name, name in vars(owner), vars(owner).get(name)))
            setattr(owner, name, forbidden)
        yield
    finally:
        # Restoring inherited methods as instance-bound attributes would change
        # the source contract, even though calling those methods still works.
        for owner, name, existed, value in reversed(saved):
            if existed:
                setattr(owner, name, value)
            else:
                delattr(owner, name)


@pytest.mark.parametrize("stochastic", (False, True), ids=("positive", "normal"))
def test_connected_optimizer_segments_match_direct_values_and_raw_gradients(
    stochastic,
):
    model, connection = _continuous(stochastic=stochastic)
    model.run(2 * DT)
    (model.pre.v.square().sum() + model.post.v.square().sum()).backward()
    before = model.pre.v.detach().clone(), model.post.v.detach().clone()
    checkpoint = _snapshot(model.state_dict_for_checkpoint())
    sources = _sources(model)
    draws = _snapshot(connection.weight.sampling_draws)
    rng = torch.get_rng_state().clone()
    _assert_clocks(model)
    assert model.detach() is model
    _assert_packet_equal(model.state_dict_for_checkpoint(), checkpoint)
    assert model.pre.mech.pas.g_param is model.post.mech.pas.g_param
    _assert_clocks(model)
    assert not model.pre.v.requires_grad and not model.post.v.requires_grad
    raw_weight = connection.weight.weights[0].param
    targets = (
        (raw_weight.mean, raw_weight.std) if stochastic else (raw_weight.rho,)
    ) + (model.pre.mech.pas.g_param, connection.transform.scale, model.pre.cm_param.rho)
    with torch.no_grad():
        model.pre.mech.pas.g_param.mul_(1.15)
        model.pre.cm_param.rho.add_(0.1)
        connection.transform.scale.add_(0.002)
        if stochastic:
            raw_weight.mean.add_(0.0004)
            raw_weight.std.mul_(1.3)
        else:
            raw_weight.rho.add_(0.3)
    for parameter in model.parameters():
        parameter.grad = None
    expected = _continuous_oracle(model, connection, before, steps=3)
    wanted = torch.autograd.grad(
        sum(value.square().sum() for value in expected), targets
    )
    with _forbid_initialization(model):
        model.longrun(3 * DT, chunklength=2)
    _close((model.pre.v, model.post.v), expected)
    actual = torch.autograd.grad(
        model.pre.v.square().sum() + model.post.v.square().sum(), targets
    )
    _close(actual, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in actual)
    _assert_packet_equal(connection.weight.sampling_draws, draws)
    _close(torch.get_rng_state(), rng)
    _close(model.t, checkpoint["t"] + 3 * DT)
    _assert_sources(model, sources)


@pytest.mark.parametrize("context", ("eval", "no_grad", "inference"))
def test_zero_and_inference_calls_defer_connected_materialization_refresh(context):
    model, connection = _continuous()
    model.run(2 * DT)
    model.post.v.square().sum().backward()
    model._set_duration_remainder(0.4 * DT)
    model.detach()
    old_weight = connection.weight.w.detach().clone()
    raw = connection.weight.weights[0].param
    with torch.no_grad():
        raw.rho.add_(0.5)
    frozen = _snapshot(model.state_dict_for_checkpoint())
    assert model.run(0.0) is None
    assert model.longrun(0.0, chunklength=2) is None
    assert model.longrun_checkpointed(0.0, chunklength=2, safe_checkpoint=True) is None
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
    _close(connection.weight.w, old_weight)
    if context == "eval":
        model.eval()
    execution = (
        torch.no_grad()
        if context == "no_grad"
        else torch.inference_mode()
        if context == "inference"
        else nullcontext()
    )
    with execution:
        model.step()
    _close(connection.weight.w, old_weight)
    before = model.pre.v.detach().clone(), model.post.v.detach().clone()
    model.train()
    expected = _continuous_oracle(model, connection, before, steps=1)
    model.step()
    _close((model.pre.v, model.post.v), expected)
    observed = torch.autograd.grad(model.post.v.square().sum(), raw.rho)[0]
    wanted = torch.autograd.grad(expected[1].square().sum(), raw.rho)[0]
    _close(observed, wanted)
    assert observed.isfinite().all() and observed.abs().sum() > 0
    _close(model._duration_remainder, frozen["duration_remainder"])


@pytest.mark.parametrize("taps", (2, 3))
def test_source_history_uses_updated_delivery_parameters_after_detach_and_checkpoint(
    taps, tmp_path
):
    model = _stim_history(taps=taps, noise=0.0, long_interval=True)
    model.run(DT)
    model.cell.v.square().sum().backward()
    connection = next(iter(model.synapses.values()))
    assert connection.source_gate_history.numel() > 0
    frozen = _snapshot(model.state_dict_for_checkpoint())
    model.detach()
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
    assert not connection.source_gate_history.requires_grad
    assert connection.source_gate_history.grad_fn is None
    functional, tensors = dn.func.make_functional_network(model)
    name = next(iter(model.synapses))
    raw_weight = connection.weight.weights[0].param.rho
    raw_delay = connection.delay_ms.weights[0].param.rho
    keys = ("weight.weights.0.param.rho", "delay_ms.weights.0.param.rho")
    for native, key, update in zip(
        (raw_weight, raw_delay), keys, (0.4, -0.3), strict=True
    ):
        with torch.no_grad():
            native.add_(update)
            tensors.parameters["synapses"][name][key].add_(update)
    expected, _ = functional.prepare_and_rollout(
        *tensors,
        steps=8,
        draws=functional.sample_runtime_draws(tensors.state, steps=8),
    )
    reference_targets = tuple(tensors.parameters["synapses"][name][key] for key in keys)
    wanted = torch.autograd.grad(
        expected["populations"]["cell"]["integrator"]["v"].square().sum(),
        reference_targets,
    )
    with _forbid_initialization(model):
        model.longrun_checkpointed(8 * DT, chunklength=3, safe_checkpoint=True)
        latest = _snapshot(model.state_dict_for_checkpoint())
        observed = torch.autograd.grad(
            model.cell.v.square().sum(), (raw_weight, raw_delay)
        )
    _close(functional.extract(model).state, expected)
    _close(observed, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in observed)
    _assert_packet_equal(model.state_dict_for_checkpoint(), latest)
    # Lazy refresh flags are not checkpoint fields. Save a pending boundary,
    # restore its raw/runtime values, then explicitly start a receiver segment.
    previous_weight = connection.weight.w.detach().clone()
    with torch.no_grad():
        raw_weight.add_(0.2)
    model.detach()
    pending = _snapshot(model.state_dict_for_checkpoint())
    receiver = _stim_history(taps=taps, noise=0.0, long_interval=True)
    receiver.load(_disk(model.state_dict(), tmp_path / "sources.pt"))
    receiver.restore_dict_from_checkpoint(_disk(pending, tmp_path / "runtime.pt"))
    receiver.detach()
    model.step()
    receiver.step()
    _assert_packet_equal(
        model.state_dict_for_checkpoint(), receiver.state_dict_for_checkpoint()
    )
    receiving = next(iter(receiver.synapses.values()))
    _close(connection.weight.w, connection.weight.derive_values())
    _close(receiving.weight.w, receiving.weight.derive_values())
    assert not torch.equal(connection.weight.w.detach(), previous_weight)
    _close(
        torch.autograd.grad(connection.weight.w.square().sum(), raw_weight),
        torch.autograd.grad(
            receiving.weight.w.square().sum(), receiving.weight.weights[0].param.rho
        ),
    )


@pytest.mark.parametrize("backend", ("bitpacked_history", "sparse_calendar"))
def test_compact_pending_payloads_and_counts_survive_detach_without_reweighting(
    backend,
):
    model, reference = (_pending_network(backend) for _ in range(2))
    reference.restore_dict_from_checkpoint(_snapshot(model.state_dict_for_checkpoint()))
    model.step()
    reference.step()
    assert _pending_count(model) == 2
    connection = next(iter(model.synapses.values()))
    pending = connection._pending_calendar
    payload = pending.values.detach().clone().requires_grad_() * 1.0
    connection._pending_calendar = pending._replace(values=payload)
    frozen = _snapshot(model.state_dict_for_checkpoint())
    sources, rng = _sources(model), torch.get_rng_state().clone()
    assert model.detach_() is None
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
    detached_payload = connection._pending_calendar.values
    assert detached_payload is not payload
    assert not detached_payload.requires_grad and detached_payload.grad_fn is None
    assert (
        detached_payload.untyped_storage().data_ptr()
        == payload.untyped_storage().data_ptr()
    )
    # These are enqueue-time weighted payloads, independent of later raw updates.
    raw = connection.weight.weights[0].param.rho
    with torch.no_grad():
        raw.add_(0.7)
    deliveries, counts = [], []
    for _ in range(10):
        model.step()
        reference.step()
        _assert_packet_equal(
            model.state_dict_for_checkpoint(), reference.state_dict_for_checkpoint()
        )
        deliveries.append(connection.delivery_buffer.clone())
        counts.append(connection.events.clone())
    assert torch.stack(deliveries).abs().sum() > 0
    if backend == "sparse_calendar":
        assert torch.stack(counts).sum() > 0
    assert _pending_count(model) == 0
    _close(torch.get_rng_state(), rng)
    _assert_sources(model, sources)


def test_detach_keeps_connection_aliases_source_ownership_and_runtime_values():
    model = _mixed_network()
    model.run(3 * DT)
    first, second = model.synapses.values()
    assert first.delivery_buffer.shape == second.delivery_buffer.shape
    second.delivery_buffer = first.delivery_buffer
    original = first.delivery_buffer
    model._set_duration_remainder(0.3 * DT)
    frozen = _snapshot(model.state_dict_for_checkpoint())
    sources, rng = _sources(model), torch.get_rng_state().clone()
    _assert_clocks(model)
    model.detach()
    assert first.delivery_buffer is second.delivery_buffer
    assert first.delivery_buffer is not original
    assert (
        first.delivery_buffer.untyped_storage().data_ptr()
        == original.untyped_storage().data_ptr()
    )
    assert (
        not first.delivery_buffer.requires_grad
        and first.delivery_buffer.grad_fn is None
    )
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
    _assert_sources(model, sources)
    _close(torch.get_rng_state(), rng)
    _assert_clocks(model)


def test_shared_rand_noise_sde_and_poisson_streams_continue_without_detach_draws():
    model, pulses = _stochastic_network()
    reference, reference_pulses = _stochastic_network()
    reference.restore_dict_from_checkpoint(_snapshot(model.state_dict_for_checkpoint()))
    for candidate in (model, reference):
        candidate.run(2 * STOCHASTIC_DT)
    frozen = _snapshot(model.state_dict_for_checkpoint())
    borrowed = tuple(pulse.generator for pulse in pulses)
    global_rng = torch.get_rng_state().clone()
    sources = _sources(model)
    model.detach()
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
    for pulse, generator in zip(pulses, borrowed, strict=True):
        assert pulse.generator is generator
    # No-grad continuation isolates the detach operation's RNG/state footprint
    # from the separate training materialization policy.
    with torch.no_grad():
        model.longrun(3 * STOCHASTIC_DT, chunklength=2)
        reference.longrun(3 * STOCHASTIC_DT, chunklength=2)
    _assert_packet_equal(
        model.state_dict_for_checkpoint(), reference.state_dict_for_checkpoint()
    )
    for pulse, expected in zip(pulses, reference_pulses, strict=True):
        _close(pulse.generator.get_state(), expected.generator.get_state())
    _close(torch.get_rng_state(), global_rng)
    _assert_sources(model, sources)


def test_updated_dynamic_delay_capacity_failure_preserves_pending_payloads_and_retries():
    model = _stim_history(taps=2, noise=0.0, long_interval=True)
    model.run(DT)
    model.cell.v.square().sum().backward()
    model.detach()
    functional, tensors = dn.func.make_functional_network(model)
    connection = next(iter(model.synapses.values()))
    raw_delay = connection.delay_ms.weights[0].param.rho
    previous = raw_delay.detach().clone()
    old_weights = connection.weight.w.detach().clone()
    old_delays = connection.delay_ms.w.detach().clone()
    with torch.no_grad():
        connection.weight.weights[0].param.rho.add_(0.2)
        raw_delay.add_(30.0)
    frozen, rng = (
        _snapshot(model.state_dict_for_checkpoint()),
        torch.get_rng_state().clone(),
    )
    with _forbid_initialization(model):
        with pytest.raises(ValueError, match="delay capacity"):
            model.step()
        _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
        _close(connection.weight.w, old_weights)
        _close(connection.delay_ms.w, old_delays)
        _close(torch.get_rng_state(), rng)
        assert model._detached_materializations_pending
    population_targets = model.cell.mech.pas.g_param, model.cell.cm_param.rho
    with torch.no_grad():
        raw_delay.copy_(previous)
        population_targets[0].mul_(1.25)
        population_targets[1].add_(0.2)
    # The failed attempt has already refreshed Population workspaces once.
    # Repairing it must also observe raw Population edits made before retry.
    native_sources = functional.populations["cell"]._parameter_sources(model.cell)
    reference_targets = []
    for target in population_targets:
        names = [name for name, value in native_sources.items() if value is target]
        assert names
        for name in names:
            with torch.no_grad():
                tensors.parameters["populations"]["cell"][name].copy_(target)
        reference_targets.append(tensors.parameters["populations"]["cell"][names[0]])
    with torch.no_grad():
        tensors.parameters["synapses"][next(iter(model.synapses))][
            "weight.weights.0.param.rho"
        ].copy_(connection.weight.weights[0].param.rho)
    expected, _ = functional.prepare_and_rollout(
        *tensors,
        steps=7,
        draws=functional.sample_runtime_draws(tensors.state, steps=7),
    )
    wanted = torch.autograd.grad(
        expected["populations"]["cell"]["integrator"]["v"].square().sum(),
        tuple(reference_targets),
    )
    with _forbid_initialization(model):
        model.step()
        model.longrun(6 * DT, chunklength=2)
    _close(model.cell.v, expected["populations"]["cell"]["integrator"]["v"])
    _close(
        model.cell.mech.pas.g, population_targets[0].expand_as(model.cell.mech.pas.g)
    )
    _close(model.cell.cm, model.cell.cm_param().expand_as(model.cell.cm))
    observed = torch.autograd.grad(model.cell.v.square().sum(), population_targets)
    _close(observed, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in observed)
    _close(model.t, frozen["t"] + 7 * DT)
    _close(connection.weight.w, connection.weight.derive_values())
    assert not torch.equal(connection.weight.w.detach(), old_weights)
    assert not model._detached_materializations_pending


def test_missing_trainable_sampling_draws_fail_before_publication_and_can_be_restored():
    model, connection = _continuous(stochastic=True)
    model.run(2 * DT)
    model.post.v.square().sum().backward()
    model.detach()
    saved_draws = _snapshot(connection.weight.sampling_draws)
    connection.weight._sampling_draws = None
    source = connection.weight.weights[0].param
    with torch.no_grad():
        source.mean.add_(0.0003)
    frozen = _snapshot(model.state_dict_for_checkpoint())
    weights, rng = connection.weight.w.detach().clone(), torch.get_rng_state().clone()
    with _forbid_initialization(model):
        with pytest.raises(RuntimeError, match="without retained explicit draws"):
            model.step()
        _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
        _close(connection.weight.w, weights)
        _close(torch.get_rng_state(), rng)
        assert model._detached_materializations_pending
        before = model.pre.v.detach().clone(), model.post.v.detach().clone()
        with torch.no_grad():
            model.pre.mech.pas.g_param.mul_(1.2)
            model.pre.cm_param.rho.add_(0.15)
        connection.weight.set_extra_state(saved_draws)
        expected = _continuous_oracle(model, connection, before, steps=1)
        targets = (
            source.mean,
            source.std,
            model.pre.mech.pas.g_param,
            model.pre.cm_param.rho,
        )
        wanted = torch.autograd.grad(
            sum(value.square().sum() for value in expected), targets
        )
        model.step()
    _close(connection.weight.w, connection.weight.derive_values(saved_draws))
    _close((model.pre.v, model.post.v), expected)
    _close(
        model.pre.mech.pas.g, model.pre.mech.pas.g_param.expand_as(model.pre.mech.pas.g)
    )
    _close(model.pre.cm, model.pre.cm_param().expand_as(model.pre.cm))
    gradients = torch.autograd.grad(
        model.pre.v.square().sum() + model.post.v.square().sum(), targets
    )
    _close(gradients, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in gradients)


class _OpaqueFrozen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.allow_sample = True

    def sample(self, n):
        if not self.allow_sample:
            raise AssertionError("A frozen opaque connection must keep its old sample")
        return torch.full((n,), 0.002, dtype=torch.float64)


def test_frozen_opaque_block_is_retained_while_deterministic_neighbor_reconnects():
    model = _stim_history(taps=None, noise=0.0, long_interval=True)
    connection = next(iter(model.synapses.values()))
    opaque = _OpaqueFrozen()
    raw = torch.nn.Parameter(torch.tensor(0.003, dtype=torch.float64))
    connection.weight = make_weight([opaque, raw], [1, 1], dtype=torch.float64).init()
    assert connection.weight.sampling_draws is None
    model.run(DT)
    model.cell.v.square().sum().backward()
    model.detach()
    opaque.allow_sample = False
    with torch.no_grad():
        raw.add_(0.001)
    with _forbid_initialization(model):
        model.longrun(4 * DT, chunklength=2)
    _close(connection.weight.w, torch.stack((raw.new_tensor(0.002), raw)))
    assert connection.weight.weights[1].param is raw
    derivative = torch.autograd.grad(model.cell.v.square().sum(), raw)[0]
    assert derivative.isfinite().all() and derivative.abs().sum() > 0


def test_indexed_schedule_sources_keep_identity_and_fresh_gradients_across_segments():
    model = _stim_network()
    model.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=False,
        diff_spiking=True,
        diff_scheduled_times=True,
        sched_width=0.8,
        train_delay_backend="dense",
    )
    connection = next(iter(model.synapses.values()))
    source = torch.nn.Parameter(torch.tensor([0.7], dtype=torch.float64))
    for owner in (model.netstim, connection):
        owner.bind_weight_source(source)
    times = [0.25 * DT, 3.25 * DT]
    model.netstim.schedule_ref([0, 0], times=times, weight_idx=[0, 0])
    connection.schedule_ref(con_indices=[0, 0], times_ms=times, weight_idx=[0, 0])
    authored = tuple(
        (owner, name, getattr(owner, name))
        for owner in (model.netstim, connection)
        for name in (
            "_sched_w_source",
            "sched_time_ms",
            "sched_weight",
            "sched_time_idx",
            "sched_weight_idx",
        )
    )
    model.run(2 * DT)
    model.cell.v.square().sum().backward()
    assert (
        source.grad is not None
        and source.grad.isfinite().all()
        and source.grad.abs().sum() > 0
    )
    model.detach()
    assert all(getattr(owner, name) is value for owner, name, value in authored)
    source.grad = None
    with torch.no_grad():
        source.add_(0.2)
    functional, tensors = dn.func.make_functional_network(model)
    name = next(iter(model.synapses))
    explicit_source = tensors.parameters["synapses"][name]["schedule"].amplitude_source
    assert explicit_source is tensors.parameters["netstim"]["schedule"].amplitude_source
    assert explicit_source is not source and explicit_source.requires_grad
    expected, _ = functional.prepare_and_rollout(
        *tensors,
        steps=4,
        draws=functional.sample_runtime_draws(tensors.state, steps=4),
    )
    wanted = torch.autograd.grad(
        expected["populations"]["cell"]["integrator"]["v"].square().sum(),
        explicit_source,
    )[0]
    model.longrun(4 * DT, chunklength=3)
    _close(functional.extract(model).state, expected)
    actual = torch.autograd.grad(model.cell.v.square().sum(), source)[0]
    _close(actual, wanted)
    assert actual.isfinite().all() and actual.abs().sum() > 0
    assert all(getattr(owner, name) is value for owner, name, value in authored)


def test_native_compiled_population_solves_refresh_connected_training_segments():
    with dn.ctx(
        JIT=0,
        JIT_NETWORK_SOLVES=1,
        JIT_NETWORK_OPS=0,
        BACKEND="aot_eager",
        FULLGRAPH=1,
    ):
        model, connection = _continuous()
        model.run(2 * DT)
        for population in model.populations.values():
            assert population.integrator.jit
            assert population.integrator.compile_scope == "network_population"
            assert population.integrator._compiled_kernels
        (model.pre.v.square().sum() + model.post.v.square().sum()).backward()
        before = model.pre.v.detach().clone(), model.post.v.detach().clone()
        raw = connection.weight.weights[0].param.rho
        model.detach()
        with torch.no_grad():
            raw.add_(0.2)
            model.pre.mech.pas.g_param.mul_(1.1)
            model.pre.cm_param.rho.add_(0.1)
        targets = raw, model.pre.mech.pas.g_param, model.pre.cm_param.rho
        expected = _continuous_oracle(model, connection, before, steps=3)
        wanted = torch.autograd.grad(
            sum(value.square().sum() for value in expected), targets
        )
        model.longrun(3 * DT, chunklength=2)
        _close((model.pre.v, model.post.v), expected)
        observed = torch.autograd.grad(
            model.pre.v.square().sum() + model.post.v.square().sum(), targets
        )
        _close(observed, wanted)
        assert all(
            value.isfinite().all() and value.abs().sum() > 0 for value in observed
        )
        for population in model.populations.values():
            assert population.integrator.jit and population.integrator._compiled_kernels
            assert population.integrator.compile_scope == "network_population"


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_connected_detach_suffix_matches_float32_values_and_raw_gradients():
    model, connection = _continuous(dtype=torch.float32, device="mps")
    model.run(2 * DT)
    model.post.v.square().sum().backward()
    before = model.pre.v.detach().clone(), model.post.v.detach().clone()
    raw = connection.weight.weights[0].param.rho
    with torch.no_grad():
        raw.add_(0.2)
        model.pre.mech.pas.g_param.mul_(1.1)
    sources, rng = _sources(model), torch.mps.get_rng_state().clone()
    model.detach()
    expected = _continuous_oracle(model, connection, before, steps=3)
    targets = raw, model.pre.mech.pas.g_param, connection.transform.scale
    wanted = torch.autograd.grad(
        sum(value.square().sum() for value in expected), targets
    )
    model.longrun(3 * DT, chunklength=2)
    torch.testing.assert_close(model.pre.v, expected[0], rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(model.post.v, expected[1], rtol=3e-5, atol=3e-5)
    observed = torch.autograd.grad(
        model.pre.v.square().sum() + model.post.v.square().sum(), targets
    )
    for actual, reference in zip(observed, wanted, strict=True):
        torch.testing.assert_close(actual, reference, rtol=3e-4, atol=3e-4)
        assert (
            actual.device.type == "mps"
            and actual.isfinite().all()
            and actual.abs().sum() > 0
        )
    assert model.pre.mech.pas.g_param is model.post.mech.pas.g_param
    _close(torch.mps.get_rng_state(), rng)
    _assert_sources(model, sources)
