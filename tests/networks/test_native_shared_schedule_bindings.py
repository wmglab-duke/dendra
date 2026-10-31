"""Native declarations bind independent live event times and weights."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_continuation_checkpoints import _disk
from tests.func.test_network import DT as NETWORK_DT
from tests.func.test_network import _stim_network
from tests.networks.test_pure_tensor_schedules import _native_netcon

DT = 0.1
pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64, IMEM=False),
    ):
        yield


def _owner(kind, *, n=3):
    if kind == "netstim":
        owner = dn.NetStim(
            N=n, start=100, interval=100, noise=0, seed=71, dtype=torch.float64
        )
        owner.set_dt(DT)
        owner.set_diff_config(diff_scheduled_times=True, sched_width=1.0)
        owner.initialize()
        return owner
    owner = _native_netcon(generators=n)
    owner.set_diff_config(
        diff_weights=True,
        diff_delays=False,
        diff_spiking=False,
        diff_scheduled_times=True,
        sched_width=1.0,
        train_delay_backend="dense",
    )
    owner.train()
    owner.initialize(reinit_weights=False, reinit_delays=False)
    return owner


def _schedule(owner, targets, **kwargs):
    if isinstance(owner, dn.NetStim):
        return owner.schedule(targets, **kwargs)
    return owner.schedule(con_indices=targets, **kwargs)


def _gate(owner, *, time=0.2, triangular=True):
    if isinstance(owner, dn.NetStim):
        owner.set_diff_config(diff_scheduled_times=triangular, sched_width=1.0)
        return owner._scheduled_tensor_gate(
            torch.tensor(time, dtype=owner.dtype()), dt=DT
        )[0]
    return owner._scheduled_gate_this_step(
        torch.tensor(round(time / DT)), use_tri_kernel=triangular
    )[0]


def _bind(owner, times, weights):
    owner.bind_time_source(times)
    owner.bind_weight_source(weights)


@pytest.mark.parametrize("kind", ("netstim", "netcon"))
def test_dual_bindings_match_independent_value_and_gradient_ledger(kind):
    owner = _owner(kind)
    times = torch.nn.Parameter(torch.tensor([0.15, 0.25], dtype=torch.float64))
    weights = torch.nn.Parameter(torch.tensor([2.0, -0.5, 0.0], dtype=torch.float64))
    _bind(owner, times, weights)
    targets, tidx, widx = [0, 0, 1, 2], [0, 1, 0, 1], [0, 1, 2, 0]
    _schedule(owner, targets, time_idx=tidx, weight_idx=widx)
    before, rng = (
        _snapshot((owner.sched_time_ms, owner.sched_weight)),
        torch.get_rng_state().clone(),
    )
    gate = _gate(owner)
    width = 1.0 if kind == "netstim" else 1.000001
    selected_times = times[torch.tensor(tidx)]
    selected_weights = weights[torch.tensor(widx)]
    kernel = (1 - ((selected_times - 0.2) / DT).abs() / width).clamp(0, 1)
    expected = torch.zeros(3, dtype=torch.float64).scatter_add(
        0, torch.tensor(targets), selected_weights * kernel
    )
    _close(gate.reshape(-1), expected)
    coefficients = torch.tensor([1.0, 0.7, -0.3], dtype=torch.float64)
    actual = torch.autograd.grad(
        (gate.reshape(-1) * coefficients).sum(), (times, weights)
    )
    wanted = torch.autograd.grad((expected * coefficients).sum(), (times, weights))
    _close(actual, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in actual)
    assert owner._sched_t_source is times and owner._sched_w_source is weights
    _assert_packet_equal((owner.sched_time_ms, owner.sched_weight), before)
    _close(torch.get_rng_state(), rng)


@pytest.mark.parametrize("kind", ("netstim", "netcon"))
def test_reference_filtering_is_frozen_at_declaration_and_sources_stay_live(kind):
    owner = _owner(kind, n=2)
    times = torch.tensor([0.1, 0.3], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([2.0, 4.0], dtype=torch.float64, requires_grad=True)
    _bind(owner, times, weights)
    if kind == "netstim":
        owner.t_last.fill_(0.2)
    else:
        owner.global_step.fill_(2)
    _schedule(owner, [0, 1], time_idx=[0, 1], weight_idx=[0, 1])
    assert owner.sched_time_idx.tolist() == owner.sched_weight_idx.tolist() == [1]
    with torch.no_grad():
        times[1] = 0.1
        weights[1] = -3.0
    # Moving an accepted time into the past does not repeat the initial filter.
    _close(
        _gate(owner, time=0.1, triangular=False).reshape(-1),
        weights.new_tensor([0, -3]),
    )
    assert owner.sched_time_idx.tolist() == [1]


def test_netstim_dual_bindings_expand_scalar_indices_over_batch_coordinates():
    owner = _owner("netstim", n=2).batch(2)
    times = torch.tensor([0.2], dtype=torch.float64)
    weights = torch.tensor([-2.0], dtype=torch.float64)
    _bind(owner, times, weights)
    _schedule(owner, [0, (1, 1)], time_idx=0, weight_idx=0)
    _close(_gate(owner, triangular=False), weights.new_tensor([[-2, 0], [-2, -2]]))
    assert owner.sched_time_idx.tolist() == owner.sched_weight_idx.tolist() == [0, 0, 0]


@pytest.mark.parametrize("kind", ("netstim", "netcon"))
def test_dual_time_gather_keeps_mixed_dtype_gradient_accumulation_order(kind):
    owner = _owner(kind)
    times = torch.tensor([0.15], dtype=torch.float32, requires_grad=True)
    weights = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    _bind(owner, times, weights)
    _schedule(owner, [0, 1, 2], time_idx=[0, 0, 0], weight_idx=[0, 0, 0])
    gate = _gate(owner).reshape(-1)
    selected = times.index_select(0, torch.zeros(3, dtype=torch.long)).to(torch.float64)
    width = 1.0 if kind == "netstim" else 1.000001
    expected = (1 - ((selected - 0.2) / DT).abs() / width).clamp(0, 1)
    coefficients = torch.tensor([1e7, 0.1, -1e7], dtype=torch.float64)
    actual = torch.autograd.grad((gate * coefficients).sum(), times)[0]
    wanted = torch.autograd.grad((expected * coefficients).sum(), times)[0]
    _close(actual, wanted)
    assert actual.item() == 0


@pytest.mark.parametrize("kind", ("netstim", "netcon"))
def test_reference_defaults_conflicts_and_invalid_indices_leave_configuration_unchanged(
    kind,
):
    owner = _owner(kind, n=1)
    _bind(
        owner,
        torch.tensor([0.2], dtype=torch.float64),
        torch.tensor([-2.0], dtype=torch.float64),
    )
    for weight in (1.0, None):
        _schedule(owner, [0], time_idx=[0], weight_idx=[0], weight=weight)
    _close(
        _gate(owner, triangular=False).reshape(-1),
        torch.tensor([-4.0], dtype=torch.float64),
    )
    _schedule(owner, [0], time_idx=[0], weight=0.0)
    _schedule(owner, [0], times_ms=[0.2], weight_idx=[0])
    snapshot = _snapshot(
        (
            owner.sched_time_ms,
            owner.sched_weight,
            owner.sched_time_idx,
            owner.sched_weight_idx,
        )
    )
    invalid = (
        (ValueError, {"times_ms": [0.2], "time_idx": [0], "weight_idx": [0]}),
        (ValueError, {"time_idx": [0], "weight_idx": [0], "weight": 2.0}),
        (ValueError, {"time_idx": [0], "weight_idx": [0], "weight": torch.tensor(1.0)}),
        (IndexError, {"time_idx": [-1], "weight_idx": [0]}),
        (IndexError, {"time_idx": [0], "weight_idx": [1]}),
    )
    for error, kwargs in invalid:
        with pytest.raises(error):
            _schedule(owner, [0], **kwargs)
        _assert_packet_equal(
            (
                owner.sched_time_ms,
                owner.sched_weight,
                owner.sched_time_idx,
                owner.sched_weight_idx,
            ),
            snapshot,
        )


def test_netstim_indexed_storage_and_hard_evaluation_are_separate_contracts():
    owner = _owner("netstim", n=1)
    _bind(
        owner,
        torch.tensor([0.2], dtype=torch.float64),
        torch.tensor([2.0], dtype=torch.float64),
    )
    with pytest.raises(ValueError):
        _schedule(owner, [0], time_idx=[0], weight_idx=[0], differentiable=False)
    _schedule(owner, [0], time_idx=[0], weight_idx=[0])
    _close(_gate(owner, triangular=False), torch.tensor([2.0], dtype=torch.float64))


def test_netstim_compatibility_helpers_keep_the_existing_negative_sentinel():
    owner = _owner("netstim", n=1)
    _bind(
        owner,
        torch.tensor([0.2], dtype=torch.float64),
        torch.tensor([2.0], dtype=torch.float64),
    )
    owner.schedule_ref(0, times=0.2, weight_idx=-1, allow_past=True)
    owner.schedule_time_ref(0, time_idx=-1, weight=3.0, allow_past=True)
    assert owner.sched_time_idx.tolist() == owner.sched_weight_idx.tolist() == [-1, -1]


@pytest.mark.parametrize("kind", ("netstim", "netcon"))
def test_compatibility_reference_helpers_keep_binding_error_precedence(kind):
    owner = _owner(kind, n=1)
    with pytest.raises(RuntimeError, match="bind_weight_source"):
        if kind == "netstim":
            owner.schedule_ref(0)
        else:
            owner.schedule_ref()
    with pytest.raises(RuntimeError, match="bind_time_source"):
        if kind == "netstim":
            owner.schedule_time_ref(0, 0)
        else:
            owner.schedule_time_ref()


def _scheduled_network():
    model = _stim_network()
    model.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=False,
        diff_spiking=True,
        diff_scheduled_times=True,
        sched_width=0.8,
        train_delay_backend="dense",
    )
    times = torch.nn.Parameter(
        torch.tensor([0.25 * NETWORK_DT, 3.25 * NETWORK_DT], dtype=torch.float64)
    )
    weights = torch.nn.Parameter(torch.tensor([0.7], dtype=torch.float64))
    for owner in (model.netstim, *model.synapses.values()):
        _bind(owner, times, weights)
        _schedule(owner, [0, 0], time_idx=[0, 1], weight_idx=[0, 0])
    return model, times, weights


def test_dual_sources_train_across_native_detach_and_portable_checkpoint(tmp_path):
    model, times, weights = _scheduled_network()
    connection = next(iter(model.synapses.values()))
    model.run(2 * NETWORK_DT)
    model.cell.v.square().sum().backward()
    assert times.grad.abs().sum() > 0 and weights.grad.abs().sum() > 0
    with torch.no_grad():
        times[1].add_(0.04 * NETWORK_DT)
        weights.add_(0.2)
    model.detach()
    saved_sources = _disk(model.state_dict(), tmp_path / "sources.pt")
    saved_runtime = _disk(
        _snapshot(model.state_dict_for_checkpoint()), tmp_path / "runtime.pt"
    )
    receiver, receiver_times, receiver_weights = _scheduled_network()
    receiver.load(saved_sources)
    receiver.restore_dict_from_checkpoint(saved_runtime)
    receiver.detach()
    functional, tensors = dn.func.make_functional_network(model)
    name = next(iter(model.synapses))
    explicit = tensors.parameters["synapses"][name]["schedule"]
    assert explicit.time_source is tensors.parameters["netstim"]["schedule"].time_source
    assert (
        explicit.amplitude_source
        is tensors.parameters["netstim"]["schedule"].amplitude_source
    )
    expected, _ = functional.prepare_and_rollout(
        *tensors, steps=4, draws=functional.sample_runtime_draws(tensors.state, steps=4)
    )
    wanted = torch.autograd.grad(
        expected["populations"]["cell"]["integrator"]["v"].square().sum(),
        (explicit.time_source, explicit.amplitude_source),
    )
    model.longrun_checkpointed(4 * NETWORK_DT, chunklength=3, safe_checkpoint=True)
    receiver.longrun(4 * NETWORK_DT, chunklength=3)
    _close(functional.extract(model).state, expected)
    _assert_packet_equal(
        model.state_dict_for_checkpoint(), receiver.state_dict_for_checkpoint()
    )
    actual = torch.autograd.grad(model.cell.v.square().sum(), (times, weights))
    restored = torch.autograd.grad(
        receiver.cell.v.square().sum(), (receiver_times, receiver_weights)
    )
    _close(actual, wanted)
    _close(restored, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in actual)
    assert model.netstim._sched_t_source is connection._sched_t_source is times
    assert model.netstim._sched_w_source is connection._sched_w_source is weights
