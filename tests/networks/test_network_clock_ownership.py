"""Network copies own their dynamic clocks across runtime lifecycle operations."""

from __future__ import annotations

import copy
import gc
import weakref

import pytest
import torch
from torch.nn.utils import parametrize

import dendra as dn
from dendra.models.mechanisms import MaterialProcess, Mechanism
from dendra.models.mod import expsyn, graded_syn

pytestmark = pytest.mark.cpu
DT = 0.125


class _ClockCurrent(Mechanism):
    Mechanism.RANGE(g=0.001)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * (-65.0 + 0.2 * self.t)


class _ClockTransfer(MaterialProcess):
    MaterialProcess.GLOBAL(rate=0.05)
    material_fields = (("pool", "amount"), ("pool", "reserve"))

    def prepare_materials(self, fields, dt):
        return (
            dt
            * self._resolve_quantity("rate", fields[("pool", "amount")])
            * (1.0 + 0.01 * self.t)
        )

    def material_updates(self, fields, prepared):
        amount, reserve = fields[("pool", "amount")], fields[("pool", "reserve")]
        transfer = prepared * (amount - reserve)
        return {
            ("pool", "amount"): amount - transfer,
            ("pool", "reserve"): reserve + transfer,
        }


def _release(value):
    return torch.sigmoid(value + 65.0)


@pytest.fixture(autouse=True)
def _environment():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        yield


def _network(*, training=False, start=0.0, initialize=True):
    cell = dn.SingleCompartment(
        N=1, C=1, v_init=-65.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    cell.material("pool", fields={"amount": 1.7, "reserve": 0.4}, conserved=False)
    cell.insert(_ClockCurrent.rename("clock_current"))
    cell.insert(_ClockTransfer.rename("clock_transfer"))
    cell.insert(expsyn.rename("events"), tau=1.2, e=0.0)
    cell.insert(graded_syn.rename("analog"), e=-10.0)
    stim = dn.NetStim(
        N=1, interval=0.25, start=-0.25, noise=0.35, max_spikes=30, seed=812
    )
    # Fixed-seed renewal uses _rng; stabilize the independent reseeding cursor
    # as well so whole owned checkpoints can be compared bit for bit.
    stim._seeder.manual_seed(27)
    network = dn.Network(
        {"cell": cell},
        netstim=stim,
        netcon_delay_backend="dense",
        netcon_train_backend="dense",
    )
    network.connect_one_to_one(
        stim[:], cell[:], cell.mech.events, threshold=None, weight=0.003, delay=2 * DT
    )
    network.connect_continuous_one_to_one(
        cell[:],
        cell[:],
        cell.mech.analog,
        weight=0.002,
        delay=2 * DT,
        transform=_release,
        input="g_pre",
        allow_autapses=True,
    )
    network.train(training)
    if not initialize:
        return network
    with torch.no_grad():
        network.initialize(DT, t=start)
    if training:
        network.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=True,
            diff_scheduled_times=False,
            train_delay_backend="dense",
        )
    return network


def _clone(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {name: _clone(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    if isinstance(value, list):
        return [_clone(item) for item in value]
    return value


def _equal(actual, expected, path=()):
    if torch.is_tensor(actual):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0,
            atol=0,
            msg=lambda message: f"{'.'.join(map(str, path))}: {message}",
        )
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for name, value in actual.items():
            _equal(value, expected[name], (*path, name))
    elif isinstance(actual, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
            _equal(value, reference, (*path, index))
    else:
        assert actual == expected


def _assert_clocks(network):
    for connection in (
        *network.synapses.values(),
        *network.continuous_synapses.values(),
    ):
        assert connection.t is network.t
    for population in network.populations.values():
        torch.testing.assert_close(
            population.t, network.t.to(population.t), rtol=0, atol=0
        )
        for mechanism in population.mech.mechanisms.values():
            assert mechanism.t is population.t
        for process in population.mech.material_processes.values():
            assert process.t is population.t


def _schedule(network, start):
    event = next(iter(network.synapses.values()))
    event.schedule(
        con_indices=[0, 0], times_ms=[start + 3 * DT, start + 5 * DT], weight=0.007
    )


@pytest.mark.parametrize("training", (False, True))
def test_deepcopy_owns_event_continuous_and_population_clocks(training):
    source, oracle = _network(training=training), _network(training=training)
    _schedule(source, 0.0)
    _schedule(oracle, 0.0)
    for _ in range(2):
        source.step()
        oracle.step()
    before = _clone(source.state_dict_for_checkpoint())
    copied = copy.deepcopy(source)

    _assert_clocks(copied)
    assert copied.t.data_ptr() != source.t.data_ptr()
    if training:
        assert next(iter(source.synapses.values())).delivery_buffer.grad_fn is not None
        assert next(iter(copied.synapses.values())).delivery_buffer.grad_fn is None
    source_parameters = dict(source.named_parameters())
    copied_parameters = dict(copied.named_parameters())
    assert source_parameters.keys() == copied_parameters.keys()
    for name, value in copied_parameters.items():
        assert value.is_leaf
        assert value.data_ptr() != source_parameters[name].data_ptr()
    _equal(copied.state_dict_for_checkpoint(), before)
    for _ in range(5):
        copied.step()
        oracle.step()
        _assert_clocks(copied)
        _equal(copied.state_dict_for_checkpoint(), oracle.state_dict_for_checkpoint())
        _equal(source.state_dict_for_checkpoint(), before)

    # Advancing the source after the copy has run cannot move copied clocks.
    copied_before = _clone(copied.state_dict_for_checkpoint())
    source.step()
    _assert_clocks(source)
    _equal(copied.state_dict_for_checkpoint(), copied_before)
    assert torch.all(copied.cell.mech.events.g > 0)
    assert torch.any(
        next(iter(copied.continuous_synapses.values())).delivery_buffer > 0
    )


@pytest.mark.parametrize("start", (-0.5, 3.25))
def test_unbuilt_network_copy_initializes_connections_against_its_own_clock(start):
    source = _network(initialize=False)
    copied = copy.deepcopy(source)
    copied.initialize(DT, t=start)
    oracle = _network(start=start)
    _schedule(copied, start)
    _schedule(oracle, start)
    _assert_clocks(copied)
    for _ in range(5):
        copied.step()
        oracle.step()
        _equal(copied.state_dict_for_checkpoint(), oracle.state_dict_for_checkpoint())
    assert not source.built
    assert source.dt is None
    assert source.t.item() == 0.0
    assert not source.synapses
    assert not source.continuous_synapses


@pytest.mark.parametrize("start", (-0.5, 3.25))
@pytest.mark.parametrize("force_rebuild", (False, True))
def test_copy_reinitialization_uses_own_absolute_clock_and_preserves_source(
    start, force_rebuild
):
    source = _network()
    for _ in range(3):
        source.step()
    before = _clone(source.state_dict_for_checkpoint())
    copied = copy.deepcopy(source)
    previous_connections = tuple(copied.synapses.values()) + tuple(
        copied.continuous_synapses.values()
    )
    copied.initialize(DT, t=start, force_rebuild=force_rebuild)
    oracle = _network(start=start)
    _schedule(copied, start)
    _schedule(oracle, start)
    connections = tuple(copied.synapses.values()) + tuple(
        copied.continuous_synapses.values()
    )
    assert all(
        (old is new) == (not force_rebuild)
        for old, new in zip(previous_connections, connections, strict=True)
    )
    _assert_clocks(copied)
    for _ in range(7):
        copied.step()
        oracle.step()
        _equal(copied.state_dict_for_checkpoint(), oracle.state_dict_for_checkpoint())
        _equal(source.state_dict_for_checkpoint(), before)
    _assert_clocks(copied)


def test_copied_network_checkpoint_restores_clock_ownership_and_exact_continuation():
    source, oracle = _network(start=3.25), _network(start=3.25)
    _schedule(source, 3.25)
    _schedule(oracle, 3.25)
    for _ in range(2):
        source.step()
        oracle.step()
    copied = copy.deepcopy(source)
    before = _clone(source.state_dict_for_checkpoint())
    state = _clone(copied.state_dict_for_checkpoint())
    for _ in range(3):
        copied.step()
    copied.restore_dict_from_checkpoint(state)
    _assert_clocks(copied)
    for _ in range(5):
        copied.step()
        oracle.step()
        _equal(copied.state_dict_for_checkpoint(), oracle.state_dict_for_checkpoint())
    _equal(source.state_dict_for_checkpoint(), before)


def test_copied_network_public_load_restores_owned_clock_and_exact_anchor():
    source = _network(start=3.25)
    source.step()
    source.step()
    copied = copy.deepcopy(source)
    before = _clone(source.state_dict_for_checkpoint())
    state = _clone(copied.state_dict())
    origin, step, time = (
        copied._clock_origin.clone(),
        copied._clock_step.clone(),
        copied.t.clone(),
    )
    copied.step()
    copied.load(state)
    _assert_clocks(copied)
    assert torch.equal(copied._clock_origin, origin)
    assert torch.equal(copied._clock_step, step)
    assert torch.equal(copied.t, time)
    copied.step()
    assert torch.equal(copied.t, time + DT)
    _assert_clocks(copied)
    _equal(source.state_dict_for_checkpoint(), before)


def test_copied_network_late_checkpoint_failure_rolls_back_owned_clock_and_runtime(
    monkeypatch,
):
    source = _network(start=-0.5)
    for _ in range(2):
        source.step()
    copied = copy.deepcopy(source)
    copied.step()
    before = _clone(copied.state_dict_for_checkpoint())
    source_before = _clone(source.state_dict_for_checkpoint())
    candidate = _clone(source_before)
    candidate["populations"]["cell"]["integrator"]["v"].add_(7.0)
    sync = copied._sync_runtime_clock
    calls = 0

    def fail_once():
        nonlocal calls
        sync()
        if calls == 0:
            calls += 1
            raise RuntimeError("injected copied-clock restore failure")

    monkeypatch.setattr(copied, "_sync_runtime_clock", fail_once)
    with pytest.raises(RuntimeError, match="injected copied-clock restore failure"):
        copied.restore_dict_from_checkpoint(candidate)
    assert calls == 1
    _assert_clocks(copied)
    _equal(copied.state_dict_for_checkpoint(), before)
    _equal(source.state_dict_for_checkpoint(), source_before)
    copied.step()
    _assert_clocks(copied)


def test_repeated_deepcopy_does_not_change_source_or_accumulate_clock_proxy_classes():
    source = _network()
    owners = (
        *source.synapses.values(),
        *source.continuous_synapses.values(),
        *source.cell.mech.mechanisms.values(),
        *source.cell.mech.material_processes.values(),
    )
    types = tuple(type(owner) for owner in owners)
    depths = tuple(len(kind.__mro__) for kind in types)
    descendant = source
    for _ in range(5):
        descendant = copy.deepcopy(descendant)
        cloned_owners = (
            *descendant.synapses.values(),
            *descendant.continuous_synapses.values(),
            *descendant.cell.mech.mechanisms.values(),
            *descendant.cell.mech.material_processes.values(),
        )
        assert tuple(len(type(owner).__mro__) for owner in cloned_owners) == depths
        assert tuple(type(owner) for owner in owners) == types
        _assert_clocks(descendant)
        _assert_clocks(source)


def test_copied_network_clock_references_do_not_keep_source_owners_alive():
    source = _network()
    source.step()
    source_network, source_population = weakref.ref(source), weakref.ref(source.cell)
    copied = copy.deepcopy(source)
    del source
    gc.collect()

    assert source_network() is None
    assert source_population() is None
    _assert_clocks(copied)
    copied.step()
    _assert_clocks(copied)


class _NetworkAffineGain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.5, dtype=torch.float64))
        self.register_buffer("offset", torch.tensor(0.25, dtype=torch.float64))

    def forward(self, value):
        return value * self.scale + self.offset


class _NetworkSplitGain(torch.nn.Module):
    def forward(self, value, multiplier):
        return value * multiplier

    def right_inverse(self, value):
        return value.clone(), torch.ones_like(value)


@pytest.mark.parametrize("multiple_originals", (False, True))
def test_parametrized_network_copy_preserves_raw_aliases_and_local_gradients(
    multiple_originals,
):
    source = _network(training=True)
    source.step()
    source.register_parameter(
        "gain", torch.nn.Parameter(torch.tensor([2.0, 3.0], dtype=torch.float64))
    )
    if multiple_originals:
        parametrize.register_parametrization(source, "gain", _NetworkSplitGain())
    parametrize.register_parametrization(source, "gain", _NetworkAffineGain())
    source_gain = source.parametrizations.gain
    raw = source_gain.original0 if multiple_originals else source_gain.original
    source.register_parameter("gain_alias", raw)
    before = _clone(source.state_dict_for_checkpoint())

    copied = copy.deepcopy(source)

    _assert_clocks(copied)
    copied_gain = copied.parametrizations.gain
    copied_raw = copied_gain.original0 if multiple_originals else copied_gain.original
    assert copied.gain_alias is copied_raw
    assert source.gain_alias is raw
    assert copied_raw is not raw
    torch.testing.assert_close(copied.gain, source.gain, rtol=0, atol=0)
    source_parameters = tuple(source_gain.parameters())
    copied_parameters = tuple(copied_gain.parameters())
    for parameter, original in zip(copied_parameters, source_parameters, strict=True):
        assert parameter.is_leaf
        assert parameter.data_ptr() != original.data_ptr()
    actual = torch.autograd.grad(copied.gain.square().sum(), copied_parameters)
    expected = torch.autograd.grad(source.gain.square().sum(), source_parameters)
    for gradient, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(gradient, reference, rtol=0, atol=0)
        assert gradient.abs().sum() > 0
    copied.step()
    _assert_clocks(copied)
    _equal(source.state_dict_for_checkpoint(), before)


class _NetworkCopyStateHook(dn.Network):
    def __getstate__(self):
        state = super().__getstate__()
        state["authored_copy_state"] = "retained"
        return state


def test_parametrized_network_copy_preserves_authored_state_hook():
    cell = dn.SingleCompartment(N=1, C=1, v_init=-65.0)
    cell.insert(_ClockCurrent.rename("clock_current"))
    source = _NetworkCopyStateHook({"cell": cell})
    source.initialize(DT)
    source.register_parameter(
        "gain", torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    )
    parametrize.register_parametrization(source, "gain", _NetworkAffineGain())

    copied = copy.deepcopy(source)

    assert isinstance(copied, _NetworkCopyStateHook)
    assert copied.authored_copy_state == "retained"
    assert "authored_copy_state" not in source.__dict__
    _assert_clocks(copied)
    torch.testing.assert_close(copied.gain, source.gain, rtol=0, atol=0)
    source_time = source.t.clone()
    copied.step()
    _assert_clocks(copied)
    assert torch.equal(source.t, source_time)
