"""Native continuation reconnects effective parameters after state detachment."""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
from torch.nn.utils.parametrize import register_parametrization

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism
from dendra.models.mod import pas
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot

DT = 0.01
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


class _Scale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.2, dtype=torch.float64))

    def forward(self, value):
        return self.scale * value


def _cell(*, parametrized=False, g=0.001, reversal=-55.0, voltage=-63.0):
    cell = dn.SingleCompartment(
        N=1, C=2, cm=1.2, v_init=voltage, integrator=dn.bwd_euler_sc(imem=False)
    )
    cell.insert(pas, g=g, e=reversal)
    cell.train()
    cell.initialize()
    if parametrized:
        register_parametrization(cell.mech.pas, "g_param", _Scale())
        cell.initialize()
    return cell


def _expected(voltage, g, cm, reversal, steps):
    for _ in range(steps):
        voltage = (cm * voltage + 1000 * DT * g * reversal) / (cm + 1000 * DT * g)
    return voltage


def _parameter_identities(module):
    return {
        name: value for name, value in module.named_parameters(remove_duplicate=False)
    }


def _assert_sources(module, sources):
    current = _parameter_identities(module)
    assert current.keys() == sources.keys()
    assert all(current[name] is value for name, value in sources.items())


@pytest.mark.parametrize(
    "parametrized", (False, True), ids=("ordinary", "torch_parametrized")
)
def test_native_detach_refreshes_optimizer_updates_and_positive_solver_physics(
    parametrized, monkeypatch
):
    cell = _cell(parametrized=parametrized)
    owner = cell.mech.pas
    if parametrized:
        original = owner.parametrizations.g_param.original
        scale = owner.parametrizations.g_param[0].scale
        targets = original, scale, cell.cm_param.rho
    else:
        targets = owner.g_param, cell.cm_param.rho
    cell.run(tstop=2 * DT, dt=DT)
    cell.v.square().sum().backward()
    assert all(
        value.grad is not None and value.grad.abs().sum() > 0 for value in targets
    )
    with torch.no_grad() if parametrized else nullcontext():
        cell.detach()
    sources = _parameter_identities(cell)
    before, start = cell.v.detach().clone(), cell.t.detach().clone()
    with torch.no_grad():
        targets[0].mul_(1.5)
        cell.cm_param.rho.add_(0.15)
    for value in cell.parameters():
        value.grad = None
    effective_g = owner.g_param
    effective_cm = cell.cm_param()
    expected = _expected(before, effective_g, effective_cm, -55.0, 3)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "continued training must not perform Population initialization or sampling"
        )

    monkeypatch.setattr(cell, "initialize", forbidden)
    monkeypatch.setattr(cell, "populate_parameter_buffers", forbidden)
    cell.longrun(tstop=3 * DT, dt=DT, chunklength=2)
    _close(cell.v, expected)
    _close(cell.t, start + 3 * DT)
    observed = torch.autograd.grad(cell.v.square().sum(), targets)
    wanted = torch.autograd.grad(expected.square().sum(), targets)
    _close(observed, wanted)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed
    )
    assert not cell._detached_materializations_pending
    _assert_sources(cell, sources)


@pytest.mark.parametrize("context", ("eval", "no_grad"))
def test_inference_preserves_pending_refresh_until_later_grad_enabled_training(context):
    cell = _cell()
    cell.run(tstop=2 * DT, dt=DT)
    cell.v.square().sum().backward()
    with torch.no_grad():
        cell.detach()
        cell.mech.pas.g_param.mul_(1.4)
    saved_g = cell.mech.pas.g.detach().clone()
    if context == "eval":
        cell.eval()
    with torch.no_grad() if context == "no_grad" else nullcontext():
        cell.step(DT)
    assert cell._detached_materializations_pending
    _close(cell.mech.pas.g, saved_g)
    before, time = cell.v.detach().clone(), cell.t.detach().clone()
    cell.train()
    cell.mech.pas.g_param.grad = None
    cell.step(DT)
    expected = _expected(before, cell.mech.pas.g_param, cell.cm_param(), -55.0, 1)
    _close(cell.v, expected)
    _close(cell.t, time + DT)
    observed = torch.autograd.grad(cell.v.square().sum(), cell.mech.pas.g_param)[0]
    wanted = torch.autograd.grad(expected.square().sum(), cell.mech.pas.g_param)[0]
    _close(observed, wanted)
    assert observed.abs().sum() > 0 and not cell._detached_materializations_pending


class _RandomLeak(Mechanism):
    Mechanism.RANGE(g=0.001, e=-60.0)
    Mechanism.GLOBALRAND(
        "offset", mu=0.00001, sigma=0.000003, seed=103, resample_on_initialize=False
    )
    Mechanism.RANGENOISE("eta", mu=0.0, sigma=0.000002, seed=107)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        return self.g, self.g * self.e + self.offset + self.eta.clone()


def _random_cell():
    cell = dn.SingleCompartment(
        N=1, C=2, v_init=-63.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    cell.insert(_RandomLeak.rename("probe"))
    cell.train()
    cell.initialize()
    return cell


def test_detach_replays_retained_rand_graphs_without_sampling_or_changing_noise_semantics(
    monkeypatch,
):
    cell, reference = _random_cell(), _random_cell()
    cell.run(tstop=2 * DT, dt=DT)
    cell.v.square().sum().backward()
    reference.restore_dict_from_checkpoint(_snapshot(cell.state_dict_for_checkpoint()))
    owner = cell.mech.probe
    cell.detach()
    reference.detach()
    sources = _parameter_identities(cell)
    sample, noise = owner.offset.detach().clone(), owner.eta.detach().clone()
    provenance = _snapshot(owner._random_parameter_state_for_checkpoint())
    before, start = cell.v.detach().clone(), cell.t.detach().clone()
    streams = owner.offset_rng.get_extra_state(), owner.eta_rng.get_extra_state()
    global_rng = torch.get_rng_state().clone()
    for candidate in (cell, reference):
        with torch.no_grad():
            candidate.mech.probe.offset_mu_param.add_(0.01)
            candidate.mech.probe.offset_sigma_param.rho.add_(0.03)

    class CheckRefresh(dn.callbacks.Callback):
        def pre_step_hook(self, model):
            _close(model.v, before)
            _close(model.t, start)
            torch.testing.assert_close(owner.offset, sample, rtol=0, atol=0)
            torch.testing.assert_close(owner.eta, noise, rtol=0, atol=0)
            assert owner.offset.grad_fn is not None and owner.offset.requires_grad
            assert not owner.eta.requires_grad and owner.eta.grad_fn is None
            _assert_packet_equal(
                owner._random_parameter_state_for_checkpoint(), provenance
            )
            _assert_packet_equal(owner.offset_rng.get_extra_state(), streams[0])
            _assert_packet_equal(owner.eta_rng.get_extra_state(), streams[1])
            _close(torch.get_rng_state(), global_rng)

    def forbidden(*args, **kwargs):
        raise AssertionError("detached RAND refresh must reuse saved explicit draws")

    monkeypatch.setattr(cell, "initialize", forbidden)
    monkeypatch.setattr(cell, "populate_parameter_buffers", forbidden)
    monkeypatch.setattr(owner, "_sample_random_parameters", forbidden)
    cell.step(DT, callbacks=[CheckRefresh()])
    with torch.no_grad():
        reference.step(DT)
    _close(cell.v, reference.v)
    _assert_packet_equal(
        owner.offset_rng.get_extra_state(),
        reference.mech.probe.offset_rng.get_extra_state(),
    )
    _assert_packet_equal(
        owner.eta_rng.get_extra_state(), reference.mech.probe.eta_rng.get_extra_state()
    )
    targets = owner.offset_mu_param, owner.offset_sigma_param.rho
    gradients = torch.autograd.grad(cell.v.square().sum(), targets)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
    )
    assert not owner.eta.requires_grad and owner.eta.grad_fn is None
    torch.testing.assert_close(owner.offset, sample, rtol=0, atol=0)
    _assert_sources(cell, sources)


def test_network_checkpoint_training_consumes_population_refresh_and_keeps_latest_state():
    shared = torch.nn.Parameter(torch.tensor(0.001, dtype=torch.float64))
    network = dn.Network(
        {
            "left": _cell(g=shared),
            "right": _cell(g=shared, reversal=-70.0, voltage=-67.0),
        }
    )
    network.train()
    network.initialize(DT)
    network.run(2 * DT)
    sum(
        population.v.square().sum() for population in network.populations.values()
    ).backward()
    for population in network.populations.values():
        population.detach()
    sources = _parameter_identities(network)
    before = {
        name: population.v.detach().clone()
        for name, population in network.populations.items()
    }
    start = network.t.detach().clone()
    with torch.no_grad():
        shared.mul_(1.3)
        network.left.cm_param.rho.add_(0.1)
    shared.grad = network.left.cm_param.rho.grad = None
    expected = {
        name: _expected(
            before[name],
            shared,
            population.cm_param(),
            -55.0 if name == "left" else -70.0,
            3,
        )
        for name, population in network.populations.items()
    }
    loss, final = network.longrun_checkpointed(
        tstop=3 * DT, chunklength=2, safe_checkpoint=True, return_final_state=True
    )
    assert loss is None
    for name, values in expected.items():
        _close(final["populations"][name]["integrator"]["v"], values)
        assert not network.populations[name]._detached_materializations_pending
    _close(network.t, start + 3 * DT)
    frozen = _snapshot(network.state_dict_for_checkpoint())
    targets = shared, network.left.cm_param.rho
    objective = sum(
        final["populations"][name]["integrator"]["v"].square().sum()
        for name in expected
    )
    observed = torch.autograd.grad(objective, targets)
    wanted = torch.autograd.grad(
        sum(value.square().sum() for value in expected.values()), targets
    )
    _close(observed, wanted)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed
    )
    _assert_packet_equal(network.state_dict_for_checkpoint(), frozen)
    _assert_sources(network, sources)


@pytest.mark.parametrize("network", (False, True), ids=("packed_population", "network"))
def test_packed_detach_refreshes_child_physics_without_advancing_child_runtime(network):
    shared = torch.nn.Parameter(torch.tensor(0.001, dtype=torch.float64))
    left = _cell(g=shared)
    right = _cell(g=shared, reversal=-70.0, voltage=-67.0)
    packed = dn.concat_models({"left": left, "right": right}, write_back=False)
    owner = dn.Network({"packed": packed}) if network else packed
    owner.train()
    owner.initialize(DT) if network else owner.initialize()
    if network:
        owner.run(2 * DT)
    else:
        owner.run(tstop=2 * DT, dt=DT)
    packed.v.square().sum().backward()
    boundary, start = packed.v.detach().clone(), owner.t.detach().clone()
    child_states = left.v.detach().clone(), right.v.detach().clone()
    packed.detach()
    _close(packed.v, boundary)
    _close(owner.t, start)
    sources = _parameter_identities(owner)
    assert packed.populations["left"].cm_param.rho is left.cm_param.rho
    assert packed.mech.pas.g_left is packed.mech.pas.g_right is shared
    with torch.no_grad():
        left.cm_param.rho.add_(0.2)
    targets = shared, left.cm_param.rho
    for value in targets:
        value.grad = None
    capacitances = left.cm_param(), right.cm_param()
    expected_parts = (
        _expected(
            boundary[..., : left.v.numel()].reshape_as(left.v),
            shared,
            capacitances[0],
            -55.0,
            3,
        ),
        _expected(
            boundary[..., left.v.numel() :].reshape_as(right.v),
            shared,
            capacitances[1],
            -70.0,
            3,
        ),
    )
    expected = torch.cat(
        tuple(value.reshape(1, -1) for value in expected_parts), dim=-1
    ).reshape_as(packed.v)
    if network:
        owner.longrun(3 * DT, chunklength=2)
    else:
        owner.longrun(tstop=3 * DT, dt=DT, chunklength=2)
    _close(packed.v, expected)
    _close(owner.t, start + 3 * DT)
    _close(packed.t, owner.t)
    _close(
        packed.cm[..., : left.v.numel()],
        capacitances[0].expand_as(left.v).reshape(1, -1),
    )
    observed = torch.autograd.grad(packed.v.square().sum(), targets)
    wanted = torch.autograd.grad(expected.square().sum(), targets)
    _close(observed, wanted)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed
    )
    for population, value in zip((left, right), child_states, strict=True):
        torch.testing.assert_close(population.v, value, rtol=0, atol=0)
    assert not packed._detached_materializations_pending
    _assert_sources(owner, sources)
