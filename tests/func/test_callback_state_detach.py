"""Callback carry truncates independently of the caller-owned model state."""

from __future__ import annotations

import os
from typing import NamedTuple

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models._recorder_storage import _RecorderHDF5Session
from dendra.models.mod import pas
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_continuation_checkpoints import _disk, _tensors
from tests.func.test_continuation_detach import _History, _initial, _loss, _program
from tests.func.test_imperative_checkpoint_callbacks import DT
from tests.func.test_lowering_api import _population, _target
from tests.func.test_native_population_functional_callbacks import (
    _PortableEnergyReducer,
)
from tests.func.test_native_step_functional_callbacks import _case, _parameter, _plan
from tests.func.test_shared_lfp_configuration import _case as _lfp_case
from tests.func.test_shared_lfp_configuration import _scoped
from tests.func.test_stimulation_rng_sharing import DT as NOISE_DT
from tests.func.test_stimulation_rng_sharing import _population as _noisy_population
from tests.test_recorder_h5_integration import local_hdf5_writer as local_hdf5_writer

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64, IMEM=True),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _native(model, network, method, callbacks, *, state=None, steps=3):
    options = {"tstop": steps * DT, "callbacks": callbacks}
    if not network:
        options["dt"] = DT
    if state is not None:
        options["callback_state"] = state
    if method != "run":
        options["chunklength"] = 2
    if method == "longrun_checkpointed":
        options.update(safe_checkpoint=True, restore_state_after_backward=True)
    result = getattr(model, method)(**options)
    if method == "longrun_checkpointed":
        assert result[0] is None
        return result[1]
    return result


def _detach_native(model, network):
    # This fixture has no connections; each Population owns its numerical
    # graphs. Network connection history remains the caller's responsibility.
    for population in model.populations.values() if network else (model,):
        population.detach()


def _objective(result):
    return (
        result["energy"]["energy"].sum() + result["views"]["trace"]["v"].square().mean()
    )


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_native_training_detaches_model_and_callback_carry_before_suffix_backward(
    network, tmp_path
):
    model, _ = _case(network)
    callbacks = _plan(model, network)
    first = _native(model, network, "run", callbacks, steps=2)
    _objective(first).backward()
    assert _parameter(model, network).grad.abs().sum() > 0
    _detach_native(model, network)
    native = _snapshot(model.state_dict_for_checkpoint())
    original = _snapshot(first.state.carries)
    state = callbacks.detach_state(first.state)
    assert state is not first.state and state.plan_token is first.state.plan_token
    assert state.emission_schemas is first.state.emission_schemas
    _close(state.carries, original)
    for name in ("views", "energy"):
        assert all(
            value.grad_fn is None and not value.requires_grad
            for value in _tensors(state[name])
        )
    _assert_packet_equal(model.state_dict_for_checkpoint(), native)
    reference, _ = _case(network)
    receiving = _plan(reference, network)
    _native(reference, network, "run", receiving, steps=0)
    reference.restore_dict_from_checkpoint(native)
    restored = receiving.restore_state(
        _disk(callbacks.export_state(state), tmp_path / "carry.pt")
    )
    _parameter(model, network).grad = None
    actual = _native(model, network, "longrun_checkpointed", callbacks, state=state)
    expected = _native(reference, network, "longrun", receiving, state=restored)
    _close(actual.outputs, expected.outputs)
    frozen = _snapshot(model.state_dict_for_checkpoint())
    observed = torch.autograd.grad(_objective(actual), _parameter(model, network))[0]
    wanted = torch.autograd.grad(_objective(expected), _parameter(reference, network))[
        0
    ]
    _close(observed, wanted)
    assert torch.isfinite(observed).all() and observed.abs().sum() > 0
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
    stream = "qualified" if network else "trace"
    assert actual[stream]["t"].shape == (1,)
    assert actual["energy"]["steps"].item() == 5


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_lowered_execution_truncates_callback_and_model_graphs_independently(network):
    program = _program(network)
    functional, callbacks = program.functional, program.callbacks
    state, first = dn.func.run(
        functional, _initial(program), steps=3, callbacks=callbacks
    )
    _loss(state, first, network).backward()
    detached = callbacks.detach_state(first["callbacks"].state)
    cut_state = pytree.tree_map_only(torch.Tensor, torch.Tensor.detach, state)
    packet = dn.func.Continuation(
        cut_state,
        first["continuation"].parameters,
        first["continuation"].constants,
        detached,
    )
    selected = _target(packet.parameters, network)
    selected.grad = None
    actual, last = dn.func.run(functional, packet, steps=2, callbacks=callbacks)
    _loss(actual, last, network).backward()
    observed = selected.grad.clone()
    expected, reference = dn.func.run(functional, packet, steps=2, callbacks=callbacks)
    wanted = torch.autograd.grad(_loss(expected, reference, network), selected)[0]
    _close(observed, wanted)
    assert torch.isfinite(observed).all() and observed.abs().sum() > 0
    _close(actual, expected)
    _close(last["callbacks"].outputs, reference["callbacks"].outputs)
    assert _target(packet.parameters, network) is _target(program.parameters, network)


@pytest.mark.parametrize("network", (False, True), ids=("population", "scoped_network"))
def test_native_lfp_optimizer_iterations_refresh_fields_without_resetting_phase(
    network,
):
    model, population, _, _ = _lfp_case(network)
    dt = 0.01
    population[:, 0].inject(dn.mono_rect(amp=2e-6, delay=0, pw=20 * dt))
    model.initialize(dt) if network else model.initialize()
    field = torch.nn.Parameter(torch.ones_like(population.v))
    callbacks = dn.func.make_callbacks(
        model,
        {
            "lfp": _scoped(dn.callbacks.LFP([field, field]), network),
            "trace": _scoped(dn.Recorder(("v", "i_membrane", "t")), network),
            "coarse": _scoped(dn.Recorder(("v", "t"), dt=2 * dt), network),
            "history": _scoped(_History(), network),
        },
        dt=dt,
    )
    conductance = population.mech.hh.gnabar_param
    optimizer = torch.optim.SGD(
        [{"params": [conductance], "lr": 1e-8}, {"params": [field], "lr": 1e-4}]
    )
    state = None
    for iteration in range(3):
        options = {"callbacks": callbacks, "tstop": 2 * dt}
        if not network:
            options["dt"] = dt
        if state is not None:
            options["callback_state"] = state
        result = model.run(**options)
        current = result["trace"]["i_membrane"].detach()
        proxy = field.detach().clone().requires_grad_()
        expected = (
            (current.unsqueeze(1) * torch.stack((proxy, proxy)).unsqueeze(0))
            .sum((-1, -2))
            .movedim(1, -1)
        )
        _close(result["lfp"], expected)
        wanted = torch.autograd.grad((expected * 1e6).square().mean(), proxy)[0]
        (
            (result["lfp"] * 1e6).square().mean()
            + 1e-3 * result["history"]["latest"].square().mean()
        ).backward()
        _close(field.grad, wanted)
        assert (
            conductance.grad is not None
            and torch.isfinite(conductance.grad).all()
            and conductance.grad.abs().sum() > 0
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _detach_native(model, network)
        for context in (torch.no_grad, torch.inference_mode):
            with context():
                frozen = callbacks.detach_state(result.state)
            assert not frozen["lfp"].requires_grad and frozen["lfp"].grad_fn is None
            assert not torch.is_inference(frozen["lfp"])
        state = callbacks.detach_state(result.state)
        assert (
            state["lfp"] is not result.state["lfp"] and state["lfp"].grad_fn is not None
        )
        _close(state["lfp"], torch.stack((field, field)))
        assert all(
            not value.requires_grad and value.grad_fn is None
            for value in _tensors(state["history"])
        )
        assert state["coarse"].item() == 2 * (iteration + 1)
        assert result["coarse"]["t"].shape[0] == (2 if iteration == 0 else 1)


class _Pair(NamedTuple):
    left: torch.Tensor
    right: torch.Tensor


class _TupleCarry(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        value = state["integrator"]["v"]
        return _Pair(value, value.t()), {"v": value}

    def update(self, carry, state, auxiliary):
        return self.initialize(state, auxiliary)

    def finalize(self, carry, emissions):
        return {"carry": carry, "samples": emissions}


def test_callback_detachment_preserves_namedtuple_aliases_views_and_emission_metadata():
    model = _population()
    functional, tensors = dn.func.make_functional(model, dt=0.025)
    callbacks = functional.make_callbacks(
        {"first": _TupleCarry(), "second": _TupleCarry()}
    )
    _, auxiliary = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        steps=2,
        callbacks=callbacks,
    )
    original = auxiliary["callbacks"].state
    value = original["first"].left
    first, second = _Pair(value, value.t()), _Pair(value, value.t())
    state = dn.func.FunctionalCallbackState(
        {"first": first, "second": second},
        original.plan_token,
        original.emission_schemas,
    )
    detached = callbacks.detach_state(state)
    assert type(detached["first"]) is _Pair
    assert detached["first"].left is detached["second"].left
    assert detached["first"].right is not detached["second"].right
    assert detached["first"].left.data_ptr() == value.data_ptr()
    assert detached["first"].right.data_ptr() == first.right.data_ptr()
    assert detached["first"].right.stride() == first.right.stride()
    assert detached.plan_token is state.plan_token
    assert detached.emission_schemas is state.emission_schemas
    assert all(
        value.grad_fn is None and not value.requires_grad
        for value in _tensors(detached.carries)
    )
    _close(detached.carries, state.carries)
    assert original["first"].left is value and value.grad_fn is not None


def test_detach_state_does_not_read_model_initialize_sample_export_or_touch_storage(
    tmp_path,
    local_hdf5_writer,  # noqa: F811 - pytest injects the imported fixture
    monkeypatch,
):
    model, _, pulse = _noisy_population()
    disk = dn.Recorder(("v", "t")).set_hdf5(
        str(tmp_path / "observations.h5"), cache_every=2
    )
    history = _History()
    callbacks = dn.func.make_callbacks(
        model, {"disk": disk, "history": history}, dt=NOISE_DT
    )
    try:
        first = model.run(tstop=3 * NOISE_DT, dt=NOISE_DT, callbacks=callbacks)
        native = _snapshot(model.state_dict_for_checkpoint())
        rng, stream = torch.get_rng_state().clone(), pulse.generator.get_state().clone()
        queued, run_number = len(local_hdf5_writer.queue.items), disk.run_number

        def forbidden(*args, **kwargs):
            raise AssertionError("callback detachment must inspect carry only")

        monkeypatch.setattr(callbacks._functional, "_validate_source", forbidden)
        for name in ("extract", "prepare", "initialize_host", "sample_runtime_draws"):
            monkeypatch.setattr(callbacks._functional, name, forbidden)
        monkeypatch.setattr(_History, "initialize", forbidden)
        monkeypatch.setattr(_History, "checkpoint_configuration", forbidden)
        monkeypatch.setattr(_RecorderHDF5Session, "__init__", forbidden)
        state = callbacks.detach_state(first.state)
        _close(state.carries, first.state.carries)
        assert state.plan_token is first.state.plan_token
        assert disk._hdf5_session is None and disk.run_number == run_number
        assert len(local_hdf5_writer.queue.items) == queued
        _assert_packet_equal(model.state_dict_for_checkpoint(), native)
        _close(torch.get_rng_state(), rng)
        _close(pulse.generator.get_state(), stream)
    finally:
        disk.close()


def test_detach_state_validates_owner_schema_and_cold_boundary_without_model_work(
    monkeypatch,
):
    model, _ = _case(False)
    callbacks = _plan(model, False)
    first = _native(model, False, "run", callbacks)
    for invalid in (None, object(), first):
        with pytest.raises(TypeError, match="FunctionalCallbackState"):
            callbacks.detach_state(invalid)
    other = _plan(model, False)
    with pytest.raises(dn.func.FunctionalizationError, match="different callback plan"):
        other.detach_state(first.state)
    malformed = dn.func.FunctionalCallbackState(
        {}, first.state.plan_token, first.state.emission_schemas
    )
    with pytest.raises(ValueError, match="keys do not match"):
        callbacks.detach_state(malformed)
    carries = {
        **first.state.carries,
        "views": {
            **first.state["views"],
            "latest": first.state["views"]["latest"].reshape(-1),
        },
    }
    malformed = dn.func.FunctionalCallbackState(
        carries, first.state.plan_token, first.state.emission_schemas
    )
    with pytest.raises((ValueError, dn.func.FunctionalizationError), match="shape"):
        callbacks.detach_state(malformed)
    with monkeypatch.context() as patch:
        patch.setattr(torch.compiler, "is_compiling", lambda: True)
        with pytest.raises(
            dn.func.FunctionalizationError,
            match="FunctionalCallbacks.detach_state.*outside",
        ):
            callbacks.detach_state(None)
    with torch.autograd.forward_ad.dual_level():
        with pytest.raises(dn.func.FunctionalizationError, match="outside"):
            callbacks.detach_state(None)
    with pytest.raises(dn.func.FunctionalizationError, match="outside"):
        torch.func.grad(lambda value: callbacks.detach_state(None))(torch.ones(()))


@pytest.mark.mps
def test_mps_native_callback_state_detachment_supports_another_backward():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DEVICE="mps", DTYPE=torch.float32, IMEM=False):
        model = dn.SingleCompartment(
            N=1, C=2, v_init=-63, integrator=dn.bwd_euler_sc(imem=False)
        )
        model.insert(pas, g=0.001, e=-60)
        model.train()
        model.initialize()
        callbacks = dn.func.make_callbacks(
            model,
            {"trace": dn.Recorder(("v", "t")), "energy": _PortableEnergyReducer()},
            dt=DT,
        )
        first = model.run(tstop=2 * DT, dt=DT, callbacks=callbacks)
        first["energy"]["energy"].sum().backward()
        model.detach()
        state = callbacks.detach_state(first.state)
        model.mech.pas.g_param.grad = None
        last = model.run(tstop=2 * DT, dt=DT, callbacks=callbacks, callback_state=state)
        last["energy"]["energy"].sum().backward()
        gradient = model.mech.pas.g_param.grad
        assert (
            gradient.device.type == "mps"
            and torch.isfinite(gradient).all()
            and gradient.abs().sum() > 0
        )
        assert last["energy"]["steps"].item() == 4
