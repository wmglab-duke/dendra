"""Prepared callbacks share argument spelling across native execution hosts."""

from __future__ import annotations

import os
import sys

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.core import _native_callback_arguments
from dendra.models.mod import pas
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_continuation_checkpoints import _disk
from tests.func.test_imperative_checkpoint_callbacks import DT
from tests.func.test_native_callback_storage import _join, _read
from tests.func.test_native_network_checkpoint_callbacks import _NetworkEnergy
from tests.func.test_native_population_functional_callbacks import (
    _PortableEnergyReducer,
)
from tests.func.test_native_step_functional_callbacks import _case, _parameter, _plan
from tests.test_recorder_h5_integration import local_hdf5_writer as local_hdf5_writer

pytestmark = pytest.mark.cpu
_METHODS = ("step", "run", "longrun", "longrun_checkpointed")


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


def _call(model, network, method, *, steps=3, **arguments):
    if method == "step":
        return model.step(**arguments) if network else model.step(DT, **arguments)
    arguments["tstop"] = steps * DT
    if not network:
        arguments["dt"] = DT
    if method != "run":
        arguments["chunklength"] = 2
    if method == "longrun_checkpointed":
        arguments.update(safe_checkpoint=True, restore_state_after_backward=True)
    return getattr(model, method)(**arguments)


def _result(value, method):
    if method == "longrun_checkpointed":
        assert type(value) is tuple and len(value) == 2 and value[0] is None
        return value[1]
    assert type(value) is dn.func.FunctionalCallbackResults
    return value


def _objective(result, network):
    value = (
        result["energy"]["energy"].sum() + result["views"]["trace"]["v"].square().mean()
    )
    return value + (result["lfp"] * 1e6).square().mean() if network else value


def _targets(model, amplitude, field, network):
    return (
        (field, amplitude, _parameter(model, network))
        if network
        else (_parameter(model, network),)
    )


def test_argument_resolution_preserves_lazy_native_objects_and_prepared_subclasses(
    monkeypatch,
):
    class NativeHooks:
        @property
        def __class__(self):
            raise AssertionError("native callback class descriptors must not be read")

        @property
        def _storage(self):
            raise AssertionError("native callback storage must not be read")

        def __iter__(self):
            raise AssertionError("argument resolution must not consume native hooks")

    native = NativeHooks()
    for unloaded in (False, True):
        with monkeypatch.context() as patch:
            if unloaded:
                patch.delitem(sys.modules, "dendra.func._callbacks")
            loaded = frozenset(sys.modules)
            hooks, observer = _native_callback_arguments(
                object(), native, None, None, None
            )
            assert hooks is native and observer is None
            assert frozenset(sys.modules) == loaded

    class Prepared(dn.func.FunctionalCallbacks):
        pass

    model, _ = _case(False)
    original = _plan(model, False)
    prepared = Prepared(original._functional, {"trace": dn.Recorder(("v", "t"))})
    hooks, observer = _native_callback_arguments(model, prepared, None, None, None)
    assert hooks is None and observer.callbacks is prepared
    assert observer.initial_callback_state is None
    # Duplicate argument checks precede source binding and carry validation.
    with pytest.raises(
        ValueError,
        match=r"^Provide either callbacks or functional_callbacks, not both\.$",
    ):
        _native_callback_arguments(object(), prepared, object(), object(), object())
    with pytest.raises(
        ValueError,
        match=r"^Provide either callback_state or functional_callback_state, not both\.$",
    ):
        _native_callback_arguments(object(), prepared, object(), None, object())


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("method", _METHODS)
def test_canonical_callback_arguments_match_legacy_results_and_strict_native_gradients(
    network, method
):
    model, amplitude = _case(network)
    reference, reference_amplitude = _case(network)
    field = torch.ones_like(model.left.v, requires_grad=True) if network else None
    reference_field = (
        torch.ones_like(reference.left.v, requires_grad=True) if network else None
    )
    callbacks = _plan(model, network, field=field)
    previous = _plan(reference, network, field=reference_field)
    observed = _result(_call(model, network, method, callbacks=callbacks), method)
    expected = _result(
        _call(reference, network, method, functional_callbacks=previous), method
    )
    _close(observed.outputs, expected.outputs)
    assert observed["energy"]["steps"].item() == (1 if method == "step" else 3)
    frozen = _snapshot(model.state_dict_for_checkpoint())
    actual_gradient = torch.autograd.grad(
        _objective(observed, network), _targets(model, amplitude, field, network)
    )
    wanted = torch.autograd.grad(
        _objective(expected, network),
        _targets(reference, reference_amplitude, reference_field, network),
    )
    _close(actual_gradient, wanted)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0
        for value in actual_gradient
    )
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_callback_state_spelling_continues_helpers_native_methods_and_portable_carry(
    network, tmp_path
):
    model, _ = _case(network)
    callbacks = _plan(model, network)
    first = dn.step(model, **({} if network else {"dt": DT}), callbacks=callbacks)
    helper = dn.step_network if network else dn.step_population
    second = helper(
        model,
        **({} if network else {"dt": DT}),
        callbacks=callbacks,
        callback_state=first.state,
    )
    empty = _call(
        model, network, "run", steps=0, callbacks=callbacks, callback_state=second.state
    )
    trace = "qualified" if network else "trace"
    assert empty[trace]["t"].shape == (0,)
    _close(empty.state.carries, second.state.carries)
    third = _call(
        model,
        network,
        "run",
        steps=1,
        functional_callbacks=callbacks,
        callback_state=empty.state,
    )
    packet = _disk(callbacks.export_state(third.state), tmp_path / "callbacks.pt")
    receiving = _plan(model, network)
    restored = receiving.restore_state(packet)
    fourth = _result(
        _call(
            model,
            network,
            "longrun_checkpointed",
            steps=1,
            callbacks=receiving,
            functional_callback_state=restored,
        ),
        "longrun_checkpointed",
    )
    fifth = _call(
        model,
        network,
        "longrun",
        steps=1,
        functional_callbacks=receiving,
        functional_callback_state=fourth.state,
    )
    reference, _ = _case(network)
    expected = _call(
        reference,
        network,
        "run",
        steps=5,
        functional_callbacks=_plan(reference, network),
    )
    _close(_join((first, second, empty, third, fourth, fifth), trace), expected[trace])
    _close(
        torch.cat(
            tuple(
                part["views"]["trace"]["v"]
                for part in (first, second, empty, third, fourth, fifth)
            )
        ),
        expected["views"]["trace"]["v"],
    )
    _close(fifth["energy"], expected["energy"])
    _close(fifth["count"], expected["count"])
    assert fifth["energy"]["steps"].item() == 5


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_callback_argument_conflicts_and_orphan_carry_reject_before_advancing(network):
    model, _ = _case(network)
    callbacks = _plan(model, network)
    initial = _call(model, network, "run", steps=0, callbacks=callbacks)
    frozen = _snapshot(model.state_dict_for_checkpoint())
    for method in _METHODS:
        with pytest.raises(ValueError, match="callbacks.*functional_callbacks"):
            _call(
                model,
                network,
                method,
                steps=0,
                callbacks=callbacks,
                functional_callbacks=callbacks,
            )
        with pytest.raises(
            ValueError, match="callback_state.*functional_callback_state"
        ):
            _call(
                model,
                network,
                method,
                steps=0,
                callbacks=callbacks,
                callback_state=initial.state,
                functional_callback_state=initial.state,
            )
        with pytest.raises(ValueError, match="callback_state.*requires"):
            _call(model, network, method, steps=0, callback_state=initial.state)
        with pytest.raises(ValueError, match="callback_state.*requires"):
            _call(
                model,
                network,
                method,
                steps=0,
                callbacks=[],
                callback_state=initial.state,
            )
        with pytest.raises(TypeError, match="FunctionalCallbackState"):
            _call(
                model,
                network,
                method,
                steps=0,
                callbacks=callbacks,
                callback_state=object(),
            )
        with pytest.raises(dn.func.FunctionalizationError, match="exact source"):
            _call(_case(network)[0], network, method, steps=0, callbacks=callbacks)
        _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
    if not network:
        with pytest.raises(dn.func.FunctionalizationError, match="dt|timestep"):
            model.step(2 * DT, callbacks=callbacks)
    else:
        with pytest.raises(TypeError, match="only valid for Population"):
            dn.step(
                model,
                ve=torch.zeros_like(model.left.v),
                callbacks=callbacks,
                callback_state=initial.state,
            )
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("method", ("run", "longrun_checkpointed"))
def test_native_hook_lists_coexist_with_legacy_collections_and_canonical_carry(
    network, method
):
    class Hooks(dn.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.calls = {"pre_loop": 0, "post_loop": 0, "pre_step": 0, "post_step": 0}

        def pre_loop_hook(self, model):
            self.calls["pre_loop"] += 1

        def post_loop_hook(self, model):
            self.calls["post_loop"] += 1

        def pre_step_hook(self, model):
            self.calls["pre_step"] += 1

        def post_step_hook(self, model):
            self.calls["post_step"] += 1
            populations = model.populations.values() if network else (model,)
            return sum(population.v.square().mean() for population in populations)

    model, _ = _case(network)
    callbacks = _plan(model, network)
    initial = _call(model, network, "run", steps=0, callbacks=callbacks)
    hooks = Hooks()
    value = _call(
        model,
        network,
        method,
        callbacks=[hooks],
        functional_callbacks=callbacks,
        callback_state=initial.state,
    )
    if method == "longrun_checkpointed":
        loss, observed = value
        _close(loss, observed["energy"]["energy"].sum())
    else:
        observed = value
    assert type(observed) is dn.func.FunctionalCallbackResults
    assert hooks.calls == {
        "pre_loop": 1,
        "post_loop": 1,
        "pre_step": 0 if method == "longrun_checkpointed" else 3,
        "post_step": 3,
    }
    assert observed["energy"]["steps"].item() == 3
    ordinary, _ = _case(network)
    native = Hooks()
    ordinary_value = _call(ordinary, network, method, callbacks=[native])
    assert (
        torch.is_tensor(ordinary_value)
        if method == "longrun_checkpointed"
        else ordinary_value is None
    )
    assert native.calls == hooks.calls


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_canonical_callbacks_keep_hdf5_storage_and_checkpoint_replay_ownership(
    network,
    tmp_path,
    local_hdf5_writer,  # noqa: F811 - pytest injects the imported fixture
):
    model, _ = _case(network)
    fields = ("left.v", "right.v", "t") if network else ("v", "hh.m", "t")
    path = tmp_path / "canonical.h5"
    disk = dn.Recorder(fields, dt=2 * DT).set_hdf5(str(path), cache_every=3)
    callbacks = dn.func.make_callbacks(
        model,
        {
            "disk": disk,
            "memory": dn.Recorder(fields, dt=2 * DT),
            "energy": _NetworkEnergy() if network else _PortableEnergyReducer(),
        },
        dt=DT,
    )
    try:
        with pytest.raises(ValueError, match="callbacks.*functional_callbacks"):
            _call(
                model,
                network,
                "run",
                callbacks=callbacks,
                functional_callbacks=callbacks,
            )
        assert disk._hdf5_session is None and disk.run_number == 0
        assert local_hdf5_writer.queue.items == []
        first = _call(model, network, "run", steps=2, callbacks=callbacks)
        last = _result(
            _call(
                model,
                network,
                "longrun_checkpointed",
                callbacks=callbacks,
                callback_state=first.state,
            ),
            "longrun_checkpointed",
        )
        assert tuple(last) == ("memory", "energy") and last.state["disk"].item() == 5
        assert disk._hdf5_session is None and disk.run_number == 2
        queued = len(local_hdf5_writer.queue.items)
        frozen = _snapshot(model.state_dict_for_checkpoint())
        gradient = torch.autograd.grad(
            last["energy"]["energy"].sum(), _parameter(model, network)
        )[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        assert len(local_hdf5_writer.queue.items) == queued
        _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
    finally:
        disk.close()
    _close(_read(path, fields), _join((first, last)))


@pytest.mark.mps
def test_mps_canonical_callback_arguments_continue_public_helpers_and_gradients():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DEVICE="mps", DTYPE=torch.float32):

        def build():
            model = dn.SingleCompartment(
                N=1, C=2, v_init=-63, integrator=dn.bwd_euler_sc(imem=False)
            )
            model.insert(pas, g=0.001, e=-60)
            model.train()
            model.initialize()
            return model

        model, reference = build(), build()
        specifications = {
            "trace": dn.Recorder(("v", "t"), dt=2 * DT),
            "energy": _PortableEnergyReducer(),
        }
        callbacks = dn.func.make_callbacks(model, specifications, dt=DT)
        first = dn.step_population(model, dt=DT, callbacks=callbacks)
        last = dn.step(model, dt=DT, callbacks=callbacks, callback_state=first.state)
        expected = reference.run(
            tstop=2 * DT,
            dt=DT,
            functional_callbacks=dn.func.make_callbacks(
                reference, specifications, dt=DT
            ),
        )
        _close(_join((first, last), "trace"), expected["trace"])
        observed = torch.autograd.grad(
            last["energy"]["energy"].sum(), model.mech.pas.g_param
        )[0]
        wanted = torch.autograd.grad(
            expected["energy"]["energy"].sum(), reference.mech.pas.g_param
        )[0]
        _close(observed, wanted)
        assert observed.device.type == "mps" and observed.abs().sum() > 0
