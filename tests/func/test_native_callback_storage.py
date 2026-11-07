"""Native and functional observation hosts share Recorder persistence."""

from __future__ import annotations

import os

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.data import H5Reader
from dendra.models._recorder_storage import _RecorderHDF5Session
from dendra.models.mod import pas
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_continuation_checkpoints import _disk
from tests.func.test_imperative_checkpoint_callbacks import DT, _gnabar, _model
from tests.func.test_native_network_checkpoint_callbacks import _NetworkEnergy
from tests.func.test_native_network_functional_callbacks import _driven
from tests.func.test_native_population_functional_callbacks import (
    _PortableEnergyReducer,
)
from tests.func.test_network_population_callbacks import _network
from tests.func.test_stimulation_rng_sharing import DT as STOCHASTIC_DT
from tests.func.test_stimulation_rng_sharing import (
    _gradient_leaves,
)
from tests.func.test_stimulation_rng_sharing import _network as _stochastic_network
from tests.func.test_stimulation_rng_sharing import (
    _population as _stochastic_population,
)
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


def _case(network):
    return _driven()[0] if network else _model(require_grad=True, stimulated=True)


def _fields(network):
    return ("left.v", "right.v", "t") if network else ("v", "hh.m", "t")


def _collection(model, disk=None, *, memory=True, dt=DT, cadence=2):
    network = isinstance(model, dn.Network)
    functional, _ = dn.func.make_functional(model, dt=dt)
    specifications = {
        "energy": _NetworkEnergy() if network else _PortableEnergyReducer(),
    }
    if disk is not None:
        specifications["disk"] = disk
    if memory:
        specifications["memory"] = dn.Recorder(_fields(network), dt=cadence * dt)
    return functional.make_callbacks(specifications)


def _execute(model, method, callbacks, *, state=None, steps=5, dt=DT, native=None):
    network = isinstance(model, dn.Network)
    options = {"functional_callbacks": callbacks, "callbacks": native}
    if state is not None:
        options["functional_callback_state"] = state
    if method == "step":
        assert steps == 1
        return model.step(**options) if network else model.step(dt, **options)
    options["tstop"] = steps * dt
    if not network:
        options["dt"] = dt
    if method != "run":
        # Neither the observation cadence nor the file cache divides this span.
        options["chunklength"] = 4
    if method == "longrun_checkpointed":
        options.update(safe_checkpoint=True, restore_state_after_backward=True)
    result = getattr(model, method)(**options)
    if method == "longrun_checkpointed":
        assert result[0] is None
        return result[1]
    return result


def _join(results, name="memory"):
    return {
        field: torch.cat(tuple(result[name][field] for result in results))
        for field in results[0][name]
    }


def _read(path, fields):
    with H5Reader(path) as reader:
        return {
            field: torch.cat(
                tuple(
                    torch.from_numpy(reader[field][run].data.compute()).clone()
                    for run in range(reader[field].n())
                )
            )
            for field in fields
        }


def _writes(writer, field):
    return [
        item[3]
        for item in writer.queue.items
        if isinstance(item, tuple) and item[0] == field
    ]


def _parameter(model):
    return (
        model.left.mech.pas.g_param if isinstance(model, dn.Network) else _gnabar(model)
    )


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("method", _METHODS)
def test_every_native_entrypoint_streams_cadenced_mechanism_state_and_keeps_gradients(
    tmp_path,
    local_hdf5_writer,  # noqa: F811 - pytest injects the imported fixture
    network,
    method,
):
    reference = _case(network)
    expected = _execute(reference, "run", _collection(reference))
    voltage_name = _fields(network)[0]
    wanted = torch.autograd.grad(
        expected["energy"]["energy"].sum()
        + expected["memory"][voltage_name].square().mean(),
        _parameter(reference),
    )
    model = _case(network)
    path = tmp_path / "observations.h5"
    recorder = dn.Recorder(_fields(network), dt=2 * DT).set_hdf5(
        str(path), cache_every=3
    )
    callbacks = _collection(model, recorder)
    try:
        results = []
        if method != "step":
            results.append(_execute(model, method, callbacks, steps=0))
            assert recorder.run_number == 1
            queued = len(local_hdf5_writer.queue.items)
            empty = _execute(model, method, callbacks, state=results[-1].state, steps=0)
            assert empty["memory"]["t"].shape == (0,)
            assert (
                recorder.run_number == 1
                and len(local_hdf5_writer.queue.items) == queued
            )
        for _ in range(5 if method == "step" else 1):
            results.append(
                _execute(
                    model,
                    method,
                    callbacks,
                    state=None if not results else results[-1].state,
                    steps=1 if method == "step" else 5,
                )
            )
        assert all(tuple(result) == ("energy", "memory") for result in results)
        _close(_join(results), expected["memory"])
        _close(results[-1]["energy"], expected["energy"])
        assert results[-1].state["disk"].item() == 5
        assert recorder.run_number == (3 if method == "step" else 2)
        assert recorder._hdf5_session is None and recorder.save_count == 0
        assert recorder.i == 0 and all(not values for values in recorder.rec.values())
        queued = len(local_hdf5_writer.queue.items)
        frozen = _snapshot(model.state_dict_for_checkpoint())
        observed = torch.autograd.grad(
            results[-1]["energy"]["energy"].sum()
            + _join(results)[voltage_name].square().mean(),
            _parameter(model),
        )
        _close(observed, wanted)
        assert observed[0].abs().sum() > 0
        _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
        assert len(local_hdf5_writer.queue.items) == queued
        writes = _writes(local_hdf5_writer, _fields(network)[0])
        assert [value.shape[0] for value in writes] == [1, 1, 1]
        assert all(
            value.device.type == "cpu"
            and not value.requires_grad
            and value.grad_fn is None
            for value in writes
        )
    finally:
        recorder.close()
    _close(_read(path, _fields(network)), expected["memory"])


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_storage_continues_across_native_methods_empty_spans_and_portable_destinations(
    tmp_path,
    local_hdf5_writer,  # noqa: F811 - pytest injects the imported fixture
    network,
):
    model, reference = _case(network), _case(network)
    expected = _execute(reference, "run", _collection(reference), steps=6)
    first_path, last_path = tmp_path / "prefix.h5", tmp_path / "suffix.h5"
    disk = dn.Recorder(_fields(network), dt=2 * DT).set_hdf5(
        str(first_path), cache_every=3
    )
    callbacks = _collection(model, disk)
    try:
        initial = _execute(model, "run", callbacks, steps=0)
        assert initial["memory"]["t"].shape == (1,) and disk.run_number == 1
        first = _execute(model, "step", callbacks, state=initial.state, steps=1)
        assert first["memory"]["t"].shape == (0,) and disk.run_number == 1
        queued = len(local_hdf5_writer.queue.items)
        empty = _execute(model, "longrun", callbacks, state=first.state, steps=0)
        assert empty["memory"]["t"].shape == (0,)
        assert len(local_hdf5_writer.queue.items) == queued and disk.run_number == 1
        middle = _execute(
            model, "longrun_checkpointed", callbacks, state=empty.state, steps=2
        )
        assert disk.run_number == 2
        native_checkpoint = _snapshot(model.state_dict_for_checkpoint())
        packet = _disk(callbacks.export_state(middle.state), tmp_path / "carry.pt")
    finally:
        disk.close()
    receiver = _case(network)
    receiver.restore_dict_from_checkpoint(native_checkpoint)
    receiving_disk = dn.Recorder(_fields(network), dt=2 * DT).set_hdf5(
        str(last_path), cache_every=2
    )
    receiving = _collection(receiver, receiving_disk)
    try:
        restored = receiving.restore_state(packet)
        last = _execute(receiver, "longrun", receiving, state=restored, steps=3)
        assert last.state["disk"].item() == 6 and receiving_disk.run_number == 1
        _close(last["energy"], expected["energy"])
    finally:
        receiving_disk.close()
    joined = _join((initial, first, empty, middle, last))
    _close(joined, expected["memory"])
    prefix, suffix = (
        _read(first_path, _fields(network)),
        _read(last_path, _fields(network)),
    )
    for field in joined:
        _close(torch.cat((prefix[field], suffix[field])), joined[field])


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_checkpoint_recomputation_never_reopens_or_writes_storage_and_preserves_rng(
    tmp_path,
    local_hdf5_writer,  # noqa: F811 - pytest injects the imported fixture
    monkeypatch,
    network,
):
    if network:
        model, _ = _stochastic_network()
        reference, _ = _stochastic_network()
    else:
        model, _, _ = _stochastic_population()
        reference, _, _ = _stochastic_population()
    reference.restore_dict_from_checkpoint(_snapshot(model.state_dict_for_checkpoint()))
    fields = ("left.v", "right.v", "t") if network else ("v", "t")
    path = tmp_path / "checkpoint.h5"
    disk = dn.Recorder(fields, dt=2 * STOCHASTIC_DT).set_hdf5(str(path), cache_every=3)
    functional, _ = dn.func.make_functional(model, dt=STOCHASTIC_DT)
    callbacks = functional.make_callbacks({"disk": disk, "memory": dn.Recorder(fields)})
    reference_functional, _ = dn.func.make_functional(reference, dt=STOCHASTIC_DT)
    reference_callbacks = reference_functional.make_callbacks(
        {"memory": dn.Recorder(fields)}
    )
    finishes = []
    finish = _RecorderHDF5Session.finish

    def record_finish(session):
        finishes.append(session)
        return finish(session)

    monkeypatch.setattr(_RecorderHDF5Session, "finish", record_finish)
    try:
        actual = _execute(model, "longrun_checkpointed", callbacks, dt=STOCHASTIC_DT)
        expected = _execute(
            reference, "longrun_checkpointed", reference_callbacks, dt=STOCHASTIC_DT
        )
        _close(actual["memory"], expected["memory"])
        parameter, reference_parameter = (
            _gradient_leaves(model)[0],
            _gradient_leaves(reference)[0],
        )
        frozen = _snapshot(model.state_dict_for_checkpoint())
        rng = torch.get_rng_state().clone()
        queued, run_number = len(local_hdf5_writer.queue.items), disk.run_number
        partial = actual["memory"][fields[0]][1:4].square().mean()
        objective = actual["memory"][fields[0]].square().mean()
        gradients = (
            torch.autograd.grad(partial, parameter, retain_graph=True)[0],
            torch.autograd.grad(objective, parameter, retain_graph=True)[0],
            torch.autograd.grad(objective, parameter)[0],
        )
        wanted = torch.autograd.grad(
            expected["memory"][fields[0]].square().mean(), reference_parameter
        )[0]
        _close(gradients[1], wanted)
        _close(gradients[2], wanted)
        assert all(
            torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
        )
        _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
        _close(torch.get_rng_state(), rng)
        assert (
            len(local_hdf5_writer.queue.items) == queued
            and disk.run_number == run_number == 1
        )
        assert len(finishes) == 1 and disk._hdf5_session is None
    finally:
        disk.close()
    _close(
        _read(path, fields),
        {name: value[::2] for name, value in expected["memory"].items()},
    )


def test_disk_only_collection_is_bounded_and_cache_intervals_count_solver_steps(
    tmp_path,
    local_hdf5_writer,  # noqa: F811 - pytest injects the imported fixture
    monkeypatch,
):
    from dendra.func._recording import _RecordingCollector

    model = _case(False)
    disk = dn.Recorder(("v", "t"), dt=4 * DT).set_hdf5(
        str(tmp_path / "sparse.h5"), cache_every=3
    )
    callbacks = _collection(model, disk, memory=False)
    functional = callbacks._functional
    tensors = functional.extract()
    _, pure = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        steps=9,
        callbacks=functional.make_callbacks(
            {"trace": dn.Recorder(("v", "t"), dt=4 * DT)}
        ),
    )
    sizes = []
    append = _RecordingCollector.append

    def record_size(collector, *args, **kwargs):
        result = append(collector, *args, **kwargs)
        sizes.append(len(collector.parts))
        return result

    monkeypatch.setattr(_RecordingCollector, "append", record_size)
    try:
        result = _execute(model, "longrun", callbacks, steps=9)
        assert tuple(result) == ("energy",) and result.state["disk"].item() == 9
        assert sizes and max(sizes) == 1
        assert [value.shape[0] for value in _writes(local_hdf5_writer, "v")] == [
            1,
            1,
            1,
        ]
    finally:
        disk.close()
    _close(_read(tmp_path / "sparse.h5", ("v", "t")), pure["callbacks"]["trace"])
    _close(
        _read(tmp_path / "sparse.h5", ("t",))["t"],
        torch.arange(0, 9, 4, dtype=model.t.dtype) * DT,
    )


def test_scoped_packed_population_storage_matches_native_region_recording(tmp_path):
    model, reference = _network(packed=True), _network(packed=True)
    fields = ("v", "i_membrane", "t")
    disk = dn.Recorder(fields, region=model.packed.left.soma, dt=2 * DT).set_hdf5(
        str(tmp_path / "packed.h5"), cache_every=3
    )
    functional, _ = dn.func.make_functional(model, dt=DT)
    callbacks = functional.make_callbacks(
        {"disk": dn.callbacks.PopulationCallback("packed", disk)}
    )
    oracle = dn.Recorder(fields, region=reference.packed.left.soma, dt=2 * DT)
    try:
        observed = _execute(model, "longrun", callbacks)
        assert tuple(observed) == () and observed.state["disk"].item() == 5
        reference.run(
            5 * DT, callbacks=[dn.callbacks.PopulationCallback("packed", oracle)]
        )
    finally:
        disk.close()
    _close(
        _read(tmp_path / "packed.h5", fields),
        {field: oracle.stack(field) for field in fields},
    )


@pytest.mark.parametrize("failure", ("forward", "write", "finish"))
def test_native_storage_releases_sessions_after_forward_write_and_finish_failures(
    tmp_path,
    local_hdf5_writer,  # noqa: F811 - pytest injects the imported fixture
    monkeypatch,
    failure,
):
    class FailedStep(dn.callbacks.Callback):
        def post_step_hook(self, model):
            raise RuntimeError("authored forward failure")

    model = _case(False)
    disk = dn.Recorder(("v", "t")).set_hdf5(
        str(tmp_path / "failure.h5"), cache_every=2 if failure == "write" else 100
    )
    callbacks = _collection(model, disk)
    before = _snapshot(model.state_dict_for_checkpoint())

    def fail(*args, **kwargs):
        raise RuntimeError(f"authored {failure} failure")

    try:
        with monkeypatch.context() as patch:
            if failure == "write":
                patch.setattr(disk, "_write_hdf5_frame", fail)
            elif failure == "finish":
                patch.setattr(_RecorderHDF5Session, "finish", fail)
            with pytest.raises(RuntimeError, match=f"authored {failure} failure"):
                _execute(
                    model,
                    "run",
                    callbacks,
                    steps=1,
                    native=[FailedStep()] if failure == "forward" else None,
                )
        assert disk._hdf5_session is None and disk.save_count == 0
        assert disk.run_number == (1 if failure == "write" else 0)
        model.restore_dict_from_checkpoint(before)
        recovered = _execute(model, "run", callbacks, steps=1)
        assert recovered.state["disk"] == () and "disk" not in recovered
        assert recovered["energy"]["steps"].item() == 1
        assert disk._hdf5_session is None and disk.save_count == 0
        assert disk.run_number == (2 if failure == "write" else 1)
        assert _writes(local_hdf5_writer, "v")[-1].shape[0] == 2
    finally:
        disk.close()


def test_same_recorder_cannot_join_native_hooks_and_pure_storage_in_one_run(tmp_path):
    model = _case(False)
    disk = dn.Recorder(("v", "t")).set_hdf5(str(tmp_path / "owned.h5"))
    callbacks = _collection(model, disk)
    try:
        with pytest.raises(RuntimeError, match="active HDF5"):
            _execute(model, "run", callbacks, native=[disk], steps=1)
        assert disk._hdf5_session is None and not disk._native_hdf5_active
        recovered = _execute(model, "run", callbacks, steps=1)
        assert recovered.state["disk"] == () and "disk" not in recovered
        assert recovered["energy"]["steps"].item() == 1
        assert disk._hdf5_session is None and disk.run_number == 1
    finally:
        disk.close()


@pytest.mark.mps
def test_mps_native_checkpoint_storage_stages_owned_cpu_frames_without_replay_writes(
    tmp_path,
    local_hdf5_writer,  # noqa: F811 - pytest injects the imported fixture
):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DEVICE="mps", DTYPE=torch.float32):
        model = dn.SingleCompartment(
            N=1, C=2, v_init=-63, integrator=dn.bwd_euler_sc(imem=False)
        )
        model.insert(pas, g=0.001, e=-60)
        model.train()
        model.initialize()
        path = tmp_path / "mps.h5"
        disk = dn.Recorder(("v", "t"), dt=2 * DT).set_hdf5(str(path), cache_every=3)
        functional, _ = dn.func.make_functional(model, dt=DT)
        callbacks = functional.make_callbacks(
            {"disk": disk, "memory": dn.Recorder(("v", "t"), dt=2 * DT)}
        )
        try:
            result = _execute(model, "longrun_checkpointed", callbacks)
            queued = len(local_hdf5_writer.queue.items)
            frozen = _snapshot(model.state_dict_for_checkpoint())
            gradient = torch.autograd.grad(
                result["memory"]["v"].square().mean(), model.mech.pas.g_param
            )[0]
            assert gradient.device.type == "mps" and gradient.abs().sum() > 0
            assert len(local_hdf5_writer.queue.items) == queued
            _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
            assert all(
                value.device.type == "cpu" for value in _writes(local_hdf5_writer, "v")
            )
        finally:
            disk.close()
    _close(
        _read(path, ("v", "t")),
        {field: value.cpu() for field, value in result["memory"].items()},
    )
