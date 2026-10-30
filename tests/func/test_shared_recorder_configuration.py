"""One Recorder declaration binds pure observation and host-owned persistence."""

from __future__ import annotations

import io
import os
import weakref
from contextlib import nullcontext

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.data import H5Reader
from dendra.models.mod import hh
from tests.func.test_lowering_api import _clone, _close
from tests.test_recorder_h5_integration import local_hdf5_writer as local_hdf5_writer

DT = 0.025
STEPS = 7
FIELDS = ("v", "hh.m", "t")
pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=True),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _population(*, initialize=True):
    model = dn.Population(
        N=1,
        C=3,
        celsius=6.3,
        v_init=-63.0,
        integrator=dn.bwd_euler_sc(imem=True),
    )
    model.insert(hh)
    model[:, [0, 2]].label("ends")
    model.train()
    if initialize:
        model.initialize(DT)
    return model


def _model(network=False):
    if not network:
        return _population()
    model = dn.Network({"cell": _population(initialize=False)})
    model.train()
    model.initialize(DT)
    return model


def _configuration(network, *, native, scoped=False, dt=2 * DT):
    fields = FIELDS if not network or scoped else ("cell.v", "cell.hh.m", "t")
    recorder = (dn.Recorder if native else dn.func.Recorder)(fields, dt=dt)
    callback = dn.func.PopulationCallback("cell", recorder) if scoped else recorder
    return recorder, callback


def _initial(tensors):
    return dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants)


def _read(path, fields, run=0):
    with H5Reader(path) as reader:
        return {
            name: torch.from_numpy(reader[name][run].data.compute()).clone()
            for name in fields
        }


def _join(first, last, name="memory"):
    return {
        field: torch.cat((first["callbacks"][name][field], values))
        for field, values in last["callbacks"][name].items()
    }


def _loss(outputs):
    return sum(values.square().sum() for name, values in outputs.items() if name != "t")


@pytest.mark.parametrize(
    "network,scoped",
    ((False, False), (True, False), (True, True)),
    ids=("population", "qualified_network", "scoped_network"),
)
def test_native_recorder_configuration_binds_population_and_network_state_without_mutation(
    network,
    scoped,
):
    model = _model(network)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    recorder, configuration = _configuration(network, native=True, scoped=scoped)
    _, equivalent = _configuration(network, native=False, scoped=scoped)
    history = recorder.rec
    initial = _initial(tensors)
    source = _clone(functional.extract().state)
    callbacks = functional.make_callbacks({"trace": configuration})
    expected, reference = dn.func.run(
        functional,
        initial,
        steps=STEPS,
        callbacks=functional.make_callbacks({"trace": equivalent}),
    )
    actual, auxiliary = dn.func.run(
        functional, initial, steps=STEPS, callbacks=callbacks
    )
    _close(actual, expected)
    _close(auxiliary["callbacks"]["trace"], reference["callbacks"]["trace"])
    fields = FIELDS if not network or scoped else ("cell.v", "cell.hh.m", "t")
    assert tuple(auxiliary["callbacks"]["trace"]) == fields
    assert auxiliary["callbacks"]["trace"][fields[0]].shape == (4, 1, 3)
    assert auxiliary["callbacks"]["trace"][fields[1]].shape == (4, 1, 3)
    assert auxiliary["callbacks"]["trace"]["t"].shape == (4,)
    assert recorder.rec is history and all(not values for values in history.values())
    assert recorder.i == recorder.run_number == 0
    assert auxiliary["callbacks"].state["trace"].item() == STEPS
    _close(functional.extract().state, source)


def test_canonical_recorder_mechanism_state_matches_native_sampling_and_shapes():
    model = _population()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    pure_configuration = dn.Recorder(FIELDS, dt=2 * DT)
    callbacks = functional.make_callbacks({"trace": pure_configuration})
    _, auxiliary = dn.func.run(
        functional, _initial(tensors), steps=6, callbacks=callbacks
    )
    native = dn.Recorder(FIELDS, dt=2 * DT)
    model.run(tstop=6 * DT, dt=DT, callbacks=[native])
    for field in FIELDS:
        expected = native.stack(field)
        assert expected.shape[0] == 4
        _close(auxiliary["callbacks"]["trace"][field], expected)
    _close(auxiliary["callbacks"]["trace"]["t"], model.t.new_tensor([0, 2, 4, 6]) * DT)
    assert pure_configuration.i == 0
    assert all(not values for values in pure_configuration.rec.values())


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_unsupported_callback_objects_are_rejected_without_reading_storage(network):
    functional, tensors = dn.func.make_functional(_model(network), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    kernel = functional.compile_rollout_chunk(2, backend="eager")

    class Unsupported:
        @property
        def _storage(self):
            raise AssertionError("unsupported callback attributes must not be read")

    invalid = Unsupported()
    for execute in (
        lambda: dn.func.run(
            functional.bind(tensors), tensors.state, steps=2, callbacks=invalid
        ),
        lambda: functional.rollout(
            tensors.parameters, prepared, tensors.state, steps=2, callbacks=invalid
        ),
        lambda: kernel(tensors.parameters, prepared, tensors.state, callbacks=invalid),
    ):
        with pytest.raises(TypeError, match="callbacks must be a FunctionalCallbacks"):
            execute()


def test_binding_snapshots_save_dt_and_selections_without_importing_native_history():
    model = _population()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    indices = torch.tensor([2, 0, 2])
    recorder = dn.Recorder(["v", "hh.m"], node_indices=indices, dt=2 * DT)
    recorder.dt = 7 * DT
    marker = torch.full_like(model.v, 123.0)
    recorder.rec["v"].append(marker)
    recorder.i = 37
    history = recorder.rec
    callbacks = functional.make_callbacks({"trace": recorder})
    reference = functional.make_callbacks(
        {
            "trace": dn.func.Recorder(("v", "hh.m"), node_indices=[2, 0, 2], dt=2 * DT),
        }
    )
    recorder.states = ("t",)
    recorder.save_dt = 4 * DT
    recorder.node_indices.fill_(1)
    _, expected = dn.func.run(
        functional, _initial(tensors), steps=STEPS, callbacks=reference
    )
    _, auxiliary = dn.func.run(
        functional, _initial(tensors), steps=STEPS, callbacks=callbacks
    )
    _close(auxiliary["callbacks"]["trace"], expected["callbacks"]["trace"])
    assert recorder.i == 37 and recorder.rec is history
    assert len(history["v"]) == 1 and history["v"][0] is marker
    assert history["hh.m"] == []


@pytest.mark.parametrize("customization", ("subclass", "hook"))
def test_imperative_recorder_customizations_require_pure_callback_authority(
    customization,
):
    functional, _ = dn.func.make_functional(_population(), dt=DT)
    if customization == "subclass":

        class Custom(dn.Recorder):
            pass

        recorder = Custom(["v"])
    else:
        recorder = dn.Recorder(["v"])
        recorder.post_step_hook = lambda model: None
    with pytest.raises(
        (TypeError, dn.func.FunctionalizationError),
        match="custom|pure|FunctionalCallback|imperative|hook",
    ):
        functional.make_callbacks({"trace": recorder})


def test_explicit_functional_callback_binding_keeps_authority_for_a_native_subclass():
    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    calls = []

    class PureRecorder(dn.Recorder, dn.func.FunctionalCallback):
        def _bind(self, plan):
            calls.append(plan)
            return self

        def initialize(self, state, auxiliary):
            return (), state["integrator"]["v"].sum()

        def update(self, carry, state, auxiliary):
            return carry, state["integrator"]["v"].sum()

        def finalize(self, carry, emissions):
            return emissions

    recorder = PureRecorder(["v"])
    callbacks = functional.make_callbacks({"trace": recorder})
    assert calls == [functional] and callbacks._plans[0].callback is recorder
    assert callbacks._plans[0].validate_updates
    _, auxiliary = dn.func.run(
        functional, _initial(tensors), steps=2, callbacks=callbacks
    )
    assert auxiliary["callbacks"]["trace"].shape == (3,)
    assert recorder.rec == {"v": []} and recorder.i == 0


def test_canonical_recorder_regions_select_physical_fields_and_reject_local_mechanism_slots():
    model = _population()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    callbacks = functional.make_callbacks(
        {
            "all": dn.Recorder(["v", "t"]),
            "ends": dn.Recorder(["v", "t"], region=model.ends),
        }
    )
    _, auxiliary = dn.func.run(
        functional, _initial(tensors), steps=3, callbacks=callbacks
    )
    _close(
        auxiliary["callbacks"]["ends"]["v"],
        auxiliary["callbacks"]["all"]["v"][..., [0, 2]],
    )
    _close(auxiliary["callbacks"]["ends"]["t"], auxiliary["callbacks"]["all"]["t"])
    with pytest.raises(
        dn.func.FunctionalizationError, match="physical|mechanism.local|region"
    ):
        functional.make_callbacks({"local": dn.Recorder(["hh.m"], region=model.ends)})


def test_compiled_canonical_recorder_supports_independent_fresh_backward_graphs():
    recorder = dn.Recorder(FIELDS, dt=2 * DT)
    program = dn.compile(
        _population(),
        dt=DT,
        steps=STEPS,
        chunk_steps=4,
        callbacks={"trace": recorder},
        inference=False,
        trainable="*gnabar_param",
        backend="aot_eager",
    )
    targets = tuple(program.trainable_parameters.values())
    assert targets
    previous = None
    for _ in range(2):
        _, auxiliary = program()
        trace = auxiliary["callbacks"]["trace"]
        gradients = torch.autograd.grad(_loss(trace), targets)
        assert all(
            torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
        )
        if previous is not None:
            _close(gradients, previous)
        previous = tuple(value.detach().clone() for value in gradients)
    assert recorder.i == 0 and all(not values for values in recorder.rec.values())


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_hdf5_streaming_matches_memory_across_compiled_spans_and_continuation_without_replay_writes(
    tmp_path,
    local_hdf5_writer,  # noqa: F811 - pytest injects the imported fixture
    monkeypatch,
    network,
    execution,
):
    fields = ("cell.v", "cell.hh.m", "t") if network else FIELDS
    path = tmp_path / f"{execution}.h5"
    cache_interval = 3 if execution == "scan" else 5
    disk = dn.Recorder(fields, dt=2 * DT).set_hdf5(
        str(path), cache_every=cache_interval
    )
    memory = dn.Recorder(fields, dt=2 * DT)
    frames = []
    write = disk._write_hdf5_frame

    def observe(values):
        frames.append(tuple(weakref.ref(value) for value in values.values()))
        return write(values)

    monkeypatch.setattr(disk, "_write_hdf5_frame", observe)
    program = dn.compile(
        _model(network),
        dt=DT,
        steps=STEPS,
        chunk_steps=4,
        callbacks={"disk": disk, "memory": memory},
        inference=execution == "while",
        execution=execution,
        checkpoint_every=3 if execution == "scan" else None,
        trainable="*gnabar_param" if execution == "scan" else (),
        backend="aot_eager",
    )
    reference_callbacks = program.functional.make_callbacks(
        {
            "memory": dn.func.Recorder(fields, dt=2 * DT),
        }
    )
    initial = dn.func.Continuation(
        program.initial_state, program.parameters, program.constants
    )
    try:
        with torch.no_grad() if execution == "while" else nullcontext():
            expected, reference = dn.func.run(
                program.functional,
                initial,
                steps=STEPS,
                callbacks=reference_callbacks,
            )
            _, first = program(steps=3)
            actual, last = program(first["continuation"], steps=4)
        if execution == "while":
            assert program.execution_report().strategy == "compiled_while_loop"
        _close(actual, expected)
        assert tuple(first["callbacks"]) == tuple(last["callbacks"]) == ("memory",)
        assert first["callbacks"].state["disk"].item() == 3
        assert last["callbacks"].state["disk"].item() == STEPS
        observed = _join(first, last)
        _close(observed, reference["callbacks"]["memory"])
        assert local_hdf5_writer.maxsize > 0
        writes = [
            item for item in local_hdf5_writer.queue.items if isinstance(item, tuple)
        ]
        assert writes and all(0 < item[3].shape[0] <= cache_interval for item in writes)
        assert all(
            item[3].device.type == "cpu"
            and not item[3].requires_grad
            and item[3].grad_fn is None
            for item in writes
        )
        queued = len(local_hdf5_writer.queue.items)
        counters = disk.run_number, disk.save_count
        if execution == "scan":
            targets = tuple(program.trainable_parameters.values())
            gradients = torch.autograd.grad(_loss(observed), targets)
            wanted = torch.autograd.grad(
                _loss(reference["callbacks"]["memory"]), targets
            )
            _close(gradients, wanted)
            assert all(
                torch.isfinite(value).all() and value.abs().sum() > 0
                for value in gradients
            )
        assert len(local_hdf5_writer.queue.items) == queued
        assert (disk.run_number, disk.save_count) == counters
        assert disk.i == memory.i == 0
        assert all(
            not values
            for recorder in (disk, memory)
            for values in recorder.rec.values()
        )
    finally:
        disk.close()
    assert frames and all(
        reference() is None for frame in frames for reference in frame
    )
    first_disk = _read(path, fields, run=0)
    last_disk = _read(path, fields, run=1)
    for field, values in observed.items():
        assert first_disk[field].shape[0] == last_disk[field].shape[0] == 2
        _close(torch.cat((first_disk[field], last_disk[field])), values.detach().cpu())
    assert disk.run_number == 2


def test_portable_recording_continuation_rebinds_a_new_hdf5_destination(tmp_path):
    first_path, last_path = tmp_path / "first.h5", tmp_path / "last.h5"
    disk = dn.Recorder(FIELDS, dt=2 * DT).set_hdf5(str(first_path), cache_every=3)
    program = dn.compile(
        _population(),
        dt=DT,
        steps=STEPS,
        chunk_steps=4,
        backend="eager",
        callbacks={"disk": disk},
    )
    callbacks = program.functional.make_callbacks(
        {"trace": dn.func.Recorder(FIELDS, dt=2 * DT)}
    )
    with torch.no_grad():
        expected, reference = dn.func.run(
            program.functional,
            dn.func.Continuation(
                program.initial_state, program.parameters, program.constants
            ),
            steps=STEPS,
            callbacks=callbacks,
        )
    try:
        _, first = program(steps=3)
        saved = program.export_continuation(first["continuation"])
        buffer = io.BytesIO()
        torch.save(saved, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
    finally:
        disk.close()
    receiving_disk = dn.Recorder(FIELDS, dt=2 * DT).set_hdf5(
        str(last_path), cache_every=2
    )
    receiver = dn.compile(
        _population(),
        dt=DT,
        steps=STEPS,
        chunk_steps=5,
        backend="eager",
        callbacks={"disk": receiving_disk},
    )
    try:
        restored = receiver.restore_continuation(saved)
        actual, last = receiver(restored, steps=4)
        _close(actual, expected)
        assert tuple(last["callbacks"]) == ()
        assert last["callbacks"].state["disk"].item() == STEPS
    finally:
        receiving_disk.close()
    first_values, last_values = _read(first_path, FIELDS), _read(last_path, FIELDS)
    for field in FIELDS:
        _close(
            torch.cat((first_values[field], last_values[field])),
            reference["callbacks"]["trace"][field].cpu(),
        )


def test_closed_hdf5_destination_does_not_silently_become_an_in_memory_recorder(
    tmp_path,
):
    disk = dn.Recorder(FIELDS).set_hdf5(str(tmp_path / "closed.h5"), cache_every=2)
    program = dn.compile(
        _population(),
        dt=DT,
        steps=3,
        callbacks={"disk": disk},
        backend="eager",
    )
    disk.close()
    with pytest.raises(
        (RuntimeError, dn.func.FunctionalizationError),
        match="HDF5|closed|writer|storage",
    ):
        program()


def test_pure_rollout_and_kernel_require_a_host_boundary_for_hdf5_storage(tmp_path):
    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    disk = dn.Recorder(FIELDS).set_hdf5(str(tmp_path / "pure.h5"), cache_every=2)
    callbacks = functional.make_callbacks({"disk": disk})
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    kernel = functional.compile_rollout_chunk(2, backend="eager")
    try:
        for execute in (
            lambda: functional.rollout(
                tensors.parameters,
                prepared,
                tensors.state,
                steps=2,
                callbacks=callbacks,
            ),
            lambda: kernel(
                tensors.parameters, prepared, tensors.state, callbacks=callbacks
            ),
        ):
            with pytest.raises(
                dn.func.FunctionalizationError,
                match="HDF5 recording requires dn.func.run or a dn.compile runner",
            ):
                execute()
        assert disk.run_number == 0 and disk.save_count == 0
    finally:
        disk.close()


def test_hdf5_span_limits_preserve_automatic_noise_sampling_and_rng_continuation(
    tmp_path,
):
    from tests.func.test_runtime_noise import DT as noise_dt
    from tests.func.test_runtime_noise import _population as noisy_population

    functional, tensors = dn.func.make_functional(noisy_population(), dt=noise_dt)
    fields = ("v", "t")
    memory = functional.make_callbacks(
        {"trace": dn.func.Recorder(fields, dt=2 * noise_dt)}
    )
    source = _clone(functional.extract().state)
    global_rng = torch.get_rng_state().clone()
    with torch.no_grad():
        expected, reference = dn.func.run(
            functional,
            _initial(tensors),
            steps=11,
            host_span_steps=7,
            callbacks=memory,
        )
    path = tmp_path / "noise.h5"
    disk = dn.Recorder(fields, dt=2 * noise_dt).set_hdf5(str(path), cache_every=3)
    callbacks = functional.make_callbacks({"disk": disk})
    try:
        with torch.no_grad():
            _, first = dn.func.run(
                functional, _initial(tensors), steps=5, callbacks=callbacks
            )
            _, empty = dn.func.run(
                functional, first["continuation"], steps=0, callbacks=callbacks
            )
            assert disk.run_number == 1
            assert tuple(empty["callbacks"]) == ()
            actual, last = dn.func.run(
                functional, empty["continuation"], steps=6, callbacks=callbacks
            )
        _close(actual, expected)
        assert tuple(first["callbacks"]) == tuple(last["callbacks"]) == ()
        assert last["callbacks"].state["disk"].item() == 11
        _close(functional.extract().state, source)
        torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
    finally:
        disk.close()
    prefix, suffix = _read(path, fields, 0), _read(path, fields, 1)
    for field in fields:
        _close(
            torch.cat((prefix[field], suffix[field])),
            reference["callbacks"]["trace"][field],
        )


@pytest.mark.mps
def test_mps_hdf5_streaming_while_keeps_cpu_owned_samples_and_explicit_carry(tmp_path):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    path = tmp_path / "mps.h5"
    with dn.ctx(DTYPE=torch.float32, DEVICE="mps"):
        disk = dn.Recorder(FIELDS, dt=2 * DT).set_hdf5(str(path), cache_every=5)
        program = dn.compile(
            _population(),
            dt=DT,
            steps=STEPS,
            chunk_steps=4,
            callbacks={"disk": disk, "memory": dn.Recorder(FIELDS, dt=2 * DT)},
            execution="while",
            backend="aot_eager",
        )
        try:
            _, auxiliary = program()
            memory = auxiliary["callbacks"]["memory"]
            assert all(value.device.type == "mps" for value in memory.values())
            assert "disk" not in auxiliary["callbacks"]
            assert auxiliary["callbacks"].state["disk"].item() == STEPS
        finally:
            disk.close()
    for field, values in _read(path, FIELDS).items():
        _close(values, memory[field].cpu())
