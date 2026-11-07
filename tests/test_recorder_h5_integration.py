import h5py
import numpy as np
import pytest
import torch

import dendra as dn
from dendra.data import H5Reader
from dendra.models import callbacks as callback_module
from dendra.models.callbacks import Recorder
from dendra.models.mod import pas

DTYPE = torch.float64


class _LocalQueue:
    def __init__(self):
        self.items = []
        self.completed = 0

    def put(self, item):
        self.items.append(item)

    def get(self):
        return self.items.pop(0)

    def task_done(self):
        self.completed += 1


class _LocalManager:
    def __init__(self):
        self.queue = _LocalQueue()
        self.closed = False

    def Queue(self, maxsize=0):
        self.maxsize = maxsize
        return self.queue

    def shutdown(self):
        self.closed = True


class _LocalProcess:
    def __init__(self, target, args):
        self.target = target
        self.args = args
        self.started = False
        self.joined = False

    def start(self):
        self.started = True

    def join(self):
        if not self.joined:
            self.target(*self.args)
            self.joined = True


@pytest.fixture(autouse=True)
def local_hdf5_writer(monkeypatch):
    """Run the real writer synchronously without OS sockets or subprocesses."""
    manager = _LocalManager()
    monkeypatch.setattr(
        callback_module.mp, "set_start_method", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(callback_module.mp, "Manager", lambda: manager)
    monkeypatch.setattr(callback_module.mp, "Process", _LocalProcess)
    return manager


def _population(batch_size=None):
    population = dn.Population(N=2, C=3, v_init=-65.0, dtype=DTYPE)
    population.insert(pas, g=0.001, e=-70.0)
    if batch_size is not None:
        population.batch(batch_size)
    population.build()
    population.initialize()
    return population


def _record_population(path, *, batch_size, n_steps, cache_every):
    population = _population(batch_size)
    expected = Recorder(["v", "t"])
    persisted = Recorder(["v", "t"]).set_hdf5(str(path), cache_every=cache_every)
    try:
        ve = torch.zeros((n_steps, *population.v.shape), dtype=DTYPE)
        population.run(ve=ve, dt=0.01, callbacks=[expected, persisted])
    finally:
        persisted.close()

    expected_values = {
        state: expected.stack(state).detach().cpu().numpy() for state in expected.states
    }
    return persisted, expected_values


@pytest.mark.parametrize(
    "batch_size,n_steps,cache_every,expected_chunk_lengths",
    [
        pytest.param(None, 3, 2, [2, 2], id="ordinary-exact-boundary"),
        pytest.param(2, 4, 3, [3, 2], id="batched-partial-final-chunk"),
    ],
)
def test_population_recorder_hdf5_round_trip_across_ranks_and_boundaries(
    tmp_path,
    batch_size,
    n_steps,
    cache_every,
    expected_chunk_lengths,
):
    path = tmp_path / "population.h5"
    persisted, expected = _record_population(
        path,
        batch_size=batch_size,
        n_steps=n_steps,
        cache_every=cache_every,
    )

    assert persisted.run_number == 1
    assert persisted.save_count == 0
    assert not persisted.cache_with_hdf5
    buffered_v = persisted.data_pinned["v"]
    assert buffered_v.device.type == "cpu"
    assert not buffered_v.is_pinned()

    with H5Reader(path) as reader:
        assert reader.vars() == ["t", "v"]
        for state, values in expected.items():
            record = reader[state][0]
            assert record.data.ndim == values.ndim
            np.testing.assert_allclose(record.data.compute(), values)

            datasets = [record.group[name] for name in record.group]
            assert [dataset.shape[0] for dataset in datasets] == (
                expected_chunk_lengths
            )
            assert all(dataset.chunks is not None for dataset in datasets)
            assert all(dataset.ndim == len(dataset.chunks) for dataset in datasets)
            if state == "v":
                assert all(dataset.chunks[1] == 1 for dataset in datasets)

    with h5py.File(path, "r") as file:
        # An exact cache boundary leaves Recorder.rec empty. post_loop_hook must
        # not materialize an extra empty dataset for that final buffer.
        assert list(file["v/run_0"]) == [
            str(i) for i in range(len(expected_chunk_lengths))
        ]


@pytest.mark.parametrize("cache_every", [True, 0, -1, 1.5])
def test_recorder_validates_cache_cadence_before_starting_writer(tmp_path, cache_every):
    recorder = Recorder(["v"])
    error = TypeError if isinstance(cache_every, (bool, float)) else ValueError
    with pytest.raises(error, match="positive integer"):
        recorder.set_hdf5(str(tmp_path / "invalid.h5"), cache_every=cache_every)
    assert not recorder.cache_with_hdf5


def test_recorder_reuse_writes_independent_runs_without_stale_or_overwritten_data(
    tmp_path,
):
    path = tmp_path / "two_runs.h5"
    population = _population()
    persisted = Recorder(["v"]).set_hdf5(str(path), cache_every=100)
    expected_runs = []
    try:
        for _ in range(2):
            expected = Recorder(["v"])
            population.run(
                ve=torch.zeros((2, *population.v.shape), dtype=DTYPE),
                dt=0.01,
                callbacks=[expected, persisted],
            )
            expected_runs.append(expected.stack("v").detach().cpu().numpy())
    finally:
        persisted.close()

    with H5Reader(path) as reader:
        assert reader["v"].n() == 2
        for run, expected in enumerate(expected_runs):
            np.testing.assert_allclose(reader["v"][run].data.compute(), expected)

    assert persisted.manager.closed
    persisted.close()


def test_cache_rejects_partial_state_frames():
    recorder = Recorder(["v", "t"])
    recorder.cache_with_hdf5 = True
    recorder.rec["v"].append(torch.tensor(1.0))
    with pytest.raises(RuntimeError, match="partial Recorder frame.*t"):
        recorder.cache_hdf5()


def test_cpu_cache_never_enters_a_cuda_stream(monkeypatch):
    recorder = Recorder(["v"])
    recorder.queue = _LocalQueue()
    recorder.rec["v"].append(torch.tensor([1.0, 2.0]))

    # CUDA availability is a property of the host, not necessarily of the
    # recorded data. A CPU recording must stay on the ordinary CPU path even
    # when the process owns a CUDA transfer stream.
    monkeypatch.setattr(callback_module, "TRANSFERSTREAM", object())

    def unexpected_cuda_stream(*args, **kwargs):
        raise AssertionError("CPU HDF5 caching must not enter torch.cuda.stream")

    monkeypatch.setattr(torch.cuda, "stream", unexpected_cuda_stream)
    recorder.cache_hdf5()

    assert recorder.data_pinned["v"].tolist() == [[1.0, 2.0]]
    assert not recorder.data_pinned["v"].is_pinned()
    assert [
        item if isinstance(item, str) else item[:3] for item in recorder.queue.items
    ] == [
        "flush",
        ("v", 0, 0),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_cache_uses_pinned_cpu_buffer_and_transfer_stream():
    recorder = Recorder(["v"])
    recorder.queue = _LocalQueue()
    recorder.rec["v"].append(torch.tensor([1.0, 2.0], device="cuda"))

    recorder.cache_hdf5()

    buffered = recorder.data_pinned["v"]
    assert buffered.device.type == "cpu"
    assert buffered.is_pinned()
    assert buffered.tolist() == [[1.0, 2.0]]
    assert all(stream.query() for stream in recorder._transfer_streams.values())


def test_hdf5_writer_flush_never_touches_cuda(tmp_path, monkeypatch):
    class UnexpectedTransferStream:
        def synchronize(self):
            raise AssertionError("the HDF5 writer process must not synchronize CUDA")

    queue = _LocalQueue()
    queue.put("flush")
    queue.put(None)
    monkeypatch.setattr(callback_module, "TRANSFERSTREAM", UnexpectedTransferStream())

    callback_module._hdf5_write(queue, str(tmp_path / "flush-only.h5"))

    assert queue.completed == 2


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multiple CUDA GPUs required")
def test_cuda_cache_uses_a_transfer_stream_on_each_source_device():
    recorder = Recorder(["v"])
    recorder.queue = _LocalQueue()

    for device_index in (0, 1):
        recorder.rec["v"] = [
            torch.tensor([float(device_index)], device=f"cuda:{device_index}")
        ]
        recorder.cache_hdf5()

    assert set(recorder._transfer_streams) == {0, 1}
    assert {stream.device.index for stream in recorder._transfer_streams.values()} == {
        0,
        1,
    }


def _stream_values(rows):
    values = torch.tensor(rows, dtype=DTYPE, requires_grad=True)
    return {
        "v": values[:, None, None].expand(-1, 2, 3),
        "hh.m": values[:, None, None].expand(-1, 1, 4),
        "t": values,
    }


def test_shared_hdf5_session_splits_sparse_rows_at_simulation_cache_boundaries(
    tmp_path, local_hdf5_writer
):
    from dendra.models._recorder_storage import _RecorderHDF5Session

    path = tmp_path / "mechanism-stream.h5"
    recorder = Recorder(["v", "hh.m", "t"]).set_hdf5(str(path), cache_every=3)
    marker = torch.tensor(37.0)
    recorder.rec["v"].append(marker)
    recorder.i = 19
    session = _RecorderHDF5Session(recorder)
    initial = _stream_values([0.0])
    session.append(initial, steps=0, positions=(0,))
    # Pending chunks own their samples, independently of subsequent mutation.
    with torch.no_grad():
        initial["t"].fill_(100.0)
    session.append(_stream_values([2.0, 4.0]), steps=5, positions=(2, 4))
    session.append(_stream_values([6.0, 8.0]), steps=3, positions=(1, 3))
    session.finish()
    assert recorder.i == 19 and recorder.rec["v"][0] is marker
    assert recorder.run_number == 1 and recorder.save_count == 0
    assert recorder._hdf5_session is None and session._pending == {}
    assert local_hdf5_writer.maxsize == 2
    assert all(not value.requires_grad for value in recorder.data_pinned.values())
    recorder.close()

    with H5Reader(path) as reader:
        for state in recorder.states:
            record = reader[state][0]
            expected = _stream_values([0.0, 2.0, 4.0, 6.0, 8.0])[state]
            np.testing.assert_array_equal(
                record.data.compute(), expected.detach().numpy()
            )
            assert [record.group[name].shape[0] for name in record.group] == [2, 1, 2]
    with pytest.raises(RuntimeError, match="session is closed"):
        session.append(_stream_values([]), steps=0, positions=())


def test_shared_hdf5_empty_spans_and_abort_do_not_reuse_written_run_identity(tmp_path):
    from dendra.models._recorder_storage import _RecorderHDF5Session

    path = tmp_path / "abort-stream.h5"
    recorder = Recorder(["v", "hh.m", "t"]).set_hdf5(str(path), cache_every=2)
    pending = _RecorderHDF5Session(recorder)
    pending.append(_stream_values([99.0]), steps=0, positions=(0,))
    pending.abort()
    assert recorder.run_number == 0

    empty = _RecorderHDF5Session(recorder)
    empty.append(_stream_values([]), steps=3, positions=())
    empty.finish()
    assert recorder.run_number == 0

    first = _RecorderHDF5Session(recorder)
    first.append(_stream_values([0.0]), steps=0, positions=(0,))
    first.append(_stream_values([]), steps=1, positions=())
    first.abort()
    assert recorder.run_number == 1

    second = _RecorderHDF5Session(recorder)
    second.append(_stream_values([2.0]), steps=1, positions=(1,))
    second.finish()
    assert recorder.run_number == 2
    recorder.close()

    with H5Reader(path) as reader:
        np.testing.assert_array_equal(reader["t"][0].data.compute(), [0.0])
        np.testing.assert_array_equal(reader["t"][1].data.compute(), [2.0])


def test_shared_hdf5_session_rejects_overlapping_and_changed_storage(tmp_path):
    from types import SimpleNamespace

    from dendra.models._recorder_storage import _RecorderHDF5Session

    recorder = Recorder(["v", "hh.m", "t"]).set_hdf5(str(tmp_path / "owned.h5"))
    session = _RecorderHDF5Session(recorder)
    for operation in (
        lambda: _RecorderHDF5Session(recorder),
        lambda: recorder.pre_loop_hook(SimpleNamespace()),
        recorder.close,
        recorder.reset,
    ):
        with pytest.raises(RuntimeError, match="active HDF5"):
            operation()
    original = recorder.hdf5_path
    recorder.hdf5_path = "changed.h5"
    with pytest.raises(RuntimeError, match="changed during execution"):
        session.append(_stream_values([]), steps=0, positions=())
    recorder.hdf5_path = original
    session.abort()
    recorder.close()


def test_shared_hdf5_session_failed_enqueue_reserves_run_and_cleans_up(
    tmp_path, monkeypatch
):
    from dendra.models._recorder_storage import _RecorderHDF5Session

    recorder = Recorder(["v", "hh.m", "t"]).set_hdf5(str(tmp_path / "failed.h5"))
    session = _RecorderHDF5Session(recorder)
    session.append(_stream_values([0.0]), steps=0, positions=(0,))
    original = recorder._write_hdf5_frame
    failure = RuntimeError("failed storage")

    def failed(values):
        raise failure

    monkeypatch.setattr(recorder, "_write_hdf5_frame", failed)
    with pytest.raises(RuntimeError, match="failed storage") as caught:
        session.finish()
    assert caught.value is failure
    assert recorder._hdf5_session is None and recorder.run_number == 1
    assert session._pending == {}
    monkeypatch.setattr(recorder, "_write_hdf5_frame", original)
    recorder.close()


def test_hdf5_producer_detects_dead_writer_instead_of_blocking(
    tmp_path, local_hdf5_writer
):
    from types import SimpleNamespace

    recorder = Recorder("v").set_hdf5(str(tmp_path / "dead.h5"))
    original = recorder.writer_thread
    recorder.writer_thread = SimpleNamespace(exitcode=7, join=lambda: None)
    with pytest.raises(RuntimeError, match="exit code 7"):
        recorder._write_hdf5_frame({"v": torch.ones(1, 2)})
    with pytest.raises(RuntimeError, match="exit code 7"):
        recorder.close()
    assert not recorder.cache_with_hdf5 and local_hdf5_writer.closed
    assert recorder.states == ("v",)
    # The local fixture has no real child process to reap.
    assert original.started


def test_native_region_recording_preserves_scalar_clock_and_per_run_sampling():
    population = _population()
    full = Recorder(["v", "t"], dt=0.02)
    region = population[:, [2, 0, 2]]
    selected = Recorder(["v", "t"], dt=0.02, region=region)
    for _ in range(2):
        population.run(
            ve=torch.zeros((3, *population.v.shape), dtype=DTYPE),
            dt=0.01,
            callbacks=[full, selected],
        )
        assert full.i == selected.i == 0
    torch.testing.assert_close(selected.stack("v"), full.stack("v")[..., [2, 0, 2]])
    torch.testing.assert_close(selected.stack("t"), full.stack("t"))
    assert selected.stack("v").shape[0] == 4
    assert not full._modules and not selected._modules


@pytest.mark.parametrize(
    "option", ("foreign", "mechanism_state", "node_indices", "maximum", "partition")
)
def test_native_region_recording_rejects_unsupported_combinations(option):
    population = _population()
    region = (_population() if option == "foreign" else population)[:, :]
    states = ["pas.i_"] if option == "mechanism_state" else ["v"]
    options = {
        "foreign": {},
        "mechanism_state": {},
        "node_indices": {"node_indices": [0]},
        "maximum": {"max_only": True},
        "partition": {"partition": [3]},
    }[option]
    recorder = Recorder(states, region=region, **options)
    with pytest.raises(ValueError, match="Recorder region"):
        recorder.pre_loop_hook(population)
    assert recorder.rec == {state: [] for state in states}
