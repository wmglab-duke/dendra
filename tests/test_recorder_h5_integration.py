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

    def Queue(self):
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
    assert not persisted.data_pinned["v"].is_pinned()

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

    monkeypatch.setattr(callback_module, "TRANSFERSTREAM", None)

    def unexpected_cuda_stream(*args, **kwargs):
        raise AssertionError("CPU HDF5 caching must not enter torch.cuda.stream")

    monkeypatch.setattr(torch.cuda, "stream", unexpected_cuda_stream)
    recorder.cache_hdf5()

    assert recorder.data_pinned["v"].tolist() == [[1.0, 2.0]]
    assert [
        item if isinstance(item, str) else item[:3] for item in recorder.queue.items
    ] == [
        "flush",
        ("v", 0, 0),
    ]
