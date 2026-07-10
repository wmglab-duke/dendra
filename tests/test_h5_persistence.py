import gc

import dask.array as da
import h5py
import numpy as np
import pytest
import torch

from dendra.data import H5Reader
from dendra.models.callbacks import _hdf5_write


def _add_run(file, variable, run, chunks):
    group = file.require_group(f"{variable}/run_{run}")
    for name, value in chunks.items():
        group.create_dataset(str(name), data=value)
    return group


class _SynchronousQueue:
    """Small queue stand-in for deterministic coverage of the HDF5 writer."""

    def __init__(self, items):
        self.items = iter(items)
        self.completed = 0

    def get(self):
        return next(self.items)

    def task_done(self):
        self.completed += 1


def test_writer_reader_round_trip_runs_chunks_metadata_and_tensor_dtype(tmp_path):
    path = tmp_path / "recordings.h5"
    first = torch.arange(24, dtype=torch.float64).reshape(2, 1, 3, 4)
    second = first.add(100)
    other_run = torch.full((1, 1, 3, 4), -7, dtype=torch.float64)
    queue = _SynchronousQueue(
        [
            ("voltage", 0, 0, first, first.shape),
            "flush",
            ("voltage", 0, 1, second, second.shape),
            ("voltage", 1, 0, other_run, other_run.shape),
            None,
        ]
    )

    _hdf5_write(queue, str(path))

    assert queue.completed == 5
    with h5py.File(path, "a") as file:
        file["voltage"].attrs["units"] = "mV"
        file["voltage/run_0"].attrs["dt"] = 0.025

    with H5Reader(path) as reader:
        assert reader.vars() == ["voltage"]
        variable = reader["voltage"]
        assert variable.n() == 2
        assert variable.keys() == ["run_0", "run_1"]
        assert variable.group.attrs["units"] == "mV"

        record = variable[0]
        assert isinstance(record.data, da.Array)
        assert record.data.shape == (4, 1, 3, 4)
        assert record.data.dtype == np.dtype(np.float64)
        assert record.group.attrs["dt"] == pytest.approx(0.025)
        np.testing.assert_array_equal(
            record.data.compute(),
            np.concatenate((first.numpy(), second.numpy()), axis=0),
        )
        np.testing.assert_array_equal(record[2:].compute(), second.numpy())
        np.testing.assert_array_equal(variable[1].data.compute(), other_run.numpy())

        assert repr(reader) == f"H5Reader {{ {path} }}"
        assert repr(variable) == f"H5Reader {{ {path} }} {{ voltage }}"
        assert repr(record) == f"H5Reader {{ {path} }} {{ voltage }} {{ run 0 }}"


def test_writer_open_mode_overwrites_an_existing_recording(tmp_path):
    path = tmp_path / "overwrite.h5"
    old = np.ones((1, 1, 1, 1), dtype=np.float32)
    new = np.full((1, 1, 1, 1), 2, dtype=np.int16)

    first_queue = _SynchronousQueue([("old", 0, 0, old, old.shape), None])
    _hdf5_write(first_queue, str(path))
    second_queue = _SynchronousQueue([("new", 0, 0, new, new.shape), None])
    _hdf5_write(second_queue, str(path))

    with H5Reader(path) as reader:
        assert reader.vars() == ["new"]
        result = reader["new"][0].data.compute()
        assert result.dtype == np.dtype(np.int16)
        np.testing.assert_array_equal(result, new)


def test_natural_ordering_applies_to_variables_runs_and_chunks(tmp_path):
    path = tmp_path / "natural.h5"
    with h5py.File(path, "w") as file:
        for variable in ("state_10", "state_2"):
            for run in (10, 2, 0):
                _add_run(
                    file,
                    variable,
                    run,
                    {
                        10: np.full((1,), 10, dtype=np.int8),
                        2: np.full((1,), 2, dtype=np.int8),
                        0: np.full((1,), 0, dtype=np.int8),
                    },
                )

    with H5Reader(path) as reader:
        assert reader.vars() == ["state_2", "state_10"]
        variable = reader["state_2"]
        assert variable.keys() == ["run_0", "run_2", "run_10"]
        np.testing.assert_array_equal(variable[0].data.compute(), [0, 2, 10])


@pytest.mark.parametrize(
    "shape",
    [
        (3,),
        (3, 2),
        (3, 2, 4),
        (3, 2, 4, 1),
        (3, 2, 4, 1, 2),
    ],
)
def test_reader_supports_recordings_of_any_non_scalar_rank(tmp_path, shape):
    path = tmp_path / f"rank_{len(shape)}.h5"
    first = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    second = first + 1000
    with h5py.File(path, "w") as file:
        _add_run(file, "value", 0, {0: first, 1: second})

    with H5Reader(path) as reader:
        result = reader["value"][0].data
        assert result.shape == (shape[0] * 2, *shape[1:])
        assert result.dtype == first.dtype
        np.testing.assert_array_equal(
            result.compute(), np.concatenate((first, second), axis=0)
        )


def test_lazy_record_keeps_its_reader_alive(tmp_path):
    path = tmp_path / "lifetime.h5"
    expected = np.arange(6).reshape(2, 3)
    with h5py.File(path, "w") as file:
        _add_run(file, "value", 0, {0: expected})

    record = H5Reader(path)["value"][0]
    gc.collect()

    assert not record.reader.closed
    np.testing.assert_array_equal(record.data.compute(), expected)
    record.reader.close()


def test_context_manager_closes_handles_and_close_is_idempotent(tmp_path):
    path = tmp_path / "close.h5"
    with h5py.File(path, "w") as file:
        _add_run(file, "value", 0, {0: np.arange(3)})

    with H5Reader(path) as reader:
        variable = reader["value"]
        record = variable[0]
        assert not reader.closed

    assert reader.closed
    reader.close()
    with pytest.raises(ValueError, match="closed"):
        reader.vars()
    with pytest.raises(ValueError, match="closed"):
        reader["value"]
    with pytest.raises(ValueError, match="closed"):
        variable.keys()
    with pytest.raises(ValueError, match="closed"):
        record[:]


def test_missing_file_variable_and_run_have_stable_failures(tmp_path):
    missing = tmp_path / "missing.h5"
    with pytest.raises((FileNotFoundError, OSError)):
        H5Reader(missing)

    path = tmp_path / "paths.h5"
    with h5py.File(path, "w") as file:
        _add_run(file, "value", 0, {0: np.arange(2)})

    with H5Reader(path) as reader:
        with pytest.raises(KeyError):
            reader["missing"]
        with pytest.raises(KeyError):
            reader["value"][2]


@pytest.mark.parametrize("index", ["0", 0.0, None, True])
def test_run_indices_must_be_integers(tmp_path, index):
    path = tmp_path / "indices.h5"
    with h5py.File(path, "w") as file:
        _add_run(file, "value", 0, {0: np.arange(2)})

    with H5Reader(path) as reader:
        with pytest.raises(TypeError, match="non-negative integer"):
            reader["value"][index]
        with pytest.raises(IndexError, match="non-negative"):
            reader["value"][-1]


def test_numpy_integer_run_index_is_supported(tmp_path):
    path = tmp_path / "numpy_index.h5"
    expected = np.arange(2)
    with h5py.File(path, "w") as file:
        _add_run(file, "value", 0, {0: expected})

    with H5Reader(path) as reader:
        np.testing.assert_array_equal(
            reader["value"][np.int64(0)].data.compute(), expected
        )


def test_top_level_variables_and_runs_must_be_groups(tmp_path):
    path = tmp_path / "groups.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("dataset_variable", data=np.arange(2))
        file.create_dataset("group_variable/run_0", data=np.arange(2))

    with H5Reader(path) as reader:
        with pytest.raises(TypeError, match="variable.*must be a group"):
            reader["dataset_variable"]
        with pytest.raises(TypeError, match="run 'run_0'.*must be a group"):
            reader["group_variable"][0]


def test_empty_run_is_rejected_as_a_malformed_schema(tmp_path):
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as file:
        file.create_group("value/run_0")

    with H5Reader(path) as reader:
        with pytest.raises(ValueError, match="contains no data chunks"):
            reader["value"][0]


def test_run_chunks_must_be_datasets_with_at_least_one_dimension(tmp_path):
    subgroup_path = tmp_path / "subgroup.h5"
    with h5py.File(subgroup_path, "w") as file:
        file.create_group("value/run_0/0")
    with H5Reader(subgroup_path) as reader:
        with pytest.raises(TypeError, match="chunk '0'.*must be a dataset"):
            reader["value"][0]

    scalar_path = tmp_path / "scalar.h5"
    with h5py.File(scalar_path, "w") as file:
        file.create_dataset("value/run_0/0", data=np.array(3.0))
    with H5Reader(scalar_path) as reader:
        with pytest.raises(ValueError, match="at least one dimension"):
            reader["value"][0]


def test_run_chunks_must_agree_outside_the_concatenation_axis(tmp_path):
    path = tmp_path / "shape_mismatch.h5"
    with h5py.File(path, "w") as file:
        _add_run(
            file,
            "value",
            0,
            {0: np.zeros((2, 3)), 1: np.zeros((2, 4))},
        )

    with H5Reader(path) as reader:
        with pytest.raises(ValueError, match="trailing shape.*expected"):
            reader["value"][0]
