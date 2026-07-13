import operator

import dask.array as da
import h5py
from natsort import natsorted


class H5Reader:
    def __init__(self, path):
        self.path = path
        # Assign before opening so finalization is safe when h5py raises while
        # constructing a reader (for example, for a missing or invalid file).
        self.file = None
        self.file = h5py.File(path, "r")

    @property
    def closed(self):
        """Whether the underlying HDF5 file handle is closed."""
        return self.file is None or not self.file.id.valid

    def _ensure_open(self):
        if self.closed:
            raise ValueError("H5Reader is closed.")

    def __getitem__(self, key):
        self._ensure_open()
        value = self.file[key]
        if not isinstance(value, h5py.Group):
            raise TypeError(
                f"HDF5 variable {key!r} must be a group, got {type(value).__name__}."
            )
        return H5Var(value, self, key)

    def close(self):
        """Close the underlying HDF5 file handle; repeated calls are safe."""
        file = getattr(self, "file", None)
        if file is not None and file.id.valid:
            file.close()

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        self.close()

    def vars(self):
        self._ensure_open()
        return natsorted(self.file.keys())

    def __repr__(self):
        return f"H5Reader {{ {self.path} }}"


class H5Var:
    def __init__(self, group, reader, name):
        self.group = group
        self.reader = reader
        self.name = name

    def __getitem__(self, key: int):
        self.reader._ensure_open()
        if isinstance(key, bool):
            raise TypeError(f"Run index must be a non-negative integer, got {key!r}.")
        try:
            key = operator.index(key)
        except TypeError as error:
            raise TypeError(
                f"Run index must be a non-negative integer, got {key!r}."
            ) from error
        if key < 0:
            raise IndexError(f"Run index must be non-negative, got {key}.")
        value = self.group[f"run_{key}"]
        if not isinstance(value, h5py.Group):
            raise TypeError(
                f"HDF5 run 'run_{key}' must be a group, got {type(value).__name__}."
            )
        return H5Rec(value, self.reader, self.name, key)

    def n(self):
        return len(self.keys())

    def keys(self):
        self.reader._ensure_open()
        return natsorted(self.group.keys())

    def __repr__(self):
        return f"{self.reader} {{ {self.name} }}"


class H5Rec:
    def __init__(self, group, reader, name, run):
        self.group = group
        self.reader = reader
        self.name = name
        self.run = run
        names = natsorted(self.group.keys())
        if not names:
            raise ValueError(
                f"HDF5 variable {name!r}, run {run}, contains no data chunks."
            )

        datasets = []
        for dataset_name in names:
            value = self.group[dataset_name]
            if not isinstance(value, h5py.Dataset):
                raise TypeError(
                    f"HDF5 chunk {dataset_name!r} in variable {name!r}, run {run}, "
                    f"must be a dataset, got {type(value).__name__}."
                )
            if value.ndim == 0:
                raise ValueError(
                    f"HDF5 chunk {dataset_name!r} in variable {name!r}, run {run}, "
                    "must have at least one dimension."
                )
            datasets.append(value)

        trailing_shape = datasets[0].shape[1:]
        for dataset_name, dataset in zip(names[1:], datasets[1:]):
            if dataset.shape[1:] != trailing_shape:
                raise ValueError(
                    f"HDF5 chunk {dataset_name!r} in variable {name!r}, run {run}, "
                    f"has trailing shape {dataset.shape[1:]}, expected {trailing_shape}."
                )

        # Recorder output is commonly four-dimensional and historically used
        # (-1, 1, -1, -1).  Construct the same pattern from the actual rank so
        # indexed/max-only recordings and simple one-dimensional traces remain
        # readable as well.
        arrays = []
        for dataset in datasets:
            chunks = tuple(1 if axis == 1 else -1 for axis in range(dataset.ndim))
            arrays.append(da.from_array(dataset, chunks=chunks))
        self.data = da.concatenate(arrays, axis=0)

    def __getitem__(self, key):
        self.reader._ensure_open()
        return self.data[key]

    def __repr__(self):
        return f"{self.reader} {{ {self.name} }} {{ run {self.run} }}"
