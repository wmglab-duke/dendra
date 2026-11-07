HDF5 recordings
===============

Configure writing with :meth:`dendra.models.callbacks.Recorder.set_hdf5` and
close the recorder before opening the file. ``H5Reader`` provides lazy Dask
arrays; compute values while the reader remains open::

    from dendra.data import H5Reader

    with H5Reader("recording.h5") as reader:
        voltage = reader["v"][0][:].compute()

The first index selects a recorded variable, the second selects its run, and
the final index slices the concatenated time axis. See
:doc:`../advanced/recording_and_callbacks` for writing and cadence.

.. autoclass:: dendra.data.H5Reader
   :members: closed, vars, close
   :special-members: __getitem__, __enter__, __exit__
   :undoc-members:

.. autoclass:: dendra.data.h5.H5Var
   :members: n, keys
   :special-members: __getitem__

.. autoclass:: dendra.data.h5.H5Rec
   :special-members: __getitem__
