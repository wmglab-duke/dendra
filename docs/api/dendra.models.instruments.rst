.. _dendra.models.instruments:

Activation threshold searches
=============================

Use :class:`~dendra.models.instruments.thresholder.Thresholder` to find the
amplitude that activates each cell. With a Population, activity callback,
spatial voltage and waveform, call
:meth:`~dendra.models.instruments.thresholder.Thresholder.calculate_thresholds`
to obtain the upper and lower bounds.

.. code-block:: python

   from dendra.models.instruments import Thresholder

   search = Thresholder(
       model, active, space=spatial_voltage, time=waveform, ub=1.0, atol=0.001,
   )
   upper, lower = search.calculate_thresholds(tstop=10.0, dt=0.025)

For a compiled Population runner, pass the name of its activity callback.
See :doc:`../advanced/functional_execution` and the runnable
:download:`thresholding example <../../examples/training/functional_thresholding.py>`.

.. autoclass:: dendra.models.instruments.thresholder.Thresholder
   :members:
