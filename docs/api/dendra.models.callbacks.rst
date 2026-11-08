Callbacks and recording
=======================

.. py:module:: dendra.callbacks

Pass :class:`dn.Recorder <dendra.models.callbacks.Recorder>` or a callback
from ``dn.callbacks`` to native execution. Recorder, LFP, threshold and
activity callbacks, and Population scopes also work with functional
execution. For custom callbacks, implement
:class:`dendra.func.FunctionalCallback` or derive from
:class:`dn.callbacks.Callback <dendra.models.callbacks.Callback>` for native
hooks. See :doc:`../advanced/recording_and_callbacks` for usage and result shapes.

Recording
---------

.. autoclass:: dendra.models.callbacks.Recorder
   :members: dt, set_partition, set_hdf5, stack, numpy, reset, close
   :undoc-members:

.. autoclass:: dendra.models.callbacks.RecorderLambda
   :members: stack, numpy, reset
   :undoc-members:

.. autoclass:: dendra.models.callbacks.LFP
   :members: lfp, t, numpy, reset
   :undoc-members:

Thresholds and activity
-----------------------

.. autoclass:: dendra.models.callbacks.ThresholdCallback
   :members: dt, reset_timer, reset_count, reset_state_cache, reset, numpy

.. autoclass:: dendra.models.callbacks.APCount
   :members: n

.. autoclass:: dendra.models.callbacks.Raster
   :members: stack, numpy, plot
   :undoc-members:

.. autoclass:: dendra.models.callbacks.Active
   :members: is_active, numpy
   :undoc-members:

.. autoclass:: dendra.models.callbacks.ActiveAL
   :members: is_active, numpy
   :undoc-members:

.. autoclass:: dendra.models.callbacks.ActiveALCount
   :members: is_active, numpy
   :undoc-members:

.. autoclass:: dendra.models.callbacks.AnomalyDetector
   :members: reset, numpy

   Construct with ``dn.callbacks.AnomalyDetector()``.

Network scopes and custom native hooks
--------------------------------------

.. autoclass:: dendra.models.callbacks.PopulationCallback

.. autoclass:: dendra.models.callbacks.Callback
   :members: pre_loop_hook, pre_chunk_hook, pre_step_hook, post_step_hook, post_chunk_hook, post_loop_hook

.. autoclass:: dendra.models.callbacks.CallbackList
