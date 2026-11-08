Functional execution
====================

.. py:module:: dendra.func

``dn.compile`` creates a reusable Population or Network runner. The
``dn.func`` lifecycle also supports explicit plans and tensor packets for
training, reproducible simulation, and checkpoint continuation. Start with
:doc:`../advanced/functional_execution` for complete workflows; the
:doc:`dendra.func.low_level` reference covers individual plans and packets.

Compilation and execution
-------------------------

Compile an initialized model with either a timestep count (``steps=``) or an
elapsed time in milliseconds (``duration=``). Use the solver timestep for
``dt``. A Network runner uses the timestep set by ``network.initialize(dt)``.

.. code-block:: python

   import dendra as dn

   program = dn.compile(model, dt=0.025, steps=400)
   state, auxiliary = program()  # Same as dn.func.run(program).
   state, auxiliary = dn.func.run(program, auxiliary["continuation"], steps=200)

A call without state starts from the runner's configured initialization.
Pass the continuation to resume with the same simulation state, effective
sources, and callback carry. When callbacks are configured, their results
are in ``auxiliary["callbacks"]``.

.. autofunction:: dendra.func.compile

.. autofunction:: dendra.func.run

.. autofunction:: dendra.func.initialize

.. autofunction:: dendra.func.resolve_steps

.. autofunction:: dendra.func.longrun

.. autofunction:: dendra.func.longrun_checkpointed

.. autoclass:: dendra.func.CompiledRunner
   :members: initial_state, initialization_inputs, parameters, named_parameters, trainable_parameters, find_parameters, parameter_name, get_parameter, constants, functional, callbacks, kernel, execution_report, warm, export_continuation, restore_continuation
   :special-members: __call__

.. autoclass:: dendra.func.CompiledNetworkRunner
   :members: initial_state, initialization_inputs, parameters, named_parameters, trainable_parameters, find_parameters, parameter_name, get_parameter, constants, functional, callbacks, kernel, execution_report, warm, export_continuation, restore_continuation
   :special-members: __call__

Runtime inputs and randomness
-----------------------------

See :doc:`../advanced/inputs_and_randomness` for input shapes, explicit draw
replay, automatic sampling, and variable extracellular stimulation.

.. autoclass:: dendra.func.Inputs
   :members:

.. autoclass:: dendra.func.VarInput
   :members:

   Use ``extra=dn.VarInput()`` when compiling to supply extracellular
   stimulation at call time.

.. autoclass:: dendra.func.RuntimeDraws
   :members:

.. autoclass:: dendra.func.InitializationDraws
   :members:

.. autofunction:: dendra.func.sample_runtime_draws

.. autofunction:: dendra.func.sample_initialization_draws

Continuation and checkpoints
----------------------------

See :doc:`../advanced/initialization_and_checkpoints` for capture, native
handoff, graph detachment, and portable checkpoint examples.

.. autoclass:: dendra.func.Continuation
   :members:

.. autofunction:: dendra.func.capture_continuation

.. autofunction:: dendra.func.detach_continuation

.. autofunction:: dendra.func.commit_continuation_

.. autofunction:: dendra.func.export_continuation

.. autofunction:: dendra.func.restore_continuation

Parameter inspection
--------------------

See :doc:`../advanced/training_and_parameters` for source selection and
optimizer groups.

.. autofunction:: dendra.func.inspect_sources

.. autoclass:: dendra.func.SourceReport
   :members:

.. autoclass:: dendra.func.SourceEntry
   :members:

Callbacks and recording
-----------------------

Native callback configurations such as ``dn.Recorder`` can be prepared for
functional execution. Custom callbacks implement ``FunctionalCallback``.
See :doc:`../advanced/recording_and_callbacks` for cadence, Network scopes,
results, storage, and continuation.

.. autofunction:: dendra.func.make_callbacks

.. autoclass:: dendra.func.FunctionalCallback
   :members:

.. autoclass:: dendra.func.FunctionalCallbacks
   :members:

.. autoclass:: dendra.func.FunctionalCallbackState
   :members:

.. autoclass:: dendra.func.FunctionalCallbackResults
   :members:

.. autoclass:: dendra.func.PopulationCallback
   :members:

.. autoclass:: dendra.func.Recorder
   :members:

.. autoclass:: dendra.func.LFP
   :members:

.. autoclass:: dendra.func.APCount
   :members:

.. autoclass:: dendra.func.Raster
   :members:

.. autoclass:: dendra.func.Active
   :members:

.. autoclass:: dendra.func.ActiveAL
   :members:

.. autoclass:: dendra.func.ActiveALCount
   :members:

.. autoclass:: dendra.func.AnomalyDetector
   :members:

   Construct with ``dn.func.AnomalyDetector()``.

Diagnostics
-----------

.. autoclass:: dendra.func.ExecutionCapabilities
   :members:

.. autoclass:: dendra.func.ExecutionReport
   :members:

.. autoexception:: dendra.func.FunctionalizationError
