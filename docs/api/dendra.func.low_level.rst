Functional plans and tensor packets
===================================

These APIs expose lowering, preparation, and individual simulation phases.
Most applications can use the runner lifecycle in :doc:`dendra.func`.
Preparation materializes raw sources; stepping consumes prepared values and
explicit state without mutating the native model.

Population and Network plans
----------------------------

.. autofunction:: dendra.func.make_functional

.. autofunction:: dendra.func.make_functional_network

.. autoclass:: dendra.func.FunctionalPopulation
   :members:

.. autoclass:: dendra.func.FunctionalNetwork
   :members:

Bindings and compiled chunks
----------------------------

Bindings hold the parameter and constant sources used for execution. Compiled
chunks execute a fixed step count and can be reused by the host scheduler.

.. autoclass:: dendra.func.BoundPopulation
   :members: functional, parameters, constants, prepared, compiled_chunk, steps
   :special-members: __call__
   :undoc-members:

.. autoclass:: dendra.func.BoundNetwork
   :members: functional, parameters, constants, compiled_chunk, steps
   :special-members: __call__
   :undoc-members:

.. autoclass:: dendra.func.CompiledPopulationChunk
   :members: bind, steps, capabilities, execution_report
   :special-members: __call__
   :undoc-members:

.. autoclass:: dendra.func.CompiledNetworkChunk
   :members: bind, steps, capabilities, execution_report
   :special-members: __call__
   :undoc-members:

Tensor and input packets
------------------------

``Inputs`` and ``RuntimeDraws`` are the common public runtime packets.
The Population- and Network-specific packets below remain supported for
explicit stepping and existing callers.

.. autoclass:: dendra.func.PopulationTensors
   :members:

.. autoclass:: dendra.func.NetworkTensors
   :members:

.. autoclass:: dendra.func.InitializationInput
   :members:

.. autoclass:: dendra.func.NetworkInitializationInput
   :members:

.. autoclass:: dendra.func.NetworkInitializationCache
   :members:

.. autoclass:: dendra.func.StepInput
   :members:

.. autoclass:: dendra.func.RolloutInput
   :members:

.. autoclass:: dendra.func.NetworkRolloutInput
   :members:

.. autoclass:: dendra.func.RandomParameterDraw
   :members:

.. autoclass:: dendra.func.RuntimeNoiseDraw
   :members:

.. autoclass:: dendra.func.StimulationDraw
   :members:

.. autoclass:: dendra.func.ConnectionSamplingDraw
   :members:

Connections and event generators
--------------------------------

These plans expose shared delivery and renewal phases for applications that
assemble their own functional networks. Ordinary Network lowering builds them
from native connection objects.

.. autofunction:: dendra.func.make_functional_netcon

.. autofunction:: dendra.func.make_functional_netstim

.. autofunction:: dendra.func.make_functional_continuous_con

.. autoclass:: dendra.func.FunctionalNetCon
   :members:

.. autoclass:: dendra.func.FunctionalNetStim
   :members:

.. autoclass:: dendra.func.FunctionalContinuousCon
   :members:

.. autoclass:: dendra.func.NetConTensors
   :members:

.. autoclass:: dendra.func.NetStimTensors
   :members:

.. autoclass:: dendra.func.ContinuousConTensors
   :members:

.. autoclass:: dendra.func.NetStimDraw
   :members:

.. autoclass:: dendra.func.NetStimOutputs
   :members:

.. autoclass:: dendra.func.TensorScheduleTensors
   :members:

Stimulation plans
-----------------

.. autoclass:: dendra.func.FunctionalIntra
   :members: shape, batch_shape, device, dtype, parameter_names, constant_names, parameter_schema, constant_schema, plan_token, validate, replace, assemble_tensors, enabled, layout_signature, is_compatible, extract, assemble, assemble_values
   :undoc-members:

.. autoclass:: dendra.func.FunctionalExtra
   :members: shape, batch_shape, device, dtype, parameter_names, constant_names, parameter_schema, constant_schema, plan_token, validate, replace, assemble_tensors, enabled, multicontact, uses_waveforms, extract, assemble, assemble_values
   :undoc-members:

.. autoclass:: dendra.func.StimulusTensors
   :members:
