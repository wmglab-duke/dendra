dendra.models.networks
======================

Connection-weight units
-----------------------

``NetCon`` weights are numerical values in the coordinate expected by their
target synapse; there is no universal weight unit. The built-in point-process
``expsyn`` and ``exp2syn`` mechanisms add incoming weights to conductance state,
so their weights are bare values in µS: ``weight=0.05`` means 0.05 µS. Do not
multiply those weights by ``dendra.units.uS``.

A density-style continuous target defines its own input unit. For example,
built-in ``graded_syn`` receives a dimensionless release gate into ``g_pre``,
so its connection weight and accumulated input are conductance densities in
S/cm². Custom event targets document the state units expected by the pure
``net_receive_values`` hook, whose updates the ``net_receive`` adapter commits.
Continuous targets document the input units expected by ``continuous_receive``.
See :doc:`../synapse_authoring` for runnable examples, and :doc:`../units` and
:doc:`dendra.models.mechanisms` for the full mechanism contract.

The ``weight`` accepted by ``NetCon.schedule`` and related scheduled-event
methods is different: it is a dimensionless per-event multiplier applied to
the connection's base weight, rather than another target-unit conductance.

Network
-------
.. autoclass:: dendra.models.networks.Network
   :members:

Functional Network lowering
---------------------------

Declare and initialize a canonical single-device Network with the standard
Network API, then lower it for pure execution with explicit parameters and
state. Connections read their registered source fields automatically, and
recurrent delivery follows native execution order. See
:doc:`../advanced/A19_functional_network_lowering` for a runnable example of
pure timesteps, parameter gradients, NetStim draw packets, and checkpoint
continuation.
Rollouts, compiled chunks, per-population input tapes, qualified
recordings, and checkpointed host execution are described in
:doc:`../advanced/A20_functional_network_rollout`.
``dn.compile(Network)`` dispatch, snapshot and fresh initialization, shared
trainable parameter selection, restart, and explicit continuation are described in
:doc:`../advanced/A21_functional_network_compile`.
Compact source-history training with native ``auto`` selection, delivery-time
parameter lookup, and checkpoint continuation is described in
:doc:`../advanced/A22_functional_source_history`.

.. autoclass:: dendra.func.CompiledNetworkRunner
   :members: initial_state, initialization_inputs, parameters, named_parameters, constants, functional, kernel, parameter_name, get_parameter, execution_report
   :special-members: __call__

.. autofunction:: dendra.func.make_functional_network

.. autoclass:: dendra.func.FunctionalNetwork
   :members: initialization_inputs, initialize, sample_initialization_draw, prepare, step, prepare_and_step, make_callbacks, sample_step_draw, sample_draw_tape, rollout, prepare_and_rollout, compile_rollout_chunk, run, extract, commit_state_

.. autoclass:: dendra.func.NetworkTensors
   :members:

.. autoclass:: dendra.func.NetworkRolloutInput
   :members:

.. autoclass:: dendra.func.NetworkInitializationInput
   :members:

.. autoclass:: dendra.func.CompiledNetworkChunk
   :members: steps, capabilities, execution_report
   :special-members: __call__

NetStim
-------
.. autoclass:: dendra.models.networks.NetStim
   :members:

Synapse slot targets
--------------------

Network endpoints can use :class:`dendra.models.slice.SynapseSlots` to address
local slots in banked point-process mechanisms. Its canonical API reference is
:doc:`dendra.models.slice`.

NetCon
------
.. autoclass:: dendra.models.networks.netcon.NetCon
   :members:
   :special-members: __init__

Functional projection tensors
-----------------------------

Lower an initialized NetCon to evaluate event delivery from explicit source
observations, weights, optional physical delays, and state. Declare connectivity
through the standard Network API. See
:doc:`../advanced/A12_functional_event_delivery` for the supported configuration
and complete runnable example. See :doc:`../advanced/A13_functional_threshold_detection`
for finite thresholds and ordered receive-before-observe execution.
Tensor schedule evaluation, differentiable values, and indexed source
bindings are described in :doc:`../advanced/A15_functional_tensor_schedules`.
Two- and three-tap differentiable delays are described in
:doc:`../advanced/A17_functional_delay_interpolation`.
Compact source-history training is described in
:doc:`../advanced/A22_functional_source_history`.

.. autofunction:: dendra.func.make_functional_netcon

.. autoclass:: dendra.func.FunctionalNetCon
   :members: runtime, initial_state, read_due, advance, step, extract, commit_state_

.. autoclass:: dendra.func.NetConTensors
   :members:

Functional continuous projections
---------------------------------

Use a functional ContinuousCon to project source values through zero- or
integer-delay connections. See
:doc:`../advanced/A18_functional_continuous_connectivity` for ordered source
observation, receiver scratch, transform parameters, and runnable continuation.

.. autofunction:: dendra.func.make_functional_continuous_con

.. autoclass:: dendra.func.FunctionalContinuousCon
   :members: read_due, advance, extract, commit_state_

.. autoclass:: dendra.func.ContinuousConTensors
   :members:

Functional NetStim renewal
--------------------------

Lower an initialized NetStim to evaluate pure initialization and renewal with
explicit parameters, random draws, and clock/RNG state.
See :doc:`../advanced/A14_functional_netstim_renewal` for sampling policy,
ordered projection composition, and checkpoint continuation.
Tensor schedules and indexed bindings are described in
:doc:`../advanced/A15_functional_tensor_schedules`.
Consuming heap calendars, explicit cursors, and legacy checkpoint compatibility
are described in :doc:`../advanced/A16_functional_heap_schedules`.

.. autofunction:: dendra.func.make_functional_netstim

.. autoclass:: dendra.func.FunctionalNetStim
   :members: initial_state, step, sample_initialization_draw, sample_step_draw, extract, commit_state_

.. autoclass:: dendra.func.NetStimTensors
   :members:

.. autoclass:: dendra.func.NetStimDraw
   :members:

.. autoclass:: dendra.func.NetStimOutputs
   :members:

.. autoclass:: dendra.func.TensorScheduleTensors
   :members:
