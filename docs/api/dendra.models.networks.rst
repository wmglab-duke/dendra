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

NetStim
-------
.. autoclass:: dendra.models.networks.NetStim
   :members:

Synapse slot targets
--------------------

Network endpoints can use :class:`dendra.models.slice.SynapseSlots` to address
local slots in banked point-process mechanisms. Its canonical API reference is
:doc:`dendra.models.slice`.

NetCon internals
----------------
.. autoclass:: dendra.models.networks.netcon.NetCon
   :members:
   :special-members: __init__

Functional projection tensors
-----------------------------

The shared detector and dense transport core can be used with explicit
source observations, weights, optional physical delays, and carry. This lowers an existing initialized
NetCon; the standard Network remains the connectivity declaration. See
:doc:`../advanced/A12_functional_event_delivery` for the supported configuration
and complete runnable example. See :doc:`../advanced/A13_functional_threshold_detection`
for finite thresholds and ordered receive-before-observe execution.
Shared tensor schedule evaluation, differentiable values, and indexed source
bindings are described in :doc:`../advanced/A15_functional_tensor_schedules`.
Shared two- and three-tap differentiable delays are described in
:doc:`../advanced/A17_functional_delay_interpolation`.

.. autofunction:: dendra.func.make_functional_netcon

.. autoclass:: dendra.func.FunctionalNetCon
   :members: read_due, advance, step, extract, commit_state_

.. autoclass:: dendra.func.NetConTensors
   :members:

Functional NetStim renewal
--------------------------

Initialized NetStims expose the shared pure initialization and
renewal core through explicit parameters, random draws, and clock/RNG carry.
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
