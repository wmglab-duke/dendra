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
Network API, then lower it with ``dn.func.make_functional(network, dt=DT)`` for
pure execution with explicit parameters and state. The supplied timestep must
match the initialized Network. ``make_functional_network(network, ...)`` remains
available with its existing implicit-timestep signature. Connections read their
registered source fields automatically, and
recurrent delivery follows native execution order. See
:doc:`../advanced/A19_functional_network_lowering` for a runnable example of
pure timesteps, parameter gradients, NetStim draw packets, and checkpoint
continuation.
Bind the returned bundle with ``plan.bind(tensors)`` and supply explicit state
to ``dn.func.run``. For fresh initialization, ``dn.func.initialize(plan, tensors)``
returns effective sources with state in a shared continuation; bind and run
that packet explicitly. Deterministic inputs use mappings from Population names
to ``dn.func.Inputs``. Supply the ``RuntimeDraws`` from
``plan.sample_runtime_draws`` separately through ``draws=``, or use automatic
host sampling. See :ref:`functional-runtime-inputs` and
:ref:`functional-runtime-draws` for the shared Population/Network contracts.
Rollouts, compiled chunks, per-population input tapes, qualified
recordings, and checkpointed host execution are described in
:doc:`../advanced/A20_functional_network_rollout`.
``dn.compile(Network)`` dispatch, snapshot and fresh initialization, shared
trainable parameter selection, restart, and explicit continuation are described in
:doc:`../advanced/A21_functional_network_compile`.
Inference-only ``execution="while"`` reuses Population's packed loop and
recording engine; see :doc:`../advanced/A25_functional_network_while`.
Qualified physical Recorders accept Population Slice regions, including named
components after ``Network.concat()``; see
:doc:`../advanced/A26_functional_network_regions`.
``PopulationCallback("left", callback)`` applies an existing functional
Population callback to one named Population. LFP, threshold-analysis, anomaly,
scoped Recorder callbacks, and custom pure callbacks support scan and
inference-only while execution. See
:doc:`../advanced/A27_functional_network_callbacks` for shapes, callback windows,
lead-field gradients, packed components, and continuation.
``trainable`` accepts exact/unique partial names, glob patterns, and explicit
regex selectors. Preview matches with ``find_parameters`` and inspect the live
selected leaves through ``trainable_parameters``; shared aliases remain visible.
``dn.func.inspect_sources(runner)`` reports default parameter/constant sources;
pass a returned ``Continuation`` for the effective sources of a particular call.
Pass ``NetworkTensors`` to inspect its stored frames, or a raw ``BoundNetwork``
to inspect its canonical parameter-owned readers. The binding's authored
constant map is unchanged. The same inspection forms apply to Population
bundles and raw bindings; prepared Population bindings do not own raw constants.
The shared report includes aliases and identity-deduplicated trainable leaves.
See :ref:`functional-source-inspection` for the inspection contract.
Compact source-history training with native ``auto`` selection, delivery-time
parameter lookup, and checkpoint continuation is described in
:doc:`../advanced/A22_functional_source_history`.
Intrinsic packed-history inference with hard source detection, current weights,
fixed delays, and native checkpoint continuation is described in
:doc:`../advanced/A23_functional_bitpacked_inference`.
Per-NetCon packed schedules and sparse-calendar inference share a pure pending
calendar with enqueue-time payloads. Capacity, overflow handling, and legacy
checkpoints are described in :doc:`../advanced/A24_functional_pending_calendar`.

Bind extracellular stimulation with
``extra={population_name: (field, waveform)}`` in ``dn.compile`` or
``make_functional(network, dt=...)``. Each selected Population also accepts a sequence
of contacts. Bound stimulation exposes trainable field and waveform leaves,
preserves shared aliases, and follows the accepted simulation clock. An
explicit ``ve`` input can supply another Population without bound stimulation.
See :doc:`../advanced/A28_functional_network_extra` for the complete workflow.

Fresh-policy runners honor the source Network's steady-state Population and
connection caches. Cached Population restoration returns effective parameters
and constants, while parameter-invariant connection histories use current
connection values. Cache packets and explicit continuation are described in
:doc:`../advanced/A29_functional_network_cache_initialization`.

For explicitly lowered execution, ``dn.func.initialize(functional, tensors)``
accepts a ``NetworkTensors`` bundle and defaults to
``functional.initialization_inputs``. Supply an explicit
``NetworkInitializationInput`` through ``initialization=`` when needed.
The host facade preserves native initialization/cache ordering and samples
missing draws from explicit RNG carry. It returns a ``Continuation`` with
effective sources and no callback carry, without running an accepted timestep
or changing the native Network. Use ``bound = functional.bind(initial)``, then
pass ``initial`` separately to ``dn.func.run`` with the intended callback
collection. The same facade
supports Population bundles; see :ref:`functional-explicit-initialization`.

Each successful convenience-runner call publishes an in-memory
``dn.func.Continuation`` at ``auxiliary["continuation"]``. Pass that packet as
the next call's state argument to retain complete Network state, effective
parameters/constants and callback carry together. Per-call parameter overrides
and fresh/cache replacements retain their selected Tensor identities, shared
aliases and gradients; they do not replace runner defaults. Input/draw tapes
and runtime extracellular recipes remain per-call arguments. See
:doc:`../advanced/A57_functional_api_overview` for the same Population/Network
contract. Raw-bound host runs also consume and publish that packet through
``dn.func.run(bound, packet, ...)``. The binding owns its sources: schemas,
Tensor identities and aliases must match, with Network shared constant readers
resolved against their parameter owners. Different or restored sources require
explicit rebinding, such as ``kernel.bind(restored)``. Eager plans and compiled
chunks also accept a ``NetworkTensors`` bundle directly. Binding reads only
parameter/constant sources and does not adopt or validate its state, callback
carry or initialization inputs; pass state and callbacks to execution
explicitly. The packet cannot accompany another constant argument, even
``None``. Existing raw mapping forms remain supported. Callback collections
still have to match the carry, and direct binding/chunk calls retain their
explicit state/callback arguments. See
:ref:`functional-bound-continuation`.

``dn.func.export_continuation(functional, packet, callbacks=callbacks)`` exports
detached ordinary storage for a lowered Population or Network. Restore with
``dn.func.restore_continuation(receiving, saved, callbacks=receiving_callbacks)``
against a compatible receiving plan, then bind the restored source leaves
explicitly before calling ``dn.func.run``. Supplied callback collections belong
to their respective plans. The same versioned plain data/Tensor format is
supported by ``runner.export_continuation(packet)`` and a compatible
reconstructed runner's ``restore_continuation(saved)``. These restore effective
sources, raw parameter/constant aliases, trainability and checked callback carry. Restore
validates source/state schemas, timestep, native mode and callback settings;
it does not initialize, prepare, sample, run or mutate runner defaults. The
checkpoint uses the same dtype/device, while model declarations, runtime tapes,
optimizer state and recording history remain separate. See
:doc:`../advanced/A58_functional_callback_checkpoints` for complete examples.

.. autoclass:: dendra.func.CompiledNetworkRunner
   :members: initial_state, initialization_inputs, parameters, named_parameters, trainable_parameters, constants, functional, callbacks, kernel, find_parameters, parameter_name, get_parameter, export_continuation, restore_continuation, execution_report
   :special-members: __call__

.. autoclass:: dendra.func.PopulationCallback

.. autofunction:: dendra.func.make_functional_network

.. autoclass:: dendra.func.FunctionalNetwork
   :members: initialization_inputs, initialize, bind, sample_initialization_draw, prepare, step, prepare_and_step, make_callbacks, sample_step_draw, sample_draw_tape, rollout, prepare_and_rollout, compile_rollout_chunk, run, extract, check_overflow, commit_state_

.. autoclass:: dendra.func.NetworkTensors
   :members:

.. autoclass:: dendra.func.NetworkRolloutInput
   :members:

.. autoclass:: dendra.func.NetworkInitializationInput
   :members:

.. autoclass:: dendra.func.NetworkInitializationCache
   :members:

.. autoclass:: dendra.func.CompiledNetworkChunk
   :members: steps, bind, capabilities, execution_report
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
Intrinsic packed-history inference is described in
:doc:`../advanced/A23_functional_bitpacked_inference`.
Sparse inference and packed connection schedules are described in
:doc:`../advanced/A24_functional_pending_calendar`.

.. autofunction:: dendra.func.make_functional_netcon

.. autoclass:: dendra.func.FunctionalNetCon
   :members: runtime, pending_capacity, initial_state, read_due, advance, step, extract, check_overflow, commit_state_

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
