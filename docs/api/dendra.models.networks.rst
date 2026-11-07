Networks, connections and event sources
=======================================

Build a Network from named Populations, add event or continuous connections,
then call ``network.initialize(dt)``. Network execution uses that fixed timestep.
For connected compiled simulations, see :doc:`../advanced/functional_execution`.

Network
-------

.. autoclass:: dendra.models.networks.Network
   :members: initialize, build, step, run, longrun, longrun_checkpointed, attach_netstim, connect, connect_one_to_one, connect_dense, connect_prob, connect_prob_n, synapse_slots, connect_to_slots, connect_one_to_one_slots, connect_continuous, connect_continuous_one_to_one, clear_synapses, set_synaptic_diff_config, train, eval, devices, dtypes, device, dtype, batch, concat, detach, steady_state, cache_state, clear_state_cache, load_state_cache, load, weights, weight_modules, concatenated_weights, delete_injections, state_dict_for_checkpoint, restore_dict_from_checkpoint
   :member-order: groupwise

Connection handles
------------------

Connections are created by Network methods. Use their returned handles to bind
live weight/time sources or schedule additional events; implementing a custom
synapse does not require subclassing NetCon.

Weights use the units expected by the receiver. Built-in ``expsyn`` and
``exp2syn`` use bare numerical conductances in µS: ``weight=0.05`` means 0.05 µS.
A scheduled event's ``weight`` is instead a dimensionless multiplier of the
connection weight. See :doc:`../synapse_authoring` and :doc:`../units`.

.. autoclass:: dendra.models.networks.netcon.NetCon
   :members: w, schedule, bind_weight_source, bind_time_source, schedule_ref, schedule_time_ref, clear_schedule, set_diff_config, detach, state_cache, state_dict_for_checkpoint, restore_dict_from_checkpoint
   :member-order: groupwise

.. autoclass:: dendra.models.networks.netcon.ContinuousCon
   :members: n_connections, detach, state_cache, state_dict_for_checkpoint, restore_dict_from_checkpoint

NetStim
-------

NetStim generates stochastic renewal events and accepts predetermined schedules.
Use explicit initialization/runtime draws for reproducible replay; see
:doc:`../advanced/inputs_and_randomness`.

.. autoclass:: dendra.models.networks.NetStim
   :members: initialize, renewal_parameters, schedule, bind_weight_source, bind_time_source, schedule_ref, schedule_time_ref, clear_schedule, set_diff_config, freeze, unfreeze, batch, detach, device, dtype, state_dict_for_checkpoint, restore_dict_from_checkpoint
   :member-order: groupwise
   :undoc-members:

Low-level pure projection interfaces are documented in :doc:`dendra.func.low_level`.
