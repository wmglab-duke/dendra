Populations and model construction
==================================

A Population owns voltage, mechanisms and simulation state. Use ``dn.SingleCompartment``
for point cells, ``dn.Cable`` or ``dn.Tree`` for native morphologies, and the axon
classes for repeating cable structures. Native execution updates the model;
:doc:`dendra.func` describes execution with explicit state and source tensors.

Population lifecycle and inputs
-------------------------------

.. autoclass:: dendra.models.core.Population
   :members: initialize, build, step, run, longrun, longrun_checkpointed, steady_state, clear_steady_state, state_dict_for_checkpoint, restore_dict_from_checkpoint, insert, delete, delete_all, bind_input, register_injection, delete_injections, batch, detach, to, cpu, cuda, float, double, half, bfloat16, find, find_not, slice, terminal_indices, set_v_init, expanded_v_init, shape, graph, area, device, dtype, material, material_geometry, register_material_geometry, register_pre_initialize_hook, register_post_initialize_hook, register_pre_initialize_transform, register_post_initialize_transform, register_parametrization, freeze_group, unfreeze_group, retain_grad, cache, restore, load, clear_jit_cache, pickleable
   :member-order: groupwise

Model families
--------------

Subclasses share the lifecycle above. Their entries describe construction and
family-specific geometry. See :doc:`dendra.models.morphology` for sections and
compartment graphs, and :doc:`../basics/02a_native_morphologies` for examples.

.. autoclass:: dendra.models.core.SingleCompartment
.. autoclass:: dendra.models.core.Cable
   :members: from_compartment_graph, from_morphology, compartment_graph, edge_resistance_ohm, material_volume
.. autoclass:: dendra.models.core.Axon
   :members: register_cid, c, csl
.. autoclass:: dendra.models.core.Unmyelinated
   :members: length
.. autoclass:: dendra.models.core.Myelinated
   :members: length, deltax
.. autoclass:: dendra.models.tree.Tree
   :members: from_morphology, from_compartment_graph, compartment_graph, material_volume
.. autoclass:: dendra.models.extcell.ExtCellAxon
.. autoclass:: dendra.models.extcell.ExtCellTree

Packing independent models
--------------------------

``concat_models`` combines independent scalar models into a ``MultiPopulation``
without adding electrical connections. Components must share device and dtype.
Named components remain selectable as Slice regions for recording and stimulation.

.. autofunction:: dendra.models.multi.concat_models
.. autoclass:: dendra.models.multi.MultiPopulation
   :members: material, material_volume, material_edge_index, material_edges, set_v_init, batch
   :undoc-members:

Axon region identifiers
-----------------------

.. autoclass:: dendra.models.heterogeneous.CompartmentID
   :members:

Intrinsic stimulation and boundary helpers
------------------------------------------

.. autofunction:: dendra.models.intrinsic.insert_intrinsic_activity
.. autofunction:: dendra.models.intrinsic.remove_intrinsic_activity
.. autofunction:: dendra.models.core.passive_end_nodes_

See also :doc:`dendra.models.slice`, :doc:`dendra.models.networks`,
:doc:`dendra.models.integrators`, and :doc:`../advanced/inputs_and_randomness`.
