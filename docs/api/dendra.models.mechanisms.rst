Mechanisms, state equations and materials
=========================================

Subclass Mechanism to define membrane currents, State to define state equations,
and the process classes below to interact with shared ions/materials. See
:doc:`../advanced/custom_models` and :doc:`../synapse_authoring` for examples.
Parameter declarations such as ``RANGE`` and ``GLOBALRAND`` are inherited from
:class:`~dendra.models.parametric.Parameterized`; their reference is
:doc:`dendra.models.parametric`.

Current units and conductance
-----------------------------

Distributed Mechanism currents are outward-positive densities in mA/cm² with
conductance in S/cm². PointProcess currents are in nA with conductance in µS;
Dendra performs the area conversion. Voltages and reversal potentials are in mV.
See :doc:`../units` before mixing distributed and point-process parameters.

An affine current may provide ``i_affine_terms(v) -> (G, J)`` for ``I=G*v-J``
and declare ``Mechanism.AFFINE("i")``. An exact ``i_with_conductance`` pair is
also supported. ``Mechanism.NUMERICAL`` opts into numerical conductance for
local deterministic currents. These declarations do not make a nonlocal or
state-mutating current valid. ``dn.ctx(MEMBRANE_ASSEMBLY="validate")`` enables
additional affine-contract diagnostics.

Mechanism and State hooks
-------------------------

.. autoclass:: dendra.models.mechanisms.Mechanism
   :members: STATE_BUNDLE, CARRY, DERIVED_BUFFER, TIMESTEP_BUFFER, ASSIGNED, SAVE_CURRENT, USEION, USEMATERIAL, NONSPECIFIC_CURRENT, EXPLICIT, AFFINE, NUMERICAL, initial_values, assigned_values, advance, derive_buffers, derive_timestep_buffers, rename, register_delayed_state, register_delayed_states, initial_delay_values, delayed_state_values, delayed_states_values, delayed_state, delayed_states, reset_delayed_states, inject, clear_injections, register_waveform_injection, injection_values, evaluate_injections, resample_random_parameters, resample_runtime_noise, state_names, material_usage, material_summary
   :member-order: groupwise
   :undoc-members:

.. autoclass:: dendra.models.mechanisms.State
   :members: STATE, INPUT, CARRY, DERIVED_BUFFER, TIMESTEP_BUFFER, DERIVATIVE, KINETIC, DIFFUSION, ASSIGNED, METHOD, state_defaults, initial_values, assigned_values, advance, derive_buffers, derive_timestep_buffers
   :member-order: groupwise

Voltage and synaptic processes
------------------------------

``VoltageProcess.voltage_values`` supplies imposed voltage. Event synapses
implement ``net_receive_values``; continuous synapses implement
``continuous_receive_values``. Their native adapters publish the returned
values. The pure hooks are shared by native and functional execution.

.. autoclass:: dendra.models.mechanisms.VoltageProcess
   :members: voltage_values, update_v
.. autoclass:: dendra.models.mechanisms.PointProcess
.. autoclass:: dendra.models.mechanisms.Synapse
   :members: net_receive_values, net_receive
.. autoclass:: dendra.models.mechanisms.ContinuousSynapse
   :members: INPUT, continuous_receive_values, continuous_receive

Ions and shared materials
-------------------------

.. autofunction:: dendra.models.mechanisms.register_ion
.. autofunction:: dendra.models.mechanisms.concentrations
.. autofunction:: dendra.models.mechanisms.equilibria
.. autofunction:: dendra.models.mechanisms.register_material
.. autofunction:: dendra.models.mechanisms.material_defaults
.. autoclass:: dendra.models.mechanisms.Material
   :members:

Material processes
------------------

MaterialProcess uses full-field ``material_fields``, ``prepare_materials`` and
``material_updates`` hooks. It does not use ordinary Mechanism initialization,
assigned-value or advance hooks. Its default native adapter publishes the pure
material replacements. Built-in transport processes implement this protocol.

.. autoclass:: dendra.models.mechanisms.MaterialProcess
   :members: METHOD, PHASE, configure_process, set_dt, material_fields, prepare_materials, material_updates, advance_materials
.. autoclass:: dendra.models.mechanisms.DiffusionProcess
   :members: DIFFUSE, RELAX, graph, material_volume
.. autoclass:: dendra.models.mechanisms.ClearanceProcess
   :members: CLEAR, DECAY, RELAX
.. autoclass:: dendra.models.mechanisms.ClampProcess
   :members: CLAMP, SET, MIN, LOWER, MAX, UPPER, BOUNDS
.. autoclass:: dendra.models.mechanisms.ExchangeProcess
   :members: EXCHANGE, COUPLE, TRANSFER
.. autoclass:: dendra.models.mechanisms.SpatialOperator1D
   :members:
.. autofunction:: dendra.models.mechanisms.solve_tridiagonal_1d

Validation and contract errors
------------------------------

.. autofunction:: dendra.models.mechanisms.validate
.. autoexception:: dendra.models.mechanisms.NumericalCurrentContractError
.. autoexception:: dendra.models.mechanisms.UnsafeAutomaticNumericalFallbackError
