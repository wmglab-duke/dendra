dendra.models.mechanisms
========================

.. autoclass:: dendra.models.mechanisms.State
   :members:
   :inherited-members: Module, object
   :member-order: groupwise
   :exclude-members: populate_parameter_buffers,
    to_column,
    from_column,
    instantiate_additional_parameters,
    instantiate_global,
    instantiate_parameters,
    instantiate_range,
    load_additional_parameters,
    apply_parametrizations,
    check_kwargs,
    detach,
    device,
    register_parametrization_in_graph,
    reshape

.. autoclass:: dendra.models.mechanisms.Mechanism
   :members:
   :inherited-members: Module, object
   :member-order: groupwise
   :exclude-members: set_dt, put_no_op, put_slice, put_fancy, register_ion, detach,
    populate, initialize, batch, states, apply_parameterizations,
    apply_parametrizations,
    check_kwargs,
    device,
    instantiate_additional_parameters,
    instantiate_global,
    instantiate_parameters,
    instantiate_range,
    load_additional_parameters,
    populate_parameter_buffers,
    register_parametrization_in_graph,
    reshape

.. autoclass:: dendra.models.mechanisms.PointProcess
   :show-inheritance:


.. autoclass:: dendra.models.mechanisms.VoltageProcess
   :members: update_v
   :show-inheritance:

.. autoclass:: dendra.models.mechanisms.Synapse
   :members: net_receive
   :show-inheritance:




State SDE support
-----------------

``State.DIFFUSION(...)`` plus ``State.METHOD("euler_maruyama")`` provides v1
Euler-Maruyama support for stochastic state variables. This applies only to
``State`` updates; stochastic voltage/cable solvers are intentionally separate
future work.

Ion Management
--------------
.. autofunction:: dendra.models.mechanisms.register_ion
