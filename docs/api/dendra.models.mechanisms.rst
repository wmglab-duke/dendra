dendra.models.mechanisms
========================

Current conductance paths
-------------------------

Dendra's implicit voltage solvers require each mechanism current to be local
(pointwise in voltage), deterministic, and free of side effects. For an affine
current such as ``g * (v - e)``, Dendra derives the conductance exactly from the
Python expression. A nonlinear pointwise current can instead define an exact
``i_with_conductance(self, v)`` method returning ``(current, conductance)``.

Dufort--Frankel requires the stronger *affine* property
``I(v) = g * v + b``, with ``g`` and ``b`` independent of ``v`` during that
evaluation, to center a current exactly between its two stored voltage levels.
Dendra infers this property for supported symbolic current expressions.
``Mechanism.AFFINE("i")`` explicitly asserts it when an analytic or
``NUMERICAL`` current path cannot be proven from source. Supplying a correct
derivative does not imply affine voltage dependence: nonlinear analytic and
numerical currents are evaluated at the current voltage with zero conductance
in Dufort--Frankel, while their conductance remains available to other implicit
solvers. ``Mechanism.EXPLICIT("i")`` takes precedence and always selects this
explicit Dufort--Frankel treatment. An inherited ``AFFINE`` assertion is
discarded when a subclass overrides the current method; the subclass must
redeclare it. Exact rename aliases preserve the assertion.

``Mechanism.NUMERICAL("i")`` is an explicit opt-in to centered finite
differences when an exact path is unavailable. The declaration asserts that
``i`` satisfies the local/pure/deterministic contract, because Dendra perturbs
all voltage elements simultaneously. It supports only float32 and float64 and
is less accurate than an exact conductance, particularly when a large
voltage-independent offset dominates the current.

On the first eager evaluation for each current implementation, input shape,
device, and dtype, Dendra transactionally probes this assertion. The current
must return a tensor with exactly the voltage shape, dtype, and device.
Repeated-input checks catch demonstrable nondeterminism, voltage-input
mutation, and registered buffer/parameter mutation (including tensor view and
storage metadata changes), while four well-spread single-coordinate
perturbations catch demonstrable tensor coupling. The probes restore Dendra and
PyTorch RNG streams, the probed voltage, and registered state before raising
:class:`NumericalCurrentContractError`. Their cost is linear in the mechanism
state and uses a fixed number of current evaluations; no full Jacobian is
constructed.

These bounded checks can disprove an invalid declaration but cannot prove it
for every coordinate, state, parameter value, or control-flow path. Authors
remain responsible for the ``NUMERICAL`` contract. Normal model initialization
performs the first eager current evaluation after ion/material binding and
initial-condition setup. A workflow that enters ``torch.compile`` without any
eager initialization skips Python validation to preserve full-graph capture;
initialize or eagerly evaluate the current once before compiling.

Dendra does not automatically switch to numerical differentiation when source
inspection or symbolic analysis fails. Such a downgrade could mistake a
directional derivative of a coupled current for the required Jacobian diagonal,
or repeatedly execute a stateful or stochastic current. The resulting error
preserves the analysis failure and directs authors to rewrite the affine
expression, provide an exact conductance pair, or use ``NUMERICAL`` only after
establishing its contract. An analytic pair supplies only a local conductance;
it does not make a genuinely coupled current safe. Nonlocal voltage coupling
must be represented outside the local mechanism-current assembly.

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

.. autoexception:: dendra.models.mechanisms.UnsafeAutomaticNumericalFallbackError

.. autoexception:: dendra.models.mechanisms.NumericalCurrentContractError

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

``State.DIFFUSION(...)`` plus ``State.METHOD("euler_maruyama")`` provides
Euler-Maruyama support for Itô stochastic state variables. ``State.METHOD("euler_heun")``
provides Euler-Heun support for Stratonovich stochastic state variables. The
Euler-Heun builder averages the old and predicted diffusion coefficients; pass
``average_drift=True`` to average the deterministic drift term as well. These
methods apply only to ``State`` updates; stochastic voltage/cable solvers are
intentionally separate future work.

Material management
-------------------

Generic :class:`~dendra.models.mechanisms.Material` fields separate their
registered initial-value source from mutable runtime state. ``initial_source``
returns the parameter or parametric module used to reset a field. Initializing a
material in training mode resolves that source without detaching it, so gradients
can flow through initial concentrations and subsequent material processes.
Evaluation initialization detaches the live field state. Both the source and
live field participate in an ordinary PyTorch ``state_dict``; runtime simulation
checkpoints restore the live field only and intentionally leave model parameters
unchanged.

.. autoclass:: dendra.models.mechanisms.Material
   :members: fields, has_field, field_spec, initial_source, initialize
   :show-inheritance:

.. autofunction:: dendra.models.mechanisms.register_material

.. autoclass:: dendra.models.mechanisms.material_defaults

Ion Management
--------------
.. autofunction:: dendra.models.mechanisms.register_ion
