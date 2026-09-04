dendra.models.mechanisms
========================

Membrane-current unit contract
------------------------------

Ordinary :class:`~dendra.models.mechanisms.Mechanism` objects are distributed
membrane mechanisms. Their current methods return outward-positive current
density in mA/cm², and their conductance or voltage derivative is in S/cm².
Voltage and reversal potentials are in mV. Dendra preserves these densities
during mechanism assembly and applies compartment area in the voltage solver.

:class:`~dendra.models.mechanisms.ContinuousSynapse` follows that distributed
contract unless it also inherits
:class:`~dendra.models.mechanisms.PointProcess`. For a continuous input driven
by a dimensionless presynaptic gate, a conductance-density target input and its
connection weight are in S/cm².

A :class:`~dendra.models.mechanisms.PointProcess` instead returns lumped,
outward-positive current in nA and conductance in µS. Dendra divides both by
``1e6 * area_cm2`` to obtain the distributed mA/cm² and S/cm² values. Built-in
``expsyn`` and ``exp2syn`` event weights are numerical values in µS:
``weight=0.05`` means 0.05 µS and must not be multiplied by
:data:`dendra.units.uS`.

See :doc:`../units` for the full unit table, scalar-conversion rules, material
units, and finite extracellular-model conventions.

Current conductance paths
-------------------------

Dendra's implicit voltage solvers require each mechanism current to be local
(pointwise in voltage), deterministic, and free of side effects. For an affine
current such as ``g * (v - e)``, Dendra derives the conductance exactly from the
Python expression. A nonlinear pointwise current can instead define an exact
``i_with_conductance(self, v)`` method returning ``(current, conductance)``.
For a distributed mechanism that pair is ``(mA/cm², S/cm²)``; for a point
process it is ``(nA, µS)`` before area normalization.

The voltage passed to ``assigned_values`` and current methods is read-only. Dendra
may gather it once and share that tensor across mechanisms with the exact same
ordered compartment support; mechanism hooks must not modify it in place.

This sharing is automatic and does not fuse or rewrite mechanism classes.
Each current method and point-process area conversion still runs independently;
adjacent contributions to the same current field are then reduced locally and
scattered once. Selectors must match exactly, including fancy-index order.
Duplicate-index supports retain separate scatters. As with other parallel
reductions, the optimization may change the final floating-point association
at overlapping supports by a few last bits.

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
:class:`~dendra.models.mechanisms.NumericalCurrentContractError`. Their cost is
linear in the mechanism state and uses a fixed number of current evaluations;
no full Jacobian is constructed.

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

Mechanism and State value roles
-------------------------------

``State.STATE("x")`` declares a solver-evolved tensor named ``x``.
``Mechanism.STATE_BUNDLE(Gates)`` has a different, structural purpose: it registers a
``State`` subclass and exposes that bundle's declared solver tensors on the
owning Mechanism. State-bundle class names and their flattened solver-tensor
names must therefore be unique within one Mechanism.

The remaining declarations describe the same lifetimes at either scope:

* ``CARRY`` is persistent, checkpointed state that is not advanced by a
  declared State equation. Use ``dtype=...`` for boolean or integer carry and
  ``shape=...`` for structural storage that is independent of morphology.
* ``ASSIGNED`` is ephemeral algebra returned by ``assigned_values(v, values)``.
  It is recomputed whenever an integrator needs it and is neither checkpointed
  nor part of functional carry.
* ``DERIVED_BUFFER`` is a prepared workspace determined by parameters,
  temperature, and geometry.
* ``TIMESTEP_BUFFER`` is a prepared workspace that additionally depends on
  ``dt``.

``Mechanism.SAVE_CURRENT("i")`` is intentionally narrower than ``CARRY``. The
name must also be declared as a nonspecific or ionic current. Dendra owns the
checkpointed ``i_`` mirror, initializes it to zero, and refreshes it when that
current participates in Dendra's current assembly. A direct call to
``mechanism.i(v)`` does not refresh it. The mirror is the raw support-local
current before scatter/reduction and, for a PointProcess, before area
normalization. Thus it uses mA/cm² for a distributed mechanism and nA for a
PointProcess. Authored ``initial_values`` and ``advance`` hooks cannot write the
mirror. In functional state it is available as
``state["mechanism_buffers"][mechanism_name]["i_"]``. Use ``CARRY`` for
counters, latches, histories, or any other user-authored persistent state.

``assigned_values`` must be pure because an integrator may evaluate it more
than once at different stage voltages. It returns exactly the declared
``ASSIGNED`` names. Counters, latches, and other persistent updates belong in
``advance(v, dt, values)``, which runs once per accepted timestep. A custom
State ``advance`` returns every declared solver tensor and may also update that
State's ``CARRY``. A Mechanism ``advance`` returns a partial update of its own
``CARRY`` and writable or additive Ion/Material locals. Every runtime output
must be a Tensor with the declared shape, dtype, and device. A State's default
``advance`` is the generated transition for its declared equations.

Authors migrating custom mechanisms from Dendra 0.24 or earlier should follow
:doc:`Custom-mechanism migration <../upgrading>`.

Derived workspaces
------------------

Use ``Mechanism.DERIVED_BUFFER(...)`` or ``State.DERIVED_BUFFER(...)`` for a
workspace that is completely determined by populated parameters, temperature,
and local geometry and then remains fixed between initializations. Implement
``derive_buffers()`` as a pure function returning exactly the declared names:

.. code-block:: python

   class TemperatureScale(Mechanism):
       Mechanism.GLOBAL(reference=1.0, q10=2.0)
       Mechanism.DERIVED_BUFFER("scale")

       def derive_buffers(self):
           return {
               "scale": self.reference * self.q10 ** ((self.celsius - 22.0) / 10.0)
           }

Dendra refreshes these tensors before State initial-value inference and
``initial_values(...)`` hooks. The builder must not
mutate module tensors or depend on voltage, evolving state, Ion/Material state,
randomness, or timestep.

Pure initialization
-------------------

Use ``State.state_defaults(v, values)`` for fallback State values. Explicit
insertion-time ``ic`` values take precedence over those defaults. Later
Mechanism and State ``initial_values(v, values)`` overlays may intentionally
replace either value; use them for pure declaration-owned initialization of
state or carry:

.. code-block:: python

   class Gate(State):
       State.STATE("x")
       State.CARRY("drive")
       State.DERIVATIVE("x' = 0.0 * x")

       def state_defaults(self, v, values):
           return {"x": torch.sigmoid(0.05 * v)}

       def initial_values(self, v, values):
           drive = torch.sigmoid(0.05 * v + 0.01 * values["celsius"])
           return {"drive": 2.0 * drive}

For fresh functional initialization, every declared State value must be resolved
by a shared-field seed, ``state_defaults``, an explicit ``ic``, or an
``initial_values`` overlay. Mechanism overlays run before ordered State overlays.
A State overlay may return only its own ``STATE`` and ``CARRY`` names; a
Mechanism overlay may additionally return its writable Ion/Material locals.

Every State bundle registered on one Mechanism must have a unique class name,
and their flattened ``STATE`` names must be disjoint. This keeps module lookup,
solver scheduling, checkpoints, and functional state schemas unambiguous. A
subclass that replaces a bundle should inherit directly from the intended
Mechanism base and redeclare the complete replacement bundle rather than
adding a second State with the same output names.

The hook must return Tensors, remain deterministic, and avoid all mutation and
implicit RNG. ``values["celsius"]`` is already padded to local voltage rank for
transform-safe broadcasting; ``values["diam"]`` supplies local geometry.

Use ``Mechanism.TIMESTEP_BUFFER(...)`` or ``State.TIMESTEP_BUFFER(...)`` when a
workspace also depends on the simulation timestep. Implement
``derive_timestep_buffers(dt)`` as a pure function returning exactly the
declared names. ``dt`` is a scalar Tensor in milliseconds on the model's device
and with its dtype:

.. code-block:: python

   class ExponentialDecay(Mechanism):
       Mechanism.GLOBAL(tau=2.0)
       Mechanism.TIMESTEP_BUFFER("decay")

       def derive_timestep_buffers(self, dt):
           return {"decay": (-dt / self.tau).exp()}

Dendra builds timestep workspaces once when an integrator configures or changes
its timestep, after effective parameters and initialization-static derived
workspaces are available. By default, each output may be scalar or
otherwise broadcastable to the owner's ``shape_p``; Dendra stores it in that
canonical local shape so serialization remains stable before and after timestep
configuration.

For a population-independent structural workspace, declare an exact shape. The
empty tuple declares a scalar, while any other tuple consists of non-negative
Python integers:

.. code-block:: python

   Mechanism.TIMESTEP_BUFFER("scalar_factor", shape=())
   Mechanism.TIMESTEP_BUFFER("stage_table", shape=(1, 11, 1))

The builder output must be broadcastable to the declared shape and Dendra
materializes an independent tensor with exactly that shape. Structural shapes
do not acquire axes from ``Population.batch()``; include intentional singleton
axes in the declaration when the mechanism's tensor algebra needs them. An
inherited timestep-buffer name must retain the same shape declaration
throughout its class hierarchy.

``TIMESTEP_BUFFER`` already declares the registered buffer, so do not repeat the
same name with ``CARRY`` or ``ASSIGNED``. The builder must not read evolving
state, mutate registered tensors, use randomness, or depend on another installed
timestep workspace.

Both workspace families are registered buffers and participate in
``state_dict`` serialization. Runtime simulation checkpoints do not treat them
as mutable carry: initialized live workspaces remain tied to the current
parameters, geometry, temperature, and integrator timestep. In the
functional-Population API both families are read-only prepared values, not
evolving state.

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
   :exclude-members: put_no_op, put_slice, put_fancy, register_ion, detach,
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

.. autoclass:: dendra.models.mechanisms.ContinuousSynapse
   :members: INPUT, reset_continuous_inputs, continuous_receive
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

Material processes
------------------

:class:`~dendra.models.mechanisms.MaterialProcess` objects update shared
population-wide Material fields after local Mechanism writes and sources have
been committed. Spatial transport normally runs in the ``transport`` phase;
Material/Ion guards and derived-field updates run afterward.

Material processes have a deliberately separate authoring contract from local
Mechanisms. Declare process parameters with ``GLOBAL``, ``RANGE``, or ``BATCH``
and material dependencies with the concrete family declaration (for example,
``DIFFUSE``, ``CLEAR``, ``CLAMP``, or ``EXCHANGE``). Ordinary Mechanism state,
current, ion/material-use, and workspace declarations are rejected with an
error because the material scheduler does not run those phases.

After insertion, Dendra binds the shared fields and calls
``configure_process(population)``. Binding or execution-map reconstruction may
call it again, so implementations must be idempotent. Dendra calls
``set_dt(dt)`` when timestep configuration changes, then
``advance_materials(dt)`` once per accepted step in ``post_local``,
``transport``, or ``post_transport`` phase order. Ordinary local Mechanism
roles and hooks are not part of this scheduler contract and fail explicitly.
Process-family implementations access bound Material fields through their
private framework adapter rather than a second public field read/write API.
Authors of new process families implement these hooks over complete Material
fields; users of the built-in families normally only need their declarations.

.. autoclass:: dendra.models.mechanisms.MaterialProcess
   :members: METHOD, PHASE, configure_process, set_dt, advance_materials
   :show-inheritance:

.. autoclass:: dendra.models.mechanisms.DiffusionProcess
   :members: DIFFUSE, RELAX, BATH
   :show-inheritance:

Regional DiffusionProcess insertion uses the induced compartment topology. An
edge is active only when both endpoints belong to the process region. Crossing
edges are sealed, disconnected selected components evolve independently, and
excluded field values remain unchanged. Repeated insertion of the same class
forms one union region.

``DiffusionProcess.RELAX(..., where="all")`` means that no *additional*
reservoir mask is applied. It never broadens or bypasses the process insertion
region. Effective reservoir support is the intersection of the process region,
the optional ``where`` mask, and compartments with nonzero ``rate``. ``BATH``
is an exact alias for ``RELAX``. Relaxation is fused into the implicit diffusion
matrix and is therefore not a separate operator-split update. It represents a
fixed/infinite reservoir and does not conserve mass in the modeled field; use
:class:`~dendra.models.mechanisms.ExchangeProcess` for a finite conservative
reservoir.

On Apple MPS, one-dimensional implicit diffusion with ``solver="auto"``,
``"thomas"``, or ``"pcr"`` uses a vectorized parallel-cyclic-reduction solve.
It takes logarithmically many reduction rounds, accepts non-power-of-two region
lengths, and does not construct a dense compartment-by-compartment matrix.
``solver="spd"`` emits a warning and selects the same MPS path. CPU and CUDA
retain their existing device-specific solver policies.

Named chemical geometry
~~~~~~~~~~~~~~~~~~~~~~~

Intracellular, extracellular, periaxonal, or other chemical domains on an
unbranched or Tree Population can use an explicitly registered finite-volume
geometry:

:meth:`~dendra.models.core.Population.register_material_geometry` accepts a
node/control-volume tensor plus either interface area and distance or a
precomputed edge factor.

Geometry components may be scalar or broadcastable tensors. Non-scalar
node/control-volume arrays normally end in ``C`` compartments. Unbranched edge
arrays end in ``C - 1`` interfaces. Tree edge arrays end in compact dimension
``E`` and follow :attr:`~dendra.models.tree.Tree.material_edge_index`, whose
columns are stable ``(parent, child)`` pairs in model-storage coordinates.
Named Tree geometry retains the morphology topology: a zero edge coefficient
may seal an existing edge, but geometry cannot add a new connection. Geometry
names refer to chemical storage and transport only; electrical
extracellular-layer parameters do not imply a chemical volume or diffusion
area.

Implicit Tree diffusion permits zero-volume algebraic junctions when every
conductive component is anchored by at least one positive-volume compartment.
Explicit Tree diffusion requires positive volume at every active node. A
regional Tree insertion is an exact induced forest, so an omitted junction is
not added implicitly and can intentionally disconnect selected branches.

Named sources must be registered Population buffers or parameters. They are
sampled when the spatial operator is configured (normally on the first step
after initialization or after explicit timestep reconfiguration), not read on
every timestep. Differentiable sources preserve autograd; use ``nn.Parameter``
for optimizer-discovered trainables. Direct ordinary tensors are registered as
Population buffers, while direct ``nn.Parameter`` values are registered as
Population parameters. Reconfigure after changing geometry, diffusivity, or
RELAX coefficients.

See :doc:`../advanced/A0_mechanisms_and_ions_materials` for regional,
edge-diffusivity, extracellular, and bath-coupling examples and the full solver
contract.

Ion Management
--------------
.. autofunction:: dendra.models.mechanisms.register_ion
