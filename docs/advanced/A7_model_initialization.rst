.. _model-initialization:

Model initialization
====================

``Population.initialize()`` establishes a complete starting state for a
simulation. It is a state transition rather than a timestep: simulation time
does not advance, and processes that describe change over time do not run.

Fresh initialization
--------------------

For a fresh initialization, Dendra performs one ordered transaction:

#. Build the model if necessary, populate Population- and integrator-owned
   parameters (including values resampled on initialization), and reset
   voltage.
#. Run pre-initialize hooks.
#. Populate Mechanism- and State-owned parameters inside the mechanism handler,
   initialize their state once, and synchronize shared Ion and Material fields.
#. Apply initial absolute writes, concentration guards, reversal potentials,
   and the accepted membrane-current state.
#. Run post-initialize hooks and mark the model ready to execute.

Every declared initialization path therefore runs once. If any phase fails, the
model remains uninitialized and cannot be stepped until initialization
succeeds.

No time-dependent transport occurs at ``t=0``. In particular, additive
``USEMATERIAL(..., source=...)`` values and Material processes such as
diffusion, clearance, and bath exchange first run during the first actual
timestep. Absolute initial concentration or Material writes are different:
they are part of establishing the starting state and are applied during
initialization.

Choosing where to set initial values
------------------------------------

Use the earliest hook that matches the value's role:

* ``set_v_init(...)`` changes the voltage used by the next fresh
  initialization.
* ``State.state_defaults(...)`` supplies fallback solver-state values below an
  explicit insertion-time ``ic``.
* ``Mechanism.initial_values(...)`` or ``State.initial_values(...)`` is
  appropriate for pure, declaration-owned state and carry initialization that
  should work through both imperative and functional execution.
* A pre-initialize transform is appropriate for declared tensor-only setup that
  should remain available through ``dn.func``.
* A pre-initialize hook is appropriate for custom setup that must be visible to
  mechanism initial conditions, temperature scaling, or derived fields.
* A post-initialize hook observes or adjusts the completed state. No automatic
  recomputation follows it, so it should not be used for values that initial
  conditions depend on.

Registering a pre-initialize hook or transform, or changing ``v_init``,
invalidates an existing steady-state cache. This prevents an older snapshot
from silently hiding the new initial condition.

Pure mechanism initialization values
------------------------------------

Mechanism authors can implement ``initial_values(v, values)`` on a
``Mechanism`` or nested ``State``. A Mechanism may return flattened solver
state, direct ``CARRY``, and owned writable Ion/Material locals. A State may
return only its own ``STATE`` and ``CARRY`` names. ``values`` contains the current
ordered initialization frame, including support-visible ``celsius``, local
``diam``, and the Ion/Material/current aliases declared by the owning
Mechanism. The ordering is shared-field seed, ``state_defaults``, insertion
``ic``, Mechanism values, then State values. Thus ``ic`` overrides
``state_defaults``, while a later authored ``initial_values`` overlay may
intentionally replace either.

These returned-value hooks must be deterministic and read-only, so imperative
and functional initialization execute the same authored computation.

Pure initialization transforms
------------------------------

Use ``register_pre_initialize_transform`` or
``register_post_initialize_transform`` when setup logic should also be
available through ``dn.func``. A transform is a stateless
``torch.nn.Module``: its tensor reads, writes, and additional inputs are
declared at registration, it returns one tuple entry per write, and actions run
in registration order. Canonical paths address raw parameters as
``parameters.<name>`` and initialized carry as ``state.<path>``. Extra inputs
become replaceable leaves in ``PopulationTensors.initialization.transforms``.
Post transforms can observe temporary clock/control writes from earlier post
transforms, but the normal final initialization reset still returns simulation
time and the duration remainder to zero.

Ordinary pre/post hooks remain the permissive imperative extension point and
still receive the Population object. Dendra deliberately does not infer a pure
functional program from those arbitrary callbacks. Use ``set_v_init(...)`` for
the configured initial voltage and an explicit transform for other declared
initialization-time tensor updates.

Steady-state restoration
------------------------

A cached steady state is already a complete initialized snapshot, so restoring
it follows a shorter path. Dendra restores the state transactionally and skips
voltage reset, parameter resampling, pre-initialize hooks, and mechanism
initial-value evaluation. Post-initialize hooks still run once against the
restored state.

The cached stochastic state is restored without consuming another random
sample. The solver workspace is rebuilt once on the first subsequent execution
and is then reused normally during evaluation.

This distinction lets repeated simulations begin from exactly the same cached
state while fresh initialization remains responsive to changed parameters and
initial-condition overrides.
