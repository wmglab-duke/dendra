.. _model-initialization:

Model initialization
====================

``Population.initialize()`` establishes a complete starting state for a
simulation. It is a state transition rather than a timestep: simulation time
does not advance, and processes that describe change over time do not run.

Fresh initialization
--------------------

For a fresh initialization, Dendra performs one ordered transaction:

#. Build the model if necessary and populate its parameters, including values
   that are resampled on initialization.
#. Reset voltage and run pre-initialize hooks.
#. Initialize mechanism state once and synchronize shared Ion and Material
   fields.
#. Apply initial absolute writes, concentration guards, reversal potentials,
   and the accepted membrane-current state.
#. Run post-initialize hooks and mark the model ready to execute.

Every mechanism ``INITIAL`` path therefore runs once. If any phase fails, the
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
* ``set_value(...)`` applies an override after voltage reset but before
  mechanism initial conditions.
* A pre-initialize hook is appropriate for custom setup that mechanism initial
  conditions, temperature scaling, or derived fields must observe.
* A post-initialize hook observes or adjusts the completed state. No automatic
  recomputation follows it, so it should not be used for values that initial
  conditions depend on.

Registering a pre-initialize hook, calling ``set_value(...)``, or changing
``v_init`` invalidates an existing steady-state cache. This prevents an older
snapshot from silently hiding the new initial condition.

Steady-state restoration
------------------------

A cached steady state is already a complete initialized snapshot, so restoring
it follows a shorter path. Dendra restores the state transactionally and skips
voltage reset, parameter resampling, pre-initialize hooks, and mechanism
``INITIAL``. Post-initialize hooks still run once against the restored state.

The cached stochastic state is restored without consuming another random
sample. The solver workspace is rebuilt once on the first subsequent execution
and is then reused normally during evaluation.

This distinction lets repeated simulations begin from exactly the same cached
state while fresh initialization remains responsive to changed parameters and
initial-condition overrides.
