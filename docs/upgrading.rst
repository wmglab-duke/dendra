.. _upgrading:

Upgrading Dendra
================

Dendra 0.25: custom Mechanism and State classes
-----------------------------------------------

Dendra 0.25 replaces the former overlapping Mechanism and State lifecycle
APIs with one declaration-driven protocol. This affects authors of custom
``Mechanism`` and ``State`` subclasses. Ordinary Population/Network execution
is unchanged; code that used the removed ``Population.set_value(...)`` helper
must also select the explicit replacement below.

There are no compatibility aliases for the removed declarations and hooks.
Failing at class definition is intentional: it prevents a model from silently
running a hook in the wrong phase or dropping mutable state from checkpoints
and functional execution.

Declaration and hook mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 32 40

   * - Dendra 0.24 and earlier
     - Dendra 0.25
     - Meaning
   * - ``Mechanism.STATE(MyState)``
     - ``Mechanism.STATE_BUNDLE(MyState)``
     - Register a structural ``State`` bundle. ``State.STATE("x")`` continues
       to declare a solver-evolved tensor.
   * - ``BUFFER("x")``
     - ``CARRY``, ``ASSIGNED``, ``DERIVED_BUFFER``, or ``TIMESTEP_BUFFER``
     - Select the declaration from the value's lifetime; see below.
   * - Legacy ``Mechanism.ASSIGNED("x")`` used as mutable storage
     - Select the role from the same four declarations
     - ``Mechanism.ASSIGNED`` now has the same pure, ephemeral meaning as
       ``State.ASSIGNED``.
   * - ``Mechanism.SAVE("i")``
     - ``Mechanism.SAVE_CURRENT("i")``
     - Retain a framework-owned mirror of a declared current only. Use
       ``CARRY`` for any author-updated persistent value.
   * - ``Mechanism.INIT(x=value)``
     - ``State.state_defaults(...)`` or insertion-time ``ic={"x": value}``
     - Put a class-owned fallback on its ``State``; use ``ic`` for a particular
       insertion override.
   * - ``State.inf(v)``
     - ``State.state_defaults(v, values)``
     - Supply low-priority initial values for solver state.
   * - ``initial(...)``, ``initial_outputs(...)``, or custom ``State.initialize(...)``
     - ``initial_values(v, values)``
     - Return declaration-owned initial state and carry without mutation.
   * - ``breakpoint(v)``
     - ``assigned_values(v, values)`` or ``advance(v, dt, values)``
     - Separate repeatable current-stage algebra from once-per-step state
       updates.
   * - Custom ``State.solve(...)`` or State/Mechanism ``_advance(...)``
     - ``advance(v, dt, values)`` on the corresponding owner
     - Override a State only when its generated transition from declared
       equations is insufficient.
   * - Custom ordinary-Mechanism ``set_dt(dt)``
     - ``TIMESTEP_BUFFER`` plus ``derive_timestep_buffers(dt)``
     - Declare timestep-dependent prepared workspace explicitly.
   * - ``has_q10``, ``calc_q10()``, or the implicit Q10 cache
     - ``DERIVED_BUFFER`` plus ``derive_buffers()``
     - Make temperature-derived coefficients explicit prepared workspace.
   * - ``method = ...`` or ``method_kwargs = ...`` on a ``State``
     - ``State.METHOD(...)``
     - Declare the integration policy at class definition.
   * - ``Population.set_value(name, value)``
     - The role-specific API
     - Use ``set_v_init`` for initial voltage, parameter assignment for model
       parameters, or a declared pre-initialize transform for tensor setup that
       must also participate in functional initialization.

Replace each former ``BUFFER`` according to how the value is produced and how
long it lives:

* Use ``CARRY`` for counters, latches, histories, and other author-updated
  values that must survive an accepted timestep, checkpoint, or functional
  transition.
* Use ``ASSIGNED`` for pure algebra that can be recomputed from the current
  voltage and value frame. An integrator may evaluate it multiple times in one
  timestep.
* Use ``DERIVED_BUFFER`` for pure prepared workspace determined by parameters,
  temperature, and geometry.
* Use ``TIMESTEP_BUFFER`` when that prepared workspace also depends on ``dt``.

Do not use persistent carry merely to cache repeatable algebra. Conversely, do
not put a counter or latch in ``ASSIGNED``: only ``advance`` is guaranteed to
run once per accepted timestep.

Canonical returned-value hooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The canonical hooks are:

* ``state_defaults(v, values)`` on ``State`` for fallback solver-state values;
* ``initial_values(v, values)`` on ``State`` or ``Mechanism`` for initialization
  overlays;
* ``assigned_values(v, values)`` on ``State`` or ``Mechanism`` for ephemeral
  algebra;
* ``advance(v, dt, values)`` on ``State`` or ``Mechanism`` for accepted-step
  updates;
* ``derive_buffers()`` and ``derive_timestep_buffers(dt)`` for prepared
  workspaces.

Treat all arguments as read-only and return a mapping of Tensors. Dendra checks
the declared names, shapes, dtypes, and devices on the first eager evaluation.
Hooks must not mutate registered or hidden state, consume implicit randomness,
or depend on undeclared mutable globals if they are to participate in
``dendra.func`` transforms.

This is a minimal State and Mechanism using the new protocol:

.. code-block:: python

   import torch

   from dendra.models.mechanisms import Mechanism, State


   class Gate(State):
       State.STATE("x")
       State.ASSIGNED("x_inf", "tau")
       State.DERIVATIVE("x' = (x_inf - x) / tau")

       def assigned_values(self, v, values):
           del values
           x_inf = torch.sigmoid((v + 40.0) / 5.0)
           return {"x_inf": x_inf, "tau": torch.ones_like(v)}

       def state_defaults(self, v, values):
           return {"x": self.assigned_values(v, values)["x_inf"]}


   class Channel(Mechanism):
       Mechanism.STATE_BUNDLE(Gate)
       Mechanism.GLOBAL(gbar=0.01, erev=-70.0)
       Mechanism.NONSPECIFIC_CURRENT("i")
       Mechanism.SAVE_CURRENT("i")

       def i(self, v):
           return self.gbar * self.x * (v - self.erev)

       def i_with_conductance(self, v):
           conductance = self.gbar * self.x
           return conductance * (v - self.erev), conductance

``SAVE_CURRENT`` creates the checkpointed ``i_`` mirror. Dendra refreshes it
during current assembly; directly calling ``mechanism.i(v)`` does not update
the mirror.

Material processes
~~~~~~~~~~~~~~~~~~

``MaterialProcess`` uses a separate population-wide scheduler. Its extension
points remain ``configure_process(population)``, ``set_dt(dt)``, and
``advance_materials(dt)`` together with the concrete process-family
declarations. Ordinary local Mechanism roles and lifecycle hooks are rejected
on a ``MaterialProcess`` subclass.

Functional compatibility
~~~~~~~~~~~~~~~~~~~~~~~~

The same declared protocol drives imperative execution and ``dendra.func``.
The imperative API remains the permissive foundation for model-specific Python
hooks and unusual mutable behavior. Functional lowering is stricter: it accepts
only state, dependencies, and effects that Dendra can represent explicitly.
Use an eager initialization or transition while developing a mechanism so
schema errors are reported before wrapping it in ``torch.compile`` or a
``torch.func`` transform.
