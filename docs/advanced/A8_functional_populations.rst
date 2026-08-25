.. _functional-populations:

Functional Populations (experimental)
=====================================

``dendra.func`` exposes a Population transition as explicit parameter,
geometry, state, and input tensors. This makes supported models usable with
PyTorch function transforms such as ``torch.func.grad``, ``jacrev``, ``vmap``,
and ``torch.compile`` without changing the normal ``Population.step()``,
``run()``, or ``longrun()`` paths.

Supported models
----------------

The current API supports initialized models with all of the following
properties:

* an ``Unmyelinated`` topology with standard cable geometry;
* dense, whole-Population distributed mechanisms;
* deterministic CPU execution in float32 or float64;
* ``bwd_euler_ub`` with membrane-current recording disabled; and
* a transform-compatible Thomas or PCR solver.

Mechanism ODE state, declared mutable buffers, Ion and Material fields, and
write-only shared-field locals are discovered automatically. Initialization-
static workspaces declared with ``Mechanism.DERIVED_BUFFER`` or
``State.DERIVED_BUFFER`` are rebuilt during preparation from current parameters,
temperature, and dense geometry, so they remain differentiable without becoming
explicit state. State variable names are not restricted to those used by the
built-in HH mechanism.

Unsupported features raise ``dendra.func.FunctionalizationError`` during
lowering. Current limitations include regional or sparse mechanism support,
PointProcesses, TABLE lookups, MaterialProcesses, custom ``set_dt`` workspaces,
registered waveform injections, delayed or stochastic state, VoltageProcesses,
callbacks, other Population topologies, accelerator execution, and
``i_membrane`` recording.

Basic use
---------

Start from an initialized Population and choose the fixed timestep for the
functional transition:

.. code-block:: python

   import dendra as dn
   from dendra.models.mod import hh

   model = dn.Unmyelinated(
       [2.0],
       L=100.0,
       dx=10.0,
       integrator=dn.bwd_euler_ub(method="thomas", imem=False),
   )
   model.insert(hh)
   model.initialize()

   fmodel, tensors = dn.func.make_functional(model, dt=0.01)
   prepared = fmodel.prepare(tensors.parameters, tensors.constants)

   next_state, auxiliary = fmodel.step(
       tensors.parameters,
       prepared,
       tensors.state,
       dn.func.StepInput(ve=ve, intra=intra),
   )

The extracted ``PopulationTensors`` contains:

``parameters``
   The Population's named raw parameters. Replace leaves in this mapping when
   optimizing model parameters.

``constants``
   Independent geometry tensors, currently ``diam`` and ``dx``.

``state``
   A cloned tensor tree containing all state required by the supported
   transition, including voltage, mechanism state and mutable buffers, shared
   Ion/Material fields, time, and the duration remainder. Derived buffers are
   intentionally absent; breakpoint-written or accumulating ``BUFFER`` values
   remain present.

``prepare`` derives parameter- and geometry-dependent coefficients once for
eager execution. A prepared object is bound to the exact tensor versions that
created it. Call ``prepare`` again after replacing or modifying a parameter or
constant.

For multiple timesteps, pass time-first drive tensors to ``rollout``:

.. code-block:: python

   final_state, auxiliary = fmodel.rollout(
       tensors.parameters,
       prepared,
       tensors.state,
       dn.func.RolloutInput(ve=ve_over_time, intra=intra_over_time),
   )

When both drives are omitted, supply ``steps=...`` explicitly. ``step`` and
``rollout`` return new state and do not modify the source Population or input
state tensors.

PyTorch transforms
------------------

Raw parameters remain differentiable through preparation and the transition:

.. code-block:: python

   import torch

   parameter_name = "integrator.mech.mechanisms.hh.gnabar_param"

   def final_voltage(gnabar):
       parameters = dict(tensors.parameters)
       parameters[parameter_name] = gnabar
       local_prepared = fmodel.prepare(parameters, tensors.constants)
       state, _ = fmodel.rollout(
           parameters,
           local_prepared,
           tensors.state,
           steps=100,
       )
       return state["integrator"]["v"]

   jacobian = torch.func.jacrev(final_voltage)(
       tensors.parameters[parameter_name]
   )

An eager prepared object may be shared across ``vmap`` lanes when parameters
and constants are shared. If they vary by lane, prepare inside the transformed
function.

Compilation
-----------

Opaque prepared objects cannot cross a ``torch.compile`` graph boundary.
Compile the atomic ``prepare_and_step`` or ``prepare_and_rollout`` methods so
coefficient preparation and consumption remain in the same graph:

.. code-block:: python

   def compiled_final_voltage(gnabar):
       parameters = dict(tensors.parameters)
       parameters[parameter_name] = gnabar
       state, _ = fmodel.prepare_and_rollout(
           parameters,
           tensors.constants,
           tensors.state,
           steps=100,
       )
       return state["integrator"]["v"]

   compiled = torch.compile(compiled_final_voltage, fullgraph=True)

Solver availability
-------------------

Native CPU Thomas and PCR execution requires a ``dendra-solvers`` release that
exports the transform-compatible ``thomas_solve_t`` and ``pcr_solve_t`` Python
functions. The pure-PyTorch PCR implementation remains available without the
extension.

Resuming ordinary execution
----------------------------

Functional execution does not update the original Population. Commit a state
explicitly before returning to ``step``, ``run``, or ``longrun``:

.. code-block:: python

   fmodel.commit_state_(model, final_state)

The target must remain structurally compatible with the functional plan. Build
a new plan after changing mechanism structure, support, dtype/device,
train/eval mode, flags, or State integration configuration.

The functional API covers the transition from an already initialized state.
Gradients through the model's initialization procedure require a separate
functional initializer and are not included in this API.
