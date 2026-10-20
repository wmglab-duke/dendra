.. _functional-populations:

Functional Populations (experimental)
=====================================

``dendra.func`` exposes a Population transition as explicit parameter,
geometry, state, and input tensors. This makes supported models usable with
PyTorch function transforms such as ``torch.func.grad``, ``jacrev``, ``vmap``,
and ``torch.compile`` without changing the normal ``Population.step()``,
``run()``, or ``longrun()`` paths.

Quick start: compiled scan or while loop
----------------------------------------

For a reusable compiled simulation, ``dn.compile`` handles
functionalization, independent parameter copies, preparation, compilation, and
binding. It is an exact top-level alias of ``dn.func.compile``, which remains
supported; both names refer to the same function. Start from an initialized,
functionally supported Population:

.. code-block:: python

   import dendra as dn

   runner = dn.compile(
       model,
       dt=0.025,
       steps=800,
       execution="scan",
       callbacks={"trace": dn.func.Recorder(("v",))},
   )
   final_state, auxiliary = runner()
   voltage = auxiliary["callbacks"]["trace"]["v"]

Recording is configured only through the named ``callbacks`` mapping; no
recorder is added implicitly. ``Recorder`` includes the initial frame, so
``voltage`` has 801 frames in this example. Use
``dn.func.Recorder(("v", "t"))`` to select fields. To record voltage from selected
compartments, use ``dn.func.Recorder(("v",), node_indices=[3])``; voltage node
indices must be nonnegative. Node selection cannot be combined with scalar
fields such as ``t``. For a named cell or physical subregion, use
``dn.func.Recorder(("v",), region=model.cell_a.soma)`` instead; see
the named-region example below. The mapping also configures threshold reducers for either
execution mode:

.. code-block:: python

   runner = dn.func.compile(
       model,
       dt=0.025,
       steps=800,
       execution="while",
       callbacks={
           "active": dn.func.Active(node_check=[0, -1]),
           "counts": dn.func.APCount(node_check=[0, -1]),
       },
   )
   final_state, auxiliary = runner()
   active = auxiliary["callbacks"]["active"]
   counts = auxiliary["callbacks"]["counts"]

The mapping accepts ``dn.func.FunctionalCallback`` configurations, not mutable
``dn.callbacks`` objects or an already-bound collection. Names such as
``"trace"`` are user-defined and have no special meaning. Threshold-only reducers
need no voltage trajectory, so the example above allocates no full-trace
storage. Omitting ``callbacks`` or passing ``callbacks={}`` selects no callbacks;
this is supported by scan, while execution requires a nonempty supported
callback collection.

Choose ``execution="while"`` for the inference-only ``torch.while_loop``
path. Both modes default to ``inference=True``: the runner manages
``torch.inference_mode()`` and, for CPU Inductor compilation, enables the C++
wrapper unless ``options={"cpp_wrapper": False}`` overrides it. The first call
captures and compiles the loop; retain the runner for subsequent calls.
Explicit compiler modes and alternate code-generation options retain their
original compiler policy rather than receiving a new wrapper preference.
Scan and while execution use Dendra's PyTorch 2.14+ baseline; supported features
are not selected by patch-version number. While execution requires at least four steps, a nonempty
collection of supported built-in callbacks, and a capturable inference step.
Supported built-ins are ``Recorder``, ``LFP``, ``AnomalyDetector``, ``Raster``, ``APCount``,
``Active``, ``ActiveAL``, and ``ActiveALCount``; scan also accepts pure custom
functional callbacks. An unsupported while request raises an error
instead of falling back to an unrolled simulation; it does not silently compile a
different execution strategy.

State stays explicit. By default, every ``runner()`` starts from the captured
``runner.initial_state`` snapshot; the runner never advances or commits state
to the source Population. To
continue, pass the returned state. Passing the callback state also preserves
reducer history and its clock without recording a duplicate boundary frame:

.. code-block:: python

   continued_state, continued_auxiliary = runner(
       final_state,
       callback_state=auxiliary["callbacks"].state,
   )

Registered stimuli are captured with the model. Explicit time-first waveform
inputs can instead be supplied with ``runner(inputs=dn.func.RolloutInput(...))``.
The runner exposes ``parameters`` as a read-only mapping of independent tensor
copies, as well as ``constants``, ``functional``, and the compiled ``kernel``.
Use ``runner.execution_report()`` after a call to inspect the selected strategy.

For ordinary first-order backpropagation, select scan and disable inference
explicitly. The default has no trainable parameter leaves; ``trainable`` accepts
the same exact or unambiguous partial parameter names as
``PopulationTensors.independent_parameters``:

.. code-block:: python

   import torch

   train_runner = dn.func.compile(
       model,
       dt=0.025,
       steps=800,
       execution="scan",
       inference=False,
       trainable=("gnabar",),
       callbacks={"trace": dn.func.Recorder(("v",))},
   )
   trainable = [p for p in train_runner.parameters.values() if p.requires_grad]
   optimizer = torch.optim.Adam(trainable, lr=1e-4)
   for _ in range(2):
       optimizer.zero_grad()
       final_state, auxiliary = train_runner()
       loss = auxiliary["callbacks"]["trace"]["v"].square().mean()
       loss.backward()
       optimizer.step()

Gradient-enabled calls prepare a fresh differentiable workspace for each
forward/backward graph. Snapshot-mode inference calls reuse preparation while its sources are
unchanged. ``inference=True`` always suppresses gradients, even for leaves
selected by ``trainable``. A complete replacement mapping may be passed with
``runner(parameters=parameters)``; explicit state tensors may carry gradients
as well. With the default ``initialization="snapshot"``, parameter changes do
not reset the captured initial state or perform differentiable initialization.

Opt-in fresh initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``initialization="fresh"`` to recompute initial state from current model
parameters every time the runner is called **without** explicit state. This
uses the pure differentiable initializer described in
:ref:`functional-fresh-initialization`, including Myelinated geometry,
finite-extracellular circuit state, packed populations, insertion-time initial
conditions, and registered pre/post initialization transforms. Unsupported
initialization hooks raise an error; there is no fallback to snapshot state.
The compiled timestep loop is reused; initialization and preparation are
invocation-local, and no training graph is retained between calls.

.. code-block:: python

   fresh_runner = dn.compile(
       model, dt=0.025, steps=800,
       initialization="fresh",
       inference=False,
       trainable=("gnabar",),
       callbacks={"trace": dn.func.Recorder(("v",))},
   )
   initial = fresh_runner.initialization_inputs
   initial.v_init.requires_grad_()
   final_state, auxiliary = fresh_runner()
   auxiliary["callbacks"]["trace"]["v"].square().mean().backward()

``runner.initialization_inputs`` contains independent tensor defaults in an
``InitializationInput(v_init, states, transforms)``. These initialization-only
leaves are separate from ``runner.parameters`` and ``trainable``. To optimize
them, enable their gradients and include the desired leaves in your optimizer.
An individual fresh call can instead receive
``runner(initialization=dn.func.InitializationInput(...))``; explicit nonleaf
tensors retain their autograd connection. State and transform mappings must
use the complete canonical names and shapes exposed by the defaults.

Explicit state always means continuation and bypasses initialization, even on
a fresh-policy runner. Every fresh-policy call returns its effective runtime
parameters at ``auxiliary["parameters"]``. If a pure initialization transform
replaces parameters (for example, to balance resting current), continuation
must carry that mapping explicitly:

.. code-block:: python

   continued_state, continued_auxiliary = fresh_runner(
       final_state,
       parameters=auxiliary["parameters"],
       callback_state=auxiliary["callbacks"].state,
   )

Without parameter-writing transforms, state-only continuation remains valid.
To replay the originally captured snapshot on a fresh-policy runner, use
``fresh_runner(fresh_runner.initial_state, parameters=fresh_runner.parameters)``.
An initialization override cannot be combined with explicit state or used on a
snapshot-policy runner. A fresh restart cannot accept callback carry without
state: restart both together, or continue both together. Fresh initialization
works with scan, checkpointed training scan, inference-only while loops, and
bound ``extra=`` stimulation; a fresh start resets the accepted simulation
clock, while explicit continuation preserves it. ``Population.run`` and
``Population.longrun`` remain unchanged.

For activation checkpointing, add ``checkpoint_every`` to the same training
scan configuration:

.. code-block:: python

   train_runner = dn.func.compile(
       model,
       dt=0.025,
       steps=20_000,
       execution="scan",
       inference=False,
       checkpoint_every=256,
       trainable=("gnabar",),
       callbacks={"trace": dn.func.Recorder(("v",), node_indices=[0])},
       options={"cpp_wrapper": True},
   )
   final_state, auxiliary = train_runner()
   loss = auxiliary["callbacks"]["trace"]["v"].square().mean()
   loss.backward()

``checkpoint_every=H`` is a positive integer controlling the maximum number of
accepted steps in each checkpoint span, not an alternative execution mode.
The scheduler runs cached compiled scan chunks and a cached shorter tail when
needed. Non-reentrant activation checkpointing replays these spans during
backward; it preserves gradients across the full simulation, initial state,
parameters, explicit inputs, and callback carry. Steps and custom callbacks must
be pure, deterministic, and replay-safe. Replay retains source and prepared-input
freshness checks and the scan donation guard.

Checkpointing exchanges recomputation for lower saved activation memory; it
does not remove requested recordings. Full voltage/gate trajectories remain a
memory floor. Final-state objectives or loss/reducer callbacks that do not emit
trajectories can save more memory than a full-state Recorder. Smaller intervals
also increase host scheduling overhead, so measure runtime and memory for the
intended model and loss rather than assuming a universally best interval.
``None`` (the default) preserves ordinary scan execution. A checkpoint request
requires ``execution="scan", inference=False``; inference and while-loop
requests raise an error instead of silently ignoring the policy. If a training
runner is called inside ``torch.no_grad()``, it uses the same spans without
activation checkpointing or backward replay.

This is an opt-in convenience layer. The lower-level functional API remains
available for explicit callback binding, independent kernel chunking, transforms, and
preparation lifetimes. Imperative ``Population.run`` and ``Population.longrun``
retain their existing behavior. See ``examples/functional_compiled_runner.py``
for a complete small-model example, or
``examples/functional_threshold_callbacks.py`` for named threshold reducers.

Supported models
----------------

The topology list below describes functional *transition* support from an
initialized tensor state. Fresh functional initialization also supports the
canonical point, cable, myelinated, Tree, finite-extracellular, and packed
families with their admitted integrators, including ``scnv`` for standalone
point models. It has additional lifecycle requirements: initialization must be
declaration-driven or use registered pure
transforms, not arbitrary imperative hooks or cached steady states. This
includes the Tigerholm/Thio models with registered pure resting-balance
transforms. See :ref:`functional-fresh-initialization` for the exact contract;
models outside that initializer contract can still start from the extracted
``tensors.state`` wherever transition admission succeeds.

The current transition API supports initialized models with all of the
following properties:

* either an exact ``Population`` or ``SingleCompartment`` topology with standard
  Population area geometry, an ``Unmyelinated`` topology with standard
  cylindrical cable geometry, a ``Myelinated`` model with Dendra's standard
  geometry and parameter transforms, an ``ExtCellAxon`` model with standard
  cylindrical membrane and extracellular-layer geometry, an exact ``Cable`` built from a
  canonical Morphology, a scalar ``Tree`` built from a canonical
  ``CompartmentGraph``, or an ``ExtCellTree`` built from a canonical
  ``CompartmentGraph`` with Dendra's standard finite-extracellular geometry,
  or an exact ``MultiPopulation`` returned by ``dn.concat_models`` whose
  components each satisfy an admitted scalar topology contract above, or
  whose components are all admitted ``ExtCellAxon``/``ExtCellTree`` models.
  Point-only packs use ``bwd_euler_sc_multi``, mixed scalar point, path, and
  Tree packs use ``dhs_multi``, and finite-extracellular packs use
  ``dhs_bt_multi``;
* zero or more explicit leading axes created with ``Population.batch()``;
* whole-Population or regional distributed mechanisms and current-producing
  ``PointProcess`` mechanisms using rectangular, shared-column, or packed
  support, including initialized regional overrides of ``RANGE`` and ``BATCH``
  parameters and regional Ion/Material access. Reads, ionic-current
  contributions, PointProcess currents, and additive Material sources support
  duplicate placements; concentration and Material replacement writes require
  injective support;
* pure ``VoltageProcess`` updates, including ``fire``, ``fire_r``, ``fire_d``,
  and ``fire_r_d``, with their declared reset and refractory carry;
* deterministic CPU or CUDA execution in float32 or float64;
* ``bwd_euler_sc`` or ``scnv`` for exact standalone ``Population`` and
  ``SingleCompartment``,
  ``bwd_euler_ub`` for ``Unmyelinated``, ``Myelinated``, and native ``Cable``,
  ``bwd_euler_bt(method="thomas")`` for ``ExtCellAxon``, ``dhs`` for scalar
  ``Tree``, ``dhs_bt`` for ``ExtCellTree``, or the automatically selected
  ``bwd_euler_sc_multi``/``dhs_multi``/``dhs_bt_multi`` integrator for
  ``MultiPopulation``, with optional membrane-current output; and
* for ``bwd_euler_ub``, a transform-compatible Thomas or PCR solver.

Solver tensors declared by nested ``State.STATE``, direct Mechanism and State
``CARRY``, saved-current mirrors, Ion and Material fields, and write-only
shared-field locals are discovered automatically. Initialization-
static workspaces declared with ``Mechanism.DERIVED_BUFFER`` or
``State.DERIVED_BUFFER`` are rebuilt during preparation from current parameters,
temperature, and the mechanism's exact support-local geometry, so they remain
differentiable without becoming explicit state. Timestep workspaces declared with
``Mechanism.TIMESTEP_BUFFER`` or ``State.TIMESTEP_BUFFER`` are rebuilt in the
same preparation from the explicit ``dt`` constant. Fresh initialization also
supports these declarations: just as with ``Population.initialize()``, all
timestep workspaces start as literal zeros in their declared shapes, before
ordinary derived buffers, State defaults, initial-value hooks, and the initial
current frame are evaluated. Timestep builders are not called during this
initialization phase; in particular, their values at ``dt=0`` need not be zero.
Runtime preparation then rebuilds the workspaces at the actual timestep,
preserving gradients through their parameters and the explicit ``dt``. State
variable names are not restricted to those used by the built-in HH mechanism.

A ``SAVE_CURRENT("i")`` mirror occupies
``state["mechanism_buffers"][mechanism_name]["i_"]`` and follows the same raw
support-local, pre-normalization current-assembly semantics as imperative
execution.

The raw tensor leaves backing initialized regional ``RANGE`` and ``BATCH``
overrides are exposed as functional parameters. ``prepare`` rematerializes the
effective local values differentiably when those leaves are replaced.

PointProcess currents retain their ordinary nA/µS unit convention. During
``prepare``, Dendra derives the ``1e6 * area_cm2`` conversion factor from the
current geometry and the mechanism's exact support. The conversion is therefore
differentiable with respect to geometry and is refreshed when geometry changes.
As in ordinary execution, it uses physical membrane area rather than the
integrator's ``area_scale`` modifier, and a PointProcess cannot occupy a
zero-area location.

Regional shared fields remain canonical full-Population state. Functional
transitions gather each declared local read from that state and return writes
through the same support mapping used by ordinary Population execution. A
non-injective concentration or Material replacement is rejected because a
single replacement operation would supply multiple local values for the same
physical destination. Additive currents and Material sources instead preserve
duplicate-slot multiplicity.

Unsupported features raise ``dendra.func.FunctionalizationError`` during
lowering. Current limitations include non-injective regional concentration or
Material replacement writes and non-injective voltage replacement support,
the reserved compact rowwise support encoding,
TABLE lookups, MaterialProcesses outside the admitted Clearance/Clamp/Exchange/Diffusion
families, mechanism-owned waveform injections, delayed or stochastic state,
legacy or otherwise impure VoltageProcess hooks, imperative callback hooks, functional
callback built-ins beyond ``Recorder``, ``LFP``, ``AnomalyDetector``, ``Raster``,
``APCount``, ``Active``, ``ActiveAL``, and ``ActiveALCount``, other
Population topologies
(including generic ``Axon``, custom ``Population``/``SingleCompartment``
subclasses, nested ``MultiPopulation``, and mixtures of scalar and
finite-extracellular components), Cable/Axon/Tree subclasses with unrecognized
custom geometry, stateful
or unregistered parameter transforms, custom integrator subclasses, the SPD
block solver, and non-CUDA accelerators (including MPS). Row-regular
selectors authored through the normal Population indexing API are already
represented as packed support and are supported; users do not need to choose a
different input form.

The ``bigMRG``, ``smolMRG``, and ``exactMRG`` models supplied by
``dendra-models`` use the admitted ``ExtCellAxon`` path. Their registered
capacitance and axial-resistivity transforms are evaluated during functional
preparation, including their explicit tensor dependencies. Custom transforms
must register numerical dependencies as parameters or buffers and keep any
remaining Python configuration immutable.

.. _functional-voltage-processes:

Functional voltage resets and refractory state
-----------------------------------------------

The built-in ``fire``, ``fire_r``, ``fire_d``, and ``fire_r_d`` VoltageProcesses
support functional transitions, fresh initialization, and explicit-state
continuation on admitted individual and packed models. Insert them as ordinary
mechanisms before initialization:

.. code-block:: python

   import dendra as dn
   from dendra.models.mod import fire_r_d, pas

   model = dn.SingleCompartment(N=1, C=1)
   model.insert(pas)
   model.insert(fire_r_d, threshold=-50.0, rest=-65.0, refractory=5.0)
   model.initialize()

   runner = dn.compile(
       model, dt=0.025, steps=800,
       initialization="fresh", inference=False,
       trainable=("threshold", "rest"),
       callbacks={"trace": dn.func.Recorder(("v", "fire_r_d.time_refractory"))},
   )
   final_state, auxiliary = runner()
   continued_state, continued = runner(
       final_state,
       parameters=auxiliary["parameters"],
       callback_state=auxiliary["callbacks"].state,
   )

Each accepted step applies the voltage processes in insertion order, using
their ordinary regional support, **before** advancing voltage-dependent
mechanism states and assembling membrane currents. This is the same phase as
imperative execution; a reset is not an additional post-solve voltage clamp.
Functional voltage replacement requires injective support, so one process
cannot supply multiple local replacements for the same physical compartment.
In finite-extracellular models, the process acts on membrane voltage for this
phase, while the circuit integrator retains its usual ``vc`` transition.

Refractory flags/countdowns and surrogate gates are declared ``CARRY`` values
and travel in ``state["mechanism_buffers"][mechanism_name]``. ``Recorder`` can
select these values through their canonical names, such as
``"fire_r_d.time_refractory"``. Fresh initialization evaluates their pure
``initial_values`` hooks but does not perform a reset, decrement a timer, or
emit a spike at t=0. Snapshot restarts retain the captured carry; explicit-state
continuation retains the supplied carry without reinitializing it.

``fire`` and ``fire_r`` use hard threshold comparisons. Their voltage reset is
piecewise differentiable, but the spike decision itself supplies no threshold
gradient. ``fire_d`` and ``fire_r_d`` have the same hard forward behavior and
the existing sigmoid straight-through surrogate backward behavior, controlled
by ``tau_gate`` and ``ste_scale``. These surrogate derivatives are deliberately
not finite-difference derivatives of the hard forward trajectory. Refractory
decisions and durations retain their hard bookkeeping semantics; a
differentiable variant does not make every event-time parameter smooth.
Scan, including checkpointed scan, supports these gradients; while execution
remains inference-only.

Custom ``VoltageProcess`` subclasses can implement the pure hook
``voltage_values(self, v, dt, values)``. It returns
``(new_voltage, updates)``, where ``updates`` is a partial mapping of declared
nested ``State.STATE`` variables and direct ``CARRY`` names to replacement
tensors. A reset can therefore replace a mechanism-owned voltage and recovery
variable before their State equations advance. Nested State ``CARRY``,
``ASSIGNED`` values, workspaces, and shared fields remain outside this output
schema. Read trainable parameters from ``self`` and evolving state from
``values``. Return out-of-place results:
do not mutate ``self``, ``v``, or tensors in ``values``. The shared base
``update_v`` adapter commits these returned replacements during imperative
execution; functional execution keeps them explicit. Ordinary Mechanism
declarations and functional eligibility checks still apply, and the pure
hooks undergo the same mutation/dependency audits as other functional hooks.
An authored legacy ``update_v`` override remains usable imperatively but is
rejected during functional lowering; it is not silently interpreted as pure.
Nested State ``setreference`` properties and Tensor attributes outside the
declared runtime frame also remain imperative-only: they can retain live
tensors outside the functional state. Lowering rejects these bindings, and
adding one invalidates an existing functional plan.

.. _functional-scnv:

Mechanism-owned voltage with scnv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``dn.scnv()`` advances mechanisms and evaluates their actual currents while
skipping the membrane-voltage solve. Functional execution supports this
integrator on exact standalone ``Population`` and ``SingleCompartment`` models,
including explicit batches. A pure ``VoltageProcess`` can own the voltage in
declared ``CARRY`` and return it explicitly:

.. code-block:: python

   import dendra as dn
   from dendra.models.mechanisms import VoltageProcess
   from dendra.models.mod import pas

   class voltage_ramp(VoltageProcess):
       VoltageProcess.RANGE(rate=2.0)  # mV/ms
       VoltageProcess.CARRY("voltage")

       def initial_values(self, v, values):
           return {"voltage": v.clone()}

       def voltage_values(self, v, dt, values):
           voltage = values["voltage"] + self.rate * dt
           return voltage, {"voltage": voltage}

   with dn.ctx(IMEM=True):
       model = dn.SingleCompartment(N=1, integrator=dn.scnv())
       model.insert(pas)
       model.insert(voltage_ramp)
   model.set_v_init(-65.0)
   model.initialize()
   runner = dn.compile(
       model, dt=0.025, steps=40, initialization="fresh", inference=False,
       trainable=("rate",),
       callbacks={"trace": dn.func.Recorder(("v", "voltage_ramp.voltage", "i_membrane"))},
   )
   final_state, auxiliary = runner()

Each accepted step applies voltage processes, advances pre-current mechanism
states, evaluates actual membrane currents, then advances post-current states.
No current, capacitance, or raw ``intra``/``ve`` input produces a voltage solve.
In the example, ``pas`` contributes current but does not change the ramp voltage.
With ``imem=True``, ``i_membrane`` includes the evaluated current and the
capacitive current from the imposed voltage change, following the shared
membrane-current contract. Mechanism-owned waveform injections remain
unsupported.

The established phase order also applies when a mechanism's ``advance`` hook
updates its voltage carry: public ``v`` may then contain the value exposed by
the preceding voltage-process phase, while the carry already holds its next
value. Record both when that distinction matters; functional execution does
not synchronize them by adding a second voltage update.

Fresh initialization, snapshot starts, explicit-state continuation, recording,
and ``commit_state_`` preserve both public voltage and mechanism carry. The
ordinary functional transforms, compiled scan, checkpointed scan, and
inference-only while execution use the same contract. Packed ``scnv`` components
are rejected: concatenation selects a packed integrator and would otherwise
replace the skipped solve with an ordinary point-voltage solve. Cable, Tree,
finite-extracellular, and custom integrator subclasses are outside this
``scnv`` admission.

.. _functional-material-processes:

Functional processes over Material fields
-----------------------------------------

Declaration-only ``ClearanceProcess``, ``ClampProcess``, ``ExchangeProcess``,
and ``DiffusionProcess`` subclasses support functional transitions, fresh
initialization, and explicit-state continuation with global or regional
insertion. This includes admitted individual and packed ``MultiPopulation``
models, including finite-extracellular models.
The existing ``CLEAR``/``DECAY``/``RELAX``, ``METHOD``, and ``PHASE`` declarations
are unchanged; no extra functional annotations or voltage-local State are
required. ``METHOD("exact")`` uses exponential relaxation,
``METHOD("implicit")`` uses backward Euler, and ``METHOD("explicit")`` uses
forward Euler. Declare trainable rates and targets with ``GLOBAL``, ``RANGE``,
or ``BATCH`` and reference their names from ``CLEAR``:

.. code-block:: python

   import torch
   import dendra as dn
   from dendra.models.mechanisms import ClearanceProcess
   from dendra.models.mod import pas

   class BathRelaxation(ClearanceProcess):
       ClearanceProcess.GLOBAL(kclear=0.01, bath=0.2)
       ClearanceProcess.CLEAR("solute", field="c", rate="kclear", target="bath")
       ClearanceProcess.METHOD("exact")
       ClearanceProcess.PHASE("post_local")

   model = dn.SingleCompartment(N=1, C=1, dtype=torch.float64)
   model.material("solute", fields={"c": 1.0}, min_values={"c": 0.0})
   model.insert(pas)
   model.insert(BathRelaxation)
   model.initialize()

   runner = dn.compile(
       model, dt=0.025, steps=8,
       execution="scan", initialization="fresh", inference=False,
       trainable=("kclear", "bath"),
       callbacks={"trace": dn.func.Recorder(("v", "solute_material.c"))},
   )
   final_state, auxiliary = runner()
   continued_state, continued = runner(
       final_state,
       parameters=auxiliary["parameters"],
       callback_state=auxiliary["callbacks"].state,
   )
   loss = continued["callbacks"]["trace"]["solute_material.c"][-1].mean()
   loss.backward()

The field is explicit state at ``state["materials"]["solute"]["c"]``;
``Recorder`` accepts its canonical name ``"solute_material.c"``. The initial
recording contains the freshly initialized field, not one clearance step.
Continuation consumes the supplied field state and does not reset it. Scan
supports gradients, including checkpointed scans; while-loop execution remains
inference-only, as for electrical state.

Each accepted step first commits local Mechanism replacement writes and adds
their Material sources once. Processes then run in ``post_local``,
``transport``, and ``post_transport`` order, preserving insertion order within
each phase. Repeated ``CLEAR`` declarations for one field are sequential: each
consumes the preceding replacement. Shared-field guards and enabled derived
updates, including Nernst reversals, run afterward, followed by local-field
refresh. Fresh initialization applies its normal shared-field transaction but
does not advance Material sources or processes at t=0.

The tensor-only process preparation/update protocol is separate from ordinary
Mechanism ``DERIVED_BUFFER``/``TIMESTEP_BUFFER`` declarations. Out-of-place
arithmetic alone does not make an arbitrary ``advance_materials`` hook pure:
the imperative adapter still commits bound Material state. Functional execution
uses explicit field inputs and returned replacements instead. Functional
preparation runs outside the accepted-step loop and may depend on parameters,
geometry, timestep, and field schemas, not evolving concentrations.
State-dependent algebra belongs in the field-update computation.
Regional insertion of the Clearance/Clamp/Exchange families uses the declared
process support, intersected with material-field availability. Custom process
lifecycle/update overrides remain unsupported by functional lowering.

``ClampProcess`` uses the same pipeline for hard assignments, lower/upper
bounds, and bounded ranges. Its existing ``where=`` masks/selectors are
prepared outside the loop and remain explicit, nondifferentiable boolean
inputs. Clamp values and bounds can be named trainable parameters. Add a
process before initializing/lowering the model, for example:

.. code-block:: python

   from dendra.models.mechanisms import ClampProcess

   class BathBounds(ClampProcess):
       ClampProcess.GLOBAL(lower=0.1, upper=2.0)
       ClampProcess.BOUNDS("solute", field="c", lower="lower", upper="upper")

Hard clamps are piecewise differentiable, not smooth at their switching
points. Dendra retains PyTorch's ``minimum``/``maximum`` tie-gradient
conventions. Fresh initialization does not apply a clamp at t=0; it runs in
the selected process phase after each accepted step. Packed models retain the
same selector semantics as imperative execution; explicit masks allow precise
selection of packed compartments.

``ExchangeProcess`` conservatively couples two modeled fields, unlike
Clearance's fixed reservoir. ``EXCHANGE``/``COUPLE``/``TRANSFER`` retain their
existing rate or conductance and volume arguments, with exact, implicit, and
explicit updates. Named parameters and tensor literals remain differentiable;
inferred volumes follow live geometry, including exact Tree/Cable volumes and
packed component membrane areas. Functional preparation does not reuse a
detached geometry snapshot. Pairs with either volume at or below ``eps`` are
identities, with finite zero coefficient/volume gradients on that inactive
branch. The active-set boundary itself is not smooth.
Packed Exchange retains the exact component chemical volumes; explicit weights
can still describe an alternative two-pool storage model.

``DiffusionProcess`` shares its finite-volume coefficient preparation and
update math with imperative execution. ``DIFFUSE`` supports implicit and
explicit updates, analytic 1-D geometry, exact native Tree/Cable geometry,
and registered named material geometries. Node- and edge-aligned diffusivity,
named volume/edge primitives, literal tensor coefficients, timestep, and
initial fields are explicit differentiable inputs. Regional insertion uses
the induced sealed subgraph, not zero diffusivity alone at crossing edges.
``RELAX``/``BATH`` terms contribute to the same implicit matrix and right-hand
side, preserving the existing unsplit reservoir coupling and single
application of mechanism Material sources.

.. code-block:: python

   from dendra.models.mechanisms import DiffusionProcess

   class BathTransport(DiffusionProcess):
       DiffusionProcess.GLOBAL(D=0.1, rate=0.03, bath=0.2)
       DiffusionProcess.DIFFUSE("solute", field="c", D="D")
       DiffusionProcess.RELAX("solute", field="c", rate="rate", target="bath")
       DiffusionProcess.METHOD("implicit")

Functional preparation runs once outside the accepted-step loop. It retains
immutable solver topology and rebuilds numerical workspace from explicit
inputs; it does not retain configure-time numerical snapshots. Fresh
initialization does not diffuse at t=0, and explicit-state continuation retains
the usual field/source ordering. Structural checks remain active under
transforms; callers changing coefficients must preserve the physical-domain
contract validated during imperative operator configuration: finite,
nonnegative diffusivity/storage and strictly positive selected storage or
edge distances wherever required by the chosen method. Implicit Tree
diffusion retains support for zero-volume algebraic junctions.

Native Thomas, PCR, and dense 1-D solves, and native DHS/dense Tree solves are
supported, with CUDA-compatible solver facades on CUDA. The implicit CPU SPD
transport solver remains imperative-only; functional lowering does not
silently substitute another solver. Explicit Tree diffusion rejects active
zero-volume junctions, for which implicit diffusion is required.

Concatenation preserves component material declarations, initial sources,
field guards, named chemical geometry, and component-local process parameters.
Each component's processes retain their support and phase ordering; in
particular, different component ``GLOBAL`` process values remain independent.
As with other mechanisms, packing reconstructs process parameters from the
authored insertion configuration, not a snapshot of later-trained runtime
parameters. Train or update the packed model's mechanism parameters after
packing.
Diffusion uses a disconnected chemical forest, preserving both component and
neuron-row boundaries and each native Tree's branches. A material field absent
from a component has zero-filled inactive recording slots, not a new chemical
pool. Process and material support seal crossing edges and leave inactive rows
unchanged. Native scalar DHS already handles this forest; packing does not add
chemical connections or require a material to exist everywhere voltage does.

The same composition is used by imperative ``run``/``longrun`` and functional
scan/while execution. Fresh initialization rebuilds packed fields from their
original sources and preserves their gradients; explicit-state continuation
does not reset them. Component-authored ``where="root"``/``"terminal"`` clamps
retain each component's original endpoints. On the packed parent these
selectors cover the endpoints of every component/neuron, rather than only the
first and last flattened slot. Material fields with the same name must have
compatible domain, units, and conservation metadata across components. Shared
ion species must also use compatible effective ``ion_style`` flags; conflicting
component policies are rejected rather than silently changing reversal updates.

Basic use
---------

Start from an initialized Population and choose its default functional
timestep:

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

Exact plain ``Population`` and ``SingleCompartment`` models use the same
functional calls with ``dn.bwd_euler_sc(imem=False)``, their default solver.
Every ``(N, C)`` slot is an independent membrane compartment, including when
``C > 1``: for example, ``dn.Population(N=2, C=3)`` contains six independent
compartments, not two three-compartment cables. Intracellular current affects
each compartment's voltage, while an extracellular ``ve`` input has no effect
because there are no axial connections. Use an admitted cable or Tree model
when spatial voltage coupling is intended. Both exact point-model classes
support fresh functional initialization; custom subclasses and generic
``Axon`` models are not admitted. Compatible extraction and commit retain the
same concrete model class rather than treating ``Population`` and
``SingleCompartment`` as interchangeable plans.
These standalone topologies also admit ``dn.scnv()`` for mechanism-owned
voltage; see :ref:`functional-scnv` for its skipped-solve semantics.

``Myelinated`` and compatible subclasses use the same calls as
``Unmyelinated``. Their canonical fiber diameters and node spacing remain
explicit constants. The full per-node ``diam_original`` source used by
PyTorch's registered diameter transform is explicit as well, so loaded or
edited parametrization state is preserved exactly. Dendra's diameter and
axial-resistivity transforms are evaluated during ``prepare``. Raw geometry
coefficients can therefore be optimized or differentiated like other model
parameters.

``ExtCellAxon`` and compatible MRG subclasses also use the same functional
calls, with ``dn.bwd_euler_bt(method="thomas", imem=False)``. Their state tree
carries both membrane voltage ``v`` and the complete three-potential circuit
state ``vc``; Dendra preserves ``v == vc[..., 0] - vc[..., 1]`` after every
transition. A current ``dendra-solvers`` installation supplies the native
transform-compatible CPU block-Thomas solver. The normal imperative integrator
continues to call the raw native operator directly. On CUDA, functional
execution uses a transform-compatible facade over the existing three-component
Triton block-Thomas kernel; no CPU solver or topology plan is selected.

The block path participates in the same lifecycle as the scalar paths:
``run``, ``longrun``, checkpointed BPTT, fixed compiled rollout chunks, and
``commit_state_`` all preserve the complete ``v``/``vc`` state. Preparation is
still performed once outside the runner or compiled chunk.

An exact native ``Cable`` constructed with ``Cable.from_morphology`` or
``Cable.from_compartment_graph`` uses the same calls with ``bwd_euler_ub``.
Its compiled membrane areas and edge resistances remain the electrical source
of truth; Dendra does not approximate tapered or heterogeneous Sections as
cylinders during functional preparation. The canonical resistance already
contains each Section's authored ``rhoa``. Consequently, the native Cable's
raw ``rhoa_param.rho`` leaf has no voltage derivative in this transition; use
the scalar ``rhoa_scale`` parameter or the explicit canonical-resistance tensor
for axial-resistance sensitivity analysis.

A scalar ``Tree`` constructed with ``Tree.from_morphology``,
``Tree.from_compartment_graph``, or ``Tree.from_graph`` uses the same calls with
``dn.dhs(imem=False)``. Its immutable compiled topology, exact membrane areas,
and child-indexed edge resistances remain the numerical source of truth; no
unbranched or cylindrical approximation is introduced. A current
``dendra-solvers`` installation supplies the transform-compatible CPU DHS
solver; CUDA uses the existing Triton DHS kernel through its functional facade.
As with native ``Cable``, the compiled resistance
already contains authored axial resistivity: use ``rhoa_scale`` or the explicit
canonical-resistance tensor for axial-resistance sensitivity rather than the
raw ``rhoa_param.rho`` leaf.

A canonical ``ExtCellTree`` uses the same functional calls with its default
``dhs_bt(imem=False)`` integrator on CPU or CUDA; import ``dhs_bt`` from
``dendra.models.integrators`` when selecting it explicitly. Its state carries
public membrane voltage ``v`` and the complete three-potential circuit state
``vc``. The exact compiled membrane area and child-indexed intracellular edge
resistance remain the intracellular source of truth, while ``dx``, ``xraxial``,
``xc``, and ``xg`` describe the two extracellular layers. The functional
transition supports the same reverse- and forward-mode differentiation,
``vmap``, compilation, host runners, checkpointed BPTT, and commit/resume
lifecycle as the other admitted topologies. It requires the transform-compatible
CPU block-DHS solver supplied by ``dendra-solvers``, or the three-component
Triton block-DHS solver on CUDA. Optional membrane-current output uses the same
``i_membrane`` state and Recorder contract as the scalar solvers.

Packed MultiPopulations
------------------------------

An exact ``MultiPopulation`` produced by ``dn.concat_models`` uses the same
functional API. A scalar pack contains independently admitted exact
``Population`` or ``SingleCompartment``, ``Unmyelinated``/``Myelinated`` and
their admitted subclasses, exact native ``Cable``, or canonical scalar ``Tree``
components.
Point components using ``scnv`` are excluded from functional packs.
A finite-extracellular pack instead contains only admitted ``ExtCellAxon``
and/or canonical ``ExtCellTree`` components, each with two extracellular
layers. Mixing scalar and finite-extracellular components, and nesting
``MultiPopulation`` objects, remain unsupported and raise an error.

A plain ``Population(N, C)`` component contributes ``N * C`` independent
point compartments even when packed alongside cables or Trees. Concatenation
does not reinterpret its ``C`` axis as an axially connected path.

Functional execution preserves the purpose of concatenation: the composite is
advanced as one packed system, not as a Python loop over components. Its state
and direct ``ve``/``intra`` inputs retain the ordinary flat composite shape
``(*batch, 1, total_width)``. There is no component-nested state tree. Component
geometry and physical parameters remain explicit preparation inputs, so a
gradient can still reach, for example, one component's capacitance or axial
resistivity without changing the packed runtime representation.

Finite-extracellular packs additionally carry ``vc`` with shape
``(*batch, 1, total_width, 3)``. Each compartment retains its intracellular,
inner-shell, and outer-shell potentials, with
``v == vc[..., 0] - vc[..., 1]``. The packed block-DHS solve treats the
components and their neuron replicas as a disconnected forest: concatenation
does not add axial connections between models or merge their extracellular
layers. Public membrane voltage and all three circuit potentials can be
recorded through ``dn.func.Recorder(("v", "vc"))``.

Component physical parameters keep names such as
``populations.<component>.*``. Packed mechanism parameters retain their normal
``integrator.*`` names, and the common temperature appears once. Geometry
constants are likewise component-namespaced, for example
``populations.axon.dx`` or
``populations.tree.canonical_area_cm2``. Calling ``prepare`` assembles those
sources into one differentiable packed workspace; it does not run the
component transitions.

For finite-extracellular components this includes the component-namespaced
``xraxial``, ``xc``, and ``xg`` tensors. ``ExtCellTree`` retains its exact
canonical area and intracellular edge resistance; ``ExtCellAxon`` retains its
cylindrical geometry. These tensors remain differentiable preparation inputs
and are shared across explicit leading batch axes, just as in the standalone
functional models.

Call ``batch`` on the composite after concatenation, as in imperative code:

.. code-block:: python

   packed = dn.concat_models(
       {"soma": soma, "axon": axon, "dendrite": dendrite},
       threads=8,
   )
   packed.batch(16)
   packed.initialize()

   fmodel, tensors = dn.func.make_functional(packed, dt=0.01)
   prepared = fmodel.prepare(tensors.parameters, tensors.constants)

The same workflow applies to a finite-extracellular pack; ``concat_models``
automatically selects ``dhs_bt_multi``:

.. code-block:: python

   packed = dn.concat_models({"axon": extcell_axon, "tree": extcell_tree})
   packed.batch(8)
   packed.initialize()
   runner = dn.func.compile(
       packed, dt=0.01, steps=100, execution="scan",
       callbacks={"trace": dn.func.Recorder(("v", "vc"))},
   )
   final_state, auxiliary = runner()

As for standalone block models, scan supports first-order backward and optional
``checkpoint_every``; while-loop execution is inference-only. Fresh functional
initialization reconstructs the packed parent state without mutating its
components. ``commit_state_`` restores the packed state explicitly. With
``write_back=True`` it also rebinds each component's ``v`` and ``vc`` to its
packed slice; ``write_back=False`` leaves component states untouched.

Registered intracellular injection and bound extracellular stimulation use the
same authored forms described below and are projected onto the flat composite.
The packed transition supports the ordinary functional runners,
differentiation, ``vmap``, and transform-then-compile compositions.
Heterogeneous packs require the transform-compatible native CPU ``dhs_multi``
solver supplied by a current ``dendra-solvers`` installation, or the existing
packed Triton DHS kernel on CUDA. The CUDA functional facade explicitly zeros
inactive padded slots and preserves gradients only for active morphology nodes.
Finite-extracellular packs instead use the three-component block-DHS solver
on their disconnected forest, requiring the transform-compatible native CPU
``dhs_bt_solve`` facade or the existing Triton block-DHS kernel on CUDA. They
do not require a new packed native extension or introduce padded circuit rows.

Current native builds let scalar DHS, block DHS, and heterogeneous packed DHS
reuse an owning, validated CPU topology plan. Initialization snapshots topology
and row assignments; timestep-dependent coefficients remain ordinary numerical
inputs. Functional lowering owns an independent plan, including through
``vmap`` and compiled scan/while execution. Imperative ``run`` and ``longrun``
keep their existing APIs and check topology tensor identities/version counters
outside compiled transitions. Normal mutation or rebinding rebuilds the plan;
versionless inference metadata or an older solver package safely uses the
fully checked raw solver. No plan stores differentiable coefficients or shares
mutable scratch between invocations. As with autograd, writes through external
aliases or ``.data`` that bypass tensor version counters are unsupported.

Tree lowering deliberately accepts only Dendra's canonical compiled topology
and geometry contract. Low-level Tree-like objects without that compiled
source, or subclasses that replace the framework-owned area or resistance
semantics, fail during lowering rather than being approximated.

The extracted ``PopulationTensors`` contains:

``parameters``
   The Population's named raw parameters and any stimulation tensor leaves.
   Replace leaves in this mapping when optimizing model or waveform
   parameters. This includes the raw leaves backing initialized regional
   ``RANGE`` and ``BATCH`` overrides.

``constants``
   Independent geometry and timestep tensors. Plain ``Population``,
   ``SingleCompartment``, ``Unmyelinated``, native ``Cable``, and scalar ``Tree``
   expose ``diam`` and ``dx``; ``Myelinated`` exposes canonical ``diameters``,
   ``diam_original``, and ``dx``; and ``ExtCellAxon`` and ``ExtCellTree`` expose
   ``diam``, ``dx``, ``xraxial``, ``xc``, and ``xg``. Every topology exposes scalar ``dt`` in
   milliseconds. Registered tensor buffers read by an admitted in-graph
   parameter transform are exposed under ``parametrizations.*`` so preparation
   has no hidden numerical inputs. A native ``Cable``, scalar ``Tree``, or
   ``ExtCellTree`` also exposes
   ``canonical_area_cm2`` and
   ``canonical_edge_resistance_ohm``. The resistance tensor is child-indexed
   and retains a zero placeholder at the root. After
   ``Population.batch()``, these retain the shared neuron/compartment core
   shape rather than being copied across replicas. Replace ``dt`` and prepare
   again when evaluating another timestep.

   ``MultiPopulation`` instead exposes each component's geometry under
   ``populations.<component>.*`` and assembles packed ``diam``/``dx`` and
   electrical workspaces during preparation. Canonical area and resistance are
   present only for components whose exact topology requires them.

   Canonical native-Cable, scalar-Tree, and ``ExtCellTree`` tensors are
   differentiable numerical inputs to the functional transition, so they can
   be used for local sensitivity analysis.
   Morphology compilation itself is not an autograd operation: gradients end
   at these tensors and do not propagate back to Section points. If area and
   resistance are replaced independently, the caller is responsible for the
   physical consistency of that counterfactual geometry.

``state``
   A cloned tensor tree containing all state required by the supported
   transition, including voltage, mechanism ``STATE`` and ``CARRY``, shared
   Ion/Material fields, time, and the duration remainder. Derived and timestep
   workspaces are intentionally absent, and ``ASSIGNED`` values are recomputed
   when needed rather than carried between steps. ``ExtCellAxon`` and
   ``ExtCellTree`` additionally carry the complete ``vc`` circuit state
   alongside public membrane voltage ``v``.

   ``MultiPopulation`` state follows the flat composite runtime rather than a
   component tree. The packed parent owns voltage, mechanism/Ion/Material
   carry, clock, and duration remainder. Finite-extracellular packs carry both
   flat ``v`` and flat ``vc``.

   When the source integrator has membrane-current output enabled, its latest
   current is also explicit at ``state["integrator"]["i_membrane"]``. It has
   the same shape as ``v`` and uses mA, following the imperative integrator's
   sign and timestep convention. A fresh initialization starts it at zero;
   extraction from an advanced model preserves the latest current. With
   membrane-current output disabled, this leaf is absent and the ordinary
   voltage-only state schema is unchanged.

``initialization``
   Initialization-only tensor inputs. ``v_init`` is the independent, fully
   expanded initial membrane voltage. ``states`` contains explicit
   insertion-time ``ic`` values, keyed by canonical names such as
   ``mechanisms.hh.m``. These leaves can be replaced or made trainable without
   changing transition preparation.

Parameter lookup and independent copies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Canonical parameter paths remain the keys accepted by ``prepare`` and the
transition, but callers do not need to hard-code those paths. The extracted
container exposes shallow, ordered ``model_parameters``, ``intra_parameters``,
and ``extra_parameters`` mappings. Stimulation mappings include every
replaceable tensor input in that namespace, including fixed waveform buffers
or extracellular fields that were originally registered as buffers rather
than ``Parameter`` objects.

Use ``find_parameters`` when multiple matches are useful, or
``parameter_name``/``get_parameter`` when exactly one result is required:

.. code-block:: python

   matches = tensors.find_parameters("bar", within="model")
   gnabar_name = tensors.parameter_name("gnabar", within="model")
   gnabar = tensors.get_parameter("gnabar", within="model")
   intra_amp = tensors.get_parameter("amp", within="intra")

Searches are case-sensitive substring matches. Exact canonical names take
precedence during single-result resolution; missing or ambiguous partial names
raise with the matching canonical paths. The available scopes are ``"all"``,
``"model"``, ``"intra"``, and ``"extra"``.

Regional overrides remain separate canonical leaves. For example, a mechanism
may expose ``gnabar_param``, ``gnabar_soma``, and ``gnabar_axon``. A singular
``parameter_name("gnabar")`` request then raises as ambiguous instead of
silently choosing a region. Use the ordered result of ``find_parameters`` to
select all matches explicitly:

.. code-block:: python

   gnabar_parameters = tensors.find_parameters("gnabar", within="model")
   parameters = tensors.independent_parameters(
       trainable=gnabar_parameters,
       within="model",
   )

Passing stable aliases when inserting regional mechanisms produces similarly
readable suffixes and permits narrower searches such as ``"gnabar_soma"``.
The base/default leaf is included when it matches, even when regional overrides
currently cover every compartment; lookup reports the complete explicit
parameter mapping rather than attempting to infer whether a leaf is active.

These lookup results are shallow dictionary projections: rebinding an entry in
the returned dictionary does not update ``tensors.parameters``, while its
Tensor leaves still refer to the extracted source leaves. Use them for
discovery, and call ``independent_parameters`` before mutation or optimization.

``independent_parameters`` clones the complete explicit parameter mapping so
it shares neither storage nor autograd history with the source Population.
Queries listed in ``trainable`` are resolved within the requested scope and
wrapped as optimizer-ready ``torch.nn.Parameter`` leaves:

.. code-block:: python

   parameters = tensors.independent_parameters(
       trainable=("gnabar", "gkbar"),
       within="model",
   )

The scope constrains trainable-name lookup; the returned mapping is still
complete and can be passed directly to ``prepare``. Perform name searches and
construct independent leaves outside compiled or transformed functions.

``prepare`` derives model-parameter-, geometry-, and timestep-dependent
coefficients once for eager execution. A prepared object is bound to the exact model tensor
versions that created it. Call ``prepare`` again after replacing or modifying
a model parameter, geometry constant, or ``dt``. Stimulation leaves may change
without preparing the model again. Declared derived and timestep workspaces are
read-only during a functional transition; values that evolve belong in
``STATE``, ``CARRY``, or a declared shared field. Lowering probes both one- and
two-step transitions on private tensors and rejects authored workspace
mutation or rebinding before a
plan can be used eagerly, under ``torch.func``, or under compilation. The same
purity probe rejects unregistered Python-state mutation and implicit global RNG
consumption; evolving or random values must be explicit tensor state or
``CARRY``.

``prepare`` may be called inside ``torch.inference_mode()``; Dendra
materializes ordinary versioned workspace tensors so freshness checks remain
effective. The supplied parameter and constant tensors must themselves be
ordinary tensors created outside inference mode.

Freshness follows ordinary PyTorch tensor identity and version-counter
semantics. Mutations made through standard PyTorch operations are detected or
invalidate the relevant prepared plan without requiring a full value scan on
every timestep. Direct writes through ``Tensor.data``, raw storage, or other
mechanisms that deliberately bypass PyTorch version tracking are outside the
functional API contract and are not guaranteed to be detected. After such an
unsafe edit, construct fresh tensors and prepare or lower the model again before
continuing.

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

Explicit Population batches
---------------------------

Functional execution preserves leading axes created by ``Population.batch()``,
including repeated batching. Mutable state and stimulation may vary by
replica, while raw dynamics parameters and geometry constants are shared,
matching the ordinary Population contract. Time and the retained
duration remainder are also scalar values shared by all replicas.

Direct ``ve`` and ``intra`` tensors use the same broadcasting order as
Population execution. Ordinary trailing PyTorch broadcasting is tried first;
only if that fails may a low-rank value broadcast over the explicit batch
shape. Consequently, when batch and compartment sizes coincide, ``[C]`` means
a shared compartment profile. Use ``[B, 1, 1]`` for an explicitly per-batch
value. ``RolloutInput`` adds the leading time axis to these per-step forms.

The functional API does not accept manually installed per-replica geometry
tensors, including native-Cable, scalar-Tree, or ExtCellTree area and
resistance. Use shared core geometry inside ``Population.batch()`` or an outer
``torch.vmap`` over complete functional calls when geometry itself must vary
between lanes. Authored ``STATE`` and ``CARRY`` updates must retain their
initialized shape, dtype, and device across timesteps.

Stimulation
-----------

Use the same stimulation authoring forms for imperative and functional
execution. Intracellular injection remains registered on a Population with
``model[slice].inject(waveform)``:

.. code-block:: python

   stimulus = dn.sin(amp=0.2, freq=1.0)
   model[:, 0].inject(stimulus)
   model.initialize()

   fmodel, tensors = dn.func.make_functional(model, dt=0.01)
   assert fmodel.intra.enabled

Registered solver-owned injections are lowered automatically. Their waveform
leaves are exposed in ``tensors.parameters`` under
``stimulation.intra.waveforms.*`` and are consumed by ``step`` and ``rollout``
without an explicit ``intra`` tensor. The read-only ``fmodel.intra`` view
reports whether injection is enabled and exposes its parameter and constant
names. For example, an amplitude can be replaced while reusing the prepared
model workspace:

.. code-block:: python

   prepared = fmodel.prepare(tensors.parameters, tensors.constants)
   parameters = dict(tensors.parameters)
   amp_name = tensors.parameter_name("amp", within="intra")
   parameters[amp_name] = 1.25 * parameters[amp_name]
   state, _ = fmodel.rollout(
       parameters,
       prepared,
       tensors.state,
       steps=100,
   )

Extracellular stimulation accepts the same ``extra=(field, waveform)`` or
multicontact list used by ``Population.step``, ``run``, and ``longrun``:

.. code-block:: python

   extra = [(field_a, waveform_a), (field_b, waveform_b)]
   state, _ = fmodel.rollout(
       tensors.parameters,
       prepared,
       tensors.state,
       steps=100,
       extra=extra,
   )

The compiled convenience entry point accepts this specification directly:

.. code-block:: python

   runner = dn.compile(
       model,
       dt=0.025,
       steps=800,
       extra=(field, waveform),
       execution="scan",
       inference=False,
       trainable=(
           "stimulation.extra.contacts.0.field",
           "stimulation.extra.waveforms.0.amp",
           "stimulation.extra.waveforms.0.freq",
           "stimulation.extra.waveforms.0.phase",
       ),
       callbacks={"trace": dn.func.Recorder(("v",))},
   )
   optimizer = torch.optim.Adam(
       [p for p in runner.parameters.values() if p.requires_grad],
       lr=1e-4,
   )
   optimizer.zero_grad()
   final_state, auxiliary = runner()
   loss = auxiliary["callbacks"]["trace"]["v"].square().mean()
   loss.backward()
   optimizer.step()

This example assumes a waveform with ``amp``, ``freq``, and ``phase`` leaves,
such as ``dn.sin``. Inspect ``runner.parameters`` for the canonical names of
other waveforms or contacts. ``trainable`` uses exact or unambiguous partial
names, and may also select model parameters. Even fields originally authored
as ordinary tensors become selectable stimulation leaves.

Optimize the independent tensors in ``runner.parameters``, not the original
field or Waveform object: runner construction snapshots and detaches authoring
inputs. The mapping is read-only, but an optimizer can update its selected
tensors. Alternatively, pass a complete replacement mapping through
``runner(parameters=parameters)``. Non-leaf replacements retain their autograd
history, so a field computed from trainable electrode coordinates can propagate
gradients to those coordinates. Updating numerical values preserves the bound
contact structure; changing that structure requires a new runner.
Omitting ``trainable`` selects no differentiable leaves, and ``inference=True``
suppresses gradients regardless of that selection. Field values multiplied by
waveform values must produce mV; analytic unit-current fields in mV/mA pair
with waveform amplitudes in mA. Hard waveform timing switches remain
piecewise/discontinuous; selecting their timing leaves does not smooth those
boundaries.

Bound ``extra`` works with scan, checkpointed training scan, and inference-only
while execution. Waveforms follow the accepted simulation clock, including
explicit continuation and checkpoint replay. Do not pass an explicit
``RolloutInput.ve`` alongside bound ``extra``. Constructor binding requires
Waveform temporal specifications; for already sampled stimulation, assemble a
time-first mV tensor and pass ``runner(inputs=dn.func.RolloutInput(ve=ve))``
instead. Both routes preserve the model's ordinary scalar-cable or
finite-extracellular interpretation of the supplied potential.

The same ``extra=extra`` keyword is accepted by ``fmodel.step`` and by
``dn.func.run``, ``longrun``, and ``longrun_checkpointed``. Dendra evaluates
and transfers the authored fields and waveforms as needed; callers do not need
to assemble a separate ``ve`` representation.

For ``torch.func`` transforms or ``torch.compile``, bind a Waveform-based
extracellular specification while lowering. This keeps structural object
lowering outside the tensor transform while retaining the same ``extra``
authoring form:

.. code-block:: python

   fmodel, tensors = dn.func.make_functional(
       model,
       dt=0.01,
       extra=extra,
   )
   prepared = fmodel.prepare(tensors.parameters, tensors.constants)
   state, _ = fmodel.rollout(
       tensors.parameters,
       prepared,
       tensors.state,
       steps=100,
   )

The bound field, waveform parameters, and fixed waveform buffers appear under
``stimulation.extra.*`` in ``tensors.parameters``. They are explicit tensor
inputs to transformed or compiled calls and, like intracellular waveform
leaves, may change without another ``prepare`` call. ``fmodel.extra`` provides
the corresponding read-only plan view. Finite tensor-temporal specifications
remain runtime ``extra`` inputs: a standalone bound ``step`` has no implicit
sequence cursor.

Do not specify two sources for the same drive. Registered functional intra
cannot be combined with an explicit ``StepInput.intra`` or
``RolloutInput.intra``; raw or bound ``extra`` cannot be combined with explicit
``ve``; and runtime ``extra`` cannot be combined with an ``extra`` already
bound by ``make_functional``.

Functional waveforms must be deterministic and side-effect-free when
evaluated. All deterministic built-in Waveforms are supported, including
compositions, fixed-schedule Poisson waveforms, and ``arbitrary`` sampled
waveforms. The latter remains compatible with ``jacrev``, ``jacfwd``, ``vmap``,
and compiled functional calls.

Dendra fails closed on mechanism-owned injections, waveform hooks, and Poisson
waveforms with ``randomize_every_call=True``. A fixed Poisson schedule is
supported as an explicit tensor leaf. To choose another random schedule, call
``regenerate_schedule_()`` before lowering and keep
``randomize_every_call=False``. Other explicit setup operations, such as
``expand()`` and ``reshape_for_intra()``, are likewise outside the evaluation
purity contract and should be completed before ``make_functional``.

A custom Waveform must opt into the audited contract by declaring
``FUNCTIONAL_PURE = True`` directly on its concrete class. An inherited marker
is not sufficient, because a subclass may change evaluation semantics.
Lowering recursively checks the complete waveform module tree for registered
tensor or Python-state mutation and RNG use. When the same authored Waveform is
needed for both registered intra and bound extra, use distinct instances so
their explicit parameter namespaces cannot silently diverge.

Host-side simulation
--------------------

Use ``dn.func.run`` to advance an explicit state through a binding. Bindings
carry the functional model, parameters, preparation, and number of timesteps
per call. Prepare once for a forward/backward graph and bind an eager step:

.. code-block:: python

   prepared = fmodel.prepare(tensors.parameters, tensors.constants)
   eager_step = fmodel.bind(tensors.parameters, prepared)
   drives = dn.func.RolloutInput(
       ve=ve_over_time,
       intra=intra_over_time,
   )
   final_state, auxiliary = dn.func.run(
       eager_step, tensors.state, inputs=drives,
   )

The returned ``BoundPopulation`` is also callable directly as
``eager_step(state, StepInput(...))``. Its parameter mapping is a read-only
snapshot of the keys and tensor references: replacing a leaf in the original
mapping does not change an existing binding. Create a new binding after
parameter replacement. The tensors themselves remain differentiable, and
in-place edits to preparation inputs trigger the usual freshness checks.

For compiled execution, bind a reusable fixed-width kernel:

.. code-block:: python

   compiled_chunk = fmodel.compile_rollout_chunk(8)
   bound_chunk = compiled_chunk.bind(tensors.parameters, prepared)
   final_state, auxiliary = dn.func.run(
       bound_chunk,
       tensors.state,
       inputs=drives,
       checkpoint_every=32,
   )

Here each compiled call consumes eight timesteps, and each checkpoint spans
32 timesteps. To group execution without checkpointing, use
``host_span_steps=32`` instead. These two options are mutually exclusive;
omitting both uses one host span covering the resolved horizon. The runner
fills each span with full compiled calls and a shorter cached tail. For example,
a four-step kernel and a ten-step span execute as ``4 + 4 + 2`` steps. A span
shorter than the kernel uses a shorter specialization. Neither setting
truncates gradients.

Keep ``compiled_chunk`` across optimization iterations. Prepare and bind again
for each forward/backward graph; kernels cache shorter specializations but do
not retain bindings or differentiable preparation. A direct
``bound_chunk(state, RolloutInput(...))`` call consumes exactly
``bound_chunk.steps`` timesteps.

The existing ``run(fmodel, step, state, ...)``, ``longrun`` and
``longrun_checkpointed`` forms remain supported, including their original
``tstop`` and ``chunklength`` conventions. They accept explicit bindings and
existing ``functools.partial`` forms. A general callable in those legacy forms
must accept ``(state, StepInput)`` and return exactly
``(next_state, auxiliary)`` for one transition from the same model.

With PyTorch 2.14 or newer, ``execution="scan"`` offers an
experimental alternative for first-order training over longer horizons:

.. code-block:: python

   compiled_chunk = fmodel.compile_rollout_chunk(
       ve_over_time.shape[0], execution="scan"
   )
   bound_chunk = compiled_chunk.bind(tensors.parameters, prepared)
   final_state, auxiliary = dn.func.run(
       bound_chunk, tensors.state, inputs=drives,
   )

Scan compiles a loop whose graph size does not grow with the number of
timesteps. Setting the chunk width to the run's horizon executes all its
timesteps in one compiled call, including callback updates. Preparation and
runner validation remain outside that call. Shorter chunks and checkpointed
runners use the same scheduling described above. Speed and memory use depend
on the model and recording requirements; scan does not guarantee an improvement
over the default compiled chunks.

Scan uses Dendra-maintained correctness protection without modifying installed
PyTorch files. It preserves positional cotangents for integer and Boolean
outputs, discrete input values during joint-graph capture, and ownership of
backward subgraph inputs. Activation checks the required compiler interfaces
and behavior instead of exact patch numbers or whole-source-file fingerprints.
It replaces only the HOP partitioner's joint-graph entry point with privately
bound backward and materialization helpers; shared Torch utilities are unchanged.
This process-wide activation persists for backward and checkpoint replay after
forward execution returns. Incompatible compiler contracts raise a descriptive
``FunctionalizationError`` rather than silently selecting horizon-unrolled
execution. Copy scheduling and wrapper-allocation policy are upstream Inductor's.

Scan supports ordinary first-order ``backward()`` and ``autograd.grad``.
``create_graph=True`` raises explicitly; use the ordinary functional
``rollout`` for higher derivatives. Calls under ``torch.func`` transforms or
direct forward AD use the existing eager recurrence, so those compositions
remain available but do not gain scan compilation. Integer callback counters
can keep their integer dtype.

For explicit bindings and the supported exact ``functools.partial`` forms,
Dendra verifies ownership, source structure, state, parameters, and
prepared-workspace freshness at runner/chunk boundaries, then calls the owned
tensor kernel. Custom wrappers remain fully checked on every public call and
must still advance exactly one timestep. Wrapping a multi-step compiled chunk
in another callable does not opt it into chunk scheduling.

Functional callbacks
~~~~~~~~~~~~~~~~~~~~

Functional callbacks are a separate, pure API; they do not run or translate
the hooks on an imperative ``Callback`` object. Give each callback a stable
result name and bind the collection once, outside repeated simulations. The
functional ``Recorder`` deliberately keeps the familiar state-selection
arguments of the ordinary recorder:

.. code-block:: python

   callbacks = fmodel.make_callbacks(
       {
           "trace": dn.func.Recorder(["v"]),
           "anomalies": dn.func.AnomalyDetector(),
           "raster": dn.func.Raster(node_check=[0, -1]),
           "spike_count": dn.func.APCount(node_check=[0, -1]),
           "active": dn.func.Active(node_check=[0, -1]),
       }
   )
   final_state, auxiliary = dn.func.run(
       eager_step,
       tensors.state,
       duration=5.0,
       host_span_steps=100,
       callbacks=callbacks,
   )
   voltage = auxiliary["callbacks"]["trace"]["v"]
   anomalous = auxiliary["callbacks"]["anomalies"]
   raster = auxiliary["callbacks"]["raster"]
   spike_count = auxiliary["callbacks"]["spike_count"]
   active = auxiliary["callbacks"]["active"]

For ``torch.func`` transforms, keep host-duration scheduling outside the
transformed function and use the fixed-step tensor operation directly:

.. code-block:: python

   final_state, auxiliary = fmodel.prepare_and_rollout(
       parameters,
       tensors.constants,
       tensors.state,
       steps=1000,
       callbacks=callbacks,
   )
   voltage = auxiliary["callbacks"]["trace"]["v"]

For a fresh run, ``Recorder`` returns the initial value followed by one value
after every step. This includes one initial sample for a zero-step run. Its
recording remains connected to autograd. ``AnomalyDetector`` instead emits no
trajectory: it keeps and returns a cumulative Boolean mask using constant
callback memory. Callback objects are immutable configuration. Functional
callbacks work with eager callables, bound compiled chunks, and
``longrun_checkpointed``. With a bound compiled chunk, the runner compiles
the simulation steps and callback updates together. Callbacks still observe
every timestep: increasing the compiled chunk size does not subsample a
recording or skip reducer updates. Checkpoint replay recomputes each pure host
span without mutating external storage.

Membrane current and lead-field recording
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Construct the source Population inside ``dn.ctx(IMEM=True)`` to enable its
integrator's membrane-current output. This works for all admitted scalar and
finite-extracellular topologies, including ``MultiPopulation``. Select
``"i_membrane"`` with an ordinary functional Recorder:

.. code-block:: python

   from dendra.models.mod import hh

   with dn.ctx(IMEM=True):
       model = dn.SingleCompartment(N=1)
       model.insert(hh)
   model.initialize()
   runner = dn.func.compile(
       model,
       dt=0.025,
       steps=800,
       callbacks={"trace": dn.func.Recorder(("v", "i_membrane"))},
   )
   final_state, auxiliary = runner()
   current_mA = auxiliary["callbacks"]["trace"]["i_membrane"]

``dn.func.LFP(lead_fields)`` records the lead-field projection directly, without
emitting a full membrane-current history. ``lead_fields`` is a nonempty list
of equally shaped tensors, one per recording contact. Fields in mV/mA
(numerically ohms) multiplied by membrane current in mA give recorded mV.
The default projection sums physical cell and compartment axes, preserving
explicit batch axes. Its output has shape ``(samples, *batch, contacts)``.
Each field must broadcast to the model voltage shape; an unbatched field can
therefore be shared across batches. Fields are moved to the functional plan's
device and dtype without detaching their gradients.

.. code-block:: python

   runner = dn.func.compile(
       model,
       dt=0.025,
       steps=800,
       inference=False,  # Differentiable scan; omit for inference-only use.
       callbacks={"lfp": dn.func.LFP(lead_fields)},
   )
   final_state, auxiliary = runner()
   potential_mV = auxiliary["callbacks"]["lfp"]

For a custom projection, ``LFP(lead_fields, rule=...)`` uses
``torch.einsum(rule, current, torch.stack(lead_fields))`` exactly. The contact
axis is first in the stacked fields. Unlike this explicit rule, the default
never sums explicit batch lanes. For unbatched models it matches the ordinary
imperative LFP callback. Both LFP and Recorder include the initial boundary
and accepted post-step values, support callback continuation, and work with
scan, checkpointed scan, and inference-only while execution. Lead fields can
be differentiated through scan and checkpoint replay.

Named cells and regions in a MultiPopulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``MultiPopulation`` state and unselected Recorder output remain flat packed
tensors, not dictionaries keyed by component name. The callback mapping's keys
name result streams; they do not select model components. Use the existing
Population Slice labels to select a component explicitly:

.. code-block:: python

   with dn.ctx(IMEM=True):
       cell_a = dn.SingleCompartment(N=1, C=3)
       cell_b = dn.SingleCompartment(N=1, C=2)
       cell_a.insert(hh)
       cell_b.insert(hh)
       cell_b[:, 0].label("soma")
       packed = dn.concat_models({"cell_a": cell_a, "cell_b": cell_b})
       packed.batch(2)
   packed.initialize()
   runner = dn.func.compile(
       packed,
       dt=0.025,
       steps=800,
       callbacks={
           "cell_a": dn.func.Recorder(("v", "i_membrane"), region=packed.cell_a),
           "cell_b_soma": dn.func.Recorder(
               ("v", "i_membrane"), region=packed.cell_b.soma
           ),
       },
   )
   final_state, auxiliary = runner()
   a_voltage = auxiliary["callbacks"]["cell_a"]["v"]
   soma_current = auxiliary["callbacks"]["cell_b_soma"]["i_membrane"]

``region`` must be a Slice of the exact source Population; names can also be
resolved with ``getattr(packed, name)``. Selection is bound once and snapshots
its indices. It preserves explicit batch selection and repeated indices.
Component names select the whole component, which may itself contain several
cells. To name one of those cells, label it before packing (for example,
``cells[1, :].label("cell_1")``), then use ``region=packed.cells.cell_1``.
Packing flattens a component's physical cell and compartment axes; indexing a
packed Slice does not restore its original cell-axis layout.
Regions select physical coordinates for ``v``, ``i_membrane``, and ``vc``;
``vc`` retains all three circuit-potential channels. Scalar ``t`` may be
recorded alongside these and is left unchanged. Mechanism-local states such
as ``hh.m`` do not yet support ``region`` because their sparse support need
not match physical voltage coordinates. They remain recordable without it.

``region`` cannot be combined with ``node_indices``, ``max_only``, or
``partition``. Those existing arguments retain their final-axis semantics;
in particular, ``node_indices`` applied to ``vc`` selects circuit channels,
not physical compartments. Use ``region`` when selecting compartments or
named cells, especially with explicit batches.

Threshold detection
^^^^^^^^^^^^^^^^^^^

``Raster``, ``APCount``, ``Active``, ``ActiveAL``, and ``ActiveALCount`` share
the imperative threshold callbacks' upward-crossing rule: a selected voltage
at or above ``threshold`` fires only while its crossing latch is armed. The
initial model frame is not inspected. The first checked post-step sample is
armed even if it is already above threshold; a subsequent below-threshold
checked sample rearms that site. The first post-step sample has callback index
zero, and the checked window uses
``int(t_start_check / dt) <= index < int(t_end_check / dt)``. Omitting
``t_end_check`` leaves the window open. These bounds use callback sample indices,
not a comparison against the state's post-step clock. ``dt`` defaults to the
functional timestep and must match it when supplied explicitly.

For ``K`` selected slots in ``node_check``, the finalized results are:

.. list-table:: Functional threshold results
   :header-rows: 1
   :widths: 22 53 25

   * - Callback
     - Result
     - Shape
   * - ``Active``
     - At least one checked site has crossed at least once.
     - ``(*batch, N)`` Boolean
   * - ``ActiveAL(at_least=k)``
     - At least ``k`` checked sites have each crossed at least once.
     - ``(*batch, N)`` Boolean
   * - ``ActiveALCount(at_least=k)``
     - At least ``k`` total crossing events across the checked slots.
     - ``(*batch, N)`` Boolean
   * - ``APCount``
     - Cumulative upward-crossing counts for each checked slot.
     - ``(*batch, N, K)`` int64
   * - ``Raster``
     - Post-step crossing events, with ``False`` outside the checked window.
     - ``(steps, *batch, N, K)`` Boolean

``ActiveAL`` counts sites, not repeated spikes at one site. In contrast,
``ActiveALCount`` sums all events, including repeated spikes. This distinction
matches the actual imperative ``is_active`` reductions. The Active family
accepts ``inv=True`` to invert the finalized Boolean result, and a keyword-only
``partition`` configuration to reduce contiguous groups of the selected slots
independently. ``at_least`` must be a positive integer; each partition length
must be at least ``at_least``, matching the existing partition contract.
For example, ``node_check=[0, 2, 5, -1], partition=(2, 2)``
reduces two pairs of checked slots and returns ``(*batch, N, 2)``. Partitions
describe this selected-slot axis, not anatomical branches or Population groups;
they are static callback configuration, not a post-run ``is_active`` argument.
The selected-slot order follows ``node_check``; repeated node indices retain
their multiplicity rather than being deduplicated.

``Active``, ``ActiveAL``, ``ActiveALCount``, and ``APCount`` emit no trajectory
and use callback memory independent of the timestep count. ``Raster`` retains
the full simulation timeline, unlike imperative Raster, which omits frames
outside its checked window. All threshold callbacks preserve their crossing
latches, accumulated results, and timing phase in explicit callback carry, so
continuation does not reset the detection window or double-count a crossing.
``runner()`` starts a fresh detection history; continue with both returned
simulation state and ``callback_state=auxiliary["callbacks"].state``.

These are observation/reduction callbacks: they do not stop the simulation
early, execute imperative hooks, or convert an existing mutable callback.
Boolean masks and integer crossing counts are hard, nondifferentiable results.
They do not interfere with gradients through recorded floating-point traces or
the final simulation state, but do not provide gradients of threshold decisions.
See ``examples/functional_threshold_callbacks.py`` for the convenience runner.

The same callback plan can observe its source model through the ordinary
``Population.longrun_checkpointed`` API. This is useful when an existing
imperative simulation needs differentiable recordings or explicit reducer
carry without being rewritten as a functional runner:

.. code-block:: python

   loss, callback_results = model.longrun_checkpointed(
       tstop=5.0,
       chunklength=100,
       dt=0.01,
       functional_callbacks=callbacks,
   )
   voltage = callback_results["trace"]["v"]

The plan must come from the ``fmodel`` made from that exact ``model``; the
bridge therefore applies only to Populations already admitted by
``make_functional``.
Ordinary ``callbacks=...`` may be supplied at the same time and retain their
existing hook and scalar-loss behavior. With ``functional_callbacks``, the
native result is appended to the usual return value: ``(loss, results)``, or
``(loss, final_state, results)`` when ``return_final_state=True``. Continue a
segmented imperative run with
``functional_callback_state=callback_results.state``. Checkpoint replay
threads the pure carry explicitly, and gradients through either an ordinary
callback loss or a functional callback result preserve the live model across
ordinary, ``autograd.grad``, and higher-order checkpoint replay.
The bridge supplies the standard Population step auxiliary mapping (currently
``{"v": ...}``); a callback that requires custom auxiliary keys belongs with
the corresponding user-authored ``dn.func`` step.

``auxiliary["callbacks"].state`` is explicit callback carry. Pass it back as
``callback_state=...`` when continuing: Dendra skips callback initialization,
``Recorder`` does not duplicate the segment boundary, and reducers such as
``AnomalyDetector`` retain their accumulated result. A resumed zero-step call
therefore returns an empty Recorder segment and the unchanged reducer result.
Treat callback state as an output/carry object and do not construct or edit it
manually; its static plan identity is part of the ``torch.func`` and compiler
contract.

Users can implement :class:`dn.func.FunctionalCallback` directly. Its three
methods form a pure reducer contract:

.. code-block:: python

   class VoltageEnergy(dn.func.FunctionalCallback):
       def initialize(self, state, auxiliary):
           del auxiliary
           v = state["integrator"]["v"]
           return v.new_zeros(()), None

       def update(self, carry, state, auxiliary):
           del auxiliary
           v = state["integrator"]["v"]
           return carry + v.square().mean(), None

       def finalize(self, carry, emissions):
           del emissions
           return carry

   callbacks = fmodel.make_callbacks({"energy": VoltageEnergy()})

``initialize`` returns explicit carry and an optional initial emission;
``update`` returns the next carry and an optional post-step emission; and
``finalize`` turns the carry and stacked emissions into the public result.
Carry, emissions, and results must be tensor PyTrees with stable structure.
Carry and individual emissions must also preserve each leaf's shape, dtype,
and device; a finalized recording may vary only along its leading sample axis.
Results from callbacks that emit no samples must preserve their complete leaf
shapes.

Use integer tensors for discrete sample indices and counts, and Boolean
tensors for masks. These leaves do not need gradients. Keep accumulated losses
as floating-point tensors connected to model state, and use tensor indexing
rather than converting an index with ``.item()``.

If a callback emits only from ``update``, its ``finalize`` method receives
``None`` for a zero-sample segment and should return the same result structure
with an explicitly shaped empty tensor where appropriate.
Implementations must be deterministic and side-effect-free: do not mutate
callback objects or input tensors, perform I/O, use hidden randomness, call
``.item()``, or branch in Python on tensor values. Use built-in containers or
explicitly registered PyTree types for custom carry. These rules preserve
autograd, ``torch.func``, compilation, and checkpoint replay.

Fixed, detached observation tensors may be stored as immutable callback
configuration and are shared across ``vmap`` lanes. Trainable, lane-specific,
or per-call data should instead be represented by an explicit tensor input;
do not hide evolving values on the callback object.

The functional ``Recorder`` currently supports explicit transition-state
names such as ``"v"``, ``"vc"``, ``"i_membrane"``, ``"t"``, and ``"hh.m"``,
plus ``node_indices``, ``max_only``, and ``partition`` for non-scalar states,
or physical Slice selection through ``region``. ``Recorder(dt=...)``,
sliding-window postprocessing, and HDF5 storage are not part of its current
pure contract; apply tensor reductions or storage after the run instead.

Functional optimization
~~~~~~~~~~~~~~~~~~~~~~~

Optimizer-owned leaves should be independent of the source Population. The
recorded-trace objective below accepts only the selected parameter PyTree, then
merges it into the complete mapping and performs preparation inside the
transform. This avoids asking ``torch.func`` to differentiate unrelated leaves:

.. code-block:: python

   import torch

   parameter_names = (
       tensors.parameter_name("gnabar", within="model"),
       tensors.parameter_name("gkbar", within="model"),
   )
   parameters = tensors.independent_parameters(
       trainable=parameter_names,
       within="model",
   )
   selected = {name: parameters[name] for name in parameter_names}
   fixed = {
       name: value
       for name, value in parameters.items()
       if name not in selected
   }

   optimizer = torch.optim.Adam(
       selected.values(),
       lr=5e-3,
   )
   callbacks = fmodel.make_callbacks({"trace": dn.func.Recorder(["v"])})

   def objective(local_selected):
       local_parameters = {**fixed, **local_selected}
       initialized = fmodel.initialize(
           local_parameters,
           tensors.constants,
           tensors.initialization,
       )
       _, auxiliary = fmodel.prepare_and_rollout(
           initialized.parameters,
           initialized.constants,
           initialized.state,
           steps=1000,
           callbacks=callbacks,
       )
       prediction = auxiliary["callbacks"]["trace"]["v"]
       return (prediction - target_voltage).square().mean()

   gradient = torch.func.grad(objective)

   for _ in range(500):
       optimizer.zero_grad(set_to_none=True)
       gradients = gradient(selected)
       for name, parameter in selected.items():
           parameter.grad = gradients[name].detach()
       optimizer.step()

This is the functional form of the HH conductance-fitting tutorial. Pure
initialization keeps initial voltage and gate construction inside the tensor
program, so transforms also see dependencies introduced before the first
timestep. A complete runnable version is provided in
``examples/functional_gradient_descent.py``.
The compact snippet above retains the full candidate trace for clarity. The
runnable example instead demonstrates a custom ``VoltageTraceMSE`` callback
whose evolving carry is only a scalar sum and frame count; Recorder is used for
the fixed target and optional plots. This bounds callback output storage, but
the target trace remains proportional to the horizon and ordinary uncheckpointed
BPTT still retains transition activations. Use ``--checkpointed`` to trade
recomputation for activation memory.

The default example keeps the objective eager so
``torch.func.grad_and_value`` can transform it. For first-order optimization,
use ``--compiled-chunk-steps`` to compile several simulation steps and their
loss updates together, with ordinary autograd:

.. code-block:: bash

   python examples/functional_gradient_descent.py --compiled-chunk-steps 4

Add ``--checkpointed --chunklength 100`` to save a checkpoint every 100 steps
while still executing four steps per compiled call. Each loss update uses the
voltage at its original timestep. Keep the compiled chunk across optimization
iterations to reuse its compilation; the example does this automatically.
Larger chunks can take longer to compile, so short fits may be faster with
smaller chunks. ``--compiled-step`` (also spelled ``--compile-step``) is equivalent to
``--compiled-chunk-steps 1``.

The binding form ``run(bound, state, ...)`` has explicit horizon rules:

* With inputs alone, their time axis determines the number of steps. Alternatively,
  pass ``steps=N`` to execute exactly N steps, including with no explicit drives.
  Either form preserves the state's retained fractional-duration remainder.
* ``duration=...`` advances by a physical duration, including any remainder from
  a previous duration call. Incomplete timesteps are retained in
  ``state["control"]["duration_remainder"]``. Duration is an amount to advance,
  not an absolute stopping time.
* ``steps`` and ``duration`` are mutually exclusive. If either is supplied along
  with drives, their time axis must match the resolved number of steps exactly.
* The default timestep comes from the bound preparation. The model's nominal
  timestep is retained when it rounds to that same prepared value. An explicit
  ``dt=...`` must agree with the preparation. Clock advance is checked after
  every host span.

For compatibility, the legacy ``run(fmodel, step, state, ...)`` still gives
explicit drives precedence over ``tstop``. Without drives it requires ``tstop``.
The two legacy long-run forms always interpret ``tstop`` as a duration and
validate drives against it. All legacy forms default to ``fmodel.dt``. In the two long-run forms,
``chunklength`` controls host/checkpoint spacing. Use the binding form
for the explicit ``steps``, ``duration``, and span options.

Time-first ``RolloutInput`` tensors remain available for already assembled
drives. Registered intracellular and bound extracellular Waveforms are evaluated
at each accepted state clock, preserving the one-step runner's sampling even
at sharp pulse boundaries. Changing the compiled kernel's timestep count does
not change this sampling. A raw ``extra`` specification is lowered once and
evaluated at each host chunk. Tensor-valued extracellular waveforms use the
corresponding global time slice, so changing host/checkpoint spacing does not
change stimulus timing.

The outer host scheduler must remain eager: do not pass ``run``, ``longrun``,
or ``longrun_checkpointed`` through ``torch.compile`` or a ``torch.func``
transform. Compile or transform the supplied tensor transition instead. The
runners inherit the caller's gradient mode and never detach state between
steps.

Use ``checkpoint_every`` when the intermediate tensors saved for backward
use too much memory. It recomputes parts of the simulation during backward,
trading additional computation for lower storage. Supply the same pure,
deterministic, and side-effect-free step callable:

.. code-block:: python

   final_state, auxiliary = dn.func.run(
       bound_chunk,
       tensors.state,
       inputs=drives,
       checkpoint_every=32,
   )
   loss = final_state["integrator"]["v"].square().mean()
   loss.backward()

``checkpoint_every`` sets the number of timesteps between checkpoints. Shorter
chunks retain more boundary states; longer chunks can need more temporary
memory during backward. Adjust this separately from the compiled chunk size
to fit your simulation's memory budget. Gradients still span the full run.

Checkpoint replay reuses the prepared workspace and reevaluates functional
waveforms for the replayed chunk without mutating the source Population. This
is why waveform evaluation must be deterministic and pure. Imperative
checkpoint options for cloning or restoring model state are unnecessary. When
gradients are disabled, the checkpointed runner uses the ordinary chunk engine
because there is no autograd graph to reduce.

A differentiable prepared object belongs to one forward/backward graph. Create
it with gradient recording enabled, reuse it across that graph's complete run,
and prepare again for the next forward pass or after an optimizer update.

PyTorch transforms
------------------

Raw parameters remain differentiable through preparation and the transition:

.. code-block:: python

   import torch

   parameter_name = tensors.parameter_name("gnabar", within="model")

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

When both the loss and differentiated argument are zero-dimensional Tensors,
direct nesting gives the scalar Hessian:

.. code-block:: python

   def loss(gnabar):
       return final_voltage(gnabar).square().mean()

   gradient = torch.func.grad(loss)(tensors.parameters[parameter_name])
   hessian = torch.func.grad(torch.func.grad(loss))(
       tensors.parameters[parameter_name]
   )

For vector arguments, multiple positional arguments, or a selected parameter
PyTree, use ``torch.func.hessian(loss)`` or
``torch.func.jacrev(torch.func.grad(loss))`` (and ``argnums`` when selecting
among positional arguments). A mapping input produces a nested mapping of
Hessian blocks because its inner gradient is itself structured.

An eager prepared object may be shared across ``vmap`` lanes when parameters
and constants are shared. If they vary by lane, prepare inside the transformed
function.

To compile a transform, apply the transform to an atomic prepare-and-consume
function first, then compile the transformed callable. This keeps every raw
parameter and geometry dependency inside the transformed graph:

.. code-block:: python

   def atomic_final_voltage(gnabar):
       parameters = dict(tensors.parameters)
       parameters[parameter_name] = gnabar
       state, _ = fmodel.prepare_and_rollout(
           parameters,
           tensors.constants,
           tensors.state,
           steps=2,
       )
       return state["integrator"]["v"]

   compiled_jacobian = torch.compile(
       torch.func.jacrev(atomic_final_voltage),
       backend="aot_eager",
       fullgraph=True,
   )
   jacobian = compiled_jacobian(tensors.parameters[parameter_name])

This ordering also applies to ``jacfwd``, ``vmap``, and higher-order composed
transforms. A prepared workspace created outside the transformed callable may
be shared when differentiating only state or drives, but it must not be treated
as independent when differentiating the raw values from which it was derived.

Compilation
-----------

For inference over one known time axis, compile the atomic
``prepare_and_rollout`` method so coefficient preparation and consumption
remain in the same graph:

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

   # Capture only this no-drive inference signature before Dynamo starts.
   fmodel.prewarm_structured_rollout(ve=False, intra=False)
   compiled = torch.compile(compiled_final_voltage, fullgraph=True)

Opaque prepared objects cannot be passed through a user-authored
``torch.compile`` boundary. Use an atomic method as above, or use
``compile_rollout_chunk`` below; its outer Python wrapper keeps the opaque plan
and freshness checks outside the compiled tensor kernel.

With PyTorch 2.14 or newer, supported no-gradient inference can use bounded
``torch.while_loop`` chunks, so the compiled graph does not grow by one copy of
the transition per timestep. One-step graph capture is lazy and specific to
the presence of ``ve`` and ``intra``. When compiling
``prepare_and_rollout`` directly, call ``prewarm_structured_rollout`` once for
that signature *before* entering the user-authored ``torch.compile`` boundary,
as above. If capture is not prewarmed or the authored transition cannot be
captured, the call remains correct and fullgraph-compatible through the
unrolled fallback, but compilation cost and graph size grow with the rollout.
``compile_rollout_chunk`` performs this prewarm automatically on its first
ordinary no-gradient call.

Built-in callbacks in compiled inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On PyTorch 2.14 or newer, a default compiled chunk can also use a bounded
inference loop when every callback is a supported functional built-in:
``Recorder``, ``LFP``, ``AnomalyDetector``, ``Raster``, ``APCount``, ``Active``,
``ActiveAL``, or ``ActiveALCount``. Run the call under ``torch.no_grad()``;
chunks of at least four timesteps use the loop when their transition can be
captured. No Recorder is required for a reducer-only callback collection.
Recorder state selection, including ``v``, mechanism states such as ``hh.m``,
``t``, node indices, and ``max_only``
reductions, retains its ordinary meaning. The runner still returns the initial
sample followed by one sample per accepted timestep:

.. code-block:: python

   callbacks = fmodel.make_callbacks(
       {"trace": dn.func.Recorder(["v", "hh.m", "t"])}
   )
   kernel = fmodel.compile_rollout_chunk(100)
   with torch.no_grad():
       prepared = fmodel.prepare(tensors.parameters, tensors.constants)
       bound = kernel.bind(tensors.parameters, prepared)
       final_state, auxiliary = dn.func.run(
           bound, tensors.state, steps=100, callbacks=callbacks,
       )
   voltage = auxiliary["callbacks"]["trace"]["v"]  # 101 time samples
   print(kernel.execution_report().strategy)       # "compiled_while_loop"

The compiled body has bounded graph size; recording storage still grows with
the number of samples and selected states. Each invocation owns fresh recording
buffers and does not modify caller-owned state or input tensors. A resumed
callback state does not repeat the initial sample. Registered intracellular
and bound extracellular waveforms are evaluated at each accepted recurrent
clock value, just as in an eager host run.

Mixed collections of these built-ins remain eligible for the loop. Custom
callbacks, gradient-enabled calls, and short chunks retain the unrolled path.
Failed transition capture also falls back to unrolling. A final one-, two-, or
three-step tail therefore reports
``compiled_unrolled`` even when preceding chunks used ``compiled_while_loop``.
Consult the execution report rather than assuming a particular dispatch.

For an explicit reference path, select
``fmodel.compile_rollout_chunk(100, execution="unrolled")``. This keeps the
ordinary compiled recurrence even under ``torch.no_grad()`` and applies to
both recorded and final-state-only runs; it does not change the numerical
model, recording contract, or gradient availability.

Grad-enabled calls, direct forward-mode AD, and calls inside an active
``torch.func`` transform retain the differentiable unrolled path. Compile and
first call an inference workload inside the same no-gradient context. A
different time-axis length may create another PyTorch specialization, but each
prewarmed structured graph retains the bounded loop body.

For first-order BPTT, bind a fixed compiled chunk to one of the host runners
shown above. The runner schedules full chunks and cached tails while retaining
the autograd graph across every call. Choose the compiled timestep count to
balance compilation cost against per-call overhead; use ``checkpoint_every`` to
choose checkpoint spacing independently. Neither setting truncates the
gradient. ``compile_rollout_chunk`` already owns a compiled tensor boundary;
call it directly rather than compiling its Python validation wrapper. Manual
composition of fixed chunks remains available for custom schedules.

Compiled chunks currently target first-order BPTT. PyTorch's default
Inductor/AOTAutograd path does not support double backward through these
already compiled kernels. For Hessians and other higher-order derivatives, use
eager ``step``/``rollout`` or transform an atomic ``prepare_and_rollout`` and
then compile it, as shown above. When a compiled chunk is called inside
``vmap``, ``jacrev``, or ``jacfwd``, Dendra automatically runs the same tensor
transition eagerly because Dynamo/AOTAutograd cannot be nested inside an active
``torch.func`` transform; the result remains transform-compatible, but that
inner chunk does not receive compiled acceleration.

Inspecting execution
~~~~~~~~~~~~~~~~~~~~

A compiled kernel exposes its execution contract and the path selected by its
most recent successful call:

.. code-block:: python

   print(compiled_chunk.capabilities.functional_transforms)  # "eager_fallback"
   report = compiled_chunk.execution_report()
   if report is not None:
       print(report.strategy, report.steps, report.callbacks)

Strategies distinguish compiled unrolling, a captured inference ``while_loop``,
compiled scan, and eager fallback under function transforms or forward AD.
``reason`` explains applicable restrictions, such as a chunk too short to use
the inference loop. ``backend`` identifies the requested compiler backend;
compiled dispatch alone does not imply native code generation or a speedup.

The report is ``None`` before the first successful call. It covers the latest
kernel invocation, including a shorter tail or callback specialization, rather
than a complete host run. Checkpoint replay can update it during backward.
Failed calls and zero-step runs leave the previous report unchanged. Capability
fields describe Dendra's execution choices; actual compilation still depends
on the selected PyTorch backend supporting the model's tensor operations.

Runnable execution examples
---------------------------

``examples/functional_execution_modes.py`` exercises the execution options on
the same small HH model. It demonstrates explicit state threading, no-gradient
inference, compiled training, recordings, checkpointing, and function transforms:

.. code-block:: bash

   python examples/functional_execution_modes.py --mode all

The default ``aot_eager`` backend exercises compilation and differentiation
without native code generation. Select ``--backend inductor`` for optimized
native kernels. The compiled-transform examples are verified with the default
AOT eager backend; backend support can differ by transform. Choose one mode
to focus on its API:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Mode
     - Demonstration
   * - ``eager``
     - Prepare once, advance one step, continue with rollout, and differentiate.
   * - ``inference``
     - Compile a no-gradient chunk, or prewarm and compile atomic preparation plus rollout.
   * - ``atomic-compile``
     - Compile preparation and a fixed rollout together with first-order backward.
   * - ``chunks``
     - Bind a compiled three-step kernel, group ``run`` into host spans, and compare legacy calls.
   * - ``checkpointed``
     - Reuse that kernel with five-step checkpoint spans and complete run gradients.
   * - ``scan`` / ``scan-checkpointed``
     - Use a full-horizon scan kernel, with optional checkpoint subdivision.
   * - ``transforms``
     - Evaluate a gradient, Jacobian, Hessian, JVP and batched simulations eagerly.
   * - ``compiled-transforms``
     - Apply those transforms first, then compile the resulting functions.
   * - ``fallback``
     - Show that transforming an already compiled chunk uses its eager fallback.

For example:

.. code-block:: bash

   python examples/functional_execution_modes.py --mode checkpointed --backend inductor
   python examples/functional_execution_modes.py --mode scan --backend inductor

Scan modes use the same PyTorch 2.14+ baseline as the other examples. ``all``
executes them instead of silently skipping an unfamiliar patch release.
The examples hold the extracted initial state fixed. For differentiation through
initialization as well, see ``examples/functional_gradient_descent.py`` above.

Solver availability
-------------------

Native CPU Thomas and PCR execution requires a ``dendra-solvers`` release that
exports the transform-compatible ``thomas_solve_t`` and ``pcr_solve_t`` Python
functions. ``ExtCellAxon`` additionally requires the transform-compatible
``solve_bt`` block-Thomas facade, scalar ``Tree`` requires ``dhs_solve``, and
``ExtCellTree`` and finite-extracellular ``MultiPopulation`` require
``dhs_bt_solve``. Heterogeneous scalar ``MultiPopulation`` requires
``dhs_multi_solve``. The pure-PyTorch PCR
implementations remain available as correctness fallbacks without the
extension where the topology supports them; the admitted Tree and
heterogeneous packed paths have no extension-free CPU fallback.

CUDA execution
~~~~~~~~~~~~~~

CUDA functional execution is experimental. It uses the existing CUDA/Triton
kernels, not a CPU fallback: scalar Thomas/PCR for unbranched cables, scalar DHS
for Trees, three-component block Thomas/DHS for finite-extracellular models,
and packed DHS for heterogeneous scalar MultiPopulations. Finite-extracellular
MultiPopulations use block DHS on a disconnected forest with the same complete
``v``/``vc`` state as CPU execution. Plain ``Population``, ``SingleCompartment``,
and point-only packed updates are ordinary PyTorch tensor operations. A CUDA
PyTorch runtime and Triton are required for the spatial solvers; the CPU
``dendra-solvers`` extension is not required for CUDA execution.

Move/build and initialize the source Population on CUDA before lowering it.
Parameters, numerical state, prepared workspaces, topology and input waveforms
then stay on that device. The host-control ``duration_remainder`` scalar retains
its CPU float64 representation. The same ``dn.func`` calls apply, for example:

.. code-block:: python

   model = dn.Unmyelinated(
       [2.0], L=100.0, dx=10.0, device="cuda",
       integrator=dn.bwd_euler_ub(method="thomas", imem=False),
   )
   model.insert(hh)
   model.initialize()
   runner = dn.func.compile(
       model, dt=0.025, steps=800, execution="scan",
       callbacks={"trace": dn.func.Recorder(("v",))},
   )
   final_state, auxiliary = runner()

CUDA supports the same float32/float64 transition contract and explicit leading
batch axes. Solver facades supply reverse AD, JVP and ``vmap`` over numerical
operands; morphology/topology must be shared, not mapped. Compiled scan supports
ordinary first-order backward; while execution remains inference-only. CUDA
does not inherit the automatic CPU C++-wrapper preference. GPU loop execution
is not a promise of one device-resident kernel: host orchestration and
per-block synchronization may remain, particularly for ``torch.while_loop``.
Cold lowering, purity validation and host runners may synchronize to inspect
tensor values; these audits are not part of a fully device-resident optimizer.

Numerical coefficients, block workspaces and linear solves retain the model's
floating-point dtype: float32 models are not automatically promoted to float64.
Scalar Tree and packed scalar DHS transitions solve for voltage increments,
assembling axial currents from edge voltage differences. This preserves the
uniform unforced voltage mode, including zero-area junctions. CUDA elimination
accumulates each edge's Schur contribution together to avoid cancelling large
diagonal updates. Exact inhomogeneous linear ``cnexp`` state updates use
``x + dt * rate / expinv(c * dt)``, where ``c`` is the signed linear coefficient
in ``x' = source + c*x`` and ``expinv(z) = z/expm1(z)``. This retains small gate
increments and their derivatives near equilibrium. These algebraic
updates use the equivalent equilibrium-plus-transient form when
``abs(c * dt) > 0.5``, preserving small equilibria under stiff decay. Both
forms keep the same integration scheme without refinement solves.

Stiff finite-extracellular systems and nearly uniform voltages can make small
geometry gradients sensitive to roundoff, even when voltage trajectories agree.
Near-equilibrium authored rates can lose relative accuracy before the state
update, so very small timestep sensitivities can also remain fragile.
Solving for voltage increments reduces cancellation but does not eliminate
conditioning limits. When accuracy of small sensitivities matters, compare
against a model explicitly constructed or converted to float64 before calling
``dn.func.make_functional`` or ``dn.func.compile``.

Branched Triton DHS kernels use atomic reductions. Last-bit scheduling
differences can therefore trigger the existing strict repeated-output purity
audit; CUDA support does not disable or weaken that audit. This remains a
hardware-validation risk for branched models, not a verified deterministic GPU
execution guarantee.

CUDA does not remove the other admission restrictions above: no
stochastic/delayed state, Network event scheduling, SPD block solver, or arbitrary
custom geometry/integrators. MPS, float16 and bfloat16 remain unsupported.

The CUDA acceptance tests compare full trajectories and gradients against
imperative execution and CPU references, and separately exercise transforms and
compiled loops. On a CUDA machine run:

.. code-block:: bash

   python -m pytest -q tests/func/test_cuda_solver_kernels.py tests/func/test_cuda_population.py -m "cuda and not slow"
   python -m pytest -q tests/func/test_cuda_population.py -m "cuda and slow"
   python -m pytest -q tests/func/test_cuda_extcell_multi_population.py -m "cuda and not slow"
   python -m pytest -q tests/func/test_cuda_extcell_multi_population.py -m "cuda and slow"
   python -m pytest -q tests/func/test_cuda_plain_population.py -m "cuda and not slow"
   python -m pytest -q tests/func/test_cuda_plain_population.py -m "cuda and slow"
   python -m pytest -q tests/func/test_cuda_membrane_current.py tests/func/test_cuda_lfp.py -m "cuda and not slow"
   python -m pytest -q tests/func/test_cuda_membrane_current.py tests/func/test_cuda_lfp.py -m "cuda and slow"
   python -m pytest -q tests/func/test_cuda_initialization_parity.py -m "cuda and not slow"
   python -m pytest -q tests/func/test_cuda_initialization_parity.py -m "cuda and slow"
   python -m pytest -q tests/func/test_cuda_convenience_extra.py -m "cuda and not slow"
   python -m pytest -q tests/func/test_cuda_convenience_extra.py -m "cuda and slow"
   python -m pytest -q tests/func/test_cuda_convenience_initialization.py -W error -m "cuda and not slow"
   python -m pytest -q tests/func/test_cuda_convenience_initialization.py -W error -m "cuda and slow"
   python -m pytest -q tests/func/test_cuda_timestep_initialization.py -W error -m "cuda and not slow"
   python -m pytest -q tests/func/test_cuda_timestep_initialization.py -W error -m "cuda and slow"
   python -m pytest -q tests/func/test_cuda_clearance_process.py -W error -m "cuda and not slow"
   python -m pytest -q tests/func/test_cuda_clearance_process.py -W error -m "cuda and slow"

The slow lane includes real Inductor scan/while compilation. Tests are skipped
when CUDA is unavailable; collecting them on a CPU-only host is not evidence
that GPU kernels or GPU code generation have passed.

Resuming ordinary execution
----------------------------

Functional execution does not update the original Population. Commit a state
explicitly before returning to ``step``, ``run``, or ``longrun``:

.. code-block:: python

   fmodel.commit_state_(model, final_state)

The target must remain structurally compatible with the functional plan.
Commit validates the complete runtime-state layout and stages the update on a
private clone; malformed state or an incompatible target leaves the target
unchanged. For ``ExtCellAxon`` and ``ExtCellTree``, this includes both ``v`` and
``vc``. For ``MultiPopulation``, the packed parent state is committed
atomically. When the composite integrator has ``write_back=True``, component
voltage views are synchronized as part of the same transaction; with
``write_back=False``, component voltages remain unchanged, matching ordinary
packed execution.
Regional selector keys are structural and retain their exact authored order,
including duplicate placements. Regional parameter aliases must also retain
the same ordered local-slot placement. Build a new plan after changing
mechanism structure or support, dtype/device, train/eval mode, flags, or State
integration configuration. Placement should be changed through Population
insertion, deletion, and rebuilding; directly editing a compiled Mechanism's
``key`` is not a supported modeling operation.

.. _functional-fresh-initialization:

Fresh functional initialization
-------------------------------

``make_functional`` still uses an initialized Population to discover the exact
tensor schema. For an admitted declaration-driven model, a fresh state can then
be constructed without mutating that Population:

.. code-block:: python

   initialized = fmodel.initialize(
       tensors.parameters,
       tensors.constants,
       tensors.initialization,
   )

   prepared = fmodel.prepare(
       initialized.parameters,
       initialized.constants,
   )
   next_state, auxiliary = fmodel.step(
       initialized.parameters,
       prepared,
       initialized.state,
   )

``tensors.initialization.v_init`` is an independent, fully expanded voltage
tensor with the model's exact shape, dtype, and device. Replacing it supports
autograd, ``torch.func`` transforms, and ``vmap`` over initial conditions.
Initialization uses an owned contiguous copy of that value, while preserving
its autograd graph, and always starts the returned clock and initialization-time
mechanism timestep at zero. Ordinary ``Population.initialize()`` uses the same
construction-fresh clock when it is called again after a simulation.
Voltage and mechanism state retain the model dtype, including float32; fresh
initialization does not promote physical calculations to float64. Clock/control
leaves retain their separately declared schemas. When an experiment requires
initial gates to remain fixed during differentiation, construct the initial
bundle under ``torch.no_grad()`` or explicitly detach the corresponding returned
state leaves before differentiating the subsequent rollout. The initializer
itself remains differentiable with respect to its explicit inputs.
``tensors.initialization.states`` exposes insertion-time ``ic`` values as
support-local tensors. Replacing these values is likewise differentiable, and
``ic`` retains precedence over ``State.state_defaults(v, values)``. Other
declared state is inferred by ``state_defaults`` from the explicit voltage,
parameters, temperature, geometry, and derived workspaces. Declared
Ion/Material write state may instead begin from its canonical shared-field
seed.

Mechanism authors can add a pure ``initial_values(v, values)`` overlay on a
``Mechanism`` or ``State``. A State may return only its own ``STATE`` and
``CARRY`` names. A Mechanism may return flattened solver state and direct
``CARRY`` plus
owned writable Ion/Material locals. See :ref:`model-initialization` for the
complete ordering and authoring contract.

Population setup that needs to participate in functional initialization can be
registered as an explicit pure transform. The transform is a stateless
``torch.nn.Module`` with declared tensor reads, writes, and additional inputs:

.. code-block:: python

   class ShiftInitialVoltage(torch.nn.Module):
       def forward(self, voltage, offset):
           return (voltage + offset,)

   model.register_pre_initialize_transform(
       "shift_voltage",
       ShiftInitialVoltage(),
       reads=("state.integrator.v",),
       writes=("state.integrator.v",),
       inputs={"offset": torch.tensor(2.0)},
   )

The additional input is then available as
``tensors.initialization.transforms["pre.shift_voltage.offset"]``. It can be
replaced, differentiated, or batched just like ``v_init`` and explicit state.
Transforms compose in registration order and must return one tuple entry for
each declared write. Supported paths are exact ``parameters.<canonical name>``
leaves and phase-available ``state.*`` leaves. A pre transform runs before
State inference; a post transform runs after the accepted initial state is
constructed. As in imperative initialization, simulation time and the duration
remainder are reset to zero after the complete post-transform sequence.
Post transforms may read accepted shared fields through
``state.ions.<ion>.<field>`` and
``state.materials.<material>.<field>``. These references are post-only and
read-only: changing a shared field would require another concentration guard,
reversal-potential update, current evaluation, and local synchronization pass.

``initialize`` returns the complete parameter/constant/state bundle, including
any raw-parameter replacements made by transforms. Pass
``initialized.parameters``—not the earlier parameter mapping—to ``prepare`` or
a combined execution method so those updates are materialized for execution.
Arbitrary callbacks registered with ``register_pre_initialize_hook`` or
``register_post_initialize_hook`` remain available imperatively but are not
guessed or traced by the functional initializer.

Fresh functional initialization supports exact ``Population`` and
``SingleCompartment`` models using ``bwd_euler_sc`` or ``scnv``; canonical
``Unmyelinated``, ``Myelinated``, and native ``Cable`` models using
``bwd_euler_ub``; exact scalar
``Tree`` models using ``dhs``; and canonical ``ExtCellAxon``/``ExtCellTree``
models using ``bwd_euler_bt``/``dhs_bt``. Exact ``MultiPopulation`` packs use
their admitted ``bwd_euler_sc_multi``, ``dhs_multi``, or ``dhs_bt_multi``
integrator. Transition geometry and integrator restrictions still apply.
The Tree path reuses its admitted immutable geometry and support maps, copies
the explicit initial voltage across every solver node, and recomputes mechanism
state only at each mechanism's declared support. Zero-area junction voltages
therefore remain part of the voltage state without acquiring an undeclared
membrane mechanism. It performs the same initialization transaction as the
other admitted families rather than restoring cached initialized state.
The plain ``Population`` initializer uses the same pure point-model transaction
as ``SingleCompartment``; it initializes every ``(N, C)`` slot independently
and does not introduce cable coupling for ``C > 1``.
Compatible subclasses, where admitted above, must retain the audited geometry
and initialization lifecycle; arbitrary subclass hooks are not inferred. The initializer accepts
multiple stateful or stateless mechanisms over dense or regional support when every declared State
value is supplied by an insertion-time ``ic``, a canonical shared write seed,
``state_defaults``, or ``initial_values``. Dendra resets ``CARRY`` to zero
before the pure initialization overlays run.

For finite-extracellular models, initialization creates ``vc`` with
intracellular potential equal to ``v_init`` and both extracellular potentials
zero, exactly as imperative initialization does. Previous extracellular state
is not carried into a fresh run. The returned ``v`` and ``vc`` remain separate
explicit state leaves; registered pre/post transforms follow the same ordering
and state-write semantics as their imperative counterparts.

Packed initialization runs the consolidated parent's mechanism transaction and
registered transforms. It does not independently initialize the components or
run their pre/post hooks. The extracted ``initialization.v_init`` already
contains the packed component defaults in canonical physical order; replace it
with a complete packed tensor to differentiate through different initial
voltages. Component geometry and parameter-derived workspaces are rebuilt from
the explicit parameter/constant mapping before mechanism state is inferred.
The result retains the flat packed state schema used by simulation and
recording, and leaves the source parent and its components unchanged.

Initialization hooks must be deterministic, tensor-native, and free of
mutation and implicit RNG. Their ``values`` mapping provides support-visible
``celsius`` and ``diam`` plus the owning Mechanism's declared local
Ion/Material/current aliases. Prefer these explicit values to hidden mutable
state. In particular, ``values["celsius"]`` is already padded to the local
voltage rank and remains safe for empty and nonempty ``vmap`` lanes.

Canonical ``Ion`` and generic ``Material`` fields are also reconstructed from
their registered tensor initial values. Initialization preserves Dendra's
ordinary shared-field semantics: current-reading states see the provisional
pre-replacement current, concentration and Material replacements are committed
once, guards and enabled reversal-potential updates are applied, and the
returned ``iion`` fields contain the final accepted current. Material sources
and ``MaterialProcess`` dynamics remain per-step operations and do not advance
the t=0 state. Initial Ion and Material tensor parameters participate in
autograd, higher-order transforms, ``vmap``, and transform-then-compile in the
same way as other functional parameters. A direct tensor initial value may be
scalar or have any visible shape that broadcasts exactly to its complete
shared-field shape; callable-module initial sources are not yet supported.

Other functional models can still start from ``tensors.state``. Fresh
``initialize`` is unavailable for arbitrary Population or Integrator
initialization hooks, steady-state caches, custom
Ion/Material initialization behavior, callable-module Material initial
sources, unsupported MaterialProcess families or custom process lifecycle
hooks, legacy or otherwise impure VoltageProcess hooks, and other topologies. Transition
execution from an initialized model remains available wherever normal
functional admission succeeds.
