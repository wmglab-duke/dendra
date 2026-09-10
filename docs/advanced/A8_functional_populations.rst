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

The topology list below describes functional *transition* support from an
initialized tensor state. Fresh functional initialization is a narrower
capability: it currently supports exact ``SingleCompartment`` and structurally
canonical ``Unmyelinated`` models, including compatible subclasses, with their
standard implicit integrators. This includes the Tigerholm/Thio models with
registered pure resting-balance transforms. Other admitted models can still
start from the extracted ``tensors.state``.

The current transition API supports initialized models with all of the
following properties:

* either an exact ``SingleCompartment`` topology with standard Population
  area geometry, an ``Unmyelinated`` topology with standard cylindrical cable
  geometry, a ``Myelinated`` model with Dendra's standard geometry and
  parameter transforms, an ``ExtCellAxon`` model with standard cylindrical
  membrane and extracellular-layer geometry, an exact ``Cable`` built from a
  canonical Morphology, a scalar ``Tree`` built from a canonical
  ``CompartmentGraph``, or an ``ExtCellTree`` built from a canonical
  ``CompartmentGraph`` with Dendra's standard finite-extracellular geometry,
  or an exact scalar ``MultiPopulation`` returned by ``dn.concat_models``
  whose components each satisfy one of the admitted scalar topology contracts
  above. Point-only packs use ``bwd_euler_sc_multi`` and mixed point, path,
  and Tree packs use ``dhs_multi``;
* zero or more explicit leading axes created with ``Population.batch()``;
* whole-Population or regional distributed mechanisms and current-producing
  ``PointProcess`` mechanisms using rectangular, shared-column, or packed
  support, including initialized regional overrides of ``RANGE`` and ``BATCH``
  parameters and regional Ion/Material access. Reads, ionic-current
  contributions, PointProcess currents, and additive Material sources support
  duplicate placements; concentration and Material replacement writes require
  injective support;
* deterministic CPU execution in float32 or float64;
* ``bwd_euler_sc`` for ``SingleCompartment``, ``bwd_euler_ub`` for
  ``Unmyelinated``, ``Myelinated``, and native ``Cable``,
  ``bwd_euler_bt(method="thomas")`` for ``ExtCellAxon``, ``dhs`` for scalar
  ``Tree``, ``dhs_bt`` for ``ExtCellTree``, or the automatically selected
  ``bwd_euler_sc_multi``/``dhs_multi`` integrator for ``MultiPopulation``, with
  membrane-current recording disabled; and
* for ``bwd_euler_ub``, a transform-compatible Thomas or PCR solver.

Solver tensors declared by nested ``State.STATE``, direct Mechanism and State
``CARRY``, saved-current mirrors, Ion and Material fields, and write-only
shared-field locals are discovered automatically. Initialization-
static workspaces declared with ``Mechanism.DERIVED_BUFFER`` or
``State.DERIVED_BUFFER`` are rebuilt during preparation from current parameters,
temperature, and the mechanism's exact support-local geometry, so they remain
differentiable without becoming explicit state. Timestep workspaces declared with
``Mechanism.TIMESTEP_BUFFER`` or ``State.TIMESTEP_BUFFER`` are rebuilt in the
same preparation from the explicit ``dt`` constant. State variable names are
not restricted to those used by the built-in HH mechanism.

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
Material replacement writes, the reserved compact rowwise support encoding,
TABLE lookups, MaterialProcesses, mechanism-owned waveform injections, delayed
or stochastic state, VoltageProcesses, imperative callback hooks, functional
callback built-ins beyond ``Recorder``, ``AnomalyDetector``, ``Raster``, and
``APCount``, other
Population topologies
(including nested ``MultiPopulation`` and packed finite-extracellular block
state), Cable/Axon/Tree subclasses with unrecognized custom geometry, stateful
or unregistered parameter transforms, custom integrator subclasses, the SPD
block solver, accelerator execution, and ``i_membrane`` recording. Row-regular
selectors authored through the normal Population indexing API are already
represented as packed support and are supported; users do not need to choose a
different input form.

The ``bigMRG``, ``smolMRG``, and ``exactMRG`` models supplied by
``dendra-models`` use the admitted ``ExtCellAxon`` path. Their registered
capacitance and axial-resistivity transforms are evaluated during functional
preparation, including their explicit tensor dependencies. Custom transforms
must register numerical dependencies as parameters or buffers and keep any
remaining Python configuration immutable.

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

``SingleCompartment`` uses the same functional calls with
``dn.bwd_euler_sc(imem=False)``. As in ordinary Population execution,
intracellular current affects its voltage, while an extracellular ``ve`` input
has no effect because there is no spatial voltage difference to drive axial
current.

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
continues to call the raw native operator directly.

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
solver, while ordinary imperative Tree execution continues to call the raw
native operator directly. As with native ``Cable``, the compiled resistance
already contains authored axial resistivity: use ``rhoa_scale`` or the explicit
canonical-resistance tensor for axial-resistance sensitivity rather than the
raw ``rhoa_param.rho`` leaf.

A canonical ``ExtCellTree`` uses the same functional calls with its default
``dhs_bt(imem=False)`` integrator on CPU; import ``dhs_bt`` from
``dendra.models.integrators`` when selecting it explicitly. Its state carries
public membrane voltage ``v`` and the complete three-potential circuit state
``vc``. The exact compiled membrane area and child-indexed intracellular edge
resistance remain the intracellular source of truth, while ``dx``, ``xraxial``,
``xc``, and ``xg`` describe the two extracellular layers. The functional
transition supports the same reverse- and forward-mode differentiation,
``vmap``, compilation, host runners, checkpointed BPTT, and commit/resume
lifecycle as the other admitted topologies. It requires the transform-compatible
CPU block-DHS solver supplied by ``dendra-solvers`` and does not currently
support ``i_membrane`` recording.

Packed scalar MultiPopulations
------------------------------

An exact ``MultiPopulation`` produced by ``dn.concat_models`` uses the same
functional API. Every component must independently satisfy an admitted scalar
contract: ``SingleCompartment``, ``Unmyelinated``/``Myelinated`` and their
admitted subclasses, exact native ``Cable``, or canonical scalar ``Tree``.
Finite-extracellular components and nested ``MultiPopulation`` objects remain
unsupported because they require a packed block-state solver.

Functional execution preserves the purpose of concatenation: the composite is
advanced as one packed system, not as a Python loop over components. Its state
and direct ``ve``/``intra`` inputs retain the ordinary flat composite shape
``(*batch, 1, total_width)``. There is no component-nested state tree. Component
geometry and physical parameters remain explicit preparation inputs, so a
gradient can still reach, for example, one component's capacitance or axial
resistivity without changing the packed runtime representation.

Component physical parameters keep names such as
``populations.<component>.*``. Packed mechanism parameters retain their normal
``integrator.*`` names, and the common temperature appears once. Geometry
constants are likewise component-namespaced, for example
``populations.axon.dx`` or
``populations.tree.canonical_area_cm2``. Calling ``prepare`` assembles those
sources into one differentiable packed workspace; it does not run the
component transitions.

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

Registered intracellular injection and bound extracellular stimulation use the
same authored forms described below and are projected onto the flat composite.
The packed transition supports the ordinary functional runners,
differentiation, ``vmap``, and transform-then-compile compositions.
Heterogeneous packs require the transform-compatible native CPU ``dhs_multi``
solver supplied by a current ``dendra-solvers`` installation.

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
   Independent geometry and timestep tensors. ``SingleCompartment``,
   ``Unmyelinated``, native ``Cable``, and scalar ``Tree`` expose ``diam`` and
   ``dx``; ``Myelinated`` exposes canonical ``diameters``, ``diam_original``, and
   ``dx``; and ``ExtCellAxon`` and ``ExtCellTree`` expose ``diam``, ``dx``,
   ``xraxial``, ``xc``, and ``xg``. Every topology exposes scalar ``dt`` in
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
   carry, clock, and duration remainder.

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

``dn.func.run``, ``dn.func.longrun``, and
``dn.func.longrun_checkpointed`` advance explicit state with a user-supplied
one-step callable or a bound fixed compiled chunk. Bind parameters and one
prepared workspace outside the runner so preparation is shared by the complete
simulation:

.. code-block:: python

   from functools import partial

   eager_step = partial(
       fmodel.step,
       tensors.parameters,
       prepared,
   )

   drives = dn.func.RolloutInput(
       ve=ve_over_time,
       intra=intra_over_time,
   )
   final_state, auxiliary = dn.func.run(
       fmodel,
       eager_step,
       tensors.state,
       drives,
   )

A general callable must accept ``(state, StepInput)`` and return exactly
``(next_state, auxiliary)`` for one transition from the same ``fmodel`` passed
to the runner. A custom wrapper is responsible for preserving this contract.
Dendra also recognizes the exact binding
``partial(compiled_chunk, parameters, prepared)`` for a chunk returned by
``fmodel.compile_rollout_chunk(C)``, where ``C >= 1``:

.. code-block:: python

   compiled_chunk = fmodel.compile_rollout_chunk(8)
   bound_chunk = partial(
       compiled_chunk,
       tensors.parameters,
       prepared,
   )

   duration = ve_over_time.shape[0] * fmodel.dt
   final_state, auxiliary = dn.func.longrun(
       fmodel,
       bound_chunk,
       tensors.state,
       duration,
       chunklength=32,
       inputs=drives,
   )

``C`` controls the number of timesteps in each compiled kernel. Independently,
``chunklength`` controls the number of timesteps in each host span and, for
``longrun_checkpointed``, each checkpoint. The runner fills each host span with
full ``C``-step calls and a shorter compiled tail when needed. For example,
``C=4`` and ``chunklength=10`` execute each full host span as ``4 + 4 + 2``
steps. A host span shorter than ``C`` uses only a shorter specialization.
``run`` uses the same scheduling within its complete resolved horizon.

Dendra caches runner kernels, including shorter specializations, on the original
chunk with its compilation options for reuse across runner calls. Keep
``compiled_chunk`` across optimization iterations; bind a newly prepared
workspace for each forward/backward graph. There is no need to construct or
schedule a tail yourself.

For the exact ``partial(fmodel.step, parameters, prepared)`` and compiled-chunk
bindings, Dendra verifies ownership, source structure, state, parameters, and
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
       }
   )
   final_state, auxiliary = dn.func.longrun(
       fmodel,
       eager_step,
       tensors.state,
       tstop=5.0,
       chunklength=100,
       callbacks=callbacks,
   )
   voltage = auxiliary["callbacks"]["trace"]["v"]
   anomalous = auxiliary["callbacks"]["anomalies"]
   raster = auxiliary["callbacks"]["raster"]
   spike_count = auxiliary["callbacks"]["spike_count"]

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

``Raster`` and ``APCount`` use the same upward-crossing rule as their imperative
counterparts: a selected voltage at or above ``threshold`` fires only when its
previous checked value was below threshold. The first checked post-step frame
is armed; the initial model frame is not emitted or inspected. ``Raster``
returns a Boolean tensor shaped ``(steps, *batch, N, K)``. It retains the full
simulation timeline and fills frames outside
``[t_start_check, t_end_check)`` with ``False``; the imperative Raster instead
omits those frames. ``APCount`` emits no trajectory and returns a cumulative
``int64`` tensor shaped ``(*batch, N, K)`` using constant callback memory.
Both preserve their latch, count, and timing phase in explicit callback carry.
Their ``dt`` must match the timestep used to create ``fmodel``; rebuild the
functional plan before using another timestep.

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
names such as ``"v"``, ``"t"``, and ``"hh.m"``, plus ``node_indices``,
``max_only``, and ``partition`` for non-scalar states. ``Recorder(dt=...)``,
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

The runners have the following duration and input rules:

* If ``run`` receives ``ve`` or ``intra``, that time axis determines the number
  of steps. ``tstop`` is ignored and the retained fractional-duration remainder
  in state is unchanged.
* Without explicit drives, ``run`` requires ``tstop``. ``longrun`` and
  ``longrun_checkpointed`` always require it. ``tstop`` is a duration to
  advance, and incomplete fixed timesteps are retained in
  ``state["control"]["duration_remainder"]`` for the next duration-based call.
* Drives passed to either long-run function must have exactly the number of
  timesteps resolved from ``tstop`` and the retained remainder.
* The runners use ``fmodel.dt`` for host duration scheduling by default. Pass
  ``dt=...`` to use another timestep, and build the supplied prepared step from
  ``tensors.constants["dt"]`` containing that same value. After every host
  chunk, the runner verifies that the model clock advanced by that timestep and
  raises instead of silently accepting a mismatched prepared step.
* Long runs use fixed ordinary-Python groups of ``chunklength`` steps followed
  by a shorter tail. ``chunklength`` controls host/checkpoint boundaries; it
  does not compile the outer simulation horizon or change the bound kernel's
  fixed timestep count.

Time-first ``RolloutInput`` tensors remain available for already assembled
drives. Registered intracellular and bound extracellular Waveforms are evaluated
at each accepted state clock, preserving the one-step runner's sampling even
at sharp pulse boundaries. Changing the compiled kernel's timestep count does
not change this sampling. A raw ``extra`` specification is lowered once and
evaluated at each host chunk. Tensor-valued extracellular waveforms use the
corresponding global time slice, so changing ``chunklength`` does not change
stimulus timing.

The outer host scheduler must remain eager: do not pass ``run``, ``longrun``,
or ``longrun_checkpointed`` through ``torch.compile`` or a ``torch.func``
transform. Compile or transform the supplied tensor transition instead. The
runners inherit the caller's gradient mode and never detach state between
steps.

Use ``longrun_checkpointed`` when the intermediate tensors saved for backward
use too much memory. It recomputes parts of the simulation during backward,
trading additional computation for lower storage. Supply the same pure,
deterministic, and side-effect-free step callable:

.. code-block:: python

   final_state, auxiliary = dn.func.longrun_checkpointed(
       fmodel,
       bound_chunk,
       tensors.state,
       duration,
       chunklength=32,
       inputs=drives,
   )
   loss = final_state["integrator"]["v"].square().mean()
   loss.backward()

``chunklength`` sets the number of timesteps between checkpoints. Shorter
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

With PyTorch 2.12 or newer, supported no-gradient inference can use bounded
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

Grad-enabled calls, direct forward-mode AD, and calls inside an active
``torch.func`` transform retain the differentiable unrolled path. Compile and
first call an inference workload inside the same no-gradient context. A
different time-axis length may create another PyTorch specialization, but each
prewarmed structured graph retains the bounded loop body.

For first-order BPTT, bind a fixed compiled chunk to one of the host runners
shown above. The runner schedules full chunks and cached tails while retaining
the autograd graph across every call. Choose the compiled timestep count to
balance compilation cost against per-call overhead; use ``chunklength`` to
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

Solver availability
-------------------

Native CPU Thomas and PCR execution requires a ``dendra-solvers`` release that
exports the transform-compatible ``thomas_solve_t`` and ``pcr_solve_t`` Python
functions. ``ExtCellAxon`` additionally requires the transform-compatible
``solve_bt`` block-Thomas facade, scalar ``Tree`` requires ``dhs_solve``, and
``ExtCellTree`` requires ``dhs_bt_solve``. Heterogeneous scalar
``MultiPopulation`` requires ``dhs_multi_solve``. The pure-PyTorch PCR
implementations remain available as correctness fallbacks without the
extension where the topology supports them; the admitted Tree and
heterogeneous packed paths have no extension-free CPU fallback.

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

Fresh functional initialization supports exact ``SingleCompartment`` models
using ``bwd_euler_sc`` and structurally canonical ``Unmyelinated`` models using
``bwd_euler_ub``. An ``Unmyelinated`` subclass may define its mechanisms,
parameterization, and registered pure initialization transforms while retaining
the standard geometry and lifecycle methods. It accepts multiple stateful or
stateless mechanisms over dense or regional support when every declared State
value is supplied by an insertion-time ``ic``, a canonical shared write seed,
``state_defaults``, or ``initial_values``. Dendra resets ``CARRY`` to zero
before the pure initialization overlays run.

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
initialization hooks, timestep workspaces, steady-state caches, custom
Ion/Material initialization behavior, callable-module Material initial
sources, MaterialProcesses, VoltageProcesses, and other topologies. Transition
execution from an initialized model remains available wherever normal
functional admission succeeds.
