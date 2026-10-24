.. _runtime-contract-validation:

A5. Runtime validation and model lifecycle
==========================================

Dendra solvers precompute workspaces from model geometry and selected runtime
parameters. Models can declare protected runtime contracts so public buffers
cannot silently disagree with those cached coefficients. The default policy is
appropriate for normal simulations; no configuration is required.

The concrete protected contract currently applies to canonical native
:class:`~dendra.models.core.Cable` models and to ``MultiPopulation`` or
``Network`` owners containing them. It does not make arbitrary mutation of
base ``Population``, legacy parameterized ``Axon``, or ``Tree`` buffers safe.

This page is primarily relevant when optimizing a manual stepping loop,
developing a solver, mutating model buffers after initialization, or managing
initialized models inside a ``MultiPopulation`` or ``Network``.

What Dendra protects
--------------------

For a native :class:`~dendra.models.core.Cable`, compiled topology and geometry
form one immutable snapshot. This includes compartment dimensions, base axial
resistivity, exact membrane areas and edge resistances, material volumes and
diffusion geometry, and the morphology fingerprint. Change the source
:class:`~dendra.models.morphology.Morphology` and compile a new ``Cable`` when
the geometry itself must change.

Some values remain configurable but are captured when a solver workspace is
built. In particular, changing ``cm``, ``cm_scale``, ``area_scale``, or
``rhoa_scale`` after initialization can make cached coefficients stale. Before
such a workspace can be reused, Dendra either rebuilds it when the direct
execution lifecycle guarantees a rebuild, or rejects execution until the
appropriate owner is reinitialized. Ordinary evolving state such as voltage
and time, documented per-step inputs, and stimuli are not frozen by this
contract.

Validation policies
-------------------

The active policy is selected with ``RUNTIME_CONTRACT_VALIDATION`` in
``dendra.ctx``. It is read at each public execution boundary.

``"versioned"`` (default)
~~~~~~~~~~~~~~~~~~~~~~~~~

This is the recommended production policy. Dendra compares inexpensive tensor
metadata—identity, PyTorch mutation version, shape and stride, storage offset,
dtype, device, layout, and storage pointer—at public execution boundaries. A
full frozen-geometry validation is repeated only after a protected tensor
changes. Inputs captured by an already-built solver workspace are also checked
before state advances.

An unchanged versioned check reads no tensor values and launches no tensor
operation, so it does not introduce a CUDA synchronization point. ``run`` and
``longrun`` validate once per public call; a user-controlled loop made from
``step`` calls crosses a public boundary on every step.

Callbacks are supported, but validation is not repeated before every internal
timestep of ``run`` or ``longrun``—nor after a callback inside a public call.
A callback must therefore not mutate protected geometry or an input captured
by the current solver workspace. For an intentional per-step change, return
control to the application between explicit ``step`` calls, use a documented
per-step input where possible, and otherwise rebuild the appropriate owner
before the next step.

``"strict"``
~~~~~~~~~~~~~

Strict mode performs the complete frozen-value validation at every public
execution boundary. It is useful while developing a solver or diagnosing
suspected geometry corruption:

.. code-block:: python

   with dn.ctx(RUNTIME_CONTRACT_VALIDATION="strict"):
       cable.run(tstop=tstop, dt=dt)

``"initialize"``
~~~~~~~~~~~~~~~~

For a trusted, performance-critical loop that controls every write, repeated
execution-boundary scans can be omitted explicitly:

.. code-block:: python

   with dn.ctx(RUNTIME_CONTRACT_VALIDATION="initialize"):
       for _ in range(n_steps):
           dn.step(cable, dt=dt)

The context must wrap the complete stepping loop because the policy is read
dynamically. This mode skips repeated execution-boundary validation, but it
does not disable mandatory checks during construction, initialization, or an
actual solver-workspace rebuild, including changes in timestep, shape, or
training mode. State and checkpoint loading or restoration and direct access
to canonical graph, area, or material geometry also retain their mandatory
checks.

Why initialization-only validation is unsafe
---------------------------------------------

``"initialize"`` is unsafe if application code, an optimizer, or an external
library can mutate protected topology or geometry buffers—or solver-affecting
inputs—between steps. Some quantities are read live while others are cached,
so mutation can otherwise produce stale or internally inconsistent results
without an early error. Leave ``"versioned"`` enabled unless the loop controls
every write and the reduced guard overhead has been shown to matter.

Mutation through ``tensor.data``, a NumPy or DLPack alias, or raw storage can
bypass PyTorch's mutation version. Such writes are unsupported under
``"versioned"``. ``"strict"`` can detect raw-alias writes that violate frozen
geometry, but raw-alias writes to otherwise mutable solver inputs remain
unsupported. Change solver inputs through ordinary PyTorch operations and
reinitialize their owner.

Tensors constructed under ``torch.inference_mode()`` do not expose mutation
versions. To retain fail-closed behavior, ``"versioned"`` falls back to a full
frozen-geometry scan at every boundary for those tensors and keeps exact value
snapshots of mutable workspace inputs. This uses more time and memory than the
ordinary fast guard. Construct ordinary tensors and use ``torch.no_grad()`` for
the lowest-overhead inference loop.

Backward-Euler voltage increments
---------------------------------

Single-compartment backward Euler, including packed independent compartments,
assembles the physical membrane and applied-current residual before adding one
voltage increment. The unbranched backward-Euler path likewise solves for an
increment with the existing exact Thomas or PCR backend, then
adds it to the old voltage. Its prepared matrix bands and row-capacitance
normalization are unchanged; unequal adjacent capacitances still require
different left- and right-row edge weights.

This formulation cancels the capacitive old-voltage terms analytically rather
than forming large absolute-voltage products and subtracting them numerically.
Axial residuals use edge voltage differences before conductance multiplication;
the old-voltage and externally applied extracellular differences remain
separate. This improves floating-point behavior near equilibrium and for weak
drives without changing the backward-Euler discretization or its local current
linearization. The final addition still rounds to the model voltage dtype; it
does not promise that every sub-ULP increment is representable.

This is shared Dendra assembly, not a requirement for a fused native solver.
Custom unbranched solver callables retain their historical absolute-voltage RHS
contract. The SPD backend also retains legacy assembly because its optional
pivot regularization need not solve the original matrix exactly; shifting the
RHS could change that existing numerical policy.
Membrane-current reporting and ion lifecycle hooks retain their existing
physical-current conventions.

Compiled kernels across initialization
--------------------------------------

Constructing a ``Population`` or constructing/rebuilding a ``Network`` does
not reset PyTorch's process-wide compiler cache. Existing compiled models can
continue running independently. Python numerical kernels and functional chunks
use separate compilation identities, so independent models do not consume a
shared method's specialization budget.

Reinitialization still reconstructs state and solver coefficients from the
current parameters. It retains compatible compiled integrator callables and
unchanged Tree topology plans. The callables read the refreshed tensors;
retaining a callable does not retain an old parameter value or autograd graph.
Changes to shape, dtype/device, topology, mechanism structure, or compiler
settings can require a new compiled specialization. External topology mutation
and failed workspace initialization retain their existing invalidation checks.

Use ``clear_jit_cache()`` to discard an owner's compiled callables explicitly.
``pickleable(reset_global_compiler=True)`` remains an explicit request to reset
the process-wide compiler cache; ordinary serialization does not require it.

Branching a native Network
--------------------------

Use ``copy.deepcopy(network)`` to branch an initialized native simulation:

.. code-block:: python

   import copy

   branch = copy.deepcopy(network)
   branch.step()

The copy owns its Network, Population, and connection clocks. Advancing either
Network does not advance the other. Dynamic clock references resolve the
current owner's time tensor, including after checkpoint restoration replaces
that tensor. Queues, scheduled-event cursors, RNG state, and exact clock anchors
retain their copied values. Shared Parameter objects within the model remain
shared within the copy, with independent storage from the source.

Stored runtime tensors are copied as detached values when they carry an
autograd graph. The copy does not differentiate through the source's earlier
trajectory; copied trainable Parameters remain independent leaves. Use explicit
functional state continuation when gradients must cross the branch boundary.

``branch.initialize(dt, t=...)`` starts a new native episode, resetting state and
delivery histories according to the ordinary initialization policy. To resume
saved traffic instead, use ``restore_dict_from_checkpoint(...)`` on a compatible
initialized Network. Native reinitialization still invalidates frozen functional
delivery plans; lower or compile again afterward. Cloning does not reset
PyTorch's process-wide compiler cache.

Functional compiled runner boundaries
-------------------------------------

An ordinary ``dn.func.compile(...)`` runner checks its source Population once
per invocation. Its internal preparation, binding and scheduling share that
invocation's source check; they still validate parameters, prepared-plan
freshness, state, inputs and callbacks. No validation exemption survives into
the next call. Independent public ``prepare``, ``bind`` and ``dn.func.run``
calls retain their own source checks. Replacing the public preparation,
binding or scheduling methods uses the fully checked public path instead.

Compiled callback chunks can borrow a validated single emission internally,
avoiding a redundant concatenation. The final public recording is still
materialized into independent storage, including when resuming with a callback
state. This does not change callback initialization, emitted frame counts or
gradient propagation, and does not change imperative ``Population.run`` or
``Population.longrun``.

Functional compiler compatibility
---------------------------------

Scan and inference-only while-loop execution share Dendra's PyTorch 2.14+
baseline. Copy scheduling and generated-wrapper allocation are left to upstream
Inductor; Dendra does not select those performance overrides by patch version.
Explicit compiler options and user hooks retain their ordinary compiler policy.

Scan retains narrowly scoped correctness protection for joint-graph capture and
backward subgraph ownership. These checks validate the required compiler
contract rather than selecting a feature menu from the PyTorch patch number.
An incompatible correctness contract raises a descriptive error; it does not
silently substitute a horizon-unrolled simulation. Numerical math, public
recording-storage ownership and imperative ``Population.run``/``longrun`` are
unchanged.

Reinitializing the correct owner
--------------------------------

After an intentional change to a solver input, rebuild every cache derived
from it before continuing:

* For a directly executed protected ``Population``—currently a canonical
  native ``Cable``—call
  ``model.initialize(populate_parameter_buffers=False)`` when an intentional
  direct buffer override must survive reinitialization.
* For a packed ``MultiPopulation``, call the same option on the packed owner so
  its concatenated fields and solver workspace are refreshed together.
* ``Network.initialize(dt)`` deliberately repopulates child parameter buffers.
  Express persistent Network values through the child population's
  parameterization or factory, then reinitialize the ``Network`` so all
  populations and coupled runtime state are rebuilt together.

Moving an initialized population to a different dtype or device also
invalidates derived solver coefficients. Direct ``Population`` and
``MultiPopulation`` execution can rebuild their workspaces lazily. A population
owned by an already-built ``Network`` requires ``Network.initialize(dt)`` so
the owner can refresh the complete runtime state.

Apple MPS supports float32 model tensors but not float64 tensors. Dendra keeps
the fractional duration carried between ``run`` calls as host-side float64
control metadata, so moving or constructing a Population on MPS does not try to
materialize that value on the accelerator. It remains part of state dictionaries
and runtime checkpoints. Time-grid and one-time geometry calculations that need
binary64 precision are likewise staged on CPU before their model-dtype results
are transferred to MPS. This includes the compartment coordinates of an
``ExtCellAxon`` constructed directly on MPS, including nonuniform compartment
lengths. This does not make float64 simulation state available on MPS;
construct the model with ``dtype=torch.float32``.

For JIT execution, Dendra applies MPS compiler compatibility as per-model
``torch.compile`` options rather than changing process-global Inductor state.
It disables the unsupported non-AOT C++ MPS wrapper and limits fusion to 30
unique input/output buffers, below Metal's 31 constant-buffer ceiling. An
explicitly smaller ``max_fusion_unique_io_buffers`` value is respected. These
effective options participate in Dendra's compiled-kernel cache identity.

For the native morphology invariants that motivate these checks, see
:doc:`../basics/02a_native_morphologies`.
