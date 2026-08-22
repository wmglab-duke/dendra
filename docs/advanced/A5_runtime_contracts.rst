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
are transferred to MPS. This does not make float64 simulation state available
on MPS; construct the model with ``dtype=torch.float32``.

For JIT execution, Dendra applies MPS compiler compatibility as per-model
``torch.compile`` options rather than changing process-global Inductor state.
It disables the unsupported non-AOT C++ MPS wrapper and limits fusion to 30
unique input/output buffers, below Metal's 31 constant-buffer ceiling. An
explicitly smaller ``max_fusion_unique_io_buffers`` value is respected. These
effective options participate in Dendra's compiled-kernel cache identity.

For the native morphology invariants that motivate these checks, see
:doc:`../basics/02a_native_morphologies`.
