.. _mechanism-support-layouts:

Mechanism support layouts
=========================

A mechanism's *support* is the ordered set of physical compartments where it
is installed. Dendra represents that placement independently from the tensors
holding RANGE values, state, saved currents, and scratch buffers. This lets the
runtime share voltage gathers between mechanisms on the same region and, when
their current destinations and scatter semantics also agree, reduce several
local current contributions before one scatter. Mechanism equations do not
change.

Population-shaped sparse storage (experimental)
------------------------------------------------

By default, a non-contiguous sparse support keeps its historical flat local
axis. For example, a mechanism installed on ``K`` identical compartment
columns in each of ``N`` cells has local shape ``(N * K,)``. A Population can
opt into the experimental population-shaped layout:

.. code-block:: python

   import dendra
   import torch
   from dendra.models.mechanisms import Mechanism

   class MyChannel(Mechanism):
       pass

   population = dendra.Population(
       N=32,
       C=101,
       preserve_mechanism_population_axis=True,
   )
   population[:, torch.tensor([0, 10, 100])].insert(MyChannel)
   population.build()

   channel = population.mech.MyChannel
   assert channel.shape_p == (32, 3)

With leading simulation batches, parameter storage retains singleton batch
axes while runtime state uses the full batch shape. For a batch of size 8 in
the example above, ``shape_p`` is ``(1, 32, 3)`` and ``shape_f`` is
``(8, 32, 3)``.

The population axis changes the meaning of BATCH storage in the intended way.
RANGE, state, SAVE, and BUFFER fields use ``(..., N, K)``. A BATCH field uses
``(..., N, 1)``: one value per cell, shared over that mechanism's local
compartment columns. Under the legacy flat layout the same sparse mechanism
has only ``(..., 1)`` BATCH storage and therefore shares one value across all
cells.

Construction policy and precedence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The base ``Population`` default remains off for compatibility. A model family
that has audited its mechanisms, batching semantics, and checkpoints may opt in
by declaring a class default:

.. code-block:: python

   class AuditedPopulation(dendra.Population):
       preserve_mechanism_population_axis_default = True

The policy is resolved once, during ``Population`` construction, in descending
order of precedence:

#. an explicit ``preserve_mechanism_population_axis=True`` or ``False``
   constructor argument;
#. the active ``PRESERVE_MECHANISM_POPULATION_AXIS`` Dendra context policy,
   initially seeded from the environment variable of the same name; then
#. the model family's boolean
   ``preserve_mechanism_population_axis_default`` class attribute.

Thus an explicit constructor argument can override either a context-wide
experiment or an audited family default:

.. code-block:: python

   with dendra.ctx(PRESERVE_MECHANISM_POPULATION_AXIS=True):
       experimental = dendra.Population(N=32, C=101)
       legacy = AuditedPopulation(
           N=32,
           C=101,
           preserve_mechanism_population_axis=False,
       )

For process-wide configuration, set the environment variable before importing
Dendra. Only ``0`` and ``1`` are boolean environment spellings; an unset,
empty, or ``default`` value defers to the family default. Invalid spellings such
as ``true`` fail loudly during import.

.. code-block:: console

   $ PRESERVE_MECHANISM_POPULATION_AXIS=1 python simulation.py

``dendra.ctx`` also accepts booleans, ``0``/``1``, ``None``, and ``default``.
Using ``None`` or ``default`` in a nested context masks an environment-level
choice and delegates to the family default for models constructed in that
scope.

This is a construction-time snapshot, not a live runtime flag. Changing the
environment or entering another ``dendra.ctx`` after a model has been created
does not alter that model, even if ``build()`` has not run yet. Treat the
resolved value as part of the model's structural ABI: select it when creating
the model and construct a new model to change it. The resolved
``population.preserve_mechanism_population_axis`` property is read-only.
Eligible mechanism tensor shapes, BATCH meaning, selector metadata, state
dictionaries, and runtime checkpoint compatibility all depend on that
decision.

Eligibility and fallback
------------------------

The initial rollout is fail-closed. Dendra uses population-shaped storage only
when all of the following are true:

* the option is enabled on the Population;
* the compiler classifies the placement as ``shared_columns``: it contains the
  same ordered, unique explicitly indexed columns in every population row;
* the Population contains more than one row;
* the class is an ordinary distributed Mechanism; and
* its class-wide RANGE/BATCH defaults, constructor overrides, and initial
  conditions have unambiguous broadcast shapes.

All other cases silently retain their established layout and numerical path.
Dense and rectangular slice placements are not transformed by this option
because their established layout already retains its native rectangular axes.
Other fallbacks include partial or different per-cell supports, ragged
supports, duplicate/copy slots, forced-flat placements, PointProcess and
synapse types, VoltageProcess, MaterialProcess (including regional diffusion),
packed MultiPopulation models, and flat-only class-wide parameter values.

The option is accepted by ``Population`` and by model families (including
Axon and Tree subclasses) that forward Population constructor keywords.
``MultiPopulation`` currently packs each component through its existing
one-population representation, so it deliberately receives no benefit yet.

A custom mechanism whose code assigns special meaning to the historical flat
slot axis can explicitly opt out:

.. code-block:: python

   class AxisSensitiveChannel(Mechanism):
       supports_population_axis_layout = False

Single delayed-state queues are elementwise and follow the selected local
shape. Multi-stream delay queues additionally assign semantic meaning to an
axis, so they are rejected on population-shaped storage unless the audited
mechanism class declares
``supports_population_axis_multistream_delays = True``.

Support-group scheduling
------------------------

At build time the handler groups mechanisms only when their ordered physical
support *and* local tensor layout are exactly equal. The schedule then reuses a
gathered local voltage for current breakpoints, initialization, and state
advancement while preserving authored mechanism order; geometry binding
similarly gathers diameter once per support. Ion, Material, and accepted
ionic-current reads are gathered once per ``(field, support)`` group; a
Material reader that also writes the same field still receives its own clone.
During current assembly, adjacent contributions to the same current
destination and exact support are summed locally and scattered once when the
selector has unambiguous scatter semantics. Duplicate-slot supports retain
separate scatters.

For the overall initialization lifecycle, see :ref:`model-initialization`.

Pure Ion and Material READ bindings should be treated as read-only during a
mechanism callback. Equal-support readers can share the same gathered tensor;
in-place mutation would therefore be visible to another reader. A Material
READ+WRITE binding remains isolated because its local value is intentionally
mutable before the ordered write-back phase.

This is scheduling reuse, not mechanism fusion. Every mechanism keeps its own
parameters, state solver, breakpoint, RNG streams, and write semantics.
Mechanism and State hooks must treat their voltage argument as read-only
because equal-support mechanisms can receive the same gathered tensor.

Structural edits are planned through the same support compiler used by
``build()``. If a partial deletion would turn opt-in ``(N, K)`` storage into a
packed axis for a mechanism that declares BATCH parameters, BATCH random
variables, or BATCH noise, Dendra rejects the edit atomically. Packed storage
cannot retain the association between a BATCH value and its population row.
Deleting complete columns from every row remains valid when the resulting
support stays grouped; deleting the whole mechanism is always valid.

Compatibility and checkpoints
-----------------------------

The option changes the mechanism storage ABI. Ordinary ``state_dict`` entries
therefore have different shapes between legacy and population-shaped models;
do not reinterpret one layout as the other. Ordinary state dictionaries are
weights for the same compiled architecture: persistent selector keys must
match the target placement exactly, and a same-sized key for another region is
rejected. Runtime checkpoints additionally store a versioned, named support
identity and reject a checkpoint for different physical compartments or a
different local layout before mutating tensors, RNG streams, or delay queues.
Checkpoints written before support identity was introduced remain loadable
under their historical shape-only rule.

For compatibility, ``mechanism.key`` currently remains the complete ordered
flat physical key even when the structured mapper operates on only ``K``
column indices. It is structural, immutable runtime data: do not modify it in
place or load it from a differently placed model. Change placement through
``Population.insert`` / ``delete`` and rebuild. This rollout establishes the
canonical support abstraction and the ``(N, K)`` tensor contract; removing or
interning the redundant flat key is a later serialization migration.

Inspection and benchmarking
---------------------------

Compiled mechanisms expose ``support_spec`` for diagnostics. Its ``kind`` is
currently one of ``dense``, ``rectangular``, ``shared_columns``, or
``packed_flat`` (available as ``support_spec.kind.value``);
``preserves_population_axis`` reports the selected storage layout and
``runtime_local_shape`` reports the unbatched mechanism-visible shape.

``scripts/benchmark_mechanism_support.py`` compares separate and grouped
flat/axis current assembly plus separate, shared, and clone-isolated field
reads in eager or compiled execution. Its output labels the ``K``-index form
as ``target-compact``: that is the representation after the compatibility key
is removed, not current serialized ``state_dict`` size.
