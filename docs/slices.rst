.. _slice-contract:

The Slice contract
==================

:class:`dendra.models.slice.Slice` is Dendra's retained, logical selection of
locations in a :class:`dendra.models.core.Population`.  It is the common
interface for inspecting and changing selected state, labelling morphology
regions, inserting and deleting mechanisms, applying intracellular stimulation,
and specifying network endpoints.

A ``Slice`` is **not** a tensor and does not own simulation state.  Reading a
field produces a non-aliasing tensor snapshot; an explicit Slice write changes
the field owned by the population or mechanism.

Indexing and morphology lookup
------------------------------

Indexing a population follows PyTorch indexing exactly.  Integers, slices,
``...``, ``None``, integer arrays, and boolean masks have the same selection
and output-shape semantics that they have for a tensor with ``population.shape``:

.. code-block:: python

   import torch
   import dendra as dn

   cells = dn.Population(N=4, C=6)

   soma = cells[:, 0]                         # shape: [4]
   band = cells[1:3, 2:5]                    # shape: [2, 3]
   paired = cells[torch.tensor([0, 3]),
                  torch.tensor([1, 4])]      # shape: [2]
   kept = cells[torch.tensor([True, False, True, False])]

Indexing an existing Slice applies the new key to the selected result, just as
two consecutive tensor indexing operations would:

.. code-block:: python

   dendrite = cells[:, 1:]
   middle = dendrite[1:3, 2:4]

   torch.testing.assert_close(
       middle.v,
       cells.v[:, 1:][1:3, 2:4],
   )

Dendra composes the two keys back into the population's coordinate system, so
``middle`` remains a retained Slice rather than a materialized tensor.

Do not confuse indexing with morphology-name lookup.  Both return a Slice, but
the latter searches compartment names attached to a Tree or imported
morphology:

.. code-block:: python

   cells[:, 0]                       # PyTorch indexing
   cell.slice("soma")                # morphology-name lookup
   cell.slice("apic", loc=0.5)      # one location per matching section

See :meth:`dendra.models.core.Population.slice` and
:meth:`dendra.models.core.Population.find` for name-matching rules.

Shape and size
--------------

The following introspection has tensor-like meaning:

``selection.shape``
   A ``torch.Size`` equal to the result shape of the corresponding indexing
   operation.

``selection.is_scalar``
   ``True`` exactly when ``selection.shape == torch.Size([])``.

``selection.is_empty``
   ``True`` when at least one result dimension has length zero.

``len(selection)``
   The length of the first result dimension, matching ``len(tensor[key])``.
   Calling ``len`` on a scalar Slice raises ``TypeError``.

``selection.numel()``
   The total number of selected entries, including repeated entries.  A scalar
   Slice has one element; an empty Slice has zero.

``selection.flat_index``
   A one-dimensional ``torch.long`` tensor containing the selected locations as
   offsets into the flattened root population.  Its order and multiplicity
   match ``selection``.  The returned tensor is an owned snapshot, so it can be
   sampled or permuted without changing the retained Slice.  It reflects the
   population layout at access time; reacquire it after batching or moving the
   population, and unravel it against the shape from that same layout.

Consequently, ``len(selection)`` and ``selection.numel()`` differ for a
multidimensional selection:

.. code-block:: python

   region = cells[1:3, 2:5]
   assert region.shape == torch.Size([2, 3])
   assert len(region) == 2
   assert region.numel() == 6

The flat numeric form makes arbitrary reordering and subsampling explicit.  To
turn transformed flat indices back into a Slice, unravel them against the
population shape:

.. code-block:: python

   flat = region.flat_index
   order = torch.randperm(flat.numel(), device=flat.device)
   shuffled_flat = flat[order]
   shuffled = cells[torch.unravel_index(shuffled_flat, cells.shape)]

   torch.testing.assert_close(
       shuffled.v,
       region.v.reshape(-1)[order],
   )

Reading state returns snapshots
-------------------------------

Attribute access, :meth:`~dendra.models.slice.Slice.inspect`, and its
:meth:`~dendra.models.slice.Slice.get` alias return non-aliasing snapshots of
the selected state:

.. code-block:: python

   soma = cells[:, 0]
   before = soma.v
   before.zero_()             # changes only the snapshot
   assert not torch.equal(before, soma.v)

This rule is the same for basic and advanced indexing and for dense and sparse
mechanisms.  Never mutate a value returned by a Slice in order to update a
model; use :meth:`~dendra.models.slice.Slice.set` or Slice attribute assignment.
The snapshot is cloned but not detached: differentiable reads retain autograd
connectivity to trainable model Parameters.

Mechanism state can be read through either public form:

.. code-block:: python

   from dendra.models.mod import hh

   cells.insert(hh)
   cells.initialize()

   soma.inspect("m", mechanism="hh")
   soma.mech.hh.m

For a mechanism inserted everywhere, its spatial fields have the population's
full layout.  A sparsely inserted mechanism stores only its supported
locations; Slice reads materialize the requested population layout while
preserving the field's dtype.  Missing locations are represented by ``NaN``
when the dtype supports ``NaN``.  For fields without a safe missing-value
representation, Dendra requires a selection wholly supported by the mechanism
and raises an informative error otherwise.  Writes to a sparse mechanism must
always be wholly inside its insertion region; missing support is an error, not
a silently ignored write.

Writing spatial state
---------------------

There are two equivalent write forms for registered, spatially indexed fields:

.. code-block:: python

   soma.set("v", -70.0)
   soma.v = -70.0

   # Mechanism fields use either explicit or attribute-style routing.
   soma.set("m", 0.1, mechanism="hh")
   soma.mech.hh.m = 0.1

A writable field must be a registered buffer or ``torch.nn.Parameter`` whose
shape supports the Slice's population coordinates.  Dendra performs the update
under ``torch.no_grad()`` and preserves the identity, registration, dtype,
device, and unrelated entries of the owning tensor.  Values use ordinary
PyTorch assignment broadcasting: a scalar, exact-shape value, or trailing-
broadcastable value is accepted.  A non-broadcastable value raises an error
that reports both the supplied and selected shapes.

Global values do not have population coordinates and therefore cannot be read
or written through a Slice.  Access them on the owning model instead:

.. code-block:: python

   configured = dn.Population(N=4, C=6, celsius=36.0)
   configured.celsius         # model-global access
   configured[:, 0].v = -70.0 # spatial Slice assignment

Assignment is deliberately closed to unknown names.  A misspelling such as
``soma.volatge = -70`` raises ``AttributeError`` instead of creating wrapper
metadata or silently leaving the model unchanged.  Likewise, requesting a
missing mechanism alias or field, a global field, a non-spatial parameter, or
an unsupported sparse location fails explicitly.

Model-wide methods and non-spatial metadata are not implicitly Slice-scoped.
For example, ``soma.batch`` and ``soma.initialize`` raise instead of
transforming the whole population behind a region-looking expression.  Use
``soma.model.initialize()`` or ``soma.model.np`` only when whole-model access is
intentional; Slice-scoped construction uses the explicit ``inject``, ``insert``,
``delete``, ``delete_all``, and ``parametrize`` methods below.

Labels
------

:meth:`~dendra.models.slice.Slice.label` gives a reusable name to a selection:

.. code-block:: python

   cells[:, 0].label("soma")
   cells[:, 1:].label("dendrite")

   cells.soma.v = -68.0

A label must be a non-private Python identifier.  It cannot collide with a
population attribute, buffer, parameter, submodule, method, reserved Slice
attribute, or another label.  Dendra raises before changing anything when a
name is unsafe.  Pass ``replace=True`` to replace an existing Slice label
explicitly; this can never replace a non-label attribute.

A top-level label belongs to the population and is stored in
``population._labels``.  A label made from a nested Slice is owned by that
specific parent Slice and is visible through its descendants, but not through
the population or an unrelated sibling:

.. code-block:: python

   dendrite = cells.dendrite
   dendrite[:, :2].label("proximal")

   dendrite.proximal           # valid
   dendrite[0].proximal        # valid when the indices commute exactly
   # cells.proximal            # not a top-level population label

Nested-label ownership prevents a short local name from leaking into the
population namespace.  Parent and nested labels remain retained selections and
follow the same lifecycle rules as other slices.

Label ownership and propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A label always has exactly one owner.  Attribute lookup on a descendant Slice
does not copy, transfer, or register that label on the descendant.  Instead,
Dendra may *propagate* the owner's label through the descendant's retained
indexing path when the two restrictions commute exactly.  This makes the two
natural readings of a cell-by-compartment model equivalent:

.. code-block:: python

   first_soma = cells[0].soma
   same_soma = cells.soma[0]

   assert first_soma.shape == same_soma.shape
   torch.testing.assert_close(first_soma.v, same_soma.v)

This is particularly useful in network code, where ``net.hh[0].soma`` reads as
"the soma of the first ``hh`` neuron".  Leading integer, slice, integer-array,
and boolean selections can propagate when replaying them on ``cells.soma``
produces exactly the same retained root coordinates, shape, order, and
multiplicity.

Propagation is intentionally not a general-purpose overlap operation.  A key
that also restricts or rearranges the label's compartment axis, a full-rank
mask, or another cross-axis advanced index may not commute with the label.  In
that case Dendra raises ``AttributeError`` instead of guessing how to combine
the selections.  Put the label first when that is the intended operation:

.. code-block:: python

   cells.soma[cell_key]

For a true physical overlap, use the explicit
:meth:`~dendra.models.slice.Slice.intersect` operation described below.  The
distinction is deliberate: attribute propagation preserves the labelled
Slice's shape semantics only when equivalence can be proved, whereas
``intersect`` always returns its documented canonical shape.

Population-owned labels are visible through descendants of that population.
A nested label remains visible only through descendants of its owning Slice;
an unrelated sibling cannot acquire it by name.  Label lookup is resolved from
the nearest owner, and creating a label that would shadow a visible label is
rejected.  Replacement and clearing update future attribute lookup without
retargeting a Slice that was already retained in a variable.

Physical labels must be selected before entering a mechanism namespace:

.. code-block:: python

   cells[0].soma.mech.hh.m       # valid
   cells.soma[0].mech.hh.m       # equivalent
   # cells[0].mech.hh.soma       # invalid: ``soma`` labels compartments

Mechanism-local point-process banks remain governed by
:class:`dendra.models.slice.SynapseSlots`; label propagation never chooses a
slot on the user's behalf.

Explicit physical intersection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~dendra.models.slice.Slice.intersect` combines two physical selections
without weakening ordinary indexing or label semantics:

.. code-block:: python

   selected = cells[torch.tensor([3, 1, 1])]
   selected_somas = selected.intersect(cells.soma)

Both operands must be population-backed Slices from the same root population.
When the intersection is formed, the result has a canonical one-dimensional
shape, including for scalar and empty inputs.  It contains the entries of the
left operand whose physical root coordinates occur anywhere in the right
operand.  Left traversal order and left duplicate occurrences are preserved;
duplicates and ordering on the right only define membership support and cannot
multiply the result.  Thus ``left.intersect(right)`` never widens ``left``, and
a disjoint intersection is a deterministic Slice with shape
``torch.Size([0])``.

The result remains an ordinary retained Slice.  If the population is batched
*after* the intersection is formed, it gains those leading batch axes just like
any other retained selection.  Recomputing the intersection after batching
again produces a canonical one-dimensional result by flattening the current
left traversal.  This lifecycle distinction preserves both the intersection
contract at creation time and Slice's established batch-rebasing semantics.

Intersection is intentionally asymmetric in order and multiplicity, even
though its physical support is set-like.  Swap the operands only when the
other Slice's traversal order and duplicate occurrences are the desired output
contract.

Lifecycle: batching, building, and device moves
------------------------------------------------

Keeping a Slice in a variable must not silently change the region it denotes.
Dendra preserves that invariant through supported in-place model lifecycle
operations:

.. list-table:: Retained Slice lifecycle
   :header-rows: 1
   :widths: 28 72

   * - Model operation
     - Slice behavior
   * - ``population.batch(B)``
     - A retained population Slice is rebased over the new leading batch axis
       and selects the same logical cells/compartments in every replica.
       Repeated batching adds another outer batch axis.
   * - ``population.build()`` or ``initialize()``
     - Population-backed slices retain their selection.  A retained mechanism
       or other submodule Slice is rebound to the corresponding live submodule.
   * - Forced rebuild after insertion/deletion/configuration changes
     - Submodule slices are rebound by their registered path.  If that path no
       longer exists or no longer supports the selection, use raises a clear
       stale-Slice error.
   * - ``population.to(...)`` and dtype conversion
     - The selection follows the model; index tensors and returned snapshots
       use the model's active device, and field snapshots preserve field dtype.
   * - Construction of a different Population or concatenated Network
     - This is a different owner.  Existing slices continue to belong to the
       original model and are never implicitly retargeted.

The same rules apply to named, unnamed, nested, and submodule slices.  In
particular, it is safe to retain a Slice before batching:

.. code-block:: python

   target = cells[:, 0]
   cells.batch(8)

   assert target.shape == torch.Size([8, 4])
   torch.testing.assert_close(target.v, cells.v[..., 0])

Network connection calls consume the selected endpoint when the connection is
registered.  Connections registered before ``Network.batch`` are rebased with
the network; connections registered afterward use the already-batched Slice.
An endpoint from another network or an unresolvable stale submodule is rejected
rather than matched by name alone.

Slice-scoped model construction
-------------------------------

The following operations record the Slice's logical locations:

``dendra.insert_intrinsic_activity(selection, onsets, tau=0.1, gmax=0.1, e=0.0)``
   Add a conductance-driven train of alpha events to the selected physical
   compartments.  A scalar creates one event, a one-dimensional value is a
   shared event train, and a two-dimensional value has explicit
   ``(events, locations)`` layout.  Repeated calls add independent events.
   Unlike ``selection.inject(waveform)``, this changes membrane current through
   the voltage-dependent relation ``g * (v - e)`` rather than prescribing an
   absolute current waveform.  This is a structural edit; call ``initialize()``
   before running the model again.

``dendra.remove_intrinsic_activity(selection)``
   Remove all events created by ``insert_intrinsic_activity`` at the selected
   physical compartments.  Activity outside the selection and ordinary
   ``alphasynapse`` insertions are retained.  Removing absent activity and
   removing from an empty Slice are no-ops.  Reinitialize before the next run.

``selection.inject(waveform)``
   Apply an intracellular waveform at those physical compartments.  Empty
   selections are a no-op.  Repeated physical indices contribute repeatedly
   and therefore accumulate current.

``selection.insert(mechanism, ...)``
   Insert a mechanism only at those locations.  Ordinary distributed
   mechanisms collapse duplicate physical indices.  Use ``copies=...`` or
   ``preserve_duplicate_indices=True`` only when independent colocated state is
   intentional.  Mechanism layout is structural and shared by batch replicas:
   insertion through a batched Slice projects the selected replicas onto the
   union of their core cell/compartment locations.  If insertion changes an
   already-built model, call ``initialize()`` before running it again.

``selection.delete(mechanism, strict=False)``
   Remove the exact mechanism class from those physical compartments.  This is
   a structural, set-valued operation: repeated Slice indices do not request
   repeated deletion, and all copied or duplicate mechanism slots at a selected
   compartment are removed.  By default, Dendra deletes the intersection of the
   selected physical set and the mechanism's support; selected compartments
   without that class are ignored.  Pass ``strict=True`` when every selected
   compartment must support the exact class.  A strict mismatch fails
   atomically without changing insertion or parameterization records.  On a
   batched population, the Slice is projected onto the shared structural core,
   so a deletion selected through any batch replica applies to that compartment
   in every replica.  An empty Slice is a no-op.

``selection.delete_all()``
   Remove every mechanism class present on the selected physical compartments.
   Support outside the Slice survives.  All affected classes and overrides are
   planned as one transaction: if any required projection is unsafe, Dendra
   changes none of them.  Batch projection, duplicate/copy removal, empty-Slice
   behavior, and rebuild requirements are the same as for ``delete``.

``selection.parametrize(name, value, alias=None)``
   Register a spatial parameter override for the selected locations.  This is
   distinct from changing current mutable state with ``set``.  Batch replicas
   share a core-region parameterization unless the parameter itself explicitly
   defines batch-varying values.  Mechanism parameter overrides are structural
   configuration and are replayed when a forced rebuild replaces the compiled
   mechanism instance.

``selection.slots(synapse, ...)``
   Select mechanism-local point-process slots supported by the physical Slice.
   A :class:`dendra.models.slice.SynapseSlots` target is required when several
   independent synapse slots share one physical compartment.  See
   :doc:`advanced/A3_synapse_banks_and_slots`.

Mechanism attribute access requires a built mechanism hierarchy.  It is valid
to retain a population Slice before insertion, but access ``selection.mech``
only after ``build()`` or ``initialize()`` has materialized the mechanisms.

Deleting mechanisms and rebuilding safely
------------------------------------------

:meth:`~dendra.models.core.Population.delete` without an ``index`` removes the
exact mechanism class from every place it is configured in the population:

.. code-block:: python

   from dendra.models.mod import hh, pas

   cells.insert(pas)
   cells.soma.insert(hh)

   cells.dendrite.delete(pas)       # pas-support intersection
   cells.soma.delete(pas, strict=True)
   cells.delete(hh)                 # exact class everywhere
   cells.delete_all()               # every remaining class everywhere

The indexed ``Population.delete(mechanism, index=..., strict=False)`` and
``Population.delete_all(index=...)`` forms address unbatched population-core
coordinates directly.  With no ``index``, ``delete`` removes the named class
everywhere and ``delete_all`` removes every mechanism class everywhere.  Prefer
the Slice forms for restricted operations, especially after batching, because
the Slice performs the structural projection explicitly.

A restricted ``delete`` subtracts the mechanism-support intersection from every
overlapping insertion alias.  ``strict=True`` first requires the entire
selected physical set to be supported; otherwise it changes nothing.
``delete_all`` takes the support intersection independently for every class
present in the target.  It is transactional across classes: an unsafe
transformation for one mechanism leaves every mechanism unchanged.

Both operations crop persistent mechanism parameterizations that were
registered through a Slice.  Scalar override values are retained.  Tensor-valued
RANGE overrides are expanded over their old physical support and projected onto
the surviving locations.  A spatial, module-valued override cannot be cropped
safely: it may remain unchanged or be removed in full, but a partial overlap
raises ``ValueError`` before any configuration is changed.  Likewise, a
non-scalar BATCH override is rejected when any of its placement survives,
because sparse deletion can change the compiled row grouping.  Replace such an
override with a scalar or Tensor RANGE value, delete its complete support, or
remove the whole mechanism instead.

When the last supported compartment is deleted, Dendra removes the mechanism's
configuration entirely, so the class is not instantiated by the next build.
Every successful non-empty ``delete`` or ``delete_all`` invalidates
initialization.  Call ``initialize()`` again before stepping or running:

.. code-block:: python

   cells.dendrite.delete(pas)
   cells.initialize()

A structural rebuild replaces compiled mechanism modules and may replace their
registered ``torch.nn.Parameter`` objects.  If an optimizer was created before
the deletion, create a new optimizer *after* reinitialization so it references
the live Parameters:

.. code-block:: python

   cells.soma.delete(hh)
   cells.initialize()
   optimizer = torch.optim.Adam(cells.parameters(), lr=1e-3)

The same replacement matters inside a :class:`~dendra.models.networks.Network`.
Queued connection specifications hold their target synapse object, and any
``SynapseSlots`` object also belongs to that compiled instance.  Before
rebuilding a population that belongs to a network, clear the network's
connections; after rebuilding, reacquire every target mechanism, recreate slot
selections, reconnect, and initialize the network:

.. code-block:: python

   network.clear_synapses()
   post.soma.delete(pas)
   post.initialize()

   live_synapse = post.mech.exp2syn
   live_slots = post.soma.slots(live_synapse)
   network.connect_to_slots(pre, live_slots, weight=0.05, delay=1.0)
   network.initialize(dt=0.025)

This is required even when the deleted mechanism is not itself the synaptic
target: rebuilding the population replaces its complete compiled mechanism
graph.  Dendra rejects stale target mechanisms and stale ``SynapseSlots``
rather than silently wiring connections to detached state.

Overlap, order, and multiplicity
--------------------------------

Advanced indexing and :func:`dendra.models.slice.concat_slices` preserve
selection order and repeated physical indices.  ``shape`` and ``numel()``
therefore describe logical entries, not the number of unique compartments.

Multiplicity has operation-specific meaning:

* reads repeat values in logical selection order;
* intracellular injections add repeated contributions;
* distributed mechanism insertion and spatial parameterization operate on the
  unique physical region unless duplicate preservation is requested explicitly;
* mechanism deletion treats the selection as a physical set and removes every
  duplicate/copy slot at each selected compartment;
* network connection rules apply their documented ``allow_multapses`` policy;
* ``set`` and attribute assignment reject an ambiguous duplicate-bearing write
  when repeated occurrences request different values.

When duplicate locations are intentional, prefer a scalar or a value that is
identical for every occurrence.  For independent colocated synaptic state, use
``SynapseSlots`` rather than treating repeated physical compartments as slots.

Failure behavior
----------------

The Slice API fails early in cases that could otherwise change model meaning:

* invalid direct or nested indices raise the corresponding PyTorch indexing
  error;
* unsafe or duplicate labels raise ``ValueError`` without partially changing
  the model;
* unknown attribute assignments raise ``AttributeError``, while known global
  or non-spatial tensor fields raise ``ValueError``;
* incompatible write shapes report the selected and supplied shapes;
* writes outside sparse mechanism support raise ``ValueError``;
* a strict partial mechanism deletion that includes unsupported compartments
  raises ``ValueError`` without changing the mechanism configuration;
* an unprojectable spatial override makes ``delete`` or ``delete_all`` fail
  without changing any affected mechanism configuration;
* a submodule Slice that cannot be rebound after rebuild raises ``RuntimeError``;
* cross-model Slice concatenation and foreign network endpoints are rejected;
* ``len`` on a scalar Slice raises ``TypeError``.

These errors are part of the contract: Dendra should not silently reinterpret
an index, ignore part of a write, create an attribute after a typo, or keep
using detached mechanism state.

API reference
-------------

See :class:`dendra.models.slice.Slice`,
:class:`dendra.models.slice.SynapseSlots`, and
:func:`dendra.models.slice.concat_slices` for the individual call signatures.
