.. _slice-contract:

The Slice contract
==================

:class:`dendra.models.slice.Slice` is Dendra's retained, logical selection of
locations in a :class:`dendra.models.core.Population`.  It is the common
interface for inspecting and changing selected state, labelling morphology
regions, inserting mechanisms, applying intracellular stimulation, and
specifying network endpoints.

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

Consequently, ``len(selection)`` and ``selection.numel()`` differ for a
multidimensional selection:

.. code-block:: python

   region = cells[1:3, 2:5]
   assert region.shape == torch.Size([2, 3])
   assert len(region) == 2
   assert region.numel() == 6

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
and ``parametrize`` methods below.

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
``population._labels``.  A label made from a nested Slice belongs only to that
specific parent Slice:

.. code-block:: python

   dendrite = cells.dendrite
   dendrite[:, :2].label("proximal")

   dendrite.proximal           # valid
   # cells.proximal            # not a top-level population label

Nested-label ownership prevents a short local name from leaking into the
population namespace.  Parent and nested labels remain retained selections and
follow the same lifecycle rules as other slices.

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
   * - Forced rebuild after insertion/configuration changes
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

``selection.parametrize(name, value, alias=None)``
   Register a spatial parameter override for the selected locations.  This is
   distinct from changing current mutable state with ``set``.  Batch replicas
   share a core-region parameterization unless the parameter itself explicitly
   defines batch-varying values.

``selection.slots(synapse, ...)``
   Select mechanism-local point-process slots supported by the physical Slice.
   A :class:`dendra.models.slice.SynapseSlots` target is required when several
   independent synapse slots share one physical compartment.  See
   :doc:`advanced/A3_synapse_banks_and_slots`.

Mechanism attribute access requires a built mechanism hierarchy.  It is valid
to retain a population Slice before insertion, but access ``selection.mech``
only after ``build()`` or ``initialize()`` has materialized the mechanisms.

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
