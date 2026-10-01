.. _native-morphologies:

02a. Native section morphologies
================================

Dendra can define a branched cable directly, without first constructing
NEURON ``Section`` objects.  A :class:`dendra.Morphology` is an editable
section-level declaration; :meth:`~dendra.Morphology.compile` turns it into an
immutable :class:`dendra.CompartmentGraph`; and
:meth:`dendra.Tree.from_morphology` constructs a general branched simulation
model from that snapshot. If the compiled graph is one unbranched material
path, :meth:`dendra.Cable.from_morphology` uses the faster tridiagonal cable
backend without approximating the canonical geometry.

NEURON import remains fully supported. :meth:`dendra.Morphology.from_swc` and
:meth:`dendra.Morphology.from_asc` now make the editable native declaration the
canonical representation after loading. The older
:meth:`dendra.Tree.from_NEURON`, :meth:`dendra.Tree.from_swc`, and
:meth:`dendra.Tree.from_asc` constructors remain available for established
workflows that intentionally want NEURON's immediate d-lambda discretization.
Native and imported morphologies ultimately use the same scalar
compartment-resistor graph contract in Dendra.

Declare sections
----------------

The native morphology API uses Dendra's ordinary numerical unit conventions:

.. list-table::
   :widths: 35 25 40
   :header-rows: 1

   * - Quantity
     - Unit
     - API names
   * - Length, diameter, and coordinates
     - µm
     - ``L``, ``diam``, pt3d ``x/y/z``
   * - Intracellular resistivity
     - Ω·cm
     - ``rhoa``
   * - Specific membrane capacitance
     - µF/cm²
     - ``cm``

The values in :mod:`dendra.units` are scalar conversion factors, so they can be
used to make the length convention explicit:

.. code-block:: python

   import dendra as dn
   from dendra.models.mod import hh, pas
   from dendra.units import um

   morphology = dn.Morphology(rhoa=100.0, cm=1.0)

   # A uniform cylinder, divided into one computational compartment.
   soma = morphology.section(
       "soma",
       L=20.0 * um,
       diam=20.0 * um,
       nseg=1,
       labels={"cell_body"},
   )

   # A piecewise-linear pt3d centerline with a linearly varying diameter.
   apic = morphology.section(
       "apic",
       points=[
           (0.0 * um, 0.0 * um, 10.0 * um, 4.0 * um),
           (0.0 * um, 0.0 * um, 110.0 * um, 2.0 * um),
           (20.0 * um, 0.0 * um, 210.0 * um, 1.0 * um),
       ],
       nseg=9,
       labels={"dendrite", "tuft_path"},
   )

   # Labels can group multiple independently named Sections.
   basal = morphology.section(
       "basal",
       L=120.0 * um,
       diam=2.0 * um,
       nseg=5,
       labels={"dendrite"},
   )

Every section has a unique non-empty name and an explicit positive integer
``nseg``.  A stylized section supplies both ``L`` and scalar ``diam``.  A pt3d
section supplies at least two ``(x, y, z, diameter)`` points and derives its
length from centerline arclength; it cannot also supply ``L`` or ``diam``.
Section-level ``rhoa`` and ``cm`` override the morphology defaults.

Consecutive pt3d controls may share xyz only when their diameters differ. This
is an abrupt diameter discontinuity, represented by the same zero-length
degenerate frustum used by NEURON. It contributes the annular membrane area
``π |d₂² - d₁²| / 4`` but no centerline length, volume, or axial resistance.
The complete Section must still have positive centerline length. Fully
identical consecutive controls are redundant and rejected by the native
authoring API; ASC import coalesces them. When a discontinuity lies exactly on
a compartment boundary, its area belongs to the lower-x compartment, matching
NEURON.

``nseg`` is explicit declaration state. Supply it directly when the desired
count is known, or replace the initial counts later with the native d-lambda
policy described below. Geometry samples remain authoring controls, not
implicit computational compartments.

Choose direct or d-lambda discretization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every Section always has one explicit positive integer ``nseg``. For a
read-only estimate, :meth:`~dendra.Section.lambda_f` returns that Section's AC
space constant in µm using its current geometry, ``rhoa``, and ``cm``:

.. code-block:: python

   wavelength_um = apic.lambda_f(freq_hz=100.0)

``freq_hz`` is deliberately named: it accepts a raw numerical frequency in
hertz, unlike Dendra's usual waveform-frequency coordinate in kHz. Pass
``100.0`` for 100 Hz, **not** ``100.0 * Hz`` (which evaluates to Dendra's
``0.1`` kHz coordinate). The method is analytical and never changes ``nseg``.
For pt3d Sections it uses NEURON's classic control-interval approximation over
the compiler-representable authored diameter profile, using the same binary64
centerline spans as native compilation. This approximation combines each
positive-length control interval with its two endpoint diameters and is
intentionally distinct from the exact tapered-frustum integration used for
area, volume, and axial resistance. A same-coordinate diameter step has zero
centerline interval and therefore contributes zero electrotonic length to this
estimate, although compilation still retains its exact annular membrane area.

Use :meth:`~dendra.Morphology.apply_d_lambda` to select and assign counts for
every current Section:

.. code-block:: python

   policy_morphology = dn.Morphology(rhoa=100.0, cm=1.0)
   policy_soma = policy_morphology.section(
       "soma",
       L=20.0 * um,
       diam=20.0 * um,
       nseg=1,
   )
   policy_dend = policy_morphology.section(
       "dend",
       L=1000.0 * um,
       diam=2.0 * um,
       nseg=1,
       labels="dendrite",
   )
   policy_dend.connect(policy_soma.at(1.0), child_end=0)

   coarse_graph = policy_morphology.compile()
   selected = policy_morphology.apply_d_lambda(
       d_lambda=0.1,   # Dimensionless fraction of lambda_f.
       freq_hz=100.0,  # Raw Hz.
   )

   assert selected == {
       section.name: section.nseg for section in policy_morphology.sections
   }
   assert all(nseg >= 1 and nseg % 2 == 1 for nseg in selected.values())

   refined_graph = policy_morphology.compile()
   assert coarse_graph.n_compartments == 2
   assert refined_graph.n_compartments == sum(selected.values())
   refined_tree = dn.Tree.from_morphology(policy_morphology)

``d_lambda`` is a positive dimensionless target for compartment length as a
fraction of ``lambda_f(freq_hz=...)``. Dendra applies the same odd-count
rounding rule as its NEURON-backed importer, so every selected ``nseg`` is odd.
The returned dictionary maps exact Section names to those counts in declaration
order. The operation is a one-time authoring edit, not a live policy: a later
geometry, ``rhoa``, or ``cm`` update does not silently rediscretize the
Morphology. Reapply d-lambda when desired, or override an individual count with
``section.update(nseg=...)``.

Application is transactional. Dendra validates the positive finite
``d_lambda`` and ``freq_hz`` values and calculates every Section's count before
changing any declaration; a failure leaves all prior ``nseg`` values intact.
Section identities, labels, connections, declaration order, and saved
locations are retained. Previously compiled graphs and instantiated models are
independent snapshots, as ``coarse_graph`` demonstrates above. Compile or
construct a new model after applying the policy. Applying it to an empty
Morphology returns an empty dictionary after validating its arguments.

Section names and region labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first argument to :meth:`~dendra.Morphology.section` is the Section's
unique authored ``name``. It is the Section's identity for connectivity and
compiled provenance; every material compartment produced from it records that
value in ``CompartmentMetadata.section_name``. Section names are automatically
included among the Section's labels, so the examples above have these label
sets in addition to their distinct identities:

.. code-block:: text

   soma   -> {"soma", "cell_body"}
   apic   -> {"apic", "dendrite", "tuft_path"}
   basal  -> {"basal", "dendrite"}

Explicit ``labels`` are reusable structural region tags. Each label covers
every one of the Section's ``nseg`` material compartments, and several Sections
may share a label. Consequently, after
:meth:`~dendra.Tree.from_morphology`, ``tree.soma`` selects the uniquely named
``soma`` Section while ``tree.dendrite`` is the union of the ``apic`` and
``basal`` Sections. Ordinary shared labels are encouraged, but Section names
are reserved: an explicit label on one Section cannot equal another Section's
name, regardless of which Section is declared first. This keeps a name-derived
selection such as ``tree.soma`` section-specific.

The compiled metadata keeps three related concepts separate:

``metadata.name``
   Generated per-node display/search provenance, for example ``"soma(0.5)"``.

``metadata.section_name``
   The single authored Section identity, such as ``"soma"``. Retained junction
   nodes use ``None`` because they do not belong to one Section.

``metadata.labels``
   The set of structural tags on each material compartment, including its
   Section name. Retained junction nodes have an empty set.

When a :class:`~dendra.Tree` or :class:`~dendra.Cable` is constructed directly
from a ``Morphology``, collision-free labels that are non-private Python
identifiers—not keywords or reserved model attributes—become population-owned
:class:`~dendra.Slice` attributes. Other labels remain available through
``compartment_graph.metadata`` and
:meth:`~dendra.CompartmentGraph.nodes_with_label`, but are not installed as
attributes. Shared Slice labels follow Section declaration order, with
compartments in increasing authored Section ``x`` within each Section; reversed
physical traversal through ``child_end=1`` does not change that public order.

Load SWC and Neurolucida ASC
----------------------------

Use :meth:`~dendra.Morphology.from_swc` when the source is a classic
seven-column SWC node tree:

.. code-block:: python

   morphology = dn.Morphology.from_swc(
       "cell.swc",
       rhoa=100.0,
       cm=1.0,
       nseg=1,
   )

   # Inspect or edit the native declaration before constructing a model.
   morphology.plot_shape()
   morphology.apply_d_lambda(d_lambda=0.1, freq_hz=100.0)
   tree = dn.Tree.from_morphology(morphology)

The SWC loader is native and does not call NEURON. It preserves finite xyz
samples, radius as ``diameter = 2 * radius``, rooted connectivity, and raw
structure type IDs. Structural node, parent, and type IDs are parsed exactly
rather than through binary64. It forms maximal root-away pt3d Sections of one
downstream type, splitting at branches and type changes. Generated names are
deterministic and collision-free—such as ``soma_0``, ``axon_0``,
``basal_dendrite_0``, and ``apical_dendrite_0``—while shared labels include
``soma``, ``axon``, ``dendrite``, and the raw provenance label
``swc_type_<id>`` where applicable. ``type_labels={42: "custom_neurite"}``
can add project-specific semantics without discarding the raw ID.

SWC carries neither Dendra electrical properties nor a numerical
discretization. ``rhoa``, ``cm``, and ``nseg`` are therefore explicit import
policy. The loader's positive integer ``nseg`` is a uniform *initial* count for
the newly declared Sections; it does not apply d-lambda while parsing, and
geometry samples are not treated as compartments. After inspecting or editing
the native declaration, either update selected Section counts directly or call
:meth:`~dendra.Morphology.apply_d_lambda` once to replace all initial counts.
This separation lets the d-lambda calculation observe any imported-geometry,
``rhoa``, or ``cm`` edits made after loading. The importer rejects forests,
cycles, duplicate IDs, missing parents, non-finite values, non-positive radii,
and zero-length edges rather than repairing them silently. A non-soma root
whose type has no same-type cable continuation is likewise rejected:
preserving that isolated annotation would otherwise require inventing geometry
or silently retyping the root.

A common SWC convention represents a soma by one type-1 point. A point alone
cannot define a positive-length cable, so the default
``single_point_soma="sphere"`` creates the familiar three-point x-axis cable
surrogate centered at that sample, with length and diameter ``2r``. Its lateral
area equals the sphere's surface area, although its cable volume is
cylindrical. Use ``single_point_soma="error"`` when implicit geometry is not
acceptable.

SWC cannot reconstruct original Section boundaries, names, labels, ``nseg``,
``rhoa``, or ``cm``. Raw imported types remain available through the read-only
:attr:`~dendra.Morphology.swc_section_types` mapping. Pass it explicitly when
re-exporting because :meth:`~dendra.Morphology.to_swc` never guesses types from
names or labels:

.. code-block:: python

   morphology.write_swc(
       "edited-cell.swc",
       section_types=morphology.swc_section_types,
   )

Neurolucida ASC is a much broader language containing contours, nested trees,
spines, markers, colors, arbitrary properties, and repair conventions.
:meth:`~dendra.Morphology.from_asc` therefore uses NEURON's mature
``Import3d_Neurolucida3`` compatibility reader, then immediately snapshots its
normalized cable interpretation into native pt3d Sections:

.. code-block:: python

   morphology = dn.Morphology.from_asc(
       "reconstruction.asc",
       rhoa=100.0,
       cm=1.0,
       nseg=1,
   )
   morphology.apply_d_lambda(d_lambda=0.1, freq_hz=100.0)

   # A file with disconnected reconstructions requires an exact generated root.
   component = dn.Morphology.from_asc(
       "reconstruction.asc",
       root="soma[0]",
   )

Generated NEURON identities such as ``soma[0]``, ``axon[3]``, ``dend[8]``,
and ``apic[1]`` become exact Section names and also receive their shared
structural base label. If several root Sections exist, Dendra lists them and
requires an explicit choice instead of silently discarding components. Exact
duplicate consecutive pt3d samples are coalesced. Equal-coordinate samples
with different diameters are instead preserved as meaningful zero-length
diameter discontinuities, including their annular membrane area. A Section
with no positive-length centerline after coalescing is rejected rather than
replaced by NEURON's fallback stylized geometry.

ASC import preserves the selected NEURON-normalized cable topology,
centerlines, diameters, logical attachments, and explicit Dendra electrical
policy. As with native SWC loading, ``nseg`` is a uniform initial declaration;
the later ``apply_d_lambda`` call is an explicit, independently repeatable
authoring step. This differs from the older :meth:`dendra.Tree.from_swc` and
:meth:`dendra.Tree.from_asc` constructors, which intentionally apply their
NEURON-backed d-lambda policy during immediate model import. Native ASC loading
is not a lossless document parser: source formatting, comments,
colors, markers, properties, spines, trace IDs, original soma contours, and
components not selected by ``root`` are not retained. The returned
``Morphology`` contains no live NEURON objects and is the editable Dendra source
for subsequent inspection, updates, compilation, and model construction.
When NEURON repairs an outlying main branch by connecting it logically to the
nearest soma, Dendra preserves that established Import3d behavior but emits one
``RuntimeWarning`` containing NEURON's repair diagnostics. The normalization is
therefore explicit rather than silent.

Update section declarations
---------------------------

A native ``Morphology`` remains editable before and after snapshots are
compiled. Section fields are read-only attributes, so an edit goes through the
validated update API rather than direct assignment:

.. code-block:: python

   # Both forms update the same stable Section object.
   soma.update(nseg=3, rhoa=120.0, cm=1.2)
   morphology.update_section("basal", rhoa=90.0)

   assert morphology.update_section(apic, nseg=11) is apic

:meth:`~dendra.Section.update` is the natural object-oriented form;
:meth:`~dendra.Morphology.update_section` additionally accepts either a
Section owned by that Morphology or its exact name. The Section's object
identity, authored ``name``, owner, declaration order, and existing
connections are retained. Consequently, previously saved
:class:`~dendra.SectionLocation` objects continue to refer to the same Section
and normalized ``x``. A Section from another Morphology, or a copied/forged
Section object that merely has the same name, is rejected.

Updates are transactional. Dendra first validates the complete candidate
declaration—including positive finite electrical values, geometry, ``nseg``,
and global name/label rules—and changes nothing if validation fails. Passing
``labels=...`` replaces the Section's explicit labels; its unique Section name
is always added back automatically. Names themselves cannot be updated because
they are the stable keys for connectivity and compiled provenance.

Every optional keyword on :meth:`~dendra.Section.update` and
:meth:`~dendra.Morphology.update_section` uses ``None`` to mean *leave the
current value unchanged*. It does not mean “inherit the Morphology default
again.” To apply a current default to an existing Section, pass it
explicitly—for example, ``soma.update(rhoa=morphology.rhoa)``. Passing
``labels=()`` removes all explicit labels while retaining the automatic
Section-name label.

Whole-section geometry follows the same distinction as declaration. A
stylized Section can update its scalar ``L`` or ``diam``; a pt3d Section can
replace its complete ``points`` sequence, from which ``L`` is derived again:

.. code-block:: python

   soma.update(L=24.0 * um, diam=18.0 * um)

   apic.update(
       points=[
           (0.0 * um, 0.0 * um, 10.0 * um, 4.0 * um),
           (5.0 * um, 0.0 * um, 115.0 * um, 2.2 * um),
           (25.0 * um, 0.0 * um, 220.0 * um, 1.0 * um),
       ]
   )

Supplying ``points`` can promote a stylized Section to pt3d geometry. The
update API does not perform the reverse pt3d-to-stylized conversion: ``L`` and
scalar ``diam`` are derived concepts for an existing pt3d Section, so attempts
to update them directly—or to mix either one with ``points``—are rejected.
Build the desired stylized Section in a new Morphology when that representation
change is required.

For a local diameter edit on a pt3d centerline, use the same normalized
location syntax used by connections:

.. code-block:: python

   updated_section = apic.at(0.4).update(diam=2.5 * um)
   assert updated_section is apic

If ``x=0.4`` is already an authored pt3d sample, its diameter is replaced. If
not, Dendra inserts a sample at the interpolated centerline coordinate and
assigns the requested diameter. Pt3d diameter is linearly interpolated between
samples, so this operation edits a diameter control point and affects its
adjacent span or spans—one at a Section endpoint and two at an interior
control point. It does not target one compiled compartment.
:meth:`~dendra.SectionLocation.update` returns the canonical owning Section,
not a new SectionLocation or a replacement Section object. Local diameter
updates are deliberately restricted to pt3d Sections. Convert a stylized
cylinder to explicit pt3d geometry before introducing a local taper.
At an abrupt diameter discontinuity, one normalized location has two authored
diameter limits. A local update there is therefore rejected as ambiguous; use
``section.update(points=...)`` to replace the explicit incoming and outgoing
controls.

``rhoa``, ``cm``, ``nseg``, and ``labels`` remain whole-Section properties and
are not accepted by :meth:`~dendra.SectionLocation.update`. A value at one
zero-width location would not define an electrical interval. Author piecewise
base ``rhoa`` with separately connected Sections whose cable spans carry the
desired scalar values, then compile and construct a new model. Changing a
post-construction tensor named ``rhoa`` is not generally equivalent: canonical
native geometry can contain precomputed edge resistances, and different model
classes expose different runtime scaling contracts.

Likewise, ``nseg`` and connection topology belong to the compiled snapshot;
model parameterization cannot change them. Section ``labels`` are compiled
provenance. A Slice label created on an existing model can organize that model,
but it does not rewrite the source Section labels or
``CompartmentGraph.metadata.labels``. After construction, vary only fields that
the relevant model explicitly documents as runtime-configurable—for example,
mechanism parameters or supported membrane/scale fields—and follow its
reinitialization requirements for solver-affecting changes.

Changing geometry does not rewrite normalized connection locations: stored
``parent_x`` and ``child_end`` values retain their logical meanings on the
updated Section. Likewise, changing ``nseg`` can change which newly compiled
parent compartment contains an interior attachment. Electrical connectivity
remains valid because it is independent of displayed coordinates, although a
geometry edit that separates connected pt3d endpoints will subsequently fail
the stricter spatial-continuity check required by SWC export.

The Morphology's ``rhoa`` and ``cm`` attributes are validated defaults for new
Sections. Each Section receives a resolved value when it is declared, so
changing a Morphology default affects only Sections declared afterward. Update
existing Sections explicitly when a new value should apply to them; Dendra
does not guess whether an existing value was inherited or supplied explicitly.

Delete and replace section subtrees
-----------------------------------

Use :meth:`~dendra.Section.delete` or
:meth:`~dendra.Morphology.delete_section` to remove authored Sections. The
object-oriented and explicit forms are equivalent:

.. code-block:: python

   deleted = morphology.delete_section("terminal")
   # Equivalent when starting from the same undeleted declaration:
   # deleted = terminal.delete()

Both methods return the deleted Section names as a tuple in their original
declaration order. The default ``recursive=False`` is deliberately safe: a
Section with children cannot be deleted because doing so would orphan them or
require Dendra to guess how they should be reparented. Set ``recursive=True``
to remove the selected Section and its complete descendant subtree. Its parent
and sibling subtrees remain untouched; Dendra removes the incoming connection
and the connections internal to the deleted subtree without reconnecting
anything implicitly.

Deletion is transactional and uses the same ownership rules as updates. An
unknown name, a Section from another Morphology, a copied/noncanonical Section,
or an already deleted Section is rejected without changing the declaration.
Any imported SWC type provenance for deleted Sections is removed as well.
Saved Section and SectionLocation handles belonging to retained Sections stay
valid, while handles into the deleted subtree become stale and are rejected.

For example, an entire branched axonal arbor can be replaced by one straight
axon while retaining its attachment to the soma:

.. code-block:: python

   editable = dn.Morphology()
   soma = editable.section("soma", L=20 * um, diam=20 * um)

   axon_root = editable.section(
       "axon[0]",
       points=[
           (0 * um, 0 * um, 0 * um, 1 * um),
           (50 * um, 0 * um, 0 * um, 1 * um),
       ],
       nseg=11,
       labels="axon",
   )
   collateral = editable.section(
       "axon[1]",
       points=[
           (25 * um, 0 * um, 0 * um, 0.8 * um),
           (25 * um, 40 * um, 0 * um, 0.6 * um),
       ],
       nseg=7,
       labels="axon",
   )
   terminal = editable.section(
       "axon[2]",
       points=[
           (50 * um, 0 * um, 0 * um, 0.8 * um),
           (90 * um, -20 * um, 0 * um, 0.5 * um),
       ],
       nseg=7,
       labels="axon",
   )

   # This retained parent location remains valid after deleting the child arbor.
   axon_attachment = soma.at(0.0)
   axon_root.connect(axon_attachment, child_end=0)
   collateral.connect(axon_root.at(0.5), child_end=0)
   terminal.connect(axon_root.at(1.0), child_end=0)

   removed = axon_root.delete(recursive=True)
   assert removed == ("axon[0]", "axon[1]", "axon[2]")

   straight_axon = editable.section(
       "axon[0]",  # A deleted name may be reused by a new canonical Section.
       points=[
           (0 * um, 0 * um, 0 * um, 1 * um),
           (500 * um, 0 * um, 0 * um, 1 * um),
       ],
       nseg=51,
       labels="axon",
   )
   straight_axon.connect(axon_attachment, child_end=0)

   revised_tree = dn.Tree.from_morphology(editable)

Deleting the sole root, including a complete tree with ``recursive=True``, is
allowed and leaves an empty editable Morphology. As usual,
:meth:`~dendra.Morphology.compile` and model construction reject an empty
declaration until a new root is authored. Previously compiled graphs and
instantiated models are immutable snapshots and are not altered by deletion;
compile or construct a new model to observe the revised morphology.

Connect complete morphologies
-----------------------------

Use :func:`dendra.connect_morphologies` when the host cell and the arbor to
attach are already complete Morphology declarations. The parent location may
be anywhere on the host tree. The child location must be endpoint ``0`` or
``1`` of the child tree's root Section; selecting any other Section would give
the child tree two parents. Both sources must validate as complete, connected
one-root trees before they can be composed.

For example, a cell ending in a short dendritic stump can be extended with a
separately authored full arbor. Here both declarations contain a Section named
``"dendrite_root"``, so ``child_prefix`` gives every copied child Section a
collision-free name:

.. code-block:: python

   host = dn.Morphology(rhoa=90.0, cm=1.0)
   soma = host.section(
       "soma",
       points=[
           (-10 * um, 0 * um, 0 * um, 20 * um),
           (10 * um, 0 * um, 0 * um, 20 * um),
       ],
   )
   stump = host.section(
       "dendrite_root",
       points=[
           (10 * um, 0 * um, 0 * um, 2 * um),
           (30 * um, 0 * um, 0 * um, 2 * um),
       ],
       labels="dendrite",
   )
   stump.connect(soma.at(1.0), child_end=0)

   arbor = dn.Morphology(rhoa=80.0, cm=1.2)
   arbor_root = arbor.section(
       "dendrite_root",
       points=[
           (30 * um, 0 * um, 0 * um, 2 * um),
           (80 * um, 0 * um, 0 * um, 1.4 * um),
       ],
       nseg=5,
       labels="dendrite",
   )
   upper = arbor.section(
       "upper",
       points=[
           (80 * um, 0 * um, 0 * um, 1.4 * um),
           (130 * um, 40 * um, 0 * um, 0.8 * um),
       ],
       nseg=5,
       labels="dendrite",
   )
   lower = arbor.section(
       "lower",
       points=[
           (80 * um, 0 * um, 0 * um, 1.4 * um),
           (130 * um, -40 * um, 0 * um, 0.8 * um),
       ],
       nseg=5,
       labels="dendrite",
   )
   upper.connect(arbor_root.at(1.0), child_end=0)
   lower.connect(arbor_root.at(1.0), child_end=0)

   combined = dn.connect_morphologies(
       stump.at(1.0),
       arbor_root.at(0.0),
       child_prefix="arbor_",
   )

   assert tuple(section.name for section in combined.sections) == (
       "soma",
       "dendrite_root",
       "arbor_dendrite_root",
       "arbor_upper",
       "arbor_lower",
   )
   tree = dn.Tree.from_morphology(combined)

Composition is pure: ``combined`` contains newly owned Section objects and
neither ``host`` nor ``arbor`` is changed. Sections retain their authored
geometry, discretization, electrical values, explicit structural labels, and
imported SWC type provenance. The result uses the host Morphology's ``rhoa``
and ``cm`` defaults for Sections declared later, and orders copied host
Sections before copied child Sections. ``child_prefix`` changes copied child
Section names, their automatic name labels, and the keys of their SWC
provenance; omit it when the two declarations' names and reserved labels are
already collision-free.

The new electrical edge has the same endpoint and resistor semantics as a
same-Morphology Section connection. Composition does not translate or rotate
either declaration's coordinates. Supply already aligned pt3d coordinates
when spatial continuity matters, particularly before SWC export or evaluation
of extracellular fields.

Connect sections
----------------

:meth:`dendra.Section.at` identifies a normalized authored location ``x`` in
the closed interval ``[0, 1]``. A child connects through exactly one of its
authored endpoints:

.. code-block:: python

   axon = morphology.section("axon", L=500 * um, diam=1 * um, nseg=21)

   # Attach child x=0 to parent x=1.
   apic.connect(soma.at(1.0), child_end=0)

   # Interior parent locations are supported.
   basal.connect(soma.at(0.5), child_end=0)

   # Attaching authored child x=1 traverses toward decreasing authored x.
   axon.connect(soma.at(0.0), child_end=1)

The equivalent explicit form is useful when both endpoints should be visible:

.. code-block:: python

   other = dn.Morphology()
   root = other.section("root", L=20 * um, diam=20 * um)
   child = other.section("child", L=100 * um, diam=2 * um, nseg=5)
   other.connect(root.at(1.0), child.at(0.0))

A section can have only one parent, a section cannot connect to itself, and the
complete declaration must form one connected acyclic tree with exactly one
root.  At an interior parent location, the child endpoint is attached to the
parent compartment containing that location.  At an exact internal
compartment boundary, Dendra selects the compartment on the ``x``-increasing
side; ``x=1`` is clamped to the final compartment.  This is the same boundary
selection convention used by Dendra's NEURON morphology importer.

Endpoint and interior connections have distinct resistor semantics.  An
endpoint-to-endpoint connection contains the adjacent half-compartment cable
on both sections.  An interior connection identifies the junction with the
containing parent compartment, so the connection edge contains the child's
adjacent half compartment but no additional parent half compartment.

If three or more cable paths meet at a physical endpoint, compilation retains
an explicit ``junction`` node.  Junctions have zero length, membrane area, and
volume, and therefore should not be targeted by membrane or point-process
mechanisms.
Sealed degree-one endpoint nodes are removed, while degree-two endpoint nodes
are exactly collapsed as two series resistors.

Native Section coordinates are stable after connection: ``child_end`` chooses
which authored endpoint faces the parent but does not rename or reflect the
Section's ``x`` coordinate. Consequently, a child attached through
``child_end=1`` is traversed away from its parent in decreasing authored ``x``.
NEURON's internal ``Segment.x`` convention for an orientation-1 Section is
different; Dendra's NEURON importer handles that pt3d mapping while preserving
the same physical cable graph.

Coordinates and electrical connections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Electrical connectivity is independent of the displayed centerline
coordinates.  Pt3d coordinates are preserved exactly and are not translated
when a section is connected.  Stylized ``L``/``diam`` sections use a canonical
local centerline from ``(0, 0, 0)`` to ``(0, 0, L)`` and likewise are not
translated or rotated by :meth:`~dendra.Morphology.connect`.

This mirrors the useful separation between NEURON's section connectivity and
its pt3d placement, but it also means that connected stylized sections may
overlap spatially.  Use pt3d sections with coherent absolute coordinates when
spatial placement matters, especially for extracellular fields.  Axial path
length and resistance follow cable arclength and connection semantics, not the
Euclidean gap between independently supplied endpoints.

Visualize and inspect declarations
----------------------------------

A native :class:`~dendra.Morphology` can be inspected directly while it is
being authored. The visualization methods take a read-only snapshot of the
current Section declarations and connections; they never construct a
``Tree``/``Cable`` or mutate the Morphology. Spatial plots, diameter profiles,
and the Section schematic render that declaration directly. The compartment
topology canonically compiles each connected component of the snapshot so its
nodes and edges match the electrical graph. This also lets an empty or
partially connected Section forest be visualized before it satisfies the
single-root model contract.

For a low-noise view of what the authored cable envelope looks like, use
:meth:`~dendra.Morphology.plot_shape` or its rotatable 3-D counterpart
:meth:`~dendra.Morphology.plot_shape_3d`:

.. code-block:: python

   fig_shape, ax_shape = morphology.plot_shape(view="y")
   fig_shape_3d, ax_shape_3d = morphology.plot_shape_3d()

These renderers build closed, linearly tapered tubes directly from authored
``x``, ``y``, ``z``, and diameter values. With the default
``diameter_scale=1``, radius is expressed in the same morphology coordinate
units as centerline position, so thickness is physically proportional rather
than a screen-space linewidth. ``radial_segments=8`` controls surface
smoothness. The quiet shape views use the same stable vivid palette as the
other native views while deliberately grouping related Sections into
structural color families. Thus ``dend[0]`` and ``dend[1]`` are both
``dend`` rather than visually unrelated cables. The canonical colors match the
default :func:`dendra.models.visualization.vis_2d` language for a native Tree:
``soma`` is ``#c8e650``, ``axon`` is ``#e65050``, ``dend`` is ``#c850e6``, and
``apic`` is ``#508ce6``. These colors have the same meaning in
``plot_shape`` and ``plot_shape_3d`` and do not depend on Section declaration
order. They are the unshaded base colors; deterministic surface lighting varies
their brightness without changing the structural family hue.

Terminal generated indices are removed for grouping: both NEURON names such
as ``dend[12]`` and native SWC names such as ``basal_dendrite_3`` map to their
structural family. The recognized aliases ``cell_body``, ``basal_dendrite``,
and ``apical_dendrite`` map to ``soma``, ``dend``, and ``apic`` respectively.
A canonical identity derived from the Section name takes precedence over its
broader labels, so ``apic[0]`` remains apical even when it also carries the
shared ``dendrite`` label. For a generic name, one unambiguous canonical label
provides its family—for example, separately named ``left`` and ``right``
Sections labelled ``dendrite`` share ``dend``. No canonical label, or
conflicting canonical labels, retains the normalized name as a deterministic,
append-stable fallback family rather than guessing.

The legend contains one entry per displayed family rather than one per
Section. ``legend="auto"`` shows this compact key for at most 16 families, so
a reconstruction with hundreds of indexed dendritic Sections can still have a
useful four-region legend. Set ``legend=False`` to hide it or ``legend=True``
to force it. Axes, titles, point controls, compartment markers, orientation
arrows, annotations, and connection glyphs are absent by default. The 3-D shape
view retains only a compact x/y/z orientation triad in its lower-left corner.
The triad is anchored to the display rather than the morphology coordinates,
follows camera rotation, and never changes the data limits. Set
``show_axis_indicator=False`` to remove it; ``show_axes=True`` independently
enables the full physical coordinate axes.

Matplotlib 3-D axes can rotate and zoom whenever they are displayed through an
event-capable backend. To request that behavior explicitly—and add scroll-wheel
zoom—select the backend before creating the figure and pass
``interactive=True``:

.. code-block:: ipython

   %matplotlib widget
   fig_shape_3d, ax_shape_3d = morphology.plot_shape_3d(interactive=True)

Left drag rotates, middle drag pans, right drag zooms, and the scroll wheel
zooms the current view. Static inline output still renders the figure but cannot
receive those events, so Dendra emits an actionable warning rather than trying
to change Matplotlib's backend. See :ref:`interactive-jupyter` for installation,
server/kernel environment, and restart guidance. A desktop GUI backend works as
well; call ``plt.show()`` if that environment does not display returned figures
automatically.

The shape renderer uses the authored declaration, not ``nseg`` or the compiled
graph, and never invents placement. Spatially coherent child tubes meet
naturally. A gapped electrical connection remains visibly gapped and emits one
warning; use :meth:`~dendra.Morphology.plot` or
:meth:`~dendra.Morphology.inspect` for the detailed connection diagnostic.
An authored zero-length diameter discontinuity appears as the corresponding
annular shoulder rather than being smoothed or assigned artificial length.
Likewise, connected stylized Sections retain their canonical local z-axis and
may overlap. This is a tapered-cable envelope, not a histological reconstruction
or a boolean surface union at branch junctions; native Morphology does not yet
define a distinct spherical-soma primitive.

For a diagnostic orthographic view, :meth:`~dendra.Morphology.plot` draws the
exact authored centerlines and optional controls. ``view`` is the axis viewed
along—and therefore omitted from the plot—so ``view="y"`` displays x against
z:

.. code-block:: python

   fig, ax = morphology.plot(
       view="y",
       color_by="diameter",
       highlight="dendrite",
       show_points=True,
       show_compartments=True,
       show_orientation=True,
       annotate_connections=True,
   )

   # The methods return ordinary Matplotlib handles and never call show().
   fig.savefig("morphology-xz.png", bbox_inches="tight")

The corresponding :meth:`~dendra.Morphology.plot_3d` view preserves all three
spatial axes and is especially useful for checking a bent pt3d centerline or a
child connected through authored ``child_end=1``:

.. code-block:: python

   fig_3d, ax_3d = morphology.plot_3d(
       color_by="section",
       show_points=True,
       show_compartments=True,
       show_orientation=True,
   )

Both spatial views preserve coordinates and relative geometry as authored.
They do not place, translate, rotate, or repair a connected child.
Consequently, connected stylized Sections still share their canonical local
centerline from
``(0, 0, 0)`` to ``(0, 0, L)`` and may overlap in a spatial plot. That overlap
is not a topology error. For coordinates near binary64's numerical limits,
the renderer may subtract a common display origin or divide an axis by a
display scale so Matplotlib can retain the local cable geometry. The figure
prints every such transform; it never silently changes the Morphology. Use
:meth:`~dendra.Morphology.plot_topology` for a coordinate-independent
view of the exact compartment connectivity graph:

.. code-block:: python

   fig_topology, ax_topology = morphology.plot_topology(
       highlight="dendrite",
   )

Every material compartment is a node colored by its exact Section name; the
legend records that mapping. The vivid HSV palette follows the visual language
of :func:`dendra.models.visualization.vis_2d`, but its fixed progressive order
ensures that appending a Section does not recolor existing Sections. By
default, an ordinary material circle's marker area scales with its compiled
arclength-mean diameter (``node_scale=8``), subject to the
``min_node_size=12`` and ``max_node_size=240`` points² display bounds. Edge
width similarly scales with the mean of its material endpoints' compiled
diameters
(``edge_scale=0.12``), bounded by ``min_edge_width=0.35`` and
``max_edge_width=3`` points. These display floors keep thin axons visible,
while the caps prevent a thick soma from obscuring a broad tree.

A diamond marks a material compartment with at least three undirected graph
neighbors. It retains its Section color and diameter encoding, with the
slightly larger ``branchpoint_scale=1.35`` and separate display bounds
(``min_branchpoint_size=24`` and ``max_branchpoint_size=300`` points²).
A neutral ``X`` instead marks a retained zero-area algebraic junction. Its
``junction_size=52`` points² is fixed because such a node has no material
diameter; incident edge width is derived from the adjacent material cable.
For a deliberately uniform schematic, explicit ``node_size``,
``branchpoint_size``, and ``edge_width`` values override the corresponding
diameter-scaled defaults with fixed display sizes. The compatibility
``branchpoint_size`` override fixes both material diamonds and junction
``X`` markers; set ``junction_size`` alone to resize only the nonmaterial
junctions.

Edges remain the canonical axial connections after endpoint removal,
degree-two junction collapse, and interior-attachment resolution. Thus both
interior and endpoint branching remain explicit without mistaking an arbitrary
degree-two solver root for a physical branchpoint. Marker area and line width
encode diameter only; node position and spacing remain deliberately schematic,
not morphological distance, and there is no per-node text on the canvas.

``highlight`` accepts the same exact Section names and shared labels as the
spatial views. Algebraic junctions have no Section label and remain neutral
structural context. Non-matching Sections are muted but remain visible so the
highlight does not erase graph context.

To use hover in Jupyter, follow :ref:`interactive-jupyter`. The setup matters
because ``ipympl`` has a Python backend in the kernel and a JavaScript frontend
in the Jupyter server. If those use separate environments, install compatible
ipympl components in both; after a first installation, restart the complete
server rather than only its kernel. For a shared server and kernel environment,
install the Jupyter extra:

.. code-block:: bash

   python -m pip install "dendra[jupyter]"

An existing notebook may use ``%pip install ipympl`` for its kernel side, but
that does not install into a separately managed server environment. After the
server restart and page refresh, select the widget backend *before* creating
the figure:

.. code-block:: ipython

   %matplotlib widget
   fig_hover, ax_hover = morphology.plot_topology(interactive=True)

A desktop GUI backend also works once the figure is displayed with
``plt.show()``. Classic Notebook versions before 7 may instead use
``%matplotlib notebook``; that fallback does not work in JupyterLab. Hovering
over a material node shows its generated name, Section, segment index,
normalized Section x, labels, length, arclength-mean diameter, integrated area
and volume, ``rhoa``, ``cm``, and spatial center. Hovering over an
algebraic-junction ``X`` instead reports its connectivity and zero-area status,
deliberately omitting copied diameter, ``cm``, ``rhoa``, and spatial fields that
are not material properties there. Static ``%matplotlib inline`` and saved
raster output cannot react to pointer motion; Dendra warns when
``interactive=True`` is requested with the inline backend. The method still
returns ``(Figure, Axes)`` and never calls ``show()``.

For a complete Morphology, this is the exact
:class:`~dendra.CompartmentGraph` produced by
:meth:`~dendra.Morphology.compile`. A partially authored forest remains
inspectable: each current connected component is compiled independently with
the same endpoint-collapse and junction rules. Use
:meth:`~dendra.Morphology.plot_section_topology` when the desired diagnostic
is instead one labelled box per authored Section with ``nseg``, ``rhoa``,
``cm``, labels, attachment locations, and spatial-gap status:

.. code-block:: python

   fig_sections, ax_sections = morphology.plot_section_topology(
       show_parameters=True,
       show_labels=True,
       show_compartments=True,
       annotate_connections=True,
   )

For taper validation, :meth:`~dendra.Morphology.plot_diameter_profile` plots
the piecewise-linear diameter controls independently for selected Sections:

.. code-block:: python

   fig_profile, ax_profile = morphology.plot_diameter_profile(
       sections=[soma, apic],
       x_axis="distance",  # µm measured from each Section's authored x=0
       show_points=True,
       show_compartments=True,
       show_connections=True,
   )

Use ``x_axis="normalized"`` for each Section's own ``[0, 1]`` coordinate.
Connected profiles are not concatenated into one global distance axis; each
Section retains its authored coordinate and orientation. This makes the
profile useful for checking local controls added with
``section.at(x).update(diam=...)``. Two controls forming a zero-length
diameter step share one horizontal position and therefore appear as a vertical
profile segment.

:meth:`~dendra.Morphology.inspect` combines the three orthographic views,
compartment topology, and diameter profile in one coordinated dashboard:

.. code-block:: python

   inspection_figure, axes = morphology.inspect(
       highlight={"dendrite", "axon"},
       show_points=True,
       show_compartments=True,
       annotate_connections=True,
   )

   axes["topology"].set_title("Electrical compartment topology")

All visualization methods return their Matplotlib figure and axes handles and
leave display policy to the caller. ``inspect`` returns an axes mapping with
``"x"``, ``"y"``, ``"z"``, ``"topology"``, and ``"diameter"`` entries. In an
interactive session, call ``plt.show()`` explicitly if the frontend does not
display the returned figure automatically.

The shared styling and selection contract is:

* In the diagnostic spatial views, ``color_by="section"`` assigns
  deterministic categorical colors to exact Section identities;
  ``"diameter"``, ``"length"``, ``"nseg"``, ``"rhoa"``, and ``"cm"`` provide
  numeric color scales. The compartment topology is intentionally always
  colored by exact Section name. The quiet ``plot_shape`` and
  ``plot_shape_3d`` views are the deliberate exception: their purpose is
  legible whole-cell form, so they use the structural family contract above.
  Shared labels are not generally a disjoint partition and therefore remain
  highlight selectors rather than a diagnostic ``color_by`` mode; shape views
  consult only unambiguous canonical structural labels. If a numeric field is
  too large for safe Matplotlib colorbar arithmetic, its displayed divisor is
  included explicitly in the colorbar label.
* ``highlight`` accepts one exact Section name or shared label, or an iterable
  of exact names/labels. A shared label highlights the union of every matching
  Section; non-matches are muted. Unknown strings are rejected rather than
  interpreted as globs or silently ignored.
* ``show_points`` marks authored pt3d controls (or a stylized Section's two
  canonical endpoints). ``show_compartments`` previews the equal-arclength
  boundaries/centers implied by ``nseg``. It does not reproduce junction
  retention/collapse; use ``plot_topology`` for the exact compiled graph.
  ``show_orientation`` draws the increasing-x arrow on one actual local
  authored span and marks a connected child's parent-facing endpoint with its
  ``0`` or ``1`` value. This preserves the distinction between x direction and
  traversal away from a parent without drawing an arrow across a bend.
* Centerline width encodes authored diameter using ``diameter_scale`` and
  ``min_linewidth`` in screen-space display points. It makes taper legible but
  is not a physically to-scale solid tube; axes, pt3d coordinates, diameter
  values, connection gaps, and distance-profile positions remain in µm. The
  renderer subdivides a sparse straight authored span only as a visual device
  when its endpoint diameters differ; this does not add morphology controls or
  alter compilation. A diagnostic centerline cannot display the annular area
  of a zero-length diameter step; ``plot_shape`` and ``plot_shape_3d`` render
  its concentric rings and shoulder instead.
* With ``show_connections=True``, coincident attachment points and spatial
  gaps are presented as connection diagnostics. A dashed gap connector records
  the electrical relationship; it is not an invented cable segment. The
  ``connection_tolerance_um`` value only classifies these visual diagnostics.
  It does not move coordinates, change electrical connectivity, validate the
  morphology for compilation, or relax the independent SWC export tolerance.

Each call snapshots the declaration at its start. Later calls reflect
subsequent :meth:`~dendra.Section.update` or
:meth:`~dendra.SectionLocation.update` edits, while already returned artists,
previously compiled graphs, and existing model instances remain independent.

Export to SWC
-------------

A native morphology can be serialized as standard seven-column SWC text or
written directly to a file:

.. code-block:: python

   swc_morphology = dn.Morphology()
   swc_soma = swc_morphology.section(
       "soma",
       points=[(0, 0, 0, 20), (0, 0, 20, 20)],
   )
   swc_apic = swc_morphology.section(
       "apic",
       points=[(0, 0, 20, 4), (0, 0, 120, 2)],
   )
   swc_apic.connect(swc_soma.at(1.0), child_end=0)

   swc_types = {
       "soma": 1,
       "apic": 4,
   }

   swc_text = swc_morphology.to_swc(section_types=swc_types)
   swc_morphology.write_swc("cell.swc", section_types=swc_types)

``section_types`` maps exact authored Section names to integer SWC type IDs.
Names omitted from the mapping use ``default_type=0`` (undefined); Dendra does
not guess a type from a Section name or label. Unknown Section names in the
mapping are rejected so that a typo cannot silently produce a mistyped file.
SWC ``x``, ``y``, ``z``, and radius are in µm. Dendra writes radius as one half
of the authored diameter.

SWC export describes the authored centerlines and section topology, not the
compiled compartment graph. Authored pt3d sample locations are retained, and
stylized sections contribute their two canonical endpoints. ``nseg`` therefore
does not add SWC samples. If a child attaches to an interior parent location,
the exporter inserts an interpolated sample at the exact junction. The
connected child endpoint and parent attachment become one shared SWC node,
avoiding a duplicate zero-length edge; at that shared location the parent's
single SWC type and radius take precedence.

Classic SWC does not carry the annular membrane surface represented by a
same-coordinate diameter discontinuity. Export therefore rejects a Morphology
containing such controls rather than silently choosing one diameter or writing
a zero-length SWC edge that Dendra could not faithfully round-trip.

Node order is deterministic. The root is written from authored ``x=0`` toward
``x=1``; each child is then written away from its connected ``child_end``.
Consequently, a child connected through ``child_end=1`` appears in reverse
authored point order. Branches follow Section declaration order, and every
parent node precedes its children.

Unlike Dendra's electrical connection model, SWC couples connectivity to
spatial geometry. A connected child endpoint must therefore coincide with its
parent attachment. Export raises an error for a spatially discontinuous
connection instead of moving a Section or inventing a connector. The
``connection_tolerance_um`` option exists only to accommodate coordinate
roundoff; use coherent absolute pt3d coordinates when an SWC file is required.

SWC is a lossy interchange format for a Dendra declaration. It does not encode
``nseg``, ``rhoa``, ``cm``, Section names, or labels. A classic SWC junction
also has only one type and radius, so the shared node retains the parent's type
and radius if the child's connected endpoint differs. Reimporting an exported
file can therefore choose a new discretization and does not reconstruct the
original ``Morphology`` declaration exactly.

Compile and construct a Tree or Cable
-------------------------------------

Compilation validates the declaration and returns an immutable snapshot:

.. code-block:: python

   compartment_graph = morphology.compile()
   tree = dn.Tree.from_morphology(morphology, N=128)

   # Section names are automatic Slice labels; shared explicit labels select
   # the union of every Section carrying them.
   tree.soma.insert(pas)
   tree.dendrite.insert(pas)
   tree.axon.insert(hh)

Adding or updating a Section on ``morphology`` after either call cannot mutate
``compartment_graph`` or ``tree``. Compilation and model construction are
snapshots, not live views of the builder. Re-run
:meth:`~dendra.Morphology.compile` to obtain a revised graph, or construct a
new :class:`~dendra.Tree`/:class:`~dendra.Cable` with ``from_morphology`` to use
the edited declaration in a simulation. ``N`` creates identical copies of the
same compiled morphology; per-cell topology or geometry variation is not part
of this first API.

For an unbranched morphology, construct a generic cable instead:

.. code-block:: python

   cable = dn.Cable.from_morphology(morphology, N=128)

``Cable`` is deliberately distinct from :class:`dendra.Axon`. ``Axon`` and its
``Unmyelinated``/``Myelinated`` subclasses use fiber-level parameters and
specialized geometric relationships; those concepts are not expressive enough
for an arbitrary tapered pt3d path. ``Cable`` is the geometry-general
unbranched abstraction, while ``Axon`` remains a specialized ``Cable`` family.

Cable eligibility and numerical contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``Cable`` must contain only material compartments and its undirected
electrical graph must be one simple path. True branches and retained zero-area
junction nodes are rejected. A valid path whose canonical rooted node lies in
its interior is reordered deterministically from one physical end to the other;
the aligned :attr:`~dendra.Cable.compartment_graph` records that storage order.
Section labels remain ordered by increasing authored Section ``x`` regardless
of physical traversal direction. Graph attachment is factory-only: do not pass
an arbitrary NetworkX graph to the low-level ``Cable`` constructor. The public
``cable.graph`` interoperability property returns a fresh copy, so editing it
cannot make voltage and material solvers observe different topologies.

The unbranched voltage kernel consumes exact compiled membrane areas and exact
center-to-center edge resistances. It does not reconstruct tapered geometry
from one representative compartment diameter and ``dx``. The same exact-edge
adapter is used when a ``Cable`` is packed into :func:`dendra.concat_models`,
so standalone and packed simulation semantics are identical. Native Cable
material diffusion retains
the canonical volumes and ``diff_geom_um`` through the graph/DHS material
backend rather than using a cylindrical approximation.

The default :func:`~dendra.models.integrators.bwd_euler_ub` integrator provides
the tridiagonal fast path. :func:`~dendra.models.integrators.dhs` is also a
compatible exact graph solver. A generic ``Cable`` rejects integrators that do
not explicitly preserve unbranched cable topology; for example, a point-model
integrator cannot silently turn the cable into independent compartments.

Compiled electrical and material geometry is immutable. ``diam``, ``dx``, base
``rhoa``, exact area and edge resistance, material volumes, diffusion geometry,
and diffusion parent indices must remain aligned to one canonical snapshot;
Dendra rejects post-compilation mutation at initialization and public execution
boundaries. State dictionaries also carry a morphology fingerprint and cannot
be loaded into a same-shaped but different native cable. A compatible restore
validates the incoming frozen geometry but retains the target Cable's locally
compiled snapshot, avoiding precision-dependent graph/buffer divergence across
dtypes. Author geometric changes—including Section resistivity—on the source
``Morphology`` with the validated update API, then compile a new ``Cable``. If
the original declaration is no longer available, build a new Morphology; do
not mutate frozen geometry on the existing Cable.

Runtime membrane and placement values remain configurable: ``cm`` and
``area_scale`` change membrane capacitance/current scaling, while ``x``/``y``/``z``
can place the already-compiled cable in an extracellular field. ``rhoa_scale``
may differ between independent cable replicas but must be spatially uniform
within each cable; after two exact half-paths have been reduced to one edge
total, distinct per-compartment scaling cannot be recovered without a new
compilation. Supply persistent scale values through the factory/constructor.
Once a timestep workspace has been built, changes to ``cm``, ``cm_scale``,
``area_scale``, or ``rhoa_scale`` require rebuilding every derived solver cache
before it can be reused; otherwise old coefficients could disagree with new
public buffers.

Runtime contract validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dendra checks these geometry and solver-workspace invariants at public
execution boundaries. The default policy is appropriate for normal
simulations. Advanced validation modes, trusted-loop performance tuning,
aliasing and callback caveats, and owner reinitialization rules are documented
in :doc:`../advanced/A5_runtime_contracts`.

The canonical compartment graph
--------------------------------

:class:`dendra.CompartmentGraph` separates immutable topology, binary64
geometry, and provenance:

``topology.parent_index``
   One child-indexed parent ID per node.  Exactly one entry is ``-1``.  Native
   compilation assigns deterministic breadth-first integer IDs, with every
   parent before its children.

``geometry``
   Node-aligned tuples ``length_um``, ``diameter_um``, ``area_um2``,
   ``volume_um3``, intracellular ``volume_i_um3``, extracellular
   ``volume_o_um3``, ``x_um``, ``y_um``, ``z_um``, ``rhoa_ohm_cm``, and
   ``cm_uF_cm2``. Child-indexed edge tuples are ``edge_length_um``,
   ``edge_resistance_ohm``, and ``edge_diff_geom_um``; the root entries are
   zero. :meth:`dendra.CompartmentGeometry.array` returns a fresh NumPy
   ``float64`` array for a named field. For a material compartment,
   ``diameter_um`` is the arclength mean over that compartment, matching
   NEURON's segment-diameter convention. It is representative metadata rather
   than an input from which the exact area, volume, or edge geometry is
   reconstructed.

``metadata``
   Node name, ``compartment``/``junction`` kind, source section name, segment
   index, normalized authored segment-center ``section_x``, and structural
   labels.

Stylized cylinders and positive-length pt3d truncated-cone intervals are
integrated in binary64. A zero-length diameter step adds its annular membrane
area to the lower-x compartment (or the first compartment at ``x=0``) while
adding no volume, path length, axial resistance, or diffusion distance. Area,
volume, inverse cross-sectional-area path integrals, axial resistance, and
diffusion geometry are therefore calculated before a
:class:`~dendra.Tree` or :class:`~dendra.Cable` deliberately casts model buffers
to its configured dtype.

NetworkX interoperability
-------------------------

NetworkX remains a supported inspection and interchange view rather than the
canonical source of truth:

.. code-block:: python

   canonical = morphology.compile()
   nx_graph = canonical.to_networkx()

   # Inspect, serialize, visualize, or pass through existing graph tooling.
   print(nx_graph.nodes[0])

   restored = dn.CompartmentGraph.from_networkx(nx_graph)
   tree = dn.Tree.from_compartment_graph(restored, N=16)

   # Valid only when restored is a material-only unbranched path.
   cable = dn.Cable.from_compartment_graph(restored, N=16)

The adapter writes the established Tree node fields ``L``, ``diam``, ``area``,
``volume``, ``volume_i``, ``volume_o``, ``Ra``, ``cm``, ``x/y/z`` and the edge
fields ``L``, ``R_ohm``, and ``diff_geom_um``. Explicit edge resistance and
diffusion geometry are preserved. For older Tree-compatible graphs that omit
them, the adapter uses the documented stylized half-cylinder fallback; that
fallback should not be mistaken for exact pt3d geometry.

Current scope
-------------

The native morphology API currently supports:

* one connected scalar resistor tree with one root;
* explicit per-Section ``nseg`` values, including transactional native
  d-lambda selection from current geometry and electrical properties; and
* construction of scalar :class:`dendra.Tree` populations, or fast
  :class:`dendra.Cable` populations for material-only unbranched paths, whose
  ``N`` members share that morphology.

It does not yet define finite-extracellular/double-cable circuit layers for
:class:`dendra.ExtCellTree` or :class:`dendra.ExtCellAxon`, a packed block-DHS
representation, or per-population-member graph variation. Those extensions can
build on the canonical contract without changing the section connection rules
described here.

See also
--------

* :doc:`02_branched_morphologies`
* :doc:`../api/dendra.models.morphology`
* :doc:`../units`
