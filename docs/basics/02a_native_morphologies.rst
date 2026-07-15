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

NEURON import remains fully supported.  In particular,
:meth:`dendra.Tree.from_NEURON`, :meth:`dendra.Tree.from_swc`, and
:meth:`dendra.Tree.from_asc` remain useful for the many established models and
morphology files already expressed through NEURON.  Native and imported
morphologies ultimately use the same scalar compartment-resistor graph
contract in Dendra.

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

``nseg`` is deliberately explicit in this first API.  Native d-lambda or other
automatic discretization policies are not yet inferred.

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

Adding another section to ``morphology`` after either call cannot mutate
``compartment_graph`` or ``tree``.  ``N`` creates identical copies of the same
compiled morphology; per-cell topology or geometry variation is not part of
this first API.

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
from center diameter and ``dx``. The same exact-edge adapter is used when a
``Cable`` is packed into :func:`dendra.concat_models`, so standalone and packed
simulation semantics are identical. Native Cable material diffusion retains
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
dtypes. Author geometric changes—including Section resistivity—on a new
``Morphology`` and compile a new ``Cable`` instead.

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
   ``float64`` array for a named field.

``metadata``
   Node name, ``compartment``/``junction`` kind, source section name, segment
   index, normalized authored segment-center ``section_x``, and structural
   labels.

Stylized cylinders and pt3d truncated-cone intervals are integrated in
binary64.  Area, volume, inverse cross-sectional-area path integrals, axial
resistance, and diffusion geometry are therefore calculated before a
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
* explicit, fixed ``nseg`` values; and
* construction of scalar :class:`dendra.Tree` populations, or fast
  :class:`dendra.Cable` populations for material-only unbranched paths, whose
  ``N`` members share that morphology.

It does not yet define finite-extracellular/double-cable circuit layers for
:class:`dendra.ExtCellTree` or :class:`dendra.ExtCellAxon`, a packed block-DHS
representation, native d-lambda discretization, or per-population-member graph
variation.  Those extensions can build on the canonical contract without
changing the section connection rules described here.

See also
--------

* :doc:`02_branched_morphologies`
* :doc:`../api/dendra.models.morphology`
* :doc:`../units`
