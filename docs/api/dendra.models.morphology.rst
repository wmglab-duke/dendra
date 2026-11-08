.. _dendra.models.morphology:

Morphology and compartment geometry
===================================

Python module ``dendra.models.morphology``.

Declare sections and their connections with
:class:`dn.Morphology <dendra.models.morphology.Morphology>`. Build a branched
model with :meth:`dn.Tree.from_morphology <dendra.models.tree.Tree.from_morphology>`
or an unbranched model with
:meth:`dn.Cable.from_morphology <dendra.models.core.Cable.from_morphology>`.
Follow the :ref:`native morphology tutorial <native-morphologies>` for a
complete simulation.

Native morphology declaration
-----------------------------

.. autoclass:: dendra.models.morphology.Morphology
   :members:

.. autoclass:: dendra.models.morphology.Section
   :members:

.. autoclass:: dendra.models.morphology.SectionLocation
   :members:

Morphology composition
----------------------

.. autofunction:: dendra.models.morphology.connect_morphologies

Canonical scalar compartment graph
----------------------------------

:meth:`Morphology.compile() <dendra.models.morphology.Morphology.compile>`
validates the sections and returns a snapshot of compartment geometry and
electrical connections. Inspect it before constructing a model if you need
to check the compartment count or the geometry.

.. autoclass:: dendra.models.morphology.CompartmentGraph
   :members:

.. autoclass:: dendra.models.morphology.CompartmentTopology
   :members:

.. autoclass:: dendra.models.morphology.CompartmentGeometry
   :members:

.. autoclass:: dendra.models.morphology.CompartmentMetadata
   :members:

Plot a compartment graph
------------------------

:meth:`~dendra.models.morphology.Morphology.plot_shape` draws the section
declaration. Use ``vis_2d`` to plot the compartment graph of a constructed
Population.

.. autofunction:: dendra.models.visualization.vis_2d
