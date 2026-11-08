Models and connectivity
=======================

Use :class:`dn.SingleCompartment <dendra.models.core.SingleCompartment>` for
point neurons, :class:`dn.Cable <dendra.models.core.Cable>` for unbranched cells,
or :class:`dn.Tree <dendra.models.tree.Tree>` for branched cells. Select
compartments with a :class:`Slice <dendra.models.slice.Slice>`. Use
:meth:`Network.connect <dendra.models.networks.Network.connect>` for events
or :meth:`Network.connect_continuous <dendra.models.networks.Network.connect_continuous>`
for continuous signals.

.. toctree::
   :maxdepth: 1

   dendra.models
   dendra.models.networks
   dendra.models.slice
   dendra.models.morphology
   dendra.models.integrators
