dendra.models
=============

Core
----
.. autoclass:: dendra.models.core.Population
   :members:
   :inherited-members: Module, object
   :member-order: groupwise

`Population` subclasses
-----------------------
.. autoclass:: dendra.models.core.SingleCompartment
.. autoclass:: dendra.models.core.Cable
   :members:
.. autoclass:: dendra.models.core.Axon
   :members:
.. autoclass:: dendra.models.core.Unmyelinated
   :members:
.. autoclass:: dendra.models.core.Myelinated
   :members:
.. autoclass:: dendra.models.tree.Tree
   :members:

.. autoclass:: dendra.models.extcell.ExtCellAxon
   :members:
.. autoclass:: dendra.models.extcell.ExtCellTree
   :members:


Native morphology construction
------------------------------

Dendra-native :class:`~dendra.models.morphology.Morphology` declarations and
NEURON-authored morphologies both produce the scalar compartment-resistor graph
consumed by :class:`~dendra.models.tree.Tree`, or by the tridiagonal
:class:`~dendra.models.core.Cable` fast path when the graph is one unbranched
material cable. See
:doc:`dendra.models.morphology` for the declaration and canonical graph APIs,
including :func:`dendra.connect_morphologies` for composing complete trees,
and :doc:`../basics/02a_native_morphologies` for the connection contract and
worked examples.


Packed scalar populations
-------------------------

:func:`dendra.concat_models` packs independent scalar systems into one solver
launch without adding electrical edges between them. Ordinary ``Population``
and ``SingleCompartment`` models are represented as one-node trees, ``Tree``
retains its rooted compartment graph, native ``Cable`` retains exact canonical
path edges, and ``Unmyelinated``/``Myelinated`` use live specialized tensor
geometry. These component types may be mixed, then explicitly batched on the
returned model:

.. code-block:: python

   packed = dendra.concat_models(
       {
           "point_cells": point_population,
           "dendrites": tree_population,
           "fibres": axon_population,
       },
       write_back=True,
   )
   packed.batch(32)

All components must initially be unbatched and share a device and dtype.
``write_back=True`` keeps each component's public voltage synchronized after a
step; ``False`` updates only the composite voltage. Finite-extracellular
``ExtCellTree``/``ExtCellAxon`` states require a block multi-solver and are
rejected by this scalar path.

.. autofunction:: dendra.models.multi.concat_models

.. autoclass:: dendra.models.multi.MultiPopulation
   :members:


Finite two-layer extracellular models
-------------------------------------

``ExtCellAxon`` and ``ExtCellTree`` solve two finite extracellular circuit
layers using NEURON's ``extracellular`` conventions. ``xraxial`` is in
MΩ/cm, ``xc`` is in µF/cm², and ``xg`` is in S/cm². The ``extra``/``ve``
input is the bath battery outside the outermost layer (NEURON's
``e_extracellular``), not the first solved extracellular node.

The block state is ordered as ``vc[..., 0] = vi``, ``vc[..., 1] = vext[0]``,
and ``vc[..., 2] = vext[1]``; public membrane voltage is therefore
``v = vi - vext[0]``. ``Myelinated`` instead implements the simpler node-only,
fully insulating-internode abstraction and should not be interpreted as this
double-cable circuit.

With ``IMEM=1``, stable voltage integrators expose absolute outward-positive
transmembrane current in mA as capacitive plus ionic/mechanism current. Axial
and applied intracellular currents influence that value through the voltage
solution but are not included as separate transmembrane-current terms.


Compartment identifiers
-----------------------

``CompartmentID`` describes repeating named regions while defining an axon
model. Register it with :meth:`dendra.models.core.Axon.register_cid` to make
those names available to Dendra's search and slicing APIs and as labelled
attributes on the axon.

.. autoclass:: dendra.models.heterogeneous.CompartmentID
   :members:


Morphology graph helpers
------------------------

These functions calculate weighted distances while treating a directed
morphology graph as undirected, so paths may be measured both toward and away
from the graph root.

.. autofunction:: dendra.models.utils.distance
.. autofunction:: dendra.models.utils.undirected_weighted_lengths


Synapse slot targets
--------------------

:class:`dendra.models.slice.SynapseSlots` explicitly selects local slots in a
banked point-process mechanism. Its canonical API reference is
:doc:`dendra.models.slice`.


Random distributions, runtime noise, and State SDEs
----------------------------------------------------

Runtime ``NOISE`` declarations are detached simulation drives. For stochastic
state dynamics, ``State.DIFFUSION`` can be paired with ``State.METHOD("euler_maruyama")``
for Itô SDEs or ``State.METHOD("euler_heun")`` for Stratonovich SDEs.

The canonical API reference for
:class:`dendra.models.random_parameters.RandomParameterSpec`,
:class:`dendra.models.random_parameters.RuntimeNoiseSpec`,
:class:`dendra.models.random_parameters.DistributionSpec`, and the distribution
registry helpers is :doc:`dendra.models.parametric`.
