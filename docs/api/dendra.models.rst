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
.. autoclass:: dendra.models.slice.SynapseSlots
   :members:


Random distributions, runtime noise, and State SDEs
----------------------------------------------------

Runtime ``NOISE`` declarations are detached simulation drives. For stochastic
state dynamics, ``State.DIFFUSION`` can be paired with ``State.METHOD("euler_maruyama")``
for Itô SDEs or ``State.METHOD("euler_heun")`` for Stratonovich SDEs.

.. autoclass:: dendra.models.random_parameters.RandomParameterSpec
   :members:
.. autoclass:: dendra.models.random_parameters.RuntimeNoiseSpec
   :members:
.. autoclass:: dendra.models.random_parameters.DistributionSpec
   :members:
.. autofunction:: dendra.models.random_parameters.register_random_distribution
.. autofunction:: dendra.models.random_parameters.available_random_distributions
