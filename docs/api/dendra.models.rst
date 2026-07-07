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


Synapse slot targets
--------------------
.. autoclass:: dendra.models.slice.SynapseSlots
   :members:


Random distributions and runtime noise
--------------------------------------
.. autoclass:: dendra.models.random_parameters.RandomParameterSpec
   :members:
.. autoclass:: dendra.models.random_parameters.RuntimeNoiseSpec
   :members:
.. autoclass:: dendra.models.random_parameters.DistributionSpec
   :members:
.. autofunction:: dendra.models.random_parameters.register_random_distribution
.. autofunction:: dendra.models.random_parameters.available_random_distributions
