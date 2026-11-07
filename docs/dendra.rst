API reference
=============

Dendra is commonly imported as ``import dendra as dn``. Top-level conveniences
such as ``dn.Population``, ``dn.Network``, ``dn.NetStim``, ``dn.Recorder``,
``dn.compile`` and ``dn.VarInput`` refer to the objects documented below.
The reference groups APIs by task; the guides show complete workflows.

Models and connectivity
-----------------------

.. toctree::
   :maxdepth: 1

   api/dendra.models
   api/dendra.models.networks
   api/dendra.models.slice
   api/dendra.models.morphology
   api/dendra.models.integrators

Functional execution and training
---------------------------------

.. toctree::
   :maxdepth: 1

   api/dendra.func
   api/dendra.func.low_level
   api/dendra.models.parametric
   api/dendra.opt

Mechanisms and stimulation
--------------------------

.. toctree::
   :maxdepth: 1

   api/dendra.models.mechanisms
   api/dendra.models.mod
   api/dendra.models.stim
   api/dendra.models.fields

Recording and analysis
----------------------

.. toctree::
   :maxdepth: 1

   api/dendra.models.callbacks
   api/dendra.data
   api/dendra.models.analysis
   api/dendra.models.instruments

Execution settings and utilities
--------------------------------

.. toctree::
   :maxdepth: 1

   api/dendra.stepping
   api/dendra.helpers
   api/dendra.diagnostics
   api/dendra.const
   api/dendra.units
   api/dendra.utils
