API reference
=============

Dendra is commonly imported as ``import dendra as dn``. Top-level conveniences
such as ``dn.Population``, ``dn.Network``, ``dn.NetStim``, ``dn.Recorder``,
``dn.compile`` and ``dn.VarInput`` refer to the objects documented below.
The :doc:`guides <advanced>` show complete workflows.

.. toctree::
   :hidden:
   :maxdepth: 2

   api/models-and-connectivity
   api/functional-execution-and-training
   api/mechanisms-and-stimulation
   api/recording-and-analysis
   api/execution-settings-and-utilities

Models and connectivity
-----------------------

:doc:`Build populations and networks <api/models-and-connectivity>` and choose
the regions, morphologies, and integrators they use.

Functional execution and training
---------------------------------

:doc:`Compile a model or set up a training loop <api/functional-execution-and-training>`
with control over trainable parameters, simulation inputs, and continuation.

Mechanisms and stimulation
--------------------------

:doc:`Configure model dynamics and stimulation <api/mechanisms-and-stimulation>`
with mechanisms, material processes, waveforms, and extracellular fields.

Recording and analysis
----------------------

:doc:`Record simulation outputs and measure activity <api/recording-and-analysis>`
using callbacks, trace measurements, and threshold searches.

Execution settings and utilities
--------------------------------

:doc:`Configure and diagnose execution <api/execution-settings-and-utilities>`
and find unit conversion factors, constants, and numerical helpers.
