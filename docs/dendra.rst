API reference
=============

The examples use ``import dendra as dn``. The :doc:`guides <advanced>` show how
to use these APIs in complete simulation and training workflows.

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

:doc:`Build and connect models <api/models-and-connectivity>` with
:class:`dn.Population <dendra.models.core.Population>`,
:class:`dn.Network <dendra.models.networks.Network>`, and event sources such as
:class:`dn.NetStim <dendra.models.networks.NetStim>`.

Functional execution and training
---------------------------------

:doc:`Run and train models <api/functional-execution-and-training>` with
:func:`dn.compile <dendra.func.compile>`. Set ``extra=`` to
:class:`dn.VarInput() <dendra.func.VarInput>` at compile time to vary
extracellular stimulation between calls.

Mechanisms and stimulation
--------------------------

:doc:`Configure model dynamics and stimulation <api/mechanisms-and-stimulation>`
with mechanisms, material processes, waveforms, and extracellular fields.

Recording and analysis
----------------------

:doc:`Record and analyze responses <api/recording-and-analysis>` with
:class:`dn.Recorder <dendra.models.callbacks.Recorder>`, activity callbacks,
trace measurements, and threshold searches.

Execution settings and utilities
--------------------------------

:doc:`Configure and diagnose execution <api/execution-settings-and-utilities>`
and find unit conversion factors, constants, and numerical helpers.
