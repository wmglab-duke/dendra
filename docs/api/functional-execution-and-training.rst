Functional execution and training
=================================

Execute a compiled program with :func:`dn.func.run <dendra.func.run>`.
Supply explicit current or voltage tapes with
:class:`dn.func.Inputs <dendra.func.Inputs>`. To resume a run, pass a
:class:`dn.func.Continuation <dendra.func.Continuation>`.
Inspect trainable sources with
:func:`dn.func.inspect_sources <dendra.func.inspect_sources>` and use
:meth:`SourceReport.select <dendra.func.SourceReport.select>` to choose subsets
for optimizer groups.

.. toctree::
   :maxdepth: 1

   dendra.func
   dendra.func.low_level
   dendra.models.parametric
   dendra.opt
