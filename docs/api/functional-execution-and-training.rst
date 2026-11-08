Functional execution and training
=================================

Use ``dn.compile`` to create a reusable Population or Network runner for
simulation or gradient-based fitting. For direct control over state, inputs,
and continuation, use the plans and tensor packets in ``dn.func``. The
parameter and optimization references cover trainable values, random
distributions, and gradient utilities.

.. toctree::
   :maxdepth: 1

   dendra.func
   dendra.func.low_level
   dendra.models.parametric
   dendra.opt
