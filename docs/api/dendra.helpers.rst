dendra.helpers
==============

Context
-------

Functional execution trusts user-authored pure hooks by default. Enable
optional diagnostics while creating a plan or compiled runner:

.. code-block:: python

   with dn.ctx(CHECK_PURITY=True):
       runner = dn.compile(network, dt=0.025, steps=100)

``CHECK_PURITY`` defaults to ``False``. The plan captures this setting when it
is created; changing the context does not change an existing plan. Diagnostic
checks can identify mutation, implicit randomness, and hidden dependencies in
supported hooks, but do not prove purity for every input. Tensor schemas,
layouts, explicit bindings, and lifecycle requirements remain checked in both
modes. Recreate a plan inside the context to diagnose an existing model.

.. autoclass:: dendra.helpers.ctx
   :members:

Timing
------

.. autofunction:: dendra.helpers.tic
.. autofunction:: dendra.helpers.toc

Others
------
.. autofunction:: dendra.helpers.allow_tf32
.. autofunction:: dendra.helpers.nojit
.. autofunction:: dendra.helpers.set_jit_enabled
.. autofunction:: dendra.helpers.set_jit_network_solves_enabled
.. autofunction:: dendra.helpers.set_jit_network_ops_enabled
.. autofunction:: dendra.helpers.set_jit_in_network_enabled
