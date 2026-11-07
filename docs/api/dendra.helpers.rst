Context and execution settings
==============================

``dn.ctx`` scopes defaults for model creation and execution. Functional plans
capture ``CHECK_PURITY`` when created. Set it to ``True`` to diagnose mutation,
implicit randomness, or hidden hook dependencies; it defaults to ``False``.
These checks are diagnostic and cannot prove purity for every input. Tensor
schemas and explicit lifecycle requirements are validated in either mode.

.. autoclass:: dendra.helpers.ctx
   :members:

Context values
--------------

.. autodata:: dendra.DEVICE
   :no-value:

.. autodata:: dendra.DTYPE
   :no-value:

.. autodata:: dendra.CHECK_PURITY
   :no-value:

.. autodata:: dendra.NATIVE_EXTENSION_POLICY
   :no-value:

.. autodata:: dendra.MEMBRANE_ASSEMBLY
   :no-value:

.. autodata:: dendra.PRESERVE_MECHANISM_POPULATION_AXIS
   :no-value:

.. autodata:: dendra.RUNTIME_CONTRACT_VALIDATION
   :no-value:

.. autodata:: dendra.COMPILE_OPTIONS
   :no-value:

Reading and normalizing settings
--------------------------------

.. autofunction:: dendra.helpers.current_device

.. autofunction:: dendra.helpers.current_dtype

.. autofunction:: dendra.helpers.current_check_purity

.. autofunction:: dendra.helpers.normalize_check_purity

.. autofunction:: dendra.helpers.current_native_extension_policy

.. autofunction:: dendra.helpers.normalize_native_extension_policy

.. autofunction:: dendra.helpers.current_membrane_assembly

.. autofunction:: dendra.helpers.normalize_membrane_assembly

.. autofunction:: dendra.helpers.current_preserve_mechanism_population_axis

.. autofunction:: dendra.helpers.normalize_preserve_mechanism_population_axis

.. autofunction:: dendra.helpers.current_runtime_contract_validation

.. autofunction:: dendra.helpers.normalize_runtime_contract_validation

.. autofunction:: dendra.helpers.current_compile_options

.. autofunction:: dendra.helpers.normalize_compile_options

Compilation controls
--------------------

.. autofunction:: dendra.helpers.set_compile_options

.. autofunction:: dendra.helpers.set_jit_enabled

.. autofunction:: dendra.helpers.set_jit_network_solves_enabled

.. autofunction:: dendra.helpers.set_jit_network_ops_enabled

.. autofunction:: dendra.helpers.set_jit_in_network_enabled

.. autofunction:: dendra.helpers.nojit

.. autofunction:: dendra.helpers.allow_tf32

Timing
------

.. autofunction:: dendra.helpers.tic

.. autofunction:: dendra.helpers.toc
