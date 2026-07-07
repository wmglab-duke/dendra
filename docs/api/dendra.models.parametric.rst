dendra.models.parametric
========================

Tools and mixins for model parameterization.

Declaration overview
--------------------

``Parameterized`` classes expose ordinary effective values through ``GLOBAL``,
``BATCH``, and ``RANGE`` declarations. They also expose constrained variants
such as ``GLOBALP`` / ``RANGEP`` and ``GLOBALN`` / ``RANGEN``.

Random parameter declarations
-----------------------------

``GLOBALRAND``, ``BATCHRAND``, and ``RANGERAND`` declare sampled buffers whose
**distribution parameters** are ordinary Dendra parameters. For example:

.. code-block:: python

   class NoisyLeak(Mechanism):
       Mechanism.RANGERAND(
           "g_scale",
           distribution="truncated_normal",
           mu=1.0,
           sigma=0.2,
           low=0.5,
           high=1.5,
       )

creates a sampled buffer ``g_scale`` plus distribution-parameter buffers such as
``g_scale_mu``, ``g_scale_sigma``, ``g_scale_low``, and ``g_scale_high``. These
can be overridden with the usual insertion-local parameter machinery.

Built-in distributions are available through
``available_random_distributions()`` and include ``normal``, ``lognormal``,
``uniform``, and ``truncated_normal``. Custom distributions can be registered
with ``register_random_distribution(...)``. A sampler has signature:

.. code-block:: python

   sampler(parameters, rng, shape, *, device, dtype) -> torch.Tensor

where ``parameters`` contains the final tensor-valued distribution parameters
after ordinary Dendra overrides and parametrizations.

.. autoclass:: dendra.models.random_parameters.RandomParameterSpec
   :members:
   :member-order: groupwise

.. autoclass:: dendra.models.random_parameters.DistributionSpec
   :members:
   :member-order: groupwise

.. autofunction:: dendra.models.random_parameters.available_random_distributions

.. autofunction:: dendra.models.random_parameters.get_random_distribution

.. autofunction:: dendra.models.random_parameters.register_random_distribution

.. autofunction:: dendra.models.random_parameters.make_random_parameter_spec

.. autofunction:: dendra.models.random_parameters.sample_random_parameter

Core parameterization classes
-----------------------------

.. autoclass:: dendra.models.parametric.Bounded
   :members:
   :member-order: groupwise

.. autoclass:: dendra.models.parametric.PositiveParam
   :members:
   :member-order: groupwise

.. autoclass:: dendra.models.parametric.SimpleParameterized
   :members:
   :member-order: groupwise

.. autoclass:: dendra.models.parametric.Parameterized
   :members:
   :member-order: groupwise
