dendra.models.parametric
========================

Tools and mixins for model parameterization.

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


Random parameters and runtime noise
-----------------------------------

``GLOBALRAND`` / ``BATCHRAND`` / ``RANGERAND`` declare sampled buffers for
initialization-time heterogeneity. ``GLOBALNOISE`` / ``BATCHNOISE`` /
``RANGENOISE`` declare detached runtime-noise buffers that can be resampled
during simulation. Runtime ``NOISE`` buffers are optimized for simulation and do
not preserve gradients through their distribution parameters. For first-class
State SDEs, use ``State.DIFFUSION`` with ``State.METHOD("euler_maruyama")`` for
Itô dynamics or ``State.METHOD("euler_heun")`` for Stratonovich dynamics.

.. autoclass:: dendra.models.random_parameters.RandomParameterSpec
   :members:

.. autoclass:: dendra.models.random_parameters.RuntimeNoiseSpec
   :members:

.. autoclass:: dendra.models.random_parameters.DistributionSpec
   :members:

.. autofunction:: dendra.models.random_parameters.available_random_distributions
.. autofunction:: dendra.models.random_parameters.get_random_distribution
.. autofunction:: dendra.models.random_parameters.register_random_distribution
.. autofunction:: dendra.models.random_parameters.make_random_parameter_spec
.. autofunction:: dendra.models.random_parameters.make_runtime_noise_spec
.. autofunction:: dendra.models.random_parameters.sample_random_parameter
.. autofunction:: dendra.models.random_parameters.sample_runtime_noise
