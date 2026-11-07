Parameters and random distributions
===================================

Raw PyTorch parameters, constrained parameter modules, and registered PyTorch
parametrizations work in native and functional execution. See
:doc:`../advanced/training_and_parameters` for selecting trainable sources,
parameter sharing, and optimizer examples.

Constrained parameters
----------------------

.. autoclass:: dendra.models.parametric.Bounded
   :members: set, batch, repeat_and_reinit, clear_cache

.. autoclass:: dendra.models.parametric.PositiveParam
   :members: set, batch, repeat_and_reinit, clear_cache

.. autoclass:: dendra.models.parametric.NegativeParam
   :members: set, batch, repeat_and_reinit, clear_cache

Parameter declarations
----------------------

Mechanisms and MaterialProcesses inherit these declarations. ``GLOBAL`` values
are shared; ``BATCH`` values vary over population rows; ``RANGE`` values vary
over sites. Their ``P`` and ``N`` variants constrain signs. ``RAND`` values are
sampled during initialization; ``NOISE`` values are detached runtime samples.
``TABLE`` creates an interpolated lookup for a named method.

.. autoclass:: dendra.models.parametric.SimpleParameterized
   :members: PARAMETER, PARAMETERP, FLAG, parameter_set_, evaluate, resolve, all_parameter_names
   :undoc-members:

.. autoclass:: dendra.models.parametric.Parameterized
   :members: GLOBAL, GLOBALP, GLOBALN, GLOBAL_SIGNED, BATCH, BATCHP, BATCHN, RANGE, RANGEP, RANGEN, GLOBALRAND, BATCHRAND, RANGERAND, GLOBALNOISE, BATCHNOISE, RANGENOISE, RNG, TABLE, resample_random_parameters, resample_runtime_noise, usetables, parameters_dict, load_parameters_dict

Distribution specifications
---------------------------

Runtime noise does not propagate gradients through distribution parameters.
For differentiable stochastic state dynamics, declare ``State.DIFFUSION``
with ``euler_maruyama`` or ``euler_heun``. See
:doc:`../advanced/custom_models` and :doc:`../advanced/inputs_and_randomness`.

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


Trainable module controls
-------------------------

Population, Network, and other model modules inherit these native selection
helpers. Source reports provide glob/regex selection and full alias groups
across native and functional models; see :func:`dendra.func.inspect_sources`.

.. autoclass:: dendra.models.modular.DNModule
   :members: freeze, freeze_, unfreeze, unfreeze_, collect_parameters, collect_named_parameters

Connection distributions
------------------------

Connection weights and delays accept distribution modules. Explicit draw hooks
allow the same sampling recipe to be replayed in native and functional
execution while differentiating through the distribution parameters.

.. autoclass:: dendra.models.distributions.Distribution
   :members: sample_draw_schema, sample_draw, sample_from_draw, sample_draw_placeholder, sample_with_draw, rsample, sample, log_prob

.. autoclass:: dendra.models.distributions.Normal
   :members: sample_draw_schema, sample_draw, sample_from_draw, sample_with_draw, rsample, sample, log_prob

.. autoclass:: dendra.models.distributions.TruncatedNormal
   :members: std, sample_draw_schema, sample_draw, sample_draw_placeholder, sample_from_draw, sample_with_draw, rsample, sample, log_prob
   :undoc-members:

Explicit distribution sampling
------------------------------

These lower-level adapters describe and evaluate individual base draws. Most
simulation code should use :func:`dendra.func.sample_initialization_draws` and
:func:`dendra.func.sample_runtime_draws` for complete model packets.

.. autofunction:: dendra.models.random_parameters.sample_random_parameter_draw
.. autofunction:: dendra.models.random_parameters.evaluate_random_parameter
.. autofunction:: dendra.models.random_parameters.sample_random_parameter_with_draw
.. autoclass:: dendra.models.random_parameters.RandomParameterDrawRequest
   :members:
.. autofunction:: dendra.models.random_parameters.sample_random_parameter_draws
.. autofunction:: dendra.models.random_parameters.prepare_rng_for_random_parameter
.. autoclass:: dendra.models.random_parameters.RuntimeNoiseDrawRequest
   :members:
.. autofunction:: dendra.models.random_parameters.sample_runtime_noise_draws
.. autofunction:: dendra.models.random_parameters.sample_runtime_noise_draw
.. autofunction:: dendra.models.random_parameters.evaluate_runtime_noise
.. autofunction:: dendra.models.random_parameters.sample_runtime_noise_with_draw
.. autofunction:: dendra.models.random_parameters.iter_initialization_random_declarations
