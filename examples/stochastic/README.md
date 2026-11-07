# Randomness and replay

Use explicit draws to compare simulations on the same random realization, train
stochastic models, or resume a sampled path from a checkpoint. Omitted draws are
sampled automatically; explicit packets make replay independent of ambient RNG
changes.

Run from the repository root:

```bash
python examples/stochastic/functional_runtime_draws.py --backend eager
python examples/stochastic/shared_initialization_draws.py
```

## Capture and replay draws

| Example | Use it to… |
| --- | --- |
| [`functional_runtime_draws.py`](functional_runtime_draws.py) | Replay `RuntimeDraws` through compiled Population and Network runs, including gradients and continuation. |
| [`shared_runtime_draw_sampling.py`](shared_runtime_draw_sampling.py) | Sample runtime draw packets directly from native Populations and Networks. |
| [`shared_runtime_draws.py`](shared_runtime_draws.py) | Replay the same one-step NOISE, SDE and NetStim draws in native and functional execution. |
| [`shared_runtime_draw_tapes.py`](shared_runtime_draw_tapes.py) | Supply draw tapes to native `run`, `longrun` and `longrun_checkpointed`. |
| [`shared_runtime_draw_horizons.py`](shared_runtime_draw_horizons.py) | Sample draws for `duration=` requests, including carried fractional time. |

## Initialize heterogeneous models

| Example | Use it to… |
| --- | --- |
| [`functional_random_parameters.py`](functional_random_parameters.py) | Sample RAND heterogeneity with explicit draws and reproducible restarts. |
| [`functional_connection_sampling.py`](functional_connection_sampling.py) | Train Normal and TruncatedNormal connection weight and delay distributions. |
| [`functional_distribution_sampling.py`](functional_distribution_sampling.py) | Implement a custom pure connection distribution with trainable sources. |
| [`functional_initialization_draws.py`](functional_initialization_draws.py) | Capture and replay initialization draws for compiled Populations and Networks. |
| [`shared_initialization_draws.py`](shared_initialization_draws.py) | Reuse one initialization realization across native and functional execution. |
| [`functional_stochastic_initialization_ordering.py`](functional_stochastic_initialization_ordering.py) | Combine sampled parameters with pre-initialization transforms in a Network. |

## Define stochastic dynamics

| Example | Use it to… |
| --- | --- |
| [`functional_runtime_noise.py`](functional_runtime_noise.py) | Drive a model with detached NOISE values and replayable draw/RNG tapes. |
| [`functional_state_sde.py`](functional_state_sde.py) | Integrate State SDEs with Euler–Maruyama or Euler–Heun and pathwise gradients. |
| [`functional_material_stochasticity.py`](functional_material_stochasticity.py) | Add sampled heterogeneity and runtime noise to a custom MaterialProcess. |
| [`functional_noise_workspaces.py`](functional_noise_workspaces.py) | Use noise-dependent derived and timestep coefficients with the intended evaluation timing. |

## Replay Poisson stimulation

| Example | Use it to… |
| --- | --- |
| [`functional_poisson_stimulation.py`](functional_poisson_stimulation.py) | Replay randomized waveforms, including custom containers and mechanism-owned injections. |
| [`shared_poisson_replay.py`](shared_poisson_replay.py) | Replay intracellular and extracellular Poisson waveforms through native Population steps. |
| [`shared_network_poisson_replay.py`](shared_network_poisson_replay.py) | Replay Poisson waveform invocations shared across Network Populations. |
| [`shared_mechanism_poisson_replay.py`](shared_mechanism_poisson_replay.py) | Supply explicit draws for mechanism-owned Poisson injections in native runs. |

## Share streams and restore RNG state

| Example | Use it to… |
| --- | --- |
| [`functional_network_rng_sharing.py`](functional_network_rng_sharing.py) | Share one random stream across functional Network Populations. |
| [`functional_stimulation_rng_sharing.py`](functional_stimulation_rng_sharing.py) | Share one generator between Poisson stimulation and model NOISE. |
| [`functional_rng_load_policies.py`](functional_rng_load_policies.py) | Restore cached numerical values while choosing which incoming random streams to retain. |

See [inputs and randomness](../../docs/advanced/inputs_and_randomness.md) for draw
packets and gradient behavior, [initialization and checkpoints](../../docs/advanced/initialization_and_checkpoints.md)
for resampling and RNG restoration, and [custom models](../../docs/advanced/custom_models.md)
for stochastic mechanism declarations.

[All examples](../README.md)
