# Recording and callbacks

Record selected fields or compute observations during a simulation. Callback
configurations work with native execution and compiled models, including
Population scopes inside a Network. In-memory observations can retain gradients;
HDF5 storage writes detached samples.

Run from the repository root:

```bash
python examples/callbacks/functional_lfp_configuration.py --section network --execution scan --backend eager
python examples/callbacks/native_callback_storage.py
```

## Recording and Population scopes

| Example | What it helps you do |
| --- | --- |
| [functional_recorder_storage.py](functional_recorder_storage.py) | Record voltage and HH mechanism state in memory or HDF5 with the same Recorder configuration and cadence. |
| [native_callback_storage.py](native_callback_storage.py) | Stream native Population and Network observations to HDF5 while continuing callback state across execution methods. |
| [functional_network_regions.py](functional_network_regions.py) | Record named physical regions, including components of concatenated MultiPopulations. |
| [functional_network_callbacks.py](functional_network_callbacks.py) | Observe selected Populations in a compiled Network with separate named callbacks. |
| [shared_population_callback_scopes.py](shared_population_callback_scopes.py) | Reuse `PopulationCallback` scopes across native and functional Networks. |

## Online analysis

| Example | What it helps you do |
| --- | --- |
| [functional_lfp_configuration.py](functional_lfp_configuration.py) | Project membrane currents into LFP observations for a Population or scoped Network, with gradients and continuation. |
| [functional_threshold_configuration.py](functional_threshold_configuration.py) | Use APCount, Raster and AnomalyDetector configurations with native-equivalent functional results. |
| [functional_activity_configuration.py](functional_activity_configuration.py) | Query activity by distinct sites or total events, including groups and inverted activity conditions. |
| [functional_threshold_callbacks.py](functional_threshold_callbacks.py) | Retain threshold counts and activity reductions without storing a full voltage trace. |

## Reuse and continue an analysis

| Example | What it helps you do |
| --- | --- |
| [native_functional_callbacks.py](native_functional_callbacks.py) | Use one pure Population analysis across native `run`, `longrun`, checkpointed execution and functional simulation. |
| [native_network_functional_callbacks.py](native_network_functional_callbacks.py) | Apply the same scoped analysis to native and functional Networks while preserving recording cadence and gradients. |
| [native_step_functional_callbacks.py](native_step_functional_callbacks.py) | Carry observations through manually controlled native timesteps and longer runs. |
| [shared_callback_preparation.py](shared_callback_preparation.py) | Prepare named callbacks from a model or an existing functional plan for reuse. |
| [shared_callback_rebinding.py](shared_callback_rebinding.py) | Rebind prepared callbacks to a compatible receiving model when handing off a simulation. |
| [functional_callback_capture.py](functional_callback_capture.py) | Capture accumulated native callback results and recording phase before continuing functionally. |
| [functional_callback_checkpoints.py](functional_callback_checkpoints.py) | Save callback continuation and restore it into an equivalent reconstructed runner. |
| [functional_callback_checkpoint_protocol.py](functional_callback_checkpoint_protocol.py) | Give a custom pure callback a portable checkpoint configuration and resume it with receiving callbacks. |

Read [recording and online analysis](../../docs/advanced/recording_and_callbacks.md)
for callback configuration, field selection, scopes and storage.
[Initialization and checkpoints](../../docs/advanced/initialization_and_checkpoints.md)
explains how observations continue with simulation state; the
[callback API reference](../../docs/api/dendra.models.callbacks.rst) lists the
available configurations.

[All examples](../README.md)
