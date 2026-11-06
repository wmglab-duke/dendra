# Inputs and schedules

Use waveforms to describe stimulation in physical time, or `dn.func.Inputs` to
drive a model with tensors produced by data or another model. These examples
cover Population and Network inputs, extracellular contacts, and scheduled
NetStim or connection events.

Run from the repository root:

```bash
python examples/inputs/shared_runtime_inputs.py
python examples/inputs/functional_runtime_extra.py --backend eager
```

## Current and extracellular inputs

| Example | What it helps you do |
| --- | --- |
| [shared_runtime_inputs.py](shared_runtime_inputs.py) | Supply the same deterministic current or extracellular-voltage packets to native and functional execution. |
| [functional_runtime_inputs.py](functional_runtime_inputs.py) | Differentiate input tapes through recorded, continued Population and Network runs. |
| [functional_runtime_extra.py](functional_runtime_extra.py) | Compile with `extra=dn.VarInput()` and change contact counts or deterministic waveform types between calls, including training. |
| [functional_network_extra.py](functional_network_extra.py) | Bind extracellular contacts per Population and train shared fields and waveform parameters. |
| [shared_step_resolution.py](shared_step_resolution.py) | Use `dn.func.resolve_steps` to allocate input tapes for an execution horizon, including fractional duration carry. |

## Scheduled events

| Example | What it helps you do |
| --- | --- |
| [shared_schedule_bindings.py](shared_schedule_bindings.py) | Bind live event times and amplitudes for NetStim and NetCon schedules, then train and checkpoint them. |
| [functional_tensor_schedules.py](functional_tensor_schedules.py) | Combine scheduled values with indexed time and amplitude sources while preserving their gradients. |
| [functional_heap_schedules.py](functional_heap_schedules.py) | Consume scheduled calendars and resume their remaining events across native and functional checkpoints. |

Read [stimulation, input tapes and randomness](../../docs/advanced/inputs_and_randomness.md)
for units, tape shapes and the relationship to `Slice.inject`.
See [connected Networks](../../docs/advanced/connected_networks.ipynb) for event
connections and [initialization and checkpoints](../../docs/advanced/initialization_and_checkpoints.md)
for continuation.

[All examples](../README.md)
