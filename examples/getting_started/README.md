# Getting started

Start here to run an ordinary Dendra model with `dn.compile`, record its
response, and differentiate a simulation. The examples also show how to take
explicit control of parameters, state and continuation when an application
needs it.

Run from the repository root:

```bash
python examples/getting_started/functional_api_overview.py --section inference --backend eager
python examples/getting_started/functional_compiled_runner.py --backend aot_eager --execution scan --grad
```

`eager` and `aot_eager` keep startup short while exercising the functional
execution paths. The examples that accept `--backend inductor` can also generate
optimized kernels; their first call includes compilation.

| Example | What it helps you do |
| --- | --- |
| [functional_api_overview.py](functional_api_overview.py) | Explore inference, training, explicit state, random draws, extracellular inputs and function transforms in small independent sections. |
| [functional_compiled_runner.py](functional_compiled_runner.py) | Compile one HH model, choose scan or while inference, and try recording, training, chunking and activation checkpointing. |
| [functional_execution_modes.py](functional_execution_modes.py) | Compare direct functional execution, compiled execution and differentiation on the same model. |
| [functional_runner_execution.py](functional_runner_execution.py) | Use `dn.func.run` with compiled Population and Network programs, then continue or restore a saved run. |
| [functional_lowering_api.py](functional_lowering_api.py) | Lower a model, inspect its sources, bind its tensors and schedule execution explicitly. |
| [functional_packet_execution.py](functional_packet_execution.py) | Run a `Continuation` through an explicit plan or compiled chunk while retaining recording and gradients. |
| [functional_runner_horizons.py](functional_runner_horizons.py) | Set `steps=` or `duration=` at compilation and per call, including continued fractional durations. |
| [shared_execution_horizons.py](shared_execution_horizons.py) | Choose equivalent execution horizons for native and functional Populations and Networks. |

Read the [functional execution guide](../../docs/advanced/functional_execution.md)
for the main workflow and
[initialization and checkpoints](../../docs/advanced/initialization_and_checkpoints.md)
for starting, continuing and restoring simulations.

[All examples](../README.md)
