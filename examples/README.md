# Examples

Runnable Python examples for building models, simulating networks, recording
responses and fitting parameters. Each folder groups scripts by task.

From the repository root, run a
[single compartment](getting_started/native_single_compartment.py) and record
its response to a current pulse.

```bash
python examples/getting_started/native_single_compartment.py
```

For a first run with `dn.compile`, choose the inference section of the
[functional overview](getting_started/functional_api_overview.py).

```bash
python examples/getting_started/functional_api_overview.py --section inference --backend eager
```

Run commands from the repository root with Dendra installed in your environment.
For a source checkout, `python -m pip install -e '.[solvers]'` also installs the
optional native solvers used by cable and tree examples. See the
[installation guide](../docs/installation.md) for requirements and device support.
Scripts with command-line options describe them in `--help`. Examples that plot
results need a graphical display unless you select a noninteractive Matplotlib
backend, such as `MPLBACKEND=Agg`.

## Find an example

| Task | Folder | A useful starting point |
| --- | --- | --- |
| Run a model and record voltage | [Getting started](getting_started/README.md) | [Single compartment](getting_started/native_single_compartment.py) |
| Connect cells, deliver events and run recurrent networks | [Networks](networks/README.md) | [Compile a Network](networks/functional_network_compile.py) |
| Supply currents, extracellular stimulation and event schedules | [Inputs](inputs/README.md) | [Shared deterministic inputs](inputs/shared_runtime_inputs.py) |
| Record voltage, LFP, spikes and activity; write HDF5 output | [Callbacks](callbacks/README.md) | [Recorder storage](callbacks/functional_recorder_storage.py) |
| Initialize models, train initial values and reuse steady-state caches | [Initialization](initialization/README.md) | [Initialize a compiled runner](initialization/functional_runner_initialization.py) |
| Resume simulations, save checkpoints and switch execution styles | [Continuation](continuation/README.md) | [Continue a simulation](continuation/functional_continuation.py) |
| Sample or replay noise, heterogeneous parameters and Poisson stimulation | [Stochastic models](stochastic/README.md) | [Shared runtime draw sampling](stochastic/shared_runtime_draw_sampling.py) |
| Write synapses, MaterialProcesses and model subclasses; choose integrators | [Model authoring](models/README.md) | [Custom event synapse](models/custom_event_synapse.py) |
| Fit parameters, select optimizer groups and differentiate descriptors | [Training and analysis](training/README.md) | [Fit HH conductances](training/functional_gradient_descent.py) |

## Native and functional execution

Native examples use `Population` and `Network` methods such as `run`.
`functional_` examples use `dn.compile` or the explicit `dn.func` lifecycle.
`shared_` examples demonstrate the same configuration or inputs across both
execution styles. These prefixes describe the workflow; the folder describes
the task.

For explanations alongside the code, see the [simulation guides](../docs/advanced.rst),
the [functional execution guide](../docs/advanced/functional_execution.md), and
the [synapse authoring tutorial](../docs/synapse_authoring.md).
