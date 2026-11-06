# Models, mechanisms and integrators

Define custom dynamics once for native and functional execution. These examples
cover synapses, material processes, state histories, constrained parameters and
several numerical integrators.

Run from the repository root:

```bash
python examples/models/custom_event_synapse.py
python examples/models/functional_spd_solvers.py
```

The SPD and Tree solver examples require the optional `dendra[solvers]` package
extra; see [installation](../../docs/installation.md).

## Author model components

| Example | Use it to… |
| --- | --- |
| [`custom_event_synapse.py`](custom_event_synapse.py) | Implement pure event reception and use the synapse in a Network and with `dn.func`. |
| [`custom_continuous_synapse.py`](custom_continuous_synapse.py) | Implement a graded synapse driven by an analog release gate from another Population. |
| [`functional_material_processes.py`](functional_material_processes.py) | Define and train a custom deterministic MaterialProcess in a connected Network. |
| [`functional_mechanism_delays.py`](functional_mechanism_delays.py) | Use single-stream and multistream mechanism histories with differentiable delays. |
| [`functional_mechanism_injections.py`](functional_mechanism_injections.py) | Train mechanism-owned waveform parameters and continue their stimulation clock. |
| [`functional_tables.py`](functional_tables.py) | Train TABLE values in Population and Network models. |
| [`functional_torch_parametrizations.py`](functional_torch_parametrizations.py) | Register a PyTorch parametrization and retain shared trainable sources through compilation. |
| [`functional_population_subclasses.py`](functional_population_subclasses.py) | Author Population, Cable, Tree, extracellular-model and Network subclasses using inherited functional support. |

## Choose an integrator

| Example | Use it to… |
| --- | --- |
| [`functional_euler.py`](functional_euler.py) | Run Euler/RK1 cable models and mixed-integrator Networks. |
| [`functional_runge_kutta.py`](functional_runge_kutta.py) | Run RK2/RK4 cable models with native parity and differentiable continuation. |
| [`functional_dufort_frankel.py`](functional_dufort_frankel.py) | Use DF/DFH cable integrators and retain their two-level history across continuation. |
| [`functional_point_integrators.py`](functional_point_integrators.py) | Use explicit and two-level methods with Population and SingleCompartment models. |
| [`functional_spd_solvers.py`](functional_spd_solvers.py) | Train a scalar/block cable Network with SPD solvers, shared parameters and checkpoints. |

See [custom models](../../docs/advanced/custom_models.md) for hook and state
contracts, and the [synapse authoring tutorial](../../docs/synapse_authoring.md)
for complete event-driven and continuous synapse walkthroughs.

[All examples](../README.md)
