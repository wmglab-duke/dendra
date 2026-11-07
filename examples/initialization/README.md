# Initialization

Choose initial conditions, reuse a steady-state cache, or start a fresh training
trial while retaining the parameters you want to optimize. These examples cover
both Populations and Networks.

Run from the repository root:

```bash
python examples/initialization/functional_explicit_initialization.py --backend eager
python examples/initialization/functional_packet_initialization.py --backend aot_eager
```

| Example | Use it to… |
| --- | --- |
| [`functional_explicit_initialization.py`](functional_explicit_initialization.py) | Initialize a lowered model from explicit sources and run the returned continuation. |
| [`functional_runner_initialization.py`](functional_runner_initialization.py) | Call `dn.func.initialize(program)` before starting a compiled run or recording. |
| [`functional_packet_initialization.py`](functional_packet_initialization.py) | Start independent training trials from a continuation's effective parameters and current RNG position. |
| [`functional_network_initialization.py`](functional_network_initialization.py) | Compile a Network with fresh initialization, explicit random draws and differentiable initial conditions. |
| [`functional_network_cache_initialization.py`](functional_network_cache_initialization.py) | Reuse Network-wide or individual Population steady-state caches during fresh initialization. |
| [`functional_initialization_sources.py`](functional_initialization_sources.py) | Train Ion and Material initializers, including constrained positive values. |
| [`functional_initialization_parametrizations.py`](functional_initialization_parametrizations.py) | Share parametrized Ion and Material initializers across Populations and differentiate their sources. |
| [`functional_initialization_ordering.py`](functional_initialization_ordering.py) | Apply initialization transforms to shared parameters in Network Population order. |

See [initialization and checkpoints](../../docs/advanced/initialization_and_checkpoints.md)
for snapshot versus fresh starts, and [model initialization](../../docs/advanced/model_initialization.rst)
for steady-state preparation. [Stochastic examples](../stochastic/README.md) cover
capturing and replaying initialization draws.

[All examples](../README.md)
