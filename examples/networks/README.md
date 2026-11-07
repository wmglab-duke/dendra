# Networks

Build connected models, deliver events or continuous signals, and run a Network
with explicit state. Start with `functional_network_compile.py` for the usual
`dn.compile(network, ...)` workflow. The ring example illustrates native Network
construction and displays the resulting voltage traces.

Run from the repository root:

```bash
python examples/networks/functional_network_compile.py
python examples/networks/ring_network.py
```

## Build and execute a Network

| Example | What it helps you do |
| --- | --- |
| [functional_network_compile.py](functional_network_compile.py) | Compile a recurrent Network, train shared connection parameters, record responses and continue a run. |
| [functional_network_lowering.py](functional_network_lowering.py) | Work with one explicit connected timestep when composing your own execution loop. |
| [functional_network_rollout.py](functional_network_rollout.py) | Run an explicit Network rollout with per-Population inputs, recording and checkpoint continuation. |
| [functional_network_while.py](functional_network_while.py) | Run connected inference with while execution, scoped recordings and explicit continuation. |
| [ring_network.py](ring_network.py) | Connect four HH neurons in a ring and visualize delayed propagation after a NetStim event. |
| [dual_pop.py](dual_pop.py) | Construct two labeled groups with recurrent excitation and cross-group inhibition, then plot their mean voltages. |

## Connections, sources and delays

| Example | What it helps you do |
| --- | --- |
| [functional_event_delivery.py](functional_event_delivery.py) | Drive a lowered NetCon with explicit event gates and apply its deliveries to a Population. |
| [functional_threshold_detection.py](functional_threshold_detection.py) | Detect threshold crossings and deliver events with the same timing as native execution. |
| [functional_netstim_renewal.py](functional_netstim_renewal.py) | Share one NetStim between projections and resume its clock and random stream through a native checkpoint. |
| [functional_connection_parameters.py](functional_connection_parameters.py) | Differentiate raw connection weight and delay sources, including constrained parameters and shared leaves. |
| [functional_netstim_parameters.py](functional_netstim_parameters.py) | Train NetStim parameter sources without losing sharing or continuation state. |
| [functional_continuous_connectivity.py](functional_continuous_connectivity.py) | Connect Populations through continuous transforms and differentiate the connected response. |
| [functional_delay_interpolation.py](functional_delay_interpolation.py) | Train interpolated delays and compare continued delivery with checkpoint replay. |

## Event storage and inference

| Example | What it helps you do |
| --- | --- |
| [functional_source_history.py](functional_source_history.py) | Train source-history connections and continue delivery after changing weights or delays. |
| [functional_bitpacked_inference.py](functional_bitpacked_inference.py) | Run packed event inference with scheduled sources and restore its state into native execution. |
| [functional_pending_calendar.py](functional_pending_calendar.py) | Run scheduled packed or sparse traffic and preserve pending deliveries across checkpoints. |

See the [connected Network guide](../../docs/advanced/connected_networks.ipynb),
the [network construction tutorial](../../docs/basics/03_networks.ipynb), and
[functional execution](../../docs/advanced/functional_execution.md).
For scheduled event inputs, see the [input examples](../inputs/README.md).

[All examples](../README.md)
