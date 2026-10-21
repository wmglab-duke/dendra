---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-event-delivery)=
# Functional NetCon event delivery

Lower an initialized `NetCon` with `dn.func.make_functional_netcon` to advance
event delivery using explicit weights, source observations, and queue state.
Declare connectivity through the ordinary `Network` API. The functional
projection returns updated queue state and the payload due for reception,
without changing the native connection or its input tensors.

Use this component interface to compose delivery with functional populations.
For complete Network execution, see {doc}`A19_functional_network_lowering` and
{doc}`A21_functional_network_compile`.

## Start with an ordinary Network

The downloadable
{download}`functional_event_delivery.py <../../examples/functional_event_delivery.py>`
example connects two source cells to two passive target cells through `expsyn`.
The source declares a dimensionless `gate` buffer. The same gate tape drives
the native and functional executions, and the script checks their queues,
synaptic conductances, voltages, and weight gradients.

```{literalinclude} ../../examples/functional_event_delivery.py
:language: python
:start-after: "# region declarations"
:end-before: "# endregion declarations"
```

```{literalinclude} ../../examples/functional_event_delivery.py
:language: python
:start-after: "# region standard-network"
:end-before: "# endregion standard-network"
```

The script's `GateInput` is an ordinary `Mechanism` with
`Mechanism.CARRY("gate")`; fresh initialization sets it to zero. The source
uses `threshold=None` and `pre_var="mech.source.gate"`, so its values are
already event gates. `diff_delays=False` selects hard-delay
semantics. Explicitly selecting the dense training backend ensures this
projection uses a dense delivery ring in both executions.

The fixed-delay example uses the active dense runtime, including tensor
schedules described in {doc}`A15_functional_tensor_schedules`. This example
uses raw gates; finite and mixed thresholds are also
supported, with explicit detector carry and ordered projection phases as
described in {doc}`A13_functional_threshold_detection`.
Two- and three-tap differentiable delay interpolation is described in
{doc}`A17_functional_delay_interpolation`. Compact source-history training is
described in {doc}`A22_functional_source_history`; intrinsic packed-history
inference is described in {doc}`A23_functional_bitpacked_inference`. `NetStim`
generation and RNG continuation are described in {doc}`A14_functional_netstim_renewal`.

## Extract the projection and its explicit tensors

```{literalinclude} ../../examples/functional_event_delivery.py
:language: python
:start-after: "# region lower-projection"
:end-before: "# endregion lower-projection"
:dedent: 4
```

`connection_tensors.weights` contains the effective per-connection weights.
The plan fixes the initialized routing, timestep, integer delays, and target
shape; use new explicit weights for optimization. It retains no evolving queue
inside the functional plan.

`connection_tensors.state` contains `delivery_buffer`, `current_time_step`, and
`global_step`. When event tracking is enabled, it also contains `events` and
`event_queue`. These names match the dense NetCon checkpoint state. Preserve
this complete tree when splitting or resuming a trajectory: synaptic state
alone cannot reconstruct pending deliveries.
Thresholded projections also include the bool `has_spiked` latch.

Source gates have the shape of the initialized source variable and must use
`projection.source_dtype` and `projection.source_device`. These are NetCon's
normalized `pre_dtype` and `pre_device`; cast a boolean source buffer to that
floating dtype before passing it as gates. The plan uses NetCon's captured
`pre_idx` to gather one value per connection. It preserves
signed and fractional values and their gradients; it does not convert them
to booleans. Weights have one entry per materialized connection. For this
point synapse, weights and delivered payloads are numerical conductances in
µS. See {doc}`../units` for other target unit contracts.

## Compose delivery and the population step

```{literalinclude} ../../examples/functional_event_delivery.py
:language: python
:start-after: "# region functional-step"
:end-before: "# endregion functional-step"
:dedent: 8
```

One projection step returns the queued payload due at this timestep and a new
connection state. It clears that due row, multiplies the current gates by the
current weights, scatters into future rows, and advances its ring and step
counters. It does not advance a Population or Network clock.

Call `FunctionalPopulation.receive` once per projection invocation, including
a zero payload. Preserve NetCon order when several projections target the same
receiver; combining their payloads can change a stateful receive rule. After
the connection phase, advance each population once. This example's gates are
explicit input tensors, so it contains no implicit event detector.

The combined projection step consumes gates already chosen by its caller. If
a receive changes a field used to observe source activity, that observation
must occur after reception. Use `projection.read_due`, receive the payload,
observe the source, then call `projection.advance`. These phases preserve the
native boundary without precomputing all source values at the start of a step;
see {doc}`A13_functional_threshold_detection` for a recurrent example.

Hard delays retain NetCon's nearest-step rounding and minimum one-step
latency. Even a physical delay rounded to zero arrives on the next step,
because reception has already consumed the current slot. A depth-one ring
still obeys this rule. Pending queue entries already contain their enqueue-time
weights: changing weights later affects newly queued events, not old traffic.

`projection.extract(connection)` snapshots compatible live connection state;
`projection.commit_state_(connection, state)` returns explicit state to a
compatible imperative connection. Population and owner clocks remain separate
state and must be managed consistently when composing or resuming execution.
This projection adapter does not commit a complete Network trajectory.

The complete example executes here:

```{code-cell} python
:load: ../../examples/functional_event_delivery.py
:tags: [remove-input]
```

## Compile and batch a composed step

Functional delivery returns new queue tensors without changing the incoming
state. Queues, weights, and source gates can therefore be mapped independently
with `torch.func.vmap`. Use `torch.compile` to compile a step that combines
projection delivery, reception, and Population integration, preserving the
connection order described above.

NetStim generation with explicit random draws and RNG continuation is described
in {doc}`A14_functional_netstim_renewal`; tensor schedules are described in
{doc}`A15_functional_tensor_schedules`, and consuming heap calendars in
{doc}`A16_functional_heap_schedules`.
Canonical single-device Network lowering is described in
{doc}`A19_functional_network_lowering`, and snapshot-based `dn.compile(Network)`
execution in {doc}`A21_functional_network_compile`. Users construct networks
through the standard Dendra API. See {doc}`A8_functional_populations` for
population execution and {doc}`../synapse_authoring` for authoring a pure receiver.
