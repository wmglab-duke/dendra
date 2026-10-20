---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-threshold-detection)=
# Shared threshold detection and ordered projection phases

An initialized dense NetCon can be lowered with finite thresholds as well as
explicit raw gates. Native and functional execution share the detector in
the network layer and the fixed-delay transport described in
{doc}`A12_functional_event_delivery`. Declare connectivity once through an
ordinary `Network`, then supply source observations and carry explicitly.

## Receive before observing the source

Each native connection reads its due payload, calls its receiver, observes
the source, detects events, and enqueues future deliveries. Preserve that
sequence for each projection in Network insertion order. After all
connections, advance each population once.

`projection.read_due(state)` returns `(delivery_state, payload)`. It publishes
the due tracked event counts in `delivery_state`, while retaining the queue,
detector latch, and counters. Deliver the payload exactly once, including a
zero payload. Then obtain the source's current value and call
`projection.advance(weights, delivery_state, source_values)` to update the
detector, clear the due row, enqueue new events, and advance the counters.

The two phases matter when reception changes a field used by a later source
observation. Precomputing all source values at the start of the connection
phase changes that behavior. The combined `projection.step(...)` remains
useful when the supplied observations are already chosen independently of
reception; its `source_gates` argument now also accepts observations for a
thresholded projection.

## Complete recurrent example

This diagnostic relay receives its own events. Every receive updates a
dimensionless `level`, including receives with zero payload. The level rises
through zero, remains above threshold, and later resets below it. A bool latch
ensures that another event requires a new upward crossing.

```{literalinclude} ../../examples/functional_threshold_detection.py
:language: python
:start-after: "# region declarations"
:end-before: "# endregion declarations"
```

The network is constructed using the standard API. Its source is the relay's
registered carry field; ordinary membrane voltage (`pre_var="v"`) is also
supported. Threshold units follow the source variable, while weights follow
the target receiver's contract.

```{literalinclude} ../../examples/functional_threshold_detection.py
:language: python
:start-after: "# region standard-network"
:end-before: "# endregion standard-network"
```

Lower with `dn.func.make_functional_netcon(connection)` and lower the target
population with `dn.func.make_functional`. The executable
{download}`functional_threshold_detection.py <../../examples/functional_threshold_detection.py>`
contains the extraction and preparation calls and checks both hard and
surrogate trajectories against ordinary Network execution. Its step is:

```{literalinclude} ../../examples/functional_threshold_detection.py
:language: python
:start-after: "# region split-step"
:end-before: "# endregion split-step"
:dedent: 8
```

On the first invocation, reception moves `level` from -0.5 to +0.5, so the
detector queues an event. Observing the source before reception would miss
that event. In a larger network, observe from the current functional source
state after the corresponding receiver; calling `read_due` alone does not
update any population.

## Detector state and gradients

Finite thresholds use the same NetCon rules in both execution paths:

- Equality counts as above threshold: `value >= threshold`.
- The `has_spiked` carry is a bool recording the previous above-threshold
  condition. Fresh initialization sets it to false, so an initially above
  observation fires.
- Staying above does not fire again; an observation below threshold rearms
  the detector.
- With `diff_spiking=True`, forward gates remain hard. Gradients follow
  `sigmoid((value - threshold) / tau) * (1 - previous_above)`. The bool latch
  carries no gradient across observations.

The initialized NetCon training configuration fixes the hard/surrogate policy
and `tau` in the lowered plan; evaluation uses hard detection. Thresholds and
their NaN mask are frozen topology metadata in this interface. Optimize the
explicit weights and source observations; lower again to adopt different
threshold metadata or detector configuration.
Use a real numeric scalar for `tau`; Tensor-valued temperatures are not part
of this frozen configuration interface.
Calling `set_diff_config` again invalidates the source-bound plan, including
when supplying the same settings. Configure the native connection before
lowering it.

Source observations use `projection.source_shape`, `source_dtype`, and
`source_device`. Thresholds are per connection, so edges sharing one source
can have different thresholds. A NaN threshold means a raw numeric gate on
that edge, retaining signed/fractional values and gradients. Entirely raw
Population projections have no `has_spiked` carry. Mixed projections retain the native
per-connection bool carry, including masked lanes.

Preserve the complete checkpoint-compatible connection state, including
`has_spiked`. Saving queues without the latch can cause an already high source
to fire again after resuming. `extract` and `commit_state_` transfer carry to
compatible initialized native connections; population and Network clocks
remain separate state. For continuation with ordinary `Network.step`, commit
at a complete timestep boundary with every projection, population state, and
owner clock synchronized.

The complete example executes here:

```{code-cell} python
:load: ../../examples/functional_threshold_detection.py
:tags: [remove-input]
```

This interface supports dense fixed-delay connections, including tensor
schedules described in {doc}`A15_functional_tensor_schedules`.
NetStim-generated values bypass threshold detection and retain the native
checkpoint schema. Shared pure generation with explicit random draws and
clock/RNG continuation is described in {doc}`A14_functional_netstim_renewal`.
Consuming heap calendars are described in {doc}`A16_functional_heap_schedules`.
Complete Network lowering remains a later extension.
