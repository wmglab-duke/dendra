---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-mechanism-delays)=
# Compile mechanisms with delay histories

Mechanisms can retain delayed copies of their own signals without defining a
Network connection. A single-stream history delays one tensor by a fixed number
of accepted steps. A multistream history groups several signals with different
integer delays into one queue. Native and functional execution use the same
pure history evaluation.

The complete
{download}`functional_mechanism_delays.py <../../examples/functional_mechanism_delays.py>`
example checks native Population trajectories, differentiates a delayed current,
and continues a compiled connected Network. Run it from the repository root
with `python examples/functional_mechanism_delays.py`.

## Author a delayed mechanism

Register a history during construction, after `Mechanism.__init__`. Declare its
queue and scalar integer pointer as `CARRY` when they should follow the ordinary
carry initialization contract. Register the full runtime payload shape,
including explicit batch axes; a broadcast parameter can be expanded to
`self.shape_f`. A deferred queue shape accommodates its history axis. Initialize it explicitly with `initial_delay_values` and return history
updates from `advance`:

```{literalinclude} ../../examples/functional_mechanism_delays.py
:language: python
:start-after: "# region single-stream"
:end-before: "# endregion single-stream"
```

`delayed_state_values(name, value, values)` returns `(delayed, updates)`.
`value` is the signal arriving on this accepted step; `values` contains the
incoming queue and pointer. Return the complete `updates` mapping from
`advance` so that the history crosses the timestep boundary. Treat every input
as read-only. The returned queue and pointer have the registered shape, dtype,
and device.

The delay is measured in calls to the helper. With `delay_steps=2`, a value
submitted on call 0 is returned on call 2. A zero delay returns the current
signal and leaves queue and pointer unchanged. The initial delayed values are
zeros in this example. Advancing the helper once in `advance` makes the delay
correspond to accepted simulation steps; current evaluation can occur more
than once per step and should only read the saved `delayed` carry.

`advance` reads the same incoming state snapshot as the mechanism's State
bundles. It does not read their newly solved states. Ordered mechanism visits,
current assembly, and connection delivery follow the ordinary native execution
order.

Native `delayed_state` and `delayed_states` retain their imperative interfaces.
Use their `_values` counterparts inside pure hooks because the imperative
methods install history updates on the module.

`VoltageProcess.voltage_values` can use the same pure helpers. Include their
queue and pointer updates in its returned `(voltage, updates)` pair to advance
the history in the voltage phase, before gating and current assembly. Update
each history in one hook per accepted step; calling its helper from both
`voltage_values` and `advance` advances it twice.

## Group heterogeneous pathways

Use `register_delayed_states` and `delayed_states_values` to evaluate several
streams together. This example prepends a three-stream axis to the mechanism's
local payload, assigns delays of 0, 1, and 3 steps, and inserts a history axis
before it:

```{literalinclude} ../../examples/functional_mechanism_delays.py
:language: python
:start-after: "# region multi-stream"
:end-before: "# endregion multi-stream"
```

`stream_axis` identifies the stream axis of the input payload. `delay_axis`
identifies where the queue's history axis is inserted. Both can select arbitrary
valid axes. Every stream shares a queue depth of `max(delay_steps) + 1` and one
circular pointer. Streams with zero delay read the current signal, including
when other streams have positive delays.

`supports_population_axis_multistream_delays=True` declares that the mechanism
handles the Population support axes explicitly. Keep the stream axis separate
from Population, compartment, and explicit batch axes; the example does so by
prepending it. A mechanism's delayed signal must retain its registered shape.
Regional support and explicit Population batches follow the mechanism's
ordinary support layout.

## Choose gradient semantics

`mode="shift"` constructs a new queue at each call. Gradients pass through the
source signal, prior history, upstream parameters, and downstream dynamics.
The integer delay counts are structural configuration and are not trainable.
Use Network connection delays when a connection with a differentiable delay is
needed; see {ref}`functional-delay-interpolation`.

`mode="circular"` detaches new writes from their submitted source signals.
Zero-delay streams pass the current signal through and retain its gradients.
With gradients enabled and an incoming graph-connected history, reverse-mode
preserves the destination's gradient identity, while forward-mode zeroes
overwritten-slot tangents, matching native `no_grad` writes. Functional execution returns new
queue tensors with these same rules. Shift mode gives an ordinary
differentiable history recurrence.
Functional calls under `torch.no_grad()` or `torch.inference_mode()` do not
retain graph connections through newly constructed histories.

`mode="auto"` selects shift mode when the mechanism is training or gradients
are enabled, and circular mode when the mechanism is in evaluation with
gradients disabled. Configure the mechanism's training mode before lowering;
compiled execution specializes to the selected execution context.

```{literalinclude} ../../examples/functional_mechanism_delays.py
:language: python
:start-after: "# region train-delay"
:end-before: "# endregion train-delay"
:dedent: 4
```

Scan execution supports autograd and checkpointed scan. While execution is
inference-only. Record the mechanism's output carry, such as `delays.delayed`,
using the ordinary Recorder interface.

## Initialize and continue

Snapshot initialization starts from the captured history and pointer. Fresh
initialization follows the mechanism's authored hooks. In this example,
`initial_delay_values(values)` returns zero queue and pointer overlays, so
fresh initialization clears the history. Pass particular registered history
names to reset only those histories.

Registration alone does not add an implicit reset to every initialization.
Registered histories without declared `CARRY` persist across native and
functional fresh initialization unless `initial_values` returns a reset
overlay. A steady-state cache restores its saved history, pointer, and
multistream delay vector, following the ordinary cache contract.

```{literalinclude} ../../examples/functional_mechanism_delays.py
:language: python
:start-after: "# region network-continuation"
:end-before: "# endregion network-continuation"
:dedent: 4
```

Calling a compiled runner without a state restarts its chosen initialization.
Passing returned simulation and callback state continues the histories and
recording. History buffers and pointers are included in functional state,
`functional.commit_state_`, native runtime checkpoints, and full model
`state_dict()` checkpoints. No separate history input tape is needed.

Multistream integer delay vectors are functional constants in each
Population's `mechanism_delays.*` namespace. Explicit replacement vectors must preserve their shape,
dtype, registered maximum delay, and whether any or all streams have zero
delay. This also permits a steady-state cache to restore a compatible saved
vector. Preserve the returned constants when continuing a cached run.

`functional.commit_state_` transfers simulation state; it does not copy a
different functional integer delay vector into the Native target. Align the
target's registered delay vector separately before Native continuation when
its routing differs. Native runtime and full model checkpoints include their
registered delay vectors along with the histories.

Configure fixed delays, queue shapes, axes, and modes before lowering. Editing
the Native registration or source delay vector after lowering requires a new
functional plan or compiled runner. Resizing a delay history during a pure
step and legacy mutating initialization hooks are currently unsupported.
General Population and Network admission rules still apply.

## Run the complete example

```{code-cell} python
:load: ../../examples/functional_mechanism_delays.py
:tags: [remove-input]
```
