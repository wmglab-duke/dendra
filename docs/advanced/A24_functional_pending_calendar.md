---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-pending-calendar)=
# Pending calendars for inference

Functional Networks support per-NetCon schedules with `bitpacked_history` and
the complete `sparse_calendar` inference runtime. Native and functional
connections share pure Tensor calendar operations. Pending entries retain their
destination, weighted payload, and delivery slot; checkpoint continuation does
not recompute them from current parameters.

The runnable
{download}`functional_pending_calendar.py <../../examples/functional_pending_calendar.py>`
example checks both runtimes against native execution, changes weights between
calls, records voltage, resumes native checkpoints with pending traffic, and
handles insufficient capacity. Run it from the repository root with
`python examples/functional_pending_calendar.py`.

## Select a runtime and schedule connections

```{literalinclude} ../../examples/functional_pending_calendar.py
:language: python
:start-after: "# region declarations"
:end-before: "# endregion declarations"
:dedent: 4
```

Use `runtime="bitpacked_history"` or `runtime="sparse_calendar"` with this
example's builder. Both require evaluation mode and native initialization before
lowering. Schedules use the existing value or indexed reference bindings
described in {doc}`A15_functional_tensor_schedules`. NetStim's own schedules
produce source activity before the connection phase.

| Runtime | Intrinsic source activity | Per-NetCon scheduled activity |
| --- | --- | --- |
| `bitpacked_history` | Packed source history, current delivery-time weights | Pending weighted payloads, enqueue-time weights |
| `sparse_calendar` | Pending weighted payloads, enqueue-time weights | Combined with intrinsic gates before enqueue |

Packed inference retains the source-level admission rules in
{doc}`A23_functional_bitpacked_inference`. Sparse inference supports
per-connection threshold detection, explicit gates, and `track_events=True`.
Its separate count calendar retains scheduled counts and positive intrinsic
events even when weights are zero or weighted payloads cancel. Inference runners
suppress gradients; differentiable source history and delay interpolation use
the training runtimes described in {doc}`A22_functional_source_history`.

## Compile and continue

```{literalinclude} ../../examples/functional_pending_calendar.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 4
```

Each projection delivers due payloads before observing its source, then enqueues
new traffic and advances its counters. Deliveries to the same destination retain
native enqueue order and connection order. Packed inference adds pending
scheduled values after its intrinsic history projection. Scheduled multipliers
are evaluated when enqueuing and combined using the native schedule semantics.
Zero gates are filtered before multiplication by connection weights.

```{literalinclude} ../../examples/functional_pending_calendar.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 4
```

Weight replacements affect new enqueues. Already queued payloads keep their
original values, while packed intrinsic history uses the replacement weights at
delivery. Inference delay offsets are rounded, clamped to at least one timestep,
and frozen at initialization. A queued slot stays fixed. Changing physical
delay parameters requires reinitialization and lowering again; inference plans
use `delays=None` in explicit parameter packets. Ring addressing preserves native
modulo behavior.

## Size calendars and handle overflow

`pending_capacity` is the maximum number of entries per connection calendar,
independent of the number of ring slots. Supply one positive integer or a mapping
from event-connection names to counts to `dn.compile`,
`dn.func.make_functional_network`, or `FunctionalNetwork`. Explicit NetCon
lowering accepts one integer through `dn.func.make_functional_netcon`.

Intrinsic-only packed connections with no NetCon schedules or imported pending
traffic use zero-capacity calendar tensors and skip calendar transport. For
other connections, the default is the largest of 16, twice the edge count, and
the number of imported pending entries. An explicit positive capacity overrides
either default. This starting capacity does not guarantee enough space for every
firing rate and delay horizon. Sparse event tracking has a separate count calendar
with the same capacity. Explicit capacities too small for a checkpoint raise
during extraction. Native calendars grow automatically; compiled calendar shapes
remain fixed.

```{literalinclude} ../../examples/functional_pending_calendar.py
:language: python
:start-after: "# region overflow"
:end-before: "# endregion overflow"
:dedent: 4
```

An enqueue that exceeds capacity preserves existing entries and sets a sticky
Tensor overflow flag. Payload and event-count enqueues succeed together or fail
together. Host steps, rollouts, compiled chunks, and `dn.compile` runners reject
overflow before returning a result. They leave incoming carry, native state,
restart snapshots, RNG state, and callback carry unchanged. Retry from a valid
checkpoint with a larger-capacity plan; incomplete results are not continuations.

Direct `torch.compile` and functional transforms propagate the flag as Tensor
carry. After returning to ordinary host code, call
`projection.check_overflow(state)` or `network_plan.check_overflow(state)` before
consuming the result. Commit rejects overflowed carry.

## Preserve checkpoints and initialization

```{literalinclude} ../../examples/functional_pending_calendar.py
:language: python
:start-after: "# region checkpoint"
:end-before: "# endregion checkpoint"
:dedent: 4
```

Functional carry contains only Tensors. The adapter translates pending tables
to the existing native slot/chunk checkpoint format and accepts legacy pending
checkpoints. Keep matching topology, timestep, runtime, and initialized delay
offsets. Align the receiving Network's parameters before continuing execution.
Extraction detaches independent snapshots; commit clones incoming carry.

Snapshot restarts retain pending events. `initialization="fresh"` clears
calendars and overflow flags in native initialization order, alongside detector
and history carry. Explicit state continues without reinitialization. NetStim
clocks, RNG state, and consumed source-calendar cursors follow
{doc}`A21_functional_network_compile`.

The complete CPU example executes both runtimes and the overflow retry:

```{code-cell} python
:load: ../../examples/functional_pending_calendar.py
:tags: [remove-input]
```
