---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-heap-schedules)=
# Consuming heap calendars

NetStim's unit-amplitude Python schedules consume due events from a fixed
calendar. Declare events through `dn.NetStim.schedule`, then lower the
initialized source with `dn.func.make_functional_netstim`. Functional execution
represents the calendar as tensor constants and the consumed position as
explicit state, so checkpoint continuation retains which events have fired.

The complete
{download}`functional_heap_schedules.py <../../examples/functional_heap_schedules.py>`
example constructs a standard Network with renewal, heap events, and tensor
events. It checks native trajectories, saves cloned carry, resumes that saved
state through both execution APIs, and compares nonzero interval and weight
gradients. Run it with `python examples/functional_heap_schedules.py`.

## Declare consuming events

Python times and unit scalar amplitude select the heap calendar by default.
Trainable tensor times, non-unit scalar amplitudes, tensor/list amplitudes,
indexed references, or `differentiable=True` select the tensor schedule
described in {doc}`A15_functional_tensor_schedules`. An explicit
`differentiable=False` forces the unit-amplitude heap path.
Both storage policies can coexist on one source:

```{literalinclude} ../../examples/functional_heap_schedules.py
:language: python
:start-after: "# region declarations"
:end-before: "# endregion declarations"
:dedent: 8
```

Index expansion and `allow_past` filtering happen when scheduling events.
Appending or clearing events changes calendar topology and requires lowering
again. Insertion inside a transformed loop is not part of this interface.
NetCon's value and indexed schedules retain tensor evaluation; the consuming
calendar belongs to NetStim sources.

## Extract constants and cursor carry

Extract the source's parameters, constants, state, and schedule packet:

```{literalinclude} ../../examples/functional_heap_schedules.py
:language: python
:start-after: "# region lower"
:end-before: "# endregion lower"
:dedent: 4
```

Every `NetStimTensors` includes calendar fields, including sources without events:

| Dictionary field | Meaning |
| --- | --- |
| `constants["heap_times"]` | One-dimensional event times, grouped in sorted rows, followed by a positive-infinity sentinel. Float64 on CPU/CUDA and the simulation dtype on MPS; these values do not require gradients. |
| `constants["heap_row_offsets"]` | Long offsets of length `number_of_state_elements + 1`, delimiting each flattened generator's row. |
| `state["heap_cursor"]` | Long vector with one absolute position per row, between that row's start and end offsets. It advances without modifying the calendar. |
| `state["heap_head_override"]` | State-shaped simulation-dtype values retained for old head-only checkpoints. |
| `state["heap_override_active"]` | State-shaped Boolean mask selecting a legacy head override instead of the calendar head. Ordinary calendars leave it false. |

These fields accompany `max_spikes`, the renewal clock, counts, `t_last`, and
both RNG states described in {doc}`A14_functional_netstim_renewal`. Leading
native batch dimensions are included in the flattened row order. Extraction
creates independent ordinary tensors even inside `torch.inference_mode()`.

Keep calendar constants unchanged throughout a lowered run. Hot calls check
their schemas without scanning their values or performing host synchronization;
they trust the calendar packet from extraction. Extraction and state commit
check topology, ordering, sentinel, and cursor bounds outside transformed code.
They reject a native destination whose calendar differs from the frozen plan.

On CPU/CUDA, calendar times retain Float64 precision even when simulation state
uses Float32. MPS uses the simulation dtype because it cannot store Float64;
native cold metadata and checkpoints retain the original Python times. Heads
and due-event comparisons cast selected entries to the simulation dtype,
preserving native comparison behavior. Native dtype/device changes rebuild
the calendar from those original times when needed; lower again afterward.

## Consume overdue events with native ordering

Generate the source once, read and receive due connection deliveries, enqueue
the current source gates, then advance populations:

```{literalinclude} ../../examples/functional_heap_schedules.py
:language: python
:start-after: "# region step"
:end-before: "# endregion step"
:dedent: 4
```

Each source step reads the calendar head before evaluating renewal.
If a finite head is due and the source has not reached `max_spikes`, it consumes
the complete overdue prefix. Duplicate and skipped-over heap events coalesce
into one Boolean source event in that call. A simultaneous stochastic or tensor
event still increases `spike_counts` at most once. Repeated or decreasing call
times do not rewind consumed positions. At the spike cap, overdue heap entries
remain unconsumed.

This is different from stateless tensor evaluation: tensor events reevaluate
when a time is repeated, and a skipped hard grid step can miss a tensor event.
Heap events wait until a sampled time reaches their finite head. Renewal still
advances at most once per call; it does not catch up every missed stochastic
event. Use Network's `origin + integer_step * dt` clock construction when
comparing with ordinary Network execution.

Heap times are non-differentiable. The surrogate gate retains the native
minimum of stochastic clock and heap head *before* renewal, including its
coupling to clock and interval gradients. Outgoing training NetCons use
`diff_spiking=True` and consume `output.spike_gate`. Hard connections consume
`output.spikes.to(projection.source_dtype)` instead. `bptt=False` detaches
the renewed stochastic clock while leaving current-step surrogate gradients
available, as described in the renewal guide.

Positive and negative infinity retain native behavior: only finite heads can
trigger heap firing. A nonfinite head alone therefore remains in place;
the surrogate minimum still follows ordinary tensor arithmetic. If a finite
head triggers firing at a positive-infinity call time, the overdue prefix can
include later positive-infinity entries. Calendars
containing NaN retain their legacy Python `heapq` execution in native mode,
because NaN does not define a sortable event order. Functional lowering rejects
those calendars explicitly instead of silently sorting them into new behavior.

## Save and resume the consumed position

Save source carry together with queues, population carry, fixed model values,
the clock origin, and the next integer timestep. Cloning carry copies storage
while retaining autograd history within this example's fixed-parameter run:

```{literalinclude} ../../examples/functional_heap_schedules.py
:language: python
:start-after: "# region checkpoint"
:end-before: "# endregion checkpoint"
:dedent: 12
```

The example resumes the saved state and checks every remaining step:

```{literalinclude} ../../examples/functional_heap_schedules.py
:language: python
:start-after: "# region functional_resume"
:end-before: "# endregion functional_resume"
:dedent: 4
```

For durable checkpoints across optimizer updates, serialize or clone/detach
parameters, constants, and schedule values as well as carry. Detached carry
starts a new gradient history; live cloned carry can continue BPTT.

`initial_state` resets the renewal clock, counts, and initialization RNG.
It preserves consumed heap cursors and clears legacy head overrides,
matching native `initialize()`. Clearing consumption requires declaring a fresh
calendar or restoring a checkpoint containing the desired earlier cursor.

Cold state commit restores clocks, consumed positions, legacy head carry, and
both RNG streams into a compatible source. Parameters, calendar constants,
and tensor schedule values remain those configured on the destination.
Synchronize each projection, population, and the Network owner's clock at a
complete timestep boundary. The example first advances a second Network to
that boundary, deliberately changes its source, then restores the saved carry:

```{literalinclude} ../../examples/functional_heap_schedules.py
:language: python
:start-after: "# region native_resume"
:end-before: "# endregion native_resume"
:dedent: 4
```

Modern native checkpoints retain the immutable calendar and cursor separately.
Restoring ordinary carry into matching topology preserves a warmed plan.
Older `sched_heaps` snapshots contain only remaining events; native restore
packs those events into a calendar, which can require lowering again because
the original consumed prefix is unavailable. Old head-only snapshots become
explicit override carry. Their first actual heap firing clears the override
and consumes the underlying overdue prefix; initialization clears it directly.

The complete example executes here:

```{code-cell} python
:load: ../../examples/functional_heap_schedules.py
:tags: [remove-input]
```

Pure calendar evaluation supports `torch.compile` and `torch.func`, including
mapped cursor state and empty mapped lanes. Sampling and cold extraction/commit
stay outside transformed graphs. Dynamic event insertion is currently
unsupported. NetStim calendar events feed dense, packed-history, or
sparse-calendar connections; compact pending transport is described in
{doc}`A24_functional_pending_calendar`. Canonical
single-device Network lowering is described in
{doc}`A19_functional_network_lowering`, and snapshot-based `dn.compile(Network)`
execution in {doc}`A21_functional_network_compile`.
