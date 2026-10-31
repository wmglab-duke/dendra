---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-tensor-schedules)=
# Tensor schedules and indexed sources

Functional NetStim and NetCon schedules accept explicit event times,
amplitudes, and indexed source vectors, including differentiable values.
Declare events through `schedule` before lowering. Each event's time and
amplitude can independently use a value or an index into a bound source;
`time_idx` and `weight_idx` can be supplied together. The `schedule_ref` and
`schedule_time_ref` convenience methods use the same declaration machinery.
Event routing and reference indices are fixed for a lowered plan; floating
values remain explicit inputs.

Index expansion and `allow_past` filtering happen when events are declared,
before lowering. Editing packet values reevaluates retained events; it does
not restore events already removed by that declaration-time filter.

The complete
{download}`functional_tensor_schedules.py <../../examples/functional_tensor_schedules.py>`
example constructs a standard Network and compares its functional voltage,
queue, RNG, and shared-source gradients with native execution. Run it with
`python examples/functional_tensor_schedules.py`.

## Declare value and reference events

Times are in milliseconds. Schedule amplitudes are dimensionless multipliers;
NetCon base weights retain the target receiver's units. Reference events gather
from bound one-dimensional source tensors by integer index. Multiple events,
and multiple schedule owners, can use the same source entry.

Choose the time and amplitude independently:

| Event time | Event amplitude | `schedule` arguments |
| --- | --- | --- |
| Value | Value | `times_ms=times, weight=amplitudes` |
| Value | Indexed source | `times_ms=times, weight_idx=amplitude_indices` |
| Indexed source | Value | `time_idx=time_indices, weight=amplitudes` |
| Indexed source | Indexed source | `time_idx=time_indices, weight_idx=amplitude_indices` |

Bind each referenced source with `bind_time_source` or `bind_weight_source`
before declaring the events. NetStim accepts its target generator indices as
`indices`; NetCon accepts `con_indices` or `pre_indices`. NetStim also accepts
`times` as an alias for `times_ms`. Existing `schedule_ref` and
`schedule_time_ref` calls select indexed amplitudes and indexed times,
respectively.

Value and reference arguments for the same field are mutually exclusive.
With `weight_idx`, omit `weight`: the indexed amplitude is used directly,
without an added unit amplitude. The default numerical `weight=1.0` and
`weight=None` also select that indexed amplitude; a Tensor or non-unit numerical
`weight` cannot accompany `weight_idx`. Canonical reference indices must be
nonnegative and within the bound source.

The complete
{download}`shared_schedule_bindings.py <../../examples/shared_schedule_bindings.py>`
example binds the same trainable time and amplitude vectors to NetStim and
NetCon, indexes both for each event, and compares native and functional
optimizer segments and checkpoint continuation:

```{literalinclude} ../../examples/shared_schedule_bindings.py
:language: python
:start-after: "# region declarations"
:end-before: "# endregion declarations"
:dedent: 4
```

Value-mode tensors remain declaration-time configuration. Expansion, filtering
and concatenation can retain nonleaf graphs even when the supplied tensor is a
leaf. `Network.detach()` does not reconstruct those expressions. For repeated
optimizer segments, use indexed source leaves for the fields that change;
replace or redeclare value-mode tensors whose graphs have been consumed.
Arbitrary external nonleaf source expressions remain caller-owned.

The following example also shows value-only schedules and the convenience
reference methods:

```{literalinclude} ../../examples/functional_tensor_schedules.py
:language: python
:start-after: "# region declarations"
:end-before: "# endregion declarations"
:dedent: 8
```

NetStim's unit-amplitude Python schedule uses consuming heap calendars. Select
tensor storage with `differentiable=True`, tensor-valued trainable times, or
non-unit/tensor amplitudes. Indexed reference methods also use tensor storage.
Any indexed field selects tensor storage for canonical `schedule` calls;
`differentiable=False` cannot be combined with indexed fields because it selects
the consuming calendar. For hard tensor scheduling, use
`set_diff_config(diff_scheduled_times=False)`.
Functional NetStim lowering also accepts consuming heap calendars with explicit
cursor carry, as described in {doc}`A16_functional_heap_schedules`.

Configure `diff_scheduled_times`, `sched_width`, and the timestep before
lowering. `Network.initialize(dt)` sets the NetStim schedule timestep;
standalone sources use `source.set_dt(dt)`. The functional source freezes this
timestep, while NetCon uses its initialized timestep and explicit `global_step`
carry. Native NetStim's per-call `dt` override is not a separate functional
input; configure the desired fixed value before extraction.

## Extract explicit packets and preserve intentional sharing

`NetStimTensors.schedule` and `NetConTensors.schedule` hold
`dn.func.TensorScheduleTensors` when events or bound sources are present.
They are `None` for an entirely unscheduled, unbound owner. A scheduled plan
requires `schedule=packet` on every numerical step or projection advance.

| Packet field | Meaning |
| --- | --- |
| `times` | Floating per-event value-mode time slots. |
| `amplitudes` | Floating per-event value-mode amplitude slots. |
| `time_source` | Bound time vector, or `None` when unbound. |
| `amplitude_source` | Bound amplitude vector, or `None` when unbound. |

The plan stores routing and reference indices. A resolved event value is its
value-mode slot plus the indexed source contribution when that reference is
present. Extraction creates independent leaves, preserving each tensor's
gradient flag and each source vector's dtype. Gradients from repeated indices
accumulate into the shared source entry with the native gather/cast policy.

Whole-Network lowering preserves shared source aliases across schedule owners.
Separate NetStim and NetCon extractions create independent snapshots even if
their native source vectors were shared. For separate extractions, preserve
intended parameter sharing by supplying the same explicit tensor to both packets:

```{literalinclude} ../../examples/functional_tensor_schedules.py
:language: python
:start-after: "# region lower"
:end-before: "# endregion lower"
:dedent: 4
```

`packet._replace(...)` is convenient for optimizer values or shared bindings.
Keep the original field shapes, dtypes, and devices; event/reference topology
does not change when floating values change. Gradients reach the explicit
leaves, independently of the source modules.

## Preserve ordering and native kernel policies

Generate NetStim activity once per timestep. Each projection then reads due
deliveries, invokes its receiver, observes the current source, evaluates its
own schedule, and enqueues new deliveries. Advance populations after all
projections. `read_due` remains independent of the schedule; pass the packet
to `advance` or the combined `step`.
If a reference vector comes from dynamic population state, supply its current
functional value when constructing the packet at that phase, after the
corresponding receive.

```{literalinclude} ../../examples/functional_tensor_schedules.py
:language: python
:start-after: "# region step"
:end-before: "# endregion step"
:dedent: 8
```

The two native scheduling policies remain distinct:

| Policy | NetStim | NetCon |
| --- | --- | --- |
| Clock | Explicit per-state absolute time `t`. | Integer `state["global_step"]`. |
| Hard schedules | First causal grid step at or after the event time. | The same causal assignment, compared with the global step. |
| Differentiable times | When enabled, triangular amplitudes around continuous time in either mode; width is `max(sched_width, 1e-6)` steps. | In training with the flag enabled, triangular amplitudes around the global step; denominator is `sched_width + 1e-6` steps. Evaluation remains hard. |
| Event bookkeeping | Boolean causal hard spikes, even while triangular amplitudes overlap. Counts coalesce to one emitted spike per generator per call and obey `max_spikes`. | Counts contributing events per connection, including zero/signed amplitudes; triangular evaluation counts nonzero kernel support. |

With `diff_scheduled_times=False`, time selection is hard and does not carry
time gradients. Schedule amplitudes can still carry gradients. With
triangular time evaluation enabled, the actual scheduled contribution is a
weighted triangular amplitude; it can be fractional or signed. It is added to intrinsic
activity. This differs from the hard-forward straight-through estimator used
for intrinsic threshold/renewal events.

For NetStim scheduled amplitude/time learning, outgoing training NetCons must
use `diff_spiking=True` and consume `output.spike_gate`. Hard/eval connections
consume `output.spikes.to(projection.source_dtype)`, which encodes boolean
events and does not encode amplitudes.
Tracked NetCon training events require native hard delay, spiking, and
scheduled-time configuration; surrogate configurations retain the native
tracking rejection.

Tensor schedule evaluation is stateless: repeating a time/step reevaluates the
same event; skipping a hard event's grid step misses it. No consumed-event
cursor is added. Heap events consume overdue entries and coalesce them into
one source event per call. Preserve these distinctions when implementing an
outer loop. For Network parity, generate times as
`origin + origin.new_tensor(integer_step) * dt`, as the example does. This
preserves the clock dtype and multiplication order; repeated floating-point
addition or multiplication first performed as Python floats can move an exact
causal boundary by an ULP.

## Checkpoint values and carry together

Keep schedule packets alongside clocks, RNG state, queues, population state,
base weights, and simulation time:

```{literalinclude} ../../examples/functional_tensor_schedules.py
:language: python
:start-after: "# region checkpoint"
:end-before: "# endregion checkpoint"
:dedent: 12
```

The shown checkpoint retains live parameter references during a fixed-parameter
forward pass. For a durable snapshot across optimizer updates, serialize it or
clone/detach parameter values as well as carry. Restore intentional source
sharing when rebuilding independently cloned packets.

For segmented training, the same indexed source leaves can remain owned by an
optimizer. After backward and the optimizer update, use `network.detach()` for
native model history, or `dn.func.detach_continuation(functional, packet)` for a
functional continuation. Indexed schedule values are gathered from the current
sources in the next segment:

```{literalinclude} ../../examples/shared_schedule_bindings.py
:language: python
:start-after: "# region training"
:end-before: "# endregion training"
:dedent: 12
```

Portable functional restoration creates owned source leaves and preserves
their aliases; create the receiving optimizer from those restored leaves.
Native source/runtime restoration uses `receiver.load(...)` followed by
`receiver.restore_dict_from_checkpoint(...)` and `receiver.detach()` before
resumed training. The complete shared-binding example checks both paths:

```console
python examples/shared_schedule_bindings.py
```

Schedule packets are configuration/parameter inputs rather than changing
carry. NetCon's native runtime checkpoint excludes schedule configuration;
NetStim's includes its tensor schedule storage. Functional `commit_state_`
transfers runtime carry and preserves the destination's schedule values and
source bindings. Set matching native values before native continuation, and
align every projection, population, and owner clock at a complete timestep
boundary as described in {doc}`A14_functional_netstim_renewal`.

Changing the event topology, bindings, timestep, or scheduling policy requires
lowering again. Ordinary state transfer into a compatible schedule preserves
plan validity. Extraction and commit audit routing, reference indices, source
schemas, and kernel configuration; explicit numerical inputs may vary between
steps. Sampling/extraction/commit stay outside transformed graphs. Pure
evaluation supports `torch.compile` and `torch.func`; use population
`prepare_and_receive` and `prepare_and_step` inside a compiled composition.

The complete example executes here:

```{code-cell} python
:load: ../../examples/functional_tensor_schedules.py
:tags: [remove-input]
```

Functional schedules use fixed tensor topology. Event projections support dense
queues, packed scheduled traffic, and sparse-calendar inference; see
{doc}`A24_functional_pending_calendar` for compact calendar capacity and overflow.
Differentiable delays are described in
{doc}`A17_functional_delay_interpolation`. Heap calendars are described in
{doc}`A16_functional_heap_schedules`.
Dynamic insertion inside transformed loops is currently unsupported. Canonical single-device
Network lowering is described in {doc}`A19_functional_network_lowering`, and
snapshot-based `dn.compile(Network)` execution in
{doc}`A21_functional_network_compile`.
