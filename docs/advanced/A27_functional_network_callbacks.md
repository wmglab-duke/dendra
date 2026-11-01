---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-network-callbacks)=
# Observe individual Populations in a Network

Use `dn.callbacks.PopulationCallback(name, callback)` to observe one Population
in a Network. The same scope configuration works with native Network execution,
`dn.compile`, and functional callback collections. The name is an exact top-level
key in the Network's `populations` mapping:

```python
callbacks = {
    "counts": dn.callbacks.PopulationCallback(
        "left", dn.callbacks.APCount(threshold=-20.0, node_check=[0, -1])
    ),
    "anomaly": dn.callbacks.PopulationCallback("right", dn.callbacks.AnomalyDetector()),
}
```

For native execution, pass the wrappers in a callback list. Retrieve results
from the supplied inner callbacks, also available through `scope.callback`.
For functional execution, pass the named mapping shown above; its keys name
result streams under `auxiliary["callbacks"]`. Binding captures the supported
inner configuration without importing or changing its native history. The
prepared callback receives the selected Population's state and step auxiliary
packet, using the same observation, carry, and finalization rules as it does
with `dn.compile(Population)`.

`dn.func.PopulationCallback(name, callback)` remains a compatible functional
entry point. Both spellings bind to the same scope and portable callback
checkpoint configuration.

(shared-population-callback-scopes)=
## Share scopes between native and functional execution

The complete
{download}`shared_population_callback_scopes.py <../../examples/shared_population_callback_scopes.py>`
example compares observations from two Populations, Recorder cadence, scan
gradients and continuation. It also restores a canonical scope's portable
checkpoint into the compatible `dn.func.PopulationCallback` configuration:

```bash
python examples/shared_population_callback_scopes.py
python examples/shared_population_callback_scopes.py --backend aot_eager
```

```{literalinclude} ../../examples/shared_population_callback_scopes.py
:language: python
:start-after: "# region scopes"
:end-before: "# endregion scopes"
:dedent: 4
```

The Recorder selections are relative to their observed Population. Native
execution stores results in each inner callback:

```{literalinclude} ../../examples/shared_population_callback_scopes.py
:language: python
:start-after: "# region native"
:end-before: "# endregion native"
:dedent: 4
```

Native `Network.step`, `run`, `longrun` and `longrun_checkpointed` forward the
inner native callback's six loop, chunk and step hooks to the selected
Population, preserving hook return values. `step` retains its `loop_hooks`
option; scope selection does not initialize or reset callback history itself.
Native subclasses of `Callback` can use this forwarding. Checkpointed native
execution retains its replay-safe callback contract: hooks must be pure
functions of model state, and scalar loss contributions from supported hooks
are accumulated by the existing executor.

The canonical wrapper can also hold a pure `FunctionalCallback` configuration.
Bind it through `dn.func.make_callbacks(target, configurations)` for functional execution or the
native Network pure-callback bridge described below. Hook forwarding through
the ordinary native `callbacks=` list requires a native `Callback` child and
rejects a pure-only child. Functional binding accepts supported built-in native
configurations and pure callbacks; wrapping an arbitrary native callback does
not make its imperative hooks functionally lowerable.

(functional-native-network-callbacks)=
## Use pure callbacks with native Network execution

Native `Network.step`, `run`, `longrun` and `longrun_checkpointed` accept the same
bound callback collection through `callbacks=` and explicit carry through
`callback_state=`. Build it from the
initialized source Network. Its timestep comes from
`network.initialize(dt)`; native Network execution uses that timestep without
a per-call `dt` argument:

```python
observations = dn.func.make_callbacks(network, {
    "trace": dn.Recorder(("left.v", "right.v", "t")),
    "counts": dn.callbacks.PopulationCallback(
        "right", dn.callbacks.APCount(node_check=[0, -1])
    ),
})
first = network.run(10.0, callbacks=observations)
second = network.longrun(
    10.0, chunklength=100, callbacks=observations,
    callback_state=first.state,
)
```

The factory uses the initialized Network timestep; an optional `dt=` must match
it. It also accepts an existing `FunctionalNetwork`, preserving its exact plan
and avoiding another lowering. `functional.make_callbacks(...)` remains
supported. Native-model preparation retains full lowering's eligibility and
setup cost and does not initialize or advance the Network. Reuse the returned
collection rather than preparing it again for each call. A newly prepared
collection has its own carry identity; use portable export/restore to transfer
carry between equivalent collections. See
{ref}`functional-shared-callback-preparation` and
{download}`shared_callback_preparation.py <../../examples/shared_callback_preparation.py>`.

A prepared collection also observes a compatible initialized native Network at
the same timestep, using that receiver's runtime state. Top-level names, model
structure, state layout, dtype/device, execution mode and source sharing must
match its prepared requirements. Qualified paths, scopes and resolved selection
indices remain fixed. Physical Slice and region configurations require the exact
source Network at preparation; reuse applies those indices to the receiving
Network. Functional execution still requires the exact owning plan, and carry
still belongs to its original collection.

For functional-to-native continuation, commit the packet with
`dn.func.commit_continuation_(program, receiver, packet)`, then pass
`callbacks=program.callbacks` and
`callback_state=program.callbacks.detach_state(packet.callback_state)` to native
execution. This retains totals and Recorder phase while truncating earlier
recurrent gradients. Custom callbacks need no portable checkpoint hooks for
same-collection reuse. See {doc}`A59_functional_native_handoff` and
{download}`shared_callback_rebinding.py <../../examples/shared_callback_rebinding.py>`.

The Network owns its advancing model and connection state. Callback carry is
separate: pass `results.state` to continue observation phase and reducer totals
without repeating the shared recording boundary. `step`, `run` and `longrun` return
the existing `FunctionalCallbackResults` mapping when opted in; their ordinary
return is unchanged when functional callbacks are omitted: `step` returns the
Network for chaining, while `run` and `longrun` return `None`.
`longrun_checkpointed` returns `(loss_or_none, results)`, or
`(loss_or_none, final_state, results)` with `return_final_state=True`.
Without a pure collection, it retains its ordinary native loss/state return.
Qualified Network Recorders and Population-scoped callbacks use the same selections,
shapes, cadence and pure hooks as functional Network execution.

Unscoped custom callbacks receive the complete Network state and auxiliary
`{"populations": {name: {"v": ...}}}`. A scoped callback receives the selected
Population state and auxiliary `{"v": ...}`. Pure updates observe accepted
state after Network clock advancement and before ordinary native post-step
hooks. Native tensor frames are snapshotted without detaching their gradients.
Ordinary native `callbacks=[hook, ...]` lists retain their existing hook
behavior. To combine them with pure observations, supply the prepared collection
through `functional_callbacks=observations` and its carry through
`callback_state=`. The `functional_callback_state=` spelling also remains
supported. Two non-`None` pure collection arguments or two non-`None` carry
arguments are rejected. Native calls accept prepared collections rather than
preparing configuration mappings on each invocation.

Fresh Recorder results include the current initial boundary; continued calls
emit only new accepted samples. Zero-step calls retain the native loop-hook
boundaries and the same fresh/resumed observation rules. Callback carry does
not initialize or restore Network simulation state. The bridge requires a
Network admitted by functional lowering. Recorders configured with `set_hdf5`
stream detached samples through the host collector; their names remain in
callback carry and are omitted from returned trajectory results. In-memory
Recorders and reducers retain gradients. Native hook-list recording retains its
independent behavior.

Use a qualified Network Recorder or a scoped local Recorder for HDF5, with the
same sampling phase across `step`, `run`, `longrun` and `longrun_checkpointed`.
Each call with samples creates a `run_N` file segment; continued calls omit the
shared boundary. `cache_every` counts solver iterations plus a fresh initial
boundary, independently of `Recorder(dt=...)` sampling cadence. Storage and
writer cleanup run on the host, outside backward replay, and do not depend on
native `loop_hooks`. Call `recorder.close()` before reading the file. See
{ref}`functional-native-callback-storage` and
{download}`native_callback_storage.py <../../examples/native_callback_storage.py>`
for a complete scoped example with file readback and gradients.

Manual `Network.step` calls use the same collection and carry, including
Population scopes and qualified recording. `loop_hooks` governs only native
loop hooks; pure initialization runs when callback carry is absent. A continued
step emits only its new selected samples and retains reducer totals. The
`dn.step` and `dn.step_network` helpers expose the same callback arguments.
See {ref}`functional-native-step-callbacks` and the complete
{download}`native_step_functional_callbacks.py <../../examples/native_step_functional_callbacks.py>`
example for manual stepping, mixed-method continuation and portable carry.

### Checkpointed native observations

Continue the same collection through checkpointed execution using
`callbacks=observations, callback_state=previous.state`. Callback carry is
threaded through each checkpointed chunk; recording collection happens outside
replay. Native hook lists may contribute an independent scalar loss and retain
their replay-safe behavior. If no native hook contributes a loss,
`loss_or_none` is `None`; construct an objective from the pure callback results.

```{literalinclude} ../../examples/native_network_functional_callbacks.py
:language: python
:start-after: "# region checkpoint"
:end-before: "# endregion checkpoint"
:dedent: 8
```

`final_state` is the existing native Network checkpoint dictionary. It is
separate from callback carry in `results.state`. With
`restore_state_after_backward=True`, checkpoint recomputation restores the live
Network around replay, including gradients taken only through pure recordings
or reducer results. `safe_checkpoint` keeps its native input-cloning policy.
The callback return and continuation conventions match the Population bridge;
native Network checkpoints retain their existing format.

The complete
{download}`native_network_functional_callbacks.py <../../examples/native_network_functional_callbacks.py>`
example connects unequal Populations and continues native `run`, checkpointed
longrun and ordinary longrun against an independent functional Network. It
checks a scoped custom summary, qualified and relative recording, cadence,
parameter gradients and live voltage/clock preservation after backward:

```console
python examples/native_network_functional_callbacks.py
```

```{literalinclude} ../../examples/native_network_functional_callbacks.py
:language: python
:start-after: "# region callbacks"
:end-before: "# endregion callbacks"
:dedent: 4
```

The complete
{download}`functional_network_callbacks.py <../../examples/functional_network_callbacks.py>`
example compares scan and while execution for two connected, batched Populations.
It checks LFP projections, all threshold-analysis callbacks, anomaly detection,
regional and cadenced recording, a custom reducer/emitter, continuation, and
scan gradients through lead fields. Run
it from the repository root with `python examples/functional_network_callbacks.py`.

## Declare the Network and select callbacks

Build and initialize the Network with the ordinary API. Enable membrane-current
output when using LFP:

```{literalinclude} ../../examples/functional_network_callbacks.py
:language: python
:start-after: "# region network"
:end-before: "# endregion network"
:dedent: 4
```

Wrap `LFP`, `Raster`, `APCount`, `Active`, `ActiveAL`, `ActiveALCount`, and
`AnomalyDetector` in `PopulationCallback`. These callbacks require an explicit
Population scope when used in a Network:

```{literalinclude} ../../examples/functional_network_callbacks.py
:language: python
:start-after: "# region callbacks"
:end-before: "# endregion callbacks"
:dedent: 4
```

A scoped Recorder uses local selections such as `"v"`, `"i_membrane"`, or
`"t"`. Its result keeps those local names. A bare Network Recorder instead uses
qualified selections such as `"left.v"`; its `"t"` selection is the Network
clock. Both forms can record a physical region from the exact owning Population.
See {doc}`A26_functional_network_regions` for labels, repeated indices, explicit
batch selection, and supported fields.

Population scope is fixed when callbacks bind. An unknown name or a nested
`PopulationCallback` is rejected, including nesting across the canonical and
compatible functional spellings. The wrapper belongs to Network callback
collections; use the underlying callback directly for a standalone Population.
Configure and bind a new collection to change a scope or selection.

Use `dn.Recorder` to configure recording in native and functional execution;
`dn.func.Recorder` remains supported. Binding captures Recorder configuration
into a separate immutable pure implementation. LFP and threshold callbacks
likewise prepare independent immutable implementations. A Recorder retains
resolved state paths,
physical regions and index or partition selections; an LFP retains its field
references and projection metadata. The supplied configuration objects remain
unchanged. The prepared implementations expose the pure callback hooks; evolving
values and recording phase
belong to explicit carry. Prepared Network Recorder paths and regions keep
their resolved meaning when used by the collection.

`dn.callbacks.LFP`, `APCount`, `Raster`, `Active`, `ActiveAL`, `ActiveALCount`
and `AnomalyDetector` configurations can also be supplied to functional binding
and `dn.compile`; their `dn.func` counterparts remain supported. Wrap them in
`PopulationCallback` for a Network. Threshold configurations use authored
selectors and timing settings, rather than importing native
counts, crossing latches, normalized indices or run counters. Negative selectors
resolve against the selected Population. An omitted `dt` uses its functional
timestep; an explicit value must match. Bind a new collection after editing
configuration. Native execution retains ownership of its callback histories.
Native activity configurations select their configured unpartitioned query.
Native LFP configuration supplies lead-field sources and projection settings
without importing or updating its recording history.

Record mechanism state with qualified names such as `"left.hh.m"`, or relative
names such as `"hh.m"` inside a Population-scoped Recorder. The result keeps the
supplied name and the mechanism's local tensor shape. Physical `region=`
selection remains limited to `v`, `vc` and `i_membrane` (with scalar `t` allowed
alongside them); mechanism slots can have sparse support.

HDF5-configured `dn.Recorder` instances use the host runner's storage path. Disk
trajectories are omitted from returned callback outputs while their sampling
carry remains in the continuation. The same selected names, shapes and cadence
are written through the existing HDF5 format and reader. See
{ref}`functional-recorder-storage` for bounded collection, gradients, file
segments and portable continuation.

## LFP shape, units, and gradients

LFP reads `i_membrane`, the total membrane current in mA at each physical
coordinate. Supply a nonempty list of equally shaped reciprocal lead-field
Tensors, one per recording contact. A field in mV/mA, numerically equivalent to
ohms, produces a potential in mV. Each field must broadcast to the selected
Population's voltage shape unless an explicit `rule` supplies a custom
`torch.einsum` reduction.

For a Population with voltage/current shape `(*batch, N, C)`, the default
projection sums the cell and compartment axes. Its result has shape
`(frames, *batch, contacts)`. Batch lanes remain separate. In the example, the
current has shape `(2, 2, 3)`, each lead field has shape `(2, 3)`, and two
contacts produce `(frames, 2, 2)`.

This default applies to native and functional LFP. Supplying `rule=...` uses
`torch.einsum(rule, current, stacked_fields)` exactly, with the contact axis
first in the stacked fields. In particular, explicitly passing
`rule="...,n...->n"` sums batch axes; omitting `rule` preserves them. Fields
are prepared in the observed Population's dtype and device without detaching
gradients.

```{literalinclude} ../../examples/functional_network_callbacks.py
:language: python
:start-after: "# region gradients"
:end-before: "# endregion gradients"
:dedent: 4
```

Scan retains gradients through lead fields, Population dynamics, and explicit
continuation. Lead fields remain invariant Tensor operands during compiled
execution. Each accepted step emits only its electrode projection; a full
membrane-current history is stored only if a Recorder also selects it. The
example records that history to verify the projection and its lead-field
gradient. Omit that Recorder when only electrode signals are needed.

LFP binding retains lead-field Tensor sources and projection metadata.
Each fresh callback initialization constructs its own
differentiable field stack, so a later fresh run supports an independent
backward pass. Continuation retains the initialized stack in explicit carry.
Authored field aliases remain shared. Changes to a native configuration's
public `v_unit` buffer made before binding are respected. Bind a new collection
after changing its buffer selection or projection rule.

(functional-shared-lfp-configuration)=
## Share LFP configuration

Use `dn.callbacks.LFP(fields)` for native runs and functional simulations;
`dn.func.LFP(fields)` remains supported. For a Network, wrap it in
`PopulationCallback(name, callback)` to select the observed Population.
Construct that Population with `IMEM=True` and an integrator that exposes
membrane-current output.

The complete
{download}`functional_lfp_configuration.py <../../examples/functional_lfp_configuration.py>`
example checks the default projection for a batched Population and Network,
independent fresh backward passes through trainable fields, live continuation,
and portable checkpoint restoration:

```bash
python examples/functional_lfp_configuration.py
python examples/functional_lfp_configuration.py --backend aot_eager
python examples/functional_lfp_configuration.py --section network --execution while
```

```{literalinclude} ../../examples/functional_lfp_configuration.py
:language: python
:start-after: "# region configuration"
:end-before: "# endregion configuration"
:dedent: 4
```

Fresh functional output includes the initial boundary. A continued call emits
only new accepted-step samples, so concatenating the segments reproduces an
uninterrupted result. Live continuation preserves the initialized field graph.
Portable continuation restores detached field carry; it does not connect to
the receiving configuration's field gradients. Native recording history is
separate from this functional carry.

(functional-shared-threshold-configuration)=
## Share spike-analysis configuration

`dn.callbacks.APCount`, `Raster` and `AnomalyDetector` configurations can be
used by native runs and functional simulations. The complete
{download}`functional_threshold_configuration.py <../../examples/functional_threshold_configuration.py>`
example covers a Population and a Network, scan gradients, while inference,
configuration ownership and portable continuation:

```bash
python examples/functional_threshold_configuration.py
python examples/functional_threshold_configuration.py --backend aot_eager
python examples/functional_threshold_configuration.py --section network --execution while
```

Its configuration selects the first and last compartments and an explicit
finite observation window. Network callbacks use the same local configuration
inside `PopulationCallback`:

```{literalinclude} ../../examples/functional_threshold_configuration.py
:language: python
:start-after: "# region configuration"
:end-before: "# endregion configuration"
:dedent: 4
```

(functional-shared-activity-configuration)=
## Share activity configuration

Use `dn.callbacks.Active`, `ActiveAL` and `ActiveALCount` to configure native
or functional activity detection. `Active` tests whether any selected slot has
fired. `ActiveAL(at_least=k)` tests how many selected slots fired at least
once; repeated events at one slot count once. `ActiveALCount(at_least=k)`
tests the total events across those slots, including repeated events. Each
returns a cumulative Boolean result, inverted when `inv=True`.

Functional binding captures the native configuration's unpartitioned
`is_active()` query. Native and `dn.func` activity constructors require a
positive integer `at_least`. Booleans, floats, Tensors and strings are rejected
with `TypeError`; zero and negative integers raise `ValueError`. Other integral
scalar types, including NumPy integers, are accepted. Binding and native
queries apply the same validation. The target is static configuration and has
no gradient. Bind a new collection after changing the query. Functional execution
does not import or update the native object's counts, crossing cache or timer.
Continue with explicit callback carry rather than the native history.

Native `is_active(partition=...)` and `numpy(partition=...)` remain post-run
queries over native counts. Native constructors do not accept `partition`.
For static grouped functional results, use `dn.func.ActiveAL` or
`dn.func.ActiveALCount` with `partition=...`. Each partition length must be at
least `at_least`.

The complete
{download}`functional_activity_configuration.py <../../examples/functional_activity_configuration.py>`
example checks activity against per-slot APCount results, inversion, static
groups, recording gradients, explicit continuation and portable checkpoints
for a Population and scoped Network:

```bash
python examples/functional_activity_configuration.py
python examples/functional_activity_configuration.py --backend aot_eager
python examples/functional_activity_configuration.py --section network --execution while
```

```{literalinclude} ../../examples/functional_activity_configuration.py
:language: python
:start-after: "# region configuration"
:end-before: "# endregion configuration"
:dedent: 4
```

## Threshold windows and result memory

Threshold preparation resolves compartment indices, timestep and window bounds.
`Active` uses the `ActiveAL(at_least=1)` rule. Configure the supplied object before
binding, and bind a new collection when changing its settings. Crossing latches,
counts and step phase travel in explicit carry.

Threshold callbacks inspect accepted post-step voltages. The initial model frame
does not trigger a crossing. The first checked sample is armed, including when
its voltage is already at or above `threshold`; a checked below-threshold sample
rearms that site. This is a hard observation rule and does not supply gradients
of threshold decisions.

The first post-step sample has callback index zero. A window checks
`int(t_start_check / dt) <= index < int(t_end_check / dt)`. Native threshold
configurations use their `1e9` ms default end when it is omitted;
`dn.func` constructors leave an omitted end open. An explicit finite end gives
the same window with either constructor. These bounds follow the callback's
phase rather than the Network's absolute clock. Starting a new collection at a
nonzero simulation time starts callback index zero at its next step. Continuing with
callback state preserves that index, the crossing latches, and accumulated
results.

For `K` checked slots in a selected Population with shape `(*batch, N, C)`:

| Callback | Result | Storage as the horizon grows |
| --- | --- | --- |
| `APCount` | `(*batch, N, K)` cumulative `torch.int32` counts | Constant callback carry |
| `Active` | `(*batch, N)` Boolean: any checked site fired | Constant callback carry |
| `ActiveAL(at_least=k)` | `(*batch, N)` Boolean: at least `k` selected slots fired | Constant callback carry |
| `ActiveALCount(at_least=k)` | `(*batch, N)` Boolean: at least `k` total events | Constant callback carry |
| `AnomalyDetector` | `(*batch, N)` Boolean: any nonfinite post-step voltage | Constant callback carry |
| `Raster` | `(samples, *batch, N, K)` Boolean events | One frame per checked post-step sample |
| `LFP` | `(steps + 1, *batch, contacts)` on a fresh run | One frame per recorded boundary |
| `Recorder` | Selected state trajectory | One frame per recorded boundary |

`ActiveAL` counts selected slots with at least one event; repeated spikes within
one slot contribute separately only to `ActiveALCount`. Both constructor families
support `inv=True`. The `dn.func` constructors additionally support static
`partition` groups over the selected-slot axis. Repeated `node_check`
indices retain their multiplicity. These settings follow the full
{doc}`A8_functional_populations` callback contract.

Native and functional threshold analyses share their numerical crossing,
counting and activity reductions. Functional execution keeps crossing latches
and phase in explicit carry; native execution owns its mutable history.
Configuration binding captures only the supplied static settings. Both APCount
entry points use exact signed 32-bit integer counts, with four bytes per checked
slot. Threshold decisions and count updates are nondifferentiable.

Both native and functional `Raster` return only checked post-step samples.
`False` denotes a checked sample without a crossing. Calls outside the window
return a zero-length sample axis, and the initial model frame contributes no
Raster row. Crossing counts and Boolean events are hard observations without
gradients of their threshold decisions.

The scan and while bodies emit fixed-size tensors. Host collection selects
Raster's checked window before retaining its history. Capturing atomic
finalization itself with `torch.compile` requires dynamic output shape support;
host `dn.func.run` keeps that selection outside compiled numerical kernels.
Reducers and anomaly detection emit no trajectory, so their callback memory
does not grow with the step count. This describes callback storage;
training may retain transition activations independently. Checkpointing reduces
saved activations, rather than the returned recording arrays.

## Compile and continue

```{literalinclude} ../../examples/functional_network_callbacks.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 4
```

Scan supports differentiable recordings and reducers. While execution is
inference-only and uses the same pure callback API, including custom callbacks,
scoped built-ins, and qualified Recorders. Unsupported explicit while
configurations raise; the runner does
not silently select another executor. See {doc}`A25_functional_network_while`
for execution reports and bounded tails.

Continue with both returned simulation state and callback state:

```{literalinclude} ../../examples/functional_network_callbacks.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 4
```

Fresh LFP and Recorder results include the initial boundary. A continued call
emits only its new boundaries, so concatenating those results does not duplicate
the shared frame. Raster emits checked post-step samples only. Reducer results returned
after continuation are cumulative; use the suffix result rather than adding
the prefix and suffix counts. `runner()` without state starts a fresh simulation
and callback history. Passing simulation state without callback state starts a
fresh callback history at that boundary.

The example's Raster window covers callback indices 2 through 5. Its returned
four rows correspond to those checked samples; samples before or after that
window are absent. The first checked row contains the armed crossing.

The example verifies the output shapes and reduction relationships:

```{literalinclude} ../../examples/functional_network_callbacks.py
:language: python
:start-after: "# region results"
:end-before: "# endregion results"
:dedent: 4
```

## Concatenated Populations

Scope names refer to top-level Populations after transformations. With the
default `network.concat()` grouping, use `"all_populations"`; a component name
such as `"exc"` is a named Slice inside that Population, rather than a separate
callback scope:

```python
network = network.concat()
packed = network.all_populations
callbacks = {
    "exc_voltage": dn.callbacks.PopulationCallback(
        "all_populations", dn.Recorder(("v",), region=packed.exc)
    ),
}
```

The packed physical layout flattens each component's cell and compartment axes.
A scoped threshold callback uses `node_check` positions in that packed
Population's final compartment axis. To select a named component, derive those
positions from its physical Slice. For the single-cell-axis packed layout:

```python
nodes = packed.exc.flat_index.remainder(packed.shape[-1]).unique()
callbacks["exc_counts"] = dn.callbacks.PopulationCallback(
    "all_populations", dn.func.APCount(node_check=nodes, threshold=-20.0)
)
```

These results retain the packed Population's axes; they do not reconstruct each
component's original cell layout. To restrict an LFP to one component, supply
full packed lead fields that are zero outside that component:

```python
mask = torch.zeros_like(packed.v)
mask[packed.exc.index] = 1
callbacks["exc_lfp"] = dn.func.PopulationCallback(
    "all_populations", dn.func.LFP([packed_lead_field * mask])
)
```

## Custom callbacks across scan and while

Implement `FunctionalCallback` once for both engines. The example accumulates a
post-step voltage statistic in scalar carry and emits a scalar mean voltage at
each boundary:

```{literalinclude} ../../examples/functional_network_callbacks.py
:language: python
:start-after: "# region custom callback"
:end-before: "# endregion custom callback"
```

`PopulationCallback("left", VoltageSummary())` supplies only that Population's
state and auxiliary mapping. A bare custom callback receives the complete
Network packet. The `summary` stream is mixed with the built-ins in the example;
scan and inference-only while use the same object and result structure.

Carry and individual emissions preserve their tensor tree structure, shape,
dtype, and device. Emissions can be `None` for a constant-memory reducer; an
emitter returns a nonempty tensor tree. Use fixed tensors and masks for sparse
observations instead of changing the output structure. Keep callback objects
immutable and implementations deterministic, side-effect-free, and compatible
with graph capture. Mutable native callbacks and their imperative hooks are not
executed by a functional callback collection.

The example compares the custom mean-voltage trajectory with the Recorder,
including its initial frame, and the accumulated energy with all post-step
voltages. Continuation keeps the carry cumulative and does not repeat the shared
boundary emission. The same bound collection supports in-memory continuation
through `callback_state=` or a returned `Continuation`. Custom bound callbacks
can support portable export and restoration by implementing
`checkpoint_configuration()`; see
{ref}`functional-callback-checkpoint-protocol`.

The independent `coarse` Recorder uses `dt=2 * DT`. It selects frames
0, 2, 4, 6, and 8 from the eight-step full recording, using the same cadence
for in-memory and HDF5 storage.
Joining the two execution segments gives the same selected frames as one
uninterrupted call. This cadence does not change the custom callback or the
other Recorders. The example uses bounded compiled chunks; the host retains
the selected `coarse` frames between those chunks. The dense `trace` stream
remains in the example for comparison and retains its full requested history.
Each loop invocation still emits dense intermediate data; see
{ref}`functional-recorder-cadence` for the storage contract.

The complete CPU example executes here:

```{code-cell} python
:load: ../../examples/functional_network_callbacks.py
:tags: [remove-input]
```
