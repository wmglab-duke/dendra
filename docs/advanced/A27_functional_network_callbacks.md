---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-network-callbacks)=
# Observe individual Populations in a Network

Use `dn.func.PopulationCallback(name, callback)` to apply a functional Population
callback to one Population in a functional Network. The name is a top-level
Population name in the Network passed to `dn.compile` or
`make_functional_network`:

```python
callbacks = {
    "counts": dn.func.PopulationCallback(
        "left", dn.func.APCount(threshold=-20.0, node_check=[0, -1])
    ),
    "anomaly": dn.func.PopulationCallback("right", dn.func.AnomalyDetector()),
}
```

The outer mapping keys name result streams. `"left"` and `"right"` identify
the observed Populations. The wrapped callback receives that Population's state
and step auxiliary packet, and uses the same observation, carry, and finalization
rules as it does with `dn.compile(Population)`.

The complete
{download}`functional_network_callbacks.py <../../examples/functional_network_callbacks.py>`
example compares scan and while execution for two connected, batched Populations.
It checks LFP projections, all threshold-analysis callbacks, anomaly detection,
regional and cadenced recording, a custom reducer/emitter, continuation, and scan gradients
through lead fields. Run
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
`PopulationCallback` is rejected. This wrapper belongs to Network callback
collections; use the underlying callback directly for a standalone Population.
Configure and bind a new collection to change a scope or selection.

Binding a Recorder, LFP or threshold callback prepares a separate immutable
instance of its public callback class. A Recorder retains resolved state paths,
physical regions and index or partition selections; an LFP retains its field
references and projection metadata. The supplied configuration objects remain
unchanged. These public classes implement the pure callback hooks; evolving values and recording phase
belong to explicit carry. Prepared Network Recorder paths and regions keep
their resolved meaning when used by the collection.

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

LFP preparation retains the original lead-field Tensor objects and builds no
autograd graph. Each fresh callback initialization constructs its own
differentiable field stack, so a later fresh run supports an independent
backward pass. Continuation retains the initialized stack in explicit carry.

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
`int(t_start_check / dt) <= index < int(t_end_check / dt)`; omitting
`t_end_check` leaves it open. These bounds follow the callback's phase rather
than the Network's absolute clock. Starting a new collection at a nonzero
simulation time starts callback index zero at its next step. Continuing with
callback state preserves that index, the crossing latches, and accumulated
results.

For `K` checked slots in a selected Population with shape `(*batch, N, C)`:

| Callback | Result | Storage as the horizon grows |
| --- | --- | --- |
| `APCount` | `(*batch, N, K)` integer counts | Constant callback carry |
| `Active` | `(*batch, N)` Boolean: any checked site fired | Constant callback carry |
| `ActiveAL(at_least=k)` | `(*batch, N)` Boolean: at least `k` selected slots fired | Constant callback carry |
| `ActiveALCount(at_least=k)` | `(*batch, N)` Boolean: at least `k` total events | Constant callback carry |
| `AnomalyDetector` | `(*batch, N)` Boolean: any nonfinite post-step voltage | Constant callback carry |
| `Raster` | `(steps, *batch, N, K)` Boolean events | One frame per post-step sample |
| `LFP` | `(steps + 1, *batch, contacts)` on a fresh run | One frame per recorded boundary |
| `Recorder` | Selected state trajectory | One frame per recorded boundary |

`ActiveAL` counts selected slots with at least one event; repeated spikes within
one slot contribute separately only to `ActiveALCount`. The Active family also supports `inv=True`
and static `partition` groups over the selected-slot axis. Repeated `node_check`
indices retain their multiplicity. These settings follow the full
{doc}`A8_functional_populations` callback contract.

Native and functional threshold analyses share their numerical crossing,
counting and activity reductions. Each execution API retains its observation
window, counter and recording policies; the functional shapes and timing are
specified here.

`Raster` retains every accepted timestep, with `False` outside its checked
window. Reducers and anomaly detection emit no trajectory, so their callback
memory does not grow with the step count. This describes callback storage;
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
the shared frame. Raster emits post-step samples only. Reducer results returned
after continuation are cumulative; use the suffix result rather than adding
the prefix and suffix counts. `runner()` without state starts a fresh simulation
and callback history. Passing simulation state without callback state starts a
fresh callback history at that boundary.

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
    "exc_voltage": dn.func.PopulationCallback(
        "all_populations", dn.func.Recorder(("v",), region=packed.exc)
    ),
}
```

The packed physical layout flattens each component's cell and compartment axes.
A scoped threshold callback uses `node_check` positions in that packed
Population's final compartment axis. To select a named component, derive those
positions from its physical Slice. For the single-cell-axis packed layout:

```python
nodes = packed.exc.flat_index.remainder(packed.shape[-1]).unique()
callbacks["exc_counts"] = dn.func.PopulationCallback(
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

The independent `coarse` Recorder uses `dt=2 * DT`. With native in-memory
timing it selects frames 0, 1, 3, 5, and 7 from the eight-step full recording.
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
