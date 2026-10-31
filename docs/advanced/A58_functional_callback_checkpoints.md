---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-callback-checkpoints)=
# Save and restore functional continuation

Functional simulation state contains clocks, delay histories, event queues and
RNG carry. A complete continuation also includes the effective parameter and
constant sources and callback state for reductions, threshold crossing latches
and sampling phase. Lowered Populations and Networks use the same portable
bridge:

```python
packet = dn.func.export_continuation(
    functional, auxiliary["continuation"], callbacks=callbacks,
)
torch.save(packet, path)

restored = dn.func.restore_continuation(
    receiving, torch.load(path, weights_only=True), callbacks=receiving_callbacks,
)
state, auxiliary = dn.func.run(
    receiving_kernel, restored, steps=100, callbacks=receiving_callbacks,
)
```

Here `functional` and `receiving` are plans returned by
`dn.func.make_functional`, and each callback collection belongs to its respective
plan. The same functions accept compiled Population and Network runners and select
their configured callbacks; omit `callbacks=` for runner targets:

```python
packet = dn.func.export_continuation(program, auxiliary["continuation"])
torch.save(packet, path)

restored = dn.func.restore_continuation(
    receiver, torch.load(path, weights_only=True),
)
state, auxiliary = receiver(restored)
```

`program.export_continuation(auxiliary["continuation"])` and
`receiver.restore_continuation(packet)`
provide equivalent method forms.

Construct and initialize the receiving model, then lower or compile it
explicitly. The file contains versioned data and Tensors, rather than executable
model configuration.
The restored `dn.func.Continuation` supplies state, parameters, constants and
callback carry to the resume call. In the runner example, both `program` and
`receiver` are constructed with `dn.compile`. The receiving plan or runner can
belong to another Python process. The original plan or runner can also restore
a checkpoint. The functions and runner methods share one packet format, so
either form can restore a compatible packet exported by the other.

Use `runner.callbacks` to access the runner's bound callback collection. It is a
read-only property and returns `None` when recording and reducers are omitted.
For an explicitly lowered Population or Network, use the collection returned by
`functional.make_callbacks(...)` instead. Both expose the same `export_state`
and `restore_state` methods.

For continuation with live graph history, pass `auxiliary["continuation"]`
directly to the next call. Portable export is a separate operation with an
explicit autograd boundary. Save the result of `export_continuation`, rather
than pickling the in-memory `Continuation` object.

For separate training segments, use
`dn.func.detach_continuation(program, auxiliary["continuation"])` after backward
and the optimizer update. This keeps the existing parameter/constant Tensor
objects, optimizer ownership and source graphs, while truncating recurrent
model-state and custom callback-carry gradients. Numerical history, RNG,
callback totals and sampling phase continue. Built-in LFP carry is refreshed
from current bound field sources so subsequent segments retain field gradients
when ambient gradients are enabled. Detached recurrent tensors share their
incoming storage; this operation keeps view strides rather than making a value
snapshot.
An existing functional plan is also a target; supply `callbacks=callbacks` when
carry is present. Runners select their configured collection. The operation
preserves lifecycle and callback metadata and performs no recording I/O.
See {ref}`functional-continuation-detach` for nonleaf-source responsibilities,
the distinction from portable restoration, and a complete optimizer-loop example.

`dn.func.initialize(program)` and `dn.func.initialize(functional, tensors)`
supply fresh initialized packets with no callback carry; see
{ref}`functional-runner-initialization` and
{ref}`functional-explicit-initialization`. Save and restore these packets against
a runner with configured callbacks, or supply a lowered plan's callback
collection explicitly. Neither operation starts recording. The first execution
starts those callbacks and records the initial boundary once. Restore resumes
the saved initialization without reapplying initialization transforms or
resampling its selected realization. The complete
{download}`functional_explicit_initialization.py <../../examples/functional_explicit_initialization.py>`
example checks initialization gradients before this boundary and trainable
effective sources after restoration.

Checkpoints from executed calls require matching callback configuration,
including zero-step calls. A checkpoint from a callback-free run cannot begin
recording when restored into a runner with callbacks. Manually constructed
continuations and version-1 checkpoints also require matching callback presence;
missing carry does not identify them as initialized packets.

The complete inference
{download}`functional_callback_checkpoints.py <../../examples/functional_callback_checkpoints.py>`
example writes actual temporary checkpoint files and reconstructs both a
Population runner and a connected Network runner:

```console
python examples/functional_callback_checkpoints.py
python examples/functional_callback_checkpoints.py --backend aot_eager
python examples/functional_callback_checkpoints.py --execution while
```

`--section population` and `--section network` select one case. The default eager
backend exercises compiler capture without native kernel generation. AOT eager
selects PyTorch's ahead-of-time graph backend. These short inference examples
verify continuation and do not measure performance.

The complete training
{download}`functional_continuation_checkpoints.py <../../examples/functional_continuation_checkpoints.py>`
example retains fresh initialization replacements, a per-call source override
and shared Network parameters. It checks disk restoration, joined recordings,
source trainability and suffix gradients after the detached boundary:

```console
python examples/functional_continuation_checkpoints.py
python examples/functional_continuation_checkpoints.py --backend aot_eager
```

The complete
{download}`functional_explicit_continuation_checkpoints.py <../../examples/functional_explicit_continuation_checkpoints.py>`
example uses only lowering, binding and host scheduling. It reconstructs both
receiving model kinds, restores from disk, binds the restored sources and checks
joined recordings, shared aliases and suffix gradients:

```console
python examples/functional_explicit_continuation_checkpoints.py
python examples/functional_explicit_continuation_checkpoints.py --backend aot_eager
```

## Configure the receiving analysis explicitly

A checkpoint contains data, rather than constructors for models or callbacks.
Construct and initialize an equivalent ordinary model, then lower it and bind
the intended callback configurations, or construct a `dn.compile` runner with
those configurations. Restore against that receiving plan or runner. Owner kind,
timestep, native mode, component layouts and the parameter/constant/state
schemas must be compatible. Saved parameter and constant values can differ
from the receiving model or runner's defaults. When callback carry is saved,
callback names and their bound configurations must match: this includes
observation paths and indices,
physical regions, Population scopes,
timestep, Recorder cadence, threshold windows, activity rules and LFP lead fields.

`dn.callbacks.PopulationCallback(name, callback)` and the compatible
`dn.func.PopulationCallback` entry point bind to the same functional scope
configuration. Portable carry can move between them when the Population name
and inner callback configuration match. Native callback histories remain
separate from exported functional carry. The
{ref}`shared scope example <shared-population-callback-scopes>` demonstrates
restoration across the two spellings without changing checkpoint formats.

For threshold callbacks, compatibility checks use prepared compartment indices,
window bounds, timestep and activity settings. Preparation is separate from the
authored configuration object and contains no changing analysis state. The
checkpoint carries crossing latches, accumulated counts where applicable, and
step phase with the per-callback carry schemas. Restoring this data
resumes the receiving callback's pure hooks without serializing a Python
callback instance.

Native `dn.callbacks.APCount`, `Raster`, `Active`, `ActiveAL`, `ActiveALCount`
and `AnomalyDetector` configurations
bind to independent pure implementations for functional execution. Exported
callback carry describes that functional execution's detection history; it does
not copy the supplied native object's `record`, crossing cache or run counter.
Construct and bind receiving configurations explicitly, then restore their
matching callback carry. Constructor namespace does not replace the resolved
configuration and schema checks. Raster configurations share the same
compact-window result and checkpoint contract when their resolved indices,
window and timestep match.

Native activity configurations capture their unpartitioned query, including
`at_least` and `inv`, while their native histories remain separate. The same
activity checkpoint contract applies to native and `dn.func` configurations
whose resolved queries match. Counts, crossing latches and step phase resume
in explicit carry; results remain cumulative Boolean masks. Native and `dn.func`
activity constructors require the same positive integer `at_least`, captured
as static configuration. Accepted integral scalar types are normalized to
Python integers. Construct a matching query before portable restoration,
including the same `at_least` and inversion settings. See the
{ref}`activity configuration example <functional-shared-activity-configuration>`
for scoped Network and Population continuation.

Native and functional APCount use `torch.int32` counts. Portable restoration
also accepts legacy `int64` and `float32` count carry when every value is a
finite, nonnegative integer within the signed 32-bit range. Restoration checks
these values before conversion and rejects unsafe values. Restored counts use
the same four-byte integer representation as fresh counts.

Both native and functional Raster return only checked post-step samples. A
restored suffix contains the remaining checked samples, or an empty sample axis
if its callback phase is outside the window. Concatenating prefix and suffix
rasters reconstructs the uninterrupted window without padding unchecked steps.

Recorder and LFP preparation follows the same separation. Recorder metadata
describes resolved paths, physical selections and cadence. LFP metadata
describes field values, projection rule, shape padding, dtype and device. A
fresh LFP run builds its differentiable field stack during initialization;
preparation retains the raw field references without caching that graph.
Portable continuation saves the detached stack as callback carry, alongside
the checked metadata.

Both `dn.callbacks.LFP` and `dn.func.LFP` bind to this same functional
configuration and carry contract. Native recording history is not imported.
The default rule preserves explicit batch lanes; a supplied einsum rule keeps
its exact reduction. Construct equivalent receiving fields and projection
settings before restoration. The
{ref}`shared LFP configuration example <functional-shared-lfp-configuration>`
checks live and portable continuation for a batched Population and Network.

`dn.Recorder` supplies the same observation configuration to native and
functional execution. HDF5 storage is a separate host session: portable
continuation includes the Recorder's numerical sampling phase, but excludes
its file path, writer process, queued samples and file counters. Construct the
receiving Recorder and destination explicitly. Restoring into a new file is
supported; reopening an existing file for cross-process append is currently
unsupported. Each host call with samples writes a separate `run_N` file segment,
and resumed callbacks emit only new samples. Calls without samples create no
datasets and consume no run number. See {ref}`functional-recorder-storage` for the
storage and result contract.

The example configures full-rate and cadenced Recorders, Raster, APCount and AnomalyDetector. Its
Network has two Populations linked continuously and an event synapse driven by
a noisy NetStim. Network callback scopes select the Population whose voltages
the threshold and anomaly analyses observe:

```{literalinclude} ../../examples/functional_callback_checkpoints.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
```

The {download}`complete script <../../examples/functional_callback_checkpoints.py>`
supplies all imports, helper functions and model constructors. Times are in ms;
the examples use CPU float64 and a fixed timestep of 0.025 ms.

(functional-callback-checkpoint-protocol)=
## Describe a custom callback's checkpoint configuration

A custom `dn.func.FunctionalCallback` can support portable continuation by
implementing `checkpoint_configuration()`. This optional hook describes the
callback's static meaning; evolving values remain in the callback carry. It
returns an exact plain dictionary with these three entries:

```python
{
    "kind": "org.example.voltage_energy",
    "version": 1,
    "fields": {"reference": -70.0, "scale": scale},
}
```

`kind` is a stable, nonempty ordinary namespaced string chosen by the callback author.
`version` is a positive ordinary integer; a boolean is not a version.
`fields` is an exact plain dictionary with ordinary string keys. Its values can contain
acyclic plain dictionaries, lists and tuples, `None`, ordinary boolean,
integer, float and string values, and ordinary `torch.Tensor` or
`torch.nn.Parameter` objects. Nested dictionary keys are also strings.
Arbitrary objects, callable values, subclasses of the allowed types and cycles
are not supported. Represent settings such as dtypes, devices or slices using
plain strings or their plain-data components.

Describe every setting that changes the meaning of the analysis, including
reference values, selections, scales or units as applicable. Choose a different
version when the meaning of the saved carry or its interpretation changes.
Equal carry shapes alone do not establish compatibility. Configuration tensors
are detached snapshots of metadata; their values, shape, dtype and declared
device participate in matching. They do not retain gradients or replace the
receiving callback's own configuration tensors.

The following callback accumulates squared, scaled voltage deviations and emits
each value and its sample number after each accepted step. Its initial emission
is `None`, so the first accepted step establishes its emission schema. Empty finalization
still returns the same result structure with explicitly shaped empty streams:

```{literalinclude} ../../examples/functional_callback_checkpoint_protocol.py
:language: python
:start-after: "# region custom_callback"
:end-before: "# endregion custom_callback"
```

`checkpoint_configuration()` must be pure and deterministic. Return metadata
without mutating the callback, consuming randomness or changing external state.
Dendra calls it during host checkpoint export and restoration, rather than
during binding, callback initialization, updates or finalization. If `_bind`
returns a separate custom callback implementation, implement the hook on that
returned object. `PopulationCallback` preserves its Population scope around the
inner callback's configuration:

```{literalinclude} ../../examples/functional_callback_checkpoint_protocol.py
:language: python
:start-after: "# region callback_configuration"
:end-before: "# endregion callback_configuration"
:dedent: 4
```

Construct the receiving callback explicitly with compatible configuration.
Restoration compares its metadata, then checks carry and emission schemas and
attaches the receiving collection's identity. The checkpoint does not import
callback code, instantiate a class or reconstruct a callback from these fields:

```{literalinclude} ../../examples/functional_callback_checkpoint_protocol.py
:language: python
:start-after: "# region custom_checkpoint"
:end-before: "# endregion custom_checkpoint"
:dedent: 4
```

When initialization emits `None`, the checkpoint retains the emission schema
discovered during execution. Restoration validates that saved schema without
running initialization or trial updates to discover a new one. Resumed updates
must match it. The configuration descriptor remains the author's declaration
that the receiving implementation gives those values the same meaning.

Custom callbacks without this hook retain ordinary execution and live in-memory
continuation. Portable callback export requires the hook when the bound
implementation does not use a supported built-in checkpoint contract.
Existing built-in checkpoint formats remain supported, including checked
conversion of legacy APCount count carry to `torch.int32`.

The complete
{download}`functional_callback_checkpoint_protocol.py <../../examples/functional_callback_checkpoint_protocol.py>`
example runs a Population and a connected Network with shared parameters and a
Population-scoped custom analysis. It writes and restores checkpoint files,
checks cumulative and emitted results, preserves a cadenced Recorder's phase,
and checks live and detached-suffix gradients:

```console
python examples/functional_callback_checkpoint_protocol.py
python examples/functional_callback_checkpoint_protocol.py --backend aot_eager
```

## Export and save a checkpoint

`dn.func.export_continuation(program, auxiliary["continuation"])`
and `program.export_continuation(auxiliary["continuation"])` return the same
versioned packet of ordinary data and Tensors. It includes complete simulation
state, the effective raw parameters/constants and checked callback carry with its
emission-schema metadata. This includes a callback schema discovered only after
the first post-step emission. It excludes already emitted recording history and
finalized results. Save earlier recordings separately when they are needed.

Export owns detached ordinary Tensor storage, including when invoked in
inference mode. It does not mutate the supplied continuation or retain its
autograd graph. Saved callback configuration is part of the compatibility metadata;
matching shapes alone is insufficient to establish that two reducers mean the
same thing.

Host runs capture their sources in `auxiliary["continuation"]`, including
explicit overrides and fresh/cache initialization replacements. Repeated Tensor
objects retain their identity ties across state, parameter and constant frames.
Export and restoration own storage independently of the input packet and
receiving defaults:

```{literalinclude} ../../examples/functional_callback_checkpoints.py
:language: python
:start-after: "# region checkpoint"
:end-before: "# endregion checkpoint"
```

`torch.load(..., weights_only=True)` reads the data/Tensor packet without
requiring Python callbacks, schedule classes or PyTree implementation classes
in the file. Trusted structured schedule values are reconstructed against the
receiving plan's schema.

Model and stimulation declarations, runtime input/draw tapes, optimizer state
and prior recording results are saved separately by the application. The
checkpoint preserves their effective execution sources, rather than serializing
the declarations that created them.

## Restore and continue

`dn.func.restore_continuation(receiving, packet, callbacks=receiving_callbacks)`
and `receiver.restore_continuation(packet)` validate the checkpoint against the
receiving source plan and callback collection. They return independently owned,
detached ordinary Tensor trees in a `Continuation`. Saved callback carry uses
the receiving collection's identity. Resume with
`dn.func.run(receiving_kernel, restored, ..., callbacks=receiving_callbacks)`,
a matching raw binding, or `receiver(restored)`.

Restoration does not initialize, prepare, sample or execute the model, advance
its clock/RNG, or replace runner defaults. Existing source, schema and required
alias checks apply. Invalid versions, malformed packets, incompatible schemas
and mismatched callback configurations reject restoration without changing the
receiver or supplied packet.

The example compares the reconstructed runner's complete state and joined
recordings with one uninterrupted run. A resumed Recorder returns only new
selected post-step samples, so joining the prefix and suffix has one initial frame.
Cadenced Recorders preserve their post-step counter. The example's `dt=3 * DT`
stream selects frames 0, 3, and 6 of an eight-step run; restoring after step
four retains that phase and selects only frame 6 from the suffix. Cadence-free
Recorder configuration and carry retain their existing checkpoint form.
Raster samples preserve their checked-window phase; APCount and anomaly results
retain their cumulative carry. Comparing complete Network state also checks pending
delivery and RNG continuation.

```{literalinclude} ../../examples/functional_callback_checkpoints.py
:language: python
:start-after: "# region demo"
:end-before: "# endregion demo"
```

When callback carry is saved, a different threshold, selection, scope, timestep
or lead field rejects restoration even if that carry has the same Tensor shapes.

### Resume through an explicit raw binding

A receiving lowered plan does not require a `dn.compile` runner or extracted
receiving frame values. Restoration checks the saved values against its trusted
metadata without extracting live native state, initializing, preparing or
sampling the model. Declare the receiving callbacks explicitly; a collection
from a different functional plan rejects even when its configuration matches.
Supply the matching collection when callback carry is saved. Initialized packets
also accept that collection before their first execution; omit it when callbacks
are not used.

```{literalinclude} ../../examples/functional_explicit_continuation_checkpoints.py
:language: python
:start-after: "# region restore"
:end-before: "# endregion restore"
:dedent: 4
```

Restoration creates owned source leaves, so construct a new raw binding with
`receiving_kernel.bind(restored)`. Binding selects only its parameter/constant
sources; it neither adopts nor validates state or callback carry. Supply the
restored packet and matching callbacks separately to `run`. The same host
scheduler also works with a receiving runner's kernel:

```python
bound = receiver.kernel.bind(restored)
print(dn.func.inspect_sources(bound))
state, auxiliary = dn.func.run(
    bound, restored, steps=STEPS, callbacks=receiver.callbacks,
)
state, auxiliary = dn.func.run(
    bound, auxiliary["continuation"], steps=STEPS, callbacks=receiver.callbacks,
)
```

`run` keeps the binding's source ownership. A packet with different Tensor
objects rejects rather than replacing that binding, even when its values and
shapes match. Successful raw-bound runs return a continuation containing their
state, effective sources and callback carry. Supplied callback collections
still have to match the packet's restored carry. Do not also pass a non-`None`
`callback_state=`. Input/draw tapes, runtime extracellular recipes and the next
horizon are supplied separately.

Inspecting that raw binding reports its effective parameter and constant
readers. Network readers owned by a parameter resolve to the same Tensor
object, including across the two frames. The authored binding maps remain
unchanged. Inspecting a `PopulationTensors` or `NetworkTensors` bundle instead
reports exactly the frames stored in that bundle. The report never performs
preparation, source validation or checkpoint restoration.

The complete
{download}`functional_continuation.py <../../examples/functional_continuation.py>`
example verifies same-binding continuation, joined recordings, spanning
gradients and explicit binding of restored sources for both model kinds. See
{ref}`functional-bound-continuation` for raw versus prepared binding ownership.

## Train after restoring

Raw parameter and constant `requires_grad` flags are retained. Trainable restored
sources are new leaves: they preserve tied identities but have no upstream
computation history. Distinct Tensor objects, including views of shared storage,
are independent snapshots; their storage sharing and upstream relationships are
not reconstructed. If initialization produced a retained value from a raw
parameter, the restored raw parameter and retained value are independent leaves,
even when both have trainability enabled.

Fresh or cached executions can use active retained materializations in the
constant frame. A resumed segment then differentiates with respect to those
effective values; a corresponding raw source may have zero or unused gradient.
Inspect the restored continuation to see these effective sources:

```python
report = dn.func.inspect_sources(restored)
print(report)
```

The display summarizes source and alias counts, trainable leaves and nonleaf
gradient-enabled sources, followed by a compact path tree with Tensor metadata.
Large frames are abbreviated with explicit omission counts; `report.entries`
retains the complete inventory. Tensor values are not printed.

The report retains every parameter/constant Tensor path and its live Tensor
reference. Aliases list other paths to the same object, including ties across
the parameter and constant frames. `report.trainable_leaves` contains each
Tensor object once when its captured `requires_grad` and `is_leaf` flags are
both true. It does not include state or callback carry. The receiving runner's
default optimization leaves remain separate.

Pass the `dn.compile` runner to inspect its defaults, or its returned
continuation to inspect a particular call's effective frames:

```{literalinclude} ../../examples/functional_continuation_checkpoints.py
:language: python
:start-after: "# region source_inspection"
:end-before: "# endregion source_inspection"
:dedent: 4
```

Nonleaf sources remain visible in `entries` but are excluded from
`trainable_leaves`. A live continuation can depend on caller-owned upstream
leaves that the report cannot recover. Reports snapshot names, aliases and
flags; Tensor values remain live. Inspect again after changing flags or source
bindings. Source inspection does not validate the model for execution, replay
initialization or traverse computation graphs. See
{ref}`functional-source-inspection` for the complete inspection contract.

```{literalinclude} ../../examples/functional_continuation_checkpoints.py
:language: python
:start-after: "# region trainability"
:end-before: "# endregion trainability"
:dedent: 4
```

The example differentiates its loss with respect to the report's eligible
leaves, rather than selecting an internal retained-value key. Its Population
case receives a raw-conductance gradient; its fresh Network case receives
gradients through saved effective constants. Eligibility alone does not prove
that a leaf affects this loss: some gradients are zero or unused. Both cases
retain raw trainability and aliases while their earlier gain parameter has no
gradient connection through the portable boundary.

After verifying continuation, the example supplies the deduplicated leaves to
an optimizer and performs a suffix update:

```{literalinclude} ../../examples/functional_continuation_checkpoints.py
:language: python
:start-after: "# region optimizer"
:end-before: "# endregion optimizer"
:dedent: 4
```

Optimizer state remains an application checkpoint. The report does not select
trainability, freeze sources or guarantee that an optimization step changes
every listed leaf.

Use live in-memory continuation when the initialization graph must span the
segment boundary. To train with a new initialization graph, construct a fresh
initialized training run explicitly. Restoration does not replay initialization
or rematerialize relationships among saved values.

## Ownership and supported configurations

Export and restoration are host operations outside compiled regions and
`torch.func` transforms. They introduce an explicit graph boundary. For
backpropagation through an in-memory segment boundary, pass the original
simulation state and callback carry directly without exporting or detaching
them. `checkpoint_every` is separate: it recomputes activations during backward
within one graph rather than making a portable detached checkpoint.

Restoration supports the same dtype and device as the receiving plan and callback
collection. Cross-device and dtype-changing restoration are currently
unsupported. Save and load a compatible model configuration rather than
changing checkpoint metadata to force acceptance.

Supported built-in callbacks are Recorder, LFP, AnomalyDetector, Raster, APCount,
Active, ActiveAL and ActiveALCount, including supported Population scopes and
physical Recorder regions. Eligibility follows the bound implementation:
authored subclasses that bind to a supported built-in implementation use its
checkpoint contract. Custom bound callbacks describe their configuration through
`checkpoint_configuration()` and use the same carry/emission validation and
restoration bridge. See {ref}`functional-callback-checkpoint-protocol`.
Their ordinary execution and in-memory continuation support both scan and
inference-only while using the same bound callback collection, including when
the optional checkpoint hook is absent. Native and pure Raster configurations
share a checkpoint contract when their resolved observation settings match.

For LFP, restored lead-field carry is detached. It does not acquire a gradient
connection to the receiving collection's lead-field Tensors. Use direct
in-memory continuation when such gradients must span the boundary.

Ordinary carry still belongs to its exact in-memory callback collection.
Restoration is the explicit checked bridge into an equivalent reconstructed
collection. The public `(state, auxiliary)` result and `callback_state=`
execution argument use the same contract for eager bindings, direct chunks,
host scheduling, scan and supported inference while loops.
Raw-bound `dn.func.run` additionally accepts the shared continuation packet;
direct calls and prepared bindings keep explicit state and callback arguments.

## Save callback carry independently

For independently saved analysis,
`callbacks.export_state(state)` and `callbacks.restore_state(packet)` provide
the callback-only bridge. Export owns detached carry and emission metadata;
restore validates the same qualified configurations and attaches the receiving
collection's identity. Pass the returned carry through `callback_state=` with
the saved simulation state and effective sources. These methods do not save
or restore simulation state, parameters, constants or recording history.
