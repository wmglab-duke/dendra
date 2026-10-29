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
bound = receiving_kernel.bind(restored)
state, auxiliary = dn.func.run(
    bound, restored, steps=100, callbacks=receiving_callbacks,
)
```

Here `functional` and `receiving` are plans returned by
`dn.func.make_functional`, and each callback collection belongs to its respective
plan. Both `dn.compile` runners also provide the bridge as methods:

```python
packet = program.export_continuation(auxiliary["continuation"])
torch.save(packet, path)

restored = receiver.restore_continuation(torch.load(path, weights_only=True))
state, auxiliary = receiver(restored)
```

Construct and initialize the receiving model, then lower or compile it
explicitly. The file contains versioned data and Tensors, rather than executable
model configuration.
The restored `dn.func.Continuation` supplies state, parameters, constants and
callback carry to the resume call. In the method example, both `program` and
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

`dn.func.initialize(functional, tensors)` supplies a fresh initialized packet
for explicit execution; see {ref}`functional-explicit-initialization`. Its
callback carry is `None`. Export that initial boundary without `callbacks=`,
or first run with the intended callbacks and export the returned continuation
with that collection. Restore resumes the saved initialization and does not
reapply initialization transforms or resample its selected realization. The
complete
{download}`functional_explicit_initialization.py <../../examples/functional_explicit_initialization.py>`
example checks initialization gradients before this boundary and trainable
effective sources after restoration.

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
from the receiving model or runner's defaults. Callback names and their bound
configurations must match: this includes observation paths and indices,
physical regions, Population scopes,
timestep, threshold windows, activity rules and LFP lead fields.

The example configures a Recorder, Raster, APCount and AnomalyDetector. Its
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

## Export and save a checkpoint

`dn.func.export_continuation(functional, auxiliary["continuation"], callbacks=callbacks)`
and `program.export_continuation(auxiliary["continuation"])` return the same
versioned packet of ordinary data and Tensors. It includes complete simulation
state, the effective raw parameters/constants and checked callback carry with its
emission-schema metadata. This includes a callback schema discovered only after
the first post-step emission. It excludes already emitted recording history and
finalized results. Save earlier recordings separately when they are needed.

Export owns detached ordinary Tensor storage, including when invoked in
inference mode. It does not mutate the supplied continuation or retain its
autograd graph. Callback configuration is part of the compatibility metadata;
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
detached ordinary Tensor trees in a `Continuation`, with the receiving callback
collection's identity. Resume with a new raw binding or `receiver(restored)`.

Restoration does not initialize, prepare, sample or execute the model, advance
its clock/RNG, or replace runner defaults. Existing source, schema and required
alias checks apply. Invalid versions, malformed packets, incompatible schemas
and mismatched callback configurations reject restoration without changing the
receiver or supplied packet.

The example compares the reconstructed runner's complete state and joined
recordings with one uninterrupted run. A resumed Recorder emits only new
post-step samples, so joining the prefix and suffix has one initial frame.
Raster emissions preserve their step phase; APCount and anomaly results retain
their cumulative carry. Comparing complete Network state also checks pending
delivery and RNG continuation.

```{literalinclude} ../../examples/functional_callback_checkpoints.py
:language: python
:start-after: "# region demo"
:end-before: "# endregion demo"
```

A different threshold, selection, scope, timestep or lead field rejects
restoration even if callback carry has the same Tensor shapes.

### Resume through an explicit raw binding

A receiving lowered plan does not require a `dn.compile` runner or extracted
receiving frame values. Restoration checks the saved values against its trusted
metadata without extracting live native state, initializing, preparing or
sampling the model. Declare the receiving callbacks explicitly; a collection
from a different functional plan rejects even when its configuration matches.
Omit `callbacks=` only when the checkpoint has no callback carry.

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

Supported callbacks are Recorder, LFP, AnomalyDetector, Raster, APCount,
Active, ActiveAL and ActiveALCount, including supported Population scopes and
physical Recorder regions. Eligibility follows the bound implementation:
authored subclasses that bind to a supported built-in implementation use its
checkpoint contract. Checkpoint export/restoration for custom bound callback
implementations is currently unsupported; their ordinary functional execution
and in-memory continuation use the existing contracts.

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
