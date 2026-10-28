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
and sampling phase. Both `dn.compile` runners provide the same portable bridge:

```python
packet = program.export_continuation(auxiliary["continuation"])
torch.save(packet, path)

restored = receiver.restore_continuation(torch.load(path, weights_only=True))
state, auxiliary = receiver(restored)
```

Construct and initialize the receiving model and runner explicitly. The file
contains versioned data and Tensors, rather than executable model configuration.
The restored `dn.func.Continuation` supplies all four operands to the existing
resume call.

Use `runner.callbacks` to access the runner's bound callback collection. It is a
read-only property and returns `None` when recording and reducers are omitted.
For an explicitly lowered Population or Network, use the collection returned by
`functional.make_callbacks(...)` instead. Both expose the same `export_state`
and `restore_state` methods.

For continuation with live graph history, pass `auxiliary["continuation"]`
directly to the next call. Portable export is a separate operation with an
explicit autograd boundary. Save the result of `export_continuation`, rather
than pickling the in-memory `Continuation` object.

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

## Configure the receiving analysis explicitly

A checkpoint contains data, rather than constructors for models or callbacks.
Construct and initialize an equivalent ordinary model, compile it with the
intended callback configurations, then restore against that runner. Owner kind,
timestep, native mode, component layouts and the parameter/constant/state
schemas must be compatible. Saved parameter and constant values can differ
from the receiving runner's defaults. Callback names and their bound configurations must match: this
includes observation paths and indices, physical regions, Population scopes,
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

`program.export_continuation(auxiliary["continuation"])` returns a versioned
packet of ordinary data and Tensors. It includes complete simulation state,
the effective raw parameters/constants and checked callback carry with its
emission-schema metadata. This includes a callback schema discovered only after
the first post-step emission. It excludes already emitted recording history and
finalized results. Save earlier recordings separately when they are needed.

Export owns detached ordinary Tensor storage, including when invoked in inference mode.
It does not mutate the supplied continuation or retain its autograd graph. Callback
configuration is part of the compatibility metadata; matching shapes alone is
insufficient to establish that two reducers mean the same thing.

The runner captures the sources in `auxiliary["continuation"]`, including
explicit overrides and fresh/cache initialization replacements. Repeated raw
parameter/constant Tensor references retain their aliases, including cross-field
ties. Export and restoration own storage independently of the input packet and
runner defaults:

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
and prior recording results are saved separately by the application. The runner
checkpoint preserves their effective execution sources, rather than serializing
the declarations that created them.

## Restore and continue

`receiver.restore_continuation(packet)` validates the checkpoint against the
receiving source plan and callback collection. It returns independently owned,
detached ordinary Tensor trees in a `Continuation`, with the receiving callback
collection's identity. Resume with `receiver(restored)`.

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

## Train after restoring

Raw parameter and constant `requires_grad` flags are retained. Trainable restored
sources are new leaves: they preserve tied identities but have no upstream
computation history. Relationships between distinct saved Tensors are not
reconstructed. If initialization produced a retained value from a raw parameter,
the restored raw parameter and retained value are independent leaves, even when
both have trainability enabled.

Fresh or cached executions can use active retained materializations in the
constant frame. A resumed segment then differentiates with respect to those
effective values; a corresponding raw source may have zero or unused gradient.
Choose the effective trainable leaves used by the model from the restored
parameter and constant frames, deduplicated by Tensor identity, when building
an optimizer. The receiving runner's default optimization leaves remain
separate, and optimizer state is an application checkpoint.

```{literalinclude} ../../examples/functional_continuation_checkpoints.py
:language: python
:start-after: "# region trainability"
:end-before: "# endregion trainability"
:dedent: 4
```

The Population case differentiates the resumed segment with respect to its raw
conductance. The fresh Network case selects the retained effective conductance
in its saved constant frame. That constant path is specific to the example's
declaration. Inspect your model's restored frames when choosing optimization
inputs. Both cases
retain raw trainability and aliases while their earlier gain parameter has no
gradient connection through the portable boundary.

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

## Save callback carry independently

For explicitly lowered execution or independently managed analysis,
`callbacks.export_state(state)` and `callbacks.restore_state(packet)` provide
the callback-only bridge. Export owns detached carry and emission metadata;
restore validates the same qualified configurations and attaches the receiving
collection's identity. Pass the returned carry through `callback_state=` with
the saved simulation state and effective sources. These methods do not save
or restore simulation state, parameters, constants or recording history.
