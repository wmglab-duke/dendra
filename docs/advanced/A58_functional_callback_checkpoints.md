---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-callback-checkpoints)=
# Save and restore callback continuation

Functional simulation state contains clocks, delay histories, event queues and
RNG carry. Functional callbacks have separate continuation state for reductions,
threshold crossing latches and sampling phase. Save both to resume a simulation
and its analysis in a reconstructed runner.

Use `runner.callbacks` to access the runner's bound callback collection. It is a
read-only property and returns `None` when recording and reducers are omitted.
For an explicitly lowered Population or Network, use the collection returned by
`functional.make_callbacks(...)` instead. Both expose the same `export_state`
and `restore_state` methods.

The complete
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

## Configure the receiving analysis explicitly

A checkpoint contains data, rather than constructors for models or callbacks.
Construct and initialize an equivalent ordinary model, compile it with the
intended callback configurations, then restore the saved carry against that
collection. Callback names and their bound configurations must match: this
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

`callbacks.export_state(auxiliary["callbacks"].state)` returns a versioned packet
of ordinary data and Tensors. It contains callback carry and emission-schema
metadata, including a schema discovered only after the first post-step emission.
It excludes the already emitted recording history and finalized results.
Save earlier recordings separately when they are needed.

Export owns detached Tensor storage, including when invoked in inference mode.
It does not mutate the supplied carry or retain its autograd graph. Callback
configuration is part of the compatibility metadata; matching shapes alone is
insufficient to establish that two reducers mean the same thing.

The example stores complete simulation state and effective parameter/constant
values alongside the callback packet. Its `snapshot` helper preserves shared
parameter aliases while making independent detached storage:

```{literalinclude} ../../examples/functional_callback_checkpoints.py
:language: python
:start-after: "# region checkpoint"
:end-before: "# endregion checkpoint"
```

`torch.load(..., weights_only=True)` reads the data/Tensor packet without
requiring Python callback objects or PyTree implementation classes in the file.
The original `FunctionalCallbackState` object is an in-memory carry type;
export it through the collection before saving a portable checkpoint.

When fresh initialization replaces parameters or constants, save
`auxiliary["parameters"]` and `auxiliary["constants"]`. Otherwise the runner's
default mappings are sufficient when no per-call override was used. If a call
supplied explicit overrides, save those effective values. Restoring callback
carry does not restore model parameters, geometry, stimulation configuration,
optimizer state or prior recording results.

## Restore and continue

`receiver.callbacks.restore_state(packet)` validates the checkpoint against the
receiving collection and returns ordinary, detached `FunctionalCallbackState`
with the receiving collection's identity. Pass that state through the existing
`callback_state=` argument alongside the saved simulation state. Restoration
does not call callback initialization or advance a model/RNG clock.

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
restoration even if its carry has the same Tensor shapes. Invalid versions,
malformed packets and incompatible tensor schemas also reject restoration.
The receiving collection and supplied packet retain their values when
validation fails.

## Ownership and supported configurations

Export and restoration are host operations outside compiled regions and
`torch.func` transforms. They introduce an explicit graph boundary. For
backpropagation through an in-memory segment boundary, pass the original
simulation state and callback carry directly without exporting or detaching
them. `checkpoint_every` is separate: it recomputes activations during backward
within one graph rather than making a portable detached checkpoint.

Restoration supports the same dtype and device as the receiving callback
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
