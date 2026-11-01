---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.13'
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(functional-native-handoff)=
# Continuing between native and functional execution

`dn.func.capture_continuation(target, native_model=None, *, callbacks=None,
callback_state=None)` captures the current native Population or Network as an
independent functional continuation, optionally retaining existing callback carry.
`dn.func.commit_continuation_(target, native_model, continuation)` transfers a
functional continuation back to a native model. Both operations include sources,
simulation state, clocks and RNG carry at the selected boundary.

## Capture current native execution

`target` is a lowered functional Population/Network plan or a `dn.compile`
program. Omit `native_model` to capture the native model used to construct that
target, or pass a compatible initialized native model explicitly. The native
model must match the plan's structure, dtype/device, mode and source sharing.
A built, initialized Population can be captured before its first native run at
the plan's timestep. If native execution has already configured its integrator,
that timestep must match the plan so existing histories retain their meaning.
A Network must be initialized at the plan's timestep.

The complete runnable
{download}`functional_continuation_capture.py <../../examples/functional_continuation_capture.py>`
example covers both target forms for a Population and a two-Population Network
with a shared parameter:

```{literalinclude} ../../examples/functional_continuation_capture.py
:language: python
:start-after: "# region capture"
:end-before: "# endregion capture"
:dedent: 4
```

Capture reads the native model's current values, including changes since
lowering or compilation. It ignores edited program defaults, the program's
`trainable` selection and its snapshot/fresh restart policy. Source Tensor
identity ties and trainability are preserved in owned detached Tensor leaves;
later edits to native tensors or program defaults do not change the packet.
Suffix gradients through the captured model sources belong to those leaves and
do not reach the native model's sources or its earlier recurrent graph. Use
`dn.func.inspect_sources(continuation).trainable_leaves` to select the captured
model sources for an optimizer. Callback-bound LFP field sources have the
separate gradient behavior described below.

Capture does not initialize a model, advance time or sample random values. It
reads retained draws and native materializations, and deterministically derives
the snapshot baselines needed to preserve those materializations. Native and
program state, defaults, RNG and execution reports retain their values. Capture
runs outside `torch.compile`, `torch.func` transforms and forward automatic
differentiation.

Omitting `callback_state=` returns a packet with no callback carry. On its first
functional execution, the receiving collection starts at the captured boundary,
including its usual initial recording sample. The packet can be exported before
that first call:

```{literalinclude} ../../examples/functional_continuation_capture.py
:language: python
:start-after: "# region checkpoint"
:end-before: "# endregion checkpoint"
:dedent: 4
```

The example supplies `callbacks` for explicit plans and uses `None` for programs,
which select their configured collection.

An extracellular `extra` recipe bound during lowering belongs to the plan;
capture snapshots its represented field and waveform Tensor sources. Recreate
equivalent bindings on a receiving plan when restoring a portable packet.
`Inputs`, runtime draw tapes and call-local `extra` recipes for `VarInput` slots
remain invocation arguments and must be supplied again when execution resumes.

### Retain native callback reductions and recording phase

Use the same prepared callback collection for native warmup and functional
execution. A runner exposes its collection as `program.callbacks`; an explicit
plan prepares one with `plan.make_callbacks(...)`. Pass the native result's
`.state` through `callback_state=` when capturing. For an explicit plan, also
supply `callbacks=callbacks`. Runners infer their configured collection and
reject a non-`None` `callbacks=` argument.

The complete
{download}`functional_callback_capture.py <../../examples/functional_callback_capture.py>`
example covers a Population and a two-Population Network through both target
forms. Its native prefix uses the target's prepared collection with a cadenced
Recorder and an APCount accumulator:

```{literalinclude} ../../examples/functional_callback_capture.py
:language: python
:start-after: "# region capture_callbacks"
:end-before: "# endregion capture_callbacks"
:dedent: 4
```

The resulting packet retains callback totals, threshold crossing state and
Recorder phase. Its first functional call emits only new selected post-step
samples; joining the native prefix's recordings with the suffix does not repeat
the boundary sample. Capture copies recurrent callback values into owned
detached storage, preserving their trainability flags while cutting the earlier
recurrent graph. Later edits to the native result's carry do not change the
packet. It does not initialize or update callbacks, finalize observations, open
recording storage, or include earlier recordings. Custom callbacks in the same
collection do not need `checkpoint_configuration()` for capture.

Built-in LFP lead-field carry is rebuilt from the callback collection's current
bound field sources and copied into owned storage. This refresh follows the
ambient gradient mode: keep gradients enabled when the functional suffix should
train those external fields. Its new field graph supports suffix gradients even
after backward has consumed the native prefix's LFP graph. Model-source and
recurrent-prefix gradients remain cut. Externally computed nonleaf field sources
retain their own upstream graph lifetime, as with
{ref}`functional-continuation-detach`.

The captured packet includes model sources, state and RNG together with callback
carry, so it can be exported before its first functional call. Portable export
retains its existing callback requirements: built-ins are supported, and custom
callbacks need the optional checkpoint protocol. The example saves and restores
the built-in Recorder and APCount state into a reconstructed receiving target:

```{literalinclude} ../../examples/functional_callback_capture.py
:language: python
:start-after: "# region checkpoint_callbacks"
:end-before: "# endregion checkpoint_callbacks"
:dedent: 4
```

Portable LFP export and restoration retain detached saved field carry. To resume
field training after restoration, call `dn.func.detach_continuation` with
gradients enabled to refresh LFP carry from the receiving collection's current
field sources before execution.

Carry belongs to one prepared collection. When native warmup uses a different
collection, first use its `export_state(...)` and the receiving collection's
`restore_state(...)` bridge, then pass the restored carry to capture. Capture
does not infer or adopt another collection's identity. Save prefix recordings
separately. See {doc}`A58_functional_callback_checkpoints` for callback
compatibility and portable custom callbacks.

Run the complete example with:

```console
python examples/functional_callback_capture.py
```

## Transfer a functional continuation to a native model

`dn.func.commit_continuation_(program_or_plan, target, continuation)` transfers a
functional continuation to an initialized native Population or Network. It
returns `target`, which can then use its ordinary `step`, `run`, `longrun` or
`longrun_checkpointed` methods.

The transfer includes supported raw parameter and constant sources, simulation
state, clocks and RNG state. Copying the sources matters after functional
training or per-call overrides: matching runtime state alone does not make the
receiving model's weights, delays or mechanism parameters match the simulation
that produced it.

The complete runnable
{download}`functional_native_handoff.py <../../examples/functional_native_handoff.py>`
example covers both a Population and a two-Population Network with a shared
parameter:

```{literalinclude} ../../examples/functional_native_handoff.py
:language: python
:start-after: "# region handoff"
:end-before: "# endregion handoff"
:dedent: 4
```

The first argument can be either kind of `dn.compile` runner or the matching
functional Population/Network plan. Pass the exact `Continuation` returned by
execution, initialization or portable restoration. The target must match the
plan's structure, timestep, dtype/device, training mode and source-sharing
relationships. Its current state and source values may differ.

## Source and gradient ownership

Handoff copies detached values into the native model. Receiving raw Parameters
and source buffers keep their identities, aliases and trainability, so
existing optimizers still refer to the receiving Parameters. Subsequent native
gradients belong to those Parameters. Gradients do not cross the handoff into
the functional prefix or the continuation's source expressions.
Optimizer state and existing gradient buffers remain under the caller's
control; model handoff does not transfer an optimizer.

```{literalinclude} ../../examples/functional_native_handoff.py
:language: python
:start-after: "# region resume"
:end-before: "# endregion resume"
:dedent: 4
```

The operation does not mutate the input continuation's containers or Tensor
objects. To retain its upstream training graph, use an independent receiving
model. If its source expressions depend on the receiving model's raw sources,
copying different values into those sources advances their version counters,
as an optimizer update would, and can invalidate the older graph. Keep an
independent continuation to continue functional training, or export it with
`dn.func.export_continuation` for portable storage. Handoff is a host operation
outside `torch.compile`, `torch.func` transforms and forward automatic
differentiation.

## Validation and native materializations

The operation validates source layout, source aliases and runtime state before
publication, and stages the corresponding native materializations. It does not
initialize the target, resample random values or advance its simulation. RNG
carry and retained draws continue from the supplied boundary.

Raw sources with a supported native write route can be transferred. Derived
workspaces and effective materializations must agree with the native values
represented by those sources and runtime state. Overrides with no supported
native representation, including a different solver timestep, are rejected
before the target is changed. Rebuild and initialize a compatible model when
changing structure or timestep.

Writable native sources must be ordinary, versioned Tensor or Parameter
objects with supported strided storage. Inference tensors, internally
overlapping layouts, and different source views sharing storage are unsupported.
Incoming inference tensors are copied into ordinary native runtime storage.
Source values that already match are left untouched, preserving their version
counters.

The existing `plan.commit_state_(target, state)` methods remain state-only
operations. They retain their existing graph behavior and require callers to
align effective native sources separately. Use `commit_continuation_` for a
detached transfer of sources and state together.

## Callbacks and call-local stimulation

Callback carry belongs to its prepared callback collection. Model handoff
leaves `continuation.callback_state` unchanged and does not install callbacks
or open storage. Reuse the same prepared collection on a compatible initialized
native receiver, including `program.callbacks`:

```{literalinclude} ../../examples/shared_callback_rebinding.py
:language: python
:start-after: "# region handoff"
:end-before: "# endregion handoff"
:dedent: 4
```

`commit_continuation_` transfers detached model sources and simulation state.
`callbacks.detach_state(packet.callback_state)` separately truncates the callback
recurrent graph, retaining reducer totals, crossing latches and Recorder cadence.
The native suffix emits only new selected samples, without repeating the shared
boundary. Omit `callback_state=` to start fresh observations at that boundary.
Keep live carry without detaching only when its earlier callback graph should
remain connected.

Native callback observations come from the receiving model. Prepared paths,
Population scopes and selection indices remain fixed; compatible receivers
match the collection's model structure, runtime layout, dtype/device, execution
mode and source-sharing requirements. Population execution checks its requested
timestep against prepared Recorder cadence and threshold timing; a Network's
initialized timestep must match the plan. Physical Slice and region
configuration still requires the exact original source model during preparation;
the resulting indices are reused on the receiver. Functional execution still
requires the collection's exact owning plan. A different collection requires
the explicit export/restore bridge, even if its configuration matches.

After model handoff and callback detachment, native model-parameter gradients
belong to the receiving Parameters. They do not reach the functional prefix,
the program's independent parameters or the source native model. External LFP
field sources remain configured on the same callback collection. Detaching
callback state with gradients enabled refreshes its field carry from those
sources and retains suffix field gradients; it does not transfer fields into
the receiving model. Externally computed nonleaf field sources retain their own
upstream graph lifetime.

Custom pure callbacks need no serialization protocol for same-collection reuse:

```{literalinclude} ../../examples/shared_callback_rebinding.py
:language: python
:start-after: "# region reducer"
:end-before: "# endregion reducer"
```

The complete
{download}`shared_callback_rebinding.py <../../examples/shared_callback_rebinding.py>`
example covers Populations and Networks, compares cadenced recordings and
reducer totals with native and functional references, and checks receiving
Parameter gradients:

```console
python examples/shared_callback_rebinding.py
```

To use an independently prepared receiving collection or save portable carry,
transfer callback state through its checkpoint protocol:

```python
saved_callbacks = program.callbacks.export_state(continuation.callback_state)
native_callbacks = dn.func.make_callbacks(
    target, {"trace": dn.Recorder(("v", "t"), dt=recording_dt)}, dt=solver_dt
)
native_callback_state = native_callbacks.restore_state(saved_callbacks)
target.run(
    tstop=duration,
    dt=solver_dt,
    callbacks=native_callbacks,
    callback_state=native_callback_state,
)
```

This portable snippet uses a Population. Use equivalent callback configurations
and the same names when preparing the receiving collection. For Networks, use the
initialized Network timestep, omit the `dt` argument to `run`, and select the
corresponding Network recorder paths or `dn.callbacks.PopulationCallback`
scopes. Recorded history and HDF5 sessions remain separately owned; see
{doc}`A58_functional_callback_checkpoints`. Portable export requires the optional
checkpoint protocol for custom callbacks.

Inputs, runtime draws and call-local `extra` arguments are not stored in a
`Continuation`. Supply deterministic inputs and stimulation again when native
execution resumes. Native `step` accepts a single `RuntimeDraws` packet for
NOISE/SDE and Network NetStim; native run-family draw tapes and randomized
Poisson replay are currently unsupported. See {doc}`A61_shared_runtime_inputs`.
Bindings that have no native owner cannot be silently installed by model handoff.

The complete example executes here:

```{code-cell} python
:load: ../../examples/functional_native_handoff.py
:tags: [remove-input]
```
