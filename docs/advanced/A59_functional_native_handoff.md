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

`dn.func.capture_continuation(target, native_model=None)` captures the current
native Population or Network as an independent functional continuation.
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
Suffix gradients belong to the captured leaves and do not reach the native
prefix. Use `dn.func.inspect_sources(continuation).trainable_leaves` to select
the captured sources for an optimizer.

Capture does not initialize a model, advance time or sample random values. It
reads retained draws and native materializations, and deterministically derives
the snapshot baselines needed to preserve those materializations. Native and
program state, defaults, RNG and execution reports retain their values. Capture
runs outside `torch.compile`, `torch.func` transforms and forward automatic
differentiation.

The returned packet has no callback carry. On its first functional execution,
the receiving collection starts at the captured boundary, including its usual
initial recording sample. The packet can be exported before that first call:

```{literalinclude} ../../examples/functional_continuation_capture.py
:language: python
:start-after: "# region checkpoint"
:end-before: "# endregion checkpoint"
:dedent: 4
```

The example supplies `callbacks` for explicit plans and uses `None` for programs,
which select their configured collection. To continue existing native callback
reductions or recording phase, export the native collection's state and restore
it into the receiving functional collection through the existing callback
bridge. Attach that restored carry explicitly to a `Continuation`; capture
does not inspect native callback histories. See
{doc}`A58_functional_callback_checkpoints`.

An extracellular `extra` recipe bound during lowering belongs to the plan;
capture snapshots its represented field and waveform Tensor sources. Recreate
equivalent bindings on a receiving plan when restoring a portable packet.
`Inputs`, runtime draw tapes and call-local `extra` recipes for `VarInput` slots
remain invocation arguments and must be supplied again when execution resumes.

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
or open storage. Transfer callback carry to a compatible receiving collection
through its existing portable state protocol:

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

This snippet uses a Population. Use equivalent callback configurations and the
same names when preparing the receiving collection. For Networks, use the
initialized Network timestep, omit the `dt` argument to `run`, and select the
corresponding Network recorder paths or `dn.callbacks.PopulationCallback`
scopes. Recorded history and HDF5 sessions remain separately owned; see
{doc}`A58_functional_callback_checkpoints`.

Inputs, runtime draw tapes and call-local `extra` arguments are not stored in a
`Continuation`. Supply them again for resumed native execution through the
corresponding native input or stimulation API. Bindings that have no native
owner cannot be silently installed by model handoff.

The complete example executes here:

```{code-cell} python
:load: ../../examples/functional_native_handoff.py
:tags: [remove-input]
```
