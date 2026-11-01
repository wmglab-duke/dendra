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

(shared-api-conformance)=
# Continuing between native, compiled and saved execution

A simulation can begin with ordinary native execution, continue through a
`dn.compile` runner, survive a portable checkpoint and return to a native model.
Use the same deterministic input packets, horizon choices and callback
configurations at each stage. State and source ownership change at the explicit
snapshot, portable restore and native handoff boundaries.

The complete runnable
{download}`shared_api_conformance.py <../../examples/shared_api_conformance.py>`
example covers a passive cable and a two-Population Network with a continuous
connection. Both use shared conductance parameters, voltage/current input tapes
and a Recorder sampling every two solver steps. The example checks continued
recording times, native/functional values, source aliases and gradients after
returning to native execution.

## Begin with native execution

Prepare a pure callback collection for the initialized model and pass it through
`callbacks=`. Its returned state carries the recording phase:

```{literalinclude} ../../examples/shared_api_conformance.py
:language: python
:start-after: "# region native_prefix"
:end-before: "# endregion native_prefix"
:dedent: 4
```

`duration=2.4 * DT` accepts two steps and retains `0.4 * DT` of physical time.
That pending duration belongs to the model's runtime state. The next stage
receives it along with the voltage and mechanism state.

Standalone `Population.initialize()` does not take a timestep. Supply `dt=`
to native Population execution and to `dn.compile`. Networks use the timestep
set by `network.initialize(dt)`; see {doc}`A62_shared_execution_horizons`.

## Compile the current snapshot

`dn.compile` captures the initialized model's current state. A no-state call
starts from that snapshot; pass the returned continuation to advance from a
previous compiled call. Training sources selected by `trainable=` are owned by
the runner, independently of the native model's Parameters.

Native and compiled callback collections have separate carry identities.
Export and restore callback state to transfer the native recording phase into
the runner's compatible collection:

```{literalinclude} ../../examples/shared_api_conformance.py
:language: python
:start-after: "# region compiled"
:end-before: "# endregion compiled"
:dedent: 4
```

The example's `trainable="*g_param"` selector matches the shared conductance
source. Source sharing within the model is preserved. Use
`dn.func.inspect_sources(program)` to inspect runner defaults, or inspect the
returned continuation for the effective sources of a particular call. Reports
retain alias paths and deduplicate trainable leaves.

`steps=` leaves pending physical duration unchanged; `duration=` uses it when
resolving complete steps. Input tapes contain exactly the accepted steps of
each call. See {doc}`A61_shared_runtime_inputs` for input ownership and
stimulation rules.

## Save and restore a portable continuation

A runner's portable continuation includes its effective parameters/constants,
runtime and RNG state, pending duration and compatible callback carry. Restore
it against a structurally compatible runner with the same callback
configurations:

```{literalinclude} ../../examples/shared_api_conformance.py
:language: python
:start-after: "# region checkpoint"
:end-before: "# endregion checkpoint"
:dedent: 4
```

Portable restoration creates independent source leaves with the saved
trainability and alias relationships. It starts a new autograd graph. An
optimizer for the restored simulation should use its restored leaves; existing
optimizers for the earlier runner continue to own their earlier sources.
`dn.func.inspect_sources(restored).trainable_leaves` exposes the restored
trainable leaves.

Input tapes and external stimulation remain call-local. Supply the next tape
segment when continuing; the checkpoint does not save future inputs or an
optimizer. Prepared callback configurations are validated by the receiving
collection. See {doc}`A58_functional_callback_checkpoints` for custom callback
configuration and portable compatibility.

## Return to native execution

Construct an initialized native target with compatible structure, source
sharing, dtype/device, timestep and training mode. Transfer callback carry to
its own prepared collection, and transfer model sources/state with
`commit_continuation_`:

```{literalinclude} ../../examples/shared_api_conformance.py
:language: python
:start-after: "# region native_handoff"
:end-before: "# endregion native_handoff"
:dedent: 4
```

Native handoff copies detached values into the target's existing raw sources.
Its Parameters retain their identities, aliases and trainability, so an existing
native optimizer still refers to the correct leaves. Handoff preserves runtime
and RNG continuation without initializing or advancing the target. It does not
install callbacks; the explicit callback state bridge preserves their phase.

The example uses an independent receiving model so the saved prefix graph
remains usable. Subsequent native gradients belong to the receiving Parameters
and the new call's inputs. They do not propagate into the saved functional
prefix:

```{literalinclude} ../../examples/shared_api_conformance.py
:language: python
:start-after: "# region gradients"
:end-before: "# endregion gradients"
:dedent: 4
```

The receiving model must satisfy the supported source-storage and native
materialization contracts. Handoff rejects source or constant overrides that
cannot be represented natively. See {doc}`A59_functional_native_handoff` for
the full compatibility and gradient ownership rules.

## Apply the workflow to a Network

The ownership transitions and callback bridges are the same for a Network.
The model-specific arguments are:

| Configuration | Population | Network |
| --- | --- | --- |
| Native timestep | Pass `dt=` at execution | Use `network.initialize(dt)` |
| Deterministic inputs | `Inputs(ve=..., intra=...)` | Mapping from Population names to `Inputs` |
| Recorder fields | `("v", "t")` | Qualified fields such as `("left.v", "right.v", "t")` |
| Scoped observations | Observe the Population directly | Use `dn.callbacks.PopulationCallback("left", callback)` |

The Network companion uses the same passive-cable constructor as the Population
example, connects it through a built-in graded synapse and shares the raw
conductance source across both Populations:

```{literalinclude} ../../examples/shared_api_conformance.py
:language: python
:start-after: "# region connection_factory"
:end-before: "# endregion connection_factory"
```

The complete companion follows the same execution transitions. Its input
mapping supplies one shared tape to both Populations; each Population could
instead receive its own packet or omit a drive field. A Network runner call
that receives callback carry also requires explicit simulation state. The first
compiled Network call supplies `program.initial_state` together with the
restored native callback state:

```{literalinclude} ../../examples/shared_api_conformance.py
:language: python
:start-after: "# region network"
:end-before: "# endregion network"
:dedent: 4
```

Shared parameters, connection queues and RNG streams remain part of the
Network's explicit continuation. Each transition preserves the execution
ordering of the selected model. Callback storage and call-local inputs retain
their own ownership rules.

Both complete workflows execute here:

```{code-cell} python
:load: ../../examples/shared_api_conformance.py
:tags: [remove-input]
```
