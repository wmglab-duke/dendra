---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-noise-workspaces)=
# Prepare workspaces from noise

Mechanism and nested State `derive_buffers()` and
`derive_timestep_buffers(dt)` hooks can read declared NOISE buffers. Native
and functional execution use the same preparation hooks and keep the sample
selected at each preparation boundary. Subsequent runtime NOISE refreshes
do not rebuild these workspaces.
`GLOBALNOISE`, `BATCHNOISE`, and `RANGENOISE` retain their declared sampling
scopes, including explicit Population batches.

The complete
{download}`functional_noise_workspaces.py <../../examples/functional_noise_workspaces.py>`
example compares a two-Population Network's trajectories, shared-parameter
gradients, and continuation with native execution. Run it with
`python examples/functional_noise_workspaces.py`.

## Declare the workspace

This leak mechanism draws initialization noise for its conductance and
uses a timestep workspace to accumulate a fixed increment:

```{literalinclude} ../../examples/functional_noise_workspaces.py
:language: python
:start-after: "# region declaration"
:end-before: "# endregion declaration"
```

`derive_buffers()` builds `conductance` after the owner's initialization
NOISE sample. State defaults and `initial_values()` can read that derived
value. Fresh initialization gives `TIMESTEP_BUFFER` values their declared
zero layout and does not call `derive_timestep_buffers(0)`.

When the integrator configures `dt`, `derive_timestep_buffers(dt)` builds
`increment` from the static conductance and the NOISE retained at that
boundary. Each simulation timestep refreshes `eta` before the state
transition. `conductance` and `increment` retain their prepared values,
while a hook that reads `eta` directly receives the current step's sample.
Configuring another timestep rebuilds the timestep workspace from the
current retained NOISE; it does not resample NOISE or rebuild the
initialization-static workspace.

Both hooks remain pure: read populated tensors, return the declared
outputs, and avoid implicit random draws or mutations. NOISE and
distribution transforms remain detached; ordinary parameters and the
explicit timestep remain differentiable through workspace algebra. Later
runtime NOISE refreshes preserve the prepared realization and its backward
graph.

The same rules apply independently to nested States. MaterialProcesses use
their `prepare_materials(fields, dt)` API rather than declaring these
Mechanism workspaces; see {doc}`A44_functional_material_stochasticity`.

## Compile and initialize

Compile the ordinary Network and select its shared raw conductance:

```{literalinclude} ../../examples/functional_noise_workspaces.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 8
```

The host runner samples omitted initialization and runtime packets.
Explicit packets expose the same boundaries for pure functional execution:

```{literalinclude} ../../examples/functional_noise_workspaces.py
:language: python
:start-after: "# region initialization"
:end-before: "# endregion initialization"
:dedent: 8
```

Initialization samples root Population declarations before pre-initialize
transforms and inserted owners afterward. Derived preparation uses its
owner's selected samples and parameter frame. Shared raw parameters retain
their aliases, including when a later Population's initialization replaces
a shared source. Repeated preparation consumes no draws. See
{doc}`A45_functional_stochastic_initialization_ordering` for sampling around
parameter-writing transforms.

## Continue the prepared realization

Runtime draw tapes advance the explicit RNG carry independently of the
prepared workspace. Continue using the returned parameter, constant, and
state packets:

```{literalinclude} ../../examples/functional_noise_workspaces.py
:language: python
:start-after: "# region execution"
:end-before: "# endregion execution"
:dedent: 8
```

Scan, checkpointed scan, and inference-only while execution retain the
same workspace realization. Snapshot initialization preserves the native
initialization and timestep preparation samples even when runtime NOISE
has advanced since configuration. Steady-state caches retain their saved
numerical workspaces and restore runtime NOISE/RNG state according to
Population or Network cache ownership. Fresh initialization resamples
according to the declarations' native policies and constructs a new
initialization-static frame.

Native runtime checkpoint commit preserves the prepared samples alongside
the independently current NOISE and RNG carry. A compatible destination can
continue without reconstructing a workspace from the wrong runtime
sample. Ordinary model state dictionaries retain the initialization NOISE
frame; loading one rebuilds timestep workspaces at the next solver
configuration. Align raw parameters separately if functional optimization changed
them. The example checks ordinary parameter gradients through the retained
preparation:

```{literalinclude} ../../examples/functional_noise_workspaces.py
:language: python
:start-after: "# region verification"
:end-before: "# endregion verification"
:dedent: 8
```

## Run the complete example

```{code-cell} python
:load: ../../examples/functional_noise_workspaces.py
:tags: [remove-input]
```
