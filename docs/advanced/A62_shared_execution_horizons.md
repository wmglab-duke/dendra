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

(shared-execution-horizons)=
# Choosing execution steps and duration

Native Population and Network `run`, `longrun` and `longrun_checkpointed`
accept the same explicit horizon choices as `dn.func.run` and compiled runners:

- `steps=N` advances exactly `N` complete solver steps.
- `duration=T` requests `T` milliseconds of additional simulation time.

Supply one horizon per call. `steps` must be a nonnegative Python integer;
booleans are rejected. `duration` must be a finite, nonnegative real scalar.
`steps=0` and `duration=0` are valid calls. Native calls advance the model's
state; functional calls return state and an explicit continuation.

The complete runnable
{download}`shared_execution_horizons.py <../../examples/shared_execution_horizons.py>`
example checks mixed horizons, native/functional gradients and registered
intracellular stimulation.

## Populations

Choose the horizon independently of the execution method:

```python
model.run(steps=100, dt=dt)
model.longrun(duration=10.0, chunklength=32, dt=dt)
model.longrun_checkpointed(
    steps=100, chunklength=32, dt=dt, safe_checkpoint=True
)
```

`longrun` and `longrun_checkpointed` require `chunklength`; it controls their
solver chunks without changing the requested horizon. Chunking does not bound
the entire input tape, retained recordings or autograd graph.

Standalone `Population.initialize()` does not accept a timestep argument.
The execution methods accept `dt=`; omitting it uses Dendra's global default
timestep (normally `0.005` ms). Pass `dt=` explicitly to use the intended
solver timestep consistently. A positional argument to `Population.initialize`
controls `force_rebuild`, rather than the timestep.

```{literalinclude} ../../examples/shared_execution_horizons.py
:language: python
:start-after: "# region population"
:end-before: "# endregion population"
:dedent: 4
```

## Networks

A Network uses the timestep supplied to `network.initialize(dt)`. The same
step count or duration applies to all of its Populations:

```{literalinclude} ../../examples/shared_execution_horizons.py
:language: python
:start-after: "# region network"
:end-before: "# endregion network"
:dedent: 4
```

All three Network methods require an explicit horizon. Their initialized
timestep remains fixed during these calls; use the existing initialization
lifecycle when changing it.

## Fractional duration and continuation

Duration requests share the native model's retained physical duration across
`run`, `longrun` and `longrun_checkpointed`. Only complete solver steps execute.
For example, at `dt=0.1`, two `duration=0.25` calls accept two and three steps,
respectively. The first call retains `0.05` ms for the next duration request.

An exact `steps=` request leaves that pending duration unchanged. With `0.05`
ms pending, `steps=10` advances ten steps and keeps the `0.05` ms available.
This also applies to zero-step calls. `duration=0` resolves the existing duration
budget; it does not add new physical time. Changing a standalone Population's
timestep can make previously pending time sufficient for a complete step.

Native runtime checkpoints preserve the pending duration and RNG state through
the existing `state_dict_for_checkpoint` / `restore_dict_from_checkpoint`
methods. Callback recording phase uses its separate explicit callback state.
Continue with the same prepared collection and pass `callback_state=` from the
previous result. Functional execution keeps these values in its continuation;
see {doc}`A57_functional_api_overview`.

## Inputs and registered stimulation

Explicit horizons determine the required length of deterministic input tapes:
`Inputs(ve=..., intra=...)` must contain exactly the steps accepted by that call.
For a duration request, that includes any retained fractional duration. Empty
tapes are valid when the call accepts no steps.

`Slice.inject` keeps its waveform binding semantics. Omitting `intra` from an
input packet leaves registered intracellular stimulation active:

```{literalinclude} ../../examples/shared_execution_horizons.py
:language: python
:start-after: "# region injection"
:end-before: "# endregion injection"
:dedent: 4
```

See {doc}`A61_shared_runtime_inputs` for broadcasting, dtype/device validation,
input ownership and competing stimulation arguments.

## Existing call forms

The native `tstop=` argument and positional call forms remain supported.
`tstop` requests elapsed duration, using the same cumulative duration budget.
It cannot be supplied together with `steps` or `duration`, even when the values
would describe the same horizon.

Population `run` can infer its horizon from input tapes when no explicit
horizon is supplied. Its legacy `ve=` argument keeps its authoritative tape
length when neither `steps` nor `duration` is supplied, including when a legacy
`tstop` is present. With `steps=` or `duration=`, the legacy voltage tape must
match the selected horizon; its existing dtype/device coercion remains intact.

Network calls and Population `longrun` / `longrun_checkpointed` require one of
`steps`, `duration` or `tstop`; they do not infer a horizon from input tapes.

The complete example executes here:

```{code-cell} python
:load: ../../examples/shared_execution_horizons.py
:tags: [remove-input]
```
