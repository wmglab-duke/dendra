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

(shared-step-resolution)=
## Resolve the input tape length

Use `dn.func.resolve_steps` to obtain the accepted timestep count before
allocating a deterministic input tape. It returns a Python integer and uses
the same timestep and retained fractional duration as execution:

```python
request = {"duration": 10.0}
steps = dn.func.resolve_steps(program, trial, **request)
population = program.functional
current = torch.zeros(
    (steps, *population.shape),
    dtype=population.dtype,
    device=population.device,
)
inputs = dn.func.Inputs(intra=current)
state, auxiliary = dn.func.run(program, trial, inputs=inputs, **request)
```

This example uses a Population runner. A Network uses the same resolved count
for each Population's tape, with its own shape, dtype and device. The complete
example below constructs both forms.

The query requires one explicit `steps=` or `duration=` request; the two are
mutually exclusive. It does not infer a runner's configured default horizon or
a chunk's width. `steps=0` returns zero, and exact step requests do not spend
pending duration. A duration query includes the incoming fractional time, so
`duration=0` can accept steps if that boundary already contains a complete
timestep. A result of zero is suitable for an empty time-leading input tape.

Initialized native Populations and Networks supply their current boundary.
For a Population, pass the same `dt` used by execution or omit it to use the
native backend default. A Network uses its initialized timestep. Functional
plans, bindings, compiled chunks and runners require explicit state or a
`Continuation`; their effective timestep sources follow
{ref}`shared-runtime-draw-horizons`. Native queries require canonical duration
budgeting but do not require stochastic draw-sampler support.

Resolution does not advance the clock, commit a remainder, sample RNG,
evaluate mechanisms or waveforms, lower a model, prepare workspaces, or refresh
caches. Keep the original request on execution. Replacing `duration=T` with
`steps=resolved_steps` would execute the same number of numerical steps but
leave the old fractional remainder in place.

The complete
{download}`shared_step_resolution.py <../../examples/shared_step_resolution.py>`
example allocates Population and Network inputs from the resolved count,
compares gradients with native checkpointed execution, and restores native and
portable functional checkpoints before resolving another duration:

```{literalinclude} ../../examples/shared_step_resolution.py
:language: python
:start-after: "# region resolution"
:end-before: "# endregion resolution"
:dedent: 4
```

```console
python examples/shared_step_resolution.py
```

(shared-runtime-draw-horizons)=
## Matching stochastic draw tapes to a duration

`dn.func.sample_runtime_draws` accepts the same `steps=` or `duration=` horizon
for initialized native models and explicit functional boundaries. Sampling
uses isolated generators and does not advance the source model, continuation,
fractional duration or RNG. Pass the same duration to sampling and execution:

```python
draws = dn.func.sample_runtime_draws(program, trial, duration=10.0)
state, auxiliary = dn.func.run(program, trial, duration=10.0, draws=draws)

# Native Population execution uses the same explicit timestep for both calls.
draws = dn.func.sample_runtime_draws(population, duration=10.0, dt=dt)
population.longrun(duration=10.0, dt=dt, chunklength=32, draws=draws)
```

For a native Network, omit `dt`; sampling and execution use its initialized
timestep. Functional plans, bindings, chunks and runners require an explicit
state or `Continuation`. A Population continuation or binding selects its
effective timestep sources. Unbound plans and chunks with plain state use
their nominal timestep; a runner with plain state uses its default sources.

At `dt=0.1`, a boundary with `0.05` ms pending accepts three steps for
`duration=0.25`. Dividing just the new duration by `dt` would produce the wrong
tape length. Sampling leaves the `0.05` ms available; replaying the duration
spends it exactly once and publishes the new remainder. Native and functional
checkpoints preserve that incoming boundary and reproduce the same tape.

Omitting both horizon arguments keeps the one-frame sampling contract.
Supplying `steps=` or `duration=` produces a time-leading tape, including a
fresh empty tape when the resolved count is zero. `duration=0` includes pending
time without adding to it, so a changed standalone Population timestep can
make that request accept complete steps. Explicit `steps=0` retains pending
time. Supplying both `steps` and `duration` raises an error.

The complete
{download}`shared_runtime_draw_horizons.py <../../examples/shared_runtime_draw_horizons.py>`
example covers Population and Network models with NOISE/SDE draws and Network
NetStim. A substep duration establishes pending time, then duration sampling,
replay and checkpoint restoration preserve that budget and training gradients:

```{literalinclude} ../../examples/shared_runtime_draw_horizons.py
:language: python
:start-after: "# region duration"
:end-before: "# endregion duration"
:dedent: 4
```

```console
python examples/shared_runtime_draw_horizons.py
```

## Inputs and registered stimulation

Explicit horizons determine the required length of deterministic input tapes:
`Inputs(ve=..., intra=...)` must contain exactly the steps accepted by that call.
For a duration request, that includes any retained fractional duration. Empty
tapes are valid when the call accepts no steps. Use
`dn.func.resolve_steps(target, ..., duration=T)` to obtain that length without
advancing the incoming boundary; see {ref}`shared-step-resolution`.

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
