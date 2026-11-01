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

(shared-runtime-inputs)=
# Deterministic inputs for native and functional execution

`dn.func.Inputs(ve=..., intra=...)` supplies deterministic extracellular voltage
and intracellular current to native Populations and Networks, functional
bindings and `dn.compile` runners. The same packet keeps its Tensor references,
views and gradients when it crosses an execution boundary.

| Field | Quantity | Units |
| --- | --- | --- |
| `ve` | Extracellular voltage at each compartment | mV |
| `intra` | Total injected current at each compartment | mA |

Both fields are optional. A native `step` consumes one frame; `run`, `longrun`
and `longrun_checkpointed` consume time-leading tapes. The dimensions after
time use the Population's existing broadcasting rules: ordinary trailing
broadcasting first, then the supported right-aligned batch-only fallback.
Explicit singleton axes distinguish batch and spatial dimensions.

The complete runnable
{download}`shared_runtime_inputs.py <../../examples/shared_runtime_inputs.py>`
example compares native runs, one-step helpers and functional values and
input gradients:

```{literalinclude} ../../examples/shared_runtime_inputs.py
:language: python
:start-after: "# region population"
:end-before: "# endregion population"
:dedent: 4
```

## Native Population calls

Pass `inputs=` to the ordinary native execution method:

```python
model.step(dt, inputs=dn.func.Inputs(ve=voltage_frame, intra=current_frame))
model.run(dt=dt, inputs=dn.func.Inputs(ve=voltage_tape, intra=current_tape))
model.longrun(duration=duration, chunklength=100, dt=dt, inputs=inputs)
model.longrun_checkpointed(
    steps=100, chunklength=100, dt=dt, inputs=inputs, safe_checkpoint=True
)
```

A Population `run` can infer the number of steps from its tapes when no horizon
is supplied. Explicit `steps=` or `duration=` requires every tape to contain
exactly the accepted step count. The same applies to `tstop` when legacy `ve=`
is absent. Duration requests include retained fractional duration from previous
calls. All supplied tapes must have the same
time length. Empty tapes represent a zero-step call when the accepted duration
contains no complete step.

`longrun` and `longrun_checkpointed` require a horizon and `chunklength`.
Use `steps=`, `duration=` or the existing `tstop` argument; see
{doc}`A62_shared_execution_horizons` for their timing contract.
Chunking divides solver execution and checkpoint replay; it
does not bound the whole input tape, recorded history or autograd graph.
Broadcast normalization can materialize a contiguous full-horizon tape.
Partition calls explicitly when supplying inputs in bounded streaming spans.

## Native Network calls

A Network accepts a mapping from its Population names to the same packets.
Omitted Populations and omitted fields keep their normal drive behavior:

```{literalinclude} ../../examples/shared_runtime_inputs.py
:language: python
:start-after: "# region network"
:end-before: "# endregion network"
:dedent: 4
```

Native Network `run`, `longrun` and `longrun_checkpointed` require a horizon
through `steps=`, `duration=` or `tstop`. Each supplied tape must match that
call's accepted step count.
`network.step(inputs={"left": dn.func.Inputs(intra=current_frame)})` consumes
frames without a separate duration. The Network's initialized timestep applies
to every Population.

## One-step helpers

`dn.step`, `dn.step_population` and `dn.step_network` accept the same `inputs=`
frames as the corresponding native method:

```python
frame = dn.func.Inputs(ve=voltage_frame, intra=current_frame)
dn.step(model, dt=dt, inputs=frame)
dn.step_population(model, dt=dt, inputs=frame)

frames = {"left": dn.func.Inputs(intra=current_frame)}
dn.step(network, inputs=frames)
dn.step_network(network, inputs=frames)
```

Each call advances one step. Helpers forward the packet unchanged to the native
method, preserving its Tensor references, views, gradients, validation and
stimulation-conflict rules. They accept the same prepared callback collection
and explicit `callback_state=` carry. Supply a Population's timestep through
`dt=`; a Network uses its initialized timestep. The runnable example steps
independent models with these helpers and compares their values and input
gradients against native runs and functional execution.

## Waveforms and existing stimulation arguments

`Slice.inject` remains the waveform binding API. Supplying `Inputs(ve=...)`
leaves its intracellular waveform active:

```{literalinclude} ../../examples/shared_runtime_inputs.py
:language: python
:start-after: "# region injection"
:end-before: "# endregion injection"
:dedent: 4
```

Explicit `intra` cannot be combined with solver-level intracellular stimulation
registered on that Population. Mechanism-owned waveform injections keep their
own current channels and can coexist with explicit `intra`.

Explicit `ve` conflicts with legacy `ve=` or extracellular `extra=` for the
same Population. `Inputs(intra=...)` can be combined with `extra=` because
they supply different drive quantities. Network conflicts are checked per
Population, so another Population can continue using its own `extra` binding.
Conflicts and malformed packets reject before waveform evaluation and random
sampling.

The legacy entry points retain their existing behavior. In particular,
Population `run(ve=...)` uses the voltage tape's time length as authoritative
when neither `steps` nor `duration` is supplied, even when `tstop` is present.
This also applies when legacy `ve=` is
combined with `Inputs(intra=...)`: the current tape must match the voltage
tape's length, and `tstop` is ignored. An explicit `steps=` or `duration=`
checks both legacy and canonical tapes against the selected horizon.

## Tensor ownership and validation

Canonical fields must be Tensors on the model's device and with its dtype.
The packet does not implicitly convert lists, change dtype or move devices.
Convert explicitly when needed; ordinary PyTorch conversion retains the
appropriate autograd connection. Legacy `ve=` and `extra=` retain their
existing coercion policies.

Input leaves remain caller-owned. Gradients propagate through broadcast,
chunking and checkpoint replay. Keep leaves unchanged until backward has
finished; mutating a saved input can invalidate checkpoint recomputation.
Runtime state and the physical duration remainder continue through the native
model's existing checkpoint methods. Supply the remaining input tape again
when resuming.

## Explicit runtime draws for native steps

`Inputs` contains deterministic drives. Supply stochastic values separately
through `draws=` on native Population/Network `step`, `dn.step`,
`dn.step_population` and `dn.step_network`. These calls consume one
`dn.func.RuntimeDraws` packet for NOISE, State SDE and optional Network NetStim.
Native `run`, `longrun` and `longrun_checkpointed` do not accept draw tapes.
Population steps also replay randomized Poisson waveforms in solver-owned
intracellular injections and call-local extracellular `extra` banks.
Mechanism-owned Poisson and native Network Poisson replay are unsupported.
Omitting `draws` retains the native sampling lifecycle.

Use the existing functional plan's sampler. Lower the initialized native model
once, capture its current boundary, then sample without advancing native state
or its generators:

```python
functional, _ = dn.func.make_functional(model, dt=dt)
boundary = dn.func.capture_continuation(functional, model)
draws = functional.sample_runtime_draws(boundary.state)
dn.step(model, dt=dt, inputs=frame, draws=draws)
```

A Network uses its initialized timestep and a Population-name mapping of input
frames. Existing plans can be reused while the model remains compatible;
`program.functional` also exposes a compiled runner's plan. Packet families,
shapes, dtype/device and stream routes follow that plan. Keep the sampled packet
intact rather than constructing its draw or RNG fields manually.

Explicit native execution uses detached base draws and publishes their
authoritative RNG frames at the normal execution boundaries. Shared Network
streams retain their Population ordering. Later native steps with omitted
`draws` continue from the published suffix. Model parameters and deterministic
inputs retain their normal gradients; draw values and RNG carry remain
detached. Recordings and callback carry use the same native step contract.

Poisson replay consumes each waveform invocation's supplied uniforms in its
prepared order, including repeated shared children and custom containers with
fixed invocation counts. Recipes require finite `n` and the existing functional
fixed-count envelope. Schedules and uniforms are detached; waveform amplitude,
container gain and extracellular field gradients retain their normal paths.
Shared waveform and generator identities remain intact, and each waveform keeps
its final invoked schedule for native checkpoints and later execution.
External `extra` waveforms and generators remain caller-owned: save their state
alongside the model when checkpointing, and supply the same configuration again.

Explicit SDE draws pass supplied tensors through native compiled kernels.
Omitting draws retains the native RNG/backend restrictions: automatic State SDE
sampling can fail under strict full-graph capture of `torch.Generator`, even
when explicit-draw steps compile successfully.

The complete
{download}`shared_runtime_draws.py <../../examples/shared_runtime_draws.py>`
example compares explicit native steps and helpers with functional references,
checks parameter gradients and recordings, then resumes ordinary native sampling:

```console
python examples/shared_runtime_draws.py
```

The complete
{download}`shared_poisson_replay.py <../../examples/shared_poisson_replay.py>`
example changes sampled uniforms, compares native and functional Population
steps for intra/extra banks, checks amplitude/field gradients and resumes
automatic sampling from a native checkpoint:

```console
python examples/shared_poisson_replay.py
```

See {doc}`A57_functional_api_overview` for functional draw tapes and host sampling.

The complete example executes here:

```{code-cell} python
:load: ../../examples/shared_runtime_inputs.py
:tags: [remove-input]
```
