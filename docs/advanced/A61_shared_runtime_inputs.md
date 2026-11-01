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

## Explicit runtime draws

`Inputs` contains deterministic drives. Supply stochastic values separately
through `draws=` on native Population/Network `step`, `dn.step`,
`dn.step_population` and `dn.step_network`. These calls consume one
`dn.func.RuntimeDraws` packet for NOISE, State SDE and optional Network NetStim.
Native `run`, `longrun` and `longrun_checkpointed` consume a time-leading
`RuntimeDraws` tape. Population and Network execution also replays randomized
Poisson waveforms in mechanism-owned injections, solver-owned intracellular
injections and call-local extracellular `extra` banks.
Omitting `draws` retains the native sampling lifecycle.

Sample directly from an initialized native Population without advancing its
state or generators:

```python
draws = dn.func.sample_runtime_draws(model, dt=dt)
dn.step(model, dt=dt, inputs=frame, draws=draws)
```

A Network uses its initialized timestep and a Population-name mapping of input
frames:

```python
draws = dn.func.sample_runtime_draws(network)
network.step(inputs={"left": dn.func.Inputs(intra=current_frame)}, draws=draws)
```

The sampler reads the model's current logical RNG boundary, including a deferred
Poisson boundary restored from a checkpoint. It leaves model tensors, generator
identities and both live and global RNG state unchanged. Native sampling supports
initialized canonical single-device Populations and Networks; it does not build
or initialize an uninitialized model.

The same function accepts a lowered plan, binding, compiled chunk or runner with
an explicit state mapping or `Continuation`:

```python
draws = dn.func.sample_runtime_draws(functional, boundary, steps=100)
next_draws = dn.func.sample_runtime_draws(program, continuation, steps=100)
```

State is required for those targets: omitted state does not select a runner's
snapshot or fresh-initialization policy. Their timestep and extracellular
bindings are fixed by the lowered plan. Existing
`functional.sample_runtime_draws(state, steps=...)` methods remain available.
Packet families, shapes, dtype/device and stream routes follow the model. Keep
the sampled packet intact rather than constructing its fields manually.

### Run-family tapes

Supply `steps=N` to the same sampler for a time-leading tape:

```python
draws = dn.func.sample_runtime_draws(population, dt=dt, steps=100)
population.longrun(
    steps=100,
    chunklength=32,
    dt=dt,
    inputs=inputs,
    draws=draws,
)
```

A Network accepts the same argument using its initialized timestep; omit `dt`
from the native call and key deterministic `inputs` by Population name. `run`
also accepts these tapes. Use `longrun_checkpointed(..., draws=draws)` for
activation-checkpointed training. The packet has one row for every accepted
solver step; its leading length must match the horizon resolved from
`steps`, `duration`, `tstop` or existing input-tape rules. Draw tapes alone do
not select a horizon. Zero-step replay advances no live simulation state or RNG
boundary.

The shared sampler uses a runtime-only Poisson invocation layout, independent of
how often an initializer evaluates its waveforms. Existing sampler methods and
initialization packets retain their formats. Both native and functional runtime
execution accept the runtime layout and compatible reserved-row layouts; all
supplied rows are validated before adaptation. Sampling under
`torch.inference_mode()` returns detached, versioned tensors that can also be
used as checkpointed-training operands.

The complete
{download}`shared_runtime_draw_sampling.py <../../examples/shared_runtime_draw_sampling.py>`
example samples directly from a Population and a Network containing NOISE, a
State SDE, repeated Poisson injections and optional NetStim input. It replays a
prefix and a checkpointed suffix while retaining waveform gradients:

```{literalinclude} ../../examples/shared_runtime_draw_sampling.py
:language: python
:start-after: "# region sample"
:end-before: "# endregion sample"
:dedent: 4
```

After restoring the prefix checkpoint, sampling reads its retained RNG boundary.
The same remaining packet can drive either native or functional execution:

```{literalinclude} ../../examples/shared_runtime_draw_sampling.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 4
```

Explicit Poisson replay evaluates one complete invocation recipe per accepted
step. A Population containing replayed Poisson banks evaluates its solver-level
intracellular and extracellular banks per accepted step. Populations without
these banks retain native whole-run/per-chunk waveform evaluation, including
when another Population in the same Network replays Poisson inputs. The supplied
random values and Poisson recipes are independent of `chunklength`; custom
deterministic waveforms retain their existing time-grid semantics. Calls with
omitted `draws` retain the native automatic sampling lifecycle, including the
existing whole-run or per-chunk preparation of solver-level waveform banks.

Mechanism-owned injections consume explicit draws when their owning hooks
evaluate them. Repeated calls, custom pure waveform containers and solver-stage
evaluations use the same fixed invocation profiles as functional execution;
registered Poisson children that are not called consume no random values.
Native Network execution preserves the order of extracellular banks,
intracellular banks and per-Population mechanism evaluations, including when
those phases share waveform objects or generators. This applies to runtime
execution from an initialized model; `draws=` does not replay native fresh
initialization.

Validation checks the entire supplied tape before executing its first frame,
including later RNG frames and Poisson recipes. Draw tensors remain
caller-owned. Keep them unchanged until backward finishes: checkpoint
recomputation reads the same frames and rejects mutated leaves. With the default
`restore_state_after_backward=True`, recomputing a checkpointed chunk restores
the live model, caller-owned Poisson schedule history and generator positions
afterward. Model parameters, waveform
parameters and deterministic input gradients retain their usual paths; supplied
base draws remain detached.

Native checkpointed training requires tape tensors with mutation counters.
Tensors created under `torch.inference_mode()` must be cloned outside that
context before the call. Ordinary replay and inference execution accept those
tensors directly; an empty tape does not require checkpoint operands.

When resuming a native checkpoint, supply the remaining draw and input slices
again. Save call-local extracellular waveform state separately when needed;
those objects remain caller-owned. A later native step with omitted draws
continues from the final replayed RNG boundary.

The complete
{download}`shared_runtime_draw_tapes.py <../../examples/shared_runtime_draw_tapes.py>`
example samples one tape, runs a native prefix and checkpointed suffix, restores
the prefix into a compatible receiver, and continues with a different chunk
size. It checks callback phase and training gradients for both Populations and
Networks:

```{literalinclude} ../../examples/shared_runtime_draw_tapes.py
:language: python
:start-after: "# region prefix"
:end-before: "# endregion prefix"
:dedent: 4
```

```console
python examples/shared_runtime_draw_tapes.py
```

The complete
{download}`shared_mechanism_poisson_replay.py <../../examples/shared_mechanism_poisson_replay.py>`
example defines a mechanism injection hook, attaches a repeated randomized
Poisson waveform through `Slice.inject`, and checks native/functional parity,
waveform gradients and checkpoint continuation for a Population and a Network:

```{literalinclude} ../../examples/shared_mechanism_poisson_replay.py
:language: python
:start-after: "# region replay"
:end-before: "# endregion replay"
:dedent: 4
```

### RNG publication and Poisson ownership

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
When resuming through `dn.func.commit_continuation_`, compatible model/extra
Parameter and generator aliases are preserved. A shared pending global RNG
boundary is consumed by its first native waveform invocation, including an
extracellular invocation before intracellular stimulation. Independent external
sources retain their own state; see {doc}`A59_functional_native_handoff`.

Network replay validates the complete packet before evaluating live banks. It
consumes extracellular banks in the supplied `extra` mapping order, then
intracellular banks in Population order, before the Network's numerical phases.
One invocation cursor spans all banks and subsequent mechanism evaluations,
including shared physical waveform children. A generator shared by several
Populations has one authoritative RNG
timeline, including when a waveform borrows a declared RAND/NOISE/SDE stream
owned by another Population. A later automatic step starts from the published
endpoint. Supply compatible bank bindings in the same order used for sampling:

```python
extra = {"right": (right_field, right_waveform), "left": (left_field, left_waveform)}
draws = dn.func.sample_runtime_draws(network, extra=extra)
network.step(extra=extra, draws=draws)
# dn.step(network, extra=extra, draws=draws) and
# dn.step_network(network, extra=extra, draws=draws) use the same contract.
```

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

The complete
{download}`shared_network_poisson_replay.py <../../examples/shared_network_poisson_replay.py>`
example replays shared waveform children across two Populations, uses reversed
extracellular mapping order, checks amplitude/field gradients and resumes
automatic sampling after native checkpoint restoration:

```console
python examples/shared_network_poisson_replay.py
```

See {doc}`A57_functional_api_overview` for functional draw tapes and host sampling.

The complete example executes here:

```{code-cell} python
:load: ../../examples/shared_runtime_inputs.py
:tags: [remove-input]
```
