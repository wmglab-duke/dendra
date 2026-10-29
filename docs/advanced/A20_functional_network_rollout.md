---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-network-rollout)=
# Run and record a functional Network

An initialized standard `Network` can execute for a complete horizon through
`dn.func.make_functional(network, dt=DT)`. Rollouts accept per-population input tapes
and NetStim draw packets, support recording and reducer callbacks, and return
state for continuation. Event delivery, continuous source observation, and
Population updates follow the native ordering described in
{doc}`A19_functional_network_lowering`.

The complete
{download}`functional_network_rollout.py <../../examples/functional_network_rollout.py>`
example simulates a recurrent Network with a noisy NetStim and per-population
current tapes. It checks checkpointed state, recordings, gradients,
segmented continuation, and native continuation from a complete boundary.
Run it with `python examples/functional_network_rollout.py`.

## Declare and lower the Network

Build the Network with ordinary Population and connection methods. The example
uses two populations of single-compartment cells, a trainable continuous release
transform, threshold events, and a NetStim that shares the event target:

```{literalinclude} ../../examples/functional_network_rollout.py
:language: python
:start-after: "# region network"
:end-before: "# endregion network"
:dedent: 4
```

`Release` is an ordinary stateless `torch.nn.Module`, defined in the complete
example. Its trainable `scale` is extracted into the explicit continuous
connection parameters. As in the previous tutorial, event weights for `expsyn`
are in µS, continuous conductance weights are in S/cm², and time is in ms.
The same canonical single-device lowering requirements apply to all execution
modes. Configure training, differentiability, and the timestep before lowering.
The supplied `dt` must match the initialized Network timestep. The compatible
`make_functional_network(network, ...)` form uses that timestep implicitly.

For a fresh initialized boundary from explicit sources, call
`initial = dn.func.initialize(plan, tensors)`, where `tensors` is the
`NetworkTensors` bundle returned by lowering. Bind `initial.parameters` and
`initial.constants`, then supply `initial` to `dn.func.run`. The default recipe
is `plan.initialization_inputs`; `initialization=` accepts an explicit
`NetworkInitializationInput`. The host initializer samples missing draw packets
from explicit RNG carry, preserves native Population/cache/connection ordering
and returns no callback carry. The first run initializes its callbacks once.
The native Network is unchanged and no accepted timestep is executed. See
{ref}`functional-explicit-initialization` for the shared Population/Network
contract, pure initialization methods and a complete training example.

## Supply canonical runtime draws

`plan.sample_runtime_draws(state, steps=N)` supplies one `RuntimeDraws` packet
for all runtime stochastic requests in the Network. Pass it through `draws=`
alongside any per-Population voltage/current tapes:

```python
draws = plan.sample_runtime_draws(initial, steps=STEPS)
final, auxiliary = dn.func.run(
    bound, initial, steps=STEPS, draws=draws, callbacks=callbacks,
)
```

The packet contains named draw families and one authoritative RNG timeline
per physical stream. It includes NetStim, model NOISE/SDE, and randomized
Poisson stimulation requests while preserving native ordering and shared
stream boundaries. Host execution with `draws=None` samples bounded packets
automatically; pure execution consumes explicit stochastic inputs. See
{ref}`functional-runtime-draws` and the complete
{download}`functional_runtime_draws.py <../../examples/functional_runtime_draws.py>`
example for replay, gradients, checkpointing, and continuation.

## Per-population tapes

Use the same deterministic `Inputs` packet as Population execution:

```python
inputs = {
    "left": dn.func.Inputs(intra=left_current),
    "right": dn.func.Inputs(intra=right_current),
}
final, auxiliary = dn.func.run(
    bound, initial, inputs=inputs, steps=STEPS, draws=runtime_draws,
)
```

Pure Network steps and eager bindings accept the same mapping with one input frame per Population;
rollouts, chunks and host runs consume time-leading tapes. Stochastic requests
use `draws=` or host automatic sampling separately from deterministic fields.
See {ref}`functional-runtime-inputs` for the shared contract and a runnable
Population/Network example.

## Compatibility NetStim packets

The existing `NetworkRolloutInput` retains two optional fields. `populations` maps the Network's
Population names to `RolloutInput(ve=..., intra=...)`; `netstim` supplies explicit
NetStim draws with their matching post-draw RNG bytes. Each supplied tensor has
one leading time axis:

```{literalinclude} ../../examples/functional_network_rollout.py
:language: python
:start-after: "# region inputs"
:end-before: "# endregion inputs"
:dedent: 4
```

The remaining dimensions follow each Population's functional input contract.
`intra` is the assembled absolute intracellular current in mA; `ve` is the
external potential in mV for integrators that accept it. Omitted Population
entries and fields retain their normal registered stimulation or default
behavior. The two populations can receive different shapes and different
drives; their tapes share the same horizon. Input tensors retain autograd
history, including across slices and checkpoint replay.

All supplied tapes must contain exactly `steps` samples. The horizon is inferred
from a tape when `steps` is omitted; supply `steps=...` when there are no tapes.
Zero steps is allowed for eager rollout and the host runner.

`plan.sample_draw_tape(state, steps)` is a host operation. It samples sequentially
from an independent generator restored from the explicit NetStim carry. It
returns `NetStimDraw(exp_rand, rng_state)` with a leading time axis on both
fields, and mutates neither the input carry, the native NetStim, nor the global
PyTorch generator. Keep both fields together when slicing or saving the tape.
Without a NetStim it returns `None`.

Pure `rollout`, `prepare_and_rollout`, and direct compiled chunks require an
explicit draw tape when a NetStim is present and the horizon is nonempty.
`dn.func.run(bound, state, ...)` can omit that tape: the host samples bounded
packets before each execution span and supplies the same packets again during checkpoint
backward. Its returned state contains the exact next RNG position. Sampling
does not belong inside a compiled timestep. See
{doc}`A14_functional_netstim_renewal` for draw semantics.

## Bind recordings and pure reducers

Bind functional callback configurations to the Network plan:

```{literalinclude} ../../examples/functional_network_rollout.py
:language: python
:start-after: "# region callbacks"
:end-before: "# endregion callbacks"
:dedent: 4
```

Recorder names begin with the Population name: `left.v` and
`right.mech.event.g` select explicit transition state. `t` selects the
authoritative Network clock. Explicit paths such as
`populations.right.integrator.v` are also accepted. A new callback collection
records the initial frame and then every accepted timestep, giving `steps + 1`
samples. The results appear under `aux["callbacks"]["trace"]` and preserve
gradients. `node_indices`, `max_only`, and `partition` follow the ordinary
functional Recorder contract.

Use `region=network.left.soma` with qualified physical fields such as `left.v`
to record a labeled Population Slice. For a concatenated Network, select
`all_populations.v` with `region=network.all_populations.exc.soma`. Labels,
batch selection, supported fields, and a runnable scan/while example are
described in {doc}`A26_functional_network_regions`.

Use `FunctionalCallback` to define a pure reducer. A Network callback receives
the full Network state:

```{literalinclude} ../../examples/functional_network_rollout.py
:language: python
:start-after: "# region reducer"
:end-before: "# endregion reducer"
```

`initialize` returns carry and an optional initial emission, `update` returns
next carry and an optional emission, and `finalize` returns the public result.
Keep schemas stable and implementations deterministic and side-effect-free;
use explicit tensor carry for evolving values. The
{doc}`A8_functional_populations` callback section explains the complete contract.
Use `PopulationCallback("left", callback)` to apply LFP, threshold-analysis,
anomaly, or local Recorder callbacks to one Population. Scoped built-ins support
scan and while execution; custom callbacks support scan. Lead-field shapes,
callback windows, packed components, and a runnable example are described in
{doc}`A27_functional_network_callbacks`.
Mutable native callbacks are not run by the functional Network.

## Choose eager rollout or compiled chunks

Use eager rollout to execute a horizon without compilation. Prepare the
coefficients before running the loop:

```{literalinclude} ../../examples/functional_network_rollout.py
:language: python
:start-after: "# region eager"
:end-before: "# endregion eager"
:dedent: 4
```

`plan.prepare_and_rollout(parameters, constants, state, inputs, ...)` combines
preparation and eager rollout in one call. Both return `(final_state, aux)` and
leave their inputs and source Network unchanged. Recompute preparation after
changing parameters or constants.

For repeated fixed horizons, construct a compiled chunk:

```python
chunk = plan.compile_rollout_chunk(3, execution="scan", backend="aot_eager")
state3, aux3 = chunk(
    parameters, prepared, initial, slice_inputs(inputs, 0, 3),
    callbacks=callbacks,
)
```

The complete example's `run_demo(compiled=True)` selects this mode. A scan has
a bounded loop graph rather than one graph expanded over the entire horizon.
`execution="unrolled"` compiles the ordinary expanded recurrence instead.
The chunk width must be positive; each call supplies exactly that many input
samples. Prepare coefficients before calling the chunk, and call it directly.

For a long horizon, use the host runner with optional checkpointing:

```{literalinclude} ../../examples/functional_network_rollout.py
:language: python
:start-after: "# region run"
:end-before: "# endregion run"
:dedent: 4
```

Here `chunk` is `None` in the default eager example, or a compiled chunk when
`run_demo(compiled=True)` is selected. Both create an explicit `BoundNetwork`
for the same `dn.func.run` scheduling contract used by Populations. Its parameter
and constant mappings snapshot the container structure while retaining the
original tensor leaves and their gradients. Replacing a mapping entry requires
a new binding. The binding holds raw constants; the host derives prepared
materializations at their execution boundaries instead of retaining a prepared
autograd graph across calls.

Direct eager bindings and compiled chunks accept `callbacks=callbacks` and
`callback_state=previous_aux["callbacks"].state` as well. A direct compiled
call consumes exactly its fixed width, and all Population input packets are
time-leading tapes. A direct eager call consumes one frame per Population when
given a mapping of `dn.func.Inputs`; the legacy `NetworkRolloutInput` form keeps
its one-step time-leading convention. The result is the same `(state, aux)`
pair used by host execution. The complete examples in
{doc}`A57_functional_api_overview` compare these interfaces and their ownership
rules.

`checkpoint_every` chooses host checkpoint spans, including automatic NetStim
draw-tape spans. Use `host_span_steps` to bound ordinary execution spans without
checkpointing. These options are mutually exclusive and independent of the
compiled kernel width; neither truncates gradients. The runner executes exact
shorter tails when the spans and compiled width do not divide the horizon.
Checkpoint backward replays the same explicit inputs and NetStim packets.
Keep parameter, constant, input-tape, and source configuration values unchanged
until backward completes.

With no explicit `steps` or `duration`, supplied tapes determine the horizon.
`steps=N` executes an exact count, including with no tapes. `duration=...`
advances a physical duration from the current state and retains incomplete
timesteps in `state["clock"]["duration_remainder"]`. Steps and duration are
mutually exclusive, and supplied tapes must match the resolved count exactly.
The binding uses the Network's fixed timestep. A host span defaults to 64 steps
for Network execution; a binding does not change that timestep or its compiled
kernel width.

The existing `plan.run(parameters, constants, state, ..., chunklength=...)`
entry point remains a compatibility adapter with its original arguments and
defaults. It schedules the same execution machinery. Explicit bindings and
`dn.func.run` provide the common scheduling interface for new code.

Scan supports ordinary first-order `backward()` and `autograd.grad` on the
supported PyTorch compiler interfaces; `create_graph=True` raises explicitly.
Use eager `step` or `rollout` for higher derivatives. Calls inside `torch.func`
transforms or direct forward AD use the eager recurrence and retain transform
semantics. AOT-compiled unrolled chunks also target first-order backward.
Recording storage still grows with the selected states and sample count;
checkpointing reduces saved transition activations, not the recording result.

## Continue both simulation and callback state

For `dn.compile(Network)`, each successful call returns an in-memory
`dn.func.Continuation` at `aux["continuation"]`:

```python
state, aux = program()
state, aux = program(aux["continuation"])
```

It carries complete simulation state, the effective raw parameters/constants
selected for that call, and callback carry. This includes per-call overrides
and fresh/cache initialization replacements. Container structure is snapshotted;
Tensor identities, shared aliases and gradients remain live. A packet supplies
all four operands, so do not combine it with non-`None` parameter, constant,
callback-state or initialization overrides. Supply input/draw tapes and runtime
extracellular recipes separately on each invocation. See
{doc}`A57_functional_api_overview` and the complete
{download}`functional_continuation.py <../../examples/functional_continuation.py>`
example for Population and Network workflows.

Raw Network bindings use the same packet through `dn.func.run`. Every successful
raw-bound host run returns `aux["continuation"]`, including a run started from
an ordinary state. To use a convenience runner's packet through its kernel:

```python
continuation = aux["continuation"]
bound = program.kernel.bind(
    continuation.parameters, constants=continuation.constants,
)
state, aux = dn.func.run(
    bound, continuation, steps=STEPS, callbacks=program.callbacks,
)
state, aux = dn.func.run(
    bound, aux["continuation"], steps=STEPS, callbacks=program.callbacks,
)
```

The binding's sources remain authoritative. A packet must have matching source
schemas, Tensor identities and aliases; the scheduler does not adopt different
sources. A restored checkpoint owns new Tensor leaves and requires an explicit
new binding. Network shared constant readers follow their canonical parameter
source without mutating the binding's authored maps. Supply the same bound
callback collection with the packet's carry, and do not also supply a
non-`None` `callback_state=`. Input/draw tapes, runtime extracellular recipes
and the horizon remain per-call arguments. See
{ref}`functional-bound-continuation` for the shared Population/Network contract.

Direct bindings and numerical chunks retain their explicit state/callback
arguments. Prepared Population bindings also use that explicit form because
they do not own raw constants.

A returned Network state includes all Population carry, delivery queues,
NetStim clock/RNG/calendar state, and the authoritative Network clock. The
explicit-state form remains supported: preserve
`aux["callbacks"].state` alongside it when continuing the same callback
collection:

```{literalinclude} ../../examples/functional_network_rollout.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 4
```

The suffix emits only its new post-step frames. Concatenating the prefix and
suffix recordings therefore matches one uninterrupted rollout, without
duplicating the boundary sample. Reducer carry continues across the boundary;
each returned recording contains that call's samples. Treat callback state as
an opaque carry object bound to the same callback collection. Passing no
callback state starts a fresh collection at the supplied Network boundary.

The example clones carry without detaching it, preserving gradients across the
prefix. Save complete detached continuation with
`dn.func.export_continuation(plan, aux["continuation"], callbacks=callbacks)`.
Restore the plain data/Tensor packet against a compatible receiving lowered plan
with `dn.func.restore_continuation(receiving, saved, callbacks=receiving_callbacks)`,
then bind its restored parameter/constant leaves before resuming with `dn.func.run`.
Each callback collection belongs to its corresponding plan. This retains
effective sources, raw parameter/constant aliases, trainability and callback
phase. Runtime tapes, model configuration, optimizer state and prior recordings
remain separate. Convenience runners provide the same portable format through
`program.export_continuation(...)` and `receiver.restore_continuation(...)`;
their bound callback collection is available as `runner.callbacks`.
Independently saved analysis can use `callbacks.export_state(...)` and
`callbacks.restore_state(...)` directly. The portable bridge validates callback settings and
source/state schemas; see
{doc}`A58_functional_callback_checkpoints` for complete Population and Network
examples. Native continuation uses `plan.commit_state_(matching_network,
state)` as described in {doc}`A19_functional_network_lowering`; commit transfers
state rather than parameters or constants.

## Differentiate the complete run

An objective can use final state, recorded samples, or pure reducer results.
The example checks checkpointed gradients against eager execution for event
weights, continuous weights, a trainable release parameter, NetStim noise, and
the intracellular input tape:

```{literalinclude} ../../examples/functional_network_rollout.py
:language: python
:start-after: "# region gradients"
:end-before: "# endregion gradients"
:dedent: 4
```

Hard event values and surrogate gradients retain the Network's configured
semantics. The sampled draw tape is explicit randomness; differentiation uses
the supplied draws and differentiates the renewal parameters, rather than the
sampling operation.

The complete CPU example executes here:

```{code-cell} python
:load: ../../examples/functional_network_rollout.py
:tags: [remove-input]
```

For `dn.compile(Network)` convenience execution with snapshot or fresh
initialization, continue with {doc}`A21_functional_network_compile`.
Bind per-population extracellular contacts and Waveforms with
`make_functional(network, dt=DT, extra={name: specification})`; see
{doc}`A28_functional_network_extra` for trainable leaves and mixed explicit inputs.
Inference-only captured while loops, recording, and bounded tails use the same
pure Network timestep; see {doc}`A25_functional_network_while`.
Source-history training and its delivery-time parameter semantics are described
in {doc}`A22_functional_source_history`. Intrinsic packed-history inference is
described in {doc}`A23_functional_bitpacked_inference`. Sparse-calendar inference
and packed NetCon scheduled traffic use {doc}`A24_functional_pending_calendar`.
Heterogeneous-device functional execution and continuous delay interpolation are
currently unsupported.
