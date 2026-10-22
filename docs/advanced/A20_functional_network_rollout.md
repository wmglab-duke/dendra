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
`dn.func.make_functional_network`. Rollouts accept per-population input tapes
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

## Supply per-population tapes and NetStim packets

`NetworkRolloutInput` has two optional fields. `populations` maps the Network's
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
`intra` is the assembled intracellular current density in mA/cm²; `ve` is the
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
`plan.run` can instead omit that tape: the host samples bounded packets before
each execution span and supplies the same packets again during checkpoint
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
`run_demo(compiled=True)` is selected. `chunklength` bounds execution chunks.
`checkpoint_every` independently chooses host checkpoint spans, including the
automatic NetStim draw-tape spans; neither option truncates gradients. The runner
executes exact shorter tails when execution, checkpoint, and compiled widths do
not divide the horizon.
Checkpoint backward replays the same explicit inputs and NetStim packets.
Keep parameter, constant, input-tape, and source configuration values unchanged
until backward completes.

Scan supports ordinary first-order `backward()` and `autograd.grad` on the
supported PyTorch compiler interfaces; `create_graph=True` raises explicitly.
Use eager `step` or `rollout` for higher derivatives. Calls inside `torch.func`
transforms or direct forward AD use the eager recurrence and retain transform
semantics. AOT-compiled unrolled chunks also target first-order backward.
Recording storage still grows with the selected states and sample count;
checkpointing reduces saved transition activations, not the recording result.

## Continue both simulation and callback state

A returned Network state includes all Population carry, delivery queues,
NetStim clock/RNG/calendar state, and the authoritative Network clock. Preserve
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
prefix. For durable independent checkpoints, save detached copies together
with matching parameters, constants, remaining inputs, and Network
configuration. Native continuation uses `plan.commit_state_(matching_network,
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
`make_functional_network(network, extra={name: specification})`; see
{doc}`A28_functional_network_extra` for trainable leaves and mixed explicit inputs.
Inference-only captured while loops, recording, and bounded tails use the same
pure Network timestep; see {doc}`A25_functional_network_while`.
Source-history training and its delivery-time parameter semantics are described
in {doc}`A22_functional_source_history`. Intrinsic packed-history inference is
described in {doc}`A23_functional_bitpacked_inference`. Sparse-calendar inference
and packed NetCon scheduled traffic use {doc}`A24_functional_pending_calendar`.
Heterogeneous-device functional execution and continuous delay interpolation are
currently unsupported.
