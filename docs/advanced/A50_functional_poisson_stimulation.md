---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-poisson-stimulation)=
# Poisson stimulation with explicit randomness

`Waveform.poisson(randomize_every_call=True)` regenerates a schedule whenever
the waveform is evaluated. Functional Populations and Networks represent that
randomness as input data: independent uniform draws for each waveform
invocation and the generator state after those draws. The numerical transition
builds the schedule and evaluates the waveform without calling a random
generator or modifying the source model.

The complete
{download}`functional_poisson_stimulation.py <../../examples/functional_poisson_stimulation.py>`
example defines a custom waveform with two randomized Poisson children, evaluates
them in a fixed authored order, and compares native values and parameter
gradients. It replays an explicit draw tape and continues a compiled rollout,
covering both intracellular stimulation and a mechanism that evaluates its
injections twice. Run it from the repository root
with `python examples/functional_poisson_stimulation.py`.

## Make randomness explicit

Create a randomized waveform with a finite event count and an optional private
Torch generator:

```{literalinclude} ../../examples/functional_poisson_stimulation.py
:language: python
:start-after: "# region waveform"
:end-before: "# endregion waveform"
:dedent: 8
```

The generator's state is captured when the Population or Network is lowered.
The functional state owns an independent copy. Host sampling reconstructs a
temporary generator from those bytes, samples the uniforms, and returns both
the draws and the generator's resulting state. It leaves the source generator
and global Torch RNG unchanged.

Pure schedule construction uses the supplied uniforms for the exponential
intervals and combines them with `interval`, `start`, `noise`, and `off`.
It preserves native scalar draw order, including the interval drawn after the
last emitted event. `noise=0` generates periodic events without consuming
randomness. Finite schedules use fixed-size tensors; unused event positions
do not contribute to the resulting waveform.

The two Poisson children in this example share the same pulse amplitude
parameter and generator. Each evaluation regenerates its own schedule.
Custom `Waveform` containers express their arithmetic and child-call order in
`fn` or `forward`:

```{literalinclude} ../../examples/functional_poisson_stimulation.py
:language: python
:start-after: "# region custom-waveform"
:end-before: "# endregion custom-waveform"
```

`OrderedPoissonWaveform` evaluates `second`, then `first`, then `first` again.
Its registered `gain` parameter scales the result. Functional lowering preserves
that order, including repeated evaluations of the same child; registration
order does not determine draw order. The child schedules use independent draws
while their shared generator advances as one stream. The underlying amplitude
parameter remains shared and receives the sum of its derivatives.

Keep children registered as module attributes, `torch.nn.ModuleList`, or
`torch.nn.ModuleDict`, and call them through those registrations. Constant
indices and keys are supported, including registered `torch.nn.Sequential`
children. Parameters and numerical constants belong in registered parameters
or buffers. The container's arithmetic must be read-only and deterministic
given its child results; Dendra supplies the registered Poisson children's
random draws explicitly. Child invocations must have a fixed count and order.
Functional execution trusts that contract by default. Optional diagnostics enabled with
`dn.ctx(CHECK_PURITY=True)` require `FUNCTIONAL_PURE = True` on each concrete
custom class and audit its evaluation. These diagnostics currently cannot
qualify calls through local module aliases or runtime-selected child indices.

Waveform arithmetic, `repeat()`, and fixed-count nested Poisson expressions
follow the same invocation machinery. A mechanism can also evaluate the same
registered injection more than once in a hook:

```{literalinclude} ../../examples/functional_poisson_stimulation.py
:language: python
:start-after: "# region repeated-mechanism"
:end-before: "# endregion repeated-mechanism"
```

The two `injection_values` calls evaluate the custom waveform twice, consuming
six Poisson invocations in total. Initialization and accepted steps have
separate invocation profiles. The sampling helpers allocate a fixed-size packet
for those profiles: a registered path evaluated once uses an event vector, and
a path evaluated repeatedly uses an invocation axis before its event axis. A
rollout adds the leading time axis. Unused invocation slots and registered
children that are not evaluated do not consume randomness.

For the explicit Population API, sample from the incoming state and pass the
packet in `RolloutInput.stimulation`:

```{literalinclude} ../../examples/functional_poisson_stimulation.py
:language: python
:start-after: "# region explicit-draws"
:end-before: "# endregion explicit-draws"
:dedent: 4
```

`sample_stimulation_draw(state)` supplies one `StepInput.stimulation` packet.
`sample_stimulation_draw_tape(state, steps)` supplies a time-first rollout
packet. Reusing the same state and packet replays the same result. Sampling a
continuation starts from the returned state, so it advances the independent
functional RNG stream rather than restarting it.

For Networks, use `functional.sample_population_stimulation_tapes(state,
steps)` and assign each non-`None` packet to its Population's
`RolloutInput.stimulation` inside `NetworkRolloutInput.populations`. The
Network sampler follows the authored Population and stimulation order while
retaining one RNG carry per physical generator.

Fresh initialization can also evaluate mechanism-owned injections while
building its initial current frame. The pure API obtains that separate packet
with `functional.sample_stimulation_initialization_draw(state)` and passes it
as `InitializationInput.stimulation`. Intra and extra assembly do not consume
Poisson draws during initialization. Host runners using
`initialization="fresh"` supply missing initialization packets, and cached
initialization follows native restoration and draw rules.
Native `initial_values` and `derive_buffers` hooks require pure builders;
randomized waveform evaluation belongs in current hooks such as `assigned_values`.

## Gradients and compiled execution

Uniform draws, Poisson event times, and generator bytes are detached, matching
native Poisson schedule generation. Gradients continue through the wrapped
waveform's registered parameters, PyTorch parametrizations, injection scales,
and extracellular fields. Custom container parameters participate in the same
gradient calculation. The example verifies both the shared pulse amplitude and
the container's gain against native execution:

```{literalinclude} ../../examples/functional_poisson_stimulation.py
:language: python
:start-after: "# region gradients"
:end-before: "# endregion gradients"
:dedent: 4
```

`dn.compile(...)` supplies missing stimulation packets at host boundaries.
Its scan and inference while loops consume explicit tensors. Activation
checkpoint replay uses the same tape, without sampling a second time during
backward. Calls without explicit state restart from the captured snapshot;
passing the returned state continues its clock and RNG:

```{literalinclude} ../../examples/functional_poisson_stimulation.py
:language: python
:start-after: "# region compiled"
:end-before: "# endregion compiled"
:dedent: 4
```

Preserve the complete functional state in a runtime checkpoint, together with
matching model parameters and constants. Generator carry and the retained
stimulation schedule belong to that state; retaining only voltages loses the
random-stream continuation. Callback state is separate and must also be
passed when continuing a recording.

After a step, every alias of a shared Poisson waveform retains the schedule
from its final invocation. Checkpoint continuation needs that latest schedule
and one generator state per physical stream, rather than the complete history
of invocation draws.

`commit_state_` restores model-owned intracellular and mechanism stimulation
when publishing functional state to a native model. Bound `extra` waveforms
remain external objects, so native commit and the model's native checkpoint
do not restore their schedules or RNG. Functional checkpoint continuation
retains that external stimulation carry. To continue natively with bound
`extra`, restore the external waveforms' state separately as well.

## Evaluation boundaries and current limits

Functional steps and compiled rollouts use repeated `Population.step` or
`Network.step` evaluation boundaries. Each timestep evaluates its bound
extracellular stimulation in the supplied mapping order, then intracellular
stimulation and mechanism-owned injections in Population order. A generator
shared by several supported waveforms owns one carry and advances in that call
order.

Native `Population.run` and `Network.run` retain their horizon-level waveform
assembly: a randomized waveform can be evaluated once for a whole time vector.
That call regenerates one schedule for the vector. Repeated native steps and
functional rollouts regenerate at each step, so the same initial seed need
not produce the same stimulation as a horizon-level native run.

Functional randomized Poisson schedules require finite `n` and a fixed
invocation structure. Supported nested parents include finite `n` with infinite
`off`, deterministic parents with `noise=0` and finite `off`, and parents with
`n=1` and finite `off`. An empty parent schedule retains the native single
infinity sentinel and its child evaluation. Randomized parents with finite
`off` whose event count changes the number of randomized child evaluations
are currently unsupported. Invocation paths whose count or order depends on
runtime values are also currently unsupported.

Nested sampling validates that the draws preserve the bound invocation
profile. Draws that shorten a parent schedule, such as a zero uniform producing
an infinite interval, are rejected. This check applies to host sampling and
explicit draw packets.

Custom `Waveform` containers with randomized Poisson children are supported
when their invocation count and order are fixed. Custom waveform leaves
wrapped by `.poisson(...)` are also supported. Changing children, their
registrations, or the invocation recipe after lowering requires a new
functional plan.
Randomized extracellular waveforms must be bound through
`make_functional(..., extra=...)` or
`dn.compile(..., extra=...)`; runtime `extra` and standalone public stimulus
assembly cannot supply the owning RNG carry. Fixed Poisson schedules
(`randomize_every_call=False`) without randomized descendants remain ordinary
explicit waveform data and do not require a stimulation draw tape.

A Poisson generator can be shared with other Poisson waveforms across
Populations. Sharing the default Torch generator with connection sampling is
also supported. Sharing an internal model-owned RAND, NOISE, or NetStim
generator with Poisson stimulation is currently unsupported; supply a separate
`torch.Generator` for that stimulation.

```{code-cell} python
:load: ../../examples/functional_poisson_stimulation.py
:tags: [remove-input]
```
