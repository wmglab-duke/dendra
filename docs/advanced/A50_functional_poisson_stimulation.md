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
randomness as input data: independent uniform draws for one step and the
generator state after those draws. The numerical transition builds the
schedule and evaluates the waveform without calling a random generator or
modifying the source model.

The complete
{download}`functional_poisson_stimulation.py <../../examples/functional_poisson_stimulation.py>`
example compares native values and amplitude gradients, replays an explicit
draw tape, and continues a compiled rollout. Run it from the repository root
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
are masked before waveform evaluation.

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

## Gradients and compiled execution

Uniform draws, Poisson event times, and generator bytes are detached, matching
native Poisson schedule generation. Gradients continue through the wrapped
waveform's registered parameters, PyTorch parametrizations, injection scales,
and extracellular fields. The example verifies the wrapped pulse amplitude
against native execution:

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

Functional randomized Poisson schedules require finite `n`. Nested randomized
Poisson schedules, repeated references to one randomized node within a single
waveform expression, and invocation paths whose count or order depends on
runtime values are currently unsupported. Custom outer `Waveform` subclasses
containing randomized Poisson children are also currently unsupported; compose
these children using waveform arithmetic or `repeat()`. Custom waveform leaves
wrapped by `.poisson(...)` are supported. A physical mechanism-owned
randomized waveform can be evaluated at most once per Population step and
at most once during that Population's fresh initialization. Multiple local
evaluations require distinct Poisson wrapper instances; their wrapped
waveforms can share parameters. Randomized extracellular waveforms must be
bound through `make_functional(..., extra=...)` or
`dn.compile(..., extra=...)`; runtime `extra` and standalone public stimulus
assembly cannot supply the owning RNG carry. Fixed Poisson schedules
(`randomize_every_call=False`) remain ordinary explicit waveform data and do
not require a stimulation draw tape.

A Poisson generator can be shared with other Poisson waveforms across
Populations. Sharing the default Torch generator with connection sampling is
also supported. Sharing an internal model-owned RAND, NOISE, or NetStim
generator with Poisson stimulation is currently unsupported; supply a separate
`torch.Generator` for that stimulation.

```{code-cell} python
:load: ../../examples/functional_poisson_stimulation.py
:tags: [remove-input]
```
