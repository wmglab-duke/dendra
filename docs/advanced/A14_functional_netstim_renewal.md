---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-netstim-renewal)=
# Shared NetStim renewal with explicit random draws

An initialized, unscheduled `dn.NetStim` can be lowered with
`dn.func.make_functional_netstim`. Native and functional execution use the
same pure initialization and renewal transitions in the network layer.
Construct the source and connections through the ordinary Network API;
functional execution supplies parameters, clocks, and random draws explicitly.

The complete example uses one noisy, seeded source to drive two projections.
It checks native trajectories, delivery queues, gradients, and continuation
after restoring the source's clock and random stream. Run it directly with
`python examples/functional_netstim_renewal.py` or download
{download}`the script <../../examples/functional_netstim_renewal.py>`.

```{literalinclude} ../../examples/functional_netstim_renewal.py
:language: python
:start-after: "# region network"
:end-before: "# endregion network"
```

## Extract parameters and replayable state

Lower after initialization and after configuring the source. Extraction
returns independent snapshots; differentiating them does not modify the
native source or reach its parameters.

```{literalinclude} ../../examples/functional_netstim_renewal.py
:language: python
:start-after: "# region lower"
:end-before: "# endregion lower"
:dedent: 4
```

`NetStimTensors` contains three dictionaries:

| Dictionary | Entries |
| --- | --- |
| `parameters` | Floating `interval`, `start`, `noise`, and `tau`, broadcast to the full source state shape. |
| `constants` | Integer `max_spikes`, also with the full state shape. |
| `state` | `next_stoch_time`, integer `spike_counts`, `t_last`, and CPU byte tensors `rng_state` and `seeder_state`. |

The last state dimension indexes generators; leading dimensions are native
batch axes. `interval` is the effective positive interval in milliseconds,
rather than the native `PositiveParam.rho` coordinate. Preserve positivity
when optimizing it, for example by passing `softplus(raw_interval)` to the
step. Preserve the usual NetStim noise range of zero to one and positive
`tau`. Native and functional parameterizations must agree when comparing
gradients. `start` affects initialization, whose clocks are detached from
autograd, matching native initialization.

The RNG state identifies the next draw, rather than just the original seed.
Both RNG streams belong in a checkpoint: `rng_state` is the active sampling
stream and `seeder_state` preserves native generator lifecycle state. A fixed
seed gives independently constructed sources the same sampling trajectory;
their seeder streams can still differ until explicitly restored.

## Sample outside the numerical graph

The cold sampling helper restores an independent `torch.Generator` from the
explicit state. It changes neither the native source nor the supplied state:

```{literalinclude} ../../examples/functional_netstim_renewal.py
:language: python
:start-after: "# region sample"
:end-before: "# endregion sample"
:dedent: 8
```

`NetStimDraw(exp_rand, rng_state)` pairs a full-shape unit-exponential draw
with the generator's **post-draw** byte state. Keep these fields together.
Custom draw packets are accepted for explicit stochastic experiments, but a
noise tensor alone cannot prove which RNG state produced it. Use the sampling
helpers when native seeded continuation is required.

Every native forward samples one full-shape uniform tensor and converts it to
`-log(U)`, including deterministic, exhausted, and currently inactive
generators. `sample_step_draw` matches this RNG consumption. Initialization
has a different policy: `sample_initialization_draw(parameters, state)` reseeds
from the lowered source's fixed seed, or continues the supplied stream when
`seed=None`. It consumes one full-shape draw only if any `noise > 0`;
otherwise it returns zeros and retains the stream after any reseeding. Pass
that packet to `initial_state(parameters, constants, state, draw)` to reset clocks
and counts while retaining the seeder state.

Sampling, extraction, and native state commits stay outside `torch.compile`
and `torch.func` transforms. The tensor-only `initial_state` and `step`
operations can be compiled or transformed with explicit draw packets.

## Generate once, then deliver in connection order

The pure step returns a new state and `NetStimOutputs(spikes, spike_gate)`:

```{literalinclude} ../../examples/functional_netstim_renewal.py
:language: python
:start-after: "# region generate"
:end-before: "# endregion generate"
:dedent: 8
```

`spikes` is boolean. `spike_gate` has the same hard forward events and a
straight-through sigmoid gradient with width `tau`. With `bptt=True`, renewal
clocks retain gradient history across calls. With `bptt=False`, returned
clocks are detached; the current output gate still carries gradients with
respect to time and `tau`. Learning `interval` or `noise` through renewal
history requires `bptt=True`. This setting is independent of whether downstream
NetCons consume hard spikes or surrogate gates.

Supply a scalar time, one time per native batch item, or a tensor broadcastable
to the full state shape. The step follows native sampled-time behavior:
equality fires, at most one renewal occurs per call, and the next interval is
added to the stored event time. Repeated times are separate invocations;
skipping time does not run an implicit catch-up loop. Keep the same time
sequence and draw consumption when comparing executions.

Generate once per Network timestep and reuse that output for every outgoing
projection. Then, in connection insertion order, read due events, receive
them, and advance the projection with this source's gates:

```{literalinclude} ../../examples/functional_netstim_renewal.py
:language: python
:start-after: "# region project"
:end-before: "# endregion project"
:dedent: 12
```

The example's training NetCons use `diff_spiking=True`, so they consume
`output.spike_gate`. For hard delivery use
`output.spikes.to(projection.source_dtype)`. NetStim sources bypass NetCon
threshold detection. Advance each population once after all connections,
preserving the ordering described in {doc}`A13_functional_threshold_detection`.

## Continue with the ordinary Network

Saving only the next event time does not preserve a noisy source's future.
Retain the entire explicit state, including both RNG byte tensors.
`generator.commit_state_(native_source, state)` restores clocks and RNG using
the native checkpoint protocol; it resets transient output buffers, which
the next source invocation recomputes. A training-mode destination preserves
the incoming clock's autograd history; an eval-mode destination detaches it,
matching native checkpoint restoration.

```{literalinclude} ../../examples/functional_netstim_renewal.py
:language: python
:start-after: "# region commit"
:end-before: "# endregion commit"
:dedent: 12
```

Commit at a complete timestep boundary. Native continuation also needs every
projection queue, population carry, and owning Network/population clock at
that same boundary. The example first aligns a receiving Network through
ordinary steps, deliberately advances its source alone, then restores source,
projection, and population state and checks the remaining native trajectory.

State commits do not apply `parameters` or `constants`. If these have changed,
set matching native values before continuing the same trajectory. Reinitialize
and lower again after changing source configuration, seed, training mode,
batch shape, device, or schedules. Schedule edits and clears, schedule-source
binding, `set_dt`, and `set_diff_config` invalidate a plan even when reapplying
the same settings; make these calls before lowering. Ordinary runtime stepping
and unscheduled checkpoint restoration do not invalidate the plan.

The complete example executes here:

```{code-cell} python
:load: ../../examples/functional_netstim_renewal.py
:tags: [remove-input]
```

This lowering currently accepts unscheduled NetStims. Native heap and tensor
schedules retain their existing behavior through schedule inputs to the same
shared renewal core. Functional schedule lowering and complete
`Network`-level lowering remain future extensions.
