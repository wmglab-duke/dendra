---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-runtime-noise)=
# Compile models with runtime noise

`GLOBALNOISE`, `BATCHNOISE`, and `RANGENOISE` declare detached stochastic
buffers. Mechanisms, nested States, and MaterialProcesses refresh these buffers
before the Population transition on each timestep. Functional Population and Network
execution uses the same distribution transforms and timestep scaling as
native execution, with explicit base draws and RNG carry.

The complete
{download}`functional_runtime_noise.py <../../examples/functional_runtime_noise.py>`
example checks native trajectories, detached noise, explicit tapes, and
Network continuation. Run it from the repository root with
`python examples/functional_runtime_noise.py`.

## Declare runtime noise

This leak mechanism adds a normally distributed current drive:

```{literalinclude} ../../examples/functional_runtime_noise.py
:language: python
:start-after: "# region declaration"
:end-before: "# endregion declaration"
```

| Declaration | Sample scope |
| --- | --- |
| `GLOBALNOISE` | One scalar for the owner. |
| `BATCHNOISE` | The owner's BATCH layout, shared over compartments. |
| `RANGENOISE` | The owner's RANGE layout, including regional mechanism support. |

`cadence="step"` and `phase="pre_state"` select the supported runtime
sampling phase. Each declaration applies one scaling rule to its
transformed sample `x`, using the current timestep `dt`:

| `scale` | Effective sample |
| --- | --- |
| `"standard"` | `x` |
| `"dW"` | `x * sqrt(dt)` |
| `"white"` | `x / sqrt(dt)` |

Dendra's simulation timestep uses milliseconds. These scaling rules do not
convert units or choose an SDE integration method.

NOISE samples are detached from distribution parameters and base draws.
Selecting `eta_mu_param` as trainable does not create a gradient through
`eta`; autograd reports an unused or zero gradient. Ordinary mechanism
parameters, input tapes, and state remain
differentiable through their use of the sampled drive:

```{literalinclude} ../../examples/functional_runtime_noise.py
:language: python
:start-after: "# region population-runner"
:end-before: "# endregion population-runner"
:dedent: 4
```

Use RAND declarations for sampled parameter heterogeneity with optional
reparameterized distribution gradients; see
{doc}`A30_functional_random_parameters`. Use State `DIFFUSION` for Itô or
Stratonovich dynamics with differentiable drift, diffusion, and timestep;
see {doc}`A32_functional_state_sde`. MaterialProcesses use explicit NOISE with
their native preparation and Material phase ordering; see
{doc}`A44_functional_material_stochasticity`. Pointwise/custom processes use
the current step's sample. Spatial diffusion retains the realization used
when `set_dt` configured its operator, alongside the independently refreshed
runtime buffer, and preserves that configured realization in checkpoints.

## Supply draw and RNG tapes

A `RuntimeNoiseDraw` contains two mappings: `draws` holds explicit base draws
for runtime NOISE and State SDEs under canonical names, and `rng` holds the
resulting state of each independent stream. A recurrent tape adds a leading time axis to every
Tensor in both mappings. Keep the base draws and matching post-draw RNG
state together so a selected prefix can continue the same stream.

The host `dn.compile` runner samples bounded tapes automatically when they
are omitted. Sampling uses independent copies of the supplied RNG carry;
it does not advance the source model or mutate the input state. Calls
without explicit state restart the captured boundary, including its RNG
carry, so repeated calls reproduce their noisy trajectory.

Pure step and rollout calls require explicit packets. For a Population,
`sample_noise_draw(state)` produces one step and
`sample_noise_draw_tape(state, steps)` produces a time-leading tape outside
compiled or transformed code. Supply them through `StepInput.noise` or
`RolloutInput.noise`:

```{literalinclude} ../../examples/functional_runtime_noise.py
:language: python
:start-after: "# region population-tape"
:end-before: "# endregion population-tape"
:dedent: 4
```

For a Network, `sample_population_noise_tapes(state, steps)` returns one
packet per top-level Population. Supply the tapes alongside that
Population's voltage and current inputs. Shared RNG owners are sampled in
timestep/Population order and retain one authoritative stream carry; see
{doc}`A47_functional_network_rng_sharing`:

```{literalinclude} ../../examples/functional_runtime_noise.py
:language: python
:start-after: "# region network-tapes"
:end-before: "# endregion network-tapes"
:dedent: 4
```

A Network's NetStim draws are separate. Use
`functional.sample_draw_tape(state, steps)` for the `netstim=` field of
`NetworkRolloutInput`; see {doc}`A14_functional_netstim_renewal` and
{doc}`A20_functional_network_rollout`. Both inputs carry their own post-draw
RNG states.

## Preserve ordering and continue checkpoints

Initialization samples NOISE buffers in native owner order, including
nested State initialization. `initialization="fresh"` uses the current
explicit parameters and RNG boundary. Its
`InitializationInput.random_draw` packet contains ordered RAND and NOISE
initialization draws, so declarations sharing an `rng_name` consume one
shared stream in the same order. Use the initialization sampling helpers in
{doc}`A30_functional_random_parameters` to supply this packet explicitly.
A NOISE `seed` initializes a persistent stream; fresh sampling consumes its
retained suffix. Snapshot initialization retains the captured buffers and RNG
suffix. A
steady-state cache restores its saved NOISE buffers and RNG suffix without
performing fresh sampling; see
{doc}`A29_functional_network_cache_initialization`. RAND declarations can
invalidate a Population-owned cache according to their resampling policy.

Within each Population timestep, each Mechanism's NOISE is refreshed before its
nested State NOISE, followed by MaterialProcess NOISE in insertion order and
then the state/voltage transition. Network connection delivery
precedes Population stepping, so a connection whose `pre_var` observes a
NOISE buffer reads its retained value from the incoming boundary.
Repeated preparation and checkpoint backward do not consume additional
draws. Each timestep's resulting RNG state is explicit carry, alongside the
current noise buffer values. Scan, checkpointed scan, and inference-only
while execution consume the same tapes.

Continue from the returned state and callback state to sample or supply the
next tape from that boundary:

```{literalinclude} ../../examples/functional_runtime_noise.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 4
```

When fresh initialization also replaces parameters or RAND/cache
materializations, carry `auxiliary["parameters"]` and
`auxiliary["constants"]` as described in
{doc}`A21_functional_network_compile`. NOISE-only continuation uses the
returned simulation and callback state. Native checkpoint commit restores the
noise buffers and RNG suffix, so a native simulation can continue from the
same boundary. Align native raw parameters separately when functional
optimizer leaves have changed.

A NOISE declaration on the Population itself retains native behavior:
initialization or an explicit native resampling call samples it; the native
timestep refreshes inserted Mechanisms and nested States. A root
Population declaration does not add a new recurrent sampling phase.
`derive_buffers` and `derive_timestep_buffers` can read declared NOISE. These
workspaces retain their initialization or timestep-configuration sample
independently of later runtime refreshes; see
{doc}`A46_functional_noise_workspaces`. Population-owned RAND and NOISE sample
before parameter-writing pre-initialize transforms;
Mechanisms, nested States, and MaterialProcesses sample afterward. Retained
realizations preserve the distribution parameters at their own sampling
boundary; see {doc}`A45_functional_stochastic_initialization_ordering`.

RNGModule owners can be shared between Network Populations. Use the Network
sampling methods to retain the shared draw sequence rather than sampling
the Populations independently. See {doc}`A47_functional_network_rng_sharing`
for initialization, reseeding, caches, and continuation. Custom distributions
require the split
`draw_sampler=` and pure `transform=` registration contract described in
{doc}`A30_functional_random_parameters`; legacy sampler-only registrations
remain available in native execution. The authored transform must remain pure
and deterministic. Functional lowering trusts this contract by default;
`dn.ctx(CHECK_PURITY=True)` enables diagnostic checks for newly created plans.
The explicit draw/RNG and tensor-schema requirements apply in either mode.
Direct NOISE-buffer recording with the
built-in `Recorder` is currently unsupported; voltage and declared State/carry
recording retain their existing APIs.

## Run the complete example

```{code-cell} python
:load: ../../examples/functional_runtime_noise.py
:tags: [remove-input]
```
