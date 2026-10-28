---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-state-sde)=
# Compile stochastic State equations

`State.DIFFUSION` declares the diffusion coefficient of a stochastic State
equation. Native and functional execution share the Euler–Maruyama and
Euler–Heun transitions. Functional execution carries the State values and RNG
boundary explicitly, and accepts replayable standard-normal draws through the
same input packet used for runtime NOISE.

The complete
{download}`functional_state_sde.py <../../examples/functional_state_sde.py>`
example checks native trajectories, pathwise gradients, explicit tapes, and
Network continuation. Run it from the repository root with
`python examples/functional_state_sde.py`.

## Declare drift and diffusion

For the Itô equation `dx = theta * (mu - x) * dt + sigma * dW`, declare the
drift with `DERIVATIVE`, the diffusion coefficient with `DIFFUSION`, and select
Euler–Maruyama:

```{literalinclude} ../../examples/functional_state_sde.py
:language: python
:start-after: "# region ou-state"
:end-before: "# endregion ou-state"
```

The numerical update is
`x_next = x + theta * (mu - x) * dt + sigma * sqrt(dt) * z`, where `z` is a
detached standard-normal base draw. Dendra's timestep is in milliseconds;
choose diffusion coefficient units consistently. `DIFFUSION` supplies the
coefficient, so do not include another `sqrt(dt)` factor in it.

`State.METHOD("euler_heun")` selects a Stratonovich interpretation. It evaluates
diffusion at the incoming State and at a predictor, using the same draw for
both evaluations. `assigned_values` is evaluated at the predictor as well,
so it can express State-dependent diffusion:

```{literalinclude} ../../examples/functional_state_sde.py
:language: python
:start-after: "# region heun-state"
:end-before: "# endregion heun-state"
```

`average_drift=True` also averages the incoming and predicted drift. With its
default `False`, drift uses the incoming State while diffusion is averaged.
Choose the method from the intended stochastic equation; the two stochastic
interpretations generally differ for multiplicative noise.

Insert the State bundle into a Mechanism with `Mechanism.STATE_BUNDLE`, then
insert that Mechanism into a Population as usual. A State can influence
membrane currents, continuous connections, and other supported mechanism
outputs through its declared values.

## Train along a sampled path

Drift and diffusion parameters retain ordinary gradients. Gradients also flow
through initial State values, input tapes, and the timestep, including the
`sqrt(dt)` scaling. Base random draws are detached; gradients through the
sampled path do not differentiate the RNG or its seed.

The host runner samples a bounded tape automatically when one is omitted.
Use `inference=False` and select trainable parameters as for deterministic
models:

```{literalinclude} ../../examples/functional_state_sde.py
:language: python
:start-after: "# region population-runner"
:end-before: "# endregion population-runner"
:dedent: 4
```

Checkpointed scan reuses the same sampled tape during backward replay.
Inference-only while execution consumes the same State transition and draws.
Sampling and checkpoint replay do not consume global PyTorch RNG state or
advance the source model.

Runtime `GLOBALNOISE`, `BATCHNOISE`, and `RANGENOISE` have a different gradient
contract: their transformed values and timestep scaling are detached from
distribution parameters and draws. Use `DIFFUSION` when drift or diffusion
coefficients should receive pathwise gradients; see
{doc}`A31_functional_runtime_noise` for detached stochastic drives and
{doc}`A30_functional_random_parameters` for sampled parameter heterogeneity.

## Supply canonical runtime draws

`functional.sample_runtime_draws(state)` supplies a single-timestep
`RuntimeDraws`; `steps=N` samples a time-leading tape. Pass it through
`draws=` to pure Population/Network execution or to the host runner:

```python
draws = functional.sample_runtime_draws(state, steps=100)
final, auxiliary = dn.func.run(
    bound, state, steps=100, draws=draws, checkpoint_every=25,
)
```

The packet combines the model's runtime NOISE and SDE base draws with any
randomized stimulation requests. Sampling follows native request order and
retains the necessary RNG boundaries for shared streams. Drift, diffusion,
and timestep gradients preserve their existing contracts; the base draws
remain detached. Host execution can sample automatically when `draws` is
omitted. Pure execution consumes explicit stochastic inputs. See
{ref}`functional-runtime-draws` for the complete sampling and continuation
contract.

### Compatibility noise/SDE packets

Pure Population step and rollout calls require explicit stochastic inputs.
`sample_noise_draw(state)` produces one `RuntimeNoiseDraw` packet;
`sample_noise_draw_tape(state, steps)` adds a leading time axis to all its
draw and RNG tensors. Both include runtime NOISE and State SDE draws when
present in the model. Supply the result through `StepInput.noise` or
`RolloutInput.noise`:

```{literalinclude} ../../examples/functional_state_sde.py
:language: python
:start-after: "# region explicit-tape"
:end-before: "# endregion explicit-tape"
:dedent: 4
```

SDE entries in `packet.draws` start with `sde.draws.` and identify the
canonical State path and field. Their tensors match that field's shape,
dtype, and device and contain unscaled standard normals. Use the sampler's
returned schema when inspecting or replacing draws. Keep `packet.rng`, the
matching post-draw RNG boundary, together with the draw values so continuation
uses the intended stream suffix. A replacement draw changes that sampled path;
it does not infer a different RNG boundary.

For a Network, sample one packet per Population and bind it alongside that
Population's other inputs:

```{literalinclude} ../../examples/functional_state_sde.py
:language: python
:start-after: "# region network-tapes"
:end-before: "# endregion network-tapes"
:dedent: 4
```

NetStim draw/RNG tapes remain a separate `NetworkRolloutInput.netstim` input;
see {doc}`A20_functional_network_rollout`.

## Preserve order and continue checkpoints

Each Population refreshes its Mechanism/State NOISE before advancing its
State equations. SDE draws then follow native State/field order. Shared RNG
owners consume one stream in that order; NOISE and SDE draws are not sampled
from independent copies of a shared stream. A Heun predictor reuses its
State's draw rather than consuming a second draw. Network connection delivery
precedes Population advancement, preserving the incoming boundary observed
by continuous source bindings.

Calling a snapshot runner again without explicit state restarts both the
simulation and RNG boundary. Continue from returned simulation and callback
state to retain the sampled suffix:

```{literalinclude} ../../examples/functional_state_sde.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 4
```

Native checkpoints contain SDE RNG streams alongside ordinary State values.
Functional state commit restores those streams, allowing native execution to
continue at the same boundary. Align native raw parameters separately if
functional optimizer leaves have changed. Fresh initialization and
steady-state cache restoration retain their existing State and RNG ordering;
see {doc}`A21_functional_network_compile` and
{doc}`A29_functional_network_cache_initialization`.

Initialization assigns State values through the ordinary initialization hooks;
Wiener draws are consumed by timesteps. Initialization draw packets therefore
contain RAND and NOISE initialization draws, together with the RNG boundary
for all streams. A cache restores its saved State values and SDE RNG suffix
without consuming fresh Wiener draws.

Functional lowering currently supports canonical generated Euler–Maruyama and
Euler–Heun State solvers. Custom SDE `State.advance` overrides, custom RNG
sampling adapters, and stochastic voltage/cable equations are currently
unsupported. Ordinary Population/Network topology and device admission rules
still apply. State SDE support does not change NOISE's detached semantics.

## Run the complete example

```{code-cell} python
:load: ../../examples/functional_state_sde.py
:tags: [remove-input]
```
