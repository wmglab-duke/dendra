---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-random-parameters)=
# Compile models with sampled parameter heterogeneity

`GLOBALRAND`, `BATCHRAND`, and `RANGERAND` declare sampled parameters whose
distribution parameters participate in ordinary model parameterization.
Functional Population and Network execution uses the same distribution
transforms as native execution, with explicit base draws and RNG carry.
Sampled values remain fixed throughout a simulation; fresh initialization
applies each declaration's resampling policy.

The complete
{download}`functional_random_parameters.py <../../examples/functional_random_parameters.py>`
example checks native trajectories, distribution-parameter gradients, explicit
draws, repeated trials, and continuation. It includes a custom distribution
used by both native and functional execution. Run it from the repository root
with `python examples/functional_random_parameters.py`.

## Declare sampled parameters

This leak mechanism samples one conductance per RANGE slot from a lognormal
distribution:

```{literalinclude} ../../examples/functional_random_parameters.py
:language: python
:start-after: "# region declaration"
:end-before: "# endregion declaration"
```

`g_mu` and `g_sigma` are the lognormal distribution parameters. Their raw
optimizer leaves follow the ordinary parameter naming and constraint rules;
`g` is the sampled effective conductance. Select distribution parameters with
`trainable=...`, then inspect `runner.trainable_parameters` to verify
the selection. The sample itself is not an independent optimizer leaf.

| Declaration | Sample scope |
| --- | --- |
| `GLOBALRAND` | One scalar for the owner. |
| `BATCHRAND` | The owner's BATCH layout, shared over compartments. |
| `RANGERAND` | The owner's RANGE layout, including regional mechanism support. |

Declarations on supported Populations, Mechanisms, nested States, and
MaterialProcesses use their own native scopes and ordering. See
{doc}`A44_functional_material_stochasticity` for sampled full-field processes.
Compatible Population-family subclasses may add constructors and RAND
declarations while inheriting their numerical and lifecycle methods; see
{doc}`A56_functional_population_subclasses`. Custom execution overrides remain
unsupported. The built-in
distributions are `normal`,
`lognormal`, `uniform`, and `truncated_normal`. Structural changes to
declarations, distribution registrations, scopes, or RNG bindings require a
new functional plan.

## Choose snapshot or fresh initialization

`initialization="snapshot"` retains the sampled parameter realization and RNG
suffix captured from the initialized model. Restarting the runner reuses that
realization. Updating distribution parameters does not request a new draw.

`initialization="fresh"` applies the declaration's `resample_on_initialize`
policy. When it is `True`, initialization samples once from the retained RNG
suffix; repeated preparation or checkpoint backward does not draw again.
When it is `False`, initialization keeps the saved realization without
advancing its RNG stream. An explicit native `resample_random_parameters`
call remains available before lowering a new snapshot; it advances the
current stream without applying the initialization reset policy.

A declared `seed` initializes the persistent RNG stream. With the default
`reseed_on_initialize=False`, it does not reseed every Population initialization.
Set `reseed_on_initialize=True` to reset that declaration's stream to its
current configured or checkpointed seed when fresh initialization resamples
it. Reseeding or loading a checkpoint can change this seed from the original
declaration. Successive initializations with that seed reuse base draws;
changing the current distribution parameters can still change the samples.
`resample_on_initialize=False` keeps the saved sample and RNG state even when
`reseed_on_initialize=True`. Shared streams reset at most once per
initialization generation in native declaration order, and snapshot restart
or explicit continuation does not request a reset.

Construction and native initialization can
already have consumed draws before lowering; a functional plan captures that
current suffix. Calls without explicit state restart the runner's captured
boundary, including its RNG carry, so repeated trials reproduce the same
sampled initialization unless their explicit draw or RNG inputs change.

For differentiable fresh runs, optimize the distribution parameters:

```{literalinclude} ../../examples/functional_random_parameters.py
:language: python
:start-after: "# region population-runner"
:end-before: "# endregion population-runner"
:dedent: 4
```

With the default `reparameterized=True`, gradients pass through the pure
distribution transform to its tensor parameters and explicit base draws.
`reparameterized=False` detaches the sampled value, matching native behavior.
Scan and checkpointed scan preserve these gradients. While execution remains
inference-only.

## Supply explicit draw packets

The host runner samples initialization packets automatically when they are
omitted. Sampling uses an independent copy of the supplied RNG carry; it does
not advance the source model or mutate the input packet.

At the lower level, call
`population_plan.sample_initialization_draw(parameters, state)` outside
compiled or transformed code, then supply the returned packet through
`InitializationInput.random_draw` or the `random_draw=` initialization
argument. `FunctionalPopulation.initialize` accepts `state=` so initialization
can use a selected RNG boundary; `initialize_host` supplies automatic host
sampling for that boundary.

For a Network, `sample_population_initialization_draws` returns a packet for
each top-level Population. Assign it to that Population's initialization input:

```{literalinclude} ../../examples/functional_random_parameters.py
:language: python
:start-after: "# region network-draws"
:end-before: "# endregion network-draws"
:dedent: 4
```

For models with NOISE declarations, this initialization packet also carries
their ordered initialization draws and shared RNG suffix. Recurrent NOISE
tapes are separate; see {doc}`A31_functional_runtime_noise`.

NetStim initialization and per-step draw tapes remain separate inputs; see
{doc}`A14_functional_netstim_renewal` and
{doc}`A21_functional_network_compile`. A shared `rng_name` within an owner
retains native draw order across its declarations. Sharing mutable RNGModule
owners between Network Populations retains one authoritative carry per
stream and preserves native Population sampling order; see
{doc}`A47_functional_network_rng_sharing`.

A randomized Poisson waveform can consume a declared stream through its
public `.rng` generator. Explicit initialization then uses matching RAND/NOISE
and stimulation packets sampled from the same incoming state. For a
Population, use `sample_initialization_draw(parameters, state)` together with
`sample_stimulation_initialization_draw(state)`. For a Network, use
`sample_population_initialization_draws(parameters, state)` together with
`sample_population_stimulation_initialization_draws(parameters, state)`.
Host runners supply both packets automatically when omitted. See
{doc}`A50_functional_poisson_stimulation` for sharing, runtime tapes, and a
complete example.

`stream.ignore_rng_on_load(True)` retains the incoming RNG seed and position
when a valid steady-state cache restores sampled values. Runtime checkpoint
restoration remains exact; see {doc}`A48_functional_rng_load_policies`.

Fresh initialization returns effective constants containing the sampled
realization and its base draws. Continue with both the returned parameters
and constants, together with simulation and callback state:

```{literalinclude} ../../examples/functional_random_parameters.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 4
```

Explicit state bypasses initialization, so continuation does not resample.
Runtime checkpoints retain the selected samples and RNG suffix for native
continuation. As with other functional commits, align native raw parameters
separately before continuing a model whose optimizer leaves have changed.

## Define a custom distribution

Register a host `draw_sampler` and a pure `transform`. The sampler receives
the declaration's RNG and requested shape; the transform receives resolved
distribution parameters and the explicit base draw. Native and functional
execution call the same transform:

```{literalinclude} ../../examples/functional_random_parameters.py
:language: python
:start-after: "# region custom-distribution"
:end-before: "# endregion custom-distribution"
```

The transform must be deterministic, tensor-native, free of mutation and
hidden RNG, and valid for functional transforms. Its output must match the
declaration's sample layout. Registering only the legacy `sampler=` callable
retains native execution; functional lowering requires the split
`draw_sampler=` and `transform=` contract. Both split functions must be
supplied together.

Functional lowering trusts the transform's purity contract by default.
Create a plan or runner inside `with dn.ctx(CHECK_PURITY=True):` to enable
diagnostic dependency, mutation, and repeated-evaluation checks. Each plan
retains its selected mode after the context exits. Explicit draw bindings and
the returned tensor's required schema are checked in either mode; diagnostics
do not prove purity for every possible draw.

## Steady-state caches and unsupported stochastic features

A Population-owned `steady_state` cache is invalidated when initialization
resamples a RAND declaration, matching native initialization. The source
Population's cache is not mutated by functional execution. Declarations with
`resample_on_initialize=False` can restore the owned cache without a new draw.
A Network-global cache takes precedence: its saved Population samples and
RNG state are restored without running Population resampling. Cached
materializations remain detached; see
{doc}`A29_functional_network_cache_initialization` for gradient boundaries.

Per-step `GLOBALNOISE`, `BATCHNOISE`, and `RANGENOISE` buffers use detached
explicit draw/RNG tapes; see {doc}`A31_functional_runtime_noise`. RAND
heterogeneity does not add random draws to the recurrent timestep. State
`DIFFUSION` supports explicit Wiener draws and pathwise gradients; see
{doc}`A32_functional_state_sde`.

Fresh initialization samples Population-owned RAND and NOISE before
parameter-writing pre-initialize transforms. Mechanisms, nested States, and
MaterialProcesses sample afterward. The earlier sample retains its selected
distribution-parameter frame when a transform changes a shared source; see
{doc}`A45_functional_stochastic_initialization_ordering` for a complete example.

## Run the complete example

```{code-cell} python
:load: ../../examples/functional_random_parameters.py
:tags: [remove-input]
```
