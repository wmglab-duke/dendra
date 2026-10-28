---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-network-rng-sharing)=
# Share random streams across populations

A Network can contain Populations whose RAND, NOISE, and State `DIFFUSION`
declarations use the same `RNGModule` object. Native and functional execution
consume that stream in Population order. A functional Network stores one
explicit RNG carry for each physical stream, so continuation preserves the
shared generator position.

The complete
{download}`functional_network_rng_sharing.py <../../examples/functional_network_rng_sharing.py>`
example compares a shared-stream Network with native execution and checks
shared-parameter gradients. Run it with
`python examples/functional_network_rng_sharing.py`.

## Share the native stream

Use `rng_name` to select an owner's stream. This mechanism shares its
initialization RAND and runtime NOISE declarations:

```{literalinclude} ../../examples/functional_network_rng_sharing.py
:language: python
:start-after: "# region declaration"
:end-before: "# endregion declaration"
```

Assign the same stream object to owners in separate Populations before
initializing their Network. Raw distribution parameters can be shared
independently of the stream:

```{literalinclude} ../../examples/functional_network_rng_sharing.py
:language: python
:start-after: "# region sharing"
:end-before: "# endregion sharing"
:dedent: 4
```

Sharing a stream shares its draw sequence; it does not give the Populations
identical samples. The left Population consumes its requests before the
right Population. Separate `RNGModule` objects with the same seed retain
independent positions. The same ownership rules cover Population-root
RAND/NOISE, inserted Mechanisms, nested States, and MaterialProcesses.
A State's Wiener stream can share an owner with its NOISE or other
Populations' streams.

Randomized Poisson waveforms can also borrow that owner's public `.rng`
generator. The waveform can belong to another Population that has no local
RAND, NOISE, or State SDE declaration. Model sampling and stimulation retain
one shared carry, including across Population boundaries. See
{doc}`A50_functional_poisson_stimulation` for a complete example.

## Compile and sample

Compile the ordinary Network:

```{literalinclude} ../../examples/functional_network_rng_sharing.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 8
```

When streams cross Population boundaries, `state["rng"]` owns the Network's
canonical stream states; Population state retains its local sampled values.
A standalone Population retains its `state["random"]["rng"]` mapping.
Networks with independent Population streams retain that Population-owned
layout. Stream ownership is fixed when the Network is lowered; changing its
RNG aliases requires lowering it again.

The host runner samples omitted draw packets from copies of the supplied
carry. Repeated snapshot calls reproduce the same trajectory without
advancing the source Network or PyTorch's global RNG. Pure initialization
uses packets from the Network sampler:

```{literalinclude} ../../examples/functional_network_rng_sharing.py
:language: python
:start-after: "# region initialization"
:end-before: "# endregion initialization"
:dedent: 8
```

Use `sample_runtime_draws(state, steps=N)` for one canonical runtime packet,
then supply it through `draws=`. The sampler orders all runtime families
together, including Network NetStim, randomized extra/intra preparation,
each Population's NOISE, mechanism, and State SDE requests:

```python
draws = functional.sample_runtime_draws(state, steps=100)
final, auxiliary = dn.func.run(bound, state, steps=100, draws=draws)
```

`RuntimeDraws.draws` contains named draw families; `RuntimeDraws.rng` is the
authoritative frame timeline for each physical stream. Ordinary shared model
streams retain per-Population post-request boundaries. Streams also consumed
by randomized Poisson stimulation retain their common physical timestep
suffix. Combining these representations does not discard those intermediate
boundaries or change when native execution publishes them.

The existing `sample_population_noise_tapes(state, steps)` and
`sample_population_stimulation_tapes(state, steps)` methods remain compatible.
Their separate packets must be obtained from the same incoming state and
supplied together for coupled consumers. Independently sampling each Population
from the same starting stream would repeat draws and lose the required ordering. Canonical
`draws=` cannot be combined with those explicit legacy packet fields.

```{literalinclude} ../../examples/functional_network_rng_sharing.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 8
```

Continue from the returned state and effective parameters/constants. Scan,
checkpointed scan, and inference-only while execution use the same packet
sequence. Backward checkpoint replay reuses the forward draws.
NetStim participates in the canonical runtime packet and retains its separate
sampler APIs for compatibility. Initialization-time sampled connection
sources retain their existing initialization RNG APIs.

## Initialization, caches, and gradients

Fresh initialization preserves native root sampling, pre-initialize
transforms, child sampling, and per-Population preparation order. A later
Population sees the shared stream position left by earlier Populations.
`resample_on_initialize=False` retains the RAND sample without consuming a
replacement draw. A RAND declaration with `reseed_on_initialize=True`
resets its selected stream once during that Population's initialization
generation. Another Population's reseeding policy applies at its own
initialization boundary. See {doc}`A30_functional_random_parameters` and
{doc}`A45_functional_stochastic_initialization_ordering`.

Population-owned steady-state caches restore their saved RNG suffix when
that Population initializes. If several caches publish the same stream,
subsequent owners consume the position most recently restored in native
order. Network-owned caches preserve their global restoration boundary.
Runtime checkpoints preserve the canonical carry and restore the original
stream aliases; incompatible state or changed ownership is rejected before
native publication. `commit_state_` restores the supplied RNG position even
when `ignore_rng_on_load=True`; that flag controls state-dict loading.
Use `stream.ignore_rng_on_load(...)` before lowering to select each stream's
model/cache load policy; see {doc}`A48_functional_rng_load_policies`.

RAND retains its distribution's reparameterized gradients. NOISE samples
remain detached from distribution parameters and base draws. State SDE
paths retain gradients through drift, diffusion, and timestep; their base
Wiener draws remain detached. Shared raw parameters receive the sum of their
uses across Populations. RNG state itself remains detached:

```{literalinclude} ../../examples/functional_network_rng_sharing.py
:language: python
:start-after: "# region verification"
:end-before: "# endregion verification"
:dedent: 8
```

See {doc}`A31_functional_runtime_noise`, {doc}`A32_functional_state_sde`, and
{doc}`A29_functional_network_cache_initialization` for the individual APIs.
Custom native RNG implementations and topology changes after lowering are
currently unsupported.

## Run the complete example

```{code-cell} python
:load: ../../examples/functional_network_rng_sharing.py
:tags: [remove-input]
```
