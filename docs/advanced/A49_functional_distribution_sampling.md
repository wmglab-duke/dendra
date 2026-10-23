---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-distribution-sampling)=
# Author connection distributions with explicit draws

Custom `dendra.models.distributions.Distribution` subclasses can supply NetCon
and ContinuousCon weights and delays through a shared sampling contract.
Native initialization and functional initialization use the same pure
transform. Registered parameters, buffers, PyTorch parametrizations, and
shared tensor identities remain explicit, and samples retain pathwise
gradients through their raw sources.

The complete
{download}`functional_distribution_sampling.py <../../examples/functional_distribution_sampling.py>`
example defines a bounded uniform distribution, shares a trainable mean
between event and continuous weights, and parametrizes positive widths. It
compares compiled checkpointed continuation, voltage recording, raw-source
gradients, and RNG carry against native execution. Run it from the repository
root with `python examples/functional_distribution_sampling.py`.

## Separate host randomness from the pure transform

Implement three hooks on a `Distribution` subclass:

| Hook | Contract |
| --- | --- |
| `sample_draw_schema(sample_shape=torch.Size())` | Return `(shape, dtype, device)` for one floating-point base-draw Tensor, without drawing randomness. |
| `sample_draw(sample_shape=torch.Size(), *, generator=None)` | Draw that Tensor with the supplied Torch generator. With `generator=None`, native initialization uses ambient Torch RNG. |
| `sample_from_draw(draw)` | Return the effective sample using the supplied draw and registered sources, without RNG or mutation. |

The inherited `sample(n)`, `rsample(sample_shape)`, and
`sample_with_draw(sample_shape, *, generator=None)` methods adapt these hooks
for native callers and retain base-draw provenance. Inherit those adapters
when using the pure connection sampling contract. Implement `log_prob` when
the distribution also needs likelihood evaluation.

The draw can have multiple axes: for example, a transform may consume several
noise coordinates per connection. Its shape describes the noise consumed by
the transform; the transformed sample must still match the connection's
requested sample layout. The draw's shape, floating dtype, and device must
match its schema. Use registered parameters and buffers for tensor
dependencies, including fixed bounds, instead of unregistered tensor state.

The optional `sample_draw_placeholder(sample_shape=torch.Size())` hook returns
a deterministic valid draw for setup-time evaluation. Its default is zeros
with the schema's shape, dtype, and device. Override it when zero lies outside
the transform's valid domain, such as an inverse CDF requiring a probability
strictly between zero and one. This hook must not consume randomness.

The example uses uniform base draws and a midpoint placeholder:

```{literalinclude} ../../examples/functional_distribution_sampling.py
:language: python
:start-after: "# region distribution"
:end-before: "# endregion distribution"
```

`sample_draw_schema`, `sample_draw_placeholder`, and `sample_from_draw` must
be deterministic and nonmutating. `sample_from_draw` must use tensor
operations supported by the chosen functional and compiler transforms.
Functional lowering trusts these contracts by default. Lower inside
`with dn.ctx(CHECK_PURITY=True):` to enable diagnostic checks for hidden
dependencies, mutation, hidden RNG, and inconsistent evaluation. Required
draw and output schemas are checked in either mode; diagnostics do not prove
correctness for every possible draw.

The host draw sampler must honor its explicit generator and return detached
random inputs. Gradients flow through the pure transform's registered sources,
including parametrization originals and transform parameters, rather than
through the random draw. Direct sampling can also use an isolated generator:

```{literalinclude} ../../examples/functional_distribution_sampling.py
:language: python
:start-after: "# region explicit-draw"
:end-before: "# endregion explicit-draw"
:dedent: 4
```

## Share and select raw parameters

Pass the same `nn.Parameter` to several distributions to retain one shared
optimizer leaf. Passing the same Distribution module shares all its
registered sources. Each connection block still receives its own independent
draw in native initialization order.

```{literalinclude} ../../examples/functional_distribution_sampling.py
:language: python
:start-after: "# region sources"
:end-before: "# endregion sources"
:dedent: 8
```

Raw source paths appear below `synapses.<connection>.weight.weights.<block>.param`
or the corresponding `delay_ms` and `continuous_synapses` paths. In this
example, `param.mean` is shared and
`param.parametrizations.half_width.original` is the unconstrained log width.
Inspect `runner.named_parameters` and `runner.find_parameters(...)` to verify
the selected names and aliases. Exact names, globs, and regex selectors are
supported:

```{literalinclude} ../../examples/functional_distribution_sampling.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 4
```

Selecting one shared alias selects its common leaf. Deduplicate
`runner.trainable_parameters.values()` by identity when creating an optimizer.
Updating a source rematerializes its effective sample from retained base
draws; it does not draw new randomness during a rollout.

## Initialize, restore, and continue

`initialization="fresh"` samples after Population initialization/cache writes,
in native connection order: event connections, continuous connections, then
NetStim initialization. Each connection samples its weights before its
delays, with declaration blocks in order. The host runner supplies the draws
and advances explicit connection RNG carry without modifying global Torch
RNG. The pure `FunctionalNetwork.initialize` API instead requires a
`ConnectionSamplingDraw`; see {doc}`A41_functional_connection_sampling` for the
explicit packet and role-specific resampling flags.

`initialization="snapshot"` preserves initialized samples and their saved
draw provenance. Model `state_dict()` retains both effective sampled values
and provenance. Legacy model checkpoints without provenance retain their
effective values; their historical raw sampling Jacobian is unavailable until
fresh sampling establishes a draw. A Network-global steady-state cache
restores Population sources before connection initialization; it does not
suppress connection resampling. The default `reinit_weights=True` and
`reinit_delays=True` refresh samples from the current restored sources before
restoring connection history. Set a role's flag to `False` to retain its
effective values and draw provenance without consuming RNG for that role.
Cached Population numerical fields retain native detachment rules; see
{doc}`A29_functional_network_cache_initialization`.

Repeated runner calls without explicit state restart at the captured boundary.
For continuation, preserve returned parameters and constants, runtime state,
and callback state:

```{literalinclude} ../../examples/functional_distribution_sampling.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 4
```

Checkpoint replay reuses the same draws. Scan and inference while execution
retain sampled connections and RNG without resampling each timestep.
Native `commit_state_` carries connection RNG checkpoint metadata without
resetting global Torch RNG. Runtime checkpoints omit static model parameters,
so use matching model data when restoring native execution. Source registration,
Python configuration, fixed integer delay routes, and the fresh draw schema
remain lowering-time configuration. Functional replay preserves a retained
draw's saved dtype. Custom distributions and built-in TruncatedNormal also
preserve retained draw dtypes across native conversion and model loading;
device conversion moves those draws without changing their dtype. Built-in
Normal follows its source dtype during conversion. Changing the dtype policy
for fresh draws requires lowering again.

The example verifies gradients through the shared mean and both parametrized
width sources:

```{literalinclude} ../../examples/functional_distribution_sampling.py
:language: python
:start-after: "# region gradients"
:end-before: "# endregion gradients"
:dedent: 4
```

## Existing sampling implementations

Subclasses that override native `sample`, `rsample`, or `sample_with_draw`
retain their authored native behavior. They can retain initialized fixed
storage with the corresponding `reinit_weights=False` or
`reinit_delays=False`, but functional fresh resampling through those custom
adapters is currently unsupported. Implement the split draw hooks and inherit
the shared adapters to support fresh functional sampling. Built-in `Normal`
and `TruncatedNormal` use the same connection sampling path.

The `dn.register_random_distribution(...)` API uses separate `draw_sampler=`
and `transform=` functions for RAND/NOISE declarations; see
{doc}`A30_functional_random_parameters` for that contract.

```{code-cell} python
:load: ../../examples/functional_distribution_sampling.py
:tags: [remove-input]
```
