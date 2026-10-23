---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-rng-load-policies)=
# Select RNG behavior when loading cached state

`stream.ignore_rng_on_load(True)` retains that stream's current seed and
generator position during model state-dict loading and steady-state cache
restoration. Other saved values, including sampled RAND parameters and NOISE
buffers, still restore. `stream.ignore_rng_on_load(False)` selects the default:
restore the saved seed and generator position as well.

Functional Population and Network initialization use these same policies.
Runtime checkpoint restoration and `commit_state_` restore the supplied exact
RNG position independently of the flag, so a resumed trajectory continues
from its saved boundary.

The complete
{download}`functional_rng_load_policies.py <../../examples/functional_rng_load_policies.py>`
example restores shared-stream Population caches, continues a compiled
Network, and commits an exact checkpoint to a destination with different
load flags. Run it with `python examples/functional_rng_load_policies.py`.

## Choose policies for each stream

This mechanism retains its RAND conductance across initialization and uses
one stream for that sample and runtime NOISE:

```{literalinclude} ../../examples/functional_rng_load_policies.py
:language: python
:start-after: "# region declaration"
:end-before: "# endregion declaration"
```

Set a policy on each stream before lowering the model. Shared aliases refer
to the same stream object and therefore the same policy:

```{literalinclude} ../../examples/functional_rng_load_policies.py
:language: python
:start-after: "# region policy"
:end-before: "# endregion policy"
:dedent: 4
```

The example saves its caches before reseeding and advancing both streams.
Cache initialization restores the numerical warmup state. The `kept` stream
retains its later seed and position; `restored` returns to its cached seed
and position. Subsequent NOISE draws follow those selected boundaries.

A Population-owned steady-state cache is invalidated when one of its RAND
declarations requests resampling on initialization. Use
`resample_on_initialize=False` for declarations whose cached realization
should be reused. This resampling choice is separate from the RNG load
policy. See {doc}`A30_functional_random_parameters` and
{doc}`A29_functional_network_cache_initialization`.

## Initialize with explicit carry

The incoming state supplies the current positions of ignored streams.
Restored streams take their positions from the selected cache. Both host
initialization and the pure API use the same rule. For standalone Populations,
omitting `state` uses the stream positions captured when lowering:

```{literalinclude} ../../examples/functional_rng_load_policies.py
:language: python
:start-after: "# region initialization"
:end-before: "# endregion initialization"
:dedent: 8
```

Ignored RNG payloads are skipped before decoding their saved seed or device
state. Saved RAND provenance, sample values, and other cache fields retain
their validation requirements. RNG payloads selected for restoration must
contain valid generator states. Loading a runtime checkpoint validates its
RNG payloads even when ordinary model loads would ignore them.

Population-root declarations, Mechanisms, nested States, and MaterialProcesses
use the policy of their selected stream. NOISE-only and State SDE-only owners
use the same carry rules. A shared Network retains one authoritative carry
per physical stream. Cache writes follow native registered-alias and
Population order; an ignored stream skips every saved occurrence of that
same object. See {doc}`A47_functional_network_rng_sharing`.

Policies are part of the source configuration captured by lowering.
Changing a source stream's flag requires lowering it again. A compatible
checkpoint destination may have different load flags because exact runtime
restoration uses the supplied continuation state.

## Continue and restore checkpoints

Pass the resulting state, effective parameters, and constants to subsequent
calls. Scan, checkpointed scan, and inference-only while execution consume
the same selected RNG suffixes:

```{literalinclude} ../../examples/functional_rng_load_policies.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 8
```

The RNG load flag selects generator-state restoration. Cached RAND values
and numerical buffers are detached materializations, retaining the cache's
gradient boundary. Returned raw parameters keep their gradient connections
and shared aliases; subsequent operations using those parameters remain
differentiable. NOISE and Wiener base draws remain detached. State SDE drift,
diffusion, and timestep retain pathwise gradients through their live operands.
Initialization and sampling leave the source model and PyTorch global RNG
unchanged. See {doc}`A29_functional_network_cache_initialization` for cache
gradient semantics.

Exact runtime restoration preserves the destination's flags and stream
aliases while publishing the saved seed and generator position:

```{literalinclude} ../../examples/functional_rng_load_policies.py
:language: python
:start-after: "# region checkpoint"
:end-before: "# endregion checkpoint"
:dedent: 8
```

See {doc}`A31_functional_runtime_noise` and {doc}`A32_functional_state_sde`
for explicit draw tapes and their gradient semantics.

## Run the complete example

```{code-cell} python
:load: ../../examples/functional_rng_load_policies.py
:tags: [remove-input]
```
