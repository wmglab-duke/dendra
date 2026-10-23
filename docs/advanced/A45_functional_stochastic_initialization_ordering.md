---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-stochastic-initialization-ordering)=
# Sample around initialization transforms

Fresh initialization samples a Population's RAND and NOISE declarations
before its pre-initialize transforms. Mechanisms, nested States, and
MaterialProcesses sample after those transforms. A transform can therefore
write a shared distribution parameter while preserving the Population's
already selected realization.

The complete
{download}`functional_stochastic_initialization_ordering.py <../../examples/functional_stochastic_initialization_ordering.py>`
example compares ordered samples, shared-parameter gradients, RNG state, and
continued execution with a native two-Population Network. Run it with
`python examples/functional_stochastic_initialization_ordering.py`.

## Share a distribution parameter

This Population declares a trial-level RAND value and initialization NOISE.
Its leak Mechanism also samples a conductance:

```{literalinclude} ../../examples/functional_stochastic_initialization_ordering.py
:language: python
:start-after: "# region declarations"
:end-before: "# endregion declarations"
```

The Network shares one raw distribution mean across both Populations and
their Mechanisms. Each Population shifts that mean in its pre-initialize
transform:

```{literalinclude} ../../examples/functional_stochastic_initialization_ordering.py
:language: python
:start-after: "# region sharing"
:end-before: "# endregion sharing"
:dedent: 4
```

The left Population samples its trial value and NOISE from the initial mean,
then writes the shifted mean and samples its leak conductance. The right
Population samples its trial values using that shared mean, applies its own
shift, and samples its leak conductance. Both Populations retain the samples
and distribution-parameter frames selected at their own sampling boundaries.
The returned raw parameter aliases refer to the final shared source.

Deterministic parameter rematerialization after a transform does not request
another draw. Registered PyTorch parametrizations and Dendra parameter
sources use the same raw-to-effective materialization boundary. Sampling
does not require a separate version of the model or an imperative sampling
hook.

## Supply explicit draws

The compiled host runner samples omitted packets automatically. To choose
the explicit boundary, sample the combined initialization packet outside
compiled or transformed code and supply it for each Population:

```{literalinclude} ../../examples/functional_stochastic_initialization_ordering.py
:language: python
:start-after: "# region draws"
:end-before: "# endregion draws"
:dedent: 8
```

The packet contains parameter-independent base draws and the resulting RNG
carry. Distribution evaluation uses the parameter tensors at the appropriate
side of the pre-initialize transform. Shared RNG streams follow native
declaration order; evaluating transforms and preparing workspaces does not
consume additional random values. Initial NOISE uses the native scaling
timestep of `1.0`.

The example checks the retained distribution parameters explicitly:

```{literalinclude} ../../examples/functional_stochastic_initialization_ordering.py
:language: python
:start-after: "# region ordering"
:end-before: "# endregion ordering"
:dedent: 8
```

RAND's `resample_on_initialize`, `reseed_on_initialize`, and
`reparameterized` settings apply at this sampling boundary. NOISE samples
remain detached from distribution parameters, base draws, and timestep.
Population-owned NOISE is sampled during initialization; inserted owners
also refresh their runtime NOISE before each state transition. See
{doc}`A30_functional_random_parameters` and
{doc}`A31_functional_runtime_noise` for those contracts.

## Differentiate and continue

Select the shared source through either qualified alias:

```{literalinclude} ../../examples/functional_stochastic_initialization_ordering.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 8
```

Pure transform outputs retain their computation graphs. Differentiate with
respect to the input parameter tensors or the transform's explicit inputs;
the returned parameter frame can contain replacements for those input
tensors. A parameter prepared before a later write retains its original
dependencies. Cache-restored operands retain their native detached
boundaries.

Continue with the returned parameter and constant packets as well as runtime
state:

```{literalinclude} ../../examples/functional_stochastic_initialization_ordering.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 8
```

Explicit continuation does not repeat initialization sampling. Scan,
checkpointed scan, and inference-only while execution use the same retained
realizations and explicit RNG carry. Runtime checkpoint commit restores
samples, provenance, NOISE, and RNG into a compatible native destination;
align its raw parameters and initialization separately before continuing.

Snapshot initialization preserves the captured native sample frame. A
Population-owned steady-state cache restores at that Population's position
in Network initialization, unless RAND resampling invalidates it. A
Network-global cache restores its saved state without running fresh
transforms or draws. See {doc}`A29_functional_network_cache_initialization`
and {doc}`A43_functional_initialization_ordering` for cache ownership and
shared deterministic preparation.

```{code-cell} python
:load: ../../examples/functional_stochastic_initialization_ordering.py
:tags: [remove-input]
```
