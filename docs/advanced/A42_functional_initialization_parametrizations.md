---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-initialization-parametrizations)=
# Parametrize Ion and Material initializers

Ion and Material initializer modules can use PyTorch's
`register_parametrization(...)`. Native initialization and functional fresh
initialization evaluate the same source recipe and registered transforms. The
functional parameter frame retains the raw originals, trainable transform
parameters, and their aliases; the effective initial value remains a computed
tensor.

The complete
{download}`functional_initialization_parametrizations.py <../../examples/functional_initialization_parametrizations.py>`
example shares one parametrized source between two Populations, checks values
and gradients against native execution, and continues a checkpointed Network
rollout. Run it with
`python examples/functional_initialization_parametrizations.py`.

## Define and attach a parametrized source

An initial-value recipe is a zero-argument module. Its registered parameter can
have a parametrization with its own trainable parameters and fixed buffers:

```{literalinclude} ../../examples/functional_initialization_parametrizations.py
:language: python
:start-after: "# region recipe"
:end-before: "# endregion recipe"
```

The recipe reads `self.raw`, so its arithmetic receives the transformed value.
The optimization variables are `parametrizations.raw.original` and
`parametrizations.raw.0.factor`. Chained transforms and transforms with multiple
originals follow PyTorch's parameter naming and evaluation order. Bounded Dendra
sources such as `PositiveParam` can also parametrize their registered raw
parameter, `rho`.

Attach the source with the ordinary Population APIs:

```{literalinclude} ../../examples/functional_initialization_parametrizations.py
:language: python
:start-after: "# region attach"
:end-before: "# endregion attach"
:dedent: 4
```

`equilibria(ena=source)` also accepts these recipes. The selected Ion style
controls whether Nernst evaluation replaces a prescribed reversal. Initializer
results follow the field's ordinary broadcast rules. Reuse the same source
module or registered raw `Parameter` to share trainable inputs; each field
retains its own runtime state.

## Select raw parameters and differentiate

Put the model in training mode and use `initialization="fresh"` to differentiate
through the derived initial fields. Wildcard selectors can select the raw
original and transform parameters together:

```{literalinclude} ../../examples/functional_initialization_parametrizations.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 8
```

`runner.trainable_parameters` reports the selected names. `get_parameter(...)`
returns the selected leaf; aliases shared across Populations refer to the same
tensor. Deduplicate identical tensor objects when constructing an optimizer
from multiple aliases. Gradients accumulate across all uses, through both the
parametrization and the initializer computation.

Fresh initialization reads current raw inputs, including when bounded sources
have an evaluation cache. Population evaluation mode retains native detached
initial-field semantics. Functional execution uses isolated module copies and
explicit parameter inputs; it does not assign through `right_inverse` at each
timestep or modify the original model.

## Restart, caches, and continuation

Snapshot initialization restarts the state captured at lowering. Changing a
source parameter does not recompute that snapshot's concentrations or Material
fields. Explicit continuation uses the supplied runtime state, parameter frame,
constant frame, and callback state:

```{literalinclude} ../../examples/functional_initialization_parametrizations.py
:language: python
:start-after: "# region continue"
:end-before: "# endregion continue"
:dedent: 8
```

Native runtime checkpoints preserve initialized fields. Steady-state caches
restore their saved originals and persistent transform buffers in native order,
preserving destination parameter identities and shared aliases. Cached runtime
fields remain detached from the warmup. See
{doc}`A29_functional_network_cache_initialization` for restoration packets and
{doc}`A38_functional_torch_parametrizations` for parametrization cache semantics.

Recipes and transforms must be deterministic and read-only, with tensor
dependencies registered as parameters or buffers. Functional lowering trusts
that contract by default. Use `dn.ctx(CHECK_PURITY=True)` around lowering to run
optional dependency, mutation, and repeated-evaluation diagnostics. Required
ownership, schema, alias, and lifecycle guards apply in either mode. Adding,
removing, or changing a transform's structure requires lowering again.

Hidden mutable dependencies, implicit randomness, custom module call/copy
dispatch, and execution hooks are currently unsupported. Configuration buffers
must be independent of runtime state and reconstructed fields. See
{doc}`A36_functional_initialization_sources` for source ownership and native
initialization ordering.

```{code-cell} python
:load: ../../examples/functional_initialization_parametrizations.py
:tags: [remove-input]
```
