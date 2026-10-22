---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-initialization-sources)=
# Train Ion and Material initial values

Ion concentrations, reversal potentials, and Material fields can use trainable
initializer modules. For example, `PositiveParam` constrains an initial
concentration to a nonnegative value, while a custom module can compute a field
from several parameters. Native initialization and functional fresh
initialization evaluate the same field derivation machinery.

The complete
{download}`functional_initialization_sources.py <../../examples/functional_initialization_sources.py>`
example trains both kinds of source, checks native trajectories, and continues a
Network with a shared initial-concentration parameter. Run it from the repository
root with `python examples/functional_initialization_sources.py`.

## Define an initial-value source

An initializer is an ordinary `torch.nn.Module` whose `forward()` takes no
arguments and returns one tensor. Register trainable values as parameters and
fixed tensor dependencies as buffers:

```{literalinclude} ../../examples/functional_initialization_sources.py
:language: python
:start-after: "# region custom-source"
:end-before: "# endregion custom-source"
```

The result must broadcast to its field's Population shape. Scalar and
compartment-wise initial values follow the same broadcast rules as direct tensor
initial values. The initializer supplies a fresh field value; it does not own the
field's runtime history. Mechanisms and Material processes own subsequent
changes to that field.

Functional fresh initialization requires deterministic, read-only sources.
A source may use its registered parameters, buffers, and
registered child modules. Hidden tensor dependencies, persistent mutation,
random draws, custom module call dispatch, and execution hooks are currently
unsupported. Parameters and buffers remain explicit inputs to functional
execution; changing a source's structure or computation requires lowering the
Population or Network again.

Functional lowering trusts the authored purity contract by default. To run
diagnostic mutation, dependency, and repeated-evaluation checks while creating
the runner, use `with dn.ctx(CHECK_PURITY=True):` around `dn.compile(...)`.
The mode is captured by the new plan; changing the context afterward does not
change an existing plan. Required field, tensor-schema, alias, and lifecycle
checks apply in either mode. Diagnostics can identify violations but do not
prove purity for every possible input.

Source configuration buffers must be independent of runtime state, geometry,
and reconstructed parameter buffers. Share registered raw parameters when a
source and mechanism need the same trainable value.

## Attach sources with the ordinary Population API

This mechanism reads sodium's reversal potential and a generic Material field:

```{literalinclude} ../../examples/functional_initialization_sources.py
:language: python
:start-after: "# region mechanism"
:end-before: "# endregion mechanism"
```

Attach a `PositiveParam` concentration and custom Material initializer before
initializing the Population:

```{literalinclude} ../../examples/functional_initialization_sources.py
:language: python
:start-after: "# region attach"
:end-before: "# endregion attach"
:dedent: 4
```

`population.concentrations(nai0=..., nao0=...)` accepts initial concentration
sources. `population.equilibria(ena=...)` accepts an explicit reversal source;
the Ion style determines whether Nernst evaluation replaces that reversal.
`population.material(..., fields={...})` accepts sources independently for each
field. Existing direct scalar and tensor initial values use the same APIs.

Initialization preserves native ordering. Ion and Material initial fields are
derived first, with the selected Ion style controlling initial Nernst
evaluation. Mechanism initialization then reads that provisional shared frame.
Accepted concentration writes, minimum-value guards, reversal updates, and
current refresh retain their native phases. A mechanism's `initial_values` hook
therefore sees the provisional values available at that phase, which can differ
from the final accepted shared fields.

## Differentiate through fresh initialization

Use `initialization="fresh"` when the run should derive initial runtime state
from the current source parameters. Put the Population or Network in training
mode before lowering to differentiate through these fields; evaluation mode
detaches them at the native shared-field reset boundary:

```{literalinclude} ../../examples/functional_initialization_sources.py
:language: python
:start-after: "# region train"
:end-before: "# endregion train"
:dedent: 4
```

The selected leaves are the initializer's raw parameters: for `PositiveParam`,
`rho` is the unconstrained optimization variable, and its constrained output is
the concentration. Inspect `runner.trainable_parameters` or query individual
leaves with `runner.get_parameter(...)` to verify the selection. Gradients flow
through the source transform, broadcasting, initialization, and subsequent
rollout. Existing minimum guards and bounded-parameter gradient rules apply.
Fresh initialization reads the current raw parameters of bounded sources such
as `PositiveParam`, including in evaluation mode. A transform's previously
cached evaluation result does not replace those current inputs.

Zero-length `torch.vmap` batches over scalar source parameters depend on
PyTorch's operator support. An authored expression that fails when mapped
directly has the same limitation during initialization; empty state batches can
use a shared source.

The default snapshot start uses the Population or Network state captured at
lowering. It does not recompute saved concentrations merely because an
initializer parameter changes. Explicit continuation also starts from the
supplied runtime fields rather than evaluating initializers again. Use fresh
initialization for a new parameter-dependent initial state.

## Share sources and continue a Network

Reuse one source module to share its raw parameters across fields or Populations:

```{literalinclude} ../../examples/functional_initialization_sources.py
:language: python
:start-after: "# region shared"
:end-before: "# endregion shared"
:dedent: 4
```

Each field retains its own runtime value. A shared source has a canonical
parameter name within each Population. Network parameter aliases refer to one
trainable leaf, so gradients from all uses accumulate on the shared parameter.
Deduplicate identical parameter objects when constructing an optimizer from
multiple parameter aliases. Concatenated MultiPopulations retain their
Population regions and native field layouts.

Across Populations, register a shared trainable value as a parameter on each
initializer that uses it, for example with `self.offset = shared_parameter`.
An initializer buffer tied to a raw parameter registered only on another
Population is currently unsupported for fresh initialization. Buffer aliases
of raw parameters registered on the same Population use its parameter binding.

Pass the resulting state, parameter frame, constant frame, and callback state
when continuing a recorded run:

```{literalinclude} ../../examples/functional_initialization_sources.py
:language: python
:start-after: "# region continue"
:end-before: "# endregion continue"
:dedent: 4
```

Native runtime checkpoints preserve the initialized Ion and Material fields.
Restoring a checkpoint and continuing does not rerun initial-value sources.
Population steady-state caches and Network population caches retain their
native restoration policy: cached runtime fields replace fresh initialization,
and cached parameter copies preserve destination parameter identity and shared
alias ordering. See {doc}`A29_functional_network_cache_initialization` for cache
selection and explicit restoration inputs.

```{code-cell} python
:load: ../../examples/functional_initialization_sources.py
:tags: [remove-input]
```
