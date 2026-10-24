---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-population-subclasses)=
# Define model subclasses

Model subclasses can assemble mechanisms, set constructor defaults, declare
parameters and random sources, and provide convenience methods while using
Dendra's inherited functional execution. This applies to `Population`,
`SingleCompartment`, `Cable`, `Tree`, `Unmyelinated`, `Myelinated`,
`ExtCellAxon`, `ExtCellTree`, and `MultiPopulation`, and to `Network`.
Each class retains the numerical integrators and topology requirements of its
supported base family.

The complete
{download}`functional_population_subclasses.py <../../examples/functional_population_subclasses.py>`
example checks native parity across point, Cable, Tree, and extracellular
classes, then compiles a connected Network with shared trainable parameters.
Run `python examples/functional_population_subclasses.py` from the repository
root.

## Assemble a reusable model

Constructors and additional helper methods can be inherited through multiple
levels or mixins:

```{literalinclude} ../../examples/functional_population_subclasses.py
:language: python
:start-after: "# region model_classes"
:end-before: "# endregion model_classes"
```

Use the inherited morphology factories for Cable and Tree families, for
example `PassiveCable.from_morphology(morphology, N=1)` or
`PassiveTree.from_morphology(morphology, N=1)`. Canonical graph geometry,
boundary conditions, and support maps remain the source of truth.
Extracellular families retain their full circuit state and supported layer
configuration.

Insert mechanisms, configure parameters, and register supported pure
initialization transforms during construction or before lowering. Initialize
the model, then use `dn.func.make_functional(...)` or `dn.compile(...)` with the
same arguments as for its base family. Constructor execution is not part of a
compiled timestep or fresh functional initialization.

## Assemble a Network

A Network subclass can create its populations and connections in its
constructor:

```{literalinclude} ../../examples/functional_population_subclasses.py
:language: python
:start-after: "# region network_class"
:end-before: "# endregion network_class"
```

The inherited Network phase ordering, source bindings, and pure projections
are used during execution:

```{literalinclude} ../../examples/functional_population_subclasses.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 4
```

Shared raw parameters keep their qualified aliases and gradient history:

```{literalinclude} ../../examples/functional_population_subclasses.py
:language: python
:start-after: "# region gradients"
:end-before: "# endregion gradients"
:dedent: 4
```

## Retain the execution contract

A compatible subclass inherits its base family's numerical, geometry,
parameter materialization, RNG, initialization, and checkpoint hooks.
Overriding these hooks, including a wrapper that calls `super()`, is currently
unsupported. This includes helpers and optional callbacks invoked during
execution or initialization, such as a Population-level `clear_cache` method.
Dynamic attribute access and descriptors that replace runtime fields also
change this contract.

Define mechanism dynamics through the pure Mechanism/State hooks rather than
overriding `Population.step`. Use registered pre/post initialization transforms
for initialization-time tensor changes rather than overriding `initialize` or
legacy initialization hooks. Optional `dn.ctx(CHECK_PURITY=True)` diagnostics
remain separate from these structural compatibility checks.

For packed models, the packed solver determines numerical execution;
concatenation does not retain each component's standalone integrator.
Compatible subclasses can be components or packed containers, subject to the
same packing restrictions as their base families.

## Initialize and continue

Snapshot and fresh initialization, steady-state caches, explicit random draws,
RNG carry, recording, and checkpoint replay use the existing APIs.
`initialization="fresh"` runs the shared pure initialization transaction for
the current parameter frame. It does not invoke the authored constructor.

Carry all returned state, effective parameters/constants, and callback state
when continuing:

```{literalinclude} ../../examples/functional_population_subclasses.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 4
```

The source model is unchanged. Population extraction and `commit_state_`
retain the concrete class and structural compatibility requirements; a
different subclass is not automatically interchangeable because it inherits
the same base. Changes to execution hooks or structure require rebuilding and
lowering again. Subclass support does not add support for new geometries,
custom numerical integrators, or additional devices.
