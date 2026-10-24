---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-dufort-frankel)=
# Dufort–Frankel models

`dn.func` and `dn.compile(...)` support `dn.df()` and `dn.dfh()` for canonical
`Population` and `SingleCompartment` containers and scalar `Unmyelinated` and
`Myelinated` Populations with standard geometry. See
{doc}`A55_functional_point_integrators` for point containers and their native
spatial semantics when `C>1`. A compatible
Population can be lowered directly, independently of Networks. Networks can
combine these Populations with other supported integrators. Native and
functional execution share the voltage equations, current evaluation,
initialization, stimulation, and checkpoint machinery.

The complete
{download}`functional_dufort_frankel.py <../../examples/functional_dufort_frankel.py>`
example lowers a standalone Population, differentiates its geometry, compiles
a connected DFH/DF Network, compares native execution, and checks filter,
stencil, and stimulation gradients. Run it from the repository root with
`python examples/functional_dufort_frankel.py`.

## Lower a standalone Population

`dn.df()` derives separate left and right coupling coefficients for
heterogeneous cable geometry. `dn.dfh()` uses the homogeneous cable stencil;
its `conv=False` and `conv=True` modes support functional lowering. Choose the
integrator when constructing the Population:

```{literalinclude} ../../examples/functional_dufort_frankel.py
:language: python
:start-after: "# region standalone"
:end-before: "# endregion standalone"
:dedent: 4
```

Preparation derives physical coefficients from explicit `diam`, `dx`, `cm`,
and `rhoa` tensors. The separate `area_scale`, `cm_scale`, and `rhoa_scale`
modifiers do not enter these integrators' equations. Effective material values
and geometry transforms retain their ordinary native meaning. Geometry stays
differentiable:

```{literalinclude} ../../examples/functional_dufort_frankel.py
:language: python
:start-after: "# region geometry"
:end-before: "# endregion geometry"
:dedent: 4
```

Both methods support singleton compartments and explicit leading batch axes.
A sealed singleton has no axial contribution, including during startup.

## Voltage history and timestep ordering

Dufort–Frankel retains two voltage levels. Functional state includes the
current voltage `state["integrator"]["v"]`, the previous voltage `v_prev`, and
explicit history-validity and timestep tensors. Keep the complete returned
state when continuing a simulation; copying only `v` discards information
required by the next step.

Without a restored cache, initialization starts without valid history. The
first timestep constructs it with one forward-Euler step of length `dt`.
Later timesteps use the two-level Dufort–Frankel update. Rebuilding coefficients
at the same timestep preserves valid history. A changed timestep invalidates the old history and
uses the Euler starter before resuming the two-level scheme. Snapshot restart,
fresh initialization, steady-state caches, and checkpoints preserve these
native lifecycle rules.

Mechanisms advance once per timestep in their pre-current and post-current
phases. Currents proven or declared affine in voltage use centered Dufort
conductance; other currents use their explicit current with zero Dufort
conductance. Current-dependent mechanisms receive the native accepted ionic
current frame. Runtime NOISE, stochastic state equations, Poisson stimulation,
and connections retain their native sampling and delivery order. The Euler
starter does not introduce an additional mechanism advance or random draw.

With `beta < 1`, spatial smoothing applies after each voltage update,
including the Euler starter. The `smooth_every` argument does not alter this
schedule. `beta=1` disables smoothing. The native smoothing-filter weights are
available as `integrator.filter.weight`; DFH's convolution stencil is
`integrator.ssd.weight` when using `conv=True`. Training symmetrizes that
stencil, while evaluation uses its raw weight. DFH's direct mode does not use
the convolution stencil.

The diffusion scheme's linear stability property does not ensure accuracy or
stability for arbitrary membrane dynamics, stimulation, or learned filters.
Choose and validate an appropriate timestep for the complete model.

## Compile a Network

Ordinary Network construction supports mixed integrators and continuous or
event-driven connections. This example connects homogeneous and
heterogeneous Dufort–Frankel Populations:

```{literalinclude} ../../examples/functional_dufort_frankel.py
:language: python
:start-after: "# region models"
:end-before: "# endregion models"
:dedent: 8
```

Record variables and select trainable parameters using Population-qualified
names:

```{literalinclude} ../../examples/functional_dufort_frankel.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 4
```

`inference=False` uses differentiable scan execution. Inference supports the
shared scan and while-loop engines. The complete voltage history participates
in scan carry and checkpoint recomputation. Gradients pass through physical
coefficients, supported currents, stimulation, connection parameters, the
convolution stencil, and the smoothing filter:

```{literalinclude} ../../examples/functional_dufort_frankel.py
:language: python
:start-after: "# region gradients"
:end-before: "# endregion gradients"
:dedent: 4
```

## Initialization and continuation

`initialization="snapshot"` restarts from the captured state, including any
valid voltage history. `initialization="fresh"` initializes from current
parameters and initialization inputs, respecting Population-owned caches.
Calls without an explicit state restart from the selected initialization.
Pass the complete returned state to continue:

```{literalinclude} ../../examples/functional_dufort_frankel.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 4
```

When fresh initialization returns changed effective parameters or constants,
also pass those returned packets on continuation as described in
{doc}`A21_functional_network_compile`. `commit_state_` transfers the complete
functional state to a compatible native source for ordinary checkpoint saving
and native continuation. Pure execution leaves the source model and its RNG
streams unchanged.

## Currently unsupported

Custom Dufort–Frankel integrators and modified convolution structures are
currently unsupported. Generic morphology-backed `Cable` and `Tree` models
require their supported geometry-aware integrators. Multilayer extracellular
and concatenated `MultiPopulation` models do not use this scalar cable path.

## Run the complete example

```{code-cell} python
:load: ../../examples/functional_dufort_frankel.py
:tags: [remove-input]
```
