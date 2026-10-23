---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-runge-kutta)=
# Runge–Kutta cable models

`dn.func` and `dn.compile(...)` support `dn.rk2` and `dn.rk4` for scalar
`Unmyelinated` and `Myelinated` Populations with standard geometry. A compatible
Population can be lowered directly or used in a Network with other supported
integrators. Both execution styles use the native voltage-stage kernels and
the shared initialization, stimulation, recording, and checkpoint machinery.

The complete
{download}`functional_runge_kutta.py <../../examples/functional_runge_kutta.py>`
example lowers a standalone RK2 Population, differentiates its geometry,
compiles a connected RK2/RK4 Network, compares its state with native execution,
and checks gradients and continuation. Run it from the repository root with
`python examples/functional_runge_kutta.py`.

## Lower a standalone Population

Choose `dn.rk2()` or `dn.rk4()` when constructing the Population. Use
`dn.func.make_functional(...)` to expose explicit parameters, workspace, and
state, or `dn.compile(...)` to create a rollout runner:

```{literalinclude} ../../examples/functional_runge_kutta.py
:language: python
:start-after: "# region standalone"
:end-before: "# endregion standalone"
:dedent: 4
```

Preparation derives the cylindrical cable coefficients from the explicit
`diam`, `dx`, `cm`, and `rhoa` tensors. As with native Euler, the separate
`area_scale`, `cm_scale`, and `rhoa_scale` modifiers do not enter these
integrators' equations. Geometry inputs remain differentiable:

```{literalinclude} ../../examples/functional_runge_kutta.py
:language: python
:start-after: "# region geometry"
:end-before: "# endregion geometry"
:dedent: 4
```

## Voltage stages and native timestep ordering

RK2 uses the midpoint voltage update. RK4 uses the classical four-stage
weighted voltage update. Each stage reevaluates axial and ionic currents at
its predicted voltage, while retaining the timestep's mechanism state,
intracellular/extracellular drives, and connection inputs. These are
voltage-stage methods: their midpoint and fourth-order formulas do not imply
second- or fourth-order integration of the full coupled system of voltage and
mechanism states.

Mechanisms advance once per native timestep in their declared pre-current and
post-current phases. RK2 accepts the midpoint ionic-current frame. RK4 accepts
the weighted frame `(i1 + 2*i2 + 2*i3 + i4) / 6`; current-dependent mechanisms
and shared ion fields receive that accepted frame. Optional `i_membrane`
recording combines the capacitive current with the corresponding accepted
ionic current.

Runtime NOISE and stochastic state updates retain their native draw ordering
and explicit draw/RNG packets. Extra voltage stages do not introduce extra
noise draws or additional mechanism advances. Connections and stimulation
retain their native timestep timing; the stages do not deliver additional
events or resample a waveform at intermediate times.

Both integrators are explicit. Choose a timestep suitable for the membrane
and spatial dynamics. More stages per step do not remove the cable diffusion
stability restriction, and compilation preserves the selected equations.

## Compile a Network

A Network can mix RK2 and RK4 Populations, or combine them with other admitted
integrators. Construct connections using the ordinary Network API:

```{literalinclude} ../../examples/functional_runge_kutta.py
:language: python
:start-after: "# region models"
:end-before: "# endregion models"
:dedent: 8
```

Qualify recorder variables and trainable parameters with their Population
names:

```{literalinclude} ../../examples/functional_runge_kutta.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 4
```

`inference=False` uses differentiable scan execution. Inference supports the
shared scan and while-loop engines. Gradients pass through every voltage
stage, connection weights and delays, waveform parameters, initial voltage,
and prepared physical coefficients. Raw parameter aliases shared by several
Populations retain their sharing.

The native spatial stencil is exposed as `integrator.ssd.weight` for a
standalone Population, or `populations.<name>.integrator.ssd.weight` in a
Network. Training mode symmetrizes it; evaluation mode uses the raw weight.
The example differentiates through the connected Network to the source
Population's stencil and stimulation amplitude:

```{literalinclude} ../../examples/functional_runge_kutta.py
:language: python
:start-after: "# region gradients"
:end-before: "# endregion gradients"
:dedent: 4
```

## Initialization and continuation

Use `initialization="fresh"` to initialize from current parameters and
initialization inputs, or `initialization="snapshot"` to restart from the
captured initialized state. Population-owned steady-state caches use native
restoration semantics. Each call without an explicit state restarts from the
selected initialization; passing the returned state continues execution:

```{literalinclude} ../../examples/functional_runge_kutta.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 4
```

Checkpoint recomputation uses the same stage equations, accepted current
frames, and explicit RNG carry. It preserves gradients without advancing the
source Population or Network. For fresh initialization that changes an
effective parameter or constant packet, also carry the returned packets as
described in {doc}`A21_functional_network_compile`.

## Currently unsupported

Custom explicit integrators and modified convolution structures are currently
unsupported. Generic morphology-backed `Cable` and `Tree` models require
their supported geometry-aware integrators; the RK2/RK4 convolution path does
not implement their canonical edge-conductance equations. Multilayer
extracellular and concatenated `MultiPopulation` models do not use this scalar
path. Dufort–Frankel voltage integration is currently unsupported by
functional lowering.
