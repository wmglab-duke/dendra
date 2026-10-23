---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-euler)=
# Forward Euler cable models

`dn.func` and `dn.compile(...)` support `dn.euler`, its `dn.rk1` alias, and
`dn.eulerv1` for scalar `Unmyelinated` and `Myelinated` Populations with standard
geometry. A Network can combine these Populations with Populations using other
supported integrators. Initialization, stimulation, connections, recording,
random draw tapes, and checkpoint continuation use the same functional APIs.
For multistage voltage integration, see {doc}`A52_functional_runge_kutta`.

The complete
{download}`functional_euler.py <../../examples/functional_euler.py>` example
lowers a standalone Population, compares a mixed-integrator Network against
native execution, differentiates
the spatial stencil and a stimulation amplitude, and verifies restart and
continuation. Run it from the repository root with
`python examples/functional_euler.py`.

## Lower a standalone Population

Euler support applies directly to a compatible Population, including a
`Myelinated` subclass such as SMF from `dendra-models`. A Network is not required.
Use `dn.func.make_functional(...)` for explicit parameter, workspace, and state
packets, or pass the Population to `dn.compile(...)` for a rollout runner:

```{literalinclude} ../../examples/functional_euler.py
:language: python
:start-after: "# region standalone"
:end-before: "# endregion standalone"
:dedent: 4
```

## Construct and compile a Network

Choose the integrator while constructing each Population:

```{literalinclude} ../../examples/functional_euler.py
:language: python
:start-after: "# region models"
:end-before: "# endregion models"
:dedent: 8
```

Compile the ordinary initialized Network and qualify recorder variables with
their Population names:

```{literalinclude} ../../examples/functional_euler.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 4
```

`inference=False` selects the differentiable scan execution path. Inference
also supports the shared while-loop engine. The numerical update uses one
current evaluation per timestep, including affine and nonaffine currents.
Mechanism state advances in native order, and current-dependent state observes
that accepted current frame. Optional `i_membrane` recording uses the native
capacitive-plus-ionic current calculation.

Forward Euler is an explicit method: choose a timestep small enough for the
model's spatial and membrane dynamics. Compilation preserves the selected
integrator's equations and does not change its stability restriction. A method
that is cheaper per step can require more steps than an implicit method.

## Trainable stencil and geometry

The raw convolution weight is exposed as `integrator.ssd.weight` in a
Population parameter mapping, or
`populations.<name>.integrator.ssd.weight` in a Network. It retains its native
three-dimensional shape. During training, the spatial stencil is symmetrized;
in evaluation mode, the raw weight is used directly. Select it with
`trainable=...` in `dn.compile`, or set its `requires_grad` flag before lowering.

```{literalinclude} ../../examples/functional_euler.py
:language: python
:start-after: "# region gradients"
:end-before: "# endregion gradients"
:dedent: 4
```

Gradients also pass through capacitance, axial resistivity, cylindrical
geometry, initial voltage, and intracellular or extracellular input tensors.
The workspace uses `diam`, `dx`, `cm`, and `rhoa`, matching native Euler;
`cm_scale`, `rhoa_scale`, and `area_scale` do not enter this integrator's
equations. Parameter sharing is retained when the same raw source is used in
several places.

`dn.eulerv1` uses the same single-stage voltage update as `dn.euler`. Models
using that integrator, including compatible `Myelinated` subclasses, retain
their stored stencil and membrane parameters during lowering. This provides
functional execution for SMF as a standalone Population or within a Network.
Population subclasses must use supported mechanism hooks. Performance depends
on the model, timestep, device,
and execution mode; measure complete rollouts for the intended workload.

## Initialization and continuation

Use `initialization="fresh"` to construct state from the model's initialization
inputs, or `initialization="snapshot"` to restart from the captured state.
Population-owned steady-state caches follow native restoration semantics.
Passing an explicit returned state continues the rollout:

```{literalinclude} ../../examples/functional_euler.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 4
```

NOISE, stochastic state updates, and randomized Poisson stimulation use their
existing explicit draw and RNG carry packets. Replaying a tape or continuing
from a checkpoint preserves the sampling order without advancing the source
model's RNG.

## Currently unsupported

Custom explicit integrator implementations and modified convolution structures
are currently unsupported. Generic morphology-backed `Cable` and `Tree` models
require their supported geometry-aware integrators; the Euler convolution path
does not implement their canonical edge-conductance equations. Multilayer extracellular
models and concatenated `MultiPopulation` models do not use this scalar Euler
path.
