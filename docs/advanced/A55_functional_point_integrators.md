---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-point-integrators)=
# Choose integrators for Population and SingleCompartment

`dn.func` and `dn.compile(...)` support `dn.euler`/`dn.rk1`, `dn.eulerv1`,
`dn.rk2`, `dn.rk4`, `dn.df`, and `dn.dfh` for canonical `Population` and
`SingleCompartment` containers. These methods use the same pure numerical
kernels as native execution and the supported scalar cable models. A Network
can combine them with other admitted integrators.

Use `N=count, C=1` for independent single-compartment cells. With `C>1`, these
explicit and Dufort–Frankel methods retain their native spatial coupling along
the final compartment axis, including reflected boundaries. Each row and
explicit leading `Population.batch()` lane retains its own voltage state.
The selected integrator determines these spatial semantics; changing from
`bwd_euler_sc` can therefore change the model's coupling as well as its time
integration.

The complete
{download}`functional_point_integrators.py <../../examples/functional_point_integrators.py>`
example lowers a spatially coupled `Population`, compiles a connected
RK4/DFH Network, checks native parity and shared gradients, and continues a
checkpointed rollout. Run it from the repository root with
`python examples/functional_point_integrators.py`.

## Lower a Population directly

Choose the integrator during construction, insert mechanisms, and initialize
the Population before lowering it:

```{literalinclude} ../../examples/functional_point_integrators.py
:language: python
:start-after: "# region standalone"
:end-before: "# endregion standalone"
:dedent: 4
```

`make_functional` exposes the same explicit parameter, constant, and state
packets as for other supported Populations. Current evaluation, accepted
ionic-current frames, optional `i_membrane`, and mechanism updates retain
their native ordering.

Euler uses one voltage-current evaluation. RK2 uses a midpoint evaluation and
RK4 uses the weighted four-stage frame. Mechanism states advance once per
timestep; additional voltage stages reuse that state and the accepted step's
stimulation and connection inputs. These voltage-stage orders do not imply
the same integration order for the complete coupled model. See
{doc}`A51_functional_euler` and {doc}`A52_functional_runge_kutta`.

## Compile a connected Network

This example uses `C=1` for both point containers and shares one raw leak
parameter:

```{literalinclude} ../../examples/functional_point_integrators.py
:language: python
:start-after: "# region models"
:end-before: "# endregion models"
:dedent: 8
```

Compile the initialized Network with qualified recorders and parameter
selectors:

```{literalinclude} ../../examples/functional_point_integrators.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 4
```

`inference=False` uses differentiable scan execution, including optional
checkpoint replay. Inference supports both scan and `execution="while"`.
The source Population or Network stays unchanged. Shared parameters retain
their identity across Population-qualified names:

```{literalinclude} ../../examples/functional_point_integrators.py
:language: python
:start-after: "# region gradients"
:end-before: "# endregion gradients"
:dedent: 4
```

Gradients pass through mechanism and connection parameters, stimulation,
initial voltage, and the native prepared physical coefficients. The spatial
stencil is `integrator.ssd.weight` where the method uses a convolution;
training symmetrizes it and evaluation uses the raw weight. DF/DFH smoothing
uses `integrator.filter.weight`. Derivatives retain native behavior for
singletons, including quantities that have no influence on a sealed cell.

## Initialization, randomness, and continuation

`initialization="fresh"` builds state from current parameters and initialization
inputs, respecting Population-owned and Network caches. Snapshot initialization
restarts the captured state. Pass the complete returned state and any effective
parameter/constant packets to continue:

```{literalinclude} ../../examples/functional_point_integrators.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 4
```

DF and DFH state includes `v_prev`, `_df_history_valid`, and `_df_history_dt`.
Initialization without restored history uses one Euler startup step; changing
the timestep invalidates the old history. Preserve all of these fields through
continuation and checkpoints. Smoothing, when enabled, applies with the native
schedule. See {doc}`A53_functional_dufort_frankel` for the two-level equations
and history lifecycle. A compiled Network fixes its timestep; reinitialize and
build a compatible runner when changing it.

RAND, runtime NOISE, State `DIFFUSION`, NetStim, and stochastic stimulation use
the existing explicit draw and RNG packets. Sampling follows native phase
ordering, with one accepted step's state/noise draws reused across RK voltage
stages. Replay and checkpoint continuation retain those RNG boundaries without
advancing the source model. See {doc}`A31_functional_runtime_noise`,
{doc}`A32_functional_state_sde`, and {doc}`A47_functional_network_rng_sharing`.

Euler and RK methods retain their explicit stability restrictions. DF/DFH's
diffusion stability does not ensure accuracy for arbitrary membrane dynamics
or learned stencils. Choose a timestep appropriate for the complete model.
Custom numerical integrators, modified convolution structures, and packed
`MultiPopulation` execution with these point-integrator profiles are currently
unsupported.
