(custom-models)=
# Custom models and explicit functional execution

Build custom dynamics with the same Mechanism and State declarations used by
native models. Pure hooks describe computations by returning values, which
lets Dendra use them in ordinary execution, compiled simulations and autograd.
You do not need to implement a separate functional neuron or Network.

## Give each value a lifetime

| Value | Where it belongs |
| --- | --- |
| User-selected numerical parameter | `GLOBAL`, `BATCH` or `RANGE`. |
| Differential-equation state | `State.STATE`, with `DERIVATIVE` or `KINETIC` equations. |
| Persistent counter, latch or history | `CARRY`, updated through a pure transition or receive hook. |
| Repeatable algebra for the current solver stage | `ASSIGNED`, returned by `assigned_values(v, values)`. |
| Coefficient derived from parameters, temperature or geometry | `DERIVED_BUFFER`, returned by `derive_buffers()`. |
| Coefficient that also depends on the solver timestep | `TIMESTEP_BUFFER`, returned by `derive_timestep_buffers(dt)`. |
| Read-only input to a State equation | `State.INPUT`, connected with the Population input-binding API. |

Do not use a buffer or `ASSIGNED` value as an undeclared timer. Current and
assigned-value methods can be evaluated several times during one accepted
step. Carry is the place for values that must survive steps and checkpoints.

## Implement one mechanism for both execution styles

This small example combines a solver-evolved gate with a step counter. The
counter is illustrative; it is not part of the current equation:

```python
import torch
import dendra as dn
from dendra.models.mechanisms import Mechanism, State

class Activation(State):
    State.STATE("x")
    State.RANGE(tau=2.0)
    State.DERIVATIVE("x' = (1.0 - x) / tau")

    def state_defaults(self, v, values):
        return {"x": torch.zeros_like(v)}

class GatedLeak(Mechanism):
    Mechanism.STATE_BUNDLE(Activation)
    Mechanism.RANGE(g=0.001, e=-65.0)
    Mechanism.CARRY("steps_seen")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def initial_values(self, v, values):
        return {"steps_seen": torch.zeros_like(v)}

    def advance(self, v, dt, values):
        return {"steps_seen": values["steps_seen"] + 1}

    def i_affine_terms(self, v):
        conductance = self.g * self.x
        return conductance, conductance * self.e

with dn.ctx(JIT=0, DEVICE="cpu", DTYPE=torch.float64, REQUIRE_GRAD=1):
    cell = dn.SingleCompartment(
        N=1, C=1, v_init=-55.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    cell.insert(GatedLeak.rename("gate"))
    cell.train()
    cell.initialize()
    program = dn.compile(
        cell, dt=0.025, steps=4, inference=False, trainable="*g_param",
        callbacks={"trace": dn.Recorder(("v", "gate.x", "gate.steps_seen"))},
        backend="eager",
    )
    state, auxiliary = dn.func.run(program)
    assert torch.isfinite(state["integrator"]["v"]).all()
```

`State.advance` normally executes the declared State equations. Override it
only when supplying a custom solver-state transition. `Mechanism.advance`
runs once per accepted step and returns a partial mapping of direct carry
and owned writable shared fields; omitted carry is retained. The two hooks
read the same incoming snapshot. A Mechanism hook cannot assume that a nested
State has already advanced.

During initialization, State defaults provide fallback values, insertion-time
`ic` overrides those defaults, and authored `initial_values` hooks may overlay
the initialized values. Initialization does not perform an accepted step.
`derive_buffers` and `derive_timestep_buffers` prepare reproducible
coefficients rather than evolving history. See the
{doc}`mechanism API reference <../api/dendra.models.mechanisms>` for exact
input/output and initialization contracts.

## Current equations, units and purity

An affine current can supply `(G, J)` through its named coefficient hook,
representing `I(v) = G*v - J`. A current named `ik` uses `ik_affine_terms`.
Other mechanisms can implement their ordinary current equation, or an exact
`i_with_conductance(v)` pair when appropriate. `NUMERICAL` explicitly opts into
finite-difference conductance for a local, deterministic current; it is not
needed for the affine example above.

Distributed currents use mA/cm² and conductances use S/cm². A `PointProcess`
uses bare nA and µS within its current methods. Do not multiply a point-synapse
weight of `0.05` µS by `dn.units.uS`. See {doc}`../units` before mixing
distributed currents, point processes and absolute stimulus currents.

Treat hook arguments and module state as read-only. In hooks that receive
`values`, read evolving values from that frame. Current methods read their
stage-bound declared state, parameters and prepared coefficients through
`self`. Return new tensors with the required shapes, dtype and device.
Do not consume hidden RNG or capture unregistered changing tensors in closures.
The {doc}`inputs_and_randomness` guide describes explicit stochastic inputs.

To diagnose purity violations during setup, construct a plan inside
`with dn.ctx(CHECK_PURITY=True):`. These optional diagnostics add setup cost and
can detect violations, but do not prove arbitrary code pure. Ordinary execution
trusts the authored purity contract; required structural/schema checks remain.

## Synapses, voltage processes and material processes

- **Event-driven synapses:** implement `Synapse.net_receive_values` to return
  State/carry replacements from a weighted delivery. Register the synapse and
  connections with the ordinary Network API; the standard adapter handles
  native writes. The {doc}`synapse tutorial <../synapse_authoring>` provides
  complete event-driven and continuous examples.
- **Continuous synapses:** declare analog `INPUT` fields and a current rule.
  Networks reset and reduce incoming connections each step. Stateless source
  functions and modules can transform the source; module parameters can be
  trained. Sigmoid release is one example of such a transform.
- **Voltage processes:** implement `VoltageProcess.voltage_values` to return
  the new voltage and supported State/carry updates. Use it for a direct
  voltage transition rather than an additional membrane current.
- **Material processes:** declare `material_fields`, build coefficients with
  `prepare_materials`, and return full-field replacements from
  `material_updates`. These are separate from Mechanism `advance` and carry
  phases. The complete {download}`functional_material_processes.py
  <../../examples/functional_material_processes.py>` example implements a
  conservative exchange process.

Lookup tables, parameterized initializers, registered PyTorch parametrizations,
mechanism delay histories and owned waveform injections retain their declared
roles in compiled execution. Use their public declarations and hooks from the
API reference; a second implementation for `dn.func` is unnecessary.

## Work with explicit plans when you need control

`dn.compile` is the usual starting point. Explicit lowering is useful when
you need to manage raw sources, compose pure tensor functions, or bind one
kernel to several parameter frames:

```python
functional, tensors = dn.func.make_functional(cell, dt=0.025)
initial = dn.func.initialize(functional, tensors)
kernel = functional.compile_rollout_chunk(2, backend="eager")
final, result = dn.func.run(kernel, initial, steps=4)
```

The plan describes the model; the tensor bundle contains parameters, constants,
state and, for a Population, its initialization recipe. A `Continuation` carries
effective sources and optional callback carry between calls. Raw `bind` forms
can retain a chosen source frame. Rebind when portable restoration creates new
parameter leaves. Advanced explicit prepared bindings have a shorter lifetime:
recreate their preparation for independent backward passes after parameters
change.

Population extraction exposes native parameter objects; Network extraction
creates detached snapshots that preserve sharing. Use `independent_parameters`
or the owned capture/checkpoint APIs when you want independent optimization
leaves. See {doc}`training_and_parameters` and the
{doc}`functional API reference <../api/dendra.func>` for these ownership rules.

Host `run`, `initialize`, sampling and checkpoint functions are not
`torch.func` transforms. For `vmap`, forward AD or higher derivatives, compose
the pure plan preparation/transition methods with explicit tensors and draws.
Supported transform combinations depend on the model and tensor operations;
compiled first-order scan and inference-only while have their own execution
contracts. The complete {download}`functional_api_overview.py
<../../examples/functional_api_overview.py>` example includes an eager
`torch.func.grad`/`vmap` comparison.

## Supported model boundaries

Canonical point, cable, tree, extracellular and concatenated Populations, and
compatible constructor-defined subclasses, can be lowered. Networks currently
require a canonical single-device execution structure. Integrator availability
depends on geometry, dtype, device and extracellular layout; use the
{doc}`integrator reference <../api/dendra.models.integrators>` to choose one.

Arbitrary native stepping/lifecycle overrides, hidden mutable hook state and
implicit random sampling inside pure hooks are currently unsupported. Pure
initialization transforms provide declared initialization-time reads and writes
in place of arbitrary native pre/post hooks. Structural model changes after
lowering require a new plan or runner. Reusing a continuation changes its
explicit simulation state; it does not rebuild the model's structure.

## Devices and specialized recipes

Construct or move the native model to its execution device before initializing
and compiling it. CPU and CUDA support float32 and float64; MPS requires float32.
Spatial cable/tree integrators require a compatible `dendra-solvers` installation.
Choose a solver supported on the device; see the
{doc}`integrator reference <../api/dendra.models.integrators>`.

These complete examples show additional model components without requiring a
different simulation lifecycle:

| Modeling task | Runnable example |
| --- | --- |
| Train SDE drift or diffusion along a sampled path | {download}`functional_state_sde.py <../../examples/functional_state_sde.py>` |
| Learn a tabulated nonlinear response | {download}`functional_tables.py <../../examples/functional_tables.py>` |
| Retain single-stream or multistream mechanism delays | {download}`functional_mechanism_delays.py <../../examples/functional_mechanism_delays.py>` |
| Give a mechanism its own waveform injection | {download}`functional_mechanism_injections.py <../../examples/functional_mechanism_injections.py>` |
| Train constrained Ion or Material initial values | {download}`functional_initialization_sources.py <../../examples/functional_initialization_sources.py>` |
| Constrain parameters with PyTorch parametrizations | {download}`functional_torch_parametrizations.py <../../examples/functional_torch_parametrizations.py>` |
| Define an explicit-draw connection distribution | {download}`functional_distribution_sampling.py <../../examples/functional_distribution_sampling.py>` |
