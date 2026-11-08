(inputs-and-randomness)=
# Stimulation, input tapes and randomness

Use waveforms when you want to describe a stimulus in physical time. Use
`dn.func.Inputs` when your application already has tensors of current or
extracellular voltage, for example the output of a trainable controller.
The same choices work with native and compiled simulations.

## Inject a waveform

`Slice.inject` attaches intracellular stimulation to selected compartments.
The waveform is evaluated on the simulation clock, including continued runs:

```python
import torch
import dendra as dn
from dendra.models.mod import pas
from dendra.units import nA

DT = 0.025  # ms

def make_cell(*, inject=False):
    with dn.ctx(JIT=0, DEVICE="cpu", DTYPE=torch.float64, REQUIRE_GRAD=1):
        cell = dn.SingleCompartment(
            N=1, C=1, v_init=-65.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        cell.insert(pas, g=0.001, e=-65.0)
        if inject:
            cell[:, 0].inject(dn.mono_rect(amp=0.1 * nA, delay=0.05, pw=0.1))
        cell.train()
        cell.initialize()
        return cell

stimulated = make_cell(inject=True)
stimulated.run(duration=0.2, dt=DT)
```

Times are in milliseconds. Intracellular current inputs use absolute current
in mA: `0.1 * nA` supplies the conversion. Mechanism current densities and
point-process currents have different units; see {doc}`../units`.

Native runs and compiled runners sample deterministic registered intracellular
and extracellular waveforms at `start + n * dt`, with `start` taken from that
call's simulation boundary. Internal chunks do not restart this sampling grid.
A continued call starts its grid at the continued clock; a manual `step`
evaluates the drive at its current accepted clock. Custom step callables passed
to low-level scan functions control their own waveform evaluation.

Omitting `Inputs.intra` retains registered `Slice.inject` stimulation.
An explicit intracellular tape conflicts with solver-level registered
injections on that Population. Choose one of those input sources.
Mechanism-owned injections retain their mechanism's own current rule.

## Supply a tensor at each accepted step

Pass `intra=` to `dn.func.Inputs` for intracellular current in mA or `ve=` for
extracellular voltage in mV. A manual `step` takes one frame. A `run` takes a
tape with time on the first axis. A full-size tape has shape
`(steps, *population.shape)`. Supported broadcast dimensions let you share
one drive across cells.

This example applies 0.1 nA to a passive cell for eight steps and records its
voltage. Allocate the tape from `cell.v` so its dtype and device match the model.

```python
cell = make_cell()
steps = 8
current = cell.v.new_full((steps, *cell.v.shape), 0.1 * nA).requires_grad_()
program = dn.compile(
    cell, dt=DT, steps=steps, inference=False,
    callbacks={"trace": dn.Recorder(("v", "t"))}, backend="eager",
)
_, auxiliary = dn.func.run(program, inputs=dn.func.Inputs(intra=current))
voltage = auxiliary["callbacks"]["trace"]["v"]
print(voltage.shape)  # torch.Size([9, 1, 1])
```

The voltage rises from its initial value of −65 mV. The tape has eight input
frames, and the Recorder returns nine voltage samples because it includes the
initial state.

To continue for a requested duration, use `resolve_steps` to find the tape
length from the returned continuation.

```python
continuation = auxiliary["continuation"]
duration = 0.06  # ms
count = dn.func.resolve_steps(program, continuation, duration=duration)
next_current = cell.v.new_full((count, *cell.v.shape), 0.1 * nA)
_, following = dn.func.run(
    program, continuation, duration=duration,
    inputs=dn.func.Inputs(intra=next_current),
)
```

With a 0.025 ms timestep this request takes two steps, so the new recording samples
are at 0.225 and 0.250 ms. The continuation carries the remaining 0.01 ms into
the next duration request. `resolve_steps` counts complete steps, including
any time carried in from previous calls. Run with the original `duration=` to
update that remainder. A `steps=` request leaves it available for a later
duration request.

To fit a stimulus to a target response, enable gradients on the tape and run
with `inference=False`, as above. A loss based on the recorded voltage can then
differentiate with respect to `current`.

```python
voltage[-1].mean().backward()
assert current.grad is not None
```

A Network takes a mapping from Population names to `Inputs`. Allocate each
Population's tape with its own shape, dtype and device. Populations omitted
from the mapping use their configured stimulation:

```python
left = make_cell()
network = dn.Network({"left": left, "right": make_cell()})
network.train()
network.initialize(DT)
net_program = dn.compile(network, dt=DT, steps=8, backend="eager")
initial = dn.func.initialize(net_program)
count = dn.func.resolve_steps(net_program, initial, duration=0.2)
left_current = left.v.new_full((count, *left.v.shape), 0.1 * nA)
net_state, net_auxiliary = dn.func.run(
    net_program, initial, duration=0.2,
    inputs={"left": dn.func.Inputs(intra=left_current)},
)
```

The complete {download}`shared_step_resolution.py
<../../examples/inputs/shared_step_resolution.py>` example also checks native
execution, gradients and checkpoint continuation with these input lengths.

## Extracellular fields and call-time contacts

For `extra=(field, waveform)`, the product `field * waveform(t)` must be in mV.
Analytic lead fields are in mV/mA and take waveforms in mA. A precomputed field
already in mV takes a dimensionless waveform. See {doc}`../units` for other
field conventions.

Bind a contact when compiling if its parameters belong to the model being
optimized. Use `extra=dn.VarInput()` to choose deterministic waveforms and
contact counts on each call.

```python
with dn.ctx(JIT=0, DEVICE="cpu", DTYPE=torch.float64):
    cable = dn.Unmyelinated(
        [2.0], L=3.0, dx=1.0,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    cable.insert(pas)
    cable.initialize()
    variable = dn.compile(
        cable, dt=DT, steps=8, extra=dn.VarInput(), backend="eager"
    )
    field = torch.linspace(-1.0, 1.0, cable.v.numel(), dtype=cable.dtype())
    field = field.reshape(cable.shape)
    pulse = dn.mono_rect(amp=0.1, delay=0.0, pw=0.1)
    sine = dn.sin(amp=0.05, freq=0.1)
    state_a, result_a = variable(extra=(field, pulse))
    state_b, result_b = variable(extra=[(field, pulse), (field.flip(-1), sine)])
```

Both calls restart the runner's snapshot because no continuation was supplied.
Pass a continuation to advance its clock. `extra=None` explicitly selects no
contacts for a variable slot. A Network uses a Population-name mapping of
contact recipes. Supplying an explicit `ve` tape conflicts with bound `extra`
for that Population. Caller-owned fields and waveform parameters can receive
gradients in training execution; see {doc}`training_and_parameters`.

Runtime slots currently support deterministic waveform recipes. Registered
or compile-time-bound Poisson waveforms use their explicit random-state
support. The complete {download}`functional_runtime_extra.py
<../../examples/inputs/functional_runtime_extra.py>` example covers changing contact
counts, gradients and continuation.

## Choose the stochastic model you mean

| Requirement | Mechanism declaration | Gradient behavior |
| --- | --- | --- |
| Sample heterogeneity at initialization | `GLOBALRAND`, `BATCHRAND`, `RANGERAND` | Supported reparameterized distributions preserve gradients through their materialized values. |
| Draw a detached random drive during execution | `GLOBALNOISE`, `BATCHNOISE`, `RANGENOISE` | Transformed noise and its timestep scaling are detached from distribution parameters and base draws. |
| Integrate an SDE with trainable coefficients | `State.DIFFUSION` and Euler–Maruyama or Euler–Heun | Drift and diffusion coefficients retain pathwise gradients. |
| Generate stochastic event times | `NetStim` or a Poisson waveform | Timing and waveform gradients follow the chosen event/schedule model. |

Random draws are sampled automatically when omitted. Checkpoints retain the
logical RNG position, including shared generators. Repeated no-state runner
calls restart their snapshot; continuing a returned packet advances its RNG.
Fresh initialization follows each declaration's resampling and
`reseed_on_initialize` settings. See {doc}`initialization_and_checkpoints` for
fresh trials and cached states.

## Replay the same sampled path

Use `RuntimeDraws` when comparing executions or optimizing against a fixed
sampled path. Capture a tape at the exact starting boundary, then pass it
separately from deterministic `Inputs`:

```python
from dendra.models.mechanisms import Mechanism

class NoisyLeak(Mechanism):
    Mechanism.RANGE(g=0.001, e=-65.0)
    Mechanism.RANGENOISE("eta", mu=0.0, sigma=1e-5, seed=17)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        return self.g, self.g * self.e + self.eta

with dn.ctx(JIT=0, DEVICE="cpu", DTYPE=torch.float64):
    noisy = dn.SingleCompartment(N=1, C=1, integrator=dn.bwd_euler_sc(imem=False))
    noisy.insert(NoisyLeak)
    noisy.initialize()
    stochastic = dn.compile(noisy, dt=DT, steps=8, backend="eager")
    start = dn.func.initialize(stochastic)
    draws = dn.func.sample_runtime_draws(stochastic, start, duration=0.2)
    first, _ = dn.func.run(stochastic, start, duration=0.2, draws=draws)
    replayed, _ = dn.func.run(stochastic, start, duration=0.2, draws=draws)
    torch.testing.assert_close(first, replayed)
```

Sampling leaves the incoming state and generators unchanged. Passing the same
packet at that boundary reproduces the path; execution publishes its RNG
continuation. Native models accept the same `draws=` tapes. Omit both `steps`
and `duration` from the sampler for one manual-step frame; either explicit
horizon produces a time-leading tape. Initialization uses the separate
`InitializationDraws` packet.

See the {doc}`functional API reference <../api/dendra.func>` and the complete
{download}`shared_runtime_draw_horizons.py
<../../examples/stochastic/shared_runtime_draw_horizons.py>` example for replay across
native, compiled and saved execution.
