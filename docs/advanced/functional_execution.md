(functional-execution)=
# Run models with explicit state

Use `dn.compile` to turn an initialized Population or Network into a reusable
program for inference or training. Each run returns `(state, auxiliary)`.
`state` holds the final simulation tensors, and `auxiliary["continuation"]`
holds what you need to continue the run.

Use native `model.run(...)` when you want the model itself to hold the evolving
state. Both interfaces accept deterministic inputs and explicit random draws,
and support the same recording and analysis configurations. See how to
{doc}`transfer a running simulation between them <initialization_and_checkpoints>`.

## Start with one Population

This example injects a 0.1 nA pulse into a passive compartment and records its
voltage for 1 ms. It uses `backend="eager"` for quick startup. Omitting
`backend` selects the default optimizing backend.

```python
import torch
import dendra as dn
from dendra.models.mod import pas
from dendra.units import nA

DT = 0.025  # ms
with dn.ctx(DEVICE="cpu", DTYPE=torch.float64, JIT=0):
    model = dn.SingleCompartment(
        N=1, C=1, v_init=-65.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    model.insert(pas, g=0.001, e=-70.0)
    model[..., 0].inject(dn.mono_rect(amp=0.1 * nA, delay=0.1, pw=0.2))
    model.initialize()
    original = model.v.detach().clone()

    program = dn.compile(
        model, dt=DT, steps=40, chunk_steps=8,
        callbacks={"trace": dn.Recorder(("v", "t"))},
        backend="eager",
    )
    state, auxiliary = dn.func.run(program)
    trace = auxiliary["callbacks"]["trace"]
    print(trace["t"].shape, trace["v"].shape)
    torch.testing.assert_close(model.v, original)
```

Read the named Recorder's results at `auxiliary["callbacks"]["trace"]`.
Both the time and voltage tensors have 41 samples along their first axis,
one at initialization and one per step.

Call `program()` or `dn.func.run(program)` to repeat the trial from the
snapshot taken at compilation. Pass the continuation to run another 40 steps
from the end of that trial.

```python
# Continue the previous example for another 40 accepted steps.
state, following = dn.func.run(program, auxiliary["continuation"])
```

The continuation carries simulation state, random stream positions, and the
effective parameters and constants. It also keeps callback state such as a
running spike count or the Recorder's sampling position. The continued trace
starts one step after the first trace ends. See
{doc}`initialization_and_checkpoints` for fresh trials and saving a continuation.

## Compile a connected Network

Build and connect a Network as you would for native execution. In this example,
a NetStim drives the source Population. Its upward crossing of −60 mV sends an
event to the target Population, raising the target's voltage.

```python
import torch
import dendra as dn
from dendra.models.mod import expsyn, pas

DT = 0.025  # ms
with dn.ctx(DEVICE="cpu", DTYPE=torch.float64, JIT=0):
    def cell():
        value = dn.SingleCompartment(
            N=1, C=1, v_init=-65.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        value.insert(pas, g=0.001, e=-65.0)
        value.insert(expsyn.rename("exc"), tau=2.0, e=0.0)
        return value

    source, target = cell(), cell()
    stimulus = dn.NetStim(N=1, start=0.0, interval=0.5, noise=0.0, seed=17)
    network = dn.Network({"source": source, "target": target}, netstim=stimulus)
    network.connect(
        stimulus[:], source[:], source.mech.exc, conn_spec="one_to_one",
        threshold=None, weight=0.1, delay=DT,
    )
    network.connect(
        source[:], target[:], target.mech.exc, conn_spec="one_to_one",
        threshold=-60.0, weight=0.1, delay=0.1,
    )
    network.eval()
    network.initialize(DT)
    program = dn.compile(
        network, dt=DT, steps=80, chunk_steps=16,
        callbacks={"trace": dn.Recorder(("source.v", "target.v", "t"))},
        backend="eager",
    )
    state, auxiliary = dn.func.run(program)
    trace = auxiliary["callbacks"]["trace"]
    print(trace["source.v"].shape, trace["target.v"].shape)
    assert trace["target.v"].max() > trace["target.v"][0].max()
```

Use the same timestep for `network.initialize(DT)` and `dn.compile(..., dt=DT)`.
The Recorder uses Population names in its field paths and returns 81 samples
for each voltage trace. Event weights take their units from the receiving
mechanism. Both `expsyn` weights above are 0.1 µS.

## Choose the amount of simulation

Set `steps=N` to run exactly N solver steps or `duration=T` to request T ms of
simulation time. Choose one when compiling to set the default run length.
Individual calls can request a different length.

```python
# Use the program from either example above.
state, auxiliary = dn.func.run(program, steps=10)
state, auxiliary = dn.func.run(
    program, auxiliary["continuation"], duration=0.25,
)
```

Duration requests carry time shorter than one timestep into the next continued
call. With `dt=0.1` ms and no pending time, a request of `duration=0.25` ms
advances two steps and carries 0.05 ms. Passing its continuation into a second
0.25 ms run advances three steps. An exact step request leaves the pending
fraction available for a later duration request.

Explicit input and draw tapes must contain the accepted number of steps.
Use `dn.func.resolve_steps` to size a tape for a duration request. See
{doc}`inputs_and_randomness` for an example.

## Choose inference or training

Compiled runs default to `inference=True` and run without gradients. For
optimization, set `inference=False` and choose parameters with `trainable=`.
Select the model's own training or evaluation behavior with `model.train()` or
`model.eval()` before initializing and compiling. See
{doc}`training_and_parameters` for a complete fitting example.

The default `execution="scan"` supports training and inference. For inference
you can also choose `execution="while"` for Populations and Networks, including
those with custom pure callbacks. A Population while runner must be compiled
for a chunk of at least four steps. See {doc}`custom_models` for supported
models and authoring requirements.

Use `chunk_steps` to limit the number of steps in each compiled chunk. For
example, a 10-step run with `chunk_steps=4` runs two 4-step chunks and a 2-step
chunk. A new final chunk length can require compilation on its first use.
Smaller chunks can reduce temporary memory. In-memory recordings still retain
the samples requested for the full run.

Create a new program after changing model structure or connectivity.

## Use ordinary connection and scheduling APIs

Choose the connection interface that matches the signal you want to deliver.

| What you want to model | Interface |
| --- | --- |
| Discrete events from a NetStim or voltage crossings | `network.connect(...)` with an event-driven target such as `expsyn`. |
| A continuous signal between Populations | `network.connect_continuous(...)` with a receiving input, optional deterministic function or module, weight and delay. |
| Predetermined event times and amplitudes | `NetStim.schedule(...)` or the connection's `schedule(...)`. |

Supply event times and amplitudes as values or indices into a tensor. To
optimize a stimulation protocol, use `bind_time_source(times)` and
`bind_weight_source(amplitudes)`, then schedule with `time_idx=` and
`weight_idx=`. Each run reads the current indexed values. See the complete
{download}`shared_schedule_bindings.py <../../examples/inputs/shared_schedule_bindings.py>`
example for native and functional training. Continuous transforms can contain
trainable parameters, so an optimizer can fit a release curve or another
conversion of the presynaptic signal.

Configure delays and interpolation on the Network before compiling. See the
{doc}`Network API <../api/dendra.models.networks>` for connection and scheduling
arguments and {doc}`inputs_and_randomness` for replaying stochastic inputs.

## Warm a program before its first timed call

The compiler prepares kernels on the first call. Use `warm()` to do that work
before timing or serving a run. It runs a representative simulation, discards
the results and returns the same program with its compiled kernels ready.
This creates and warms a program for the Population from the first example.

```python
program = dn.compile(model, dt=DT, steps=40).warm(verbose=True)
state, auxiliary = program()
```

`warm()` takes the same arguments as `program(...)`. For example, use a
representative input tape and pass a continuation to prepare resumed runs.

```python
example_inputs = dn.func.Inputs(ve=torch.zeros(
    (40, *model.v.shape), dtype=model.v.dtype, device=model.v.device
))
program.warm(inputs=example_inputs, verbose=True)
program.warm(auxiliary["continuation"], inputs=example_inputs, verbose=True)
```

Warm-up follows the requested run length and recording cadence, with recording
file writes suppressed. Subsequent calls still start from their selected
snapshot or continuation. With `verbose=True`, progress and elapsed time are
printed to standard error. Accelerator work completes before `warm()` returns.

A different input shape, gradient configuration or run length may require
additional compilation. In training, `warm()` prepares the forward pass, and
the first backward pass may compile separately. See {doc}`diagnostics` for
timing advice and `program.execution_report()`.

## Find activation thresholds

`Thresholder` brackets the stimulus amplitude that triggers an activity
criterion for each model instance in a Population. Pass a native Population
with its activity callback, or a compiled Population program with the name
of its Boolean callback result.

The example below compiles two passive cables with a runtime `extra` slot.
Each trial scales the spatial field by its candidate amplitude. The
`"activation"` callback tests for a crossing of −64 mV at the first
compartment. `upper` and `lower` each contain one threshold bound per cable.

```{literalinclude} ../../examples/training/functional_thresholding.py
:language: python
:start-after: "# region compiled_threshold"
:end-before: "# endregion compiled_threshold"
:dedent: 4
```

Supply an upper bound with `ub=` for a compiled program. The search uses the
program's timestep and default run length. An explicit `dt` must match the
program, and `tstop` sets an absolute stop time in ms. To search with precomputed
time-first `bases`, compile with `extra=None`. The basis tape supplies the
extracellular voltage and determines the run length.

Each amplitude starts an independent inference trial with new callback history.
With snapshot initialization, the trials start from the same simulation and
random stream state. With `initialization="fresh"`, the program initializes for
each trial.

The `"peak_voltage"` callback records the largest compartment voltage at each
sample time. With `block_possible=True`, the search uses the peak over the trial
to distinguish a subthreshold response from possible conduction block. For a
custom activity callback, return one Boolean per searched instance or partition
and supply an explicit `block_threshold=` when checking for conduction block.
Set callback partitions before compiling so they match the search partitions.

The {doc}`thresholding tutorial <../basics/01d_thresholding>` introduces the
search. The complete
{download}`functional_thresholding.py <../../examples/training/functional_thresholding.py>`
example compares native and compiled threshold brackets and supports both scan
and while inference. Compiled Network threshold searches are currently unsupported.

## Bound pending event storage

When a compiled Network uses an inference event calendar, it has a fixed
capacity for events waiting to be delivered. For dense bursts or long delays,
increase `pending_capacity` when compiling. Pass an integer or a mapping from
connection names to capacities.

If a run raises an overflow error, retry from its starting state. For a snapshot
run, compile the source model with a larger capacity. For a resumed run, transfer
the continuation from before the failed call into a compatible native Network
with `dn.func.commit_continuation_`, then compile that Network with the larger
capacity. This gives the new program a starting state with the new calendar
shape. See the {doc}`handoff guide <initialization_and_checkpoints>` and the complete
{download}`functional_pending_calendar.py <../../examples/networks/functional_pending_calendar.py>`
example.

## Next steps

- {doc}`inputs_and_randomness`: stimulation, tensor inputs and reproducible draws.
- {doc}`recording_and_callbacks`: fields, regions, cadence, analysis and HDF5.
- {doc}`training_and_parameters`: parameter selection, optimizers and gradients.
- {doc}`initialization_and_checkpoints`: fresh trials, portable saves and handoff.
- {doc}`custom_models`: model hooks, supported components and explicit lowering.

The complete
{download}`functional_api_overview.py <../../examples/getting_started/functional_api_overview.py>`
example combines Population inference, connected-Network training, explicit
bindings, stochastic replay and runtime extracellular stimulation.
