(functional-execution)=
# Run models with explicit state

Use `dn.compile` when you want a reusable simulation function for inference or
training. Define and initialize an ordinary Dendra Population or Network, then
compile it. The function returns tensors and an explicit continuation;
it does not advance the original model.

Use native `model.run(...)` when you want the model itself to own its evolving
state. Both interfaces support deterministic inputs, stochastic replay and
the same callback configurations. You can also
{doc}`move between them <initialization_and_checkpoints>`.

## Start with one Population

This complete example stimulates a passive compartment and records its voltage.
It selects the `eager` compiler backend for quick startup; omit `backend` to use
the default optimizing backend.

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

`state` contains the final simulation tensors. The named Recorder returns its
sampled values at `auxiliary["callbacks"]["trace"]`. A fresh recording includes
the initial boundary, so this trace has 41 samples.

The source model is unchanged. Calling `program()` is equivalent to
`dn.func.run(program)` and starts again from its captured initial snapshot.
Pass the returned continuation to advance the same simulation:

```python
# Continue the previous example for another 40 accepted steps.
state, following = dn.func.run(program, auxiliary["continuation"])
```

The continuation carries the effective parameters, constants, simulation/RNG
state and callback phase together. Continued recordings contain only the new
samples, without repeating the joining boundary.

## Compile a connected Network

Connections are configured before compilation using the ordinary Network API.
This complete example sends NetStim events into one Population; its voltage
crossings drive an event synapse on another Population.

```python
import torch
import dendra as dn
from dendra.models.mod import expsyn, pas

DT = 0.025
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
        threshold=None, weight=0.002, delay=DT,
    )
    network.connect(
        source[:], target[:], target.mech.exc, conn_spec="one_to_one",
        threshold=-20.0, weight=0.002, delay=0.1,
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
```

Recorder field names use the Network's Population names. Event-connection
weights are defined by their target mechanism: `expsyn` uses bare numerical
values in µS. Networks can also use continuous connections, custom deterministic
release transforms, schedules and delays; see {doc}`custom_models` and the
{doc}`Network API <../api/dendra.models.networks>`.

## Use ordinary connection and scheduling APIs

Configure connectivity on the native Network, then compile that configuration:

| What you want to model | Interface |
| --- | --- |
| Discrete events from a NetStim or voltage crossings | `network.connect(...)` with an event-driven target such as `expsyn`. |
| A continuous signal between Populations | `network.connect_continuous(...)` with a receiving input, optional deterministic function or module, weight and delay. |
| Predetermined event times and amplitudes | `NetStim.schedule(...)` or the connection's `schedule(...)`. |

Schedules accept values or indexed live time/amplitude sources. Supply one time
form and one amplitude form; incompatible alternatives are rejected. This is
useful for optimizing a stimulation protocol without replacing its connectivity.
Bind shared Tensor leaves with `bind_time_source(times)` and
`bind_weight_source(amplitudes)`, then schedule using `time_idx=` and
`weight_idx=`. The live indexed form is useful across repeated optimizer
segments because each execution reads the current leaves. See the complete
{download}`shared_schedule_bindings.py <../../examples/shared_schedule_bindings.py>`
example for native and functional training. Continuous transforms can contain
trainable parameters, so an optimizer can fit a release curve or another
conversion of the presynaptic signal.

Delays and their interpolation policies are configured with the Network before
compilation. `dn.compile` uses that ordinary Network configuration.
Runtime randomness and replay are described in
{doc}`inputs_and_randomness`; connection signatures and source bindings are in the
{doc}`Network API <../api/dendra.models.networks>`.

## Choose the amount of simulation

`steps=N` accepts exactly N solver steps. `duration=T` requests elapsed time in
ms and retains any fraction smaller than a complete timestep. Supply one of
them at compile time; a call may override that default:

```python
# Uses the program and continuation from either example above.
state, auxiliary = dn.func.run(program, steps=10)
state, auxiliary = dn.func.run(
    program, auxiliary["continuation"], duration=0.25,
)
```

With `dt=0.1`, two continued requests of `duration=0.25` accept two and three
steps. Exact step requests leave the pending fraction available for later
duration requests. A no-state call restarts the selected initialization policy.

Explicit input and draw tapes must contain the accepted number of steps.
Use `dn.func.resolve_steps` when constructing a tape for a duration request;
see {doc}`inputs_and_randomness`.

## Choose inference or training

The default runner uses `inference=True`, so simulation outputs do not retain
training graphs. This flag does not change the source model's training/evaluation
configuration; call `model.train()` or `model.eval()` before initialization and
compilation as appropriate. Set `inference=False` and select trainable sources for
optimization; see {doc}`training_and_parameters`.

`execution="scan"` supports training and inference. `execution="while"` is
an inference option for both Population and Network runners, including custom
pure callbacks. Population while runners require a compiled width of at least
four steps. Supported models and authoring requirements are described in
{doc}`custom_models`.

`chunk_steps` controls the compiled width. It need not divide the requested
horizon: the host scheduler executes full chunks and the remaining tail.
Changing the call's horizon reuses the runner; a previously unseen tail width
may require compilation. Chunking bounds execution workspaces, but requested
recordings still require storage for their retained samples.

Compilation is lazy, so the first call includes compilation work. Inspect
`program.execution_report()` after a call when diagnosing its selected strategy.
Source structure and connectivity are captured at compilation: after structural
model changes, construct a new program.

## Bound pending event storage

Inference event calendars use fixed-capacity storage. For dense event bursts or
long delays, supply `pending_capacity=N` to Network compilation, or a mapping
from connection names to capacities. Overflow raises before a result is
published. Recompile from the unchanged native snapshot with enough capacity. For an
already continued simulation, transfer its pre-call continuation into a compatible
native receiver with `commit_continuation_`, then compile that receiver with the
larger capacity. Calendar shapes belong to the plan; a small-plan packet is not
directly interchangeable with a larger-plan packet. Do not continue from a
partial result. The complete
{download}`functional_pending_calendar.py <../../examples/functional_pending_calendar.py>`
example demonstrates this recovery. Delivery and parameter-lookup timing follow
the Network's selected connection policy.

## Next steps

- {doc}`inputs_and_randomness`: stimulation, tensor inputs and reproducible draws.
- {doc}`recording_and_callbacks`: fields, regions, cadence, analysis and HDF5.
- {doc}`training_and_parameters`: parameter selection, optimizers and gradients.
- {doc}`initialization_and_checkpoints`: fresh trials, portable saves and handoff.
- {doc}`custom_models`: model hooks, supported components and explicit lowering.

The complete
{download}`functional_api_overview.py <../../examples/functional_api_overview.py>`
example combines Population inference, connected-Network training, explicit
bindings, stochastic replay and runtime extracellular stimulation.
