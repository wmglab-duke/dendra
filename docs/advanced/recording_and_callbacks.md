(recording-and-callbacks)=
# Recording and online analysis

Use `dn.Recorder` to retain a trajectory, and callbacks such as `APCount` or
`LFP` to compute observations during simulation. The same configurations work
with native models and `dn.compile`. In-memory observations can retain
gradients; HDF5 recording stores detached samples.

## Record a compiled simulation

A compiled runner takes a named callback mapping. Results appear under those
names in `auxiliary["callbacks"]`:

```python
import torch
import dendra as dn
from dendra.models.mod import pas

DT = 0.025

def make_cell():
    with dn.ctx(JIT=0, DEVICE="cpu", DTYPE=torch.float64, REQUIRE_GRAD=1):
        cell = dn.SingleCompartment(
            N=2, C=1, v_init=-65.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        cell.insert(pas.rename("leak"), g=0.001, e=-55.0)
        cell.train()
        cell.initialize()
        return cell

model = make_cell()
program = dn.compile(
    model, dt=DT, steps=8, inference=False, trainable="*g_param",
    callbacks={"trace": dn.Recorder(("v", "t"), dt=2 * DT)},
    backend="eager",
)
state, auxiliary = dn.func.run(program)
trace = auxiliary["callbacks"]["trace"]
voltage, time = trace["v"], trace["t"]
assert voltage.shape[0] == 5  # Initial boundary, then steps 2, 4, 6 and 8.
```

`Recorder(dt=None)` records the initial boundary and every accepted step.
With `dt=record_dt`, the sampling stride is `int(record_dt / solver_dt)`;
choose a recording interval at least as large as the solver timestep.
The recording interval does not change the solver timestep. Use `"t"` to
obtain the actual sample times, especially with subsampling or continuation.

To continue the simulation and recording phase, pass the returned packet:

```python
next_state, next_auxiliary = dn.func.run(program, auxiliary["continuation"], steps=4)
joined = torch.cat((voltage, next_auxiliary["callbacks"]["trace"]["v"]))
assert joined.shape[0] == 7  # The continuation does not repeat its boundary.
```

Each returned trace contains samples collected by that call. Reducers retain
their accumulated carry. A fresh no-state runner call starts a new recording;
see {doc}`initialization_and_checkpoints` for explicit fresh initialization.

## Choose fields and Population scopes

Use `"v"` for voltage, `"t"` for time, and mechanism paths such as `"hh.m"`
for local mechanism state. A Network Recorder uses qualified fields such as
`"left.v"`; the root `"t"` is the Network clock. `node_indices` selects
compartments along the recorded field's final axis.

`region=` accepts a physical `Slice`, including a named component within a
concatenated MultiPopulation. It selects physical voltage and current fields;
mechanism-local state currently requires its own field/support selection.
Region selection cannot be combined with `node_indices`, `max_only` or
`partition`.

For a callback that observes one Population, use `PopulationCallback`. A
scoped Recorder uses local paths, just as it would for a standalone model:

```python
network = dn.Network({"left": make_cell(), "right": make_cell()})
network.train()
network.initialize(DT)
observations = {
    "both": dn.Recorder(("left.v", "right.v", "t")),
    "left_only": dn.callbacks.PopulationCallback("left", dn.Recorder(("v", "t"))),
    "spikes": dn.callbacks.PopulationCallback(
        "right", dn.callbacks.APCount(node_check=[0], threshold=-20.0, dt=DT)
    ),
}
net_program = dn.compile(network, dt=DT, steps=8, callbacks=observations, backend="eager")
net_state, net_auxiliary = dn.func.run(net_program)
assert "v" in net_auxiliary["callbacks"]["left_only"]
```

Scopes select one exact top-level Population name. They also work with native
Network callbacks. Use a region inside that Population when selecting a child
of a concatenated model. After `network.concat()`, a Recorder can select a
named child region with
`dn.Recorder(("all_populations.v",), region=network.all_populations.exc.soma)`
when the model defines an `exc` component with a `soma` region. The complete
{download}`functional_network_regions.py <../../examples/functional_network_regions.py>`
example defines and records these concatenated regions.

## Use the same observations in native execution

Prepare a named pure collection once, then keep its returned state between
native calls. Native `run`, `longrun`, `longrun_checkpointed` and `step`
support these observations:

```python
native = make_cell()
callbacks = dn.func.make_callbacks(
    native, {"trace": dn.Recorder(("v", "t"), dt=2 * DT)}, dt=DT
)
first = native.run(steps=4, dt=DT, callbacks=callbacks)
last = native.longrun(
    steps=4, chunklength=2, dt=DT,
    callbacks=callbacks, callback_state=first.state,
)
native_trace = torch.cat((first["trace"]["v"], last["trace"]["v"]))
assert native_trace.shape[0] == 5
```

Preparation binds field names and selections and retains its model/timestep
compatibility requirements. Functional execution uses the collection's exact
owning plan; native reuse accepts compatible receivers. Carry belongs to its
prepared collection. Transfer it between independently prepared collections
with their callback checkpoint methods.

The existing native list interface, `callbacks=[recorder]`, retains its mutable
callback history and hook behavior. Read those observations from the callback
object. A named prepared collection instead returns explicit results and carry.

## Online spike analysis and LFP

| Configuration | Result and purpose |
| --- | --- |
| `dn.callbacks.APCount` | Int32 counts of hard upward threshold crossings. |
| `dn.callbacks.Raster` | Boolean crossing rows only for timesteps within its observation window. |
| `dn.callbacks.Active`, `ActiveAL`, `ActiveALCount` | Boolean activity tests: any checked site, a target number of distinct sites, or a target total crossing count including repeats. |
| `dn.callbacks.AnomalyDetector` | Accumulates whether nonfinite post-step voltage has been observed. |
| `dn.callbacks.LFP` | Projects absolute transmembrane current through supplied lead fields. |

These threshold counts are not differentiable spike surrogates. They are
useful for measuring responses; choose a suitable differentiable loss when
training spike generation. Observation windows and checked nodes are described
in the {doc}`callback API reference <../api/dendra.models.callbacks>`.

LFP requires membrane-current output, for example an integrator constructed
with `imem=True`. Supply one unit-current field per recording contact. In a
Network, wrap LFP with `PopulationCallback` to choose its source Population.
The complete {download}`functional_lfp_configuration.py
<../../examples/functional_lfp_configuration.py>` example shows contact and
batch dimensions, field gradients and continued recordings.

## Stream long recordings to HDF5

Configure storage on `dn.Recorder` before preparing or compiling it:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

def write_recording():
    with TemporaryDirectory() as directory:
        disk = dn.Recorder(("v", "t"), dt=2 * DT)
        disk.set_hdf5(str(Path(directory) / "trace.h5"), cache_every=4)
        stored_program = dn.compile(
            model, dt=DT, steps=8, callbacks={"trace": disk}, backend="eager"
        )
        stored_state, stored_auxiliary = dn.func.run(stored_program)
        disk.close()

if __name__ == "__main__":
    write_recording()
```

HDF5 storage starts worker processes; standalone scripts use the
`if __name__ == "__main__":` guard shown above. In a notebook, call
`write_recording()` directly. Use a persistent directory to keep the file. Disk-backed histories are written
by host execution rather than returned as an in-memory trajectory. Their
callback carry still supports continuation. Native pure collections use the
same storage configuration. Stored samples are detached: use in-memory
observations for gradient-based losses. See the complete
{download}`native_callback_storage.py <../../examples/native_callback_storage.py>`
example for readback and checkpointed execution.

## Write a pure callback

A custom callback stores evolving values in explicit carry. This reducer sums
post-step mean squared voltage without keeping a trajectory:

```python
class SquaredVoltageSum(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        return state["integrator"]["v"].new_zeros(()), None

    def update(self, carry, state, auxiliary):
        return carry + state["integrator"]["v"].square().mean(), None

    def finalize(self, carry, emissions):
        return carry

    def checkpoint_configuration(self):
        return {"kind": "example.squared_voltage_sum", "version": 1, "fields": {}}

summary_program = dn.compile(
    model, dt=DT, steps=8, inference=False, callbacks={"summary": SquaredVoltageSum()},
    backend="eager",
)
summary_state, summary_auxiliary = dn.func.run(summary_program)
summary = summary_auxiliary["callbacks"]["summary"]
```

Keep configuration immutable and use tensor operations without hidden state,
I/O or random sampling. Carry and emissions need stable structures and tensor
shapes. `initialize` runs once for fresh observations, `update` runs after each
accepted step, and `finalize` produces each call's result. The same callback
can run with scan and inference-only while execution when its operations are
capture-compatible. `checkpoint_configuration` is needed for portable custom
callback checkpoints; include every setting that affects its interpretation.
See {class}`dendra.func.FunctionalCallback` for the full authoring contract.
