(recording-and-callbacks)=
# Recording and online analysis

Use `dn.Recorder` to collect voltage, time, or mechanism state. Other callbacks
measure activity or compute summaries during simulation. In-memory results can
retain gradients. HDF5 stores detached samples and keeps long recordings off
the autograd graph.

## Record a native simulation

Pass a Recorder in `callbacks=[recorder]`, then read each field from the Recorder.

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

native = make_cell()
recorder = dn.Recorder(("v", "t"), dt=2 * DT)
native.run(steps=8, dt=DT, callbacks=[recorder])

native_voltage = recorder.stack("v")
native_time = recorder.stack("t")
assert native_voltage.shape == (5, 2, 1)  # Samples, cells, compartments.
assert native_time.shape == (5,)
```

A fresh Recorder samples the initial boundary, then every accepted step by
default. With `dt=record_dt`, it samples every `int(record_dt / solver_dt)`
steps. Here the samples are the initial boundary and steps 2, 4, 6 and 8.
Pass the solver timestep to `run` and the sampling interval to `Recorder`.
Choose a sampling interval at least as large as the solver timestep. Record
`"t"` to read the actual sample times in ms. `recorder.numpy("v")` detaches
the same time-leading data and moves it to CPU for plotting.

## Record a compiled simulation

Use the same Recorder configuration in a named mapping. The compiled runner
returns the trace under that name in `auxiliary["callbacks"]`. This example uses
the eager backend to keep startup short.

```python
model = make_cell()
program = dn.compile(
    model, dt=DT, steps=8, inference=False, trainable="*g_param",
    callbacks={"trace": dn.Recorder(("v", "t"), dt=2 * DT)},
    backend="eager",
)
state, auxiliary = dn.func.run(program)
trace = auxiliary["callbacks"]["trace"]
voltage, time = trace["v"], trace["t"]
torch.testing.assert_close(voltage, native_voltage)
torch.testing.assert_close(time, native_time)
```

Pass the returned continuation to keep the simulation state and sampling phase.

```python
next_state, next_auxiliary = dn.func.run(program, auxiliary["continuation"], steps=4)
joined = torch.cat((voltage, next_auxiliary["callbacks"]["trace"]["v"]))
assert joined.shape[0] == 7  # The continuation does not repeat its boundary.
```

Each returned trace contains the samples from that call. Reducers retain their
accumulated values across continuation. A call without state or callback carry
starts a fresh recording. See {doc}`initialization_and_checkpoints` for
initialization and checkpoint workflows.

## Use the same observations in native execution

Prepare a named collection when you want explicit results and sampling carry
across native calls. Each result maps callback names to observations and
provides `.state` for the next call.

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
torch.testing.assert_close(native_trace, voltage)
```

Prepared collections also work with `step` and `longrun_checkpointed`. The
checkpointed call returns `(hook_loss, result)` when using a prepared collection.
Preparation resolves field names and selections for the model and timestep.
Pass the carry back to the same collection, or transfer it to an independently
prepared collection with the callback checkpoint methods. Native execution can
reuse a prepared collection on compatible models.

## Stream long recordings to HDF5

Configure `set_hdf5` before passing the Recorder to execution. Close the writer
before opening the file. This standalone script writes and reads one recording.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import dendra as dn
from dendra.data import H5Reader
from dendra.models.mod import pas


def record_and_read(path):
    dt = 0.025
    with dn.ctx(JIT=0, DEVICE="cpu"):
        model = dn.SingleCompartment(
            N=2, C=1, v_init=-65.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        model.insert(pas.rename("leak"), g=0.001, e=-55.0)
        model.eval()
        model.initialize()

        recorder = dn.Recorder(("v", "t"), dt=2 * dt)
        recorder.set_hdf5(str(path), cache_every=4)
        try:
            model.run(steps=8, dt=dt, callbacks=[recorder])
        finally:
            recorder.close()

    with H5Reader(path) as reader:
        voltage = reader["v"][0][:].compute()
        time = reader["t"][0][:].compute()
    return voltage, time


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        voltage, time = record_and_read(Path(directory) / "trace.h5")
        assert voltage.shape == (5, 2, 1)
        assert time.shape == (5,)
```

`H5Reader` indexes by field, run number, then time slice. Compute each lazy
array while the reader is open. Choose a persistent path to keep the file.
Keep the `if __name__ == "__main__":` guard in scripts because the writer
starts worker processes. `cache_every` controls the flush interval in solver
iterations independently of the sampling cadence. A fresh recording counts
its initial boundary toward the first flush.

Compiled runners and prepared native collections accept the same storage
configuration. Read their disk-backed samples through `H5Reader`. The returned
callback state carries the sampling phase for continuation. Use an in-memory
Recorder when the samples will enter a gradient-based loss. The complete
{download}`native_callback_storage.py <../../examples/callbacks/native_callback_storage.py>`
example includes readback and checkpointed execution.

## Choose fields and Population scopes

Use `"v"` for voltage, `"t"` for time, and paths such as `"hh.m"` for mechanism
state. A Network Recorder uses qualified fields such as `"left.v"`. Its root
`"t"` is the Network clock. `node_indices` selects compartments along the
recorded field's final axis.

`region=` accepts a physical `Slice`, including a named component within a
concatenated MultiPopulation. It selects physical voltage and current fields.
For mechanism-local state, select the mechanism field and its support.
Choose either `region` or the index and reduction options
`node_indices`, `max_only` and `partition`.

Use `PopulationCallback` to run a callback on one named Population. A scoped
Recorder uses local paths, as it does on a standalone model.

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

Scopes select an exact top-level Population name in both native and compiled
Networks. Use a region inside that Population to select a child of a
concatenated model. After `network.concat()`, a Recorder can select a named
child region with
`dn.Recorder(("all_populations.v",), region=network.all_populations.exc.soma)`
when the model defines an `exc` component with a `soma` region. The
{download}`functional_network_regions.py <../../examples/callbacks/functional_network_regions.py>`
example defines and records these regions.

## Online spike analysis and LFP

| Configuration | Result |
| --- | --- |
| `dn.callbacks.APCount` | Int32 upward threshold-crossing counts per checked compartment. |
| `dn.callbacks.Raster` | Time-leading Boolean crossing rows for samples within the observation window. |
| `dn.callbacks.Active`, `ActiveAL`, `ActiveALCount` | Boolean tests for any active site, a target number of distinct active sites, or a target number of crossings including repeats. |
| `dn.callbacks.AnomalyDetector` | Cumulative Boolean flag per cell for nonfinite post-step voltage. |
| `dn.callbacks.LFP` | Extracellular potentials in mV, shaped `(samples, *batch, contacts)` by default. |

Use the hard threshold callbacks to measure responses. For training, choose a
differentiable loss on recorded or reduced model outputs. See the
{doc}`callback API reference <../api/dendra.models.callbacks>` for observation
windows and checked nodes.

LFP requires membrane-current output, for example an integrator constructed
with `imem=True`. Supply one reciprocal lead field in mV/mA per contact. In a
Network, use `PopulationCallback` to select the source Population. The
{download}`functional_lfp_configuration.py <../../examples/callbacks/functional_lfp_configuration.py>`
example shows contact and batch dimensions, field gradients and continuation.

## Write a pure callback

Keep evolving values in carry. This reducer sums the mean squared voltage
after each accepted step and emits no trajectory.

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

`initialize` starts the carry, `update` advances it, and `finalize` turns carry
and any retained emissions into a result for the call. Keep configuration
immutable. Use deterministic tensor operations with stable carry and emission
shapes.
Capture-compatible callbacks work with scan and inference-only while
execution. For portable checkpoints, include every setting that affects the
callback's interpretation in `checkpoint_configuration`. See
{class}`dendra.func.FunctionalCallback` for the authoring contract.
