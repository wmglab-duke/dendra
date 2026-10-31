---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-api-overview)=
# Choose a functional workflow

Use `dn.compile(model, ...)` for an ordinary simulation or training loop built
from a supported, initialized Population or Network. It owns independent default
parameter tensors, manages preparation and optional runtime sampling, and
returns explicit simulation state. Execute the returned program with
`dn.func.run(program, ...)` or `program(...)`. Use `make_functional(model, dt=...)`
for either kind of model when an application needs to control bindings, chunk
scheduling, explicit draws, or PyTorch function transforms.

The same `dn.compile` function returns `CompiledRunner` for one Population,
including a concatenated `MultiPopulation`, or `CompiledNetworkRunner` for a
Network. The Network runner coordinates its Populations, connections, optional
NetStim, and shared clock/RNG scheduling. Both use the same calling and
continuation interface; Network inputs and parameter trees are grouped by
Population/component, and observation paths use names such as `"left.v"`.

The complete
{download}`functional_api_overview.py <../../examples/functional_api_overview.py>`
example contains six independently runnable sections:

```console
python examples/functional_api_overview.py --section inference
python examples/functional_api_overview.py --section training
python examples/functional_api_overview.py --section explicit
python examples/functional_api_overview.py --section draws
python examples/functional_api_overview.py --section extra
python examples/functional_api_overview.py --section transforms
```

Omit `--section` to run all six. The default `--backend eager` exercises compiler
capture without native kernel generation. `--backend aot_eager` also exercises
AOT autograd; `--backend inductor` requests generated kernels and includes
their compilation cost. The examples use short CPU simulations in float64,
check continuation and replay, and check finite nonzero training gradients.
They do not measure performance or establish numerical convergence.

## Start with an ordinary model

Model construction uses the standard Dendra API. The cable in the example has
a passive membrane and an optional intracellular waveform registered through a
Slice:

```{literalinclude} ../../examples/functional_api_overview.py
:language: python
:start-after: "# region cable"
:end-before: "# endregion cable"
```

The runnable file supplies the imports, constants, helper functions and a
`dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False)`
context. Times are in ms. `dn.units.nA` scales the injected current amplitude
to mA. Registered `Slice.inject(...)` stimulation remains part of the
lowered model; the caller does not need to reconstruct its current tape.

Construct and initialize a Network in the desired training or inference mode
before lowering it. Functional Network lowering requires a single device;
{doc}`A19_functional_network_lowering` describes the supported configurations.
Its timestep, topology, parameter alias relationships and
execution mode form part of the functional plan. The example connects two
single-compartment Populations continuously and drives an event synapse from a
noisy NetStim. Its continuous transform is an ordinary deterministic function:

```python
def release(voltage):
    return torch.sigmoid(0.1 * (voltage + 60.0))
```

```{literalinclude} ../../examples/functional_api_overview.py
:language: python
:start-after: "# region network"
:end-before: "# endregion network"
```

Functions and stateless modules can implement continuous transforms, including
modules with trainable parameters. Their tensor state must obey the functional
contract. Optional purity diagnostics are controlled through `dn.ctx`'s
`CHECK_PURITY` setting; normal execution trusts authored implementations.

(functional-runner-execution)=
## Run a compiled program

`dn.func.run` accepts compiled Population and Network programs as well as
explicit bindings. For a compiled program, it uses the same defaults, source
ownership and execution policy as calling the program directly:

```python
program = dn.compile(model, dt=DT, steps=100, chunk_steps=16)
state, auxiliary = dn.func.run(program)
state, auxiliary = dn.func.run(
    program, auxiliary["continuation"], duration=25.0,
)
```

Supply an ordinary state or a `Continuation` as the second positional argument,
or through `state=`. Omitted state applies the program's configured snapshot or
fresh restart policy. A continuation supplies its effective parameters,
constants and callback carry. Both forms return `(state, auxiliary)` and publish
the next packet at `auxiliary["continuation"]`.

The call accepts the same runner keywords as `program(...)`: `steps`, `duration`,
`parameters`, `constants`, `inputs`, `draws`, `callback_state`, `initialization`
and declared runtime `extra`. The configured horizon applies unless `steps=` or
`duration=` replaces it; input tapes alone do not replace a runner's horizon.
Configure the timestep, callbacks, inference policy, initialization policy,
compiled chunk width and activation checkpoint span with `dn.compile`.
Non-`None` call-time `callbacks=`, `dt=`, `tstop=`, `checkpoint_every=` and
`host_span_steps=` overrides are not supported for program targets. Explicit
binding and legacy callable forms retain their separate scheduling arguments.
Runner preparation caches, automatic sampling, callback phase and gradients
follow the same rules through either spelling.

The complete
{download}`functional_runner_execution.py <../../examples/functional_runner_execution.py>`
example checks Population and Network restart, direct/canonical equivalence,
initialized and portable continuation, shared aliases and live gradients:

```console
python examples/functional_runner_execution.py
python examples/functional_runner_execution.py --backend aot_eager
```

## Inference: compile, run, retain state explicitly

`dn.compile` defaults to `inference=True` and `initialization="snapshot"`.
It enters inference mode during execution. Every no-state call restarts from
the initialized snapshot; an explicit state continues that simulation. It does
not advance the native model or overwrite its snapshot.

Choose a default horizon with either `steps` or `duration`; `chunk_steps`
selects the compiled width for both Populations and Networks:

```python
runner = dn.compile(model, dt=DT, steps=1_000, chunk_steps=128)
# All 1,000 steps, including the shorter tail.
state, auxiliary = dn.func.run(runner)
state, auxiliary = dn.func.run(runner, steps=250)  # Restart with another horizon.
state, auxiliary = dn.func.run(runner, auxiliary["continuation"], duration=25.0)
```

`duration` is elapsed simulation time in ms. It is finite and nonnegative;
compile-time `steps` is a positive integer, and a per-call step override may
also be zero. `steps` and `duration` are mutually exclusive at construction and
at invocation. Supplying either per call replaces the default horizon:

```python
runner = dn.compile(model, dt=DT, duration=100.0, chunk_steps=128)
state, auxiliary = runner()  # Advance the default duration from the snapshot.
state, auxiliary = runner(auxiliary["continuation"], steps=250)
```

Duration calls carry fractional-step remainder in the returned continuation.
For example, with `dt=0.1`, two continued `duration=0.25` calls accept two and
three steps respectively. Explicit step calls leave that remainder unchanged.
A no-state call restarts its default initialization policy and recording phase;
zero-step continuation retains its phase without adding a boundary sample.
Input and draw tapes must match the selected call's accepted step count.

The positive compiled width need not divide a call's horizon. A step default
caps it at `steps`; a duration default uses its nominal accepted step count,
with a minimum width of one for scan or four for Population while execution.
These minimum widths do not force a short call to advance a full chunk.
Omission uses the ordinary default horizon, or at most 64 steps for runtime
`extra=dn.VarInput()` slots. Registered and compile-time
bound stimulation use the ordinary setting. Changing the call's horizon leaves
the compiled width unchanged and reuses cached tails; a previously unseen tail
width may require compilation. Chunking does not remove requested recording
storage.

The complete
{download}`functional_runner_horizons.py <../../examples/functional_runner_horizons.py>`
example checks step and duration defaults, per-call overrides, fractional
continuation and Recorder phase for both model families.

Recording is an explicit named callback. `dn.Recorder` configures recording
for native models and functional runners; `dn.func.Recorder` remains supported.
By default a fresh Recorder includes the initial sample, followed by every
accepted step. `Recorder(dt=record_dt)` selects the initial boundary and accepted
steps that are multiples of `int(record_dt / solver_dt)`, using the same rule
for in-memory and HDF5 storage. Its phase is explicit carry and survives chunks
and continuation. Different Recorders can use different
cadences. Host schedules retain selected frames between emission segments.
With bounded chunks or host/checkpoint spans, that Recorder's payload scales
with selected history plus bounded dense segments. Each compiled invocation still
emits dense data; full-rate/custom streams, autograd and input tapes have
separate storage requirements. Atomic rollouts retain dense finalization; see
{ref}`functional-recorder-cadence` for timing and compilation details.
`Recorder.set_hdf5(path, cache_every=...)` routes its sampled trajectory to the
host HDF5 writer instead of returned callback history. Its sampling carry
remains available for continuation. In-memory recordings retain gradients;
persisted samples are detached. Functional host schedules and native pure
callback collections use the same storage path; direct tensor kernels reject
HDF5 configurations. See {ref}`functional-recorder-storage` for file segments,
bounded storage and the complete runnable examples.

Use `dn.callbacks.APCount`, `Raster`, `Active`, `ActiveAL`, `ActiveALCount`
and `AnomalyDetector` to configure native or functional spike analysis and
anomaly detection. Functional binding captures
these settings into independent pure callbacks; native histories and counters
remain separate. The corresponding `dn.func` constructors are also supported.
For a Network, select the observed Population with
`dn.callbacks.PopulationCallback(name, callback)`. This scope also forwards a
native callback's hooks to the selected Population in native Network execution.
Use a callback list for native runs and a named mapping for functional results;
`dn.func.PopulationCallback` remains a compatible functional entry point.
Scopes select exact top-level Population names and cannot be nested. A scoped
Recorder uses local paths such as `"v"` or `"hh.m"`; named Slice regions also
select components within a concatenated MultiPopulation. Native results belong
to the inner callback; functional results and continuation use independent
explicit carry. Pure callback children bind through `functional.make_callbacks`
and can also observe native Network `step`, `run`, `longrun` and `longrun_checkpointed`
through that collection.
They do not run through the ordinary imperative `callbacks=` hook list.
See the {ref}`shared scope example <shared-population-callback-scopes>` and
{doc}`A27_functional_network_callbacks` for scopes and configuration ownership.
Both Raster entry points return only checked post-step samples in the
observation window, with an empty sample axis when none are checked. Their
crossing latches and window phase remain explicit carry.
Both APCount entry points return `torch.int32` counts, with four bytes per
checked slot. These exact integer observations do not require gradients;
gradients through model state and floating-point recordings remain available.

`ActiveAL(at_least=k)` compares the number of selected slots that fired at
least once against `k`; `ActiveALCount(at_least=k)` compares total crossing
events, including repeats. `inv=True` inverts the Boolean result. Native
configurations select their unpartitioned query; use the `dn.func` constructors
with `partition=...` for static grouped output. Both constructor families
require a positive integer `at_least`; booleans, floats and Tensors are not
accepted. The target is static callback configuration. See the
{ref}`activity configuration example <functional-shared-activity-configuration>`
for inversion, groups and continuation.

`dn.callbacks.LFP(fields)` configures native and functional extracellular
recording; `dn.func.LFP` remains supported. Wrap it in `PopulationCallback` for
a Network and enable membrane-current output on the observed Population.
The default projection sums cell and compartment axes, preserving batch lanes;
an explicit `rule` is applied exactly. Fields in mV/mA multiplied by membrane
current in mA produce mV. Each fresh functional run builds its own
differentiable field stack. Live continuation retains that graph; portable
restoration uses detached field carry. See the
{ref}`complete LFP example <functional-shared-lfp-configuration>` for both
Population and Network execution.

(functional-native-callback-execution)=
### Use pure callbacks with native execution

A bound pure callback collection also works with native `Population.step`, `run`,
`longrun` and `longrun_checkpointed`. Build the collection from
`make_functional` for the exact source Population and the same timestep:

```python
functional, tensors = dn.func.make_functional(model, dt=DT)
observations = functional.make_callbacks({"trace": dn.Recorder(("v", "t"))})
first = model.run(tstop=10.0, dt=DT, functional_callbacks=observations)
second = model.longrun(
    tstop=10.0, chunklength=100, dt=DT,
    functional_callbacks=observations,
    functional_callback_state=first.state,
)
```

The Population owns its advancing model state; callback carry is separate.
`step`, `run` and `longrun` return `FunctionalCallbackResults` when this option is
supplied, with named results and a `.state` carry. Without it, `step` returns
the model for chaining and `run`/`longrun` return `None`. Checkpointed execution
keeps its existing `(loss_or_none, results)` or
`(loss_or_none, final_state, results)` form.
Native `callbacks=` can coexist independently. Pure updates observe accepted
state after clock advancement and before native post-step hooks, with a
gradient-preserving tensor-frame snapshot. Continued Recorder segments omit
the shared boundary and preserve cadence; reducer results stay cumulative.

One `FunctionalCallback` definition supplies the same pure analysis for native
and functional Population dynamics. The native Population auxiliary packet
contains `{"v": ...}`. Recorders configured with `set_hdf5` use host storage
through `functional_callbacks=` as well as ordinary native hooks in `callbacks=`.
Disk histories are omitted from returned results while their sampling carry
remains in `results.state`; in-memory recordings and reducers retain gradients.
See
{ref}`functional-native-population-callbacks` for the complete contract and
{download}`native_functional_callbacks.py <../../examples/native_functional_callbacks.py>`
for a runnable custom reducer, cadence, native loss, gradients and replay:

```console
python examples/native_functional_callbacks.py
```

Native `Network.step`, `run`, `longrun` and `longrun_checkpointed` accept the same
two keyword arguments. Create the bound collection with
`dn.func.make_functional_network(network)` from the exact initialized source
Network. Qualified Recorders select fields
such as `"left.v"`; `dn.callbacks.PopulationCallback("right", callback)`
selects one Population for a relative Recorder or custom pure analysis. The
Network retains its simulation state while `results.state` continues callback
carry. `step`, `run` and `longrun` return `FunctionalCallbackResults` only when
opted in; checkpointed execution appends it to the existing loss/state return, producing
`(loss_or_none, results)` or `(loss_or_none, final_state, results)`.
They use the timestep established by `network.initialize(dt)`.

The native Network auxiliary packet is
`{"populations": {name: {"v": ...}}}`; scoped callbacks receive `{"v": ...}`.
Callback carry remains explicit through checkpointed chunks, with recording
collection outside replay. With the default `restore_state_after_backward=True`,
recomputation preserves the live Network even when the objective uses only
pure callback results and no native hook loss. See
{ref}`functional-native-network-callbacks` and the complete
{download}`native_network_functional_callbacks.py <../../examples/native_network_functional_callbacks.py>`
example for unequal connected Populations, scope parity, cadence, gradients
and checkpoint replay:

```console
python examples/native_network_functional_callbacks.py
```

For manual control loops, pass callback carry through consecutive native
`step` calls. `loop_hooks` controls only ordinary native loop hooks; pure
initialization runs when carry is absent. A fresh full-rate Recorder emits its
initial and post-step frames; continued steps emit only new frames, preserving
cadence and reducer totals. `dn.step`, `dn.step_population` and `dn.step_network`
accept the same callback keywords and return results only when opted in.
See {ref}`functional-native-step-callbacks` and
{download}`native_step_functional_callbacks.py <../../examples/native_step_functional_callbacks.py>`
for a Population/Network control-loop example with mixed-method continuation,
gradients and portable callback carry:

```console
python examples/native_step_functional_callbacks.py
```

HDF5 storage works across these native methods for Populations and Networks,
including scoped Network Recorders. File writes occur outside checkpoint replay,
so backward does not repeat them. Each call with selected samples creates a
`run_N` segment; callback carry preserves cadence and omits the shared boundary.
Calls without samples create no datasets. `cache_every` counts solver iterations
plus a fresh initial boundary; `Recorder(dt=...)` controls sampling separately.
Close each configured Recorder after its last call and before reading its file.
See {ref}`functional-native-callback-storage` and the temporary-file example:

```console
python examples/native_callback_storage.py
```

### Continue functional execution

`auxiliary["continuation"]` carries
the returned simulation state, effective bindings and callback state together.
Pass it to the next call to continue without recording the shared boundary twice:

```python
state, auxiliary = runner()
state, auxiliary = runner(auxiliary["continuation"])
```

The complete example checks restart, recording and shared continuation:

```{literalinclude} ../../examples/functional_api_overview.py
:language: python
:start-after: "# region inference"
:end-before: "# endregion inference"
```

The separate simulation-state and `callback_state=` form remains available
when an application manages those operands individually.

For final-state inference, omit `callbacks` or pass `callbacks={}` with scan or
while execution. This works for both Populations and Networks. The auxiliary
mapping has no `"callbacks"` entry, its continuation has `callback_state=None`,
and `execution_report().callbacks` is `False`. Resume the continuation in the
same way as a recorded run; no recording or callback carry is allocated.

Network Recorders qualify physical fields, such as `left.v`, `right.v` and
`right.mech.event.g`. `t` selects the Network clock. `region=some_slice` selects
a named Population or compartment region, including a labeled region of a
concatenated MultiPopulation. `PopulationCallback("left", callback)` scopes
LFP, spike-analysis and anomaly callbacks to one Population. See
{doc}`A26_functional_network_regions` and {doc}`A27_functional_network_callbacks`.

Use `execution_report()` after a successful call to inspect the selected loop
strategy. A report describes dispatch; it does not demonstrate a performance
improvement. Measure compilation and warm execution separately on the target
device and workload.

## Training: select and inspect raw parameters

Training requires `inference=False`. `trainable` accepts exact names, unique
partial names, glob patterns and explicit regular expressions (`"re:..."` or a
compiled `re.Pattern`). `find_parameters(query)` previews named matches, while
`trainable_parameters` reports live default leaves that require gradients.
An empty selection trains no model parameters. Inference mode suppresses
gradients regardless of the selection.

Shared sources keep their aliases. An optimizer receives each Tensor object
once, even when several names select it. The example shares the two leak
conductances and trains them alongside event and continuous connection weights:

```{literalinclude} ../../examples/functional_api_overview.py
:language: python
:start-after: "# region training"
:end-before: "# endregion training"
```

Each forward constructs its own differentiable preparation. Retaining the
runner across optimizer iterations does not retain an old preparation graph.
`checkpoint_every=H` recomputes exact H-step spans during backward; gradients
continue through the entire simulation. It does not truncate history or remove
the storage required by requested full recordings.
For ordinary runners, H is independent of `chunk_steps=C`, including when
neither divides the other. Runtime `VarInput` assembly retains checkpoint spans
bounded by `min(H, C)`.

The runner's parameter mappings are read-only containers with editable Tensor
leaves. Per-call `parameters=` and `constants=` overrides retain the supplied
tensor identities and autograd history and must preserve the plan's schema and
ties. Those overrides do not replace the runner-owned defaults reported by its
inspection properties. Low-level Population extraction exposes native raw
parameter tensors; Network extraction returns independent detached snapshots.
`tensors.independent_parameters(...)` constructs independent optimization
leaves while preserving their aliases in either case.

(functional-source-inspection)=
## Inspect explicit sources and runner defaults

`dn.func.inspect_sources` uses the same report throughout lowering, binding
and execution:

| Subject | Parameter and constant frames reported |
| --- | --- |
| `PopulationTensors`, `NetworkTensors` | Exactly the frames stored in the bundle |
| Raw `BoundPopulation`, `BoundNetwork` | The binding's sources, with canonical shared readers for a Network |
| Either `dn.compile(...)` runner | The runner's defaults |
| `Continuation` | The effective frames captured for that call or checkpoint |

The complete
{download}`functional_lowering_api.py <../../examples/functional_lowering_api.py>`
example inspects extracted bundles, raw bindings and returned continuations
for both a Population and a connected Network:

```python
functional, tensors = dn.func.make_functional(model, dt=DT)
print(dn.func.inspect_sources(tensors))
parameters = tensors.independent_parameters(trainable="*.g_param")
kernel = functional.compile_rollout_chunk(2, backend="aot_eager")
bound = kernel.bind(tensors._replace(parameters=parameters))
print(dn.func.inspect_sources(bound))
```

A bundle reports its stored Tensor objects, even when independently selected
parameters have been used for a later binding. Raw Network bindings resolve
parameter-owned constant readers through the same projection used by
execution and continuation. This can differ from the authored constant frame;
inspection leaves that frame and the binding's containers unchanged. Prepared
Population bindings do not own raw constants and are currently unsupported by
source inspection. Pass the original tensor bundle or create a raw binding
with `constants=` when those are the sources you intend to inspect.

For convenience runners, inspect the runner's defaults or a particular call's
effective frames, including overrides and fresh/cache initialization replacements:

```python
program = dn.compile(model, dt=DT, steps=STEPS, inference=False)
defaults = dn.func.inspect_sources(program)
print(defaults)

state, auxiliary = program()
current = dn.func.inspect_sources(auxiliary["continuation"])
print(current)
```

Every supported subject returns a read-only `SourceReport`. Printing it, or displaying it in
an interactive session, shows a summary and compact path tree: parameter and
constant path counts, unique Tensors, alias groups, trainable leaves and
gradient-enabled nonleaves. Rows show shape, dtype, device and gradient/leaf
status; alias arrows point to the first path for the same Tensor object.
The display abbreviates large frames, prioritizes gradient-enabled entries and
marks omitted paths. It does not print Tensor values or transfer data between
devices. Use `report.entries` for the full inventory.

Its ordered `entries` retain every
Tensor path: parameters first, then constants, following their frame order.
Paths have `parameters.` or `constants.` prefixes. Network paths also include
their collection and Population names; structured schedule fields retain their
names. Each `SourceEntry` provides:

| Attribute | Meaning |
| --- | --- |
| `path` | Full source path |
| `domain` | `"parameters"` or `"constants"` |
| `tensor` | The original live Tensor reference |
| `aliases` | Other paths referring to that exact Tensor object |
| `requires_grad`, `is_leaf` | Flags captured during inspection |

`report.trainable_leaves` is a tuple containing each Tensor object once, in
first-path order, when both captured flags are true. It includes qualifying
constant leaves as well as parameters and can be supplied explicitly to an
optimizer. Distinct Tensor views remain distinct objects even when they share
storage.

Inspection captures names, aliases and flags without copying or detaching
Tensors. Tensor values remain live. Inspect again after replacing sources or
changing their flags. Inspection performs no initialization, preparation,
sampling, execution or execution-validity checks. State, callback carry,
initialization inputs and call-local input/draw/extra sources are outside its
scope. Call it on the host, outside `torch.compile`.

Eligibility does not establish that a source affects a particular loss or has a
nonzero gradient. Nonleaf sources with `requires_grad=True` appear in `entries`
but are absent from `trainable_leaves`. Their upstream optimizer leaves can be
caller-owned; inspection does not traverse their computation graphs. Existing
`runner.trainable_parameters`, `find_parameters` and related selectors retain
their default-parameter scope.

The complete
{download}`functional_continuation_checkpoints.py <../../examples/functional_continuation_checkpoints.py>`
example compares default and effective inspection, lists aliases, restores
trainable leaves and optimizes a resumed segment without selecting internal
materialization keys. See {doc}`A58_functional_callback_checkpoints` for the
detached checkpoint boundary.

After a raw-bound run, its returned continuation reports the same effective
binding sources:

```{literalinclude} ../../examples/functional_lowering_api.py
:language: python
:start-after: "# region inspect_continuation"
:end-before: "# endregion inspect_continuation"
:dedent: 4
```

(functional-source-selection)=
### Select sources for optimizer groups

`report.select(queries)` returns another `SourceReport` over the captured
inventory. Queries use full report paths, including their `parameters.` or
`constants.` prefix. They accept exact names, unique partial names, glob
patterns (including `"glob:..."`), `"re:..."` expressions and compiled string
regular expressions. An iterable takes the union of its queries. A query that
matches nothing, or a plain partial name that matches several paths, raises
`KeyError`; `report.select([])` returns an empty report.

Selecting any name includes every path in the report that refers to the same
Tensor object, even across parameter and constant domains. Entries, aliases
and trainable leaves retain their original order, independent of query order.
Chaining `select` searches the current view and cannot recover groups excluded
by an earlier selection. Values and Tensor identities remain live; selection
retains the inspection's captured gradient/leaf flags and performs no cloning,
flag changes or graph traversal. Inspect again after changing bindings or flags.
Use inspection and selection on the host, outside `torch.compile`.

Different selections can supply different optimizers or parameter groups. For
a model with trainable `leak` and `other` mechanisms, as in the runnable example:

```python
report = dn.func.inspect_sources(program)
leak = report.select("parameters.*.leak.g_param")
other = report.select("parameters.*.other.g_param")
print(leak)

optimizer = torch.optim.Adam([
    {"params": leak.trainable_leaves, "lr": 1e-4},
    {"params": other.trainable_leaves, "lr": 1e-5},
])
```

Use `dn.func.inspect_sources(auxiliary["continuation"])` when selecting the
effective parameter/constant sources from a particular call, including its
replacements. Selection filters the existing trainable-leaf view: matching a
Tensor with `requires_grad=False` does not make it trainable, and captured
nonleaf sources remain excluded from optimizer leaves. State, callback carry
and call-local inputs, draws and `extra` retain their separate ownership.

Each selection deduplicates its own trainable leaves by identity. Different
selections can overlap because several names refer to a shared Tensor. Check
identity sets when independent groups are intended:

```python
assert {id(p) for p in leak.trainable_leaves}.isdisjoint(
    id(p) for p in other.trainable_leaves
)

leak_optimizer = torch.optim.Adam(leak.trainable_leaves, lr=1e-4)
other_optimizer = torch.optim.SGD(other.trainable_leaves, lr=1e-5)
```

PyTorch requires parameter groups within one optimizer to be identity-disjoint.
Separate optimizers can reference the same Tensor; stepping both applies both
optimizers' updates to that shared source. Selecting an alias does not create
an independent parameter. The complete
{download}`functional_source_selection.py <../../examples/functional_source_selection.py>`
example checks Population and Network groups, shared aliases, effective-frame
selection, gradients and updates with both optimizer arrangements:

```console
python examples/functional_source_selection.py
python examples/functional_source_selection.py --backend aot_eager
```

## Deterministic drives and runtime draws are separate inputs

Use one deterministic packet for a Population:

```python
inputs = dn.func.Inputs(ve=external_voltage, intra=injected_current)
```

Use the same packets by Population name for a Network:

```python
inputs = {
    "left": dn.func.Inputs(intra=left_current),
    "right": dn.func.Inputs(ve=right_external_voltage),
}
state, auxiliary = runner(inputs=inputs)
```

`ve` is extracellular voltage in mV for a compatible integrator. `intra` is
the total injected current per compartment in mA. A step or eager binding
consumes one frame. A rollout, compiled chunk, host `run`, or `dn.compile`
runner consumes a time-leading tape, including a compiled chunk of width one.
The remaining dimensions follow the Population's broadcasting rules. Tensor
views and gradients remain owned by the caller.

An eager Network binding selects frame semantics when passed a mapping of
`Inputs`, including `{}` for a draws-only step. Its legacy
`NetworkRolloutInput` and `inputs=None` forms retain their one-step tape
convention. The pure `FunctionalNetwork.prepare_and_step` entry point consumes
frames.

Omitted fields retain their registered drive or default behavior. Supplying
explicit `intra` to a Population with registered intracellular injections is
an error; supplying explicit `ve` alongside bound extracellular `extra` for
the same Population is also an error. Other Populations can use either source
independently.

Use `draws=RuntimeDraws(...)` separately for runtime stochastic values and their
RNG timeline. Obtain a packet from
`functional.sample_runtime_draws(state, steps=N)`, rather than constructing
its component schema by hand. A step uses the no-`steps` form. The packet
covers the model's NetStim, NOISE/SDE and randomized Poisson requests and
preserves shared generator ordering. Draw values are detached from sampling;
model and distribution parameters retain their normal gradient paths.

Pure calls require explicit stochastic packets. Host `dn.func.run` and
`dn.compile` sample omitted packets in bounded spans from the supplied state's
RNG carry. Sampling does not advance the source model or the global generator.
Keep draw values and RNG timelines together when slicing or saving packets:

```{literalinclude} ../../examples/functional_api_overview.py
:language: python
:start-after: "# region draws"
:end-before: "# endregion draws"
```

The existing `StepInput`, `RolloutInput`, `NetworkRolloutInput` and family-specific
draw packets remain supported through adapters. New code can use `Inputs` and
`RuntimeDraws` without selecting a different input representation for each
stochastic subsystem. Do not supply the same stochastic family through both
`draws=` and a legacy packet. See {ref}`functional-runtime-draws` and
{ref}`functional-runtime-inputs` for their complete contracts.

## Explicit plans, bindings and chunks

`make_functional(model, dt=...)` returns a plan and explicit tensor bundle for
an initialized Population or Network. A Network's supplied `dt` must match its
initialized timestep. The factory returns the concrete `FunctionalPopulation` /
`PopulationTensors` or `FunctionalNetwork` / `NetworkTensors` types; it preserves
their distinct state layouts and extraction ownership. The existing
`make_functional_network(network, ...)` spelling remains available and uses the
initialized Network timestep implicitly. Initialize an explicit continuation,
then run the plan eagerly or run a reusable compiled chunk:

```python
initial = dn.func.initialize(functional, tensors)
state, auxiliary = dn.func.run(functional, initial, steps=100, callbacks=callbacks)
kernel = functional.compile_rollout_chunk(2, backend="aot_eager")
state, auxiliary = dn.func.run(kernel, initial, steps=100, callbacks=callbacks)
```

The plan or chunk selects execution; the continuation supplies its effective
parameters, constants, state and callback carry. See
{ref}`functional-packet-execution` for continuation and checkpoint examples.
Explicit bindings remain useful when sources have a longer lifetime than one
invocation. `bind` accepts the matching `PopulationTensors` or `NetworkTensors`
bundle, or a `Continuation` from initialization, execution or portable
restoration:

```python
initial = dn.func.initialize(functional, tensors)
bound = kernel.bind(initial)
state, auxiliary = dn.func.run(bound, initial, steps=100, callbacks=callbacks)
```

Binding reads only `parameters` and `constants`. State, callback carry and
initialization inputs are not selected or validated by `bind`; supply the
intended state or continuation and callbacks to execution explicitly. A binding
retains source Tensor references and snapshots their containers, without
retaining the packet itself. It does not clone, detach, initialize, sample or
prepare the model. A bundle must match the plan's model family. When binding a
bundle or continuation, omit `constants` and `prepared` entirely: even an
explicit `None` conflicts with the packet's sources.

The mapping form `bind(parameters, constants=constants)` remains available on
eager plans and compiled chunks. It uses the same raw binding ownership and
preparation policies as the packet form.

The complete
{download}`functional_lowering_api.py <../../examples/functional_lowering_api.py>`
example applies this shared lowering, binding and scheduling flow to both a
Population and a connected Network, including recording, gradients and resumed
execution.

An eager binding executes one step; a compiled binding executes its fixed
width. Direct calls accept the same `callbacks=` and `callback_state=` contract
as rollouts. Bind callback configurations once with `functional.make_callbacks`.
Keep that collection when resuming its returned carry: carry is associated
with its callback plan.

For arbitrary horizons, use `dn.func.run(bound, state, ...)`. It owns tape
slicing, automatic sampling and tail dispatch. `host_span_steps` bounds host
work without checkpointing; `checkpoint_every` requests backward replay. Both
accept `steps` or `duration` with the binding's timestep. Omit `steps` when the
supplied input/draw tapes determine the horizon.
Raw-bound host runs also return `auxiliary["continuation"]`; pass that packet
as `dn.func.run(bound, packet, ...)` to continue state and callback carry with
the same source binding. See {ref}`functional-bound-continuation`.

```{literalinclude} ../../examples/functional_api_overview.py
:language: python
:start-after: "# region explicit"
:end-before: "# endregion explicit"
```

A raw binding snapshots mapping containers and retains their Tensor leaves.
In-place optimizer changes to those leaves remain visible. Replacing a mapping
entry requires a new binding. Raw bindings manage invocation-local preparation;
Population raw bindings also reuse safe inference preparation across calls.
Direct `run(plan_or_kernel, continuation, ...)` constructs an invocation-local
binding and does not retain that preparation for the next call. Keep an explicit
binding for repeated Population inference when this reuse matters. Compiled
kernels can be reused with other valid bindings; they do not own optimization
leaves or an autograd graph.

Population `bind(parameters, prepared=prepared)` remains available for applications that
explicitly own `functional.prepare(parameters, constants)`. Its second positional
argument is **prepared values**; a Network binding's second positional argument
is **raw constants**. Use `constants=` or `prepared=` to make the ownership
explicit. A prepared
Population binding must be rebuilt for each independent backward graph, and
preparation must be recomputed after its sources change. Preparing inside
`prepare_and_rollout(parameters, constants, state, ...)` provides one explicit
preparation-and-execution function for transforms.

## Runtime extracellular recipes

`extra=dn.VarInput()` declares a call-local extracellular slot. Each invocation
supplies `extra=(field, waveform)`, a list of contacts, or `None` for zero drive.
Different calls can use different deterministic Waveform types and contact
counts. The host assembles a bounded `ve` tape; a change to the contact recipe
does not specialize the numerical timestep on that Python recipe. Compiler
specialization can still depend on the numerical tensor contract.

Caller-owned field and Waveform parameters retain their gradients without a
`trainable` selector. They are not added to `runner.parameters`. The example
checks both gradients and continuation through two calls:

```{literalinclude} ../../examples/functional_api_overview.py
:language: python
:start-after: "# region extra"
:end-before: "# endregion extra"
```

Networks accept `extra={"left": dn.VarInput(), ...}` for selected Populations or
`extra=dn.VarInput()` for all Populations. A call supplies only declared slots;
omitted declared Population names receive zero drive. Runtime-extra waveforms
must be deterministic. Randomized Poisson containers belong in registered or
compile-time bound stimulation. See {ref}`functional-runtime-extra`.

## Initialization and saved continuation

(functional-runner-initialization)=
### Initialize a compiled runner

`dn.func.initialize(program)` returns an initialized `Continuation` for either
kind of compiled runner:

```python
initial = dn.func.initialize(program)
print(dn.func.inspect_sources(initial))
state, auxiliary = dn.func.run(program, initial, duration=25.0)
```

This explicit operation always follows the fresh initialization protocol,
including native cache precedence and RNG policies. The runner's
`initialization="snapshot"` or `"fresh"` setting still controls ordinary calls
without state. Initialization reads the runner's default parameters, constants,
initial state and initialization inputs. Optional `parameters=`, `constants=`
and `initialization=` arguments replace those sources for this initialization;
they do not change the defaults.

The returned packet retains the effective initialized sources, including
parameter replacements, aliases and gradient history. It has
`callback_state=None`: initialization starts no callback or recording, performs
no rollout and captures no execution kernel. Initialization itself builds the
coefficients and samples the initialization draws required by the model's
existing protocol. Runtime input tapes, `RuntimeDraws` and call-local `extra` belong
to the subsequent run; a runner with `extra=dn.VarInput()` needs no runtime
recipe for this initialization call.

Explicit initialization follows the caller's gradient mode and native model
policy. The runner's `inference` setting governs execution when the packet is
passed to `dn.func.run(program, ...)` or `program(...)`. Inspect, bind or export
the initialized continuation
through the existing APIs. The first execution starts the configured callbacks
and records their initial boundary once; subsequent continuation calls retain
their carry and sampling phase.

The shared checkpoint functions accept the runner directly, including when it
has configured callbacks:

```python
saved = dn.func.export_continuation(program, initial)
restored = dn.func.restore_continuation(program, saved)
state, auxiliary = dn.func.run(program, restored)
```

Restoration creates detached source leaves, following the checkpoint contract
below. The runner starts its configured callbacks on the restored initial state.
`program.export_continuation(initial)` and `program.restore_continuation(saved)`
provide equivalent method forms.

The complete
{download}`functional_runner_initialization.py <../../examples/functional_runner_initialization.py>`
example initializes snapshot-policy Population and Network runners with raw
parameter and voltage overrides. It checks effective shared sources, live
gradients, joined recordings and disk restoration before recording begins:

```console
python examples/functional_runner_initialization.py
python examples/functional_runner_initialization.py --backend aot_eager
```

(functional-explicit-initialization)=
### Initialize explicit sources

`dn.func.initialize` initializes a lowered Population or Network from its
matching tensor bundle and returns a `dn.func.Continuation`:

```python
functional, tensors = dn.func.make_functional(model, dt=DT)
initial = dn.func.initialize(functional, tensors)
callbacks = functional.make_callbacks({"trace": dn.Recorder(("v", "t"))})
state, auxiliary = dn.func.run(functional, initial, steps=100, callbacks=callbacks)
```

For a Network, use qualified Recorder paths such as `"left.v"`. Run with the
returned continuation: initialization transforms and caches can replace the
incoming sources. Their effective identities, aliases and
gradients travel with the returned state. The packet has `callback_state=None`;
the first `run` initializes the supplied callbacks once, including their
initial boundary sample. Subsequent calls with `auxiliary["continuation"]`
continue without reinitializing the model or repeating that sample.

The default initialization recipe is `tensors.initialization` for a
`PopulationTensors` bundle and `functional.initialization_inputs` for a
`NetworkTensors` bundle. Supply the corresponding existing
`InitializationInput` or `NetworkInitializationInput` through
`initialization=` to override it. The bundle's state supplies explicit incoming
carry; each model's existing initialization rules determine its resulting clock
and histories. No accepted timestep is executed, and the native model and its
RNG are unchanged.

This host operation samples missing stochastic initialization packets from
explicit RNG carry in native order. Explicit draw fields are replayed. In
particular, draws retained in an initialized `PopulationTensors.initialization`
are kept when that bundle is supplied again. Clear the intended draw fields
explicitly when host sampling should supply them. Runtime `draws=` tapes remain
separate arguments to `run`. Steady-state caches keep their native precedence
and detached warmup semantics.

Initialization follows ambient gradient mode. With gradients enabled, derived
state and effective sources retain their parameter and initialization-input
connections. Inference-mode calls produce ordinary Tensor results without
constructing a training graph. Call `dn.func.initialize` outside
`torch.compile`, `torch.func` and forward AD. The existing pure
`functional.initialize(...)` methods and their model-specific typed results
remain available for transformed code with explicit initialization draws.

The complete
{download}`functional_explicit_initialization.py <../../examples/functional_explicit_initialization.py>`
example supplies a raw parameter override and a registered pre-initialization
transform for both model kinds. It checks effective shared aliases, live
initialization gradients, portable continuation and suffix optimization leaves:

```console
python examples/functional_explicit_initialization.py
python examples/functional_explicit_initialization.py --backend aot_eager
```

(functional-initialization-draws)=
### Capture and replay initialization draws

`dn.func.InitializationDraws` groups the stochastic inputs for a fresh
initialization. Capture a packet from a compiled runner, then replay it with
the same initialization recipe and effective sources:

```python
draws = dn.func.sample_initialization_draws(program)
initial = dn.func.initialize(program, draws=draws)
state, auxiliary = dn.func.run(program, initial)
```

For a lowered Population or Network, both calls accept its matching tensor
bundle:

```python
draws = dn.func.sample_initialization_draws(functional, tensors)
initial = dn.func.initialize(functional, tensors, draws=draws)
```

The sampler accepts the same `initialization=` override as `initialize`.
Compiled runners also accept `parameters=` and `constants=` overrides; supply
the same overrides when replaying. For explicit models, replace the intended
bundle fields before capture and replay.

| Packet field | Population | Network |
| --- | --- | --- |
| `random` | RAND/initialization NOISE draw packet | Draw packets keyed by Population name |
| `stimulation` | Poisson initialization draw packet | Draw packets keyed by Population name |
| `netstim` | `None` | NetStim initialization draw packet |
| `connections` | `None` | Connection weight/delay sampling draw packet |

Unused families or Population entries use `None`. The supplied packet must
contain the draws required by the selected initialization; it does not request
automatic sampling of omitted stochastic families. Newly sampled base draws
and their resulting RNG carry are detached. When an initialization recipe
already supplies legacy draw fields, capture retains their Tensor identities
and gradient flags without cloning or detaching them. Parameters used to
materialize samples retain their normal initialization gradients.

Capture follows one complete ordered host initialization under `no_grad` to
resolve sampling around transforms, caches and per-Population writes. It
returns only the draw packet, discarding the initialized state. It leaves
native sources, global RNG, runner defaults, execution caches and recording
unchanged. Replay is a separate initialization call: its transforms execute
once in the caller's gradient mode using the captured draws. Keep authored
transforms functionally pure. Capture and replay run those transforms in two
separate calls; they do not cache transform outputs between calls.

For ordinary fresh initialization, `dn.func.initialize(...)` samples missing
draws as part of its single ordered execution. There is no need to capture a
packet first. Explicit capture is useful for comparing executions, replaying a
chosen realization, or separating draw generation from differentiable
initialization. The pure Population and Network `initialize` methods also
accept `draws=InitializationDraws(...)`.

Existing family-specific samplers and initialization fields remain supported.
Choose either the canonical packet or the legacy stochastic fields:
`draws=` cannot be combined with `InitializationInput.random_draw`,
`InitializationInput.stimulation`, `NetworkInitializationInput.netstim_draw`
or `NetworkInitializationInput.connection_draw`, including per-Population
legacy fields. Conflicts are rejected before initialization hooks or sampling.
Clear retained legacy draw fields before supplying a canonical packet.

Initialization draws and runtime `RuntimeDraws` are distinct packet types.
`initialize(..., draws=...)` consumes `InitializationDraws`;
`run(..., draws=...)` consumes runtime draws. After initialization, use the
returned `Continuation` and the existing portable checkpoint API to retain the
resulting state, effective sources and RNG position.

The complete
{download}`functional_initialization_draws.py <../../examples/functional_initialization_draws.py>`
example checks automatic-versus-explicit replay, source RNG isolation, shared
parameters, scan gradients and portable continuation for both model kinds:

```console
python examples/functional_initialization_draws.py
python examples/functional_initialization_draws.py --backend aot_eager
```

```{literalinclude} ../../examples/functional_initialization_draws.py
:language: python
:start-after: "# region draws"
:end-before: "# endregion draws"
:dedent: 4
```

### Fresh starts with a convenience runner

Choose `initialization="fresh"` when no-state calls should initialize from the
current parameters, rather than restart a captured state. Edit
`runner.initialization_inputs` or supply `initialization=...` for a fresh call.
Bound steady-state caches restore their saved state and detached
materializations according to native cache semantics.

Every successful convenience-runner call returns a `dn.func.Continuation` at
`auxiliary["continuation"]`. It captures the actual effective `parameters` and
`constants` for that call, including per-call overrides and fresh/cache
initialization replacements, with its simulation state and callback carry:

```python
fresh_runner = dn.compile(
    make_cable(), dt=DT, steps=STEPS, initialization="fresh",
    callbacks={"trace": dn.Recorder(("v", "t"))},
)
state, auxiliary = fresh_runner()
continued, next_auxiliary = fresh_runner(auxiliary["continuation"])
```

A packet has `state`, `parameters`, `constants` and `callback_state` attributes.
It snapshots their container structure and retains Tensor identities, aliases
and gradient history. Its attributes are fixed; nested dictionaries and lists
are independent ordinary containers. Editing a packet does not bypass source,
schema or callback-state validation when it is resumed.
The runner does not retain the returned packet. Tensor
leaves remain live: keep their values unchanged until backward completes.
Runtime preparation is separate from the packet; emitted recordings are saved
separately when needed.
Packet calls prepare from their explicit sources; they do not replace the
runner's default bindings or populate its default inference cache.
The `(state, auxiliary)` return value and explicit keyword calls are also
supported; fresh-policy calls expose their effective mappings at
`auxiliary["parameters"]` and `auxiliary["constants"]`.

A packet supplies all four continuation operands. Do not combine it with
non-`None` `parameters=`, `constants=`, `callback_state=` or `initialization=`
overrides. `inputs=`, `draws=` and call-local `extra=` remain inputs to each
invocation; the packet does not save or replay those tapes or waveform recipes.
No-state calls retain the configured snapshot/fresh restart behavior.

The complete
{download}`functional_continuation.py <../../examples/functional_continuation.py>`
example checks Population and connected-Network continuation, per-call
parameter overrides, aliases, joined recordings, checkpointed gradients and
fresh effective bindings. It also checks raw-bound continuation and explicitly
binds restored sources. Run
`python examples/functional_continuation.py --backend aot_eager`.

(functional-packet-execution)=
### Execute an explicit continuation

`dn.func.run(functional, continuation, ...)` executes a lowered Population or
Network eagerly. Supply its compiled chunk instead to use the same host
scheduler with compiled execution:

```{literalinclude} ../../examples/functional_packet_execution.py
:language: python
:start-after: "# region packet_run"
:end-before: "# endregion packet_run"
:dedent: 4
```

The supported targets are exact `FunctionalPopulation`, `FunctionalNetwork`,
`CompiledPopulationChunk` and `CompiledNetworkChunk` objects. Supply the packet
as the second argument or as `state=`, once. This form requires a
`Continuation`; a tensor bundle or an ordinary state is not a complete source
and continuation frame. Use `dn.func.initialize(functional, tensors)` explicitly
to create a fresh packet. Execution does not initialize the model or apply a
runner's restart policy.

The packet owns the effective parameter and constant Tensor objects. Each call
binds those sources for that invocation without cloning, detaching or retaining
a new persistent source/preparation cache. Returned continuations retain live
aliases and gradients. A compiled chunk retains its kernel specializations,
including tails, and can execute another compatible packet with different
source objects.

Supply `steps=` or `duration=`, or input/draw tapes that determine the horizon.
The chunk width does not set a default run length. `host_span_steps` bounds
ordinary host work; `checkpoint_every` requests activation checkpointing, and
the two are mutually exclusive. Inputs, draws and extracellular recipes follow
the explicit bound scheduler's per-call semantics. Supply the callback
collection with `callbacks=`; it must belong to the target's functional plan and
match any retained callback carry. The first run of an initialized packet
starts callbacks, and resumed runs emit only the suffix's samples.
Source and initialization overrides are not accepted in this form, and a
non-`None` `callback_state=` cannot compete with the packet's carry.

Portable restoration also returns a packet that can be executed directly:

```{literalinclude} ../../examples/functional_packet_execution.py
:language: python
:start-after: "# region packet_restore"
:end-before: "# endregion packet_restore"
:dedent: 4
```

Inspect the restored packet with `dn.func.inspect_sources(restored)` to select
its effective optimizer leaves. Detached retained materializations can be
independent trainable sources after restoration; a suffix gradient need not
flow through their raw parameter. Live in-memory continuations preserve the
earlier gradient graph.

The complete
{download}`functional_packet_execution.py <../../examples/functional_packet_execution.py>`
example runs both a Population and a connected Network, preserving shared
sources, recording phase and nonzero gradients through live continuation and
a detached checkpoint boundary:

```console
python examples/functional_packet_execution.py
python examples/functional_packet_execution.py --backend aot_eager
```

(functional-bound-continuation)=
### Continue through a raw binding

`dn.func.run(bound, continuation, ...)` accepts a raw `BoundPopulation` or
`BoundNetwork`. Successful raw-bound host runs publish
`auxiliary["continuation"]` whether their input is an ordinary state or a packet.
An eager plan and a compiled chunk expose the same raw `bind` form.
`dn.func.run(runner, continuation)` retains the runner's configured policies.
For persistent source binding and explicit scheduling ownership, use
`bound = runner.kernel.bind(continuation)` and pass that continuation separately
to `dn.func.run`. The equivalent explicit mapping form is:

```{literalinclude} ../../examples/functional_continuation.py
:language: python
:start-after: "# region binding_bridge"
:end-before: "# endregion binding_bridge"
:dedent: 4
```

The binding remains the owner of its parameter and constant sources. Packet
Tensor identities, schemas and aliases must match those sources; equal values
in a cloned Tensor do not establish the same binding. The scheduler does not
adopt a different packet's parameters or constants. A returned packet can be
resumed through that same binding:

```{literalinclude} ../../examples/functional_continuation.py
:language: python
:start-after: "# region binding_resume"
:end-before: "# endregion binding_resume"
:dedent: 4
```

Supply the bound callback collection with `callbacks=` when continuing its
carry. The collection must match the packet's callback state, and a non-`None`
`callback_state=` cannot compete with the carry in the packet. Recording emits
only the suffix's accepted steps, without repeating the boundary sample.
`inputs=`, `draws=`, runtime `extra=`, and the horizon remain per-call arguments.
Packet containers are independent snapshots; source Tensor identities and
gradient history remain live.

For Networks, shared constant readers follow their canonical parameter source;
the binding's authored maps remain unchanged. Prepared Population bindings do
not own raw constants and use their existing explicit state/callback-state
contract. Direct binding calls, direct chunks and the legacy
`run(functional, step, state, ...)` form retain their explicit contracts.

When retaining a binding, different sources, including the owned leaves from
portable restoration, require a new raw binding. Alternatively, use
`dn.func.run(kernel, restored, ...)` to bind the restored sources for that call.
The explicit binding form is:

```{literalinclude} ../../examples/functional_continuation.py
:language: python
:start-after: "# region restored_binding"
:end-before: "# endregion restored_binding"
:dedent: 4
```

### Save and restore a detached continuation

`Continuation` retains in-memory graph history. For an independent portable
checkpoint, export against its lowered Population or Network plan:

```python
saved = dn.func.export_continuation(
    functional, auxiliary["continuation"], callbacks=callbacks,
)
torch.save(saved, path)

# Construct and lower a compatible initialized model explicitly.
restored = dn.func.restore_continuation(
    receiving, torch.load(path, weights_only=True), callbacks=receiving_callbacks,
)
state, auxiliary = dn.func.run(
    receiving_kernel, restored, steps=100, callbacks=receiving_callbacks,
)
```

Each supplied callback collection must belong to its corresponding functional
plan. Omit it when callbacks are not used. Restoration creates new owned
source leaves; passing the restored packet selects those objects for execution.
The complete
{download}`functional_explicit_continuation_checkpoints.py <../../examples/functional_explicit_continuation_checkpoints.py>`
example checks both model kinds, disk restoration, aliases, recording and
suffix gradients.

The same functions accept compiled Population and Network runners and select
their configured callbacks; omit `callbacks=` for runner targets:

```python
saved = dn.func.export_continuation(runner, auxiliary["continuation"])
torch.save(saved, path)

# Construct and initialize a compatible model and runner explicitly.
restored = dn.func.restore_continuation(
    receiver, torch.load(path, weights_only=True),
)
state, auxiliary = receiver(restored)
```

`runner.export_continuation(auxiliary["continuation"])` and
`receiver.restore_continuation(saved)`
are equivalent methods. The functions and methods use the same versioned packet
format and accept compatible checkpoints from one another. Initialized packets
begin callbacks on first execution; checkpoints from executed calls retain their
callback configuration, including when a call has zero steps.

The versioned data/Tensor packet owns detached ordinary storage and preserves
raw parameter/constant aliases and trainability flags. Restored trainable
sources are new leaves; upstream computation graphs are not restored.
Relationships between distinct saved Tensors are also cut. Fresh/cache
initialization can leave active retained values in the constant frame, so
suffix gradients may reach those effective leaves while their raw sources
have zero or unused gradients. `dn.func.inspect_sources(restored)` reports their
paths and identity-deduplicated `trainable_leaves` for explicit suffix
optimization. This reports eligibility, rather than discovering loss
dependencies. Use live in-memory continuation when initialization graphs must
remain connected.
The receiving plan validates source, state and binding schemas, required aliases
and callback compatibility. Restoration uses the same dtype/device, does not
extract native state, initialize, prepare, sample or execute a model, and leaves
receiver defaults and RNG unchanged.
Save runtime input/draw tapes, model declarations, optimizer state and earlier
recordings separately. See {doc}`A58_functional_callback_checkpoints` and the
complete
{download}`functional_continuation_checkpoints.py <../../examples/functional_continuation_checkpoints.py>`
example for fresh replacements, shared sources and suffix gradients.

Custom pure callbacks support portable continuation by implementing
`FunctionalCallback.checkpoint_configuration()`, which describes their static
configuration as versioned plain data and Tensor metadata. Construct receiving
callbacks explicitly; restoration checks their configuration and carry/emission
schemas. Callbacks without the optional hook retain live in-memory continuation.
See {ref}`functional-callback-checkpoint-protocol` for the contract and a complete
Population/scoped-Network example.

Callback carry is tagged with its in-memory collection's identity. Portable
restoration uses the checked callback export/restore bridge to attach
the receiving identity. Independently managed callbacks use
`callbacks.export_state(...)` and `callbacks.restore_state(...)` directly.
Starting a collection without restored carry starts new reductions and includes
its initial boundary sample. Native state is published only through an explicit
`functional.commit_state_(...)` call.

## Choose an execution level deliberately

| Need | Entry point | Important boundary |
| --- | --- | --- |
| A reusable ordinary simulation or optimizer loop | `dn.compile`, then `dn.func.run(program, ...)` or `program(...)` | Default or per-call horizon; no-state calls restart |
| Explicit sources and scheduling ownership | `dn.func.run(plan_or_kernel, continuation, ...)`, or a raw bound step/chunk | Host operation; handles sampling, spans and tails |
| Exactly one step or a fixed width | Eager binding or compiled chunk | Caller supplies the frame/tape and stochastic packet |
| `torch.func` transforms or an outer `torch.compile` | Pure `prepare_and_step` / `prepare_and_rollout` | Keep host scheduling and sampling outside the transform |

`dn.compile` selects scan by default. First-order training is supported by scan
and compiled unrolled chunks. `execution="while"` is inference-only. Both engines
accept built-in and capture-compatible custom `FunctionalCallback` objects,
including Population scopes in a Network, with the same explicit carry and
emission contract. Population while runners require a compiled
chunk width of at least four steps. Both kinds of runner support callback-free
while inference; Networks also support short main chunks. Requested unsupported
while execution raises rather than silently choosing another strategy for the
main chunk. Shorter tails retain their bounded dispatch rules.

Raw Population and Network chunks also accept `execution="while"`:

```python
kernel = functional.compile_rollout_chunk(128, execution="while")
bound = kernel.bind(tensors)
with torch.no_grad():
    state, auxiliary = dn.func.run(bound, tensors.state, steps=1000)
```

For Populations, explicit while chunks use the shared inference engine and
reject unavailable capture for widths of at least four steps. Raw widths and
host tails of one to three steps use a bounded unrolled path. The Population
default selector retains automatic inference selection and first-order
training; selecting explicit while does not alter other kernels or a
`dn.compile` runner's exposed kernel. `disable=True` retains eager diagnostic
execution, and active transforms retain the eager fallback described below.

`vmap`, forward-mode AD and higher-order differentiation have a different
execution contract from ordinary first-order compiled training. Supported
transforms can execute through the eager pure path or its fallback; a compiled
wrapper does not establish that the transformed recurrence is accelerated.
Apply transforms to a pure combined preparation-and-execution function, keep
its tensors explicit, and qualify the particular transform/model combination.
Do not place host runners or automatic sampling inside `torch.func` or an outer
compiled region. See {doc}`A8_functional_populations` for supported transform
compositions and limitations.

The complete example differentiates a deterministic single-compartment
rollout with respect to a scalar current multiplier, and batches several
multiplier values. It compares `torch.func.grad` with ordinary autograd and
`vmap` with separate runs. Preparation belongs inside the transformed function;
the plan, raw sources and initial state are explicit captured operands:

```{literalinclude} ../../examples/functional_api_overview.py
:language: python
:start-after: "# region transforms"
:end-before: "# endregion transforms"
```

This section executes eagerly. Its parity checks establish the transform
contract for the example; they do not claim compiled acceleration.

The main remaining API asymmetries are visible at the advanced boundary:
Population and Network retain different state layouts; Population extraction
exposes native parameters while Network extraction takes detached snapshots;
prepared Population bindings and raw Network bindings use different positional
arguments; eager frames and compiled tapes differ by a leading time dimension;
and the legacy packet/callable scheduler forms remain available. The raw
`constants=` binding, `Inputs`, `RuntimeDraws`, named callbacks and `run` contract
give application code a consistent route across those differences. Use the
additional forms only when their explicit ownership or compatibility behavior
is needed.
