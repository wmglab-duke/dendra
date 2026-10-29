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
returns explicit simulation state. Use `make_functional(model, dt=...)`
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

## Inference: compile, call, retain state explicitly

`dn.compile` defaults to `inference=True` and `initialization="snapshot"`.
It enters inference mode during execution. Every no-state call restarts from
the initialized snapshot; an explicit state continues that simulation. It does
not advance the native model or overwrite its snapshot.

`steps` fixes each call's horizon; `chunk_steps` selects the compiled width for
both Populations and Networks:

```python
runner = dn.compile(model, dt=DT, steps=1_000, chunk_steps=128)
state, auxiliary = runner()  # All 1,000 steps, including the shorter tail.
```

The positive chunk width is capped at `steps` and need not divide it. Omission
uses the full horizon for ordinary runners, or at most 64 steps for runtime
`extra=dn.VarInput()` slots. Registered and compile-time bound stimulation use
the ordinary setting. Chunking does not remove requested recording storage.

Recording is an explicit named callback. By default a fresh Recorder includes
the initial sample, followed by every accepted step. `Recorder(dt=record_dt)`
selects a cadence with the native in-memory sampling rule; its phase is explicit
carry and survives chunks and continuation. Different Recorders can use different
cadences. Host schedules retain selected frames between emission segments.
With bounded chunks or host/checkpoint spans, that Recorder's payload scales
with selected history plus bounded dense segments. Each compiled invocation still
emits dense data; full-rate/custom streams, autograd and input tapes have
separate storage requirements. Atomic rollouts retain dense finalization; see
{ref}`functional-recorder-cadence` for timing and compilation details.
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

## Explicit bindings and direct chunks

`make_functional(model, dt=...)` returns a plan and explicit tensor bundle for
an initialized Population or Network. A Network's supplied `dt` must match its
initialized timestep. The factory returns the concrete `FunctionalPopulation` /
`PopulationTensors` or `FunctionalNetwork` / `NetworkTensors` types; it preserves
their distinct state layouts and extraction ownership. The existing
`make_functional_network(network, ...)` spelling remains available and uses the
initialized Network timestep implicitly. Use the same raw binding spelling for
eager and compiled execution:

```python
eager = functional.bind(tensors)
kernel = functional.compile_rollout_chunk(2, backend="aot_eager")
bound = kernel.bind(tensors)
```

`bind` accepts the matching `PopulationTensors` or `NetworkTensors` bundle, or
a `Continuation` from initialization, execution or portable restoration:

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
Population raw bindings also reuse safe inference preparation. Compiled kernels
can be reused with other valid
bindings; they do not own optimization leaves or an autograd graph.

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

(functional-explicit-initialization)=
### Initialize explicit sources

`dn.func.initialize` initializes a lowered Population or Network from its
matching tensor bundle and returns a `dn.func.Continuation`:

```python
functional, tensors = dn.func.make_functional(model, dt=DT)
initial = dn.func.initialize(functional, tensors)
callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("v", "t"))})
bound = functional.bind(initial)
state, auxiliary = dn.func.run(bound, initial, steps=100, callbacks=callbacks)
```

For a Network, use qualified Recorder paths such as `"left.v"`. Bind the
returned parameter and constant objects: initialization transforms and caches
can replace the incoming sources. Their effective identities, aliases and
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
    callbacks={"trace": dn.func.Recorder(("v", "t"))},
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

(functional-bound-continuation)=
### Continue through a raw binding

`dn.func.run(bound, continuation, ...)` accepts a raw `BoundPopulation` or
`BoundNetwork`. Successful raw-bound host runs publish
`auxiliary["continuation"]` whether their input is an ordinary state or a packet.
An eager plan and a compiled chunk expose the same raw `bind` form. To execute
a runner's continuation through its compiled kernel, use
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

Different sources, including the owned leaves from portable restoration,
require an explicit new raw binding. The reusable kernel and receiving
callback collection can still be used:

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
bound = receiving_kernel.bind(restored)
state, auxiliary = dn.func.run(
    bound, restored, steps=100, callbacks=receiving_callbacks,
)
```

Each supplied callback collection must belong to its corresponding functional
plan. Omit it for callback-free continuation. Restoration creates new owned
source leaves; bind those exact restored objects before resuming. The complete
{download}`functional_explicit_continuation_checkpoints.py <../../examples/functional_explicit_continuation_checkpoints.py>`
example checks both model kinds, disk restoration, aliases, recording and
suffix gradients.

Both convenience runners provide the same operations as methods:

```python
saved = runner.export_continuation(auxiliary["continuation"])
torch.save(saved, path)

# Construct and initialize a compatible model and runner explicitly.
restored = receiver.restore_continuation(torch.load(path, weights_only=True))
state, auxiliary = receiver(restored)
```

The functions and methods use the same versioned packet format and accept
compatible checkpoints from one another.

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
| A reusable ordinary simulation or optimizer loop | `dn.compile` | Fixed configured horizon; no-state calls restart |
| An explicit scheduler for variable horizons | `dn.func.run` with a raw bound step/chunk | Host operation; handles sampling, spans and tails |
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
