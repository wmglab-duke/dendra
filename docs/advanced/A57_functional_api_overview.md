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
returns explicit simulation state. Use `make_functional` or
`make_functional_network` when an application needs to control bindings, chunk
scheduling, explicit draws, or PyTorch function transforms.

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

Recording is an explicit named callback. A fresh Recorder includes the initial
sample, followed by every accepted step. Supply both simulation state and
callback state to continue without recording the shared boundary twice:

```{literalinclude} ../../examples/functional_api_overview.py
:language: python
:start-after: "# region inference"
:end-before: "# endregion inference"
```

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

The runner's parameter mappings are read-only containers with editable Tensor
leaves. Per-call `parameters=` and `constants=` overrides retain the supplied
tensor identities and autograd history and must preserve the plan's schema and
ties. Those overrides do not replace the runner-owned defaults reported by its
inspection properties. Low-level Population extraction exposes native raw
parameter tensors; Network extraction returns independent detached snapshots.
`tensors.independent_parameters(...)` constructs independent optimization
leaves while preserving their aliases in either case.

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

`make_functional(population, dt=...)` and `make_functional_network(network)`
return a plan and explicit tensor bundles. Use the same raw binding spelling
for eager and compiled execution:

```python
eager = functional.bind(parameters, constants=constants)
kernel = functional.compile_rollout_chunk(2, backend="aot_eager")
bound = kernel.bind(parameters, constants=constants)
```

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

Choose `initialization="fresh"` when no-state calls should initialize from the
current parameters, rather than restart a captured state. Edit
`runner.initialization_inputs` or supply `initialization=...` for a fresh call.
Bound steady-state caches restore their saved state and detached
materializations according to native cache semantics.

Fresh initialization can return replacement parameter or constant mappings.
The result exposes these effective mappings explicitly. A continuation carries
them with the simulation and callback state:

```python
fresh_runner = dn.compile(
    make_cable(), dt=DT, steps=STEPS, initialization="fresh",
    callbacks={"trace": dn.func.Recorder(("v", "t"))},
)
state, auxiliary = fresh_runner()
continued, next_auxiliary = fresh_runner(
    state,
    parameters=auxiliary["parameters"],
    constants=auxiliary["constants"],
    callback_state=auxiliary["callbacks"].state,
)
```

For an in-memory training continuation, retain tensors with their existing
graphs. For an independent checkpoint, detach and clone the tensor trees. Save
effective parameters/constants as needed and complete simulation state,
including clocks, queues, delay histories and RNG carry. Tensor checkpoints can
be serialized with PyTorch and restored against a compatible functional plan.
Callback carry is tagged with the identity of its in-memory callback collection.
Pass it directly when continuing with that collection. For a reconstructed
runner, save `runner.callbacks.export_state(auxiliary["callbacks"].state)` and
restore it through `receiver.callbacks.restore_state(packet)` before continuing.
The checked checkpoint bridge supports built-ins and retains detached carry,
phase and discovered emission metadata; see {doc}`A58_functional_callback_checkpoints`
for a complete disk round-trip example and compatibility rules. Starting a new
callback collection without restored carry starts new reductions and includes
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
and compiled unrolled chunks. `execution="while"` is inference-only and supports
its defined built-in callbacks, including supported Population scopes in a
Network; custom callbacks use scan. Population while runners require at least
four steps and a nonempty supported callback collection. Network while runners
also support callback-free execution and short tails. Requested unsupported
while execution raises rather than silently choosing another strategy.

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
Population and Network have separate lowering functions and state layouts;
Population extraction exposes native parameters while Network extraction takes
detached snapshots;
prepared Population bindings and raw Network bindings use different positional
arguments; eager frames and compiled tapes differ by a leading time dimension;
and the legacy packet/callable scheduler forms remain available. The raw
`constants=` binding, `Inputs`, `RuntimeDraws`, named callbacks and `run` contract
give application code a consistent route across those differences. Use the
additional forms only when their explicit ownership or compatibility behavior
is needed.
