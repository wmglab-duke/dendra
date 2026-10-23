---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-network-compile)=
# Compile an ordinary Network

Pass an initialized standard `Network` directly to `dn.compile` to obtain a
reusable functional runner. Calls return `(state, aux)` with explicit state,
qualified recordings, and gradients. The source Network remains unchanged.
Network declarations, pure mechanism hooks, event delivery, and continuous
transforms follow the contracts in {doc}`A19_functional_network_lowering` and
{doc}`A20_functional_network_rollout`.

The complete
{download}`functional_network_compile.py <../../examples/functional_network_compile.py>`
example compiles the ordinary recurrent Network from the previous tutorial.
That model contains two Populations, a noisy NetStim, threshold connections,
and a trainable continuous release transform. The example shares one
weight Tensor between its two event connections and checks compiled results
against an uninterrupted eager rollout. Run from the repository root with
`python examples/functional_network_compile.py` or
`python -m examples.functional_network_compile`.

## Compile an initialized snapshot

Configure the Network's mode, timestep, and differentiability before compiling.
The example's `make_network()` performs ordinary construction, training-mode
initialization, and synaptic differentiation setup:

```{literalinclude} ../../examples/functional_network_compile.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 4
```

`dt` must equal the initialized Network timestep. `steps` is the fixed positive
number of accepted timesteps per runner call. The default `inference=True`
disables gradient recording; select `inference=False` for differentiable
execution, using a Network initialized in training mode. Training mode,
topology, component schemas, and timestep are fixed by the compiled plan.
Changing them requires a new runner; initialize the Network again when its
configuration requires it, including after changing the timestep.

Network runners support `initialization="snapshot"` (the default),
`initialization="fresh"`, and `execution="scan"` or inference-only
`execution="while"`. The snapshot includes Population carry, connection queues,
NetStim clock/RNG/calendar state, and the authoritative Network clock. A call
without explicit state restarts that complete snapshot; compilation does not
advance or reinitialize the source Network. `runner.initial_state` exposes the
restart packet, whose Tensor leaves can be edited deliberately.

With snapshot initialization, initial state is not recomputed after a parameter
update and gradients do not include initialization. Select fresh initialization
to include that computation. Inference while loops share Population's packed
carry and recording engine; see {doc}`A25_functional_network_while` for execution,
recording, and continuation examples. Scan remains the default and supports
training gradients and custom callbacks.
Apply Population observation callbacks within a Network using
`PopulationCallback("left", callback)`. LFP, threshold analysis, anomaly
detection, and scoped Recorders support scan and while execution; see
{doc}`A27_functional_network_callbacks`.
Use a canonical single-device Network with supported dense event queues,
source-history training, packed-history or sparse-calendar inference, as
described in {doc}`A19_functional_network_lowering`,
{doc}`A22_functional_source_history`, {doc}`A23_functional_bitpacked_inference`,
and {doc}`A24_functional_pending_calendar`.

`backend="aot_eager"` makes this example portable on CPU. The compiler default
is Inductor. For CPU inference with Inductor, Dendra requests a C++ wrapper
unless explicit compiler options choose otherwise. Compilation is lazy,
and the runner is called from Python. Use the lower-level pure API when
composing `torch.func` or forward-AD transforms.

Scan and while use the same tensor-view ownership rules as Population chunks.
Expanded and strided parameter, prepared-buffer, and input-tape views retain
their authored strides and storage offsets. Invocation-local copies isolate
loop operands without modifying caller tensors; scan gradients accumulate
through shared sources and overlapping views across checkpointed spans and
explicit continuation.

## Initialize from current parameters

Select `initialization="fresh"` to construct initial state from current
parameters on each call without explicit state. Population initialization,
connection resets, and NetStim initialization use the same pure components as
their native adapters. Initialization runs once before rollout; checkpoint
backward replays rollout spans without rerunning initialization or consuming
another initialization draw.

The complete
{download}`functional_network_initialization.py <../../examples/functional_network_initialization.py>`
example uses a noisy NetStim and differentiates through fresh Population
initialization and two continued, checkpointed calls:

```{literalinclude} ../../examples/functional_network_initialization.py
:language: python
:start-after: "# region fresh-compile"
:end-before: "# endregion fresh-compile"
:dedent: 4
```

`runner.initialization_inputs` is an independent `NetworkInitializationInput`.
Its `populations` mapping contains one ordinary `InitializationInput` per
Population, with expanded `v_init`, insertion-time `states`, and registered
pure transform inputs. Edit their Tensor leaves or supply a complete
`initialization=` override. These inputs are separate from runtime parameters
and are not selected by `trainable`; set `requires_grad_()` on them to include
them in an objective. `t` is the finite scalar CPU Float64 starting time and
defaults to zero. It controls the Network clock and absolute schedule position;
it is detached during initialization.

```{literalinclude} ../../examples/functional_network_initialization.py
:language: python
:start-after: "# region initialization-inputs"
:end-before: "# endregion initialization-inputs"
:dedent: 4
```

`netstim_draw` optionally supplies an explicit initialization `NetStimDraw`.
When omitted, the host runner samples once from its restart state's RNG carry.
A fixed NetStim seed reseeds initialization; `seed=None` continues that supplied
stream. Deterministic initialization consumes no random draw. The initial
NetStim clock kick is detached, matching native initialization; subsequent
renewal retains its configured gradient policy. Consumed heap cursors and
seeder state are preserved, while delivery queues and detector carry reset.
Neither the source Network's RNG nor its state is changed.

The lower-level `FunctionalNetwork.initialize` consumes explicit parameters,
constants, prior carry, and initialization inputs, returning `NetworkTensors`.
Supply an initialization draw when the graph contains a NetStim. The prior
carry supplies calendar cursors and RNG/seeder state; it does not replace
fresh Population state. To prepare runtime draw tapes, sample from the returned
initialized state:

```{literalinclude} ../../examples/functional_network_initialization.py
:language: python
:start-after: "# region explicit-initialization"
:end-before: "# endregion explicit-initialization"
:dedent: 4
```

Initialization follows native order: Populations in declaration order, Network
clock reset and Population clock synchronization, event connections,
continuous connections, then NetStim. Parameter replacements from each
Population initialization propagate across every shared parameter alias before
the next Population initializes or connections materialize their weights and
delays. `NetworkInitializationInput.reinit_weights` and `reinit_delays` default
to `True`, refreshing those values from their raw sources. Set either flag to
`False` to retain its preceding explicit materialization. Return and reuse
`aux["constants"]` to preserve that materialization across fresh initialization
boundaries. Fresh-policy calls return the resulting
effective parameter packet in `aux["parameters"]`; initialization does not
overwrite the runner's independent optimizer leaves. If an initialization
transform writes parameters, continue with that effective packet explicitly:

```{literalinclude} ../../examples/functional_network_initialization.py
:language: python
:start-after: "# region fresh-continuation"
:end-before: "# endregion fresh-continuation"
:dedent: 4
```

Explicit state bypasses fresh initialization, and cannot be combined with an
`initialization=` override. Gradients through the prefix remain attached to
the initialized state and effective parameters:

```{literalinclude} ../../examples/functional_network_initialization.py
:language: python
:start-after: "# region initialization-gradients"
:end-before: "# endregion initialization-gradients"
:dedent: 4
```

Fresh initialization uses the existing frozen topology. When the source
Network has a Population state cache, it restores the cached Population
parameters, constants, and carry, and rebuilds connection state using native
cache semantics. Fresh results return both effective `aux["parameters"]` and
`aux["constants"]` for explicit continuation. See
{doc}`A29_functional_network_cache_initialization` for cache capture, current
connection parameters, gradient boundaries, and required continuation packets.
Without a cache, fresh initialization uses current explicit parameters as
described above. Ion and Material initial values can use `PositiveParam` or
deterministic zero-argument modules with registered parameters and buffers;
shared sources retain one raw parameter binding across populations. See
{doc}`A36_functional_initialization_sources` for source purity, gradients, and
cache behavior. Custom `MaterialProcess` implementations use
their shared pure field hooks for native and functional execution; see
{doc}`A37_functional_material_processes` for process authoring, gradients, and
continued Network runs. MaterialProcess RAND and NOISE declarations use the
same explicit sampling packets; see {doc}`A44_functional_material_stochasticity`.
Initialization-time RAND declarations use explicit draw and
RNG packets, including their resampling and ``reseed_on_initialize`` policies;
see {doc}`A30_functional_random_parameters`. Per-stream model/cache load
policies use `ignore_rng_on_load(...)`; see
{doc}`A48_functional_rng_load_policies`. Runtime NOISE declarations use
per-Population draw/RNG tapes, with automatic host sampling and explicit
continuation; see {doc}`A31_functional_runtime_noise`. RNGModule owners shared
across Populations use one authoritative carry per stream and preserve native
sampling and reseeding order; see {doc}`A47_functional_network_rng_sharing`.
State `DIFFUSION` uses
the same packet for explicit Wiener draws and RNG carry, with pathwise
drift/diffusion gradients; see {doc}`A32_functional_state_sde`. Custom native
initialization/build adapters are currently unsupported.
Shared runtime parameters and parameter/configuration aliases follow authored
Population initialization order. Earlier Populations retain their prepared
numerical values when a later initializer writes a shared source; the final raw
parameter frame preserves its shared aliases. Explicit continuation carries
those per-Population materializations in the returned constants. See
{doc}`A43_functional_initialization_ordering` for transforms, gradients, and
mixed cached/fresh initialization. Parameter and buffer aliases within one
Population use its canonical parameter binding when available.
Pure PyTorch `register_parametrization(...)` transforms use the shared
parameter materializer with explicit original tensors, transform parameters,
and fixed buffers. Shared trainable parameters retain all their qualified
Network aliases. See {doc}`A38_functional_torch_parametrizations` for parameter
selection, initialization, caches, and continued checkpointed gradients.
Population initialization hooks must satisfy the pure initialization contracts
in {doc}`A8_functional_populations`; unsupported hooks raise an error without
falling back to snapshot state. Explicit continuation and snapshot execution
retain their existing admission rules.

## Keep shared trainable parameters shared

`runner.parameters` is the nested explicit parameter packet. Its container
structure is read-only; Tensor leaves are independent, editable copies of the
source Network values. `runner.named_parameters` presents the same leaves
under collection/component-qualified dotted names:

```{literalinclude} ../../examples/functional_network_compile.py
:language: python
:start-after: "# region parameters"
:end-before: "# endregion parameters"
:dedent: 4
```

For example, Population names prefix their raw parameter paths. A connection
weight supplied as a `PositiveParam` uses a raw path such as
`synapses.<connection-name>.weight.weights.0.param.rho`. Delay sources use the
corresponding `delay_ms.weights` path. The numerical weight and physical delay
are derived during preparation, retaining their constraints and any supported
PyTorch parametrizations. Continuous
transform parameters use
`continuous_synapses.<connection-name>.transform_parameters.<parameter-name>`.
NetStim renewal sources appear under `netstim.parameters`, including the
positive interval's raw `interval.rho` and PyTorch parametrization originals
and transform parameters. Schedule values retain named fields such as
`schedule.times` and `schedule.amplitudes`. See
{doc}`A39_functional_connection_parameters` for connection source registration,
shared optimizer leaves, and refresh policies, and
{doc}`A40_functional_netstim_parameters` for NetStim source selection,
initialization, and random-state continuation.
Built-in and custom connection distributions with explicit-draw hooks retain
their raw sources and support fresh sampling with explicit draws and RNG carry;
see
{doc}`A41_functional_connection_sampling` for initialization, resampling flags,
and checkpointed gradients, and {doc}`A49_functional_distribution_sampling`
for the custom distribution contract.

`trainable` accepts exact names, unique partial names, glob patterns, and
explicit regex selectors. For example:

```python
import re

runner = dn.compile(
    network, dt=dt, steps=100, inference=False,
    trainable=(
        "synapses.*.weight.weights.*.param.*",
        re.compile(r"^populations\.cell\..*\.leak\.g_param$"),
    ),
)
```

Globs match the complete dotted name case-sensitively; `*` spans dots.
`glob:synapses.*.weight.weights.*.param.*` is equivalent to the bare glob above.
Regex selectors
use `re:` strings or compiled string patterns, with regex `search` semantics.
For example, `r"re:^synapses\..*\.(weight|delay_ms)\.weights\."` selects the
available raw event weight and delay source parameters. A fixed delay remains
frozen routing configuration; changing its rounded timestep requires lowering
the Network again. Empty/invalid selectors and unmatched
`trainable` selectors raise an error. A plain substring still must identify
one name, while a pattern can select several. Selected leaves must have a
floating-point or complex dtype; narrow broad patterns that also match
integer or Boolean buffers. The full matching rules also apply to Population
runners and `independent_parameters`, as described in
{ref}`functional-parameter-selection`.

Preview candidates before compiling, or search a runner's independent copies:

```python
_, tensors = dn.func.make_functional_network(network)
print(tuple(tensors.find_parameters("synapses.*.weight.weights.*")))
print(tuple(runner.find_parameters(r"re:\.(weight|delay_ms)\.weights\.")))
```

`runner.parameter_name(query)` resolves an exact name or one unique selector
match; `runner.get_parameter(query)` returns that leaf. Both reject ambiguous
matches, including when several names alias the same Tensor.

Selecting any alias in `trainable` turns the shared independent leaf into one
`torch.nn.Parameter`. All aliases still refer to that same object, so their
gradient contributions accumulate. Distinct Tensor views are independent
leaves unless they were the same Tensor object in the original packet.
Unselected leaves are detached ordinary tensors. Inspect the actual selected
leaves with `runner.trainable_parameters`:

```python
print(tuple(runner.trainable_parameters))
for name, parameter in runner.trainable_parameters.items():
    print(name, tuple(parameter.shape), parameter.dtype, parameter.requires_grad)
```

This named mapping contains every alias whose live Tensor has
`requires_grad=True`, including aliases outside the selector that share a
selected object. Each access reflects explicit freezing or unfreezing. It
describes the runner's default parameter packet, rather than an invocation-local
packet supplied with `parameters=`. `inference=True` suppresses simulation gradients
even when this mapping is nonempty; use `inference=False` for training.

An optimizer should receive each shared object only once:

```python
unique_trainable = {
    id(value): value
    for value in runner.trainable_parameters.values()
}
optimizer = torch.optim.Adam(list(unique_trainable.values()), lr=1e-3)
```

Use `runner(parameters=packet, ...)` for an explicit nested replacement packet.
Replacement tensors retain their autograd history and must preserve the
Network's shared parameter identities and component schema. The runner does
not retain a replacement packet or its preparation graph between calls.
`NetworkTensors` from `make_functional_network` also supplies
`named_parameters`, `trainable_parameters`, `find_parameters`, `parameter_name`,
`get_parameter`, and `independent_parameters(trainable=...)` for lower-level
use.

The complete CPU
{download}`functional_parameter_selection.py <../../examples/functional_parameter_selection.py>`
example checks wildcard and regex selection, parameter inspection, and backward
gradients for both a Population and a Network with shared connection weights:

```{code-cell} python
:load: ../../examples/functional_parameter_selection.py
:tags: [remove-input]
```

## Supply Population inputs and explicit NetStim tapes

Supply runtime inputs with `NetworkRolloutInput`:

```{literalinclude} ../../examples/functional_network_compile.py
:language: python
:start-after: "# region inputs"
:end-before: "# endregion inputs"
:dedent: 4
```

Each provided field has one leading time axis. Each runner call receives
exactly `runner.kernel.steps` samples. The example prepares six samples and
slices them into two three-step calls. Per-population `intra` values are
assembled current densities in mA/cm²; `ve` contains external potentials in
mV for integrators that accept them. Their remaining dimensions follow each
Population's input contract. Tensor slices preserve gradient history.

`runner.functional.sample_draw_tape(state, steps)` samples NetStim packets
outside compiled execution, from an independent copy of explicit RNG state.
Keep `exp_rand` and matching post-draw `rng_state` bytes together when slicing.
The runner can also omit the NetStim tape and sample bounded packets from its
input state's RNG carry. Checkpoint backward reuses the forward packets.
The example checks that automatic sampling matches the supplied tape:

```{literalinclude} ../../examples/functional_network_compile.py
:language: python
:start-after: "# region automatic-draws"
:end-before: "# endregion automatic-draws"
:dedent: 4
```

Snapshot calls without state restart the NetStim RNG snapshot and event clock.
Fresh calls apply the initialization sampling policy described above.
Explicit continuation advances from the returned RNG position. Constructor
`extra={population_name: specification}` binds each selected Population's
extracellular contacts and Waveforms. Their field and waveform leaves support
trainable selection, shared aliases, and the same continuation clock; see
{doc}`A28_functional_network_extra`. An explicit `ve` tape remains available for
any Population without bound `extra`.

## Restart or continue explicitly

Repeated no-state calls with the same parameters and input tapes produce the
same result, without consuming the native Network's clock or RNG:

```{literalinclude} ../../examples/functional_network_compile.py
:language: python
:start-after: "# region restart"
:end-before: "# endregion restart"
:dedent: 4
```

Pass returned state to continue. Preserve `aux["callbacks"].state` when
continuing recordings and reducers from the same runner:

```{literalinclude} ../../examples/functional_network_compile.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 4
```

A fresh callback collection records the initial frame, then one frame after
each accepted timestep. Continued collections emit only new frames, so joined
recordings contain one copy of the boundary. The suffix recording belongs to
that call; concatenate it with the prefix to retain the whole trace. Supplying
`callback_state` requires explicit simulation state as well.

Keep returned state attached to its autograd graph to differentiate through
the prefix. Detach independent copies when intentionally starting a separate
optimization problem. To continue natively, publish state with
`runner.functional.commit_state_(matching_network, state)` and align its
parameters separately; runner calls do not commit anything automatically.

To branch a native Network independently, use `copy.deepcopy(network)`. Its
event, continuous-connection, and Population clocks refer to the copied owners.
Stored runtime graphs are detached during copying; use explicit functional
continuation to retain gradients through a prefix. See
{ref}`runtime-contract-validation` for native copy and reinitialization semantics.

## Preserve gradients through checkpoints and continuation

With `inference=False`, `checkpoint_every=H` trades replay of H-step spans for
lower activation storage. Exact shorter tails are executed when H does not
divide the call's horizon. Gradients include the entire prefix and suffix;
checkpointing does not truncate their history. Requested recordings still
occupy their normal output storage.

The example compares the compiled two-call run to one uninterrupted eager
rollout using the same explicit parameter leaves, drive tape, and NetStim
packets. Its objective differentiates shared event weights, the release
transform, NetStim noise, and the Population input tape:

```{literalinclude} ../../examples/functional_network_compile.py
:language: python
:start-after: "# region gradients"
:end-before: "# endregion gradients"
:dedent: 4
```

Keep parameters, constants, initialization-only inputs, draw packets, runtime
input tapes, and source configuration unchanged until backward finishes.
Hard event values and configured surrogate gradients
retain their native meaning. Scan supports ordinary first-order reverse-mode
gradients; use lower-level eager execution for higher derivatives.

`runner.execution_report()` returns the most recent successful kernel
dispatch, including a shorter tail or checkpoint replay. It contains scalar
metadata and starts as `None` before execution. It is not a whole-run timing
report. `runner.kernel.capabilities` describes the execution and
differentiation contract:

```{literalinclude} ../../examples/functional_network_compile.py
:language: python
:start-after: "# region report"
:end-before: "# endregion report"
:dedent: 4
```

The complete CPU example executes here:

```{code-cell} python
:tags: [remove-input]
from pathlib import Path
import sys

sys.path.insert(0, str(Path("../../examples").resolve()))
```

```{code-cell} python
:load: ../../examples/functional_network_compile.py
:tags: [remove-input]
```

The fresh-initialization example executes here:

```{code-cell} python
:load: ../../examples/functional_network_initialization.py
:tags: [remove-input]
```
