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

Network runners currently support `initialization="snapshot"` and
`execution="scan"`. The snapshot includes Population carry, connection queues,
NetStim clock/RNG/calendar state, and the authoritative Network clock. A call
without explicit state restarts that complete snapshot; compilation does not
advance or reinitialize the source Network. `runner.initial_state` exposes the
restart packet, whose Tensor leaves can be edited deliberately.

Initialization is not recomputed after a parameter update. Gradients therefore
do not include fresh Network initialization. Fresh initialization
(`initialization="fresh"` and runtime initialization overrides) and inference
while loops are currently unsupported. Use a canonical single-device Network
with supported dense event queues, as described in
{doc}`A19_functional_network_lowering`.

`backend="aot_eager"` makes this example portable on CPU. The compiler default
is Inductor. For CPU inference with Inductor, Dendra requests a C++ wrapper
unless explicit compiler options choose otherwise. Compilation is lazy,
and the runner is called from Python. Use the lower-level pure API when
composing `torch.func` or forward-AD transforms.

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

For example, Population names prefix their raw parameter paths, while
connection weights use `synapses.<connection-name>.weights`. Continuous
transform parameters use
`continuous_synapses.<connection-name>.transform_parameters.<parameter-name>`.
NetStim renewal parameters appear under `netstim.parameters`, and schedule
values retain named fields such as `schedule.times` and
`schedule.amplitudes`. `runner.parameter_name(query)` resolves an exact name
or one unique partial match; `runner.get_parameter(query)` returns that leaf.
Ambiguous queries raise with the matching names.

Selecting any alias in `trainable` turns the shared independent leaf into one
`torch.nn.Parameter`. All aliases still refer to that same object, so their
gradient contributions accumulate. Distinct Tensor views are independent
leaves unless they were the same Tensor object in the original packet.
Unselected leaves are detached ordinary tensors.

An optimizer should receive each shared object only once:

```python
unique_trainable = {
    id(value): value
    for value in runner.named_parameters.values()
    if value.requires_grad
}
optimizer = torch.optim.Adam(list(unique_trainable.values()), lr=1e-3)
```

Use `runner(parameters=packet, ...)` for an explicit nested replacement packet.
Replacement tensors retain their autograd history and must preserve the
Network's shared parameter identities and component schema. The runner does
not retain a replacement packet or its preparation graph between calls.
`NetworkTensors` from `make_functional_network` also supplies
`named_parameters`, `find_parameters`, `parameter_name`, `get_parameter`, and
`independent_parameters(trainable=...)` for lower-level use.

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

No-state calls restart the NetStim RNG snapshot as well as its event clock.
Explicit continuation advances from the returned RNG position. Constructor
`extra=` binding is currently unsupported for Network runners; supply
per-population `ve` tapes or supported registered stimulation.

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

Keep parameters, constants, input tapes, and source configuration unchanged
until backward finishes. Hard event values and configured surrogate gradients
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
