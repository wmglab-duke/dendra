---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-network-while)=
# Network inference with while loops

`dn.compile(Network, execution="while", inference=True)` executes the existing
pure Network timestep inside a captured `torch.while_loop`. Population and
Network inference share the same packed carry and recording engine. The graph
contains a four-step body and a tail of at most three steps, independently of
the requested horizon. Calls return explicit `(state, aux)` packets.

The runnable
{download}`functional_network_while.py <../../examples/functional_network_while.py>`
example compares while and scan execution for two Populations with recurrent
event/continuous traffic, noisy NetStim, shared parameters, input tapes,
recordings, and native checkpoint continuation. Run it from the repository root
with `python examples/functional_network_while.py`.

## Compile and record inference

```{literalinclude} ../../examples/functional_network_while.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 4
```

The Network must be initialized before compilation and satisfy
{doc}`A19_functional_network_lowering`. The selected native mode, connection
runtimes, timestep, topology, and parameter ties remain fixed. Inference
suppresses gradients, including when the native Network was initialized in
training mode. Use `execution="scan", inference=False` for gradients and
`checkpoint_every` for training recomputation.

While execution supports qualified `dn.func.Recorder` selections, built-ins
such as LFP and threshold analysis wrapped in `dn.func.PopulationCallback`,
and custom pure `FunctionalCallback` implementations. A custom callback may
observe the complete Network packet or one scoped Population. Its carry and
emissions have fixed tensor schemas, using the same `initialize`, `update`,
and `finalize` methods as scan. See {doc}`A27_functional_network_callbacks` for
scoping, reducers, and callback-relative windows. Omit `callbacks` when only the
final state is needed. Explicit while requests raise when capture or compilation
is unsupported for the configuration.
Compilation must be enabled for the convenience runner.

Physical recordings can use a Population Slice, for example
`dn.func.Recorder(("left.v", "t"), region=network.left.soma)`. Named regions,
concatenated components, batch selection, and continuation use the same selector
as scan; see {doc}`A26_functional_network_regions`.

Recording buffers are allocated within each invocation. Callback phase and
continuation carry are explicit; neither buffers nor live parameter/preparation
graphs are retained in kernel caches. The public return has independent
storage, including leaves that were repeated within private packed carry.

## Supply inputs and replay NetStim draws

```{literalinclude} ../../examples/functional_network_while.py
:language: python
:start-after: "# region inputs"
:end-before: "# endregion inputs"
:dedent: 4
```

Per-Population `ve` and `intra` tapes use `NetworkRolloutInput`. Explicit NetStim
draw/RNG packets are indexed at each accepted timestep. If omitted, the host
scheduler samples packets from an independent copy of the supplied state's RNG
before entering a compiled span. Source renewal, schedules, threshold detection,
continuous resets, projections, event reception, and Population solves retain
native phase order. Capture does not sample extra random draws.

Constructor `extra={population_name: specification}` binds extracellular
contacts and Waveforms for while execution as well as scan. Their samples
follow the accepted simulation clock. An explicit `ve` tape can drive another
Population without bound `extra`; see {doc}`A28_functional_network_extra`.

The example supplies explicit draws for snapshot restarts and uses automatic
draws for fresh initialization. Fresh initialization runs once before each call
without state, using the source Network's cache-aware initialization policy
and retained RNG/seeder carry. It is
outside the while body. The initialization policy follows
{doc}`A21_functional_network_compile`; cache restoration and its effective
parameter/constant packets are described in
{doc}`A29_functional_network_cache_initialization`.

## Continue state and recordings

```{literalinclude} ../../examples/functional_network_while.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 4
```

Calls without state restart the captured snapshot or initialize fresh according
to the runner policy. Explicit state continues without initialization. Pass
`callback_state=aux["callbacks"].state` to continue a recording without
duplicating its boundary frame. Current parameters and aliases remain explicit;
effective parameter replacements from fresh initialization are returned in
`aux["parameters"]` and must be supplied when continuing if initialization
transforms wrote them.

```{literalinclude} ../../examples/functional_network_while.py
:language: python
:start-after: "# region checkpoint"
:end-before: "# endregion checkpoint"
:dedent: 4
```

Carry includes Population state, queues or histories, pending calendars, source
and schedule cursors, RNG bytes, and the authoritative Network clock. Native
checkpoint commit uses the same boundary and compatibility checks as scan.
Align the receiving Native Network's parameters before continuing with changed
explicit parameters.

Pending calendars retain their enqueue-time semantics and fixed capacity.
Overflow flags survive the loop and are checked on the host before returning
a result. Failure leaves supplied state/RNG, callback carry, restart snapshots,
and the prior execution report unchanged. Increase `pending_capacity` and retry
from a valid checkpoint; see {doc}`A24_functional_pending_calendar`.

## Inspect execution and bounded tails

```{literalinclude} ../../examples/functional_network_while.py
:language: python
:start-after: "# region report"
:end-before: "# endregion report"
:dedent: 4
```

`compiled_while_loop` reports a selected captured loop with at least one
four-step iteration. A horizon of one to three steps executes only the bounded
unrolled tail and reports `compiled_unrolled` with a reason. The requested
execution remains `"while"`. These reports describe dispatch; they do not promise
native code generation or performance for a particular compiler backend.

```{literalinclude} ../../examples/functional_network_while.py
:language: python
:start-after: "# region tail"
:end-before: "# endregion tail"
:dedent: 4
```

Lower-level `compile_rollout_chunk(..., execution="while")` accepts reusable
prepared values and explicit tapes, or participates in host `plan.run` scheduling.
Training-mode plans require `torch.no_grad()` or inference mode for captured
while execution. Functional transforms and direct forward AD use the existing
eager recurrence and report their fallback. A lower-level chunk with
`disable=True` reports `eager_disabled`; the explicit convenience while runner
rejects disabled compilation. Zero-step host runs make no kernel dispatch.

The complete CPU example executes both initialization policies:

```{code-cell} python
:tags: [remove-input]
from pathlib import Path
import sys

sys.path.insert(0, str(Path("../../examples").resolve()))
```

```{code-cell} python
:load: ../../examples/functional_network_while.py
:tags: [remove-input]
```
