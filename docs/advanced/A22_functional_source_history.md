---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-source-history)=
# Source-history training

Functional Networks and NetCon projections support the native source-history
training runtime. It stores one gate per unique presynaptic source in a history
ring and reconstructs each delivery using the current weights and delays.
`netcon_train_backend="auto"` selects it when native source-level specialization
preserves the connection's semantics. Functional lowering retains that choice.

The complete
{download}`functional_source_history.py <../../examples/functional_source_history.py>`
example compiles an ordinary NetStim-driven Network, changes parameters between
two checkpointed calls, checks values and gradients against native execution,
and commits its state for native continuation. Run it from the repository root
with `python examples/functional_source_history.py`.

## Configure the ordinary Network

```{literalinclude} ../../examples/functional_source_history.py
:language: python
:start-after: "# region declarations"
:end-before: "# endregion declarations"
:dedent: 4
```

The training policy is `auto` by default. Use `source_history` for strict
admission, or `dense` to select weighted delivery queues. Source-history
training supports intrinsic NetStim activity, explicit source gate fields, and
source-level finite threshold detection. The native policy retains dense
fallback for scheduled traffic and configurations that require per-connection
detection or event introspection. Strict source-history requests reject
incompatible configurations.

`diff_spiking=True` stores floating gates and preserves surrogate gradients.
`diff_spiking=False` stores packed hard spikes; gradients still flow through
explicit weights and differentiable delays for realized events. Fixed delays
use the initialized integer offsets. With `diff_delays=True`, two- and
three-tap interpolation follows the native clamp, floor, coefficient, and modulo
policies described in {doc}`A17_functional_delay_interpolation`.

## Compile and continue with explicit parameters

```{literalinclude} ../../examples/functional_source_history.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 4
```

`runner.functional.synapses[name].runtime` reports `"source_history"` or
`"dense"`. Both snapshot and fresh initialization preserve the selected
layout. Fresh initialization clears history and detector carry in native
order; explicit state continues without reinitialization. NetStim draw/RNG
packets and initialization follow {doc}`A21_functional_network_compile`.

Source history retains past source activity, so parameter replacements affect
the delivery of that activity. The example supplies new weights and physical
delays for its second call:

```{literalinclude} ../../examples/functional_source_history.py
:language: python
:start-after: "# region changed-parameters"
:end-before: "# endregion changed-parameters"
:dedent: 4
```

These are out-of-place replacements. Keep tensors used by the prefix unchanged
until backward finishes. A delay edit changes which history rows contribute to
the current delivery; a weight edit rescales the reconstructed payload. Dense
queues retain weighted enqueue-time payloads, so they have a different policy
for edits made while traffic is pending. Choose the native backend policy
appropriate for the model.

The example compares both calls and the joined recording with native execution,
then checks gradients with respect to the original weight and delay leaves:

```{literalinclude} ../../examples/functional_source_history.py
:language: python
:start-after: "# region gradients"
:end-before: "# endregion gradients"
:dedent: 4
```

`checkpoint_every` replays exact rollout spans using their explicit parameter
packets. Initialization and host RNG sampling run outside those spans.
Continuation retains the prefix gradient history, including floating source
gates saved in the ring. Packed history has hard event semantics and carries no
source-gate gradient. Recording and callback boundary frames use the same
contracts as {doc}`A20_functional_network_rollout`.

## Compose the split projection phases

Lower a connection with `dn.func.make_functional_netcon(connection)`. For a
source-history plan, supply current parameters when reading the due payload:

```python
delivery_state, payload = projection.read_due(
    state, weights=weights, delays=delays,
)
population_state = population.prepare_and_receive(
    parameters, constants, population_state,
    target="syn", payload=payload,
)
source_values = observe_source(population_state)
state = projection.advance(
    weights, delivery_state, source_values, delays=delays,
)
```

Use `delays=None` for fixed-delay plans. `advance` records source activity and
advances the ring/global counters; it does not weight new events into a delivery
queue. Calling `read_due` before observing the source preserves native
receive-before-detect ordering. This matters when a receive changes the source
field of the same or a later recurrent connection. `projection.step` combines
the phases for source values already chosen by the caller.

Dense projections continue to support `read_due(state)`. The aggregate Network
adapter binds weights, delays, and source observations automatically.

## Save and commit compact carry

Functional source-history state is a Tensor-only tree. It contains counters,
the native delivery scratch, the active `source_gate_history` or
`spike_history_packed`, and source-level detector carry when present. Extraction
creates independent detached snapshots. Clone live carry to preserve its
gradient history across a saved prefix, or detach it for a new optimization
problem.

The adapter translates this tree to the native `backend_state` checkpoint
format during commit. Native checkpoint format and backward compatibility are
unchanged. The example aligns the receiving model's current parameters before
committing a complete Network boundary:

```{literalinclude} ../../examples/functional_source_history.py
:language: python
:start-after: "# region commit"
:end-before: "# endregion commit"
:dedent: 4
```

Routing, unique-source maps, detector configuration, timestep, runtime, and ring
capacity are frozen at lowering. Reinitialize and lower again after changing
them. Explicit weights and interpolated delays can vary between invocations.
As with native source history, delays address the fixed ring modulo its
capacity; constrain learned delays to the intended horizon.

Source-history training with nonempty legacy side calendars is currently
unsupported by functional lowering. Scheduled configurations use the native
dense fallback. Intrinsic packed-history inference is described in
{doc}`A23_functional_bitpacked_inference`; its hard source semantics differ from
surrogate training history. Functional sparse-calendar inference and packed
per-NetCon schedules are described in {doc}`A24_functional_pending_calendar`.

The complete CPU example executes both floating and packed training modes:

```{code-cell} python
:load: ../../examples/functional_source_history.py
:tags: [remove-input]
```
