---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-bitpacked-inference)=
# Packed-history inference

Functional Networks support native `bitpacked_history` inference for intrinsic
source traffic. The runtime stores one hard spike per unique presynaptic source
in an `int64` history ring. Deliveries read that history using the current
connection weights and the initialized integer delay offsets. Native and
functional execution use shared source detection, history, and projection
kernels.

The complete
{download}`functional_bitpacked_inference.py <../../examples/functional_bitpacked_inference.py>`
example compares compiled and native execution with noisy NetStim renewal,
source schedules, nonempty snapshot restarts, fresh initialization, weight
changes, recording, and native continuation. Run it from the repository root
with `python examples/functional_bitpacked_inference.py`.

## Configure and compile the ordinary Network

```{literalinclude} ../../examples/functional_bitpacked_inference.py
:language: python
:start-after: "# region declarations"
:end-before: "# endregion declarations"
:dedent: 4
```

Select `netcon_delay_backend="bitpacked_history"`, switch the Network to
evaluation mode, and initialize it before lowering. Native source-level
admission rules apply: finite thresholds must agree across the outgoing edges
of each source, explicit gate fields must have binary event semantics, and
per-connection event tracking is currently unsupported by this runtime. Mixed
finite/NaN thresholds and heterogeneous-device packed connections are currently
unsupported. Ineligible configurations raise; lowering does not change the
selected backend to a dense queue.

```{literalinclude} ../../examples/functional_bitpacked_inference.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 4
```

`runner.functional.synapses[name].runtime` reports `"bitpacked_history"`.
Use `inference=True` for a Network initialized in evaluation mode. Inference
execution suppresses gradients. For differentiable simulation, use native
training mode and the dense or source-history training policies described in
{doc}`A22_functional_source_history`.

The `aot_eager` backend makes the example portable on CPU; the compiler default
is Inductor. Snapshot calls restart the captured carry, including nonempty
packed history. Fresh calls clear connection history, detector latches, and
scratch in native initialization order. NetStim clocks, RNG, and consumed
calendar cursors follow {doc}`A21_functional_network_compile`.

## Preserve hard source semantics and ordering

NetStim sources provide their hard `spikes` mask. Their own tensor schedules
and consuming calendars are supported. Tensor schedule timing determines hard
spikes independently of amplitude, subject to the source's spike limit: negative,
zero, and fractional amplitudes do not rescale a packed source spike. Outgoing
weights scale the delivered event; the connection does not consume the floating
`spike_gate`. Explicit numeric gate fields fire when positive; Boolean fields
supply their Boolean values. Finite thresholds detect rising edges using one
latch per unique source. Surrogate source gradients are not part of packed
inference.

Each projection reads and delivers due history before observing its current
source. It then records hard source activity and advances its counters. This
preserves native ordering when reception changes the source of a recurrent or
later connection. Each event projection invokes its receiver even
when the payload is zero, following the same event-reception contract as
{doc}`A13_functional_threshold_detection`.

Delays are rounded, clamped to at least one timestep, and frozen at native
initialization. Changing physical delay parameters requires reinitialization
and lowering again. Differentiable delay interpolation is a training policy;
it is currently unsupported in packed inference. The fixed ring retains native
modulo behavior.

## Change weights and continue

Saved packed history is unweighted. Current weights rescale deliveries from
already recorded source spikes. The example replaces the explicit parameter
packet for its second call:

```{literalinclude} ../../examples/functional_bitpacked_inference.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 4
```

Dense queues retain weighted enqueue-time values and therefore have different
semantics for weight edits while traffic is pending. Choose the native policy
that fits the model. Recording and callback boundary frames follow
{doc}`A20_functional_network_rollout`; resumed collections emit only new frames.

## Use explicit projections and native checkpoints

`dn.func.make_functional_netcon(connection)` also supports this runtime. Supply
current `weights` to `projection.read_due(state, weights=weights)`, deliver the
returned payload, observe the source, then call
`projection.advance(weights, delivery_state, source_values)`. Fixed-delay
plans use `delays=None`. `projection.step` combines the phases when the caller
has already chosen source values. Lower-level pure projections retain
gradients with respect to explicit floating weights; the Network inference
runner suppresses simulation gradients as configured.

The returned read state contains the current payload in `delivery_buffer`,
matching native packed inference's scratch row. Source-history training leaves
its scratch row inert. Both use explicit history and detector carry;
the adapter translates these Tensor-only packets to native `backend_state`
checkpoints without changing native serialization. Extraction creates detached
independent snapshots, and native commit clones incoming carry.

```{literalinclude} ../../examples/functional_bitpacked_inference.py
:language: python
:start-after: "# region commit"
:end-before: "# endregion commit"
:dedent: 4
```

Align the receiving model's parameters before committing state. Keep matching
topology, timestep, source configuration, and ring capacity with a checkpoint.

Per-NetCon scheduled events and nonempty legacy side calendars are currently
unsupported by functional packed inference. NetStim's own schedules are
supported because they produce source spikes before connection delivery.
Functional `sparse_calendar` inference is currently unsupported.

The complete CPU example executes both initialization policies:

```{code-cell} python
:load: ../../examples/functional_bitpacked_inference.py
:tags: [remove-input]
```
