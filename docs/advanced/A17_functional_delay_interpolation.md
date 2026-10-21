---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-delay-interpolation)=
# Differentiable connection delays

NetCon's functional adapter supports two- and three-tap dense delay
interpolation. Declare connectivity with the ordinary Network API, configure
interpolated training delays, then lower the initialized connection.
Physical delays become explicit differentiable inputs; the routing, timestep,
interpolation policy, and queue capacity remain fixed in the plan.

The complete
{download}`functional_delay_interpolation.py <../../examples/functional_delay_interpolation.py>`
example compares connected voltages, queues, weight/delay/source gradients, and
saved-state continuation with native execution. Run it with
`python examples/functional_delay_interpolation.py`.

## Configure an ordinary connection

Delays are in milliseconds. Use `diff_delays=True` with a dense training
runtime; `taps=2` selects linear interpolation and `taps=3` selects the native
Gaussian policy. `sigma` is static metadata measured in timesteps, not a
trainable parameter in this interface. Event tracking retains its native
restriction to hard delays, hard spikes, and hard scheduled times.

```{literalinclude} ../../examples/functional_delay_interpolation.py
:language: python
:start-after: "# region declarations"
:end-before: "# endregion declarations"
:dedent: 8
```

Configure the source before lowering. The interpolation path applies to the
active training runtime; evaluation retains integer-delay execution. Compact
training and inference backends are currently unsupported by this functional
interface.

## Supply explicit effective delays

```{literalinclude} ../../examples/functional_delay_interpolation.py
:language: python
:start-after: "# region lower"
:end-before: "# endregion lower"
:dedent: 4
```

`NetConTensors` has fields `(weights, state, schedule=None, delays=None)`.
`schedule` and `delays` are optional constructor arguments. Unpacking the
complete tuple returns all four fields.

| Field | Meaning |
| --- | --- |
| `weights` | Effective target-unit weights, one per materialized connection. |
| `delays` | Effective physical delay tensor in milliseconds for interpolation plans; `None` for fixed-delay plans. |
| `state` | Dense queue, ring/global-step counters, and detector carry. |
| `schedule` | Optional tensor schedule packet, independent of delay interpolation. |

Extracted parameters and schedule values are independent ordinary leaf
snapshots that preserve their gradient flags. Extracted runtime carry is
detached from any existing native autograd history. The effective delay values
are independent of the native parameter transform. An optimizer may update them
directly or supply values computed from its own parameterization. Gradients
flow to the explicit inputs rather than the original Network.

Interpolation plans require `delays=...` on every `advance` or `step` call.
The tensor must retain the extracted shape, dtype, and device. Fixed-delay
plans use calls without a delay argument and reject a
non-`None` delay input, since their integer routing is frozen.

## Preserve receive-before-observe ordering

Read the queued payload, receive it, observe the source, enqueue the current
weighted gates with the current delays, then integrate the population:

```{literalinclude} ../../examples/functional_delay_interpolation.py
:language: python
:start-after: "# region step"
:end-before: "# endregion step"
:dedent: 8
```

The example supplies explicit source gates. For thresholded or recurrent
sources, observe the current functional source after reception as described
in {doc}`A13_functional_threshold_detection`. Tensor schedules contribute
through the same packet and policy as {doc}`A15_functional_tensor_schedules`.

Population `prepare_and_receive` and `prepare_and_step` keep preparation inside
a composed compilation boundary. The example's `run_demo(compiled=True)` uses
full-graph `torch.compile` with `backend="aot_eager"`; the pure projection also
supports Inductor and `torch.func` transforms, including independently mapped
delay tensors and empty mapped lanes.

## Native interpolation and ring semantics

For each physical delay `d`, native interpolation forms
`position = clamp_min(d / dt, 1)` and `k = floor(position)`.

- Two taps write fractions `1 - (position - k)` and `position - k` at offsets
  `k` and `k + 1`.
- Three taps write at offsets `k - 1`, `k`, and `k + 1`, with weights given by
  `softmax(-0.5 * ((position - offsets) / (sigma + 1e-6))**2)`.

All destinations use the initialized ring's modulo addressing. Native and
functional execution use the same clamp/floor autograd conventions and
operation order. Delay gradients are piecewise; comparisons at delay boundaries should
use the native analytical policy rather than assume a smooth finite difference.

Three-tap offsets can include zero when `k == 1`. That tap is written after
the current row has already been read, so it arrives when the ring visits that
row again, after a full ring wrap. This behavior applies to native and
functional dense execution.

Queue capacity does not grow when explicit delay values change. Delays outside
the initialized horizon wrap modulo that capacity, as they do natively. Choose
a suitable native capacity before lowering and constrain learned values to the
intended range. Changing capacity requires initialization and lowering again.
Hot calls check tensor schemas without scanning delay values or adding host
synchronization.

## Pending deliveries and checkpoint continuation

Weights, gates, and delays are evaluated at enqueue time. Updating parameters
later affects new events; it neither moves nor rescales already queued values.
The queue retains the autograd history of the values used to create it.

Save connection and population carry, effective parameter/schedule values, and
the owner clock together:

```{literalinclude} ../../examples/functional_delay_interpolation.py
:language: python
:start-after: "# region checkpoint"
:end-before: "# endregion checkpoint"
:dedent: 12
```

The example clones carry without detaching it, preserving BPTT history through
the saved prefix. It resumes that saved state and checks each
remaining step and the complete gradients:

```{literalinclude} ../../examples/functional_delay_interpolation.py
:language: python
:start-after: "# region replay"
:end-before: "# endregion replay"
:dedent: 4
```

For durable checkpoints across optimizer updates, serialize or clone/detach
parameter and schedule values as well as carry. Detached carry begins a new
gradient history; live cloned carry can continue the existing one.

`commit_state_` transfers runtime queues, counters, and detector carry into a
compatible native connection. It does not apply weights or delay parameters.
Align those destination values separately, and synchronize population state
and the owning Network clock at a complete timestep boundary. Interpolation
plans accept compatible native delay values with different integer inference
quantization when routing, timestep, policy, and capacity remain unchanged.
The example deliberately changes the receiving connection before restoring:

```{literalinclude} ../../examples/functional_delay_interpolation.py
:language: python
:start-after: "# region commit"
:end-before: "# endregion commit"
:dedent: 4
```

This continuation uses the runtime `state_dict_for_checkpoint` protocol. The
`state_cache()` / `initialize_from_state_cache()` gate-history API is a
different operation: native cold rematerialization uses current
integer inference delays. It is not interpolation-aware runtime restoration.

The complete example executes here:

```{code-cell} python
:load: ../../examples/functional_delay_interpolation.py
:tags: [remove-input]
```

Continuous projections are described in
{doc}`A18_functional_continuous_connectivity`. Canonical single-device Network
composition is described in {doc}`A19_functional_network_lowering`. Compact
queue backends are currently unsupported by the functional adapter. The native
and functional adapters use the same dense interpolation policy.
