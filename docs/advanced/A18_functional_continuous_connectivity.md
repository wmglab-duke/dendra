---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-continuous-connectivity)=
# Shared continuous connectivity

Continuous connections sample a source variable each timestep. Declare them
with the standard `Network.connect_continuous` API, then lower the initialized
`ContinuousCon` and its target Population. Native and functional execution use
the same pure transport, receiver operations, and user-authored source transform.

The complete
{download}`functional_continuous_connectivity.py <../../examples/functional_continuous_connectivity.py>`
example checks connected voltages, previous inputs, queues, gradients, and
checkpoint continuation against an ordinary Network. Run it with
`python examples/functional_continuous_connectivity.py`.

## Declare and lower ordinary connections

The example supplies dimensionless source gates. Its `graded_syn` target
receives conductance density, so weights and `g_pre` are in S/cm². Delays are
in milliseconds and retain native integer rounding. A delay rounded to zero
delivers immediately, before the population solve.

```{literalinclude} ../../examples/functional_continuous_connectivity.py
:language: python
:start-after: "# region declarations"
:end-before: "# endregion declarations"
:dedent: 8
```

```{literalinclude} ../../examples/functional_continuous_connectivity.py
:language: python
:start-after: "# region lower"
:end-before: "# endregion lower"
:dedent: 4
```

`ContinuousConTensors` separates effective `weights`, runtime `state`, and
`transform_parameters`. For a module transform, that mapping contains all named
parameters and read-only buffers, including nested names. It is `None` for
functions and modules without registered tensors. The state contains the value queue, boolean presence
queue, ring cursor, and absolute step counter. Routing, integer delays,
timestep, input name, and reduction mode are frozen metadata. Reinitialize
and lower again after changing the connection configuration.

Extracted tensors have independent storage. Effective parameters retain their
gradient flags; extracted carry is detached from existing native history.
Supply explicit tensors computed from your own parameterization when needed.
Weights affect new samples at enqueue time; changing them does not rescale
already pending deliveries.

## Reset once, then preserve connection order

`population.reset_continuous_inputs(state)` saves each input as its declared
`*_old` carry, zeros current inputs, and returns `(state, scratch)`. Scratch
contains explicit per-input first-delivery masks. Pass its returned value
between receives within this timestep; reset once before all connections.

`population.continuous_receive(...)` returns `(state, scratch)` without
advancing time. Its preparation-combining counterpart is
`prepare_and_continuous_receive`. Inputs use target-local order and include
explicit population batch axes. A projection returns both its weighted
payload and its boolean delivery mask; pass both to the receiver.
Use `input=projection.input` and `reduce=projection.reduce` to retain the
connection's declared target input and reduction policy.

The native ordering depends on the initialized delay partition:

| Connection | Ordered operations |
| --- | --- |
| Empty | Advance counters without reading a source or calling a receiver. |
| All immediate | Read source, receive projected values. |
| All delayed | Read due values, receive them, read source, enqueue new values. |
| Mixed immediate/delayed | Read source, receive due plus immediate values in one call, enqueue delayed values. |

`projection.receive_before_source` identifies the all-delayed case. Call
`read_due` and receive its payload before observing a recurrent source in that
case. `advance` samples the supplied explicit source values and returns the
new queue state, payload, and mask. Its all-delayed payload is the same due
delivery: receive it once, using the earlier phase. For mixed and immediate
connections, receive the payload returned by `advance`.
Skip reception when `projection.has_connections` is false; a custom hook may
have effects even when its numeric payload is zero.

```{literalinclude} ../../examples/functional_continuous_connectivity.py
:language: python
:start-after: "# region step"
:end-before: "# endregion step"
:dedent: 8
```

The example supplies independent source gates. For recurrent connectivity,
read source values from the current functional state at the indicated phase.
Across a complete Network timestep, reset continuous targets once, visit
continuous projections in native order, visit event projections, and then
advance populations.

## Reduction and receiver hooks

Duplicate edges sum within one projection before delivery. Between
projections, receivers support `sum`/`add`, `set`/`replace`/`last`, `min`, and
`max`. A true zero delivery participates in every reduction. An untargeted
scatter slot does not participate. First-delivery masks ensure that resetting
an input to zero does not bias a positive minimum or a negative maximum.

Most `ContinuousSynapse` subclasses only declare `INPUT` and implement their
current. Custom receive rules implement the pure hook:

```python
def continuous_receive_values(self, value, dt, values, *,
                              input, reduce, mask, received):
    # This illustrative custom rule always adds masked input.
    update = torch.where(mask, value, torch.zeros_like(value))
    return {"carry": {input: values[input] + update}}
```

The hook receives explicit runtime values and masks and returns partial
`states`, `carry`, or nested `state_carry` replacements, following the event
receive contract. `dt` has shape `(1,)`. Arguments and evolving module state
are read-only; read parameters through `self`. The canonical imperative
`continuous_receive` adapter applies the same result. Legacy mutating receive
or reset overrides remain available to imperative Networks; functional
reception requires the canonical adapters and a pure supported hook.

## Source transforms and gradients

Supply a pure function or a memoryless `torch.nn.Module` using its ordinary
`forward(values)` contract. The transform receives the selected, connection-ordered
source vector, before multiplication by weights. Its output must broadcast with
those weights to one value per connection, using the same dtype and device.
There is no list of permitted transform classes. The example defines both:

```{literalinclude} ../../examples/functional_continuous_connectivity.py
:language: python
:start-after: "# region transforms"
:end-before: "# endregion transforms"
```

Stateless means no hidden evolving simulation history; parameters and read-only
buffers are supported. Functional execution uses the same authored `forward`
through `torch.func.functional_call`, with every registered parameter and buffer
supplied explicitly in `transform_parameters`. Pass this mapping on every
`advance` call, including buffer values. An optimizer can update its parameter
leaves or supply values computed from another parameterization. Extracted tensor
leaves preserve their original gradient flags. Tied registered tensors retain
their alias contract.

Ordinary Python configuration, such as a float threshold, is frozen metadata.
Register a `torch.nn.Parameter` when a value should be trainable, and use
`register_buffer` for a tensor coefficient that stays read-only during a call.
Unregistered tensor captures in functions, closures, or module attributes must
be made explicit through a module. Pure functions may use immutable constants
and helpers; their dependencies are checked as part of lowering.

`sigmoid_release` follows this same generic module path, using its existing
`forward`. Its default `theta` and `sigma` float attributes remain static
configuration. It has no privileged lowering and needs no extra hook.

Lowering qualifies a disposable transform copy and checks source-code
dependencies, arguments, registered tensors, Python state, repeated outputs, and
global random-generator state. In-place input changes, parameter/buffer writes,
hidden counters, mutable closure/global dependencies, implicit randomness, and
global RNG seed/state reads are rejected. Replaced call dispatch and hooks are
also unsupported. Modules, including nested children, must retain canonical
metadata traversal and copying/serialization behavior. Qualification can therefore
reject an otherwise memoryless module; express that operation in a pure function
or a module with canonical hooks instead. For example, a pure function calling
`torch.softmax(values, dim=-1)` avoids `nn.Softmax`'s custom serialization hook.
Qualification does not advance or mutate the source Network; RNG state is restored after
probes. Changing static configuration or callable bindings invalidates the plan;
lower again to adopt those changes. Explicit parameter and buffer values can
change without rebuilding the plan when their schemas stay the same.

Gradients flow through explicit source values, weights, transform parameters,
receiver updates, and already queued values. Integer delays are static in
this interface. The example's `run_demo(compiled=True)` compiles a complete
composed step with `backend="aot_eager", fullgraph=True`; pure components
also support `torch.func` transforms. The example compares gradients for the
custom module's scale and bias as well as weights and source values with native
execution, including pending queues and saved-prefix continuation.

On the tested PyTorch 2.14.1 build, an empty `vmap` over a zero-dimensional
parameter can raise `IndexError` even for plain `torch.ones(4) * parameter`.
This is a PyTorch batching limitation. Use a broadcastable parameter shape
`(1,)` when empty parameter batches are needed, or avoid empty mapped batches.
The adapter preserves authored tensor shapes and arithmetic.

## Save and resume at a timestep boundary

Save population and connection carry together with their effective parameters
and owner clock. Clone live carry to retain gradients through a saved prefix:

```{literalinclude} ../../examples/functional_continuous_connectivity.py
:language: python
:start-after: "# region checkpoint"
:end-before: "# endregion checkpoint"
:dedent: 12
```

```{literalinclude} ../../examples/functional_continuous_connectivity.py
:language: python
:start-after: "# region replay"
:end-before: "# endregion replay"
:dedent: 4
```

`commit_state_` transfers runtime queue/mask/counter state into a compatible
native connection. Population commit transfers its carry, including current
and previous continuous inputs. Synchronize weights, transform settings, and
the owning Network clock separately. The example aligns that clock first:

```{literalinclude} ../../examples/functional_continuous_connectivity.py
:language: python
:start-after: "# region commit"
:end-before: "# endregion commit"
:dedent: 4
```

Receiver scratch is per-timestep working state, separate from native
checkpoint data. Resume at a complete timestep boundary and create fresh
scratch at the next reset. The runtime checkpoint protocol preserves pending
payloads and their gradient history. Native `state_cache()` is a separate,
detached cold-start protocol that rematerializes recorded source history
using current parameters; its existing behavior is retained.

The complete example executes here:

```{code-cell} python
:load: ../../examples/functional_continuous_connectivity.py
:tags: [remove-input]
```

Canonical single-device `Network` lowering is described in
{doc}`A19_functional_network_lowering`. Differentiable continuous delay
interpolation remains a separate extension.
