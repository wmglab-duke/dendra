---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-netstim-parameters)=
# Train NetStim parameter sources

`dn.compile(Network)` retains NetStim's raw parameter sources. The positive
interval uses its native constraint transform, and PyTorch parametrizations
retain their original tensors, trainable transform parameters, and registered
configuration buffers. Native execution and functional preparation evaluate
the same pure materializer.

The complete
{download}`functional_netstim_parameters.py <../../examples/functional_netstim_parameters.py>`
example trains a parametrized interval and noise, shares the interval source
with Population capacitance and a connection weight, and checks checkpointed
continuation against native voltage, gradients, and clock/RNG state. Run it
from the repository root with
`python examples/functional_netstim_parameters.py`.

## Define and share sources

`dn.NetStim(...)` creates a positive interval parameter. Register a PyTorch
parametrization on `stimulus.interval.rho` to constrain or transform its raw
value. A transform registers optimizer tensors as `nn.Parameter` and fixed
tensor dependencies with `register_buffer`:

```{literalinclude} ../../examples/functional_netstim_parameters.py
:language: python
:start-after: "# region transform"
:end-before: "# endregion transform"
```

The same source can be used by a Population or connection. Passing the same
module preserves the shared original tensor and its transform dependencies.
The example deliberately shares one source across interval, capacitance, and
weight to demonstrate alias handling:

```{literalinclude} ../../examples/functional_netstim_parameters.py
:language: python
:start-after: "# region network"
:end-before: "# endregion network"
:dedent: 4
```

Source evaluation must be deterministic and must not mutate inputs, registered
tensors, or external state. Register tensor dependencies, preserve their
shape, dtype, and device, and lower the Network again after changing Python
configuration or registration structure. Functional lowering trusts this
purity contract by default; create the plan inside
`dn.ctx(CHECK_PURITY=True)` to enable optional diagnostics. The NetStim renewal
process uses separate explicit random draws rather than source-local sampling.

## Select raw parameters

NetStim source names appear beneath `netstim.parameters`:

| Source | Example path |
| --- | --- |
| Positive interval | `netstim.parameters.interval.rho` |
| Parametrized interval original | `netstim.parameters.interval.parametrizations.rho.original` |
| Trainable interval transform | `netstim.parameters.interval.parametrizations.rho.0.factor` |
| Noise | `netstim.parameters.noise` |
| Parametrized noise original | `netstim.parameters.parametrizations.noise.original` |

`start` and `tau` also retain their native source names. These primary fields
remain explicit parameter inputs when their native source is a buffer or
scalar; `trainable` controls which runner leaves require gradients.
Parametrized sources
can have different registered paths; inspect `runner.named_parameters` or
`runner.find_parameters(...)` instead of assuming every source contains
`rho`. Registered fixed transform buffers appear in
`constants["netstim"]["sources"]`. Integer `max_spikes` and calendar metadata
remain configuration.

Exact names, globs, and regex selectors operate on these raw paths. Selecting
one alias selects the shared optimizer leaf for all matching Population,
connection, and NetStim readers:

```{literalinclude} ../../examples/functional_netstim_parameters.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 4
```

If a primary NetStim input shares a tensor with a registered configuration
buffer, the explicit parameter also supplies that buffer's value. Updating the
parameter therefore updates every shared reader; the corresponding constants
entry does not provide an independent value.

`runner.trainable_parameters` includes alias names. Deduplicate its values by
tensor identity when constructing an optimizer from that mapping. Selecting
`interval.rho` trains the unconstrained original through the native positive
transform rather than replacing the physical interval with a free tensor.

## Initialization, caches, and random state

`initialization="snapshot"` starts from the captured clocks, calendars, and RNG
state. `initialization="fresh"` completes Population initialization and cache
restoration, connection initialization and history restoration, then NetStim
initialization. A Population initializer or cache that writes a shared raw
parameter or registered transform buffer affects the NetStim values used at
that boundary.

The host runner samples the initialization draw after those writes and after
materializing the effective noise. Deterministic initialization consumes no
random draw. A fixed NetStim seed resets its initialization RNG; `seed=None`
continues the supplied RNG state. The initial stochastic clock kick is
detached, matching native execution. Subsequent renewal intervals and
surrogate gates differentiate through the raw interval, noise, and gate
parameters.

For an explicit initialization draw, pass the corresponding source constants
when sampling, particularly when a transform reads configuration buffers:

```python
plan, packet = dn.func.make_functional_network(network)
draw = plan.sample_initialization_draw(
    packet.parameters, packet.state, constants=packet.constants
)
inputs = plan.initialization_inputs._replace(netstim_draw=draw)
initialized = plan.initialize(*packet, inputs)
```

Use the runner's fresh initialization when the initialization phases can
change effective noise and the draw should be selected conditionally from
those resulting values. Explicit draw and RNG tapes remain available for pure
transformed execution.

## Continue without restarting

Pass the returned parameter and constants packets alongside runtime state and
callback carry. This retains source replacements made by initialization,
updated cache values, gradients, random state, and calendar cursors:

```{literalinclude} ../../examples/functional_netstim_parameters.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 4
```

The continued loss differentiates the shared original, its trainable transform,
and noise:

```{literalinclude} ../../examples/functional_netstim_parameters.py
:language: python
:start-after: "# region gradients"
:end-before: "# endregion gradients"
:dedent: 4
```

The lower-level `make_functional_netstim` API accepts effective numerical
`interval`, `start`, `noise`, and `tau` tensors. It remains useful when an
application manages source materialization itself. Source-local stochastic
sampling and changes to the frozen generator shape are currently unsupported
by Network source materialization. A parametrized module used as an Ion or
Material initialization recipe is also currently unsupported by fresh
initialization. Share a registered raw tensor with an ordinary initializer,
or use a supported runtime source such as `cm_param`, when those values should
participate in the same parameter binding.

The complete example executes its native, compiled, and gradient checks:

```{code-cell} python
:load: ../../examples/functional_netstim_parameters.py
:tags: [remove-input]
```
