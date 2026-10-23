---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-connection-parameters)=
# Train connection parameter sources

`dn.compile(Network)` retains the raw sources of event and continuous
connection weights and delays. A `PositiveParam` retains its positivity
transform, and a source using PyTorch parametrizations retains its original
tensors and trainable transform parameters. Native initialization and
functional preparation evaluate the same deterministic materializer.

The complete
{download}`functional_connection_parameters.py <../../examples/functional_connection_parameters.py>`
example shares a parametrized positive weight between an event connection and
a continuous connection, trains an event delay, and checks a continued
checkpointed rollout and its gradients against native execution. Run from the
repository root with `python examples/functional_connection_parameters.py`.

## Define sources in the ordinary Network

A source can be a supported Dendra parameter module such as `PositiveParam`,
or a deterministic module whose zero-argument `forward` returns the scalar or
vector values accepted by the connection. Register optimizer tensors as
`nn.Parameter` and fixed tensor dependencies with `register_buffer`. Sources
must preserve their output shape, dtype, and device and must not mutate inputs,
registered tensors, or external state. Implicit random draws and evolving
simulation state are currently unsupported in these deterministic sources.

This transform scales the raw `PositiveParam.rho` before its positivity
transform:

```{literalinclude} ../../examples/functional_connection_parameters.py
:language: python
:start-after: "# region transform"
:end-before: "# endregion transform"
```

Register it before initializing the Network, and pass the same weight module to
both connections:

```{literalinclude} ../../examples/functional_connection_parameters.py
:language: python
:start-after: "# region network"
:end-before: "# endregion network"
:dedent: 4
```

Functional lowering trusts the purity contract by default. Create the plan
inside `dn.ctx(CHECK_PURITY=True)` to enable optional diagnostics. Source
bindings and fixed tensor schemas remain required with either setting.
Change a source's Python configuration or registration structure by lowering
the Network again.

## Select raw parameters and inspect sharing

Connection parameter names follow native registered paths beneath
`synapses.<name>` or `continuous_synapses.<name>`:

| Source | Example suffix |
| --- | --- |
| Positive weight | `weight.weights.0.param.rho` |
| Positive event delay | `delay_ms.weights.0.param.rho` |
| Parametrized raw weight | `weight.weights.0.param.parametrizations.rho.original` |
| Trainable transform dependency | `weight.weights.0.param.parametrizations.rho.0.factor` |

The integer identifies a connection declaration block within the materialized
connection group. Source modules can have different paths; inspect
`runner.named_parameters` or `runner.find_parameters(...)` instead of assuming
every source contains `rho`. Registered fixed buffers appear in the
corresponding connection constants packet.

Exact names, globs, and regex selectors operate on these raw paths. Selecting
one alias preserves a shared optimizer leaf across every corresponding event,
continuous, or Population source binding. The two weight contributions in this
example accumulate into the same original and transform parameters:

```{literalinclude} ../../examples/functional_connection_parameters.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 4
```

`runner.trainable_parameters` includes alias names. Deduplicate its values by
tensor identity when constructing an optimizer from that mapping.

## Refresh and retain initialized values

`initialization="snapshot"` retains the initialized native value at the
lowering boundary, including a native value retained with
`reinit_weights=False`. Subsequent functional preparation remains
differentiable with respect to the explicit raw source tensors.

`initialization="fresh"` follows native order: Population initialization and
cache restoration, clock synchronization, connection source materialization,
then connection history restoration and NetStim initialization. A Population
initialization transform or cache that writes a shared raw source affects the
connection materialization at its position in that order.

`runner.initialization_inputs` contains `reinit_weights=True` and
`reinit_delays=True`. To retain an initialized value at a fresh boundary, supply
an input with the corresponding flag set to `False`:

```python
inputs = runner.initialization_inputs._replace(reinit_weights=False)
state, auxiliary = runner(initialization=inputs)
```

Retained values and their source baseline are explicit constants. Reuse both
`auxiliary["parameters"]` and `auxiliary["constants"]` when passing results to
another initialization boundary. Runtime continuation also carries both
packets, alongside state and callback carry:

```{literalinclude} ../../examples/functional_connection_parameters.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 4
```

The continued loss differentiates the original weight, the shared transform
parameter, and the positive delay source:

```{literalinclude} ../../examples/functional_connection_parameters.py
:language: python
:start-after: "# region gradients"
:end-before: "# endregion gradients"
:dedent: 4
```

## Supported delay and sampling policies

Event delays selected with `diff_delays=True` retain the existing native
interpolation and delay-gradient rules. Fixed event delays and continuous
delays retain integer routing frozen by lowering. Their raw sources remain
explicit, but a source update that changes the rounded routing timestep
requires lowering the Network again.

`Normal` and `TruncatedNormal` weight/delay distributions retain raw trainable
sources and support fresh sampling with explicit base draws and RNG carry.
See {doc}`A41_functional_connection_sampling` for resampling, shared sources,
and checkpoint continuation. Other initialized distribution implementations
can retain their fixed sampled storage with the corresponding reinitialization
flag set to `False`; fresh sampling through those implementations is currently
unsupported.

The lower-level `make_functional_netcon` projection API accepts effective
numerical weights and interpolated delays. `make_functional_continuous_con`
accepts effective weights and retains its frozen integer delay routes. These
APIs remain useful when an application manages source materialization itself.
NetStim source parameters retain their raw originals and registered transform
dependencies through the same Network preparation boundary. See
{doc}`A40_functional_netstim_parameters` for renewal-source selection and
initialization examples.

The complete example executes its native, compiled, and gradient checks:

```{code-cell} python
:load: ../../examples/functional_connection_parameters.py
:tags: [remove-input]
```
