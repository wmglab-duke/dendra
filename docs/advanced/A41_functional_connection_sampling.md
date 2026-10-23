---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-connection-sampling)=
# Sample and train connection distributions

`dn.compile(Network)` supports `Normal` and `TruncatedNormal` sources for
NetCon and ContinuousCon weights and delays. Native and functional execution
share the distribution's pure draw transform and connection materializer.
Raw distribution parameters, registered PyTorch parametrizations, and shared
tensor identities remain explicit; sampled effective tensors retain pathwise
gradients through those sources.

The complete
{download}`functional_connection_sampling.py <../../examples/functional_connection_sampling.py>`
example shares a mean between event and continuous weights, samples event
delays, and checks compiled checkpointed continuation against native voltage,
raw-source gradients, and RNG. Run it from the repository root with
`python examples/functional_connection_sampling.py`.

## Define distributions and share raw sources

Import the distribution classes from `dendra.models.distributions`. Passing
the same `nn.Parameter` preserves one optimizer leaf across distributions;
passing the same Distribution module shares all of its registered sources.
Each connection block still receives independent draws in native order:

```{literalinclude} ../../examples/functional_connection_sampling.py
:language: python
:start-after: "# region sources"
:end-before: "# endregion sources"
:dedent: 8
```

Connection blocks can mix tensors, deterministic source modules, and the
supported distributions. A registered PyTorch parametrization can constrain
or transform a distribution's raw tensor and have trainable parameters of its
own. Source transforms follow the same deterministic, nonmutating contract
as {doc}`A39_functional_connection_parameters`. Optional diagnostics are
enabled by lowering inside `dn.ctx(CHECK_PURITY=True)`.

## Select trainable distribution parameters

Names follow registered source paths under `synapses.<name>` or
`continuous_synapses.<name>`:

| Source | Example suffix |
| --- | --- |
| Normal mean | `weight.weights.0.param.mean` |
| Normal scale | `weight.weights.0.param.std` |
| TruncatedNormal log scale | `delay_ms.weights.0.param._log_std` |
| Parametrized mean original | `weight.weights.0.param.parametrizations.mean.original` |

The integer is the declaration block index. `Normal` retains its native scale
representation; `TruncatedNormal` uses log scale. Fixed registered dependencies,
including truncation bounds, live in the corresponding constants packet.
Inspect `runner.named_parameters` or `runner.find_parameters(...)` for the
model's actual paths. Exact names, globs, and regex selectors all work:

```{literalinclude} ../../examples/functional_connection_sampling.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 4
```

Selecting one alias trains the common leaf used by every reader. Deduplicate
`runner.trainable_parameters.values()` by identity when constructing an
optimizer. Base draws are detached random inputs and are not optimizer leaves.

## Initialize, resample, and retain values

`initialization="snapshot"` uses the initialized native samples and their
retained draw provenance. `initialization="fresh"` applies Population
initialization/cache restoration, samples and initializes event connections,
then continuous connections, restores connection history, and initializes
NetStim. Each connection samples its weights before its delays; each role
samples its declaration blocks in order. Shared distributions consume a
separate draw at each sampling position.

`NetworkInitializationInput.reinit_weights` and `reinit_delays` default to
`True`. Setting a flag to `False` retains that role's previous materialization
and base draws, consumes no RNG for it, and retains its raw-source gradient
identity. Pass the resulting constants to subsequent fresh boundaries or
continued runs. Integer routing and delay capacity remain frozen by lowering.

The host runner samples automatically after Population/cache writes. For
explicit pure initialization, supply a `ConnectionSamplingDraw`:

```python
plan, packet = dn.func.make_functional_network(network)
draw = plan.sample_connection_initialization_draw(
    packet.parameters, packet.constants, packet.state,
    reinit_weights=True, reinit_delays=False,
)
inputs = plan.initialization_inputs._replace(
    connection_draw=draw, reinit_delays=False,
    netstim_draw=plan.sample_initialization_draw(
        packet.parameters, packet.state, constants=packet.constants,
    ),
)
initialized = plan.initialize(*packet, inputs)
```

The packet's `values` map collection names, connection names, `weights` or
`delays`, and stochastic block indices to base-draw tensors. Its `rng` maps
draw devices to generator-state byte vectors. Preserve the sampler's tensor
shapes, dtypes, devices, and detached-random-input semantics when supplying
your own draws. Pure initialization consumes the supplied values without
calling a random generator. A fresh stochastic role requires an explicit draw
in this pure API; the host runner provides the automatic sampling adapter.

## Carry RNG and continue checkpoints

Native distribution sampling uses Torch's global generators. Lowering
captures their current states for independent functional sampling; seed Torch
before constructing the functional plan or runner to choose that starting
point. The functional sampler and runner consume no global Torch RNG. Their
resulting carry is `state["connection_sampling"]["rng"]`, present on Networks
with supported stochastic connections. CPU and CUDA generator checkpoints
remain CPU byte vectors even when samples live on an accelerator.

Repeated calls without explicit state restart from the runner's captured
sampling state. Explicit continuation retains sampled constants, RNG, runtime
carry, and gradients:

```{literalinclude} ../../examples/functional_connection_sampling.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 4
```

A fresh-policy runner requires `aux["constants"]` when continuing stochastic
samples. Scan checkpoint replay and inference while execution carry the same
sampling state without resampling runtime connections. Native
`commit_state_` preserves explicit RNG in the Network runtime checkpoint's
optional `connection_sampling_rng` metadata without resetting global Torch
RNG. A subsequent native `initialize()` samples from ambient Torch RNG and
establishes a new native boundary.

Native model `state_dict()` retains sampled draw provenance alongside the
static weight/delay materializations. Runtime checkpoints continue to omit
those model parameters, so load or construct matching static model data before
restoring runtime state. Legacy model checkpoints without provenance retain
their exact effective sampled values. Their historical raw sampling Jacobian
cannot be reconstructed, so raw gradients through those sampled roles are
zero until explicit fresh sampling establishes provenance.

The example compares gradients of the continued voltage loss:

```{literalinclude} ../../examples/functional_connection_sampling.py
:language: python
:start-after: "# region gradients"
:end-before: "# endregion gradients"
:dedent: 4
```

## Current limits

Built-in Normal and TruncatedNormal sampling hooks are supported. Custom
distribution sampling implementations retain their initialized effective
storage; fresh functional sampling through them is currently unsupported.
Changing a fixed event/continuous delay's integer route requires lowering
again. Interpolated event delays can vary within the frozen delay capacity.
Changing source registration, Python configuration, or the cold sampling
dtype policy after lowering also requires a new compatible native boundary
and plan.

The complete example executes its native and compiled checks:

```{code-cell} python
:load: ../../examples/functional_connection_sampling.py
:tags: [remove-input]
```
