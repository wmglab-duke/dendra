---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-torch-parametrizations)=
# Use PyTorch parametrizations

Use `register_parametrization(...)` to compute a Population tensor, or a tensor
on one of its supported registered submodules, from original tensors and a
module. The same deterministic transform works in native execution,
`dn.func.make_functional`, and `dn.compile(Population)` or
`dn.compile(Network)`. Functional preparation uses Dendra's shared parameter
materializer and retains the registered transform order.

The complete
{download}`functional_torch_parametrizations.py <../../examples/functional_torch_parametrizations.py>`
example shares a trainable capacitance transform between two Populations,
continues a checkpointed Network rollout, and checks its voltage recordings
and gradient against native execution. Run from the repository root with
`python examples/functional_torch_parametrizations.py`.

## Define and register a transform

This module scales its original capacitance tensor and adds a fixed offset:

```{literalinclude} ../../examples/functional_torch_parametrizations.py
:language: python
:start-after: "# region transform"
:end-before: "# endregion transform"
```

Register it on the effective `cm` buffer before initialization:

```{literalinclude} ../../examples/functional_torch_parametrizations.py
:language: python
:start-after: "# region network"
:end-before: "# endregion network"
```

`Population.register_parametrization(name, module, unsafe=True)` delegates to
`torch.nn.utils.parametrize.register_parametrization`. The PyTorch function
also accepts supported registered submodules within a lowered Population as
its owner, including Mechanisms, States, parameter-source modules, and
Waveforms. The target must
be a registered tensor: a Dendra source such as `cm_param` can itself be a
module, rather than a tensor, and cannot be passed as a PyTorch tensor target.
Choose the registered effective tensor, or a registered raw `Parameter`
inside its source module, according to the parameterization you intend.
After parametrizing `cm`, its value comes from
`parametrizations.cm.original` and the registered transforms. The separate
`cm_param` source does not feed that computed property. Parametrize a raw
source tensor instead when that source should remain the input.

The module's `forward` returns the effective tensor. Register trainable
dependencies as `nn.Parameter` and fixed tensor dependencies with
`register_buffer`. Register child modules normally. Do not mutate original
tensors, transform parameters/buffers, the owner, or external state. Do not
draw implicit random values or read evolving simulation state. Computation
must retain its fixed shape, dtype, and device schema across preparation
calls. A transform that updates running statistics in training mode does not
satisfy this contract.

Functional lowering trusts this purity contract by default. Enable optional
diagnostics while creating a plan:

```python
with dn.ctx(CHECK_PURITY=True):
    runner = dn.compile(network, dt=DT, steps=STEPS, inference=False)
```

The plan retains that setting after the context exits. Tensor bindings,
alias relationships, target schemas, and supported lifecycle checks remain
required in both modes. Diagnostics help detect hidden dependencies and
mutation; they do not prove purity for every input.

## Inspect originals, constants, and shared parameters

PyTorch stores each transform list under `owner.parametrizations.<target>`.
An ordinary one-input transform has an `original` tensor. A `right_inverse`
that decomposes its input into several tensors produces `original0`,
`original1`, and so on. Those originals remain explicit functional inputs;
transform modules retain their normal numbered paths, such as
`parametrizations.cm.0.log_scale`.

An original `Parameter` and the transform's trainable parameters appear in
the functional parameter packet. An original buffer and the transform's
registered fixed buffers appear in the constants packet. Parametrizing a
buffer does not turn that original into a trainable `Parameter`. In this
example `cm` starts as a buffer, so its original is a constant; `log_scale`
is the trainable parameter. PyTorch's `right_inverse` establishes the
original tensors at registration or explicit native assignment. Functional
preparation evaluates `forward` from the supplied originals.

Network query names include the Population name. Glob and regular-expression
selectors can select transform parameters across Populations:

```{literalinclude} ../../examples/functional_torch_parametrizations.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 4
```

The two transforms reference the same `nn.Parameter`. Their qualified names
resolve to one independent runner-owned optimizer leaf, and both Population
contributions accumulate into its gradient. `runner.trainable_parameters`
shows the selected names, including aliases; deduplicate by tensor identity
when constructing an optimizer from that mapping. `runner.get_parameter(...)`
resolves one exact, partial, glob, or regular-expression query. The runner's
parameters remain independent of the source Network.

Connection weight/delay sources and NetStim parameter sources do not retain
their raw transformations in functional packets. Connection weights, event
delays selected for differentiation, and NetStim's `interval`, `start`,
`noise`, and `tau` are exposed as effective numerical values. Trainable
selectors refer to those effective fields; parameters inside their source
modules are currently unavailable as optimizer leaves. Fixed connection
delays and NetStim's integer `max_spikes` remain configuration, rather than
trainable parameters. Registering a PyTorch parametrization directly on
NetStim is currently unsupported. Continuous connections retain their separately
registered stateless transform parameters, and tensor schedules retain their
explicit times and amplitudes. See
{doc}`A18_functional_continuous_connectivity` and
{doc}`A15_functional_tensor_schedules` for those inputs.

## Continue a differentiable rollout

Prepared effective tensors are invocation-local. Updating an original or a
transform parameter and calling the runner again rebuilds the effective
values and numerical workspace without retaining a previous autograd graph.
Supply the effective packets when continuing a fresh-initialized run:

```{literalinclude} ../../examples/functional_torch_parametrizations.py
:language: python
:start-after: "# region continue"
:end-before: "# endregion continue"
:dedent: 4
```

Checkpoint recomputation uses the same explicit parameter and constant
inputs. It does not assign new originals or call `right_inverse` during each
simulation step. Both continued state and recordings retain their gradients:

```{literalinclude} ../../examples/functional_torch_parametrizations.py
:language: python
:start-after: "# region native"
:end-before: "# endregion native"
:dedent: 4
```

`initialization="snapshot"` restarts the captured runtime state while
preparing the current parameter values. `initialization="fresh"` evaluates
initialization from those current values. A bound Population or Network
steady-state cache restores its saved original tensors and persistent
transform buffers according to native state-dict ordering. Cached numerical
values are detached from the warmup, and restored raw Parameters retain their
incoming gradient identity. An ordinary saved effective buffer remains
detached. A PyTorch-owned effective tensor such as this example's `cm` is a
computed property rather than a saved effective buffer: it is recomputed from
the restored originals and transform parameters, retaining their native
gradients. Current runner values do not override restored forward values.
Concatenated models also restore their saved assembled physical fields;
parametrizing a component does not make a saved parent field a computed
property. Quantities the solver reads from component properties retain
their native gradients.
Nonpersistent transform buffers retain their current values unless a
persistent alias writes the same storage. See
{doc}`A29_functional_network_cache_initialization` for cache and continuation
packets.

Transform lists, module bindings, original layouts, and aliases are part of
the lowered model structure. Adding, removing, or reordering parametrizations
requires a new plan. Registered execution hooks, hidden mutable dependencies,
implicit randomness, and stateful transform forwards are currently
unsupported. Functional lowering supports the same canonical Population
topologies as {doc}`A8_functional_populations`; a PyTorch-generated
parametrized wrapper retains its underlying topology contract.
Parametrize individual component Populations before concatenating them.
Parametrizations directly on a `MultiPopulation` packed parent are currently
unsupported: native packing assigns assembled fields through the property's
`right_inverse`, while functional preparation evaluates its pure forward
computation. Component parametrizations retain their native bindings and
gradients.
`Myelinated` retains its canonical node-diameter and axial-resistivity
transforms, including framework-owned passive-end overrides. Authored
replacement or extension of that geometry stack is currently unsupported;
other supported tensor targets can have their own parametrizations.
Targets that are evolving simulation state are currently unsupported. A
PyTorch-owned effective target also cannot have regional parameter overrides
or an in-graph transform writing that same target; use one coherent
parameterization path for it.

The native and compiled checks in the complete example can be run directly:

```{code-cell} python
:load: ../../examples/functional_torch_parametrizations.py
:tags: [remove-input]
```
