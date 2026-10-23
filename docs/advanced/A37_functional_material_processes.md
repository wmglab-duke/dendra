---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-custom-material-processes)=
# Author a custom MaterialProcess

A `MaterialProcess` updates complete shared Material fields. Use one for
population-wide reactions, transfer between stores, or spatial transport.
Implement its pure field hooks once and use the same class with native
Population/Network execution and `dn.compile`.

The complete
{download}`functional_material_processes.py <../../examples/functional_material_processes.py>`
example transfers a stored amount between two Materials, trains the transfer
rates in a two-Population Network, and checks native trajectories and
conservation after checkpointed scan continuation. Run it from the repository
root with `python examples/functional_material_processes.py`.

## Define the field computation

This process moves the fraction `1 - exp(-dt * rate)` of the pool into the
reservoir on each accepted step. It preserves their sum at each compartment:

```{literalinclude} ../../examples/functional_material_processes.py
:language: python
:start-after: "# region process"
:end-before: "# endregion process"
```

The three pure hooks have separate roles:

- `material_fields` returns a nonempty tuple of canonical `(material_name,
  field_name)` pairs. These identify the complete input fields; the tuple stays
  fixed for the lowered model. Generic Material names use their declared name,
  such as `"pool"`, rather than a recording alias such as `"pool_material"`.
- `prepare_materials(fields, dt)` returns a tensor-only PyTree of numerical
  workspace, or an empty tuple when no workspace is needed. `None` is reserved
  by the native adapter. Preparation uses current explicit parameters, geometry,
  timestep, and field schemas. Deterministic preparation runs outside the
  functional accepted-step loop; custom processes with NOISE prepare at their
  Material phase using the current explicit noise. It must not read evolving field
  values. In the example, `pool`
  supplies the shape, dtype, and device needed to resolve the rate;
  `fraction` is independent of the stored amount.
- `material_updates(fields, prepared)` reads the current complete fields and
  returns a dictionary of full-field replacements. Its fixed output keys are a
  subset of `material_fields`, and each tensor matches that input field's
  shape, dtype, and device. State-dependent reactions belong here. Omitted
  fields retain their values.

Neither numerical hook mutates its inputs, the process, or external state.
Register trainable values with `GLOBAL`, `RANGE`, or `BATCH`; fixed tensor
dependencies must be registered buffers. Keep parameters and buffers explicit
rather than capturing hidden tensors in a function closure or module global.
Preparation remains differentiable through its numerical parameters and
geometry. Its workspace has a fixed tensor schema and is rebuilt when the
functional model is prepared from a new parameter or constant packet.
`material_updates` receives each prepared tensor with the shape returned by
preparation.

Functional lowering trusts the authored purity contract by default.
`with dn.ctx(CHECK_PURITY=True):` around `dn.compile(...)` or
`dn.func.make_functional(...)` enables diagnostic dependency, mutation, RNG,
and repeated-evaluation checks for the new plan. The plan retains that mode
after the context exits. Required field/workspace schemas, tensor layouts,
and lifecycle checks remain active in either mode. Diagnostics can detect
violations; they do not prove purity for every possible input.

Use the shared `_resolve_quantity(name, like)` helper to expand named process
operands into the complete field frame. This handles `RANGE`/`BATCH` values
authored on a regional insertion or a component of a concatenated
MultiPopulation. Return `_mask_material_updates(fields, updates)` to retain
unchanged values outside the process insertion and material-field support.
The hooks receive full fields even for a regional insertion; they do not
receive an automatically gathered local field slice.

The process inherits `configure_process`, `set_dt`, and `advance_materials`
from `MaterialProcess`. The shared native `advance_materials` adapter calls
these pure numerical hooks and commits their replacements. Custom imperative
lifecycle overrides are currently unsupported by functional lowering. Ordinary
Mechanism `STATE`, `CARRY`, `initial_values`, `advance`, and workspace phases
do not apply to MaterialProcesses. Process-level `RAND` and `NOISE` use
explicit draws and RNG carry; see {doc}`A44_functional_material_stochasticity`.
Named NOISE operands resolved with `_resolve_quantity` are copied for the
accepted step. Direct reads such as `self.eta` need an explicit `.clone()` when
backward may save that sample, because native sampling refreshes it in place.
Implicit random draws, execution hooks, hidden dependencies, and persistent
mutation are currently unsupported.

## Insert the same process in native Populations

Declare the fields and insert the process before initialization:

```{literalinclude} ../../examples/functional_material_processes.py
:language: python
:start-after: "# region population"
:end-before: "# endregion population"
```

Material fields remain shared runtime state, independent of process workspace.
Each accepted step commits local Mechanism replacement writes and adds their
Material sources once. MaterialProcesses then run in `post_local`, `transport`,
and `post_transport` order, preserving insertion order within each phase. Each
process sees preceding replacements. Shared-field guards and enabled derived
updates, including Nernst reversals, run afterward, followed by local-field
refresh. Choose `PHASE(...)` to place a process in this order.

Fresh initialization derives initial fields and applies its normal guards and
shared-field transaction. It does not advance Material sources or processes at
t=0, so the first recorded pool amount in this example is `2.0`. Generic
Material sources, Ion styles, and parameter sharing retain their normal
initialization contracts; see {doc}`A36_functional_initialization_sources`.

## Compile, record, and differentiate

`dn.compile(Network)` lowers the same authored process and field bindings:

```{literalinclude} ../../examples/functional_material_processes.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 4
```

The wildcard selects both Populations' rate parameters. Inspect
`runner.trainable_parameters` or use `runner.get_parameter(...)` to verify the
selected leaves. Process parameters use the same shared-alias rules as other
registered Network parameters. Record complete fields with names such as
`"left.pool_material.amount"`; the explicit state contains the same value at
`state["populations"]["left"]["materials"]["pool"]["amount"]`.

Explicit continuation consumes the supplied fields and carries. Pass the
effective parameter/constant packets and callback state with that state:

```{literalinclude} ../../examples/functional_material_processes.py
:language: python
:start-after: "# region continue"
:end-before: "# endregion continue"
:dedent: 4
```

Continuation does not initialize the fields or apply an extra process step.
The first segment records its initial sample; the continued segment appends
accepted samples without duplicating that boundary. Gradients remain attached
through both segments, including the preparation and checkpointed scan:

```{literalinclude} ../../examples/functional_material_processes.py
:language: python
:start-after: "# region gradients"
:end-before: "# endregion gradients"
:dedent: 4
```

Snapshot starts and runtime checkpoints preserve the saved Material fields.
Steady-state cache restoration retains its native field and parameter policy;
it does not execute a process while restoring t=0. Runtime preparation derives
workspace from the effective restored inputs. See
{doc}`A29_functional_network_cache_initialization` for Population-owned and
Network-global cache inputs. Scan supports training and checkpoint replay;
while execution uses the same field updates for inference.

Fresh cache restoration currently does not support a custom process
configuration tensor that overlaps only part of a cached runtime tensor's
storage. Use independent registered configuration buffers or an exact shared
tensor binding.

```{code-cell} python
:load: ../../examples/functional_material_processes.py
:tags: [remove-input]
```
