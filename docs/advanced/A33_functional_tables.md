---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-tables)=
# Compile models with lookup tables

`Mechanism.TABLE` and `State.TABLE` tabulate a single-input method over a
uniform grid. Native and functional execution share the same pure linear
interpolation. Fixed tables are constants; tables declared with
`learnable=True` expose their sampled values as trainable parameters.

The complete
{download}`functional_tables.py <../../examples/functional_tables.py>`
example checks native Population trajectories, computes table-value gradients, and
continues a compiled Network. Run it from the repository root with
`python examples/functional_tables.py`.

## Declare and select a table

Declare the method's name, grid bounds, and number of samples. Call the method
normally from pure hooks such as `initial_values` and `advance`:

```{literalinclude} ../../examples/functional_tables.py
:language: python
:start-after: "# region mechanism-table"
:end-before: "# endregion mechanism-table"
```

Construction evaluates `curve` at the declared grid points and stores the
interpolation samples. A learnable table exposes those samples as
`curve_table.y`. Calls to `self.curve(v)` use the table when
`usetable_curve` is `True`. Interpolation clamps queries outside the declared
range to the endpoint values. The input's shape is preserved, including
regional support and explicit Population batches.

Set `mechanism.usetables(False)` to use the original methods, or set an
individual `usetable_<method>` flag to select one method. `Mechanism.usetables`
also applies to its nested State bundles. Configure these flags before
functional lowering; they are part of the lowered model's fixed structure.
Changing a flag requires a new functional plan or compiled runner.

States declare tables with the same API. This gate interpolates its target
from voltage and relaxes toward it using `cnexp`:

```{literalinclude} ../../examples/functional_tables.py
:language: python
:start-after: "# region state-table"
:end-before: "# endregion state-table"
```

`learnable=False`, the default, stores fixed sampled values. With
`learnable=True`, select the values using the ordinary `trainable` expressions:

```{literalinclude} ../../examples/functional_tables.py
:language: python
:start-after: "# region train-table"
:end-before: "# endregion train-table"
:dedent: 4
```

Inspect `runner.trainable_parameters` to see every selected optimizer leaf.
`runner.get_parameter("*curve_table.y")` returns a unique matching parameter;
use a population or State-qualified expression if the model has several
tables with that name. A State table's path includes its `DE.<StateName>`
owner. Shared table parameters remain shared when selected through either
alias, and gradients accumulate from all uses.

## Understand table gradients

Gradients pass through linear interpolation to its selected table values and
query values. A query between two grid points sends weighted gradients to
those two samples. Gradients through query inputs, other mechanism parameters,
State evolution, and downstream membrane voltage follow ordinary autograd
rules. Scan, checkpointed scan, and pure step execution use this same
interpolation. While execution is inference-only.

The grid and table samples are created during native model construction.
Changing a parameter used by the original `curve` implementation does not
regenerate the sampled table or connect that parameter to table evaluation.
Optimize `curve_table.y` to learn the tabulated function, or disable the table
before lowering to differentiate through the original method. The grid
positions are fixed structural data.

Functional execution receives every table tensor explicitly. It does not
update a lookup workspace or mutate the source table during evaluation.
Ordinary Native and functional checkpoint/continuation rules still apply;
no additional table input packet is required.

Keep authored table methods deterministic and read-only. Functional lowering
trusts this contract by default; creating a plan inside
`with dn.ctx(CHECK_PURITY=True):` enables diagnostic purity checks. Registered
table bindings, grid layout, and returned tensor schemas retain their required
validation in either mode.

## Initialize and continue a Network

Snapshot initialization starts from the captured Population states. Fresh
initialization runs the ordinary pure hooks using the supplied current table
values. It preserves learned table samples instead of rebuilding them from
the original method. A steady-state cache restores its saved table values,
state, and materializations using the existing cache contract, preserving
parameter identity and gradients. Clear the cache before initializing from
updated table values when its saved realization is no longer wanted.

```{literalinclude} ../../examples/functional_tables.py
:language: python
:start-after: "# region network-continuation"
:end-before: "# endregion network-continuation"
:dedent: 4
```

Calling the runner without a state restarts its chosen initialization.
Passing returned simulation and callback state continues the same run.
Full model `state_dict()` checkpoints include registered table tensors.
Runtime `state_dict_for_checkpoint()` packets and `functional.commit_state_`
transfer simulation state. Align native raw parameters and fixed table data
separately when they differ from the functional inputs.

Functional lowering supports declared Mechanism and State tables with the
canonical generated dispatch and interpolation modules. Custom dispatch
overrides, replaced table implementations, and changes to grid structure
after lowering are currently unsupported. General Population and Network
admission rules still apply.

## Run the complete example

```{code-cell} python
:load: ../../examples/functional_tables.py
:tags: [remove-input]
```
