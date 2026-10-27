---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-network-extra)=
# Bind extracellular stimulation to a Network

Pass `extra={population_name: specification}` to `dn.compile(Network, ...)` or
`dn.func.make_functional_network`. Each mapping value uses the ordinary
Population extracellular specification: `(field, waveform)` for one contact,
or a sequence of these pairs for multiple contacts. Omit a Population, or map
it to `None`, to leave its extracellular input unbound.

```python
extra = {
    "left": [(field_a, waveform_a), (field_b, waveform_b)],
    "right": (field_a, waveform_a),
}
runner = dn.compile(network, dt=0.05, steps=100, extra=extra)
state, auxiliary = runner()
```

The binding uses the Population stimulation contract described in
{doc}`A8_functional_populations`. Fields and waveform values multiply and sum
to produce the extracellular potential in mV. Dendra evaluates each waveform
at the accepted simulation clock, including during explicit continuation and
checkpoint replay. The bound contacts are part of the functional plan; calls
receive their numerical values through explicit parameter leaves.

The complete
{download}`functional_network_extra.py <../../examples/functional_network_extra.py>`
example builds two connected, batched passive cables. It compares scan and
while execution, continues recordings across calls, verifies shared field and
waveform aliases, and differentiates through the stimulation. Run it from the
repository root with `python examples/functional_network_extra.py`.

## Build the Network and describe its contacts

Initialize the ordinary Network before lowering or compiling it. Select its
mode and timestep in the same way as in
{doc}`A21_functional_network_compile`:

```{literalinclude} ../../examples/functional_network_extra.py
:language: python
:start-after: "# region network"
:end-before: "# endregion network"
:dedent: 4
```

Use exact top-level Population names as mapping keys. An unknown name is an
error. Spatial fields follow the selected Population's shape and broadcasting
rules. In this example, each cable has shape `(batch=2, cells=1, compartments=5)`;
a field of shape `(1, 5)` applies the same spatial profile in both batch lanes.
Fields with explicit batch axes can assign different profiles to those lanes.

```{literalinclude} ../../examples/functional_network_extra.py
:language: python
:start-after: "# region extra"
:end-before: "# endregion extra"
:dedent: 4
```

The left Population receives two contacts, while the right receives one.
Both reuse the same first field Tensor and Waveform object. Their explicit
parameter leaves retain those shared identities, so an update to a shared
leaf changes the contact in both Populations.
Sharing follows source Tensor object identity, including the leaves of a
reused Waveform; views that share storage are separate leaves. A shared source
requiring incompatible conversions for its participating Populations, such as
different dtypes, is rejected. Use separate source Tensors for those contracts.

Complete concatenation, batching, and other structural transformations before
binding. A concatenated MultiPopulation is one top-level owner. For example,
after `network = network.concat(packed=["exc", "inh"])`, bind under `"packed"`
and supply a field in that packed Population's coordinates. `"exc"` and
`"packed.exc"` are not separate binding keys. Pad or mask the packed spatial
field to stimulate only a component. Changing a constructor binding's contact
structure requires a new functional plan or runner. A runtime slot, described
below, lets each call choose its contact structure.

Constructor binding requires a `Waveform` temporal specification for every
contact. Tensor temporal specifications are currently unsupported in these
bindings. Already sampled stimulation uses an explicit time-first `ve` tape,
as described below. Deterministic built-in waveforms, compositions, and
supported custom pure Waveforms follow the contract in
{doc}`A8_functional_populations`. Finite-count Poisson waveforms with
`randomize_every_call=True` use explicit stimulation draw/RNG packets;
see {doc}`A50_functional_poisson_stimulation`. Hidden random draws and waveform
hooks are currently unsupported.

## Select and inspect trainable stimulation

Bound stimulation leaves appear under each Population's parameter namespace:

```text
populations.left.stimulation.extra.contacts.0.field
populations.left.stimulation.extra.waveforms.0.amp
populations.left.stimulation.extra.waveforms.0.freq
populations.left.stimulation.extra.waveforms.0.phase
```

The exact waveform leaf names depend on its type and composition. Inspect
`runner.named_parameters`, `runner.find_parameters(...)`, or
`runner.trainable_parameters` to see available names and the selected leaves.
Use exact names, glob patterns, or regular expressions with `trainable`:

```{literalinclude} ../../examples/functional_network_extra.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 4
```

`inference=False` enables gradients through scan execution. Omitting
`trainable` selects no differentiable runner leaves; `inference=True`
suppresses gradient recording regardless of that selection.

Runner construction snapshots and detaches the authored inputs. Optimize the
independent leaves in `runner.parameters`, rather than the original field or
Waveform object. Fields and waveform buffers are selectable even if they were
authored as ordinary nontrainable Tensors. Shared aliases appear under each
participating name, but refer to the same leaf. Deduplicate them when building
an optimizer:

```{literalinclude} ../../examples/functional_network_extra.py
:language: python
:start-after: "# region parameters"
:end-before: "# endregion parameters"
:dedent: 4
```

Numerical updates retain the frozen contact structure. A complete
`parameters=` override may contain non-leaf Tensors, such as a field computed
from trainable electrode coordinates; their autograd history is preserved.
Preserve shared aliases in replacement packets. Changes to stimulation
leaves do not require another `prepare` call.
When re-extracting authored values, tied Waveform leaves must retain their
shared Tensor identity; splitting those ties requires a new plan.

`plan.extract(other_network)` reads that Network's registered parameters and
intracellular stimulation while retaining the plan's external `extra` recipe.
If copying a Network separates registered sources from shared external
contacts, extraction rejects the broken aliases. Continue with the existing
explicit packet, or lower the copied Network with matching contact bindings.

## Continue the waveform clock and its gradients

A call without state restarts the runner's snapshot, or applies
`initialization="fresh"` when selected. Pass returned simulation state to
continue from its clock. Waveform time follows that state and does not restart
at each call or chunk. Pass callback state as well to continue the same
recording without duplicating the boundary frame:

```{literalinclude} ../../examples/functional_network_extra.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 4
```

The example compares two continued calls with one uninterrupted call.
Checkpointed scan preserves the same waveform samples and gradients without
consuming hidden randomness. The continued state remains connected to its
prefix graph:

```{literalinclude} ../../examples/functional_network_extra.py
:language: python
:start-after: "# region gradients"
:end-before: "# endregion gradients"
:dedent: 4
```

Field values multiplied by waveform values must produce mV. For example,
unit-current fields in mV/mA pair with waveform amplitudes in mA. Selecting a
timing parameter does not smooth a waveform's hard timing switches;
derivatives at these boundaries remain piecewise or discontinuous.

Inference-only while execution uses the same contacts and accepted clock.
Use `execution="while", inference=True`; a full loop block reports
`compiled_while_loop`, and a shorter remainder uses the bounded tail described
in {doc}`A25_functional_network_while`.

## Choose Network contacts at call time

Compile with `extra=dn.VarInput()` to declare a runtime extracellular slot for
every top-level Population. Supply a mapping of ordinary Population contact
specifications on each call:

```python
runtime = dn.compile(
    network,
    dt=0.05,
    steps=100,
    extra=dn.VarInput(),
    chunk_steps=32,
    inference=False,
    callbacks={"trace": dn.func.Recorder(("left.v", "right.v"))},
)
state, auxiliary = runtime(extra={"left": (field_a, waveform_a)})
other_state, other_auxiliary = runtime(
    extra={
        "left": [(field_a, waveform_b), (field_b, waveform_a)],
        "right": (field_b, waveform_b),
    },
)
```

An explicitly supplied mapping can omit declared slots; omitted slots receive
zero extracellular potential. Mapping a slot to `None` has the same effect.
Pass `extra=None` to supply zero potential to all runtime slots. The whole
`extra` call argument is required whenever the runner declares any slots;
omitting it raises `TypeError`.

To restrict runtime input to selected Populations, put `VarInput()` values in
the constructor mapping. Other entries can bind fixed contacts:

```python
selective = dn.compile(
    network,
    dt=0.05,
    steps=100,
    extra={"left": dn.VarInput(), "right": (field_a, waveform_a)},
    chunk_steps=32,
)
state, auxiliary = selective(extra={"left": (field_b, waveform_b)})
```

Only declared runtime names are accepted in a call mapping. Unknown names,
statically bound names, and unbound names without a slot are rejected. A
runner with no runtime slots rejects call-local `extra` overrides, including
`None`. Exact top-level names and packed Population coordinates follow the
same rules as constructor bindings.

Each slot accepts different deterministic Waveform types, compositions, and
contact counts on subsequent calls. Tensor temporal specifications are also
accepted, with a trailing time axis containing exactly `steps` samples for
that call and the ordinary Population broadcasting rules. Within one
Population's specification, all contacts must use Waveforms or all must use
temporal tensors. Dendra slices temporal tensors across chunks using their
global sample index for the call. Runtime specifications use caller-owned
tensor values and preserve their autograd history with `inference=False`,
including fields computed from upstream differentiable tensors. They create
no `stimulation.extra.*` runner parameter leaves and do not use `trainable`.

The compiled Network consumes assembled extracellular voltage in bounded
chunks. Changes to runtime contact recipes reuse its model kernel;
call-local contact plans and autograd graphs are not retained between calls.
`chunk_steps` is a positive integer accepted only when runtime slots are
declared. It defaults to 64, with the kernel width capped at `steps`.
In training scan, `checkpoint_every=H` uses spans of at most
`min(H, chunk_steps)` steps. Recordings retain their requested trajectory.
Inference-only while execution keeps the usual bounded tail behavior;
chunks shorter than four steps use the bounded unrolled tail.

Runtime Waveforms follow the accepted simulation clock across continuation
and checkpoint replay. Continue with both returned simulation state and
callback state as in the bound example above. Runtime slots conflict with
explicit `ve` on the same Population, including when a slot receives zero
potential; other unbound Populations can receive explicit voltage tapes.
The source Network, fields, and Waveforms remain unchanged.

Runtime Waveform evaluation must be deterministic and side-effect-free.
Fixed Poisson schedules with `randomize_every_call=False` are supported.
Randomized Poisson Waveforms are unsupported in call-local `extra`; bind them
at construction to use the explicit stimulation RNG contract in
{doc}`A50_functional_poisson_stimulation`.

## Combine bindings and explicit voltage tapes

Each Population receives one source of extracellular potential. An explicit
`ve` input conflicts with bound `extra` on that same Population. This rule
applies to both `StepInput.ve` and `RolloutInput.ve`. Runtime `extra` also
conflicts with a constructor binding on the same owner. Other Populations may
receive explicit `ve` tapes independently.

This example binds contacts only to `left` and supplies an assembled time-first
voltage tape to unbound `right`. It verifies the result against the fully bound
execution:

```{literalinclude} ../../examples/functional_network_extra.py
:language: python
:start-after: "# region mixed-inputs"
:end-before: "# endregion mixed-inputs"
:dedent: 4
```

The tape's leading axis is time. Its payload follows the selected Population's
ordinary voltage-input broadcasting rules; the example expands it to
`(steps, *population.shape)` explicitly. For precomputed stimulation, supply
these mV values directly rather than wrapping a sampled time Tensor in a
constructor `extra` specification.

Lower-level bound plans use the same inputs in `step`, `rollout`, and `run`.
Bind outside an active `torch.func` transform or `torch.compile` capture, then
transform the explicit Tensor packets. A plan and its runner leave the source
Network unchanged. `commit_state_` publishes runtime carry; it does not bind
external stimulation to the native Network. Supply the corresponding `extra`
specification separately and align any optimized numerical stimulation values
when continuing with native execution. Native `Network.step` and `run` accept
one `(field, waveform)` pair per Population; mapping entries containing a list
of contact pairs, as used for `left` in this example, are currently unsupported
there. Native continuation therefore requires a compatible native stimulation
specification.

## Run the complete example

```{code-cell} python
:load: ../../examples/functional_network_extra.py
:tags: [remove-input]
```
