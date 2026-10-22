---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-mechanism-injections)=
# Compile mechanisms that own waveform injections

Some mechanisms interpret an injected waveform within their own current or
voltage equations. A mechanism can accept a waveform supplied through
`population[slice].inject(waveform)`, evaluate it with an explicit clock, and
use the same equations in native and functional execution. Ordinary mechanisms
that do not accept injections leave their delivery to the Population's solver.

The complete
{download}`functional_mechanism_injections.py <../../examples/functional_mechanism_injections.py>`
example differentiates a waveform inside a membrane-current mechanism and a
`VoltageProcess`, checks both against native trajectories, and continues a
compiled connected Network. Run it from the repository root with
`python examples/functional_mechanism_injections.py`.

## Accept and evaluate an injection

Override `Mechanism.inject` and delegate cold registration to
`register_waveform_injection`. Declare a repeatable current with `ASSIGNED`,
then return its value from the pure `assigned_values` hook:

```{literalinclude} ../../examples/functional_mechanism_injections.py
:language: python
:start-after: "# region current-owner"
:end-before: "# endregion current-owner"
```

`injection_values(v, *, t, current_name="i_inj")` returns the sum of matching
waveforms padded to the mechanism-local shape of `v`. It does not install the
result on the module. The explicit scalar `t` is measured in ms; `self.t` in a
pure hook refers to the execution's current Population clock. Treat input
tensors and waveform modules as read-only. The helper preserves gradients
through waveform parameters and tensor-valued time.

`ASSIGNED` describes algebra that can be evaluated repeatedly from the current
inputs. A waveform current is normally assigned algebra, rather than `CARRY`:
it has no independent history to preserve between accepted steps. If a mechanism
also integrates an adaptation variable or retains another value, declare that
separate value as State or `CARRY` and update it through its pure transition
hook. See {doc}`A8_functional_populations` for the hook contracts.

In the example, positive waveform values contribute inward current through
`gain * i_inj`. Accepting a waveform makes the mechanism responsible for its
meaning and units within that equation; merely registering it does not add a
membrane current. Use a gain or conversion factor appropriate to the model.

Attach a waveform through the ordinary Slice API:

```{literalinclude} ../../examples/functional_mechanism_injections.py
:language: python
:start-after: "# region attach"
:end-before: "# endregion attach"
:dedent: 4
```

An injection is offered to mechanisms in their native order. The first mechanism
returning `True` accepts complete ownership; the solver does not deliver that
waveform a second time. Registration succeeds only when the mechanism covers
every targeted compartment. Partial overlap returns `False` without storing a
registration, allowing another mechanism or the solver to deliver the complete
injection. A regional mechanism can accept a slice contained within its support.
Registration and changes to injection targets take place before lowering.

## Named currents, scales, and spatial values

Pass `current_name="i_exc"` or another name to `register_waveform_injection`
to group waveforms. Read that group with
`self.injection_values(v, t=self.t, current_name="i_exc")` and return it under a
matching declared `ASSIGNED` name. Multiple registrations in the same group
sum; different groups remain independent. An empty group returns zeros.

The registration's `scale` multiplies its waveform before padding. Scalar
scales are convenient for unit conversion or sign changes. Tensor scales must
broadcast with the waveform's time-last output. Registration materializes a
detached scale buffer; trainable amplitudes belong in the waveform's parameters.

Waveform outputs can be scalars, tensors broadcastable to the mechanism-local
voltage shape, or vectors containing one value per targeted local compartment.
Selected-compartment vectors can have explicit batch axes. The helper removes
the singleton time axis after evaluating one time value, then applies the
registered mask. Regional support, explicit Population batches, and concatenated
MultiPopulations retain their ordinary layout.

## Use a waveform in a voltage transition

A `VoltageProcess` can evaluate the waveform directly inside `voltage_values`;
an exposed assigned current is optional when no other equation needs it:

```{literalinclude} ../../examples/functional_mechanism_injections.py
:language: python
:start-after: "# region voltage-owner"
:end-before: "# endregion voltage-owner"
```

This example uses `dn.scnv`, which commits the voltage produced by the process.
The process retains its integrated voltage in the declared `owned` carry and
returns it on each transition. SCNV native runtime checkpoints retain the
mechanism's declared State and carry; keeping integrated voltage there makes
native checkpoint continuation complete.
The voltage phase reads the incoming step's clock and runs before gating and
current assembly. A mechanism current is evaluated during its ordinary current
assembly phase. Both retain native ordering; the helper does not advance time.

`evaluate_injections(v=None, t=None, current_name="i_inj")` remains the native
adapter for code that needs to assign an exposed current. Pure hooks use
`injection_values` and return their results instead of calling this mutating
adapter.

## Differentiate, record, and continue

Waveform parameters are available through the usual functional parameter tree
under the owning mechanism's `injected_waveforms` children. Wildcards select
the desired leaves, and `runner.trainable_parameters` shows the complete
selection:

```{literalinclude} ../../examples/functional_mechanism_injections.py
:language: python
:start-after: "# region differentiate"
:end-before: "# endregion differentiate"
:dedent: 4
```

The compiled run leaves the source Population unchanged. Scan execution supports
gradients through the waveform, mechanism parameters, incoming voltage and
clock, including checkpointed rollouts. The waveform's own timing semantics,
including any hard gates and surrogate gradients, apply unchanged.

The same waveform module can be shared across mechanisms or Populations.
Functional parameter aliases preserve that sharing; deduplicate identical
parameter objects before constructing an optimizer when inspecting multiple
aliases. This Network uses one waveform for both Populations:

```{literalinclude} ../../examples/functional_mechanism_injections.py
:language: python
:start-after: "# region shared-network"
:end-before: "# endregion shared-network"
:dedent: 4
```

Pass explicit state and callback state to continue a recorded rollout:

```{literalinclude} ../../examples/functional_mechanism_injections.py
:language: python
:start-after: "# region continue-network"
:end-before: "# endregion continue-network"
:dedent: 4
```

Fresh initialization and steady-state cache restoration follow the native
Population or Network policy. Waveform currents are derived at the restored
clock; waveform parameters remain parameter inputs. Snapshot execution, explicit
continuation, native checkpoint restoration, and `commit_state_` retain the
same injection ownership. There is no separate waveform clock or hidden cursor.

Waveforms used during functional execution must be pure. Custom `Waveform`
subclasses declare `FUNCTIONAL_PURE = True` on each concrete class after
ensuring evaluation is deterministic and read-only. Custom modules that
mutate their inputs or persistent state are currently unsupported. Numerical
dependencies used by waveform helpers belong in registered parameters or
buffers. Define custom recipes with `fn` or `forward`; overriding module call
or attribute dispatch is currently unsupported. Waveforms that draw randomness
during evaluation and hooks on source
waveform modules or their parameters are currently unsupported. Changing the
registration count, targets, groups, or ownership after lowering requires a new
functional plan. Runtime injection registration inside captured execution is
currently unsupported.

Custom waveform hooks call registered children directly, for example
`self.child(t)` or `self.child.recipe(t)`. Calling through a local module alias
or dispatching a helper through an indexed `ModuleList` is currently
unsupported. Built-in waveform compositions support their ordinary APIs.

```{code-cell} python
:load: ../../examples/functional_mechanism_injections.py
:tags: [remove-input]
```
