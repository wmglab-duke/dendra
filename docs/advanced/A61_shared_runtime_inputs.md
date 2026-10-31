---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.13'
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(shared-runtime-inputs)=
# Deterministic inputs for native and functional execution

`dn.func.Inputs(ve=..., intra=...)` supplies deterministic extracellular voltage
and intracellular current to native Populations and Networks, functional
bindings and `dn.compile` runners. The same packet keeps its Tensor references,
views and gradients when it crosses an execution boundary.

| Field | Quantity | Units |
| --- | --- | --- |
| `ve` | Extracellular voltage at each compartment | mV |
| `intra` | Total injected current at each compartment | mA |

Both fields are optional. A native `step` consumes one frame; `run`, `longrun`
and `longrun_checkpointed` consume time-leading tapes. The dimensions after
time use the Population's existing broadcasting rules: ordinary trailing
broadcasting first, then the supported right-aligned batch-only fallback.
Explicit singleton axes distinguish batch and spatial dimensions.

The complete runnable
{download}`shared_runtime_inputs.py <../../examples/shared_runtime_inputs.py>`
example compares native and functional values and gradients:

```{literalinclude} ../../examples/shared_runtime_inputs.py
:language: python
:start-after: "# region population"
:end-before: "# endregion population"
:dedent: 4
```

## Native Population calls

Pass `inputs=` to the ordinary native execution method:

```python
model.step(dt, inputs=dn.func.Inputs(ve=voltage_frame, intra=current_frame))
model.run(dt=dt, inputs=dn.func.Inputs(ve=voltage_tape, intra=current_tape))
model.longrun(duration, chunklength=100, dt=dt, inputs=inputs)
model.longrun_checkpointed(
    duration, chunklength=100, dt=dt, inputs=inputs, safe_checkpoint=True
)
```

A Population `run` can infer the number of steps from its tapes when `tstop`
is omitted. Without legacy `ve=`, an explicit `tstop` requires every tape to
contain exactly the accepted step count. That count uses the ordinary duration
budget, including a retained
fractional duration from previous calls. All supplied tapes must have the same
time length. Empty tapes represent a zero-step call when the accepted duration
contains no complete step.

`longrun` and `longrun_checkpointed` retain their required duration and
`chunklength`. Chunking divides solver execution and checkpoint replay; it
does not bound the whole input tape, recorded history or autograd graph.
Broadcast normalization can materialize a contiguous full-horizon tape.
Partition calls explicitly when supplying inputs in bounded streaming spans.

## Native Network calls

A Network accepts a mapping from its Population names to the same packets.
Omitted Populations and omitted fields keep their normal drive behavior:

```{literalinclude} ../../examples/shared_runtime_inputs.py
:language: python
:start-after: "# region network"
:end-before: "# endregion network"
:dedent: 4
```

Native Network `run`, `longrun` and `longrun_checkpointed` require their usual
duration. Each supplied tape must match that call's accepted step count.
`network.step(inputs={"left": dn.func.Inputs(intra=current_frame)})` consumes
frames without a separate duration. The Network's initialized timestep applies
to every Population.

## Waveforms and existing stimulation arguments

`Slice.inject` remains the waveform binding API. Supplying `Inputs(ve=...)`
leaves its intracellular waveform active:

```{literalinclude} ../../examples/shared_runtime_inputs.py
:language: python
:start-after: "# region injection"
:end-before: "# endregion injection"
:dedent: 4
```

Explicit `intra` cannot be combined with solver-level intracellular stimulation
registered on that Population. Mechanism-owned waveform injections keep their
own current channels and can coexist with explicit `intra`.

Explicit `ve` conflicts with legacy `ve=` or extracellular `extra=` for the
same Population. `Inputs(intra=...)` can be combined with `extra=` because
they supply different drive quantities. Network conflicts are checked per
Population, so another Population can continue using its own `extra` binding.
Conflicts and malformed packets reject before waveform evaluation and random
sampling.

The legacy entry points retain their existing behavior. In particular,
Population `run(ve=...)` uses the voltage tape's time length as authoritative,
even when `tstop` is also supplied. This also applies when legacy `ve=` is
combined with `Inputs(intra=...)`: the current tape must match the voltage
tape's length, and `tstop` is ignored. Without legacy `ve=`, the canonical
`inputs=` form checks an explicit duration against its tapes instead.

## Tensor ownership and validation

Canonical fields must be Tensors on the model's device and with its dtype.
The packet does not implicitly convert lists, change dtype or move devices.
Convert explicitly when needed; ordinary PyTorch conversion retains the
appropriate autograd connection. Legacy `ve=` and `extra=` retain their
existing coercion policies.

Input leaves remain caller-owned. Gradients propagate through broadcast,
chunking and checkpoint replay. Keep leaves unchanged until backward has
finished; mutating a saved input can invalidate checkpoint recomputation.
Runtime state and the physical duration remainder continue through the native
model's existing checkpoint methods. Supply the remaining input tape again
when resuming.

`Inputs` contains deterministic drives only. It does not add an explicit
`draws=` argument to native execution. Native RNG ownership and sampling order
remain part of the native lifecycle; functional draw packets remain described
in {doc}`A57_functional_api_overview`.

The complete example executes here:

```{code-cell} python
:load: ../../examples/shared_runtime_inputs.py
:tags: [remove-input]
```
