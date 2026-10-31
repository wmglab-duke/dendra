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

(native-source-inspection)=
# Inspecting native model sources

`dn.func.inspect_sources(model)` accepts native Populations and Networks as
well as functional runners, tensor bundles, raw bindings and continuations.
The returned `SourceReport` identifies raw Tensor sources, their aliases and
the unique trainable leaves that can be passed to a PyTorch optimizer.

Native inspection does not lower the model, build it, initialize it or execute
its mechanisms, parametrizations or waveforms. It reads registered metadata
without evaluating source properties, copying Tensor values, sampling random
values or preparing solver workspaces. It does not establish whether a model
can be functionally lowered.

The complete runnable
{download}`native_source_inspection.py <../../examples/native_source_inspection.py>`
example prints Population and Network reports and forms two optimizer groups:

```{literalinclude} ../../examples/native_source_inspection.py
:language: python
:start-after: "# region inspect"
:end-before: "# endregion inspect"
:dedent: 8
```

## Native names and live objects

Native report paths follow registered model names, with a `parameters.` or
`constants.` prefix. For example:

| Source | Native report path |
| --- | --- |
| A Population mechanism parameter | `parameters.integrator.mech.mechanisms.leak.g_param` |
| A Network Population's mechanism parameter | `parameters.left.integrator.mech.mechanisms.leak.g_param` |
| A registered connection parameter | `parameters.synapses.<connection_name>.weight.weights.0.param.rho` |
| A Population geometry source | `constants.diam` |
| A Network Population geometry source | `constants.left.diam` |
| An owned intracellular waveform parameter | `parameters.injections.0.waveform.amplitude` |

Names reflect the native registrations. Functional packets retain their own
explicit source names, including `parameters.populations.left...` for a
Network. Existing runner, packet and binding reports keep those names and
their existing ownership rules.

Population-owned injection declarations expose their Waveform sources under
`injections.<index>.waveform...` or `mechanism_injections.<index>.waveform...`,
with a Population-name prefix inside a Network. These routes refer to the
already-owned Waveform's registered leaves; inspection does not prepare the
stimulation handler or evaluate the Waveform.

Each entry's `tensor` is the live source Tensor object. Every registered source
alias path is retained, including aliases between parameters and raw constant
readers. Traversal stops ancestor cycles while retaining independent alias
paths. Distinct Tensor views remain distinct sources even when they share
storage; aliases mean the same Tensor object, rather than equal values.

The `parameters` and `constants` domains describe source roles, not gradient
flags. A raw configuration buffer can require gradients and contribute a
trainable leaf. Recognized runtime-state registrations remain excluded even
when they alias a reported source.

`requires_grad` and `is_leaf` are captured when the report is created. The
Tensor values remain live. Reinspect after changing flags, replacing source
objects or rebuilding a model; existing reports keep their captured paths,
flags and references. `print(report)` shows a bounded, human-readable source
tree with shapes, dtypes, devices, gradient flags and aliases. `report.entries`
contains every reported path, including paths omitted from a large display.

## Selecting optimizer groups

`report.select(...)` uses exact names, unique partial names, glob selectors,
`re:` patterns or compiled regular expressions. Selecting one alias retains
the complete identity group. Each selected report's `trainable_leaves` contains
every captured trainable leaf once, in its original order:

```{literalinclude} ../../examples/native_source_inspection.py
:language: python
:start-after: "# region optimizers"
:end-before: "# endregion optimizers"
:dedent: 8
```

Selection does not change `requires_grad`, discover upstream optimizer leaves
for nonleaf source expressions, or prove that a source affects an output.
Deduplication applies within each selection. Check for shared Tensor identities
when combining selections into disjoint optimizer groups or assigning them to
different optimizers.

## Configuration sources and runtime state

Native reports include registered raw Parameters and supported raw
configuration constants, including source buffers used by parametrizations,
connection and NetStim parameter recipes, schedule bindings and registered
intracellular waveforms. They exclude simulation state, mechanism CARRY,
saved currents, derived and timestep workspaces, solver storage, random
samples, RNG state, event queues and Poisson histories.

Inspection works on built and uninitialized native models. Before building,
it reports existing registered sources, together with the Waveforms already
owned by its injection declarations. Pending mechanism or connection declarations
that have not become registered modules or Tensor slots are not materialized
by inspection. Build or initialize through the ordinary model lifecycle when
needed, then obtain a fresh report to inspect the resulting registrations.

Python scalar configuration is not converted into synthetic Tensor entries.
Native reports describe existing source objects; packet reports describe their
explicit Tensor frames.

Unbound call-local inputs, runtime stimulation and random draw tapes remain
outside the report. To inspect the effective explicit sources used by a
particular functional invocation, inspect its returned `Continuation` instead
of the native model. See {doc}`A57_functional_api_overview` for the shared
functional lifecycle.

The complete example executes here:

```{code-cell} python
:load: ../../examples/native_source_inspection.py
:tags: [remove-input]
```
