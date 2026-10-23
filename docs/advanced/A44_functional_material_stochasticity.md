---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-material-stochasticity)=
# Sample MaterialProcess parameters and runtime noise

MaterialProcesses use `GLOBALRAND`, `BATCHRAND`, and `RANGERAND` for sampled
parameter heterogeneity, and `GLOBALNOISE`, `BATCHNOISE`, and `RANGENOISE` for
detached noise refreshed on each timestep. The declarations and pure Material
field hooks work with native execution, functional Populations, and
`dn.compile(Network)`.

The complete
{download}`functional_material_stochasticity.py <../../examples/functional_material_stochasticity.py>`
example compares a two-Population conservative transfer process, a shared
distribution-parameter gradient, explicit draws, and continued RNG state with
native execution. Run it with
`python examples/functional_material_stochasticity.py`.

## Declare heterogeneity and runtime noise

This process samples a positive transfer rate at initialization and modulates
it with independent normal noise on each timestep:

```{literalinclude} ../../examples/functional_material_stochasticity.py
:language: python
:start-after: "# region process"
:end-before: "# endregion process"
```

`rate` remains fixed for the simulation. Its default
`reparameterized=True` allows gradients through the sampled rate to
`rate_mu_param`, `rate_sigma_param`, and supplied base draws. Set
`reparameterized=False` to detach the sample. The `eta` sample is detached from
its distribution parameters, base draw, and timestep; ordinary trainable
operands such as `gain` remain differentiable when multiplied by that sample.
See {doc}`A30_functional_random_parameters` for distribution registration and
resampling policies, and {doc}`A31_functional_runtime_noise` for noise scaling.

`GLOBAL`, `BATCH`, and `RANGE` scopes retain their native layouts, including
regional insertion and component support in concatenated MultiPopulations.
Use `_resolve_quantity(name, like)` to expand a declared operand to a full
Material field and `_mask_material_updates(fields, updates)` to preserve
values outside the process support. The returned fields preserve the same
conservation law as their native equivalents.

NOISE sampling occurs before the Population's state and voltage transition.
Pointwise and custom processes evaluate `prepare_materials(fields, dt)` at
their Material phase, after preceding Material writes, with the current
timestep's explicit noise.
The hook retains its pure, fixed-schema contract: it uses field metadata rather
than evolving field values. Field-dependent arithmetic belongs in
`material_updates`. Preparation does not sample randomness or mutate inputs.
MaterialProcess `PHASE` orders field updates; random draws follow insertion
order independently of that phase ordering.

Native runtime sampling refreshes NOISE buffers in place. If a custom hook
reads a buffer directly and its value can be saved for another operand's
backward pass, use a copy such as `self.eta.clone()` for that step. The shared
`_resolve_quantity("eta", like)` helper already copies a named NOISE operand,
so built-in process families and custom hooks using that helper preserve the
accepted sample through backward.

`DiffusionProcess` retains its native timestep configuration: `set_dt`
prepares spatial operator coefficients, and accepted steps reuse those
coefficients. A NOISE declaration used by that preparation supplies its value
at configuration time; refreshing the buffer does not rebuild the operator
on every step. Explicit state retains both the configured workspace and the
NOISE realization used to configure it. Functional preparation uses that held
realization with the explicit deterministic parameters, RAND samples, and
geometry, preserving their gradients without substituting later noise samples.
Snapshot and runtime checkpoint continuation retain this distinction between
the current NOISE buffer and configured operator.

Fresh initialization configures the operator from its initialization NOISE.
Initializing from a steady-state cache rebuilds the operator from the restored
time-zero NOISE and restored parameter frame, following native solver setup.
It does not continue the operator that happened to be active when the cache
was saved.

## Select parameters and supply initialization draws

Train distribution parameters with the same selectors used for other model
parameters. Qualified aliases of a shared parameter refer to one optimizer
leaf:

```{literalinclude} ../../examples/functional_material_stochasticity.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 4
```

The host runner samples missing packets automatically. To supply them
explicitly, sample outside compiled or transformed code and assign each
Population's packet to `InitializationInput.random_draw`:

```{literalinclude} ../../examples/functional_material_stochasticity.py
:language: python
:start-after: "# region initialization"
:end-before: "# endregion initialization"
:dedent: 4
```

Each packet contains the ordered RAND and initialization NOISE draws and their
post-draw RNG carry. Within a Population, initialization visits ordinary
Mechanisms and nested States, then MaterialProcesses in insertion order. Each
MaterialProcess samples RAND before its initialization NOISE. Initial NOISE
uses the native initialization scaling timestep of `1.0`; runtime noise uses
the simulation timestep. Declarations with the same `rng_name` consume one
shared stream. Sampling leaves the source model and input packets unchanged.
Sharing mutable RNG owners between different Network Populations is currently
unsupported.

## Continue from an explicit RNG boundary

`sample_population_noise_tapes(state, steps)` returns time-leading base draws
and the corresponding post-draw RNG boundaries. Supply each tape through that
Population's `RolloutInput.noise`. A continued run consumes the returned
parameter, constant, and state packets:

```{literalinclude} ../../examples/functional_material_stochasticity.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 4
```

`noise_inputs` in the example uses this helper to sample from the selected
boundary. The second tape starts from `prefix`; it neither reinitializes the
RAND sample nor repeats a runtime draw. Scan, checkpointed scan, and
inference-only while execution consume the same explicit tapes. Checkpoint
backward and repeated parameter preparation do not advance the RNG.

The example checks that the shared RAND distribution mean has the same
gradient as native execution, while the runtime noise mean has no gradient:

```{literalinclude} ../../examples/functional_material_stochasticity.py
:language: python
:start-after: "# region gradients"
:end-before: "# endregion gradients"
:dedent: 4
```

Snapshot initialization retains the captured realization, Material fields,
NOISE buffers, and RNG suffix. Fresh initialization applies RAND's
`resample_on_initialize` and `reseed_on_initialize` policies. A resampled RAND
declaration invalidates a Population-owned steady-state cache; a declaration
which retains its sample can restore that cache. A Network-global cache
restores its saved samples, fields, and RNG without fresh draws. See
{doc}`A29_functional_network_cache_initialization` for cache ownership and
{doc}`A43_functional_initialization_ordering` for shared initialization writes.

Runtime checkpoints include sampled provenance, current NOISE buffers, RNG
carry, and configured spatial workspace with its held NOISE realization.
`commit_state_` restores these to a compatible native model; align its raw
parameters and initialization separately before continuing.
A checkpoint can be restored before the target's first native run. Its first
solver setup retains the saved spatial operator when the timestep matches;
changing the timestep rebuilds the operator from the current NOISE frame.
Ordinary `load_state_dict` restoration invalidates spatial preparation and
rebuilds it from the restored inputs at the next solver setup.

MaterialProcess does not provide a State `DIFFUSION` declaration. Use the
supported State SDE API in {doc}`A32_functional_state_sde` for Itô or
Stratonovich state equations. Implicit random draws inside MaterialProcess
hooks and custom imperative sampling adapters are currently unsupported.
`dn.ctx(CHECK_PURITY=True)` enables optional purity diagnostics when lowering;
explicit schemas and canonical lifecycle requirements apply in either mode.

```{code-cell} python
:load: ../../examples/functional_material_stochasticity.py
:tags: [remove-input]
```
