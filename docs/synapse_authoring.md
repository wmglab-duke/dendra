---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(writing-synapses)=
# Writing your own synapse

A synapse combines a current with a rule for receiving presynaptic input.
For discrete weighted deliveries, inherit `Synapse` and implement the pure
`net_receive_values` hook. For a signal sampled every timestep, inherit
`ContinuousSynapse` and declare its named analog inputs. A prescribed
conductance waveform, such as the built-in `alphasynapse`, needs neither
connection interface.

This tutorial implements both kinds of connected synapse from scratch. The
examples use CPU float64 and disable compilation so the first run is quick.
They run with the installed Dendra package and require no companion models or
NEURON simulation. Download the complete scripts:
{download}`custom_event_synapse.py <../examples/custom_event_synapse.py>` and
{download}`custom_continuous_synapse.py <../examples/custom_continuous_synapse.py>`.
Run them with `python custom_event_synapse.py` and
`python custom_continuous_synapse.py`. The documentation executes these same
files and checks their numerical results.

## Choose a value's lifetime first

The declarations determine how Dendra initializes, advances, checkpoints, and
functionally represents each value. Use the same roles in a synapse as in any
other mechanism:

| Value | Declaration and producer |
| --- | --- |
| Solver-evolved conductance or gate | `State.STATE`, with `DERIVATIVE` or `KINETIC` equations. Register the bundle with `Mechanism.STATE_BUNDLE`. |
| Persistent counter, latch, or history | `CARRY`. Initialize it with `initial_values`; update it through `advance` or an event receive hook. |
| Repeatable current-stage algebra | `ASSIGNED`, returned by pure `assigned_values(v, values)`. Return every declared assigned name. |
| Workspace derived from parameters, temperature, or geometry | `DERIVED_BUFFER`, returned by pure `derive_buffers()`. |
| Prepared workspace that also depends on the timestep | `TIMESTEP_BUFFER`, returned by pure `derive_timestep_buffers(dt)`. |
| A continuously delivered projection | `ContinuousSynapse.INPUT`, reset and filled by the Network every timestep. |
| A read-only input to a nested State equation | `State.INPUT`, supplied from a declared field in the same Population through `Population.bind_input`. |

The returned-value hooks shown here return mappings containing Tensor outputs;
current methods return a Tensor. Treat arguments and module state as read-only.
Read evolving values from the explicit `values` frame in returned-value hooks;
parameters are available through `self`. Runtime replacements preserve the
declared shape, dtype, and device. Avoid in-place additions such as
`values["g"].add_(weights)`.

`State.state_defaults(v, values)` supplies fallback solver-state initial
conditions. An insertion-time `ic` overrides those defaults. Later
`initial_values(v, values)` overlays may intentionally replace initial State
or carry values; they do not advance time. Declared carry starts at zero when
freshly initialized unless an initial-value overlay changes it. Prepared
workspaces are reproducible coefficients rather than simulation history:
reinitialize an imperative model after changing preparation inputs, or call
functional `prepare` again with the new parameters/constants.

The two scopes of `advance` have different jobs:

- A State's default `advance(v, dt, values)` executes its declared equations.
  Override it only for a custom transition; then return every declared solver
  tensor and any updated carry belonging to that State.
- `Mechanism.advance(v, dt, values)` returns a partial mapping of its direct
  carry and owned writable/additive shared fields once per accepted timestep.
  Omitted carry is retained. It does not replace the State solver. State and
  Mechanism transitions read the same incoming snapshot, so the Mechanism hook
  cannot assume it sees a newly solved State value.

`assigned_values` and current methods may be evaluated several times during
one step. Keep timers and counters out of them. See the
{doc}`mechanism authoring guide <advanced/A0_mechanisms_and_ions_materials>`
and {doc}`API reference <api/dendra.models.mechanisms>` for the complete hook
contracts and worked preparation/initialization examples.

(custom-event-synapse)=
## An event-driven exponential point synapse

Our first synapse stores conductance `g`, adds each weighted delivery to it,
and decays it between deliveries:

$$
g \leftarrow g + w, \qquad
\frac{dg}{dt} = -\frac{g}{\tau}, \qquad
I_\mathrm{syn} = g(v-e).
$$

`Synapse` supplies the receive interface. Combining it with `PointProcess`
selects a lumped current: `g` and weights are numerical conductances in µS,
`v` and `e` are in mV, `tau` is in ms, and the outward-positive current is
in nA. Thus `weight=0.05` means 0.05 µS; do not multiply that weight by
`dendra.units.uS`. Dendra handles the target compartment's area conversion.

### Declare the state and current

```{literalinclude} ../examples/custom_event_synapse.py
:language: python
:start-after: "# region imports"
:end-before: "# endregion imports"
```

```{literalinclude} ../examples/custom_event_synapse.py
:language: python
:start-after: "# region custom-synapse"
:end-before: "# endregion custom-synapse"
```

`state_defaults` creates one zero conductance per local slot. Using
`zeros_like(v)` preserves its shape, dtype, and device. `STATE_BUNDLE`
registers the State class; its `g` tensor becomes available to the owning
synapse's current as `self.g`. `tau` belongs to the State and can still be
configured when inserting the synapse.

The receive hook returns a partial, namespaced update. Here `states` replaces
the flattened State variable `g`; more elaborate receivers can also return
direct `carry` or nested `state_carry` replacements. Omitted values are
retained. Reception itself does not advance the clock or perform ODE decay.
Leave the standard `net_receive` adapter in place: it commits these same
returned updates during ordinary Network execution.

### Insert it and deliver a Network event

```{literalinclude} ../examples/custom_event_synapse.py
:language: python
:start-after: "# region population"
:end-before: "# endregion population"
```

The distributed passive leak in this helper uses S/cm², while the synapse
uses µS. Each mechanism keeps its own unit contract. The helper initializes
the population for standalone use; `Network.initialize` subsequently
establishes the complete network starting state.

```{literalinclude} ../examples/custom_event_synapse.py
:language: python
:start-after: "# region network"
:end-before: "# endregion network"
```

`network_demo()` supplies one deterministic `NetStim` event. The first step
detects and queues it; the configured 0.1 ms delay delivers it on the next
step. Delivery occurs before the population solve, so the conductance
observed after that step is already decayed. The assertions check both the
delivery and the following event-free decay. This linear State equation uses
Dendra's generated exponential update, even though the membrane integrator
is backward Euler.

### Use the same receiver with explicit functional state

```{literalinclude} ../examples/custom_event_synapse.py
:language: python
:start-after: "# region functional"
:end-before: "# endregion functional"
```

`functional_demo()` checks the conductance jump, unchanged clock during
reception, subsequent decay, payload gradient, and unchanged source
Population. The payload is already weighted and follows the synapse's local
slot order. `full_like` gives it the correct dtype, device, and runtime shape,
including any explicit batch axes. A smaller Tensor may be broadcast to that
shape; a returned replacement must have the exact declared shape.

The gradient of the next conductance with respect to its payload is
$\exp(-0.1/2) \approx 0.95122942$. Functional reception handles this local
transition. A shared dense fixed-delay NetCon core supplies explicit queued
payloads as described in {doc}`advanced/A12_functional_event_delivery`.
Shared threshold detection and ordered source observation are described in
{doc}`advanced/A13_functional_threshold_detection`. Shared NetStim generation
with explicit draws and clock/RNG state is described in
{doc}`advanced/A14_functional_netstim_renewal`. Shared tensor schedules and
indexed bindings are described in {doc}`advanced/A15_functional_tensor_schedules`.
Consuming heap calendars are described in {doc}`advanced/A16_functional_heap_schedules`.
Complete Network execution still uses the ordinary Network API. See
{ref}`functional-event-receive` for the full functional receive contract.

The complete event example runs here:

```{code-cell} python
:load: ../examples/custom_event_synapse.py
:tags: [remove-input]
```

### Keep timestep recovery separate from delivery

A receiver may be called several times in one timestep by different NetCons,
including calls with zero payloads. A call is a connection delivery, not
necessarily a single biological event. Adding zero leaves this synapse
unchanged, while a plasticity rule must explicitly define how it responds.
Preserve call order for receivers with counters, gates, or other effects.

Put continuous conductance decay in a State equation, as above, and use
`Mechanism.advance` for once-per-accepted-step carry updates. Applying a full
timestep of decay inside each receive call would multiply the decay when
several connections target the same synapse. For independent contacts at the
same compartment, use separate local slots as described in
{doc}`advanced/A3_synapse_banks_and_slots`.

(custom-continuous-synapse)=
## A continuous conductance synapse

For graded transmission, a Network samples a presynaptic variable on every
step instead of detecting a discrete event. A dimensionless presynaptic gate
`s` can drive a weighted conductance input `g_pre`:

$$
g_\mathrm{pre} = \sum_j w_j s_j, \qquad
I_\mathrm{syn} = g_\mathrm{pre}(v-e).
$$

The second script declares the source gate, the target's continuous input and
current, and their connection. It uses the ordinary distributed
`ContinuousSynapse` contract: weights and `g_pre` are in S/cm², and the
outward-positive current is in mA/cm². To model a lumped continuous point
synapse instead, combine `PointProcess` with `ContinuousSynapse` and document
weights in µS and current in nA.

### Declare a source gate and a receiving current

```{literalinclude} ../examples/custom_continuous_synapse.py
:language: python
:start-after: "# region imports"
:end-before: "# endregion imports"
```

```{literalinclude} ../examples/custom_continuous_synapse.py
:language: python
:start-after: "# region declarations"
:end-before: "# endregion declarations"
```

The source's `release` is solver-evolved State. Its instantaneous target
`release_inf` is ephemeral `ASSIGNED` algebra; `assigned_values` can safely
be reevaluated at each required voltage. `state_defaults` supplies an initial
release of 0.25. The source contributes no membrane current itself.

`AFFINE("i")` explicitly asserts that the current is affine in the local
voltage while the delivered conductance is held fixed during current assembly.
Dendra can also infer this property from a simple expression such as this one.

The target declares `ContinuousSynapse.INPUT("g_pre")`. Before each round of
continuous delivery, the Network saves the previous input as `g_pre_old` and
resets `g_pre` to zero. Connections then accumulate their weighted values with
`reduce="sum"`, before the target population advances. Inputs describe the
current timestep's projection; they do not accumulate indefinitely across
timesteps. Declare `keep_old=False` if the previous-input buffer is unnecessary.

`ContinuousSynapse.INPUT` and `State.INPUT` serve different scopes. The former
declares persistent, checkpointed delivery buffers managed by the Network.
The latter exposes a read-only, ephemeral value to a nested State equation;
`Population.bind_input` supplies it from a declared field within the same
Population. Declaring `State.INPUT` alone creates no intercellular connection.
See {ref}`functional-input-bindings` for the binding contract.

### Connect and verify the analog projection

```{literalinclude} ../examples/custom_continuous_synapse.py
:language: python
:start-after: "# region network"
:end-before: "# endregion network"
```

`pre_var="mech.source.release"` selects the live presynaptic State by its
inserted mechanism alias and flattened State name. Each source cell connects
to its corresponding target, with weights of 0.002 and 0.004 S/cm². The
dimensionless gate multiplied by this weight produces the conductance density
expected by `g_pre`. The source voltage is held at its passive reversal, so
the gate trajectory has a simple analytic reference.

```{literalinclude} ../examples/custom_continuous_synapse.py
:language: python
:start-after: "# region simulation"
:end-before: "# endregion simulation"
```

Call `run_demo()` to initialize the Network and run four timesteps. The
recorder includes the initial frame. Zero-delay connections read the source
at the start of a timestep, before either population advances, so the input
in recorded frame `k+1` equals the weight times release in frame `k`.
The assertions check this ordering, input history, the analytic gate dynamics,
and the current's inward sign below reversal.

The complete continuous example runs here:

```{code-cell} python
:load: ../examples/custom_continuous_synapse.py
:tags: [remove-input]
```

The ordinary Network owns continuous input resets, sampling, reductions, and
delays. These operations share pure tensor cores with the explicit functional
projection and receiver APIs. See
{doc}`advanced/A18_functional_continuous_connectivity` for a complete connected
example, differentiable weights and source values, and checkpoint continuation.
Custom reception uses `continuous_receive_values`; leave the standard reset
and receive adapters in place. Complete functional Network orchestration is
a subsequent step.

## What to check when adapting these examples

Initialize a new mechanism eagerly before compiling it. Check that one known
delivery changes the intended local state or input, that current has the
expected sign and units, and that decay or recovery runs once per timestep.
For differentiable models, also check a gradient against a known expression
or a finite difference. Reuse the exact declared shapes, devices, and dtypes
in returned-value hooks.

For migration from the earlier mutation-based hook API, see
{doc}`upgrading`. Initialization precedence and workspace timing are detailed
in {doc}`advanced/A7_model_initialization` and the
{doc}`mechanism API reference <api/dendra.models.mechanisms>`.
