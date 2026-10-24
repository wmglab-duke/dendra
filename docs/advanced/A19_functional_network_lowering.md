---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-network-lowering)=
# Lower a standard Network

Declare and initialize a `Network` using the ordinary Dendra API, then call
`dn.func.make_functional_network(network)`. The result is a plan for executing
connected timesteps with explicit parameters, constants, and state. Functional
execution preserves the native Network's phase order and supports gradients,
replay, and checkpoint continuation without advancing the source Network.

The complete
{download}`functional_network_lowering.py <../../examples/functional_network_lowering.py>`
example contains a recurrent pair of populations. One connection delivers
threshold-triggered events; the reverse connection samples voltage through a
trainable memoryless release transform. It checks native trajectories,
gradients, functional replay, and native continuation after a Network commit.
Run it with `python examples/functional_network_lowering.py`.

## Declare the ordinary Network

Functional lowering supports initialized Networks and compatible subclasses
whose simulation components use one device. Model subclasses retain their base
family's execution contract; see {doc}`A56_functional_population_subclasses`.
Event connections support dense queues and compact
source-history training, including compatible native `auto` selections, and
packed-history inference with connection schedules, and sparse-calendar
inference. Pending calendar capacity and overflow handling are described in
{doc}`A24_functional_pending_calendar`.
Continuous connections retain native zero/integer-delay semantics.
Configure training and differentiability before lowering. Populations,
mechanisms, and connections must have distinct mutable owners; registering the
same evolving module through multiple Network paths is unsupported. Raw
parameter Tensor ties are allowed and retain shared explicit leaves.

```{literalinclude} ../../examples/functional_network_lowering.py
:language: python
:start-after: "# region standard_network"
:end-before: "# endregion standard_network"
:dedent: 4
```

The event weights are in µS for `expsyn`; the continuous weights are in S/cm²
for the dimensionless release gate received by `graded_syn`. Delays and the
Network timestep are in milliseconds. See {doc}`../units` and
{doc}`../synapse_authoring` for mechanism and weight conventions.

## Lower and execute a timestep

```{literalinclude} ../../examples/functional_network_lowering.py
:language: python
:start-after: "# region lower"
:end-before: "# endregion lower"
:dedent: 4
```

`NetworkTensors` has three fields: `parameters`, `constants`, and `state`.
They are ordinary tensor trees. Component names match those in
`network.populations`, `network.synapses`, and `network.continuous_synapses`.
Lower with `extra={population_name: specification}` to bind extracellular
contacts and Waveforms. Their explicit leaves live under the selected
Population's parameter namespace; see {doc}`A28_functional_network_extra`.

| Tree | Contents |
| --- | --- |
| `parameters["populations"]` | Raw Population parameter mappings. |
| `parameters["synapses"]` | Raw event weight/delay source parameters and schedule packets. |
| `parameters["continuous_synapses"]` | Raw continuous weight/delay source parameters and explicit transform parameters/buffers. |
| `parameters["netstim"]` | NetStim parameter mapping and schedule packet, or `None`. |
| `constants` | Population constants, fixed connection source buffers and initialization values, and optional NetStim calendar constants. |
| `state` | Population carry, event queue/history and continuous queue carry, optional NetStim clock/RNG carry, and the Network clock. |

Extraction creates independent tensor storage and detaches existing native
simulation history. Parameter leaves retain their gradient flags.
Shared registered parameter identities remain shared in the aggregate tree;
optimize each distinct leaf once. Values computed from another parameterization
can be supplied explicitly instead of optimizing the extracted leaves.

```{literalinclude} ../../examples/functional_network_lowering.py
:language: python
:start-after: "# region step"
:end-before: "# endregion step"
:dedent: 8
```

`prepare_and_step(parameters, constants, state, ...)` returns `(next_state, aux)`
without committing to the source Network. The split API is
`prepared = plan.prepare(parameters, constants)` followed by
`plan.step(parameters, prepared, state, ...)`. Recompute preparation after
changing parameters or constants. Pass per-population explicit `inputs`,
`extra`, and `intra` values through the corresponding keyword mappings when
needed, following the functional Population contracts. `inputs` maps names to
`dn.func.StepInput` values; an `intra` entry overrides that Population's
`StepInput.intra`. Registered Population stimulation remains active.
Imperative callbacks are not executed by this timestep.

For a reusable runner with snapshot or fresh restart and recording, use `dn.compile`
as described in {doc}`A21_functional_network_compile`. To compose timesteps
in your own execution loop, compile the pure step directly; the example's
`run_demo(compiled=True)` uses `backend="aot_eager", fullgraph=True`.
For eager rollouts and fixed-horizon compiled chunks, see
{doc}`A20_functional_network_rollout`, including per-population input tapes,
NetStim draw tapes, recording, and checkpointed execution.

## Observe the source at the correct phase

Dendra resolves each connection's registered `pre_var` automatically during
functional execution.
Boolean and integer source fields use the same floating dtype conversion as
the native connection adapters.
It observes the state at the same phase as the native Network:

1. Advance the optional NetStim using the current Network time.
2. Reset every continuous target once.
3. Visit continuous connections in native order.
4. Visit event connections in native order.
5. Advance populations, then publish the next authoritative Network clock.

An all-delayed continuous connection delivers due values before observing its
source. A mixed immediate/delayed connection observes its source before its
combined delivery. Every event connection delivers due values before observing
its source. These distinctions matter in recurrent networks: an earlier
receive can change a field used by the same or a later connection.

Each continuous target resets its per-timestep inputs once, then accumulates
receives in connection order. First-delivery `min`/`max` behavior and true zero
deliveries retain native semantics. A returned Network state is a complete
timestep boundary; partial delivery-phase continuation is currently unsupported.

## Keep NetStim draws explicit

When a Network has a NetStim, call `plan.sample_step_draw(state)` outside
compiled or transformed code and pass the returned packet as `netstim_draw`.
The sampler uses an independent generator initialized from the explicit state;
it mutates neither that state nor the native NetStim. Without a NetStim it
returns `None`.

Keep the draw and its matching post-draw RNG bytes together. Native NetStim
consumes one full draw every step, including deterministic renewal and steps
with no emitted spike. The returned Network state carries the updated clock,
RNG, and schedule cursors. Saving it is sufficient to sample the correct next
packet during continuation. For a fixed supplied draw sequence, the pure
timestep has no implicit RNG dependency. See
{doc}`A14_functional_netstim_renewal` for initialization and sampling details.

## Checkpoints and native continuation

The Network clock contains `t`, `origin`, integer `step`, and
`duration_remainder`. Time is derived from `origin + step * dt`, matching native
execution rather than accumulating floating additions. Population clocks are
aligned at complete timestep boundaries. A one-step call retains the duration
remainder used by native duration-based execution. Clock metadata may reside on
CPU even when the simulation components use an accelerator.

```{literalinclude} ../../examples/functional_network_lowering.py
:language: python
:start-after: "# region checkpoint"
:end-before: "# endregion checkpoint"
:dedent: 12
```

Clone live carry without detaching it when gradients must cross the saved
prefix. Serialize or detach it deliberately for an independent saved run. Keep
the matching parameters, constants, and frozen Network configuration with the
checkpoint. Dense queued payloads retain enqueue-time weights and transform
values. Source-history training reconstructs deliveries from past activity
using current weights and delays; see {doc}`A22_functional_source_history`.

`plan.commit_state_(matching_network, state)` validates and commits a complete
boundary, including the authoritative clock and optional NetStim RNG state.
It does not transfer parameters or constants. Before native continuation, the
receiving Network must have the same effective weights, physical delays,
transform values, schedule values, and Population/NetStim parameters that
produced the saved state. The example constructs matching parameters and can
commit directly without first running the prefix:

```{literalinclude} ../../examples/functional_network_lowering.py
:language: python
:start-after: "# region commit"
:end-before: "# endregion commit"
:dedent: 4
```

Changing wiring, placement, timestep, training mode, source bindings, or static
transform configuration requires lowering again, and may require native
initialization first. Runtime values with compatible schemas can be changed
explicitly. Populations, connections, and transforms must satisfy their
functional purity and eligibility contracts, including restrictions on custom
dispatch, storage, and copy/serialization hooks. See {doc}`A12_functional_event_delivery`,
{doc}`A17_functional_delay_interpolation`, and
{doc}`A18_functional_continuous_connectivity`.

The complete example executes here:

```{code-cell} python
:load: ../../examples/functional_network_lowering.py
:tags: [remove-input]
```

Multi-step execution is available through {doc}`A20_functional_network_rollout`,
and `dn.compile(Network)` execution with snapshot or fresh initialization is
described in {doc}`A21_functional_network_compile`.
Source-history training is described in {doc}`A22_functional_source_history`.
Packed inference is described in {doc}`A23_functional_bitpacked_inference`.
Sparse-calendar inference and packed NetCon scheduled traffic use the shared
pending calendar described in {doc}`A24_functional_pending_calendar`.
Heterogeneous-device functional execution and continuous delay interpolation are
currently unsupported.
