---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(functional-network-cache-initialization)=
# Initialize a Network from a steady-state cache

`dn.compile(network, initialization="fresh", ...)` follows the native
Network's cache-aware initialization policy. A Network-global state cache
restores all Populations and saved connection carry. Without a global cache,
each Population restores its own `steady_state` cache when present; other
Populations use ordinary pure fresh initialization as described in
{doc}`A21_functional_network_compile`. Calls with explicit state continue that
state without restoring either kind of cache.

This supports repeated trials that begin from a saved resting state while
using current connection weights, delays, and continuous transforms. It is
different from snapshot restart: a snapshot retains already prepared runtime
carry, while cache restoration can reconstruct pending traffic from saved
presynaptic histories using the current connection parameters.

The complete
{download}`functional_network_cache_initialization.py <../../examples/functional_network_cache_initialization.py>`
example prepares a Network warmup cache, changes a connection's explicit
weights, and compares continued scan and while execution. It also assembles a
Network containing one Population with its own saved resting state and one
uncached Population. It checks detached cache traffic, subsequent connection
gradients, and recording continuation. Run it from the repository root with
`python examples/functional_network_cache_initialization.py`.

## Prepare and capture a native cache

Use the ordinary `Network.steady_state` workflow before compiling:

```{literalinclude} ../../examples/functional_network_cache_initialization.py
:language: python
:start-after: "# region cache"
:end-before: "# endregion cache"
:dedent: 4
```

`steady_state` records detached Population state and connection source
histories during its warmup. Its `max_delay_ms` should cover the largest delay
needed by subsequent trials. Increasing a delay later cannot recover source
samples beyond the retained history. After warmup, initialize the selected
native training or inference runtime before lowering it.

Construct a fresh-policy runner from that initialized Network:

```{literalinclude} ../../examples/functional_network_cache_initialization.py
:language: python
:start-after: "# region compile"
:end-before: "# endregion compile"
:dedent: 4
```

`runner.initialization_inputs.cache` is an independent, detached
`NetworkInitializationCache` with three fields:

| Field | Contents |
| --- | --- |
| `populations` | Saved native Population state dictionaries, including their parameters and buffers. |
| `synapses` | Normalized event-connection cache packets. |
| `continuous_synapses` | Normalized continuous-connection cache packets. |

The lower-level `make_functional_network` captures the same defaults in
`functional.initialization_inputs.cache`. A connection cache alone, without a
Network-global Population cache, does not restore connection carry. An
individual Population's cache restores its own Population; it does not restore
NetCon or NetStim state.

## Use individual Population resting states

Call `Population.steady_state` before assembling or initializing the Network.
Its saved state belongs to that Population, so a Network can combine saved
resting states with uncached fresh starts:

```{literalinclude} ../../examples/functional_network_cache_initialization.py
:language: python
:start-after: "# region population-cache"
:end-before: "# endregion population-cache"
:dedent: 4
```

The individual cache is exposed as
`runner.initialization_inputs.populations[name].cache`. It is an independent,
detached mapping using the saved Population's native state-dictionary names.
The Network-global `runner.initialization_inputs.cache` remains `None`:

```{literalinclude} ../../examples/functional_network_cache_initialization.py
:language: python
:start-after: "# region population-cache-runner"
:end-before: "# endregion population-cache-runner"
:dedent: 4
```

A Network-global cache takes precedence over individual Population caches.
When it is captured, the per-Population `InitializationInput.cache` fields are
`None` and the global packet controls restoration. Without a global cache,
Populations initialize in authored Network order. A cached Population restores
its saved values at its position in that order; an uncached Population runs its
fresh initializer there. Shared parameter writes follow that same order.
Shared runtime buffers between cached and fresh Populations that retain
different initialization materializations are currently unsupported. Use
independent buffers for those Populations.

Standalone `dn.compile(population, initialization="fresh", ...)` also restores
the Population's own `steady_state` cache. Its mapping is exposed directly at
`runner.initialization_inputs.cache`, the `cache` field of
`dn.func.InitializationInput`. See {doc}`A8_functional_populations` for the
standalone initialization API.

## Replace compatible cache inputs

Cache presence and structure are fixed when lowering. A cache-bound plan
requires its cache packet; passing `cache=None` does not switch it to uncached
initialization. To choose that policy, call `network.clear_state_cache()` or
`population.clear_steady_state()` as appropriate, then lower a new plan. To use
another saved cache with the same frozen schemas, replace the corresponding
global or per-Population input cache.
Its Tensor leaves may change values, while Population names, connection names,
packet layouts, shapes, dtypes, and devices must remain compatible. Raw legacy
formats are normalized when the source Network is lowered, rather than
interpreted inside a compiled timestep.
Keep cached parameters and their saved expanded buffers consistent; editing a
raw parameter alone is not a replacement for producing a coherent native
cache.
State dictionaries can contain multiple entries referring to the same storage,
including packed MultiPopulation state and its component views. Restoration
follows native loading order: later writes determine the value at overlapping
locations. Shared storage does not merge separate Parameter gradient identities.
Cache bindings with conjugate or negative view flags, shared-storage views
with different dtypes, internally overlapping copy destinations, and external
views only partly covered by cache writes are currently unsupported for cached
initialization.

Frozen topology fields, index maps, and spatial coordinates must retain their
values, even when an edited Tensor has the same shape. Structural changes
require a new plan. Validate edited packets with an ordinary eager
`FunctionalNetwork.initialize` call before applying `torch.func` transforms
such as `vmap`: value assertions are skipped inside those transforms.
Ordinary `torch.compile` capture retains graph assertions; combining a
transform with compilation still requires this concrete prevalidation.

Population-owned cached initialization currently requires canonical cache
restoration without arbitrary legacy post-initialization hooks. RAND
declarations that resample on initialization invalidate an owned cache and use
fresh initialization, matching native behavior. A Network-global cache takes
precedence and restores its saved samples and RNG state. See
{doc}`A30_functional_random_parameters` for explicit draws and resampling
policies. Structured pure initialization transforms are skipped when their
owner restores a cache, matching native initialization.

## Distinguish cached Population values from current connection values

Restoring a Population cache loads its saved parameters and buffers. It
preserves the cached voltage and mechanism state rather than applying
`v_init`, evaluating initial values, or running fresh initialization
transforms again. Saved geometry and parameterization buffers may replace
functional constants as well as parameters.

Consequently, editing a Population optimizer leaf does not change that
Population's cached forward value. Update the compatible cache or choose an
uncached initializer when those initial values should change. Cache inputs
are detached; gradients do not flow into the warmup or into edited cache
leaves. The saved effective and materialized Population buffers are also
detached. Dynamics using those buffers therefore do not acquire gradients to
the raw Population parameters by restoring the cache. Use uncached fresh
initialization when optimizing Population parameters through their effect on
the dynamics.

Returned raw Population parameters retain the incoming Parameter gradient
identity, matching native state-dictionary loading. A loss that directly uses
those returned raw parameters can differentiate to the supplied parameter
leaves. That identity is separate from the detached materialized buffers
consumed by the cached dynamics.

Connection parameters follow a different rule. Parameter-invariant caches
store raw source histories and rebuild pending traffic using current weights
and delays. Continuous caches storing raw source values also apply the current
stateless transform. This lets a connection optimizer update affect the next
trial without rerunning the warmup:

```{literalinclude} ../../examples/functional_network_cache_initialization.py
:language: python
:start-after: "# region current-weights"
:end-before: "# endregion current-weights"
:dedent: 4
```

Compatibility caches that contain already weighted pending payloads retain
those payloads. Older continuous caches containing transformed source values
do not apply the transform again. Their interpretation follows native cache
restoration; changing current parameters cannot reconstruct information that
the stored format does not contain.

Gradient boundaries also follow each native runtime. Dense pending payloads
rebuilt from detached histories are detached. Compact source-history training
performs its normal current-parameter lookup when delivering later. Subsequent
new connection transitions retain their ordinary parameter gradients.
Checkpoint backward replays
rollout spans without restoring the cache or sampling initialization again.

## Continue with the effective parameters and constants

Fresh cache restoration returns its effective numerical packets in
`aux["parameters"]` and `aux["constants"]`. It leaves the runner's independent
optimizer/default leaves unchanged. Preserve both effective packets during
explicit continuation:

```{literalinclude} ../../examples/functional_network_cache_initialization.py
:language: python
:start-after: "# region continuation"
:end-before: "# endregion continuation"
:dedent: 4
```

Explicit simulation state bypasses initialization. Continuing a cache-bound
fresh runner requires its effective parameter and constant packets; callback
state continues the same recording without repeating its shared boundary.
The lower-level `FunctionalNetwork.initialize` returns the same three packets
as `NetworkTensors` for explicit preparation and execution.

Calls without state restore the captured cache inputs for a new trial. The starting
time comes from `initialization_inputs.t` and defaults to zero, rather than
the cache's warmup time. Population clocks are synchronized to that Network
clock. NetStim initializes after Population and connection restoration;
automatic host sampling follows its configured seed and the retained RNG
carry. Supplied initialization draws and per-step draw tapes remain explicit,
as described in {doc}`A21_functional_network_compile`.

`initialization="snapshot"` and explicit-state calls retain their existing
restart and continuation behavior. They do not restore steady-state caches
as a side effect. Native checkpoint continuation uses the complete returned
runtime boundary, distinct from the parameter-invariant warmup cache; see
{doc}`A19_functional_network_lowering`.

## Training, inference, and validation

Use `execution="scan", inference=False` for training, including checkpointed
rollout and gradients through continued state:

```{literalinclude} ../../examples/functional_network_cache_initialization.py
:language: python
:start-after: "# region gradients"
:end-before: "# endregion gradients"
:dedent: 4
```

While execution is inference-only and uses the same cache initialization
before entering its captured loop. Cache restoration is outside the recurrent
body for both scan and while.

Invalid cache packets and pending-calendar overflow raise before a public
result is returned. Failed calls leave the source Network, supplied packets,
restart snapshot, and previous execution report unchanged. Calendar capacity
remains fixed by `pending_capacity`; use a larger-capacity plan and retry from
the valid input packet when restoration would exceed it. See
{doc}`A24_functional_pending_calendar` for calendar capacity and overflow.

## Run the complete example

```{code-cell} python
:load: ../../examples/functional_network_cache_initialization.py
:tags: [remove-input]
```
