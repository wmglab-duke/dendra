(initialization-and-checkpoints)=
# Initialize, continue and save simulations

Choose whether a call starts another trial or continues an existing simulation.
The explicit `Continuation` groups simulation/RNG state, effective parameters,
constants and callback carry, so those pieces travel together.

## Choose the starting state

| Operation | Starting boundary |
| --- | --- |
| `dn.func.run(program)` | Runner's captured snapshot by default. |
| Compile with `initialization="fresh"`, then call without state | Fresh initialization using the selected sources and cache policy. |
| `dn.func.run(program, continuation)` | Continue that packet's state and sources. |
| `dn.func.initialize(program)` | Explicit fresh initialization from runner defaults and its captured incoming RNG state. |
| `dn.func.initialize(program, continuation)` | Fresh initialization from that packet's effective sources and current RNG carry. |

Snapshot restart is useful for repeating an experiment from one known boundary.
Fresh initialization is useful when optimized parameters affect the initial
conditions, distributions or other initialized values. Fresh initialization
follows the native model's initialization order, reseeding and cache rules.

`initialize` returns a continuation without advancing a timestep or starting
callbacks. Its first execution starts fresh recording. Continuing an executed
packet preserves callback phase and accumulated reductions, without duplicating
the joining sample.

For a lower-level plan, initialize its tensor bundle with
`dn.func.initialize(functional, tensors)`. A Population continuation requires an
explicit `initialization=recipe`, because that packet does not store a recipe;
retain `tensors.initialization` from lowering. A Network plan retains its default
`initialization_inputs`. Runners expose their recipe as
`program.initialization_inputs`.

## Start independent optimizer trials

Detaching a continuation ends its recurrent graph; initializing it starts fresh
model and callback state while retaining the effective source ownership and
incoming RNG position. This is useful for independently differentiated trials
without resetting every stochastic stream to the runner's original snapshot.

```{literalinclude} ../../examples/functional_packet_initialization.py
:language: python
:start-after: "# region training"
:end-before: "# endregion training"
:dedent: 4
```

The complete
{download}`functional_packet_initialization.py <../../examples/functional_packet_initialization.py>`
example defines both Population and Network models, shares optimizer leaves and
checks that each trial records from its initialized boundary.

Initialization transforms run on the supplied effective sources. A transform
that replaces a parameter is applied again; initialization does not recover an
earlier pre-transform baseline. If you construct source expressions such as
`raw * gain` outside initialization, recreate them when another independent
backward pass needs a fresh graph. See {doc}`training_and_parameters`.

## Replay initialization randomness

Use `sample_initialization_draws` when comparing initializations or replaying a
specific random realization. It returns an `InitializationDraws` packet rather
than initialized simulation state. Capturing and replaying are separate
initialization executions; use the same sources and deterministic recipe.

The complete
{download}`functional_initialization_draws.py <../../examples/functional_initialization_draws.py>`
example defines a randomized leak and compiles a `program`, then captures and
replays its initialization draws:

```{literalinclude} ../../examples/functional_initialization_draws.py
:language: python
:start-after: "# region draws"
:end-before: "# endregion draws"
:dedent: 4
```

The same sampler accepts an initialized native model. Replay its packet with
`model.initialize(draws=packet)` for a Population or
`network.initialize(dt, draws=packet)` for a Network. Native capture does not
lower the model; explicit native replay does not support forced rebuilds. See the
complete
{download}`shared_initialization_draws.py <../../examples/shared_initialization_draws.py>`
example for cross-interface replay.

For another trial from current functional RNG carry, pass its continuation to
both capture and replay. Missing draws in an automatic initialization are sampled
in native order. Canonical draw packets cannot be combined with legacy
initialization draw fields. Runtime noise and Poisson draws use the separate
`RuntimeDraws` packet described in {doc}`inputs_and_randomness`.

## Reuse a resting-state cache

Prepare a steady-state cache with the native model before compiling. Fresh
initialization can restore a Network-wide cache, an individual Population's
cache, or a mixture of cached and uncached Populations. A Network-wide cache
takes precedence over individual Population caches. Explicit continuation calls
do not reload caches.

Cached state and prepared numerical values are detached snapshots. Subsequent
simulation can remain differentiable, but the cached warmup graph is not part of
that training graph. For delayed connections, cache enough source history for
the largest delay used by your trials; increasing a delay cannot recover history
that was not retained.

RNG streams default to restoring their saved seed and position. Setting
`stream.ignore_rng_on_load(True)` retains the incoming seed and position while
cached numerical values restore. RAND declarations' `resample_on_initialize` and
`reseed_on_initialize` policies also apply. Runtime checkpoint continuation
restores its exact saved RNG boundary independently of those cache policies.
Use `resample_on_initialize=False` for RAND declarations whose cached realization
must be retained; resampling can invalidate a Population steady-state cache.

Cache presence and structure are fixed by the lowered plan. To switch between
cached and uncached initialization, clear the native cache and lower a new plan;
compatible cache data can be replaced through the initialization recipe.
The complete
{download}`functional_network_cache_initialization.py <../../examples/functional_network_cache_initialization.py>`
example demonstrates Network and Population caches, current connection parameters
and recording continuation.

## Save a portable continuation

This complete example saves a Population runner and resumes it using a separately
constructed compatible runner. Network checkpoints use the same functions.

```python
import io
import torch
import dendra as dn
from dendra.models.mod import pas

def make_program():
    model = dn.SingleCompartment(
        N=1, C=1, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    model.insert(pas, g=0.001, e=-70.0)
    model.initialize()
    return dn.compile(
        model, dt=0.025, steps=4,
        callbacks={"trace": dn.Recorder(("v", "t"))}, backend="eager",
    )

with dn.ctx(DEVICE="cpu", DTYPE=torch.float64, JIT=0):
    program = make_program()
    _, first = dn.func.run(program)
    saved = dn.func.export_continuation(program, first["continuation"])
    file = io.BytesIO()  # Use a filesystem path for a persistent save.
    torch.save(saved, file)
    file.seek(0)

    receiver = make_program()
    restored = dn.func.restore_continuation(
        receiver, torch.load(file, weights_only=True)
    )
    state, following = dn.func.run(receiver, restored)
    print(following["callbacks"]["trace"]["t"])
```

Portable saves preserve effective parameters/constants, Tensor aliases,
trainability, numerical/RNG state and compatible callback carry. They omit
upstream autograd graphs, optimizer state, input/draw tapes and recording files.
Restoration creates independent leaves: construct an optimizer from
`dn.func.inspect_sources(restored).trainable_leaves` when resuming training.
Keep the receiver's model, timestep and callback configuration compatible with
the save. Portable LFP carry contains detached field values. Before training a
suffix against the receiver's live lead fields, call
`dn.func.detach_continuation(receiver, restored)` with gradients enabled to
refresh those bindings. Custom callbacks provide a checkpoint configuration hook; see
{doc}`recording_and_callbacks`.

For an in-memory differentiable continuation, pass the original packet directly
to `run`; portable export deliberately cuts the prefix graph. The complete
{download}`functional_continuation_checkpoints.py <../../examples/functional_continuation_checkpoints.py>`
example checks aliases, restored trainability and suffix optimization.

## Move between native and functional execution

`dn.func.capture_continuation(program)` copies the native source model's current
boundary into owned, detached functional sources. Supply a compatible
`native_model=` to capture another model. Prepared callback carry can be supplied
to preserve recording cadence and accumulated analysis results; recording file
history remains host-owned.

To resume natively, use
`dn.func.commit_continuation_(program, native_receiver, continuation)`. It transfers
values into an already initialized compatible receiver while retaining that
receiver's Parameter objects and aliases. The native suffix builds its own graph;
gradients do not cross the detached handoff back into the functional prefix.
Transfer callback carry through compatible prepared collections when continuing
their analysis or recording phase.

The complete
{download}`functional_native_handoff.py <../../examples/functional_native_handoff.py>`
example demonstrates value transfer, preserved receiver ownership and suffix
gradients. The
{download}`shared_api_conformance.py <../../examples/shared_api_conformance.py>`
example joins native, compiled, saved and native execution with continued
recording. For signatures and compatibility checks, see the
{doc}`functional API reference <../api/dendra.func>`.
