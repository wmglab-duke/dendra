dendra.models
=============

Core
----
.. autoclass:: dendra.models.core.Population
   :members:
   :inherited-members: Module, object
   :member-order: groupwise


Functional Population parameters
--------------------------------

``dn.func.make_functional(model, dt=...)`` lowers either an initialized Population
or a Network. The concrete plan and tensor-bundle types, state layouts and
extraction ownership follow the model: Population raw parameters refer to native
sources, while Network extraction produces independent detached snapshots with
shared aliases. Use ``independent_parameters`` for independent optimization
leaves. Network ``dt`` must match its initialized timestep. See
:doc:`../advanced/A19_functional_network_lowering` for the shared lowering and
binding flow and the compatible ``make_functional_network`` form.

.. autofunction:: dendra.func.make_functional

``dn.func.initialize(functional, tensors, initialization=...)`` initializes the
matching ``PopulationTensors`` or ``NetworkTensors`` bundle and returns a shared
``Continuation``. Use ``dn.func.run(functional, initial, steps=...)`` to execute
it eagerly, or supply a compiled chunk as the target. The default recipe
is ``tensors.initialization`` for a Population and
``functional.initialization_inputs`` for a Network. Explicit
draw packets are replayed; missing packets are sampled from explicit RNG carry
in native order. No accepted timestep runs, and the native model is unchanged.
The returned callback carry is ``None``; callbacks initialize on the first run.
This host entry point follows ambient gradient mode and runs outside
``torch.compile``, ``torch.func`` and forward AD. The existing pure plan methods
and typed initialization results remain available. See
:ref:`functional-explicit-initialization` for ownership, caches, stochastic
replay and complete examples.

``dn.func.initialize(program_or_plan, continuation)`` starts fresh initialization
from the packet's effective parameters, constants and current RNG carry. The
returned packet has no callback carry. Runners and Network plans retain their
default initialization recipes; a Population plan with a continuation requires
explicit ``initialization=``. Runner source overrides cannot be combined with
a continuation. See :ref:`functional-packet-initialization` for source ownership,
native RNG/cache policies and independent optimization trials.

.. autofunction:: dendra.func.initialize

``dn.func.sample_initialization_draws`` captures a canonical
``InitializationDraws`` packet from a compiled runner or a lowered model and
matching tensor bundle. Replay it with ``dn.func.initialize(..., draws=packet)``
or the pure plan's ``initialize`` method. The packet groups ``random``,
``stimulation``, ``netstim`` and ``connections`` draws; for Networks, the first
two fields map Population names to their existing component packets.
Unused families or Population entries use ``None``.

Functional capture also accepts a continuation with the same target and recipe
rules as ``initialize``. It samples from the packet's current RNG carry and
effective sources without advancing or modifying that packet.

Capture executes one ordered host initialization under ``no_grad`` to resolve
sampling around transforms, cache restoration and shared writes. It returns
newly sampled detached base draws and RNG carry, discards initialized state,
and leaves native sources, global RNG and runner defaults unchanged. Supplied
legacy draw fields are captured as provided, retaining Tensor identities and
gradient flags without cloning or detaching them. Replay is a separate
initialization execution and preserves ordinary parameter gradients. Automatic
initialization without a packet samples within its single initialization call.
Supplied canonical draws cannot be combined with legacy stochastic initialization
fields; conflicts are rejected before hooks or sampling. Initialization and
runtime draw packets are distinct types. See :ref:`functional-initialization-draws`
for both target forms, ordering, compatibility and the runnable example.

.. autoclass:: dendra.func.InitializationDraws
   :members: random, stimulation, netstim, connections

.. autofunction:: dendra.func.sample_initialization_draws

``dn.compile(Population)`` returns independent parameter leaves selectable by
exact/unique partial names, glob patterns, or explicit regex selectors. Use
``find_parameters`` to preview matches and ``trainable_parameters`` to inspect
the live ``requires_grad`` subset. See
:ref:`functional-parameter-selection` for matching rules and runnable examples.
The :doc:`functional API overview <../advanced/A57_functional_api_overview>`
provides complete Population and connected-Network workflows.

.. autoclass:: dendra.func.CompiledRunner
   :members: initial_state, initialization_inputs, parameters, named_parameters, trainable_parameters, constants, functional, callbacks, kernel, find_parameters, parameter_name, get_parameter, export_continuation, restore_continuation, execution_report
   :special-members: __call__

.. autoclass:: dendra.func.PopulationTensors
   :members: parameters, trainable_parameters, model_parameters, intra_parameters, extra_parameters, find_parameters, parameter_name, get_parameter, independent_parameters

.. autoclass:: dendra.func.VarInput

.. autoclass:: dendra.func.Inputs
   :members: ve, intra

``Inputs`` carries deterministic voltage/current frames or time-leading tapes.
Networks accept mappings from Population names to the same packet. Stochastic
inputs use ``draws=`` separately. See :ref:`functional-runtime-inputs` for shape,
ownership, sampling, and compatibility contracts.

.. autoclass:: dendra.func.RuntimeDraws
   :members: draws, rng

``RuntimeDraws`` is the canonical runtime stochastic input packet for both
Populations and Networks. Obtain it with ``dn.func.sample_runtime_draws``
and supply it through ``draws=``. Omit ``steps`` and ``duration`` for one frame;
``steps=N`` or ``duration=T`` returns a time-leading tape. Duration sampling
uses the execution timestep and retained fractional duration without advancing
the incoming boundary. Native models supply their current state; functional
targets require explicit state or a ``Continuation``. See
:ref:`shared-runtime-draw-horizons` for elapsed-duration replay and
:ref:`functional-runtime-draws` for sampling, slicing, continuation, and the
compatibility packet forms.

.. autofunction:: dendra.func.sample_runtime_draws


Source inspection
-----------------

``dn.func.inspect_sources(runner)`` inspects the defaults of either runner
returned by ``dn.compile``. Pass a ``Continuation`` to inspect effective
parameter/constant frames from a call or restored checkpoint instead.
``PopulationTensors`` and ``NetworkTensors`` report exactly their stored
frames. Exact raw ``BoundPopulation`` and ``BoundNetwork`` bindings report
their sources, with Network parameter-owned constant readers resolved through
the execution projection. Authored maps remain unchanged. Prepared Population
bindings do not own raw constants and are currently unsupported. Ordered
entries expose live Tensor references, aliases and captured gradient/leaf flags.
``report.trainable_leaves`` contains each qualifying leaf once by identity.
``report.select(queries)`` filters full report paths with exact/unique partial,
glob and regex selectors, retaining complete identity-alias groups, captured
flags and source order. It returns another report with deduplicated optimizer
leaves and performs no Tensor cloning or trainability changes. Use disjoint
selections for independent optimizer groups; separate selections can contain
the same shared Tensor. See :ref:`functional-source-selection` for selector
semantics, multiple optimizers and a runnable example.
State and callback carry are outside this inventory. Inspection neither executes
the model nor proves that a source affects a particular output. See
:ref:`functional-source-inspection` for metadata ownership and runnable examples.

.. autofunction:: dendra.func.inspect_sources

.. autoclass:: dendra.func.SourceReport
   :members: entries, trainable_leaves, select

.. autoclass:: dendra.func.SourceEntry
   :members: path, domain, tensor, aliases, requires_grad, is_leaf


Functional continuation
------------------------

``dn.compile(Population)`` returns a ``CompiledRunner``; a concatenated
``MultiPopulation`` is one Population. ``dn.compile(Network)`` returns a
``CompiledNetworkRunner`` that coordinates its Populations, connections,
optional NetStim and shared clock/RNG scheduling. Both return
``(state, auxiliary)`` and publish ``auxiliary["continuation"]`` for the same
in-memory resume call: ``state, auxiliary = dn.func.run(runner, packet)``.
The direct ``runner(packet)`` form is equivalent.

The packet captures effective raw parameter/constant bindings, including call
overrides and fresh/cache replacements, with complete simulation state and
callback carry. Attributes are fixed; container snapshots are independent
ordinary dictionaries/lists/tuples, and Tensor identities, aliases and gradient
history are retained. Runtime preparation and recording history are separate.
Passing a packet cannot be combined with non-``None`` ``parameters=``,
``constants=``, ``callback_state=`` or ``initialization=`` overrides. Runtime
inputs, draws and declared call-local ``extra=`` remain per-call arguments.

.. autoclass:: dendra.func.Continuation

``dn.func.capture_continuation(target, native_model=None, *, callbacks=None,
callback_state=None)`` creates an owned snapshot from a current initialized
native Population or Network, detaching its model sources and recurrent state.
Targets are lowered functional plans or compiled programs; omission captures
the target's original native model. A supplied model must match the plan's
structure, dtype/device, mode and source sharing. A configured native timestep
must match the plan; an initialized Population can also be captured before its
first native run. Capture uses native
values and trainability, including shared sources, rather than edited program
defaults or trainable selectors. It preserves clocks, pending traffic, retained
materializations and RNG carry without initializing, advancing or sampling a
model. It derives deterministic snapshot baselines from native materializations.
Omitting ``callback_state=`` starts fresh callbacks at the captured boundary.
To retain reductions and recording phase, use the target's prepared collection
for native warmup and pass its result's ``.state`` to capture. An explicit plan
also requires its bound ``callbacks=`` collection when carry is supplied;
runners infer ``program.callbacks`` and reject non-``None`` ``callbacks=``.
The packet owns detached recurrent callback carry, retaining its trainability
flags; suffix recordings contain only new selected samples, without repeating
the boundary. Built-in LFP field carry is rebuilt from the collection's current
bound field sources into owned storage using the ambient gradient mode. Enabled
gradients preserve suffix gradients to those external fields, including after
native-prefix backward; captured model-source and earlier recurrent graphs stay
detached. Externally computed nonleaf field sources retain their upstream graph
lifetime. Capture excludes prior recordings and does not initialize callbacks
or open storage. Same-collection custom callbacks need no portable checkpoint
hook for capture; exporting the packet retains the existing checkpoint
requirements. Portable LFP restoration keeps detached saved field carry; call
``detach_continuation`` with gradients enabled before resuming field training
to refresh fields from the receiving collection. Transfer carry between different
collections through their explicit export/restore bridge before capture. See
:doc:`../advanced/A59_functional_native_handoff` and
:download:`functional_continuation_capture.py <../../examples/functional_continuation_capture.py>`
and :download:`functional_callback_capture.py <../../examples/functional_callback_capture.py>`.

.. autofunction:: dendra.func.capture_continuation

``dn.func.detach_continuation(target, packet, callbacks=...)`` truncates recurrent
gradient history while retaining parameter/constant Tensor identities, aliases,
trainability and upstream graphs. It detaches model state and custom callback
carry and rebuilds built-in LFP field carry from current bound sources, retaining
future field gradients when ambient gradients are enabled. Recurrent tensors
share their incoming storage and preserve view strides; repeated state/carry
Tensor identities remain shared. Source/recurrent identity ties split at the
boundary while raw source objects remain unchanged.
Numerical state, clocks, RNG, reducers, sampling phase,
callback metadata and initialization lifecycle are preserved.

Targets are compiled Population/Network runners and existing functional plans.
Runners use their configured callbacks; omit ``callbacks=``. Explicit plans
require their matching collection when carry is present. Bindings, kernels and
native models are not targets. The helper runs outside compilation, tensor
transforms and forward AD, validates the receiving frames and ownership, and
performs no initialization, sampling, timesteps or recording I/O. Source graphs
computed upstream remain the caller's responsibility; portable export/restore
instead creates independent detached source leaves. See
:ref:`functional-continuation-detach` and
:download:`functional_continuation_detach.py <../../examples/functional_continuation_detach.py>`.

.. autofunction:: dendra.func.detach_continuation

Existing explicit-state/keyword calls retain their contracts. Raw bindings
support ``dn.func.run(bound, packet, ...)`` and successful raw-bound host runs
publish ``auxiliary["continuation"]``. Create the binding with
``packet`` directly, or with ``packet.parameters`` and
``constants=packet.constants``; its source Tensor identities, schemas and aliases
remain authoritative. A packet with different
sources requires an explicit new binding, including after portable restoration.
Prepared Population bindings, direct calls and legacy scheduling keep explicit
state/callback arguments. See :ref:`functional-bound-continuation`.
Save portable detached checkpoints against a lowered Population or Network with
``dn.func.export_continuation(functional, packet, callbacks=callbacks)``. Restore
against a compatible receiving plan with
``dn.func.restore_continuation(receiving, saved, callbacks=receiving_callbacks)``.
Each explicitly supplied callback collection must belong to its respective
plan; omit it for callback-free continuation. Use
``dn.func.run(receiving_kernel, restored, steps=..., callbacks=receiving_callbacks)``
to execute its restored sources and state. For compiled
Population and Network runners, use ``dn.func.export_continuation(runner, packet)``
and ``dn.func.restore_continuation(receiver, saved)``; they select the runners'
configured callbacks. ``runner.export_continuation(packet)`` and
``receiver.restore_continuation(saved)`` are equivalent methods.
The versioned plain data/Tensor format
preserves effective raw sources, tied Tensor identities and trainability.
Restored sources are independent leaves; receiver defaults and RNG are
unchanged, and restoration does not extract native state, initialize, prepare,
sample or execute a model. Same dtype/device and source/state schemas are
required. Packets returned by ``dn.func.initialize`` can be saved before
recording begins; restored packets start the configured callbacks on first
execution. Executed packets require matching callback configuration, including
zero-step calls. The functions and methods reuse the callback
export/restore bridge; the in-memory
``Continuation`` object itself is not the portable format. See
:doc:`../advanced/A57_functional_api_overview` and the complete
:download:`functional_continuation.py <../../examples/functional_continuation.py>`
in-memory example, or :doc:`../advanced/A58_functional_callback_checkpoints`
for disk checkpoints.

.. autofunction:: dendra.func.export_continuation

.. autofunction:: dendra.func.restore_continuation


Functional host scheduling
--------------------------

``dn.func.run(program, ...)`` executes either compiled Population or Network
runner with the same call-level state/continuation, horizon, source, input, draw
and runtime-extra arguments as ``program(...)``. With no state, it applies the
configured snapshot/fresh restart policy. The runner owns callbacks, inference,
initialization, chunk width, checkpoint span and preparation caching; select
these with ``dn.compile``. The explicit scheduling overrides are
not supported for runner targets. See :ref:`functional-runner-execution` for
the complete lifecycle and runnable example.

``dn.func.run(functional_or_kernel, continuation, ...)`` executes exact
``FunctionalPopulation``, ``FunctionalNetwork``, ``CompiledPopulationChunk`` and
``CompiledNetworkChunk`` targets with explicit packet ownership. ``state=packet``
is equivalent to the second positional argument; supply it once. The packet
provides effective parameters, constants, simulation state and callback carry.
Create a fresh packet with ``dn.func.initialize(functional, tensors)`` explicitly.
Ordinary states and tensor bundles are not accepted by this form. It uses the
existing raw binding and scheduler for that invocation, without cloning or
detaching Tensor objects, initializing the model, or retaining a new persistent
source/preparation cache. Source and initialization overrides are not accepted.
Pass ``callbacks=`` explicitly and omit a competing non-``None``
``callback_state=``. Provide steps/duration or tapes determining the horizon;
chunk width does not supply a default horizon. Host spans, checkpoint spans,
inputs, draws and runtime extracellular recipes use the same explicit scheduler
contract as bindings. See :ref:`functional-packet-execution` and the complete
:download:`functional_packet_execution.py <../../examples/functional_packet_execution.py>`
example for initialization, live continuation and portable restoration.

``dn.func.run`` also schedules Population and Network bindings with explicit
steps or duration, host spans, activation checkpointing, and callback continuation.
Both use ``functional.bind(tensors)`` or ``functional.bind(continuation)``, or
the same form on a compiled chunk. The bundle must match the plan's model
family. These forms read only parameter/constant sources; state, callback carry
and initialization inputs are not bound or validated. A binding does not retain
the packet. Supply the intended state or continuation and callbacks explicitly
to execution. Omit additional ``constants``/``prepared`` arguments when binding
a packet, including explicit ``None``. The existing
``bind(parameters, constants=constants)`` form remains available. Raw
``BoundPopulation`` and ``BoundNetwork`` constructors accept the same source
forms. Bindings snapshot mapping containers, retain tensor aliases and gradients,
and expose read-only ``parameters`` and ``constants``.
Independent runs derive fresh training preparation; safe Population inference
preparation is reused until its inputs change. In-place optimizer updates are
visible on the next run. Replacing a mapping entry requires a new binding.
Keep a persistent Population binding when preparation reuse across invocations
matters; direct packet execution binds separately on each call. Compiled kernels
retain no source leaves or preparation graphs in either form.
Constructing a raw binding does not clone or detach Tensors, initialize, sample
or prepare the model. Bindings do not own simulation state. Population's explicit
``bind(parameters, prepared)`` form remains available for graph-local control.
Direct bindings and compiled chunks accept ``callbacks=`` and
``callback_state=`` and return ``(state, auxiliary)``. Bind a collection once
with ``dn.func.make_callbacks(functional, configurations)`` or the compatible
``functional.make_callbacks`` method; its results and continuation carry appear
under ``auxiliary["callbacks"]``.
For raw-bound host scheduling, ``run(bound, packet, ...)`` carries state and
callback memory together and returns the next ``auxiliary["continuation"]``.
Supply the matching callback collection with ``callbacks=``; a non-``None``
``callback_state=`` cannot also accompany a packet. Runtime inputs, draws,
extracellular recipes and scheduling horizons remain per-call arguments.
See :doc:`../advanced/A8_functional_populations` and
:doc:`../advanced/A20_functional_network_rollout` for the execution and
preparation lifetime contracts.

.. autoclass:: dendra.func.BoundPopulation
   :members: functional, parameters, constants, prepared, compiled_chunk, steps
   :special-members: __call__

.. autoclass:: dendra.func.BoundNetwork
   :members: functional, parameters, constants, compiled_chunk, steps
   :special-members: __call__

.. autofunction:: dendra.func.run

.. automethod:: dendra.func.FunctionalPopulation.bind

.. automethod:: dendra.func.CompiledPopulationChunk.bind


Functional callback checkpoints
-------------------------------

.. autofunction:: dendra.func.make_callbacks

``dn.func.make_callbacks(target, configurations, dt=...)`` prepares a named
collection from an initialized native Population/Network or an existing
``FunctionalPopulation``/``FunctionalNetwork``. Native Populations require
``dt``; initialized Networks and existing plans supply their timestep, and an
optional ``dt`` must match it. Existing plan methods remain supported.
Native preparation retains full functional lowering's eligibility and setup
cost; it does not initialize, advance, sample or move the model. Existing-plan
preparation reuses that exact plan without lowering again.

After preparation, a collection can observe compatible initialized native
models at its timestep. The receiver supplies runtime state and must match the
prepared structure, state layout, dtype/device, execution mode and source
sharing. Paths, scopes and selection indices stay fixed. Physical Slice and
region configuration must refer to the exact source model at preparation;
native reuse applies the resolved indices to the receiver. Functional execution
still requires the collection's exact owning plan.
Population execution checks its requested timestep against prepared Recorder
cadence and threshold timing and can configure its integrator for that call.
A Network's initialized timestep and mode must match the prepared plan.

Keep the returned collection across repeated calls: every new collection has
its own live carry identity. Portable export/restore transfers detached carry
into equivalent newly prepared collections. The factory does not accept
runners, bindings, tensor bundles or continuations. Select ``program.functional``
or ``bound.functional`` explicitly when preparing for an existing plan; runner
callbacks remain constructor-owned. See
:ref:`functional-shared-callback-preparation` and
:download:`shared_callback_preparation.py <../../examples/shared_callback_preparation.py>`.

``runner.callbacks`` exposes the bound collection, or ``None`` when callbacks
are omitted. Its ``export_state`` and ``restore_state`` methods provide checked
portable continuation into an equivalent reconstructed collection. Exported
carry is detached and excludes earlier recording results. Direct in-memory
calls retain the existing ``callback_state=`` contract; raw-bound host runs
also accept it as part of a shared continuation packet. See
:doc:`../advanced/A58_functional_callback_checkpoints` for complete Population
and Network examples, ownership, and compatibility rules.

For functional-to-native handoff, commit model sources and state with
``dn.func.commit_continuation_(program, receiver, packet)``, then call the native
receiver with ``callbacks=program.callbacks`` and
``callback_state=program.callbacks.detach_state(packet.callback_state)``.
Callback detachment retains reducer totals and Recorder cadence while cutting
their earlier recurrent gradients. Native model gradients belong to receiving
Parameters; external LFP fields remain configured on the same collection and
refresh under the ambient gradient mode. Custom callbacks need no serialization
protocol for same-collection reuse. Different collections use export/restore.
See :doc:`../advanced/A59_functional_native_handoff` and
:download:`shared_callback_rebinding.py <../../examples/shared_callback_rebinding.py>`.

``dn.Recorder`` configures native and functional recording;
``dn.func.Recorder`` remains supported. Functional binding captures observation
configuration into an independent immutable pure Recorder and retains native
recording history separately. LFP and threshold configurations likewise prepare
independent immutable implementations. ``Recorder`` resolves paths, regions and
selections; ``LFP`` retains raw field Tensor references and projection metadata;
threshold callbacks resolve indices, windows and timestep metadata. These
prepared implementations
expose the pure hooks. ``Active`` uses the ``ActiveAL(at_least=1)``
rule. Changing analysis state belongs to explicit carry, and supplied
configuration objects remain unchanged. Each fresh LFP initialization builds a
new differentiable field stack from its bound Tensor sources. See
:doc:`../advanced/A27_functional_network_callbacks` for timing, result shapes
and Population scopes.

Native ``dn.callbacks.LFP``, ``APCount``, ``Raster``, ``Active``, ``ActiveAL``,
``ActiveALCount`` and ``AnomalyDetector``
configurations are accepted by functional callback collections and ``dn.compile``.
Their ``dn.func`` entry points remain supported. Population-scoped Network
bindings use ``dn.callbacks.PopulationCallback(name, callback)``;
``dn.func.PopulationCallback`` remains compatible for functional execution.
The canonical wrapper also forwards all six native callback hooks to one
Population in ``Network.step``, ``run``, ``longrun`` and
``longrun_checkpointed``. Native results belong to the supplied inner callback.
Functional binding captures supported observation configuration into independent
pure carry, without importing or changing native history. Native execution
through the ordinary ``callbacks=`` hook list requires a native ``Callback``
child. Pure ``FunctionalCallback`` children use the prepared collection in
functional execution or the native pure-callback bridge. Arbitrary imperative
callbacks are not lowered by adding a scope.

Scope names are exact top-level Population keys. Scopes cannot be nested,
including across the two entry points. Recorder selections inside a scope are
relative to that Population, and ``region=`` can select a named physical Slice
inside a concatenated MultiPopulation. Both scope spellings share functional
checkpoint configuration. See :ref:`shared-population-callback-scopes` for the
complete example.

Configuration
capture excludes native history and counters. Authored negative compartment
indices resolve against the receiving Population; an omitted threshold
``dt`` uses its functional timestep, while an explicit value must match.
Both native and functional Raster return only checked post-step samples inside
their window, including a correctly shaped empty result when none are checked.
Raster configuration and carry use the same checkpoint contract for either
constructor.
Native and functional APCount return exact ``torch.int32`` counts, with four
bytes per checked slot. Threshold detection and count updates are
nondifferentiable. Portable restoration accepts valid legacy ``int64`` and
``float32`` count carry after checking that its values are finite, nonnegative
integers within the ``int32`` range; unsafe values are rejected.

Native activity configurations select their configured unpartitioned
``is_active()`` query. ``ActiveAL`` compares the number of selected slots that
fired at least once; ``ActiveALCount`` compares the total crossing events,
including repeats. ``inv=True`` inverts the Boolean result. Native and
``dn.func`` activity constructors require a positive integer ``at_least``.
Booleans, floats, Tensors and strings raise ``TypeError``; zero or negative
integers raise ``ValueError``. Integral scalar types such as NumPy integers
are accepted and normalized to Python integers. Binding and native queries
apply the same validation. Native
``is_active(partition=...)`` and ``numpy(partition=...)``
remain post-run methods. For static grouped functional output, use the
``dn.func`` activity constructors with ``partition=...``. Native constructors
do not accept ``partition``.
See :ref:`functional-shared-activity-configuration` for the runnable example.

Native and functional LFP share a batch-preserving default projection, summing
cell and compartment axes. Explicit ``rule`` strings select exactly that
``torch.einsum`` reduction, including rules that sum batch lanes. Lead fields
are prepared in the observed Population's dtype and device without detaching
their gradients. Authored field aliases remain shared, and public native
``v_unit`` buffer edits made before binding are respected. Functional binding
does not import native recording history. Bind a new collection after changing
the native buffer selection or projection rule. Live continuation retains field
gradients; portable restoration retains detached field carry. See
:ref:`functional-shared-lfp-configuration` for the runnable example.

Recorder cadence selects the initial boundary and accepted-step multiples of
``int(record_dt / solver_dt)`` for both in-memory and HDF5 output. Configure
HDF5 with ``dn.Recorder(...).set_hdf5(path, cache_every=...)``. Host runners
and native pure-callback execution stream its detached samples, omit that
trajectory from callback outputs, and retain numerical sampling carry in the
continuation. Paths, writer sessions and queued samples are excluded from
portable checkpoints. Existing HDF5 layout and ``H5Reader`` access apply.
Direct tensor kernels reject HDF5 configurations. See
:ref:`functional-recorder-storage` for timing, storage bounds, gradients and
file-session continuation.

Custom ``FunctionalCallback`` implementations support portable continuation with
the optional ``checkpoint_configuration()`` hook. It returns a plain dictionary
with a stable namespaced ``kind`` string, positive integer ``version`` and a
plain string-keyed ``fields`` dictionary describing all static configuration.
Nested plain acyclic data and ordinary Tensor/Parameter metadata are supported.
Explicit receiving callbacks must have matching metadata, carry and emission
schemas. Restoration does not import or reconstruct callback code. The hook is
called only during cold checkpoint export/restore; callbacks without it retain
ordinary execution and in-memory continuation. Existing built-in checkpoint
formats remain supported. See :ref:`functional-callback-checkpoint-protocol`
for the full contract and runnable custom reducer/emitter example.

.. autoclass:: dendra.func.FunctionalCallback
   :members: initialize, update, finalize, checkpoint_configuration

.. autoclass:: dendra.func.FunctionalCallbacks
   :members: detach_state, export_state, restore_state

``callbacks.detach_state(state)`` truncates recurrent callback history for that
same collection. It preserves carry values, totals, crossing latches, cadence,
token and emission metadata, and shares custom carry storage/aliases/view
strides. Built-in LFP carry is refreshed from current bound sources; local field
gradients follow the ambient gradient mode. No callback initialization, update,
finalization, sampling, resets or recording I/O are performed. Model state,
connections, parameters and RNG remain unchanged and separately owned.

For native Population or Network training, call ``model.detach()`` and
``callbacks.detach_state(results.state)`` after backward and the optimizer
update. ``Network.detach()`` truncates runtime histories while preserving their
values, raw parameter source identities, clocks and RNG. Its next positive-step,
grad-enabled training call refreshes connection weight/delay graphs from current
sources and retained stochastic draws, without resampling or changing pending
enqueue-time payload values. Initialized quantized delay routes remain fixed
until ``initialize()``; differentiable event delays read current values within
the allocated history. ``detach()`` returns the Network; ``detach_()``
returns ``None``. See :ref:`functional-shared-network-detach` and
:download:`shared_network_detach.py <../../examples/shared_network_detach.py>`.
``dn.func.detach_continuation`` shares these callback rules for full functional
packets. Source expressions and earlier outputs retain their own graphs. See
:ref:`functional-shared-callback-state-detach` and
:download:`shared_callback_state_detach.py <../../examples/shared_callback_state_detach.py>`.

Native ``Population.step``, ``run``, ``longrun`` and ``longrun_checkpointed`` accept
a prepared ``FunctionalCallbacks`` collection through ``callbacks=`` and explicit
continuation through ``callback_state=results.state``. The collection must match
the receiving Population's structure and timestep; prepare it directly with
``dn.func.make_callbacks(model, configurations, dt=...)``. Native model state remains
on the Population. Opted-in ``step``, ``run`` and ``longrun`` return the existing
``FunctionalCallbackResults`` mapping. Without a pure collection, ``step``
returns the model for chaining and ``run``/``longrun`` return ``None``.
Checkpointed execution retains its loss/result tuples.
Ordinary native ``callbacks=[hook, ...]`` lists retain their hook, loss and return
contracts. To combine a native list with pure observations, pass the collection
through ``functional_callbacks=`` and its carry through ``callback_state=``.
``functional_callbacks=`` and ``functional_callback_state=`` remain supported.
Two non-``None`` pure
collection arguments or two non-``None`` carry arguments are rejected.
Native calls accept prepared collections rather than automatically preparing
configuration mappings. The bridge supplies a
gradient-preserving state snapshot and auxiliary ``{"v": ...}``, with pure
updates after clock advancement and before native post-step hooks. HDF5
Recorders use the shared host collector, outside numerical checkpoint replay.
See
:ref:`functional-native-population-callbacks` and
the complete
:download:`native_functional_callbacks.py <../../examples/native_functional_callbacks.py>`
example.

Native ``Network.step``, ``run``, ``longrun`` and ``longrun_checkpointed`` accept
the same ``callbacks`` and ``callback_state`` keywords and compatibility spellings.
Prepare the collection with ``dn.func.make_callbacks(network, configurations)``
from an initialized source Network, or reuse a compatible prepared collection.
The receiving Network's initialized timestep remains
authoritative. Qualified Network
Recorders and Population-scoped pure callbacks keep their functional selections,
cadence and outputs. Unscoped auxiliary data is
``{"populations": {name: {"v": ...}}}``; scoped callbacks receive ``{"v": ...}``.
Native Network state remains on the Network while explicit callback carry
continues its observation phase. Ordinary native hooks can coexist independently.
Opted-in ``step``, ``run`` and ``longrun`` return ``FunctionalCallbackResults``;
default ``step`` keeps its chainable Network return.
Checkpointed execution returns ``(loss_or_none, results)``, or
``(loss_or_none, final_state, results)`` with ``return_final_state=True``.
Callback carry crosses checkpoint boundaries explicitly and emission collection
stays outside replay. With ``restore_state_after_backward=True``, recomputation
preserves the live Network for callback-only gradients as well as native loss
hooks. Native returns and checkpoint formats remain unchanged when the pure
collection is omitted. See :ref:`functional-native-network-callbacks` and
:download:`native_network_functional_callbacks.py <../../examples/native_network_functional_callbacks.py>`.

``loop_hooks`` controls native imperative loop hooks only. Pure initialization
runs when callback carry is omitted; supplied carry resumes sampling phase and
reducer totals. Fresh full-rate recording includes initial and post-step frames;
continued steps emit only new selected frames. ``dn.step``, ``dn.step_population``
and ``dn.step_network`` accept the same callback keywords and retain the same
default/opt-in return conventions. Callback carry can continue across native
step and run-family calls. Portable callback carry uses the existing
``export_state``/``restore_state`` protocol and restores no native model state.
See :ref:`functional-native-step-callbacks` and
:download:`native_step_functional_callbacks.py <../../examples/native_step_functional_callbacks.py>`.

``Recorder.set_hdf5`` supports all native methods above, including scoped
Network recording. Disk histories are omitted from results; sampling carry,
in-memory histories and reducer outputs keep their existing ownership and
gradient contracts. Each call with samples writes a separate ``run_N`` segment
without repeating a continued boundary. ``cache_every`` counts solver iterations
and a fresh initial boundary; recording cadence selects rows separately.
Host storage lifetime does not depend on native ``loop_hooks``. Close the
Recorder before file readback. See :ref:`functional-native-callback-storage` and
:download:`native_callback_storage.py <../../examples/native_callback_storage.py>`.

Intrinsic activity
------------------

.. autofunction:: dendra.models.intrinsic.insert_intrinsic_activity

.. autofunction:: dendra.models.intrinsic.remove_intrinsic_activity

`Population` subclasses
-----------------------
.. autoclass:: dendra.models.core.SingleCompartment
.. autoclass:: dendra.models.core.Cable
   :members:
.. autoclass:: dendra.models.core.Axon
   :members:
.. autoclass:: dendra.models.core.Unmyelinated
   :members:
.. autoclass:: dendra.models.core.Myelinated
   :members:
.. autoclass:: dendra.models.tree.Tree
   :members:

.. autoclass:: dendra.models.extcell.ExtCellAxon
   :members:
.. autoclass:: dendra.models.extcell.ExtCellTree
   :members:


Native morphology construction
------------------------------

Dendra-native :class:`~dendra.models.morphology.Morphology` declarations and
NEURON-authored morphologies both produce the scalar compartment-resistor graph
consumed by :class:`~dendra.models.tree.Tree`, or by the tridiagonal
:class:`~dendra.models.core.Cable` fast path when the graph is one unbranched
material cable. See
:doc:`dendra.models.morphology` for the declaration and canonical graph APIs,
including :func:`dendra.connect_morphologies` for composing complete trees,
and :doc:`../basics/02a_native_morphologies` for the connection contract and
worked examples.


Packed scalar populations
-------------------------

:func:`dendra.concat_models` packs independent scalar systems into one solver
launch without adding electrical edges between them. Ordinary ``Population``
and ``SingleCompartment`` models are represented as one-node trees, ``Tree``
retains its rooted compartment graph, native ``Cable`` retains exact canonical
path edges, and ``Unmyelinated``/``Myelinated`` use live specialized tensor
geometry. These component types may be mixed, then explicitly batched on the
returned model:

.. code-block:: python

   packed = dendra.concat_models(
       {
           "point_cells": point_population,
           "dendrites": tree_population,
           "fibres": axon_population,
       },
       write_back=True,
   )
   packed.batch(32)

All components must initially be unbatched and share a device and dtype.
``write_back=True`` keeps each component's public voltage synchronized after a
step; ``False`` updates only the composite voltage. Finite-extracellular
``ExtCellTree``/``ExtCellAxon`` states require a block multi-solver and are
rejected by this scalar path.

.. autofunction:: dendra.models.multi.concat_models

.. autoclass:: dendra.models.multi.MultiPopulation
   :members:


Finite two-layer extracellular models
-------------------------------------

``ExtCellAxon`` and ``ExtCellTree`` solve two finite extracellular circuit
layers using NEURON's ``extracellular`` conventions. ``xraxial`` is in
MΩ/cm, ``xc`` is in µF/cm², and ``xg`` is in S/cm². The ``extra``/``ve``
input is the bath battery outside the outermost layer (NEURON's
``e_extracellular``), not the first solved extracellular node.

The block state is ordered as ``vc[..., 0] = vi``, ``vc[..., 1] = vext[0]``,
and ``vc[..., 2] = vext[1]``; public membrane voltage is therefore
``v = vi - vext[0]``. ``Myelinated`` instead implements the simpler node-only,
fully insulating-internode abstraction and should not be interpreted as this
double-cable circuit.

With ``IMEM=1``, stable voltage integrators expose absolute outward-positive
transmembrane current in mA as capacitive plus ionic/mechanism current. Axial
and applied intracellular currents influence that value through the voltage
solution but are not included as separate transmembrane-current terms.


Compartment identifiers
-----------------------

``CompartmentID`` describes repeating named regions while defining an axon
model. Register it with :meth:`dendra.models.core.Axon.register_cid` to make
those names available to Dendra's search and slicing APIs and as labelled
attributes on the axon.

.. autoclass:: dendra.models.heterogeneous.CompartmentID
   :members:


Morphology graph helpers
------------------------

These functions calculate weighted distances while treating a directed
morphology graph as undirected, so paths may be measured both toward and away
from the graph root.

.. autofunction:: dendra.models.utils.distance
.. autofunction:: dendra.models.utils.undirected_weighted_lengths


Synapse slot targets
--------------------

:class:`dendra.models.slice.SynapseSlots` explicitly selects local slots in a
banked point-process mechanism. Its canonical API reference is
:doc:`dendra.models.slice`.


Random distributions, runtime noise, and State SDEs
----------------------------------------------------

Runtime ``NOISE`` declarations are detached simulation drives. For stochastic
state dynamics, ``State.DIFFUSION`` can be paired with ``State.METHOD("euler_maruyama")``
for Itô SDEs or ``State.METHOD("euler_heun")`` for Stratonovich SDEs.

The canonical API reference for
:class:`dendra.models.random_parameters.RandomParameterSpec`,
:class:`dendra.models.random_parameters.RuntimeNoiseSpec`,
:class:`dendra.models.random_parameters.DistributionSpec`, and the distribution
registry helpers is :doc:`dendra.models.parametric`.
