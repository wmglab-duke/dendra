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
Populations and Networks. Obtain it with ``functional.sample_runtime_draws``
and supply it through ``draws=``. See
:ref:`functional-runtime-draws` for sampling, slicing, continuation, and the
compatibility packet forms.


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
State and callback carry are outside this inventory. Inspection neither executes
the model nor proves that a source affects a particular output. See
:ref:`functional-source-inspection` for metadata ownership and runnable examples.

.. autofunction:: dendra.func.inspect_sources

.. autoclass:: dendra.func.SourceReport
   :members: entries, trainable_leaves

.. autoclass:: dendra.func.SourceEntry
   :members: path, domain, tensor, aliases, requires_grad, is_leaf


Functional continuation
------------------------

``dn.compile(Population)`` returns a ``CompiledRunner``; a concatenated
``MultiPopulation`` is one Population. ``dn.compile(Network)`` returns a
``CompiledNetworkRunner`` that coordinates its Populations, connections,
optional NetStim and shared clock/RNG scheduling. Both return
``(state, auxiliary)`` and publish ``auxiliary["continuation"]`` for the same
in-memory resume call: ``state, auxiliary = runner(packet)``.

The packet captures effective raw parameter/constant bindings, including call
overrides and fresh/cache replacements, with complete simulation state and
callback carry. Attributes are fixed; container snapshots are independent
ordinary dictionaries/lists/tuples, and Tensor identities, aliases and gradient
history are retained. Runtime preparation and recording history are separate.
Passing a packet cannot be combined with non-``None`` ``parameters=``,
``constants=``, ``callback_state=`` or ``initialization=`` overrides. Runtime
inputs, draws and declared call-local ``extra=`` remain per-call arguments.

.. autoclass:: dendra.func.Continuation

Existing explicit-state/keyword calls retain their contracts. Raw bindings
support ``dn.func.run(bound, packet, ...)`` and successful raw-bound host runs
publish ``auxiliary["continuation"]``. Create the binding with
``packet.parameters`` and ``constants=packet.constants``; its source Tensor
identities, schemas and aliases remain authoritative. A packet with different
sources requires an explicit new binding, including after portable restoration.
Prepared Population bindings, direct calls and legacy scheduling keep explicit
state/callback arguments. See :ref:`functional-bound-continuation`.
Save portable detached checkpoints against a lowered Population or Network with
``dn.func.export_continuation(functional, packet, callbacks=callbacks)``. Restore
against a compatible receiving plan with
``dn.func.restore_continuation(receiving, saved, callbacks=receiving_callbacks)``.
Each explicitly supplied callback collection must belong to its respective
plan; omit it for callback-free continuation. Bind the returned sources with
``receiving_kernel.bind(restored.parameters, constants=restored.constants)``
before using ``dn.func.run``. The same format is supported by
``runner.export_continuation(packet)`` and
``receiver.restore_continuation(saved)``. The versioned plain data/Tensor format
preserves effective raw sources, tied Tensor identities and trainability.
Restored sources are independent leaves; receiver defaults and RNG are
unchanged, and restoration does not extract native state, initialize, prepare,
sample or execute a model. Same dtype/device, source/state schemas and callback
compatibility are required. The functions and methods reuse the callback
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

``dn.func.run`` schedules Population and Network bindings with explicit steps
or duration, host spans, activation checkpointing, and callback continuation.
Both use ``functional.bind(parameters, constants=constants)`` or the same form
on a compiled chunk. Bindings snapshot mapping containers, retain tensor
aliases and gradients, and expose read-only ``parameters`` and ``constants``.
Independent runs derive fresh training preparation; safe Population inference
preparation is reused until its inputs change. In-place optimizer updates are
visible on the next run. Replacing a mapping entry requires a new binding.
Bindings do not own simulation state. Population's explicit
``bind(parameters, prepared)`` form remains available for graph-local control.
Direct bindings and compiled chunks accept ``callbacks=`` and
``callback_state=`` and return ``(state, auxiliary)``. Bind a collection once
with ``functional.make_callbacks``; its results and continuation carry appear
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


Functional callback checkpoints
-------------------------------

``runner.callbacks`` exposes the bound collection, or ``None`` when callbacks
are omitted. Its ``export_state`` and ``restore_state`` methods provide checked
portable continuation into an equivalent reconstructed collection. Exported
carry is detached and excludes earlier recording results. Direct in-memory
calls retain the existing ``callback_state=`` contract; raw-bound host runs
also accept it as part of a shared continuation packet. See
:doc:`../advanced/A58_functional_callback_checkpoints` for complete Population
and Network examples, ownership, and compatibility rules.

.. autoclass:: dendra.func.FunctionalCallbacks
   :members: export_state, restore_state

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
