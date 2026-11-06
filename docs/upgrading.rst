.. _upgrading:

Upgrading Dendra
================

Dendra 0.28: upgrading from 0.27.x
----------------------------------

Dependencies
~~~~~~~~~~~~

Dendra requires PyTorch 2.14 or newer. Choose a PyTorch build appropriate for
your CPU or accelerator; see :doc:`installation`. The optional
``dendra[solvers]`` extra requires ``dendra-solvers>=0.5.1``. Upgrade that
package alongside Dendra if you install the solver backend separately.

Recording and callback results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Recorder`` and ``dn.func.Recorder`` do not accept ``sliding_window``;
``dendra.models.callbacks.sliding_window_average`` is also removed. Apply
temporal filtering to recorded tensors as a separate postprocessing step.
For example, a five-sample average without edge padding is:

.. code-block:: python

   voltage = recorder.stack("v")
   moving_mean = voltage.unfold(0, 5, 1).mean(dim=-1)

The filter preserves any gradients carried by the recorded tensors and returns
four fewer samples. Choose padding explicitly if you need an output at every
original sample. If a native ``Recorder`` call
passed ``partition`` positionally after ``sliding_window``, remove the obsolete
argument and pass ``partition=...`` by keyword.

In-memory native ``Recorder(dt=...)`` uses the same sampling phase as HDF5
recording. A fresh recording includes the initial boundary, then accepted
timesteps ``k, 2*k, ...``, where ``k = int(recording_dt / solver_dt)``. Older
in-memory recordings could have a different phase. Record ``"t"`` alongside
your variables and use those timestamps when comparing runs. Recording cadence
does not change the solver timestep; ``dt=None`` records every accepted step
and the initial boundary.

Callback outputs also have consistent native and functional conventions:

* ``APCount`` uses ``torch.int32`` counters instead of native ``float32`` or
  functional ``int64`` counters. Cast explicitly if subsequent analysis needs
  floating-point tensors. These hard threshold-crossing counts are not
  differentiable spike surrogates.
* ``ActiveAL`` and ``ActiveALCount`` require a positive integer ``at_least``;
  fractional values and booleans are rejected.
* Functional ``Raster`` returns only the rows inside its configured checking
  window, matching native ``Raster``. It does not return a full timeline with
  zeros outside the window. Code that assumed one row per simulated step must
  account for the selected window. Neither path records the initial boundary
  as a spike sample.
* The default ``LFP`` projection preserves batch axes and sums over cells and
  compartments. Its result has shape ``(samples, *batch, contacts)`` in both
  execution modes, including the initial boundary. Lead fields in mV/mA and
  membrane currents in mA produce values in mV. Older native defaults also
  summed batch axes; pass ``rule="...,n...->n"`` explicitly if that reduction
  is intentional.

Supported callback checkpoint restoration validates legacy ``float32`` and
``int64`` count tensors before converting them to ``int32``. Counts must be
finite, non-negative integers within the representable range. This conversion
does not make an otherwise incompatible callback configuration or checkpoint
compatible. See :doc:`advanced/recording_and_callbacks` for configuration,
storage, and continuation examples.

Built-in Hodgkin-Huxley mechanism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The built-in ``hh`` mechanism exposes one total current, ``i(v)``. The separate
``ina(v)``, ``ik(v)``, and ``il(v)`` methods are removed. For component-current
diagnostics on an initialized model with ``hh`` inserted over all compartments:

.. code-block:: python

   channel = model.mech.hh
   v = model.v
   ina = channel.gnabar * channel.m**3 * channel.h * (v - channel.ena)
   ik = channel.gkbar * channel.n**4 * (v - channel.ek)
   il = channel.gl * (v - channel.el)
   total = channel.i(v)

These are outward-positive current densities in mA/cm². Subclasses that changed
the component methods must instead override ``i_affine_terms(v)``, returning
``(G, J)`` for ``i = G*v - J``, and redeclare ``Mechanism.AFFINE("i")``; see
:doc:`advanced/custom_models` for current declarations.

The gates ``m``, ``h``, and ``n`` and their parameters are retained. Their
assigned fields are rate coefficients ``m_a``, ``m_b``, ``h_a``, ``h_b``,
``n_a``, and ``n_b`` for ``x' = x_a - x_b*x``. Replace diagnostic uses of
``minf`` and ``mtau`` with ``m_a / m_b`` and ``1 / m_b`` from an evaluated
assigned-value frame, respectively. Apply the same mapping to ``hinf``/``htau``
and ``ninf``/``ntau``. The rate coefficients include temperature scaling; do
not apply Q10 a second time. The former assigned field names are not retained.

Morphology loading and spatial discretization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Tree.from_swc``, ``Tree.from_asc``, and ``Tree.from_neurolucida`` construct
models through native Morphology declarations. They load geometry, apply
``Morphology.apply_d_lambda``, and pass the resulting declaration to
``Tree.from_morphology``. Their ``d_lambda`` and ``freq`` arguments retain
their meanings: ``d_lambda`` is dimensionless and ``freq`` is a raw frequency
in Hz. Their electrical defaults remain ``rhoa=35.4`` Ω·cm and
``cm=1.0`` µF/cm².

For control over editing and discretization, use the explicit native workflow:

.. code-block:: python

   import dendra as dn

   morphology = dn.Morphology.from_swc("cell.swc", rhoa=35.4, cm=1.0)
   # Inspect or edit the declaration before choosing compartment counts.
   morphology.apply_d_lambda(d_lambda=0.1, freq_hz=100.0)
   tree = dn.Tree.from_morphology(morphology, N=128)

Use ``Morphology.from_asc`` for Neurolucida ASC files in the same workflow.
``Tree.from_asc``, ``Tree.from_neurolucida``, and ``Morphology.from_asc`` require
an explicit ``root`` when a file contains several disconnected cable trees.
Pass an exact generated root name, for example
``dn.Tree.from_asc("reconstruction.asc", root="soma[0]")``. Omitting ``root``
reports the available roots rather than selecting the first and silently
discarding other components.

Standalone Morphology loaders default to ``rhoa=100.0`` rather than the Tree
convenience constructors' ``35.4``; specify ``rhoa`` explicitly to reproduce
the latter's discretization. Their ``nseg`` argument sets a uniform initial
count and does not apply d-lambda. See :ref:`native-morphologies` for editing,
geometry validation, and the discretization contract.

Native Sections have an explicit positive ``nseg`` (default ``1``); loading
geometry alone does not select a frequency-dependent grid. Reapply
``Morphology.apply_d_lambda`` after geometry edits when you want to recompute
compartment counts, then rebuild the Tree from the updated declaration.

SWC imports follow ``Morphology.from_swc`` naming and validation rather than
NEURON's SWC importer. Generated exact names include ``soma_0``,
``basal_dendrite_0``, and ``apical_dendrite_0``. ``Tree.from_swc`` retains the
shared ``soma``, ``axon``, ``dend``, and ``apic`` labels where applicable, along
with the native labels and raw ``swc_type_<id>`` provenance. Code that selects
an exact generated Section name should use the native names. Malformed or
ambiguous SWC topology is rejected according to the native loader's strict
rules rather than repaired silently.

The NEURON-object helpers ``dendra.models.io.lambda_f`` and
``dendra.models.io.apply_d_lambda`` are removed. For native declarations, use
``Section.lambda_f(freq_hz=...)`` to estimate an AC space constant and
``Morphology.apply_d_lambda(d_lambda=..., freq_hz=...)`` to select odd
compartment counts. These methods operate on Dendra's native objects.

``Tree.from_NEURON`` and ``dendra.models.io.neuron_to_dendra_graph`` capture
the source NEURON model's existing grid without changing ``nseg``. Set any
desired NEURON compartment counts before capture. This path remains useful
when NEURON Section objects are the model's source of truth.

The low-level ``io.read_swc``, ``io.read_neurolucida``, and ``io.read_asc``
helpers return a geometry graph and live NEURON ``id2seg`` map without
applying d-lambda. They accept ``file_path`` and keyword-only ``data_func``,
``rhoa``, and ``cm``; remove ``d_lambda`` and ``freq`` arguments from these
calls. Use the native workflow above for discretization. To attach
NEURON-specific segment metadata, use ``data_func`` with a low-level reader
and pass its graph to ``Tree.from_graph``; Tree file constructors do not
accept ``data_func``.

The documentation's ASC reconstruction is named
``docs/basics/L23_PC_cADpyr229_1.asc``. Update references to the former
``docs/basics/example.asc`` path in local copies of the tutorials.

Gradient-clipped Thomas
~~~~~~~~~~~~~~~~~~~~~~~

Remove ``clip_scale_backward`` from ``bwd_euler_ub(...)`` calls. The option is
unsupported and raises ``TypeError``; ordinary Thomas execution uses the
unmodified solve derivatives. The low-level ``thomas_solve_t_gc`` operator is
also unavailable.

Strict state-dictionary loading rejects the obsolete ``integrator.clip_scale``
buffer from a clipped configuration. Reconstruct the model without the removed
option. If migrating a state dictionary, remove that one known obsolete buffer
only after checking that the model configuration and all other keys match;
keep strict loading enabled. Do not discard arbitrary unexpected keys or use
``strict=False`` to hide a schema mismatch.

Native runtime checkpoints store simulation state rather than clipping
configuration. Restore them with ``restore_dict_from_checkpoint`` on a matching
model and check any schema or configuration errors explicitly. They do not
reproduce the former clipped gradients. Models saved as full-object pickles
that reference removed operators, helpers, or constructor options must be
rebuilt. Keep the original environment available to export known parameters
and simulation state before migrating such files. See
:doc:`advanced/initialization_and_checkpoints` for supported checkpoint APIs.

Dendra 0.25: custom Mechanism and State classes
-----------------------------------------------

Dendra 0.25 replaces the former overlapping Mechanism and State lifecycle
APIs with one declaration-driven protocol. This affects authors of custom
``Mechanism`` and ``State`` subclasses. Ordinary Population/Network execution
is unchanged; code that used the removed ``Population.set_value(...)`` helper
must also select the explicit replacement below.

There are no compatibility aliases for the removed declarations and hooks.
Failing at class definition is intentional: it prevents a model from silently
running a hook in the wrong phase or dropping mutable state from checkpoints
and functional execution.

Declaration and hook mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 32 40

   * - Dendra 0.24 and earlier
     - Dendra 0.25
     - Meaning
   * - ``Mechanism.STATE(MyState)``
     - ``Mechanism.STATE_BUNDLE(MyState)``
     - Register a structural ``State`` bundle. ``State.STATE("x")`` continues
       to declare a solver-evolved tensor.
   * - ``BUFFER("x")``
     - ``CARRY``, ``ASSIGNED``, ``DERIVED_BUFFER``, or ``TIMESTEP_BUFFER``
     - Select the declaration from the value's lifetime; see below.
   * - Legacy ``Mechanism.ASSIGNED("x")`` used as mutable storage
     - Select the role from the same four declarations
     - ``Mechanism.ASSIGNED`` has the same pure, ephemeral meaning as
       ``State.ASSIGNED``.
   * - ``Mechanism.SAVE("i")``
     - ``Mechanism.SAVE_CURRENT("i")``
     - Retain a framework-owned mirror of a declared current only. Use
       ``CARRY`` for any author-updated persistent value.
   * - ``Mechanism.INIT(x=value)``
     - ``State.state_defaults(...)`` or insertion-time ``ic={"x": value}``
     - Put a class-owned fallback on its ``State``; use ``ic`` for a particular
       insertion override.
   * - ``State.inf(v)``
     - ``State.state_defaults(v, values)``
     - Supply low-priority initial values for solver state.
   * - ``initial(...)``, ``initial_outputs(...)``, or custom ``State.initialize(...)``
     - ``initial_values(v, values)``
     - Return declaration-owned initial state and carry without mutation.
   * - ``breakpoint(v)``
     - ``assigned_values(v, values)`` or ``advance(v, dt, values)``
     - Separate repeatable current-stage algebra from once-per-step state
       updates.
   * - Custom ``State.solve(...)`` or State/Mechanism ``_advance(...)``
     - ``advance(v, dt, values)`` on the corresponding owner
     - Override a State only when its generated transition from declared
       equations is insufficient.
   * - Custom ordinary-Mechanism ``set_dt(dt)``
     - ``TIMESTEP_BUFFER`` plus ``derive_timestep_buffers(dt)``
     - Declare timestep-dependent prepared workspace explicitly.
   * - ``has_q10``, ``calc_q10()``, or the implicit Q10 cache
     - ``DERIVED_BUFFER`` plus ``derive_buffers()``
     - Make temperature-derived coefficients explicit prepared workspace.
   * - ``method = ...`` or ``method_kwargs = ...`` on a ``State``
     - ``State.METHOD(...)``
     - Declare the integration policy at class definition.
   * - ``Population.set_value(name, value)``
     - The role-specific API
     - Use ``set_v_init`` for initial voltage, parameter assignment for model
       parameters, or a declared pre-initialize transform for tensor setup that
       must also participate in functional initialization.

Replace each former ``BUFFER`` according to how the value is produced and how
long it lives:

* Use ``CARRY`` for counters, latches, histories, and other author-updated
  values that must survive an accepted timestep, checkpoint, or functional
  transition.
* Use ``ASSIGNED`` for pure algebra that can be recomputed from the current
  voltage and value frame. An integrator may evaluate it multiple times in one
  timestep.
* Use ``DERIVED_BUFFER`` for pure prepared workspace determined by parameters,
  temperature, and geometry.
* Use ``TIMESTEP_BUFFER`` when that prepared workspace also depends on ``dt``.

Do not use persistent carry merely to cache repeatable algebra. Conversely, do
not put a counter or latch in ``ASSIGNED``: only ``advance`` is guaranteed to
run once per accepted timestep.

Canonical returned-value hooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a full authoring walkthrough, see the
:doc:`mechanism guide <advanced/mechanisms_and_materials>` and
:doc:`synapse_authoring`. The latter demonstrates the pure synaptic receive
hook alongside ordinary State equations and continuous inputs.

The canonical hooks are:

* ``state_defaults(v, values)`` on ``State`` for fallback solver-state values;
* ``initial_values(v, values)`` on ``State`` or ``Mechanism`` for initialization
  overlays;
* ``assigned_values(v, values)`` on ``State`` or ``Mechanism`` for ephemeral
  algebra;
* ``advance(v, dt, values)`` on ``State`` or ``Mechanism`` for accepted-step
  updates;
* ``derive_buffers()`` and ``derive_timestep_buffers(dt)`` for prepared
  workspaces;
* ``net_receive_values(weights, dt, values)`` on ``Synapse`` for explicit
  event-triggered State and carry replacements. Leave the standard
  ``net_receive`` adapter in place so imperative and functional execution
  share those equations.
* ``continuous_receive_values(value, dt, values, *, input, reduce, mask, received)``
  on ``ContinuousSynapse`` for custom analog reception. Its default implementation
  handles declared ``INPUT`` buffers and the standard reductions. Keep the
  canonical reset/receive adapters for functional delivery; see
  :doc:`advanced/functional_execution`.

Treat all arguments as read-only and return a mapping of Tensors. Dendra checks
the declared names, shapes, dtypes, and devices on the first eager evaluation.
Hooks must not mutate registered or hidden state, consume implicit randomness,
or depend on undeclared mutable globals if they are to participate in
``dendra.func`` transforms.

This is a minimal State and Mechanism using this protocol:

.. code-block:: python

   import torch

   from dendra.models.mechanisms import Mechanism, State


   class Gate(State):
       State.STATE("x")
       State.ASSIGNED("x_inf", "tau")
       State.DERIVATIVE("x' = (x_inf - x) / tau")

       def assigned_values(self, v, values):
           del values
           x_inf = torch.sigmoid((v + 40.0) / 5.0)
           return {"x_inf": x_inf, "tau": torch.ones_like(v)}

       def state_defaults(self, v, values):
           return {"x": self.assigned_values(v, values)["x_inf"]}


   class Channel(Mechanism):
       Mechanism.STATE_BUNDLE(Gate)
       Mechanism.GLOBAL(gbar=0.01, erev=-70.0)
       Mechanism.NONSPECIFIC_CURRENT("i")
       Mechanism.SAVE_CURRENT("i")

       def i(self, v):
           return self.gbar * self.x * (v - self.erev)

       def i_with_conductance(self, v):
           conductance = self.gbar * self.x
           return conductance * (v - self.erev), conductance

``SAVE_CURRENT`` creates the checkpointed ``i_`` mirror. Dendra refreshes it
during current assembly; directly calling ``mechanism.i(v)`` does not update
the mirror.

Material processes
~~~~~~~~~~~~~~~~~~

``MaterialProcess`` uses a separate population-wide scheduler. To share a
custom process between native and functional execution, expose the affected
``(material, field)`` pairs through ``material_fields``, prepare tensor-only
workspace with ``prepare_materials(fields, dt)``, and return replacement fields
from ``material_updates(fields, prepared)``. The inherited
``advance_materials`` adapter publishes those replacements during native
execution.

``configure_process(population)`` and ``set_dt(dt)`` remain available for
persistent configuration and native timestep setup. A process that implements
only a custom mutable ``advance_materials`` does not supply the pure hooks
required for functional lowering. Ordinary local Mechanism roles and lifecycle
hooks, including State bundles, ``CARRY``, and ``advance``, are rejected on a
``MaterialProcess`` subclass. See the
:doc:`mechanism and material guide <advanced/mechanisms_and_materials>` for
process authoring.

Functional compatibility
~~~~~~~~~~~~~~~~~~~~~~~~

The same declared protocol drives imperative execution and ``dendra.func``.
Functional lowering accepts state, dependencies, and effects that Dendra can
represent explicitly; custom implementations should use the shared pure hooks
described above. Optional purity diagnostics can be enabled with
``dn.ctx(CHECK_PURITY=True)`` during lowering; they are disabled by default.
Use an eager initialization or transition while developing a mechanism so
schema errors are reported before wrapping it in ``torch.compile`` or a
``torch.func`` transform. See :doc:`advanced/functional_execution` for supported
components and complete execution examples.
