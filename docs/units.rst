Units used in Dendra
====================

Dendra APIs use documented numerical unit conventions rather than attaching a
runtime dimension to every value. The names in :mod:`dendra.units` are ordinary
scalar conversion factors, not a dimensional-analysis system. The expected unit
therefore depends on the API receiving a value.

Core model conventions
----------------------

.. list-table::
   :widths: 26 22 52
   :header-rows: 1

   * - Quantity
     - Numerical unit
     - Typical uses
   * - Time
     - ms
     - ``dt``, ``tstop``, delays, durations, and mechanism time constants
   * - Frequency
     - kHz (equivalently ``1/ms``)
     - Repetition rates and scheduled stimulation
   * - Voltage
     - mV
     - Membrane, reversal, threshold, and extracellular potentials
   * - Morphology length
     - µm
     - ``diam``, ``dx``, and spatial coordinates
   * - Axial resistivity
     - Ω·cm
     - ``rhoa`` / ``Ra``
   * - Specific membrane capacitance
     - µF/cm²
     - ``cm``
   * - Absolute current
     - mA
     - Intracellular stimuli and public ``model.i_membrane``
   * - Ion concentration
     - mM
     - Built-in intracellular and extracellular ion fields
   * - Spatial diffusivity
     - µm²/ms
     - Material and ion diffusion processes
   * - Temperature
     - °C
     - ``celsius``

Geometry supplied in µm is converted internally where a solver requires
membrane area in cm² or axial length in cm. Public ``model.i_membrane`` is an
absolute, outward-positive transmembrane current in mA; it is not a current
density.

Native :class:`~dendra.models.morphology.Morphology` declarations use these
same morphology units. Stylized ``L``/``diam`` values and pt3d coordinates are
in µm, ``rhoa`` is in Ω·cm, and ``cm`` is in µF/cm². The same conventions apply
to :meth:`~dendra.models.morphology.Section.update`,
:meth:`~dendra.models.morphology.SectionLocation.update`, and
:meth:`~dendra.models.morphology.Morphology.update_section`; in particular, a
location-scoped ``diam`` update is expressed in µm. Compilation retains the
canonical geometry in binary64 before a constructed model deliberately casts
its buffers to the configured model dtype. A compiled material compartment's
``diameter_um`` is its arclength-mean diameter, matching NEURON's segment
convention; exact membrane area, volume, and axial resistance are integrated
independently and are not reconstructed from that representative value.
Consecutive pt3d controls at the same xyz with different diameters represent a
zero-length step: they contribute annular membrane area in µm² but no length,
volume, or axial resistance. Generic unbranched
:class:`~dendra.models.core.Cable` models preserve the compiled edge resistance
in Ω and membrane area in cm² when using the fast tridiagonal solver. See
:ref:`native-morphologies` for the complete section and connection contract.
When :meth:`~dendra.models.morphology.Morphology.to_swc` or
:meth:`~dendra.models.morphology.Morphology.write_swc` exports the authored
centerlines, SWC ``x``, ``y``, ``z``, and radius are likewise written in µm;
the SWC radius is one half of the authored diameter. SWC does not carry
Dendra's electrical units or discretization fields such as ``rhoa``, ``cm``,
or ``nseg``.

:meth:`~dendra.models.morphology.Morphology.from_swc` reads SWC coordinates and
radii in µm and converts each radius to an authored diameter in µm.
:meth:`~dendra.models.morphology.Morphology.from_asc` snapshots NEURON's
normalized Neurolucida centerlines and diameters in µm, preserving
same-coordinate diameter steps. Classic SWC has no faithful representation for
their annular membrane surface, so SWC export rejects a Morphology containing
one. Neither format defines Dendra's ``rhoa``, ``cm``, or ``nseg``; those are
explicit loader arguments. In particular, loader ``nseg`` is a uniform initial
positive integer for the native Section declarations, not a d-lambda policy or
a reinterpretation of geometry samples as compartments.

:meth:`~dendra.models.morphology.Section.lambda_f` returns an AC space constant
in µm from the Section's current geometry, ``rhoa`` in Ω·cm, and ``cm`` in
µF/cm². Its ``freq_hz`` argument is a raw numerical frequency in Hz.
:meth:`~dendra.models.morphology.Morphology.apply_d_lambda` uses the same raw-Hz
argument; its ``d_lambda`` argument is a positive dimensionless fraction, and
it assigns dimensionless odd integer ``nseg`` counts. These APIs intentionally
use the ``_hz`` suffix to distinguish raw hertz from the normal Dendra
frequency coordinate used by waveforms. For example, pass ``freq_hz=100.0``
for 100 Hz; do not pass ``100.0 * Hz``, because the latter evaluates to
``0.1`` in Dendra's kHz coordinate.

Native Morphology visualizations label authored coordinates, diameters,
connection gaps, and distance-profile positions in µm. Centerline linewidth is
only a relative screen-space diameter encoding and must not be interpreted as
a metrically to-scale tube thickness. In contrast,
:meth:`~dendra.models.morphology.Morphology.plot_shape` and
:meth:`~dendra.models.morphology.Morphology.plot_shape_3d` construct tube radii
in morphology data units; ``diameter_scale=1`` preserves the authored physical
ratio between centerline length and diameter.

Distributed membrane-mechanism contract
----------------------------------------

An ordinary :class:`~dendra.models.mechanisms.Mechanism` is a distributed
membrane mechanism. A
:class:`~dendra.models.mechanisms.ContinuousSynapse` follows the same contract
unless it also inherits :class:`~dendra.models.mechanisms.PointProcess`.

.. list-table::
   :widths: 35 25 40
   :header-rows: 1

   * - Mechanism quantity
     - Numerical unit
     - Convention
   * - ``v`` and reversal potentials
     - mV
     - Compartment-local membrane voltage
   * - Declared current methods
     - mA/cm²
     - Outward-positive current density
   * - Conductance or ``dI/dV``
     - S/cm²
     - Conductance density
   * - Conductance-density parameters
     - S/cm²
     - Enter the numerical density directly

These units make the usual Ohmic expression numerically direct:

.. math::

   I_\mathrm{density} = g_\mathrm{density}(V-E),
   \qquad
   \mathrm{S/cm^2}\;\mathrm{mV} = \mathrm{mA/cm^2}.

Dendra keeps the returned current and conductance as densities during mechanism
assembly. The voltage solver applies compartment membrane area when an absolute
current is needed. Ionic currents declared through ``USEION`` use this same
mA/cm² outward-positive contract.

For example, the built-in passive and Hodgkin--Huxley values are:

.. list-table::
   :widths: 34 33 33
   :header-rows: 1

   * - Parameter
     - Dendra input (S/cm²)
     - Equivalent (mS/cm²)
   * - ``pas.g``
     - ``0.001``
     - 1
   * - ``hh.gnabar``
     - ``0.12``
     - 120
   * - ``hh.gkbar``
     - ``0.036``
     - 36
   * - ``hh.gl``
     - ``0.0003``
     - 0.3

The first value in each row is entered directly in Dendra. In particular,
``pas.g=0.001`` means 0.001 S/cm², not 0.001 mS/cm².

Point-process contract
----------------------

A :class:`~dendra.models.mechanisms.PointProcess` is authored in lumped local
coordinates:

.. list-table::
   :widths: 34 24 42
   :header-rows: 1

   * - Point-process quantity
     - Numerical unit
     - Convention
   * - ``v`` and reversal potentials
     - mV
     - Same voltage convention as distributed mechanisms
   * - Current methods
     - nA
     - Lumped, outward-positive current
   * - Conductance or ``dI/dV``
     - µS
     - Lumped conductance
   * - Built-in ``expsyn`` / ``exp2syn`` weights
     - µS
     - Added to conductance state

Here the corresponding identity is ``µS * mV = nA``. Dendra divides a point
process's current and conductance by ``1e6 * area_cm2`` before adding them to
the distributed mA/cm² and S/cm² membrane totals. A point process therefore
cannot be inserted at a zero-area branchpoint.

Point-process nA/µS values are a local coordinate convention. For the built-in
conductance synapses, write ``weight=0.05`` to mean 0.05 µS. Do **not** write
``weight=0.05 * dendra.units.uS``: ``uS`` is an absolute-S conversion factor
and would make the point-process value one million times too small. To express
50 nS in the required µS coordinate using the conversion scalars, use
``50 * nS / uS``.

Connection weights are otherwise target-defined. A custom synapse must document
the units of the input changed by its ``net_receive`` or ``continuous_receive``
method. For the built-in density-style ``graded_syn``, a dimensionless release
gate uses a connection weight and ``g_pre`` in S/cm².

Materials and ions
------------------

Generic :class:`~dendra.models.mechanisms.Material` fields do not have one
universal unit: their model author defines the field's physical meaning and
must document it. Built-in ion concentrations use mM, ion reversal potentials
use mV, and ionic current methods use the distributed mA/cm² contract.

Material-process conventions include:

* diffusion coefficients in µm²/ms and geometry in µm, µm², or µm³ as
  documented by the process;
* clearance and first-order exchange rates in ``1/ms``;
* direct exchange conductance in the chosen geometry/mass-weight units per ms
  (for example µm³/ms for a volume or µm²/ms for a membrane area); and
* additive ``USEMATERIAL(..., source=...)`` buffers as per-step increments in
  the target field's units, not rates. Multiply a rate by ``dt`` explicitly.

The optional ``units`` metadata on a generic material field is descriptive. It
records the model author's declaration for inspection and documentation;
Dendra does not use it to convert or validate arithmetic at runtime.

Extracellular fields and stimulation
------------------------------------

Every extracellular potential reaching a cable solver is in mV. A direct
``ve`` value is already a potential in mV. For ``extra=(spatial, waveform)``,
the product ``spatial * waveform(t)`` must be in mV; Dendra cannot infer or
repair the normalization.

Analytic point and line sources return lead fields numerically in mV/mA (the mV
potential produced by a 1 mA source), so pair them with temporal waveforms in
mA. Precomputed scalar fields retain the caller's normalization: an absolute
mV field takes a dimensionless waveform, while a lead field in mV per input
unit takes a waveform in that input unit. Electric-field interpolators accept
field vectors in V/m and morphology coordinates in µm, and return integrated
quasipotentials in mV.

Extracellular recording uses the reciprocal convention. Public
``model.i_membrane`` values are absolute currents in mA, and
``callbacks.LFP`` expects one reciprocal lead field per contact in mV/mA; their
dot product is the recorded potential in mV.

Finite extracellular / double-cable models
-------------------------------------------

:class:`~dendra.models.extcell.ExtCellAxon` and
:class:`~dendra.models.extcell.ExtCellTree` follow NEURON's ``extracellular``
parameter convention:

* ``xraxial``: MΩ/cm;
* ``xc``: µF/cm²;
* ``xg``: S/cm²; and
* supplied ``extra`` / ``ve`` bath potential: mV.

Using conversion scalars
------------------------

Use :mod:`dendra.units` to convert a value into the simple base coordinate
expected by an API:

.. code-block:: python

    from dendra.units import Hz, V, mm, ms, nA, s

    length = 1.0 * mm       # 1000 µm
    duration = 0.5 * s      # 500 ms
    current = 2.0 * nA      # 2e-6 mA for an intracellular stimulus
    frequency = 20.0 * Hz   # 0.02 kHz = 0.02 / ms
    delay = 0.25 * ms       # 0.25 ms
    voltage = 0.05 * V      # 50 mV

Because the exported names are floats, they do not track or cancel dimensions.
Do not construct a density as ``value * S / cm**2`` or
``value * uF / cm**2``. Instead, pass the numerical value in the compound unit
documented by that API (for example, ``g=0.001`` in S/cm² or ``cm=1.0`` in
µF/cm²).

API-specific exceptions
-----------------------

Parameter-specific documentation overrides the general table. In particular,
waveform frequencies use Dendra's kHz coordinate, but the ``freq`` argument of
``Tree.from_swc`` / ``Tree.from_asc`` and related D-lambda morphology-import
helpers is explicitly in Hz. APIs whose names end in ``_hz`` likewise expect
raw Hz. Electric-field data are in V/m even though integrated extracellular
potentials are in mV. The native ``Morphology.from_swc`` and
``Morphology.from_asc`` loaders do not apply d-lambda: their ``nseg`` argument
is an explicit initial positive integer, and SWC/ASC geometry samples are not
numerical compartments. Call
:meth:`~dendra.models.morphology.Morphology.apply_d_lambda` with, for example,
``d_lambda=0.1`` and ``freq_hz=100.0`` after loading or editing when native
d-lambda selection is desired.

See also
--------

* :doc:`mechanisms`
* :doc:`advanced/A0_mechanisms_and_ions_materials`
* :doc:`api/dendra.models.mechanisms`
* :ref:`native-morphologies`
