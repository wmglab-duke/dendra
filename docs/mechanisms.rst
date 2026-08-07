.. _mechanisms:

Mechanisms
==========

Dendra has two explicit membrane-current unit lanes:

.. list-table::
   :widths: 30 25 25 20
   :header-rows: 1

   * - Mechanism type
     - Current method
     - Conductance / ``dI/dV``
     - Voltage
   * - Ordinary distributed ``Mechanism`` (including the default
       ``ContinuousSynapse``)
     - mA/cm², outward-positive
     - S/cm²
     - mV
   * - ``PointProcess``
     - nA, outward-positive
     - µS
     - mV

For a distributed mechanism, ``g * (v - e)`` is in mA/cm² when ``g`` is in
S/cm². Dendra does not multiply a distributed current by area until the voltage
solver needs an absolute current. For a point process, the same expression is
in nA when ``g`` is in µS; Dendra divides by ``1e6 * area_cm2`` to convert it to
a membrane density. Point processes cannot therefore be attached to zero-area
branchpoints.

Built-in ``expsyn`` and ``exp2syn`` weights are bare numerical values in the
point-process µS coordinate: ``weight=0.05`` means 0.05 µS. Do not multiply
these weights by ``dendra.units.uS``. Connection weights for other target
mechanisms inherit the units of the receiving input and must be documented by
that mechanism.

Placement and removal
---------------------

Mechanism placement is structural and shared by every batch replica.
``population.insert(mechanism)`` places a class everywhere, while
``population_region.insert(mechanism)`` places it on the Slice's physical
support.  The inverse operations are:

.. code-block:: python

   population.dendrite.delete(pas)       # remove pas where it is present
   population.soma.delete(pas, strict=True)
   population.delete(pas)                # remove pas everywhere
   population.dendrite.delete_all()      # remove every class in this region
   population.delete_all()               # remove every class everywhere

For example, an active region can be converted to a passive-only region with
``region.delete_all()`` followed by ``region.insert(pas, ...)``.

By default, restricted ``delete`` removes the physical intersection with the
exact class's current support.  Pass ``strict=True`` to require every selected
compartment to host that class; a mismatch then fails atomically.  Deletion
subtracts from every overlapping insertion record and removes every colocated
duplicate/copy slot.

``delete_all`` removes each mechanism class present in the target while
preserving its support outside the target.  The complete multi-class change is
transactional: it fails without changing any class if one affected
parameterization cannot be projected safely.  Whole-population removal also
drops the corresponding persistent Slice parameterizations.

Call ``initialize()`` after changing placement; then recreate optimizers that
referenced the old compiled mechanism Parameters.  Network populations
additionally require clearing and reconnecting synapses with newly acquired
target mechanisms and slot selections.

RANGE insertion values follow each selected region.  A scalar BATCH value is
stable under deletion; partial deletion with a non-scalar BATCH override is
rejected because changing sparse support can change its compiled row groups.
GLOBAL values and mechanism initial conditions belong to the single compiled
mechanism class and must agree across its region records.  Concatenating
component populations with incompatible GLOBAL values for the same exact class
therefore raises instead of silently choosing one; use distinct renamed
mechanism classes when the components genuinely require different GLOBAL
configuration.

See :ref:`slice-contract` for the complete deletion, batching,
parameter-projection, rebuild, optimizer, and network-lifecycle contract.

See :doc:`units` for the complete numerical-unit contract,
:doc:`advanced/A0_mechanisms_and_ions_materials` for mechanism authoring, and
:doc:`api/dendra.models.mechanisms` for the API reference.
