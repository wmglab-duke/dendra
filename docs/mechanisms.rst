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

See :doc:`units` for the complete numerical-unit contract,
:doc:`advanced/A0_mechanisms_and_ions_materials` for mechanism authoring, and
:doc:`api/dendra.models.mechanisms` for the API reference.
