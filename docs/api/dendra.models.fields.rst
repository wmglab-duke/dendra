dendra.models.fields
====================

Field normalization and units
-----------------------------

Field modules return a spatial tensor; they do not attach physical-unit
metadata.  When the result is used as ``ve_s`` in
``extra=(ve_s, waveform)``, the product ``ve_s * waveform(t)`` must be an
extracellular potential in mV.  This supports two common representations:

* an absolute spatial potential in mV driven by a dimensionless relative
  waveform; or
* a normalized lead field in mV per input unit driven by a waveform expressed
  in that input unit.

The analytic point and line sources use the second convention.  Coordinates
are in µm, resistivity is in Ω·cm, and their return value is numerically in
mV/mA (equivalently, the potential in mV produced by a 1 mA source).  Therefore
use a temporal waveform in mA.  The product is then mV because
``(mV/mA) * mA = mV``.

The scalar precomputed interpolators preserve the values supplied by the
caller without unit conversion.  Their data may therefore be absolute mV or a
lead field with any declared normalization, provided the temporal waveform
makes the final product mV.  ``PreComputedInterpolate1D.from_ascent`` converts
ASCENT values stored in V to mV, but that representation conversion does not
change the source data's reference-amplitude normalization.

The E-field interpolators expect electric-field vectors in V/m and model
coordinates in µm.  They integrate those vectors along the morphology and
return quasipotentials in mV.  Treat that result as an absolute mV field with a
dimensionless relative waveform, or normalize it to a chosen input amplitude
before pairing it with a dimensional waveform.

See :doc:`dendra.models.stim` for accepted shapes and broadcasting, and
:doc:`../units` for Dendra's broader unit conventions.

Analytic lead fields
--------------------

.. autoclass:: dendra.models.fields.isotropic_point
   :members:
.. autoclass:: dendra.models.fields.anisotropic_point
   :members:
.. autoclass:: dendra.models.fields.line3d
   :members:
.. autoclass:: dendra.models.fields.arbitrary_line
   :members:
.. autoclass:: dendra.models.fields.arc_line
   :members:
.. autoclass:: dendra.models.fields.helix_line
   :members:

Parametric E-fields
-------------------

.. autoclass:: dendra.models.fields.parametric_efield
   :members:

Precomputed
-----------
.. autoclass:: dendra.models.fields.PreComputedInterpolate1D
   :members:
   :exclude-members: forward
.. autoclass:: dendra.models.fields.precomputed_interpolate_1d
   :members:

.. autoclass:: dendra.models.fields.PreComputedInterpolate3DRect
   :members:
   :exclude-members: forward
.. autoclass:: dendra.models.fields.precomputed_interpolate_3d_rect
   :members:

.. autoclass:: dendra.models.fields.PreComputedInterpolate3DScattered
   :members:
   :exclude-members: forward
.. autoclass:: dendra.models.fields.precomputed_interpolate_3d_scattered
   :members:

.. autoclass:: dendra.models.fields.EfieldInterpolate3DRect
   :members:
   :exclude-members: forward
.. autoclass:: dendra.models.fields.efield_interpolate_3d_rect
   :members:

.. autoclass:: dendra.models.fields.EfieldInterpolate3DScattered
   :members:
   :exclude-members: forward
.. autoclass:: dendra.models.fields.efield_interpolate_3d_scattered
   :members:
