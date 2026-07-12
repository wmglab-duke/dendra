dendra.models.integrators
=========================

.. automodule:: dendra.models.integrators
   :members:
   :undoc-members:
   :show-inheritance:


Membrane-current reporting
----------------------------

With ``imem=True`` (or ``IMEM=1``), every stable integrator reports
``model.i_membrane`` as absolute outward-positive transmembrane current in mA.
It is the capacitive current plus ionic/mechanism current under that
integrator's per-step discretization. Applied intracellular stimulus and axial
cable currents affect the reported value through the voltage solution, but are
not themselves added to or subtracted from it as separate terms.


Experimental IMEX integrator
----------------------------

.. warning::

   :mod:`dendra.models.integrators.imex` is experimental. It is intentionally
   not re-exported from :mod:`dendra.models.integrators` or :mod:`dendra`.
   Its underscored classes and numerical helpers may change without the
   compatibility guarantees of the stable integrator API.

The current implementation is an exponential time-differencing method that
uses Arnoldi or Lanczos Krylov projections. Importing it directly is an
explicit opt-in to the experimental API. Its per-step membrane-current
discretization is not yet defined; constructing it with ``imem=True`` raises
``NotImplementedError`` rather than exposing a stale or misleading value.

.. autoclass:: dendra.models.integrators.imex._krylov_etd1
