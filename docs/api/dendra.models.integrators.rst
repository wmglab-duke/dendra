dendra.models.integrators
=========================

.. automodule:: dendra.models.integrators
   :members:
   :undoc-members:
   :show-inheritance:


Experimental IMEX integrator
----------------------------

.. warning::

   :mod:`dendra.models.integrators.imex` is experimental. It is intentionally
   not re-exported from :mod:`dendra.models.integrators` or :mod:`dendra`.
   Its underscored classes and numerical helpers may change without the
   compatibility guarantees of the stable integrator API.

The current implementation is an exponential time-differencing method that
uses Arnoldi or Lanczos Krylov projections. Importing it directly is an
explicit opt-in to the experimental API.

.. autoclass:: dendra.models.integrators.imex._krylov_etd1
