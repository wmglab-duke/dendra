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

Shared membrane assembly
------------------------

Single-compartment implicit Euler (including packed independent compartments),
unbranched implicit Euler, scalar and packed DHS, block tridiagonal Euler, and
block DHS consume the same membrane
linearization frame. An eligible model supplies conductance ``G`` and source
``J`` directly; otherwise the frame retains its authored current ``I`` and the
solver uses its existing ``G*v-I`` arithmetic. Single compartments use the
equivalent direct form ``((C/dt)*v + J + drive)/(C/dt + G)`` only when eligible.

All participating currents must be affine in voltage with stage state fixed,
as inferred by Dendra or asserted with ``Mechanism.AFFINE``. Authors may supply
``i_affine_terms(v) -> (G, J)`` to avoid constructing and canceling ``I``;
existing affine currents and analytic conductance pairs need no extra method.
``dn.ctx(MEMBRANE_ASSEMBLY="off")`` disables this assembly for every current,
``"auto"`` selects it automatically, and ``"validate"`` adds diagnostic
contract checks. The policy is read at evaluation time, not permanently chosen
when a model is built. Validation is not supported inside ``torch.vmap``;
ordinary population batch dimensions remain supported.

Geometry, capacitance, area, solver permutation and extracellular block signs
remain the responsibility of each integrator. ``imem=True`` requires actual
current evaluation, retaining the established reporting discretization.
Explicit Euler/Runge--Kutta, Dufort--Frankel and imposed-voltage current
evaluation retain their existing current paths. See
:doc:`mechanism eligibility <dendra.models.mechanisms>` for selection and
fallback rules.


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
