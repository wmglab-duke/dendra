Numerical integrators
=====================

Integrator factories return a configured integrator class. Pass that class to
model construction, for example ``dn.Cable(..., integrator=dn.rk4())``.
Choose an integrator compatible with the model's spatial topology and
extracellular layers. Functional lowering validates the same numerical family;
unsupported combinations raise a functionalization error.

Explicit methods
----------------

.. autofunction:: dendra.models.integrators.euler
.. autofunction:: dendra.models.integrators.eulerv1
.. autofunction:: dendra.models.integrators.rk1
.. autofunction:: dendra.models.integrators.rk2
.. autofunction:: dendra.models.integrators.rk4
.. autofunction:: dendra.models.integrators.dufort_frankel
.. autofunction:: dendra.models.integrators.dufort_frankel_homogeneous

.. py:function:: dendra.models.integrators.df(**kwargs)

   Alias of :func:`dendra.models.integrators.dufort_frankel`.

.. py:function:: dendra.models.integrators.dfh(**kwargs)

   Alias of :func:`dendra.models.integrators.dufort_frankel_homogeneous`.

Implicit methods
----------------

.. autofunction:: dendra.models.integrators.bwd_euler_sc
.. autofunction:: dendra.models.integrators.bwd_euler_sc_multi
.. autofunction:: dendra.models.integrators.bwd_euler_ub
.. autofunction:: dendra.models.integrators.bwd_euler_bt
.. autofunction:: dendra.models.integrators.dhs
.. autofunction:: dendra.models.integrators.dhs_multi
.. autofunction:: dendra.models.integrators.dhs_bt
.. autofunction:: dendra.models.integrators.dhs_bt_multi

Imposed voltage
---------------

.. autofunction:: dendra.models.integrators.scnv

Membrane-current output
-----------------------

With ``imem=True`` (or ``IMEM=1``), stable integrators report
``model.i_membrane`` as absolute outward-positive transmembrane current in mA:
the capacitive current plus mechanism current under that integrator's
per-step discretization. Intracellular stimulus and axial currents influence
the voltage solution; they are not separately added to the reported current.
LFP callbacks consume this output.

``scnv`` evaluates mechanism currents without a voltage solve. Its reported
membrane current includes the capacitive contribution from the imposed voltage
change. Solver-level ``intra`` and ``ve`` do not drive voltage through this
integrator; a VoltageProcess supplies voltage instead.

Current assembly
----------------

Implicit solvers can assemble affine currents as ``I = G*v - J``. Custom
mechanisms may declare this contract and provide affine terms; see
:doc:`dendra.models.mechanisms`. ``dn.ctx(MEMBRANE_ASSEMBLY="off")`` disables
this optimization, ``"auto"`` selects eligible currents, and ``"validate"``
adds diagnostics. The policy is read when evaluating currents. Validation
inside ``torch.vmap`` is currently unsupported.

Experimental solvers
--------------------

``dendra.models.integrators.imex`` is an explicit opt-in experimental module,
not part of the integrator exports above. Its private implementations may
change without stable-API guarantees; membrane-current output is currently
unsupported.
