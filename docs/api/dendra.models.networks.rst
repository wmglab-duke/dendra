dendra.models.networks
======================

Connection-weight units
-----------------------

``NetCon`` weights are numerical values in the coordinate expected by their
target synapse; there is no universal weight unit. The built-in point-process
``expsyn`` and ``exp2syn`` mechanisms add incoming weights to conductance state,
so their weights are bare values in µS: ``weight=0.05`` means 0.05 µS. Do not
multiply those weights by ``dendra.units.uS``.

A density-style continuous target defines its own input unit. For example,
built-in ``graded_syn`` receives a dimensionless release gate into ``g_pre``,
so its connection weight and accumulated input are conductance densities in
S/cm². Custom targets must document the units of the state or input changed by
``net_receive`` or ``continuous_receive``. See :doc:`../units` and
:doc:`dendra.models.mechanisms` for the full mechanism contract.

The ``weight`` accepted by ``NetCon.schedule`` and related scheduled-event
methods is different: it is a dimensionless per-event multiplier applied to
the connection's base weight, rather than another target-unit conductance.

Network
-------
.. autoclass:: dendra.models.networks.Network
   :members:

NetStim
-------
.. autoclass:: dendra.models.networks.NetStim
   :members:

Synapse slot targets
--------------------

Network endpoints can use :class:`dendra.models.slice.SynapseSlots` to address
local slots in banked point-process mechanisms. Its canonical API reference is
:doc:`dendra.models.slice`.

NetCon internals
----------------
.. autoclass:: dendra.models.networks.netcon.NetCon
   :members:
   :special-members: __init__
