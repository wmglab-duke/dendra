dendra.models.mod
=================

Built-in mechanisms. Distributed ``hh`` and ``pas`` use conductance density in
S/cm² and return current density in mA/cm². Point-process ``alphasynapse``,
``alphasynapse_d``, ``expsyn``, and ``exp2syn`` use conductance in µS and return
lumped current in nA before Dendra's area normalization. For the event-driven
point processes, a bare connection ``weight=0.05`` means 0.05 µS; do not
multiply it by ``dendra.units.uS``. See :doc:`../units` for the complete
contract.

.. autoclass:: dendra.models.mod.alphasynapse
.. autoclass:: dendra.models.mod.alphasynapse_d
.. autoclass:: dendra.models.mod.apcount
   :members:
   :exclude-members: initial, breakpoint
.. autoclass:: dendra.models.mod.apcount_d
   :members:
   :exclude-members: initial, breakpoint
.. autoclass:: dendra.models.mod.exp2syn
.. autoclass:: dendra.models.mod.expsyn
.. autoclass:: dendra.models.mod.fire_r
.. autoclass:: dendra.models.mod.fire_r_d
.. autoclass:: dendra.models.mod.fire
.. autoclass:: dendra.models.mod.fire_d
.. autoclass:: dendra.models.mod.hh
.. autoclass:: dendra.models.mod.pas
.. autoclass:: dendra.models.mod.spikedetect

Spike detector usage
--------------------

Use ``spikedetect`` when many outgoing connections share one voltage threshold.
Connect from ``pre_var="mech.spikedetect.spikes"`` with ``threshold=None``
to avoid duplicate thresholding inside every NetCon.

Graded / continuous synapses
----------------------------

.. automodule:: dendra.models.mod.GRADED_SYNAPSE
   :members: sigmoid_release, graded_release_gate, graded_syn
   :exclude-members: __init__, initial, breakpoint, inf, forward
   :show-inheritance:
   :no-inherited-members:
