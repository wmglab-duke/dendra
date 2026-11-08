Mechanisms and stimulation
==========================

Insert :class:`dn.mod.hh <dendra.models.mod.hh>` for Hodgkin–Huxley dynamics or
subclass :class:`Mechanism <dendra.models.mechanisms.Mechanism>` for your own
state equations. Use :meth:`Slice.inject <dendra.models.slice.Slice.inject>`
with :class:`dn.mono_rect <dendra.models.stim.waveform.mono_rect>` for an
intracellular current pulse, or use
:class:`dn.isotropic_point <dendra.models.fields.isotropic_point>` to compute a
point-source lead field for extracellular stimulation.

.. toctree::
   :maxdepth: 1

   dendra.models.mechanisms
   dendra.models.mod
   dendra.models.stim
   dendra.models.fields
