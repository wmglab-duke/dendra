Recording and analysis
======================

Use :meth:`Recorder.set_hdf5 <dendra.models.callbacks.Recorder.set_hdf5>` to
store long recordings on disk. Record extracellular signals with
:class:`dn.callbacks.LFP <dendra.models.callbacks.LFP>` or count spikes with
:class:`dn.callbacks.APCount <dendra.models.callbacks.APCount>`. Use
:class:`Thresholder <dendra.models.instruments.thresholder.Thresholder>` to
find the stimulus amplitude at which a Population meets an activity criterion.

.. toctree::
   :maxdepth: 1

   dendra.models.callbacks
   dendra.data
   dendra.models.analysis
   dendra.models.instruments
