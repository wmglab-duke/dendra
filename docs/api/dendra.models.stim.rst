Stimulation and waveforms
=========================

Use :meth:`dendra.models.slice.Slice.inject` to attach an intracellular
waveform to selected sites. Supply ``extra=(field, waveform)`` or a list of
such pairs for extracellular stimulation. Intracellular current is in mA;
the extracellular field-waveform product is in mV. Overlapping intracellular
injections add, and each waveform retains its own parameters.
See :doc:`dendra.models.fields` for field normalization and
:doc:`../advanced/inputs_and_randomness` for sampled inputs, Poisson replay,
and runtime-variable extracellular stimulation.

Waveform construction
---------------------

A waveform is a PyTorch module evaluated at times in milliseconds. Temporal
parameters use milliseconds; amplitude units follow the stimulation target.
Arithmetic composes waveforms without collapsing their parameter sources.

.. autoclass:: dendra.models.stim.Waveform
   :members: fn, forward, repeat, poisson, assemble

.. autoclass:: dendra.models.stim.waveform.constant
   :members:

.. autoclass:: dendra.models.stim.waveform.sin
   :members:

.. autoclass:: dendra.models.stim.waveform.cos
   :members:

.. autoclass:: dendra.models.stim.waveform.mono_rect
   :members:

.. autoclass:: dendra.models.stim.waveform.bi_rect
   :members:

.. autoclass:: dendra.models.stim.waveform.bi_rect_balanced
   :members:

.. autoclass:: dendra.models.stim.waveform.bi_rect_symm
   :members:

.. autoclass:: dendra.models.stim.waveform.arbitrary
   :members:

.. autofunction:: dendra.models.stim.waveform.energy

Stimulation handlers
--------------------

The execution methods create these handlers from the model's registered
injections and supplied extracellular configuration. Most users work through
``Slice.inject``, ``extra=`` and :class:`dendra.func.Inputs`.

.. autoclass:: dendra.models.stim.Intra
   :members: init
   :special-members: __call__
   :undoc-members:

.. autoclass:: dendra.models.stim.extra.Extra
   :members: forward
   :undoc-members:
