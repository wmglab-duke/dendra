dendra.models.stim
==================

Stimulation shape contract
--------------------------

The following conventions apply to both standalone
:class:`~dendra.models.core.Population` objects and populations owned by a
:class:`~dendra.models.networks.net.Network`.  Let ``N`` be the population
size, ``C`` the number of compartments, ``T`` the number of evaluated time
points, and ``*B`` the (possibly empty) tuple of explicit batch dimensions.
The full model shape is ``[*B, N, C]``; each call to ``batch(n)`` prepends one
batch dimension.

Time axes
~~~~~~~~~

All :class:`~dendra.models.stim.waveform.core.Waveform` objects use **time as
the last axis**.  Evaluating a scalar waveform on ``T`` times returns ``[T]``;
a waveform with value/sweep axes ``[*V]`` returns ``[*V, T]``.  A temporal
tensor supplied in a standalone Population ``extra=(ve_s, time)`` tuple follows
the same time-last convention.

The one exception is direct, precomputed extracellular voltage passed to
``Population.run(ve=...)``: its time axis is first, so its shape is
``[T, *V]``.  ``Population.step(ve=...)`` accepts one per-step value without a
time axis (or with one leading singleton time axis).

Broadcasting and ambiguous axes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ordinary trailing PyTorch broadcasting is authoritative.  Dendra first tries
to broadcast the non-time value axes directly to the model or selected shape.
Only if that fails, a low-rank value may use the explicit batch axes: its rank
must be no greater than the number of batch axes, it must broadcast to ``*B``,
and Dendra then appends singleton spatial axes.  PyTorch broadcasting
right-aligns a partial value within ``*B``.  For example, for batch shape
``[B0, B1]``, a fallback value ``[B1]`` expands as ``[1, B1]`` before spatial
axes are appended.

For intracellular selections, this fallback is enabled only when the complete
batch grid remains a leading Cartesian prefix, as it does for ordinary model
and labeled slices.  Integer, ``None``, boolean, or advanced indexing may
remove, insert, pair, or reorder axes; in those cases use an explicit shape
that succeeds under ordinary trailing broadcasting.

Trailing broadcasting also resolves ambiguous equal sizes.  For a model with
shape ``[3, N, 3]``, a per-step vector ``[3]`` is compartment-aligned because
that ordinary broadcast succeeds.  Use ``[3, 1, 1]`` to state batch intent
explicitly.  The corresponding waveform output shapes are ``[3, T]`` and
``[3, 1, 1, T]``.  Similarly, when the batch size equals ``N``, temporal
``[N, T]`` remains population-aligned and shared over batches; ``[N, 1, T]``
is explicitly batch-aligned.  Singleton axes are the portable way to
disambiguate batch and spatial intent.

Intracellular input
~~~~~~~~~~~~~~~~~~~

``slice.inject(waveform)`` evaluates the waveform with time last, removes the
last axis one step at a time, and broadcasts each sample to the **selected**
model shape ``S``.  A sample is valid when it is:

* scalar;
* exactly ``S``;
* compatible with ``S`` under ordinary trailing PyTorch broadcasting; or
* after trailing broadcasting fails, low-rank data that broadcasts to the
  selected explicit batch shape and is shared over selected spatial axes.

For an unbatched selection ``S=[Ns, Cs]``, the usual output shapes are ``[T]``
(shared), ``[Cs, T]`` (compartment-aligned), ``[Ns, 1, T]`` (cell-aligned),
and ``[Ns, Cs, T]`` (fully specified).  There is no batch fallback for an
unbatched selection.

For a batched selection ``S=[*B, Ns, Cs]``:

* the same unbatched forms remain valid and keep their ordinary trailing
  meaning, shared over batch dimensions;
* low-rank ``[*V, T]`` may vary over right-aligned batch axes and be shared
  spatially, but only when its ordinary trailing broadcast fails;
* ``[*B, 1, 1, T]`` states per-batch, spatially shared intent explicitly; and
* ``[*B, Ns, Cs, T]`` can vary at every selected location.

Injections may be registered before or after ``Population.batch()`` or
``Network.batch()``.  Existing injection indices are rebased when batching.
The same rules apply to Population and Network execution through ``step``,
``run``, ``longrun``, and ``longrun_checkpointed``.

For example, this selected soma has spatial shape ``[1, 1]``.  A frequency
vector produces ``[B, T]``; its per-step ``[B]`` value cannot broadcast to the
trailing ``[1, 1]`` axes, so the batch fallback drives one frequency per
network replica:

.. code-block:: python

   B = 9
   network = build_windup_network(celsius=36.0, dt_ms=0.0125).batch(B)
   frequencies = torch.linspace(10, 90, B) * Hz
   stimulus = dn.mono_rect(amp=1.0 * nA, pw=1 * ms).repeat(frequencies)
   network.SG.slice("soma", loc=0.5).inject(stimulus)

Extracellular ``extra`` input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A standalone Population accepts ``extra=(ve_s, time)``.  It also accepts a
sequence of these tuples for multiple contacts.  A Network accepts a mapping
``extra={population_name: (ve_s, waveform)}``; Network temporal values must be
Waveform objects.  In a multi-contact Population call, all contacts must use
the same temporal type: either Waveforms or time-last tensors.

The spatial field ``ve_s`` broadcasts to ``[*B, N, C]``:

* for a regular Population, use ``[C]``, ``[1, C]``, or ``[N, C]``;
* after ``batch()``, those forms keep their trailing meaning and are shared by
  all replicas;
* low-rank ``[*V]`` may fall back to right-aligned batch-only broadcasting when
  trailing broadcasting fails;
* ``[*B, 1, 1]`` explicitly supplies one scalar field factor per replica; and
* ``[*B, N, C]`` supplies a replica-specific field.

The temporal value broadcasts to ``[*B, N, T]`` and always has time last:

* for a regular Population, use ``[T]``, ``[1, T]``, or ``[N, T]``;
* after ``batch()``, those forms keep their trailing meaning and are shared by
  all replicas;
* low-rank ``[*V, T]`` may fall back to right-aligned batch-only broadcasting
  when its leading axes cannot broadcast to the population axis;
* ``[*B, 1, T]`` explicitly supplies a per-replica waveform shared over the
  population; and
* ``[*B, N, T]`` may vary by replica and cell.

Spatial and temporal inputs may independently be shared or batch-specific.
For example:

.. code-block:: python

   # One spatial field, one stimulation frequency per network replica.
   field = torch.ones(network.SG.np, network.SG.nc)
   # [B, 1, T] makes the batch axis explicit even if B happens to equal N.
   sweep = dn.mono_rect(amp=1.0, pw=1 * ms).repeat(frequencies[:, None])
   network.run(1 * s, extra={"SG": (field, sweep)})

Direct precomputed ``ve``
~~~~~~~~~~~~~~~~~~~~~~~~~

``Population.run(ve=...)`` normalizes a **time-first** tensor to
``[T, *B, N, C]``.  For an unbatched Population, ``[T]``, ``[T, C]``, and
``[T, N, C]`` are shared, compartment-specific, and fully specified forms.
After ``batch()``, Dendra applies the same trailing-first rule to the per-step
payload after the leading time axis.  The unbatched forms therefore stay shared
over replicas.  A low-rank ``[T, *V]`` payload may use the batch-only fallback
when trailing broadcasting fails; ``[T, *B, 1, 1]`` makes that intent explicit,
and ``[T, *B, N, C]`` is fully specified.  Networks do not expose a direct
``ve`` argument; use the population-specific ``extra`` mapping instead.

.. code-block:: python

   # T timesteps, B replicas, explicit singleton cell/compartment axes.
   ve = torch.randn(T, B, 1, 1)
   population.run(ve=ve, dt=0.025)

Base Class
----------
.. autoclass:: dendra.models.stim.waveform.core.Waveform
   :members:
   :exclude-members: forward

Implementations
---------------
.. autoclass:: dendra.models.stim.waveform.implementations.sin
   :members:
   :exclude-members: fn
.. autoclass:: dendra.models.stim.waveform.implementations.cos
   :members:
   :exclude-members: fn
.. autoclass:: dendra.models.stim.waveform.implementations.mono_rect
   :members:
   :exclude-members: fn
.. autoclass:: dendra.models.stim.waveform.implementations.bi_rect
   :members:
   :exclude-members: fn
.. autoclass:: dendra.models.stim.waveform.implementations.bi_rect_balanced
   :members:
   :exclude-members: fn
.. autoclass:: dendra.models.stim.waveform.implementations.bi_rect_symm
   :members:
   :exclude-members: fn
.. autoclass:: dendra.models.stim.waveform.implementations.arbitrary
   :members:
   :exclude-members: fn
