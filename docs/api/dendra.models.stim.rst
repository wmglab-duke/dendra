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

Extracellular-voltage units
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every extracellular potential that reaches the cable solver is expressed in
**mV**.  The two input forms differ only in where the spatial/temporal product
is formed:

* Direct ``ve`` values supplied to ``Population.step(ve=...)`` or
  ``Population.run(ve=...)`` are already extracellular potentials and must be
  in mV.
* For ``extra=(ve_s, time)``, Dendra multiplies the spatial value ``ve_s`` by
  the temporal waveform value.  Their product must be in mV at every
  compartment and time point.  Multiple contacts are summed after forming
  each contact's product.

Dendra's unit helpers are scalar conversion factors, so the runtime cannot
infer or repair a mismatched normalization.  The analytic point and line
sources return a lead field numerically expressed in mV/mA; pair those fields
with a waveform in mA.  Precomputed scalar fields are intentionally more
general: a field in mV can be paired with a dimensionless relative waveform,
or a field normalized in mV per input unit can be paired with a waveform in
that input unit.  See :doc:`dendra.models.fields` and :doc:`../units` for the
full field and unit contracts.

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
the same temporal type: either Waveforms or time-last tensors.  In every case,
``ve_s * time`` must be an extracellular potential in mV.

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
``[T, *B, N, C]``.  Its values are already potentials in mV; unlike ``extra``,
no temporal scaling is applied.  For an unbatched Population, ``[T]``, ``[T, C]``, and
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

Optimizing waveform timing
--------------------------

``dn.sin`` and ``dn.cos`` support surrogate gradients for ``delay``, ``off``,
and ``off_after``. Rectangular pulses use the same gate for their delays,
pulse widths, and inter-phase intervals. Forward values keep their abrupt
edges; backward differentiation uses a sigmoid approximation with temperature
``tau`` in ms. These gradients describe that approximation, so they need not
match finite differences of the abruptly switched waveform.

For sine and cosine, ``delay`` shifts both the onset and the carrier phase.
The effective stop is ``min(off, delay + off_after)``. Only the earlier stop
constraint receives its edge gradient; equal finite constraints split it
in half. When the relative cutoff wins, changing ``delay`` also moves the
stop. Usually optimize one cutoff and leave the other at its default infinity.

Initialize the cutoff you want to learn to a **finite** value. Infinity means
no cutoff and has zero gradient. Gate contributions to timing gradients are concentrated within a
few ``tau`` of each edge; if ``tau`` is much smaller than your sampling interval,
they can become numerically negligible. ``tau`` controls the gradient width,
while the forward edges remain abrupt.

Enable gradients on the waveform parameters you want to fit, and pass them
to the optimizer explicitly:

.. code-block:: python

   import torch
   import dendra as dn
   from dendra.units import Hz

   waveform = dn.sin(freq=100 * Hz, phase=0.4, delay=2.0,
                     off_after=5.0, tau=0.05)
   waveform.requires_grad_(False)
   waveform.delay.requires_grad_(True)
   waveform.off_after.requires_grad_(True)
   optimizer = torch.optim.Adam(
       [waveform.delay, waveform.off_after], lr=0.01
   )

   t = torch.arange(0.0, 10.0, 0.05)  # ms
   target = dn.sin(freq=100 * Hz, phase=0.4, delay=2.3,
                   off_after=4.5)(t).detach()
   optimizer.zero_grad(set_to_none=True)
   loss = (waveform(t) - target).square().mean()
   loss.backward()
   optimizer.step()

The same waveform can be injected into a model, with the loss computed from
simulation output. Keep its timing parameters in the optimizer when doing so.
To learn an absolute cutoff instead, supply finite ``off``, leave ``off_after``
at infinity, and optimize ``waveform.off``.

For a complete fit from **voltage recordings**, run the
:download:`sinusoidal stimulation example <../../examples/fit_sinusoid_voltage.py>`:

.. code-block:: bash

   python examples/fit_sinusoid_voltage.py --noise-std-mv 0.1 --noise-seed 0 --output-dir voltage-fit
   python examples/fit_sinusoid_voltage.py --membrane hh --noise-std-mv 0.5 --noise-seed 0 --output-dir voltage-hh

This example uses ordinary ``model.run()`` and autograd to fit frequency,
delay, and ``off_after`` from recordings at compartments 0, 2, and 3 of a
five-compartment cable (250 µm long, 2 µm in diameter). The reference and fitted
models share known membrane and cable properties, initial voltage (-65 mV),
amplitude, and phase (0.4 radians). Passive defaults are 0.05 nA,
``dt=0.125`` ms, and ``tau=0.25`` ms. ``--membrane hh`` uses fixed sodium,
potassium, and leak channels at 6.3°C, with 0.2 nA, ``dt=0.0625`` ms, and
``tau=0.5`` ms; the reference includes an action potential.
``--amplitude-na`` changes the known amplitude. JIT is enabled by default;
``--no-jit`` is available for debugging.

``--noise-std-mv`` defaults to zero. Gaussian observation noise is drawn once,
independently at each time and compartment, and reused for all iterations and
starts; ``--noise-seed`` defaults to zero. Fitting and selection of the best
iteration and start use only observed voltage MSE, normalized by observed
response energy. Clean-reference RMSE is evaluated after selection. Saved
arrays and plots include reference and observed voltages, fitted traces,
residuals, and parameter trajectories.

Use ``--cutoff off`` to fit an absolute cutoff. Supply one or more initial
absolute stop times with ``--initial-stop-ms 10.25 12.0``, including when
fitting ``off_after``. Updates bound frequency to 0.001 Hz through the sampling
Nyquist limit, and keep stimulation within the 16 ms recording window for
at least one time step.

Keep ``tau`` wide enough relative to ``dt`` for nearby samples to receive
timing gradients. Forward edges remain abrupt: moving a cutoff within one
sampling interval can leave the voltage trace unchanged. Interpret fitted
cutoffs at that resolution. Multiple starts and a wider surrogate can help
explore other intervals, but optimization can settle in a local minimum.
This example assumes the generating model is known; it does not guarantee
parameter recovery with noise or model mismatch.
Compare with ``--noise-std-mv 0`` before attributing parameter error to noise:
a noiseless fit can also retain parameter error.

Repeated waveforms also support timing gradients for the repetition window:

.. code-block:: python

   train = dn.mono_rect(amp=1.0, pw=0.2).repeat(
       200 * Hz, delay=2.0, off=7.1, tau=0.05
   )
   train.delay.requires_grad_(True)
   train.off.requires_grad_(True)

Here ``train.tau`` controls the repetition window's onset/stop surrogate.
The inner pulse retains its own ``tau`` for its pulse edges. Repetition keeps
its existing period and abrupt cutoff behavior; its ``off`` gradient is
weighted by the repeated signal near the cutoff and may be zero between
pulses. Poisson schedule settings such as ``start`` and ``off`` remain fixed
schedule controls and are not trainable timing parameters.

Optimizing waveform frequency
-----------------------------

Sine and cosine frequencies are expressed in **kHz**, and time is expressed in
**ms**. For optimization, you can instead store the number of cycles over a
fixed reference duration:

.. math::

   q = f T_{\mathrm{ref}}, \qquad f = q / T_{\mathrm{ref}}.

For example, 20 Hz is 0.02 kHz, or ``q = 20`` cycles over a 1000 ms reference
window. This leaves the waveform unchanged and scales its frequency gradient
by ``1 / T_ref``. For an active sine with elapsed time ``t - delay``,

.. math::

   \frac{\partial y}{\partial q}
   = A\,2\pi\frac{t-\mathrm{delay}}{T_{\mathrm{ref}}}
     \cos\!\left(2\pi\frac{q}{T_{\mathrm{ref}}}
                      (t-\mathrm{delay})+\phi\right).

Choose a reference duration comparable to the elapsed time since waveform
onset, and keep it fixed while optimizing. When ``0 <= t - delay <= T_ref``,
the elapsed-time factor is at most one. The loss and optimizer still determine
the size of an update, so select a learning rate for the new coordinate.

PyTorch's ``parametrize`` utility lets an ordinary Dendra waveform expose its
physical ``freq`` while storing the scaled coordinate. The ``right_inverse``
method preserves the starting frequency when registering the parametrization:

.. literalinclude:: ../../examples/scaled_waveform_frequency.py
   :language: python
   :start-at: import math
   :end-before: def main():

Register it before creating the optimizer:

.. code-block:: python

   waveform = dn.sin(freq=20 * Hz).to(dtype=torch.float64)
   transform = FrequencyFromCycles(1000.0).to(waveform.freq)
   parametrize.register_parametrization(waveform, "freq", transform)

   waveform.requires_grad_(False)  # Fit frequency only in this example.
   cycles = waveform.parametrizations.freq.original
   cycles.requires_grad_(True)
   optimizer = torch.optim.Adam([cycles], lr=0.01)

   # waveform.freq is still the physical frequency in kHz.
   values = waveform(torch.linspace(0.0, 1000.0, 1001, dtype=torch.float64))
   # You can also inject this waveform into a model as usual:
   # model[:, 0].inject(waveform)

The same mapping works for ``dn.cos`` and tensor-valued frequencies, including
batched and multitone waveforms. Keep the waveform reference and include
``cycles`` explicitly in your optimizer when fitting stimulation through a
model. To assign a new physical frequency after registration, use
``waveform.freq = new_frequency_tensor``; update ``cycles`` when working in the
scaled coordinate.

Construct the waveform first, then register the parametrization as above.
Passing an expression such as ``dn.sin(freq=cycles / reference_ms)`` instead
creates a new waveform parameter and disconnects the external ``cycles``
tensor from its gradient.

The :download:`complete fitting example <../../examples/scaled_waveform_frequency.py>`
optimizes a sine wave against a target signal. This approach uses ordinary
waveform calls and can also be used with an injected waveform during
``model.run()``. For :doc:`functional Population calls <../advanced/A8_functional_populations>`,
use an ordinary, unparametrized waveform and put ``cycles / reference_ms`` in
the call's parameter dictionary: functional lowering does not currently accept
PyTorch's dynamically parametrized waveform classes.

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
