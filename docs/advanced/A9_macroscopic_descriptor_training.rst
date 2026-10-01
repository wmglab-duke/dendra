.. _macroscopic-descriptor-training:

Training with differentiable macroscopic descriptors
=====================================================

Macroscopic descriptors let a model be trained toward an electrophysiological
behavior without supplying a target voltage trajectory. Examples include an
action-potential width, a firing frequency, a chronaxie, or a paired-pulse
recovery ratio. The experimental protocol still defines the behavior: the
caller chooses the stimulus, recording sites, observation window, and hard
event rule. Functions in ``dendra.models.analysis`` turn the resulting voltage
or threshold measurements into graph-connected PyTorch tensors.

A useful training workflow keeps two roles separate:

* The **hard descriptor** is the scientific measurement used to evaluate the
  current model and proposed updates.
* The **differentiable descriptor** has the same forward value where stated,
  or a declared related objective, and supplies a local parameter direction.

Run the complete hard protocol again after an update. A derivative describes
the current local branch: the event identities, recording sites, and adjacent
voltage-sample pairs selected in that replay. It does not promise that a finite
update remains on that branch. Recomputing the descriptor after every accepted
update naturally accommodates a spike crossing moving to another sample
interval or a new event branch becoming active.

Choosing a descriptor
---------------------

The recommended interface depends on the behavior being trained.

.. list-table::
   :header-rows: 1
   :widths: 23 31 46

   * - Goal
     - Differentiable interface
     - Gradient represented
   * - AP half-width or full width
     - ``differentiable_action_potential_width``
     - Derivative of the hard, linearly interpolated crossings on the selected
       spike branch. The recording and return window must contain the event.
   * - Repeated firing frequency or interspike intervals
     - ``differentiable_spike_timing``
     - Derivative of existing interpolated crossing times. It does not provide
       a birth or loss derivative when the hard event count changes.
   * - Activity-dependent slowing
     - ``differentiable_activity_dependent_slowing``
     - Derivative of the hard selected arrival times and their downstream
       velocity calculation for pulses with complete responses.
   * - Time-resolved firing rate
     - ``differentiable_firing_rate_trajectory``
     - Derivative of a caller-scaled causal kernel applied to existing event
       times. Score the tail long enough to observe the requested stopping or
       slowing behavior.
   * - Activation threshold
     - ``select_trace_tangent_threshold_probe`` or
       ``trace_tangent_threshold_proxy``
     - A hard-forward threshold with a voltage-trace projection in backward.
       Validate the selected probe against complete hard threshold searches.
   * - Chronaxie
     - ``chronaxie_from_threshold_proxies``
     - Current-domain Weiss fit through hard-forward threshold proxies at the
       caller's pulse widths.
   * - Paired-pulse recovery
     - ``paired_pulse_recovery_ratio_from_trace_tangents``
     - Quotient derivative through matched conditioned and unconditioned
       hard-forward threshold proxies.
   * - Noise-averaged hard spike count or following fraction
     - ``gaussian_expected_hard_from_transition_roots``
     - Derivative of a declared Gaussian expectation of the complete hard
       descriptor through all supplied signed transition roots.

Frequency-following optimization and voltage-margin directions for creating
or removing spikes remain experimental. See :ref:`event-margin-training` and
:ref:`hard-transition-scans` for their stronger protocol and validation
requirements.

One training iteration
----------------------

The following structure applies to all descriptor families:

1. Initialize the model and run the complete protocol with the parameters at
   the current optimizer state.
2. Keep the recorded voltage connected to the simulator graph. Do not convert
   it to NumPy or detach it before calling the differentiable descriptor.
3. Reduce the descriptor to one scalar task loss and call ``backward()`` or
   ``torch.autograd.grad``.
4. Propose a parameter update. Positive physical parameters are often easier
   to control with :math:`p=p_0\exp(q)`, where the optimizer updates the
   dimensionless log coordinate :math:`q`.
5. Rerun the complete hard protocol at the proposed parameters. Record the
   hard descriptor, event count or identity, and any required collateral
   constraints.
6. Accept the update according to the declared task, then rebuild the graph
   and recompute the local direction. A branch change is diagnostic
   information; the hard task determines whether the update was useful.

For a physical parameter :math:`p=p_0\exp(q)`, convert a gradient in the
stored parameter to the log coordinate with
:math:`\partial L/\partial q=p\,\partial L/\partial p`. If one coordinate
scales several physical coefficients, sum these terms before applying the
optimizer update.

Example: make an HH neuron fire faster
--------------------------------------

The :download:`complete example
<../../examples/macroscopic_descriptor_training.py>` trains Dendra's built-in
single-compartment HH model under a fixed current. It changes two log
coordinates that jointly scale the forward and backward rates of sodium
activation and inactivation. Scaling both rates preserves each gate's
steady-state value while changing its time constant.

The optimization readout is the interpolated span frequency of complete
0-mV upward crossings in a late recording window:

.. math::

   f_{\mathrm{span}}=
   \frac{1000(N-1)}{t_N-t_1}\ \mathrm{Hz},

where :math:`N` is the number of retained crossings and :math:`t_1` and
:math:`t_N` are the first and last interpolated crossing times in
milliseconds. This objective responds to spike-time shifts before a fixed
window count changes. The example nevertheless measures the full hard
``APCount`` after every proposal and rejects a step that reduces it.

Run it from the repository root:

.. code-block:: console

   python examples/macroscopic_descriptor_training.py

A short source-checkout smoke run is available:

.. code-block:: console

   python examples/macroscopic_descriptor_training.py \
       --updates 1 --tstop-ms 60 --objective-start-ms 20 \
       --skip-holdout --no-jit

The short command checks the example environment and normally finishes in a
few seconds. The full default run performs nine differentiable updates plus
hard backtracking and a 400-ms holdout evaluation; it took about two minutes
on one CPU thread in our test.

The core descriptor call is independent of the HH model:

.. code-block:: python

   from dendra.models.analysis import differentiable_spike_timing

   timing = differentiable_spike_timing(
       voltage_mV,                 # shape (time, neuron, compartment)
       dt_ms,
       node_mask=readout_sites,
       V_spk=0.0,
       time_window_ms=(40.0, 160.0),
   )
   if not bool(timing["valid"].all()):
       raise RuntimeError("Each neuron needs at least two retained spikes")

   frequency_hz = timing["span_frequency_hz"]
   loss = (frequency_hz - target_hz).square().mean()
   loss.backward()

``event_times_ms`` and ``interspike_intervals_ms`` are also graph-connected.
Use the individual intervals when a loss should see redistribution among
interior spikes: span frequency depends only on the first and last event while
the count is fixed.

Training AP width, slowing, and rate trajectories
-------------------------------------------------

The trace-based descriptor functions accept the voltage tensor rather than a
model class. A typical AP-width objective is:

.. code-block:: python

   import torch

   from dendra.models.analysis import differentiable_action_potential_width

   width = differentiable_action_potential_width(
       voltage_mV,
       dt_ms,
       node_mask=readout_sites,
       baseline_mean_mode="continuous_time",
       full_width_post_ms=8.0,
   )
   if (
       not bool(torch.isfinite(width["width_ms"]).all())
       or not bool(width["branch_matches_hard"].all())
   ):
       raise RuntimeError("AP width is undefined or its branch does not match")
   loss = (width["width_ms"] - target_width_ms).square().mean()

``differentiable_activity_dependent_slowing`` additionally receives the pulse
times and the response window. ``differentiable_firing_rate_trajectory``
receives a causal-kernel scale ``tau_ms`` and returns a ``(time, neuron)`` rate
tensor. In each case, the caller's site mask and time windows are part of the
measurement protocol and should be saved with the trained model.

Threshold-derived descriptors
-----------------------------

A deterministic activation threshold is discontinuous as a function of a
model parameter. Its ordinary derivative is therefore zero away from a
transition and undefined at the transition. Dendra uses two measurements for
training instead:

* ``Thresholder`` supplies the reported value :math:`T` and a final bracket
  whose lower amplitude is inactive and upper amplitude is active.
* A subthreshold voltage replay at a nearby probe amplitude supplies a local
  direction for moving that boundary.

The :download:`complete threshold and chronaxie example
<../../examples/threshold_descriptor_gradients.py>` implements both parts for
Dendra's built-in myelinated HH model. It differentiates every threshold and
the fitted chronaxie with respect to the logarithm of sodium conductance, then
checks those derivatives with fresh complete hard searches at perturbed
conductances. Run it from the repository root:

.. code-block:: console

   python examples/threshold_descriptor_gradients.py

The model contains one independent population lane per pulse width. All lanes
share the scalar sodium conductance. Batching the widths matters because one
voltage replay then retains every threshold graph until the chronaxie gradient
has been calculated.

Hard search and bracket
~~~~~~~~~~~~~~~~~~~~~~~

The example constructs its complete search as follows. ``Active`` defines the
hard event rule; the field, waveform, checked sites, time step, and recording
horizon are all parts of the protocol.

.. code-block:: python

   from dendra.models.callbacks import Active
   from dendra.models.instruments import Thresholder

   criterion = Active(
       threshold=0.0,
       node_check=monitored_nodes,
       dt=dt_ms,
   )
   thresholder = Thresholder(
       model,
       criterion,
       space=field,
       time=waveform,
       lb=0.0,
       ub=0.2,
       rtol=1e-5,
   )
   upper, lower = thresholder.calculate_thresholds(
       tstop=tstop_ms,
       dt=dt_ms,
       block_possible=True,
   )
   hard_threshold = 0.5 * (lower + upper)

``calculate_thresholds`` returns ``(upper_active, lower_inactive)`` in that
order, and returns both tensors on the CPU. In contrast,
``select_trace_tangent_threshold_probe`` accepts its bracket as
``(inactive, active)``. Keep the names explicit when moving the bounds to the
model's device.

The midpoint is a numerical report of the hard threshold. The bracket width
is its search uncertainty; it is not a differentiable transition width. Check
that a fresh complete run classifies ``lower`` as inactive and ``upper`` as
active before constructing a training direction.

Voltage and amplitude tangent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`V(A,q)` be the recorded voltage for stimulus amplitude :math:`A`
and model parameters :math:`q`. A trace-tangent proxy needs a
graph-connected :math:`V(A_p,q)` at a fixed inactive probe :math:`A_p` and the
amplitude tangent

.. math::

   J_A = \left.\frac{\partial V(A,q)}{\partial A}\right|_{A=A_p}.

Here is the complete ordering used by the example:

.. code-block:: python

   from dendra.models.callbacks import Recorder

   def run_voltage(self, amplitudes):
       recorder = Recorder(["v"], node_indices=self.monitored_nodes)
       self.model.initialize()
       self.model.run(
           extra=(self.field * amplitudes[:, None], self.waveform),
           tstop=self.tstop_ms,
           dt=self.dt_ms,
           callbacks=[recorder],
           progressbar=False,
       )
       return recorder.stack("v")

   def run_voltage_and_amplitude_tangent(
       self,
       probe_amplitudes,
       check_amplitudes,
   ):
       _, amplitude_tangent = torch.autograd.functional.jvp(
           self.run_voltage,
           (probe_amplitudes,),
           (torch.ones_like(probe_amplitudes),),
           create_graph=False,
           strict=True,
       )

       with torch.no_grad():
           check_voltage = self.run_voltage(check_amplitudes).detach().clone()

       # Keep this as the final simulation before using the proxy gradients.
       voltage = self.run_voltage(probe_amplitudes)
       return voltage, amplitude_tangent.detach(), check_voltage

``run_voltage`` initializes the model, resets its recorder, runs the complete
stimulus, and returns a ``(time, lane, site)`` voltage tensor. The vector of
ones gives one amplitude derivative per lane only because the population lanes
are independent. For a coupled population, calculate the required Jacobian
columns separately.

The amplitude tangent is deliberately detached. It defines the fixed local
projection used for this training iteration; the reverse pass differentiates
the final voltage replay with respect to every connected model parameter and
does not require simulator second derivatives. The detached check replay tests
the local amplitude linearization. Dendra's imperative simulator reuses
internal state, so the graph-bearing voltage replay runs last and its gradients
are consumed before another simulation. Batch compatible pulse widths as in
the example, or calculate and consume each width's parameter vector-Jacobian
product sequentially.

An exact amplitude JVP is convenient but not required by the analysis API. A
same-side finite secant can be used instead:

.. math::

   J_A \approx
   \frac{V(A_p,q)-V(A_c,q)}{A_p-A_c},

where :math:`A_c` is a nearby amplitude on the same hard side of the bracket.
Detach this secant before constructing the proxy, keep the fresh
graph-connected probe replay, and use the linearization diagnostic to reject a
probe whose local response is inconsistent.

Constructing threshold and chronaxie gradients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a chosen probe, the lower-level construction is:

.. code-block:: python

   from dendra.models.analysis import trace_tangent_threshold_proxy

   result = trace_tangent_threshold_proxy(
       hard_threshold,
       probe_amplitude,
       probe_voltage,
       amplitude_tangent,
       mask=response_window_and_site_mask,
       check_amplitude=check_amplitude,
       check_voltage=check_voltage,
       min_tangent_norm=1e-9,
       max_relative_linearization_error=0.05,
   )
   if not result.valid:
       raise RuntimeError(result.rejection_reasons)

   (d_threshold_d_parameter,) = torch.autograd.grad(
       result.proxy,
       model_parameter,
   )

``result.proxy`` has exactly ``hard_threshold`` as its forward value. Its
backward direction projects parameter-induced voltage changes onto the local
amplitude-response direction. It is a proposed threshold direction, so the
complete hard threshold search remains the authority after an update.

The optional ``mask`` contains fixed nonnegative weights over time and site.
A zero excludes a sample; a positive value includes it, and larger values give
that sample more influence. Binary weights selecting the hard protocol's
recording sites and response window are a good default. A common scale factor
cancels from the projection. Hold the weights fixed during one derivative and
recompute them after accepting an update if the protocol requires it.

``select_trace_tangent_threshold_probe`` automates the probe-distance search
when a single threshold is being treated. The callback returns ``ProbeTrace``:

.. code-block:: python

   import torch

   from dendra.models.analysis import (
       ProbeTrace,
       select_trace_tangent_threshold_probe,
   )

   def trace_and_tangent(amplitude, need_tangent):
       amplitude = torch.as_tensor(
           amplitude,
           dtype=model_parameter.dtype,
           device=model_parameter.device,
       )
       if not need_tangent:
           with torch.no_grad():
               return ProbeTrace(run_voltage(amplitude).detach().clone())

       _, tangent = torch.autograd.functional.jvp(
           run_voltage,
           (amplitude,),
           (torch.ones_like(amplitude),),
           create_graph=False,
           strict=True,
       )
       # The fresh replay must be last so its parameter graph stays valid.
       voltage = run_voltage(amplitude)
       return ProbeTrace(voltage, tangent.detach())

   selected = select_trace_tangent_threshold_probe(
       hard_threshold,
       (inactive_bound, active_bound),
       hard_is_active=run_complete_hard_trial,
       trace_and_tangent=trace_and_tangent,
       mask=response_window_and_site_mask,
       min_tangent_norm=1e-9,
       max_relative_linearization_error=0.05,
       verify_bracket_endpoints=True,
   )

The callback shown here assumes a scalar-amplitude protocol; the downloadable
example contains the corresponding batched implementation and all model setup.
The selector first checks the hard side of each proposed probe, then checks
local amplitude linearity. Inspect ``selected.attempts`` and
``selected.rejection_reasons`` when no candidate is valid.

Chronaxie composes one valid threshold result per pulse width:

.. code-block:: python

   from dendra.models.analysis import chronaxie_from_threshold_proxies

   # When the automatic selector was used at each width:
   if not all(selection.valid for selection in selections):
       raise RuntimeError("At least one pulse width has no valid threshold proxy")
   threshold_results = [selection.threshold for selection in selections]

   fit = chronaxie_from_threshold_proxies(
       pulse_widths_ms,
       threshold_results,
       weights=fit_weights,
   )
   if not fit.valid:
       raise RuntimeError(fit.rejection_reasons)

   (d_chronaxie_d_parameter,) = torch.autograd.grad(
       fit.chronaxie_ms,
       model_parameter,
   )

Pass the ``TraceTangentThresholdResult`` stored in ``selection.threshold``,
rather than the enclosing ``ProbeSelectionResult`` or a detached proxy value.

The helper fits the current-domain Weiss relation
:math:`T(d)=r+b/d`, where :math:`d` is pulse width, :math:`r` is rheobase,
:math:`b` is the strength-duration slope, and :math:`c=b/r` is chronaxie. It
rejects incomplete, rank-deficient, or nonphysiological fits. For a positive
physical parameter represented by :math:`p=p_0\exp(q)`, report or optimize the
log-coordinate derivative as
:math:`\partial c/\partial q=p\,\partial c/\partial p`.

For paired-pulse recovery, make separate threshold proxies for the conditioned
and time-matched unconditioned protocols, then compose both terms:

.. code-block:: python

   from dendra.models.analysis import (
       paired_pulse_recovery_ratio_from_trace_tangents,
   )

   recovery = paired_pulse_recovery_ratio_from_trace_tangents(
       conditioned_threshold,
       unconditioned_threshold,
   )
   if recovery.valid:
       loss = (recovery.ratio - target_ratio).square().mean()

Do not detach the denominator. Its parameter sensitivity is part of the
recovery-ratio gradient.

Expected hard descriptors at event transitions
----------------------------------------------

For a hard descriptor with signed jumps :math:`\Delta_j` at ordered stimulus
roots :math:`r_j`, a declared Gaussian stimulus distribution produces a
continuous expected hard descriptor:

.. math::

   \mathbb{E}[H(A)] = H_{\mathrm{left}} +
   \sum_j \Delta_j\,\Phi\!\left(\frac{\mu-r_j}{\sigma}\right),
   \qquad A\sim\mathcal{N}(\mu,\sigma^2).

Here :math:`H_{\mathrm{left}}` is the hard value below the first root,
:math:`\Delta_j` is the signed change at root :math:`r_j`, :math:`\mu` is the
stimulus center, :math:`\sigma>0` is the declared stimulus standard deviation,
and :math:`\Phi` is the standard-normal cumulative distribution function.

.. code-block:: python

   from dendra.models.analysis import (
       gaussian_expected_hard_from_transition_roots,
   )

   expected_count = gaussian_expected_hard_from_transition_roots(
       center=stimulus_center,
       roots=torch.stack(root_proxies),
       jumps=signed_jumps,
       left_value=hard_value_below_all_roots,
       noise_std=stimulus_noise_std,
   )

This derivative belongs to the stated noise-averaged protocol. It is not the
pointwise derivative of a deterministic spike count. The hard transition scan
must cover the stimulus range with meaningful Gaussian probability, include
nonmonotone signed jumps, and independently validate every root direction.

Practical checks
----------------

Before relying on a training run, check the following:

* **Graph connection:** the descriptor tensor should require gradients, and
  every intended parameter should receive a finite nonzero gradient.
* **Forward measurement:** compare the differentiable output with its matching
  hard descriptor at the same parameters and protocol.
* **Recording horizon:** leave enough time for every response and for the tail
  of a rate-trajectory target. An event close to the final sample can create an
  apparent improvement by moving out of view.
* **Local direction:** on representative operating points, compare small steps
  in both gradient directions with fresh complete hard runs. Finite differences
  are validation runs; they are not required for every training coordinate.
* **Branch record:** save crossing-pair or event signatures. An adjacent pair
  change may follow the same continuously moving event, while a changed spike,
  invalid response, or selected peak needs closer inspection.
* **Recomputation:** initialize and rerun the model after every accepted update.
  Do not reuse a stale voltage graph, trace tangent, or threshold bracket.

If a descriptor reports invalid data or explicit rejection reasons, skip that
update and change the protocol, recording horizon, or probe design before
continuing.
