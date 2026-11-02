dendra.models.analysis
======================

``dendra.models.analysis`` contains hard measurement functions and
graph-connected macroscopic descriptors. The training guide explains how to
choose a protocol, build a loss, update model parameters, and rerun the hard
measurement after each proposal: :ref:`macroscopic-descriptor-training`.

Supported trace descriptors
---------------------------

These interfaces take a voltage tensor from any Dendra model. Their forward
values use the matching hard event choices; their gradients follow the
linearly interpolated crossings on the branch selected in the current replay.

.. currentmodule:: dendra.models.analysis

.. autofunction:: branch_conditioned_conduction_velocity
.. autofunction:: differentiable_action_potential_width
.. autofunction:: differentiable_activity_dependent_slowing
.. autofunction:: differentiable_spike_timing
.. autofunction:: differentiable_firing_rate_trajectory
.. autofunction:: compare_branch_signatures
.. autofunction:: hard_spike_timing
.. autofunction:: hard_firing_rate_trajectory

The implementation names ``branch_conditioned_action_potential_width``,
``branch_conditioned_activity_dependent_slowing``,
``branch_conditioned_spike_timing``, and
``branch_conditioned_firing_rate_trajectory`` are equivalent public aliases.
``differentiable_conduction_velocity`` is an alias for the preferred explicit
name ``branch_conditioned_conduction_velocity``.

Standalone conduction velocity
------------------------------

``branch_conditioned_conduction_velocity`` has the exact
``hard_conduction_velocity`` value on its forward pass. It selects the first
hard upward threshold crossing at every included site, linearly interpolates
each selected crossing with the voltage graph intact, and differentiates the
spatial ordinary-least-squares fit through those arrival times. A valid
velocity requires at least two included sites with crossings and arrival-time
variance greater than ``eps``. Use ``node_mask`` and the optional time window
to select recording sites and isolate the propagated action potential of
interest.

The derivative is local to those event choices. It does not describe the
creation of a missing distal response, the loss of a response, or a switch to
a different spike. Rerun the full protocol and recompute the descriptor after
every parameter update so that the selected branch follows the current model.

The ``conduction_velocity`` function is a distinct smooth
estimator based on soft upstroke localization and soft spike-presence weights.
It estimates a smooth response rather than the exact hard-forward
measurement above.

Threshold-derived training objectives
-------------------------------------

A complete hard search supplies each threshold and bracket. A nearby voltage
replay supplies a validated local direction. The composition helpers preserve
the hard threshold values on the forward pass and propagate their directions
through activation probability, a strength-duration fit, or a paired-pulse
ratio. The :download:`complete threshold and chronaxie example
<../../examples/threshold_descriptor_gradients.py>` shows the required hard
search, amplitude JVP, replay ordering, parameter gradients, and independent
hard finite-difference checks.

.. autofunction:: select_trace_tangent_threshold_probe
.. autofunction:: trace_tangent_threshold_proxy
.. autofunction:: gaussian_expected_activation
.. autofunction:: chronaxie_from_threshold_proxies
.. autofunction:: paired_pulse_recovery_ratio_from_trace_tangents

.. autoclass:: ProbeTrace
   :members:

.. autoclass:: ProbeAttempt
   :members:

.. autoclass:: ProbeSelectionResult
   :members:

.. autoclass:: TraceTangentDiagnostics
   :members:

.. autoclass:: TraceTangentThresholdResult
   :members:

.. autoclass:: TraceTangentChronaxieResult
   :members:

.. autoclass:: TraceTangentRecoveryResult
   :members:

General hard-transition expectation
-----------------------------------

Use this lower-level composition when a complete hard scan contains more than
one transition or signed, nonmonotone jumps.

.. autofunction:: gaussian_expected_hard_from_transition_roots

Additional measurements
-----------------------

These functions measure responses and protocol results. For training, use the
hard-forward interfaces above when their measurement contracts apply. ``frequency_following`` is experimental as an optimization
objective; use it for diagnostics unless its direction has been validated for
the declared model and protocol.

.. autofunction:: action_potential_width
.. autofunction:: active
.. autofunction:: activity_dependent_slowing
.. autofunction:: chronaxie_from_trials
.. autofunction:: compute_rheobase_chronaxie
.. autofunction:: conduction_velocity
.. autofunction:: firing_rate
.. autofunction:: frequency_following
.. autofunction:: hard_action_potential_width
.. autofunction:: hard_active
.. autofunction:: hard_activity_dependent_slowing
.. autofunction:: hard_chronaxie_from_trials
.. autofunction:: hard_conduction_velocity
.. autofunction:: hard_firing_rate
.. autofunction:: hard_frequency_following
.. autofunction:: hard_paired_pulse_recovery_from_trials
.. autofunction:: hard_spike_arrival_times
.. autofunction:: paired_pulse_recovery_from_trials
.. autofunction:: plot_activation_heatmap_from_chronaxie_output

Experimental event creation and removal
---------------------------------------

Voltage-margin objectives for creating or extinguishing events require
additional carrier and hard-rerun checks. They remain available from
``dendra.models.analysis.event_margin_training`` and are described in
:ref:`event-margin-training`.
