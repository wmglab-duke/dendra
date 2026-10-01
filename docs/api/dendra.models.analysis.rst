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

Historical descriptor interface
-------------------------------

The functions below retain their historical imports and behavior for
compatibility. For new training code, prefer the supported hard-forward
interfaces above. ``frequency_following`` is experimental as an optimization
objective; use it for diagnostics unless its direction has been validated for
the declared model and protocol.

.. automodule:: dendra.models.analysis.descriptors
   :members:
   :undoc-members:
   :show-inheritance:

Experimental event creation and removal
---------------------------------------

Voltage-margin objectives for creating or extinguishing events require
additional carrier and hard-rerun checks. They remain available from
``dendra.models.analysis.event_margin_training`` and are described in
:ref:`event-margin-training`.
