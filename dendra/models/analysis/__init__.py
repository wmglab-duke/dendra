"""Hard measurements and differentiable macroscopic descriptors.

Historical descriptor imports remain available for compatibility.  New code
should prefer the ``differentiable_*`` hard-forward, branch-conditioned
interfaces for AP width, activity-dependent slowing, spike timing, and firing
rate trajectories.  Threshold-derived helpers attach validated local training
directions to complete hard-protocol values and compose them into activation,
chronaxie, and paired-pulse objectives.

Additional research diagnostics remain available from their named submodules.
"""

from .branch_conditioned_metrics import (
    branch_conditioned_action_potential_width,
    branch_conditioned_activity_dependent_slowing,
    compare_branch_signatures,
)
from .descriptors import (
    ArrayLike1D,
    LengthLike,
    _as_F_vector,
    _coerce_compartment_weights,
    _coerce_lengths_um,
    _coerce_positive_scalar,
    _coerce_pulse_times_ms,
    _coerce_train_frequency_hz,
    _finite_mean_over_indices,
    _frequency_following_summary_from_ads,
    _frequency_group_ids,
    _gather_time_windows_FNKC,
    _grouped_mean_by_frequency,
    _hard_width_crossings_around_peak,
    _piecewise_linear_window_mean,
    _soft_arrival_and_spike_gate,
    _weighted_mean_over_pulses,
    _weighted_mean_over_time_indices,
    _weighted_weiss_fit,
    action_potential_width,
    active,
    activity_dependent_slowing,
    chronaxie_from_trials,
    compute_rheobase_chronaxie,
    conduction_velocity,
    firing_rate,
    frequency_following,
    hard_action_potential_width,
    hard_active,
    hard_activity_dependent_slowing,
    hard_chronaxie_from_trials,
    hard_conduction_velocity,
    hard_firing_rate,
    hard_frequency_following,
    hard_paired_pulse_recovery_from_trials,
    hard_spike_arrival_times,
    paired_pulse_recovery_from_trials,
    plot_activation_heatmap_from_chronaxie_output,
)
from .firing_rate_trajectory import (
    branch_conditioned_firing_rate_trajectory,
    hard_firing_rate_trajectory,
)
from .spike_timing import branch_conditioned_spike_timing, hard_spike_timing
from .threshold_trace_probe import (
    ProbeAttempt,
    ProbeSelectionResult,
    ProbeTrace,
    select_trace_tangent_threshold_probe,
)
from .threshold_trace_tangent import (
    TraceTangentChronaxieResult,
    TraceTangentDiagnostics,
    TraceTangentRecoveryResult,
    TraceTangentThresholdResult,
    chronaxie_from_threshold_proxies,
    paired_pulse_recovery_ratio_from_trace_tangents,
    trace_tangent_threshold_proxy,
)
from .threshold_transition_metrics import (
    gaussian_expected_activation,
    gaussian_expected_hard_from_transition_roots,
)

# Descriptive names for the supported training interfaces.  Keep the
# implementation names public as well so diagnostics and existing notebooks
# can describe the selected-branch construction precisely.
differentiable_action_potential_width = branch_conditioned_action_potential_width
differentiable_activity_dependent_slowing = (
    branch_conditioned_activity_dependent_slowing
)
differentiable_spike_timing = branch_conditioned_spike_timing
differentiable_firing_rate_trajectory = branch_conditioned_firing_rate_trajectory

__all__ = [
    "ArrayLike1D",
    "LengthLike",
    "action_potential_width",
    "active",
    "activity_dependent_slowing",
    "branch_conditioned_action_potential_width",
    "branch_conditioned_activity_dependent_slowing",
    "branch_conditioned_firing_rate_trajectory",
    "branch_conditioned_spike_timing",
    "chronaxie_from_trials",
    "chronaxie_from_threshold_proxies",
    "compare_branch_signatures",
    "compute_rheobase_chronaxie",
    "conduction_velocity",
    "differentiable_action_potential_width",
    "differentiable_activity_dependent_slowing",
    "differentiable_firing_rate_trajectory",
    "differentiable_spike_timing",
    "firing_rate",
    "frequency_following",
    "gaussian_expected_activation",
    "gaussian_expected_hard_from_transition_roots",
    "hard_action_potential_width",
    "hard_active",
    "hard_activity_dependent_slowing",
    "hard_chronaxie_from_trials",
    "hard_conduction_velocity",
    "hard_firing_rate",
    "hard_firing_rate_trajectory",
    "hard_frequency_following",
    "hard_paired_pulse_recovery_from_trials",
    "hard_spike_arrival_times",
    "hard_spike_timing",
    "paired_pulse_recovery_from_trials",
    "paired_pulse_recovery_ratio_from_trace_tangents",
    "plot_activation_heatmap_from_chronaxie_output",
    "ProbeAttempt",
    "ProbeSelectionResult",
    "ProbeTrace",
    "select_trace_tangent_threshold_probe",
    "TraceTangentChronaxieResult",
    "TraceTangentDiagnostics",
    "TraceTangentRecoveryResult",
    "TraceTangentThresholdResult",
    "trace_tangent_threshold_proxy",
]
