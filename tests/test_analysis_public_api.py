"""Public import contracts for supported macroscopic-descriptor training APIs."""

import torch

import dendra.models.analysis as analysis
from dendra.models.analysis.branch_conditioned_metrics import (
    branch_conditioned_action_potential_width,
    branch_conditioned_activity_dependent_slowing,
)
from dendra.models.analysis.firing_rate_trajectory import (
    branch_conditioned_firing_rate_trajectory,
)
from dendra.models.analysis.spike_timing import branch_conditioned_spike_timing


def test_differentiable_descriptor_aliases_identify_supported_implementations():
    assert (
        analysis.differentiable_action_potential_width
        is branch_conditioned_action_potential_width
    )
    assert (
        analysis.differentiable_activity_dependent_slowing
        is branch_conditioned_activity_dependent_slowing
    )
    assert analysis.differentiable_spike_timing is branch_conditioned_spike_timing
    assert (
        analysis.differentiable_firing_rate_trajectory
        is branch_conditioned_firing_rate_trajectory
    )


def test_supported_training_surfaces_are_declared_public():
    expected = {
        "differentiable_action_potential_width",
        "differentiable_activity_dependent_slowing",
        "differentiable_spike_timing",
        "differentiable_firing_rate_trajectory",
        "hard_spike_timing",
        "hard_firing_rate_trajectory",
        "trace_tangent_threshold_proxy",
        "select_trace_tangent_threshold_probe",
        "chronaxie_from_threshold_proxies",
        "paired_pulse_recovery_ratio_from_trace_tangents",
        "gaussian_expected_activation",
        "gaussian_expected_hard_from_transition_roots",
    }
    assert expected <= set(analysis.__all__)
    assert all(hasattr(analysis, name) for name in expected)


def test_public_gaussian_activation_composes_with_autograd_threshold():
    threshold = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    probability = analysis.gaussian_expected_activation(0.35, threshold, 0.1)
    (slope,) = torch.autograd.grad(probability, threshold)
    assert probability.requires_grad
    assert slope < 0
