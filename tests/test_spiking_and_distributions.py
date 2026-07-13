import math

import pytest
import torch

import dendra as dn
from dendra.models.distributions import Distribution, Normal, TruncatedNormal
from dendra.models.networks.spiking import (
    crossing_spike,
    crossing_spikes,
    level_spike,
    level_spikes,
    sigmoid_ste,
    update_active,
    update_active_diff,
)


def test_hard_and_differentiable_active_updates_agree_on_rising_edges():
    has_spiked = torch.tensor([False, True, False])
    voltage = torch.tensor([-1.0, 1.0, 2.0], requires_grad=True)
    threshold = torch.tensor(0.0)

    active, spikes = update_active(has_spiked, voltage, threshold)
    active_diff, active_gate, spike_gate = update_active_diff(
        has_spiked, voltage, threshold, tau=0.5
    )

    assert active.tolist() == [False, True, True]
    assert spikes.tolist() == [False, False, True]
    assert torch.equal(active_diff, active)
    assert active_gate.tolist() == pytest.approx([0.0, 1.0, 1.0])
    assert spike_gate.tolist() == pytest.approx([0.0, 0.0, 1.0])
    (active_gate.sum() + spike_gate.sum()).backward()
    assert torch.isfinite(voltage.grad).all()
    assert torch.count_nonzero(voltage.grad) > 0


def test_sigmoid_ste_has_hard_forward_values_and_surrogate_gradients():
    x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    tau = torch.tensor(0.2, requires_grad=True)

    gate = sigmoid_ste(x, tau)

    assert gate.tolist() == [0.0, 1.0, 1.0]
    gate.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad) == 3
    assert torch.isfinite(tau.grad)


def test_crossing_and_level_surrogates_preserve_hard_forward_contracts():
    old = torch.tensor([-1.0, 1.0, -1.0], requires_grad=True)
    new = torch.tensor([1.0, 2.0, -0.5], requires_grad=True)
    threshold = torch.tensor(0.0)

    crossing = crossing_spike(old, new, threshold)
    level = level_spike(new, threshold)

    assert crossing.tolist() == [1.0, 0.0, 0.0]
    assert level.tolist() == [1.0, 1.0, 0.0]
    assert torch.equal(crossing_spikes(old, new, threshold), crossing)
    assert torch.equal(level_spikes(new, threshold), level)
    (crossing.sum() + level.sum()).backward()
    assert torch.isfinite(old.grad).all()
    assert torch.isfinite(new.grad).all()


def test_distribution_base_class_requires_sampling_and_log_prob_implementations():
    distribution = Distribution()
    with pytest.raises(NotImplementedError):
        distribution.rsample()
    with pytest.raises(NotImplementedError):
        distribution.log_prob(torch.tensor(0.0))


def test_learnable_normal_matches_torch_distribution_and_backpropagates():
    with dn.ctx(REQUIRE_GRAD=1):
        distribution = Normal(
            mean=torch.tensor([0.0, 1.0], dtype=torch.float64),
            std=torch.tensor([1.0, 2.0], dtype=torch.float64),
        )
    value = torch.tensor([0.5, -1.0], dtype=torch.float64)
    expected = torch.distributions.Normal(distribution.mean, distribution.std).log_prob(
        value
    )

    assert torch.allclose(distribution.log_prob(value), expected)
    sample = distribution.sample(4)
    assert sample.shape == (4, 2)
    sample.sum().backward()
    assert distribution.mean.grad is not None
    assert distribution.std.grad is not None


def test_truncated_normal_samples_within_support_and_has_normalized_log_prob():
    torch.manual_seed(123)
    with dn.ctx(REQUIRE_GRAD=1):
        distribution = TruncatedNormal(
            mean=torch.tensor(0.0, dtype=torch.float64),
            std=torch.tensor(1.0, dtype=torch.float64),
            low=-1.0,
            high=2.0,
        )

    samples = distribution.sample(1000)
    assert torch.all(samples >= -1.0)
    assert torch.all(samples <= 2.0)

    values = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64)
    log_prob = distribution.log_prob(values)
    assert torch.isneginf(log_prob[[0, 2]]).all()
    normalizer = torch.distributions.Normal(0.0, 1.0).cdf(torch.tensor(2.0)) - (
        torch.distributions.Normal(0.0, 1.0).cdf(torch.tensor(-1.0))
    )
    expected_center = -0.5 * math.log(2 * math.pi) - torch.log(normalizer)
    assert log_prob[1].item() == pytest.approx(expected_center.item(), rel=1e-6)

    samples.mean().backward()
    assert distribution.mean.grad is not None
    assert distribution._log_std.grad is not None
