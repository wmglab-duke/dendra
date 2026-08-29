import torch

from ..mechanisms import Mechanism as M


class apcount_d(M):
    r"""
    Differentiable analog of :class:`apcount` with hard-forward, surrogate-backward
    spike counting.

    The forward pass preserves event/count semantics:

    * ``spikes`` is exactly a 0/1 upward-threshold-crossing indicator.
    * ``n`` is exactly the accumulated number of upward crossings, represented as
      a floating tensor so it can carry an autograd graph.

    The backward pass uses a straight-through estimator (STE), following the same
    pattern as :class:`spikedetect`: the emitted hard event has the forward value
    of the boolean crossing, but gradients are supplied by a smooth onset
    surrogate.

    Let

    .. math::

        g_t = \sigma((v_t - \theta) / \tau_g)

    and let ``h_prev`` store the previous gate. The hard crossing is

    .. math::

        s_t^\mathrm{hard} = \mathbf{1}[v_t > \theta] \;\mathbf{1}[v_{t-1} \le \theta],

    implemented using the hard boolean state ``active``. The surrogate used for
    gradients is

    .. math::

        s_t^\mathrm{soft} = \max(g_t - g_{t-1}, 0).

    The buffer-backed spike variable is

    .. math::

        s_t = s_t^\mathrm{hard}
              + \alpha (s_t^\mathrm{soft} - \operatorname{stopgrad}(s_t^\mathrm{soft})),

    so its forward value is exactly hard while its backward gradient is shaped by
    the smooth gate. The count is then ``n <- n + spikes``.

    Parameters
    ----------
    threshold : float
        Membrane-potential threshold in mV. Default is 0.0 mV.
    tau_gate : float
        Smoothness/steepness of the voltage threshold gate in mV. Smaller values
        make the surrogate sharper but increase gradient magnitude near
        threshold. Default is 0.5 mV.
    ste_scale : float
        Multiplicative scale applied to the surrogate-gradient contribution.
        Default is 1.0.

    Buffer variables
    ------------------
    n : torch.Tensor
        Accumulated hard-forward spike count with surrogate gradients.
    spikes : torch.Tensor
        Current-step hard-forward upward-crossing indicator with surrogate
        gradients.
    active : torch.Tensor
        Boolean hard state indicating whether the previous voltage was above
        threshold.
    h_prev : torch.Tensor
        Previous smooth threshold gate used only for the surrogate gradient.
    """

    M.RANGE(threshold=0.0, tau_gate=0.5, ste_scale=1.0)
    M.CARRY("n", "spikes", "h_prev")
    M.CARRY("active", dtype=torch.bool)

    @staticmethod
    def _tensor_like(x, ref):
        if torch.is_tensor(x):
            return x.to(dtype=ref.dtype, device=ref.device)
        return torch.tensor(x, dtype=ref.dtype, device=ref.device)

    def _gate(self, v):
        threshold = self._tensor_like(self.threshold, v)
        tau = self._tensor_like(self.tau_gate, v).clamp_min(1.0e-3)
        return torch.sigmoid((v - threshold) / tau)

    def initial_values(self, v, values):
        threshold = self._tensor_like(self.threshold, v)
        gate0 = self._gate(v)

        # Initialize to the current hard state so starting above threshold does
        # not create a spurious first-step crossing. This matches spikedetect's
        # upward-crossing semantics.
        return {
            "n": torch.zeros_like(v),
            "spikes": torch.zeros_like(v),
            "active": v > threshold,
            "h_prev": gate0,
        }

    def advance(self, v, dt, values):
        del dt
        threshold = self._tensor_like(self.threshold, v)
        ste = self._tensor_like(self.ste_scale, v)

        gate = self._gate(v)
        above_threshold = v > threshold

        # Hard upward crossing used for the forward value.
        hard_spikes = above_threshold & ~values["active"]
        hard_spikes = hard_spikes.to(dtype=v.dtype)

        # Smooth positive gate change used only for the backward value.
        soft_spikes = torch.relu(gate - values["h_prev"])

        # Straight-through estimator: forward == hard_spikes, backward ==
        # ste_scale * d soft_spikes / d inputs.
        spikes = hard_spikes + ste * (soft_spikes - soft_spikes.detach())

        # Update memories after computing the event.
        return {
            "n": values["n"] + spikes,
            "spikes": spikes,
            "active": above_threshold,
            "h_prev": gate,
        }
        # If truncated BPTT is desired, use: self.h_prev = gate.detach()
