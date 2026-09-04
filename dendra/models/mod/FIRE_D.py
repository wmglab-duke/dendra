import torch

from ..mechanisms import VoltageProcess


class fire_d(VoltageProcess):
    r"""
    Differentiable analog of :class:`fire` with hard-forward, surrogate-backward
    reset semantics.

    The forward pass is exactly the original hard reset:

    .. math::

        v_\mathrm{hard} = \begin{cases}
            V_\mathrm{rest}, & v > \theta \\
            v, & v \le \theta.
        \end{cases}

    The backward pass is supplied by a smooth reset surrogate

    .. math::

        r(v) = \sigma((v - \theta) / \tau_g), \qquad
        v_\mathrm{soft} = v + r(v)(V_\mathrm{rest} - v).

    The returned voltage uses a straight-through estimator:

    .. math::

        v_\mathrm{out} = \operatorname{stopgrad}(v_\mathrm{hard})
                         + \alpha (v_\mathrm{soft}
                         - \operatorname{stopgrad}(v_\mathrm{soft})).

    Therefore the forward voltage is identical to ``fire``, while gradients are
    available near the threshold through the sigmoid reset gate.

    Parameters
    ----------
    threshold : float
        Voltage threshold in mV. Default is -50.0 mV.
    rest : float
        Reset/resting voltage in mV. Default is -65.0 mV.
    tau_gate : float
        Smoothness/steepness of the reset gate in mV. Default is 0.5 mV.
    ste_scale : float
        Multiplicative scale applied to the surrogate-gradient contribution.
        Default is 1.0.

    Buffer variables
    ------------------
    reset_gate : torch.Tensor
        Hard-forward 0/1 reset indicator with a smooth surrogate gradient.
    """

    VoltageProcess.RANGE(threshold=-50.0, rest=-65.0, tau_gate=0.5, ste_scale=1.0)
    VoltageProcess.CARRY("reset_gate")

    @staticmethod
    def _tensor_like(x, ref):
        if torch.is_tensor(x):
            return x.to(dtype=ref.dtype, device=ref.device)
        return torch.tensor(x, dtype=ref.dtype, device=ref.device)

    def initial_values(self, v, values):
        return {"reset_gate": torch.zeros_like(v)}

    def update_v(self, v):
        threshold = self._tensor_like(self.threshold, v)
        rest = self._tensor_like(self.rest, v)
        tau = self._tensor_like(self.tau_gate, v).clamp_min(1.0e-3)
        ste = self._tensor_like(self.ste_scale, v)

        hard_reset = v > threshold
        hard_v = torch.where(hard_reset, rest, v)

        smooth_gate = torch.sigmoid((v - threshold) / tau)
        smooth_v = v + smooth_gate * (rest - v)

        self.reset_gate = hard_reset.to(dtype=v.dtype) + ste * (
            smooth_gate - smooth_gate.detach()
        )

        return hard_v.detach() + ste * (smooth_v - smooth_v.detach())
