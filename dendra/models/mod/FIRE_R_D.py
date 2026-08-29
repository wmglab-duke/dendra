import torch

from ..mechanisms import VoltageProcess


class fire_r_d(VoltageProcess):
    r"""
    Differentiable analog of :class:`fire_r` with hard-forward,
    surrogate-backward refractory-reset semantics.

    The forward pass preserves the original hard refractory logic:

    * ``is_refractory`` is a boolean refractory state.
    * ``time_refractory`` is the hard refractory countdown.
    * a new spike is emitted only when the cell is not refractory and
      ``v > threshold``.
    * the returned voltage is exactly ``rest`` while refractory and exactly ``v``
      otherwise.

    The backward pass uses a smooth sigmoid threshold surrogate for the new-spike
    and reset decisions when the cell is available to spike. During an already
    active refractory period, the smooth reset gate is one, so the voltage is
    clamped to ``rest`` in both the forward and surrogate backward paths.

    Parameters
    ----------
    threshold : float
        Voltage threshold in mV. Default is -50.0 mV.
    rest : float
        Reset/resting voltage in mV. Default is -65.0 mV.
    refractory : float
        Hard refractory duration in ms. Default is 5.0 ms.
    tau_gate : float
        Smoothness/steepness of the voltage threshold gate in mV. Default is
        0.5 mV.
    ste_scale : float
        Multiplicative scale applied to the surrogate-gradient contribution.
        Default is 1.0.

    Buffer variables
    ------------------
    is_refractory : torch.Tensor
        Boolean hard refractory state used by the forward pass.
    time_refractory : torch.Tensor
        Hard refractory countdown in ms.
    spike_gate : torch.Tensor
        Hard-forward 0/1 new-spike indicator with a smooth surrogate gradient.
    reset_gate : torch.Tensor
        Hard-forward 0/1 reset/refractory indicator with a smooth surrogate
        gradient.
    """

    VoltageProcess.RANGE(
        threshold=-50.0,
        rest=-65.0,
        refractory=5.0,
        tau_gate=0.5,
        ste_scale=1.0,
    )
    VoltageProcess.CARRY("is_refractory", dtype=torch.bool)
    VoltageProcess.CARRY("time_refractory", "spike_gate", "reset_gate")

    @staticmethod
    def _tensor_like(x, ref):
        if torch.is_tensor(x):
            return x.to(dtype=ref.dtype, device=ref.device)
        return torch.tensor(x, dtype=ref.dtype, device=ref.device)

    def initial_values(self, v, values):
        return {
            "is_refractory": torch.zeros_like(v, dtype=torch.bool),
            "time_refractory": torch.zeros_like(v),
            "spike_gate": torch.zeros_like(v),
            "reset_gate": torch.zeros_like(v),
        }

    def update_v(self, v):
        threshold = self._tensor_like(self.threshold, v)
        rest = self._tensor_like(self.rest, v)
        refractory = self._tensor_like(self.refractory, v)
        tau = self._tensor_like(self.tau_gate, v).clamp_min(1.0e-3)
        ste = self._tensor_like(self.ste_scale, v)

        # ----- Original hard refractory bookkeeping -----
        still_ref = self.is_refractory
        time_refractory = torch.where(
            still_ref, self.time_refractory - self.dt, self.time_refractory
        )

        recovered = still_ref & (time_refractory <= 0)
        is_refractory_before_spike = torch.where(recovered, False, still_ref)

        can_spike = ~is_refractory_before_spike
        hard_new_spike_bool = can_spike & (v > threshold)

        self.is_refractory = torch.where(
            hard_new_spike_bool, True, is_refractory_before_spike
        )
        self.time_refractory = torch.where(
            hard_new_spike_bool, refractory, time_refractory
        )

        hard_v = torch.where(self.is_refractory, rest, v)

        # ----- Smooth surrogate path for gradients -----
        availability = can_spike.to(dtype=v.dtype)
        refractory_before = is_refractory_before_spike.to(dtype=v.dtype)
        voltage_gate = torch.sigmoid((v - threshold) / tau)

        # New spikes are possible only outside the hard refractory period.
        soft_spike = voltage_gate * availability
        hard_new_spike = hard_new_spike_bool.to(dtype=v.dtype)
        self.spike_gate = hard_new_spike + ste * (soft_spike - soft_spike.detach())

        # Reset is hard-on during an existing refractory period, otherwise it is
        # driven by the threshold surrogate for a newly triggered spike.
        soft_reset_gate = 1.0 - (1.0 - refractory_before) * (1.0 - soft_spike)
        hard_reset = self.is_refractory.to(dtype=v.dtype)
        self.reset_gate = hard_reset + ste * (
            soft_reset_gate - soft_reset_gate.detach()
        )

        smooth_v = v + soft_reset_gate * (rest - v)
        return hard_v.detach() + ste * (smooth_v - smooth_v.detach())
