import torch

from ..mechanisms import VoltageProcess


class fire(VoltageProcess):
    """
    A VoltageProcess that implements a leaky integrate-and-fire neuron model.
    When the membrane potential `v` crosses the specified threshold, it is reset to the
    resting potential.

    Parameters
    ----------
    threshold : float
        The membrane potential threshold for spike detection. Default is -50.0 mV.
    rest : float
        The resting membrane potential (in mV) to reset to after a spike. Default is -65.0 mV.
    """

    VoltageProcess.RANGE(threshold=-50.0, rest=-65.0)

    def update_v(self, v):
        return torch.where(v > self.threshold, self.rest, v)
