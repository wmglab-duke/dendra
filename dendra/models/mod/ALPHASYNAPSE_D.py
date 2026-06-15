import torch

from ..mechanisms import PointProcess as M


class alphasynapse_d(M):
    r"""
    Differentiable analog of :class:`alphasynapse`.

    The original ``alphasynapse.alpha`` uses hard support checks
    ``x < 0`` and ``x > 10``.  This analog removes those boolean cutoffs by
    replacing ``x`` with a smooth non-negative approximation before evaluating
    the alpha waveform:

    .. math::

        x_+ \approx \max(x, 0), \qquad
        \alpha_d(x) = x_+ \exp(1 - x_+).

    The result is smooth at onset and naturally decays to zero for large
    positive ``x`` without a hard upper cutoff.

    Parameters
    ----------
    onset : float
        Activation onset time in ms. Default is 0.0 ms.
    tau : float
        Alpha-function time constant in ms. Default is 0.1 ms.
    gmax : float
        Maximum conductance scale in µS. Default is 0.0 µS.
    e : float
        Reversal potential in mV. Default is 0.0 mV.
    smooth_eps : float
        Width of the smooth non-negative approximation in dimensionless alpha
        time units. Smaller values more closely approximate ``max(x, 0)``.
        Default is 1e-6.
    """

    M.RANGE(onset=0.0, tau=0.1, gmax=0.0, e=0.0, smooth_eps=1.0e-6)
    M.NONSPECIFIC_CURRENT("i")

    @property
    def g(self):
        return self.gmax * alpha_d((self.t - self.onset) / self.tau, self.smooth_eps)

    def i(self, v):
        return self.g * (v - self.e)


def alpha_d(x: torch.Tensor, smooth_eps=1.0e-6) -> torch.Tensor:
    r"""
    Smooth non-negative alpha function.

    Parameters
    ----------
    x : torch.Tensor
        Dimensionless time ``(t - onset) / tau``.
    smooth_eps : float or torch.Tensor
        Positive smoothing width for the smooth approximation to ``max(x, 0)``.

    Returns
    -------
    torch.Tensor
        ``x_pos * exp(1 - x_pos)``, where ``x_pos`` is a smooth approximation to
        the positive part of ``x``.
    """

    eps = torch.as_tensor(smooth_eps, dtype=x.dtype, device=x.device).clamp_min(1.0e-12)
    x_pos = 0.5 * (x + torch.sqrt(x * x + eps * eps))
    return x_pos * torch.exp(1.0 - x_pos)
