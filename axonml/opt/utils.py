import torch
from torch.optim import Optimizer


def sanitize_grad(
    optimizer: Optimizer, clip_norm: float = None, clip_value: float = None
):
    """Sanitize gradients in-place by replacing NaNs/Infs with zeros. Optionally
    clip gradients by norm or by value.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer whose gradients will be sanitized.
    clip_norm : float, optional
        Maximum allowed norm of the gradients. If provided, gradients will be
        clipped to this norm.
    clip_value : float, optional
        Maximum allowed absolute value of the gradients. If provided, gradients
        will be clipped to this value.
    """
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                p.grad.data = torch.nan_to_num(p.grad.data)
                if clip_value is not None:
                    p.grad.data.clamp_(-clip_value, clip_value)
                if clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(p, clip_norm)
