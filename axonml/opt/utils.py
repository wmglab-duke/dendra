import torch
from torch.optim import Optimizer


def sanitize_grad(optimizer: Optimizer, clip_value: float = 1e6):
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                p.grad.data = torch.nan_to_num(p.grad.data)
                p.grad.data.clamp_(-clip_value, clip_value)
