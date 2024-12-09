import torch

exp = torch.exp
expm1 = torch.expm1
expit = torch.sigmoid


@torch.jit.script
def exprelr(x, y):
    q = x / y
    val = x / torch.expm1(q)
    approx = y - (x / 2)
    return torch.where(q.abs() < 1e-6, approx, val)
