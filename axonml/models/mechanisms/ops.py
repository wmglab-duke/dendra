import torch

exp = torch.exp
expm1 = torch.expm1
expit = torch.sigmoid

@torch.jit.script
def exprelr(x, y):
    val = x / expm1(x / y)
    approx = y - (x / 2)
    return torch.where(x.abs() < 1e-6, approx, val)
