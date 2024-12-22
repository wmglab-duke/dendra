import torch

exp = torch.exp
expm1 = torch.expm1
expit = torch.sigmoid
sigmoid = torch.sigmoid
log = torch.log


@torch.jit.script
def exprelr(x, y):
    q = x / y
    val = x / torch.expm1(q)
    approx = y - (x / 2)
    return torch.where(q.abs() < 1e-6, approx, val)


@torch.jit.script
def expinv(x):
    val = x / torch.expm1(x)
    approx = 1 - 0.5 * x
    return torch.where(x.abs() < 1e-6, approx, val)


def all_ops():
    return {"exp", "expm1", "expit", "sigmoid", "log", "exprelr", "expinv"}
