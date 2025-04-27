import torch

exp = torch.exp
expm1 = torch.expm1
expit = torch.sigmoid
sigmoid = torch.sigmoid
log = torch.log
log10 = torch.log10
log2 = torch.log2
log1p = torch.log1p


@torch.jit._overload
def exprelr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: pass

@torch.jit._overload
def exprelr(x: torch.Tensor, y: float) -> torch.Tensor: pass

@torch.jit._overload
def exprelr(x: float, y: torch.Tensor) -> torch.Tensor: pass

@torch.jit._overload
def vtrap(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: pass

@torch.jit._overload
def vtrap(x: torch.Tensor, y: float) -> torch.Tensor: pass

@torch.jit._overload
def vtrap(x: float, y: torch.Tensor) -> torch.Tensor: pass


def exprelr(x, y):
    q = x / y
    val = x / torch.expm1(q)
    approx = y - (x / 2)
    return torch.where(q.abs() < 1e-6, approx, val)

vtrap = exprelr

@torch.jit.script
def expinv(x):
    val = x / torch.expm1(x)
    approx = 1 - 0.5 * x
    return torch.where(x.abs() < 1e-6, approx, val)


@torch.jit.script
def safe_exp(x):
    exp_ = torch.exp(x)
    return torch.where(torch.isfinite(exp_), exp_, torch.tensor(1e20))


def all_ops():
    return {"exp", "expm1", "expit", "sigmoid", "log", "exprelr", "expinv", "safe_exp", "vtrap"}
