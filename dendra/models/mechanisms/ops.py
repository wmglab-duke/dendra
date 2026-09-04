import torch

exp = torch.exp
expm1 = torch.expm1
expit = torch.sigmoid
sigmoid = torch.sigmoid
log = torch.log
log10 = torch.log10
log2 = torch.log2
log1p = torch.log1p


def exprelr(x, y):
    """
    Compute x / (exp(x / y) - 1) with a series fallback for small x / y.

    This function implements a numerically stable evaluation of
    f(x, y) = x / (exp(x / y) - 1). For small q = x / y, direct evaluation
    of expm1(q) = exp(q) - 1 suffers from catastrophic cancellation.
    Using the series expansion expm1(q) ≈ q + q^2/2 + q^3/6 + ..., we obtain

        f(x, y) = x / expm1(q) = (q y) / expm1(q)
                 = y / (1 + q/2 + q^2/6 + ...)
                 ≈ y (1 - q/2 + q^2/12).

    The implementation switches to this second-order series when |q| < 1e-6.

    Parameters
    ----------
    x : torch.Tensor
        Numerator tensor.
    y : torch.Tensor
        Scaling tensor in the exponent; must be broadcastable with x.

    Returns
    -------
    torch.Tensor
        The value of x / expm1(x / y), using a series approximation
        when |x / y| < 1e-6.

    Notes
    -----
    - Exactly: f(x, y) = x / (e^{x / y} - 1).
    - For small q = x / y: f ≈ y - x/2 + x^2/(12 y) - ... .
    - Often this function will be referred to as 'vtrap' in e.g. NMODL files.

    Examples
    --------
    >>> import torch
    >>> exprelr(torch.tensor(1e-12), torch.tensor(1.0))
    tensor(1.)
    """
    q = x / y
    small = q.abs() < 1e-6
    # ``torch.where`` evaluates both branches. Feeding the exact singular
    # expression ``0 / expm1(0)`` to the inactive branch gives a finite primal
    # but a NaN derivative (the masked backward still encounters 0 * NaN).
    # Make that branch finite before selecting the series approximation.
    q_safe = torch.where(small, torch.ones_like(q), q)
    val = x / torch.expm1(q_safe)
    # ``addcmul`` retains the exact Hessian at the removable singularity while
    # fusing the only extra product needed beyond the historical first-order
    # branch. This helper is used in hot mechanism assigned-value paths.
    approx = torch.addcmul(y - x / 2, x, q, value=1 / 12)
    return torch.where(small, approx, val)


vtrap = exprelr


def expinv(x):
    """
    Compute x / (exp(x) - 1) with a series fallback for small x.

    This is the numerically stable reciprocal of the "exprel" function:
    expinv(x) = x / (exp(x) - 1) = 1 / exprel(x), where exprel(x) = (exp(x) - 1) / x.
    For small x, using expm1(x) ≈ x + x^2/2 + x^3/6 + ... yields

        x / expm1(x) = 1 - x/2 + x^2/12 - x^4/720 + ... .

    The implementation switches to a second-order series when |x| < 1e-6.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The value of x / expm1(x), using a series approximation
        when |x| < 1e-6.

    Notes
    -----
    - Stable around x = 0 to avoid loss of precision due to cancellation.
    - The small-input branch retains terms through second order.

    Examples
    --------
    >>> import torch
    >>> expinv(torch.tensor(1e-12))
    tensor(1.)
    """
    small = x.abs() < 1e-6
    x_safe = torch.where(small, torch.ones_like(x), x)
    val = x / torch.expm1(x_safe)
    approx = torch.addcmul(1 - 0.5 * x, x, x, value=1 / 12)
    return torch.where(small, approx, val)


def safe_exp(x):
    exp_ = torch.exp(x)
    return torch.where(
        torch.isfinite(exp_), exp_, torch.tensor(1e20, device=x.device, dtype=x.dtype)
    )


def all_ops():
    return {
        "exp",
        "expm1",
        "expit",
        "sigmoid",
        "log",
        "exprelr",
        "expinv",
        "safe_exp",
        "vtrap",
        "log10",
        "log2",
        "log1p",
    }
