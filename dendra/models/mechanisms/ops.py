import torch

exp = torch.exp
expm1 = torch.expm1
expit = torch.sigmoid
sigmoid = torch.sigmoid
log = torch.log
log10 = torch.log10
log2 = torch.log2
log1p = torch.log1p


def _reciprocal_exprel_even(q):
    """Even part of the local Bernoulli series, excluding its q**2 factor.

    Thus ``q / expm1(q) = 1 - q/2 + q**2 * result``. The degree is twelve
    for float32 on |q| <= 1 and eighteen for float64 on |q| <= 0.5. Bounds
    for the omitted value/first/second derivative terms are respectively
    1.4e-11/2.0e-10/2.6e-9 and 2.2e-22/8.5e-21/3.3e-19. They follow from
    |B_(2n)|/(2n)! = 2*zeta(2n)/(2*pi)**(2n), summing the decreasing tail.
    Floating-point evaluation, not Taylor truncation, dominates the error.

    Callers mask nonlocal inputs before evaluating the polynomial, including
    its inactive branch. This helper is not a global polynomial approximation.
    """
    # Keep scalar field arithmetic vector-shaped: Torch 2.14 func.jvp can
    # otherwise promote float32 tangents when Python floating constants occur.
    # These are views only; non-scalar mechanism fields take the unchanged path.
    scalar = q.ndim == 0
    if scalar:
        q = q.unsqueeze(0)
    q2 = q * q
    if q.dtype == torch.float64:
        value = q2 * (43867 / 5109094217170944000) - 3617 / 10670622842880000
        value = value * q2 + 1 / 74724249600
        value = value * q2 - 691 / 1307674368000
        value = value * q2 + 1 / 47900160
    else:
        value = q2 * (-691 / 1307674368000) + 1 / 47900160
    value = value * q2 - 1 / 1209600
    value = value * q2 + 1 / 30240
    value = value * q2 - 1 / 720
    value = value * q2 + 1 / 12
    return value.squeeze(0) if scalar else value


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

    For float32, a degree-twelve Bernoulli series is used when |q| <= 1;
    float64 uses degree eighteen when |q| <= 0.5. Outside that region the
    ordinary ``exp(q) - 1`` denominator is bounded away from zero. This compact
    evaluation targets physiological voltage/rate ranges; it does not extend
    the historical exponential's nonlocal overflow or underflow range.
    Reduced-precision and complex inputs retain the historical
    expm1/second-order-series path.

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
        near the removable singularity.

    Notes
    -----
    - Exactly: f(x, y) = x / (e^{x / y} - 1).
    - For small q = x / y: f ≈ y - x/2 + x^2/(12 y) - ... .
    - Either sign of a nonzero scale y is supported. Outside the local series,
      the ordinary exponential retains its historical representational limits;
      no extension is promised for an overflowed quotient or exponential.
    - Often this function will be referred to as 'vtrap' in e.g. NMODL files.

    Examples
    --------
    >>> import torch
    >>> exprelr(torch.tensor(1e-12), torch.tensor(1.0))
    tensor(1.)
    """
    q = x / y
    if q.dtype == torch.float32 or q.dtype == torch.float64:
        threshold = 1.0 if q.dtype == torch.float32 else 0.5
        small = q.abs() <= threshold

        # x*q, rather than y*q*q, preserves the scale and avoids reconstructing
        # the numerator. Keep the unused polynomial bounded at nonlocal inputs.
        small_q = torch.where(small, q, torch.zeros_like(q))
        small_x = torch.where(small, x, torch.zeros_like(q))
        small_y = torch.where(small, y, torch.zeros_like(q))
        even = _reciprocal_exprel_even(small_q)
        approx = torch.addcmul(small_y - small_x / 2, small_x, small_q * even)
        # Neither zero nor cancellation-prone small inputs reach exp(q)-1.
        # q_safe=1 also keeps its inactive branch finite near the singularity.
        q_safe = torch.where(small, torch.ones_like(q), q)
        val = x / (torch.exp(q_safe) - 1)
        return torch.where(small, approx, val)

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

    Float32 and float64 use the same dtype-aware local-series evaluation as
    exprelr(x, 1), retaining the ordinary exponential's nonlocal range limits.
    Other dtypes retain the second-order series at |x| < 1e-6.

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
    - The real float32/float64 small-input branches retain terms through
      degree twelve/eighteen respectively.

    Examples
    --------
    >>> import torch
    >>> expinv(torch.tensor(1e-12))
    tensor(1.)
    """
    if x.dtype == torch.float32 or x.dtype == torch.float64:
        return exprelr(x, torch.ones_like(x))
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
