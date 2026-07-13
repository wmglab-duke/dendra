import torch

from dendra.utils import interp1d

from .core import Waveform

__all__ = [
    "sin",
    "cos",
    "mono_rect",
    "bi_rect",
    "bi_rect_balanced",
    "bi_rect_symm",
    "arbitrary",
    "constant",
]


def _as_tensor_like(x, ref: torch.Tensor) -> torch.Tensor:
    """
    Convert python/numpy scalars to a tensor on ref.device/ref.dtype.
    Leave torch.Tensors (incl. nn.Parameter) untouched to preserve grads.
    """
    if torch.is_tensor(x):
        return x
    return ref.new_tensor(x)


def _time_broadcast_param(x, t: torch.Tensor) -> torch.Tensor:
    """
    Ensure x broadcasts against t (shape [T]) with time as the LAST dim.

    Rules:
      - scalars (0-dim) are fine as-is
      - scalar-time evaluation preserves the parameter shape
      - if the last dim is already 1, keep that explicit broadcast axis
      - otherwise append a trailing singleton time dim, e.g. [B] -> [B,1]
        and [B,C] -> [B,C,1]

    Parameter axes are never inferred to be time merely because their length
    happens to equal the number of evaluation points. The old heuristic made
    output rank depend on ``T`` and silently confused population/compartment
    axes with time.
    """
    x = _as_tensor_like(x, t)
    if x.ndim == 0 or t.ndim == 0:
        return x
    if x.shape[-1] == 1:
        return x
    return x.unsqueeze(-1)


def _rect_gate(t, start, stop, tau, inclusive_stop=False):
    # Canonicalize to broadcast across time
    start = _time_broadcast_param(start, t)
    stop = _time_broadcast_param(stop, t)
    tau = _time_broadcast_param(tau, t)

    tau = torch.clamp(tau, min=1e-6)
    soft = torch.sigmoid((t - start) / tau) * torch.sigmoid((stop - t) / tau)

    if inclusive_stop:
        hard = ((t >= start) & (t <= stop)).to(soft.dtype)
    else:
        hard = ((t >= start) & (t < stop)).to(soft.dtype)

    # Straight-through gate: hard in forward, soft for gradients.
    return hard + (soft - soft.detach())


def _oscillator_broadcast_param(x, t: torch.Tensor) -> torch.Tensor:
    """
    Canonicalize parameters for sin/cos multicomponent oscillators.

    Convention for oscillator parameters:
      - scalar: one component, no batch
      - [K]: K oscillator components
      - [B, K]: B batches, K oscillator components per batch
      - [..., K, T] or [..., K, 1]: explicit time axis; component axis is -2

    This intentionally differs from _time_broadcast_param for 1-D and 2-D
    tensors: in sin/cos, a 1-D tensor is components, and a 2-D tensor is
    batch-by-components, not a time-varying vector.
    """
    x = _as_tensor_like(x, t)
    if x.ndim == 0:
        return x

    # 1-D means [K] components. 2-D means [B, K] batched components.
    # In both cases, append a trailing singleton time axis.
    if x.ndim in (1, 2):
        return x.unsqueeze(-1)

    # For >=3-D tensors, allow an explicit trailing time axis.
    T = t.shape[-1]
    if x.shape[-1] in (1, T):
        return x
    return x.unsqueeze(-1)


def _sum_oscillator_components(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Sum over the oscillator component axis.

    The component axis is the dimension immediately before time. If all
    oscillator parameters are scalar, y has the same rank as t and is returned
    unchanged.
    """
    if y.ndim == t.ndim:
        return y
    return y.sum(dim=-2)


def _trig_oscillator_sum(
    t: torch.Tensor,
    amp,
    freq,
    phase,
    delay,
    off,
    off_after,
    tau,
    trig,
) -> torch.Tensor:
    """
    Shared multicomponent implementation for sin/cos.

    Returns:
      - [T] for scalar or unbatched component parameters
      - [B, T] for [B, K] batched component parameters
      - [..., T] for higher-rank batched component parameters
    """
    amp = _oscillator_broadcast_param(amp, t)
    freq = _oscillator_broadcast_param(freq, t)
    phase = _oscillator_broadcast_param(phase, t)
    delay = _oscillator_broadcast_param(delay, t)
    off = _oscillator_broadcast_param(off, t)
    off_after = _oscillator_broadcast_param(off_after, t)
    tau = _oscillator_broadcast_param(tau, t)

    off_eff = torch.minimum(off, off_after + delay)
    arg = 2 * torch.pi * freq * (t - delay) + phase
    gate = _rect_gate(t, delay, off_eff, tau)

    y = amp * gate * trig(arg)
    return _sum_oscillator_components(y, t)


def _first_tensor_from_waveform(waveform):
    """
    Find a tensor on a waveform/module so generated integration grids inherit
    the user's dtype/device when possible.
    """
    if torch.is_tensor(waveform):
        return waveform

    if hasattr(waveform, "parameters"):
        for p in waveform.parameters(recurse=True):
            return p

    if hasattr(waveform, "buffers"):
        for b in waveform.buffers(recurse=True):
            return b

    if hasattr(waveform, "__dict__"):
        for value in vars(waveform).values():
            if torch.is_tensor(value):
                return value

    return torch.empty((), dtype=torch.get_default_dtype())


def _make_time_grid(tstart, tstop, dt, ref: torch.Tensor, include_endpoint=False):
    """
    Build a 1-D time grid on ref.device/ref.dtype.

    The default is endpoint-exclusive, matching the usual fixed-step simulator
    convention: t = tstart, tstart + dt, ..., < tstop.
    """
    tstart = float(tstart)
    tstop = float(tstop)
    dt = float(dt)

    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if tstop < tstart:
        raise ValueError("tstop must be greater than or equal to tstart.")

    if include_endpoint:
        # Small tolerance keeps ordinary decimal dt values from missing the end.
        n = int(torch.floor(torch.tensor((tstop - tstart) / dt + 1e-12)).item()) + 1
    else:
        n = int(torch.ceil(torch.tensor((tstop - tstart) / dt - 1e-12)).item())

    return tstart + dt * torch.arange(n, device=ref.device, dtype=ref.dtype)


def _integrate_square_last_dim(y: torch.Tensor, t: torch.Tensor, method="trapezoid"):
    """
    Integrate y**2 over the last dimension using t as the time vector.
    """
    if y.shape[-1] != t.shape[-1]:
        raise ValueError(
            "The waveform output must use time as its last dimension; "
            f"got waveform shape {tuple(y.shape)} and t shape {tuple(t.shape)}."
        )

    if t.numel() < 2:
        return torch.zeros(y.shape[:-1], device=y.device, dtype=y.dtype)

    y2 = y.square()
    dt = t.diff()

    method = method.lower()
    if method in {"trapezoid", "trapz"}:
        return (0.5 * (y2[..., :-1] + y2[..., 1:]) * dt).sum(dim=-1)
    if method in {"left", "riemann", "rectangle"}:
        return (y2[..., :-1] * dt).sum(dim=-1)
    if method == "right":
        return (y2[..., 1:] * dt).sum(dim=-1)

    raise ValueError("method must be 'trapezoid', 'left', or 'right'.")


def energy(
    waveform,
    t: torch.Tensor = None,
    *,
    tstart=0.0,
    tstop=None,
    dt=None,
    include_endpoint=False,
    time_scale=1e-3,
    resistance=None,
    mode="current",
    method="trapezoid",
):
    """
    Compute waveform energy by integrating the squared waveform over time.

    This function is intentionally numerical rather than analytic. It therefore
    works for scalar sin/cos, multitone sin/cos, batched multitone sin/cos, and
    any other Waveform whose output has time on the last dimension.

    Parameters
    ----------
    waveform : callable
        Waveform/module to evaluate. For example, ``sin(...)`` or ``cos(...)``.
    t : torch.Tensor, optional
        Explicit time vector. If omitted, ``tstart``, ``tstop``, and ``dt`` are
        used to construct one. Times should use the same unit as the waveform's
        delay/off/frequency convention, usually milliseconds in dendra.
    tstart : float, optional
        Start time for an internally constructed grid. Default is 0.0.
    tstop : float, optional
        Stop time for an internally constructed grid. Required if ``t`` is not
        supplied. The default grid is endpoint-exclusive.
    dt : float, optional
        Time step for an internally constructed grid. Required if ``t`` is not
        supplied.
    include_endpoint : bool, optional
        Include ``tstop`` in the constructed grid. Default is False.
    time_scale : float, optional
        Multiplier converting the supplied time unit to seconds. For ms, use
        ``1e-3``. For seconds, use ``1.0``. Default is ``1e-3``.
    resistance : float or torch.Tensor, optional
        If provided, convert the normalized integral to physical energy using
        ``mode``.
    mode : {'current', 'voltage'}, optional
        Unit interpretation. With ``resistance=None``, the return value is just
        ``integral waveform(t)^2 dt_seconds``. If waveform is current in amps,
        this has units J/Ohm. If waveform is voltage in volts, this has units
        J*Ohm. With resistance supplied, ``mode='current'`` returns
        ``R * integral I^2 dt`` in joules, while ``mode='voltage'`` returns
        ``integral V^2/R dt`` in joules.
    method : {'trapezoid', 'left', 'right'}, optional
        Quadrature method. Default is trapezoid.

    Returns
    -------
    torch.Tensor
        Energy integrated over the last/time dimension. For a scalar waveform
        this is a scalar tensor. For a batched waveform with output ``[B, T]``,
        this has shape ``[B]``.

    Examples
    --------
    >>> w = sin(amp=1.0, freq=5.0, off_after=10.0)
    >>> e_norm = energy(w, tstop=10.0, dt=0.005)  # integral I^2 dt_seconds

    >>> w = sin(amp=torch.ones(4, 3), freq=torch.ones(4, 3) * 5.0)
    >>> e_norm = energy(w, tstop=1.0, dt=0.005)  # shape [4]
    """
    if t is None:
        if tstop is None or dt is None:
            raise ValueError(
                "Provide either an explicit t vector or both tstop and dt."
            )
        ref = _first_tensor_from_waveform(waveform)
        t = _make_time_grid(tstart, tstop, dt, ref, include_endpoint=include_endpoint)
    else:
        ref = _first_tensor_from_waveform(waveform)
        t = _as_tensor_like(t, ref)

    t_seconds = t * _as_tensor_like(time_scale, t)
    y = waveform(t)

    e = _integrate_square_last_dim(y, t_seconds, method=method)

    if resistance is not None:
        R = _as_tensor_like(resistance, e)
        mode = mode.lower()
        if mode == "current":
            e = e * R
        elif mode == "voltage":
            e = e / R
        else:
            raise ValueError("mode must be 'current' or 'voltage'.")

    return e


class sin(Waveform):
    """
    Sinusoidal waveform generator.

    Generates either a single sinusoid or a sum of sinusoidal oscillator
    components. The waveform is zero before each component's delay and after
    each component's effective off time.

    Parameters
    ----------
    amp : float or torch.Tensor, optional
        Amplitude in the desired output units (for example, mA for a current
        waveform). Default is 1.0.
    freq : float or torch.Tensor, optional
        Frequency in kHz (equivalently, ms⁻¹). For example, use
        ``100 * dendra.units.Hz`` for 100 Hz. Default is 1.0 kHz.
    phase : float or torch.Tensor, optional
        Phase offset in radians. Default is 0.0.
    delay : float or torch.Tensor, optional
        Start time in ms. Default is 0.0.
    off : float or torch.Tensor, optional
        Absolute off time in ms. Default is infinity.
    off_after : float or torch.Tensor, optional
        Off time relative to delay, in ms. The effective off time is
        ``min(off, delay + off_after)``. Default is infinity.
    tau : float or torch.Tensor, optional
        Sigmoid temperature in ms for differentiable delay/off edges. The
        forward pass uses a hard gate with soft straight-through gradients.
        Default is 0.01 ms.

    Notes
    -----
    For ``amp``, ``freq``, ``phase``, ``delay``, ``off``, ``off_after``, and
    ``tau``:

    - scalar: one oscillator component, output shape ``[T]``.
    - ``[K]``: ``K`` oscillator components, summed to output shape ``[T]``.
    - ``[B, K]``: ``B`` batches of ``K`` oscillator components, summed over
      components to output shape ``[B, T]``.
    - ``[..., K, T]`` or ``[..., K, 1]``: explicit time axis; the component
      axis is the dimension immediately before time, and the output shape is
      ``[..., T]``.

    Scalars and singleton dimensions broadcast over components and batches.
    For example, ``delay`` with shape ``[B, 1]`` gives one delay per batch,
    shared by all components in that batch.

    For component ``k``:

    .. math::
        y_k(t) = a_k g_k(t) \\sin(2\\pi f_k (t - d_k) + \\phi_k)

    and the returned waveform is ``sum_k y_k(t)``.

    Examples
    --------
    >>> import torch
    >>> import dendra as dn
    >>> from dendra.units import Hz, ms
    >>> t = torch.linspace(0, 100 * ms, 1000)
    >>> waveform = dn.sin(amp=2.0, freq=10 * Hz)
    >>> values = waveform(t)  # [T]

    >>> waveform = dn.sin(
    ...     amp=torch.tensor([1.0, 0.5, 0.25]),
    ...     freq=torch.tensor([5.0, 7.0, 11.0]) * Hz,
    ... )
    >>> values = waveform(t)  # [T], sum of 3 components

    >>> waveform = dn.sin(
    ...     amp=torch.ones(4, 3),
    ...     freq=torch.tensor([[5.0, 7.0, 11.0]]).expand(4, 3) * Hz,
    ... )
    >>> values = waveform(t)  # [4, T], 4 batched multitone waveforms
    """

    Waveform.PARAMETER(
        amp=1.0,
        freq=1.0,
        phase=0.0,
        delay=0.0,
        off=torch.inf,
        off_after=torch.inf,
        tau=0.01,
    )

    def fn(self, t):
        return _trig_oscillator_sum(
            t=t,
            amp=self.amp,
            freq=self.freq,
            phase=self.phase,
            delay=self.delay,
            off=self.off,
            off_after=self.off_after,
            tau=self.tau,
            trig=torch.sin,
        )


class cos(Waveform):
    """
    Cosine waveform generator.

    Generates either a single cosine or a sum of cosine oscillator components.
    The waveform is zero before each component's delay and after each
    component's effective off time.

    Parameters
    ----------
    amp : float or torch.Tensor, optional
        Amplitude in the desired output units (for example, mA for a current
        waveform). Default is 1.0.
    freq : float or torch.Tensor, optional
        Frequency in kHz (equivalently, ms⁻¹). For example, use
        ``100 * dendra.units.Hz`` for 100 Hz. Default is 1.0 kHz.
    phase : float or torch.Tensor, optional
        Phase offset in radians. Default is 0.0.
    delay : float or torch.Tensor, optional
        Start time in ms. Default is 0.0.
    off : float or torch.Tensor, optional
        Absolute off time in ms. Default is infinity.
    off_after : float or torch.Tensor, optional
        Off time relative to delay, in ms. The effective off time is
        ``min(off, delay + off_after)``. Default is infinity.
    tau : float or torch.Tensor, optional
        Sigmoid temperature in ms for differentiable delay/off edges. The
        forward pass uses a hard gate with soft straight-through gradients.
        Default is 0.01 ms.

    Notes
    -----
    For ``amp``, ``freq``, ``phase``, ``delay``, ``off``, ``off_after``, and
    ``tau``:

    - scalar: one oscillator component, output shape ``[T]``.
    - ``[K]``: ``K`` oscillator components, summed to output shape ``[T]``.
    - ``[B, K]``: ``B`` batches of ``K`` oscillator components, summed over
      components to output shape ``[B, T]``.
    - ``[..., K, T]`` or ``[..., K, 1]``: explicit time axis; the component
      axis is the dimension immediately before time, and the output shape is
      ``[..., T]``.

    Scalars and singleton dimensions broadcast over components and batches.
    For example, ``delay`` with shape ``[B, 1]`` gives one delay per batch,
    shared by all components in that batch.

    For component ``k``:

    .. math::
        y_k(t) = a_k g_k(t) \\cos(2\\pi f_k (t - d_k) + \\phi_k)

    and the returned waveform is ``sum_k y_k(t)``.

    Examples
    --------
    >>> import torch
    >>> import dendra as dn
    >>> from dendra.units import Hz, ms
    >>> t = torch.linspace(0, 100 * ms, 1000)
    >>> waveform = dn.cos(amp=2.0, freq=10 * Hz)
    >>> values = waveform(t)  # [T]

    >>> waveform = dn.cos(
    ...     amp=torch.tensor([1.0, 0.5, 0.25]),
    ...     freq=torch.tensor([5.0, 7.0, 11.0]) * Hz,
    ... )
    >>> values = waveform(t)  # [T], sum of 3 components

    >>> waveform = dn.cos(
    ...     amp=torch.ones(4, 3),
    ...     freq=torch.tensor([[5.0, 7.0, 11.0]]).expand(4, 3) * Hz,
    ... )
    >>> values = waveform(t)  # [4, T], 4 batched multitone waveforms
    """

    Waveform.PARAMETER(
        amp=1.0,
        freq=1.0,
        phase=0.0,
        delay=0.0,
        off=torch.inf,
        off_after=torch.inf,
        tau=0.01,
    )

    def fn(self, t):
        return _trig_oscillator_sum(
            t=t,
            amp=self.amp,
            freq=self.freq,
            phase=self.phase,
            delay=self.delay,
            off=self.off,
            off_after=self.off_after,
            tau=self.tau,
            trig=torch.cos,
        )


class mono_rect(Waveform):
    """
    Monophasic rectangular pulse waveform generator.

    Generates a single rectangular pulse with configurable amplitude,
    delay, and pulse width (duration). The waveform is zero outside the pulse
    duration.

    Parameters
    ----------
    amp : float, optional
        Amplitude of the rectangular pulse. Default is -1.0.
    delay : float, optional
        Time delay before the pulse starts in ms. Default is 0.0.
    pw : float, optional
        Width of the pulse in ms. Default is 1.0.
    tau : float, optional
        Sigmoid temperature for differentiable edges in ms. Default is 0.1.

    Notes
    -----
    The waveform is defined as:

    .. math::
        f(t) =
        \\begin{cases}
        \\text{amp} & \\text{if } \\text{delay} \\leq t \\leq \\text{delay} + \\text{duration} \\\\
        0 & \\text{otherwise}
        \\end{cases}

    Examples
    --------
    >>> import torch
    >>> import dendra as dn
    >>> waveform = dn.mono_rect(amp=-2.0, pw=0.5)
    >>> t = torch.linspace(0, 2, 100)
    >>> values = waveform(t)
    """

    Waveform.PARAMETER(amp=1.0, delay=0.0, pw=1.0, tau=0.1)

    def fn(self, t):
        amp = _time_broadcast_param(self.amp, t)
        delay = _time_broadcast_param(self.delay, t)
        pw = _time_broadcast_param(self.pw, t)
        tau = _time_broadcast_param(self.tau, t)

        gate = _rect_gate(t, delay, delay + pw, tau)
        return amp * gate


class bi_rect(Waveform):
    """
    Biphasic rectangular pulse waveform generator.

    Generates a two-phase rectangular pulse with configurable amplitudes,
    pulse widths, delay, and inter-phase interval. The waveform is zero
    outside the pulse durations.

    Parameters
    ----------
    amp1 : float, optional
        Amplitude of the first phase. Default is -1.0.
    amp2 : float, optional
        Amplitude of the second phase. Default is 1.0.
    delay : float, optional
        Time delay before the pulse starts in ms. Default is 0.0.
    pw1 : float, optional
        Pulse width of the first phase in ms. Default is 1.0.
    pw2 : float, optional
        Pulse width of the second phase in ms. Default is 1.0.
    interval : float, optional
        Time interval between the two phases in ms. Default is 0.0.
    tau : float, optional
        Sigmoid temperature for differentiable edges in ms. Default is 0.1.

    Notes
    -----
    The waveform is defined as:

    .. math::
        f(t) =
        \\begin{cases}
        \\text{amp1} & \\text{if } \\text{delay} \\leq t \\leq \\text{delay} + \\text{pw1} \\\\
        \\text{amp2} & \\text{if } \\text{delay} + \\text{pw1} + \\text{interval} \\leq t \\leq \\text{delay} + \\text{pw1} + \\text{interval} + \\text{pw2} \\\\
        0 & \\text{otherwise}
        \\end{cases}

    Examples
    --------
    >>> import torch
    >>> import dendra as dn
    >>> waveform = dn.bi_rect(amp1=-2.0, amp2=1.0, pw1=0.5, pw2=1.0)
    >>> t = torch.linspace(0, 3, 100)
    >>> values = waveform(t)
    """

    Waveform.PARAMETER(
        amp1=-1.0,
        amp2=1.0,
        delay=0.0,
        pw1=1.0,
        pw2=1.0,
        interval=0.0,
        tau=0.1,
    )

    def fn(self, t):
        amp1 = _time_broadcast_param(self.amp1, t)
        amp2 = _time_broadcast_param(self.amp2, t)
        delay = _time_broadcast_param(self.delay, t)
        pw1 = _time_broadcast_param(self.pw1, t)
        pw2 = _time_broadcast_param(self.pw2, t)
        interval = _time_broadcast_param(self.interval, t)
        tau = _time_broadcast_param(self.tau, t)

        t1_start = delay
        t1_stop = delay + pw1
        t2_start = t1_stop + interval
        t2_stop = t2_start + pw2

        gate1 = _rect_gate(t, t1_start, t1_stop, tau, inclusive_stop=True)
        gate2 = _rect_gate(t, t2_start, t2_stop, tau, inclusive_stop=True)

        return amp1 * gate1 + amp2 * gate2


class bi_rect_balanced(Waveform):
    """
    Charge-balanced biphasic rectangular pulse waveform generator.

    Generates a two-phase rectangular pulse where the second phase amplitude
    is automatically adjusted to maintain charge balance based on the pulse widths.
    The waveform is zero outside the pulse durations.

    Parameters
    ----------
    amp : float, optional
        Amplitude of the first phase. Default is 1.0.
    delay : float, optional
        Time delay before the pulse starts in ms. Default is 0.0.
    pw1 : float, optional
        Pulse width of the first phase in ms. Default is 1.0.
    pw2 : float, optional
        Pulse width of the second phase in ms. Default is 1.0.
    interval : float, optional
        Time interval between the two phases in ms. Default is 0.0.
    tau : float, optional
        Sigmoid temperature for differentiable edges in ms. Default is 0.1.

    Notes
    -----
    The waveform is defined as:

    .. math::
        f(t) =
        \\begin{cases}
        \\text{amp} & \\text{if } \\text{delay} \\leq t \\leq \\text{delay} + \\text{pw1} \\\\
        -\\text{amp} \\cdot \\frac{\\text{pw1}}{\\text{pw2}} & \\text{if } \\text{delay} + \\text{pw1} + \\text{interval} \\leq t \\leq \\text{delay} + \\text{pw1} + \\text{interval} + \\text{pw2} \\\\
        0 & \\text{otherwise}
        \\end{cases}

    The amplitude of the second phase is scaled to ensure charge balance, where:

    .. math::
        \\text{amp2} = -\\text{amp} \\cdot \\frac{\\text{pw1}}{\\text{pw2}}

    Examples
    --------
    >>> import torch
    >>> import dendra as dn
    >>> waveform = dn.bi_rect_balanced(amp=2.0, pw1=0.5, pw2=1.0)
    >>> t = torch.linspace(0, 3, 100)
    >>> values = waveform(t)
    """

    Waveform.PARAMETER(amp=1.0, delay=0.0, pw1=1.0, pw2=1.0, interval=0.0, tau=0.1)

    def fn(self, t):
        amp = _time_broadcast_param(self.amp, t)
        delay = _time_broadcast_param(self.delay, t)
        pw1 = _time_broadcast_param(self.pw1, t)
        pw2 = _time_broadcast_param(self.pw2, t)
        interval = _time_broadcast_param(self.interval, t)
        tau = _time_broadcast_param(self.tau, t)

        t1_start = delay
        t1_stop = delay + pw1
        t2_start = t1_stop + interval
        t2_stop = t2_start + pw2

        gate1 = _rect_gate(t, t1_start, t1_stop, tau, inclusive_stop=True)
        gate2 = _rect_gate(t, t2_start, t2_stop, tau, inclusive_stop=True)

        pw2_safe = torch.clamp(pw2, min=1e-12)
        amp2 = -amp * (pw1 / pw2_safe)

        return amp * gate1 + amp2 * gate2


class bi_rect_symm(Waveform):
    """
    Symmetric biphasic rectangular pulse waveform generator.

    Generates a two-phase rectangular pulse with equal but opposite amplitudes
    and identical pulse widths. The waveform is zero outside the pulse durations.

    Parameters
    ----------
    amp : float, optional
        Amplitude of the first phase. Default is 1.0.
    delay : float, optional
        Time delay before the pulse starts in ms. Default is 0.0.
    pw : float, optional
        Pulse width for each phase in ms. Default is 1.0.
    interval : float, optional
        Time interval between the two phases in ms. Default is 0.0.
    tau : float, optional
        Sigmoid temperature for differentiable edges in ms. Default is 0.1.

    Notes
    -----
    The waveform is defined as:

    .. math::
        f(t) =
        \\begin{cases}
        \\text{amp} & \\text{if } \\text{delay} \\leq t \\leq \\text{delay} + \\text{pw} \\\\
        -\\text{amp} & \\text{if } \\text{delay} + \\text{pw} + \\text{interval} \\leq t \\leq \\text{delay} + 2\\text{pw} + \\text{interval} \\\\
        0 & \\text{otherwise}
        \\end{cases}

    This waveform is charge-balanced by design due to the equal duration and
    opposite amplitude of the two phases.

    Examples
    --------
    >>> import torch
    >>> import dendra as dn
    >>> waveform = dn.bi_rect_symm(amp=2.0, pw=0.5, interval=0.1)
    >>> t = torch.linspace(0, 3, 100)
    >>> values = waveform(t)
    """

    Waveform.PARAMETER(amp=1.0, delay=0.0, pw=1.0, interval=0.0, tau=0.1)

    def fn(self, t):
        amp = _time_broadcast_param(self.amp, t)
        delay = _time_broadcast_param(self.delay, t)
        pw = _time_broadcast_param(self.pw, t)
        interval = _time_broadcast_param(self.interval, t)
        tau = _time_broadcast_param(self.tau, t)

        t1_start = delay
        t1_stop = delay + pw
        t2_start = t1_stop + interval
        t2_stop = t2_start + pw

        gate1 = _rect_gate(t, t1_start, t1_stop, tau, inclusive_stop=True)
        gate2 = _rect_gate(t, t2_start, t2_stop, tau, inclusive_stop=True)

        return amp * gate1 - amp * gate2


class arbitrary(Waveform):
    """
    Arbitrary waveform generator using linear interpolation.

    Generates a waveform by linearly interpolating between specified amplitude
    values at given time points. Values outside the specified time range are
    set to zero.

    Parameters
    ----------
    values : list or torch.Tensor, optional
        List of amplitude values at specified time points. Default is [0.0, 0.0].
    tpoints : list or torch.Tensor, optional
        List of time points in ms corresponding to amplitude values. Default is [0.0, 1.0].

    Notes
    -----
    The waveform is defined by linear interpolation between the specified points.
    For a time point t, if t is within the range of tpoints, the value is linearly interpolated;
    if t is outside the range of tpoints, the value is 0.

    Examples
    --------
    >>> import torch
    >>> import dendra as dn
    >>> # Create a triangular pulse
    >>> waveform = dn.arbitrary(tpoints=[0.0, 0.5, 1.0], values=[0.0, 1.0, 0.0])
    >>> t = torch.linspace(0, 1.5, 100)
    >>> values = waveform(t)
    """

    Waveform.PARAMETER(values=[0.0, 0.0], tpoints=[0.0, 1.0])

    def fn(self, t):
        if self.values.ndim > 1:
            t = t.unsqueeze(0)
            t = t.expand(self.values.shape[0], -1)
        return interp1d(self.tpoints, self.values, t)


class constant(Waveform):
    """
    Constant waveform generator.

    Generates a constant value over time.

    Parameters
    ----------
    value : float, optional
        The constant value to generate. Default is 0.0.

    Notes
    -----
    The waveform is defined as:

    .. math::
        f(t) = \\text{value}

    Examples
    --------
    >>> import torch
    >>> import dendra as dn
    >>> waveform = dn.constant(value=5.0)
    >>> t = torch.linspace(0, 1, 100)
    >>> values = waveform(t)
    """

    Waveform.PARAMETER(value=0.0)

    def fn(self, t):
        value = _time_broadcast_param(self.value, t)
        return value + torch.zeros_like(t)
