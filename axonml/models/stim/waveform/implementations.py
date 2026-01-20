import torch

from axonml.utils import interp1d

from .core import Waveform

__all__ = [
    "sin",
    "cos",
    "mono_rect",
    "bi_rect",
    "bi_rect_balanced",
    "bi_rect_symm",
    "arbitrary",
]


def _rect_gate(t, start, stop, tau, inclusive_stop=False):
    tau = torch.clamp(tau, min=1e-6)
    soft = torch.sigmoid((t - start) / tau) * torch.sigmoid((stop - t) / tau)
    if inclusive_stop:
        hard = ((t >= start) & (t <= stop)).to(soft.dtype)
    else:
        hard = ((t >= start) & (t < stop)).to(soft.dtype)
    # Straight-through gate: hard in forward, soft for gradients.
    return hard + (soft - soft.detach())


class sin(Waveform):
    """
    Sinusoidal waveform generator.

    Generates a sine wave with configurable amplitude, frequency, phase,
    and delay. The waveform is zero before the specified delay time.

    Parameters
    ----------
    amp : float, optional
        Amplitude of the sine wave. Default is 1.0.
    freq : float, optional
        Frequency of the sine wave in kHz. Default is 1.0.
    phase : float, optional
        Phase offset in radians. Default is 0.0.
    delay : float, optional
        Time delay before the waveform starts in ms. Default is 0.0.
    off : float, optional
        Time at which the waveform turns off in ms. Default is infinity.
    off_after : float, optional
        Time at which waveform turns of after delay. Default is infinity.

    Notes
    -----
    The waveform is defined as:

    .. math::
        f(t) =
        \\begin{cases}
        \\text{amp} \\cdot \\sin(2\\pi \\cdot \\text{freq} \\cdot (t - \\text{delay}) + \\text{phase}) & \\text{if } t \\geq \\text{delay} \\\\
        0 & \\text{otherwise}
        \\end{cases}

    Examples
    --------
    >>> import torch
    >>> import axonml as ax
    >>> waveform = ax.sin(amp=2.0, freq=10.0)
    >>> t = torch.linspace(0, 1, 100)
    >>> values = waveform(t)
    """

    Waveform.PARAMETER(
        amp=1.0, freq=1.0, phase=0.0, delay=0.0, off=torch.inf, off_after=torch.inf
    )

    def fn(self, t):
        off = torch.minimum(self.off, self.off_after + self.delay)
        w = torch.sin(2 * torch.pi * self.freq * (t - self.delay) + self.phase)
        return self.amp * torch.where((t >= self.delay) & (t < off), w, 0.0)


class cos(Waveform):
    """
    Cosine waveform generator.

    Generates a cosine wave with configurable amplitude, frequency, phase,
    and delay. The waveform is zero before the specified delay time.

    Parameters
    ----------
    amp : float, optional
        Amplitude of the cosine wave. Default is 1.0.
    freq : float, optional
        Frequency of the cosine wave in kHz. Default is 1.0.
    phase : float, optional
        Phase offset in radians. Default is 0.0.
    delay : float, optional
        Time delay before the waveform starts in ms. Default is 0.0.
    off : float, optional
        Time at which the waveform turns off in ms. Default is infinity.
    off_after : float, optional
        Time at which waveform turns of after delay. Default is infinity.

    Notes
    -----
    The waveform is defined as:

    .. math::
        f(t) =
        \\begin{cases}
        \\text{amp} \\cdot \\cos(2\\pi \\cdot \\text{freq} \\cdot (t - \\text{delay}) + \\text{phase}) & \\text{if } t \\geq \\text{delay} \\\\
        0 & \\text{otherwise}
        \\end{cases}

    Examples
    --------
    >>> import torch
    >>> import axonml as ax
    >>> waveform = ax.cos(amp=2.0, freq=10.0)
    >>> t = torch.linspace(0, 1, 100)
    >>> values = waveform(t)
    """

    Waveform.PARAMETER(
        amp=1.0, freq=1.0, phase=0.0, delay=0.0, off=torch.inf, off_after=torch.inf
    )

    def fn(self, t):
        off = torch.minimum(self.off, self.off_after + self.delay)
        w = torch.cos(2 * torch.pi * self.freq * (t - self.delay) + self.phase)
        return self.amp * torch.where((t >= self.delay) & (t < off), w, 0.0)


class mono_rect(Waveform):
    """
    Monophasic rectangular pulse waveform generator.

    Generates a single rectangular pulse with configurable amplitude,
    delay, and duration. The waveform is zero outside the pulse duration.

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
    >>> impotr axonml as ax
    >>> waveform = ax.mono_rect(amp=-2.0, duration=0.5)
    >>> t = torch.linspace(0, 2, 100)
    >>> values = waveform(t)
    """

    Waveform.PARAMETER(amp=1.0, delay=0.0, pw=1.0, tau=0.1)

    def fn(self, t):
        gate = _rect_gate(t, self.delay, self.delay + self.pw, self.tau)
        return self.amp * gate


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
    >>> import axonml as ax
    >>> waveform = ax.bi_rect(amp1=-2.0, amp2=1.0, pw1=0.5, pw2=1.0)
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
        t1_start = self.delay
        t1_stop = self.delay + self.pw1
        t2_start = self.delay + self.pw1 + self.interval
        t2_stop = t2_start + self.pw2
        gate1 = _rect_gate(t, t1_start, t1_stop, self.tau, inclusive_stop=True)
        gate2 = _rect_gate(t, t2_start, t2_stop, self.tau, inclusive_stop=True)
        return self.amp1 * gate1 + self.amp2 * gate2


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
    >>> import axonml as ax
    >>> waveform = ax.bi_rect_balanced(amp=2.0, pw1=0.5, pw2=1.0)
    >>> t = torch.linspace(0, 3, 100)
    >>> values = waveform(t)
    """

    Waveform.PARAMETER(amp=1.0, delay=0.0, pw1=1.0, pw2=1.0, interval=0.0, tau=0.1)

    def fn(self, t):
        t1_start = self.delay
        t1_stop = self.delay + self.pw1
        t2_start = self.delay + self.pw1 + self.interval
        t2_stop = t2_start + self.pw2
        gate1 = _rect_gate(t, t1_start, t1_stop, self.tau, inclusive_stop=True)
        gate2 = _rect_gate(t, t2_start, t2_stop, self.tau, inclusive_stop=True)
        amp2 = -self.amp / (self.pw1 / self.pw2)
        return self.amp * gate1 + amp2 * gate2


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
    >>> import axonml as ax
    >>> waveform = ax.bi_rect_symm(amp=2.0, pw=0.5, interval=0.1)
    >>> t = torch.linspace(0, 3, 100)
    >>> values = waveform(t)
    """

    Waveform.PARAMETER(amp=1.0, delay=0.0, pw=1.0, interval=0.0, tau=0.1)

    def fn(self, t):
        t1_start = self.delay
        t1_stop = self.delay + self.pw
        t2_start = self.delay + self.pw + self.interval
        t2_stop = self.delay + 2 * self.pw + self.interval
        gate1 = _rect_gate(t, t1_start, t1_stop, self.tau, inclusive_stop=True)
        gate2 = _rect_gate(t, t2_start, t2_stop, self.tau, inclusive_stop=True)
        return self.amp * gate1 - self.amp * gate2


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
    >>> import axonml as ax
    >>> # Create a triangular pulse
    >>> waveform = ax.arbitrary(tpoints=[0.0, 0.5, 1.0], values=[0.0, 1.0, 0.0])
    >>> t = torch.linspace(0, 1.5, 100)
    >>> values = waveform(t)
    """

    Waveform.PARAMETER(values=[0.0, 0.0], tpoints=[0.0, 1.0])

    def fn(self, t):
        t = t.unsqueeze(0)
        if self.values.ndim > 1:
            t = t.expand(self.values.shape[0], -1)
        return interp1d(self.tpoints, self.values, t)
