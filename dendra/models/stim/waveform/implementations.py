import torch

from dendra.utils import interp1d

from ._gates import _as_tensor_like, _rect_gate, _time_broadcast_param
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
    """Sinusoidal waveform generator.

    Generates either a single sinusoid or a sum of sinusoidal oscillator
    components. Each component is active on the half-open interval
    ``delay <= t < min(off, delay + off_after)`` and zero elsewhere.

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
    - For rank three or higher, a final dimension of length ``1`` or ``T``
      is an explicit time axis. The preceding axis contains components, which
      are summed, leaving output shape ``[..., T]``.
    - Otherwise a time axis is appended and the final parameter axis contains
      components. Include an explicit time axis to avoid ambiguity when the
      component count equals ``T``.

    A trailing singleton is therefore not an additional batch axis. For one
    component over two leading axes, use parameter shape ``[B, N, 1, 1]``;
    ``[B, N, 1]`` instead treats ``N`` as the component axis and sums it.
    These conventions describe evaluation on a one-dimensional time grid.

    Timing parameters use surrogate gradients while the forward waveform keeps
    abrupt edges. ``delay`` moves both the onset gate and the carrier phase;
    when ``off_after`` determines the stop, ``delay`` moves that edge too.
    Only the earlier of ``off`` and ``delay + off_after`` receives the stop
    gradient; equal finite cutoffs split it equally. Initialize a cutoff to a
    finite value to optimize it. An infinite cutoff has zero gradient.
    Enable gradients on the waveform parameters to be optimized, for example
    ``waveform.delay.requires_grad_(True)``. Samples near an edge supply its
    gradient, so a ``tau`` much smaller than the sampling interval can make
    edge contributions numerically negligible.

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

    FUNCTIONAL_PURE = True

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
    """Cosine waveform generator.

    Generates either a single cosine or a sum of cosine oscillator components.
    Each component is active on the half-open interval
    ``delay <= t < min(off, delay + off_after)`` and zero elsewhere.

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
    - For rank three or higher, a final dimension of length ``1`` or ``T``
      is an explicit time axis. The preceding axis contains components, which
      are summed, leaving output shape ``[..., T]``.
    - Otherwise a time axis is appended and the final parameter axis contains
      components. Include an explicit time axis to avoid ambiguity when the
      component count equals ``T``.

    A trailing singleton is therefore not an additional batch axis. For one
    component over two leading axes, use parameter shape ``[B, N, 1, 1]``;
    ``[B, N, 1]`` instead treats ``N`` as the component axis and sums it.
    These conventions describe evaluation on a one-dimensional time grid.

    Timing parameters use surrogate gradients while the forward waveform keeps
    abrupt edges. ``delay`` moves both the onset gate and the carrier phase;
    when ``off_after`` determines the stop, ``delay`` moves that edge too.
    Only the earlier of ``off`` and ``delay + off_after`` receives the stop
    gradient; equal finite cutoffs split it equally. Initialize a cutoff to a
    finite value to optimize it. An infinite cutoff has zero gradient.
    Enable gradients on the waveform parameters to be optimized, for example
    ``waveform.delay.requires_grad_(True)``. Samples near an edge supply its
    gradient, so a ``tau`` much smaller than the sampling interval can make
    edge contributions numerically negligible.

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

    FUNCTIONAL_PURE = True

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
    """A rectangular pulse on the half-open interval ``[delay, delay + pw)``.

    Parameters
    ----------
    amp : float or Tensor, default 1.0
        Pulse amplitude in the desired output units, such as mA for current.
    delay : float or Tensor, default 0.0
        Pulse onset in milliseconds.
    pw : float or Tensor, default 1.0
        Pulse width in milliseconds.
    tau : float or Tensor, default 0.1
        Sigmoid temperature in milliseconds for surrogate edge gradients.

    Notes
    -----
    The forward value is ``amp`` when ``delay <= t < delay + pw`` and zero
    elsewhere. The forward edges remain abrupt; backward evaluation uses the
    sigmoid gate, with ``tau`` clamped to at least 1e-6 ms.

    For a one-dimensional time grid of length ``T``, scalar parameters produce
    ``[T]``. Non-scalar parameters gain a final time axis unless they already end
    in a singleton dimension. That trailing ``1`` is the explicit time axis even
    for tensors of rank three or higher. For example, ``[B, N, C, 1]`` retains
    all three leading axes and produces ``[B, N, C, T]``, including when ``C=1``.
    Parameter dimensions broadcast together before evaluation.

    Examples
    --------
    >>> import torch
    >>> import dendra as dn
    >>> waveform = dn.mono_rect(amp=-2.0, pw=0.5)
    >>> values = waveform(torch.linspace(0, 2, 100))
    """

    FUNCTIONAL_PURE = True

    Waveform.PARAMETER(amp=1.0, delay=0.0, pw=1.0, tau=0.1)

    def fn(self, t):
        amp = _time_broadcast_param(self.amp, t)
        delay = _time_broadcast_param(self.delay, t)
        pw = _time_broadcast_param(self.pw, t)
        tau = _time_broadcast_param(self.tau, t)

        gate = _rect_gate(t, delay, delay + pw, tau)
        return amp * gate


class bi_rect(Waveform):
    """Two rectangular phases with independently configured amplitudes and widths.

    Parameters
    ----------
    amp1 : float or Tensor, default -1.0
        First-phase amplitude in the desired output units.
    amp2 : float or Tensor, default 1.0
        Second-phase amplitude in the same units.
    delay : float or Tensor, default 0.0
        First-phase onset in milliseconds.
    pw1 : float or Tensor, default 1.0
        First-phase width in milliseconds.
    pw2 : float or Tensor, default 1.0
        Second-phase width in milliseconds.
    interval : float or Tensor, default 0.0
        Gap between the first-phase stop and second-phase start, in milliseconds.
    tau : float or Tensor, default 0.1
        Sigmoid temperature in milliseconds for surrogate edge gradients.

    Notes
    -----
    The phases occupy ``[delay, delay + pw1]`` and
    ``[delay + pw1 + interval, delay + pw1 + interval + pw2]``. Both endpoints
    are included and the phase contributions add wherever they overlap. Thus
    ``interval=0`` returns ``amp1 + amp2`` at the shared boundary.

    Parameter broadcasting and the hard-forward/surrogate-gradient edge contract
    follow :class:`~dendra.models.stim.waveform.mono_rect`. In particular, a
    trailing singleton parameter dimension is an explicit time axis.

    Examples
    --------
    >>> import dendra as dn
    >>> waveform = dn.bi_rect(amp1=-2.0, amp2=1.0, pw1=0.5, pw2=1.0)
    """

    FUNCTIONAL_PURE = True

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
    """Biphasic rectangular pulse with equal and opposite continuous phase areas.

    Parameters
    ----------
    amp : float or Tensor, default 1.0
        First-phase amplitude in the desired output units.
    delay : float or Tensor, default 0.0
        First-phase onset in milliseconds.
    pw1 : float or Tensor, default 1.0
        First-phase width in milliseconds.
    pw2 : float or Tensor, default 1.0
        Second-phase width in milliseconds. Use positive widths for charge balance.
    interval : float or Tensor, default 0.0
        Gap between the phases in milliseconds.
    tau : float or Tensor, default 0.1
        Sigmoid temperature in milliseconds for surrogate edge gradients.

    Notes
    -----
    The first phase has amplitude ``amp`` and the second has amplitude
    ``-amp * pw1 / max(pw2, 1e-12)``. The phases occupy the same inclusive
    intervals as :class:`~dendra.models.stim.waveform.bi_rect`; contributions
    add at overlapping endpoints. Charge balance describes the continuous pulse
    areas for widths above the denominator's 1e-12 ms floor, rather than an
    exact sum on every sampled grid.

    Parameter broadcasting and surrogate gradients follow
    :class:`~dendra.models.stim.waveform.mono_rect`, including the explicit
    trailing singleton time axis.

    Examples
    --------
    >>> import dendra as dn
    >>> waveform = dn.bi_rect_balanced(amp=2.0, pw1=0.5, pw2=1.0)
    """

    FUNCTIONAL_PURE = True

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
    """Biphasic rectangular pulse with equal widths and opposite amplitudes.

    Parameters
    ----------
    amp : float or Tensor, default 1.0
        First-phase amplitude; the second phase has amplitude ``-amp``.
    delay : float or Tensor, default 0.0
        First-phase onset in milliseconds.
    pw : float or Tensor, default 1.0
        Width of each phase in milliseconds.
    interval : float or Tensor, default 0.0
        Gap between the phases in milliseconds.
    tau : float or Tensor, default 0.1
        Sigmoid temperature in milliseconds for surrogate edge gradients.

    Notes
    -----
    The phases occupy ``[delay, delay + pw]`` and
    ``[delay + pw + interval, delay + 2*pw + interval]``. Both endpoints are
    included; opposite contributions cancel at the shared boundary when
    ``interval=0``. The continuous phase areas are equal and opposite.

    Parameter broadcasting and surrogate gradients follow
    :class:`~dendra.models.stim.waveform.mono_rect`, including the explicit
    trailing singleton time axis.

    Examples
    --------
    >>> import dendra as dn
    >>> waveform = dn.bi_rect_symm(amp=2.0, pw=0.5, interval=0.1)
    """

    FUNCTIONAL_PURE = True

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
    """Piecewise-linear waveform through supplied time/amplitude samples.

    Parameters
    ----------
    values : array-like or Tensor, optional
        Amplitudes at the interpolation knots. Shape ``[Q]`` or ``[B, Q]``;
        defaults to ``[0.0, 0.0]``.
    tpoints : array-like or Tensor, optional
        Corresponding knot times in milliseconds, sorted in ascending order.
        Shape ``[Q]`` or ``[B, Q]``; defaults to ``[0.0, 1.0]``. Use at least
        two distinct knots. One shared row can broadcast over amplitude rows.

    Notes
    -----
    Values outside the knot range are zero; values at the endpoints are retained.
    A one-dimensional evaluation grid of length ``T`` produces ``[T]`` for one
    unbatched table, or ``[B, T]`` for batched tables. Knot values and times must
    have matching final dimensions and broadcastable row counts.

    Gradients propagate through amplitudes, knot times, and query times within
    the selected interpolation segments. Segment selection is discrete; preserve
    sorted knot order when optimizing times.

    Examples
    --------
    >>> import dendra as dn
    >>> waveform = dn.arbitrary(tpoints=[0.0, 0.5, 1.0], values=[0.0, 1.0, 0.0])
    """

    FUNCTIONAL_PURE = True

    Waveform.PARAMETER(values=[0.0, 0.0], tpoints=[0.0, 1.0])

    def fn(self, t):
        if self.values.ndim > 1:
            t = t.unsqueeze(0)
            t = t.expand(self.values.shape[0], -1)
        # The uniform-grid shortcut treats spacing as structural, so its
        # derivative with respect to trainable tpoints is not the derivative of
        # general interpolation. Retain that fast path only for ordinary eager
        # evaluation in which no tpoint derivative can be recorded.
        transformed = (
            torch.compiler.is_compiling() or torch._C._are_functorch_transforms_active()
        )
        if not transformed:
            # Direct make_fx scan capture sets neither flag above.
            from torch.fx.experimental.proxy_tensor import get_proxy_mode

            transformed = get_proxy_mode() is not None
        if transformed:
            differentiate_tpoints = True
        else:
            tangent = torch.autograd.forward_ad.unpack_dual(self.tpoints).tangent
            differentiate_tpoints = (
                torch.is_grad_enabled() and self.tpoints.requires_grad
            ) or tangent is not None
        uniform = "never" if differentiate_tpoints else "auto"
        return interp1d(self.tpoints, self.values, t, uniform=uniform)


class constant(Waveform):
    """Constant amplitude broadcast over the evaluation time grid.

    Parameters
    ----------
    value : float or Tensor, default 0.0
        Output amplitude, in units determined by the stimulation target.

    Notes
    -----
    For a one-dimensional time grid of length ``T``, a scalar produces ``[T]``.
    A non-scalar value gains a final time axis unless it already ends in a
    singleton dimension. That trailing ``1`` is an explicit time axis even for
    rank-three or higher values: ``[B, N, C, 1]`` produces ``[B, N, C, T]``.
    The value is not interpreted as time-varying because an axis happens to have
    length ``T``.

    Examples
    --------
    >>> import dendra as dn
    >>> waveform = dn.constant(value=5.0)
    """

    FUNCTIONAL_PURE = True

    Waveform.PARAMETER(value=0.0)

    def fn(self, t):
        value = _time_broadcast_param(self.value, t)
        return value + torch.zeros_like(t)
