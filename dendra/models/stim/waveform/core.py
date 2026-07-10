import math
from numbers import Integral, Number
from typing import Optional

import numpy as np
import torch

from dendra.models.parametric import SimpleParameterized


class Waveform(SimpleParameterized):
    """
    Base class for creating waveform generators.

    This abstract class provides the foundation for implementing various types of
    waveforms (like sine waves, rectangular pulses, etc.) that can be used for
    neural stimulation. It inherits from torch.nn.Module.

    Parameters
    ----------
    **kwargs : dict
        Parameter values to initialize the waveform. These are passed to
        the Parameterized class's instantiate_parameters method. These parameters
        should be defined in subclasses using the PARAMETER class method. These parameters
        can be scalars or tensors, allowing for flexible waveform definitions. They will
        be automatically broadcasted to match the shape of the input time tensor `t`
        when the waveform is evaluated. All parameters are accessible as attributes of the waveform instance.

    Notes
    -----
    When subclassing Waveform, you need to:
        1. Define parameters using the PARAMETER class method
        2. Implement the fn(t) method to define the waveform's behavior

    Examples
    --------
    Creating a custom triangular wave:

    >>> import torch
    >>> from dendra.models.stim.waveform.core import Waveform
    >>>
    >>> class triangle(Waveform):
    ...     '''Triangular waveform generator'''
    ...     Waveform.PARAMETER(amp=1.0, freq=1.0, delay=0.0)
    ...
    ...     def fn(self, t):
    ...         t_adjusted = t - self.delay
    ...         period = 1.0 / self.freq
    ...         phase = torch.fmod(t_adjusted, period) / period
    ...         tri = 2 * torch.abs(2 * phase - 1) - 1
    ...         return self.amp * torch.where(t >= self.delay, tri, 0.0)
    >>>
    >>> # Using the custom waveform
    >>> waveform = triangle(amp=2.0, freq=5.0)
    >>> t = torch.linspace(0, 1, 100)
    >>> values = waveform(t)

    Waveform arithmetic:

    You can perform arithmetic operations with waveforms. For example, you can add two waveforms together or add a constant to a waveform:

    >>> w1 = triangle(amp=1.0, freq=2.0)
    >>> w2 = triangle(amp=0.5, freq=2.0)
    >>> w_sum = w1 + w2
    >>> w_const = w1 + 3.0

    This will create new waveform instances representing the sum of the two waveforms and the waveform with a constant added, respectively.
    Arithmetic operations supported include addition (+), subtraction (-), multiplication (*), and division (/).

    """

    def __init__(self, **kwargs):
        self.check_kwargs(kwargs)
        super(Waveform, self).__init__(**kwargs)

    def expand(self, shape):
        for p in self.parameters():
            if p.dim() == 0:
                pass
            else:
                p.data = p.data.expand(shape)
        return self

    def reshape_for_intra(self):
        if getattr(self, "_reshaped_for_intra", False):
            return self
        for p in self.parameters():
            if p.dim() == 0:
                pass
            else:
                p.data = p.data.unsqueeze(0)
        self._reshaped_for_intra = True
        return self

    def fn(self, t):
        r"""Core waveform implementation. Must be implemented by subclasses.

        .. note::
            Although the waveform recipe needs to be defined within
            this function, one should call the :class:`Waveform` instance
            afterwards instead of this since the former takes care of
            running any registered hooks while the latter silently ignores them.
        """
        raise NotImplementedError

    def forward(self, t):
        return self.fn(torch.as_tensor(t))

    def repeat(self, freq: float, delay: float = 0.0, off: float = torch.inf):
        """Return a periodically repeating copy of *this* waveform.

        Parameters
        ----------
        freq : float
            Frequency of repetition in kHz.
        delay : float, optional
            Delay before the first repetition in ms. Default is 0.0.
        off : float, optional
            Time after which the waveform stops repeating in ms. Default is infinity.

        Returns
        -------
        _repeat
            A waveform that repeats periodically according to the specified parameters.
        """
        return _repeat(self, freq, delay, off)

    def poisson(
        self,
        interval: float,
        n: Optional[int] = 10,
        start: float = 0.0,
        noise: float = 1.0,
        off: float = torch.inf,
        randomize_every_call: bool = False,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ):
        """Return a Poisson-distributed copy of *this* waveform.

        Parameters
        ----------
        interval : float
            Mean interval between events in ms.
        n : int, optional
            Number of events to generate. Default is 10.
        start : float, optional
            Start time in ms. Default is 0.0.
        noise : float, optional
            Noise factor for interval variability. Default is 1.0.
        off : float, optional
            Time after which the waveform stops. Default is infinity.
        randomize_every_call : bool, optional
            If True, generates a new Poisson schedule on each call. Default is False.
        generator : Optional[torch.Generator], optional
            A PyTorch random number generator for reproducibility. Default is None.

        Returns
        -------
        _poisson
            A waveform that follows a Poisson distribution according to the specified parameters.
        """
        return _poisson(
            self,
            interval,
            n,
            start,
            noise,
            off,
            randomize_every_call,
            generator,
            **kwargs,
        )

    def assemble(self, start, end, dt):
        ref = next(self.parameters(), None)
        if ref is None:
            ref = next(self.buffers(), torch.empty(()))
        t = torch.arange(start, end, dt, device=ref.device, dtype=ref.dtype)
        return self(t)

    def assemble_chunked(self, dt, chunks):
        ref = next(self.parameters(), None)
        if ref is None:
            ref = next(self.buffers(), torch.empty(()))
        t = torch.arange(0, self._tstop, dt, device=ref.device, dtype=ref.dtype)
        t = torch.tensor_split(t, chunks)
        for t_ in t:
            yield self(t_)

    # ----- + and - -----
    def __add__(self, other):
        if isinstance(other, Waveform):
            return Sum(self, other)
        elif isinstance(other, Number):
            return Sum(self, Constant(other))
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Waveform):
            return Sum(self, other, scale=[1.0, -1.0])
        elif isinstance(other, Number):
            return Sum(self, Constant(other), scale=[1.0, -1.0])
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, Number):
            return Sum(Constant(other), self, scale=[1.0, -1.0])
        return NotImplemented

    # ----- * and / -----
    def __mul__(self, other):
        if isinstance(other, Waveform):
            return Product(self, other)
        elif isinstance(other, Number):
            return Product(self, gain=float(other))
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Waveform):
            return Product(self, Reciprocal(other))
        elif isinstance(other, Number):
            return Product(self, gain=1.0 / float(other))
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, Number):
            return Product(Constant(other), Reciprocal(self))
        return NotImplemented

    def __neg__(self):
        return (-1.0) * self


# --- Primitives -------------------------------------------------------


class Constant(Waveform):
    def __init__(self, c: float):
        super().__init__()
        self.c = float(c)

    def fn(self, t):
        # Multiplication promotes integer time grids to the default floating
        # dtype instead of truncating fractional constants as full_like does.
        return torch.ones_like(t) * self.c

    def __repr__(self):
        return f"Constant({self.c})"


class Reciprocal(Waveform):
    """Represents 1 / wf."""

    def __init__(self, wf: Waveform, eps: float = 1e-12):
        super().__init__()
        self.wf = wf
        self.eps = float(eps)  # optional stabilizer if you ever want it

    def fn(self, t):
        return 1.0 / (self.wf.fn(t) + self.eps)

    def __repr__(self):
        if self.eps == 0.0:
            return f"Reciprocal({repr(self.wf)})"
        return f"Reciprocal({repr(self.wf)}, eps={self.eps})"


def _repeat_broadcast_param(x, t: torch.Tensor) -> torch.Tensor:
    """
    Canonicalize repeat controls so sweep axes broadcast over trailing time.

    For repeat parameters, 1-D and 2-D tensors are interpreted as sweep axes,
    not time-varying vectors. Higher-rank tensors may provide an explicit
    trailing time axis.
    """
    if not torch.is_tensor(x):
        x = t.new_tensor(x)
    if x.ndim == 0 or t.ndim == 0:
        return x
    if x.ndim in (1, 2):
        return x.unsqueeze(-1)
    T = t.shape[-1]
    if x.shape[-1] in (1, T):
        return x
    return x.unsqueeze(-1)


class _repeat(Waveform):
    Waveform.PARAMETER(freq=1.0, delay=0.0, off=torch.inf)

    def __init__(
        self, waveform, freq: float, delay: float = 0.0, off: float = torch.inf
    ):
        freq_value = torch.as_tensor(freq).detach()
        if not torch.isfinite(freq_value).all() or torch.any(freq_value <= 0):
            raise ValueError("Repeat frequency must be finite and positive.")
        super(_repeat, self).__init__(freq=freq, delay=delay, off=off)
        self.waveform = waveform

    def fn(self, t):
        freq = _repeat_broadcast_param(self.freq, t)
        delay = _repeat_broadcast_param(self.delay, t)
        off = _repeat_broadcast_param(self.off, t)

        t_adjusted = t - delay
        mask = (t >= delay) & (t < off)
        t_periodic = torch.fmod(t_adjusted, 1.0 / freq)
        return torch.where(mask, self.waveform.fn(t_periodic), 0.0)

    def __repr__(self):
        return f"Repeat({self.waveform}, freq={self.freq}, delay={self.delay})"


class Sum(Waveform):
    def __init__(self, *waveforms, scale=None):
        super().__init__()

        if scale is None:
            scale = [1.0] * len(waveforms)
        else:
            if len(scale) != len(waveforms):
                raise ValueError("Scale length must match number of waveforms.")

        flat_wfs = []
        flat_scales = []

        def _add(wf, s):
            if isinstance(wf, Sum):
                for child, child_s in zip(wf.waveforms, wf.scale):
                    _add(child, s * child_s)
            else:
                flat_wfs.append(wf)
                flat_scales.append(float(s))

        for wf, s in zip(waveforms, scale):
            _add(wf, s)

        self.waveforms = torch.nn.ModuleList(flat_wfs)
        self.scale = flat_scales

    def fn(self, t):
        out = None
        for s, wf in zip(self.scale, self.waveforms):
            term = s * wf.fn(t)
            out = term if out is None else out + term
        return out

    def __repr__(self):
        parts = [f"{s}*{repr(wf)}" for s, wf in zip(self.scale, self.waveforms)]
        return "Sum(" + ", ".join(parts) + ")"

    # Scale-aware overrides make scalar ops cheaper than building Product
    def __mul__(self, other):
        if isinstance(other, Number):
            return Sum(*self.waveforms, scale=[other * s for s in self.scale])
        return super().__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Number):
            inv = 1.0 / float(other)
            return Sum(*self.waveforms, scale=[inv * s for s in self.scale])
        return super().__truediv__(other)


# --- Product (auto-flatten + gain) -----------------------------------


class Product(Waveform):
    """
    Product of factors with an overall scalar gain.
    - Auto-flattens nested Product.
    - Scalars are absorbed into `gain`.
    """

    def __init__(self, *waveforms, gain: float = 1.0):
        super().__init__()
        flat = []
        total_gain = float(gain)

        def _add(wf):
            nonlocal total_gain
            if isinstance(wf, Product):
                # absorb child's gain and flatten its factors
                total_gain *= wf.gain
                for child in wf.waveforms:
                    _add(child)
            elif isinstance(wf, Constant):
                # Constant factor can fold into gain
                total_gain *= wf.c
            else:
                flat.append(wf)

        for wf in waveforms:
            if isinstance(wf, Number):
                total_gain *= float(wf)
            else:
                _add(wf)

        self.gain = float(total_gain)
        self.waveforms = torch.nn.ModuleList(flat)

    def fn(self, t):
        result = torch.ones_like(t) * self.gain
        for wf in self.waveforms:
            result = result * wf.fn(t)
        return result

    def __repr__(self):
        parts = [repr(wf) for wf in self.waveforms]
        head = f"{self.gain}*" if self.gain != 1.0 else ""
        return f"Product({head}{', '.join(parts)})"

    # Cheap scalar tweaks
    def __mul__(self, other):
        if isinstance(other, Number):
            return Product(*self.waveforms, gain=self.gain * float(other))
        return super().__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Number):
            return Product(*self.waveforms, gain=self.gain / float(other))
        return super().__truediv__(other)


class _poisson(Waveform):
    """
    Wrap a single-waveform generator so that it is emitted at
    irregular, Poisson-distributed onset times.

    Parameters
    ----------
    waveform : Waveform
        The (single-shot) wave shape to replicate - e.g. a
        rectangular pulse, a biphasic stim, …
    interval : float
        Mean inter-spike-interval Δt  [ms].
        (Think 10 ms  ⇒ 100 Hz mean rate.)
    n : Optional[int]
        Optional upper bound on the number of spikes.  If given,
        generation stops after this many onsets even if `off`
        has not been reached.
    start : float
        Most-likely time of the first spike [ms].
    noise : float ∈ [0,1]
        0 → perfectly periodic (Δt = interval every time)
        1 → pure Poisson (Δt ~Exp(rate=1/interval))
        values in between give a convex mixture:
            Δt = (1-noise)*interval + noise*Exp(...)
    off : float
        Do not schedule spikes at or beyond this time [ms].
    randomize_every_call : bool
        If True, generate a new Poisson schedule on each call to fn(t).
        Otherwise, the schedule is generated once at init time and
        re-used on every call.
    generator : Optional[torch.Generator]
        A PyTorch random number generator for reproducibility.
        If None, the default generator is used.
    """

    def __init__(
        self,
        waveform: Waveform,
        interval: float,
        n: Optional[int] = None,
        start: float = 0.0,
        noise: float = 1.0,
        off: float = torch.inf,
        randomize_every_call: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()  # <- no kwargs
        self.waveform = waveform
        self.interval = float(interval)
        if not math.isfinite(self.interval) or self.interval <= 0.0:
            raise ValueError("Poisson interval must be finite and positive.")
        if n is not None:
            if isinstance(n, bool) or not isinstance(n, Integral):
                raise TypeError("Poisson n must be a non-negative integer or None.")
            if n < 0:
                raise ValueError("Poisson n must be a non-negative integer or None.")
            n = int(n)
        self.n = n
        self.start = float(start)
        self.noise = float(noise)
        self.off = float(off)
        if not math.isfinite(self.start):
            raise ValueError("Poisson start must be finite.")
        if math.isnan(self.off):
            raise ValueError("Poisson off must not be NaN.")
        if not math.isfinite(self.noise) or not 0.0 <= self.noise <= 1.0:
            raise ValueError("Poisson noise must be between 0 and 1 inclusive.")
        self.randomize_every_call = bool(randomize_every_call)
        self.generator = generator or torch.default_generator
        self._spike_time_trailing_dims = 0
        if np.isinf(self.off) and self.n is None:
            raise ValueError(
                "Poisson schedule needs a finite `off` time or a finite `n` "
                "(number of spikes) to terminate."
            )
        self.register_buffer("_spike_times", self._make_schedule(self.generator))

    def reshape_for_intra(self):
        already_reshaped = getattr(self, "_reshaped_for_intra", False)
        super().reshape_for_intra()
        if not already_reshaped:
            self._spike_time_trailing_dims = 2
            self._spike_times = self._spike_times.reshape(
                self._spike_times.shape + (1, 1)
            )
        return self

    # ---------- helper ---------------------------------------------------
    def _next_dt(self, gen: torch.Generator):
        """Draw the next Δt according to noise parameter."""
        if self.noise == 0.0:
            # perfectly regular
            return self.interval
        # exponential sample (mean = interval)
        u = torch.rand((), generator=gen, device=gen.device)  # uniform (0,1)
        exp_sample = -u.log() * self.interval  # Exp(λ=1/interval)
        return (1.0 - self.noise) * self.interval + self.noise * exp_sample.item()

    def _make_schedule(self, gen: torch.Generator) -> torch.Tensor:
        """Generate all spike onset times once, store as buffer."""
        times = []
        t = self.start
        k = 0
        while t < self.off and (self.n is None or k < self.n):
            times.append(t)
            t += self._next_dt(gen)
            k += 1
        if not times:  # handle edge‑case: no spikes at all
            times.append(torch.inf)
        ref = next(self.waveform.parameters(), None)
        if ref is None:
            ref = next(self.waveform.buffers(), torch.empty(()))
        return torch.tensor(times, device=ref.device, dtype=ref.dtype)

    def regenerate_schedule_(self) -> None:
        schedule = self._make_schedule(self.generator)
        if hasattr(self, "_spike_times"):
            schedule = schedule.to(self._spike_times)
        if self._spike_time_trailing_dims:
            schedule = schedule.reshape(
                schedule.shape + (1,) * self._spike_time_trailing_dims
            )
        self._spike_times = schedule

    # ---------- core -----------------------------------------------------
    def fn(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sum a copy of `waveform` at every scheduled spike time.
        Assumes the wrapped waveform returns 0 for t<0 or t>duration.
        """
        if self.randomize_every_call:
            self.regenerate_schedule_()
        # Evaluate each onset independently so the spike axis can never alias a
        # population, compartment, oscillator-component, or other leading axis
        # of the wrapped waveform merely because their lengths happen to match.
        responses = [
            self.waveform.fn(t - onset.unsqueeze(-1))
            for onset in self._spike_times.unbind(dim=0)
        ]
        return torch.stack(responses, dim=0).sum(dim=0)

    def __repr__(self):
        return f"Poisson({self.waveform}, interval={self.interval}, noise={self.noise})"
