from typing import Optional

import torch
import numpy as np

from axonml.models.parametric import SimpleParameterized


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
        the Parameterized class's instantiate_parameters method.

    Attributes
    ----------
    Any attributes defined using the PARAMETER decorator in subclasses.

    Methods
    -------
    fn(t)
        Core implementation method that calculates the waveform value at time t.
        Must be implemented by subclasses.
    repeat(freq, delay=0.0, off=torch.inf)
        Creates a repeating version of the waveform. Frequency should be given in
        kHz and delay and off in ms. Off is the time after which the waveform stops repeating.

    Notes
    -----
    When subclassing Waveform, you need to:
    1. Define parameters using the PARAMETER decorator
    2. Implement the fn(t) method to define the waveform's behavior

    Examples
    --------
    Creating a custom triangular wave:

    >>> import torch
    >>> from axonml.models.stim.waveform.core import Waveform
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
    """

    def __init__(self, **kwargs):
        super(Waveform, self).__init__(**kwargs)

    def expand(self, shape):
        for p in self.parameters():
            if p.dim() == 0:
                pass
            else:
                p.data = p.data.expand(shape)
        return self

    def reshape_for_intra(self):
        for p in self.parameters():
            if p.dim() == 0:
                pass
            else:
                p.data = p.data.unsqueeze(0)
        return self

    def fn(self, t):
        raise NotImplementedError

    def forward(self, t):
        return self.fn(t)

    def repeat(self, freq: float, delay: float = 0.0, off: float = torch.inf):
        return _repeat(self, freq, delay, off)
    
    def poisson(
        self,
        interval: float,
        n: Optional[int] = 10,
        start: float = 0.0,
        noise: float = 1.0,
        off: float = torch.inf,
        **kwargs,
    ):
        """Return a Poisson-scheduled copy of *this* waveform."""
        return _poisson(self, interval, n, start, noise, off, **kwargs)

    def assemble(self, start, end, dt):
        t = torch.arange(start, end, dt, device=self.device())
        return self(t)

    def assemble_chunked(self, dt, chunks):
        t = torch.arange(0, self._tstop, dt, device=self.device())
        t = torch.tensor_split(t, chunks)
        for t_ in t:
            yield self(t_)


class _repeat(Waveform):
    def __init__(
        self, waveform, freq: float, delay: float = 0.0, off: float = torch.inf
    ):
        super(_repeat, self).__init__()
        self.waveform = waveform
        self.freq = freq
        self.delay = delay
        self.off = off

    def fn(self, t):
        t_adjusted = t - self.delay
        mask = (t >= self.delay) & (t < self.off)
        t_periodic = torch.fmod(t_adjusted, 1.0 / self.freq)
        return torch.where(mask, self.waveform.fn(t_periodic), 0.0)

    def __repr__(self):
        return f"Repeat({self.waveform}, freq={self.freq}, delay={self.delay})"


class Sum(Waveform):
    def __init__(self, *waveforms):
        super(Sum, self).__init__()
        self.waveforms = torch.nn.ModuleList(waveforms)

    def fn(self, t):
        return sum(waveform.fn(t) for waveform in self.waveforms)

    def __repr__(self):
        return f"Sum({', '.join(map(repr, self.waveforms))})"


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
    """
    def __init__(
        self,
        waveform: Waveform,
        interval: float,
        n: Optional[int] = None,
        start: float = 0.0,
        noise: float = 1.0,
        off: float = torch.inf,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()                        # <- no kwargs
        self.waveform = waveform
        self.interval = float(interval)
        self.n        = n
        self.start    = float(start)
        self.noise    = float(noise)
        self.off      = float(off)
        if np.isinf(self.off) and self.n is None:
            raise ValueError(
                "Poisson schedule needs a finite `off` time or a finite `n` "
                "(number of spikes) to terminate."
            )
        self.register_buffer(
            "_spike_times",
            self._make_schedule(generator or torch.default_generator)
        )

    def reshape_for_intra(self):
        super().reshape_for_intra()
        self._spike_times = self._spike_times.unsqueeze(-1).unsqueeze(-1)
        return self

    # ---------- helper ---------------------------------------------------
    def _next_dt(self, gen: torch.Generator):
        """Draw the next Δt according to noise parameter."""
        if self.noise == 0.0:
            # perfectly regular
            return self.interval
        # exponential sample (mean = interval)
        u = torch.rand((), generator=gen)   # uniform (0,1)
        exp_sample = -u.log() * self.interval              # Exp(λ=1/interval)
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
        if not times:            # handle edge‑case: no spikes at all
            times.append(torch.inf)
        return torch.tensor(times)

    # ---------- core -----------------------------------------------------
    def fn(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sum a copy of `waveform` at every scheduled spike time.
        Assumes the wrapped waveform returns 0 for t<0 or t>duration.
        """
        # broadcast: (#spikes, |t|)  – never moves _spike_times to CPU
        tt = t.unsqueeze(0) - self._spike_times.unsqueeze(-1)
        return self.waveform.fn(tt).sum(dim=0)

    def __repr__(self):
        return (f"Poisson({self.waveform},"
                f" interval={self.interval}, noise={self.noise})")
