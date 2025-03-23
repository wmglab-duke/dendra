from typing import Optional

import torch
from axonml.models.parametric import Parameterized


class Waveform(torch.jit.ScriptModule, Parameterized):
    """
    Base class for creating waveform generators.

    This abstract class provides the foundation for implementing various types of
    waveforms (like sine waves, rectangular pulses, etc.) that can be used for
    neural stimulation. It inherits from PyTorch's ScriptModule for JIT compilation
    and Parameterized for parameter management.

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
    forward(t)
        Evaluates the waveform at given time points. Calls fn() internally.
    repeat(freq)
        Creates a repeating version of the waveform.
        Not implemented in the base class.

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
    >>> from axonml.models.declarations import PARAMETER
    >>>
    >>> class triangle(Waveform):
    ...     '''Triangular waveform generator'''
    ...     PARAMETER(amp=1.0, freq=1.0, delay=0.0)
    ...
    ...     def fn(self, t):
    ...         t_adjusted = t - self.delay
    ...         period = 1.0 / self.freq
    ...         # Create sawtooth wave and then take absolute value
    ...         phase = torch.fmod(t_adjusted, period) / period
    ...         tri = 2 * torch.abs(2 * phase - 1) - 1
    ...         return self.amp * torch.where(t >= self.delay, tri, 0.0)
    >>>
    >>> # Using the custom waveform
    >>> waveform = triangle(amp=2.0, freq=5.0)
    >>> t = torch.linspace(0, 1, 100)
    >>> values = waveform(t)
    """

    _tstop: Optional[float]

    def __init__(self, **kwargs):
        super(Waveform, self).__init__()
        self._tstop = None
        self.check_kwargs(kwargs)
        self.instantiate_parameters(**kwargs)

    def fn(self, t):
        raise NotImplementedError

    def forward(self, t):
        return torch.atleast_2d(self.fn(torch.as_tensor(t)))

    def repeat(self, freq: float, delay: float = 0.0):
        return _repeat(self, freq, delay)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.parameters_repr()})"

    def parameters_repr(self):
        return ", ".join(f"{k}={v}" for k, v in self.named_parameters())

    def tstop(self, tstop):
        self._tstop = tstop
        return self

    def assemble(self, dt):
        t = torch.arange(0, self._tstop, dt)
        return self(t)


class _repeat(Waveform):
    def __init__(self, waveform, freq: float, delay: float = 0.0):
        super(_repeat, self).__init__()
        self.waveform = waveform
        self.freq = freq
        self.delay = delay

    def fn(self, t):
        t_adjusted = t - self.delay
        mask = t >= self.delay
        t_periodic = torch.fmod(t_adjusted, 1.0 / self.freq)
        return torch.where(mask, self.waveform.fn(t_periodic), 0.0)

    def __repr__(self):
        return f"Repeat({self.waveform}, freq={self.freq}, delay={self.delay})"


class Sum(Waveform):
    def __init__(self, *waveforms):
        super(Sum, self).__init__()
        self.waveforms = waveforms

    def fn(self, t):
        return sum(waveform.fn(t) for waveform in self.waveforms)

    def __repr__(self):
        return f"Sum({', '.join(map(repr, self.waveforms))})"
