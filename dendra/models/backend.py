"""Global backend state for Dendra simulations."""

import torch


class __Backend:
    """Singleton container for Dendra runtime configuration.

    Attributes
    ----------
    dt : float
        Simulation time-step in milliseconds. Defaults to ``0.005``.
    device : str
        Default torch device, ``'cuda'`` when available, otherwise ``'cpu'``.
    """

    __defaults__ = {
        "dt": 0.005,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    _instance = None  # Keep instance reference

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        for k, v in self.__defaults__.items():
            setattr(self, f"_{k}", v)

    @property
    def dt(self):
        """Get the global simulation time-step.

        Returns
        -------
        float
            Time-step in milliseconds.
        """
        return self._dt

    @dt.setter
    def dt(self, value):
        """Set the global simulation time-step.

        Parameters
        ----------
        value : float
            Time-step in milliseconds.
        """
        self._dt = value

    @property
    def device(self):
        """Get the default torch device.

        Returns
        -------
        str
            Device identifier, for example ``'cpu'`` or ``'cuda'``.
        """
        return self._device

    @device.setter
    def device(self, value):
        """Set the global torch device.

        Parameters
        ----------
        value : str
            Device identifier, for example ``'cpu'`` or ``'cuda'``.
        """
        self._device = value


Backend = __Backend()
