from contextlib import ContextDecorator

import torch

from dendra.helpers import DEBUG

from ..parametric import to_param
from ._materials import Material

# default reversal potentials from NEURON
REVERSAL = {"ena": 50.0, "ek": -77.0, "eca": 132.0}

VALENCES = {"na": 1.0, "k": 1.0, "ca": 2.0}

# default initial concentrations from NEURON
CINIT = {
    "nao0": 140.0,
    "nai0": 10.0,
    "ko0": 2.5,
    "ki0": 54.4,
    "cao0": 2.0,
    "cai0": 5e-5,
}

MIN_CONCENTRATION = {
    "na": 1e-12,
    "k": 1e-12,
    "ca": 1e-12,
}


def valid_ions():
    global VALENCES
    return list(VALENCES.keys())


def valid_concentrations():
    global CINIT
    return [s[:-1] for s in CINIT.keys()]


def reversals():
    global REVERSAL
    return REVERSAL


def cinits():
    global CINIT
    return CINIT


def min_concentrations():
    global MIN_CONCENTRATION
    return MIN_CONCENTRATION


def set_min_concentration(ion, min_concentration):
    """
    Set the minimum concentration for a specific ion.

    Parameters
    ----------
    ion : str
        The name of the ion (e.g., 'na', 'k', 'ca').
    min_concentration : float
        The minimum concentration of the ion (in mM).
    """
    global MIN_CONCENTRATION
    MIN_CONCENTRATION[ion] = min_concentration


def register_ion(ion, valence, e, i0, o0, min_concentration=None):
    """
    Register a new ion species with its properties.

    Parameters
    ----------
    ion : str
        The name of the ion (e.g., 'na', 'k', 'ca').
    valence : float
        The valence of the ion (e.g., 1.0 for Na+, 2.0 for Ca2+).
    e : float
        The reversal potential (in mV) for the ion.
    i0 : float
        The initial intracellular concentration of the ion (in mM).
    o0 : float
        The initial extracellular concentration of the ion (in mM).
    min_concentration : float, optional
        The minimum concentration of the ion (in mM). If None, a default value is used.
    """
    global VALENCES
    global REVERSAL
    global CINIT
    global MIN_CONCENTRATION
    VALENCES[ion] = valence
    REVERSAL[f"e{ion}"] = e
    CINIT[f"{ion}o0"] = o0
    CINIT[f"{ion}i0"] = i0
    min_concentration = min_concentration if min_concentration is not None else 1e-12
    MIN_CONCENTRATION[ion] = min_concentration


def ion_register(ion, valence, e, i0, o0, min_concentration=None):
    """Backward-compatible alias for :func:`register_ion`."""
    return register_ion(ion, valence, e, i0, o0, min_concentration)


class equilibria(ContextDecorator):
    _last = {}

    def __init__(self, use_last=False, **kwargs):
        global REVERSAL
        not_in_reversal = [k for k in kwargs if k not in REVERSAL]
        if not_in_reversal:
            raise ValueError(f"Reversal potential not found: {not_in_reversal}")
        self.updates = kwargs
        if not use_last:
            equilibria._last = self.updates
            if DEBUG:
                print(equilibria._last)
        if use_last:
            self.updates.update(equilibria._last)
            if DEBUG:
                print(self.updates)
            equilibria._last = {}
        self.original_values = {}

    def __enter__(self):
        global REVERSAL
        # Store original values of the keys to be updated
        self.original_values = {k: REVERSAL[k] for k in self.updates if k in REVERSAL}
        # Update the global dictionary
        REVERSAL.update(self.updates)

    def __exit__(self, exc_type, exc_value, traceback):
        global REVERSAL
        # Restore original values
        REVERSAL.update(self.original_values)
        return False  # Propagate exceptions if any


class concentrations(ContextDecorator):
    _last = {}

    def __init__(self, use_last=False, **kwargs):
        global CINIT
        not_in_cinit = [k for k in kwargs if k not in CINIT]
        if not_in_cinit:
            raise ValueError(f"Initial concentration not found: {not_in_cinit}")
        self.updates = kwargs
        if not use_last:
            concentrations._last = self.updates
            if DEBUG:
                print(concentrations._last)
        if use_last:
            self.updates.update(concentrations._last)
            if DEBUG:
                print(self.updates)
            concentrations._last = {}
        self.original_values = {}

    def __enter__(self):
        global CINIT
        # Store original values of the keys to be updated
        self.original_values = {k: CINIT[k] for k in self.updates if k in CINIT}
        # Update the global dictionary
        CINIT.update(self.updates)

    def __exit__(self, exc_type, exc_value, traceback):
        global CINIT
        # Restore original values
        CINIT.update(self.original_values)
        return False  # Propagate exceptions if any


R = 1e3 * 8.31446261815324
FARADAY = 96485.33212331001


def _is_scalar(x):
    if isinstance(x, float):
        return True
    if isinstance(x, int):
        return True
    return torch.is_tensor(x) and x.ndim == 0


def _make_into_shape(shape, value):
    if _is_scalar(value):
        return torch.full(shape, value)
    else:
        return value.expand(shape).clone()


class Ion(Material):
    __constants__ = "init_e_reversal", "advance_e"

    def __init__(self, name, shape, einit, eadvance):
        # Do not assign Parameters before torch.nn.Module.__init__ has run.
        # Material.__init__ calls Module.__init__, so pass plain defaults into
        # the Material field specs first, then register learnable/init values.
        e0 = reversals()[f"e{name}"]
        i0 = cinits()[f"{name}i0"]
        o0 = cinits()[f"{name}o0"]
        min_concentration = min_concentrations()[name]

        super().__init__(
            name,
            shape,
            fields={
                f"i{name}": 0.0,
                f"e{name}": e0,
                f"{name}i": i0,
                f"{name}o": o0,
            },
            min_values={
                f"{name}i": min_concentration,
                f"{name}o": min_concentration,
            },
        )

        self.rzf = R / (VALENCES[name] * FARADAY)
        self.e_init = to_param(e0)
        self.i_init = to_param(i0)
        self.o_init = to_param(o0)
        self.min_concentration = min_concentration

        self.init_e_reversal = einit != 0
        self.advance_e = eadvance != 0

    @staticmethod
    def _expand_init_like(value, like):
        if torch.is_tensor(value):
            value_t = value.to(device=like.device, dtype=like.dtype)
        else:
            value_t = torch.as_tensor(value, device=like.device, dtype=like.dtype)
        if value_t.ndim == 0 or value_t.numel() == 1:
            return value_t.reshape(()).expand_as(like).clone()
        return value_t.expand_as(like).clone()

    def initialize(self, celsius) -> None:
        # Ion initialization should use e_init/i_init/o_init so that explicit
        # equilibria()/concentrations() values, including trainable parameter
        # declarations handled by to_param, remain the source of truth.
        name = self.name
        i_buf = self._buffers[f"i{name}"]
        self._buffers[f"i{name}"] = torch.zeros_like(i_buf)
        self._buffers[f"e{name}"] = self._expand_init_like(
            self.e_init, self._buffers[f"e{name}"]
        )
        self._buffers[f"{name}i"] = self._expand_init_like(
            self.i_init, self._buffers[f"{name}i"]
        )
        self._buffers[f"{name}o"] = self._expand_init_like(
            self.o_init, self._buffers[f"{name}o"]
        )
        self.einit(celsius)
        if not self.training:
            self.detach()

    def detach(self):
        super().detach()
        return self

    def einit(self, celsius) -> None:
        if self.init_e_reversal:
            name = self.name
            iono = self._buffers[f"{name}o"]
            ioni = self._buffers[f"{name}i"]
            self._buffers[f"e{name}"] = (
                torch.log(iono / ioni) * self.rzf * (273.15 + celsius)
            )

    def advance(self, celsius) -> None:
        # Clamp intracellular/extracellular concentrations through Material.advance.
        super().advance(celsius)

        if not self.advance_e:
            return

        name = self.name
        iono = self._buffers[f"{name}o"]
        ioni = self._buffers[f"{name}i"]
        self._buffers[f"e{name}"] = (
            torch.log(iono / ioni) * self.rzf * (273.15 + celsius)
        )
