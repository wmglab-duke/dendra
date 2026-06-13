from contextlib import ContextDecorator
from types import MethodType

import torch

from dendra.helpers import DEBUG

from ..parametric import to_param

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


def _make_like(reference, value):
    """Materialize ``value`` with the same shape/device/dtype as ``reference``."""
    if torch.is_tensor(value):
        return (
            value.to(device=reference.device, dtype=reference.dtype)
            .expand_as(reference)
            .clone()
        )
    return torch.full_like(reference, float(value))


def _clamp_ion_concentrations(iono_t, ioni_t, min_concentration_t):
    """Replace non-positive concentrations without rebinding modules."""
    min_val = min_concentration_t.to(device=iono_t.device, dtype=iono_t.dtype)
    return (
        torch.where(iono_t <= 0, min_val, iono_t),
        torch.where(ioni_t <= 0, min_val, ioni_t),
    )


def _nernst_potential(iono_t, ioni_t, celsius, rzf_t):
    """Compute the Nernst reversal potential using tensor-valued scalar constants."""
    rzf = rzf_t.to(device=iono_t.device, dtype=iono_t.dtype)
    celsius = torch.as_tensor(celsius, device=iono_t.device, dtype=iono_t.dtype)
    return torch.log(iono_t / ioni_t) * rzf * (273.15 + celsius)


def _safe_impl_suffix(name):
    return "".join(ch if ch.isalnum() else "_" for ch in str(name)) or "ion"


_ION_ADVANCE_IMPL_CACHE = {}
_ION_EINIT_IMPL_CACHE = {}


def _make_ion_advance_impl(name, advance_e):
    """Create a per-ion advance method with literal buffer names.

    TorchDynamo caches by Python code object. The previous Ion.advance used one
    shared code object for Na/K/Ca and reached different ``nn.Module.__setattr__``
    branches for names such as ``nao`` and ``ko``. This factory gives each ion
    layout a distinct, monomorphic code object and updates ``_buffers`` directly.
    """
    name = str(name)
    advance_e = bool(advance_e)
    key = (name, advance_e)
    if key in _ION_ADVANCE_IMPL_CACHE:
        return _ION_ADVANCE_IMPL_CACHE[key]

    suffix = _safe_impl_suffix(name)
    fn_name = f"_advance_ion_{suffix}_{int(advance_e)}"
    iono_name = f"{name}o"
    ioni_name = f"{name}i"
    e_name = f"e{name}"

    src = f"""
def {fn_name}(self, celsius):
    iono_t = self._buffers[{iono_name!r}]
    ioni_t = self._buffers[{ioni_name!r}]
    iono_t, ioni_t = _clamp_ion_concentrations(
        iono_t,
        ioni_t,
        self._buffers[\"_min_concentration_t\"],
    )
    self._buffers[{iono_name!r}] = iono_t
    self._buffers[{ioni_name!r}] = ioni_t
"""
    if advance_e:
        src += f"""    self._buffers[{e_name!r}] = _nernst_potential(
        iono_t,
        ioni_t,
        celsius,
        self._buffers[\"_rzf_t\"],
    )
"""
    src += """    return None
"""

    namespace = {
        "_clamp_ion_concentrations": _clamp_ion_concentrations,
        "_nernst_potential": _nernst_potential,
    }
    exec(src, namespace)
    impl = namespace[fn_name]
    impl.__module__ = __name__
    globals()[fn_name] = impl
    _ION_ADVANCE_IMPL_CACHE[key] = impl
    return impl


def _make_ion_einit_impl(name, init_e_reversal):
    """Create a per-ion initial reversal-potential method with literal names."""
    name = str(name)
    init_e_reversal = bool(init_e_reversal)
    key = (name, init_e_reversal)
    if key in _ION_EINIT_IMPL_CACHE:
        return _ION_EINIT_IMPL_CACHE[key]

    suffix = _safe_impl_suffix(name)
    fn_name = f"_einit_ion_{suffix}_{int(init_e_reversal)}"
    iono_name = f"{name}o"
    ioni_name = f"{name}i"
    e_name = f"e{name}"

    if init_e_reversal:
        src = f"""
def {fn_name}(self, celsius):
    self._buffers[{e_name!r}] = _nernst_potential(
        self._buffers[{iono_name!r}],
        self._buffers[{ioni_name!r}],
        celsius,
        self._buffers[\"_rzf_t\"],
    )
    return None
"""
    else:
        src = f"""
def {fn_name}(self, celsius):
    return None
"""

    namespace = {"_nernst_potential": _nernst_potential}
    exec(src, namespace)
    impl = namespace[fn_name]
    impl.__module__ = __name__
    globals()[fn_name] = impl
    _ION_EINIT_IMPL_CACHE[key] = impl
    return impl


class Ion(torch.nn.Module):
    __constants__ = (
        "name",
        "init_e_reversal",
        "advance_e",
        "_current_name",
        "_e_name",
        "_ioni_name",
        "_iono_name",
    )

    def __init__(self, name, shape, einit, eadvance):
        super().__init__()
        self.name = str(name)

        self._current_name = f"i{self.name}"
        self._e_name = f"e{self.name}"
        self._ioni_name = f"{self.name}i"
        self._iono_name = f"{self.name}o"

        self.rzf = R / (VALENCES[self.name] * FARADAY)

        self.e_init = to_param(reversals()[self._e_name])
        self.i_init = to_param(cinits()[f"{self.name}i0"])
        self.o_init = to_param(cinits()[f"{self.name}o0"])

        self.min_concentration = min_concentrations()[self.name]

        scalar_ref = torch.zeros((), dtype=torch.get_default_dtype())
        self.register_buffer(
            "_rzf_t",
            scalar_ref.new_tensor(self.rzf),
            persistent=False,
        )
        self.register_buffer(
            "_min_concentration_t",
            scalar_ref.new_tensor(self.min_concentration),
            persistent=False,
        )

        self.register_buffer(self._current_name, torch.zeros(shape))
        self.register_buffer(self._e_name, _make_into_shape(shape, self.e_init))
        self.register_buffer(self._ioni_name, _make_into_shape(shape, self.i_init))
        self.register_buffer(self._iono_name, _make_into_shape(shape, self.o_init))

        self.init_e_reversal = einit != 0
        self.advance_e = eadvance != 0

        # Shadow the generic class methods with per-ion generated methods. This
        # avoids one shared Ion.advance code object specializing alternately on
        # ``nao``, ``ko``, ``cao``, etc. during torch.compile.
        self.einit = MethodType(
            _make_ion_einit_impl(self.name, self.init_e_reversal), self
        )
        self.advance = MethodType(
            _make_ion_advance_impl(self.name, self.advance_e), self
        )

    def initialize(self, celsius) -> None:
        current = self._buffers[self._current_name]
        e = self._buffers[self._e_name]
        ioni = self._buffers[self._ioni_name]
        iono = self._buffers[self._iono_name]

        # Rebind rather than copy_ so training-mode concentration trajectories
        # remain compatible with autograd, matching the previous behavior.
        self._buffers[self._current_name] = torch.zeros_like(current)
        self._buffers[self._e_name] = _make_like(e, self.e_init)
        self._buffers[self._ioni_name] = _make_like(ioni, self.i_init)
        self._buffers[self._iono_name] = _make_like(iono, self.o_init)

        self.einit(celsius)
        if not self.training:
            self.detach()

    def detach(self):
        self._buffers[self._current_name] = self._buffers[self._current_name].detach()
        self._buffers[self._ioni_name] = self._buffers[self._ioni_name].detach()
        self._buffers[self._iono_name] = self._buffers[self._iono_name].detach()
        self._buffers[self._e_name] = self._buffers[self._e_name].detach()

    def einit(self, celsius) -> None:
        """Fallback; instances replace this with a per-ion generated method."""
        if self.init_e_reversal:
            self._buffers[self._e_name] = _nernst_potential(
                self._buffers[self._iono_name],
                self._buffers[self._ioni_name],
                celsius,
                self._buffers["_rzf_t"],
            )

    def advance(self, celsius) -> None:
        """Fallback; instances replace this with a per-ion generated method."""
        iono_t = self._buffers[self._iono_name]
        ioni_t = self._buffers[self._ioni_name]
        iono_t, ioni_t = _clamp_ion_concentrations(
            iono_t,
            ioni_t,
            self._buffers["_min_concentration_t"],
        )
        self._buffers[self._iono_name] = iono_t
        self._buffers[self._ioni_name] = ioni_t

        if self.advance_e:
            self._buffers[self._e_name] = _nernst_potential(
                iono_t,
                ioni_t,
                celsius,
                self._buffers["_rzf_t"],
            )
