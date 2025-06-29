import torch
from contextlib import ContextDecorator

from axonml.helpers import DEBUG

# default reversal potentials from NEURON
REVERSAL = {
    "ena": 50.0, 
    "ek" : -77.0, 
    "eca": 132.0
}

VALENCES = {
    "na": 1.0,
    "k" : 1.0,
    "ca": 2.0
}

# default initial concentrations from NEURON
CINIT = {
    "nao0": 140.0,
    "nai0": 10.0,
    "ko0" : 2.5,
    "ki0" : 54.4,
    "cao0": 2.0,
    "cai0": 5e-5,
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


def ion_register(ion, valence, e, i0, o0):
    global VALENCES
    global REVERSAL
    global CINIT
    VALENCES[ion] = valence
    REVERSAL[f"e{ion}"] = e
    CINIT[f"{ion}o0"] = o0
    CINIT[f"{ion}i0"] = i0


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


class Ion(torch.nn.Module):

    __constants__ = "init_e_reversal", "advance_e"

    def __init__(self, name, shape, cstyle, estyle, einit, eadvance, cinit):
        super().__init__()
        self.name = name
        self.rzf = R / (VALENCES[name] * FARADAY)

        self.e_init = reversals()[f"e{name}"]
        self.i_init = cinits()[f"{name}i0"]
        self.o_init = cinits()[f"{name}o0"]

        self.register_buffer(f"i{name}", torch.zeros(shape))
        self.register_buffer(f"e{name}", torch.full(shape, self.e_init))
        self.register_buffer(f"{name}i", torch.full(shape, self.i_init))
        self.register_buffer(f"{name}o", torch.full(shape, self.o_init))

        self.init_e_reversal = einit != 0
        self.advance_e = eadvance != 0

    def initialize(self, celsius) -> None:
        name = self.name
        i = getattr(self, f"i{name}")
        e = getattr(self, f"e{name}")
        ioni = getattr(self, f"{name}i")
        iono = getattr(self, f"{name}o")
        setattr(self, f"i{name}", torch.full(i.shape, 0.0, dtype=i.dtype, device=i.device))
        setattr(self, f"e{name}", torch.full(e.shape, self.e_init, dtype=e.dtype, device=e.device))
        setattr(self, f"{name}i", torch.full(ioni.shape, self.i_init, dtype=ioni.dtype, device=ioni.device))
        setattr(self, f"{name}o", torch.full(iono.shape, self.o_init, dtype=iono.dtype, device=iono.device))
        self.einit(celsius)
        self.detach()
    
    def detach(self):
        name = self.name
        getattr(self, f"i{name}").detach_()
        getattr(self, f"{name}i").detach_()
        getattr(self, f"{name}o").detach_()
        getattr(self, f"e{name}").detach_()

    def einit(self, celsius) -> None:
        if self.init_e_reversal:
            name = self.name
            iono = getattr(self, f"{name}o")
            ioni = getattr(self, f"{name}i")
            new_val = torch.log(iono / ioni) * self.rzf * (273.15 + celsius)
            setattr(self, f"e{name}", new_val)

    def advance(self, celsius) -> None:
        if not self.advance_e:
            return
        name = self.name
        iono = getattr(self, f"{name}o")
        ioni = getattr(self, f"{name}i")
        new_val = torch.log(iono / ioni) * self.rzf * (273.15 + celsius)
        setattr(self, f"e{name}", new_val)
