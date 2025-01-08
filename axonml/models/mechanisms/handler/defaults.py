from contextlib import ContextDecorator

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


class e_context(ContextDecorator):
    def __init__(self, **kwargs):
        global REVERSAL
        not_in_reversal = [k for k in kwargs if k not in REVERSAL]
        if not_in_reversal:
            raise ValueError(f"Reversal potential not found: {not_in_reversal}")
        self.updates = kwargs
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


class c_context(ContextDecorator):
    def __init__(self, **kwargs):
        global CINIT
        not_in_cinit = [k for k in kwargs if k not in CINIT]
        if not_in_cinit:
            raise ValueError(f"Initial concentration not found: {not_in_cinit}")
        self.updates = kwargs
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
