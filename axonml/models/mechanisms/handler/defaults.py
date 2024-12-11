# default reversal potentials from NEURON
REVERSAL = {"na": 50.0, "k": -77.0, "ca": 132.0}

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

def ion_register(ion, valence, e, i0, o0):
    global VALENCES
    global REVERSAL
    global CINIT
    VALENCES[ion] = valence
    REVERSAL[ion] = e
    CINIT[f"{ion}o0"] = o0
    CINIT[f"{ion}i0"] = i0
