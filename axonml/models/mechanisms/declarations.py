import sys
from functools import partial


def add_to_namespace(namespace, name, *args):
    if name in namespace:
        for arg in args:
            namespace[name].add(arg)
    else:
        namespace[name] = set()
        for arg in args:
            namespace[name].add(arg)


def add_to_namespace_dict(namespace, name, dct):
    if name in namespace:
        namespace[name].update(dct)
    else:
        namespace[name] = dct


def _declare(name, *args):
    # Get the calling frame (the frame where myfunc was called)
    frame = sys._getframe(1)

    # Get the local namespace of the calling frame
    namespace = frame.f_locals

    add_to_namespace(namespace, name, *args)


def _declare_parameters(name, dct):
    frame = sys._getframe(1)
    namespace = frame.f_locals
    add_to_namespace_dict(namespace, name, dct)


def USEQ10():
    frame = sys._getframe(1)
    namespace = frame.f_locals
    namespace["is_q10"] = True


STATE = partial(_declare, "_states")
PARAMETER = partial(_declare_parameters, "_params")
CONDUCTANCE = partial(_declare_parameters, "_conductances")
INITIAL = partial(_declare_parameters, "_init")


def NONSPECIFIC_CURRENT(*args):
    frame = sys._getframe(1)
    namespace = frame.f_locals
    if "_currents" not in namespace:
        namespace["_currents"] = {}
    namespace["_currents"].setdefault("nonspecific", []).extend(args)


def DERIVATIVE(f, pade=False):
    frame = sys._getframe(1)
    namespace = frame.f_locals
    namespace["_derivative"] = (f, pade)
