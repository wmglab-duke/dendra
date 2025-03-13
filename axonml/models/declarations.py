import sys
from functools import partial


def add_to_namespace(namespace, name, *args):
    """Add values to a set in the given namespace.
    
    Parameters
    ----------
    namespace : dict
        The namespace to modify.
    name : str
        The key in the namespace to add values to.
    *args : any
        The values to add to the set.
    """
    if name in namespace:
        for arg in args:
            namespace[name].add(arg)
    else:
        namespace[name] = set()
        for arg in args:
            namespace[name].add(arg)


def add_to_namespace_dict(namespace, name, **kwargs):
    """Add key-value pairs to a dictionary in the given namespace.
    
    Parameters
    ----------
    namespace : dict
        The namespace to modify.
    name : str
        The key in the namespace whose value will be updated.
    **kwargs : any
        The key-value pairs to add to the dictionary.
    """
    if name in namespace:
        namespace[name].update(kwargs)
    else:
        namespace[name] = kwargs


def _declare(name, *args):
    """Internal function to declare variables in the calling namespace.
    
    Parameters
    ----------
    name : str
        The name of the set in the namespace.
    *args : any
        The values to add to the set.
    """
    # Get the calling frame (the frame where myfunc was called)
    frame = sys._getframe(1)

    # Get the local namespace of the calling frame
    namespace = frame.f_locals

    add_to_namespace(namespace, name, *args)


def _declare_parameters(name, inherit=None, **kwargs):
    """Internal function to declare parameters in the calling namespace.
    
    Parameters
    ----------
    name : str
        The name of the dictionary in the namespace.
    inherit : object, optional
        An object to inherit parameters from.
    **kwargs : any
        The key-value pairs to add to the dictionary.
    """
    if inherit is not None:
        data = getattr(inherit, name, {})
        kwargs = dict(data, **kwargs)
    frame = sys._getframe(1)
    namespace = frame.f_locals
    add_to_namespace_dict(namespace, name, **kwargs)


STATE = partial(_declare, "_states")
PARAMETER = partial(_declare_parameters, "_params")
INITIAL = partial(_declare_parameters, "_init")
RANGE = partial(_declare, "_range")
ASSIGNED = partial(_declare, "_assigned")
BUFFERS = partial(_declare, "_buffers")


def USEQ10():
    """Enable temperature scaling for the mechanism using Q10 rule.
    
    When enabled, rate constants in the mechanism will be scaled 
    according to the temperature using the Q10 rule.
    """
    frame = sys._getframe(1)
    namespace = frame.f_locals
    namespace["is_q10"] = True


def NONSPECIFIC_CURRENT(*args):
    """Declare nonspecific current contributions from a mechanism.
    
    Parameters
    ----------
    *args : str
        Names of current variables that don't correspond to a specific ion type.
    """
    frame = sys._getframe(1)
    namespace = frame.f_locals
    if "_currents" not in namespace:
        namespace["_currents"] = {}
    namespace["_currents"].setdefault("nonspecific", []).extend(args)


def DERIVATIVE(f, pade=False):
    """Specify derivative function(s) for state variables.
    
    Parameters
    ----------
    f : str, List[str]
        The derivative function(s) for the state variable(s).
    pade : bool, optional
        Whether to use Padé approximation, by default False.
    """
    frame = sys._getframe(1)
    namespace = frame.f_locals
    namespace["_derivative"] = (f, pade)


def DIFFUSION(D, method='strang'):
    """Specify diffusion parameters for a mechanism.
    
    Parameters
    ----------
    D : float
        Diffusion constant.
    method : {'strang', 'lie'}, optional
        The splitting method to use, by default 'strang'.
        
    Raises
    ------
    ValueError
        If the method is not one of 'strang' or 'lie'.
    """
    if method not in {'strang', 'lie'}:
        raise ValueError(f"Method must be one of 'strang' or 'lie', got {method}")
    frame = sys._getframe(1)
    namespace = frame.f_locals
    namespace["_diffusion"] = (D, method)
