from functools import partial, partialmethod
from typing import Any, Type

from .explicit import (
    _dufort_frankel,
    _dufort_frankel_homogeneous,
    _euler,
    _eulerv1,
    _rk1,
    _rk2,
    _rk4,
)

#from .imex import _krylov_etd1
from .implicit import (
    _bwd_euler_bt,
    _bwd_euler_sc,
    _bwd_euler_sc_multi,
    _bwd_euler_sc_skip,
    _bwd_euler_ub,
)
from .tree import _dhs, _dhs_multi
from .tree_bt import _dhs_bt


def partial_class(cls: Type[Any], /, **kwargs):
    """
    Return a subclass of *cls* whose __init__ is pre-filled with *args/kwargs*.
    Because it is a real subclass, all class attributes, methods,
    and isinstance checks continue to behave as expected.
    """

    class _Partial(cls):
        __init__ = partialmethod(cls.__init__, **kwargs)

    _Partial.__name__ = f"{cls.__name__}"
    _Partial.__qualname__ = _Partial.__name__
    return _Partial

def make_partial_integrator(cls: Type[Any]) -> Type[Any]:
    partial_func = partial(partial_class, cls)
    partial_func.__doc__ = cls.__doc__
    return partial_func

euler = make_partial_integrator(_euler)
rk1 = make_partial_integrator(_rk1)
rk2 = make_partial_integrator(_rk2)
rk4 = make_partial_integrator(_rk4)
dufort_frankel = make_partial_integrator(_dufort_frankel)
dufort_frankel_homogeneous = make_partial_integrator(_dufort_frankel_homogeneous)
eulerv1 = make_partial_integrator(_eulerv1)

#krylov_etd1 = partial(partial_class, _krylov_etd1)
bwd_euler_sc = make_partial_integrator(_bwd_euler_sc)
bwd_euler_sc_multi = make_partial_integrator(_bwd_euler_sc_multi)
bwd_euler_ub = make_partial_integrator(_bwd_euler_ub)
bwd_euler_bt = make_partial_integrator(_bwd_euler_bt)
scnv = make_partial_integrator(_bwd_euler_sc_skip)

dhs = make_partial_integrator(_dhs)
dhs_multi = make_partial_integrator(_dhs_multi)
dhs_bt = make_partial_integrator(_dhs_bt)

df = dufort_frankel
dfh = dufort_frankel_homogeneous


__all__ = [
    "euler",
    "eulerv1",
    "rk1",
    "rk2",
    "rk4",
    "dufort_frankel",
    "dufort_frankel_homogeneous",
    #"krylov_etd1",
    "bwd_euler_sc",
    "bwd_euler_sc_multi",
    "bwd_euler_ub",
    "bwd_euler_bt",
    "dhs",
    "dhs_multi",
    "dhs_bt",
    "df",
    "dfh",
    "scnv",
]
