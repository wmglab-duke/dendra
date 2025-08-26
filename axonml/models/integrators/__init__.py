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
from .implicit import _bwd_euler_bt, _bwd_euler_sc, _bwd_euler_ub
from .tree import _dhs
from .tree_bt import _dhs_bt


def partial_class(cls: Type[Any], /, *args, **kwargs) -> Type[Any]:
    """
    Return a subclass of *cls* whose __init__ is pre-filled with *args/kwargs*.
    Because it is a real subclass, all class attributes, methods,
    and isinstance checks continue to behave as expected.
    """

    class _Partial(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    _Partial.__name__ = f"{cls.__name__}_solver"
    _Partial.__qualname__ = _Partial.__name__
    return _Partial


euler = partial(partial_class, _euler)
rk1 = partial(partial_class, _rk1)
rk2 = partial(partial_class, _rk2)
rk4 = partial(partial_class, _rk4)
dufort_frankel = partial(partial_class, _dufort_frankel)
dufort_frankel_homogeneous = partial(partial_class, _dufort_frankel_homogeneous)
eulerv1 = partial(partial_class, _eulerv1)

#krylov_etd1 = partial(partial_class, _krylov_etd1)
bwd_euler_sc = partial(partial_class, _bwd_euler_sc)
bwd_euler_ub = partial(partial_class, _bwd_euler_ub)
bwd_euler_bt = partial(partial_class, _bwd_euler_bt)

dhs = partial(partial_class, _dhs)
dhs_bt = partial(partial_class, _dhs_bt)

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
    "bwd_euler_ub",
    "bwd_euler_bt",
    "dhs",
    "dhs_bt",
    "df",
    "dfh",
]
