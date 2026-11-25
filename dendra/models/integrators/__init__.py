from functools import partial, partialmethod
from numbers import Integral
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

# IMEX/Krylov implementations remain direct, explicit imports from ``.imex``
# while that module is experimental.
from .implicit import (
    _bwd_euler_bt,
    _bwd_euler_sc,
    _bwd_euler_sc_multi,
    _bwd_euler_sc_skip,
    _bwd_euler_ub,
)
from .tree import _dhs, _dhs_multi
from .tree_bt import _dhs_bt, _dhs_bt_multi


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

bwd_euler_sc = make_partial_integrator(_bwd_euler_sc)
bwd_euler_sc_multi = make_partial_integrator(_bwd_euler_sc_multi)
bwd_euler_ub = make_partial_integrator(_bwd_euler_ub)
bwd_euler_bt = make_partial_integrator(_bwd_euler_bt)
scnv = make_partial_integrator(_bwd_euler_sc_skip)

dhs = make_partial_integrator(_dhs)
dhs_multi = make_partial_integrator(_dhs_multi)
dhs_bt = make_partial_integrator(_dhs_bt)
dhs_bt_multi = make_partial_integrator(_dhs_bt_multi)


def tree_contraction(max_segment_interior_nodes=16, *, imem=None):
    """Configure a scalar Tree integrator that solves by tree contraction.

    Pass the returned factory as ``integrator=dn.tree_contraction()`` when
    constructing a Tree. The solver supports float32 and float64 on CUDA and
    float32 on Apple MPS.

    ``max_segment_interior_nodes`` is a positive integer counting only the
    interior nodes of each chain segment. Pass ``None`` to skip chain
    condensation and use fine-grained contraction.

    Set ``imem`` to accumulate membrane currents each step, or leave it as
    ``None`` to inherit the current Dendra context.
    """
    if max_segment_interior_nodes is not None:
        if isinstance(max_segment_interior_nodes, bool) or not isinstance(
            max_segment_interior_nodes, Integral
        ):
            raise TypeError(
                "max_segment_interior_nodes must be a positive integer or None"
            )
        if max_segment_interior_nodes <= 0:
            raise ValueError("max_segment_interior_nodes must be positive")
        max_segment_interior_nodes = int(max_segment_interior_nodes)
    return partial_class(
        _dhs,
        _tree_contraction_options=(
            max_segment_interior_nodes,
            None if imem is None else bool(imem),
        ),
    )


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
    "bwd_euler_sc",
    "bwd_euler_sc_multi",
    "bwd_euler_ub",
    "bwd_euler_bt",
    "dhs",
    "dhs_multi",
    "dhs_bt",
    "dhs_bt_multi",
    "tree_contraction",
    "df",
    "dfh",
    "scnv",
]
