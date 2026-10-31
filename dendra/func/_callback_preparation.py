"""Cold callback preparation for native models and existing functional plans."""

from __future__ import annotations

import torch

from ._callbacks import FunctionalCallbacks
from ._types import FunctionalizationError


def make_callbacks(
    target, callbacks, *, dt: float | None = None
) -> FunctionalCallbacks:
    """Prepare named pure observations for a native model or functional plan.

    Parameters
    ----------
    target : Population, Network, FunctionalPopulation or FunctionalNetwork
        Initialized native model or existing functional plan. Existing plans
        retain their exact owner and authoritative ``make_callbacks`` method.
        Native models are lowered once using the existing lowering machinery;
        preparation does not initialize, advance or resample the model.
    callbacks : mapping of str to callback configurations
        Named pure callbacks or supported built-in configurations, including
        Recorders, HDF5 storage and Network ``PopulationCallback`` scopes.
    dt : float, optional
        Required for a native Population. A native Network uses its initialized
        timestep when omitted. For Networks and existing functional plans, a
        supplied timestep must match the target's timestep.

    Returns
    -------
    FunctionalCallbacks
        Bound observations with explicit results and continuation carry. Reuse
        this collection across calls; ordinary preparation creates a new carry
        token. Use portable callback checkpoints to transfer carry between
        equivalent independently prepared collections.

    Notes
    -----
    Preparation runs outside compilation, function transforms and forward AD.
    Native targets retain the existing functional lowering cost and supported
    model restrictions. Existing plans are reused without lowering again.
    Storage sessions open during host execution, rather than preparation.
    The existing plan methods and compile-time callback configuration remain
    supported. Runners, bindings and tensor packets are not preparation targets;
    use an owning functional plan when one is available.
    """
    if (
        torch.compiler.is_compiling()
        or torch._C._are_functorch_transforms_active()
        or torch.autograd.forward_ad._current_level >= 0
    ):
        raise FunctionalizationError(
            "make_callbacks must run outside torch.compile, torch.func and forward AD"
        )

    from dendra.models.core import Population, _validate_time_scalar
    from dendra.models.networks.net import Network

    from ._network import FunctionalNetwork
    from ._population import FunctionalPopulation

    target_type = type(target)
    if issubclass(target_type, (FunctionalPopulation, FunctionalNetwork)):
        if dt is not None:
            timestep = _validate_time_scalar(dt, name="dt", positive=True)
            if timestep != target.dt:
                raise ValueError("dt must match the functional plan timestep")
        return target.make_callbacks(callbacks)

    if issubclass(target_type, Network):
        from ._lowering import _lower_network

        timestep = (
            None if dt is None else _validate_time_scalar(dt, name="dt", positive=True)
        )
        functional, _ = _lower_network(target, dt=timestep)
    elif issubclass(target_type, Population):
        if dt is None:
            raise TypeError(
                "dt is required when preparing callbacks for a native Population"
            )
        from ._lowering import make_functional

        functional, _ = make_functional(target, dt=dt)
    else:
        raise TypeError(
            "make_callbacks target must be a Population, Network, "
            "FunctionalPopulation or FunctionalNetwork"
        )
    return functional.make_callbacks(callbacks)


__all__ = ["make_callbacks"]
