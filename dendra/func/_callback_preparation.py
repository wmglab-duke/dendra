"""Cold callback preparation for native models and existing functional plans."""

from __future__ import annotations

import torch

from ._callbacks import FunctionalCallbacks
from ._types import FunctionalizationError


def make_callbacks(
    target, callbacks, *, dt: float | None = None
) -> FunctionalCallbacks:
    """Prepare named observations for a native model or functional plan.

    Parameters
    ----------
    target : Population, Network, FunctionalPopulation or FunctionalNetwork
        Initialized native model or existing lowered plan. Native targets are
        lowered once to bind selections; preparation does not initialize, advance
        or resample the source.
    callbacks : mapping of str to callback configurations
        Supported built-in callbacks or pure FunctionalCallback objects. Recorder
        HDF5 settings and Network PopulationCallback scopes are supported.
    dt : float, optional
        Required for a native Population. A native Network uses its initialized
        timestep; any explicit Network/functional timestep must match the plan.

    Returns
    -------
    FunctionalCallbacks
        Bound collection with explicit result/carry state. Reuse with compatible
        native receivers, or its exact owning functional plan.

    Notes
    -----
    Carry belongs to this collection. Use export_state/restore_state to transfer
    it between equivalent independently prepared collections. Slice selections
    refer to the original source during preparation; native reuse applies their
    resolved indices to the receiver. LFP fields retain their bound sources.

    This host operation runs outside compilation, function transforms and forward
    AD. Native preparation has functional lowering's supported-model restrictions.
    Storage opens during host execution, not preparation. Runners, bindings and
    tensor bundles are not preparation targets; use their owning functional plan.
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
