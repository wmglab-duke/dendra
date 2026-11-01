"""Detached current-native boundaries for explicit functional execution."""

from __future__ import annotations

import torch

from ._continuation import _make_continuation
from ._continuation_checkpoints import _cold_boundary, _validate_frames
from ._network import FunctionalNetwork
from ._network_bindings import AliasSnapshots
from ._population import FunctionalPopulation
from ._snapshot import _snapshot_frames
from ._types import FunctionalizationError


def capture_continuation(functional, native_model=None):
    """Snapshot a native Population or Network at its current boundary.

    ``functional`` is a lowered Population/Network plan or a ``dn.compile``
    runner. ``native_model`` defaults to that plan's native source; an explicit
    model must match the plan's structure and configured timestep. Capturing
    does not initialize, advance, or sample the model. Existing extraction
    evaluates deterministic materializations needed to represent native
    prepared workspaces faithfully.

    Return a :class:`Continuation` with owned detached Tensor leaves, current
    raw parameters/constants, clock, solver history and RNG carry. Source
    identity ties and current trainability are retained; distinct Tensor views
    are independent leaves. Runner defaults, trainable selectors and fresh
    initialization policy do not affect capture. Subsequent gradients belong
    to the captured sources, not the native model or its upstream graph.

    Callback carry and recording history are not captured. Configured
    functional callbacks start at the captured boundary on first execution;
    transfer existing carry through the callback export/restore protocol.
    Inputs, draw tapes and runtime ``extra`` remain invocation arguments.
    Capture runs outside compilation, function transforms and forward AD.
    """
    _cold_boundary("capture_continuation")
    from ._convenience import CompiledRunner
    from ._network_compile import CompiledNetworkRunner

    if type(functional) in (CompiledRunner, CompiledNetworkRunner):
        functional = functional.functional
    if type(functional) not in (FunctionalPopulation, FunctionalNetwork):
        raise TypeError(
            "capture_continuation requires a Population/Network plan or compiled runner"
        )
    with torch.inference_mode(False), torch.no_grad():
        functional._validate_source()
        snapshots = AliasSnapshots(preserve_native_grad=True)
        if type(functional) is FunctionalNetwork:
            native_model = functional._source if native_model is None else native_model
            functional._validate_compatible(native_model)
            populations = (
                (plan, native_model.populations[name])
                for name, plan in functional.populations.items()
            )
        else:
            native_model = (
                functional._population if native_model is None else native_model
            )
            functional._validate_compatible_population(native_model)
            populations = ((functional, native_model),)
        for plan, population in populations:
            if not population.initialized:
                raise FunctionalizationError(
                    "Initialize the native Population before capturing a continuation"
                )
            integrator = population.integrator
            # Native configuration rounds dt to the Population dtype before
            # retaining its Python scalar; compare in that same numeric domain.
            if integrator.initialized and not torch.equal(
                torch.as_tensor(integrator.dt, dtype=plan.dtype, device="cpu"),
                torch.as_tensor(plan.dt, dtype=plan.dtype, device="cpu"),
            ):
                raise FunctionalizationError(
                    "capture_continuation requires the native Population timestep "
                    "to match the functional plan"
                )
        tensors = functional._extract(native_model, snapshots=snapshots)
        parameters, constants, state = _snapshot_frames(functional, tensors)
        continuation = _make_continuation(
            state, parameters, constants, lifecycle="unstarted"
        )
        _validate_frames(functional, continuation)
        return continuation
