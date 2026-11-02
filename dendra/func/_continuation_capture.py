"""Detached current-native boundaries for explicit functional execution."""

from __future__ import annotations

import torch

from ._callback_detach import _map_recurrent_values
from ._continuation import _make_continuation
from ._continuation_checkpoints import (
    _cold_boundary,
    _is_runner_target,
    _validate_frames,
)
from ._network import FunctionalNetwork
from ._network_bindings import AliasSnapshots
from ._snapshot import _snapshot_frames
from ._types import FunctionalizationError


def capture_continuation(
    functional, native_model=None, *, callbacks=None, callback_state=None
):
    """Capture an initialized native model's current boundary as owned tensors.

    Parameters
    ----------
    functional : FunctionalPopulation, FunctionalNetwork or compiled runner
        Receiving plan/runner defining the compatible model schema.
    native_model : Population or Network, optional
        Compatible initialized source; omission uses the plan's native model.
    callbacks : FunctionalCallbacks, optional
        Explicit plan's prepared callback collection. Runners use their configured
        collection and require this argument omitted.
    callback_state : FunctionalCallbackState, optional
        Carry from that collection, retaining reductions and recording phase.

    Returns
    -------
    Continuation
        Detached owned sources/state with aliases and source trainability retained.
        Omitted callback carry starts fresh observations at the captured boundary.

    Notes
    -----
    No initialization, timestep or random sampling runs. Current native
    materializations are represented in the snapshot; the native model and runner
    defaults are unchanged. Captured leaves own subsequent gradients, cutting the
    native prefix graph. Recording payloads are not captured.

    Callback recurrent history is detached. Bound LFP fields are refreshed with
    the caller's gradient mode so suffix field gradients remain available.
    External nonleaf expressions remain caller-owned. Call outside compilation,
    function transforms and forward AD.
    """
    _cold_boundary("capture_continuation")
    runner = functional if _is_runner_target(functional, callbacks) else None
    if runner is not None:
        functional, callbacks = runner.functional, runner.callbacks
    if callback_state is not None:
        if callbacks is None:
            raise FunctionalizationError(
                "callback_state requires a prepared callback collection"
            )
        callbacks._validate_state(callback_state)
    grad_enabled = torch.is_grad_enabled()
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
    owned = AliasSnapshots()
    with torch.inference_mode(False), torch.no_grad():
        parameters, constants, state = _snapshot_frames(
            functional, tensors, snapshots=owned
        )
    carry = None
    if callback_state is not None:

        def snapshot_recurrent(value):
            return owned.snapshot(None, snapshots.resolve(value))

        def snapshot_invariant(original, value):
            # Raw sources were captured above. Refresh only recurrent roles
            # of an invariant alias, retaining the newly owned field graph.
            return owned.remember(snapshots.resolve(original), value)

        with torch.inference_mode(False), torch.set_grad_enabled(grad_enabled):
            state, carry = _map_recurrent_values(
                tensors.state,
                callbacks,
                callback_state,
                snapshot_recurrent,
                invariant_transform=snapshot_invariant,
            )
    with torch.inference_mode(False), torch.no_grad():
        continuation = _make_continuation(
            state,
            parameters,
            constants,
            carry,
            lifecycle="unstarted" if carry is None else "active",
        )
        _validate_frames(functional, continuation)
        return continuation
