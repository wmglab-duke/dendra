"""Public one-step simulation helper."""

from __future__ import annotations

from typing import Optional, Sequence

from .models.callbacks import Callback


def step(
    model,
    dt: Optional[float] = None,
    *,
    ve=None,
    extra=None,
    callbacks: Optional[Sequence[Callback]] = None,
    loop_hooks: bool = False,
    functional_callbacks=None,
    functional_callback_state=None,
):
    """Advance a Dendra Population or Network by one timestep.

    Parameters
    ----------
    model : Population or Network
        Object to advance.
    dt : float, optional
        Population timestep in milliseconds. For ``Network`` objects, the
        timestep is fixed by ``network.initialize(dt)``/``network.build(dt)``;
        if ``dt`` is provided here it must match ``network.dt``.
    ve : Tensor, optional
        Population-only precomputed extracellular voltage for this single step.
    extra : optional
        Extracellular stimulation specification. For populations, this has the
        same meaning as ``Population.run(extra=...)``. For networks, this is the
        same population-name mapping accepted by ``Network.run(extra=...)``.
    callbacks : sequence of Callback, optional
        Callbacks to execute around the step. By default only per-step hooks are
        called; set ``loop_hooks=True`` to also call loop setup/finalization
        hooks around the single step.
    loop_hooks : bool, optional
        Whether to call ``pre_loop_hook`` and ``post_loop_hook`` around the
        single step.
    functional_callbacks : dendra.func.FunctionalCallbacks, optional
        Pure observations bound to this exact model. They observe the accepted
        state after clock advancement and before native post-step hooks.
    functional_callback_state : dendra.func.FunctionalCallbackState, optional
        Carry from the same callback collection. Supply the previous result's
        ``state`` to continue sampling phase and reducer totals without another
        initial observation. Native ``loop_hooks`` remain independent.

    Returns
    -------
    Population, Network or dendra.func.FunctionalCallbackResults
        The advanced object for chaining when pure observations are omitted.
        With pure observations, returns their named results and explicit callback
        carry; simulation state remains on the model.
    """
    # Import lazily to keep the top-level helper lightweight and avoid creating
    # extra import-time coupling between dendra.__init__ and the model classes.
    from .models import Network, Population

    callback_options = {}
    if functional_callbacks is not None or functional_callback_state is not None:
        callback_options = {
            "functional_callbacks": functional_callbacks,
            "functional_callback_state": functional_callback_state,
        }

    if isinstance(model, Population):
        return model.step(
            dt=dt,
            ve=ve,
            extra=extra,
            callbacks=callbacks,
            loop_hooks=loop_hooks,
            **callback_options,
        )

    if isinstance(model, Network):
        if ve is not None:
            raise TypeError(
                "dn.step(..., ve=...) is only valid for Population objects."
            )
        if dt is not None:
            if model.dt is None:
                raise RuntimeError(
                    "Network has no simulation timestep. Call initialize(dt) or "
                    "build(dt) before dn.step(network)."
                )
            if float(dt) != float(model.dt):
                raise ValueError(
                    f"Network timestep is {float(model.dt)} ms, but dt={float(dt)} "
                    "was passed to dn.step(). Reinitialize/build the network with "
                    "the desired timestep instead."
                )
        return model.step(
            extra=extra, callbacks=callbacks, loop_hooks=loop_hooks, **callback_options
        )

    raise TypeError(
        f"dn.step expects a dendra Population or Network; got {type(model).__name__}."
    )


def step_population(
    population,
    dt: Optional[float] = None,
    *,
    ve=None,
    extra=None,
    callbacks: Optional[Sequence[Callback]] = None,
    loop_hooks: bool = False,
    functional_callbacks=None,
    functional_callback_state=None,
):
    """Advance a Dendra Population by one timestep.

    Parameters
    ----------
    population : Population
        Object to advance.
    dt : float, optional
        Population timestep in milliseconds.
    ve : Tensor, optional
        Precomputed extracellular voltage for this single step.
    extra : optional
        Extracellular stimulation specification. This has the same meaning as
        ``Population.run(extra=...)``.
    callbacks : sequence of Callback, optional
        Callbacks to execute around the step. By default only per-step hooks are
        called; set ``loop_hooks=True`` to also call loop setup/finalization
        hooks around the single step.
    loop_hooks : bool, optional
        Whether to call ``pre_loop_hook`` and ``post_loop_hook`` around the
        single step.
    functional_callbacks : dendra.func.FunctionalCallbacks, optional
        Pure observations bound to this exact Population.
    functional_callback_state : dendra.func.FunctionalCallbackState, optional
        Carry from the same callback collection. Supply the previous result's
        ``state`` to continue its sampling phase and reducer totals without
        another initial observation.

    Returns
    -------
    Population or dendra.func.FunctionalCallbackResults
        The advanced Population for chaining when pure observations are omitted.
        With pure observations, returns their named results and explicit callback
        carry; simulation state remains on the Population.
    """
    callback_options = {}
    if functional_callbacks is not None or functional_callback_state is not None:
        callback_options = {
            "functional_callbacks": functional_callbacks,
            "functional_callback_state": functional_callback_state,
        }
    return population.step(
        dt=dt,
        ve=ve,
        extra=extra,
        callbacks=callbacks,
        loop_hooks=loop_hooks,
        **callback_options,
    )


def step_network(
    network,
    *,
    extra=None,
    callbacks: Optional[Sequence[Callback]] = None,
    loop_hooks: bool = False,
    functional_callbacks=None,
    functional_callback_state=None,
):
    """Advance a Dendra Network by one timestep.

    Parameters
    ----------
    network : Network
        Object to advance.
    extra : optional
        Extracellular stimulation specification. This is the same population-name
        mapping accepted by ``Network.run(extra=...)``.
    callbacks : sequence of Callback, optional
        Callbacks to execute around the step. By default only per-step hooks are
        called; set ``loop_hooks=True`` to also call loop setup/finalization
        hooks around the single step.
    loop_hooks : bool, optional
        Whether to call ``pre_loop_hook`` and ``post_loop_hook`` around the
        single step.
    functional_callbacks : dendra.func.FunctionalCallbacks, optional
        Pure observations bound to this exact Network.
    functional_callback_state : dendra.func.FunctionalCallbackState, optional
        Carry from the same callback collection. Supply the previous result's
        ``state`` to continue its sampling phase and reducer totals without
        another initial observation.

    Returns
    -------
    Network or dendra.func.FunctionalCallbackResults
        The advanced Network for chaining when pure observations are omitted.
        With pure observations, returns their named results and explicit callback
        carry; simulation state remains on the Network.
    """
    callback_options = {}
    if functional_callbacks is not None or functional_callback_state is not None:
        callback_options = {
            "functional_callbacks": functional_callbacks,
            "functional_callback_state": functional_callback_state,
        }
    return network.step(
        extra=extra, callbacks=callbacks, loop_hooks=loop_hooks, **callback_options
    )
