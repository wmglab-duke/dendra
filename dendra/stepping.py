"""Public one-step simulation helper."""

from __future__ import annotations

from typing import Optional, Sequence

from .models.callbacks import Callback


def step(
    model,
    dt: Optional[float] = None,
    *,
    inputs=None,
    draws=None,
    ve=None,
    extra=None,
    callbacks: Optional[Sequence[Callback]] = None,
    loop_hooks: bool = False,
    callback_state=None,
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
    inputs : dendra.func.Inputs or mapping of str to dendra.func.Inputs, optional
        Deterministic voltage and current frames. A Population consumes one
        packet; a Network consumes a mapping keyed by exact Population names.
        Frames pass unchanged to the native method, which validates broadcasting,
        dtype/device and conflicts with existing stimulation sources.
    draws : dendra.func.RuntimeDraws, optional
        One explicit native NOISE/SDE frame, including NetStim for a Network.
        The native method validates and consumes it, publishing its RNG suffixes.
        Populations also replay solver-owned intra/extra Poisson waveform banks.
        Network and mechanism-owned Poisson replay are currently unsupported.
    ve : Tensor, optional
        Population-only precomputed extracellular voltage for this single step.
    extra : optional
        Extracellular stimulation specification. For populations, this has the
        same meaning as ``Population.run(extra=...)``. For networks, this is the
        same population-name mapping accepted by ``Network.run(extra=...)``.
    callbacks : sequence of Callback or dendra.func.FunctionalCallbacks, optional
        Native hook sequence, or a pure collection prepared once with
        ``dn.func.make_callbacks`` for this model or a compatible initialized
        receiver. Observations read receiver state with fixed prepared selections.
        Population ``dt`` must honor prepared cadence and threshold timing;
        a Network uses its matching initialized timestep and mode.
        Pure observations run after clock advancement and before native
        post-step hooks. Native hooks retain their loop/chunk/step timing.
    loop_hooks : bool, optional
        Whether to call ``pre_loop_hook`` and ``post_loop_hook`` around the
        single step.
    callback_state : dendra.func.FunctionalCallbackState, optional
        Carry from the same prepared collection supplied through
        ``callbacks`` or ``functional_callbacks``. Resumes sampling
        phase and reducer totals without another initial observation.
    functional_callbacks : dendra.func.FunctionalCallbacks, optional
        Compatibility spelling for the prepared pure collection. Use it
        alongside ``callbacks=[...]`` to combine native hooks and pure
        observations. Supply the collection through only one argument.
    functional_callback_state : dendra.func.FunctionalCallbackState, optional
        Compatibility spelling for ``callback_state``. Supply carry
        through only one argument; either collection spelling is accepted.

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

    step_options = {}
    if functional_callbacks is not None or functional_callback_state is not None:
        step_options = {
            "functional_callbacks": functional_callbacks,
            "functional_callback_state": functional_callback_state,
        }
    if callback_state is not None:
        step_options["callback_state"] = callback_state
    if inputs is not None:
        step_options["inputs"] = inputs
    if draws is not None:
        step_options["draws"] = draws

    if isinstance(model, Population):
        return model.step(
            dt=dt,
            ve=ve,
            extra=extra,
            callbacks=callbacks,
            loop_hooks=loop_hooks,
            **step_options,
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
            extra=extra, callbacks=callbacks, loop_hooks=loop_hooks, **step_options
        )

    raise TypeError(
        f"dn.step expects a dendra Population or Network; got {type(model).__name__}."
    )


def step_population(
    population,
    dt: Optional[float] = None,
    *,
    inputs=None,
    draws=None,
    ve=None,
    extra=None,
    callbacks: Optional[Sequence[Callback]] = None,
    loop_hooks: bool = False,
    callback_state=None,
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
    inputs : dendra.func.Inputs, optional
        Deterministic extracellular voltage and intracellular current frames.
        The native method validates broadcasting, dtype/device and stimulation
        conflicts. Tensor references and gradients are retained.
    draws : dendra.func.RuntimeDraws, optional
        One explicit NOISE/SDE frame and authoritative RNG suffixes, validated
        by the native method. Solver-owned intra/extra Poisson waveform banks
        consume invocation-indexed draws; mechanism-owned replay is unsupported.
    ve : Tensor, optional
        Precomputed extracellular voltage for this single step.
    extra : optional
        Extracellular stimulation specification. This has the same meaning as
        ``Population.run(extra=...)``.
    callbacks : sequence of Callback or dendra.func.FunctionalCallbacks, optional
        Native hook sequence, or a pure collection prepared once with
        ``dn.func.make_callbacks`` for this Population or a compatible initialized
        receiver. Observations read receiver state with fixed prepared selections.
        Requested ``dt`` must honor prepared cadence and threshold timing.
        Pure observations run after clock advancement and before native
        post-step hooks. Native hooks retain their loop/chunk/step timing.
    loop_hooks : bool, optional
        Whether to call ``pre_loop_hook`` and ``post_loop_hook`` around the
        single step.
    callback_state : dendra.func.FunctionalCallbackState, optional
        Carry from the same prepared collection supplied through
        ``callbacks`` or ``functional_callbacks``. Resumes sampling
        phase and reducer totals without another initial observation.
    functional_callbacks : dendra.func.FunctionalCallbacks, optional
        Compatibility spelling for the prepared pure collection. Use it
        alongside ``callbacks=[...]`` to combine native hooks and pure
        observations. Supply the collection through only one argument.
    functional_callback_state : dendra.func.FunctionalCallbackState, optional
        Compatibility spelling for ``callback_state``. Supply carry
        through only one argument; either collection spelling is accepted.

    Returns
    -------
    Population or dendra.func.FunctionalCallbackResults
        The advanced Population for chaining when pure observations are omitted.
        With pure observations, returns their named results and explicit callback
        carry; simulation state remains on the Population.
    """
    step_options = {}
    if functional_callbacks is not None or functional_callback_state is not None:
        step_options = {
            "functional_callbacks": functional_callbacks,
            "functional_callback_state": functional_callback_state,
        }
    if callback_state is not None:
        step_options["callback_state"] = callback_state
    if inputs is not None:
        step_options["inputs"] = inputs
    if draws is not None:
        step_options["draws"] = draws
    return population.step(
        dt=dt,
        ve=ve,
        extra=extra,
        callbacks=callbacks,
        loop_hooks=loop_hooks,
        **step_options,
    )


def step_network(
    network,
    *,
    extra=None,
    inputs=None,
    draws=None,
    callbacks: Optional[Sequence[Callback]] = None,
    loop_hooks: bool = False,
    callback_state=None,
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
    inputs : mapping of str to dendra.func.Inputs, optional
        Deterministic voltage and current frames keyed by exact Population names.
        The native method validates broadcasting, dtype/device and stimulation
        conflicts. Tensor references and gradients are retained.
    draws : dendra.func.RuntimeDraws, optional
        One explicit NOISE/SDE and NetStim frame and authoritative RNG
        boundaries, validated by the native method. Randomized Poisson replay
        is currently unsupported.
    callbacks : sequence of Callback or dendra.func.FunctionalCallbacks, optional
        Native hook sequence, or a pure collection prepared once with
        ``dn.func.make_callbacks`` for this Network or a compatible initialized
        receiver with matching timestep and mode. Observations read receiver
        state with fixed prepared selections.
        Pure observations run after clock advancement and before native
        post-step hooks. Native hooks retain their loop/chunk/step timing.
    loop_hooks : bool, optional
        Whether to call ``pre_loop_hook`` and ``post_loop_hook`` around the
        single step.
    callback_state : dendra.func.FunctionalCallbackState, optional
        Carry from the same prepared collection supplied through
        ``callbacks`` or ``functional_callbacks``. Resumes sampling
        phase and reducer totals without another initial observation.
    functional_callbacks : dendra.func.FunctionalCallbacks, optional
        Compatibility spelling for the prepared pure collection. Use it
        alongside ``callbacks=[...]`` to combine native hooks and pure
        observations. Supply the collection through only one argument.
    functional_callback_state : dendra.func.FunctionalCallbackState, optional
        Compatibility spelling for ``callback_state``. Supply carry
        through only one argument; either collection spelling is accepted.

    Returns
    -------
    Network or dendra.func.FunctionalCallbackResults
        The advanced Network for chaining when pure observations are omitted.
        With pure observations, returns their named results and explicit callback
        carry; simulation state remains on the Network.
    """
    step_options = {}
    if functional_callbacks is not None or functional_callback_state is not None:
        step_options = {
            "functional_callbacks": functional_callbacks,
            "functional_callback_state": functional_callback_state,
        }
    if callback_state is not None:
        step_options["callback_state"] = callback_state
    if inputs is not None:
        step_options["inputs"] = inputs
    if draws is not None:
        step_options["draws"] = draws
    return network.step(
        extra=extra, callbacks=callbacks, loop_hooks=loop_hooks, **step_options
    )
