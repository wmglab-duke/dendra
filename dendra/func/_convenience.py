"""Opt-in compiled runners; imperative Population scheduling is unchanged."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from types import MappingProxyType

import torch
from torch.utils import _pytree as pytree

from ._callbacks import FunctionalCallback
from ._population import make_functional
from ._runners import run
from ._types import FunctionalizationError


def _inference_compile_options(functional, inference, backend, options, keywords):
    """Prefer a native CPU loop without overriding explicit compiler choices."""
    result = dict(keywords)
    result["backend"] = backend
    selected = None if options is None else dict(options)
    if (
        inference
        and functional.device.type == "cpu"
        and backend == "inductor"
        and not result.get("disable", False)
        and result.get("mode") is None
    ):
        from torch._inductor import config

        candidate = selected or {}
        normalized = {
            name.replace("-", "_"): value for name, value in candidate.items()
        }
        if not (
            normalized.get("use_aoti", False)
            or normalized.get("disable_cpp_codegen", config.disable_cpp_codegen)
            or normalized.get("fx_wrapper", config.fx_wrapper)
        ):
            selected = dict(candidate)
            if "cpp_wrapper" not in normalized:
                selected["cpp_wrapper"] = True
    if selected is not None:
        result["options"] = selected
    return result


class CompiledRunner:
    """Reusable, fixed-horizon scan or inference-only while-loop execution.

    Construct with :func:`dendra.func.compile`. A call returns ``(state, aux)``
    and never commits state or parameters to the source Population. With no
    state argument, every call restarts from :attr:`initial_state`. Parameters
    are independent copies; select differentiable leaves with ``trainable``
    and ``inference=False``, or pass an explicit parameter mapping per call.

    ``callbacks`` is a named mapping of functional callback configurations.
    Recording is explicit: use ``callbacks={'trace': dn.func.Recorder(('v',))}``
    to retain voltage at ``aux['callbacks']['trace']['v']``. Omitting callbacks
    or passing an empty mapping records no trajectory. Callback names have no
    special meaning. Scan accepts pure custom callbacks; while supports the
    built-in callbacks and requires a nonempty mapping.

    Compilation is lazy. Inference preparation is reused with version checks;
    training and explicit parameter overrides receive fresh preparation. No
    output, live binding, or training preparation is retained between calls.
    This is a host-side convenience object, not a torch.compile/torch.func
    transform target; those uses retain the lower-level functional API.
    """

    def __init__(
        self,
        population,
        *,
        dt,
        steps,
        execution="scan",
        callbacks=None,
        inference=True,
        trainable=(),
        backend="inductor",
        options=None,
        **compile_options,
    ):
        if torch.compiler.is_compiling():
            raise FunctionalizationError(
                "construct compiled runners outside torch.compile"
            )
        removed = [
            name for name in ("record", "node_indices") if name in compile_options
        ]
        if removed:
            raise TypeError(
                f"Removed compile argument(s): {', '.join(removed)}; configure "
                "recording through callbacks={'trace': "
                "dn.func.Recorder(..., node_indices=...)}"
            )
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if execution not in ("scan", "while"):
            raise ValueError("execution must be 'scan' or 'while'")
        if not isinstance(inference, bool):
            raise TypeError("inference must be a bool")
        if callbacks is not None and not isinstance(callbacks, Mapping):
            raise TypeError(
                "callbacks must be a named mapping of FunctionalCallback configurations"
            )
        configured_callbacks = {} if callbacks is None else dict(callbacks)
        for name, callback in configured_callbacks.items():
            if not isinstance(name, str) or not name:
                raise ValueError("functional callback names must be non-empty strings")
            if not isinstance(callback, FunctionalCallback):
                raise TypeError(
                    "callback values must be FunctionalCallback configurations, not imperative callbacks"
                )
        if execution == "while":
            if not inference:
                raise ValueError(
                    "execution='while' is inference-only; use scan for gradients"
                )
            if not configured_callbacks:
                raise ValueError(
                    "execution='while' requires a nonempty callbacks mapping"
                )
            if steps < 4:
                raise ValueError("execution='while' requires at least four steps")
            if compile_options.get("disable", False):
                raise ValueError("execution='while' requires compilation to be enabled")
        self._inference = inference
        self._execution = execution
        self._steps = steps
        # Versioned snapshots must remain ordinary tensors even when the
        # convenience constructor is called inside inference_mode().
        with torch.inference_mode(False), torch.no_grad():
            self._functional, tensors = make_functional(population, dt=dt)
            self._parameters = tensors.independent_parameters(trainable=trainable)
            self._constants = dict(tensors.constants)
            self._initial_state = pytree.tree_map_only(
                torch.Tensor, lambda value: value.detach().clone(), tensors.state
            )
            self._callbacks = (
                None
                if not configured_callbacks
                else self._functional.make_callbacks(configured_callbacks)
            )
        keywords = _inference_compile_options(
            self._functional, inference, backend, options, compile_options
        )
        self._kernel = self._functional.compile_rollout_chunk(
            steps,
            execution="default" if execution == "while" else "scan",
            **keywords,
        )
        self._prepared = None

    @property
    def initial_state(self):
        """Explicit restart state; tensor leaves may be edited by the caller."""
        return self._initial_state

    @property
    def parameters(self):
        """Read-only mapping of independent, live parameter tensor leaves."""
        return MappingProxyType(self._parameters)

    @property
    def constants(self):
        """Read-only mapping of independent, version-checked constants."""
        return MappingProxyType(self._constants)

    @property
    def functional(self):
        return self._functional

    @property
    def kernel(self):
        return self._kernel

    def execution_report(self):
        """Return the underlying kernel's latest successful execution report."""
        return self._kernel.execution_report()

    def _prepare(self, parameters, *, cache):
        if cache and self._prepared is not None:
            try:
                self._functional._validate_prepared_freshness(
                    self._prepared, parameters
                )
            except FunctionalizationError as exc:
                # Refresh normal model/constant edits, but do not hide a
                # corrupted workspace, invalid schema, or foreign plan.
                if str(exc) not in (
                    "parameters changed after prepare(); call prepare() again",
                    "constants changed after prepare(); call prepare() again",
                ):
                    raise
            else:
                return self._prepared
        prepared = self._functional.prepare(parameters, self._constants)
        if cache:
            self._prepared = prepared
        return prepared

    def _require_while(self, parameters, prepared, state, inputs, callback_state):
        """Fail before a rejected structured loop could compile an unrolled horizon."""
        functional = self._functional
        functional._validate_state(state)
        values = functional._validate_prepared(prepared, parameters)
        ve, intra, _ = functional._resolve_rollout_inputs(inputs, self._steps)
        callbacks = self._callbacks
        carry = (
            callbacks._initialize(state)[0]
            if callback_state is None
            else callbacks._validate_state(callback_state)
        )
        chunk = self._kernel._runner_callback_chunk(callbacks, self._steps)
        executor = chunk._recorder_while
        if executor is None:
            raise FunctionalizationError(
                "execution='while' has no functional callback loop executor"
            )
        kernel, reason = executor.kernel_for(
            parameters, values, state, ve, intra, carry.carries
        )
        if kernel is None:
            raise FunctionalizationError(f"execution='while' is unavailable: {reason}")

    def __call__(
        self, state=None, *, parameters=None, inputs=None, callback_state=None
    ):
        if torch.compiler.is_compiling():
            raise FunctionalizationError("call compiled runners outside torch.compile")
        if (
            torch._C._are_functorch_transforms_active()
            or torch.autograd.forward_ad._current_level >= 0
        ):
            raise FunctionalizationError(
                "compiled runners are host-side callables; use the lower-level "
                "functional API for torch.func or forward-mode AD transforms"
            )
        selected_state = self._initial_state if state is None else state
        selected_parameters = self._parameters if parameters is None else parameters
        context = torch.inference_mode() if self._inference else nullcontext()
        with context:
            prepared = self._prepare(
                selected_parameters, cache=self._inference and parameters is None
            )
            bound = self._kernel.bind(selected_parameters, prepared)
            if self._execution == "while":
                self._require_while(
                    bound._parameters, prepared, selected_state, inputs, callback_state
                )
            return run(
                bound,
                selected_state,
                inputs=inputs,
                steps=self._steps,
                callbacks=self._callbacks,
                callback_state=callback_state,
            )


def compile(
    population,
    *,
    dt,
    steps,
    execution="scan",
    callbacks=None,
    inference=True,
    trainable=(),
    backend="inductor",
    options=None,
    **compile_options,
):
    """Create a reusable :class:`CompiledRunner` (scan by default).

    Recording and reductions are configured only through named ``callbacks``.
    For example, ``callbacks={'trace': dn.func.Recorder(('v',))}`` records
    voltage at ``aux['callbacks']['trace']['v']`` after ``state, aux = runner()``.
    Omit callbacks to return the final state without a trajectory. Select
    compartments with ``dn.func.Recorder(..., node_indices=[...])``, not a
    runner argument. CPU inference with ordinary Inductor defaults to
    ``options={'cpp_wrapper': True}``; explicit compiler choices win. Use
    ``inference=False, trainable=(...)`` for scan gradients. See
    :class:`CompiledRunner` for all constructor options.
    """
    return CompiledRunner(
        population,
        dt=dt,
        steps=steps,
        execution=execution,
        callbacks=callbacks,
        inference=inference,
        trainable=trainable,
        backend=backend,
        options=options,
        **compile_options,
    )


__all__ = ["CompiledRunner", "compile"]
