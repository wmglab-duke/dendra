"""Execution adapters for independent amplitude trials in Thresholder."""

import math
from collections.abc import Mapping

import torch

from dendra.models.callbacks import Recorder
from dendra.models.core import _validate_time_scalar


class _NativeThresholdTrials:
    compiled = False

    def __init__(self, model, active, *, peak, block_threshold):
        if isinstance(active, str):
            raise TypeError("active callback names require a compiled runner")
        if peak is not None or block_threshold is not None:
            raise TypeError("peak and block_threshold apply to compiled thresholding")
        self.model = model
        self.active = active
        self.recorder = Recorder(["v"], max_only=True)

    @property
    def shape(self):
        return tuple(getattr(self.model, "shape", (self.model.np, self.model.nc)))

    @property
    def np(self):
        return self.model.np

    @property
    def nc(self):
        return self.model.nc

    @property
    def device(self):
        return self.model.device()

    @property
    def dtype(self):
        return self.model.dtype()

    @property
    def diameters(self):
        return getattr(self.model, "diameters", None)

    @property
    def threshold(self):
        return self.active.threshold

    def cast(self, dtype):
        self.model = (
            self.model.float() if dtype == torch.float32 else self.model.double()
        )
        return self.model

    def horizon(self, tstop, dt, *, basis=False):
        del basis
        return tstop, dt

    def configure(self, owner):
        del owner

    def validate(self, owner, *, block_possible=False):
        del owner, block_possible

    def partition(self, model_partition, active_partition):
        del active_partition
        self.recorder.set_partition(model_partition)

    def reset(self, owner):
        owner.rec.reset()
        owner.active.reset()

    def evaluate(self, owner, bound, tstop, dt, *, with_peak):
        owner.active.reset()
        if with_peak:
            owner.rec.reset()
        callbacks = [owner.active] + ([owner.rec] if with_peak else [])
        with torch.no_grad():
            # Preserve initialization order for live field aliases: block
            # diagnostics read their field after initialization.
            if with_peak:
                self.model.initialize()
            ve, waveform = owner._trial_stimulus(bound)
            if not with_peak:
                self.model.initialize()
            if waveform is None:
                if with_peak:
                    self.model.run(ve, dt=dt, callbacks=callbacks)
                else:
                    self.model.run(ve=ve, dt=dt, callbacks=callbacks)
            else:
                kwargs = {
                    "extra": (ve, waveform),
                    "tstop": tstop,
                    "dt": dt,
                    "callbacks": callbacks,
                }
                if owner.chunklength is None:
                    self.model.run(**kwargs)
                else:
                    self.model.longrun(**kwargs, chunklength=owner.chunklength)
        activity = owner.active.is_active(partition=owner.active_partition)
        peak = owner.rec.stack("v").amax(dim=0) if with_peak else None
        return activity, peak


class _CompiledThresholdTrials:
    compiled = True

    def __init__(self, program, active, *, peak, block_threshold):
        from dendra.func._callbacks import ActiveAL
        from dendra.func._callbacks import Recorder as PureRecorder
        from dendra.func._callbacks import _BoundCadencedRecorder

        self.model = program
        self.functional = program.functional
        self.shape = self.functional.shape
        self.np, self.nc = self.shape[-2:]
        self.device, self.dtype = self.functional.device, self.functional.dtype
        self.diameters = None
        self.active = self._plan(active, "active")
        schema = self.active.result_schema
        if (
            not schema.spec.is_leaf()
            or len(schema.leaves) != 1
            or schema.leaves[0].dtype != torch.bool
        ):
            raise TypeError("active must name a callback returning a Boolean Tensor")
        self.peak = None if peak is None else self._plan(peak, "peak")
        self.recorder = None
        if self.peak is not None:
            callback = self.peak.callback
            if type(callback) is _BoundCadencedRecorder:
                if callback.save_every != 1:
                    raise ValueError("peak Recorder must sample every solver step")
                callback = callback.recorder
            if isinstance(callback, PureRecorder):
                if (
                    callback.states != ("v",)
                    or not callback.max_only
                    or callback.region is not None
                ):
                    raise ValueError(
                        "peak Recorder must record only 'v' with max_only=True"
                    )
                if callback._host_storage is not None:
                    raise ValueError("peak Recorder must retain in-memory samples")
                self.recorder = callback
            elif (
                not self.peak.result_schema.spec.is_leaf()
                or not self.peak.result_schema.leaves[0].dtype.is_floating_point
            ):
                raise TypeError(
                    "peak must name a voltage Recorder or a floating-point Tensor result"
                )
        self.threshold = (
            self.active.callback.threshold
            if isinstance(self.active.callback, ActiveAL)
            else None
        )
        if block_threshold is not None:
            if isinstance(block_threshold, bool):
                raise TypeError("block_threshold must be a finite voltage")
            self.threshold = float(block_threshold)
            if not math.isfinite(self.threshold):
                raise ValueError("block_threshold must be a finite voltage")

    def _plan(self, name, argument):
        if not isinstance(name, str) or not name:
            raise TypeError(f"{argument} must name a compiled callback result")
        callbacks = self.model.callbacks
        if callbacks is not None:
            for plan in callbacks._plans:
                if plan.name == name:
                    return plan
        raise ValueError(f"compiled callback {name!r} does not exist")

    def cast(self, dtype):
        if dtype != self.dtype:
            raise TypeError("compile a new runner to change the thresholding dtype")
        return self.model

    def horizon(self, tstop, dt, *, basis=False):
        from dendra.func._runners import _population_timestep

        effective_dt = _population_timestep(self.functional, self.model.constants["dt"])
        if (
            dt is not None
            and _validate_time_scalar(dt, name="dt", positive=True) != effective_dt
        ):
            raise ValueError("dt must match the compiled runner timestep")
        if basis:
            return None, effective_dt
        if tstop is not None:
            tstop = _validate_time_scalar(tstop, name="tstop", positive=False)
            if tstop < self._start_time():
                raise ValueError("tstop must not precede the compiled trial start time")
        return tstop, effective_dt

    def _start_time(self):
        return (
            0.0
            if self.model._initialization == "fresh"
            else float(self.model.initial_state["clock"]["t"].detach().cpu())
        )

    def partition(self, model_partition, active_partition):
        from dendra.func._callbacks import ActiveAL

        callback = self.active.callback
        if isinstance(callback, ActiveAL) and callback.partition != active_partition:
            raise ValueError(
                "active_partition must match the compiled activity callback partition"
            )
        if self.recorder is not None and self.recorder.partition != model_partition:
            raise ValueError(
                "model_partition must match the compiled peak Recorder partition"
            )

    def configure(self, owner):
        if owner.chunklength is not None:
            raise ValueError("set chunk_steps in dn.compile instead of chunklength")
        if owner.bases is None:
            if not self.model._runtime_extra:
                raise ValueError(
                    "space/time thresholding requires compile(..., extra=dn.VarInput())"
                )
        elif self.model._runtime_extra or self.functional.extra._enabled:
            raise ValueError(
                "bases require a compiled runner without extracellular bindings"
            )

    def validate(self, owner, *, block_possible=False):
        self.configure(owner)
        shape = self.active.result_schema.leaves[0].shape
        if shape != tuple(owner.ub.shape):
            raise ValueError(
                f"compiled activity result shape {shape} must match threshold lane shape {tuple(owner.ub.shape)}; configure partitions before searching"
            )
        if owner.model_partition is not None or block_possible:
            self.partition(owner.model_partition, owner.active_partition)
        if block_possible:
            if self.peak is None:
                raise ValueError(
                    "block_possible=True requires peak to name a compiled diagnostic"
                )
            if owner.threshold is None:
                raise ValueError(
                    "custom activity callbacks require block_threshold for block_possible=True"
                )
            if self.recorder is None:
                shape = self.peak.result_schema.leaves[0].shape
                expected = tuple(owner.ub.shape)
                allowed = (
                    (expected,)
                    if owner.model_partition is not None
                    else (expected, (*expected, 1))
                )
                if shape not in allowed:
                    raise ValueError(
                        "peak Tensor result shape must match threshold lanes"
                    )

    def reset(self, owner):
        # Every runner call below omits neural state and callback carry.
        del owner

    def evaluate(self, owner, bound, tstop, dt, *, with_peak):
        from dendra.func import Inputs

        del dt
        with torch.no_grad():
            ve, waveform = owner._trial_stimulus(bound)
            if waveform is None:
                kwargs = {"inputs": Inputs(ve=ve), "steps": int(ve.shape[0])}
            else:
                kwargs = {"extra": (ve, waveform)}
                if tstop is not None:
                    kwargs["duration"] = tstop - self._start_time()
            _, auxiliary = self.model(**kwargs)
        results = auxiliary["callbacks"]
        activity = results[self.active.name]
        maximum = None
        if with_peak:
            maximum = results[self.peak.name]
            if self.recorder is not None:
                if not isinstance(maximum, Mapping) or "v" not in maximum:
                    raise TypeError("peak Recorder must return voltage samples")
                maximum = maximum["v"].amax(dim=0)
        return activity, maximum


def _threshold_trials(model, active, *, peak, block_threshold):
    # Import only when constructing a Thresholder, after model modules exist.
    from dendra.func._convenience import CompiledRunner
    from dendra.func._network_compile import CompiledNetworkRunner

    if isinstance(model, CompiledNetworkRunner):
        raise TypeError(
            "Thresholder currently supports compiled Population runners, not Networks"
        )
    cls = (
        _CompiledThresholdTrials
        if isinstance(model, CompiledRunner)
        else _NativeThresholdTrials
    )
    return cls(model, active, peak=peak, block_threshold=block_threshold)
