import inspect
import math
from typing import Optional, Sequence, Tuple

import torch

from dendra._bootstrap import (
    reset_torch_compiler,
    torch_compiler_warning_context,
)
from dendra.helpers import (
    BACKEND,
    COMPILE_MODE,
    DYNAMIC,
    FULLGRAPH,
    IMEM,
    compile_options_key,
    current_compile_options,
    detach_vars,
    jit_enabled_for_scope,
    normalize_compile_options,
)


def get_init_defaults(cls):
    signature = inspect.signature(cls.__init__)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty and k != "self"
    }


def _cfg_value(value):
    return getattr(value, "value", value)


def _shape_tuple(shape: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(x) for x in shape)


def solver_shape_from_voltage_shape(shape: Sequence[int]) -> Tuple[int, int]:
    """Return canonical flattened cable-solver shape ``(B, K)``."""
    shape = _shape_tuple(shape)
    if len(shape) < 1:
        raise ValueError(f"voltage shape must have at least one dimension; got {shape}")
    K = int(shape[-1])
    B = int(math.prod(shape[:-1])) if len(shape) > 1 else 1
    return B, K


def solver_shape(model) -> Tuple[int, int]:
    return solver_shape_from_voltage_shape(model.shape)


def flatten_model_tensor(
    x: torch.Tensor,
    target_shape: Sequence[int],
    *,
    core_ndim: int = 1,
) -> torch.Tensor:
    """
    Broadcast ``x`` to ``target_shape`` and flatten all non-core axes.

    ``core_ndim=1`` preserves the final compartment axis and returns ``(B, K)``.
    ``core_ndim=2`` preserves the final ``(K, M)`` block axes and returns
    ``(B, K, M)``. This lets integrators expose public states in batched model
    shape while feeding flattened matrices to numerical solvers.
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    target_shape = _shape_tuple(target_shape)
    if core_ndim < 1 or core_ndim > len(target_shape):
        raise ValueError(
            f"core_ndim must be in [1, {len(target_shape)}]; got {core_ndim}."
        )

    lead_shape = target_shape[:-core_ndim]
    core_shape = target_shape[-core_ndim:]
    flat_shape = (int(math.prod(lead_shape)) if lead_shape else 1, *core_shape)

    if tuple(x.shape) == flat_shape:
        return x

    try:
        x_view = x
        if x_view.ndim < len(target_shape):
            x_view = x_view.reshape(
                (1,) * (len(target_shape) - x_view.ndim) + tuple(x_view.shape)
            )
        return x_view.expand(target_shape).reshape(flat_shape)
    except RuntimeError as err:
        if int(x.numel()) == int(math.prod(target_shape)):
            return x.reshape(flat_shape)
        raise RuntimeError(
            f"Cannot broadcast tensor with shape {tuple(x.shape)} to target_shape "
            f"{target_shape} before flattening to {flat_shape}."
        ) from err


def flatten_voltage(v: torch.Tensor) -> torch.Tensor:
    return flatten_model_tensor(v, tuple(v.shape), core_ndim=1)


def flatten_optional_model_tensor(
    x: Optional[torch.Tensor],
    target_shape: Sequence[int],
    *,
    core_ndim: int = 1,
) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return flatten_model_tensor(x, target_shape, core_ndim=core_ndim)


def unflatten_voltage(
    v_flat: torch.Tensor, target_shape: Sequence[int]
) -> torch.Tensor:
    return v_flat.reshape(_shape_tuple(target_shape))


def ensure_model_buffer(
    model, name: str, shape: Sequence[int], *, dtype=None, device=None
):
    """Ensure ``model.<name>`` exists as a tensor buffer with ``shape``."""
    shape = _shape_tuple(shape)
    device = model.device() if device is None else device
    dtype = model.dtype() if dtype is None else dtype
    current = getattr(model, name, None)
    if torch.is_tensor(current) and tuple(current.shape) == shape:
        return current
    value = torch.zeros(shape, device=device, dtype=dtype)
    if name in getattr(model, "_buffers", {}):
        setattr(model, name, value)
    else:
        model.register_buffer(name, value)
    return getattr(model, name)


def _model_solve_shape(model) -> Tuple[int, int]:
    return solver_shape(model)


def _flatten_to_solve(
    x: Optional[torch.Tensor],
    K: Optional[int] = None,
    target_shape: Optional[Sequence[int]] = None,
) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if target_shape is not None:
        return flatten_model_tensor(x, target_shape, core_ndim=1)
    K = int(x.shape[-1] if K is None else K)
    return x.reshape(-1, K)


def _broadcast_to_shape(x: torch.Tensor, target_shape: Sequence[int]) -> torch.Tensor:
    """Broadcast ``x`` to ``target_shape``, allowing omitted leading dimensions."""
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    target_shape = _shape_tuple(target_shape)
    if x.ndim > len(target_shape):
        raise ValueError(
            f"Cannot broadcast tensor with shape {tuple(x.shape)} to target shape {target_shape}."
        )
    if x.ndim < len(target_shape):
        x = x.reshape((1,) * (len(target_shape) - x.ndim) + tuple(x.shape))
    return x.expand(target_shape)


def _as_solve_matrix(x: torch.Tensor, model) -> torch.Tensor:
    """Broadcast ``x`` to ``model.shape`` and flatten to ``(B, K)``."""
    return flatten_model_tensor(x, tuple(model.shape), core_ndim=1)


def _as_solve_block(x: torch.Tensor, model, trailing_shape) -> torch.Tensor:
    """Broadcast ``x`` to ``model.shape + trailing_shape`` and flatten to ``(B, K, *trailing)``."""
    trailing_shape = tuple(int(v) for v in trailing_shape)
    return flatten_model_tensor(
        x, tuple(model.shape) + trailing_shape, core_ndim=1 + len(trailing_shape)
    )


def _expanded_v_init(model):
    """
    Return ``model.v_init`` expanded to ``model.v.shape``.

    Population implements ``expanded_v_init`` directly. The fallback keeps
    integrators robust for Network/MultiIntegrator-like containers that expose
    only ``v`` and ``v_init``.
    """
    if hasattr(model, "expanded_v_init"):
        return model.expanded_v_init()

    target = model.v
    target_shape = tuple(target.shape)
    if len(target_shape) < 1:
        raise ValueError(
            f"model.v must have at least one dimension; got {target_shape}."
        )

    v0 = torch.as_tensor(model.v_init, device=target.device, dtype=target.dtype)

    if v0.ndim == 0 or v0.numel() == 1:
        return v0.reshape(()).expand_as(target)

    n_comp = int(getattr(model, "nc", target_shape[-1]))
    if target_shape[-1] != n_comp:
        n_comp = int(target_shape[-1])

    if v0.ndim == 1:
        if v0.numel() != n_comp:
            raise ValueError(
                f"v_init has length {v0.numel()}, but the voltage state expects "
                f"length {n_comp}. Use a scalar or a vector matching model.nc/the "
                "last voltage dimension."
            )
        view_shape = (1,) * (target.ndim - 1) + (n_comp,)
        return v0.reshape(view_shape).expand_as(target)

    if tuple(v0.shape) == target_shape:
        return v0

    if target.ndim >= 2 and tuple(v0.shape) == tuple(target_shape[-2:]):
        view_shape = (1,) * (target.ndim - 2) + tuple(target_shape[-2:])
        return v0.reshape(view_shape).expand_as(target)

    if target.ndim >= 2 and tuple(v0.shape) == (1, target_shape[-1]):
        view_shape = (1,) * (target.ndim - 2) + (1, target_shape[-1])
        return v0.reshape(view_shape).expand_as(target)

    raise ValueError(
        "Unsupported v_init shape. Expected a scalar, a 1D vector matching "
        f"model.nc/the last voltage dimension ({n_comp}), the model core shape, "
        f"or full voltage shape {target_shape}; got shape {tuple(v0.shape)}."
    )


class Integrator(torch.nn.Module):
    r"""Base class for all integrators.

    Notes
    -----
    When ``imem`` is enabled, every stable voltage integrator exposes
    ``model.i_membrane`` as absolute transmembrane current in mA: capacitive
    current plus ionic/mechanism current, using that integrator's per-step
    voltage and current discretization. Dendra's mechanism-current convention
    is outward-positive. Applied intracellular stimulus and axial cable
    currents influence ``i_membrane`` through the solved voltage, but are not
    themselves added to or subtracted from the reported transmembrane current.
    """

    __constants__ = {"imem"}
    v_vars = ["v"]

    def __init__(self, model, mech, imem=None):
        super().__init__()
        imem = imem if imem is not None else IMEM
        self.imem = bool(imem)
        self.mech = mech
        self.initialized = False
        self.dt = None
        self.shape = None
        self._compiled_kernels = {}
        self.compile_scope = "population"
        self.configure_jit(model, scope="population")

    def _advance_pre_current(self, v, dt, temp):
        """Advance mechanism state that does not depend on this step's current.

        ``MechanismHandler`` exposes a phased scheduler. The fallback keeps
        lightweight/custom mechanism handlers compatible with the integrator
        API used by tests and downstream projects.
        """
        advance = getattr(self.mech, "advance_pre_current", None)
        if advance is None:
            self.mech.advance(v, dt, temp)
        else:
            advance(v, dt, temp)

    def _advance_post_current(self, v, dt, temp, ion_current_frame):
        """Commit an accepted ionic-current frame and finish mechanism state."""
        advance = getattr(self.mech, "advance_post_current", None)
        if advance is not None:
            advance(v, dt, temp, ion_current_frame)

    def _capture_ion_current_frame(self):
        """Capture the current evaluation's reader-facing per-ion currents."""
        capture = getattr(self.mech, "capture_ion_current_frame", None)
        return () if capture is None else capture()

    def _capture_ion_conductance_frame(self):
        """Capture per-ion conductances aligned with the current frame."""
        capture = getattr(self.mech, "capture_ion_conductance_frame", None)
        return () if capture is None else capture()

    @staticmethod
    def _combine_ion_current_frames(frames, weights):
        """Combine solver-stage frames without detaching their autograd graphs."""
        if not frames or not frames[0]:
            return ()
        return tuple(
            sum(weight * frame[index] for frame, weight in zip(frames, weights))
            for index in range(len(frames[0]))
        )

    @staticmethod
    def _linearize_ion_current_frame(
        current_frame,
        conductance_frame,
        delta_v,
        conductance_scale=1.0,
    ):
        """Evaluate an affine per-ion frame at a solver-selected voltage."""
        return tuple(
            current + conductance_scale * conductance * delta_v
            for current, conductance in zip(current_frame, conductance_frame)
        )

    def configure_jit(self, model, *, scope: str = "population"):
        """Copy compiler settings from the owner for a specific execution scope.

        ``scope="population"`` is used for standalone Population.run/longrun.
        ``scope="network_population"`` is used when a Network steps this
        population.  The old unconditional ``model.jit or model.jit_in_network``
        rule is intentionally avoided so that network-only solve JIT does not
        leak into standalone population runs.
        """
        owner_compile_options = getattr(model, "compile_options", None)
        if owner_compile_options is None:
            owner_compile_options = current_compile_options()
        compile_options = normalize_compile_options(owner_compile_options)
        new_config = (
            bool(jit_enabled_for_scope(scope, model)),
            getattr(model, "backend", _cfg_value(BACKEND)),
            bool(getattr(model, "fullgraph", bool(FULLGRAPH))),
            bool(getattr(model, "dynamic", bool(DYNAMIC))),
            getattr(model, "compile_mode", _cfg_value(COMPILE_MODE)),
            compile_options_key(compile_options),
            scope,
        )
        old_config = (
            getattr(self, "jit", None),
            getattr(self, "backend", None),
            getattr(self, "fullgraph", None),
            getattr(self, "dynamic", None),
            getattr(self, "compile_mode", None),
            getattr(self, "compile_options_key", None),
            getattr(self, "compile_scope", None),
        )
        (
            self.jit,
            self.backend,
            self.fullgraph,
            self.dynamic,
            self.compile_mode,
            self.compile_options_key,
            self.compile_scope,
        ) = new_config
        self.compile_options = compile_options
        if new_config != old_config:
            self._compiled_kernels.clear()
        return self

    def _compile_kwargs(self):
        kwargs = dict(
            backend=self.backend,
            fullgraph=self.fullgraph,
            dynamic=self.dynamic,
        )
        if self.compile_mode is not None:
            kwargs["mode"] = self.compile_mode
        if self.compile_options is not None:
            kwargs["options"] = dict(self.compile_options)
        return kwargs

    def clear_jit_cache(self):
        """Drop lazily generated ``torch.compile`` kernels.

        Compiled callables retain TorchDynamo/Inductor configuration objects that
        are process-local implementation details and are not pickleable.  The
        kernels are generated lazily by :meth:`_kernel`, so clearing this cache
        is safe: the next compiled step recreates the functions if JIT is still
        enabled.
        """
        if hasattr(self, "_compiled_kernels"):
            self._compiled_kernels.clear()
        return self

    def pickleable(
        self,
        *,
        inplace: bool = False,
        clone: bool = False,
        reset_global_compiler: bool = False,
    ):
        """Return a pickle-friendly integrator handle.

        By default this is non-mutating and relies on :meth:`__getstate__` to
        omit compiled kernels from the serialized state.  Use ``inplace=True``
        only when you intentionally want to drop the live integrator's compiled
        cache.
        """
        if inplace and clone:
            raise ValueError(
                "pickleable(...): choose at most one of inplace=True or clone=True."
            )
        if clone:
            import copy as _copy

            obj = _copy.deepcopy(self)
            obj.clear_jit_cache()
        elif inplace:
            obj = self.clear_jit_cache()
        else:
            obj = self
        if reset_global_compiler:
            reset_torch_compiler()
        return obj

    def __getstate__(self):
        """Serialize without process-local compiled kernels."""
        state = self.__dict__.copy()
        if "_compiled_kernels" in state:
            state["_compiled_kernels"] = {}
        return state

    def _kernel(self, name: str, *args, **kwargs):
        """
        Call an integrator-owned numerical kernel, compiling it if requested.

        ``step(model, ...)`` should stay as an ordinary Python state-commit
        wrapper.  That keeps ``model.v = ...`` / ``model.vc = ...`` outside
        Dynamo while still compiling the tensor-heavy transition.
        """
        fn = getattr(self, name)
        if not self.jit:
            return fn(*args, **kwargs)
        key = (
            name,
            self.compile_scope,
            self.backend,
            self.fullgraph,
            self.dynamic,
            self.compile_mode,
            self.compile_options_key,
        )
        with torch_compiler_warning_context():
            compiled = self._compiled_kernels.get(key)
            if compiled is None:
                compiled = torch.compile(fn, **self._compile_kwargs())
                self._compiled_kernels[key] = compiled
            return compiled(*args, **kwargs)

    def _sample_runtime_noises(self):
        """Refresh detached runtime NOISE buffers before the compiled step kernel.

        Keeping this eager avoids putting the fast in-place NOISE path inside the
        integrator's ``torch.compile`` region. State-level SDE increments remain
        the responsibility of the generated State solver.
        """
        if hasattr(self.mech, "sample_runtime_noises_"):
            self.mech.sample_runtime_noises_(dt=self.dt, phase="pre_state")

    def _call_kernel(self, name: str, *args, **kwargs):
        self._sample_runtime_noises()
        return self._kernel(name, *args, **kwargs)

    def _refresh_solver_shape(self, model, *, block_dim=None):
        self.shape = tuple(model.shape)
        self.B, self.K = solver_shape(model)
        self.base_shape = (
            self.shape if block_dim is None else tuple(self.shape) + (int(block_dim),)
        )

    def _flat_voltage(self, x: Optional[torch.Tensor], K: Optional[int] = None):
        if x is None:
            return None
        K = int(self.K if K is None else K)
        return x.reshape(-1, K)

    def _flat_block_voltage(
        self, x: Optional[torch.Tensor], M: int, K: Optional[int] = None
    ):
        if x is None:
            return None
        K = int(self.K if K is None else K)
        return x.reshape(-1, K, int(M))

    def _restore_voltage(self, x: torch.Tensor, shape=None):
        return x.reshape(self.shape if shape is None else tuple(shape))

    def _restore_block_voltage(self, x: torch.Tensor, shape=None):
        return x.reshape(self.base_shape if shape is None else tuple(shape))

    def initialize(self, model, dt):
        raise NotImplementedError

    def step(self, model, dt, ve=None, intra=None):
        raise NotImplementedError

    def needs_to_be_initialized(self, model, dt, force=False):
        if force:
            return True
        return not self.initialized or self.dt != float(dt) or self.shape != model.shape

    def _initialize(self, model, dt, force=False, *, compile_scope: str = "population"):
        needs_initialize = self.needs_to_be_initialized(model, dt, force)
        if needs_initialize:
            # Geometry-derived workspaces must never be built from a model that
            # violates its static contracts. This hook is independent of the
            # public runtime-validation policy and runs before this integrator
            # mutates compiler settings, timestep state, or solver buffers.
            validate = getattr(model, "_validate_integrator_rebuild_contracts", None)
            if validate is not None:
                validate()

        self.configure_jit(model, scope=compile_scope)
        if needs_initialize:
            previous_dt = self.dt
            previous_shape = self.shape
            previous_initialized = self.initialized
            self.dt = float(dt)
            self.shape = model.shape
            self._compiled_kernels.clear()
            try:
                if hasattr(self.mech, "set_dt"):
                    self.mech.set_dt(dt)
                else:
                    for mech in self.mech.mechanisms.values():
                        mech.set_dt(dt)
                self.initialize(model, dt)
                record = getattr(model, "_record_integrator_workspace_contracts", None)
                if record is not None:
                    record()
            except Exception:
                # An initializer may have rebound only some workspaces before
                # failing. Never advertise that partial state as initialized or
                # let a retry at the same dt/shape skip reconstruction.
                self.dt = previous_dt
                self.shape = previous_shape
                self.initialized = False
                self._compiled_kernels.clear()
                if previous_initialized and previous_dt is not None:
                    try:
                        if hasattr(self.mech, "set_dt"):
                            self.mech.set_dt(previous_dt)
                        else:
                            for mech in self.mech.mechanisms.values():
                                mech.set_dt(previous_dt)
                    except Exception:
                        # The integrator remains explicitly invalid. A later
                        # initialize call must rebuild all timestep-dependent
                        # state rather than using any partially restored cache.
                        pass
                raise
            self.initialized = True
            complete_reinitialization = getattr(
                model,
                "_complete_integrator_reinitialization",
                None,
            )
            if complete_reinitialization is not None:
                complete_reinitialization()

    def init_v(self, model):
        model.v = _expanded_v_init(model).clone().detach().contiguous()
        if self.imem:
            model.i_membrane = torch.zeros_like(model.v).detach()

    def detach(self, model):
        detach_vars(model, self.v_vars)
        for n, b in model.named_buffers():
            setattr(model, n, b.detach())
        if self.imem:
            model.i_membrane = model.i_membrane.detach()
        self.mech.detach()

    def mutable_state_dict(self, model):
        dct = {}
        for var in self.v_vars:
            dct[var] = getattr(model, var)
        if self.imem:
            dct["i_membrane"] = model.i_membrane
        return dct

    def restore_mutable_state_dict(self, model, state_dict):
        for var in self.v_vars:
            setattr(model, var, state_dict[var])
        if self.imem:
            model.i_membrane = state_dict["i_membrane"]


@torch._dynamo.disable
def _write_back(model, split_at):
    # write v back to the constituent populations
    v_f = model.v
    splits = torch.tensor_split(v_f, split_at, dim=-1)
    for split, pop in zip(splits, model.populations.values()):
        pop.v = split.reshape_as(pop.v)


class MultiIntegrator(Integrator):
    def __init__(self, model, mech, imem=None, write_back=True):
        super().__init__(model, mech, imem)
        self.write_back = write_back
        self.split_at = None

    def _write_back(self, model):
        if self.write_back:
            _write_back(model, self.split_at)

    def _calc_splits(self, models):
        split_lengths = [m.numelc() for m in models]
        self.split_at = torch.cumsum(torch.tensor(split_lengths), dim=0)[:-1].tolist()

    def init_v(self, model):
        model.v = _expanded_v_init(model).clone().detach().contiguous()
        if self.write_back:
            self._calc_splits(model)
            _write_back(model, self.split_at)
        if self.imem:
            model.i_membrane = torch.zeros_like(model.v).detach()
