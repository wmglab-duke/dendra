import inspect
import math
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import FunctionType, MethodType
from typing import Optional, Sequence, Tuple

import torch

from dendra._bootstrap import (
    isolate_compilation_callable,
    reset_torch_compiler,
    torch_compiler_warning_context,
)
from dendra.helpers import (
    BACKEND,
    COMPILE_MODE,
    DYNAMIC,
    FULLGRAPH,
    IMEM,
    MEMBRANE_ASSEMBLY,
    compile_options_for_device,
    compile_options_key,
    current_compile_options,
    detach_vars,
    jit_enabled_for_scope,
    normalize_compile_options,
)

try:
    from torch.utils._device import DeviceContext as _POPULATION_DEVICE_CONTEXT

    _POPULATION_DEVICE_TORCH_FUNCTION = _POPULATION_DEVICE_CONTEXT.__torch_function__
except (ImportError, AttributeError):
    _POPULATION_DEVICE_CONTEXT = None
    _POPULATION_DEVICE_TORCH_FUNCTION = None


def get_init_defaults(cls):
    signature = inspect.signature(cls.__init__)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty and k != "self"
    }


def _cfg_value(value):
    return getattr(value, "value", value)


def _kernel_callable_source(function):
    """Keep source identity without relying on transient bound-method objects."""
    owner = function.__self__ if isinstance(function, MethodType) else None
    source = function.__func__ if isinstance(function, MethodType) else function
    if isinstance(source, FunctionType):
        return (
            owner,
            source,
            source.__code__,
            source.__defaults__,
            source.__kwdefaults__,
        )
    return (owner, source)


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


@dataclass(frozen=True)
class _DeferredIntra:
    """One-step stimulus inputs for a proven noise-free built-in transition.

    Population run/longrun construct this payload after validating their runtime
    and callback eligibility. It is never stored in model or integrator state.
    """

    source: object
    stimuli: object
    indices: object


@dataclass(frozen=True)
class _NoiseFreeStep:
    """One-call proof that the ordinary runtime-NOISE traversal is empty.

    This payload never becomes persistent model state, so failures and nested
    execution cannot leave sampling disabled for a later call.
    """

    intra: object = None


@dataclass(frozen=True)
class _FunctionalIntegratorSpec:
    """Private capability record consumed by ``dendra.func`` lowering."""

    operator_kind: str
    implementation: type
    workspace_schema: tuple[tuple[str, str], ...]
    critical_methods: tuple[str, ...]


def _population_runtime_modes_plain(*, check_saved_tensors=True):
    """Fail closed when ambient Torch execution can invoke authored code.

    This runs outside Dynamo at a segment boundary. The ordinary default-device
    context used by ``dendra.ctx`` is permitted; dispatch modes, other function
    modes, and saved-tensor hooks keep per-step dynamic validation and sampling.
    Private introspection is feature-detected so unavailable APIs only disable
    this optimization, never ordinary execution.

    Operators that save no tensors may omit the saved-tensor hook check while
    retaining all dispatch and function-mode checks.
    """
    try:
        dispatch_length = getattr(torch._C, "_len_torch_dispatch_stack", None)
        function_length = getattr(torch._C, "_len_torch_function_stack", None)
        function_at = getattr(torch._C, "_get_function_stack_at", None)
        pre_dispatch_length = getattr(
            getattr(torch, "_ops", None), "_len_torch_dispatch_stack_pre_dispatch", None
        )
        saved_hooks = getattr(
            getattr(torch._C, "_autograd", None),
            "_top_saved_tensors_default_hooks",
            None,
        )
        if not all(
            callable(probe)
            for probe in (
                dispatch_length,
                function_length,
                function_at,
                pre_dispatch_length,
                saved_hooks,
            )
        ):
            return False
        # Ignore a tracing flag that could otherwise hide active user hooks.
        if check_saved_tensors and saved_hooks(True) is not None:
            return False
        dispatch_count = dispatch_length()
        pre_dispatch_count = pre_dispatch_length()
        function_count = function_length()
        if (
            type(dispatch_count) is not int
            or dispatch_count != 0
            or type(pre_dispatch_count) is not int
            or pre_dispatch_count != 0
            or type(function_count) is not int
            or function_count < 0
        ):
            return False
        for index in range(function_count):
            mode = function_at(index)
            mode_type = type(mode)
            if (
                mode_type is not _POPULATION_DEVICE_CONTEXT
                or mode_type.__torch_function__ is not _POPULATION_DEVICE_TORCH_FUNCTION
                or mode_type.__getattribute__ is not object.__getattribute__
                or getattr(mode_type, "__getattr__", None) is not None
            ):
                return False
            namespace = object.__getattribute__(mode, "__dict__")
            if (
                type(namespace) is not dict
                or "__torch_function__" in namespace
                or type(namespace.get("device")) is not torch.device
            ):
                return False
        return True
    except Exception:
        return False


def _plain_membrane_metadata(value, *, depth=8):
    """Reject metadata containers that can dispatch authored iteration/truth."""
    if type(value) in (str, int, float, bool, type(None)):
        return True
    if depth == 0:
        return False
    if type(value) is dict:
        return all(
            _plain_membrane_metadata(key, depth=depth - 1)
            and _plain_membrane_metadata(item, depth=depth - 1)
            for key, item in value.items()
        )
    if type(value) in (tuple, list, set, frozenset):
        return all(_plain_membrane_metadata(item, depth=depth - 1) for item in value)
    return False


def _population_membrane_metadata_plain(handler):
    """Check only structural fields whose repeated eligibility walk is skipped."""
    if type(handler._membrane_map_supported) is not bool or any(
        type(getattr(handler, name)) is not torch.nn.ModuleDict
        for name in (
            "mechanisms",
            "ions",
            "materials",
            "voltage_processes",
            "material_processes",
        )
    ):
        return False
    if not all(
        _plain_membrane_metadata(getattr(handler, name))
        for name in (
            "currents",
            "read_ion",
            "write_ion_c",
            "read_material",
            "write_material",
            "source_material",
            "_ion_current_reads",
            "_membrane_current_signature",
            "_membrane_owner_currents",
        )
    ):
        return False
    owners = handler._membrane_owners
    registered = handler.mechanisms._modules
    if (
        type(owners) is not tuple
        or type(registered) is not dict
        or any(type(name) is not str for name in registered)
    ):
        return False
    for entry in owners:
        if type(entry) is not tuple or len(entry) != 2 or type(entry[0]) is not str:
            return False
        mechanism = entry[1]
        if registered.get(entry[0]) is not mechanism:
            return False
        if type(mechanism._support_key_values_valid) is not bool or not all(
            _plain_membrane_metadata(getattr(mechanism, name))
            for name in (
                "_save",
                "_write_ion",
                "_read_ion",
                "_read_material",
                "_write_material",
                "_source_material",
                "_currents",
            )
        ):
            return False
    return True


@contextmanager
def _owned_runtime_noise_frame(module):
    """Lend a compiled training call detached NOISE inputs it can retain.

    Native sampling updates its persistent buffers in place. AOT backward can
    retain those inputs even when eager autograd would only save an expression
    derived from them. Bind one private copy per physical NOISE tensor for this
    invocation, including all registered aliases, then restore the live buffer
    registrations before the caller observes or samples the model again.
    """
    modules = tuple(module.modules())
    identities = {
        id(owner._buffers[name])
        for owner in modules
        for name in getattr(owner, "runtime_noises", {})
    }
    if not identities:
        yield
        return
    copies, originals = {}, []
    try:
        for owner in modules:
            for name, value in owner._buffers.items():
                if id(value) not in identities:
                    continue
                if id(value) not in copies:
                    copies[id(value)] = value.detach().clone(
                        memory_format=torch.preserve_format
                    )
                originals.append((owner, name, value))
                owner._buffers[name] = copies[id(value)]
        yield
    finally:
        for owner, name, value in reversed(originals):
            owner._buffers[name] = value


@contextmanager
def _population_step_context(integrator, *, stable=False):
    """Temporarily reuse a proof supplied by an observer-stable host loop."""
    previous = integrator._direct_membrane_structure_prevalidated
    integrator._direct_membrane_structure_prevalidated = False
    try:
        if (
            stable
            and not integrator.imem
            and MEMBRANE_ASSEMBLY.value != "off"
            and _population_runtime_modes_plain()
        ):
            from ..mechanisms._handler import (
                _MEMBRANE_COEFFICIENT_FRAME,
                _MEMBRANE_DIRECT_ELIGIBILITY,
                MechanismHandler,
            )

            handler = integrator.mech
            if (
                type(handler) is MechanismHandler
                and type(handler)._evaluate_membrane_frame
                is _MEMBRANE_COEFFICIENT_FRAME
                and type(handler)._can_evaluate_direct_membrane_frame
                is _MEMBRANE_DIRECT_ELIGIBILITY
                and "_evaluate_membrane_frame" not in handler.__dict__
                and "_can_evaluate_direct_membrane_frame" not in handler.__dict__
                and _population_membrane_metadata_plain(handler)
                and handler._can_evaluate_direct_membrane_frame()
            ):
                integrator._direct_membrane_structure_prevalidated = True
        yield
    finally:
        integrator._direct_membrane_structure_prevalidated = previous


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
    _PREPARED_WORKSPACE_SCHEMA = ()
    _FUNCTIONAL_OPERATOR_KIND = None
    _FUNCTIONAL_CRITICAL_METHODS = ()
    _DIRECT_MEMBRANE_VOLTAGE_UPDATE = None
    # Old serialized integrators predate the instance marker. An absent field
    # must retain the ordinary per-step structural validation after restoration.
    _direct_membrane_structure_prevalidated = False

    def __init__(self, model, mech, imem=None):
        super().__init__()
        imem = imem if imem is not None else IMEM
        self.imem = bool(imem)
        self.mech = mech
        self.initialized = False
        self.dt = None
        self.shape = None
        self._compiled_kernels = {}
        self._compiled_kernel_sources = {}
        self._direct_membrane_structure_prevalidated = False
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

    def _membrane_frame(self, v, *, call_local=False, require_current=None):
        """Return ``(G, J, I, ion_I, ion_G)`` for the current solver stage.

        ``J = G*v-I`` is the linearized membrane source. A capable handler can
        supply proven direct coefficients with ``I=None`` when no current
        observer needs the authored current. Consumers keep their original
        arithmetic whenever ``I`` is present, including observer and custom
        handler fallbacks. Mechanism advance and frame publication stay owned
        by the integrator, outside this evaluation.
        """
        if require_current is None:
            require_current = self.imem
        evaluate = getattr(self.mech, "_evaluate_membrane_frame", None)
        if evaluate is not None:
            if self._direct_membrane_structure_prevalidated:
                return evaluate(
                    v,
                    call_local=call_local,
                    require_current=require_current,
                    _structure_prevalidated=True,
                )
            return evaluate(v, call_local=call_local, require_current=require_current)
        if call_local:
            evaluate = getattr(self.mech, "_evaluate_current_frame", None)
            if evaluate is None:
                raise RuntimeError(
                    "call-local current evaluation requires a MechanismHandler "
                    "with _evaluate_current_frame()"
                )
            current, conductance, ion_i, ion_g = evaluate(v)
        else:
            current, conductance = self.mech.i(v)
            ion_i = self._capture_ion_current_frame()
            ion_g = self._capture_ion_conductance_frame()
        return conductance, conductance * v - current, current, ion_i, ion_g

    def _population_step_scope(self, *, stable=False):
        """Reuse one structural proof only inside an observer-stable host loop.

        The caller proves that built-in execution and read-only callbacks cannot
        replace current declarations within this run/chunk. Tensor values,
        compiler guards, dtype checks, and membrane-assembly policy remain live.
        Unknown nested execution deliberately disables any enclosing proof.
        """
        # Keep this class method source-owned: the runtime stability proof
        # rejects authored/contextlib executable wrappers on module classes.
        return _population_step_context(self, stable=stable)

    def _prepared_workspace_schema(self):
        """Return the validated private tensor-workspace contract.

        Concrete integrators declare ``(buffer_name, shape_role)`` entries and
        implement a pure ``_prepare_workspace(dt, **physical_inputs)`` builder.
        The functional backend interprets shape roles; the base Integrator owns
        structural validation so imperative and functional preparation cannot
        silently disagree about names or tensor ownership.
        """
        schema = tuple(self._PREPARED_WORKSPACE_SCHEMA)
        names = []
        for entry in schema:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not all(isinstance(value, str) and value for value in entry)
            ):
                raise TypeError(
                    f"{type(self).__qualname__}._PREPARED_WORKSPACE_SCHEMA must "
                    "contain non-empty (buffer_name, shape_role) string pairs."
                )
            names.append(entry[0])
        if len(names) != len(set(names)):
            raise ValueError(
                f"{type(self).__qualname__}._PREPARED_WORKSPACE_SCHEMA contains "
                "duplicate buffer names."
            )
        if "dt" in names:
            raise ValueError(
                f"{type(self).__qualname__}._PREPARED_WORKSPACE_SCHEMA cannot "
                "declare reserved entry 'dt'."
            )
        return schema

    def _derive_prepared_workspace(self, dt, **physical_inputs):
        """Build and validate a pure tensor workspace for one timestep.

        Topology adapters remain responsible for producing effective physical
        inputs in the integrator's documented layout. This method is the shared
        numerical-preparation boundary used by both ordinary initialization and
        ``dendra.func`` lowering.
        """
        builder = getattr(self, "_prepare_workspace", None)
        if not callable(builder):
            raise NotImplementedError(
                f"{type(self).__qualname__} does not define a pure "
                "_prepare_workspace() builder."
            )
        schema = self._prepared_workspace_schema()
        workspace = builder(dt, **physical_inputs)
        if not isinstance(workspace, Mapping):
            raise TypeError(
                f"{type(self).__qualname__}._prepare_workspace() must return a mapping."
            )

        # Resolve and validate the complete generation before returning it. In
        # particular, a missing late entry cannot expose a partially installed
        # workspace to the imperative model.
        staged = tuple((name, workspace[name]) for name, _role in schema)
        expected = {name for name, _role in schema}
        unexpected = set(workspace) - expected
        if unexpected:
            raise KeyError(
                f"{type(self).__qualname__}._prepare_workspace() returned "
                f"unexpected entries {sorted(unexpected)}."
            )
        non_tensors = [name for name, value in staged if not torch.is_tensor(value)]
        if non_tensors:
            raise TypeError(
                f"{type(self).__qualname__}._prepare_workspace() returned "
                f"non-Tensor entries {non_tensors}."
            )
        return dict(staged)

    def _install_prepared_workspace(self, workspace):
        """Atomically install one already validated workspace generation."""
        schema = self._prepared_workspace_schema()
        staged = tuple((name, workspace[name]) for name, _role in schema)
        missing_buffers = [name for name, _value in staged if name not in self._buffers]
        if missing_buffers:
            raise RuntimeError(
                f"{type(self).__qualname__} workspace entries must be registered "
                f"buffers before installation; missing {missing_buffers}."
            )
        for name, value in staged:
            setattr(self, name, value)
        return workspace

    def _functional_spec(self):
        """Describe this integrator's private functional capability, if any."""
        implementation = None
        operator_kind = None
        for owner in type(self).__mro__:
            declared = owner.__dict__.get("_FUNCTIONAL_OPERATOR_KIND")
            if declared is not None:
                implementation = owner
                operator_kind = declared
                break
        if implementation is None:
            return None

        common_methods = (
            "_functional_spec",
            "_functional_solver",
            "_step",
            "_membrane_frame",
            "_voltage_update",
            "_DIRECT_MEMBRANE_VOLTAGE_UPDATE",
            "_prepare_workspace",
            "_prepared_workspace_schema",
            "_derive_prepared_workspace",
            "_install_prepared_workspace",
            "initialize",
            "step",
        )
        critical_methods = tuple(
            dict.fromkeys((*common_methods, *self._FUNCTIONAL_CRITICAL_METHODS))
        )
        return _FunctionalIntegratorSpec(
            operator_kind=str(operator_kind),
            implementation=implementation,
            workspace_schema=self._prepared_workspace_schema(),
            critical_methods=critical_methods,
        )

    def _functional_solver(self):
        """Return a transform-compatible solver facade, if one is needed."""
        return None

    def _step(
        self,
        v,
        dt,
        temp,
        ve=None,
        intra=None,
        *,
        solver=None,
        call_local_currents=False,
    ):
        """Run one linearized implicit transition.

        Single-compartment and unbranched-cable implicit Euler share this
        lifecycle exactly; concrete integrators own only their numerical
        :meth:`_voltage_update`. Ordinary execution uses handler-owned scratch.
        Functional execution selects call-local current frames and must isolate
        registered mechanism state around this method.
        """
        v_state = self.mech.update_v(v)
        self._advance_pre_current(v_state, dt, temp)
        direct_update = (
            "_voltage_update" not in self.__dict__
            and type(self)._voltage_update is type(self)._DIRECT_MEMBRANE_VOLTAGE_UPDATE
        )
        (
            gtot,
            membrane_source,
            itot,
            ion_current_frame,
            ion_conductance_frame,
        ) = self._membrane_frame(
            v_state,
            call_local=call_local_currents,
            require_current=self.imem or not direct_update,
        )
        if direct_update:
            v_new, i_membrane = self._voltage_update(
                v_state,
                dt,
                itot,
                gtot,
                ve,
                intra,
                solver=solver,
                membrane_source=membrane_source,
            )
        else:
            # An authored numerical override retains its original signature
            # and actual-current inputs, even when its model is affine.
            v_new, i_membrane = self._voltage_update(
                v_state,
                dt,
                itot,
                gtot,
                ve,
                intra,
                solver=solver,
            )
        accepted_frame = self._linearize_ion_current_frame(
            ion_current_frame,
            ion_conductance_frame,
            v_new - v_state,
        )
        self._advance_post_current(v_state, dt, temp, accepted_frame)
        return v_new, i_membrane

    def _voltage_update(
        self,
        v,
        dt,
        itot,
        gtot,
        ve=None,
        intra=None,
        *,
        solver=None,
        membrane_source=None,
    ):
        """Return the pure numerical voltage update shared by both frontends."""
        raise NotImplementedError(
            f"{type(self).__qualname__} does not define _voltage_update()."
        )

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
        requested_compile_options = normalize_compile_options(owner_compile_options)
        model_device = model.device()
        compile_device_type = torch.device(model_device).type
        backend = getattr(model, "backend", _cfg_value(BACKEND))
        dynamic = bool(getattr(model, "dynamic", bool(DYNAMIC)))
        requested_compile_mode = getattr(
            model, "compile_mode", _cfg_value(COMPILE_MODE)
        )
        mechanism_handler = getattr(model, "mech", None)
        if mechanism_handler is None:
            # During Population.build(), the handler is passed to the
            # integrator before Population.mech is assigned.
            mechanism_handler = self.mech
        requires_inductor_python_wrapper = bool(
            getattr(
                mechanism_handler,
                "requires_inductor_python_wrapper",
                False,
            )
        )
        compile_options = compile_options_for_device(
            requested_compile_options,
            backend=backend,
            device=model_device,
            mode=requested_compile_mode,
            dynamic=dynamic,
            requires_inductor_python_wrapper=requires_inductor_python_wrapper,
        )
        compile_mode = requested_compile_mode if compile_options is None else None
        new_config = (
            bool(jit_enabled_for_scope(scope, model)),
            backend,
            bool(getattr(model, "fullgraph", bool(FULLGRAPH))),
            dynamic,
            compile_mode,
            compile_options_key(compile_options),
            compile_device_type,
            scope,
        )
        old_config = (
            getattr(self, "jit", None),
            getattr(self, "backend", None),
            getattr(self, "fullgraph", None),
            getattr(self, "dynamic", None),
            getattr(self, "compile_mode", None),
            getattr(self, "compile_options_key", None),
            getattr(self, "compile_device_type", None),
            getattr(self, "compile_scope", None),
        )
        (
            self.jit,
            self.backend,
            self.fullgraph,
            self.dynamic,
            self.compile_mode,
            self.compile_options_key,
            self.compile_device_type,
            self.compile_scope,
        ) = new_config
        self.requested_compile_mode = requested_compile_mode
        self.requested_compile_options = requested_compile_options
        self.compile_options = compile_options
        if new_config != old_config:
            self.clear_jit_cache()
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
        if hasattr(self, "_compiled_kernel_sources"):
            self._compiled_kernel_sources.clear()
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
        if "_compiled_kernel_sources" in state:
            state["_compiled_kernel_sources"] = {}
        state["_direct_membrane_structure_prevalidated"] = False
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
            self.compile_device_type,
        )
        with torch_compiler_warning_context():
            compiled = self._compiled_kernels.get(key)
            sources = self.__dict__.setdefault("_compiled_kernel_sources", {})
            source = _kernel_callable_source(fn)
            previous = sources.get(key)
            if (
                compiled is None
                or previous is None
                or any(old is not new for old, new in zip(previous, source))
                or len(previous) != len(source)
            ):
                # A cached bound method otherwise keeps calling its original
                # body after an instance/class method replacement.
                compiled = torch.compile(
                    isolate_compilation_callable(fn), **self._compile_kwargs()
                )
                self._compiled_kernels[key] = compiled
                sources[key] = source
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
        if args and isinstance(args[-1], _NoiseFreeStep):
            return self._kernel(name, *args[:-1], args[-1].intra, **kwargs)
        if args and isinstance(args[-1], _DeferredIntra):
            # Population run/longrun only defer scatter after proving that every
            # mechanism/State lacks runtime NOISE and uses a built-in sampler.
            # Rewalking that empty sampler tree cannot change this transition.
            # Ordinary/custom/noisy paths retain their sampling order below.
            intra = args[-1]
            return self._kernel(
                "_step_with_intra",
                name,
                args[:-1],
                intra.source,
                intra.stimuli,
                intra.indices,
                kwargs,
            )
        self._sample_runtime_noises()
        if self.jit and torch.is_grad_enabled():
            with _owned_runtime_noise_frame(self):
                return self._kernel(name, *args, **kwargs)
        return self._kernel(name, *args, **kwargs)

    def _step_with_intra(self, name, args, source, stimuli, indices, kwargs):
        """Scatter and advance in one graph, leaving state commit eager."""
        intra = source(stimuli, indices)
        return getattr(self, name)(*args, intra, **kwargs)

    def _supports_fused_intra_step(
        self,
        _canonical_call_kernel=_call_kernel,
        _canonical_kernel=_kernel,
        _canonical_sampler=_sample_runtime_noises,
        _canonical_fused_kernel=_step_with_intra,
    ):
        """Fail closed for custom step/dispatch wrappers and unknown solvers.

        The listed built-in wrappers pass ``intra`` as the final positional
        kernel argument, then commit the returned state eagerly. Partial-class
        constructors and subclasses inheriting these wrappers remain eligible.
        """
        from .explicit import (
            _dufort_frankel,
            _dufort_frankel_homogeneous,
            _euler,
            _eulerv1,
            _rk2,
            _rk4,
        )
        from .implicit import (
            _bwd_euler_bt,
            _bwd_euler_sc,
            _bwd_euler_sc_multi,
            _bwd_euler_sc_skip,
            _bwd_euler_ub,
        )
        from .tree import _dhs, _dhs_multi
        from .tree_bt import _dhs_bt

        owners = (
            _euler,
            _eulerv1,
            _rk2,
            _rk4,
            _dufort_frankel,
            _dufort_frankel_homogeneous,
            _bwd_euler_sc,
            _bwd_euler_sc_skip,
            _bwd_euler_sc_multi,
            _bwd_euler_ub,
            _bwd_euler_bt,
            _dhs,
            _dhs_multi,
            _dhs_bt,
        )
        # A class monkeypatch must not authorize itself by replacing both the
        # resolved step and the class attribute used for comparison. Built-in
        # methods originate in their defining class's source file. Inherited
        # kernels (for example Integrator._step used by bwd_euler_sc) are valid.
        step_wrapper = getattr(self.step, "__func__", None)
        step_kernel = getattr(self._step, "__func__", None)

        def declared_method_matches(owner, name, bound):
            for defining_class in owner.__mro__:
                if name in vars(defining_class):
                    declared = vars(defining_class)[name]
                    return (
                        bound is declared
                        and inspect.isfunction(declared)
                        and inspect.getfile(declared) == inspect.getfile(defining_class)
                    )
            return False

        canonical_pair = any(
            declared_method_matches(owner, "step", step_wrapper)
            and declared_method_matches(owner, "_step", step_kernel)
            for owner in owners
        )
        return (
            canonical_pair
            and getattr(self._call_kernel, "__func__", None) is _canonical_call_kernel
            and getattr(self._kernel, "__func__", None) is _canonical_kernel
            and getattr(self._sample_runtime_noises, "__func__", None)
            is _canonical_sampler
            and getattr(self._step_with_intra, "__func__", None)
            is _canonical_fused_kernel
        )

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

    def _configure_mechanism_timestep(self, dt):
        """Configure handler or fallback mechanism timestep state safely."""

        handler_set_dt = getattr(self.mech, "set_dt", None)
        if callable(handler_set_dt):
            handler_set_dt(dt)
            return

        # Lightweight downstream/test handlers may expose only a ``mechanisms``
        # collection. Ordinary Dendra Mechanisms no longer have a public
        # ``set_dt`` hook, so stage them through the framework-owned base
        # implementation while preserving the legacy protocol for non-Dendra
        # leaves and the separate MaterialProcess family lifecycle.
        from dendra.models.mechanisms._mechanism import Mechanism

        canonical = []
        family_or_legacy = []
        for mechanism in self.mech.mechanisms.values():
            if isinstance(mechanism, Mechanism) and not getattr(
                type(mechanism), "_material_process_set_dt_lifecycle", False
            ):
                canonical.append(
                    (
                        mechanism,
                        Mechanism._stage_timestep_configuration(mechanism, dt),
                    )
                )
            else:
                setter = getattr(mechanism, "set_dt", None)
                if not callable(setter):
                    raise TypeError(
                        "Fallback mechanism handlers require each non-Dendra "
                        "leaf to provide a callable set_dt(dt) method."
                    )
                family_or_legacy.append(setter)

        for setter in family_or_legacy:
            setter(dt)
        for mechanism, configuration in canonical:
            Mechanism._commit_timestep_configuration(mechanism, configuration)

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
            try:
                # Rebuild numerical workspaces without discarding compiled
                # callables. Kernels read the current buffers/parameters, and
                # Dynamo guards specialize changes to their metadata or Python
                # configuration. State resets with an unchanged compilation
                # contract can therefore retain their existing wrappers.
                self._configure_mechanism_timestep(dt)
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
                self.clear_jit_cache()
                if previous_initialized and previous_dt is not None:
                    try:
                        self._configure_mechanism_timestep(previous_dt)
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
