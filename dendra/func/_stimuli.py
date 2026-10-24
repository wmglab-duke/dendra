"""Pure tensor lowering for authored intracellular and extracellular stimuli.

This module deliberately separates a structural stimulation plan from the
tensor leaves used to evaluate it.  The imperative authoring objects remain
untouched: waveform modules are deep-copied into an execution bank and are
called with explicit parameters and buffers through :func:`functional_call`.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.func import functional_call

from dendra.helpers import current_check_purity, op_mc, op_sc
from dendra.models.core import Population
from dendra.models.parametric import SimpleParameterized, cacheable
from dendra.models.stim.intra import (
    _canonicalize_index_for_index_put,
    _retained_leading_batch_shape,
)
from dendra.models.stim.waveform import Waveform
from dendra.utils.tensor_ops import _logical_tensor_bytes

from ._module_traversal import _raw_module_namespace, _registered_modules
from ._rng_diagnostics import _numpy_rng_state_equal
from ._types import FunctionalizationError


@dataclass(frozen=True)
class TensorSchema:
    """Shape, dtype, and device contract for one explicit tensor leaf."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class StimulusTensors:
    """Plan-bound explicit tensor leaves for one functional stimulus.

    Use :meth:`FunctionalIntra.replace` or :meth:`FunctionalExtra.replace` for
    checked partial replacement. The bundle is a registered PyTorch PyTree:
    ``parameters`` and ``constants`` contain its tensor children, while the
    intentionally opaque plan token is static structural context that prevents
    bundles from two different plans from being confused.
    """

    parameters: dict[str, torch.Tensor]
    constants: dict[str, torch.Tensor]
    _token: object

    @property
    def plan_token(self) -> object:
        """Opaque identity of the plan that produced this bundle."""

        return self._token


def _flatten_stimulus_tensors(
    tensors: StimulusTensors,
) -> tuple[list[object], object]:
    """Expose only tensor-bearing mappings as PyTree children.

    The opaque plan token is structural context rather than differentiable
    data. Keeping it in the TreeSpec lets transforms reconstruct a bundle that
    remains tied to the exact plan that produced it.
    """

    return [tensors.parameters, tensors.constants], tensors.plan_token


def _unflatten_stimulus_tensors(
    children: Sequence[object], plan_token: object
) -> StimulusTensors:
    parameters, constants = children
    return StimulusTensors(parameters, constants, plan_token)


torch.utils._pytree.register_pytree_node(
    StimulusTensors,
    _flatten_stimulus_tensors,
    _unflatten_stimulus_tensors,
)


def _shape(value: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(size) for size in value.shape)


def _population_contract(
    population: Population,
) -> tuple[tuple[int, ...], tuple[int, ...], torch.device, torch.dtype]:
    if not isinstance(population, Population):
        raise TypeError("population must be a dendra Population")
    try:
        shape = tuple(int(size) for size in population.v.shape)
        device = torch.device(population.device())
        dtype = population.dtype()
    except (AttributeError, RuntimeError) as exc:
        raise FunctionalizationError(
            "Functional stimulation requires an initialized Population."
        ) from exc
    if not shape:
        raise FunctionalizationError(
            "Functional stimulation requires a non-scalar Population."
        )
    if len(shape) < 2:
        raise FunctionalizationError(
            "Population stimulation expects neuron and compartment axes."
        )
    return shape, shape[:-2], device, dtype


def _tensor_on(
    value: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _schema(value: torch.Tensor) -> TensorSchema:
    return TensorSchema(_shape(value), value.dtype, value.device)


_FUNCTIONAL_PURITY_CAPABILITY = "FUNCTIONAL_PURE"
_MODULE_INTERNAL_STATE = {
    "_parameters",
    "_buffers",
    "_non_persistent_buffers_set",
    "_backward_pre_hooks",
    "_backward_hooks",
    "_is_full_backward_hook",
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
    "_modules",
}
_SIMPLE_PARAMETERIZED_CONSTRUCTOR_MIRRORS = {"params", "params_p", "params_n"}
_WAVEFORM_MODULE_DISPATCH = {
    name: inspect.getattr_static(torch.nn.Module, name)
    for name in ("__call__", "_call_impl", "_wrapped_call_impl", "__getattr__")
}
_WAVEFORM_MODULE_DISPATCH["__getattribute__"] = object.__getattribute__


def _validate_waveform_dispatch(module: torch.nn.Module, label: str) -> None:
    """Keep evaluated recipes on the forward/registered-attribute contract."""
    for name, standard in _WAVEFORM_MODULE_DISPATCH.items():
        if inspect.getattr_static(
            type(module), name
        ) is not standard or name in object.__getattribute__(module, "__dict__"):
            raise FunctionalizationError(
                f"Waveform {label!r} requires canonical module call/attribute "
                "dispatch; express custom behavior in forward or fn and use "
                "registered parameters, buffers, and child modules."
            )
    if getattr(module, "_compiled_call_impl", None) is not None:
        raise FunctionalizationError(
            f"Waveform {label!r} cannot retain compiled custom call dispatch."
        )


def _waveform_registered_modules(waveform: Waveform):
    """Visit raw registered children, qualifying dispatch before reading state."""

    def qualified():
        for path, module in _registered_modules(
            waveform, namespace=_raw_module_namespace
        ):
            _validate_waveform_dispatch(module, path or "waveform")
            yield path, module

    return qualified()


def _is_internal_module_state(module: torch.nn.Module, name: str) -> bool:
    if name in _MODULE_INTERNAL_STATE:
        return True
    return isinstance(module, SimpleParameterized) and (
        name in _SIMPLE_PARAMETERIZED_CONSTRUCTOR_MIRRORS
    )


def _find_callable_state(value: Any, path: str, seen: set[int]) -> str | None:
    if callable(value) and not isinstance(value, torch.nn.Module):
        return path
    if value is None or isinstance(value, (bool, int, float, complex, str)):
        return None
    if torch.is_tensor(value) or isinstance(value, (torch.nn.Module, torch.Generator)):
        return None
    identity = id(value)
    if identity in seen:
        return None
    seen.add(identity)
    if isinstance(value, (tuple, list, set, frozenset)):
        for index, item in enumerate(value):
            found = _find_callable_state(item, f"{path}[{index}]", seen)
            if found is not None:
                return found
    elif isinstance(value, dict):
        for key, item in value.items():
            found = _find_callable_state(item, f"{path}[{key!r}]", seen)
            if found is not None:
                return found
    return None


def _strict_state_signature(value: Any, *, label: str) -> Any:
    """Fingerprint auditable Waveform state or reject it explicitly."""

    if value is None or isinstance(value, (bool, int, float, complex, str)):
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return (type(value).__name__, str(value))
    if isinstance(value, slice):
        return ("slice", value.start, value.stop, value.step)
    if isinstance(value, np.generic):
        scalar = np.asarray(value)
        return (
            type(value).__name__,
            str(scalar.dtype),
            hashlib.sha256(scalar.tobytes()).hexdigest(),
        )
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return (
            "ndarray",
            tuple(int(size) for size in value.shape),
            str(value.dtype),
            hashlib.sha256(contiguous.tobytes()).hexdigest(),
        )
    if isinstance(value, torch.Generator):
        state = value.get_state().detach().cpu().contiguous().numpy().tobytes()
        return (
            "torch.Generator",
            str(value.device),
            hashlib.sha256(state).hexdigest(),
        )
    if isinstance(value, (tuple, list)):
        return (
            type(value).__name__,
            tuple(
                _strict_state_signature(item, label=f"{label}[{index}]")
                for index, item in enumerate(value)
            ),
        )
    if isinstance(value, (set, frozenset)):
        items = [_strict_state_signature(item, label=label) for item in value]
        return (type(value).__name__, tuple(sorted(items, key=repr)))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                (
                    _strict_state_signature(key, label=f"{label}.key"),
                    _strict_state_signature(item, label=f"{label}[{key!r}]"),
                )
                for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
            ),
        )
    if callable(value):
        raise FunctionalizationError(
            f"Waveform state {label!r} is callable. Callable instance "
            "state has no stable functional compatibility contract; express "
            "the operation in the Waveform class or as tensor state."
        )
    raise FunctionalizationError(
        f"Waveform state {label!r} has unauditable type "
        f"{type(value).__module__}.{type(value).__qualname__}. Use immutable "
        "literal state, registered tensors, or an explicit child Module."
    )


def _trusted_state_signature(value: Any) -> Any:
    """Retain recipe structure without inspecting hidden numerical state."""
    if value is None or isinstance(value, (bool, int, float, complex, str)):
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return (type(value).__name__, str(value))
    if isinstance(value, slice):
        return ("slice", value.start, value.stop, value.step)
    if isinstance(value, torch.Generator):
        return ("torch.Generator", str(value.device))
    if isinstance(value, np.generic):
        return (type(value).__name__, str(value.dtype), value.item())
    if isinstance(value, np.ndarray):
        return ("ndarray", tuple(value.shape), str(value.dtype))
    if isinstance(value, (tuple, list)):
        return (
            type(value).__name__,
            tuple(_trusted_state_signature(item) for item in value),
        )
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                (_trusted_state_signature(key), _trusted_state_signature(item))
                for key, item in value.items()
            ),
        )
    return ("opaque", type(value), id(value))


def _module_python_state_signature(
    module: torch.nn.Module, label: str, *, check_purity: bool = True
) -> tuple[Any, ...]:
    state = []
    for name, value in sorted(vars(module).items()):
        if _is_internal_module_state(module, name):
            continue
        if name == "_functional_poisson_rng":
            continue
        from dendra.models.stim.waveform.core import _randomized_poisson

        if type(module) is _randomized_poisson and name == "generator":
            state.append((name, ("poisson-generator", str(value.device))))
            continue
        if name == "_cache" and isinstance(module, cacheable):
            continue
        if not check_purity:
            if not isinstance(value, torch.nn.Module):
                state.append((name, _trusted_state_signature(value)))
            continue
        if name == "_cache":
            raise FunctionalizationError(
                f"Waveform module {label!r} uses reserved mutable state '_cache'."
            )
        if torch.is_tensor(value):
            raise FunctionalizationError(
                f"Waveform module state {label + '.' + name!r} is an "
                "unregistered Tensor; register it as a parameter or buffer."
            )
        if isinstance(value, torch.nn.Module):
            continue
        state.append((name, _strict_state_signature(value, label=f"{label}.{name}")))
    return tuple(state)


def _validate_waveform_contract(
    waveform: Waveform, label: str, *, check_purity: bool | None = None
) -> None:
    """Validate hooks, determinism capability, and callable instance state.

    Optional diagnostics require ``FUNCTIONAL_PURE = True`` on each concrete
    Waveform class. Canonical dispatch and registered hooks remain structural
    execution contracts in either mode.
    """

    if check_purity is None:
        check_purity = current_check_purity()
    hook_attributes = (
        "_forward_pre_hooks",
        "_forward_hooks",
        "_backward_pre_hooks",
        "_backward_hooks",
    )
    for module_name, module in _waveform_registered_modules(waveform):
        module_label = label if not module_name else f"{label}.{module_name}"
        for attribute in hook_attributes:
            if getattr(module, attribute, None):
                raise FunctionalizationError(
                    f"Waveform {module_label!r} has registered module hooks; "
                    "functional stimulation currently fails closed on hooks."
                )
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if getattr(parameter, "_backward_hooks", None):
                raise FunctionalizationError(
                    f"Waveform parameter "
                    f"{module_label + '.' + parameter_name!r} has registered "
                    "gradient hooks; functional stimulation currently fails "
                    "closed on hooks."
                )
        if not check_purity:
            continue
        from dendra.models.stim.waveform.core import _randomized_poisson

        shared_poisson = type(module) is _randomized_poisson
        if shared_poisson:
            from ._poisson_stimulation import randomized_nodes

            randomized_nodes(module)
        for name, value in vars(module).items():
            if _is_internal_module_state(module, name):
                continue
            callable_path = _find_callable_state(value, f"{module_label}.{name}", set())
            if callable_path is not None:
                raise FunctionalizationError(
                    f"Waveform state {callable_path!r} is callable. Callable "
                    "instance state is not supported because its behavior "
                    "cannot be represented in the compatibility signature."
                )
        _module_python_state_signature(module, module_label)
        if isinstance(module, Waveform):
            from ._torch_parametrizations import canonical_type

            if (
                canonical_type(module).__dict__.get(_FUNCTIONAL_PURITY_CAPABILITY)
                is not True
                and not shared_poisson
            ):
                raise FunctionalizationError(
                    f"Waveform {module_label!r} has no functional purity "
                    "contract. Declare FUNCTIONAL_PURE = True on the concrete "
                    "class only after ensuring forward is deterministic and "
                    "does not mutate tensor or Python state."
                )


def _module_tensor_snapshot(module: torch.nn.Module) -> dict[str, tuple[Any, ...]]:
    values = {}
    for kind, iterator in (
        ("parameter", module.named_parameters(remove_duplicate=False)),
        ("buffer", module.named_buffers(remove_duplicate=False)),
    ):
        for name, value in iterator:
            values[f"{kind}:{name}"] = (
                id(value),
                value._version,
                _shape(value),
                value.dtype,
                value.device,
                value.detach().clone(memory_format=torch.preserve_format),
            )
    return values


def _tensor_values_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare tensor values exactly while treating paired NaNs as equal."""

    if (
        left.shape != right.shape
        or left.dtype != right.dtype
        or left.device != right.device
    ):
        return False
    if torch.equal(left, right):
        return True
    if torch.is_floating_point(left) or torch.is_complex(left):
        return torch.allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True)
    return False


def _assert_module_tensors_unchanged(
    module: torch.nn.Module,
    expected: Mapping[str, tuple[Any, ...]],
    *,
    label: str,
) -> None:
    actual = _module_tensor_snapshot(module)
    if set(actual) != set(expected):
        raise FunctionalizationError(
            f"Waveform {label!r} changed its registered tensor layout "
            "during the functional purity audit."
        )
    for name, snapshot in expected.items():
        identity, version, shape, dtype, device, value = snapshot
        current = actual[name]
        if (
            current[0] != identity
            or current[1] != version
            or current[2] != shape
            or current[3] != dtype
            or current[4] != device
            or not _tensor_values_equal(current[5], value)
        ):
            raise FunctionalizationError(
                f"Waveform {label!r} mutated registered tensor "
                f"{name.removeprefix('parameter:').removeprefix('buffer:')!r} "
                "during evaluation. Functional Waveforms must be stateless."
            )


def _module_tree_state_signature(
    waveform: Waveform, label: str
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    return tuple(
        (
            path,
            _module_python_state_signature(
                module,
                label if not path else f"{label}.{path}",
            ),
        )
        for path, module in waveform.named_modules()
    )


def _audit_waveform_purity(
    waveform: Waveform,
    *,
    label: str,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Exercise an opted-in recipe without leaking RNG side effects."""

    tensor_state = _module_tensor_snapshot(waveform)
    python_state = _module_tree_state_signature(waveform, label)
    torch_rng = torch.random.get_rng_state()
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    cuda_rng = None
    if device.type == "cuda":
        cuda_rng = torch.cuda.get_rng_state(device)
    mps_rng = None
    if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        mps_rng = torch.mps.get_rng_state()

    times = torch.tensor((0.0, 0.125, 0.375), device=device, dtype=dtype)
    try:
        with torch.no_grad():
            first = waveform(times)
        if not torch.is_tensor(first):
            raise FunctionalizationError(f"Waveform {label!r} must return a Tensor.")
        _assert_module_tensors_unchanged(waveform, tensor_state, label=label)
        if _module_tree_state_signature(waveform, label) != python_state:
            raise FunctionalizationError(
                f"Waveform {label!r} mutated Python instance state "
                "during evaluation. Functional Waveforms must be stateless."
            )
        if not torch.equal(torch.random.get_rng_state(), torch_rng):
            raise FunctionalizationError(
                f"Waveform {label!r} consumed the global PyTorch RNG. "
                "Functional Waveforms must be deterministic."
            )
        if random.getstate() != python_rng:
            raise FunctionalizationError(
                f"Waveform {label!r} consumed the global Python RNG. "
                "Functional Waveforms must be deterministic."
            )
        if not _numpy_rng_state_equal(np.random.get_state(), numpy_rng):
            raise FunctionalizationError(
                f"Waveform {label!r} consumed the global NumPy RNG. "
                "Functional Waveforms must be deterministic."
            )
        if cuda_rng is not None and not torch.equal(
            torch.cuda.get_rng_state(device), cuda_rng
        ):
            raise FunctionalizationError(
                f"Waveform {label!r} consumed the CUDA RNG. Functional "
                "Waveforms must be deterministic."
            )
        if mps_rng is not None and not torch.equal(torch.mps.get_rng_state(), mps_rng):
            raise FunctionalizationError(
                f"Waveform {label!r} consumed the MPS RNG. Functional "
                "Waveforms must be deterministic."
            )

        first_value = first.detach().clone(memory_format=torch.preserve_format)
        with torch.no_grad():
            second = waveform(times)
        _assert_module_tensors_unchanged(waveform, tensor_state, label=label)
        if _module_tree_state_signature(waveform, label) != python_state:
            raise FunctionalizationError(
                f"Waveform {label!r} mutated Python instance state "
                "during repeated evaluation."
            )
        if not torch.is_tensor(second) or not _tensor_values_equal(second, first_value):
            raise FunctionalizationError(
                f"Waveform {label!r} produced non-deterministic outputs "
                "for identical inputs."
            )
    except FunctionalizationError:
        raise
    except Exception as exc:
        raise FunctionalizationError(
            f"Waveform {label!r} failed its functional purity audit."
        ) from exc
    finally:
        torch.random.set_rng_state(torch_rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        if mps_rng is not None:
            torch.mps.set_rng_state(mps_rng)


class _WaveformBank(torch.nn.Module):
    """Small module whose only mutable tensor state is supplied explicitly."""

    def __init__(self, waveforms: Sequence[Waveform]):
        super().__init__()
        self.waveforms = torch.nn.ModuleList(waveforms)

    def forward(self, times: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(waveform(times) for waveform in self.waveforms)


@dataclass(frozen=True)
class _BankLeaf:
    public_name: str
    internal_name: str
    waveform_index: int
    source_name: str
    schema: TensorSchema


class _FunctionalWaveformBank:
    """Deep-copied Waveforms plus explicit parameter/buffer namespaces."""

    def __init__(
        self,
        waveforms: Sequence[Waveform],
        *,
        namespace: str,
        device: torch.device,
        dtype: torch.dtype,
        check_purity: bool | None = None,
    ):
        self._namespace = namespace.rstrip(".")
        self._device = device
        self._dtype = dtype
        self._sources = tuple(waveforms)
        self._check_purity = (
            current_check_purity() if check_purity is None else check_purity
        )

        clones = []
        for index, waveform in enumerate(self._sources):
            if not isinstance(waveform, Waveform):
                raise TypeError(
                    f"waveform {index} has type {type(waveform).__name__}; "
                    "expected Waveform"
                )
            _validate_waveform_contract(
                waveform, f"waveforms.{index}", check_purity=self._check_purity
            )
            from ._torch_parametrizations import failures, install_output_schemas

            parametrization_failures = failures(waveform, diagnostic=self._check_purity)
            if parametrization_failures:
                raise FunctionalizationError("; ".join(parametrization_failures))
            try:
                clone = copy.deepcopy(waveform)
            except Exception as exc:  # pragma: no cover - custom Waveform detail
                raise FunctionalizationError(
                    f"Waveform {index} could not be isolated by deepcopy."
                ) from exc
            # This mutates only the private execution copy.  In particular, a
            # CPU-authored Waveform is never moved out from under its caller.
            clone.to(device=device, dtype=dtype)
            from ._poisson_stimulation import configure_execution, randomized_nodes

            poisson_nodes = randomized_nodes(waveform)
            for path, source in poisson_nodes:
                from ._poisson_stimulation import normalize_schedule

                target = clone.get_submodule(path) if path else clone
                target._spike_times = normalize_schedule(source, target._spike_times)
            if poisson_nodes:
                configure_execution(clone)
            else:
                from ._poisson_execution import isolate_waveform_aliases

                # Every explicit alias path must have its own registration
                # owner so functional_call restores caller operands exactly.
                isolate_waveform_aliases(clone)
            # Preserve every authored Waveform/module train/eval flag. Only
            # cache-bearing parameter modules use training mode internally so
            # explicit functional leaves remain authoritative and evaluation
            # cannot mutate a hidden `_cache` tensor.
            for module in clone.modules():
                if isinstance(module, cacheable) and not isinstance(module, Waveform):
                    module.clear_cache()
                    module.training = True
            install_output_schemas(clone)
            if self._check_purity:
                _audit_waveform_purity(
                    clone,
                    label=f"waveforms.{index}",
                    device=device,
                    dtype=dtype,
                )
            clones.append(clone)

        self._bank = _WaveformBank(clones)
        template_parameters = dict(self._bank.named_parameters(remove_duplicate=False))
        template_constants = dict(self._bank.named_buffers(remove_duplicate=False))
        parameter_leaves: list[_BankLeaf] = []
        constant_leaves: list[_BankLeaf] = []
        canonical_parameters: dict[int, str] = {}
        canonical_constants: dict[int, str] = {}

        for index, source in enumerate(self._sources):
            for source_name, value in source.named_parameters(remove_duplicate=False):
                internal_name = f"waveforms.{index}.{source_name}"
                if internal_name not in template_parameters:
                    raise FunctionalizationError(
                        f"Waveform parameter layout for contact {index} is not "
                        "stable under deepcopy."
                    )
                public_name = canonical_parameters.setdefault(
                    id(value), self._public_name(index, source_name)
                )
                parameter_leaves.append(
                    _BankLeaf(
                        public_name,
                        internal_name,
                        index,
                        source_name,
                        _schema(template_parameters[internal_name]),
                    )
                )
            for source_name, value in source.named_buffers(remove_duplicate=False):
                internal_name = f"waveforms.{index}.{source_name}"
                if internal_name not in template_constants:
                    raise FunctionalizationError(
                        f"Waveform buffer layout for contact {index} is not "
                        "stable under deepcopy."
                    )
                parent, _separator, field = source_name.rpartition(".")
                owner = source.get_submodule(parent) if parent else source
                from dendra.models.stim.waveform.core import _randomized_poisson

                public_name = (
                    self._public_name(index, source_name)
                    if field == "_spike_times" and type(owner) is _randomized_poisson
                    else canonical_constants.setdefault(
                        id(value), self._public_name(index, source_name)
                    )
                )
                constant_leaves.append(
                    _BankLeaf(
                        public_name,
                        internal_name,
                        index,
                        source_name,
                        _schema(template_constants[internal_name]),
                    )
                )

        self._parameter_leaves = tuple(parameter_leaves)
        self._constant_leaves = tuple(constant_leaves)

    @property
    def source_identities(self) -> frozenset[int]:
        """Identities of authored Waveforms retained by this plan."""

        return frozenset(id(waveform) for waveform in self._sources)

    def _public_name(self, index: int, name: str) -> str:
        relative = f"waveforms.{index}.{name}"
        return f"{self._namespace}.{relative}" if self._namespace else relative

    @property
    def parameter_schema(self) -> dict[str, TensorSchema]:
        return {leaf.public_name: leaf.schema for leaf in self._parameter_leaves}

    @property
    def constant_schema(self) -> dict[str, TensorSchema]:
        return {leaf.public_name: leaf.schema for leaf in self._constant_leaves}

    def _tensor_sources(self) -> dict[str, torch.Tensor]:
        """Map explicit names to authored Tensors before placement or cloning.

        This cold correspondence records identity, rather than shared storage.
        Aggregate snapshots use it to preserve aliases across stimulus plans.
        """
        sources = {}
        for leaves, accessor in (
            (self._parameter_leaves, "named_parameters"),
            (self._constant_leaves, "named_buffers"),
        ):
            for leaf in leaves:
                waveform = self._sources[leaf.waveform_index]
                source = dict(getattr(waveform, accessor)(remove_duplicate=False))[
                    leaf.source_name
                ]
                if (
                    leaf.public_name in sources
                    and sources[leaf.public_name] is not source
                ):
                    raise FunctionalizationError(
                        "Waveform source aliases changed after lowering; "
                        "lower the Population or Network again."
                    )
                sources[leaf.public_name] = source
        return sources

    def extract(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        parameters: dict[str, torch.Tensor] = {}
        constants: dict[str, torch.Tensor] = {}
        for leaf in self._parameter_leaves:
            source = dict(
                self._sources[leaf.waveform_index].named_parameters(
                    remove_duplicate=False
                )
            )[leaf.source_name]
            # Device/dtype conversion is differentiable and does not mutate the
            # authored source Parameter.  Matching leaves remain the originals.
            parameters[leaf.public_name] = source.to(
                device=leaf.schema.device,
                dtype=leaf.schema.dtype,
            )
        for leaf in self._constant_leaves:
            source = dict(
                self._sources[leaf.waveform_index].named_buffers(remove_duplicate=False)
            )[leaf.source_name]
            parent, _separator, field = leaf.source_name.rpartition(".")
            from dendra.models.stim.waveform.core import _randomized_poisson

            owner = self._sources[leaf.waveform_index]
            owner = owner.get_submodule(parent) if parent else owner
            if field == "_spike_times" and type(owner) is _randomized_poisson:
                from ._poisson_stimulation import normalize_schedule

                source = normalize_schedule(owner, source)
            constants[leaf.public_name] = (
                source.detach()
                .to(
                    device=leaf.schema.device,
                    dtype=leaf.schema.dtype,
                )
                .clone(memory_format=torch.preserve_format)
            )
        return parameters, constants

    def evaluate(
        self,
        parameters: Mapping[str, torch.Tensor],
        constants: Mapping[str, torch.Tensor],
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        mapping: dict[str, torch.Tensor] = {}
        for leaf in self._parameter_leaves:
            mapping[leaf.internal_name] = parameters[leaf.public_name]
        for leaf in self._constant_leaves:
            mapping[leaf.internal_name] = constants[leaf.public_name]
            if leaf.internal_name.endswith("._spike_times"):
                from dendra.models.stim.waveform._poisson import (
                    POISSON_INVOCATION_POSITION,
                    POISSON_INVOCATION_STACK,
                )

                path = leaf.internal_name.rsplit(".", 1)[0]
                owner = self._bank.get_submodule(path)
                if getattr(owner, "_functional_poisson_recipe", False):
                    stack = constants.get(
                        leaf.public_name + ".__invocations",
                        getattr(owner, POISSON_INVOCATION_STACK),
                    )
                    mapping[f"{path}.{POISSON_INVOCATION_STACK}"] = stack
                    mapping[f"{path}.{POISSON_INVOCATION_POSITION}"] = torch.zeros(
                        (1,), dtype=torch.int64, device=stack.device
                    )
        return functional_call(
            self._bank,
            mapping,
            (times,),
            tie_weights=False,
            strict=True,
        )


class _FunctionalStimulus:
    """Shared schema, extraction, and checked replacement behavior."""

    _parameter_schema: dict[str, TensorSchema]
    _constant_schema: dict[str, TensorSchema]

    def _validate_deterministic_assembly(self):
        from ._poisson_stimulation import randomized_nodes

        if any(randomized_nodes(waveform) for waveform in self._waveforms._sources):
            raise FunctionalizationError(
                "Randomized Poisson stimulation requires explicit draws and RNG "
                "carry; bind it through make_functional(..., extra=...) or "
                "dn.compile and use the Population stimulation input."
            )

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._batch_shape

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self._parameter_schema)

    @property
    def constant_names(self) -> tuple[str, ...]:
        return tuple(self._constant_schema)

    @property
    def parameter_schema(self) -> dict[str, TensorSchema]:
        return dict(self._parameter_schema)

    @property
    def constant_schema(self) -> dict[str, TensorSchema]:
        return dict(self._constant_schema)

    @property
    def plan_token(self) -> object:
        return self._token

    @property
    def _waveform_source_identities(self) -> frozenset[int]:
        """Authored Waveform identities used to detect cross-plan aliases."""

        return self._waveforms.source_identities

    def _tensor_sources(self) -> dict[str, torch.Tensor]:
        """Return original Tensor correspondence for cold aggregate extraction."""
        return self._waveforms._tensor_sources()

    def _validate_mapping(
        self,
        values: Mapping[str, torch.Tensor],
        schema: Mapping[str, TensorSchema],
        *,
        label: str,
    ) -> None:
        if not isinstance(values, Mapping):
            raise TypeError(f"{label} must be a mapping of named Tensors")
        expected = set(schema)
        actual = set(values)
        if actual != expected:
            raise KeyError(
                f"stimulation {label} mismatch; "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        for name, contract in schema.items():
            value = values[name]
            if not torch.is_tensor(value):
                raise TypeError(f"stimulation leaf {name!r} must be a Tensor")
            if _shape(value) != contract.shape:
                raise ValueError(
                    f"stimulation leaf {name!r} has shape {_shape(value)}; "
                    f"expected {contract.shape}"
                )
            if value.device != contract.device or value.dtype != contract.dtype:
                raise ValueError(
                    f"stimulation leaf {name!r} must use "
                    f"{contract.device}/{contract.dtype}"
                )

    def validate(self, tensors: StimulusTensors) -> None:
        if not isinstance(tensors, StimulusTensors):
            raise TypeError("tensors must be StimulusTensors")
        if tensors.plan_token is not self._token:
            raise FunctionalizationError(
                "StimulusTensors belong to a different functional stimulation plan."
            )
        self._validate_mapping(
            tensors.parameters, self._parameter_schema, label="parameters"
        )
        self._validate_mapping(
            tensors.constants, self._constant_schema, label="constants"
        )

    def replace(
        self,
        tensors: StimulusTensors | None = None,
        *,
        parameters: Mapping[str, torch.Tensor] | None = None,
        constants: Mapping[str, torch.Tensor] | None = None,
    ) -> StimulusTensors:
        """Return a checked plan-bound bundle with partial leaf replacements."""

        base = self.extract() if tensors is None else tensors
        self.validate(base)
        new_parameters = dict(base.parameters)
        new_constants = dict(base.constants)
        if parameters is not None:
            unknown = set(parameters) - set(self._parameter_schema)
            if unknown:
                raise KeyError(f"unexpected stimulation parameters: {sorted(unknown)}")
            new_parameters.update(parameters)
        if constants is not None:
            unknown = set(constants) - set(self._constant_schema)
            if unknown:
                raise KeyError(f"unexpected stimulation constants: {sorted(unknown)}")
            new_constants.update(constants)
        result = StimulusTensors(new_parameters, new_constants, self._token)
        self.validate(result)
        return result

    def assemble_tensors(
        self, tensors: StimulusTensors, times: torch.Tensor, **assemble_kwargs
    ) -> torch.Tensor | None:
        """Token-check a bundle and assemble its complete time-first drive."""

        self.validate(tensors)
        return self.assemble(
            tensors.parameters,
            tensors.constants,
            times,
            **assemble_kwargs,
        )

    def _validate_inputs(
        self,
        parameters: Mapping[str, torch.Tensor],
        constants: Mapping[str, torch.Tensor],
    ) -> None:
        self._validate_mapping(parameters, self._parameter_schema, label="parameters")
        self._validate_mapping(constants, self._constant_schema, label="constants")

    def _times(self, times: Any) -> torch.Tensor:
        if torch.is_tensor(times):
            result = times.to(device=self.device, dtype=self.dtype)
        else:
            result = torch.as_tensor(times, device=self.device, dtype=self.dtype)
        if result.ndim == 0:
            result = result.unsqueeze(0)
        if result.ndim != 1:
            raise ValueError(
                "functional stimulation times must be a scalar or 1-D Tensor; "
                f"got shape {_shape(result)}"
            )
        return result


def _normalize_time_last_output(
    value: torch.Tensor,
    *,
    n_steps: int,
    name: str,
) -> torch.Tensor:
    if value.ndim == 0:
        return value.expand(n_steps)
    if int(value.shape[-1]) != n_steps:
        raise ValueError(
            f"{name} must put time on its last axis; got shape {_shape(value)} "
            f"for {n_steps} timepoints"
        )
    return value


def _expand_time_first_to_selection(
    values: torch.Tensor,
    selection_shape: tuple[int, ...],
    batch_shape: tuple[int, ...],
    *,
    name: str,
) -> torch.Tensor:
    """Vectorized counterpart of Intra's per-step broadcast convention."""

    n_steps = int(values.shape[0])
    payload_shape = tuple(int(size) for size in values.shape[1:])
    target_shape = (n_steps, *selection_shape)
    if any(size == 0 for size in selection_shape):
        return values.new_empty(target_shape)

    trailing_error: RuntimeError | None = None
    if len(payload_shape) <= len(selection_shape):
        candidate_shape = (
            n_steps,
            *(1,) * (len(selection_shape) - len(payload_shape)),
            *payload_shape,
        )
        try:
            return torch.broadcast_to(values.reshape(candidate_shape), target_shape)
        except RuntimeError as exc:
            trailing_error = exc

    batch_rank = len(batch_shape)
    if batch_rank and 0 < len(payload_shape) <= batch_rank:
        batch_candidate_shape = (
            n_steps,
            *(1,) * (batch_rank - len(payload_shape)),
            *payload_shape,
        )
        try:
            batch_values = torch.broadcast_to(
                values.reshape(batch_candidate_shape),
                (n_steps, *batch_shape),
            )
            candidate = batch_values.reshape(
                n_steps,
                *batch_shape,
                *(1,) * (len(selection_shape) - batch_rank),
            )
            return torch.broadcast_to(candidate, target_shape)
        except RuntimeError:
            pass

    message = (
        f"{name} sample shape {payload_shape} cannot broadcast to selected "
        f"model shape {selection_shape}. Ordinary trailing PyTorch "
        "broadcasting is tried first; a batch-only fallback accepts values "
        f"broadcastable to explicit batch shape {batch_shape}. Use singleton "
        "axes to disambiguate batch and spatial intent."
    )
    if trailing_error is None:
        raise ValueError(message)
    raise ValueError(message) from trailing_error


def _index_signature(value: Any) -> Any:
    if torch.is_tensor(value):
        return (
            "tensor",
            id(value),
            value._version,
            _shape(value),
            value.dtype,
            value.device,
        )
    if isinstance(value, slice):
        return ("slice", value.start, value.stop, value.step)
    if isinstance(value, tuple):
        return ("tuple", tuple(_index_signature(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_index_signature(item) for item in value))
    return (type(value), value)


def _tensor_content_signature(value: torch.Tensor) -> tuple[Any, ...]:
    detached = value.detach().to(device="cpu").contiguous()
    try:
        payload = detached.numpy().tobytes()
    except TypeError:
        # NumPy does not currently expose every PyTorch dtype (for example,
        # bfloat16). Viewing the bytes avoids a lossy dtype conversion.
        payload = _logical_tensor_bytes(detached).numpy().tobytes()
    return (_shape(value), value.dtype, hashlib.sha256(payload).hexdigest())


def _waveform_layout_signature(
    waveform: Waveform, *, check_purity: bool | None = None
) -> tuple[Any, ...]:
    if check_purity is None:
        check_purity = current_check_purity()
    modules = []
    for path, module in waveform.named_modules():
        from dendra.models.stim.waveform.core import _randomized_poisson

        def buffer_shape(name, value):
            if (
                name == "_spike_times"
                and type(module) is _randomized_poisson
                and module.n is not None
            ):
                return (max(module.n, 1),) + (1,) * module._spike_time_trailing_dims
            return _shape(value)

        module_label = "waveform" if not path else f"waveform.{path}"
        static = _module_python_state_signature(
            module,
            module_label,
            check_purity=check_purity,
        )
        modules.append(
            (
                path,
                type(module).__module__,
                type(module).__qualname__,
                tuple(
                    (name, _shape(value), value.dtype)
                    for name, value in module.named_parameters(recurse=False)
                ),
                tuple(
                    (name, buffer_shape(name, value), value.dtype)
                    for name, value in module.named_buffers(recurse=False)
                ),
                static,
            )
        )
    return tuple(modules)


def _solver_injection_specs(population: Population):
    accepted = list(getattr(population, "mechanism_injection_accepted", ()))
    injections = list(getattr(population, "injections", ()))
    if len(accepted) < len(injections):
        accepted.extend([False] * (len(injections) - len(accepted)))
    return [spec for spec, claimed in zip(injections, accepted) if not claimed]


def _injection_signature(
    population: Population, *, check_purity: bool = True
) -> tuple[Any, ...]:
    accepted = list(getattr(population, "mechanism_injection_accepted", ()))
    injections = list(getattr(population, "injections", ()))
    if len(accepted) < len(injections):
        accepted.extend([False] * (len(injections) - len(accepted)))
    return tuple(
        (
            id(waveform),
            _waveform_layout_signature(waveform, check_purity=check_purity),
            tuple(int(size) for size in registered_shape),
            _index_signature(index),
            bool(claimed),
        )
        for (waveform, registered_shape, index), claimed in zip(injections, accepted)
    )


class FunctionalIntra(_FunctionalStimulus):
    """Pure plan for solver-owned ``model[slice].inject(waveform)`` specs.

    Mechanism-owned injections are deliberately excluded: their effect belongs
    in functional mechanism lowering.  An instance is always constructible;
    :attr:`enabled` is false and :meth:`assemble` returns ``None`` when no
    solver-owned injection is registered.
    """

    def __init__(self, population: Population, *, _check_purity: bool | None = None):
        shape, batch_shape, device, dtype = _population_contract(population)
        self._shape = shape
        self._batch_shape = batch_shape
        self._device = device
        self._dtype = dtype
        self._population = population
        self._check_purity = (
            current_check_purity() if _check_purity is None else _check_purity
        )
        self._source_signature = _injection_signature(
            population, check_purity=self._check_purity
        )
        self._token = object()

        specs = _solver_injection_specs(population)
        waveforms = []
        indices = []
        selected_batch_shapes = []
        numel = 1
        for size in shape:
            numel *= size
        linear_index = torch.arange(numel, device=device, dtype=torch.long).reshape(
            shape
        )
        for waveform, _registered_shape, index in specs:
            if not isinstance(waveform, Waveform):
                raise TypeError(
                    "model[slice].inject(...) must receive a Waveform for "
                    "functional lowering"
                )
            canonical = _canonicalize_index_for_index_put(
                index,
                shape,
                device=device,
                linear_index=linear_index,
            )
            canonical = tuple(item.detach().clone() for item in canonical)
            waveforms.append(waveform)
            indices.append(canonical)
            selected_batch_shapes.append(
                _retained_leading_batch_shape(canonical, batch_shape)
            )
        self._indices = tuple(indices)
        self._selected_batch_shapes = tuple(selected_batch_shapes)
        self._waveforms = _FunctionalWaveformBank(
            waveforms,
            namespace="stimulation.intra",
            device=device,
            dtype=dtype,
            check_purity=self._check_purity,
        )
        self._parameter_schema = self._waveforms.parameter_schema
        self._constant_schema = self._waveforms.constant_schema
        self._layout_signature = (
            "FunctionalIntra.v1",
            self.shape,
            self.batch_shape,
            self.device,
            self.dtype,
            tuple(
                tuple(_tensor_content_signature(item) for item in canonical)
                for canonical in self._indices
            ),
            self._selected_batch_shapes,
            tuple(
                _waveform_layout_signature(waveform, check_purity=self._check_purity)
                for waveform in waveforms
            ),
            tuple(self._parameter_schema.items()),
            tuple(self._constant_schema.items()),
        )

    @property
    def enabled(self) -> bool:
        return bool(self._indices)

    @property
    def layout_signature(self) -> tuple[Any, ...]:
        """Read-only injection topology and Waveform recipe contract."""

        return self._layout_signature

    def is_compatible(self, other: FunctionalIntra) -> bool:
        """Whether ``other`` can supply leaves to this execution plan."""

        return isinstance(other, FunctionalIntra) and (
            other.layout_signature == self.layout_signature
        )

    def _validate_source(self) -> None:
        if (
            _injection_signature(self._population, check_purity=self._check_purity)
            != self._source_signature
        ):
            raise FunctionalizationError(
                "Population injection structure changed after FunctionalIntra "
                "was created; lower the Population again."
            )

    def _tensor_sources(
        self, population: Population | None = None
    ) -> dict[str, torch.Tensor]:
        if population is not None and population is not self._population:
            candidate = FunctionalIntra(population, _check_purity=self._check_purity)
            if not self.is_compatible(candidate):
                raise FunctionalizationError(
                    "Population injection topology or Waveform structure does "
                    "not match this FunctionalIntra plan."
                )
            return candidate._tensor_sources()
        self._validate_source()
        return super()._tensor_sources()

    def extract(self, population: Population | None = None) -> StimulusTensors:
        if population is not None and population is not self._population:
            candidate = FunctionalIntra(population, _check_purity=self._check_purity)
            if not self.is_compatible(candidate):
                raise FunctionalizationError(
                    "Population injection topology or Waveform structure does "
                    "not match this FunctionalIntra plan."
                )
            candidate_tensors = candidate.extract()
            result = StimulusTensors(
                dict(candidate_tensors.parameters),
                dict(candidate_tensors.constants),
                self._token,
            )
            self.validate(result)
            return result
        self._validate_source()
        parameters, constants = self._waveforms.extract()
        result = StimulusTensors(parameters, constants, self._token)
        self.validate(result)
        return result

    def assemble(
        self,
        parameters: Mapping[str, torch.Tensor],
        constants: Mapping[str, torch.Tensor],
        times: torch.Tensor,
    ) -> torch.Tensor | None:
        """Assemble a contiguous ``[time, *population.shape]`` current."""

        self._validate_deterministic_assembly()
        self._validate_source()
        self._validate_inputs(parameters, constants)
        if not self.enabled:
            return None
        times = self._times(times)
        return self.assemble_values(parameters, constants, times)

    def assemble_values(
        self,
        parameters: Mapping[str, torch.Tensor],
        constants: Mapping[str, torch.Tensor],
        times: torch.Tensor,
    ) -> torch.Tensor | None:
        """Tensor-only lowering used inside compiled functional transitions.

        The caller must provide already validated plan-local mappings and a
        one-dimensional time tensor on this plan's device/dtype.
        """

        if not self.enabled:
            return None
        n_steps = int(times.shape[0])
        evaluated = self._waveforms.evaluate(parameters, constants, times)
        result = torch.zeros(
            (n_steps, *self.shape), device=self.device, dtype=self.dtype
        )

        for waveform_index, (value, indices, selected_batch_shape) in enumerate(
            zip(
                evaluated,
                self._indices,
                self._selected_batch_shapes,
                strict=True,
            )
        ):
            value = value.to(device=self.device, dtype=self.dtype)
            value = _normalize_time_last_output(
                value,
                n_steps=n_steps,
                name=f"intracellular waveform {waveform_index}",
            )
            samples = value.movedim(-1, 0)
            selection_shape = tuple(
                int(size)
                for size in torch.broadcast_shapes(
                    *(tuple(item.shape) for item in indices)
                )
            )
            samples = _expand_time_first_to_selection(
                samples,
                selection_shape,
                selected_batch_shape,
                name=f"intracellular waveform {waveform_index}",
            )

            time_shape = (n_steps, *(1,) * len(selection_shape))
            time_index = (
                torch.arange(n_steps, device=self.device, dtype=torch.long)
                .reshape(time_shape)
                .expand(n_steps, *selection_shape)
            )
            assembled_indices = [time_index]
            assembled_indices.extend(
                index.unsqueeze(0).expand(n_steps, *selection_shape)
                for index in indices
            )
            # Out-of-place index_put is essential here: its transform rule is
            # vmap-safe, and accumulate=True preserves duplicate/overlap sums.
            result = result.index_put(
                tuple(assembled_indices), samples, accumulate=True
            )
        return result.contiguous()


def _parse_extra(extra: Any) -> list[tuple[Any, Any]]:
    if extra is None:
        return []
    if isinstance(extra, tuple):
        tuple_of_pairs = bool(extra) and all(
            isinstance(item, tuple) and len(item) == 2 for item in extra
        )
        pairs = list(extra) if tuple_of_pairs else [extra]
    else:
        try:
            pairs = list(extra)
        except TypeError as exc:
            raise TypeError(
                "extra must be a (field, temporal) tuple or a sequence of pairs"
            ) from exc
    if not pairs:
        raise ValueError("If 'extra' is provided, it must not be empty.")
    result = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"extra contact {index} must be a (field, temporal) pair")
        result.append((pair[0], pair[1]))
    return result


def _normalize_spatial(
    value: torch.Tensor,
    *,
    shape: tuple[int, ...],
    batch_shape: tuple[int, ...],
    name: str,
) -> torch.Tensor:
    if value.ndim == 0:
        raise ValueError(f"{name} needs at least one spatial or batch dimension")
    try:
        return torch.broadcast_to(value, shape)
    except RuntimeError as exc:
        trailing_error = exc
    if batch_shape and 0 < value.ndim <= len(batch_shape):
        try:
            batch_value = torch.broadcast_to(value, batch_shape)
            candidate = batch_value.reshape(*batch_shape, 1, 1)
            return torch.broadcast_to(candidate, shape)
        except RuntimeError:
            pass
    raise ValueError(
        f"{name} shape {_shape(value)} is not broadcastable to model shape "
        f"{shape}. Ordinary trailing PyTorch broadcasting is tried first; a "
        "batch-only fallback accepts values broadcastable to explicit batch "
        f"shape {batch_shape}."
    ) from trailing_error


def _normalize_temporal(
    value: torch.Tensor,
    *,
    shape: tuple[int, ...],
    batch_shape: tuple[int, ...],
    n_steps: int,
    name: str,
) -> torch.Tensor:
    value = _normalize_time_last_output(value, n_steps=n_steps, name=name)
    target_shape = (*shape[:-1], n_steps)
    try:
        return torch.broadcast_to(value, target_shape)
    except RuntimeError as exc:
        trailing_error = exc
    leading = _shape(value)[:-1]
    if batch_shape and 0 < len(leading) <= len(batch_shape):
        try:
            batch_value = torch.broadcast_to(value, (*batch_shape, n_steps))
            candidate = batch_value.reshape(*batch_shape, 1, n_steps)
            return torch.broadcast_to(candidate, target_shape)
        except RuntimeError:
            pass
    raise ValueError(
        f"{name} leading shape {leading} is not broadcastable to model "
        f"batch/population shape {shape[:-1]}. Ordinary trailing PyTorch "
        "broadcasting is tried first; a batch-only fallback accepts values "
        f"broadcastable to explicit batch shape {batch_shape}."
    ) from trailing_error


class FunctionalExtra(_FunctionalStimulus):
    """Pure lowering of Population's existing raw ``ExtraSpec`` syntax.

    Construct with ``FunctionalExtra(population, extra)`` or through
    ``FunctionalPopulation.make_extra(extra)``. Spatial fields and tensor
    temporal specifications are exposed as ``contacts.*`` constants; Waveform
    leaves use ``waveforms.*``. ``extra=None`` creates a disabled always-present
    plan.
    """

    def __init__(self, population: Population, extra: Any, *, namespace: str = ""):
        if extra is not None and (
            torch.compiler.is_compiling() or torch._C._are_functorch_transforms_active()
        ):
            raise FunctionalizationError(
                "Raw extra specifications cannot be lowered inside an active "
                "torch.func transform or torch.compile. Bind the same raw spec "
                "outside the transform with "
                "make_functional(population, dt=..., extra=...), then transform "
                "the resulting explicit tensor leaves."
            )
        shape, batch_shape, device, dtype = _population_contract(population)
        self._shape = shape
        self._batch_shape = batch_shape
        self._device = device
        self._dtype = dtype
        self._token = object()
        self._namespace = namespace.rstrip(".")
        self._check_purity = current_check_purity()
        pairs = _parse_extra(extra)
        self._pairs = tuple(pairs)
        self._enabled = extra is not None

        temporal_is_waveform = [
            isinstance(temporal, Waveform) for _field, temporal in pairs
        ]
        if any(temporal_is_waveform) and not all(temporal_is_waveform):
            raise ValueError(
                "All contacts in 'extra' must use the same temporal type; "
                "mixing Waveform and tensor time specifications is not supported."
            )
        self._uses_waveforms = bool(pairs) and all(temporal_is_waveform)
        waveforms = (
            [temporal for _field, temporal in pairs] if self._uses_waveforms else []
        )
        self._waveforms = _FunctionalWaveformBank(
            waveforms,
            namespace=self._namespace,
            device=device,
            dtype=dtype,
            check_purity=self._check_purity,
        )

        parameter_schema = self._waveforms.parameter_schema
        constant_schema = self._waveforms.constant_schema
        contact_sources: dict[str, Any] = {}
        for index, (field, temporal) in enumerate(pairs):
            field_name = self._contact_name(index, "field")
            field_tensor = _tensor_on(field, device=device, dtype=dtype)
            # Validate spatial compatibility at lowering time without retaining
            # the potentially expanded view in the explicit tensor mapping.
            _normalize_spatial(
                field_tensor,
                shape=shape,
                batch_shape=batch_shape,
                name=f"extracellular field {index}",
            )
            constant_schema[field_name] = _schema(field_tensor)
            contact_sources[field_name] = field
            if not self._uses_waveforms:
                time_name = self._contact_name(index, "time")
                time_tensor = _tensor_on(temporal, device=device, dtype=dtype)
                if time_tensor.ndim == 0:
                    raise ValueError(
                        "extra time tensor needs a trailing time axis; "
                        f"contact {index} is scalar"
                    )
                constant_schema[time_name] = _schema(time_tensor)
                contact_sources[time_name] = temporal
        self._contact_sources = contact_sources
        self._parameter_schema = parameter_schema
        self._constant_schema = constant_schema

    def _contact_name(self, index: int, leaf: str) -> str:
        relative = f"contacts.{index}.{leaf}"
        return f"{self._namespace}.{relative}" if self._namespace else relative

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def multicontact(self) -> bool:
        return len(self._pairs) > 1

    @property
    def uses_waveforms(self) -> bool:
        return self._uses_waveforms

    def _tensor_sources(self) -> dict[str, torch.Tensor]:
        sources = super()._tensor_sources()
        # Python-valued fields have no authored Tensor identity. Omitting them
        # leaves each materialized contact independent in aggregate snapshots.
        sources.update(
            (name, source)
            for name, source in self._contact_sources.items()
            if torch.is_tensor(source)
        )
        return sources

    def extract(self) -> StimulusTensors:
        parameters, constants = self._waveforms.extract()
        for name, source in self._contact_sources.items():
            contract = self._constant_schema[name]
            value = _tensor_on(
                source,
                device=contract.device,
                dtype=contract.dtype,
            )
            # Clone retains an autograd edge to differentiable field/time
            # tensors while preventing aliasing with mutable authored inputs.
            constants[name] = value.clone(memory_format=torch.preserve_format)
        result = StimulusTensors(parameters, constants, self._token)
        self.validate(result)
        return result

    def assemble(
        self,
        parameters: Mapping[str, torch.Tensor],
        constants: Mapping[str, torch.Tensor],
        times: torch.Tensor,
        *,
        temporal_slice: slice | None = None,
        total_steps: int | None = None,
    ) -> torch.Tensor | None:
        """Assemble contiguous extracellular voltage ``[time, *shape]``."""
        self._validate_deterministic_assembly()

        self._validate_inputs(parameters, constants)
        if not self.enabled:
            return None
        times = self._times(times)
        temporal_bounds = self._temporal_bounds(
            times,
            temporal_slice=temporal_slice,
            total_steps=total_steps,
        )
        if temporal_bounds is not None:
            start, stop, global_steps = temporal_bounds
            for index in range(len(self._pairs)):
                value = constants[self._contact_name(index, "time")]
                if int(value.shape[-1]) != global_steps:
                    raise ValueError(
                        f"extra time tensor {index} has {int(value.shape[-1])} "
                        f"samples; expected total_steps={global_steps}"
                    )
            compiled_bounds = (start, stop)
        else:
            compiled_bounds = None
        return self.assemble_values(
            parameters,
            constants,
            times,
            temporal_slice=compiled_bounds,
        )

    def _temporal_bounds(
        self,
        times: torch.Tensor,
        *,
        temporal_slice: slice | None,
        total_steps: int | None,
    ) -> tuple[int, int, int] | None:
        """Resolve and validate host chunk metadata outside compiled code."""

        if self.uses_waveforms:
            if temporal_slice is not None or total_steps is not None:
                raise ValueError(
                    "temporal_slice/total_steps apply only to tensor-temporal "
                    "extra specs; Waveforms evaluate directly on each time chunk"
                )
            return None
        if temporal_slice is None:
            if total_steps is not None:
                if isinstance(total_steps, bool):
                    raise TypeError("total_steps must be a non-negative integer")
                try:
                    total_steps = int(total_steps)
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        "total_steps must be a non-negative integer"
                    ) from exc
                if total_steps != int(times.shape[0]):
                    raise ValueError(
                        "total_steps without temporal_slice must equal the local "
                        "number of timepoints"
                    )
            return None
        if not isinstance(temporal_slice, slice):
            raise TypeError("temporal_slice must be a slice")
        if total_steps is None:
            raise ValueError("total_steps is required with temporal_slice")
        if isinstance(total_steps, bool):
            raise TypeError("total_steps must be a non-negative integer")
        try:
            total_steps = int(total_steps)
        except (TypeError, ValueError) as exc:
            raise TypeError("total_steps must be a non-negative integer") from exc
        if total_steps < 0:
            raise ValueError("total_steps must be non-negative")
        start, stop, step = temporal_slice.indices(total_steps)
        if step != 1:
            raise ValueError("temporal_slice must be contiguous (step 1)")
        if stop - start != int(times.shape[0]):
            raise ValueError(
                "temporal_slice length must equal the local number of timepoints"
            )
        return start, stop, total_steps

    def assemble_values(
        self,
        parameters: Mapping[str, torch.Tensor],
        constants: Mapping[str, torch.Tensor],
        times: torch.Tensor,
        *,
        temporal_slice: tuple[int, int] | None = None,
    ) -> torch.Tensor | None:
        """Tensor-only extracellular lowering for compiled transitions.

        For tensor-temporal specs, ``temporal_slice=(start, stop)`` selects a
        host-resolved contiguous chunk from each globally bound time tensor.
        """

        if not self.enabled:
            return None
        n_steps = int(times.shape[0])

        fields = []
        for index in range(len(self._pairs)):
            fields.append(
                _normalize_spatial(
                    constants[self._contact_name(index, "field")],
                    shape=self.shape,
                    batch_shape=self.batch_shape,
                    name=f"extracellular field {index}",
                )
            )

        temporal_values: tuple[torch.Tensor, ...] | list[torch.Tensor]
        if self.uses_waveforms:
            temporal_values = self._waveforms.evaluate(parameters, constants, times)
        else:
            temporal_values = []
            for index in range(len(self._pairs)):
                value = constants[self._contact_name(index, "time")]
                if temporal_slice is not None:
                    start, stop = temporal_slice
                    value = value[..., start:stop]
                temporal_values.append(value)
        normalized_temporal = []
        for index, value in enumerate(temporal_values):
            normalized_temporal.append(
                _normalize_temporal(
                    value.to(device=self.device, dtype=self.dtype),
                    shape=self.shape,
                    batch_shape=self.batch_shape,
                    n_steps=n_steps,
                    name=(
                        f"extracellular waveform {index}"
                        if self.uses_waveforms
                        else f"extra time tensor {index}"
                    ),
                )
            )

        if self.multicontact:
            spatial = torch.stack(fields, dim=0)
            temporal = torch.stack(normalized_temporal, dim=0)
            result = op_mc(spatial, temporal)
        else:
            result = op_sc(fields[0], normalized_temporal[0])
        return result.contiguous()


__all__ = [
    "FunctionalExtra",
    "FunctionalIntra",
    "StimulusTensors",
    "TensorSchema",
]
