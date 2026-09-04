import copy
import hashlib
import inspect
import random
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, MethodType

import numpy as np
import torch

from dendra.models._class_declarations import (
    consume_class_values,
    declare_class_value,
    merge_buffer_schemas,
    merge_timestep_buffer_shapes,
    normalize_buffer_schema,
    normalize_timestep_buffer_shape,
)
from dendra.models.parametric import Parameterized, cacheable
from dendra.models.rng import RNGModule
from dendra.utils.tensor_ops import _logical_tensor_bytes

from ._bufferimplicit import build_bufferimplicit
from ._cnexp import build_cnexp
from ._derivimplicit import build_derivimplicit
from ._euler_heun import build_euler_heun
from ._euler_maruyama import build_euler_maruyama
from ._kinetic import kinetic_to_derivatives
from ._linearimplicit import build_linearimplicit
from ._mechanism import classproperty
from ._rosenbrock import build_rosenbrock1


def _extend_unique(target, values):
    """Append declarations in authored/MRO order without duplicates."""

    seen = set(target)
    for value in values:
        if value not in seen:
            target.append(value)
            seen.add(value)


def _registered_tensor_items(module):
    """Yield canonical registered tensor slots, including nested modules."""

    for module_path, owner in module.named_modules(remove_duplicate=False):
        for collection_name in ("_parameters", "_buffers"):
            collection = getattr(owner, collection_name)
            for name, value in collection.items():
                if value is None:
                    continue
                yield (module_path, collection_name, name), value


def _registered_tensor_snapshot(module):
    """Snapshot enough registered state to reject mutations by a pure builder."""

    snapshot = {}
    for key, value in _registered_tensor_items(module):
        try:
            version = value._version
        except RuntimeError:
            # Tensors created in inference_mode deliberately have no version
            # counter. Preserve a value copy only for that uncommon lifecycle;
            # ordinary initialization keeps the cheap identity/version path.
            version = None
            fallback = value.detach().clone(memory_format=torch.preserve_format)
        else:
            fallback = None
        snapshot[key] = (
            id(value),
            version,
            tuple(value.shape),
            tuple(value.stride()),
            value.storage_offset(),
            value.dtype,
            value.device,
            value.requires_grad,
            fallback,
        )
    return snapshot


def _registered_tensors_unchanged(module, snapshot):
    """Return whether registered identities, metadata, and values are unchanged."""

    current = dict(_registered_tensor_items(module))
    if set(current) != set(snapshot):
        return False
    for key, value in current.items():
        (
            identity,
            version,
            shape,
            stride,
            storage_offset,
            dtype,
            device,
            requires_grad,
            fallback,
        ) = snapshot[key]
        if (
            id(value) != identity
            or tuple(value.shape) != shape
            or tuple(value.stride()) != stride
            or value.storage_offset() != storage_offset
            or value.dtype != dtype
            or value.device != device
            or value.requires_grad != requires_grad
        ):
            return False
        if version is not None:
            try:
                if value._version != version:
                    return False
            except RuntimeError:
                return False
        else:
            # Compare logical values bit-for-bit. torch.equal reports matching
            # NaNs as unequal, while a zero-tolerance allclose would miss a
            # +0.0/-0.0 mutation that can affect reciprocal expressions.
            if value.numel() == 0:
                continue
            value_bytes = _logical_tensor_bytes(value)
            fallback_bytes = _logical_tensor_bytes(fallback)
            if not torch.equal(value_bytes, fallback_bytes):
                return False
    return True


def _clone_registered_tensor(value):
    """Clone a registered tensor while retaining its autograd connection."""

    if value.layout == torch.strided:
        return value.clone(memory_format=torch.preserve_format)
    return value.clone()


def _seed_isolated_deepcopy_memo(value, memo, seen):
    """Teach ``deepcopy`` how to isolate immutable mapping views and tensors.

    Random-parameter declarations intentionally expose their distribution
    metadata through :class:`types.MappingProxyType`.  Python's default
    ``deepcopy`` cannot pickle those views, even though their contents are
    perfectly safe to copy.  Seed the memo with an equivalent read-only view
    and with graph-connected Tensor clones before copying the surrounding
    module/specification tree.
    """

    identity = id(value)
    if identity in memo or identity in seen:
        return
    seen.add(identity)

    if torch.is_tensor(value):
        memo[identity] = _clone_registered_tensor(value)
        return
    if isinstance(value, MappingProxyType):
        backing = {}
        memo[identity] = MappingProxyType(backing)
        for key, item in value.items():
            _seed_isolated_deepcopy_memo(key, memo, seen)
            _seed_isolated_deepcopy_memo(item, memo, seen)
        for key, item in value.items():
            backing[copy.deepcopy(key, memo)] = copy.deepcopy(item, memo)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _seed_isolated_deepcopy_memo(key, memo, seen)
            _seed_isolated_deepcopy_memo(item, memo, seen)
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            _seed_isolated_deepcopy_memo(item, memo, seen)
        return
    if isinstance(value, torch.nn.Module):
        for item in vars(value).values():
            _seed_isolated_deepcopy_memo(item, memo, seen)
        return
    if callable(value) or isinstance(value, type):
        return
    if hasattr(value, "__dict__"):
        for item in vars(value).values():
            _seed_isolated_deepcopy_memo(item, memo, seen)


def _isolated_deepcopy(value, memo=None):
    """Deep-copy builder-owned state while preserving Tensor gradient edges."""

    memo = {} if memo is None else memo
    _seed_isolated_deepcopy_memo(value, memo, set())
    return copy.deepcopy(value, memo)


_PARAMETERIZED_INSTANCE_SLOTS = frozenset(
    {
        "_init_device",
        "_init_dtype",
        "_random_parameter_generation",
        "_random_parameter_initialized",
        "additional_parameters",
        "batch_n",
        "batch_p",
        "batch_t",
        "flags",
        "global_n",
        "globals",
        "globals_p",
        "in_graph_parametrizations",
        "keys",
        "params",
        "params_n",
        "params_p",
        "random_parameters",
        "range",
        "range_n",
        "range_p",
        "rng",
        "runtime_noises",
        "shape_f",
        "shape_p",
        "training",
    }
)


def _parameterized_workspace_slots(cls):
    """Return names installed by :class:`Parameterized` construction."""

    logical_names = set(cls.all_parameter_names())
    physical_names = set().union(
        *(
            set(getattr(cls, collection, ()))
            for collection in (
                "_global",
                "_global_p",
                "_global_n",
                "_range",
                "_range_p",
                "_range_n",
                "_batch",
                "_batch_p",
                "_batch_n",
            )
        )
    )
    slots = set(_PARAMETERIZED_INSTANCE_SLOTS)
    slots.update(logical_names)
    slots.update(f"{name}_param" for name in physical_names)
    slots.update(getattr(cls, "_rng", ()))
    for spec in (
        *getattr(cls, "_random_parameters", {}).values(),
        *getattr(cls, "_runtime_noises", {}).values(),
    ):
        slots.add(spec.name)
        slots.add(spec.effective_rng_name)
    return slots


@dataclass(frozen=True)
class _TensorClassRestore:
    storage: object
    storage_value: torch.Tensor
    storage_nbytes: int
    storage_offset: int
    size: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout
    requires_grad: bool


@dataclass(frozen=True)
class _NumpyClassRestore:
    value: np.ndarray
    shape: tuple[int, ...]
    dtype: np.dtype


def _snapshot_tensor_class_restore(value):
    """Save a class Tensor's original storage, layout, and complete contents."""

    if value.layout != torch.strided:
        raise RuntimeError(
            "Cannot safely isolate mutable class Tensor state with non-strided "
            f"layout {value.layout}. Move it to declared tensor state."
        )
    storage = value.untyped_storage()
    storage_nbytes = storage.nbytes()
    if storage_nbytes:
        storage_view = torch.empty(
            0,
            device=value.device,
            dtype=torch.uint8,
        ).set_(storage, 0, (storage_nbytes,), (1,))
        storage_value = storage_view.clone()
    else:
        storage_value = torch.empty(
            0,
            device=value.device,
            dtype=torch.uint8,
        )
    return _TensorClassRestore(
        storage=storage,
        storage_value=storage_value,
        storage_nbytes=storage_nbytes,
        storage_offset=value.storage_offset(),
        size=tuple(value.shape),
        stride=tuple(value.stride()),
        dtype=value.dtype,
        device=value.device,
        layout=value.layout,
        requires_grad=value.requires_grad,
    )


def _tensor_argument_snapshot(value):
    """Capture a tensor argument, including inference tensors without versions."""

    try:
        version = value._version
    except RuntimeError:
        version = None
        fallback = value.detach().clone(memory_format=torch.preserve_format)
    else:
        fallback = None
    return (
        id(value),
        version,
        tuple(value.shape),
        tuple(value.stride()),
        value.storage_offset(),
        value.dtype,
        value.device,
        value.requires_grad,
        fallback,
    )


def _tensor_argument_unchanged(value, snapshot):
    (
        identity,
        version,
        shape,
        stride,
        storage_offset,
        dtype,
        device,
        requires_grad,
        fallback,
    ) = snapshot
    if (
        id(value) != identity
        or tuple(value.shape) != shape
        or tuple(value.stride()) != stride
        or value.storage_offset() != storage_offset
        or value.dtype != dtype
        or value.device != device
        or value.requires_grad != requires_grad
    ):
        return False
    if version is not None:
        try:
            return value._version == version
        except RuntimeError:
            return False
    if value.numel() == 0:
        return True
    value_bytes = _logical_tensor_bytes(value)
    fallback_bytes = _logical_tensor_bytes(fallback)
    return torch.equal(value_bytes, fallback_bytes)


def _shared_value_signature(value, seen):
    """Fingerprint authored class state without executing user code."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (float, complex)):
        return type(value).__name__, repr(value)
    if isinstance(value, (torch.dtype, torch.device)):
        return type(value).__name__, str(value)
    if isinstance(value, slice):
        return "slice", value.start, value.stop, value.step
    if torch.is_tensor(value):
        detached = value.detach()
        if detached.layout != torch.strided:
            detached = detached.to_dense()
        if detached.numel() == 0:
            payload = b""
        else:
            payload = _logical_tensor_bytes(detached).numpy().tobytes()
        digest = hashlib.sha256(payload).hexdigest()
        try:
            version = value._version
        except RuntimeError:
            version = None
        return (
            "Tensor",
            id(value),
            version,
            tuple(value.shape),
            value.dtype,
            value.device,
            value.layout,
            tuple(value.stride()) if value.layout == torch.strided else None,
            value.storage_offset() if value.layout == torch.strided else None,
            value.requires_grad,
            digest,
        )
    if isinstance(value, np.generic):
        scalar = np.asarray(value)
        return type(value).__name__, str(scalar.dtype), scalar.tobytes()
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return (
            "ndarray",
            id(value),
            tuple(value.shape),
            str(value.dtype),
            hashlib.sha256(contiguous.tobytes()).hexdigest(),
        )
    if isinstance(value, torch.Generator):
        state = value.get_state().detach().cpu().contiguous().numpy().tobytes()
        return "torch.Generator", id(value), str(value.device), state
    if callable(value):
        return "callable", id(value), type(value).__module__, type(value).__qualname__

    identity = id(value)
    if identity in seen:
        return "reference", identity
    seen.add(identity)
    if isinstance(value, (tuple, list)):
        return (
            type(value).__name__,
            identity,
            tuple(_shared_value_signature(item, seen) for item in value),
        )
    if isinstance(value, (set, frozenset)):
        items = [_shared_value_signature(item, seen) for item in value]
        return type(value).__name__, identity, tuple(sorted(items, key=repr))
    if isinstance(value, dict):
        items = [
            (
                _shared_value_signature(key, seen),
                _shared_value_signature(item, seen),
            )
            for key, item in value.items()
        ]
        return "dict", identity, tuple(sorted(items, key=repr))
    if hasattr(value, "__dict__"):
        attributes = tuple(
            (name, _shared_value_signature(item, seen))
            for name, item in sorted(vars(value).items())
        )
        return (
            "object",
            identity,
            type(value).__module__,
            type(value).__qualname__,
            attributes,
        )
    return (
        "opaque",
        identity,
        type(value).__module__,
        type(value).__qualname__,
        repr(value),
    )


def _authored_builder_classes(module):
    """Return authored classes shared by every copy of the builder tree."""

    classes = []
    seen = set()
    for owner in module.modules():
        owned_classes = []
        found_framework_base = False
        for cls in type(owner).__mro__:
            if cls.__name__ in {"Mechanism", "State"} and cls.__module__.startswith(
                "dendra.models.mechanisms"
            ):
                found_framework_base = True
                break
            owned_classes.append(cls)
        if not found_framework_base:
            continue
        for cls in owned_classes:
            if cls not in seen:
                seen.add(cls)
                classes.append(cls)
    return tuple(classes)


def _snapshot_authored_class_state(module):
    """Capture restorable class attributes shared by every deepcopy."""

    snapshot = {}
    for cls in _authored_builder_classes(module):
        attributes = {}
        for name, value in vars(cls).items():
            if name.startswith("__") and name.endswith("__"):
                continue
            signature = _shared_value_signature(value, set())
            if isinstance(value, (classmethod, staticmethod, property)) or callable(
                value
            ):
                restored = value
            elif torch.is_tensor(value):
                restored = _snapshot_tensor_class_restore(value)
            elif isinstance(value, np.ndarray):
                restored = _NumpyClassRestore(
                    value=value.copy(),
                    shape=tuple(value.shape),
                    dtype=value.dtype,
                )
            elif isinstance(value, torch.Generator):
                restored = torch.Generator(device=value.device)
                restored.set_state(value.get_state())
            else:
                try:
                    restored = _isolated_deepcopy(value)
                except Exception as exc:
                    raise RuntimeError(
                        "Cannot safely isolate mutable class state "
                        f"{cls.__module__}.{cls.__qualname__}.{name}; move it "
                        "to declared tensor state or an immutable literal."
                    ) from exc
            attributes[name] = signature, value, restored
        snapshot[cls] = attributes
    return snapshot


def _restore_authored_class_state(snapshot):
    """Restore shared authored class state and report changed attributes."""

    changed = set()
    for cls, attributes in snapshot.items():
        current_names = {
            name
            for name in vars(cls)
            if not (name.startswith("__") and name.endswith("__"))
        }
        expected_names = set(attributes)
        for name in current_names - expected_names:
            changed.add(f"{cls.__module__}.{cls.__qualname__}.{name}")
            delattr(cls, name)
        for name, (signature, original, restored) in attributes.items():
            current = vars(cls).get(name)
            current_signature = (
                None
                if name not in vars(cls)
                else _shared_value_signature(current, set())
            )
            if current_signature == signature:
                continue
            changed.add(f"{cls.__module__}.{cls.__qualname__}.{name}")
            if isinstance(original, dict):
                original.clear()
                original.update(_isolated_deepcopy(restored))
                restored_value = original
            elif isinstance(original, list):
                original[:] = _isolated_deepcopy(restored)
                restored_value = original
            elif isinstance(original, set):
                original.clear()
                original.update(_isolated_deepcopy(restored))
                restored_value = original
            elif isinstance(original, np.ndarray):
                if original.dtype != restored.dtype:
                    original.dtype = restored.dtype
                original.resize(restored.shape, refcheck=False)
                np.copyto(original, restored.value)
                restored_value = original
            elif torch.is_tensor(original):
                with torch.no_grad():
                    # Repair the captured storage itself before reattaching the
                    # original Tensor object. This restores aliases as well as
                    # mutate-then-rebind attempts and overlapping/expanded
                    # layouts for which logical ``copy_`` is not writable.
                    if restored.storage.nbytes() != restored.storage_nbytes:
                        restored.storage.resize_(restored.storage_nbytes)
                    if restored.storage_nbytes:
                        storage_view = torch.empty(
                            0,
                            device=restored.device,
                            dtype=torch.uint8,
                        ).set_(
                            restored.storage,
                            0,
                            (restored.storage_nbytes,),
                            (1,),
                        )
                        storage_view.copy_(restored.storage_value)
                        restored_view = torch.empty(
                            0,
                            device=restored.device,
                            dtype=restored.dtype,
                            layout=restored.layout,
                        ).set_(
                            restored.storage,
                            restored.storage_offset,
                            restored.size,
                            restored.stride,
                        )
                        # ``set_`` alone cannot restore dtype/device metadata
                        # changed through ``tensor.data = ...``. Reattach the
                        # exact typed view while preserving the original Python
                        # identity and captured shared raw storage.
                        original.data = restored_view
                    if original.requires_grad != restored.requires_grad:
                        original.requires_grad_(restored.requires_grad)
                restored_value = original
            elif isinstance(original, torch.Generator):
                original.set_state(restored.get_state())
                restored_value = original
            else:
                restored_value = original if current is not original else restored
            setattr(cls, name, restored_value)
    return changed


def _snapshot_global_rng(device):
    snapshot = {
        "torch": torch.random.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if device.type == "cuda":  # pragma: no cover - CUDA test slice
        snapshot["cuda"] = torch.cuda.get_rng_state(device)
    if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        snapshot["mps"] = torch.mps.get_rng_state()
    return snapshot


def _numpy_rng_state_equal(left, right):
    return all(
        (
            np.array_equal(left_item, right_item)
            if isinstance(left_item, np.ndarray)
            else left_item == right_item
        )
        for left_item, right_item in zip(left, right, strict=True)
    )


def _global_rng_changes(snapshot, device):
    changed = set()
    if not torch.equal(torch.random.get_rng_state(), snapshot["torch"]):
        changed.add("global PyTorch")
    if random.getstate() != snapshot["python"]:
        changed.add("global Python")
    if not _numpy_rng_state_equal(np.random.get_state(), snapshot["numpy"]):
        changed.add("global NumPy")
    if "cuda" in snapshot and not torch.equal(
        torch.cuda.get_rng_state(device), snapshot["cuda"]
    ):
        changed.add("CUDA")
    if "mps" in snapshot and not torch.equal(
        torch.mps.get_rng_state(), snapshot["mps"]
    ):
        changed.add("MPS")
    return changed


def _restore_global_rng(snapshot, device):
    torch.random.set_rng_state(snapshot["torch"])
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    if "cuda" in snapshot:  # pragma: no cover - CUDA test slice
        torch.cuda.set_rng_state(snapshot["cuda"], device)
    if "mps" in snapshot:
        torch.mps.set_rng_state(snapshot["mps"])


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


def _python_instance_state_snapshot(module):
    """Fingerprint unregistered Python state throughout a builder module."""

    snapshot = {
        "<module-tree>": tuple(
            (path, id(owner), type(owner).__module__, type(owner).__qualname__)
            for path, owner in module.named_modules()
        )
    }
    for path, owner in module.named_modules():
        label = path or "<root>"
        for name, value in sorted(vars(owner).items()):
            if name in _MODULE_INTERNAL_STATE:
                continue
            if name == "_cache" and isinstance(owner, cacheable):
                continue
            snapshot[f"{label}.{name}"] = _shared_value_signature(value, set())
    return snapshot


class _IsolatedBuilderCall(torch.nn.Module):
    """Run an authored workspace builder and audit its disposable module."""

    def __init__(self, module, method_name):
        super().__init__()
        self.candidate = module
        self.method_name = method_name
        self.registered_tensors_changed = False
        self.tensor_arguments_changed = False
        self.python_instance_state_changed = set()

    def forward(self, *args):
        before = _registered_tensor_snapshot(self.candidate)
        python_state_before = _python_instance_state_snapshot(self.candidate)
        argument_leaves, argument_spec = torch.utils._pytree.tree_flatten(args)
        argument_snapshots = tuple(
            _tensor_argument_snapshot(argument) if torch.is_tensor(argument) else None
            for argument in argument_leaves
        )
        try:
            return getattr(self.candidate, self.method_name)(*args)
        finally:
            self.registered_tensors_changed = not _registered_tensors_unchanged(
                self.candidate, before
            )
            current_leaves, current_spec = torch.utils._pytree.tree_flatten(args)
            self.tensor_arguments_changed = (
                current_spec != argument_spec
                or len(current_leaves) != len(argument_snapshots)
                or any(
                    snapshot is not None
                    and (
                        not torch.is_tensor(argument)
                        or not _tensor_argument_unchanged(argument, snapshot)
                    )
                    for argument, snapshot in zip(
                        current_leaves,
                        argument_snapshots,
                        strict=False,
                    )
                )
            )
            python_state_after = _python_instance_state_snapshot(self.candidate)
            self.python_instance_state_changed = {
                key
                for key in python_state_before.keys() | python_state_after.keys()
                if python_state_before.get(key) != python_state_after.get(key)
            }


def _evaluate_registered_builder(module, method_name, *args):
    """Evaluate a pure builder without exposing the source module to writes.

    Every registered tensor in the disposable module is an explicit clone with
    an autograd edge back to its source.  ``functional_call`` supplies those
    tensors as the complete state of the probe while the wrapper snapshots the
    probe *inside* that context.  Authored in-place writes and registered-slot
    rebindings are therefore observable, but cannot change a source tensor's
    identity, value, or version counter.
    """

    memo = {}
    for _key, value in _registered_tensor_items(module):
        memo.setdefault(id(value), _clone_registered_tensor(value))
    candidate = _isolated_deepcopy(module, memo)
    probe = _IsolatedBuilderCall(candidate, method_name)

    replacements = {}
    for name, value in candidate.named_parameters():
        replacements[f"candidate.{name}"] = value
    for name, value in candidate.named_buffers():
        replacements[f"candidate.{name}"] = value

    isolated_args = tuple(
        (
            _clone_registered_tensor(argument)
            if torch.is_tensor(argument)
            else _isolated_deepcopy(argument)
        )
        for argument in args
    )
    class_state = _snapshot_authored_class_state(module)
    rng_state = _snapshot_global_rng(module.diam.device)
    try:
        values = torch.func.functional_call(
            probe,
            replacements,
            isolated_args,
            strict=True,
        )
    finally:
        try:
            changed_class_state = _restore_authored_class_state(class_state)
        finally:
            try:
                consumed_rng = _global_rng_changes(rng_state, module.diam.device)
            finally:
                # Shared RNG restoration is unconditional, including when an
                # exotic class-state repair itself raises.
                _restore_global_rng(rng_state, module.diam.device)

    if changed_class_state:
        raise RuntimeError(
            f"{type(module).__name__}.{method_name}() must not mutate shared class "
            f"state: {sorted(changed_class_state)}."
        )
    if consumed_rng:
        raise RuntimeError(
            f"{type(module).__name__}.{method_name}() must not consume implicit "
            f"RNG state: {sorted(consumed_rng)}; use explicit declared RNG state."
        )
    if probe.python_instance_state_changed:
        raise RuntimeError(
            f"{type(module).__name__}.{method_name}() must not mutate Python "
            "instance state; declare it as explicit tensor state instead: "
            f"{sorted(probe.python_instance_state_changed)}."
        )
    if probe.registered_tensors_changed or probe.tensor_arguments_changed:
        raise RuntimeError(
            f"{type(module).__name__}.{method_name}() must not mutate registered "
            "parameters, buffers, or tensor arguments; return the derived tensors "
            "instead."
        )
    return values


def _normalize_initial_values(
    module,
    values,
    references,
    *,
    method_name="initial_values",
):
    """Validate and clone one authored initialization-value mapping.

    ``references`` is the declaration-owned output schema.  Hooks may return a
    partial mapping, but every returned name must already be owned by the
    receiver and every Tensor must preserve that leaf's device and dtype.  A
    scalar (or other broadcastable view) is expanded to the declared local
    shape and cloned so returned aliases never become mutable runtime carry.
    """

    if not isinstance(values, Mapping):
        raise TypeError(
            f"{type(module).__qualname__}.{method_name}() must return a mapping"
        )

    unknown = tuple(name for name in values if name not in references)
    if unknown:
        raise KeyError(
            f"{type(module).__qualname__}.{method_name}() returned undeclared "
            f"outputs {sorted(map(str, unknown))}"
        )

    normalized = {}
    for name, value in values.items():
        reference = references[name]
        if not torch.is_tensor(value):
            raise TypeError(
                f"{type(module).__qualname__}.{method_name}()[{name!r}] must "
                "be a Tensor"
            )
        if value.device != reference.device or value.dtype != reference.dtype:
            raise ValueError(
                f"{type(module).__qualname__}.{method_name}()[{name!r}] must "
                f"use {reference.device}/{reference.dtype}; got "
                f"{value.device}/{value.dtype}"
            )
        if _is_unresolved_deferred_buffer(module, name):
            normalized[name] = value.clone(memory_format=torch.preserve_format)
            continue
        try:
            broadcast_shape = torch.broadcast_shapes(
                tuple(value.shape), tuple(reference.shape)
            )
        except RuntimeError as exc:
            raise ValueError(
                f"{type(module).__qualname__}.{method_name}()[{name!r}] with "
                f"shape {tuple(value.shape)} is not broadcastable to declared "
                f"shape {tuple(reference.shape)}"
            ) from exc
        if tuple(broadcast_shape) != tuple(reference.shape):
            raise ValueError(
                f"{type(module).__qualname__}.{method_name}()[{name!r}] with "
                f"shape {tuple(value.shape)} is not broadcastable to declared "
                f"shape {tuple(reference.shape)}"
            )
        normalized[name] = value.expand_as(reference).clone(
            memory_format=torch.preserve_format
        )
    return normalized


def _validate_runtime_outputs(
    module,
    method_name,
    outputs,
    references,
    *,
    required=(),
):
    """Validate one runtime transition schema without cloning live tensors."""
    if not isinstance(outputs, Mapping):
        raise TypeError(
            f"{type(module).__qualname__}.{method_name}() must return a mapping"
        )
    names = tuple(outputs)
    unknown = tuple(name for name in names if name not in references)
    if unknown:
        raise KeyError(
            f"{type(module).__qualname__}.{method_name}() returned undeclared "
            f"outputs {sorted(map(str, unknown))}"
        )
    missing = tuple(name for name in required if name not in outputs)
    if missing:
        raise KeyError(
            f"{type(module).__qualname__}.{method_name}() did not return "
            f"declared outputs {list(missing)}"
        )
    for name, value in outputs.items():
        reference = references[name]
        if not torch.is_tensor(value):
            raise TypeError(
                f"{type(module).__qualname__}.{method_name}()[{name!r}] must "
                "be a Tensor"
            )
        if (
            value.device != reference.device
            or value.dtype != reference.dtype
            or tuple(value.shape) != tuple(reference.shape)
        ):
            raise ValueError(
                f"{type(module).__qualname__}.{method_name}()[{name!r}] must "
                f"match {reference.device}/{reference.dtype}/"
                f"{tuple(reference.shape)}; got {value.device}/{value.dtype}/"
                f"{tuple(value.shape)}"
            )
    return outputs


def _reset_runtime_output_validation(module):
    """Invalidate one-shot transition schemas after tensor layout conversion."""
    module._assigned_schema_validated = False
    module._advance_schema_validated = False
    module._advance_return_names = ()


def _deferred_resolution_map(module, name):
    """Return the per-instance resolution map for a deferred declared buffer."""

    carry_specs = getattr(module, "_carry_specs", {})
    if name in carry_specs and carry_specs[name][1] == "deferred":
        return getattr(module, "_carry_resolved_shapes", None)
    derived_specs = getattr(module, "_derived_buffer_specs", {})
    if name in derived_specs and derived_specs[name][1] == "deferred":
        return getattr(module, "_derived_resolved_shapes", None)
    return None


def _is_unresolved_deferred_buffer(module, name) -> bool:
    resolutions = _deferred_resolution_map(module, name)
    return resolutions is not None and resolutions.get(name) is None


def _install_declared_buffer_value(module, name, value):
    """Install a declared tensor while freezing any deferred shape once."""

    resolutions = _deferred_resolution_map(module, name)
    if resolutions is not None:
        shape = tuple(value.shape)
        resolved = resolutions.get(name)
        if resolved is None:
            resolutions[name] = shape
        elif tuple(resolved) != shape:
            raise ValueError(
                f"{type(module).__qualname__}.{name} has frozen deferred shape "
                f"{tuple(resolved)}, but received {shape}"
            )
    setattr(module, name, value)


def _unresolved_deferred_buffer_names(module) -> tuple[str, ...]:
    """Return declared deferred buffers that have not established a shape."""

    unresolved = []
    for attribute in ("_carry_resolved_shapes", "_derived_resolved_shapes"):
        for name, shape in getattr(module, attribute, {}).items():
            if shape is None:
                unresolved.append(name)
    return tuple(unresolved)


def _resolve_deferred_buffers_from_state_dict(module, state_dict, prefix):
    """Resolve fresh deferred slots from incoming checkpoint tensor shapes."""

    for name in _unresolved_deferred_buffer_names(module):
        key = f"{prefix}{name}"
        incoming = state_dict.get(key)
        if not torch.is_tensor(incoming):
            continue
        current = module._buffers.get(name)
        if current is None:
            continue
        module._buffers[name] = current.new_empty(tuple(incoming.shape))
        resolutions = _deferred_resolution_map(module, name)
        resolutions[name] = tuple(incoming.shape)


def _support_visible_value(value, like):
    """Pad a registered scalar/low-rank input to the local support rank.

    The padding is explicit in the authored ``initial_values`` input frame;
    the registered attribute itself keeps its established visible shape.  This
    makes empty and non-empty ``vmap`` lanes behave alike without silently
    changing the semantics of ``self.celsius`` elsewhere.
    """

    if value.ndim >= like.ndim:
        return value
    return value.reshape((1,) * (like.ndim - value.ndim) + tuple(value.shape))


def _materialize_derived_buffers(module):
    """Validate and install one module's initialization-static workspaces."""

    expected = set(module._derived_buffers)
    if not expected:
        return

    values = _evaluate_registered_builder(module, "derive_buffers")

    if not isinstance(values, Mapping):
        raise TypeError(
            f"{type(module).__name__}.derive_buffers() must return a mapping."
        )
    actual = set(values)
    if actual != expected:
        missing = sorted(map(str, expected - actual))
        unexpected = sorted(map(str, actual - expected))
        raise ValueError(
            f"{type(module).__name__}.derive_buffers() keys do not match "
            f"DERIVED_BUFFER declarations; missing={missing}, "
            f"unexpected={unexpected}."
        )

    staged = {}
    for name in sorted(expected):
        value = values[name]
        reference = module._buffers[name]
        if not torch.is_tensor(value):
            raise TypeError(
                f"{type(module).__name__}.derive_buffers()[{name!r}] must be a Tensor."
            )
        if value.device != reference.device or value.dtype != reference.dtype:
            raise ValueError(
                f"{type(module).__name__}.derive_buffers()[{name!r}] must use "
                f"{reference.device}/{reference.dtype}; got "
                f"{value.device}/{value.dtype}."
            )
        if not _is_unresolved_deferred_buffer(module, name):
            target_shape = tuple(reference.shape)
            try:
                broadcast_shape = torch.broadcast_shapes(
                    tuple(value.shape), target_shape
                )
            except RuntimeError as exc:
                raise ValueError(
                    f"{type(module).__name__}.derive_buffers()[{name!r}] with "
                    f"shape {tuple(value.shape)} is not broadcastable to "
                    f"declared shape {target_shape}."
                ) from exc
            if tuple(broadcast_shape) != target_shape:
                raise ValueError(
                    f"{type(module).__name__}.derive_buffers()[{name!r}] with "
                    f"shape {tuple(value.shape)} is not broadcastable to "
                    f"declared shape {target_shape}."
                )
            value = value.expand_as(reference)
        # A registered buffer must remain independently writable by ordinary
        # state_dict/checkpoint restoration. Builders may naturally return a
        # view (for example scalar.expand_as(diam)); clone it so overlapping
        # strides or dependency aliases do not turn the installed buffer into
        # an unwritable checkpoint destination. clone() retains autograd.
        staged[name] = value.clone(memory_format=torch.preserve_format)

    # Install only after the whole mapping has passed validation. Assignment is
    # deliberately out of place so gradients through the builder are retained.
    for name, value in staged.items():
        _install_declared_buffer_value(module, name, value)


def _canonical_timestep_buffer_shape(module, name) -> tuple[int, ...]:
    """Resolve one declared timestep workspace to its registered shape."""

    declared = getattr(module, "_timestep_buffer_shapes", {}).get(name, "local")
    if declared == "local":
        # ``diam`` intentionally omits explicit Population batch prefixes,
        # while RANGE parameters and their local workspaces retain them.
        # ``shape_p`` is therefore the authored parameter/workspace shape.
        return tuple(getattr(module, "shape_p", module.diam.shape))
    return tuple(declared)


def _timestep_buffer_schema(module, values=None):
    """Return declaration and realized-shape metadata for checkpoint audits."""

    declarations = tuple(
        (name, getattr(module, "_timestep_buffer_shapes", {}).get(name, "local"))
        for name in sorted(getattr(module, "_timestep_buffers", ()))
    )
    if values is None:
        realized = tuple(
            (name, _canonical_timestep_buffer_shape(module, name))
            for name, _ in declarations
        )
    else:
        realized = tuple((name, tuple(values[name].shape)) for name, _ in declarations)
    return declarations, realized


def _stage_timestep_buffers(module, dt):
    """Validate one module's timestep workspaces without installing them."""

    expected = set(getattr(module, "_timestep_buffers", ()))
    if not expected:
        return {}

    values = _evaluate_registered_builder(
        module,
        "derive_timestep_buffers",
        dt,
    )

    if not isinstance(values, Mapping):
        raise TypeError(
            f"{type(module).__name__}.derive_timestep_buffers() must return a mapping."
        )
    actual = set(values)
    if actual != expected:
        missing = sorted(map(str, expected - actual))
        unexpected = sorted(map(str, actual - expected))
        raise ValueError(
            f"{type(module).__name__}.derive_timestep_buffers() keys do not match "
            f"TIMESTEP_BUFFER declarations; missing={missing}, "
            f"unexpected={unexpected}."
        )

    reference = module.diam
    staged = {}
    for name in sorted(expected):
        canonical_shape = _canonical_timestep_buffer_shape(module, name)
        value = values[name]
        if not torch.is_tensor(value):
            raise TypeError(
                f"{type(module).__name__}.derive_timestep_buffers()[{name!r}] "
                "must be a Tensor."
            )
        if value.device != reference.device or value.dtype != reference.dtype:
            raise ValueError(
                f"{type(module).__name__}.derive_timestep_buffers()[{name!r}] "
                f"must use device/dtype {reference.device}/{reference.dtype}; got "
                f"{value.device}/{value.dtype}."
            )
        try:
            broadcast_shape = torch.broadcast_shapes(
                tuple(value.shape), canonical_shape
            )
        except RuntimeError as exc:
            raise ValueError(
                f"{type(module).__name__}.derive_timestep_buffers()[{name!r}] "
                f"with shape {tuple(value.shape)} is not broadcastable to "
                f"canonical shape {canonical_shape}."
            ) from exc
        if tuple(broadcast_shape) != canonical_shape:
            raise ValueError(
                f"{type(module).__name__}.derive_timestep_buffers()[{name!r}] "
                f"with shape {tuple(value.shape)} is not broadcastable to "
                f"canonical shape {canonical_shape}."
            )
        # Store every timestep workspace in one canonical layout.  Accepting a
        # scalar or another broadcastable view is convenient for authors, but
        # installing that raw shape makes the registered-buffer schema depend on
        # whether timestep configuration has already run.  That in turn
        # prevents a strict
        # state_dict round trip into an equivalent fresh model.  Expanding to
        # the declared canonical shape before cloning gives the slot a
        # lifecycle-stable schema and retains the builder's autograd connection
        # through BroadcastBackward/CloneBackward. ``shape="local"`` follows
        # shape_p; explicit structural shapes remain Population-independent.
        staged[name] = torch.broadcast_to(value, canonical_shape).clone(
            memory_format=torch.preserve_format
        )

    return staged


def _install_timestep_buffers(module, staged):
    """Commit a previously validated timestep-workspace mapping."""

    if not staged:
        return
    for name, value in staged.items():
        module._buffers[name] = value
    module._timestep_buffer_schema = _timestep_buffer_schema(module, staged)


# Integration-method registry -------------------------------------------------
#
# State subclasses can declare their solver with State.METHOD(...).  The registry
# keeps method dispatch out of State.__init__ and makes it straightforward to add
# new generated solvers such as linearimplicit/sparse, rosenbrock1, or
# bufferimplicit without editing the State class again.
_INTEGRATION_BUILDERS = {}
_INTEGRATION_ALIASES = {}


def _canonical_method_name(method):
    if method is None:
        method = "cnexp"
    if not isinstance(method, str):
        raise TypeError(
            f"State integration method must be a string; got {type(method).__name__}."
        )
    method = method.strip().lower().replace("-", "_")
    return _INTEGRATION_ALIASES.get(method, method)


def register_integration_method(name, builder, *, aliases=()):
    """Register a generated integration-function builder.

    Parameters
    ----------
    name : str
        Canonical method name, e.g. ``"cnexp"`` or ``"derivimplicit"``.
    builder : callable
        Function with signature ``builder(states, assigned, derivative,
        eliminate=None, **method_kwargs)`` returning a generated ``solve``
        function.
    aliases : iterable[str], optional
        Additional names accepted by :meth:`State.METHOD`.
    """
    canonical = str(name).strip().lower().replace("-", "_")
    if not canonical:
        raise ValueError("Integration method name cannot be empty.")
    _INTEGRATION_BUILDERS[canonical] = builder
    _INTEGRATION_ALIASES[canonical] = canonical
    for alias in aliases:
        alias = str(alias).strip().lower().replace("-", "_")
        if not alias:
            raise ValueError("Integration method alias cannot be empty.")
        _INTEGRATION_ALIASES[alias] = canonical
    return builder


def valid_integration_methods():
    """Return the currently registered canonical integration method names."""
    return tuple(sorted(_INTEGRATION_BUILDERS))


def _merge_method_config(current_method, current_kwargs, method, kwargs):
    """Merge inherited/class-body method declarations.

    ``method=None`` means keep the current/inherited method and only update the
    options.  If the method changes, inherited options are intentionally dropped:
    carrying ``max_iter`` from ``derivimplicit`` into ``cnexp`` would be a subtle
    configuration bug.
    """
    kwargs = dict(kwargs or {})
    if method is None:
        current_kwargs.update(kwargs)
        return current_method, current_kwargs

    method = _canonical_method_name(method)
    if method != current_method:
        current_kwargs = {}
    current_kwargs.update(kwargs)
    return method, current_kwargs


def build_integration_func(
    states,
    assigned,
    derivative,
    method,
    eliminate=None,
    diffusion=None,
    **method_kwargs,
):
    """Build the integration function for a State subclass."""
    method = _canonical_method_name(method)
    try:
        builder = _INTEGRATION_BUILDERS[method]
    except KeyError as exc:
        valid = ", ".join(valid_integration_methods())
        raise ValueError(
            f"Unknown integration method: {method!r}. Valid methods are: {valid}."
        ) from exc

    if diffusion:
        if method not in {"euler_maruyama", "euler_heun"}:
            raise ValueError(
                "State.DIFFUSION(...) requires State.METHOD('euler_maruyama') "
                "or State.METHOD('euler_heun') in v1. Voltage/cable SDE solvers "
                "are intentionally out of scope."
            )
        return builder(
            states,
            assigned,
            derivative,
            eliminate=eliminate,
            diffusion=diffusion,
            **method_kwargs,
        )

    if method in {"euler_maruyama", "euler_heun"}:
        return builder(
            states,
            assigned,
            derivative,
            eliminate=eliminate,
            diffusion=(),
            **method_kwargs,
        )

    return builder(
        states,
        assigned,
        derivative,
        eliminate=eliminate,
        **method_kwargs,
    )


register_integration_method("cnexp", build_cnexp, aliases=("rush_larsen", "rl"))
register_integration_method(
    "derivimplicit",
    build_derivimplicit,
    aliases=("deriv_implicit", "implicit", "backward_euler", "be"),
)
register_integration_method(
    "bufferimplicit",
    build_bufferimplicit,
    aliases=(
        "buffer",
        "implicit_buffer",
        "ca_buffer",
        "calcium_buffer",
        "calciumbuffer",
    ),
)
register_integration_method(
    "linearimplicit",
    build_linearimplicit,
    aliases=(
        "linear_implicit",
        "implicitlinear",
        "implicit_linear",
        "affineimplicit",
        "affine_implicit",
        "sparse",
        "be_linear",
        "backward_euler_linear",
        "linear_be",
    ),
)
register_integration_method(
    "euler_maruyama",
    build_euler_maruyama,
    aliases=("em", "sde", "euler-maruyama", "ito"),
)
register_integration_method(
    "euler_heun",
    build_euler_heun,
    aliases=("eh", "stochastic_heun", "stratonovich", "euler-heun"),
)

register_integration_method(
    "rosenbrock",
    build_rosenbrock1,
    aliases=(
        "rosenbrock1",
        "rosenbrock_euler",
        "rosenbrock-euler",
        "linearlyimplicit",
        "linearly_implicit",
        "linearized_implicit",
        "linearly_implicit_euler",
        "linimplicit",
        "semiimplicit",
        "semi_implicit",
    ),
)


class State(Parameterized):
    """
    Helper mixin for declaring per-compartment state variables and their dynamics.

    Define subclasses inside a :class:`Mechanism` and register them with
    :meth:`Mechanism.STATE_BUNDLE`. Use uppercase classmethods (``STATE``, ``DERIVATIVE``,
    ``KINETIC``, ``ASSIGNED``, ``CARRY``, ``GLOBAL``, ``RANGE``) at class
    definition time to declare state variables, ODEs/kinetics, per-compartment
    parameters, and auxiliary buffers. Override lowercase hooks to implement
    behavior:

    - ``state_defaults(self, v, values)``: return lower-priority defaults for
      declared solver state.
    - ``initial_values(self, v, values)``: return a pure initialization overlay
      for declared state and persistent carry.
    - ``assigned_values(self, v, values)``: compute ephemeral ``ASSIGNED`` values
      needed by the generated solver.
    - ``advance(self, v, dt, values)``: return the next declared state/carry.

    Notes
    -----
    State equations advance on Dendra's millisecond time coordinate. A
    derivative of a state ``x`` therefore has units ``unit(x)/ms``; an SDE
    diffusion coefficient has units ``unit(x)/sqrt(ms)``. Voltage arguments are
    in mV, ``self.celsius`` is in °C, and compartment diameters are in µm.
    """

    _carry = ()
    _carry_specs = {}
    _carry_declarations = []

    _state = ()
    _state_declarations = []

    _derivative = set()
    _derivative_declarations = []

    _kinetic = set()
    _kinetic_declarations = []

    _diffusion = set()
    _diffusion_declarations = []

    _assigned = ()
    _assigned_declarations = []

    _derived_buffers = set()
    _derived_buffer_specs = {}
    _derived_buffer_declarations = []

    _timestep_buffers = set()
    _timestep_buffer_shapes = {}
    _timestep_buffer_declarations = []

    # Public introspection aliases are assigned to each concrete subclass after
    # its METHOD declarations have been resolved. Author integration policy with
    # State.METHOD(...), never by assigning these attributes in a class body.
    method = "cnexp"
    method_kwargs = {}

    _method = "cnexp"
    _method_kwargs = {}
    _method_declarations = []

    def __init_subclass__(cls, **kwargs):
        """
        This special method is called automatically whenever a class
        inherits from Parameterized.
        """
        forbidden_hooks = {
            "breakpoint": "assigned_values",
            "inf": "state_defaults",
            "initial": "initial_values",
            "initial_outputs": "initial_values",
            "initialize": "initial_values",
            "solve": "advance",
            "_advance": "advance",
        }
        authored_forbidden = {
            name: replacement
            for name, replacement in forbidden_hooks.items()
            if name in cls.__dict__
        }
        if authored_forbidden:
            details = ", ".join(
                f"{name} -> {replacement}"
                for name, replacement in sorted(authored_forbidden.items())
            )
            raise TypeError(
                f"State {cls.__qualname__} defines removed lifecycle hooks: {details}."
            )
        direct_method = {"method", "method_kwargs"} & set(cls.__dict__)
        if direct_method:
            raise TypeError(
                f"State {cls.__qualname__} assigns {sorted(direct_method)} "
                "directly; declare the solver with State.METHOD(...) instead."
            )

        # Call the parent's __init_subclass__ WITHOUT our custom kwargs,
        # as the base 'object' class does not accept them.
        super().__init_subclass__(**kwargs)

        # Start with a fresh dictionary for the new class's parameters.
        new_state = []
        new_carry = []
        new_carry_specs = {}
        new_derivative = set()
        new_assigned = []
        new_derived_buffers = set()
        new_derived_buffer_specs = {}
        new_timestep_buffers = set()
        new_timestep_buffer_shapes = {}
        new_kinetic = set()
        new_diffusion = set()
        new_method = "cnexp"
        new_method_kwargs = {}

        # Walk MRO in reverse to build up params from parent to child.
        for base in reversed(cls.__mro__):
            if "_state" in base.__dict__:
                _extend_unique(new_state, base._state)
            if "_carry" in base.__dict__:
                _extend_unique(new_carry, base._carry)
                inherited_specs = {
                    name: base.__dict__.get("_carry_specs", {}).get(
                        name, (None, "local")
                    )
                    for name in base._carry
                }
                merge_buffer_schemas(
                    new_carry_specs,
                    inherited_specs,
                    owner=cls,
                    declaration="CARRY",
                )
            if "_derivative" in base.__dict__:
                new_derivative.update(base._derivative)
            if "_kinetic" in base.__dict__:
                new_kinetic.update(base._kinetic)
            if "_diffusion" in base.__dict__:
                new_diffusion.update(base._diffusion)
            if "_assigned" in base.__dict__:
                _extend_unique(new_assigned, base._assigned)
            if "_derived_buffers" in base.__dict__:
                new_derived_buffers.update(base._derived_buffers)
                inherited_specs = {
                    name: base.__dict__.get("_derived_buffer_specs", {}).get(
                        name, (None, "local")
                    )
                    for name in base._derived_buffers
                }
                merge_buffer_schemas(
                    new_derived_buffer_specs,
                    inherited_specs,
                    owner=cls,
                    declaration="DERIVED_BUFFER",
                )
            if "_timestep_buffers" in base.__dict__:
                new_timestep_buffers.update(base._timestep_buffers)
                inherited_shapes = {
                    name: base.__dict__.get("_timestep_buffer_shapes", {}).get(
                        name, "local"
                    )
                    for name in base._timestep_buffers
                }
                merge_timestep_buffer_shapes(
                    new_timestep_buffer_shapes,
                    inherited_shapes,
                    owner=cls,
                )

            if "_method" in base.__dict__:
                new_method, new_method_kwargs = _merge_method_config(
                    new_method,
                    new_method_kwargs,
                    base.__dict__["_method"],
                    base.__dict__.get("_method_kwargs", {}),
                )

        for s_list in consume_class_values(
            cls, "state.state", State._state_declarations
        ):
            _extend_unique(new_state, s_list)
        for declaration in consume_class_values(
            cls, "state.carry", State._carry_declarations
        ):
            merge_buffer_schemas(
                new_carry_specs,
                declaration,
                owner=cls,
                declaration="CARRY",
            )
            _extend_unique(new_carry, declaration)
        for d_list in consume_class_values(
            cls, "state.derivative", State._derivative_declarations
        ):
            new_derivative.update(d_list)
        for k_list in consume_class_values(
            cls, "state.kinetic", State._kinetic_declarations
        ):
            new_kinetic.update(k_list)
        for d_list in consume_class_values(
            cls, "state.diffusion", State._diffusion_declarations
        ):
            new_diffusion.update(d_list)
        for a_list in consume_class_values(
            cls, "state.assigned", State._assigned_declarations
        ):
            _extend_unique(new_assigned, a_list)
        for declaration in consume_class_values(
            cls,
            "state.derived_buffers",
            State._derived_buffer_declarations,
        ):
            if isinstance(declaration, dict):
                specs = declaration
            else:
                specs = {name: (None, "local") for name in declaration}
            merge_buffer_schemas(
                new_derived_buffer_specs,
                specs,
                owner=cls,
                declaration="DERIVED_BUFFER",
            )
            new_derived_buffers.update(specs)
        for declaration in consume_class_values(
            cls,
            "state.timestep_buffers",
            State._timestep_buffer_declarations,
        ):
            if isinstance(declaration, dict):
                shapes = declaration
            else:
                shapes = {name: "local" for name in declaration}
            merge_timestep_buffer_shapes(
                new_timestep_buffer_shapes,
                shapes,
                owner=cls,
            )
            new_timestep_buffers.update(shapes)
        overlap = new_derived_buffers & new_timestep_buffers
        if overlap:
            raise ValueError(
                "State buffers cannot be both DERIVED_BUFFER and "
                f"TIMESTEP_BUFFER: {sorted(overlap)}"
            )
        state_names = set(new_state)
        assigned_names = set(new_assigned)
        carry_names = set(new_carry)
        workspace_buffers = new_derived_buffers | new_timestep_buffers
        overlap = carry_names & workspace_buffers
        if overlap:
            raise ValueError(
                "State DERIVED_BUFFER/TIMESTEP_BUFFER already declares persistent "
                f"carry; do not also declare it with CARRY: {sorted(overlap)}"
            )
        overlap = state_names & workspace_buffers
        if overlap:
            raise ValueError(
                "State variables cannot also be DERIVED_BUFFER/TIMESTEP_BUFFER "
                "workspaces: "
                f"{sorted(overlap)}"
            )
        overlap = assigned_names & workspace_buffers
        if overlap:
            raise ValueError(
                "State ASSIGNED values cannot also be "
                "DERIVED_BUFFER/TIMESTEP_BUFFER workspaces: "
                f"{sorted(overlap)}"
            )
        overlap = state_names & carry_names
        if overlap:
            raise ValueError(
                "State variables cannot also be persistent CARRY values: "
                f"{sorted(overlap)}"
            )
        overlap = assigned_names & carry_names
        if overlap:
            raise ValueError(
                "State ASSIGNED values cannot also be persistent CARRY values: "
                f"{sorted(overlap)}"
            )
        overlap = state_names & assigned_names
        if overlap:
            raise ValueError(
                f"State variables cannot also be ASSIGNED values: {sorted(overlap)}"
            )
        execution_slots = _parameterized_workspace_slots(cls) | {
            "_name",
            "_sde_rng_names",
            "celsius",
            "diam",
            "dt",
            "key",
            "method",
            "method_kwargs",
            "_solve",
        }
        execution_slots.update(f"{name}_dW_rng" for name in new_state)
        class_slots = {
            name
            for name in workspace_buffers
            if any(name in base.__dict__ for base in cls.__mro__)
        }
        overlap = workspace_buffers & (execution_slots | class_slots)
        if overlap:
            raise ValueError(
                "State DERIVED_BUFFER/TIMESTEP_BUFFER names conflict with "
                "parameters, methods, or reserved execution slots: "
                f"{sorted(overlap)}"
            )
        carry_class_slots = {
            name
            for name in carry_names
            if any(name in base.__dict__ for base in cls.__mro__)
        }
        overlap = carry_names & (execution_slots | carry_class_slots)
        if overlap:
            raise ValueError(
                "State CARRY names conflict with parameters, methods, or "
                f"reserved execution slots: {sorted(overlap)}"
            )
        for method_name, method_kwargs in consume_class_values(
            cls, "state.method", State._method_declarations
        ):
            new_method, new_method_kwargs = _merge_method_config(
                new_method, new_method_kwargs, method_name, method_kwargs
            )

        new_method = _canonical_method_name(new_method)
        if new_method not in _INTEGRATION_BUILDERS:
            valid = ", ".join(valid_integration_methods())
            raise ValueError(
                f"Unknown integration method for State {cls.__name__}: "
                f"{new_method!r}. Valid methods are: {valid}."
            )

        cls._state = tuple(new_state)
        cls._carry = tuple(new_carry)
        cls._carry_specs = new_carry_specs
        cls._derivative = new_derivative
        cls._kinetic = new_kinetic
        cls._diffusion = new_diffusion
        cls._assigned = tuple(new_assigned)
        cls._derived_buffers = new_derived_buffers
        cls._derived_buffer_specs = new_derived_buffer_specs
        cls._timestep_buffers = new_timestep_buffers
        cls._timestep_buffer_shapes = new_timestep_buffer_shapes
        cls._method = new_method
        cls._method_kwargs = dict(new_method_kwargs)

        # Public/introspection aliases.
        cls.method = cls._method
        cls.method_kwargs = dict(cls._method_kwargs)

        set_dt_owner = next(
            (base for base in cls.__mro__ if "set_dt" in base.__dict__), None
        )
        if set_dt_owner is not None:
            raise TypeError(
                "State.set_dt is not a supported lifecycle hook; declare "
                "TIMESTEP_BUFFER values and implement "
                "derive_timestep_buffers(dt) instead."
            )

    def __init__(
        self,
        celsius,
        diameters,
        key,
        shape,
        shape_f,
        additional_parameters=None,
        **kwargs,
    ):
        if not self._state:
            raise ValueError(
                f"State {self.__class__.__name__} has no state variables defined."
                "Use State.STATE(<state vars>) in State implementation to define them."
            )
        super().__init__(
            shape, shape_f, additional_parameters=additional_parameters, **kwargs
        )
        self._name = self.__class__.__name__
        self.key = key

        self.register_buffer("celsius", celsius)
        self.register_buffer("diam", diameters)

        _derivative = list(self._derivative)

        cinfo = None

        if self._kinetic:
            _kinetic = list(self._kinetic)
            deriv_list, _, cinfo, _, _ = kinetic_to_derivatives(self._state, _kinetic)
            _derivative.extend(deriv_list)

        buffer_order = (
            *self._carry,
            *sorted(self._derived_buffers),
            *sorted(self._timestep_buffers),
        )
        self._carry_resolved_shapes = {}
        self._derived_resolved_shapes = {}
        for b in buffer_order:
            if b in self._carry_specs:
                dtype, carry_shape = self._carry_specs[b]
                canonical_shape = (
                    shape_f
                    if carry_shape == "local"
                    else (() if carry_shape == "deferred" else carry_shape)
                )
                value = self.diam.new_zeros(
                    canonical_shape,
                    dtype=self.diam.dtype if dtype is None else dtype,
                )
                self._carry_resolved_shapes[b] = (
                    None if carry_shape == "deferred" else tuple(canonical_shape)
                )
            elif b in self._timestep_buffers:
                # Match the canonical materialized layout before timestep
                # configuration is
                # first called, so a fresh State accepts a strict state_dict
                # produced by an initialized equivalent State.
                value = torch.zeros(
                    _canonical_timestep_buffer_shape(self, b),
                    device=self.diam.device,
                    dtype=self.diam.dtype,
                )
            else:
                dtype, derived_shape = self._derived_buffer_specs.get(
                    b, (None, "local")
                )
                canonical_shape = (
                    shape_f
                    if derived_shape == "local"
                    else (() if derived_shape == "deferred" else derived_shape)
                )
                value = self.diam.new_zeros(
                    canonical_shape,
                    dtype=self.diam.dtype if dtype is None else dtype,
                )
                self._derived_resolved_shapes[b] = (
                    None if derived_shape == "deferred" else tuple(canonical_shape)
                )
            self.register_buffer(b, value)

        # Canonical immutable initialization plans keep declaration discovery
        # outside compiled functional execution.
        self._initial_state_names = tuple(self._state)
        self._initial_carry_names = tuple(self._carry)
        self._has_authored_initial_values = (
            type(self).initial_values is not State.initial_values
        )
        self._has_authored_assigned_values = (
            type(self).assigned_values is not State.assigned_values
        )
        self._has_authored_advance = type(self).advance is not State.advance
        self._assigned_schema_validated = False
        self._advance_schema_validated = False
        self._advance_return_names = ()

        self._timestep_buffer_schema = _timestep_buffer_schema(self)

        # Integration-method configuration is fixed at State-subclass definition
        # time.  Copy it onto the instance for introspection and to keep generated
        # solver compilation monomorphic.  Instance kwargs may still override
        # legacy global options such as ``pade``.
        self.method = self.__class__._method
        self.method_kwargs = dict(self.__class__._method_kwargs)
        if "pade" in kwargs:
            self.method_kwargs["pade"] = kwargs["pade"]
        else:
            self.method_kwargs.setdefault("pade", False)

        ifunc = build_integration_func(
            self._state,
            self._assigned,
            _derivative,
            self.method,
            eliminate=cinfo,
            diffusion=self._diffusion,
            **self.method_kwargs,
        )
        self._solve = MethodType(ifunc, self)

        self._sde_rng_names = ()
        if self.method in {"euler_maruyama", "euler_heun"}:
            self._sde_rng_names = tuple(f"{state}_dW_rng" for state in self._state)
            for rng_name in self._sde_rng_names:
                if not hasattr(self, rng_name):
                    setattr(
                        self,
                        rng_name,
                        RNGModule(None, shape_p=self.shape_p, shape_f=self.shape_f),
                    )

    def _apply(self, fn, recurse=True):
        """Preserve explicitly declared CARRY dtypes across module conversion."""

        result = super()._apply(fn, recurse=recurse)
        # `Module.to(dtype=...)` converts every floating buffer by default, but
        # an explicit CARRY dtype is part of the fixed simulation-state schema.
        buffer_specs = {
            **self._carry_specs,
            **self._derived_buffer_specs,
        }
        for name, (dtype, _shape) in buffer_specs.items():
            if dtype is not None and name in self._buffers:
                value = self._buffers[name]
                if value.dtype != dtype:
                    self._buffers[name] = value.to(dtype=dtype)
        _reset_runtime_output_validation(self)
        return result

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Resolve fresh deferred layouts once from incoming checkpoint tensors."""

        _resolve_deferred_buffers_from_state_dict(self, state_dict, prefix)
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _derive_initial_state_values(
        self,
        v,
        *,
        overrides=None,
        seeds=None,
        require_complete=False,
        return_seeded=False,
    ):
        """Purely derive this module's declared state at local voltage ``v``.

        Effective parameters, temperature, geometry, and DERIVED_BUFFER values
        must already be installed by the caller. Explicit overrides take
        precedence over :meth:`state_defaults`; when every state is overridden,
        ``state_defaults`` is deliberately not evaluated. The returned tensors
        are independent but remain attached to their input autograd graphs.

        ``require_complete=False`` preserves the imperative lifecycle's legacy
        allowance for an entirely empty defaults mapping when an authored hook
        initializes the state. A nonempty inference mapping must contain every
        non-overridden state, as it did before this helper was introduced. Pure
        functional initialization uses the strict form, which also rejects an
        empty incomplete mapping.
        """

        overrides = {} if overrides is None else overrides
        seeds = {} if seeds is None else seeds
        if not isinstance(overrides, Mapping):
            raise TypeError("initial state overrides must be a mapping")
        if not isinstance(seeds, Mapping):
            raise TypeError("initial state seeds must be a mapping")

        state_names = tuple(self._state)
        unknown = set(overrides) - set(state_names)
        if unknown:
            raise KeyError(
                f"initial state overrides for {type(self).__qualname__} contain "
                f"unknown states {sorted(unknown)}"
            )
        unknown_seeds = set(seeds) - set(state_names)
        if unknown_seeds:
            raise KeyError(
                f"initial state seeds for {type(self).__qualname__} contain "
                f"unknown states {sorted(unknown_seeds)}"
            )

        visible_values = {}
        for name, value in (*tuple(seeds.items()), *tuple(overrides.items())):
            value = torch.as_tensor(value, device=v.device, dtype=v.dtype)
            try:
                visible_values[name] = value.expand_as(v)
            except RuntimeError as exc:
                raise ValueError(
                    f"initial input for {type(self).__qualname__}.{name} with "
                    f"shape {tuple(value.shape)} is not broadcastable to local "
                    f"voltage shape {tuple(v.shape)}"
                ) from exc
        visible_values["celsius"] = _support_visible_value(self.celsius, v)
        visible_values["diam"] = self.diam

        missing = tuple(name for name in state_names if name not in overrides)
        if missing:
            inferred = self.state_defaults(v, visible_values)
            if not isinstance(inferred, Mapping):
                raise TypeError(
                    f"{type(self).__qualname__}.state_defaults() must return a mapping"
                )
            unknown_inferred = set(inferred) - set(state_names)
            if unknown_inferred:
                raise KeyError(
                    f"{type(self).__qualname__}.state_defaults() returned "
                    f"undeclared states {sorted(unknown_inferred)}"
                )
        else:
            inferred = {}

        absent = tuple(
            name for name in missing if name not in inferred and name not in seeds
        )
        if absent and (require_complete or inferred):
            raise KeyError(
                f"{type(self).__qualname__}.state_defaults() did not return declared "
                f"states {list(absent)}"
            )

        values = {}
        seeded_only = set()
        for name in state_names:
            if name in overrides:
                value = overrides[name]
            elif name in inferred:
                value = inferred[name]
            elif name in seeds:
                value = seeds[name]
                seeded_only.add(name)
            else:
                continue

            if torch.is_tensor(value):
                value = value.to(device=v.device, dtype=v.dtype)
            else:
                value = torch.as_tensor(value, device=v.device, dtype=v.dtype)
            try:
                value = value.expand_as(v).clone()
            except RuntimeError as exc:
                raise ValueError(
                    f"initial value for {type(self).__qualname__}.{name} with "
                    f"shape {tuple(value.shape)} is not broadcastable to local "
                    f"voltage shape {tuple(v.shape)}"
                ) from exc
            values[name] = value
        if return_seeded:
            return values, frozenset(seeded_only)
        return values

    def _derive_initial_values(self, v, values, *, isolate):
        """Evaluate this State's pure authored initialization overlay."""

        authored = self._has_authored_initial_values
        if not torch.compiler.is_compiling():
            authored = (
                "initial_values" in self.__dict__
                or type(self).initial_values is not State.initial_values
            )
        if not authored:
            return {}

        references = {name: v for name in self._initial_state_names}
        references.update(
            {name: getattr(self, name) for name in self._initial_carry_names}
        )
        outputs = (
            _evaluate_registered_builder(self, "initial_values", v, values)
            if isolate
            else self.initial_values(v, values)
        )
        return _normalize_initial_values(
            self,
            outputs,
            references,
            method_name="initial_values",
        )

    @staticmethod
    def STATE(*args):
        """
        Declare state variables for the State subclass.

        Parameters
        ----------
        *args : str
            Names of state variables advanced by the integrator.
        """
        declare_class_value("state.state", args, State._state_declarations)

    @staticmethod
    def CARRY(*args, dtype=None, shape="local"):
        """
        Declare persistent per-compartment non-ODE state.

        Carry is checkpointed and exposed by functional execution, but is not
        advanced automatically by the declared ODE/SDE solver. Initialize or
        update it through returned values from :meth:`initial_values` or a
        custom :meth:`advance` implementation.

        Parameters
        ----------
        *args : str
            Persistent carry names to allocate.
        dtype : torch.dtype, optional
            Fixed carry dtype. ``None`` follows the State dtype.
        shape : {"local", "deferred", tuple of int}, optional
            ``"local"`` follows the State runtime shape, including explicit
            Population batch axes. ``"deferred"`` lets the first pure
            initialization output establish a custom shape, which is then
            frozen for the lifetime of that State instance. A tuple declares
            structural storage independent of Population batching.
        """
        schema = normalize_buffer_schema(dtype, shape, declaration="CARRY")
        declare_class_value(
            "state.carry",
            {name: schema for name in args},
            State._carry_declarations,
        )

    @staticmethod
    def DERIVED_BUFFER(*args, dtype=None, shape="local"):
        """Declare initialization-static buffers built by ``derive_buffers``.

        Derived buffers retain registered-buffer and ``state_dict``
        registration, but additionally declare that their accepted value is a
        pure function of populated parameters, temperature, and local geometry.
        They are refreshed before State defaults and authored
        :meth:`initial_values` hooks.

        ``derive_buffers()`` must return a mapping containing exactly the
        declared names. It must not mutate module tensors and must not depend on
        voltage, dynamic state, ions/materials, randomness, or timestep.

        Parameters
        ----------
        *args : str
            Names of derived buffers to allocate and materialize.
        dtype : torch.dtype, optional
            Fixed workspace dtype. ``None`` follows the State dtype.
        shape : {"local", "deferred", tuple of int}, optional
            ``"local"`` follows the State runtime shape, including explicit
            Population batch axes.
            ``"deferred"`` lets the first pure builder result establish a
            custom shape, which is then frozen for this instance. A tuple is a
            fixed structural shape.
        """
        schema = normalize_buffer_schema(dtype, shape, declaration="DERIVED_BUFFER")
        declare_class_value(
            "state.derived_buffers",
            {name: schema for name in args},
            State._derived_buffer_declarations,
        )

    @staticmethod
    def TIMESTEP_BUFFER(*args, shape="local"):
        """Declare buffers built by ``derive_timestep_buffers(dt)``.

        Timestep buffers retain registered-buffer and ``state_dict``
        registration, but are rebuilt whenever an integrator configures its
        timestep. The builder must be pure and may depend on the explicit scalar
        ``dt`` argument as well as populated parameters, temperature, local
        geometry, and initialization-static derived buffers. Do not repeat the
        same name in a separate :meth:`CARRY` declaration.

        Parameters
        ----------
        *args : str
            Names of timestep-derived buffers to materialize.
        shape : {"local", tuple of int}, optional
            Canonical registered-buffer shape. ``"local"`` (the default)
            follows the owning State's ``shape_p``. An explicit tuple,
            including ``()`` for a scalar, is structural and remains independent
            of Population batching.
        """
        shape = normalize_timestep_buffer_shape(shape)
        declare_class_value(
            "state.timestep_buffers",
            {name: shape for name in args},
            State._timestep_buffer_declarations,
        )

    @staticmethod
    def DERIVATIVE(*args):
        """
        Declare ODEs for state variables using symbolic strings.

        Parameters
        ----------
        *args : str
            Derivative expressions like ``"m' = (minf - m) / tau"``.
        """
        declare_class_value("state.derivative", args, State._derivative_declarations)

    @staticmethod
    def KINETIC(*args):
        """
        Declare kinetic/Markov schemes between states.

        Parameters
        ----------
        *args : str
            Kinetic expressions like ``"~ a <-> b (alpha, beta)"``.
        """
        declare_class_value("state.kinetic", args, State._kinetic_declarations)

    @staticmethod
    def DIFFUSION(*args):
        """Declare diffusion coefficients for Euler-Maruyama state SDEs.

        Each declaration has the form ``"x = sigma"`` and represents the
        multiplicative noise coefficient. ``State.METHOD("euler_maruyama")``
        interprets this as an Itô SDE. ``State.METHOD("euler_heun")`` interprets
        it as a Stratonovich SDE. Voltage-equation noise is intentionally handled
        by future stochastic voltage/cable integrators.
        """
        declare_class_value("state.diffusion", args, State._diffusion_declarations)

    @staticmethod
    def ASSIGNED(*args):
        """
        Declare computed per-compartment variables used in derivatives.

        State ASSIGNED variables form an explicit contract with
        :meth:`assigned_values`: each declared name must be computed there and
        returned in its mapping. For persistent auxiliary storage
        that does not participate in this return contract, use
        :meth:`CARRY`.

        Parameters
        ----------
        *args : str
            Names of values returned by :meth:`assigned_values`.
        """
        declare_class_value("state.assigned", args, State._assigned_declarations)

    @staticmethod
    def METHOD(method=None, **kwargs):
        """Declare the integration method for this State subclass.

        Parameters
        ----------
        method : str, optional
            Integration method name.  Currently registered methods include
            ``"cnexp"``, ``"derivimplicit"``, ``"bufferimplicit"``,
            ``"linearimplicit"``/``"sparse"``, ``"rosenbrock"``,
            ``"euler_maruyama"`` for Itô State SDEs, and ``"euler_heun"`` for
            Stratonovich State SDEs. Passing ``None`` leaves the inherited method
            unchanged and only updates method kwargs.
        **kwargs
            Method-specific options captured at class-definition time and passed
            to the generated solver builder.  For example::

                State.METHOD("derivimplicit", max_iter=4, line_search=False)

            or::

                State.METHOD("cnexp", pade=True)
        """
        if method is not None:
            method = _canonical_method_name(method)
        declare_class_value(
            "state.method", (method, dict(kwargs)), State._method_declarations
        )

    def assigned_values(self, v, values):
        """Return ephemeral values declared with :meth:`ASSIGNED`.

        The mapping is evaluated immediately before the generated state solver
        needs it and may be evaluated more than once by a multi-stage method.
        Treat ``v`` and ``values`` as read-only and do not update persistent
        carry here.

        """
        return {}

    def _derive_assigned_values(self, v, values):
        """Evaluate ASSIGNED algebra and validate its exact schema once."""
        if not self._assigned and not self._has_authored_assigned_values:
            return {}
        outputs = self.assigned_values(v, values)
        if not self._assigned_schema_validated and not torch.compiler.is_compiling():
            _validate_runtime_outputs(
                self,
                "assigned_values",
                outputs,
                {name: v for name in self._assigned},
                required=self._assigned,
            )
            self._assigned_schema_validated = True
        return outputs

    def advance(self, v, dt, values):
        """Return the next declared state from a read-only current-value frame."""
        if self.method == "euler_heun":
            return self._solve(v, dt, values)
        assigned = self._derive_assigned_values(v, values)
        # A State ASSIGNED name may intentionally shadow a parent Mechanism
        # input with the same spelling. The freshly evaluated State value owns
        # the solver slot; merging first also avoids duplicate-keyword errors.
        return self._solve(dt, **{**values, **assigned})

    def _derive_advance_values(self, v, dt, values):
        """Evaluate State advance and validate custom output structure once."""
        outputs = self.advance(v, dt, values)
        if (
            self._has_authored_advance
            and not self._advance_schema_validated
            and not torch.compiler.is_compiling()
        ):
            references = {name: values[name] for name in self._state}
            references.update({name: getattr(self, name) for name in self._carry})
            _validate_runtime_outputs(
                self,
                "advance",
                outputs,
                references,
                required=self._state,
            )
            self._advance_return_names = tuple(outputs)
            self._advance_schema_validated = True
        return outputs

    def _sde_randn_like(self, state_name: str, like: torch.Tensor) -> torch.Tensor:
        """Detached standard-normal increment for State SDE solvers."""
        rng_name = f"{state_name}_dW_rng"
        rng = getattr(self, rng_name)
        if isinstance(rng, RNGModule):
            rng.init(like.device)
        with torch.no_grad():
            return rng.randn(tuple(like.shape), device=like.device, dtype=like.dtype)

    def init_rng(self):
        super().init_rng()
        for rng_name in getattr(self, "_sde_rng_names", ()):
            rng = getattr(self, rng_name)
            if isinstance(rng, RNGModule):
                rng.init(self._init_device)

    def reset_rng(self):
        super().reset_rng()
        for rng_name in getattr(self, "_sde_rng_names", ()):
            rng = getattr(self, rng_name)
            if isinstance(rng, RNGModule):
                rng.reset()

    def initial_values(self, v, values):
        """Return a pure initialization overlay for this State.

        The optional mapping may contain this State's declared ``STATE`` and
        ``CARRY`` names. ``values`` exposes their current initialized values,
        support-visible ``celsius`` and local ``diam`` inputs, and any
        support-local Ion/Material/current aliases installed by the owning
        Mechanism for this ordered initialization phase.
        Returned values must be Tensors broadcastable to the local voltage
        shape. The hook must not mutate module state or consume implicit RNG.

        """
        return {}

    def derive_buffers(self):
        """Return initialization-static buffers declared by DERIVED_BUFFER."""
        return {}

    def derive_timestep_buffers(self, dt):
        """Return timestep workspaces declared by TIMESTEP_BUFFER."""
        return {}

    def state_defaults(self, v, values):
        """Return lower-priority defaults for declared solver state.

        Used once per State during initialization when at least one declared
        state lacks an insertion-time ``ic`` value. Return a mapping from every
        such state name to a Tensor (or a value broadcastable to the
        support-local voltage shape).

        ``values`` explicitly exposes available shared seeds/overrides plus
        support-visible ``celsius`` and local ``diam``. The hook must be pure,
        deterministic, and complete for every otherwise unresolved State name.

        """
        del values
        return {}

    @classproperty
    def code(cls):
        """
        Returns the source code of the mechanism.
        This is useful for debugging and introspection.
        """
        source_code = inspect.getsource(cls)
        return textwrap.dedent(source_code)
