"""Runtime contracts for explicitly numerically differentiated currents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


class NumericalCurrentContractError(RuntimeError):
    """Raised when a runtime probe disproves a ``NUMERICAL`` declaration.

    ``Mechanism.NUMERICAL`` is valid only for currents returning a tensor with
    the voltage's shape, dtype, and device that are pointwise in voltage,
    deterministic, non-mutating with respect to the voltage input, and free of
    registered-state side effects. Dendra checks those properties with bounded
    probes on the first eager evaluation for each input signature. Passing the
    probes is useful evidence, not a proof for unprobed or state-dependent
    behavior.
    """


def _same_tensor_values(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return exact equality while treating colocated NaNs as equal."""
    if (
        left.shape != right.shape
        or left.dtype != right.dtype
        or left.device != right.device
    ):
        return False
    if left.is_floating_point() or left.is_complex():
        same = torch.eq(left, right) | (torch.isnan(left) & torch.isnan(right))
        return bool(torch.all(same).item())
    return torch.equal(left, right)


@dataclass(frozen=True)
class _TensorSnapshot:
    """Tensor value and view metadata needed for transactional restoration."""

    tensor: torch.Tensor
    value: torch.Tensor
    storage: object
    shape: torch.Size
    stride: tuple[int, ...]
    storage_offset: int


def _tensor_snapshot(tensor: torch.Tensor) -> _TensorSnapshot:
    storage = tensor.untyped_storage()
    return _TensorSnapshot(
        tensor=tensor,
        value=tensor.detach().clone(),
        storage=storage,
        shape=tensor.shape,
        stride=tensor.stride(),
        storage_offset=tensor.storage_offset(),
    )


def _tensor_mutations(snapshot: _TensorSnapshot) -> list[str]:
    """Describe in-place value, view, or storage changes to one tensor."""
    tensor = snapshot.tensor
    mutations = []
    if tensor.shape != snapshot.shape:
        mutations.append(
            f"shape changed from {tuple(snapshot.shape)!r} to {tuple(tensor.shape)!r}"
        )
    if tensor.stride() != snapshot.stride:
        mutations.append(
            f"stride changed from {snapshot.stride!r} to {tensor.stride()!r}"
        )
    if tensor.storage_offset() != snapshot.storage_offset:
        mutations.append(
            "storage offset changed from "
            f"{snapshot.storage_offset} to {tensor.storage_offset()}"
        )
    if tensor.untyped_storage() is not snapshot.storage:
        mutations.append("underlying storage was replaced")
    if (
        tensor.shape == snapshot.shape
        and tensor.dtype == snapshot.value.dtype
        and tensor.device == snapshot.value.device
        and not _same_tensor_values(tensor, snapshot.value)
    ):
        mutations.append("values changed")
    return mutations


def _restore_tensor(snapshot: _TensorSnapshot) -> None:
    """Restore a tensor's original storage, view metadata, and values."""
    with torch.no_grad():
        snapshot.tensor.set_(
            snapshot.storage,
            snapshot.storage_offset,
            snapshot.shape,
            snapshot.stride,
        )
        snapshot.tensor.copy_(snapshot.value)


def _registered_tensor_snapshot(module: torch.nn.Module):
    """Snapshot registered tensors without traversing the state-dict API."""
    snapshot = []
    for module_name, child in module.named_modules():
        for registry_name in ("_buffers", "_parameters"):
            registry = getattr(child, registry_name)
            values = {
                name: (
                    tensor,
                    None if tensor is None else _tensor_snapshot(tensor),
                )
                for name, tensor in registry.items()
            }
            snapshot.append((module_name, child, registry_name, values))
    return snapshot


def _registered_tensor_mutations(snapshot) -> list[str]:
    """Describe registered tensor mutations relative to ``snapshot``."""
    mutations = []
    for module_name, module, registry_name, expected in snapshot:
        live = getattr(module, registry_name)
        kind = "buffer" if registry_name == "_buffers" else "parameter"
        prefix = f"{module_name}." if module_name else ""
        expected_names = tuple(expected)
        live_names = tuple(live)
        if live_names != expected_names:
            mutations.append(
                f"registered {kind} names changed from {expected_names!r} "
                f"to {live_names!r}"
            )
        for name, (original, saved_state) in expected.items():
            qualified = f"{prefix}{name}"
            if name not in live:
                mutations.append(f"registered {kind} {qualified!r} was removed")
                continue
            current = live[name]
            if current is not original:
                mutations.append(f"registered {kind} {qualified!r} was replaced")
                continue
            if current is not None and (changes := _tensor_mutations(saved_state)):
                mutations.append(
                    f"registered {kind} {qualified!r} was modified in place "
                    f"({', '.join(changes)})"
                )
    return mutations


def _restore_registered_tensors(snapshot) -> list[str]:
    """Restore registry membership, object identity, and tensor values."""
    failures = []
    for module_name, module, registry_name, expected in snapshot:
        try:
            registry = getattr(module, registry_name)
            registry.clear()
            registry.update(
                {name: original for name, (original, _) in expected.items()}
            )
        except Exception as cause:  # pragma: no cover - defensive corruption path
            prefix = f"{module_name}." if module_name else ""
            failures.append(
                f"could not restore {prefix}{registry_name}: "
                f"{type(cause).__name__}: {cause}"
            )
    for module_name, _, registry_name, expected in snapshot:
        kind = "buffer" if registry_name == "_buffers" else "parameter"
        prefix = f"{module_name}." if module_name else ""
        for name, (original, saved_state) in expected.items():
            if original is None:
                continue
            try:
                _restore_tensor(saved_state)
            except Exception as cause:  # pragma: no cover - defensive corruption path
                failures.append(
                    f"could not restore registered {kind} {prefix}{name!r}: "
                    f"{type(cause).__name__}: {cause}"
                )
    return failures


def _module_rng_snapshot(module: torch.nn.Module):
    """Capture Dendra-style per-module RNG streams used by current methods."""
    snapshot = []
    for child in module.modules():
        get_state = getattr(child, "rng_state", None)
        set_state = getattr(child, "set_rng_state", None)
        if callable(get_state) and callable(set_state):
            snapshot.append((set_state, get_state()))
    return snapshot


def _restore_module_rngs(snapshot) -> None:
    for set_state, state in snapshot:
        set_state(state)


def _raise_contract_violation(mechanism, current: str, detail: str, *, cause=None):
    mechanism_name = type(mechanism).__qualname__
    error = NumericalCurrentContractError(
        f"Runtime validation disproved Mechanism.NUMERICAL({current!r}) for "
        f"current {current!r} on {mechanism_name}: {detail}. Numerical currents "
        "must return a tensor with exactly the voltage shape and must be "
        "deterministic, side-effect free, and pointwise in voltage. Use a "
        "supported affine expression or, for a pointwise current, provide an "
        f"exact {current}_with_conductance(self, v) implementation instead. "
        "Genuinely coupled voltage dependence must be modeled outside local "
        "mechanism-current assembly. Passing "
        "the bounded runtime probes is not proof for unprobed or "
        "state-dependent behavior."
    )
    if cause is None:
        raise error
    raise error from cause


def _validate_output_shape(mechanism, current: str, voltage, output, stage: str):
    if not torch.is_tensor(output):
        _raise_contract_violation(
            mechanism,
            current,
            f"the {stage} call returned {type(output).__name__}, not a tensor",
        )
    if output.shape != voltage.shape:
        _raise_contract_violation(
            mechanism,
            current,
            f"the {stage} call returned shape {tuple(output.shape)!r}, but the "
            f"voltage shape is {tuple(voltage.shape)!r}",
        )
    if output.device != voltage.device:
        _raise_contract_violation(
            mechanism,
            current,
            f"the {stage} call returned a tensor on {output.device}, but voltage "
            f"is on {voltage.device}",
        )
    if output.dtype != voltage.dtype:
        _raise_contract_violation(
            mechanism,
            current,
            f"the {stage} call returned dtype {output.dtype}, but voltage has "
            f"dtype {voltage.dtype}",
        )
    return output.detach().clone()


def _invoke_probe(
    mechanism,
    current: str,
    function: Callable,
    voltage: torch.Tensor,
    stage: str,
    tensor_snapshot,
):
    voltage_snapshot = _tensor_snapshot(voltage)
    try:
        output = function(voltage)
    except Exception as cause:
        mutations = _registered_tensor_mutations(tensor_snapshot)
        voltage_mutations = _tensor_mutations(voltage_snapshot)
        if voltage_mutations:
            mutations.insert(
                0,
                f"voltage input was modified in place ({', '.join(voltage_mutations)})",
            )
        if mutations:
            restoration_failures = []
            if voltage_mutations:
                try:
                    _restore_tensor(voltage_snapshot)
                except Exception as restore_cause:  # pragma: no cover - defensive
                    restoration_failures.append(
                        "could not restore voltage input: "
                        f"{type(restore_cause).__name__}: {restore_cause}"
                    )
            restoration_failures.extend(_restore_registered_tensors(tensor_snapshot))
            detail = f"the {stage} call mutated " + "; ".join(mutations)
            if restoration_failures:
                detail += "; transactional restoration failed: " + "; ".join(
                    restoration_failures
                )
            _raise_contract_violation(
                mechanism,
                current,
                detail,
                cause=cause,
            )
        raise

    mutations = _registered_tensor_mutations(tensor_snapshot)
    voltage_mutations = _tensor_mutations(voltage_snapshot)
    if voltage_mutations:
        mutations.insert(
            0,
            f"voltage input was modified in place ({', '.join(voltage_mutations)})",
        )
    if mutations:
        restoration_failures = []
        if voltage_mutations:
            try:
                _restore_tensor(voltage_snapshot)
            except Exception as cause:  # pragma: no cover - defensive corruption path
                restoration_failures.append(
                    f"could not restore voltage input: {type(cause).__name__}: {cause}"
                )
        restoration_failures.extend(_restore_registered_tensors(tensor_snapshot))
        detail = f"the {stage} call mutated " + "; ".join(mutations)
        if restoration_failures:
            detail += "; transactional restoration failed: " + "; ".join(
                restoration_failures
            )
        _raise_contract_violation(
            mechanism,
            current,
            detail,
        )
    return _validate_output_shape(mechanism, current, voltage, output, stage)


def _well_spread_probe_indices(size: int) -> tuple[int, ...]:
    if size <= 1:
        return ()
    candidates = (0, size // 3, (2 * size) // 3, size - 1)
    return tuple(dict.fromkeys(candidates))


def _current_identity(function: Callable) -> tuple[int, int | None]:
    unbound = getattr(function, "__func__", function)
    code = getattr(unbound, "__code__", None)
    return id(unbound), None if code is None else id(code)


def validate_declared_numerical_current(
    mechanism, current: str, voltage: torch.Tensor
) -> None:
    """Run bounded, transactional checks for one explicit numerical current.

    Validation is cached per current implementation, input device, dtype, and
    shape.  It performs a constant number of current evaluations and never
    constructs a full Jacobian.  Four well-spread voltage coordinates are
    perturbed independently; any changed output outside the perturbed coordinate
    demonstrates forbidden tensor coupling.
    """
    function = getattr(mechanism, current)
    signature = (
        current,
        _current_identity(function),
        voltage.device.type,
        voltage.device.index,
        voltage.dtype,
        tuple(voltage.shape),
    )
    cache = mechanism.__dict__.setdefault("_dendra_validated_numerical_currents", set())
    if signature in cache:
        return

    probe_voltage = voltage.detach().clone().contiguous()
    tensor_snapshot = _registered_tensor_snapshot(mechanism)
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = None
    if voltage.device.type == "cuda":
        cuda_rng_state = torch.cuda.get_rng_state(voltage.device)
    module_rng_states = _module_rng_snapshot(mechanism)

    try:
        with torch.no_grad():
            baseline = _invoke_probe(
                mechanism,
                current,
                function,
                probe_voltage,
                "first repeated-input",
                tensor_snapshot,
            )
            repeated = _invoke_probe(
                mechanism,
                current,
                function,
                probe_voltage,
                "second repeated-input",
                tensor_snapshot,
            )
            if not _same_tensor_values(baseline, repeated):
                _raise_contract_violation(
                    mechanism,
                    current,
                    "repeated calls with the same voltage returned different values, "
                    "demonstrating nondeterminism",
                )

            flat_baseline = baseline.reshape(-1)
            for index in _well_spread_probe_indices(probe_voltage.numel()):
                flat_voltage = probe_voltage.reshape(-1).clone()
                value = flat_voltage[index]
                rel_step = torch.finfo(probe_voltage.dtype).eps ** (1.0 / 3.0)
                step = rel_step * torch.maximum(
                    torch.abs(value), torch.ones_like(value)
                )
                flat_voltage[index] = value + step
                perturbed_voltage = flat_voltage.reshape(probe_voltage.shape)
                perturbed = _invoke_probe(
                    mechanism,
                    current,
                    function,
                    perturbed_voltage,
                    f"locality probe at flattened voltage index {index}",
                    tensor_snapshot,
                ).reshape(-1)
                outside = torch.ones(
                    probe_voltage.numel(),
                    dtype=torch.bool,
                    device=probe_voltage.device,
                )
                outside[index] = False
                if not _same_tensor_values(flat_baseline[outside], perturbed[outside]):
                    _raise_contract_violation(
                        mechanism,
                        current,
                        f"perturbing flattened voltage index {index} changed "
                        "current values at other indices, demonstrating tensor "
                        "coupling",
                    )
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, voltage.device)
        _restore_module_rngs(module_rng_states)

    cache.add(signature)
