"""Private building blocks for structured population initialization transforms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

_STATE_REFERENCES = {
    "state.integrator.v",
    "state.clock.t",
    "state.control.duration_remainder",
}

_NN_MODULE_STATE = {
    "training",
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


def _normalize_references(values, *, label: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence of canonical reference strings.")
    references = tuple(values)
    if any(not isinstance(reference, str) for reference in references):
        raise TypeError(f"{label} must contain only canonical reference strings.")
    for reference in references:
        _validate_reference(reference)
    return references


def _validate_reference(reference: str) -> None:
    if not reference:
        raise ValueError("Initialization-transform references must not be empty.")
    if reference.startswith("parameters."):
        parameter_name = reference.removeprefix("parameters.")
        if (
            not parameter_name
            or parameter_name.startswith(".")
            or parameter_name.endswith(".")
        ):
            raise ValueError(
                f"Invalid initialization-transform parameter reference {reference!r}."
            )
        return
    if reference in _STATE_REFERENCES:
        return
    if reference.startswith("state.mechanisms."):
        path = reference.removeprefix("state.mechanisms.")
        if "." in path:
            mechanism_name, state_name = path.rsplit(".", 1)
            if mechanism_name and state_name:
                return
    for prefix in ("state.ions.", "state.materials."):
        if reference.startswith(prefix):
            path = reference.removeprefix(prefix)
            if "." in path:
                owner_name, field_name = path.rsplit(".", 1)
                if owner_name and field_name:
                    return
    raise ValueError(
        f"Unsupported initialization-transform reference {reference!r}. Expected "
        "'parameters.<named_parameters key>', 'state.integrator.v', "
        "'state.clock.t', 'state.control.duration_remainder', or "
        "'state.mechanisms.<mechanism>.<state>', "
        "'state.ions.<ion>.<field>', or "
        "'state.materials.<material>.<field>'."
    )


def _validate_stateless_module(module: torch.nn.Module) -> None:
    if not isinstance(module, torch.nn.Module):
        raise TypeError("initialization transform module must be an nn.Module")

    parameters = tuple(module.named_parameters(remove_duplicate=False))
    buffers = tuple(module.named_buffers(remove_duplicate=False))
    children = tuple(module.named_children())
    if parameters or buffers or children:
        details = []
        if parameters:
            details.append(f"parameters={[name for name, _ in parameters]!r}")
        if buffers:
            details.append(f"buffers={[name for name, _ in buffers]!r}")
        if children:
            details.append(f"submodules={[name for name, _ in children]!r}")
        raise ValueError(
            "initialization transform modules must be stateless; explicit tensor "
            "dependencies belong in inputs (" + ", ".join(details) + ")"
        )

    hook_registries = (
        module._backward_pre_hooks,
        module._backward_hooks,
        module._forward_hooks,
        module._forward_pre_hooks,
    )
    if any(registry for registry in hook_registries):
        raise ValueError(
            "initialization transform modules must not have forward or backward hooks"
        )
    authored_state = tuple(sorted(set(vars(module)) - _NN_MODULE_STATE))
    if authored_state:
        raise ValueError(
            "initialization transform modules must not own hidden Python instance "
            f"state; supply tensor values through inputs instead: {authored_state!r}"
        )


class _InitializationTransformAction(torch.nn.Module):
    """One declarative, stateless initialization tensor transform.

    The supplied authored module owns no tensor state. Framework-owned explicit
    inputs are registered on this action so ordinary ``Module.to`` and
    state-dictionary operations retain their usual semantics.
    """

    def __init__(
        self,
        name: str,
        phase: str,
        module: torch.nn.Module,
        *,
        reads=(),
        writes=(),
        inputs=None,
    ):
        super().__init__()
        if not isinstance(name, str):
            raise TypeError("initialization transform name must be a string")
        if not name:
            raise ValueError("initialization transform name must not be empty")
        if "." in name:
            raise ValueError("initialization transform names must not contain '.'")
        if phase not in {"pre", "post"}:
            raise ValueError("initialization transform phase must be 'pre' or 'post'")

        _validate_stateless_module(module)
        normalized_reads = _normalize_references(reads, label="reads")
        normalized_writes = _normalize_references(writes, label="writes")
        if not normalized_writes:
            raise ValueError(
                "initialization transforms must declare at least one write"
            )
        if len(normalized_writes) != len(set(normalized_writes)):
            raise ValueError("initialization transform writes must be unique")

        if inputs is None:
            inputs = {}
        if not isinstance(inputs, Mapping):
            raise TypeError("initialization transform inputs must be a mapping")
        input_items = tuple(inputs.items())
        for input_name, value in input_items:
            if not isinstance(input_name, str) or not input_name:
                raise TypeError(
                    "initialization transform input names must be non-empty strings"
                )
            if not torch.is_tensor(value):
                raise TypeError(
                    f"initialization transform input {input_name!r} must be a Tensor"
                )

        self._action_name = name
        self._phase = phase
        self._reads = normalized_reads
        self._writes = normalized_writes
        self._input_names = tuple(name for name, _ in input_items)
        self.transform = module
        # Initialization transforms have one fixed inference interpretation;
        # Population.train()/eval() must not introduce a hidden input.
        self.transform.eval()
        for index, (_input_name, value) in enumerate(input_items):
            self.register_buffer(
                f"_input_{index}",
                value.detach().clone(memory_format=torch.preserve_format),
            )

    @property
    def name(self) -> str:
        return self._action_name

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def reads(self) -> tuple[str, ...]:
        return self._reads

    @property
    def writes(self) -> tuple[str, ...]:
        return self._writes

    @property
    def input_names(self) -> tuple[str, ...]:
        return self._input_names

    def input_values(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            self._buffers[f"_input_{index}"] for index in range(len(self._input_names))
        )

    def train(self, mode: bool = True):
        """Keep the authored transform in its fixed evaluation interpretation."""
        super().train(mode)
        self.transform.eval()
        return self

    def forward(self, *values):
        expected = len(self.reads) + len(self.input_names)
        if len(values) != expected:
            raise TypeError(
                f"Initialization transform {self.name!r} expected {expected} tensor "
                f"arguments ({len(self.reads)} reads and {len(self.input_names)} "
                f"inputs), got {len(values)}."
            )
        return self.transform(*values)


class _InitializationTransformHook:
    """Callable hook-list adapter for one structured transform action."""

    __slots__ = ("action_name", "phase")

    def __init__(self, action_name: str, phase: str):
        self.action_name = action_name
        self.phase = phase

    def __call__(self, population):
        return population._execute_initialization_transform(
            self.action_name,
            phase=self.phase,
        )
