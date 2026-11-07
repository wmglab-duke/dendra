"""Conservative batching of repeated generated State numerical programs.

Plans contain existing module owners and static metadata only. Sampling,
initialization, shared-field publication and checkpoint ownership stay with
the ordinary handler. Unknown programs retain their original advance calls.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import cache
from types import BuiltinFunctionType, FunctionType, MethodType, SimpleNamespace

import torch

from . import _cnexp, _solvers
from ._mechanism import Mechanism
from ._state import State

_SOLVER_GROUPING_ENABLED = True
_ROSENBROCK_PRELUDE = {
    ast.dump(statement)
    for statement in ast.parse(
        "dt = torch.as_tensor(dt, dtype=__ref.dtype, device=__ref.device)\n"
        "if dt.ndim == 0 and __ref.ndim > 0:\n"
        "    dt = dt.reshape((1,) * __ref.ndim)\n"
    ).body
}
_ELEMENTAL = frozenset(
    {
        "abs",
        "exp",
        "expm1",
        "log",
        "log1p",
        "log2",
        "log10",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "tanh",
        "sinh",
        "cosh",
        "sigmoid",
        "where",
        "zeros_like",
        "ones_like",
    }
)
_HELPERS = {
    "_linear_exact_step": (_cnexp, _cnexp._linear_exact_step),
    "_solve_linear_small": (_solvers, _solvers._solve_linear_small),
}
_HELPER_CODES = {name: function.__code__ for name, (_, function) in _HELPERS.items()}
_HOOKS = (
    "_forward_hooks",
    "_forward_pre_hooks",
    "_backward_hooks",
    "_backward_pre_hooks",
)


def _plain(value):
    return (
        type(value) in (torch.Tensor, torch.nn.Parameter)
        and value.layout == torch.strided
        and value.device.type != "meta"
        and not value.is_quantized
        and not getattr(value, "_backward_hooks", None)
    )


def _metadata(value):
    return (tuple(value.shape), value.dtype, value.device)


def _state_metadata(value):
    return (*_metadata(value), tuple(value.stride()))


def _field_metadata(value, rank):
    shape, dtype, device = _metadata(value)
    return ((1,) * (rank - len(shape)) + shape, dtype, device)


@cache
def _trace_mode_contracts():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import (
        PreDispatchTorchFunctionMode,
        ProxyTorchDispatchMode,
        TorchFunctionMetadataMode,
    )

    result = {}
    for kind, method in (
        (FakeTensorMode, "__torch_dispatch__"),
        (ProxyTorchDispatchMode, "__torch_dispatch__"),
        (PreDispatchTorchFunctionMode, "__torch_function__"),
        (TorchFunctionMetadataMode, "__torch_function__"),
    ):
        function = getattr(kind, method)
        chain = []
        while type(function) is FunctionType:
            chain.append((function, function.__code__))
            wrapped = getattr(function, "__wrapped__", None)
            if wrapped is None:
                break
            function = wrapped
        if not chain or not chain[-1][1].co_filename.replace("\\", "/").endswith(
            kind.__module__.replace(".", "/") + ".py"
        ):
            return None
        result[kind] = (method, tuple(chain))
    return result


def _solver_grouping_runtime_plain():
    """Batch inference only; keep per-member AD dependency graphs separate."""
    if torch.compiler.is_compiling() or torch.is_grad_enabled():
        return False
    # no_grad does not disable forward AD. Packing would also turn absent
    # tangents into zero tangents for unrelated members, so transforms retain
    # the ordinary per-member calls. Missing introspection fails closed.
    try:
        level = getattr(torch.autograd.forward_ad, "_current_level", None)
        transforms_active = getattr(torch._C, "_are_functorch_transforms_active", None)
        if (
            type(level) is not int
            or level >= 0
            or type(transforms_active) is not BuiltinFunctionType
            or transforms_active()
        ):
            return False
    except Exception:
        return False
    from ..integrators import core

    if core._population_runtime_modes_plain(check_saved_tensors=False):
        return True
    try:
        contracts = _trace_mode_contracts()
        if contracts is None:
            return False
        dispatch = [
            torch._C._get_dispatch_stack_at(index)
            for index in range(torch._C._len_torch_dispatch_stack())
        ]
        pre_dispatch = [
            torch._ops._get_dispatch_mode_pre_dispatch(key)
            for key in (
                torch._C._TorchDispatchModeKey.PROXY,
                torch._C._TorchDispatchModeKey.FUNCTIONAL,
            )
        ]
        pre_dispatch = [mode for mode in pre_dispatch if mode is not None]
        if len(pre_dispatch) != torch._ops._len_torch_dispatch_stack_pre_dispatch():
            return False
        functions = [
            torch._C._get_function_stack_at(index)
            for index in range(torch._C._len_torch_function_stack())
        ]
        for mode in (*dispatch, *pre_dispatch, *functions):
            kind = type(mode)
            if kind.__getattribute__ is not object.__getattribute__:
                return False
            namespace = vars(mode)
            if kind is core._POPULATION_DEVICE_CONTEXT:
                if (
                    kind.__torch_function__
                    is not core._POPULATION_DEVICE_TORCH_FUNCTION
                    or getattr(kind, "__getattr__", None) is not None
                    or "__torch_function__" in namespace
                    or type(namespace.get("device")) is not torch.device
                ):
                    return False
                continue
            if kind not in contracts:
                return False
            method, chain = contracts[kind]
            if (
                method in namespace
                or getattr(kind, method) is not chain[0][0]
                or any(function.__code__ is not code for function, code in chain)
            ):
                return False
        return True
    except Exception:
        return False


def _options(values):
    if type(values) is dict and all(type(name) is str for name in values):
        result = tuple(sorted(values.items()))
        if all(
            type(value) in (str, int, float, bool, type(None)) for _, value in result
        ):
            return result
    return None


def _generated_program(state):
    method = state.method
    if method not in {"cnexp", "rosenbrock"}:
        return None
    bound = state._solve
    if type(bound) is not MethodType or bound.__self__ is not state:
        return None
    function = bound.__func__
    source = getattr(function, "__source__", None)
    filename = getattr(function, "__generated_filename__", "")
    if (
        type(function) is not FunctionType
        or type(source) is not str
        or not filename.startswith(f"<dendra.{method}.solve:")
        or function.__code__.co_filename != filename
        or function.__code__.co_freevars
        or function.__defaults__ is not None
        or function.__kwdefaults__ is not None
    ):
        return None
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return None
    definition = tree.body[0]
    compiled = compile(source, filename, "exec", dont_inherit=True)
    expected = next(
        value for value in compiled.co_consts if isinstance(value, type(compiled))
    )
    actual = function.__code__
    if any(
        getattr(actual, name) != getattr(expected, name)
        for name in ("co_code", "co_consts", "co_names", "co_varnames", "co_flags")
    ):
        return None
    inputs = tuple(arg.arg for arg in definition.args.args[2:])
    if set(inputs) != set(state._state) | set(state._assigned):
        return None
    self_fields, globals_used = set(), {}
    if method == "rosenbrock":
        if function.__globals__.get("torch") is not torch:
            return None
        globals_used[("torch", "as_tensor")] = torch.as_tensor
    prelude_nodes = {
        id(node)
        for statement in definition.body
        if method == "rosenbrock" and ast.dump(statement) in _ROSENBROCK_PRELUDE
        for node in ast.walk(statement)
    }
    for node in ast.walk(definition):
        if id(node) in prelude_nodes:
            continue
        if isinstance(
            node,
            (ast.For, ast.While, ast.Lambda, ast.ListComp, ast.DictComp, ast.Subscript),
        ):
            return None
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                if not isinstance(node.ctx, ast.Load):
                    return None
                self_fields.add(node.attr)
            elif isinstance(node.value, ast.Name) and node.value.id == "torch":
                if function.__globals__.get("torch") is not torch or not hasattr(
                    torch, node.attr
                ):
                    return None
                globals_used[("torch", node.attr)] = getattr(torch, node.attr)
            else:
                return None
        if not isinstance(node, ast.Call):
            continue
        call = node.func
        if isinstance(call, ast.Name) and call.id in _HELPERS:
            module, helper = _HELPERS[call.id]
            if (
                function.__globals__.get(call.id) is not helper
                or getattr(module, call.id) is not helper
                or helper.__code__ is not _HELPER_CODES[call.id]
            ):
                return None
            globals_used[("helper", call.id)] = helper
        elif isinstance(call, ast.Attribute) and isinstance(call.value, ast.Name):
            if call.value.id == "torch":
                if call.attr not in _ELEMENTAL:
                    return None
            else:
                return None
        else:
            return None
    return _Program(
        method,
        source,
        function,
        inputs,
        tuple(sorted(self_fields)),
        tuple(sorted(globals_used.items())),
    )


@dataclass(frozen=True)
class _Program:
    method: str
    source: str
    function: object
    inputs: tuple[str, ...]
    self_fields: tuple[str, ...]
    globals_used: tuple

    def key(self):
        return (
            self.method,
            self.source,
            self.inputs,
            self.self_fields,
            tuple((name, id(value)) for name, value in self.globals_used),
        )

    def current(self, state):
        bound = state._solve
        if (
            type(bound) is not MethodType
            or bound.__self__ is not state
            or bound.__func__ is not self.function
        ):
            return False
        for (kind, name), value in self.globals_used:
            if kind == "torch":
                if (
                    self.function.__globals__.get("torch") is not torch
                    or getattr(torch, name) is not value
                ):
                    return False
            else:
                module, _ = _HELPERS[name]
                if (
                    self.function.__globals__.get(name) is not value
                    or getattr(module, name) is not value
                    or value.__code__ is not _HELPER_CODES[name]
                ):
                    return False
        return True


def _candidate(mechanism, support_index):
    if (
        len(mechanism.DE) != 1
        or mechanism._has_authored_advance
        or mechanism._has_authored_assigned_values
        or mechanism._carry
        or mechanism._delay_history_buffer_names
        or mechanism._injection_specs
        or mechanism._delayed_state_specs
        or type(mechanism).advance is not Mechanism.advance
        or type(mechanism).assigned_values is not Mechanism.assigned_values
        or type(mechanism)._runtime_value_frame is not Mechanism._runtime_value_frame
        or any(getattr(mechanism, name, None) for name in _HOOKS)
    ):
        return None
    advance = mechanism._advance_states
    if not getattr(
        getattr(advance, "__func__", None), "_dendra_monomorphic_advance_states", False
    ):
        return None
    state = next(iter(mechanism.DE.values()))
    if (
        type(state).advance is not State.advance
        or type(state)._derive_assigned_values is not State._derive_assigned_values
        or state._carry
        or state._inputs
        or state._diffusion
        or state._modules
        or any(isinstance(value, torch.nn.Module) for value in vars(state).values())
        or any(getattr(state, name, None) for name in _HOOKS)
    ):
        return None
    options = _options(state.method_kwargs)
    if options is None:
        return None
    program = _generated_program(state)
    if program is None:
        return None
    states = tuple(getattr(mechanism, name) for name in state._state)
    if not states or not all(_plain(value) for value in states):
        return None
    metadata = _state_metadata(states[0])
    if metadata[1] not in (torch.float32, torch.float64):
        return None
    if any(_state_metadata(value) != metadata for value in states):
        return None
    fields = []
    for name in program.self_fields:
        if name not in state._buffers and name not in state._parameters:
            return None
        value = getattr(state, name)
        if (
            not _plain(value)
            or value.dtype != metadata[1]
            or value.device != metadata[2]
        ):
            return None
        if torch.broadcast_shapes(value.shape, metadata[0]) != metadata[0]:
            return None
        fields.append(_field_metadata(value, len(metadata[0])))
    key = (
        support_index,
        program.key(),
        options,
        tuple(state._state),
        tuple(state._assigned),
        metadata,
        tuple(fields),
    )
    return key, state, program, metadata, tuple(fields)


@dataclass(frozen=True)
class _SolverGroup:
    members: tuple
    states: tuple
    program: _Program
    programs: tuple[_Program, ...]
    metadata: tuple
    field_metadata: tuple

    def _solve(self, proxy, dt, packed):
        return self.program.function(proxy, dt, **packed)

    def advance(self, local_voltages, dt):
        def ordinary():
            for mechanism, index in self.members:
                mechanism._advance_states(local_voltages[index], dt)

        if (
            not _plain(dt)
            or dt.ndim != 0
            or dt.dtype != self.metadata[1]
            or dt.device != self.metadata[2]
        ):
            return ordinary()
        inputs, fields = [], []
        for (mechanism, index), state, program in zip(
            self.members, self.states, self.programs, strict=True
        ):
            if not program.current(state) or any(
                not _plain(value := getattr(mechanism, name))
                or _state_metadata(value) != self.metadata
                for name in state._state
            ):
                return ordinary()
            values = tuple(getattr(state, name) for name in self.program.self_fields)
            if any(
                not _plain(value)
                or _field_metadata(value, len(self.metadata[0])) != expected
                for value, expected in zip(values, self.field_metadata, strict=True)
            ):
                return ordinary()
            fields.append(values)
        # Custom ASSIGNED hooks retain their original visible shape, input
        # identities and call order. A late mismatch reuses these cached frames.
        for (mechanism, index), state in zip(self.members, self.states, strict=True):
            frame = mechanism._runtime_value_frame()
            assigned = state._derive_assigned_values(local_voltages[index], frame)
            inputs.append({**frame, **assigned})
        compatible = all(
            name in frame
            and _plain(frame[name])
            and _metadata(frame[name]) == self.metadata[:3]
            for frame in inputs
            for name in self.program.inputs
        )
        if compatible:
            packed = {
                name: torch.stack([frame[name] for frame in inputs], dim=0)
                for name in self.program.inputs
            }
            proxy = SimpleNamespace(
                **{
                    name: torch.stack(
                        [values[index].reshape(expected[0]) for values in fields],
                        dim=0,
                    )
                    for index, (name, expected) in enumerate(
                        zip(self.program.self_fields, self.field_metadata, strict=True)
                    )
                }
            )
            outputs = self._solve(proxy, dt, packed)
            results = [
                {name: outputs[name][index].clone() for name in state._state}
                for index, state in enumerate(self.states)
            ]
        else:
            results = [
                state._solve(dt, **frame)
                for state, frame in zip(self.states, inputs, strict=True)
            ]
        for (mechanism, _), state, result in zip(
            self.members, self.states, results, strict=True
        ):
            for name in state._state:
                mechanism._buffers[name] = result[name]


def _blocked_mutable_states(handler):
    """Reject overlaps; irregular views conservatively cover their storage."""
    blocked, seen, spans = set(), set(), []
    for mechanism in handler.mechanisms.values():
        for state in mechanism.DE.values():
            for name in state._state:
                value = getattr(mechanism, name)
                if not _plain(value):
                    return None
                identity = id(value)
                if identity in seen:
                    blocked.add(identity)
                seen.add(identity)
                storage = value.untyped_storage()
                key = (value.device.type, value.device.index or -1, storage.data_ptr())
                if value.is_contiguous():
                    start = value.storage_offset() * value.element_size()
                    stop = start + value.numel() * value.element_size()
                else:
                    blocked.add(identity)
                    start, stop = 0, storage.nbytes()
                if stop > start:
                    spans.append((key, start, stop, identity))
    previous, end, owner = None, 0, None
    for key, start, stop, identity in sorted(spans):
        if key != previous:
            previous, end, owner = key, stop, identity
        else:
            if start < end:
                blocked.update((identity, owner))
            if stop > end:
                end, owner = stop, identity
    return blocked


def _install_solver_grouping(population):
    """Describe contiguous compatible calls on an isolated execution handler."""
    handler = population.mech
    plans = {}
    from ..integrators.core import _population_runtime_modes_plain

    enabled = _SOLVER_GROUPING_ENABLED and _population_runtime_modes_plain()
    if enabled:
        _trace_mode_contracts()
    try:
        blocked = _blocked_mutable_states(handler) if enabled else None
    except Exception:
        blocked = None
    enabled = enabled and blocked is not None
    for phase, attribute in (
        ("all", "_state_advance_plan"),
        ("pre", "_pre_state_advance_plan"),
        ("post", "_post_state_advance_plan"),
    ):
        actions, pending, candidate = [], [], None

        def flush():
            if len(pending) < 2:
                actions.extend(pending)
            else:
                actions.append(
                    _SolverGroup(
                        members=tuple(pending),
                        states=tuple(item[1] for item in details),
                        program=candidate[2],
                        programs=tuple(item[2] for item in details),
                        metadata=candidate[3],
                        field_metadata=candidate[4],
                    )
                )
            pending.clear()
            details.clear()

        details = []
        for mechanism, index in getattr(handler, attribute, ()):
            selected = None
            if enabled:
                try:
                    selected = _candidate(mechanism, index)
                    if selected is not None and any(
                        id(getattr(mechanism, name)) in blocked
                        for name in selected[1]._state
                    ):
                        selected = None
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    SyntaxError,
                    StopIteration,
                ):
                    selected = None
            if selected is None or candidate is None or selected[0] != candidate[0]:
                flush()
                candidate = selected
            if selected is None:
                actions.append((mechanism, index))
            else:
                pending.append((mechanism, index))
                details.append(selected)
        flush()
        plans[phase] = tuple(actions)
    if any(
        isinstance(action, _SolverGroup)
        for actions in plans.values()
        for action in actions
    ):
        handler._solver_grouping_plans = plans
    else:
        handler.__dict__.pop("_solver_grouping_plans", None)


__all__ = []
