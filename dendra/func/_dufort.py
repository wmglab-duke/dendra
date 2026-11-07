"""Tensor-only dispatch for the two native Dufort current phases."""

from __future__ import annotations

import warnings

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._python_dispatch import TorchDispatchMode

from ._dufort_autograd import capture_autograd_program


class _CurrentAutogradPolicy(TorchDispatchMode):
    """Observe AD policy changes while establishing a pure current schema."""

    supports_higher_order_operators = True

    def __init__(self):
        self.detached_arithmetic = False

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if (
            not torch.is_grad_enabled() or function is torch.ops.aten.detach.default
        ) and any(
            torch.is_tensor(value) and value.requires_grad
            for value in torch.utils._pytree.tree_leaves((args, kwargs))
        ):
            # Function.forward, explicit no_grad, and nondifferentiable identity
            # outputs can carry AD contracts absent from primal ATen capture.
            self.detached_arithmetic = True
        return function(*args, **(kwargs or {}))


def _contains_custom_autograd(values):
    pending = [
        value.grad_fn
        for value in torch.utils._pytree.tree_leaves(values)
        if torch.is_tensor(value) and value.grad_fn is not None
    ]
    seen = set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        if isinstance(node, torch.autograd.function.BackwardCFunction):
            return True
        pending.extend(parent for parent, _index in node.next_functions if parent)
    return False


@torch.compiler.assume_constant_result
def _has_empty_transform_axis(value):
    """Inspect transform shape metadata without reading numerical data."""
    functorch = torch._C._functorch
    while functorch.is_functorch_wrapped_tensor(value):
        batched = functorch.is_batchedtensor(value)
        axis = functorch.maybe_get_bdim(value)
        value = functorch.get_unwrapped(value)
        if batched and value.shape[axis] == 0:
            return True
    return False


def _capture_autograd_program(evaluate, arguments):
    """Suppress AOT's anomaly warning only for an unsupported native VJP."""
    expected_failure = False
    recorded = []
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.filterwarnings(
                "always",
                message=r"Error detected in .*Backward",
                category=UserWarning,
                module=r"torch\.autograd\.graph",
            )
            try:
                return capture_autograd_program(evaluate, arguments)
            except NotImplementedError:
                expected_failure = True
                raise
    finally:
        for warning in recorded:
            if (
                expected_failure
                and issubclass(warning.category, UserWarning)
                and str(warning.message).startswith("Error detected in ")
                and warning.filename.replace("\\", "/").endswith(
                    "/torch/autograd/graph.py"
                )
            ):
                continue
            warnings.warn_explicit(
                warning.message,
                warning.category,
                warning.filename,
                warning.lineno,
                source=warning.source,
            )


def _fixed_cond_operand_layout(value):
    """Give conditional operands an explicit contiguous storage layout.

    Inductor can realize a shared pointwise producer inside a convolution's
    channel stack after conditional branches have captured contiguous operand
    strides. An explicit storage view makes that layout a consumer requirement,
    so branch code and its caller agree even when ordinary clones are removed.
    The singleton axis keeps this view distinct from a removable identity view.
    """
    contiguous = value.clone(memory_format=torch.contiguous_format)
    return (
        torch.as_strided(
            contiguous,
            (1, *contiguous.shape),
            (contiguous.numel(), *contiguous.stride()),
        )
        .squeeze(0)
        .clone(memory_format=torch.contiguous_format)
    )


def _expand_scalar_binary_operands(program):
    """Retain empty hidden vmap lanes in scalar/spatial ATen arithmetic.

    A captured authored parametrization can multiply a mapped scalar by a
    shared spatial tensor before current assembly expands its result. PyTorch's
    pointwise vmap rule selects lane zero while preparing that scalar, which
    fails for zero lanes. Explicit scalar expansion preserves the same visible
    arithmetic and native zero-dimensional dtype promotion.
    """
    binary = {
        torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.pow.Tensor_Tensor,
    }
    graph = program.graph
    for node in tuple(graph.nodes):
        if node.op != "call_function" or node.target not in binary:
            continue
        output = node.meta.get("val")
        if not torch.is_tensor(output) or not output.ndim:
            continue
        operands = node.args[:2]
        metadata = tuple(
            value.meta.get("val") if isinstance(value, torch.fx.Node) else None
            for value in operands
        )
        if not all(torch.is_tensor(value) for value in metadata):
            continue
        dtype = torch.result_type(*metadata)
        arguments = list(node.args)
        for index, value in enumerate(metadata):
            if value.ndim:
                continue
            with graph.inserting_before(node):
                scalar = arguments[index]
                if value.dtype != dtype:
                    scalar = graph.call_function(
                        torch.ops.aten._to_copy.default,
                        (scalar,),
                        {"dtype": dtype},
                    )
                arguments[index] = graph.call_function(
                    torch.ops.aten.expand.default,
                    (scalar, tuple(output.shape)),
                )
        node.args = tuple(arguments)
    graph.lint()
    program.recompile()
    return program


class _DufortCurrentPhase(torch.nn.Module):
    def __init__(self, handler, use_history, clock_owner):
        super().__init__()
        self.handler = handler
        self.use_history = use_history
        if clock_owner is not None:
            # Mechanism clocks resolve through framework-owned references to
            # this live owner rather than through registered handler buffers.
            # The capture mapping isolates its clock just like other inputs.
            self.clock_owner = clock_owner

    def forward(self, voltage, previous):
        if self.use_history:
            return self.handler._evaluate_dufort_current_frame(voltage, previous)
        current, current_frame = self.handler._evaluate_explicit_current_frame(voltage)
        return (
            current,
            torch.zeros_like(current),
            current_frame,
            tuple(torch.zeros_like(value) for value in current_frame),
        )


class DufortCurrentPrograms:
    """Capture the shared current evaluators before conditional execution.

    Native current evaluation installs ASSIGNED and SAVE_CURRENT products on
    registered module buffers. Those rebindings are isolated by functional_call
    during capture; the tensor forward/backward programs receive every
    registered raw parameter and buffer explicitly and return mutable products.
    Runtime conditionals execute the captured programs, and only accepted
    products are installed on the functional owner's private handler.
    """

    def __init__(self, handler, voltage, *, clock_owner=None):
        self.handler = handler
        self._clock_owner = clock_owner
        self._grad_raw = False
        self._authored_autograd = False
        registered = tuple(
            (path, owner, name, value)
            for path, owner in handler.named_modules()
            for registry in (owner._parameters, owner._buffers)
            for name, value in registry.items()
            if torch.is_tensor(value)
        )
        self._slots = tuple(
            (
                f"handler.{path + '.' if path else ''}{name}",
                owner,
                name,
            )
            for path, owner, name, _value in registered
        )
        mutable = {
            (id(owner), name)
            for owner in handler.mechanisms.values()
            for name in (*owner._assigned, *owner._saved_current_buffer_names)
            if name in owner._buffers
        }
        self._output_slots = tuple(
            (slot, owner, name)
            for slot, owner, name in self._slots
            if (id(owner), name) in mutable
        )

        def capture(use_history, values, *, preserve_autograd=False):
            phase = _DufortCurrentPhase(handler, use_history, clock_owner)

            def evaluate(voltage, previous, clock, *values):
                mapping = {
                    slot: value.clone(memory_format=torch.contiguous_format)
                    for (slot, _owner, _name), value in zip(
                        self._slots, values, strict=True
                    )
                }
                if clock_owner is not None:
                    mapping["clock_owner.t"] = clock.clone()
                frame = torch.func.functional_call(
                    phase,
                    mapping,
                    (voltage, previous),
                    tie_weights=False,
                )
                frame = torch.utils._pytree.tree_map(
                    lambda value: value.clone(memory_format=torch.contiguous_format),
                    frame,
                )
                return frame, tuple(
                    mapping[slot].clone(memory_format=torch.contiguous_format)
                    for slot, _owner, _name in self._output_slots
                )

            arguments = (
                voltage.clone(memory_format=torch.contiguous_format),
                voltage.clone(memory_format=torch.contiguous_format),
                clock.clone(memory_format=torch.contiguous_format),
                *values,
            )
            # First eager evaluation establishes the declaration-owned schemas
            # and bounded NUMERICAL contracts before tracing tensor operations.
            with _CurrentAutogradPolicy() as policy:
                example = evaluate(*arguments)
            if preserve_autograd and (
                policy.detached_arithmetic or _contains_custom_autograd(example)
            ):
                self._authored_autograd = True
                try:
                    return _capture_autograd_program(evaluate, arguments)
                except NotImplementedError:
                    # Forward-only authored Functions are valid in inference.
                    # Native evaluation decides whether a runtime gradient
                    # actually crosses their unsupported backward boundary.
                    self._grad_raw = True
                    return None
            return _expand_scalar_binary_operands(
                make_fx(evaluate, tracing_mode="real")(*arguments)
            )

        with torch.inference_mode(False), torch.no_grad():
            voltage = voltage.detach().clone(memory_format=torch.contiguous_format)
            clock = (
                voltage.new_zeros(())
                if clock_owner is None
                else clock_owner._buffers["t"]
                .detach()
                .clone(memory_format=torch.contiguous_format)
            )
            values = tuple(
                value.detach().clone(memory_format=torch.contiguous_format)
                for *_metadata, value in registered
            )
            self._initial = capture(False, values)
            self._history = capture(True, values)
            with torch.enable_grad():
                # Shared delay helpers select their native shift/circular
                # policy from gradient mode as well as module training mode.
                # Both policies retain explicit tensor inputs and gradients.
                voltage.requires_grad_(True)
                clock.requires_grad_(True)
                values = tuple(
                    value.requires_grad_(
                        value.is_floating_point() or value.is_complex()
                    )
                    for value in values
                )
                self._grad_initial = capture(False, values, preserve_autograd=True)
                self._grad_history = capture(True, values, preserve_autograd=True)

    def __call__(self, voltage, previous, use_history):
        values = tuple(
            owner._buffers[name] if name in owner._buffers else owner._parameters[name]
            for _slot, owner, name in self._slots
        )
        clock = (
            voltage.new_zeros(())
            if self._clock_owner is None
            else self._clock_owner._buffers["t"]
        )
        operands = (voltage, previous, clock, *values)
        initial_program, history_program = (
            (self._grad_initial, self._grad_history)
            if torch.is_grad_enabled()
            else (self._initial, self._history)
        )
        if (
            torch._C._are_functorch_transforms_active()
            or torch.autograd.forward_ad._current_level >= 0
        ):
            if self._authored_autograd and not any(
                _has_empty_transform_axis(value) for value in operands
            ):
                # Forward AD and vmap use the author's own JVP/batching rules.
                # These shared pure evaluations retain accepted products while
                # mechanism advance and RNG consumption still happen once.
                return self.diagnostic(voltage, previous, use_history)
            if self._authored_autograd:
                initial_program, history_program = self._initial, self._history
            # torch.cond cannot carry forward AD through GradTrackingTensor
            # views or direct dual tensors, and its vmap rule rejects
            # constant/dynamic output strides.
            # Both captured programs are pure tensor graphs: selecting their
            # products retains the accepted module state without executing an
            # authored hook, mechanism advance, or random draw a second time.
            initial = initial_program(*operands)
            history = history_program(*operands)
            frame, products = torch.utils._pytree.tree_map(
                lambda accepted, starter: torch.where(use_history, accepted, starter),
                history,
                initial,
            )
        else:
            if torch.is_grad_enabled() and self._grad_raw:
                return self.diagnostic(voltage, previous, use_history)
            # Shared parameters and framework read bindings can occupy several
            # registered slots. The conditional accepts each live tensor only
            # once; its alias checker rejects repeated input storage even when
            # every captured current operation is out of place.
            unique = []
            indices = []
            for value in operands:
                index = next(
                    (index for index, item in enumerate(unique) if value is item),
                    len(unique),
                )
                if index == len(unique):
                    unique.append(value)
                indices.append(index)
            operands = tuple(_fixed_cond_operand_layout(value) for value in unique)

            def history_phase(*values):
                return history_program(*(values[index] for index in indices))

            def initial_phase(*values):
                return initial_program(*(values[index] for index in indices))

            frame, products = torch.cond(
                use_history,
                history_phase,
                initial_phase,
                operands,
            )
        for (_slot, owner, name), value in zip(
            self._output_slots, products, strict=True
        ):
            owner._buffers[name] = value
        return frame

    def diagnostic(self, voltage, previous, use_history):
        """Evaluate raw shared phases for diagnostics and authored AD rules.

        Captured programs intentionally contain only tensor operations. The
        diagnostic owner observes registered inputs, Python state, and RNG
        effects. Transform execution uses the same shared handlers to preserve
        authored JVP and batching rules. Both repeatable phases are evaluated,
        with only selected ASSIGNED and SAVE_CURRENT products retained.
        """
        before = tuple(
            owner._buffers[name] for _slot, owner, name in self._output_slots
        )
        current, current_frame = self.handler._evaluate_explicit_current_frame(voltage)
        initial_frame = (
            current,
            torch.zeros_like(current),
            current_frame,
            tuple(torch.zeros_like(value) for value in current_frame),
        )
        initial_products = tuple(
            owner._buffers[name].clone() for _slot, owner, name in self._output_slots
        )
        for (_slot, owner, name), value in zip(self._output_slots, before, strict=True):
            owner._buffers[name] = value
        history_frame = self.handler._evaluate_dufort_current_frame(voltage, previous)
        history_products = tuple(
            owner._buffers[name] for _slot, owner, name in self._output_slots
        )
        frame, products = torch.utils._pytree.tree_map(
            lambda history, initial: torch.where(use_history, history, initial),
            (history_frame, history_products),
            (initial_frame, initial_products),
        )
        for (_slot, owner, name), value in zip(
            self._output_slots, products, strict=True
        ):
            owner._buffers[name] = value
        return frame
