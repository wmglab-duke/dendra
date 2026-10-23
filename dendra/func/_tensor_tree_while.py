"""Shared owned tensor-tree boundaries for bounded inference while loops.

Adapters supply one pure timestep. Recording buffers belong to a compiled
invocation; recurrent state is packed by dtype/device and callback invariants
remain explicit operands. No example values or preparation graphs are cached.
"""

from __future__ import annotations

from functools import wraps

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree as pytree

from dendra.helpers import MEMBRANE_ASSEMBLY

from ._callback_carry import _CallbackCarryPartition
from ._carry_packing import _PackedCarryLayout
from ._hop_inputs import (
    _input_usage,
    _isolate_dense_lifted_inputs,
    _LiftedViewIsolation,
)
from ._while_inductor import _compile_while


def _clone_tree(tree):
    return pytree.tree_map_only(
        torch.Tensor,
        lambda value: value.clone(memory_format=torch.contiguous_format),
        tree,
    )


def _schema(tree):
    values, spec = pytree.tree_flatten(tree)
    return spec, tuple(
        (
            tuple(value.shape),
            value.dtype,
            value.device,
            tuple(value.stride()),
            value.storage_offset(),
            value.requires_grad,
            value.is_leaf,
            value.is_conj(),
            value.is_neg(),
            torch.is_inference(value),
        )
        if torch.is_tensor(value)
        else (type(value), value)
        for value in values
    )


class _CompiledTensorTreeWhile:
    """Capture and compile an adapter's fixed-horizon inference recurrence.

    ``body(parameters, prepared, state, carries, sample)`` returns
    ``((next_state, next_carries), emissions)``. The tape is an arbitrary
    Tensor/None PyTree with a leading time axis on each Tensor. No callbacks
    and empty emissions are supported. Public adapters own admission policy,
    host sampling, validation, execution reports and gradient-mode dispatch.

    ``kernel_for`` returns a callable with the same five operand arguments as
    this object. Capture errors propagate to the adapter; the shared engine
    does not select a fallback. Horizons below four execute only the bounded
    tail; larger horizons capture a real while loop plus at most three steps.
    """

    def __init__(
        self,
        body,
        *,
        steps,
        compile_options,
        device,
        plans=None,
        clone_state=None,
        tracer=None,
        compiler=None,
    ):
        if type(steps) is not int or steps < 1:
            raise ValueError("compiled tensor-tree loops require a positive step count")
        self._body, self._steps = body, steps
        self._options = dict(compile_options)
        self._device = torch.device(device)
        self._partition = _CallbackCarryPartition.from_plans(plans)
        self._clone_state = _clone_tree if clone_state is None else clone_state
        self._tracer = make_fx if tracer is None else tracer
        self._compiler = _compile_while if compiler is None else compiler
        self._kernels = {}

    def _samples(self, tape):
        values, spec = pytree.tree_flatten(tape)
        tensors, positions = [], []
        for value in values:
            if value is None:
                positions.append(None)
            elif (
                not torch.is_tensor(value)
                or value.ndim < 1
                or value.shape[0] != self._steps
            ):
                raise ValueError(
                    "while input tapes require a matching leading time axis"
                )
            else:
                positions.append(len(tensors))
                tensors.append(value)
        return tuple(tensors), (spec, tuple(positions))

    def kernel_for(self, parameters, prepared, state, carries, tape):
        samples, sample_layout = self._samples(tape)
        signature = (
            _schema((parameters, prepared, state, carries, samples)),
            sample_layout,
            MEMBRANE_ASSEMBLY.value,
        )
        if signature not in self._kernels:
            with torch.inference_mode(False), torch.no_grad():
                self._kernels[signature] = self._capture(
                    parameters, prepared, state, carries, samples, sample_layout
                )
        return self._kernels[signature]

    def __call__(self, parameters, prepared, state, carries, tape):
        return self.kernel_for(parameters, prepared, state, carries, tape)(
            parameters, prepared, state, carries, tape
        )

    def _capture(self, parameters, prepared, state, carries, samples, sample_layout):
        body, partition, steps = self._body, self._partition, self._steps
        clone_state = self._clone_state
        block, full_steps = 4, (steps // 4) * 4
        # Invariant operands are read-only by the authored step contract.
        # Trace them directly under no_grad: cloning changes non-dense views,
        # offsets and conjugate/negative bits, while detaching changes is_leaf.
        # FX records argument placeholders, not these values or their graphs.
        state = clone_state(state)
        recurrent, invariant = partition.split(carries)
        layout = _PackedCarryLayout.from_tree((state, recurrent))
        # Capture the logical views actually presented by the inference HOP:
        # packing owns contiguous storage and does not retain AD history.
        state, recurrent = layout.unpack(layout.pack((state, recurrent)))

        def authored(parameters, prepared, state, recurrent, invariant, sample):
            spec, positions = sample_layout
            sample = pytree.tree_unflatten(
                [None if index is None else sample[index] for index in positions], spec
            )
            (next_state, next_carries), emitted = body(
                parameters,
                prepared,
                state,
                partition.merge(recurrent, invariant),
                sample,
            )
            next_recurrent, _ = partition.split(next_carries)
            return (next_state, next_recurrent), emitted or {}

        graph = self._tracer(authored)(
            parameters,
            prepared,
            state,
            recurrent,
            invariant,
            tuple(
                value.index_select(
                    0, torch.zeros(1, dtype=torch.long, device=value.device)
                ).squeeze(0)
                for value in samples
            ),
        )
        parameter_use, prepared_use, _, _, invariant_use, sample_use = _input_usage(
            graph, (parameters, prepared, state, recurrent, invariant, samples)
        )
        isolate_views = _LiftedViewIsolation.from_usage(
            (
                parameter_use,
                prepared_use,
                invariant_use,
                # The loop indexes every supplied tape before calling the
                # authored graph, including tapes that graph does not read.
                pytree.tree_map(lambda _value: True, sample_use),
            )
        )
        # FX already recorded the emission tensors' metadata. Discover their
        # schema without executing another numerical model transition.
        output = next(node for node in graph.graph.nodes if node.op == "output")
        output_values = pytree.tree_map(
            lambda value: value.meta["val"]
            if isinstance(value, torch.fx.Node)
            else value,
            output.args[0],
        )
        if hasattr(graph, "_out_spec"):
            output_values = pytree.tree_unflatten(list(output_values), graph._out_spec)
        emission_values = output_values[1]
        emission_leaves, emission_spec = pytree.tree_flatten(emission_values)
        shapes = tuple(
            (tuple(value.shape), value.dtype, value.device) for value in emission_leaves
        )
        # A Network can combine accelerator draw values with CPU RNG bytes.
        # Independent synchronized counters avoid copying a scalar index
        # between those devices on every accepted timestep.
        devices = tuple(
            dict.fromkeys(
                (self._device,)
                + tuple(value.device for value in samples)
                + tuple(device for _, _, device in shapes)
            )
        )
        sample_devices = tuple(devices.index(value.device) for value in samples)
        emission_devices = tuple(devices.index(device) for _, _, device in shapes)

        def execute(parameters, prepared, state, carries, samples):
            recurrent, invariant = partition.split(carries)
            parameters, prepared, invariant, samples = _isolate_dense_lifted_inputs(
                (parameters, prepared, invariant, samples)
            )
            buffers = tuple(
                torch.empty((steps, *shape), dtype=dtype, device=device)
                for shape, dtype, device in shapes
            )
            initial = layout.pack((clone_state(state), recurrent))

            def apply_step(counters, offset, current, current_carries):
                selectors = tuple((counter + offset).reshape(1) for counter in counters)
                selected = tuple(
                    value.index_select(0, selectors[device]).squeeze(0)
                    for value, device in zip(samples, sample_devices, strict=True)
                )
                (next_state, next_carries), emitted = graph(
                    parameters, prepared, current, current_carries, invariant, selected
                )
                leaves, _ = pytree.tree_flatten(emitted)
                for buffer, value, device in zip(
                    buffers, leaves, emission_devices, strict=True
                ):
                    buffer.index_copy_(0, selectors[device], value.unsqueeze(0))
                return next_state, next_carries

            def condition(counters, packed):
                return counters[0] < full_steps

            def advance(counters, packed):
                current, current_carries = layout.unpack(packed)
                for offset in range(block):
                    current, current_carries = apply_step(
                        counters, offset, current, current_carries
                    )
                return (
                    tuple(counter + block for counter in counters),
                    layout.pack((current, current_carries)),
                )

            counters = tuple(
                torch.zeros((), dtype=torch.long, device=d) for d in devices
            )
            packed = initial
            if full_steps:
                counters, packed = torch.while_loop(
                    condition, advance, (counters, packed)
                )
            if steps != full_steps:
                final, final_carries = layout.unpack(packed)
                for offset in range(steps - full_steps):
                    final, final_carries = apply_step(
                        counters, offset, final, final_carries
                    )
                packed = layout.pack((final, final_carries))
            return packed, pytree.tree_unflatten(list(buffers), emission_spec)

        compiled = self._compiler(execute, self._options)

        @wraps(compiled)
        def materialize(parameters, prepared, state, carries, tape):
            samples, _ = self._samples(tape)
            recurrent, invariant = partition.split(carries)
            parameters, prepared, invariant, samples = isolate_views(
                (parameters, prepared, invariant, samples)
            )
            packed, emitted = compiled(
                parameters,
                prepared,
                state,
                partition.merge(recurrent, invariant),
                samples,
            )
            final, updated = layout.unpack(packed)
            final = clone_state(final)
            updated = partition.merge(updated, partition.split(carries)[1])
            updated = pytree.tree_map(
                lambda value, template: (
                    torch.empty_like(template).copy_(value)
                    if torch.is_tensor(value)
                    else value
                ),
                updated,
                carries,
            )
            return final, updated, emitted

        return materialize


__all__ = []
