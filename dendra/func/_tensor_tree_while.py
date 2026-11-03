"""Shared owned tensor-tree boundaries for bounded inference while loops.

Adapters supply one pure timestep. Recording buffers belong to a compiled
invocation; recurrent state is packed by dtype/device and callback invariants
remain explicit operands. No example values or preparation graphs are cached.
"""

from __future__ import annotations

from functools import wraps
from types import MethodDescriptorType

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree as pytree

from dendra.helpers import MEMBRANE_ASSEMBLY

from ._callback_carry import _CallbackCarryPartition
from ._carry_packing import _clone_tree_like, _PackedCarryLayout
from ._hop_inputs import (
    _input_usage,
    _isolate_dense_lifted_inputs,
    _LiftedViewIsolation,
)
from ._state_projection import (
    _can_project_state,
    _plain_projection_context,
    _plain_state_tree,
    _tensor_descriptor,
)
from ._tensor_contracts import _clone_contiguous_tree as _clone_tree
from ._tensor_contracts import capture_schema as _schema
from ._while_inductor import _compile_while

_SAMPLE_METHODS = ("index_select", "squeeze", "stride")
_SAMPLE_DESCRIPTORS = tuple(_tensor_descriptor(name) for name in _SAMPLE_METHODS)


def _own_samples(samples):
    # Authored steps receive fresh index_select results, independent of the
    # tape's view layout or inference status. Establish that same private
    # boundary before specialization and Dynamo's input guards. Keep observable
    # Tensor methods, subclasses and execution modes on their existing path.
    if (
        not samples
        or not _plain_projection_context()
        or not all(
            type(descriptor) is MethodDescriptorType
            and _tensor_descriptor(name) is descriptor
            for name, descriptor in zip(
                _SAMPLE_METHODS, _SAMPLE_DESCRIPTORS, strict=True
            )
        )
        or not _plain_state_tree(samples)
    ):
        return samples, False
    with torch.inference_mode(False), torch.no_grad():
        return (
            tuple(
                value.clone(memory_format=torch.contiguous_format) for value in samples
            ),
            True,
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
        project_state=None,
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
        self._project_state = project_state
        self._tracer = make_fx if tracer is None else tracer
        self._compiler = _compile_while if compiler is None else compiler
        self._kernels = {}
        self._layouts = {}

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

    def _pack(self, state, carries):
        # The authored graph already sees owned packed views. Establish that
        # same private boundary before specialization and Dynamo's guards:
        # public continuation leaves can have different offsets or aliases.
        project = _can_project_state(
            self._clone_state,
            self._project_state,
            state,
            carries,
            self._partition,
        )
        with torch.inference_mode(False), torch.no_grad():
            state = (self._project_state if project else self._clone_state)(state)
            recurrent, invariant = self._partition.split(carries)
            if project:
                values, spec = pytree.tree_flatten((state, recurrent))
                signature = (
                    spec,
                    tuple(
                        None
                        if value is None
                        else (
                            tuple(value.shape),
                            value.dtype,
                            value.device,
                            value.layout,
                        )
                        for value in values
                    ),
                )
                layout = self._layouts.get(signature)
                if layout is None:
                    # Use the shared construction contract for a new layout;
                    # subsequent frames reuse its metadata and flattened leaves.
                    layout = _PackedCarryLayout.from_tree((state, recurrent))
                    self._layouts[signature] = layout
                initial = layout.pack_leaves(values, spec)
            else:
                layout = _PackedCarryLayout.from_tree((state, recurrent))
                initial = layout.pack((state, recurrent))
        return layout, initial, invariant

    def _kernel_for(
        self,
        parameters,
        prepared,
        layout,
        initial,
        invariant,
        samples,
        sample_layout,
        *,
        owned_samples=False,
    ):
        signature = (
            layout,
            _schema((parameters, prepared, initial, invariant, samples)),
            sample_layout,
            MEMBRANE_ASSEMBLY.value,
            owned_samples,
        )
        kernel = self._kernels.get(signature)
        if kernel is None:
            with torch.inference_mode(False), torch.no_grad():
                kernel = self._capture(
                    parameters,
                    prepared,
                    layout,
                    initial,
                    invariant,
                    samples,
                    sample_layout,
                    owned_samples=owned_samples,
                )
                self._kernels[signature] = kernel
        return kernel

    def kernel_for(self, parameters, prepared, state, carries, tape):
        # Admission captures eagerly, without retaining these example packs.
        # The returned raw-operand callable prepares its own current values.
        samples, sample_layout = self._samples(tape)
        layout, initial, invariant = self._pack(state, carries)
        samples, owned_samples = _own_samples(samples)
        kernel = self._kernel_for(
            parameters,
            prepared,
            layout,
            initial,
            invariant,
            samples,
            sample_layout,
            owned_samples=owned_samples,
        )

        @wraps(kernel)
        def materialize(parameters, prepared, state, carries, tape):
            return self(parameters, prepared, state, carries, tape)

        return materialize

    def __call__(self, parameters, prepared, state, carries, tape):
        samples, sample_layout = self._samples(tape)
        layout, initial, invariant = self._pack(state, carries)
        samples, owned_samples = _own_samples(samples)
        kernel = self._kernel_for(
            parameters,
            prepared,
            layout,
            initial,
            invariant,
            samples,
            sample_layout,
            owned_samples=owned_samples,
        )
        return kernel(parameters, prepared, initial, invariant, samples, carries)

    def _capture(
        self,
        parameters,
        prepared,
        layout,
        initial,
        invariant,
        samples,
        sample_layout,
        *,
        owned_samples=False,
    ):
        body, partition, steps = self._body, self._partition, self._steps
        clone_state = self._clone_state
        block, full_steps = 4, (steps // 4) * 4
        # Invariant operands are read-only by the authored step contract.
        # Trace them directly under no_grad: cloning changes non-dense views,
        # offsets and conjugate/negative bits, while detaching changes is_leaf.
        # FX records argument placeholders, not these values or their graphs.
        # Capture the logical views actually presented by the inference HOP:
        # packing owns contiguous storage and does not retain AD history.
        state, recurrent = layout.unpack(initial)

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

        def execute(parameters, prepared, initial, invariant, samples):
            if owned_samples:
                parameters, prepared, invariant = _isolate_dense_lifted_inputs(
                    (parameters, prepared, invariant)
                )
            else:
                parameters, prepared, invariant, samples = _isolate_dense_lifted_inputs(
                    (parameters, prepared, invariant, samples)
                )
            buffers = tuple(
                torch.empty((steps, *shape), dtype=dtype, device=device)
                for shape, dtype, device in shapes
            )

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
        def materialize(parameters, prepared, initial, invariant, samples, carries):
            parameters, prepared, invariant, samples = isolate_views(
                (parameters, prepared, invariant, samples)
            )
            packed, emitted = compiled(
                parameters,
                prepared,
                initial,
                invariant,
                samples,
            )
            final, updated = layout.unpack(packed)
            final = clone_state(final)
            updated = partition.restore_invariants(updated, carries)
            updated = _clone_tree_like(updated, carries)
            return final, updated, emitted

        return materialize


__all__ = []
