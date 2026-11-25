"""Explicit CUDA Graph replay at the compiled Population scan boundary."""

from __future__ import annotations

from threading import RLock

import torch
from torch.utils import _pytree as pytree


def _metadata(value):
    return (value.shape, value.stride(), value.dtype, value.device, value.requires_grad)


def _clone(value):
    return value.clone(memory_format=torch.preserve_format)


def _host_duration_passthrough(plans):
    from ._callbacks import Recorder, _BoundCadencedRecorder

    # Custom callbacks receive the whole state and can observe the CPU
    # duration carry. Their captures keep the original host input value.
    return all(
        type(plan.callback) in (Recorder, _BoundCadencedRecorder)
        for plan in plans or ()
    )


class _GraphOperands:
    """Flatten operands, preserving repeated tensors and frozen host values."""

    def __init__(self, args):
        leaves, self.spec = pytree.tree_flatten(args)
        self.positions = {}
        self.host = {}
        self.metadata = []
        tensors = []
        identities = {}
        for index, value in enumerate(leaves):
            if isinstance(value, torch.Tensor) and value.is_cuda:
                position = identities.get(id(value))
                if position is None:
                    position = identities[id(value)] = len(tensors)
                    tensors.append(value)
                    self.metadata.append(_metadata(value))
                self.positions[index] = position
            else:
                self.host[index] = (
                    _clone(value) if isinstance(value, torch.Tensor) else value
                )
        self.count = len(leaves)

    def unflatten(self, tensors):
        leaves = [None] * self.count
        for index, position in self.positions.items():
            leaves[index] = tensors[position]
        for index, value in self.host.items():
            leaves[index] = value
        return pytree.tree_unflatten(leaves, self.spec)

    def match(self, args, device):
        leaves, spec = pytree.tree_flatten(args)
        if spec != self.spec:
            return None
        tensors = [None] * len(self.metadata)
        storages = set()
        for index, position in self.positions.items():
            value = leaves[index]
            if (
                type(value) not in (torch.Tensor, torch.nn.Parameter)
                or value.layout != torch.strided
                or value.device != device
                or value.is_conj()
                or value.is_neg()
                or _metadata(value) != self.metadata[position]
            ):
                return None
            previous = tensors[position]
            if previous is not None:
                if previous is not value:
                    return None
                continue
            if value.numel():
                storage = value.untyped_storage().data_ptr()
                if storage in storages:
                    return None
                storages.add(storage)
            tensors[position] = value
        for index, expected in self.host.items():
            value = leaves[index]
            if isinstance(expected, torch.Tensor):
                if (
                    not isinstance(value, torch.Tensor)
                    or value.device.type != "cpu"
                    or value.requires_grad
                    or _metadata(value) != _metadata(expected)
                    or not torch.equal(value, expected)
                ):
                    return None
            elif (
                type(expected)
                not in (type(None), bool, int, float, str, torch.dtype, torch.device)
                or type(value) is not type(expected)
                or value != expected
            ):
                return None
        return tuple(tensors)


class _GraphResults:
    """Keep CUDA results unique while retaining the ordinary output PyTree."""

    def __init__(self, example):
        leaves, self.spec = pytree.tree_flatten(example)
        self.positions = {}
        self.host = {}
        self.host_requires_grad = False
        identities = {}
        for index, value in enumerate(leaves):
            if isinstance(value, torch.Tensor) and value.is_cuda:
                self.positions[index] = identities.setdefault(id(value), index)
            elif isinstance(value, torch.Tensor):
                self.host_requires_grad |= value.requires_grad
                self.host[index] = _clone(value.detach())
            else:
                self.host[index] = value
        self.unique = tuple(identities.values())
        self.count = len(leaves)

    def flatten(self, result):
        leaves = pytree.tree_leaves(result)
        return tuple(leaves[index] for index in self.unique)

    def unflatten(self, tensors):
        selected = dict(zip(self.unique, tensors, strict=True))
        leaves = [None] * self.count
        for index, position in self.positions.items():
            leaves[index] = selected[position]
        for index, value in self.host.items():
            leaves[index] = _clone(value) if isinstance(value, torch.Tensor) else value
        return pytree.tree_unflatten(leaves, self.spec)


class _InferenceGraph:
    def __init__(self, function, inputs, stream):
        self.static = tuple(_clone(value) for value in inputs)
        warmup = torch.cuda.Stream(device=stream.device)
        warmup.wait_stream(stream)
        try:
            with torch.cuda.stream(warmup):
                for _ in range(3):
                    self._copy_inputs(inputs)
                    function(*self.static)
                self._copy_inputs(inputs)
        finally:
            stream.wait_stream(warmup)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=warmup):
            self.outputs = function(*self.static)

    def _copy_inputs(self, inputs):
        for target, value in zip(self.static, inputs, strict=True):
            target.copy_(value)

    def try_replay(self, *inputs):
        self._copy_inputs(inputs)
        self.graph.replay()
        return tuple(_clone(value) for value in self.outputs)


class _CudaGraphCall:
    """Keep one fixed-signature replay slot per compiled scan executor."""

    def __init__(self, execute, device, *, detach_host_outputs=False):
        self._execute = execute
        self._device = torch.device(device)
        self._detach_host_outputs = detach_host_outputs
        self._lock = RLock()
        self._operands = None
        self._replay = None

    def __call__(self, *args):
        if torch.is_grad_enabled():
            from torch._functorch import config

            # A captured adjoint can replay more than once when the caller
            # retains the graph. Its saved activations cannot be donated.
            with config.patch(donated_buffer=False):
                return self._call(args)
        return self._call(args)

    def _call(self, args):
        from dendra.models.integrators.core import _population_runtime_modes_plain

        if torch.is_autocast_enabled("cuda") or not _population_runtime_modes_plain():
            return self._execute(*args)
        with torch.cuda.device(self._device):
            if torch.cuda.is_current_stream_capturing():
                return self._execute(*args)
            with self._lock:
                result = self._try_replay(args)
        return self._execute(*args) if result is None else result

    def _try_replay(self, args):
        stream = torch.cuda.current_stream(self._device)
        mode = (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
        if self._operands is None:
            operands = _GraphOperands(args)
            tensors = operands.match(args, self._device)
            if not tensors:
                return None
            # The traced scan already fixes authored layout-dependent behavior.
            # Warm its executor on owned buffers so that it specializes on the
            # capture layout, including dense copies of broadcast operands.
            example = self._execute(*args)
            results = _GraphResults(example)
            if results.host_requires_grad and not self._detach_host_outputs:
                self._operands = operands
                return example

            def execute(*inputs):
                return results.flatten(self._execute(*operands.unflatten(inputs)))

            if mode[0] and any(value.requires_grad for value in tensors):
                from ._cuda_graph_training import capture_training

                replay = capture_training(execute, tensors)
            else:
                replay = (
                    _InferenceGraph(execute, tensors, stream)
                    if results.unique
                    else None
                )
            self._operands = operands
            self._results = results
            self._stream = stream
            self._mode = mode
            self._replay = replay
            return example
        if self._replay is None or stream != self._stream or mode != self._mode:
            return None
        tensors = self._operands.match(args, self._device)
        if tensors is None:
            return None
        result = self._replay.try_replay(*tensors)
        return None if result is None else self._results.unflatten(result)
