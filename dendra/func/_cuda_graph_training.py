"""First-order CUDA graph replay for one fixed tensor-only scan chunk."""

from threading import Lock
from weakref import finalize, ref

import torch

from ._types import FunctionalizationError


def _metadata(tensor):
    return tensor.shape, tensor.stride(), tensor.dtype, tensor.device, tensor.requires_grad


def _supported_inputs(inputs, device):
    if any(
        type(value) not in (torch.Tensor, torch.nn.Parameter)
        or value.device != device
        or value.layout != torch.strided
        or value.is_conj()
        or value.is_neg()
        for value in inputs
    ):
        return False
    # Preserve alias-dependent behavior by using uncaptured execution for
    # distinct views of one operand. The caller can deduplicate identical leaves.
    return len({value.untyped_storage() for value in inputs}) == len(inputs)


def _output_grad_mask(outputs, device):
    if (
        not isinstance(outputs, tuple)
        or any(
            type(value) is not torch.Tensor
            or value.device != device
            or value.layout != torch.strided
            for value in outputs
        )
    ):
        return ()
    return tuple(value.requires_grad for value in outputs)


def _gradients(outputs, output_mask, inputs, cotangents, *, retain_graph=False):
    return torch.autograd.grad(
        tuple(value for value, active in zip(outputs, output_mask, strict=True) if active),
        inputs,
        cotangents,
        allow_unused=True,
        retain_graph=retain_graph,
    )


class _Lease:
    def __init__(self, graph):
        self.graph = ref(graph)
        self.ran_backward = False

    def release(self):
        graph = self.graph()
        if graph is not None:
            graph._release(self)

    def release_after_backward(self):
        # A user-installed saved-tensor pack hook can drop the Python tensor
        # during forward. In that case keep the lease until the node dies.
        if self.ran_backward:
            self.release()


class _Replay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, graph, lease, *inputs):
        ctx.graph = graph
        ctx.lease = lease
        ctx.set_materialize_grads(False)
        # Backward releases saved tensors unless retain_graph=True. Retaining
        # this marker keeps the graph's activations reserved just as long.
        marker = torch.empty(0, device="cpu")
        finalize(marker, lease.release_after_backward)
        finalize(ctx, lease.release)
        ctx.save_for_backward(marker)
        outputs = graph._forward(inputs)
        ctx.mark_non_differentiable(*(
            value for value, active in zip(outputs, graph._output_mask, strict=True) if not active
        ))
        return outputs

    @staticmethod
    def backward(ctx, *cotangents):
        if torch.is_grad_enabled():
            raise FunctionalizationError(
                "CUDA graph scan replay supports first-order reverse-mode gradients only"
            )
        # Use the default autograd error for a second traversal without retain_graph.
        _ = ctx.saved_tensors
        gradients = ctx.graph._backward(ctx.lease, cotangents)
        return None, None, *gradients


class TrainingGraph:
    """A single graph slot with private forward activations and VJP buffers."""

    def __init__(
        self, stream, input_metadata, static_inputs, outputs, output_mask,
        forward, cotangents, gradients, backward,
    ):
        self._stream = stream
        self._device = stream.device
        self._input_metadata = input_metadata
        self._static_inputs = static_inputs
        self._outputs = outputs
        self._output_mask = output_mask
        self._forward_graph = forward
        self._cotangents = cotangents
        self._gradients = gradients
        self._backward_graph = backward
        self._lease = None
        self._lock = Lock()

    def try_replay(self, *inputs):
        if (
            not torch.is_grad_enabled()
            or torch.is_inference_mode_enabled()
            or torch._C._are_functorch_transforms_active()
            or not _supported_inputs(inputs, self._device)
            or tuple(map(_metadata, inputs)) != self._input_metadata
        ):
            return None
        with torch.cuda.device(self._device):
            if torch.cuda.current_stream() != self._stream or torch.cuda.is_current_stream_capturing():
                return None
        with self._lock:
            if self._lease is not None:
                return None
            lease = _Lease(self)
            self._lease = lease
        try:
            return _Replay.apply(self, lease, *inputs)
        except BaseException:
            lease.release()
            raise

    def _release(self, lease):
        with self._lock:
            if self._lease is lease:
                self._lease = None

    def _forward(self, inputs):
        for destination, source in zip(self._static_inputs, inputs, strict=True):
            destination.copy_(source)
        self._forward_graph.replay()
        return tuple(value.clone() for value in self._outputs)

    def _backward(self, lease, cotangents):
        if torch.cuda.current_stream(self._device) != self._stream:
            raise FunctionalizationError("CUDA graph scan backward requires its forward CUDA stream")
        with self._lock:
            if self._lease is not lease:
                raise FunctionalizationError("CUDA graph scan backward has no matching forward")
            incoming = (
                value for value, active in zip(cotangents, self._output_mask, strict=True) if active
            )
            for destination, source in zip(self._cotangents, incoming, strict=True):
                if source is None:
                    destination.zero_()
                else:
                    destination.copy_(source)
            self._backward_graph.replay()
            gradients = tuple(value.clone() if value is not None else None for value in self._gradients)
            lease.ran_backward = True
            return gradients


def capture_training(function, sample_args, *, warmup_iters=3):
    """Capture a pure tensor callable and its VJP on a side stream, when supported."""
    sample_args = tuple(sample_args)
    if (
        not sample_args
        or not isinstance(sample_args[0], torch.Tensor)
        or not sample_args[0].is_cuda
        or not torch.is_grad_enabled()
        or torch.is_inference_mode_enabled()
        or torch._C._are_functorch_transforms_active()
    ):
        return None
    device = sample_args[0].device
    if not _supported_inputs(sample_args, device) or not any(arg.requires_grad for arg in sample_args):
        return None

    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            return None
        stream = torch.cuda.current_stream(device)
        side_stream = torch.cuda.Stream(device=device)
        side_stream.wait_stream(stream)
        with torch.cuda.stream(side_stream):
            for value in sample_args:
                value.record_stream(side_stream)
            static_inputs = tuple(
                value.detach().clone().requires_grad_(value.requires_grad) for value in sample_args
            )
            input_metadata = tuple(map(_metadata, sample_args))
            grad_inputs = tuple(value for value in static_inputs if value.requires_grad)
            for _ in range(max(1, warmup_iters)):
                outputs = function(*static_inputs)
                output_mask = _output_grad_mask(outputs, device)
                if not any(output_mask):
                    return None
                cotangents = tuple(value.new_zeros(value.shape) for value in outputs if value.requires_grad)
                _gradients(outputs, output_mask, grad_inputs, cotangents)
                del outputs, cotangents

            forward = torch.cuda.CUDAGraph()
            backward = torch.cuda.CUDAGraph()
            pool = torch.cuda.graph_pool_handle()
            with torch.cuda.graph(forward, stream=side_stream, pool=pool):
                outputs = function(*static_inputs)
            if _output_grad_mask(outputs, device) != output_mask:
                return None
            cotangents = tuple(value.new_zeros(value.shape) for value in outputs if value.requires_grad)
            with torch.cuda.graph(backward, stream=side_stream, pool=pool):
                # Keep forward activations live while choosing graph-private
                # addresses so repeated VJPs can read the same forward state.
                input_gradients = iter(_gradients(
                    outputs, output_mask, grad_inputs, cotangents, retain_graph=True,
                ))
                gradients = tuple(
                    next(input_gradients) if value.requires_grad else None for value in static_inputs
                )
        stream.wait_stream(side_stream)

    return TrainingGraph(
        stream, input_metadata, static_inputs,
        tuple(value.detach() for value in outputs), output_mask,
        forward, cotangents, gradients, backward,
    )
