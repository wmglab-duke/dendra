"""Utilities for reshaping tensors across batch dimensions."""


def expand_and_reshape(tensor, batched_shape, n_batch_dims, target_shape):
    """Expand a tensor across batch axes and reshape to a target layout.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor to broadcast.
    batched_shape : Sequence[int]
        Expanded shape including the newly introduced batch axes.
    n_batch_dims : int
        Number of leading batch axes to prepend before expansion.
    target_shape : Sequence[int]
        Final shape to reshape the expanded tensor into.

    Returns
    -------
    torch.Tensor
        Contiguous tensor with shape ``target_shape``.
    """
    for _ in range(n_batch_dims):
        tensor = tensor.unsqueeze(0)
    tensor = tensor.expand(batched_shape).reshape(*target_shape).contiguous()
    return tensor
