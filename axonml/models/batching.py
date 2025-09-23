def expand_and_reshape(tensor, batched_shape, n_batch_dims, target_shape):
    for _ in range(n_batch_dims):
        tensor = tensor.unsqueeze(0)
    tensor = tensor.expand(batched_shape).reshape(*target_shape).contiguous()
    return tensor
