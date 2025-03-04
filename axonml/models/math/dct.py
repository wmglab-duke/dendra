import torch

def dct1_rfft_impl(x):
    return torch.view_as_real(torch.fft.rfft(x, dim=1))


@torch.jit.script
def dct1(x: torch.Tensor):
    """
    Discrete Cosine Transform, Type I

    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x = x.squeeze()
    x = torch.cat([x, x.flip([1])[:, 1:-1]], dim=1)

    return dct1_rfft_impl(x)[:, :, 0].unsqueeze(1)


@torch.jit.script
def idct1(x: torch.Tensor, n: int):
    """
    The inverse of DCT-I, which is just a scaled DCT-I

    Our definition if idct1 is such that idct1(dct1(x)) == x

    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    return dct1(x) / (2 * (n - 1))