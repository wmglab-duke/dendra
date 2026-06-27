import torch


def dct1_rfft_impl(x):
    return torch.view_as_real(torch.fft.rfft(x, dim=1))


def dct1(x: torch.Tensor):
    """Discrete Cosine Transform, Type I.

    Parameters
    ----------
    x : torch.Tensor
        The input signal.

    Returns
    -------
    torch.Tensor
        The DCT-I of the signal over the last dimension.
    """
    # x = x.squeeze()
    x = torch.cat([x, x.flip([1])[:, 1:-1]], dim=1)

    return dct1_rfft_impl(x)[:, :, 0].unsqueeze(1)


def idct1(x: torch.Tensor, n: int):
    """The inverse of DCT-I, which is just a scaled DCT-I.

    Our definition of idct1 is such that idct1(dct1(x)) == x.

    Parameters
    ----------
    x : torch.Tensor
        The input signal.
    n : int
        Size of the original signal.

    Returns
    -------
    torch.Tensor
        The inverse DCT-I of the signal over the last dimension.
    """
    return dct1(x) / (2 * (n - 1))
