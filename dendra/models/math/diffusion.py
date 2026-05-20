import math

import torch

from dendra.models.math.dct import dct1, idct1


@torch.jit.script
def diffuse_step_neumann_dct1(
    state: torch.Tensor, dt: float, D: float, L: float, n: int
) -> torch.Tensor:
    """
    Diffuse a state using a DCT-I based Neumann boundary condition.

    This function performs a diffusion step on the input state tensor using the Discrete Cosine Transform Type-I (DCT-I)
    to handle Neumann boundary conditions. The diffusion process is governed by the diffusion coefficient and the time step.

    Args:
        state (torch.Tensor): The state tensor to diffuse. It is expected to have a shape of (batch_size, channels, N).
        dt (float): The time step for the diffusion process.
        D (float): The diffusion coefficient.
        L (float): The length of the domain.
        n (int): The number of points in the domain.

    Returns:
        torch.Tensor: The state tensor after the diffusion step.
    """
    M_hat = dct1(state)
    k = torch.arange(n, device=state.device, dtype=state.dtype)
    freq_sq = (k * math.pi / L) ** 2
    decay = torch.exp(-D * dt * freq_sq)
    M_hat = M_hat * decay
    m_new = idct1(M_hat, n)

    return m_new
