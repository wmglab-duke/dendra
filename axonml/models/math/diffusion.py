from axonml.models.math.dct import idct1, dct1
import torch
import math


@torch.jit.script
def diffuse_step_neumann_dct1(
    state: torch.Tensor,
    dt: float,
    D: float,
    L: float,
):
    """
    Diffuse a state using a DCT-I based Neumann boundary condition

    :param state: the state to diffuse
    :param dt: the time step
    :param D: the diffusion coefficient
    :param L: the length of the domain
    :return: the state after diffusion
    """
    device = state.device
    _, _, N = state.shape
    M_hat = dct1(state)
    k = torch.arange(N, device=device, dtype=state.dtype)
    freq_sq = (k * math.pi / L) ** 2
    decay = torch.exp(-D * dt * freq_sq)
    M_hat = M_hat * decay
    m_new = idct1(M_hat)

    return m_new