import numbers
from typing import Dict, Tuple

import torch

from axonml.units import um
from axonml.models.integrators import bwd_euler_sc


def torch_isscalar(x) -> bool:
    """
    True for Python scalars (int, float, bool, complex, etc.)
    and for 0-dim (scalar) torch.Tensors.
    """
    # 1) plain Python scalar?
    if isinstance(x, numbers.Number):
        return True

    # 2) torch scalar tensor?
    if torch.is_tensor(x) and x.dim() == 0:
        return True

    return False


class Population:
    def __init__(self, name, base_model, diam=[10.0 * um], L=10.0 * um, temp=37.0, v_init=-70.0):
        self.name = name
        self.base_model = base_model
        self.diam = diam
        self.L = L
        self.temp = temp
        self.v_init = v_init
        self.n = len(diam)

    def build(self, N=1, P=1, **kwargs):
        """
        Build the population model.

        Parameters
        ----------
        N : int
            Number of axons.
        P : int
            Number of compartments per axon.
        kwargs : dict
            Additional parameters for the model.

        Returns
        -------
        Axon
            The built axon model.
        """
        if self.n == 1:
            diameters = self.diam[0] * torch.ones((N, P, 1))
        else:
            diameters = torch.as_tensor(self.diam, dtype=torch.float)
            diameters = diameters.view(1, 1, self.n).expand(N, P, self.n)

        return _single_compartment(
            N, 
            P,
            self.n, 
            self.base_model, 
            diameters, 
            L=self.L, 
            temp=self.temp, 
            v_init=self.v_init, 
            **kwargs
        )


def _single_compartment(N, P, C, model, diameters, L=10 * um, **kwargs):
    """
    Create a single compartment model for the given diameters.

    Parameters
    ----------
    model : Axon
        The axon model to be used.
    diameters : array_like
        Diameters of the axons in μm. Can be a single value, list, or tensor.

    Returns
    -------
    Axon
        A new axon model with a single compartment.
    """
    kwargs["L"] = L
    kwargs["dx"] = L
    kwargs["integrator"] = bwd_euler_sc(N=N, P=P, C=C)
    m = model(diameters, **kwargs)
    return m


def _make_mask(n_pre: int, n_post: int, p) -> torch.Tensor:
    if torch_isscalar(p):
        if not (0.0 <= p <= 1.0):
            raise ValueError("Probability must be in [0,1].")
        return torch.rand(n_pre, n_post) < p

    p = torch.astensor(p, dtype=float)
    if p.shape != (n_pre, n_post):
        raise ValueError("Probability/mask shape mismatch.")
    if p.dtype == bool:
        return p.copy()
    return torch.rand(n_pre, n_post) < p


def connect(pre: str, post: str, p, weight, delay):
    """
    Creates one projection  (`pre` → `post`).

    Parameters
    ----------
    pre, post : populations
    p         : scalar in [0,1]           OR bool/float mask (n_pre x n_post)
    weight    : scalar or ndarray         (broadcasted to mask's shape)
    delay     : scalar or ndarray         (broadcasted to mask's shape)

    Returns
    -------
    dict describing this projection; collect these in a list
    pass to `assemble_global_adjacencies`.
    """
    n_pre, n_post = pre.n, post.n

    mask = _make_mask(n_pre, n_post, p)  # bool matrix
    weight = torch.as_tensor(weight)
    delay = torch.as_tensor(delay)
    weight_arr = torch.broadcast_to(weight, mask.shape).to(float)
    delay_arr = torch.broadcast_to(delay, mask.shape).to(float)

    return {
        "pre": pre.name,
        "post": post.name,
        "mask": mask,
        "weight": weight_arr,
        "delay": delay_arr,
    }


def assemble_global_adjacencies(populations, connections, dt, max_delay=None):
    """
    Parameters
    ----------
    connections : list returned by repeated calls to `connect`.

    Returns
    -------
    weight_mat, delay_mat : (N x N) dense float64 arrays
        - row  = presynaptic global ID
        - col  = postsynaptic global ID
    """

    N = sum([pop.n for pop in populations])  # total number of cells

    # Pre-compute slice indices for fast look-ups
    _offsets: Dict[str, Tuple[int, int]] = {}
    start = 0
    for pop in populations:
        _offsets[pop.name] = (start, start + pop.n)  # (inclusive, exclusive)
        start += pop.n

    W = torch.zeros((N, N), dtype=torch.float)
    D = torch.zeros((N, N), dtype=torch.float)

    for c in connections:
        pre0, pre1 = _offsets[c["pre"]]  # slice of pre-cells
        post0, post1 = _offsets[c["post"]]  # slice of post-cells

        # write directly into the global matrices
        mask = c["mask"]
        W_block = c["weight"] * mask
        D_block = c["delay"] * mask

        W[pre0:pre1, post0:post1] = W_block
        D[pre0:pre1, post0:post1] = D_block

    if max_delay is not None:
        if max_delay < 0:
            raise ValueError("max_delay must be non-negative.")
        if max_delay < dt:
            raise ValueError("max_delay must be greater than dt.")
        max_delay = int(max_delay / dt)

    D = torch.ceil(D / dt).to(int).clamp(min=1, max=max_delay)
    M = (W != 0).float()

    return W, M, D
