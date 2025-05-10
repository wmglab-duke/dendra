import torch, torch.nn as nn, math
from typing import List

from .population import assemble_global_adjacencies
from ..mechanisms.compilers.utils import indent


advance_v_template = """
self.{pop}.v = self.{pop}.integrator._step_no_intra(self.{pop}.v, self.dt, self.{pop}.temp_c)
"""


net_receive_template = """
self.{pop}.mech.net_receive({pop}_weights)
"""


def advance_v(populations):
    """
    Generate the code to advance the voltage of each population.
    """
    advance_v = []
    for pop in populations:
        advance_v.append(advance_v_template.format(pop=pop))
    return indent("\n".join(advance_v), 2)


def net_receive(populations):
    """
    Generate the code to advance the voltage of each population.
    """
    net_receive = []
    for pop in populations:
        net_receive.append(net_receive_template.format(pop=pop))
    return indent("\n".join(net_receive), 2)


template = """
class _network(torch.jit.ScriptModule):
    def __init__(self, populations, synapses, check_active, dt):
        super().__init__()
        self.synapses = synapses
        for name, pop in populations.items():
            setattr(self, name, pop)
        self.register_buffer("dt", torch.as_tensor(dt))
        self.n_per_pop: List[int] = [pop.n for pop in populations.values()]

    @torch.ijt.script_method
    def step(self):
        {advance_v}
        {compute_spikes}
        {compute_weights}
        {split_weights}
        {net_receive}
"""


@torch.no_grad()
def weights_by_delay(weight: torch.Tensor, delay: torch.Tensor) -> torch.Tensor:
    D_max = int(delay.max().item())
    Wd: List[torch.Tensor] = []
    for d in range(1, D_max + 1):
        Wd.append(weight * (delay == d))
    return torch.stack(Wd, dim=0)  # (D,N,C,C)


class VariableDelaySynapse(torch.jit.ScriptModule):
    """
    Forward signature
    -----------------
    spikes : (N, P, C)  bool / 0-1
    returns: (N, P, C)  float
    """

    __constants__ = ["has_intrinsic"]

    def __init__(
        self,
        W: torch.Tensor,  # (N,C,C)
        M: torch.Tensor,  # (N,C,C)
        D: torch.Tensor,  # (N,C,C)
        P: int,
        dt: float,
        rate_int: torch.Tensor,  # (N,C) or scalar
        w_int: torch.Tensor,  # (N,C) or scalar
        d_int: int = 1,
    ):
        super().__init__()

        W_by_delay = weights_by_delay(W, D)  # (D,N,C,C)

        # parameters
        self.register_buffer("M", M)  # (D,N,C,C)
        self.register_buffer("W", W_by_delay)  # (D,N,C,C)
        self.D_max = W_by_delay.size(0)
        self.N = W_by_delay.size(1)
        self.C = W_by_delay.size(2)
        self.P = int(P)

        # intrinsic parameters
        self.dt = float(dt)
        self.register_buffer(
            "rate_int",
            torch.as_tensor(rate_int, dtype=W_by_delay.dtype).expand(self.N, self.C),
        )  # (N,C)
        self.register_buffer(
            "w_int",
            torch.as_tensor(w_int, dtype=W_by_delay.dtype).expand(self.N, self.C),
        )  # (N,C)
        self.d_int = int(d_int)

        self.has_intrinsic = (self.rate_int > 0).any()

        # circular delay line  (D, N, P, C)
        self.register_buffer(
            "queue",
            torch.zeros(
                self.D_max,
                self.N,
                self.P,
                self.C,
                dtype=W_by_delay.dtype,
                device=W_by_delay.device,
            ),
        )
        self.register_buffer("head", torch.zeros((), dtype=torch.long))

    @torch.jit.export
    def set_weights(self, W: torch.Tensor, D: torch.Tensor) -> None:
        """
        Set weights and delays for all synapses.
        """
        W = W * self.M
        W_by_delay = weights_by_delay(W, D)  # (D,N,C,C)
        self.W.copy_(W_by_delay)  # (D,N,C,C)

    # ------------------------------------------------------------------
    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        spikes : (N,P,C)  bool / 0-1
        """
        # deliver events due now
        out = self.queue[self.head].clone()  # (N,P,C)
        self.queue[self.head].zero_()

        # schedule future events
        #     spikes: (N,P,C)  → (N,P,1,C)   (for matmul)
        s = spikes.to(self.W.dtype).unsqueeze(-2)  # (N,P,1,C)

        for d in range(self.D_max):
            #   W[d] : (N,C,C) → (N,1,C,C)  (broadcast along P)
            contrib = torch.matmul(s, self.W[d].unsqueeze(1))  # (N,P,1,C)
            contrib = contrib.squeeze(-2)  # (N,P,C)

            slot = (self.head + d) % self.D_max
            self.queue[slot] += contrib

        # intrinsic Poisson drive
        if self.has_intrinsic:
            # rate_int, w_int : (N,C) → (N,1,C) → broadcast (N,P,C)
            p = 1.0 - torch.exp(-self.rate_int * self.dt)  # (N,C)
            p = p.unsqueeze(1)  # (N,1,C)
            w_i = self.w_int.unsqueeze(1)  # (N,1,C)

            rand = torch.rand(
                self.N, self.P, self.C, dtype=out.dtype, device=out.device
            )
            intrinsic = (rand < p).to(out.dtype) * w_i  # (N,P,C)

            slot_int = (self.head + self.d_int - 1) % self.D_max
            self.queue[slot_int] += intrinsic

        # advance time
        self.head = (self.head + 1) % self.D_max
        return out


def build_network(
    populations,
    connections,
    dt,
    N=1,
    P=1,
    max_delay=None,
    intrinsic_rate=0.0,
    intrinsic_weight=0.0,
):
    W, M, D = assemble_global_adjacencies(populations, connections, dt, max_delay)
    n = W.shape[0]

    W = W.view(1, n, n).expand(N, n, n).copy()
    M = M.view(1, n, n).expand(N, n, n).copy()
    D = D.view(1, n, n).expand(N, n, n).copy()

    populations = {pop.name: pop.build(N, P) for pop in populations}

    synapses = VariableDelaySynapse(W, M, D, P, dt, intrinsic_rate, intrinsic_weight)
