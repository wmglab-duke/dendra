import linecache
from typing import List

from tqdm.auto import tqdm
import torch, torch.nn as nn, math

from axonml.helpers import DEBUG
from axonml.models.callbacks import CallbackList, Callback

from .population import assemble_global_adjacencies
from ..mechanisms.compilers.utils import indent


advance_v_template = """
self.{pop}.v = self.{pop}.integrator._step_no_intra(self.{pop}.v, self.dt, self.{pop}.temp_c)
"""


net_receive_template = """
self.{pop}.mech.net_receive({pop}_weights)
"""


compute_spikes_template = """
spikes = self.check_active(torch.cat([{all_v}], dim=-1))
"""


compute_weights = indent("weights = self.synapses(spikes)", 2)


split_weights_template = """
{split_weights} = torch.split(weights, self.n_per_pop, dim=-1)
"""


init_v_template = """
self.{pop}.integrator.init_v(self.{pop})
"""


init_mech_template = """
self.{pop}.integrator.mech.initialize(self.{pop}.v, self.{pop}.v_init_c, self.{pop}.temp_c)
"""


init_integrator_template = """
self.{pop}.integrator.initialize(self.{pop}, self.dt)
"""


def advance_v(populations):
    """
    Generate the code to advance the voltage of each population.
    """
    advance_v = []
    for pop in populations:
        advance_v.append(advance_v_template.format(pop=pop))
    return indent("\n".join(advance_v), 2)


def compute_spikes(populations):
    """
    Generate the code to compute the spikes of each population.
    """
    all_v = []
    for pop in populations:
        all_v.append(f"self.{pop}.v")
    compute_spikes = compute_spikes_template.format(all_v=", ".join(all_v))
    return indent(compute_spikes, 2)


def split_weights(populations):
    """
    Generate the code to split the weights for each population.
    """
    split_weights = []
    for pop in populations:
        split_weights.append(f"{pop}_weights")
    split_weights = split_weights_template.format(
        split_weights=", ".join(split_weights)
    )
    return indent(split_weights, 2)


def net_receive(populations):
    """
    Generate the code to advance the voltage of each population.
    """
    net_receive = []
    for pop in populations:
        net_receive.append(net_receive_template.format(pop=pop))
    return indent("\n".join(net_receive), 2)


def init_v(populations):
    """
    Generate the code to advance the voltage of each population.
    """
    init_v = []
    for pop in populations:
        init_v.append(init_v_template.format(pop=pop))
    return indent("\n".join(init_v), 2)


def init_mech(populations):
    """
    Generate the code to advance the voltage of each population.
    """
    init_mech = []
    for pop in populations:
        init_mech.append(init_mech_template.format(pop=pop))
    return indent("\n".join(init_mech), 2)


def init_integrator(populations):
    """
    Generate the code to advance the voltage of each population.
    """
    init_integrator = []
    for pop in populations:
        init_integrator.append(init_integrator_template.format(pop=pop))
    return indent("\n".join(init_integrator), 2)


template = """
class _network(torch.nn.Module):
    def __init__(self, populations, synapses, check_active, dt):
        super().__init__()
        self.synapses = synapses
        self.check_active = check_active
        for name, pop in populations.items():
            setattr(self, name, pop)
        self.register_buffer("dt", torch.as_tensor(dt))
        self.n_per_pop: List[int] = [pop.v.shape[-1] for pop in populations.values()]

    def step(self):
{advance_v}
{compute_spikes}
{compute_weights}
{split_weights}
{net_receive}

    @torch.jit.ignore
    def initialize(self):
        self.synapses.initialize()
        self.check_active.above_threshold.zero_()
{init_v}
{init_integrator}
{init_mech}
"""


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

    __constants__ = ["has_intrinsic", "P_greater_than_N"]

    def __init__(
        self,
        W: torch.Tensor,  # (N,C,C)
        M: torch.Tensor,  # (N,C,C)
        D: torch.Tensor,  # (N,C,C)
        P: int,
        dt: float,
        rate_int: torch.Tensor,     # (N,C) or scalar
        w_int: torch.Tensor,        # (N,C) or scalar
        d_int: int = 1,
    ):
        super().__init__()

        self.W = torch.nn.Parameter(W, requires_grad=False)  # (N,C,C)
        self.register_buffer("D", D)  # (N,C,C)

        W_by_delay = weights_by_delay(W, D)  # (D,N,C,C)

        # parameters
        self.register_buffer("M", M)  # (D,N,C,C)
        self.D_max = W_by_delay.size(0)
        self.D_buf = self.D_max + 1 
        self.N = W_by_delay.size(1)
        self.C = W_by_delay.size(2)
        self.P = int(P)

        self.P_greater_than_N = self.P >= self.N

        if False:
            Wd = W_by_delay.unsqueeze(2).contiguous()  # (D,N,1,C,C)
        else:
            Wd = W_by_delay.contiguous()
        
        self.register_buffer("Wd", Wd)

        # intrinsic parameters
        self.dt = float(dt)
        self.register_buffer(
            "rate_int",
            torch.as_tensor(rate_int, dtype=W_by_delay.dtype).expand(self.N, self.C).clone(),
        )  # (N,C)
        self.register_buffer(
            "w_int",
            torch.as_tensor(w_int, dtype=W_by_delay.dtype).expand(self.N, self.C).clone(),
        )  # (N,C)
        self.d_int = int(d_int)

        self.has_intrinsic = bool(torch.any(self.rate_int > 0))

        # circular delay line  (D, N, P, C)
        self.register_buffer(
            "queue",
            torch.zeros(
                self.D_buf,
                self.N,
                self.P,
                self.C,
                dtype=W_by_delay.dtype,
                device=W_by_delay.device,
            ),
        )
        self.register_buffer("head", torch.zeros((), dtype=torch.long))

        # pre-compute tensor of slot offsets  [1,2,…,D_max]
        self.register_buffer(
            "delay_offsets",
            torch.arange(self.D_max, dtype=torch.long, device=W_by_delay.device) + 1
        )

        self.eval()

    @torch.jit.export
    def initialize(self) -> None:
        """
        Initialize the synapse state.
        """
        self.queue.zero_().detach_()
        self.head.zero_().detach_()
        if self.training:
            self.Wd.detach_()
            self.Wd = weights_by_delay(self.W, self.D)  # (D,N,C,C)

    @torch.jit.export
    def float16(self) -> None:
        """
        Convert the synapse weights to float16.
        """
        self.Wd = self.Wd.to(torch.float16)
        self.M = self.M.to(torch.float16)
        self.D = self.D.to(torch.float16)
        self.W = self.W.to(torch.float16)
        self.rate_int = self.rate_int.to(torch.float16)
        self.w_int = self.w_int.to(torch.float16)
        self.queue = self.queue.to(torch.float16)

    @torch.jit.export
    def set_weights(self, W: torch.Tensor, D: torch.Tensor) -> None:
        """
        Set weights and delays for all synapses.
        """
        W = W * self.M
        W_by_delay = weights_by_delay(W, D)  # (D,N,C,C)
        self.Wd.copy_(W_by_delay)  # (D,N,C,C)

    # ------------------------------------------------------------------
    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        spikes : (N,P,C)
        returns: (N,P,C)
        """

        # ── 1. advance head, deliver, clear ─────────────────
        h = (self.head + 1) % self.D_buf              # NEW head slot
        out = self.queue[h].clone()                   # deliver
        self.queue[h] = 0.0                           # clear for reuse
        self.head = h                                 # save back

        # ── 2. schedule recurrent events ─────────────———
        if True:
            # --- path A: batched GEMM (einsum)  fast when P is long -------
            contrib = torch.einsum(           # (D,N,P,C)
                'npj,dnji->dnpi',
                spikes.to(self.Wd.dtype),
                self.Wd
            )

        #  scatter-add into the ring buffer (no loop)
        slots = (h + self.delay_offsets) % self.D_buf         # (D,)
        self.queue.view(self.D_buf, -1).index_add_(
            0,
            slots,
            contrib.reshape(self.D_max, -1)
        )

        # ── 3. intrinsic Poisson drive ─────────────———
        if self.has_intrinsic:
            p   = 1.0 - torch.exp(-self.rate_int * self.dt)   # (N,C)
            p   = p.unsqueeze(1)                              # (N,1,C)
            w_i = self.w_int.unsqueeze(1)                     # (N,1,C)

            rand = torch.rand(self.N, self.P, self.C,
                              dtype=out.dtype, device=out.device)
            intrinsic = (rand < p).to(out.dtype) * w_i        # (N,P,C)

            slot_int = (h + self.d_int) % self.D_buf
            self.queue[slot_int] += intrinsic

        return out


class CheckActive(torch.jit.ScriptModule):
    def __init__(self, N, P, C, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold
        self.register_buffer("above_threshold", torch.zeros((N, P, C), dtype=torch.bool))

    @torch.jit.script_method
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        v : (N,P,C)  float
        """
        # check if above threshold
        above_threshold = v >= self.threshold
        active = above_threshold & ~self.above_threshold
        self.above_threshold = above_threshold

        return active


class Network(torch.jit.ScriptModule):
    """
    Network class for AxonML.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.dt : float = net.dt.item()
        self.t_ind : int = 0

    @property
    def t(self):
        """
        Current time in the network.
        """
        return self.t_ind * self.dt

    @property
    def synapses(self):
        """
        Synapse object.
        """
        return self.net.synapses

    @torch.jit.script_method
    def step(self) -> None:
        """
        Step the network forward in time.
        """
        self.net.step()

    def run(self, tstop: float, progressbar=True, reinit=False, callbacks=None) -> None:
        """
        Run the network for a given time.
        """
        dt = self.dt
        n_steps = int(math.ceil(tstop / dt))

        if reinit:
            self.net.initialize()
            self.t_ind = 0

        if callbacks is not None:
            for callback in callbacks:
                callback.dt = dt

        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(callbacks)

        callbacks.pre_loop_hook(self)

        if progressbar:
            progressbar = tqdm(total=n_steps, desc=f"{self.t:.3f} ms")

        for _ in range(n_steps):
            self.step()
            callbacks.post_step_hook(self)
            self.t_ind += 1

            if progressbar:
                progressbar.update(1)
                if self.t_ind % 100 == 0:
                    progressbar.set_description(f"{self.t:.3f} ms")

        if progressbar:
            progressbar.close()
        
        callbacks.post_loop_hook(self)


def build_network(
    populations,
    connections,
    dt,
    N=1,
    P=1,
    max_delay=None,
    intrinsic_rate=0.0,
    intrinsic_weight=0.0,
    threshold=0.0,
):
    W, M, D = assemble_global_adjacencies(populations, connections, dt, max_delay)
    n = W.shape[0]

    W = W.view(1, n, n).expand(N, n, n).clone()
    M = M.view(1, n, n).expand(N, n, n).clone()
    D = D.view(1, n, n).expand(N, n, n).clone()

    populations = {pop.name: pop.build(N, P) for pop in populations}

    synapses = VariableDelaySynapse(W, M, D, P, dt, intrinsic_rate, intrinsic_weight)

    forward = template.format(
        advance_v=advance_v(populations),
        compute_spikes=compute_spikes(populations),
        compute_weights=compute_weights,
        split_weights=split_weights(populations),
        net_receive=net_receive(populations),
        init_integrator=init_integrator(populations),
        init_v=init_v(populations),
        init_mech=init_mech(populations),
    )

    if DEBUG: print(forward)

    # Create the network class
    filename = f"<_network_template>"
    code = compile(forward, filename, "exec")
    exec(code)

    lines = [line + "\n" for line in forward.splitlines()]
    linecache.cache[filename] = (len(forward), None, lines, filename)

    check_active = CheckActive(N, P, n, threshold)

    n = locals()["_network"](
        populations, synapses, check_active, dt
    )

    net = Network(n)

    return net
