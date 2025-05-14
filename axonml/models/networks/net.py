import linecache
from typing import List

from tqdm.auto import tqdm
import torch, torch.nn as nn, math

from axonml.helpers import DEBUG
from axonml.models.callbacks import CallbackList, Callback

from .population import assemble_global_adjacencies, blocks_by_projection
from ..mechanisms.compilers.utils import indent


advance_v_template = """
self.{pop}.v = self.{pop}.integrator._step_no_intra(self.{pop}.v, self.dt, self.{pop}.temp_c)
"""


net_receive_template = """
self.{pop}.mech.net_receive(weights_{pop})
"""


compute_spikes_template = """
spikes_{pop} = self.check_active_{pop}(self.{pop}.v)
"""


compute_weights_template = """
{pre}_{post}_weights = self.synapses_{pre}_{post}(spikes_{pre})
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
    compute_spikes = []
    for pop in populations:
        compute_spikes.append(compute_spikes_template.format(pop=pop))
    return indent("\n".join(compute_spikes), 2)


def compute_weights(connections):
    """
    Generate the code to compute the weights for each connection.
    """
    compute_weights = []
    for c in connections:
        compute_weights.append(compute_weights_template.format(pre=c['pre'], post=c['post']))
    return indent("\n".join(compute_weights), 2)


def total_weights(populations, connections, intrinsic):
    """
    Generate the code to compute the total weights for each population.
    """
    total_weights = []
    for pop in populations:
        relevant_weights = []
        for c in connections:
            if c['post'] == pop:
                relevant_weights.append(f"{c['pre']}_{c['post']}_weights")
        if not relevant_weights:
            if not pop in intrinsic:
                continue
            total_weights.append(f"weights_{pop} = self.{pop}_intrinsic()")
            continue
        relevant_weights_sum = f" + ".join(relevant_weights)
        if pop in intrinsic:
            relevant_weights_sum = f"self.{pop}_intrinsic() + {relevant_weights_sum}"
        total_weights.append(f"weights_{pop} = {relevant_weights_sum}")
    return indent("\n".join(total_weights), 2)


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


def synapses_init(synapses):
    """
    Generate the code to advance the voltage of each population.
    """
    synapses_init = []
    for s in synapses:
        synapses_init.append(f"self.synapses_{s}.initialize()")
    return indent("\n".join(synapses_init), 2)


def check_active_init(populations):
    """
    Generate the code to advance the voltage of each population.
    """
    check_active_init = []
    for pop in populations:
        check_active_init.append(f"self.check_active_{pop}.above_threshold.zero_()")
    return indent("\n".join(check_active_init), 2)


template = """
class _network(torch.nn.Module):
    def __init__(self, populations, synapses, check_active, intrinsic, dt):
        super().__init__()

        for name, synapse in synapses.items():
            setattr(self, f"synapses_{{name}}", synapse)

        for name, check in check_active.items():
            setattr(self, f"check_active_{{name}}", check)

        for name, pop in populations.items():
            setattr(self, name, pop)

        if intrinsic is not None:
            for name, intrinsic in intrinsic.items():
                setattr(self, f"{{name}}_intrinsic", intrinsic)

        self.register_buffer("dt", torch.as_tensor(dt))
        self.n_per_pop: List[int] = [pop.v.shape[-1] for pop in populations.values()]

    def step(self):
{advance_v}
{compute_spikes}
{compute_weights}
{total_weights}
{net_receive}

    @torch.jit.ignore
    def initialize(self):
{synapses_init}
{check_active_init}
{init_v}
{init_integrator}
{init_mech}
"""


def weights_by_delay_(weight: torch.Tensor, delay: torch.Tensor) -> torch.Tensor:
    D_max = int(delay.max().item())
    Wd: List[torch.Tensor] = []
    for d in range(1, D_max + 1):
        Wd.append(weight * (delay == d))
    return torch.stack(Wd, dim=0)  # (D,N,C,C)


def weights_by_delay(
    W: torch.Tensor,          # (N, n_pre, n_post)  float / half
    D: torch.Tensor           # (N, n_pre, n_post)  int64
) -> torch.Tensor:
    """
    Rearranges a weight matrix `W` and its corresponding delay matrix `D`
    into a delay-indexed tensor suitable for VariableDelaySynapse.

    Returns
    -------
    W_by_D : (D_max, N, n_pre, n_post)
             W_by_D[d-1, n, i, j] holds W[n,i,j] **iff** D[n,i,j] == d,
             otherwise 0.  (Delay values must be >= 1.)
    """
    if W.shape != D.shape:
        raise ValueError("W and D must have the same shape")

    if not torch.is_floating_point(W):
        raise TypeError("W must be a floating-point tensor")
    if D.dtype != torch.int64:
        raise TypeError("D must be int64 (torch.long)")

    D_max = int(D.max().item())        # largest delay present
    if D_max < 1:
        raise ValueError("all delays must be ≥ 1")

    # Broadcast masks & zero-out in one pass per delay
    blocks = []
    for d in range(1, D_max + 1):
        mask = (D == d)
        blocks.append(W.masked_fill(~mask, 0.0))

    # → (D_max, N, n_pre, n_post)
    return torch.stack(blocks, dim=0)


class VariableDelaySynapse(torch.jit.ScriptModule):
    """
    Forward signature
    -----------------
    spikes : (N, P, C)  bool / 0-1
    returns: (N, P, C)  float
    """

    def __init__(
        self,
        W: torch.Tensor,  # (N,C,C)
        M: torch.Tensor,  # (N,C,C)
        D: torch.Tensor,  # (N,C,C)
        P: int,
        dt: float,
    ):
        super().__init__()

        self.W = torch.nn.Parameter(W, requires_grad=False)  # (N,C,C)
        self.register_buffer("D", D)  # (N,C,C)

        W_by_delay = weights_by_delay(W, D)  # (D,N,C,C)

        # parameters
        self.register_buffer("M", M)         # (D,N,C,C)
        self.D_max = W_by_delay.size(0)
        self.D_buf = self.D_max + 1 
        self.N = W_by_delay.size(1)
        self.C = W_by_delay.size(3)
        self.P = int(P)

        Wd = W_by_delay.contiguous()
        
        self.register_buffer("Wd", Wd)
    
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
    def bfloat16(self) -> None:
        """
        Convert the synapse weights to bfloat16.
        """
        self.Wd = self.Wd.to(torch.bfloat16)
        self.M = self.M.to(torch.bfloat16)
        self.D = self.D.to(torch.bfloat16)
        self.W = self.W.to(torch.bfloat16)
        self.rate_int = self.rate_int.to(torch.bfloat16)
        self.w_int = self.w_int.to(torch.bfloat16)
        self.queue = self.queue.to(torch.bfloat16)

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
        spikes : (N,P,C_pre)
        returns: (N,P,C_post)
        """

        # ── 1. advance head, deliver, clear ─────────────────
        h = (self.head + 1) % self.D_buf              # NEW head slot
        out = self.queue[h].clone()                   # deliver
        self.queue[h] = 0.0                           # clear for reuse
        self.head = h                                 # save back

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

        return out


class Intrinsic(torch.jit.ScriptModule):

    def __init__(
        self,
        N: int,
        P: int,
        C: int,
        dt: float,
        rate_int: torch.Tensor,  # (N,C) or scalar
        w_int: torch.Tensor,     # (N,C) or scalar
    ):
        super().__init__()
        self.N = N
        self.P = P
        self.C = C

        self.dt = float(dt)
        self.register_buffer(
            "rate_int",
            torch.as_tensor(rate_int, dtype=torch.float).expand(self.N, self.C).clone(),
        )  # (N,C)
        self.register_buffer(
            "w_int",
            torch.as_tensor(w_int, dtype=torch.float).expand(self.N, self.C).clone(),
        )  # (N,C)

    def forward(self):
        """
        intrinsic : (N,P,C)
        """
        p = 1.0 - torch.exp(-self.rate_int * self.dt)
        p = p.unsqueeze(1)  # (N,1,C)
        w_i = self.w_int.unsqueeze(1)  # (N,1,C)
        rand = torch.rand(self.N, self.P, self.C,
                          dtype=self.w_int.dtype, device=self.w_int.device)
        intrinsic = (rand < p).to(self.w_int.dtype) * w_i  # (N,P,C)
        return intrinsic


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
        self.eval()

    @property
    def t(self):
        """
        Current time in the network.
        """
        return self.t_ind * self.dt

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

        with torch.set_grad_enabled(self.training):

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
    threshold=0.0,
    intrinsic=None,
):
    n = {
        pop.name: pop.n for pop in populations
    }

    if not isinstance(threshold, dict):
        threshold = {pop.name: threshold for pop in populations}

    check_active = {
        pop.name: CheckActive(N, P, pop.n, threshold[pop.name]) for pop in populations
    }

    synapse_data = blocks_by_projection(connections, n, dt, max_delay)

    synapses = {}

    for name, (W, D) in synapse_data.items():
        M = (W != 0).float()
        W = torch.tile(W, (N, 1, 1))  # (N, n_pre, n_post)
        D = torch.tile(D, (N, 1, 1))  # (N, n_pre, n_post)
        M = torch.tile(M, (N, 1, 1))  # (N, n_pre, n_post)
        synapses[name] = VariableDelaySynapse(
            W=W,
            M=M,
            D=D,
            P=P,
            dt=dt,
        )

    if intrinsic is not None:
        intrinsic = {
            name: Intrinsic(N, P, n[name], dt, r, w) for name, (r, w) in intrinsic.items()
        }

    populations = {pop.name: pop.build(N, P) for pop in populations}

    forward = template.format(
        advance_v=advance_v(populations),
        compute_spikes=compute_spikes(populations),
        compute_weights=compute_weights(connections),
        total_weights=total_weights(populations, connections, intrinsic),
        net_receive=net_receive(populations),
        synapses_init=synapses_init(synapses),
        check_active_init=check_active_init(populations),
        init_v=init_v(populations),
        init_integrator=init_integrator(populations),
        init_mech=init_mech(populations),
    )

    if DEBUG: print(forward)

    # Create the network class
    filename = f"<_network_template>"
    code = compile(forward, filename, "exec")
    exec(code)

    lines = [line + "\n" for line in forward.splitlines()]
    linecache.cache[filename] = (len(forward), None, lines, filename)

    n = locals()["_network"](
        populations, synapses, check_active, intrinsic, dt
    )

    net = Network(n)

    return net
