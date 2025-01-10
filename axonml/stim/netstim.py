from typing import Tuple, Optional
import torch


class NetStim(torch.jit.ScriptModule):
    """
    A PyTorch implementation of NEURON's NetStim-like spike generator.

    Args:
        shape (tuple): The shape (m, n) indicating the number of synapses.
        interval (float): Mean inter-spike interval in ms.
        start (float): Start time (ms) after which synapses can begin spiking.
        max_spikes (int): Maximum number of spikes each synapse can deliver.
        noise (float): 0 <= noise <= 1, controls how randomly the intervals vary.
        device (str): The PyTorch device (e.g., 'cpu' or 'cuda').
        dtype (torch.dtype): Floating-point precision.
        seed (int or None): Seed for reproducible random number generation. If None,
                            it will use the global RNG state.
    """

    __constants__ = ["seed"]

    def __init__(
        self,
        interval: float,
        max_spikes: int,
        start: float = 0.0,
        noise: float = 0.0,
        shape=None,
        device="cpu",
        seed: Optional[int] = None,
    ):
        super().__init__()
        # Store parameters
        self.shape: Tuple[int, int] = shape
        self.interval: float = interval
        self.start: float = start
        self.max_spikes: int = max_spikes
        self.noise: float = noise
        self.seed: Optional[int] = seed

        # If a seed is provided, create a Generator and seed it for reproducible behavior
        self._rng = torch.Generator(device=device)

        self.register_buffer('next_spike_time', torch.zeros(1, device=device))
        self.register_buffer('spike_counts', torch.zeros(1, device=device, dtype=torch.long))

    def device(self):
        return self.next_spike_time.device
    
    def dtype(self):
        return self.next_spike_time.dtype

    def init_rng(self):
        if self._rng.device != self.device():
            self._rng = torch.Generator(device=self.device())
        if self.seed is not None:
            self._rng.manual_seed(self.seed)

    def init(self, n_ax, n_node):
        self.shape = (n_ax, n_node)
        self.init_rng()

        device = self.device()
        dtype = self.dtype()

        # Initialize next_spike_time:
        # If noise=0, the first spike time = start (no randomization).
        # Otherwise, draw from an exponential distribution with mean = noise * interval,
        # so that E[next_spike_time] = start + noise * interval.
        self.next_spike_time = torch.full(
            self.shape, self.start, device=device, dtype=dtype
        )
        if self.noise > 0:
            # Draw from Exp(1 / (noise*interval)) so that mean = noise*interval
            randvals = torch.rand(
                self.shape, generator=self._rng, device=device, dtype=dtype
            )
            # Exponential variable with mean = noise*interval => -log(U) * (noise*interval)
            init_offsets = -(self.noise * self.interval) * torch.log(randvals)
            self.next_spike_time += init_offsets

        # spike_counts: how many spikes each synapse has emitted
        self.spike_counts = torch.zeros(
            self.shape, device=device, dtype=torch.long
        )
        return self

    @torch.jit.script_method
    def event(self, t: float):
        """
        Returns an m x n binary tensor indicating which synapses spike at time `t`.
        Updates internal states for spiking synapses (increments spike count,
        and schedules their next spike time).

        Args:
            t (float): Current simulation time in ms.

        Returns:
            A binary tensor of shape (m, n) indicating which synapses spike.
        """

        # Identify which synapses are still allowed to spike
        can_spike = self.spike_counts < self.max_spikes

        # Determine which synapses spike exactly at this time
        # (i.e., t >= next_spike_time)
        is_spiking_now = can_spike & (t >= self.next_spike_time)

        # Create the output mask (1 = spike, 0 = no spike)
        output = is_spiking_now.to(self.dtype())

        # Get the indices of synapses that spike
        spiking_indices = is_spiking_now.nonzero()
        r_inds, c_inds = spiking_indices.unbind(1)

        # Increment the spike count for those synapses
        self.spike_counts[r_inds, c_inds] += 1

        # Compute the next inter-spike interval for those synapses:
        # if noise=0, interval is constant
        # if noise=1, intervals are purely exponential with mean=interval
        # for partial noise: next_interval = interval*(1 - noise) + interval*noise*Exp(1/interval).
        num_spiking = r_inds.numel()
        if num_spiking > 0:
            # Exponential random deviates (for partial or full noise)
            exp_rand = -torch.log(
                torch.rand(
                    num_spiking,
                    generator=self._rng,
                    device=self.device(),
                    dtype=self.dtype(),
                )
            )
            # Weighted combination of deterministic + random
            next_interval = (
                self.interval * (1 - self.noise) + self.interval * self.noise * exp_rand
            )

            self.next_spike_time[r_inds, c_inds] += next_interval

        # Any synapse that has just reached its maximum number of spikes
        # will no longer spike (set next_spike_time = inf)
        done_r_indices, done_c_indices = (self.spike_counts >= self.max_spikes).nonzero().unbind(1)
        self.next_spike_time[done_r_indices, done_c_indices] = float("inf")

        return output

    def __call__(self, t: float):
        return self.event(t)


if __name__ == "__main__":
    # Example usage:

    # Suppose you have 3x4 synapses
    m, n = 3, 4

    # Create a NetStim with:
    # - Mean interval of 10 ms
    # - Max 5 spikes per synapse
    # - Start time at 5 ms
    # - noise = 0.5 (partially stochastic intervals)
    # - seed = 123 for reproducible random intervals
    netstim = NetStim(
        shape=(m, n),
        interval=10.0,
        start=5.0,
        max_spikes=5,
        noise=0.5,
        device="cpu",
        dtype=torch.float,
        seed=123,
    )

    # Simulate from t=0 to t=50 ms in 1 ms steps
    for t in range(51):
        events = netstim.event(t)
        if events.sum() > 0:
            print(f"Time={t} ms, spikes:\n{events}")
