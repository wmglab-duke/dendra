from typing import Tuple, Optional
import torch


class NetStim(torch.nn.Module):
    """
    A PyTorch implementation of NEURON's NetStim-like spike generator.

    This class generates spike events according to a stochastic process with
    configurable timing parameters, similar to NEURON's NetStim mechanism.

    Parameters
    ----------
    interval : float
        Mean inter-spike interval in ms.
    start : float, optional
        Start time (ms) after which synapses can begin spiking. Default is 0.0.
    noise : float, optional
        Controls randomness of intervals, between 0 and 1.
        0 = deterministic intervals, 1 = fully random (exponential distribution).
        Default is 0.0.
    max_spikes : int, optional
        Maximum number of spikes each synapse can deliver. Default is 1e9.
    seed : int, optional
        Seed for reproducible random number generation. If None,
        uses non-deterministic seeding. Default is None.

    Attributes
    ----------
    shape : tuple of int
        Shape of the output spike tensor (n_ax, n_comp).
    interval : float
        Mean inter-spike interval in ms.
    start : float
        Start time in ms.
    noise : float
        Randomness factor (0-1).
    max_spikes : int
        Maximum number of spikes per synapse.
    seed : int or None
        Random seed.
    next_spike_time : torch.Tensor
        Tensor storing the next spike time for each synapse.
    spike_counts : torch.Tensor
        Tensor counting how many spikes each synapse has emitted.
    """

    __constants__ = ["seed"]

    def __init__(
        self,
        N: int = 1,
        interval: float | list[float] = 10.0,
        start: float | list[float] = 0.0,
        noise: float | list[float] = 0.0,
        max_spikes: int | list[int] = 1e9,
        seed: Optional[int] = None,
    ):
        """
        Initialize the NetStim spike generator.

        Parameters
        ----------
        N : int, optional
            Number of independent event generators. Default is 1.
        interval : float | list[float], optional
            Mean inter-spike interval in ms. Default is 10.0.
        start : float | list[float], optional
            Start time (ms) after which synapses can begin spiking. Default is 0.0.
        noise : float | list[float], optional
            Controls randomness of intervals, between 0 and 1. Default is 0.0.
        max_spikes : int | list[int], optional
            Maximum number of spikes each synapse can deliver. Default is 1e9.
        seed : int, optional
            Seed for reproducible random number generation. If None,
            uses non-deterministic seeding. Default is None.
        """
        super().__init__()
        # Store parameters
        self.N = N

        if noise < 0:
            noise = 0.0
        if noise > 1:
            noise = 1.0

        self.register_buffer("noise", torch.as_tensor(noise, dtype=torch.float32))
        self.register_buffer("interval", torch.as_tensor(interval, dtype=torch.float32))
        self.register_buffer("start", torch.as_tensor(start, dtype=torch.float32))
        self.register_buffer(
            "max_spikes", torch.as_tensor(max_spikes, dtype=torch.long)
        )

        self.seed: Optional[int] = seed

        # each NetStim gets its own Generator
        self._seeder = torch.Generator()
        self._rng = torch.Generator().manual_seed(self._seeder.seed())
        self.register_buffer("next_spike_time", torch.zeros(self.N))
        self.register_buffer("spike_counts", torch.zeros(self.N, dtype=torch.long))
        self.register_buffer("spikes", torch.zeros(self.N, dtype=self.dtype()))

    def device(self):
        """
        Get the device on which the module's tensors reside.

        Returns
        -------
        torch.device
            The computation device (CPU or CUDA).
        """
        return self.next_spike_time.device

    def dtype(self):
        """
        Get the data type of the module's tensors.

        Returns
        -------
        torch.dtype
            The data type used for computations.
        """
        return self.next_spike_time.dtype

    def init_rng(self):
        """
        Initialize the random number generator.

        This method ensures the RNG is on the correct device and
        sets the seed if specified.
        """
        if self._rng.device != self.device():
            self._rng = torch.Generator(device=self.device()).manual_seed(
                self._seeder.seed()
            )
        if self.seed is not None:
            self._rng.manual_seed(self.seed)

    def initialize(self):
        """
        Initialize the spike generator for a given shape.

        Parameters
        ----------
        n_ax : int
            Number of axons (rows in the output tensor).
        n_comp : int
            Number of nodes (columns in the output tensor).

        Returns
        -------
        self : NetStim
            Returns self for method chaining.

        Notes
        -----
        This method initializes next_spike_time for each synapse:
        - If noise=0, first spike will occur exactly at start time
        - If noise>0, first spike times follow start + exponential(noise*interval)
        """
        self.init_rng()

        device = self.device()
        dtype = self.dtype()

        # Initialize next_spike_time:
        # If noise=0, the first spike time = start (no randomization).
        # Otherwise, draw from an exponential distribution with mean = noise * interval,
        # so that E[next_spike_time] = start + noise * interval.
        self.next_spike_time = self.start

        if self.noise > 0:
            # Draw from Exp(1 / (noise*interval)) so that mean = noise*interval
            randvals = torch.rand(
                (self.N,), generator=self._rng, device=device, dtype=dtype
            )
            # Exponential variable with mean = noise*interval => -log(U) * (noise*interval)
            init_offsets = -(self.noise * self.interval) * torch.log(randvals)
            self.next_spike_time += init_offsets

        # spike_counts: how many spikes each synapse has emitted
        self.spike_counts = torch.zeros(self.N, device=device, dtype=torch.long)
        return self

    def forward(self, t):
        """
        Check which synapses spike at the given time and update their states.

        Parameters
        ----------
        t : float
            Current simulation time in ms.

        Returns
        -------
        torch.Tensor
            Binary tensor of shape (n_ax, n_comp) where True/1.0 indicates
            a spike at this time step.

        Notes
        -----
        This method:
        1. Identifies which synapses spike at time t
        2. Increments the spike count for those synapses
        3. Computes their next spike time based on the noise parameter
        4. Disables synapses that have reached max_spikes
        """
        with torch.no_grad():
            # Identify which synapses are still allowed to spike
            can_spike = self.spike_counts < self.max_spikes

            # Determine which synapses spike exactly at this time
            # (i.e., t >= next_spike_time)
            is_spiking_now = can_spike & (self.next_spike_time <= t)

            # Create the output mask (1 = spike, 0 = no spike)
            self.spikes = is_spiking_now.to(self.dtype())

            # Get the indices of synapses that spike
            spiking_indices = is_spiking_now != 0

            # Increment the spike count for those synapses
            torch.masked_add_(self.spike_counts, spiking_indices, 1)

            # Compute the next inter-spike interval for those synapses:
            # if noise=0, interval is constant
            # if noise=1, intervals are purely exponential with mean=interval
            # for partial noise: next_interval = interval*(1 - noise) + interval*noise*Exp(1/interval).
            # Exponential random deviates (for partial or full noise)
            exp_rand = -torch.log(
                torch.rand(
                    (self.N,),
                    generator=self._rng,
                    device=self.device(),
                    dtype=self.dtype(),
                )
            )
            # Weighted combination of deterministic + random
            next_interval = (
                self.interval * (1 - self.noise) + self.interval * self.noise * exp_rand
            )

            # Update next_spike_time for those synapses that spiked
            torch.masked_add_(
                self.next_spike_time, is_spiking_now, next_interval[is_spiking_now]
            )

            # Any synapse that has just reached its maximum number of spikes
            # will no longer spike (set next_spike_time = inf)
            done = self.spike_counts >= self.max_spikes

            torch.masked_fill_(self.next_spike_time, done, float("inf"))
