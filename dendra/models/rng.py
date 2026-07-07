"""Per-module random number generation helpers."""

import os

import torch
from torch import nn


class RNGMixin(nn.Module):
    """Mixin providing per-device random number generators.

    Parameters
    ----------
    seed : int or None, optional
        Base seed used to initialise all derived :class:`torch.Generator`
        instances. A random seed is chosen when ``None``.
    """

    def __init__(self, seed: int | None = None):
        super().__init__()
        # Store a reproducible base seed and serialize it with the module
        self._base_seed = int(
            seed if seed is not None else int.from_bytes(os.urandom(8), "little")
        )
        # One CPU generator (always exists) + lazily-created device-specific generators
        self._cpu_gen = torch.Generator(device="cpu").manual_seed(self._base_seed)
        self._device_gens: dict[torch.device, torch.Generator] = {}
        self._ignore_rng_on_load = False

    def ignore_rng_on_load(self, ignore: bool = True) -> None:
        """Toggle restoration of RNG state when loading a checkpoint.

        Parameters
        ----------
        ignore : bool, optional
            If ``True``, RNG state is ignored during :meth:`load_state_dict`.
        """
        self._ignore_rng_on_load = ignore

    # ---------- public API ----------
    def reseed(self, seed: int) -> None:
        """Reset all generators with a new base seed.

        Parameters
        ----------
        seed : int
            Base seed used to initialise every generator.
        """
        self._base_seed = int(seed)
        self._cpu_gen.manual_seed(self._base_seed)
        # Derive stable, distinct seeds per device from the base seed
        for dev, gen in self._device_gens.items():
            # Any simple bijection is fine; include device index to avoid collisions
            off = (dev.index or 0) + (hash(dev.type) & 0xFFFF)
            gen.manual_seed(self._base_seed + 0x9E3779B97F4A7C15 * (1 + off))

    def rng_state(self) -> dict:
        """Snapshot the state of every generator.

        Returns
        -------
        dict
            Mapping of device identifier to tensor-encoded RNG state.
        """
        state = {"cpu": self._cpu_gen.get_state().cpu()}
        for dev, gen in self._device_gens.items():
            state[str(dev)] = gen.get_state().cpu()
        return state

    def set_rng_state(self, state: dict) -> None:
        """Restore the state of every generator.

        Parameters
        ----------
        state : dict
            Mapping created by :meth:`rng_state`.
        """
        if "cpu" in state:
            self._cpu_gen.set_state(state["cpu"].cpu())
        for k, v in state.items():
            if k == "cpu":
                continue
            dev = torch.device(k)
            g = self._device_gens.get(dev)
            if g is None:
                g = torch.Generator(device=dev)
                self._device_gens[dev] = g
            g.set_state(v.to("cpu"))  # states are stored on CPU tensors

    # ---------- serialization hooks ----------
    def get_extra_state(self):
        """Return RNG metadata for PyTorch serialization.

        Returns
        -------
        dict
            Dictionary containing the base seed and generator states.
        """
        return {"base_seed": self._base_seed, "rng_state": self.rng_state()}

    def set_extra_state(self, extra_state):
        """Load RNG metadata produced by :meth:`get_extra_state`.

        Parameters
        ----------
        extra_state : dict
            Serialized dictionary containing RNG state and base seed.
        """
        if self._ignore_rng_on_load:
            return
        self._base_seed = int(extra_state.get("base_seed", 0))
        self.reseed(self._base_seed)
        if "rng_state" in extra_state:
            self.set_rng_state(extra_state["rng_state"])

    # ---------- internal helper ----------
    def _rng(self, device: torch.device | str | None) -> torch.Generator:
        """Return the generator associated with ``device``.

        Parameters
        ----------
        device : torch.device or str or None
            Device identifier. ``None`` selects the CPU generator.

        Returns
        -------
        torch.Generator
            Generator tied to the specified device.
        """
        dev = torch.device(device) if device is not None else torch.device("cpu")
        if dev.type == "cpu":
            return self._cpu_gen
        # Lazily create a generator for this device the first time it's needed
        g = self._device_gens.get(dev)
        if g is None:
            g = torch.Generator(device=dev)
            off = (dev.index or 0) + (hash(dev.type) & 0xFFFF)
            g.manual_seed(self._base_seed + 0x9E3779B97F4A7C15 * (1 + off))
            self._device_gens[dev] = g
        return g


class RNGModule(RNGMixin):
    def __init__(self, seed, shape_p, shape_f):
        super().__init__(seed)
        self.shape_p = shape_p
        self.shape_f = shape_f
        self.rng = None

    def init(self, device: torch.device | str | None = None):
        """Initialize the RNG for a specific device."""
        self.rng = self._rng(device)

    def _generator_for(self, device=None):
        if device is None:
            if self.rng is None:
                self.init("cpu")
            return self.rng
        dev = torch.device(device)
        if self.rng is None or self.rng.device != dev:
            try:
                self.init(dev)
            except (RuntimeError, TypeError):
                self.init("cpu")
        return self.rng

    def rand(self, shape=None, *, device=None, dtype=None):
        """Generate uniform random numbers in [0, 1)."""
        if shape is None:
            shape = self.shape_f
        gen = self._generator_for(device)
        dev = gen.device if device is None else torch.device(device)
        try:
            return torch.rand(shape, generator=gen, device=dev, dtype=dtype)
        except (RuntimeError, TypeError):
            cpu_gen = self._rng("cpu")
            return torch.rand(shape, generator=cpu_gen, dtype=dtype).to(dev)

    def randn(self, shape=None, *, device=None, dtype=None):
        """Generate standard normal random numbers."""
        if shape is None:
            shape = self.shape_f
        gen = self._generator_for(device)
        dev = gen.device if device is None else torch.device(device)
        try:
            return torch.randn(shape, generator=gen, device=dev, dtype=dtype)
        except (RuntimeError, TypeError):
            cpu_gen = self._rng("cpu")
            return torch.randn(shape, generator=cpu_gen, dtype=dtype).to(dev)

    def rand_like(self, tensor):
        """Generate uniform random numbers matching ``tensor``."""
        return self.rand(tuple(tensor.shape), device=tensor.device, dtype=tensor.dtype)

    def randn_like(self, tensor):
        """Generate normal random numbers matching ``tensor``."""
        return self.randn(tuple(tensor.shape), device=tensor.device, dtype=tensor.dtype)

    def binomial(self, n, p, shape=None):
        """Generate binomial random numbers."""
        if shape is None:
            shape = self.shape_f
        if self.rng is None:
            self.init("cpu")
        n = torch.as_tensor(n, device=self.rng.device).expand(shape)
        p = torch.as_tensor(p, device=self.rng.device).expand(shape)
        return torch.binomial(n, p, generator=self.rng)

    def reset(self):
        """Reseed the RNG to its initial state."""
        if self.rng is None:
            self.init("cpu")
        self.rng.manual_seed(self._base_seed)
