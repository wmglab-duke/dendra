import os

import torch
from torch import nn


class RNGMixin(nn.Module):
    """
    Mixin that provides a per-module RNG (one Generator per device).
    Use self._rng(device) to get the right Generator and pass it to random ops.
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

    # ---------- public API ----------
    def reseed(self, seed: int) -> None:
        """Reset this module's RNGs (CPU + any created device gens)."""
        self._base_seed = int(seed)
        self._cpu_gen.manual_seed(self._base_seed)
        # Derive stable, distinct seeds per device from the base seed
        for dev, gen in self._device_gens.items():
            # Any simple bijection is fine; include device index to avoid collisions
            off = (dev.index or 0) + (hash(dev.type) & 0xFFFF)
            gen.manual_seed(self._base_seed + 0x9E3779B97F4A7C15 * (1 + off))

    def rng_state(self) -> dict:
        """Snapshot current generator states (so sequences continue after save/load)."""
        state = {"cpu": self._cpu_gen.get_state().cpu()}
        for dev, gen in self._device_gens.items():
            state[str(dev)] = gen.get_state().cpu()
        return state

    def set_rng_state(self, state: dict) -> None:
        """Restore generator states."""
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
        # Persist base seed and the live generator states (so sequences resume exactly)
        return {"base_seed": self._base_seed, "rng_state": self.rng_state()}

    def set_extra_state(self, extra_state):
        self._base_seed = int(extra_state.get("base_seed", 0))
        self.reseed(self._base_seed)
        if "rng_state" in extra_state:
            self.set_rng_state(extra_state["rng_state"])

    # ---------- internal helper ----------
    def _rng(self, device: torch.device | str | None) -> torch.Generator:
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
