"""Tensor RNG frames and isolated host samplers shared by functional adapters."""

import torch

from dendra.models.rng import RNGModule


def _seed_tensor(seed, device):
    return torch.tensor(
        list((int(seed) % (1 << 64)).to_bytes(8, "little")),
        dtype=torch.uint8,
        device=device,
    )


def _rng_values(rng, device):
    extra = rng if isinstance(rng, dict) else rng.get_extra_state()
    device = torch.device(device)
    if device.type != "cpu" and str(device) not in extra["rng_state"]:
        # A cold accelerator stream can have only its unindexed route so far.
        # Tensor sampling uses the receiving device's exact indexed route. Seed
        # that lazy route on an isolated sampler so the carried schema is fixed
        # before the first draw, without changing native generator ownership.
        sampler = RNGModule.from_checkpoint(extra)
        sampler._rng(device)
        extra = sampler.get_extra_state()
    return {
        "seed": _seed_tensor(extra["base_seed"], device),
        "negative": torch.tensor(
            extra["base_seed"] < 0, device=device, dtype=torch.bool
        ),
        **{
            key: value.detach().to(device).clone()
            for key, value in extra["rng_state"].items()
        },
    }


def _rng_extra(values):
    seed = int.from_bytes(bytes(values["seed"].detach().cpu().tolist()), "little")
    if bool(values["negative"].detach().cpu()):
        seed -= 1 << 64
    return {
        "base_seed": seed,
        "rng_state": {
            key: value.detach().cpu().clone()
            for key, value in values.items()
            if key not in {"seed", "negative"}
        },
    }


def _sampler_from_frame(frame):
    """Restore one fresh sampler without changing its caller's stream boundary."""
    return RNGModule.from_checkpoint(_rng_extra(frame))
