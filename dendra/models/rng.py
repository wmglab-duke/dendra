"""Per-module random number generation helpers."""

import os
import zlib
from collections.abc import Mapping
from numbers import Integral

import torch
from torch import nn

_GOLDEN_RATIO_64 = 0x9E3779B97F4A7C15
_UINT64_MASK = (1 << 64) - 1
_MIN_MANUAL_SEED = -(1 << 63)


def should_restore_rng(*, policy: str, ignore_rng_on_load: bool = False) -> bool:
    """Choose whether a saved RNG frame owns a restoration boundary.

    Model and cache loads respect ``ignore_rng_on_load``. Runtime checkpoint
    continuation restores the saved stream regardless of that preference.
    This predicate also applies to explicit functional RNG tensor frames.
    """
    if policy not in {"cache", "checkpoint"}:
        raise ValueError("RNG restoration policy must be 'cache' or 'checkpoint'.")
    return policy == "checkpoint" or not ignore_rng_on_load


def _snapshot_rng_lifecycle(module):
    """Retain physical sampler handles for an enclosing checkpoint transaction.

    Module traversal deduplicates shared streams. The existing serialized
    checkpoint owns seed and generator bytes; these records retain only the
    bindings that a successful inner restore can replace or evict.
    """
    return tuple(
        (
            rng,
            rng._cpu_gen,
            dict(rng._device_gens),
            "rng" in vars(rng),
            vars(rng).get("rng"),
        )
        for rng in module.modules()
        if isinstance(rng, RNGMixin)
    )


def _restore_rng_lifecycle(snapshot):
    """Rebind original handles before replaying the preceding checkpoint."""
    for rng, cpu_generator, device_generators, has_active, active in snapshot:
        rng._cpu_gen = cpu_generator
        rng._device_gens = dict(device_generators)
        if has_active:
            rng.rng = active
        else:
            vars(rng).pop("rng", None)


def _device_seed(base_seed: int, device: torch.device) -> int:
    """Derive a stable uint64 seed for a non-CPU device generator."""
    device_type = zlib.crc32(device.type.encode("utf-8"))
    device_index = 0 if device.index is None else int(device.index)
    offset = device_type + device_index
    return (int(base_seed) + _GOLDEN_RATIO_64 * (1 + offset)) & _UINT64_MASK


def _validate_base_seed(seed) -> int:
    """Return a seed accepted by ``torch.Generator.manual_seed``.

    Serialized RNG metadata is deliberately stricter than ``int(seed)``: a
    float, string, or boolean should not silently change the future random
    stream while loading a checkpoint.
    """

    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("RNG base_seed must be an integer.")
    seed = int(seed)
    if seed < _MIN_MANUAL_SEED or seed > _UINT64_MASK:
        raise ValueError(
            "RNG base_seed must be between "
            f"{_MIN_MANUAL_SEED} and {_UINT64_MASK}, inclusive."
        )
    return seed


def _validate_state_tensor(state, device: torch.device) -> torch.Tensor:
    """Clone and validate one tensor-encoded generator state.

    Accelerator states are fully checked when their backend is available.  If
    the checkpoint's device cannot be constructed on this machine, structural
    validation is the most that can be performed; the state remains safe to
    defer until that exact backend is requested.
    """

    if not torch.is_tensor(state):
        raise TypeError(f"RNG state for device {str(device)!r} must be a tensor.")
    if state.dtype != torch.uint8:
        raise TypeError(
            f"RNG state for device {str(device)!r} must have dtype torch.uint8."
        )
    if state.device.type != "cpu":
        raise ValueError(f"RNG state for device {str(device)!r} must be stored on CPU.")
    if state.numel() == 0:
        raise ValueError(f"RNG state for device {str(device)!r} must not be empty.")

    state = state.detach().clone()
    try:
        validator = torch.Generator(device=device)
    except (RuntimeError, TypeError):
        if device.type == "cpu":
            raise
        return state
    try:
        validator.set_state(state)
    except (RuntimeError, TypeError) as error:
        raise RuntimeError(
            f"RNG state for device {str(device)!r} is not a valid generator state."
        ) from error
    return state


def _validate_rng_state(state) -> dict[str, torch.Tensor]:
    """Validate and clone a complete device-to-generator-state mapping.

    A valid snapshot always includes the primary CPU generator.  Device keys
    are normalized to their canonical string representation so aliases cannot
    silently overwrite one another during restoration.
    """

    if not isinstance(state, Mapping):
        raise TypeError("RNG state must be a mapping of devices to tensors.")

    validated = {}
    for key, value in state.items():
        if not isinstance(key, (str, torch.device)):
            raise TypeError("RNG state device keys must be strings or torch.device.")
        try:
            device = torch.device(key)
        except (RuntimeError, TypeError) as error:
            raise ValueError(f"Invalid RNG state device key {key!r}.") from error
        canonical = str(device)
        if canonical in validated:
            raise ValueError(f"Duplicate RNG state for device {canonical!r}.")
        validated[canonical] = _validate_state_tensor(value, device)

    if "cpu" not in validated:
        raise KeyError("RNG state must contain a valid 'cpu' generator state.")
    return validated


def _validate_rng_checkpoint_payload(payload, *, allow_legacy: bool):
    """Validate modern RNG metadata or an explicitly allowed legacy snapshot.

    Modern payloads contain both ``base_seed`` and ``rng_state``.  Early
    runtime-checkpoint payloads stored the raw device-state mapping instead; the
    legacy form is accepted only at boundaries that opt in explicitly.

    Returns
    -------
    tuple[int | None, dict[str, torch.Tensor]]
        The normalized base seed (``None`` for a legacy raw snapshot) and a
        cloned, validated RNG-state mapping.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("RNG checkpoint payload must be a mapping.")

    modern_keys_present = {
        name for name in ("base_seed", "rng_state") if name in payload
    }
    if modern_keys_present:
        missing = {"base_seed", "rng_state"} - modern_keys_present
        if missing:
            raise KeyError(
                "Modern RNG checkpoint payload requires both 'base_seed' and "
                f"'rng_state'; missing={sorted(missing)}."
            )
        return (
            _validate_base_seed(payload["base_seed"]),
            _validate_rng_state(payload["rng_state"]),
        )

    if not allow_legacy:
        raise KeyError("RNG extra state requires 'base_seed' and 'rng_state' entries.")
    return None, _validate_rng_state(payload)


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
        # Checkpoints may contain accelerator generators that cannot be
        # constructed on the machine doing the load. Keep those states on CPU
        # until that exact device is requested later.
        self._pending_device_states: dict[str, torch.Tensor] = {}
        self._ignore_rng_on_load = False

    def ignore_rng_on_load(self, ignore: bool = True) -> None:
        """Toggle RNG restoration for model and initialization-cache loads.

        Parameters
        ----------
        ignore : bool, optional
            If ``True``, RNG state is ignored during :meth:`load_state_dict`.
            Explicit runtime checkpoint restoration retains its saved stream.
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
        # An explicit reseed supersedes every saved generator suffix, including
        # states deferred because their accelerator was unavailable at load time.
        self._pending_device_states = {}
        self._cpu_gen.manual_seed(self._base_seed)
        # Derive stable, distinct seeds per device from the base seed
        for dev, gen in self._device_gens.items():
            gen.manual_seed(_device_seed(self._base_seed, dev))

    def rng_state(self) -> dict:
        """Snapshot the state of every generator.

        Returns
        -------
        dict
            Mapping of device identifier to tensor-encoded RNG state.
        """
        state = {"cpu": self._cpu_gen.get_state().cpu()}
        # ``getattr`` keeps full-module pickles created before deferred states
        # were introduced usable; unpickling does not call ``__init__``.
        pending_states = getattr(self, "_pending_device_states", {})
        for dev, pending_state in pending_states.items():
            state[dev] = pending_state.detach().cpu().clone()
        for dev, gen in self._device_gens.items():
            state[str(dev)] = gen.get_state().cpu()
        return state

    def _apply_validated_rng_state(self, state: dict[str, torch.Tensor]) -> None:
        """Apply a state mapping already checked by :func:`_validate_rng_state`."""

        prepared_generators = {}
        pending_states = {}
        for key, device_state in state.items():
            if key == "cpu":
                continue
            device = torch.device(key)
            generator = self._device_gens.get(device)
            if generator is not None:
                continue
            try:
                generator = torch.Generator(device=device)
            except (RuntimeError, TypeError):
                pending_states[key] = device_state.clone()
                continue
            # Prepare new generators completely before mutating any live state.
            generator.set_state(device_state)
            prepared_generators[device] = generator

        self._cpu_gen.set_state(state["cpu"])
        for key, device_state in state.items():
            if key == "cpu":
                continue
            device = torch.device(key)
            generator = self._device_gens.get(device)
            if generator is not None:
                generator.set_state(device_state)

        self._device_gens = {
            device: generator
            for device, generator in self._device_gens.items()
            if str(device) in state
        }
        self._device_gens.update(prepared_generators)
        active = getattr(self, "rng", None)
        if (
            active is not None
            and active is not self._cpu_gen
            and all(active is not generator for generator in self._device_gens.values())
        ):
            self.rng = None
        # ``state`` is a complete snapshot. Do not retain deferred states from
        # an older snapshot that are absent from this one.
        self._pending_device_states = pending_states

    def set_rng_state(self, state: Mapping) -> None:
        """Restore the state of every generator.

        Parameters
        ----------
        state : Mapping
            Mapping created by :meth:`rng_state`.
        """
        validated = _validate_rng_state(state)
        previous_state = self.rng_state()
        previous_generators = self._device_gens.copy()
        missing = object()
        previous_active = getattr(self, "rng", missing)
        try:
            self._apply_validated_rng_state(validated)
        except Exception:
            self._device_gens = previous_generators
            try:
                self._apply_validated_rng_state(previous_state)
                if previous_active is not missing:
                    self.rng = previous_active
            except Exception as rollback_error:
                raise RuntimeError(
                    "RNG-state restoration failed and rollback could not recover "
                    "the previous generator state."
                ) from rollback_error
            raise

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
        extra_state : Mapping
            Serialized dictionary containing RNG state and base seed.
        """
        self.restore_rng_checkpoint(extra_state, policy="cache")

    def restore_rng_checkpoint(
        self, payload, *, policy: str = "checkpoint", allow_legacy: bool = False
    ) -> None:
        """Restore seed and generator suffix under an explicit load policy.

        ``cache`` honors :meth:`ignore_rng_on_load` before inspecting the
        payload. ``checkpoint`` restores exact continuation independently of
        that flag. With ``allow_legacy=True``, a raw device-state mapping keeps
        the current base seed. Failed restores recover the preceding frame.
        """
        if not should_restore_rng(
            policy=policy,
            ignore_rng_on_load=getattr(self, "_ignore_rng_on_load", False),
        ):
            return
        base_seed, rng_state = _validate_rng_checkpoint_payload(
            payload, allow_legacy=allow_legacy
        )
        previous_seed = self._base_seed
        previous_state = self.rng_state()
        previous_generators = self._device_gens.copy()
        missing = object()
        previous_active = getattr(self, "rng", missing)
        try:
            if base_seed is not None:
                self.reseed(base_seed)
            self._apply_validated_rng_state(rng_state)
        except Exception:
            self._device_gens = previous_generators
            self._base_seed = previous_seed
            try:
                self._apply_validated_rng_state(previous_state)
                if previous_active is not missing:
                    self.rng = previous_active
            except Exception as rollback_error:
                raise RuntimeError(
                    "RNG checkpoint restoration failed and rollback could not "
                    "recover the previous generator state."
                ) from rollback_error
            raise

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
            device_key = str(dev)
            if not hasattr(self, "_pending_device_states"):
                self._pending_device_states = {}
            pending_state = self._pending_device_states.get(device_key)
            if pending_state is None:
                g.manual_seed(_device_seed(self._base_seed, dev))
            else:
                # Only consume the pending state after it has been accepted by
                # the newly available backend.
                g.set_state(pending_state)
            self._device_gens[dev] = g
            if pending_state is not None:
                self._pending_device_states.pop(device_key)
        return g


class RNGModule(RNGMixin):
    def __init__(self, seed, shape_p, shape_f):
        super().__init__(seed)
        self.shape_p = shape_p
        self.shape_f = shape_f
        self.rng = None

    @classmethod
    def from_checkpoint(cls, payload, *, shape_p=(), shape_f=()):
        """Construct an isolated host sampler from explicit generator metadata.

        Sampling the returned module changes neither the source checkpoint nor
        any live model RNG. Device generators retain their saved byte states;
        missing devices derive their stream from the saved base seed.
        """
        base_seed, state = _validate_rng_checkpoint_payload(payload, allow_legacy=False)
        module = cls(base_seed, shape_p=shape_p, shape_f=shape_f)
        module._apply_validated_rng_state(state)
        return module

    def reseed_for_generation(self, generation):
        """Reset once at the first initialization request sharing this RNG.

        A missing generation denotes manual sampling and does not reset the
        stream. Other declarations sharing this RNG continue in Native order.
        """
        if generation is None:
            return False
        previous = getattr(self, "_random_parameter_reseed_generation", None)
        if previous is generation or previous == generation:
            return False
        self.reset()
        self._random_parameter_reseed_generation = generation
        return True

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
        n = torch.as_tensor(n, device=self.rng.device)
        p = torch.as_tensor(p, device=self.rng.device)
        dtype = torch.promote_types(n.dtype, p.dtype)
        if not dtype.is_floating_point:
            dtype = torch.get_default_dtype()
        n = n.to(dtype=dtype).expand(shape)
        p = p.to(dtype=dtype).expand(shape)
        return torch.binomial(n, p, generator=self.rng)

    def reset(self):
        """Reseed the RNG to its initial state."""
        self.reseed(self._base_seed)
        if self.rng is None:
            self.init("cpu")
