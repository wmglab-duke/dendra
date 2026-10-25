"""Cold coordinates for initialization-retained effective operands."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class _RetainedMaterializationSlot:
    path: str
    value_key: str
    active_key: str
    baseline_key: str
    shape: tuple
    dtype: object
    device: object
    prepared_path: tuple | None


@dataclass(frozen=True)
class _RetainedMaterializationInventory:
    """Own ordered slot metadata, independently of tensor values and graphs."""

    slots: tuple = ()
    by_path: dict = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "by_path", {slot.path: slot for slot in self.slots})

    @classmethod
    def from_specs(cls, specs, prepared_paths):
        return cls(
            tuple(
                _RetainedMaterializationSlot(
                    path=path,
                    value_key=name,
                    active_key=f"materializations.active.{path}",
                    baseline_key=f"materializations.baseline.{path}",
                    shape=shape,
                    dtype=dtype,
                    device=device,
                    prepared_path=prepared_paths.get(path),
                )
                for name, path, shape, dtype, device in specs
            )
        )
