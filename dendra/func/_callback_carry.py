"""Static, built-in-only callback carry partitioning for compiled loops.

LFP lead fields are explicit runtime tensors, but never change during a run.
They are lifted loop operands rather than recurrent carry: packing them each
timestep would copy and save contact-by-compartment arrays unnecessarily.
Custom callback carry is always recurrent, regardless of its apparent values.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._callbacks import _BoundLFP, _callback_implementation


@dataclass(frozen=True)
class _CallbackCarryPartition:
    names: tuple[str, ...]
    invariant_names: tuple[str, ...]

    @classmethod
    def from_plans(cls, plans):
        plans = () if plans is None else plans
        return cls(
            tuple(plan.name for plan in plans),
            tuple(
                plan.name
                for plan in plans
                if type(_callback_implementation(plan.callback)) is _BoundLFP
            ),
        )

    def split(self, carries):
        if not self.invariant_names:
            return carries, {}
        return (
            {
                name: carries[name]
                for name in self.names
                if name not in self.invariant_names
            },
            {name: carries[name] for name in self.invariant_names},
        )

    def merge(self, recurrent, invariant):
        if not self.invariant_names:
            return recurrent
        return {
            name: invariant[name] if name in self.invariant_names else recurrent[name]
            for name in self.names
        }

    def restore_invariants(self, updated, original):
        """Restore invariants from this invocation's original callback carry."""
        return self.merge(updated, self.split(original)[1])


__all__ = []
