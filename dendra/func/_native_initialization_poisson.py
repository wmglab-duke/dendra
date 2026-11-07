"""Native initialization Poisson invocations over the existing shared recipes."""

from collections import Counter

from dendra.models.stim.waveform._poisson import (
    _is_default_poisson_generator,
    defer_poisson_generator_restore,
    native_poisson_schedule_from_draws,
    restore_pending_poisson_rng,
)

from ._native_runtime_draws import _native_poisson_plan
from ._types import StimulationDraw


class NativePoissonInitialization:
    def __deepcopy__(self, memo):
        return self

    def __init__(self, population, packet, *, capture, cached):
        self.capture, self._supplied, self._skipped = capture, packet, cached
        self.plan = (
            None
            if cached
            else _native_poisson_plan(population, None, _initialization=True)
        )
        self.position, self.positions = 0, Counter()
        self.draws = {}
        self.updates = []
        self.enabled = (
            self.plan is not None and self.plan.has_initialization and not cached
        )
        if self.plan is not None:
            self.draws = self.plan._placeholder_draws()
        if not capture:
            if not self.enabled:
                if packet is not None:
                    raise ValueError(
                        "This initialization does not consume Poisson draws"
                    )
            else:
                if packet is None:
                    raise ValueError("Initialization requires explicit Poisson draws")
                self.plan.validate(packet)
                self.updates = [
                    (generator, packet.rng[name].detach().cpu().clone())
                    for name, generator in self.plan.streams.items()
                ]

    def slots(self):
        if self.plan is None:
            return
        seen = set()
        for entry in self.plan.entries:
            if id(entry.source) not in seen:
                yield entry.source, "_native_initialization_draws", self
                seen.add(id(entry.source))

    def poisson_schedule(self, waveform, generator):
        if not self.enabled or self.position >= len(self.plan._initialization_entries):
            raise ValueError("Poisson initialization exceeded its invocation profile")
        entry = self.plan._initialization_entries[self.position]
        if entry.source is not waveform and vars(waveform).get(
            "_native_initialization_origin"
        ) != id(entry.source):
            raise ValueError("Poisson initialization invocation order changed")
        index = self.positions[entry.key]
        if self.capture:
            restore_pending_poisson_rng(entry.source)
            draw = self.plan._sample_entry(entry, self.plan.streams[entry.stream])
            self.plan._assign_draw(self.draws, entry, index, draw)
        else:
            values = self._supplied.draws[entry.key]
            draw = values if self.plan._capacities[entry.key] == 1 else values[index]
        self.positions[entry.key] += 1
        self.position += 1
        reference = next(waveform.waveform.parameters(), None)
        if reference is None:
            reference = next(waveform.waveform.buffers(), waveform._spike_times)
        schedule = native_poisson_schedule_from_draws(
            waveform, draw, dtype=reference.dtype, device=reference.device
        )
        entry.source._spike_times = schedule
        return schedule.reshape(-1)

    def skip(self):
        self._skipped = True

    def finish(self):
        if (
            not self._skipped
            and self.enabled
            and self.position != len(self.plan._initialization_entries)
        ):
            raise ValueError(
                "Poisson initialization did not consume its invocation profile"
            )

    def packet(self):
        self.finish()
        if self._skipped or not self.enabled:
            return None
        return StimulationDraw(
            self.draws,
            {
                name: generator.get_state().clone()
                for name, generator in self.plan.streams.items()
            },
        )

    def publish(self, *, model_generators=()):
        if self._skipped:
            return
        published = set(model_generators)
        deferred = set()
        for generator, state in self.updates:
            if _is_default_poisson_generator(generator):
                nodes = [
                    entry.source
                    for entry in self.plan.entries
                    if entry.source.generator is generator
                ]
                defer_poisson_generator_restore(nodes, state)
                deferred.add(id(generator))
            else:
                generator.set_state(state)
                published.add(id(generator))
        if self.plan is not None:
            for entry in self.plan.entries:
                holder = vars(entry.source).get("_functional_poisson_rng")
                if (
                    holder is not None
                    and holder.generator is entry.source.generator
                    and id(holder.generator) in published
                    and id(holder.generator) not in deferred
                ):
                    holder.state = None
