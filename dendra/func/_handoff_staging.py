"""Private native staging with coherent caller-owned stimulation bindings.

The execution plans remain immutable templates. Staging copies only the native
graph and represented external objects, then lends existing readers a shallow
binding projection. Original source validation remains authoritative; none of
the projections is published or returned through a public API.
"""

from __future__ import annotations

import copy

import torch


class _StagingGraph(torch.nn.Module):
    def __init__(self, target, extras):
        super().__init__()
        # Populate only the disposable registries. Ordinary registration could
        # invoke global Torch hooks on these still-live authored modules.
        self._modules["native"] = target
        # Register external modules so the established execution copier also
        # detaches their buffers and authored non-leaf caches in its shared memo.
        waveforms = torch.nn.ModuleList()
        waveforms._modules.update(
            (str(index), waveform)
            for index, waveform in enumerate(
                waveform for extra in extras for waveform in extra._waveforms._sources
            )
        )
        self._modules["waveforms"] = waveforms
        self.contacts = tuple(extra._contact_sources for extra in extras)
        self.pairs = tuple(extra._pairs for extra in extras)


def _project_population(plan, waveforms, contacts, pairs):
    projected = copy.copy(plan)
    # Source guards belong to the admitted live plan, not the private clone's
    # tensor/module identities. Target compatibility still runs normally.
    projected._validate_source = plan._validate_source
    projected._extra = copy.copy(plan._extra)
    projected._extra._waveforms = copy.copy(plan._extra._waveforms)
    projected._extra._waveforms._sources = tuple(waveforms)
    projected._extra._contact_sources = contacts
    projected._extra._pairs = pairs
    if plan._poisson_plan is not None:
        projected._poisson_plan = copy.copy(plan._poisson_plan)
        projected._poisson_plan.validate_source = plan._poisson_plan.validate_source
        projected._poisson_plan._extra_sources = tuple(waveforms)
        projected._stimulation_rng_plan = copy.copy(plan._stimulation_rng_plan)
        projected._stimulation_rng_plan.validate_source = (
            plan._stimulation_rng_plan.validate_source
        )
        projected._stimulation_rng_plan.functional = projected
        projected._stimulation_rng_plan.poisson = projected._poisson_plan
    return projected


def stage_native_handoff(functional, target):
    """Return one detached native clone and its private source-reader projection.

    Every represented extra field, temporal tensor and Waveform participates in
    the model's copy memo. Thus aliases to native parameters, buffers, modules
    and generators survive isolated validation without making external-only
    values writable native handoff destinations. Execution metadata and tensor
    schemas are reused; no lowering, initialization or sampling occurs here.
    """
    from ._network import FunctionalNetwork
    from ._population import _deepcopy_execution_module

    network = isinstance(functional, FunctionalNetwork)
    populations = tuple(functional.populations.values()) if network else (functional,)
    if not any(plan._extra.enabled for plan in populations):
        return _deepcopy_execution_module(target), functional

    graph = _deepcopy_execution_module(
        _StagingGraph(target, tuple(plan._extra for plan in populations))
    )
    waveforms, offset = tuple(graph.waveforms), 0
    projected_populations = []
    for index, plan in enumerate(populations):
        end = offset + len(plan._extra._waveforms._sources)
        projected_populations.append(
            _project_population(
                plan, waveforms[offset:end], graph.contacts[index], graph.pairs[index]
            )
        )
        offset = end
    if not network:
        return graph.native, projected_populations[0]

    projected = copy.copy(functional)
    projected._validate_source = functional._validate_source
    projected.populations = dict(
        zip(functional.populations, projected_populations, strict=True)
    )
    projected._stimulation_plan = copy.copy(functional._stimulation_plan)
    projected._stimulation_plan.validate_source = (
        functional._stimulation_plan.validate_source
    )
    projected._stimulation_plan.functional = projected
    projected._stimulation_plan.plans = {
        name: plan._poisson_plan
        for name, plan in projected.populations.items()
        if plan._poisson_plan is not None
    }
    # A staging generator is intentionally private even when its live source
    # was a default Torch stream. Retain that already-qualified ownership role
    # without installing the clone as ambient randomness.
    projected._stimulation_plan._staged_default_streams = {
        projected._stimulation_plan.local[name][entry.stream]: source.generator
        for name, plan in projected._stimulation_plan.plans.items()
        for entry in plan.entries
        for source in (plan._resolve_source(entry, graph.native.populations[name]),)
        if projected._stimulation_plan.local[name][entry.stream]
        in projected._stimulation_plan.default_streams
    }
    projected._stimulation_rng_plan = copy.copy(functional._stimulation_rng_plan)
    projected._stimulation_rng_plan.validate_source = (
        functional._stimulation_rng_plan.validate_source
    )
    projected._stimulation_rng_plan.functional = projected
    projected._stimulation_rng_plan.stimulation = projected._stimulation_plan
    return graph.native, projected


__all__ = []
