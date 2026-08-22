"""Microbenchmark population-preserving mechanism support operations.

This covers current assembly, shared field reads, and support-plan construction.
``--channels`` is also used as the number of field-reading mechanisms so the
command-line surface stays small.  Channels are assigned to contiguous authored
clusters on ``--distinct-supports`` exact ordered supports.  This makes each
support one adjacent current-reduction run, matching Dendra's authored-order
scheduler.

The accounting report deliberately separates selector storage from execution
grouping.  Dendra still serializes one flattened compatibility key per
mechanism, even when the runtime support registry interns the corresponding
support plan.  ``flattened-interned`` and ``compact-interned`` therefore show
the two later storage targets; they are not claims about the current checkpoint
format.  Reported storage is raw ``torch.long`` tensor payload only; it excludes
Python metadata and checkpoint-container overhead.  ``--compile`` measures
warmed compiled execution and deliberately excludes first-call compilation
latency.

Run outside pytest so timing never affects correctness CI, for example::

    python scripts/benchmark_mechanism_support.py --population 1 5 32 256
    python scripts/benchmark_mechanism_support.py --distinct-supports 2
    python scripts/benchmark_mechanism_support.py --device cuda --compile
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch
from torch.utils.benchmark import Compare, Timer

import dendra as dn  # noqa: F401 - configure TorchInductor before importing Torch
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.mechanisms._support import SupportMap, SupportSpec
from dendra.models.mechanisms._support_registry import SupportRegistry


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _maybe_compile(fn, enabled: bool, device: torch.device):
    if not enabled:
        return fn
    compiled = torch.compile(fn, fullgraph=True)
    compiled()
    compiled()
    _synchronize(device)
    return compiled


def _support_columns(
    *,
    compartments: int,
    support: int,
    distinct_supports: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Build deterministic, exact ordered supports for the benchmark."""

    base = torch.linspace(0, compartments - 1, support, device=device, dtype=torch.long)
    return tuple(
        (base + offset).remainder(compartments) for offset in range(distinct_supports)
    )


def _support_assignments(item_count: int, distinct_supports: int) -> tuple[int, ...]:
    """Assign contiguous authored clusters to non-empty support groups."""

    if item_count < 1:
        raise ValueError("channels/readers must be positive")
    if distinct_supports < 1:
        raise ValueError("distinct-supports must be positive")
    if distinct_supports > item_count:
        raise ValueError("distinct-supports must not exceed channels/readers")
    group_size, extra = divmod(item_count, distinct_supports)
    return tuple(
        support_index
        for support_index in range(distinct_supports)
        for _ in range(group_size + (support_index < extra))
    )


def _group_members(
    assignments: Sequence[int], distinct_supports: int
) -> tuple[tuple[int, ...], ...]:
    """Return mechanism indices in authored order for each support."""

    return tuple(
        tuple(
            item_index
            for item_index, support_index in enumerate(assignments)
            if support_index == group_index
        )
        for group_index in range(distinct_supports)
    )


def _integer_metric(registry, name: str) -> int | None:
    """Read an optional integer registry diagnostic without fixing its API."""

    if registry is None or not hasattr(registry, name):
        return None
    value = getattr(registry, name)
    value = value() if callable(value) else value
    return int(value)


class _SyntheticMechanismSupport:
    """Minimal mechanism-shaped object used for live registry accounting."""

    base_ndim = 2
    is_composable = False
    _support_key_values_valid = True

    def __init__(self, *, name, key, support_spec, support_map):
        self.name = name
        self.key = key
        self.support_spec = support_spec
        self.support_map = support_map
        self.shape_f = support_spec.runtime_local_shape


def _build_synthetic_mechanisms(
    *,
    population: int,
    compartments: int,
    support: int,
    mechanisms: int,
    distinct_supports: int,
) -> tuple[_SyntheticMechanismSupport, ...]:
    """Build one authored mechanism sequence for accounting and planning."""

    columns = _support_columns(
        compartments=compartments,
        support=support,
        distinct_supports=distinct_supports,
        device=torch.device("cpu"),
    )
    assignments = _support_assignments(mechanisms, distinct_supports)
    rows = torch.arange(population, dtype=torch.long)[:, None] * compartments
    synthetic_mechanisms = []
    for mechanism_index, support_index in enumerate(assignments):
        key = (rows + columns[support_index]).reshape(-1)
        spec = SupportSpec.from_compiled(
            core_shape=(population, compartments),
            key=key,
            is_composable=False,
            local_shape=(key.numel(),),
        )
        if population > 1:
            spec = spec.with_population_axis()
        synthetic_mechanisms.append(
            _SyntheticMechanismSupport(
                name=f"mechanism_{mechanism_index}",
                key=key.clone(),
                support_spec=spec,
                support_map=SupportMap(spec),
            )
        )

    return tuple(synthetic_mechanisms)


def _planning_phases(mechanisms: Sequence[_SyntheticMechanismSupport]):
    """Return six representative handler phase sequences.

    The mix mirrors the repeated all/pre-current/current-reader/current-source
    partitions constructed by ``MechanismHandler.make_maps()`` without
    pretending that the synthetic mechanisms declare real ion dependencies.
    """

    mechanisms = tuple(mechanisms)
    reader_count = max(1, len(mechanisms) // 3)
    pre_current = mechanisms[:-reader_count]
    current_readers = mechanisms[-reader_count:]
    ion_sources = mechanisms[: max(1, len(mechanisms) - reader_count)]
    return (
        mechanisms,
        pre_current,
        current_readers,
        mechanisms,
        ion_sources,
        mechanisms,
    )


def planning_measurements(
    *,
    mechanisms: Sequence[_SyntheticMechanismSupport],
    population: int,
    compartments: int,
    support: int,
    distinct_supports: int,
    min_run_time: float,
):
    """Compare repeated legacy partitioning with the handler-wide registry."""

    mechanisms = tuple(mechanisms)
    phases = _planning_phases(mechanisms)
    registry = SupportRegistry()

    def legacy_repartition_each_phase():
        return tuple(
            MechanismHandler._partition_current_supports(phase) for phase in phases
        )

    def registry_rebuild_then_partition():
        registry.rebuild(mechanisms)
        return tuple(registry.partition(phase) for phase in phases)

    legacy_result = legacy_repartition_each_phase()
    registry_result = registry_rebuild_then_partition()
    for phase, legacy_partition, registry_partition in zip(
        phases, legacy_result, registry_result, strict=True
    ):
        legacy_by_mechanism = legacy_partition[2]
        registry_by_mechanism = registry_partition[2]
        if len(legacy_partition[0]) != len(registry_partition[0]):
            raise AssertionError("Legacy and registry planning disagree on groups.")
        for left in phase:
            for right in phase:
                legacy_same = (
                    legacy_by_mechanism[id(left)] == legacy_by_mechanism[id(right)]
                )
                registry_same = (
                    registry_by_mechanism[id(left)] == registry_by_mechanism[id(right)]
                )
                if legacy_same != registry_same:
                    raise AssertionError(
                        "Legacy and registry planning disagree on support identity."
                    )

    description = (
        f"P={population}, C={compartments}, K={support}, "
        f"M={len(mechanisms)}, S={distinct_supports}, "
        f"Q={len(phases)}, cpu, Python"
    )
    functions = {
        "legacy/repartition-each-phase": legacy_repartition_each_phase,
        "registry/rebuild-once+partition": registry_rebuild_then_partition,
    }
    return [
        Timer(
            stmt="fn()",
            globals={"fn": fn},
            label="mechanism support plan construction",
            sub_label=description,
            description=name,
        ).blocked_autorange(min_run_time=min_run_time)
        for name, fn in functions.items()
    ]


def _print_support_accounting(
    *,
    population: int,
    support: int,
    mechanisms: int,
    distinct_supports: int,
    registry=None,
) -> None:
    """Report scheduling counts and current/future selector storage.

    ``registry`` is optional so this remains a standalone tensor benchmark.
    When a live handler or :class:`SupportRegistry` is supplied, its registry
    diagnostics take precedence over the scenario estimates.
    """

    if registry is not None and hasattr(registry, "support_registry"):
        registry = registry.support_registry

    index_bytes = torch.tensor([], dtype=torch.long).element_size()
    serialized_indices = mechanisms * population * support
    runtime_indices = serialized_indices
    flattened_interned_indices = distinct_supports * population * support
    compact_interned_indices = distinct_supports * support

    if registry is not None:
        accounting = getattr(registry, "accounting", None)
        distinct_supports = len(registry)
        if accounting is not None:
            mechanisms = int(accounting.registered_mechanisms)
            serialized_indices = int(accounting.compatibility_indices)
            runtime_indices = serialized_indices
            flattened_interned_indices = int(accounting.unique_legacy_indices)
            compact_interned_indices = int(accounting.compact_indices)
        registry_legacy_indices = _integer_metric(registry, "legacy_index_count")
        if registry_legacy_indices is not None:
            flattened_interned_indices = registry_legacy_indices
        registry_compact_indices = _integer_metric(registry, "compact_index_count")
        if registry_compact_indices is not None:
            compact_interned_indices = registry_compact_indices
        registry_runtime_indices = _integer_metric(registry, "runtime_index_count")
        if registry_runtime_indices is None:
            registry_runtime_indices = _integer_metric(registry, "interned_index_count")
    else:
        registry_runtime_indices = None

    print(
        f"support plan P={population}: mechanisms={mechanisms}, "
        f"distinct-supports={distinct_supports}"
    )
    print(
        "  current assembly: "
        f"gathers separate/grouped={mechanisms}/{distinct_supports}, "
        f"scatters separate/grouped={mechanisms}/{distinct_supports}"
    )
    print(
        "  shared field reads: "
        f"gathers separate/grouped={mechanisms}/{distinct_supports}"
    )
    print(
        "  raw torch.long selector payload bytes: "
        f"serialized-per-mechanism={serialized_indices * index_bytes:,}, "
        f"runtime-per-mechanism={runtime_indices * index_bytes:,}, "
        f"flattened-interned={flattened_interned_indices * index_bytes:,}, "
        f"compact-interned={compact_interned_indices * index_bytes:,}"
    )
    if registry_runtime_indices is not None:
        print(
            "  live registry selector bytes: "
            f"{registry_runtime_indices * index_bytes:,}"
        )
    elif registry is not None:
        print(
            "  live registry: owns no selector tensors; current runtime keys "
            "remain per mechanism"
        )


def measurements(
    *,
    population: int,
    compartments: int,
    support: int,
    channels: int,
    distinct_supports: int,
    device: torch.device,
    dtype: torch.dtype,
    compile_: bool,
    min_run_time: float,
):
    if support > compartments:
        raise ValueError("support must not exceed compartments")
    assignments = _support_assignments(channels, distinct_supports)
    groups = _group_members(assignments, distinct_supports)
    columns = _support_columns(
        compartments=compartments,
        support=support,
        distinct_supports=distinct_supports,
        device=device,
    )
    flat_indices = tuple(
        (
            torch.arange(population, device=device)[:, None] * compartments
            + support_columns
        ).reshape(-1)
        for support_columns in columns
    )
    rowwise_indices = tuple(
        support_columns.expand(population, -1) for support_columns in columns
    )
    voltage = torch.randn(population, compartments, device=device, dtype=dtype)
    weights = torch.linspace(0.25, 1.25, channels, device=device, dtype=dtype)

    def flat_separate():
        destination = torch.zeros_like(voltage).reshape(-1)
        source = voltage.reshape(-1)
        for mechanism_index, weight in enumerate(weights):
            indices = flat_indices[assignments[mechanism_index]]
            local = source.index_select(0, indices) * weight
            destination.scatter_add_(0, indices, local)
        return destination

    def flat_grouped():
        destination = torch.zeros_like(voltage).reshape(-1)
        source = voltage.reshape(-1)
        for support_index, members in enumerate(groups):
            indices = flat_indices[support_index]
            local_voltage = source.index_select(0, indices)
            local_total = torch.zeros_like(local_voltage)
            for mechanism_index in members:
                local_total = local_total + local_voltage * weights[mechanism_index]
            destination.scatter_add_(0, indices, local_total)
        return destination

    def axis_separate():
        destination = torch.zeros_like(voltage)
        for mechanism_index, weight in enumerate(weights):
            support_index = assignments[mechanism_index]
            local = voltage.index_select(-1, columns[support_index]) * weight
            destination.scatter_add_(-1, rowwise_indices[support_index], local)
        return destination

    def axis_grouped():
        destination = torch.zeros_like(voltage)
        for support_index, members in enumerate(groups):
            local_voltage = voltage.index_select(-1, columns[support_index])
            local_total = torch.zeros_like(local_voltage)
            for mechanism_index in members:
                local_total = local_total + local_voltage * weights[mechanism_index]
            destination.scatter_add_(-1, rowwise_indices[support_index], local_total)
        return destination

    eager = {
        "flat/separate": flat_separate,
        "flat/grouped": flat_grouped,
        "axis/separate": axis_separate,
        "axis/grouped": axis_grouped,
    }
    compiled = {
        name: _maybe_compile(fn, compile_, device) for name, fn in eager.items()
    }

    reference = flat_separate().reshape_as(voltage)
    for name, fn in compiled.items():
        torch.testing.assert_close(fn().reshape_as(voltage), reference)
        _synchronize(device)

    description = (
        f"P={population}, C={compartments}, K={support}, M={channels}, "
        f"S={distinct_supports}, "
        f"{device.type}, {str(dtype).removeprefix('torch.')}, "
        f"{'compiled' if compile_ else 'eager'}"
    )
    return [
        Timer(
            stmt="fn()",
            globals={"fn": fn},
            label="mechanism support current assembly",
            sub_label=description,
            description=name,
        ).blocked_autorange(min_run_time=min_run_time)
        for name, fn in compiled.items()
    ]


def field_read_measurements(
    *,
    population: int,
    compartments: int,
    support: int,
    readers: int,
    distinct_supports: int,
    device: torch.device,
    dtype: torch.dtype,
    compile_: bool,
    min_run_time: float,
):
    """Measure binding one field to mechanisms on one exact shared support.

    A pure reader can safely retain the gathered tensor, while a consumer that
    also writes the same field needs a private clone.  The latter mirrors the
    alias-isolation rule used by the material-read scheduler.
    """
    if support > compartments:
        raise ValueError("support must not exceed compartments")
    assignments = _support_assignments(readers, distinct_supports)

    columns = _support_columns(
        compartments=compartments,
        support=support,
        distinct_supports=distinct_supports,
        device=device,
    )
    field = torch.randn(population, compartments, device=device, dtype=dtype)

    def separate_reads():
        return tuple(
            field.index_select(-1, columns[support_index])
            for support_index in assignments
        )

    def shared_pure_reads():
        locals_ = tuple(
            field.index_select(-1, support_columns) for support_columns in columns
        )
        return tuple(locals_[support_index] for support_index in assignments)

    def shared_reads_with_writer():
        bindings = list(shared_pure_reads())
        bindings[-1] = bindings[-1].clone()
        return tuple(bindings)

    eager = {
        "separate/readers": separate_reads,
        "shared/pure-readers": shared_pure_reads,
        "shared/one-read-write": shared_reads_with_writer,
    }

    reference = separate_reads()
    shared = shared_pure_reads()
    for support_index in range(distinct_supports):
        members = tuple(
            index
            for index, assigned_support in enumerate(assignments)
            if assigned_support == support_index
        )
        if not all(shared[index] is shared[members[0]] for index in members):
            raise AssertionError("pure readers must share each gathered tensor")
    read_write = shared_reads_with_writer()
    same_support_readers = tuple(
        index
        for index, support_index in enumerate(assignments[:-1])
        if support_index == assignments[-1]
    )
    if same_support_readers and read_write[-1] is read_write[same_support_readers[0]]:
        raise AssertionError("the read/write consumer must receive a clone")
    if same_support_readers:
        reader_index = same_support_readers[0]
        preserved_reader = read_write[reader_index].clone()
        read_write[-1].add_(1)
        torch.testing.assert_close(read_write[reader_index], preserved_reader)

    compiled = {
        name: _maybe_compile(fn, compile_, device) for name, fn in eager.items()
    }
    for name, fn in compiled.items():
        actual = fn()
        if len(actual) != len(reference):
            raise AssertionError(f"{name} returned the wrong number of bindings")
        for actual_value, expected_value in zip(actual, reference, strict=True):
            torch.testing.assert_close(actual_value, expected_value)
        _synchronize(device)

    description = (
        f"P={population}, C={compartments}, K={support}, R={readers}, "
        f"S={distinct_supports}, "
        f"{device.type}, {str(dtype).removeprefix('torch.')}, "
        f"{'compiled' if compile_ else 'eager'}"
    )
    return [
        Timer(
            stmt="fn()",
            globals={"fn": fn},
            label="shared mechanism field reads",
            sub_label=description,
            description=name,
        ).blocked_autorange(min_run_time=min_run_time)
        for name, fn in compiled.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", nargs="+", type=int, default=[1, 5, 32, 256])
    parser.add_argument("--compartments", type=int, default=1101)
    parser.add_argument("--support", type=int, default=101)
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--distinct-supports", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="benchmark warmed torch.compile execution (compile time excluded)",
    )
    parser.add_argument("--min-run-time", type=float, default=0.5)
    args = parser.parse_args()

    if args.min_run_time <= 0:
        parser.error("--min-run-time must be positive")
    if any(population < 1 for population in args.population):
        parser.error("--population values must be positive")
    if args.compartments < 1:
        parser.error("--compartments must be positive")
    if not 1 <= args.support <= args.compartments:
        parser.error("--support must be between 1 and --compartments")
    if args.channels < 1:
        parser.error("--channels must be positive")
    if not 1 <= args.distinct_supports <= args.channels:
        parser.error("--distinct-supports must be between 1 and --channels")
    if args.distinct_supports > args.compartments:
        parser.error(
            "--distinct-supports must not exceed --compartments in this "
            "synthetic scenario"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = getattr(torch, args.dtype)

    results = []
    for population in args.population:
        synthetic_mechanisms = _build_synthetic_mechanisms(
            population=population,
            compartments=args.compartments,
            support=args.support,
            mechanisms=args.channels,
            distinct_supports=args.distinct_supports,
        )
        results.extend(
            measurements(
                population=population,
                compartments=args.compartments,
                support=args.support,
                channels=args.channels,
                distinct_supports=args.distinct_supports,
                device=device,
                dtype=dtype,
                compile_=args.compile,
                min_run_time=args.min_run_time,
            )
        )
        results.extend(
            field_read_measurements(
                population=population,
                compartments=args.compartments,
                support=args.support,
                readers=args.channels,
                distinct_supports=args.distinct_supports,
                device=device,
                dtype=dtype,
                compile_=args.compile,
                min_run_time=args.min_run_time,
            )
        )
        results.extend(
            planning_measurements(
                mechanisms=synthetic_mechanisms,
                population=population,
                compartments=args.compartments,
                support=args.support,
                distinct_supports=args.distinct_supports,
                min_run_time=args.min_run_time,
            )
        )
        registry = SupportRegistry(synthetic_mechanisms)
        if len(registry) != args.distinct_supports:
            raise AssertionError(
                "The live SupportRegistry did not preserve the requested "
                "number of exact synthetic supports."
            )
        _print_support_accounting(
            population=population,
            support=args.support,
            mechanisms=args.channels,
            distinct_supports=args.distinct_supports,
            registry=registry,
        )

    Compare(results).print()


if __name__ == "__main__":
    main()
