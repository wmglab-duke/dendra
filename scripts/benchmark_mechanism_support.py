"""Microbenchmark population-preserving mechanism support operations.

This covers both current assembly and shared field reads.  ``--channels`` is
also used as the number of field-reading mechanisms so the command-line
surface stays small.

Run outside pytest so timing never affects correctness CI, for example::

    python scripts/benchmark_mechanism_support.py --population 1 5 32 256
    python scripts/benchmark_mechanism_support.py --device cuda --compile
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.benchmark import Compare, Timer


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


def measurements(
    *,
    population: int,
    compartments: int,
    support: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
    compile_: bool,
):
    if support > compartments:
        raise ValueError("support must not exceed compartments")
    columns = torch.linspace(
        0, compartments - 1, support, device=device, dtype=torch.long
    )
    flat_indices = (
        torch.arange(population, device=device)[:, None] * compartments + columns
    ).reshape(-1)
    rowwise_indices = columns.expand(population, -1)
    voltage = torch.randn(population, compartments, device=device, dtype=dtype)
    weights = torch.linspace(0.25, 1.25, channels, device=device, dtype=dtype)

    def flat_separate():
        destination = torch.zeros_like(voltage).reshape(-1)
        source = voltage.reshape(-1)
        for weight in weights:
            local = source.index_select(0, flat_indices) * weight
            destination.scatter_add_(0, flat_indices, local)
        return destination

    def flat_grouped():
        destination = torch.zeros_like(voltage).reshape(-1)
        local_voltage = voltage.reshape(-1).index_select(0, flat_indices)
        local_total = torch.zeros_like(local_voltage)
        for weight in weights:
            local_total = local_total + local_voltage * weight
        destination.scatter_add_(0, flat_indices, local_total)
        return destination

    def axis_separate():
        destination = torch.zeros_like(voltage)
        for weight in weights:
            local = voltage.index_select(-1, columns) * weight
            destination.scatter_add_(-1, rowwise_indices, local)
        return destination

    def axis_grouped():
        destination = torch.zeros_like(voltage)
        local_voltage = voltage.index_select(-1, columns)
        local_total = torch.zeros_like(local_voltage)
        for weight in weights:
            local_total = local_total + local_voltage * weight
        destination.scatter_add_(-1, rowwise_indices, local_total)
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
        ).blocked_autorange(min_run_time=0.5)
        for name, fn in compiled.items()
    ]


def field_read_measurements(
    *,
    population: int,
    compartments: int,
    support: int,
    readers: int,
    device: torch.device,
    dtype: torch.dtype,
    compile_: bool,
):
    """Measure binding one field to mechanisms on one exact shared support.

    A pure reader can safely retain the gathered tensor, while a consumer that
    also writes the same field needs a private clone.  The latter mirrors the
    alias-isolation rule used by the material-read scheduler.
    """
    if support > compartments:
        raise ValueError("support must not exceed compartments")
    if readers < 1:
        raise ValueError("readers must be positive")

    columns = torch.linspace(
        0, compartments - 1, support, device=device, dtype=torch.long
    )
    field = torch.randn(population, compartments, device=device, dtype=dtype)

    def separate_reads():
        return tuple(field.index_select(-1, columns) for _ in range(readers))

    def shared_pure_reads():
        local = field.index_select(-1, columns)
        return (local,) * readers

    def shared_reads_with_writer():
        local = field.index_select(-1, columns)
        return (local,) * (readers - 1) + (local.clone(),)

    eager = {
        "separate/readers": separate_reads,
        "shared/pure-readers": shared_pure_reads,
        "shared/one-read-write": shared_reads_with_writer,
    }

    reference = separate_reads()
    shared = shared_pure_reads()
    if not all(value is shared[0] for value in shared):
        raise AssertionError("pure readers must share the gathered tensor")
    read_write = shared_reads_with_writer()
    if readers > 1 and read_write[-1] is read_write[0]:
        raise AssertionError("the read/write consumer must receive a clone")
    if readers > 1:
        preserved_reader = read_write[0].clone()
        read_write[-1].add_(1)
        torch.testing.assert_close(read_write[0], preserved_reader)

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
        ).blocked_autorange(min_run_time=0.5)
        for name, fn in compiled.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", nargs="+", type=int, default=[1, 5, 32, 256])
    parser.add_argument("--compartments", type=int, default=1101)
    parser.add_argument("--support", type=int, default=101)
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = getattr(torch, args.dtype)

    results = []
    for population in args.population:
        results.extend(
            measurements(
                population=population,
                compartments=args.compartments,
                support=args.support,
                channels=args.channels,
                device=device,
                dtype=dtype,
                compile_=args.compile,
            )
        )
        results.extend(
            field_read_measurements(
                population=population,
                compartments=args.compartments,
                support=args.support,
                readers=args.channels,
                device=device,
                dtype=dtype,
                compile_=args.compile,
            )
        )
        flat_bytes = (
            population
            * args.support
            * torch.tensor([], dtype=torch.long).element_size()
        )
        column_bytes = args.support * torch.tensor([], dtype=torch.long).element_size()
        print(
            f"index bytes P={population}: flat={flat_bytes:,}, "
            f"target-compact={column_bytes:,}, "
            f"ratio={flat_bytes / column_bytes:.1f}x"
        )

    Compare(results).print()


if __name__ == "__main__":
    main()
