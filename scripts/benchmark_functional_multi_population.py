"""Probe setup and execution scaling for functional ``MultiPopulation``.

This benchmark is deliberately non-gating: elapsed time is printed for human
comparison, while deterministic structural counters expose where setup work is
performed.  Each component is an independent three-node Unmyelinated/HH cable;
``concat_models`` packs all of them into one scalar DHS solve.

Examples
--------
Run the standard setup-scaling matrix::

    python scripts/benchmark_functional_multi_population.py

Use shorter samples while iterating on lowering::

    python scripts/benchmark_functional_multi_population.py \
        --components 2 8 --lowering-repeats 1 --min-run-time 0.05

The ``solver calls`` column is a structural observation from one eager
functional rollout.  It must equal ``--steps`` regardless of component count:
components are packed into one solve rather than advanced independently.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import gc
import statistics
import time
from dataclasses import dataclass
from unittest import mock

import torch
from torch.utils.benchmark import Timer

import dendra as dn
import dendra.func._population as population_lowering
from dendra.models.core import Population
from dendra.models.mod import hh
from dendra.models.multi import MultiPopulation


@dataclass(frozen=True)
class ProbeResult:
    components: int
    nodes: int
    lowering_seconds: float
    prepare_seconds: float
    step_seconds: float
    rollout_seconds: float
    solver_calls: int
    clone_counts: dict[str, int]
    initialize_counts: dict[str, int]
    plan_counts: dict[str, int]
    lowering_prepare_counts: dict[str, int]
    explicit_prepare_counts: dict[str, int]
    adapter_parameter_slots: int
    adapter_buffer_slots: int
    adapter_tensor_elements: int
    adapter_runtime_slots: tuple[str, ...]
    transition_component_modules: int
    preparation_component_modules: int
    transition_component_mapping_slots: int
    preparation_component_mapping_slots: int


def _counter_kind(population) -> str:
    return "packed" if isinstance(population, MultiPopulation) else "component"


def _model(*, components: int, size: int, dtype: torch.dtype):
    if components < 1:
        raise ValueError("components must be positive")
    if size < 2:
        raise ValueError("size must be at least two")

    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        populations = {}
        for index in range(components):
            component = dn.Unmyelinated(
                [1.0 + 0.01 * index],
                L=float(size - 1),
                dx=1.0,
                dtype=dtype,
                integrator=dn.bwd_euler_ub(method="thomas", imem=False),
            )
            component.insert(hh)
            populations[f"c{index:03d}"] = component
        model = dn.concat_models(populations, threads=2, write_back=False)
        model.initialize()
        model.train()
    return model


@contextlib.contextmanager
def _instrument_lowering():
    """Count deterministic setup operations without changing their results."""
    counts = {
        "clone": collections.Counter(),
        "initialize": collections.Counter(),
        "plan": collections.Counter(),
        "prepare": collections.Counter(),
    }
    original_clone = population_lowering._clone_execution_population
    original_initialize = Population.initialize
    original_plan_init = population_lowering.FunctionalPopulation.__init__
    original_prepare = population_lowering.FunctionalPopulation._prepare_values

    def counted_clone(population):
        counts["clone"][_counter_kind(population)] += 1
        return original_clone(population)

    def counted_initialize(population, *args, **kwargs):
        counts["initialize"][_counter_kind(population)] += 1
        return original_initialize(population, *args, **kwargs)

    def counted_plan_init(plan, population, *args, **kwargs):
        counts["plan"][_counter_kind(population)] += 1
        return original_plan_init(plan, population, *args, **kwargs)

    def counted_prepare(plan, *args, **kwargs):
        counts["prepare"][plan._topology_kind] += 1
        return original_prepare(plan, *args, **kwargs)

    with (
        mock.patch.object(
            population_lowering,
            "_clone_execution_population",
            counted_clone,
        ),
        mock.patch.object(Population, "initialize", counted_initialize),
        mock.patch.object(
            population_lowering.FunctionalPopulation,
            "__init__",
            counted_plan_init,
        ),
        mock.patch.object(
            population_lowering.FunctionalPopulation,
            "_prepare_values",
            counted_prepare,
        ),
    ):
        yield counts


def _timed_lowering(model, *, dt: float, repeats: int):
    samples = []
    last = None
    for _index in range(repeats):
        gc.collect()
        start = time.perf_counter()
        last = dn.func.make_functional(model, dt=dt)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), last


def _blocked_time(fn, *, min_run_time: float) -> float:
    return (
        Timer(stmt="fn()", globals={"fn": fn})
        .blocked_autorange(min_run_time=min_run_time)
        .median
    )


def _adapter_inventory(functional):
    adapters = getattr(
        functional._preparation,
        "component_physical_preparations",
        (),
    )
    parameter_slots = 0
    buffer_slots = 0
    tensor_elements = 0
    runtime_slots = set()
    for adapter in adapters:
        parameters = tuple(adapter.named_parameters(remove_duplicate=False))
        buffers = tuple(adapter.named_buffers(remove_duplicate=False))
        parameter_slots += len(parameters)
        buffer_slots += len(buffers)
        tensor_elements += sum(value.numel() for _name, value in parameters)
        tensor_elements += sum(value.numel() for _name, value in buffers)
        for name, _value in (*parameters, *buffers):
            if name in {"population.v", "population.t"} or name.startswith(
                ("population.integrator.", "population.mech.")
            ):
                runtime_slots.add(name)
    return (
        parameter_slots,
        buffer_slots,
        tensor_elements,
        tuple(sorted(runtime_slots)),
    )


def _retained_parent_inventory(functional):
    transition_population = functional._transition.population
    preparation_population = functional._preparation.population
    return (
        len(transition_population.populations),
        len(preparation_population.populations),
        sum(
            name.startswith("population.populations.")
            for name in functional._base_mapping
        ),
        sum(
            name.startswith("population.populations.")
            for name in functional._preparation_base_mapping
        ),
    )


def _probe_case(
    *,
    components: int,
    size: int,
    steps: int,
    dt: float,
    dtype: torch.dtype,
    lowering_repeats: int,
    min_run_time: float,
) -> ProbeResult:
    model = _model(components=components, size=size, dtype=dtype)

    with _instrument_lowering() as lowering_counts:
        instrumented_start = time.perf_counter()
        functional, tensors = dn.func.make_functional(model, dt=dt)
        instrumented_lowering_seconds = time.perf_counter() - instrumented_start

    # Use uninstrumented repetitions for the reported wall-clock baseline. A
    # single-repeat smoke run reuses the instrumented sample to avoid doubling
    # setup time while developing the lowering itself.
    if lowering_repeats == 1:
        lowering_seconds = instrumented_lowering_seconds
    else:
        lowering_seconds, (functional, tensors) = _timed_lowering(
            model,
            dt=dt,
            repeats=lowering_repeats,
        )

    (
        adapter_parameter_slots,
        adapter_buffer_slots,
        adapter_tensor_elements,
        adapter_runtime_slots,
    ) = _adapter_inventory(functional)
    (
        transition_component_modules,
        preparation_component_modules,
        transition_component_mapping_slots,
        preparation_component_mapping_slots,
    ) = _retained_parent_inventory(functional)

    explicit_prepare_counts = collections.Counter()
    original_prepare = population_lowering.FunctionalPopulation._prepare_values

    def counted_explicit_prepare(plan, *args, **kwargs):
        key = "packed" if plan is functional else "component"
        explicit_prepare_counts[key] += 1
        return original_prepare(plan, *args, **kwargs)

    with mock.patch.object(
        population_lowering.FunctionalPopulation,
        "_prepare_values",
        counted_explicit_prepare,
    ):
        prepared = functional.prepare(tensors.parameters, tensors.constants)

    def prepare_once():
        functional.prepare(tensors.parameters, tensors.constants)

    prepare_seconds = _blocked_time(prepare_once, min_run_time=min_run_time)

    ve = torch.linspace(
        -1.0,
        1.0,
        steps * model.v.numel(),
        dtype=model.dtype(),
        device=model.device(),
    ).reshape(steps, *model.shape)
    intra = torch.linspace(
        -1.0e-9,
        1.0e-9,
        steps * model.v.numel(),
        dtype=model.dtype(),
        device=model.device(),
    ).reshape(steps, *model.shape)
    step_input = dn.func.StepInput(ve=ve[0], intra=intra[0])
    rollout_input = dn.func.RolloutInput(ve=ve, intra=intra)

    def step_once():
        with torch.no_grad():
            functional.step(
                tensors.parameters,
                prepared,
                tensors.state,
                step_input,
            )

    def rollout_once():
        with torch.no_grad():
            functional.rollout(
                tensors.parameters,
                prepared,
                tensors.state,
                rollout_input,
            )

    step_seconds = _blocked_time(step_once, min_run_time=min_run_time)
    rollout_seconds = _blocked_time(rollout_once, min_run_time=min_run_time)
    solver_calls = 0
    original_solver = functional._transition.solver

    def counted_solver(*args, **kwargs):
        nonlocal solver_calls
        solver_calls += 1
        return original_solver(*args, **kwargs)

    functional._transition.solver = counted_solver
    try:
        rollout_once()
    finally:
        functional._transition.solver = original_solver
    if solver_calls != steps:
        raise RuntimeError(
            f"expected one packed solver call per step, got {solver_calls} "
            f"calls for {steps} steps"
        )

    return ProbeResult(
        components=components,
        nodes=model.v.numel(),
        lowering_seconds=lowering_seconds,
        prepare_seconds=prepare_seconds,
        step_seconds=step_seconds,
        rollout_seconds=rollout_seconds,
        solver_calls=solver_calls,
        clone_counts=dict(lowering_counts["clone"]),
        initialize_counts=dict(lowering_counts["initialize"]),
        plan_counts=dict(lowering_counts["plan"]),
        lowering_prepare_counts=dict(lowering_counts["prepare"]),
        explicit_prepare_counts=dict(explicit_prepare_counts),
        adapter_parameter_slots=adapter_parameter_slots,
        adapter_buffer_slots=adapter_buffer_slots,
        adapter_tensor_elements=adapter_tensor_elements,
        adapter_runtime_slots=adapter_runtime_slots,
        transition_component_modules=transition_component_modules,
        preparation_component_modules=preparation_component_modules,
        transition_component_mapping_slots=transition_component_mapping_slots,
        preparation_component_mapping_slots=preparation_component_mapping_slots,
    )


def _format_counts(counts: dict[str, int]) -> str:
    return (
        ", ".join(f"{name}={value}" for name, value in sorted(counts.items())) or "none"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--components", type=int, nargs="+", default=(2, 8, 24))
    parser.add_argument("--size", type=int, default=3)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--lowering-repeats", type=int, default=3)
    parser.add_argument("--min-run-time", type=float, default=0.2)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="PyTorch CPU threads used by the probe",
    )
    args = parser.parse_args()
    if any(value < 1 for value in args.components):
        parser.error("--components values must be positive")
    if args.size < 2:
        parser.error("--size must be at least two")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.lowering_repeats < 1:
        parser.error("--lowering-repeats must be positive")
    if args.min_run_time <= 0.0:
        parser.error("--min-run-time must be positive")
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")

    torch.set_num_threads(args.torch_threads)
    dtype = getattr(torch, args.dtype)
    results = [
        _probe_case(
            components=components,
            size=args.size,
            steps=args.steps,
            dt=args.dt,
            dtype=dtype,
            lowering_repeats=args.lowering_repeats,
            min_run_time=args.min_run_time,
        )
        for components in args.components
    ]

    print(
        f"torch={torch.__version__}, dtype={args.dtype}, size={args.size}, "
        f"steps={args.steps}, dt={args.dt}, torch_threads={args.torch_threads}, "
        f"lowering_repeats={args.lowering_repeats}"
    )
    print(
        "components  nodes  lowering (ms)  prepare (ms)  step (ms)  "
        "rollout (ms)  solver calls"
    )
    for result in results:
        print(
            f"{result.components:10d}  {result.nodes:5d}  "
            f"{result.lowering_seconds * 1e3:13.3f}  "
            f"{result.prepare_seconds * 1e3:12.3f}  "
            f"{result.step_seconds * 1e3:9.3f}  "
            f"{result.rollout_seconds * 1e3:12.3f}  "
            f"{result.solver_calls:12d}"
        )
        print(f"  lowering clones:       {_format_counts(result.clone_counts)}")
        print(f"  lowering initializes:  {_format_counts(result.initialize_counts)}")
        print(f"  lowering plans:        {_format_counts(result.plan_counts)}")
        print(
            f"  lowering preparations: {_format_counts(result.lowering_prepare_counts)}"
        )
        print(
            f"  one public prepare:    {_format_counts(result.explicit_prepare_counts)}"
        )
        runtime_slots = ", ".join(result.adapter_runtime_slots) or "none"
        print(
            "  adapter inventory:     "
            f"parameters={result.adapter_parameter_slots}, "
            f"buffers={result.adapter_buffer_slots}, "
            f"elements={result.adapter_tensor_elements}, "
            f"runtime slots={runtime_slots}"
        )
        print(
            "  retained parents:      "
            f"transition components/mappings="
            f"{result.transition_component_modules}/"
            f"{result.transition_component_mapping_slots}, "
            f"preparation components/mappings="
            f"{result.preparation_component_modules}/"
            f"{result.preparation_component_mapping_slots}"
        )


if __name__ == "__main__":
    main()
