"""Measure functional training across independent kernel/checkpoint sizes.

Each configuration/repeat runs serially in a fresh process with its own whole
TMPDIR and Inductor cache. Timed samples include differentiable preparation,
forward/loss, and backward; input cloning is reported separately. Every sample
is checked against an eager host recurrence outside the timed regions.

Examples::

    python scripts/benchmark_functional_training.py --output /tmp/training.json
    python scripts/benchmark_functional_training.py --cases hh_axon \
        --steps 32 128 --chunk-steps 1 4 8 --checkpoint-steps 16 32 \
        --repeats 3 --samples 5 --output /tmp/hh-training.json

The default is a CPU float64 matrix using Inductor. Memory diagnostics report
tracked tensor storage (including nested checkpoint inputs), not allocator
peak activation memory. Worker peak RSS includes compilation and excludes
compiler child processes. No timing threshold belongs in pytest.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import random
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

MODES = ("eager", "eager_checkpointed", "compiled", "compiled_checkpointed")
CASE_NAMES = ("hh_axon", "hh_tree", "hh_extcell", "sweeney1987")


@dataclass(frozen=True)
class Configuration:
    case: str = "hh_axon"
    steps: int = 32
    chunk_steps: int = 1
    checkpoint_steps: int = 16
    mode: str = "eager"
    dtype: str = "float64"
    batch: int = 2
    size: int = 17
    dt: float = 0.01
    backend: str = "inductor"
    threads: int = 1
    seed: int = 0
    samples: int = 5
    warmup: int = 1
    repeat: int = 0

    def validate(self):
        for name in (
            "steps",
            "chunk_steps",
            "checkpoint_steps",
            "batch",
            "size",
            "threads",
            "samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("warmup", "repeat", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be positive and finite")
        if self.case not in CASE_NAMES or self.mode not in MODES:
            raise ValueError("unknown model case or execution mode")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if self.backend not in ("inductor", "aot_eager", "eager"):
            raise ValueError("unknown compilation backend")


@dataclass
class Problem:
    case: Any
    functional: Any
    tensors: Any
    inputs: Any
    config: Configuration
    setup_seconds: float
    lowering_seconds: float


@dataclass
class Execution:
    config: Configuration
    kernel: Any = None
    setup_seconds: float = 0.0


@dataclass
class SampleResult:
    state: Any
    loss: Any
    gradients: dict[str, Any]
    timings: dict[str, float]
    memory: dict[str, Any] | None = None


def build_problem(config: Configuration) -> Problem:
    import torch
    from _functional_training_cases import build_case

    import dendra as dn

    config.validate()
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    start = time.perf_counter()
    case = build_case(
        config.case,
        dtype=getattr(torch, config.dtype),
        batch=config.batch,
        size=config.size,
    )
    setup = time.perf_counter() - start
    start = time.perf_counter()
    functional, tensors = dn.func.make_functional(case.model, dt=config.dt)
    lowering = time.perf_counter() - start
    dtype = getattr(torch, config.dtype)
    shape = (config.steps, *case.model.shape)
    spatial = torch.linspace(0, 1, case.model.shape[-1], dtype=dtype)
    temporal = torch.linspace(-1, 1, config.steps, dtype=dtype).reshape(
        config.steps, *([1] * len(case.model.shape))
    )
    inputs = dn.func.RolloutInput(
        ve=(temporal * torch.sin(torch.pi * spatial)).expand(shape).clone(),
        intra=(1e-9 * temporal * torch.exp(-((spatial - 0.4) * 5).square()))
        .expand(shape)
        .clone(),
    )
    return Problem(case, functional, tensors, inputs, config, setup, lowering)


def build_execution(problem: Problem, config: Configuration) -> Execution:
    config.validate()
    start = time.perf_counter()
    kernel = None
    if config.mode.startswith("compiled"):
        kernel = problem.functional.compile_rollout_chunk(
            config.chunk_steps, backend=config.backend, fullgraph=True, dynamic=False
        )
    return Execution(config, kernel, time.perf_counter() - start)


def _clone_tree(tree):
    import torch

    return torch.utils._pytree.tree_map(lambda tensor: tensor.detach().clone(), tree)


def run_sample(
    problem: Problem, execution: Execution, *, collect_memory=False
) -> SampleResult:
    from functools import partial

    import torch
    from _functional_training_memory import SavedTensorStats

    import dendra as dn

    config = execution.config
    start = time.perf_counter()
    parameters = _clone_tree(problem.tensors.parameters)
    constants = _clone_tree(problem.tensors.constants)
    state = _clone_tree(problem.tensors.state)
    parameter = parameters[problem.case.parameter_name].requires_grad_()
    geometry = constants[problem.case.geometry_name].requires_grad_()
    voltage = state["integrator"]["v"].requires_grad_()
    if "vc" in state["integrator"]:
        # Block state is canonical: preserve v == vi - vext[0] while exposing
        # the same initial membrane-voltage derivative as scalar models.
        vc = state["integrator"]["vc"]
        state["integrator"]["vc"] = torch.cat(
            ((voltage + vc[..., 1]).unsqueeze(-1), vc[..., 1:]), dim=-1
        )
    inputs = dn.func.RolloutInput(
        ve=problem.inputs.ve.detach().clone().requires_grad_(),
        intra=problem.inputs.intra.detach().clone().requires_grad_(),
    )
    targets = {
        "parameter": parameter,
        "geometry": geometry,
        "initial_voltage": voltage,
        "ve": inputs.ve,
        "intra": inputs.intra,
    }
    clone_seconds = time.perf_counter() - start
    stats = SavedTensorStats() if collect_memory else None
    hooks = (
        torch.autograd.graph.saved_tensors_hooks(stats.pack, stats.unpack)
        if stats
        else contextlib.nullcontext()
    )
    boundaries = stats.observe_checkpoints() if stats else contextlib.nullcontext()
    start = time.perf_counter()
    with hooks, boundaries:
        prepared = problem.functional.prepare(parameters, constants)
        preparation_seconds = time.perf_counter() - start
        if stats:
            prepared_values = problem.functional._validate_prepared(
                prepared, parameters
            )
            stats.retain_tree(
                "explicit_inputs",
                (parameters, constants, state, inputs, prepared_values),
            )
        bound = partial(
            execution.kernel
            if execution.kernel is not None
            else problem.functional.step,
            parameters,
            prepared,
        )
        runner = (
            dn.func.longrun_checkpointed
            if config.mode.endswith("checkpointed")
            else dn.func.longrun
        )
        final, auxiliary = runner(
            problem.functional,
            bound,
            state,
            config.steps * config.dt,
            config.checkpoint_steps,
            inputs,
        )
        voltage = final["integrator"]["v"]
        weights = torch.linspace(0.5, 1.5, voltage.shape[-1], dtype=voltage.dtype)
        loss = (voltage.square() * weights).mean() + 0.01 * auxiliary["v"].sin().mean()
    forward_seconds = time.perf_counter() - start
    memory = stats.summary() if stats else None
    start = time.perf_counter()
    gradients = torch.autograd.grad(loss, tuple(targets.values()), allow_unused=False)
    backward_seconds = time.perf_counter() - start
    return SampleResult(
        _clone_tree(final),
        loss.detach().clone(),
        {
            name: value.detach().clone()
            for name, value in zip(targets, gradients, strict=True)
        },
        {
            "clone_seconds": clone_seconds,
            "prepare_seconds": preparation_seconds,
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "forward_backward_seconds": forward_seconds + backward_seconds,
        },
        memory,
    )


def assert_parity(actual: SampleResult, expected: SampleResult) -> dict:
    import torch
    from torch.utils._pytree import tree_flatten_with_path

    rtol, atol = (3e-5, 3e-6) if expected.loss.dtype == torch.float32 else (3e-8, 3e-10)
    metrics = {}
    for group in ("state", "loss", "gradients"):
        a_leaves, a_spec = tree_flatten_with_path(getattr(actual, group))
        e_leaves, e_spec = tree_flatten_with_path(getattr(expected, group))
        if a_spec != e_spec:
            raise AssertionError(f"{group} tree schema differs from reference")
        errors = {}
        for (path, a), (_, e) in zip(a_leaves, e_leaves, strict=True):
            label = "/".join(str(entry) for entry in path) or group
            if not torch.isfinite(a).all() or not torch.isfinite(e).all():
                raise AssertionError(f"nonfinite {group} {label}")
            torch.testing.assert_close(
                a, e, rtol=rtol, atol=atol, msg=lambda msg: f"{group} {label}: {msg}"
            )
            if a.dtype == torch.bool:
                error = float(torch.count_nonzero(a != e))
                scale = float(torch.count_nonzero(e))
            else:
                error = float((a - e).abs().max()) if a.numel() else 0.0
                scale = float(e.abs().max()) if e.numel() else 0.0
            if group == "gradients":
                if scale == 0:
                    raise AssertionError(
                        f"reference gradient {label} is identically zero"
                    )
                # Gradients carry different physical units. Bound each
                # target's relative infinity-norm error so absolute tolerances
                # cannot hide a lost small derivative. A componentwise relative
                # comparison is ill-conditioned at cancellation-level entries.
                if error / scale > rtol:
                    raise AssertionError(
                        f"normalized gradient {label}: relative infinity-norm "
                        f"error {error / scale:.6g} exceeds {rtol:.6g}"
                    )
            errors[label] = {"max_abs_error": error, "reference_max_abs": scale}
            if group == "gradients":
                errors[label]["relative_linf_error"] = error / scale
        metrics[group] = errors
    return metrics


def _peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _git_metadata(root):
    def command(*args):
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else None

    diff = command("diff", "HEAD")
    return {
        "root": str(root),
        "revision": command("rev-parse", "HEAD"),
        "status": command("status", "--short"),
        "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest()
        if diff is not None
        else None,
    }


def _metadata(problem):
    import torch
    from torch._inductor import config as compiler_config

    import dendra as dn

    modules = {}
    for name in ("dendra", "dendra_solvers", "dendra_models"):
        try:
            module = __import__(name)
            modules[name] = {
                "version": getattr(module, "__version__", None),
                "path": module.__file__,
                "source": _git_metadata(Path(module.__file__).resolve().parent.parent),
            }
        except ImportError:
            modules[name] = None
    cpu = (
        subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        if sys.platform == "darwin"
        else platform.processor()
    )
    files = [
        Path(__file__),
        Path(__file__).with_name("_functional_training_cases.py"),
        Path(__file__).with_name("_functional_training_memory.py"),
    ]
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu": cpu,
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "compile_threads": compiler_config.compile_threads,
        "modules": modules,
        "source": _git_metadata(Path(dn.__file__).resolve().parent.parent),
        "benchmark_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "TMPDIR",
                "TORCHINDUCTOR_CACHE_DIR",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE",
                "TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE",
                "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE",
                "TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE",
            )
        },
        "model_shape": list(problem.case.model.shape),
        "solver": problem.case.solver,
        "parameter_name": problem.case.parameter_name,
        "geometry_name": problem.case.geometry_name,
        "cache_policy": "fresh process, whole TMPDIR and Inductor cache per configuration/repeat; remote caches disabled",
        "memory_policy": "tracked tensor-storage proxy incl explicit inputs and nested checkpoint args; RSS is compile-inclusive worker high-water, excludes compiler children",
        "timing_policy": "candidate first call before eager oracle; forward includes preparation and runner validation; input cloning/setup, parity checks and result detachment excluded; serial workers",
        "parity_policy": "pointwise state/loss/gradient closeness plus per-target gradient relative infinity-norm error <= 3e-8 (float64) or 3e-5 (float32); finite nonzero reference gradients required",
    }


def _compiler_counters():
    from torch._dynamo.utils import counters

    return {
        group: {
            str(key): value
            for key, value in entries.items()
            if isinstance(value, (int, float))
        }
        for group, entries in counters.items()
        if entries
    }


def execute_worker(config):
    import torch
    from torch._dynamo.utils import counters

    from dendra._bootstrap import torch_compiler_warning_context

    problem = build_problem(config)
    execution = build_execution(problem, config)
    counters.clear()
    with torch.enable_grad(), torch_compiler_warning_context():
        cold = run_sample(problem, execution)
        cold_peak = _peak_rss_bytes()
        cold_counters = _compiler_counters()
        reference = run_sample(
            problem, build_execution(problem, replace(config, mode="eager"))
        )
        parity = assert_parity(cold, reference)
        for _ in range(config.warmup):
            assert_parity(run_sample(problem, execution), reference)
        before_samples_counters = _compiler_counters()
        warmed = []
        for _ in range(config.samples):
            sample = run_sample(problem, execution)
            assert_parity(sample, reference)
            warmed.append(sample.timings)
        warm_counters = _compiler_counters()
        prediagnostic_peak = _peak_rss_bytes()
        diagnostic = run_sample(problem, execution, collect_memory=True)
        assert_parity(diagnostic, reference)
        diagnostic_counters = _compiler_counters()
    graphs_added = warm_counters.get("stats", {}).get(
        "unique_graphs", 0
    ) - before_samples_counters.get("stats", {}).get("unique_graphs", 0)
    return {
        "status": "ok",
        "config": asdict(config),
        "metadata": _metadata(problem),
        "setup": {
            "model_seconds": problem.setup_seconds,
            "lowering_seconds": problem.lowering_seconds,
            "kernel_seconds": execution.setup_seconds,
        },
        "cold": cold.timings,
        "warm_samples": warmed,
        "warm_median": {
            key: statistics.median(sample[key] for sample in warmed)
            for key in warmed[0]
        },
        "memory": diagnostic.memory,
        "cold_worker_peak_rss_bytes": cold_peak,
        "worker_prediagnostic_peak_rss_bytes": prediagnostic_peak,
        "worker_lifetime_peak_rss_bytes": _peak_rss_bytes(),
        "parity": parity,
        "compiler_counters_after_cold": cold_counters,
        "compiler_counters_before_samples": before_samples_counters,
        "compiler_counters_after_warm": warm_counters,
        "compiler_counters_after_diagnostic": diagnostic_counters,
        "compiler_graphs_added_during_samples": graphs_added,
        "warm_samples_are_compiled": config.mode.startswith("compiled")
        and graphs_added == 0,
    }


def matrix_configurations(args):
    configs = []
    for repeat in range(args.repeats):
        block = []
        for case in args.cases:
            for steps in args.steps:
                for span in args.checkpoint_steps:
                    for mode in args.modes:
                        widths = (
                            args.chunk_steps if mode.startswith("compiled") else [1]
                        )
                        for width in widths:
                            config = Configuration(
                                case=case,
                                steps=steps,
                                chunk_steps=width,
                                checkpoint_steps=span,
                                mode=mode,
                                dtype=args.dtype,
                                batch=args.batch,
                                size=args.size,
                                dt=args.dt,
                                backend=args.backend,
                                threads=args.threads,
                                seed=args.seed,
                                samples=args.samples,
                                warmup=args.warmup,
                                repeat=repeat,
                            )
                            config.validate()
                            block.append(config)
        random.Random(args.seed + repeat).shuffle(block)
        configs.extend(block)
    return configs


def aggregate_records(records, expected_repeats=None):
    groups = {}
    for record in records:
        key = json.dumps(
            {k: v for k, v in record["config"].items() if k != "repeat"}, sort_keys=True
        )
        groups.setdefault(key, []).append(record)
    result = []
    for key, observed in groups.items():
        values = [value for value in observed if value["status"] == "ok"]
        metrics = {}
        sources = {
            "cold_forward_seconds": lambda r: r["cold"]["forward_seconds"],
            "cold_backward_seconds": lambda r: r["cold"]["backward_seconds"],
            "cold_forward_backward_seconds": lambda r: r["cold"][
                "forward_backward_seconds"
            ],
            **{
                f"warm_{name}": lambda r, name=name: r["warm_median"][name]
                for name in (values[0]["warm_median"] if values else {})
            },
            "worker_prediagnostic_peak_rss_bytes": lambda r: r[
                "worker_prediagnostic_peak_rss_bytes"
            ],
            "worker_lifetime_peak_rss_bytes": lambda r: r[
                "worker_lifetime_peak_rss_bytes"
            ],
        }
        for name, get in sources.items():
            if not values:
                continue
            numbers = [get(value) for value in values]
            metrics[name] = {
                "median": statistics.median(numbers),
                "min": min(numbers),
                "max": max(numbers),
            }
        result.append(
            {
                "config": json.loads(key),
                "worker_repeats": len(values),
                "observed_repeats": len(observed),
                "expected_repeats": expected_repeats
                if expected_repeats is not None
                else len(observed),
                "failed_repeats": len(observed) - len(values),
                "recompiling_repeats": sum(
                    value["compiler_graphs_added_during_samples"] > 0
                    for value in values
                ),
                "metrics": metrics,
                "memory": values[0]["memory"] if values else None,
                "memory_representative_repeat": values[0]["config"]["repeat"]
                if values
                else None,
                "memory_identical_across_repeats": all(
                    value["memory"] == values[0]["memory"] for value in values
                ),
            }
        )
    return result


def _write_report(path, records, args):
    document = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": sys.argv,
        "configuration_count": len(records),
        "success_count": sum(record["status"] == "ok" for record in records),
        "failure_count": sum(record["status"] != "ok" for record in records),
        "records": records,
        "summary": aggregate_records(records, args.repeats),
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cases", nargs="+", choices=CASE_NAMES, default=list(CASE_NAMES)
    )
    parser.add_argument("--steps", nargs="+", type=int, default=[32])
    parser.add_argument("--chunk-steps", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--checkpoint-steps", nargs="+", type=int, default=[16])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument(
        "--backend", choices=["inductor", "aot_eager", "eager"], default="inductor"
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--size", type=int, default=17)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-config", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_config:
        config = Configuration(**json.loads(args.worker_config))
        try:
            result = execute_worker(config)
        except Exception as exc:
            result = {
                "status": "error",
                "config": asdict(config),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        args.worker_result.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        return 0 if result["status"] == "ok" else 1
    if args.output is None:
        parser.error("--output is required")
    if args.repeats < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--repeats and --timeout must be positive")
    try:
        configs = matrix_configurations(args)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logs = args.output.with_suffix("") / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    records = []
    for index, config in enumerate(configs):
        print(
            f"[{index + 1}/{len(configs)}] {config.case} {config.mode} T={config.steps} C={config.chunk_steps} H={config.checkpoint_steps} repeat={config.repeat + 1}",
            flush=True,
        )
        scratch = Path(
            tempfile.mkdtemp(
                prefix="dendra-training-",
                dir="/private/tmp" if sys.platform == "darwin" else None,
            )
        )
        result_path = scratch / "result.json"
        env = os.environ.copy()
        env.update(
            {
                "TMPDIR": str(scratch),
                "TMP": str(scratch),
                "TEMP": str(scratch),
                "TORCHINDUCTOR_CACHE_DIR": str(scratch / "inductor"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE": "0",
                "TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE": "0",
                "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE": "0",
                "TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE": "0",
                "OMP_NUM_THREADS": str(config.threads),
                "MKL_NUM_THREADS": str(config.threads),
            }
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-config",
            json.dumps(asdict(config)),
            "--worker-result",
            str(result_path),
        ]
        log_path = logs / f"worker-{index:04d}.log"
        started = time.perf_counter()
        try:
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    # Inductor can own compiler grandchildren. Stop the entire
                    # group before deleting caches or timing another worker.
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise
            if result_path.exists():
                record = json.loads(result_path.read_text())
            else:
                record = {
                    "status": "error",
                    "config": asdict(config),
                    "error": f"worker exited {process.returncode} without a result",
                }
        except subprocess.TimeoutExpired:
            record = {
                "status": "error",
                "config": asdict(config),
                "error": f"worker timed out after {args.timeout} seconds",
            }
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        record["worker_wall_seconds"] = time.perf_counter() - started
        record["log"] = str(log_path)
        records.append(record)
        _write_report(args.output, records, args)
        if record["status"] == "ok":
            print(
                f"  passed; warm forward+backward {record['warm_median']['forward_backward_seconds'] * 1000:.3f} ms; first call {record['cold']['forward_backward_seconds']:.3f} s",
                flush=True,
            )
            if record["compiler_graphs_added_during_samples"]:
                print(
                    "  WARNING: timed samples compiled new graphs; these are not steady-state timings",
                    flush=True,
                )
        else:
            print(f"  FAILED: {record['error']}", flush=True)
    print(f"Saved {len(records)} worker records to {args.output}", flush=True)
    return int(any(record["status"] != "ok" for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
