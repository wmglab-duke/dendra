"""Guard benchmark correctness and accounting; never assert performance timings."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators.implicit import DENDRA_SOLVERS_AVAILABLE

SCRIPT = Path(__file__).parents[2] / "scripts" / "benchmark_functional_training.py"
if not SCRIPT.exists():
    # Also permit validating the complete benchmark before installing its files.
    SCRIPT = Path(__file__).with_name("benchmark_functional_training.py")

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


@pytest.fixture(scope="module")
def benchmark():
    spec = importlib.util.spec_from_file_location(
        "dendra_functional_training_benchmark", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(SCRIPT.parent))


@pytest.fixture(scope="module")
def saved_stats(benchmark):
    from _functional_training_memory import SavedTensorStats

    return SavedTensorStats


@pytest.fixture(scope="module")
def tiny_config(benchmark):
    return benchmark.Configuration(
        case="hh_axon",
        steps=5,
        chunk_steps=2,
        checkpoint_steps=3,
        mode="eager",
        dtype="float64",
        batch=1,
        size=3,
        backend="aot_eager",
        threads=1,
        seed=17,
        samples=1,
        warmup=0,
    )


@pytest.fixture(scope="module")
def problem(benchmark, tiny_config):
    if not DENDRA_SOLVERS_AVAILABLE:
        pytest.skip("training benchmark models require native dendra-solvers")
    return benchmark.build_problem(tiny_config)


@pytest.fixture(scope="module")
def eager_result(benchmark, problem, tiny_config):
    execution = benchmark.build_execution(problem, tiny_config)
    return benchmark.run_sample(problem, execution)


def _clone_tree(tree):
    return torch.utils._pytree.tree_map(lambda value: value.detach().clone(), tree)


def _source_tensors(problem):
    return (
        problem.tensors.parameters,
        problem.tensors.constants,
        problem.tensors.state,
        problem.inputs.ve,
        problem.inputs.intra,
    )


def test_training_samples_start_fresh_and_leave_the_problem_unchanged(
    benchmark, problem, tiny_config
):
    source = _source_tensors(problem)
    before = _clone_tree(source)
    source_leaves = torch.utils._pytree.tree_leaves(source)
    requires_grad = [value.requires_grad for value in source_leaves]
    execution = benchmark.build_execution(problem, tiny_config)

    first = benchmark.run_sample(problem, execution)
    second = benchmark.run_sample(problem, execution)

    benchmark.assert_parity(second, first)
    torch.testing.assert_close(_source_tensors(problem), before, rtol=0.0, atol=0.0)
    assert [value.requires_grad for value in source_leaves] == requires_grad
    assert all(value.grad is None for value in source_leaves)
    for result in (first, second):
        for value in torch.utils._pytree.tree_leaves(
            (result.state, result.loss, result.gradients)
        ):
            assert value.grad_fn is None
            assert not value.requires_grad
        assert result.gradients
        assert set(result.gradients) == {
            "parameter",
            "geometry",
            "initial_voltage",
            "ve",
            "intra",
        }
        assert all(torch.isfinite(value).all() for value in result.gradients.values())
        assert all(value >= 0.0 for value in result.timings.values())


@pytest.mark.parametrize(
    "mode", ["eager_checkpointed", "compiled", "compiled_checkpointed"]
)
def test_benchmark_modes_match_eager_across_chunk_checkpoint_and_tail_boundaries(
    benchmark, problem, tiny_config, eager_result, mode
):
    config = replace(tiny_config, mode=mode)
    execution = benchmark.build_execution(problem, config)
    with torch_compiler_warning_context():
        actual = benchmark.run_sample(problem, execution, collect_memory=True)

    # Five steps, two-step kernels, and three-step checkpoint spans require
    # recurrent derivatives to survive every boundary, including one-step tails.
    benchmark.assert_parity(actual, eager_result)
    assert actual.memory is not None
    assert actual.memory["sources"]["explicit_inputs"]["unique_storage_bytes"] > 0
    if mode.endswith("checkpointed"):
        assert actual.memory["sources"]["checkpoint_inputs"]["count"] > 0
    assert actual.gradients.keys() == eager_result.gradients.keys()
    for name, expected in eager_result.gradients.items():
        assert torch.count_nonzero(expected) > 0, name


@pytest.mark.parametrize("changed", ["state", "loss", "gradient", "missing_gradient"])
def test_parity_gate_rejects_numerical_or_gradient_disagreement(
    benchmark, eager_result, changed
):
    candidate = replace(
        eager_result,
        state=_clone_tree(eager_result.state),
        loss=eager_result.loss.clone(),
        gradients=_clone_tree(eager_result.gradients),
    )
    if changed == "state":
        candidate.state["integrator"]["v"].add_(10.0)
    elif changed == "loss":
        candidate = replace(candidate, loss=candidate.loss + 100.0)
    elif changed == "gradient":
        name = next(iter(candidate.gradients))
        candidate.gradients[name].add_(100.0)
    else:
        candidate.gradients.pop(next(iter(candidate.gradients)))

    with pytest.raises((AssertionError, ValueError)):
        benchmark.assert_parity(candidate, eager_result)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_parity_gate_rejects_nonfinite_gradients_even_when_both_results_agree(
    benchmark, eager_result, nonfinite
):
    gradients = _clone_tree(eager_result.gradients)
    gradients[next(iter(gradients))].fill_(nonfinite)
    candidate = replace(eager_result, gradients=gradients)

    with pytest.raises((AssertionError, ValueError)):
        benchmark.assert_parity(candidate, candidate)


def _tiny_gradient_result(benchmark):
    return benchmark.SampleResult(
        state={"integrator": {"v": torch.tensor([-65.0], dtype=torch.float64)}},
        loss=torch.tensor(1.0, dtype=torch.float64),
        gradients={"geometry": torch.tensor([1.0e-12, 0.0], dtype=torch.float64)},
        timings={},
    )


def test_parity_gate_rejects_lost_small_gradient_despite_absolute_tolerance(
    benchmark,
):
    expected = _tiny_gradient_result(benchmark)
    actual = replace(
        expected, gradients={"geometry": torch.zeros(2, dtype=torch.float64)}
    )

    # The whole derivative fits below the ordinary raw absolute tolerance.
    # Its removal is nevertheless a complete loss of this target dependency.
    with pytest.raises((AssertionError, ValueError)):
        benchmark.assert_parity(actual, expected)


def test_parity_gate_accepts_cancellation_noise_relative_to_gradient_target(
    benchmark,
):
    expected = _tiny_gradient_result(benchmark)
    actual = replace(
        expected,
        gradients={"geometry": torch.tensor([1.0e-12, 1.0e-21], dtype=torch.float64)},
    )

    # The infinity-norm relative error is 1e-9, below the 3e-8 float64 bound.
    # A cancelling component must not impose a tighter elementwise tolerance.
    benchmark.assert_parity(actual, expected)


def test_retained_storage_counts_aliases_once_but_reports_every_saved_reference(
    saved_stats,
):
    stats = saved_stats()
    base = torch.arange(8, dtype=torch.float64, requires_grad=True)
    view = base[2:4]
    independent = view.detach().clone()

    for tensor in (base, base, view, independent, torch.empty(0)):
        packed = stats.pack(tensor)
        assert packed.grad_fn is None
        assert not packed.requires_grad
        torch.testing.assert_close(stats.unpack(packed), tensor)

    stats.retain_tree("explicit_inputs", {"view": base[1:2], "metadata": None})
    stats.retain_tree("prepared", torch.ones(3, dtype=base.dtype))
    summary = stats.summary()
    saved = summary["sources"]["autograd_saved"]
    element_bytes = base.element_size()
    assert saved["count"] == 5
    assert saved["logical_bytes"] == (8 + 8 + 2 + 2) * element_bytes
    assert saved["unique_storage_bytes"] == (8 + 2) * element_bytes
    assert summary["sources"]["explicit_inputs"]["logical_bytes"] == element_bytes
    assert summary["sources"]["explicit_inputs"]["unique_storage_bytes"] == (
        8 * element_bytes
    )
    # Explicit input aliases overlap with saved tensors; summing the groups
    # would incorrectly count the base allocation twice.
    assert summary["tracked_storage_bytes"] == (8 + 2 + 3) * element_bytes


def test_checkpoint_observer_tracks_nested_inputs_and_restores_runner_on_failure(
    saved_stats, monkeypatch
):
    from dendra.func import _runners

    def execute(function, *args, **kwargs):
        return function(*args)

    monkeypatch.setattr(_runners, "checkpoint", execute)
    stats = saved_stats()
    backing = torch.arange(10, dtype=torch.float64)
    state = {"voltage": backing[:2], "nested": (backing[4:6],)}

    with pytest.raises(RuntimeError, match="interrupt diagnostic"):
        with stats.observe_checkpoints():
            actual = _runners.checkpoint(lambda value: value, state)
            assert actual is state
            raise RuntimeError("interrupt diagnostic")

    assert _runners.checkpoint is execute
    checkpoints = stats.summary()["sources"]["checkpoint_inputs"]
    assert checkpoints["count"] == 2
    assert checkpoints["logical_bytes"] == 4 * backing.element_size()
    assert checkpoints["unique_storage_bytes"] == 10 * backing.element_size()


def test_matrix_repeats_preserve_identical_workloads_and_reproducible_order(
    benchmark, tiny_config
):
    args = SimpleNamespace(
        **{
            key: value
            for key, value in asdict(tiny_config).items()
            if key
            not in {
                "case",
                "steps",
                "chunk_steps",
                "checkpoint_steps",
                "mode",
                "repeat",
            }
        },
        cases=["hh_axon", "hh_tree"],
        steps=[5, 7],
        chunk_steps=[1, 2],
        checkpoint_steps=[3, 6],
        modes=["eager", "compiled_checkpointed"],
        repeats=2,
    )
    configs = benchmark.matrix_configurations(args)

    assert configs == benchmark.matrix_configurations(args)
    assert len(configs) == 2 * 2 * 2 * (1 + 2) * 2
    assert {config.seed for config in configs} == {tiny_config.seed}
    assert {config.chunk_steps for config in configs if config.mode == "eager"} == {1}
    assert {
        config.chunk_steps for config in configs if config.mode.startswith("compiled")
    } == {1, 2}
    repeats = Counter(
        json.dumps(
            {key: value for key, value in asdict(config).items() if key != "repeat"},
            sort_keys=True,
        )
        for config in configs
    )
    assert set(repeats.values()) == {2}
    assert Counter(config.repeat for config in configs) == {0: 24, 1: 24}


def test_aggregate_reports_failed_and_missing_repeats_without_biasing_timings(
    benchmark, tiny_config
):
    def passed(repeat, duration):
        return {
            "status": "ok",
            "config": asdict(replace(tiny_config, repeat=repeat)),
            "cold": {
                "forward_seconds": duration,
                "backward_seconds": duration * 2,
                "forward_backward_seconds": duration * 3,
            },
            "warm_median": {"forward_backward_seconds": duration},
            "worker_prediagnostic_peak_rss_bytes": 80 * repeat,
            "worker_lifetime_peak_rss_bytes": 100 * repeat,
            "compiler_graphs_added_during_samples": int(repeat == 2),
            "memory": {"tracked_storage_bytes": 10 * repeat},
        }

    records = [
        {
            "status": "error",
            "config": asdict(replace(tiny_config, repeat=0)),
            "error": "worker timed out",
        },
        passed(2, 4.0),
        passed(1, 2.0),
        *[
            {
                "status": "error",
                "config": asdict(replace(tiny_config, case="hh_tree", repeat=repeat)),
                "error": "worker failed",
            }
            for repeat in (0, 1)
        ],
    ]
    summaries = {
        summary["config"]["case"]: summary
        for summary in benchmark.aggregate_records(records, expected_repeats=4)
    }

    assert set(summaries) == {"hh_axon", "hh_tree"}
    mixed = summaries["hh_axon"]
    assert mixed["worker_repeats"] == 2
    assert mixed["observed_repeats"] == 3
    assert mixed["expected_repeats"] == 4
    assert mixed["failed_repeats"] == 1
    assert mixed["recompiling_repeats"] == 1
    assert mixed["metrics"]["warm_forward_backward_seconds"] == {
        "median": 3.0,
        "min": 2.0,
        "max": 4.0,
    }
    assert not mixed["memory_identical_across_repeats"]
    representative = next(
        record
        for record in records
        if record["status"] == "ok"
        and record["config"]["repeat"] == mixed["memory_representative_repeat"]
    )
    assert mixed["memory"] == representative["memory"]

    failed = summaries["hh_tree"]
    assert failed["worker_repeats"] == 0
    assert failed["observed_repeats"] == failed["failed_repeats"] == 2
    assert failed["expected_repeats"] == 4
    assert failed["recompiling_repeats"] == 0
    assert failed["metrics"] == {}
    assert failed["memory"] is None
    assert failed["memory_representative_repeat"] is None


@pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE,
    reason="successful native Tree benchmark requires dendra-solvers",
)
def test_matrix_cli_preserves_success_and_failure_records_in_one_report(
    benchmark, tmp_path
):
    output = tmp_path / "training.json"
    # Four compartments is valid for the tree, but invalid for a centered
    # Unmyelinated axon. A failed worker must not discard the passing result.
    command = [
        sys.executable,
        str(SCRIPT),
        "--cases",
        "hh_axon",
        "hh_tree",
        "--size",
        "4",
        "--batch",
        "1",
        "--modes",
        "eager",
        "--steps",
        "2",
        "--checkpoint-steps",
        "1",
        "--samples",
        "1",
        "--warmup",
        "0",
        "--repeats",
        "1",
        "--backend",
        "eager",
        "--dtype",
        "float64",
        "--dt",
        "0.02",
        "--seed",
        "23",
        "--threads",
        "1",
        "--timeout",
        "60",
        "--output",
        str(output),
    ]
    process = subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=120
    )

    assert process.returncode == 1, process.stdout + process.stderr
    report = json.loads(output.read_text())
    assert report["schema_version"] == 1
    assert report["configuration_count"] == 2
    assert report["success_count"] == report["failure_count"] == 1
    records = {record["config"]["case"]: record for record in report["records"]}
    failed = records["hh_axon"]
    assert failed["status"] == "error"
    assert "odd size" in failed["error"]
    passed = records["hh_tree"]
    assert passed["status"] == "ok"
    assert len(passed["warm_samples"]) == 1
    assert set(passed["parity"]) == {"state", "loss", "gradients"}
    assert len(passed["parity"]["gradients"]) == 5
    for record in records.values():
        assert record["config"]["seed"] == 23
        assert record["config"]["dt"] == 0.02
        assert record["config"]["repeat"] == 0
        assert record["config"]["backend"] == "eager"
        assert record["config"]["checkpoint_steps"] == 1
        assert record["config"]["samples"] == 1
        assert record["config"]["warmup"] == 0
        assert Path(record["log"]).is_file()
    assert len(report["summary"]) == 2
    summaries = {row["config"]["case"]: row for row in report["summary"]}
    assert summaries["hh_tree"]["worker_repeats"] == 1
    assert summaries["hh_tree"]["memory_identical_across_repeats"]
    assert summaries["hh_tree"]["memory_representative_repeat"] == 0
    assert summaries["hh_axon"]["worker_repeats"] == 0
    assert summaries["hh_axon"]["failed_repeats"] == 1
    assert summaries["hh_axon"]["metrics"] == {}
    assert summaries["hh_axon"]["memory"] is None
    for summary in summaries.values():
        assert summary["observed_repeats"] == summary["expected_repeats"] == 1
