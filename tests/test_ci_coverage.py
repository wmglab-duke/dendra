"""Sharded CI must measure the same branches and tests as one complete run."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import coverage
import pytest

from scripts.combine_ci_coverage import validate_shards

SCRIPT = Path(__file__).parents[1] / "scripts" / "combine_ci_coverage.py"
TARGET = "dendra/branch_target.py"
SOURCE = """def classify(value):
    if value < 0:
        return "negative"
    if value == 0:
        return "zero"
    return "positive"
"""
CONFIG = "[run]\nbranch = True\nrelative_files = True\nsource = dendra\n"


def _write_checkout(path):
    (path / "dendra").mkdir(parents=True)
    (path / "dendra" / "__init__.py").write_text("", encoding="utf-8")
    (path / TARGET).write_text(SOURCE, encoding="utf-8")
    (path / ".coveragerc.cpu").write_text(CONFIG, encoding="utf-8")
    (path / "driver.py").write_text(
        'import runpy, sys\nclassify = runpy.run_path("dendra/branch_target.py")["classify"]\n'
        "for argument in sys.argv[1:]:\n    classify(int(argument))\n",
        encoding="utf-8",
    )


def _measure(checkout, destination, values):
    env = {key: value for key, value in os.environ.items() if not key.startswith("COV")}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=.coveragerc.cpu",
            f"--data-file={destination}",
            "driver.py",
            *map(str, values),
        ],
        cwd=checkout,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def shard_artifacts(tmp_path):
    aggregation = tmp_path / "aggregate-checkout"
    _write_checkout(aggregation)
    (aggregation / "coverage-floors.toml").write_text(
        f'[coverage-floors]\n"{TARGET}" = 100\n', encoding="utf-8"
    )
    artifacts = aggregation / "ci-results"
    selected = [f"tests/test_branch.py::test_case[{index}]" for index in range(1, 5)]
    for index, value in enumerate((-1, 0, 1, -1), 1):
        directory = artifacts / str(index)
        directory.mkdir(parents=True)
        worker = tmp_path / f"separate-worker-checkout-{index}"
        _write_checkout(worker)
        _measure(worker, directory / ".coverage", [value])
        manifest = {
            "schema_version": 1,
            "index": index,
            "total": 4,
            "selected_nodeids": selected,
            "assigned_nodeids": [selected[index - 1]],
            "executed_nodeids": [selected[index - 1]],
            "exitstatus": 0,
            "collect_only": False,
        }
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return aggregation, artifacts


def _aggregate(checkout, *options):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--shards=4", *options],
        cwd=checkout,
        check=False,
        capture_output=True,
        text=True,
    )


def _alter_manifest(artifacts, shard_index, **changes):
    path = artifacts / str(shard_index) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(changes)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _data(path):
    data = coverage.CoverageData(basename=str(path))
    data.read()
    return data


def test_combined_relocated_shards_equal_monolithic_branch_coverage(shard_artifacts):
    checkout, artifacts = shard_artifacts
    monolithic_path = checkout / "monolithic.coverage"
    _measure(checkout, monolithic_path, [-1, 0, 1, -1])
    # A stale aggregate must not affect this run, and auxiliary artifacts are safe.
    (checkout / ".coverage").write_text("stale", encoding="utf-8")
    (artifacts / "1" / "junit.xml").write_text("<testsuite/>", encoding="utf-8")

    result = _aggregate(checkout, "--fail-under=100")

    assert result.returncode == 0, result.stdout + result.stderr
    combined = _data(checkout / ".coverage")
    monolithic = _data(monolithic_path)
    assert combined.has_arcs() and monolithic.has_arcs()
    combined_executed = {
        filename for filename in combined.measured_files() if combined.lines(filename)
    }
    monolithic_executed = {
        filename
        for filename in monolithic.measured_files()
        if monolithic.lines(filename)
    }
    assert combined_executed == monolithic_executed == {TARGET}
    assert set(combined.lines(TARGET)) == set(monolithic.lines(TARGET))
    assert set(combined.arcs(TARGET)) == set(monolithic.arcs(TARGET))
    for report in ("coverage.json", "coverage.xml", "htmlcov/index.html"):
        assert (checkout / report).is_file()
    report = json.loads((checkout / "coverage.json").read_text(encoding="utf-8"))
    assert report["totals"]["percent_covered"] == 100
    assert report["totals"]["num_branches"] > 0
    assert all(
        (artifacts / str(index) / ".coverage").is_file() for index in range(1, 5)
    )


@pytest.mark.parametrize("missing", ["manifest.json", ".coverage", "directory"])
def test_missing_shard_artifact_fails_closed(shard_artifacts, missing):
    checkout, artifacts = shard_artifacts
    directory = artifacts / "4"
    if missing == "directory":
        for path in directory.iterdir():
            path.unlink()
        directory.rmdir()
    else:
        (directory / missing).unlink()
    result = _aggregate(checkout)
    assert result.returncode == 2
    assert "aggregation error" in result.stderr
    assert not (checkout / "coverage.json").exists()


@pytest.mark.parametrize("contents", [b"", b"not a coverage database"])
def test_empty_or_unreadable_coverage_fails_closed(shard_artifacts, contents):
    checkout, artifacts = shard_artifacts
    (artifacts / "4" / ".coverage").write_bytes(contents)
    result = _aggregate(checkout)
    assert result.returncode == 2
    assert "aggregation error" in result.stderr


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({"selected_nodeids": ["different::test"]}, "collection disagrees"),
        ({"assigned_nodeids": ["unknown::test"]}, "unselected tests"),
        (
            {"assigned_nodeids": ["tests/test_branch.py::test_case[1]"]},
            "multiple shards",
        ),
        ({"executed_nodeids": ["not-executed::test"]}, "executed tests differ"),
        ({"assigned_nodeids": []}, "unique, nonempty"),
        ({"index": 1}, "index must equal 4"),
        ({"total": 3}, "total must equal 4"),
        ({"schema_version": 2}, "schema_version must equal 1"),
        ({"exitstatus": 1}, "did not finish successfully"),
        ({"exitstatus": None}, "did not finish successfully"),
        ({"collect_only": True}, "collect-only"),
    ],
)
def test_conflicting_or_incomplete_manifests_fail_closed(
    shard_artifacts, changes, expected
):
    _, artifacts = shard_artifacts
    _alter_manifest(artifacts, 4, **changes)
    with pytest.raises(ValueError, match=expected):
        validate_shards(artifacts, 4)


def test_agreed_collection_cannot_omit_an_unassigned_test(shard_artifacts):
    _, artifacts = shard_artifacts
    for index in range(1, 5):
        path = artifacts / str(index) / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["selected_nodeids"].append("tests/test_missing.py::test_omitted")
        path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="do not cover the full collection"):
        validate_shards(artifacts, 4)


def test_duplicate_manifest_test_id_fails_closed(shard_artifacts):
    _, artifacts = shard_artifacts
    duplicate = "tests/test_branch.py::test_case[4]"
    _alter_manifest(artifacts, 4, assigned_nodeids=[duplicate, duplicate])
    with pytest.raises(ValueError, match="unique, nonempty"):
        validate_shards(artifacts, 4)


def test_unexpected_shard_directory_fails_closed(shard_artifacts):
    _, artifacts = shard_artifacts
    (artifacts / "5").mkdir()
    with pytest.raises(ValueError, match="expected exactly shard directories"):
        validate_shards(artifacts, 4)


def test_line_only_data_cannot_reduce_branch_coverage(shard_artifacts):
    _, artifacts = shard_artifacts
    path = artifacts / "4" / ".coverage"
    path.unlink()
    data = coverage.CoverageData(basename=str(path))
    data.add_lines({TARGET: [1, 2, 3]})
    data.write()
    with pytest.raises(ValueError, match="branch coverage is required"):
        validate_shards(artifacts, 4)


@pytest.mark.parametrize("absolute", [False, True])
def test_empty_or_unrelocatable_branch_data_fails_closed(shard_artifacts, absolute):
    _, artifacts = shard_artifacts
    path = artifacts / "4" / ".coverage"
    path.unlink()
    data = coverage.CoverageData(basename=str(path))
    filename = str((artifacts / TARGET).resolve()) if absolute else TARGET
    data.add_arcs({filename: [(-1, 1), (1, -1)] if absolute else []})
    data.write()
    expected = "must be relative" if absolute else "no executed source lines"
    with pytest.raises(ValueError, match=expected):
        validate_shards(artifacts, 4)


@pytest.mark.parametrize("gate", ["overall", "module"])
def test_failed_gate_still_writes_all_combined_reports(shard_artifacts, gate):
    checkout, artifacts = shard_artifacts
    # All tests ran, but deliberately measure only one branch in every shard.
    for index in range(1, 5):
        path = artifacts / str(index) / ".coverage"
        path.unlink()
        _measure(checkout, path, [-1])
    options = [] if gate == "overall" else ["--fail-under=0"]
    if gate == "overall":
        (checkout / "coverage-floors.toml").write_text(
            f'[coverage-floors]\n"{TARGET}" = 0\n', encoding="utf-8"
        )

    result = _aggregate(checkout, *options)

    assert result.returncode == 1, result.stdout + result.stderr
    assert (
        "below" in result.stderr
        if gate == "overall"
        else "floor(s) failed" in result.stderr
    )
    for report in ("coverage.json", "coverage.xml", "htmlcov/index.html"):
        assert (checkout / report).is_file()


def _fingerprint(record):
    import hashlib

    return {
        **record,
        "sha256": hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


@pytest.fixture
def partition_artifacts(tmp_path):
    import hashlib
    from copy import deepcopy

    root = tmp_path / "partition-checkout"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "ci_shard.py").write_text("# selected shard plugin\n")
    (root / "dendra").mkdir()
    (root / "dendra" / "__init__.py").write_text("VALUE = 1\n")
    (root / "tests").mkdir()
    modules = [f"tests/test_branch_{index}.py" for index in range(1, 5)]
    for module in modules:
        (root / module).write_text("def test_one(): pass\ndef test_two(): pass\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*.py")
    }
    runtime = _fingerprint(
        {
            "files": {"dendra/__init__.py": files["dendra/__init__.py"]},
            "environment": {"packages": [["torch", "2.15.0+cpu"]]},
        }
    )
    config_files = {name: value for name, value in files.items() if name not in modules}
    config_files["@ci_shard_plugin"] = files["scripts/ci_shard.py"]
    config = _fingerprint({"files": config_files})
    selected = [
        f"{module}::test_{name}" for module in modules for name in ("one", "two")
    ]
    artifacts = root / "ci-results"
    for index, module in enumerate(modules, 1):
        assigned = [node for node in selected if node.startswith(module + "::")]
        initial = {
            "modules": {module: files[module]},
            "runtime": runtime,
            "config": config,
        }
        snapshot = {**initial, "final": deepcopy(initial), "unchanged": True}
        manifest = {
            "schema_version": 1,
            "index": index,
            "total": 4,
            "selected_nodeids": selected,
            "assigned_nodeids": assigned,
            "executed_nodeids": assigned,
            "collect_only": False,
            "exitstatus": 0,
            "module_assignments": {
                name: position for position, name in enumerate(modules, 1)
            },
            "source_snapshot": snapshot,
        }
        directory = artifacts / str(index)
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps(manifest))
    return root, artifacts


def _partition_change(artifacts, index, change):
    from copy import deepcopy

    path = artifacts / str(index) / "manifest.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    snapshot = manifest["source_snapshot"]
    for domain in ("runtime", "config"):
        record = snapshot[domain]
        snapshot[domain] = _fingerprint(
            {k: v for k, v in record.items() if k != "sha256"}
        )
    snapshot["final"] = deepcopy(
        {name: snapshot[name] for name in ("modules", "runtime", "config")}
    )
    path.write_text(json.dumps(manifest))


def test_partition_only_validates_sources_without_weakening_default_coverage(
    partition_artifacts,
):
    root, artifacts = partition_artifacts
    assert validate_shards(artifacts, 4, require_coverage=False, source_root=root) == []
    with pytest.raises(ValueError, match="raw coverage data is missing"):
        validate_shards(artifacts, 4)
    result = _aggregate(root, "--partition-only", "--torch-spec=torch==2.15.0")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Verified 8 tests across 4 shards with Torch 2.15.0+cpu" in result.stdout
    assert not (root / ".coverage").exists()
    assert not (root / "coverage.json").exists()


@pytest.mark.parametrize("spec", ["torch==2.14.0", "torch>=2.16,<3"])
def test_partition_only_rejects_wrong_actual_torch_version(partition_artifacts, spec):
    root, _artifacts = partition_artifacts
    result = _aggregate(root, "--partition-only", f"--torch-spec={spec}")
    assert result.returncode == 2
    assert "does not satisfy" in result.stderr


@pytest.mark.parametrize("domain", ["runtime", "config"])
def test_partition_only_rejects_mixed_fingerprints(partition_artifacts, domain):
    root, artifacts = partition_artifacts

    def change(manifest):
        record = manifest["source_snapshot"][domain]
        if domain == "runtime":
            record["environment"]["packages"] = [["torch", "2.16.0"]]
        else:
            record["extra_configuration"] = "another worker configuration"

    _partition_change(artifacts, 4, change)
    with pytest.raises(
        ValueError, match="runtime, configuration or module assignments disagree"
    ):
        validate_shards(artifacts, 4, require_coverage=False, source_root=root)


@pytest.mark.parametrize(
    "filename", ["tests/test_branch_4.py", "scripts/ci_shard.py", "dendra/__init__.py"]
)
def test_partition_only_rejects_stale_current_source(partition_artifacts, filename):
    root, artifacts = partition_artifacts
    with (root / filename).open("a") as stream:
        stream.write("# changed after worker execution\n")
    with pytest.raises(ValueError, match="current .*source differs"):
        validate_shards(artifacts, 4, require_coverage=False, source_root=root)


@pytest.mark.parametrize("change", ["missing", "changed", "digest"])
def test_partition_only_rejects_missing_or_invalid_attestation(
    partition_artifacts, change
):
    root, artifacts = partition_artifacts
    path = artifacts / "4" / "manifest.json"
    manifest = json.loads(path.read_text())
    if change == "missing":
        manifest.pop("source_snapshot")
    elif change == "changed":
        manifest["source_snapshot"]["unchanged"] = False
    else:
        manifest["source_snapshot"]["runtime"]["sha256"] = "invalid"
        manifest["source_snapshot"]["final"]["runtime"]["sha256"] = "invalid"
    path.write_text(json.dumps(manifest))
    with pytest.raises(
        ValueError, match="attestation|sources changed|source fingerprint"
    ):
        validate_shards(artifacts, 4, require_coverage=False, source_root=root)


def test_partition_only_rejects_split_module_assignments(partition_artifacts):
    root, artifacts = partition_artifacts

    def change(manifest):
        manifest["module_assignments"]["tests/test_branch_4.py"] = 3

    _partition_change(artifacts, 4, change)
    with pytest.raises(ValueError, match="whole-module assignment disagrees"):
        validate_shards(artifacts, 4, require_coverage=False, source_root=root)
