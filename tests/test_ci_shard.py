"""Exercise the CI plugin against independent, lightweight pytest suites."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import ci_shard

_ROOT = Path(ci_shard.__file__).resolve().parents[1]


@pytest.fixture
def shard_suite(tmp_path: Path) -> Path:
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    cpu\n    neuron\n    cuda\n    slow\n",
        encoding="utf-8",
    )
    for name, count, markers in (
        ("cpu", 5, "[pytest.mark.cpu]"),
        ("neuron", 3, "[pytest.mark.neuron, pytest.mark.slow]"),
        ("cuda", 2, "[pytest.mark.neuron, pytest.mark.cuda]"),
        ("other", 1, "[pytest.mark.cpu]"),
    ):
        (tmp_path / f"test_{name}.py").write_text(
            "import pytest\n"
            f"pytestmark = {markers}\n"
            f"@pytest.mark.parametrize('value', range({count}))\n"
            f"def test_{name}(value):\n    assert value >= 0\n",
            encoding="utf-8",
        )
    return tmp_path


def _run(suite: Path, *args: str, paths: tuple[str, ...] = (".",)):
    environment = os.environ.copy()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONPATH"] = str(_ROOT)
    environment.pop("PYTEST_ADDOPTS", None)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "scripts.ci_shard",
            "-c",
            str(suite / "pytest.ini"),
            f"--rootdir={suite}",
            f"--confcutdir={suite}",
            "-q",
            "-W",
            "error",
            *args,
            *paths,
        ],
        cwd=suite,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _shard(suite: Path, index: int, total: int = 4, *args: str):
    output = suite / "artifacts" / str(index)
    result = _run(
        suite,
        f"--ci-shard-index={index}",
        f"--ci-shard-total={total}",
        f"--ci-shard-output={output}",
        *args,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def test_shards_are_a_deterministic_disjoint_complete_partition(shard_suite):
    manifests = [
        _shard(shard_suite, index, 4, "--collect-only") for index in range(1, 5)
    ]
    selected = manifests[0]["selected_nodeids"]
    assert len(selected) == 11
    assert all(manifest["selected_nodeids"] == selected for manifest in manifests)
    assigned = [
        nodeid for manifest in manifests for nodeid in manifest["assigned_nodeids"]
    ]
    assert len(assigned) == len(set(assigned)) == len(selected)
    assert set(assigned) == set(selected)
    for manifest in manifests:
        assert manifest["collect_only"] is True
        assert manifest["executed_nodeids"] == []
        assert manifest["test_duration_seconds"] == 0
        assert manifest["estimated_loads"] == [5, 3, 2, 1]
        assert all(
            manifest["module_assignments"][nodeid.split("::", 1)[0]]
            == manifest["index"]
            for nodeid in manifest["assigned_nodeids"]
        )
    repeated = _shard(shard_suite, 1, 4, "--collect-only")
    assert repeated["assigned_nodeids"] == manifests[0]["assigned_nodeids"]
    assert repeated["module_assignments"] == manifests[0]["module_assignments"]


def test_marker_and_keyword_selection_precede_sharding(shard_suite):
    manifests = [
        _shard(
            shard_suite,
            index,
            4,
            "--collect-only",
            "-m",
            "cpu or (neuron and not cuda)",
            "-k",
            "not other and not cpu[0]",
        )
        for index in range(1, 5)
    ]
    selected = manifests[0]["selected_nodeids"]
    assert len(selected) == 7
    assert all("cuda" not in nodeid and "other" not in nodeid for nodeid in selected)
    assert "test_cpu.py::test_cpu[0]" not in selected
    assigned = [
        nodeid for manifest in manifests for nodeid in manifest["assigned_nodeids"]
    ]
    assert len(assigned) == len(set(assigned)) == len(selected)
    assert set(assigned) == set(selected)


def test_unsharded_execution_is_unchanged(shard_suite):
    result = _run(shard_suite)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "11 passed" in result.stdout
    assert not (shard_suite / "artifacts").exists()


def test_execution_records_real_timings_and_skipped_cases(shard_suite):
    (shard_suite / "test_cpu.py").write_text(
        "import time\nimport pytest\n"
        "@pytest.mark.parametrize('value', range(5))\n"
        "def test_cpu(value):\n"
        "    if value == 0:\n        pytest.skip('expected skip')\n"
        "    time.sleep(0.002)\n",
        encoding="utf-8",
    )
    manifest = _shard(shard_suite, 1)
    assert manifest["exitstatus"] == 0
    assert manifest["collect_only"] is False
    assert manifest["executed_nodeids"] == manifest["assigned_nodeids"]
    assert set(manifest["test_durations_seconds"]) == set(manifest["assigned_nodeids"])
    assert manifest["collection_duration_seconds"] > 0
    assert manifest["test_duration_seconds"] >= 0.008
    assert manifest["test_duration_seconds"] == pytest.approx(
        sum(manifest["test_durations_seconds"].values())
    )


def test_failed_execution_finalizes_manifest(shard_suite):
    (shard_suite / "test_other.py").write_text(
        "def test_other():\n    assert False\n", encoding="utf-8"
    )
    result = _run(
        shard_suite,
        "--ci-shard-index=4",
        "--ci-shard-total=4",
        "--ci-shard-output=failed-artifacts",
    )
    assert result.returncode == 1, result.stdout + result.stderr
    manifest = json.loads(
        (shard_suite / "failed-artifacts" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["exitstatus"] == 1
    assert manifest["executed_nodeids"] == manifest["assigned_nodeids"]


@pytest.mark.parametrize(
    "options",
    [
        ("--ci-shard-index=1",),
        ("--ci-shard-total=4", "--ci-shard-output=artifacts"),
        ("--ci-shard-index=0", "--ci-shard-total=4", "--ci-shard-output=artifacts"),
        ("--ci-shard-index=5", "--ci-shard-total=4", "--ci-shard-output=artifacts"),
        ("--ci-shard-index=1", "--ci-shard-total=0", "--ci-shard-output=artifacts"),
        ("--ci-shard-index=1", "--ci-shard-total=4", "--ci-shard-output="),
        (
            "--ci-shard-index=1",
            "--ci-shard-index=2",
            "--ci-shard-total=4",
            "--ci-shard-output=artifacts",
        ),
    ],
)
def test_invalid_or_incomplete_options_fail(shard_suite, options):
    result = _run(shard_suite, *options)
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert not (shard_suite / "artifacts" / "manifest.json").exists()


def test_duplicate_selected_nodeids_fail(shard_suite):
    result = _run(
        shard_suite,
        "--ci-shard-index=1",
        "--ci-shard-total=4",
        "--ci-shard-output=artifacts",
        "--keep-duplicates",
        paths=("test_cpu.py", "test_cpu.py"),
    )
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, (
        result.stdout + result.stderr
    )
    assert "unique selected test node IDs" in result.stdout + result.stderr
    assert not (shard_suite / "artifacts" / "manifest.json").exists()


def _duration_profile(suite, *, modules=None, provenance=None):
    profile = suite / "durations.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provenance": {"source_revision": "synthetic-fixture"}
                if provenance is None
                else provenance,
                "modules": {
                    "test_cpu.py": {"seconds": 10.0, "tests": 5},
                    "test_neuron.py": {"seconds": 60.0, "tests": 3},
                    "test_cuda.py": {"seconds": 60.0, "tests": 2},
                }
                if modules is None
                else modules,
            }
        ),
        encoding="utf-8",
    )
    return profile


def test_duration_shards_keep_every_module_and_record_profile_provenance(shard_suite):
    profile = _duration_profile(shard_suite)
    manifests = [
        _shard(
            shard_suite, index, 4, "--collect-only", f"--ci-shard-durations={profile}"
        )
        for index in range(1, 5)
    ]
    selected = manifests[0]["selected_nodeids"]
    assigned = [node for manifest in manifests for node in manifest["assigned_nodeids"]]
    assert len(assigned) == len(set(assigned)) == len(selected) == 11
    assert set(assigned) == set(selected)
    assert all(manifest["selected_nodeids"] == selected for manifest in manifests)
    assert all(
        manifest["estimated_loads"] == [60.0, 60.0, 13.0, 10.0]
        for manifest in manifests
    )
    assert all(manifest["estimated_load_units"] == "seconds" for manifest in manifests)
    assert all(
        manifest["duration_profile"]["provenance"]
        == {"source_revision": "synthetic-fixture"}
        for manifest in manifests
    )
    assert len({manifest["duration_profile"]["sha256"] for manifest in manifests}) == 1
    for manifest in manifests:
        assert all(
            manifest["module_assignments"][node.split("::", 1)[0]] == manifest["index"]
            for node in manifest["assigned_nodeids"]
        )
    repeated = _shard(
        shard_suite, 1, 4, "--collect-only", f"--ci-shard-durations={profile}"
    )
    assert repeated["assigned_nodeids"] == manifests[0]["assigned_nodeids"]


def test_duration_weights_scale_after_marker_and_keyword_selection(shard_suite):
    profile = _duration_profile(shard_suite)
    manifests = [
        _shard(
            shard_suite,
            index,
            2,
            "--collect-only",
            f"--ci-shard-durations={profile}",
            "-m",
            "cpu or (neuron and not cuda)",
            "-k",
            "not other and not cpu[0]",
        )
        for index in range(1, 3)
    ]
    assert all(manifest["estimated_loads"] == [60.0, 8.0] for manifest in manifests)
    selected = manifests[0]["selected_nodeids"]
    assigned = [node for manifest in manifests for node in manifest["assigned_nodeids"]]
    assert len(assigned) == len(set(assigned)) == len(selected) == 7
    assert set(assigned) == set(selected)


@pytest.mark.parametrize(
    "seconds,tests",
    [
        (True, 1),
        (0, 1),
        (-1, 1),
        (float("nan"), 1),
        (float("inf"), 1),
        ("1", 1),
        (1, True),
        (1, 0),
        (1, -1),
        (1, 1.5),
    ],
)
def test_invalid_duration_records_fail_before_writing_artifacts(
    shard_suite, seconds, tests
):
    profile = _duration_profile(
        shard_suite, modules={"test_cpu.py": {"seconds": seconds, "tests": tests}}
    )
    result = _run(
        shard_suite,
        "--ci-shard-index=1",
        "--ci-shard-total=4",
        "--ci-shard-output=bad-artifacts",
        f"--ci-shard-durations={profile}",
    )
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert "duration profile" in result.stdout + result.stderr
    assert not (shard_suite / "bad-artifacts").exists()


@pytest.mark.parametrize(
    "mode",
    ["missing", "malformed", "version", "empty", "provenance", "alone", "duplicate"],
)
def test_duration_profile_admission_is_explicit_and_strict(shard_suite, mode):
    profile = _duration_profile(shard_suite)
    if mode == "missing":
        profile.unlink()
    elif mode == "malformed":
        profile.write_text("{", encoding="utf-8")
    elif mode in ("version", "empty", "provenance"):
        data = json.loads(profile.read_text())
        if mode == "version":
            data["schema_version"] = True
        elif mode == "empty":
            data["modules"] = {}
        else:
            data["provenance"] = {}
        profile.write_text(json.dumps(data), encoding="utf-8")
    options = (
        []
        if mode == "alone"
        else [
            "--ci-shard-index=1",
            "--ci-shard-total=4",
            "--ci-shard-output=bad-artifacts",
        ]
    )
    options.append(f"--ci-shard-durations={profile}")
    if mode == "duplicate":
        options.append(f"--ci-shard-durations={profile}")
    result = _run(shard_suite, *options)
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert not (shard_suite / "bad-artifacts").exists()


def test_duration_execution_keeps_skips_and_actual_timings(shard_suite):
    profile = _duration_profile(shard_suite)
    (shard_suite / "test_cpu.py").write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('value', range(5))\n"
        "def test_cpu(value):\n"
        "    if value == 0:\n        pytest.skip('expected skip')\n",
        encoding="utf-8",
    )
    manifest = _shard(shard_suite, 1, 1, f"--ci-shard-durations={profile}")
    assert manifest["exitstatus"] == 0
    assert manifest["collect_only"] is False
    assert len(manifest["executed_nodeids"]) == 11
    assert (
        manifest["executed_nodeids"]
        == manifest["assigned_nodeids"]
        == manifest["selected_nodeids"]
    )
    assert set(manifest["test_durations_seconds"]) == set(manifest["assigned_nodeids"])
    assert manifest["test_duration_seconds"] > 0
    assert manifest["test_duration_seconds"] == pytest.approx(
        sum(manifest["test_durations_seconds"].values())
    )


def _plugin():
    return ci_shard


def test_outcomes_include_all_phases_and_exact_junit_ids(shard_suite):
    import xml.etree.ElementTree as ET

    (shard_suite / "test_outcomes.py").write_text(
        "import pytest\n"
        "def test_pass(): pass\n"
        "@pytest.mark.skip(reason='not available')\n"
        "def test_skip(): pass\n"
        "@pytest.mark.xfail(reason='known defect')\n"
        "def test_xfail(): assert False\n"
        "@pytest.mark.xfail(reason='fixed defect', strict=False)\n"
        "def test_xpass(): pass\n"
        "@pytest.fixture\n"
        "def broken():\n    yield\n    assert False, 'teardown defect'\n"
        "def test_teardown(broken): pass\n"
        "def test_subtest(subtests):\n"
        "    with subtests.test('child'): assert False, 'child defect'\n"
    )
    output = shard_suite / "artifacts" / "outcomes"
    result = _run(
        shard_suite,
        "--ci-shard-index=1",
        "--ci-shard-total=1",
        f"--ci-shard-output={output}",
        f"--junitxml={output / 'junit.xml'}",
    )
    assert result.returncode == 1, result.stdout + result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    outcomes = manifest["test_outcomes"]
    assert (
        set(outcomes)
        == set(manifest["assigned_nodeids"])
        == set(manifest["executed_nodeids"])
    )
    for name, expected in (
        ("pass", "passed"),
        ("skip", "skipped"),
        ("xfail", "xfailed"),
        ("xpass", "xpassed"),
        ("teardown", "failed"),
        ("subtest", "failed"),
    ):
        assert outcomes[f"test_outcomes.py::test_{name}"]["outcome"] == expected
    assert outcomes["test_outcomes.py::test_skip"]["reason"] == "Skipped: not available"
    assert outcomes["test_outcomes.py::test_xfail"]["wasxfail"] == "known defect"
    assert outcomes["test_outcomes.py::test_teardown"]["phases"] == {
        "setup": "passed",
        "call": "passed",
        "teardown": "failed",
    }
    cases = ET.parse(output / "junit.xml").findall(".//testcase")
    ids = [
        case.find("./properties/property[@name='ci_nodeid']").get("value")
        for case in cases
    ]
    # A failing subtest and its parent may each emit a JUnit entry, but both
    # retain the authored node ID rather than reconstructing parameter names.
    assert set(ids) == set(manifest["assigned_nodeids"])
    assert manifest["source_snapshot"]["unchanged"] is True


def test_recovery_filters_whole_modules_after_global_selection(shard_suite):
    output = shard_suite / "artifacts" / "retry"
    result = _run(
        shard_suite,
        "--ci-shard-index=1",
        "--ci-shard-total=1",
        f"--ci-shard-output={output}",
        "--ci-recovery-module=test_other.py",
        "-m",
        "cpu",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["selected_nodeids"]) == 6
    assert manifest["assigned_nodeids"] == ["test_other.py::test_other[0]"]
    assert manifest["recovery_modules"] == ["test_other.py"]
    assert set(manifest["source_snapshot"]["modules"]) == {"test_other.py"}
    # CUDA-only modules are part of global support, since their imports are
    # measured even though their tests do not belong to this lane.
    assert "test_cuda.py" in manifest["source_snapshot"]["config"]["files"]
    assert "test_cpu.py" not in manifest["source_snapshot"]["config"]["files"]


def test_recovery_rejects_absent_or_partially_named_modules(shard_suite):
    for module in ("missing.py", "test_cpu.py::test_cpu[0]", "test_cuda.py"):
        result = _run(
            shard_suite,
            "--ci-shard-index=1",
            "--ci-shard-total=1",
            "--ci-shard-output=artifacts",
            f"--ci-recovery-module={module}",
            "-m",
            "cpu",
        )
        assert result.returncode == pytest.ExitCode.USAGE_ERROR, (
            result.stdout + result.stderr
        )


def test_snapshots_distinguish_leaf_changes_from_shared_resources(shard_suite):
    plugin = _plugin()
    modules = [f"test_{name}.py" for name in ("cpu", "cuda", "neuron", "other")]
    (shard_suite / "data.asc").write_text("geometry")
    (shard_suite / "_fixtures.py").write_text("VALUE = 1\n")
    before = plugin._config_snapshot(shard_suite, modules)
    assert before["selective_recovery_safe"] is True
    with (shard_suite / "test_cpu.py").open("a") as stream:
        stream.write("\n# leaf-only patch\n")
    assert plugin._config_snapshot(shard_suite, modules) == before
    (shard_suite / "data.asc").write_text("changed geometry")
    assert plugin._config_snapshot(shard_suite, modules)["sha256"] != before["sha256"]
    shared = shard_suite / "test_helper.py"
    shared.write_text("VALUE = 1\n")
    with (shard_suite / "test_other.py").open("a") as stream:
        stream.write("\nfrom test_helper import VALUE\n")
    snapshot = plugin._config_snapshot(shard_suite, [*modules, "test_helper.py"])
    assert snapshot["shared_test_modules"] == ["test_helper.py"]
    assert "test_helper.py" in snapshot["files"]
    shared.write_text("VALUE = 2\n")
    assert (
        plugin._config_snapshot(shard_suite, [*modules, "test_helper.py"])["sha256"]
        != snapshot["sha256"]
    )


def test_unknown_dynamic_test_imports_disable_selective_reuse(shard_suite):
    plugin = _plugin()
    with (shard_suite / "test_other.py").open("a") as stream:
        stream.write(
            "\nimport importlib\ndef load(name): return importlib.import_module(name)\n"
        )
    snapshot = plugin._config_snapshot(shard_suite, ["test_other.py", "test_cpu.py"])
    assert snapshot["selective_recovery_safe"] is False
    assert snapshot["unsafe_test_imports"] == [
        {"module": "test_other.py", "kind": "unresolved-dynamic-import"}
    ]
    assert "test_cpu.py" in snapshot["files"]
    assert "test_other.py" in snapshot["files"]


def test_source_change_during_execution_is_not_reusable(shard_suite):
    (shard_suite / "test_mutator.py").write_text(
        "from pathlib import Path\n"
        "def test_mutator():\n"
        "    path = Path(__file__)\n"
        "    path.write_text(path.read_text() + '\\n# changed during test\\n')\n"
    )
    output = shard_suite / "artifacts" / "mutator"
    result = _run(
        shard_suite,
        "--ci-shard-index=1",
        "--ci-shard-total=1",
        f"--ci-shard-output={output}",
    )
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, (
        result.stdout + result.stderr
    )
    snapshot = json.loads((output / "manifest.json").read_text())["source_snapshot"]
    assert snapshot["unchanged"] is False
    assert (
        snapshot["modules"]["test_mutator.py"]
        != snapshot["final"]["modules"]["test_mutator.py"]
    )
    assert "modules.test_mutator.py (modified)" in snapshot["error"]
    evidence = output / "source-drift-execution.json"
    diagnostic = json.loads(evidence.read_text())
    assert diagnostic["phase"] == "execution"
    assert diagnostic["before"]["modules"] == snapshot["modules"]
    assert diagnostic["after"]["modules"] == snapshot["final"]["modules"]
    assert {"path": ["modules", "test_mutator.py"], "change": "modified"} in diagnostic[
        "changes"
    ]
    assert str(evidence) in result.stdout + result.stderr


def test_snapshots_are_json_stable_and_global_git_inventory_is_complete(tmp_path):
    plugin = _plugin()
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("artifacts/\n")
    (tmp_path / "test_leaf.py").write_text("def test_leaf(): pass\n")
    (tmp_path / "guide.md").write_text("user docs")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    before = plugin._config_snapshot(tmp_path, ["test_leaf.py"])
    assert before["inventory_policy"] == "git-tracked-and-nonignored"
    (tmp_path / "new_resource.hoc").write_text("new resource")
    after = plugin._config_snapshot(tmp_path, ["test_leaf.py"])
    assert "new_resource.hoc" in after["files"]
    assert before["sha256"] != after["sha256"]
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "manifest.json").write_text("{}")
    assert plugin._config_snapshot(tmp_path, ["test_leaf.py"]) == after
    runtime = plugin._runtime_snapshot(tmp_path)
    assert json.loads(json.dumps(runtime)) == runtime


def test_sources_changed_during_collection_fail_before_execution(shard_suite):
    (shard_suite / "test_collection_mutation.py").write_text(
        "from pathlib import Path\n"
        "path = Path(__file__)\n"
        "path.write_text(path.read_text() + '\\n# changed during import\\n')\n"
        "def test_never_runs(): raise AssertionError('execution must not start')\n"
    )
    result = _run(
        shard_suite,
        "--ci-shard-index=1",
        "--ci-shard-total=1",
        "--ci-shard-output=artifacts",
    )
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, (
        result.stdout + result.stderr
    )
    assert "sources changed during collection" in result.stdout + result.stderr
    assert "execution must not start" not in result.stdout + result.stderr
    assert not (shard_suite / "artifacts" / "manifest.json").exists()
    assert (
        "config.files.test_collection_mutation.py (modified)"
        in result.stdout + result.stderr
    )
    evidence = shard_suite / "artifacts" / "source-drift-collection.json"
    diagnostic = json.loads(evidence.read_text())
    assert diagnostic["phase"] == "collection"
    assert (
        diagnostic["before"]["config"]["files"]["test_collection_mutation.py"]
        != diagnostic["after"]["config"]["files"]["test_collection_mutation.py"]
    )
    assert {
        "path": ["config", "files", "test_collection_mutation.py"],
        "change": "modified",
    } in diagnostic["changes"]


def test_source_difference_diagnostics_name_files_and_environment_fields():
    before = {
        "runtime": {
            "files": {"old.py": "a", "kept.py": "same"},
            "environment": {"packages": [["torch", "2.14.0"]]},
        },
        "config": {"files": {"modified.asc": "before"}},
    }
    after = {
        "runtime": {
            "files": {"new.py": "b", "kept.py": "same"},
            "environment": {"packages": [["torch", "2.15.0"]]},
        },
        "config": {"files": {"modified.asc": "after"}},
    }
    changes = ci_shard._source_differences(before, after)
    assert changes == [
        {"path": ["config", "files", "modified.asc"], "change": "modified"},
        {"path": ["runtime", "environment", "packages"], "change": "modified"},
        {"path": ["runtime", "files", "new.py"], "change": "added"},
        {"path": ["runtime", "files", "old.py"], "change": "removed"},
    ]


def test_source_drift_messages_are_bounded_and_complete_snapshots_are_saved(tmp_path):
    plugin = ci_shard._ShardPlugin(1, 1, tmp_path / "artifacts")
    before = {
        "config": {"files": {f"file_{index:03}.py": "before" for index in range(100)}}
    }
    after = {
        "config": {"files": {f"file_{index:03}.py": "after" for index in range(100)}}
    }
    message = plugin._source_drift("collection", before, after)
    assert "config.files.file_000.py (modified)" in message
    assert "file_099.py" not in message
    assert "more changes" in message
    assert len(message) < 2_000
    report = json.loads((plugin.output / "source-drift-collection.json").read_text())
    assert len(report["changes"]) == 100
    assert report["before"] == before
    assert report["after"] == after


def test_source_drift_diagnostic_write_failure_keeps_primary_failure_detail(
    tmp_path, monkeypatch
):
    plugin = ci_shard._ShardPlugin(1, 1, tmp_path / "artifacts")

    def denied(*args, **kwargs):
        raise OSError("diagnostic output unavailable")

    monkeypatch.setattr(Path, "mkdir", denied)
    message = plugin._source_drift(
        "collection",
        {"modules": {"test_a.py": "before"}},
        {"modules": {"test_a.py": "after"}},
    )
    assert "modules.test_a.py (modified)" in message
    assert "unable to write snapshots" in message


def test_untracked_bytecode_does_not_change_git_source_snapshot(tmp_path):
    plugin = _plugin()
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "test_leaf.py").write_text("def test_leaf(): pass\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    before = plugin._config_snapshot(tmp_path, ["test_leaf.py"])
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "test_leaf.cpython-312.pyc").write_bytes(b"generated")
    assert plugin._config_snapshot(tmp_path, ["test_leaf.py"]) == before


def test_shared_helper_import_edges_are_global_provenance(shard_suite):
    plugin = _plugin()
    (shard_suite / "test_helper.py").write_text("VALUE = 1\n")
    for name in ("a", "c"):
        (shard_suite / f"test_{name}.py").write_text(
            "from test_helper import VALUE\ndef test_value(): assert VALUE\n"
        )
    selected = [
        f"test_{name}.py" for name in ("a", "c", "cpu", "cuda", "neuron", "other")
    ]
    before = plugin._config_snapshot(shard_suite, selected)
    assert before["test_import_edges"] == [
        {"importer": "test_a.py", "module": "test_helper.py"},
        {"importer": "test_c.py", "module": "test_helper.py"},
    ]
    with (shard_suite / "test_a.py").open("a") as stream:
        stream.write("\n# isolated fixture patch\n")
    assert plugin._config_snapshot(shard_suite, selected) == before
    (shard_suite / "test_a.py").write_text("def test_value(): assert True\n")
    after = plugin._config_snapshot(shard_suite, selected)
    assert after["shared_test_modules"] == before["shared_test_modules"]
    assert after["test_import_edges"] != before["test_import_edges"]
    assert after["sha256"] != before["sha256"]


def test_aliased_unknown_dynamic_import_also_fails_closed(shard_suite):
    plugin = _plugin()
    with (shard_suite / "test_other.py").open("a") as stream:
        stream.write(
            "\nfrom importlib import import_module as load\ndef helper(name): return load(name)\n"
        )
    assert (
        plugin._config_snapshot(shard_suite, ["test_other.py"])[
            "selective_recovery_safe"
        ]
        is False
    )


def test_untracked_resource_containing_artifact_is_not_an_output_directory(tmp_path):
    plugin = _plugin()
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "artifact_geometry.asc").write_text("authored model geometry")
    (tmp_path / "tests" / "artifacts").mkdir(parents=True)
    (tmp_path / "tests" / "artifacts" / "example.asc").write_text(
        "authored fixture data"
    )
    snapshot = plugin._config_snapshot(tmp_path)
    assert "artifact_geometry.asc" in snapshot["files"]
    assert "tests/artifacts/example.asc" in snapshot["files"]


def test_shadowed_import_target_name_cannot_be_proven_static(shard_suite):
    plugin = _plugin()
    with (shard_suite / "test_other.py").open("a") as stream:
        stream.write(
            "\nimport importlib\nTARGET = 'test_cpu'\n"
            "def load(TARGET): return importlib.import_module(TARGET)\n"
        )
    assert (
        plugin._config_snapshot(shard_suite, ["test_other.py", "test_cpu.py"])[
            "selective_recovery_safe"
        ]
        is False
    )


@pytest.mark.parametrize("importer", ["conftest.py", "support.py", "dendra/bridge.py"])
def test_project_support_importers_make_test_targets_global(shard_suite, importer):
    path = shard_suite / importer
    path.parent.mkdir(exist_ok=True)
    path.write_text("from test_cpu import test_cpu\n")
    snapshot = ci_shard._config_snapshot(shard_suite, ["test_cpu.py", "test_other.py"])
    assert "test_cpu.py" in snapshot["shared_test_modules"]
    assert "test_cpu.py" in snapshot["files"]
    assert {"importer": importer, "module": "test_cpu.py"} in snapshot[
        "test_import_edges"
    ]
    assert "test_other.py" not in snapshot["files"]


@pytest.mark.parametrize("declaration", ["'test_cpu'", "['test_cpu']", "('test_cpu',)"])
def test_literal_pytest_plugins_are_global_test_support(shard_suite, declaration):
    (shard_suite / "conftest.py").write_text(f"pytest_plugins = {declaration}\n")
    snapshot = ci_shard._config_snapshot(shard_suite, ["test_cpu.py"])
    assert snapshot["selective_recovery_safe"] is True
    assert snapshot["shared_test_modules"] == ["test_cpu.py"]
    assert snapshot["test_import_edges"] == [
        {"importer": "conftest.py", "module": "test_cpu.py"}
    ]
    assert "test_cpu.py" in snapshot["files"]


@pytest.mark.parametrize(
    "source",
    [
        "pytest_plugins = get_plugins()\n",
        "pytest_plugins = []\npytest_plugins.append('test_cpu')\n",
        "from support import pytest_plugins\n",
    ],
)
def test_computed_pytest_plugins_disable_module_recovery(shard_suite, source):
    (shard_suite / "conftest.py").write_text(source)
    snapshot = ci_shard._config_snapshot(shard_suite, ["test_cpu.py"])
    assert snapshot["selective_recovery_safe"] is False
    assert snapshot["unsafe_test_imports"] == [
        {"module": "conftest.py", "kind": "dynamic-pytest-plugins"}
    ]
    assert "test_cpu.py" in snapshot["files"]


@pytest.mark.parametrize(
    "before,after",
    [
        ("FLAG = 1", "FLAG = 2"),
        ("return 1", "return 2"),
        ("CLASS_VALUE = 1", "CLASS_VALUE = 2"),
        ("def test_case(value=1):", "def test_case(value=2):"),
        ("def test_case(value=1):", "@pytest.mark.cpu\ndef test_case(value=1):"),
    ],
)
def test_only_plain_top_level_test_bodies_are_locally_reusable(
    shard_suite, before, after
):
    path = shard_suite / "test_isolation.py"
    source = (
        "import pytest\nFLAG = 1\n"
        "def helper():\n    return 1\n"
        "class Container:\n    CLASS_VALUE = 1\n"
        "def test_case(value=1):\n    assert False\n"
    )
    path.write_text(source)
    original = ci_shard._config_snapshot(shard_suite, ["test_isolation.py"])
    path.write_text(source.replace("assert False", "assert True"))
    body_only = ci_shard._config_snapshot(shard_suite, ["test_isolation.py"])
    assert body_only == original
    path.write_text(source.replace(before, after))
    changed_collection = ci_shard._config_snapshot(shard_suite, ["test_isolation.py"])
    assert changed_collection["collection_sources"] != original["collection_sources"]
    assert changed_collection["sha256"] != original["sha256"]


@pytest.fixture
def metadata_site(tmp_path, monkeypatch):
    site = tmp_path / "site"
    vendor = site / "setuptools" / "_vendor"
    vendor.mkdir(parents=True)
    monkeypatch.setattr(sys, "path", [str(site)])
    monkeypatch.setattr(
        ci_shard.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(submodule_search_locations=[str(vendor.parent)])
        if name == "setuptools"
        else None,
    )
    return site, vendor


def _write_metadata(root, name, version):
    directory = root / f"{name}-{version}.dist-info"
    directory.mkdir()
    (directory / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")
    return directory


def test_metadata_roots_are_physical_and_include_lazy_vendoring_without_imports(
    tmp_path, monkeypatch, metadata_site
):
    import builtins

    site, vendor = metadata_site
    _write_metadata(site, "installed", "1.0")
    _write_metadata(vendor, "bundled", "2.0")
    alias = tmp_path / "site_alias"
    alias.symlink_to(site, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "path",
        ["", str(tmp_path), str(site), str(site / ".." / "site"), str(alias)],
    )
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name != "setuptools", "metadata discovery must not execute setuptools"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert ci_shard._metadata_search_paths() == sorted(
        map(str, [tmp_path.resolve(), site.resolve(), vendor.resolve()])
    )
    before = ci_shard._runtime_snapshot(tmp_path)
    assert before["environment"]["packages"] == [
        ["bundled", "2.0"],
        ["installed", "1.0"],
    ]
    # Both pytest's duplicate root insertion and setuptools's lazy vendor path
    # publication leave the same physical metadata inventory.
    sys.path.extend([str(site), str(vendor), str(alias)])
    assert ci_shard._runtime_snapshot(tmp_path) == before


@pytest.mark.parametrize("location", ["installed", "vendored"])
@pytest.mark.parametrize("mutation", ["add", "remove", "version"])
def test_runtime_inventory_detects_real_installed_and_vendored_metadata_changes(
    tmp_path, metadata_site, location, mutation
):
    site, vendor = metadata_site
    root = site if location == "installed" else vendor
    distribution = _write_metadata(root, "tracked", "1.0")
    before = ci_shard._runtime_snapshot(tmp_path)
    if mutation == "add":
        _write_metadata(root, "new-package", "2.0")
    elif mutation == "remove":
        shutil.rmtree(distribution)
    else:
        (distribution / "METADATA").write_text("Name: tracked\nVersion: 2.0\n")
    after = ci_shard._runtime_snapshot(tmp_path)
    assert after["environment"]["packages"] != before["environment"]["packages"]
    assert after["sha256"] != before["sha256"]
    assert after["files"] == before["files"]


def test_distinct_metadata_roots_preserve_duplicate_distribution_records(
    tmp_path, metadata_site
):
    site, vendor = metadata_site
    for root in (site, vendor):
        _write_metadata(root, "shared-name", "1.0")
    assert ci_shard._runtime_snapshot(tmp_path)["environment"]["packages"] == [
        ["shared-name", "1.0"],
        ["shared-name", "1.0"],
    ]


def test_lazy_setuptools_import_does_not_look_like_collection_source_drift(shard_suite):
    (shard_suite / "test_vendor.py").write_text(
        "import setuptools\ndef test_vendor(): pass\n"
    )
    manifest = _shard(shard_suite, 1, 1, "--collect-only")
    assert "test_vendor.py::test_vendor" in manifest["selected_nodeids"]
    assert manifest["source_snapshot"]["unchanged"] is True
