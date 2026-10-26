"""Exercise the CI plugin against independent, lightweight pytest suites."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


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
