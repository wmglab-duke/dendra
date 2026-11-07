from __future__ import annotations

import configparser
import json
import subprocess
import sys
import tomllib
from pathlib import Path

from scripts import run_ci_shards

SCRIPT = Path(__file__).parents[1] / "scripts" / "check_coverage_floors.py"
ROOT = Path(__file__).parents[1]
CPU_COVERAGE_CONFIG = ROOT / ".coveragerc.cpu"
COVERAGE_FLOORS_CONFIG = ROOT / "coverage-floors.toml"
CPU_KERNEL_OMISSIONS = {
    "dendra/models/integrators/triton/bt_kernel.py",
    "dendra/models/integrators/triton/bt_spd_kernel.py",
    "dendra/models/integrators/triton/dhs_kernel.py",
    "dendra/models/integrators/triton/dhs_kernel_bt.py",
    "dendra/models/integrators/triton/dhs_kernel_multi.py",
    "dendra/models/integrators/triton/pcr_kernel_thomas.py",
    "dendra/models/integrators/triton/t_kernel_thomas.py",
}


def _write_policy_fixture(tmp_path, *, actual=None, floor=80.0, include=True):
    module = "dendra/example.py"
    config = tmp_path / "coverage-floors.toml"
    config.write_text(
        f'[coverage-floors]\n"{module}" = {floor}\n',
        encoding="utf-8",
    )
    files = {}
    if include:
        files[module] = {"summary": {"percent_covered": actual}}
    report = tmp_path / "coverage.json"
    report.write_text(json.dumps({"files": files}), encoding="utf-8")
    return config, report


def _run_policy(config, report):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(report), "--config", str(config)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_coverage_policy_accepts_module_at_floor(tmp_path):
    config, report = _write_policy_fixture(tmp_path, actual=80.0)
    result = _run_policy(config, report)
    assert result.returncode == 0
    assert "PASS" in result.stdout


def test_coverage_policy_rejects_low_or_missing_modules(tmp_path):
    config, report = _write_policy_fixture(tmp_path, actual=79.99)
    low = _run_policy(config, report)
    assert low.returncode == 1
    assert "FAIL" in low.stdout

    config, report = _write_policy_fixture(tmp_path, include=False)
    missing = _run_policy(config, report)
    assert missing.returncode == 1
    assert "missing" in missing.stdout


def test_coverage_policy_reports_invalid_configuration(tmp_path):
    config = tmp_path / "coverage-floors.toml"
    config.write_text("[coverage]\n", encoding="utf-8")
    report = tmp_path / "coverage.json"
    report.write_text(json.dumps({"files": {}}), encoding="utf-8")

    result = _run_policy(config, report)
    assert result.returncode == 2
    assert "configuration error" in result.stderr


def test_coverage_floors_live_in_dedicated_configuration():
    with COVERAGE_FLOORS_CONFIG.open("rb") as stream:
        floors = tomllib.load(stream)["coverage-floors"]
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)

    assert floors
    assert "coverage-floors" not in project.get("tool", {}).get("dendra", {})


def test_cpu_coverage_omits_only_accelerator_kernel_bodies():
    config = configparser.ConfigParser()
    assert config.read(CPU_COVERAGE_CONFIG) == [str(CPU_COVERAGE_CONFIG)]

    omissions = {
        path.strip() for path in config.get("run", "omit").splitlines() if path.strip()
    }
    assert omissions == CPU_KERNEL_OMISSIONS
    assert all("*" not in path for path in omissions)

    # CPU-testable interfaces around the kernels must stay measurable.
    assert "dendra/models/integrators/triton/__init__.py" not in omissions
    assert "dendra/models/integrators/triton/_contracts.py" not in omissions
    assert "dendra/models/networks/netcon_bitpack_ops_triton.py" not in omissions
    assert "dendra/utils/gpu.py" not in omissions


def test_required_coverage_commands_use_cpu_scope():
    ci = (ROOT / ".gitlab-ci.yml").read_text(encoding="utf-8")
    testing_guide = (ROOT / "docs" / "testing.md").read_text(encoding="utf-8")
    command = run_ci_shards.command(ROOT, Path("ci-results/1"), 1)
    assert "python scripts/run_ci_shards.py" in ci
    assert "--cov-config=.coveragerc.cpu" in command
    assert "--cov=dendra" in command and "--cov-branch" in command
    assert "--cov-config=.coveragerc.cpu" in testing_guide
    assert "--fail-under=84" in ci
    aggregate = f"python scripts/combine_ci_coverage.py --shards {run_ci_shards.UNITS}"
    assert aggregate in ci and aggregate in testing_guide
    assert f"--ci-shard-total={run_ci_shards.UNITS}" in command
    assert "--ci-shard-durations=scripts/ci_durations.json" in command
    assert "scripts/ci_durations.json" in testing_guide


def test_shard_coverage_uses_checkout_relative_source_paths():
    config = configparser.ConfigParser()
    config.read(CPU_COVERAGE_CONFIG)

    assert config.getboolean("run", "relative_files")
    assert config.get("run", "source").strip() == "dendra"


def test_gitlab_full_suite_and_aggregate_are_required_gates():
    ci = (ROOT / ".gitlab-ci.yml").read_text(encoding="utf-8")
    shards = ci.split("\ntest_shards:\n", 1)[1].split("\ntest_coverage:\n", 1)[0]
    aggregate = ci.split("\ntest_coverage:\n", 1)[1].split(
        "\ntest_functional_compatibility:\n", 1
    )[0]
    release = ci.split("\nrelease_bump:\n", 1)[1]
    docs = ci.split("\nbuild_docs:\n", 1)[1].split("\nrelease_bump:\n", 1)[0]
    command = run_ci_shards.command(ROOT, Path("ci-results/1"), 1)

    assert f"parallel: {run_ci_shards.JOBS}" in shards
    assert '--job-index="$CI_NODE_INDEX"' in shards
    assert "--workers=1 --output-dir=ci-results" in shards
    assert ("-m", "cpu or (neuron and not cuda)") in set(zip(command, command[1:]))
    assert "--cov-report=" in command
    assert not any(option.startswith("--cov-fail-under") for option in command)
    assert "--ci-shard-index=1" in command
    assert f"--ci-shard-total={run_ci_shards.UNITS}" in command
    assert "--ci-shard-durations=scripts/ci_durations.json" in command
    assert "job: test_shards\n      artifacts: true" in aggregate
    assert "job: test_coverage\n      artifacts: false" in release
    # Jobs without rules default to except:merge_requests in GitLab. Feature
    # pushes with open MRs are suppressed, so docs must explicitly join MRs.
    assert "rules:\n    - when: on_success" in docs
    for job in (shards, aggregate):
        assert "extends: .full_test_rules" in job
        assert "allow_failure:" not in job
        assert "optional:" not in job
    rules = ci.split("\n.full_test_rules:\n", 1)[1].split("\ntest_base_install:\n", 1)[
        0
    ]
    assert '$CI_PIPELINE_SOURCE == "merge_request_event"' in rules
    assert '$CI_COMMIT_BRANCH == "main"' in rules
    assert '$CI_COMMIT_BRANCH == "develop"' in rules
