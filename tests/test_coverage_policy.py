from __future__ import annotations

import configparser
import json
import subprocess
import sys
import tomllib
from pathlib import Path

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
    assert "--cov-config=.coveragerc.cpu" in ci
    assert "--cov-config=.coveragerc.cpu" in testing_guide
    assert "--cov-fail-under=84" in ci
