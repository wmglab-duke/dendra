from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "check_coverage_floors.py"


def _write_policy_fixture(tmp_path, *, actual=None, floor=80.0, include=True):
    module = "dendra/example.py"
    config = tmp_path / "pyproject.toml"
    config.write_text(
        f'[tool.dendra.coverage-floors]\n"{module}" = {floor}\n',
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
    config = tmp_path / "pyproject.toml"
    config.write_text("[tool.dendra]\n", encoding="utf-8")
    report = tmp_path / "coverage.json"
    report.write_text(json.dumps({"files": {}}), encoding="utf-8")

    result = _run_policy(config, report)
    assert result.returncode == 2
    assert "configuration error" in result.stderr
