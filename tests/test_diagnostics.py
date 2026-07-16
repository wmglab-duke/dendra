from __future__ import annotations

import json

import dendra as dn
import dendra.cli as cli
import dendra.diagnostics as diagnostics
from dendra.diagnostics import DoctorCheck, DoctorReport
from dendra.models.networks import netcon_bitpack_ops


def test_default_doctor_inspection_never_loads_or_builds_native_extension(
    monkeypatch,
):
    def forbidden_load():
        raise AssertionError("default doctor must not load the native extension")

    monkeypatch.setattr(netcon_bitpack_ops, "_load_extension", forbidden_load)

    report = diagnostics.collect_doctor_report()
    public_report = dn.doctor(output=False)

    assert isinstance(report, DoctorReport)
    assert isinstance(public_report, DoctorReport)
    assert report.probe_native_bitpack is False
    probe = next(
        check for check in report.checks if check.name == "native bitpack probe"
    )
    assert probe.status == "info"
    assert "skipped" in probe.message
    assert "no native build attempted" in report.to_text()


def test_doctor_reports_invalid_environment_policy(monkeypatch):
    def invalid_policy():
        raise ValueError("invalid native policy")

    monkeypatch.setattr(diagnostics, "current_native_extension_policy", invalid_policy)
    report = diagnostics.collect_doctor_report()

    policy = next(
        check for check in report.checks if check.name == "native extension policy"
    )
    assert policy.status == "error"
    assert report.ok is False


def test_native_probe_requires_explicit_opt_in(monkeypatch):
    calls = []
    monkeypatch.setattr(
        diagnostics,
        "_probe_native_bitpack",
        lambda: calls.append("probe") or {"loaded": True},
    )

    diagnostics.collect_doctor_report(probe_native_bitpack=False)
    assert calls == []

    report = diagnostics.collect_doctor_report(probe_native_bitpack=True)
    assert calls == ["probe"]
    probe = next(
        check for check in report.checks if check.name == "native bitpack probe"
    )
    assert probe.status == "ok"


def test_cli_doctor_supports_text_and_json(monkeypatch, capsys):
    report = DoctorReport(
        (DoctorCheck("runtime", "ok", "ready"),),
        require_cuda=False,
        probe_native_bitpack=False,
    )
    monkeypatch.setattr(cli, "collect_doctor_report", lambda **kwargs: report)

    assert cli.main(["doctor"]) == 0
    assert "Dendra doctor" in capsys.readouterr().out

    assert cli.main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["checks"][0]["message"] == "ready"
