"""Deterministically partition the complete selected pytest suite for CI.

Load with ``python -m pytest -p scripts.ci_shard`` and provide the required
index, total and output options. Ordinary marker and keyword selection happens before
sharding. Modules stay together to preserve module-scoped fixture reuse; a
largest-first greedy assignment balances collected case counts by default.
An explicit duration profile balances estimated runtime while retaining every
selected test. The manifest records actual durations and profile provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import pytest


class _UniqueOption(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            raise argparse.ArgumentError(self, "may only be specified once")
        setattr(namespace, self.dest, values)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("ci-shard", "complete-suite CI sharding")
    group.addoption(
        "--ci-shard-index",
        type=int,
        default=None,
        action=_UniqueOption,
        help="One-based index of this CI shard (requires total and output).",
    )
    group.addoption(
        "--ci-shard-total",
        type=int,
        default=None,
        action=_UniqueOption,
        help="Total number of CI shards (requires index and output).",
    )
    group.addoption(
        "--ci-shard-output",
        default=None,
        action=_UniqueOption,
        help="Unique shard artifact directory containing manifest.json.",
    )

    group.addoption(
        "--ci-shard-durations",
        default=None,
        action=_UniqueOption,
        help="Optional measured module-duration JSON profile (requires sharding).",
    )


def _load_duration_profile(path: Path) -> tuple[dict, float, dict]:
    raw = path.read_bytes()
    profile = json.loads(raw)
    if (
        not isinstance(profile, dict)
        or type(profile.get("schema_version")) is not int
        or profile["schema_version"] != 1
    ):
        raise ValueError("duration profile schema_version must equal 1")
    modules = profile.get("modules")
    if not isinstance(modules, dict) or not modules:
        raise ValueError("duration profile modules must be a nonempty mapping")
    for module, record in modules.items():
        if not isinstance(module, str) or not module or not isinstance(record, dict):
            raise ValueError("duration profile module entries must name module records")
        seconds, tests = record.get("seconds"), record.get("tests")
        if (
            type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError(
                f"duration profile {module!r} seconds must be finite and positive"
            )
        if type(tests) is not int or tests <= 0:
            raise ValueError(
                f"duration profile {module!r} tests must be a positive integer"
            )
    provenance = profile.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("duration profile provenance must be a nonempty mapping")
    fallback = sum(record["seconds"] for record in modules.values()) / sum(
        record["tests"] for record in modules.values()
    )
    if not math.isfinite(fallback) or fallback <= 0:
        raise ValueError("duration profile fallback must be finite and positive")
    return (
        modules,
        fallback,
        {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "provenance": provenance,
            "fallback_seconds_per_test": fallback,
        },
    )


def _module_weights(counts: dict[str, int], profile: tuple | None) -> dict:
    if profile is None:
        return dict(counts)
    modules, fallback, _metadata = profile
    weights = {
        module: count
        * (
            modules[module]["seconds"] / modules[module]["tests"]
            if module in modules
            else fallback
        )
        for module, count in counts.items()
    }
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights.values()):
        raise pytest.UsageError("CI duration profile produced invalid module weights")
    return weights


def pytest_configure(config: pytest.Config) -> None:
    duration_path = config.getoption("ci_shard_durations")
    index = config.getoption("ci_shard_index")
    total = config.getoption("ci_shard_total")
    output = config.getoption("ci_shard_output")
    if index is None and total is None and output is None and duration_path is None:
        return
    if index is None or total is None or output is None:
        raise pytest.UsageError(
            "--ci-shard-index, --ci-shard-total, and --ci-shard-output "
            "must be supplied together"
        )
    if total < 1 or not 1 <= index <= total:
        raise pytest.UsageError("CI shard index must satisfy 1 <= index <= total")
    if not output.strip():
        raise pytest.UsageError("--ci-shard-output must name a directory")
    if config.getoption("numprocesses", default=None):
        raise pytest.UsageError("CI sharding requires one pytest process per shard")
    profile = None
    if duration_path is not None:
        try:
            profile = _load_duration_profile(Path(duration_path))
        except (OSError, ValueError, OverflowError) as exc:
            raise pytest.UsageError(
                f"Invalid CI duration profile {duration_path!r}: {exc}"
            ) from exc
    config.pluginmanager.register(
        _ShardPlugin(index, total, Path(output), profile), "ci-shard"
    )


class _ShardPlugin:
    def __init__(
        self, index: int, total: int, output: Path, profile: tuple | None = None
    ) -> None:
        self.profile = profile
        self.index = index
        self.total = total
        self.output = output
        self.manifest: dict | None = None
        self.started = 0.0
        self.durations: dict[str, float] = {}

    def pytest_sessionstart(self) -> None:
        self.started = time.perf_counter()

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(
        self, config: pytest.Config, items: list[pytest.Item]
    ):
        # As the outermost wrapper, resume after ordinary selection and any
        # other wrappers have finished modifying the selected item list.
        yield
        selected = [item.nodeid for item in items]
        if len(selected) != len(set(selected)):
            raise pytest.UsageError(
                "CI sharding requires unique selected test node IDs"
            )

        counts = Counter(nodeid.split("::", 1)[0] for nodeid in selected)
        weights = _module_weights(counts, self.profile)
        loads = [0] * self.total
        assignments = {}
        for module, weight in sorted(
            weights.items(), key=lambda pair: (-pair[1], pair[0])
        ):
            shard = min(range(self.total), key=lambda value: (loads[value], value))
            assignments[module] = shard + 1
            loads[shard] += weight

        assigned, deselected = [], []
        for item in items:
            module = item.nodeid.split("::", 1)[0]
            (assigned if assignments[module] == self.index else deselected).append(item)
        items[:] = assigned
        config.hook.pytest_deselected(items=deselected)
        self.manifest = {
            "schema_version": 1,
            "index": self.index,
            "total": self.total,
            "selected_nodeids": selected,
            "assigned_nodeids": [item.nodeid for item in assigned],
            "module_assignments": assignments,
            "estimated_loads": loads,
            "collection_duration_seconds": time.perf_counter() - self.started,
            "test_duration_seconds": 0.0,
            "executed_nodeids": [],
            "test_durations_seconds": {},
            "exitstatus": None,
            "collect_only": bool(config.option.collectonly),
        }
        if self.profile is not None:
            self.manifest["duration_profile"] = self.profile[2]
            self.manifest["estimated_load_units"] = "seconds"
        self._write_manifest()

    def pytest_collection_finish(self) -> None:
        if self.manifest is not None:
            self.manifest["collection_duration_seconds"] = (
                time.perf_counter() - self.started
            )

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self.durations[report.nodeid] = (
            self.durations.get(report.nodeid, 0.0) + report.duration
        )

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        if self.manifest is None:
            return
        # A valid partition can be empty when there are more shards than
        # selected modules. An empty overall selection still fails normally.
        if (
            exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED
            and self.manifest["selected_nodeids"]
            and not self.manifest["assigned_nodeids"]
        ):
            exitstatus = pytest.ExitCode.OK
            session.exitstatus = exitstatus
        self.manifest.update(
            exitstatus=int(exitstatus),
            executed_nodeids=[
                nodeid
                for nodeid in self.manifest["assigned_nodeids"]
                if nodeid in self.durations
            ],
            test_duration_seconds=sum(self.durations.values()),
            test_durations_seconds=self.durations,
        )
        self._write_manifest()

    def _write_manifest(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / "manifest.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
