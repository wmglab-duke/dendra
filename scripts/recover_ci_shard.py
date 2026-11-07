"""Recover a CI unit with complete module retries and owned coverage contexts.

Original artifacts are immutable. Test IDs must remain unchanged. Shared test
support, runtime or global configuration changes require broader invalidation.
Legacy/contextless coverage cannot satisfy partial recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path, PurePosixPath, PureWindowsPath

import coverage

try:
    from .ci_shard import _config_snapshot, _runtime_snapshot
except ImportError:
    from ci_shard import _config_snapshot, _runtime_snapshot


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _ids(manifest, key):
    values = manifest.get(key)
    _require(
        isinstance(values, list)
        and values
        and all(isinstance(x, str) and x for x in values),
        f"{key}: unique nonempty IDs required",
    )
    _require(len(values) == len(set(values)), f"{key}: duplicate IDs")
    return values


def _module(nodeid):
    return nodeid.split("::", 1)[0]


def _relative(path):
    _require(
        isinstance(path, str)
        and path
        and not PurePosixPath(path).is_absolute()
        and not PureWindowsPath(path).is_absolute()
        and not PureWindowsPath(path).drive
        and ".." not in PurePosixPath(path).parts
        and ".." not in PureWindowsPath(path).parts,
        f"Unsafe or absolute source path: {path!r}",
    )
    return path


def _read_artifact(directory):
    files = {
        name: directory / name for name in ("manifest.json", ".coverage", "junit.xml")
    }
    _require(
        all(path.is_file() for path in files.values()),
        f"{directory}: missing manifest, coverage or JUnit artifact",
    )
    paths = list(directory.rglob("*"))
    _require(
        not any(path.is_symlink() for path in paths),
        f"{directory}: symlink artifacts are unsupported",
    )
    files = {
        path.relative_to(directory).as_posix(): path for path in paths if path.is_file()
    }
    manifest = json.loads(files["manifest.json"].read_text())
    _require(
        manifest.get("schema_version") == 1 and manifest.get("collect_only") is False,
        f"{directory}: executed schema-1 artifact required",
    )
    assigned = _ids(manifest, "assigned_nodeids")
    selected = _ids(manifest, "selected_nodeids")
    _require(set(assigned) <= set(selected), f"{directory}: unselected assigned IDs")
    _require(
        set(manifest.get("test_outcomes", {})) == set(assigned),
        f"{directory}: incomplete outcome inventory",
    )
    executed = manifest.get("executed_nodeids")
    _require(
        isinstance(executed, list)
        and all(isinstance(node, str) for node in executed)
        and len(executed) == len(set(executed))
        and set(executed) <= set(assigned),
        f"{directory}: invalid executed inventory",
    )
    snapshot = manifest.get("source_snapshot", {})
    modules = {_module(node) for node in assigned}
    _require(
        set(snapshot.get("modules", {})) == modules,
        f"{directory}: missing module source hashes",
    )
    _require(
        snapshot.get("unchanged") is True,
        f"{directory}: source changed during execution",
    )
    _require(
        snapshot.get("final")
        == {name: snapshot.get(name) for name in ("modules", "runtime", "config")},
        f"{directory}: final source attestation differs",
    )
    contexts = manifest.get("coverage_contexts", {})
    _require(
        contexts.get("mode") == "module"
        and contexts.get("active") is True
        and contexts.get("owned_only") is True,
        f"{directory}: owned module coverage is required; legacy/contextless recovery is unsupported",
    )
    _require(
        set(contexts.get("modules", {})) == modules,
        f"{directory}: incomplete owned context inventory",
    )
    names = [contexts.get("global"), *contexts["modules"].values()]
    _require(
        all(isinstance(name, str) and name for name in names)
        and len(names) == len(set(names)),
        f"{directory}: invalid or aliased context names",
    )
    data = coverage.CoverageData(basename=str(files[".coverage"].resolve()))
    try:
        data.read()
        _require(data.has_arcs(), f"{directory}: branch coverage is required")
        measured = sorted(data.measured_files())
        _require(measured, f"{directory}: no measured source files")
        for filename in measured:
            _relative(filename)
        for context in data.measured_contexts() - set(names):
            data.set_query_contexts(["^" + re.escape(context) + "$"])
            _require(
                not any(data.arcs(name) for name in measured),
                f"{directory}: unknown or unowned nonempty coverage context {context!r}",
            )
        data.set_query_contexts(None)
    finally:
        data.close()
    xml = ET.parse(files["junit.xml"])
    cases = {}
    for case in xml.findall(".//testcase"):
        properties = [
            prop.get("value")
            for prop in case.findall("./properties/property")
            if prop.get("name") == "ci_nodeid"
        ]
        _require(
            len(properties) == 1 and properties[0] in assigned,
            f"{directory}: JUnit requires one exact ci_nodeid property per case",
        )
        cases.setdefault(properties[0], []).append(case)
    return {
        "path": directory,
        "files": files,
        "hashes": {name: _sha(path) for name, path in files.items()},
        "manifest": manifest,
        "data": data,
        "cases": cases,
    }


def _validate_outcome(record, *, previous=None):
    outcome, phases = record.get("outcome"), record.get("phases", {})
    _require(
        isinstance(phases, dict) and set(phases) <= {"setup", "call", "teardown"},
        "Invalid test phase inventory",
    )
    _require(
        phases.get("teardown") == "passed",
        "Test teardown did not complete successfully",
    )
    if outcome in {"passed", "xpassed"}:
        _require(
            phases == {"setup": "passed", "call": "passed", "teardown": "passed"},
            "Passed outcome has incomplete or failed phases",
        )
    elif outcome in {"skipped", "xfailed"}:
        valid_phases = phases in (
            {"setup": "skipped", "teardown": "passed"},
            {"setup": "passed", "call": "skipped", "teardown": "passed"},
        )
        _require(
            record.get("reason") and valid_phases,
            "Skip/xfail requires a recorded reason and successful remaining phases",
        )
        if previous is not None:
            _require(
                all(
                    record.get(key) == previous.get(key)
                    for key in ("outcome", "reason", "wasxfail")
                ),
                "Replacement may not turn a failing/passing test into a skip or change its expected-failure reason",
            )
    else:
        raise ValueError(f"Unsuccessful or incomplete test outcome: {outcome!r}")


def _case(artifact, nodeid, outcome):
    cases = artifact["cases"].get(nodeid, [])
    _require(len(cases) == 1, f"{nodeid}: exactly one final JUnit case required")
    case = cases[0]
    _require(
        not case.findall("failure") and not case.findall("error"),
        f"{nodeid}: unsuccessful JUnit replacement",
    )
    _require(
        bool(case.findall("skipped")) == (outcome in {"skipped", "xfailed"}),
        f"{nodeid}: JUnit/outcome disagreement",
    )
    return deepcopy(case)


def _duration(manifest, nodeid):
    value = manifest.get("test_durations_seconds", {}).get(nodeid)
    _require(
        type(value) in (int, float) and math.isfinite(value) and value >= 0,
        f"{nodeid}: missing or invalid execution duration",
    )
    return value


def _copy_context(source, target, source_name, target_name):
    try:
        source.set_query_contexts(["^" + re.escape(source_name) + "$"])
        target.set_context(target_name)
        target.add_arcs(
            {name: source.arcs(name) or [] for name in source.measured_files()}
        )
        tracers = {
            name: source.file_tracer(name)
            for name in source.measured_files()
            if source.file_tracer(name)
        }
        if tracers:
            target.add_file_tracers(tracers)
    finally:
        source.set_query_contexts(None)
        source.close()


def recover(
    original: Path, replacements: list[Path], output: Path, source_root: Path
) -> dict:
    """Publish a new complete unit, replacing entire modules without data union."""
    _require(replacements, "At least one complete module replacement is required")
    _require(not output.exists(), "Recovery output already exists")
    original, output, source_root = (
        original.resolve(),
        output.resolve(),
        source_root.resolve(),
    )
    inputs = [
        _read_artifact(original),
        *(_read_artifact(path.resolve()) for path in replacements),
    ]
    _require(
        all(not output.is_relative_to(item["path"]) for item in inputs),
        "Output must be separate from immutable input artifacts",
    )
    base = inputs[0]["manifest"]
    _require(
        type(base.get("exitstatus")) is int and base["exitstatus"] in {0, 1},
        "Original unit did not finish a normal pytest execution",
    )
    assigned = base["assigned_nodeids"]
    groups = {
        module: {node for node in assigned if _module(node) == module}
        for module in base["source_snapshot"]["modules"]
    }
    runtime = _runtime_snapshot(source_root)
    selected_modules = {_module(node) for node in base["selected_nodeids"]}
    config = _config_snapshot(source_root, selected_modules)
    _require(
        config.get("selective_recovery_safe") is True,
        "Unsafe dynamic test imports require broader invalidation",
    )
    owners = {}
    for item in inputs:
        snapshot = item["manifest"]["source_snapshot"]
        _require(
            snapshot["runtime"] == runtime,
            "Runtime/environment changed: invalidate all affected units",
        )
        _require(
            snapshot["config"] == config,
            "Global test configuration or shared support changed: invalidate all affected units",
        )
    for item in inputs[1:]:
        replacement = item["manifest"]
        _require(
            replacement["selected_nodeids"] == base["selected_nodeids"],
            "Replacement full selected inventory differs",
        )
        _require(replacement.get("exitstatus") == 0, "Replacement execution failed")
        _require(
            set(replacement["executed_nodeids"])
            == set(replacement["assigned_nodeids"]),
            "Replacement execution is incomplete",
        )
        for module, sha in replacement["source_snapshot"]["modules"].items():
            _require(
                module in groups and module not in owners,
                "Replacement has unknown or duplicate modules",
            )
            actual_ids = {
                node
                for node in replacement["assigned_nodeids"]
                if _module(node) == module
            }
            _require(
                actual_ids == groups[module],
                f"{module}: replacement must contain the complete unchanged module ID inventory",
            )
            _require(
                sha == _sha(source_root / _relative(module)),
                f"{module}: replacement source hash is stale",
            )
            owners[module] = item
    for module, sha in base["source_snapshot"]["modules"].items():
        if module not in owners:
            _require(
                sha == _sha(source_root / _relative(module)),
                f"{module}: changed module has no complete replacement",
            )

    final = deepcopy(base)
    final["test_outcomes"], final["test_durations_seconds"] = {}, {}
    junit = ET.Element("testsuites")
    suite = ET.SubElement(junit, "testsuite", name="pytest")
    for nodeid in assigned:
        item = owners.get(_module(nodeid), inputs[0])
        record = item["manifest"]["test_outcomes"][nodeid]
        _require(
            nodeid in item["manifest"]["executed_nodeids"],
            f"{nodeid}: test did not execute",
        )
        _validate_outcome(
            record,
            previous=base["test_outcomes"][nodeid] if item is not inputs[0] else None,
        )
        suite.append(_case(item, nodeid, record["outcome"]))
        final["test_outcomes"][nodeid] = deepcopy(record)
        final["test_durations_seconds"][nodeid] = _duration(item["manifest"], nodeid)
    suite.set("tests", str(len(assigned)))
    suite.set("errors", "0")
    suite.set("failures", "0")
    suite.set("skipped", str(sum(bool(case.findall("skipped")) for case in suite)))
    suite.set("time", str(sum(float(case.get("time", "0")) for case in suite)))
    final.update(
        exitstatus=0,
        executed_nodeids=list(assigned),
        test_duration_seconds=sum(final["test_durations_seconds"].values()),
    )
    for module, item in owners.items():
        final["source_snapshot"]["modules"][module] = item["manifest"][
            "source_snapshot"
        ]["modules"][module]
    final["source_snapshot"]["final"] = {
        name: deepcopy(final["source_snapshot"][name])
        for name in ("modules", "runtime", "config")
    }
    final["recovery"] = {
        "schema_version": 1,
        "replaced_modules": sorted(owners),
        "provenance": "recovery.json",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=output.parent))
    merged = None
    try:
        merged = coverage.CoverageData(basename=str(staging / ".coverage"))
        contexts = base["coverage_contexts"]
        _copy_context(inputs[0]["data"], merged, contexts["global"], contexts["global"])
        for module in groups:
            item = owners.get(module, inputs[0])
            _copy_context(
                item["data"],
                merged,
                item["manifest"]["coverage_contexts"]["modules"][module],
                contexts["modules"][module],
            )
        merged.write()
        merged.close()
        ET.ElementTree(junit).write(
            staging / "junit.xml", encoding="utf-8", xml_declaration=True
        )
        (staging / "manifest.json").write_text(json.dumps(final, indent=2) + "\n")
        evidence = []
        for index, item in enumerate(inputs):
            relative = Path("evidence") / (
                "original" if index == 0 else f"replacement-{index:03d}"
            )
            for name, path in item["files"].items():
                if name == ".coverage":
                    continue
                destination = staging / relative / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
            evidence.append(
                {
                    "path": str(item["path"]),
                    "evidence": relative.as_posix(),
                    "sha256": item["hashes"],
                    "exitstatus": item["manifest"]["exitstatus"],
                    "source_snapshot": item["manifest"]["source_snapshot"],
                    "previous_recovery": json.loads(
                        item["files"]["recovery.json"].read_text()
                    )
                    if "recovery.json" in item["files"]
                    else None,
                }
            )
        provenance = {
            "schema_version": 1,
            "replaced_modules": sorted(owners),
            "coverage_policy": "Original global context plus exactly one current owner per module; overridden original and replacement-global contexts are not unioned.",
            "inputs": evidence,
            "final_outcome_counts": {
                name: sum(
                    record["outcome"] == name
                    for record in final["test_outcomes"].values()
                )
                for name in {
                    record["outcome"] for record in final["test_outcomes"].values()
                }
            },
        }
        (staging / "recovery.json").write_text(json.dumps(provenance, indent=2) + "\n")
        (staging / "pytest.log").write_text(
            "Recovered unit; original and replacement executions remain unchanged. See recovery.json and evidence/ for preserved execution records and artifact hashes. Original coverage databases are identified by hash; final coverage replaces their owned contexts.\n"
        )
        for item in inputs:
            _require(
                item["hashes"]
                == {name: _sha(path) for name, path in item["files"].items()},
                "Input artifacts changed during recovery",
            )
            item["data"].close()
        _require(
            _runtime_snapshot(source_root) == runtime
            and _config_snapshot(source_root, selected_modules) == config,
            "Runtime or global configuration changed during recovery",
        )
        for module, sha in final["source_snapshot"]["modules"].items():
            _require(
                _sha(source_root / _relative(module)) == sha,
                f"{module}: source changed during recovery",
            )
        _require(not output.exists(), "Recovery output appeared during validation")
        staging.rename(output)
    except BaseException:
        if merged is not None:
            merged.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--replacement", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        recover(args.original, args.replacement, args.output, args.source_root)
    except (
        OSError,
        ValueError,
        KeyError,
        coverage.CoverageException,
        ET.ParseError,
    ) as error:
        parser.exit(2, f"CI recovery rejected: {error}\n")
    print(f"Recovered complete CI unit: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
