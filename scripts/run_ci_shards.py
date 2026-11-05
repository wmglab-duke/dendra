"""Run/resume 40 complete CPU/NEURON units with bounded process concurrency.

GitLab keeps ten physical jobs, each owning four serial units by default.
Without a job index, a local invocation owns all units. This orchestrator never
changes pytest selection or splits a test module. Recovery replaces complete
modules and delegates outcome/coverage reconciliation to recover_ci_shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

try:
    from .recover_ci_shard import _case, _duration, _read_artifact, _validate_outcome
except ImportError:
    from recover_ci_shard import _case, _duration, _read_artifact, _validate_outcome

UNITS = 40
JOBS = 10
MARKERS = "cpu or (neuron and not cuda)"
THREADS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MAX_JOBS": "1",
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
}
ARTIFACTS = (
    "manifest.json",
    "junit.xml",
    ".coverage",
    "pytest.log",
    "invocation.json",
    "runner-context.json",
    "completion.json",
)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def units_for_job(index):
    if index is None:
        return list(range(1, UNITS + 1))
    if type(index) is not int or not 1 <= index <= JOBS:
        raise ValueError(f"job index must be between 1 and {JOBS}")
    return list(range(index, UNITS + 1, JOBS))


def repository():
    for parent in Path(__file__).resolve().parents:
        if (parent / "scripts/ci_shard.py").is_file() and (
            parent / "pyproject.toml"
        ).is_file():
            return parent
    raise ValueError("Run this script from a Dendra source checkout")


def validate_output(repo, output):
    """Keep runner artifacts outside the globally fingerprinted source inventory."""
    repo, output = repo.resolve(), output.resolve()
    for directory in (output, output.with_name(output.name + "-runner")):
        if directory.is_relative_to(repo):
            relative = directory.relative_to(repo).as_posix() + "/"
            status = subprocess.run(
                ["git", "check-ignore", "--quiet", "--", relative], cwd=repo
            ).returncode
            if status:
                raise ValueError(
                    f"output and runner directories must be Gitignored or outside the checkout: {directory}"
                )


def environment(output):
    env = dict(os.environ)
    scripts = str(Path(__file__).resolve().parent)
    env.update(THREADS)
    env["PYTHONPATH"] = scripts + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["COVERAGE_FILE"] = str(output / ".coverage")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def command(repo, output, index, *, total=UNITS, modules=None, collect=False):
    result = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "ci_shard",
        "tests",
        "-v",
        "-W",
        "error",
        "-m",
        MARKERS,
        f"--ci-shard-index={index}",
        f"--ci-shard-total={total}",
        f"--ci-shard-output={output}",
        "--ci-shard-durations=scripts/ci_durations.json",
        f"--junitxml={output / 'junit.xml'}",
        "--durations=25",
        "--cov=dendra",
        "--cov-branch",
        "--cov-config=.coveragerc.cpu",
        "--cov-report=",
    ]
    if collect:
        result.append("--collect-only")
    if modules:
        result.extend(f"--ci-recovery-module={module}" for module in modules)
    return result


def execute(repo, output, cmd, context):
    output.mkdir(parents=True)
    write(
        output / "runner-context.json",
        {
            key: context[key]
            for key in ("sha256", "configuration", "action")
            if key in context
        },
    )
    invocation = {"command": cmd, "started": stamp(), "cwd": str(repo)}
    invocation["revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    invocation["ci_job_url"] = os.environ.get("CI_JOB_URL")
    write(output / "invocation.json", invocation)
    started = time.monotonic()
    with (output / "pytest.log").open("wb") as log:
        status = subprocess.run(
            cmd, cwd=repo, env=environment(output), stdout=log, stderr=subprocess.STDOUT
        ).returncode
    write(
        output / "completion.json",
        {
            **invocation,
            "exitcode": status,
            "finished": stamp(),
            "seconds": time.monotonic() - started,
        },
    )
    seal(output)
    return status


def seal(directory):
    write(
        directory / "artifact-hashes.json",
        {
            path.relative_to(directory).as_posix(): digest(path.read_bytes())
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path != directory / "artifact-hashes.json"
        },
    )


def recover_unit(repo, original, replacement, recovered, log):
    """Use the same interpreter search path and cwd as pytest for attestation."""
    cmd = [
        sys.executable,
        "-m",
        "recover_ci_shard",
        "--original",
        str(original),
        "--replacement",
        str(replacement),
        "--output",
        str(recovered),
        "--source-root",
        str(repo),
    ]
    with log.open("wb") as stream:
        status = subprocess.run(
            cmd,
            cwd=repo,
            env=environment(recovered),
            stdout=stream,
            stderr=subprocess.STDOUT,
        ).returncode
    if status:
        raise ValueError(log.read_text()[-2000:].strip())
    shutil.copyfile(log, recovered / "recovery-command.log")
    return cmd


def context(repo, collection):
    snapshot = collection["source_snapshot"]
    profile = repo / "scripts/ci_durations.json"
    plugin = Path(__file__).with_name("ci_shard.py")
    recovery = Path(__file__).with_name("recover_ci_shard.py")
    config = {
        "schema_version": 1,
        "units": UNITS,
        "markers": MARKERS,
        "pytest_options": [
            "-v",
            "-W",
            "error",
            "--cov=dendra",
            "--cov-branch",
            "--cov-config=.coveragerc.cpu",
        ],
        "profile_sha256": digest(profile.read_bytes()),
        "threads": THREADS,
        "runner_sha256": digest(Path(__file__).read_bytes()),
        "plugin_sha256": digest(plugin.read_bytes()),
        "recovery_sha256": digest(recovery.read_bytes()),
        "runtime_sha256": snapshot["runtime"]["sha256"],
        "config_sha256": snapshot["config"]["sha256"],
    }
    selected = collection["selected_nodeids"]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Collection must contain unique nonempty selected IDs")
    return {
        "configuration": config,
        "sha256": digest(encoded(config)),
        "selected_nodeids": selected,
        "module_assignments": collection["module_assignments"],
        "estimated_loads": collection["estimated_loads"],
        "source_snapshot": snapshot,
    }


def assigned(current, unit):
    return [
        nodeid
        for nodeid in current["selected_nodeids"]
        if current["module_assignments"][nodeid.split("::", 1)[0]] == unit
    ]


def module_hashes(repo, nodeids):
    return {
        module: digest((repo / module).read_bytes())
        for module in sorted({nodeid.split("::", 1)[0] for nodeid in nodeids})
    }


def inspect_artifacts(directory, unit, current):
    """Admit authentic cached evidence; failures may still support recovery."""
    if directory.is_symlink() or any(
        path.is_symlink() for path in directory.rglob("*")
    ):
        raise ValueError("unit artifacts must be owned files, not symbolic links")
    hashes = read(directory / "artifact-hashes.json")
    actual_files = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path != directory / "artifact-hashes.json"
    }
    if set(hashes) != actual_files:
        raise ValueError("published artifact inventory changed")
    for name, expected in hashes.items():
        path = directory / name
        if (
            not path.resolve().is_relative_to(directory.resolve())
            or digest(path.read_bytes()) != expected
        ):
            raise ValueError(f"changed artifact: {name}")
    for name in ARTIFACTS:
        path = directory / name
        if (
            not path.is_file()
            or not path.stat().st_size
            or hashes.get(name) != digest(path.read_bytes())
        ):
            raise ValueError(f"missing or changed artifact: {name}")
    previous = read(directory / "runner-context.json")
    if previous["sha256"] != current["sha256"]:
        raise ValueError("runtime/configuration/environment/native fingerprint changed")
    manifest = read(directory / "manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("index") != unit
        or manifest.get("total") != UNITS
    ):
        raise ValueError("logical unit identity changed")
    if manifest.get("collect_only") is not False:
        raise ValueError("collect-only artifacts cannot resume execution")
    completion = read(directory / "completion.json")
    if (
        type(completion.get("exitcode")) is not int
        or completion["exitcode"] != manifest["exitstatus"]
    ):
        raise ValueError("pytest process and manifest completion disagree")
    if (
        manifest["selected_nodeids"] != current["selected_nodeids"]
        or manifest["module_assignments"] != current["module_assignments"]
    ):
        raise ValueError("selected IDs or assignment changed; rerun this unit")
    if manifest["assigned_nodeids"] != assigned(current, unit):
        raise ValueError("assigned IDs changed")
    if len(manifest["executed_nodeids"]) != len(
        set(manifest["executed_nodeids"])
    ) or not set(manifest["executed_nodeids"]) <= set(manifest["assigned_nodeids"]):
        raise ValueError("executed IDs are invalid")
    if (
        manifest["duration_profile"]["sha256"]
        != current["configuration"]["profile_sha256"]
    ):
        raise ValueError("duration profile changed")
    snapshot = manifest["source_snapshot"]
    if snapshot.get("final") != {
        domain: snapshot.get(domain) for domain in ("modules", "runtime", "config")
    }:
        raise ValueError("final source attestation differs from the starting snapshot")
    for domain in ("runtime", "config"):
        if snapshot[domain]["sha256"] != current["source_snapshot"][domain]["sha256"]:
            raise ValueError(f"{domain} fingerprint changed")
    if snapshot.get("unchanged") is not True:
        raise ValueError("source changed during unit execution")
    junit = ET.parse(directory / "junit.xml")
    if manifest["exitstatus"] == 0 and any(
        row.tag in ("failure", "error") for row in junit.iter()
    ):
        raise ValueError("successful unit contains JUnit failure/error")
    evidence = _read_artifact(directory)
    try:
        data = evidence["data"]
        if not any(data.lines(name) for name in data.measured_files()):
            raise ValueError("raw relative-path branch coverage is missing")
        if manifest["exitstatus"] == 0:
            for nodeid in manifest["assigned_nodeids"]:
                record = manifest["test_outcomes"][nodeid]
                _validate_outcome(record)
                _case(evidence, nodeid, record["outcome"])
                _duration(manifest, nodeid)
    finally:
        evidence["data"].close()
    return manifest


def recovery_modules(repo, manifest):
    modules = module_hashes(repo, manifest["assigned_nodeids"])
    changed = {
        name
        for name, value in modules.items()
        if manifest["source_snapshot"]["modules"].get(name) != value
    }
    executed = set(manifest["executed_nodeids"])
    outcomes = manifest.get("test_outcomes", {})
    for nodeid in manifest["assigned_nodeids"]:
        outcome = outcomes.get(nodeid, {})
        try:
            _validate_outcome(outcome)
            complete = nodeid in executed
        except ValueError:
            complete = False
        if not complete:
            changed.add(nodeid.split("::", 1)[0])
    return sorted(changed)


def publish(staged, output, workspace):
    if output.exists():
        output.rename(workspace / "superseded")
    staged.rename(output)


def run_unit(repo, output_root, state, current, unit):
    output = output_root / str(unit)
    workspace = state / f"unit-{unit}-{time.time_ns()}"
    workspace.mkdir(parents=True)
    reason = "no previous unit"
    if output.exists():
        try:
            manifest = inspect_artifacts(output, unit, current)
        except Exception as exc:
            reason = str(exc)
        else:
            modules = recovery_modules(repo, manifest)
            if (
                not modules
                and manifest["exitstatus"] == 0
                and set(manifest["executed_nodeids"])
                == set(manifest["assigned_nodeids"])
            ):
                return {"unit": unit, "status": "resumed", "output": str(output)}
            if (
                modules
                and manifest["source_snapshot"]["config"].get("selective_recovery_safe")
                is True
            ):
                # Keep source artifacts immutable and preserve failed-attempt evidence.
                original = workspace / "original"
                shutil.copytree(output, original)
                replacement = workspace / "replacement"
                status = execute(
                    repo,
                    replacement,
                    command(repo, replacement, 1, total=1, modules=modules),
                    current,
                )
                if status:
                    return {
                        "unit": unit,
                        "status": "failed",
                        "reason": "module retry failed",
                        "modules": modules,
                        "retry": str(replacement),
                    }
                try:
                    recovered = workspace / "recovered"
                    recovery_command = recover_unit(
                        repo,
                        original,
                        replacement,
                        recovered,
                        workspace / "recovery.log",
                    )
                    write(
                        recovered / "invocation.json",
                        {
                            "action": "complete-module recovery",
                            "command": recovery_command,
                        },
                    )
                    write(
                        recovered / "runner-context.json",
                        {key: current[key] for key in ("sha256", "configuration")},
                    )
                    write(
                        recovered / "completion.json",
                        {
                            "exitcode": 0,
                            "finished": stamp(),
                            "action": "module recovery",
                            "original": str(original),
                            "replacement": str(replacement),
                        },
                    )
                    if not (recovered / "pytest.log").exists():
                        (recovered / "pytest.log").write_text(
                            f"Recovered from {original} and {replacement}; see recovery.json.\n"
                        )
                    seal(recovered)
                    result = inspect_artifacts(recovered, unit, current)
                    if result["exitstatus"] != 0 or recovery_modules(repo, result):
                        raise ValueError("recovered unit is not a complete success")
                    publish(recovered, output, workspace)
                    return {
                        "unit": unit,
                        "status": "recovered",
                        "modules": modules,
                        "output": str(output),
                    }
                except Exception as exc:
                    return {
                        "unit": unit,
                        "status": "failed",
                        "reason": f"recovery rejected: {exc}",
                        "retry": str(replacement),
                    }
            reason = "unsuccessful evidence has no safely identifiable failed module"
    staged = workspace / "fresh"
    status = execute(repo, staged, command(repo, staged, unit), current)
    try:
        result = inspect_artifacts(staged, unit, current)
        if status or result["exitstatus"] != 0 or recovery_modules(repo, result):
            raise ValueError(f"pytest/unit failed with exit {status}")
        passed = True
    except Exception as exc:
        passed, reason = False, f"{reason}; {exc}"
    publish(staged, output, workspace)
    return {
        "unit": unit,
        "status": "passed" if passed else "failed",
        "reason": reason,
        "output": str(output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("ci-results"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--job-index", type=int, default=os.environ.get("CI_NODE_INDEX")
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if (
        args.job_index is not None
        and int(os.environ.get("CI_NODE_TOTAL", JOBS)) != JOBS
    ):
        parser.error(f"GitLab ownership requires exactly {JOBS} jobs")
    try:
        units = units_for_job(args.job_index)
        repo = repository()
        output = args.output_dir.resolve()
        validate_output(repo, output)
        output.mkdir(parents=True, exist_ok=True)
        state = (
            output.with_name(output.name + "-runner")
            / f"job-{args.job_index or 'local'}-{time.time_ns()}"
        )
        state.mkdir(parents=True)
        for cached in output.iterdir():
            if not cached.is_dir() or cached.name not in {
                str(unit) for unit in range(1, UNITS + 1)
            }:
                raise ValueError(
                    f"unexpected file/directory in runner-owned output: {cached}"
                )
            if int(cached.name) not in units:
                quarantine = state / "unowned-cache"
                quarantine.mkdir(exist_ok=True)
                cached.rename(quarantine / cached.name)
        collection = state / "collection"
        status = execute(
            repo,
            collection,
            command(repo, collection, 1, collect=True),
            {"action": "collect"},
        )
        if status:
            raise ValueError(
                f"Full-suite collection failed; see {collection / 'pytest.log'}"
            )
        current = context(repo, read(collection / "manifest.json"))
        records = []
        with ThreadPoolExecutor(max_workers=args.workers) as workers:
            pending = {
                workers.submit(run_unit, repo, output, state, current, unit): unit
                for unit in units
            }
            for future in as_completed(pending):
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "unit": pending[future],
                        "status": "failed",
                        "reason": str(exc),
                    }
                records.append(result)
                print(json.dumps(result), flush=True)
                write(
                    state / "results.json",
                    {"units": units, "workers": args.workers, "results": records},
                )
        return int(any(row["status"] == "failed" for row in records))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Shard runner rejected the invocation: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
