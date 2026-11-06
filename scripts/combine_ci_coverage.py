#!/usr/bin/env python3
"""Validate complete CI shards and enforce coverage on their combined data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import coverage
from coverage.exceptions import CoverageException

if __package__:
    from . import check_coverage_floors
else:
    import check_coverage_floors


def _nodeids(manifest: dict, key: str, path: Path) -> list[str]:
    values = manifest.get(key)
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value for value in values)
        or len(values) != len(set(values))
    ):
        raise ValueError(f"{path}: {key} must contain unique, nonempty test IDs")
    return values


def _digest(record: dict) -> str:
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _source_hash(root: Path, name: str) -> str:
    path = root / name
    if Path(name).is_absolute() or not path.resolve().is_relative_to(root):
        raise ValueError(f"invalid partition source path: {name!r}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_partition_source(manifest: dict, path: Path, root: Path) -> dict:
    snapshot = manifest.get("source_snapshot", {})
    if not isinstance(snapshot, dict):
        raise ValueError(f"{path}: source attestation is missing")
    initial = {key: snapshot.get(key) for key in ("modules", "runtime", "config")}
    if snapshot.get("unchanged") is not True or snapshot.get("final") != initial:
        raise ValueError(f"{path}: sources changed or source attestation is missing")
    modules = {node.split("::", 1)[0] for node in manifest["assigned_nodeids"]}
    if not isinstance(initial["modules"], dict) or set(initial["modules"]) != modules:
        raise ValueError(f"{path}: assigned module source inventory disagrees")
    for name, expected in initial["modules"].items():
        if _source_hash(root, name) != expected:
            raise ValueError(f"{path}: current module source differs: {name}")
    for domain in ("runtime", "config"):
        record = initial[domain]
        if not isinstance(record, dict) or record.get("sha256") != _digest(
            {key: value for key, value in record.items() if key != "sha256"}
        ):
            raise ValueError(f"{path}: invalid {domain} source fingerprint")
        if not isinstance(record.get("files"), dict):
            raise ValueError(f"{path}: missing {domain} source inventory")
    environment = initial["runtime"].get("environment", {})
    packages = environment.get("packages") if isinstance(environment, dict) else None
    if not isinstance(packages, list) or any(
        not isinstance(pair, list)
        or len(pair) != 2
        or any(not isinstance(value, str) for value in pair)
        for pair in packages
    ):
        raise ValueError(f"{path}: runtime package inventory is missing or malformed")
    try:
        tracked = (
            subprocess.run(
                ["git", "-C", str(root), "ls-files", "--cached", "-z"],
                check=True,
                capture_output=True,
            )
            .stdout.decode()
            .split("\0")
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            "partition source validation requires a Git checkout"
        ) from error
    selected_modules = {node.split("::", 1)[0] for node in manifest["selected_nodeids"]}
    config_files = initial["config"]["files"]
    for name in filter(None, tracked):
        # A selected leaf is checked by its owning shard. Every other tracked
        # source must match the current checkout. Generated worker constraints
        # and installed dependencies are compared between shards below.
        if name in selected_modules and name not in config_files:
            continue
        if _source_hash(root, name) != config_files.get(name):
            raise ValueError(f"{path}: current project source differs: {name}")
    if config_files.get("@ci_shard_plugin") != _source_hash(
        root, "scripts/ci_shard.py"
    ):
        raise ValueError(f"{path}: shard plugin differs from current source")
    for name, expected in initial["runtime"]["files"].items():
        if not name.startswith("@") and _source_hash(root, name) != expected:
            raise ValueError(f"{path}: current runtime source differs: {name}")
    return {key: initial[key] for key in ("runtime", "config")}


def validate_shards(
    input_dir: Path,
    shards: int,
    *,
    require_coverage: bool = True,
    source_root: Path | None = None,
) -> list[Path]:
    """Require successful, disjoint execution of one agreed full collection."""
    if shards < 1:
        raise ValueError("the shard count must be positive")
    expected_dirs = {str(index) for index in range(1, shards + 1)}
    if not input_dir.is_dir():
        raise ValueError(f"shard artifact directory is missing: {input_dir}")
    actual_dirs = {path.name for path in input_dir.iterdir() if path.is_dir()}
    unexpected_files = [path.name for path in input_dir.iterdir() if not path.is_dir()]
    if actual_dirs != expected_dirs or unexpected_files:
        raise ValueError(
            f"expected exactly shard directories {sorted(expected_dirs)} in {input_dir}; "
            f"found directories {sorted(actual_dirs)}, files {sorted(unexpected_files)}"
        )

    selected: list[str] | None = None
    assigned: set[str] = set()
    data_paths: list[Path] = []
    sources: dict | None = None
    assignments: dict | None = None
    for index in range(1, shards + 1):
        directory = input_dir / str(index)
        manifest_path = directory / "manifest.json"
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if not isinstance(manifest, dict):
            raise ValueError(f"{manifest_path}: expected a JSON object")
        for key, expected in (
            ("schema_version", 1),
            ("index", index),
            ("total", shards),
        ):
            if type(manifest.get(key)) is not int or manifest[key] != expected:
                raise ValueError(f"{manifest_path}: {key} must equal {expected}")
        if type(manifest.get("exitstatus")) is not int or manifest["exitstatus"] != 0:
            raise ValueError(f"{manifest_path}: shard did not finish successfully")
        if manifest.get("collect_only") is not False:
            raise ValueError(f"{manifest_path}: collect-only data cannot satisfy CI")

        shard_selected = _nodeids(manifest, "selected_nodeids", manifest_path)
        shard_assigned = _nodeids(manifest, "assigned_nodeids", manifest_path)
        executed = _nodeids(manifest, "executed_nodeids", manifest_path)
        if selected is None:
            selected = shard_selected
        elif shard_selected != selected:
            raise ValueError(
                f"{manifest_path}: full selected test collection disagrees"
            )
        shard_set = set(shard_assigned)
        if not shard_set <= set(selected):
            raise ValueError(f"{manifest_path}: assignment contains unselected tests")
        if assigned & shard_set:
            raise ValueError(f"{manifest_path}: tests are assigned to multiple shards")
        if set(executed) != shard_set:
            raise ValueError(
                f"{manifest_path}: executed tests differ from assigned tests"
            )
        assigned.update(shard_set)

        if not require_coverage:
            current_sources = _validate_partition_source(
                manifest, manifest_path, (source_root or Path.cwd()).resolve()
            )
            current_assignments = manifest.get("module_assignments")
            if not isinstance(current_assignments, dict) or any(
                current_assignments.get(node.split("::", 1)[0]) != index
                for node in shard_assigned
            ):
                raise ValueError(f"{manifest_path}: whole-module assignment disagrees")
            if sources is None:
                sources, assignments = current_sources, current_assignments
            elif current_sources != sources or current_assignments != assignments:
                raise ValueError(
                    f"{manifest_path}: runtime, configuration or module assignments disagree"
                )
            continue

        data_path = directory / ".coverage"
        if not data_path.is_file() or data_path.stat().st_size == 0:
            raise ValueError(f"{data_path}: raw coverage data is missing or empty")
        data = coverage.CoverageData(basename=str(data_path.resolve()))
        data.read()
        if not data.has_arcs():
            raise ValueError(f"{data_path}: branch coverage is required")
        measured = data.measured_files()
        if not measured or not any(data.lines(filename) for filename in measured):
            raise ValueError(f"{data_path}: no executed source lines were measured")
        if any(Path(filename).is_absolute() for filename in measured):
            raise ValueError(
                f"{data_path}: coverage paths must be relative to the checkout"
            )
        data_paths.append(data_path.resolve())

    if assigned != set(selected or []):
        missing = sorted(set(selected or []) - assigned)
        raise ValueError(
            f"shard assignments do not cover the full collection: {missing}"
        )
    return data_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", required=True, type=int)
    parser.add_argument("--input-dir", default=Path("ci-results"), type=Path)
    parser.add_argument("--coverage-config", default=Path(".coveragerc.cpu"), type=Path)
    parser.add_argument(
        "--floors-config", default=Path("coverage-floors.toml"), type=Path
    )
    parser.add_argument("--fail-under", default=84.0, type=float)
    parser.add_argument(
        "--partition-only",
        action="store_true",
        help="Validate executed compatibility shards without changing coverage gates.",
    )
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--torch-spec", help="Required Torch version range for a partition."
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.fail_under) or not 0 <= args.fail_under <= 100:
        parser.error("--fail-under must be between 0 and 100")

    try:
        data_paths = validate_shards(
            args.input_dir,
            args.shards,
            require_coverage=not args.partition_only,
            source_root=args.source_root,
        )
        if args.partition_only:
            manifest = json.loads((args.input_dir / "1" / "manifest.json").read_text())
            packages = manifest["source_snapshot"]["runtime"]["environment"]["packages"]
            torch_versions = [value for name, value in packages if name == "torch"]
            if len(torch_versions) != 1:
                raise ValueError(
                    "partition must record exactly one PyTorch distribution"
                )
            if args.torch_spec:
                from packaging.requirements import Requirement

                required = Requirement(args.torch_spec)
                if (
                    required.name != "torch"
                    or torch_versions[0] not in required.specifier
                ):
                    raise ValueError(
                        f"partition Torch {torch_versions[0]} does not satisfy {args.torch_spec}"
                    )
            print(
                f"Verified {len(manifest['selected_nodeids'])} tests across "
                f"{args.shards} shards with Torch {torch_versions[0]}."
            )
            return 0
        if args.torch_spec:
            raise ValueError("--torch-spec requires --partition-only")
        combined = coverage.Coverage(
            config_file=str(args.coverage_config), data_file=".coverage"
        )
        # Never let a stale combined file fill a gap in the current shard data.
        combined.erase()
        combined.combine(
            data_paths=[str(path) for path in data_paths], strict=True, keep=True
        )
        combined.save()
        # Publish useful diagnostics even when either coverage gate fails.
        combined.json_report(outfile="coverage.json")
        combined.xml_report(outfile="coverage.xml")
        combined.html_report(directory="htmlcov")
        total = combined.report(show_missing=True)
    except (OSError, ValueError, CoverageException) as exc:
        print(f"CI coverage aggregation error: {exc}", file=sys.stderr)
        return 2

    global_failed = total < args.fail_under
    if global_failed:
        print(
            f"Combined coverage {total:.2f}% is below the {args.fail_under:g}% minimum",
            file=sys.stderr,
        )
    floor_status = check_coverage_floors.main(
        ["coverage.json", "--config", str(args.floors_config)]
    )
    return max(int(global_failed), floor_status)


if __name__ == "__main__":
    raise SystemExit(main())
