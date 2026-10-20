#!/usr/bin/env python3
"""Validate complete CI shards and enforce coverage on their combined data."""

from __future__ import annotations

import argparse
import json
import math
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


def validate_shards(input_dir: Path, shards: int) -> list[Path]:
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
    args = parser.parse_args(argv)
    if not math.isfinite(args.fail_under) or not 0 <= args.fail_under <= 100:
        parser.error("--fail-under must be between 0 and 100")

    try:
        data_paths = validate_shards(args.input_dir, args.shards)
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
