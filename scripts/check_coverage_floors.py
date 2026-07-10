#!/usr/bin/env python3
"""Enforce per-module combined line-and-branch coverage floors."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path


def load_floors(config_path: Path) -> dict[str, float]:
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    try:
        raw_floors = config["tool"]["dendra"]["coverage-floors"]
    except KeyError as exc:
        raise ValueError(
            f"{config_path} has no [tool.dendra.coverage-floors] table"
        ) from exc

    floors = {str(module): float(value) for module, value in raw_floors.items()}
    invalid = {
        module: value for module, value in floors.items() if not 0 <= value <= 100
    }
    if invalid:
        raise ValueError(f"coverage floors must be between 0 and 100: {invalid}")
    return floors


def load_module_coverage(coverage_path: Path) -> dict[str, float]:
    with coverage_path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    try:
        files = report["files"]
    except KeyError as exc:
        raise ValueError(f"{coverage_path} is not a coverage.py JSON report") from exc

    return {
        str(module): float(data["summary"]["percent_covered"])
        for module, data in files.items()
    }


def check_floors(
    actual: dict[str, float], floors: dict[str, float]
) -> list[tuple[str, float | None, float]]:
    return [
        (module, actual.get(module), floor)
        for module, floor in sorted(floors.items())
        if actual.get(module, -1.0) < floor
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "coverage_json",
        nargs="?",
        default="coverage.json",
        type=Path,
        help="coverage.py JSON report (default: coverage.json)",
    )
    parser.add_argument(
        "--config",
        default=Path("pyproject.toml"),
        type=Path,
        help="TOML file containing [tool.dendra.coverage-floors]",
    )
    args = parser.parse_args(argv)

    try:
        floors = load_floors(args.config)
        actual = load_module_coverage(args.coverage_json)
    except (OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        print(f"coverage floor configuration error: {exc}", file=sys.stderr)
        return 2

    failures = check_floors(actual, floors)
    for module, floor in sorted(floors.items()):
        value = actual.get(module)
        rendered = "missing" if value is None else f"{value:.2f}%"
        status = "FAIL" if any(row[0] == module for row in failures) else "PASS"
        print(f"{status:4}  {module}: {rendered} (floor {floor:.2f}%)")

    if failures:
        print(f"{len(failures)} critical coverage floor(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
