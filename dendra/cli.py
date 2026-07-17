"""Command-line entry points for Dendra deployment diagnostics."""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from .diagnostics import collect_doctor_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dendra")
    subparsers = parser.add_subparsers(dest="command")
    doctor = subparsers.add_parser(
        "doctor", help="inspect CPU, CUDA, and optional native-extension readiness"
    )
    doctor.add_argument(
        "--require-cuda",
        action="store_true",
        help="treat unavailable CUDA/PyTorch CUDA support as an error",
    )
    doctor.add_argument(
        "--probe-native-bitpack",
        action="store_true",
        help="opt into JIT build/load and tiny native bitpack kernel launches",
    )
    doctor.add_argument(
        "--json",
        action="store_true",
        help="emit the structured report as JSON",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command != "doctor":
        parser.print_help()
        return 2

    report = collect_doctor_report(
        require_cuda=args.require_cuda,
        probe_native_bitpack=args.probe_native_bitpack,
    )
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        print(report.to_text())
    return 0 if report.ok else 1
