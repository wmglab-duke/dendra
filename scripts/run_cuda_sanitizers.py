#!/usr/bin/env python3
"""Run NVIDIA Compute Sanitizer over Dendra's native CUDA bitpack kernels."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")
KERNELS = (
    "pack_source_spikes_words_kernel",
    "build_delivery_kernel",
    "build_delivery_uniform_kernel",
)
PREFLIGHT = """
from dendra.models.networks import netcon_bitpack_ops as ops

available = ops.is_available()
print(f"Native CUDA bitpack extension available: {available}")
print(ops.build_info())
if not available:
    print(f"Native CUDA bitpack extension error: {ops.last_error()!r}")
raise SystemExit(0 if available else 1)
"""
CUDA_RUNTIME_INFO = """
import json
import torch
from torch.utils.cpp_extension import CUDA_HOME

print(json.dumps({"torch_cuda": torch.version.cuda, "cuda_home": CUDA_HOME}))
"""


def _supports_cpp20(executable: str) -> bool:
    try:
        result = subprocess.run(
            (
                executable,
                "-std=c++20",
                "-x",
                "c++",
                "-fsyntax-only",
                "-",
            ),
            input="int main() { return 0; }\n",
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def _configure_compiler(environment: dict[str, str]) -> None:
    explicit = environment.get("CXX")
    if explicit:
        executable = shutil.which(explicit) or explicit
        if not _supports_cpp20(executable):
            raise RuntimeError(
                f"CXX={explicit!r} cannot compile a C++20 translation unit"
            )
        return

    candidates = (
        "c++",
        "g++",
        *(f"g++-{version}" for version in range(14, 9, -1)),
    )
    for candidate in candidates:
        executable = shutil.which(candidate)
        if executable is None or not _supports_cpp20(executable):
            continue
        environment["CXX"] = executable
        name = Path(executable).name
        if name.startswith("g++"):
            gcc = Path(executable).with_name(name.replace("g++", "gcc", 1))
            if gcc.is_file():
                environment.setdefault("CC", str(gcc))
        print(f"Using C++20 compiler: {executable}", flush=True)
        return
    raise RuntimeError(
        "No C++20-capable host compiler was found. Set CC and CXX to a "
        "compiler pair supported by the installed CUDA toolkit."
    )


def _sanitizer_binary() -> str:
    executable = shutil.which("compute-sanitizer")
    if executable is not None:
        return executable
    candidate = Path("/usr/local/cuda/bin/compute-sanitizer")
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(
        "compute-sanitizer was not found on PATH or under /usr/local/cuda/bin"
    )


def _cuda_runtime_info(
    environment: dict[str, str], root: Path
) -> tuple[str | None, Path | None]:
    result = subprocess.run(
        (sys.executable, "-c", CUDA_RUNTIME_INFO),
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return None, None
    try:
        info = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        return None, None
    torch_cuda = info.get("torch_cuda")
    cuda_home = info.get("cuda_home")
    return (
        str(torch_cuda) if torch_cuda else None,
        Path(cuda_home) if cuda_home else None,
    )


def _cuda_toolkit_version(
    cuda_home: Path | None, environment: dict[str, str]
) -> str | None:
    candidates = []
    if cuda_home is not None:
        candidates.append(cuda_home / "bin" / "nvcc")
    on_path = shutil.which("nvcc", path=environment.get("PATH"))
    if on_path:
        candidates.append(Path(on_path))

    for executable in candidates:
        try:
            result = subprocess.run(
                (str(executable), "--version"),
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            continue
        match = re.search(r"\brelease\s+(\d+\.\d+)", result.stdout)
        if result.returncode == 0 and match:
            return match.group(1)
    return None


def _major(version: str | None) -> int | None:
    if version is None:
        return None
    match = re.match(r"(\d+)", version)
    return int(match.group(1)) if match else None


def _command(tool: str, sanitizer: str, *, launch_timeout: int) -> list[str]:
    command = [
        sanitizer,
        "--tool",
        tool,
        "--launch-timeout",
        str(launch_timeout),
        "--kill",
        "--error-exitcode",
        "99",
        "--target-processes",
        "application-only",
    ]
    for kernel in KERNELS:
        command.extend(("--kernel-name", f"kns={kernel}"))
    command.extend(
        (
            sys.executable,
            "-m",
            "pytest",
            "tests/networks/test_cuda_bitpack_kernels.py",
            "tests/networks/test_cuda_bitpack_stress.py",
            "-k",
            "native",
            "-W",
            "error",
            "-q",
            "-s",
            "-rs",
        )
    )
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tool",
        choices=("all", *TOOLS),
        default="all",
        help="sanitizer tool to run (default: all)",
    )
    parser.add_argument(
        "--launch-timeout",
        type=int,
        default=120,
        help="seconds allowed for Compute Sanitizer to attach (default: 120)",
    )
    args = parser.parse_args(argv)
    if args.launch_timeout <= 0:
        parser.error("--launch-timeout must be a positive integer")

    try:
        sanitizer = _sanitizer_binary()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    environment = os.environ.copy()
    # Windows TMP/TEMP paths inherited by WSL prevent Compute Sanitizer from
    # launching even /bin/true. Keep its injection files and local socket on
    # the Linux filesystem.
    environment.update(
        {
            "TMP": "/tmp",
            "TEMP": "/tmp",
            "TMPDIR": "/tmp",
            "NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE": "uds",
            "DENDRA_NETCON_BITPACK_LINEINFO": "1",
            "DENDRA_INDUCTOR_CLEANUP": "0",
        }
    )
    python_bin = str(Path(sys.executable).resolve().parent)
    environment["PATH"] = os.pathsep.join((python_bin, environment.get("PATH", "")))
    try:
        _configure_compiler(environment)
    except RuntimeError as exc:
        print(f"Native CUDA compiler configuration failed: {exc}", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parents[1]
    print(
        "Preparing native CUDA bitpack extension with line information...",
        flush=True,
    )
    preflight = subprocess.run(
        (sys.executable, "-c", PREFLIGHT),
        cwd=root,
        env=environment,
        check=False,
    )
    if preflight.returncode:
        print(
            "Native CUDA bitpack extension preflight failed. Activate the "
            "supported Dendra build environment and resolve the reported "
            "compiler or CUDA error before running Compute Sanitizer.",
            file=sys.stderr,
        )
        return preflight.returncode

    selected = TOOLS if args.tool == "all" else (args.tool,)
    torch_cuda, cuda_home = _cuda_runtime_info(environment, root)
    toolkit_cuda = _cuda_toolkit_version(cuda_home, environment)
    if (
        "initcheck" in selected
        and _major(torch_cuda) is not None
        and _major(toolkit_cuda) is not None
        and _major(torch_cuda) != _major(toolkit_cuda)
    ):
        message = (
            "CUDA initcheck is not reliable for this build: PyTorch uses "
            f"CUDA {torch_cuda}, while the native extension is compiled with "
            f"the CUDA {toolkit_cuda} toolkit. Initcheck tracks CUDA memset "
            "and memcpy initialization, and mixed runtime majors can produce "
            "false uninitialized-memory reports. Use a PyTorch build and CUDA "
            "toolkit with matching major versions for trustworthy initcheck "
            "coverage."
        )
        if args.tool == "initcheck":
            print(message, file=sys.stderr)
            return 2
        print(f"Skipping initcheck. {message}", flush=True)
        selected = tuple(tool for tool in selected if tool != "initcheck")

    for tool in selected:
        print(f"Running CUDA {tool}...", flush=True)
        result = subprocess.run(
            _command(
                tool,
                sanitizer,
                launch_timeout=args.launch_timeout,
            ),
            cwd=root,
            env=environment,
            check=False,
        )
        if result.returncode:
            print(
                f"CUDA {tool} failed with exit code {result.returncode}. "
                "On WSL/WDDM, verify that the Windows driver supports and "
                "permits GPU debugging as described by NVIDIA's Compute "
                "Sanitizer documentation.",
                file=sys.stderr,
            )
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
