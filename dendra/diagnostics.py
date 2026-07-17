"""Deployment diagnostics for Dendra CPU, CUDA, and native extension paths."""

from __future__ import annotations

import os
import platform
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, TextIO, Tuple

import torch

from .helpers import current_native_extension_policy
from .models.networks import netcon_bitpack_ops

DoctorStatus = Literal["ok", "warning", "error", "info"]


@dataclass(frozen=True)
class DoctorCheck:
    """One independently actionable environment diagnostic."""

    name: str
    status: DoctorStatus
    message: str
    details: Optional[Dict[str, object]] = None

    def as_dict(self) -> Dict[str, object]:
        result: Dict[str, object] = {
            "name": self.name,
            "status": self.status,
            "message": self.message,
        }
        if self.details:
            result["details"] = dict(self.details)
        return result


@dataclass(frozen=True)
class DoctorReport:
    """Structured result returned by :func:`doctor`."""

    checks: Tuple[DoctorCheck, ...]
    require_cuda: bool
    probe_native_bitpack: bool

    @property
    def ok(self) -> bool:
        return all(check.status != "error" for check in self.checks)

    def as_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "require_cuda": self.require_cuda,
            "probe_native_bitpack": self.probe_native_bitpack,
            "checks": [check.as_dict() for check in self.checks],
        }

    def to_text(self) -> str:
        mode = (
            "native probe requested"
            if self.probe_native_bitpack
            else "inspection only; no native build attempted"
        )
        lines = [f"Dendra doctor ({mode})"]
        for check in self.checks:
            lines.append(f"[{check.status.upper():7}] {check.name}: {check.message}")
        lines.append(f"Result: {'PASS' if self.ok else 'FAIL'}")
        return "\n".join(lines)


def _dendra_version() -> str:
    root_package = sys.modules.get(__package__.split(".", 1)[0])
    return str(getattr(root_package, "__version__", "unknown"))


def _cuda_major(value: object) -> Optional[int]:
    if value is None:
        return None
    match = re.search(r"(?:release\s+)?(\d+)\.", str(value), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.expanduser()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _probe_native_bitpack() -> Dict[str, object]:
    """Build/load and exercise every native bitpack entry point."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not netcon_bitpack_ops.is_available():
        cause = netcon_bitpack_ops.last_error()
        raise RuntimeError(
            f"native NetCon bitpack extension is unavailable: {cause!r}"
        ) from cause

    device = torch.device("cuda", torch.cuda.current_device())
    spikes = torch.tensor([True], device=device)
    history = torch.zeros((2, 1), dtype=torch.int64, device=device)
    pack_step = torch.tensor([0], dtype=torch.int64, device=device)
    netcon_bitpack_ops.pack_source_spikes(spikes, history, pack_step)

    delivery_step = torch.tensor([1], dtype=torch.int64, device=device)
    delay_steps = torch.tensor([1], dtype=torch.int64, device=device)
    word_idx = torch.tensor([0], dtype=torch.int64, device=device)
    bit_mask = torch.tensor([1], dtype=torch.int64, device=device)
    post_idx = torch.tensor([0], dtype=torch.int64, device=device)
    weight = torch.tensor([2.0], dtype=torch.float32, device=device)
    delivery = torch.zeros(1, dtype=torch.float32, device=device)
    uniform_delivery = torch.zeros_like(delivery)
    netcon_bitpack_ops.build_delivery(
        history,
        delivery_step,
        delay_steps,
        word_idx,
        bit_mask,
        post_idx,
        weight,
        delivery,
    )
    netcon_bitpack_ops.build_delivery_uniform(
        history,
        delivery_step,
        1,
        word_idx,
        bit_mask,
        post_idx,
        weight,
        uniform_delivery,
    )
    torch.cuda.synchronize(device)
    if int(history[0, 0].item()) != 1:
        raise RuntimeError("native source-spike packing smoke test returned wrong bits")
    if float(delivery.item()) != 2.0 or float(uniform_delivery.item()) != 2.0:
        raise RuntimeError("native delivery smoke test returned the wrong value")
    return netcon_bitpack_ops.build_info()


def collect_doctor_report(
    *, require_cuda: bool = False, probe_native_bitpack: bool = False
) -> DoctorReport:
    """Collect deployment diagnostics without building unless a probe is requested."""
    checks = []

    def add(
        name: str,
        status: DoctorStatus,
        message: str,
        details: Optional[Dict[str, object]] = None,
    ) -> None:
        checks.append(DoctorCheck(name, status, message, details))

    add(
        "runtime",
        "ok",
        f"Dendra {_dendra_version()}, Python {platform.python_version()}, "
        f"PyTorch {torch.__version__} on {platform.system()} {platform.machine()}",
    )
    try:
        policy = current_native_extension_policy()
    except ValueError as exc:
        add("native extension policy", "error", str(exc))
    else:
        add(
            "native extension policy",
            "ok",
            f"{policy!r} is active",
        )

    cuda_required = bool(require_cuda or probe_native_bitpack)
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        devices = []
        try:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "index": index,
                        "name": properties.name,
                        "capability": ".".join(
                            map(str, torch.cuda.get_device_capability(index))
                        ),
                        "total_memory": properties.total_memory,
                    }
                )
            summary = ", ".join(
                f"cuda:{item['index']} {item['name']} (sm_{str(item['capability']).replace('.', '')})"
                for item in devices
            )
        except Exception as exc:
            add(
                "CUDA devices",
                "error",
                f"CUDA initialized but device query failed: {exc!r}",
            )
        else:
            add(
                "CUDA devices",
                "ok",
                summary or "CUDA is available",
                {"devices": devices},
            )
    else:
        add(
            "CUDA devices",
            "error" if cuda_required else "warning",
            "CUDA is not available to PyTorch",
        )

    torch_cuda = torch.version.cuda
    add(
        "PyTorch CUDA runtime",
        "ok" if torch_cuda is not None else ("error" if cuda_required else "info"),
        str(torch_cuda) if torch_cuda is not None else "CPU-only PyTorch build",
    )

    try:
        plan = netcon_bitpack_ops.planned_build_info()
    except Exception as exc:
        plan = None
        add(
            "native build plan",
            "error" if probe_native_bitpack else "warning",
            f"could not resolve the non-building plan: {exc!r}",
        )
    else:
        toolchain = dict(plan["toolchain"])
        nvcc_path = toolchain.get("nvcc_path")
        nvcc_version = toolchain.get("nvcc_version")
        cxx_path = toolchain.get("cxx_path")
        cxx_version = toolchain.get("cxx_version")
        add(
            "CUDA toolkit",
            "ok" if nvcc_path and nvcc_version else "warning",
            (
                f"{nvcc_path}: {nvcc_version}"
                if nvcc_path and nvcc_version
                else "NVCC was not found; eager CUDA can run, but native JIT builds cannot"
            ),
            {"cuda_home": toolchain.get("cuda_home")},
        )
        runtime_major = _cuda_major(torch_cuda)
        toolkit_major = _cuda_major(nvcc_version)
        if runtime_major is not None and toolkit_major is not None:
            matching = runtime_major == toolkit_major
            add(
                "CUDA version alignment",
                "ok" if matching else "warning",
                (
                    f"PyTorch and NVCC both use CUDA {runtime_major}"
                    if matching
                    else f"PyTorch uses CUDA {runtime_major}, NVCC uses CUDA {toolkit_major}"
                ),
            )
        add(
            "C++ compiler",
            "ok" if cxx_path and cxx_version else "warning",
            (
                f"{cxx_path}: {cxx_version}"
                if cxx_path and cxx_version
                else "no usable C++ compiler was detected for native JIT builds"
            ),
        )
        ninja_path = shutil.which("ninja")
        add(
            "Ninja",
            "ok" if ninja_path else "warning",
            ninja_path or "not found; native JIT builds require Ninja",
        )

        missing_sources = list(plan["missing_sources"])
        add(
            "native sources",
            "error" if missing_sources else "ok",
            (
                f"missing packaged sources: {missing_sources}"
                if missing_sources
                else "NetCon C++/CUDA sources are packaged"
            ),
        )
        cache_dir = Path(str(plan["build_directory"]))
        writable_parent = _nearest_existing_parent(cache_dir)
        writable = writable_parent.is_dir() and os.access(writable_parent, os.W_OK)
        add(
            "extension cache",
            "ok" if writable else "error",
            f"{cache_dir} ({'writable' if writable else 'not writable'})",
            {"existing_parent": str(writable_parent)},
        )
        artifacts = list(plan["cached_artifacts"])
        add(
            "cached native bitpack",
            "ok" if artifacts else "info",
            (
                f"found {len(artifacts)} artifact(s) for the current build identity"
                if artifacts
                else "no artifact found for the current build identity"
            ),
            {"artifacts": artifacts, "arch_list": list(plan["arch_list"])},
        )

    state = netcon_bitpack_ops.build_info()
    if state.get("loaded"):
        add("native bitpack process state", "ok", "extension is already loaded")
    elif state.get("disabled"):
        add(
            "native bitpack process state",
            "warning",
            f"a prior load failed: {state.get('last_error')}",
        )
    else:
        add(
            "native bitpack process state",
            "info",
            "not loaded or tested in this process",
        )

    if probe_native_bitpack:
        try:
            details = _probe_native_bitpack()
        except Exception as exc:
            add("native bitpack probe", "error", repr(exc))
        else:
            add(
                "native bitpack probe",
                "ok",
                "pack, general delivery, and uniform delivery kernels passed",
                details,
            )
    else:
        add(
            "native bitpack probe",
            "info",
            "skipped; pass --probe-native-bitpack to opt into build/load execution",
        )

    return DoctorReport(tuple(checks), bool(require_cuda), bool(probe_native_bitpack))


def doctor(
    *,
    require_cuda: bool = False,
    probe_native_bitpack: bool = False,
    output: bool = True,
    stream: Optional[TextIO] = None,
) -> DoctorReport:
    """Inspect a Dendra deployment and optionally probe native CUDA bitpacking.

    By default this function is read-only with respect to native extensions: it
    does not compile, load, or launch one. Set ``probe_native_bitpack=True`` to
    explicitly opt into a JIT build/load and a tiny end-to-end kernel smoke test.
    """
    report = collect_doctor_report(
        require_cuda=require_cuda,
        probe_native_bitpack=probe_native_bitpack,
    )
    if output:
        print(report.to_text(), file=stream or sys.stdout)
    return report
