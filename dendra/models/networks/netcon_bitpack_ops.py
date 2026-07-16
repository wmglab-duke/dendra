"""Lazy C++/CUDA kernels for NetCon bitpacked source-spike history.

``NetCon`` dispatches to this module for its normal accelerated CUDA path. The
separate Triton implementation mirrors the core API for direct experimental and
reference use, but ``NetCon`` does not select it automatically:

    pack_source_spikes(source_spikes, packed_history, current_time_step)
    build_delivery(packed_history, current_time_step, delay_steps,
                   conn_word_idx, conn_bit_mask, post_idx, weight, delivery_out)
    build_delivery_uniform(..., delay_step, conn_word_idx, conn_bit_mask, ...)

The extension is compiled with ``torch.utils.cpp_extension.load`` on first use.
Unlike the vanilla PyTorch JIT-extension defaults, this loader:

* resolves CUDA arch flags dynamically for Dendra's bitpack extension only;
* passes explicit ``-gencode`` flags instead of mutating TORCH_CUDA_ARCH_LIST;
* uses a Dendra-specific persistent build/cache directory; and
* hashes source files, arch flags, Python, PyTorch, and CUDA metadata into the
  extension name/build directory so new kernels can reuse the compiled .so when
  the build inputs are unchanged.

Relevant environment overrides:

    DENDRA_TORCH_EXTENSIONS_DIR
        Root directory for all Dendra torch extensions.  Default:
        ``$XDG_CACHE_HOME/dendra/torch_extensions`` or
        ``~/.cache/dendra/torch_extensions``.

    DENDRA_NETCON_BITPACK_BUILD_DIR
        Exact build directory for this extension.  Overrides the root above.

    DENDRA_NETCON_BITPACK_ARCH_LIST
        Dendra-specific arch list, e.g. ``"8.9"`` or ``"8.0;8.6;9.0"``.
        Special values: ``auto``, ``native``, ``visible``, ``current``.

    DENDRA_CUDA_ARCH_LIST
        Broader Dendra-specific arch-list fallback.

    TORCH_CUDA_ARCH_LIST
        Honored only if the Dendra-specific arch-list variables are unset.

    DENDRA_NETCON_BITPACK_ARCH_MODE
        Used when no arch list is set.  ``visible`` compiles for all visible GPU
        capabilities; ``current`` compiles only for ``torch.cuda.current_device()``.

    DENDRA_NETCON_BITPACK_INCLUDE_PTX
        If true, adds PTX for the highest requested arch.  Disabled by default
        for performance on known deployment GPUs; automatically enabled when a
        visible device capability must be clamped to the maximum CUDA arch that
        the installed PyTorch/NVCC stack appears to support.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .netcon_bitpack_contracts import (
    validate_delivery_structure,
    validate_delivery_uniform_structure,
    validate_pack_structure,
)

_EXTENSION = None
_LAST_ERROR: Optional[BaseException] = None
_DISABLED = False
_LAST_BUILD_INFO: Dict[str, object] = {}

_BITS_PER_WORD = 63
_DEFAULT_BASE_EXTENSION_NAME = "dendra_netcon_bitpack_ops"
_BUILD_KEY_VERSION = "toolchain_cache_v2"

_NAMED_ARCHES = {
    # Keep these aligned with PyTorch's conventional TORCH_CUDA_ARCH_LIST names.
    # Users can still pass explicit values such as "8.9" or "9.0+PTX".
    "Kepler+Tesla": "3.7",
    "Kepler": "3.5+PTX",
    "Maxwell+Tegra": "5.3",
    "Maxwell": "5.0;5.2+PTX",
    "Pascal": "6.0;6.1+PTX",
    "Volta+Tegra": "7.2",
    "Volta": "7.0+PTX",
    "Turing": "7.5+PTX",
    "Ampere+Tegra": "8.7",
    "Ampere": "8.0;8.6+PTX",
    "Ada": "8.9+PTX",
    "Hopper": "9.0+PTX",
    "Blackwell+Tegra": "11.0",
    "Blackwell": "10.0;10.3;12.0;12.1+PTX",
}


def _set_error(exc: BaseException) -> None:
    global _LAST_ERROR
    _LAST_ERROR = exc


def _disable(exc: BaseException) -> None:
    global _DISABLED, _LAST_ERROR
    _DISABLED = True
    _LAST_ERROR = exc


def _truthy_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _sanitize_token(value: object, *, max_len: int = 80) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")
    return text[:max_len] or "x"


def _dendra_cache_root() -> Path:
    explicit = os.environ.get("DENDRA_TORCH_EXTENSIONS_DIR")
    if explicit:
        return Path(explicit).expanduser()

    dendra_cache = os.environ.get("DENDRA_CACHE_DIR")
    if dendra_cache:
        return Path(dendra_cache).expanduser() / "torch_extensions"

    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "dendra" / "torch_extensions"

    return Path.home() / ".cache" / "dendra" / "torch_extensions"


def _parse_arch_token(token: str) -> str:
    """Normalize an arch token to PyTorch-style ``M.m`` or ``M.m+PTX``."""
    token = token.strip()
    if not token:
        raise ValueError("empty CUDA arch token")

    ptx = token.upper().endswith("+PTX")
    if ptx:
        token = token[:-4]

    token = token.strip().lower()
    token = token.removeprefix("sm_").removeprefix("compute_")

    if "." in token:
        major, minor = token.split(".", 1)
        if not major.isdigit() or not re.fullmatch(r"\d+[a-z]?", minor):
            raise ValueError(f"invalid CUDA arch token: {token!r}")
        norm = f"{int(major)}.{minor}"
    else:
        # Accept forms such as 86, 90a, 120, 121a.
        m = re.fullmatch(r"(\d+)([a-z]?)", token)
        if m is None:
            raise ValueError(f"invalid CUDA arch token: {token!r}")
        digits, suffix = m.groups()
        if len(digits) < 2:
            raise ValueError(f"invalid CUDA arch token: {token!r}")
        major = digits[:-1]
        minor = digits[-1] + suffix
        norm = f"{int(major)}.{minor}"

    return norm + ("+PTX" if ptx else "")


def _expand_named_arches(raw: str) -> str:
    for name, archs in sorted(
        _NAMED_ARCHES.items(), key=lambda kv: len(kv[0]), reverse=True
    ):
        raw = re.sub(re.escape(name), archs, raw, flags=re.IGNORECASE)
    return raw


def _split_arch_list(raw: str) -> List[str]:
    raw = _expand_named_arches(raw.strip())
    if raw.lower() in ("", "auto", "native", "visible", "current"):
        return []

    # Match the separators PyTorch users usually employ: spaces, semicolons,
    # or commas.  Keep +PTX attached to its base token.
    raw = raw.replace(",", ";").replace(" ", ";")
    tokens = [_parse_arch_token(tok) for tok in raw.split(";") if tok.strip()]
    if not tokens:
        return []

    # Deduplicate by base arch while preserving whether any token requested PTX.
    by_base: Dict[str, bool] = {}
    for token in tokens:
        base = token[:-4] if token.endswith("+PTX") else token
        by_base[base] = by_base.get(base, False) or token.endswith("+PTX")

    return [
        base + ("+PTX" if ptx else "")
        for base, ptx in sorted(by_base.items(), key=lambda kv: _arch_sort_key(kv[0]))
    ]


def _arch_sort_key(token: str) -> Tuple[int, int, str]:
    base = token[:-4] if token.endswith("+PTX") else token
    major, minor = base.split(".", 1)
    m = re.fullmatch(r"(\d+)([a-z]?)", minor)
    if m is None:
        return (int(major), 0, minor)
    return (int(major), int(m.group(1)), m.group(2))


def _compiled_sm_capabilities() -> List[Tuple[int, int]]:
    """Return CUDA SM capabilities compiled into this PyTorch build."""
    caps = []
    try:
        arch_list = torch.cuda.get_arch_list()
    except Exception:
        return []

    for item in arch_list:
        if not item.startswith("sm_"):
            continue
        try:
            norm = _parse_arch_token(item)
            base = norm[:-4] if norm.endswith("+PTX") else norm
            major, minor = base.split(".", 1)
            m = re.fullmatch(r"(\d+)", minor)
            if m is None:
                # get_device_capability cannot report the architecture suffix
                # variants, so ignore suffix-specific entries here.
                continue
            caps.append((int(major), int(m.group(1))))
        except Exception:
            continue
    return sorted(set(caps))


def _auto_cuda_arch_list(mode: str) -> List[str]:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return []

    mode = mode.strip().lower()
    if mode in ("", "auto", "native", "visible"):
        devices = list(range(torch.cuda.device_count()))
    elif mode == "current":
        devices = [torch.cuda.current_device()]
    else:
        raise ValueError(
            "DENDRA_NETCON_BITPACK_ARCH_MODE must be one of "
            "'visible', 'current', 'auto', or 'native'"
        )

    supported = _compiled_sm_capabilities()
    max_supported = supported[-1] if supported else None
    include_ptx = _truthy_env("DENDRA_NETCON_BITPACK_INCLUDE_PTX", False)

    by_base: Dict[str, bool] = {}
    for dev in devices:
        cap = tuple(int(v) for v in torch.cuda.get_device_capability(dev))
        needs_ptx = False
        if max_supported is not None and cap > max_supported:
            # Match PyTorch's default strategy in spirit: do not ask an older
            # NVCC/toolchain for a newer SASS target it likely cannot build;
            # compile PTX for the highest supported architecture instead.
            cap = max_supported
            needs_ptx = True
        base = f"{cap[0]}.{cap[1]}"
        by_base[base] = by_base.get(base, False) or needs_ptx

    if include_ptx and by_base:
        highest = sorted(by_base, key=_arch_sort_key)[-1]
        by_base[highest] = True

    return [
        base + ("+PTX" if ptx else "")
        for base, ptx in sorted(by_base.items(), key=lambda kv: _arch_sort_key(kv[0]))
    ]


def _resolve_cuda_arch_list() -> List[str]:
    # Dendra-specific arch settings win.  Global TORCH_CUDA_ARCH_LIST is honored
    # only as a fallback so users with existing setups are not surprised.
    raw = (
        os.environ.get("DENDRA_NETCON_BITPACK_ARCH_LIST")
        or os.environ.get("DENDRA_CUDA_ARCH_LIST")
        or os.environ.get("TORCH_CUDA_ARCH_LIST")
    )
    if raw is not None:
        lowered = raw.strip().lower()
        if lowered in ("", "auto", "native", "visible", "current"):
            return _auto_cuda_arch_list(lowered)
        return _split_arch_list(raw)

    return _auto_cuda_arch_list(
        os.environ.get("DENDRA_NETCON_BITPACK_ARCH_MODE", "visible")
    )


def _cuda_gencode_flags(arch_list: Sequence[str]) -> List[str]:
    flags: List[str] = []
    for token in arch_list:
        ptx = token.endswith("+PTX")
        base = token[:-4] if ptx else token
        major, minor = base.split(".", 1)
        num = f"{int(major)}{minor}"
        flags.append(f"-gencode=arch=compute_{num},code=sm_{num}")
        if ptx:
            flags.append(f"-gencode=arch=compute_{num},code=compute_{num}")
    return sorted(set(flags))


def _hash_sources(sources: Sequence[str]) -> str:
    h = hashlib.sha256()
    for source in sources:
        p = Path(source)
        h.update(p.name.encode("utf8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _resolved_cuda_home() -> Optional[Path]:
    explicit = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if explicit:
        return Path(os.path.expandvars(explicit)).expanduser()
    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:
        return None
    return Path(CUDA_HOME).expanduser() if CUDA_HOME else None


def _resolve_command(
    command: Optional[str], fallbacks: Sequence[str]
) -> Tuple[str, Optional[str]]:
    raw = command.strip() if command else ""
    if not raw:
        raw = next((candidate for candidate in fallbacks if candidate), "")
    if not raw:
        return "", None
    try:
        parts = shlex.split(os.path.expandvars(raw))
    except ValueError:
        parts = [raw]
    if not parts:
        return raw, None
    executable = str(Path(parts[0]).expanduser())
    resolved = shutil.which(executable)
    if resolved is None and Path(executable).is_file():
        resolved = executable
    if resolved is not None:
        try:
            resolved = str(Path(resolved).resolve())
        except OSError:
            pass
    return raw, resolved


def _command_version(executable: Optional[str]) -> Optional[str]:
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "--version"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = " ".join(result.stdout.split())
    return output[:2048] or None


def native_toolchain_info() -> Dict[str, object]:
    """Return non-mutating native build-toolchain diagnostics and cache inputs."""
    cuda_home = _resolved_cuda_home()
    nvcc_fallbacks = []
    if cuda_home is not None:
        nvcc_fallbacks.append(str(cuda_home / "bin" / "nvcc"))
    nvcc_fallbacks.append("nvcc")
    nvcc_command, nvcc_path = _resolve_command(
        os.environ.get("CUDACXX"), nvcc_fallbacks
    )
    cxx_command, cxx_path = _resolve_command(os.environ.get("CXX"), ("c++", "g++"))
    return {
        "cuda_home": str(cuda_home) if cuda_home is not None else None,
        "cudacxx": os.environ.get("CUDACXX", ""),
        "nvcc_command": nvcc_command,
        "nvcc_path": nvcc_path,
        "nvcc_version": _command_version(nvcc_path),
        "cc": os.environ.get("CC", ""),
        "cxx_command": cxx_command,
        "cxx_path": cxx_path,
        "cxx_version": _command_version(cxx_path),
        "cuda_host_cxx": os.environ.get("CUDAHOSTCXX", ""),
        "nvcc_prepend_flags": os.environ.get("NVCC_PREPEND_FLAGS", ""),
        "cpath": os.environ.get("CPATH", ""),
        "cplus_include_path": os.environ.get("CPLUS_INCLUDE_PATH", ""),
        "library_path": os.environ.get("LIBRARY_PATH", ""),
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "torch_cxx11_abi": getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None),
        "lineinfo": _truthy_env("DENDRA_NETCON_BITPACK_LINEINFO", False),
        "os_name": os.name,
    }


def _extension_identity(
    sources: Sequence[str],
    arch_list: Sequence[str],
    arch_flags: Sequence[str],
    *,
    toolchain: Optional[Dict[str, object]] = None,
) -> Tuple[str, Path, str]:
    source_hash = _hash_sources(sources)
    if toolchain is None:
        toolchain = native_toolchain_info()
    metadata = "|".join(
        [
            _BUILD_KEY_VERSION,
            f"python={sys.version_info.major}.{sys.version_info.minor}",
            f"torch={torch.__version__}",
            f"torch_cuda={torch.version.cuda}",
            f"arch_list={';'.join(arch_list)}",
            f"arch_flags={';'.join(arch_flags)}",
            f"source_hash={source_hash}",
            "toolchain=" + json.dumps(toolchain, sort_keys=True, separators=(",", ":")),
        ]
    )
    build_key = hashlib.sha256(metadata.encode("utf8")).hexdigest()[:16]

    base_name = _sanitize_token(
        os.environ.get(
            "DENDRA_NETCON_BITPACK_EXTENSION_BASE_NAME", _DEFAULT_BASE_EXTENSION_NAME
        ),
        max_len=48,
    )
    name = f"{base_name}_{build_key}"

    explicit_build_dir = os.environ.get("DENDRA_NETCON_BITPACK_BUILD_DIR")
    if explicit_build_dir:
        build_dir = Path(explicit_build_dir).expanduser()
    else:
        build_dir = _dendra_cache_root() / name
    return name, build_dir, source_hash


def _cached_extension_artifacts(build_dir: Path, extension_name: str) -> List[str]:
    if not build_dir.is_dir():
        return []
    suffixes = (".so", ".pyd", ".dll", ".dylib")
    patterns = tuple(f"{extension_name}*{suffix}" for suffix in suffixes)
    return sorted(
        {
            str(path)
            for pattern in patterns
            for path in build_dir.glob(pattern)
            if path.is_file()
        }
    )


def planned_build_info() -> Dict[str, object]:
    """Return the native bitpack build plan without compiling or loading it."""
    here = Path(__file__).resolve().parent
    sources = [
        str(here / "netcon_bitpack_kernel.cpp"),
        str(here / "netcon_bitpack_kernel.cu"),
    ]
    arch_list = _resolve_cuda_arch_list()
    arch_flags = _cuda_gencode_flags(arch_list)
    toolchain = native_toolchain_info()
    name, build_dir, source_hash = _extension_identity(
        sources, arch_list, arch_flags, toolchain=toolchain
    )
    cached_artifacts = _cached_extension_artifacts(build_dir, name)
    return {
        "extension_name": name,
        "build_directory": str(build_dir),
        "cache_root": str(_dendra_cache_root()),
        "sources": sources,
        "missing_sources": [source for source in sources if not Path(source).is_file()],
        "source_hash": source_hash[:16],
        "arch_list": list(arch_list),
        "arch_flags": list(arch_flags),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": (
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
        "toolchain": toolchain,
        "cached_artifacts": cached_artifacts,
    }


def _load_extension():
    global _EXTENSION, _LAST_BUILD_INFO
    if _EXTENSION is not None:
        return _EXTENSION
    if _DISABLED:
        return None
    if not torch.cuda.is_available():
        _set_error(RuntimeError("CUDA is not available"))
        return None

    try:
        from torch.utils.cpp_extension import load

        plan = planned_build_info()
        _LAST_BUILD_INFO = dict(plan)
        sources = list(plan["sources"])
        missing = list(plan["missing_sources"])
        if missing:
            raise FileNotFoundError(
                f"missing NetCon bitpack extension sources: {missing}"
            )

        arch_flags = list(plan["arch_flags"])
        name = str(plan["extension_name"])
        build_dir = Path(str(plan["build_directory"]))
        build_dir.mkdir(parents=True, exist_ok=True)

        verbose = _truthy_env("DENDRA_NETCON_BITPACK_VERBOSE", False)
        extra_cuda_cflags = ["-O3"] + arch_flags
        if _truthy_env("DENDRA_NETCON_BITPACK_LINEINFO", False):
            extra_cuda_cflags.append("-lineinfo")

        extra_cflags = ["/O2"] if os.name == "nt" else ["-O3"]

        _EXTENSION = load(
            name=name,
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            build_directory=str(build_dir),
            verbose=verbose,
            with_cuda=True,
            keep_intermediates=True,
        )
        _LAST_BUILD_INFO["cached_artifacts"] = _cached_extension_artifacts(
            build_dir, name
        )
        return _EXTENSION
    except Exception as exc:  # pragma: no cover - depends on CUDA toolchain
        _disable(exc)
        return None


def is_available() -> bool:
    """Return True when the C++/CUDA extension can be attempted in this process."""
    if _DISABLED or not torch.cuda.is_available():
        return False
    # Force a load attempt here because NetCon calls is_available() before using
    # the kernel.  After the first successful load this is just a Python pointer
    # check; after the first failed load the module is disabled for the process.
    return _load_extension() is not None


def last_error() -> Optional[BaseException]:
    return _LAST_ERROR


def build_info() -> Dict[str, object]:
    """Return diagnostic information about the last attempted extension build."""
    info = dict(_LAST_BUILD_INFO)
    if _LAST_ERROR is not None:
        info["last_error"] = repr(_LAST_ERROR)
    info["disabled"] = _DISABLED
    info["loaded"] = _EXTENSION is not None
    return info


def _require_cuda_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise RuntimeError(f"{name} must be contiguous")


def pack_source_spikes(
    source_spikes: torch.Tensor,
    packed_history: torch.Tensor,
    current_time_step: torch.Tensor,
) -> None:
    _, n_words = validate_pack_structure(
        source_spikes, packed_history, current_time_step
    )
    _require_cuda_contiguous("source_spikes", source_spikes)
    _require_cuda_contiguous("packed_history", packed_history)
    _require_cuda_contiguous("current_time_step", current_time_step)
    if n_words == 0:
        return
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("NetCon bitpack C++/CUDA kernels are not available")
    ext.pack_source_spikes(source_spikes, packed_history, current_time_step)


def build_delivery(
    packed_history: torch.Tensor,
    current_time_step: torch.Tensor,
    delay_steps: torch.Tensor,
    conn_word_idx: torch.Tensor,
    conn_bit_mask: torch.Tensor,
    post_idx: torch.Tensor,
    weight: torch.Tensor,
    delivery_out: torch.Tensor,
) -> None:
    n_conn, _, _ = validate_delivery_structure(
        packed_history,
        current_time_step,
        delay_steps,
        conn_word_idx,
        conn_bit_mask,
        post_idx,
        weight,
        delivery_out,
    )
    tensors = (
        packed_history,
        current_time_step,
        delay_steps,
        conn_word_idx,
        conn_bit_mask,
        post_idx,
        weight,
        delivery_out,
    )
    if not all(t.is_cuda for t in tensors):
        raise RuntimeError("NetCon bitpack C++/CUDA kernels require CUDA tensors")
    for name, tensor in (
        ("packed_history", packed_history),
        ("current_time_step", current_time_step),
        ("delay_steps", delay_steps),
        ("conn_word_idx", conn_word_idx),
        ("conn_bit_mask", conn_bit_mask),
        ("post_idx", post_idx),
        ("weight", weight),
        ("delivery_out", delivery_out),
    ):
        _require_cuda_contiguous(name, tensor)
    if n_conn == 0:
        return
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("NetCon bitpack C++/CUDA kernels are not available")
    ext.build_delivery(
        packed_history,
        current_time_step,
        delay_steps,
        conn_word_idx,
        conn_bit_mask,
        post_idx,
        weight,
        delivery_out,
    )


def build_delivery_uniform(
    packed_history: torch.Tensor,
    current_time_step: torch.Tensor,
    delay_step: int,
    conn_word_idx: torch.Tensor,
    conn_bit_mask: torch.Tensor,
    post_idx: torch.Tensor,
    weight: torch.Tensor,
    delivery_out: torch.Tensor,
) -> None:
    n_conn, _, _ = validate_delivery_uniform_structure(
        packed_history,
        current_time_step,
        conn_word_idx,
        conn_bit_mask,
        post_idx,
        weight,
        delivery_out,
    )
    for name, tensor in (
        ("packed_history", packed_history),
        ("current_time_step", current_time_step),
        ("conn_word_idx", conn_word_idx),
        ("conn_bit_mask", conn_bit_mask),
        ("post_idx", post_idx),
        ("weight", weight),
        ("delivery_out", delivery_out),
    ):
        _require_cuda_contiguous(name, tensor)
    if n_conn == 0:
        return
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("NetCon bitpack C++/CUDA kernels are not available")
    ext.build_delivery_uniform(
        packed_history,
        current_time_step,
        int(delay_step),
        conn_word_idx,
        conn_bit_mask,
        post_idx,
        weight,
        delivery_out,
    )
