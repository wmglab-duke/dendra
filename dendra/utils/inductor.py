from __future__ import annotations

import getpass
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union

CleanupMode = Literal["stale", "all"]
DiscoveryMode = Literal["auto", "env_only", "temp_only", "explicit_only"]

_PCH_EXTS = (".pch", ".gch")  # clang uses .pch; gch is gcc-style (rare but harmless)
_PCH_FROM_LOG_RE = re.compile(r"precompiled header '([^']+?\.(?:pch|gch))'")


@dataclass
class InductorPCHCleanupReport:
    cache_dirs_scanned: List[Path]
    precompiled_header_dirs_found: List[Path]
    removed_files: List[Path]
    removed_dirs: List[Path]
    kept_files: List[Path]
    errors: List[Tuple[Path, str]]


def _try_get_torchinductor_cache_dir_from_torch() -> Optional[Path]:
    """
    Best-effort attempt to ask torch where Inductor's cache_dir is.
    Uses internal APIs if they exist, but gracefully falls back if not.
    """
    try:
        import torch  # noqa: F401
    except Exception:
        return None

    # Most reliable (when present): torch._inductor.utils.cache_dir()
    try:
        from torch._inductor.utils import cache_dir as _cache_dir_fn  # type: ignore

        p = _cache_dir_fn()
        if p:
            return Path(p).expanduser()
    except Exception:
        pass

    # Try a few other known/likely internals
    try:
        from torch._inductor import config as _cfg  # type: ignore

        for attr in ("cache_dir", "inductor_cache_dir"):
            v = getattr(_cfg, attr, None)
            if isinstance(v, (str, os.PathLike)):
                return Path(v).expanduser()

        if hasattr(_cfg, "default_cache_dir"):
            v = _cfg.default_cache_dir()
            if isinstance(v, (str, os.PathLike)):
                return Path(v).expanduser()
    except Exception:
        pass

    return None


def discover_torchinductor_cache_dirs(
    *,
    discovery: DiscoveryMode = "auto",
    extra: Optional[Sequence[Union[str, os.PathLike]]] = None,
    include_all_torchinductor_under_temp: bool = False,
) -> List[Path]:
    """
    Discover candidate TORCHINDUCTOR_CACHE_DIR locations.

    Parameters
    ----------
    discovery:
        - "auto": env var -> torch internal -> default tempdir/torchinductor_<user>
        - "env_only": only TORCHINDUCTOR_CACHE_DIR if set
        - "temp_only": only tempdir/torchinductor_<user> (and optionally glob)
        - "explicit_only": only from `extra`
    extra:
        Optional explicit paths to treat as cache roots.
    include_all_torchinductor_under_temp:
        If True, also include any tempdir children matching torchinductor_*.
        Useful if you have multiple environments/usersuffixes, but more aggressive.

    Returns
    -------
    list[Path]
    """
    candidates: List[Path] = []

    def add(p: Optional[Union[str, os.PathLike]]) -> None:
        if not p:
            return
        pp = Path(p).expanduser()
        # do not resolve() here; temp dirs can disappear; keep it robust
        if pp not in candidates:
            candidates.append(pp)

    if discovery in ("auto", "env_only"):
        add(os.environ.get("TORCHINDUCTOR_CACHE_DIR"))

    if discovery in ("auto",) and (not os.environ.get("TORCHINDUCTOR_CACHE_DIR")):
        add(_try_get_torchinductor_cache_dir_from_torch())

    if discovery in ("auto", "temp_only"):
        user = "unknown"
        try:
            user = getpass.getuser()
        except Exception:
            pass
        add(Path(tempfile.gettempdir()) / f"torchinductor_{user}")

        if include_all_torchinductor_under_temp:
            tmp = Path(tempfile.gettempdir())
            try:
                for d in tmp.glob("torchinductor_*"):
                    add(d)
            except Exception:
                # ignore glob issues on odd temp implementations
                pass

    if discovery in ("auto", "explicit_only") and extra:
        for p in extra:
            add(p)

    # If explicit_only: ignore everything else
    if discovery == "explicit_only":
        candidates = []
        if extra:
            for p in extra:
                add(p)

    return candidates


def _iter_precompiled_header_dirs(cache_root: Path) -> Iterable[Path]:
    """
    Yield precompiled_headers dirs under a given cache root.
    Usually it's just cache_root/precompiled_headers.
    """
    p = cache_root / "precompiled_headers"
    if p.is_dir():
        yield p


def _is_stale_pch(pch_path: Path) -> bool:
    """
    Heuristic:
      - stale if header is missing, OR
      - stale if header mtime is newer than pch mtime
    This matches the common clang failure mode: header changed after pch build.
    """
    try:
        if not pch_path.is_file():
            return False
        header_path = pch_path.with_suffix("")  # .../foo.h.pch -> .../foo.h
        if not header_path.exists():
            return True
        h_mtime = header_path.stat().st_mtime
        p_mtime = pch_path.stat().st_mtime
        return h_mtime > p_mtime
    except Exception:
        # if we cannot stat reliably, treat as stale to allow recovery
        return True


def refresh_torchinductor_precompiled_headers(
    *,
    cache_dirs: Optional[Sequence[Union[str, os.PathLike]]] = None,
    discovery: DiscoveryMode = "auto",
    mode: CleanupMode = "stale",
    include_all_torchinductor_under_temp: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> InductorPCHCleanupReport:
    """
    Delete stale TorchInductor precompiled header artifacts.

    Parameters
    ----------
    cache_dirs:
        Explicit cache root(s) (i.e., TORCHINDUCTOR_CACHE_DIR). If None, we discover.
    discovery:
        How to discover cache dirs when cache_dirs is None.
    mode:
        - "stale": remove only stale *.pch/*.gch files under precompiled_headers/
        - "all":   remove the entire precompiled_headers/ directory (forces full rebuild)
    include_all_torchinductor_under_temp:
        When discovering from temp, also sweep temp/torchinductor_* directories.
    dry_run:
        If True, do not delete; just report what would be deleted.
    verbose:
        If True, print actions to stdout.

    Returns
    -------
    InductorPCHCleanupReport
    """
    if mode not in ("stale", "all"):
        raise ValueError("mode must be 'stale' or 'all'")

    if cache_dirs is None:
        roots = discover_torchinductor_cache_dirs(
            discovery=discovery,
            include_all_torchinductor_under_temp=include_all_torchinductor_under_temp,
        )
    else:
        roots = [Path(p).expanduser() for p in cache_dirs]

    report = InductorPCHCleanupReport(
        cache_dirs_scanned=roots,
        precompiled_header_dirs_found=[],
        removed_files=[],
        removed_dirs=[],
        kept_files=[],
        errors=[],
    )

    for root in roots:
        for pch_dir in _iter_precompiled_header_dirs(root):
            report.precompiled_header_dirs_found.append(pch_dir)

            if mode == "all":
                if verbose:
                    print(f"[inductor-pch] remove dir: {pch_dir}")
                if not dry_run:
                    try:
                        shutil.rmtree(pch_dir, ignore_errors=False)
                    except FileNotFoundError:
                        pass
                    except Exception as e:
                        report.errors.append((pch_dir, repr(e)))
                        continue
                report.removed_dirs.append(pch_dir)
                continue

            # mode == "stale": delete only stale .pch/.gch
            try:
                for entry in pch_dir.iterdir():
                    if not entry.is_file():
                        continue
                    if entry.suffix not in _PCH_EXTS:
                        continue

                    if _is_stale_pch(entry):
                        if verbose:
                            print(f"[inductor-pch] remove stale: {entry}")
                        if not dry_run:
                            try:
                                entry.unlink(missing_ok=True)  # py3.8+: ok
                            except TypeError:
                                # compatibility for older python
                                try:
                                    entry.unlink()
                                except FileNotFoundError:
                                    pass
                            except Exception as e:
                                report.errors.append((entry, repr(e)))
                                continue
                        report.removed_files.append(entry)
                    else:
                        report.kept_files.append(entry)
            except Exception as e:
                report.errors.append((pch_dir, repr(e)))

    return report


def refresh_torchinductor_precompiled_headers_from_error_text(
    error_text: str,
    *,
    dry_run: bool = False,
    verbose: bool = False,
) -> InductorPCHCleanupReport:
    """
    Targeted cleanup: parse an Inductor/clang error log and remove the exact .pch/.gch
    paths referenced (usually enough to fix the next retry).
    """
    paths: List[Path] = []
    for m in _PCH_FROM_LOG_RE.finditer(error_text):
        paths.append(Path(m.group(1)))

    report = InductorPCHCleanupReport(
        cache_dirs_scanned=[],
        precompiled_header_dirs_found=[],
        removed_files=[],
        removed_dirs=[],
        kept_files=[],
        errors=[],
    )

    for p in paths:
        if verbose:
            print(f"[inductor-pch] remove (from log): {p}")
        if not dry_run:
            try:
                p.unlink(missing_ok=True)
            except TypeError:
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
            except Exception as e:
                report.errors.append((p, repr(e)))
                continue
        report.removed_files.append(p)

    return report
