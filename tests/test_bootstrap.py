"""Regression coverage for process-wide Dendra bootstrap helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch

from dendra._bootstrap import (
    _should_configure_torchinductor_cache_on_import,
    reset_torch_compiler,
    torch_compiler_warning_context,
)

_REPOSITORY_ROOT = Path(__file__).parents[1]
_PROBE_PREFIX = "DENDRA_BOOTSTRAP_PROBE="
_CACHE_ENVIRONMENT_VARIABLES = {
    "DENDRA_CONFIGURE_TORCHINDUCTOR_CACHE",
    "DENDRA_INDUCTOR_CACHE_POLICY",
    "DENDRA_INDUCTOR_CACHE_ROOT",
    "DENDRA_INDUCTOR_SESSION_ID",
    "DENDRA_INDUCTOR_DISABLE_PCH",
    "DENDRA_INDUCTOR_COMPILE_THREADS",
    "DENDRA_INDUCTOR_CLEANUP",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS",
    "TORCHINDUCTOR_COMPILE_THREADS",
    "_DENDRA_TEST_IMPORT_TORCH_FIRST",
}
_IMPORT_PROBE = f"""
import json
import os
import warnings
from pathlib import Path

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    if os.environ.get("_DENDRA_TEST_IMPORT_TORCH_FIRST") == "1":
        import torch
    import dendra

from torch._inductor import cpu_vec_isa

cache_root = Path(os.environ["DENDRA_INDUCTOR_CACHE_ROOT"])
cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
result = {{
    "cache_dir": cache_dir,
    "cache_dir_exists": bool(cache_dir and Path(cache_dir).is_dir()),
    "cache_root_exists": cache_root.exists(),
    "compile_threads": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
    "cpu_isa_cache_exists": (Path.home() / ".cache" / "dendra").exists(),
    "cpu_isa_hook_module": cpu_vec_isa.valid_vec_isa_list.__module__,
    "pch": os.environ.get("TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS"),
    "warnings": [str(item.message) for item in caught],
}}
print("{_PROBE_PREFIX}" + json.dumps(result, sort_keys=True))
"""


def _probe_dendra_import(tmp_path, overrides=None, *, import_torch_first=False):
    environment = os.environ.copy()
    for name in _CACHE_ENVIRONMENT_VARIABLES:
        environment.pop(name, None)

    cache_root = tmp_path / "dendra-inductor-cache"
    environment.update(
        {
            "DENDRA_INDUCTOR_CACHE_ROOT": os.fspath(cache_root),
            "DENDRA_INDUCTOR_SESSION_ID": "pytest-session",
        }
    )
    if import_torch_first:
        environment["_DENDRA_TEST_IMPORT_TORCH_FIRST"] = "1"
    if overrides:
        environment.update(overrides)

    completed = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        cwd=os.fspath(_REPOSITORY_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    matches = [
        line.removeprefix(_PROBE_PREFIX)
        for line in completed.stdout.splitlines()
        if line.startswith(_PROBE_PREFIX)
    ]
    assert len(matches) == 1, completed.stdout
    return json.loads(matches[0]), cache_root


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("0", False),
        ("false", False),
        ("1", True),
        ("TRUE", True),
        ("yes", True),
        ("On", True),
    ],
)
def test_torchinductor_cache_import_gate_is_default_off(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("DENDRA_CONFIGURE_TORCHINDUCTOR_CACHE", raising=False)
    else:
        monkeypatch.setenv("DENDRA_CONFIGURE_TORCHINDUCTOR_CACHE", value)

    assert _should_configure_torchinductor_cache_on_import() is expected


@pytest.mark.parametrize("gate", [None, "0", "false", "off"])
def test_dendra_import_does_not_configure_cache_without_opt_in(tmp_path, gate):
    overrides = {
        "DENDRA_INDUCTOR_CACHE_POLICY": "process",
        "HOME": os.fspath(tmp_path / "home"),
    }
    if gate is not None:
        overrides["DENDRA_CONFIGURE_TORCHINDUCTOR_CACHE"] = gate

    result, cache_root = _probe_dendra_import(tmp_path, overrides)

    assert result["cache_dir"] != os.fspath(cache_root / "pytest-session")
    assert result["cache_root_exists"] is False
    assert result["compile_threads"] is None
    assert result["cpu_isa_cache_exists"] is False
    assert result["cpu_isa_hook_module"] == "torch._inductor.cpu_vec_isa"
    assert result["pch"] is None
    assert not cache_root.exists()


def test_dendra_import_configures_process_cache_when_opted_in(tmp_path):
    result, cache_root = _probe_dendra_import(
        tmp_path,
        {"DENDRA_CONFIGURE_TORCHINDUCTOR_CACHE": "1"},
    )

    expected_cache = cache_root / "pytest-session"
    assert result["cache_dir"] == os.fspath(expected_cache)
    assert result["cache_dir_exists"] is True
    assert result["cache_root_exists"] is True
    assert result["compile_threads"] == "1"
    assert result["pch"] == "0"
    assert not any("torch was imported before Dendra" in w for w in result["warnings"])


def test_dendra_import_gate_still_respects_disabled_policy(tmp_path):
    result, cache_root = _probe_dendra_import(
        tmp_path,
        {
            "DENDRA_CONFIGURE_TORCHINDUCTOR_CACHE": "1",
            "DENDRA_INDUCTOR_CACHE_POLICY": "off",
            "HOME": os.fspath(tmp_path / "home"),
        },
    )

    assert result["cache_dir"] != os.fspath(cache_root / "pytest-session")
    assert result["cache_root_exists"] is False
    assert result["compile_threads"] is None
    assert result["cpu_isa_cache_exists"] is False
    assert result["cpu_isa_hook_module"] == "torch._inductor.cpu_vec_isa"
    assert result["pch"] is None
    assert not cache_root.exists()


def test_dendra_import_opt_in_preserves_user_torchinductor_settings(tmp_path):
    custom_cache = tmp_path / "custom-cache"
    result, _ = _probe_dendra_import(
        tmp_path,
        {
            "DENDRA_CONFIGURE_TORCHINDUCTOR_CACHE": "1",
            "TORCHINDUCTOR_CACHE_DIR": os.fspath(custom_cache),
            "TORCHINDUCTOR_COMPILE_THREADS": "7",
            "TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS": "1",
        },
    )

    assert result["cache_dir"] == os.fspath(custom_cache)
    assert result["compile_threads"] == "7"
    assert result["pch"] == "1"


def test_dendra_import_opt_in_warns_and_skips_if_torch_is_already_loaded(tmp_path):
    result, cache_root = _probe_dendra_import(
        tmp_path,
        {"DENDRA_CONFIGURE_TORCHINDUCTOR_CACHE": "1"},
        import_torch_first=True,
    )

    assert result["cache_dir"] != os.fspath(cache_root / "pytest-session")
    assert result["cache_root_exists"] is False
    assert not cache_root.exists()
    assert any("torch was imported before Dendra" in w for w in result["warnings"])


def test_compiler_warning_context_filters_only_known_deprecations():
    autograd_message = (
        "<class 'torch.autograd.function.Function'> should not be instantiated. "
        "Methods on autograd functions are all static."
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with torch_compiler_warning_context():
            warnings.warn_explicit(
                "torch.jit.script_method is deprecated",
                DeprecationWarning,
                filename="torch/jit/_script.py",
                lineno=365,
                module="torch.jit._script",
            )
            warnings.warn(autograd_message, DeprecationWarning, stacklevel=1)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(DeprecationWarning, match="unrelated deprecation"):
            with torch_compiler_warning_context():
                warnings.warn("unrelated deprecation", DeprecationWarning, stacklevel=1)


def test_compiler_reset_filters_only_pytorch_script_method_deprecation(monkeypatch):
    calls = []

    def reset_with_known_warning():
        calls.append("known")
        warnings.warn_explicit(
            "torch.jit.script_method is deprecated",
            DeprecationWarning,
            filename="torch/jit/_script.py",
            lineno=365,
            module="torch.jit._script",
        )

    monkeypatch.setattr(torch.compiler, "reset", reset_with_known_warning)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reset_torch_compiler()
    assert calls == ["known"]

    def dynamo_reset_with_known_warning():
        calls.append("dynamo")
        warnings.warn_explicit(
            "torch.jit.script_method is deprecated",
            DeprecationWarning,
            filename="torch/jit/_script.py",
            lineno=365,
            module="torch.jit._script",
        )

    monkeypatch.setattr(torch._dynamo, "reset", dynamo_reset_with_known_warning)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reset_torch_compiler(prefer_public=False)
    assert calls == ["known", "dynamo"]

    def reset_with_unrelated_warning():
        calls.append("unrelated")
        warnings.warn("unrelated compiler warning", RuntimeWarning, stacklevel=1)

    monkeypatch.setattr(torch.compiler, "reset", reset_with_unrelated_warning)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeWarning, match="unrelated compiler warning"):
            reset_torch_compiler()
    assert calls == ["known", "dynamo", "unrelated"]
