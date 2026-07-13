"""Packaging contracts for runtime-loaded native sources."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_netcon_bitpack_extension_sources_are_shipped_as_package_data():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = config["tool"]["setuptools"]["package-data"]
    declared = set(package_data["dendra.models.networks"])
    expected = {"netcon_bitpack_kernel.cpp", "netcon_bitpack_kernel.cu"}

    assert expected <= declared
    for source in expected:
        assert (ROOT / "dendra" / "models" / "networks" / source).is_file()
