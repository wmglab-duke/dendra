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


def test_jupyter_extra_declares_the_interactive_matplotlib_backend():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pyproject_extra = config["project"]["optional-dependencies"]["jupyter"]

    assert pyproject_extra == ["ipympl >= 0.9.5"]


def test_jupyter_install_docs_cover_shared_and_split_environments():
    text = (ROOT / "docs" / "installation.md").read_text(encoding="utf-8")
    section = text.split("(interactive-jupyter)=", 1)[1].split("\n## ", 1)[0]
    shared, split = " ".join(section.split()).split(
        "If the server and kernel use separate environments", 1
    )

    assert 'python -m pip install "dendra[jupyter]"' in shared
    assert "restart the Jupyter server" in shared
    assert shared.index("python -m jupyter lab") < shared.index("%matplotlib widget")
    assert "install Dendra and ipympl in the kernel environment" in split
    assert "Install ipympl in the server environment" in split
    assert "restart both" in split
    assert "prebuilt extension" in split
    assert "jupyter lab build" not in split
    assert "jupyter labextension list" in split
    assert "Failed to load model class 'MPLCanvasModel'" in split


def test_doctor_console_script_and_module_entrypoint_are_packaged():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert config["project"]["scripts"]["dendra"] == "dendra.cli:main"
    assert (ROOT / "dendra" / "cli.py").is_file()
    assert (ROOT / "dendra" / "__main__.py").is_file()
