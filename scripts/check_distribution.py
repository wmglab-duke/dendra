#!/usr/bin/env python3
"""Reject stale, incomplete or incorrectly versioned release distributions.

This check does not import Dendra or require its runtime dependencies. Run it
against the wheel and source archive built from the current source revision::

    python scripts/check_distribution.py dist --ref refs/tags/v0.27.0
"""

from __future__ import annotations

import argparse
import ast
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path

NATIVE_SOURCES = (
    "dendra/models/networks/netcon_bitpack_kernel.cpp",
    "dendra/models/networks/netcon_bitpack_kernel.cu",
)
TORCH_REQUIREMENT = "torch>=2.14.0"


def source_version(root: Path) -> str:
    """Read the literal version without importing the simulation runtime."""
    tree = ast.parse((root / "dendra/__init__.py").read_text(encoding="utf-8"))
    versions = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        )
    ]
    if len(versions) != 1 or not isinstance(versions[0], str):
        raise ValueError("Expected one literal dendra.__version__ assignment.")
    configured = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    if configured["project"]["name"] != "dendra":
        raise ValueError("The configured project name is not dendra.")
    if configured["tool"]["commitizen"]["version"] != versions[0]:
        raise ValueError("Commitizen and dendra.__version__ disagree.")
    return versions[0]


def source_licenses(root: Path) -> tuple[str, dict[str, bytes]]:
    """Read the declared expression and expanded license-file inventory."""
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    expression = project["license"]
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("The configured project license must be an expression.")
    files = {}
    for pattern in project["license-files"]:
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        if not matches:
            raise ValueError(f"License-file pattern {pattern!r} matches no files.")
        for path in matches:
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    if "LICENSE.md" not in files:
        raise ValueError("The original Duke license must be declared in license-files.")
    return expression, files


def check_metadata(
    data: bytes,
    version: str,
    artifact: Path,
    *,
    license_expression: str,
    license_files: dict[str, bytes],
) -> None:
    metadata = BytesParser().parsebytes(data)
    if metadata["Name"] != "dendra" or metadata["Version"] != version:
        raise ValueError(
            f"{artifact.name}: project name or version disagrees with source."
        )
    if metadata["Requires-Python"] != ">=3.11":
        raise ValueError(
            f"{artifact.name}: Python support metadata is missing or changed."
        )
    dependencies = metadata.get_all("Requires-Dist", [])
    if TORCH_REQUIREMENT not in (item.replace(" ", "") for item in dependencies):
        raise ValueError(
            f"{artifact.name}: expected runtime dependency {TORCH_REQUIREMENT}."
        )
    if "solvers" not in metadata.get_all("Provides-Extra", []):
        raise ValueError(f"{artifact.name}: the solvers extra is missing.")
    if metadata.get_all("License-Expression", []) != [license_expression]:
        raise ValueError(
            f"{artifact.name}: License-Expression is missing or differs from source."
        )
    declared = metadata.get_all("License-File", [])
    if set(declared) != license_files.keys() or len(declared) != len(license_files):
        raise ValueError(
            f"{artifact.name}: License-File headers differ from declared license-files."
        )


def check_license_contents(
    contents: dict[str, bytes],
    license_files: dict[str, bytes],
    artifact: Path,
    *,
    prefix: str = "",
) -> None:
    for name, expected in license_files.items():
        if contents.get(prefix + name) != expected:
            if name == "LICENSE.md" and prefix:
                raise ValueError(
                    f"{artifact.name}: the original Duke license is missing or changed."
                )
            raise ValueError(
                f"{artifact.name}: {name} is missing or differs from source."
            )


def check_package_contents(
    contents: dict[str, bytes], root: Path, artifact: Path
) -> None:
    expected = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in (root / "dendra").rglob("*.py")
    }
    packaged_modules = {
        name for name in contents if name.startswith("dendra/") and name.endswith(".py")
    }
    if packaged_modules != expected.keys():
        missing = sorted(expected.keys() - packaged_modules)
        unexpected = sorted(packaged_modules - expected.keys())
        raise ValueError(
            f"{artifact.name}: Python modules differ from source; "
            f"missing={missing}, unexpected={unexpected}."
        )
    expected.update({name: (root / name).read_bytes() for name in NATIVE_SOURCES})
    for name, data in expected.items():
        if contents.get(name) != data:
            raise ValueError(
                f"{artifact.name}: {name} is missing or differs from source."
            )
    for name in contents:
        if name.startswith("dendra/") and name.endswith(
            (".so", ".pyd", ".dll", ".dylib")
        ):
            raise ValueError(f"{artifact.name}: unexpected compiled extension {name}.")


def check_distributions(directory: Path, root: Path, ref: str = "") -> str:
    version = source_version(root)
    if ref.startswith("refs/tags/") and ref != f"refs/tags/v{version}":
        raise ValueError(f"Release tag {ref} must be refs/tags/v{version}.")
    license_expression, license_files = source_licenses(root)

    wheel = directory / f"dendra-{version}-py3-none-any.whl"
    sdist = directory / f"dendra-{version}.tar.gz"
    found = {path for path in directory.iterdir() if path.is_file()}
    if found != {wheel, sdist}:
        raise ValueError(
            "Expected exactly one universal wheel and one source archive: "
            f"{wheel.name}, {sdist.name}; found {sorted(path.name for path in found)}."
        )

    with zipfile.ZipFile(wheel) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError(f"{wheel.name}: duplicate archive members.")
        wheel_contents = {
            name: archive.read(name)
            for name in archive.namelist()
            if not name.endswith("/")
        }
    dist_info = f"dendra-{version}.dist-info"
    check_metadata(
        wheel_contents[f"{dist_info}/METADATA"],
        version,
        wheel,
        license_expression=license_expression,
        license_files=license_files,
    )
    check_package_contents(wheel_contents, root, wheel)
    check_license_contents(
        wheel_contents,
        license_files,
        wheel,
        prefix=f"{dist_info}/licenses/",
    )

    with tarfile.open(sdist) as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        if len(members) != len({member.name for member in members}):
            raise ValueError(f"{sdist.name}: duplicate archive members.")
        sdist_contents = {}
        for member in members:
            prefix = f"dendra-{version}/"
            if not member.name.startswith(prefix):
                raise ValueError(
                    f"{sdist.name}: unexpected archive root {member.name}."
                )
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"{sdist.name}: unreadable member {member.name}.")
            sdist_contents[member.name.removeprefix(prefix)] = stream.read()
    check_metadata(
        sdist_contents["PKG-INFO"],
        version,
        sdist,
        license_expression=license_expression,
        license_files=license_files,
    )
    check_package_contents(sdist_contents, root, sdist)
    check_license_contents(sdist_contents, license_files, sdist)
    for name in ("README.md", "pyproject.toml", "setup.py"):
        if sdist_contents.get(name) != (root / name).read_bytes():
            raise ValueError(f"{sdist.name}: {name} is missing or differs from source.")
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--ref", default="", help="Git ref selected for this release")
    args = parser.parse_args()
    try:
        version = check_distributions(
            args.directory, Path(__file__).resolve().parents[1], args.ref
        )
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f"Distribution check failed: {error}") from error
    print(f"Dendra {version}: versions, package contents and license verified.")


if __name__ == "__main__":
    main()
