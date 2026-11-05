"""Publication must reject mismatched versions, stale code and missing data."""

from __future__ import annotations

import io
import re
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.check_distribution import NATIVE_SOURCES, check_distributions

LICENSE_EXPRESSION = "LicenseRef-Duke-Research-NonCommercial AND BSD-3-Clause"


class DistributionChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dist = self.root / "dist"
        self.dist.mkdir()
        self.license_files = (
            "LICENSE.md",
            "THIRD_PARTY_NOTICES.md",
            "LICENSES/PyTorch.txt",
        )
        self.source = {
            "dendra/__init__.py": b'__version__ = "0.27.0"\n',
            "dendra/models/analysis/__init__.py": b'"""Analysis."""\n',
            "pyproject.toml": (
                '[project]\nname = "dendra"\n'
                f'license = "{LICENSE_EXPRESSION}"\n'
                'license-files = ["LICENSE.md", "THIRD_PARTY_NOTICES.md", '
                '"LICENSES/PyTorch.txt"]\n'
                '[tool.commitizen]\nversion = "0.27.0"\n'
            ).encode(),
            "THIRD_PARTY_NOTICES.md": (
                b"PyTorch-derived code retains its BSD-3-Clause license.\n"
            ),
            "LICENSES/PyTorch.txt": b"Original PyTorch license and copyrights\n",
            "README.md": b"Dendra\n",
            "LICENSE.md": b"Original custom research license\n",
            "setup.py": b"from setuptools import setup\nsetup()\n",
            **{name: b"// Runtime source\n" for name in NATIVE_SOURCES},
        }
        for name, data in self.source.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.metadata = (
            "Metadata-Version: 2.4\nName: dendra\nVersion: 0.27.0\n"
            "Requires-Python: >=3.11\nRequires-Dist: torch>=2.14.0\n"
            "Provides-Extra: solvers\n"
            f"License-Expression: {LICENSE_EXPRESSION}\n"
            + "".join(f"License-File: {name}\n" for name in self.license_files)
            + "\n"
        ).encode()

    def build_archives(
        self,
        *,
        missing=None,
        stale=None,
        metadata=None,
        wheel_changes=None,
        sdist_changes=None,
    ):
        package = {
            name: data
            for name, data in self.source.items()
            if name.startswith("dendra/") and name != missing
        }
        if stale:
            package[stale] = b"# Stale build artifact\n"
        wheel_contents = {
            **package,
            "dendra-0.27.0.dist-info/METADATA": (
                self.metadata if metadata is None else metadata
            ),
            **{
                f"dendra-0.27.0.dist-info/licenses/{name}": self.source[name]
                for name in self.license_files
            },
        }
        for name, data in (wheel_changes or {}).items():
            if data is None:
                wheel_contents.pop(name, None)
            else:
                wheel_contents[name] = data
        with zipfile.ZipFile(
            self.dist / "dendra-0.27.0-py3-none-any.whl", "w"
        ) as archive:
            for name, data in wheel_contents.items():
                archive.writestr(name, data)
        sdist_contents = {**self.source, "PKG-INFO": self.metadata}
        for name, data in (sdist_changes or {}).items():
            if data is None:
                sdist_contents.pop(name, None)
            else:
                sdist_contents[name] = data
        with tarfile.open(self.dist / "dendra-0.27.0.tar.gz", "w:gz") as archive:
            for name, data in sdist_contents.items():
                member = tarfile.TarInfo(f"dendra-0.27.0/{name}")
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))

    def test_valid_release_and_candidate_branch(self):
        self.build_archives()
        for ref in ("refs/tags/v0.27.0", "refs/heads/release-candidate/0.27.0"):
            self.assertEqual(check_distributions(self.dist, self.root, ref), "0.27.0")

    def test_reject_incorrect_tag(self):
        self.build_archives()
        with self.assertRaisesRegex(ValueError, "Release tag"):
            check_distributions(self.dist, self.root, "refs/tags/v0.28.0")

    def test_reject_commitizen_version_disagreement(self):
        self.build_archives()
        config = self.root / "pyproject.toml"
        config.write_bytes(self.source["pyproject.toml"].replace(b"0.27.0", b"0.28.0"))
        with self.assertRaisesRegex(ValueError, "Commitizen"):
            check_distributions(self.dist, self.root)

    def test_reject_missing_analysis_module(self):
        self.build_archives(missing="dendra/models/analysis/__init__.py")
        with self.assertRaisesRegex(ValueError, "Python modules differ"):
            check_distributions(self.dist, self.root)

    def test_reject_missing_runtime_cuda_source(self):
        self.build_archives(missing=NATIVE_SOURCES[1])
        with self.assertRaisesRegex(ValueError, "missing or differs"):
            check_distributions(self.dist, self.root)

    def test_reject_stale_python_module(self):
        self.build_archives(stale="dendra/__init__.py")
        with self.assertRaisesRegex(ValueError, "missing or differs"):
            check_distributions(self.dist, self.root)

    def test_reject_mismatched_wheel_metadata(self):
        self.build_archives(metadata=self.metadata.replace(b"0.27.0", b"0.28.0"))
        with self.assertRaisesRegex(ValueError, "version disagrees"):
            check_distributions(self.dist, self.root)

    def test_reject_outdated_torch_minimum(self):
        self.build_archives(
            metadata=self.metadata.replace(b"torch>=2.14.0", b"torch>=2.12.0")
        )
        with self.assertRaisesRegex(ValueError, "expected runtime dependency torch"):
            check_distributions(self.dist, self.root)

    def test_reject_extra_distribution(self):
        self.build_archives()
        (self.dist / "old-release.whl").touch()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            check_distributions(self.dist, self.root)

    def test_reject_missing_changed_or_duplicate_license_expression(self):
        line = f"License-Expression: {LICENSE_EXPRESSION}\n".encode()
        replacements = (
            b"",
            b"License-Expression: LicenseRef-Duke-Research-NonCommercial\n",
            line + line,
        )
        for artifact, name in (
            ("wheel", "dendra-0.27.0.dist-info/METADATA"),
            ("sdist", "PKG-INFO"),
        ):
            for replacement in replacements:
                with self.subTest(artifact=artifact, expression=replacement):
                    self.build_archives(
                        **{
                            f"{artifact}_changes": {
                                name: self.metadata.replace(line, replacement)
                            }
                        }
                    )
                    with self.assertRaisesRegex(ValueError, "License-Expression"):
                        check_distributions(self.dist, self.root)

    def test_reject_missing_changed_extra_or_duplicate_license_file_headers(self):
        header_changes = [
            (f"License-File: {name}\n".encode(), b"") for name in self.license_files
        ] + [
            (
                b"License-File: LICENSES/PyTorch.txt\n",
                b"License-File: LICENSES/Other.txt\n",
            ),
            (
                b"Provides-Extra: solvers\n",
                b"Provides-Extra: solvers\nLicense-File: OTHER_LICENSE.txt\n",
            ),
            (
                b"License-File: LICENSE.md\n",
                b"License-File: LICENSE.md\nLicense-File: LICENSE.md\n",
            ),
        ]
        for artifact, name in (
            ("wheel", "dendra-0.27.0.dist-info/METADATA"),
            ("sdist", "PKG-INFO"),
        ):
            for old, new in header_changes:
                with self.subTest(artifact=artifact, old=old, new=new):
                    self.build_archives(
                        **{
                            f"{artifact}_changes": {
                                name: self.metadata.replace(old, new)
                            }
                        }
                    )
                    with self.assertRaisesRegex(ValueError, "License-File headers"):
                        check_distributions(self.dist, self.root)

    def test_reject_missing_or_changed_license_and_notice_files(self):
        for artifact, prefix in (
            ("wheel", "dendra-0.27.0.dist-info/licenses/"),
            ("sdist", ""),
        ):
            for name in self.license_files:
                for replacement in (None, self.source[name].replace(b"\n", b"\r\n")):
                    with self.subTest(artifact=artifact, name=name, value=replacement):
                        self.build_archives(
                            **{f"{artifact}_changes": {prefix + name: replacement}}
                        )
                        diagnostic = (
                            "original Duke license"
                            if artifact == "wheel" and name == "LICENSE.md"
                            else re.escape(name)
                        )
                        with self.assertRaisesRegex(ValueError, diagnostic):
                            check_distributions(self.dist, self.root)

    def test_license_file_patterns_expand_to_metadata_and_archive_paths(self):
        self.source["pyproject.toml"] = self.source["pyproject.toml"].replace(
            b'"LICENSES/PyTorch.txt"', b'"LICENSES/*.txt"'
        )
        (self.root / "pyproject.toml").write_bytes(self.source["pyproject.toml"])
        self.build_archives()
        self.assertEqual(check_distributions(self.dist, self.root), "0.27.0")

    def test_reject_unmatched_license_file_pattern(self):
        self.source["pyproject.toml"] = self.source["pyproject.toml"].replace(
            b'"LICENSES/PyTorch.txt"', b'"LICENSES/*.missing"'
        )
        (self.root / "pyproject.toml").write_bytes(self.source["pyproject.toml"])
        self.build_archives()
        with self.assertRaisesRegex(
            ValueError, "License-file pattern.*matches no files"
        ):
            check_distributions(self.dist, self.root)

    def test_reject_configuration_omitting_original_duke_license(self):
        self.source["pyproject.toml"] = self.source["pyproject.toml"].replace(
            b'"LICENSE.md", ', b""
        )
        (self.root / "pyproject.toml").write_bytes(self.source["pyproject.toml"])
        self.build_archives()
        with self.assertRaisesRegex(
            ValueError, "original Duke license must be declared"
        ):
            check_distributions(self.dist, self.root)


if __name__ == "__main__":
    unittest.main()
