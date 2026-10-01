"""Publication must reject mismatched versions, stale code and missing data."""

from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.check_distribution import NATIVE_SOURCES, check_distributions


class DistributionChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dist = self.root / "dist"
        self.dist.mkdir()
        self.source = {
            "dendra/__init__.py": b'__version__ = "0.27.0"\n',
            "dendra/models/analysis/__init__.py": b'"""Analysis."""\n',
            "pyproject.toml": (
                b'[project]\nname = "dendra"\n[tool.commitizen]\nversion = "0.27.0"\n'
            ),
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
            b"Metadata-Version: 2.4\nName: dendra\nVersion: 0.27.0\n"
            b"Requires-Python: >=3.11\nRequires-Dist: torch>=2.12.0\n"
            b"Provides-Extra: solvers\n\n"
        )

    def build_archives(self, *, missing=None, stale=None, metadata=None):
        package = {
            name: data
            for name, data in self.source.items()
            if name.startswith("dendra/") and name != missing
        }
        if stale:
            package[stale] = b"# Stale build artifact\n"
        with zipfile.ZipFile(
            self.dist / "dendra-0.27.0-py3-none-any.whl", "w"
        ) as archive:
            for name, data in package.items():
                archive.writestr(name, data)
            archive.writestr(
                "dendra-0.27.0.dist-info/METADATA", metadata or self.metadata
            )
            archive.writestr(
                "dendra-0.27.0.dist-info/licenses/LICENSE.md", self.source["LICENSE.md"]
            )
        with tarfile.open(self.dist / "dendra-0.27.0.tar.gz", "w:gz") as archive:
            for name, data in {**self.source, "PKG-INFO": self.metadata}.items():
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

    def test_reject_extra_distribution(self):
        self.build_archives()
        (self.dist / "old-release.whl").touch()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            check_distributions(self.dist, self.root)


if __name__ == "__main__":
    unittest.main()
