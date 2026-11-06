"""Exercise collection, failed-unit recovery and resume on a tiny fake project."""

import json
import subprocess
import tempfile
import unittest
from contextlib import chdir
from pathlib import Path
from unittest.mock import patch

from scripts import run_ci_shards as runner


class RunnerIntegrationTests(unittest.TestCase):
    def test_failed_leaf_module_recovers_after_commit_without_reexecuting_retained_module(
        self,
    ):
        with tempfile.TemporaryDirectory(
            prefix="dendra-runner-integration-"
        ) as temporary:
            base = Path(temporary)
            root, output, state = base / "project", base / "units", base / "state"
            for path in (root, output, state):
                path.mkdir()
            for name in ("dendra", "tests", "scripts"):
                (root / name).mkdir()
            (root / "dendra/__init__.py").write_text("from .probe import value\n")
            (root / "dendra/probe.py").write_text(
                "def value(flag):\n    if flag:\n        return 2\n    return 3\n"
            )
            (root / "pyproject.toml").write_text(
                '[tool.pytest.ini_options]\ntestpaths=["tests"]\n'
                'markers=["cpu: portable tests", "neuron: simulator tests", '
                '"cuda: accelerator tests"]\naddopts="--strict-markers"\n'
            )
            (root / ".coveragerc.cpu").write_text(
                "[run]\nrelative_files=True\nbranch=True\nsource=dendra\n"
            )
            modules = {}
            for index in range(41):
                path = f"tests/test_{index:02}.py"
                source = "import pytest\nimport dendra\n\n@pytest.mark.cpu\ndef test_one():\n"
                source += (
                    "    assert dendra.value(True) == "
                    + ("999" if index == 0 else "2")
                    + "\n"
                )
                if index == 0:
                    source += "\n@pytest.mark.cpu\ndef test_two():\n    assert dendra.value(False) == 3\n"
                (root / path).write_text(source)
                modules[path] = {"tests": 2 if index == 0 else 1, "seconds": 1.0}
            (root / "scripts/ci_durations.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "provenance": {"kind": "tiny synthetic fixture"},
                        "modules": modules,
                    }
                )
            )

            def git(*args):
                return subprocess.run(
                    [
                        "git",
                        "-c",
                        "user.name=Prototype Check",
                        "-c",
                        "user.email=prototype@example.invalid",
                        *args,
                    ],
                    cwd=root,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

            git("init", "-q")
            git("add", ".")
            git("commit", "-qm", "Initial synthetic fixture")

            def collect(name):
                target = state / name
                status = runner.execute(
                    root,
                    target,
                    runner.command(root, target, 1, collect=True),
                    {"action": "collect"},
                )
                self.assertEqual(status, 0, (target / "pytest.log").read_text())
                return runner.context(root, runner.read(target / "manifest.json"))

            with chdir(root):
                initial = collect("initial-collection")
                self.assertEqual(initial["module_assignments"]["tests/test_00.py"], 1)
                self.assertEqual(initial["module_assignments"]["tests/test_40.py"], 1)
                result = runner.run_unit(root, output, state, initial, 1)
                self.assertEqual(result["status"], "failed")
                original = runner.read(output / "1/manifest.json")
                self.assertEqual(original["exitstatus"], 1)
                retained = "tests/test_40.py::test_one"
                self.assertEqual(
                    original["test_outcomes"][retained]["outcome"], "passed"
                )

                leaf = root / "tests/test_00.py"
                leaf.write_text(leaf.read_text().replace("== 999", "== 2"))
                git("add", "tests/test_00.py")
                git("commit", "-qm", "Fix one leaf-module assertion")
                current = collect("fixed-collection")
                self.assertEqual(
                    initial["configuration"],
                    current["configuration"],
                    {
                        key: (
                            initial["source_snapshot"].get(key),
                            current["source_snapshot"].get(key),
                        )
                        for key in ("runtime", "config")
                        if initial["source_snapshot"].get(key)
                        != current["source_snapshot"].get(key)
                    },
                )
                self.assertEqual(initial["sha256"], current["sha256"])
                self.assertEqual(
                    initial["selected_nodeids"], current["selected_nodeids"]
                )
                result = runner.run_unit(root, output, state, current, 1)
                self.assertEqual(result["status"], "recovered", result)
                self.assertEqual(result["modules"], ["tests/test_00.py"])
                recovered = runner.inspect_artifacts(output / "1", 1, current)
                self.assertEqual(recovered["exitstatus"], 0)
                self.assertEqual(
                    recovered["test_outcomes"][retained],
                    original["test_outcomes"][retained],
                )
                self.assertTrue((output / "1/recovery.json").is_file())
                with patch.object(
                    runner,
                    "execute",
                    side_effect=AssertionError("Valid recovery must resume"),
                ):
                    result = runner.run_unit(root, output, state, current, 1)
                self.assertEqual(result["status"], "resumed")


if __name__ == "__main__":
    unittest.main()
