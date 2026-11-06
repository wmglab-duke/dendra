"""Small artifact/orchestration checks; no simulation or pytest subprocesses."""

import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import coverage

from scripts import run_ci_shards as runner

SCRIPT = Path(runner.__file__).resolve()


class ShardRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        (self.repo / "tests").mkdir()
        self.output = self.repo / "ci-results"
        self.output.mkdir()
        self.state = self.repo / "state"
        self.state.mkdir()
        self.module = "tests/test_a.py"
        (self.repo / self.module).write_text(
            "def test_one(): pass\ndef test_two(): pass\n"
        )
        self.nodes = [self.module + "::test_one", self.module + "::test_two"]
        other = [f"tests/test_{index}.py::test_one" for index in range(2, 41)]
        assignments = {
            self.module: 1,
            **{node.split("::")[0]: index for index, node in enumerate(other, 2)},
        }
        self.current = {
            "sha256": "global-current",
            "configuration": {"profile_sha256": "profile"},
            "selected_nodeids": self.nodes + other,
            "module_assignments": assignments,
            "estimated_loads": [1.0] * 40,
            "source_snapshot": {
                "runtime": {"sha256": "runtime"},
                "config": {"sha256": "config", "selective_recovery_safe": True},
            },
        }

    def artifact(self, directory, *, failure=False, fingerprint=None):
        directory.mkdir(parents=True)
        manifest = {
            "schema_version": 1,
            "index": 1,
            "total": 40,
            "collect_only": False,
            "exitstatus": int(failure),
            "selected_nodeids": self.current["selected_nodeids"],
            "assigned_nodeids": self.nodes,
            "executed_nodeids": self.nodes,
            "module_assignments": self.current["module_assignments"],
            "duration_profile": {"sha256": "profile"},
            "coverage_contexts": {
                "mode": "module",
                "active": True,
                "owned_only": True,
                "global": "ci-global",
                "modules": {self.module: "ci-module:" + self.module},
            },
            "test_durations_seconds": {node: 0.1 for node in self.nodes},
            "source_snapshot": {
                **self.current["source_snapshot"],
                "unchanged": True,
                "modules": runner.module_hashes(self.repo, self.nodes),
            },
            "test_outcomes": {
                node: {
                    "outcome": "failed"
                    if failure and node == self.nodes[0]
                    else "passed",
                    "phases": {
                        "setup": "passed",
                        "call": "failed"
                        if failure and node == self.nodes[0]
                        else "passed",
                        "teardown": "passed",
                    },
                }
                for node in self.nodes
            },
        }
        manifest["source_snapshot"]["final"] = {
            key: manifest["source_snapshot"][key]
            for key in ("modules", "runtime", "config")
        }
        runner.write(directory / "manifest.json", manifest)
        runner.write(
            directory / "runner-context.json",
            {"sha256": fingerprint or self.current["sha256"]},
        )
        runner.write(directory / "completion.json", {"exitcode": int(failure)})
        runner.write(
            directory / "invocation.json",
            {"command": ["python", "-m", "pytest", "tests"]},
        )
        (directory / "pytest.log").write_text("Complete pytest output\n")
        (directory / "junit.xml").write_text(
            "<testsuite>"
            + "".join(
                f"<testcase name='{node}'><properties><property name='ci_nodeid' value='{node}'/></properties></testcase>"
                for node in self.nodes
            )
            + "</testsuite>"
        )
        data = coverage.CoverageData(basename=str(directory / ".coverage"))
        data.set_context("ci-module:" + self.module)
        data.add_arcs({"dendra/model.py": [(1, 2)]})
        data.write()
        runner.seal(directory)
        return manifest

    def test_ten_jobs_cover_all_forty_units_once(self):
        owners = [runner.units_for_job(index) for index in range(1, 11)]
        self.assertTrue(all(len(units) == 4 for units in owners))
        self.assertEqual(
            sorted(unit for units in owners for unit in units), list(range(1, 41))
        )
        self.assertEqual(runner.units_for_job(None), list(range(1, 41)))
        self.assertEqual(runner.units_for_job(1), [1, 11, 21, 31])
        for index in (0, 11, True, 1.5):
            with self.subTest(index=index), self.assertRaises(ValueError):
                runner.units_for_job(index)

    def test_cli_rejects_unignored_output_before_starting_collection(self):
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        arguments = [str(SCRIPT), "--output-dir", str(self.repo / "unignored")]
        error = io.StringIO()
        with (
            patch.object(sys, "argv", arguments),
            patch.object(runner, "repository", return_value=self.repo),
            patch.object(
                runner, "execute", side_effect=AssertionError("No collection")
            ),
            redirect_stderr(error),
            self.assertRaises(SystemExit) as failure,
        ):
            runner.main()
        self.assertEqual(failure.exception.code, 2)
        self.assertIn("Gitignored or outside the checkout", error.getvalue())
        (self.repo / ".gitignore").write_text("ci-results/\n")
        with self.assertRaisesRegex(ValueError, "ci-results-runner"):
            runner.validate_output(self.repo, self.output)
        (self.repo / ".gitignore").write_text("ci-results/\nci-results-runner/\n")
        runner.validate_output(self.repo, self.output)
        runner.validate_output(self.repo, self.repo.parent / "outside-results")

    def test_collection_failure_exposes_bounded_traceback_and_stops_execution(self):
        log_content = (
            "".join(f"<Function test_collected_{index}>\n" for index in range(10_000))
            + "Traceback (most recent call last):\n"
            + '  File "tests/test_example.py", line 3, in <module>\n'
            + "ImportError: cannot import name 'expected_hook'\n"
            + "Interrupted: 1 error during collection\n"
        )

        def execute(repo, output, command, context):
            self.assertIn("--collect-only", command)
            self.assertIn("--cov-branch", command)
            self.assertEqual(command[command.index("-W") + 1], "error")
            output.mkdir()
            (output / "pytest.log").write_text(log_content)
            return 2

        error = io.StringIO()
        with (
            patch.object(sys, "argv", [str(SCRIPT), "--output-dir", str(self.output)]),
            patch.dict(
                runner.os.environ, {"CI_NODE_INDEX": "1", "CI_NODE_TOTAL": "10"}
            ),
            patch.object(runner, "repository", return_value=self.repo),
            patch.object(runner, "validate_output"),
            patch.object(runner, "execute", side_effect=execute) as called,
            patch.object(
                runner, "run_unit", side_effect=AssertionError("No test execution")
            ),
            redirect_stderr(error),
            self.assertRaises(SystemExit) as failure,
        ):
            runner.main()
        self.assertEqual(failure.exception.code, 2)
        self.assertEqual(called.call_count, 1)
        log_path = called.call_args.args[1] / "pytest.log"
        self.assertEqual(log_path.read_text(), log_content)
        message = error.getvalue()
        self.assertIn("Full-suite collection failed (pytest exit 2)", message)
        self.assertIn(str(log_path), message)
        self.assertIn("ImportError: cannot import name 'expected_hook'", message)
        self.assertIn("Interrupted: 1 error during collection", message)
        self.assertIn("earlier collection output omitted", message)
        self.assertNotIn("<Function test_collected_0>", message)
        self.assertLessEqual(len(message.splitlines()), 83)
        self.assertLess(len(message), 17_000)

    def test_successful_collection_does_not_echo_its_log(self):
        def execute(repo, output, command, context):
            output.mkdir()
            (output / "pytest.log").write_text("Successful collection detail\n")
            runner.write(output / "manifest.json", {})
            return 0

        error, output = io.StringIO(), io.StringIO()
        with (
            patch.object(sys, "argv", [str(SCRIPT), "--output-dir", str(self.output)]),
            patch.dict(
                runner.os.environ, {"CI_NODE_INDEX": "1", "CI_NODE_TOTAL": "10"}
            ),
            patch.object(runner, "repository", return_value=self.repo),
            patch.object(runner, "validate_output"),
            patch.object(runner, "execute", side_effect=execute),
            patch.object(runner, "context", return_value=self.current),
            patch.object(runner, "units_for_job", return_value=[1]),
            patch.object(
                runner, "run_unit", return_value={"unit": 1, "status": "passed"}
            ),
            redirect_stderr(error),
            redirect_stdout(output),
        ):
            self.assertEqual(runner.main(), 0)
        self.assertEqual(error.getvalue(), "")
        self.assertNotIn("Successful collection detail", output.getvalue())

    def test_collection_diagnostics_handle_missing_empty_and_non_utf8_logs(self):
        log = self.state / "pytest.log"
        missing = runner.collection_failure(log, 3)
        self.assertIn("pytest exit 3", missing)
        self.assertIn("Unable to read collection log", missing)
        log.write_bytes(b"")
        self.assertIn("collection log is empty", runner.collection_failure(log, 2))
        log.write_bytes(b"invalid output: \xff\nRuntimeError: collection failed\n")
        self.assertIn(
            "RuntimeError: collection failed", runner.collection_failure(log, 2)
        )
        log.write_bytes(b"x" * 100_000 + b"\nRuntimeError: final error\n")
        bounded = runner.collection_failure(log, 2)
        self.assertLess(len(bounded), 17_000)
        self.assertIn("RuntimeError: final error", bounded)

    def test_valid_unit_resumes_without_invoking_tests(self):
        self.artifact(self.output / "1")
        with patch.object(
            runner, "execute", side_effect=AssertionError("No test job is needed")
        ):
            result = runner.run_unit(
                self.repo, self.output, self.state, self.current, 1
            )
        self.assertEqual(result["status"], "resumed")

    def test_runtime_configuration_mismatch_invalidates_resume(self):
        directory = self.output / "1"
        self.artifact(directory, fingerprint="old-global")
        with self.assertRaisesRegex(ValueError, "fingerprint changed"):
            runner.inspect_artifacts(directory, 1, self.current)

    def test_changed_artifact_or_missing_coverage_invalidates_resume(self):
        directory = self.output / "1"
        self.artifact(directory)
        (directory / "junit.xml").write_text("<testsuite/>")
        with self.assertRaisesRegex(ValueError, "changed artifact"):
            runner.inspect_artifacts(directory, 1, self.current)
        runner.seal(directory)
        (directory / ".coverage").unlink()
        with self.assertRaises(ValueError):
            runner.inspect_artifacts(directory, 1, self.current)

    def test_failed_or_changed_test_requires_the_entire_module(self):
        directory = self.output / "1"
        manifest = self.artifact(directory, failure=True)
        self.assertEqual(runner.recovery_modules(self.repo, manifest), [self.module])
        manifest["test_outcomes"][self.nodes[0]] = {
            "outcome": "passed",
            "phases": {"call": "passed"},
        }
        (self.repo / self.module).write_text(
            "# fixture change with unchanged node IDs\n"
        )
        self.assertEqual(runner.recovery_modules(self.repo, manifest), [self.module])

    def test_changed_selection_never_reuses_a_unit(self):
        directory = self.output / "1"
        self.artifact(directory)
        self.current["selected_nodeids"] = self.current["selected_nodeids"] + [
            self.module + "::test_added"
        ]
        with self.assertRaisesRegex(ValueError, "selected IDs or assignment changed"):
            runner.inspect_artifacts(directory, 1, self.current)

    def test_pristine_resume_rejects_incomplete_phases_or_unowned_coverage(self):
        directory = self.output / "1"
        manifest = self.artifact(directory)
        manifest["test_outcomes"][self.nodes[0]]["phases"].pop("teardown")
        runner.write(directory / "manifest.json", manifest)
        runner.seal(directory)
        with self.assertRaisesRegex(ValueError, "teardown"):
            runner.inspect_artifacts(directory, 1, self.current)
        manifest["test_outcomes"][self.nodes[0]]["phases"]["teardown"] = "passed"
        manifest["coverage_contexts"]["owned_only"] = False
        runner.write(directory / "manifest.json", manifest)
        runner.seal(directory)
        with self.assertRaisesRegex(ValueError, "owned module coverage"):
            runner.inspect_artifacts(directory, 1, self.current)

    def test_missing_artifact_runs_a_complete_fresh_unit(self):
        directory = self.output / "1"
        self.artifact(directory)
        (directory / "junit.xml").unlink()

        def execute(repo, output, command, context):
            self.assertIn("tests", command)
            self.assertIn("--ci-shard-total=40", command)
            self.artifact(output)
            return 0

        with patch.object(runner, "execute", side_effect=execute) as called:
            result = runner.run_unit(
                self.repo, self.output, self.state, self.current, 1
            )
        self.assertEqual(called.call_count, 1)
        self.assertEqual(result["status"], "passed")

    def test_module_retry_failure_does_not_publish_a_success(self):
        self.artifact(self.output / "1", failure=True)
        with patch.object(runner, "execute", return_value=1) as called:
            result = runner.run_unit(
                self.repo, self.output, self.state, self.current, 1
            )
        self.assertEqual(result["status"], "failed")
        command = called.call_args.args[2]
        self.assertIn("--ci-recovery-module=" + self.module, command)
        self.assertIn("tests", command)
        self.assertIn("--ci-shard-total=1", command)

    def test_unsafe_dynamic_import_failure_reruns_the_complete_unit(self):
        self.current["source_snapshot"]["config"]["selective_recovery_safe"] = False
        self.artifact(self.output / "1", failure=True)

        def execute(repo, output, command, context):
            self.assertIn("--ci-shard-total=40", command)
            self.assertFalse(
                any(arg.startswith("--ci-recovery-module=") for arg in command)
            )
            self.artifact(output)
            return 0

        with patch.object(runner, "execute", side_effect=execute):
            result = runner.run_unit(
                self.repo, self.output, self.state, self.current, 1
            )
        self.assertEqual(result["status"], "passed")

    def test_changed_module_delegates_recovery_and_revalidates(self):
        self.artifact(self.output / "1")
        (self.repo / self.module).write_text("# changed fixture, stable test IDs\n")

        def execute(repo, output, command, context):
            self.assertIn("--ci-recovery-module=" + self.module, command)
            self.assertIn("tests", command)
            self.assertIn("--ci-shard-total=1", command)
            output.mkdir()
            return 0

        def recover(source_root, original, replacement, output, log):
            self.assertTrue((original / "manifest.json").is_file())
            self.assertTrue(replacement.is_dir())
            self.assertEqual(source_root, self.repo)
            self.artifact(output)
            return ["python", "-m", "recover_ci_shard"]

        with (
            patch.object(runner, "execute", side_effect=execute),
            patch.object(runner, "recover_unit", side_effect=recover),
        ):
            result = runner.run_unit(
                self.repo, self.output, self.state, self.current, 1
            )
        self.assertEqual(result["status"], "recovered")
        self.assertEqual(result["modules"], [self.module])
        runner.inspect_artifacts(self.output / "1", 1, self.current)


if __name__ == "__main__":
    unittest.main()
