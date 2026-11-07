"""Selective recovery replaces real branch/context data and fails closed."""

import hashlib
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import coverage
import pytest

from scripts import recover_ci_shard as recovery

SCRIPTS = Path(recovery.__file__).resolve().parent

A, B = "tests/test_left.py", "tests/test_right.py"
IDS = [A + "::test_case[0]", A + "::test_case[1]", B + "::test_keep"]
ENGINE = "def choose(value):\n    if value == 'obsolete':\n        return 1\n    if value == 'retry':\n        return 2\n    return 3\n"


def _outcome(name="passed", reason=""):
    phase = (
        "passed"
        if name in {"passed", "xpassed"}
        else "skipped"
        if name in {"skipped", "xfailed"}
        else "failed"
    )
    return {
        "outcome": name,
        "phases": {"setup": "passed", "call": phase, "teardown": "passed"},
        "reason": reason,
        "wasxfail": reason if name == "xfailed" else None,
    }


def _snapshot(root, assigned):
    snapshot = {
        "modules": {
            module: hashlib.sha256((root / module).read_bytes()).hexdigest()
            for module in {x.split("::")[0] for x in assigned}
        },
        "runtime": recovery._runtime_snapshot(root),
        "config": recovery._config_snapshot(root, {A, B}),
    }
    snapshot["final"] = deepcopy(snapshot)
    snapshot["unchanged"] = True
    return snapshot


def _coverage(root, path, contexts, traces):
    previous = Path.cwd()
    try:
        os.chdir(root)
        measured = coverage.Coverage(
            data_file=str(path),
            branch=True,
            source=[str(root / "dendra")],
            config_file=False,
        )
        measured.set_option("run:relative_files", True)
        measured.erase()
        measured.start()
        measured.switch_context(contexts["global"])
        namespace = {}
        exec(compile(ENGINE, str(root / "dendra/engine.py"), "exec"), namespace)
        for context, arguments in traces.items():
            measured.switch_context(context)
            for argument in arguments:
                namespace["choose"](argument)
        measured.stop()
        measured.save()
        measured.get_data().close()
    finally:
        os.chdir(previous)


def _artifact(
    root, directory, assigned, *, outcomes=None, traces=None, prefix="", failed=False
):
    directory.mkdir(parents=True)
    modules = {node.split("::")[0] for node in assigned}
    contexts = {
        "mode": "module",
        "active": True,
        "owned_only": True,
        "global": prefix + "ci-global",
        "modules": {module: prefix + "ci-module:" + module for module in modules},
    }
    outcomes = {node: _outcome() for node in assigned} if outcomes is None else outcomes
    if failed:
        outcomes[IDS[1]] = _outcome("failed")
    manifest = {
        "schema_version": 1,
        "index": 1,
        "total": 1,
        "selected_nodeids": list(IDS),
        "assigned_nodeids": list(assigned),
        "executed_nodeids": list(assigned),
        "module_assignments": {A: 1, B: 1},
        "collect_only": False,
        "test_outcomes": outcomes,
        "test_durations_seconds": {node: 0.01 for node in assigned},
        "test_duration_seconds": 0.01 * len(assigned),
        "collection_duration_seconds": 0.02,
        "exitstatus": int(any(x["outcome"] == "failed" for x in outcomes.values())),
        "source_snapshot": _snapshot(root, assigned),
        "coverage_contexts": contexts,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    if traces is None:
        traces = {
            contexts["modules"][module]: ["obsolete" if module == A else "keep"]
            for module in modules
        }
    _coverage(root, directory / ".coverage", contexts, traces)
    xml = ET.Element("testsuites")
    suite = ET.SubElement(xml, "testsuite", name="pytest")
    for node in assigned:
        case = ET.SubElement(
            suite,
            "testcase",
            name=node.split("::", 1)[1],
            classname=node.split("::", 1)[0],
            time="0.01",
        )
        props = ET.SubElement(case, "properties")
        ET.SubElement(props, "property", name="ci_nodeid", value=node)
        kind = outcomes[node]["outcome"]
        if kind == "failed":
            ET.SubElement(case, "failure", message="original assertion failed")
        elif kind in {"skipped", "xfailed"}:
            ET.SubElement(case, "skipped", message=outcomes[node]["reason"])
    ET.ElementTree(xml).write(directory / "junit.xml")
    (directory / "pytest.log").write_text("Original execution evidence\n")
    return directory


@pytest.fixture
def scenario(tmp_path):
    root = tmp_path / "checkout"
    (root / "dendra").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "dendra/__init__.py").write_text("")
    (root / "dendra/engine.py").write_text(ENGINE)
    (root / A).write_text(
        "import pytest\n@pytest.mark.parametrize('value', [0, 1])\ndef test_case(value):\n    assert value == 0\n"
    )
    (root / B).write_text("def test_keep():\n    assert True\n")
    (root / "pytest.ini").write_text("[pytest]\n")
    original = _artifact(root, tmp_path / "original", IDS, failed=True)
    (root / A).write_text((root / A).read_text().replace("value == 0", "value >= 0"))
    replacement = _artifact(
        root,
        tmp_path / "replacement",
        IDS[:2],
        prefix="retry|",
        traces={"retry|ci-global": ["obsolete"], "retry|ci-module:" + A: ["retry"]},
    )
    return root, original, replacement, tmp_path / "aggregate/1"


def _read(path):
    return json.loads((path / "manifest.json").read_text())


def _edit(path, change):
    record = _read(path)
    change(record)
    (path / "manifest.json").write_text(json.dumps(record))


def _arcs(path, context=None):
    data = coverage.CoverageData(basename=str(path / ".coverage"))
    data.read()
    if context:
        data.set_query_contexts(["^" + re.escape(context) + "$"])
    result = set(data.arcs("dendra/engine.py") or ())
    data.close()
    return result


def test_full_module_recovery_replaces_stale_arcs_junit_and_outcomes(scenario):
    root, original, replacement, output = scenario
    before = {
        path: path.read_bytes()
        for directory in (original, replacement)
        for path in directory.iterdir()
    }
    assert (2, 3) in _arcs(original)

    final = recovery.recover(original, [replacement], output, root)

    assert final["exitstatus"] == 0
    assert final["selected_nodeids"] == IDS and final["assigned_nodeids"] == IDS
    assert final["executed_nodeids"] == IDS
    assert all(
        record["outcome"] == "passed" for record in final["test_outcomes"].values()
    )
    assert (2, 3) not in _arcs(
        output
    )  # Neither failed old module nor retry-global arcs leak.
    assert (4, 5) in _arcs(output) and (4, 6) in _arcs(output)
    assert (4, 5) in _arcs(output, "ci-module:" + A)
    assert (4, 6) in _arcs(output, "ci-module:" + B)
    assert all(path.read_bytes() == content for path, content in before.items())
    junit = ET.parse(output / "junit.xml")
    assert len(junit.findall(".//testcase")) == 3
    assert not junit.findall(".//failure") and not junit.findall(".//error")
    provenance = json.loads((output / "recovery.json").read_text())
    assert provenance["inputs"][0]["exitstatus"] == 1
    assert provenance["replaced_modules"] == [A]
    assert provenance["inputs"][0]["sha256"]["manifest.json"]
    assert (
        output / provenance["inputs"][0]["evidence"] / "pytest.log"
    ).read_bytes() == (original / "pytest.log").read_bytes()
    assert not (output / provenance["inputs"][0]["evidence"] / ".coverage").exists()


def test_shared_arc_remains_when_an_unreplaced_module_really_executed_it(scenario):
    root, original, replacement, output = scenario
    contexts = _read(original)["coverage_contexts"]
    _coverage(
        root,
        original / ".coverage",
        contexts,
        {
            contexts["modules"][A]: ["obsolete"],
            contexts["modules"][B]: ["obsolete", "keep"],
        },
    )

    recovery.recover(original, [replacement], output, root)

    assert (2, 3) in _arcs(output, "ci-module:" + B)
    assert (2, 3) not in _arcs(output, "ci-module:" + A)


def test_recovered_unit_still_satisfies_existing_strict_inventory_gate(scenario):
    root, original, replacement, output = scenario
    recovery.recover(original, [replacement], output, root)
    from scripts.combine_ci_coverage import validate_shards

    assert validate_shards(output.parent, 1) == [(output / ".coverage").resolve()]


@pytest.mark.parametrize("field", ["assigned_nodeids", "executed_nodeids"])
def test_incomplete_module_replacement_is_rejected(scenario, field):
    root, original, replacement, output = scenario
    _edit(replacement, lambda record: record[field].pop())
    with pytest.raises(ValueError, match="inventory|incomplete|complete"):
        recovery.recover(original, [replacement], output, root)
    assert not output.exists()


def test_failed_module_replacement_is_rejected(scenario):
    root, original, replacement, output = scenario
    _edit(replacement, lambda record: record.update(exitstatus=1))
    with pytest.raises(ValueError, match="Replacement execution failed"):
        recovery.recover(original, [replacement], output, root)


def test_duplicate_executed_ids_are_rejected(scenario):
    root, original, replacement, output = scenario
    _edit(replacement, lambda record: record["executed_nodeids"].append(IDS[0]))
    with pytest.raises(ValueError, match="invalid executed inventory"):
        recovery.recover(original, [replacement], output, root)


@pytest.mark.parametrize(
    "kind", ["runtime", "global", "other_module", "stale_replacement"]
)
def test_changed_sources_invalidate_the_required_scope(scenario, kind):
    root, original, replacement, output = scenario
    target = {
        "runtime": root / "dendra/engine.py",
        "global": root / "pytest.ini",
        "other_module": root / B,
        "stale_replacement": root / A,
    }[kind]
    target.write_text(target.read_text() + "\n# after execution\n")
    with pytest.raises(ValueError, match="Runtime|Global|changed module|source hash"):
        recovery.recover(original, [replacement], output, root)
    assert not output.exists()


@pytest.mark.parametrize(
    "kind", ["legacy", "unknown_context", "foreign_module", "mutated_during_run"]
)
def test_unowned_or_unattested_evidence_is_rejected(scenario, kind):
    root, original, replacement, output = scenario
    if kind == "legacy":
        _edit(original, lambda record: record.pop("coverage_contexts"))
    elif kind == "mutated_during_run":
        _edit(
            original, lambda record: record["source_snapshot"].update(unchanged=False)
        )
    else:
        contexts = _read(original)["coverage_contexts"]
        extra = "" if kind == "unknown_context" else "ci-module:tests/test_foreign.py"
        _coverage(root, original / ".coverage", contexts, {extra: ["obsolete"]})
    with pytest.raises(ValueError, match="coverage|context|source changed"):
        recovery.recover(original, [replacement], output, root)


def test_failed_test_cannot_be_hidden_by_a_new_skip(scenario):
    root, original, replacement, output = scenario
    _edit(
        replacement,
        lambda record: record["test_outcomes"].update(
            {IDS[1]: _outcome("skipped", "new escape")}
        ),
    )
    with pytest.raises(ValueError, match="may not turn"):
        recovery.recover(original, [replacement], output, root)


def test_unchanged_legitimate_skip_is_retained(scenario):
    root, original, replacement, output = scenario
    for path in (original, replacement):
        _edit(
            path,
            lambda record: record["test_outcomes"].update(
                {IDS[0]: _outcome("skipped", "known platform")}
            ),
        )
        xml = ET.parse(path / "junit.xml")
        ET.SubElement(
            xml.findall(".//testcase")[0], "skipped", message="known platform"
        )
        xml.write(path / "junit.xml")
    final = recovery.recover(original, [replacement], output, root)
    assert final["test_outcomes"][IDS[0]]["outcome"] == "skipped"
    assert len(ET.parse(output / "junit.xml").findall(".//skipped")) == 1


def test_incomplete_or_failed_teardown_cannot_claim_a_passing_test(scenario):
    root, original, replacement, output = scenario
    _edit(
        replacement,
        lambda record: record["test_outcomes"][IDS[0]]["phases"].update(
            teardown="failed"
        ),
    )
    with pytest.raises(ValueError, match="teardown"):
        recovery.recover(original, [replacement], output, root)


def test_junit_requires_exact_case_identity(scenario):
    root, original, replacement, output = scenario
    xml = ET.parse(replacement / "junit.xml")
    xml.findall(".//testcase")[0].remove(xml.findall(".//testcase/properties")[0])
    xml.write(replacement / "junit.xml")
    with pytest.raises(ValueError, match="ci_nodeid"):
        recovery.recover(original, [replacement], output, root)


def test_duplicate_module_replacements_and_output_overwrite_are_rejected(scenario):
    root, original, replacement, output = scenario
    with pytest.raises(ValueError, match="duplicate modules"):
        recovery.recover(original, [replacement, replacement], output, root)
    output.mkdir(parents=True)
    with pytest.raises(ValueError, match="already exists"):
        recovery.recover(original, [replacement], output, root)


def test_second_recovery_retains_portable_history_without_obsolete_arcs(scenario):
    root, original, replacement, first = scenario
    recovery.recover(original, [replacement], first, root)
    (root / A).write_text((root / A).read_text() + "\n# another fixture repair\n")
    retry = _artifact(
        root,
        first.parent.parent / "next-retry",
        IDS[:2],
        traces={"ci-module:" + A: ["keep"]},
    )
    final = first.parent / "next-unit"

    recovery.recover(first, [retry], final, root)

    assert (4, 5) not in _arcs(final)
    provenance = json.loads((final / "recovery.json").read_text())
    previous = provenance["inputs"][0]["previous_recovery"]
    assert previous["inputs"][0]["exitstatus"] == 1
    assert (final / "evidence/original/evidence/original/pytest.log").read_bytes() == (
        original / "pytest.log"
    ).read_bytes()


def test_replacement_must_retain_the_full_ordinary_selected_inventory(scenario):
    root, original, replacement, output = scenario
    _edit(replacement, lambda record: record.update(selected_nodeids=IDS[:2]))
    with pytest.raises(ValueError, match="full selected inventory"):
        recovery.recover(original, [replacement], output, root)


def test_dynamic_test_import_requires_broader_invalidation(scenario):
    root, original, replacement, output = scenario
    (root / A).write_text(
        (root / A).read_text()
        + "\nimport importlib\ndef resolve(name):\n    return importlib.import_module(name)\n"
    )
    with pytest.raises(ValueError, match="Unsafe dynamic test imports"):
        recovery.recover(original, [replacement], output, root)


def test_recovery_rechecks_sources_and_artifacts_before_publishing(
    scenario, monkeypatch
):
    root, original, replacement, output = scenario
    copy_context = recovery._copy_context

    def mutate_input(*args):
        copy_context(*args)
        (replacement / "pytest.log").write_text("changed during recovery")

    monkeypatch.setattr(recovery, "_copy_context", mutate_input)
    with pytest.raises(ValueError, match="Input artifacts changed"):
        recovery.recover(original, [replacement], output, root)
    assert not output.exists() and not list(output.parent.glob("1.tmp-*"))


def test_real_collection_ownership_recovery_and_unfiltered_union(tmp_path):
    """A patched module/class collection cannot leak; unselected C stays global."""
    root = tmp_path / "checkout"
    (root / "dendra").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "dendra/__init__.py").write_text("")
    engine = "def pick(tag):\n    if tag == 'old':\n        return 1\n    if tag == 'new':\n        return 2\n    if tag == 'cuda':\n        return 3\n    return 4\n"
    (root / "dendra/engine.py").write_text(engine)
    (root / "pytest.ini").write_text("[pytest]\nmarkers =\n    cpu\n    cuda\n")
    (root / "coverage.ini").write_text(
        "[run]\nbranch = true\nrelative_files = true\nsource = dendra\n"
    )
    source_a = (
        "from dendra.engine import pick\n"
        "def test_profile():\n"
        '    """old"""\n'
        "    assert True\n"
        "assert pick(test_profile.__doc__) in {1, 2}\n"
        "class TestParametrized:\n"
        "    @staticmethod\n"
        "    def pytest_generate_tests(metafunc):\n"
        "        assert pick(test_profile.__doc__) in {1, 2}\n"
        "        metafunc.parametrize('flag', [0])\n"
        "    def test_a(self, flag):\n"
        "        assert test_profile.__doc__ == 'new'\n"
    )
    (root / "tests/test_a.py").write_text(source_a)
    (root / "tests/test_b.py").write_text(
        "from dendra.engine import pick\nassert pick('keep') == 4\ndef test_b():\n    assert pick('keep') == 4\n"
    )
    (root / "tests/test_cuda.py").write_text(
        "import pytest\nfrom dendra.engine import pick\npytestmark = pytest.mark.cuda\nassert pick('cuda') == 3\ndef test_cuda():\n    assert False\n"
    )
    environment = os.environ.copy()
    environment.update(
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTHONPATH=str(SCRIPTS) + os.pathsep + str(root),
    )
    environment.pop("PYTEST_ADDOPTS", None)

    def run(name, index=None, *, recovery_module=None):
        output = root / "artifacts" / name
        output.mkdir(parents=True)
        environment["COVERAGE_FILE"] = str(output / ".coverage")
        command = [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-p",
            "pytest_cov",
            "-c",
            str(root / "pytest.ini"),
            "--rootdir=" + str(root),
            "--confcutdir=" + str(root),
            "-q",
            "-m",
            "not cuda",
            "--cov=dendra",
            "--cov-config=" + str(root / "coverage.ini"),
            "--cov-report=",
            "--junitxml=" + str(output / "junit.xml"),
        ]
        if index is not None:
            command += [
                "-p",
                "ci_shard",
                "--ci-shard-index=" + str(index),
                "--ci-shard-total=2",
                "--ci-shard-output=" + str(output),
            ]
        if recovery_module:
            command += ["--ci-recovery-module=" + recovery_module]
        command.append("tests")
        result = subprocess.run(
            command,
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        (output / "pytest.log").write_text(result.stdout + result.stderr)
        assert result.returncode == (1 if name == "original-a" else 0), (
            result.stdout + result.stderr
        )
        return output

    original = run("original-a", 1)
    cached_b = run("cached-b", 2)
    assert (2, 3) in _arcs(original, "ci-module:tests/test_a.py")
    assert (2, 3) not in _arcs(original, "ci-global")
    assert (2, 3) not in _arcs(cached_b)
    assert (6, 7) in _arcs(cached_b, "ci-global")
    (root / "tests/test_a.py").write_text(source_a.replace('"""old"""', '"""new"""'))
    retry = run("retry-a", 1, recovery_module="tests/test_a.py")
    recovered = root / "artifacts/recovered-a"
    # Match the fixture subprocess environment: this outer pytest checkout has
    # extra editable distribution metadata on sys.path that the fixture does not.
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SCRIPTS / "recover_ci_shard.py"),
            "--original",
            str(original),
            "--replacement",
            str(retry),
            "--output",
            str(recovered),
            "--source-root",
            str(root),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    baseline = run("unfiltered-baseline")

    # Compare the complete source denominator and branch arc union, independent
    # of module contexts, with a fresh ordinary full-lane coverage run.
    combined = {}
    for path in (recovered, cached_b):
        data = coverage.CoverageData(basename=str(path / ".coverage"))
        data.read()
        for name in data.measured_files():
            combined.setdefault(name, set()).update(data.arcs(name) or ())
        data.close()
    unfiltered = coverage.CoverageData(basename=str(baseline / ".coverage"))
    unfiltered.read()
    assert set(combined) == unfiltered.measured_files()
    assert all(
        combined[name] == set(unfiltered.arcs(name) or ())
        for name in unfiltered.measured_files()
    )
    assert (2, 3) not in _arcs(recovered) and (4, 5) in _arcs(recovered)
    assert (6, 7) in _arcs(recovered, "ci-global")
    unfiltered.close()
