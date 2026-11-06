"""Deterministically partition the complete selected pytest suite for CI.

Load with ``python -m pytest -p scripts.ci_shard`` and provide the required
index, total and output options. Ordinary marker and keyword selection happens before
sharding. Modules stay together to preserve module-scoped fixture reuse; a
largest-first greedy assignment balances collected case counts by default.
An explicit duration profile balances estimated runtime while retaining every
selected test. The manifest records actual durations and profile provenance.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pytest

_SOURCE_SUFFIXES = {".py", ".pyi", ".c", ".cpp", ".cu", ".h", ".hpp", ".metal"}
_NATIVE_SUFFIXES = _SOURCE_SUFFIXES | {".so", ".pyd", ".dll", ".dylib"}
_IGNORED_PARTS = {"__pycache__", ".git", ".pytest_cache"}
_GENERATED_DIRECTORIES = {"artifacts", "failed-artifacts", "ci-results", "htmlcov"}
_GLOBAL_CONTEXT = "ci-global"
_MODULE_CONTEXT_PREFIX = "ci-module:"


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_files(directory: Path, suffixes: set[str]):
    if directory.is_dir():
        for path in sorted(directory.rglob("*")):
            if (
                path.is_file()
                and path.suffix in suffixes
                and not _IGNORED_PARTS.intersection(path.parts)
            ):
                yield path


def _metadata_search_paths() -> list[str]:
    """Inventory each physical metadata root once, including lazy vendoring."""
    paths = {
        str(Path(path or ".").resolve()) for path in sys.path if isinstance(path, str)
    }
    try:
        spec = importlib.util.find_spec("setuptools")
    except (ImportError, ValueError, AttributeError):
        spec = None
    if spec is not None:
        # Setuptools can append this existing directory to sys.path on import.
        # Inventory it upfront without executing setuptools or its import hooks.
        for location in spec.submodule_search_locations or ():
            vendor = Path(location) / "_vendor"
            if vendor.is_dir():
                paths.add(str(vendor.resolve()))
    return sorted(paths)


def _runtime_snapshot(root: Path) -> dict:
    """Hash runtime sources, the native solver installation and environment."""
    root = Path(root).resolve()
    files = {
        path.relative_to(root).as_posix(): _hash_file(path)
        for path in _source_files(root / "dendra", _SOURCE_SUFFIXES)
    }
    try:
        spec = importlib.util.find_spec("dendra_solvers")
    except (ImportError, ValueError, AttributeError):
        spec = None
    if spec is not None:
        locations = sorted(spec.submodule_search_locations or ())
        for index, location in enumerate(locations):
            directory = Path(location)
            for path in _source_files(directory, _NATIVE_SUFFIXES):
                name = (
                    f"@dendra_solvers/{index}/{path.relative_to(directory).as_posix()}"
                )
                files[name] = _hash_file(path)
    # Editable model installations can change without a distribution version.
    try:
        model_spec = importlib.util.find_spec("dendra_models")
    except (ImportError, ValueError, AttributeError):
        model_spec = None
    if model_spec is not None:
        for index, location in enumerate(
            sorted(model_spec.submodule_search_locations or ())
        ):
            directory = Path(location)
            for path in sorted(directory.rglob("*")):
                if path.is_file() and not _IGNORED_PARTS.intersection(path.parts):
                    name = f"@dendra_models/{index}/{path.relative_to(directory).as_posix()}"
                    files[name] = _hash_file(path)
    packages = sorted(
        [
            (distribution.metadata.get("Name") or "").lower().replace("_", "-"),
            distribution.version,
        ]
        for distribution in importlib.metadata.distributions(
            path=_metadata_search_paths()
        )
    )
    environment = {
        "python": sys.version,
        "implementation": sys.implementation.name,
        "platform": sys.platform,
        "machine": platform.machine(),
        "packages": packages,
    }
    record = {"files": dict(sorted(files.items())), "environment": environment}
    return {**record, "sha256": _digest(record)}


def _is_test_module(path: Path) -> bool:
    return fnmatch.fnmatch(path.name, "test_*.py") or fnmatch.fnmatch(
        path.name, "*_test.py"
    )


def _shared_test_support(
    root: Path, candidates: tuple[Path, ...]
) -> tuple[list, list, list, dict]:
    """Inventory test targets from project imports and test/plugin scaffolding."""
    test_paths = {
        path.relative_to(root).as_posix(): path
        for path in candidates
        if _is_test_module(path)
    }
    by_stem = {}
    for name, path in test_paths.items():
        by_stem.setdefault(path.stem, set()).add(name)
    shared, unsafe, edges, trees = set(), set(), set(), {}

    def resolve(name, importer, level=0):
        if not name:
            return set()
        if level:
            parent = importer.parent
            for _ in range(level - 1):
                parent = parent.parent
            path = parent.joinpath(*name.split(".")).with_suffix(".py")
            relative = (
                path.relative_to(root).as_posix() if path.is_relative_to(root) else ""
            )
            return {relative} if relative in test_paths else set()
        relative = name.replace(".", "/") + ".py"
        if relative in test_paths:
            return {relative}
        return set(by_stem.get(name, ())) if "." not in name else set()

    for path in candidates:
        name = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_bytes(), filename=name)
        except (SyntaxError, ValueError):
            unsafe.add((name, "unparseable"))
            continue
        trees[name] = tree
        # Runtime/compiler and standalone campaign scripts have their own
        # dynamic import machinery. Static test targets are recognized there;
        # unresolved dynamic imports fail closed in test/support scaffolding.
        support_importer = (
            _is_test_module(path)
            or path.name == "conftest.py"
            or path.relative_to(root).parts[0]
            not in {"dendra", "scripts", "examples", "benchmarks"}
        )
        dynamic_names = {"import_module", "__import__"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in {
                "importlib",
                "builtins",
            }:
                dynamic_names.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name in {"import_module", "__import__"}
                )
        for node in ast.walk(tree):
            found = set()
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.update(resolve(alias.name, path))
            elif isinstance(node, ast.ImportFrom):
                if any(
                    (alias.asname or alias.name) == "pytest_plugins"
                    for alias in node.names
                ):
                    unsafe.add((name, "dynamic-pytest-plugins"))
                found.update(resolve(node.module or "", path, node.level))
                for alias in node.names:
                    target = ".".join(filter(None, (node.module, alias.name)))
                    found.update(resolve(target, path, node.level))
            elif isinstance(node, ast.Call):
                function = node.func
                if (
                    isinstance(function, ast.Attribute)
                    and isinstance(function.value, ast.Name)
                    and function.value.id == "pytest_plugins"
                ):
                    unsafe.add((name, "dynamic-pytest-plugins"))
                function_name = (
                    function.id
                    if isinstance(function, ast.Name)
                    else (function.attr if isinstance(function, ast.Attribute) else "")
                )
                if support_importer and function_name in dynamic_names and node.args:
                    argument = node.args[0]
                    # A name can be rebound or shadowed in a function. Only
                    # literal targets are statically proven; computed names
                    # conservatively require whole-unit invalidation.
                    value = (
                        argument.value if isinstance(argument, ast.Constant) else None
                    )
                    if isinstance(value, str):
                        if value.startswith("."):
                            unsafe.add((name, "dynamic-relative-import"))
                        else:
                            found.update(resolve(value, path))
                    else:
                        unsafe.add((name, "unresolved-dynamic-import"))
            plugin_value = None
            declares_plugins = False
            if isinstance(node, ast.Assign):
                declares_plugins = any(
                    isinstance(target, ast.Name) and target.id == "pytest_plugins"
                    for target in node.targets
                )
                plugin_value = node.value
            elif isinstance(node, ast.AnnAssign):
                declares_plugins = (
                    isinstance(node.target, ast.Name)
                    and node.target.id == "pytest_plugins"
                )
                plugin_value = node.value
            elif (
                isinstance(node, ast.AugAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "pytest_plugins"
            ):
                unsafe.add((name, "dynamic-pytest-plugins"))
            if declares_plugins:
                try:
                    plugins = ast.literal_eval(plugin_value)
                except (ValueError, TypeError, SyntaxError):
                    plugins = None
                if isinstance(plugins, str):
                    plugins = [plugins]
                if isinstance(plugins, (list, tuple)) and all(
                    isinstance(plugin, str) for plugin in plugins
                ):
                    for plugin in plugins:
                        found.update(resolve(plugin, path))
                else:
                    unsafe.add((name, "dynamic-pytest-plugins"))
            shared.update(found.difference({name}))
            edges.update((name, target) for target in found.difference({name}))
    return (
        sorted(shared),
        [{"module": module, "kind": kind} for module, kind in sorted(unsafe)],
        [
            {"importer": importer, "module": module}
            for importer, module in sorted(edges)
        ],
        trees,
    )


def _project_files(root: Path) -> tuple[list[Path], str]:
    """Use Git's exact project inventory; fixture suites have no Git metadata."""
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if Path(top).resolve() == root:

            def names(*options):
                return (
                    subprocess.run(
                        ["git", "-C", str(root), "ls-files", *options, "-z"],
                        check=True,
                        capture_output=True,
                    )
                    .stdout.decode()
                    .split("\0")
                )

            tracked = {name for name in names("--cached") if name}
            untracked = {
                name
                for name in names("--others", "--exclude-standard")
                if name
                and not _IGNORED_PARTS.intersection(Path(name).parts)
                and Path(name).name != ".coverage"
                and not Path(name).name.startswith(".coverage.")
                and Path(name).name
                not in {"coverage.xml", "coverage.json", "junit.xml"}
            }
            return [
                root / name
                for name in sorted(tracked | untracked)
                if (root / name).is_file()
            ], "git-tracked-and-nonignored"
    except (OSError, subprocess.CalledProcessError):
        pass
    # Only synthetic standalone fixtures use this policy. Published outputs have
    # dedicated artifact directories and coverage's dot-prefixed data files.
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and not _IGNORED_PARTS.intersection(path.relative_to(root).parts)
        and not _GENERATED_DIRECTORIES.intersection(path.relative_to(root).parts)
        and path.name != ".coverage"
        and not path.name.startswith(".coverage.")
        and path.name not in {"junit.xml", "coverage.xml", "coverage.json"}
    ]
    return sorted(paths), "standalone-fixture"


def _config_snapshot(root: Path, selected_modules=()) -> dict:
    """Only isolated selected leaf test modules are eligible for local recovery."""
    root = Path(root).resolve()
    paths, policy = _project_files(root)
    candidates = tuple(path for path in paths if path.suffix == ".py")
    shared, unsafe, edges, trees = _shared_test_support(root, candidates)
    ordinary_modules = {
        path.relative_to(root).as_posix()
        for path in candidates
        if _is_test_module(path)
    }
    leaf_modules = (
        set(selected_modules).intersection(ordinary_modules).difference(shared)
        if not unsafe
        else set()
    )
    collection_sources = {}
    for name in sorted(leaf_modules):
        tree = trees[name]
        # Only an ordinary top-level test body is isolated. Helpers, classes,
        # hooks, module code, decorators, signatures and defaults remain global.
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                node.body = [ast.Pass()]
        collection_sources[name] = _digest(ast.dump(tree, include_attributes=False))
    files = {
        path.relative_to(root).as_posix(): _hash_file(path)
        for path in paths
        if path.relative_to(root).as_posix() not in leaf_modules
    }
    files["@ci_shard_plugin"] = _hash_file(Path(__file__))
    record = {
        "files": dict(sorted(files.items())),
        "inventory_policy": policy,
        "shared_test_modules": shared,
        "test_import_edges": edges,
        "collection_sources": collection_sources,
        "unsafe_test_imports": unsafe,
        "selective_recovery_safe": not unsafe,
    }
    return {**record, "sha256": _digest(record)}


def _module_snapshot(root: Path, modules) -> dict:
    return {name: _hash_file(root / name) for name in sorted(modules)}


def _source_differences(before: dict, after: dict, path=()) -> list[dict]:
    """Name changed snapshot components without putting their values in logs."""
    changes = []
    for name in sorted(before.keys() | after.keys()):
        location = (*path, name)
        if name not in before:
            changes.append({"path": list(location), "change": "added"})
        elif name not in after:
            changes.append({"path": list(location), "change": "removed"})
        elif isinstance(before[name], dict) and isinstance(after[name], dict):
            changes.extend(_source_differences(before[name], after[name], location))
        elif before[name] != after[name]:
            changes.append({"path": list(location), "change": "modified"})
    return changes


def _report_reason(report) -> str | None:
    representation = report.longrepr
    if not representation:
        return None
    if isinstance(representation, tuple):
        return str(representation[-1])
    return str(representation).splitlines()[-1]


def _copy_coverage_context(source, target, context: str, destination: str) -> None:
    source.set_query_contexts([r"^" + re.escape(context) + r"$"])
    target.set_context(destination)
    for filename in source.measured_files():
        values = source.arcs(filename) if source.has_arcs() else source.lines(filename)
        if values:
            (target.add_arcs if source.has_arcs() else target.add_lines)(
                {filename: values}
            )


def _coverage_context(label: str) -> str | None:
    module = sys.modules.get("coverage")
    if module is None:
        return None
    coverage = module.Coverage.current()
    if coverage is None:
        return None
    if coverage.config.dynamic_context:
        raise pytest.UsageError(
            "CI module contexts require coverage dynamic_context to be unset"
        )
    coverage.switch_context(label)
    prefix = coverage.config.context
    return f"{prefix}|{label}" if prefix else label


@pytest.hookimpl(tryfirst=False)
def pytest_load_initial_conftests(early_config, parser, args):
    # pytest-cov starts at tryfirst; pytest loads conftests at trylast. Establish
    # a global context between those hooks, before project startup executes.
    options = early_config.known_args_namespace
    if getattr(options, "ci_shard_index", None) is not None:
        _coverage_context(_GLOBAL_CONTEXT)


class _UniqueOption(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            raise argparse.ArgumentError(self, "may only be specified once")
        setattr(namespace, self.dest, values)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("ci-shard", "complete-suite CI sharding")
    group.addoption(
        "--ci-shard-index",
        type=int,
        default=None,
        action=_UniqueOption,
        help="One-based index of this CI shard (requires total and output).",
    )
    group.addoption(
        "--ci-shard-total",
        type=int,
        default=None,
        action=_UniqueOption,
        help="Total number of CI shards (requires index and output).",
    )
    group.addoption(
        "--ci-shard-output",
        default=None,
        action=_UniqueOption,
        help="Unique shard artifact directory containing manifest.json.",
    )

    group.addoption(
        "--ci-recovery-module",
        action="append",
        default=[],
        help="Select a complete module after full ordinary collection (repeatable).",
    )
    group.addoption(
        "--ci-shard-durations",
        default=None,
        action=_UniqueOption,
        help="Optional measured module-duration JSON profile (requires sharding).",
    )


def _load_duration_profile(path: Path) -> tuple[dict, float, dict]:
    raw = path.read_bytes()
    profile = json.loads(raw)
    if (
        not isinstance(profile, dict)
        or type(profile.get("schema_version")) is not int
        or profile["schema_version"] != 1
    ):
        raise ValueError("duration profile schema_version must equal 1")
    modules = profile.get("modules")
    if not isinstance(modules, dict) or not modules:
        raise ValueError("duration profile modules must be a nonempty mapping")
    for module, record in modules.items():
        if not isinstance(module, str) or not module or not isinstance(record, dict):
            raise ValueError("duration profile module entries must name module records")
        seconds, tests = record.get("seconds"), record.get("tests")
        if (
            type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError(
                f"duration profile {module!r} seconds must be finite and positive"
            )
        if type(tests) is not int or tests <= 0:
            raise ValueError(
                f"duration profile {module!r} tests must be a positive integer"
            )
    provenance = profile.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("duration profile provenance must be a nonempty mapping")
    fallback = sum(record["seconds"] for record in modules.values()) / sum(
        record["tests"] for record in modules.values()
    )
    if not math.isfinite(fallback) or fallback <= 0:
        raise ValueError("duration profile fallback must be finite and positive")
    return (
        modules,
        fallback,
        {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "provenance": provenance,
            "fallback_seconds_per_test": fallback,
        },
    )


def _module_weights(counts: dict[str, int], profile: tuple | None) -> dict:
    if profile is None:
        return dict(counts)
    modules, fallback, _metadata = profile
    weights = {
        module: count
        * (
            modules[module]["seconds"] / modules[module]["tests"]
            if module in modules
            else fallback
        )
        for module, count in counts.items()
    }
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights.values()):
        raise pytest.UsageError("CI duration profile produced invalid module weights")
    return weights


def pytest_configure(config: pytest.Config) -> None:
    duration_path = config.getoption("ci_shard_durations")
    index = config.getoption("ci_shard_index")
    total = config.getoption("ci_shard_total")
    output = config.getoption("ci_shard_output")
    recovery_modules = config.getoption("ci_recovery_module")
    if (
        index is None
        and total is None
        and output is None
        and duration_path is None
        and not recovery_modules
    ):
        return
    if index is None or total is None or output is None:
        raise pytest.UsageError(
            "--ci-shard-index, --ci-shard-total, and --ci-shard-output "
            "must be supplied together"
        )
    if total < 1 or not 1 <= index <= total:
        raise pytest.UsageError("CI shard index must satisfy 1 <= index <= total")
    if not output.strip():
        raise pytest.UsageError("--ci-shard-output must name a directory")
    if config.getoption("numprocesses", default=None):
        raise pytest.UsageError("CI sharding requires one pytest process per shard")
    if config.getoption("cov_context", default=None) == "test":
        raise pytest.UsageError(
            "CI module contexts cannot be combined with --cov-context=test"
        )
    if config.getoption("cov_fail_under", default=0):
        raise pytest.UsageError(
            "CI coverage thresholds must be evaluated after combining all units"
        )
    profile = None
    if duration_path is not None:
        try:
            profile = _load_duration_profile(Path(duration_path))
        except (OSError, ValueError, OverflowError) as exc:
            raise pytest.UsageError(
                f"Invalid CI duration profile {duration_path!r}: {exc}"
            ) from exc
    config.pluginmanager.register(
        _ShardPlugin(index, total, Path(output), profile, recovery_modules), "ci-shard"
    )


class _ShardPlugin:
    def __init__(
        self,
        index: int,
        total: int,
        output: Path,
        profile: tuple | None = None,
        recovery_modules=(),
    ) -> None:
        self.profile = profile
        self.index = index
        self.total = total
        self.output = output
        self.recovery_modules = sorted(set(recovery_modules))
        self.manifest: dict | None = None
        self.started = 0.0
        self.durations: dict[str, float] = {}
        self.root = None
        self.collected_modules = set()
        self.context = _GLOBAL_CONTEXT
        self.context_names = {}
        self.coverage_active = False
        self.collection_snapshot = None

    def _source_drift(self, phase, before, after):
        changes = _source_differences(before, after)
        details = self.output / f"source-drift-{phase}.json"
        report = {
            "schema_version": 1,
            "phase": phase,
            "changes": changes,
            "before": before,
            "after": after,
        }
        try:
            self.output.mkdir(parents=True, exist_ok=True)
            temporary = details.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            temporary.replace(details)
            evidence = f"snapshots: {details}"
        except OSError as error:
            # Diagnostics must not replace the existing UsageError/exit policy.
            evidence = f"unable to write snapshots {details}: {error}"
        labels = [
            f"{'.'.join(change['path'])[:200]} ({change['change']})"
            for change in changes[:8]
        ]
        if len(changes) > 8:
            labels.append(f"{len(changes) - 8} more changes")
        return f"{'; '.join(labels)}; {evidence}"

    def _switch(self, label):
        self.context = label
        actual = _coverage_context(label)
        if actual is not None:
            self.coverage_active = True
            self.context_names[label] = actual

    def pytest_sessionstart(self, session) -> None:
        self.root = Path(session.config.rootpath).resolve()
        self.started = time.perf_counter()
        self._switch(_GLOBAL_CONTEXT)
        self.collection_snapshot = {
            "runtime": _runtime_snapshot(self.root),
            "config": _config_snapshot(self.root),
        }

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_make_collect_report(self, collector):
        previous = self.context
        owner = (
            collector
            if isinstance(collector, pytest.Module)
            else collector.getparent(pytest.Module)
        )
        if owner is not None:
            module = owner.nodeid.split("::", 1)[0]
            self.collected_modules.add(module)
            self._switch(_MODULE_CONTEXT_PREFIX + module)
        try:
            return (yield)
        finally:
            self._switch(previous)

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(
        self, config: pytest.Config, items: list[pytest.Item]
    ):
        yield
        selected = [item.nodeid for item in items]
        after_collection = {
            "runtime": _runtime_snapshot(self.root),
            "config": _config_snapshot(self.root),
        }
        if after_collection != self.collection_snapshot:
            details = self._source_drift(
                "collection", self.collection_snapshot, after_collection
            )
            raise pytest.UsageError(
                "CI sources changed during collection; recollect before execution. "
                + details
            )
        if len(selected) != len(set(selected)):
            raise pytest.UsageError(
                "CI sharding requires unique selected test node IDs"
            )
        counts = Counter(nodeid.split("::", 1)[0] for nodeid in selected)
        missing = set(self.recovery_modules).difference(counts)
        if missing:
            raise pytest.UsageError(
                "CI recovery modules must be globally selected complete modules: "
                + ", ".join(sorted(missing))
            )
        weights = _module_weights(counts, self.profile)
        loads = [0] * self.total
        assignments = {}
        for module, weight in sorted(
            weights.items(), key=lambda pair: (-pair[1], pair[0])
        ):
            shard = min(range(self.total), key=lambda value: (loads[value], value))
            assignments[module] = shard + 1
            loads[shard] += weight
        assigned, deselected = [], []
        for item in items:
            module = item.nodeid.split("::", 1)[0]
            owns = (
                module in self.recovery_modules
                if self.recovery_modules
                else assignments[module] == self.index
            )
            (assigned if owns else deselected).append(item)
        items[:] = assigned
        config.hook.pytest_deselected(items=deselected)
        assigned_modules = {item.nodeid.split("::", 1)[0] for item in assigned}
        self.manifest = {
            "schema_version": 1,
            "index": self.index,
            "total": self.total,
            "selected_nodeids": selected,
            "assigned_nodeids": [item.nodeid for item in assigned],
            "module_assignments": assignments,
            "estimated_loads": loads,
            "collection_duration_seconds": time.perf_counter() - self.started,
            "test_duration_seconds": 0.0,
            "executed_nodeids": [],
            "test_durations_seconds": {},
            "exitstatus": None,
            "collect_only": bool(config.option.collectonly),
            "recovery_modules": self.recovery_modules,
            "test_outcomes": {
                item.nodeid: {
                    "outcome": "incomplete",
                    "phases": {},
                    "reason": None,
                    "wasxfail": None,
                }
                for item in assigned
            },
            "source_snapshot": {
                "modules": _module_snapshot(self.root, assigned_modules),
                "runtime": _runtime_snapshot(self.root),
                "config": _config_snapshot(self.root, counts),
                "unchanged": False,
            },
            "coverage_contexts": {
                "mode": "module",
                "active": self.coverage_active,
                "global": self.context_names.get(_GLOBAL_CONTEXT, _GLOBAL_CONTEXT),
                "modules": {
                    module: self.context_names.get(
                        _MODULE_CONTEXT_PREFIX + module, _MODULE_CONTEXT_PREFIX + module
                    )
                    for module in sorted(assigned_modules)
                },
                "owned_only": False,
            },
        }
        if self.profile is not None:
            self.manifest["duration_profile"] = self.profile[2]
            self.manifest["estimated_load_units"] = "seconds"
        self._write_manifest()

    def pytest_collection_finish(self) -> None:
        if self.manifest is not None:
            self.manifest["collection_duration_seconds"] = (
                time.perf_counter() - self.started
            )

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_runtest_protocol(self, item, nextitem):
        if not any(name == "ci_nodeid" for name, _value in item.user_properties):
            item.user_properties.append(("ci_nodeid", item.nodeid))
        previous = self.context
        self._switch(_MODULE_CONTEXT_PREFIX + item.nodeid.split("::", 1)[0])
        try:
            return (yield)
        finally:
            self._switch(previous)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self.durations[report.nodeid] = (
            self.durations.get(report.nodeid, 0.0) + report.duration
        )
        if self.manifest is None or report.nodeid not in self.manifest["test_outcomes"]:
            return
        record = self.manifest["test_outcomes"][report.nodeid]
        # Subtests emit additional call reports. Their failures remain sticky;
        # a skipped subtest does not turn a passing containing test into a skip.
        if hasattr(report, "context"):
            if report.failed:
                record["outcome"] = "failed"
                record["reason"] = _report_reason(report)
            return
        record["phases"][report.when] = report.outcome
        wasxfail = getattr(report, "wasxfail", None)
        if wasxfail:
            record["wasxfail"] = str(wasxfail)
        if report.failed:
            record["outcome"] = "failed"
            record["reason"] = _report_reason(report)
        elif record["outcome"] != "failed":
            if report.skipped:
                record["outcome"] = "xfailed" if wasxfail else "skipped"
                record["reason"] = str(wasxfail) if wasxfail else _report_reason(report)
            elif report.when == "call":
                record["outcome"] = "xpassed" if wasxfail else "incomplete"
            elif report.when == "teardown" and record["outcome"] == "incomplete":
                if (
                    record["phases"].get("setup")
                    == record["phases"].get("call")
                    == "passed"
                ):
                    record["outcome"] = "passed"

    def _filter_coverage(self, session):
        metadata = self.manifest["coverage_contexts"]
        if not self.coverage_active:
            metadata["owned_only"] = True
            return
        plugin = session.config.pluginmanager.getplugin("_cov")
        controller = getattr(plugin, "cov_controller", None)
        coverage = getattr(controller, "cov", None)
        if coverage is None:
            raise pytest.UsageError(
                "CI owned coverage requires pytest-cov-managed coverage"
            )
        module = sys.modules["coverage"]
        data = coverage.get_data()
        filtered = module.CoverageData(no_disk=True)
        if data.has_arcs():
            filtered.add_arcs({})
        else:
            filtered.add_lines({})
        global_context = metadata["global"]
        owned = set(metadata["modules"].values()) | {global_context}
        unselected = self.collected_modules.difference(
            self.manifest["module_assignments"]
        )
        unselected_contexts = {
            self.context_names.get(
                _MODULE_CONTEXT_PREFIX + name, _MODULE_CONTEXT_PREFIX + name
            )
            for name in unselected
        }
        foreign = {
            self.context_names.get(
                _MODULE_CONTEXT_PREFIX + name, _MODULE_CONTEXT_PREFIX + name
            )
            for name in self.manifest["module_assignments"]
        }.difference(owned)
        for context in sorted(data.measured_contexts()):
            if context in owned:
                _copy_coverage_context(data, filtered, context, context)
            elif context in unselected_contexts:
                _copy_coverage_context(data, filtered, context, global_context)
            elif context not in foreign:
                data.set_query_contexts([r"^" + re.escape(context) + r"$"])
                measured = any(
                    data.arcs(name) if data.has_arcs() else data.lines(name)
                    for name in data.measured_files()
                )
                if measured:
                    raise pytest.UsageError(
                        f"CI coverage contains unknown/contextless measurements: {context!r}"
                    )
        data.set_query_contexts(None)
        filtered.touch_files(data.measured_files())
        tracers = {
            filename: data.file_tracer(filename)
            for filename in data.measured_files()
            if data.file_tracer(filename)
        }
        if tracers:
            filtered.add_file_tracers(tracers)
        # CoverageData.update uses SQLite ATTACH: an in-memory source is not
        # attachable, and replacing a previously used destination loses its mode.
        # Reinsert each owned context through the public data API instead.
        coverage.erase()
        data = coverage.get_data()
        (data.add_arcs if filtered.has_arcs() else data.add_lines)({})
        for context in sorted(filtered.measured_contexts()):
            _copy_coverage_context(filtered, data, context, context)
        data.touch_files(filtered.measured_files())
        if tracers:
            data.add_file_tracers(tracers)
        coverage.save()
        metadata.update(
            owned_only=True, data_file=str(Path(coverage.config.data_file).resolve())
        )
        # pytest-cov reports before sessionfinish. Regenerate its configured
        # reports from owned data so XML/JSON and its terminal summary agree.
        if plugin.cov_total is not None:
            plugin.cov_report.seek(0)
            plugin.cov_report.truncate()
            plugin.cov_total = controller.summary(plugin.cov_report)

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        if self.manifest is None:
            return
        if (
            exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED
            and self.manifest["selected_nodeids"]
            and not self.manifest["assigned_nodeids"]
        ):
            exitstatus = pytest.ExitCode.OK
            session.exitstatus = exitstatus
        snapshot = self.manifest["source_snapshot"]
        try:
            self._filter_coverage(session)
            final = {
                "modules": _module_snapshot(self.root, snapshot["modules"]),
                "runtime": _runtime_snapshot(self.root),
                "config": _config_snapshot(
                    self.root, self.manifest["module_assignments"]
                ),
            }
            snapshot["final"] = final
            snapshot["unchanged"] = all(
                snapshot[key] == value for key, value in final.items()
            )
            if not snapshot["unchanged"]:
                before = {
                    key: snapshot[key] for key in ("modules", "runtime", "config")
                }
                details = self._source_drift("execution", before, final)
                snapshot["error"] = "Sources changed during execution. " + details
                print(snapshot["error"], file=sys.stderr)
                session.exitstatus = exitstatus = pytest.ExitCode.USAGE_ERROR
        except (OSError, ValueError, pytest.UsageError) as exc:
            snapshot["error"] = str(exc)
            session.exitstatus = exitstatus = pytest.ExitCode.USAGE_ERROR
        self.manifest.update(
            exitstatus=int(exitstatus),
            executed_nodeids=[
                nodeid
                for nodeid in self.manifest["assigned_nodeids"]
                if nodeid in self.durations
            ],
            test_duration_seconds=sum(self.durations.values()),
            test_durations_seconds=self.durations,
        )
        self._write_manifest()

    def _write_manifest(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / "manifest.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
