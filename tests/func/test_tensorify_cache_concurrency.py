"""Real Inductor artifacts and concurrent owned compilation keep scalar guards."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
import torch

from dendra._bootstrap import compile_owned
from dendra._tensorify_compat import (
    _compatibility_status,
    install_tensorify_compatibility,
)

pytestmark = pytest.mark.cpu
_USER_CACHE_TAG = "dendra-test-user-prefix"
_EXPECTED_CACHE_MARKER = "dendra.tensorify-eq-v1"


def _tensor_math(value):
    return value.sin() * value


def _python_float_factory(value, amplitude):
    return value.sin() + torch.full(
        (2,), amplitude, dtype=value.dtype, device=value.device
    )


def _item_factory(value):
    return torch.full((2,), value.item(), dtype=value.dtype, device=value.device)


def _mixed_factory(value, scalar_source):
    return value.sin() + _item_factory(scalar_source)


def _cosine_math(value):
    return value.cos() * value


def _thread_moved_math(value):
    return value.square() + value.cos()


def _norm_pure(value, scalar):
    return torch.nn.functional.rms_norm(value, (4,), eps=scalar + 0.001)


def _norm_mixed(value, scalar):
    return torch.nn.functional.rms_norm(value * scalar, (4,), eps=scalar + 0.001)


def _huber_pure(value, scalar):
    return torch.nn.functional.huber_loss(
        value, torch.zeros_like(value), delta=scalar + 0.1
    )


def _huber_mixed(value, scalar):
    return torch.nn.functional.huber_loss(
        value * scalar, torch.zeros_like(value), delta=scalar + 0.1
    )


def _resize_pure(value, scalar):
    return torch.nn.functional.interpolate(value[None], scale_factor=1.0 + scalar)


def _resize_mixed(value, scalar):
    return torch.nn.functional.interpolate(
        (value * scalar)[None], scale_factor=1.0 + scalar
    )


def _counters():
    from torch._dynamo.utils import counters

    return {
        "aot_hit": counters["aot_autograd"]["autograd_cache_hit"],
        "aot_saved": counters["aot_autograd"]["autograd_cache_saved"],
        "aot_miss": counters["aot_autograd"]["autograd_cache_miss"],
        "aot_bypass": counters["aot_autograd"]["autograd_cache_bypass"],
        "aot_guard_miss": counters["aot_autograd"]["autograd_cache_guard_miss"],
        "fx_hit": counters["inductor"]["fxgraph_cache_hit"],
        "fx_miss": counters["inductor"]["fxgraph_cache_miss"],
    }


def _delta(before, after):
    return {name: after[name] - before[name] for name in before}


def _checked_call(function, compiled, scalar, *, rank_one=False, python_float=False):
    values = [scalar, scalar + 0.25] if rank_one else scalar
    value = torch.tensor(
        values, dtype=torch.float64, requires_grad=function is not _item_factory
    )
    if python_float:
        value = torch.tensor([0.4, -0.2], dtype=torch.float64, requires_grad=True)
    original, version = value.detach().clone(), value._version
    reference = original.clone().requires_grad_(value.requires_grad)
    arguments = (value, scalar) if python_float else (value,)
    reference_arguments = (reference, scalar) if python_float else (reference,)
    if function is _mixed_factory:
        # A separate factory source has only item() users. Mixing tensor math
        # and item() on the same source has an independent stock-Torch cache
        # guard defect when capture_scalar_outputs is explicitly enabled.
        scalar_source = torch.tensor(scalar, dtype=torch.float64)
        source_original = scalar_source.clone()
        source_version = scalar_source._version
        arguments = (value, scalar_source)
        reference_arguments = (reference, source_original)
    expected = function(*reference_arguments)
    actual = compiled(*arguments)
    torch.testing.assert_close(
        actual,
        expected,
        msg=lambda message: f"{function.__name__}({scalar}): {message}",
    )
    if value.requires_grad:
        expected_gradient = torch.autograd.grad(expected.sum(), reference)[0]
        gradient = torch.autograd.grad(actual.sum(), value)[0]
        torch.testing.assert_close(gradient, expected_gradient)
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    torch.testing.assert_close(value, original, rtol=0, atol=0)
    assert value._version == version
    if function is _mixed_factory:
        torch.testing.assert_close(scalar_source, source_original, rtol=0, atol=0)
        assert scalar_source._version == source_version
    return actual.detach().tolist()


def _compile(function, *, owned=True, automatic_dynamic=False):
    compiler = compile_owned if owned else torch.compile
    keywords = {} if automatic_dynamic else {"dynamic": False}
    return compiler(
        function,
        backend="inductor",
        fullgraph=True,
        options={"compile_threads": 1, "cpp_wrapper": False},
        **keywords,
    )


def _cache_worker(role):
    # No reset or clear: this process reads exactly the artifacts left by its
    # predecessor in the shared cache. Input values and gradients remain live.
    reader = role in ("read", "stock-read")
    stock_reader = role == "stock-read"
    sequence = (2.7, 1.3, 3.4) if reader else (1.3, 2.7, 1.3)
    initial_identity = _implementation_identity()
    initial_status = _compatibility_status()
    if role == "stock-write":
        assert initial_status == "uninitialized"
    records = []
    cases = (
        ("scalar_tensor", _tensor_math, False, False),
        ("rank_one_tensor", _tensor_math, True, False),
        ("python_float_factory", _python_float_factory, True, True),
        ("tensor_item_factory", _item_factory, False, False),
        ("mixed_tensor_item_factory", _mixed_factory, False, False),
    )
    for name, function, rank_one, python_float in cases:
        before = _counters()
        compiled = _compile(function, owned=role != "stock-write")
        outputs = [
            _checked_call(
                function, compiled, value, rank_one=rank_one, python_float=python_float
            )
            for value in sequence
        ]
        changes = _delta(before, _counters())
        assert changes["aot_bypass"] == 0, (name, changes)
        # A pure item factory has no differentiable input dependency. Keep its
        # real FX-cache control without requiring a deferred backward artifact.
        aot_eligible = function is not _item_factory
        if not reader:
            counter = "aot_saved" if aot_eligible else "fx_miss"
            assert changes[counter] > 0, (name, changes)
        elif stock_reader and _compatibility_status() == "installed":
            # Repaired compilers use a distinct namespace, including when the
            # old artifact was numerically safe for this particular graph.
            counter = "aot_miss" if aot_eligible else "fx_miss"
            assert changes[counter] > 0, (name, changes)
        else:
            # AOT hits count only after its guards and nested FX artifact load
            # succeed. FXGraphCache.load_with_key increments fxgraph_cache_hit.
            if aot_eligible:
                assert changes["aot_hit"] > 0, (name, changes)
            assert changes["fx_hit"] > 0, (name, changes)
        records.append(
            {
                "case": name,
                "aot_eligible": aot_eligible,
                "counter_changes": changes,
                "outputs": outputs,
            }
        )
    if role == "stock-write":
        assert _implementation_identity() == initial_identity
        assert _compatibility_status() == initial_status
    return {
        "qualified": True,
        "role": role,
        "initial_compatibility_status": initial_status,
        "final_compatibility_status": _compatibility_status(),
        "cases": records,
        "cache_reused": reader and not stock_reader,
        "stock_artifact_migration": stock_reader,
    }


def _derived_checked_call(function, compiled, scalar):
    value = (torch.arange(8, dtype=torch.float64).reshape(2, 4) / 9).requires_grad_()
    original, version = value.detach().clone(), value._version
    reference = original.clone().requires_grad_()
    expected = function(reference, scalar)
    actual = compiled(value, scalar)
    torch.testing.assert_close(actual, expected)
    expected_gradient = torch.autograd.grad(expected.sum(), reference)[0]
    gradient = torch.autograd.grad(actual.sum(), value)[0]
    torch.testing.assert_close(gradient, expected_gradient)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
    torch.testing.assert_close(value, original, rtol=0, atol=0)
    assert value._version == version
    return {
        "scalar": scalar,
        "shape": list(actual.shape),
        "output": actual.detach().tolist(),
        "gradient": gradient.detach().tolist(),
    }


def _derived_cache_worker(role):
    # Leave automatic dynamic-float policy at Torch's default. A derived
    # schema-only float must stay guarded even when another use was tensorified.
    stock = role == "derived-stock-write"
    migration = role == "derived-migration-read"
    reader = role in ("derived-read", "derived-migration-read")
    sequence = (0.25, 0.9, 2.0, 0.5) if reader else (0.9, 0.5, 0.25, 2.0)
    if stock:
        # Seed artifacts that stock Torch builds correctly at the first two
        # values; its later dynamic-float reuse is the migration hazard.
        sequence = (0.9, 0.5)
    initial_identity = _implementation_identity()
    initial_status = _compatibility_status()
    if stock:
        assert initial_status == "uninitialized"
    cases = (
        _norm_pure,
        _norm_mixed,
        _huber_pure,
        _huber_mixed,
        _resize_pure,
        _resize_mixed,
    )
    if stock or migration:
        cases = (_norm_mixed, _huber_mixed, _resize_mixed)
    records = []
    for function in cases:
        before = _counters()
        compiled = _compile(function, owned=not stock, automatic_dynamic=True)
        outputs = [
            _derived_checked_call(function, compiled, scalar) for scalar in sequence
        ]
        changes = _delta(before, _counters())
        assert changes["aot_bypass"] == 0, (function.__name__, changes)
        if not reader:
            assert changes["aot_saved"] > 0, (function.__name__, changes)
        elif migration:
            # For the recognized repaired body, corrected specialization must
            # not load the old graph for unseen values. Future stock passes
            # can reuse compatible artifacts when all value oracles hold.
            if _compatibility_status() == "installed":
                assert changes["aot_miss"] > 0, (function.__name__, changes)
        else:
            assert changes["aot_hit"] > 0, (function.__name__, changes)
            assert changes["fx_hit"] > 0, (function.__name__, changes)
        records.append(
            {"case": function.__name__, "counter_changes": changes, "outputs": outputs}
        )
    if stock:
        assert _implementation_identity() == initial_identity
        assert _compatibility_status() == initial_status
    return {
        "qualified": True,
        "role": role,
        "initial_compatibility_status": initial_status,
        "final_compatibility_status": _compatibility_status(),
        "cases": records,
        "cache_reused": reader and not migration,
        "stock_artifact_migration": migration,
        "automatic_dynamic_policy": True,
    }


def _implementation_identity():
    try:
        module = importlib.import_module("torch.fx.passes._tensorify_python_scalars")
    except ImportError:
        return None
    function = getattr(module, "_tensorify_impl", None)
    return None if function is None else id(function)


def _effective_cache_tag():
    from torch._inductor.codecache import FxGraphHashDetails

    return FxGraphHashDetails(None, [], {}, []).cache_key_tag


def _assert_owned_cache_tag(prefix=_USER_CACHE_TAG):
    import torch.compiler.config as compiler_config

    assert compiler_config.cache_key_tag == prefix
    expected = prefix
    if _compatibility_status() == "installed":
        expected = (_EXPECTED_CACHE_MARKER, prefix)
    assert _effective_cache_tag() == expected
    assert compiler_config.cache_key_tag == prefix
    return expected


def _concurrency_worker():
    initial_identity = _implementation_identity()
    initial_status = _compatibility_status()
    constructors, executions = Barrier(3), Barrier(3)
    cases = (
        (_tensor_math, False, False),
        (_cosine_math, True, False),
        (_python_float_factory, True, True),
    )

    def first(index):
        function, rank_one, python_float = cases[index]
        constructors.wait(timeout=30)
        compiled = _compile(function)
        identity = _implementation_identity()
        cache_tag = _assert_owned_cache_tag()
        executions.wait(timeout=30)
        outputs = [
            _checked_call(
                function, compiled, scalar, rank_one=rank_one, python_float=python_float
            )
            for scalar in (1.3, 2.7, 1.3)
        ]
        return compiled, {
            "index": index,
            "implementation_identity": identity,
            "cache_tag": cache_tag,
            "outputs": outputs,
        }

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(first, index) for index in range(3)]
        completed = [future.result(timeout=180) for future in futures]
    selected_identity = _implementation_identity()
    assert all(
        row["implementation_identity"] == selected_identity
        for _compiled, row in completed
    )
    installed = install_tensorify_compatibility()
    assert _implementation_identity() == selected_identity
    repeats = Barrier(3)

    def repeat(index):
        compiled, _record = completed[index]
        function, rank_one, python_float = cases[index]
        repeats.wait(timeout=30)
        repeated = install_tensorify_compatibility()
        assert repeated == installed
        assert _implementation_identity() == selected_identity
        cache_tag = _assert_owned_cache_tag()
        output = _checked_call(
            function, compiled, 3.4, rank_one=rank_one, python_float=python_float
        )
        return {
            "index": index,
            "installed": repeated,
            "cache_tag": cache_tag,
            "output": output,
        }

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(repeat, index) for index in range(3)]
        repeated = [future.result(timeout=180) for future in futures]
    assert _implementation_identity() == selected_identity

    # Construction and actual backend execution can occur in different
    # contexts. The global hash-constructor hook must use each execution
    # context's user prefix even though compile_owned was called in main.
    moved = _compile(_thread_moved_math)
    transferred = Barrier(3)
    before_transfer = _counters()

    def execute_transferred(index):
        import torch.compiler.config as compiler_config

        prefix = _USER_CACHE_TAG + "-execution-" + str(index)
        with compiler_config.patch(cache_key_tag=prefix):
            transferred.wait(timeout=30)
            output = _checked_call(_thread_moved_math, moved, 0.8 + index)
            cache_tag = _assert_owned_cache_tag(prefix)
            assert _implementation_identity() == selected_identity
        assert compiler_config.cache_key_tag == _USER_CACHE_TAG
        return {
            "index": index,
            "user_prefix": prefix,
            "cache_tag": cache_tag,
            "output": output,
        }

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(execute_transferred, index) for index in range(3)]
        transferred_results = [future.result(timeout=180) for future in futures]
    transfer_changes = _delta(before_transfer, _counters())
    assert transfer_changes["aot_miss"] > 0
    _assert_owned_cache_tag()
    return {
        "qualified": True,
        "workers": 3,
        "initial_identity": initial_identity,
        "initial_status": initial_status,
        "selected_identity": selected_identity,
        "final_status": _compatibility_status(),
        "installed": installed,
        "first": [row for _compiled, row in completed],
        "repeated": repeated,
        "main_constructed_worker_executed": transferred_results,
        "transfer_counter_changes": transfer_changes,
        "global_reset_during_threads": False,
    }


def _worker_main():
    import torch._dynamo.config as dynamo_config
    import torch._functorch.config as autograd_config
    import torch._inductor.config as inductor_config

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "role",
        choices=(
            "write",
            "stock-write",
            "read",
            "stock-read",
            "threads",
            "derived-write",
            "derived-read",
            "derived-stock-write",
            "derived-migration-read",
        ),
    )
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    initial_rng = torch.get_rng_state().clone()
    import torch.compiler.config as compiler_config

    initial_tag = compiler_config.cache_key_tag
    assert initial_tag == _USER_CACHE_TAG
    with (
        dynamo_config.patch(capture_scalar_outputs=True),
        autograd_config.patch(
            enable_autograd_cache=True,
            enable_remote_autograd_cache=False,
            strict_autograd_cache=True,
        ),
        inductor_config.patch(
            fx_graph_cache=True,
            fx_graph_remote_cache=False,
            force_disable_caches=False,
            compile_threads=1,
        ),
    ):
        if arguments.role == "threads":
            result = _concurrency_worker()
        elif arguments.role.startswith("derived-"):
            result = _derived_cache_worker(arguments.role)
        else:
            result = _cache_worker(arguments.role)
    torch.testing.assert_close(torch.get_rng_state(), initial_rng, rtol=0, atol=0)
    result["global_rng_unchanged"] = True
    result["initial_cache_tag"] = initial_tag
    result["final_cache_tag"] = compiler_config.cache_key_tag
    assert result["final_cache_tag"] == initial_tag
    result["effective_hash_tag"] = _assert_owned_cache_tag(initial_tag)
    result["expected_cache_marker"] = _EXPECTED_CACHE_MARKER
    status = _compatibility_status()
    result["final_compatibility_status"] = status
    if arguments.role in ("stock-write", "derived-stock-write"):
        assert status == "uninitialized"
        assert result["effective_hash_tag"] == initial_tag
    result["cache_dir"] = os.environ["TORCHINDUCTOR_CACHE_DIR"]
    cache = Path(result["cache_dir"])
    all_paths = {
        str(path.relative_to(cache)) for path in cache.rglob("*") if path.is_file()
    }
    artifacts = _cache_file_hashes(cache)
    result["cache_files"] = len(all_paths)
    result["compiled_artifact_files"] = len(artifacts)
    result["cache_byte_invariant_excluded_paths"] = sorted(all_paths - artifacts.keys())
    result["pgo_bookkeeping_paths"] = sorted(
        name for name in all_paths if Path(name).parts[0] == "dynamo"
    )
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")


def _worker(tmp_path, role, cache):
    output = tmp_path / f"{role}.json"
    environment = os.environ.copy()
    root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = root + os.pathsep + environment.get("PYTHONPATH", "")
    environment.update(
        TORCHINDUCTOR_CACHE_DIR=str(cache),
        TORCH_COMPILE_CACHE_KEY_TAG=_USER_CACHE_TAG,
        TORCHINDUCTOR_COMPILE_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    process = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), role, str(output)],
        env=environment,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads(output.read_text())
    assert result["qualified"] and result["global_rng_unchanged"]
    return result


def _cache_file_hashes(cache):
    # PGO and compiler bookkeeping can legitimately change as the reader
    # learns new values. Preserve the actual persisted AOT/FX artifacts.
    return {
        str(path.relative_to(cache)): hashlib.sha256(path.read_bytes()).hexdigest()
        for namespace in ("aotautograd", "fxgraph")
        for path in (cache / namespace).rglob("*")
        if path.is_file()
    }


def _assert_cached_files_unchanged(cache, before):
    assert {Path(name).parts[0] for name in before} == {"aotautograd", "fxgraph"}
    after = _cache_file_hashes(cache)
    assert all(after.get(name) == digest for name, digest in before.items())


@pytest.mark.parametrize(
    "writer", ("write", "stock-write"), ids=("owned-writer", "stock-writer")
)
def test_owned_inductor_persistent_aot_artifacts_follow_compiler_cache_policy(
    tmp_path,
    writer,
):
    cache = tmp_path / "persistent-cache"
    written = _worker(tmp_path, writer, cache)
    assert written["cache_files"] > 0
    cache_before = _cache_file_hashes(cache)
    loaded = _worker(tmp_path, "read" if writer == "write" else "stock-read", cache)
    assert loaded["cache_reused"] == (writer == "write")
    assert written["cache_dir"] == loaded["cache_dir"]
    _assert_cached_files_unchanged(cache, cache_before)
    if writer == "write":
        assert written["effective_hash_tag"] == loaded["effective_hash_tag"]
    elif loaded["final_compatibility_status"] == "installed":
        assert loaded["effective_hash_tag"] == [
            _EXPECTED_CACHE_MARKER,
            written["effective_hash_tag"],
        ]
    assert len(written["cases"]) == len(loaded["cases"]) == 5
    for before, after in zip(written["cases"], loaded["cases"], strict=True):
        assert before["case"] == after["case"]
        assert before["aot_eligible"] == after["aot_eligible"]
        if (
            writer == "stock-write"
            and loaded["final_compatibility_status"] == "installed"
        ):
            counter = "aot_miss" if after["aot_eligible"] else "fx_miss"
            assert after["counter_changes"][counter] > 0
        else:
            if after["aot_eligible"]:
                assert after["counter_changes"]["aot_hit"] > 0
            assert after["counter_changes"]["fx_hit"] > 0
        assert after["outputs"][0] == before["outputs"][1]
        assert after["outputs"][1] == before["outputs"][0]


@pytest.fixture
def _derived_cache_repair():
    # These two workflows assert repaired scalar specialization rather than
    # stock behavior. Generic cache and fallback tests remain unconditional.
    if not install_tensorify_compatibility():
        pytest.skip(
            "Installed Torch compiler is outside the recognized repair contract"
        )


def test_owned_inductor_reuses_derived_scalar_artifacts_with_automatic_dynamic_floats(
    tmp_path,
    _derived_cache_repair,
):
    cache = tmp_path / "derived-cache"
    written = _worker(tmp_path, "derived-write", cache)
    cache_before = _cache_file_hashes(cache)
    loaded = _worker(tmp_path, "derived-read", cache)
    _assert_cached_files_unchanged(cache, cache_before)
    assert written["effective_hash_tag"] == loaded["effective_hash_tag"]
    assert written["cache_files"] > 0 and loaded["cache_reused"]
    assert written["automatic_dynamic_policy"] and loaded["automatic_dynamic_policy"]
    assert len(written["cases"]) == len(loaded["cases"]) == 6
    for before, after in zip(written["cases"], loaded["cases"], strict=True):
        assert before["case"] == after["case"]
        expected = {row["scalar"]: row for row in before["outputs"]}
        assert len(expected) == len(after["outputs"]) == 4
        assert after["outputs"] == [expected[row["scalar"]] for row in after["outputs"]]


def test_owned_inductor_avoids_incompatible_stock_derived_scalar_artifacts(
    tmp_path, _derived_cache_repair
):
    cache = tmp_path / "stock-derived-cache"
    written = _worker(tmp_path, "derived-stock-write", cache)
    cache_before = _cache_file_hashes(cache)
    loaded = _worker(tmp_path, "derived-migration-read", cache)
    _assert_cached_files_unchanged(cache, cache_before)
    if loaded["final_compatibility_status"] == "installed":
        assert loaded["effective_hash_tag"] == [
            _EXPECTED_CACHE_MARKER,
            written["effective_hash_tag"],
        ]
    assert written["cache_files"] > 0 and loaded["stock_artifact_migration"]
    assert len(written["cases"]) == len(loaded["cases"]) == 3
    for before, after in zip(written["cases"], loaded["cases"], strict=True):
        assert before["case"] == after["case"]
        expected = {row["scalar"]: row for row in before["outputs"]}
        assert len(expected) == 2 and len(after["outputs"]) == 4
        if loaded["final_compatibility_status"] == "installed":
            assert after["counter_changes"]["aot_miss"] > 0
        for row in after["outputs"]:
            if row["scalar"] in expected:
                assert row == expected[row["scalar"]]


def test_owned_inductor_concurrent_install_compile_and_backward_are_idempotent(
    tmp_path,
):
    result = _worker(tmp_path, "threads", tmp_path / "concurrent-cache")
    assert result["workers"] == 3
    assert len(result["first"]) == len(result["repeated"]) == 3
    assert len(result["main_constructed_worker_executed"]) == 3
    assert result["transfer_counter_changes"]["aot_miss"] > 0
    assert result["global_reset_during_threads"] is False


if __name__ == "__main__":
    _worker_main()
