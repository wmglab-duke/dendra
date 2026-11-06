"""Compiler compatibility CI keeps one complete selection for every runtime."""

from pathlib import Path

ROOT = Path(__file__).parents[1]
SELECTION = "scripts/functional_compatibility_tests.txt"
SPECS = ("torch==2.14.0", "torch==2.15.0", "torch>=2.15,<3")


def test_compiler_selection_is_shared_and_preserves_the_serial_alias_check():
    selections = (ROOT / SELECTION).read_text().splitlines()
    assert len(selections) == len(set(selections)) == 29
    assert all(
        value.startswith("tests/func/") and ".py" in value for value in selections
    )
    assert selections[-1] == (
        "tests/func/test_convenience_runner.py::"
        "test_top_level_compile_is_the_exported_functional_compile_alias"
    )
    for name in (".gitlab-ci.yml", ".github/workflows/publish.yml"):
        text = (ROOT / name).read_text()
        assert "@" + SELECTION in text
        # Native and compiler assertions remain in the selected test files.
        assert "tests/func/test_scan_chunk.py" not in text
        assert "tests/func/test_material_processes.py" not in text


def test_github_compiler_gate_requires_all_four_shards_for_each_torch():
    text = (ROOT / ".github/workflows/publish.yml").read_text()
    shards = text.split("  test_functional_cpu_shards:\n", 1)[1].split(
        "\n  test_functional_cpu:\n", 1
    )[0]
    gate = text.split("  test_functional_cpu:\n", 1)[1].split("\n  check_dist:\n", 1)[0]
    check = text.split("  check_dist:\n", 1)[1].split("\n  publish_testpypi:\n", 1)[0]
    assert "shard: [1, 2, 3, 4]" in shards
    assert "timeout-minutes: 90" in shards
    assert "python -m pytest -q -p scripts.ci_shard" in shards
    assert "--ci-shard-total=4" in shards
    assert "--ci-shard-durations=scripts/ci_durations.json" in shards
    assert "needs: test_functional_cpu_shards" in gate
    assert "needs: [build, test_install, test_functional_cpu]" in check
    assert "functional-compatibility-${{ matrix.torch.label }}-*" in gate
    assert "merge-multiple: true" in gate
    assert "--partition-only --shards=4" in gate
    assert '--torch-spec="$TORCH_SPEC"' in gate
    for block in (shards, gate):
        assert all(spec in block for spec in SPECS)
        assert "continue-on-error" not in block
        assert "--cov" not in block
    assert "if: always()" not in gate


def test_gitlab_compiler_partition_gate_leaves_full_coverage_required():
    text = (ROOT / ".gitlab-ci.yml").read_text()
    shards = text.split("test_functional_compatibility_shards:\n", 1)[1].split(
        "\ntest_functional_compatibility:\n", 1
    )[0]
    gate = text.split("\ntest_functional_compatibility:\n", 1)[1].split(
        "\nbuild_docs:\n", 1
    )[0]
    coverage = text.split("\ntest_coverage:\n", 1)[1].split(
        "\ntest_functional_compatibility_shards:\n", 1
    )[0]
    release = text.split("\nrelease_bump:\n", 1)[1].split("\npages:\n", 1)[0]
    assert 'COMPAT_SHARD: ["1", "2", "3", "4"]' in shards
    assert "timeout: 90 minutes" in shards
    assert "python -m pytest -q -p scripts.ci_shard" in shards
    assert "--ci-shard-total=4" in shards
    assert "--ci-shard-durations=scripts/ci_durations.json" in shards
    assert "job: test_functional_compatibility_shards\n      artifacts: true" in gate
    assert "--partition-only --shards=4" in gate
    assert '--input-dir="ci-results/functional-compatibility/$TORCH_LABEL"' in gate
    assert '--torch-spec="$TORCH_SPEC"' in gate
    assert "job: test_functional_compatibility\n      artifacts: false" in release
    assert "--shards 40 --input-dir ci-results --fail-under=84" in coverage
    assert "--partition-only" not in coverage
    for block in (shards, gate):
        assert all(spec in block for spec in SPECS)
        assert "extends: .full_test_rules" in block
        assert "allow_failure:" not in block
        assert "when: always" not in gate
        assert "--cov" not in block


def test_github_keeps_base_mac_install_and_adds_published_solver_mac_install():
    text = (ROOT / ".github/workflows/publish.yml").read_text()
    installs = text.split("  test_install:\n", 1)[1].split(
        "\n  test_functional_cpu_shards:\n", 1
    )[0]
    assert (
        'os: macos-15\n            python: "3.12"\n            torch: "torch==2.14.0"\n            distribution: wheel\n            solvers: absent'
        in installs
    )
    assert (
        'os: macos-15\n            python: "3.12"\n            torch: "torch==2.15.0"\n            distribution: wheel\n            solvers: present'
        in installs
    )
