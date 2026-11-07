"""Collect real pytest cases to verify required and optional backend lanes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_BACKENDS = {"cpu", "neuron", "cuda", "mps"}


@pytest.fixture
def classification_suite(tmp_path: Path) -> Path:
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\naddopts = --strict-markers\nmarkers =\n"
        "    cpu\n    neuron\n    cuda\n    mps\n    slow\n    stochastic\n",
        encoding="utf-8",
    )
    classifier = (_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    (tmp_path / "conftest.py").write_text(
        classifier
        + "\nimport json\n"
        + "\ndef pytest_collection_finish(session):\n"
        + "    inventory = {item.name: sorted({mark.name for mark in "
        + "item.iter_markers()} & {'cpu', 'neuron', 'cuda', 'mps'}) "
        + "for item in session.items}\n"
        + "    (session.config.rootpath / 'inventory.json').write_text("
        + "json.dumps(inventory), encoding='utf-8')\n",
        encoding="utf-8",
    )
    (tmp_path / "test_devices.py").write_text(
        "import pytest\n"
        "\n@pytest.mark.cpu\n"
        "@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])\n"
        "def test_metadata(device):\n    pass\n"
        "\n@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])\n"
        "def test_hardware(device):\n    pass\n"
        "\n@pytest.mark.cuda\n"
        "def test_explicit_accelerator():\n    pass\n"
        "\n@pytest.mark.skipif(False, reason='CUDA is required')\n"
        "def test_cuda_guard():\n    pass\n"
        "\n@pytest.mark.skipif(False, reason='GPU is required')\n"
        "def test_gpu_guard():\n    pass\n"
        "\n@pytest.mark.neuron\n"
        "@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])\n"
        "def test_simulator(device):\n    pass\n"
        "\ndef test_default():\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "test_mixed_devices.py").write_text(
        "import pytest\n"
        "pytestmark = pytest.mark.cpu\n"
        "\n@pytest.mark.parametrize('device', ['cuda:1'])\n"
        "def test_inherited_metadata(device):\n    pass\n"
        "\n@pytest.mark.cuda\n"
        "def test_inherited_explicit_accelerator():\n    pass\n"
        "\n@pytest.mark.skipif(False, reason='CUDA is required')\n"
        "def test_inherited_cuda_guard():\n    pass\n"
        "\n@pytest.mark.skipif(False, reason='GPU is required')\n"
        "def test_inherited_gpu_guard():\n    pass\n"
        "\n@pytest.mark.parametrize('device', [\n"
        "    pytest.param('cpu', marks=pytest.mark.cuda), 'cuda:0'])\n"
        "def test_inherited_parameter(device):\n    pass\n"
        "\n@pytest.mark.neuron\n"
        "def test_inherited_simulator():\n    pass\n"
        "\n@pytest.mark.cpu\n@pytest.mark.cuda\n"
        "def test_local_explicit_accelerator():\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "test_class_devices.py").write_text(
        "import pytest\n"
        "\n@pytest.mark.cpu\n"
        "class TestMixedBackends:\n"
        "    @pytest.mark.parametrize('device', ['cuda:0'])\n"
        "    def test_class_metadata(self, device):\n        pass\n"
        "    @pytest.mark.cuda\n"
        "    def test_class_accelerator(self):\n        pass\n",
        encoding="utf-8",
    )
    return tmp_path


def _collect(suite: Path, marker: str | None = None) -> dict[str, set[str]]:
    environment = os.environ.copy()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONPATH"] = str(_ROOT)
    environment.pop("PYTEST_ADDOPTS", None)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        str(suite / "pytest.ini"),
        f"--rootdir={suite}",
        f"--confcutdir={suite}",
        "--collect-only",
        "-q",
        "-W",
        "error",
    ]
    if marker is not None:
        command.extend(("-m", marker))
    result = subprocess.run(
        command,
        cwd=suite,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    inventory = json.loads((suite / "inventory.json").read_text(encoding="utf-8"))
    return {name: set(markers) & _BACKENDS for name, markers in inventory.items()}


def test_authored_cpu_metadata_does_not_override_hardware_classification(
    classification_suite,
):
    assert _collect(classification_suite) == {
        "test_metadata[cpu]": {"cpu"},
        "test_metadata[cuda:0]": {"cpu"},
        "test_hardware[cpu]": {"cpu"},
        "test_hardware[cuda:0]": {"cuda"},
        "test_explicit_accelerator": {"cuda"},
        "test_cuda_guard": {"cuda"},
        "test_gpu_guard": {"cuda"},
        "test_simulator[cpu]": {"neuron"},
        "test_simulator[cuda:0]": {"neuron", "cuda"},
        "test_default": {"cpu"},
        "test_inherited_metadata[cuda:1]": {"cpu"},
        "test_inherited_explicit_accelerator": {"cuda"},
        "test_inherited_cuda_guard": {"cuda"},
        "test_inherited_gpu_guard": {"cuda"},
        "test_inherited_parameter[cpu]": {"cuda"},
        "test_inherited_parameter[cuda:0]": {"cpu"},
        "test_inherited_simulator": {"cpu", "neuron"},
        "test_local_explicit_accelerator": {"cuda"},
        "test_class_metadata[cuda:0]": {"cpu"},
        "test_class_accelerator": {"cuda"},
    }


def test_required_lane_keeps_metadata_and_non_cuda_simulator_cases(
    classification_suite,
):
    assert set(_collect(classification_suite, "cpu or (neuron and not cuda)")) == {
        "test_metadata[cpu]",
        "test_metadata[cuda:0]",
        "test_hardware[cpu]",
        "test_simulator[cpu]",
        "test_default",
        "test_inherited_metadata[cuda:1]",
        "test_inherited_parameter[cuda:0]",
        "test_inherited_simulator",
        "test_class_metadata[cuda:0]",
    }


def test_optional_lane_keeps_hardware_parameters_explicit_markers_and_guards(
    classification_suite,
):
    assert set(_collect(classification_suite, "cuda")) == {
        "test_hardware[cuda:0]",
        "test_explicit_accelerator",
        "test_cuda_guard",
        "test_gpu_guard",
        "test_simulator[cuda:0]",
        "test_inherited_explicit_accelerator",
        "test_inherited_cuda_guard",
        "test_inherited_gpu_guard",
        "test_inherited_parameter[cpu]",
        "test_local_explicit_accelerator",
        "test_class_accelerator",
    }


def test_mps_lane_is_separate_from_cuda_and_required_cpu_cases(classification_suite):
    (classification_suite / "test_mps_devices.py").write_text(
        "import pytest\n"
        "@pytest.mark.mps\ndef test_mps_explicit():\n    pass\n"
        "@pytest.mark.skipif(False, reason='MPS GPU is required')\n"
        "def test_mps_guard():\n    pass\n"
        "@pytest.mark.skipif(False, reason='Metal GPU is required')\n"
        "def test_metal_guard():\n    pass\n"
        "@pytest.mark.parametrize('device', ['mps', 'mps:0'])\n"
        "def test_mps_hardware(device):\n    pass\n"
        "@pytest.mark.cpu\n@pytest.mark.parametrize('device', ['mps'])\n"
        "def test_mps_metadata(device):\n    pass\n",
        encoding="utf-8",
    )
    assert set(_collect(classification_suite, "mps")) == {
        "test_mps_explicit",
        "test_mps_guard",
        "test_metal_guard",
        "test_mps_hardware[mps]",
        "test_mps_hardware[mps:0]",
    }
    assert "test_mps_metadata[mps]" in _collect(classification_suite, "cpu")
    cuda = _collect(classification_suite, "cuda")
    assert not any(name.startswith("test_mps_") for name in cuda)
    assert "test_metal_guard" not in cuda


def test_unrelated_reason_substrings_do_not_change_backend_classification(
    classification_suite,
):
    (classification_suite / "test_reason_substrings.py").write_text(
        "import pytest\n"
        "@pytest.mark.skipif(False, reason='diagnostic dumps are enabled')\n"
        "def test_dumps_cpu():\n    pass\n"
        "@pytest.mark.skipif(False, reason='GPU diagnostic dumps are enabled')\n"
        "def test_dumps_cuda():\n    pass\n",
        encoding="utf-8",
    )
    inventory = _collect(classification_suite)
    assert inventory["test_dumps_cpu"] == {"cpu"}
    assert inventory["test_dumps_cuda"] == {"cuda"}
