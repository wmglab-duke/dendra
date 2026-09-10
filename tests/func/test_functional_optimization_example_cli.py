"""Keep compiled kernel width separate from the fitting example's checkpoints."""

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).parents[2] / "examples" / "functional_gradient_descent.py"


def _parse(monkeypatch, *arguments):
    spec = importlib.util.spec_from_file_location(
        "dendra_functional_gradient_descent_cli_example", EXAMPLE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [str(EXAMPLE), *arguments])
    args = module.parse_args()
    module.validate_args(args)
    return args


def test_compiled_chunk_width_is_independent_of_checkpoint_spacing(monkeypatch):
    args = _parse(
        monkeypatch,
        "--compiled-chunk-steps",
        "4",
        "--checkpointed",
        "--chunklength",
        "11",
    )
    assert args.compiled_chunk_steps == 4
    assert args.chunklength == 11
    assert args.checkpointed
    assert not args.compiled_step


@pytest.mark.parametrize("flag", ["--compiled-step", "--compile-step"])
def test_single_step_compilation_aliases_remain_available(monkeypatch, flag):
    args = _parse(monkeypatch, flag)
    assert args.compiled_step
    assert args.compiled_chunk_steps is None


@pytest.mark.parametrize("steps", ["0", "-1"])
def test_compiled_chunk_width_must_be_positive(monkeypatch, steps):
    with pytest.raises(ValueError, match="compiled-chunk-steps must be positive"):
        _parse(monkeypatch, "--compiled-chunk-steps", steps)


def test_compilation_flags_cannot_conflict(monkeypatch):
    with pytest.raises(SystemExit) as error:
        _parse(monkeypatch, "--compiled-step", "--compiled-chunk-steps", "4")
    assert error.value.code == 2
