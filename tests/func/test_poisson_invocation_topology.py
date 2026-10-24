"""Nested event counts reject changed call graphs and ignore unused phase slots."""

from unittest.mock import patch

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func import _poisson_stimulation
from tests.func.test_poisson_invocation_guards import _case, _RepeatedInitializer
from tests.func.test_poisson_stimulation import DT, DTYPE, _clone, _close, _waveform

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _nested_case():
    generator = torch.Generator().manual_seed(149)
    inner = _waveform(generator=generator)
    outer = inner.poisson(
        interval=0.08,
        n=2,
        start=-0.025,
        noise=0.6,
        randomize_every_call=True,
        generator=generator,
    )
    population, _ = _case(_RepeatedInitializer, waveform=outer)
    return population, outer, generator


def test_runtime_count_guard_ignores_unused_initialization_invocation_rows():
    source, outer, generator = _nested_case()
    before_rng = generator.get_state().clone()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    plan = functional._poisson_plan
    entry = next(entry for entry in plan.entries if entry.source is outer)
    runtime_count = plan._runtime_counts[entry.key]
    assert plan._initialization_counts[entry.key] > runtime_count > 0
    original = functional.sample_stimulation_draw(tensors.state)
    draws = _clone(original.draws)
    draws[entry.key][runtime_count:] = 0.0
    unused_zeros = original._replace(draws=draws)
    plan.validate(unused_zeros)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    expected, _ = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.StepInput(stimulation=original),
    )
    actual, _ = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.StepInput(stimulation=unused_zeros),
    )
    _close(actual, expected)
    _close(generator.get_state(), before_rng)


def test_host_sampling_rejects_short_outer_schedule_before_sampling_children(
    monkeypatch,
):
    source, outer, generator = _nested_case()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    before_rng = generator.get_state().clone()
    calls = []
    sample = _poisson_stimulation.sample_poisson_draws

    def zero_first_interval(**kwargs):
        calls.append(kwargs["n"])
        assert kwargs["n"] == outer.n
        with patch(
            "torch.rand",
            side_effect=lambda *args, **options: torch.zeros(
                (), dtype=options["dtype"], device=options["device"]
            ),
        ):
            return sample(**kwargs)

    monkeypatch.setattr(
        _poisson_stimulation, "sample_poisson_draws", zero_first_interval
    )
    with pytest.raises(
        dn.func.FunctionalizationError, match="change the fixed nested invocation count"
    ):
        functional.sample_stimulation_draw(tensors.state)
    assert calls == [outer.n]
    _close(generator.get_state(), before_rng)
