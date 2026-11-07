"""Canonical packets retain absent-family semantics in compiled scans."""

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_convenience_validation import _stochastic_runner
from tests.func.test_poisson_stimulation import _clone, _close
from tests.func.test_stimulation_rng_sharing import _environment as _environment

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("family", ("noise", "poisson"))
def test_canonical_population_scan_preserves_the_absent_runtime_family(family):
    with torch_compiler_warning_context():
        source, waveform, runner = _stochastic_runner(family)
        native, _, _ = _stochastic_runner(family)
        functional = runner.functional
        original = _clone(functional.extract())
        global_rng = torch.get_rng_state().clone()
        source_rng = (
            None if waveform is None else waveform.generator.get_state().clone()
        )
        packet = functional.sample_runtime_draws(runner.initial_state, steps=4)
        present, absent = (
            ("noise", "stimulation") if family == "noise" else ("stimulation", "noise")
        )
        assert packet.draws[present] is not None
        assert packet.draws[absent] is None
        legacy = dn.func.RolloutInput(
            noise=functional.sample_noise_draw_tape(runner.initial_state, 4),
            stimulation=functional.sample_stimulation_draw_tape(
                runner.initial_state, 4
            ),
        )
        prepared = functional.prepare(runner.parameters, runner.constants)
        kernel = functional.compile_rollout_chunk(
            4, execution="scan", backend="aot_eager"
        )
        with torch.no_grad():
            actual, _ = kernel(
                runner.parameters, prepared, runner.initial_state, draws=packet
            )
            expected, _ = functional.rollout(
                runner.parameters, prepared, runner.initial_state, legacy, steps=4
            )
            for _ in range(4):
                native.step(functional.dt)
        _close(actual, expected)
        _close(actual, functional.extract(native).state)
        _close(functional.extract(), original)
        _close(torch.get_rng_state(), global_rng)
        if waveform is not None:
            _close(waveform.generator.get_state(), source_rng)
