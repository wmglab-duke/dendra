"""Authored canonical samplers cannot invalidate captured preparation inputs."""

import pytest
import torch

import dendra as dn
from tests.func.test_poisson_stimulation import DT
from tests.func.test_stimulation_rng_sharing import _environment as _environment
from tests.func.test_stimulation_rng_sharing import _lower, _population

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("execution", ("run", "runtime_extra_while"))
@pytest.mark.parametrize("source", ("parameters", "constants"))
def test_canonical_sampler_rechecks_independent_prepared_inputs(
    monkeypatch, execution, source
):
    model, _, _ = _population()
    if execution == "run":
        functional, tensors = _lower(model)
        parameters = tensors.independent_parameters()
        constants = {
            name: value.detach().clone() for name, value in tensors.constants.items()
        }
        prepared = functional.prepare(parameters, constants)
        bound = functional.bind(parameters, prepared)

        def execute():
            return dn.func.run(bound, tensors.state, steps=4)

    else:
        runner = dn.compile(
            model,
            dt=DT,
            steps=4,
            extra=dn.VarInput(),
            chunk_steps=4,
            execution="while",
            callbacks={"trace": dn.func.Recorder(("v", "t"))},
            backend="aot_eager",
        )
        functional = runner.functional
        parameters = runner.parameters
        constants = {
            name: value.detach().clone() for name, value in runner.constants.items()
        }

        def execute():
            return runner(constants=constants, extra=None)

    target = (
        next(value for name, value in parameters.items() if name.endswith(".g_param"))
        if source == "parameters"
        else constants["diam"]
    )
    original_sampler = functional.sample_runtime_draws
    calls = 0

    def sample_and_mutate(state, *, steps=None):
        nonlocal calls
        packet = original_sampler(state, steps=steps)
        with torch.no_grad():
            target.add_(1.0)
        calls += 1
        return packet

    monkeypatch.setattr(functional, "sample_runtime_draws", sample_and_mutate)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match=rf"{source} changed after prepare\(\)",
    ):
        execute()
    assert calls == 1
