import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from dendra._bootstrap import torch_compiler_warning_context

EXAMPLE = Path(__file__).parents[2] / "examples" / "functional_gradient_descent.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location(
        "dendra_functional_gradient_descent_example",
        EXAMPLE,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode_args", [(), ("--checkpointed",), ("--compile-step",)])
def test_functional_gradient_descent_example_cli_smoke(
    monkeypatch,
    capsys,
    mode_args,
):
    module = _load_example_module()
    if "--compile-step" in mode_args:
        original_compile = module.dn.func.FunctionalPopulation.compile_rollout_chunk

        def compile_with_portable_backend(functional, steps, **options):
            options.setdefault("backend", "aot_eager")
            return original_compile(functional, steps, **options)

        monkeypatch.setattr(
            module.dn.func.FunctionalPopulation,
            "compile_rollout_chunk",
            compile_with_portable_backend,
        )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(EXAMPLE),
            "--iterations",
            "1",
            "--tstop",
            "0.01",
            "--chunklength",
            "2",
            "--report-every",
            "1",
            *mode_args,
        ],
    )

    with torch_compiler_warning_context():
        module.main()

    output = capsys.readouterr().out
    assert "Step    0, loss:" in output
    assert "Final loss:" in output
    assert "Optimized parameters:" in output


def test_custom_mse_reducer_matches_recorded_trace_loss_and_gradient():
    module = _load_example_module()
    dt = 0.005
    steps = 2

    reference = module.build_model(gnabar=0.12, gkbar=0.036)
    reference_functional, reference_tensors = module.dn.func.make_functional(
        reference,
        dt=dt,
    )
    reference_callbacks = reference_functional.make_callbacks(
        {"voltage": module.dn.func.Recorder(["v"])}
    )
    with torch.no_grad():
        target = module.simulate_callbacks(
            reference_functional,
            reference_tensors,
            reference_tensors.parameters,
            reference_callbacks,
            steps=steps,
            chunklength=2,
        )["voltage"]["v"]

    candidate = module.build_model(gnabar=0.05, gkbar=0.05)
    functional, tensors = module.dn.func.make_functional(candidate, dt=dt)
    parameter_names = (
        tensors.parameter_name(module.GNABAR_QUERY, within="model"),
        tensors.parameter_name(module.GKBAR_QUERY, within="model"),
    )
    parameters = tensors.independent_parameters(
        trainable=parameter_names,
        within="model",
    )
    selected = {name: parameters[name] for name in parameter_names}
    fixed = {name: value for name, value in parameters.items() if name not in selected}
    trace_callbacks = functional.make_callbacks(
        {"voltage": module.dn.func.Recorder(["v"])}
    )
    mse_callbacks = functional.make_callbacks({"mse": module.VoltageTraceMSE(target)})

    def trace_objective(local_selected):
        local_parameters = {**fixed, **local_selected}
        trace = module.simulate_callbacks(
            functional,
            tensors,
            local_parameters,
            trace_callbacks,
            steps=steps,
            chunklength=2,
        )["voltage"]["v"]
        return (trace - target).square().mean()

    def mse_objective(local_selected):
        local_parameters = {**fixed, **local_selected}
        return module.simulate_callbacks(
            functional,
            tensors,
            local_parameters,
            mse_callbacks,
            steps=steps,
            chunklength=2,
        )["mse"]

    expected_gradients, expected_loss = torch.func.grad_and_value(trace_objective)(
        selected
    )
    actual_gradients, actual_loss = torch.func.grad_and_value(mse_objective)(selected)
    torch.testing.assert_close(actual_loss, expected_loss)
    for name in parameter_names:
        torch.testing.assert_close(actual_gradients[name], expected_gradients[name])

    _state, auxiliary = functional.prepare_and_rollout(
        parameters,
        tensors.constants,
        tensors.state,
        steps=steps,
        callbacks=mse_callbacks,
    )
    carry = auxiliary["callbacks"].state["mse"]
    assert set(carry) == {"sse", "frames"}
    assert all(value.shape == () for value in carry.values())

    zero_state, zero_auxiliary = functional.prepare_and_rollout(
        parameters,
        tensors.constants,
        tensors.state,
        steps=0,
        callbacks=mse_callbacks,
    )
    first_state, first_auxiliary = functional.prepare_and_rollout(
        parameters,
        tensors.constants,
        zero_state,
        steps=1,
        callbacks=mse_callbacks,
        callback_state=zero_auxiliary["callbacks"].state,
    )
    second_state, second_auxiliary = functional.prepare_and_rollout(
        parameters,
        tensors.constants,
        first_state,
        steps=1,
        callbacks=mse_callbacks,
        callback_state=first_auxiliary["callbacks"].state,
    )
    torch.testing.assert_close(second_auxiliary["callbacks"]["mse"], actual_loss)
    _same_state, resumed_zero_auxiliary = functional.prepare_and_rollout(
        parameters,
        tensors.constants,
        second_state,
        steps=0,
        callbacks=mse_callbacks,
        callback_state=second_auxiliary["callbacks"].state,
    )
    torch.testing.assert_close(
        resumed_zero_auxiliary["callbacks"]["mse"],
        second_auxiliary["callbacks"]["mse"],
    )

    compiled = torch.compile(
        torch.func.grad_and_value(mse_objective),
        backend="aot_eager",
        fullgraph=True,
        dynamic=False,
    )
    with torch_compiler_warning_context():
        compiled_gradients, compiled_loss = compiled(selected)
    torch.testing.assert_close(compiled_loss, actual_loss)
    for name in parameter_names:
        torch.testing.assert_close(compiled_gradients[name], actual_gradients[name])
