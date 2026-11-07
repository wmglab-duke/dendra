"""Fresh Network runners initialize once and continue explicit carry/parameters."""

from collections.abc import Mapping

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from tests.func.test_network import (
    DT,
    DTYPE,
    _clone,
    _close,
    _loss,
    _mixed_network,
    _stim_network,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _runner(model, *, steps=3, **keywords):
    return dn.compile(
        model,
        dt=DT,
        steps=steps,
        initialization="fresh",
        backend="aot_eager",
        **keywords,
    )


def _callbacks():
    return {"trace": dn.func.Recorder(("cell.v", "t"))}


@pytest.mark.parametrize("inference", (False, True))
def test_fresh_network_restart_matches_native_initialization_and_preserves_snapshots(
    inference,
):
    model, native = _mixed_network(), _mixed_network()
    for network in (model, native):
        network._reset_runtime_clock(2.75)
        with torch.no_grad():
            for _ in range(3):
                network.step()
    runner = _runner(model, inference=inference, callbacks=_callbacks())
    captured = _clone(runner.initial_state, detached=True)
    source = runner.functional.extract().state
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    initial_voltage = native.cell.v.clone()
    with torch.no_grad():
        for _ in range(3):
            native.step()
    actual, auxiliary = runner()
    repeated, _ = runner()
    _close(actual, runner.functional.extract(native).state)
    _close(repeated, actual)
    torch.testing.assert_close(
        auxiliary["callbacks"]["trace"]["cell.v"][0], initial_voltage
    )
    assert actual["clock"]["step"].item() == 3
    assert auxiliary["callbacks"]["trace"]["t"][0].item() == 0
    _close(runner.initial_state, captured)
    _close(runner.functional.extract().state, source)
    assert "parameters" in auxiliary


@pytest.mark.parametrize("checkpoint_every", (None, 2))
def test_fresh_network_initialization_gradients_span_explicit_continuation(
    checkpoint_every,
):
    model = _mixed_network()
    name = f"synapses.{next(iter(model.synapses))}.weight.weights.0.param.rho"
    runner = _runner(
        model,
        inference=False,
        checkpoint_every=checkpoint_every,
        trainable=(name,),
        callbacks=_callbacks(),
    )
    initialization = runner.initialization_inputs
    population_inputs = dict(initialization.populations)
    upstream = population_inputs["cell"].v_init.detach().clone().requires_grad_()
    population_inputs["cell"] = population_inputs["cell"]._replace(
        v_init=0.97 * upstream + 0.4
    )
    initialization = initialization._replace(populations=population_inputs)
    initialized = runner.functional.initialize(
        runner.parameters, runner.constants, runner.initial_state, initialization
    )
    expected, oracle_aux = runner.functional.prepare_and_rollout(
        initialized.parameters,
        initialized.constants,
        initialized.state,
        steps=6,
        callbacks=runner.functional.make_callbacks(_callbacks()),
    )
    prefix, prefix_aux = runner(initialization=initialization)
    final, final_aux = runner(
        prefix,
        parameters=prefix_aux["parameters"],
        callback_state=prefix_aux["callbacks"].state,
    )
    trace = torch.cat(
        (
            prefix_aux["callbacks"]["trace"]["cell.v"],
            final_aux["callbacks"]["trace"]["cell.v"],
        )
    )
    _close(final, expected)
    torch.testing.assert_close(trace, oracle_aux["callbacks"]["trace"]["cell.v"])
    targets = (upstream, runner.get_parameter(name))
    actual_gradient = torch.autograd.grad(
        _loss(final) + 0.01 * trace.square().sum(), targets, retain_graph=True
    )
    oracle_gradient = torch.autograd.grad(
        _loss(expected)
        + 0.01 * oracle_aux["callbacks"]["trace"]["cell.v"].square().sum(),
        targets,
    )
    _close(actual_gradient, oracle_gradient)
    assert all(value.abs().sum() > 0 for value in actual_gradient)


def test_fresh_netstim_draw_and_initialization_run_once_outside_checkpoint_replay(
    monkeypatch,
):
    model = _stim_network()
    name = f"synapses.{next(iter(model.synapses))}.weight.weights.0.param.rho"
    runner = _runner(model, inference=False, checkpoint_every=2, trainable=(name,))
    source = runner.functional.extract().state
    calls = {"sample": 0, "initialize": 0}
    sample = runner.functional.sample_initialization_draw
    initialize = runner.functional._initialize_host

    def counted_sample(*args, **kwargs):
        calls["sample"] += 1
        return sample(*args, **kwargs)

    def counted_initialize(*args, **kwargs):
        calls["initialize"] += 1
        return initialize(*args, **kwargs)

    monkeypatch.setattr(runner.functional, "sample_initialization_draw", counted_sample)
    monkeypatch.setattr(runner.functional, "_initialize_host", counted_initialize)
    prefix, prefix_aux = runner()
    final, _ = runner(prefix, parameters=prefix_aux["parameters"])
    repeated, _ = runner()
    _close(repeated, prefix)
    assert calls == {"sample": 2, "initialize": 2}
    loss = final["populations"]["cell"]["integrator"]["v"].square().sum()
    gradient = torch.autograd.grad(loss, runner.get_parameter(name))[0]
    assert torch.isfinite(gradient).all()
    assert calls == {"sample": 2, "initialize": 2}
    _close(runner.functional.extract().state, source)


class _ScaleParameter(torch.nn.Module):
    def forward(self, value, factor):
        return (value * factor,)


def _with_parameter_transform(model):
    key = "parameters.integrator.mech.mechanisms.pas.g_param"
    model.cell.register_post_initialize_transform(
        "conductance_scale",
        _ScaleParameter(),
        reads=(key,),
        writes=(key,),
        inputs={"factor": model.cell.v.new_tensor(0.8)},
    )
    return model


def test_fresh_network_parameter_replacements_are_explicit_and_not_cached():
    model = _with_parameter_transform(_mixed_network())
    name = "populations.cell.integrator.mech.mechanisms.pas.g_param"
    runner = _runner(model, inference=False, trainable=(name,), checkpoint_every=2)
    original = runner.get_parameter(name).detach().clone()
    effective_name = "integrator.mech.mechanisms.pas.g_param"
    for factor in (1.0, 1.1):
        with torch.no_grad():
            runner.get_parameter(name).copy_(factor * original)
        owned = runner.get_parameter(name).detach().clone()
        first, auxiliary = runner()
        effective = auxiliary["parameters"]["populations"]["cell"][effective_name]
        torch.testing.assert_close(effective, 0.8 * runner.get_parameter(name))
        with pytest.raises(ValueError, match="parameters|continu"):
            runner(first)
        continued, continued_aux = runner(
            first,
            parameters=auxiliary["parameters"],
            constants=auxiliary["constants"],
        )
        assert (
            continued_aux["parameters"]["populations"]["cell"][effective_name]
            is effective
        )
        torch.autograd.grad(_loss(continued), runner.get_parameter(name))
        torch.testing.assert_close(runner.get_parameter(name), owned)


def test_fresh_network_inference_initialization_parameters_have_version_counters():
    model = _with_parameter_transform(_mixed_network())
    with torch.inference_mode():
        runner = _runner(model)
        _, auxiliary = runner()
    parameters = auxiliary["parameters"]["populations"]["cell"]
    assert all(not value.is_inference() for value in parameters.values())
    assert all(value._version >= 0 for value in parameters.values())


def test_fresh_network_initialization_overrides_and_continuation_are_unambiguous():
    runner = _runner(_mixed_network(), inference=False)
    assert isinstance(runner.initialization_inputs, dn.func.NetworkInitializationInput)
    with pytest.raises(TypeError, match="NetworkInitializationInput"):
        runner(initialization={})
    first, auxiliary = runner()
    with pytest.raises(ValueError, match="explicit state|initialization"):
        runner(first, initialization=runner.initialization_inputs)
    with pytest.raises(ValueError, match="callback_state|state"):
        runner(callback_state=object())
    continued, _ = runner(first, parameters=auxiliary["parameters"])
    assert continued["clock"]["step"].item() == 6


class _ModeChangingMapping(Mapping):
    def __init__(self, values, model):
        self.values, self.model = values, model

    def __getitem__(self, key):
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def items(self):
        self.model.eval()
        return self.values.items()


@pytest.mark.parametrize("packet", ("initialization", "runtime"))
def test_fresh_network_materializes_public_mappings_before_source_validation(packet):
    model = _mixed_network()
    runner = _runner(model, inference=False)
    keywords = {}
    if packet == "initialization":
        keywords["initialization"] = runner.initialization_inputs._replace(
            populations=_ModeChangingMapping(
                runner.initialization_inputs.populations, model
            )
        )
    else:
        keywords["inputs"] = dn.func.NetworkRolloutInput(
            _ModeChangingMapping({}, model)
        )
    with pytest.raises(
        dn.func.FunctionalizationError, match="changed|configuration|mode"
    ):
        runner(**keywords)


def _legacy_hook(population):
    del population


def test_unsupported_fresh_network_initialization_does_not_fall_back_to_snapshot():
    model = _mixed_network()
    model.cell.register_pre_initialize_hook(_legacy_hook)
    runner = _runner(model, inference=False)
    with pytest.raises(
        dn.func.FunctionalizationError, match="initialization|initialize"
    ):
        runner()
    continued, _ = runner(runner.initial_state)
    assert torch.isfinite(continued["populations"]["cell"]["integrator"]["v"]).all()


@pytest.mark.parametrize("execution", ("pure", "runner"))
def test_late_shared_runtime_parameter_writes_preserve_native_materializations(
    execution,
):
    def network():
        with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE, DEVICE="cpu"):
            populations = {}
            for name in ("left", "right"):
                population = dn.SingleCompartment(
                    N=1, C=1, integrator=dn.bwd_euler_sc(imem=False), v_init=-60.0
                )
                population.insert(pas, g=0.001, e=-65.0)
                population.build()
                populations[name] = population
            populations["right"].cm_scale_param.rho = populations[
                "left"
            ].cm_scale_param.rho
            model = dn.Network(populations)
            model.train()
            model.initialize(DT)
            # The later write updates the shared raw source, while LEFT retains
            # the capacitance and solver coefficients prepared before that write.
            populations["right"].register_pre_initialize_transform(
                "late_capacitance_scale",
                _ScaleParameter(),
                reads=("parameters.cm_scale_param.rho",),
                writes=("parameters.cm_scale_param.rho",),
                inputs={"factor": populations["right"].v.new_tensor(1.5)},
            )
        return model

    model, native = network(), network()
    functional, tensors = dn.func.make_functional_network(model)
    frozen = _clone(tensors, detached=True)
    native.initialize(DT)
    if execution == "pure":
        initialized = functional.initialize(*tensors, functional.initialization_inputs)
        left = initialized.parameters["populations"]["left"]["cm_scale_param.rho"]
        right = initialized.parameters["populations"]["right"]["cm_scale_param.rho"]
        assert left is right
        actual, _ = functional.prepare_and_step(*initialized)
        native.step()
    else:
        runner = _runner(model, inference=False)
        actual, _ = runner()
        for _ in range(3):
            native.step()
    assert not torch.allclose(native.left.cm_scale, native.right.cm_scale)
    _close(actual, functional.extract(native).state)
    _close(tensors, frozen)
