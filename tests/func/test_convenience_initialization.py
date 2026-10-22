"""Opt-in fresh compiled restarts keep physiological initialization in AD."""

from __future__ import annotations

import inspect
from collections.abc import Mapping

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_convenience_extra import (
    _extra,
    _selected,
)
from tests.func.test_functional_initialization import (
    AM1,
    _assert_module_snapshot,
    _assert_tree_close,
    _module_snapshot,
)
from tests.func.test_initialization_parity import (
    CHANNELS,
    DT,
    IC,
    PROBE,
    _close_history,
    _imperative_history,
)
from tests.func.test_initialization_parity import _loss as _initialization_loss
from tests.func.test_initialization_parity import _model as _initialization_model
from tests.func.test_initialization_parity import (
    _read_path,
    _record_paths,
    _voltage,
)

STEPS = 4
GNABAR = "integrator.mech.mechanisms.hh.gnabar_param"
pytestmark = pytest.mark.cpu


@pytest.fixture(params=(1, 4), ids=("threads1", "threads4"))
def cpu_threads(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    try:
        yield request.param
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _clone(value):
    return pytree.tree_map_only(
        torch.Tensor, lambda tensor: tensor.detach().clone(), value
    )


def _model(
    kind="myelinated",
    *,
    device="cpu",
    dtype=torch.float64,
    shell_resistance_scale=1.0,
):
    model = _initialization_model(
        kind,
        device=device,
        dtype=dtype,
        batch=(2,),
        shell_resistance_scale=shell_resistance_scale,
    )
    with torch.no_grad():
        model.v.add_(1.5)
        if hasattr(model, "vc"):
            model.vc[..., 0].add_(1.5)
        model.mech.hh.m.add_(0.025)
        model.t.fill_(2.0)
        model._duration_remainder.fill_(0.004)
    return model


def _runner(
    model,
    *,
    execution="scan",
    checkpoint_every=None,
    backend="eager",
    initialization="fresh",
    bound_extra=False,
):
    extra = _extra(model) if bound_extra else None
    with torch.inference_mode(False), torch.no_grad():
        functional, tensors = dn.func.make_functional(model, dt=DT, extra=extra)
    trainable = tuple(
        f"integrator.mech.mechanisms.hh.{name}_param" for name in CHANNELS
    ) + (AM1, "celsius_param")
    trainable += tuple(name for name in tensors.parameters if _selected(name))
    options = {}
    if backend == "inductor":
        options = {
            "fullgraph": True,
            "options": {
                "cpp_wrapper": model.device().type == "cpu",
                "compile_threads": 1,
            },
        }
    return dn.compile(
        model,
        dt=DT,
        steps=STEPS,
        extra=extra,
        execution=execution,
        initialization=initialization,
        inference=execution == "while",
        trainable=trainable,
        checkpoint_every=checkpoint_every,
        callbacks={"trace": dn.func.Recorder(tuple(_record_paths(functional)))},
        backend=backend,
        **options,
    )


def _arguments(runner, model):
    parameters = _clone(dict(runner.parameters))
    targets = {
        name: parameters[f"integrator.mech.mechanisms.hh.{name}_param"].requires_grad_()
        for name in CHANNELS
    }
    parameters[AM1] = (parameters[AM1] * 1.17).requires_grad_()
    parameters["celsius_param"] = (parameters["celsius_param"] + 0.4).requires_grad_()
    targets["rate"] = parameters[AM1]
    targets["celsius"] = parameters["celsius_param"]
    if "noded2" in parameters:
        parameters["noded2"] = (parameters["noded2"] * 1.04).requires_grad_()
        targets["geometry"] = parameters["noded2"]
    else:
        # This is an owned raw geometry leaf, not a prepared workspace. Its
        # identity is intentionally retained when preparing the eager oracle.
        targets["geometry"] = runner.constants[
            "populations.block_path.diam"
        ].requires_grad_()
    for name, value in parameters.items():
        if _selected(name):
            targets[name] = value.requires_grad_()
    initialization = _clone(runner.initialization_inputs)
    voltage = _voltage(model).requires_grad_()
    states = dict(initialization.states)
    states[IC] = (states[IC] + 0.07).requires_grad_()
    targets["initial_voltage"] = voltage
    targets["ic"] = states[IC]
    initialization = initialization._replace(v_init=voltage, states=states)
    shape = (STEPS, *model.shape)
    drive = torch.linspace(
        -2.0e-9,
        3.0e-9,
        STEPS * model.v.numel(),
        device=model.device(),
        dtype=model.dtype(),
    ).reshape(shape)
    amplitude = model.v.new_ones((), requires_grad=True)
    targets["stimulus"] = amplitude
    return (
        parameters,
        initialization,
        dn.func.RolloutInput(intra=amplitude * drive),
        targets,
    )


def _recorded(result):
    return dict(result[1]["callbacks"]["trace"])


def _history(recorded):
    integrator = {
        name: recorded[name] for name in ("v", "vc", "i_membrane") if name in recorded
    }
    return {
        "integrator": integrator,
        "mechanisms": {
            "hh": {"m": recorded["hh.m"]},
            PROBE: {
                name: recorded[f"{PROBE}.{name}"] for name in ("inferred", "fixed")
            },
        },
        "mechanism_buffers": {PROBE: {"seed": recorded[f"{PROBE}.seed"]}},
    }


def _loss(recorded):
    value = _initialization_loss(_history(recorded))
    if "vc" in recorded:
        shells = recorded["vc"][..., 1:]
        weights = torch.linspace(
            0.7, 1.3, shells.shape[-2], device=shells.device, dtype=shells.dtype
        )
        value = value + 0.1 * (weights.unsqueeze(-1) * shells).mean()
    return value


def _gradients(recorded, targets, *, retain_graph=False):
    values = torch.autograd.grad(
        _loss(recorded), tuple(targets.values()), retain_graph=retain_graph
    )
    for name, value in zip(targets, values, strict=True):
        assert torch.isfinite(value).all() and torch.count_nonzero(value) > 0, name
    return dict(zip(targets, values, strict=True))


def _close(actual, expected, *, dtype=torch.float64):
    _close_history(actual, expected, dtype=dtype)


def _explicit_reference(runner, parameters, initialization, inputs):
    functional = runner.functional
    initialized = functional.initialize(parameters, runner.constants, initialization)
    prepared = functional.prepare(initialized.parameters, initialized.constants)
    final, auxiliary = dn.func.run(
        functional.bind(initialized.parameters, prepared),
        initialized.state,
        steps=STEPS,
        inputs=inputs,
        callbacks=runner._callbacks,
    )
    return final, {**auxiliary, "parameters": initialized.parameters}


def _exercise(
    kind="myelinated",
    *,
    device="cpu",
    dtype=torch.float64,
    shell_resistance_scale=1.0,
    backend="eager",
    execution="scan",
    checkpoint_every=None,
    bound_extra=False,
):
    model = _model(
        kind, device=device, dtype=dtype, shell_resistance_scale=shell_resistance_scale
    )
    runner = _runner(
        model,
        backend=backend,
        execution=execution,
        checkpoint_every=checkpoint_every,
        bound_extra=bound_extra,
    )
    source = _module_snapshot(model)
    owned_parameters = _clone(dict(runner.parameters))
    owned_initialization = _clone(runner.initialization_inputs)
    parameters, initialization, inputs, targets = _arguments(runner, model)
    actual_result = runner(
        parameters=parameters, initialization=initialization, inputs=inputs
    )
    actual = _recorded(actual_result)
    gradients = _gradients(actual, targets) if execution == "scan" else {}
    (
        reference_parameters,
        reference_initialization,
        reference_inputs,
        reference_targets,
    ) = _arguments(runner, model)
    expected_result = _explicit_reference(
        runner, reference_parameters, reference_initialization, reference_inputs
    )
    _close(actual_result[0], expected_result[0], dtype=dtype)
    # Infinite built-in waveform off-times are valid authoring parameters.
    _assert_tree_close(
        actual_result[1]["parameters"],
        expected_result[1]["parameters"],
        rtol=2.0e-4 if dtype == torch.float32 else 3.0e-9,
        atol=2.0e-5 if dtype == torch.float32 else 3.0e-10,
    )
    _close(actual, _recorded(expected_result), dtype=dtype)
    if execution == "scan":
        _close(
            gradients,
            _gradients(_recorded(expected_result), reference_targets),
            dtype=dtype,
        )
        assert runner._prepared is None  # Do not retain a training preparation/graph.
    else:
        assert not actual["v"].requires_grad
    assert actual["v"].shape == (STEPS + 1, *model.shape)
    assert actual["t"][0].item() == 0.0
    assert set(actual) == set(_record_paths(runner.functional))
    _assert_tree_close(dict(runner.parameters), owned_parameters)
    _assert_tree_close(runner.initialization_inputs, owned_initialization)
    _assert_module_snapshot(model, source)
    return actual, gradients


@pytest.mark.parametrize("kind", ("myelinated", "multi_block"))
@pytest.mark.parametrize(
    ("execution", "checkpoint_every"), (("scan", None), ("scan", 2), ("while", None))
)
def test_fresh_runner_full_traces_and_initialization_gradients(
    kind, execution, checkpoint_every, cpu_threads
):
    _exercise(kind, execution=execution, checkpoint_every=checkpoint_every)


@pytest.mark.slow
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_fresh_runner_real_cpp_trace_and_gradient_composition(execution, cpu_threads):
    _exercise(
        "multi_block",
        backend="inductor",
        execution=execution,
        checkpoint_every=2 if execution == "scan" else None,
    )


def test_fresh_runner_bound_extra_uses_fresh_zero_clock_and_retains_gradients(
    cpu_threads,
):
    _exercise("myelinated", checkpoint_every=2, bound_extra=True)


@pytest.mark.parametrize("factory", (dn.compile, dn.func.CompiledRunner))
def test_initialization_policy_and_input_override_are_explicit(factory):
    assert inspect.signature(factory).parameters["initialization"].default == "snapshot"
    assert (
        "initialization"
        in inspect.signature(dn.func.CompiledRunner.__call__).parameters
    )


@pytest.mark.parametrize("policy", (None, True, "auto", "initialize"))
def test_invalid_initialization_policy_fails_before_lowering(monkeypatch, policy):
    import dendra.func._convenience as convenience

    def must_not_lower(*args, **kwargs):
        pytest.fail("invalid policy must fail before functionalization")

    monkeypatch.setattr(convenience, "make_functional", must_not_lower)
    with pytest.raises((TypeError, ValueError), match="initialization|snapshot|fresh"):
        dn.compile(object(), dt=DT, steps=STEPS, initialization=policy, backend="eager")


def test_initialization_inputs_are_independent_versioned_editable_leaves():
    model = _model()
    _, tensors = dn.func.make_functional(model, dt=DT)
    with torch.inference_mode():
        runner = _runner(model)
    assert isinstance(runner.initialization_inputs, dn.func.InitializationInput)
    source_leaves = [
        value
        for value in pytree.tree_leaves(tensors.initialization)
        if torch.is_tensor(value)
    ]
    owned_leaves = [
        value
        for value in pytree.tree_leaves(runner.initialization_inputs)
        if torch.is_tensor(value)
    ]
    assert len(source_leaves) == len(owned_leaves)
    for owned, source in zip(owned_leaves, source_leaves, strict=True):
        assert owned.data_ptr() != source.data_ptr()
        assert owned.is_leaf and not owned.requires_grad
        assert isinstance(owned._version, int)
        assert all(owned.data_ptr() != p.data_ptr() for p in runner.parameters.values())
    source = _module_snapshot(model)
    first = _recorded(runner())["v"]
    runner.initialization_inputs.v_init.add_(1.0)
    second = _recorded(runner())["v"]
    assert not torch.equal(first, second)
    _assert_module_snapshot(model, source)


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_snapshot_default_and_fresh_restart_match_their_actual_imperative_frames(
    execution,
):
    source = _model()
    snapshot = _runner(source, execution=execution, initialization="snapshot")
    fresh = _runner(source, execution=execution)
    snapshot_result, fresh_result = snapshot(), fresh()
    assert "parameters" not in snapshot_result[1]
    _assert_tree_close(_recorded(snapshot_result)["v"][0], source.v)
    torch.testing.assert_close(_recorded(snapshot_result)["t"][0], source.t)
    reference = _model()
    reference.initialize()
    inputs = dn.func.RolloutInput(
        ve=reference.v.new_zeros((STEPS, *reference.shape)),
        intra=reference.v.new_zeros((STEPS, *reference.shape)),
    )
    with torch.no_grad():
        history = _imperative_history(fresh.functional, reference, inputs)
    expected = {
        name: _read_path(history, path)
        for name, path in _record_paths(fresh.functional).items()
    }
    _close(_recorded(fresh_result), expected)
    _close(_recorded(fresh()), _recorded(fresh_result))
    assert not torch.equal(
        _recorded(snapshot_result)["hh.m"][0], _recorded(fresh_result)["hh.m"][0]
    )


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_explicit_continuation_bypasses_fresh_initializer_and_rejects_ambiguous_inputs(
    monkeypatch, execution
):
    runner = _runner(_model(), execution=execution)
    first = runner()
    with pytest.raises(ValueError, match="state|continu|initialization"):
        runner(first[0], initialization=runner.initialization_inputs)
    with pytest.raises(ValueError, match="state|continu|callback"):
        runner(callback_state=first[1]["callbacks"].state)

    def forbidden(*args, **kwargs):
        pytest.fail("an explicit continuation state must bypass initialization")

    monkeypatch.setattr(runner.functional, "initialize", forbidden)
    actual = runner(
        first[0],
        parameters=first[1]["parameters"],
        callback_state=first[1]["callbacks"].state,
    )
    prepared = runner.functional.prepare(first[1]["parameters"], runner.constants)
    with torch.no_grad():
        expected = dn.func.run(
            runner.functional.bind(first[1]["parameters"], prepared),
            first[0],
            steps=STEPS,
            callbacks=runner._callbacks,
            callback_state=first[1]["callbacks"].state,
        )
    _close(actual[0], expected[0])
    _close(_recorded(actual), _recorded(expected))
    assert _recorded(actual)["v"].shape[0] == STEPS


def test_snapshot_rejects_initialization_override_but_accepts_ordinary_continuation():
    # Parameter-writing transforms must not change snapshot-mode continuation
    # rules: that policy does not execute those transforms in the first place.
    runner = _runner(_with_transforms(_model()), initialization="snapshot")
    with pytest.raises(ValueError, match="snapshot|fresh|initialization"):
        runner(initialization=runner.initialization_inputs)
    first = runner()
    second = runner(first[0], callback_state=first[1]["callbacks"].state)
    assert "parameters" not in second[1]
    assert second[0]["clock"]["t"] > first[0]["clock"]["t"]


def test_fresh_initializer_gradients_span_two_continued_scans(cpu_threads):
    model = _with_transforms(_model())
    runner = _runner(model)

    def evaluate(reference):
        parameters, initialization, inputs, targets = _arguments(runner, model)
        for name, value in initialization.transforms.items():
            targets[name] = value.requires_grad_()
        if reference:
            functional = runner.functional
            initialized = functional.initialize(
                parameters, runner.constants, initialization
            )
            prepared = functional.prepare(initialized.parameters, initialized.constants)
            result = dn.func.run(
                functional.bind(initialized.parameters, prepared),
                initialized.state,
                steps=2 * STEPS,
                inputs=dn.func.RolloutInput(
                    intra=torch.cat((inputs.intra, inputs.intra))
                ),
                callbacks=runner._callbacks,
            )
            recorded = _recorded(result)
        else:
            first = runner(
                parameters=parameters, initialization=initialization, inputs=inputs
            )
            second = runner(
                first[0],
                parameters=first[1]["parameters"],
                inputs=inputs,
                callback_state=first[1]["callbacks"].state,
            )
            recorded = {
                name: torch.cat((_recorded(first)[name], _recorded(second)[name]))
                for name in _recorded(first)
            }
            assert recorded["v"].shape[0] == 2 * STEPS + 1
            result = second
        return result[0], recorded, _gradients(recorded, targets)

    actual_state, actual, gradients = evaluate(False)
    expected_state, expected, expected_gradients = evaluate(True)
    _close(actual_state, expected_state)
    _close(actual, expected)
    _close(gradients, expected_gradients)


@pytest.mark.parametrize("part", ("states", "transforms"))
def test_external_initialization_mappings_are_materialized_before_source_validation(
    part,
):
    model = _with_transforms(_model())
    runner = _runner(model)

    class ModeChangingMapping(Mapping):
        def __init__(self, values):
            self.values = values

        def __getitem__(self, key):
            return self.values[key]

        def __iter__(self):
            model.eval()
            return iter(self.values)

        def __len__(self):
            return len(self.values)

    initialization = runner.initialization_inputs._replace(
        **{part: ModeChangingMapping(getattr(runner.initialization_inputs, part))}
    )
    with pytest.raises(
        dn.func.FunctionalizationError, match="training|train|mode|source|changed"
    ):
        runner(initialization=initialization)


def test_fresh_override_rejects_wrong_type_and_initial_voltage_schema():
    runner = _runner(_model())
    with pytest.raises(TypeError, match="InitializationInput"):
        runner(initialization={"v_init": runner.initialization_inputs.v_init})
    initialization = runner.initialization_inputs._replace(
        v_init=runner.initialization_inputs.v_init[..., :-1]
    )
    with pytest.raises(ValueError, match="v_init|voltage|shape"):
        runner(initialization=initialization)


def _legacy_hook(population):
    del population


def test_unsupported_fresh_initialization_never_falls_back_to_snapshot():
    source = _model()
    source.register_pre_initialize_hook(_legacy_hook)
    runner = _runner(source)
    with pytest.raises(
        dn.func.FunctionalizationError, match="initialization|initialize"
    ):
        runner()
    # Transition admission is broader: explicit continuation remains usable.
    continued = runner(runner.initial_state)
    assert torch.isfinite(continued[0]["integrator"]["v"]).all()
    _runner(source, initialization="snapshot")()


class _ShiftVoltage(torch.nn.Module):
    def forward(self, voltage, offset):
        return (voltage + offset,)


class _ScaleParameter(torch.nn.Module):
    def forward(self, value, factor):
        return (value * factor,)


def _with_transforms(model):
    model.register_pre_initialize_transform(
        "voltage_shift",
        _ShiftVoltage(),
        reads=("state.integrator.v",),
        writes=("state.integrator.v",),
        inputs={"offset": model.v.new_tensor(0.2)},
    )
    parameter = "parameters." + GNABAR
    model.register_post_initialize_transform(
        "conductance_scale",
        _ScaleParameter(),
        reads=(parameter,),
        writes=(parameter,),
        inputs={"factor": model.v.new_tensor(0.9)},
    )
    return model


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_fresh_parameter_writes_use_effective_mapping_without_committing_it(
    execution, cpu_threads
):
    model = _with_transforms(_model())
    runner = _runner(model, execution=execution)
    parameters, initialization, inputs, targets = _arguments(runner, model)
    transforms = _clone(initialization.transforms)
    transforms["pre.voltage_shift.offset"] = model.v.new_tensor(
        0.35, requires_grad=True
    )
    transforms["post.conductance_scale.factor"] = model.v.new_tensor(
        0.8, requires_grad=True
    )
    targets.update(transforms)
    initialization = initialization._replace(transforms=transforms)
    source = _module_snapshot(model)
    owned = _clone(dict(runner.parameters))
    result = runner(parameters=parameters, initialization=initialization, inputs=inputs)
    expected = _explicit_reference(runner, parameters, initialization, inputs)
    _close(_recorded(result), _recorded(expected))
    _close(result[1]["parameters"], expected[1]["parameters"])
    torch.testing.assert_close(
        result[1]["parameters"][GNABAR],
        parameters[GNABAR] * transforms["post.conductance_scale.factor"],
    )
    if execution == "scan":
        gradients = _gradients(_recorded(result), targets, retain_graph=True)
        _close(gradients, _gradients(_recorded(expected), targets))
    _assert_tree_close(dict(runner.parameters), owned)
    _assert_module_snapshot(model, source)
    with pytest.raises(ValueError, match="parameters|continu"):
        runner(result[0])
    resumed = runner(
        result[0],
        parameters=result[1]["parameters"],
        callback_state=result[1]["callbacks"].state,
    )
    # Explicit continuation must neither reapply a multiplicative transform
    # nor silently substitute the runner's untransformed owned parameter.
    torch.testing.assert_close(
        resumed[1]["parameters"][GNABAR], result[1]["parameters"][GNABAR]
    )
    assert _recorded(resumed)["v"].shape[0] == STEPS
    with torch.no_grad():
        prepared = runner.functional.prepare(result[1]["parameters"], runner.constants)
        expected_resumed = dn.func.run(
            runner.functional.bind(result[1]["parameters"], prepared),
            result[0],
            steps=STEPS,
            callbacks=runner._callbacks,
            callback_state=result[1]["callbacks"].state,
        )
    _close(resumed[0], expected_resumed[0])
    _close(_recorded(resumed), _recorded(expected_resumed))


def test_fresh_inference_parameter_replacements_remain_versioned_under_outer_inference_mode():
    runner = _runner(_with_transforms(_model()), execution="while")
    with torch.inference_mode():
        result = runner()
    assert not _recorded(result)["v"].requires_grad
    torch.testing.assert_close(
        result[1]["parameters"][GNABAR],
        runner.parameters[GNABAR]
        * runner.initialization_inputs.transforms["post.conductance_scale.factor"],
    )
    for value in result[1]["parameters"].values():
        assert not value.is_inference()
        assert isinstance(value._version, int)


def test_fresh_nonleaf_initial_voltage_overrides_retain_upstream_autograd(cpu_threads):
    model = _model()
    runner = _runner(model, checkpoint_every=2)

    def evaluate(reference):
        parameters, initialization, inputs, targets = _arguments(runner, model)
        upstream = initialization.v_init.detach().clone().requires_grad_()
        initialization = initialization._replace(v_init=0.98 * upstream - 0.5)
        targets["initial_voltage"] = upstream
        result = (
            _explicit_reference(runner, parameters, initialization, inputs)
            if reference
            else runner(
                parameters=parameters, initialization=initialization, inputs=inputs
            )
        )
        return _recorded(result), _gradients(_recorded(result), targets)

    actual, gradients = evaluate(False)
    expected, expected_gradients = evaluate(True)
    _close(actual, expected)
    _close(gradients, expected_gradients)


def test_fresh_checkpointed_calls_do_not_reuse_graphs_and_respect_outer_no_grad():
    model = _model()
    runner = _runner(model, checkpoint_every=2)
    runner.initialization_inputs.v_init.requires_grad_()
    optimizer = torch.optim.Adam(
        [p for p in runner.parameters.values() if p.requires_grad]
        + [runner.initialization_inputs.v_init],
        lr=1.0e-4,
    )
    with torch.no_grad():
        assert not _recorded(runner())["v"].requires_grad
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        recorded = _recorded(runner())
        _loss(recorded).backward()
        assert runner.initialization_inputs.v_init.grad is not None
        assert torch.isfinite(runner.initialization_inputs.v_init.grad).all()
        optimizer.step()
        assert runner._prepared is None


def test_fresh_checkpoint_replay_rejects_changed_model_parameters(cpu_threads):
    model = _model()
    runner = _runner(model, checkpoint_every=2, backend="aot_eager")
    recorded = _recorded(runner())
    with torch.no_grad():
        runner.parameters[GNABAR].add_(0.001)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"parameters changed after (checkpoint forward|prepare)",
    ):
        torch.autograd.grad(_loss(recorded), runner.parameters[GNABAR])
