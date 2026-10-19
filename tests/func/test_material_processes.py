"""Functional built-in MaterialProcesses share imperative lifecycle semantics.

Complete recorded state is checked against actual imperative execution. An
independent field recurrence additionally checks phase order, duplicate writes,
and source application; explicit functional recurrence supplies the AD oracle.
Only two tiny configurations per family use real CPU Inductor.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import ClampProcess, ClearanceProcess
from tests.func.test_clearance_process import (
    DT,
    NAI,
    POOL,
    PRODUCTION,
    STEPS,
    _clone,
    _close,
    _explicit_reference,
    _key,
    _loss,
    _MaterialSourceCurrent,
    _runner,
)
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _assert_tree_close,
    _module_snapshot,
)
from tests.func.test_initialization_parity import (
    _constant_target,
    _live_state,
    _read_path,
    _record_paths,
)
from tests.func.test_membrane_current import CHANNELS
from tests.func.test_membrane_current import _model as _membrane_model

pytestmark = pytest.mark.cpu


class _BeforeClamp(ClearanceProcess):
    ClearanceProcess.GLOBAL(rate=0.2, target=1.65)
    ClearanceProcess.CLEAR("pool", field="amount", rate="rate", target="target")
    ClearanceProcess.CLEAR("na", field="nai", rate=0.23, target=8.1)


class _BathClamp(ClampProcess):
    ClampProcess.GLOBAL(value=1.62, lower=1.35, upper=1.9, sodium=8.6)
    ClampProcess.SET("pool", field="amount", value="value", where=0)
    # The second write must read the result of the first, not the old field.
    ClampProcess.BOUNDS("pool", field="amount", lower="lower", upper="upper")
    ClampProcess.SET("na", field="nai", value="sodium", where=-1)


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


def _model(
    kind="path",
    *,
    family="clamp",
    device="cpu",
    dtype=torch.float64,
    shell_resistance_scale=1.0,
    process=None,
):
    assert family == "clamp"
    model = _membrane_model(kind, device=device, dtype=dtype, batch=None)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True, DEVICE=device, DTYPE=dtype):
        components = (
            model.populations.values() if kind.startswith("multi_") else (model,)
        )
        for component in components:
            if hasattr(component, "xraxial"):
                component.xraxial.mul_(shell_resistance_scale)
        initial = torch.linspace(
            1.1, 2.1, model.v.numel(), device=device, dtype=dtype
        ).reshape(model.shape)
        model.material(
            "pool",
            fields={"amount": torch.nn.Parameter(initial)},
            min_values={"amount": 0.0},
            domain="intracellular",
        )
        model.concentrations(nai0=torch.nn.Parameter(model.v.new_tensor(9.25)))
        model.ion_style("na", 1, 1)
        model.insert(_MaterialSourceCurrent)
        # Reverse insertion order catches accidental insertion-order execution.
        model.insert(_BathClamp if process is None else process)
        model.insert(_BeforeClamp)
        model.batch(2)
        model.train()
        model.initialize()
        model.t.fill_(1.25)
    model._material_test_kind = kind
    model._material_test_family = family
    model._material_test_process = process
    model._material_test_shell_scale = shell_resistance_scale
    return model


def _arguments(runner, model):
    parameters = _clone(dict(runner.parameters))
    targets = {
        channel: parameters[
            _key(parameters, f"mechanisms.hh.{channel}_param")
        ].requires_grad_()
        for channel in CHANNELS
    }
    for name, value in parameters.items():
        if ".material_processes." in name:
            short = name.split(".material_processes.", 1)[1].removesuffix("_param")
            targets[short] = value.requires_grad_()
    for label, suffix in (
        ("initial_pool", "materials.pool._initial_sources.amount"),
        ("initial_nai", "ions.na.i_init"),
    ):
        targets[label] = parameters[_key(parameters, suffix)].requires_grad_()
    if "noded2" in parameters:
        targets["geometry"] = parameters["noded2"].requires_grad_()
    else:
        canonical = [
            name for name in runner.constants if name.endswith("canonical_area_cm2")
        ]
        geometry = (
            canonical[0]
            if canonical
            else next(name for name in runner.constants if name.endswith("diam"))
        )
        targets["geometry"] = runner.constants[geometry].requires_grad_()
    targets["dt"] = runner.constants["dt"].requires_grad_()
    voltage = torch.linspace(
        -72.0, -52.0, model.v.numel(), device=model.device(), dtype=model.dtype()
    ).reshape(model.shape)
    targets["initial_voltage"] = voltage.requires_grad_()
    amplitude = model.v.new_ones((), requires_grad=True)
    targets["stimulus"] = amplitude
    drive = torch.linspace(
        -2e-9, 3e-9, STEPS * model.v.numel(), device=model.device(), dtype=model.dtype()
    ).reshape(STEPS, *model.shape)
    ve = torch.linspace(
        -0.3, 0.4, drive.numel(), device=model.device(), dtype=model.dtype()
    ).reshape_as(drive)
    initialization = _clone(runner.initialization_inputs)._replace(v_init=voltage)
    return (
        parameters,
        initialization,
        dn.func.RolloutInput(ve, drive * amplitude),
        targets,
    )


def _gradients(recorded, targets):
    values = torch.autograd.grad(_loss(recorded), tuple(targets.values()))
    gradients = dict(zip(targets, values, strict=True))
    for name, value in gradients.items():
        assert torch.isfinite(value).all() and torch.count_nonzero(value), name
    return gradients


def _imperative_reference(
    model, runner, parameters, initialization, inputs, *, steps=STEPS
):
    reference = _model(
        model._material_test_kind,
        family=model._material_test_family,
        device=model.device(),
        dtype=model.dtype(),
        process=model._material_test_process,
        shell_resistance_scale=model._material_test_shell_scale,
    )
    with torch.no_grad():
        raw = dict(reference.named_parameters())
        for name, value in parameters.items():
            raw[name].copy_(value)
        for name, value in runner.constants.items():
            if name != "dt" and not name.startswith("parametrizations."):
                target = (
                    _constant_target(reference.mech, name)
                    if name.startswith("material_processes.")
                    else _constant_target(reference, name)
                )
                if not torch.equal(target, value):
                    target.copy_(value)
        reference.set_v_init(initialization.v_init)
        reference.initialize()
        paths = _record_paths(runner.functional)
        state = _live_state(runner.functional, reference)
        history = [{name: _read_path(state, path) for name, path in paths.items()}]
        dt = runner.constants["dt"].detach()
        reference.integrator._initialize(
            reference, dt, force=True, compile_scope="population"
        )
        for index in range(steps):
            reference.integrator.step(
                reference, dt, inputs.ve[index], inputs.intra[index]
            )
            reference.t = reference.t + dt
            state = _live_state(runner.functional, reference)
            history.append(
                {name: _read_path(state, path) for name, path in paths.items()}
            )
    return {name: torch.stack([frame[name] for frame in history]) for name in paths}


def _parameter(parameters, owner, name):
    return parameters[_key(parameters, f"material_processes.{owner}.{name}_param")]


def _material_oracle(recorded, parameters, dt, *, family="clamp"):
    assert family == "clamp"
    pool, sodium = recorded[POOL][0], recorded[NAI][0]
    pools, ions = [pool], [sodium]
    root = torch.zeros_like(pool, dtype=torch.bool)
    root[..., 0] = True
    end = torch.zeros_like(sodium, dtype=torch.bool)
    end[..., -1] = True
    for _ in range(recorded[POOL].shape[0] - 1):
        pool = pool + dt * PRODUCTION * (1.0 + 0.01 * sodium)
        rate = _parameter(parameters, "_BeforeClamp", "rate")
        target = _parameter(parameters, "_BeforeClamp", "target")
        pool = target + (pool - target) * torch.exp(-dt * rate)
        sodium = 8.1 + (sodium - 8.1) * torch.exp(-dt * 0.23)
        pool = torch.where(root, _parameter(parameters, "_BathClamp", "value"), pool)
        pool = torch.maximum(
            torch.minimum(pool, _parameter(parameters, "_BathClamp", "upper")),
            _parameter(parameters, "_BathClamp", "lower"),
        )
        sodium = torch.where(
            end, _parameter(parameters, "_BathClamp", "sodium"), sodium
        )
        pools.append(pool)
        ions.append(sodium)
    _close(
        {POOL: recorded[POOL], NAI: recorded[NAI]},
        {POOL: torch.stack(pools), NAI: torch.stack(ions)},
        dtype=recorded[POOL].dtype,
    )


def _exercise(
    kind="path",
    *,
    family="clamp",
    device="cpu",
    dtype=torch.float64,
    shell_resistance_scale=1.0,
    backend="eager",
    execution="scan",
    checkpoint_every=None,
    imperative=True,
):
    model = _model(
        kind,
        family=family,
        device=device,
        dtype=dtype,
        shell_resistance_scale=shell_resistance_scale,
    )
    runner = _runner(
        model, backend=backend, execution=execution, checkpoint_every=checkpoint_every
    )
    source = _module_snapshot(model)
    parameters, initialization, inputs, targets = _arguments(runner, model)
    result = runner(parameters=parameters, initialization=initialization, inputs=inputs)
    actual = dict(result[1]["callbacks"]["trace"])
    gradients = _gradients(actual, targets) if execution == "scan" else {}
    (
        reference_parameters,
        reference_initialization,
        reference_inputs,
        reference_targets,
    ) = _arguments(runner, model)
    expected = _explicit_reference(
        runner, reference_parameters, reference_initialization, reference_inputs
    )
    _assert_tree_close(
        result[0],
        expected[0],
        rtol=2e-4 if dtype == torch.float32 else 3e-9,
        atol=2e-5 if dtype == torch.float32 else 3e-10,
    )
    _close(actual, dict(expected[1]["callbacks"]["trace"]), dtype=dtype)
    if execution == "scan":
        _close(
            gradients,
            _gradients(dict(expected[1]["callbacks"]["trace"]), reference_targets),
            dtype=dtype,
        )
        assert len(gradients) == 18
    if imperative:
        _close(
            actual,
            _imperative_reference(model, runner, parameters, initialization, inputs),
            dtype=dtype,
        )
    _material_oracle(actual, parameters, runner.constants["dt"], family=family)
    torch.testing.assert_close(
        actual[POOL][0],
        parameters[
            _key(parameters, "materials.pool._initial_sources.amount")
        ].expand_as(actual[POOL][0]),
    )
    torch.testing.assert_close(
        actual[NAI][0],
        parameters[_key(parameters, "ions.na.i_init")].expand_as(actual[NAI][0]),
    )
    assert actual["v"].shape == (STEPS + 1, *model.shape)
    assert actual["t"][0].item() == 0
    assert torch.count_nonzero(actual["i_membrane"][0]) == 0
    assert {
        POOL,
        NAI,
        "na_ion.ena",
        "na_ion.ina",
        "hh.m",
        "hh.h",
        "hh.n",
    } <= actual.keys()
    _assert_module_snapshot(model, source)
    return actual, gradients


@pytest.mark.parametrize(
    "kind", ("point", "path", "block_path", "multi_scalar", "multi_block")
)
def test_clamp_full_traces_and_all_gradients(kind, cpu_threads):
    _exercise(kind)


@pytest.mark.parametrize("kind", ("path", "multi_block"))
def test_clamp_checkpointed_continuation_full_traces_and_all_gradients(
    kind, cpu_threads
):
    model = _model(kind)
    runner = _runner(model, checkpoint_every=2)
    parameters, initialization, inputs, targets = _arguments(runner, model)
    state, auxiliary = runner(
        parameters=parameters, initialization=initialization, inputs=inputs
    )
    final, continued = runner(
        state,
        parameters=auxiliary["parameters"],
        inputs=inputs,
        callback_state=auxiliary["callbacks"].state,
    )
    recorded = {
        name: torch.cat((value, continued["callbacks"]["trace"][name]))
        for name, value in auxiliary["callbacks"]["trace"].items()
    }
    gradients = _gradients(recorded, targets)
    (
        reference_parameters,
        reference_initialization,
        reference_inputs,
        reference_targets,
    ) = _arguments(runner, model)
    doubled = dn.func.RolloutInput(
        *(torch.cat((value, value)) for value in reference_inputs)
    )
    expected = _explicit_reference(
        runner, reference_parameters, reference_initialization, doubled, steps=2 * STEPS
    )
    _assert_tree_close(final, expected[0])
    expected_recorded = dict(expected[1]["callbacks"]["trace"])
    _close(recorded, expected_recorded)
    _close(gradients, _gradients(expected_recorded, reference_targets))
    _close(
        recorded,
        _imperative_reference(
            model,
            runner,
            parameters,
            initialization,
            dn.func.RolloutInput(*(torch.cat((value, value)) for value in inputs)),
            steps=2 * STEPS,
        ),
    )
    _material_oracle(recorded, parameters, runner.constants["dt"])
    assert recorded["v"].shape[0] == 2 * STEPS + 1


def test_clamp_snapshot_restarts_preserve_unclamped_initial_fields(cpu_threads):
    model = _model("point")
    source = _module_snapshot(model)
    runner = _runner(model, initialization="snapshot")
    first, second = runner(), runner()
    _assert_tree_close(first[0], second[0])
    _close(dict(first[1]["callbacks"]["trace"]), dict(second[1]["callbacks"]["trace"]))
    trace = first[1]["callbacks"]["trace"]
    assert trace["t"][0].item() == 1.25
    torch.testing.assert_close(trace[POOL][0], model.mech.materials["pool"].amount)
    assert trace[POOL][0].min() < 1.35 and trace[POOL][0].max() > 1.9
    _assert_module_snapshot(model, source)


def test_clamp_bound_initial_field_and_dt_gradients_match_central_differences(
    cpu_threads,
):
    model = _model("point")
    runner = _runner(model)
    parameters, initialization, inputs, targets = _arguments(runner, model)
    recorded = dict(
        runner(parameters=parameters, initialization=initialization, inputs=inputs)[1][
            "callbacks"
        ]["trace"]
    )
    gradients = _gradients(recorded, targets)
    for label in (
        "_BathClamp.value",
        "_BathClamp.lower",
        "_BathClamp.upper",
        "_BathClamp.sodium",
        "initial_pool",
        "initial_voltage",
        "dt",
    ):
        leaf = targets[label]
        step = 1e-5 if label != "dt" else 1e-6
        values = []
        for sign in (1, -1):
            with torch.no_grad():
                leaf.add_(sign * step)
            values.append(
                _loss(
                    _imperative_reference(
                        model, runner, parameters, initialization, inputs
                    )
                )
            )
            with torch.no_grad():
                leaf.sub_(sign * step)
        torch.testing.assert_close(
            gradients[label].sum(),
            (values[0] - values[1]) / (2 * step),
            rtol=3e-5,
            atol=2e-7,
        )


@pytest.mark.parametrize(
    ("kind", "mode", "where"),
    (
        ("point", "set", "all"),
        ("point", "min", "root"),
        ("point", "max", "terminal"),
        ("point", "range", slice(1, None)),
        ("point", "set", torch.tensor([True, False, True])),
        ("point", "min", 1),
        ("point", "set", False),
        ("point", "set", True),
        ("tree", "set", "root"),
        ("tree", "max", "terminal"),
    ),
)
def test_clamp_modes_regions_and_piecewise_gradients(kind, mode, where, cpu_threads):
    class RegionClamp(ClampProcess):
        ClampProcess.GLOBAL(value=1.6, lower=1.35, upper=1.9)
        ClampProcess.CLAMP(
            "pool",
            field="amount",
            mode=mode,
            where=where,
            value=("lower", "upper") if mode == "range" else "value",
        )

    model = _model(kind, process=RegionClamp)
    runner = _runner(model)
    parameters, initialization, inputs, _targets = _arguments(runner, model)
    actual = dict(
        runner(parameters=parameters, initialization=initialization, inputs=inputs)[1][
            "callbacks"
        ]["trace"]
    )
    pool, sodium = actual[POOL][0], actual[NAI][0]
    mask = torch.zeros_like(pool, dtype=torch.bool)
    if isinstance(where, str):
        if where == "all":
            mask.fill_(True)
        elif getattr(model, "graph", None) is not None:
            graph = (
                model.graph[0]
                if isinstance(model.graph, (list, tuple))
                else model.graph
            )
            if where == "root":
                indices = [int(n) for n in graph.nodes if graph.in_degree(n) == 0]
            else:
                indices = [int(n) for n in graph.nodes if graph.out_degree(n) == 0]
            mask[..., indices] = True
        elif where == "root":
            mask[..., 0] = True
        else:
            mask[..., (0, -1)] = True
    elif torch.is_tensor(where):
        mask = where.to(device=pool.device).expand_as(pool)
    elif isinstance(where, bool):
        mask.fill_(where)
    else:
        mask[..., where] = True
    dt = runner.constants["dt"]
    history = [pool]
    for _ in range(STEPS):
        pool = pool + dt * PRODUCTION * (1 + 0.01 * sodium)
        target = _parameter(parameters, "_BeforeClamp", "target")
        rate = _parameter(parameters, "_BeforeClamp", "rate")
        pool = target + (pool - target) * torch.exp(-rate * dt)
        sodium = 8.1 + (sodium - 8.1) * torch.exp(-0.23 * dt)
        value = _parameter(parameters, "RegionClamp", "value")
        if mode == "range":
            clamped = torch.maximum(
                torch.minimum(pool, _parameter(parameters, "RegionClamp", "upper")),
                _parameter(parameters, "RegionClamp", "lower"),
            )
        elif mode == "min":
            clamped = torch.maximum(pool, value)
        elif mode == "max":
            clamped = torch.minimum(pool, value)
        else:
            clamped = value.expand_as(pool)
        pool = torch.where(mask, clamped, pool)
        history.append(pool)
    expected = torch.stack(history)
    _close(
        actual, _imperative_reference(model, runner, parameters, initialization, inputs)
    )
    torch.testing.assert_close(actual[POOL], expected)
    names = ("lower", "upper") if mode == "range" else ("value",)
    leaves = tuple(_parameter(parameters, "RegionClamp", name) for name in names)
    actual_derivatives = torch.autograd.grad(
        actual[POOL].sum(), leaves, retain_graph=True
    )
    reference_derivatives = torch.autograd.grad(expected.sum(), leaves)
    for result, reference in zip(
        actual_derivatives, reference_derivatives, strict=True
    ):
        torch.testing.assert_close(result, reference)
        assert torch.isfinite(result).all()


@pytest.mark.parametrize("mode", ("min", "max"))
def test_clamp_at_bound_preserves_torch_half_gradient_convention(mode, cpu_threads):
    # Hard bounds are nonsmooth. At equality maximum/minimum assign half the
    # derivative to each operand; away from equality the active branch wins.
    class TieClamp(ClampProcess):
        ClampProcess.GLOBAL(value=1.6)
        ClampProcess.CLAMP("pool", field="amount", value="value", mode=mode)

    model = _model("point", process=TieClamp)
    process = model.mech.material_processes["TieClamp"]
    c = torch.full_like(model.mech.materials["pool"].amount, 1.6, requires_grad=True)
    value = process.value_param
    prepared = process.prepare_materials({("pool", "amount"): c}, c.new_tensor(DT))
    output = process.material_updates({("pool", "amount"): c}, prepared)[
        ("pool", "amount")
    ]
    dc, dbound = torch.autograd.grad(output.sum(), (c, value))
    torch.testing.assert_close(dc, torch.full_like(c, 0.5))
    torch.testing.assert_close(dbound, torch.full_like(value, 0.5 * c.numel()))


def test_clamp_literal_values_are_explicit_differentiable_constants(cpu_threads):
    literal = torch.tensor(1.6, dtype=torch.float64)

    class LiteralClamp(ClampProcess):
        ClampProcess.SET("pool", field="amount", value=literal, where=1)

    model = _model("point", process=LiteralClamp)
    runner = _runner(model)
    name = next(name for name in runner.constants if "_clamp_literal_" in name)
    constant = runner.constants[name]
    assert constant is not literal
    constant.requires_grad_()
    result = runner()
    recorded = dict(result[1]["callbacks"]["trace"])
    expected = literal.expand_as(recorded[POOL][1:, ..., 1])
    torch.testing.assert_close(recorded[POOL][1:, ..., 1], expected)
    derivative = torch.autograd.grad(recorded[POOL].sum(), constant)[0]
    torch.testing.assert_close(
        derivative, derivative.new_tensor(STEPS * model.shape[0] * model.shape[1])
    )
    torch.testing.assert_close(literal, literal.new_tensor(1.6))


def test_clamp_vmap_empty_vmap_and_jvp_preserve_full_state(cpu_threads):
    model = _model("point")
    runner = _runner(model)
    parameters, initialization, inputs, _targets = _arguments(runner, model)
    functional = runner.functional
    key = _key(parameters, "material_processes._BathClamp.value_param")

    def rollout(value, voltage, drive):
        initialized = functional.initialize(
            {**parameters, key: value},
            runner.constants,
            initialization._replace(v_init=voltage),
        )
        _, auxiliary = functional.prepare_and_rollout(
            initialized.parameters,
            initialized.constants,
            initialized.state,
            steps=STEPS,
            inputs=inputs._replace(intra=drive),
            callbacks=runner._callbacks,
        )
        return dict(auxiliary["callbacks"]["trace"])

    def objective(value, voltage, drive):
        return _loss(rollout(value, voltage, drive))

    value, voltage, drive = parameters[key], initialization.v_init, inputs.intra
    values = torch.stack((value * 0.95, value * 1.05))
    voltages = torch.stack((voltage - 0.4, voltage + 0.6))
    drives = torch.stack((drive * 0.9, drive * 1.1))
    actual = torch.func.vmap(rollout)(values, voltages, drives)
    expected = pytree.tree_map(
        lambda *leaves: torch.stack(leaves),
        *[rollout(values[i], voltages[i], drives[i]) for i in range(2)],
    )
    _assert_tree_close(actual, expected, rtol=3e-9, atol=3e-10)
    empty = torch.func.vmap(rollout)(values[:0], voltages[:0], drives[:0])
    assert empty[POOL].shape == (0, STEPS + 1, *model.shape)
    derivative = torch.func.grad(objective, argnums=(0, 1, 2))
    expected_gradients = pytree.tree_map(
        lambda *leaves: torch.stack(leaves),
        *[derivative(values[i], voltages[i], drives[i]) for i in range(2)],
    )
    _assert_tree_close(
        torch.func.vmap(derivative)(values, voltages, drives), expected_gradients
    )

    def material_sum(value, voltage, drive):
        return rollout(value, voltage, drive)[POOL].sum()

    empty_gradients = torch.func.vmap(torch.func.grad(material_sum, argnums=(0, 1, 2)))(
        values[:0], voltages[:0], drives[:0]
    )
    assert all(value.shape[0] == 0 for value in empty_gradients)
    tangents = (
        torch.ones_like(value) * 0.01,
        torch.ones_like(voltage) * 0.01,
        torch.ones_like(drive) * 1e-10,
    )
    _, tangent = torch.func.jvp(objective, (value, voltage, drive), tangents)
    gradients = derivative(value, voltage, drive)
    reference = sum(
        (gradient * direction).sum()
        for gradient, direction in zip(gradients, tangents, strict=True)
    )
    torch.testing.assert_close(tangent, reference, rtol=3e-9, atol=3e-10)


def test_custom_clamp_update_fails_closed():
    class CustomClamp(_BathClamp):
        def material_updates(self, fields, prepared):
            return super().material_updates(fields, prepared)

    model = _model("point", process=CustomClamp)
    with pytest.raises(dn.func.FunctionalizationError):
        dn.func.make_functional(model, dt=DT)


@pytest.mark.parametrize("kind", ("path", "multi_block"))
def test_clamp_commit_then_imperative_continuation(kind, cpu_threads):
    model = _model(kind)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    state, _ = functional.step(tensors.parameters, prepared, tensors.state)
    expected, _ = functional.step(tensors.parameters, prepared, state)
    target = _model(kind)
    functional.commit_state_(target, state)
    dt = tensors.constants["dt"]
    target.integrator._initialize(target, dt, force=True, compile_scope="population")
    target.integrator.step(target, dt, None, None)
    target.t = target.t + dt
    _assert_tree_close(_live_state(functional, target), expected, rtol=3e-9, atol=3e-10)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("kind", "execution", "checkpoint_every"),
    (("path", "scan", 2), ("multi_block", "while", None)),
)
def test_inductor_clamp_scan_gradients_and_while_full_traces(
    kind, execution, checkpoint_every, cpu_threads
):
    _exercise(
        kind, backend="inductor", execution=execution, checkpoint_every=checkpoint_every
    )
