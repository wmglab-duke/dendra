"""Voltage-skip transitions preserve exposed voltage, owned carry, and currents."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, VoltageProcess
from dendra.models.mod import pas
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _assert_tree_close,
    _module_snapshot,
)
from tests.func.test_initialization_parity import _live_state

DT = 0.05
STEPS = 4
pytestmark = pytest.mark.cpu


class _OwnedVoltage(VoltageProcess):
    VoltageProcess.GLOBAL(offset=1.25, drift=0.7, feedback=0.03)
    VoltageProcess.CARRY("owned", "ticks")

    def initial_values(self, v, values):
        del values
        return {"owned": v + self.offset, "ticks": torch.zeros_like(v)}

    def voltage_values(self, v, dt, values):
        del v, dt
        return values["owned"], {}

    def advance(self, v, dt, values):
        return {
            "owned": values["owned"] + dt * (self.drift + self.feedback * (v + 60)),
            "ticks": values["ticks"] + 1,
        }


class _MaterialCurrent(Mechanism):
    Mechanism.GLOBAL(g=0.002, ena=-58.0)
    Mechanism.USEMATERIAL("pool", read=["amount"])
    Mechanism.USEION("na", write=["ina"])
    Mechanism.AFFINE("ina")

    def ina(self, v):
        return self.g * (1 + 0.1 * self.amount) * (v - self.ena)


class _AcceptedCurrent(Mechanism):
    Mechanism.GLOBAL(production=0.2)
    Mechanism.CARRY("seen")
    Mechanism.USEION("na", read=["ina"])
    Mechanism.USEMATERIAL("pool", source={"amount": "delta"})

    def initial_values(self, v, values):
        del values
        return {"seen": torch.zeros_like(v)}

    def advance(self, v, dt, values):
        del v
        return {"seen": values["ina"], "delta": dt * self.production * values["ina"]}


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _key(parameters, suffix):
    names = [name for name in parameters if name.endswith(suffix)]
    assert len(names) == 1, (suffix, names)
    return names[0]


def _model(*, kind="population", batch=2, imem=True, device="cpu", dtype=torch.float64):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE=device, DTYPE=dtype, IMEM=imem):
        cls = dn.Population if kind == "population" else dn.SingleCompartment
        model = cls(N=1, C=3, dtype=dtype, integrator=dn.scnv(imem=imem))
        model.material("pool", fields={"amount": 0.6}, domain="intracellular")
        model.insert(_OwnedVoltage.rename("driver"))
        model.insert(_MaterialCurrent.rename("channel"))
        model.insert(_AcceptedCurrent.rename("probe"))
        if batch is not None:
            model.batch(batch)
        model.set_v_init(
            torch.linspace(
                -65, -61, model.v.numel(), device=device, dtype=dtype
            ).reshape(model.shape)
        )
        model.train()
        model.initialize()
    return model


def _observe(state):
    result = {
        "v": state["integrator"]["v"],
        "driver.owned": state["mechanism_buffers"]["driver"]["owned"],
        "driver.ticks": state["mechanism_buffers"]["driver"]["ticks"],
        "probe.seen": state["mechanism_buffers"]["probe"]["seen"],
        "pool_material.amount": state["materials"]["pool"]["amount"],
        "na_ion.ina": state["ions"]["na"]["ina"],
        "t": state["clock"]["t"],
    }
    if "i_membrane" in state["integrator"]:
        result["i_membrane"] = state["integrator"]["i_membrane"]
    return result


def _reference(model, parameters, constants, state, steps):
    """Independent imposed-voltage recurrence; no Dendra stepping or kernels."""
    frame = _observe(state)
    frames = [frame]
    dt = constants["dt"]
    drift = parameters[_key(parameters, ".driver.drift_param")]
    feedback = parameters[_key(parameters, ".driver.feedback_param")]
    conductance = parameters[_key(parameters, ".channel.g_param")]
    reversal = parameters[_key(parameters, ".channel.ena_param")]
    production = parameters[_key(parameters, ".probe.production_param")]
    cm = F.softplus(parameters[_key(parameters, "cm_param.rho")])
    cm = cm * F.softplus(parameters[_key(parameters, "cm_scale_param.rho")])
    area = torch.pi * constants["diam"] * constants["dx"] * 1e-8
    area = area * F.softplus(parameters[_key(parameters, "area_scale_param.rho")])
    for _ in range(steps):
        exposed = frame["driver.owned"]
        current = (
            conductance
            * (1 + 0.1 * frame["pool_material.amount"])
            * (exposed - reversal)
        )
        next_frame = {
            "v": exposed,
            "driver.owned": exposed + dt * (drift + feedback * (exposed + 60)),
            "driver.ticks": frame["driver.ticks"] + 1,
            "probe.seen": current,
            "pool_material.amount": frame["pool_material.amount"]
            + dt * production * current,
            "na_ion.ina": current,
            "t": frame["t"] + dt,
        }
        if model.integrator.imem:
            next_frame["i_membrane"] = (
                1e-3 * cm / dt * (exposed - frame["v"]) + current
            ) * area
        frames.append(next_frame)
        frame = next_frame
    return {name: torch.stack([frame[name] for frame in frames]) for name in frames[0]}


def _loss(trace):
    result = (
        0.001 * trace["v"].square().mean()
        + 0.001 * trace["driver.owned"].square().mean()
    )
    result = (
        result
        + trace["pool_material.amount"].square().mean()
        + trace["probe.seen"].square().mean()
    )
    if "i_membrane" in trace:
        result = result + (1e6 * trace["i_membrane"]).square().mean()
    return result


def _close(actual, expected, *, dtype=torch.float64):
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        other = expected[name]
        if name == "i_membrane":
            value, other = value * 1e6, other * 1e6
        torch.testing.assert_close(
            value,
            other,
            rtol=3e-4 if dtype == torch.float32 else 2e-9,
            atol=2e-5 if dtype == torch.float32 else 2e-10,
            msg=name,
        )


def _runner(
    model,
    *,
    steps=STEPS,
    initialization="fresh",
    backend="eager",
    execution="scan",
    checkpoint_every=None,
):
    functional, tensors = dn.func.make_functional(model, dt=DT)
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
        steps=steps,
        initialization=initialization,
        backend=backend,
        execution=execution,
        inference=execution == "while",
        checkpoint_every=checkpoint_every,
        callbacks={"trace": dn.func.Recorder(tuple(_observe(tensors.state)))},
        **options,
    )


def _exercise(
    *,
    device="cpu",
    dtype=torch.float64,
    kind="population",
    imem=True,
    backend="eager",
    execution="scan",
    checkpoint_every=None,
):
    model = _model(device=device, dtype=dtype, kind=kind, imem=imem)
    snapshot = _module_snapshot(model)
    runner = _runner(
        model, backend=backend, execution=execution, checkpoint_every=checkpoint_every
    )
    parameters = _clone(dict(runner.parameters))
    targets = [
        parameters[_key(parameters, suffix)].requires_grad_()
        for suffix in (
            ".driver.drift_param",
            ".driver.offset_param",
            ".channel.g_param",
            ".probe.production_param",
            "cm_param.rho",
        )
    ]
    initialization = _clone(runner.initialization_inputs)
    targets.append(initialization.v_init.requires_grad_())
    state, auxiliary = runner(parameters=parameters, initialization=initialization)
    trace = dict(auxiliary["callbacks"]["trace"])
    initialized = runner.functional.initialize(
        parameters, runner.constants, initialization
    )
    reference = _reference(
        model, initialized.parameters, initialized.constants, initialized.state, STEPS
    )
    _close(trace, reference, dtype=dtype)
    assert not torch.equal(
        state["integrator"]["v"], state["mechanism_buffers"]["driver"]["owned"]
    )
    assert tuple(model.integrator.v_vars) == ()
    if execution == "scan":
        actual_gradients = torch.autograd.grad(
            _loss(trace), targets, retain_graph=True, allow_unused=True
        )
        expected_gradients = torch.autograd.grad(
            _loss(reference), targets, allow_unused=True
        )
        actual_gradients = tuple(
            torch.zeros_like(target) if value is None else value
            for target, value in zip(targets, actual_gradients, strict=True)
        )
        expected_gradients = tuple(
            torch.zeros_like(target) if value is None else value
            for target, value in zip(targets, expected_gradients, strict=True)
        )
        for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
            torch.testing.assert_close(
                actual,
                expected,
                rtol=3e-4 if dtype == torch.float32 else 2e-8,
                atol=2e-5 if dtype == torch.float32 else 2e-10,
            )
            assert torch.isfinite(actual).all()
        assert torch.count_nonzero(actual_gradients[0])
    else:
        assert not trace["v"].requires_grad
    _assert_module_snapshot(model, snapshot)
    return trace


@pytest.mark.parametrize("kind,imem", (("population", True), ("single", False)))
def test_scnv_independent_full_trace_and_gradient_oracle(kind, imem):
    _exercise(kind=kind, imem=imem)


def test_scnv_float32_full_trace_and_gradient_oracle():
    _exercise(dtype=torch.float32)


def test_scnv_currentless_voltage_process_has_only_capacitive_membrane_current():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True):
        model = dn.Population(
            N=1, C=2, dtype=torch.float64, integrator=dn.scnv(imem=True)
        )
        model.insert(_OwnedVoltage.rename("driver"))
        model.set_v_init(torch.tensor([[-65.0, -61.0]], dtype=torch.float64))
        model.train()
        model.initialize()
    source = _module_snapshot(model)
    assert not model.mech.currents and not model.mech.ions and not model.mech.materials
    assert model.integrator._functional_solver() is None
    functional, tensors = dn.func.make_functional(model, dt=DT)
    actual, _ = functional.prepare_and_rollout(
        tensors.parameters, tensors.constants, tensors.state, steps=3
    )

    parameters, constants = tensors.parameters, tensors.constants
    voltage = model.v.detach().clone()
    owned = voltage + parameters[_key(parameters, ".driver.offset_param")]
    drift = parameters[_key(parameters, ".driver.drift_param")]
    feedback = parameters[_key(parameters, ".driver.feedback_param")]
    capacitance = F.softplus(parameters[_key(parameters, "cm_param.rho")])
    capacitance = capacitance * F.softplus(
        parameters[_key(parameters, "cm_scale_param.rho")]
    )
    area = torch.pi * constants["diam"] * constants["dx"] * 1e-8
    area = area * F.softplus(parameters[_key(parameters, "area_scale_param.rho")])
    for _ in range(3):
        exposed = owned
        membrane_current = (
            1e-3 * capacitance / constants["dt"] * (exposed - voltage) * area
        )
        owned = owned + constants["dt"] * (drift + feedback * (exposed + 60))
        voltage = exposed
    torch.testing.assert_close(actual["integrator"]["v"], voltage)
    torch.testing.assert_close(actual["mechanism_buffers"]["driver"]["owned"], owned)
    torch.testing.assert_close(
        actual["mechanism_buffers"]["driver"]["ticks"], torch.full_like(voltage, 3)
    )
    torch.testing.assert_close(
        actual["integrator"]["i_membrane"] * 1e6,
        membrane_current * 1e6,
        rtol=2e-9,
        atol=2e-10,
    )
    assert not torch.equal(actual["integrator"]["v"], owned)
    _assert_module_snapshot(model, source)


def test_scnv_imperative_trace_and_commit_resume_preserve_both_voltage_frames():
    model = _model()
    runner = _runner(model, initialization="snapshot")
    source = _module_snapshot(model)
    state, auxiliary = runner()
    reference = _model()
    dt = reference.v.new_tensor(DT)
    reference.integrator._initialize(
        reference, dt, force=True, compile_scope="population"
    )
    frames = [_observe(_live_state(runner.functional, reference))]
    for _ in range(STEPS):
        reference.integrator.step(reference, dt)
        reference.t = reference.t + dt
        frames.append(_observe(_live_state(runner.functional, reference)))
    _close(
        dict(auxiliary["callbacks"]["trace"]),
        {name: torch.stack([frame[name] for frame in frames]) for name in frames[0]},
    )
    _assert_module_snapshot(model, source)
    runner.functional.commit_state_(model, state)
    torch.testing.assert_close(model.v, state["integrator"]["v"])
    torch.testing.assert_close(
        model.mech.driver.owned, state["mechanism_buffers"]["driver"]["owned"]
    )
    assert not torch.equal(model.v, model.mech.driver.owned)
    _assert_tree_close(
        _live_state(runner.functional, model), state, rtol=2e-9, atol=2e-10
    )
    expected, _ = runner.functional.prepare_and_step(
        runner.parameters, runner.constants, state
    )
    model.integrator._initialize(model, dt, force=True, compile_scope="population")
    model.integrator.step(model, dt)
    model.t = model.t + dt
    _assert_tree_close(
        _live_state(runner.functional, model), expected, rtol=2e-9, atol=2e-10
    )


def _exercise_continuation(*, device="cpu", dtype=torch.float64):
    model = _model(device=device, dtype=dtype)
    runner = _runner(model, steps=2)
    source = _module_snapshot(model)
    state, _ = runner()
    state = _clone(state)
    state["integrator"]["v"] = (state["integrator"]["v"] - 3).requires_grad_()
    state["mechanism_buffers"]["driver"]["owned"] = (
        state["mechanism_buffers"]["driver"]["owned"] + 2
    ).requires_grad_()
    targets = (state["integrator"]["v"], state["mechanism_buffers"]["driver"]["owned"])
    _, auxiliary = runner(state)
    actual = dict(auxiliary["callbacks"]["trace"])
    expected = _reference(model, runner.parameters, runner.constants, state, 2)
    _close(actual, expected, dtype=dtype)
    for value, original in zip(
        torch.autograd.grad(_loss(actual), targets),
        torch.autograd.grad(_loss(expected), targets),
        strict=True,
    ):
        torch.testing.assert_close(
            value,
            original,
            rtol=3e-4 if dtype == torch.float32 else 2e-9,
            atol=2e-5 if dtype == torch.float32 else 2e-10,
        )
        assert torch.count_nonzero(value)
    _assert_module_snapshot(model, source)


def test_scnv_explicit_continuation_keeps_independent_public_voltage_and_owned_carry():
    _exercise_continuation()


def test_scnv_advanced_snapshot_and_changed_fresh_initialization_are_distinct():
    model = _model()
    dt = model.v.new_tensor(DT)
    model.integrator._initialize(model, dt, force=True, compile_scope="population")
    for _ in range(2):
        model.integrator.step(model, dt)
        model.t = model.t + dt
    snapshot_runner = _runner(model, initialization="snapshot", steps=1)
    fresh_runner = _runner(model, initialization="fresh", steps=1)
    source = _module_snapshot(model)
    _, snapshot_aux = snapshot_runner()
    snapshot_trace = snapshot_aux["callbacks"]["trace"]
    torch.testing.assert_close(snapshot_trace["v"][0], model.v)
    torch.testing.assert_close(
        snapshot_trace["driver.owned"][0], model.mech.driver.owned
    )
    torch.testing.assert_close(
        snapshot_trace["driver.ticks"][0], model.mech.driver.ticks
    )
    assert not torch.equal(snapshot_trace["v"][0], snapshot_trace["driver.owned"][0])

    parameters = _clone(dict(fresh_runner.parameters))
    offset = _key(parameters, ".driver.offset_param")
    parameters[offset] = parameters[offset] + 0.4
    initial = _clone(fresh_runner.initialization_inputs)
    initial = initial._replace(v_init=initial.v_init + 0.3)
    _, fresh_aux = fresh_runner(parameters=parameters, initialization=initial)
    fresh_trace = fresh_aux["callbacks"]["trace"]
    torch.testing.assert_close(fresh_trace["v"][0], initial.v_init)
    torch.testing.assert_close(
        fresh_trace["driver.owned"][0], initial.v_init + parameters[offset]
    )
    assert torch.count_nonzero(fresh_trace["driver.ticks"][0]) == 0
    assert torch.count_nonzero(fresh_trace["probe.seen"][0]) == 0
    assert fresh_trace["t"][0] == 0
    _assert_module_snapshot(model, source)


def test_scnv_raw_drives_do_not_solve_voltage_and_split_runs_keep_callback_clock():
    model = _model()
    runner = _runner(model)
    source = _module_snapshot(model)
    baseline, auxiliary = runner()
    drive = model.v.new_ones((STEPS, *model.shape), requires_grad=True)
    driven, driven_aux = runner(inputs=dn.func.RolloutInput(100 * drive, 1e-4 * drive))
    _assert_tree_close(driven, baseline, rtol=0, atol=0)
    _close(
        dict(driven_aux["callbacks"]["trace"]), dict(auxiliary["callbacks"]["trace"])
    )
    drive_gradient = torch.autograd.grad(
        _loss(driven_aux["callbacks"]["trace"]), drive, allow_unused=True
    )[0]
    assert drive_gradient is None or torch.count_nonzero(drive_gradient) == 0
    chunk = _runner(model, steps=2)
    state, first = chunk()
    final, second = chunk(
        state, parameters=first["parameters"], callback_state=first["callbacks"].state
    )
    _assert_tree_close(final, baseline, rtol=2e-9, atol=2e-10)
    _close(
        {
            name: torch.cat((first["callbacks"]["trace"][name], value))
            for name, value in second["callbacks"]["trace"].items()
        },
        dict(auxiliary["callbacks"]["trace"]),
    )
    _assert_module_snapshot(model, source)


def test_scnv_vmap_empty_lanes_and_forward_reverse_jacobians():
    model = _model(batch=None)
    source = _module_snapshot(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    key = _key(tensors.parameters, ".driver.drift_param")

    def evaluate(drift, voltage):
        parameters = {**tensors.parameters, key: drift}
        initialized = functional.initialize(
            parameters,
            tensors.constants,
            tensors.initialization._replace(v_init=voltage),
        )
        state, _ = functional.prepare_and_rollout(
            initialized.parameters, initialized.constants, initialized.state, steps=2
        )
        width = model.v.numel()
        return torch.cat(
            (
                state["integrator"]["v"].reshape(width),
                state["mechanism_buffers"]["driver"]["owned"].reshape(width),
                state["integrator"]["i_membrane"].reshape(width) * 1e6,
            )
        )

    inputs = (tensors.parameters[key], tensors.initialization.v_init)
    lanes = tuple(torch.stack((value, value + 0.1)) for value in inputs)
    torch.testing.assert_close(
        torch.vmap(evaluate)(*lanes),
        torch.stack([evaluate(*(value[i] for value in lanes)) for i in range(2)]),
    )
    assert torch.vmap(evaluate)(*(value[:0] for value in lanes)).shape[0] == 0
    for actual, expected in zip(
        torch.func.jacfwd(evaluate, argnums=(0, 1))(*inputs),
        torch.func.jacrev(evaluate, argnums=(0, 1))(*inputs),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-10)
    _assert_module_snapshot(model, source)


@pytest.mark.parametrize("mixed", (False, True))
def test_scnv_components_are_rejected_before_packed_voltage_solves(mixed):
    first = _model(batch=None)
    second = _model(batch=None)
    if mixed:
        with dn.ctx(JIT=0, REQUIRE_GRAD=1):
            second = dn.Population(N=1, C=2, dtype=torch.float64)
            second.insert(pas)
            second.initialize()
    packed = dn.concat_models({"skip": first, "other": second})
    packed.initialize()
    with pytest.raises(dn.func.FunctionalizationError, match="scnv|skip|component"):
        dn.func.make_functional(packed, dt=DT)


@pytest.mark.slow
@pytest.mark.parametrize("execution,checkpoint_every", (("scan", 2), ("while", None)))
def test_scnv_inductor_checkpointed_scan_and_inference_while(
    execution, checkpoint_every
):
    _exercise(
        backend="inductor", execution=execution, checkpoint_every=checkpoint_every
    )
