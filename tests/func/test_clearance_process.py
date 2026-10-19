"""Population-level clearance preserves process order, replay, and physiology.

The independent material recurrence detects duplicate SOURCE application and
noncommuting process reordering. Actual imperative initialization/stepping is
the trace oracle; explicit functional recurrence is the AD oracle. Only two
tiny configurations use real Inductor, each on one and four CPU threads.
"""

from __future__ import annotations

from types import MethodType

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import ClearanceProcess, MaterialProcess, Mechanism
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
from tests.func.test_membrane_current import _close as _close_recorded
from tests.func.test_membrane_current import _model as _membrane_model

DT = 0.01
STEPS = 4
POOL = "pool_material.amount"
NAI = "na_ion.nai"
PRODUCTION = 0.17
pytestmark = pytest.mark.cpu


class _MaterialSourceCurrent(Mechanism):
    Mechanism.GLOBAL(production=PRODUCTION, g=2.0e-5)
    Mechanism.USEMATERIAL("pool", read=["amount"], source={"amount": "delta"})
    Mechanism.USEION("na", read=["ena", "nai"], write=["ina"])
    Mechanism.AFFINE("ina")

    def advance(self, v, dt, values):
        del v
        return {"delta": dt * self.production * (1.0 + 0.01 * values["nai"])}

    def ina(self, v):
        return self.g * (1.0 + 0.1 * self.amount) * (v - self.ena)


class _ExactClearance(ClearanceProcess):
    ClearanceProcess.GLOBAL(rate=0.35, target=1.1, na_rate=0.23, na_target=7.5)
    ClearanceProcess.METHOD("exact")
    ClearanceProcess.CLEAR("pool", field="amount", rate="rate", target="target")
    ClearanceProcess.CLEAR("pool", field="amount", rate=0.7, target=0.65)
    ClearanceProcess.CLEAR("na", field="nai", rate="na_rate", target="na_target")


class _ImplicitClearance(ClearanceProcess):
    ClearanceProcess.GLOBAL(rate=0.35, target=1.1, na_rate=0.23, na_target=7.5)
    ClearanceProcess.METHOD("implicit")
    ClearanceProcess.CLEAR("pool", field="amount", rate="rate", target="target")
    ClearanceProcess.CLEAR("pool", field="amount", rate=0.7, target=0.65)
    ClearanceProcess.CLEAR("na", field="nai", rate="na_rate", target="na_target")


class _ExplicitClearance(ClearanceProcess):
    ClearanceProcess.GLOBAL(rate=0.35, target=1.1, na_rate=0.23, na_target=7.5)
    ClearanceProcess.METHOD("explicit")
    ClearanceProcess.CLEAR("pool", field="amount", rate="rate", target="target")
    ClearanceProcess.CLEAR("pool", field="amount", rate=0.7, target=0.65)
    ClearanceProcess.CLEAR("na", field="nai", rate="na_rate", target="na_target")


class _TransportClearance(ClearanceProcess):
    ClearanceProcess.PHASE("transport")
    ClearanceProcess.CLEAR("pool", field="amount", rate=0.11, target=2.0)


class _FinalClearance(ClearanceProcess):
    ClearanceProcess.GLOBAL(rate=0.19, target=2.4)
    ClearanceProcess.PHASE("post_transport")
    ClearanceProcess.CLEAR("pool", field="amount", rate="rate", target="target")


METHODS = {
    "exact": _ExactClearance,
    "implicit": _ImplicitClearance,
    "explicit": _ExplicitClearance,
}


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


def _clone(tree):
    return pytree.tree_map_only(torch.Tensor, lambda x: x.detach().clone(), tree)


def _key(parameters, suffix):
    names = [name for name in parameters if name.endswith(suffix)]
    assert len(names) == 1, (suffix, names)
    return names[0]


def _model(
    kind="path",
    *,
    method="exact",
    device="cpu",
    dtype=torch.float64,
    shell_resistance_scale=1.0,
):
    model = _membrane_model(kind, device=device, dtype=dtype, batch=None)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True, DEVICE=device, DTYPE=dtype):
        components = (
            model.populations.values() if kind.startswith("multi_") else (model,)
        )
        for component in components:
            if hasattr(component, "xraxial"):
                component.xraxial.mul_(shell_resistance_scale)
        model.material(
            "pool",
            fields={"amount": torch.nn.Parameter(model.v.new_tensor(1.7))},
            min_values={"amount": 0.0},
            domain="intracellular",
        )
        model.concentrations(nai0=torch.nn.Parameter(model.v.new_tensor(9.25)))
        model.ion_style("na", 1, 1)
        model.insert(_MaterialSourceCurrent)
        # Reverse insertion order must not replace scheduler phase order.
        model.insert(_FinalClearance)
        model.insert(_TransportClearance)
        model.insert(METHODS[method])
        model.batch(2)
        model.train()
        model.initialize()
        model.t.fill_(1.25)
    model._clearance_test_kind = kind
    model._clearance_test_method = method
    model._clearance_test_shell_scale = shell_resistance_scale
    return model


def _runner(
    model,
    *,
    backend="eager",
    execution="scan",
    checkpoint_every=None,
    initialization="fresh",
    steps=STEPS,
):
    functional, _ = dn.func.make_functional(model, dt=DT)
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
        inference=execution == "while",
        execution=execution,
        checkpoint_every=checkpoint_every,
        callbacks={"trace": dn.func.Recorder(tuple(_record_paths(functional)))},
        backend=backend,
        **options,
    )


def _arguments(runner, model):
    parameters = _clone(dict(runner.parameters))
    targets = {
        channel: parameters[
            _key(parameters, f"mechanisms.hh.{channel}_param")
        ].requires_grad_()
        for channel in CHANNELS
    }
    process = METHODS[model._clearance_test_method].__name__
    for owner, names in (
        (process, ("rate", "target", "na_rate", "na_target")),
        ("_FinalClearance", ("rate", "target")),
    ):
        for name in names:
            full = _key(parameters, f"material_processes.{owner}.{name}_param")
            targets[f"{owner}.{name}"] = parameters[full].requires_grad_()
    for label, suffix in (
        ("initial_pool", "materials.pool._initial_sources.amount"),
        ("initial_nai", "ions.na.i_init"),
    ):
        targets[label] = parameters[_key(parameters, suffix)].requires_grad_()
    if "noded2" in parameters:
        targets["geometry"] = parameters["noded2"].requires_grad_()
    else:
        # Canonical morphology models use their explicit physical area/edge
        # geometry, not the display diameter, for numerical assembly.
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


def _loss(recorded):
    value = 0.001 * recorded["v"].square().mean()
    value = value + 1e-4 * (recorded["i_membrane"] * 1e6).square().mean()
    value = value + 0.1 * recorded["hh.m"].square().mean()
    value = value + 0.2 * recorded[POOL].square().mean()
    value = value + 0.01 * recorded[NAI].square().mean()
    value = value + 0.0001 * recorded["na_ion.ena"].square().mean()
    if "vc" in recorded:
        shells = recorded["vc"][..., 1:]
        weights = torch.linspace(
            0.7, 1.3, shells.shape[-2], device=shells.device, dtype=shells.dtype
        )
        value = value + 0.1 * (weights.unsqueeze(-1) * shells).mean()
    return value


def _gradients(recorded, targets):
    values = torch.autograd.grad(_loss(recorded), tuple(targets.values()))
    gradients = dict(zip(targets, values, strict=True))
    for name, value in gradients.items():
        assert torch.isfinite(value).all() and torch.count_nonzero(value), name
    return gradients


def _close(actual, expected, *, dtype=torch.float64):
    _close_recorded(actual, expected, dtype=dtype)


def _explicit_reference(runner, parameters, initialization, inputs, *, steps=STEPS):
    functional = runner.functional
    initialized = functional.initialize(parameters, runner.constants, initialization)
    prepared = functional.prepare(initialized.parameters, initialized.constants)
    return dn.func.run(
        functional.bind(initialized.parameters, prepared),
        initialized.state,
        inputs=inputs,
        steps=steps,
        callbacks=runner._callbacks,
    )


def _imperative_reference(model, runner, parameters, initialization, inputs):
    reference = _model(
        model._clearance_test_kind,
        method=model._clearance_test_method,
        device=model.device(),
        dtype=model.dtype(),
        shell_resistance_scale=model._clearance_test_shell_scale,
    )
    with torch.no_grad():
        raw = dict(reference.named_parameters())
        for name, value in parameters.items():
            raw[name].copy_(value)
        for name, value in runner.constants.items():
            if name != "dt" and not name.startswith("parametrizations."):
                target = _constant_target(reference, name)
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
        for index in range(STEPS):
            reference.integrator.step(
                reference, dt, inputs.ve[index], inputs.intra[index]
            )
            reference.t = reference.t + dt
            state = _live_state(runner.functional, reference)
            history.append(
                {name: _read_path(state, path) for name, path in paths.items()}
            )
    return {name: torch.stack([frame[name] for frame in history]) for name in paths}


def _relax(c, rate, target, dt, method):
    if method == "explicit":
        return c - dt * rate * (c - target)
    if method == "implicit":
        return (c + dt * rate * target) / (1 + dt * rate)
    return target + (c - target) * torch.exp(-rate * dt)


def _material_oracle(recorded, parameters, dt, method):
    process = METHODS[method].__name__

    def parameter(owner, name):
        return parameters[_key(parameters, f"material_processes.{owner}.{name}_param")]

    pool, sodium = recorded[POOL][0], recorded[NAI][0]
    pools, ions = [pool], [sodium]
    for _ in range(recorded[POOL].shape[0] - 1):
        pool = pool + dt * PRODUCTION * (1 + 0.01 * sodium)
        pool = _relax(
            pool, parameter(process, "rate"), parameter(process, "target"), dt, method
        )
        pool = _relax(pool, pool.new_tensor(0.7), pool.new_tensor(0.65), dt, method)
        sodium = _relax(
            sodium,
            parameter(process, "na_rate"),
            parameter(process, "na_target"),
            dt,
            method,
        )
        pool = _relax(pool, pool.new_tensor(0.11), pool.new_tensor(2.0), dt, "exact")
        pool = _relax(
            pool,
            parameter("_FinalClearance", "rate"),
            parameter("_FinalClearance", "target"),
            dt,
            "exact",
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
    method="exact",
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
        method=method,
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
    _material_oracle(actual, parameters, runner.constants["dt"], method)
    for key, suffix in (
        (POOL, "materials.pool._initial_sources.amount"),
        (NAI, "ions.na.i_init"),
    ):
        torch.testing.assert_close(
            actual[key][0],
            parameters[_key(parameters, suffix)].expand_as(actual[key][0]),
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


@pytest.mark.parametrize("kind", ("path", "block_path", "multi_scalar", "multi_block"))
def test_clearance_full_traces_and_all_gradients(kind, cpu_threads):
    _exercise(kind)


@pytest.mark.parametrize("method", tuple(METHODS))
def test_clearance_methods_source_once_and_noncommuting_phase_order(
    method, cpu_threads
):
    _exercise("point", method=method)


@pytest.mark.parametrize(
    "kind",
    ("point", "single", "myelinated", "cable", "tree", "block_tree", "multi_point"),
)
def test_clearance_other_admitted_topologies(kind):
    _exercise(kind)


@pytest.mark.parametrize("kind", ("path", "multi_block"))
def test_clearance_checkpointed_continuation_preserves_full_traces_and_gradients(
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
    actual_gradients = _gradients(recorded, targets)
    (
        reference_parameters,
        reference_initialization,
        reference_inputs,
        reference_targets,
    ) = _arguments(runner, model)
    doubled = dn.func.RolloutInput(*(torch.cat((x, x)) for x in reference_inputs))
    expected = _explicit_reference(
        runner, reference_parameters, reference_initialization, doubled, steps=2 * STEPS
    )
    _assert_tree_close(final, expected[0])
    expected_recorded = dict(expected[1]["callbacks"]["trace"])
    _close(recorded, expected_recorded)
    _close(actual_gradients, _gradients(expected_recorded, reference_targets))
    assert recorded["v"].shape[0] == 2 * STEPS + 1


def test_clearance_snapshot_restarts_do_not_advance_source_or_initial_materials(
    cpu_threads,
):
    model = _model("path")
    source = _module_snapshot(model)
    runner = _runner(model, initialization="snapshot")
    first = runner()
    second = runner()
    _assert_tree_close(first[0], second[0])
    _close(dict(first[1]["callbacks"]["trace"]), dict(second[1]["callbacks"]["trace"]))
    _assert_tree_close(
        first[1]["callbacks"].state.carries, second[1]["callbacks"].state.carries
    )
    trace = first[1]["callbacks"]["trace"]
    assert trace["t"][0].item() == 1.25
    torch.testing.assert_close(trace[POOL][0], model.mech.materials["pool"].amount)
    _assert_module_snapshot(model, source)


def test_clearance_rate_initial_field_and_dt_gradients_match_central_differences(
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
    for label in ("_ExactClearance.rate", "initial_pool", "initial_voltage", "dt"):
        leaf = targets[label]
        step = 1e-5 if label != "dt" else 1e-6
        values = []
        for sign in (1, -1):
            with torch.no_grad():
                leaf.add_(sign * step)
            trace = _imperative_reference(
                model, runner, parameters, initialization, inputs
            )
            values.append(_loss(trace))
            with torch.no_grad():
                leaf.sub_(sign * step)
        difference = (values[0] - values[1]) / (2 * step)
        torch.testing.assert_close(
            gradients[label].sum(), difference, rtol=3e-5, atol=2e-7
        )


class _UnsupportedProcess(MaterialProcess):
    def advance_materials(self, dt):
        del dt


class _MutatingClearance(_ExactClearance):
    def advance_materials(self, dt):
        self._test_mutation = True
        return super().advance_materials(dt)


@pytest.mark.parametrize("process", (_UnsupportedProcess, _MutatingClearance))
def test_unsupported_or_mutating_processes_fail_closed(process):
    model = _model("path")
    model.insert(process)
    model.initialize()
    source = _module_snapshot(model)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="MaterialProcess|Clearance|clearance|process",
    ):
        dn.func.make_functional(model, dt=DT)
    _assert_module_snapshot(model, source)


class _LiteralClearance(ClearanceProcess):
    ClearanceProcess.CLEAR(
        "pool", field="amount", rate=torch.tensor(0.29), target=torch.tensor(0.85)
    )


def test_literal_clearance_operands_are_explicit_differentiable_constants(cpu_threads):
    model = _model("point")
    model.insert(_LiteralClearance)
    model.initialize()
    runner = _runner(model)
    parameters, initialization, inputs, targets = _arguments(runner, model)
    for name in ("rate", "target"):
        key = _key(
            runner.constants,
            f"material_processes._LiteralClearance._clearance_literal_0_{name}",
        )
        targets[f"literal_{name}"] = runner.constants[key].requires_grad_()
    source = _module_snapshot(model)
    actual = runner(parameters=parameters, initialization=initialization, inputs=inputs)
    recorded = dict(actual[1]["callbacks"]["trace"])
    gradients = _gradients(recorded, targets)
    assert len(gradients) == 20
    (
        reference_parameters,
        reference_initialization,
        reference_inputs,
        reference_targets,
    ) = _arguments(runner, model)
    for name in ("rate", "target"):
        key = _key(
            runner.constants,
            f"material_processes._LiteralClearance._clearance_literal_0_{name}",
        )
        reference_targets[f"literal_{name}"] = runner.constants[key]
    expected = _explicit_reference(
        runner, reference_parameters, reference_initialization, reference_inputs
    )
    reference = dict(expected[1]["callbacks"]["trace"])
    _close(recorded, reference)
    _close(gradients, _gradients(reference, reference_targets))
    _assert_module_snapshot(model, source)


def test_nested_literal_class_audit_restores_tensor_identity_storage_and_aliases():
    from dendra.func._population import _class_state_snapshot, _restore_class_state
    from dendra.models.mechanisms._material_process import ClearanceSpec

    rate = torch.tensor(0.29, dtype=torch.float64)
    original = rate.clone()
    owner = torch.nn.Module()
    owner.register_buffer("rate", rate)

    class LiteralConfiguration:
        config = {
            "specs": (ClearanceSpec("pool", "amount", rate, 0.85),),
            "metadata": {"phase": "post_local", "order": [1, 2]},
        }

    storage = rate.untyped_storage()
    pointer = storage.data_ptr()
    snapshot = _class_state_snapshot((LiteralConfiguration,))
    with torch.no_grad():
        rate.add_(3.0)
    LiteralConfiguration.config["metadata"]["phase"] = "transport"
    LiteralConfiguration.config["metadata"]["order"].append(3)
    changed = _restore_class_state(snapshot)
    assert changed
    restored = LiteralConfiguration.config["specs"][0]
    assert restored.rate is rate and owner.rate is rate
    assert rate.untyped_storage() is storage and storage.data_ptr() == pointer
    assert not rate.requires_grad
    torch.testing.assert_close(rate, original, rtol=0, atol=0)
    assert LiteralConfiguration.config["metadata"] == {
        "phase": "post_local",
        "order": [1, 2],
    }


@pytest.mark.parametrize(
    "mutation", ("method", "phase", "specs", "process_resolver", "handler_resolver")
)
def test_clearance_structural_changes_invalidate_prepared_execution(
    monkeypatch, mutation
):
    model = _model("point")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    process = model.mech.material_processes["_ExactClearance"]
    if mutation == "method":
        monkeypatch.setattr(process, "_clearance_method", "implicit")
    elif mutation == "phase":
        monkeypatch.setattr(type(process), "_material_process_phase", "post_transport")
    elif mutation == "specs":
        monkeypatch.setattr(
            type(process), "_clearance_specs", type(process)._clearance_specs[::-1]
        )
    else:
        owner = process if mutation == "process_resolver" else model.mech
        name = (
            "_material_resolver" if mutation == "process_resolver" else "_get_material"
        )
        original = getattr(owner, name)

        def resolver(self, material):
            del self
            return original(material)

        monkeypatch.setattr(owner, name, MethodType(resolver, owner))
    with pytest.raises(
        dn.func.FunctionalizationError, match="structure|changed|stale|lower"
    ):
        functional.step(tensors.parameters, prepared, tensors.state)


@pytest.mark.parametrize("owner", ("process", "handler"))
def test_custom_material_resolvers_fail_closed(owner):
    model = _model("point")
    if owner == "process":
        target = model.mech.material_processes["_ExactClearance"]
        original = target._material_resolver
        name = "_material_resolver"
    else:
        target = model.mech
        original = target._get_material
        name = "_get_material"

    def resolver(self, material):
        del self
        return original(material)

    setattr(target, name, MethodType(resolver, target))
    with pytest.raises(
        dn.func.FunctionalizationError, match="resolver|canonical|standard"
    ):
        dn.func.make_functional(model, dt=DT)


def test_clearance_atomic_transforms_include_empty_vmap_and_jvp(cpu_threads):
    model = _model("point")
    runner = _runner(model)
    parameters, initialization, inputs, _targets = _arguments(runner, model)
    functional = runner.functional
    rate_name = _key(parameters, "material_processes._ExactClearance.rate_param")
    source = _module_snapshot(model)

    def rollout(rate, voltage, drive):
        raw = {**parameters, rate_name: rate}
        initialized = functional.initialize(
            raw, runner.constants, initialization._replace(v_init=voltage)
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

    def objective(rate, voltage, drive):
        return _loss(rollout(rate, voltage, drive))

    rate = parameters[rate_name]
    voltage = initialization.v_init
    drives = torch.stack((inputs.intra * 0.8, inputs.intra * 1.2))
    rates = torch.stack((rate * 0.9, rate * 1.1))
    voltages = torch.stack((voltage - 0.5, voltage + 0.7))
    expected = torch.stack(
        [objective(rates[i], voltages[i], drives[i]) for i in range(2)]
    )
    torch.testing.assert_close(
        torch.func.vmap(objective)(rates, voltages, drives), expected
    )
    empty_recording = torch.func.vmap(rollout)(rates[:0], voltages[:0], drives[:0])
    assert empty_recording["v"].shape == (0, STEPS + 1, *model.shape)
    derivative = torch.func.grad(objective, argnums=(0, 1, 2))
    expected_gradients = pytree.tree_map(
        lambda *values: torch.stack(values),
        *[derivative(rates[i], voltages[i], drives[i]) for i in range(2)],
    )
    actual_gradients = torch.func.vmap(derivative)(rates, voltages, drives)
    _assert_tree_close(actual_gradients, expected_gradients, rtol=3e-9, atol=3e-10)

    # PyTorch cannot multiply an empty mapped scalar by a Python scalar after
    # a reduction. Test Dendra's full recording directly above and use a pure
    # material sum for zero-lane AD, without that unrelated loss arithmetic.
    def material_sum(rate, voltage, drive):
        return rollout(rate, voltage, drive)[POOL].sum()

    empty = torch.func.vmap(torch.func.grad(material_sum, argnums=(0, 1, 2)))(
        rates[:0], voltages[:0], drives[:0]
    )
    assert all(value.shape[0] == 0 for value in empty)
    primals = (rate, voltage, inputs.intra)
    tangents = (
        torch.ones_like(rate) * 0.01,
        torch.ones_like(voltage) * 0.01,
        torch.ones_like(inputs.intra) * 1e-10,
    )
    _, tangent = torch.func.jvp(objective, primals, tangents)
    gradients = derivative(*primals)
    expected_tangent = sum(
        (gradient * direction).sum()
        for gradient, direction in zip(gradients, tangents, strict=True)
    )
    torch.testing.assert_close(tangent, expected_tangent, rtol=3e-9, atol=3e-10)
    _assert_module_snapshot(model, source)


@pytest.mark.parametrize("kind", ("path", "multi_block"))
def test_clearance_commit_state_then_imperative_continuation(kind):
    model = _model(kind)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    state, _ = functional.step(tensors.parameters, prepared, tensors.state)
    expected, _ = functional.step(tensors.parameters, prepared, state)
    source = _module_snapshot(model)
    target = _model(kind)
    functional.commit_state_(target, state)
    dt = tensors.constants["dt"]
    target.integrator._initialize(target, dt, force=True, compile_scope="population")
    target.integrator.step(target, dt, None, None)
    target.t = target.t + dt
    _assert_tree_close(_live_state(functional, target), expected, rtol=3e-9, atol=3e-10)
    _assert_module_snapshot(model, source)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("kind", "execution", "checkpoint_every"),
    (("path", "scan", 2), ("multi_block", "while", None)),
)
def test_inductor_clearance_scan_gradients_and_while_full_traces(
    kind, execution, checkpoint_every, cpu_threads
):
    _exercise(
        kind, backend="inductor", execution=execution, checkpoint_every=checkpoint_every
    )
