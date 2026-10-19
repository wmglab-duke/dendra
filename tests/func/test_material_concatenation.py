"""Packed chemistry retains imperative and pure functional lifecycle parity.

The full traces are independently checked by stepping the original children;
AD is additionally checked against explicit functional recurrence. Real compiler
coverage is deliberately one scan and one inference while-loop, not unrolling.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from dendra.models.multi import MultiPopulation
from tests.func.test_clearance_process import _clone, _key
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _assert_tree_close,
    _module_snapshot,
)
from tests.test_material_concatenation import (
    CHANNELS,
    DT,
    PROCESSES,
    STEPS,
    _assert_close,
    _assert_independent_traces,
    _components,
    _gradients,
    _independent_traces,
    _loss,
    _model,
    _targets,
    _trace,
)
from tests.test_material_concatenation_contracts import (
    _initialize,
    _IonClearance,
    _point,
)

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


def _runner(
    model,
    *,
    backend="eager",
    execution="scan",
    checkpoint_every=None,
    steps=STEPS,
    initialization="fresh",
):
    fields = ("v", "i_membrane", "hh.m", "hh.h", "hh.n", "pool_material.amount")
    if "reservoir" in model.mech.materials:
        fields = (*fields, "reservoir_material.amount")
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
        callbacks={"trace": dn.func.Recorder(fields)},
        backend=backend,
        **options,
    )


def _arguments(runner, model, *, steps=STEPS):
    parameters = _clone(dict(runner.parameters))
    targets = {
        name: value.requires_grad_()
        for name, value in parameters.items()
        if value.is_floating_point()
        and (".material_processes." in name or "material" in name or "chemical" in name)
    }
    for name in CHANNELS:
        targets[f"hh.{name}"] = parameters[
            _key(parameters, f"mechanisms.hh.{name}_param")
        ].requires_grad_()
    initialization = _clone(runner.initialization_inputs)
    targets["initial_voltage"] = initialization.v_init.requires_grad_()
    targets["dt"] = runner.constants["dt"].requires_grad_()
    targets["stimulus"] = model.v.new_ones((), requires_grad=True)
    drive = torch.stack(
        [model.v.new_full(model.shape, (index + 1) * 1e-10) for index in range(steps)]
    )
    return (
        parameters,
        initialization,
        dn.func.RolloutInput(None, drive * targets["stimulus"]),
        targets,
    )


def _explicit_reference(runner, parameters, initialization, inputs, *, steps=STEPS):
    initialized = runner.functional.initialize(
        parameters, runner.constants, initialization
    )
    prepared = runner.functional.prepare(initialized.parameters, initialized.constants)
    return dn.func.run(
        runner.functional.bind(initialized.parameters, prepared),
        initialized.state,
        inputs=inputs,
        steps=steps,
        callbacks=runner._callbacks,
    )


def _independent_gradient_oracle(
    model, runner, recorded, gradients, reference, amplitude
):
    """Compare AD to separate children, not just two uses of the same lowering."""
    targets = {"stimulus": [amplitude]}
    for component, child in reference.items():
        for name, value in _targets(child, component=component).items():
            if name.endswith("initial_voltage"):
                continue
            targets.setdefault(name, []).append(value)
    flattened = {
        f"{name}:{index}": value
        for name, values in targets.items()
        for index, value in enumerate(values)
    }
    expected = _gradients(sum(_loss(trace) for trace in recorded.values()), flattened)
    expected = {
        name: sum(expected[f"{name}:{index}"] for index in range(len(values)))
        for name, values in targets.items()
    }
    original = dict(model.named_parameters())
    actual = {"stimulus": gradients["stimulus"]}
    for name, source in _targets(model).items():
        if name.endswith("initial_voltage"):
            continue
        if name.startswith("hh."):
            actual[name] = gradients[name]
        else:
            source_name = next(
                key for key, value in original.items() if value is source
            )
            actual[name] = gradients[source_name]
    _assert_close(actual, expected, dtype=model.dtype())


def _independent_initial_voltage_differences(model, gradients):
    """Imperative init_v detaches; independent fresh values still check its AD."""
    step = 0.05 if model.dtype() == torch.float32 else 1e-4
    offset = 0
    for component, original in model.populations.items():
        direction = torch.linspace(
            0.5,
            1.5,
            original.v.numel(),
            device=original.device(),
            dtype=original.dtype(),
        ).reshape(original.shape)
        losses = []
        for sign in (1, -1):
            child = _components(**model._concat_test_options)[component]
            with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True), torch.no_grad():
                child.set_v_init(
                    torch.nn.Parameter(
                        child.expanded_v_init(child.shape) + sign * step * direction,
                        requires_grad=False,
                    )
                )
                child.clear_steady_state()
                trace = _trace(
                    child, dt=child.v.new_tensor(DT), amplitude=child.v.new_ones(())
                )
                losses.append(_loss(trace))
        count = original.v.numel()
        actual = (
            gradients["initial_voltage"][..., offset : offset + count].reshape(
                original.shape
            )
            * direction
        ).sum()
        expected = (losses[0] - losses[1]) / (2 * step)
        torch.testing.assert_close(
            actual,
            expected,
            rtol=5e-3 if model.dtype() == torch.float32 else 2e-5,
            atol=2e-3 if model.dtype() == torch.float32 else 2e-7,
            msg=f"{component}: fresh initial-voltage finite difference",
        )
        offset += count


def _exercise(
    family="diffusion",
    *,
    backend="eager",
    execution="scan",
    checkpoint_every=None,
    **options,
):
    model = _model(family, **options)
    runner = _runner(
        model, backend=backend, execution=execution, checkpoint_every=checkpoint_every
    )
    snapshot = _module_snapshot(model)
    parameters, initialization, inputs, targets = _arguments(runner, model)
    result = runner(parameters=parameters, initialization=initialization, inputs=inputs)
    recorded = dict(result[1]["callbacks"]["trace"])
    gradients = _gradients(_loss(recorded), targets) if execution == "scan" else {}
    rp, ri, rx, rt = _arguments(runner, model)
    expected = _explicit_reference(runner, rp, ri, rx)
    _assert_tree_close(
        result[0],
        expected[0],
        rtol=3e-4 if model.dtype() == torch.float32 else 2e-9,
        atol=3e-5 if model.dtype() == torch.float32 else 2e-10,
    )
    expected_recorded = dict(expected[1]["callbacks"]["trace"])
    _assert_close(recorded, expected_recorded, dtype=model.dtype())
    if gradients:
        _assert_close(
            gradients, _gradients(_loss(expected_recorded), rt), dtype=model.dtype()
        )
        for name in (
            "stimulus",
            "initial_voltage",
            "dt",
            *(f"hh.{name}" for name in CHANNELS),
        ):
            assert torch.count_nonzero(gradients[name]), name
        assert any(
            torch.count_nonzero(value)
            for name, value in gradients.items()
            if "material_processes" in name
        )
    reference = _components(family, **options)
    reference_amplitude = model.v.new_ones((), requires_grad=True)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True):
        independent = _independent_traces(
            reference, dt=model.v.new_tensor(DT), amplitude=reference_amplitude
        )
    _assert_independent_traces(model, recorded, independent)
    if gradients:
        _independent_gradient_oracle(
            model, runner, independent, gradients, reference, reference_amplitude
        )
        _independent_initial_voltage_differences(model, gradients)
    assert recorded["v"].shape == (STEPS + 1, *model.shape)
    assert torch.count_nonzero(recorded["i_membrane"][0]) == 0
    _assert_module_snapshot(model, snapshot)
    return recorded, gradients


@pytest.mark.parametrize("family", tuple(PROCESSES))
def test_concatenated_processes_fresh_full_traces_and_all_gradients(
    family, cpu_threads
):
    _exercise(family)


def test_packed_diffusion_vmap_empty_vmap_and_jvp(cpu_threads):
    """Independent chemical forests retain the ordinary functional transforms."""
    model = _model("named_diffusion")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters = _clone(dict(tensors.parameters))
    initialization = _clone(tensors.initialization)
    diffusion_key = next(
        name
        for name in parameters
        if ".material_processes." in name and name.endswith("D_param")
    )
    callbacks = functional.make_callbacks(
        {
            "trace": dn.func.Recorder(
                ("v", "hh.m", "hh.h", "hh.n", "i_membrane", "pool_material.amount")
            )
        }
    )

    def rollout(diffusion, voltage, drive):
        initialized = functional.initialize(
            {**parameters, diffusion_key: diffusion},
            tensors.constants,
            initialization._replace(v_init=voltage),
        )
        _, auxiliary = functional.prepare_and_rollout(
            initialized.parameters,
            initialized.constants,
            initialized.state,
            steps=STEPS,
            inputs=dn.func.RolloutInput(None, drive),
            callbacks=callbacks,
        )
        return dict(auxiliary["callbacks"]["trace"])

    def objective(diffusion, voltage, drive):
        return _loss(rollout(diffusion, voltage, drive))

    values = (
        parameters[diffusion_key],
        initialization.v_init,
        model.v.new_full((STEPS, *model.shape), 1e-10),
    )
    lanes = tuple(torch.stack((value * 0.97, value * 1.03)) for value in values)

    def compare_lanes(evaluate):
        expected = pytree.tree_map(
            lambda *leaves: torch.stack(leaves),
            *[evaluate(*(value[index] for value in lanes)) for index in range(2)],
        )
        _assert_tree_close(
            torch.func.vmap(evaluate)(*lanes), expected, rtol=3e-8, atol=3e-9
        )

    compare_lanes(rollout)
    derivative = torch.func.grad(objective, argnums=(0, 1, 2))
    compare_lanes(derivative)
    empty = torch.func.vmap(rollout)(*(value[:0] for value in lanes))
    assert all(value.shape == (0, STEPS + 1, *model.shape) for value in empty.values())
    directions = tuple(
        torch.full_like(value, 0.01 if index < 2 else 1e-10)
        for index, value in enumerate(values)
    )
    _, tangent = torch.func.jvp(objective, values, directions)
    expected_tangent = sum(
        (gradient * direction).sum()
        for gradient, direction in zip(derivative(*values), directions, strict=True)
    )
    torch.testing.assert_close(tangent, expected_tangent, rtol=3e-8, atol=3e-9)


def _exercise_explicitly_batched_named_diffusion(*, device="cpu"):
    model = _model("named_diffusion", device=device)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True):
        model.batch(2).train().initialize()
        functional, tensors = dn.func.make_functional(model, dt=DT)
        runner = SimpleNamespace(
            parameters=tensors.parameters,
            constants=tensors.constants,
            initialization_inputs=tensors.initialization,
        )
        parameters, initialization, inputs, targets = _arguments(runner, model)
        initialized = functional.initialize(
            parameters, tensors.constants, initialization
        )
        callbacks = functional.make_callbacks(
            {
                "trace": dn.func.Recorder(
                    (
                        "v",
                        "hh.m",
                        "hh.h",
                        "hh.n",
                        "i_membrane",
                        "pool_material.amount",
                    )
                )
            }
        )
        _, auxiliary = functional.prepare_and_rollout(
            initialized.parameters,
            initialized.constants,
            initialized.state,
            steps=STEPS,
            inputs=inputs,
            callbacks=callbacks,
        )
        actual = dict(auxiliary["callbacks"]["trace"])
        gradients = _gradients(_loss(actual), targets)
        for name in (
            "stimulus",
            "initial_voltage",
            "dt",
            *(f"hh.{name}" for name in CHANNELS),
        ):
            assert torch.isfinite(gradients[name]).all() and torch.count_nonzero(
                gradients[name]
            ), name
        imperative = _trace(
            model, dt=model.v.new_tensor(DT), amplitude=model.v.new_ones(())
        )
        _assert_close(actual, imperative)
        reference = _components("named_diffusion", device=device)
        for child in reference.values():
            child.batch(2)
        amplitude = model.v.new_ones((), requires_grad=True)
        independent = _independent_traces(
            reference, dt=model.v.new_tensor(DT), amplitude=amplitude
        )
        offset = 0
        for name, child in reference.items():
            count = child.shape[-2] * child.shape[-1]
            _assert_close(
                {
                    field: value[..., offset : offset + count].reshape_as(
                        independent[name][field]
                    )
                    for field, value in actual.items()
                },
                independent[name],
            )
            offset += count
        _independent_gradient_oracle(
            model, runner, independent, gradients, reference, amplitude
        )
        assert actual["pool_material.amount"].shape == (
            STEPS + 1,
            2,
            *model.core_shape(),
        )


def test_explicitly_batched_named_diffusion_full_traces_and_gradients(cpu_threads):
    _exercise_explicitly_batched_named_diffusion()


@pytest.mark.parametrize("missing", ("material", "process"))
def test_concatenated_diffusion_sparse_component_support(missing, cpu_threads):
    _exercise("diffusion", **{f"missing_{missing}": True})


def test_concatenated_diffusion_checkpointed_continuation(cpu_threads):
    model = _model("named_diffusion")
    runner = _runner(model, steps=2, checkpoint_every=2)
    parameters, initialization, inputs, targets = _arguments(runner, model)
    first_inputs = dn.func.RolloutInput(None, inputs.intra[:2])
    state, first = runner(
        parameters=parameters, initialization=initialization, inputs=first_inputs
    )
    final, second = runner(
        state,
        parameters=first["parameters"],
        inputs=dn.func.RolloutInput(None, inputs.intra[2:]),
        callback_state=first["callbacks"].state,
    )
    recorded = {
        name: torch.cat((value, second["callbacks"]["trace"][name]))
        for name, value in first["callbacks"]["trace"].items()
    }
    gradients = _gradients(_loss(recorded), targets)
    rp, ri, rx, rt = _arguments(runner, model)
    expected = _explicit_reference(runner, rp, ri, rx)
    _assert_tree_close(final, expected[0], rtol=2e-9, atol=2e-10)
    expected_recorded = dict(expected[1]["callbacks"]["trace"])
    _assert_close(recorded, expected_recorded)
    _assert_close(gradients, _gradients(_loss(expected_recorded), rt))


def test_concatenated_process_snapshot_restarts_preserve_materials(cpu_threads):
    model = _model("named_diffusion")
    runner = _runner(model, initialization="snapshot")
    first, second = runner(), runner()
    _assert_tree_close(first[0], second[0])
    _assert_close(
        dict(first[1]["callbacks"]["trace"]), dict(second[1]["callbacks"]["trace"])
    )
    torch.testing.assert_close(
        first[1]["callbacks"]["trace"]["pool_material.amount"][0],
        model.mech.materials["pool"].amount,
    )


def test_fresh_initialization_uses_reconfigured_child_material_source(cpu_threads):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64):
        child = _point(2)
        child.insert(pas)
        child.material("pool", fields={"c": 1.0})
        _initialize(child)
        initial = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
        child.material("pool", fields={"c": initial})
        packed = _initialize(MultiPopulation(child=child))
        functional, tensors = dn.func.make_functional(packed, dt=DT)
        initialized = functional.initialize(
            tensors.parameters, tensors.constants, tensors.initialization
        )
        prepared = functional.prepare(initialized.parameters, initialized.constants)
        callbacks = functional.make_callbacks(
            {"trace": dn.func.Recorder(("pool_material.c",))}
        )
        _state, output = dn.func.run(
            functional.bind(initialized.parameters, prepared),
            initialized.state,
            steps=2,
            callbacks=callbacks,
        )
        recorded = output["callbacks"]["trace"]["pool_material.c"]
        torch.testing.assert_close(recorded, initial.new_full((3, 1, 2), 2.0))
        torch.testing.assert_close(
            torch.autograd.grad(recorded.sum(), initial)[0], initial.new_tensor(6.0)
        )


@pytest.mark.parametrize("configured", (False, True))
def test_fresh_initialization_keeps_trained_child_ion_sources(configured, cpu_threads):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64):
        child = _point(2)
        child.insert(pas)
        child.insert(_IonClearance)
        if configured:
            child.concentrations(nai0=12.0, nao0=130.0)
            child.equilibria(ena=55.0)
        _initialize(child)
        ion = child.mech.ions["na"]
        sources = (ion.i_init, ion.o_init, ion.e_init)
        with torch.no_grad():
            for source, value in zip(sources, (17.0, 123.0, 45.0)):
                source.fill_(value)
        packed = _initialize(MultiPopulation(child=child))
        functional, tensors = dn.func.make_functional(packed, dt=DT)
        initialized = functional.initialize(
            tensors.parameters, tensors.constants, tensors.initialization
        )
        prepared = functional.prepare(initialized.parameters, initialized.constants)
        fields = ("na_ion.nai", "na_ion.nao", "na_ion.ena")
        callbacks = functional.make_callbacks({"trace": dn.func.Recorder(fields)})
        _state, output = dn.func.run(
            functional.bind(initialized.parameters, prepared),
            initialized.state,
            steps=2,
            callbacks=callbacks,
        )
        recorded = output["callbacks"]["trace"]
        factors = torch.exp(-0.2 * DT * torch.arange(3, dtype=torch.float64))
        for field, value, scale in zip(
            fields,
            (17.0, 123.0, 45.0),
            (factors, torch.ones_like(factors), torch.ones_like(factors)),
        ):
            torch.testing.assert_close(
                recorded[field], (value * scale).reshape(3, 1, 1).expand(3, 1, 2)
            )
        gradients = torch.autograd.grad(
            sum(recorded[field].sum() for field in fields), sources
        )
        for actual, expected in zip(
            gradients,
            (2 * factors.sum(), factors.new_tensor(6.0), factors.new_tensor(6.0)),
        ):
            torch.testing.assert_close(actual, expected)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("execution", "kinds", "checkpoint_every"),
    (("scan", ("path", "path"), 2), ("while", ("path", "tree"), None)),
)
def test_inductor_concatenated_diffusion_scan_and_while(
    execution, kinds, checkpoint_every, cpu_threads
):
    _exercise(
        "named_diffusion",
        kinds=kinds,
        backend="inductor",
        execution=execution,
        checkpoint_every=checkpoint_every,
    )


def _exercise_ion_sources(*, device="cpu", dtype=torch.float64):
    """Pre-authored ion style, initial sources, and rates retain their owners."""

    def components():
        children, sources = {}, {}
        for index, count in enumerate((2, 3)):
            child = dn.Population(
                N=1,
                C=count,
                celsius=6.3,
                dtype=dtype,
                integrator=dn.bwd_euler_sc(),
            )
            source = torch.nn.Parameter(child.v.new_tensor(12.0 + 11.0 * index))
            child.concentrations(nai0=source)
            child.ion_style("na", 1, 1)
            child.insert(pas)
            child.insert(_IonClearance, rate=0.2 + 0.07 * index)
            children[f"cell{index}"] = child
            sources[f"initial_{index}"] = source
        return children, sources

    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE=device, DTYPE=dtype):
        children, sources = components()
        model = dn.concat_models(children, celsius=6.3, threads=1, write_back=False)
        model.train().initialize()
        runner = dn.compile(
            model,
            dt=DT,
            steps=STEPS,
            initialization="fresh",
            execution="scan",
            inference=False,
            backend="eager",
            callbacks={"trace": dn.func.Recorder(("na_ion.nai", "na_ion.ena"))},
        )
        parameters = _clone(dict(runner.parameters))
        raw = dict(model.named_parameters())
        targets = {
            label: parameters[
                next(name for name, value in raw.items() if value is source)
            ].requires_grad_()
            for label, source in sources.items()
        }
        for process in model.mech.material_processes.values():
            component = type(process)._material_process_component
            index = tuple(children).index(component)
            key = next(
                name for name, value in raw.items() if value is process.rate_param
            )
            targets[f"rate_{index}"] = parameters[key].requires_grad_()
        recorded = dict(runner(parameters=parameters)[1]["callbacks"]["trace"])

        def loss(trace):
            return (
                0.2 * trace["na_ion.nai"].square().sum()
                + 0.01 * trace["na_ion.ena"].square().sum()
            )

        actual_gradients = _gradients(loss(recorded), targets)
        reference, reference_targets = components()
        histories = []
        for index, child in enumerate(reference.values()):
            child.train().initialize()
            process = next(iter(child.mech.material_processes.values()))
            reference_targets[f"rate_{index}"] = process.rate_param
            dt = child.v.new_tensor(DT)
            child.integrator._initialize(
                child, dt, force=True, compile_scope="population"
            )
            ion = child.mech.ions["na"]
            history = [{"na_ion.nai": ion.nai, "na_ion.ena": ion.ena}]
            for _ in range(STEPS):
                child.integrator.step(child, dt, None, None)
                child.t = child.t + dt
                history.append({"na_ion.nai": ion.nai, "na_ion.ena": ion.ena})
            histories.append(
                {
                    name: torch.stack([frame[name] for frame in history])
                    for name in history[0]
                }
            )
        expected = {
            name: torch.cat([history[name] for history in histories], dim=-1)
            for name in recorded
        }
        _assert_close(recorded, expected, dtype=dtype)
        _assert_close(
            actual_gradients, _gradients(loss(expected), reference_targets), dtype=dtype
        )
        for name, value in actual_gradients.items():
            assert torch.isfinite(value).all() and torch.count_nonzero(value), name
        assert recorded["na_ion.nai"].shape == (STEPS + 1, 1, 5)
        assert not torch.equal(recorded["na_ion.ena"][0], recorded["na_ion.ena"][-1])
        return recorded, actual_gradients


def test_preconfigured_ion_sources_styles_and_rate_gradients(cpu_threads):
    _exercise_ion_sources()
