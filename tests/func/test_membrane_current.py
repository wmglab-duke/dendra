"""Opt-in accepted-step membrane currents retain imperative solver semantics."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators import dhs_bt
from dendra.models.integrators.cable import (
    _cylindrical_edge_conductance,
    _cylindrical_membrane_area,
    _layered_edge_conductance,
)
from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE
from dendra.models.integrators.tree_bt import _tree_layered_edge_conductance
from dendra.models.mod import hh
from dendra.models.multi import MultiPopulation
from tests.func.test_extcell_multi_population import _graph
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _module_snapshot,
)
from tests.func.test_native_cable import _morphology

DT = 0.01
STEPS = 4
CHANNELS = ("gnabar", "gkbar", "gl", "ena", "ek", "el")
KINDS = (
    "point",
    "single",
    "path",
    "myelinated",
    "cable",
    "tree",
    "block_path",
    "block_tree",
    "multi_point",
    "multi_scalar",
    "multi_block",
)
NATIVE_KINDS = {"tree", "block_path", "block_tree", "multi_scalar", "multi_block"}
pytestmark = pytest.mark.cpu


@pytest.fixture(params=(1, 4), ids=("threads1", "threads4"))
def cpu_threads(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    try:
        yield request.param
    finally:
        torch.set_num_threads(previous)


def _model(
    kind="point",
    *,
    device="cpu",
    dtype=torch.float64,
    imem=True,
    batch=2,
    method="pcr",
):
    if device == "cpu" and kind in NATIVE_KINDS and not DENDRA_SOLVERS_AVAILABLE:
        pytest.skip("this topology requires native CPU dendra-solvers")
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=imem, DEVICE=device, DTYPE=dtype):
        common = {"celsius": 6.3, "dtype": dtype, "v_init": -63.0}
        if kind in {"point", "single"}:
            cls = dn.Population if kind == "point" else dn.SingleCompartment
            model = cls(N=2, C=3, integrator=dn.bwd_euler_sc(imem=imem), **common)
        elif kind == "path":
            model = dn.Unmyelinated(
                [2.0, 3.0],
                L=20.0,
                dx=10.0,
                integrator=dn.bwd_euler_ub(method=method, imem=imem),
                **common,
            )
        elif kind == "myelinated":
            model = dn.Myelinated(
                [8.0, 12.0],
                n_node=3,
                integrator=dn.bwd_euler_ub(method=method, imem=imem),
                **common,
            )
        elif kind == "cable":
            model = dn.Cable.from_morphology(
                _morphology(),
                N=2,
                integrator=dn.bwd_euler_ub(method=method, imem=imem),
                **common,
            )
        elif kind == "tree":
            model = dn.Tree.from_graph(
                _graph(zero_area_branchpoint=True),
                N=2,
                integrator=dn.dhs(threads=2, imem=imem),
                **common,
            )
        elif kind == "block_path":
            model = dn.ExtCellAxon(
                [5.0, 8.0],
                n_comp=3,
                integrator=dn.bwd_euler_bt(method="thomas", imem=imem),
                **common,
            )
            # Exercise finite shell dynamics, rather than the constructor's
            # near-ideal bath clamp whose xg sensitivity is nearly zero.
            for field, lower, upper in (
                ("dx", 8.0, 14.0),
                ("xraxial", 1.5, 3.0),
                ("xc", 0.08, 0.18),
                ("xg", 1.0e-4, 3.0e-4),
            ):
                value = getattr(model, field)
                value.copy_(
                    torch.linspace(
                        lower, upper, value.numel(), device=device, dtype=dtype
                    ).reshape_as(value)
                )
        elif kind == "block_tree":
            model = dn.ExtCellTree.from_graph(
                _graph(zero_area_branchpoint=True),
                N=2,
                integrator=dhs_bt(threads=2, imem=imem),
                **common,
            )
        else:
            component_kinds = {
                "multi_point": ("point", "single"),
                "multi_scalar": ("point", "path", "tree"),
                "multi_block": ("block_path", "block_tree"),
            }[kind]
            components = {
                name: _model(name, device=device, dtype=dtype, imem=imem, batch=None)
                for name in component_kinds
            }
            model = dn.concat_models(components, celsius=6.3, threads=2)
        if not kind.startswith("multi_"):
            model.insert(hh)
        if batch is not None:
            model.batch(batch)
        model.initialize()
        model.train()
        # Nonuniform membrane and shell potentials expose ordering mistakes.
        model.v = torch.linspace(
            -66.0, -58.0, model.v.numel(), device=device, dtype=dtype
        ).reshape(model.shape)
        if hasattr(model, "vc"):
            inner = torch.linspace(
                -0.4, 0.6, model.v.numel(), device=device, dtype=dtype
            ).reshape(model.shape)
            model.vc = torch.stack((model.v + inner, inner, -0.3 * inner), dim=-1)
    return model


def _clone(value):
    return pytree.tree_map(lambda tensor: tensor.detach().clone(), value)


def _inputs(model, steps=STEPS):
    options = {"device": model.device(), "dtype": model.dtype()}
    shape = (steps, *model.shape)
    count = steps * model.v.numel()
    return (
        torch.linspace(-0.5, 0.8, count, **options).reshape(shape),
        torch.linspace(-2.0e-9, 3.0e-9, count, **options).reshape(shape),
    )


def _parameter_names(parameters):
    channels = [f"integrator.mech.mechanisms.hh.{name}_param" for name in CHANNELS]
    # A raw capacitance (or capacitance multiplier for canonical geometry) is
    # an independent physical-parameter target, not a prepared-workspace leaf.
    prefix = (
        "populations."
        if any(name.startswith("populations.") for name in parameters)
        else ""
    )
    cm = next(
        name
        for name in parameters
        if name.startswith(prefix)
        and name.endswith(("cm_param.rho", "cm_param", "cm_scale_param.rho"))
    )
    return (*channels, cm)


def _observable(model):
    names = ("v", "i_membrane", "vc") if hasattr(model, "vc") else ("v", "i_membrane")
    result = {name: getattr(model, name) for name in names}
    result.update(
        {f"hh.{name}": getattr(model.mech.hh, name) for name in ("m", "h", "n")}
    )
    return result


def _loss(recorded):
    # Current is in mA, so express its contribution in nA for a meaningful
    # derivative signal beside membrane voltage. Include nonuniform weights.
    current = recorded["i_membrane"] * 1.0e6
    weights = torch.linspace(
        0.7, 1.3, current.shape[-1], device=current.device, dtype=current.dtype
    )
    return (
        (weights * current.square()).mean()
        + 0.001 * recorded["v"].square().mean()
        + 0.1 * recorded["hh.m"].mean()
    )


def _functional_result(model, *, compiled=None):
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters, constants, state = map(
        _clone, (tensors.parameters, tensors.constants, tensors.state)
    )
    targets = {
        name: parameters[name].requires_grad_() for name in _parameter_names(parameters)
    }
    targets["initial_voltage"] = state["integrator"]["v"].requires_grad_()
    if "vc" in state["integrator"]:
        vc = state["integrator"]["vc"]
        state["integrator"]["vc"] = torch.cat(
            (state["integrator"]["v"][..., None] + vc[..., 1:2], vc[..., 1:]),
            dim=-1,
        )
    ve, drive = _inputs(model)
    amplitude = model.v.new_ones((), requires_grad=True)
    targets["stimulus"] = amplitude
    intra = drive * amplitude
    prepared = functional.prepare(parameters, constants)
    names = tuple(_observable(model))
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(names)})
    if compiled is None:
        step = functional.bind(parameters, prepared)
    else:
        step = functional.compile_rollout_chunk(
            STEPS,
            execution=compiled,
            backend="inductor",
            fullgraph=True,
            options={
                "cpp_wrapper": model.device().type == "cpu",
                "compile_threads": 1,
            },
        ).bind(parameters, prepared)
    _, auxiliary = dn.func.run(
        step,
        state,
        steps=STEPS,
        inputs=dn.func.RolloutInput(ve=ve, intra=intra),
        callbacks=callbacks,
    )
    recorded = dict(auxiliary["callbacks"]["trace"])
    gradients = torch.autograd.grad(_loss(recorded), tuple(targets.values()))
    return recorded, dict(zip(targets, gradients, strict=True))


def _imperative_result(model):
    parameters = dict(model.named_parameters())
    targets = {
        name: parameters[name].requires_grad_() for name in _parameter_names(parameters)
    }
    model.populate_parameter_buffers()
    model.v = model.v.detach().clone().requires_grad_()
    targets["initial_voltage"] = model.v
    if hasattr(model, "vc"):
        model.vc = torch.cat(
            (model.v[..., None] + model.vc[..., 1:2], model.vc[..., 1:]), dim=-1
        )
    ve, drive = _inputs(model)
    amplitude = model.v.new_ones((), requires_grad=True)
    targets["stimulus"] = amplitude
    intra = drive * amplitude
    dt = model.v.new_tensor(DT)
    model.integrator._initialize(model, dt, force=True, compile_scope="population")
    states = [_observable(model)]
    for index in range(STEPS):
        model.integrator.step(model, dt, ve[index], intra[index])
        states.append(_observable(model))
    recorded = {
        name: torch.stack([state[name] for state in states]) for name in states[0]
    }
    gradients = torch.autograd.grad(_loss(recorded), tuple(targets.values()))
    return recorded, dict(zip(targets, gradients, strict=True))


def _close(actual, expected, *, dtype=torch.float64):
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        reference = expected[name]
        assert torch.isfinite(value).all() and torch.isfinite(reference).all(), name
        if name == "i_membrane":
            # Public current is in mA; compare in nA so the absolute tolerance
            # cannot silently accept a wrong (or identically zero) small trace.
            value, reference = value * 1.0e6, reference * 1.0e6
        torch.testing.assert_close(
            value,
            reference,
            rtol=1.0e-4 if dtype == torch.float32 else 2.0e-9,
            atol=1.0e-6 if dtype == torch.float32 else 2.0e-10,
            msg=lambda message: f"{name}: {message}",
        )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_membrane_current_full_traces_and_gradients(kind, dtype, cpu_threads):
    source = _model(kind, dtype=dtype)
    snapshot = _module_snapshot(source)
    actual, gradients = _functional_result(source)
    expected, expected_gradients = _imperative_result(_model(kind, dtype=dtype))
    _close(actual, expected, dtype=dtype)
    _close(gradients, expected_gradients, dtype=dtype)
    assert actual["i_membrane"].shape == actual["v"].shape
    assert torch.count_nonzero(actual["i_membrane"][0]) == 0
    _assert_module_snapshot(source, snapshot)


@pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE,
    reason="native scalar Thomas requires dendra-solvers",
)
def test_native_thomas_membrane_current_full_traces_and_gradients(cpu_threads):
    source = _model("path", method="thomas")
    source.integrator._initialize(
        source, source.v.new_tensor(DT), force=True, compile_scope="population"
    )
    # Fail rather than accidentally validating a fallback under this name.
    assert source.integrator._solve is torch.ops.dendra_solvers.thomas_solve_t
    snapshot = _module_snapshot(source)
    actual, gradients = _functional_result(source)
    expected, reference_gradients = _imperative_result(_model("path", method="thomas"))
    assert len(gradients) == 9  # Standard eight plus raw physical capacitance.
    _close(actual, expected)
    _close(gradients, reference_gradients)
    _assert_module_snapshot(source, snapshot)


@pytest.mark.parametrize(
    "kind", ("point", "multi_point", "multi_scalar", "multi_block")
)
def test_membrane_current_commit_extract_and_restart(kind, cpu_threads):
    model = _model(kind)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _inputs(model)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    state, _ = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.StepInput(ve=ve[0], intra=intra[0]),
    )
    assert torch.count_nonzero(state["integrator"]["i_membrane"]) > 0
    functional.commit_state_(model, state)
    restored = functional.extract(model).state
    _close(restored["integrator"], state["integrator"])
    lowered_again, extracted = dn.func.make_functional(model, dt=DT)
    torch.testing.assert_close(
        extracted.state["integrator"]["i_membrane"],
        state["integrator"]["i_membrane"],
        rtol=0.0,
        atol=0.0,
    )
    resumed, _ = lowered_again.step(
        extracted.parameters,
        lowered_again.prepare(extracted.parameters, extracted.constants),
        extracted.state,
        dn.func.StepInput(ve=ve[1], intra=intra[1]),
    )
    uninterrupted, _ = functional.step(
        tensors.parameters, prepared, state, dn.func.StepInput(ve=ve[1], intra=intra[1])
    )
    _close(resumed["integrator"], uninterrupted["integrator"])


def test_membrane_current_fresh_initialization_and_disabled_schema(cpu_threads):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    tensors.state["integrator"]["i_membrane"].fill_(2.0)
    initialized = functional.initialize(
        tensors.parameters, tensors.constants, tensors.initialization
    )
    assert torch.count_nonzero(initialized.state["integrator"]["i_membrane"]) == 0
    disabled, disabled_tensors = dn.func.make_functional(_model(imem=False), dt=DT)
    assert "i_membrane" not in disabled_tensors.state["integrator"]
    with pytest.raises(dn.func.FunctionalizationError, match="unsupported selection"):
        disabled.make_callbacks({"trace": dn.func.Recorder("i_membrane")})


@pytest.mark.slow
def test_membrane_current_cpp_scan(cpu_threads):
    with torch_compiler_warning_context():
        actual, gradients = _functional_result(_model(), compiled="scan")
    expected, expected_gradients = _imperative_result(_model())
    _close(actual, expected)
    _close(gradients, expected_gradients)


@pytest.mark.slow
def test_membrane_current_cpp_while(cpu_threads):
    model = _model()
    expected, _ = _imperative_result(_model())
    ve, intra = _inputs(model)
    runner = dn.func.compile(
        model,
        dt=DT,
        steps=STEPS,
        execution="while",
        inference=True,
        callbacks={"trace": dn.func.Recorder(tuple(_observable(model)))},
    )
    with torch_compiler_warning_context():
        final, auxiliary = runner(inputs=dn.func.RolloutInput(ve=ve, intra=intra))
    assert runner.execution_report().strategy == "compiled_while_loop"
    _close(dict(auxiliary["callbacks"]["trace"]), expected)
    torch.testing.assert_close(
        final["integrator"]["i_membrane"], expected["i_membrane"][-1]
    )


@pytest.mark.parametrize("kind", ("point", "block_tree"))
def test_membrane_current_torch_func_transforms(kind, cpu_threads):
    model = _model(kind)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _inputs(model)

    def response(amplitude):
        state = tensors.state
        for index in range(STEPS):
            state, _ = functional.step(
                tensors.parameters,
                prepared,
                state,
                dn.func.StepInput(ve=ve[index], intra=intra[index] * amplitude),
            )
        return state["integrator"]["i_membrane"] * 1.0e6

    amplitude = model.v.new_tensor(0.8, requires_grad=True)
    reverse = torch.func.jacrev(response)(amplitude)
    forward = torch.func.jvp(response, (amplitude,), (torch.ones_like(amplitude),))[1]
    torch.testing.assert_close(reverse, forward, rtol=2.0e-9, atol=2.0e-10)
    assert torch.count_nonzero(reverse) > 0
    amplitudes = model.v.new_tensor([0.6, 1.1])
    batched = torch.func.vmap(response)(amplitudes)
    explicit = torch.stack([response(value) for value in amplitudes])
    torch.testing.assert_close(batched, explicit, rtol=2.0e-9, atol=2.0e-10)


@pytest.mark.parametrize("imem", (False, True))
def test_scalar_pack_reordered_tree_edges_match_independent_components(
    imem, cpu_threads
):
    model = _model("multi_scalar", imem=imem)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _inputs(model)
    references = []
    offset = 0
    for name, component in model.populations.items():
        reference = _model(name, imem=imem)
        width = component.numelc()
        selection = (..., 0, slice(offset, offset + width))
        reference.v = model.v[selection].reshape(reference.shape).clone()
        dt = reference.v.new_tensor(DT)
        reference.integrator._initialize(
            reference, dt, force=True, compile_scope="population"
        )
        references.append((reference, selection))
        offset += width
    state = tensors.state
    for index in range(STEPS):
        state, _ = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        for reference, selection in references:
            reference.integrator.step(
                reference,
                reference.v.new_tensor(DT),
                ve[index][selection].reshape(reference.shape),
                intra[index][selection].reshape(reference.shape),
            )
        names = ("v", "i_membrane") if imem else ("v",)
        for name in names:
            expected = torch.cat(
                [getattr(reference, name).flatten(-2) for reference, _ in references],
                dim=-1,
            ).unsqueeze(-2)
            _close({name: state["integrator"][name]}, {name: expected})


def _physical_arguments(model, tensors):
    parameters, constants, state = map(
        _clone, (tensors.parameters, tensors.constants, tensors.state)
    )
    targets = {}
    prefixes = (
        ("populations.block_path.", "populations.block_tree.")
        if isinstance(model, MultiPopulation)
        else ("",)
    )
    for prefix in prefixes:
        for name in ("cm_scale_param.rho", "rhoa_scale_param.rho"):
            key = prefix + name
            targets[key] = parameters[key].requires_grad_()
        fields = ["dx", "xraxial", "xc", "xg"]
        if prefix != "populations.block_path.":
            fields.extend(("canonical_area_cm2", "canonical_edge_resistance_ohm"))
        for name in fields:
            key = prefix + name
            targets[key] = constants[key].requires_grad_()
    return parameters, constants, state, targets


def _current_trace(functional, parameters, constants, state, ve, intra):
    prepared = functional.prepare(parameters, constants)
    currents = [state["integrator"]["i_membrane"]]
    for index in range(STEPS):
        state, _ = functional.step(
            parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        currents.append(state["integrator"]["i_membrane"])
    return torch.stack(currents)


def _imperative_physical_current_trace(model, parameters, constants, state, ve, intra):
    """Independent imperative rollout with explicit physical tensor inputs.

    Canonical Tree area/resistance are immutable Python geometry, so ordinary
    imperative initialization cannot differentiate those tensor inputs. Bind
    raw parameters independently, then derive/install the normal integrator
    workspace directly from those inputs before taking imperative steps. This
    avoids using FunctionalPopulation.prepare as its own gradient oracle.
    """
    for name, value in parameters.items():
        owner, _, leaf = name.rpartition(".")
        model.get_submodule(owner)._parameters[leaf] = value
    model.populate_parameter_buffers()
    integrator = model.integrator
    dt = model.v.new_tensor(DT)
    integrator._initialize(model, dt, force=True, compile_scope="population")

    def expanded(value, tail=()):
        return torch.broadcast_to(value, (*model.shape, *tail)).reshape(
            integrator.B, integrator.K, *tail
        )

    cm = expanded(model.cm * model.cm_scale)
    dx = expanded(constants["dx"])
    xraxial = expanded(constants["xraxial"], (2,))
    if isinstance(model, dn.ExtCellTree):
        child, parent = integrator.edge_child_orig, integrator.edge_parent_orig
        resistance = expanded(constants["canonical_edge_resistance_ohm"])
        rhoa_scale = expanded(model.rhoa_scale)
        intracellular = (
            resistance.index_select(-1, child) * rhoa_scale.index_select(-1, child)
        ).reciprocal()
        extracellular = _tree_layered_edge_conductance(xraxial, dx, child, parent)
        area = expanded(constants["canonical_area_cm2"]) * expanded(model.area_scale)
        ordering = {"edge_child_orig": child, "solver_order": integrator.solver_order}
    else:
        diam = expanded(constants["diam"])
        intracellular = _cylindrical_edge_conductance(
            diam, dx, expanded(model.rhoa * model.rhoa_scale)
        )
        extracellular = _layered_edge_conductance(xraxial, dx)
        area = _cylindrical_membrane_area(diam, dx) * expanded(model.area_scale)
        ordering = {}
    workspace = integrator._derive_prepared_workspace(
        dt,
        cm=cm,
        area=area,
        intracellular_edge_conductance=intracellular,
        extracellular_edge_conductance=extracellular,
        xc=expanded(constants["xc"], (2,)),
        xg=expanded(constants["xg"], (2,)),
        **ordering,
    )
    integrator._install_prepared_workspace(workspace)
    for name, value in state["integrator"].items():
        setattr(model, name, value)
    for gate in ("m", "h", "n"):
        setattr(model.mech.hh, gate, state["mechanisms"]["hh"][gate])
    currents = [model.i_membrane]
    for index in range(STEPS):
        integrator.step(model, dt, ve[index], intra[index])
        currents.append(model.i_membrane)
    return torch.stack(currents)


def _independent_physical_current_trace(model, parameters, constants, state, ve, intra):
    if not isinstance(model, MultiPopulation):
        return _imperative_physical_current_trace(
            _model("block_tree"), parameters, constants, state, ve, intra
        )
    records = []
    offset = 0
    for name, component in model.populations.items():
        reference = _model(name)
        width = component.numelc()
        prefix = f"populations.{name}."
        local_parameters = {
            key: parameters.get(prefix + key, parameters.get(key, value))
            for key, value in reference.named_parameters()
        }
        _, local_tensors = dn.func.make_functional(reference, dt=DT)
        local_constants = {
            key: constants.get(prefix + key, value)
            for key, value in local_tensors.constants.items()
        }

        def view(value, *, tail=()):
            selected = (
                value[..., 0, offset : offset + width, :]
                if tail
                else value[..., 0, offset : offset + width]
            )
            return selected.reshape(*reference.shape, *tail)

        local_state = {
            "integrator": {
                key: view(value, tail=(3,) if key == "vc" else ())
                for key, value in state["integrator"].items()
            },
            "mechanisms": {
                "hh": {
                    gate: view(state["mechanisms"]["hh"][gate])
                    for gate in ("m", "h", "n")
                }
            },
        }
        local_ve = torch.stack([view(value) for value in ve])
        local_intra = torch.stack([view(value) for value in intra])
        current = _imperative_physical_current_trace(
            reference,
            local_parameters,
            local_constants,
            local_state,
            local_ve,
            local_intra,
        )
        records.append(current.flatten(-2))
        offset += width
    return torch.cat(records, dim=-1).unsqueeze(-2)


@pytest.mark.parametrize("kind", ("block_tree", "multi_block"))
def test_current_only_physical_gradients_match_independent_imperative(
    kind, cpu_threads
):
    model = _model(kind)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _inputs(model)
    parameters, constants, state, targets = _physical_arguments(model, tensors)
    actual = _current_trace(functional, parameters, constants, state, ve, intra)

    def loss(current):
        current_na = current * 1.0e6
        weights = torch.linspace(
            0.7, 1.3, current.shape[-1], device=current.device, dtype=current.dtype
        )
        return (weights * current_na.square()).mean()

    gradients = torch.autograd.grad(loss(actual), tuple(targets.values()))
    parameters, constants, state, reference_targets = _physical_arguments(
        model, tensors
    )
    expected = _independent_physical_current_trace(
        model, parameters, constants, state, ve, intra
    )
    reference_gradients = torch.autograd.grad(
        loss(expected), tuple(reference_targets.values())
    )
    _close({"i_membrane": actual}, {"i_membrane": expected})
    for name, gradient, reference in zip(
        targets, gradients, reference_gradients, strict=True
    ):
        norm = torch.linalg.vector_norm(reference.double())
        assert norm > 0 and torch.isfinite(gradient).all(), name
        relative = torch.linalg.vector_norm((gradient - reference).double()) / norm
        assert relative <= 2.0e-7, (name, relative.item())
