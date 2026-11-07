"""Packed finite-extracellular functional execution and independent oracles.

The reference advances each component separately, rather than using the packed
forest implementation being tested. All recording retains both membrane and
circuit voltages, including nonzero extracellular shell carry.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators import dhs_bt
from dendra.models.integrators.tree_bt import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mod import hh

DT = 0.01
STEPS = 4
CHANNELS = ("gnabar", "gkbar", "gl", "ena", "ek", "el")
RECORD = ("v", "vc", "hh.m", "hh.h", "hh.n", "t")

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        not DENDRA_SOLVERS_AVAILABLE,
        reason="packed ExtCell CPU execution requires dendra-solvers",
    ),
]


@pytest.fixture(params=(1, 4), ids=("threads1", "threads4"))
def cpu_threads(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    try:
        yield request.param
    finally:
        torch.set_num_threads(previous)


def _graph(*, zero_area_branchpoint=False):
    graph = nx.DiGraph()
    for node in range(4):
        is_junction = zero_area_branchpoint and node == 3
        graph.add_node(
            node,
            name=f"branch.{node}",
            kind="junction" if is_junction else "compartment",
            L=0.0 if is_junction else 9.0 + node,
            diam=1.5 + 0.2 * node,
            Ra=90.0 + 9.0 * node,
            cm=0.8 + 0.1 * node,
            area=0.0 if is_junction else 120.0 + 15.0 * node,
            xraxial=[1.8 + 0.2 * node, 2.4 + 0.3 * node],
            xc=[0.08 + 0.01 * node, 0.12 + 0.015 * node],
            xg=[1.4e-4 + 0.2e-4 * node, 2.4e-4 + 0.3e-4 * node],
        )
    # Storage order deliberately differs from topology order.
    for parent, child, resistance in ((3, 0, 1.7e8), (3, 2, 2.9e8), (2, 1, 4.3e8)):
        graph.add_edge(parent, child, R_ohm=resistance)
    return graph


def _components(*, device="cpu", dtype=torch.float64, zero_area_branchpoint=False):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DEVICE=device, DTYPE=dtype):
        axon = dn.ExtCellAxon(
            diameters=[5.0, 8.0],
            n_comp=3,
            celsius=6.3,
            v_init=torch.tensor([-66.0, -63.0, -60.0], device=device, dtype=dtype),
            dtype=dtype,
            integrator=dn.bwd_euler_bt(method="thomas", imem=False),
        )
        axon.dx.copy_(
            torch.tensor(
                [[8.0, 10.0, 12.0], [9.0, 11.0, 14.0]], device=device, dtype=dtype
            )
        )
        for name, lower, upper in (
            ("xraxial", 1.5, 3.0),
            ("xc", 0.08, 0.18),
            ("xg", 1.0e-4, 3.5e-4),
        ):
            value = getattr(axon, name)
            value.copy_(
                torch.linspace(
                    lower, upper, value.numel(), device=device, dtype=dtype
                ).reshape_as(value)
            )
        tree = dn.ExtCellTree.from_graph(
            _graph(zero_area_branchpoint=zero_area_branchpoint),
            N=2,
            celsius=6.3,
            v_init=torch.tensor(
                [-64.0, -61.0, -65.0, -62.0], device=device, dtype=dtype
            ),
            dtype=dtype,
            integrator=dhs_bt(threads=2, imem=False),
        )
        tree.rhoa_scale_param.set(1.15)
        tree.cm_scale_param.set(0.9)
        tree.area_scale_param.set(1.1)
        for component in (axon, tree):
            component.insert(hh)
    return {"axon": axon, "tree": tree}


def _seed_circuit(model):
    inner = torch.linspace(
        -1.5, 0.8, model.v.numel(), device=model.device(), dtype=model.dtype()
    ).reshape(model.shape)
    outer = torch.linspace(
        0.6, -1.0, model.v.numel(), device=model.device(), dtype=model.dtype()
    ).reshape(model.shape)
    model.vc = torch.stack((model.v + inner, inner, outer), dim=-1)


def _model(
    *,
    device="cpu",
    dtype=torch.float64,
    batch_calls=(),
    write_back=True,
    zero_area_branchpoint=False,
    shell_resistance_scale=1.0,
):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DEVICE=device, DTYPE=dtype):
        components = _components(
            device=device, dtype=dtype, zero_area_branchpoint=zero_area_branchpoint
        )
        for component in components.values():
            component.xraxial.mul_(shell_resistance_scale)
        model = dn.concat_models(
            components,
            celsius=6.3,
            threads=2,
            write_back=write_back,
        )
        for size in batch_calls:
            model.batch(size)
        model.initialize()
        model.train()
        _seed_circuit(model)
    return model


def _clone(value):
    return pytree.tree_map(lambda tensor: tensor.detach().clone(), value)


def _drives(model, steps=STEPS):
    count = steps * model.v.numel()
    arguments = {"device": model.device(), "dtype": model.dtype()}
    ve = torch.linspace(-1.5, 2.0, count, **arguments).reshape(steps, *model.shape)
    intra = torch.linspace(-1.0e-9, 2.0e-9, count, **arguments).reshape(
        steps, *model.shape
    )
    return ve, intra


def _view(value, model, component, offset, *, tail=()):
    """Retain explicit batches while unpacking one component's core axes."""
    leading = tuple(model.shape[:-2])
    width = math.prod(component.core_shape())
    if tail:
        return value.reshape(*leading, model.shape[-1], *tail)[
            ..., offset : offset + width, :
        ].reshape(*component.shape, *tail)
    return value.reshape(*leading, model.shape[-1])[
        ..., offset : offset + width
    ].reshape(component.shape)


def _recorded(states):
    result = {
        field: torch.stack([state["integrator"][field] for state in states])
        for field in ("v", "vc")
    }
    result["t"] = torch.stack([state["clock"]["t"] for state in states])
    for gate in ("m", "h", "n"):
        result["hh." + gate] = torch.stack(
            [state["mechanisms"]["hh"][gate] for state in states]
        )
    return result


def _trajectory(functional, parameters, constants, state, ve, intra):
    prepared = functional.prepare(parameters, constants)
    states = [state]
    for index in range(ve.shape[0]):
        state, _ = functional.step(
            parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        states.append(state)
    return states


def _standalone_recorded(
    model, parameters, constants, state, ve, intra, *, references=None
):
    """Advance independent component plans with the exact same public leaves."""
    records = []
    offset = 0
    if references is None:
        references = _components(
            device=model.device(),
            dtype=model.dtype(),
            zero_area_branchpoint=bool(torch.any(model.populations["tree"].area == 0)),
        )
    for name in model.populations:
        component = references[name]
        # Never initialize or otherwise mutate children of the packed source.
        # Their runtime state is not authoritative when write_back=False.
        for size in reversed(model.shape[:-2]):
            component.batch(size)
        component.initialize()
        component.train()
        functional, tensors = dn.func.make_functional(component, dt=DT)
        prefix = f"populations.{name}."
        local_parameters = dict(tensors.parameters)
        for key in local_parameters:
            if prefix + key in parameters:
                local_parameters[key] = parameters[prefix + key]
            elif key in parameters:
                local_parameters[key] = parameters[key]
        local_constants = {
            key: constants.get(prefix + key, value)
            for key, value in tensors.constants.items()
        }
        local_state = _clone(tensors.state)
        local_state["integrator"] = {
            "v": _view(state["integrator"]["v"], model, component, offset),
            "vc": _view(state["integrator"]["vc"], model, component, offset, tail=(3,)),
        }
        for gate in ("m", "h", "n"):
            local_state["mechanisms"]["hh"][gate] = _view(
                state["mechanisms"]["hh"][gate], model, component, offset
            )
        local_state["clock"] = state["clock"]
        local_ve = torch.stack([_view(value, model, component, offset) for value in ve])
        local_intra = torch.stack(
            [_view(value, model, component, offset) for value in intra]
        )
        records.append(
            _recorded(
                _trajectory(
                    functional,
                    local_parameters,
                    local_constants,
                    local_state,
                    local_ve,
                    local_intra,
                )
            )
        )
        offset += math.prod(component.core_shape())
    assert offset == model.shape[-1]
    packed = {"t": records[0]["t"]}
    for field in RECORD:
        if field == "t":
            continue
        tail = (3,) if field == "vc" else ()
        parts = [
            record[field].reshape(ve.shape[0] + 1, *model.shape[:-2], -1, *tail)
            for record in records
        ]
        packed[field] = torch.cat(parts, dim=-2 if tail else -1).reshape(
            ve.shape[0] + 1, *model.shape, *tail
        )
    return packed


def _arguments(model, tensors):
    parameters, constants, state = map(
        _clone, (tensors.parameters, tensors.constants, tensors.state)
    )
    targets = {}
    for channel in CHANNELS:
        key = f"integrator.mech.mechanisms.hh.{channel}_param"
        targets[channel] = parameters[key].requires_grad_()
    for component in ("axon", "tree"):
        for field in ("xraxial", "xc", "xg"):
            key = f"populations.{component}.{field}"
            targets[f"{component}.{field}"] = constants[key].requires_grad_()
    for component, field in (
        ("axon", "dx"),
        ("tree", "canonical_area_cm2"),
        ("tree", "canonical_edge_resistance_ohm"),
    ):
        key = f"populations.{component}.{field}"
        targets[f"{component}.{field}"] = constants[key].requires_grad_()
    for field in ("cm_scale_param.rho", "rhoa_scale_param.rho"):
        key = f"populations.tree.{field}"
        targets[f"tree.{field}"] = parameters[key].requires_grad_()
    voltage = state["integrator"]["v"].requires_grad_()
    shells = state["integrator"]["vc"][..., 1:].detach().clone().requires_grad_()
    state["integrator"]["vc"] = torch.cat(
        (voltage.unsqueeze(-1) + shells[..., :1], shells), dim=-1
    )
    targets["initial_voltage"] = voltage
    targets["initial_shells"] = shells
    ve, intra = _drives(model)
    targets["bath_stimulus"] = ve.requires_grad_()
    targets["intracellular_stimulus"] = intra.requires_grad_()
    return parameters, constants, state, ve, intra, targets


def _loss(recorded):
    weights = torch.linspace(
        0.6,
        1.4,
        recorded["v"].shape[-1],
        device=recorded["v"].device,
        dtype=recorded["v"].dtype,
    )
    return (
        (recorded["v"].square() * weights).mean()
        + 0.03 * (recorded["vc"].square() * weights.unsqueeze(-1)).mean()
        + 0.01 * recorded["hh.m"].mean()
    )


def _gradients(recorded, targets):
    values = torch.autograd.grad(_loss(recorded), tuple(targets.values()))
    return dict(zip(targets, values, strict=True))


def _close(actual, expected, *, dtype=torch.float64, cross_device=False):
    def canonical(value):
        if isinstance(value, dict):
            return {key: canonical(value[key]) for key in sorted(value)}
        if isinstance(value, list):
            return [canonical(item) for item in value]
        if isinstance(value, tuple):
            return tuple(canonical(item) for item in value)
        return value

    actual, expected = canonical(actual), canonical(expected)
    actual_leaves, actual_spec = pytree.tree_flatten_with_path(actual)
    expected_leaves, expected_spec = pytree.tree_flatten_with_path(expected)
    assert actual_spec == expected_spec
    for (path, value), (other_path, reference) in zip(
        actual_leaves, expected_leaves, strict=True
    ):
        assert path == other_path
        assert value.shape == reference.shape
        assert torch.isfinite(value).all() and torch.isfinite(reference).all(), path
        if cross_device:
            value, reference = value.cpu().double(), reference.cpu().double()
        torch.testing.assert_close(
            value,
            reference,
            rtol=5.0e-4 if dtype == torch.float32 else 3.0e-8,
            atol=3.0e-4 if dtype == torch.float32 else 3.0e-9,
            msg=lambda message: f"{path}: {message}",
        )


def _close_gradients(actual, expected, *, dtype=torch.float64, cross_device=False):
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        reference = expected[name]
        assert value.shape == reference.shape
        assert torch.isfinite(value).all() and torch.isfinite(reference).all(), name
        if cross_device:
            value, reference = value.cpu(), reference.cpu()
        norm = torch.linalg.vector_norm(reference.double())
        assert norm > 0, name
        error = torch.linalg.vector_norm((value.double() - reference.double()))
        relative = error / norm
        assert relative <= (2.0e-3 if dtype == torch.float32 else 2.0e-7), (
            name,
            relative.item(),
        )


@pytest.mark.parametrize("batch_calls", [(), (2,), (2, 2)])
def test_extcell_multi_full_v_vc_and_gate_traces_match_independent_components(
    cpu_threads, batch_calls
):
    model = _model(batch_calls=batch_calls)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    assert set(tensors.state["integrator"]) == {"v", "vc"}
    assert tensors.state["integrator"]["vc"].shape == (*model.shape, 3)
    assert torch.count_nonzero(tensors.state["integrator"]["vc"][..., 1:])
    ve, intra = _drives(model)
    actual = _recorded(
        _trajectory(
            functional, tensors.parameters, tensors.constants, tensors.state, ve, intra
        )
    )
    expected = _standalone_recorded(
        model, tensors.parameters, tensors.constants, tensors.state, ve, intra
    )
    _close(actual, expected)
    torch.testing.assert_close(actual["v"], actual["vc"][..., 0] - actual["vc"][..., 1])


@pytest.mark.parametrize("batch_calls", [(), (2,)])
def test_extcell_multi_all_source_gradients_match_independent_components(
    cpu_threads, batch_calls
):
    model = _model(batch_calls=batch_calls)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters, constants, state, ve, intra, targets = _arguments(model, tensors)
    actual = _recorded(_trajectory(functional, parameters, constants, state, ve, intra))
    actual_gradients = _gradients(actual, targets)
    parameters, constants, state, ve, intra, targets = _arguments(model, tensors)
    expected = _standalone_recorded(model, parameters, constants, state, ve, intra)
    expected_gradients = _gradients(expected, targets)
    assert len(actual_gradients) == 21
    _close(actual, expected)
    _close_gradients(actual_gradients, expected_gradients)


def test_extcell_multi_float32_matches_float64_full_trace_and_gradients(cpu_threads):
    results = []
    for dtype in (torch.float64, torch.float32):
        # The original low-resistance shell fixture stays in the float64
        # independent-component and compiled checks. Its implicit matrix has
        # condition number about 9.4e5: both standalone and packed float32 lose
        # common-mode shell accuracy. These finite shell resistances reduce
        # that condition number to about 9.4e3 without dropping gradient targets
        # or relaxing tolerances.
        model = _model(dtype=dtype, shell_resistance_scale=100.0)
        functional, tensors = dn.func.make_functional(model, dt=DT)
        parameters, constants, state, ve, intra, targets = _arguments(model, tensors)
        recorded = _recorded(
            _trajectory(functional, parameters, constants, state, ve, intra)
        )
        results.append((recorded, _gradients(recorded, targets)))
    expected, actual = results
    _close(actual[0], expected[0], dtype=torch.float32, cross_device=True)
    _close_gradients(actual[1], expected[1], dtype=torch.float32, cross_device=True)


@pytest.mark.parametrize("write_back", [True, False])
def test_extcell_multi_commit_syncs_both_voltage_views_and_resumes(
    cpu_threads, write_back
):
    model = _model(batch_calls=(2,), write_back=write_back)
    target = _model(batch_calls=(2,), write_back=write_back)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model)
    states = _trajectory(
        functional, tensors.parameters, tensors.constants, tensors.state, ve, intra
    )
    child_before = {
        name: tuple(
            getattr(component, field).clone() if hasattr(component, field) else None
            for field in ("v", "vc")
        )
        for name, component in target.populations.items()
    }
    malformed = _clone(states[2])
    malformed["integrator"]["vc"] = malformed["integrator"]["vc"][..., :2]
    with pytest.raises((ValueError, dn.func.FunctionalizationError)):
        functional.commit_state_(target, malformed)
    _close(functional.extract(target).state, tensors.state)
    functional.commit_state_(target, states[2])
    _close(functional.extract(target).state, states[2])
    offset = 0
    for name, component in target.populations.items():
        for field, before in zip(("v", "vc"), child_before[name], strict=True):
            if not write_back and before is None:
                assert not hasattr(component, field)
                continue
            expected = (
                _view(
                    states[2]["integrator"][field],
                    target,
                    component,
                    offset,
                    tail=(3,) if field == "vc" else (),
                )
                if write_back
                else before
            )
            torch.testing.assert_close(
                getattr(component, field), expected, rtol=0, atol=0
            )
        offset += math.prod(component.core_shape())
    timestep = torch.as_tensor(DT, device=target.device(), dtype=target.dtype())
    target.integrator._initialize(
        target, timestep, force=True, compile_scope="population"
    )
    for index in range(2, STEPS):
        target.integrator.step(target, timestep, ve[index], intra[index])
        target.t = target.t + timestep
    _close(functional.extract(target).state, states[-1])


def test_extcell_multi_prepared_constants_and_state_schema_are_checked(cpu_threads):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    constants = _clone(tensors.constants)
    prepared = functional.prepare(tensors.parameters, constants)
    constants["populations.tree.xg"].add_(1.0e-5)
    with pytest.raises(dn.func.FunctionalizationError, match="constants changed"):
        functional.step(tensors.parameters, prepared, tensors.state)
    refreshed = functional.prepare(tensors.parameters, constants)
    functional.step(tensors.parameters, refreshed, tensors.state)
    malformed = _clone(tensors.state)
    malformed["integrator"].pop("vc")
    with pytest.raises((KeyError, ValueError, dn.func.FunctionalizationError)):
        functional.step(tensors.parameters, refreshed, malformed)


def test_extcell_multi_preserves_zero_area_branchpoint_physics(cpu_threads):
    model = _model(zero_area_branchpoint=True)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    assert (
        torch.count_nonzero(
            tensors.constants["populations.tree.canonical_area_cm2"] == 0
        )
        == 2
    )
    ve, intra = _drives(model)
    actual = _recorded(
        _trajectory(
            functional, tensors.parameters, tensors.constants, tensors.state, ve, intra
        )
    )
    expected = _standalone_recorded(
        model, tensors.parameters, tensors.constants, tensors.state, ve, intra
    )
    _close(actual, expected)


def _one_node_components():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False):
        graph = _graph().subgraph([3]).copy()
        tree = dn.ExtCellTree.from_graph(
            graph,
            N=2,
            celsius=6.3,
            dtype=torch.float64,
            v_init=-64.0,
            integrator=dhs_bt(threads=2, imem=False),
        )
        axon = dn.ExtCellAxon(
            diameters=[5.0, 8.0],
            n_comp=1,
            celsius=6.3,
            dtype=torch.float64,
            v_init=-64.0,
            integrator=dn.bwd_euler_bt(method="thomas", imem=False),
        )
        for component in (axon, tree):
            component.insert(hh)
    return {"axon": axon, "tree": tree}


@pytest.mark.parametrize("component_names", [("axon",), ("tree",), ("axon", "tree")])
def test_extcell_multi_homogeneous_and_empty_edge_forests(cpu_threads, component_names):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False):
        components = _one_node_components()
        model = dn.concat_models(
            {name: components[name] for name in component_names},
            celsius=6.3,
            threads=2,
        )
        model.batch(2)
        model.initialize()
        model.train()
        _seed_circuit(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model)
    actual = _recorded(
        _trajectory(
            functional, tensors.parameters, tensors.constants, tensors.state, ve, intra
        )
    )
    expected = _standalone_recorded(
        model,
        tensors.parameters,
        tensors.constants,
        tensors.state,
        ve,
        intra,
        references=_one_node_components(),
    )
    _close(actual, expected)


def test_extcell_multi_rejects_tampered_voltage_state_contract(cpu_threads):
    model = _model()
    model.integrator.v_vars = ["v"]
    with pytest.raises(dn.func.FunctionalizationError, match="v.*vc|block"):
        dn.func.make_functional(model, dt=DT)


@pytest.mark.parametrize("inference_tensors", [False, True])
def test_extcell_multi_changed_child_shell_physics_requires_reinitialize(
    cpu_threads, inference_tensors
):
    with (
        dn.ctx(JIT=0, REQUIRE_GRAD=0, IMEM=False),
        torch.inference_mode(inference_tensors),
    ):
        model = _model()
        model.requires_grad_(False)
        model.eval()
        ve, _ = _drives(model, steps=1)
        model.run(ve=ve, dt=DT)
        model.populations["tree"].xg.add_(1.0e-5)
        with pytest.raises(RuntimeError, match="initialize"):
            model.run(ve=ve, dt=DT)
        model.initialize()
        model.run(ve=ve, dt=DT)
        assert torch.isfinite(model.vc).all()


def test_extcell_multi_failed_commit_restores_exact_v_and_vc_bindings(
    cpu_threads, monkeypatch
):
    source = _model(batch_calls=(2,))
    target = _model(batch_calls=(2,))
    functional, tensors = dn.func.make_functional(source, dt=DT)
    ve, intra = _drives(source)
    checkpoint = _trajectory(
        functional, tensors.parameters, tensors.constants, tensors.state, ve, intra
    )[-1]
    # Both packed voltages and the child views carry graphs whose identities
    # must survive a failure after the parent checkpoint has already restored.
    for field in ("v", "vc"):
        leaf = getattr(target, field).detach().clone().requires_grad_()
        setattr(target, field, leaf + leaf.new_zeros(()))
    offset = 0
    for component in target.populations.values():
        for field in ("v", "vc"):
            setattr(
                component,
                field,
                _view(
                    getattr(target, field),
                    target,
                    component,
                    offset,
                    tail=(3,) if field == "vc" else (),
                ),
            )
        offset += math.prod(component.core_shape())
    before_state = functional.extract(target).state
    before_buffers = dict(target.named_buffers(remove_duplicate=False))
    before_voltages = {
        (name, field): getattr(component, field)
        for name, component in target.populations.items()
        for field in ("v", "vc")
    }
    original_restore = target.restore_dict_from_checkpoint
    restore_calls = 0

    def failing_restore(checkpoint):
        nonlocal restore_calls
        restore_calls += 1
        original_restore(checkpoint)
        raise RuntimeError("injected post-restore failure")

    original_validate = functional._validate_compatible_population
    with monkeypatch.context() as fault:
        # Canonical checkpoint hooks are checked before commit. Inject the
        # restore failure after that preflight to exercise the transaction.
        def validate_then_inject(population):
            original_validate(population)
            fault.setitem(
                target.__dict__, "restore_dict_from_checkpoint", failing_restore
            )

        fault.setitem(
            functional.__dict__,
            "_validate_compatible_population",
            validate_then_inject,
        )
        with pytest.raises(RuntimeError, match="injected post-restore failure"):
            functional.commit_state_(target, checkpoint)
    assert restore_calls == 1
    _close(functional.extract(target).state, before_state)
    for name, tensor in target.named_buffers(remove_duplicate=False):
        assert tensor is before_buffers[name]
    for (name, field), tensor in before_voltages.items():
        assert getattr(target.populations[name], field) is tensor


def _compiled_recorded(
    functional,
    parameters,
    constants,
    state,
    ve,
    intra,
    *,
    backend="inductor",
    options=None,
):
    kernel = functional.compile_rollout_chunk(
        STEPS, execution="scan", backend=backend, options=options, fullgraph=True
    )
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(RECORD)})
    final, auxiliary = dn.func.run(
        kernel.bind(parameters, functional.prepare(parameters, constants)),
        state,
        inputs=dn.func.RolloutInput(ve=ve, intra=intra),
        steps=STEPS,
        callbacks=callbacks,
    )
    assert kernel.execution_report().strategy == "compiled_scan"
    return final, dict(auxiliary["callbacks"]["trace"])


@pytest.mark.slow
def test_extcell_multi_cpp_scan_full_recording_and_all_gradients(cpu_threads):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters, constants, state, ve, intra, targets = _arguments(model, tensors)
    expected_states = _trajectory(functional, parameters, constants, state, ve, intra)
    expected = _recorded(expected_states)
    expected_gradients = _gradients(expected, targets)
    parameters, constants, state, ve, intra, targets = _arguments(model, tensors)
    with torch_compiler_warning_context():
        final, actual = _compiled_recorded(
            functional,
            parameters,
            constants,
            state,
            ve,
            intra,
            options={"cpp_wrapper": True},
        )
        gradients = _gradients(actual, targets)
    _close(final, expected_states[-1])
    _close(actual, expected)
    _close_gradients(gradients, expected_gradients)


@pytest.mark.slow
def test_extcell_multi_cpp_while_full_recording(cpu_threads):
    model = _model(batch_calls=(2,))
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model)
    expected_states = _trajectory(
        functional, tensors.parameters, tensors.constants, tensors.state, ve, intra
    )
    runner = dn.func.compile(
        model,
        dt=DT,
        steps=STEPS,
        execution="while",
        inference=True,
        callbacks={"trace": dn.func.Recorder(RECORD)},
    )
    with torch_compiler_warning_context():
        final, auxiliary = runner(inputs=dn.func.RolloutInput(ve=ve, intra=intra))
    assert runner.execution_report().strategy == "compiled_while_loop"
    assert runner.kernel._compile_options["options"]["cpp_wrapper"] is True
    _close(final, expected_states[-1])
    _close(dict(auxiliary["callbacks"]["trace"]), _recorded(expected_states))
