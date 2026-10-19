"""Concatenation retains authored chemistry and independent component domains.

The reference advances the original component models separately. In particular,
each two-neuron child has another chemical row boundary inside its packed slice;
checking only boundaries between child objects would not catch that error.
"""

from __future__ import annotations

import networkx as nx
import pytest
import torch

import dendra as dn
from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mechanisms import (
    ClampProcess,
    ClearanceProcess,
    DiffusionProcess,
    ExchangeProcess,
)
from dendra.models.mod import hh

DT = 0.025
STEPS = 4
CHANNELS = ("gnabar", "gkbar", "gl", "ena", "ek", "el")
pytestmark = pytest.mark.cpu


class _ConcatClearance(ClearanceProcess):
    ClearanceProcess.GLOBAL(rate=0.35, target=0.8)
    ClearanceProcess.CLEAR("pool", field="amount", rate="rate", target="target")


class _ConcatClamp(ClampProcess):
    ClampProcess.GLOBAL(root_value=0.7, terminal_value=1.3)
    ClampProcess.SET("pool", field="amount", value="terminal_value", where="terminal")
    ClampProcess.SET("pool", field="amount", value="root_value", where="root")


class _ConcatExchange(ExchangeProcess):
    ExchangeProcess.GLOBAL(rate=0.4, volume_a=1.5, volume_b=2.3)
    ExchangeProcess.EXCHANGE(
        "pool.amount",
        "reservoir.amount",
        rate="rate",
        volume_a="volume_a",
        volume_b="volume_b",
    )


class _ConcatDiffusion(DiffusionProcess):
    DiffusionProcess.GLOBAL(D=0.65)
    DiffusionProcess.METHOD("implicit", solver="auto")
    DiffusionProcess.DIFFUSE("pool", field="amount", D="D")


class _ConcatNamedDiffusion(DiffusionProcess):
    DiffusionProcess.GLOBAL(D=0.65)
    DiffusionProcess.METHOD("implicit", solver="auto")
    DiffusionProcess.DIFFUSE("pool", field="amount", D="D", geometry="chemical")


PROCESSES = {
    "clearance": _ConcatClearance,
    "clamp": _ConcatClamp,
    "exchange": _ConcatExchange,
    "diffusion": _ConcatDiffusion,
    "named_diffusion": _ConcatNamedDiffusion,
}


@pytest.fixture(params=(1, 4), ids=("threads1", "threads4"))
def cpu_threads(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    try:
        yield request.param
    finally:
        torch.set_num_threads(previous)


def _graph():
    graph = nx.DiGraph()
    for node in range(4):
        graph.add_node(
            node,
            name=f"chemical.{node}",
            kind="compartment",
            L=1.0,
            diam=2.0,
            Ra=100.0,
            cm=1.0,
            area=6.0 + node,
            volume_i=1.0 + 0.2 * node,
            volume_o=0.0,
            x=float(node),
            y=0.0,
            z=0.0,
        )
    # Neither the root nor topological order matches flattened storage order.
    for parent, child in ((3, 0), (3, 2), (2, 1)):
        graph.add_edge(parent, child, diff_geom_um=0.8, R_ohm=1e8, L=1.0)
    return graph


def _components(
    family="diffusion",
    *,
    kinds=("path", "tree"),
    missing_material=False,
    missing_process=False,
    device="cpu",
    dtype=torch.float64,
):
    if device == "cpu" and "tree" in kinds and not DENDRA_SOLVERS_AVAILABLE:
        pytest.skip("packed Tree CPU execution requires dendra-solvers")
    result = {}
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True, DEVICE=device, DTYPE=dtype):
        for index, kind in enumerate(kinds):
            common = {"celsius": 6.3, "dtype": dtype}
            if kind == "tree":
                component = dn.Tree.from_graph(
                    _graph(), N=2, integrator=dn.dhs(threads=1, imem=True), **common
                )
            else:
                component = dn.Population(
                    N=2, C=3, integrator=dn.bwd_euler_sc(imem=True), **common
                )
                component.diam.fill_(2.0)
                component.dx.copy_(
                    torch.linspace(0.8, 1.3, 6, device=device, dtype=dtype).reshape(
                        2, 3
                    )
                )
            component.insert(hh)
            voltage = torch.nn.Parameter(
                torch.tensor(-65.0 + index, device=device, dtype=dtype)
            )
            component.set_v_init(voltage)
            sources = {"initial_voltage": voltage}
            if not (missing_material and index == len(kinds) - 1):
                initial = torch.nn.Parameter(
                    torch.linspace(
                        0.5 + 2 * index,
                        1.6 + 2 * index,
                        component.v.numel(),
                        device=device,
                        dtype=dtype,
                    ).reshape(component.shape)
                )
                component.material(
                    "pool",
                    fields={"amount": initial},
                    min_values={"amount": 0.0},
                    domain="intracellular",
                    units="mM",
                    conserved=True,
                )
                sources["initial_pool"] = initial
                if family == "exchange":
                    reservoir = torch.nn.Parameter(initial.detach().clone() * 0.3)
                    component.material(
                        "reservoir",
                        fields={"amount": reservoir},
                        min_values={"amount": 0.0},
                        domain="membrane",
                    )
                    sources["initial_reservoir"] = reservoir
                if family == "named_diffusion":
                    volume = torch.nn.Parameter(
                        torch.ones_like(initial) * (1.2 + index)
                    )
                    edge = torch.nn.Parameter(
                        initial.new_full((2, component.shape[-1] - 1), 0.9 + index)
                    )
                    component.register_material_geometry(
                        "chemical",
                        domain="intracellular",
                        volume=volume,
                        edge_factor=edge,
                    )
                    sources.update(chemical_volume=volume, chemical_edge_factor=edge)
                if not (missing_process and index == len(kinds) - 1):
                    # GLOBAL means global within its authored child, not one
                    # shared bath/rate imposed on every concatenated cell.
                    overrides = {
                        "clearance": {"rate": 0.35 + 0.1 * index},
                        "clamp": {"root_value": 0.7 + 0.1 * index},
                        "exchange": {"rate": 0.4 + 0.1 * index},
                        "diffusion": {"D": 0.65 + 0.15 * index},
                        "named_diffusion": {"D": 0.65 + 0.15 * index},
                    }[family]
                    component.insert(PROCESSES[family], **overrides)
            component._concat_test_sources = sources
            result[f"{kind}{index}"] = component
    return result


def _model(family="diffusion", **options):
    components = _components(family, **options)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True):
        model = dn.concat_models(components, celsius=6.3, threads=1, write_back=False)
        model.train()
        model.initialize()
    model._concat_test_options = {"family": family, **options}
    return model


def _record(model):
    values = {name: getattr(model, name) for name in ("v", "i_membrane")}
    values.update(
        {f"hh.{name}": getattr(model.mech.hh, name) for name in ("m", "h", "n")}
    )
    for name in ("pool", "reservoir"):
        if name in model.mech.materials:
            values[f"{name}_material.amount"] = model.mech.materials[name].amount
    return values


def _trace(model, *, dt, amplitude, steps=STEPS):
    model.train()
    model.initialize()
    model.integrator._initialize(
        model, dt.detach(), force=True, compile_scope="population"
    )
    history = [_record(model)]
    for index in range(steps):
        drive = model.v.new_full(model.shape, (index + 1) * 1e-10) * amplitude
        model.integrator.step(model, dt, None, drive)
        model.t = model.t + dt
        history.append(_record(model))
    return {
        name: torch.stack([frame[name] for frame in history]) for name in history[0]
    }


def _targets(model, *, component=None):
    result = {}
    components = (
        model.populations if hasattr(model, "populations") else {component: model}
    )
    for name, child in components.items():
        for label, value in child._concat_test_sources.items():
            result[f"{name}.{label}"] = value
    for name in CHANNELS:
        result[f"hh.{name}"] = getattr(model.mech.hh, name + "_param")
    for process in model.mech.material_processes.values():
        owner = getattr(type(process), "_material_process_component", component)
        if isinstance(owner, tuple):
            owner = owner[0]
        for name, value in process.named_parameters():
            result[f"{owner}.process.{name}"] = value
    return result


def _loss(recorded):
    loss = 0.001 * recorded["v"].square().sum()
    loss = loss + 0.1 * recorded["hh.m"].square().sum()
    loss = loss + 1e-4 * (recorded["i_membrane"] * 1e6).square().sum()
    for name in ("pool_material.amount", "reservoir_material.amount"):
        if name in recorded:
            loss = loss + 0.2 * recorded[name].square().sum()
    return loss


def _gradients(loss, targets):
    values = torch.autograd.grad(loss, tuple(targets.values()), allow_unused=True)
    return {
        name: torch.zeros_like(targets[name]) if value is None else value
        for name, value in zip(targets, values, strict=True)
    }


def _assert_close(actual, expected, *, dtype=torch.float64):
    tolerance = (
        {"rtol": 3e-4, "atol": 3e-5}
        if dtype == torch.float32
        else {"rtol": 2e-9, "atol": 2e-10}
    )
    assert actual.keys() == expected.keys()
    for name in actual:
        assert torch.isfinite(actual[name]).all(), name
        torch.testing.assert_close(
            actual[name],
            expected[name],
            **tolerance,
            msg=lambda message, name=name: f"{name}: {message}",
        )


def _independent_traces(components, *, dt, amplitude, steps=STEPS):
    return {
        name: _trace(child, dt=dt, amplitude=amplitude, steps=steps)
        for name, child in components.items()
    }


def _assert_independent_traces(model, actual, independent):
    offset = 0
    for name, child in model.populations.items():
        count = child.v.numel()
        for field, expected in independent[name].items():
            selected = actual[field][..., offset : offset + count].reshape_as(expected)
            torch.testing.assert_close(
                selected,
                expected,
                rtol=3e-4 if expected.dtype == torch.float32 else 2e-9,
                atol=3e-5 if expected.dtype == torch.float32 else 2e-10,
                msg=f"{name}: {field}",
            )
        if "pool_material.amount" not in independent[name]:
            assert (
                torch.count_nonzero(
                    actual["pool_material.amount"][..., offset : offset + count]
                )
                == 0
            )
        offset += count


def _exercise(family="diffusion", **options):
    model = _model(family, **options)
    reference = _components(family, **options)
    dt = model.v.new_tensor(DT, requires_grad=True)
    amplitude = model.v.new_ones((), requires_grad=True)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True):
        actual = _trace(model, dt=dt, amplitude=amplitude)
        expected = _independent_traces(reference, dt=dt, amplitude=amplitude)
    _assert_independent_traces(model, actual, expected)
    actual_targets = {**_targets(model), "dt": dt, "stimulus": amplitude}
    actual_gradients = _gradients(_loss(actual), actual_targets)
    reference_targets = {}
    for name, child in reference.items():
        for label, value in _targets(child, component=name).items():
            reference_targets.setdefault(label, []).append(value)
    independent_targets = {
        f"{label}:{index}": value
        for label, values in reference_targets.items()
        for index, value in enumerate(values)
    }
    independent_targets.update(dt=dt, stimulus=amplitude)
    reference_gradients = _gradients(
        sum(_loss(value) for value in expected.values()), independent_targets
    )
    gradients = {
        label: sum(
            reference_gradients[f"{label}:{index}"] for index in range(len(values))
        )
        for label, values in reference_targets.items()
    }
    gradients.update(
        dt=reference_gradients["dt"], stimulus=reference_gradients["stimulus"]
    )
    _assert_close(actual_gradients, gradients, dtype=model.dtype())
    for label in ("stimulus", "dt", *(f"hh.{name}" for name in CHANNELS)):
        assert torch.count_nonzero(actual_gradients[label]), label
    for label, value in actual_gradients.items():
        if label.endswith(
            (
                "initial_pool",
                "initial_reservoir",
                "chemical_volume",
                "chemical_edge_factor",
            )
        ):
            assert torch.count_nonzero(value), label
    return actual, actual_gradients


@pytest.mark.parametrize("family", tuple(PROCESSES))
def test_preinserted_material_processes_match_independent_traces_and_gradients(
    family, cpu_threads
):
    _exercise(family)


@pytest.mark.parametrize("missing", ("material", "process"))
def test_diffusion_support_does_not_leak_into_absent_component(missing, cpu_threads):
    _exercise("diffusion", **{f"missing_{missing}": True})


def test_material_metadata_and_named_geometry_survive_concatenation(cpu_threads):
    model = _model("named_diffusion")
    spec = model.mech.materials["pool"].field_spec("amount")
    assert spec.domain == "intracellular"
    assert spec.units == "mM"
    assert spec.conserved
    assert torch.as_tensor(spec.min_value).eq(0.0).all()
    assert all(
        child.material_geometry("chemical") for child in model.populations.values()
    )


@pytest.mark.parametrize("execution", ("run", "longrun"))
def test_public_imperative_runners_retain_packed_chemistry(execution, cpu_threads):
    model = _model("named_diffusion")
    reference = _components("named_diffusion")
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True):
        history = [_record(model)]
        for _ in range(STEPS):
            options = {"chunklength": 2} if execution == "longrun" else {}
            getattr(model, execution)(tstop=DT, dt=DT, **options)
            history.append(_record(model))
        expected = _independent_traces(
            reference, dt=model.v.new_tensor(DT), amplitude=model.v.new_zeros(())
        )
    actual = {
        name: torch.stack([frame[name] for frame in history]) for name in history[0]
    }
    _assert_independent_traces(model, actual, expected)


def _disjoint_field_components(*, device="cpu", dtype=torch.float64):
    children = _components(
        "diffusion", missing_material=True, device=device, dtype=dtype
    )
    first, second = children.values()
    first.delete(_ConcatDiffusion)
    other = torch.nn.Parameter(second.v.new_full(second.shape, 4.2))
    second.material(
        "pool",
        fields={"other": other},
        min_values={"other": 0.0},
        domain="intracellular",
    )
    second._concat_test_sources["initial_other"] = other
    return children


@pytest.mark.parametrize("process", (_ConcatDiffusion, _ConcatClamp))
def test_postpack_global_process_respects_per_field_availability(process, cpu_threads):
    children = _disjoint_field_components()
    reference = _disjoint_field_components()
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True):
        model = dn.concat_models(children, celsius=6.3, threads=1, write_back=False)
        model.insert(process)
        model.train()
        model.initialize()
        first, second = reference.values()
        first.insert(process)
        for child in reference.values():
            child.train()
            child.initialize()
        count = first.v.numel()
        actual_history, expected_history = [], []
        for _ in range(STEPS):
            model.step(dt=DT)
            for child in reference.values():
                child.step(dt=DT)
            pool = model.mech.materials["pool"]
            expected_amount = first.mech.materials["pool"].amount
            expected_other = second.mech.materials["pool"].other
            torch.testing.assert_close(
                pool.amount[..., :count].reshape_as(expected_amount), expected_amount
            )
            torch.testing.assert_close(
                pool.other[..., count:].reshape_as(expected_other), expected_other
            )
            assert torch.count_nonzero(pool.amount[..., count:]) == 0
            assert torch.count_nonzero(pool.other[..., :count]) == 0
            actual_history.append(
                pool.amount.square().sum() + pool.other.square().sum()
            )
            expected_history.append(
                expected_amount.square().sum() + expected_other.square().sum()
            )
    actual_targets = {
        "amount": children["path0"]._concat_test_sources["initial_pool"],
        "other": children["tree1"]._concat_test_sources["initial_other"],
    }
    expected_targets = {
        "amount": first._concat_test_sources["initial_pool"],
        "other": second._concat_test_sources["initial_other"],
    }
    _assert_close(
        _gradients(sum(actual_history), actual_targets),
        _gradients(sum(expected_history), expected_targets),
    )


def test_postpack_root_and_terminal_clamps_use_each_child_topology(cpu_threads):
    children = _components("clamp")
    reference = _components("clamp")
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True):
        for child in children.values():
            child.delete(_ConcatClamp)
        for child in reference.values():
            child.delete(_ConcatClamp)
            child.insert(_ConcatClamp)
        model = dn.concat_models(children, celsius=6.3, threads=1, write_back=False)
        model.insert(_ConcatClamp)
        actual = _trace(
            model, dt=model.v.new_tensor(DT), amplitude=model.v.new_zeros(())
        )
        independent = _independent_traces(
            reference, dt=model.v.new_tensor(DT), amplitude=model.v.new_zeros(())
        )
    _assert_independent_traces(model, actual, independent)


def _one_sided_exchange(*, device="cpu", dtype=torch.float64):
    children = _disjoint_field_components(device=device, dtype=dtype)
    first, second = children.values()
    reservoir = torch.nn.Parameter(second.v.new_full(second.shape, 2.7))
    second.material("reservoir", fields={"amount": reservoir}, domain="membrane")
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True):
        model = dn.concat_models(children, celsius=6.3, threads=1, write_back=False)
        model.insert(_ConcatExchange)
        model.train()
        model.initialize()
        initial = {
            name: model.mech.materials[name].amount.clone()
            for name in ("pool", "reservoir")
        }
        for _ in range(STEPS):
            model.step(dt=DT)
            for name, value in initial.items():
                torch.testing.assert_close(
                    model.mech.materials[name].amount, value, rtol=0, atol=0
                )
        targets = {
            "amount": first._concat_test_sources["initial_pool"],
            "reservoir": reservoir,
        }
        gradients = _gradients(
            sum(model.mech.materials[name].amount.square().sum() for name in initial),
            targets,
        )
    _assert_close(
        gradients, {name: 2 * value for name, value in targets.items()}, dtype=dtype
    )
    return initial, gradients


def test_postpack_exchange_requires_both_endpoint_fields(cpu_threads):
    _one_sided_exchange()
