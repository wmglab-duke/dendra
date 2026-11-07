"""Chemical forests coexist with independent finite-extracellular circuits."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators.tree_bt import DENDRA_SOLVERS_AVAILABLE
from tests.func.test_clearance_process import _clone, _key
from tests.func.test_extcell_multi_population import CHANNELS, DT, STEPS, _components
from tests.func.test_material_concatenation import _explicit_reference
from tests.test_material_concatenation import _ConcatNamedDiffusion, _gradients

FIELDS = ("v", "vc", "hh.m", "hh.h", "hh.n", "pool_material.amount")


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


def _chemical_components(device="cpu", dtype=torch.float64):
    components = _components(device=device, dtype=dtype)
    for index, component in enumerate(components.values()):
        initial = torch.nn.Parameter(
            torch.linspace(
                0.7 + index,
                1.5 + index,
                component.v.numel(),
                device=device,
                dtype=dtype,
            ).reshape(component.shape)
        )
        volume = torch.nn.Parameter(torch.full_like(initial, 0.8 + 0.2 * index))
        edge = torch.nn.Parameter(
            torch.linspace(
                0.6,
                1.0,
                component.shape[-2] * (component.shape[-1] - 1),
                device=device,
                dtype=dtype,
            ).reshape(component.shape[-2], -1)
        )
        component.material("pool", fields={"amount": initial}, domain="intracellular")
        component.register_material_geometry(
            "chemical", domain="intracellular", volume=volume, edge_factor=edge
        )
        component.insert(_ConcatNamedDiffusion)
        component._chemical_sources = {
            "initial": initial,
            "volume": volume,
            "edge": edge,
        }
    return components


def _record(model):
    return {
        "v": model.v,
        "vc": model.vc,
        **{f"hh.{gate}": getattr(model.mech.hh, gate) for gate in ("m", "h", "n")},
        "pool_material.amount": model.mech.materials["pool"].amount,
    }


def _trace(model, amplitude):
    model.train().initialize()
    model.integrator._initialize(model, DT, force=True, compile_scope="population")
    frames = [_record(model)]
    for index in range(STEPS):
        drive = model.v.new_full(model.shape, (index + 1) * 1e-10) * amplitude
        model.integrator.step(model, DT, None, drive)
        model.t = model.t + DT
        frames.append(_record(model))
    return {field: torch.stack([frame[field] for frame in frames]) for field in FIELDS}


def _loss(recorded):
    return (
        0.001 * recorded["v"].square().sum()
        + 0.001 * recorded["vc"].square().sum()
        + 0.1 * recorded["hh.m"].square().sum()
        + 0.2 * recorded["pool_material.amount"].square().sum()
    )


def _chemical_targets(model, *, component=None):
    components = (
        model.populations if hasattr(model, "populations") else {component: model}
    )
    targets = {
        f"{name}.{role}": value
        for name, child in components.items()
        for role, value in child._chemical_sources.items()
    }
    for process in model.mech.material_processes.values():
        name = getattr(type(process), "_material_process_component", component)
        targets[f"{name}.D"] = process.D_param
    return targets


def _exercise(*, device="cpu", dtype=torch.float64, execution="scan"):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DEVICE=device, DTYPE=dtype):
        model = dn.concat_models(
            _chemical_components(device, dtype),
            celsius=6.3,
            threads=2,
            write_back=False,
        ).train()
        model.initialize()
        runner = dn.compile(
            model,
            dt=DT,
            steps=STEPS,
            initialization="fresh",
            execution=execution,
            inference=execution == "while",
            backend="eager",
            callbacks={"trace": dn.func.Recorder(FIELDS)},
        )
        parameters = _clone(dict(runner.parameters))
        initialization = _clone(runner.initialization_inputs)
        initialization.v_init.requires_grad_()
        amplitude = model.v.new_tensor(0.8, requires_grad=True)
        inputs = dn.func.RolloutInput(
            None,
            torch.stack(
                [
                    model.v.new_full(model.shape, (index + 1) * 1e-10)
                    for index in range(STEPS)
                ]
            )
            * amplitude,
        )
        originals = dict(model.named_parameters())
        targets = {"stimulus": amplitude, "initial_voltage": initialization.v_init}
        for label, value in _chemical_targets(model).items():
            name = next(
                name for name, original in originals.items() if original is value
            )
            targets[label] = parameters[name].requires_grad_()
        for channel in CHANNELS:
            targets[f"hh.{channel}"] = parameters[
                _key(parameters, f"mechanisms.hh.{channel}_param")
            ].requires_grad_()
        result = runner(
            parameters=parameters, initialization=initialization, inputs=inputs
        )
        recorded = dict(result[1]["callbacks"]["trace"])
        if execution == "scan":
            actual_gradients = dict(
                zip(
                    targets,
                    torch.autograd.grad(
                        _loss(recorded), tuple(targets.values()), retain_graph=True
                    ),
                    strict=True,
                )
            )
            reference = _explicit_reference(
                runner, parameters, initialization, inputs, steps=STEPS
            )
            reference_gradients = _gradients(
                _loss(dict(reference[1]["callbacks"]["trace"])), targets
            )
            for label in targets:
                torch.testing.assert_close(
                    actual_gradients[label],
                    reference_gradients[label],
                    rtol=3e-8,
                    atol=3e-9,
                    msg=label,
                )
        else:
            actual_gradients = {}
            assert not recorded["v"].requires_grad

        independent = _chemical_components(device, dtype)
        independent_amplitude = model.v.new_tensor(0.8, requires_grad=True)
        independent_records = {
            name: _trace(child, independent_amplitude)
            for name, child in independent.items()
        }
        offset = 0
        for name, child in independent.items():
            count = child.v.numel()
            for field in FIELDS:
                actual = recorded[field]
                selected = (
                    actual[..., offset : offset + count, :]
                    if field == "vc"
                    else actual[..., offset : offset + count]
                ).reshape_as(independent_records[name][field])
                torch.testing.assert_close(
                    selected,
                    independent_records[name][field],
                    rtol=3e-8,
                    atol=3e-9,
                    msg=f"{name}: {field}",
                )
            offset += count
        if execution == "scan":
            independent_targets = {"stimulus": independent_amplitude}
            for name, child in independent.items():
                independent_targets.update(_chemical_targets(child, component=name))
                for channel in CHANNELS:
                    independent_targets[f"{name}.hh.{channel}"] = getattr(
                        child.mech.hh, channel + "_param"
                    )
            expected = _gradients(
                sum(_loss(value) for value in independent_records.values()),
                independent_targets,
            )
            for label, value in expected.items():
                if ".hh." in label:
                    continue
                torch.testing.assert_close(
                    actual_gradients[label], value, rtol=3e-8, atol=3e-9, msg=label
                )
            for channel in CHANNELS:
                torch.testing.assert_close(
                    actual_gradients[f"hh.{channel}"],
                    sum(expected[f"{name}.hh.{channel}"] for name in independent),
                    rtol=3e-8,
                    atol=3e-9,
                    msg=channel,
                )
        return recorded, actual_gradients


@pytest.mark.cpu
@pytest.mark.skipif(not DENDRA_SOLVERS_AVAILABLE, reason="requires native CPU solvers")
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_extcell_packed_diffusion_traces_and_gradients(cpu_threads, execution):
    _exercise(execution=execution)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_cuda_extcell_packed_diffusion_traces_and_gradients(execution):
    recorded, gradients = _exercise(device="cuda", execution=execution)
    assert all(value.device.type == "cuda" for value in recorded.values())
    assert all(value.device.type == "cuda" for value in gradients.values())
