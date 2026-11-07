"""Physical Slice selection for functional recording, including packed cells."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._regions import _RecorderRegion
from dendra.models.mod import hh

pytestmark = pytest.mark.cpu
DT = 0.01


def _logical_case():
    with dn.ctx(JIT=0, IMEM=False, DTYPE=torch.float64):
        source = dn.Population(N=2, C=4, dtype=torch.float64)
        source.batch(2).batch(3)
    functional = SimpleNamespace(
        _population=source, shape=source.shape, device=source.device()
    )
    return source, functional


@pytest.mark.parametrize(
    "key",
    [
        (..., slice(1, 4, 2)),
        (..., [3, 0, 3]),
        (..., []),
        (..., torch.tensor([True, False, True, False])),
        (..., torch.tensor([[True, False, True, False], [False, True, False, True]])),
        (None, ..., 2, None),
        (1, 0, 1, 3),
        ([2, 0, 2], slice(None), 1, [3, 0, 3]),
        (slice(None), [1, 0], ..., [0, 1], slice(None)),
        (True, ..., slice(0, 2)),
        (torch.tensor(False), ...),
        (slice(None), torch.tensor([True, False]), ..., 1),
    ],
)
def test_region_matches_exact_logical_slice_with_optional_payload(key):
    source, functional = _logical_case()
    region = source[key]
    bound = _RecorderRegion.bind(functional, region)
    voltage = torch.arange(source.v.numel(), dtype=torch.float64).reshape(source.shape)
    channels = torch.stack((voltage, voltage + 1000, voltage + 2000), dim=-1)
    torch.testing.assert_close(bound.select(voltage), voltage[region.index])
    torch.testing.assert_close(
        bound.select(channels, payload_ndim=1),
        channels[region.index + (slice(None),)]
        if not any(item is Ellipsis for item in region.index)
        else torch.stack(
            [channels[..., layer][region.index] for layer in range(3)], -1
        ),
    )
    assert bound.selected_shape == tuple(region.shape)
    assert bound.physical_shape == tuple(source.shape)
    assert bound.select(channels, payload_ndim=1).shape == (*region.shape, 3)
    assert not any(item is Ellipsis for item in bound.index)
    assert all(
        not torch.is_tensor(item) or item.dtype != torch.bool for item in bound.index
    )


def test_region_snapshots_indices_outside_inference_mode():
    source, functional = _logical_case()
    with torch.inference_mode():
        region = source[..., torch.tensor([3, 0, 3])]
        bound = _RecorderRegion.bind(functional, region)
    voltage = torch.arange(source.v.numel(), dtype=torch.float64).reshape(source.shape)
    expected = voltage[..., [3, 0, 3]].clone()
    with torch.inference_mode():
        region.index[-1].fill_(1)
    assert all(
        not torch.is_tensor(item) or not item.is_inference() for item in bound.index
    )
    torch.testing.assert_close(bound.select(voltage), expected)


def test_region_selection_preserves_vmap_and_repeated_index_gradients():
    source, functional = _logical_case()
    region = source[[2, 0, 2], :, 1, [3, 0, 3]]
    bound = _RecorderRegion.bind(functional, region)
    values = torch.randn(2, *source.shape, 3, dtype=torch.float64, requires_grad=True)
    actual = torch.func.vmap(lambda value: bound.select(value, payload_ndim=1))(values)
    expected = torch.stack(
        [
            torch.stack([value[..., layer][region.index] for layer in range(3)], -1)
            for value in values
        ]
    )
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual.square().sum(), values)[0]
    expected_gradient = torch.autograd.grad(expected.square().sum(), values)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_region_rejects_foreign_non_slice_and_wrong_payload_shapes():
    source, functional = _logical_case()
    foreign, _ = _logical_case()
    with pytest.raises(dn.func.FunctionalizationError, match="exact source"):
        _RecorderRegion.bind(functional, foreign[:])
    with pytest.raises(dn.func.FunctionalizationError, match="Population Slice"):
        _RecorderRegion.bind(functional, "cell_a")
    bound = _RecorderRegion.bind(functional, source[:])
    with pytest.raises(dn.func.FunctionalizationError, match="full-coordinate"):
        bound.select(source.v[..., :1])
    with pytest.raises(dn.func.FunctionalizationError, match="trailing payload"):
        bound.select(source.v, payload_ndim=1)
    source.batch(2)
    with pytest.raises(dn.func.FunctionalizationError, match="source shape"):
        _RecorderRegion.bind(functional, source[:])


def test_recorder_region_selects_vc_compartments_without_selecting_channels():
    source, functional = _logical_case()
    voltage = torch.arange(source.v.numel(), dtype=torch.float64).reshape(source.shape)
    state = {
        "integrator": {
            "v": voltage,
            "vc": torch.stack((voltage, voltage + 1000, voltage + 2000), dim=-1),
            "i_membrane": voltage * 1.0e-6,
        },
        "clock": {"t": torch.tensor(1.25, dtype=torch.float64)},
    }
    paths = [("integrator", name) for name in state["integrator"]] + [("clock", "t")]
    functional._state_layout = tuple(
        SimpleNamespace(public_path=path, checkpoint_key=None) for path in paths
    )
    functional._state_schema = {
        section: {
            name: SimpleNamespace(shape=value.shape) for name, value in values.items()
        }
        for section, values in state.items()
    }
    region = source[..., [3, 1, 3]]
    callback = dn.func.Recorder(("v", "vc", "i_membrane", "t"), region=region)._bind(
        functional
    )
    carry, observed = callback.initialize(state, None)
    assert carry == ()
    assert observed["vc"].shape == (*region.shape, 3)
    for name in ("v", "i_membrane"):
        torch.testing.assert_close(
            observed[name], state["integrator"][name][region.index]
        )
    torch.testing.assert_close(
        observed["vc"],
        torch.stack(
            [state["integrator"]["vc"][..., layer][region.index] for layer in range(3)],
            -1,
        ),
    )
    torch.testing.assert_close(observed["t"], state["clock"]["t"])


@pytest.fixture(params=(1, 4), ids=("threads1", "threads4"))
def cpu_threads(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    try:
        yield request.param
    finally:
        torch.set_num_threads(previous)


def _point_pack():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64):
        left = dn.Population(N=2, C=3, celsius=6.3, dtype=torch.float64)
        right = dn.SingleCompartment(N=1, C=2, celsius=6.3, dtype=torch.float64)
        left[:, 0].label("soma")
        left[1, :].label("cell_1")
        left.insert(hh)
        right.insert(hh)
        source = dn.concat_models({"cell_a": left, "cell_b": right}, celsius=6.3)
        source.batch(2)
        source.initialize()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    return source, functional, tensors


def test_named_multi_region_full_traces_and_gradients(cpu_threads):
    source, functional, tensors = _point_pack()
    region = source.cell_a.soma
    parameters = dict(tensors.parameters)
    name = "integrator.mech.mechanisms.hh.gnabar_param"
    parameters[name] = parameters[name].detach().clone().requires_grad_()
    state = {**tensors.state, "integrator": dict(tensors.state["integrator"])}
    state["integrator"]["v"] = (
        state["integrator"]["v"].detach().clone().requires_grad_()
    )
    callbacks = functional.make_callbacks(
        {
            "all": dn.func.Recorder(("v", "t")),
            "cell_a_soma": dn.func.Recorder(("v", "t"), region=region),
            "one_cell": dn.func.Recorder("v", region=source.cell_a.cell_1),
            "partial_batch": dn.func.Recorder(
                "v", region=source[[1, 0, 1], :, [5, 1, 5]]
            ),
        }
    )
    _, auxiliary = functional.rollout(
        parameters,
        functional.prepare(parameters, tensors.constants),
        state,
        steps=4,
        callbacks=callbacks,
    )
    traces = auxiliary["callbacks"]
    expected = torch.stack([value[region.index] for value in traces["all"]["v"]])
    torch.testing.assert_close(traces["cell_a_soma"]["v"], expected)
    torch.testing.assert_close(traces["cell_a_soma"]["t"], traces["all"]["t"])
    torch.testing.assert_close(traces["one_cell"]["v"], traces["all"]["v"][..., 3:6])
    torch.testing.assert_close(
        traces["partial_batch"]["v"],
        torch.stack([value[[1, 0, 1], :, [5, 1, 5]] for value in traces["all"]["v"]]),
    )
    targets = (state["integrator"]["v"], parameters[name])
    actual_gradients = torch.autograd.grad(
        traces["cell_a_soma"]["v"].square().sum(), targets, retain_graph=True
    )
    reference_gradients = torch.autograd.grad(expected.square().sum(), targets)
    for actual, reference in zip(actual_gradients, reference_gradients, strict=True):
        torch.testing.assert_close(actual, reference)
        assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0


@pytest.mark.parametrize(
    "extra", [{"node_indices": [0]}, {"max_only": True}, {"partition": [8]}]
)
def test_recorder_region_rejects_ambiguous_legacy_selection(extra):
    source, functional, _ = _point_pack()
    with pytest.raises(dn.func.FunctionalizationError, match="cannot be combined"):
        functional.make_callbacks(
            {"trace": dn.func.Recorder("v", region=source.cell_a, **extra)}
        )


def test_recorder_region_rejects_mechanism_local_state_and_submodule_slices():
    source, functional, _ = _point_pack()
    with pytest.raises(dn.func.FunctionalizationError, match="mechanism-local"):
        functional.make_callbacks(
            {"trace": dn.func.Recorder("hh.m", region=source.cell_a)}
        )
    with pytest.raises(dn.func.FunctionalizationError, match="mechanism submodule"):
        _RecorderRegion.bind(functional, source[:].mech.hh)


def test_named_multi_region_preserves_hidden_vmap_batches(cpu_threads):
    source, functional, tensors = _point_pack()
    region = source.cell_a.soma
    callbacks = functional.make_callbacks(
        {"trace": dn.func.Recorder("v", region=region)}
    )
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    voltages = torch.stack(
        (tensors.state["integrator"]["v"] - 2, tensors.state["integrator"]["v"] + 3)
    ).requires_grad_()

    def simulate(voltage):
        state = {**tensors.state, "integrator": dict(tensors.state["integrator"])}
        state["integrator"]["v"] = voltage
        _, auxiliary = functional.rollout(
            tensors.parameters, prepared, state, steps=2, callbacks=callbacks
        )
        return auxiliary["callbacks"]["trace"]["v"]

    actual = torch.func.vmap(simulate)(voltages)
    expected = torch.stack([simulate(voltage) for voltage in voltages])
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual.square().sum(), voltages)[0]
    expected_gradient = torch.autograd.grad(expected.square().sum(), voltages)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.slow
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_cpp_named_multi_regions_preserve_traces_and_checkpoint_gradients(
    cpu_threads, execution
):
    source, functional, tensors = _point_pack()
    specifications = {
        "named": dn.func.Recorder(("v", "t"), region=source.cell_a.soma),
        "mask": dn.func.Recorder(
            "v",
            region=source[
                ..., torch.tensor([True, False, True, False, False, True, False, True])
            ],
        ),
        "partial": dn.func.Recorder("v", region=source[[1, 0, 1], :, [5, 1, 5]]),
    }
    scan = execution == "scan"
    runner = dn.func.compile(
        source,
        dt=DT,
        steps=4,
        execution=execution,
        inference=not scan,
        checkpoint_every=2 if scan else None,
        trainable=("hh.gnabar",) if scan else (),
        callbacks=specifications,
        backend="inductor",
        options={"cpp_wrapper": True, "compile_threads": 1},
    )
    parameters = dict(runner.parameters)
    state = {
        **runner.initial_state,
        "integrator": dict(runner.initial_state["integrator"]),
    }
    state["integrator"]["v"] = (
        state["integrator"]["v"].detach().clone().requires_grad_(scan)
    )
    intra = torch.linspace(-1.0e-7, 2.0e-7, 4 * source.v.numel(), dtype=source.dtype())
    intra = intra.reshape(4, *source.shape).requires_grad_(scan)
    inputs = dn.func.RolloutInput(intra=intra)
    _, expected_auxiliary = functional.rollout(
        parameters,
        functional.prepare(parameters, tensors.constants),
        state,
        inputs,
        callbacks=functional.make_callbacks(specifications),
    )
    with torch_compiler_warning_context():
        _, auxiliary = runner(state, inputs=inputs)
    for callback_name, expected_states in expected_auxiliary["callbacks"].items():
        for state_name, expected in expected_states.items():
            torch.testing.assert_close(
                auxiliary["callbacks"][callback_name][state_name],
                expected,
                rtol=2.0e-10,
                atol=2.0e-11,
            )
    assert runner.execution_report().strategy == (
        "compiled_scan" if scan else "compiled_while_loop"
    )
    if scan:
        name = "integrator.mech.mechanisms.hh.gnabar_param"
        targets = (state["integrator"]["v"], parameters[name], intra)
        expected_loss = sum(
            expected_auxiliary["callbacks"][key]["v"].square().mean()
            for key in specifications
        )
        actual_loss = sum(
            auxiliary["callbacks"][key]["v"].square().mean() for key in specifications
        )
        reference = torch.autograd.grad(expected_loss, targets)
        gradients = torch.autograd.grad(actual_loss, targets)
        for actual, expected in zip(gradients, reference, strict=True):
            torch.testing.assert_close(actual, expected, rtol=2.0e-8, atol=2.0e-10)
            assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
