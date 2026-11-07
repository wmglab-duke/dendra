"""Shared initialization writes retain each Population's native preparation."""

from __future__ import annotations

import pytest
import torch
from torch.nn.utils.parametrize import register_parametrization

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from tests.func.test_network import DT, DTYPE, _clone, _close
from tests.func.test_network_population_caches import _late_runtime_cache_network

pytestmark = pytest.mark.cpu

RAW_CM = "cm_scale_param.rho"
RAW_G = "integrator.mech.mechanisms.pas.g_param"


class _Affine(torch.nn.Module):
    def forward(self, value, factor, offset):
        return (factor * value + offset,)


class _SqrtCapacitance(torch.nn.Module):
    def forward(self, value, raw):
        return value * torch.sqrt(raw)


class _SharedScale(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, value):
        return value * self.scale


@pytest.fixture(autouse=True)
def _environment():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_num_threads(threads)


def _network(*, phase="pre", cached=None):
    shared = torch.nn.Parameter(torch.tensor(0.001, dtype=DTYPE))
    cells = {}
    for name in ("left", "right"):
        cell = dn.SingleCompartment(
            N=1, C=1, integrator=dn.bwd_euler_sc(imem=False), v_init=-50.0
        )
        cell.insert(pas, g=shared, e=-65.0)
        cell.build()
        cells[name] = cell
    cells["right"].cm_scale_param.rho = cells["left"].cm_scale_param.rho
    model = dn.Network(cells)
    model.train()
    # Start from coherent snapshot values. The late action runs on fresh starts.
    with torch.no_grad():
        model.initialize(DT)
    register = getattr(cells["right"], f"register_{phase}_initialize_transform")
    register(
        "scale",
        _Affine(),
        reads=(f"parameters.{RAW_CM}",),
        writes=(f"parameters.{RAW_CM}",),
        inputs={
            "factor": torch.tensor(1.4, dtype=DTYPE),
            "offset": torch.tensor(0.2, dtype=DTYPE),
        },
    )
    if cached is not None:
        cells[cached].cache("_steady_state")
    return model


def _voltage(state):
    return torch.cat(
        tuple(
            value["integrator"]["v"].flatten()
            for value in state["populations"].values()
        )
    )


def _native_voltage(network):
    return torch.cat(tuple(cell.v.flatten() for cell in network.populations.values()))


def _loss(voltage):
    return voltage.square().sum() + 0.13 * voltage.prod()


@pytest.mark.parametrize("phase", ("pre", "post"))
@pytest.mark.parametrize("compiled", (False, True))
def test_late_pre_and_post_writes_match_native_values_and_shared_parameter_vjp(
    phase, compiled
):
    source, native = _network(phase=phase), _network(phase=phase)
    functional, tensors = dn.func.make_functional_network(source)
    frozen = _clone((tensors, functional.initialization_inputs), detached=True)
    initialize = functional.initialize
    advance = functional.prepare_and_step
    if compiled:
        initialize = torch.compile(initialize, backend="aot_eager", fullgraph=True)
        advance = torch.compile(advance, backend="aot_eager", fullgraph=True)
    initial = initialize(*tensors, functional.initialization_inputs)
    assert (
        initial.parameters["populations"]["left"][RAW_CM]
        is initial.parameters["populations"]["right"][RAW_CM]
    )
    assert (
        initial.parameters["populations"]["left"][RAW_G]
        is initial.parameters["populations"]["right"][RAW_G]
    )
    native.initialize(DT)
    assert not torch.allclose(native.left.cm_scale, native.right.cm_scale)
    state = initial.state
    for _ in range(4):
        before = _clone(state, detached=True)
        incoming = state
        state, _ = advance(initial.parameters, initial.constants, incoming)
        native.step()
        _close(incoming, before)
        _close(state, functional.extract(native).state)
    actual = torch.autograd.grad(
        _loss(_voltage(state)), tensors.parameters["populations"]["left"][RAW_G]
    )[0]
    expected = torch.autograd.grad(
        _loss(_native_voltage(native)), native.left.get_parameter(RAW_G)
    )[0]
    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)
    assert actual.abs().sum() > 0
    _close((tensors, functional.initialization_inputs), frozen)


@pytest.mark.parametrize("phase", ("pre", "post"))
def test_ordered_materializations_preserve_transform_vjp_jvp_and_mapped_inputs(phase):
    functional, tensors = dn.func.make_functional_network(_network(phase=phase))
    defaults = functional.initialization_inputs

    def voltage(factor, offset):
        right = defaults.populations["right"]._replace(
            transforms={
                f"{phase}.scale.factor": factor,
                f"{phase}.scale.offset": offset,
            }
        )
        initial = functional.initialize(
            *tensors,
            defaults._replace(populations={**defaults.populations, "right": right}),
        )
        final, _ = functional.prepare_and_rollout(*initial, steps=3)
        return _voltage(final)

    factor = torch.tensor(1.4, dtype=DTYPE)
    offset = torch.tensor(0.2, dtype=DTYPE)
    reverse = torch.func.jacrev(voltage, argnums=(0, 1))(factor, offset)
    forward = torch.func.jacfwd(voltage, argnums=(0, 1))(factor, offset)
    _close(reverse, forward)
    # The later write changes RIGHT's preparation, while LEFT's earlier
    # materialization has no path to the later transform's explicit inputs.
    for value in reverse:
        torch.testing.assert_close(value[0], value.new_zeros(()))
        assert value[1].abs() > 0
    _, tangent = torch.func.jvp(
        voltage, (factor, offset), (factor.new_tensor(0.3), offset.new_tensor(-0.2))
    )
    torch.testing.assert_close(tangent, 0.3 * reverse[0] - 0.2 * reverse[1])
    factors = factor.new_tensor([1.2, 1.8])
    offsets = offset.new_tensor([0.1, 0.3])
    mapped = torch.vmap(voltage)(factors, offsets)
    torch.testing.assert_close(
        mapped,
        torch.stack([voltage(a, b) for a, b in zip(factors, offsets, strict=True)]),
    )


@pytest.mark.parametrize("cached", ("left", "right"))
def test_cache_and_fresh_initialization_keep_native_owner_preparation_and_vjp(cached):
    source, native = _network(cached=cached), _network(cached=cached)
    functional, tensors = dn.func.make_functional_network(source)
    frozen = _clone((tensors, functional.initialization_inputs), detached=True)
    initial = functional.initialize(*tensors, functional.initialization_inputs)
    native.initialize(DT)
    state = initial.state
    for _ in range(4):
        state, _ = functional.prepare_and_step(
            initial.parameters, initial.constants, state
        )
        native.step()
        _close(state, functional.extract(native).state)
    actual = torch.autograd.grad(
        _loss(_voltage(state)), tensors.parameters["populations"]["left"][RAW_G]
    )[0]
    expected = torch.autograd.grad(
        _loss(_native_voltage(native)), native.left.get_parameter(RAW_G)
    )[0]
    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)
    _close((tensors, functional.initialization_inputs), frozen)


@pytest.mark.parametrize("initialization", ("fresh", "snapshot"))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_ordered_runner_restart_and_checkpoint_continuation_match_native(
    initialization, execution
):
    source, native = _network(), _network()
    if initialization == "snapshot":
        # Capture an initialized Network whose raw shared parameter is newer
        # than LEFT's effective capacitance and integrator preparation.
        source.initialize(DT)
        native.initialize(DT)
    options = dict(
        dt=DT,
        steps=3,
        initialization=initialization,
        execution=execution,
        inference=execution == "while",
        trainable=(f"populations.left.{RAW_G}",),
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
        backend="aot_eager",
    )
    if execution == "scan":
        options["checkpoint_every"] = 2
    runner = dn.compile(source, **options)
    frozen = _clone((runner.initial_state, runner.initialization_inputs), detached=True)
    first, prefix = runner()
    repeated, again = runner()
    _close(first, repeated)
    _close(prefix["callbacks"].outputs, again["callbacks"].outputs)
    final, suffix = runner(
        first,
        parameters=prefix.get("parameters", runner.parameters),
        constants=prefix.get("constants", runner.constants),
        callback_state=prefix["callbacks"].state,
    )
    if initialization == "fresh":
        native.initialize(DT)
    frames = {
        name: [cell.v.detach().clone()] for name, cell in native.populations.items()
    }
    for _ in range(6):
        native.step()
        for name, cell in native.populations.items():
            frames[name].append(cell.v.detach().clone())
    _close(final, runner.functional.extract(native).state)
    for name in ("left", "right"):
        trace = torch.cat(
            (
                prefix["callbacks"]["trace"][f"{name}.v"],
                suffix["callbacks"]["trace"][f"{name}.v"],
            )
        )
        torch.testing.assert_close(trace, torch.stack(frames[name]))
    if execution == "scan":
        gradient = torch.autograd.grad(
            _loss(_voltage(final)), runner.get_parameter(f"populations.left.{RAW_G}")
        )[0]
        expected = torch.autograd.grad(
            _loss(_native_voltage(native)), native.left.get_parameter(RAW_G)
        )[0]
        torch.testing.assert_close(gradient, expected, rtol=2e-8, atol=2e-9)
    _close((runner.initial_state, runner.initialization_inputs), frozen)

    target = _network()
    # State commits require a compatible initialized target; the explicit
    # parameter/constant packet remains part of functional continuation.
    target.initialize(DT)
    runner.functional.commit_state_(target, final)
    resumed = _network()
    resumed.initialize(DT)
    resumed.restore_dict_from_checkpoint(target.state_dict_for_checkpoint())
    continued, _ = runner.functional.prepare_and_step(
        suffix.get("parameters", runner.parameters),
        suffix.get("constants", runner.constants),
        final,
    )
    resumed.step()
    _close(continued, runner.functional.extract(resumed).state)


def test_later_cache_raw_overwrite_retains_native_fresh_parameter_gradient():
    source, native = _late_runtime_cache_network(), _late_runtime_cache_network()
    functional, tensors = dn.func.make_functional_network(source)
    initial = functional.initialize(*tensors, functional.initialization_inputs)
    native.initialize(DT)
    state, _ = functional.prepare_and_rollout(*initial, steps=4)
    for _ in range(4):
        native.step()
    actual = torch.autograd.grad(
        _loss(_voltage(state)), tensors.parameters["populations"]["fresh"][RAW_G]
    )[0]
    expected = torch.autograd.grad(
        _loss(_native_voltage(native)), native.populations["fresh"].get_parameter(RAW_G)
    )[0]
    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)
    assert actual.abs().sum() > 0


def test_fresh_ordered_continuation_requires_its_returned_materializations():
    runner = dn.compile(
        _network(),
        dt=DT,
        steps=1,
        initialization="fresh",
        inference=False,
        backend="aot_eager",
    )
    first, auxiliary = runner()
    with pytest.raises(ValueError, match="constants|continu"):
        runner(first, parameters=auxiliary["parameters"])
    continued, _ = runner(
        first,
        parameters=auxiliary["parameters"],
        constants=auxiliary["constants"],
    )
    assert continued["clock"]["step"].item() == 2


def _singular_later_source_network():
    shared = torch.nn.Parameter(torch.tensor(0.04, dtype=DTYPE))
    cells = {}
    for name in ("left", "right"):
        cell = dn.SingleCompartment(
            N=1, C=1, integrator=dn.bwd_euler_sc(imem=False), v_init=-50.0
        )
        cell.insert(pas, g=0.001, e=-65.0)
        cell.register_parameter("capacitance_source", shared)
        if name == "left":
            cell.register_parametrization_in_graph(
                "cm", _SqrtCapacitance(), args=("capacitance_source",)
            )
        cells[name] = cell
    model = dn.Network(cells)
    model.train()
    with torch.no_grad():
        model.initialize(DT)
    cells["right"].register_pre_initialize_transform(
        "zero_source",
        _Affine(),
        reads=("parameters.capacitance_source",),
        writes=("parameters.capacitance_source",),
        inputs={"factor": shared.new_tensor(0.0), "offset": shared.new_tensor(0.0)},
    )
    return model


def test_inactive_singular_rematerialization_preserves_native_finite_gradient():
    source, native = _singular_later_source_network(), _singular_later_source_network()
    functional, tensors = dn.func.make_functional_network(source)
    initial = functional.initialize(*tensors, functional.initialization_inputs)
    native.initialize(DT)
    state, _ = functional.prepare_and_rollout(*initial, steps=3)
    for _ in range(3):
        native.step()
    left = state["populations"]["left"]["integrator"]["v"]
    torch.testing.assert_close(left, native.left.v)
    actual = torch.autograd.grad(
        left.square().sum(),
        tensors.parameters["populations"]["left"]["capacitance_source"],
    )[0]
    expected = torch.autograd.grad(
        native.left.v.square().sum(), native.left.capacitance_source
    )[0]
    assert torch.isfinite(actual).all() and actual.abs().sum() > 0
    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)


def test_stale_native_snapshot_retains_baseline_without_freezing_parameter_updates():
    source, native = _network(), _network()
    source.initialize(DT)
    native.initialize(DT)
    functional, tensors = dn.func.make_functional_network(source)
    frozen = _clone(tensors, detached=True)
    original = tensors.parameters["populations"]["left"][RAW_CM]

    def voltage(raw):
        parameters = {
            **tensors.parameters,
            "populations": {
                name: {**values, RAW_CM: raw}
                for name, values in tensors.parameters["populations"].items()
            },
        }
        state, _ = functional.prepare_and_step(
            parameters, tensors.constants, tensors.state
        )
        return _voltage(state)

    baseline = voltage(original)
    native.step()
    torch.testing.assert_close(baseline, _native_voltage(native))
    shifted = original + 0.15
    assert torch.all((voltage(shifted) - baseline).abs() > 1e-6)
    reverse = torch.func.jacrev(voltage)(shifted)
    forward = torch.func.jacfwd(voltage)(shifted)
    torch.testing.assert_close(reverse, forward)
    epsilon = shifted.new_tensor(1e-5)
    finite_difference = (voltage(shifted + epsilon) - voltage(shifted - epsilon)) / (
        2 * epsilon
    )
    torch.testing.assert_close(reverse, finite_difference, rtol=1e-7, atol=1e-8)
    assert torch.all(reverse.abs() > 0)
    _close(tensors, frozen)


def test_inductor_ordered_initialization_and_rollout_keep_native_parameter_vjp():
    source, native = _network(), _network()
    runner = dn.compile(
        source,
        dt=DT,
        steps=3,
        initialization="fresh",
        inference=False,
        trainable=(f"populations.left.{RAW_G}",),
        backend="inductor",
    )
    final, _ = runner()
    native.initialize(DT)
    for _ in range(3):
        native.step()
    _close(final, runner.functional.extract(native).state)
    actual = torch.autograd.grad(
        _loss(_voltage(final)), runner.get_parameter(f"populations.left.{RAW_G}")
    )[0]
    expected = torch.autograd.grad(
        _loss(_native_voltage(native)), native.left.get_parameter(RAW_G)
    )[0]
    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)


def _packed_network():
    components = {}
    for name, cm in (("one", 0.8), ("two", 1.2)):
        cell = dn.SingleCompartment(
            N=1,
            C=1,
            cm=cm,
            v_init=-50.0,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        cell.insert(pas, g=0.001, e=-65.0)
        components[name] = cell
    packed = dn.concat_models(components)
    late = dn.SingleCompartment(
        N=1, C=1, v_init=-50.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    late.insert(pas, g=0.001, e=-65.0)
    late.build()
    late.cm_scale_param.rho = packed.populations["one"].cm_scale_param.rho
    network = dn.Network({"packed": packed, "late": late})
    network.train()
    with torch.no_grad():
        network.initialize(DT)
    late.register_pre_initialize_transform(
        "scale",
        _Affine(),
        reads=(f"parameters.{RAW_CM}",),
        writes=(f"parameters.{RAW_CM}",),
        inputs={
            "factor": torch.tensor(1.4, dtype=DTYPE),
            "offset": torch.tensor(0.2, dtype=DTYPE),
        },
    )
    return network


@pytest.mark.parametrize("snapshot", (False, True))
def test_packed_component_materialization_survives_later_shared_raw_write(snapshot):
    source, native = _packed_network(), _packed_network()
    if snapshot:
        source.initialize(DT)
        native.initialize(DT)
    functional, tensors = dn.func.make_functional_network(source)
    frozen = _clone(tensors, detached=True)
    initial = (
        tensors
        if snapshot
        else functional.initialize(*tensors, functional.initialization_inputs)
    )
    if not snapshot:
        native.initialize(DT)
    packed_key = "populations.one.cm_scale_param.rho"
    assert (
        initial.parameters["populations"]["packed"][packed_key]
        is initial.parameters["populations"]["late"][RAW_CM]
    )
    state = initial.state
    for _ in range(3):
        state, _ = functional.prepare_and_step(
            initial.parameters, initial.constants, state
        )
        native.step()
        _close(state, functional.extract(native).state)
    _close(tensors, frozen)


def _torch_physical_network():
    model = _network()
    register_parametrization(
        model.left, "cm", _SharedScale(model.right.cm_scale_param.rho)
    )
    model.initialize(DT)
    return model


def test_live_torch_getter_snapshot_keeps_native_earlier_integrator_workspace():
    source, native = _torch_physical_network(), _torch_physical_network()
    functional, tensors = dn.func.make_functional_network(source)
    original = tensors.parameters["populations"]["left"][RAW_CM]
    # Intra-Population aliases use one canonical explicit parameter leaf.
    assert source.left.parametrizations.cm[0].scale is source.left.cm_scale_param.rho
    assert original is tensors.parameters["populations"]["right"][RAW_CM]
    frozen = _clone(tensors, detached=True)

    def voltage(raw):
        population_parameters = {
            name: {**values, RAW_CM: raw}
            for name, values in tensors.parameters["populations"].items()
        }
        parameters = {**tensors.parameters, "populations": population_parameters}
        final, _ = functional.prepare_and_rollout(
            parameters, tensors.constants, tensors.state, steps=3
        )
        return _voltage(final)

    baseline = voltage(original)
    for _ in range(3):
        native.step()
    torch.testing.assert_close(baseline, _native_voltage(native), rtol=2e-8, atol=2e-9)
    shifted = original + 0.15
    assert torch.all((voltage(shifted) - baseline).abs() > 1e-6)
    reverse = torch.func.jacrev(voltage)(shifted)
    forward = torch.func.jacfwd(voltage)(shifted)
    torch.testing.assert_close(reverse, forward)
    epsilon = shifted.new_tensor(1e-5)
    finite_difference = (voltage(shifted + epsilon) - voltage(shifted - epsilon)) / (
        2 * epsilon
    )
    torch.testing.assert_close(reverse, finite_difference, rtol=1e-7, atol=1e-8)
    _close(tensors, frozen)
