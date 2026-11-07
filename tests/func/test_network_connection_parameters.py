"""Network raw connection sources preserve native coordinates and shared VJPs."""

from __future__ import annotations

import pytest
import torch
from torch.nn.utils.parametrize import register_parametrization

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import graded_syn, pas
from dendra.models.parametric import PositiveParam
from tests.func.test_network import (
    DT,
    DTYPE,
    _clone,
    _close,
    _loss,
    _native_loss,
    _Release,
)
from tests.networks.test_functional_event_delivery import ThresholdRelay

pytestmark = pytest.mark.cpu
RAW_WEIGHT = "weight.weights.0.param.rho"
RAW_DELAY = "delay_ms.weights.0.param.rho"


class _AffineRawSource(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.factor = torch.nn.Parameter(torch.tensor(1.1, dtype=DTYPE))
        self.register_buffer("offset", torch.tensor(0.1, dtype=DTYPE))

    def forward(self, value):
        return value * self.factor + self.offset


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _network(*, parametrized=False, diff_delays=False):
    shared = PositiveParam(torch.tensor(0.002, dtype=DTYPE))
    if parametrized:
        register_parametrization(shared, "rho", _AffineRawSource())
    cell = dn.SingleCompartment(
        N=2,
        C=1,
        cm=shared,
        integrator=dn.bwd_euler_sc(imem=False),
        v_init=-65.0,
    )
    cell.insert(pas, g=0.001, e=-65.0)
    cell.insert(ThresholdRelay.rename("relay"))
    cell.insert(graded_syn.rename("analog"), e=-10.0)
    model = dn.Network(
        {"cell": cell},
        netcon_delay_backend="dense",
        netcon_train_backend="dense",
    )
    for field, threshold, delay in (
        ("level", 0.0, 2.4 * DT if diff_delays else DT),
        ("mirror", 1.5, 3.2 * DT if diff_delays else DT),
    ):
        model.connect_one_to_one(
            cell[:],
            cell[:],
            cell.mech.relay,
            pre_var=f"mech.relay.{field}",
            threshold=threshold,
            weight=shared,
            delay=delay,
            allow_autapses=True,
        )
    for field, delay, transform in (
        ("level", DT, _Release()),
        ("mirror", 3 * DT, torch.sigmoid),
    ):
        model.connect_continuous_one_to_one(
            cell[:],
            cell[:],
            cell.mech.analog,
            pre_var=f"mech.relay.{field}",
            input="g_pre",
            weight=shared,
            delay=delay,
            transform=transform,
            allow_autapses=True,
        )
    model.train()
    model.initialize(DT)
    model.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=diff_delays,
        diff_spiking=True,
        diff_scheduled_times=False,
        tau=0.23,
        taps=3,
        sigma=0.43,
        train_delay_backend="dense",
    )
    model.initialize(DT)
    return model, shared


def _source_paths(parametrized):
    if parametrized:
        return (
            "cm_param.parametrizations.rho.original",
            "cm_param.parametrizations.rho.0.factor",
        )
    return ("cm_param.rho",)


def _connection_paths(parametrized):
    if parametrized:
        return (
            "weight.weights.0.param.parametrizations.rho.original",
            "weight.weights.0.param.parametrizations.rho.0.factor",
        )
    return (RAW_WEIGHT,)


def _native_targets(shared, parametrized):
    if parametrized:
        return shared.parametrizations.rho.original, shared.parametrizations.rho[
            0
        ].factor
    return (shared.rho,)


def _packet_targets(parameters, parametrized):
    return tuple(
        parameters["populations"]["cell"][name] for name in _source_paths(parametrized)
    )


def _selection(parametrized):
    return tuple(f"populations.cell.{name}" for name in _source_paths(parametrized))


def _assert_shared(parameters, parametrized):
    for population_path, connection_path in zip(
        _source_paths(parametrized), _connection_paths(parametrized), strict=True
    ):
        canonical = parameters["populations"]["cell"][population_path]
        assert canonical.ndim == 0
        for collection in ("synapses", "continuous_synapses"):
            for values in parameters[collection].values():
                assert values[connection_path] is canonical
                assert "weights" not in values and "delays" not in values


@pytest.mark.parametrize("check_purity", (False, True))
@pytest.mark.parametrize("parametrized", (False, True))
def test_raw_sources_keep_population_and_projection_aliases_before_expansion(
    check_purity, parametrized
):
    model, source = _network(parametrized=parametrized)
    with dn.ctx(CHECK_PURITY=check_purity):
        functional, tensors = dn.func.make_functional_network(model)
    _assert_shared(tensors.parameters, parametrized)
    independent = tensors.independent_parameters(_selection(parametrized))
    _assert_shared(independent, parametrized)
    materialized = functional._materialize_connections(independent, tensors.constants)
    for collection in ("synapses", "continuous_synapses"):
        for name, connection in getattr(model, collection).items():
            value = materialized[collection][name]["weights"]
            assert value.shape == (2,)
            torch.testing.assert_close(value, connection.weight.w)
            torch.testing.assert_close(value, source._compute().expand_as(value))
    frozen = _clone((tensors.constants, tensors.state), detached=True)
    native_before = tuple(
        value.detach().clone() for value in _native_targets(source, parametrized)
    )
    original = _packet_targets(independent, parametrized)[0]
    with torch.no_grad():
        original.add_(0.2)
    updated = functional._materialize_connections(independent, tensors.constants)
    values = tuple(
        fields["weights"]
        for collection in ("synapses", "continuous_synapses")
        for fields in updated[collection].values()
    )
    assert all((value > 0).all() for value in values)
    for value in values:
        torch.testing.assert_close(value, values[0])
    assert torch.count_nonzero(
        values[0] - materialized["synapses"][next(iter(model.synapses))]["weights"]
    )
    for value, before in zip(
        _native_targets(source, parametrized), native_before, strict=True
    ):
        torch.testing.assert_close(value, before)
    _close((tensors.constants, tensors.state), frozen)


@pytest.mark.parametrize("parametrized", (False, True))
def test_shared_raw_weight_and_interpolated_delay_vjps_match_native_network(
    parametrized,
):
    model, _ = _network(parametrized=parametrized, diff_delays=True)
    native, native_source = _network(parametrized=parametrized, diff_delays=True)
    functional, tensors = dn.func.make_functional_network(model)
    event = next(iter(model.synapses))
    delay_name = f"synapses.{event}.{RAW_DELAY}"
    parameters = tensors.independent_parameters((*_selection(parametrized), delay_name))
    actual, _ = functional.prepare_and_rollout(
        parameters, tensors.constants, tensors.state, steps=7
    )
    for _ in range(7):
        native.step()
    _close(actual, functional.extract(native).state)
    targets = (
        *_packet_targets(parameters, parametrized),
        parameters["synapses"][event][RAW_DELAY],
    )
    native_delay = native.synapses[event].delay_ms.weights[0].param.rho
    observed = torch.autograd.grad(_loss(actual), targets)
    oracle = torch.autograd.grad(
        _native_loss(native),
        (*_native_targets(native_source, parametrized), native_delay),
    )
    _close(observed, oracle)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in observed)


@pytest.mark.parametrize("execution", ("scan", "while"))
@pytest.mark.parametrize("parametrized", (False, True))
def test_raw_connection_sources_continue_compiled_network_and_native_checkpoint(
    execution, parametrized
):
    model, _ = _network(parametrized=parametrized)
    native, native_source = _network(parametrized=parametrized)
    options = {"checkpoint_every": 2} if execution == "scan" else {}
    runner = dn.compile(
        model,
        dt=DT,
        steps=4,
        inference=execution == "while",
        execution=execution,
        trainable=_selection(parametrized),
        callbacks={"trace": dn.func.Recorder(("cell.v", "t"))},
        backend="aot_eager",
        **options,
    )
    _assert_shared(runner.parameters, parametrized)
    initial = _clone(runner.initial_state, detached=True)
    prefix, first = runner()
    repeated, _ = runner()
    _close(prefix, repeated)
    final, second = runner(prefix, callback_state=first["callbacks"].state)
    frames = [native.cell.v.detach().clone()]
    for _ in range(8):
        native.step()
        frames.append(native.cell.v.detach().clone())
    _close(final, runner.functional.extract(native).state)
    trace = torch.cat(
        (first["callbacks"]["trace"]["cell.v"], second["callbacks"]["trace"]["cell.v"])
    )
    torch.testing.assert_close(trace, torch.stack(frames))
    assert trace.shape[0] == 9
    if execution == "scan":
        observed = torch.autograd.grad(
            _loss(final), _packet_targets(runner.parameters, parametrized)
        )
        oracle = torch.autograd.grad(
            _native_loss(native), _native_targets(native_source, parametrized)
        )
        _close(observed, oracle)
        assert all(value.abs().sum() > 0 for value in observed)
    else:
        assert runner.execution_report().strategy == "compiled_while_loop"
    _close(runner.initial_state, initial)
    target, _ = _network(parametrized=parametrized)
    runner.functional.commit_state_(target, final)
    checkpoint = target.state_dict_for_checkpoint()
    resumed, _ = _network(parametrized=parametrized)
    resumed.restore_dict_from_checkpoint(checkpoint)
    continued, _ = runner.functional.prepare_and_step(
        runner.parameters, runner.constants, final
    )
    resumed.step()
    _close(continued, runner.functional.extract(resumed).state)


def test_inductor_raw_connection_sources_keep_native_shared_gradient():
    model, _ = _network(parametrized=True)
    native, source = _network(parametrized=True)
    runner = dn.compile(
        model,
        dt=DT,
        steps=2,
        inference=False,
        trainable=_selection(True),
        backend="inductor",
    )
    final, _ = runner()
    for _ in range(2):
        native.step()
    _close(final, runner.functional.extract(native).state)
    observed = torch.autograd.grad(
        _loss(final), _packet_targets(runner.parameters, True)
    )
    oracle = torch.autograd.grad(_native_loss(native), _native_targets(source, True))
    _close(observed, oracle)
    assert all(value.abs().sum() > 0 for value in observed)


@pytest.mark.parametrize("delay", (0.0, 0.4 * DT))
def test_fixed_zero_and_subhalf_event_delays_keep_native_one_step_routing(delay):
    model, _ = _network()
    native, source = _network()
    for network in (model, native):
        for connection in network.synapses.values():
            connection.delay_ms.weights[0].param.set(delay)
        network.initialize(DT)
    functional, tensors = dn.func.make_functional_network(model)
    advance = torch.compile(
        functional.prepare_and_step, backend="aot_eager", fullgraph=True
    )
    eager, compiled = tensors.state, tensors.state
    for _ in range(4):
        eager, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, eager
        )
        compiled, _ = advance(tensors.parameters, tensors.constants, compiled)
        native.step()
        _close(compiled, eager)
        _close(compiled, functional.extract(native).state)
    targets = _packet_targets(tensors.parameters, False)
    observed = torch.autograd.grad(_loss(compiled), targets)
    eager_gradient = torch.autograd.grad(_loss(eager), targets)
    oracle = torch.autograd.grad(_native_loss(native), _native_targets(source, False))
    _close(observed, eager_gradient)
    _close(observed, oracle)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in observed)


class _OtherSource(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.002, dtype=DTYPE))

    def forward(self):
        raise AssertionError("An incompatible target source must not execute")


def test_compatible_target_raw_source_mismatch_rejects_commit_atomically():
    model, _ = _network()
    target, _ = _network()
    functional, tensors = dn.func.make_functional_network(model)
    state, _ = functional.prepare_and_step(*tensors)
    connection = next(iter(target.synapses.values()))
    connection.weight.weights[0].param = _OtherSource()
    frozen = _clone(target.state_dict_for_checkpoint(), detached=True)
    with pytest.raises(dn.func.FunctionalizationError, match="source|structure"):
        functional.commit_state_(target, state)
    _close(target.state_dict_for_checkpoint(), frozen)


@pytest.mark.parametrize("collection", ("synapses", "continuous_synapses"))
@pytest.mark.parametrize("transform", ("grad", "vmap", "vmap_grad", "nested_vmap"))
def test_transformed_fixed_delays_validate_every_lane(collection, transform):
    model, _ = _network()
    functional, tensors = dn.func.make_functional_network(model)
    name = next(iter(getattr(model, collection)))
    connection = getattr(model, collection)[name]
    source = connection.delay_ms.weights[0].param
    original = tensors.parameters[collection][name][RAW_DELAY]
    different_route = source._inverse_transform(original.new_tensor(4 * DT))

    def loss(delay):
        parameters = {
            **tensors.parameters,
            collection: {
                **tensors.parameters[collection],
                name: {**tensors.parameters[collection][name], RAW_DELAY: delay},
            },
        }
        value = functional._materialize_connections(parameters, tensors.constants)
        # Keep an independent derivative to check that assertion inspection
        # never unwraps or detaches the authored numerical computation.
        return value[collection][name]["weights"].sum() + delay.square().sum()

    if transform == "grad":
        transformed = torch.func.grad(loss)
        valid = original
        invalid = different_route
        expected = 2 * valid
    elif transform == "vmap_grad":
        transformed = torch.vmap(torch.func.grad(loss))
        valid = original.expand(3).clone()
        invalid = torch.stack((original, original, different_route))
        expected = 2 * valid
    elif transform == "nested_vmap":
        transformed = torch.vmap(torch.vmap(loss))
        valid = original.expand(2, 3).clone()
        invalid = valid.clone()
        invalid[-1, -1] = different_route
        expected = loss(original).expand(2, 3)
    else:
        transformed = torch.vmap(loss)
        valid = original.expand(3).clone()
        invalid = torch.stack((original, original, different_route))
        expected = loss(original).expand(3)

    torch.testing.assert_close(transformed(valid), expected)
    with pytest.raises(RuntimeError, match="frozen integer routing"):
        transformed(invalid)
