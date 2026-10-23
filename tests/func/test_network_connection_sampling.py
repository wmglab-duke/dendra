"""Stochastic connection recipes share raw sources, explicit draws and RNG carry."""

from __future__ import annotations

import pytest
import torch
from torch.nn.utils.parametrize import register_parametrization
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.distributions import Normal, TruncatedNormal
from dendra.models.mod import graded_syn, pas
from dendra.models.parametric import PositiveParam
from tests.func.test_network import DT, DTYPE, _clone, _close, _loss, _native_loss
from tests.networks.test_functional_event_delivery import ThresholdRelay

pytestmark = pytest.mark.cpu
_COLLECTIONS = ("synapses", "continuous_synapses")


class _AffineMean(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.factor = torch.nn.Parameter(torch.tensor(1.1, dtype=DTYPE))
        self.register_buffer("offset", torch.tensor(0.0001, dtype=DTYPE))

    def forward(self, value):
        return self.factor * value + self.offset


@pytest.fixture(autouse=True)
def _environment():
    previous_threads = torch.get_num_threads()
    previous_rng = torch.get_rng_state()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_rng_state(previous_rng)
        torch.set_num_threads(previous_threads)


def _network(
    *,
    parametrized=False,
    interpolated=False,
    mixed=True,
    reverse_distributions=False,
    shared_population=False,
):
    torch.manual_seed(2478)
    weights = (
        TruncatedNormal(mean=0.003, std=0.0002, low=0.0025, high=0.0035)
        if reverse_distributions
        else Normal(mean=0.003, std=0.0002)
    )
    if parametrized:
        register_parametrization(weights, "mean", _AffineMean())
    delays = (
        Normal(mean=1.1 * DT, std=0.02 * DT)
        if reverse_distributions
        else TruncatedNormal(mean=1.1 * DT, std=0.02 * DT, low=1.05 * DT, high=1.2 * DT)
    )
    deterministic = PositiveParam(torch.tensor(0.002, dtype=DTYPE))
    cell = dn.SingleCompartment(
        N=2,
        C=1,
        integrator=dn.bwd_euler_sc(imem=False),
        v_init=-65.0,
        cm=PositiveParam(torch.tensor(0.7, dtype=DTYPE)) if shared_population else 1.0,
    )
    if shared_population:
        if parametrized:
            weights.parametrizations.mean.original = cell.cm_param.rho
        else:
            weights.mean = cell.cm_param.rho
    cell.insert(pas, g=0.001, e=-65.0)
    cell.insert(ThresholdRelay.rename("relay"))
    cell.insert(graded_syn.rename("analog"), e=-10.0)
    model = dn.Network(
        {"cell": cell},
        netcon_delay_backend="dense",
        netcon_train_backend="dense",
    )
    blocks = ((weights, delays), (deterministic, DT)) if mixed else ((weights, delays),)
    for weight, delay in blocks:
        model.connect_one_to_one(
            cell[:],
            cell[:],
            cell.mech.relay,
            pre_var="mech.relay.level",
            weight=weight,
            delay=delay,
            allow_autapses=True,
        )
        model.connect_continuous_one_to_one(
            cell[:],
            cell[:],
            cell.mech.analog,
            pre_var="mech.relay.level",
            input="g_pre",
            weight=weight,
            delay=delay,
            transform=torch.sigmoid,
            allow_autapses=True,
        )
    model.train()
    model.initialize(DT)
    model.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=interpolated,
        diff_spiking=True,
        diff_scheduled_times=False,
        tau=0.23,
        taps=3,
        sigma=0.43,
        train_delay_backend="dense",
    )
    model.initialize(DT)
    assert len(model.synapses) == len(model.continuous_synapses) == 1
    return model, weights, delays, deterministic


def _materialized(functional, packet):
    return {
        collection: {
            name: functional._connection_parameter_plans[collection, name].materialize(
                packet.parameters[collection][name], packet.constants[collection][name]
            )
            for name in getattr(functional, collection)
        }
        for collection in _COLLECTIONS
    }


def _check_native(functional, packet, native):
    values = _materialized(functional, packet)
    for collection in _COLLECTIONS:
        for name, connection in getattr(native, collection).items():
            torch.testing.assert_close(
                values[collection][name]["weights"], connection.weight.w
            )
            torch.testing.assert_close(
                values[collection][name]["delays"], connection.delay_ms.w
            )


def _parameters(packet):
    return next(iter(packet.parameters["synapses"].values()))


def _targets(packet, parametrized):
    values = _parameters(packet)
    suffixes = (
        ("parametrizations.mean.original", "parametrizations.mean.0.factor")
        if parametrized
        else ("mean",)
    )
    return (
        *(values[f"weight.weights.0.param.{name}"] for name in suffixes),
        values["weight.weights.0.param.std"]
        if "weight.weights.0.param.std" in values
        else values["weight.weights.0.param._log_std"],
        values["delay_ms.weights.0.param.mean"],
        values["delay_ms.weights.0.param.std"]
        if "delay_ms.weights.0.param.std" in values
        else values["delay_ms.weights.0.param._log_std"],
    )


def _native_targets(weights, delays, parametrized):
    mean = (
        (
            weights.parametrizations.mean.original,
            weights.parametrizations.mean[0].factor,
        )
        if parametrized
        else (weights.mean,)
    )
    return (
        *mean,
        weights.std if isinstance(weights, Normal) else weights._log_std,
        delays.mean,
        delays.std if isinstance(delays, Normal) else delays._log_std,
    )


def _materialized_loss(functional, packet):
    values = _materialized(functional, packet)
    return sum(
        (index + 1) * fields["weights"].square().sum()
        + (index + 0.5) * fields["delays"].square().sum()
        for index, collection in enumerate(_COLLECTIONS)
        for fields in values[collection].values()
    )


def _native_materialized_loss(model):
    return sum(
        (index + 1) * connection.weight.w.square().sum()
        + (index + 0.5) * connection.delay_ms.w.square().sum()
        for index, collection in enumerate(_COLLECTIONS)
        for connection in getattr(model, collection).values()
    )


def _draw(functional, packet, **flags):
    return functional.sample_connection_initialization_draw(
        packet.parameters, packet.constants, packet.state, **flags
    )


@pytest.mark.parametrize("parametrized", (False, True))
@pytest.mark.parametrize("check_purity", (False, True))
def test_snapshot_preserves_mixed_blocks_shared_raw_sources_and_native_gradients(
    parametrized, check_purity
):
    model, weights, delays, _ = _network(parametrized=parametrized)
    before_rng = torch.get_rng_state().clone()
    before_native = _clone(model.state_dict(), detached=True)
    with dn.ctx(CHECK_PURITY=check_purity):
        functional, packet = dn.func.make_functional_network(model)
    _check_native(functional, packet, model)
    values = [
        _parameters(packet),
        next(iter(packet.parameters["continuous_synapses"].values())),
    ]
    for key, value in values[0].items():
        if key.startswith(("weight.weights.", "delay_ms.weights.0.")):
            assert value is values[1][key]
    assert "weight.w" not in values[0]
    assert "weight.weights.1.param.rho" in values[0]
    assert all(value.ndim == 0 for value in _targets(packet, parametrized))
    observed = torch.autograd.grad(
        _materialized_loss(functional, packet), _targets(packet, parametrized)
    )
    expected = torch.autograd.grad(
        _native_materialized_loss(model), _native_targets(weights, delays, parametrized)
    )
    _close(observed, expected)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in observed)
    torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)
    _close(model.state_dict(), before_native)
    event = next(iter(model.synapses.values()))
    continuous = next(iter(model.continuous_synapses.values()))
    assert not torch.equal(event.weight.w[:2], continuous.weight.w[:2])
    assert not torch.equal(event.delay_ms.w[:2], continuous.delay_ms.w[:2])


@pytest.mark.parametrize("reinit_weights", (False, True))
@pytest.mark.parametrize("reinit_delays", (False, True))
def test_explicit_refresh_matches_native_draw_order_and_retention_without_global_rng(
    reinit_weights, reinit_delays
):
    model, *_ = _network()
    native, *_ = _network()
    functional, packet = dn.func.make_functional_network(model)
    flags = dict(reinit_weights=reinit_weights, reinit_delays=reinit_delays)
    frozen = _clone(packet, detached=True)
    global_rng = torch.get_rng_state().clone()
    draw = _draw(functional, packet, **flags)
    repeated = _draw(functional, packet, **flags)
    assert isinstance(draw, dn.func.ConnectionSamplingDraw)
    _close(draw, repeated)
    inputs = functional.initialization_inputs._replace(connection_draw=draw, **flags)
    initialized = functional.initialize(*packet, inputs)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
    _close(packet, frozen)
    native.initialize(DT, **flags)
    _check_native(functional, initialized, native)
    torch.testing.assert_close(
        initialized.state["connection_sampling"]["rng"]["cpu"],
        torch.get_rng_state(),
        rtol=0,
        atol=0,
    )
    for collection in _COLLECTIONS:
        for roles in draw.values[collection].values():
            assert bool(roles["weights"]) is reinit_weights
            assert bool(roles["delays"]) is reinit_delays
    if not (reinit_weights or reinit_delays):
        _close(draw.rng, packet.state["connection_sampling"]["rng"])
    actual, _ = functional.prepare_and_step(*initialized)
    native.step()
    _close(actual, functional.extract(native).state)


@pytest.mark.parametrize("parametrized", (False, True))
@pytest.mark.parametrize("reverse_distributions", (False, True))
def test_fresh_sampling_reconnects_native_gradients_to_raw_distribution_parameters(
    parametrized, reverse_distributions
):
    options = dict(
        parametrized=parametrized, reverse_distributions=reverse_distributions
    )
    model, *_ = _network(**options)
    native, native_weights, native_delays, _ = _network(**options)
    functional, packet = dn.func.make_functional_network(model)
    draw = _draw(functional, packet)
    initialized = functional.initialize(
        *packet, functional.initialization_inputs._replace(connection_draw=draw)
    )
    native.initialize(DT)
    _check_native(functional, initialized, native)
    observed = torch.autograd.grad(
        _materialized_loss(functional, initialized), _targets(initialized, parametrized)
    )
    expected = torch.autograd.grad(
        _native_materialized_loss(native),
        _native_targets(native_weights, native_delays, parametrized),
    )
    _close(observed, expected)


def test_pure_initialization_requires_draw_only_for_refreshed_stochastic_roles():
    model, *_ = _network()
    functional, packet = dn.func.make_functional_network(model)
    frozen = _clone(packet, detached=True)
    with pytest.raises(
        (ValueError, dn.func.FunctionalizationError), match="[Dd]raw|sampling"
    ):
        functional.initialize(*packet, functional.initialization_inputs)
    retained = functional.initialize(
        *packet,
        functional.initialization_inputs._replace(
            reinit_weights=False, reinit_delays=False
        ),
    )
    _check_native(functional, retained, model)
    _close(packet, frozen)


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_sampled_sources_continue_compiled_chunks_and_native_checkpoint(execution):
    model, *_ = _network(parametrized=True, interpolated=execution == "scan")
    native, native_weights, native_delays, _ = _network(
        parametrized=True, interpolated=execution == "scan"
    )
    event_name = next(iter(model.synapses))
    trainable = (
        f"synapses.{event_name}.weight.weights.0.param.parametrizations.mean.original",
        f"synapses.{event_name}.weight.weights.0.param.parametrizations.mean.0.factor",
    )
    options = {"checkpoint_every": 2} if execution == "scan" else {}
    runner = dn.compile(
        model,
        dt=DT,
        steps=4,
        inference=execution == "while",
        execution=execution,
        initialization="fresh",
        trainable=trainable,
        callbacks={"trace": dn.func.Recorder(("cell.v", "t"))},
        backend="aot_eager",
        **options,
    )
    initial = _clone(runner.initial_state, detached=True)
    global_rng = torch.get_rng_state().clone()
    prefix, first = runner()
    repeated, _ = runner()
    _close(prefix, repeated)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
    parameters, constants = first["parameters"], first["constants"]
    with pytest.raises(ValueError, match="constants=aux"):
        runner(prefix, parameters=parameters)
    final, second = runner(
        prefix,
        parameters=parameters,
        constants=constants,
        callback_state=first["callbacks"].state,
    )
    native.initialize(DT)
    frames = [native.cell.v.detach().clone()]
    for _ in range(8):
        native.step()
        frames.append(native.cell.v.detach().clone())
    _close(final, runner.functional.extract(native).state)
    trace = torch.cat(
        (first["callbacks"]["trace"]["cell.v"], second["callbacks"]["trace"]["cell.v"])
    )
    torch.testing.assert_close(trace, torch.stack(frames))
    if execution == "scan":
        targets = tuple(
            parameters["synapses"][event_name][
                key.removeprefix(f"synapses.{event_name}.")
            ]
            for key in trainable
        )
        observed = torch.autograd.grad(_loss(final), targets)
        expected = torch.autograd.grad(
            _native_loss(native),
            _native_targets(native_weights, native_delays, True)[:2],
        )
        _close(observed, expected)
        assert all(
            value.isfinite().all() and value.abs().sum() > 0 for value in observed
        )
    else:
        assert runner.execution_report().strategy == "compiled_while_loop"
    _close(runner.initial_state, initial)
    # Native runtime checkpoints retain explicit RNG carry after publication;
    # static samples and raw recipes travel with their normal model state dict.
    target, *_ = _network(parametrized=True, interpolated=execution == "scan")
    target.load_state_dict(native.state_dict())
    target.initialize(DT, reinit_weights=False, reinit_delays=False)
    runner.functional.commit_state_(target, final)
    checkpoint = _clone(target.state_dict_for_checkpoint(), detached=True)
    resumed, *_ = _network(parametrized=True, interpolated=execution == "scan")
    resumed.load_state_dict(target.state_dict())
    resumed.initialize(DT, reinit_weights=False, reinit_delays=False)
    resumed.restore_dict_from_checkpoint(checkpoint)
    restored = runner.functional.extract(resumed)
    _close(restored.state["connection_sampling"], final["connection_sampling"])
    continued, _ = runner.functional.prepare_and_step(parameters, constants, final)
    resumed.step()
    _close(continued, runner.functional.extract(resumed).state)


def test_changing_fixed_delay_route_fails_without_mutating_operands():
    model, *_ = _network()
    functional, packet = dn.func.make_functional_network(model)
    original = _parameters(packet)["delay_ms.weights.0.param.mean"]
    replacements = {
        id(original): original.detach().new_tensor(3.1 * DT).requires_grad_()
    }
    for collection in _COLLECTIONS:
        for values in packet.constants[collection].values():
            for key, value in values.items():
                if key.endswith(".param.low"):
                    replacements[id(value)] = value.new_tensor(3.05 * DT)
                elif key.endswith(".param.high"):
                    replacements[id(value)] = value.new_tensor(3.2 * DT)
    parameters, constants = pytree.tree_map_only(
        torch.Tensor,
        lambda value: replacements.get(id(value), value),
        (packet.parameters, packet.constants),
    )
    changed = packet._replace(parameters=parameters, constants=constants)
    frozen = _clone(changed, detached=True)
    draw = _draw(functional, changed)
    with pytest.raises(dn.func.FunctionalizationError, match="delay|routing|lower"):
        functional.initialize(
            *changed, functional.initialization_inputs._replace(connection_draw=draw)
        )
    _close(changed, frozen)


@pytest.mark.parametrize("operation", ("commit", "restore"))
def test_invalid_sampling_rng_cannot_partially_publish_native_checkpoint(operation):
    model, *_ = _network()
    functional, packet = dn.func.make_functional_network(model)
    functional.commit_state_(model, packet.state)
    before = _clone(model.state_dict_for_checkpoint(), detached=True)
    global_rng = torch.get_rng_state().clone()
    invalid = torch.zeros_like(packet.state["connection_sampling"]["rng"]["cpu"])
    with pytest.raises((ValueError, RuntimeError), match="[Ss]tate|RNG|generator"):
        if operation == "commit":
            state = {
                **packet.state,
                "connection_sampling": {"rng": {"cpu": invalid}},
            }
            functional.commit_state_(model, state)
        else:
            model.restore_dict_from_checkpoint(
                {**before, "connection_sampling_rng": {"cpu": invalid}}
            )
    _close(model.state_dict_for_checkpoint(), before)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)


class _AddToSource(torch.nn.Module):
    def forward(self, value, amount):
        return (value + amount,)


def _network_with_population_source_phase(mode):
    model, *_ = _network(shared_population=True)
    if mode == "transform":
        model.cell.register_pre_initialize_transform(
            "adjust",
            _AddToSource(),
            reads=("parameters.cm_param.rho",),
            writes=("parameters.cm_param.rho",),
            inputs={"amount": model.t.new_tensor(0.02)},
        )
    else:
        if mode == "network_cache":
            model._enable_synapse_state_cache_recording()
        with torch.no_grad():
            for _ in range(2):
                model.step()
            if mode == "network_cache":
                model.cache_state()
            else:
                model.cell.cache("_steady_state")
            model.cell.cm_param.rho.add_(0.04)
        if mode == "network_cache":
            model._disable_synapse_state_cache_recording(release=True)
        # Keep live Population materializations coherent with the edited raw
        # source while retaining the older detached steady-state inputs.
        saved_network = model._state_cache
        saved_population = model.cell._caches.pop("_steady_state", None)
        model._state_cache = {}
        model.initialize(DT, reinit_weights=False, reinit_delays=False)
        model._state_cache = saved_network
        if saved_population is not None:
            model.cell._caches["_steady_state"] = saved_population
    return model


@pytest.mark.parametrize("mode", ("transform", "population_cache", "network_cache"))
def test_population_source_writes_precede_host_sampling_and_keep_raw_alias_gradients(
    mode,
):
    from dendra.func._network_initialization import initialize_host

    model = _network_with_population_source_phase(mode)
    native = _network_with_population_source_phase(mode)
    functional, packet = dn.func.make_functional_network(model)
    global_rng = torch.get_rng_state().clone()
    initialized = initialize_host(functional, *packet, functional.initialization_inputs)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
    raw = initialized.parameters["populations"]["cell"]["cm_param.rho"]
    for collection in _COLLECTIONS:
        for values in initialized.parameters[collection].values():
            assert values["weight.weights.0.param.mean"] is raw
    native.initialize(DT)
    _check_native(functional, initialized, native)
    _close(initialized.state, functional.extract(native).state)
    observed = torch.autograd.grad(
        _materialized_loss(functional, initialized),
        packet.parameters["populations"]["cell"]["cm_param.rho"],
    )[0]
    expected = torch.autograd.grad(
        _native_materialized_loss(native), native.cell.cm_param.rho
    )[0]
    torch.testing.assert_close(observed, expected)
    assert observed.isfinite().all() and observed.abs().sum() > 0


@pytest.mark.parametrize("initialization", ("snapshot", "fresh"))
@pytest.mark.parametrize(
    ("captured_dtype", "fresh_dtype"),
    ((torch.float64, torch.float32), (torch.float32, torch.float64)),
)
def test_compilation_outside_construction_default_dtype_preserves_samples_and_gradients(
    initialization, captured_dtype, fresh_dtype
):
    previous_dtype = torch.get_default_dtype()
    try:
        # Model tensors stay float64; TruncatedNormal's historical uniform
        # sampler uses the ambient default dtype at the sampling call.
        torch.set_default_dtype(captured_dtype)
        model, *_ = _network(interpolated=True)
        native, native_weights, native_delays, _ = _network(interpolated=True)
        original = {
            collection: {
                name: connection.delay_ms.w.detach().clone()
                for name, connection in getattr(model, collection).items()
            }
            for collection in _COLLECTIONS
        }
        assert all(
            connection.delay_ms.sampling_draws["0"].dtype == captured_dtype
            for collection in _COLLECTIONS
            for connection in getattr(model, collection).values()
        )
        torch.set_default_dtype(fresh_dtype)
        global_rng = torch.get_rng_state().clone()
        runner = dn.compile(
            model,
            dt=DT,
            steps=4,
            inference=False,
            initialization=initialization,
            trainable="*.weights.0.param.*",
            backend="aot_eager",
        )
        final, auxiliary = runner()
        torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
        parameters = auxiliary.get("parameters", runner.parameters)
        constants = auxiliary.get("constants", runner.constants)
        packet = dn.func.NetworkTensors(parameters, constants, final)
        if initialization == "fresh":
            native.initialize(DT)
            assert all(
                connection.delay_ms.sampling_draws["0"].dtype == fresh_dtype
                for collection in _COLLECTIONS
                for connection in getattr(native, collection).values()
            )
        else:
            for collection, connections in _materialized(
                runner.functional, packet
            ).items():
                for name, values in connections.items():
                    torch.testing.assert_close(
                        values["delays"], original[collection][name], rtol=0, atol=0
                    )
        _check_native(runner.functional, packet, native)
        for _ in range(4):
            native.step()
        _close(final, runner.functional.extract(native).state)
        observed = torch.autograd.grad(_loss(final), _targets(packet, False))
        expected = torch.autograd.grad(
            _native_loss(native), _native_targets(native_weights, native_delays, False)
        )
        _close(observed, expected)
        assert all(
            value.dtype == DTYPE and value.isfinite().all() for value in observed
        )
        torch.testing.assert_close(
            final["connection_sampling"]["rng"]["cpu"],
            torch.get_rng_state(),
            rtol=0,
            atol=0,
        )
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize("change", ("broken", "introduced"))
def test_compatible_target_checks_exact_storage_ties_without_accepting_alias_changes(
    change,
):
    from tests.func.test_network_cache_aliases import _shared_network

    source, _, _ = _shared_network("constant")
    target, _, _ = _shared_network("constant")
    if change == "broken":
        target.cell.factor = target.cell.factor.clone()
    else:
        source.cell.factor = source.cell.factor.clone()
        # Different Tensor identities can still read the exact same storage.
        target.cell.factor = target.cell.factor.detach()
        assert target.cell.factor is not target.netstim.start
        assert (
            target.cell.factor.untyped_storage().data_ptr()
            == target.netstim.start.untyped_storage().data_ptr()
        )
    source.initialize(DT, reinit_weights=False, reinit_delays=False)
    target.initialize(DT, reinit_weights=False, reinit_delays=False)
    functional, packet = dn.func.make_functional_network(source)
    before = _clone(target.state_dict_for_checkpoint(), detached=True)
    with pytest.raises(dn.func.FunctionalizationError, match="source sharing"):
        functional.extract(target)
    with pytest.raises(dn.func.FunctionalizationError, match="source sharing"):
        functional.commit_state_(target, packet.state)
    _close(target.state_dict_for_checkpoint(), before)


def test_explicit_draw_packet_must_match_retained_role_flags():
    model, *_ = _network()
    functional, packet = dn.func.make_functional_network(model)
    draw = functional.sample_connection_initialization_draw(*packet)
    saved = _clone(packet, detached=True)
    global_rng = torch.get_rng_state().clone()
    inputs = functional.initialization_inputs._replace(
        connection_draw=draw,
        reinit_weights=False,
        reinit_delays=False,
    )
    with pytest.raises(ValueError, match="resampling flags"):
        functional.initialize(*packet, inputs)
    _close(packet, saved)
    assert torch.equal(torch.get_rng_state(), global_rng)


def test_inductor_fresh_sampling_keeps_shared_native_weight_and_delay_gradients():
    model, *_ = _network(parametrized=True, interpolated=True)
    native, native_weights, native_delays, _ = _network(
        parametrized=True, interpolated=True
    )
    runner = dn.compile(
        model,
        dt=DT,
        steps=3,
        inference=False,
        initialization="fresh",
        trainable="*.weights.0.param.*",
        backend="inductor",
    )
    initial = _clone(runner.initial_state, detached=True)
    global_rng = torch.get_rng_state().clone()
    final, auxiliary = runner()
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
    packet = dn.func.NetworkTensors(
        auxiliary["parameters"], auxiliary["constants"], final
    )
    event_raw = _parameters(packet)
    continuous_raw = next(iter(packet.parameters["continuous_synapses"].values()))
    for key, value in event_raw.items():
        if key.startswith(("weight.weights.", "delay_ms.weights.0.")):
            assert continuous_raw[key] is value
    native.initialize(DT)
    _check_native(runner.functional, packet, native)
    for _ in range(3):
        native.step()
    _close(final, runner.functional.extract(native).state)
    observed = torch.autograd.grad(_loss(final), _targets(packet, True))
    expected = torch.autograd.grad(
        _native_loss(native), _native_targets(native_weights, native_delays, True)
    )
    _close(observed, expected)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in observed)
    _close(runner.initial_state, initial)
