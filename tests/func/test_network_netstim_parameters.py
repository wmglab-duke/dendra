"""Network NetStim sources retain native coordinates, aliases and renewal carry."""

from __future__ import annotations

import pytest
import torch
from torch.nn.utils.parametrize import register_parametrization

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import expsyn, pas
from dendra.models.parametric import PositiveParam
from tests.func.test_network import DT, DTYPE, _clone, _close

pytestmark = pytest.mark.cpu


class _Affine(torch.nn.Module):
    def __init__(self, factor=1.1, offset=0.03):
        super().__init__()
        self.factor = torch.nn.Parameter(torch.tensor(factor, dtype=DTYPE))
        self.register_buffer("offset", torch.tensor(offset, dtype=DTYPE))

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


def _network(*, parametrized=False, other_sources=False, seed=317):
    shared = PositiveParam(torch.tensor(0.5, dtype=DTYPE))
    if parametrized:
        register_parametrization(shared, "rho", _Affine())
    cell = dn.SingleCompartment(
        N=2,
        C=1,
        cm=shared,
        integrator=dn.bwd_euler_sc(imem=False),
        v_init=-65.0,
    )
    cell.insert(pas, g=0.001, e=-65.0)
    cell.insert(expsyn.rename("syn"), tau=1.2, e=0.0)
    stimulus = dn.NetStim(
        N=2, interval=0.5, start=0.0, noise=0.6, max_spikes=20, seed=seed
    )
    # One authored source participates in cell dynamics, connection expansion,
    # and the NetStim renewal process before any functional snapshots exist.
    stimulus.interval = shared
    stimulus.tau = torch.nn.Parameter(torch.tensor(0.17, dtype=DTYPE))
    stimulus.noise.requires_grad_()
    if other_sources:
        stimulus.start.requires_grad_()
        register_parametrization(stimulus, "noise", _Affine(0.8, 0.02))
        register_parametrization(stimulus, "start", _Affine(0.9, 0.01))
        register_parametrization(stimulus, "tau", _Affine(0.9, 0.01))
    model = dn.Network(
        {"cell": cell},
        netstim=stimulus,
        netcon_delay_backend="dense",
        netcon_train_backend="dense",
    )
    model.connect_one_to_one(
        stimulus[:], cell[:], cell.mech.syn, weight=shared, delay=DT
    )
    model.train()
    model.initialize(DT)
    model.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=False,
        diff_spiking=True,
        diff_scheduled_times=False,
        train_delay_backend="dense",
    )
    return model, shared


def _raw_names(parametrized):
    if parametrized:
        return (
            "interval.parametrizations.rho.original",
            "interval.parametrizations.rho.0.factor",
        )
    return ("interval.rho",)


def _targets(parameters, parametrized):
    values = parameters["netstim"]["parameters"]
    return tuple(values[name] for name in _raw_names(parametrized))


def _native_targets(source, parametrized):
    if parametrized:
        return source.parametrizations.rho.original, source.parametrizations.rho[
            0
        ].factor
    return (source.rho,)


def _trainable(parametrized):
    return tuple(f"netstim.parameters.{name}" for name in _raw_names(parametrized))


def _assert_aliases(parameters, parametrized):
    for name in _raw_names(parametrized):
        suffix = name.removeprefix("interval.")
        value = parameters["netstim"]["parameters"][name]
        assert value.ndim == 0
        assert value is parameters["populations"]["cell"][f"cm_param.{suffix}"]
        assert (
            value
            is next(iter(parameters["synapses"].values()))[
                f"weight.weights.0.param.{suffix}"
            ]
        )
    assert "interval" not in parameters["netstim"]["parameters"]


def _loss(state):
    return (
        state["populations"]["cell"]["integrator"]["v"].square().sum()
        + 0.1 * state["netstim"]["next_stoch_time"].square().sum()
    )


def _native_loss(model):
    return (
        model.cell.v.square().sum() + 0.1 * model.netstim.next_stoch_time.square().sum()
    )


@pytest.mark.parametrize("check_purity", (False, True))
@pytest.mark.parametrize("parametrized", (False, True))
def test_shared_raw_netstim_sources_materialize_without_source_or_carry_mutation(
    check_purity, parametrized
):
    model, source = _network(parametrized=parametrized)
    with dn.ctx(CHECK_PURITY=check_purity):
        plan, tensors = dn.func.make_functional_network(model)
    _assert_aliases(tensors.parameters, parametrized)
    parameters = tensors.independent_parameters(_trainable(parametrized))
    _assert_aliases(parameters, parametrized)
    frozen = _clone((tensors.constants, tensors.state), detached=True)
    native_raw = tuple(
        value.detach().clone() for value in _native_targets(source, parametrized)
    )
    actual = plan._materialize_netstim_parameters(parameters, tensors.constants)
    torch.testing.assert_close(actual["interval"], source._compute().expand(2))
    with torch.no_grad():
        _targets(parameters, parametrized)[0].add_(0.2)
    changed = plan._materialize_netstim_parameters(parameters, tensors.constants)
    assert (changed["interval"] > actual["interval"]).all()
    for value, reference in zip(
        _native_targets(source, parametrized), native_raw, strict=True
    ):
        torch.testing.assert_close(value, reference)
    _close((tensors.constants, tensors.state), frozen)


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_parametrized_shared_netstim_sources_restart_and_continue_checkpoint(execution):
    model, _ = _network(parametrized=True)
    native, native_source = _network(parametrized=True)
    native.netstim.restore_dict_from_checkpoint(
        _clone(model.netstim.state_dict_for_checkpoint(), detached=True)
    )
    options = {"checkpoint_every": 2} if execution == "scan" else {}
    runner = dn.compile(
        model,
        dt=DT,
        steps=4,
        inference=execution == "while",
        execution=execution,
        trainable=_trainable(True),
        callbacks={"trace": dn.func.Recorder(("cell.v", "t"))},
        backend="aot_eager",
        **options,
    )
    _assert_aliases(runner.parameters, True)
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
        observed = torch.autograd.grad(_loss(final), _targets(runner.parameters, True))
        expected = torch.autograd.grad(
            _native_loss(native), _native_targets(native_source, True)
        )
        _close(observed, expected)
        assert all(
            value.isfinite().all() and value.abs().sum() > 0 for value in observed
        )
    else:
        assert runner.execution_report().strategy == "compiled_while_loop"
    _close(runner.initial_state, initial)
    target, _ = _network(parametrized=True)
    runner.functional.commit_state_(target, final)
    resumed, _ = _network(parametrized=True)
    resumed.restore_dict_from_checkpoint(target.state_dict_for_checkpoint())
    continued, _ = runner.functional.prepare_and_step(
        runner.parameters,
        runner.constants,
        final,
        netstim_draw=runner.functional.sample_step_draw(final),
    )
    resumed.step()
    _close(continued, runner.functional.extract(resumed).state)


@pytest.mark.parametrize("compiled", (False, True))
def test_parametrized_netstim_buffers_keep_originals_and_native_raw_vjps(compiled):
    model, _ = _network(other_sources=True)
    native, _ = _network(other_sources=True)
    native.netstim.restore_dict_from_checkpoint(
        _clone(model.netstim.state_dict_for_checkpoint(), detached=True)
    )
    plan, tensors = dn.func.make_functional_network(model)
    raw = tensors.parameters["netstim"]["parameters"]
    for name in ("noise", "start", "tau"):
        assert f"parametrizations.{name}.original" in raw
        assert f"parametrizations.{name}.0.factor" in raw
        assert name not in raw
    assert raw["parametrizations.tau.original"].ndim == 0
    values = plan._materialize_netstim_parameters(tensors.parameters, tensors.constants)
    for name in ("noise", "start", "tau"):
        torch.testing.assert_close(
            values[name], torch.broadcast_to(getattr(model.netstim, name), (2,))
        )
    tape = plan.sample_draw_tape(tensors.state, 7)
    inputs = dn.func.NetworkRolloutInput(netstim=tape)
    prepared = plan.prepare(tensors.parameters, tensors.constants)
    if compiled:
        execute = plan.compile_rollout_chunk(7, execution="scan", backend="aot_eager")
    else:
        execute = plan.rollout
    final, _ = execute(tensors.parameters, prepared, tensors.state, inputs)
    for _ in range(7):
        native.step()
    _close(final, plan.extract(native).state)
    names = (
        "interval.rho",
        "parametrizations.noise.original",
        "parametrizations.noise.0.factor",
        "parametrizations.tau.original",
        "parametrizations.tau.0.factor",
    )
    actual = torch.autograd.grad(_loss(final), tuple(raw[name] for name in names))
    expected = torch.autograd.grad(
        _native_loss(native),
        (
            native.netstim.interval.rho,
            native.netstim.parametrizations.noise.original,
            native.netstim.parametrizations.noise[0].factor,
            native.netstim.parametrizations.tau.original,
            native.netstim.parametrizations.tau[0].factor,
        ),
    )
    _close(actual, expected)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in actual)
    start = raw["parametrizations.start.original"]
    start_gradient = torch.autograd.grad(values["start"].sum(), start)[0]
    torch.testing.assert_close(
        start_gradient, model.netstim.parametrizations.start[0].factor.expand_as(start)
    )


def test_inductor_parametrized_netstim_sources_preserve_shared_native_vjp():
    model, _ = _network(parametrized=True)
    native, native_source = _network(parametrized=True)
    native.netstim.restore_dict_from_checkpoint(
        _clone(model.netstim.state_dict_for_checkpoint(), detached=True)
    )
    runner = dn.compile(
        model,
        dt=DT,
        steps=3,
        inference=False,
        trainable=_trainable(True),
        backend="inductor",
    )
    final, _ = runner()
    for _ in range(3):
        native.step()
    _close(final, runner.functional.extract(native).state)
    observed = torch.autograd.grad(_loss(final), _targets(runner.parameters, True))
    expected = torch.autograd.grad(
        _native_loss(native), _native_targets(native_source, True)
    )
    _close(observed, expected)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in observed)


def _mixed_config_network(*, cache=None, shared_noise=False):
    model, source = _network(parametrized=True)
    # The same ordinary Tensor is a primary, selectable NetStim input and a
    # registered config buffer in Population and connection source transforms.
    del model.netstim.tau
    model.netstim.tau = source.parametrizations.rho[0].offset
    if shared_noise:
        model.netstim.noise = model.netstim.tau
    model.initialize(DT)
    if cache is not None:
        with torch.no_grad():
            if cache == "network":
                model.cache_state()
            else:
                model.cell.cache("_steady_state")
            model.netstim.tau.add_(0.12)
        # Keep live materializations coherent with the changed raw config while
        # retaining the independent cache that fresh initialization restores.
        saved_network = model._state_cache
        saved_population = model.cell._caches.pop("_steady_state", None)
        model._state_cache = {}
        model.initialize(DT)
        model._state_cache = saved_network
        if saved_population is not None:
            model.cell._caches["_steady_state"] = saved_population
    return model


def _mixed_targets(parameters):
    return parameters["netstim"]["parameters"]["tau"]


def _assert_mixed_constants(plan, parameters, constants):
    from dendra.func._network import _at

    target = _mixed_targets(parameters)
    readers = [
        reader
        for path, paths in plan._parameter_constant_aliases
        if _at(parameters, path) is target
        for reader in paths
    ]
    assert {path[0] for path in readers} >= {"populations", "synapses"}
    assert all(_at(constants, path) is target for path in readers)


@pytest.mark.parametrize("source_requires_grad", (False, True))
def test_mixed_primary_and_config_aliases_keep_all_materialization_gradients(
    source_requires_grad,
):
    model, native = _mixed_config_network(), _mixed_config_network()
    model.netstim.tau.requires_grad_(source_requires_grad)
    plan, packet = dn.func.make_functional_network(model)
    parameters = packet.independent_parameters(("netstim.parameters.tau",))
    target = _mixed_targets(parameters)
    saved_constants = _clone(packet.constants, detached=True)
    native.netstim.tau.requires_grad_()
    native.initialize(DT)
    values = plan._prepare_values(parameters, packet.constants)
    connection = next(iter(model.synapses))
    actual = (
        values["populations"]["cell"]["population"]["cm"].sum(),
        values["connections"]["synapses"][connection]["weights"].sum(),
        values["netstim"]["parameters"]["interval"].sum(),
    )
    expected = (
        native.cell.cm.sum(),
        native.synapses[connection].weight.w.sum(),
        native.netstim.renewal_parameters()["interval"].sum(),
    )
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right)
        observed = torch.autograd.grad(left, target, retain_graph=True)[0]
        reference = torch.autograd.grad(right, native.netstim.tau, retain_graph=True)[0]
        torch.testing.assert_close(observed, reference)
        assert observed.abs() > 0
    with torch.no_grad():
        target.add_(0.4)
    changed = plan._prepare_values(parameters, packet.constants)
    assert (
        changed["populations"]["cell"]["population"]["cm"]
        > values["populations"]["cell"]["population"]["cm"]
    ).all()
    assert (
        changed["connections"]["synapses"][connection]["weights"]
        > values["connections"]["synapses"][connection]["weights"]
    ).all()
    _close(packet.constants, saved_constants)


def test_mixed_config_preparation_keeps_fresh_graphs_and_checks_alias_freshness():
    model = _mixed_config_network()
    plan, packet = dn.func.make_functional_network(model)
    parameters = packet.independent_parameters(("netstim.parameters.tau",))
    target = _mixed_targets(parameters)
    prepared = plan.prepare(parameters, packet.constants)
    gradients = []
    for _ in range(2):
        final, _ = plan.step(
            parameters,
            prepared,
            packet.state,
            netstim_draw=plan.sample_step_draw(packet.state),
        )
        gradients.append(torch.autograd.grad(_loss(final), target)[0])
    torch.testing.assert_close(*gradients)
    replacement = {
        **parameters,
        "netstim": {
            **parameters["netstim"],
            "parameters": {
                key: value.clone() if value is target else value
                for key, value in parameters["netstim"]["parameters"].items()
            },
        },
    }
    # Preserve the intra-NetStim alias while replacing its canonical Tensor.
    new_target = replacement["netstim"]["parameters"]["tau"]
    replacement["netstim"]["parameters"] = {
        key: new_target if value is target else value
        for key, value in parameters["netstim"]["parameters"].items()
    }
    with pytest.raises(dn.func.FunctionalizationError, match="prepare"):
        plan.step(
            replacement,
            prepared,
            packet.state,
            netstim_draw=plan.sample_step_draw(packet.state),
        )
    with torch.no_grad():
        target.add_(0.1)
    with pytest.raises(dn.func.FunctionalizationError, match="prepare"):
        plan.step(
            parameters,
            prepared,
            packet.state,
            netstim_draw=plan.sample_step_draw(packet.state),
        )


@pytest.mark.parametrize("initialization", ("snapshot", "fresh"))
def test_compiled_mixed_config_aliases_preserve_native_updates_and_raw_vjp(
    initialization,
):
    model, native = _mixed_config_network(), _mixed_config_network()
    native.netstim.restore_dict_from_checkpoint(
        _clone(model.netstim.state_dict_for_checkpoint(), detached=True)
    )
    runner = dn.compile(
        model,
        dt=DT,
        steps=6,
        initialization=initialization,
        inference=False,
        trainable=("netstim.parameters.tau",),
        backend="aot_eager",
        checkpoint_every=2,
    )
    _assert_mixed_constants(runner.functional, runner.parameters, runner.constants)
    target = _mixed_targets(runner.parameters)
    native.netstim.tau.requires_grad_()
    native.initialize(DT)
    actual, _ = runner()
    for _ in range(6):
        native.step()
    _close(actual, runner.functional.extract(native).state)
    observed = torch.autograd.grad(_loss(actual), target)[0]
    expected = torch.autograd.grad(_native_loss(native), native.netstim.tau)[0]
    torch.testing.assert_close(observed, expected)
    assert observed.isfinite() and observed.abs() > 0
    # A continuation override keeps returned constants bound to its new raw
    # input, including fresh-policy runners that publish continuation packets.
    parameters = runner.functional.extract().independent_parameters(
        ("netstim.parameters.tau",)
    )
    with torch.no_grad():
        _mixed_targets(parameters).add_(0.1)
    continued, auxiliary = runner(actual, parameters=parameters)
    assert continued["clock"]["step"].item() == 12
    if initialization == "fresh":
        _assert_mixed_constants(
            runner.functional, auxiliary["parameters"], auxiliary["constants"]
        )


@pytest.mark.parametrize("cache", ("population", "network"))
def test_cache_restored_mixed_config_aliases_keep_initialization_values_and_vjp(cache):
    model, native = (
        _mixed_config_network(cache=cache),
        _mixed_config_network(cache=cache),
    )
    native.netstim.restore_dict_from_checkpoint(
        _clone(model.netstim.state_dict_for_checkpoint(), detached=True)
    )
    plan, packet = dn.func.make_functional_network(model)
    parameters = packet.independent_parameters(("netstim.parameters.tau",))
    target = _mixed_targets(parameters)
    native.netstim.tau.requires_grad_()
    initialized = plan._initialize_host(
        parameters, packet.constants, packet.state, plan.initialization_inputs
    )
    native.initialize(DT)
    _assert_mixed_constants(plan, initialized.parameters, initialized.constants)
    torch.testing.assert_close(
        _mixed_targets(initialized.parameters), native.netstim.tau
    )
    actual, _ = plan.run(*initialized, steps=6)
    for _ in range(6):
        native.step()
    _close(actual, plan.extract(native).state)
    observed = torch.autograd.grad(_loss(actual), target)[0]
    expected = torch.autograd.grad(_native_loss(native), native.netstim.tau)[0]
    torch.testing.assert_close(observed, expected)


def test_initialization_sampler_uses_replaced_shared_noise_input():
    model = _mixed_config_network(shared_noise=True)
    plan, packet = dn.func.make_functional_network(model)
    parameters = packet.independent_parameters(("netstim.parameters.tau",))
    with torch.no_grad():
        _mixed_targets(parameters).zero_()
    draw = plan.sample_initialization_draw(
        parameters, packet.state, constants=packet.constants
    )
    assert torch.equal(draw.exp_rand, torch.zeros_like(draw.exp_rand))
    inputs = plan.initialization_inputs._replace(netstim_draw=draw)
    initialized = plan.initialize(parameters, packet.constants, packet.state, inputs)
    _assert_mixed_constants(plan, initialized.parameters, initialized.constants)
    assert torch.equal(
        initialized.state["netstim"]["next_stoch_time"],
        model.netstim.start,
    )


@pytest.mark.parametrize("shared_config", (False, True))
def test_compatible_targets_preserve_mixed_source_alias_topology(shared_config):
    models = []
    for shared in (shared_config, not shared_config):
        model = _mixed_config_network()
        # The interval has independent raw Parameters, so the only external
        # NetStim tie is tau's plain Tensor/config-buffer relationship.
        model.netstim.interval = PositiveParam(torch.tensor(0.5, dtype=DTYPE))
        if not shared:
            model.netstim.tau = model.netstim.tau.clone()
        model.initialize(DT)
        models.append(model)
    model, target = models
    plan, packet = dn.func.make_functional_network(model)
    before = _clone(target.state_dict_for_checkpoint(), detached=True)
    with pytest.raises(dn.func.FunctionalizationError, match="source sharing"):
        plan.extract(target)
    with pytest.raises(dn.func.FunctionalizationError, match="source sharing"):
        plan.commit_state_(target, packet.state)
    _close(target.state_dict_for_checkpoint(), before)
    model.netstim.tau = model.netstim.tau.clone()
    with pytest.raises(dn.func.FunctionalizationError, match="structure"):
        plan.prepare(packet.parameters, packet.constants)
