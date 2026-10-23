"""Raw NetStim renewal sources retain native coordinates and pure evaluation."""

import copy

import pytest
import torch
from torch.nn.utils.parametrize import register_parametrization

import dendra as dn
from dendra.func._netstim_parameters import NetStimParameterPlan
from dendra.func._network_bindings import AliasSnapshots

pytestmark = pytest.mark.cpu
DTYPE = torch.float64


class Affine(torch.nn.Module):
    def __init__(self, *, scale=1.1):
        super().__init__()
        self.factor = torch.nn.Parameter(torch.tensor(scale, dtype=DTYPE))
        self.register_buffer("offset", torch.tensor(0.01, dtype=DTYPE))

    def forward(self, value):
        return self.factor * value + self.offset


class RandomTransform(torch.nn.Module):
    def forward(self, value):
        return value + torch.rand_like(value)


class CountingTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return value


def _source(*, parametrized=False, batch=None, tau=0.15):
    with dn.ctx(JIT=0, REQUIRE_GRAD=True):
        source = dn.NetStim(
            N=2,
            interval=(0.4, 0.7),
            start=(-0.1, 0.2),
            noise=(0.2, 0.6),
            tau=tau,
            dtype=DTYPE,
            seed=31,
        )
        if batch is not None:
            source.batch(batch)
        if parametrized:
            register_parametrization(source.interval, "rho", Affine())
            register_parametrization(source, "noise", Affine(scale=0.8))
            register_parametrization(source, "start", Affine(scale=0.9))
        source.initialize()
    return source


def _close(actual, expected):
    assert set(actual) == set(expected)
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key])


@pytest.mark.parametrize("parametrized", [False, True])
@pytest.mark.parametrize("batch", [None, 2])
def test_sources_materialize_native_fields_and_raw_vjps(parametrized, batch):
    source = _source(parametrized=parametrized, batch=batch)
    plan = NetStimParameterPlan(source)
    parameters, constants = plan.extract()
    effective = plan.materialize(parameters, constants)
    native = source.renewal_parameters()
    _close(effective, native)
    assert "interval" not in parameters
    assert (
        "interval.parametrizations.rho.original" in parameters
        if parametrized
        else "interval.rho" in parameters
    )
    assert {"tau"}.issubset(parameters)
    native_sources = plan.parameter_sources()
    trainable = tuple(
        name for name, value in native_sources.items() if value.requires_grad
    )
    actual = torch.autograd.grad(
        sum(value.square().sum() for value in effective.values()),
        [parameters[name] for name in trainable],
    )
    expected = torch.autograd.grad(
        sum(value.square().sum() for value in native.values()),
        [native_sources[name] for name in trainable],
    )
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right)
    if parametrized:
        assert "parametrizations.noise.0.offset" in constants
        assert "interval.parametrizations.rho.0.offset" in constants


@pytest.mark.parametrize("parametrized", [False, True])
def test_fullgraph_and_reusable_prepared_values_have_fresh_backward_graphs(
    parametrized,
):
    source = _source(parametrized=parametrized)
    plan = NetStimParameterPlan(source)
    parameters, constants = plan.extract()
    prepared = plan.prepare(parameters, constants)
    key = next(name for name in parameters if name.startswith("interval."))
    gradient = None
    for _ in range(2):
        values = plan.validate_prepared(prepared, parameters)
        current = torch.autograd.grad(
            values["interval"].square().sum(), parameters[key]
        )[0]
        if gradient is not None:
            torch.testing.assert_close(current, gradient)
        gradient = current
    compiled = torch.compile(plan.materialize, backend="eager", fullgraph=True)
    _close(compiled(parameters, constants), plan.materialize(parameters, constants))


def test_plain_tensor_tau_is_explicit_and_keeps_cross_component_identity():
    tau = torch.tensor(0.15, dtype=DTYPE, requires_grad=True)
    source = _source(tau=tau)
    plan = NetStimParameterPlan(source)
    snapshots = AliasSnapshots()
    outside = snapshots.snapshot(tau, tau)
    parameters, constants = plan.extract(snapshots)
    assert parameters["tau"] is outside
    assert plan.parameter_sources()["tau"] is tau
    effective = plan.materialize(parameters, constants)
    gradient = torch.autograd.grad(effective["tau"].sum(), outside)[0]
    torch.testing.assert_close(gradient, torch.tensor(2.0, dtype=DTYPE))


def test_float_tau_and_fixed_noise_remain_selectable_raw_inputs():
    source = _source()
    plan = NetStimParameterPlan(source)
    parameters, constants = plan.extract()
    assert parameters["tau"].shape == ()
    assert "noise" in parameters and "start" in parameters
    parameters["tau"].requires_grad_()
    parameters["noise"].requires_grad_()
    values = plan.materialize(parameters, constants)
    gradients = torch.autograd.grad(
        values["tau"].sum() + values["noise"].sum(),
        [parameters["tau"], parameters["noise"]],
    )
    torch.testing.assert_close(gradients[0], torch.tensor(2.0, dtype=DTYPE))
    torch.testing.assert_close(gradients[1], torch.ones(2, dtype=DTYPE))


def test_native_eval_materialization_does_not_reuse_stale_interval_cache():
    source = _source().eval()
    previous = source.interval().clone()
    assert source.interval._cache is not None
    with torch.no_grad():
        source.interval.rho.add_(0.2)
    actual = source.renewal_parameters()["interval"]
    torch.testing.assert_close(actual, source.interval._compute())
    assert not torch.equal(actual, previous)
    plan = NetStimParameterPlan(source)
    _close(plan.materialize(*plan.extract()), source.renewal_parameters())


def test_native_interval_forward_hooks_still_run_and_functional_adapter_rejects_them():
    source = _source()
    handle = source.interval.register_forward_hook(
        lambda module, args, value: value * 2
    )
    try:
        torch.testing.assert_close(
            source.renewal_parameters()["interval"], source.interval._compute() * 2
        )
        with pytest.raises(dn.func.FunctionalizationError, match="hooks"):
            NetStimParameterPlan(source)
    finally:
        handle.remove()


@pytest.mark.parametrize("source_type", [RandomTransform, CountingTransform])
def test_purity_diagnostics_are_opt_in_and_restore_global_rng(source_type):
    source = _source()
    register_parametrization(source, "noise", source_type())
    NetStimParameterPlan(source)
    rng = torch.random.get_rng_state().clone()
    with dn.ctx(CHECK_PURITY=True):
        with pytest.raises(dn.func.FunctionalizationError):
            NetStimParameterPlan(source)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)


def test_compatible_sources_preserve_alias_topology_and_target_values():
    source = _source(parametrized=True)
    plan = NetStimParameterPlan(source)
    target = copy.deepcopy(source)
    with torch.no_grad():
        target.interval.parametrizations.rho.original.add_(0.3)
    target.tau = 0.25
    raw, constants = plan.extract(source=target)
    _close(plan.materialize(raw, constants), target.renewal_parameters())
    assert raw["tau"].item() == 0.25


def test_source_binding_replacement_and_prepared_mutation_are_guarded():
    source = _source(parametrized=True)
    plan = NetStimParameterPlan(source)
    raw, constants = plan.extract()
    prepared = plan.prepare(raw, constants)
    with torch.no_grad():
        prepared.values["interval"].add_(1)
    with pytest.raises(dn.func.FunctionalizationError, match="prepare again"):
        plan.validate_prepared(prepared, raw)
    source.interval.parametrizations.rho[0].factor = torch.nn.Parameter(
        torch.tensor(1.2, dtype=DTYPE)
    )
    with pytest.raises(dn.func.FunctionalizationError, match="structure"):
        plan.materialize(raw, constants)


@pytest.mark.parametrize("view", [False, True])
def test_source_buffers_cannot_alias_native_runtime_storage(view):
    source = _source(parametrized=True)
    source.parametrizations.noise[0].offset = (
        source.next_stoch_time.view(-1) if view else source.next_stoch_time
    )
    with pytest.raises(dn.func.FunctionalizationError, match="aliases runtime"):
        NetStimParameterPlan(source)


def test_low_level_effective_api_accepts_torch_parametrized_native_source():
    source = _source(parametrized=True)
    plan, tensors = dn.func.make_functional_netstim(source)
    _close(tensors.parameters, source.renewal_parameters())
    assert set(tensors.parameters) == {"interval", "start", "noise", "tau"}
    draw = plan.sample_step_draw(tensors.state)
    state, output = plan.step(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        torch.tensor(0.3, dtype=DTYPE),
        draw,
        bptt=True,
    )
    source(torch.tensor(0.3, dtype=DTYPE), bptt=True)
    torch.testing.assert_close(state["next_stoch_time"], source.next_stoch_time)
    torch.testing.assert_close(output.spike_gate, source.spike_gate)


def test_instance_materialization_overrides_cannot_escape_source_binding():
    source = _source()
    source.renewal_parameters = lambda: {}
    with pytest.raises(dn.func.FunctionalizationError, match="canonical"):
        NetStimParameterPlan(source)
    with pytest.raises(dn.func.FunctionalizationError, match="canonical"):
        dn.func.make_functional_netstim(source)


@pytest.mark.parametrize("interval_alias", [False, True])
def test_primary_aliases_retain_each_native_selection_path(interval_alias):
    source = _source()
    source.noise = source.interval.rho if interval_alias else source.start
    source.initialize()
    plan = NetStimParameterPlan(source)
    raw, constants = plan.extract()
    other = "interval.rho" if interval_alias else "start"
    assert other in raw and "noise" in raw
    assert raw[other] is raw["noise"]
    _close(plan.materialize(raw, constants), source.renewal_parameters())
    broken = {**raw, "noise": raw["noise"].clone()}
    with pytest.raises(dn.func.FunctionalizationError, match="shared tensor aliases"):
        plan.materialize(broken, constants)


def test_transform_config_aliases_retain_native_paths_and_shared_identity():
    source = _source(parametrized=True)
    source.parametrizations.noise[0].offset = source.interval.parametrizations.rho[
        0
    ].offset
    plan = NetStimParameterPlan(source)
    raw, constants = plan.extract()
    left, right = (
        "parametrizations.noise.0.offset",
        "interval.parametrizations.rho.0.offset",
    )
    assert constants[left] is constants[right]
    _close(plan.materialize(raw, constants), source.renewal_parameters())
    with pytest.raises(dn.func.FunctionalizationError, match="shared tensor aliases"):
        plan.materialize(raw, {**constants, left: constants[left].clone()})


@pytest.mark.parametrize("view", [False, True])
def test_source_config_cannot_alias_unregistered_runtime_gate(view):
    source = _source(parametrized=True)
    source.parametrizations.noise[0].offset = (
        source.spike_gate.view(-1) if view else source.spike_gate
    )
    with pytest.raises(dn.func.FunctionalizationError, match="aliases runtime"):
        NetStimParameterPlan(source)


@pytest.mark.parametrize("parametrized", [False, True])
def test_explicit_primary_inputs_can_read_population_storage(parametrized):
    source = _source()
    cell = dn.SingleCompartment(N=2, C=1, dtype=DTYPE)
    source.start = cell.v.reshape(-1)
    source.tau = cell.celsius
    if parametrized:
        register_parametrization(source, "start", Affine())
    plan = NetStimParameterPlan(source, owners=(cell,))
    _close(plan.materialize(*plan.extract()), source.renewal_parameters())


def test_hidden_transform_config_cannot_borrow_population_state():
    source = _source(parametrized=True)
    cell = dn.SingleCompartment(N=2, C=1, dtype=DTYPE)
    source.parametrizations.noise[0].offset = cell.v.reshape(-1)
    with pytest.raises(dn.func.FunctionalizationError, match="aliases runtime"):
        NetStimParameterPlan(source, owners=(cell,))


def test_primary_inputs_still_cannot_read_netstim_runtime_gate():
    source = _source()
    source.start = source.spike_gate
    with pytest.raises(dn.func.FunctionalizationError, match="aliases runtime"):
        NetStimParameterPlan(source)
