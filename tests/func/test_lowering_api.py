"""One lowering entry point preserves the concrete Population/Network contracts."""

from __future__ import annotations

import inspect

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._population import make_functional as _make_population
from dendra.models.mod import pas

DT = 0.025
G = "integrator.mech.mechanisms.leak.g_param"
pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
    ):
        yield


def _population(*, initialize=True):
    population = dn.SingleCompartment(
        N=2, C=1, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    population.insert(pas.rename("leak"), g=0.001, e=-45.0)
    population.train()
    if initialize:
        population.initialize(DT)
    return population


def _network(*, initialize=True):
    left, right = _population(initialize=False), _population(initialize=False)
    network = dn.Network({"left": left, "right": right})
    right.mech.leak.g_param = left.mech.leak.g_param
    network.train()
    if initialize:
        network.initialize(DT)
    return network


def _legacy(model, network, **keywords):
    if network:
        return dn.func.make_functional_network(model, **keywords)
    return _make_population(model, dt=DT, **keywords)


def _close(actual, expected):
    leaves, spec = pytree.tree_flatten(actual)
    references, other = pytree.tree_flatten(expected)
    assert spec == other
    for value, reference in zip(leaves, references, strict=True):
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, reference)
        else:
            assert value == reference


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _failure(call):
    try:
        call()
    except Exception as error:
        return type(error), str(error)
    raise AssertionError("the malformed lowering request must fail")


def _voltage(state, network):
    if network:
        return torch.cat(
            [
                value["integrator"]["v"].flatten()
                for value in state["populations"].values()
            ]
        )
    return state["integrator"]["v"]


def _target(parameters, network):
    return parameters["populations"]["left"][G] if network else parameters[G]


def _trace(network):
    return {
        "trace": dn.func.Recorder(("left.v", "right.v", "t") if network else ("v", "t"))
    }


@pytest.mark.parametrize("network", (False, True))
def test_common_lowering_preserves_concrete_packets_and_extraction_ownership(network):
    model = _network() if network else _population()
    functional, tensors = dn.func.make_functional(population=model, dt=DT)
    legacy, previous = _legacy(model, network)
    assert type(functional) is (
        dn.func.FunctionalNetwork if network else dn.func.FunctionalPopulation
    )
    assert type(tensors) is (
        dn.func.NetworkTensors if network else dn.func.PopulationTensors
    )
    assert type(functional) is type(legacy)
    assert functional.dt == DT
    _close(tensors, previous)
    _close(functional.extract().state, tensors.state)
    raw = _target(tensors.parameters, network)
    native = model.left.mech.leak.g_param if network else model.mech.leak.g_param
    if network:
        assert raw is tensors.parameters["populations"]["right"][G]
        assert raw is not native and raw.data_ptr() != native.data_ptr()
        assert raw.is_leaf and raw.grad_fn is None
    else:
        assert raw is native
    frozen = _clone(tensors.state)
    with torch.no_grad():
        raw.add_(0.0002)
    torch.testing.assert_close(native, raw if not network else raw - 0.0002)
    _close(tensors.state, frozen)


@pytest.mark.parametrize("network", (False, True))
def test_common_raw_binding_run_retains_gradients_and_recorded_checkpoint_continuation(
    network,
):
    model = _network() if network else _population()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters = tensors.independent_parameters(trainable="*g_param")
    bound = functional.bind(parameters, constants=tensors.constants)
    callbacks = functional.make_callbacks(_trace(network))
    initial = _clone(tensors.state)
    expected, whole = dn.func.run(bound, initial, steps=5, callbacks=callbacks)
    reference = torch.autograd.grad(
        _voltage(expected, network).square().sum(), _target(parameters, network)
    )[0]
    prefix, first = dn.func.run(
        bound, initial, steps=2, checkpoint_every=1, callbacks=callbacks
    )
    final, second = dn.func.run(
        bound,
        prefix,
        steps=3,
        checkpoint_every=2,
        callbacks=callbacks,
        callback_state=first["callbacks"].state,
    )
    _close(final, expected)
    observed = torch.autograd.grad(
        _voltage(final, network).square().sum(), _target(parameters, network)
    )[0]
    torch.testing.assert_close(observed, reference)
    assert torch.isfinite(observed).all() and observed.abs().sum() > 0
    for name, trace in whole["callbacks"]["trace"].items():
        head = first["callbacks"]["trace"][name]
        tail = second["callbacks"]["trace"][name]
        assert head.shape[0] == 3 and tail.shape[0] == 3
        torch.testing.assert_close(torch.cat((head, tail)), trace)
    _close(tensors.state, initial)
    _close(functional.extract().state, initial)


def test_network_legacy_to_canonical_disk_checkpoint_uses_the_same_state_and_callback_contract(
    tmp_path,
):
    source = _network()
    functional, tensors = dn.func.make_functional_network(source)
    receiver, received = dn.func.make_functional(_network(), dt=DT)
    bound = functional.bind(tensors.parameters, constants=tensors.constants)
    callbacks = functional.make_callbacks(_trace(True))
    receiving_callbacks = receiver.make_callbacks(_trace(True))
    with torch.no_grad():
        expected, whole = dn.func.run(
            bound, tensors.state, steps=5, callbacks=callbacks
        )
        prefix, first = dn.func.run(bound, tensors.state, steps=2, callbacks=callbacks)
        path = tmp_path / "lowering.pt"
        torch.save(
            {
                "state": _clone(prefix),
                "parameters": tensors.parameters,
                "constants": tensors.constants,
                "callbacks": callbacks.export_state(first["callbacks"].state),
            },
            path,
        )
        saved = torch.load(path, weights_only=True)
        receiving_bound = receiver.bind(
            saved["parameters"], constants=saved["constants"]
        )
        final, last = dn.func.run(
            receiving_bound,
            saved["state"],
            steps=3,
            callbacks=receiving_callbacks,
            callback_state=receiving_callbacks.restore_state(saved["callbacks"]),
        )
    _close(final, expected)
    for name, trace in whole["callbacks"]["trace"].items():
        torch.testing.assert_close(
            torch.cat(
                (first["callbacks"]["trace"][name], last["callbacks"]["trace"][name])
            ),
            trace,
        )
    assert (
        saved["parameters"]["populations"]["left"][G]
        is saved["parameters"]["populations"]["right"][G]
    )
    _close(received.state, tensors.state)


def test_legacy_network_factory_keeps_its_signature_and_network_only_validation():
    signature = inspect.signature(dn.func.make_functional_network)
    assert tuple(signature.parameters) == ("network", "extra", "pending_capacity")
    assert signature.parameters["extra"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        signature.parameters["pending_capacity"].kind is inspect.Parameter.KEYWORD_ONLY
    )
    with pytest.raises(
        dn.func.FunctionalizationError, match=r"^Expected an initialized Network\.$"
    ):
        dn.func.make_functional_network(_population())
    with pytest.raises(TypeError, match="unexpected keyword argument 'dt'"):
        dn.func.make_functional_network(_network(), dt=DT)


@pytest.mark.parametrize("network", (False, True))
def test_canonical_factory_requires_explicit_dt_for_each_model_kind(network):
    model = _network() if network else _population()
    with pytest.raises(TypeError, match="required keyword-only argument: 'dt'"):
        dn.func.make_functional(model)


@pytest.mark.parametrize("dt", (True, None, float("nan"), 0.0, -DT))
def test_population_dt_diagnostics_and_validation_order_match_the_original_factory(dt):
    model = object()
    assert _failure(lambda: dn.func.make_functional(model, dt=dt)) == _failure(
        lambda: _make_population(model, dt=dt)
    )


@pytest.mark.parametrize(
    ("dt", "error", "message"),
    (
        (True, TypeError, "dt must be a real scalar, not a boolean."),
        (None, TypeError, "dt must be a real scalar."),
        (float("inf"), ValueError, "dt must be finite."),
        (0.0, ValueError, "dt must be positive."),
    ),
)
def test_network_dt_scalar_validation_precedes_lowering(dt, error, message):
    model = _network(initialize=False)
    failure = _failure(
        lambda: dn.func.make_functional(model, dt=dt, extra={"missing": None})
    )
    assert failure == (error, message)


@pytest.mark.parametrize("change", ("uninitialized", "step", "extra", "capacity"))
def test_network_component_diagnostics_precede_dt_mismatch_and_match_legacy(change):
    model = _network(initialize=change != "uninitialized")
    options = {}
    if change == "step":
        model.step = lambda: None
    elif change == "extra":
        options["extra"] = {"missing": None}
    elif change == "capacity":
        options["pending_capacity"] = 0
    expected = _failure(lambda: dn.func.make_functional_network(model, **options))
    actual = _failure(lambda: dn.func.make_functional(model, dt=2 * DT, **options))
    assert actual == expected


def test_network_timestep_mismatch_rejects_without_reinitializing_or_mutating_source():
    model = _network()
    original, tensors = dn.func.make_functional_network(model)
    snapshot = _clone(tensors.state)
    with pytest.raises(
        ValueError, match="dt must match the initialized Network timestep"
    ):
        dn.func.make_functional(model, dt=2 * DT)
    assert model.dt == DT
    _close(original.extract().state, snapshot)


def test_population_rejects_network_capacity_before_attempting_lowering():
    with pytest.raises(
        ValueError, match=r"^pending_capacity applies only to Network lowering$"
    ):
        dn.func.make_functional(object(), dt=True, pending_capacity=8)


@pytest.mark.parametrize("dt", (DT, True))
def test_typed_population_runner_keeps_population_only_admission_and_dt_precedence(dt):
    model = _network()
    expected = _failure(lambda: _make_population(model, dt=dt))
    actual = _failure(
        lambda: dn.func.CompiledRunner(model, dt=dt, steps=2, backend="eager")
    )
    assert actual == expected


@pytest.mark.parametrize("dt", (DT, True))
def test_typed_network_runner_keeps_network_only_admission_and_dt_precedence(dt):
    model = _population()
    expected = (
        (TypeError, "dt must be a real scalar, not a boolean.")
        if dt is True
        else _failure(lambda: dn.func.make_functional_network(model))
    )
    actual = _failure(
        lambda: dn.func.CompiledNetworkRunner(model, dt=dt, steps=2, backend="eager")
    )
    assert actual == expected


@pytest.mark.parametrize(
    "options",
    (
        {"_bind_owned_cache": False},
        {"_retain_initialization_materializations": True},
        {"_network_stimulation_rng_scope": True},
    ),
)
def test_network_rejects_population_only_private_lowering_options(options):
    with pytest.raises(
        TypeError,
        match="Population-only lowering options cannot be used with a Network",
    ):
        dn.func.make_functional(_network(), dt=DT, **options)


@pytest.mark.parametrize("backend", ("bitpacked_history", "sparse_calendar"))
@pytest.mark.parametrize("mapped", (False, True))
def test_canonical_network_lowering_forwards_pending_capacity_and_pending_traffic(
    backend, mapped
):
    from tests.func.test_network_pending_calendar import DT as calendar_dt
    from tests.func.test_network_pending_calendar import _network as calendar_network

    model = calendar_network(backend)
    name = next(iter(model.synapses))
    capacity = {name: 32} if mapped else 32
    functional, tensors = dn.func.make_functional(
        model, dt=calendar_dt, pending_capacity=capacity
    )
    legacy, previous = dn.func.make_functional_network(model, pending_capacity=capacity)
    assert (
        functional.synapses[name].pending_capacity
        == legacy.synapses[name].pending_capacity
        == 32
    )
    _close(tensors, previous)
    bound = functional.bind(tensors.parameters, constants=tensors.constants)
    old_bound = legacy.bind(previous.parameters, constants=previous.constants)
    with torch.no_grad():
        actual, _ = dn.func.run(bound, tensors.state, steps=5)
        expected, _ = dn.func.run(old_bound, previous.state, steps=5)
    _close(actual, expected)
    assert not torch.equal(
        actual["populations"]["cell"]["integrator"]["v"],
        tensors.state["populations"]["cell"]["integrator"]["v"],
    )


def test_canonical_network_extra_retains_shared_field_waveform_sources_and_gradients():
    from tests.func.test_network_extra import PREFIX, _extra
    from tests.func.test_network_extra import _network as stimulated_network

    model = stimulated_network(batch=None)
    extra = _extra(model, shared=True)
    functional, tensors = dn.func.make_functional(model, dt=model.dt, extra=extra)
    legacy, previous = dn.func.make_functional_network(model, extra=extra)
    _close(tensors, previous)
    leaves = tensors.parameters["populations"]
    field = leaves["left"][PREFIX + "contacts.0.field"]
    amplitude = leaves["left"][PREFIX + "waveforms.0.amp"]
    assert field is leaves["right"][PREFIX + "contacts.0.field"]
    assert amplitude is leaves["right"][PREFIX + "waveforms.0.amp"]
    assert field is not extra["left"][0]
    assert amplitude is not extra["left"][1].amp
    # Prefix-qualified names are available through the same selector vocabulary.
    parameters = tensors.independent_parameters(trainable="*stimulation.extra.*")
    targets = (
        parameters["populations"]["left"][PREFIX + "contacts.0.field"],
        parameters["populations"]["left"][PREFIX + "waveforms.0.amp"],
    )
    state, _ = dn.func.run(
        functional.bind(parameters, constants=tensors.constants), tensors.state, steps=3
    )
    expected, _ = legacy.prepare_and_rollout(
        parameters, previous.constants, previous.state, steps=3
    )
    _close(state, expected)
    observed = torch.autograd.grad(_voltage(state, True).square().sum(), targets)
    reference = torch.autograd.grad(_voltage(expected, True).square().sum(), targets)
    _close(observed, reference)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed
    )
