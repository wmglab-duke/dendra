"""Network fresh starts restore individual Population-owned steady states."""

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from tests.func.test_network import DT, DTYPE, _clone, _close, _stim_network
from tests.func.test_network_bitpacked_inference import _sync_stimulus
from tests.func.test_network_cache_initialization import (
    _cache_initialized_native_packet,
)
from tests.func.test_network_initialization import PARAMETER, _shared_network

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch_compiler_warning_context():
            with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
                yield
    finally:
        torch.set_num_threads(previous)


def _mixed_owned_network():
    left = dn.SingleCompartment(
        N=1, C=1, integrator=dn.bwd_euler_sc(imem=False), v_init=-40.0
    )
    left.insert(pas, g=0.001, e=-65.0)
    left.steady_state(dt=DT, tstop=4 * DT, with_ve=False, with_intra=False)
    right = dn.SingleCompartment(
        N=1, C=1, integrator=dn.bwd_euler_sc(imem=False), v_init=-45.0
    )
    right.insert(pas, g=0.001, e=-65.0)
    network = dn.Network({"left": left, "right": right})
    network.train()
    network.initialize(DT)
    assert not network._state_cache
    return network


def _owned_stim_network():
    network = _stim_network()
    network.cell.v_init = -45.0
    network.cell.steady_state(dt=DT, tstop=4 * DT, with_ve=False, with_intra=False)
    network.initialize(DT, reinit_weights=False, reinit_delays=False)
    assert not network._state_cache
    return network


def test_network_mixes_population_owned_cache_and_uncached_fresh_initialization():
    source, native = _mixed_owned_network(), _mixed_owned_network()
    functional, tensors = dn.func.make_functional_network(source)
    defaults = functional.initialization_inputs
    assert defaults.cache is None
    assert defaults.populations["left"].cache is not None
    assert defaults.populations["right"].cache is None
    frozen = _clone((defaults, tensors), detached=True)
    initialized = functional.initialize(
        *tensors, defaults._replace(t=defaults.t.new_tensor(0.25))
    )
    native.initialize(DT, t=0.25, reinit_weights=False, reinit_delays=False)
    expected = _cache_initialized_native_packet(functional, native)
    _close(initialized, expected)
    left = initialized.state["populations"]["left"]["integrator"]["v"]
    right = initialized.state["populations"]["right"]["integrator"]["v"]
    torch.testing.assert_close(left, source.left.v)
    assert not torch.allclose(left, defaults.populations["left"].v_init)
    torch.testing.assert_close(right, torch.full_like(right, -45))
    actual, _ = functional.prepare_and_step(*initialized)
    native.step()
    _close(actual, functional.extract(native).state)
    _close((defaults, tensors), frozen)


@pytest.mark.parametrize("owner", ("left", "right"))
def test_population_cache_and_fresh_transforms_publish_shared_parameter_writes_in_order(
    owner,
):
    def network():
        model = _shared_network()
        model.populations[owner].cache("_steady_state")
        with torch.no_grad():
            model.left.get_parameter(PARAMETER).fill_(0.009)
        return model

    source, native = network(), network()
    functional, tensors = dn.func.make_functional_network(source)
    initialized = functional.initialize(*tensors, functional.initialization_inputs)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    _close(initialized, _cache_initialized_native_packet(functional, native))
    left = initialized.parameters["populations"]["left"][PARAMETER]
    right = initialized.parameters["populations"]["right"][PARAMETER]
    assert left is right
    torch.testing.assert_close(
        left, torch.full_like(left, 0.0044 if owner == "left" else 0.0022)
    )
    actual, _ = functional.prepare_and_step(*initialized)
    native.step()
    _close(actual, functional.extract(native).state)


def _retained_alias_network(cached_first):
    cells = {
        name: dn.SingleCompartment(
            N=1, C=1, integrator=dn.bwd_euler_sc(imem=False), v_init=-50.0
        )
        for name in (("cached", "fresh") if cached_first else ("fresh", "cached"))
    }
    for population in cells.values():
        population.insert(pas, g=0.001, e=-65.0)
    network = dn.Network(cells)
    network.train()
    network.initialize(DT)
    fresh = next(iter(cells["fresh"].mech.mechanisms.values()))
    cached = next(iter(cells["cached"].mech.mechanisms.values()))
    if cached_first:
        # A later fresh population writes through the cached owner's retained
        # effective buffer, after that owner has already prepared its solver.
        cached.g = fresh.g
        field = "integrator.mech.mechanisms.pas.g"
    else:
        # An otherwise unused saved field still writes through an earlier
        # fresh Population's effective conductance when its cache is restored.
        cells["cached"].register_buffer("shared_effective_g", fresh.g)
        field = "shared_effective_g"
    cells["cached"].cache("_steady_state")
    cells["cached"]._caches["_steady_state"][field].fill_(0.003)
    return network


@pytest.mark.parametrize("cached_first", (False, True))
def test_owned_cache_shared_retained_materializations_follow_native_order(cached_first):
    network, native = (
        _retained_alias_network(cached_first),
        _retained_alias_network(cached_first),
    )
    functional, tensors = dn.func.make_functional_network(network)
    before = _clone(tensors, detached=True)
    initialized = functional.initialize(*tensors, functional.initialization_inputs)
    _close(tensors, before)
    native.initialize(DT)
    _close(initialized.state, functional.extract(native).state)
    actual = initialized.state
    for _ in range(3):
        actual, _ = functional.prepare_and_step(
            initialized.parameters, initialized.constants, actual
        )
        native.step()
        _close(actual, functional.extract(native).state)
    fresh = next(iter(native.populations["fresh"].mech.mechanisms.values()))
    cached = next(iter(native.populations["cached"].mech.mechanisms.values()))
    expected = 0.001 if cached_first else 0.003
    torch.testing.assert_close(fresh.g, torch.full_like(fresh.g, expected))
    if cached_first:
        torch.testing.assert_close(cached.g, fresh.g)


def _late_runtime_cache_network():
    shared = torch.nn.Parameter(torch.tensor(0.001, dtype=DTYPE))
    cells = {
        name: dn.SingleCompartment(N=1, C=1, integrator=dn.bwd_euler_sc(imem=False))
        for name in ("fresh", "cached")
    }
    for population in cells.values():
        population.insert(pas, g=shared)
    network = dn.Network(cells)
    network.train()
    network.initialize(DT)
    mechanisms = [
        next(iter(population.mech.mechanisms.values())) for population in cells.values()
    ]
    assert all(mechanism.g_param is shared for mechanism in mechanisms)
    cells["cached"].cache("_steady_state")
    cells["cached"]._caches["_steady_state"][
        "integrator.mech.mechanisms.pas.g_param"
    ].fill_(0.003)
    return network


def test_later_owned_cache_shared_runtime_parameter_preserves_earlier_fresh_reader():
    network, native = _late_runtime_cache_network(), _late_runtime_cache_network()
    functional, tensors = dn.func.make_functional_network(network)
    frozen = _clone((tensors, functional.initialization_inputs), detached=True)
    initialized = functional.initialize(*tensors, functional.initialization_inputs)
    key = "integrator.mech.mechanisms.pas.g_param"
    assert (
        initialized.parameters["populations"]["fresh"][key]
        is initialized.parameters["populations"]["cached"][key]
    )
    native.initialize(DT)
    actual = initialized.state
    for _ in range(3):
        actual, _ = functional.prepare_and_step(
            initialized.parameters, initialized.constants, actual
        )
        native.step()
        _close(actual, functional.extract(native).state)
    _close((tensors, functional.initialization_inputs), frozen)


def test_network_global_cache_takes_precedence_over_population_owned_cache():
    source, native = _mixed_owned_network(), _mixed_owned_network()
    for network in (source, native):
        network.cache_state()
        network._state_cache["left"]["v"].fill_(-31)
        network._state_cache["right"]["v"].fill_(-32)
        network.left._caches["_steady_state"]["v"].fill_(-20)
    functional, tensors = dn.func.make_functional_network(source)
    defaults = functional.initialization_inputs
    assert defaults.cache is not None
    assert all(value.cache is None for value in defaults.populations.values())
    initialized = functional.initialize(*tensors, defaults)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    _close(initialized, _cache_initialized_native_packet(functional, native))
    for name, voltage in (("left", -31), ("right", -32)):
        value = initialized.state["populations"][name]["integrator"]["v"]
        torch.testing.assert_close(value, torch.full_like(value, voltage))


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_owned_cache_restart_rng_recording_and_checkpoint_continuation_match_native(
    execution,
):
    source, native = _owned_stim_network(), _owned_stim_network()
    _sync_stimulus(source, native)
    options = {
        "dt": DT,
        "steps": 4,
        "initialization": "fresh",
        "execution": execution,
        "inference": execution == "while",
        "trainable": (
            f"synapses.{next(iter(source.synapses))}.weight.weights.0.param.rho",
        ),
        "callbacks": {"trace": dn.func.Recorder(("cell.v", "t"))},
        "backend": "aot_eager",
    }
    if execution == "scan":
        options["checkpoint_every"] = 2
    runner = dn.compile(source, **options)
    frozen = _clone((runner.initial_state, runner.initialization_inputs), detached=True)
    prefix, first = runner()
    repeated, again = runner()
    _close(prefix, repeated)
    _close(first["callbacks"].outputs, again["callbacks"].outputs)
    final, second = runner(
        prefix,
        parameters=first["parameters"],
        constants=first["constants"],
        callback_state=first["callbacks"].state,
    )
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
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
        weight = runner.parameters["synapses"][next(iter(source.synapses))][
            "weight.weights.0.param.rho"
        ]
        gradient = torch.autograd.grad(trace.sum(), weight)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    else:
        assert runner.execution_report().strategy == "compiled_while_loop"
    _close((runner.initial_state, runner.initialization_inputs), frozen)

    target = _owned_stim_network()
    runner.functional.commit_state_(target, final)
    checkpoint = target.state_dict_for_checkpoint()
    resumed = _owned_stim_network()
    resumed.restore_dict_from_checkpoint(checkpoint)
    continued, _ = runner.functional.prepare_and_step(
        first["parameters"],
        first["constants"],
        final,
        netstim_draw=runner.functional.sample_step_draw(final),
    )
    resumed.step()
    _close(continued, runner.functional.extract(resumed).state)


def test_population_owned_cache_defaults_are_detached_and_independent_of_native_edits():
    source = _mixed_owned_network()
    functional, tensors = dn.func.make_functional_network(source)
    defaults = functional.initialization_inputs
    saved = defaults.populations["left"].cache["v"].clone()
    source.left._caches["_steady_state"]["v"].add_(7)
    torch.testing.assert_close(defaults.populations["left"].cache["v"], saved)
    inputs = defaults._replace(
        populations={
            **defaults.populations,
            "left": defaults.populations["left"]._replace(
                cache={
                    **defaults.populations["left"].cache,
                    "v": saved.new_full(saved.shape, -29),
                }
            ),
        }
    )
    initialized = functional.initialize(*tensors, inputs)
    value = initialized.state["populations"]["left"]["integrator"]["v"]
    torch.testing.assert_close(value, torch.full_like(value, -29))
    assert not value.requires_grad
    torch.testing.assert_close(defaults.populations["left"].cache["v"], saved)


def test_mixed_population_cache_initialization_compiles_as_one_full_graph():
    functional, tensors = dn.func.make_functional_network(_mixed_owned_network())
    execute = torch.compile(functional.initialize, backend="aot_eager", fullgraph=True)
    defaults = functional.initialization_inputs
    _close(execute(*tensors, defaults), functional.initialize(*tensors, defaults))


@pytest.mark.parametrize("lanes", (0, 2))
def test_mixed_population_cache_initialization_maps_saved_and_fresh_voltages(lanes):
    functional, tensors = dn.func.make_functional_network(_mixed_owned_network())
    defaults = functional.initialization_inputs

    def initialize(saved, fresh):
        inputs = defaults._replace(
            populations={
                "left": defaults.populations["left"]._replace(
                    cache={**defaults.populations["left"].cache, "v": saved}
                ),
                "right": defaults.populations["right"]._replace(v_init=fresh),
            }
        )
        state = functional.initialize(*tensors, inputs).state["populations"]
        return state["left"]["integrator"]["v"], state["right"]["integrator"]["v"]

    shape = defaults.populations["left"].cache["v"].shape
    saved = torch.full((lanes, *shape), -31.0, dtype=DTYPE)
    fresh = torch.full((lanes, *shape), -42.0, dtype=DTYPE)
    actual_saved, actual_fresh = torch.vmap(initialize)(saved, fresh)
    torch.testing.assert_close(actual_saved, saved)
    torch.testing.assert_close(actual_fresh, fresh)
