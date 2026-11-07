"""Native and functional initialization share explicit stochastic boundaries."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.distributions import Normal
from dendra.models.mechanisms import Mechanism
from dendra.models.mod import expsyn, pas
from dendra.models.parametric import PositiveParam
from dendra.models.random_parameters import register_random_distribution
from dendra.models.stim.waveform._poisson import (
    _poisson_waveforms,
    poisson_stream_state,
)
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_native_handoff import _capture, _unchanged
from tests.func.test_netstim_parameter_initialization import _noise_cache_network
from tests.func.test_network_connection_sampling import (
    _check_native,
    _materialized_loss,
    _network_with_population_source_phase,
)
from tests.func.test_network_initialization_rng import _shared_noise_network
from tests.func.test_stimulation_rng_sharing import (
    DT,
    _functional_gradient_leaves,
    _gradient_leaves,
    _lower,
)
from tests.func.test_stimulation_rng_sharing import _network as _shared_network
from tests.func.test_stimulation_rng_sharing import (
    _one_stream,
)
from tests.func.test_stimulation_rng_sharing import _population as _shared_population
from tests.func.test_stimulation_rng_sharing import (
    _SharedStochasticLeak,
)
from tests.func.test_stochastic_initialization_ordering import (
    _population as _ordered_population,
)
from tests.func.test_stochastic_initialization_ordering import (
    _RandomPopulation,
    _ReseededPopulation,
    _StickyPopulation,
)
from tests.test_native_runtime_draw_tapes import _loss, _state_loss
from tests.test_native_runtime_draws import _environment as _environment
from tests.test_native_runtime_draws import _forbid_owned_sampling
from tests.test_shared_runtime_draw_sampling import _forbid_lowering

pytestmark = pytest.mark.cpu


def _make_shared(network, mode="resample"):
    options = {"reseed": mode == "reseed", "retained": mode == "retained"}
    if network:
        return _shared_network(**options)[0]
    return _shared_population(**options)[0]


def _generators(model):
    generators = {
        id(waveform.generator): waveform.generator
        for waveform in _poisson_waveforms(model).values()
    }
    if isinstance(model, dn.Network) and model.netstim is not None:
        generators[id(model.netstim._rng)] = model.netstim._rng
    return tuple(generators.values())


def _initialize(model, draws, **options):
    result = (
        model.initialize(model.dt, draws=draws, **options)
        if isinstance(model, dn.Network)
        else model.initialize(draws=draws, **options)
    )
    assert result is model
    if not isinstance(model, dn.Network):
        model.integrator._initialize(model, model.t.new_tensor(DT), force=True)
    return result


def _state_close(actual, expected):
    if isinstance(expected, Mapping):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            # Pure initialization retains its pre-write parameter frame. Native
            # extraction reconnects that provenance to current source bindings.
            if isinstance(key, str) and key.startswith("random.initialization_frame."):
                continue
            _state_close(actual[key], value)
    else:
        torch.testing.assert_close(actual, expected)


def _slice(packet, start, stop):
    return pytree.tree_map_only(torch.Tensor, lambda value: value[start:stop], packet)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("mode", ("resample", "retained", "reseed"))
def test_native_capture_matches_functional_packets_without_lowering_or_source_changes(
    network, mode, monkeypatch
):
    model = _make_shared(network, mode)
    functional, tensors = _lower(model)
    expected = dn.func.sample_initialization_draws(functional, tensors)
    saved = _capture(model)
    generators = _generators(model)
    states = tuple(generator.get_state().clone() for generator in generators)
    with monkeypatch.context() as guard:
        _forbid_lowering(guard)
        _forbid_owned_sampling(guard, model, *generators)
        with torch.inference_mode():
            actual = dn.func.sample_initialization_draws(model)
        repeated = dn.func.sample_initialization_draws(model)
    _assert_packet_equal(actual, expected)
    _assert_packet_equal(repeated, expected)
    _unchanged(model, saved)
    for generator, state in zip(generators, states, strict=True):
        _close(generator.get_state(), state)
    for leaf in pytree.tree_leaves(actual):
        if torch.is_tensor(leaf):
            assert not leaf.requires_grad and leaf.grad_fn is None
            assert not leaf.is_inference()
            assert isinstance(leaf._version, int)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_native_replay_preserves_shared_rng_gradients_and_checkpoint_suffix(
    network, monkeypatch
):
    source, native = _make_shared(network), _make_shared(network)
    native.restore_dict_from_checkpoint(_snapshot(source.state_dict_for_checkpoint()))
    functional, tensors = _lower(source)
    draws = dn.func.sample_initialization_draws(source)
    frozen = _snapshot(draws)
    initial = dn.func.initialize(functional, tensors, draws=draws)
    parameters = dict(native.named_parameters(remove_duplicate=False))
    with monkeypatch.context() as guard:
        _forbid_owned_sampling(guard, native, *_generators(native))
        _initialize(native, draws)
    current = dict(native.named_parameters(remove_duplicate=False))
    assert current.keys() == parameters.keys()
    assert all(current[name] is value for name, value in parameters.items())
    _state_close(functional.extract(native).state, initial.state)
    _one_stream(initial.state)
    _assert_packet_equal(draws, frozen)

    runtime = dn.func.sample_runtime_draws(functional, initial, steps=4)
    final, _ = dn.func.run(functional, initial, steps=4, draws=runtime)
    options = {} if network else {"dt": DT}
    native.run(steps=2, draws=_slice(runtime, 0, 2), **options)
    checkpoint = _snapshot(native.state_dict_for_checkpoint())
    native.longrun_checkpointed(
        steps=2,
        chunklength=1,
        safe_checkpoint=True,
        draws=_slice(runtime, 2, 4),
        **options,
    )
    _state_close(functional.extract(native).state, final)
    actual = torch.autograd.grad(_loss(native), _gradient_leaves(native))
    expected = torch.autograd.grad(
        _state_loss(final, network), _functional_gradient_leaves(initial)
    )
    _close(actual, expected)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in actual)

    receiver = _make_shared(network)
    receiver.restore_dict_from_checkpoint(checkpoint)
    suffix = dn.func.sample_runtime_draws(receiver, steps=2, **options)
    _assert_packet_equal(suffix, _slice(runtime, 2, 4))
    receiver.longrun(steps=2, chunklength=1, draws=suffix, **options)
    _state_close(functional.extract(receiver).state, final)


@pytest.mark.parametrize(
    "network,cache", ((False, "population"), (True, "population"), (True, "network"))
)
@pytest.mark.parametrize("ignore", (False, True))
def test_native_cache_capture_skips_unused_draws_and_preserves_rng_load_policy(
    network, cache, ignore
):
    model = _make_shared(network, "retained")
    cells = tuple(model.populations.values()) if network else (model,)
    for cell in cells:
        cell.mech.probe.stream.ignore_rng_on_load(ignore)
    if cache == "network":
        model.cache_state()
    else:
        for cell in cells:
            cell.cache("_steady_state")
    for _ in range(2):
        model.step() if network else model.step(DT)
    functional, tensors = _lower(model)
    expected_draws = dn.func.sample_initialization_draws(functional, tensors)
    saved = _capture(model)
    draws = dn.func.sample_initialization_draws(model)
    _assert_packet_equal(draws, expected_draws)
    assert (
        draws.random
        == draws.stimulation
        == (dict.fromkeys(model.populations) if network else None)
    )
    _unchanged(model, saved)
    expected = dn.func.initialize(functional, tensors, draws=draws)
    _initialize(model, draws)
    _state_close(functional.extract(model).state, expected.state)


@pytest.mark.parametrize(
    "population_type", (_RandomPopulation, _StickyPopulation, _ReseededPopulation)
)
def test_native_ordered_capture_preserves_root_child_material_sampling_and_aliases(
    population_type,
):
    model = _ordered_population(population_type)
    functional, tensors = dn.func.make_functional(model, dt=0.125)
    saved = _capture(model)
    draws = dn.func.sample_initialization_draws(model)
    _assert_packet_equal(
        draws, dn.func.sample_initialization_draws(functional, tensors)
    )
    _unchanged(model, saved)
    expected = dn.func.initialize(functional, tensors, draws=draws)
    assert model.initialize(draws=draws) is model
    model.integrator._initialize(model, model.t.new_tensor(0.125), force=True)
    _state_close(functional.extract(model).state, expected.state)
    assert model.trial_mu_param is model.mech.leak.g_mu_param
    assert model.trial_mu_param is model.mech.leak.DE._RandomGate.tau_mu_param
    assert (
        model.trial_mu_param is model.mech.material_processes["transfer"].rate_mu_param
    )


@pytest.mark.parametrize("initial_noise,increment", ((0.0, 0.6), (0.6, -0.6)))
def test_native_netstim_capture_observes_post_population_noise_sources(
    initial_noise, increment
):
    source, _ = _shared_noise_network(
        seed=812, initial_noise=initial_noise, increment=increment
    )
    model, _ = _shared_noise_network(
        seed=812, initial_noise=initial_noise, increment=increment
    )
    model.restore_dict_from_checkpoint(_snapshot(source.state_dict_for_checkpoint()))
    functional, tensors = dn.func.make_functional_network(source)
    saved = _capture(model)
    draws = dn.func.sample_initialization_draws(model)
    _assert_packet_equal(
        draws, dn.func.sample_initialization_draws(functional, tensors)
    )
    assert bool(draws.netstim.exp_rand.any()) is bool(initial_noise + increment)
    _unchanged(model, saved)
    expected = dn.func.initialize(functional, tensors, draws=draws)
    _initialize(model, draws)
    _state_close(functional.extract(model).state, expected.state)
    _close(model.netstim._rng.get_state(), draws.netstim.rng_state)


@pytest.mark.parametrize("noise", (0.0, 0.6))
def test_native_netstim_capture_observes_cache_restored_sources(noise):
    source = _noise_cache_network("population", noise)
    model = _noise_cache_network("population", noise)
    model.restore_dict_from_checkpoint(_snapshot(source.state_dict_for_checkpoint()))
    functional, tensors = dn.func.make_functional_network(source)
    draws = dn.func.sample_initialization_draws(model)
    _assert_packet_equal(
        draws, dn.func.sample_initialization_draws(functional, tensors)
    )
    assert draws.random == draws.stimulation == {"cell": None}
    assert bool(draws.netstim.exp_rand.any()) is bool(noise)
    expected = dn.func.initialize(functional, tensors, draws=draws)
    _initialize(model, draws)
    _state_close(functional.extract(model).state, expected.state)


@pytest.mark.parametrize("mode", ("transform", "population_cache", "network_cache"))
def test_native_connection_capture_uses_ordered_sources_and_preserves_gradients(mode):
    source = _network_with_population_source_phase(mode)
    native = _network_with_population_source_phase(mode)
    native.restore_dict_from_checkpoint(_snapshot(source.state_dict_for_checkpoint()))
    functional, tensors = dn.func.make_functional_network(source)
    saved = _capture(source)
    draws = dn.func.sample_initialization_draws(source)
    _assert_packet_equal(
        draws, dn.func.sample_initialization_draws(functional, tensors)
    )
    _unchanged(source, saved)
    expected = dn.func.initialize(functional, tensors, draws=draws)
    global_rng = torch.get_rng_state().clone()
    _initialize(native, draws)
    _close(torch.get_rng_state(), global_rng)
    _check_native(functional, expected, native)
    _state_close(functional.extract(native).state, expected.state)
    _close(native._functional_connection_sampling_rng, draws.connections.rng)
    reference = expected.parameters["populations"]["cell"]["cm_param.rho"]
    actual_gradient = torch.autograd.grad(
        sum(
            (index + 1) * connection.weight.w.square().sum()
            + (index + 0.5) * connection.delay_ms.w.square().sum()
            for index, name in enumerate(("synapses", "continuous_synapses"))
            for connection in getattr(native, name).values()
        ),
        native.cell.cm_param.rho,
    )[0]
    expected_gradient = torch.autograd.grad(
        _materialized_loss(functional, expected), reference
    )[0]
    _close(actual_gradient, expected_gradient)
    assert actual_gradient.isfinite().all() and actual_gradient.abs().sum() > 0


def test_native_connection_replay_preserves_disabled_resampling_flags():
    source = _network_with_population_source_phase("transform")
    model = _network_with_population_source_phase("transform")
    model.restore_dict_from_checkpoint(_snapshot(source.state_dict_for_checkpoint()))
    functional, tensors = dn.func.make_functional_network(source)
    initialization = functional.initialization_inputs._replace(
        reinit_weights=False, reinit_delays=False
    )
    draws = dn.func.sample_initialization_draws(
        functional, tensors, initialization=initialization
    )
    expected = dn.func.initialize(
        functional, tensors, initialization=initialization, draws=draws
    )
    _initialize(model, draws, reinit_weights=False, reinit_delays=False)
    _state_close(functional.extract(model).state, expected.state)
    _check_native(functional, expected, model)


class _RepairPositive(torch.nn.Module):
    def forward(self, value):
        return (value.abs(),)


class _PositiveOnlyWeight(torch.nn.Module):
    def __init__(self, raw):
        super().__init__()
        self.raw = raw

    def forward(self):
        if bool(torch.any(self.raw <= 0)):
            raise RuntimeError("Connection getter ran before the Population repair")
        return 0.003 * self.raw


def _repair_noise_draw(rng, shape, *, device, dtype):
    return rng.randn(shape, device=device, dtype=dtype)


def _repair_noise_transform(parameters, draw):
    if bool(torch.any(parameters["sigma"] <= 0)):
        raise ValueError("NOISE transform ran before the Population repair")
    return parameters["mu"] + parameters["sigma"] * draw


register_random_distribution(
    "shared_initialization_positive_noise",
    parameter_defaults={"mu": 0.0, "sigma": 0.1},
    parameter_constraints={"mu": "real", "sigma": "real"},
    draw_sampler=_repair_noise_draw,
    transform=_repair_noise_transform,
)


class _NoiseRepairPopulation(dn.SingleCompartment):
    dn.SingleCompartment.RANGE(repair_sigma=0.1)


class _NoiseRepairLeak(Mechanism):
    Mechanism.RANGE(g=0.001, e=-65.0)
    Mechanism.GLOBALNOISE(
        "eta",
        distribution="shared_initialization_positive_noise",
        mu=0.0,
        sigma=0.1,
        seed=919,
    )
    Mechanism.ASSIGNED("i_inj")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def inject(self, waveform, *, index=None, model_shape=None, model=None, **kwargs):
        del kwargs
        return self.register_waveform_injection(
            waveform, index=index, model_shape=model_shape, model=model
        )

    def assigned_values(self, v, values):
        del values
        return {"i_inj": self.injection_values(v, t=self.t)}

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e + self.eta.clone() + 0.001 * self.i_inj


def _noise_repair_population(*, poisson=False):
    model = _NoiseRepairPopulation(
        N=1, C=1, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    model.insert(_NoiseRepairLeak.rename("leak"))
    model.train()
    model.initialize()
    model.mech.leak.eta_sigma_param = model.repair_sigma_param
    model.register_pre_initialize_transform(
        "repair",
        _RepairPositive(),
        reads=("parameters.repair_sigma_param",),
        writes=("parameters.repair_sigma_param",),
    )
    if poisson:
        waveform = dn.mono_rect(amp=0.2, pw=0.045).poisson(
            interval=0.055,
            n=5,
            start=-0.035,
            noise=0.7,
            randomize_every_call=True,
            generator=torch.Generator().manual_seed(941),
        )
        model[:, 0].inject(waveform)
    model.initialize()
    model.integrator._initialize(model, model.t.new_tensor(DT), force=True)
    return model


@pytest.mark.parametrize("poisson", (False, True), ids=("noise", "noise_poisson"))
def test_native_noise_metadata_preserves_post_transform_parameter_domain(poisson):
    source, native = (
        _noise_repair_population(poisson=poisson),
        _noise_repair_population(poisson=poisson),
    )
    functional, tensors = _lower(source)
    shared = tensors.parameters["repair_sigma_param"]
    with torch.no_grad():
        source.repair_sigma_param.fill_(-0.5)
        native.repair_sigma_param.fill_(-0.5)
        shared.fill_(-0.5)
    saved = _capture(source)
    draws = dn.func.sample_initialization_draws(source)
    _assert_packet_equal(
        draws, dn.func.sample_initialization_draws(functional, tensors)
    )
    _unchanged(source, saved)
    initial = dn.func.initialize(functional, tensors, draws=draws)
    _initialize(native, draws)
    _state_close(functional.extract(native).state, initial.state)
    assert native.repair_sigma_param is native.mech.leak.eta_sigma_param
    _close(native.repair_sigma_param, torch.full_like(native.repair_sigma_param, 0.5))


def _repair_connection_network():
    cell = dn.SingleCompartment(
        N=1,
        C=1,
        cm=PositiveParam(torch.tensor(1.0, dtype=torch.float64)),
        integrator=dn.bwd_euler_sc(imem=False),
    )
    cell.insert(pas)
    cell.insert(expsyn.rename("syn"), tau=1.2, e=0.0)
    cell.train()
    cell.initialize()
    model = dn.Network(
        {"cell": cell},
        netcon_delay_backend="dense",
        netcon_train_backend="dense",
    )
    model.connect_one_to_one(
        cell[:],
        cell[:],
        cell.mech.syn,
        weight=_PositiveOnlyWeight(cell.cm_param.rho),
        delay=DT,
        allow_autapses=True,
    )
    model.train()
    model.initialize(DT)
    cell.register_pre_initialize_transform(
        "repair",
        _RepairPositive(),
        reads=("parameters.cm_param.rho",),
        writes=("parameters.cm_param.rho",),
    )
    model.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=False,
        diff_spiking=False,
        train_delay_backend="dense",
    )
    return model


def test_native_connection_getters_run_after_population_source_repair():
    source, native = _repair_connection_network(), _repair_connection_network()
    functional, tensors = _lower(source)
    with torch.no_grad():
        source.cell.cm_param.rho.fill_(-0.5)
        native.cell.cm_param.rho.fill_(-0.5)
        tensors.parameters["populations"]["cell"]["cm_param.rho"].fill_(-0.5)
    saved = _capture(source)
    draws = dn.func.sample_initialization_draws(source)
    _assert_packet_equal(
        draws, dn.func.sample_initialization_draws(functional, tensors)
    )
    _unchanged(source, saved)
    expected = dn.func.initialize(functional, tensors, draws=draws)
    _initialize(native, draws)
    _state_close(functional.extract(native).state, expected.state)
    _check_native(functional, expected, native)
    assert native.cell.cm_param.rho.gt(0).all()


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("kind", ("mapping", "runtime", "random_family", "missing"))
def test_native_replay_rejects_malformed_draws_without_sampling(
    network, kind, monkeypatch
):
    model = _make_shared(network)
    packet = dn.func.sample_initialization_draws(model)
    invalid = {
        "mapping": {},
        "runtime": dn.func.RuntimeDraws({}, {}),
        "random_family": packet._replace(random=object()),
        "missing": packet._replace(random=None),
    }[kind]
    saved = _capture(model)
    with monkeypatch.context() as guard:
        _forbid_owned_sampling(guard, model, *_generators(model))
        with pytest.raises((TypeError, ValueError, dn.func.FunctionalizationError)):
            _initialize(model, invalid)
    _unchanged(model, saved)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("keyword", ("initialization", "parameters", "constants"))
def test_native_capture_rejects_functional_overrides_before_initialization(
    network, keyword
):
    model = _make_shared(network)
    saved = _capture(model)
    with pytest.raises(TypeError):
        dn.func.sample_initialization_draws(model, **{keyword: {}})
    _unchanged(model, saved)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_native_capture_rejects_a_positional_tensor_bundle(network):
    model = _make_shared(network)
    saved = _capture(model)
    with pytest.raises(TypeError):
        dn.func.sample_initialization_draws(model, object())
    _unchanged(model, saved)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_native_capture_rejects_implicit_build_and_initialization(network):
    cell = dn.SingleCompartment(N=1, C=1, integrator=dn.bwd_euler_sc(imem=False))
    cell.insert(pas)
    model = dn.Network({"cell": cell}) if network else cell
    built = cell.is_built
    saved = _snapshot(model.state_dict())
    with pytest.raises(dn.func.FunctionalizationError, match="[Ii]nitializ"):
        dn.func.sample_initialization_draws(model)
    _assert_packet_equal(dict(model.state_dict()), saved)
    assert not cell.initialized and cell.is_built is built


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_native_replay_rejects_force_rebuild_before_mutating_sources(network):
    model = _make_shared(network)
    draws = dn.func.sample_initialization_draws(model)
    saved = _capture(model)
    with pytest.raises((ValueError, dn.func.FunctionalizationError), match="rebuild"):
        _initialize(model, draws, force_rebuild=True)
    _unchanged(model, saved)


def test_native_network_capture_preserves_a_physical_waveform_shared_by_populations():
    model, _ = _shared_network()
    waveform = model.left.mech.probe.injected_waveforms[0]
    model.right.delete_injections()
    model.right[:, 1].inject(waveform)
    model.initialize(DT)
    functional, tensors = _lower(model)
    saved = _capture(model)
    draws = dn.func.sample_initialization_draws(model)
    _assert_packet_equal(
        draws, dn.func.sample_initialization_draws(functional, tensors)
    )
    _unchanged(model, saved)
    assert model.left.mech.probe.injected_waveforms[0] is waveform
    assert model.right.mech.probe.injected_waveforms[0] is waveform
    expected = dn.func.initialize(functional, tensors, draws=draws)
    _initialize(model, draws)
    _state_close(functional.extract(model).state, expected.state)


def test_native_capture_omits_solver_bank_poisson_draws_from_initialization():
    model, _, pulse = _shared_population("intra")
    functional, tensors = _lower(model)
    saved = _capture(model)
    draws = dn.func.sample_initialization_draws(model)
    assert draws.stimulation is None
    _assert_packet_equal(
        draws, dn.func.sample_initialization_draws(functional, tensors)
    )
    _unchanged(model, saved)
    expected = dn.func.initialize(functional, tensors, draws=draws)
    _initialize(model, draws)
    _state_close(functional.extract(model).state, expected.state)
    _close(pulse.generator.get_state(), _one_stream(expected.state))


def _default_stream_network():
    cell = dn.SingleCompartment(
        N=1, C=1, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    cell.insert(_SharedStochasticLeak.rename("probe"))
    cell.insert(expsyn.rename("syn"), tau=1.2, e=0.0)
    cell.train()
    cell.initialize()
    owner = cell.mech.probe
    owner.DE._SharedStochasticGate.stream = owner.stream
    owner.DE._SharedStochasticGate.x_dW_rng = owner.stream
    pulse = dn.mono_rect(amp=0.2, pw=0.045).poisson(
        interval=0.055,
        n=5,
        start=-0.035,
        noise=0.7,
        randomize_every_call=True,
        generator=torch.default_generator,
    )
    cell[:, 0].inject(pulse + pulse)
    model = dn.Network(
        {"cell": cell},
        netcon_delay_backend="dense",
        netcon_train_backend="dense",
    )
    model.connect_one_to_one(
        cell[:],
        cell[:],
        cell.mech.syn,
        weight=Normal(mean=0.003, std=0.0002),
        delay=DT,
        allow_autapses=True,
    )
    model.train()
    model.initialize(DT)
    model.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=False,
        diff_spiking=False,
        train_delay_backend="dense",
    )
    return model, pulse


def test_native_default_torch_poisson_and_connections_share_initialization_and_runtime_rng():
    source, _ = _default_stream_network()
    functional, tensors = _lower(source)
    saved = _capture(source)
    draws = dn.func.sample_initialization_draws(source)
    _assert_packet_equal(
        draws, dn.func.sample_initialization_draws(functional, tensors)
    )
    _unchanged(source, saved)
    initial = dn.func.initialize(functional, tensors, draws=draws)
    runtime = dn.func.sample_runtime_draws(functional, initial, steps=2)
    final, _ = dn.func.run(functional, initial, steps=2, draws=runtime)
    original = _snapshot(source.state_dict_for_checkpoint())
    model, pulse = _default_stream_network()
    model.restore_dict_from_checkpoint(original)
    ambient = torch.get_rng_state().clone()
    _initialize(model, draws)
    _close(torch.get_rng_state(), ambient)
    _state_close(functional.extract(model).state, initial.state)
    _close(model._functional_connection_sampling_rng, draws.connections.rng)
    _close(poisson_stream_state((pulse,)), draws.connections.rng["cpu"])

    checkpoint = _snapshot(model.state_dict_for_checkpoint())
    receiver, _ = _default_stream_network()
    receiver.restore_dict_from_checkpoint(checkpoint)
    ambient = torch.get_rng_state().clone()
    _assert_packet_equal(dn.func.sample_runtime_draws(model, steps=2), runtime)
    _close(torch.get_rng_state(), ambient)
    model.run(steps=2)
    _state_close(functional.extract(model).state, final)
    receiver.run(steps=2, draws=runtime)
    _state_close(functional.extract(receiver).state, final)


@pytest.mark.parametrize("change", ("dt", "device_signature"))
def test_native_network_replay_rejects_rebuild_triggers_before_mutating_sources(change):
    model = _make_shared(True)
    draws = dn.func.sample_initialization_draws(model)
    if change == "device_signature":
        # Native build invalidates its prepared plan when the cached device
        # signature differs, regardless of whether the tensors already moved.
        model._device_sig = ("stale-device-signature",)
    saved = _capture(model)
    timestep = 2 * DT if change == "dt" else DT
    with pytest.raises((ValueError, dn.func.FunctionalizationError), match="rebuild"):
        model.initialize(timestep, draws=draws)
    _unchanged(model, saved)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_native_initialization_capture_and_replay_with_purity_diagnostics(network):
    source, native = _make_shared(network), _make_shared(network)
    native.restore_dict_from_checkpoint(_snapshot(source.state_dict_for_checkpoint()))
    functional, tensors = _lower(source)
    saved = _capture(source)
    with dn.ctx(CHECK_PURITY=True):
        draws = dn.func.sample_initialization_draws(source)
        _initialize(native, draws)
    expected = dn.func.initialize(functional, tensors, draws=draws)
    _state_close(functional.extract(native).state, expected.state)
    _unchanged(source, saved)


@pytest.mark.parametrize(
    "inference_owned,inference", ((False, False), (False, True), (True, True))
)
def test_native_connection_replay_preserves_inference_mode_with_purity_diagnostics(
    inference_owned, inference
):
    source = _network_with_population_source_phase("transform")
    with torch.inference_mode(inference_owned):
        native = _network_with_population_source_phase("transform")
        native.restore_dict_from_checkpoint(
            _snapshot(source.state_dict_for_checkpoint())
        )
    functional, tensors = dn.func.make_functional_network(source)
    draws = dn.func.sample_initialization_draws(source)
    expected = dn.func.initialize(functional, tensors, draws=draws)
    with dn.ctx(CHECK_PURITY=True), torch.inference_mode(inference):
        _initialize(native, draws)
    _state_close(functional.extract(native).state, expected.state)
    _check_native(functional, expected, native)
    assert all(
        connection.weight.w.is_inference() is inference
        for connection in native.synapses.values()
    )


class _CountingInitialization(torch.nn.Module):
    calls = 0

    def forward(self, value):
        _CountingInitialization.calls += 1
        return (value,)


@pytest.mark.parametrize("enabled", (False, True))
def test_native_capture_class_mutation_diagnostics_follow_the_optional_flag(enabled):
    model = dn.SingleCompartment(N=1, C=1, integrator=dn.bwd_euler_sc(imem=False))
    model.insert(pas)
    model.initialize()
    parameter = next(
        name for name, value in model.named_parameters() if name.endswith("g_param")
    )
    model.register_pre_initialize_transform(
        "counted",
        _CountingInitialization(),
        reads=(f"parameters.{parameter}",),
        writes=(f"parameters.{parameter}",),
    )
    saved = _capture(model)
    _CountingInitialization.calls = 0
    try:
        with dn.ctx(CHECK_PURITY=enabled):
            if enabled:
                with pytest.raises(
                    dn.func.FunctionalizationError, match="(?i)class|mutat|pure"
                ):
                    dn.func.sample_initialization_draws(model)
                assert _CountingInitialization.calls == 0
            else:
                assert (
                    dn.func.sample_initialization_draws(model)
                    == dn.func.InitializationDraws()
                )
                assert _CountingInitialization.calls == 1
        _unchanged(model, saved)
    finally:
        _CountingInitialization.calls = 0


class _MPSLeak(Mechanism):
    Mechanism.RANGERAND("g", distribution="lognormal", mu=-7.0, sigma=0.1, seed=719)
    Mechanism.GLOBALNOISE("eta", mu=0.0, sigma=0.00001, seed=723)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, -65.0 * self.g + self.eta.clone()


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_native_initialization_replay_keeps_gradients_and_rng_continuation():
    with dn.ctx(DEVICE="mps", DTYPE=torch.float32):
        model = dn.SingleCompartment(
            N=1, C=1, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        model.insert(_MPSLeak.rename("leak"))
        model.train()
        model.initialize()
        model.integrator._initialize(model, model.t.new_tensor(DT), force=True)
        functional, tensors = dn.func.make_functional(model, dt=DT)
        saved = _capture(model)
        global_rng = torch.mps.get_rng_state().clone()
        with torch.inference_mode():
            draws = dn.func.sample_initialization_draws(model)
        _unchanged(model, saved)
        _close(torch.mps.get_rng_state(), global_rng)
        expected = dn.func.initialize(functional, tensors, draws=draws)
        _initialize(model, draws)
        runtime = dn.func.sample_runtime_draws(functional, expected, steps=2)
        final, _ = dn.func.run(functional, expected, steps=2, draws=runtime)
        model.longrun_checkpointed(
            steps=2, dt=DT, chunklength=1, safe_checkpoint=True, draws=runtime
        )
        _state_close(functional.extract(model).state, final)
        actual_gradient = torch.autograd.grad(
            model.v.square().sum(), model.mech.leak.g_mu_param
        )[0]
        key = next(name for name in tensors.parameters if name.endswith("g_mu_param"))
        expected_gradient = torch.autograd.grad(
            final["integrator"]["v"].square().sum(), tensors.parameters[key]
        )[0]
        _close(actual_gradient, expected_gradient)
        assert actual_gradient.device.type == "mps" and actual_gradient.abs().sum() > 0
