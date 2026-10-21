"""Fresh Network initialization retains native clocks, RNG, and calendars."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import expsyn, pas
from tests.func.test_network import DT, DTYPE, _clone, _close, _stim_network
from tests.func.test_population_initialization_transforms import _Add

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _models(*, seed, noise, calendar=False):
    model, native = _stim_network(), _stim_network()
    for network in (model, native):
        network.netstim.seed = seed
        network.netstim.noise.detach().fill_(noise)
        if calendar:
            network.netstim.schedule([0, 0, 0], [0.0, DT, 2.0])
    for _ in range(4):
        model.step()
    model._set_duration_remainder(0.3 * DT)
    native.restore_dict_from_checkpoint(_clone(model.state_dict_for_checkpoint()))
    return model, native


def _fresh(functional, tensors, *, t=0.0, inputs=None, draw=None):
    inputs = functional.initialization_inputs if inputs is None else inputs
    inputs = inputs._replace(t=inputs.t.new_tensor(t))
    draw = (
        functional.sample_initialization_draw(tensors.parameters, tensors.state)
        if draw is None
        else draw
    )
    return functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        inputs,
        netstim_draw=draw,
    )


@pytest.mark.parametrize("seed", (None, 812))
@pytest.mark.parametrize("noise", (0.0, 0.6))
@pytest.mark.parametrize("t", (-0.2, 0.0, 2.75))
def test_fresh_network_native_clock_rng_reset_and_first_step_parity(seed, noise, t):
    model, native = _models(seed=seed, noise=noise)
    functional, tensors = dn.func.make_functional_network(model)
    frozen = _clone(tensors)
    native_before = _clone(model.state_dict_for_checkpoint())
    draw = functional.sample_initialization_draw(tensors.parameters, tensors.state)
    _close(
        draw, functional.sample_initialization_draw(tensors.parameters, tensors.state)
    )
    initialized = _fresh(functional, tensors, t=t, draw=draw)
    _close(tensors, frozen)
    _close(model.state_dict_for_checkpoint(), native_before)
    native.initialize(DT, reinit_weights=False, reinit_delays=False, t=t)
    _close(initialized.state, functional.extract(native).state)
    assert initialized.state["clock"]["t"].item() == t
    assert initialized.state["clock"]["origin"].item() == t
    assert initialized.state["clock"]["step"].item() == 0
    assert initialized.state["clock"]["duration_remainder"].item() == 0
    assert not initialized.state["clock"]["t"].requires_grad
    assert initialized.state["populations"]["cell"]["clock"]["t"].item() == t
    if seed is None and noise == 0:
        torch.testing.assert_close(
            initialized.state["netstim"]["rng_state"],
            tensors.state["netstim"]["rng_state"],
        )
    step_draw = functional.sample_step_draw(initialized.state)
    actual, _ = functional.prepare_and_step(
        initialized.parameters,
        initialized.constants,
        initialized.state,
        netstim_draw=step_draw,
    )
    native.step()
    _close(actual, functional.extract(native).state)


@pytest.mark.parametrize("seed", (None, 812))
def test_fresh_network_preserves_consumed_calendar_and_seeder_and_clears_override(seed):
    model, native = _models(seed=seed, noise=0.6, calendar=True)
    model.netstim.heap_head_override.fill_(99)
    model.netstim.heap_override_active.fill_(True)
    native.restore_dict_from_checkpoint(_clone(model.state_dict_for_checkpoint()))
    functional, tensors = dn.func.make_functional_network(model)
    before = _clone(tensors.state["netstim"])
    assert before["heap_cursor"].item() == 2
    initialized = _fresh(functional, tensors, t=0.75)
    native.initialize(DT, reinit_weights=False, reinit_delays=False, t=0.75)
    _close(initialized.state, functional.extract(native).state)
    reset = initialized.state["netstim"]
    torch.testing.assert_close(reset["heap_cursor"], before["heap_cursor"])
    torch.testing.assert_close(reset["seeder_state"], before["seeder_state"])
    assert not reset["heap_override_active"].any()
    assert torch.isinf(reset["heap_head_override"]).all()
    assert reset["spike_counts"].eq(0).all()
    assert torch.isneginf(reset["t_last"]).all()
    _close(tensors.state["netstim"], before)


def test_network_initialization_random_kick_is_detached_but_renewal_keeps_gradients():
    model, _ = _models(seed=812, noise=0.6)
    functional, tensors = dn.func.make_functional_network(model)
    parameters = tensors.parameters["netstim"]["parameters"]
    targets = tuple(
        parameters[name].requires_grad_() for name in ("interval", "noise", "start")
    )
    draw = functional.sample_initialization_draw(tensors.parameters, tensors.state)
    draw = draw._replace(exp_rand=draw.exp_rand.requires_grad_())
    initialized = _fresh(functional, tensors, t=10.0, draw=draw)
    assert not initialized.state["netstim"]["next_stoch_time"].requires_grad
    step_draw = functional.sample_step_draw(initialized.state)
    actual, _ = functional.prepare_and_step(
        initialized.parameters,
        initialized.constants,
        initialized.state,
        netstim_draw=step_draw,
    )
    gradient = torch.autograd.grad(
        actual["netstim"]["next_stoch_time"].sum(),
        (*targets, draw.exp_rand),
        allow_unused=True,
    )
    assert gradient[0].abs().sum() > 0
    assert gradient[1].abs().sum() > 0
    assert gradient[2] is None
    assert gradient[3] is None
    torch.testing.assert_close(
        gradient[0], 1 - parameters["noise"] + parameters["noise"] * step_draw.exp_rand
    )
    torch.testing.assert_close(
        gradient[1], parameters["interval"] * (step_draw.exp_rand - 1)
    )


def test_fresh_network_checkpoint_and_explicit_continuation_keep_rng_and_gradients():
    model, _ = _models(seed=None, noise=0.6)
    functional, tensors = dn.func.make_functional_network(model)
    parameters = tensors.parameters["netstim"]["parameters"]
    targets = tuple(parameters[name].requires_grad_() for name in ("interval", "noise"))
    initialized = _fresh(functional, tensors)
    expected, _ = functional.run(
        initialized.parameters, initialized.constants, initialized.state, steps=8
    )
    kernel = functional.compile_rollout_chunk(3, disable=True)
    prefix, _ = functional.run(
        initialized.parameters,
        initialized.constants,
        initialized.state,
        steps=3,
        checkpoint_every=2,
        compiled_chunk=kernel,
    )
    actual, _ = functional.run(
        initialized.parameters,
        initialized.constants,
        prefix,
        steps=5,
        checkpoint_every=2,
        compiled_chunk=kernel,
    )
    _close(actual, expected)
    actual_gradient = torch.autograd.grad(
        actual["netstim"]["next_stoch_time"].sum(), targets, retain_graph=True
    )
    expected_gradient = torch.autograd.grad(
        expected["netstim"]["next_stoch_time"].sum(), targets
    )
    _close(actual_gradient, expected_gradient)
    assert all(value.abs().sum() > 0 for value in actual_gradient)


@pytest.mark.parametrize("cache", ("_state_cache", "_syn_cache"))
@pytest.mark.parametrize("when", ("before", "after"))
def test_fresh_network_rejects_native_state_cache_restoration(cache, when):
    model, _ = _models(seed=812, noise=0.6)
    if when == "before":
        getattr(model, cache)["cell"] = {}
    functional, tensors = dn.func.make_functional_network(model)
    if when == "after":
        getattr(model, cache)["cell"] = {}
    with pytest.raises(dn.func.FunctionalizationError, match="cache"):
        _fresh(functional, tensors)


@pytest.mark.parametrize(
    "hook",
    (
        "initialize",
        "build",
        "init_synapses",
        "_reset_runtime_clock",
        "_clear_duration_remainder",
    ),
)
def test_fresh_network_rejects_custom_native_initialization_adapters(hook):
    model, _ = _models(seed=812, noise=0.6)
    functional, tensors = dn.func.make_functional_network(model)
    setattr(model, hook, lambda *args, **kwargs: model)
    with pytest.raises(dn.func.FunctionalizationError):
        _fresh(functional, tensors)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_fresh_network_rejects_nonfinite_starting_time(value):
    model, _ = _models(seed=812, noise=0.6)
    functional, tensors = dn.func.make_functional_network(model)
    with pytest.raises((ValueError, dn.func.FunctionalizationError), match="finite"):
        _fresh(functional, tensors, t=value)


def test_network_initialization_starting_time_is_detached_like_native_clock_reset():
    model, _ = _models(seed=812, noise=0.6)
    functional, tensors = dn.func.make_functional_network(model)
    time = functional.initialization_inputs.t.new_tensor(2.75, requires_grad=True)
    inputs = functional.initialization_inputs._replace(t=time)
    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        inputs,
        netstim_draw=functional.sample_initialization_draw(
            tensors.parameters, tensors.state
        ),
    )
    assert initialized.state["clock"]["t"].item() == 2.75
    assert not initialized.state["clock"]["t"].requires_grad
    assert not initialized.state["clock"]["origin"].requires_grad
    assert not initialized.state["populations"]["cell"]["clock"]["t"].requires_grad


def _shared_noise_network(*, seed, initial_noise, increment):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        shared = torch.nn.Parameter(torch.full((1,), initial_noise, dtype=DTYPE))
        cell = dn.SingleCompartment(N=1, C=1, integrator=dn.bwd_euler_sc(imem=False))
        cell.material(
            "pool",
            fields={"amount": shared},
            min_values={"amount": 0.0},
            conserved={"amount": False},
        )
        cell.insert(pas, g=0.001, e=-65.0)
        cell.insert(expsyn.rename("syn"), tau=1.2, e=0.0)
        stimulus = dn.NetStim(
            N=1, interval=0.5, start=0.0, noise=0.6, max_spikes=20, seed=seed
        )
        stimulus.noise = shared
        network = dn.Network(
            {"cell": cell},
            netstim=stimulus,
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        network.connect_one_to_one(
            stimulus[:], cell[:], cell.mech.syn, weight=0.003, delay=DT
        )
        network.train()
        network.initialize(DT)
        parameter = next(
            name for name, value in cell.named_parameters() if value is shared
        )
        cell.register_post_initialize_transform(
            "change_noise",
            _Add(),
            reads=(f"parameters.{parameter}",),
            writes=(f"parameters.{parameter}",),
            inputs={"increment": torch.tensor(increment, dtype=DTYPE)},
        )
        network.initialize(DT, reinit_weights=False, reinit_delays=False)
        network.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=True,
            diff_scheduled_times=False,
            train_delay_backend="dense",
        )
        with torch.no_grad():
            shared.fill_(initial_noise)
    return network, parameter


@pytest.mark.parametrize("seed", (None, 812))
@pytest.mark.parametrize("initial_noise,increment", ((0.0, 0.6), (0.6, -0.6)))
def test_fresh_runner_samples_initialization_after_shared_noise_parameter_transform(
    seed, initial_noise, increment
):
    model, parameter = _shared_noise_network(
        seed=seed, initial_noise=initial_noise, increment=increment
    )
    native, _ = _shared_noise_network(
        seed=seed, initial_noise=initial_noise, increment=increment
    )
    native.restore_dict_from_checkpoint(_clone(model.state_dict_for_checkpoint()))
    before = _clone(model.state_dict_for_checkpoint())
    runner = dn.compile(
        model,
        dt=DT,
        steps=2,
        initialization="fresh",
        inference=False,
        trainable=(f"populations.cell.{parameter}",),
        disable=True,
    )
    actual, auxiliary = runner()
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    for _ in range(2):
        native.step()
    _close(actual, runner.functional.extract(native).state)
    effective = auxiliary["parameters"]
    assert (
        effective["populations"]["cell"][parameter]
        is effective["netstim"]["parameters"]["noise"]
    )
    torch.testing.assert_close(
        effective["netstim"]["parameters"]["noise"],
        torch.full((1,), initial_noise + increment, dtype=DTYPE),
    )
    _close(model.state_dict_for_checkpoint(), before)


def test_pure_network_initialization_compiles_fullgraph_at_nonzero_starting_time():
    model, _ = _models(seed=812, noise=0.6)
    functional, tensors = dn.func.make_functional_network(model)
    inputs = functional.initialization_inputs._replace(
        t=torch.tensor(2.75, dtype=DTYPE)
    )
    draw = functional.sample_initialization_draw(tensors.parameters, tensors.state)
    compiled = torch.compile(functional.initialize, backend="aot_eager", fullgraph=True)
    actual = compiled(
        tensors.parameters, tensors.constants, tensors.state, inputs, netstim_draw=draw
    )
    expected = functional.initialize(
        tensors.parameters, tensors.constants, tensors.state, inputs, netstim_draw=draw
    )
    _close(actual, expected)


@pytest.mark.parametrize("lanes", (0, 3))
def test_pure_network_initialization_vmap_starting_times_including_empty_lanes(lanes):
    model, _ = _models(seed=812, noise=0.6)
    functional, tensors = dn.func.make_functional_network(model)
    inputs = functional.initialization_inputs
    draw = functional.sample_initialization_draw(tensors.parameters, tensors.state)
    times = torch.arange(lanes, dtype=DTYPE) - 0.25

    def initialize_at(time):
        return functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            inputs._replace(t=time),
            netstim_draw=draw,
        ).state

    actual = torch.vmap(initialize_at)(times)
    torch.testing.assert_close(actual["clock"]["t"], times)
    if lanes:
        for lane, time in enumerate(times):
            selected = torch.utils._pytree.tree_map_only(
                torch.Tensor, lambda value: value[lane], actual
            )
            _close(selected, initialize_at(time))
    else:
        reference = initialize_at(inputs.t)
        expected = torch.utils._pytree.tree_map_only(
            torch.Tensor, lambda value: value.new_empty((0, *value.shape)), reference
        )
        _close(actual, expected)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_compiled_network_initialization_rejects_nonfinite_starting_time(value):
    model, _ = _models(seed=812, noise=0.6)
    functional, tensors = dn.func.make_functional_network(model)
    inputs = functional.initialization_inputs
    draw = functional.sample_initialization_draw(tensors.parameters, tensors.state)
    compiled = torch.compile(functional.initialize, backend="aot_eager", fullgraph=True)
    compiled(
        tensors.parameters, tensors.constants, tensors.state, inputs, netstim_draw=draw
    )
    with pytest.raises(RuntimeError, match="finite"):
        compiled(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            inputs._replace(t=inputs.t.new_tensor(value)),
            netstim_draw=draw,
        )
