import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_imperative_checkpoint_callbacks import EnergyReducer
from tests.func.test_network_rollout import _driven_network
from tests.networks.test_native_network_runtime_inputs import DT as NETWORK_DT
from tests.networks.test_native_network_runtime_inputs import _drives as _network_drives
from tests.networks.test_native_network_runtime_inputs import _inputs as _network_inputs
from tests.networks.test_native_network_runtime_inputs import _loss as _network_loss
from tests.test_native_runtime_inputs import DT as POPULATION_DT
from tests.test_native_runtime_inputs import _model as _input_population
from tests.test_native_runtime_inputs import _tapes as _population_inputs


class CountCallback(dn.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.pre = 0
        self.post = 0
        self.pre_loop = 0
        self.post_loop = 0

    def pre_loop_hook(self, model):
        self.pre_loop += 1

    def post_loop_hook(self, model):
        self.post_loop += 1

    def pre_step_hook(self, model):
        self.pre += 1

    def post_step_hook(self, model):
        self.post += 1


def _cell():
    cell = dn.SingleCompartment(N=1, C=1, v_init=-65.0)
    cell.insert(pas, g=0.001, e=-70.0)
    cell.build()
    cell.initialize()
    return cell


def test_population_step_matches_one_step_run():
    dt = 0.01
    stepped = _cell()
    run = _cell()

    dn.step(stepped, dt=dt)
    run.run(tstop=dt, dt=dt)

    assert torch.allclose(stepped.v, run.v)
    assert torch.allclose(stepped.t, run.t)


def test_population_step_callbacks_are_per_step_by_default():
    cell = _cell()
    cb = CountCallback()

    dn.step(cell, dt=0.01, callbacks=[cb])

    assert cb.pre == 1
    assert cb.post == 1
    assert cb.pre_loop == 0
    assert cb.post_loop == 0


def test_population_step_loop_hooks_are_optional():
    cell = _cell()
    cb = CountCallback()

    dn.step(cell, dt=0.01, callbacks=[cb], loop_hooks=True)

    assert cb.pre == 1
    assert cb.post == 1
    assert cb.pre_loop == 1
    assert cb.post_loop == 1


def test_network_step_matches_one_step_run():
    dt = 0.01
    net_step = dn.Network({"cell": _cell()})
    net_run = dn.Network({"cell": _cell()})
    net_step.initialize(dt)
    net_run.initialize(dt)

    dn.step(net_step)
    net_run.run(dt)

    assert torch.allclose(net_step.cell.v, net_run.cell.v)
    assert torch.allclose(net_step.t, net_run.t)


def test_population_step_accepts_singleton_time_voltage_and_returns_model():
    cell = _cell()
    ve = torch.zeros((1, *cell.v.shape), dtype=cell.dtype())

    returned = dn.step(cell, dt=0.01, ve=ve)

    assert returned is cell
    assert cell.t.item() == pytest.approx(0.01)


def test_population_step_validates_initialization_and_extracellular_inputs():
    cell = dn.SingleCompartment(N=1, C=1, v_init=-65.0)
    cell.insert(pas, g=0.001, e=-70.0)
    cell.build()

    with pytest.raises(ValueError, match="initialized"):
        dn.step(cell, dt=0.01)

    cell.initialize()
    with pytest.raises(ValueError, match="either 've' or 'extra'"):
        dn.step(cell, dt=0.01, ve=torch.zeros_like(cell.v), extra=(None, None))


def test_public_step_rejects_unsupported_objects():
    with pytest.raises(TypeError, match="Population or Network"):
        dn.step(object(), dt=0.01)


def test_network_step_validates_timestep_and_population_only_voltage():
    net = dn.Network({"cell": _cell()})

    with pytest.raises(RuntimeError, match="no simulation timestep"):
        dn.step(net)

    net.initialize(0.01)
    with pytest.raises(ValueError, match="timestep is"):
        dn.step(net, dt=0.02)
    with pytest.raises(TypeError, match="only valid for Population"):
        dn.step(net, ve=torch.zeros_like(net.cell.v))

    assert dn.step(net, dt=0.01) is net


def test_network_instance_step_rejects_stale_wiring():
    net = dn.Network({"cell": _cell()})
    net.initialize(0.01)
    net.built = False

    with pytest.raises(RuntimeError, match="wiring has changed"):
        net.step()


class _StepTarget:
    """Avoid constructing numerical models when checking helper forwarding."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.initialized = self.built = True
        self.dt = 0.01
        self.synapses = {}
        self._mode_requires_initialize = False
        self.received = None


class _PopulationStepTarget(_StepTarget, dn.Population):
    def step(self, dt=None, *, ve=None, extra=None, callbacks=None, loop_hooks=False):
        self.received = {
            "dt": dt,
            "ve": ve,
            "extra": extra,
            "callbacks": callbacks,
            "loop_hooks": loop_hooks,
        }
        return self


class _NetworkStepTarget(_StepTarget, dn.Network):
    def step(self, *, extra=None, callbacks=None, loop_hooks=False):
        self.received = {
            "extra": extra,
            "callbacks": callbacks,
            "loop_hooks": loop_hooks,
        }
        return self


_STEP_HELPERS = (
    (dn.step, _PopulationStepTarget),
    (dn.step_population, _PopulationStepTarget),
    (dn.step, _NetworkStepTarget),
    (dn.step_network, _NetworkStepTarget),
)
_INPUT_HELPERS = (
    pytest.param(dn.step, False, id="step_population"),
    pytest.param(dn.step_population, False, id="specialized_population"),
    pytest.param(dn.step, True, id="step_network"),
    pytest.param(dn.step_network, True, id="specialized_network"),
)


@pytest.mark.parametrize("helper,target", _STEP_HELPERS)
def test_step_helpers_forward_callback_packets_and_return_results_unchanged(
    helper, target, monkeypatch
):
    model, returned, received = target(), object(), []

    def capture(**kwargs):
        received.append(kwargs)
        return returned

    monkeypatch.setattr(model, "step", capture)
    options = {
        "inputs": object(),
        "draws": object(),
        "extra": object(),
        "callbacks": [object()],
        "loop_hooks": True,
        "functional_callbacks": object(),
        "functional_callback_state": object(),
    }
    if isinstance(model, dn.Population):
        options.update(dt=0.02, ve=object())
    assert helper(model, **options) is returned
    assert len(received) == 1 and received[0] == options
    assert all(received[0][key] is value for key, value in options.items())


@pytest.mark.parametrize("helper,target", _STEP_HELPERS)
@pytest.mark.parametrize("explicit_none", (False, True), ids=("omitted", "none"))
def test_step_helpers_keep_authored_legacy_step_signatures_when_inputs_are_omitted_or_none(
    helper, target, explicit_none
):
    model = target()
    assert (
        helper(model, **({"inputs": None, "draws": None} if explicit_none else {}))
        is model
    )
    expected = {"extra": None, "callbacks": None, "loop_hooks": False}
    if isinstance(model, dn.Population):
        expected.update(dt=None, ve=None)
    assert model.received == expected


@pytest.mark.parametrize("helper,target", _STEP_HELPERS)
def test_step_helpers_forward_orphan_callback_state_to_the_native_guard(
    helper, target, monkeypatch
):
    model = target()
    base = dn.Population if isinstance(model, dn.Population) else dn.Network
    monkeypatch.setattr(model, "step", base.step.__get__(model, type(model)))
    with pytest.raises(ValueError, match="requires functional_callbacks"):
        helper(model, functional_callback_state=object())


def test_public_step_keeps_network_voltage_and_timestep_validation_before_forwarding():
    model = _NetworkStepTarget()
    options = {
        "inputs": object(),
        "functional_callbacks": object(),
        "functional_callback_state": object(),
    }
    with pytest.raises(TypeError, match="only valid for Population"):
        dn.step(model, dt=0.02, ve=object(), **options)
    with pytest.raises(ValueError, match="timestep is"):
        dn.step(model, dt=0.02, **options)
    assert model.received is None


@pytest.mark.parametrize("helper,target", _STEP_HELPERS)
def test_step_helpers_keep_input_packet_tensor_views_and_aliases(
    helper, target, monkeypatch
):
    model = target()
    source = torch.arange(8.0, requires_grad=True)
    view = source[::2]
    packet = dn.func.Inputs(ve=view, intra=view)
    inputs = (
        {"left": packet, "right": packet} if isinstance(model, dn.Network) else packet
    )
    returned = object()
    received = []

    def capture(**kwargs):
        received.append(kwargs["inputs"])
        return returned

    monkeypatch.setattr(model, "step", capture)
    assert helper(model, inputs=inputs) is returned
    assert len(received) == 1 and received[0] is inputs
    actual = received[0]["left"] if isinstance(model, dn.Network) else received[0]
    assert actual is packet and actual.ve is actual.intra is view
    assert actual.ve.grad_fn is view.grad_fn


@pytest.fixture
def _input_environment():
    threads = torch.get_num_threads()
    rng = torch.get_rng_state().clone()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False
            ),
        ):
            yield
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


def _input_case(network):
    return _driven_network() if network else _input_population(batch=True)[0]


def _input_tapes(model, network, steps):
    if network:
        return _network_drives(model, steps)
    return tuple(_population_inputs(model, steps, broadcast=True))


def _input_frame(leaves, network, index):
    ve, intra = (value[index] for value in leaves)
    return _network_inputs(ve, intra) if network else dn.func.Inputs(ve=ve, intra=intra)


def _input_step(helper, model, network, **options):
    if not network:
        options["dt"] = POPULATION_DT
    return model.step(**options) if helper is None else helper(model, **options)


def _input_callbacks(model, network):
    energy = EnergyReducer()
    if network:
        energy = dn.callbacks.PopulationCallback("left", energy)
    return dn.func.make_callbacks(
        model,
        {
            "trace": dn.Recorder(
                ("left.v", "right.v", "t") if network else ("v", "t"),
                dt=2 * (NETWORK_DT if network else POPULATION_DT),
            ),
            "energy": energy,
        },
        dt=NETWORK_DT if network else POPULATION_DT,
    )


@pytest.mark.cpu
@pytest.mark.usefixtures("_input_environment")
@pytest.mark.parametrize("helper,network", _INPUT_HELPERS)
def test_step_helper_inputs_match_direct_frames_gradients_and_callback_continuation(
    helper, network
):
    model, reference = _input_case(network), _input_case(network)
    leaves = _input_tapes(model, network, 2)
    reference_leaves = tuple(
        value.detach().clone().requires_grad_() for value in leaves
    )
    frozen = _snapshot(leaves)
    callbacks, reference_callbacks = (
        _input_callbacks(model, network),
        _input_callbacks(reference, network),
    )
    hooks, reference_hooks = CountCallback(), CountCallback()
    carry = reference_carry = None
    for index in range(2):
        result = _input_step(
            helper,
            model,
            network,
            inputs=_input_frame(leaves, network, index),
            callbacks=[hooks],
            functional_callbacks=callbacks,
            callback_state=carry,
            loop_hooks=True,
        )
        expected = _input_step(
            None,
            reference,
            network,
            inputs=_input_frame(reference_leaves, network, index),
            callbacks=[reference_hooks],
            functional_callbacks=reference_callbacks,
            callback_state=reference_carry,
            loop_hooks=True,
        )
        _close(result.outputs, expected.outputs)
        _close(result.state.carries, expected.state.carries)
        assert result["trace"]["t"].shape == (1,)
        carry, reference_carry = result.state, expected.state
    assert result["energy"]["steps"].item() == 2
    assert carry.plan_token is callbacks._state_token
    assert (hooks.pre, hooks.post, hooks.pre_loop, hooks.post_loop) == (2, 2, 2, 2)
    assert (
        reference_hooks.pre,
        reference_hooks.post,
        reference_hooks.pre_loop,
        reference_hooks.post_loop,
    ) == (2, 2, 2, 2)
    _assert_packet_equal(
        model.state_dict_for_checkpoint(), reference.state_dict_for_checkpoint()
    )
    loss = (
        _network_loss(cell.v for cell in model.populations.values())
        if network
        else model.v.square().mean()
    )
    reference_loss = (
        _network_loss(cell.v for cell in reference.populations.values())
        if network
        else reference.v.square().mean()
    )
    actual = torch.autograd.grad(loss, leaves)
    wanted = torch.autograd.grad(reference_loss, reference_leaves)
    _close(actual, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in actual)
    _close(leaves, frozen)


@pytest.mark.cpu
@pytest.mark.usefixtures("_input_environment")
@pytest.mark.parametrize("helper,network", _INPUT_HELPERS)
@pytest.mark.parametrize("invalid", ("dtype", "packet", "extra"))
def test_step_helper_input_errors_match_native_guards_before_clock_advancement(
    helper, network, invalid, monkeypatch
):
    model, reference = _input_case(network), _input_case(network)
    population = model.left if network else model
    value = torch.zeros(population.shape, dtype=population.dtype())
    packet = dn.func.Inputs(ve=value.float() if invalid == "dtype" else value)
    inputs = {"left": packet} if network else packet
    options = {}
    if invalid == "packet":
        inputs = {"left": value} if network else {"ve": value}
    elif invalid == "extra":
        waveform = dn.constant(value=0.1)

        def forbidden(*args, **kwargs):
            raise AssertionError("input rejection must precede waveform evaluation")

        monkeypatch.setattr(waveform, "forward", forbidden)
        extra = (torch.ones_like(population.v), waveform)
        options["extra"] = {"left": extra} if network else extra
    saved, reference_saved = (
        _snapshot(model.state_dict_for_checkpoint()),
        _snapshot(reference.state_dict_for_checkpoint()),
    )
    global_rng = torch.get_rng_state().clone()
    with pytest.raises((TypeError, ValueError)) as expected:
        _input_step(None, reference, network, inputs=inputs, **options)
    with pytest.raises(type(expected.value)) as actual:
        _input_step(helper, model, network, inputs=inputs, **options)
    assert str(actual.value) == str(expected.value)
    _assert_packet_equal(model.state_dict_for_checkpoint(), saved)
    _assert_packet_equal(reference.state_dict_for_checkpoint(), reference_saved)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
