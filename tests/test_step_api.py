import pytest
import torch

import dendra as dn
from dendra.models.mod import pas


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
def test_step_helpers_keep_authored_legacy_step_signatures_when_callbacks_are_omitted(
    helper, target
):
    model = target()
    assert helper(model) is model
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
    options = {"functional_callbacks": object(), "functional_callback_state": object()}
    with pytest.raises(TypeError, match="only valid for Population"):
        dn.step(model, dt=0.02, ve=object(), **options)
    with pytest.raises(ValueError, match="timestep is"):
        dn.step(model, dt=0.02, **options)
    assert model.received is None
