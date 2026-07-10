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
