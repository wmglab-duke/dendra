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
