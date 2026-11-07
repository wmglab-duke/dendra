#!/usr/bin/env python3
"""Lower authored model classes through Dendra's inherited pure machinery."""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mod import graded_syn, pas

DT = 0.01
STEPS = 4
DTYPE = torch.float64


# region model_classes
class Passive:
    def __init__(self, *args, g=0.0004, **kwargs):
        super().__init__(*args, **kwargs)
        self.insert(pas, g=g, e=-70.0)

    def description(self):
        return f"{type(self).__name__}: {tuple(self.shape)}"


class PassiveCell(Passive, dn.SingleCompartment):
    pass


class PassiveCable(Passive, dn.Cable):
    pass


class PassiveTree(Passive, dn.Tree):
    pass


class PassiveExtCellAxon(Passive, dn.ExtCellAxon):
    pass


class PassiveExtCellTree(Passive, dn.ExtCellTree):
    pass


# endregion model_classes


def morphology():
    result = dn.Morphology()
    root = result.section("root", L=20.0, diam=5.0, nseg=2)
    tail = result.section("tail", L=30.0, diam=2.0, nseg=3)
    tail.connect(root.at(1.0))
    return result


def check_family(make):
    source, native = make(), make()
    source.initialize()
    native.initialize()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    initialized = functional.initialize(
        tensors.parameters, tensors.constants, tensors.initialization
    )
    native.initialize()
    final, _ = functional.prepare_and_rollout(
        initialized.parameters, initialized.constants, initialized.state, steps=STEPS
    )
    for _ in range(STEPS):
        native.step(DT)
    torch.testing.assert_close(final, functional.extract(native).state)
    torch.testing.assert_close(functional.extract(source), tensors)
    print(source.description())


# region network_class
class PairNetwork(dn.Network):
    def __init__(self):
        shared = torch.nn.Parameter(torch.tensor(0.0004, dtype=DTYPE))
        left = PassiveCell(N=1, C=1, g=shared, v_init=-61.0)
        right = PassiveCell(N=1, C=1, g=shared, v_init=-65.0)
        right.insert(graded_syn.rename("syn"), e=0.0)
        super().__init__({"left": left, "right": right})
        self.connect_continuous_one_to_one(
            left[:],
            right[:],
            right.mech.syn,
            input="g_pre",
            transform=torch.sigmoid,
            weight=0.002,
            delay=DT,
        )


# endregion network_class


def check_network():
    network, native = PairNetwork(), PairNetwork()
    network.train()
    native.train()
    network.initialize(DT)
    native.initialize(DT)
    # region compile
    runner = dn.compile(
        network,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=False,
        trainable="*pas.g_param",
        checkpoint_every=2,
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
        backend="aot_eager",
    )
    final, auxiliary = runner()
    # endregion compile
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    for _ in range(STEPS):
        native.step()
    torch.testing.assert_close(final, runner.functional.extract(native).state)

    # region gradients
    gain = runner.get_parameter(
        "populations.left.integrator.mech.mechanisms.pas.g_param"
    )
    assert gain is runner.get_parameter(
        "populations.right.integrator.mech.mechanisms.pas.g_param"
    )
    loss = final["populations"]["right"]["integrator"]["v"].square().sum()
    gradient = torch.autograd.grad(loss, gain)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    # endregion gradients

    # region continuation
    continued, _ = runner(
        final,
        parameters=auxiliary["parameters"],
        constants=auxiliary["constants"],
        callback_state=auxiliary["callbacks"].state,
    )
    # endregion continuation
    for _ in range(STEPS):
        native.step()
    torch.testing.assert_close(continued, runner.functional.extract(native).state)


if __name__ == "__main__":
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        for make in (
            lambda: PassiveCell(N=1, C=1),
            lambda: PassiveCable.from_morphology(morphology(), N=1),
            lambda: PassiveTree.from_morphology(morphology(), N=1),
            lambda: PassiveExtCellAxon([8.0], n_comp=3),
            lambda: PassiveExtCellTree.from_graph(
                morphology().compile().to_networkx(), N=1
            ),
        ):
            check_family(make)
        check_network()
    print("Subclass native parity, shared gradients, and continuation passed.")
