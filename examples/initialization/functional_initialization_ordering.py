"""Initialize shared Network parameters in Population order.

Run: python examples/initialization/functional_initialization_ordering.py
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mod import pas

DT = 0.125
STEPS = 3
G = "integrator.mech.mechanisms.pas.g_param"
CM = "cm_scale_param.rho"


# region transform
class Affine(torch.nn.Module):
    def forward(self, value, factor, offset):
        return (factor * value + offset,)


# endregion transform


def make_network():
    conductance = torch.nn.Parameter(torch.tensor(0.001, dtype=torch.float64))
    cells = {}
    for name in ("left", "right"):
        cell = dn.SingleCompartment(
            N=1, C=1, v_init=-50.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        cell.insert(pas, g=conductance, e=-65.0)
        cell.build()
        cells[name] = cell
    # region sharing
    cells["right"].cm_scale_param.rho = cells["left"].cm_scale_param.rho
    network = dn.Network(cells)
    network.train()
    with torch.no_grad():
        network.initialize(DT)
    cells["right"].register_pre_initialize_transform(
        "scale",
        Affine(),
        reads=("parameters.cm_scale_param.rho",),
        writes=("parameters.cm_scale_param.rho",),
        inputs={
            "factor": torch.tensor(1.4, dtype=torch.float64),
            "offset": torch.tensor(0.2, dtype=torch.float64),
        },
    )
    # endregion sharing
    return network


def voltage(state, name):
    return state["populations"][name]["integrator"]["v"]


def loss(state):
    return sum(voltage(state, name).square().sum() for name in ("left", "right"))


def main():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        network, native = make_network(), make_network()
        # region compile
        runner = dn.compile(
            network,
            dt=DT,
            steps=STEPS,
            initialization="fresh",
            inference=False,
            trainable=(f"populations.*.{G}",),
            checkpoint_every=2,
            callbacks={"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
            backend="aot_eager",
        )
        shared = runner.get_parameter(f"populations.left.{G}")
        assert shared is runner.get_parameter(f"populations.right.{G}")
        # endregion compile
        # region continuation
        prefix, first = runner()
        final, second = runner(
            prefix,
            parameters=first["parameters"],
            constants=first["constants"],
            callback_state=first["callbacks"].state,
        )
        gradient = torch.autograd.grad(loss(final), shared)[0]
        # endregion continuation

        native.initialize(DT)
        assert not torch.allclose(native.left.cm_scale, native.right.cm_scale)
        assert native.left.cm_scale_param.rho is native.right.cm_scale_param.rho
        expected_trace = [native.left.v.detach().clone()]
        for _ in range(2 * STEPS):
            native.step()
            expected_trace.append(native.left.v.detach().clone())
        native_gradient = torch.autograd.grad(
            native.left.v.square().sum() + native.right.v.square().sum(),
            native.left.get_parameter(G),
        )[0]
        for name in ("left", "right"):
            torch.testing.assert_close(voltage(final, name), native.populations[name].v)
        torch.testing.assert_close(gradient, native_gradient)
        recorded = torch.cat(
            (
                first["callbacks"]["trace"]["left.v"],
                second["callbacks"]["trace"]["left.v"],
            )
        )
        torch.testing.assert_close(recorded, torch.stack(expected_trace))
        parameters = first["parameters"]["populations"]
        assert parameters["left"][CM] is parameters["right"][CM]
        assert network.t.item() == 0.0
    print("Ordered values, shared-parameter gradients, and continuation match native.")


if __name__ == "__main__":
    main()
