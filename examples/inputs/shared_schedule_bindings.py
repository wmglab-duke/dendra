#!/usr/bin/env python3
"""Train shared schedule times and amplitudes in native and functional Networks.

Run from the repository root::

    python examples/inputs/shared_schedule_bindings.py

The bounded CPU example indexes both sources on the same events, compares
native and functional optimizer segments, and resumes portable checkpoints.
"""

from __future__ import annotations

from io import BytesIO

import torch

import dendra as dn
from dendra.models.mod import expsyn, pas

DT, STEPS, SEGMENTS = 0.05, 4, 3


def make_network():
    stimulus = dn.NetStim(N=1, start=10.0, seed=71)
    post = dn.SingleCompartment(
        N=1, C=1, v_init=-65.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    post.insert(pas, g=0.001, e=-65.0)
    post.insert(expsyn.rename("event"), tau=2.0, e=0.0)
    network = dn.Network(
        {"post": post},
        netstim=stimulus,
        netcon_delay_backend="dense",
        netcon_train_backend="dense",
    )
    network.connect_one_to_one(
        stimulus[:], post[:], post.mech.event, weight=0.002, delay=DT
    )
    network.train()
    network.initialize(DT)
    network.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=False,
        diff_spiking=True,
        diff_scheduled_times=True,
        sched_width=1.3,
        train_delay_backend="dense",
    )
    connection = next(iter(network.synapses.values()))
    # region declarations
    times = torch.nn.Parameter(torch.tensor([0.075, 0.275, 0.475], dtype=torch.float64))
    amplitudes = torch.nn.Parameter(torch.tensor([0.4, 0.5, 0.6], dtype=torch.float64))
    for owner in (stimulus, connection):
        owner.bind_time_source(times)
        owner.bind_weight_source(amplitudes)
    stimulus.schedule([0, 0, 0], time_idx=[0, 1, 2], weight_idx=[0, 1, 2])
    connection.schedule(con_indices=[0, 0, 0], time_idx=[0, 1, 2], weight_idx=[0, 1, 2])
    # endregion declarations
    return network, times, amplitudes


def schedule_sources(continuation):
    source = continuation.parameters["netstim"]["schedule"]
    projection = next(iter(continuation.parameters["synapses"].values()))["schedule"]
    assert source.time_source is projection.time_source
    assert source.amplitude_source is projection.amplitude_source
    return source.time_source, source.amplitude_source


def optimizer(sources):
    return torch.optim.SGD(
        [
            {"params": [sources[0]], "lr": 1e-6},
            {"params": [sources[1]], "lr": 1e-3},
        ]
    )


def portable(value):
    buffer = BytesIO()
    torch.save(value, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=True)


def main():
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        native, times, amplitudes = make_network()
        source, _, _ = make_network()
        functional, tensors = dn.func.make_functional(source, dt=DT)
        continuation = dn.func.initialize(functional, tensors)
        native_sources = (times, amplitudes)
        functional_sources = schedule_sources(continuation)
        native_optimizer = optimizer(native_sources)
        functional_optimizer = optimizer(functional_sources)
        previous_voltage = None
        for segment in range(SEGMENTS):
            native.zero_grad(set_to_none=True)
            functional_optimizer.zero_grad()
            if segment == 1:
                assert (
                    native.longrun_checkpointed(
                        STEPS * DT, chunklength=2, safe_checkpoint=True
                    )
                    is None
                )
            else:
                native.run(STEPS * DT)
            final, auxiliary = dn.func.run(
                functional,
                continuation,
                steps=STEPS,
                checkpoint_every=2,
            )
            voltage = final["populations"]["post"]["integrator"]["v"]
            torch.testing.assert_close(voltage, native.post.v)
            native_loss = (native.post.v / 65).square().sum()
            functional_loss = (voltage / 65).square().sum()
            if previous_voltage is not None:
                assert (
                    torch.autograd.grad(
                        native_loss,
                        previous_voltage,
                        allow_unused=True,
                        retain_graph=True,
                    )[0]
                    is None
                )
            previous_voltage = native.post.v
            # region training
            native_loss.backward()
            functional_loss.backward()
            for observed, expected in zip(
                native_sources, functional_sources, strict=True
            ):
                torch.testing.assert_close(observed.grad, expected.grad)
                assert observed.grad is not None
                assert observed.grad.isfinite().all() and observed.grad.abs().sum() > 0
            native_optimizer.step()
            functional_optimizer.step()
            native.detach()
            continuation = dn.func.detach_continuation(
                functional, auxiliary["continuation"]
            )
            # endregion training
            for observed, expected in zip(
                native_sources, functional_sources, strict=True
            ):
                torch.testing.assert_close(observed, expected)
            assert schedule_sources(continuation)[0] is functional_sources[0]
            assert schedule_sources(continuation)[1] is functional_sources[1]

            if segment == 0:
                # region checkpoint
                native_saved = portable(
                    {
                        "sources": native.state_dict(),
                        "runtime": native.state_dict_for_checkpoint(),
                    }
                )
                receiving, times, amplitudes = make_network()
                receiving.load(native_saved["sources"])
                receiving.restore_dict_from_checkpoint(native_saved["runtime"])
                receiving.detach()
                native = receiving
                native_sources = (times, amplitudes)
                native_optimizer = optimizer(native_sources)
                saved = portable(dn.func.export_continuation(functional, continuation))
                continuation = dn.func.restore_continuation(functional, saved)
                # Portable restoration creates owned leaves for a new optimizer.
                functional_sources = schedule_sources(continuation)
                functional_optimizer = optimizer(functional_sources)
                # endregion checkpoint
        torch.testing.assert_close(native.t, native.t.new_tensor(SEGMENTS * STEPS * DT))
        torch.testing.assert_close(final["clock"]["t"], native.t)
        print(
            "Shared schedule bindings: native/functional values, local source gradients, "
            "optimizer updates and portable continuation passed"
        )


if __name__ == "__main__":
    main()
