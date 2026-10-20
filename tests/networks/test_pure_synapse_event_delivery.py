"""The pure receive adapter retains the ordered NetCon receive boundary."""

import torch

import dendra as dn
from dendra.models.mechanisms import Synapse

DT = 0.125


class _ReceiveLedger(Synapse):
    Synapse.GLOBAL(scale=1.0)
    Synapse.CARRY("calls", dtype=torch.int64)
    Synapse.CARRY("ledger", "elapsed")

    def net_receive_values(self, weights, dt, values):
        return {
            "carry": {
                "calls": values["calls"] + 1,
                "ledger": 10 * values["ledger"] + weights * self.scale + 1,
                "elapsed": values["elapsed"] + dt,
            }
        }


def test_network_keeps_zero_payload_calls_and_ordered_receive_updates():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False):
        post = dn.Population(
            N=1, C=1, dtype=torch.float64, integrator=dn.scnv(imem=False)
        )
        post.insert(_ReceiveLedger.rename("receiver"))
        sources = {
            name: dn.Population(
                N=1,
                C=1,
                v_init=0.0,
                dtype=torch.float64,
                integrator=dn.scnv(imem=False),
            )
            for name in ("source_a", "source_b")
        }
        network = dn.Network(
            {**sources, "post": post},
            netcon_delay_backend="dense",
        )
        for source, weight in zip(sources.values(), (1.0, 2.0), strict=True):
            network.connect_one_to_one(
                source[:],
                post[:],
                post.mech.receiver,
                threshold=None,
                weight=weight,
                delay=DT,
            )
        network.initialize(DT)
    connections = tuple(network.synapses.values())
    assert len(connections) == 2
    for connection in connections:
        connection.schedule(con_indices=[0], times_ms=[0.0])

    # First-step reception is zero for both connections; current source events
    # enter their future slots only after that connection's receive call.
    network.step()
    receiver = network.post.mech.receiver
    torch.testing.assert_close(receiver.calls, torch.full_like(receiver.calls, 2))
    torch.testing.assert_close(receiver.ledger, torch.full_like(receiver.ledger, 11))
    torch.testing.assert_close(
        receiver.elapsed, torch.full_like(receiver.elapsed, 2 * DT)
    )

    # Each connection delivers separately, in insertion order: weights 1 then 2.
    network.step()
    torch.testing.assert_close(receiver.calls, torch.full_like(receiver.calls, 4))
    torch.testing.assert_close(receiver.ledger, torch.full_like(receiver.ledger, 1123))
    torch.testing.assert_close(
        receiver.elapsed, torch.full_like(receiver.elapsed, 4 * DT)
    )
