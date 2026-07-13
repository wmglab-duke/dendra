"""Public timing contracts for fixed-step network event delivery.

Absolute event times and connection delays answer different questions.  An
absolute event cannot be observed before it occurs, so it is assigned to the
first grid point at or after its time.  A physical NetCon delay is quantized to
the nearest grid interval (ties to even), with one step as the earliest possible
delivery because the current receive slot has already been consumed.
"""

import pytest
import torch

import dendra as dn
from dendra.models.mod import expsyn

DT = 0.125  # exactly representable in float32 and float64
DTYPE = torch.float64


def _built_netcon(*, delays, weights=None, tau=1.0):
    delay_spec = torch.as_tensor(delays, dtype=DTYPE).flatten()
    n_source = int(delay_spec.numel())
    if weights is None:
        weights = torch.ones(n_source, dtype=DTYPE)
    else:
        weights = torch.as_tensor(weights, dtype=DTYPE).flatten()

    post = dn.SingleCompartment(N=1, C=1, v_init=-65.0, dtype=DTYPE)
    post.insert(expsyn.rename("syn"), e=0.0, tau=tau)
    stim = dn.NetStim(
        N=n_source,
        interval=100.0,
        start=100.0,
        noise=0.0,
        max_spikes=1,
        dtype=DTYPE,
    )
    net = dn.Network({"post": post}, netstim=stim, track_netcon_events=True)
    net.connect_dense(
        stim[:],
        post[:],
        post.mech.syn,
        threshold=None,
        weight=weights,
        delay=delay_spec,
    )
    net.initialize(DT)
    return net, next(iter(net.synapses.values()))


@pytest.mark.parametrize("storage", ["value", "weight_ref", "time_ref"])
def test_absolute_event_times_use_the_first_grid_point_at_or_after_the_event(
    storage,
):
    """Adjacent representable times must remain causal at a grid boundary."""
    net, netcon = _built_netcon(delays=[DT, DT, DT], weights=[1.0, 2.0, 4.0])
    boundary = torch.tensor(2 * DT, dtype=DTYPE)
    before = torch.nextafter(boundary, torch.tensor(-torch.inf, dtype=DTYPE))
    after = torch.nextafter(boundary, torch.tensor(torch.inf, dtype=DTYPE))

    times = torch.stack((before, boundary, after))
    if storage == "value":
        netcon.schedule(con_indices=[0, 1, 2], times_ms=times)
    elif storage == "weight_ref":
        netcon.bind_weight_source(torch.ones(3, dtype=DTYPE))
        netcon.schedule_ref(con_indices=[0, 1, 2], times_ms=times, weight_idx=[0, 1, 2])
    else:
        netcon.bind_time_source(times)
        netcon.schedule_time_ref(con_indices=[0, 1, 2], time_idx=[0, 1, 2])

    # The event immediately before and the event exactly on the grid point are
    # visible at step 2.  The event immediately after it cannot be visible until
    # step 3.  In particular, retaining float64 here prevents a just-after event
    # from being collapsed onto the boundary by float32 storage.
    assert netcon.sched_abs_step.tolist() == [2, 2, 3]

    scheduled = []
    delivered = []
    conductance = []
    for _ in range(5):
        netcon.advance()
        scheduled.append(netcon.sched_counts.detach().clone())
        delivered.append(netcon.events.detach().clone())
        conductance.append(net.post.mech.syn.g.item())

    assert [value.tolist() for value in scheduled] == [
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ]
    assert [value.tolist() for value in delivered] == [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 0],
        [0, 0, 1],
    ]
    assert conductance == pytest.approx([0.0, 0.0, 0.0, 3.0, 7.0])


def test_physical_delays_use_nearest_even_with_a_one_step_minimum():
    """Delay quantization is explicit, including exact half-step ties."""
    ratios = torch.tensor([0.49, 0.5, 0.51, 1.49, 1.5, 1.51, 2.5, 3.5], dtype=DTYPE)
    weights = 10.0 ** torch.arange(ratios.numel(), dtype=DTYPE)
    net, netcon = _built_netcon(delays=ratios * DT, weights=weights)
    assert netcon.delay_ms().dtype == netcon.dt.dtype == DTYPE

    # torch.round uses nearest-even: 0.5 -> 0, 1.5 -> 2, 2.5 -> 2,
    # and 3.5 -> 4.  Once the current receive slot has been consumed, a
    # detected event cannot be delivered in that same slot, so zero-step delays
    # are promoted to one step for inference.
    assert netcon.delay_steps.tolist() == [0, 0, 1, 1, 2, 2, 2, 4]
    assert netcon.inference_delay_steps.tolist() == [1, 1, 1, 1, 2, 2, 2, 4]

    netcon.schedule(
        con_indices=list(range(ratios.numel())),
        times_ms=[0.0] * ratios.numel(),
    )

    event_trace = []
    conductance = []
    for _ in range(5):
        netcon.advance()
        event_trace.append(netcon.events.detach().clone())
        conductance.append(net.post.mech.syn.g.item())

    assert [value.tolist() for value in event_trace] == [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]
    assert conductance == pytest.approx(
        [0.0, 1_111.0, 1_111_111.0, 1_111_111.0, 11_111_111.0]
    )


def test_colliding_events_survive_chunked_continuation_and_sum_before_advance():
    """Events queued in separate run chunks still collide at one receive phase."""
    tau = 1.0
    net, netcon = _built_netcon(
        delays=[4 * DT, 2 * DT, DT], weights=[1.0, 10.0, 100.0], tau=tau
    )
    netcon.schedule(
        con_indices=[0, 1, 2],
        times_ms=[0.0, 2 * DT, 3 * DT],
        weight=torch.tensor([2.0, 3.0, 4.0], dtype=DTYPE),
    )

    # The first event is already pending when the first run returns.  The next
    # two are detected in the continuation; all three are due at absolute step 4.
    net.run(2 * DT)
    assert net.t.item() == pytest.approx(2 * DT)
    assert net.post.mech.syn.g.item() == 0.0

    net.run(3 * DT)

    assert net.t.item() == pytest.approx(5 * DT)
    assert netcon.events.tolist() == [1, 1, 1]
    # NetCon delivery precedes the population mechanism advance.  The three
    # payloads are first summed to 1*2 + 10*3 + 100*4 = 432, then ExpSyn decays
    # that newly received state during the same network step.
    expected = 432.0 * torch.exp(torch.tensor(-DT / tau, dtype=DTYPE)).item()
    assert net.post.mech.syn.g.item() == pytest.approx(expected)
