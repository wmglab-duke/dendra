# tests/test_netstim.py
import itertools

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import dendra.models.networks.netstim as M

# -------------------------- utilities & fixtures ----------------------------------


def have_cuda():
    return torch.cuda.is_available()


@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    return torch.device(request.param)


def times_from_schedule(starts, intervals, max_spikes, dtype=torch.float32):
    """
    Build expected spike times using the *same dtype arithmetic* as the model
    (float32), to avoid float64-vs-float32 drift at boundary comparisons.
    """
    out = []
    for s, dt, m in zip(starts, intervals, max_spikes):
        s_t = torch.tensor(float(s), dtype=dtype)
        dt_t = torch.tensor(float(dt), dtype=dtype)
        seq = []
        for _ in range(int(m)):
            seq.append(float(s_t.item()))
            s_t = s_t + dt_t  # float32 addition like the model
        out.append(seq)
    return out


def step_model(ns: M.NetStim, ts):
    """
    Advance NetStim over times in ts; collect spikes per time (bool tensor).
    Returns list of bool tensors.
    """
    outs = []
    for t in ts:
        ns.forward(float(t))
        outs.append(ns.spikes.clone())
    return outs


# ------------------------------ validation ----------------------------------------


def test_validate_parameter_lengths_mismatch_raises():
    with pytest.raises(ValueError):
        M.NetStim(N=3, interval=[10.0, 10.0], start=0.0, noise=0.0, max_spikes=10)

    with pytest.raises(ValueError):
        M.NetStim(N=2, interval=10.0, start=[0.0, 1.0, 2.0], noise=0.0, max_spikes=10)

    with pytest.raises(ValueError):
        M.NetStim(N=2, interval=10.0, start=0.0, noise=[0.0, 0.5, 1.0], max_spikes=10)

    with pytest.raises(ValueError):
        M.NetStim(N=2, interval=10.0, start=0.0, noise=0.0, max_spikes=[1, 2, 3])


# ------------------------------ clamping ------------------------------------------


def test_noise_is_clamped_scalar_and_vector():
    # Scalar negative / >1 clamp
    a = M.NetStim(N=1, noise=-0.5)
    assert float(a.noise) == pytest.approx(0.0)

    b = M.NetStim(N=1, noise=7.5)
    assert float(b.noise) == pytest.approx(1.0)

    # Vector clamp should work before and after initialize().
    vec = [-1.0, 0.25, 1.3]
    c = M.NetStim(N=3, noise=vec)
    assert torch.allclose(c.noise, torch.tensor([0.0, 0.25, 1.0], dtype=torch.float32))


# ------------------------------ device & dtype ------------------------------------


def test_device_and_dtype_report_buffers(device):
    ns = M.NetStim(N=4, interval=5.0, start=0.0, noise=0.0).to(device)
    ns.initialize()

    # Compare device type; index handling differs when you pass 'cuda' without an index.
    assert ns.device().type == device.type
    if device.type == "cuda":
        expected_idx = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        assert ns.device().index == expected_idx

    assert ns.dtype() == ns.next_stoch_time.dtype == torch.float32
    assert ns.next_stoch_time.device.type == device.type


# ------------------------------ initialize() --------------------------------------


def test_initialize_noise_zero_sets_next_equal_start():
    N = 3
    starts = [0.0, 5.0, 2.0]
    ns = M.NetStim(N=N, interval=10.0, start=starts, noise=0.0, max_spikes=5)
    ns.initialize()
    assert torch.allclose(ns.next_stoch_time, torch.tensor(starts, dtype=torch.float32))
    assert torch.equal(ns.spike_counts, torch.zeros(N, dtype=torch.long))


def test_initialize_with_seed_makes_offsets_reproducible_when_noise_positive():
    # Two identical instances with the same seed → identical next_stoch_time after initialize
    N = 5
    cfg = dict(N=N, interval=7.0, start=3.0, noise=1.0, max_spikes=10, seed=12345)
    a = M.NetStim(**cfg).initialize()
    b = M.NetStim(**cfg).initialize()
    assert torch.allclose(a.next_stoch_time, b.next_stoch_time)

    # Different seed → very likely different offsets
    c = M.NetStim(**{**cfg, "seed": 54321}).initialize()
    if torch.allclose(a.next_stoch_time, c.next_stoch_time):
        a.forward(float(torch.max(a.next_stoch_time).item()))
        c.forward(float(torch.max(c.next_stoch_time).item()))
        assert not torch.allclose(a.next_stoch_time, c.next_stoch_time)


# ------------------------------ forward() deterministic (noise=0) -----------------


def test_forward_noise_zero_spikes_on_schedule_and_respects_max_spikes():
    # Three units with different starts/intervals/max_spikes, all deterministic (noise=0)
    N = 3
    starts = [0.0, 2.0, 1.5]
    intervals = [5.0, 3.0, 7.0]
    max_spikes = [3, 2, 4]
    ns = M.NetStim(
        N=N, interval=intervals, start=starts, noise=0.0, max_spikes=max_spikes, seed=77
    )
    ns.initialize()

    # Expected spike times for each unit
    exp_times = times_from_schedule(starts, intervals, max_spikes)
    all_times = sorted(set(itertools.chain.from_iterable(exp_times)))

    outs = step_model(ns, all_times)

    # Check spikes are exactly at expected times
    for time_idx, t in enumerate(all_times):
        spikes = outs[time_idx]
        for i in range(N):
            should_spike = t in exp_times[i]
            assert bool(spikes[i].item()) is should_spike

    # After all events, spike_counts should equal max_spikes
    assert torch.equal(ns.spike_counts, torch.tensor(max_spikes, dtype=torch.long))

    # No further spikes once max reached
    last_t = all_times[-1] + 100.0
    ns.forward(last_t)
    assert not ns.spikes.any()


# ------------------------------ forward() stochastic parity -----------------------


def test_forward_same_seed_same_spike_sequence_when_noise_positive():
    cfg = dict(N=4, interval=5.0, start=0.0, noise=1.0, max_spikes=5, seed=2024)
    a = M.NetStim(**cfg).initialize()
    b = M.NetStim(**cfg).initialize()

    ts = [0.0] + [k * 2.5 for k in range(1, 15)]
    a_outs = step_model(a, ts)
    b_outs = step_model(b, ts)
    for sa, sb in zip(a_outs, b_outs):
        assert torch.equal(sa, sb)
    assert torch.equal(a.spike_counts, b.spike_counts)
    assert torch.allclose(a.next_stoch_time, b.next_stoch_time)


def test_forward_different_seed_probably_different_sequence_when_noise_positive():
    cfg1 = dict(N=6, interval=4.0, start=0.0, noise=1.0, max_spikes=6, seed=1)
    cfg2 = dict(N=6, interval=4.0, start=0.0, noise=1.0, max_spikes=6, seed=2)
    a = M.NetStim(**cfg1).initialize()
    b = M.NetStim(**cfg2).initialize()
    ts = [k * 3.0 for k in range(12)]
    a_outs = step_model(a, ts)
    b_outs = step_model(b, ts)
    any_diff = any(not torch.equal(x, y) for x, y in zip(a_outs, b_outs))
    if not any_diff:
        a.forward(9999.0)
        b.forward(9999.0)
        assert not torch.allclose(a.next_stoch_time, b.next_stoch_time)


# ------------------------------ per-unit broadcasting (noise=0) -------------------


def test_per_unit_interval_start_maxspikes_work_with_noise_zero():
    N = 4
    intervals = [2.0, 3.0, 5.0, 7.0]
    starts = [0.0, 1.0, 0.0, 2.5]
    msp = [1, 2, 3, 1]
    ns = M.NetStim(
        N=N, interval=intervals, start=starts, noise=0.0, max_spikes=msp, seed=9
    )
    ns.initialize()

    exp = times_from_schedule(starts, intervals, msp)
    all_times = sorted(set(itertools.chain.from_iterable(exp)))
    outs = step_model(ns, all_times)

    for ti, t in enumerate(all_times):
        s = outs[ti]
        for i in range(N):
            assert bool(s[i]) == (t in exp[i])


# ------------------------------ CUDA (optional) -----------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_behavior_generators_on_device_and_spiking(device=torch.device("cuda")):
    ns = M.NetStim(N=3, interval=5.0, start=0.0, noise=0.0, seed=42).to(device)
    ns.initialize()
    ns.forward(0.0)
    assert ns.next_stoch_time.device.type == "cuda"
    assert ns.spikes.device.type == "cuda"
    assert ns.spike_counts.device.type == "cuda"


# ------------------------------ Hypothesis property (noise=0) ---------------------


@settings(max_examples=60)
@given(
    N=st.integers(min_value=1, max_value=5),
    starts=st.lists(
        st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=5,
    ),
    intervals=st.lists(
        st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=5,
    ),
    maxsp=st.lists(st.integers(min_value=1, max_value=4), min_size=1, max_size=5),
)
def test_noise_zero_schedule_matches_exact(N, starts, intervals, maxsp):
    N = min(N, len(starts), len(intervals), len(maxsp))

    # Quantize to float32 to match module buffers and avoid subnormal surprises.
    starts32 = torch.tensor(starts[:N], dtype=torch.float32).tolist()
    intervals32 = torch.tensor(intervals[:N], dtype=torch.float32).tolist()
    maxsp = [int(x) for x in maxsp[:N]]

    ns = M.NetStim(
        N=N, interval=intervals32, start=starts32, noise=0.0, max_spikes=maxsp, seed=777
    ).initialize()

    exp = times_from_schedule(starts32, intervals32, maxsp)
    all_times = sorted(set(itertools.chain.from_iterable(exp)))
    outs = step_model(ns, all_times)

    for ti, t in enumerate(all_times):
        s = outs[ti]
        assert s.shape == (N,)
        for i in range(N):
            assert bool(s[i]) == (t in exp[i])

    assert torch.equal(ns.spike_counts, torch.tensor(maxsp, dtype=torch.long))


def test_noise_zero_tiny_positive_start_does_not_fire_at_zero():
    zero = torch.tensor(0.0, dtype=torch.float32)
    tiny = torch.nextafter(zero, torch.tensor(1.0, dtype=torch.float32)).item()
    ns = M.NetStim(
        N=2,
        interval=[1.0, 1.0],
        start=[0.0, tiny],
        noise=0.0,
        max_spikes=[1, 1],
        seed=777,
    ).initialize()

    ns.forward(0.0)
    assert torch.equal(ns.spikes, torch.tensor([True, False]))

    ns.forward(tiny)
    assert torch.equal(ns.spikes, torch.tensor([False, True]))


def test_noise_zero_positive_param_one_ulp_interval_drift_still_fires():
    # PositiveParam(1.5845052003860474) reconstructs the interval one float32
    # ulp larger on current PyTorch, so the exact float32 schedule time can be
    # just below the internal next_stoch_time. The event comparison should be
    # tolerant to that local ulp drift without using a coarse absolute tolerance.
    interval = torch.tensor(1.5845052203122405, dtype=torch.float32).item()
    ns = M.NetStim(
        N=1,
        interval=[interval],
        start=[0.0],
        noise=0.0,
        max_spikes=[2],
        seed=777,
    ).initialize()

    ns.forward(0.0)
    assert bool(ns.spikes[0])

    ns.forward(interval)
    assert bool(ns.spikes[0])


# ------------------------------ misc ---------------------------------------------


def test_numel_reports_N():
    ns = M.NetStim(N=7)
    assert ns.numel() == 7


def test_spikes_buffer_dtype_and_shape_after_forward():
    ns = M.NetStim(N=5, interval=1.0, start=0.0, noise=0.0).initialize()
    ns.forward(0.0)
    assert ns.spikes.dtype == torch.bool
    assert ns.spikes.shape == (5,)


def test_initialize_vector_noise_randomizes_only_positive_entries():
    N = 5
    starts = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    noise = [0.0, 0.5, 1.0, 0.0, 0.2]  # per-unit noise
    interval = 2.0  # scalar broadcast

    ns1 = M.NetStim(
        N=N,
        interval=interval,
        start=starts.tolist(),
        noise=noise,
        max_spikes=10,
        seed=123,
    ).initialize()

    # noise == 0 → first spike exactly at start; noise > 0 → strictly later than start
    nz = torch.tensor(noise, dtype=torch.float32)
    for i in range(N):
        if nz[i] == 0:
            assert ns1.next_stoch_time[i].item() == pytest.approx(starts[i].item())
        else:
            assert ns1.next_stoch_time[i].item() > starts[i].item()

    # Same seed + same config → identical initialization
    ns2 = M.NetStim(
        N=N,
        interval=interval,
        start=starts.tolist(),
        noise=noise,
        max_spikes=10,
        seed=123,
    ).initialize()
    assert torch.allclose(ns1.next_stoch_time, ns2.next_stoch_time)

    # Different seed → at least one randomized entry should differ
    ns3 = M.NetStim(
        N=N,
        interval=interval,
        start=starts.tolist(),
        noise=noise,
        max_spikes=10,
        seed=999,
    ).initialize()
    # only check indices where noise > 0
    pos_idx = (nz > 0).nonzero(as_tuple=True)[0]
    if len(pos_idx) > 0:
        assert not torch.allclose(
            ns1.next_stoch_time[pos_idx], ns3.next_stoch_time[pos_idx]
        )


def test_batch_prepends_dimensions_and_preserves_generator_axis():
    ns = M.NetStim(N=3, interval=[1.0, 2.0, 3.0], start=0.0, noise=0.0)
    ns.initialize()
    ns.batch(4)

    assert ns.N == 3
    assert ns.shape == (4, 3)
    assert ns.numel() == 12
    for name in (
        "noise",
        "start",
        "start_cache",
        "max_spikes",
        "next_stoch_time",
        "next_sched_time",
        "spike_counts",
        "spikes",
    ):
        assert tuple(getattr(ns, name).shape) == (4, 3)

    ns.batch(2)
    assert ns.shape == (2, 4, 3)
    assert ns.numel() == 24


def test_batched_forward_accepts_per_batch_time_and_preserves_shape():
    ns = M.NetStim(N=2, interval=100.0, start=100.0, noise=0.0).batch(3).initialize()
    ns.schedule(1, 2.0)  # bare generator index applies to all batch elements

    spikes = ns(torch.tensor([2.0, 1.0, 2.0]))
    expected = torch.tensor(
        [[False, True], [False, False], [False, True]], dtype=torch.bool
    )
    assert spikes.shape == ns.shape == (3, 2)
    assert torch.equal(spikes.cpu(), expected)


def test_batched_schedule_full_coordinate_targets_single_state_element():
    ns = M.NetStim(N=2, interval=100.0, start=100.0, noise=0.0).batch(3).initialize()
    ns.schedule((1, 0), 2.0)

    spikes = ns(2.0)
    expected = torch.tensor(
        [[False, False], [True, False], [False, False]], dtype=torch.bool
    )
    assert torch.equal(spikes.cpu(), expected)
