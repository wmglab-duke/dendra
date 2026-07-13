from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra.models.mod import expsyn
from dendra.models.networks import netcon_bitpack_ops as bitpack_ops

DT = 0.1
BACKENDS = ("dense", "sparse_calendar", "bitpacked_history")
WEIGHTS = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
DELAYS = torch.tensor([0.1, 0.2, 0.3, 0.1, 0.2, 0.3], dtype=torch.float64)


def _fan_network(backend: str):
    post = dn.Population(N=2, C=1, v_init=-65.0, dtype=torch.float64)
    post.insert(expsyn.rename("syn"), e=0.0, tau=1.0)
    stim = dn.NetStim(N=3, dtype=torch.float64)
    net = dn.Network(
        {"post": post},
        netstim=stim,
        track_netcon_events=False,
        netcon_delay_backend=backend,
    )
    net.connect_dense(
        net.netstim[:],
        post[:],
        post.mech.syn,
        threshold=None,
        weight=WEIGHTS,
        delay=DELAYS,
    )
    net.build(DT)
    net.init_synapses()
    return net, next(iter(net.synapses.values()))


def _run_tape(net, netcon, tape):
    trace = []
    for spikes in tape:
        net.netstim.spikes.copy_(torch.as_tensor(spikes, dtype=torch.bool))
        netcon.advance()
        trace.append(netcon.syn.g.detach().flatten().clone())
    return torch.stack(trace)


def _threshold_network(backend: str):
    pre = dn.Population(N=3, C=1, v_init=-65.0, dtype=torch.float64)
    post = dn.Population(N=2, C=1, v_init=-65.0, dtype=torch.float64)
    post.insert(expsyn.rename("syn"), e=0.0, tau=1.0)
    net = dn.Network(
        {"pre": pre, "post": post},
        track_netcon_events=False,
        netcon_delay_backend=backend,
    )
    net.connect_dense(
        pre[:],
        post[:],
        post.mech.syn,
        threshold=-30.0,
        weight=WEIGHTS,
        delay=DELAYS,
    )
    net.build(DT)
    net.init_synapses()
    return net, next(iter(net.synapses.values()))


def _reference_trace(tape):
    # Network.connect_dense orders edges source-major, then target-major.
    pre_idx = [0, 0, 1, 1, 2, 2]
    post_idx = [0, 1, 0, 1, 0, 1]
    delay_steps = [1, 2, 3, 1, 2, 3]
    pending = {}
    state = torch.zeros(2, dtype=torch.float64)
    trace = []
    for step, spikes in enumerate(tape):
        state = state + pending.pop(step, torch.zeros_like(state))
        trace.append(state.clone())
        for edge, (source, target, delay) in enumerate(
            zip(pre_idx, post_idx, delay_steps)
        ):
            if spikes[source]:
                due = pending.setdefault(step + delay, torch.zeros_like(state))
                due[target] += WEIGHTS[edge]
    return torch.stack(trace)


def test_all_cpu_backends_match_known_fan_in_out_tape_across_ring_wraparound():
    tape = (
        (True, True, False),
        (False, False, True),
        (True, False, True),
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (False, False, False),
        (False, False, False),
        (False, False, False),
    )
    expected = _reference_trace(tape)
    traces = {}

    for backend in BACKENDS:
        net, netcon = _fan_network(backend)
        traces[backend] = _run_tape(net, netcon, tape)

        assert torch.allclose(traces[backend], expected, atol=1e-6)
        assert netcon.current_time_step.item() == len(tape) % netcon.max_delay_steps

    assert torch.allclose(traces["dense"], traces["sparse_calendar"])
    assert torch.allclose(traces["dense"], traces["bitpacked_history"])


def test_thresholded_source_crossings_match_across_all_backends():
    voltage_tape = (
        (-65.0, -65.0, -65.0),
        (-20.0, -65.0, -20.0),
        (-10.0, -10.0, -20.0),  # source 0/2 stay high; source 1 crosses
        (-65.0, -10.0, -65.0),
        (-20.0, -65.0, -65.0),  # source 0 crosses a second time
        (-65.0, -65.0, -65.0),
        (-65.0, -65.0, -65.0),
        (-65.0, -65.0, -65.0),
    )
    traces = {}
    for backend in BACKENDS:
        net, netcon = _threshold_network(backend)
        trace = []
        for voltages in voltage_tape:
            net.pre.v.copy_(torch.tensor(voltages, dtype=torch.float64).view(3, 1))
            netcon.advance()
            trace.append(netcon.syn.g.detach().flatten().clone())
        traces[backend] = torch.stack(trace)

        if backend == "bitpacked_history":
            assert netcon._bitpack_mode == "threshold"
            assert netcon.bitpack_source_has_spiked.shape == (3,)

    assert torch.allclose(traces["dense"], traces["sparse_calendar"], atol=1e-6)
    assert torch.allclose(traces["dense"], traces["bitpacked_history"], atol=1e-6)


def test_scheduled_events_match_across_backends_with_collisions_and_negative_weight():
    traces = {}
    for backend in BACKENDS:
        net, netcon = _fan_network(backend)
        netcon.schedule(
            con_indices=[0, 3, 5, 0, 2],
            times_ms=[0.0, 0.1, 0.2, 0.4, 0.0],
            weight=torch.tensor([2.0, -0.5, 1.5, 0.25, 3.0]),
        )
        traces[backend] = _run_tape(net, netcon, [(False, False, False)] * 9)

    assert torch.allclose(traces["dense"], traces["sparse_calendar"], atol=1e-6)
    assert torch.allclose(traces["dense"], traces["bitpacked_history"], atol=1e-6)
    assert traces["dense"][-1].tolist() == pytest.approx([11.25, 7.0], abs=1e-6)


def test_bitpacked_history_crosses_the_63_bit_word_boundary():
    traces = {}
    active = torch.zeros(65, dtype=torch.bool)
    active[[0, 62, 63, 64]] = True
    tape = [active, torch.zeros_like(active)]

    for backend in ("dense", "bitpacked_history"):
        post = dn.Population(N=65, C=1, v_init=-65.0, dtype=torch.float64)
        post.insert(expsyn.rename("syn"), e=0.0, tau=1.0)
        stim = dn.NetStim(N=65, dtype=torch.float64)
        net = dn.Network(
            {"post": post},
            netstim=stim,
            netcon_delay_backend=backend,
        )
        net.connect_one_to_one(
            net.netstim[:],
            post[:],
            post.mech.syn,
            threshold=None,
            weight=torch.ones(65, dtype=torch.float64),
            delay=torch.full((65,), DT, dtype=torch.float64),
        )
        net.build(DT)
        net.init_synapses()
        netcon = next(iter(net.synapses.values()))
        traces[backend] = _run_tape(net, netcon, tape)

        if backend == "bitpacked_history":
            assert netcon.spike_history_packed.shape[1] == 2
            assert netcon.bitpack_source_word_idx[[0, 62, 63, 64]].tolist() == [
                0,
                0,
                1,
                1,
            ]

    assert torch.equal(traces["dense"], traces["bitpacked_history"])
    assert torch.equal(
        torch.nonzero(traces["dense"][1]).flatten(), active.nonzero().flatten()
    )


def test_failed_native_bitpack_delivery_does_not_contaminate_torch_fallback(
    monkeypatch,
):
    _, netcon = _fan_network("bitpacked_history")
    active_word = int(netcon.bitpack_source_bit_mask.sum().item())
    netcon.spike_history_packed.fill_(active_word)
    scratch = torch.empty(netcon._syn_numel, dtype=netcon.dtype)

    def partially_writing_failure(_cur_idx, output):
        output.fill_(1000.0)
        return False

    monkeypatch.setattr(
        netcon,
        "_try_bitpack_delivery_kernel",
        partially_writing_failure,
    )

    actual = netcon._bitpack_build_delivery_from_history(
        netcon.current_time_step,
        scratch,
    )
    expected = torch.zeros_like(scratch)
    expected.index_add_(0, netcon.post_idx, netcon.weight())

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, torch.tensor([9.0, 12.0], dtype=actual.dtype))


def test_bitpacked_state_cache_resumes_pending_intrinsic_and_scheduled_events():
    original, original_netcon = _fan_network("bitpacked_history")
    original_netcon.schedule(con_indices=[5], times_ms=[0.0], weight=2.0)
    _run_tape(
        original,
        original_netcon,
        [(True, True, False), (False, False, True)],
    )
    cache = original_netcon.state_cache()

    assert cache["param_invariant"] is True
    assert cache["backend_state"]["history_layout"] == "age"
    assert "source_spike_history_packed" in cache["backend_state"]
    assert cache["backend_state"]["sparse_calendar"]

    resumed, resumed_netcon = _fan_network("bitpacked_history")
    resumed_netcon.initialize_from_state_cache(cache, rebuild_delays=False)
    with torch.no_grad():
        resumed_netcon.syn.g.copy_(original_netcon.syn.g)

    continuation = (
        (True, False, True),
        (False, False, False),
        (False, True, False),
        (False, False, False),
    )
    expected = _run_tape(original, original_netcon, continuation)
    actual = _run_tape(resumed, resumed_netcon, continuation)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_bitpacked_eval_and_hard_source_history_training_have_forward_parity():
    tape = (
        (True, False, True),
        (False, True, False),
        (True, True, False),
        (False, False, False),
        (False, False, True),
        (False, False, False),
    )
    eval_net, eval_netcon = _fan_network("bitpacked_history")
    eval_trace = _run_tape(eval_net, eval_netcon, tape)

    train_net, train_netcon = _fan_network("bitpacked_history")
    train_netcon.train()
    train_netcon.set_diff_config(
        diff_weights=True,
        diff_delays=False,
        diff_spiking=False,
        diff_scheduled_times=False,
        train_delay_backend="source_history",
    )
    train_netcon.initialize(reinit_weights=False, reinit_delays=False)
    assert train_netcon.advance.__name__ == "advance_diff_source_history"
    assert train_netcon.delivery_buffer.shape[0] == 1

    train_trace = _run_tape(train_net, train_netcon, tape)
    assert torch.allclose(train_trace, eval_trace, atol=1e-6)

    train_netcon.eval()
    train_netcon.initialize(reinit_weights=False, reinit_delays=False)
    assert train_netcon.advance.__name__ == "advance_non_diff_bitpacked_history"


def test_bitpacked_backend_rejects_inexact_or_unsupported_contracts():
    with pytest.raises(ValueError, match="track_events=True"):
        post = dn.Population(N=1, C=1, dtype=torch.float64)
        post.insert(expsyn.rename("syn"))
        stim = dn.NetStim(N=1, dtype=torch.float64)
        net = dn.Network(
            {"post": post},
            netstim=stim,
            track_netcon_events=True,
            netcon_delay_backend="bitpacked_history",
        )
        net.connect_one_to_one(stim[:], post[:], post.mech.syn, threshold=None)
        net.build(DT)

    pre = dn.Population(N=1, C=1, v_init=-65.0, dtype=torch.float64)
    post = dn.Population(N=2, C=1, v_init=-65.0, dtype=torch.float64)
    post.insert(expsyn.rename("syn"))
    net = dn.Network(
        {"pre": pre, "post": post}, netcon_delay_backend="bitpacked_history"
    )
    net.connect_dense(
        pre[:],
        post[:],
        post.mech.syn,
        threshold=torch.tensor([-30.0, -20.0], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="threshold differs across outgoing edges"):
        net.build(DT)

    _, netcon = _fan_network("bitpacked_history")
    netcon.schedule(con_indices=[0], times_ms=[0.0])
    netcon.train()
    with pytest.raises(RuntimeError, match="intrinsic source-level events only"):
        netcon.set_diff_config(
            diff_delays=False,
            diff_spiking=False,
            diff_scheduled_times=False,
            train_delay_backend="source_history",
        )


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("sm_86", "8.6"),
        ("compute_90a", "9.0a"),
        ("12.1+PTX", "12.1+PTX"),
        ("75", "7.5"),
    ],
)
def test_bitpack_arch_tokens_are_normalized(raw, expected):
    assert bitpack_ops._parse_arch_token(raw) == expected


@pytest.mark.parametrize("raw", ["", "8", "abc", "8.x", "sm_"])
def test_bitpack_arch_tokens_reject_malformed_values(raw):
    with pytest.raises(ValueError, match="CUDA arch token|empty CUDA"):
        bitpack_ops._parse_arch_token(raw)


def test_bitpack_arch_lists_expand_names_deduplicate_and_emit_gencode():
    archs = bitpack_ops._split_arch_list("Ampere;8.6;sm_75;9.0+PTX")
    assert archs == ["7.5", "8.0", "8.6+PTX", "9.0+PTX"]
    assert bitpack_ops._cuda_gencode_flags(["8.6", "9.0+PTX"]) == [
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_90,code=compute_90",
        "-gencode=arch=compute_90,code=sm_90",
    ]


def test_auto_arch_resolution_clamps_newer_devices_and_includes_ptx(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device: [(8, 9), (9, 0)][device],
    )
    monkeypatch.setattr(
        bitpack_ops, "_compiled_sm_capabilities", lambda: [(8, 0), (8, 6)]
    )
    monkeypatch.delenv("DENDRA_NETCON_BITPACK_INCLUDE_PTX", raising=False)

    assert bitpack_ops._auto_cuda_arch_list("visible") == ["8.6+PTX"]
    with pytest.raises(ValueError, match="ARCH_MODE"):
        bitpack_ops._auto_cuda_arch_list("invalid")


def test_extension_identity_is_stable_and_sensitive_to_sources(tmp_path, monkeypatch):
    source = tmp_path / "kernel.cpp"
    source.write_text("one", encoding="utf8")
    build_dir = tmp_path / "build"
    monkeypatch.setenv("DENDRA_NETCON_BITPACK_BUILD_DIR", str(build_dir))
    monkeypatch.setenv("DENDRA_NETCON_BITPACK_EXTENSION_BASE_NAME", "custom-name")

    first = bitpack_ops._extension_identity([str(source)], ["8.6"], ["flag"])
    again = bitpack_ops._extension_identity([str(source)], ["8.6"], ["flag"])
    source.write_text("two", encoding="utf8")
    changed = bitpack_ops._extension_identity([str(source)], ["8.6"], ["flag"])

    assert first == again
    assert first[0].startswith("custom_name_")
    assert first[1] == build_dir
    assert first[0] != changed[0]
    assert first[2] != changed[2]


def test_bitpack_kernel_wrappers_validate_types_and_cpu_contract(monkeypatch):
    fake_extension = SimpleNamespace(pack_source_spikes=lambda *args: None)
    monkeypatch.setattr(bitpack_ops, "_load_extension", lambda: fake_extension)
    spikes = torch.tensor([True, False])
    history = torch.zeros((2, 1), dtype=torch.int64)
    step = torch.tensor([0], dtype=torch.long)

    with pytest.raises(TypeError, match="source_spikes"):
        bitpack_ops.pack_source_spikes(spikes.float(), history, step)
    with pytest.raises(TypeError, match="packed_history"):
        bitpack_ops.pack_source_spikes(spikes, history.float(), step)
    with pytest.raises(TypeError, match="current_time_step"):
        bitpack_ops.pack_source_spikes(spikes, history, step.float())
    with pytest.raises(RuntimeError, match="source_spikes must be a CUDA tensor"):
        bitpack_ops.pack_source_spikes(spikes, history, step)

    with pytest.raises(RuntimeError, match="require CUDA tensors"):
        bitpack_ops.build_delivery(
            history,
            step,
            torch.ones(2, dtype=torch.long),
            torch.zeros(2, dtype=torch.long),
            torch.ones(2, dtype=torch.long),
            torch.arange(2),
            torch.ones(2),
            torch.zeros(2),
        )


def test_bitpack_pack_wrapper_dispatches_after_validation(monkeypatch):
    calls = []
    fake_extension = SimpleNamespace(
        pack_source_spikes=lambda *args: calls.append(args),
    )
    monkeypatch.setattr(bitpack_ops, "_load_extension", lambda: fake_extension)
    monkeypatch.setattr(bitpack_ops, "_require_cuda_contiguous", lambda *args: None)
    spikes = torch.tensor([True, False])
    history = torch.zeros((2, 1), dtype=torch.int64)
    step = torch.tensor([0], dtype=torch.long)

    assert bitpack_ops.pack_source_spikes(spikes, history, step) is None
    assert calls == [(spikes, history, step)]
