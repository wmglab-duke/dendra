"""Mechanism replay preserves shared model streams and unrelated bank cadence."""

import pytest
import torch

import dendra as dn
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_mechanism_injection_execution import _InjectedLeak
from tests.func.test_poisson_invocation_execution import _native_state
from tests.func.test_stimulation_rng_sharing import (
    DT,
    _mixed_network,
    _network,
    _one_stream,
    _population,
)
from tests.test_native_mechanism_poisson_replay import _case
from tests.test_native_runtime_draw_tapes import (
    METHODS,
    _at,
    _functional_sources,
    _loss,
    _MeanGrid,
    _run,
    _state_loss,
)
from tests.test_native_runtime_draws import (
    _changed_draws,
)
from tests.test_native_runtime_draws import _environment as _environment
from tests.test_native_runtime_draws import (
    _forbid_owned_sampling,
)

pytestmark = pytest.mark.cpu
STEPS = 5


def _shared_case(kind):
    if kind == "population":
        model, extra, pulse = _population()
        return model, extra, pulse.generator
    if kind == "network":
        model, pulses = _network()
        return model, None, pulses[0].generator
    model, extra, pulses = _mixed_network()
    return model, extra, pulses[0].generator


@pytest.mark.parametrize("kind", ("population", "network", "mixed"))
@pytest.mark.parametrize("method", ("step", *METHODS))
def test_mechanism_replay_shares_one_rng_with_noise_sde_and_other_populations(
    kind, method, monkeypatch
):
    source, source_extra, _ = _shared_case(kind)
    model, extra, generator = _shared_case(kind)
    functional, tensors = dn.func.make_functional(source, dt=DT, extra=source_extra)
    steps = 1 if method == "step" else STEPS
    draws = _changed_draws(functional.sample_runtime_draws(tensors.state, steps=steps))
    assert not draws.rng["stimulation"]
    expected, _ = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        steps=steps,
        draws=draws,
    )
    with monkeypatch.context() as guard:
        _forbid_owned_sampling(guard, model, generator)
        if method == "step":
            model.step(
                **({"dt": DT} if kind == "population" else {}),
                draws=_at(draws, 0),
                extra=extra,
            )
        else:
            _run(model, method, DT, steps=steps, draws=draws, extra=extra)
    _close(_native_state(model, extra), expected)
    _close(generator.get_state(), _one_stream(expected))
    saved = _snapshot(model.state_dict_for_checkpoint())

    def selected(owner):
        return tuple(
            value
            for name, value in owner.named_parameters()
            if name.endswith(("waveform.amp", "offset_mu_param"))
        )

    references = _functional_sources(functional, tensors, selected(source))
    wanted = torch.autograd.grad(
        _state_loss(expected, kind != "population"), references
    )
    actual = torch.autograd.grad(_loss(model), selected(model))
    _close(actual, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in wanted)
    _assert_packet_equal(model.state_dict_for_checkpoint(), saved)
    _close(generator.get_state(), _one_stream(expected))

    receiver, receiving_extra, _ = _shared_case(kind)
    receiver.restore_dict_from_checkpoint(saved)
    following, _ = dn.func.run(
        functional,
        dn.func.Continuation(expected, tensors.parameters, tensors.constants),
        steps=1,
    )
    receiver.step(**({"dt": DT} if kind == "population" else {}), extra=receiving_extra)
    _close(_native_state(receiver, receiving_extra), following)


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("method", METHODS)
def test_mechanism_only_replay_keeps_deterministic_extra_grid_and_gradients(
    network, method, monkeypatch
):
    def make():
        cells, extras, waves = {}, {}, []
        for name in ("left", "right") if network else ("cell",):
            cell, _, _ = _case("repeated")
            waveform = _MeanGrid(0.7)
            field = torch.linspace(
                -0.2, 0.3, cell.v.numel(), dtype=cell.dtype()
            ).reshape(cell.shape)
            cells[name], extras[name] = cell, (field, waveform)
            waves.append(waveform)
        model = dn.Network(cells) if network else cells["cell"]
        if network:
            model.train()
            model.initialize(DT)
        return model, extras if network else extras["cell"], waves

    model, extra, waveforms = make()
    baseline, baseline_extra, baseline_waveforms = make()
    functional, tensors = dn.func.make_functional(model, dt=DT, extra=extra)
    draws = functional.sample_runtime_draws(tensors.state, steps=STEPS)
    observed, owned = [], {id(waveform) for waveform in waveforms}
    original = _MeanGrid.fn

    def observe(self, times):
        if id(self) in owned:
            observed.append(times.numel())
        return original(self, times)

    with monkeypatch.context() as guard:
        guard.setattr(_MeanGrid, "fn", observe)
        _run(model, method, DT, steps=STEPS, draws=draws, extra=extra)
    _run(baseline, method, DT, steps=STEPS, extra=baseline_extra)
    _assert_packet_equal(
        model.state_dict_for_checkpoint(), baseline.state_dict_for_checkpoint()
    )
    counts = [STEPS] if method == "run" else [3, 2]
    assert sorted(observed) == sorted(counts * len(waveforms))
    wanted = torch.autograd.grad(
        _loss(baseline), tuple(waveform.amp for waveform in baseline_waveforms)
    )
    actual = torch.autograd.grad(
        _loss(model), tuple(waveform.amp for waveform in waveforms)
    )
    _close(actual, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in wanted)


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
@pytest.mark.parametrize("network", (False, True))
def test_mps_mechanism_replay_preserves_checkpointed_values_and_gradients(network):
    def make():
        cells = {}
        generator = torch.Generator().manual_seed(719)
        for name in ("left", "right") if network else ("cell",):
            cell = dn.SingleCompartment(
                N=1, C=2, v_init=-47.0, integrator=dn.bwd_euler_sc(imem=False)
            )
            cell.insert(_InjectedLeak.rename("owner"))
            pulse = dn.mono_rect(amp=0.2, pw=0.045).poisson(
                interval=0.055,
                n=5,
                start=-0.035,
                noise=0.7,
                randomize_every_call=True,
                generator=generator,
            )
            cell[:, 0].inject(pulse + pulse)
            cells[name] = cell
        model = dn.Network(cells) if network else cells["cell"]
        model.train()
        model.initialize(DT) if network else model.initialize()
        return model

    with dn.ctx(DTYPE=torch.float32, DEVICE="mps"):
        source, model = make(), make()
        functional, tensors = dn.func.make_functional(source, dt=DT)
        draws = functional.sample_runtime_draws(tensors.state, steps=3)
        expected, _ = dn.func.run(
            functional,
            dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
            steps=3,
            draws=draws,
        )
        _run(model, "longrun_checkpointed", DT, steps=3, chunklength=2, draws=draws)
        torch.testing.assert_close(functional.extract(model).state, expected)
        saved = _snapshot(model.state_dict_for_checkpoint())

        def selected(owner):
            return tuple(
                value
                for name, value in owner.named_parameters()
                if name.endswith("amp")
            )

        wanted = torch.autograd.grad(
            _state_loss(expected, network),
            _functional_sources(functional, tensors, selected(source)),
        )
        actual = torch.autograd.grad(_loss(model), selected(model))
        torch.testing.assert_close(actual, wanted)
        assert all(value.isfinite().all() and value.abs().sum() > 0 for value in wanted)
        _assert_packet_equal(model.state_dict_for_checkpoint(), saved)
