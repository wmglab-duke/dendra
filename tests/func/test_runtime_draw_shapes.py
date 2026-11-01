"""Runtime Poisson layouts are independent of initializer reservation widths."""

import pytest
import torch

import dendra as dn
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_poisson_invocation_guards import _case, _RepeatedInitializer
from tests.func.test_poisson_stimulation import DT, _waveform
from tests.test_native_runtime_draws import _environment as _environment

pytestmark = pytest.mark.cpu


class _RepeatedRuntime(_RepeatedInitializer):
    def assigned_values(self, v, values):
        del values
        return {
            "i_inj": self.injection_values(v, t=self.t)
            + 0.25 * self.injection_values(v, t=self.t)
        }


def _plan(*, cached=False, waveform=None, mechanism=_RepeatedInitializer):
    source, pulse = _case(mechanism, waveform=waveform)
    source.train()
    if cached:
        source.cache("_steady_state")
    source.detach()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    return source, pulse, functional, tensors


@pytest.mark.parametrize("cached", (False, True), ids=("reserved", "cached"))
def test_runtime_packet_frame_replays_with_legacy_values_and_parameter_gradients(
    cached,
):
    source, pulse, functional, tensors = _plan(cached=cached)
    saved_source = _snapshot(source.state_dict_for_checkpoint())
    legacy = functional.sample_runtime_draws(tensors.state)
    saved_packet = _snapshot(legacy)
    packet = dn.func.sample_runtime_draws(functional, tensors.state)
    (entry,) = functional._poisson_plan.entries
    assert legacy.draws["stimulation"][entry.key].shape == (
        (pulse.n,) if cached else (3, pulse.n)
    )
    assert packet.draws["stimulation"][entry.key].shape == (pulse.n,)
    _close(packet.rng, legacy.rng)
    assert packet.draws["noise"] is legacy.draws["noise"] is None
    _close(
        packet.draws["stimulation"][entry.key],
        legacy.draws["stimulation"][entry.key]
        if cached
        else legacy.draws["stimulation"][entry.key][0],
    )

    expected, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state, draws=legacy
    )
    actual, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state, draws=packet
    )
    _close(actual, expected)
    amplitude = next(
        value
        for name, value in tensors.parameters.items()
        if name.endswith("waveform.amp")
    )
    wanted = torch.autograd.grad(
        expected["integrator"]["v"].square().mean(), amplitude, retain_graph=True
    )[0]
    observed = torch.autograd.grad(
        actual["integrator"]["v"].square().mean(), amplitude
    )[0]
    _close(observed, wanted)
    assert wanted.abs().sum() > 0
    _assert_packet_equal(legacy, saved_packet)
    _assert_packet_equal(source.state_dict_for_checkpoint(), saved_source)


@pytest.mark.parametrize("steps", (None, 0, 3), ids=("frame", "empty", "tape"))
@pytest.mark.parametrize("mechanism", (_RepeatedInitializer, _RepeatedRuntime))
def test_runtime_and_reserved_ingress_preserve_shared_leaves_and_waveform_gradients(
    steps, mechanism
):
    first, second = _waveform(seed=19), _waveform(seed=23)
    _source, _pulse, functional, tensors = _plan(
        waveform=first + second, mechanism=mechanism
    )
    legacy = functional.sample_runtime_draws(tensors.state, steps=steps)
    packet = dn.func.sample_runtime_draws(functional, tensors.state, steps=steps)
    keys = tuple(legacy.draws["stimulation"])
    assert len(keys) == 2
    runtime = 1 if mechanism is _RepeatedInitializer else 2
    shape = (first.n,) if runtime == 1 else (runtime, first.n)
    assert packet.draws["stimulation"][keys[0]].shape == (
        shape if steps is None else (steps, *shape)
    )
    bases = tuple(
        value.draws["stimulation"][keys[0]].detach().clone().requires_grad_()
        for value in (legacy, packet)
    )
    expected_draws = bases[0][:runtime] if steps is None else bases[0][:, :runtime]
    if runtime == 1:
        expected_draws = expected_draws.squeeze(0 if steps is None else 1)
    _close(bases[1], expected_draws)
    legacy = legacy._replace(
        draws={
            **legacy.draws,
            "stimulation": dict.fromkeys(keys, bases[0]),
        }
    )
    packet = packet._replace(
        draws={**packet.draws, "stimulation": dict.fromkeys(keys, bases[1])}
    )
    frozen = tuple((base.detach().clone(), base._version) for base in bases)
    results = []
    for value in (legacy, packet):
        if steps is None:
            state, _ = functional.prepare_and_step(
                tensors.parameters, tensors.constants, tensors.state, draws=value
            )
        else:
            state, _ = functional.prepare_and_rollout(
                tensors.parameters,
                tensors.constants,
                tensors.state,
                steps=steps,
                draws=value,
            )
        results.append(state)
    _close(results[1], results[0])
    for value, base, (saved, version) in zip(
        (legacy, packet), bases, frozen, strict=True
    ):
        assert all(leaf is base for leaf in value.draws["stimulation"].values())
        assert base._version == version
        _close(base, saved)
    if steps != 0:
        amplitudes = tuple(
            value
            for name, value in tensors.parameters.items()
            if name.endswith("waveform.amp")
        )
        assert len(amplitudes) == 2
        expected = torch.autograd.grad(
            results[0]["integrator"]["v"].square().mean(),
            amplitudes,
            retain_graph=True,
        )
        actual = torch.autograd.grad(
            results[1]["integrator"]["v"].square().mean(), amplitudes
        )
        _close(actual, expected)
        assert all(gradient.abs().sum() > 0 for gradient in expected)


@pytest.mark.parametrize("steps", (None, 0), ids=("frame", "empty_tape"))
def test_zero_event_runtime_layout_retains_sentinel_and_zero_step_state(steps):
    waveform = dn.mono_rect(amp=0.2, pw=0.045).poisson(
        interval=0.055,
        n=0,
        start=-0.035,
        noise=0.7,
        randomize_every_call=True,
        generator=torch.Generator().manual_seed(29),
    )
    _source, _pulse, functional, tensors = _plan(waveform=waveform)
    legacy = functional.sample_runtime_draws(tensors.state, steps=steps)
    packet = dn.func.sample_runtime_draws(functional, tensors.state, steps=steps)
    (value,) = packet.draws["stimulation"].values()
    assert value.shape == ((0,) if steps is None else (0, 0))
    if steps is None:
        expected, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, tensors.state, draws=legacy
        )
        actual, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, tensors.state, draws=packet
        )
    else:
        expected = tensors.state
        actual, _ = functional.prepare_and_rollout(
            tensors.parameters, tensors.constants, tensors.state, steps=0, draws=packet
        )
    _close(actual, expected)


@pytest.mark.parametrize("canonical_entry", (0, 1))
def test_mixed_runtime_and_reserved_entry_widths_reject_as_one_population_schema(
    canonical_entry,
):
    _source, _pulse, functional, tensors = _plan(
        waveform=_waveform(seed=31) + _waveform(seed=37)
    )
    legacy = functional.sample_runtime_draws(tensors.state)
    canonical = dn.func.sample_runtime_draws(functional, tensors.state)
    for packet in (legacy, canonical):
        functional.prepare_and_step(
            tensors.parameters, tensors.constants, tensors.state, draws=packet
        )
    mixed = _snapshot(legacy)
    key = tuple(mixed.draws["stimulation"])[canonical_entry]
    mixed.draws["stimulation"][key] = canonical.draws["stimulation"][key]
    with pytest.raises(
        ValueError, match="Poisson stimulation draw changed Tensor schema"
    ):
        functional.prepare_and_step(
            tensors.parameters, tensors.constants, tensors.state, draws=mixed
        )


@pytest.mark.parametrize("defect", ("width", "unused_nan"))
def test_runtime_ingress_validates_unused_reserved_rows_before_execution(
    defect,
):
    _source, _pulse, functional, tensors = _plan()
    legacy = _snapshot(functional.sample_runtime_draws(tensors.state))
    (key,) = legacy.draws["stimulation"]
    if defect == "width":
        legacy.draws["stimulation"][key] = legacy.draws["stimulation"][key][:2]
        message = "Poisson stimulation draw changed Tensor schema"
    else:
        legacy.draws["stimulation"][key][-1].fill_(torch.nan)
        message = "uniforms must be finite"
    with pytest.raises(ValueError, match=message):
        functional.prepare_and_step(
            tensors.parameters, tensors.constants, tensors.state, draws=legacy
        )


def test_legacy_stimulation_and_initialization_packets_keep_reserved_layouts():
    _source, pulse, functional, tensors = _plan()
    legacy = functional.sample_stimulation_draw(tensors.state)
    initial = functional.sample_stimulation_initialization_draw(tensors.state)
    canonical = dn.func.sample_runtime_draws(functional, tensors.state)
    assert all(value.shape == (3, pulse.n) for value in legacy.draws.values())
    assert all(value.shape == (3, pulse.n) for value in initial.draws.values())
    functional._poisson_plan.validate(legacy)
    functional._poisson_plan.validate(initial)
    with pytest.raises(
        ValueError, match="Poisson stimulation draw changed Tensor schema"
    ):
        functional.prepare_and_step(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            dn.func.StepInput(
                stimulation=dn.func.StimulationDraw(
                    canonical.draws["stimulation"], legacy.rng
                )
            ),
        )
