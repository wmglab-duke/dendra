"""Deterministic correctness matrix for stimulation waveforms and assembly."""

from __future__ import annotations

import math

import pytest
import torch

import dendra as dn
from dendra.models.mod import pas
from dendra.models.stim.extra import Extra, op_mc, op_sc, ve_from_s_t
from dendra.models.stim.intra import (
    Intra,
    _canonicalize_index_for_index_put,
    avoid_smart_indexing,
    n,
)
from dendra.models.stim.waveform import (
    Waveform,
    arbitrary,
    bi_rect,
    bi_rect_balanced,
    bi_rect_symm,
    constant,
    cos,
    energy,
    mono_rect,
    sin,
)
from dendra.models.stim.waveform.core import Constant, Product, Reciprocal, Sum
from dendra.models.stim.waveform.implementations import (
    _integrate_square_last_dim,
    _make_time_grid,
)

DTYPE = torch.float64


class _Unimplemented(Waveform):
    pass


class _ScalarWaveform(Waveform):
    def fn(self, t):
        return t.new_tensor(2.0)


class _ShortWaveform(Waveform):
    def fn(self, t):
        return t[:-1]


class _ShapedWaveform(Waveform):
    def __init__(self, rows):
        super().__init__()
        self.rows = rows

    def fn(self, t):
        return t.new_zeros((self.rows, t.numel()))


class _RecordingWaveform(Waveform):
    def __init__(self):
        super().__init__()
        self.times = []

    def fn(self, t):
        self.times.append(t.detach().clone())
        return torch.zeros_like(t)


class _FakeModel:
    def __init__(self, shape=(2, 3), dtype=DTYPE):
        self.v = torch.zeros(shape, dtype=dtype)

    def device(self):
        return self.v.device

    def dtype(self):
        return self.v.dtype


def _population():
    pop = dn.Population(N=1, C=2, v_init=-65.0, dtype=DTYPE)
    pop.insert(pas, g=0.001, e=-70.0)
    pop[:, 0].inject(mono_rect(amp=2.0, delay=0.0, pw=0.02, tau=0.001))
    pop.build()
    pop.initialize()
    return pop


def test_base_waveform_requires_an_implementation_and_rejects_unknown_kwargs():
    with pytest.raises(NotImplementedError):
        _Unimplemented()(torch.tensor([0.0]))
    with pytest.raises(ValueError, match="Unknown parameter"):
        mono_rect(not_a_parameter=1.0)


def test_waveform_arithmetic_matches_direct_tensor_arithmetic():
    t = torch.tensor([0.0, 0.25, 0.75, 1.25], dtype=DTYPE)
    a = mono_rect(amp=2.0, pw=1.0)
    b = mono_rect(amp=4.0, delay=0.5, pw=1.0)

    torch.testing.assert_close((a + b)(t), a(t) + b(t))
    torch.testing.assert_close((a - b)(t), a(t) - b(t))
    torch.testing.assert_close((3.0 + a)(t), 3.0 + a(t))
    torch.testing.assert_close((3.0 - a)(t), 3.0 - a(t))
    torch.testing.assert_close((a * b)(t), a(t) * b(t))
    torch.testing.assert_close((3.0 * a)(t), 3.0 * a(t))
    torch.testing.assert_close((a / 2.0)(t), a(t) / 2.0)
    torch.testing.assert_close((-a)(t), -a(t))


def test_waveform_division_composition_and_flattening_are_stable():
    t = torch.tensor([0.25, 0.75], dtype=DTYPE)
    a = constant(value=8.0)
    b = constant(value=2.0)
    quotient = a / b
    reverse = 16.0 / a
    nested_sum = (a + b) - constant(value=1.0)
    nested_product = (2.0 * a) * b

    torch.testing.assert_close(quotient(t), torch.full_like(t, 4.0))
    torch.testing.assert_close(reverse(t), torch.full_like(t, 2.0))
    torch.testing.assert_close(nested_sum(t), torch.full_like(t, 9.0))
    torch.testing.assert_close(nested_product(t), torch.full_like(t, 32.0))
    assert isinstance(nested_sum, Sum) and len(nested_sum.waveforms) == 3
    assert isinstance(nested_product, Product) and nested_product.gain == 2.0
    assert "Sum(" in repr(nested_sum)
    assert "Product(" in repr(nested_product)
    assert "Reciprocal(" in repr(Reciprocal(a))
    assert repr(Reciprocal(a, eps=0.0)).startswith("Reciprocal(")


def test_waveform_arithmetic_rejects_unsupported_operands_and_bad_scale():
    waveform = constant(value=1.0)
    with pytest.raises(TypeError):
        _ = waveform + object()
    with pytest.raises(TypeError):
        _ = object() - waveform
    with pytest.raises(TypeError):
        _ = waveform * object()
    with pytest.raises(TypeError):
        _ = object() / waveform
    with pytest.raises(ValueError, match="Scale length"):
        Sum(waveform, waveform, scale=[1.0])
    with pytest.raises(ZeroDivisionError):
        _ = waveform / 0.0


def test_constant_primitives_preserve_shape_dtype_and_repr():
    t = torch.arange(4, dtype=DTYPE)
    internal = Constant(2.5)
    public = constant(value=2.5).to(dtype=DTYPE)

    torch.testing.assert_close(internal(t), torch.full_like(t, 2.5))
    torch.testing.assert_close(public(t), torch.full_like(t, 2.5))
    assert public(t).shape == t.shape
    assert public(t).dtype == DTYPE
    assert repr(internal) == "Constant(2.5)"


def test_fractional_arithmetic_promotes_integer_time_grids_without_truncation():
    t = torch.arange(3)
    torch.testing.assert_close(
        Constant(2.5)(t), torch.full((3,), 2.5, dtype=torch.get_default_dtype())
    )
    scaled = mono_rect(amp=1.0, pw=10.0) * 0.5
    torch.testing.assert_close(
        scaled(t), torch.full((3,), 0.5, dtype=torch.get_default_dtype())
    )


@pytest.mark.parametrize("n_time", [2, 3, 4])
def test_vector_parameters_never_alias_an_equal_length_time_axis(n_time):
    values = torch.tensor([1.0, 2.0, 3.0])
    t = torch.arange(n_time, dtype=values.dtype)
    expected = values.unsqueeze(-1).expand(-1, n_time)

    assert constant(value=values)(t).shape == (3, n_time)
    torch.testing.assert_close(constant(value=values)(t), expected)
    torch.testing.assert_close(mono_rect(amp=values, pw=10.0)(t), expected)


def test_sum_and_product_parameter_dtype_movement_and_gradients():
    amp = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
    waveform = (mono_rect(amp=amp, pw=1.0) + constant(value=1.0)) * 3.0
    waveform = waveform.to(dtype=DTYPE)
    t = torch.tensor([0.25, 0.75], dtype=DTYPE)
    loss = waveform(t).sum()
    loss.backward()

    assert next(waveform.parameters()).dtype == DTYPE
    assert amp.grad is not None
    torch.testing.assert_close(amp.grad, torch.tensor(6.0, dtype=DTYPE))


def test_expand_and_reshape_for_intra_transform_vector_parameters():
    waveform = mono_rect(amp=torch.tensor([1.0, 2.0]), pw=0.5)
    assert waveform.expand((3, 2)) is waveform
    assert waveform.amp.shape == (3, 2)
    assert waveform.reshape_for_intra() is waveform
    assert waveform.amp.shape == (1, 3, 2)
    assert waveform.pw.ndim == 0


def test_assemble_and_chunked_assembly_match_direct_evaluation():
    waveform = sin(amp=2.0, freq=1.0, off=1.0).to(dtype=DTYPE)
    direct = waveform(torch.arange(0.0, 1.0, 0.1, dtype=DTYPE))
    assembled = waveform.assemble(0.0, 1.0, 0.1)
    waveform._tstop = 1.0
    chunked = torch.cat(list(waveform.assemble_chunked(0.1, chunks=3)))

    torch.testing.assert_close(assembled, direct)
    torch.testing.assert_close(chunked, direct)


def test_parameterless_waveform_assembly_uses_default_dtype_fallback():
    waveform = Constant(3.0)
    assembled = waveform.assemble(0.0, 0.3, 0.1)
    waveform._tstop = 0.3
    chunked = torch.cat(list(waveform.assemble_chunked(0.1, chunks=2)))
    torch.testing.assert_close(assembled, torch.full((3,), 3.0))
    torch.testing.assert_close(chunked, assembled)


@pytest.mark.parametrize(
    "waveform,expected",
    [
        (
            mono_rect(amp=2.0, delay=1.0, pw=1.0),
            [0.0, 2.0, 2.0, 2.0],
        ),
        (
            bi_rect(amp1=-2.0, amp2=3.0, delay=1.0, pw1=0.5, pw2=0.5),
            [0.0, -2.0, 1.0, 3.0],
        ),
        (
            bi_rect_symm(amp=2.0, delay=1.0, pw=0.5),
            [0.0, 2.0, 0.0, -2.0],
        ),
    ],
)
def test_rectangular_waveform_boundary_times(waveform, expected):
    t = torch.tensor([0.999, 1.0, 1.5, 1.999])
    torch.testing.assert_close(waveform(t), torch.tensor(expected))


def test_balanced_biphasic_waveform_has_equal_and_opposite_charge():
    waveform = bi_rect_balanced(amp=2.0, pw1=0.5, pw2=1.0, interval=0.2)
    t = torch.arange(0.0, 1.7, 0.001, dtype=DTYPE)
    charge = torch.trapezoid(waveform(t), t)
    assert charge.item() == pytest.approx(0.0, abs=3e-3)


def test_rectangular_parameter_sweeps_broadcast_over_time():
    waveform = mono_rect(
        amp=torch.tensor([1.0, 2.0]),
        delay=torch.tensor([0.0, 0.5]),
        pw=torch.tensor([0.25, 0.25]),
    )
    t = torch.tensor([0.0, 0.2, 0.5, 0.7, 1.0])
    expected = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 2.0, 0.0]])
    torch.testing.assert_close(waveform(t), expected)


def test_rectangular_straight_through_gate_provides_finite_edge_gradients():
    delay = torch.nn.Parameter(torch.tensor(0.5, dtype=DTYPE))
    waveform = mono_rect(amp=2.0, delay=delay, pw=0.5, tau=0.1)
    waveform(torch.tensor([0.5], dtype=DTYPE)).sum().backward()

    assert delay.grad is not None
    assert torch.isfinite(delay.grad)
    assert delay.grad.abs() > 0


def test_sine_and_cosine_scalar_phase_delay_and_off_boundaries():
    t = torch.tensor([-0.1, 0.0, 0.25, 0.5, 0.75], dtype=DTYPE)
    sine = sin(amp=2.0, freq=1.0, delay=0.0, off=0.75, tau=0.01).to(dtype=DTYPE)
    cosine = cos(amp=3.0, freq=1.0, phase=math.pi / 2, off_after=0.75).to(dtype=DTYPE)

    expected = torch.tensor([0.0, 0.0, 2.0, 0.0, 0.0], dtype=DTYPE)
    torch.testing.assert_close(sine(t), expected, atol=1e-14, rtol=0.0)
    torch.testing.assert_close(cosine(t), -1.5 * sine(t), atol=2e-7, rtol=0.0)


def test_multitone_and_batched_oscillators_match_manual_sums():
    t = torch.linspace(0.0, 0.5, 9, dtype=DTYPE)
    amp = torch.tensor([[1.0, 0.5], [2.0, 1.0]], dtype=DTYPE)
    freq = torch.tensor([[1.0, 2.0], [1.5, 3.0]], dtype=DTYPE)
    waveform = sin(amp=amp, freq=freq, off=1.0)
    got = waveform(t)
    expected = (
        amp.unsqueeze(-1) * torch.sin(2 * torch.pi * freq.unsqueeze(-1) * t)
    ).sum(dim=1)

    assert got.shape == (2, t.numel())
    torch.testing.assert_close(got, expected)


def test_explicit_time_axis_oscillator_parameters_are_supported():
    t = torch.linspace(0.0, 0.5, 6)
    amp = torch.ones(2, 3, t.numel())
    freq = torch.tensor([1.0, 2.0, 3.0]).reshape(1, 3, 1)
    waveform = cos(amp=amp, freq=freq, off=torch.ones(1, 3, 1))
    expected = torch.cos(2 * torch.pi * freq * t).sum(dim=1).expand(2, -1)
    torch.testing.assert_close(waveform(t), expected)


def test_arbitrary_waveform_interpolates_batches_and_zeros_outside():
    t = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=DTYPE)
    waveform = arbitrary(
        tpoints=torch.tensor([0.0, 1.0], dtype=DTYPE),
        values=torch.tensor([[0.0, 2.0], [2.0, 4.0]], dtype=DTYPE),
    ).to(dtype=DTYPE)
    expected = torch.tensor(
        [[0.0, 0.0, 1.0, 2.0, 0.0], [0.0, 2.0, 3.0, 4.0, 0.0]],
        dtype=DTYPE,
    )
    torch.testing.assert_close(waveform(t), expected)


def test_arbitrary_waveform_gradients_flow_through_values_and_query_time():
    values = torch.nn.Parameter(torch.tensor([0.0, 2.0], dtype=DTYPE))
    t = torch.tensor([0.25, 0.75], dtype=DTYPE, requires_grad=True)
    waveform = arbitrary(tpoints=torch.tensor([0.0, 1.0]), values=values)
    waveform(t).sum().backward()

    torch.testing.assert_close(values.grad, torch.tensor([1.0, 1.0], dtype=DTYPE))
    torch.testing.assert_close(t.grad, torch.tensor([2.0, 2.0], dtype=DTYPE))


def test_repeat_delay_period_and_off_boundaries_are_exact():
    waveform = mono_rect(amp=2.0, pw=0.2).repeat(freq=2.0, delay=0.1, off=1.1)
    t = torch.tensor([0.0, 0.1, 0.29, 0.30, 0.60, 1.09, 1.10])
    expected = torch.tensor([0.0, 2.0, 2.0, 0.0, 2.0, 0.0, 0.0])
    torch.testing.assert_close(waveform(t), expected)
    assert repr(waveform).startswith("Repeat(")


@pytest.mark.parametrize("freq", [0.0, -1.0, torch.tensor([1.0, 0.0])])
def test_repeat_rejects_nonpositive_frequency(freq):
    with pytest.raises(ValueError, match="positive"):
        mono_rect().repeat(freq=freq)


def test_poisson_noise_zero_has_a_known_periodic_schedule_and_overlap_sum():
    waveform = mono_rect(amp=2.0, pw=0.6).poisson(
        interval=0.5, n=3, start=0.25, noise=0.0
    )
    expected_times = torch.tensor([0.25, 0.75, 1.25])
    torch.testing.assert_close(waveform._spike_times, expected_times)
    t = torch.tensor([0.0, 0.25, 0.80, 1.30, 1.90])
    torch.testing.assert_close(waveform(t), torch.tensor([0.0, 2.0, 4.0, 4.0, 0.0]))


def test_seeded_poisson_schedule_is_reused_until_regenerated():
    generator = torch.Generator().manual_seed(1234)
    waveform = mono_rect(pw=0.1).poisson(
        interval=1.0, n=4, generator=generator, randomize_every_call=False
    )
    first_schedule = waveform._spike_times.clone()
    t = torch.linspace(0.0, 10.0, 20)
    first = waveform(t)
    second = waveform(t)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(waveform._spike_times, first_schedule)

    waveform.regenerate_schedule_()
    assert not torch.equal(waveform._spike_times, first_schedule)
    assert repr(waveform).startswith("Poisson(")


def test_randomized_poisson_regenerates_each_call_and_preserves_dtype():
    generator = torch.Generator().manual_seed(99)
    waveform = (
        mono_rect(pw=0.1)
        .poisson(
            interval=1.0,
            n=4,
            generator=generator,
            randomize_every_call=True,
        )
        .to(dtype=DTYPE)
    )
    t = torch.linspace(0.0, 10.0, 20, dtype=DTYPE)
    waveform(t)
    first = waveform._spike_times.clone()
    waveform(t)
    second = waveform._spike_times.clone()

    assert first.dtype == second.dtype == DTYPE
    assert not torch.equal(first, second)


def test_poisson_empty_schedule_is_zero_and_can_be_reshaped_for_intra():
    waveform = mono_rect(pw=0.1).poisson(
        interval=1.0, n=3, start=2.0, off=1.0, noise=0.0
    )
    assert torch.isinf(waveform._spike_times).all()
    torch.testing.assert_close(waveform(torch.tensor([0.0, 1.0])), torch.zeros(2))
    assert waveform.reshape_for_intra() is waveform
    assert waveform._spike_times.shape == (1, 1, 1)


def test_poisson_intra_reshape_is_idempotent_and_survives_regeneration():
    waveform = mono_rect(pw=0.1).poisson(
        interval=1.0,
        n=4,
        generator=torch.Generator().manual_seed(123),
        randomize_every_call=True,
    )
    assert waveform.reshape_for_intra() is waveform
    first_shape = waveform._spike_times.shape
    assert waveform.reshape_for_intra() is waveform
    assert waveform._spike_times.shape == first_shape == (4, 1, 1)

    waveform.regenerate_schedule_()
    assert waveform._spike_times.shape == first_shape
    result = waveform(torch.linspace(0.0, 5.0, 8))
    assert result.shape == (1, 1, 8)
    assert waveform._spike_times.shape == first_shape


def test_poisson_keeps_spike_and_batched_waveform_axes_independent():
    pulse = mono_rect(amp=torch.tensor([1.0, 2.0]), pw=0.4)
    waveform = pulse.poisson(interval=1.0, n=2, start=0.0, noise=0.0)
    t = torch.tensor([0.0, 0.2, 0.5, 1.0, 1.2])

    expected = sum(pulse(t - onset) for onset in waveform._spike_times)
    result = waveform(t)
    assert result.shape == (2, t.numel())
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"interval": 0.0, "n": 1}, "interval"),
        ({"interval": -1.0, "n": 1}, "interval"),
        ({"interval": 1.0, "n": -1}, "n"),
        ({"interval": 1.0, "n": 1.5}, "n"),
        ({"interval": 1.0, "n": 1, "noise": -0.1}, "noise"),
        ({"interval": 1.0, "n": 1, "noise": 1.1}, "noise"),
        ({"interval": 1.0, "n": 1, "start": torch.inf}, "start"),
        ({"interval": 1.0, "n": 1, "off": float("nan")}, "off"),
        ({"interval": 1.0, "n": None, "off": torch.inf}, "finite"),
    ],
)
def test_poisson_rejects_invalid_schedules(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        mono_rect().poisson(**kwargs)


@pytest.mark.parametrize("method", ["trapezoid", "trapz", "left", "right"])
def test_energy_of_constant_waveform_is_exact_for_all_quadratures(method):
    waveform = constant(value=2.0).to(dtype=DTYPE)
    got = energy(
        waveform,
        tstart=0.0,
        tstop=2.0,
        dt=0.5,
        include_endpoint=True,
        time_scale=1.0,
        method=method,
    )
    torch.testing.assert_close(got, torch.tensor(8.0, dtype=DTYPE))


def test_energy_current_voltage_resistance_and_batched_outputs():
    t = torch.tensor([0.0, 1.0, 2.0], dtype=DTYPE)
    waveform = constant(value=torch.tensor([2.0, 4.0], dtype=DTYPE))
    normalized = energy(waveform, t=t, time_scale=1.0)
    current = energy(waveform, t=t, time_scale=1.0, resistance=3.0)
    voltage = energy(waveform, t=t, time_scale=1.0, resistance=4.0, mode="voltage")

    torch.testing.assert_close(normalized, torch.tensor([8.0, 32.0], dtype=DTYPE))
    torch.testing.assert_close(current, 3.0 * normalized)
    torch.testing.assert_close(voltage, normalized / 4.0)


def test_energy_grid_and_validation_edges():
    ref = torch.empty((), dtype=DTYPE)
    torch.testing.assert_close(
        _make_time_grid(0.0, 1.0, 0.4, ref),
        torch.tensor([0.0, 0.4, 0.8], dtype=DTYPE),
    )
    torch.testing.assert_close(
        _make_time_grid(0.0, 0.8, 0.4, ref, include_endpoint=True),
        torch.tensor([0.0, 0.4, 0.8], dtype=DTYPE),
    )
    with pytest.raises(ValueError, match="positive"):
        _make_time_grid(0.0, 1.0, 0.0, ref)
    with pytest.raises(ValueError, match="greater than"):
        _make_time_grid(1.0, 0.0, 0.1, ref)
    with pytest.raises(ValueError, match="Provide either"):
        energy(constant(value=1.0))
    with pytest.raises(ValueError, match="method"):
        energy(constant(value=1.0), t=torch.arange(3.0), method="simpson")
    with pytest.raises(ValueError, match="mode"):
        energy(
            constant(value=1.0),
            t=torch.arange(3.0),
            resistance=1.0,
            mode="power",
        )
    with pytest.raises(ValueError, match="time as its last dimension"):
        _integrate_square_last_dim(torch.ones(2), torch.ones(3))
    torch.testing.assert_close(
        _integrate_square_last_dim(torch.ones(2, 1), torch.zeros(1)),
        torch.zeros(2),
    )


def test_extra_contraction_operators_match_direct_broadcasted_products():
    space = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=DTYPE)
    time = torch.tensor([[2.0, 3.0], [5.0, 7.0]], dtype=DTYPE)
    expected_single = torch.stack([space * time[:, i, None] for i in range(2)])
    torch.testing.assert_close(op_sc(space, time), expected_single)
    assert op_sc(space, time).is_contiguous()

    space_mc = torch.stack([space, 2.0 * space])
    time_mc = torch.stack([time, 3.0 * time])
    expected_mc = op_sc(space, time) + op_sc(2.0 * space, 3.0 * time)
    torch.testing.assert_close(op_mc(space_mc, time_mc), expected_mc)
    assert op_mc(space_mc, time_mc).is_contiguous()


def test_public_extra_contraction_operators_support_batched_ellipses():
    space = torch.arange(2 * 3 * 4 * 5, dtype=DTYPE).reshape(2, 3, 4, 5)
    time = (1.0 + torch.arange(2 * 3 * 4 * 6, dtype=DTYPE)).reshape(2, 3, 4, 6)
    expected_single = time.movedim(-1, 0).unsqueeze(-1) * space

    actual_single = op_sc(space, time)
    assert actual_single.shape == (6, 2, 3, 4, 5)
    assert actual_single.is_contiguous()
    torch.testing.assert_close(actual_single, expected_single)

    space_mc = torch.stack((space, 2.0 * space))
    time_mc = torch.stack((time, 3.0 * time))
    expected_mc = (space_mc.unsqueeze(-2) * time_mc.unsqueeze(-1)).sum(0)
    expected_mc = expected_mc.movedim(-2, 0)

    actual_mc = op_mc(space_mc, time_mc)
    assert actual_mc.shape == (6, 2, 3, 4, 5)
    assert actual_mc.is_contiguous()
    torch.testing.assert_close(actual_mc, expected_mc)


@pytest.mark.parametrize("multicontact", [False, True])
def test_ve_from_s_t_expands_single_population_inputs(multicontact):
    n_pop = 3
    if multicontact:
        space = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
        time = torch.tensor([[[2.0, 5.0]], [[7.0, 11.0]]])
        expected = op_mc(space.expand(-1, n_pop, -1), time.expand(-1, n_pop, -1))
    else:
        space = torch.tensor([[1.0, 2.0]])
        time = torch.tensor([[2.0, 5.0]])
        expected = op_sc(space.expand(n_pop, -1), time.expand(n_pop, -1))
    torch.testing.assert_close(
        ve_from_s_t(space, time, n_pop, device="cpu", multicontact=multicontact),
        expected,
    )


def test_extra_functional_fields_sum_and_initialize_resets_iteration():
    fields = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=DTYPE),
        torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=DTYPE),
    ]
    extra = Extra(
        [
            (fields[0], constant(value=2.0)),
            (fields[1], mono_rect(amp=3.0, pw=0.5)),
        ]
    )
    model = _FakeModel(shape=(2, 2))
    times = torch.tensor([0.0, 0.5, 1.0], dtype=DTYPE)
    extra.initialize(model, times)

    torch.testing.assert_close(extra(), 2.0 * fields[0] + 3.0 * fields[1])
    torch.testing.assert_close(extra(), 2.0 * fields[0])
    extra.initialize(model, times)
    torch.testing.assert_close(extra(), 2.0 * fields[0] + 3.0 * fields[1])
    assert extra.fields.dtype == DTYPE
    assert all(next(w.parameters()).dtype == DTYPE for w in extra.waveforms)


def test_extra_broadcasts_scalar_and_population_specific_waveforms():
    fields = [torch.ones(2, 2, dtype=DTYPE), torch.ones(2, 2, dtype=DTYPE)]
    extra = Extra(
        [
            (fields[0], _ScalarWaveform()),
            (fields[1], constant(value=torch.tensor([3.0, 4.0]))),
        ]
    )
    extra.initialize(_FakeModel(shape=(2, 2)), torch.tensor([0.0, 0.5, 1.0]))
    torch.testing.assert_close(
        extra(), torch.tensor([[5.0, 5.0], [6.0, 6.0]], dtype=DTYPE)
    )


def test_extra_population_parameter_does_not_alias_equal_time_length():
    field = torch.ones(2, 3, dtype=DTYPE)
    extra = Extra([(field, constant(value=torch.tensor([3.0, 4.0], dtype=DTYPE)))])
    extra.initialize(_FakeModel(shape=(2, 3)), torch.tensor([0.0, 1.0]))
    expected = torch.tensor([[3.0] * 3, [4.0] * 3], dtype=DTYPE)
    torch.testing.assert_close(extra(), expected)
    torch.testing.assert_close(extra(), expected)


def test_extra_runtime_waveform_cache_is_excluded_from_state_dict():
    config = [(torch.ones(2, 2), constant(value=torch.tensor([2.0, 3.0])))]
    initialized = Extra(config)
    initialized.initialize(_FakeModel(shape=(2, 2)), torch.arange(3.0))
    state = initialized.state_dict()
    assert "waveform_stacked" not in state

    fresh = Extra(config)
    assert fresh.load_state_dict(state).missing_keys == []


def test_extra_precomputed_iterates_last_axis_and_resets():
    values = torch.arange(12, dtype=DTYPE).reshape(2, 2, 3)
    extra = Extra.from_precomputed(values)
    model = _FakeModel(shape=(2, 2))
    extra.initialize(model, torch.arange(3, dtype=DTYPE))

    torch.testing.assert_close(extra(), values[..., 0])
    torch.testing.assert_close(extra(), values[..., 1])
    extra.initialize(model, torch.arange(3, dtype=DTYPE))
    torch.testing.assert_close(extra(), values[..., 0])
    assert extra.precomputed.dtype == DTYPE
    extra()
    extra()
    with pytest.raises(IndexError, match="exhausted.*3 samples"):
        extra()


def test_extra_validates_spatial_broadcast_and_precomputed_time_length():
    model = _FakeModel(shape=(2, 3))
    wrong_space = Extra.from_precomputed(torch.zeros(4, 3, 2))
    with pytest.raises(ValueError, match="not broadcastable"):
        wrong_space.initialize(model, torch.arange(2.0))

    wrong_time = Extra.from_precomputed(torch.zeros(2, 3, 4))
    with pytest.raises(ValueError, match="one sample per timepoint"):
        wrong_time.initialize(model, torch.arange(3.0))

    functional = Extra([(torch.ones(4, 3), constant(value=1.0))])
    with pytest.raises(ValueError, match="not broadcastable"):
        functional.initialize(model, torch.arange(2.0))


def test_extra_validates_empty_and_malformed_configuration():
    with pytest.raises(ValueError, match="at least one"):
        Extra([])
    with pytest.raises(TypeError, match="Field"):
        Extra([([[1.0]], constant(value=1.0))])
    with pytest.raises(TypeError, match="Waveform"):
        Extra([(torch.ones(1, 1), object())])
    with pytest.raises((TypeError, ValueError), match="precomputed"):
        Extra.from_precomputed([[1.0, 2.0]])
    with pytest.raises(ValueError, match="time axis"):
        Extra.from_precomputed(torch.tensor(1.0))
    with pytest.raises(ValueError, match="same shape"):
        Extra(
            [
                (torch.ones(1, 2), constant(value=1.0)),
                (torch.ones(2, 2), constant(value=1.0)),
            ]
        )


def test_extra_validates_waveform_time_axis_and_broadcast_compatibility():
    model = _FakeModel(shape=(2, 2))
    times = torch.arange(3, dtype=DTYPE)
    wrong_time = Extra([(torch.ones(2, 2), _ShortWaveform())])
    with pytest.raises(ValueError, match="time as its last dimension"):
        wrong_time.initialize(model, times)

    incompatible = Extra(
        [
            (torch.ones(2, 2), _ShapedWaveform(2)),
            (torch.ones(2, 2), _ShapedWaveform(3)),
        ]
    )
    with pytest.raises(ValueError, match="compatible shapes"):
        incompatible.initialize(model, times)


@pytest.mark.parametrize(
    "idx,expected",
    [
        ((slice(None), slice(None)), torch.ones(2, 3)),
        ((0, slice(None)), torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])),
        (
            (slice(None), [0, 2]),
            torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
        ),
        (
            (torch.tensor([0, 1]), torch.tensor([2, 0])),
            torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        ),
        (
            torch.tensor([[True, False, True], [False, True, False]]),
            torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        ),
        ((Ellipsis, 1), torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])),
    ],
)
def test_intra_index_matrix_matches_native_tensor_selection(idx, expected):
    intra = Intra(_FakeModel(), [(constant(value=1.0), None, idx)])
    got = intra([torch.tensor(1.0, dtype=DTYPE)], intra.indices)
    torch.testing.assert_close(got, expected.to(DTYPE))


def test_intra_duplicate_indices_accumulate_and_empty_masks_are_safe():
    duplicate = Intra(
        _FakeModel(),
        [(constant(value=1.0), None, ([0, 0], [1, 1]))],
    )
    got = duplicate([torch.tensor(2.0, dtype=DTYPE)], duplicate.indices)
    expected = torch.zeros(2, 3, dtype=DTYPE)
    expected[0, 1] = 4.0
    torch.testing.assert_close(got, expected)

    empty = Intra(
        _FakeModel(),
        [(constant(value=1.0), None, torch.zeros(2, 3, dtype=torch.bool))],
    )
    torch.testing.assert_close(
        empty([torch.tensor(1.0, dtype=DTYPE)], empty.indices),
        torch.zeros(2, 3, dtype=DTYPE),
    )


def test_intra_init_whole_grid_matches_one_time_at_a_time():
    model = _FakeModel()
    intra = Intra(
        model,
        [
            (mono_rect(amp=2.0, pw=0.2), None, (slice(None), 0)),
            (constant(value=-1.0), None, (1, 2)),
        ],
    )
    t = torch.tensor([0.0, 0.1, 0.2], dtype=DTYPE)
    waves, indices = intra.init(t)
    whole = [intra([wave[i] for wave in waves], indices) for i in range(t.numel())]
    stepped = []
    for time in t:
        step_waves, step_indices = intra.init(time.reshape(1))
        stepped.append(intra([wave[0] for wave in step_waves], step_indices))
    torch.testing.assert_close(torch.stack(whole), torch.stack(stepped))


def test_intra_compartment_parameter_does_not_alias_equal_time_length():
    values = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)
    intra = Intra(
        _FakeModel(),
        [(constant(value=values), None, (0, slice(None)))],
    )
    waves, indices = intra.init(torch.arange(3, dtype=DTYPE))
    per_step = [intra([stim], indices) for stim in waves[0].unbind(-1)]
    expected = torch.zeros(2, 3, dtype=DTYPE)
    expected[0] = values
    for result in per_step:
        torch.testing.assert_close(result, expected)


def test_intra_reuses_one_linear_index_grid_across_injections(monkeypatch):
    calls = 0
    arange = torch.arange

    def counted_arange(*args, **kwargs):
        nonlocal calls
        calls += 1
        return arange(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", counted_arange)
    waveform = constant(value=1.0)
    Intra(
        _FakeModel(),
        [
            (waveform, None, (0, 0)),
            (waveform, None, (0, 1)),
            (waveform, None, (1, 2)),
        ],
    )
    assert calls == 1


def test_intra_validates_stimulation_type_and_index_errors():
    with pytest.raises(TypeError, match="Expected Waveform"):
        Intra(_FakeModel(), [(torch.tensor(1.0), None, (0, 0))])
    with pytest.raises(IndexError):
        _canonicalize_index_for_index_put((4, 0), (2, 3), torch.device("cpu"))
    with pytest.raises(IndexError):
        _canonicalize_index_for_index_put(
            torch.ones(2, 2, dtype=torch.bool), (2, 3), torch.device("cpu")
        )
    with pytest.raises(ValueError, match="non-scalar"):
        _canonicalize_index_for_index_put((), (), torch.device("cpu"))
    with pytest.raises(ValueError, match="linear_index has shape"):
        _canonicalize_index_for_index_put(
            (0, 0),
            (2, 3),
            torch.device("cpu"),
            linear_index=torch.zeros(3, 2, dtype=torch.long),
        )


def test_intra_small_helpers_have_explicit_contracts():
    assert avoid_smart_indexing([4]) == 4
    assert avoid_smart_indexing([4, 5]) == [4, 5]
    assert avoid_smart_indexing(None) is None
    assert avoid_smart_indexing(3) == 3
    assert n(3) == 1
    assert n([1, 2, 3]) == 3
    assert n(slice(1, 8, 2)) == 4
    assert n(slice(8, 1, -2)) == 4
    with pytest.raises(ValueError, match="Unbounded"):
        n(slice(None))
    with pytest.raises(ValueError, match="Unsupported"):
        n(torch.tensor([1]))


def test_population_run_step_and_longrun_are_equivalent_with_intra_stimulation():
    run_pop = _population()
    step_pop = _population()
    long_pop = _population()

    run_pop.run(tstop=0.04, dt=0.01)
    for _ in range(4):
        step_pop.step(dt=0.01)
    long_pop.longrun(tstop=0.04, chunklength=3, dt=0.01)

    torch.testing.assert_close(step_pop.v, run_pop.v)
    torch.testing.assert_close(long_pop.v, run_pop.v)
    torch.testing.assert_close(step_pop.t, run_pop.t)
    torch.testing.assert_close(long_pop.t, run_pop.t)


@pytest.mark.parametrize("runner", ["run", "longrun", "longrun_checkpointed"])
def test_nonzero_origin_intra_and_functional_extra_receive_exact_time_grid(runner):
    intra_waveform = _RecordingWaveform()
    extra_waveform = _RecordingWaveform()
    pop = dn.Population(N=1, C=2, v_init=-65.0, dtype=DTYPE)
    pop.insert(pas, g=0.001, e=-70.0)
    pop[:, 0].inject(intra_waveform)
    pop.build()
    pop.initialize()

    for dt in (0.04, 0.04, 0.09):
        pop.step(dt=dt)
    intra_waveform.times.clear()

    start = pop.t.detach().clone()
    duration = 0.07
    dt = 0.01
    expected = start + torch.arange(7, dtype=DTYPE) * dt
    extra = (torch.zeros_like(pop.v), extra_waveform)

    if runner == "run":
        pop.run(tstop=duration, dt=dt, extra=extra)
    elif runner == "longrun":
        pop.longrun(tstop=duration, dt=dt, chunklength=3, extra=extra)
    else:
        pop.longrun_checkpointed(
            tstop=duration,
            dt=dt,
            chunklength=3,
            extra=extra,
            safe_checkpoint=True,
        )

    recorded_intra = torch.cat(intra_waveform.times)
    recorded_extra = torch.cat(extra_waveform.times)
    torch.testing.assert_close(recorded_intra, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(recorded_extra, expected, rtol=0.0, atol=0.0)
