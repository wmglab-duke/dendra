"""Regression tests for the functional Waveform purity contract."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from dendra.models.stim.waveform import Waveform
from dendra.models.stim.waveform.core import (
    Constant,
    Product,
    Reciprocal,
    Sum,
    _poisson,
    _randomized_poisson,
    _repeat,
)
from dendra.models.stim.waveform.implementations import (
    arbitrary,
    bi_rect,
    bi_rect_balanced,
    bi_rect_symm,
    constant,
    cos,
    mono_rect,
    sin,
)
from dendra.utils import interp1d

DTYPE = torch.float64
DT = 0.025


class _UnmarkedSin(sin):
    """A subclass whose inherited marker must not count as an opt-in."""


class _MarkedSin(sin):
    FUNCTIONAL_PURE = True


class _ModuleSpoofedWaveform(Waveform):
    """An unmarked extension pretending to live in a built-in module."""

    __module__ = "dendra.models.stim.waveform.implementations"

    def fn(self, t):
        return torch.ones_like(t)


class _PythonStateMutatingChild(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # This name is an inert constructor mirror only for Dendra's
        # SimpleParameterized classes, not generic nn.Module state.
        self.params = {"calls": 0}

    def forward(self, t):
        self.params["calls"] += 1
        return torch.ones_like(t)


class _MarkedWaveformWithMutatingChild(Waveform):
    FUNCTIONAL_PURE = True

    def __init__(self):
        super().__init__()
        self.child = _PythonStateMutatingChild()

    def fn(self, t):
        return self.child(t)


class _MarkedWaveformWithNumpyNaN(Waveform):
    FUNCTIONAL_PURE = True

    def __init__(self):
        super().__init__()
        self.sentinel = np.float32(np.nan)

    def fn(self, t):
        return torch.ones_like(t)


PURE_CONCRETE_TYPES = (
    Constant,
    Reciprocal,
    _repeat,
    Sum,
    Product,
    _poisson,
    sin,
    cos,
    mono_rect,
    bi_rect,
    bi_rect_balanced,
    bi_rect_symm,
    arbitrary,
    constant,
)


def _model():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE):
        model = dn.Unmyelinated(
            [2.0, 2.5],
            L=4.0,
            dx=1.0,
            v_init=-65.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(pas, g=1.0e-4, e=-65.0)
        model.initialize()
    return model


def _fixed_poisson():
    return dn.mono_rect(amp=0.2, pw=0.05).poisson(
        interval=0.1,
        n=4,
        randomize_every_call=False,
        generator=torch.Generator().manual_seed(1203),
    )


PURE_WAVEFORM_FACTORIES = (
    pytest.param(lambda: Constant(0.2), id="core-constant"),
    pytest.param(lambda: Reciprocal(dn.constant(value=2.0)), id="reciprocal"),
    pytest.param(
        lambda: dn.mono_rect(amp=0.2, pw=0.05).repeat(freq=2.0),
        id="repeat",
    ),
    pytest.param(
        lambda: Sum(dn.constant(value=0.1), dn.constant(value=0.2)),
        id="sum",
    ),
    pytest.param(
        lambda: Product(dn.constant(value=0.2), dn.constant(value=0.5)),
        id="product",
    ),
    pytest.param(_fixed_poisson, id="fixed-poisson"),
    pytest.param(lambda: dn.sin(amp=0.2, freq=0.7), id="sin"),
    pytest.param(lambda: dn.cos(amp=0.2, freq=0.7), id="cos"),
    pytest.param(lambda: dn.mono_rect(amp=0.2, pw=0.05), id="mono-rect"),
    pytest.param(
        lambda: dn.bi_rect(amp1=0.2, amp2=-0.1, pw1=0.05, pw2=0.05),
        id="bi-rect",
    ),
    pytest.param(
        lambda: dn.bi_rect_balanced(amp=0.2, pw1=0.05, pw2=0.1),
        id="bi-rect-balanced",
    ),
    pytest.param(
        lambda: dn.bi_rect_symm(amp=0.2, pw=0.05),
        id="bi-rect-symm",
    ),
    pytest.param(
        lambda: dn.arbitrary(
            tpoints=[0.0, 0.04, 0.11, 0.2],
            values=[0.0, 0.2, -0.1, 0.0],
        ),
        id="arbitrary",
    ),
    pytest.param(lambda: dn.constant(value=0.2), id="public-constant"),
)


def test_concrete_builtin_waveforms_own_an_explicit_purity_marker():
    assert Waveform.__dict__["FUNCTIONAL_PURE"] is False
    assert _randomized_poisson.__dict__["FUNCTIONAL_PURE"] is False

    for waveform_type in PURE_CONCRETE_TYPES:
        assert waveform_type.__dict__.get("FUNCTIONAL_PURE") is True, (
            f"{waveform_type.__module__}.{waveform_type.__qualname__} must own "
            "FUNCTIONAL_PURE = True"
        )


def test_poisson_factory_separates_fixed_and_randomized_evaluation_contracts():
    fixed_generator = torch.Generator().manual_seed(7)
    fixed = dn.mono_rect(amp=1.0, pw=0.02).poisson(
        interval=0.1,
        n=5,
        generator=fixed_generator,
    )
    randomized_generator = torch.Generator().manual_seed(7)
    randomized = dn.mono_rect(amp=1.0, pw=0.02).poisson(
        interval=0.1,
        n=5,
        randomize_every_call=True,
        generator=randomized_generator,
    )

    assert type(fixed) is _poisson
    assert fixed.randomize_every_call is False
    assert type(randomized) is _randomized_poisson
    assert randomized.randomize_every_call is True

    times = torch.linspace(0.0, 1.0, 101)
    fixed_schedule = fixed._spike_times.clone()
    fixed_rng_state = fixed_generator.get_state().clone()
    first = fixed(times)
    second = fixed(times)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(fixed._spike_times, fixed_schedule)
    assert torch.equal(fixed_generator.get_state(), fixed_rng_state)

    randomized_schedule = randomized._spike_times.clone()
    randomized_rng_state = randomized_generator.get_state().clone()
    randomized(times)
    assert not torch.equal(randomized._spike_times, randomized_schedule)
    assert not torch.equal(randomized_generator.get_state(), randomized_rng_state)


def test_private_legacy_poisson_constructor_preserves_randomized_mode():
    waveform = _poisson(
        dn.mono_rect(amp=1.0, pw=0.02),
        interval=0.1,
        n=5,
        randomize_every_call=True,
        generator=torch.Generator().manual_seed(11),
    )

    assert type(waveform) is _randomized_poisson
    assert waveform.randomize_every_call is True

    waveform.randomize_every_call = False
    assert type(waveform) is _poisson
    assert waveform.randomize_every_call is False

    waveform.randomize_every_call = True
    assert type(waveform) is _randomized_poisson
    assert waveform.randomize_every_call is True


def test_legacy_randomized_poisson_pickle_restores_stateful_concrete_type():
    legacy = dn.mono_rect(amp=1.0, pw=0.02).poisson(
        interval=0.1,
        n=5,
        generator=torch.Generator().manual_seed(31),
    )
    assert type(legacy) is _poisson

    # Before the fixed/randomized type split, both modes were serialized as
    # _poisson and this flag selected the call-time behavior.
    legacy.__dict__["randomize_every_call"] = True
    assert type(legacy) is _poisson
    restored = pickle.loads(pickle.dumps(legacy))

    assert type(restored) is _randomized_poisson
    assert restored.randomize_every_call is True
    schedule = restored._spike_times.clone()
    rng_state = restored.generator.get_state().clone()
    restored(torch.linspace(0.0, 1.0, 101))
    assert not torch.equal(restored._spike_times, schedule)
    assert not torch.equal(restored.generator.get_state(), rng_state)


@pytest.mark.parametrize("waveform_factory", PURE_WAVEFORM_FACTORIES)
def test_every_pure_builtin_waveform_functionalizes(waveform_factory):
    model = _model()
    waveform = waveform_factory()
    model[:, 1].inject(waveform)

    functional, _tensors = dn.func.make_functional(model, dt=DT)
    stimulation = functional.intra.extract()
    times = torch.tensor([0.0, 0.05, 0.125], dtype=DTYPE)
    assembled = functional.intra.assemble_tensors(stimulation, times)

    assert functional.intra.enabled
    assert assembled is not None
    assert assembled.shape == (times.numel(), *model.shape)
    assert torch.isfinite(assembled).all()


def test_subclass_of_marked_builtin_must_redeclare_purity():
    unmarked = _model()
    unmarked[:, 1].inject(_UnmarkedSin(amp=0.2, freq=0.7))
    with pytest.raises(dn.func.FunctionalizationError, match="FUNCTIONAL_PURE"):
        dn.func.make_functional(unmarked, dt=DT)

    marked = _model()
    marked[:, 1].inject(_MarkedSin(amp=0.2, freq=0.7))
    functional, tensors = dn.func.make_functional(marked, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    state, _auxiliary = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
    )
    assert torch.isfinite(state["integrator"]["v"]).all()


@pytest.mark.parametrize(
    "waveform",
    [
        pytest.param(
            Sum(dn.constant(value=0.1), _UnmarkedSin(amp=0.2, freq=0.7)),
            id="unmarked-composite-child",
        ),
        pytest.param(_ModuleSpoofedWaveform(), id="spoofed-builtin-module"),
    ],
)
def test_purity_validation_is_recursive_and_does_not_trust_module_names(waveform):
    model = _model()
    model[:, 1].inject(waveform)

    with pytest.raises(dn.func.FunctionalizationError, match="FUNCTIONAL_PURE"):
        dn.func.make_functional(model, dt=DT)


def test_purity_audit_includes_non_waveform_module_children():
    model = _model()
    waveform = _MarkedWaveformWithMutatingChild()
    model[:, 1].inject(waveform)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="mutated Python instance state",
    ):
        dn.func.make_functional(model, dt=DT)

    # The audit runs against a private clone, never the user's Waveform.
    assert waveform.child.params["calls"] == 0


def test_interp1d_mixed_dtype_preserves_allocation_and_out_contracts():
    x = torch.tensor([0.0, 0.25, 1.0], dtype=torch.float32)
    y = torch.tensor([1.0, 3.0, 9.0], dtype=torch.float64)
    query = torch.tensor([0.125, 0.625], dtype=torch.float32)

    allocated = interp1d(x, y, query, uniform="never")
    assert allocated.dtype == x.dtype
    torch.testing.assert_close(
        allocated,
        torch.tensor([2.0, 6.0], dtype=x.dtype),
    )

    out = torch.empty(query.shape, dtype=torch.float64)
    returned = interp1d(x, y, query, out=out, uniform="never")
    assert returned.dtype == out.dtype
    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(
        returned,
        torch.tensor([2.0, 6.0], dtype=out.dtype),
    )


def test_interp1d_respects_ambient_no_grad():
    x = torch.tensor([0.0, 0.25, 1.0], requires_grad=True)
    y = torch.tensor([1.0, 3.0, 9.0], requires_grad=True)
    query = torch.tensor([0.125, 0.625], requires_grad=True)

    with torch.no_grad():
        result = interp1d(x, y, query, uniform="never")

    assert result.requires_grad is False


def test_purity_audit_accepts_deterministic_nan_values():
    model = _model()
    model[:, 1].inject(dn.constant(value=torch.nan))

    functional, _tensors = dn.func.make_functional(model, dt=DT)
    stimulation = functional.intra.extract()
    assembled = functional.intra.assemble_tensors(
        stimulation,
        torch.tensor([0.0, 0.05], dtype=DTYPE),
    )

    assert assembled is not None
    assert torch.isnan(assembled).any()


def test_purity_audit_accepts_stable_numpy_nan_state():
    model = _model()
    model[:, 1].inject(_MarkedWaveformWithNumpyNaN())

    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    state, _auxiliary = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
    )

    assert torch.isfinite(state["integrator"]["v"]).all()


def test_arbitrary_waveform_composes_with_functional_transforms_and_compile():
    with dn.ctx(REQUIRE_GRAD=1, DTYPE=DTYPE):
        waveform = dn.arbitrary(
            tpoints=torch.tensor([0.0, 0.5, 1.0, 1.5], dtype=DTYPE),
            values=torch.tensor([0.0, 0.25, 1.0, 2.25], dtype=DTYPE),
        ).double()

    parameters = dict(waveform.named_parameters())
    values = parameters["values"]
    times = torch.tensor([-0.1, 0.25, 0.75, 1.25, 1.6], dtype=DTYPE)

    def response(candidate_values):
        local_parameters = dict(parameters)
        local_parameters["values"] = candidate_values
        return torch.func.functional_call(
            waveform,
            local_parameters,
            (times,),
            strict=True,
        )

    reverse = torch.func.jacrev(response)(values)
    with torch_compiler_warning_context():
        forward = torch.func.jacfwd(response)(values)
    torch.testing.assert_close(reverse, forward, rtol=1.0e-12, atol=1.0e-12)

    batch = torch.stack((values, values + 0.2, -0.5 * values))
    vmapped = torch.vmap(response)(batch)
    expected_batch = torch.stack(tuple(response(row) for row in batch))
    torch.testing.assert_close(vmapped, expected_batch, rtol=1.0e-12, atol=1.0e-12)

    def tpoint_response(candidate_tpoints):
        local_parameters = dict(parameters)
        local_parameters["tpoints"] = candidate_tpoints
        return torch.func.functional_call(
            waveform,
            local_parameters,
            (times,),
            strict=True,
        )

    (eager_tpoint_gradient,) = torch.autograd.grad(
        waveform(times).sum(),
        (parameters["tpoints"],),
        retain_graph=True,
    )
    reverse_tpoints = torch.func.jacrev(tpoint_response)(parameters["tpoints"])
    with torch_compiler_warning_context():
        forward_tpoints = torch.func.jacfwd(tpoint_response)(parameters["tpoints"])
    torch.testing.assert_close(
        reverse_tpoints,
        forward_tpoints,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    transformed_tpoint_gradient = reverse_tpoints.sum(dim=0)
    torch.testing.assert_close(
        eager_tpoint_gradient,
        transformed_tpoint_gradient,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    tpoint_batch = torch.stack(
        (
            parameters["tpoints"],
            torch.tensor([0.0, 0.25, 0.8, 1.5], dtype=DTYPE),
            torch.tensor([0.0, 0.4, 1.0, 1.6], dtype=DTYPE),
        )
    )
    vmapped_tpoints = torch.vmap(tpoint_response)(tpoint_batch)
    expected_tpoints = torch.stack(tuple(tpoint_response(row) for row in tpoint_batch))
    torch.testing.assert_close(
        vmapped_tpoints,
        expected_tpoints,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    compiled = torch.compile(response, backend="aot_eager", fullgraph=True)
    compiled_reverse = torch.compile(
        torch.func.jacrev(response),
        backend="aot_eager",
        fullgraph=True,
    )
    compiled_tpoint_reverse = torch.compile(
        torch.func.jacrev(tpoint_response),
        backend="aot_eager",
        fullgraph=True,
    )
    with torch_compiler_warning_context():
        torch.testing.assert_close(
            compiled(values), response(values), rtol=1.0e-12, atol=1.0e-12
        )
        torch.testing.assert_close(
            compiled_reverse(values), reverse, rtol=1.0e-12, atol=1.0e-12
        )
        torch.testing.assert_close(
            compiled_tpoint_reverse(parameters["tpoints"]),
            reverse_tpoints,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
