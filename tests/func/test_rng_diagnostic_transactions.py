"""Optional RNG diagnostics restore ambient state at their probe boundaries."""

import random

import numpy as np
import pytest
import torch
from torch.utils import _pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._random_parameters import _current_draw_leaves
from dendra.func._runtime_noise import RuntimeNoisePlan
from dendra.models import random_parameters as random_core
from dendra.models.random_parameters import register_random_distribution
from dendra.models.rng import RNGModule
from tests.func.test_runtime_noise import _population as _noise_population
from tests.func.test_stimulation_rng_sampling_diagnostics import (
    DT,
)
from tests.func.test_stimulation_rng_sampling_diagnostics import (
    _population as _shared_population,
)

pytestmark = pytest.mark.cpu


def _snapshot():
    return torch.random.get_rng_state(), random.getstate(), np.random.get_state()


def _restore(snapshot):
    torch.random.set_rng_state(snapshot[0])
    random.setstate(snapshot[1])
    np.random.set_state(snapshot[2])


def _assert_snapshot(expected):
    actual = _snapshot()
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    assert actual[1] == expected[1]
    assert actual[2][0] == expected[2][0]
    np.testing.assert_array_equal(actual[2][1], expected[2][1])
    assert actual[2][2:] == expected[2][2:]


def _consume_globals():
    torch.rand(())
    random.random()
    np.random.random()


@pytest.fixture(autouse=True)
def _environment():
    snapshot, threads = _snapshot(), torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0,
                REQUIRE_GRAD=1,
                IMEM=False,
                DTYPE=torch.float64,
                DEVICE="cpu",
                CHECK_PURITY=False,
            ),
        ):
            yield
    finally:
        _restore(snapshot)
        torch.set_num_threads(threads)


def _random_probe(*, consumes=False, failure=None, checked=True):
    def sampler(rng, shape, *, device, dtype):
        draw = rng.rand(shape, device=device, dtype=dtype)
        if consumes:
            _consume_globals()
        if failure is not None:
            raise failure
        return draw

    distribution = register_random_distribution(
        "rng_diagnostic_transaction_probe",
        {},
        draw_sampler=sampler,
        transform=lambda parameters, draw: draw,
        overwrite=True,
    )
    sample = torch.zeros(2, dtype=torch.float64)
    rng = RNGModule(991, shape_p=(), shape_f=sample.shape)
    before = rng.get_extra_state()
    try:
        return _current_draw_leaves(
            distribution,
            rng,
            sample,
            _pytree.tree_structure(sample),
            (sample,),
            check_purity=checked,
        )
    finally:
        after = rng.get_extra_state()
        assert after["base_seed"] == before["base_seed"]
        torch.testing.assert_close(
            after["rng_state"], before["rng_state"], rtol=0, atol=0
        )


@pytest.mark.parametrize("raises", (False, True))
def test_random_draw_probe_restores_all_global_rngs_and_preserves_failure(raises):
    before = _snapshot()
    failure = RuntimeError("random draw body failure") if raises else None
    expected = RuntimeError if raises else dn.func.FunctionalizationError
    with pytest.raises(expected) as caught:
        _random_probe(consumes=True, failure=failure)
    if raises:
        assert caught.value is failure
    else:
        assert str(caught.value) == (
            "RAND draw sampler consumed global RNG state "
            "['global NumPy', 'global PyTorch', 'global Python']; "
            "use its explicit RNGModule"
        )
    _assert_snapshot(before)


@pytest.mark.parametrize("phase", ("draw", "transform"))
@pytest.mark.parametrize("raises", (False, True))
def test_noise_probe_restores_all_global_rngs_and_preserves_failure(
    phase, raises, monkeypatch
):
    source = _noise_population()
    name = "sample_runtime_noise_draw" if phase == "draw" else "evaluate_runtime_noise"
    original = getattr(random_core, name)
    failure = RuntimeError(f"noise {phase} body failure") if raises else None

    def probe(*args, **kwargs):
        value = original(*args, **kwargs)
        _consume_globals()
        if failure is not None:
            raise failure
        return value

    monkeypatch.setattr(random_core, name, probe)
    before = _snapshot()
    expected = RuntimeError if raises else dn.func.FunctionalizationError
    with dn.ctx(CHECK_PURITY=True), pytest.raises(expected) as caught:
        RuntimeNoisePlan(source)
    if raises:
        assert caught.value is failure
    else:
        assert str(caught.value) == (
            "NOISE draw sampler consumed global RNG; use its explicit RNGModule"
            if phase == "draw"
            else "NOISE transform must preserve its explicit parameters, draws, dt, "
            "and global RNG state"
        )
    _assert_snapshot(before)


@pytest.mark.parametrize("kind", ("random", "noise"))
def test_disabled_probe_scopes_do_not_inspect_or_restore_global_rng(kind, monkeypatch):
    from dendra.func import _rng_diagnostics as diagnostics

    source = _noise_population() if kind == "noise" else None

    def unexpected(*args, **kwargs):
        pytest.fail("Disabled optional RNG diagnostics inspected ambient state")

    for name in ("_global_rng_snapshot", "_global_rng_changes", "_restore_global_rng"):
        monkeypatch.setattr(diagnostics, name, unexpected)
    if kind == "random":
        _random_probe(checked=False)
    else:
        plan = RuntimeNoisePlan(source)
        assert plan.check_purity is False


@pytest.mark.parametrize("checked", (False, True))
def test_shared_sampler_uses_captured_policy_outside_creation_context(
    checked, monkeypatch
):
    from dendra.func import _rng_diagnostics as diagnostics

    source, pulse = _shared_population(raises=False)
    with dn.ctx(CHECK_PURITY=checked):
        functional, tensors = dn.func.make_functional(source, dt=DT)
    before, source_rng = _snapshot(), pulse.generator.get_state().clone()
    if not checked:

        def unexpected(*args, **kwargs):
            pytest.fail("A captured disabled policy inspected ambient RNG state")

        for name in (
            "_global_rng_snapshot",
            "_global_rng_changes",
            "_restore_global_rng",
        ):
            monkeypatch.setattr(diagnostics, name, unexpected)
    with dn.ctx(CHECK_PURITY=not checked):
        if checked:
            with pytest.raises(dn.func.FunctionalizationError) as caught:
                functional.sample_noise_draw_tape(tensors.state, 128)
            assert str(caught.value) == (
                "Stochastic draw sampler consumed global RNG; use its explicit RNGModule"
            )
            _assert_snapshot(before)
        else:
            functional.sample_noise_draw_tape(tensors.state, 128)
            assert not torch.equal(torch.random.get_rng_state(), before[0])
    torch.testing.assert_close(pulse.generator.get_state(), source_rng, rtol=0, atol=0)


def test_nested_scopes_restore_their_own_rng_entry_frame():
    from dendra.func._rng_diagnostics import _rng_diagnostic_scope

    before = _snapshot()
    with _rng_diagnostic_scope(torch.device("cpu"), enabled=True) as outer:
        _consume_globals()
        intermediate = _snapshot()
        with _rng_diagnostic_scope(torch.device("cpu"), enabled=True) as inner:
            _consume_globals()
        _assert_snapshot(intermediate)
    assert (
        inner.changes
        == outer.changes
        == {
            "global PyTorch",
            "global Python",
            "global NumPy",
        }
    )
    _assert_snapshot(before)


@pytest.mark.parametrize("backend", ("cuda", "mps"))
def test_scope_restores_selected_device_rng_without_requiring_accelerator(
    backend, monkeypatch
):
    from dendra.func._rng_diagnostics import _rng_diagnostic_scope

    device = torch.device("cuda:1" if backend == "cuda" else "mps")
    saved = torch.tensor((11, 23, 37), dtype=torch.uint8)
    current = {"state": saved.clone()}

    def get_state(*args):
        assert args == ((device,) if backend == "cuda" else ())
        return current["state"].clone()

    def set_state(frame, *args):
        assert args == ((device,) if backend == "cuda" else ())
        current["state"] = frame.clone()

    target = getattr(torch, backend)
    monkeypatch.setattr(target, "get_rng_state", get_state)
    monkeypatch.setattr(target, "set_rng_state", set_state)
    with _rng_diagnostic_scope(device, enabled=True) as diagnostic:
        current["state"][1] ^= 1
    assert diagnostic.changes == {backend.upper()}
    torch.testing.assert_close(current["state"], saved, rtol=0, atol=0)


def test_comparison_failure_still_restores_global_rng(monkeypatch):
    from dendra.func import _rng_diagnostics as diagnostics

    failure = RuntimeError("RNG comparison failed")

    def compare(*args, **kwargs):
        raise failure

    monkeypatch.setattr(diagnostics, "_global_rng_changes", compare)
    before = _snapshot()
    with pytest.raises(RuntimeError) as caught:
        with diagnostics._rng_diagnostic_scope(torch.device("cpu"), enabled=True):
            _consume_globals()
    assert caught.value is failure
    _assert_snapshot(before)
