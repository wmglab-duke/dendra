"""Population-owned cache starts detach warm-up graphs and own broadcast buffers."""

from contextlib import nullcontext

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from tests.func.test_membrane_current import _model
from tests.networks.test_population_cache_restore import _broadcast_buffers, _snapshot

pytestmark = pytest.mark.cpu
DT = 0.01
DTYPE = torch.float64


@pytest.fixture(autouse=True)
def _execution_environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch_compiler_warning_context():
            yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("training_warmup", (False, True))
@pytest.mark.parametrize("gradient_warmup", (False, True))
def test_steady_restore_cuts_warmup_graphs_but_keeps_parameters_and_runtime_gradients(
    training_warmup, gradient_warmup
):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE, DEVICE="cpu"):
        model = dn.SingleCompartment(
            N=1, C=1, integrator=dn.bwd_euler_sc(imem=False), v_init=-50.0
        )
        model.insert(pas, g=0.001, e=-65.0)
        model.train(training_warmup)
        with nullcontext() if gradient_warmup else torch.no_grad():
            model.steady_state(dt=DT, tstop=2 * DT, with_ve=False, with_intra=False)
        mechanism = next(iter(model.mech.mechanisms.values()))
        assert mechanism.g.requires_grad is gradient_warmup
        parameters = dict(model.named_parameters())
        model.train()
        model.initialize()

        assert all(
            dict(model.named_parameters())[name] is value
            for name, value in parameters.items()
        )
        assert mechanism.g.grad_fn is None
        assert not mechanism.g.requires_grad
        assert not model.v.requires_grad
        drive = torch.full_like(model.v, 1e-9, requires_grad=True)
        timestep = model.v.new_tensor(DT)
        model.integrator._initialize(model, timestep, force=True)
        model.integrator.step(model, timestep, intra=drive)
        drive_gradient, parameter_gradient = torch.autograd.grad(
            model.v.sum() + mechanism.g_param.sum(), (drive, mechanism.g_param)
        )
        assert torch.isfinite(drive_gradient).all()
        assert drive_gradient.abs().sum() > 0
        torch.testing.assert_close(
            parameter_gradient, torch.ones_like(mechanism.g_param)
        )


class _Add(torch.nn.Module):
    def forward(self, value, increment):
        return (value + increment,)


def test_steady_restore_detaches_before_legacy_post_hooks_and_skips_structured_post():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE, DEVICE="cpu"):
        model = dn.SingleCompartment(N=1, C=1, v_init=-50.0)
        model.insert(pas)
        model.register_post_initialize_transform(
            "raise_voltage",
            _Add(),
            reads=("state.integrator.v",),
            writes=("state.integrator.v",),
            inputs={"increment": torch.tensor(5.0, dtype=DTYPE)},
        )
        model.initialize()
        model.cache("_steady_state")
        saved = model.v.detach().clone()
        observed = []

        def legacy_post(population):
            observed.append(
                (
                    torch.is_grad_enabled(),
                    population.cm.requires_grad,
                    population.v.detach().clone(),
                )
            )
            population.v.add_(1)

        model.register_post_initialize_hook(legacy_post)
        model.initialize()

    assert len(observed) == 1
    assert observed[0][:2] == (False, False)
    torch.testing.assert_close(observed[0][2], saved)
    torch.testing.assert_close(model.v, saved + 1)
    assert model.initializing_from_state_cache
    assert model._integrator_reinit_pending
    assert not model._restoring_steady_state


def test_failed_steady_restore_detachment_is_fail_closed_and_retryable(monkeypatch):
    model = _warmed("single")
    model.cache("_steady_state")
    parameters = dict(model.named_parameters())
    saved = model.v.detach().clone()
    post_calls = []
    model.register_post_initialize_hook(lambda _population: post_calls.append(True))

    def fail_detach():
        raise RuntimeError("injected cached detachment failure")

    monkeypatch.setattr(model, "detach", fail_detach)
    with pytest.raises(RuntimeError, match="cached detachment failure"):
        model.initialize()
    assert not model.initialized
    assert not model.integrator.initialized
    assert not model.initializing_from_state_cache
    assert not model._restoring_steady_state
    assert not post_calls
    with pytest.raises(ValueError, match="initialized"):
        model.step(dt=DT)

    monkeypatch.delattr(model, "detach")
    model.initialize()
    assert model.initialized
    assert model.initializing_from_state_cache
    assert post_calls == [True]
    assert all(
        dict(model.named_parameters())[name] is value
        for name, value in parameters.items()
    )
    torch.testing.assert_close(model.v, saved)
    with torch.no_grad():
        model.step(dt=DT)


def _warmed(kind):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True, DTYPE=DTYPE, DEVICE="cpu"):
        model = _model(kind, batch=2)
        with torch.no_grad():
            for _ in range(3):
                model.step(dt=DT)
    return model


@pytest.mark.parametrize("kind", ("tree", "block_tree", "multi_scalar", "multi_block"))
def test_batched_population_owned_cache_matches_uninterrupted_step(kind):
    model, reference = _warmed(kind), _warmed(kind)
    assert _broadcast_buffers(model)
    model.cache("_steady_state")
    saved = model.v.detach().clone()
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        model.v.add_(7)
    model.initialize()
    torch.testing.assert_close(model.v, saved)
    assert all(
        dict(model.named_parameters())[name] is value
        for name, value in parameters.items()
    )
    with torch.no_grad():
        model.step(dt=DT)
        reference.step(dt=DT)
    torch.testing.assert_close(model.v, reference.v, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(
        model.i_membrane, reference.i_membrane, rtol=2e-9, atol=2e-12
    )
    if hasattr(model, "vc"):
        torch.testing.assert_close(model.vc, reference.vc, rtol=2e-9, atol=2e-10)


def test_failed_named_cache_restore_preserves_broadcast_alias_objects_and_values():
    model = _warmed("multi_scalar")
    module, field, original = next(
        (module, name, value)
        for module, name, value in _broadcast_buffers(model)
        if name == "area"
    )
    module.register_buffer("_cache_area_alias", original)
    model.cache("named")
    before = _snapshot(model)
    cache = model._caches["named"]
    topology = next(
        name
        for name in cache
        if name.startswith("populations.tree.") and name.endswith("diff_parent_index")
    )
    cache[topology].reshape(-1)[0].bitwise_xor_(1)
    cache[next(iter(dict(model.named_parameters())))].add_(1)
    with pytest.raises(
        RuntimeError, match="canonical|morphology|topology|geometry|graph"
    ):
        model.restore("named")
    assert module._buffers[field] is original
    assert module._buffers["_cache_area_alias"] is original
    torch.testing.assert_close(_snapshot(model), before)
