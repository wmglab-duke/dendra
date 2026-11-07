"""Inspection follows runner defaults and effective continuation frames."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_continuation import _assign, _close, _plain, _runner, _snapshot
from tests.func.test_continuation_checkpoint_initialization import _fresh_program
from tests.func.test_lowering_api import G, _target, _voltage

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
    ):
        yield


def _path(network, domain="parameters"):
    return f"{domain}.populations.left.{G}" if network else f"{domain}.{G}"


def _entries(report):
    return {entry.path: entry for entry in report.entries}


@pytest.mark.parametrize("network", (False, True))
def test_compile_runner_inspection_keeps_default_selectors_names_and_aliases(network):
    runner = _runner(network, callbacks=False)
    named = dict(runner.named_parameters)
    selected = dict(runner.trainable_parameters)
    found = runner.find_parameters("*g_param")
    report = dn.func.inspect_sources(runner)
    parameter_entries = [
        entry for entry in report.entries if entry.domain == "parameters"
    ]
    assert [entry.path for entry in parameter_entries] == [
        f"parameters.{name}" for name in named
    ]
    for entry, value in zip(parameter_entries, named.values(), strict=True):
        assert entry.tensor is value
    unique = tuple(dict.fromkeys(map(id, selected.values())))
    assert tuple(map(id, report.trainable_leaves)) == unique
    assert tuple(map(id, runner.trainable_parameters.values())) == tuple(
        map(id, selected.values())
    )
    assert tuple(runner.find_parameters("*g_param")) == tuple(found)
    assert all(
        runner.find_parameters("*g_param")[key] is value for key, value in found.items()
    )
    target = _entries(report)[_path(network)]
    assert target.tensor is _target(runner.parameters, network)
    if network:
        assert target.aliases == (f"parameters.populations.right.{G}",)
    else:
        assert target.aliases == ()


@pytest.mark.parametrize("network", (False, True))
def test_continuation_inspection_separates_call_overrides_from_runner_defaults(network):
    runner = _runner(network, callbacks=False)
    default = _target(runner.parameters, network)
    parameters = _plain(runner.parameters)
    upstream = default.detach().clone().requires_grad_()
    override = upstream * 1.3
    _assign(parameters, network, override)
    _, auxiliary = runner(parameters=parameters)
    effective = dn.func.inspect_sources(auxiliary["continuation"])
    entry = _entries(effective)[_path(network)]
    assert entry.tensor is override and entry.requires_grad and not entry.is_leaf
    assert all(
        value is not override and value is not upstream
        for value in effective.trainable_leaves
    )
    assert _entries(dn.func.inspect_sources(runner))[_path(network)].tensor is default
    assert all(entry.tensor is not upstream for entry in effective.entries)
    assert all(
        entry.tensor is not override
        for entry in dn.func.inspect_sources(runner).entries
    )
    assert _target(runner.parameters, network) is default


@pytest.mark.parametrize("network", (False, True))
def test_inspection_has_no_validation_lifecycle_rng_or_report_side_effects(
    network, monkeypatch
):
    runner = _runner(network, callbacks=False)
    report_before = runner.execution_report()
    values_before = _snapshot(
        (runner.parameters, runner.constants, runner.initial_state)
    )
    rng_before = torch.random.get_rng_state().clone()

    def reject(*args, **kwargs):
        raise AssertionError("inspection must only read explicit sources")

    for name in (
        "_validate_source",
        "_validate_parameters",
        "_validate_state",
        "_validate_constants",
        "prepare",
        "prepare_and_rollout",
        "sample_runtime_draws",
        "_initialize_host" if network else "initialize_host",
    ):
        if hasattr(runner.functional, name):
            monkeypatch.setattr(runner.functional, name, reject)
    report = dn.func.inspect_sources(runner)
    assert report.entries
    assert runner.execution_report() is report_before
    _close((runner.parameters, runner.constants, runner.initial_state), values_before)
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before)


@pytest.mark.parametrize("network", (False, True))
def test_fresh_portable_sources_supply_deduplicated_effective_optimizer_leaves(
    network, tmp_path
):
    source, receiver = _fresh_program(network), _fresh_program(network)
    parameters = _plain(source.parameters)
    upstream = _target(parameters, network).detach().clone().requires_grad_()
    _assign(parameters, network, upstream * 1.2)
    _, first = source(parameters=parameters)
    current = first["continuation"]
    raw_before = _entries(dn.func.inspect_sources(current))[_path(network)]
    assert raw_before.requires_grad and not raw_before.is_leaf
    path = tmp_path / "sources.pt"
    torch.save(source.export_continuation(current), path)
    restored = receiver.restore_continuation(torch.load(path, weights_only=True))
    report = dn.func.inspect_sources(restored)
    raw = _entries(report)[_path(network)]
    assert raw.is_leaf and raw.requires_grad
    assert len(set(map(id, report.trainable_leaves))) == len(report.trainable_leaves)
    assert any(value is raw.tensor for value in report.trainable_leaves)
    retained = [
        entry
        for entry in report.entries
        if entry.domain == "constants" and entry.requires_grad and entry.is_leaf
    ]
    if network:
        assert retained
        assert raw.aliases == (f"parameters.populations.right.{G}",)
    defaults = _snapshot((receiver.parameters, receiver.constants))
    optimizer = torch.optim.SGD(report.trainable_leaves, lr=1e-9)
    state, _ = receiver(restored)
    loss = _voltage(state, network).square().sum()
    loss.backward()
    gradients = [
        value.grad for value in report.trainable_leaves if value.grad is not None
    ]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)
    assert any(value.abs().sum() > 0 for value in gradients)
    if network:
        assert any(
            entry.tensor.grad is not None and entry.tensor.grad.abs().sum() > 0
            for entry in retained
        )
    assert upstream.grad is None
    before = [value.detach().clone() for value in report.trainable_leaves]
    optimizer.step()
    assert any(
        not torch.equal(value, saved)
        for value, saved in zip(report.trainable_leaves, before, strict=True)
    )
    _close((receiver.parameters, receiver.constants), defaults)
