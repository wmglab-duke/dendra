"""Canonical packets survive first-run and mid-run portable boundaries."""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_continuation import _callbacks, _close, _joined, _snapshot
from tests.func.test_continuation_checkpoints import _disk, _leaf_flags
from tests.func.test_explicit_initialization import _case as _population_case
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _module_snapshot,
)
from tests.func.test_lowering_api import DT as POPULATION_DT
from tests.func.test_lowering_api import G
from tests.func.test_network_stochastic_conformance import DT as NETWORK_DT
from tests.func.test_network_stochastic_conformance import (
    RAW,
)
from tests.func.test_network_stochastic_conformance import (
    _assert_module_snapshot as _assert_network_snapshot,
)
from tests.func.test_network_stochastic_conformance import (
    _module_snapshot as _network_snapshot,
)
from tests.func.test_network_stochastic_conformance import (
    _network,
    _recording,
)

pytestmark = pytest.mark.cpu
STEPS = 2


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
    ):
        yield


def _make(network):
    if network:
        model = _network(training=False, cache_kind="owned")
        functional, tensors = dn.func.make_functional(model, dt=NETWORK_DT)
        return model, functional, tensors, NETWORK_DT, _recording(), ()
    functional, tensors = _population_case(False, scaled=True)
    return (
        functional._population,
        functional,
        tensors,
        POPULATION_DT,
        _callbacks(False),
        "*g_param",
    )


def _assert_aliases(packet, network):
    if network:
        assert (
            packet.parameters["populations"]["cable"][RAW]
            is packet.parameters["populations"]["cell"][RAW]
        )


@pytest.mark.parametrize(
    "network", (False, True), ids=("training_population", "cached_stochastic_network")
)
def test_first_run_and_resumed_portable_packets_match_fresh_convenience_execution(
    network, tmp_path
):
    model, functional, tensors, dt, specifications, trainable = _make(network)
    program = dn.compile(
        model,
        dt=dt,
        steps=STEPS,
        initialization="fresh",
        inference=network,
        trainable=trainable,
        callbacks=specifications,
        backend="eager",
        disable=True,
    )
    snapshot = _network_snapshot(model) if network else _module_snapshot(model)
    defaults = _snapshot((program.parameters, program.constants, program.initial_state))
    original = tensors._replace(parameters=tensors.independent_parameters(trainable))
    global_rng = torch.get_rng_state().clone()
    expected_prefix, expected_first = program()
    expected, expected_last = program(expected_first["continuation"])
    expected_trace = _joined(expected_first, expected_last)
    context = torch.no_grad if network else nullcontext
    with context():
        initialized = dn.func.initialize(functional, original)
    assert initialized.callback_state is None
    _assert_aliases(initialized, network)
    # Saving an initialized frame does not require configuring callbacks. The
    # receiving callback collection starts at that accepted initial state.
    saved_initial = _disk(
        dn.func.export_continuation(functional, initialized), tmp_path / "initial.pt"
    )
    receiving, _ = dn.func.make_functional(model, dt=dt)
    restored = dn.func.restore_continuation(receiving, saved_initial)
    assert restored.callback_state is None
    assert _leaf_flags(restored.parameters) == _leaf_flags(initialized.parameters)
    _assert_aliases(restored, network)
    callbacks = receiving.make_callbacks(specifications)
    bound = receiving.bind(restored)
    options = (
        {"draws": receiving.sample_runtime_draws(restored.state, steps=STEPS)}
        if network
        else {}
    )
    with context():
        prefix, first = dn.func.run(
            bound, restored, steps=STEPS, callbacks=callbacks, **options
        )
    _close(prefix, expected_prefix)
    _close(first["callbacks"].outputs, expected_first["callbacks"].outputs)
    _assert_aliases(first["continuation"], network)
    assert all(
        value.shape[0] == STEPS + 1 for value in first["callbacks"]["trace"].values()
    )
    saved_middle = _disk(
        dn.func.export_continuation(
            receiving, first["continuation"], callbacks=callbacks
        ),
        tmp_path / "middle.pt",
    )
    final_owner, _ = dn.func.make_functional(model, dt=dt)
    final_callbacks = final_owner.make_callbacks(specifications)
    resumed = dn.func.restore_continuation(
        final_owner, saved_middle, callbacks=final_callbacks
    )
    assert resumed.callback_state.plan_token is final_callbacks._state_token
    assert (
        resumed.callback_state.plan_token
        is not first["continuation"].callback_state.plan_token
    )
    _assert_aliases(resumed, network)
    with context():
        actual, last = dn.func.run(
            final_owner.bind(resumed),
            resumed,
            steps=STEPS,
            callbacks=final_callbacks,
        )
    _close(actual, expected)
    _close(_joined(first, last), expected_trace)
    _close(last["callbacks"].outputs, expected_last["callbacks"].outputs)
    assert all(value.shape[0] == STEPS for value in last["callbacks"]["trace"].values())
    _assert_aliases(last["continuation"], network)
    # The same portable format resumes through a convenience runner. Gradients
    # span its suffix; the graph preceding each portable boundary is detached.
    program_packet = program.restore_continuation(saved_middle)
    compatible, compatible_aux = program(program_packet)
    _close(compatible, actual)
    _close(compatible_aux["callbacks"].outputs, last["callbacks"].outputs)
    report = dn.func.inspect_sources(resumed)
    if not network:
        selected = resumed.parameters[G]
        assert selected.is_leaf and selected.requires_grad and selected.grad_fn is None
        assert any(value is selected for value in report.trainable_leaves)
        loss = actual["integrator"]["v"].square().sum()
        observed, upstream = torch.autograd.grad(
            loss, (selected, original.parameters[G]), allow_unused=True
        )
        reference = torch.autograd.grad(
            compatible["integrator"]["v"].square().sum(), program_packet.parameters[G]
        )[0]
        _close(observed, reference)
        assert torch.isfinite(observed).all() and observed.abs().sum() > 0
        assert upstream is None
    _close((program.parameters, program.constants, program.initial_state), defaults)
    torch.testing.assert_close(torch.get_rng_state(), global_rng)
    if network:
        _assert_network_snapshot(model, snapshot)
    else:
        _assert_module_snapshot(model, snapshot)
