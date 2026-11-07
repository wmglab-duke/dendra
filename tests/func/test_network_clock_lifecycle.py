"""Copied native clocks retain functional restart and checkpoint boundaries."""

from __future__ import annotations

import copy
import gc
import weakref

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_network import (
    DT,
    DTYPE,
    _clone,
    _close,
    _loss,
    _mixed_network,
    _native_loss,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _assert_clock_owners(network):
    for connection in (
        *network.synapses.values(),
        *network.continuous_synapses.values(),
    ):
        assert connection.t is network.t
    for population in network.populations.values():
        for mechanism in population.mech.mechanisms.values():
            assert mechanism.t is population.t


@pytest.mark.parametrize("initialization", ("snapshot", "fresh"))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_copied_network_compiles_and_continues_from_its_own_clock(
    initialization, execution
):
    source = _mixed_network()
    if execution == "while":
        source.eval()
        source.initialize(DT, reinit_weights=False, reinit_delays=False)
    with torch.no_grad():
        for _ in range(2):
            source.step()
    copied = copy.deepcopy(source)
    _assert_clock_owners(copied)
    saved = _clone(copied.state_dict_for_checkpoint(), detached=True)
    with torch.no_grad():
        for _ in range(3):
            source.step()
    source_snapshot = _clone(source.state_dict_for_checkpoint(), detached=True)
    assert copied.t.item() != source.t.item()
    _assert_clock_owners(copied)

    native = _mixed_network()
    if execution == "while":
        native.eval()
        native.initialize(DT, reinit_weights=False, reinit_delays=False)
    native.restore_dict_from_checkpoint(saved)
    runner = dn.compile(
        copied,
        dt=DT,
        steps=2,
        initialization=initialization,
        inference=execution == "while",
        execution=execution,
        trainable=("*transform_parameters.scale",),
        callbacks={"trace": dn.func.Recorder(("cell.v", "t"))},
        backend="aot_eager",
        **({"checkpoint_every": 1} if execution == "scan" else {}),
    )
    prefix, first = runner()
    final, second = runner(
        prefix,
        parameters=first.get("parameters", runner.parameters),
        constants=first.get("constants", runner.constants),
        callback_state=first["callbacks"].state,
    )
    if initialization == "fresh":
        native.initialize(DT, reinit_weights=False, reinit_delays=False)
    for _ in range(4):
        native.step()
    _close(final, runner.functional.extract(native).state)
    if execution == "scan":
        actual = torch.autograd.grad(
            _loss(final), runner.get_parameter("*transform_parameters.scale")
        )[0]
        transform = next(
            connection.transform
            for connection in native.continuous_synapses.values()
            if isinstance(connection.transform, torch.nn.Module)
        )
        expected = torch.autograd.grad(_native_loss(native), transform.scale)[0]
        torch.testing.assert_close(actual, expected)
        assert actual.abs().sum() > 0

    # Publishing through the functional adapter stages another native deepcopy.
    # It must retain the receiving Network as clock owner after buffer replacement.
    runner.functional.commit_state_(copied, final)
    _assert_clock_owners(copied)
    _close(runner.functional.extract(copied).state, final)
    with torch.no_grad():
        copied.step()
        native.step()
    _close(
        runner.functional.extract(copied).state, runner.functional.extract(native).state
    )
    _close(source.state_dict_for_checkpoint(), source_snapshot)


def test_copied_clock_descriptors_do_not_retain_original_network_or_population():
    def branch():
        source = _mixed_network()
        return copy.deepcopy(source), weakref.ref(source), weakref.ref(source.cell)

    copied, network_reference, population_reference = branch()
    gc.collect()
    assert network_reference() is None
    assert population_reference() is None
    _assert_clock_owners(copied)
    with torch.no_grad():
        copied.step()
    _assert_clock_owners(copied)


@pytest.mark.parametrize("check_purity", (False, True))
def test_reinitialized_copy_retains_proxy_types_and_requires_fresh_lowering(
    check_purity,
):
    source = copy.deepcopy(_mixed_network())
    with dn.ctx(CHECK_PURITY=check_purity):
        functional, tensors = dn.func.make_functional_network(source)
    classes = tuple(type(connection) for connection in source.synapses.values())
    source.initialize(DT, reinit_weights=False, reinit_delays=False)
    assert tuple(type(connection) for connection in source.synapses.values()) == classes
    # Native initialization invalidates the frozen delivery generation even
    # when the owner's proxy classes and topology are unchanged.
    with pytest.raises(dn.func.FunctionalizationError, match="changed|lower"):
        functional.prepare_and_step(*tensors)
    with dn.ctx(CHECK_PURITY=check_purity):
        updated, current = dn.func.make_functional_network(source)
    initialized = updated.initialize(*current, updated.initialization_inputs)
    _close(initialized.state, updated.extract(source).state)
    actual, _ = updated.prepare_and_step(*initialized)
    source.step()
    _close(actual, updated.extract(source).state)
