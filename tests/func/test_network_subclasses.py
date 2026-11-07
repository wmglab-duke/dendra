"""Authored construction helpers retain canonical Network execution."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import expsyn, graded_syn, pas
from tests.func.test_network import DT, DTYPE, _clone, _close
from tests.func.test_network_cache_initialization import _warm_cache

pytestmark = pytest.mark.cpu
GAIN = "integrator.mech.mechanisms.pas.g_param"


class _Cell(dn.SingleCompartment):
    def __init__(self, shared, voltage):
        super().__init__(N=1, C=1, v_init=voltage)
        self.insert(pas, g=shared, e=-70.0)
        self.insert(graded_syn.rename("analog"), e=0.0)
        self.insert(expsyn.rename("events"), tau=1.2, e=0.0)


class _Description:
    def describe(self):
        return tuple(self.populations)


class _Circuit(_Description, dn.Network):
    def __init__(self, *, stimulus=False):
        shared = torch.nn.Parameter(torch.tensor(0.001, dtype=DTYPE))
        left, right = _Cell(shared, -61.0), _Cell(shared, -65.0)
        netstim = (
            dn.NetStim(N=1, start=0.0, interval=0.5, noise=0.6, seed=812)
            if stimulus
            else None
        )
        super().__init__({"left": left, "right": right}, netstim=netstim)
        self.connect_continuous_one_to_one(
            left[:],
            right[:],
            right.mech.analog,
            input="g_pre",
            transform=torch.sigmoid,
            weight=0.002,
            delay=DT,
        )
        if stimulus:
            self.connect_one_to_one(
                netstim[:], right[:], right.mech.events, weight=0.003, delay=DT
            )


class _DerivedCircuit(_Circuit):
    pass


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


def _network(*, stimulus=False, training=True, cached=False):
    result = _DerivedCircuit(stimulus=stimulus)
    result.train(training)
    result.initialize(DT)
    result.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=False,
        diff_spiking=False,
        diff_scheduled_times=False,
        train_delay_backend="dense",
    )
    if cached:
        _warm_cache(result)
        result.initialize(DT)
    return result


@pytest.mark.parametrize("fresh", (False, True))
def test_network_subclasses_preserve_native_gradients_and_parameter_sharing(fresh):
    source, native = _network(), _network()
    functional, tensors = dn.func.make_functional_network(source)
    frozen = _clone(tensors, detached=True)
    current = tensors
    if fresh:
        current = functional.initialize(*current, functional.initialization_inputs)
        native.initialize(DT, reinit_weights=False, reinit_delays=False)
    state = current.state
    for _ in range(4):
        state, _ = functional.prepare_and_step(
            current.parameters, current.constants, state
        )
        native.step()
    _close(state, functional.extract(native).state)
    gain = current.parameters["populations"]["left"][GAIN]
    assert gain is current.parameters["populations"]["right"][GAIN]
    loss = sum(
        pop["integrator"]["v"].square().sum() for pop in state["populations"].values()
    )
    actual = torch.autograd.grad(loss, gain)[0]
    expected_loss = native.left.v.square().sum() + native.right.v.square().sum()
    expected = torch.autograd.grad(expected_loss, native.left.mech.pas.g_param)[0]
    _close(actual, expected)
    assert actual.abs().sum() > 0
    _close(functional.extract(source), frozen)
    assert source.describe() == ("left", "right")


@pytest.mark.parametrize(
    "initialization,inference,execution,cached,backend",
    (
        ("snapshot", False, "scan", False, "aot_eager"),
        ("fresh", False, "scan", False, "aot_eager"),
        ("fresh", True, "while", False, "aot_eager"),
        ("fresh", True, "scan", True, "aot_eager"),
        ("snapshot", True, "while", False, "inductor"),
    ),
)
def test_network_subclasses_compiled_recording_cache_and_checkpoint_continuation(
    initialization, inference, execution, cached, backend
):
    source = _network(training=not inference, cached=cached)
    native = _network(training=not inference, cached=cached)
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        backend=backend,
        initialization=initialization,
        inference=inference,
        execution=execution,
        checkpoint_every=None if inference else 2,
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
    )
    before = runner.functional.extract(source)
    if initialization == "fresh":
        native.initialize(DT, reinit_weights=False, reinit_delays=False)
    final, aux = runner()
    for _ in range(4):
        native.step()
    _close(final, runner.functional.extract(native).state)
    assert aux["callbacks"]["trace"]["right.v"].shape[0] == 5
    repeated, _ = runner()
    _close(repeated, final)
    continued, _ = runner(
        final,
        parameters=aux.get("parameters", runner.parameters),
        constants=aux.get("constants", runner.constants),
        callback_state=aux["callbacks"].state,
    )
    for _ in range(4):
        native.step()
    _close(continued, runner.functional.extract(native).state)
    receiver = _network(training=not inference, cached=cached)
    assert runner.functional.commit_state_(receiver, continued) is receiver
    receiver.step()
    native.step()
    _close(
        runner.functional.extract(receiver).state,
        runner.functional.extract(native).state,
    )
    _close(runner.functional.extract(source), before)


def test_network_subclasses_preserve_explicit_netstim_rng_checkpoint_suffix():
    source, native = _network(stimulus=True), _network(stimulus=True)
    native.restore_dict_from_checkpoint(
        _clone(source.state_dict_for_checkpoint(), detached=True)
    )
    functional, tensors = dn.func.make_functional_network(source)
    state = tensors.state
    for _ in range(3):
        draw = functional.sample_step_draw(state)
        state, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, state, netstim_draw=draw
        )
        native.step()
    _close(state, functional.extract(native).state)
    receiver = _network(stimulus=True)
    functional.commit_state_(receiver, state)
    checkpoint = receiver.state_dict_for_checkpoint()
    restored = _network(stimulus=True)
    restored.restore_dict_from_checkpoint(checkpoint)
    for _ in range(4):
        draw = functional.sample_step_draw(state)
        state, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, state, netstim_draw=draw
        )
        native.step()
        restored.step()
    _close(state, functional.extract(native).state)
    _close(state, functional.extract(restored).state)


@pytest.mark.parametrize(
    "hook",
    (
        "step",
        "_advance_runtime_clock",
        "initialize",
        "_refresh_compile_config_from_ctx",
        "checkpoint_structure",
    ),
)
def test_network_subclasses_reject_replaced_execution_contracts(hook):
    def changed(self, *args, **kwargs):
        return getattr(super(type(self), self), hook)(*args, **kwargs)

    incompatible = type("IncompatibleCircuit", (_Circuit,), {hook: changed})
    # Replace after native construction so an override need not execute to
    # produce the structural rejection.
    source = _network()
    source.__class__ = incompatible
    with pytest.raises(dn.func.FunctionalizationError, match=f"canonical.*{hook}"):
        dn.func.make_functional_network(source)


def test_network_subclass_hook_changes_invalidate_a_warmed_compiled_runner(monkeypatch):
    source = _network(training=False)
    runner = dn.compile(source, dt=DT, steps=4, inference=True, backend="aot_eager")
    runner()
    monkeypatch.setattr(_Description, "step", lambda self: None, raising=False)
    with pytest.raises(
        dn.func.FunctionalizationError, match="adapters changed|execution hooks"
    ):
        runner()


def test_network_source_class_change_requires_lowering_again():
    source = _network()
    functional, _ = dn.func.make_functional_network(source)
    source.__class__ = _Circuit
    with pytest.raises(dn.func.FunctionalizationError, match="configuration changed"):
        functional.extract()


def test_fresh_network_initialization_checks_its_inherited_refresh_helper(monkeypatch):
    source = _network()
    source.__class__ = dn.Network
    functional, tensors = dn.func.make_functional_network(source)
    calls = []
    monkeypatch.setattr(
        source, "_refresh_compile_config_from_ctx", lambda: calls.append("refresh")
    )
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="canonical.*_refresh_compile_config_from_ctx",
    ):
        functional.initialize(*tensors, functional.initialization_inputs)
    assert calls == []


@pytest.mark.parametrize("name", ("t", "populations", "_clock_step"))
def test_network_subclass_runtime_descriptors_are_rejected_without_evaluation(name):
    calls = []

    def getter(self):
        calls.append(name)
        raise AssertionError("authored runtime descriptor must not execute")

    source = _network()
    source.__class__ = type("MaskedCircuit", (_Circuit,), {name: property(getter)})
    with pytest.raises(
        dn.func.FunctionalizationError, match="masks canonical.*runtime"
    ):
        dn.func.make_functional_network(source)
    assert calls == []


@pytest.mark.parametrize("name", ("t", "populations", "__getattribute__"))
def test_network_retained_source_contract_is_checked_before_authored_access(
    name, monkeypatch
):
    calls = []
    source = _network()
    functional, tensors = dn.func.make_functional_network(source)

    def changed(self, *args):
        calls.append(name)
        raise AssertionError("authored source accessor must not execute")

    replacement = changed if name == "__getattribute__" else property(changed)
    monkeypatch.setattr(_DerivedCircuit, name, replacement, raising=False)
    with pytest.raises(dn.func.FunctionalizationError, match="adapters changed"):
        functional.prepare_and_step(
            tensors.parameters, tensors.constants, tensors.state
        )
    assert calls == []
