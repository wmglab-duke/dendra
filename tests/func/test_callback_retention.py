"""Host collectors retain sampled callback history without changing pure results."""

from __future__ import annotations

import gc
import os
import weakref
from collections import defaultdict
from contextlib import nullcontext

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._callbacks import (
    FunctionalCallbacks,
    _CallbackCollector,
    _host_callback_parts,
    _ImperativeCallbackBinding,
)
from tests.func.test_lowering_api import DT, _close, _network, _population, _target
from tests.func.test_recorder_cadence import (
    _callbacks,
    _indices,
    _inputs,
    _loss,
    _select,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False
            ),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _case(network, *, steps=9):
    model = _network() if network else _population()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    shape = model.left.shape if network else model.shape
    tape = torch.linspace(1e-7, 4e-7, steps * torch.Size(shape).numel()).reshape(
        steps, *shape
    )
    return functional, tensors, tape.requires_grad_()


def _atomic(functional, tensors, tape, callbacks, network):
    """The unchanged pure rollout finalizes dense emissions as the reference."""
    return functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=tape.shape[0],
        inputs=_inputs(network, tape),
        callbacks=callbacks,
    )


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_host_retention_matches_atomic_results_gradients_and_offphase_continuation(
    network, execution
):
    functional, tensors, tape = _case(network)
    callbacks = _callbacks(functional, network)
    kernel = functional.compile_rollout_chunk(
        4, execution=execution, backend="aot_eager"
    )
    bound = kernel.bind(tensors)
    options = {"checkpoint_every": 7} if execution == "scan" else {"host_span_steps": 7}
    with torch.no_grad() if execution == "while" else nullcontext():
        expected, reference = _atomic(functional, tensors, tape, callbacks, network)
        actual, result = dn.func.run(
            bound,
            tensors.state,
            inputs=_inputs(network, tape),
            steps=9,
            callbacks=callbacks,
            **options,
        )
    _close(actual, expected)
    _close(result["callbacks"], reference["callbacks"])
    assert result["callbacks"]["dense"]["t"].shape == (10,)
    assert result["callbacks"]["slow"]["t"].shape == (6,)
    assert result["callbacks"]["sparse"]["t"].shape == (3,)
    assert result["callbacks"]["every"]["samples"].shape == (9,)
    if execution == "scan":
        targets = (_target(tensors.parameters, network), tape)
        wanted = torch.autograd.grad(_loss(reference["callbacks"], network), targets)
        observed = torch.autograd.grad(_loss(result["callbacks"], network), targets)
        _close(observed, wanted)
        assert all(
            torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed
        )
    with torch.no_grad() if execution == "while" else nullcontext():
        _, prefix = dn.func.run(
            bound,
            tensors.state,
            steps=1,
            inputs=_inputs(network, tape[:1]),
            callbacks=callbacks,
            **options,
        )
        report = kernel.execution_report()
        unchanged, empty = dn.func.run(
            bound, prefix["continuation"], steps=0, callbacks=callbacks, **options
        )
        _close(unchanged, prefix["continuation"].state)
        assert kernel.execution_report() is report
        _, middle = dn.func.run(
            bound,
            prefix["continuation"],
            steps=4,
            inputs=_inputs(network, tape[1:5]),
            callbacks=callbacks,
            **options,
        )
        final, suffix = dn.func.run(
            bound,
            middle["continuation"],
            steps=4,
            inputs=_inputs(network, tape[5:]),
            callbacks=callbacks,
            **options,
        )
    _close(final, expected)
    assert middle["callbacks"]["sparse"]["t"].shape == (0,)
    joined = {}
    for name in ("dense", "slow", "sparse"):
        joined[name] = {
            field: torch.cat(
                [part["callbacks"][name][field] for part in (prefix, middle, suffix)]
            )
            for field in result["callbacks"][name]
        }
        _close(joined[name], result["callbacks"][name])
        assert all(value.shape[0] == 0 for value in empty["callbacks"][name].values())
    _close(suffix["callbacks"].state, result["callbacks"].state)
    if execution == "scan":
        key = "left.v" if network else "v"
        empty_gradients = torch.autograd.grad(
            middle["callbacks"]["sparse"][key].sum(), targets, retain_graph=True
        )
        assert all(torch.count_nonzero(value) == 0 for value in empty_gradients)
        observed = torch.autograd.grad(_loss(joined, network), targets)
        _close(observed, wanted)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_checkpoint_spans_smaller_than_kernel_keep_retained_phase_and_vjps(network):
    functional, tensors, tape = _case(network)
    callbacks = _callbacks(functional, network)
    kernel = functional.compile_rollout_chunk(4, execution="scan", backend="eager")
    expected, reference = _atomic(functional, tensors, tape, callbacks, network)
    actual, result = dn.func.run(
        kernel.bind(tensors),
        tensors.state,
        steps=9,
        inputs=_inputs(network, tape),
        callbacks=callbacks,
        checkpoint_every=2,
    )
    _close(actual, expected)
    _close(result["callbacks"], reference["callbacks"])
    targets = (_target(tensors.parameters, network), tape)
    wanted = torch.autograd.grad(_loss(reference["callbacks"], network), targets)
    observed = torch.autograd.grad(_loss(result["callbacks"], network), targets)
    _close(observed, wanted)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_host_retention_owns_only_selected_payload_and_releases_dense_chunks(
    monkeypatch, network, execution
):
    functional, tensors, tape = _case(network, steps=17)
    names = ("left.v", "t") if network else ("v", "t")
    callbacks = functional.make_callbacks(
        {"trace": dn.func.Recorder(names, dt=8.5 * DT)}
    )
    kernel = functional.compile_rollout_chunk(4, execution=execution, backend="eager")
    events, raw_references = [], []
    append = _CallbackCollector.append

    def observe(collector, callback_state, emitted, *, retained=False):
        raw = emitted["trace"]
        rows = next(iter(raw.values())).shape[0]
        raw_storage = {
            name: value.untyped_storage()._cdata for name, value in raw.items()
        }
        if not retained and rows > 1:
            assert all(reference() is None for reference in raw_references)
            raw_references.extend(weakref.ref(value) for value in raw.values())
        append(collector, callback_state, emitted, retained=retained)
        selected = collector.parts[-1]["trace"]
        kept = next(iter(selected.values())).shape[0]
        metadata = []
        for name, value in selected.items():
            metadata.append(
                (
                    value.untyped_storage().nbytes(),
                    value.numel() * value.element_size(),
                    value.untyped_storage()._cdata == raw_storage[name],
                )
            )
        events.append((id(collector), retained, rows, kept, tuple(metadata)))

    monkeypatch.setattr(_CallbackCollector, "append", observe)
    with torch.no_grad():
        expected, reference = _atomic(functional, tensors, tape, callbacks, network)
        actual, result = dn.func.run(
            kernel.bind(tensors),
            tensors.state,
            steps=17,
            inputs=_inputs(network, tape),
            callbacks=callbacks,
        )
    _close(actual, expected)
    _close(result["callbacks"], reference["callbacks"])
    _close(
        callbacks.export_state(result["callbacks"].state),
        callbacks.export_state(reference["callbacks"].state),
    )
    assert result["callbacks"]["trace"]["t"].shape == (4,)
    assert events and any(
        not retained and rows == 4 for _, retained, rows, _, _ in events
    )
    assert all(rows <= 4 for _, retained, rows, _, _ in events if not retained)
    retained_rows = defaultdict(int)
    for collector, retained, rows, kept, storage in events:
        retained_rows[collector] += kept
        for actual_bytes, payload_bytes, shares_dense_storage in storage:
            assert actual_bytes == payload_bytes
            if not retained and kept < rows:
                assert not shares_dense_storage
    assert max(retained_rows.values()) == 4
    gc.collect()
    assert raw_references and all(reference() is None for reference in raw_references)


@pytest.mark.parametrize("kind", ("default", "subclass", "finalize", "has"))
def test_host_retention_keeps_legacy_list_and_hook_contracts(monkeypatch, kind):
    functional, tensors, _tape = _case(False)
    configurations = {
        "trace": dn.func.Recorder("v", dt=None if kind == "default" else 2.5 * DT)
    }

    class Derived(FunctionalCallbacks):
        pass

    callbacks = (
        Derived(functional, configurations)
        if kind == "subclass"
        else functional.make_callbacks(configurations)
    )
    events = []
    if kind == "finalize":
        finalize = callbacks._finalize

        def authored_finalize(state, emissions):
            events.append("finalize")
            return finalize(state, emissions)

        monkeypatch.setattr(callbacks, "_finalize", authored_finalize)
    elif kind == "has":
        has_emissions = callbacks._has_emissions

        def authored_has(emissions):
            events.append("has")
            return has_emissions(emissions)

        monkeypatch.setattr(callbacks, "_has_emissions", authored_has)
    state, emission = callbacks._initialize(tensors.state)
    original = [callbacks._stack([emission])]
    assert _host_callback_parts(callbacks, state, original) is original
    with torch.no_grad():
        expected, reference = functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            steps=5,
            callbacks=callbacks,
        )
        events.clear()
        actual, result = dn.func.run(
            functional.bind(tensors), tensors.state, steps=5, callbacks=callbacks
        )
    _close(actual, expected)
    _close(result["callbacks"], reference["callbacks"])
    if kind == "finalize":
        assert events == ["finalize"]
    elif kind == "has":
        assert events and all(event == "has" for event in events)


def test_host_retention_rechecks_all_owned_hook_identities(monkeypatch):
    functional, tensors, _tape = _case(False)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder("v", dt=2.5 * DT)})
    state, emission = callbacks._initialize(tensors.state)
    original = [callbacks._stack([emission])]
    assert isinstance(
        _host_callback_parts(callbacks, state, original), _CallbackCollector
    )
    for name in (
        "_initialize",
        "_validate_state",
        "_update",
        "_stack",
        "_wrap_compiled_update",
        "_concatenate",
        "_has_emissions",
        "_capture_emission_schemas",
        "_finalize",
        "_materialize_state",
    ):
        owned = getattr(FunctionalCallbacks, name)

        def authored(self, *args, **kwargs):
            return owned(self, *args, **kwargs)

        with monkeypatch.context() as scope:
            scope.setattr(FunctionalCallbacks, name, authored)
            assert _host_callback_parts(callbacks, state, original) is original
        assert isinstance(
            _host_callback_parts(callbacks, state, original), _CallbackCollector
        )


def test_native_checkpoint_retention_preserves_vjp_and_lazy_zero_resume(monkeypatch):
    from tests.func.test_imperative_checkpoint_callbacks import DT as bridge_dt
    from tests.func.test_imperative_checkpoint_callbacks import (
        _callbacks as bridge_callbacks,
    )
    from tests.func.test_imperative_checkpoint_callbacks import (
        _checkpoint,
        _gnabar,
    )
    from tests.func.test_imperative_checkpoint_callbacks import _model as bridge_model

    reference_model = bridge_model(require_grad=True)
    functional, tensors = dn.func.make_functional(reference_model, dt=bridge_dt)
    dense_callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("v", "t"))})
    _, reference = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=7,
        callbacks=dense_callbacks,
    )
    expected = _select(reference["callbacks"]["trace"], _indices(7, 3))
    expected_gradient = torch.autograd.grad(
        expected["v"].square().sum(),
        tensors.parameters[tensors.parameter_name("gnabar")],
    )[0]
    model = bridge_model(require_grad=True)
    callbacks = bridge_callbacks(
        model, {"trace": dn.func.Recorder(("v", "t"), dt=3.5 * bridge_dt)}
    )
    _, first = _checkpoint(model, callbacks, steps=2)
    reads = []
    state = _ImperativeCallbackBinding.state

    def read(binding):
        reads.append(True)
        return state(binding)

    monkeypatch.setattr(_ImperativeCallbackBinding, "state", read)
    loss, empty = _checkpoint(model, callbacks, steps=0, callback_state=first.state)
    assert loss is None and reads == []
    assert all(value.shape[0] == 0 for value in empty["trace"].values())
    _close(empty.state, first.state)
    _, initial_only = _checkpoint(model, callbacks, steps=0)
    assert reads == [True]
    assert initial_only["trace"]["t"].shape == (1,)
    _, middle = _checkpoint(model, callbacks, steps=2, callback_state=first.state)
    _, last = _checkpoint(model, callbacks, steps=3, callback_state=middle.state)
    observed = {
        name: torch.cat([part["trace"][name] for part in (first, middle, last)])
        for name in expected
    }
    _close(observed, expected)
    final_voltage, final_time = model.v.detach().clone(), model.t.detach().clone()
    gradient = torch.autograd.grad(observed["v"].square().sum(), _gnabar(model))[0]
    _close(gradient, expected_gradient)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    _close(model.v, final_voltage)
    _close(model.t, final_time)


@pytest.mark.mps
def test_mps_host_retention_preserves_cadence_and_continuation():
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DTYPE=torch.float32, DEVICE="mps"):
        model = _population()
        program = dn.compile(
            model,
            dt=DT,
            steps=9,
            chunk_steps=4,
            execution="while",
            backend="aot_eager",
            callbacks={"trace": dn.func.Recorder(("v", "t"), dt=6.5 * DT)},
        )
        with torch.no_grad():
            expected, reference = program.functional.prepare_and_rollout(
                program.parameters,
                program.constants,
                program.initial_state,
                steps=18,
                callbacks=program.callbacks,
            )
        _, prefix = program()
        actual, suffix = program(prefix["continuation"])
        _close(actual, expected)
        observed = {
            name: torch.cat((prefix["callbacks"]["trace"][name], value))
            for name, value in suffix["callbacks"]["trace"].items()
        }
        _close(observed, reference["callbacks"]["trace"])
        assert prefix["callbacks"]["trace"]["t"].shape == (3,)
        assert suffix["callbacks"]["trace"]["t"].shape == (1,)
        assert all(value.device.type == "mps" for value in observed.values())
