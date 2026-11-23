"""Public opt-in scan correctness, compatibility scope, and cache lifetime."""

import gc
import importlib
import warnings
import weakref
from functools import partial
from types import FunctionType

import pytest
import torch
from test_runner_compiled_callbacks import _TraceMSE
from test_runner_compiled_chunks import _run
from test_runners import (
    DT,
    GNABAR,
    _assert_tree_close,
    _clone_state,
    _differentiable_case,
    _drives,
    _model,
)

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func import _scan_compat
from dendra.models.mechanisms import Mechanism, State

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


@pytest.fixture(autouse=True)
def _known_compiler_warnings():
    with torch_compiler_warning_context(), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.jit\.script` is deprecated.*",
            category=FutureWarning,
            module=r"torch\.jit\._script",
        )
        yield


class _AffineRelaxationState(State):
    State.STATE("x")
    State.GLOBAL(rate=2.0, source=1.0)
    State.DERIVATIVE("x' = source - rate*x")
    State.METHOD("derivimplicit")

    def state_defaults(self, v, values):
        return {"x": torch.full_like(v, 0.5)}


class _AffineRelaxation(Mechanism):
    Mechanism.STATE_BUNDLE(_AffineRelaxationState)


class _CoupledReactionState(State):
    State.STATE("u", "w")
    State.GLOBAL(rate=1.25)
    State.DERIVATIVE("u' = -rate*u*w", "w' = rate*u*w")
    State.METHOD("derivimplicit")

    def state_defaults(self, v, values):
        return {"u": torch.full_like(v, 0.75), "w": torch.full_like(v, 0.25)}


class _CoupledReaction(Mechanism):
    Mechanism.STATE_BUNDLE(_CoupledReactionState)


@pytest.mark.cpu
def test_compiled_affine_derivimplicit_preserves_equilibrium_gradients():
    dt, steps = 0.1, 4
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64, IMEM=False):
        model = dn.SingleCompartment(N=2, C=1, dtype=torch.float64).train()
        model.insert(_AffineRelaxation)
        model.initialize()
        program = dn.compile(
            model,
            dt=dt,
            steps=steps,
            execution="scan",
            backend="eager",
            inference=False,
            trainable=("*rate_param", "*source_param"),
        )
        state, _ = program()

    actual = state["mechanisms"][_AffineRelaxation.__name__]["x"]
    torch.testing.assert_close(actual, torch.full_like(actual, 0.5), rtol=0, atol=0)
    parameters = {
        name.rsplit(".", 1)[-1]: value
        for name, value in program.named_parameters.items()
        if value.requires_grad
    }
    rate, source = parameters["rate_param"], parameters["source_param"]
    rate_vjp, source_vjp = torch.autograd.grad(actual.sum(), (rate, source))

    # At equilibrium the backward Euler recurrence retains this fraction of x0.
    retained = (1 + dt * 2.0) ** -steps
    source_derivative = actual.numel() * (1 - retained) / 2.0
    torch.testing.assert_close(
        rate_vjp, torch.full_like(rate, -0.5 * source_derivative)
    )
    torch.testing.assert_close(source_vjp, torch.full_like(source, source_derivative))


@pytest.mark.cpu
def test_compiled_nonlinear_derivimplicit_matches_coupled_backward_euler():
    dt, steps = 0.1, 4
    name = _CoupledReaction.__name__
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64, IMEM=False):
        model = dn.SingleCompartment(N=2, C=1, dtype=torch.float64).train()
        model.insert(_CoupledReaction)
        model.initialize()
        program = dn.compile(
            model,
            dt=dt,
            steps=steps,
            execution="scan",
            backend="eager",
            inference=False,
            trainable="*rate_param",
        )
        initial = program.initial_state
        mechanisms = initial["mechanisms"]
        reference = mechanisms[name]["u"]
        u = reference.new_tensor([[0.75], [0.25]], requires_grad=True)
        w = reference.new_tensor([[0.3], [0.0]], requires_grad=True)
        incoming = {**initial, "mechanisms": {**mechanisms, name: {"u": u, "w": w}}}
        (rate,) = program.trainable_parameters.values()

        state, _ = program(incoming)
        actual = (state["mechanisms"][name]["u"], state["mechanisms"][name]["w"])

        # u+w is conserved by backward Euler, so each step reduces to a quadratic.
        total = u + w
        scale = dt * rate
        linear = 1 + scale * total
        expected_u = u
        for _ in range(steps):
            root = (linear.square() - 4 * scale * expected_u).sqrt()
            expected_u = 2 * expected_u / (linear + root)
        expected = (expected_u, total - expected_u)
        cotangents = (u.new_tensor([[0.8], [-0.2]]), w.new_tensor([[-0.1], [0.7]]))
        actual_vjp = torch.autograd.grad(actual, (u, w, rate), cotangents)
        expected_vjp = torch.autograd.grad(expected, (u, w, rate), cotangents)
        torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-8)
        torch.testing.assert_close(actual_vjp, expected_vjp, rtol=1e-7, atol=1e-8)

        # In the second lane, u+w=0 and 4*dt*rate*u>1 give a negative discriminant.
        failed_inputs = {
            "u": u.new_tensor([[0.75], [3.0]], requires_grad=True),
            "w": w.new_tensor([[0.3], [-3.0]], requires_grad=True),
        }
        failed = {**initial, "mechanisms": {**mechanisms, name: failed_inputs}}
        with pytest.raises(
            RuntimeError, match=r"derivimplicit.*chunk step 0, batch index \(1, 0\)"
        ):
            program(failed)
        replay, _ = program(incoming)
        torch.testing.assert_close(
            replay["mechanisms"][name], state["mechanisms"][name]
        )


def _assert_gradients(actual, expected):
    assert actual.keys() == expected.keys()
    for name, reference in expected.items():
        scale = reference.abs().amax()
        assert torch.isfinite(reference).all() and scale > 0, name
        torch.testing.assert_close(
            actual[name] / scale,
            reference / scale,
            rtol=2e-8,
            atol=2e-10,
        )


def _evaluate(functional, tensors, ve, intra, chunk, presence=(True, True), offset=0):
    parameters, constants, state, local_ve, local_intra, targets = _differentiable_case(
        tensors, ve, intra
    )
    with torch.no_grad():
        parameters[GNABAR].add_(offset)
    if not presence[0]:
        local_ve = None
        del targets["ve"]
    if not presence[1]:
        local_intra = None
        del targets["intra"]
    prepared = functional.prepare(parameters, constants)
    inputs = dn.func.RolloutInput(local_ve, local_intra)
    original = _clone_state((parameters, constants, state))
    if chunk is None:
        final, auxiliary = functional.rollout(
            parameters, prepared, state, inputs, steps=ve.shape[0]
        )
    else:
        final, auxiliary = chunk(parameters, prepared, state, inputs)
    assert not final["control"]["duration_remainder"].requires_grad
    saved = _clone_state((final, auxiliary))
    loss = final["integrator"]["v"].square().mean()
    gradients = dict(
        zip(
            targets,
            torch.autograd.grad(loss, tuple(targets.values()), retain_graph=True),
            strict=True,
        )
    )
    repeated = dict(
        zip(targets, torch.autograd.grad(loss, tuple(targets.values())), strict=True)
    )
    _assert_tree_close((parameters, constants, state), original, rtol=0, atol=0)
    _assert_tree_close((final, auxiliary), saved, rtol=0, atol=0)
    _assert_gradients(repeated, gradients)
    return final, auxiliary, gradients


@pytest.mark.parametrize(
    "presence", [(False, False), (True, False), (False, True), (True, True)]
)
def test_scan_chunk_preserves_independent_first_order_oracles(presence):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 4)
    expected = _evaluate(functional, tensors, ve, intra, None, presence)
    chunk = functional.compile_rollout_chunk(4, execution="scan", backend="aot_eager")
    with torch_compiler_warning_context():
        actual = _evaluate(functional, tensors, ve, intra, chunk, presence)
    _assert_tree_close(actual[:2], expected[:2], rtol=2e-10, atol=2e-11)
    _assert_gradients(actual[2], expected[2])


def test_scan_chunk_reuses_graphs_with_new_parameter_and_prepared_values():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 3)
    graphs = []

    def backend(graph, _examples):
        graphs.append(graph)
        return graph.forward

    chunk = functional.compile_rollout_chunk(3, execution="scan", backend=backend)
    count = None
    results = []
    with torch_compiler_warning_context():
        for offset in (0.0, 0.007):
            expected = _evaluate(functional, tensors, ve, intra, None, offset=offset)
            actual = _evaluate(functional, tensors, ve, intra, chunk, offset=offset)
            _assert_tree_close(actual[:2], expected[:2], rtol=2e-10, atol=2e-11)
            _assert_gradients(actual[2], expected[2])
            results.append(actual[0]["integrator"]["v"].detach())
            if count is None:
                count = len(graphs)
                assert count > 0
            else:
                assert len(graphs) == count
    assert not torch.equal(*results)


def _callbacks(functional, steps):
    target = torch.linspace(-64, -52, (steps + 1) * 10, dtype=torch.float64).reshape(
        steps + 1, *functional.shape
    )
    options = {
        "threshold": -60.0,
        "node_check": [0, 1],
        "t_start_check": DT,
        "t_end_check": (steps - 1) * DT,
    }
    return functional.make_callbacks(
        {
            "mse": _TraceMSE(target),
            "trace": dn.func.Recorder(["v", "hh.m", "t"]),
            "raster": dn.func.Raster(**options),
            "count": dn.func.APCount(**options),
            "anomaly": dn.func.AnomalyDetector(),
        }
    )


def test_scan_checkpoint_callbacks_preserve_discrete_state_and_gradients():
    steps = 10
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, steps)
    signs = intra.new_tensor([1, 1, -1, -1, 1, 1, -1, -1, 1, 1])
    intra = signs.reshape(steps, 1, 1).expand_as(intra) * 1e-8
    callbacks = _callbacks(functional, steps)
    chunk = functional.compile_rollout_chunk(7, execution="scan", backend="aot_eager")

    def evaluate(compiled):
        parameters, constants, state, ve_local, intra_local, targets = (
            _differentiable_case(tensors, ve, intra)
        )
        prepared = functional.prepare(parameters, constants)
        final, auxiliary = _run(
            "longrun_checkpointed" if compiled else "run",
            functional,
            partial(chunk if compiled else functional.step, parameters, prepared),
            state,
            dn.func.RolloutInput(ve_local, intra_local),
            steps=steps,
            chunklength=4,
            callbacks=callbacks,
        )
        result = auxiliary["callbacks"]
        loss = result["mse"] + result["trace"]["v"].square().mean()
        gradients = dict(
            zip(
                targets, torch.autograd.grad(loss, tuple(targets.values())), strict=True
            )
        )
        return final, result, gradients

    expected = evaluate(False)
    with torch_compiler_warning_context():
        actual = evaluate(True)
    _assert_tree_close(actual[:2], expected[:2], rtol=2e-10, atol=2e-11)
    _assert_gradients(actual[2], expected[2])
    result = actual[1]
    assert result["trace"]["v"].shape[0] == steps + 1
    assert result.state["mse"]["frames"].dtype == torch.int64
    assert result.state["mse"]["frames"].item() == steps + 1
    assert result["raster"].dtype == torch.bool
    assert result["raster"].shape[0] == steps - 2
    assert result["count"].dtype == torch.int32
    assert bool((result["count"] >= 2).all())
    torch.testing.assert_close(
        result["count"], result["raster"].sum(0, dtype=torch.int32)
    )
    assert result.state["count"][2].item() == steps
    assert result["anomaly"].dtype == torch.bool
    assert not result["anomaly"].any()
    for value in torch.utils._pytree.tree_leaves(result):
        if value.dtype in (torch.bool, torch.int32, torch.int64):
            assert not value.requires_grad


@pytest.mark.parametrize("with_callbacks", [False, True])
def test_scan_cache_releases_callback_collection_and_live_training_graph(
    with_callbacks,
):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 3)
    chunk = functional.compile_rollout_chunk(3, execution="scan", backend="eager")

    def iteration():
        callbacks = _callbacks(functional, 3) if with_callbacks else None
        parameters, constants, state, local_ve, local_intra, targets = (
            _differentiable_case(tensors, ve, intra)
        )
        prepared = functional.prepare(parameters, constants)
        references = {
            "prepared": weakref.ref(prepared),
            "parameter": weakref.ref(parameters[GNABAR]),
            "geometry": weakref.ref(constants["diam"]),
            "workspace": weakref.ref(prepared.values["integrator"]["diag_base"]),
        }
        if callbacks is not None:
            references["callbacks"] = weakref.ref(callbacks)
        with torch_compiler_warning_context():
            inputs = dn.func.RolloutInput(local_ve, local_intra)
            if callbacks is None:
                final, _auxiliary = chunk(parameters, prepared, state, inputs)
                loss = final["integrator"]["v"].square().mean()
            else:
                _state, auxiliary = dn.func.run(
                    functional,
                    partial(chunk, parameters, prepared),
                    state,
                    inputs,
                    callbacks=callbacks,
                )
                loss = auxiliary["callbacks"]["mse"]
            torch.autograd.grad(loss, tuple(targets.values()))
        return references

    references = iteration()
    gc.collect()
    assert {
        name: reference() for name, reference in references.items()
    } == dict.fromkeys(references)
    assert not chunk._callback_chunks


class _InitializationOnly(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        voltage = state["integrator"]["v"]
        return voltage.new_zeros((), dtype=torch.int64), voltage

    def update(self, carry, state, auxiliary):
        return carry + 1, None

    def finalize(self, carry, emissions):
        assert torch.is_tensor(emissions)
        return emissions


def test_scan_initialization_only_emissions_have_empty_resumed_schema():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    callbacks = functional.make_callbacks({"initial": _InitializationOnly()})
    chunk = functional.compile_rollout_chunk(2, execution="scan", backend="eager")
    step = partial(chunk, tensors.parameters, prepared)
    with torch.no_grad(), torch_compiler_warning_context():
        first, auxiliary = dn.func.run(
            functional, step, tensors.state, tstop=2 * DT, callbacks=callbacks
        )
        result = auxiliary["callbacks"]
        torch.testing.assert_close(
            result["initial"], tensors.state["integrator"]["v"][None]
        )
        final, resumed = dn.func.run(
            functional,
            step,
            first,
            tstop=2 * DT,
            callbacks=callbacks,
            callback_state=result.state,
        )
        assert resumed["callbacks"]["initial"].shape == (0, *functional.shape)
        assert resumed["callbacks"].state["initial"].item() == 4
        _zero, zero = dn.func.run(
            functional,
            step,
            final,
            tstop=0.0,
            callbacks=callbacks,
            callback_state=resumed["callbacks"].state,
        )
        assert zero["callbacks"]["initial"].shape == (0, *functional.shape)
        assert zero["callbacks"].state["initial"].item() == 4


def test_scan_direct_forward_ad_falls_back_without_compilation():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    def unexpected(_graph, _examples):
        raise AssertionError("direct forward AD must not enter compiled scan")

    chunk = functional.compile_rollout_chunk(2, execution="scan", backend=unexpected)
    with torch.autograd.forward_ad.dual_level(), torch.no_grad():
        voltage = tensors.state["integrator"]["v"]
        dual = torch.autograd.forward_ad.make_dual(voltage, torch.ones_like(voltage))
        state = {
            **tensors.state,
            "integrator": {**tensors.state["integrator"], "v": dual},
        }
        expected, _ = functional.rollout(tensors.parameters, prepared, state, steps=2)
        actual, _ = chunk(tensors.parameters, prepared, state)
        for a, e in zip(
            torch.utils._pytree.tree_leaves(actual),
            torch.utils._pytree.tree_leaves(expected),
            strict=True,
        ):
            torch.testing.assert_close(
                torch.autograd.forward_ad.unpack_dual(a),
                torch.autograd.forward_ad.unpack_dual(e),
                rtol=0,
                atol=0,
            )
    assert not chunk._compiled._kernels


def test_scan_public_chunk_rejects_recorded_backward_through_preparation():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 2)
    parameters, constants, state, ve, intra, _targets = _differentiable_case(
        tensors, ve, intra
    )
    prepared = functional.prepare(parameters, constants)
    chunk = functional.compile_rollout_chunk(2, execution="scan", backend="aot_eager")
    with torch_compiler_warning_context():
        final, _auxiliary = chunk(
            parameters, prepared, state, dn.func.RolloutInput(ve, intra)
        )
        loss = final["integrator"]["v"].square().mean()
        with pytest.raises(dn.func.FunctionalizationError, match="first-order"):
            torch.autograd.grad(loss, constants["diam"], create_graph=True)


@pytest.mark.parametrize("version", ["2.14.0", "2.14.2", "2.15.0", "2.99.0"])
def test_scan_uses_the_installed_contract_not_its_version_label(monkeypatch, version):
    functional, _tensors = dn.func.make_functional(_model(), dt=DT)
    monkeypatch.setattr(torch, "__version__", version)
    # This checks label independence on the installed contract, not support
    # for an uninstalled future compiler's implementation.
    functional.compile_rollout_chunk(2, execution="scan", backend="eager")
    functional.compile_rollout_chunk(2, backend="eager")
    with pytest.raises(ValueError, match="execution must be"):
        functional.compile_rollout_chunk(2, execution="unknown", backend="eager")


def test_scan_compatibility_rejects_incompatible_signature_before_mutation(monkeypatch):
    partitioner = importlib.import_module("torch._higher_order_ops.partitioner")
    original = partitioner.create_hop_joint_graph
    if _scan_compat._INSTALLED is not None:
        original = original.__wrapped__
    monkeypatch.setattr(partitioner, "create_hop_joint_graph", original)
    monkeypatch.setattr(_scan_compat, "_INSTALLED", None)

    def wrong_signature(value):
        return value

    incompatible = FunctionType(
        wrong_signature.__code__, partitioner.__dict__, "create_hop_joint_graph"
    )
    monkeypatch.setattr(partitioner, "create_hop_joint_graph", incompatible)
    with pytest.raises(dn.func.FunctionalizationError, match="incompatible signature"):
        _scan_compat.install_scan_compatibility()
    assert partitioner.create_hop_joint_graph is incompatible
    assert _scan_compat._INSTALLED is None


def test_scan_compatibility_behavior_failure_is_atomic(monkeypatch):
    partitioner = importlib.import_module("torch._higher_order_ops.partitioner")
    original = partitioner.create_hop_joint_graph
    if _scan_compat._INSTALLED is not None:
        original = original.__wrapped__
    monkeypatch.setattr(partitioner, "create_hop_joint_graph", original)
    monkeypatch.setattr(_scan_compat, "_INSTALLED", None)

    def reject_probe(_entry):
        raise dn.func.FunctionalizationError("injected joint behavior failure")

    monkeypatch.setattr(_scan_compat, "_probe_scan_contract", reject_probe)
    with pytest.raises(dn.func.FunctionalizationError, match="joint behavior failure"):
        _scan_compat.install_scan_compatibility()
    assert partitioner.create_hop_joint_graph is original
    assert _scan_compat._INSTALLED is None


def test_scan_behavior_probe_is_cpu_local_under_an_ambient_meta_device():
    _scan_compat.install_scan_compatibility()
    entry = _scan_compat._INSTALLED[1]
    original_device = torch.get_default_device()
    observed = []

    def observe_entry(function, primals, **kwargs):
        observed.append(tuple(value.device.type for value in primals))
        return entry(function, primals, **kwargs)

    with torch.device("meta"):
        assert torch.get_default_device().type == "meta"
        _scan_compat._probe_scan_contract(observe_entry)
        assert torch.get_default_device().type == "meta"
    assert observed == [("cpu", "cpu", "cpu", "cpu")]
    assert torch.get_default_device() == original_device


def test_scan_compatibility_rejects_replaced_callable_before_mutation(monkeypatch):
    utils = importlib.import_module("torch._higher_order_ops.utils")
    partitioner = importlib.import_module("torch._higher_order_ops.partitioner")
    original = partitioner.create_hop_joint_graph
    if _scan_compat._INSTALLED is not None:
        original = original.__wrapped__
    monkeypatch.setattr(partitioner, "create_hop_joint_graph", original)
    monkeypatch.setattr(_scan_compat, "_INSTALLED", None)
    monkeypatch.setattr(utils, "create_bw_fn", lambda *_args: None)
    with pytest.raises(dn.func.FunctionalizationError, match="already been replaced"):
        _scan_compat.install_scan_compatibility()
    assert partitioner.create_hop_joint_graph is original


@pytest.mark.parametrize("target", ["entry", "shared_helper", "partitioner_helper"])
def test_scan_compatibility_rechecks_active_callable_identities(monkeypatch, target):
    utils = importlib.import_module("torch._higher_order_ops.utils")
    partitioner = importlib.import_module("torch._higher_order_ops.partitioner")
    _scan_compat.install_scan_compatibility()
    active = partitioner.create_hop_joint_graph
    if target == "entry":
        monkeypatch.setattr(partitioner, "create_hop_joint_graph", lambda *_args: None)
    elif target == "shared_helper":
        monkeypatch.setattr(utils, "create_bw_fn", lambda *_args: None)
    else:
        monkeypatch.setattr(partitioner, "create_bw_fn", lambda *_args: None)
    with pytest.raises(
        dn.func.FunctionalizationError, match="active .* changed|replaced"
    ):
        _scan_compat.install_scan_compatibility()
    if target != "entry":
        assert partitioner.create_hop_joint_graph is active


@pytest.mark.parametrize("multithreading", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_scan_joint_capture_scopes_autograd_multithreading(multithreading, fail):
    utils = importlib.import_module("torch._higher_order_ops.utils")
    aot = importlib.import_module("torch._functorch.aot_autograd")
    observed = []

    def create_joint(*args, **kwargs):
        def joint(primals, tangents):
            observed.append(torch.autograd.is_multithreading_enabled())
            if fail:
                raise RuntimeError("joint capture failed")
            return [primals[0].square()], [2 * primals[0] * tangents[0]]

        return joint

    primal = torch.tensor([0.2, -0.4], dtype=torch.float64, requires_grad=True)
    tangent = torch.ones_like(primal)
    factory = _scan_compat._make_full_output_backward(
        utils, aot.AOTConfig, create_joint
    )
    backward = factory(lambda value: value.square(), (primal,))
    with torch.autograd.set_multithreading_enabled(multithreading):
        if fail:
            with pytest.raises(RuntimeError, match="joint capture failed"):
                backward(primal, tangent)
        else:
            torch.testing.assert_close(backward(primal, tangent), [2 * primal])
        assert torch.autograd.is_multithreading_enabled() is multithreading
    assert observed == [False]


def test_scan_compatibility_preserves_mixed_output_cotangent_positions():
    """Prove the audited upstream bug and that the local repair fixes its cause."""
    from torch.fx.experimental.proxy_tensor import make_fx

    utils = importlib.import_module("torch._higher_order_ops.utils")
    aot = importlib.import_module("torch._functorch.aot_autograd")
    x = torch.tensor([0.2, -0.4], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0.7, 0.9], dtype=torch.float64, requires_grad=True)
    index = torch.tensor(3, dtype=torch.int64)
    flag = torch.tensor([True, False])

    def function(x, y, index, flag):
        # Distinct float cotangents, interleaved with nondifferentiable slots,
        # detect both dropping slots and silently shifting gradient positions.
        return index + 1, x.square() + y, ~flag, x * y

    primals = (x, y, index, flag)
    outputs = function(*primals)
    tangents = (
        torch.zeros_like(index),
        torch.full_like(x, 1.3),
        torch.zeros_like(flag),
        torch.full_like(x, -0.4),
    )
    unpatched = utils.create_bw_fn(function, primals)
    with pytest.raises(AssertionError, match="outs_to_grad length.*tangents length"):
        make_fx(unpatched)(*primals, *tangents)

    repaired_factory = _scan_compat._make_full_output_backward(
        utils, aot.AOTConfig, aot.create_joint
    )
    repaired = repaired_factory(function, primals, return_fw_outputs=True)
    graph = make_fx(repaired)(*primals, *tangents)
    result = graph(*primals, *tangents)
    torch.testing.assert_close(result[:4], outputs)
    expected = (
        2 * x * 1.3 - 0.4 * y,
        1.3 - 0.4 * x,
        torch.zeros_like(index),
        torch.zeros_like(flag),
    )
    torch.testing.assert_close(tuple(result[4:]), expected)


def test_scan_mixed_integer_boolean_carry_and_emissions_match_eager():
    from torch._higher_order_ops.scan import scan

    _scan_compat.install_scan_compatibility()

    def evaluate(structured):
        initial = torch.tensor([0.2, -0.3], dtype=torch.float64, requires_grad=True)
        parameter = torch.tensor([0.7, 0.8], dtype=torch.float64, requires_grad=True)
        drives = torch.linspace(-0.1, 0.2, 8, dtype=torch.float64).reshape(4, 2)
        drives.requires_grad_()

        def body(carry, drive):
            index, value, flag = carry
            value = torch.tanh(value * parameter + drive)
            index = index + 1
            flag = torch.logical_xor(flag, value > 0)
            return (index, value, flag), (index.clone(), value.clone(), flag.clone())

        carry = (torch.tensor(0), initial, torch.tensor([True, False]))
        if structured:
            kernel = torch.compile(
                lambda carry, drives: scan(body, carry, drives),
                backend="aot_eager",
                fullgraph=True,
            )
            final, traces = kernel(carry, drives)
        else:
            emissions = []
            for drive in drives:
                carry, emitted = body(carry, drive)
                emissions.append(emitted)
            final = carry
            traces = tuple(
                torch.stack(values) for values in zip(*emissions, strict=True)
            )
        loss = final[1].square().sum() + traces[1].square().mean()
        gradients = torch.autograd.grad(loss, (initial, parameter, drives))
        assert final[0].dtype == torch.int64
        assert final[2].dtype == torch.bool
        return final, traces, gradients

    expected = evaluate(False)
    actual = evaluate(True)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_scan_compatibility_is_idempotent_and_preserves_shared_while_backward():
    utils = importlib.import_module("torch._higher_order_ops.utils")
    partitioner = importlib.import_module("torch._higher_order_ops.partitioner")
    original_bw = utils.create_bw_fn
    original_masks = utils.prepare_fw_with_masks_all_requires_grad
    _scan_compat.install_scan_compatibility()
    active = partitioner.create_hop_joint_graph
    _scan_compat.install_scan_compatibility()
    assert partitioner.create_hop_joint_graph is active
    assert utils.create_bw_fn is original_bw
    assert utils.prepare_fw_with_masks_all_requires_grad is original_masks

    def evaluate(structured):
        initial = torch.tensor(
            [0.2, -0.3, 0.5], dtype=torch.float64, requires_grad=True
        )
        parameter = torch.tensor(
            [0.7, 0.8, 0.9], dtype=torch.float64, requires_grad=True
        )

        def condition(index, _value):
            return index < 4

        def body(index, value):
            return index + 1, torch.tanh(value * parameter + 0.1)

        carry = torch.zeros((), dtype=torch.int64), initial
        if structured:
            count, value = torch.while_loop(condition, body, carry)
        else:
            for _ in range(4):
                carry = body(*carry)
            count, value = carry
        gradients = torch.autograd.grad(value.square().sum(), (initial, parameter))
        assert all(torch.isfinite(g).all() and g.abs().amax() > 0 for g in gradients)
        return count, value, gradients

    expected = evaluate(False)
    with torch_compiler_warning_context():
        actual = evaluate(True)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)
