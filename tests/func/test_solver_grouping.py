"""Inference groups programs; differentiation keeps independent member graphs."""

from __future__ import annotations

import importlib
from collections.abc import Mapping

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State

DT = 0.025
NAMES = ("first", "second", "third")
pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"
            ),
        ):
            yield
    finally:
        torch.set_num_threads(threads)


class _Decay(State):
    State.STATE("x")
    State.RANGE(rate=0.4, bias=0.2)
    State.ASSIGNED("drive")
    State.DERIVATIVE("x' = drive - rate*x")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.full_like(v, 0.3)}

    def assigned_values(self, v, values):
        return {"drive": self.bias + 0.001 * v + 0.01 * values["x"]}


class _Coupled(State):
    State.STATE("x", "y")
    State.RANGE(rate=0.4, bias=0.2)
    State.ASSIGNED("drive")
    State.DERIVATIVE("x' = drive - rate*x*x + 0.25*y", "y' = rate*x - 0.6*y")
    State.METHOD("rosenbrock", gamma=0.75, damping=0.8)

    def state_defaults(self, v, values):
        del values
        return {"x": torch.full_like(v, 0.3), "y": torch.full_like(v, 0.1)}

    assigned_values = _Decay.assigned_values


class _Current(Mechanism):
    Mechanism.RANGE(g=0.002, e=-63.0)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e + 0.003 * self.x


class _DecayCurrent(_Current):
    Mechanism.STATE_BUNDLE(_Decay)


class _CoupledCurrent(_Current):
    Mechanism.STATE_BUNDLE(_Coupled)


class _GlobalDecay(State):
    State.STATE("x")
    State.GLOBAL(k=0.4, beta=0.2)
    State.ASSIGNED("drive")
    State.DERIVATIVE("x' = drive - k*x")

    state_defaults = _Decay.state_defaults

    def assigned_values(self, v, values):
        return {"drive": self.beta + 0.001 * v + 0.01 * values["x"]}


class _GlobalCurrent(_Current):
    Mechanism.STATE_BUNDLE(_GlobalDecay)


class _PeerState(_Decay):
    def assigned_values(self, v, values):
        drive = self.bias + 0.001 * v + 0.01 * values["x"]
        if "peer" in self._modules:
            drive = drive + 0.2 * self.peer.x
        return {"drive": drive}


class _PeerCurrent(_Current):
    Mechanism.STATE_BUNDLE(_PeerState)


class _OtherOptions(_Coupled):
    State.METHOD("rosenbrock", gamma=0.5, damping=0.6)


class _OtherOptionsCurrent(_Current):
    Mechanism.STATE_BUNDLE(_OtherOptions)


class _TwoDecay(State):
    State.STATE("x", "y")
    State.RANGE(rate=0.4, bias=0.2)
    State.ASSIGNED("drive")
    State.DERIVATIVE("x' = drive - rate*x", "y' = -rate*y")

    state_defaults = _Coupled.state_defaults
    assigned_values = _Decay.assigned_values


class _TwoDecayCurrent(_Current):
    Mechanism.STATE_BUNDLE(_TwoDecay)


class _AuthoredAdvance(_Decay):
    def advance(self, v, dt, values):
        return super().advance(v, dt, values)


class _AuthoredAdvanceCurrent(_Current):
    Mechanism.STATE_BUNDLE(_AuthoredAdvance)


class _CarryCurrent(_DecayCurrent):
    Mechanism.CARRY("memory")


class _MechanismAdvanceCurrent(_DecayCurrent):
    def advance(self, v, dt, values):
        del v, dt, values
        return {}


class _BadAssigned(_Decay):
    def assigned_values(self, v, values):
        del values
        return {"drive": v.new_tensor(0.1)}


class _BadAssignedCurrent(_Current):
    Mechanism.STATE_BUNDLE(_BadAssigned)


def _grouping():
    return importlib.import_module("dendra.models.mechanisms._solver_grouping")


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _close(actual, expected, dtype=torch.float64):
    tolerance = (5e-5, 5e-6) if dtype == torch.float32 else (2e-10, 2e-12)
    torch.testing.assert_close(actual, expected, rtol=tolerance[0], atol=tolerance[1])


def _state(mechanism):
    assert len(mechanism.DE) == 1
    return next(iter(mechanism.DE.values()))


def _model(
    kind=_DecayCurrent,
    *,
    shared=False,
    dtype=torch.float64,
    kinds=None,
    supports=None,
    names=NAMES,
    fields=("rate", "bias"),
):
    population = dn.SingleCompartment(
        N=2, C=3, v_init=-55.0, dtype=dtype, integrator=dn.bwd_euler_sc(imem=False)
    )
    for index, (name, mechanism) in enumerate(
        zip(names, kinds or (kind,) * len(names), strict=True)
    ):
        target = population if supports is None else population[:, supports[index]]
        target.insert(mechanism.rename(name))
    # Insertion records recipes; initialize constructs their handler/modules.
    population.train()
    population.initialize()
    for index, name in enumerate(names):
        state = _state(population.mech.mechanisms[name])
        with torch.no_grad():
            getattr(state, fields[0] + "_param").fill_(0.4 + index * 0.15)
            getattr(state, fields[1] + "_param").fill_(0.2 + index * 0.1)
    if shared:
        _state(population.mech.second).rate_param = _state(
            population.mech.first
        ).rate_param
    # Refresh effective fields from the changed raw sources and aliases before
    # installing distinct recurrent values. Reinitialization must not erase them.
    population.initialize()
    for index, name in enumerate(names):
        mechanism = population.mech.mechanisms[name]
        mechanism.x = torch.full_like(mechanism.x, 0.2 + index * 0.25)
        if "y" in _state(mechanism)._state:
            mechanism.y = torch.full_like(mechanism.y, 0.1 + index * 0.1)
    population.integrator._initialize(
        population, population.t.new_tensor(DT), force=True
    )
    population.detach()
    return population


def _lower(population, monkeypatch, *, enabled=True):
    with monkeypatch.context() as patcher:
        patcher.setattr(_grouping(), "_SOLVER_GROUPING_ENABLED", enabled)
        return dn.func.make_functional(population, dt=DT)


def _groups(functional):
    plans = getattr(
        functional._transition.population.mech, "_solver_grouping_plans", {}
    )
    return tuple(
        action
        for action in plans.get("all", ())
        if isinstance(action, _grouping()._SolverGroup)
    )


def _key(tensors, member, name):
    matches = [
        key
        for key in tensors.parameters
        if f".{member}." in key and key.endswith(name + "_param")
    ]
    assert matches and len({id(tensors.parameters[key]) for key in matches}) == 1, (
        matches
    )
    return matches[0]


def _vector(state):
    return torch.cat(
        (
            state["integrator"]["v"].flatten(),
            *(
                value.flatten()
                for name in NAMES
                for value in state["mechanisms"][name].values()
            ),
        )
    )


def _evaluation(functional, tensors, *, fields=("rate", "bias")):
    rate, bias = _key(tensors, "first", fields[0]), _key(tensors, "third", fields[1])
    rate_aliases = {
        key
        for key, value in tensors.parameters.items()
        if value is tensors.parameters[rate]
    }
    bias_aliases = {
        key
        for key, value in tensors.parameters.items()
        if value is tensors.parameters[bias]
    }

    def evaluate(rate_value, bias_value, initial, dt):
        parameters = {
            key: rate_value
            if key in rate_aliases
            else bias_value
            if key in bias_aliases
            else value
            for key, value in tensors.parameters.items()
        }
        constants = {**tensors.constants, "dt": dt}
        state = {
            **tensors.state,
            "mechanisms": {
                **tensors.state["mechanisms"],
                "first": {
                    **tensors.state["mechanisms"]["first"],
                    "x": initial,
                },
            },
        }
        final, _ = functional.prepare_and_rollout(parameters, constants, state, steps=3)
        return _vector(final)

    return evaluate, (
        tensors.parameters[rate].detach(),
        tensors.parameters[bias].detach(),
        tensors.state["mechanisms"]["first"]["x"],
        tensors.constants["dt"],
    )


@pytest.mark.parametrize(
    "kind", (_DecayCurrent, _CoupledCurrent), ids=("cnexp", "rosenbrock")
)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_grouped_solvers_preserve_live_sources_shared_gradients_and_inputs(
    kind, dtype, monkeypatch
):
    model = _model(kind, shared=True, dtype=dtype)
    functional, tensors = _lower(model, monkeypatch)
    baseline, baseline_tensors = _lower(
        _model(kind, shared=True, dtype=dtype), monkeypatch, enabled=False
    )
    groups = _groups(functional)
    assert len(groups) == 1 and len(groups[0].members) == 3
    assert not _groups(baseline)
    evaluate, operands = _evaluation(functional, tensors)
    reference, _ = _evaluation(baseline, baseline_tensors)
    source = _clone((model.state_dict(), tensors))
    source_tensors = [
        value
        for value in pytree.tree_leaves((model.state_dict(), tensors))
        if isinstance(value, torch.Tensor)
    ]
    versions = tuple(value._version for value in source_tensors)
    group_type = _grouping()._SolverGroup
    original_solve = group_type._solve
    grouped_calls = []

    def observed_solve(group, proxy, dt, packed):
        grouped_calls.append(
            {
                name: tuple(value.shape)
                for name, value in packed.items()
                if isinstance(value, torch.Tensor)
            }
        )
        return original_solve(group, proxy, dt, packed)

    first = None
    for factor in (1.0, 1.6):
        arguments = tuple(
            (value * factor if index < 2 else value).detach().clone().requires_grad_()
            for index, value in enumerate(operands)
        )
        before_calls = len(grouped_calls)
        with monkeypatch.context() as patcher:
            patcher.setattr(group_type, "_solve", observed_solve)
            with torch.no_grad():
                inference = evaluate(*arguments)
            inference_calls = len(grouped_calls)
            actual = evaluate(*arguments)
            assert len(grouped_calls) == inference_calls
        calls = grouped_calls[before_calls:]
        assert len(calls) == 3
        packed_state_shape = (3, *operands[2].shape)
        assert all(packed_state_shape in row.values() for row in calls)
        expected = reference(*arguments)
        _close(inference, expected, dtype)
        _close(actual, expected, dtype)
        actual_gradients = torch.autograd.grad(actual.square().sum(), arguments)
        expected_gradients = torch.autograd.grad(expected.square().sum(), arguments)
        _close(actual_gradients, expected_gradients, dtype)
        assert all(
            torch.isfinite(value).all() and torch.count_nonzero(value)
            for value in actual_gradients
        )
        if first is None:
            first = actual.detach().clone()
        else:
            assert not torch.equal(first, actual)
    _close((model.state_dict(), tensors), source, dtype)
    assert tuple(value._version for value in source_tensors) == versions


@pytest.mark.parametrize(
    "kind", (_DecayCurrent, _CoupledCurrent), ids=("cnexp", "rosenbrock")
)
def test_grouped_solvers_preserve_noncontiguous_vmap_and_empty_lanes(kind, monkeypatch):
    functional, tensors = _lower(_model(kind), monkeypatch)
    assert _groups(functional)
    evaluate, operands = _evaluation(functional, tensors)
    solve = _grouping()._SolverGroup._solve
    calls = []

    def observed(group, proxy, dt, packed):
        calls.append(True)
        return solve(group, proxy, dt, packed)

    monkeypatch.setattr(_grouping()._SolverGroup, "_solve", observed)
    noncontiguous = torch.stack((operands[2], operands[2]), dim=-1)[..., 0]
    assert not noncontiguous.is_contiguous()
    operands = (*operands[:2], noncontiguous, operands[3])
    batched = tuple(torch.stack((value, value + 0.02)) for value in operands)
    expected = torch.stack(
        [evaluate(*(value[index] for value in batched)) for index in range(2)]
    )
    _close(torch.vmap(evaluate)(*batched), expected)
    empty = torch.vmap(evaluate)(*(value[:0] for value in batched))
    assert empty.shape == (0, expected.shape[-1]) and empty.dtype == expected.dtype
    directions = tuple(torch.full_like(value, 0.1) for value in operands)
    _, tangent = torch.func.jvp(evaluate, operands, directions)
    baseline, original = _lower(_model(kind), monkeypatch, enabled=False)
    reference, _ = _evaluation(baseline, original)
    _close(empty, torch.vmap(reference)(*(value[:0] for value in batched)))
    _, wanted = torch.func.jvp(reference, operands, directions)
    _close(tangent, wanted)
    assert not calls, (
        "Differentiation and function transforms retain independent solves"
    )


@pytest.mark.parametrize(
    "kind", (_DecayCurrent, _CoupledCurrent), ids=("cnexp", "rosenbrock")
)
def test_training_partial_member_loss_preserves_unused_parameter_none(
    kind, monkeypatch
):
    functional, tensors = _lower(_model(kind), monkeypatch)
    baseline, original = _lower(_model(kind), monkeypatch, enabled=False)
    assert _groups(functional)
    calls = []
    solve = _grouping()._SolverGroup._solve

    def observed(group, proxy, dt, packed):
        calls.append(True)
        return solve(group, proxy, dt, packed)

    with monkeypatch.context() as patcher:
        patcher.setattr(_grouping()._SolverGroup, "_solve", observed)
        actual, _ = functional.step(
            tensors.parameters,
            functional.prepare(tensors.parameters, tensors.constants),
            tensors.state,
        )
    assert not calls
    expected, _ = baseline.step(
        original.parameters,
        baseline.prepare(original.parameters, original.constants),
        original.state,
    )
    keys = [_key(tensors, name, "rate") for name in ("first", "second")]
    actual_gradients = torch.autograd.grad(
        actual["mechanisms"]["first"]["x"].sum(),
        tuple(tensors.parameters[key] for key in keys),
        allow_unused=True,
    )
    expected_gradients = torch.autograd.grad(
        expected["mechanisms"]["first"]["x"].sum(),
        tuple(original.parameters[key] for key in keys),
        allow_unused=True,
    )
    _close(actual, expected)
    _close(actual_gradients[0], expected_gradients[0])
    assert actual_gradients[1] is expected_gradients[1] is None


@pytest.mark.parametrize(
    "kind", (_DecayCurrent, _CoupledCurrent), ids=("cnexp", "rosenbrock")
)
def test_mixed_source_trainability_preserves_each_state_gradient_flag(
    kind, monkeypatch
):
    def model():
        population = _model(kind)
        for parameter in population.parameters():
            parameter.requires_grad_(False)
        _state(population.mech.first).rate_param.requires_grad_(True)
        return population

    functional, tensors = _lower(model(), monkeypatch)
    baseline, original = _lower(model(), monkeypatch, enabled=False)
    assert _groups(functional)
    actual, _ = functional.step(
        tensors.parameters,
        functional.prepare(tensors.parameters, tensors.constants),
        tensors.state,
    )
    expected, _ = baseline.step(
        original.parameters,
        baseline.prepare(original.parameters, original.constants),
        original.state,
    )
    _close(actual, expected)
    flags = {
        name: tuple(
            value.requires_grad for value in actual["mechanisms"][name].values()
        )
        for name in NAMES
    }
    expected_flags = {
        name: tuple(
            value.requires_grad for value in expected["mechanisms"][name].values()
        )
        for name in NAMES
    }
    assert flags == expected_flags
    assert all(flags["first"]) and not any(flags["second"]) and not any(flags["third"])


@pytest.mark.parametrize(
    "kind", (_DecayCurrent, _CoupledCurrent), ids=("cnexp", "rosenbrock")
)
def test_forward_ad_under_no_grad_retains_independent_solver_calls(kind, monkeypatch):
    functional, tensors = _lower(_model(kind), monkeypatch)
    baseline, original = _lower(_model(kind), monkeypatch, enabled=False)
    evaluate, operands = _evaluation(functional, tensors)
    reference, _ = _evaluation(baseline, original)
    calls = []
    solve = _grouping()._SolverGroup._solve

    def observed(group, proxy, dt, packed):
        calls.append(True)
        return solve(group, proxy, dt, packed)

    with monkeypatch.context() as patcher:
        patcher.setattr(_grouping()._SolverGroup, "_solve", observed)
        with torch.no_grad(), torch.autograd.forward_ad.dual_level():
            initial = torch.autograd.forward_ad.make_dual(
                operands[2], torch.ones_like(operands[2])
            )
            actual = torch.autograd.forward_ad.unpack_dual(
                evaluate(*operands[:2], initial, operands[3])
            )
            expected = torch.autograd.forward_ad.unpack_dual(
                reference(*operands[:2], initial, operands[3])
            )
            assert not calls
            assert actual.tangent is not None and expected.tangent is not None
            _close(actual.primal, expected.primal)
            _close(actual.tangent, expected.tangent)


def test_distinct_solver_families_form_separate_active_groups(monkeypatch):
    names = ("a", "b", "c", "d")
    kinds = (_DecayCurrent, _DecayCurrent, _CoupledCurrent, _CoupledCurrent)
    functional, tensors = _lower(_model(kinds=kinds, names=names), monkeypatch)
    baseline, original = _lower(
        _model(kinds=kinds, names=names), monkeypatch, enabled=False
    )
    groups = _groups(functional)
    assert len(groups) == 2
    assert [group.program.method for group in groups] == ["cnexp", "rosenbrock"]
    assert all(len(group.members) == 2 for group in groups)
    original_solve = _grouping()._SolverGroup._solve
    calls = []

    def solve(group, proxy, dt, packed):
        calls.append(group.program.method)
        return original_solve(group, proxy, dt, packed)

    with monkeypatch.context() as patcher:
        patcher.setattr(_grouping()._SolverGroup, "_solve", solve)
        with torch.no_grad():
            inference, _ = functional.prepare_and_rollout(
                tensors.parameters, tensors.constants, tensors.state, steps=3
            )
        inference_calls = len(calls)
        actual, _ = functional.prepare_and_rollout(
            tensors.parameters, tensors.constants, tensors.state, steps=3
        )
        assert len(calls) == inference_calls
    expected, _ = baseline.prepare_and_rollout(
        original.parameters, original.constants, original.state, steps=3
    )
    assert calls == ["cnexp", "rosenbrock"] * 3
    _close(actual, expected)
    _close(inference, expected)

    def loss(state):
        return state["integrator"]["v"].square().sum() + sum(
            value.square().sum()
            for name in names
            for value in state["mechanisms"][name].values()
        )

    keys = [_key(tensors, name, "rate") for name in (names[0], names[2])]
    observed = torch.autograd.grad(
        loss(actual), tuple(tensors.parameters[key] for key in keys)
    )
    wanted = torch.autograd.grad(
        loss(expected), tuple(original.parameters[key] for key in keys)
    )
    _close(observed, wanted)
    assert all(torch.count_nonzero(value) for value in observed)


@pytest.mark.parametrize("field_dt", (False, True), ids=("scalar-dt", "field-dt"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_shared_rosenbrock_preserves_mapped_timesteps_and_empty_lanes(field_dt, dtype):
    from types import SimpleNamespace

    from dendra.models.mechanisms._rosenbrock import build_rosenbrock1

    solve = build_rosenbrock1(["x"], set(), ["x' = -rate*x*x"], gamma=0.75, damping=0.8)
    state = torch.tensor((0.2, 0.6), dtype=dtype)
    owner = SimpleNamespace(rate=state.new_tensor(0.4), training=True)

    def step(dt):
        return solve(owner, dt, x=state)["x"]

    durations = state.new_tensor(
        ((0.01, 0.02), (0.03, 0.04)) if field_dt else (0.01, 0.03)
    )
    expected = torch.stack(
        [
            state
            - 0.8
            * dt
            * owner.rate
            * state.square()
            / (1 + 0.75 * dt * 2 * owner.rate * state)
            for dt in durations
        ]
    )
    actual = torch.vmap(step)(durations)
    _close(actual, expected, dtype)
    empty = torch.vmap(step)(durations[:0])
    assert empty.shape == (0, 2) and empty.dtype == dtype


def test_global_coefficients_use_the_grouped_path_after_materialization(monkeypatch):
    fields = ("k", "beta")
    functional, tensors = _lower(_model(_GlobalCurrent, fields=fields), monkeypatch)
    baseline, original = _lower(
        _model(_GlobalCurrent, fields=fields), monkeypatch, enabled=False
    )
    assert len(_groups(functional)) == 1
    evaluate, operands = _evaluation(functional, tensors, fields=fields)
    reference, _ = _evaluation(baseline, original, fields=fields)
    solve = _grouping()._SolverGroup._solve
    shapes = []

    def observed(group, proxy, dt, packed):
        shapes.append(tuple(proxy.k.shape))
        return solve(group, proxy, dt, packed)

    arguments = tuple(value.detach().clone().requires_grad_() for value in operands)
    with monkeypatch.context() as patcher:
        patcher.setattr(_grouping()._SolverGroup, "_solve", observed)
        with torch.no_grad():
            inference = evaluate(*arguments)
        inference_calls = len(shapes)
        actual = evaluate(*arguments)
        assert len(shapes) == inference_calls
    assert shapes == [(3, *(1,) * operands[2].ndim)] * 3
    expected = reference(*arguments)
    _close(actual, expected)
    _close(inference, expected)
    _close(
        torch.autograd.grad(actual.square().sum(), arguments),
        torch.autograd.grad(expected.square().sum(), arguments),
    )


def _ordinary_dicts(tree):
    if isinstance(tree, Mapping):
        return {key: _ordinary_dicts(value) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_ordinary_dicts(value) for value in tree)
    return tree


def test_runtime_custom_dispatch_declines_but_fx_capture_keeps_grouping(monkeypatch):
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch.utils._python_dispatch import TorchDispatchMode

    from dendra.func._inputs import _RuntimeInputs

    class Operations(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.calls = []

        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            self.calls.append(
                (
                    str(function),
                    tuple(
                        tuple(value.shape)
                        for value in pytree.tree_leaves((args, kwargs))
                        if isinstance(value, torch.Tensor)
                    ),
                )
            )
            return function(*args, **(kwargs or {}))

    functional, tensors = _lower(_model(), monkeypatch)
    baseline, original = _lower(_model(), monkeypatch, enabled=False)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    reference_prepared = baseline.prepare(original.parameters, original.constants)
    solve = _grouping()._SolverGroup._solve
    calls = []

    def observed(group, proxy, dt, packed):
        calls.append(True)
        return solve(group, proxy, dt, packed)

    with monkeypatch.context() as patcher:
        patcher.setattr(_grouping()._SolverGroup, "_solve", observed)
        with torch.no_grad():
            direct, _ = functional.step(tensors.parameters, prepared, tensors.state)
        assert calls
        with torch.no_grad():
            baseline.step(original.parameters, reference_prepared, original.state)
        calls.clear()
        actual_mode, reference_mode = Operations(), Operations()
        with torch.no_grad(), actual_mode:
            actual, _ = functional.step(tensors.parameters, prepared, tensors.state)
        assert not calls
        with torch.no_grad(), reference_mode:
            expected, _ = baseline.step(
                original.parameters, reference_prepared, original.state
            )
        assert actual_mode.calls == reference_mode.calls
        _close(actual, expected)

        def step(parameters, values, state):
            return functional._step_values(parameters, values, state, _RuntimeInputs())[
                0
            ]

        operands = _ordinary_dicts((tensors.parameters, prepared.values, tensors.state))
        with torch.no_grad():
            graph = make_fx(step)(*operands)
        assert calls, "The compiler's own FX capture must retain batched solves"
        _close(graph(*operands), direct)


def test_registered_peer_reads_preserve_sequential_member_commits(monkeypatch):
    def model():
        population = _model(_PeerCurrent)
        _state(population.mech.second).peer = population.mech.first
        _state(population.mech.third).peer = population.mech.second
        return population

    actual, expected = model(), model()
    with monkeypatch.context() as patcher:
        patcher.setattr(_grouping(), "_SOLVER_GROUPING_ENABLED", False)
        _grouping()._install_solver_grouping(expected)
    _grouping()._install_solver_grouping(actual)
    assert not any(
        isinstance(action, _grouping()._SolverGroup)
        for action in getattr(actual.mech, "_solver_grouping_plans", {}).get("all", ())
    )
    actual.step(DT)
    expected.step(DT)
    _close(actual.v, expected.v)
    _close(
        tuple(actual.mech.mechanisms[name].x for name in NAMES),
        tuple(expected.mech.mechanisms[name].x for name in NAMES),
    )


def test_indexed_generated_coefficients_keep_the_ungrouped_axis(monkeypatch):
    from types import MethodType

    from dendra.utils.dynamic_compilation import compile_generated_function

    def model():
        population = _model()
        for mechanism in population.mech.mechanisms.values():
            state = _state(mechanism)
            original = state._solve.__func__
            source = original.__source__
            assert "self.rate" in source
            # Indexed public DERIVATIVE syntax is unsupported by the symbolic
            # builder. Exercise the private classifier with indexed generated
            # source without expanding the author's equation grammar.
            function = compile_generated_function(
                source.replace("self.rate", "self.rate[0]"),
                func_name="solve",
                filename_prefix="dendra.cnexp.solve",
                global_ns=original.__globals__,
            )
            state._solve = MethodType(function, state)
        return population

    actual, expected = model(), model()
    with monkeypatch.context() as patcher:
        patcher.setattr(_grouping(), "_SOLVER_GROUPING_ENABLED", False)
        _grouping()._install_solver_grouping(expected)
    _grouping()._install_solver_grouping(actual)
    assert not any(
        isinstance(action, _grouping()._SolverGroup)
        for action in getattr(actual.mech, "_solver_grouping_plans", {}).get("all", ())
    )
    actual.step(DT)
    expected.step(DT)
    _close(actual.v, expected.v)
    _close(
        tuple(actual.mech.mechanisms[name].x for name in NAMES),
        tuple(expected.mech.mechanisms[name].x for name in NAMES),
    )


@pytest.mark.parametrize(
    "kind", (_AuthoredAdvanceCurrent, _CarryCurrent, _MechanismAdvanceCurrent)
)
def test_authored_or_carried_transitions_keep_the_original_path(kind, monkeypatch):
    functional, _ = _lower(_model(kind), monkeypatch)
    assert not _groups(functional)


@pytest.mark.parametrize(
    "kinds",
    (
        (_CoupledCurrent, _OtherOptionsCurrent, _CoupledCurrent),
        (_DecayCurrent, _TwoDecayCurrent, _DecayCurrent),
    ),
    ids=("solver-options", "state-schema"),
)
def test_method_options_schemas_and_intervening_programs_do_not_form_a_group(
    kinds, monkeypatch
):
    model = _model(kinds=kinds)
    functional, _ = _lower(model, monkeypatch)
    assert not _groups(functional)


def test_different_ordered_supports_do_not_form_a_group(monkeypatch):
    model = _model(supports=(slice(0, 2), slice(1, 3), slice(0, 2)))
    functional, _ = _lower(model, monkeypatch)
    assert not _groups(functional)


def test_cross_member_mutable_state_aliases_keep_the_original_path(monkeypatch):
    model = _model()
    model.mech.second._buffers["x"] = model.mech.first._buffers["x"]
    model.mech.third._buffers["x"] = model.mech.first._buffers["x"]
    functional, _ = _lower(model, monkeypatch)
    assert not _groups(functional)


@pytest.mark.parametrize("enabled", (False, True))
def test_grouping_preserves_assigned_output_validation(enabled, monkeypatch):
    with dn.ctx(CHECK_PURITY=False):
        functional, tensors = _lower(
            _model(_BadAssignedCurrent), monkeypatch, enabled=enabled
        )
        prepared = functional.prepare(tensors.parameters, tensors.constants)
        with pytest.raises(ValueError, match="assigned_values.*must match"):
            functional.step(tensors.parameters, prepared, tensors.state)


def _runner(population, monkeypatch, *, enabled):
    with monkeypatch.context() as patcher:
        patcher.setattr(_grouping(), "_SOLVER_GROUPING_ENABLED", enabled)
        return dn.compile(
            population,
            dt=DT,
            steps=2,
            chunk_steps=2,
            checkpoint_every=2,
            inference=False,
            backend="aot_eager",
            trainable=("*rate_param", "*bias_param"),
            callbacks={
                "trace": dn.func.Recorder(("v", "first.x", "second.x", "third.x"))
            },
        )


def _loss(result):
    state, auxiliary = result
    return (
        _vector(state).square().sum()
        + auxiliary["callbacks"]["trace"]["v"].square().sum()
    )


@pytest.mark.parametrize(
    "kind", (_DecayCurrent, _CoupledCurrent), ids=("cnexp", "rosenbrock")
)
def test_grouped_scan_checkpoints_preserve_recording_and_portable_owner_gradients(
    kind, monkeypatch
):
    runner = _runner(_model(kind, shared=True), monkeypatch, enabled=True)
    baseline = _runner(_model(kind, shared=True), monkeypatch, enabled=False)
    assert _groups(runner.functional) and not _groups(baseline.functional)
    first = runner()
    expected_first = baseline()
    _close(first[0], expected_first[0])
    _close(first[1]["callbacks"]["trace"], expected_first[1]["callbacks"]["trace"])
    packet = first[1]["continuation"]
    expected_packet = expected_first[1]["continuation"]
    portable_data = runner.export_continuation(packet)
    portable = runner.restore_continuation(portable_data)
    expected_portable = baseline.restore_continuation(portable_data)
    final, expected = runner(packet), baseline(expected_packet)
    _close(final[0], expected[0])
    _close(final[1]["callbacks"]["trace"], expected[1]["callbacks"]["trace"])
    assert first[1]["callbacks"]["trace"]["v"].shape[0] == 3
    assert final[1]["callbacks"]["trace"]["v"].shape[0] == 2
    actual_leaves = tuple(dn.func.inspect_sources(packet).trainable_leaves)
    expected_leaves = tuple(dn.func.inspect_sources(expected_packet).trainable_leaves)
    _close(
        torch.autograd.grad(_loss(final), actual_leaves),
        torch.autograd.grad(_loss(expected), expected_leaves),
    )
    restored, expected_restored = runner(portable), baseline(expected_portable)
    _close(restored[0], expected_restored[0])
    _close(
        restored[1]["callbacks"]["trace"], expected_restored[1]["callbacks"]["trace"]
    )
    restored_leaves = tuple(dn.func.inspect_sources(portable).trainable_leaves)
    assert len(restored_leaves) == len(actual_leaves)
    assert not {id(value) for value in restored_leaves} & {
        id(value) for value in actual_leaves
    }
    _close(
        torch.autograd.grad(_loss(restored), restored_leaves),
        torch.autograd.grad(
            _loss(expected_restored),
            tuple(dn.func.inspect_sources(expected_portable).trainable_leaves),
        ),
    )


@pytest.mark.parametrize(
    "kind", (_DecayCurrent, _CoupledCurrent), ids=("cnexp", "rosenbrock")
)
def test_scan_inference_groups_during_capture_and_preserves_continuations(
    kind, monkeypatch
):
    from tests._call_observer import observe_calls

    model, reference = _model(kind, shared=True), _model(kind, shared=True)
    owners = (model, reference)
    before = _clone(tuple(owner.state_dict() for owner in owners))
    tensors = tuple(
        value for owner in owners for value in (*owner.parameters(), *owner.buffers())
    )
    identities = tuple(
        (id(value), value._version, value.requires_grad) for value in tensors
    )

    def compile_runner(owner, enabled):
        with monkeypatch.context() as patcher:
            patcher.setattr(_grouping(), "_SOLVER_GROUPING_ENABLED", enabled)
            return dn.compile(
                owner,
                dt=DT,
                steps=2,
                chunk_steps=2,
                inference=True,
                execution="scan",
                backend="aot_eager",
                callbacks={
                    "trace": dn.func.Recorder(("v", "first.x", "second.x", "third.x"))
                },
            )

    runner, baseline = compile_runner(model, True), compile_runner(reference, False)
    assert _groups(runner.functional) and not _groups(baseline.functional)
    calls = []
    with observe_calls(
        {_grouping()._SolverGroup._solve.__code__},
        lambda frame: calls.append(len(frame.f_locals["self"].members)),
    ):
        first = runner()
    assert calls and all(count == 3 for count in calls)
    expected_first = baseline()
    _close(first[0], expected_first[0])
    _close(first[1]["callbacks"]["trace"], expected_first[1]["callbacks"]["trace"])
    packet = first[1]["continuation"]
    second = runner(packet)
    expected_second = baseline(expected_first[1]["continuation"])
    _close(second[0], expected_second[0])
    _close(second[1]["callbacks"]["trace"], expected_second[1]["callbacks"]["trace"])
    assert first[1]["callbacks"]["trace"]["v"].shape[0] == 3
    assert second[1]["callbacks"]["trace"]["v"].shape[0] == 2

    exported = runner.export_continuation(packet)
    portable = runner.restore_continuation(exported)
    restored = runner(portable)
    expected_restored = baseline(baseline.restore_continuation(exported))
    _close(restored[0], expected_restored[0])
    _close(
        restored[1]["callbacks"]["trace"], expected_restored[1]["callbacks"]["trace"]
    )
    _close(restored[0], second[0])
    _close(restored[1]["callbacks"]["trace"], second[1]["callbacks"]["trace"])
    assert restored[1]["callbacks"]["trace"]["v"].shape[0] == 2
    _close(tuple(owner.state_dict() for owner in owners), before)
    assert (
        tuple((id(value), value._version, value.requires_grad) for value in tensors)
        == identities
    )
