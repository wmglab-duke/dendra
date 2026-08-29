"""CPU lifecycle contracts for temperature-dependent mechanism kinetics."""

from __future__ import annotations

import networkx as nx
import pytest
import torch

import dendra as dn
from dendra.models.mod import hh

DTYPE = torch.float64
LOW_CELSIUS = 6.3
HIGH_CELSIUS = 26.3
INITIAL_V = torch.tensor([-40.0, -61.0, -72.0], dtype=DTYPE)
DT = 0.01
MODEL_KINDS = ("single_compartment", "tree", "unmyelinated", "myelinated")


class _ReplaceParameter(torch.nn.Module):
    def forward(self, value):
        return (value,)


def _tree_graph():
    graph = nx.DiGraph()
    for node in range(3):
        graph.add_node(
            node,
            name=("Cell.soma[0](0.5)" if node == 0 else f"Cell.dend[{node}](0.5)"),
            L=10.0,
            diam=2.0 - 0.2 * node,
            Ra=100.0,
            cm=1.0,
            area=12.0 - node,
            volume=8.0 - node,
            volume_i=8.0 - node,
            volume_o=0.0,
            x=float(node),
            y=0.0,
            z=0.0,
        )
    graph.add_edge(0, 1, R_ohm=8.0e7, diff_geom_um=0.8)
    graph.add_edge(1, 2, R_ohm=9.0e7, diff_geom_um=0.6)
    return graph


def _new_model(kind, celsius, *, training=False, initialize=True):
    common = dict(celsius=celsius, dtype=DTYPE)
    if kind == "single_compartment":
        model = dn.SingleCompartment(N=1, C=1, v_init=-40.0, **common)
    elif kind == "unmyelinated":
        model = dn.Unmyelinated(
            diameters=[2.0], L=3.0, dx=1.0, v_init=INITIAL_V, **common
        )
    elif kind == "myelinated":
        model = dn.Myelinated(
            diameters=[6.0], n_node=3, node_length=1.0, v_init=INITIAL_V, **common
        )
    elif kind == "tree":
        model = dn.Tree.from_graph(_tree_graph(), N=1, v_init=INITIAL_V, **common)
    else:  # pragma: no cover - helper guard
        raise ValueError(kind)
    model.insert(hh)
    model.train(training)
    if initialize:
        model.initialize()
    return model


def _hh_parts(model):
    mechanism = next(iter(model.mech.mechanisms.values()))
    return mechanism, mechanism.DE["mhn"]


def _expected_q10(celsius):
    return 3.0 ** ((float(celsius) - 6.3) / 10.0)


def _runtime_state(model):
    mechanism, state = _hh_parts(model)
    return {
        "v": model.v,
        "m": mechanism.m,
        "h": mechanism.h,
        "n": mechanism.n,
        "q10": state.q10,
        "celsius": model.celsius,
        "state_celsius": state.celsius,
        "t": model.t,
    }


def _assert_runtime_equal(actual, expected):
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0.0, atol=0.0)


def _assert_temperature_views_are_current(model, expected):
    mechanism, state = _hh_parts(model)
    assert model.mech.celsius is model.celsius
    assert mechanism.celsius is model.celsius
    assert state.celsius is model.celsius
    torch.testing.assert_close(model.celsius, torch.as_tensor(expected, dtype=DTYPE))
    torch.testing.assert_close(
        state.q10,
        torch.full_like(state.q10, _expected_q10(expected)),
    )


def _advance(model, steps=5):
    for _ in range(steps):
        model.step(dt=DT)


@pytest.mark.parametrize("kind", MODEL_KINDS)
@pytest.mark.parametrize("training", [False, True])
def test_celsius_parameter_update_reinitializes_q10_and_matches_fresh_model(
    kind, training
):
    updated = _new_model(kind, LOW_CELSIUS, training=training)
    _advance(updated, 2)  # dirty voltage/gates before the lifecycle reset

    updated.parameter_set_(celsius_param=HIGH_CELSIUS)
    # GLOBAL sources are materialized only at the initialize boundary.
    assert updated.celsius.item() == pytest.approx(LOW_CELSIUS)
    updated.initialize()

    fresh = _new_model(kind, HIGH_CELSIUS, training=training)
    _assert_temperature_views_are_current(updated, HIGH_CELSIUS)
    assert updated.training is training
    mechanism, state = _hh_parts(updated)
    assert updated.mech.training is training
    assert mechanism.training is training
    assert state.training is training
    _assert_runtime_equal(_runtime_state(updated), _runtime_state(fresh))

    _advance(updated)
    _advance(fresh)
    _assert_runtime_equal(_runtime_state(updated), _runtime_state(fresh))


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_pre_transform_updates_temperature_q10_and_persists(kind):
    transformed = _new_model(kind, LOW_CELSIUS, initialize=False)
    transformed.register_pre_initialize_transform(
        "replace_temperature",
        _ReplaceParameter(),
        writes=("parameters.celsius_param",),
        inputs={"value": torch.tensor(HIGH_CELSIUS, dtype=DTYPE)},
    )
    transformed.initialize()

    assert transformed.celsius_param.item() == pytest.approx(HIGH_CELSIUS)
    _assert_temperature_views_are_current(transformed, HIGH_CELSIUS)
    transformed.initialize()
    _assert_temperature_views_are_current(transformed, HIGH_CELSIUS)

    fresh = _new_model(kind, HIGH_CELSIUS)
    _assert_runtime_equal(_runtime_state(transformed), _runtime_state(fresh))
    _advance(transformed)
    _advance(fresh)
    _assert_runtime_equal(_runtime_state(transformed), _runtime_state(fresh))


def test_direct_celsius_buffer_mutation_and_rebinding_are_not_source_updates():
    model = _new_model("single_compartment", LOW_CELSIUS)
    _, state = _hh_parts(model)
    initial_q10 = state.q10.clone()

    with torch.no_grad():
        model.celsius.fill_(HIGH_CELSIUS)
    assert model.celsius.item() == pytest.approx(HIGH_CELSIUS)
    assert model.celsius_param.item() == pytest.approx(LOW_CELSIUS)
    torch.testing.assert_close(state.q10, initial_q10)

    # initialize() rematerializes the unchanged source and rebuilds Q10.
    model.initialize()
    _assert_temperature_views_are_current(model, LOW_CELSIUS)

    old_state_temperature = state.celsius
    model.celsius = torch.tensor(HIGH_CELSIUS, dtype=DTYPE)
    assert state.celsius is old_state_temperature
    assert state.celsius is not model.celsius
    torch.testing.assert_close(state.q10, initial_q10)
    model.initialize()
    _assert_temperature_views_are_current(model, LOW_CELSIUS)


def _clone_nested(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {name: _clone_nested(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_nested(item) for item in value)
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    return value


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_fixed_temperature_checkpoint_replays_exact_hh_suffix(kind):
    source = _new_model(kind, HIGH_CELSIUS)
    _advance(source, 3)
    checkpoint = _clone_nested(source.state_dict_for_checkpoint())

    _advance(source, 6)
    expected = {
        name: value.detach().clone() for name, value in _runtime_state(source).items()
    }

    resumed = _new_model(kind, HIGH_CELSIUS)
    resumed.restore_dict_from_checkpoint(checkpoint)
    _assert_temperature_views_are_current(resumed, HIGH_CELSIUS)
    _advance(resumed, 6)
    _assert_runtime_equal(_runtime_state(resumed), expected)


class _SigmoidTemperature(torch.nn.Module):
    def __init__(self, theta):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(theta, dtype=DTYPE))

    def forward(self):
        return 6.3 + 20.0 * torch.sigmoid(self.theta)


def _trajectory_loss(model):
    _advance(model, 12)
    mechanism, _ = _hh_parts(model)
    return (
        0.01 * model.v.square().sum()
        + mechanism.m.square().sum()
        + 0.3 * mechanism.h.sum()
        + 0.2 * mechanism.n.square().sum()
    )


def _updated_temperature_model(theta, *, training):
    temperature = _SigmoidTemperature(0.2)
    model = _new_model("single_compartment", temperature, training=training)
    with torch.no_grad():
        temperature.theta.fill_(theta)
    model.initialize()
    return model, temperature


def test_reinitialized_nonlinear_temperature_gradient_matches_finite_difference():
    target_theta = 1.2
    temperature = _SigmoidTemperature(0.2)
    model = _new_model("single_compartment", temperature, training=True)

    first_loss = _trajectory_loss(model)
    first_loss.backward()
    first_gradient = temperature.theta.grad.detach().clone()
    assert abs(float(first_gradient)) > 1.0e-3

    temperature.theta.grad = None
    with torch.no_grad():
        temperature.theta.fill_(target_theta)
    model.initialize()
    _assert_temperature_views_are_current(model, temperature().detach())
    loss = _trajectory_loss(model)
    loss.backward()
    gradient = temperature.theta.grad.detach().clone()

    eps = 1.0e-4
    plus, _ = _updated_temperature_model(target_theta + eps, training=False)
    minus, _ = _updated_temperature_model(target_theta - eps, training=False)
    finite_difference = (_trajectory_loss(plus) - _trajectory_loss(minus)) / (2 * eps)

    assert abs(float(gradient)) > 1.0e-3
    torch.testing.assert_close(gradient, finite_difference, rtol=2.0e-7, atol=2.0e-8)
    assert model.v.requires_grad

    evaluated, _ = _updated_temperature_model(target_theta, training=False)
    _trajectory_loss(evaluated)
    evaluated_mechanism, _ = _hh_parts(evaluated)
    assert not evaluated.v.requires_grad
    assert not evaluated_mechanism.m.requires_grad
