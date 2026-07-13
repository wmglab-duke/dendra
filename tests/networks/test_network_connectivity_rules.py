"""Invariant tests for public NEST-style network connectivity rules."""

from __future__ import annotations

from collections import Counter

import pytest
import torch

import dendra as dn
from dendra.models.mod import expsyn


def _rule_network(*, n_pre=4, n_post=4, same_population=False, seed=1729):
    if same_population:
        population = dn.Population(N=n_pre, C=1, v_init=-65.0, dtype=torch.float64)
        population.insert(expsyn.rename("rule_syn"), e=0.0, tau=1.0)
        network = dn.Network({"population": population}, seed=seed)
        return network, population, population, population.mech.rule_syn

    pre = dn.Population(N=n_pre, C=1, v_init=-65.0, dtype=torch.float64)
    post = dn.Population(N=n_post, C=1, v_init=-65.0, dtype=torch.float64)
    post.insert(expsyn.rename("rule_syn"), e=0.0, tau=1.0)
    network = dn.Network({"pre": pre, "post": post}, seed=seed)
    return network, pre, post, post.mech.rule_syn


def _queued_edges(network):
    if not network.synapse_spec:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty
    assert len(network.synapse_spec) == 1
    specs = next(iter(network.synapse_spec.values()))
    return torch.cat([spec[0] for spec in specs]), torch.cat(
        [spec[1] for spec in specs]
    )


@pytest.mark.parametrize("size", [1, 2, 5])
def test_all_to_all_rule_has_exact_unique_non_autaptic_pairs(size):
    network, population, _, synapse = _rule_network(
        n_pre=size,
        same_population=True,
    )

    network.connect(
        population[:],
        population[:],
        synapse,
        conn_spec={"rule": "all_to_all"},
        allow_autapses=False,
        allow_multapses=False,
    )

    pre_idx, post_idx = _queued_edges(network)
    pairs = list(zip(pre_idx.tolist(), post_idx.tolist()))
    assert len(pairs) == size * max(0, size - 1)
    assert len(set(pairs)) == len(pairs)
    assert all(source != target for source, target in pairs)


@pytest.mark.parametrize("probability, expected_edges", [(0.0, 0), (1.0, 12)])
def test_pairwise_bernoulli_boundary_probabilities_are_exact(
    probability,
    expected_edges,
):
    network, pre, post, synapse = _rule_network(n_pre=3, n_post=4)

    network.connect(
        pre[:],
        post[:],
        synapse,
        conn_spec={"rule": "pairwise_bernoulli", "p": probability},
    )

    pre_idx, post_idx = _queued_edges(network)
    assert pre_idx.numel() == post_idx.numel() == expected_edges


@pytest.mark.parametrize(
    "rule, degree_key",
    [("fixed_indegree", "indegree"), ("fixed_outdegree", "outdegree")],
)
def test_fixed_degree_rules_have_exact_degrees_without_duplicates_or_autapses(
    rule,
    degree_key,
):
    network, population, _, synapse = _rule_network(same_population=True)

    network.connect(
        population[:],
        population[:],
        synapse,
        conn_spec={"rule": rule, degree_key: 2},
        allow_autapses=False,
        allow_multapses=False,
    )

    pre_idx, post_idx = _queued_edges(network)
    pairs = list(zip(pre_idx.tolist(), post_idx.tolist()))
    assert len(pairs) == 8
    assert len(set(pairs)) == len(pairs)
    assert all(source != target for source, target in pairs)
    counted = Counter(
        post_idx.tolist() if rule == "fixed_indegree" else pre_idx.tolist()
    )
    assert counted == Counter({index: 2 for index in range(4)})


@pytest.mark.parametrize(
    "conn_spec",
    [
        {"rule": "fixed_total_number", "N": 7},
        {"rule": "pairwise_bernoulli", "p": 0.45},
        {"rule": "pairwise_poisson", "pairwise_avg_num_conns": 0.75},
        {"rule": "fixed_indegree", "indegree": 2},
        {"rule": "fixed_outdegree", "outdegree": 2},
    ],
)
def test_stochastic_connection_rules_replay_from_the_network_seed(conn_spec):
    generated = []
    for _ in range(2):
        network, pre, post, synapse = _rule_network(seed=2468)
        network.connect(
            pre[:],
            post[:],
            synapse,
            conn_spec=conn_spec,
            allow_multapses=conn_spec["rule"] == "pairwise_poisson",
        )
        generated.append(_queued_edges(network))

    torch.testing.assert_close(generated[0][0], generated[1][0])
    torch.testing.assert_close(generated[0][1], generated[1][1])


def test_fixed_total_rule_rejects_negative_counts_and_honors_zero():
    network, pre, post, synapse = _rule_network()
    with pytest.raises(ValueError, match="non-negative"):
        network.connect(
            pre[:],
            post[:],
            synapse,
            conn_spec={"rule": "fixed_total_number", "N": -1},
        )
    assert network.synapse_spec == {}

    network.connect(
        pre[:],
        post[:],
        synapse,
        conn_spec={"rule": "fixed_total_number", "N": 0},
    )
    assert network.synapse_spec == {}


@pytest.mark.parametrize(
    "conn_spec, message",
    [
        ({"rule": "pairwise_bernoulli", "p": -0.1}, "0 <= p <= 1"),
        ({"rule": "pairwise_bernoulli", "p": 1.1}, "0 <= p <= 1"),
        (
            {"rule": "pairwise_poisson", "pairwise_avg_num_conns": -0.1},
            "non-negative mean",
        ),
        ({"rule": "fixed_indegree", "indegree": -1}, "non-negative"),
        ({"rule": "fixed_outdegree", "outdegree": -1}, "non-negative"),
    ],
)
def test_invalid_connection_rule_parameters_fail_before_queuing(conn_spec, message):
    network, pre, post, synapse = _rule_network()

    with pytest.raises(ValueError, match=message):
        network.connect(pre[:], post[:], synapse, conn_spec=conn_spec)

    assert network.synapse_spec == {}
