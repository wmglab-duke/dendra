import math
import sys

import networkx as nx
import pytest
import torch

import dendra as dn
from dendra.models.core import _duration_step_count
from dendra.models.mod import pas
from dendra.models.stim.waveform import constant

DTYPE = torch.float64


def _population(*, N=2, C=3, v_init=-65.0, initialize=True):
    pop = dn.Population(N=N, C=C, v_init=v_init, dtype=DTYPE)
    pop.insert(pas, g=0.001, e=-70.0)
    pop.build()
    if initialize:
        pop.initialize()
    return pop


def _clone_nested(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_nested(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_nested(item) for item in value)
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    return value


@pytest.mark.parametrize(
    "v_init,expected",
    [
        (-60.0, [[-60.0, -60.0, -60.0], [-60.0, -60.0, -60.0]]),
        ([-60.0, -61.0, -62.0], [[-60.0, -61.0, -62.0]] * 2),
        (
            [[-60.0, -61.0, -62.0], [-63.0, -64.0, -65.0]],
            [[-60.0, -61.0, -62.0], [-63.0, -64.0, -65.0]],
        ),
        ([[-60.0, -61.0, -62.0]], [[-60.0, -61.0, -62.0]] * 2),
    ],
)
def test_expanded_v_init_supports_documented_shapes(v_init, expected):
    pop = dn.Population(N=2, C=3, v_init=v_init, dtype=DTYPE)
    assert pop.expanded_v_init().tolist() == expected

    batched = pop.expanded_v_init((4, 2, 3))
    assert batched.shape == (4, 2, 3)
    assert batched[0].tolist() == expected


def test_set_v_init_validates_without_losing_previous_value():
    pop = dn.Population(N=2, C=3, v_init=-65.0, dtype=DTYPE)
    assert pop.set_v_init([-60.0, -61.0, -62.0]) is pop
    previous = pop.v_init

    with pytest.raises(ValueError, match="v_init has length"):
        pop.set_v_init([-1.0, -2.0])
    assert pop.v_init is previous
    with pytest.raises(ValueError, match="at least population and compartment"):
        pop.expanded_v_init((3,))


def test_population_cache_is_a_snapshot_and_restore_recovers_state():
    pop = _population()
    pop.step(dt=0.01)
    assert pop.cache() is pop
    cached_v = pop._caches["latest"]["v"].clone()
    cached_t = pop._caches["latest"]["t"].clone()

    with torch.no_grad():
        pop.v.add_(20.0)
        pop.t.add_(1.0)

    assert torch.equal(pop._caches["latest"]["v"], cached_v)
    assert torch.equal(pop._caches["latest"]["t"], cached_t)
    assert pop.restore() is pop
    assert torch.equal(pop.v, cached_v)
    assert torch.equal(pop.t, cached_t)
    assert pop.initialized


def test_named_cache_aliases_and_missing_cache_validation():
    pop = _population()
    assert pop.cache_("baseline") is None
    pop.v.fill_(-10.0)
    assert pop.restore_("baseline") is None
    assert torch.all(pop.v == -65.0)
    with pytest.raises(KeyError):
        pop.restore("missing")


def test_population_checkpoint_restores_integrator_mechanism_and_time_state():
    pop = _population()
    pop.step(dt=0.01)
    checkpoint = _clone_nested(pop.state_dict_for_checkpoint())
    expected_v = pop.v.clone()
    expected_t = pop.t.clone()

    pop.step(dt=0.01)
    pop.restore_dict_from_checkpoint(checkpoint)

    assert torch.equal(pop.v, expected_v)
    assert torch.equal(pop.t, expected_t)


def test_initialization_hooks_and_set_value_execute_in_order():
    pop = dn.Population(N=1, C=2, v_init=-65.0, dtype=DTYPE)
    pop.insert(pas, g=0.001, e=-70.0)
    calls = []
    pop.register_pre_initialize_hook(lambda model: calls.append(("pre", model)))
    pop.register_post_initialize_hook(lambda model: calls.append(("post", model)))
    pop.set_value("v", torch.tensor([[-55.0, -56.0]], dtype=DTYPE))

    assert pop.initialize() is pop
    assert [name for name, _ in calls] == ["pre", "post"]
    assert all(model is pop for _, model in calls)
    assert pop.v.tolist() == [[-55.0, -56.0]]
    assert pop.initialize_() is None


def test_set_value_reports_unknown_state_during_initialization():
    pop = dn.Population(N=1, C=1, dtype=DTYPE)
    pop.insert(pas, g=0.001, e=-70.0)
    pop.set_value("not_a_state", torch.tensor(1.0))
    with pytest.raises(AttributeError, match="not_a_state"):
        pop.initialize()


def test_population_batch_shapes_state_coordinates_and_rejects_nonpositive_size():
    pop = _population()
    original_v = pop.v.clone()
    assert pop.batch(4) is pop
    assert pop.shape == (4, 2, 3)
    assert pop.is_batched()
    assert pop.core_shape() == (2, 3)
    assert pop.batched_shape() == (8, 3)
    assert pop.n_batch_dimensions() == 1
    assert pop._calc_shape_p() == (1, 2, 3)
    assert torch.equal(pop.v[0], original_v)
    pop.v[0, 0, 0] = 1.0
    assert pop.v[1, 0, 0] != 1.0

    with pytest.raises(ValueError, match="positive"):
        _population().batch(0)


def test_single_contact_tensor_extra_has_known_spatiotemporal_product():
    pop = _population()
    time = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=DTYPE)
    t_global = torch.arange(4, dtype=DTYPE) * 0.1
    spatial = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)

    cfg = pop._prepare_extra((spatial, time), t_global, n_chunks=2)
    first = pop._compute_extra_chunk(cfg, 0, t_global[:2])
    second = pop._compute_extra_chunk(cfg, 1, t_global[2:])

    assert cfg.enabled and not cfg.multicontact and not cfg.functional
    assert len(first) == len(second) == 2
    assert first[0].tolist() == [[1.0, 2.0, 3.0]] * 2
    assert second[1].tolist() == [[4.0, 8.0, 12.0]] * 2


def test_multicontact_and_functional_extra_are_combined_correctly():
    pop = _population()
    t_global = torch.tensor([0.0, 0.1, 0.2], dtype=DTYPE)
    ones = torch.ones(3, dtype=DTYPE)
    twos = torch.full((3,), 2.0, dtype=DTYPE)
    time_a = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)
    time_b = torch.tensor([4.0, 5.0, 6.0], dtype=DTYPE)

    cfg = pop._prepare_extra([(ones, time_a), (twos, time_b)], t_global, n_chunks=1)
    values = pop._compute_extra_chunk(cfg, 0, t_global)
    assert cfg.multicontact and not cfg.functional
    assert values[1].tolist() == [[12.0, 12.0, 12.0]] * 2

    functional = pop._prepare_extra((ones, constant(value=2.0)), t_global, n_chunks=1)
    functional_values = pop._compute_extra_chunk(functional, 0, t_global)
    assert functional.functional
    assert len(functional_values) == 3
    assert functional_values[0].tolist() == [[2.0, 2.0, 2.0]] * 2


def test_extra_normalization_validates_shapes_lengths_and_temporal_types():
    pop = _population()
    t_global = torch.tensor([0.0, 0.1, 0.2], dtype=DTYPE)
    with pytest.raises(ValueError, match="must not be empty"):
        pop._prepare_extra([], t_global, n_chunks=1)
    with pytest.raises(ValueError, match="leading dimension"):
        pop._normalize_spatial(torch.ones(4, 3, dtype=DTYPE))
    with pytest.raises(ValueError, match="compartment"):
        pop._normalize_spatial(torch.ones(2, 4, dtype=DTYPE))
    with pytest.raises(ValueError, match="leading dimension"):
        pop._normalize_time_tensor(torch.ones(4, 3, dtype=DTYPE), t_global)
    with pytest.raises(ValueError, match="length"):
        pop._normalize_time_tensor(torch.ones(2, 4, dtype=DTYPE), t_global)
    with pytest.raises(ValueError, match="chunk length"):
        pop._expand_eval_time(torch.ones(4, dtype=DTYPE), n_steps=3)
    with pytest.raises(ValueError, match="mixing Waveform"):
        pop._prepare_extra(
            [
                (torch.ones(3), constant(value=1.0)),
                (torch.ones(3), torch.ones(3)),
            ],
            t_global,
            n_chunks=1,
        )


def test_terminal_indices_labels_and_pretty_summary():
    graph = nx.DiGraph()
    for node in range(3):
        graph.add_node(
            node,
            name=f"Cell.dend[{node}](0.5)",
            L=10.0,
            diam=1.0,
            Ra=100.0,
            cm=1.0,
            area=31.4,
            x=float(node),
            y=0.0,
            z=0.0,
        )
    graph.add_edge(0, 1, R_ohm=1e5, L=10.0, diff_geom_um=1.0)
    graph.add_edge(0, 2, R_ohm=1e5, L=10.0, diff_geom_um=1.0)
    pop = dn.Tree.from_graph(graph, N=1)
    pop.insert(pas, g=0.001, e=-70.0)
    pop.slice("dend[1]").label("branch_one")

    assert pop.terminal_indices() == [1, 2]
    not_root = pop.find_not("dend[0]")
    if isinstance(not_root, slice):
        not_root = list(range(*not_root.indices(3)))
    else:
        not_root = not_root.tolist()
    assert not_root == [1, 2]
    pending = pop.pretty(show_kwargs=True)
    assert "pas" in pending and pending.startswith("Tree {")

    pop.initialize()
    built = pop.pretty(
        show_mechanism_parameters=True, tensor_stats="full", show_kwargs=True
    )
    assert "pas" in built and "parameters {" in built
    pop.clear_labels()
    assert not hasattr(pop, "branch_one")


def test_insert_rejects_invalid_copies_and_duplicate_everywhere_mechanism():
    pop = dn.Population(N=1, C=2, dtype=DTYPE)
    with pytest.raises(ValueError, match="region-restricted"):
        pop.insert(pas, copies=2)
    with pytest.raises(ValueError, match="positive integer"):
        pop[:, 0].insert(pas.rename("bad"), copies=0)

    pop.insert(pas, g=0.001, e=-70.0)
    with pytest.raises(ValueError, match="already inserted everywhere"):
        pop[:, 0].insert(pas, g=0.002, e=-65.0)


def test_checkpointed_longrun_matches_regular_run_and_returns_resumable_state():
    regular = _population(N=1, C=2)
    checkpointed = _population(N=1, C=2)

    regular.run(tstop=0.04, dt=0.01)
    loss, final_state = checkpointed.longrun_checkpointed(
        tstop=0.04,
        chunklength=2,
        dt=0.01,
        safe_checkpoint=True,
        return_final_state=True,
    )

    assert loss is None
    assert torch.allclose(checkpointed.v, regular.v)
    assert checkpointed.t.item() == pytest.approx(regular.t.item())
    checkpointed.step(dt=0.01)
    checkpointed.restore_dict_from_checkpoint(final_state)
    assert torch.allclose(checkpointed.v, regular.v)


def test_checkpointed_longrun_callback_loss_backpropagates_and_restores_final_state():
    class VoltageLoss(dn.callbacks.Callback):
        def post_step_hook(self, model):
            return model.v.square().mean()

    with dn.ctx(REQUIRE_GRAD=1):
        pop = _population(N=1, C=2)
    pop.train()
    loss, final_state = pop.longrun_checkpointed(
        tstop=0.03,
        chunklength=2,
        dt=0.01,
        callbacks=[VoltageLoss()],
        safe_checkpoint=True,
        restore_state_after_backward=True,
        return_final_state=True,
    )
    expected_t = pop.t.clone()

    assert loss.requires_grad
    assert final_state is not None
    loss.backward()
    assert pop.mech.pas.g_param.grad is not None
    assert torch.isfinite(pop.mech.pas.g_param.grad)
    assert torch.equal(pop.t, expected_t)


def test_checkpointed_longrun_validates_chunk_length_and_empty_run_contract():
    pop = _population(N=1, C=1)
    with pytest.raises(ValueError, match="chunklength"):
        pop.longrun_checkpointed(tstop=0.1, chunklength=0, dt=0.01)

    loss, state = pop.longrun_checkpointed(
        tstop=0.0, chunklength=1, dt=0.01, return_final_state=True
    )
    assert loss is None
    assert state["t"].item() == pytest.approx(0.0)


def test_population_run_and_longrun_share_complete_step_callback_semantics():
    class Lifecycle(dn.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.pre = 0
            self.post_times = []
            self.pre_chunks = 0
            self.post_chunks = 0

        def pre_step_hook(self, model):
            self.pre += 1

        def post_step_hook(self, model):
            self.post_times.append(float(model.t))

        def pre_chunk_hook(self, model, chunk):
            self.pre_chunks += 1

        def post_chunk_hook(self, model, chunk):
            self.post_chunks += 1

    run_pop = _population(N=1, C=1)
    run_cb = Lifecycle()
    run_pop.run(tstop=0.03, dt=0.01, callbacks=[run_cb])
    assert run_cb.pre == 3
    assert run_cb.post_times == pytest.approx([0.01, 0.02, 0.03])

    long_pop = _population(N=1, C=1)
    long_cb = Lifecycle()
    long_pop.longrun(tstop=0.03, chunklength=2, dt=0.01, callbacks=[long_cb])
    assert long_cb.pre == 3
    assert long_cb.post_times == pytest.approx([0.01, 0.02, 0.03])
    assert long_cb.pre_chunks == long_cb.post_chunks == 2

    with pytest.raises(ValueError, match="chunklength"):
        long_pop.longrun(tstop=0.1, chunklength=0, dt=0.01)


def test_repeated_run_duration_is_independent_of_accumulated_time_roundoff():
    class StepCounter(dn.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.post_times = []

        def post_step_hook(self, model):
            self.post_times.append(float(model.t))

    pop = _population(N=1, C=1)
    callback = StepCounter()
    expected_time = 0.0

    # At the fourth call, naive arange(start, start + dt, dt) can include the
    # nominally excluded endpoint and execute two 0.09 ms steps.
    for step_index, dt in enumerate((0.04, 0.04, 0.09, 0.09, 0.09), start=1):
        pop.run(tstop=dt, dt=dt, callbacks=[callback])
        expected_time += dt
        assert len(callback.post_times) == step_index
        assert pop.t.item() == pytest.approx(expected_time, abs=1e-15)

    assert callback.post_times == pytest.approx(
        [0.04, 0.08, 0.17, 0.26, 0.35], abs=1e-15
    )


def test_positive_subnormal_duration_still_requests_one_step():
    assert _duration_step_count(math.ulp(0.0), sys.float_info.max) == 1


@pytest.mark.parametrize(
    "duration,expected_steps",
    [(0.005, 1), (0.015, 2), (0.07, 7), (0.071, 8)],
)
def test_run_and_longrun_share_half_open_duration_step_count(duration, expected_steps):
    class StepCounter(dn.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.steps = 0

        def post_step_hook(self, model):
            self.steps += 1

    run_pop = _population(N=1, C=1)
    long_pop = _population(N=1, C=1)
    checkpointed_pop = _population(N=1, C=1)
    for model in (run_pop, long_pop, checkpointed_pop):
        for dt in (0.04, 0.04, 0.09):
            model.step(dt=dt)

    run_callback = StepCounter()
    long_callback = StepCounter()
    run_pop.run(tstop=duration, dt=0.01, callbacks=[run_callback])
    long_pop.longrun(
        tstop=duration,
        chunklength=3,
        dt=0.01,
        callbacks=[long_callback],
    )
    _, final_state = checkpointed_pop.longrun_checkpointed(
        tstop=duration,
        chunklength=3,
        dt=0.01,
        safe_checkpoint=True,
        return_final_state=True,
    )

    assert run_callback.steps == long_callback.steps == expected_steps
    torch.testing.assert_close(run_pop.t, long_pop.t, rtol=0.0, atol=0.0)
    torch.testing.assert_close(run_pop.t, checkpointed_pop.t, rtol=0.0, atol=0.0)
    torch.testing.assert_close(final_state["t"], checkpointed_pop.t)
    torch.testing.assert_close(run_pop.v, long_pop.v, rtol=0.0, atol=0.0)
    torch.testing.assert_close(run_pop.v, checkpointed_pop.v, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "duration,dt",
    [(sys.float_info.max, sys.float_info.min), (1.0e20, 1.0)],
)
def test_run_variants_reject_unrepresentable_step_counts_without_advancing(
    duration, dt
):
    for method in ("run", "longrun", "longrun_checkpointed"):
        pop = _population(N=1, C=1)
        initial_v = pop.v.clone()
        initial_t = pop.t.clone()

        with pytest.raises(ValueError, match="too many simulation steps"):
            if method == "run":
                pop.run(tstop=duration, dt=dt)
            elif method == "longrun":
                pop.longrun(tstop=duration, chunklength=3, dt=dt)
            else:
                pop.longrun_checkpointed(
                    tstop=duration,
                    chunklength=3,
                    dt=dt,
                    safe_checkpoint=True,
                )

        torch.testing.assert_close(pop.t, initial_t, rtol=0.0, atol=0.0)
        torch.testing.assert_close(pop.v, initial_v, rtol=0.0, atol=0.0)
