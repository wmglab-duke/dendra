"""Shape contracts for extracellular inputs before and after ``batch()``."""

import pytest
import torch

import dendra as dn
from dendra.models.networks.net import prepare_extra

DTYPE = torch.float64


def _population(*, batch=None):
    pop = dn.Population(N=2, C=3, dtype=DTYPE)
    pop.build()
    if batch is not None:
        pop.batch(batch)
    pop.initialize(force_rebuild=True)
    return pop


def _capture_population_steps(pop):
    captured = []

    def capture(integrator, model, dt, ve, intra):
        captured.append(None if ve is None else ve.detach().clone())

    pop._step = capture
    return captured


def _network(*, batch=None, dt=0.25):
    network = dn.Network({"cells": dn.Population(N=2, C=3, dtype=DTYPE)})
    if batch is not None:
        network.batch(batch)
    network.initialize(dt)
    return network


def _capture_network_steps(network):
    captured = []

    def capture(*args, extra=None, intra=None, **kwargs):
        captured.append(
            {
                name: value.detach().clone()
                for name, value in (extra if extra is not None else {}).items()
            }
        )

    network._step = capture
    return captured


def test_population_spatial_and_temporal_contract_covers_batch_fallbacks():
    pop = _population(batch=4)
    shared_spatial = torch.arange(6, dtype=DTYPE).reshape(2, 3)
    spatial = pop._normalize_spatial(shared_spatial)
    assert spatial.shape == (4, 2, 3)
    torch.testing.assert_close(spatial[0], shared_spatial)
    torch.testing.assert_close(spatial[3], shared_spatial)

    batch_spatial = pop._normalize_spatial(torch.arange(4, dtype=DTYPE))
    assert batch_spatial.shape == (4, 2, 3)
    torch.testing.assert_close(batch_spatial[:, 0, 0], torch.arange(4, dtype=DTYPE))

    time = torch.arange(3, dtype=DTYPE)
    batch_temporal = torch.arange(12, dtype=DTYPE).reshape(4, 3)
    temporal = pop._normalize_time_tensor(batch_temporal, time)
    assert temporal.shape == (4, 2, 3)
    torch.testing.assert_close(temporal[:, 0], batch_temporal)
    torch.testing.assert_close(temporal[:, 1], batch_temporal)


def test_equal_batch_and_compartment_sizes_preserve_spatial_suffix_meaning():
    pop = _population(batch=3)
    compartment_values = torch.tensor([10.0, 20.0, 30.0], dtype=DTYPE)

    shared = pop._normalize_spatial(compartment_values)
    expected_shared = compartment_values.expand(3, 2, 3)
    torch.testing.assert_close(shared, expected_shared)

    explicit_batch = pop._normalize_spatial(compartment_values[:, None, None])
    expected_batch = compartment_values[:, None, None].expand(3, 2, 3)
    torch.testing.assert_close(explicit_batch, expected_batch)


def test_repeated_batching_preserves_np_nc_suffix_and_explicit_batch_grid():
    pop = dn.Population(N=2, C=3, dtype=DTYPE)
    pop.batch(3).batch(2)
    assert pop.shape == (2, 3, 2, 3)

    np_nc = torch.arange(6, dtype=DTYPE).reshape(2, 3)
    shared = pop._normalize_spatial(np_nc)
    expected_shared = np_nc.expand(2, 3, 2, 3)
    torch.testing.assert_close(shared, expected_shared)

    batch_grid = (10.0 + torch.arange(6, dtype=DTYPE)).reshape(2, 3)
    explicit_batch = pop._normalize_spatial(batch_grid[..., None, None])
    expected_batch = batch_grid[..., None, None].expand(2, 3, 2, 3)
    torch.testing.assert_close(explicit_batch, expected_batch)


def test_multidimensional_batch_fallback_right_aligns_inside_batch_shape():
    pop = dn.Population(N=2, C=3, dtype=DTYPE)
    pop.batch(5).batch(4)
    assert pop.shape == (4, 5, 2, 3)

    inner_batch = torch.arange(5, dtype=DTYPE)
    spatial = pop._normalize_spatial(inner_batch)
    expected_spatial = inner_batch[None, :, None, None].expand(4, 5, 2, 3)
    torch.testing.assert_close(spatial, expected_spatial)

    time = torch.arange(3, dtype=DTYPE)
    temporal_input = torch.arange(15, dtype=DTYPE).reshape(5, 3)
    temporal = pop._normalize_time_tensor(temporal_input, time)
    expected_temporal = temporal_input[None, :, None, :].expand(4, 5, 2, 3)
    torch.testing.assert_close(temporal, expected_temporal)


def test_equal_batch_and_population_sizes_preserve_temporal_suffix_meaning():
    pop = _population(batch=2)
    time = torch.arange(3, dtype=DTYPE)
    np_time = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=DTYPE)

    shared = pop._normalize_time_tensor(np_time, time)
    expected_shared = np_time.expand(2, 2, 3)
    torch.testing.assert_close(shared, expected_shared)

    explicit_batch = pop._normalize_time_tensor(np_time[:, None, :], time)
    expected_batch = np_time[:, None, :].expand(2, 2, 3)
    torch.testing.assert_close(explicit_batch, expected_batch)


def test_population_extra_executes_multiple_explicit_batch_dimensions():
    pop = dn.Population(N=2, C=3, dtype=DTYPE)
    pop.build()
    pop.batch(2).batch(3)
    pop.initialize(force_rebuild=True)
    captured = _capture_population_steps(pop)
    spatial = torch.arange(6, dtype=DTYPE).reshape(2, 3)
    temporal = torch.arange(12, dtype=DTYPE).reshape(3, 2, 1, 2)

    pop.run(tstop=0.2, dt=0.1, extra=(spatial, temporal))

    actual = torch.stack(captured)
    expected = temporal[..., 0, :].movedim(-1, 0)[..., None, None] * spatial
    assert actual.shape == (2, 3, 2, 2, 3)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("mode", ["run", "longrun", "checkpointed"])
@pytest.mark.parametrize("functional", [False, True])
def test_batched_population_extra_entrypoints_preserve_batch_sweeps(mode, functional):
    pop = _population(batch=2)
    captured = _capture_population_steps(pop)
    spatial = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=DTYPE)
    temporal = torch.tensor([[1.0, 2.0, 3.0], [5.0, 7.0, 11.0]], dtype=DTYPE)
    time_spec = (
        dn.constant(value=temporal[:, :1, None]) if functional else temporal[:, None, :]
    )

    if mode == "run":
        pop.run(tstop=0.3, dt=0.1, extra=(spatial, time_spec))
    elif mode == "longrun":
        pop.longrun(tstop=0.3, chunklength=2, dt=0.1, extra=(spatial, time_spec))
    else:
        pop.longrun_checkpointed(
            tstop=0.3,
            chunklength=2,
            dt=0.1,
            extra=(spatial, time_spec),
            safe_checkpoint=True,
        )

    actual = torch.stack(captured)
    factors = temporal[:, :1].expand(-1, 3) if functional else temporal
    expected = factors.T[:, :, None, None] * spatial
    assert actual.shape == (3, 2, 2, 3)
    torch.testing.assert_close(actual, expected)


def test_population_step_extra_and_direct_ve_use_the_same_batch_fallback_rule():
    extra_pop = _population(batch=2)
    extra_captured = _capture_population_steps(extra_pop)
    extra_pop.step(
        dt=0.1,
        extra=(
            torch.ones(2, 3, dtype=DTYPE),
            dn.constant(value=[[[2.0]], [[5.0]]]),
        ),
    )

    ve_pop = _population(batch=2)
    ve_captured = _capture_population_steps(ve_pop)
    ve_pop.step(dt=0.1, ve=torch.tensor([2.0, 5.0], dtype=DTYPE))

    expected = torch.tensor([2.0, 5.0], dtype=DTYPE)[:, None, None].expand(2, 2, 3)
    torch.testing.assert_close(extra_captured[0], expected)
    torch.testing.assert_close(ve_captured[0], expected)


def test_population_precomputed_ve_is_time_first_then_batch_prefixed():
    pop = _population(batch=2)
    captured = _capture_population_steps(pop)
    ve = torch.tensor([[1.0, 4.0], [2.0, 8.0], [3.0, 12.0]], dtype=DTYPE)

    pop.run(ve=ve, dt=0.1)

    expected = ve[:, :, None, None].expand(3, 2, 2, 3)
    torch.testing.assert_close(torch.stack(captured), expected)


def test_precomputed_ve_preserves_time_and_equal_compartment_suffix_axes():
    pop = _population(batch=3)
    captured = _capture_population_steps(pop)
    ve = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=DTYPE)

    pop.run(ve=ve, dt=0.1)

    actual = torch.stack(captured)
    expected = ve[:, None, None, :].expand(2, 3, 2, 3)
    torch.testing.assert_close(actual, expected)


def test_precomputed_ve_uses_singletons_for_ambiguous_batch_intent():
    pop = _population(batch=3)
    captured = _capture_population_steps(pop)
    batch_values = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=DTYPE)
    explicit_batch_ve = batch_values[..., None, None]

    pop.run(ve=explicit_batch_ve, dt=0.1)

    actual = torch.stack(captured)
    expected = explicit_batch_ve.expand(2, 3, 2, 3)
    torch.testing.assert_close(actual, expected)


def test_repeated_batch_precomputed_ve_preserves_np_nc_payload_suffix():
    pop = dn.Population(N=2, C=3, dtype=DTYPE)
    pop.build()
    pop.batch(3).batch(2)
    pop.initialize(force_rebuild=True)
    captured = _capture_population_steps(pop)
    ve = torch.arange(12, dtype=DTYPE).reshape(2, 2, 3)

    pop.run(ve=ve, dt=0.1)

    actual = torch.stack(captured)
    expected = ve[:, None, None, :, :].expand(2, 2, 3, 2, 3)
    torch.testing.assert_close(actual, expected)


def test_population_extra_shape_errors_name_received_and_target_shapes():
    pop = _population(batch=2)
    with pytest.raises(ValueError, match=r"ve shape \(2, 4\).*model shape \(2, 2, 3\)"):
        pop.step(dt=0.1, ve=torch.ones(2, 4, dtype=DTYPE))
    with pytest.raises(
        ValueError,
        match=r"Precomputed ve per-step shape \(2, 4\).*model shape \(2, 2, 3\)",
    ):
        pop.run(ve=torch.ones(3, 2, 4, dtype=DTYPE), dt=0.1)
    with pytest.raises(ValueError, match=r"ve_s.*model shape \(2, 2, 3\)"):
        pop._normalize_spatial(torch.ones(2, 4, dtype=DTYPE))


def test_prepare_extra_indexes_trailing_time_after_normalization():
    spatial = torch.arange(12, dtype=DTYPE).reshape(2, 2, 3)
    temporal = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        dtype=DTYPE,
    )

    first = prepare_extra({"cells": (spatial, temporal)}, 0)["cells"]
    second = prepare_extra({"cells": (spatial, temporal)}, 1)["cells"]

    torch.testing.assert_close(first, spatial * temporal[..., 0, None])
    torch.testing.assert_close(second, spatial * temporal[..., 1, None])


@pytest.mark.parametrize("mode", ["run", "longrun", "checkpointed"])
def test_batched_population_multicontact_extra_executes_in_every_chunked_lane(mode):
    pop = _population(batch=2)
    captured = _capture_population_steps(pop)
    spatial_shared = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=DTYPE)
    spatial_batched = torch.stack((spatial_shared, 2.0 * spatial_shared))
    temporal_batched = torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], dtype=DTYPE)
    temporal_shared = torch.tensor([7.0, 8.0, 9.0], dtype=DTYPE)
    extra = [
        (spatial_shared, temporal_batched),
        (spatial_batched, temporal_shared),
    ]

    if mode == "run":
        pop.run(tstop=0.3, dt=0.1, extra=extra)
    elif mode == "longrun":
        pop.longrun(tstop=0.3, chunklength=2, dt=0.1, extra=extra)
    else:
        pop.longrun_checkpointed(
            tstop=0.3,
            chunklength=2,
            dt=0.1,
            extra=extra,
            safe_checkpoint=True,
        )

    actual = torch.stack(captured)
    expected = (
        temporal_batched[:, 0].T[:, :, None, None] * spatial_shared
        + temporal_shared[:, None, None, None] * spatial_batched
    )
    assert actual.shape == (3, 2, 2, 3)
    torch.testing.assert_close(actual, expected)


def test_population_multicontact_tuple_of_tuples_matches_list_form():
    pop = _population(batch=2)
    time = torch.tensor([0.0, 0.1, 0.2], dtype=DTYPE)
    spatial_a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=DTYPE)
    spatial_b = torch.stack((spatial_a, 2.0 * spatial_a))
    temporal_a = torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], dtype=DTYPE)
    temporal_b = torch.tensor([7.0, 8.0, 9.0], dtype=DTYPE)
    contacts = [(spatial_a, temporal_a), (spatial_b, temporal_b)]

    list_cfg = pop._prepare_extra(contacts, time, n_chunks=1)
    tuple_cfg = pop._prepare_extra(tuple(contacts), time, n_chunks=1)
    list_values = torch.stack(pop._compute_extra_chunk(list_cfg, 0, time))
    tuple_values = torch.stack(pop._compute_extra_chunk(tuple_cfg, 0, time))

    assert list_cfg.multicontact
    assert tuple_cfg.multicontact
    torch.testing.assert_close(tuple_values, list_values)


def test_batched_population_extra_preserves_spatial_and_temporal_gradients():
    pop = _population(batch=2)
    pop.train()
    captured = []

    def capture(integrator, model, dt, ve, intra):
        captured.append(ve)

    pop._step = capture
    spatial = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=DTYPE,
        requires_grad=True,
    )
    temporal = torch.tensor(
        [[[1.0, 2.0, 3.0]], [[5.0, 7.0, 11.0]]],
        dtype=DTYPE,
        requires_grad=True,
    )

    pop.run(tstop=0.3, dt=0.1, extra=(spatial, temporal))
    torch.stack(captured).sum().backward()

    expected_spatial_grad = torch.full_like(spatial, temporal.detach().sum())
    expected_temporal_grad = torch.full_like(temporal, spatial.detach().sum())
    torch.testing.assert_close(spatial.grad, expected_spatial_grad)
    torch.testing.assert_close(temporal.grad, expected_temporal_grad)


@pytest.mark.parametrize("mode", ["step", "run", "longrun", "checkpointed"])
def test_batched_network_frequency_sweep_is_shared_across_each_replica(mode):
    network = _network(batch=2)
    captured = _capture_network_steps(network)
    spatial = torch.ones(2, 3, dtype=DTYPE)
    waveform = dn.mono_rect(amp=1.0, pw=0.2).repeat(
        torch.tensor([[1.0], [2.0]], dtype=DTYPE)
    )
    extra = {"cells": (spatial, waveform)}

    if mode == "step":
        for _ in range(3):
            network.step(extra=extra)
    elif mode == "run":
        network.run(0.75, extra=extra)
    elif mode == "longrun":
        network.longrun(0.75, chunklength=2, extra=extra)
    else:
        network.longrun_checkpointed(
            0.75,
            chunklength=2,
            extra=extra,
            safe_checkpoint=True,
        )

    actual = torch.stack([item["cells"] for item in captured])
    expected_factors = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    expected = expected_factors[:, :, None, None].expand(3, 2, 2, 3)
    assert actual.shape == (3, 2, 2, 3)
    torch.testing.assert_close(actual, expected)


def test_regular_network_population_specific_waveform_remains_supported():
    network = _network()
    captured = _capture_network_steps(network)
    spatial = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=DTYPE)
    waveform = dn.constant(value=torch.tensor([2.0, 5.0], dtype=DTYPE))

    network.run(0.25, extra={"cells": (spatial, waveform)})

    expected = spatial * torch.tensor([2.0, 5.0], dtype=DTYPE)[:, None]
    torch.testing.assert_close(captured[0]["cells"], expected)


def test_batched_network_equal_population_size_preserves_population_waveform_axis():
    network = _network(batch=2)
    captured = _capture_network_steps(network)
    spatial = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=DTYPE)
    waveform = dn.constant(value=torch.tensor([2.0, 5.0], dtype=DTYPE))

    network.run(0.25, extra={"cells": (spatial, waveform)})

    per_population = spatial * torch.tensor([2.0, 5.0], dtype=DTYPE)[:, None]
    expected = per_population.expand(2, 2, 3)
    torch.testing.assert_close(captured[0]["cells"], expected)


@pytest.mark.parametrize("mode", ["step", "run", "longrun", "checkpointed"])
def test_network_shared_waveform_respects_each_population_dtype_in_all_lanes(mode):
    network = dn.Network(
        {
            "f32": dn.Population(N=1, C=1, dtype=torch.float32),
            "f64": dn.Population(N=1, C=1, dtype=torch.float64),
        }
    )
    network.batch(2)
    network.initialize(0.1)
    captured = _capture_network_steps(network)
    waveform = dn.constant(value=torch.tensor([[[1.0]], [[2.0]]], dtype=torch.float32))
    extra = {
        "f32": (torch.ones(1, 1), waveform),
        "f64": (torch.ones(1, 1), waveform),
    }

    if mode == "step":
        network.step(extra=extra)
    elif mode == "run":
        network.run(0.1, extra=extra)
    elif mode == "longrun":
        network.longrun(0.1, chunklength=1, extra=extra)
    else:
        network.longrun_checkpointed(
            0.1,
            chunklength=1,
            extra=extra,
            safe_checkpoint=True,
        )

    assert len(captured) == 1
    expected_values = torch.tensor([1.0, 2.0])[:, None, None]
    for name, dtype in (("f32", torch.float32), ("f64", torch.float64)):
        actual = captured[0][name]
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, expected_values.to(dtype=dtype))


def test_network_rejects_temporal_shapes_outside_batch_population_contract():
    network = _network(batch=2)
    spatial = torch.ones(2, 3, dtype=DTYPE)
    malformed = dn.constant(value=torch.ones(2, 4, dtype=DTYPE))

    with pytest.raises(
        ValueError,
        match=r"Waveform output leading dimension shape \(2, 4\).*\(2, 2\)",
    ):
        network.run(0.25, extra={"cells": (spatial, malformed)})
