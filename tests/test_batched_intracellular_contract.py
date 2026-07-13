"""Shape and execution contracts for batched intracellular stimulation."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mod import pas
from dendra.models.stim.intra import Intra

DTYPE = torch.float64
DT = 0.25
BATCH_SIZE = 3


class _FakeModel:
    def __init__(self, shape):
        self.v = torch.zeros(shape, dtype=DTYPE)

    def device(self):
        return self.v.device

    def dtype(self):
        return self.v.dtype


def _target(population):
    """Select one cell/compartment while retaining every logical axis."""
    batch_axes = (slice(None),) * (len(population.shape) - 2)
    return population[batch_axes + (slice(0, 1), slice(0, 1))]


def _sweep_waveform(batched):
    frequency = torch.tensor([1.0, 2.0, 4.0], dtype=DTYPE) if batched else 1.0
    return dn.mono_rect(amp=1.0, pw=0.1, tau=0.01).repeat(frequency)


def _expected_currents(batched):
    if not batched:
        return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE).reshape(4, 1, 1)
    return torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=DTYPE,
    ).reshape(4, BATCH_SIZE, 1, 1)


def _population(batch_order):
    population = dn.Population(N=1, C=1, dtype=DTYPE)
    population.insert(pas, g=0.0, e=-65.0)
    batched = batch_order != "unbatched"

    if batch_order == "before":
        _target(population).inject(_sweep_waveform(batched=True))

    population.build()
    if batched:
        population.batch(BATCH_SIZE)

    if batch_order in ("unbatched", "after"):
        _target(population).inject(_sweep_waveform(batched=batched))

    population.initialize(force_rebuild=True)
    return population


def _network(batch_order):
    population = dn.Population(N=1, C=1, dtype=DTYPE)
    population.insert(pas, g=0.0, e=-65.0)
    batched = batch_order != "unbatched"

    if batch_order == "before":
        _target(population).inject(_sweep_waveform(batched=True))

    network = dn.Network({"cell": population})
    if batched:
        network.batch(BATCH_SIZE)

    if batch_order in ("unbatched", "after"):
        _target(network.cell).inject(_sweep_waveform(batched=batched))

    network.initialize(DT)
    return network


def _run_population(population, runner, currents):
    def capture_step(integrator, model, dt, ve, intra):
        currents.append(intra.detach().clone())

    population._step = capture_step
    if runner == "step":
        for _ in range(4):
            population.step(dt=DT)
    elif runner == "run":
        population.run(tstop=1.0, dt=DT)
    elif runner == "longrun":
        population.longrun(tstop=1.0, dt=DT, chunklength=3)
    else:
        population.longrun_checkpointed(
            tstop=1.0,
            dt=DT,
            chunklength=3,
            safe_checkpoint=True,
        )


def _run_network(network, runner, currents):
    def capture_step(*args, **kwargs):
        currents.append(kwargs["intra"]["cell"].detach().clone())

    network._step = capture_step
    if runner == "step":
        for _ in range(4):
            network.step()
    elif runner == "run":
        network.run(1.0)
    elif runner == "longrun":
        network.longrun(1.0, chunklength=3)
    else:
        network.longrun_checkpointed(
            1.0,
            chunklength=3,
            safe_checkpoint=True,
        )


def test_intra_trailing_broadcast_precedes_explicit_batch_broadcast():
    model = _FakeModel((3, 1, 3))
    intra = Intra(
        model,
        [(dn.constant(value=1.0), None, (slice(None), slice(None), slice(None)))],
    )

    # B == C is deliberately ambiguous. Preserve legacy PyTorch semantics:
    # trailing broadcasting wins, so [C] remains compartment-aligned.
    compartment = intra([torch.tensor([10.0, 20.0, 30.0], dtype=DTYPE)], intra.indices)
    torch.testing.assert_close(
        compartment,
        torch.tensor([10.0, 20.0, 30.0], dtype=DTYPE)
        .reshape(1, 1, 3)
        .expand_as(compartment),
    )

    # Explicit singleton spatial axes make batch intent unambiguous even when
    # the batch and compartment dimensions have the same size.
    batch = intra(
        [torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE).reshape(3, 1, 1)],
        intra.indices,
    )
    torch.testing.assert_close(
        batch,
        torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE).reshape(3, 1, 1).expand_as(batch),
    )

    with pytest.raises(ValueError, match=r"sample shape \(4,\).*selected model shape"):
        intra([torch.ones(4, dtype=DTYPE)], intra.indices)


def test_soma_sweep_falls_back_to_explicit_batch_broadcast():
    model = _FakeModel((BATCH_SIZE, 1, 1))
    intra = Intra(
        model,
        [(dn.constant(value=1.0), None, (slice(None), slice(None), slice(None)))],
    )

    # [B] cannot trailing-broadcast to [B, 1, 1], so it is right-aligned
    # within the explicit batch shape and padded with singleton spatial axes.
    sweep = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)
    actual = intra([sweep], intra.indices)
    torch.testing.assert_close(actual, sweep.reshape(BATCH_SIZE, 1, 1))


def test_batch_fallback_is_disabled_when_an_integer_removes_the_batch_axis():
    model = _FakeModel((2, 3, 1))
    index = (0, slice(None), slice(None))
    intra = Intra(
        model,
        [(dn.constant(value=1.0), tuple(model.v[index].shape), index)],
    )

    # The selected shape is [N, C], with no batch axis left. [N] therefore
    # cannot use the model's batch fallback and must be made spatially explicit.
    with pytest.raises(ValueError, match=r"sample shape \(3,\).*selected model shape"):
        intra([torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)], intra.indices)

    explicit_cells = torch.tensor([[1.0], [2.0], [3.0]], dtype=DTYPE)
    actual = intra([explicit_cells], intra.indices)
    expected = torch.zeros_like(model.v)
    expected[0] = explicit_cells
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("model_shape", "index", "sample", "explicit"),
    [
        (
            (3, 1, 1),
            (None, slice(None), slice(None), slice(None)),
            torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE),
            torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE).reshape(1, 3, 1, 1),
        ),
        (
            (2, 3, 4),
            (torch.tensor([0, 1]), slice(None), torch.tensor([0, 1])),
            torch.tensor([1.0, 2.0], dtype=DTYPE),
            torch.tensor([1.0, 2.0], dtype=DTYPE).reshape(2, 1),
        ),
    ],
    ids=["leading-none", "reordered-advanced"],
)
def test_batch_fallback_does_not_guess_reordered_selected_axes(
    model_shape, index, sample, explicit
):
    model = _FakeModel(model_shape)
    intra = Intra(
        model,
        [(dn.constant(value=1.0), tuple(model.v[index].shape), index)],
    )

    with pytest.raises(ValueError, match=r"sample shape .*selected model shape"):
        intra([sample], intra.indices)

    # Users can still state the selected-axis intent explicitly using ordinary
    # trailing broadcasting, without asking Dendra to infer reordered axes.
    actual = intra([explicit], intra.indices)
    expected = torch.zeros_like(model.v)
    expected.index_put_(intra.indices[0], explicit.expand(model.v[index].shape))
    torch.testing.assert_close(actual, expected)


def test_batched_builtin_waveform_preserves_autograd_through_intra():
    model = _FakeModel((BATCH_SIZE, 1, 1))
    index = (slice(None), slice(None), slice(None))
    waveform = dn.constant(value=torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE))
    waveform.value.requires_grad_(True)
    intra = Intra(model, [(waveform, tuple(model.v[index].shape), index)])

    values, indices = intra.init(torch.tensor([0.0], dtype=DTYPE))
    assert values[0].shape == (BATCH_SIZE, 1)
    current = intra([values[0][..., 0]], indices)

    weights = torch.tensor([2.0, 3.0, 5.0], dtype=DTYPE).reshape(BATCH_SIZE, 1, 1)
    (current * weights).sum().backward()
    torch.testing.assert_close(waveform.value.grad, weights.reshape(BATCH_SIZE))


@pytest.mark.parametrize("container", ["population", "network"])
def test_repeated_batch_preserves_an_existing_inner_batch_sweep(container):
    population = dn.Population(N=1, C=1, dtype=DTYPE)
    population.insert(pas, g=0.0, e=-65.0)

    if container == "population":
        population.build()
        population.batch(BATCH_SIZE)
        _target(population).inject(_sweep_waveform(batched=True))
        population.batch(2)
        population.initialize(force_rebuild=True)
        currents = []
        _run_population(population, "run", currents)
    else:
        model = dn.Network({"cell": population})
        model.batch(BATCH_SIZE)
        _target(model.cell).inject(_sweep_waveform(batched=True))
        model.batch(2)
        model.initialize(DT)
        currents = []
        _run_network(model, "run", currents)
        population = model.cell

    expected = _expected_currents(batched=True).unsqueeze(1).expand(-1, 2, -1, -1, -1)
    assert population.shape == (2, BATCH_SIZE, 1, 1)
    assert population.injections[0][1] == population.shape
    assert population.mechanism_injections[0][1] == population.shape
    torch.testing.assert_close(torch.stack(currents), expected)


@pytest.mark.parametrize("runner", ["step", "run", "longrun", "checkpointed"])
@pytest.mark.parametrize("batch_order", ["unbatched", "before", "after"])
def test_population_execution_lanes_preserve_intracellular_sweep_axes(
    runner, batch_order
):
    population = _population(batch_order)
    currents = []
    _run_population(population, runner, currents)

    expected_shape = population.shape
    assert population.injections[0][1] == expected_shape
    assert population.mechanism_injections[0][1] == expected_shape
    torch.testing.assert_close(
        torch.stack(currents),
        _expected_currents(batched=batch_order != "unbatched"),
    )


@pytest.mark.parametrize("runner", ["step", "run", "longrun", "checkpointed"])
@pytest.mark.parametrize("batch_order", ["unbatched", "before", "after"])
def test_network_execution_lanes_preserve_intracellular_sweep_axes(runner, batch_order):
    network = _network(batch_order)
    currents = []
    _run_network(network, runner, currents)

    expected_shape = network.cell.shape
    assert network.cell.injections[0][1] == expected_shape
    assert network.cell.mechanism_injections[0][1] == expected_shape
    torch.testing.assert_close(
        torch.stack(currents),
        _expected_currents(batched=batch_order != "unbatched"),
    )


@pytest.mark.parametrize("runner", ["longrun", "checkpointed"])
def test_chunked_population_lazily_builds_injections_added_after_initialize(runner):
    population = dn.Population(N=1, C=1, dtype=DTYPE)
    population.insert(pas, g=0.0, e=-65.0)
    population.initialize()
    _target(population).inject(_sweep_waveform(batched=False))

    currents = []
    _run_population(population, runner, currents)
    torch.testing.assert_close(torch.stack(currents), _expected_currents(False))
