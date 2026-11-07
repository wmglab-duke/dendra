"""Multiple current injections retain independent sources and ordered sums."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.core import make_intra
from dendra.models.stim.intra import Intra, _expand_stimulus_to_selection

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64):
        yield


def _model(specs, *, dtype=torch.float64, batch=None):
    model = dn.Population(N=2, C=6, dtype=dtype)
    if batch is not None:
        model.batch(batch)
    for target, value in specs:
        waveform = dn.constant(value=torch.as_tensor(value, dtype=dtype))
        model[target].inject(waveform)
    return model


def _native(model):
    return Intra(model, model.injections)


def _ordered_reference(intra, values, indices=None):
    """The original per-injection index_put contract, including rounding."""
    result = torch.zeros(intra.shape, dtype=intra.dtype, device=intra.device)
    for sample, index, batch_shape in zip(
        values,
        intra.indices if indices is None else indices,
        intra.selected_batch_shapes,
    ):
        selection = torch.broadcast_shapes(*(item.shape for item in index))
        sample = _expand_stimulus_to_selection(sample, selection, batch_shape)
        result.index_put_(index, sample, accumulate=True)
    return result


def _native_tape(intra, times):
    values, indices = intra.init(times)
    return torch.stack(
        [
            intra([value[..., step] for value in values], indices)
            for step in range(len(times))
        ]
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_multiple_injections_preserve_cancellation_and_duplicate_order(dtype):
    # Interleave disjoint and overlapping targets. Combining all writes into
    # one unordered reduction would permit losing the final small contribution.
    specs = [
        ((slice(None), 0), 1.0e20),
        ((slice(None), 1), 2.0),
        ((slice(None), 0), -1.0e20),
        ((slice(None), [2, 2]), 0.125),
        ((slice(None), 0), 3.25),
        ((slice(None), 3), -0.5),
        ((slice(None), 2), 0.5),
    ]
    model = _model(specs, dtype=dtype)
    intra = _native(model)
    times = torch.tensor([0.0, 0.125], dtype=dtype)
    values, _ = intra.init(times)
    reference = _ordered_reference(intra, [value[..., 0] for value in values])
    expected = torch.tensor([3.25, 2.0, 0.75, -0.5, 0.0, 0.0], dtype=dtype)
    torch.testing.assert_close(reference, expected.expand_as(reference), rtol=0, atol=0)
    torch.testing.assert_close(
        _native_tape(intra, times), reference.expand(2, -1, -1), rtol=0, atol=0
    )

    plan = dn.func.FunctionalIntra(model)
    tensors = plan.extract()
    actual = plan.assemble(tensors.parameters, tensors.constants, times)
    torch.testing.assert_close(actual, reference.expand(2, -1, -1), rtol=0, atol=0)


@pytest.mark.parametrize("execution", ("eager", "aot_eager", "vmap"))
def test_disjoint_injections_preserve_independent_and_shared_parameter_gradients(
    execution,
):
    model = _model(
        [
            ((slice(None), [0, 2]), 0.25),
            ((slice(None), 1), -0.5),
            ((slice(None), [3, 5]), 0.75),
        ]
    )
    waveforms = [spec[0] for spec in model.injections]
    # Distinct waveform objects can share a raw source Parameter. The other
    # waveform remains independent and must retain its own gradient.
    waveforms[2].value = waveforms[0].value
    plan = dn.func.FunctionalIntra(model)
    tensors = plan.extract()
    assert len(tensors.parameters) == 2
    assert all(
        parameter is waveform.value
        for parameter, waveform in zip(
            tensors.parameters.values(), waveforms[:2], strict=True
        )
    )
    times = torch.tensor([0.0, 0.125, 0.25], dtype=torch.float64)

    def assemble(first, second):
        parameters = dict(zip(tensors.parameters, (first, second), strict=True))
        return plan.assemble_values(parameters, tensors.constants, times)

    first, second = tuple(tensors.parameters.values())
    if execution == "aot_eager":
        with torch_compiler_warning_context():
            actual = torch.compile(assemble, backend="aot_eager", fullgraph=True)(
                first, second
            )
    elif execution == "vmap":
        actual = torch.vmap(assemble)(
            torch.stack((first, 2 * first)), torch.stack((second, 3 * second))
        )[0]
    else:
        actual = assemble(first, second)
    reference = _native_tape(_native(model), times)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    weights = torch.arange(1, 13, dtype=actual.dtype).reshape(1, 2, 6)
    actual_gradients = torch.autograd.grad(
        (actual * weights).sum(), (first, second), retain_graph=True
    )
    reference_gradients = torch.autograd.grad(
        (reference * weights).sum(), (first, second)
    )
    for actual_gradient, reference_gradient in zip(
        actual_gradients, reference_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, reference_gradient, rtol=0, atol=0)
        assert reference_gradient != 0
    assert waveforms[2].value is first
    assert waveforms[1].value is second


def test_native_assembly_observes_mutated_and_replacement_indices():
    model = _model([((slice(None), index), index + 1.0) for index in range(3)])
    intra = _native(model)
    values, indices = intra.init(torch.tensor([0.0], dtype=torch.float64))
    samples = [value[..., 0] for value in values]
    initial = intra(samples, indices)
    frozen = initial.clone()

    # A callback may mutate the public index tensors between accepted steps.
    # A previously disjoint assembly must not reuse a stale disjoint grouping.
    indices[1][-1].fill_(0)
    actual = intra(samples, indices)
    torch.testing.assert_close(
        actual, _ordered_reference(intra, samples), rtol=0, atol=0
    )
    torch.testing.assert_close(actual[:, 0], torch.full((2,), 3.0), rtol=0, atol=0)

    replacement = [tuple(item.clone() for item in index) for index in indices]
    replacement[2][-1].fill_(0)
    replaced = intra(samples, replacement)
    torch.testing.assert_close(
        replaced, _ordered_reference(intra, samples, replacement), rtol=0, atol=0
    )
    torch.testing.assert_close(replaced[:, 0], torch.full((2,), 6.0), rtol=0, atol=0)
    torch.testing.assert_close(initial, frozen, rtol=0, atol=0)
    assert initial.untyped_storage().data_ptr() != actual.untyped_storage().data_ptr()


@pytest.mark.parametrize("n_steps", (0, 3))
def test_multiple_batched_injections_preserve_broadcast_and_empty_selections(n_steps):
    model = _model(
        [
            ((slice(None), slice(None), slice(0, 1)), [0.25, 0.5, 0.75]),
            (
                (slice(None), slice(None), slice(2, 3)),
                [[[[-0.125]]], [[[-0.25]]], [[[-0.5]]]],
            ),
            ((slice(None), slice(None), []), [10.0, 20.0, 30.0, 40.0]),
        ],
        batch=3,
    )
    times = torch.arange(n_steps, dtype=torch.float64) * 0.125
    plan = dn.func.FunctionalIntra(model)
    tensors = plan.extract()
    actual = plan.assemble(tensors.parameters, tensors.constants, times)
    assert actual.shape == (n_steps, 3, 2, 6)
    assert actual.is_contiguous()
    if n_steps:
        reference = _native_tape(_native(model), times)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(
            actual[0, :, :, 0],
            torch.tensor([0.25, 0.5, 0.75])[:, None].expand(3, 2),
            rtol=0,
            atol=0,
        )


def test_native_injection_dtype_error_precedes_later_broadcast_error():
    model = _model([((slice(None), 0), 0.25), ((slice(None), 1), 0.5)])
    intra = _native(model)
    with pytest.raises(RuntimeError, match="source and destination dtypes match"):
        intra([torch.tensor(1.0, dtype=torch.float32), torch.ones(3)], intra.indices)


def test_multiple_poisson_injections_keep_shared_generator_evaluation_order():
    generator = torch.Generator().manual_seed(715)
    model = dn.Population(N=2, C=6, dtype=torch.float64)
    for compartment, amplitude in ((0, 0.25), (2, 0.5), (4, 0.75)):
        waveform = dn.mono_rect(amp=amplitude, pw=0.045).poisson(
            interval=0.055,
            n=5,
            start=-0.035,
            noise=0.7,
            randomize_every_call=True,
            generator=generator,
        )
        model[:, compartment].inject(waveform)
    intra = _native(model)
    before = generator.get_state().clone()
    times = torch.arange(8, dtype=torch.float64) * 0.025
    expected_values = [waveform(times).to(intra.dtype) for waveform in intra.stims]
    expected_rng = generator.get_state().clone()
    generator.set_state(before)
    actual_values, indices = intra.init(times)
    torch.testing.assert_close(generator.get_state(), expected_rng, rtol=0, atol=0)
    for actual, expected in zip(actual_values, expected_values, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for step in range(len(times)):
        samples = [value[..., step] for value in actual_values]
        actual = intra(samples, indices)
        reference = _ordered_reference(intra, samples)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    # Assembly consumes already evaluated values and must not touch the stream.
    torch.testing.assert_close(generator.get_state(), expected_rng, rtol=0, atol=0)


def _scatter_count(graph):
    return sum(
        node.op == "call_method" and node.target in ("index_put", "index_put_")
        for node in graph.graph.nodes
    )


def test_compiled_native_call_prepares_groups_and_refreshes_after_index_mutation():
    model = _model([((slice(None), index), index + 1.0) for index in range(3)])
    intra = _native(model)
    values, indices = intra.init(torch.tensor([0.0], dtype=torch.float64))
    samples = [value[..., 0] for value in values]
    scatter_counts = []

    def backend(graph, _inputs):
        scatter_counts.append(_scatter_count(graph))
        return graph.forward

    model.make_intra = torch.compile(make_intra, backend=backend, fullgraph=True)
    with torch_compiler_warning_context():
        initial = model._call_make_intra(intra, samples, indices)
    torch.testing.assert_close(
        initial, _ordered_reference(intra, samples), rtol=0, atol=0
    )
    assert scatter_counts == [1]
    assert [count for count, _ in intra._prepare_assembly(samples, indices).groups] == [
        3
    ]

    indices[1][-1].fill_(0)
    with torch_compiler_warning_context():
        changed = model._call_make_intra(intra, samples, indices)
    torch.testing.assert_close(
        changed, _ordered_reference(intra, samples), rtol=0, atol=0
    )
    assert scatter_counts[-1] == 2

    replacement = [tuple(item.clone() for item in index) for index in indices]
    replacement[2][-1].fill_(0)
    with torch_compiler_warning_context():
        changed = model._call_make_intra(intra, samples, replacement)
    torch.testing.assert_close(
        changed, _ordered_reference(intra, samples, replacement), rtol=0, atol=0
    )
    assert scatter_counts[-1] == 3


def test_direct_compiled_intra_preserves_live_indices_without_prepared_proof():
    model = _model([((slice(None), index), index + 1.0) for index in range(3)])
    intra = _native(model)
    values, indices = intra.init(torch.tensor([0.0], dtype=torch.float64))
    samples = [value[..., 0] for value in values]
    scatter_counts = []

    def backend(graph, _inputs):
        scatter_counts.append(_scatter_count(graph))
        return graph.forward

    compiled = torch.compile(intra, backend=backend, fullgraph=True)
    with torch_compiler_warning_context():
        initial = compiled(samples, indices)
    torch.testing.assert_close(
        initial, _ordered_reference(intra, samples), rtol=0, atol=0
    )
    assert scatter_counts == [3]
    indices[1][-1].fill_(0)
    with torch_compiler_warning_context():
        changed = compiled(samples, indices)
    torch.testing.assert_close(
        changed, _ordered_reference(intra, samples), rtol=0, atol=0
    )

    replacement = [tuple(item.clone() for item in index) for index in indices]
    replacement[2][-1].fill_(0)
    with torch_compiler_warning_context():
        changed = compiled(samples, replacement)
    torch.testing.assert_close(
        changed, _ordered_reference(intra, samples, replacement), rtol=0, atol=0
    )
    assert all(count == 3 for count in scatter_counts)


def test_warm_native_grouping_preserves_eager_and_compiled_error_contracts():
    model = _model([((slice(None), index), index + 1.0) for index in range(3)])
    intra = _native(model)
    values, indices = intra.init(torch.tensor([0.0], dtype=torch.float64))
    samples = [value[..., 0] for value in values]
    model.make_intra = torch.compile(make_intra, backend="eager", fullgraph=True)
    with torch_compiler_warning_context():
        actual = model._call_make_intra(intra, samples, indices)
    torch.testing.assert_close(
        actual, _ordered_reference(intra, samples), rtol=0, atol=0
    )
    prepared = intra._prepare_assembly(samples, indices)
    assert [count for count, _ in prepared.groups] == [3]

    invalid = [samples[0].to(torch.float32), torch.ones(3), samples[2]]
    # A warm coordinate proof may be reused, but it cannot certify replacement
    # sample contracts. Eager execution retains the original first dtype error.
    assert intra._prepare_assembly(invalid, indices) is prepared
    with pytest.raises(RuntimeError, match="source and destination dtypes match"):
        intra(invalid, prepared)

    # Dynamo's fake index_put does not reject the dtype before tracing the
    # later broadcast. Compare against the original sequential compiled path,
    # rather than requiring its tracing exception to match eager execution.
    legacy = torch.compile(intra, backend="eager", fullgraph=True)
    with torch_compiler_warning_context():
        torch.testing.assert_close(legacy(samples, indices), actual, rtol=0, atol=0)
        with pytest.raises(RuntimeError) as baseline_failure:
            legacy(invalid, indices)
        with pytest.raises(type(baseline_failure.value)) as grouped_failure:
            model._call_make_intra(intra, invalid, indices)
    for message in (
        "Observed exception",
        "raised exception ValueError",
        "Intracellular waveform sample shape",
        "cannot broadcast to selected model shape (2,)",
    ):
        assert message in str(baseline_failure.value)
        assert message in str(grouped_failure.value)


def test_native_grouping_refreshes_after_index_storage_and_shape_changes():
    model = _model([((slice(None), index), index + 1.0) for index in range(3)])
    intra = _native(model)
    values, indices = intra.init(torch.tensor([0.0], dtype=torch.float64))
    samples = [value[..., 0] for value in values]
    initial = intra._prepare_assembly(samples, indices)
    assert [count for count, _ in initial.groups] == [3]

    coordinate = indices[1][-1]
    coordinate.set_(torch.zeros_like(coordinate))
    rebound = intra._prepare_assembly(samples, indices)
    assert rebound is not initial
    assert [count for count, _ in rebound.groups] == [1, 2]
    torch.testing.assert_close(
        intra(samples, indices), _ordered_reference(intra, samples), rtol=0, atol=0
    )

    # A singleton coordinate still broadcasts over both cells. Shape changes
    # must refresh the cached concatenation even when target values are equal.
    coordinate.resize_(1)
    resized = intra._prepare_assembly(samples, indices)
    assert resized is not rebound
    assert tuple(coordinate.shape) == (1,)
    torch.testing.assert_close(
        intra(samples, indices), _ordered_reference(intra, samples), rtol=0, atol=0
    )


def test_native_grouping_keeps_independent_version_counters_for_shared_storage():
    model = _model([((0, 0), 1.0), ((1, 0), 2.0)])
    intra = _native(model)
    values, _ = intra.init(torch.tensor([0.0], dtype=torch.float64))
    samples = [value[..., 0] for value in values]
    first = torch.tensor([0], dtype=torch.long)
    second = torch.empty(0, dtype=torch.long).set_(
        first.untyped_storage(), 0, first.shape, first.stride()
    )
    # set_ ties storage without tying version counters. A storage-address memo
    # would miss the second coordinate's mutation when retaining only first.
    assert first._base is None and second._base is None
    assert first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
    indices = [
        (torch.tensor([0], dtype=torch.long), first),
        (torch.tensor([1], dtype=torch.long), second),
    ]
    model.make_intra = torch.compile(make_intra, backend="eager", fullgraph=True)
    with torch_compiler_warning_context():
        initial = model._call_make_intra(intra, samples, indices)
    prepared = intra._prepare_assembly(samples, indices)
    assert [count for count, _ in prepared.groups] == [2]
    torch.testing.assert_close(
        initial, _ordered_reference(intra, samples, indices), rtol=0, atol=0
    )
    first_version, second_version = first._version, second._version

    second.fill_(2)
    assert first._version == first_version
    assert second._version == second_version + 1
    assert first.item() == second.item() == 2
    refreshed = intra._prepare_assembly(samples, indices)
    assert refreshed is not prepared
    with torch_compiler_warning_context():
        changed = model._call_make_intra(intra, samples, indices)
    torch.testing.assert_close(
        changed, _ordered_reference(intra, samples, indices), rtol=0, atol=0
    )
    torch.testing.assert_close(changed[:, 0], torch.zeros(2), rtol=0, atol=0)
    torch.testing.assert_close(changed[:, 2], torch.tensor([1.0, 2.0]), rtol=0, atol=0)


def test_cpu_inductor_uses_live_indices_across_eager_cache_and_backend_changes():
    model = _model([((slice(None), index), index + 1.0) for index in range(3)])
    intra = _native(model)
    values, indices = intra.init(torch.tensor([0.0], dtype=torch.float64))
    samples = [value[..., 0] for value in values]
    prepared = intra._prepare_assembly(samples, indices)
    assert [count for count, _ in prepared.groups] == [3]
    assert intra._prepare_assembly(samples, indices, backend="inductor") is indices

    indices[1][-1].fill_(0)
    assert intra._prepare_assembly(samples, indices, backend="inductor") is indices
    refreshed = intra._prepare_assembly(samples, indices, backend=None)
    assert refreshed is not prepared
    assert [count for count, _ in refreshed.groups] == [1, 2]
    torch.testing.assert_close(
        intra(samples, refreshed), _ordered_reference(intra, samples), rtol=0, atol=0
    )

    class Backend:
        def __eq__(self, other):
            raise AssertionError("Backend equality is authored execution")

    class BackendName(str):
        def __eq__(self, other):
            raise AssertionError("A backend name subclass can override equality")

    for backend in (Backend(), BackendName("inductor")):
        assert intra._prepare_assembly(samples, indices, backend=backend) is refreshed


def test_native_first_dtype_error_precedes_later_incompatible_coordinate_shapes():
    model = _model([((slice(None), index), index + 1.0) for index in range(2)])
    intra = _native(model)
    values, indices = intra.init(torch.tensor([0.0], dtype=torch.float64))
    samples = [value[..., 0] for value in values]
    assert [count for count, _ in intra._prepare_assembly(samples, indices).groups] == [
        2
    ]
    incompatible = [
        indices[0],
        (
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([0, 1, 2], dtype=torch.long),
        ),
    ]
    invalid = [samples[0].to(torch.float32), samples[1]]
    with pytest.raises(RuntimeError, match="source and destination dtypes match"):
        _ordered_reference(intra, invalid, incompatible)
    with pytest.raises(RuntimeError, match="source and destination dtypes match"):
        intra(invalid, incompatible)


def test_native_plain_coordinate_view_does_not_read_authored_base_version():
    class GuardedVersion(torch.Tensor):
        @property
        def _version(self):
            raise AssertionError("Authored base version property must not be read")

    owner = torch.Tensor._make_subclass(
        GuardedVersion, torch.tensor([0, 0], dtype=torch.long), False
    )
    plain = owner.as_subclass(torch.Tensor)
    assert type(plain) is torch.Tensor
    assert plain._base is owner
    model = _model([((slice(None), 0), 1.0), ((slice(None), 2), 2.0)])
    intra = _native(model)
    values, indices = intra.init(torch.tensor([0.0], dtype=torch.float64))
    samples = [value[..., 0] for value in values]
    indices = [(indices[0][0], plain), indices[1]]
    prepared = intra._prepare_assembly(samples, indices)
    assert [count for count, _ in prepared.groups] == [2]
    assert intra._prepare_assembly(samples, indices) is prepared
    torch.testing.assert_close(
        intra(samples, indices),
        _ordered_reference(intra, samples, indices),
        rtol=0,
        atol=0,
    )

    before = plain._version
    plain.fill_(3)
    assert plain._version == before + 1
    refreshed = intra._prepare_assembly(samples, indices)
    assert refreshed is not prepared
    changed = intra(samples, indices)
    torch.testing.assert_close(
        changed, _ordered_reference(intra, samples, indices), rtol=0, atol=0
    )
    torch.testing.assert_close(changed[:, 0], torch.zeros(2), rtol=0, atol=0)
    torch.testing.assert_close(changed[:, 3], torch.ones(2), rtol=0, atol=0)


def test_native_inference_indices_use_live_fallback_without_mutation_counters():
    model = _model([((slice(None), index), index + 1.0) for index in range(3)])
    intra = _native(model)
    values, _ = intra.init(torch.tensor([0.0], dtype=torch.float64))
    samples = [value[..., 0] for value in values]
    with torch.inference_mode():
        indices = [tuple(item.clone() for item in index) for index in intra.indices]
        assert intra._prepare_assembly(samples, indices) is indices
        initial = intra(samples, indices)
        torch.testing.assert_close(
            initial, _ordered_reference(intra, samples, indices), rtol=0, atol=0
        )
        indices[1][-1].fill_(0)
        changed = intra(samples, indices)
        torch.testing.assert_close(
            changed, _ordered_reference(intra, samples, indices), rtol=0, atol=0
        )


def test_grouped_functional_assembly_preserves_forward_ad():
    model = _model([((slice(None), index), index + 1.0) for index in range(3)])
    plan = dn.func.FunctionalIntra(model)
    tensors = plan.extract()
    assert [count for count, _ in plan._current_groups] == [3]
    times = torch.tensor([0.0, 0.125], dtype=torch.float64)
    values = tuple(tensors.parameters.values())
    tangents = tuple(
        torch.full_like(value, index + 0.25) for index, value in enumerate(values)
    )

    def assemble(*parameters):
        return plan.assemble_values(
            dict(zip(tensors.parameters, parameters, strict=True)),
            tensors.constants,
            times,
        )

    primal, tangent = torch.func.jvp(assemble, values, tangents)
    reference = _native_tape(_native(model), times)
    expected_tangent = torch.zeros_like(reference)
    for index, value in enumerate(tangents):
        expected_tangent[:, :, index] = value
    torch.testing.assert_close(primal, reference, rtol=0, atol=0)
    torch.testing.assert_close(tangent, expected_tangent, rtol=0, atol=0)
