"""Reverse scan preserves chronological outputs and differentiation."""

import pytest
import torch
from torch._higher_order_ops.scan import scan

from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._scan_compat import install_scan_compatibility
from dendra.func._scan_inductor import _compile_scan

pytestmark = pytest.mark.cpu


def _step(carry, tapes):
    signal, fraction, flag, sign = tapes
    previous = torch.where(flag, carry, -carry)
    updated = previous + (signal * fraction) * sign
    return updated.clone(), (updated.square(), torch.logical_not(flag))


@pytest.mark.parametrize("reverse", (False, True), ids=("backward", "manual_reverse"))
def test_reverse_scan_matches_strided_and_broadcast_tapes(reverse):
    install_scan_compatibility()
    steps, batch, width = 5, 2, 3
    dtype = torch.float64
    source = (
        torch.arange((steps + 2) * batch * 2 * width, dtype=dtype).reshape(
            steps + 2, batch, 2 * width
        )
        / 64
    )
    fraction = (
        torch.arange(1, batch * width + 1, dtype=dtype).reshape(1, batch, width) / 8
    )
    seed = torch.arange(batch * width, dtype=dtype).reshape(batch, width) / 16
    flag = torch.tensor(
        [[True, False], [False, True], [True, True], [False, False], [True, False]]
    ).unsqueeze(-1)
    sign = torch.tensor([-1, 2, -3, 4, -5])

    def samples(signal, fraction):
        return (
            signal[1:-1, :, 1::2],
            fraction.expand(steps, -1, -1),
            flag,
            sign,
        )

    def numerical(initial, signal, fraction):
        return scan(_step, initial, samples(signal, fraction), reverse=reverse)

    def reference(initial, signal, fraction):
        inputs = samples(signal, fraction)
        indexes = range(steps - 1, -1, -1) if reverse else range(steps)
        outputs = [None] * steps
        for index in indexes:
            initial, outputs[index] = _step(
                initial, tuple(tape[index] for tape in inputs)
            )
        emissions = tuple(torch.stack(leaves) for leaves in zip(*outputs, strict=True))
        return initial, emissions

    compiled = _compile_scan(
        numerical,
        {
            "backend": "inductor",
            "fullgraph": True,
            "dynamic": False,
            "options": {"compile_threads": 1, "cpp.threads": 1, "cpp_wrapper": False},
        },
    )
    originals = (seed, source, fraction)
    actual = tuple(value.clone().requires_grad_() for value in originals)
    expected = tuple(value.clone().requires_grad_() for value in originals)
    versions = tuple(value._version for value in actual)
    # A nonuniform, strided output cotangent distinguishes reverse indexing
    # from a forward sweep on any accidentally unflipped saved input.
    cotangent = source.sub(0.5)[1:-1, :, ::2]
    terminal = seed * 2 - 0.25
    with torch_compiler_warning_context():
        result = compiled(*actual)
        target = reference(*expected)
        grads = torch.autograd.grad(
            (result[0], result[1][0]), actual, (terminal, cotangent)
        )
        wanted = torch.autograd.grad(
            (target[0], target[1][0]), expected, (terminal, cotangent)
        )

    torch.testing.assert_close(result, target, rtol=0, atol=0)
    torch.testing.assert_close(grads, wanted, rtol=0, atol=0)
    for value, original, version in zip(actual, originals, versions, strict=True):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
        assert value._version == version
        assert not torch._C._is_alias_of(result[0], value)
    assert not torch._C._is_alias_of(result[0], result[1][0])


def test_reverse_scan_keeps_storage_sensitive_reads_and_escaped_flip():
    install_scan_compatibility()

    def numerical(seed, tape):
        reversed_tape = tape.flip(0)

        def step(carry, sample):
            # A fixed backing-storage offset repeatedly reads the first row of
            # the reversed allocation. Reversing only the sample index would
            # change this access if the compiler admitted the storage view.
            first = sample.as_strided((1,), (1,), 0)[0]
            updated = carry + sample + first
            return updated, updated.clone()

        final, emitted = scan(step, seed, reversed_tape)
        return final, emitted, reversed_tape, tape.flip(1)

    source = torch.arange(18.0, dtype=torch.float64).reshape(6, 3)[1:-1]
    seed = source.new_zeros(3)
    compiled = _compile_scan(
        numerical,
        {
            "backend": "inductor",
            "fullgraph": True,
            "dynamic": False,
            "options": {"compile_threads": 1, "cpp.threads": 1, "cpp_wrapper": False},
        },
    )
    reversed_tape = source.flip(0)
    expected = (reversed_tape + reversed_tape[0, 0]).cumsum(0)
    with torch_compiler_warning_context(), torch.no_grad():
        final, emitted, escaped, cross_axis = compiled(seed, source)
    torch.testing.assert_close(
        (final, emitted, escaped, cross_axis),
        (expected[-1], expected, reversed_tape, source.flip(1)),
        rtol=0,
        atol=0,
    )
    assert not torch._C._is_alias_of(escaped, source)


@pytest.mark.parametrize(
    ("observer", "storage"),
    (
        pytest.param(None, "ordinary", id="logical"),
        pytest.param(None, "aliased", id="aliased_tape"),
        pytest.param(None, "lazy_zero", id="lazy_zero_tape"),
        pytest.param(torch.ops.aten.storage_offset.default, "ordinary", id="offset"),
        pytest.param(
            torch.ops.aten.sym_storage_offset.default, "ordinary", id="symbolic_offset"
        ),
        pytest.param(
            torch.ops.aten._has_same_storage_numel.default,
            "ordinary",
            id="backing_size",
        ),
        pytest.param(
            torch.ops.aten.is_set_to.default, "ordinary", id="storage_identity"
        ),
        pytest.param(torch.ops.aten.set.source_Tensor, "ordinary", id="set_alias"),
    ),
)
def test_reverse_scan_eligibility_preserves_storage(observer, storage):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from dendra.func._scan_reverse import _scan_reversal

    aten = torch.ops.aten
    body_graph = torch.fx.Graph()
    carry = body_graph.placeholder("carry")
    sample = body_graph.placeholder("sample")
    sibling = body_graph.placeholder("sibling")
    updated = body_graph.call_function(aten.add.Tensor, (carry, sample))
    updated = body_graph.call_function(aten.add.Tensor, (updated, sibling))
    if observer is None:
        observed = body_graph.call_function(aten.clone.default, (sample,))
    elif observer is aten.set.source_Tensor:
        observed = body_graph.call_function(observer, (sibling, sample))
    else:
        comparison = observer in (
            aten._has_same_storage_numel.default,
            aten.is_set_to.default,
        )
        args = (sample, sibling) if comparison else (sample,)
        metadata = body_graph.call_function(observer, args)
        observed = body_graph.call_function(aten.scalar_tensor.default, (metadata,))
    body_graph.output((updated, observed))
    body = torch.fx.GraphModule({}, body_graph)

    graph = torch.fx.Graph()
    initial = graph.placeholder("initial")
    source = graph.placeholder("tape")
    sibling_tape = graph.placeholder("sibling_tape")
    reversed_tape = graph.call_function(torch.ops.prims.rev.default, (source, [0]))
    combine = graph.get_attr("body")
    scan_node = graph.call_function(
        torch.ops.higher_order.scan,
        (combine, [initial], [reversed_tape, sibling_tape], ()),
    )
    graph.output(scan_node)
    module = torch.fx.GraphModule({"body": body}, graph)
    with FakeTensorMode():
        initial.meta["val"] = torch.empty(2)
        # Prefix view has the required zero offset and matching reversed
        # strides, but the size of its backing allocation is still observable.
        source.meta["val"] = (
            torch._efficientzerotensor((4, 2))
            if storage == "lazy_zero"
            else torch.empty(6, 2)[:4]
        )
        reversed_tape.meta["val"] = source.meta["val"].flip(0)
        sibling_tape.meta["val"] = (
            source.meta["val"].view(4, 2) if storage == "aliased" else torch.empty(4, 2)
        )
        shape = (4, 2) if observer in (None, aten.set.source_Tensor) else (4,)
        scan_node.meta["val"] = (torch.empty(2), torch.empty(shape))
    assert source.meta["val"].storage_offset() == 0
    assert tuple(reversed_tape.users) == (scan_node,)
    assert module.graph is scan_node.graph
    result = _scan_reversal(scan_node)
    assert (result is not None) == (observer is None and storage == "ordinary")
