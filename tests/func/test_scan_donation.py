"""Composed scans retain lifted-parameter gradients without global policy edits.

Tests target Dendra's own scan compiler boundary. Arbitrary user-selected
backends remain outside this compatibility repair's contract.
"""

from copy import deepcopy
from types import FunctionType, SimpleNamespace

import pytest
import torch
from torch._functorch import config as aot_config
from torch._higher_order_ops.scan import scan
from torch._inductor import config
from torch._inductor import graph as graph_module
from torch._inductor.codegen import wrapper as wrapper_module
from torch._inductor.custom_graph_pass import CustomSchedulerPass
from torch._inductor.graph import SubgraphLowering
from torch._inductor.virtualized import V

from dendra._bootstrap import torch_compiler_warning_context
from dendra.func import _scan_compat
from dendra.func import _scan_donation as donation
from dendra.func._scan_compat import install_scan_compatibility
from dendra.func._scan_inductor import _compile_scan
from dendra.func._types import FunctionalizationError


def _update(value, sample, gain):
    private = value.view_as(value).clone()
    private.add_(sample * gain)
    return private.tanh()


def _program(kind, segments):
    def program(initial, samples, gain):
        def body(value, sample):
            updated = _update(value, sample, gain)
            return updated.clone(), updated.view_as(updated).clone()

        if kind == "nested":

            def outer(value, block):
                final, frames = scan(body, value, block)
                return final.clone(), frames.clone()

            final, frames = scan(outer, initial, samples.reshape(segments, 2, -1))
            return final, torch.cat((initial.unsqueeze(0), frames.flatten(0, 1)))

        value, parameter = initial, gain
        parts = [initial.unsqueeze(0)]
        for index in range(segments):
            block = samples[2 * index : 2 * (index + 1)]
            if kind == "carried":

                def carried_body(carry, sample):
                    current, live_gain = carry
                    updated = _update(current, sample, live_gain)
                    return (updated.clone(), live_gain.clone()), updated.clone()

                (value, parameter), frames = scan(
                    carried_body, (value, parameter), block
                )
            else:
                value, frames = scan(body, value, block)
            parts.append(frames)
        return value, torch.cat(parts)

    return program


def _reference(initial, samples, gain):
    value, frames = initial, [initial]
    for sample in samples.unbind():
        value = _update(value, sample, gain)
        frames.append(value)
    return value, torch.stack(frames)


def _loss(result):
    final, frames = result
    return final.square().sum() + frames.square().mean()


def _inputs(steps, offset):
    return (
        torch.linspace(-0.3, 0.2, 8, dtype=torch.float64),
        torch.linspace(-0.2, 0.25, steps * 8, dtype=torch.float64).reshape(steps, 8)
        + offset / 10,
        torch.tensor(0.2 + offset, dtype=torch.float64),
    )


@pytest.mark.slow
@pytest.mark.parametrize("cpp_wrapper", [False, True], ids=["python", "cpp"])
@pytest.mark.parametrize(
    "kind,segments",
    [("shared", 2), ("shared", 3), ("carried", 2), ("nested", 2)],
)
def test_composed_scans_preserve_full_frames_and_all_input_gradients(
    cpp_wrapper, kind, segments
):
    install_scan_compatibility()
    global_policy = (
        config._pre_fusion_custom_pass,
        config.inplace_buffers,
        config.allow_buffer_reuse,
        aot_config.donated_buffer,
    )
    compiled = _compile_scan(
        _program(kind, segments),
        {
            "backend": "inductor",
            "fullgraph": True,
            "dynamic": False,
            "options": {
                "cpp_wrapper": cpp_wrapper,
                "compile_threads": 1,
                "fx_graph_cache": False,
            },
        },
    )
    assert compiled.get_compiler_config()["cpp_wrapper"] is cpp_wrapper
    # New forward/backward graphs must reuse the compiled callable, not its
    # earlier operands. Ordinary backward (retain_graph=False) is the primary
    # check: retaining the graph can independently inhibit donated buffers.
    first_receipt_count = None
    for offset in (0.0, 0.07):
        values = _inputs(2 * segments, offset)
        actual = tuple(value.clone().requires_grad_() for value in values)
        expected = tuple(value.clone().requires_grad_() for value in values)
        saved = tuple(value.detach().clone() for value in actual)
        versions = tuple(value._version for value in actual)
        assert actual[2].ndim == 0
        with torch_compiler_warning_context():
            output = compiled(*actual)
            reference = _reference(*expected)
            gradients = torch.autograd.grad(_loss(output), actual)
            wanted = torch.autograd.grad(_loss(reference), expected)
        assert output[1].shape == (2 * segments + 1, 8)
        for left, right in zip(output, reference, strict=True):
            torch.testing.assert_close(left, right, rtol=2e-11, atol=2e-12)
        for left, right in zip(gradients, wanted, strict=True):
            assert torch.isfinite(left).all() and torch.count_nonzero(right) > 0
            torch.testing.assert_close(left, right, rtol=2e-10, atol=2e-11)
        for current, before, version in zip(actual, saved, versions, strict=True):
            torch.testing.assert_close(current, before, rtol=0, atol=0)
            assert current._version == version
        receipts = compiled._dendra_scan_donation_diagnostics()
        protected = [item for item in receipts if item.get("guard_applied")]
        assert protected and all(
            item["is_backward"]
            and item["is_subgraph"]
            and item["protected_input_count"] == item["input_count"]
            and all(value["is_never_reuse"] for value in item["inputs"])
            for item in protected
        )
        assert not any(
            item.get("guard_applied")
            for item in receipts
            if not item.get("is_subgraph")
        )
        if first_receipt_count is None:
            first_receipt_count = len(receipts)
        else:
            assert len(receipts) == first_receipt_count
    assert (
        config._pre_fusion_custom_pass,
        config.inplace_buffers,
        config.allow_buffer_reuse,
        aot_config.donated_buffer,
    ) == global_policy


@pytest.mark.slow
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_composed_scans_preserve_shared_lifted_gradient():
    """The device-generic guard is qualified on CUDA when a user runs it there."""
    install_scan_compatibility()
    compiled = _compile_scan(
        _program("shared", 3),
        {
            "backend": "inductor",
            "fullgraph": True,
            "dynamic": False,
            "options": {
                "cpp_wrapper": False,
                "compile_threads": 1,
                "fx_graph_cache": False,
            },
        },
    )
    values = tuple(value.to("cuda") for value in _inputs(6, 0.0))
    actual = tuple(value.clone().requires_grad_() for value in values)
    expected = tuple(value.clone().requires_grad_() for value in values)
    with torch_compiler_warning_context():
        output, reference = compiled(*actual), _reference(*expected)
        gradients = torch.autograd.grad(_loss(output), actual)
        wanted = torch.autograd.grad(_loss(reference), expected)
    for left, right in zip((*output, *gradients), (*reference, *wanted), strict=True):
        torch.testing.assert_close(left, right, rtol=2e-9, atol=2e-10)
    receipts = compiled._dendra_scan_donation_diagnostics()
    assert any(item.get("guard_applied") for item in receipts)


def _graph(*, subgraph=True, backward=True, parent=None):
    graph = object.__new__(SubgraphLowering) if subgraph else SimpleNamespace()
    graph.name = "same_graph_name"
    graph.parent = parent
    graph.is_backward = backward
    graph.graph_input_names = ["shared", "carry"]
    graph.graph_inputs_original = {name: object() for name in graph.graph_input_names}
    graph.never_reuse_buffers = {"existing", "shared"}
    graph.bw_donated_idxs = [0]
    return graph


@pytest.mark.parametrize(
    "subgraph,backward", [(True, True), (True, False), (False, True), (False, False)]
)
def test_only_backward_child_inputs_become_non_reusable(subgraph, backward):
    parent = _graph(subgraph=False)
    graph = _graph(subgraph=subgraph, backward=backward, parent=parent)
    graph_inputs = graph.graph_input_names
    input_ir = dict(graph.graph_inputs_original)
    donor_indices = graph.bw_donated_idxs
    parent_before = set(parent.never_reuse_buffers)
    nodes = [object()]
    guard = donation._scan_donation_pass()
    with V.set_graph_handler(graph):
        assert guard(nodes) is nodes
        assert V.graph is graph
    assert graph.graph_input_names is graph_inputs
    assert graph.graph_inputs_original == input_ir
    assert graph.bw_donated_idxs is donor_indices and donor_indices == [0]
    assert parent.never_reuse_buffers == parent_before
    assert graph.never_reuse_buffers == (
        {"existing", "shared", "carry"}
        if subgraph and backward
        else {"existing", "shared"}
    )
    receipt = guard.diagnostics()[0]
    assert receipt["guard_applied"] is (subgraph and backward)
    assert receipt["status"] == (
        "guarded" if subgraph and backward else "not_applicable"
    )
    assert receipt["inputs"][0]["was_never_reuse"] is True
    assert receipt["inputs"][1]["was_never_reuse"] is False
    receipt["inputs"][0]["name"] = "external mutation"
    assert guard.diagnostics()[0]["inputs"][0]["name"] == "shared"


class _PreviousPass(CustomSchedulerPass):
    def __init__(self, identity="prior", result=None):
        self.identity, self.result, self.calls = identity, result, []

    def uuid(self):
        return self.identity

    def __call__(self, nodes):
        self.calls.append(nodes)
        V.graph.graph_input_names.append("added_by_previous")
        return nodes if self.result is None else self.result


def test_cache_identity_and_previous_callback_order_are_preserved():
    replacement = [object()]
    previous = _PreviousPass(result=replacement)
    guard = donation._scan_donation_pass(previous)
    assert isinstance(guard, CustomSchedulerPass) and guard.previous is previous
    assert guard.uuid() == donation._scan_donation_pass(_PreviousPass()).uuid()
    assert (
        guard.uuid() != donation._scan_donation_pass(_PreviousPass("different")).uuid()
    )
    graph, nodes = _graph(), [object()]
    with V.set_graph_handler(graph):
        assert guard(nodes) is replacement
    assert previous.calls == [nodes]
    assert "added_by_previous" in graph.never_reuse_buffers
    assert guard.diagnostics()[0]["input_count"] == 3


@pytest.mark.parametrize("kind", ["raw", "none_uuid", "empty_uuid"])
def test_uncacheable_callbacks_keep_plain_callable_bypass_and_still_guard(kind):
    from torch._functorch._aot_autograd.autograd_cache import (
        AOTAutogradCacheDetails,
        BypassAOTAutogradCache,
    )
    from torch._inductor.codecache import BypassFxGraphCache, CacheabilityValidator

    calls = []

    def previous(nodes):
        calls.append(nodes)
        return nodes

    selected = (
        previous
        if kind == "raw"
        else _PreviousPass(None if kind == "none_uuid" else "")
    )
    guard = donation._scan_donation_pass(selected)
    assert isinstance(guard, donation._UncacheableScanDonationPass)
    assert not isinstance(guard, CustomSchedulerPass) and not hasattr(guard, "uuid")
    nodes, graph = [object()], _graph()
    with V.set_graph_handler(graph):
        assert guard(nodes) is nodes
    assert set(graph.graph_input_names).issubset(graph.never_reuse_buffers)
    assert calls == [nodes] if kind == "raw" else selected.calls == [nodes]
    fx_graph = torch.fx.Graph()
    argument = fx_graph.placeholder("value")
    fx_graph.output(argument)
    module = torch.fx.GraphModule(torch.nn.Module(), fx_graph)
    with config.patch(_pre_fusion_custom_pass=guard):
        with pytest.raises(
            BypassFxGraphCache, match="Unsupported _pre_fusion_custom_pass"
        ):
            CacheabilityValidator(module, require_shape_env=False).validate()
        # Call only AOT's cacheability initializer, not compilation or tracing.
        # It must translate the same FX-validator rejection into AOT bypass.
        details = object.__new__(AOTAutogradCacheDetails)
        with pytest.raises(
            BypassAOTAutogradCache, match="Unsupported _pre_fusion_custom_pass"
        ):
            details._init_fx_graph_hash_details(module, (), {})


def test_guard_restores_compiler_context_when_previous_callback_raises():
    graph = _graph()

    def previous(_nodes):
        raise RuntimeError("user scheduler failure")

    guard = donation._scan_donation_pass(previous)
    with V.set_graph_handler(graph):
        with pytest.raises(RuntimeError, match="user scheduler failure"):
            guard([object()])
        assert V.graph is graph
    assert graph.never_reuse_buffers == {"existing", "shared"}


def test_receipt_limits_do_not_limit_actual_input_protection(monkeypatch):
    monkeypatch.setattr(donation, "_MAX_RECEIPTS", 1)
    monkeypatch.setattr(donation, "_MAX_INPUTS", 1)
    guard = donation._scan_donation_pass()
    for _index in range(3):
        graph = _graph()
        with V.set_graph_handler(graph):
            guard([])
        assert "carry" in graph.never_reuse_buffers
    receipts = guard.diagnostics()
    assert len(receipts) == 2
    assert receipts[0]["inputs_truncated"] and len(receipts[0]["inputs"]) == 1
    assert receipts[0]["input_count"] == receipts[0]["protected_input_count"] == 2
    assert receipts[1] == {"status": "truncated", "receipts_truncated": True}


def test_unsupported_contract_rejected_before_factory_mutates_compiler_state(
    monkeypatch,
):
    before = config._pre_fusion_custom_pass

    def reject_contract():
        raise FunctionalizationError("injected ownership contract failure")

    monkeypatch.setattr(donation, "validate_inductor_scan_contract", reject_contract)
    with pytest.raises(FunctionalizationError, match="ownership contract failure"):
        donation._scan_donation_pass()
    assert config._pre_fusion_custom_pass is before


@pytest.mark.parametrize("version", ["2.14.0", "2.14.2", "2.15.0", "2.99.0"])
def test_donation_uses_the_installed_contract_not_its_version_label(
    monkeypatch, version
):
    before = config._pre_fusion_custom_pass
    monkeypatch.setattr(torch, "__version__", version)
    assert donation._scan_donation_pass() is not None
    assert config._pre_fusion_custom_pass is before


def test_replaced_upstream_method_rejected_after_contract_validation(monkeypatch):
    donation._validate_contract()
    monkeypatch.setattr(SubgraphLowering, "__init__", lambda *_args, **_kwargs: None)
    with pytest.raises(FunctionalizationError, match=r"SubgraphLowering\.__init__"):
        donation._scan_donation_pass()


def test_replaced_upstream_graph_class_is_rejected_without_method_replacement(
    monkeypatch,
):
    donation._validate_contract()
    original = SubgraphLowering.__init__

    class Replacement:
        pass

    monkeypatch.setattr(graph_module, "SubgraphLowering", Replacement)
    assert SubgraphLowering.__init__ is original
    with pytest.raises(FunctionalizationError, match="SubgraphLowering"):
        donation._scan_donation_pass()


def test_failed_reuse_behavior_probe_is_rejected_without_global_patch(
    monkeypatch,
):
    before = config._pre_fusion_custom_pass
    monkeypatch.setattr(_scan_compat, "_INDUCTOR_CONTRACT", None)

    def always_reusable(self, node):
        return True

    replacement = FunctionType(
        always_reusable.__code__, wrapper_module.__dict__, "can_reuse"
    )
    monkeypatch.setattr(wrapper_module.PythonWrapperCodegen, "can_reuse", replacement)
    with pytest.raises(FunctionalizationError, match="does not prevent buffer reuse"):
        donation._scan_donation_pass()
    assert _scan_compat._INDUCTOR_CONTRACT is None
    assert config._pre_fusion_custom_pass is before


def test_reuse_probe_supports_separate_freeable_non_reusable_buffers(monkeypatch):
    global_policy = (
        config._pre_fusion_custom_pass,
        config.inplace_buffers,
        config.allow_buffer_reuse,
        aot_config.donated_buffer,
    )
    monkeypatch.setattr(_scan_compat, "_INDUCTOR_CONTRACT", None)

    def can_reuse(self, node):
        return (
            node.get_name() not in V.graph.never_reuse_buffers
            and node.get_name() not in V.graph.never_reuse_but_free_buffers
        )

    replacement = FunctionType(can_reuse.__code__, wrapper_module.__dict__, "can_reuse")
    monkeypatch.setattr(wrapper_module.PythonWrapperCodegen, "can_reuse", replacement)
    guard = donation._scan_donation_pass()
    assert _scan_compat._INDUCTOR_CONTRACT is not None
    graph = _graph()
    graph.never_reuse_but_free_buffers = set()
    buffer = SimpleNamespace(get_name=lambda: "carry")
    nodes = [object()]
    with V.set_graph_handler(graph):
        assert replacement(None, buffer)
        assert guard(nodes) is nodes
        assert not replacement(None, buffer)
    assert graph.never_reuse_buffers == {"existing", "shared", "carry"}
    assert graph.never_reuse_but_free_buffers == set()
    assert (
        config._pre_fusion_custom_pass,
        config.inplace_buffers,
        config.allow_buffer_reuse,
        aot_config.donated_buffer,
    ) == global_policy


@pytest.mark.parametrize("delegates", (False, True), ids=("direct", "delegated"))
def test_reuse_probe_tracks_only_delegated_freeability_helper(monkeypatch, delegates):
    global_policy = (
        config._pre_fusion_custom_pass,
        config.inplace_buffers,
        config.allow_buffer_reuse,
        aot_config.donated_buffer,
    )
    monkeypatch.setattr(_scan_compat, "_INDUCTOR_CONTRACT", None)

    def can_free(self, node):
        return node.get_name() not in V.graph.never_reuse_buffers

    def direct(self, node):
        return node.get_name() not in V.graph.never_reuse_buffers

    def delegated(self, node):
        return self.can_free(node) and (
            node.get_name() not in V.graph.never_reuse_but_free_buffers
        )

    owner = wrapper_module.PythonWrapperCodegen
    for name, function in (
        ("can_free", can_free),
        ("can_reuse", delegated if delegates else direct),
    ):
        replacement = FunctionType(function.__code__, wrapper_module.__dict__, name)
        monkeypatch.setattr(owner, name, replacement, raising=False)
    guard = donation._scan_donation_pass()
    can_reuse = owner.can_reuse
    monkeypatch.setattr(owner, "can_free", lambda self, node: True)
    assert owner.can_reuse is can_reuse
    graph = _graph()
    graph.never_reuse_but_free_buffers = set()
    protected = set(graph.never_reuse_buffers)
    nodes = [object()]
    if delegates:
        with pytest.raises(
            FunctionalizationError, match=r"PythonWrapperCodegen\.can_free"
        ):
            donation._scan_donation_pass()
        with V.set_graph_handler(graph):
            with pytest.raises(
                FunctionalizationError, match=r"PythonWrapperCodegen\.can_free"
            ):
                guard(nodes)
        assert graph.never_reuse_buffers == protected
    else:
        assert donation._scan_donation_pass() is not None
        with V.set_graph_handler(graph):
            assert guard(nodes) is nodes
        assert graph.never_reuse_buffers == protected | {"carry"}
    assert graph.never_reuse_but_free_buffers == set()
    assert (
        config._pre_fusion_custom_pass,
        config.inplace_buffers,
        config.allow_buffer_reuse,
        aot_config.donated_buffer,
    ) == global_policy


def test_compile_options_are_not_mutated_by_donation_compatibility(monkeypatch):
    captured = {}

    def compiled(*_args):
        return None

    def compile_stub(function, **options):
        captured.update(function=function, options=options)
        return compiled

    monkeypatch.setattr(torch, "compile", compile_stub)
    supplied = {
        "backend": "inductor",
        "fullgraph": True,
        "dynamic": False,
        "options": {"cpp_wrapper": False, "compile_threads": 1},
    }
    before = deepcopy(supplied)
    result = _compile_scan(lambda value: value.sin(), supplied)
    assert result is compiled and supplied == before
    assert captured["options"]["backend"].config["cpp_wrapper"] is False


@pytest.mark.parametrize(
    "kind",
    ["missing_map", "null_map", "not_mapping", "null_get", "missing_name", "bad_name"],
)
def test_malformed_graph_protocol_fails_before_protection_or_receipts(kind):
    graph = _graph()
    if kind == "missing_map":
        del graph.graph_inputs_original
    elif kind == "null_map":
        graph.graph_inputs_original = None
    elif kind == "not_mapping":
        graph.graph_inputs_original = object()
    elif kind == "null_get":
        graph.graph_inputs_original = SimpleNamespace(get=None)
    elif kind == "missing_name":
        del graph.name
    else:
        graph.name = 3
    before = set(graph.never_reuse_buffers)
    guard = donation._scan_donation_pass()
    with (
        V.set_graph_handler(graph),
        pytest.raises(FunctionalizationError, match="input-protection protocol"),
    ):
        guard([])
    assert graph.never_reuse_buffers == before
    assert guard.diagnostics() == ()


@pytest.mark.parametrize("name", [None, "", "child"])
def test_compatible_graph_name_and_optional_parent_diagnostics(name):
    graph = _graph(parent=SimpleNamespace())
    graph.name = name
    guard = donation._scan_donation_pass()
    with V.set_graph_handler(graph):
        guard([])
    assert graph.never_reuse_buffers == {"existing", "shared", "carry"}
    diagnostic = guard.diagnostics()[0]
    assert diagnostic["graph_name"] == name
    assert diagnostic["parent_name"] is None
