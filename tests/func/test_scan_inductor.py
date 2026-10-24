"""Scan-local copy lowering and preservation of compiler selection/options."""

from copy import deepcopy

import pytest
import torch

from dendra.func._scan_inductor import (
    _as_strided_copy,
    _compile_scan,
    _ScanInductorBackend,
    _select_copy,
    _slice_copy,
)


@pytest.mark.parametrize("layout", ["contiguous", "transposed", "expanded", "empty"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.int64, torch.bool])
def test_scan_copy_lowering_preserves_semantics(layout, dtype):
    raw = torch.arange(24).reshape(2, 3, 4)
    if layout == "transposed":
        raw = raw.transpose(1, 2)
    elif layout == "expanded":
        raw = raw[:1].expand(3, -1, -1)
    elif layout == "empty":
        raw = raw[:, :, :0]
    value = raw.to(dtype)
    before = value.clone()
    cases = [
        (_select_copy, torch.ops.aten.select_copy.int, (0, -1)),
        (_slice_copy, torch.ops.aten.slice_copy.Tensor, (-1, None, None, 1)),
        (_slice_copy, torch.ops.aten.slice_copy.Tensor, (0, 0, 1, 1)),
        (_slice_copy, torch.ops.aten.slice_copy.Tensor, (1, 0, 3, 2)),
        (_slice_copy, torch.ops.aten.slice_copy.Tensor, (0, 1, 1, 1)),
    ]
    for replacement, original, arguments in cases:
        actual = replacement(value, *arguments)
        expected = original(value, *arguments)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.stride() == expected.stride()
        assert actual.dtype == expected.dtype
        assert actual.device == expected.device
        assert not torch._C._is_alias_of(actual, value)
    torch.testing.assert_close(value, before, rtol=0, atol=0)


def _capture_compile(monkeypatch):
    captured = {}

    def compiled():
        pass

    def compile_stub(function, **options):
        captured.update(function=function, options=options)
        return compiled

    monkeypatch.setattr(torch, "compile", compile_stub)
    return captured, compiled


def test_scan_inductor_preserves_mode_dynamic_name_and_lifecycle(monkeypatch):
    captured, compiled = _capture_compile(monkeypatch)
    options = {
        "backend": "inductor",
        "mode": "max-autotune-no-cudagraphs",
        "dynamic": True,
        "name": "scan_validation",
        "fullgraph": True,
    }
    before = deepcopy(options)
    result = _compile_scan(lambda: None, options)
    backend = captured["options"]["backend"]
    assert isinstance(backend, _ScanInductorBackend)
    assert backend.dynamic is True
    assert backend.name == "scan_validation"
    assert backend.config["max_autotune"] is True
    assert captured["options"]["dynamic"] is True
    assert captured["options"]["fullgraph"] is True
    assert "mode" not in captured["options"]
    assert "options" not in captured["options"]
    assert result is compiled
    assert result.get_compiler_config == backend.get_compiler_config
    assert backend.reset.__func__ is torch._TorchCompileInductorWrapper.reset
    assert options == before


def test_scan_inductor_preserves_guard_options_without_consuming_inputs(monkeypatch):
    captured, _ = _capture_compile(monkeypatch)

    def guard_filter(guards):
        return [True] * len(guards)

    nested = {"compile-threads": 1, "guard_filter_fn": guard_filter, "use_aoti": False}
    options = {"backend": "inductor", "options": nested, "dynamic": False}
    before = dict(nested)
    _compile_scan(lambda: None, options)
    backend = captured["options"]["backend"]
    assert backend.config["compile_threads"] == 1
    assert captured["options"]["options"] == {
        "guard_filter_fn": guard_filter,
        "use_aoti": False,
    }
    assert nested == before and options["options"] is nested


@pytest.mark.parametrize("backend", ["eager", "aot_eager", lambda graph, inputs: graph])
def test_scan_custom_backends_are_unchanged(monkeypatch, backend):
    captured, _ = _capture_compile(monkeypatch)
    options = {"backend": backend, "options": {"custom_option": 17}, "dynamic": True}
    _compile_scan(lambda: None, options)
    assert captured["options"] == options
    assert captured["options"]["options"] is not options["options"]


@pytest.mark.parametrize("default_backend", ["inductor", "aot_eager"])
def test_scan_respects_configured_default_backend(monkeypatch, default_backend):
    from torch._dynamo.backends import registry

    captured, _ = _capture_compile(monkeypatch)
    monkeypatch.setattr(registry, "get_default_backend", lambda: default_backend)
    _compile_scan(lambda: None, {"dynamic": False})
    selected = captured["options"].get("backend")
    if default_backend == "inductor":
        assert isinstance(selected, _ScanInductorBackend)
    else:
        assert selected is None


@pytest.mark.parametrize("extra", [{"disable": True}, {"options": {"use_aoti": True}}])
def test_scan_preserves_disabled_and_aoti_routes(monkeypatch, extra):
    captured, _ = _capture_compile(monkeypatch)
    options = {"backend": "inductor", **extra}
    _compile_scan(lambda: None, options)
    assert captured["options"] == options


def test_scan_preserves_mode_options_conflict():
    with pytest.raises(RuntimeError, match="Either mode or options"):
        _compile_scan(
            lambda: None, {"backend": "inductor", "mode": "default", "options": {}}
        )


def test_scan_inductor_owns_decompositions_and_forwards_overrides(monkeypatch):
    from torch._inductor import compile_fx as compile_module
    from torch._inductor.decomposition import select_decomp_table

    from dendra.func._dufort_autograd import _inductor_decompositions

    shared = select_decomp_table()
    before = dict(shared)
    captured = {}
    marker = object()

    def compile_stub(graph, inputs, **kwargs):
        captured.update(kwargs)
        return marker

    monkeypatch.setattr(compile_module, "compile_fx", compile_stub)
    backend = _ScanInductorBackend(None, {"compile_threads": 1}, False, "scan_test")
    result = backend(object(), [], config_patches={"compile_threads": 2})
    assert result is marker
    assert captured["config_patches"]["compile_threads"] == 2
    assert captured["compile_region_name"] == "scan_test"
    assert backend.config["compile_threads"] == 1
    selected = captured["decompositions"]
    assert selected is not shared
    assert selected[torch.ops.aten.select_copy.int] is _select_copy
    assert selected[torch.ops.aten.slice_copy.Tensor] is _slice_copy
    assert selected[torch.ops.aten.as_strided_copy.default] is _as_strided_copy
    for operator, decomposition in _inductor_decompositions().items():
        assert selected[operator] is decomposition
    assert shared == before
    selected.clear()
    assert shared == before
    ordinary = torch._TorchCompileInductorWrapper(
        None, {"compile_threads": 1}, False, "scan_test"
    )
    assert backend != ordinary and ordinary != backend
    assert backend == _ScanInductorBackend(
        None, {"compile_threads": 1}, False, "scan_test"
    )


def test_scan_inductor_real_compiled_copy_gradients():
    def operators(value):
        selected = torch.ops.aten.select_copy.int(value, 0, 1)
        sliced = torch.ops.aten.slice_copy.Tensor(selected, -1, 1, 4, 2)
        return (sliced.sin() * sliced).sum()

    compiled = _compile_scan(
        operators,
        {"backend": "inductor", "fullgraph": True, "options": {"compile_threads": 1}},
    )
    left = torch.linspace(-0.7, 0.8, 24, dtype=torch.float64).reshape(2, 3, 4)
    left = left.transpose(1, 2).detach().requires_grad_()
    right = left.detach().clone().requires_grad_()
    actual = compiled(left)
    expected = operators(right)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual_gradient = torch.autograd.grad(actual, left)[0]
    expected_gradient = torch.autograd.grad(expected, right)[0]
    torch.testing.assert_close(
        actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.int64, torch.bool])
@pytest.mark.parametrize(
    "size,stride,offset",
    [
        ((3, 2), (1, 3), None),
        ((2, 3), (4, 1), 6),
        ((2, 3), (0, 1), 6),
        ((0, 3), (3, 1), None),
    ],
)
def test_scan_storage_view_copy_preserves_offsets_layouts_and_independent_storage(
    dtype, size, stride, offset
):
    value = torch.arange(32).to(dtype)[4:28].reshape(4, 6)
    before = value.clone()
    actual = _as_strided_copy(value, size, stride, offset)
    expected = torch.ops.aten.as_strided_copy.default(value, size, stride, offset)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.stride() == expected.stride()
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    assert not torch._C._is_alias_of(actual, value)
    torch.testing.assert_close(value, before, rtol=0, atol=0)


def test_scan_storage_view_copy_preserves_repeated_element_gradients():
    raw = torch.linspace(-1, 1, 32, dtype=torch.float64, requires_grad=True)
    value = raw[4:28].reshape(4, 6)
    actual = _as_strided_copy(value, (2, 3), (0, 1), 6)
    expected = torch.ops.aten.as_strided_copy.default(value, (2, 3), (0, 1), 6)
    weights = torch.arange(1, 7, dtype=torch.float64).reshape(2, 3)
    actual_gradient = torch.autograd.grad((actual * weights).sum(), raw)[0]
    expected_gradient = torch.autograd.grad((expected * weights).sum(), raw)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_gradient[6:9], torch.tensor([5.0, 7.0, 9.0], dtype=raw.dtype)
    )
    assert torch.count_nonzero(actual_gradient) == 3
