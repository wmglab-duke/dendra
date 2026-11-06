"""Automatic CPU scan forward wrapper policy without compiling numerical graphs."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from dendra.func import _scan, _scan_tree
from dendra.func._scan_inductor import _compile_scan, _ScanInductorBackend

pytestmark = pytest.mark.cpu


def _capture(monkeypatch):
    captured = {}

    def compiled():
        pass

    def compile_stub(function, **options):
        captured.update(function=function, options=options)
        return compiled

    monkeypatch.setattr(torch, "compile", compile_stub)
    return captured, compiled


@pytest.mark.parametrize("prefer", [False, True])
@pytest.mark.parametrize("explicit", [None, False, True])
@pytest.mark.parametrize("spelling", ["cpp_wrapper", "cpp-wrapper"])
def test_private_cpp_preference_respects_normalized_explicit_options(
    monkeypatch, prefer, explicit, spelling
):
    captured, compiled = _capture(monkeypatch)
    options = {"backend": "inductor", "options": {"compile-threads": 1}}
    if explicit is not None:
        options["options"][spelling] = explicit
    original = deepcopy(options)
    result = _compile_scan(lambda: None, options, prefer_cpp_wrapper=prefer)
    backend = captured["options"]["backend"]
    assert isinstance(backend, _ScanInductorBackend)
    assert backend.config["compile_threads"] == 1
    if explicit is None and not prefer:
        assert "cpp_wrapper" not in backend.config
    else:
        assert backend.config["cpp_wrapper"] is (
            prefer if explicit is None else explicit
        )
    assert result is compiled
    assert options == original


def test_private_cpp_preference_preserves_mode_and_outer_compile_options(monkeypatch):
    captured, compiled = _capture(monkeypatch)
    options = {
        "backend": "inductor",
        "mode": "max-autotune-no-cudagraphs",
        "dynamic": True,
        "name": "scan_native_loop",
        "fullgraph": True,
    }
    original = deepcopy(options)
    result = _compile_scan(lambda: None, options, prefer_cpp_wrapper=True)
    backend = captured["options"]["backend"]
    assert backend.config["cpp_wrapper"] is True
    assert backend.config["max_autotune"] is True
    assert backend.dynamic is True and backend.name == "scan_native_loop"
    assert captured["options"]["dynamic"] is True
    assert captured["options"]["fullgraph"] is True
    assert (
        result is compiled and result.get_compiler_config == backend.get_compiler_config
    )
    assert options == original


@pytest.mark.parametrize("setting", ["disable_cpp_codegen", "fx_wrapper"])
@pytest.mark.parametrize("source", ["options", "ambient"])
def test_private_cpp_preference_preserves_other_codegen_choices(
    monkeypatch, setting, source
):
    from torch._inductor import config

    captured, _ = _capture(monkeypatch)
    options = {"backend": "inductor"}
    if source == "options":
        options["options"] = {setting.replace("_", "-"): True}
    previous = getattr(config, setting)
    # These virtual attributes need Inductor's restoration protocol; generic
    # setattr undo can delete the entry instead of restoring its value.
    with config.patch({setting: True} if source == "ambient" else {}):
        _compile_scan(lambda: None, options, prefer_cpp_wrapper=True)
        assert "cpp_wrapper" not in captured["options"]["backend"].config
    assert getattr(config, setting) is previous


@pytest.mark.parametrize(
    "options",
    [
        {"backend": "eager", "options": {"custom_option": 7}},
        {"backend": "aot_eager"},
        {"backend": lambda graph, examples: graph},
        {"backend": "inductor", "disable": True},
        {"backend": "inductor", "options": {"use_aoti": True}},
    ],
)
def test_private_cpp_preference_does_not_change_other_routes(monkeypatch, options):
    captured, _ = _capture(monkeypatch)
    original = deepcopy(options)
    _compile_scan(lambda: None, options, prefer_cpp_wrapper=True)
    assert captured["options"] == original
    assert options == original


@pytest.mark.parametrize("backend", ["inductor", "aot_eager"])
def test_private_cpp_preference_respects_configured_default_backend(
    monkeypatch, backend
):
    from torch._dynamo.backends import registry

    captured, _ = _capture(monkeypatch)
    monkeypatch.setattr(registry, "get_default_backend", lambda: backend)
    _compile_scan(lambda: None, {"dynamic": False}, prefer_cpp_wrapper=True)
    selected = captured["options"].get("backend")
    if backend == "inductor":
        assert selected.config["cpp_wrapper"] is True
    else:
        assert selected is None and captured["options"] == {"dynamic": False}


def test_private_cpp_preference_preserves_mode_options_conflict():
    with pytest.raises(RuntimeError, match="Either mode or options"):
        _compile_scan(
            lambda: None,
            {"backend": "inductor", "mode": "default", "options": {}},
            prefer_cpp_wrapper=True,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_population_inference_capture_selects_cpp_from_immutable_device(
    monkeypatch, device
):
    captured, marker = {}, object()
    functional = SimpleNamespace(
        device=torch.device(device),
        intra=SimpleNamespace(enabled=False),
        extra=SimpleNamespace(enabled=False),
        _clone_state_tree=lambda state: state,
    )
    chunk = object.__new__(_scan._CompiledScanPopulationChunk)
    chunk._functional, chunk._steps, chunk._options = functional, 1, {}
    chunk._plans, chunk._stepwise = None, False

    # Retain the authored callable without executing it or allocating on the
    # plan's device; these tests inspect compiler selection, not numerical work.
    def make_fx_stub(function, *, pre_dispatch):
        captured["pre_dispatch"] = pre_dispatch
        return lambda *examples: function

    monkeypatch.setattr(_scan, "make_fx", make_fx_stub)

    def compile_stub(function, options, *, prefer_cpp_wrapper=False):
        captured["prefer"] = prefer_cpp_wrapper
        return marker

    monkeypatch.setattr(_scan, "_compile_scan", compile_stub)
    with torch.no_grad():
        result = chunk._capture(
            {},
            {},
            {},
            torch.zeros(1),
            None,
            {},
            None,
        )
    assert result.__wrapped__ is marker
    assert captured["prefer"] is (device == "cpu")
    assert captured["pre_dispatch"] is False


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
@pytest.mark.parametrize("original_grad_enabled", [False, True])
def test_shared_scan_capture_selects_cpp_from_original_mode_and_immutable_device(
    monkeypatch, device, original_grad_enabled
):
    captured, marker = {}, object()

    def body(parameters, prepared, state, carries, sample):
        return (state, carries), {}

    def make_fx_stub(function, *, pre_dispatch):
        captured["pre_dispatch"] = pre_dispatch
        return lambda *examples: function

    def compile_stub(function, options, *, prefer_cpp_wrapper=False):
        captured["prefer"] = prefer_cpp_wrapper
        return marker

    monkeypatch.setattr(_scan_tree, "make_fx", make_fx_stub)
    monkeypatch.setattr(_scan_tree, "_compile_scan", compile_stub)
    engine = _scan_tree._CompiledTensorTreeScan(
        body, steps=1, compile_options={}, device=device
    )
    # CPU examples keep this compiler-policy probe independent of accelerator
    # availability. Only the immutable device hint chooses the wrapper policy.
    samples, sample_layout = engine._samples({"drive": torch.zeros(1)})
    with torch.no_grad():
        compiled, _layout, _isolate_views = engine._capture(
            {"scale": torch.tensor(0.7, requires_grad=True)},
            {},
            {"v": torch.zeros(1)},
            {},
            samples,
            sample_layout,
            grad_enabled=original_grad_enabled,
        )
    assert compiled is marker
    assert captured["prefer"] is (device == "cpu" and not original_grad_enabled)
    assert captured["pre_dispatch"] is original_grad_enabled


@pytest.mark.parametrize("device,forward_block", [("cpu", 4), ("cuda", 1), ("mps", 1)])
def test_population_scan_routes_training_and_forward_to_separate_engines(
    monkeypatch, device, forward_block
):
    chunk = _scan._CompiledScanPopulationChunk(
        SimpleNamespace(device=torch.device(device)), steps=8, compile_options={}
    )
    records = []

    def capture_stub(*operands, **options):
        records.append(("inference", torch.is_grad_enabled(), options["block"]))
        return lambda *operands: ({}, {})

    def training_stub(*operands):
        records.append(("training", torch.is_grad_enabled()))
        return {}, {}

    monkeypatch.setattr(chunk, "_capture", capture_stub)
    monkeypatch.setattr(chunk, "_training_values", training_stub)
    monkeypatch.setattr(_scan, "_FORWARD_SCAN_BLOCK", 4)
    with torch.no_grad():
        chunk({}, {}, {}, None, None)
    with torch.enable_grad():
        chunk({}, {}, {}, None, None)
    with torch.no_grad():
        chunk({}, {}, {}, None, None)
    assert records == [("inference", False, forward_block), ("training", True)]
    assert len(chunk._kernels) == 1
