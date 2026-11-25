"""Automatic Newton forward routing without allocating CUDA tensors."""

import pytest
import torch
from torch._dynamo.backends import registry
from torch._inductor import config, list_mode_options

import dendra as dn
from dendra.func import _scan
from dendra.func._scan import _resolve_derivimplicit_forward
from dendra.models.mechanisms import Mechanism, _derivimplicit_forward

pytestmark = pytest.mark.cpu


def _custom_backend(graph, examples):
    return graph


class _NoCurrent(Mechanism):
    pass


@pytest.mark.parametrize(
    "device,training,compile_options,expected",
    [
        pytest.param(torch.device("cuda:1"), True, {}, "separate", id="cuda-training"),
        pytest.param("cuda", False, {}, "inline", id="eval"),
        pytest.param("cpu", True, {}, "inline", id="cpu"),
        pytest.param("mps", True, {}, "inline", id="mps"),
        pytest.param("cuda", True, {"backend": "eager"}, "inline", id="eager"),
        pytest.param("cuda", True, {"backend": "aot_eager"}, "inline", id="aot-eager"),
        pytest.param(
            "cuda", True, {"backend": _custom_backend}, "inline", id="callable"
        ),
        pytest.param("cuda", True, {"disable": True}, "inline", id="disabled"),
        pytest.param(
            "cuda", True, {"options": {"use_aoti": True}}, "inline", id="aot-inductor"
        ),
    ],
)
def test_auto_selects_separate_only_for_cuda_training_on_ordinary_inductor(
    device, training, compile_options, expected
):
    options = {"backend": "inductor", **compile_options}
    with config.patch({"triton.cudagraphs": False}):
        assert (
            _resolve_derivimplicit_forward(
                "auto", device=device, training=training, compile_options=options
            )
            == expected
        )


@pytest.mark.parametrize(
    "backend,expected", [("inductor", "separate"), ("aot_eager", "inline")]
)
def test_auto_respects_dynamo_default_backend(monkeypatch, backend, expected):
    monkeypatch.setattr(registry, "get_default_backend", lambda: backend)
    with config.patch({"triton.cudagraphs": False}):
        assert (
            _resolve_derivimplicit_forward(
                "auto", device="cuda", training=True, compile_options={}
            )
            == expected
        )


@pytest.mark.parametrize(
    "ambient,compile_options,expected",
    [
        pytest.param(True, {}, "inline", id="ambient"),
        pytest.param(True, {"mode": "default"}, "inline", id="ambient-with-default"),
        pytest.param(
            False, {"mode": "reduce-overhead"}, "inline", id="reduce-overhead"
        ),
        pytest.param(False, {"mode": "max-autotune"}, "inline", id="max-autotune"),
        pytest.param(
            False,
            {"mode": "max-autotune-no-cudagraphs"},
            "separate",
            id="max-autotune-no-cudagraphs",
        ),
        pytest.param(
            False, {"options": {"triton.cudagraphs": True}}, "inline", id="explicit-on"
        ),
        pytest.param(
            True,
            {"options": {"triton.cudagraphs": False}},
            "separate",
            id="explicit-off",
        ),
        pytest.param(
            False,
            {"options": {"triton.cudagraph-trees": True}},
            "separate",
            id="trees-without-cudagraphs",
        ),
    ],
)
def test_auto_uses_effective_cudagraph_configuration(
    ambient, compile_options, expected
):
    with config.patch({"triton.cudagraphs": ambient}):
        assert (
            _resolve_derivimplicit_forward(
                "auto",
                device="cuda",
                training=True,
                compile_options={"backend": "inductor", **compile_options},
            )
            == expected
        )


def test_auto_respects_ambient_cudagraphs_with_no_cudagraphs_mode():
    mode = "max-autotune-no-cudagraphs"
    # Torch's preset may leave this key unset, so ambient settings still apply.
    enabled = list_mode_options(mode).get("triton.cudagraphs", True)
    with config.patch({"triton.cudagraphs": True}):
        assert _resolve_derivimplicit_forward(
            "auto",
            device="cuda",
            training=True,
            compile_options={"backend": "inductor", "mode": mode},
        ) == ("inline" if enabled else "separate")


def test_separate_cuda_primal_disables_graph_capture(monkeypatch):
    graph = torch.fx.symbolic_trace(torch.nn.Identity())
    calls = []

    def compile_primal(forward, **options):
        calls.append(options)
        return forward

    monkeypatch.setattr(_derivimplicit_forward, "compile_owned", compile_primal)
    with config.patch({"triton.cudagraphs": True}):
        _derivimplicit_forward._ForwardProgram(graph, "inductor", torch.device("cuda"))
    (call,) = calls
    assert call["options"]["triton.cudagraphs"] is False


@pytest.mark.parametrize(
    "policy,device,compile_options,expected",
    [
        pytest.param("inline", "cuda", {}, "inline", id="inline-override"),
        pytest.param("separate", "cpu", {}, "separate", id="separate-on-cpu"),
        pytest.param(
            "separate",
            "cuda",
            {"mode": "reduce-overhead"},
            "separate",
            id="explicit-mode",
        ),
        pytest.param(
            "separate", "cuda", {"disable": True}, "inline", id="disabled-override"
        ),
    ],
)
def test_explicit_newton_forward_selection(policy, device, compile_options, expected):
    with config.patch({"triton.cudagraphs": False}):
        assert (
            _resolve_derivimplicit_forward(
                policy,
                device=device,
                training=True,
                compile_options={"backend": "inductor", **compile_options},
            )
            == expected
        )


@pytest.mark.parametrize("inference", [False, True], ids=["training", "inference"])
def test_public_cpu_default_keeps_inline_execution(monkeypatch, inference):
    resolved = []

    def record_policy(policy, **kwargs):
        mode = _resolve_derivimplicit_forward(policy, **kwargs)
        resolved.append((policy, mode))
        return mode

    monkeypatch.setattr(_scan, "_resolve_derivimplicit_forward", record_policy)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float32, IMEM=False):
        model = dn.SingleCompartment(N=1, C=1, dtype=torch.float32).train(not inference)
        model.insert(_NoCurrent)
        model.initialize()
        runner = dn.compile(
            model, dt=0.1, steps=1, inference=inference, backend="eager"
        )
        initial_voltage = runner.initial_state["integrator"]["v"].clone()
        final, _ = runner()

    assert resolved == ([] if inference else [("auto", "inline")])
    assert runner.execution_report().strategy == "compiled_scan"
    torch.testing.assert_close(final["integrator"]["v"], initial_voltage)
