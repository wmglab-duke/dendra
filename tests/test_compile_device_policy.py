import torch

import dendra as dn
from dendra.helpers import compile_options_for_device
from dendra.models.integrators.core import Integrator


class _CompileProbeModel:
    def __init__(self, device="cpu"):
        self._device = torch.device(device)
        self.jit = True
        self.jit_network_solves = False
        self.backend = "inductor"
        self.fullgraph = True
        self.dynamic = False
        self.compile_mode = "default"
        self.compile_options = None

    def device(self):
        return self._device


class _CompileProbeIntegrator(Integrator):
    def initialize(self, model, dt):
        return None

    def step(self, model, dt, ve=None, intra=None):
        return None

    def _probe(self, value):
        return value + 1


class _PythonWrapperMechanism(torch.nn.Module):
    requires_inductor_python_wrapper = True


def test_compile_options_are_device_local_and_do_not_mutate_input():
    requested = {"epilogue_fusion": False, "cpp_wrapper": True}

    effective = compile_options_for_device(
        requested,
        backend="inductor",
        device="mps:0",
        mode="default",
    )

    assert effective == {
        "epilogue_fusion": False,
        "cpp_wrapper": False,
        "max_fusion_unique_io_buffers": 30,
    }
    assert requested == {"epilogue_fusion": False, "cpp_wrapper": True}
    assert compile_options_for_device(
        None, backend="inductor", device="cpu", mode="default"
    ) == {"cpp_wrapper": True}
    assert compile_options_for_device(
        {"cpp_wrapper": False},
        backend="inductor",
        device="cuda:0",
        mode="default",
    ) == {"cpp_wrapper": False}
    assert (
        compile_options_for_device(None, backend="eager", device="mps", mode="default")
        is None
    )


def test_inductor_mode_is_folded_into_effective_options():
    effective = compile_options_for_device(
        {"max_autotune": False},
        backend="inductor",
        device="mps",
        mode="max-autotune",
    )

    assert effective["coordinate_descent_tuning"] is True
    assert effective["triton.cudagraphs"] is True
    assert effective["max_autotune"] is False
    assert effective["cpp_wrapper"] is False
    assert effective["max_fusion_unique_io_buffers"] == 30


def test_mps_fusion_cap_preserves_a_stricter_user_limit():
    stricter = compile_options_for_device(
        {"max_fusion_unique_io_buffers": 12},
        backend="inductor",
        device="mps",
        mode="default",
    )
    unsafe = compile_options_for_device(
        {"max_fusion_unique_io_buffers": 64},
        backend="inductor",
        device="mps",
        mode="default",
    )

    assert stricter["max_fusion_unique_io_buffers"] == 12
    assert unsafe["max_fusion_unique_io_buffers"] == 30


def test_integrator_uses_effective_mps_options_without_mode(monkeypatch):
    model = _CompileProbeModel("mps")
    integrator = _CompileProbeIntegrator(model, torch.nn.Module())
    compile_calls = []

    def fake_compile(function, **kwargs):
        compile_calls.append(kwargs)
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    result = integrator._call_kernel("_probe", torch.tensor(1.0))

    assert result.item() == 2.0
    assert compile_calls == [
        {
            "backend": "inductor",
            "fullgraph": True,
            "dynamic": False,
            "options": {
                "cpp_wrapper": False,
                "max_fusion_unique_io_buffers": 30,
            },
        }
    ]
    assert integrator.requested_compile_mode == "default"
    assert integrator.compile_mode is None
    assert integrator.compile_device_type == "mps"


def test_integrator_invalidates_cache_when_execution_device_changes():
    model = _CompileProbeModel("cpu")
    model.compile_options = {"cpp_wrapper": False}
    integrator = _CompileProbeIntegrator(model, torch.nn.Module())
    integrator._compiled_kernels["sentinel"] = object()

    # Effective options are identical on these devices, so the device type must
    # independently participate in the compiled-kernel configuration key.
    model._device = torch.device("mps")
    integrator.configure_jit(model)

    assert not integrator._compiled_kernels
    assert integrator.compile_device_type == "mps"
    assert integrator.compile_options == {
        "cpp_wrapper": False,
        "max_fusion_unique_io_buffers": 30,
    }


def test_complex_handler_forces_python_wrapper_on_cpu():
    model = _CompileProbeModel("cpu")
    model.compile_options = {"cpp_wrapper": True}
    mechanism = _PythonWrapperMechanism()

    integrator = _CompileProbeIntegrator(model, mechanism)

    assert integrator.compile_options == {"cpp_wrapper": False}
    assert "mode" not in integrator._compile_kwargs()
    assert integrator._compile_kwargs()["options"]["cpp_wrapper"] is False


def test_population_make_intra_uses_same_mps_compile_policy(monkeypatch):
    population = dn.Population(N=1, C=1, dtype=torch.float32)
    monkeypatch.setattr(population, "device", lambda: torch.device("mps"))
    compile_calls = []

    def fake_compile(function, **kwargs):
        compile_calls.append(kwargs)
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    with dn.ctx(JIT=1, BACKEND="inductor", COMPILE_MODE="default"):
        population._refresh_compile_config_from_ctx()

    assert compile_calls == [
        {
            "backend": "inductor",
            "fullgraph": False,
            "dynamic": False,
            "options": {
                "cpp_wrapper": False,
                "max_fusion_unique_io_buffers": 30,
            },
        }
    ]
