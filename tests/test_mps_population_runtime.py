"""MPS-specific Population construction and duration execution contracts."""

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.helpers import compile_options_for_device
from dendra.models.core import _time_grid_from_step_count
from dendra.models.mod import pas

MPS_AVAILABLE = bool(
    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
)

pytestmark = pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS is not available")


def _mps_population():
    pop = dn.SingleCompartment(
        N=1,
        C=1,
        v_init=-65.0,
        device="mps",
        dtype=torch.float32,
    )
    pop.insert(pas, g=0.001, e=-70.0)
    pop.initialize()
    return pop


def test_mps_population_keeps_precise_duration_metadata_on_cpu():
    pop = _mps_population()

    assert pop.v.device.type == "mps"
    assert pop.v.dtype == torch.float32
    assert pop._duration_remainder.device.type == "cpu"
    assert pop._duration_remainder.dtype == torch.float64


def test_mps_population_duration_calls_use_supported_time_grid():
    pop = _mps_population()

    for _ in range(10):
        pop.run(tstop=0.01, dt=0.1)

    assert pop.t.item() == pytest.approx(0.1)
    assert pop._duration_remainder.item() == pytest.approx(0.0, abs=1.0e-15)


def test_mps_time_grid_matches_binary64_staging_reference():
    start = torch.tensor(1.234567, device="mps", dtype=torch.float32)
    actual = _time_grid_from_step_count(
        start,
        2000,
        0.005,
        device=torch.device("mps"),
    )
    expected = (
        start.cpu().to(torch.float64) + torch.arange(2000, dtype=torch.float64) * 0.005
    ).to(torch.float32)

    assert actual.device.type == "mps"
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.0, atol=0.0)


def test_cpu_population_can_move_to_mps_before_initialization():
    pop = dn.SingleCompartment(N=1, C=1, v_init=-65.0, dtype=torch.float64)
    pop.insert(pas, g=0.001, e=-70.0)

    pop.to(device="mps", dtype=torch.float32)
    pop.initialize()
    pop.run(tstop=0.1, dt=0.1)

    assert pop.t.item() == pytest.approx(0.1)
    assert pop._duration_remainder.device.type == "cpu"
    assert pop._duration_remainder.dtype == torch.float64


def test_mps_population_file_load_stages_checkpoint_on_cpu(tmp_path):
    source = _mps_population()
    source.run(tstop=0.04, dt=0.1)
    path = tmp_path / "mps_population.pt"
    torch.save(source.state_dict(), path)

    restored = _mps_population()
    restored.load(path)

    assert restored._duration_remainder.device.type == "cpu"
    assert restored._duration_remainder.dtype == torch.float64
    assert restored._duration_remainder.item() == pytest.approx(0.04)
    restored.run(tstop=0.06, dt=0.1)
    assert restored.t.item() == pytest.approx(0.1)


def test_extcell_axon_constructs_directly_on_mps():
    axon = dn.ExtCellAxon(
        diameters=[8.0],
        n_comp=3,
        device="mps",
        dtype=torch.float32,
    )

    assert axon.x.device.type == "mps"
    assert axon.x.dtype == torch.float32
    torch.testing.assert_close(
        axon.x,
        torch.tensor([[-10.0, 0.0, 10.0]], device="mps"),
    )


def test_mps_population_jit_uses_non_cpp_inductor_wrapper():
    # Keep JIT active through run(): Population refreshes compiler policy from
    # the active Dendra context at each public execution boundary.
    with dn.ctx(JIT=1, BACKEND="inductor", COMPILE_MODE="default"):
        pop = dn.SingleCompartment(
            N=1,
            C=1,
            v_init=-65.0,
            device="mps",
            dtype=torch.float32,
        )
        pop.insert(pas, g=0.001, e=-70.0)
        pop.initialize()
        pop.run(tstop=0.1, dt=0.1)

    assert pop.t.item() == pytest.approx(0.1)
    assert torch.isfinite(pop.v).all()
    assert pop.integrator.compile_options["cpp_wrapper"] is False
    assert pop.integrator.compile_options["max_fusion_unique_io_buffers"] == 30


def test_mps_compile_policy_splits_kernels_above_metal_buffer_limit():
    options = compile_options_for_device(
        None,
        backend="inductor",
        device="mps",
        mode="default",
    )

    def add_all(*values):
        return sum(values)

    with torch_compiler_warning_context():
        compiled = torch.compile(
            add_all,
            backend="inductor",
            fullgraph=True,
            options=options,
        )
        values = [torch.full((8,), value, device="mps") for value in range(40)]
        actual = compiled(*values)

    torch.testing.assert_close(actual, torch.full_like(actual, 780.0))
