from pathlib import Path

import numpy as np
import pytest
import torch

import dendra as dn
from dendra.models.io import apply_d_lambda
from dendra.models.mod import hh
from dendra.units import nA

neuron = pytest.importorskip("neuron")
h = neuron.h


def sim_and_rec_neuron(d_lambda):
    h.load_file("import3d.hoc")

    class Cell:
        def __init__(self, importer):
            importer.instantiate(self)

        def __repr__(self):
            return "NeuronCell"

    reader = h.Import3d_Neurolucida3()
    reader.quiet = 1
    reader.input(str(Path(__file__).parent / "111200A.asc"))
    importer = h.Import3d_GUI(reader, 0)

    cell = Cell(importer)
    apply_d_lambda(cell.all, d_lambda, 100.0)

    for sec in cell.all:
        sec.insert("hh")

    # stim and record
    stim = h.IClamp(cell.soma[0](0.5))
    stim.delay = 1
    stim.dur = 1
    stim.amp = 5
    rec1 = h.Vector()
    rec1.record(cell.soma[0](0.5)._ref_v)

    rec2 = h.Vector()
    rec2.record(cell.dend[86](0.5)._ref_v)

    h.dt = 0.025
    h.finitialize()
    while h.t < 10.0:
        h.fadvance()
    return np.array(rec1), np.array(rec2)


def assert_neuron_close(actual, expected, *, label, atol=2e-2, rtol=1e-5):
    """Assert simulator agreement with diagnostics useful for drift debugging.

    NEURON and Dendra differ in implementation details and may vary slightly
    across NEURON/PyTorch/platform versions.  This test is intended to catch
    meaningful numerical regressions, not fail on a few ulps to sub-microvolt
    solver drift.
    """
    expected = expected[: actual.shape[0]]
    diff = actual - expected
    max_abs = float(np.nanmax(np.abs(diff)))
    rmse = float(np.sqrt(np.nanmean(diff * diff)))
    assert np.allclose(actual, expected, atol=atol, rtol=rtol), (
        f"Mismatch in {label}: max_abs={max_abs:.6g}, rmse={rmse:.6g}, "
        f"actual={actual}, expected={expected}"
    )


@pytest.mark.parametrize("d_lambda", [0.1, 0.5, 1.0])
def test_against_neuron(d_lambda):
    rec1, rec2 = sim_and_rec_neuron(d_lambda)
    asc_file = str(Path(__file__).parent / "111200A.asc")

    with dn.ctx(DTYPE="float64"):
        cell = dn.Tree.from_asc(asc_file, d_lambda=d_lambda, celsius=6.3)
    cell.insert(hh)
    cell.soma.inject(dn.mono_rect(amp=5 * nA, delay=1.0, pw=1.0))
    r_ind = cell.find("soma", loc=0.5, as_list=True)
    r_ind += cell.find("dend[86]", loc=0.5, as_list=True)

    rec = dn.callbacks.Recorder(states=["v"], node_indices=r_ind)

    cell.initialize()
    cell.run(tstop=10.0, dt=0.025, callbacks=[rec])
    v = rec.numpy("v")

    r1 = v[:, 0, 0]
    r2 = v[:, 0, 1]

    assert_neuron_close(r1, rec1, label="soma voltage")
    assert_neuron_close(r2, rec2, label="dend[86] voltage")


@pytest.mark.parametrize("d_lambda", [0.1, 0.5, 1.0])
def test_against_neuron_longrun(d_lambda):
    rec1, rec2 = sim_and_rec_neuron(d_lambda)
    asc_file = str(Path(__file__).parent / "111200A.asc")

    with dn.ctx(DTYPE="float64"):
        cell = dn.Tree.from_asc(asc_file, d_lambda=d_lambda, celsius=6.3)
    cell.insert(hh)
    cell.soma.inject(dn.mono_rect(amp=5 * nA, delay=1.0, pw=1.0))
    r_ind = cell.find("soma", loc=0.5, as_list=True)
    r_ind += cell.find("dend[86]", loc=0.5, as_list=True)

    rec = dn.callbacks.Recorder(states=["v"], node_indices=r_ind)

    cell.initialize()
    cell.longrun(tstop=10.0, dt=0.025, chunklength=100, callbacks=[rec])
    v = rec.numpy("v")

    r1 = v[:, 0, 0]
    r2 = v[:, 0, 1]

    assert_neuron_close(r1, rec1, label="soma voltage")
    assert_neuron_close(r2, rec2, label="dend[86] voltage")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("d_lambda", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("threads", [8, 16, 32])
@pytest.mark.parametrize("N", [1, 2, 4])
def test_against_neuron_cuda(d_lambda, threads, N):
    rec1, rec2 = sim_and_rec_neuron(d_lambda)
    asc_file = str(Path(__file__).parent / "111200A.asc")

    integrator = dn.dhs(threads=threads)

    cell = dn.Tree.from_asc(
        asc_file, N=N, d_lambda=d_lambda, celsius=6.3, integrator=integrator
    )
    cell.insert(hh)
    cell.soma.inject(dn.mono_rect(amp=5 * nA, delay=1.0, pw=1.0))
    r_ind = cell.find("soma", loc=0.5, as_list=True)
    r_ind += cell.find("dend[86]", loc=0.5, as_list=True)

    rec = dn.callbacks.Recorder(states=["v"], node_indices=r_ind)

    cell.cuda().double().initialize()
    cell.run(tstop=10.0, dt=0.025, callbacks=[rec])
    v = rec.numpy("v")

    r1 = v[:, 0, 0]
    r2 = v[:, 0, 1]

    assert_neuron_close(r1, rec1, label="soma voltage")
    assert_neuron_close(r2, rec2, label="dend[86] voltage")
