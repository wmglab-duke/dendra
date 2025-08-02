from pathlib import Path

import numpy as np
import pytest
from neuron import h

import axonml as ax
from axonml.models.io import apply_d_lambda
from axonml.models.mod import hh
from axonml.units import nA


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


@pytest.mark.parametrize("d_lambda", [0.1, 0.5, 1.0])
def test_against_neuron(d_lambda):
    rec1, rec2 = sim_and_rec_neuron(d_lambda)
    asc_file = str(Path(__file__).parent / "111200A.asc")

    cell = ax.Tree.from_asc(asc_file, d_lambda=d_lambda, celsius=6.3).double()
    cell.insert(hh)
    cell.soma.inject(ax.mono_rect(amp=5 * nA, delay=1.0, pw=1.0))
    r_ind = cell.find("soma", loc=0.5, as_list=True)
    r_ind += cell.find("dend[86]", loc=0.5, as_list=True)

    rec = ax.callbacks.Recorder(states=["v"], node_indices=r_ind)

    cell.initialize()
    cell.run(tstop=10.0, dt=0.025, callbacks=[rec])
    v = rec.numpy("v")

    r1 = v[:, 0, 0]
    r2 = v[:, 0, 1]

    assert np.allclose(r1, rec1[:-1], atol=1e-3), (
        f"Mismatch in soma voltage: {r1} vs {rec1}"
    )
    assert np.allclose(r2, rec2[:-1], atol=1e-3), (
        f"Mismatch in dend[86] voltage: {r2} vs {rec2}"
    )
