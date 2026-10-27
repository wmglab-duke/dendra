"""Tree graph metadata stages on CPU under an MPS default-device context."""

import networkx as nx
import pytest
import torch

import dendra as dn
from dendra.models.integrators import dhs_bt
from dendra.models.integrators._native import _mps_solver_available
from dendra.models.mod import hh

try:
    import dendra_solvers
except ImportError:
    dendra_solvers = None

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        not _mps_solver_available(dendra_solvers),
        reason="MPS-enabled dendra-solvers is unavailable",
    ),
]


def _graph(*, block=False, chain=False):
    graph = nx.DiGraph()
    for node in range(4):
        fields = dict(
            name="soma.root" if node == 3 else f"dend.child{node}",
            L=8.000000123456789 + node,
            diam=1.200000234567891 + 0.2 * node,
            x=0.100000123456789 + node,
            y=0.200000234567891 * node,
            z=0.300000345678912 * node,
            Ra=80.000000123456789 + 7.0 * node,
            cm=0.750000234567891 + 0.1 * node,
            area=85.000000345678912 + 35.0 * node,
        )
        if block:
            fields.update(xraxial=[1.4, 2.2], xc=[0.06, 0.11], xg=[1.2e-4, 2.1e-4])
        graph.add_node(node, **fields)
    edges = ((3, 2), (2, 1), (1, 0)) if chain else ((3, 0), (3, 2), (2, 1))
    for parent, child in edges:
        graph.add_edge(parent, child, R_ohm=1.7e8, diff_geom_um=0.123456789123456)
    return graph


def _model(kind):
    # Rely on the public DEVICE/DTYPE context instead of constructor overrides.
    if kind == "packed":
        parts = {
            "branched": dn.Tree.from_graph(_graph(), N=2),
            "chain": dn.Tree.from_graph(_graph(chain=True), N=1),
        }
        for part in parts.values():
            part.insert(hh)
        return dn.concat_models(parts, threads=2)
    block = kind == "block"
    cls = dn.ExtCellTree if block else dn.Tree
    model = cls.from_graph(
        _graph(block=block),
        N=2,
        integrator=dhs_bt(threads=2) if block else dn.dhs(threads=2),
    )
    model.insert(hh)
    return model


@pytest.mark.parametrize("kind", ("scalar", "block", "packed"))
def test_mps_default_context_constructs_and_steps_hh_trees(kind):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    traces, models = [], []
    try:
        for device in ("cpu", "mps"):
            with dn.ctx(DEVICE=device, DTYPE=torch.float32, JIT=0, REQUIRE_GRAD=False):
                assert torch.empty(()).device.type == device
                model = _model(kind)
                model.initialize()
                frames = [model.v.detach().cpu().clone()]
                for _ in range(5):
                    model.run(tstop=0.01, dt=0.01)
                    frames.append(model.v.detach().cpu().clone())
                assert model.v.device.type == device
                assert model.v.dtype == torch.float32
                assert torch.isfinite(model.v).all().item()
                assert model.t.item() == pytest.approx(0.05)
                models.append(model)
                traces.append(torch.stack(frames))
    finally:
        torch.set_num_threads(previous)
    torch.testing.assert_close(traces[1], traces[0], rtol=4e-5, atol=4e-5)
    assert not torch.equal(traces[1][0], traces[1][-1])
    if kind != "packed":
        cpu, mps = models
        for name in ("dx", "diam", "x", "y", "z", "volume", "diff_geom_um", "area"):
            torch.testing.assert_close(
                getattr(mps, name).cpu(), getattr(cpu, name), rtol=0, atol=0
            )
        for name in ("diff_parent_index", "diff_edge_parent", "diff_edge_child"):
            value = getattr(mps, name)
            assert value.device.type == "mps" and value.dtype == torch.long
            assert torch.equal(value.cpu(), getattr(cpu, name))
