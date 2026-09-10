"""Deterministic CPU workloads for the functional training benchmark.

``size`` is the compartment count (the node count for Sweeney1987), and
``batch`` is the number of independent population members.  The HH axon needs
an odd compartment count because Unmyelinated centers its discretization.
Native solver availability is checked explicitly so a missing extension never
silently changes the measured solver to a Python implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib.util import find_spec

import networkx as nx
import torch

import dendra as dn
from dendra.models.integrators.implicit import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mod import hh

CASE_NAMES = ("hh_axon", "hh_tree", "hh_extcell", "sweeney1987")
DT = 0.01
HH_GNABAR = "integrator.mech.mechanisms.hh.gnabar_param"


class CaseUnavailable(RuntimeError):
    """A requested workload lacks an optional dependency."""


@dataclass(frozen=True)
class BenchCase:
    name: str
    model: dn.Population
    parameter_name: str
    geometry_name: str
    solver: str


def _branched_graph(size: int) -> nx.DiGraph:
    """Make a true binary branch with exact positive compartment geometry."""
    graph = nx.DiGraph()
    for node in range(size):
        length = 18.0 + 2.0 * (node % 4)
        diameter = 8.0 if node == 0 else 1.5 + 0.1 * (node % 5)
        graph.add_node(
            node,
            name=f"compartment.{node}",
            kind="compartment",
            L=length,
            diam=diameter,
            Ra=100.0,
            cm=1.0,
            area=math.pi * diameter * length,
        )
        if node:
            graph.add_edge(
                (node - 1) // 2,
                node,
                R_ohm=1.0e8 * (1.0 + 0.05 * (node % 5)),
                L=length,
            )
    return graph


def _diameters(batch: int, low: float, high: float, dtype: torch.dtype):
    return torch.linspace(low, high, batch, dtype=dtype, device="cpu")


def build_case(
    name: str,
    *,
    dtype: torch.dtype = torch.float64,
    batch: int = 2,
    size: int = 17,
) -> BenchCase:
    """Build one initialized, train-mode population and its gradient targets.

    Geometry names refer to ``PopulationTensors.constants`` after lowering;
    parameter names refer to ``PopulationTensors.parameters``.  Initialization
    is deliberately outside the benchmark's differentiated transition, so the
    benchmark measures an explicit initial-state optimization problem.
    """
    if name not in CASE_NAMES:
        raise ValueError(f"Unknown case {name!r}; choose from {CASE_NAMES}.")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Functional training benchmarks support float32/float64.")
    if batch < 1 or size < 3:
        raise ValueError("batch must be positive and size must be at least 3.")
    if name == "hh_axon" and size % 2 == 0:
        raise ValueError("hh_axon requires an odd size for centered compartments.")
    if not DENDRA_SOLVERS_AVAILABLE:
        raise CaseUnavailable(f"{name} requires the native dendra-solvers package.")
    if name == "sweeney1987" and find_spec("dendra_models") is None:
        raise CaseUnavailable(
            "sweeney1987 requires the optional dendra-models package."
        )

    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=dtype):
        voltage = torch.linspace(-65.0, -62.0, size, dtype=dtype, device="cpu")
        if name == "hh_axon":
            model = dn.Unmyelinated(
                _diameters(batch, 2.0, 2.5, dtype),
                L=(size - 1) * 10.0,
                dx=10.0,
                celsius=6.3,
                v_init=voltage,
                dtype=dtype,
                device="cpu",
                integrator=dn.bwd_euler_ub(method="thomas", imem=False),
            )
            geometry_name = "diam"
            solver = "native Thomas"
        elif name == "hh_tree":
            model = dn.Tree.from_graph(
                _branched_graph(size),
                N=batch,
                celsius=6.3,
                v_init=voltage,
                dtype=dtype,
                device="cpu",
                integrator=dn.dhs(threads=1, imem=False),
            )
            geometry_name = "canonical_area_cm2"
            solver = "native DHS (one solver thread)"
        elif name == "hh_extcell":
            model = dn.ExtCellAxon(
                diameters=_diameters(batch, 5.0, 8.0, dtype),
                n_comp=size,
                celsius=6.3,
                v_init=voltage,
                dtype=dtype,
                device="cpu",
                integrator=dn.bwd_euler_bt(method="thomas", imem=False),
            )
            # Finite coefficients exercise genuine coupled shell dynamics.
            # The constructor defaults instead describe a limiting regime.
            for key, start, stop in (
                ("xraxial", 1.5, 3.0),
                ("xc", 0.08, 0.20),
                ("xg", 1.0e-4, 4.0e-4),
            ):
                target = getattr(model, key)
                target.copy_(
                    torch.linspace(
                        start, stop, target.numel(), dtype=dtype, device="cpu"
                    ).reshape_as(target)
                )
            geometry_name = "diam"
            solver = "native block Thomas (two extracellular layers)"
        else:
            from dendra_models.models.cells.peripheral import Sweeney1987

            model = Sweeney1987(
                diameters=_diameters(batch, 8.0, 10.0, dtype),
                n_node=size,
                v_init=-80.0,
                integrator=dn.bwd_euler_ub(method="thomas", imem=False),
            ).to(device="cpu", dtype=dtype)
            geometry_name = "diameters"
            solver = "native Thomas"

        if name != "sweeney1987":
            model.insert(hh)
        model.initialize()
        model.train()

        if name == "hh_extcell":
            # Preserve a nonzero extracellular carry. The first voltage plane
            # is intracellular potential, and v is membrane voltage.
            with torch.no_grad():
                inner = torch.linspace(
                    -0.2, 0.1, model.v.numel(), dtype=dtype, device="cpu"
                ).reshape(model.shape)
                outer = torch.linspace(
                    0.05, -0.15, model.v.numel(), dtype=dtype, device="cpu"
                ).reshape(model.shape)
                model.vc[..., 1].copy_(inner)
                model.vc[..., 2].copy_(outer)
                model.vc[..., 0].copy_(model.v + inner)

    parameter_name = (
        "integrator.mech.mechanisms.sweeney.gnabar_param.rho"
        if name == "sweeney1987"
        else HH_GNABAR
    )
    return BenchCase(name, model, parameter_name, geometry_name, solver)
