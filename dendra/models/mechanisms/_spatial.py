"""Reusable one-dimensional spatial linear operators for Dendra.

The initial operators here target Dendra's own unbranched Axon classes, where
all compartments are ordered along the final tensor dimension.  They are kept
ion/material agnostic so voltage, material diffusion, and future cable-like
processes can share the same tridiagonal solve path.
"""

from __future__ import annotations

import warnings
from collections import deque
from functools import partial
from typing import Literal

import numpy as np
import torch

try:  # Registers torch.ops.dendra_solvers when available.
    import dendra_solvers  # noqa: F401

    DENDRA_SOLVERS_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on local installation
    DENDRA_SOLVERS_AVAILABLE = False

try:
    from dendra.models.integrators.tridiag import pcr_solve_t
except Exception:  # pragma: no cover - import robustness for partial installs
    pcr_solve_t = None

try:
    from dendra.models.integrators.triton import (
        dhs_solve_cuda,
        pcr_solve_cuda_t,
        thomas_solve_cuda_t,
    )
except Exception:  # pragma: no cover - triton may be unavailable
    dhs_solve_cuda = None
    pcr_solve_cuda_t = None
    thomas_solve_cuda_t = None


def _as_tensor_like(value, like: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)


def _broadcast_to(value, target: torch.Tensor) -> torch.Tensor:
    value = _as_tensor_like(value, target)
    if value.ndim == 0:
        return value.expand_as(target)
    while value.ndim < target.ndim:
        value = value.unsqueeze(0)
    return value.expand_as(target)


def _flatten_last(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    shape = tuple(x.shape)
    if x.ndim < 1:
        raise ValueError("spatial operators require at least one spatial dimension")
    return x.reshape(-1, shape[-1]), shape


def _dense_tridiagonal_solve(a, b, c, rhs):
    """Debug/fallback batched tridiagonal solve.

    a, c: (..., K-1), b/rhs: (..., K).  The solve is over the final dimension.
    """
    rhs2, shape = _flatten_last(rhs)
    K = rhs2.shape[-1]
    a2 = a.reshape(-1, max(K - 1, 0))
    b2 = b.reshape(-1, K)
    c2 = c.reshape(-1, max(K - 1, 0))
    B = rhs2.shape[0]
    A = rhs2.new_zeros((B, K, K))
    idx = torch.arange(K, device=rhs2.device)
    A[:, idx, idx] = b2
    if K > 1:
        eidx = torch.arange(K - 1, device=rhs2.device)
        A[:, eidx + 1, eidx] = a2
        A[:, eidx, eidx + 1] = c2
    out = torch.linalg.solve(A, rhs2.unsqueeze(-1)).squeeze(-1)
    return out.reshape(shape)


def _normalize_solver_name(solver: str | None) -> str:
    mode = str(solver or "auto").lower()
    aliases = {
        "inv": "thomas",
        "custom": "thomas",
        "cuda": "thomas",
        "cpu": "thomas",
        "torch": "dense",
        "linalg": "dense",
    }
    return aliases.get(mode, mode)


def select_tridiagonal_solver(solver: str | None, device: torch.device | str):
    """Select a tridiagonal solver using Dendra voltage-integrator conventions.

    The policy mirrors the unbranched voltage integrator:

    - ``solver='pcr'`` explicitly requests PCR/Torch fallback.
    - ``solver='spd'`` uses the CPU SPD tridiagonal solver when available and
      falls back to Thomas-style solvers otherwise.
    - ``solver='thomas'``/``'auto'`` uses Triton Thomas on CUDA, Dendra's CPU
      extension on CPU when available, and the pure-PyTorch PCR/Thomas fallback
      when needed.
    """
    mode = _normalize_solver_name(solver)
    dev_type = torch.device(device).type if not isinstance(device, str) else device

    if mode in {"dense", "debug"}:
        return None, "dense"

    if mode == "pcr":
        if dev_type == "cuda":
            if pcr_solve_cuda_t is None:
                raise ImportError(
                    "solver='pcr' on CUDA requires dendra.models.integrators.triton.pcr_solve_cuda_t."
                )
            return pcr_solve_cuda_t, "pcr_cuda"
        if pcr_solve_t is None:
            raise ImportError(
                "solver='pcr' requires dendra.models.integrators.tridiag.pcr_solve_t."
            )
        return pcr_solve_t, "pcr_cpu"

    if mode == "spd":
        if dev_type == "cuda":
            warnings.warn(
                "Material diffusion solver='spd' is not implemented on CUDA; falling back to Thomas."
            )
            mode = "thomas"
        else:
            try:
                from dendra_solvers import solve_tri_spd
            except ImportError:
                warnings.warn(
                    "Material diffusion solver='spd' requested on CPU but dendra_solvers.solve_tri_spd "
                    "is unavailable; falling back to Thomas/PCR."
                )
                mode = "thomas"
            else:
                return solve_tri_spd, "spd_cpu"

    if mode not in {"auto", "thomas"}:
        warnings.warn(
            f"Unknown material tridiagonal solver {solver!r}; falling back to 'thomas'."
        )
        mode = "thomas"

    if dev_type == "cuda":
        if thomas_solve_cuda_t is not None:
            return thomas_solve_cuda_t, "thomas_cuda"
        if pcr_solve_cuda_t is not None:
            warnings.warn(
                "Triton Thomas solver unavailable; falling back to CUDA PCR solver."
            )
            return pcr_solve_cuda_t, "pcr_cuda"
        if mode == "thomas":
            raise ImportError(
                "solver='thomas' on CUDA requires dendra.models.integrators.triton.thomas_solve_cuda_t."
            )
        return None, "dense"

    if dev_type == "cpu":
        if DENDRA_SOLVERS_AVAILABLE:
            return torch.ops.dendra_solvers.thomas_solve_t, "thomas_cpu"
        if pcr_solve_t is not None:
            warnings.warn(
                "Using material diffusion on CPU without dendra_solvers installed; falling back to PCR/Thomas torch solver."
            )
            return pcr_solve_t, "pcr_cpu"
        return None, "dense"

    return None, "dense"


def solve_tridiagonal_1d(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    rhs: torch.Tensor,
    *,
    solver: Literal[
        "auto", "custom", "torch", "dense", "thomas", "pcr", "spd", "inv"
    ] = "auto",
) -> torch.Tensor:
    """Solve a batched tridiagonal system along the final dimension.

    This function remains as a public/debug helper.  Hot material-process paths
    should prefer :class:`SpatialOperator1D`, which selects the solver and
    precomputes static bands outside the timestep loop.
    """
    if rhs.shape[-1] == 1:
        return rhs / b

    rhs2, shape = _flatten_last(rhs)
    K = rhs2.shape[-1]
    a2 = a.reshape(-1, K - 1).contiguous()
    b2 = b.reshape(-1, K).contiguous()
    c2 = c.reshape(-1, K - 1).contiguous()
    d2 = rhs2.contiguous()

    mode = _normalize_solver_name(solver)
    if mode in {"dense", "torch", "linalg", "debug"}:
        return _dense_tridiagonal_solve(a, b, c, rhs)

    solve_fn, _ = select_tridiagonal_solver(mode, d2.device)
    if solve_fn is None:
        return _dense_tridiagonal_solve(a, b, c, rhs)
    out = solve_fn(a2, b2, c2, d2)
    return out.reshape(shape)


class SpatialOperator1D(torch.nn.Module):
    """Finite-volume helper for 1D compartment chains.

    The operator assumes the final tensor dimension is the compartment axis; all
    leading dimensions are independent batches/fibers.  Geometry is expected in
    micrometers.  Diffusivities are therefore expected in ``um^2 / ms`` by
    default, and the implicit solve advances concentration-like fields in ms.

    Unlike the convenience :func:`solve_tridiagonal_1d`, this module follows the
    same split used by Dendra's voltage integrators: solver selection and static
    geometry/band construction happen in ``configure_diffusion(...)``; the
    per-step ``diffuse_*_configured(...)`` methods only assemble the RHS and call
    the preselected solver.
    """

    def __init__(self, *, solver: str = "auto", boundary: str = "sealed"):
        super().__init__()
        self.solver = _normalize_solver_name(solver)
        self.boundary = str(boundary or "sealed").lower()
        if self.boundary not in {"sealed", "no_flux", "noflux"}:
            raise NotImplementedError(
                "SpatialOperator1D MVP supports only sealed/no-flux boundaries."
            )
        self.configured = False
        self.solver_name = "unconfigured"
        self._solve = None
        self.base_shape: tuple[int, ...] | None = None
        self.B = 0
        self.K = 0

    def _set_buffer(self, name: str, value: torch.Tensor) -> None:
        if name in self._buffers:
            self._buffers[name] = value
        else:
            self.register_buffer(name, value)

    @staticmethod
    def cylinder_cross_section_um2(diam_um: torch.Tensor) -> torch.Tensor:
        return torch.pi * (diam_um * 0.5) ** 2

    @staticmethod
    def cylinder_volume_um3(diam_um: torch.Tensor, dx_um: torch.Tensor) -> torch.Tensor:
        return SpatialOperator1D.cylinder_cross_section_um2(diam_um) * dx_um

    def finite_volume_geometry(
        self,
        diam_um: torch.Tensor,
        dx_um: torch.Tensor,
        *,
        volume_fraction=1.0,
        area_fraction=1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return compartment volumes and interface areas for a 1D cable."""
        dx_um = _broadcast_to(dx_um, diam_um)
        area = self.cylinder_cross_section_um2(diam_um)
        volume = area * dx_um * _broadcast_to(volume_fraction, diam_um)
        edge_area = 0.5 * (area[..., :-1] + area[..., 1:])
        edge_area = edge_area * _broadcast_to(area_fraction, edge_area)
        return volume, edge_area

    def edge_conductance(
        self,
        D_um2_per_ms,
        diam_um: torch.Tensor,
        dx_um: torch.Tensor,
        *,
        volume_fraction=1.0,
        area_fraction=1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(volume, g_edge)`` for finite-volume diffusion."""
        volume, edge_area = self.finite_volume_geometry(
            diam_um,
            dx_um,
            volume_fraction=volume_fraction,
            area_fraction=area_fraction,
        )
        dx_um = _broadcast_to(dx_um, diam_um)
        edge_len = 0.5 * (dx_um[..., :-1] + dx_um[..., 1:])

        D_full = _broadcast_to(D_um2_per_ms, diam_um)
        D_edge = 0.5 * (D_full[..., :-1] + D_full[..., 1:])
        tiny = torch.finfo(edge_len.dtype).tiny
        g_edge = D_edge * edge_area / edge_len.clamp_min(tiny)
        return volume, g_edge

    def configure_diffusion(
        self,
        c_like: torch.Tensor,
        dt,
        D_um2_per_ms,
        diam_um: torch.Tensor,
        dx_um: torch.Tensor,
        *,
        volume_fraction=1.0,
        area_fraction=1.0,
        solver: str | None = None,
    ) -> "SpatialOperator1D":
        """Precompute geometry, solver choice, and timestep-scaled bands."""
        if c_like.ndim < 1:
            raise ValueError("diffusion fields must have a final compartment dimension")
        if c_like.shape[-1] <= 0:
            raise ValueError("diffusion fields must have at least one compartment")

        c_ref = c_like
        dt_t = _as_tensor_like(dt, c_ref)
        diam_um = _broadcast_to(diam_um, c_ref)
        dx_um = _broadcast_to(dx_um, c_ref)
        volume, g_edge = self.edge_conductance(
            D_um2_per_ms,
            diam_um,
            dx_um,
            volume_fraction=volume_fraction,
            area_fraction=area_fraction,
        )

        vol2, base_shape = _flatten_last(volume)
        K = int(vol2.shape[-1])
        B = int(vol2.shape[0])
        self.B = B
        self.K = K
        self.base_shape = base_shape

        dt_t = dt_t.to(device=c_ref.device, dtype=c_ref.dtype)
        if dt_t.ndim != 0:
            # Broadcastable tensor dt is allowed, but store it in flattened solve
            # layout so the compiled hot path never repeats shape normalization.
            dt_t = _broadcast_to(dt_t, c_ref).reshape(B, K)
        self._set_buffer("dt", dt_t)

        if K == 1:
            self._set_buffer("volume", vol2)
            self._set_buffer(
                "inv_volume", 1.0 / vol2.clamp_min(torch.finfo(vol2.dtype).tiny)
            )
            self._set_buffer("g_edge", vol2.new_empty((B, 0)))
            self._set_buffer("lower", vol2.new_empty((B, 0)))
            self._set_buffer("upper", vol2.new_empty((B, 0)))
            self._set_buffer("main", vol2)
            self._solve = None
            self.solver_name = "trivial"
            self.configured = True
            return self

        g2 = g_edge.reshape(B, K - 1).contiguous()
        zeros = vol2.new_zeros((B, 1))
        left_g = torch.cat((zeros, g2), dim=-1)
        right_g = torch.cat((g2, zeros), dim=-1)

        if dt_t.ndim == 0:
            lower = -dt_t * g2
            upper = lower
            main = vol2 + dt_t * (left_g + right_g)
        else:
            dt_left = 0.5 * (dt_t[:, :-1] + dt_t[:, 1:])
            lower = -dt_left * g2
            upper = lower
            main = vol2 + dt_t * (left_g + right_g)

        self._set_buffer("volume", vol2.contiguous())
        self._set_buffer(
            "inv_volume",
            (1.0 / vol2.clamp_min(torch.finfo(vol2.dtype).tiny)).contiguous(),
        )
        self._set_buffer("g_edge", g2)
        self._set_buffer("left_g", left_g.contiguous())
        self._set_buffer("right_g", right_g.contiguous())
        self._set_buffer("lower", lower.contiguous())
        self._set_buffer("upper", upper.contiguous())
        self._set_buffer("main", main.contiguous())

        solve_fn, solver_name = select_tridiagonal_solver(
            solver or self.solver, c_ref.device
        )
        self._solve = solve_fn
        self.solver_name = solver_name
        self.configured = True
        return self

    def _require_configured(self) -> None:
        if not self.configured:
            raise RuntimeError(
                "SpatialOperator1D has not been configured. Call configure_diffusion(...) "
                "from the owning MaterialProcess.set_dt(...) before advancing."
            )

    def diffuse_implicit_configured(self, c: torch.Tensor) -> torch.Tensor:
        """Backward-Euler sealed-boundary update using precomputed bands."""
        self._require_configured()
        if self.K <= 1:
            return c
        c2 = c.reshape(self.B, self.K).contiguous()
        rhs = self._buffers["volume"] * c2
        if self._solve is None:
            out = _dense_tridiagonal_solve(
                self._buffers["lower"],
                self._buffers["main"],
                self._buffers["upper"],
                rhs,
            ).reshape(self.B, self.K)
        else:
            out = self._solve(
                self._buffers["lower"],
                self._buffers["main"],
                self._buffers["upper"],
                rhs,
            )
        return out.reshape(self.base_shape)

    def diffuse_explicit_configured(self, c: torch.Tensor) -> torch.Tensor:
        """Explicit sealed-boundary finite-volume update using precomputed geometry."""
        self._require_configured()
        if self.K <= 1:
            return c
        c2 = c.reshape(self.B, self.K)
        edge_flux = self._buffers["g_edge"] * (c2[:, 1:] - c2[:, :-1])
        net = torch.zeros_like(c2)
        net[:, :-1] = net[:, :-1] + edge_flux
        net[:, 1:] = net[:, 1:] - edge_flux
        dt_t = self._buffers["dt"]
        return (c2 + dt_t * net * self._buffers["inv_volume"]).reshape(self.base_shape)

    # ------------------------------------------------------------------
    # Backward-compatible convenience methods.  These recompute coefficients
    # and should not be used in the hot MaterialProcess path.
    # ------------------------------------------------------------------
    def diffuse_implicit(
        self,
        c: torch.Tensor,
        dt,
        D_um2_per_ms,
        diam_um: torch.Tensor,
        dx_um: torch.Tensor,
        *,
        volume_fraction=1.0,
        area_fraction=1.0,
        solver: str | None = None,
    ) -> torch.Tensor:
        self.configure_diffusion(
            c,
            dt,
            D_um2_per_ms,
            diam_um,
            dx_um,
            volume_fraction=volume_fraction,
            area_fraction=area_fraction,
            solver=solver,
        )
        return self.diffuse_implicit_configured(c)

    def diffuse_explicit(
        self,
        c: torch.Tensor,
        dt,
        D_um2_per_ms,
        diam_um: torch.Tensor,
        dx_um: torch.Tensor,
        *,
        volume_fraction=1.0,
        area_fraction=1.0,
    ) -> torch.Tensor:
        self.configure_diffusion(
            c,
            dt,
            D_um2_per_ms,
            diam_um,
            dx_um,
            volume_fraction=volume_fraction,
            area_fraction=area_fraction,
            solver=self.solver,
        )
        return self.diffuse_explicit_configured(c)


# ---------------------------------------------------------------------------
# Tree / DHS finite-volume operators
# ---------------------------------------------------------------------------

THREADS_PER_WARP = 32


def _build_tree_morphology(
    parent_idx: list[int],
) -> tuple[torch.Tensor, list[list[int]], torch.Tensor]:
    """Build DHS/Hines morphology metadata from a parent-index vector.

    This mirrors :func:`dendra.models.integrators.tree.build_morphology` but is
    kept local to avoid importing the full voltage-integrator module from the
    mechanism/material namespace.
    """
    K = len(parent_idx)
    children: list[list[int]] = [[] for _ in range(K)]
    root = None
    for i, p in enumerate(parent_idx):
        p = int(p)
        if p == -1:
            if root is not None:
                raise ValueError("tree morphology has more than one root")
            root = i
        else:
            if p < 0 or p >= K:
                raise ValueError(f"invalid parent index {p} for node {i}")
            children[p].append(i)

    if root is None:
        raise ValueError("tree morphology must have one root with parent=-1")

    depth = torch.zeros(K, dtype=torch.int32)
    visited = torch.zeros(K, dtype=torch.bool)
    visited[root] = True
    q = deque([root])
    while q:
        u = q.popleft()
        for child in children[u]:
            if visited[child]:
                raise ValueError("tree morphology contains a cycle")
            visited[child] = True
            depth[child] = depth[u] + 1
            q.append(child)

    if not bool(torch.all(visited)):
        raise ValueError(
            "tree morphology must be connected; unreachable nodes may form a cycle"
        )

    return torch.as_tensor(parent_idx, dtype=torch.int32), children, depth


def _build_dhs_layers(
    depth: torch.Tensor, threads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build DHS elimination order/layer pointers, deepest nodes first."""
    depth_cpu = depth.detach().cpu().numpy()
    max_d = int(depth_cpu.max()) if depth_cpu.size else 0

    order: list[int] = []
    layer_ptr: list[int] = [0]
    bins: list[list[int]] = [[] for _ in range(max_d + 1)]
    for i, d in enumerate(depth_cpu):
        bins[int(d)].append(i)

    for d in range(max_d, -1, -1):
        bucket = bins[d]
        for start in range(0, len(bucket), threads):
            order.extend(bucket[start : start + threads])
            layer_ptr.append(len(order))

    return torch.as_tensor(order, dtype=torch.int32), torch.as_tensor(
        layer_ptr, dtype=torch.int32
    )


def _edge_diff_geom_from_graph(graph, parent, child) -> float:
    """Return edge diffusion geometry in µm from a graph edge.

    Preferred graph metadata is ``edge['diff_geom_um']`` generated by Dendra's
    NEURON import path.  The resistance-based fallback is only for older or
    hand-written graphs and assumes a single effective axial resistivity.
    """
    edge = graph.edges[parent, child]
    if "diff_geom_um" in edge:
        return float(edge["diff_geom_um"])

    R_ohm = edge.get("R_ohm", None)
    if R_ohm is not None and float(R_ohm) != 0.0:
        parent_Ra = graph.nodes[parent].get("Ra", None)
        child_Ra = graph.nodes[child].get("Ra", None)
        if parent_Ra is None and child_Ra is None:
            raise KeyError(
                "Edge is missing diff_geom_um and cannot recover it from R_ohm "
                "because neither endpoint has Ra."
            )
        if parent_Ra is None:
            rhoa = float(child_Ra)
        elif child_Ra is None:
            rhoa = float(parent_Ra)
        else:
            rhoa = 0.5 * (float(parent_Ra) + float(child_Ra))
        return float(rhoa * 1e4 / float(R_ohm))

    # Last-resort stylized fallback, useful for minimal tests.
    L_edge = float(edge.get("L", 0.0) or 0.0)
    if L_edge <= 0.0:
        return 0.0
    d_parent = float(graph.nodes[parent].get("diam", 0.0) or 0.0)
    d_child = float(graph.nodes[child].get("diam", 0.0) or 0.0)
    if d_parent <= 0.0 or d_child <= 0.0:
        return 0.0
    a_parent = float(np.pi * (0.5 * d_parent) ** 2)
    a_child = float(np.pi * (0.5 * d_child) ** 2)
    integral = 0.5 * L_edge / a_parent + 0.5 * L_edge / a_child
    return float(1.0 / integral) if integral > 0.0 else 0.0


def graph_to_parent_and_diffusion(graphs, dtype: torch.dtype = torch.float32):
    """Convert tree graph(s) into DHS parent indices and diffusion geometry.

    Returns
    -------
    parent_idx : torch.Tensor
        Length-K parent vector in solver/topological node order.
    diff_geom_um : torch.Tensor
        Shape ``(n_graphs, K)``. Entry ``[..., i]`` is the diffusion geometry of
        the edge connecting node ``i`` to its parent. The root entry is zero.
    node_order : list
        Mapping from solver index to original graph node label.
    """
    if not isinstance(graphs, (list, tuple)):
        graphs = [graphs]
    if len(graphs) == 0:
        raise ValueError("at least one graph is required")

    # Import networkx lazily so non-tree material diffusion does not require it.
    import networkx as nx  # noqa: PLC0415

    nodes = list(nx.topological_sort(graphs[0]))
    idx_of = {node: i for i, node in enumerate(nodes)}
    K = len(nodes)

    parent_idx = np.full(K, -1, dtype=np.int32)
    diff_geom = np.zeros((len(graphs), K), dtype=np.float64)

    # Topology is expected to be shared.  We check enough to produce useful
    # errors without importing Dendra's graph helper here.
    edge_signature = None
    for gi, graph in enumerate(graphs):
        graph_nodes = list(nx.topological_sort(graph))
        if graph_nodes != nodes:
            raise AssertionError(
                "all tree diffusion graphs must share the same topological order"
            )
        signature = []
        for child in nodes:
            preds = list(graph.predecessors(child))
            if not preds:
                signature.append((child, None))
                continue
            if len(preds) > 1:
                raise ValueError(
                    f"Node {child!r} has {len(preds)} parents; material diffusion requires a rooted tree."
                )
            parent = preds[0]
            signature.append((child, parent))

            i = idx_of[child]
            p = idx_of[parent]
            if gi == 0:
                parent_idx[i] = p
            diff_geom[gi, i] = _edge_diff_geom_from_graph(graph, parent, child)

        if edge_signature is None:
            edge_signature = signature
        elif signature != edge_signature:
            raise AssertionError(
                "all tree diffusion graphs must share the same parent/child topology"
            )

    return (
        torch.as_tensor(parent_idx, dtype=torch.int32),
        torch.as_tensor(diff_geom, dtype=dtype),
        nodes,
    )


def _select_dhs_solver(solver: str | None, device: torch.device | str, *, threads: int):
    # Resolve tree-specific automatic aliases before the generic tridiagonal
    # aliases: the latter maps ``custom`` to an explicit Thomas request.
    mode = str(solver or "auto").lower()
    aliases = {"dhs": "auto", "hines": "auto", "tree": "auto", "custom": "auto"}
    mode = aliases.get(mode, _normalize_solver_name(mode))
    dev_type = torch.device(device).type if not isinstance(device, str) else device

    if mode in {"dense", "debug", "torch", "linalg"}:
        return None, "dense"
    if mode not in {"auto", "thomas"}:
        warnings.warn(
            f"Unknown tree material solver {solver!r}; falling back to DHS auto solver."
        )
        mode = "auto"

    if dev_type == "cuda":
        if dhs_solve_cuda is None:
            if mode == "thomas":
                raise ImportError(
                    "Tree material diffusion on CUDA requires dendra.models.integrators.triton.dhs_solve_cuda."
                )
            return None, "dense"
        return partial(dhs_solve_cuda, threads=threads), "dhs_cuda"

    if dev_type == "cpu":
        if DENDRA_SOLVERS_AVAILABLE:
            return torch.ops.dendra_solvers.dhs_solve, "dhs_cpu"
        if mode == "thomas":
            raise ImportError(
                "Tree material diffusion solver='thomas' on CPU requires dendra_solvers.dhs_solve."
            )
        warnings.warn(
            "Using tree material diffusion on CPU without dendra_solvers installed; falling back to dense torch.linalg.solve."
        )
        return None, "dense"

    return None, "dense"


def _dense_tree_solve(
    d_mem: torch.Tensor,
    a_geom: torch.Tensor,
    rhs: torch.Tensor,
    parent_idx: torch.Tensor,
) -> torch.Tensor:
    """Dense debug fallback for a DHS/Hines tree system.

    ``d_mem`` is the mass/membrane diagonal term and ``a_geom[..., child]`` is
    the parent-child coupling for each child.  The final matrix diagonal is
    ``d_mem + sum(edge couplings incident to node)`` with off-diagonal entries
    ``-g``.
    """
    B, K = rhs.shape
    A = rhs.new_zeros((B, K, K))
    idx = torch.arange(K, device=rhs.device)
    A[:, idx, idx] = d_mem
    for child in range(K):
        p = int(parent_idx[child].item())
        if p < 0:
            continue
        g = a_geom[:, child]
        A[:, child, child] = A[:, child, child] + g
        A[:, p, p] = A[:, p, p] + g
        A[:, child, p] = A[:, child, p] - g
        A[:, p, child] = A[:, p, child] - g
    return torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)


class SpatialOperatorTree(torch.nn.Module):
    """Finite-volume implicit diffusion operator for branched tree morphologies.

    This operator mirrors Dendra's DHS voltage integrator.  The same tree
    topology/elimination layout is used, but the matrix represents material
    conservation rather than membrane voltage:

    ``d_mem = volume / dt`` and ``a_geom = D * diff_geom_um``.

    Branchpoints with zero volume are handled naturally by the implicit solve as
    algebraic flux-balance junctions.  Explicit diffusion rejects zero-volume
    nodes because they would require division by zero.
    """

    def __init__(
        self, *, solver: str = "auto", boundary: str = "sealed", threads: int = 16
    ):
        super().__init__()
        threads = int(threads)
        if (
            threads <= 0
            or threads > THREADS_PER_WARP
            or THREADS_PER_WARP % threads != 0
        ):
            raise ValueError(
                "threads must be positive, divide 32, and be <= 32 for DHS-style tree solves"
            )
        self.solver = _normalize_solver_name(solver)
        self.boundary = str(boundary or "sealed").lower()
        if self.boundary not in {"sealed", "no_flux", "noflux"}:
            raise NotImplementedError(
                "SpatialOperatorTree MVP supports only sealed/no-flux boundaries."
            )
        self.threads = threads
        self.configured = False
        self.solver_name = "unconfigured"
        self._solve = None
        self.base_shape: tuple[int, ...] | None = None
        self.B = 0
        self.K = 0

    def _set_buffer(self, name: str, value: torch.Tensor) -> None:
        if name in self._buffers:
            self._buffers[name] = value
        else:
            self.register_buffer(name, value)

    @staticmethod
    def _domain_volume(model, domain: str | None):
        domain = str(domain or "intracellular").lower()
        if hasattr(model, "material_volume"):
            return model.material_volume(domain)
        if domain in {
            "i",
            "inside",
            "intracellular",
            "cytosol",
            "cytoplasm",
        } and hasattr(model, "volume_i"):
            return model.volume_i
        if domain in {"o", "outside", "extracellular"} and hasattr(model, "volume_o"):
            return model.volume_o
        if domain in {"membrane", "surface"} and hasattr(model, "area"):
            return model.area
        if hasattr(model, "volume"):
            return model.volume
        raise AttributeError(
            "Tree material diffusion requires a model volume buffer. Apply the Tree material-geometry patch first."
        )

    def configure_diffusion(
        self,
        c_like: torch.Tensor,
        dt,
        D_um2_per_ms,
        model,
        *,
        domain: str | None = "intracellular",
        solver: str | None = None,
    ) -> "SpatialOperatorTree":
        """Precompute DHS topology, mass term, diffusion couplings, and solver."""
        if c_like.ndim < 1:
            raise ValueError("diffusion fields must have a final compartment dimension")
        K = int(c_like.shape[-1])
        if K <= 0:
            raise ValueError("diffusion fields must have at least one compartment")

        graph = getattr(model, "graph", None)
        if graph is None:
            try:
                graph = model.assemble_graphs()
            except Exception as err:  # pragma: no cover - model dependent
                raise ValueError(
                    "Tree material diffusion requires `model.graph` or `model.assemble_graphs()`."
                ) from err

        graph_arg = graph if isinstance(graph, (list, tuple)) else [graph]
        parent_idx_t, diff_geom_t, node_order = graph_to_parent_and_diffusion(
            graph_arg, dtype=c_like.dtype
        )
        if len(node_order) != K:
            raise ValueError(
                f"tree graph has {len(node_order)} compartments, but field has final dimension {K}."
            )

        parent_idx, _, depth = _build_tree_morphology(parent_idx_t.tolist())
        order, layer_ptr = _build_dhs_layers(depth, self.threads)

        device = c_like.device
        dtype = c_like.dtype
        solver_order = torch.as_tensor(node_order, dtype=torch.long, device=device)
        inv_solver_order = torch.argsort(solver_order, dim=0)

        volume = self._domain_volume(model, domain).to(device=device, dtype=dtype)
        volume = _broadcast_to(volume, c_like)
        volume2, base_shape = _flatten_last(volume)
        B = int(volume2.shape[0])
        self.B = B
        self.K = K
        self.base_shape = base_shape

        # Align volume and field-space quantities to solver/topological order.
        volume_s = volume2.index_select(-1, solver_order).contiguous()

        D_full = _broadcast_to(D_um2_per_ms, c_like).reshape(B, K)
        D_s = D_full.index_select(-1, solver_order)

        parent_idx_dev = parent_idx.to(dtype=torch.long, device=device)
        D_edge_s = D_s.clone()
        for child in range(K):
            p = int(parent_idx[child].item())
            if p < 0:
                D_edge_s[:, child] = 0.0
            else:
                D_edge_s[:, child] = 0.5 * (D_s[:, child] + D_s[:, p])

        diff_geom_s = diff_geom_t.to(device=device, dtype=dtype)
        # graph_to_parent_and_diffusion returns one row per graph.  Most Tree
        # models have one graph and B population rows.  If multiple graph rows
        # are supplied and they match B, keep them; otherwise broadcast row 0.
        if diff_geom_s.shape[0] == 1:
            diff_geom_s = diff_geom_s.expand(B, -1)
        elif diff_geom_s.shape[0] == B:
            pass
        else:
            raise ValueError(
                f"diffusion graph batch has {diff_geom_s.shape[0]} rows, but material field has {B} solve rows."
            )

        g_diff_s = (D_edge_s * diff_geom_s).contiguous()

        dt_t = _as_tensor_like(dt, c_like)
        if dt_t.ndim != 0:
            dt_t = (
                _broadcast_to(dt_t, c_like).reshape(B, K).index_select(-1, solver_order)
            )
        self._set_buffer("dt", dt_t.to(device=device, dtype=dtype))

        tiny = torch.finfo(dtype).tiny
        if dt_t.ndim == 0:
            dmem = volume_s / dt_t.clamp_min(tiny)
        else:
            dmem = volume_s / dt_t.clamp_min(tiny)

        self._set_buffer("volume", volume2.contiguous())
        self._set_buffer("volume_solver", volume_s.contiguous())
        self._set_buffer("inv_volume", (1.0 / volume2.clamp_min(tiny)).contiguous())
        self._set_buffer("dmem", dmem.contiguous())
        self._set_buffer("a_geom", g_diff_s.contiguous())
        self._set_buffer("solver_order", solver_order)
        self._set_buffer("inv_solver_order", inv_solver_order)
        self._set_buffer("parent_idx", parent_idx_dev)
        self._set_buffer("order", order.to(dtype=torch.long, device=device))
        self._set_buffer("layer_ptr", layer_ptr.to(dtype=torch.long, device=device))

        solve_fn, solver_name = _select_dhs_solver(
            solver or self.solver, device, threads=self.threads
        )
        self._solve = solve_fn
        self.solver_name = solver_name
        self.configured = True
        return self

    def _require_configured(self) -> None:
        if not self.configured:
            raise RuntimeError(
                "SpatialOperatorTree has not been configured. Call configure_diffusion(...) "
                "from the owning MaterialProcess.set_dt(...) before advancing."
            )

    def diffuse_implicit_configured(self, c: torch.Tensor) -> torch.Tensor:
        """Backward-Euler sealed-boundary diffusion update on a tree."""
        self._require_configured()
        if self.K <= 1:
            return c
        c2 = c.reshape(self.B, self.K).contiguous()
        c_s = c2.index_select(-1, self._buffers["solver_order"])
        rhs_s = self._buffers["dmem"] * c_s

        if self._solve is None:
            out_s = _dense_tree_solve(
                self._buffers["dmem"],
                self._buffers["a_geom"],
                rhs_s,
                self._buffers["parent_idx"],
            )
        else:
            out_s = self._solve(
                self._buffers["dmem"],
                self._buffers["a_geom"],
                rhs_s,
                self._buffers["parent_idx"],
                self._buffers["order"],
                self._buffers["layer_ptr"],
            )

        out = out_s.index_select(-1, self._buffers["inv_solver_order"])
        return out.reshape(self.base_shape)

    def diffuse_explicit_configured(self, c: torch.Tensor) -> torch.Tensor:
        """Explicit tree diffusion update for debug use.

        This rejects zero-volume nodes, so it is not appropriate for Tree graphs
        containing branchpoints.  Use implicit diffusion for imported branched
        morphologies.
        """
        self._require_configured()
        if self.K <= 1:
            return c
        c2 = c.reshape(self.B, self.K).contiguous()
        c_s = c2.index_select(-1, self._buffers["solver_order"])
        volume_s = self._buffers["volume_solver"]
        if torch.any(volume_s <= 0):
            raise RuntimeError(
                "Explicit tree diffusion is undefined for zero-volume nodes/branchpoints; use implicit diffusion."
            )

        net = torch.zeros_like(c_s)
        parent_idx = self._buffers["parent_idx"]
        g = self._buffers["a_geom"]
        for child in range(self.K):
            p = int(parent_idx[child].item())
            if p < 0:
                continue
            flux = g[:, child] * (c_s[:, p] - c_s[:, child])
            net[:, child] = net[:, child] + flux
            net[:, p] = net[:, p] - flux

        dt_t = self._buffers["dt"]
        c_next_s = c_s + dt_t * net / volume_s
        c_next = c_next_s.index_select(-1, self._buffers["inv_solver_order"])
        return c_next.reshape(self.base_shape)

    # Backward-compatible convenience wrappers.
    def diffuse_implicit(
        self,
        c: torch.Tensor,
        dt,
        D_um2_per_ms,
        model,
        *,
        domain: str | None = "intracellular",
        solver: str | None = None,
    ) -> torch.Tensor:
        self.configure_diffusion(
            c, dt, D_um2_per_ms, model, domain=domain, solver=solver
        )
        return self.diffuse_implicit_configured(c)

    def diffuse_explicit(
        self,
        c: torch.Tensor,
        dt,
        D_um2_per_ms,
        model,
        *,
        domain: str | None = "intracellular",
    ) -> torch.Tensor:
        self.configure_diffusion(
            c, dt, D_um2_per_ms, model, domain=domain, solver=self.solver
        )
        return self.diffuse_explicit_configured(c)
