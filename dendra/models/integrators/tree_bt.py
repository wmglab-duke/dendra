from collections import deque
from functools import partial
from typing import List, Tuple

import networkx as nx
import torch

from .core import (
    Integrator,
    _as_solve_block,
    _as_solve_matrix,
    _expanded_v_init,
    _flatten_to_solve,
    _model_solve_shape,
)
from .triton import dhs_bt_solve_cuda

try:
    import dendra_solvers  # noqa:F401

    DENDRA_SOLVERS_AVAILABLE = True
except ImportError:
    DENDRA_SOLVERS_AVAILABLE = False


# ---------------- Topology helpers (local, to avoid extra deps) ----------------
def _topo_parent_depth(G: nx.DiGraph) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    nodes = list(nx.topological_sort(G))
    idx_of = {n: i for i, n in enumerate(nodes)}
    K = len(nodes)
    parent_idx = torch.full((K,), -1, dtype=torch.int64)
    depth = torch.zeros(K, dtype=torch.int64)
    roots = [n for n in nodes if G.in_degree(n) == 0]
    q = deque((r, 0) for r in roots)
    seen = set()
    while q:
        u, d = q.popleft()
        if u in seen:
            continue
        seen.add(u)
        for v in G.successors(u):
            parent_idx[idx_of[v]] = idx_of[u]
            depth[idx_of[v]] = d + 1
            q.append((v, d + 1))
    return parent_idx, depth, nodes


def _build_layers(
    depth: torch.Tensor, k_threads: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_d = int(depth.max().item()) if depth.numel() > 0 else 0
    bins = [[] for _ in range(max_d + 1)]
    for i, d in enumerate(depth.tolist()):
        bins[d].append(i)
    order: List[int] = []
    layer_ptr: List[int] = [0]
    for d in range(max_d, -1, -1):
        bucket = bins[d]
        for s in range(0, len(bucket), k_threads):
            chunk = bucket[s : s + k_threads]
            order.extend(chunk)
            layer_ptr.append(len(order))
    return torch.as_tensor(order, dtype=torch.int64), torch.as_tensor(
        layer_ptr, dtype=torch.int64
    )


def assemble_rhs(v_prev, c_rad, d, xg, e_ext):
    """RHS for block system (vi, ve0, ve1) following _bwd_euler_bt.
    Only *capacitive* radial terms appear in the RHS (cm_dt, xc_dt). Resistive xg
    is handled on the LHS, except for the outer-bath driving term xg[..., 1] * e_ext.
    """
    rhs = torch.zeros_like(v_prev)

    v_c = c_rad[..., :-1] * (v_prev[..., :-1] - v_prev[..., 1:])

    rhs[..., :-1] += v_c
    rhs[..., 1:] -= v_c

    rhs[..., 0] += d
    rhs[..., 1] -= d

    if e_ext is not None:
        rhs[..., -1] += xg[..., -1] * e_ext + c_rad[..., -1] * v_prev[..., -1]
    else:
        rhs[..., -1] += c_rad[..., -1] * v_prev[..., -1]

    return rhs


class _dhs_bt(Integrator):
    r"""
    DHS integrator for block-tridiagonal (3x3 per node) tree systems.

    Extends DHS to models with intracellular voltage and two extracellular
    shells (vi, ve0, ve1) per node, assembling block diagonals and axial
    conductances, then solving in DHS order with CUDA or CPU backends.

    Parameters
    ----------
    imem : bool or None, optional
        If truthy, accumulate membrane currents each step. Default None.
    threads : int, optional
        Threads per warp lane for DHS elimination (must divide 32). Default 16.
    """

    v_vars = ["v", "vc"]

    def __init__(self, model, mech, imem=None, threads=16):
        assert 32 % threads == 0 and threads <= 32
        super().__init__(model, mech, imem)
        self.threads = threads

        _, K = _model_solve_shape(model)

        self.K = K

        self.register_buffer("parent_idx", torch.empty(K, dtype=torch.int64))
        self.register_buffer("order", torch.empty(K, dtype=torch.int64))
        self.register_buffer("layer_ptr", torch.empty(1, dtype=torch.int64))
        self.register_buffer("solver_order", torch.empty(K, dtype=torch.int64))
        self.register_buffer("inv_solver_order", torch.empty(K, dtype=torch.int64))

        # Geometry-dependent buffers
        self.register_buffer("area", torch.empty(1, K))  # (1,K)
        self.register_buffer("cm_dt", torch.empty(1, K))  # (1,K)
        self.register_buffer("xc_dt", torch.empty(1, K, 2))  # (1,K,2)
        self.register_buffer("xg", torch.empty(1, K, 2))  # (1,K,2)

        # Per-node radial block (no axial degree baked in)
        self.register_buffer("main_blocks", torch.empty(1, K, 3, 3))
        # Per-edge diag g to parent (0:vi, 1:ve0, 2:ve1), 0 for roots
        self.register_buffer("g_to_parent", torch.empty(1, K, 3))

    def initialize(self, model, dt):
        dev, dtyp = model.device(), model.dtype()
        self.to(dev)

        if dev.type == "cpu" and not DENDRA_SOLVERS_AVAILABLE:
            raise ImportError(
                "DHS_BT integrator requires dendra_solvers package for CPU execution. "
                "Please install it with `pip install dendra_solvers`."
            )

        if dev.type == "cuda":
            self.solve = partial(dhs_bt_solve_cuda, threads=self.threads)
        elif dev.type == "cpu":
            self.solve = torch.ops.dendra_solvers.dhs_bt_solve
        else:
            raise NotImplementedError(f"Unsupported device type: {dev.type}")

        # Topology
        parent_idx, depth, node_order = _topo_parent_depth(model.graph)
        order, layer_ptr = _build_layers(depth.to(torch.int32), self.threads)
        self.solver_order.copy_(
            torch.as_tensor(node_order, dtype=torch.int64, device=dev)
        )
        self.inv_solver_order.copy_(torch.argsort(self.solver_order, dim=0))
        self.parent_idx.copy_(parent_idx.to(device=dev, dtype=torch.int64))
        self.order.copy_(order.to(device=dev, dtype=torch.int64))
        self.layer_ptr = layer_ptr.to(device=dev, dtype=torch.int64)

        self._refresh_solver_shape(model, block_dim=3)
        B = self.B
        K = self.K
        dt_s = dt * 1e-3  # convert to seconds for capacitance

        # Areas: recompute from geometry to guarantee cm² (matches unbranched BT)
        area_cm2 = _as_solve_matrix(model.area, model)
        self.area = area_cm2

        xc = _as_solve_block(model.xc, model, (2,))
        xg_param = _as_solve_block(model.xg, model, (2,))
        cm = _as_solve_matrix(model.cm, model)

        # Capacitances / conductances (per node)
        cm_dt = 1e-6 * cm * area_cm2 / dt_s  # (B,K)   F/s -> S
        xc_dt = 1e-6 * xc * area_cm2.unsqueeze(-1) / dt_s  # (B,K,2) F/s -> S
        xg = xg_param * area_cm2.unsqueeze(-1)  # (B,K,2) S

        # Zero-area branch points: explicitly zero radial terms (prevents NaNs)
        area_zero = area_cm2 == 0
        if area_zero.any():
            cm_dt = cm_dt.masked_fill(area_zero, 0.0)
            xc_dt = xc_dt.masked_fill(area_zero.unsqueeze(-1), 0.0)
            xg = xg.masked_fill(area_zero.unsqueeze(-1), 0.0)

        # Assemble radial 3x3 block
        main = torch.zeros(B, K, 3, 3, device=dev, dtype=dtyp)
        # diags
        main[..., 0, 0] = cm_dt
        main[..., 1, 1] = cm_dt + xc_dt[..., 0] + xg[..., 0]
        main[..., 2, 2] = xc_dt[..., 0] + xg[..., 0] + xc_dt[..., 1] + xg[..., 1]
        # off-diags
        main[..., 0, 1] = -cm_dt
        main[..., 1, 0] = -cm_dt
        coup01 = -(xc_dt[..., 0] + xg[..., 0])
        main[..., 1, 2] = coup01
        main[..., 2, 1] = coup01

        # ------------------ Per-edge conductances (to parent) ------------------
        from .tree import graph_to_parent_and_axial

        # Intracellular axial (returned in SOLVER order)
        _, g_intra_solver, _ = graph_to_parent_and_axial(
            model.graph, dtype_axial=dtyp
        )  # (K,) S
        g_intra_solver = g_intra_solver.to(device=dev)
        # Map to MECHANISM order so it matches dx/area/xraxial layout
        # solver_order[s] == mechanism index of the s-th topo (solver) node
        g_intra_mech = torch.empty_like(g_intra_solver)
        g_intra_mech.index_copy_(0, self.solver_order, g_intra_solver)

        # parent_idx is in SOLVER order. Map it to MECHANISM order for geometry gathers.
        parent_solver = self.parent_idx.to(torch.long)  # (K,) solver index space
        solver2mech = self.solver_order.to(torch.long)  # maps solver -> mechanism
        parent_mech = torch.full_like(parent_solver, -1)  # (K,)
        mask_nr = parent_solver >= 0
        parent_mech[mask_nr] = solver2mech[
            parent_solver[mask_nr]
        ]  # (K,) mechanism index space

        # Masks in mechanism order
        non_root = (parent_mech >= 0).view(1, -1, 1)  # (1,K,1)

        # Child/parent edge lengths (cm), mechanism order
        dx_cm = 1e-4 * _as_solve_matrix(model.dx, model)  # (B,K)
        dx_parent = dx_cm.gather(
            1, parent_mech.clamp_min(0).view(1, -1).expand(B, -1)
        )  # (B,K)

        # Per-shell resistivities per side (Ω·cm), mechanism order
        xrax_child = _as_solve_block(model.xraxial, model, (2,))  # (B,K,2)
        xrax_parent = xrax_child.gather(
            1, parent_mech.clamp_min(0).view(1, -1, 1).expand(B, -1, 2)
        )  # (B,K,2)

        # Edge resistance = average of half-segments (Ω); then conductance (S)
        # R_edge = 0.5 * (xrax_child * dx_child + xrax_parent * dx_parent) * 1e6 [Ω]
        R_edge = (
            0.5
            * (xrax_child * dx_cm.unsqueeze(-1) + xrax_parent * dx_parent.unsqueeze(-1))
            * 1e6
        )
        g_layers = torch.zeros_like(R_edge)
        good = non_root & torch.isfinite(R_edge) & (R_edge > 0)
        g_layers[good] = 1.0 / R_edge[good]  # (B,K,2) S

        # Package per-edge g to parent (mechanism order). Reindex to solver order at solve time.
        g_to_parent = torch.zeros(B, K, 3, device=dev, dtype=dtyp)
        g_to_parent[..., 0] = g_intra_mech.expand(B, -1)  # intracellular S
        g_to_parent[..., 1:] = g_layers  # shells S

        # Save
        self.cm_dt = cm_dt
        self.xc_dt = xc_dt
        self.xg = xg
        self.main_blocks = main.index_select(1, self.solver_order)
        self.g_to_parent = g_to_parent.index_select(1, self.solver_order)

        self.base_shape = tuple(list(model.shape) + [3])

        # State vectors
        if not hasattr(model, "vc"):
            model.register_buffer(
                "vc", torch.zeros(*model.shape, 3, device=dev, dtype=dtyp)
            )
        elif tuple(model.vc.shape) != tuple(model.shape) + (3,):
            model.vc = torch.zeros(*model.shape, 3, device=dev, dtype=dtyp)
        v0 = _expanded_v_init(model)
        model.vc[..., 0] = v0
        model.vc[..., 1] = 0.0
        model.vc[..., 2] = 0.0
        if not hasattr(model, "v"):
            model.register_buffer(
                "v", torch.zeros(*model.shape, device=dev, dtype=dtyp)
            )
        model.v[:] = v0

    def step(self, model, dt, ve=None, intra=None):
        vc_new, v_new = self._call_kernel(
            "_step",
            self._flat_block_voltage(model.vc, 3),
            model.v,
            dt,
            model.celsius,
            ve,
            intra,
        )
        model.vc = vc_new
        model.v = v_new

    def _step(self, vc, v, dt, temp, ve=None, intra=None):
        # Update mechanisms in mV / mA/cm^2
        v = self.mech.update_v(v)
        self.mech.advance(v, dt, temp)
        itot, gtot = self.mech.i(v)  # itot: mA/cm^2, gtot: S/cm^2

        # RHS (mechanism order), keep everything in mV/mA/S:
        # d_lin = (g*v - itot) * area  [mA]
        v_flat = self._flat_voltage(v)
        d_lin = (
            self._flat_voltage(gtot) * v_flat - self._flat_voltage(itot)
        ) * self.area
        if intra is not None:
            d_lin = d_lin + _flatten_to_solve(
                intra, self.K, self.shape
            )  # assume intra is already in mA

        # Capacitive+shell terms (S) operate on mV to yield mA
        c_rad = torch.cat(
            [self.cm_dt.unsqueeze(-1), self.xc_dt], dim=-1
        )  # (B,K,3) in S
        ve_flat = _flatten_to_solve(ve, self.K, self.shape) if ve is not None else None
        rhs_mech = assemble_rhs(vc, c_rad, d_lin, self.xg, ve_flat)  # (B,K,3) in mA

        # Reorder into solver order
        idx = self.solver_order
        rhs_ = rhs_mech.index_select(1, idx)  # (B,K,3)
        G_ = self.g_to_parent

        # Inject membrane gtot (scaled by area) into [vi, ve0] block (solver order)
        g_mech = self._flat_voltage(gtot) * self.area  # (B,K) S (mechanism order)
        g_ = g_mech.index_select(1, idx)  # (B,K) S (solver order)

        Dm = self.main_blocks.clone()  # (B, K, 3, 3)
        Dm[..., 0, 0] += g_
        Dm[..., 1, 1] += g_
        Dm[..., 0, 1] -= g_
        Dm[..., 1, 0] -= g_

        # Solve in solver order (kernel expects S @ mV = mA)
        X_ = self.solve(
            Dm,
            G_,
            rhs_,
            self.parent_idx,
            self.order,
            self.layer_ptr,
        )  # (B,K,3) in mV

        # Map solution back to mechanism order
        inv = self.inv_solver_order
        vc_out = X_.index_select(1, inv).reshape(self.base_shape)  # (B,K,3) mV
        v_out = vc_out[..., 0] - vc_out[..., 1]  # membrane (mV)

        return vc_out, v_out

    def init_v(self, model):
        v0 = _expanded_v_init(model).clone().detach().contiguous()
        if not hasattr(model, "vc") or tuple(model.vc.shape) != tuple(model.shape) + (
            3,
        ):
            model.register_buffer(
                "vc",
                torch.zeros(
                    *model.shape, 3, device=model.device(), dtype=model.dtype()
                ),
            )
        model.vc.zero_()
        model.vc[..., 0] = v0
        model.v = v0.clone()
        model.vc = model.vc.detach()
        model.v = model.v.detach()
        if self.imem:
            model.i_membrane = torch.zeros_like(model.v).detach()
