from collections import deque
from typing import List, Tuple

import networkx as nx
import torch

from .core import Integrator
from .triton import dhs_bt_solve_cuda

try:
    import axonml_solvers  # noqa: F401

    AXONML_SOLVERS_AVAILABLE = True
except ImportError:
    AXONML_SOLVERS_AVAILABLE = False


THREADS_PER_WARP = 32


def _topo_parent_depth(G: nx.DiGraph) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    nodes = list(nx.topological_sort(G))
    idx_of = {n: i for i, n in enumerate(nodes)}
    K = len(nodes)
    parent_idx = torch.full((K,), -1, dtype=torch.int64)
    depth = torch.zeros(K, dtype=torch.int64)

    # BFS from roots to get depth (for multi-root graphs, treat each root)
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


def _harmonic_edge_g_to_parent(
    val_node: torch.Tensor, parent_idx: torch.Tensor
) -> torch.Tensor:
    """
    Given per-node positive values `val_node` (shape: (B, K, ...)),
    compute edge value to parent using series-half rule:
        R_edge = 0.5 * (R_node + R_parent)   then g = 1/R_edge
    For roots, return zeros.
    """
    B, K = val_node.shape[:2]

    parent_clamped = parent_idx.clamp_min(0)
    # Broadcast to match remaining dims
    gather_idx = parent_clamped.view(1, -1, *([1] * (val_node.dim() - 2))).expand(
        B, -1, *val_node.shape[2:]
    )
    parent_vals = val_node.gather(1, gather_idx)
    R_edge = 0.5 * (val_node + parent_vals)  # series half
    with torch.no_grad():
        # Avoid div-by-zero – assume large R⇒small g when R==0 is not physical
        mask_pos = R_edge > 0
    g = torch.zeros_like(R_edge)
    g[mask_pos] = 1.0 / R_edge[mask_pos]
    # zero out roots
    is_root = parent_idx < 0
    if is_root.any():
        root_mask = is_root.view(1, -1, *([1] * (g.dim() - 2))).expand(
            B, -1, *g.shape[2:]
        )
        g[root_mask] = 0.0
    return g


def assemble_rhs(v_prev, c_rad, d, xg, e_ext):
    """
    Copy of helper from implicit._bwd_euler_bt for building RHS of the block system.
    v_prev : (B,K,M) previous [vi, ve0, ..., ve_{M-1}]
    c_rad  : (B,K,M) radial capacitances scaled by 1/dt  [cm^2*µF / s], first entry is cm_dt
    d      : (B,K)   ionic linear term  (gtot*v - itot)*area  (+ intra)
    xg     : (B,K,M-1) radial conductances (membrane & shells) [S]
    e_ext  : (B,K) or None  driving field for outermost shell (bath)
    """
    rhs = torch.zeros_like(v_prev)
    # flows between consecutive shells (including vi↔ve0)
    v_c = c_rad[:, :, :-1] * (v_prev[:, :, :-1] - v_prev[:, :, 1:])
    rhs[:, :, :-1] += v_c
    rhs[:, :, 1:] -= v_c

    # ionic current acts as +d on vi and -d on ve0 (matches vi - ve0 potential diff)
    rhs[:, :, 0] += d
    rhs[:, :, 1] -= d

    if e_ext is not None:
        rhs[:, :, -1] += xg[:, :, -1] * e_ext + c_rad[:, :, -1] * v_prev[:, :, -1]
    else:
        rhs[:, :, -1] += c_rad[:, :, -1] * v_prev[:, :, -1]

    return rhs


class _dhs_bt(Integrator):
    """
    DHS on trees with *block* unknowns per compartment (vi, ve0, ve1).

    This combines the scalar DHS solver structure with the 3×3 block assembly
    used by the unbranched block-tridiagonal integrator.
    """

    def __init__(self, model, mech, imem=None, threads=16):
        assert THREADS_PER_WARP % threads == 0, (
            f"threads must divide THREADS_PER_WARP ({THREADS_PER_WARP})"
        )
        assert threads <= THREADS_PER_WARP, (
            f"threads must be less than or equal to THREADS_PER_WARP ({THREADS_PER_WARP})"
        )
        super().__init__(model, mech, imem)
        self.threads = threads

        _, K = model.np, model.nc
        M = getattr(model, "n_layers", 2) + 1  # include vi
        if M != 3:
            raise NotImplementedError(
                "Current DHS block implementation supports exactly M=3 (vi, ve0, ve1)."
            )

        # Topology tables
        self.register_buffer("parent_idx", torch.empty(K, dtype=torch.int64))
        self.register_buffer("order", torch.empty(K, dtype=torch.int64))
        self.register_buffer("layer_ptr", torch.empty(1, dtype=torch.int64))
        self.register_buffer("solver_order", torch.empty(K, dtype=torch.int64))
        self.register_buffer("inv_solver_order", torch.empty(K, dtype=torch.int64))

        # Geometry-dependent buffers
        self.register_buffer("area", torch.empty(1, K))  # (1,K)
        self.register_buffer("cm_dt", torch.empty(1, K))  # (1,K)
        self.register_buffer("xc_dt", torch.empty(1, K, M - 1))  # (1,K,2)
        self.register_buffer("xg", torch.empty(1, K, M - 1))  # (1,K,2)

        # Block matrix and edge conductances → parent
        self.register_buffer("main_blocks", torch.empty(1, K, M, M))
        self.register_buffer(
            "g_to_parent", torch.empty(1, K, M)
        )  # diag per edge (0 at roots)

    def initialize(self, model, dt):
        dev, dtyp = model.device(), model.dtype()
        self.to(dev)

        if dev.type == "cpu" and not AXONML_SOLVERS_AVAILABLE:
            raise ImportError("DHS block solver on CPU requires 'axonml_solvers'.")

        # --- Topology ---
        parent_idx, depth, node_order = _topo_parent_depth(model.graph)
        order, layer_ptr = _build_layers(depth.to(torch.int32), self.threads)

        self.solver_order.copy_(
            torch.as_tensor(node_order, dtype=torch.int64, device=dev)
        )
        self.inv_solver_order.copy_(torch.argsort(self.solver_order, dim=0))

        self.parent_idx.copy_(parent_idx.to(device=dev, dtype=torch.int64))
        self.order.copy_(order.to(device=dev, dtype=torch.int64))
        self.layer_ptr.copy_(layer_ptr.to(device=dev, dtype=torch.int64))

        B, K = model.np, model.nc
        dt_s = dt * 1e-3  # ms→s

        # --- Radial / capacitance terms ---
        # Area (cm²) per compartment is expected to be provided (as in scalar DHS)
        area = model.area  # (1,K) or (B,K) – broadcast OK
        if area.dim() == 2 and area.shape[0] == 1:
            area_cm2 = area
        else:
            area_cm2 = area  # assume already broadcasted

        Cm = 1e-6 * model.cm * area_cm2  # F
        cm_dt = Cm / dt_s  # F/s
        xc_dt = 1e-6 * model.xc * area_cm2.unsqueeze(-1) / dt_s  # F/s, (B,K,M-1)
        xg = model.xg * area_cm2.unsqueeze(-1)  # S, (B,K,M-1)

        # --- Assemble per-node 3×3 radial block (WITHOUT axial degree terms) ---
        # Unknowns per node: [vi, ve0, ve1]
        Bsz = B
        M = 3
        main = torch.zeros(Bsz, K, M, M, device=dev, dtype=dtyp)

        # Diagonals
        # vi
        main[..., 0, 0] = cm_dt
        # ve0: membrane + shell-0
        main[..., 1, 1] = cm_dt + xc_dt[..., 0] + xg[..., 0]
        # ve1: shell-0 inward + shell-1 outward
        main[..., 2, 2] = xc_dt[..., 0] + xg[..., 0] + xc_dt[..., 1] + xg[..., 1]

        # Off-diagonals (radial couplings)
        # vi <-> ve0
        main[..., 0, 1] = -cm_dt
        main[..., 1, 0] = -cm_dt
        # ve0 <-> ve1
        coup01 = -(xc_dt[..., 0] + xg[..., 0])
        main[..., 1, 2] = coup01
        main[..., 2, 1] = coup01

        # --- Edge conductances to parent per component ---
        # (0) intracellular uses geometry-based axial conductance
        from .tree import graph_to_parent_and_axial  # reuse tested util

        parent_idx_t, a_geom_t, _ = graph_to_parent_and_axial(
            model.graph, dtype_axial=dtyp
        )

        g0_to_parent = a_geom_t.expand(Bsz, -1).to(device=dev, dtype=dtyp)  # (B,K)

        # (1,2) extracellular shells – use harmonic mean of per-node axial *resistance* to parent
        # R_node = xraxial * L (Ω). Try to infer L: prefer model.dx (µm); fall back to area/(π*diam).
        if hasattr(model, "dx"):
            L_cm = 1e-4 * model.dx  # (B,K)
        elif hasattr(model, "diam"):
            # L = area / (π * diam)
            diam_cm = 1e-4 * model.diam
            L_cm = area_cm2 / (torch.pi * diam_cm).clamp_min(1e-12)
        else:
            raise AttributeError(
                "Need model.dx or model.diam to compute extracellular axial resistances."
            )

        # node-level resistance per layer
        R_layers = model.xraxial * L_cm.unsqueeze(-1) * 1e6  # Ω, (B,K,2)
        # Edge conductance to parent
        g_layers_to_parent = _harmonic_edge_g_to_parent(
            R_layers, self.parent_idx
        )  # (B,K,2) after conversion inside

        # Combine to (B,K,3)
        g_to_parent = torch.zeros(Bsz, K, 3, device=dev, dtype=dtyp)
        g_to_parent[..., 0] = g0_to_parent
        g_to_parent[..., 1:] = g_layers_to_parent

        # Save buffers (broadcast B if necessary)
        self.area = area_cm2
        self.cm_dt = cm_dt
        self.xc_dt = xc_dt
        self.xg = xg
        self.main_blocks = main
        self.g_to_parent = g_to_parent

        # Allocate model state vectors (block unknowns)
        if not hasattr(model, "vc"):
            model.register_buffer("vc", torch.zeros(B, K, 3, device=dev, dtype=dtyp))
            model.vc[..., 0] = model.v_init
            model.vc[..., 1] = 0.0
            model.vc[..., 2] = 0.0
        if not hasattr(model, "v"):
            model.register_buffer("v", torch.zeros(B, K, device=dev, dtype=dtyp))
            model.v[:] = model.v_init

    def step(self, model, dt, ve=None, intra=None):
        model.vc, model.v = self._step(model.vc, model.v, dt, model.celsius, ve, intra)

    def _step(self, vc, v, dt, temp, ve=None, intra=None):
        # Update mechanisms (ionic) on the *membrane* voltage
        v = self.mech.update_v(v, dt)
        self.mech.advance(v, dt, temp)
        itot, gtot = self.mech.i(v)  # (B,K)

        # Ionic linear term scaled to currents [A] via area
        d_lin = (gtot * v - itot) * self.area  # (B,K)
        if intra is not None:
            d_lin = d_lin + intra

        # Assemble RHS in Mechanism/Model order first
        c_rad = torch.cat([self.cm_dt.unsqueeze(-1), self.xc_dt], dim=-1)  # (B,K,3)
        rhs_mech = assemble_rhs(vc, c_rad, d_lin, self.xg, ve)  # (B,K,3)

        # Reorder *all per-node arrays* into solver order expected by DHS
        idx = self.solver_order
        rhs_ = rhs_mech.index_select(1, idx)  # (B,K,3)
        D_ = self.main_blocks.index_select(1, idx)  # (B,K,3,3)
        G_ = self.g_to_parent.index_select(1, idx)  # (B,K,3)

        # Solve in solver order

        # Incorporate ionic conductance gtot into the per-node 3×3 blocks
        # (radial membrane term coupling vi ↔ ve0), exactly like unbranched BT:
        #   B[...,0,0] += g; B[...,1,1] += g; B[...,0,1] -= g; B[...,1,0] -= g
        g = gtot * self.area  # (B,K)
        Dm = D_.clone()
        Dm[..., 0, 0] += g
        Dm[..., 1, 1] += g
        Dm[..., 0, 1] -= g
        Dm[..., 1, 0] -= g

        # Solve in solver order
        X_ = dhs_bt_solve_cuda(
            Dm,
            G_,
            rhs_,
            self.parent_idx,
            self.order,
            self.layer_ptr,
            threads=self.threads,
        )  # (B,K,3) in solver order

        # Bring solution back to Mechanism order
        inv = self.inv_solver_order
        vc_out = X_.index_select(1, inv)  # (B,K,3)
        v_out = vc_out[..., 0] - vc_out[..., 1]  # (B,K)
        return vc_out, v_out
