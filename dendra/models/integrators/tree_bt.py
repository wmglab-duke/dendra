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
from .implicit import assemble_rhs, assemble_rhs_into
from .tree import (
    _compiled_tree_graph_view,
    _validate_dhs_threads,
    _validate_tree_graph,
    graph_to_parent_and_axial,
)
from .triton import dhs_bt_solve_cuda

try:
    import dendra_solvers  # noqa:F401

    DENDRA_SOLVERS_AVAILABLE = True
except ImportError:
    DENDRA_SOLVERS_AVAILABLE = False


# ---------------- Topology helpers (local, to avoid extra deps) ----------------
def _topo_parent_depth(G: nx.DiGraph) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    _validate_tree_graph(G)
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


def _tree_layered_edge_conductance(
    xraxial: torch.Tensor,
    dx: torch.Tensor,
    edge_child_orig: torch.Tensor,
    edge_parent_orig: torch.Tensor,
) -> torch.Tensor:
    """Return compact extracellular-shell edge conductance in storage order.

    ``xraxial`` has units MOhm/cm and ``dx`` has units micrometres.  Each edge
    joins the centres of its parent and child compartments, so its resistance is
    the sum of the two half-segment resistances.  Invalid/non-positive edge
    resistance retains the historical sealed-edge conductance of zero.

    This helper is deliberately tensor-only: both imperative initialization and
    functional topology lowering can use the same differentiable expression.
    """
    child_dx = dx.index_select(-1, edge_child_orig)
    parent_dx = dx.index_select(-1, edge_parent_orig)
    child_xraxial = xraxial.index_select(-2, edge_child_orig)
    parent_xraxial = xraxial.index_select(-2, edge_parent_orig)
    resistance = (
        0.5
        * (
            child_xraxial * child_dx.unsqueeze(-1)
            + parent_xraxial * parent_dx.unsqueeze(-1)
        )
        * 1.0e2
    )
    valid = torch.isfinite(resistance) & (resistance > 0)
    return torch.where(valid, resistance.reciprocal(), torch.zeros_like(resistance))


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
    _PREPARED_WORKSPACE_SCHEMA = (
        ("area", "node"),
        ("cm_dt", "node"),
        ("xc_dt", "shell_node"),
        ("c_rad", "block_node"),
        ("xg", "shell_node"),
        ("main_blocks", "block_matrix"),
        ("g_to_parent", "block_node"),
    )
    _FUNCTIONAL_OPERATOR_KIND = "block_tree"
    _FUNCTIONAL_CRITICAL_METHODS = ("_select_solver",)

    def __init__(self, model, mech, imem=None, threads=16):
        threads = _validate_dhs_threads(threads)
        super().__init__(model, mech, imem)
        self.threads = threads

        B, K = _model_solve_shape(model)
        M = int(getattr(model, "n_layers", 2)) + 1
        if M != 3:
            raise ValueError(
                "_dhs_bt currently requires exactly 3 unknowns per compartment "
                f"(model.n_layers + 1 == 3); got {M}."
            )

        self.B = B
        self.K = K
        self.M = M

        self.register_buffer("parent_idx", torch.empty(K, dtype=torch.int64))
        self.register_buffer("order", torch.empty(K, dtype=torch.int64))
        self.register_buffer("layer_ptr", torch.empty(1, dtype=torch.int64))
        self.register_buffer("solver_order", torch.empty(K, dtype=torch.int64))
        self.register_buffer("inv_solver_order", torch.empty(K, dtype=torch.int64))
        self.register_buffer("parent_idx_orig", torch.empty(K, dtype=torch.int64))
        self.register_buffer(
            "edge_child_orig", torch.empty(max(K - 1, 0), dtype=torch.int64)
        )
        self.register_buffer(
            "edge_parent_orig", torch.empty(max(K - 1, 0), dtype=torch.int64)
        )

        # Geometry-dependent buffers
        self.register_buffer("area", torch.empty(B, K))
        self.register_buffer("cm_dt", torch.empty(B, K))
        self.register_buffer("xc_dt", torch.empty(B, K, 2))
        self.register_buffer("c_rad", torch.empty(B, K, M))
        self.register_buffer("xg", torch.empty(B, K, 2))

        # Per-node radial block (no axial degree baked in)
        self.register_buffer("main_blocks", torch.empty(B, K, M, M))
        # Per-edge diag g to parent (0:vi, 1:ve0, 2:ve1), 0 for roots
        self.register_buffer("g_to_parent", torch.empty(B, K, M))

    def _select_solver(self, device):
        device = torch.device(device)
        if device.type == "cuda":
            self._solve = partial(dhs_bt_solve_cuda, threads=self.threads)
        elif device.type == "cpu":
            if not DENDRA_SOLVERS_AVAILABLE:
                raise ImportError(
                    "DHS_BT integrator requires the dendra-solvers package for CPU "
                    "execution. Install it with "
                    "`python -m pip install --upgrade "
                    '--only-binary=dendra-solvers "dendra-solvers>=0.3.1"`.'
                )
            self._solve = torch.ops.dendra_solvers.dhs_bt_solve
        else:
            raise NotImplementedError(f"Unsupported device type: {device.type}")

    def _functional_solver(self):
        """Return the transform-compatible native CPU block-DHS facade.

        Ordinary execution retains direct dispatcher access.  The public
        dendra-solvers facade owns the JVP and nested ``torch.func`` contracts
        required by functional ExtCellTree lowering.
        """
        solver = self._solve
        solver_name = getattr(solver, "__name__", None)
        solver_module = getattr(solver, "__module__", None)
        if (
            solver_name == "dhs_bt_solve"
            and solver_module == "torch._ops.dendra_solvers"
        ):
            facade = (
                None
                if not DENDRA_SOLVERS_AVAILABLE
                else getattr(dendra_solvers, "dhs_bt_solve", None)
            )
            if callable(facade):
                return facade
            raise RuntimeError(
                "The selected native CPU solver requires a dendra-solvers build "
                "that exports the torch.func-compatible dhs_bt_solve facade. "
                "Upgrade it with `python -m pip install --upgrade "
                '--only-binary=dendra-solvers "dendra-solvers>=0.3.1"`.'
            )
        raise RuntimeError(
            "The selected block-DHS solver is not yet transform-compatible. "
            "Functional ExtCellTree execution currently requires CPU "
            "dendra_solvers.dhs_bt_solve."
        )

    @staticmethod
    def _prepare_workspace(
        dt,
        *,
        cm,
        area,
        intracellular_edge_conductance,
        extracellular_edge_conductance,
        xc,
        xg,
        edge_child_orig,
        solver_order,
    ):
        """Purely derive the two-layer block-tree implicit-Euler workspace.

        Node tensors use flattened original/model storage order.  Both edge
        conductance tensors use compact original order (increasing child storage
        index); topology tensors place them into child-indexed node planes and
        then gather solver-owned matrices into DHS order.  ``cm``/``xc`` are in
        microfarads per square centimetre, ``xg`` in siemens per square
        centimetre, ``area`` in square centimetres, and edge inputs in siemens.
        """
        if torch.is_tensor(dt):
            dt_s = dt.reshape(1) * 1.0e-3
        else:
            dt_s = dt * 1.0e-3

        cm_dt = 1.0e-6 * cm * area / dt_s
        xc_dt = 1.0e-6 * xc * area.unsqueeze(-1) / dt_s
        radial_g = xg * area.unsqueeze(-1)

        # Historical zero-volume branch points carry no radial membrane terms.
        area_zero = area == 0
        cm_dt = torch.where(area_zero, torch.zeros_like(cm_dt), cm_dt)
        xc_dt = torch.where(area_zero.unsqueeze(-1), torch.zeros_like(xc_dt), xc_dt)
        radial_g = torch.where(
            area_zero.unsqueeze(-1), torch.zeros_like(radial_g), radial_g
        )

        shell_0 = xc_dt[..., 0] + radial_g[..., 0]
        zero = torch.zeros_like(cm_dt)
        radial_main = torch.stack(
            (
                torch.stack((cm_dt, -cm_dt, zero), dim=-1),
                torch.stack((-cm_dt, cm_dt + shell_0, -shell_0), dim=-1),
                torch.stack(
                    (
                        zero,
                        -shell_0,
                        shell_0 + xc_dt[..., 1] + radial_g[..., 1],
                    ),
                    dim=-1,
                ),
            ),
            dim=-2,
        )

        edge_index = edge_child_orig.reshape(
            (1,) * (intracellular_edge_conductance.ndim - 1) + (-1,)
        ).expand_as(intracellular_edge_conductance)
        intracellular_node = torch.zeros_like(area).scatter(
            -1,
            edge_index,
            intracellular_edge_conductance,
        )
        shell_edge_index = edge_child_orig.reshape(
            (1,) * (extracellular_edge_conductance.ndim - 2) + (-1, 1)
        ).expand_as(extracellular_edge_conductance)
        extracellular_node = torch.zeros_like(xc).scatter(
            -2,
            shell_edge_index,
            extracellular_edge_conductance,
        )
        g_to_parent = torch.cat(
            (intracellular_node.unsqueeze(-1), extracellular_node), dim=-1
        ).index_select(-2, solver_order)

        return {
            "area": area.clone(memory_format=torch.preserve_format),
            "cm_dt": cm_dt,
            "xc_dt": xc_dt,
            "c_rad": torch.cat((cm_dt.unsqueeze(-1), xc_dt), dim=-1),
            "xg": radial_g,
            "main_blocks": radial_main.index_select(-3, solver_order),
            "g_to_parent": g_to_parent,
        }

    def initialize(self, model, dt):
        dev, dtyp = model.device(), model.dtype()
        self.to(dev)

        self._select_solver(dev)
        self._refresh_solver_shape(model, block_dim=self.M)
        B = self.B
        K = self.K

        # Topology and axial geometry come from the same immutable compiled
        # morphology snapshot as scalar DHS and material transport. The public
        # NetworkX graph is an interoperability view, not a recompilation API.
        graph = _compiled_tree_graph_view(model)
        if isinstance(graph, list):
            if len(graph) != 1:
                raise ValueError(
                    "ExtCellTree block-DHS requires one shared compiled morphology."
                )
            graph = graph[0]
        parent_idx, depth, node_order = _topo_parent_depth(graph)
        order, layer_ptr = _build_layers(depth.to(torch.int32), self.threads)
        self.solver_order.copy_(
            torch.as_tensor(node_order, dtype=torch.int64, device=dev)
        )
        self.inv_solver_order.copy_(torch.argsort(self.solver_order, dim=0))
        self.parent_idx.copy_(parent_idx.to(device=dev, dtype=torch.int64))
        self.order.copy_(order.to(device=dev, dtype=torch.int64))
        self.layer_ptr = layer_ptr.to(device=dev, dtype=torch.int64)

        # Retain explicit storage-order topology for pure preparation and future
        # functional extraction. ``parent_idx`` itself remains in DHS order.
        parent_orig = torch.full((K,), -1, dtype=torch.int64, device=dev)
        non_root_solver = self.parent_idx >= 0
        child_orig = self.solver_order[non_root_solver]
        parent_orig[child_orig] = self.solver_order[self.parent_idx[non_root_solver]]
        edge_child_orig = torch.arange(K, dtype=torch.int64, device=dev)[
            parent_orig >= 0
        ]
        edge_parent_orig = parent_orig.index_select(0, edge_child_orig)
        self.parent_idx_orig.copy_(parent_orig)
        self.edge_child_orig.copy_(edge_child_orig)
        self.edge_parent_orig.copy_(edge_parent_orig)

        rhoa_scale = _as_solve_matrix(model.rhoa_scale, model)
        canonical_resistance = getattr(model, "edge_resistance_ohm", None)
        if canonical_resistance is not None:
            resistance = _as_solve_matrix(canonical_resistance, model).index_select(
                1, edge_child_orig
            )
            intracellular_edge_conductance = (
                resistance * rhoa_scale.index_select(1, edge_child_orig)
            ).reciprocal()
        else:
            # Legacy/lightweight Tree-like models have no immutable compiled
            # resistance tensor. Preserve their graph fallback while applying
            # the same runtime rhoa_scale contract as canonical ExtCellTree.
            _, graph_conductance, _ = graph_to_parent_and_axial(graph, dtype_axial=dtyp)
            graph_conductance = graph_conductance.to(device=dev, dtype=dtyp).expand(
                B, -1
            )
            mechanism_conductance = graph_conductance.index_select(
                1, self.inv_solver_order
            )
            intracellular_edge_conductance = mechanism_conductance.index_select(
                1, edge_child_orig
            ) / rhoa_scale.index_select(1, edge_child_orig)

        dx = _as_solve_matrix(model.dx, model)
        xraxial = _as_solve_block(model.xraxial, model, (self.M - 1,))
        extracellular_edge_conductance = _tree_layered_edge_conductance(
            xraxial,
            dx,
            edge_child_orig,
            edge_parent_orig,
        )
        area = _as_solve_matrix(model.area, model) * _as_solve_matrix(
            model.area_scale, model
        )
        cm = _as_solve_matrix(model.cm, model) * _as_solve_matrix(model.cm_scale, model)
        workspace = self._derive_prepared_workspace(
            dt,
            cm=cm,
            area=area,
            intracellular_edge_conductance=intracellular_edge_conductance,
            extracellular_edge_conductance=extracellular_edge_conductance,
            xc=_as_solve_block(model.xc, model, (self.M - 1,)),
            xg=_as_solve_block(model.xg, model, (self.M - 1,)),
            edge_child_orig=edge_child_orig,
            solver_order=self.solver_order,
        )
        self._install_prepared_workspace(workspace)

        # Initializing solver geometry must not reset a live simulation.  A
        # missing or shape-stale block state is seeded from the current
        # membrane voltage; explicit resets remain the job of ``init_v``.
        expected_vc_shape = tuple(model.shape) + (self.M,)
        if not hasattr(model, "vc") or tuple(model.vc.shape) != expected_vc_shape:
            vc = torch.zeros(expected_vc_shape, device=dev, dtype=dtyp)
            vc[..., 0] = model.v.to(device=dev, dtype=dtyp)
            if "vc" in getattr(model, "_buffers", {}):
                model.vc = vc
            else:
                model.register_buffer("vc", vc)

    def step(self, model, dt, ve=None, intra=None):
        vc_new, v_new, i_membrane = self._call_kernel(
            "_step",
            self._flat_block_voltage(model.vc, self.M),
            model.v,
            dt,
            model.celsius,
            ve,
            intra,
        )
        if self.imem:
            model.i_membrane = i_membrane
        model.vc = vc_new
        model.v = v_new

    def _step(
        self,
        vc,
        v,
        dt,
        temp,
        ve=None,
        intra=None,
        *,
        solver=None,
        call_local_currents=False,
    ):
        # Update mechanisms in mV / mA/cm^2
        v_state = self.mech.update_v(v)
        self._advance_pre_current(v_state, dt, temp)
        if call_local_currents:
            evaluate = getattr(self.mech, "_evaluate_current_frame", None)
            if evaluate is None:
                raise RuntimeError(
                    "call-local current evaluation requires a MechanismHandler "
                    "with _evaluate_current_frame()"
                )
            (
                itot,
                gtot,
                ion_current_frame,
                ion_conductance_frame,
            ) = evaluate(v_state)
        else:
            itot, gtot = self.mech.i(v_state)
            ion_current_frame = self._capture_ion_current_frame()
            ion_conductance_frame = self._capture_ion_conductance_frame()

        # RHS (mechanism order), keep everything in mV/mA/S:
        # d_lin = (g*v - itot) * area  [mA]
        public_voltage_shape = self.base_shape[:-1]
        if call_local_currents:
            v_flat = _flatten_to_solve(v_state, self.K, public_voltage_shape)
            gtot_flat = _flatten_to_solve(gtot, self.K, public_voltage_shape)
            itot_flat = _flatten_to_solve(itot, self.K, public_voltage_shape)
        else:
            v_flat = self._flat_voltage(v_state)
            gtot_flat = self._flat_voltage(gtot)
            itot_flat = self._flat_voltage(itot)
        d_lin = (gtot_flat * v_flat - itot_flat) * self.area
        if intra is not None:
            d_lin = d_lin + _flatten_to_solve(
                intra,
                self.K,
                public_voltage_shape if call_local_currents else None,
            )  # assume intra is already in mA

        ve_flat = (
            _flatten_to_solve(
                ve,
                self.K,
                public_voltage_shape if call_local_currents else None,
            )
            if ve is not None
            else None
        )
        xg_outer = self.xg[..., -1]
        if call_local_currents:
            # Shared out-of-place block assembly is safe when any operand owns
            # a hidden torch.func lane, including an empty lane batch.
            rhs_mech = assemble_rhs(
                vc,
                self.c_rad,
                d_lin,
                xg_outer,
                ve_flat,
            )
        else:
            # Keep the ordinary allocation-efficient fill path. ``rhs_mech`` is
            # fresh disposable storage and never aliases model carry.
            rhs_mech = assemble_rhs_into(
                torch.empty_like(vc),
                vc,
                self.c_rad,
                d_lin,
                xg_outer,
                ve_flat,
            )

        # Reorder into solver order
        idx = self.solver_order
        rhs_ = rhs_mech.index_select(1, idx)  # (B,K,3)
        G_ = self.g_to_parent

        # Inject membrane gtot (scaled by area) into [vi, ve0] block (solver order)
        g_mech = gtot_flat * self.area  # (B,K) S (mechanism order)
        g_ = g_mech.index_select(1, idx)  # (B,K) S (solver order)

        if call_local_currents:
            zero = torch.zeros_like(g_)
            ionic_blocks = torch.stack(
                (
                    torch.stack((g_, -g_, zero), dim=-1),
                    torch.stack((-g_, g_, zero), dim=-1),
                    torch.stack((zero, zero, zero), dim=-1),
                ),
                dim=-2,
            )
            Dm = self.main_blocks + ionic_blocks
        else:
            Dm = self.main_blocks.clone()  # (B, K, 3, 3)
            Dm[..., 0, 0] += g_
            Dm[..., 1, 1] += g_
            Dm[..., 0, 1] -= g_
            Dm[..., 1, 0] -= g_

        # Solve in solver order (kernel expects S @ mV = mA)
        solve = self._solve if solver is None else solver
        X_ = solve(
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

        i_membrane = None
        if self.imem:
            if call_local_currents:
                v_new = _flatten_to_solve(v_out, self.K, public_voltage_shape)
            else:
                v_new = self._flat_voltage(v_out)
            v_old = v_flat
            g_abs = gtot_flat * self.area
            i_abs_old = itot_flat * self.area
            i_membrane = ((self.cm_dt + g_abs) * (v_new - v_old) + i_abs_old).reshape(
                self.shape
            )

        accepted_frame = self._linearize_ion_current_frame(
            ion_current_frame,
            ion_conductance_frame,
            v_out - v_state,
        )
        self._advance_post_current(v_state, dt, temp, accepted_frame)

        return vc_out, v_out, i_membrane

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
