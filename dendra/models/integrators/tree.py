from collections import deque
from dataclasses import dataclass
from functools import partial
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch

from ..graph import share_topology_labeled
from .cable import unbranched_edge_conductance
from .core import (
    Integrator,
    MultiIntegrator,
    _as_solve_matrix,
    _broadcast_to_shape,
    _flatten_to_solve,
)
from .triton import dhs_solve_cuda, dhs_solve_multi_cuda

try:
    import dendra_solvers

    DENDRA_SOLVERS_AVAILABLE = True

    def _dhs_multi_solve_cpu(
        d_mem,
        a_geom,
        b,
        P_cat,
        ORDER_cat,
        LAYER_PTR_cat,
        WARP_P_OFF,
        WARP_ORDER_OFF,
        WARP_LPTR_OFF,
        WARP_L,
        WARP_ROW_BASE,
        WARP_ROW_COUNT,
        K_stride: int,
        L_max: int,
        grid_x: int = None,
    ):
        return torch.ops.dendra_solvers.dhs_multi_solve(
            d_mem,
            a_geom,
            b,
            P_cat,
            ORDER_cat,
            LAYER_PTR_cat,
            WARP_P_OFF,
            WARP_ORDER_OFF,
            WARP_LPTR_OFF,
            WARP_L,
            WARP_ROW_BASE,
            WARP_ROW_COUNT,
        )

except ImportError:
    dendra_solvers = None
    _dhs_multi_solve_cpu = None
    DENDRA_SOLVERS_AVAILABLE = False


THREADS_PER_WARP = 32


@dataclass(frozen=True)
class _ScalarDHSGroup:
    """One component's topology and geometry in packed DHS solver order."""

    B: int
    K: int
    parent_idx: torch.Tensor
    solver_order: torch.Tensor
    a_geom: torch.Tensor


def _compiled_tree_graph_view(model):
    """Return the morphology snapshot used by Tree numerical solvers.

    ``Tree.graph`` is retained as a mutable NetworkX interoperability view.
    Factories also retain an immutable CompartmentGraph; prefer that compiled
    snapshot so mutating the view cannot split electrical and material
    topology. Low-level/legacy Tree subclasses without a snapshot keep the
    historical graph/assemble_graphs fallback.
    """
    canonical = getattr(model, "compartment_graph", None)
    if canonical is not None:
        return canonical.to_networkx()
    compiled = getattr(model, "_compiled_graph", None)
    if compiled is not None:
        return compiled
    graph = model.graph
    if graph is None:
        try:
            graph = model.assemble_graphs()
        except Exception as err:
            raise ValueError(
                "Model must expose compiled compartment_graph, `graph`, or "
                "implement `assemble_graphs()` returning morphology graphs."
            ) from err
    return graph


def _scalar_group_planes(value, model, P, B, K, name, *, device):
    """Broadcast a scalar-group field to ``(P, B, K)`` without detaching it."""
    value = value.to(device=device, dtype=model.dtype())
    try:
        value = torch.broadcast_to(value, tuple(model.shape))
    except RuntimeError as err:
        raise ValueError(
            f"{name} for a scalar DHS group must broadcast to model shape "
            f"{tuple(model.shape)}; got {tuple(value.shape)}."
        ) from err
    return value.reshape(P, B, K)


def _scalar_dhs_group(model, P: int, device: torch.device) -> _ScalarDHSGroup:
    """Adapt a supported scalar population to the universal DHS row contract.

    Ordinary ``Population(N, C)`` instances are represented by ``N * C``
    independent one-node trees.  Tree models retain their imported rooted
    topology. Cable models use an identity-ordered path and obtain axial
    conductance from exact canonical edges when available, otherwise from the
    specialized Axon tensor geometry.
    """
    # Lazy imports avoid the core -> integrators -> core import cycle.
    from ..core import Cable, Population
    from ..multi import MultiPopulation
    from ..tree import Tree

    dtype = model.dtype()

    if isinstance(model, Tree):
        B = int(model.shape[-2])
        K = int(model.shape[-1])
        graph = _compiled_tree_graph_view(model)
        graphs = graph if isinstance(graph, list) else [graph]
        if len(graphs) not in (1, B):
            raise ValueError(
                "A scalar Tree group must provide one shared morphology graph "
                f"or one graph per neuron ({B}); got {len(graphs)}."
            )
        parent_idx, a_geom, node_order = graph_to_parent_and_axial(
            graphs, dtype_axial=dtype
        )
        solver_order = torch.as_tensor(node_order, dtype=torch.int64, device=device)
        a_geom = a_geom.to(device=device, dtype=dtype).expand(B, -1)
        rhoa_scale = _scalar_group_planes(
            model.rhoa_scale,
            model,
            P,
            B,
            K,
            "rhoa_scale",
            device=device,
        ).index_select(-1, solver_order)
        return _ScalarDHSGroup(
            B=B,
            K=K,
            parent_idx=parent_idx.to(dtype=torch.int64, device=device),
            solver_order=solver_order,
            a_geom=a_geom.unsqueeze(0).expand(P, -1, -1) / rhoa_scale,
        )

    if isinstance(model, Cable):
        B = int(model.shape[-2])
        K = int(model.shape[-1])
        solver_order = torch.arange(K, dtype=torch.int64, device=device)
        parent_idx = solver_order - 1
        edge_conductance = unbranched_edge_conductance(model).reshape(P, B, K - 1)
        if K == 1:
            a_geom = torch.zeros(P, B, 1, dtype=dtype, device=device)
        else:
            root = torch.zeros(P, B, 1, dtype=dtype, device=device)
            a_geom = torch.cat((root, edge_conductance), dim=-1)
        return _ScalarDHSGroup(
            B=B,
            K=K,
            parent_idx=parent_idx,
            solver_order=solver_order,
            a_geom=a_geom,
        )

    if isinstance(model, Population) and not isinstance(model, MultiPopulation):
        B = int(np.prod(model.core_shape()))
        K = 1
        return _ScalarDHSGroup(
            B=B,
            K=K,
            parent_idx=torch.tensor([-1], dtype=torch.int64, device=device),
            solver_order=torch.zeros(1, dtype=torch.int64, device=device),
            a_geom=torch.zeros(P, B, K, dtype=dtype, device=device),
        )

    raise TypeError(
        f"{type(model).__name__} does not expose a supported scalar DHS "
        "topology; expected ordinary Population, Tree, or scalar Cable."
    )


def _validate_dhs_threads(threads: int) -> int:
    """Validate the lane count accepted by the CPU and CUDA DHS kernels."""
    if isinstance(threads, bool) or not isinstance(threads, int):
        raise TypeError("threads must be a positive integer that divides 32")
    if threads <= 0 or threads > THREADS_PER_WARP:
        raise ValueError("threads must be in [1, 32]")
    if THREADS_PER_WARP % threads != 0:
        raise ValueError("threads must divide 32 (warp size)")
    return threads


def _validate_tree_graph(graph: nx.DiGraph) -> None:
    """Reject graph shapes that the Hines/DHS parent representation cannot encode."""
    if not isinstance(graph, nx.DiGraph) or graph.is_multigraph():
        raise TypeError("morphology graph must be a networkx.DiGraph")
    if graph.number_of_nodes() == 0:
        raise ValueError("morphology graph must contain at least one node")
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("morphology graph must be acyclic")

    for node, degree in graph.in_degree():
        if degree > 1:
            raise ValueError(
                f"Node {node} has {degree} parents; morphology must be a rooted tree."
            )

    roots = [node for node, degree in graph.in_degree() if degree == 0]
    if len(roots) != 1:
        raise ValueError(
            f"morphology graph must have exactly one root; found {len(roots)}"
        )

    root = roots[0]
    if len(nx.descendants(graph, root)) + 1 != graph.number_of_nodes():
        raise ValueError("all morphology nodes must be reachable from the root")

    expected_nodes = set(range(graph.number_of_nodes()))
    if set(graph.nodes()) != expected_nodes:
        raise ValueError(
            "morphology node labels must be consecutive integers from 0 to K - 1"
        )


def build_morphology(
    parent_idx: List[int],
) -> Tuple[
    torch.Tensor,  # parent (K,)
    List[List[int]],  # children adjacency
    torch.Tensor,  # depth (K,)
]:
    """
    Parameters
    ----------
    parent_idx : length-K list with soma parent = -1.

    Returns
    -------
    parent      int32[K]  : constant on GPU, maps each compartment to its parent
    children    list[K]   : python list of child lists (only needed offline)
    depth       int32[K]  : depth of every node from the soma
    """
    K = len(parent_idx)
    if K == 0:
        raise ValueError("parent_idx must contain at least one node")

    roots = [i for i, parent in enumerate(parent_idx) if parent == -1]
    if len(roots) != 1:
        raise ValueError(
            f"parent_idx must contain exactly one root; found {len(roots)}"
        )

    children = [[] for _ in range(K)]
    root = roots[0]
    for i, p in enumerate(parent_idx):
        if p == -1:
            continue
        if isinstance(p, bool) or not isinstance(p, (int, np.integer)):
            raise TypeError(f"parent index at node {i} must be an integer; got {p!r}")
        if p < 0 or p >= K:
            raise ValueError(
                f"parent index at node {i} must be in [0, {K - 1}]; got {p}"
            )
        if p == i:
            raise ValueError(f"node {i} cannot be its own parent")
        children[p].append(i)

    depth = torch.zeros(K, dtype=torch.int32)
    q = deque([root])
    visited = set()
    while q:
        u = q.popleft()
        if u in visited:
            raise ValueError("parent_idx contains a cycle")
        visited.add(u)
        for c in children[u]:
            depth[c] = depth[u] + 1
            q.append(c)

    if len(visited) != K:
        raise ValueError(
            "parent_idx contains a cycle or nodes unreachable from the root"
        )

    return (torch.as_tensor(parent_idx, dtype=torch.int32), children, depth)


def graph_to_parent_and_axial(
    G: nx.DiGraph | list[nx.DiGraph], dtype_axial: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a compartmental morphology stored in a DiGraph into

        parent_idx : int32[K]
        a_geom     : (dtype_axial)[K]   - axial conductance (Siemens)

    Preference order for axial resistance:
        1.  Use `G.edges[parent, child]['R_ohm']`  if present (exact value
            taken from NEURON's ri()).
        2.  Otherwise compute it from the node-level geometric attributes
            (L, diam, Ra).

    The topological sort guarantees that every parent index < child index,
    matching the requirements of DHS / Hines matrix preprocessing.
    """
    graphs = [G] if isinstance(G, nx.DiGraph) else list(G)
    if not graphs:
        raise ValueError("at least one morphology graph is required")
    for graph in graphs:
        _validate_tree_graph(graph)

    same_topology, reason = share_topology_labeled(graphs)
    if not same_topology:
        raise ValueError(
            "All morphology graphs must share the same labeled topology: " + reason
        )
    # ------------------------------------------------------------------
    # 0. topological order and quick look‑ups
    # ------------------------------------------------------------------
    nodes = list(nx.topological_sort(graphs[0]))  # length K
    idx_of = {n: i for i, n in enumerate(nodes)}
    K = len(nodes)

    parent_idx = np.full(K, -1, dtype=np.int32)
    # Compute in double precision and cast only at the public tensor boundary.
    # The previous float32 staging array silently truncated float64 morphologies.
    a_geom = np.zeros((len(graphs), K), dtype=np.float64)

    # constant: 1 µm = 1 e‑4 cm
    microns_to_cm = 1e-4
    pi = np.pi

    def positive_geometry(graph, node, name):
        value = graph.nodes[node][name]
        try:
            value = float(value)
        except (TypeError, ValueError) as err:
            raise TypeError(
                f"Geometry attribute {name!r} on node {node} must be a scalar; "
                f"got {value!r}."
            ) from err
        if not np.isfinite(value):
            raise ValueError(
                f"Geometry attribute {name!r} on node {node} must be finite; "
                f"got {value!r}."
            )
        if value <= 0:
            raise ValueError(
                f"Geometry attribute {name!r} on node {node} must be positive; "
                f"got {value!r}."
            )
        return value

    # ------------------------------------------------------------------
    # 1. iterate over all nodes except the roots
    # ------------------------------------------------------------------
    for j, g in enumerate(graphs):
        for child in nodes:
            i = idx_of[child]
            preds = list(g.predecessors(child))

            if not preds:  # soma / root compartment
                continue
            if len(preds) > 1:
                raise ValueError(
                    f"Node {child} has {len(preds)} parents — "
                    "morphology must be a rooted tree for the Hines matrix."
                )

            parent = preds[0]
            p = idx_of[parent]
            parent_idx[i] = p

            # --------------------------------------------------------------
            # 1a.  Attempt to use the pre‑computed exact resistance
            # --------------------------------------------------------------
            edge_data = g.get_edge_data(parent, child, default={})
            R_total = edge_data.get("R_ohm", None)  # Ω or None

            # --------------------------------------------------------------
            # 1b.  If not present, fall back to geometric half‑segment calc
            # --------------------------------------------------------------
            if R_total is None:
                try:
                    # child geometry
                    L_i = positive_geometry(g, child, "L") * microns_to_cm  # cm
                    d_i_cm = positive_geometry(g, child, "diam") * microns_to_cm
                    r_i_cm = 0.5 * d_i_cm
                    rho_i = positive_geometry(g, child, "Ra")  # Ω·cm

                    # parent geometry
                    L_p = positive_geometry(g, parent, "L") * microns_to_cm
                    d_p_cm = positive_geometry(g, parent, "diam") * microns_to_cm
                    r_p_cm = 0.5 * d_p_cm
                    rho_p = positive_geometry(g, parent, "Ra")  # Ω·cm
                except KeyError as err:
                    raise KeyError(
                        f"Missing geometry attribute {err} on node; "
                        "cannot compute axial resistance and no R_ohm present "
                        "on the edge."
                    ) from err

                R_half_i = rho_i * (L_i / 2) / (pi * r_i_cm**2)
                R_half_p = rho_p * (L_p / 2) / (pi * r_p_cm**2)
                R_total = R_half_i + R_half_p  # Ω

            # --------------------------------------------------------------
            # 1c.  Store axial conductance  (Siemens = 1 / Ω)
            # --------------------------------------------------------------
            if not np.isfinite(R_total) or R_total <= 0:
                raise ValueError(
                    f"Axial resistance for edge ({parent}, {child}) must be "
                    f"finite and positive; got {R_total!r}."
                )
            a_geom[j, i] = 1.0 / R_total

    # ------------------------------------------------------------------
    # 2. cast to torch tensors
    # ------------------------------------------------------------------
    parent_idx_t = torch.as_tensor(parent_idx, dtype=torch.int32)
    a_geom_t = torch.as_tensor(a_geom, dtype=dtype_axial)

    return parent_idx_t, a_geom_t, nodes


def build_dhs_layers(
    depth: torch.Tensor, k_threads: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    O(K) construction of DHS order/layers.

    Parameters
    ----------
    depth      int32[K]    depth[i] = 0 for the soma
    k_threads  int         warp size, e.g. 32

    Returns
    -------
    order      int32[K]        elimination order (children before parent)
    layer_ptr  int32[L+1]      layer_ptr[m] … layer_ptr[m+1]-1  is layer m
    """
    k_threads = _validate_dhs_threads(k_threads)
    if not torch.is_tensor(depth):
        raise TypeError("depth must be a one-dimensional integer tensor")
    if depth.ndim != 1:
        raise ValueError(
            f"depth must be one-dimensional; got shape {tuple(depth.shape)}"
        )
    if depth.numel() == 0:
        raise ValueError("depth must contain at least one node")
    if depth.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError(f"depth must use an integer dtype; got {depth.dtype}")
    if bool((depth < 0).any()):
        raise ValueError("depth values must be nonnegative")
    depth_cpu = depth.cpu().numpy()
    max_d = int(depth_cpu.max())

    order: List[int] = []
    layer_ptr: List[int] = [0]

    # bucket sort by depth
    bins: List[List[int]] = [[] for _ in range(max_d + 1)]
    for i, d in enumerate(depth_cpu):
        bins[d].append(i)

    # deepest → root
    for d in range(max_d, -1, -1):
        bucket = bins[d]
        # slice bucket into chunks of at most k
        for s in range(0, len(bucket), k_threads):
            chunk = bucket[s : s + k_threads]
            order.extend(chunk)
            layer_ptr.append(len(order))

    return (
        torch.as_tensor(order, dtype=torch.int32),
        torch.as_tensor(layer_ptr, dtype=torch.int32),
    )


def _edge_currents(
    edge_child: torch.Tensor,
    edge_parent: torch.Tensor,
    edge_gax: torch.Tensor,
    ve: torch.Tensor,
):  # (B,K)
    # diff shape (B,E)
    # ``v`` is transmembrane voltage, while axial current is driven by the
    # intracellular potential ``v + ve``.  The extracellular contribution on
    # an oriented parent -> child edge is therefore g * (ve_child - ve_parent):
    # it enters the parent RHS and leaves the child RHS.
    diff = ve.index_select(1, edge_child) - ve.index_select(1, edge_parent)
    return diff * edge_gax


class _dhs(Integrator):
    r"""
    Dendritic Hierarchical Scheduling (DHS) integrator for tree morphologies.

    Preprocesses a rooted tree into elimination layers and solves the resulting
    Hines tridiagonal system with a warp-friendly DHS ordering. Supports CUDA
    and CPU backends (CPU requires ``dendra_solvers``). Extracellular coupling
    is included by accumulating edge currents into the RHS when ``ve`` is
    provided.

    Parameters
    ----------
    imem : bool or None, optional
        If truthy, accumulate membrane currents each step. Default None.
    threads : int, optional
        Threads per warp lane for DHS elimination (must divide 32). Default 16.

    Notes
    -----
    Based on Zhang et al., Nat. Commun. 14, 5798 (2023).
    """

    supports_unbranched_cable = True
    # Integrator-owned tensor workspace. Tree topology is immutable solver-plan
    # metadata; these four tensors are the parameter/timestep-dependent numerical
    # inputs shared by imperative initialization and functional preparation.
    _PREPARED_WORKSPACE_SCHEMA = (
        ("area", "node"),
        ("cm_dt", "node"),
        ("axial_conductance", "node"),
        ("edge_conductance", "edge"),
    )
    _FUNCTIONAL_OPERATOR_KIND = "scalar_tree"
    _FUNCTIONAL_CRITICAL_METHODS = ("_select_solver",)

    def __init__(self, model, mech, imem=None, threads=16):
        threads = _validate_dhs_threads(threads)

        super().__init__(model, mech, imem)
        self.threads = threads

        N = model.shape[-1]
        B = int(np.prod(model.shape[:-1]))
        self.B = B
        self.K = N
        self.base_shape = model.shape

        self.register_buffer("parent_idx", torch.empty(N, dtype=torch.int64))
        self.register_buffer(
            "order", torch.empty(N, dtype=torch.int64)
        )  # order of forward elimination

        self.register_buffer(
            "solver_order", torch.empty(N, dtype=torch.int64)
        )  # (N,) node order for the graph
        self.register_buffer(
            "inv_solver_order", torch.empty(N, dtype=torch.int64)
        )  # (N,) inverse node order
        self.register_buffer("area", torch.empty(B, N))  # (B,N) membrane area
        self.register_buffer("cm_dt", torch.empty(B, N))  # (B,N) capacitance / dt
        self.register_buffer(
            "axial_conductance", torch.empty(B, N)
        )  # (B,N) child-indexed, solver order
        self.register_buffer(
            "edge_conductance", torch.empty(B, max(N - 1, 0))
        )  # (B,E) compact mechanism order

    # Historical public/internal names remain read-only views of the canonical
    # workspace. Do not register duplicate aliases: functional_call must have one
    # unambiguous tensor slot for every prepared value.
    @property
    def scale(self):
        return self.area

    @property
    def cmdt(self):
        return self.cm_dt

    @property
    def a_geom(self):
        return self.axial_conductance

    @property
    def edge_gax_orig(self):
        return self.edge_conductance

    def _select_solver(self, device):
        device = torch.device(device)
        if device.type == "cuda":
            self._solve = partial(dhs_solve_cuda, threads=self.threads)
        elif device.type == "cpu":
            if not DENDRA_SOLVERS_AVAILABLE:
                raise ImportError(
                    "DHS integrator requires the dendra-solvers package for CPU "
                    "execution. Install it with "
                    "`python -m pip install --upgrade "
                    '--only-binary=dendra-solvers "dendra-solvers>=0.3.1"`.'
                )
            self._solve = torch.ops.dendra_solvers.dhs_solve
        else:
            raise NotImplementedError(
                f"DHS integrator is not implemented for device type {device.type}."
            )

    def _functional_solver(self):
        """Return the transform-compatible native CPU DHS facade.

        Imperative execution intentionally retains the raw dispatcher operator;
        its Python facade owns the forward-mode and nested ``torch.func``
        contracts needed by functional Population lowering. CUDA remains
        fail-closed for this CPU functional milestone.
        """
        solver = self._solve
        solver_name = getattr(solver, "__name__", None)
        solver_module = getattr(solver, "__module__", None)
        if solver_name == "dhs_solve" and solver_module == "torch._ops.dendra_solvers":
            facade = (
                None
                if not DENDRA_SOLVERS_AVAILABLE
                else getattr(dendra_solvers, "dhs_solve", None)
            )
            if callable(facade):
                return facade
            raise RuntimeError(
                "The selected native CPU solver requires a dendra-solvers build "
                "that exports the torch.func-compatible dhs_solve facade. Upgrade "
                "it with `python -m pip install --upgrade "
                '--only-binary=dendra-solvers "dendra-solvers>=0.3.1"`.'
            )
        raise RuntimeError(
            "The selected DHS solver is not yet transform-compatible. "
            "Functional scalar Tree execution currently requires CPU "
            "dendra_solvers.dhs_solve."
        )

    @staticmethod
    def _prepare_workspace(
        dt,
        *,
        cm,
        area,
        axial_conductance,
        edge_conductance,
    ):
        """Purely derive the scalar-tree implicit-Euler tensor workspace.

        ``cm`` is effective specific capacitance in uF/cm^2, ``area`` is
        effective membrane area in cm^2, ``axial_conductance`` is child-indexed
        in solver order (including the zero root), and ``edge_conductance`` uses
        the compact parent/child mechanism order used for extracellular drives.
        Topology adapters own graph preprocessing and provide all four physical
        tensors in flattened solve layout.
        """
        if torch.is_tensor(dt):
            # Preserve hidden vmap lanes, including an empty outer batch, until
            # the scalar timestep is combined with spatial tensors.
            dt_s = dt.reshape(1) * 1.0e-3
        else:
            dt_s = dt * 1.0e-3

        capacitance = 1.0e-6 * cm * area
        return {
            "area": area.clone(memory_format=torch.preserve_format),
            "cm_dt": capacitance / dt_s,
            "axial_conductance": axial_conductance.clone(
                memory_format=torch.preserve_format
            ),
            "edge_conductance": edge_conductance.clone(
                memory_format=torch.preserve_format
            ),
        }

    def initialize(self, model, dt):
        self.B = int(np.prod(model.shape[:-1]))

        self.base_shape = model.shape

        device = model.device()
        self.to(device)

        self._select_solver(device)

        graph = _compiled_tree_graph_view(model)
        if not isinstance(graph, list):
            graph = [graph]

        parent_idx_t, a_geom_t, node_order = graph_to_parent_and_axial(
            graph, dtype_axial=model.dtype()
        )
        parent_idx, _, depth = build_morphology(parent_idx_t.tolist())
        order, layer_ptr = build_dhs_layers(depth, self.threads)

        self.solver_order.copy_(
            torch.as_tensor(node_order, dtype=torch.int64, device=device)
        )  # (N,)
        self.inv_solver_order.copy_(torch.argsort(self.solver_order, dim=0))  # (N,)

        self.register_buffer(
            "layer_ptr", layer_ptr.to(dtype=torch.int64, device=device)
        )  # (L+1,)
        self.order.copy_(order.to(dtype=torch.int64, device=device))
        self.parent_idx.copy_(parent_idx.to(dtype=torch.int64, device=device))  # (N,)
        if getattr(model, "_canonical_edge_resistance_ohm", None) is not None:
            # The canonical model buffer is the numerical source of truth for
            # both unbranched solvers. In particular, ``float32 -> float64``
            # widens that buffer but cannot recreate the CompartmentGraph's
            # original binary64 values. Reading graph R here would therefore
            # make DHS disagree with UB after an otherwise valid dtype move.
            canonical_edges = unbranched_edge_conductance(model)
            node_conductance = torch.cat(
                (
                    torch.zeros(
                        canonical_edges.shape[0],
                        1,
                        device=canonical_edges.device,
                        dtype=canonical_edges.dtype,
                    ),
                    canonical_edges,
                ),
                dim=1,
            )
            axial_conductance = node_conductance.index_select(
                1, self.solver_order
            ).contiguous()
        else:
            rhoa_scale_mech = _as_solve_matrix(model.rhoa_scale, model)
            rhoa_scale_solver = rhoa_scale_mech.index_select(1, self.solver_order)
            axial_conductance = (
                _as_solve_matrix(a_geom_t.to(device=device, dtype=model.dtype()), model)
                .clone()
                .contiguous()
            ) / rhoa_scale_solver  # (B,N), solver order

        # extracellular
        # We will need the original node IDs from the graph for this
        # Assuming G.nodes() provides the original order [0, 1, ..., N-1]
        graph0 = graph[0]
        original_nodes = list(range(graph0.number_of_nodes()))
        original_idx_of = {n: i for i, n in enumerate(original_nodes)}

        # --- Create edge indices in the ORIGINAL node order ---
        edge_child_orig_list = []
        edge_parent_orig_list = []
        # Iterate in canonical mechanism order. Graph insertion order is not
        # semantic and may differ across a heterogeneous geometry batch.
        for child_node in original_nodes:
            preds = list(graph0.predecessors(child_node))
            if not preds:
                continue
            parent_node = preds[0]
            edge_child_orig_list.append(original_idx_of[child_node])
            edge_parent_orig_list.append(original_idx_of[parent_node])

        # Convert lists to tensors and register them as buffers
        edge_child_orig = torch.tensor(
            edge_child_orig_list, dtype=torch.int64, device=device
        )
        self.register_buffer("edge_child_orig", edge_child_orig)
        self.register_buffer(
            "edge_parent_orig",
            torch.tensor(edge_parent_orig_list, dtype=torch.int64, device=device),
        )
        # Map the already effective child-indexed conductance back to mechanism
        # order once, then gather the compact edge plane. This keeps the voltage
        # solve and extracellular-current path on one numerical source of truth.
        axial_mechanism_order = axial_conductance.index_select(1, self.inv_solver_order)
        edge_conductance = axial_mechanism_order.index_select(1, edge_child_orig)

        area_cm2 = _as_solve_matrix(
            model.area.to(device=device, dtype=model.dtype()), model
        ) * _as_solve_matrix(model.area_scale, model)
        cm = _as_solve_matrix(
            model.cm.to(device=device, dtype=model.dtype()), model
        ) * _as_solve_matrix(model.cm_scale, model)
        workspace = self._derive_prepared_workspace(
            dt,
            cm=cm,
            area=area_cm2,
            axial_conductance=axial_conductance,
            edge_conductance=edge_conductance,
        )
        self._install_prepared_workspace(workspace)

    def step(self, model, dt, ve=None, intra=None):
        v_new, i_membrane = self._call_kernel(
            "_step", model.v, dt, model.celsius, ve, intra
        )
        model.v = v_new
        if self.imem:
            model.i_membrane = i_membrane

    def _step(
        self,
        v,
        dt,
        temp,
        ve=None,
        intra=None,
        *,
        solver=None,
        call_local_currents=False,
    ):
        K = self.K
        v = self.mech.update_v(v)  # apply voltage processes
        v_old = v

        self._advance_pre_current(v_old, dt, temp)

        itot = None
        gtot_flat = None

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
            ) = evaluate(v_old)
        elif self.mech.currents:
            itot, gtot = self.mech.i(v_old)  # shapes: base_shape
            ion_current_frame = self._capture_ion_current_frame()
            ion_conductance_frame = self._capture_ion_conductance_frame()
        else:
            itot = None
            gtot = None
            ion_current_frame = self._capture_ion_current_frame()
            ion_conductance_frame = self._capture_ion_conductance_frame()

        if itot is not None:
            itot_flat = itot.reshape(self.B, K)  # (B,K), mA/cm^2
            gtot_flat = gtot.reshape(self.B, K)  # (B,K), mA/(cm^2 mV)
            area = self.area.reshape(self.B, K)  # (B,K), cm^2

            # ionic "reversal" term + scale to absolute mA
            f_n = (gtot_flat * v_old.reshape(self.B, K) - itot_flat) * area  # (B,K), mA
        else:
            # no ionic currents: zero contribution
            itot_flat = None
            gtot_flat = torch.zeros_like(self.cm_dt)  # (B,K)
            f_n = torch.zeros_like(self.cm_dt)

        if ve is not None:
            ve_flat = _flatten_to_solve(
                ve,
                K,
                self.base_shape if call_local_currents else None,
            )
            I_edge = _edge_currents(
                self.edge_child_orig,
                self.edge_parent_orig,
                self.edge_conductance,
                ve_flat,
            )  # (B, E) mA
            if call_local_currents:
                # Functional execution cannot mutate a shared destination when
                # only the edge drive carries a hidden vmap lane.
                S = (
                    torch.zeros_like(f_n)
                    .index_add(1, self.edge_child_orig, -I_edge)
                    .index_add(1, self.edge_parent_orig, I_edge)
                )
            else:
                # Keep the allocation-efficient imperative fast path.
                S = torch.zeros_like(f_n)
                S.scatter_add_(
                    1,
                    self.edge_child_orig.expand_as(I_edge),
                    -I_edge,
                )
                S.scatter_add_(
                    1,
                    self.edge_parent_orig.expand_as(I_edge),
                    I_edge,
                )
            f_n = f_n + S  # (B, K) mA

        if intra is not None:
            f_n = f_n + _flatten_to_solve(
                intra,
                K,
                self.base_shape if call_local_currents else None,
            )

        v_old_flat = v_old.reshape(self.B, K)  # (B,K)
        RHS = f_n + (self.cm_dt * v_old_flat)  # (B,K), mA

        area = self.area.reshape(self.B, K)  # (B,K), cm^2
        main = self.cm_dt + (gtot_flat * area)

        d_ = main.index_select(-1, self.solver_order)  # (B, N)
        b_ = RHS.index_select(-1, self.solver_order)  # (B, N)
        a = self.axial_conductance  # (B, N) axial conductance

        solve = self._solve if solver is None else solver
        v_out = solve(
            d_,
            a,
            b_,
            self.parent_idx,
            self.order,
            self.layer_ptr,
        )

        v_new = v_out.index_select(-1, self.inv_solver_order).reshape(
            self.base_shape
        )  # (B, N)

        # ---- i_membrane: net membrane current (cap + ionic) in mA ----
        i_membrane = None
        if self.imem:
            v_new_flat = v_new.reshape(self.B, K)  # (B,K), mV
            dv = v_new_flat - v_old_flat  # (B,K), mV

            dmem = main  # (B,K), A/V
            if self.mech.currents:
                # absolute ionic current at old step (mA)
                i_abs_old = itot_flat * area  # (B,K), mA
            else:
                i_abs_old = torch.zeros_like(dmem)

            # I_mem = (C/dt + G_abs)*Δv + I_ion_old   (mA, code units)
            i_mem_flat = dmem * dv + i_abs_old  # (B,K), mA
            i_membrane = i_mem_flat.reshape(self.base_shape)

        accepted_frame = self._linearize_ion_current_frame(
            ion_current_frame,
            ion_conductance_frame,
            v_new - v_old,
        )
        self._advance_post_current(v_old, dt, temp, accepted_frame)

        return v_new, i_membrane


class _dhs_multi(MultiIntegrator):
    r"""
    Multi-model DHS integrator for heterogeneous scalar cable systems.

    Packs ordinary independent populations as one-node trees, branched Tree
    morphologies, and Cable paths into padded flat buffers (shared stride
    ``K_stride = max K_g``). Per-group offsets allow one CUDA/CPU kernel to
    process every scalar group in one launch. Optional extracellular coupling
    is applied through the same child/parent edge representation.

    Parameters
    ----------
    imem : bool or None, optional
        If truthy, accumulate membrane currents each step. Default None.
    threads : int, optional
        Threads per warp lane for DHS elimination (must divide 32). Default 16.
    write_back : bool, optional
        If True, write updated voltages back into each group's population after
        stepping. Default True.
    """

    # The packed topology and index maps are immutable plan metadata.  These
    # four tensors are the complete parameter/timestep-dependent numerical
    # workspace consumed by a transition.  ``dendra.func`` prepares them from
    # explicit component physical tensors and installs them with
    # ``functional_call``; ordinary initialization uses the same pure builder.
    _PREPARED_WORKSPACE_SCHEMA = (
        ("a_geom_flat", "multi_plane"),
        ("CMDT_MECH", "multi_node"),
        ("SCALE_MECH", "multi_node"),
        ("EDGE_GAX_FLAT", "multi_edge"),
    )
    _FUNCTIONAL_OPERATOR_KIND = "scalar_multi"
    _FUNCTIONAL_CRITICAL_METHODS = ("_select_solver",)

    def __init__(
        self, model, mech, imem=None, threads: int = 16, write_back: bool = True
    ):
        from ..multi import MultiPopulation

        threads = _validate_dhs_threads(threads)
        assert len(model) > 0 and isinstance(model, MultiPopulation), (
            "model must be a non-empty MultiPopulation instance"
        )

        super().__init__(model, mech, imem, write_back)

        self.threads = threads
        self.num_groups = len(model)

        # Per-group shapes (filled in initialize)
        self.group_B: List[int] = []
        self.group_K: List[int] = []
        self.group_L: List[int] = []
        self.base_shapes: List[Tuple[int, int]] = []  # [(B_g, K_g), ...]

        # Global packed shapes (filled in initialize)
        self.B_total: int = 0
        self.K_stride: int = 0
        self.L_max: int = 0
        self.P: int = 1

        # Launch geometry
        self.NPW = THREADS_PER_WARP // self.threads  # neurons per warp
        self.grid_x = 0

        # Will be set in initialize
        self._solve = None

        # Step-time scratch planes (allocated in initialize)
        self._d_plane = None
        self._b_plane = None

        # --- plan & scratch caches (filled lazily in _step) ---
        self._plan_cache = {}  # key: (P, device) -> dict with tiled plan
        self._scratch_sig = None

    def __getstate__(self):
        """Exclude process-local tensor views from copies and checkpoints."""
        state = super().__getstate__()
        state["_plan_cache"] = {}
        return state

    def _select_solver(self, device):
        device = torch.device(device)
        if device.type == "cuda":
            self._solve = partial(dhs_solve_multi_cuda, threads=self.threads)
        elif device.type == "cpu":
            if not DENDRA_SOLVERS_AVAILABLE:
                raise ImportError(
                    "DHS integrator requires the dendra-solvers package for CPU "
                    "execution. Install it with "
                    "`python -m pip install --upgrade "
                    '--only-binary=dendra-solvers "dendra-solvers>=0.3.1"`.'
                )
            self._solve = _dhs_multi_solve_cpu
        else:
            raise NotImplementedError(
                f"DHS multi integrator is not implemented for device type "
                f"{device.type}."
            )

    def _functional_solver(self):
        """Return the transform-compatible native packed CPU facade.

        The imperative path deliberately keeps ``_dhs_multi_solve_cpu`` (or
        the CUDA launcher) and its existing call signature.  The public facade
        owns the forward-mode and nested ``torch.func`` contracts and accepts
        the twelve tensor operands of the native operator directly.
        """
        if self._solve is _dhs_multi_solve_cpu:
            facade = (
                None
                if not DENDRA_SOLVERS_AVAILABLE
                else getattr(dendra_solvers, "dhs_multi_solve", None)
            )
            if callable(facade):
                return facade
            raise RuntimeError(
                "The selected native CPU solver requires a dendra-solvers build "
                "that exports the torch.func-compatible dhs_multi_solve facade. "
                "Upgrade it with `python -m pip install --upgrade "
                '--only-binary=dendra-solvers "dendra-solvers>=0.3.1"`.'
            )
        raise RuntimeError(
            "The selected packed DHS solver is not yet transform-compatible. "
            "Functional MultiPopulation execution currently requires CPU "
            "dendra_solvers.dhs_multi_solve."
        )

    @staticmethod
    def _prepare_workspace(
        dt,
        *,
        a_geom_flat,
        cm,
        area,
        edge_conductance,
    ):
        """Purely derive the packed scalar implicit-Euler workspace.

        Parameters
        ----------
        dt
            Timestep in milliseconds.
        a_geom_flat
            Padded child-indexed axial conductance planes in solver order,
            shaped ``(P, B_total, K_stride)``.
        cm, area
            Effective specific capacitance and membrane area in packed
            mechanism order.  Either ``(P, N_total)`` or the public packed
            voltage layout ``(*batch, 1, N_total)`` is accepted.
        edge_conductance
            Compact parent/child edge conductance planes in mechanism order,
            shaped ``(P, E_total)``.

        The returned tensors own independent storage while retaining autograd
        edges to every explicit physical input.  Structural padding and index
        maps remain outside this workspace.
        """
        if torch.is_tensor(dt):
            # Keep hidden vmap lanes intact while giving a scalar timestep one
            # visible singleton dimension for packed-node broadcasting.
            dt_s = dt.reshape(1) * 1.0e-3
        else:
            dt_s = dt * 1.0e-3

        plane_count = a_geom_flat.shape[0]
        n_total = cm.shape[-1]
        area_mech = area.reshape(plane_count, n_total)
        cm_mech = cm.reshape(plane_count, n_total)
        return {
            "a_geom_flat": a_geom_flat.clone(memory_format=torch.preserve_format),
            "CMDT_MECH": (1.0e-6 * cm_mech * area_mech) / dt_s,
            "SCALE_MECH": area_mech.clone(memory_format=torch.preserve_format),
            "EDGE_GAX_FLAT": edge_conductance.clone(
                memory_format=torch.preserve_format
            ),
        }

    def initialize(self, models, dt: float):
        assert len(models) == self.num_groups

        # ``initialize`` is also the shape/dt reinitialization path.  Keep all
        # Python-side group metadata idempotent rather than appending a second
        # copy of the old plan on every rebuild.
        self.group_B = []
        self.group_K = []
        self.group_L = []
        self.base_shapes = []
        self._plan_cache.clear()
        self._scratch_sig = None

        if self.write_back:
            self._calc_splits(models)

        # --- device / dtype from first group ---
        dev0 = models.device()
        dtype0 = models.dtype()
        for m in models:
            assert m.device().type == dev0.type, "All models must be on the same device"
            assert m.dtype() == dtype0, "All models must share the same dtype"
        self.to(dev0)

        self._select_solver(dev0)

        P = int(np.prod(models.shape[:-2])) if len(models.shape) > 2 else 1
        self.P = P

        P_list, ORDER_list, LPTR_list = [], [], []
        L_list = []
        SOLVER_list, INV_SOLVER_list = [], []
        B_list, K_list = [], []
        a_rows, cm_rows, area_rows = [], [], []
        row_off, mech_off, solver_off = [], [], []
        row_cursor = 0
        mech_cursor = 0
        solver_cursor = 0

        # ---------- per-group topology & params ----------
        for g, mdl in enumerate(models):
            group = _scalar_dhs_group(mdl, P, dev0)
            B_g, K_g = group.B, group.K

            self.base_shapes.append((B_g, K_g))
            B_list.append(B_g)
            K_list.append(K_g)

            parent_idx, _, depth = build_morphology(group.parent_idx.tolist())
            order_g, layer_ptr_g = build_dhs_layers(depth, self.threads)

            solver_order_g = group.solver_order
            inv_solver_g = torch.argsort(solver_order_g, dim=0)

            # Match the physical scaling contract of the scalar DHS solver.
            # These are population-specific, so applying the composite model's
            # globals after concatenation would be incorrect.  Retain all
            # leading batch planes: RANGE parameters can legitimately differ
            # between those planes after ``MultiPopulation.batch``.
            area_cm2 = _scalar_group_planes(
                mdl.area, mdl, P, B_g, K_g, "area", device=dev0
            )
            area_cm2 = area_cm2 * _scalar_group_planes(
                mdl.area_scale,
                mdl,
                P,
                B_g,
                K_g,
                "area_scale",
                device=dev0,
            )
            cm_density = _scalar_group_planes(
                mdl.cm, mdl, P, B_g, K_g, "cm", device=dev0
            )
            cm_scale = _scalar_group_planes(
                mdl.cm_scale,
                mdl,
                P,
                B_g,
                K_g,
                "cm_scale",
                device=dev0,
            )
            cm_g = (cm_density * cm_scale).contiguous()

            a_geom_g = group.a_geom
            area_g = area_cm2.contiguous()

            P_list.append(parent_idx.to(dtype=torch.int64, device=dev0))
            ORDER_list.append(order_g.to(dtype=torch.int64, device=dev0))
            LPTR_list.append(layer_ptr_g.to(dtype=torch.int64, device=dev0))
            L_list.append(int(layer_ptr_g.numel() - 1))

            SOLVER_list.append(solver_order_g)
            INV_SOLVER_list.append(inv_solver_g)

            a_rows.append(a_geom_g)
            cm_rows.append(cm_g)
            area_rows.append(area_g)

            row_off.append(row_cursor)
            mech_off.append(mech_cursor)
            solver_off.append(solver_cursor)

            row_cursor += B_g
            mech_cursor += B_g * K_g
            solver_cursor += K_g

        # ---------- globals ----------
        self.group_B, self.group_K, self.group_L = B_list, K_list, L_list
        self.B_total = int(sum(B_list))
        self.K_stride = int(max(K_list)) if K_list else 0
        self.L_max = int(max(L_list)) if L_list else 0

        self.register_buffer(
            "ROW_OFF", torch.as_tensor(row_off, dtype=torch.int64, device=dev0)
        )
        self.register_buffer(
            "MECH_OFF", torch.as_tensor(mech_off, dtype=torch.int64, device=dev0)
        )
        self.register_buffer(
            "SOLVER_OFF", torch.as_tensor(solver_off, dtype=torch.int64, device=dev0)
        )

        # ---------- concat topology ----------
        P_cat = (
            torch.cat(P_list, 0)
            if P_list
            else torch.empty(0, dtype=torch.int64, device=dev0)
        )
        ORDER_cat = (
            torch.cat(ORDER_list, 0)
            if ORDER_list
            else torch.empty(0, dtype=torch.int64, device=dev0)
        )
        LAYER_PTR_cat = (
            torch.cat(LPTR_list, 0)
            if LPTR_list
            else torch.empty(0, dtype=torch.int64, device=dev0)
        )
        SOLVER_cat = (
            torch.cat(SOLVER_list, 0)
            if SOLVER_list
            else torch.empty(0, dtype=torch.int64, device=dev0)
        )
        INV_SOLVER_cat = (
            torch.cat(INV_SOLVER_list, 0)
            if INV_SOLVER_list
            else torch.empty(0, dtype=torch.int64, device=dev0)
        )

        P_OFF = torch.as_tensor(
            np.cumsum([0] + K_list[:-1]).tolist(), dtype=torch.int64, device=dev0
        )
        ORDER_OFF = torch.as_tensor(
            np.cumsum([0] + K_list[:-1]).tolist(), dtype=torch.int64, device=dev0
        )
        LPTR_OFF = torch.as_tensor(
            np.cumsum([0] + [lp.numel() for lp in LPTR_list][:-1]).tolist(),
            dtype=torch.int64,
            device=dev0,
        )
        L_per_group = torch.as_tensor(L_list, dtype=torch.int32, device=dev0)

        self.register_buffer("P_cat", P_cat)
        self.register_buffer("ORDER_cat", ORDER_cat)
        self.register_buffer("LAYER_PTR_cat", LAYER_PTR_cat)
        self.register_buffer("P_OFF", P_OFF)
        self.register_buffer("ORDER_OFF", ORDER_OFF)
        self.register_buffer("LPTR_OFF", LPTR_OFF)
        self.register_buffer("L_per_group", L_per_group)
        self.register_buffer("SOLVER_cat", SOLVER_cat)
        self.register_buffer("INV_SOLVER_cat", INV_SOLVER_cat)

        # ---------- warp plan (one morphology per warp) ----------
        NPW = self.NPW
        warp_p_off, warp_o_off, warp_lptr_off = [], [], []
        warp_L, warp_row_base, warp_row_count = [], [], []
        for g in range(self.num_groups):
            B_g = int(self.group_B[g])
            n_warps_g = (B_g + NPW - 1) // NPW
            p_off = int(P_OFF[g].item())
            o_off = int(ORDER_OFF[g].item())
            lptr_off = int(LPTR_OFF[g].item())
            Lg = int(L_per_group[g].item())
            row0 = int(self.ROW_OFF[g].item())

            for t in range(n_warps_g):
                warp_p_off.append(p_off)
                warp_o_off.append(o_off)
                warp_lptr_off.append(lptr_off)
                warp_L.append(Lg)
                base = row0 + t * NPW
                cnt = min(NPW, B_g - t * NPW)
                warp_row_base.append(base)
                warp_row_count.append(cnt)

        self.grid_x = len(warp_row_base)
        self.register_buffer(
            "WARP_P_OFF", torch.tensor(warp_p_off, dtype=torch.int64, device=dev0)
        )
        self.register_buffer(
            "WARP_ORDER_OFF", torch.tensor(warp_o_off, dtype=torch.int64, device=dev0)
        )
        self.register_buffer(
            "WARP_LPTR_OFF", torch.tensor(warp_lptr_off, dtype=torch.int64, device=dev0)
        )
        self.register_buffer(
            "WARP_L", torch.tensor(warp_L, dtype=torch.int32, device=dev0)
        )
        self.register_buffer(
            "WARP_ROW_BASE", torch.tensor(warp_row_base, dtype=torch.int64, device=dev0)
        )
        self.register_buffer(
            "WARP_ROW_COUNT",
            torch.tensor(warp_row_count, dtype=torch.int32, device=dev0),
        )

        # ---------- padded flat planes ----------
        if self.B_total == 0 or self.K_stride == 0:
            self.register_buffer(
                "a_geom_flat", torch.empty(0, 0, device=dev0, dtype=dtype0)
            )
            self.register_buffer(
                "cmdt_flat", torch.empty(0, 0, device=dev0, dtype=dtype0)
            )
            self.register_buffer(
                "scale_flat", torch.empty(0, 0, device=dev0, dtype=dtype0)
            )
            return

        plane_shape = (P, self.B_total, self.K_stride)
        a_flat = torch.zeros(plane_shape, device=dev0, dtype=dtype0)
        for g, (B_g, K_g) in enumerate(zip(B_list, K_list)):
            r0 = int(self.ROW_OFF[g])
            r1 = r0 + B_g
            a_flat[:, r0:r1, :K_g].copy_(a_rows[g])  # solver order

        # ---------- vectorized mech↔solver mapping ----------
        mech_rows, cm_mech_flat, area_mech_flat = [], [], []
        mech_cols_inv = []
        for g, (B_g, K_g) in enumerate(zip(self.group_B, self.group_K)):
            r0 = int(self.ROW_OFF[g])
            r1 = r0 + B_g
            rows_g = torch.arange(r0, r1, device=dev0).repeat_interleave(
                K_g
            )  # (B_g*K_g,)
            mech_rows.append(rows_g)

            soff = int(self.SOLVER_OFF[g])
            inv_cols = self.INV_SOLVER_cat[soff : soff + K_g]  # mechanism m -> solver s
            mech_cols_inv.append(inv_cols.repeat(B_g))  # (B_g*K_g,)

            cm_mech_flat.append(cm_rows[g].reshape(P, -1))
            area_mech_flat.append(area_rows[g].reshape(P, -1))

        self.register_buffer("MECH_ROWS", torch.cat(mech_rows, 0))
        self.register_buffer("MECH_COLS_INV", torch.cat(mech_cols_inv, 0))
        cm_mech = torch.cat(cm_mech_flat, 1)
        area_mech = torch.cat(area_mech_flat, 1)

        self.N_mech = int(area_mech.shape[1])
        PLANE_LIN_BASE = self.MECH_ROWS * self.K_stride + self.MECH_COLS_INV
        self.register_buffer("PLANE_LIN_BASE", PLANE_LIN_BASE.to(torch.long))

        # ---------- extracellular edge maps (for ve) ----------
        edge_child_idx_flat_all = []
        edge_parent_idx_flat_all = []
        edge_gax_flat_all = []

        for g, (B_g, K_g) in enumerate(zip(self.group_B, self.group_K)):
            # Derive original mechanism columns from the canonical solver
            # topology. This is valid for every adapter, including one-node
            # point groups, and avoids reconstructing Cable graphs through
            # scalar ``.item()`` calls.
            soff = int(self.SOLVER_OFF[g])
            solver_nodes = self.SOLVER_cat[soff : soff + K_g]
            parent_solver = P_list[g]
            edge_solver_col = torch.arange(K_g, dtype=torch.int64, device=dev0)[
                parent_solver >= 0
            ]
            if edge_solver_col.numel() == 0:
                continue  # degenerate, no edges
            edge_parent_solver = parent_solver.index_select(0, edge_solver_col)
            edge_child = solver_nodes.index_select(0, edge_solver_col)
            edge_parent = solver_nodes.index_select(0, edge_parent_solver)

            # Preserve per-neuron heterogeneous geometry and the rhoa_scale
            # dependency without round-tripping through Python scalars.
            edge_gax = a_rows[g].index_select(2, edge_solver_col)  # (P, B_g, E_g)

            # Repeat across batch rows and convert to FLAT mechanism indices
            rows = torch.arange(B_g, device=dev0, dtype=torch.int64).repeat_interleave(
                edge_child.numel()
            )  # (B_g*E_g,)
            child_cols = edge_child.repeat(B_g)  # (B_g*E_g,)
            parent_cols = edge_parent.repeat(B_g)  # (B_g*E_g,)
            g_mech_off = int(self.MECH_OFF[g])

            child_flat = g_mech_off + rows * K_g + child_cols
            parent_flat = g_mech_off + rows * K_g + parent_cols
            gax_flat = edge_gax.reshape(P, -1)  # (P, B_g*E_g)

            edge_child_idx_flat_all.append(child_flat)
            edge_parent_idx_flat_all.append(parent_flat)
            edge_gax_flat_all.append(gax_flat)

        if edge_child_idx_flat_all:
            self.register_buffer(
                "EDGE_CHILD_IDX_FLAT", torch.cat(edge_child_idx_flat_all, 0)
            )
            self.register_buffer(
                "EDGE_PARENT_IDX_FLAT", torch.cat(edge_parent_idx_flat_all, 0)
            )
            edge_gax_flat = torch.cat(edge_gax_flat_all, 1)
        else:
            # empty placeholders
            self.register_buffer(
                "EDGE_CHILD_IDX_FLAT", torch.empty(0, dtype=torch.int64, device=dev0)
            )
            self.register_buffer(
                "EDGE_PARENT_IDX_FLAT", torch.empty(0, dtype=torch.int64, device=dev0)
            )
            edge_gax_flat = torch.empty(P, 0, dtype=dtype0, device=dev0)

        workspace = self._derive_prepared_workspace(
            dt,
            a_geom_flat=a_flat,
            cm=cm_mech,
            area=area_mech,
            edge_conductance=edge_gax_flat,
        )
        for name, _shape_role in self._prepared_workspace_schema():
            if name not in self._buffers:
                self.register_buffer(name, workspace[name])
        self._install_prepared_workspace(workspace)

        # Retain the historical padded diagnostic views.  They are derived
        # from the canonical mechanism-order workspace and are not transition
        # inputs of their own.
        cmdt_flat = (
            torch.zeros(
                P,
                self.B_total * self.K_stride,
                device=dev0,
                dtype=dtype0,
            )
            .index_copy(1, self.PLANE_LIN_BASE, self.CMDT_MECH)
            .reshape(P, self.B_total, self.K_stride)
        )
        scale_flat = (
            torch.zeros_like(cmdt_flat)
            .reshape(P, -1)
            .index_copy(1, self.PLANE_LIN_BASE, self.SCALE_MECH)
            .reshape_as(cmdt_flat)
        )
        if "cmdt_flat" in self._buffers:
            self.cmdt_flat = cmdt_flat
            self.scale_flat = scale_flat
        else:
            self.register_buffer("cmdt_flat", cmdt_flat)
            self.register_buffer("scale_flat", scale_flat)

        # ---------- allocate step scratch ----------
        rows_total = P * self.B_total
        self._d_plane = torch.empty(
            (rows_total, self.K_stride), device=dev0, dtype=dtype0
        )
        self._b_plane = torch.empty_like(self._d_plane)
        plan = self._get_tiled_plan(P, dev0)
        self._install_fixed_tiled_plan(plan)

    def _install_fixed_tiled_plan(self, plan):
        """Register the initialized batch shape's immutable tiled metadata.

        Functional transitions must not enter the Python plan cache while a
        transform is active.  The explicit Population batch shape is fixed at
        lowering time, so materialize its repeated warp metadata once.  The
        differentiable axial plane remains a prepared workspace tensor and is
        deliberately excluded from this static plan.
        """
        entries = {
            "_TILED_WARP_P_OFF": plan["WARP_P_OFF"],
            "_TILED_WARP_ORDER_OFF": plan["WARP_ORDER_OFF"],
            "_TILED_WARP_LPTR_OFF": plan["WARP_LPTR_OFF"],
            "_TILED_WARP_L": plan["WARP_L"],
            "_TILED_WARP_ROW_BASE": plan["WARP_ROW_BASE"],
            "_TILED_WARP_ROW_COUNT": plan["WARP_ROW_COUNT"],
            "_TILED_PLIN_FLAT": plan["PLIN_flat"],
        }
        for name, value in entries.items():
            if name in self._buffers:
                setattr(self, name, value)
            else:
                self.register_buffer(name, value)
        self._tiled_grid_x = int(plan["grid_x"])
        self._tiled_rows_total = int(plan["rows_total"])

    def _get_tiled_plan(self, P: int, device: torch.device):
        """
        Returns a dict with:
          a_geom_eff, WARP_P_OFF, WARP_ORDER_OFF, WARP_LPTR_OFF, WARP_L,
          WARP_ROW_BASE, WARP_ROW_COUNT, grid_x, PLIN_flat, rows_total
        All tensors are on `device`; a_geom_eff uses `dtype`.
        """
        key = (int(P), str(device))
        cached = self._plan_cache.get(key, None)
        if cached is not None:
            return cached
        if self.a_geom_flat.shape[0] != P:
            raise ValueError(
                "Packed multi-tree geometry is stale for the requested batch "
                f"planes: initialized for {self.a_geom_flat.shape[0]}, got {P}."
            )

        Btot, Kstride = self.B_total, self.K_stride
        plane_stride = Btot * Kstride
        rows_total = P * Btot

        # Batched linear indices for (P, N_mech) -> (rows_total, Kstride)
        # PLIN[b, j] = PLANE_LIN_BASE[j] + b * plane_stride
        batch_offsets = (
            torch.arange(P, device=device, dtype=torch.long) * plane_stride
        ).unsqueeze(1)  # (P,1)
        PLIN_flat = (self.PLANE_LIN_BASE.unsqueeze(0) + batch_offsets).reshape(
            -1
        )  # (P*N_mech,)

        if P == 1:
            plan = dict(
                a_geom_eff=self.a_geom_flat[0],  # (Btot, Kstride)
                WARP_P_OFF=self.WARP_P_OFF,
                WARP_ORDER_OFF=self.WARP_ORDER_OFF,
                WARP_LPTR_OFF=self.WARP_LPTR_OFF,
                WARP_L=self.WARP_L,
                WARP_ROW_BASE=self.WARP_ROW_BASE,
                WARP_ROW_COUNT=self.WARP_ROW_COUNT,
                grid_x=self.grid_x,
                PLIN_flat=PLIN_flat,
                rows_total=rows_total,
            )
        else:
            # Geometry can differ across leading batch planes.  Flatten the
            # already packed planes in the same P-major order as the RHS.
            a_geom_eff = self.a_geom_flat.reshape(P * Btot, Kstride).contiguous()

            W = int(self.grid_x)
            row_base_offsets = (
                torch.arange(P, device=device, dtype=torch.long) * Btot
            ).repeat_interleave(W)  # (P*W,)

            plan = dict(
                a_geom_eff=a_geom_eff,
                WARP_P_OFF=self.WARP_P_OFF.repeat(P),
                WARP_ORDER_OFF=self.WARP_ORDER_OFF.repeat(P),
                WARP_LPTR_OFF=self.WARP_LPTR_OFF.repeat(P),
                WARP_L=self.WARP_L.repeat(P),
                WARP_ROW_BASE=self.WARP_ROW_BASE.repeat(P) + row_base_offsets,
                WARP_ROW_COUNT=self.WARP_ROW_COUNT.repeat(P),
                grid_x=W * P,
                PLIN_flat=PLIN_flat,
                rows_total=rows_total,
            )

        self._plan_cache[key] = plan
        return plan

    def step(self, model, dt, ve=None, intra=None):
        v_new, i_mem = self._call_kernel(
            "_step", model.v, dt, getattr(model, "celsius", None), ve, intra
        )
        model.v = v_new
        if self.imem:
            model.i_membrane = i_mem
        self._write_back(model)

    def _step(
        self,
        v,
        dt,
        temp=None,
        ve=None,
        intra=None,
        *,
        solver=None,
        call_local_currents=False,
    ):
        if self._solve is None and solver is None:
            raise NotImplementedError("Multi-morph kernel not connected.")

        orig_shape = v.shape  # keep whatever the caller gave us
        assert orig_shape[-2] == 1, "expect shape (..., 1, N_total)"
        N_total = orig_shape[-1]

        # Flatten leading batch dims into P
        P = self.P

        # 1) mechanisms on flattened vector
        v = self.mech.update_v(v)
        v_old = v

        self._advance_pre_current(v_old, dt, temp)
        if call_local_currents:
            evaluate = getattr(self.mech, "_evaluate_current_frame", None)
            if evaluate is None:
                raise RuntimeError(
                    "call-local current evaluation requires a MechanismHandler "
                    "with _evaluate_current_frame()"
                )
            (
                itot_flat,
                gtot_flat,
                ion_current_frame,
                ion_conductance_frame,
            ) = evaluate(v_old)
        else:
            itot_flat, gtot_flat = self.mech.i(v)  # both original shape
            ion_current_frame = self._capture_ion_current_frame()
            ion_conductance_frame = self._capture_ion_conductance_frame()

        intra_flat = (
            0.0
            if intra is None
            else _broadcast_to_shape(intra, tuple(orig_shape)).reshape(P, N_total)
        )

        SCALE_MECH = self.SCALE_MECH.reshape(P, N_total)

        # 2) assemble in mechanism order
        v_old_flat = v_old.reshape(P, N_total)  # (P, N_total)
        itot_mech = itot_flat.reshape(P, N_total)
        gtot_mech = gtot_flat.reshape(P, N_total)

        f_n_flat = (gtot_mech * v_old_flat - itot_mech) * SCALE_MECH + intra_flat

        # --- extracellular coupling (ve), vectorized on flattened indices) ---
        if ve is not None and self.EDGE_CHILD_IDX_FLAT.numel() > 0:
            ve_flat = _broadcast_to_shape(ve, tuple(orig_shape)).reshape(P, N_total)
            I_edge = _edge_currents(
                self.EDGE_CHILD_IDX_FLAT,
                self.EDGE_PARENT_IDX_FLAT,
                self.EDGE_GAX_FLAT,
                ve_flat,
            )

            if call_local_currents:
                # Functional execution cannot mutate a shared destination
                # when only the edge drive or conductance owns a hidden vmap
                # lane (including a zero-sized lane batch).
                S_flat = (
                    torch.zeros_like(f_n_flat)
                    .index_add(1, self.EDGE_CHILD_IDX_FLAT, -I_edge)
                    .index_add(1, self.EDGE_PARENT_IDX_FLAT, I_edge)
                )
            else:
                # Preserve the allocation-efficient imperative fast path.
                S_flat = torch.zeros_like(f_n_flat)
                edge_child = self.EDGE_CHILD_IDX_FLAT.unsqueeze(0).expand(P, -1)
                edge_parent = self.EDGE_PARENT_IDX_FLAT.unsqueeze(0).expand(P, -1)
                S_flat.scatter_add_(1, edge_child, -I_edge)
                S_flat.scatter_add_(1, edge_parent, I_edge)
            f_n_flat = f_n_flat + S_flat

        CMDT_MECH = self.CMDT_MECH.reshape(P, N_total)

        RHS_flat = f_n_flat + CMDT_MECH * v_old_flat
        MAIN_flat = CMDT_MECH + gtot_mech * SCALE_MECH

        device = v_old.device
        if call_local_currents:
            # Build disposable planes from current-bearing tensors so hidden
            # torch.func lanes propagate even when the packed topology and
            # index map are shared.  Indexing within each explicit P plane is
            # equivalent to the imperative tiled linear map without entering
            # the Python plan cache.
            plane_width = self.B_total * self.K_stride
            zero_rhs = RHS_flat[..., :1] * 0.0
            zero_main = MAIN_flat[..., :1] * 0.0
            b_plane = (
                zero_rhs.expand(P, plane_width)
                .index_copy(1, self.PLANE_LIN_BASE, RHS_flat)
                .reshape(self._tiled_rows_total, self.K_stride)
            )
            d_plane = (
                zero_main.expand(P, plane_width)
                .index_copy(1, self.PLANE_LIN_BASE, MAIN_flat)
                .reshape(self._tiled_rows_total, self.K_stride)
            )
            a_geom_eff = (
                self.a_geom_flat[0]
                if P == 1
                else self.a_geom_flat.reshape(self._tiled_rows_total, self.K_stride)
            )
            warp_p_off = self._TILED_WARP_P_OFF
            warp_order_off = self._TILED_WARP_ORDER_OFF
            warp_lptr_off = self._TILED_WARP_LPTR_OFF
            warp_l = self._TILED_WARP_L
            warp_row_base = self._TILED_WARP_ROW_BASE
            warp_row_count = self._TILED_WARP_ROW_COUNT
        else:
            plan = self._get_tiled_plan(P, device)
            PLIN_flat = plan["PLIN_flat"]

            # Zero & scatter by linear indices (fast, vectorized).  The packed
            # solver's autograd rule saves its diagonal input for the adjoint
            # solve. Reusing and mutating scratch on a later timestep would
            # invalidate that saved tensor during BPTT, so differentiable
            # imperative solves receive fresh storage.
            differentiable_solve = torch.is_grad_enabled() and any(
                value.requires_grad
                for value in (MAIN_flat, RHS_flat, plan["a_geom_eff"])
            )
            if differentiable_solve:
                b_plane = (
                    torch.zeros_like(self._b_plane)
                    .view(-1)
                    .index_copy(0, PLIN_flat, RHS_flat.view(-1))
                    .view_as(self._b_plane)
                )
                d_plane = (
                    torch.zeros_like(self._d_plane)
                    .view(-1)
                    .index_copy(0, PLIN_flat, MAIN_flat.view(-1))
                    .view_as(self._d_plane)
                )
            else:
                self._b_plane.view(-1).zero_().index_copy_(
                    0, PLIN_flat, RHS_flat.view(-1)
                )
                self._d_plane.view(-1).zero_().index_copy_(
                    0, PLIN_flat, MAIN_flat.view(-1)
                )
                b_plane = self._b_plane
                d_plane = self._d_plane
            a_geom_eff = plan["a_geom_eff"]
            warp_p_off = plan["WARP_P_OFF"]
            warp_order_off = plan["WARP_ORDER_OFF"]
            warp_lptr_off = plan["WARP_LPTR_OFF"]
            warp_l = plan["WARP_L"]
            warp_row_base = plan["WARP_ROW_BASE"]
            warp_row_count = plan["WARP_ROW_COUNT"]

        solve = self._solve if solver is None else solver
        solver_args = (
            d_plane,
            a_geom_eff,
            b_plane,
            self.P_cat,
            self.ORDER_cat,
            self.LAYER_PTR_cat,
            warp_p_off,
            warp_order_off,
            warp_lptr_off,
            warp_l,
            warp_row_base,
            warp_row_count,
        )
        if solver is None:
            v_out_solver = solve(
                *solver_args,
                K_stride=self.K_stride,
                L_max=self.L_max,
                grid_x=(self._tiled_grid_x if call_local_currents else plan["grid_x"]),
            )
        else:
            v_out_solver = solve(*solver_args)

        # 6) gather back to mechanism order with the same index map
        if call_local_currents:
            v_sel = v_out_solver.reshape(P, plane_width).index_select(
                1, self.PLANE_LIN_BASE
            )
        else:
            v_sel = v_out_solver.view(-1).index_select(0, PLIN_flat)
        v_new = v_sel.reshape(orig_shape)

        # --- net membrane current: cap + ionic, per mechanism index ---
        i_mem = None
        if self.imem:
            v_new_flat = v_new.reshape(P, N_total)  # (P, N_total)
            dv_flat = v_new_flat - v_old_flat  # ΔV, mV

            # absolute ionic current at old step (mA)
            i_abs_old_flat = itot_mech * SCALE_MECH  # (P, N_total), mA

            # dmem = C/dt + G_abs (A/V)
            dmem_flat = MAIN_flat  # (P, N_total), A/V

            # I_mem = dmem * ΔV + I_ion_old  (mA, code units)
            i_mem_flat = dmem_flat * dv_flat + i_abs_old_flat  # (P, N_total)
            i_mem = i_mem_flat.reshape(orig_shape)  # (..., 1, N_total)

        accepted_frame = self._linearize_ion_current_frame(
            ion_current_frame,
            ion_conductance_frame,
            v_new - v_old,
        )
        self._advance_post_current(v_old, dt, temp, accepted_frame)

        return v_new, i_mem
