from collections import deque
from functools import partial
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch

from .core import Integrator
from .triton import dhs_solve_cuda, dhs_solve_multi_cuda

try:
    import axonml_solvers  # noqa: F401

    AXONML_SOLVERS_AVAILABLE = True
except ImportError:
    AXONML_SOLVERS_AVAILABLE = False


THREADS_PER_WARP = 32


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
    children = [[] for _ in range(K)]
    root = None
    for i, p in enumerate(parent_idx):
        if p == -1:
            root = i
        else:
            children[p].append(i)

    depth = torch.zeros(K, dtype=torch.int32)
    q = deque([root])
    while q:
        u = q.popleft()
        for c in children[u]:
            depth[c] = depth[u] + 1
            q.append(c)

    return (torch.as_tensor(parent_idx, dtype=torch.int32), children, depth)


def graph_to_parent_and_axial(
    G: nx.DiGraph, dtype_axial: torch.dtype = torch.float32
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
    # ------------------------------------------------------------------
    # 0. topological order and quick look‑ups
    # ------------------------------------------------------------------
    nodes = list(nx.topological_sort(G))  # length K
    idx_of = {n: i for i, n in enumerate(nodes)}
    K = len(nodes)

    parent_idx = np.full(K, -1, dtype=np.int32)
    a_geom = np.zeros(K, dtype=np.float32)

    # constant: 1 µm = 1 e‑4 cm
    microns_to_cm = 1e-4
    pi = np.pi

    # ------------------------------------------------------------------
    # 1. iterate over all nodes except the roots
    # ------------------------------------------------------------------
    for child in nodes:
        i = idx_of[child]
        preds = list(G.predecessors(child))

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
        edge_data = G.get_edge_data(parent, child, default={})
        R_total = edge_data.get("R_ohm", None)  # Ω or None

        # --------------------------------------------------------------
        # 1b.  If not present, fall back to geometric half‑segment calc
        # --------------------------------------------------------------
        if R_total is None:
            try:
                # child geometry
                L_i = G.nodes[child]["L"] * microns_to_cm  # cm
                d_i_cm = G.nodes[child]["diam"] * microns_to_cm
                r_i_cm = 0.5 * d_i_cm
                rho_i = G.nodes[child]["Ra"]  # Ω·cm

                # parent geometry
                L_p = G.nodes[parent]["L"] * microns_to_cm
                d_p_cm = G.nodes[parent]["diam"] * microns_to_cm
                r_p_cm = 0.5 * d_p_cm
                rho_p = G.nodes[parent]["Ra"]  # Ω·cm
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
        a_geom[i] = 1.0 / R_total

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
    diff = ve.index_select(1, edge_parent) - ve.index_select(1, edge_child)
    return diff * edge_gax


class _dhs(Integrator):
    """
    Implements Dendritic Hierarchical Scheduling (DHS) for efficient integration of dendritic tree models.

    This integrator leverages a hierarchical elimination order to solve the compartmental equations
    of dendritic structures in a scalable, multi-threaded manner. It sets up buffers for various
    morphological and biophysical properties, constructs a scheduling of computational layers, and
    performs forward elimination on the system matrix, ultimately updating the membrane potentials.

    Parameters:
        model: Neural model object containing morphological (e.g., diameters, distances) and
               biophysical properties.
        mech: MechanismHandler to be advanced during simulation steps.
        imem: Flag whether to store membrane current.
        threads: Number of threads to use for parallel elimination in the hierarchical
                 scheduling procedure (default is 32).

    Core Methods:
        initialize(model, dt):
            Configures internal buffers by converting geometrical information to biophysical
            parameters, constructing the dendritic morphology (parent indices and elimination
            order), and setting up the computational layers.

        step(model, dt, ve=None, intra=None):
            Executes a simulation step by advancing the mechanism state and solving the
            modified system using the DHS strategy.

        _step(v, dt, temp, ve=None, intra=None):
            Internal method that advances the simulation state by performing the forward
            elimination via the DHS algorithm.

    Reference:
        Zhang, Y., He, G., Ma, L. et al. A GPU-based computational framework that bridges
        neuron simulation and artificial intelligence. Nat Commun 14, 5798 (2023).
        https://doi.org/10.1038/s41467-023-41553-7
    """

    def __init__(self, model, mech, imem=None, threads=16):
        assert THREADS_PER_WARP % threads == 0, "threads must divide 32 (warp size)"
        assert threads <= 32, "threads must be ≤ 32 (warp size)"

        super().__init__(model, mech, imem)
        self.threads = threads

        N = model.shape[-1]
        B = np.prod(model.shape[:-1])
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
        self.register_buffer("scale", torch.empty(1, N))  # (1, N) scale factor

        self.register_buffer("a_geom", torch.empty(1, N))  # (B,N) axial conductance
        self.register_buffer("cmdt", torch.empty(1, N))  # (B,N) capacitance * dt

    def initialize(self, model, dt):
        self.B = np.prod(model.shape[:-1])
        B = self.B

        self.base_shape = model.shape

        dt_s = dt * 1e-3

        device = model.device()
        self.to(device)

        if device.type == "cpu" and not AXONML_SOLVERS_AVAILABLE:
            raise ImportError(
                "DHS integrator requires axonml_solvers package for CPU execution. "
                "Please install it with `pip install axonml_solvers`."
            )

        if device.type == "cuda":
            self.solve = partial(dhs_solve_cuda, threads=self.threads)
        elif device.type == "cpu":
            self.solve = torch.ops.axonml_solvers.dhs_solve
        else:
            raise NotImplementedError(
                f"DHS integrator is not implemented for device type {device.type}."
            )

        parent_idx_t, a_geom_t, node_order = graph_to_parent_and_axial(model.graph)
        parent_idx, _, depth = build_morphology(parent_idx_t.tolist())
        order, layer_ptr = build_dhs_layers(depth, self.threads)

        self.solver_order.copy_(
            torch.as_tensor(node_order, dtype=torch.int64, device=device)
        )  # (N,)
        self.inv_solver_order.copy_(torch.argsort(self.solver_order, dim=0))  # (N,)

        area_cm2 = model.area.to(device=device)  # cm²

        self.register_buffer(
            "layer_ptr", layer_ptr.to(dtype=torch.int64, device=device)
        )  # (L+1,)
        self.order.copy_(order.to(dtype=torch.int64, device=device))
        self.parent_idx.copy_(parent_idx.to(dtype=torch.int64, device=device))  # (N,)
        self.a_geom = a_geom_t.expand(B, -1).to(
            device=device, dtype=model.dtype()
        )  # (B,N)

        self.scale = area_cm2

        cm = 1e-6 * model.cm.to(device=device) * area_cm2  # convert from µF / cm2 to F
        self.cmdt = (
            (cm / dt_s).expand(model.shape).reshape(B, self.K)
        )  # (B,N) (F/s = S)

        # extracellular
        # We will need the original node IDs from the graph for this
        # Assuming G.nodes() provides the original order [0, 1, ..., N-1]
        original_nodes = list(range(model.graph.number_of_nodes()))
        original_idx_of = {n: i for i, n in enumerate(original_nodes)}

        # --- Create edge indices in the ORIGINAL node order ---
        edge_child_orig_list = []
        edge_parent_orig_list = []
        edge_gax_orig_list = []

        node_order_list = node_order

        for child_node, _ in model.graph.nodes(data=True):
            preds = list(model.graph.predecessors(child_node))
            if not preds:
                continue  # Skip root nodes

            parent_node = preds[0]

            # Get the ORIGINAL index (0 to N-1) of the parent and child
            child_idx_orig = original_idx_of[child_node]
            parent_idx_orig = original_idx_of[parent_node]

            edge_child_orig_list.append(child_idx_orig)
            edge_parent_orig_list.append(parent_idx_orig)

            # a_geom_t is in solver_order, so we need to find the child's
            # index in the solver order to get its correct conductance.
            # We can use the list `node_order_list` returned from graph_to_parent_and_axial
            # which maps solver_order_index -> original_node_id
            solver_idx_of_child = node_order_list.index(child_node)
            edge_gax_orig_list.append(a_geom_t[solver_idx_of_child])

        # Convert lists to tensors and register them as buffers
        self.register_buffer(
            "edge_child_orig",
            torch.tensor(edge_child_orig_list, dtype=torch.int64, device=device),
        )
        self.register_buffer(
            "edge_parent_orig",
            torch.tensor(edge_parent_orig_list, dtype=torch.int64, device=device),
        )
        self.register_buffer(
            "edge_gax_orig",
            torch.tensor(edge_gax_orig_list, dtype=a_geom_t.dtype, device=device),
        )

    def step(self, model, dt, ve=None, intra=None):
        model.v = self._step(model.v, dt, model.celsius, ve, intra)

    def _step(self, v, dt, temp, ve=None, intra=None):
        v = self.mech.update_v(v, dt)  # apply voltage processes

        self.mech.advance(v, dt, temp)

        if self.mech.currents:
            itot, gtot = self.mech.i(v)
            f_n = (gtot * v - itot).view(-1, self.K) * self.scale
            gtot = gtot.view(-1, self.K)
        else:
            gtot = torch.tensor(0.0, dtype=v.dtype, device=v.device)
            f_n = torch.tensor(0.0, dtype=v.dtype, device=v.device)

        if ve is not None:
            I_edge = _edge_currents(
                self.edge_child_orig, self.edge_parent_orig, self.edge_gax_orig, ve
            )  # (B, E) mA
            S = torch.zeros_like(f_n)  # (B, K)
            S.scatter_add_(
                1, self.edge_child_orig.expand_as(I_edge), -I_edge
            )  # child gets -I
            S.scatter_add_(
                1, self.edge_parent_orig.expand_as(I_edge), I_edge
            )  # parent gets +I
            f_n = f_n + S  # (B, K) mA

        if intra is not None:
            f_n += intra

        RHS = f_n + (self.cmdt * v.view(-1, self.K))  # mA
        main = self.cmdt + (gtot * self.scale)  # S

        d_ = main.index_select(-1, self.solver_order)  # (B, N)
        b_ = RHS.index_select(-1, self.solver_order)  # (B, N)
        a = self.a_geom  # (B, N) axial conductance

        v_out = self.solve(
            d_,
            a,
            b_,
            self.parent_idx,
            self.order,
            self.layer_ptr,
        )

        v = v_out.index_select(-1, self.inv_solver_order).reshape(
            self.base_shape
        )  # (B, N)

        return v  # (B, N) mV


class _dhs_multi(Integrator):
    """
    Multi-model DHS integrator.

    Packs all morphology/geometry into padded, flat buffers with a global
    row pitch K_stride = max(K_g) and records per-group offsets so a single
    GPU kernel can process all groups in one launch.
    """

    def __init__(
        self, model, mech, imem=None, threads: int = 16, write_back: bool = True
    ):
        from ..multi import MultiPopulation

        assert THREADS_PER_WARP % threads == 0, "threads must divide 32 (warp size)"
        assert threads <= 32, "threads must be ≤ 32 (warp size)"
        assert len(model) > 0 and isinstance(model, MultiPopulation), (
            "model must be a non-empty MultiPopulation instance"
        )

        super().__init__(model, mech, imem)

        self.threads = threads
        self.write_back = write_back
        self.split_at = None
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
        self.solve = None

        # Step-time scratch planes (allocated in initialize)
        self._d_plane = None
        self._b_plane = None

        # --- plan & scratch caches (filled lazily in _step) ---
        self._plan_cache = {}  # key: (P, device) -> dict with tiled plan
        self._scratch_sig = None

    def _calc_splits(self, models):
        split_lengths = [m.numelc() for m in models]
        self.split_at = torch.cumsum(torch.tensor(split_lengths), dim=0)[:-1].tolist()

    def initialize(self, models, dt: float):
        assert len(models) == self.num_groups

        if self.write_back:
            self._calc_splits(models)

        # --- device / dtype from first group ---
        dev0 = models.device()
        dtype0 = models.dtype()
        for m in models:
            assert m.device().type == dev0.type, "All models must be on the same device"
            assert m.dtype() == dtype0, "All models must share the same dtype"
        self.to(dev0)

        # Backend
        if dev0.type == "cuda":
            self.solve = partial(dhs_solve_multi_cuda, threads=self.threads)
        else:
            if not AXONML_SOLVERS_AVAILABLE:
                raise ImportError(
                    "DHS integrator requires axonml_solvers package for CPU execution. "
                    "Please install it with `pip install axonml_solvers`."
                )
            self.solve = torch.ops.axonml_solvers.dhs_multi_solve

        dt_s = dt * 1e-3

        P_list, ORDER_list, LPTR_list = [], [], []
        L_list = []
        SOLVER_list, INV_SOLVER_list = [], []
        B_list, K_list = [], []
        a_rows, cmdt_rows, scale_rows = [], [], []
        row_off, mech_off, solver_off = [], [], []
        row_cursor = 0
        mech_cursor = 0
        solver_cursor = 0

        # ---------- per-group topology & params ----------
        for g, mdl in enumerate(models):
            K_g = int(mdl.shape[-1])
            B_g = int(mdl.shape[-2]) if len(mdl.shape) > 1 else 1

            n_batch_dims = len(mdl.shape) - 2
            if n_batch_dims > 0:
                selection = tuple([0] * n_batch_dims)

            self.base_shapes.append((B_g, K_g))
            B_list.append(B_g)
            K_list.append(K_g)

            parent_idx_t, a_geom_t, node_order = graph_to_parent_and_axial(
                mdl.graph, dtype_axial=mdl.dtype()
            )
            parent_idx, _, depth = build_morphology(parent_idx_t.tolist())
            order_g, layer_ptr_g = build_dhs_layers(depth, self.threads)

            solver_order_g = torch.as_tensor(node_order, dtype=torch.int64, device=dev0)
            inv_solver_g = torch.argsort(solver_order_g, dim=0)

            area_cm2 = mdl.area.to(device=dev0, dtype=mdl.dtype())  # (1, K_g)
            cm = 1e-6 * mdl.cm.to(device=dev0, dtype=mdl.dtype())
            if n_batch_dims > 0:
                cm = cm[selection]
            cm = cm * area_cm2
            cmdt_g = (cm / dt_s).expand(B_g, -1).contiguous()  # (B_g, K_g)
            a_geom_g = a_geom_t.expand(B_g, -1).to(device=dev0, dtype=mdl.dtype())
            scale_g = area_cm2.expand(B_g, -1).contiguous()  # (B_g, K_g)

            P_list.append(parent_idx.to(dtype=torch.int64, device=dev0))
            ORDER_list.append(order_g.to(dtype=torch.int64, device=dev0))
            LPTR_list.append(layer_ptr_g.to(dtype=torch.int64, device=dev0))
            L_list.append(int(layer_ptr_g.numel() - 1))

            SOLVER_list.append(solver_order_g)
            INV_SOLVER_list.append(inv_solver_g)

            a_rows.append(a_geom_g)
            cmdt_rows.append(cmdt_g)
            scale_rows.append(scale_g)

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

        a_flat = torch.zeros((self.B_total, self.K_stride), device=dev0, dtype=dtype0)
        c_flat = torch.zeros((self.B_total, self.K_stride), device=dev0, dtype=dtype0)
        s_flat = torch.zeros((self.B_total, self.K_stride), device=dev0, dtype=dtype0)
        for g, (B_g, K_g) in enumerate(zip(B_list, K_list)):
            r0 = int(self.ROW_OFF[g])
            r1 = r0 + B_g
            a_flat[r0:r1, :K_g].copy_(a_rows[g])  # solver order
            c_flat[r0:r1, :K_g].copy_(cmdt_rows[g])  # mechanism order
            s_flat[r0:r1, :K_g].copy_(scale_rows[g])  # mechanism order
        self.register_buffer("a_geom_flat", a_flat)
        self.register_buffer("cmdt_flat", c_flat)
        self.register_buffer("scale_flat", s_flat)

        # ---------- vectorized mech↔solver mapping ----------
        mech_rows, cmdt_mech_flat, scale_mech_flat = [], [], []
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

            cmdt_mech_flat.append(self.cmdt_flat[r0:r1, :K_g].reshape(-1))
            scale_mech_flat.append(self.scale_flat[r0:r1, :K_g].reshape(-1))

        self.register_buffer("MECH_ROWS", torch.cat(mech_rows, 0))
        self.register_buffer("MECH_COLS_INV", torch.cat(mech_cols_inv, 0))
        self.register_buffer("CMDT_MECH", torch.cat(cmdt_mech_flat, 0))
        self.register_buffer("SCALE_MECH", torch.cat(scale_mech_flat, 0))

        self.N_mech = int(
            self.CMDT_MECH.numel()
        )  # total # of mechanism elements (sum_g B_g*K_g)
        PLANE_LIN_BASE = self.MECH_ROWS * self.K_stride + self.MECH_COLS_INV
        self.register_buffer("PLANE_LIN_BASE", PLANE_LIN_BASE.to(torch.long))

        # reset caches whenever we (re)initialize
        self._plan_cache.clear()
        self._scratch_sig = None

        # ---------- extracellular edge maps (for ve) ----------
        edge_child_idx_flat_all = []
        edge_parent_idx_flat_all = []
        edge_gax_flat_all = []

        for g, (mdl, B_g, K_g) in enumerate(zip(models, self.group_B, self.group_K)):
            G = mdl.graph

            # Build mapping: node_label -> solver index (fast lookup)
            # `self.SOLVER_cat[soff:soff+K_g]` maps solver_index -> original_node_label
            soff = int(self.SOLVER_OFF[g])
            solver_nodes = self.SOLVER_cat[soff : soff + K_g].tolist()
            solver_idx_of = {node_label: s for s, node_label in enumerate(solver_nodes)}

            # Edge lists in ORIGINAL node order
            edge_child = []
            edge_parent = []
            edge_gax = []

            # If nodes are 0..K_g-1 (typical), this is O(E). If labels differ, this still works.
            for child_node, _data in G.nodes(data=True):
                preds = list(G.predecessors(child_node))
                if not preds:
                    continue
                parent_node = preds[0]

                # original indices (mechanism columns) are the node labels
                child_idx_orig = int(child_node)
                parent_idx_orig = int(parent_node)

                edge_child.append(child_idx_orig)
                edge_parent.append(parent_idx_orig)

                # axial for the child's connection: take from a_geom_t at child's solver index
                s_child = solver_idx_of[child_node]
                edge_gax.append(
                    a_rows[g][0, s_child].item()
                )  # (B_g,K_g) -> any row, same geom

            if len(edge_child) == 0:
                continue  # degenerate, no edges

            edge_child = torch.tensor(
                edge_child, dtype=torch.int64, device=dev0
            )  # (E_g,)
            edge_parent = torch.tensor(
                edge_parent, dtype=torch.int64, device=dev0
            )  # (E_g,)
            edge_gax = torch.tensor(edge_gax, dtype=dtype0, device=dev0)  # (E_g,)

            # Repeat across batch rows and convert to FLAT mechanism indices
            rows = torch.arange(B_g, device=dev0, dtype=torch.int64).repeat_interleave(
                edge_child.numel()
            )  # (B_g*E_g,)
            child_cols = edge_child.repeat(B_g)  # (B_g*E_g,)
            parent_cols = edge_parent.repeat(B_g)  # (B_g*E_g,)
            g_mech_off = int(self.MECH_OFF[g])

            child_flat = g_mech_off + rows * K_g + child_cols
            parent_flat = g_mech_off + rows * K_g + parent_cols
            gax_flat = edge_gax.repeat(B_g)  # (B_g*E_g,)

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
            self.register_buffer("EDGE_GAX_FLAT", torch.cat(edge_gax_flat_all, 0))
        else:
            # empty placeholders
            self.register_buffer(
                "EDGE_CHILD_IDX_FLAT", torch.empty(0, dtype=torch.int64, device=dev0)
            )
            self.register_buffer(
                "EDGE_PARENT_IDX_FLAT", torch.empty(0, dtype=torch.int64, device=dev0)
            )
            self.register_buffer(
                "EDGE_GAX_FLAT", torch.empty(0, dtype=dtype0, device=dev0)
            )

        # ---------- allocate step scratch ----------
        self._d_plane = torch.empty(
            (self.B_total, self.K_stride), device=dev0, dtype=dtype0
        )
        self._b_plane = torch.empty(
            (self.B_total, self.K_stride), device=dev0, dtype=dtype0
        )

        P = int(np.prod(models.shape[:-2])) if len(models.shape) > 2 else 1
        self.P = P
        self._get_tiled_plan(P, dev0)

        if models.is_batched():
            Btot, Kstride = self.B_total, self.K_stride
            rows_total = P * Btot
            if (
                self._d_plane is None
                or self._d_plane.shape[0] != rows_total
                or self._d_plane.shape[1] != Kstride
            ):
                self._d_plane = torch.empty(
                    (rows_total, Kstride), device=dev0, dtype=dtype0
                )
                self._b_plane = torch.empty_like(self._d_plane)

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
                a_geom_eff=self.a_geom_flat,  # (Btot, Kstride)
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
            # Tile geometry rows and warp plan P times
            a_geom_eff = self.a_geom_flat.repeat(
                P, 1
            ).contiguous()  # materialize – kernel expects proper row stride

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

    def init_v(self, model):
        model.v = model.v_init.expand_as(model.v).clone().detach().contiguous()
        if self.write_back:
            self._calc_splits(model)
            _write_back(model, self.split_at)
        if self.imem:
            model.i_membrane = torch.zeros(
                model.i_membrane.shape,
                dtype=model.i_membrane.dtype,
                device=model.i_membrane.device,
            ).detach()

    def step(self, model, dt, ve=None, intra=None):
        model.v = self._step(model.v, dt, getattr(model, "celsius", None), ve, intra)
        if self.write_back:
            _write_back(model, self.split_at)

    def _step(self, v, dt, temp=None, ve=None, intra=None):
        if self.solve is None:
            raise NotImplementedError("Multi-morph kernel not connected.")

        orig_shape = v.shape  # keep whatever the caller gave us
        assert orig_shape[-2] == 1, "expect shape (..., 1, N_total)"
        N_total = orig_shape[-1]

        # Flatten leading batch dims into P
        P = self.P

        # 1) mechanisms on flattened vector
        v = self.mech.update_v(v, dt)
        self.mech.advance(v, dt, temp)
        itot_flat, gtot_flat = self.mech.i(v)  # both original shape

        intra_flat = 0.0 if intra is None else intra.reshape(P, N_total)

        SCALE_MECH = self.SCALE_MECH.reshape(1, -1)

        # 2) assemble in mechanism order
        f_n_flat = (gtot_flat * v - itot_flat).reshape(
            P, N_total
        ) * SCALE_MECH + intra_flat

        # --- extracellular coupling (ve), vectorized on flattened indices) ---
        if ve is not None and self.EDGE_CHILD_IDX_FLAT.numel() > 0:
            ve_flat = ve.reshape(1, -1)
            # edge potential differences per (row, edge)
            dV_edge = ve_flat.index_select(
                1, self.EDGE_PARENT_IDX_FLAT
            ) - ve_flat.index_select(1, self.EDGE_CHILD_IDX_FLAT)
            I_edge = dV_edge * self.EDGE_GAX_FLAT  # (sum_g B_g*E_g,)

            S_flat = torch.zeros_like(f_n_flat)
            S_flat.scatter_add_(1, self.EDGE_CHILD_IDX_FLAT, -I_edge)
            S_flat.scatter_add_(1, self.EDGE_PARENT_IDX_FLAT, I_edge)
            f_n_flat = f_n_flat + S_flat

        CMDT_MECH = self.CMDT_MECH.reshape(1, -1)

        RHS_flat = f_n_flat + CMDT_MECH * v.reshape(P, N_total)
        MAIN_flat = CMDT_MECH + gtot_flat * SCALE_MECH

        device = v.device
        plan = self._get_tiled_plan(P, device)
        PLIN_flat = plan["PLIN_flat"]

        # Zero & scatter by linear indices (fast, vectorized)
        self._b_plane.view(-1).zero_().index_copy_(0, PLIN_flat, RHS_flat.view(-1))
        self._d_plane.view(-1).zero_().index_copy_(0, PLIN_flat, MAIN_flat.view(-1))

        v_out_solver = self.solve(
            self._d_plane,
            plan["a_geom_eff"],
            self._b_plane,
            self.P_cat,
            self.ORDER_cat,
            self.LAYER_PTR_cat,
            plan["WARP_P_OFF"],
            plan["WARP_ORDER_OFF"],
            plan["WARP_LPTR_OFF"],
            plan["WARP_L"],
            plan["WARP_ROW_BASE"],
            plan["WARP_ROW_COUNT"],
            K_stride=self.K_stride,
            L_max=self.L_max,
            grid_x=plan["grid_x"],
        )  # -> (rows_total, K_stride)

        # 6) gather back to mechanism order with the same index map
        v_sel = v_out_solver.view(-1).index_select(0, PLIN_flat)  # (P * N_total,)
        return v_sel.reshape(orig_shape)


@torch.compile
def _write_back(model, split_at):
    # write v back to the constituent populations
    v_f = model.v
    splits = torch.tensor_split(v_f, split_at, dim=-1)
    for split, pop in zip(splits, model.populations.values()):
        pop.v = split.reshape_as(pop.v)
