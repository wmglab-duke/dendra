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
        B = self.B
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

        area_cm2 = model.area  # cm²

        self.register_buffer(
            "layer_ptr", layer_ptr.to(dtype=torch.int64, device=device)
        )  # (L+1,)
        self.order.copy_(order.to(dtype=torch.int64, device=device))
        self.parent_idx.copy_(parent_idx.to(dtype=torch.int64, device=device))  # (N,)
        self.a_geom = a_geom_t.expand(B, -1).to(
            device=device, dtype=model.dtype()
        )  # (B,N)

        self.scale = area_cm2

        cm = 1e-6 * model.cm * area_cm2  # convert from µF / cm2 to F
        self.cmdt = (cm / dt_s).expand(model.shape).view(B, self.K)  # (B,N) (F/s = S)

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
        itot, gtot = self.mech.i(v)

        f_n = (gtot * v - itot).view(-1, self.K) * self.scale

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
        main = self.cmdt + (gtot.view(-1, self.K) * self.scale)  # S

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
    Multi-model DHS initializer.

    Packs all morphology/geometry into padded, flat buffers with a global
    row pitch K_stride = max(K_g) and records per-group offsets so a single
    GPU kernel can process all groups in one launch.

    After `initialize`, the following attributes are ready for a single-launch solve:
      - B_total, K_stride, L_max
      - Per-group B, K, L, offsets: ROW_OFF[g], MECH_OFF[g]
      - Concatenated topology tables: P_cat, ORDER_cat, LAYER_PTR_cat
      - Per-group topology offsets: P_OFF, ORDER_OFF, LPTR_OFF, L_per_group
      - Flat parameter planes (padded to K_stride): a_geom_flat, cmdt_flat, scale_flat
      - Per-group solver<->mechanism mappings: SOLVER_cat, INV_SOLVER_cat, SOLVER_OFF
    """

    def __init__(self, model: List, mech, imem=None, threads: int = 16):
        assert THREADS_PER_WARP % threads == 0, "threads must divide 32 (warp size)"
        assert threads <= 32, "threads must be ≤ 32 (warp size)"
        assert isinstance(model, (list, tuple)) and len(model) > 0, (
            "model must be a non-empty list of per-morph models"
        )

        # It is ok to pass the list upstream; Integrator typically stores it.
        super().__init__(model, mech, imem)

        self.threads = threads
        self.num_groups = len(model)

        # Basic per-group shapes (filled in initialize)
        self.group_B: List[int] = []
        self.group_K: List[int] = []
        self.group_L: List[int] = []

        # Global packed shapes (filled in initialize)
        self.B_total: int = 0
        self.K_stride: int = 0
        self.L_max: int = 0

        # Per-group row and mechanism offsets (filled in initialize)
        # ROW_OFF[g] = starting neuron row for group g in flat (B_total, K_stride) planes
        # MECH_OFF[g] = starting index in concatenated mechanism-order vector (sum B_g*K_g)
        self.ROW_OFF: torch.Tensor = None  # int64[num_groups]
        self.MECH_OFF: torch.Tensor = None  # int64[num_groups]

        # Topology (concatenated) + per-group offsets (registered in initialize)
        self.P_cat = None  # int64[sum K_g]
        self.ORDER_cat = None  # int64[sum K_g]
        self.LAYER_PTR_cat = None  # int64[sum (L_g+1)]
        self.P_OFF = None  # int64[num_groups]
        self.ORDER_OFF = None  # int64[num_groups]
        self.LPTR_OFF = None  # int64[num_groups]
        self.L_per_group = None  # int32[num_groups]

        # Solver<->mechanism maps (concatenated over groups)
        self.SOLVER_cat = None  # int64[sum K_g]  (solver_order per group, catted)
        self.INV_SOLVER_cat = None  # int64[sum K_g]
        self.SOLVER_OFF = None  # int64[num_groups]  (offsets into *_cat above)

        # Padded, flat parameter planes with row pitch K_stride
        # Shapes: (B_total, K_stride); dtype/device = first group's
        self.a_geom_flat = None
        self.cmdt_flat = None
        self.scale_flat = None

        # Keep per-group base shapes for convenient reshaping/scattering later
        self.base_shapes: List[Tuple[int, int]] = []  # [(B_g, K_g), ...]

        self.NPW = THREADS_PER_WARP // self.threads  # neurons per warp
        # Per-warp descriptors (set in initialize)
        self.WARP_P_OFF = self.WARP_ORDER_OFF = self.WARP_LPTR_OFF = None  # int64[W]
        self.WARP_L = None  # int32[W]
        self.WARP_ROW_BASE = None  # int64[W]  (starting row index in [0, B_total))
        self.WARP_ROW_COUNT = (
            None  # int32[W]  (1..NPW, how many valid neurons in this warp)
        )
        self.grid_x = 0  # number of warps to launch

        # (Optional future hookup)
        self.solve = None  # will be set in initialize if you plug in a kernel

    def initialize(self, models: List, dt: float):
        assert isinstance(models, (list, tuple)) and len(models) == self.num_groups

        # ---------------------------------------------------------------------
        # 0) Device / dtype sanity
        # ---------------------------------------------------------------------
        dev0 = models[0].device()
        dtype0 = models[0].dtype()
        for m in models:
            assert m.device().type == dev0.type, "All models must be on the same device"
            assert m.dtype() == dtype0, "All models must share the same dtype"

        self.to(dev0)

        # (Optional) choose solver backend — left as a placeholder per request
        if dev0.type == "cuda":
            # from .triton import dhs_solve_multi_cuda
            self.solve = partial(dhs_solve_multi_cuda, threads=self.threads)
        elif dev0.type == "cpu":
            # self.solve = torch.ops.axonml_solvers.dhs_solve_multi  # if/when available
            raise NotImplementedError("CPU support is not implemented.")
        else:
            raise NotImplementedError(f"Device type {dev0.type} is not supported.")

        # Time-step in seconds for cmdt
        dt_s = dt * 1e-3

        # Scratch collectors
        P_list, ORDER_list, LPTR_list = [], [], []
        L_list = []  # scalar L_g per group
        SOLVER_list, INV_SOLVER_list = [], []
        B_list, K_list = [], []

        # Parameter planes per group
        a_rows, cmdt_rows, scale_rows = [], [], []

        # Offsets
        row_off = []
        mech_off = []
        solver_off = []

        # Running cursors
        row_cursor = 0  # over B_total
        mech_cursor = 0  # over sum(B_g*K_g)
        p_cursor = 0  # over sum K_g
        o_cursor = 0  # over sum K_g
        lptr_cursor = 0  # over sum (L_g+1)
        solver_cursor = 0  # over sum K_g

        # ---------------------------------------------------------------------
        # 1) Per-group topology & parameters
        # ---------------------------------------------------------------------
        for g, model in enumerate(models):
            # Shapes
            K_g = int(model.shape[-1])
            B_g = int(np.prod(model.shape[:-1])) if len(model.shape) > 1 else 1

            B_list.append(B_g)
            K_list.append(K_g)
            self.base_shapes.append((B_g, K_g))

            # Morphology/geometry
            parent_idx_t, a_geom_t, node_order = graph_to_parent_and_axial(
                model.graph, dtype_axial=model.dtype()
            )
            parent_idx, _, depth = build_morphology(parent_idx_t.tolist())
            order_g, layer_ptr_g = build_dhs_layers(depth, self.threads)

            # Solver <-> mechanism maps
            solver_order_g = torch.as_tensor(
                node_order, dtype=torch.int64, device=dev0
            )  # (K_g,)
            inv_solver_g = torch.argsort(solver_order_g, dim=0)  # (K_g,)

            # Area, cm, cmdt, scale
            area_cm2 = model.area.to(device=dev0, dtype=model.dtype())  # (1, K_g)
            cm = (
                1e-6 * model.cm.to(device=dev0, dtype=model.dtype()) * area_cm2
            )  # (1, K_g) [F]
            cmdt_g = (cm / dt_s).expand(B_g, -1).contiguous()  # (B_g, K_g) [S]
            a_geom_g = a_geom_t.expand(B_g, -1).to(device=dev0, dtype=model.dtype())
            scale_g = area_cm2.expand(B_g, -1).contiguous()  # (B_g, K_g)

            # Stash topology (device int64)
            P_list.append(parent_idx.to(dtype=torch.int64, device=dev0))  # (K_g,)
            ORDER_list.append(order_g.to(dtype=torch.int64, device=dev0))  # (K_g,)
            LPTR_list.append(layer_ptr_g.to(dtype=torch.int64, device=dev0))  # (L_g+1,)
            L_list.append(int(layer_ptr_g.numel() - 1))

            # Stash solver maps
            SOLVER_list.append(solver_order_g)
            INV_SOLVER_list.append(inv_solver_g)

            # Stash parameter planes (will pad to K_stride later)
            a_rows.append(a_geom_g)
            cmdt_rows.append(cmdt_g)
            scale_rows.append(scale_g)

            # Record offsets
            row_off.append(row_cursor)
            mech_off.append(mech_cursor)
            solver_off.append(solver_cursor)

            row_cursor += B_g
            mech_cursor += B_g * K_g
            solver_cursor += K_g
            p_cursor += K_g
            o_cursor += K_g
            lptr_cursor += layer_ptr_g.numel()

        # ---------------------------------------------------------------------
        # 2) Global shapes & constants
        # ---------------------------------------------------------------------
        self.group_B = B_list
        self.group_K = K_list
        self.group_L = L_list

        self.B_total = int(sum(B_list))
        self.K_stride = int(max(K_list)) if len(K_list) > 0 else 0
        self.L_max = int(max(L_list)) if len(L_list) > 0 else 0

        # Offsets as buffers
        self.ROW_OFF = torch.as_tensor(row_off, dtype=torch.int64, device=dev0)  # (G,)
        self.MECH_OFF = torch.as_tensor(
            mech_off, dtype=torch.int64, device=dev0
        )  # (G,)
        self.SOLVER_OFF = torch.as_tensor(
            solver_off, dtype=torch.int64, device=dev0
        )  # (G,)

        # ---------------------------------------------------------------------
        # 3) Concatenate topology and solver maps; register as buffers
        # ---------------------------------------------------------------------
        P_cat = (
            torch.cat(P_list, dim=0)
            if len(P_list)
            else torch.empty(0, dtype=torch.int64, device=dev0)
        )
        ORDER_cat = (
            torch.cat(ORDER_list, dim=0)
            if len(ORDER_list)
            else torch.empty(0, dtype=torch.int64, device=dev0)
        )
        LAYER_PTR_cat = (
            torch.cat(LPTR_list, dim=0)
            if len(LPTR_list)
            else torch.empty(0, dtype=torch.int64, device=dev0)
        )
        SOLVER_cat = (
            torch.cat(SOLVER_list, dim=0)
            if len(SOLVER_list)
            else torch.empty(0, dtype=torch.int64, device=dev0)
        )
        INV_SOLVER_cat = (
            torch.cat(INV_SOLVER_list, dim=0)
            if len(INV_SOLVER_list)
            else torch.empty(0, dtype=torch.int64, device=dev0)
        )

        # Per-group offsets into concatenated topology
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

        # Register buffers so they move with .to()
        self.register_buffer("P_cat", P_cat)
        self.register_buffer("ORDER_cat", ORDER_cat)
        self.register_buffer("LAYER_PTR_cat", LAYER_PTR_cat)
        self.register_buffer("P_OFF", P_OFF)
        self.register_buffer("ORDER_OFF", ORDER_OFF)
        self.register_buffer("LPTR_OFF", LPTR_OFF)
        self.register_buffer("L_per_group", L_per_group)

        self.register_buffer("SOLVER_cat", SOLVER_cat)
        self.register_buffer("INV_SOLVER_cat", INV_SOLVER_cat)
        self.register_buffer("SOLVER_OFF", self.SOLVER_OFF)

        NPW = self.NPW
        warp_p_off, warp_o_off, warp_lptr_off = [], [], []
        warp_L, warp_row_base, warp_row_count = [], [], []

        for g in range(self.num_groups):
            B_g = int(self.group_B[g])
            n_warps_g = (B_g + NPW - 1) // NPW
            p_off = int(self.P_OFF[g].item())
            o_off = int(self.ORDER_OFF[g].item())
            lptr_off = int(self.LPTR_OFF[g].item())
            Lg = int(self.L_per_group[g].item())
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

        # Register buffers
        dev0 = self.a_geom_flat.device
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

        # ---------------------------------------------------------------------
        # 4) Allocate padded flat planes and materialize parameters
        #     Layout: rows enumerate neurons across all groups:
        #       rows [ROW_OFF[g] : ROW_OFF[g] + B_g] belong to group g,
        #       columns [0:K_g] active, [K_g:K_stride] are padding.
        # ---------------------------------------------------------------------
        if self.B_total == 0 or self.K_stride == 0:
            # Degenerate case; keep empty planes
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
            # Fill leading K_g columns; remainder stays zero (padding)
            a_flat[r0:r1, :K_g].copy_(a_rows[g])
            c_flat[r0:r1, :K_g].copy_(cmdt_rows[g])
            s_flat[r0:r1, :K_g].copy_(scale_rows[g])

        self.register_buffer("a_geom_flat", a_flat)  # (B_total, K_stride)
        self.register_buffer("cmdt_flat", c_flat)  # (B_total, K_stride)
        self.register_buffer("scale_flat", s_flat)  # (B_total, K_stride)

        mech_rows = []
        mech_cols_solver = []
        cmdt_mech_flat = []
        scale_mech_flat = []

        for g, (B_g, K_g) in enumerate(zip(self.group_B, self.group_K)):
            r0 = int(self.ROW_OFF[g])
            r1 = r0 + B_g
            # per-group (B_g, K_g) → 1D mechanism-order segment of length B_g*K_g
            # rows: repeat each row index K_g times
            rows_g = torch.arange(r0, r1, device=self.device).repeat_interleave(
                K_g
            )  # (B_g*K_g,)

            # cols in SOLVER order: solver permutation repeated across rows
            soff = int(self.SOLVER_OFF[g])
            solver_cols = self.SOLVER_cat[soff : soff + K_g]  # (K_g,)
            cols_g = solver_cols.repeat(B_g)  # (B_g*K_g,)

            mech_rows.append(rows_g)
            mech_cols_solver.append(cols_g)

            # pull active columns (0:K_g) from the padded planes, then flatten in mechanism order
            cmdt_mech_flat.append(self.cmdt_flat[r0:r1, :K_g].reshape(-1))
            scale_mech_flat.append(self.scale_flat[r0:r1, :K_g].reshape(-1))

        # Concatenate across groups once
        MECH_ROWS = torch.cat(mech_rows, dim=0)  # (sum B_g*K_g,)
        MECH_COLS_SOLVER = torch.cat(mech_cols_solver, dim=0)  # (sum B_g*K_g,)
        CMDT_MECH = torch.cat(cmdt_mech_flat, dim=0)  # (sum B_g*K_g,)
        SCALE_MECH = torch.cat(scale_mech_flat, dim=0)  # (sum B_g*K_g,)

        self.register_buffer("MECH_ROWS", MECH_ROWS)
        self.register_buffer("MECH_COLS_SOLVER", MECH_COLS_SOLVER)
        self.register_buffer("CMDT_MECH", CMDT_MECH)
        self.register_buffer("SCALE_MECH", SCALE_MECH)

    def step(self, model, dt, ve=None, intra=None):
        """
        model.v is a 1-D flattened vector in MECHANISM order, formed by
        concatenating all groups' (B_g, K_g) rows: [g0, g1, ..., g_{G-1}].
        After solving, we write back the same 1-D layout.
        """
        model.v = self._step(model.v, dt, getattr(model, "celsius", None), ve, intra)

    def _step(self, v_flat, dt, temp=None, ve=None, intra=None):
        if ve is not None:
            raise NotImplementedError("ve support is not wired yet in _dhs_multi.step")
        if self.solve is None:
            raise NotImplementedError("Multi-morph kernel not connected.")

        device, dtype = v_flat.device, v_flat.dtype
        Btot, Kstride = int(self.B_total), int(self.K_stride)

        # --- 1) Mechanisms on the flat, concatenated vector ---
        v_flat = self.mech.update_v(v_flat, dt)
        self.mech.advance(v_flat, dt, temp)
        itot_flat, gtot_flat = self.mech.i(v_flat)  # 1-D, mech order

        # Optional additive current in mech order
        if intra is None:
            intra_flat = 0.0
        else:
            intra_flat = intra

        # --- 2) Assemble RHS/main in mechanism order (all vectorized) ---
        # f_n = (gtot * v - itot) * scale + intra
        f_n_flat = (gtot_flat * v_flat - itot_flat) * self.SCALE_MECH + intra_flat
        RHS_flat = f_n_flat + self.CMDT_MECH * v_flat
        MAIN_flat = self.CMDT_MECH + gtot_flat * self.SCALE_MECH

        # --- 3) Scatter once into padded solver-order planes ---
        d_plane = torch.zeros((Btot, Kstride), device=device, dtype=dtype)
        b_plane = torch.zeros((Btot, Kstride), device=device, dtype=dtype)

        # elementwise (row, col) assignment; no loops
        b_plane[self.MECH_ROWS, self.MECH_COLS_SOLVER] = RHS_flat
        d_plane[self.MECH_ROWS, self.MECH_COLS_SOLVER] = MAIN_flat

        # --- 4) Single multi-morph solve (solver-order planes) ---
        v_out_solver = self.solve(
            d_plane,
            self.a_geom_flat,
            b_plane,
            self.P_cat,
            self.ORDER_cat,
            self.LAYER_PTR_cat,
            self.WARP_P_OFF,
            self.WARP_ORDER_OFF,
            self.WARP_LPTR_OFF,
            self.WARP_L,
            self.WARP_ROW_BASE,
            self.WARP_ROW_COUNT,
            K_stride=self.K_stride,
            L_max=self.L_max,
            threads=self.threads,
            grid_x=self.grid_x,
        )
        # Expected: (B_total, K_stride), solver order

        # --- 5) Gather back to mechanism order in one shot ---
        v_new_flat = v_out_solver[self.MECH_ROWS, self.MECH_COLS_SOLVER]

        return v_new_flat
