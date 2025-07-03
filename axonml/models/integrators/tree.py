from collections import deque
from functools import partial
from typing import NamedTuple, List, Tuple

import numpy as np
import torch, networkx as nx

from .core import Integrator
from .triton import dhs_solve_cuda

from axonml.helpers import tic, toc

try:
    import axonml_solvers
    AXONML_SOLVERS_AVAILABLE = True
except ImportError:
    AXONML_SOLVERS_AVAILABLE = False


def build_morphology(parent_idx: List[int]) -> Tuple[
        torch.Tensor,     # parent (K,)
        List[List[int]],  # children adjacency
        torch.Tensor,     # depth (K,)
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

    return (torch.as_tensor(parent_idx, dtype=torch.int32),
            children,
            depth)


def graph_to_parent_and_axial(
    G: nx.DiGraph,
    dtype_axial: torch.dtype = torch.float32
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
    nodes   = list(nx.topological_sort(G))          # length K
    idx_of  = {n: i for i, n in enumerate(nodes)}
    K       = len(nodes)

    parent_idx = np.full(K, -1, dtype=np.int32)
    a_geom     = np.zeros(K, dtype=np.float32)

    # constant: 1 µm = 1 e‑4 cm
    microns_to_cm = 1e-4
    pi = np.pi

    # ------------------------------------------------------------------
    # 1. iterate over all nodes except the roots
    # ------------------------------------------------------------------
    for child in nodes:
        i = idx_of[child]
        preds = list(G.predecessors(child))

        if not preds:                      # soma / root compartment
            continue
        if len(preds) > 1:
            raise ValueError(
                f"Node {child} has {len(preds)} parents — "
                "morphology must be a rooted tree for the Hines matrix."
            )

        parent = preds[0]
        p      = idx_of[parent]
        parent_idx[i] = p

        # --------------------------------------------------------------
        # 1a.  Attempt to use the pre‑computed exact resistance
        # --------------------------------------------------------------
        edge_data = G.get_edge_data(parent, child, default={})
        R_total = edge_data.get('R_ohm', None)      # Ω or None

        # --------------------------------------------------------------
        # 1b.  If not present, fall back to geometric half‑segment calc
        # --------------------------------------------------------------
        if R_total is None:
            try:
                # child geometry
                L_i    = G.nodes[child]['L']    * microns_to_cm   # cm
                d_i_cm = G.nodes[child]['diam'] * microns_to_cm
                r_i_cm = 0.5 * d_i_cm
                rho_i  = G.nodes[child]['Ra']                     # Ω·cm

                # parent geometry
                L_p    = G.nodes[parent]['L']    * microns_to_cm
                d_p_cm = G.nodes[parent]['diam'] * microns_to_cm
                r_p_cm = 0.5 * d_p_cm
                rho_p  = G.nodes[parent]['Ra']                   # Ω·cm
            except KeyError as err:
                raise KeyError(
                    f"Missing geometry attribute {err} on node; "
                    "cannot compute axial resistance and no R_ohm present "
                    "on the edge."
                ) from err

            R_half_i = rho_i * (L_i / 2) / (pi * r_i_cm**2)
            R_half_p = rho_p * (L_p / 2) / (pi * r_p_cm**2)
            R_total  = R_half_i + R_half_p           # Ω

        # --------------------------------------------------------------
        # 1c.  Store axial conductance  (Siemens = 1 / Ω)
        # --------------------------------------------------------------
        a_geom[i] = 1.0 / R_total

    # ------------------------------------------------------------------
    # 2. cast to torch tensors
    # ------------------------------------------------------------------
    parent_idx_t = torch.as_tensor(parent_idx, dtype=torch.int32)
    a_geom_t     = torch.as_tensor(a_geom,     dtype=dtype_axial)

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
    max_d     = int(depth_cpu.max())

    order:     List[int]  = []
    layer_ptr: List[int]  = [0]

    # bucket sort by depth
    bins: List[List[int]] = [[] for _ in range(max_d + 1)]
    for i, d in enumerate(depth_cpu):
        bins[d].append(i)

    # deepest → root
    for d in range(max_d, -1, -1):
        bucket = bins[d]
        # slice bucket into chunks of at most k
        for s in range(0, len(bucket), k_threads):
            chunk = bucket[s:s + k_threads]
            order.extend(chunk)
            layer_ptr.append(len(order))

    return (torch.as_tensor(order,     dtype=torch.int32),
            torch.as_tensor(layer_ptr, dtype=torch.int32))


def get_area_from_graph(G: nx.DiGraph) -> torch.Tensor:
    """
    Extracts the area from the graph's nodes if available.

    Parameters
    ----------
    G : nx.DiGraph
        The directed graph representing the tree structure.

    Returns
    -------
    torch.Tensor or None
        A tensor containing the area in µm² for each node, or None if not available.
    """
    areas = []
    for n in range(len(G.nodes)):
        area = G.nodes[n].get('area', None)
        if area is not None:
            areas.append(area)
        else:
            return None  # If any node lacks area, return None

    return torch.tensor(areas, dtype=torch.float32)


def _edge_currents(
        edge_child: torch.Tensor, 
        edge_parent: torch.Tensor,
        edge_gax: torch.Tensor,
        ve: torch.Tensor
    ): # (B,K)
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

    def __init__(self, model, mech, imem=None, threads=32):
        super().__init__(model, mech, imem)
        self.threads = threads

        B, N = model.np, model.nc

        self.register_buffer("parent_idx",  torch.empty(N, dtype=torch.int32))  # (N,)
        self.register_buffer("order",       torch.empty(N, dtype=torch.int32))  # (N,) order of forward elimination

        self.register_buffer("lower",       torch.empty(B, N))    # (B,N) lower diagonal

        self.register_buffer("solver_order",        torch.empty(N, dtype=torch.int64))  # (N,) node order for the graph
        self.register_buffer("inv_solver_order",    torch.empty(N, dtype=torch.int64))  # (N,) inverse node order
        self.register_buffer("scale",               torch.empty(B, N))    # (N,) scale factor

        self.register_buffer("a_geom",          torch.empty(B, N))    # (B,N) axial conductance
        self.register_buffer("cmdt",            torch.empty(B, N))    # (B,N) capacitance * dt


    def initialize(self, model, dt):
        B, N = model.np, model.nc
        dt_s = dt * 1e-3

        device = model.device()

        if device.type == 'cpu' and not AXONML_SOLVERS_AVAILABLE:
            raise ImportError(
                "DHS integrator requires axonml_solvers package for CPU execution. "
                "Please install it with `pip install axonml_solvers`."
            )
        
        if device.type == 'cuda':
            self.solve = partial(
                dhs_solve_cuda,
                threads=self.threads
            )
        elif device.type == 'cpu':
            self.solve = torch.ops.axonml_solvers.dhs_solve
        else:
            raise NotImplementedError(
                f"DHS integrator is not implemented for device type {device.type}."
            )

        parent_idx_t, a_geom_t, node_order = graph_to_parent_and_axial(model.graph)
        parent_idx, children, depth = build_morphology(parent_idx_t.tolist())
        order, layer_ptr = build_dhs_layers(depth, self.threads)

        self.solver_order.copy_(torch.as_tensor(node_order, dtype=torch.int64, device=device))   # (N,)
        self.inv_solver_order.copy_(torch.argsort(self.solver_order, dim=0))        # (N,)

        radius_cm = 1e-4 * model.diam / 2.0                 # µm → cm   (N,)
        dx_cm     = 1e-4 * model.dx                         # µm → cm   (N,)
        area_cm2  = 2 * torch.pi * radius_cm * dx_cm        # cm²

        area_um2 = get_area_from_graph(model.graph).to(device=device)
        if area_um2 is not None:
            area_cm2 = 1e-8 * area_um2                      # convert from µm² to cm²

        self.register_buffer("layer_ptr", layer_ptr.to(device))  # (L+1,)
        self.order.copy_(order.to(dtype=torch.int32, device=device))
        self.parent_idx.copy_(parent_idx.to(dtype=torch.int32, device=device))  # (N,)
        self.a_geom.copy_(a_geom_t.expand(B, -1))

        K = parent_idx.numel()

        # add children contributions to their parent’s diagonal
        valid = parent_idx >= 0

        self.scale.copy_(area_cm2)

        cm = model.cm * 1e-6 * area_cm2      # convert from µF to F
        self.cmdt.copy_(cm / dt_s)           # (B,N) (F/s = S)

        # extracellular
        parent = parent_idx                                         # (N,)
        child  = (parent >= 0).nonzero(as_tuple=False).squeeze(1)   # (E,)
        self.register_buffer("edge_child",  child.to(torch.int64))  # (E,) child indices in solver order
        self.register_buffer("edge_parent", parent[child].to(torch.int64))  # (E,) parent indices in solver order

        # axial conductance per edge (S) already in a_geom_t[child]
        self.register_buffer("edge_gax", a_geom_t[child])  # (E,) axial conductance in solver order

    def step(self, model, dt, ve=None, intra=None):
        model.v = self._step(model.v, dt, model.celsius, ve, intra)

    def _step(self, v, dt, temp, ve=None, intra=None):
        self.mech.advance(v, dt, temp)
        itot, gtot = self.mech.i(v)

        f_n  = (gtot * v - itot) * self.scale

        if ve is not None:
            I_edge = _edge_currents(
                self.edge_child,
                self.edge_parent,
                self.edge_gax,
                ve
            ) # (B, E) mA
            S = torch.zeros_like(f_n) # (B, K)
            S.scatter_add_(1, self.edge_child .expand_as(I_edge), -I_edge)  # child gets -I
            S.scatter_add_(1, self.edge_parent.expand_as(I_edge),  I_edge)  # parent gets +I
            f_n = f_n + S  # (B, K) mA

        if intra is not None:
            f_n += intra

        RHS  = f_n + self.cmdt * v                      # mA
        main = self.cmdt + gtot * self.scale            # S

        d_ = main.index_select(-1, self.solver_order)   # (B, N)
        b_ = RHS.index_select(-1, self.solver_order)    # (B, N)
        a  = self.a_geom                                # (B, N) axial conductance

        v_out = self.solve(
            d_, a, b_,
            self.parent_idx.to(d_.device, dtype=torch.int64),
            self.order.to(d_.device, dtype=torch.int64),
            self.layer_ptr.to(d_.device, dtype=torch.int64),
        )

        v = v_out.index_select(-1, self.inv_solver_order)  # (B, N)

        return v  # (B, N) mV


class _df_branched(Integrator):
    """
    Dufort-Frankel integrator for a branched (tree) morphology.
    """

    __constants__ = [
        "beta", "smoothing", "smooth_every", "imem", 
        "parent_idx", "g_axial_sum", "coeff1_inv", "coeff2"
    ]

    def __init__(self, model, mech, beta=1.0, smooth_every=100, imem=None):
        super().__init__(model, mech, imem)

        B, N = model.np, model.nc

        # Previous voltage state needed for DF
        model.register_buffer(
            "v_prev", torch.full((B, N), model.v_init)
        )

        # Morphological and pre-computed coefficients
        self.register_buffer("parent_idx",  torch.empty(N, dtype=torch.int64))
        self.register_buffer("a_geom",      torch.empty(B, N))
        self.register_buffer("g_axial_sum", torch.empty(B, N))
        self.register_buffer("coeff1_inv",  torch.empty(B, N))
        self.register_buffer("coeff2",      torch.empty(B, N))
        self.register_buffer("area",        torch.empty(B, N))
        self.register_buffer("s1",          torch.empty(B, N)) # For i_cap calculation

        self.smooth_every = smooth_every
        self.beta = beta
        self.smoothing = bool((1 - beta))

        if self.smoothing:
            self.filter = torch.nn.Conv1d(
                B, B, 5, padding=2, bias=False, padding_mode="reflect", groups=B
            )
            filter_weights = torch.tensor([0.25, 0.25, 0, 0.25, 0.25], dtype=torch.float)
            self.filter.weight.data = filter_weights.reshape(1, 1, 5).repeat(B, 1, 1)
            for p in self.filter.parameters():
                p.requires_grad = False

    def initialize(self, model, dt):
        B, N = model.np, model.nc
        device = model.device()
        
        # 1. Get morphology from graph (parent indices and axial conductances)
        # We need the graph in the solver's topological order
        parent_idx_t, a_geom_t, node_order = graph_to_parent_and_axial(model.graph)
        self.parent_idx.copy_(parent_idx_t.to(device, dtype=torch.int64))
        self.a_geom.copy_(a_geom_t.to(device).expand(B, -1))

        # 2. Calculate compartment area and capacitance
        # Using model.diam and model.dx, assuming they are in µm
        radius_cm = 1e-4 * model.diam / 2.0
        dx_cm = 1e-4 * model.dx
        area_cm2 = 2 * torch.pi * radius_cm * dx_cm
        self.area.copy_(area_cm2.expand(B, -1))
        
        # cm is in µF/cm^2, convert to F
        cm_F = (model.cm * 1e-6 * area_cm2).expand(B, -1) 

        # 3. Pre-compute sum of all axial conductances for each compartment
        # g_axial_sum[i] = g_parent_i + sum(g_children_of_i)
        g_sum = torch.zeros_like(self.a_geom)
        
        # Add parent conductance
        g_sum += self.a_geom

        # Add children conductances via scatter_add
        # For each child `i`, its `a_geom[i]` is the conductance to its parent.
        # So we scatter `a_geom` values to the parent indices.
        valid_mask = self.parent_idx >= 0
        parent_indices = self.parent_idx[valid_mask].unsqueeze(0).expand(B, -1)
        child_conductances = self.a_geom[:, valid_mask]
        g_sum.scatter_add_(1, parent_indices, child_conductances)
        self.g_axial_sum.copy_(g_sum)

        # 4. Pre-compute DF coefficients
        # Based on: v_new * C1 = v_prev * C2 + (neighbor terms) - I_ion
        c_term = cm_F / (2 * dt * 1e-3) # dt is in ms
        g_term = self.g_axial_sum / 2.0

        coeff1 = c_term + g_term
        self.coeff1_inv.copy_(1.0 / coeff1)
        self.coeff2.copy_(c_term - g_term)
        
        # Other constants
        self.s1.copy_(2 * (dt * 1e-3) / cm_F) # For i_cap, in s/F

    def step(self, model, dt, ve=None, intra=None):
        # Note: Dufort-Frankel does not naturally handle ve in this form.
        # This implementation ignores ve. A more complex derivation would be needed.
        if ve is not None:
            print("Warning: Branched Dufort-Frankel does not currently support ve. It will be ignored.")
            
        model.v, model.v_prev = self._step_intra(
            model.v,
            model.v_prev,
            dt,
            model.celsius,
            intra if intra is not None else torch.zeros_like(model.v),
            model.t_ind
        )

    def _step_intra(self, v, v_prev, dt, temp, intra, t_ind: int):
        B, N = v.shape
        device = v.device
        
        # 1. Calculate semi-implicit ionic current and conductance
        # This part is identical to your original DF solver
        i_ion, gtot = self.mech.idf(v, v_prev) # i_ion is in mA/cm^2, gtot in S/cm^2
        i_ion_total = i_ion * self.area    # Convert to total current (mA)
        gtot_total = gtot * self.area      # Convert to total conductance (S)

        # 2. Calculate the sum of neighboring voltages weighted by conductance
        # Σ g_neighbor * v_neighbor
        sum_g_v_neighbors = torch.zeros_like(v)

        # Contribution from parent: g_p,i * v_p
        valid_mask = self.parent_idx >= 0
        parent_indices = self.parent_idx[valid_mask]
        parent_v = v[:, parent_indices]
        parent_g = self.a_geom[:, valid_mask]
        sum_g_v_neighbors[:, valid_mask] += parent_g * parent_v
        
        # Contribution from children: Σ g_c,i * v_c
        # For each child `i`, its contribution `a_geom[i] * v[i]` goes to its parent.
        # This is another scatter_add.
        parent_indices_expanded = parent_indices.unsqueeze(0).expand(B, -1)
        source_vals = self.a_geom[:, valid_mask] * v[:, valid_mask]
        sum_g_v_neighbors.scatter_add_(1, parent_indices_expanded, source_vals)
        
        # 3. Assemble the full update equation
        # v_new * (C1 + 0.5*gtot) = v_prev * (C2 - 0.5*gtot) + sum_g_v_neighbors - (i_ion - intra)
        # Note: The gtot term comes from a semi-implicit handling of I_ion.
        # I_ion_new ≈ I_ion_old + gtot * (v_new - v_old)
        # The DF scheme uses (v_new - v_prev) / (2*dt), leading to the 0.5*gtot factor.
        
        numerator = (
            v_prev * (self.coeff2 - 0.5 * gtot_total) 
            + sum_g_v_neighbors 
            - (i_ion_total - intra)
        )
        denominator_inv = 1.0 / (1.0/self.coeff1_inv + 0.5 * gtot_total)
        
        v_new = numerator * denominator_inv

        # 4. Advance gating variables for the next step
        self.mech.itot(v) # Update internal currents if needed
        self.mech.advance(v, dt, temp)

        # 6. Optional membrane current calculation
        if self.imem:
            i_cap = (v_new - v_prev) / self.s1
            self.i_membrane = i_cap + self.mech.imem * self.area

        return v_new, v
        
    def init_v(self, model):
        model.v = torch.full(model.v.shape, model.v_init, dtype=model.v.dtype, device=model.v.device)
        model.v.detach_()
        model.v_prev = torch.full(model.v_prev.shape, model.v_init, dtype=model.v_prev.dtype, device=model.v_prev.device)
        model.v_prev.detach_()
        if self.imem:
            model.i_membrane = torch.zeros(model.i_membrane.shape, dtype=model.i_membrane.dtype, device=model.i_membrane.device)
            model.i_membrane.detach_()

    def detach(self, model):
        model.v.detach_()
        model.v_prev.detach_()
        if self.imem:
            model.i_membrane.detach_()
        self.mech.detach()