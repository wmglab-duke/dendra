from collections import deque
from typing import NamedTuple, List, Tuple

import numpy as np
import torch, networkx as nx

from axonml.models.mechanisms.compilers import MechCompiler, ImplicitCompiler
from axonml.models.mechanisms.handler.builders import ImplicitHandlerBuilder

from .core import Integrator
from .triton import dhs_solve
from axonml.helpers import tic, toc


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
    dtype_axial=torch.float32
    ):
    """
    Convert a compartmental morphology stored in a DiGraph to

        parent_idx : int32[K]
        a_geom     : fp32[K]   (static axial conductance)

    The node order (0 … K-1) follows a topological sort so that
    every parent appears before its children - exactly what the
    DHS pre-processing requires.
    """
    # topological order and index map
    nodes = list(nx.topological_sort(G))            # length K
    idx_of = {n: i for i, n in enumerate(nodes)}
    K = len(nodes)

    parent_idx = np.full(K, -1, dtype=np.int32)     # default −1
    a_geom     = np.zeros(K, dtype=np.float32)

    # per‑compartment geometry
    # 1 µm = 1e-4 cm
    microns_to_cm = 1e-4

    for child in nodes:
        i = idx_of[child]

        preds = list(G.predecessors(child))         # parents of `child`
        if not preds:                               # soma / root
            continue
        if len(preds) > 1:
            raise ValueError(f"Node {child} has {len(preds)} parents — "
                             "branches must be a tree for the Hines matrix")

        parent = preds[0]
        p = idx_of[parent]
        parent_idx[i] = p

        # ------------ axial conductance ------------
        # child geometry
        L_i    = G.nodes[child]['L']    * microns_to_cm   # cm
        d_i_cm = G.nodes[child]['diam'] * microns_to_cm
        r_i_cm = 0.5 * d_i_cm
        rho_i  = G.nodes[child]['Ra']                     # Ω·cm

        # parent geometry
        L_p    = G.nodes[parent]['L']    * microns_to_cm
        d_p_cm = G.nodes[parent]['diam'] * microns_to_cm
        r_p_cm = 0.5 * d_p_cm
        rho_p  = G.nodes[parent]['Ra']

        R_half_i = rho_i * (L_i / 2) / (np.pi * r_i_cm**2)
        R_half_p = rho_p * (L_p / 2) / (np.pi * r_p_cm**2)

        R_total  = R_half_i + R_half_p
        a_geom[i] = 1.0 / R_total                         # Siemens

    # ---------- 3. cast to torch tensors ----------
    parent_idx_t = torch.as_tensor(parent_idx, dtype=torch.int32)
    a_geom_t     = torch.as_tensor(a_geom,     dtype=dtype_axial)

    return parent_idx_t, a_geom_t


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

    order:     List[int]      = []
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

    compiler = ImplicitCompiler
    builder = ImplicitHandlerBuilder
    is_df = False

    def __init__(self, model, mech, imem=None, threads=32):
        super().__init__(model, mech, imem)
        self.threads = threads

        B, N = model.np, model.nc

        self.register_buffer("parent_idx",  torch.empty(N, dtype=torch.int32))  # (N,)
        self.register_buffer("order",       torch.empty(N, dtype=torch.int32))  # (N,) order of forward elimination

        self.register_buffer("diag_base",   torch.zeros(B, N))    # (B,N) base diagonal
        self.register_buffer("lower",       torch.empty(B, N))    # (B,N) lower diagonal

        self.register_buffer("scale",       torch.empty(B, N))    # (B,N) scale factor

        self.register_buffer("g_ax",        torch.empty(B, N))    # (B,N) axial conductance
        self.register_buffer("cmdt",        torch.empty(B, N))    # (B,N) capacitance * dt


    def initialize(self, model, dt):
        B, N = model.np, model.nc
        dt_s = dt * 1e-3

        radius_cm = 1e-4 * model.diam / 2.0                 # µm → cm   (B,K)
        dx_cm     = 1e-4 * model.dx                         # µm → cm   (B,K)
        area_cm2  = 2 * torch.pi * radius_cm * dx_cm        # cm²

        Cm     = 1e-6 * model.cm * area_cm2         # F   (B,K)
        Cm_inv = 1.0 / Cm                           # 1/F

        parent_idx_t, a_geom_t = graph_to_parent_and_axial(model.graph)
        parent_idx, children, depth = build_morphology(parent_idx_t.tolist())
        order, layer_ptr = build_dhs_layers(depth, self.threads)

        self.register_buffer("layer_ptr", layer_ptr.to(model.device()))  # (L+1,)
        self.order.copy_(order.to(dtype=torch.int32))
        self.parent_idx.copy_(parent_idx.to(dtype=torch.int32))  # (N,)
        self.lower.copy_(-a_geom_t.expand(B, -1))

        K = parent_idx.numel()
        g_ax = a_geom_t.clone()

        # add children contributions to their parent’s diagonal
        valid = parent_idx >= 0
        g_ax.index_add_(0, parent_idx[valid], a_geom_t[valid])
        self.g_ax.copy_(g_ax.expand(B, -1))  # (B,N)

        self.scale.copy_(area_cm2)

        cm = model.cm * 1e-6 * area_cm2      # convert from µF to F
        self.cmdt.copy_(cm / dt_s)           # (B,N) (F/s = S)

        # extracellular
        parent = parent_idx                # (K,)
        child  = (parent >= 0).nonzero(as_tuple=False).squeeze(1)   # (E,)
        self.register_buffer("edge_child",  child.to(torch.int64))
        self.register_buffer("edge_parent", parent[child].to(torch.int64))

        # axial conductance per edge (S) already in a_geom_t[child]
        self.register_buffer("edge_gax", a_geom_t[child])

    def step(self, model, dt, ve=None, intra=None):
        model.v = self._step(model.v, dt, model.celsius, ve, intra)

    def _step(self, v, dt, temp, ve=None, intra=None):

        self.mech.advance(v, dt, temp)

        itot = self.mech.i(v)
        gtot = self.mech.gtot(v)

        f_n = (gtot * v - itot) * self.scale

        if ve is not None:
            I_edge = _edge_currents(
                self.edge_child,
                self.edge_parent,
                self.edge_gax,
                ve
            ) # (B, E) mA
            S = torch.zeros_like(f_n) # (B, K)
            S.scatter_add_(1, self.edge_child.expand_as(I_edge), -I_edge)  # child gets -I
            S.scatter_add_(1, self.edge_parent.expand_as(I_edge), I_edge)  # parent gets +I
            f_n = f_n + S # (B, K) mA

        if intra is not None:
            f_n += intra

        RHS  = f_n + self.cmdt * v                       # mA
        main = self.g_ax + self.cmdt + gtot * self.scale # S

        d_ = main
        b_ = RHS
        a  = self.lower

        v_out = dhs_solve(
            d_, a, b_,
            self.parent_idx.to(d_.device, dtype=torch.int32),
            self.order.to(d_.device, dtype=torch.int32),
            self.layer_ptr.to(d_.device, dtype=torch.int32),
            threads=self.threads
        )
        return v_out