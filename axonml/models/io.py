try:
    from neuron import h
    NEURON_INSTALLED = True
except ImportError:
    NEURON_INSTALLED = False

import networkx as nx
from typing import Dict, Tuple, Optional, List


def is_neuron_installed() -> bool:
    """Check if NEURON is installed."""
    return NEURON_INSTALLED


def read_swc(file_path: str):
    """Read an SWC file and return the contents."""
    if not is_neuron_installed():
        raise ImportError("NEURON is not installed. Cannot read SWC files.")
    
    h.load_file("stdrun.hoc")
    h.load_file("import3d.hoc")
    
    reader = h.Import3d_SWC_read(file_path)
    importer = h.Import3d_Importer(reader)
    importer.import_swc()
    
    return reader

# convert NEURON sections to a directed acyclic graph (DAG)

def _sec_children(sec):
    """
    Yield (child_section, x_on_parent)  where 0 ≤ x ≤ 1 is the location
    on the *parent* section where the child connects.
    """
    sr = h.SectionRef(sec=sec)
    for i in range(int(sr.nchild())):
        child_sec = sr.child[i]                 # this is already a Section
        parent_seg = child_sec.parentseg()      # Segment on parent
        yield child_sec, parent_seg.x


def _compartments(sec) -> List[Tuple[float, int]]:
    """Return list of (x, idx) for segment centres in `sec` (proximal→distal)."""
    n = sec.nseg
    return [((i + 0.5) / n, i) for i in range(n)]


def _first_child_compartment(child_sec, parent_seg):
    """Return (child_seg, idx) for the compartment that actually touches parent_seg,
    irrespective of child orientation.  Works for any nseg, even nseg==1."""
    d0 = h.distance(parent_seg, child_sec(0))  # parent ↔ child x=0
    d1 = h.distance(parent_seg, child_sec(1))  # parent ↔ child x=1
    if d0 <= d1:
        idx = 0                                # x=0 end is proximal
    else:
        idx = child_sec.nseg - 1               # x=1 end is proximal
    x_center = (idx + 0.5) / child_sec.nseg
    return child_sec(x_center), idx


def r_ohm(parent_seg, child_seg):
    if parent_seg.sec is child_seg.sec:
        return child_seg.ri() * 1e6
    else:
        if child_seg.sec.parentseg().x == 0 or child_seg.sec.parentseg().x == 1:
            child_r = child_seg.ri() * 1e6
            parent_r = child_seg.sec.parentseg().ri() * 1e6
            return child_r + parent_r
        return child_seg.ri() * 1e6


def neuron_to_axonml_graph(
    root_sec: Optional['h.Section'] = None,
    *,
    attach_objects: bool = True,
    **data_kwargs
) -> Tuple[nx.DiGraph, Dict[int, 'h.Segment']]:
    """
    Build a directed acyclic graph whose nodes are NEURON compartments.

    Edge attributes
    ---------------
    L : float
        Centre-to-centre intracellular distance (µm), obtained with
        ``h.distance(seg_prox, seg_dist)`` — identical to NEURON's own
        geometry handling, including 3-D pt3d morphology.
    R_ohm : float
        Axial resistance (Ω) between those centres, taken from the distal
        segment's ``seg.ri()`` (which returns megohms) and scaled by 1e6.
    """

    # 0. discover root sections -------------------------------------------------
    if root_sec is None:
        roots = [s for s in h.allsec() if not h.SectionRef(sec=s).has_parent()]
    else:
        roots = [root_sec]

    assert len(roots) == 1, "There is more than one candidate root section in the hoc namespace. Please specify one."

    G          = nx.DiGraph()
    id2seg     = {}                 # node‑id → Segment
    segkey2id  = {}                 # (Section, idx) → node‑id

    def node_for(seg, idx):
        """Return existing node-id or create one for (seg.sec, idx)."""
        key = (seg.sec, idx)
        if key not in segkey2id:
            nid = len(segkey2id)
            segkey2id[key] = nid
            id2seg[nid]    = seg
            if attach_objects:
                data = {
                    k:v(seg) for k,v in data_kwargs.items()
                }
                G.add_node(
                    nid,
                    diam=seg.diam,
                    L=seg.sec.L / seg.sec.nseg,
                    Ra=seg.sec.Ra,
                    cm=seg.cm,
                    name=str(seg),
                    area=seg.area(),
                    **data
                )
            else:
                G.add_node(nid)
        return segkey2id[key]

    # 1. depth‑first traversal --------------------------------------------------
    stack, visited = list(roots), set()

    while stack:
        sec = stack.pop()
        if sec in visited:
            continue
        visited.add(sec)

        # 1a. axial neighbours inside this section -----------------------------
        prev_id  = None
        prev_seg = None
        for x, idx in _compartments(sec):            # proximal → distal
            seg = sec(x)
            nid = node_for(seg, idx)

            if prev_id is not None:
                # exact centre‑to‑centre distance (µm)
                L_um = h.distance(prev_seg, seg)
                # exact axial resistance from NEURON (Ω)
                R_ohm = r_ohm(prev_seg, seg)
                G.add_edge(prev_id, nid,
                           L=L_um,
                           R_ohm=R_ohm)

            prev_id, prev_seg = nid, seg

        # 1b. parent → child ----------------------------------------------------
        for child_sec, x_on_parent in _sec_children(sec):

            # -------- parent compartment (centre of hosting segment) -----
            nseg_p  = sec.nseg
            idx_p   = min(int(x_on_parent * nseg_p), nseg_p - 1)
            parent_seg = sec((idx_p + 0.5) / nseg_p)
            parent_id  = node_for(parent_seg, idx_p)

            # -------- child compartment that is *actually connected* -----
            child_seg, idx_c = _first_child_compartment(child_sec, parent_seg)
            child_id = node_for(child_seg, idx_c)

            # exact geometry & resistance
            L_um  = h.distance(parent_seg, child_seg)     # µm
            R_ohm = r_ohm(parent_seg, child_seg)

            G.add_edge(parent_id, child_id,
                       L=L_um,
                       R_ohm=R_ohm)

            stack.append(child_sec)

    return G, id2seg