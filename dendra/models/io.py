try:
    from neuron import h, nrn  # type: ignore
except ImportError:
    pass

import re
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from ..helpers import requires_packages


def xyz(seg, extcell=None):
    """
    Return interpolated 3-D coordinates for a NEURON segment.

    Notes
    -----
    NEURON segment positions ``seg.x`` are normalized to the interval [0, 1],
    while ``sec.arc3d(i)`` is reported in physical distance units (typically µm)
    along the pt3d centerline.  We therefore convert the normalized position to
    physical arclength before interpolation.

    If ``extcell`` is provided, extracellular mechanism data are also copied from
    the segment in the same way as the original implementation.
    """
    sec = seg.sec
    n3d = int(sec.n3d())

    if n3d < 1:
        dat = {"x": 0.0, "y": 0.0, "z": 0.0}
    else:
        arc_l = np.array([sec.arc3d(i) for i in range(n3d)], dtype=float)
        x_arr = np.array([sec.x3d(i) for i in range(n3d)], dtype=float)
        y_arr = np.array([sec.y3d(i) for i in range(n3d)], dtype=float)
        z_arr = np.array([sec.z3d(i) for i in range(n3d)], dtype=float)

        seg_x = float(seg.x)
        seg_x = min(max(seg_x, 0.0), 1.0)

        total_arc = float(arc_l[-1]) if len(arc_l) else 0.0
        if total_arc <= 0.0:
            x = float(x_arr[0])
            y = float(y_arr[0])
            z = float(z_arr[0])
        else:
            s_um = seg_x * total_arc
            x = float(np.interp(s_um, arc_l, x_arr))
            y = float(np.interp(s_um, arc_l, y_arr))
            z = float(np.interp(s_um, arc_l, z_arr))

        dat = {"x": x, "y": y, "z": z}

    if extcell is not None:
        xraxial, xc, xg = [], [], []
        for i in range(int(extcell)):
            xraxial.append(seg.xraxial[i])
            xc.append(seg.xc[i])
            xg.append(seg.xg[i])
        dat["xraxial"] = xraxial
        dat["xc"] = xc
        dat["xg"] = xg

    return dat


@requires_packages("neuron")
def lambda_f(sec, freq_hz):
    """
    Python clone of the HOC function `lambda_f()`.

    Parameters
    ----------
    sec      : h.Section   -- NEURON section for which λ is computed
    freq_hz  : float       -- frequency [Hz]

    Returns
    -------
    float  -- space constant λ (microns) for `sec` at `freq_hz`
    """
    from math import pi, sqrt

    # make sure diam/3‑D info are up to date
    h.define_shape()

    n3d = int(h.n3d(sec=sec))
    if n3d < 2:  # no 3‑D points → uniform cylinder shortcut
        return 1e5 * sqrt(sec.diam / (4 * pi * freq_hz * sec.Ra * sec.cm))

    # --- piecewise integration along 3‑D centre line ------------------------
    x1 = h.arc3d(0, sec=sec)
    d1 = h.diam3d(0, sec=sec)
    lam = 0.0
    for i in range(1, n3d):
        x2 = h.arc3d(i, sec=sec)
        d2 = h.diam3d(i, sec=sec)
        lam += (x2 - x1) / sqrt(d1 + d2)
        x1, d1 = x2, d2

    # convert to “length‑in‑units‑of‑λ”
    lam *= sqrt(2.0) * 1e-5 * sqrt(4 * pi * freq_hz * sec.Ra * sec.cm)

    # return the actual λ (µm)
    return sec.L / lam


@requires_packages("neuron")
def apply_d_lambda(all_sections: List, d_lambda: float = 0.1, freq: float = 100.0):
    """
    Apply a d_lambda value to all sections in the NEURON model.

    Parameters:
        all_sections (List[nrn.Section]): List of all sections in the NEURON model.
        d_lambda (float): The d_lambda value to apply.
        freq (float): The frequency for the d_lambda application.
    """
    h.define_shape()

    for sec in all_sections:
        lam = lambda_f(sec, freq)  # λ(freq) in this section
        nseg = int((sec.L / (d_lambda * lam) + 0.9) / 2) * 2 + 1
        sec.nseg = max(1, nseg)  # safeguard: nseg must be ≥ 1


@requires_packages("neuron")
def read_swc(
    file_path: str, d_lambda=0.1, freq=100.0, data_func=None, **kwargs
) -> Tuple[nx.DiGraph, Dict[int, "nrn.Segment"]]:
    """Read an SWC file and return the contents."""

    if data_func is None:
        data_func = xyz

    h.load_file("import3d.hoc")

    rhoa = kwargs.get("rhoa", None)  # intracellular resistivity (Ω·cm)
    cm = kwargs.get("cm", None)  # specific membrane capacitance (µF/cm²)

    class Cell:
        def __init__(self, importer):
            importer.instantiate(self)

        def __repr__(self):
            return "SWCCell"

    reader = h.Import3d_SWC_read()
    reader.input(file_path)
    importer = h.Import3d_GUI(reader, 0)

    cell = Cell(importer)

    if rhoa is not None:
        for sec in cell.all:
            sec.Ra = rhoa
    if cm is not None:
        for sec in cell.all:
            sec.cm = cm

    apply_d_lambda(cell.all, d_lambda, freq)

    return neuron_to_dendra_graph(root_sec=cell.all[0], data_func=data_func)


@requires_packages("neuron")
def read_neurolucida(
    file_path: str, d_lambda=0.1, freq=100.0, data_func=None, **kwargs
) -> Tuple[nx.DiGraph, Dict[int, "nrn.Segment"]]:
    """Read a Neurolucida file and return the contents."""

    if data_func is None:
        data_func = xyz

    rhoa = kwargs.get("rhoa", None)  # intracellular resistivity (Ω·cm)
    cm = kwargs.get("cm", None)  # specific membrane capacitance (µF/cm²)

    h.load_file("import3d.hoc")

    class Cell:
        def __init__(self, importer):
            importer.instantiate(self)

        def __repr__(self):
            return "NeurolucidaCell"

    reader = h.Import3d_Neurolucida3()
    reader.quiet = 1
    reader.input(file_path)
    importer = h.Import3d_GUI(reader, 0)

    cell = Cell(importer)
    if rhoa is not None:
        for sec in cell.all:
            sec.Ra = rhoa
    if cm is not None:
        for sec in cell.all:
            sec.cm = cm

    apply_d_lambda(cell.all, d_lambda, freq)

    return neuron_to_dendra_graph(root_sec=cell.all[0], data_func=data_func)


read_asc = read_neurolucida


# convert NEURON sections to a directed acyclic graph (DAG)


def _sec_children(sec):
    """
    Yield (child_section, x_on_parent)  where 0 ≤ x ≤ 1 is the location
    on the *parent* section where the child connects.
    """
    sr = h.SectionRef(sec=sec)
    for i in range(int(sr.nchild())):
        child_sec = sr.child[i]  # this is already a Section
        parent_seg = child_sec.parentseg()  # Segment on parent
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
        idx = 0  # x=0 end is proximal
    else:
        idx = child_sec.nseg - 1  # x=1 end is proximal
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


def _section_axis_arrays(sec):
    """Return pt3d/stylized arclength and diameter arrays in µm.

    NEURON sections with pt3d morphology are represented as a sequence of
    truncated cones.  For stylized sections without pt3d points we fall back
    to a uniform cylinder using ``sec.L`` and ``sec.diam``.
    """
    try:
        n3d = int(sec.n3d())
    except Exception:
        n3d = int(h.n3d(sec=sec))

    if n3d >= 2:
        try:
            arc = np.array([sec.arc3d(i) for i in range(n3d)], dtype=float)
            diam = np.array([sec.diam3d(i) for i in range(n3d)], dtype=float)
        except Exception:
            arc = np.array([h.arc3d(i, sec=sec) for i in range(n3d)], dtype=float)
            diam = np.array([h.diam3d(i, sec=sec) for i in range(n3d)], dtype=float)

        # Ensure a nondecreasing arclength grid and remove duplicate points.
        order = np.argsort(arc, kind="stable")
        arc = arc[order]
        diam = diam[order]
        keep = np.concatenate(([True], np.diff(arc) > 0.0))
        arc = arc[keep]
        diam = diam[keep]

        if arc.size >= 2 and float(arc[-1] - arc[0]) > 0.0:
            if arc[0] != 0.0:
                arc = arc - arc[0]
            return arc, diam

    L = float(getattr(sec, "L", 0.0))
    d = float(getattr(sec, "diam", 0.0))
    if L <= 0.0:
        return np.array([0.0, 0.0], dtype=float), np.array([d, d], dtype=float)
    return np.array([0.0, L], dtype=float), np.array([d, d], dtype=float)


def _segment_index(seg) -> int:
    """Return the integer NEURON nseg index for a segment centre."""
    nseg = int(seg.sec.nseg)
    idx = int(round(float(seg.x) * nseg - 0.5))
    return max(0, min(nseg - 1, idx))


def _section_s_um(sec, x: float) -> float:
    """Map normalized section coordinate x in [0, 1] to arclength µm."""
    arc, _ = _section_axis_arrays(sec)
    total = float(arc[-1]) if arc.size else float(getattr(sec, "L", 0.0))
    x = min(max(float(x), 0.0), 1.0)
    return x * total


def _diam_at_s_um(s_um: float, arc: np.ndarray, diam: np.ndarray) -> float:
    if arc.size == 0:
        return 0.0
    if arc.size == 1 or float(arc[-1] - arc[0]) <= 0.0:
        return float(diam[0])
    return float(np.interp(float(s_um), arc, diam))


def _integrate_frustum_volume_um3(
    s0: float,
    s1: float,
    d0: float,
    d1: float,
    *,
    diameter_eps: float = 1e-12,
) -> float:
    """Volume of a truncated cone segment in µm³."""
    L = max(0.0, float(s1) - float(s0))
    if L <= 0.0:
        return 0.0
    r0 = max(float(d0), diameter_eps) * 0.5
    r1 = max(float(d1), diameter_eps) * 0.5
    return float(np.pi * L * (r0 * r0 + r0 * r1 + r1 * r1) / 3.0)


def _integrate_inv_area_um_inv(
    s0: float,
    s1: float,
    d0: float,
    d1: float,
    *,
    diameter_eps: float = 1e-12,
) -> float:
    """Exact ∫ ds/A(s) for a conical section in units µm⁻¹.

    Diameter is assumed to vary linearly between d0 and d1 over the interval.
    Since A(s) = π d(s)^2 / 4, the integral over a conical frustum is
    4 L / (π d0 d1), with the constant-diameter case included.
    """
    L = max(0.0, float(s1) - float(s0))
    if L <= 0.0:
        return 0.0
    d0 = max(float(d0), diameter_eps)
    d1 = max(float(d1), diameter_eps)
    return float(4.0 * L / (np.pi * d0 * d1))


def _integrate_section_interval(
    sec,
    s0: float,
    s1: float,
    *,
    quantity: str,
) -> float:
    """Integrate a pt3d/stylized section interval.

    ``quantity`` is either ``"volume"`` for µm³ or ``"inv_area"`` for
    ∫ds/A in µm⁻¹.
    """
    if s1 < s0:
        s0, s1 = s1, s0
    arc, diam = _section_axis_arrays(sec)
    if arc.size < 2 or s1 <= s0:
        return 0.0

    total = float(arc[-1])
    lo_all = max(0.0, min(float(s0), total))
    hi_all = max(0.0, min(float(s1), total))
    if hi_all <= lo_all:
        return 0.0

    out = 0.0
    for a0, a1, d0_raw, d1_raw in zip(arc[:-1], arc[1:], diam[:-1], diam[1:]):
        lo = max(lo_all, float(a0))
        hi = min(hi_all, float(a1))
        if hi <= lo:
            continue
        d_lo = _diam_at_s_um(lo, arc, diam)
        d_hi = _diam_at_s_um(hi, arc, diam)
        if quantity == "volume":
            out += _integrate_frustum_volume_um3(lo, hi, d_lo, d_hi)
        elif quantity == "inv_area":
            out += _integrate_inv_area_um_inv(lo, hi, d_lo, d_hi)
        else:
            raise ValueError(f"Unsupported section integral quantity: {quantity!r}.")
    return float(out)


def segment_volume_um3(seg) -> float:
    """Return the physical volume of a NEURON segment in µm³.

    If pt3d data are available, the volume is computed by clipping the section's
    truncated-cone sequence to the segment interval.  Otherwise this falls back
    to the stylized cylinder ``π(seg.diam/2)^2 * L/nseg``.
    """
    sec = seg.sec
    nseg = int(sec.nseg)
    if nseg <= 0:
        return 0.0

    try:
        n3d = int(sec.n3d())
    except Exception:
        n3d = int(h.n3d(sec=sec))

    if n3d < 2:
        L = float(getattr(sec, "L", 0.0)) / nseg
        d = float(getattr(seg, "diam", getattr(sec, "diam", 0.0)))
        if L <= 0.0 or d <= 0.0:
            return 0.0
        return float(np.pi * L * (0.5 * d) ** 2)

    idx = _segment_index(seg)
    total = _section_s_um(sec, 1.0)
    if total <= 0.0:
        return 0.0
    s0 = total * idx / nseg
    s1 = total * (idx + 1) / nseg
    return _integrate_section_interval(sec, s0, s1, quantity="volume")


def _section_inv_area_between_x(sec, x0: float, x1: float) -> float:
    """Return ∫ds/A between two normalized coordinates on one section."""
    s0 = _section_s_um(sec, x0)
    s1 = _section_s_um(sec, x1)
    return _integrate_section_interval(sec, s0, s1, quantity="inv_area")


def edge_inv_area_integral_um_inv(parent_seg, child_seg) -> float:
    """Return effective axial ∫ds/A for a graph edge, in µm⁻¹.

    The directed graph edge is assumed to connect the centre of ``parent_seg``
    to the centre of ``child_seg``.  For inter-section edges, the path is split
    into parent-centre→section-connection and child-connection→child-centre.
    """
    if parent_seg.sec is child_seg.sec:
        return _section_inv_area_between_x(parent_seg.sec, parent_seg.x, child_seg.x)

    parent_on_parent = child_seg.sec.parentseg()
    if parent_on_parent is not None and parent_on_parent.sec is parent_seg.sec:
        parent_part = _section_inv_area_between_x(
            parent_seg.sec, parent_seg.x, parent_on_parent.x
        )
        try:
            d0 = h.distance(parent_on_parent, child_seg.sec(0.0))
            d1 = h.distance(parent_on_parent, child_seg.sec(1.0))
            child_end_x = 0.0 if d0 <= d1 else 1.0
        except Exception:
            child_end_x = 0.0
        child_part = _section_inv_area_between_x(
            child_seg.sec, child_end_x, child_seg.x
        )
        return float(parent_part + child_part)

    raise ValueError(
        "Cannot infer axial section path for edge; parent/child segments do not "
        "appear to be adjacent in NEURON section topology."
    )


def _edge_diff_geom_from_resistance_um(parent_seg, child_seg, R_ohm: float) -> float:
    """Fallback conversion from electrical resistance to diffusion geometry.

    If R = ρ ∫ds/A and R is in Ω while ρ is in Ω·cm, then
    (∫ds_um/A_um²)^-1 = ρ * 1e4 / R, in µm.  This is exact only when a
    single effective axial resistivity applies across the edge.
    """
    if R_ohm is None or float(R_ohm) <= 0.0:
        return 0.0
    rho_parent = float(parent_seg.sec.Ra)
    rho_child = float(child_seg.sec.Ra)
    if parent_seg.sec is child_seg.sec:
        rho_eff = rho_child
    else:
        rho_eff = 0.5 * (rho_parent + rho_child)
    return float(rho_eff * 1e4 / float(R_ohm))


def edge_diff_geom_um(parent_seg, child_seg, R_ohm: float = None) -> float:
    """Return diffusion edge geometry factor in µm.

    A diffusion coefficient ``D`` in µm²/ms gives an edge conductance
    ``g = D * diff_geom_um`` in µm³/ms.
    """
    try:
        inv_area = edge_inv_area_integral_um_inv(parent_seg, child_seg)
        if inv_area > 0.0 and np.isfinite(inv_area):
            return float(1.0 / inv_area)
    except Exception:
        pass
    if R_ohm is not None:
        return _edge_diff_geom_from_resistance_um(parent_seg, child_seg, R_ohm)
    return 0.0


def _node_geometry_attrs(seg) -> Dict[str, float]:
    """Geometry attrs shared by NEURON-imported material/voltage paths."""
    vol = float(segment_volume_um3(seg))
    return {
        "volume": vol,
        "volume_um3": vol,
        "volume_i": vol,
    }


def _edge_geometry_attrs(parent_seg, child_seg, R_ohm_value: float) -> Dict[str, float]:
    diff_geom = float(edge_diff_geom_um(parent_seg, child_seg, R_ohm_value))
    return {
        "diff_geom_um": diff_geom,
    }


@requires_packages("neuron")
def neuron_to_dendra_graph(
    root_sec: Optional["nrn.Section"] = None,
    *,
    attach_objects: bool = True,
    data_func=None,
    extcell=None,
    exclude=None,
) -> Tuple[nx.DiGraph, Dict[int, "nrn.Segment"]]:
    """
    Build a directed acyclic graph whose nodes are NEURON compartments.
    If `data_func` is provided, it will be used to extract compartment data. Otherwise,
    a default function will be used (extracts xyz coordinates, and extracellular mechanism
    properties for extcell # of layers if extcell is provided). cm, Ra, L, diam are
    always extracted.

    Node attributes
    ---------------
    volume : float
        Compartment volume in µm³.  For pt3d morphologies this is computed by
        clipping the section's truncated-cone sequence to the segment interval;
        for stylized sections this falls back to a uniform cylinder.
    volume_i : float
        Intracellular material volume in µm³.  Currently identical to
        ``volume`` and provided as a domain-specific alias for material processes.

    Edge attributes
    ---------------
    L : float
        Centre-to-centre intracellular distance (µm), obtained with
        ``h.distance(seg_prox, seg_dist)`` — identical to NEURON's own
        geometry handling, including 3-D pt3d morphology.
    R_ohm : float
        Axial resistance (Ω) between those centres, taken from the distal
        segment's ``seg.ri()`` (which returns megohms) and scaled by 1e6.
    diff_geom_um : float
        Diffusion geometry factor in µm.  A diffusion coefficient ``D`` in
        µm²/ms gives an edge diffusive conductance ``D * diff_geom_um`` in
        µm³/ms.
    """
    if data_func is None:
        data_func = partial(xyz, extcell=extcell)

    # 0. discover root sections -------------------------------------------------
    if root_sec is None:
        roots = [s for s in h.allsec() if not h.SectionRef(sec=s).has_parent()]
    else:
        roots = [root_sec]

    assert len(roots) == 1, (
        "There is more than one candidate root section in the hoc namespace. Please specify one."
    )

    G = nx.DiGraph()
    id2seg = {}  # node‑id → Segment
    segkey2id = {}  # (Section, idx) → node‑id

    def node_for(seg, idx):
        """Return existing nodeid or create one for (seg.sec, idx)."""
        key = (seg.sec, idx)
        if key not in segkey2id:
            nid = len(segkey2id)
            segkey2id[key] = nid
            id2seg[nid] = seg
            if attach_objects:
                data = data_func(seg) if data_func else {}
                G.add_node(
                    nid,
                    diam=seg.diam,
                    L=seg.sec.L / seg.sec.nseg,
                    Ra=seg.sec.Ra,
                    cm=seg.cm,
                    name=str(seg),
                    area=seg.area(),
                    **_node_geometry_attrs(seg),
                    **data,
                )
            else:
                G.add_node(nid)
        return segkey2id[key]

    # 1. depth‑first traversal --------------------------------------------------
    stack, visited = list(roots), set()

    if exclude is not None:
        exclude_set = set(exclude)
        stack = [sec for sec in stack if sec not in exclude_set]

    while stack:
        sec = stack.pop()
        if sec in visited or (exclude is not None and sec in exclude_set):
            continue
        visited.add(sec)

        # 1a. axial neighbours inside this section -----------------------------
        prev_id = None
        prev_seg = None
        for x, idx in _compartments(sec):  # proximal → distal
            seg = sec(x)
            nid = node_for(seg, idx)

            if prev_id is not None:
                # exact centre‑to‑centre distance (µm)
                L_um = h.distance(prev_seg, seg)
                # exact axial resistance from NEURON (Ω)
                R_ohm = r_ohm(prev_seg, seg)
                G.add_edge(
                    prev_id,
                    nid,
                    L=L_um,
                    R_ohm=R_ohm,
                    **_edge_geometry_attrs(prev_seg, seg, R_ohm),
                )

            prev_id, prev_seg = nid, seg

        # 1b. parent → child ----------------------------------------------------
        for child_sec, x_on_parent in _sec_children(sec):
            # -------- parent compartment (centre of hosting segment) -----
            nseg_p = sec.nseg
            idx_p = min(int(x_on_parent * nseg_p), nseg_p - 1)
            parent_seg = sec((idx_p + 0.5) / nseg_p)
            parent_id = node_for(parent_seg, idx_p)

            # -------- child compartment that is *actually connected* -----
            child_seg, idx_c = _first_child_compartment(child_sec, parent_seg)
            child_id = node_for(child_seg, idx_c)

            # exact geometry & resistance
            L_um = h.distance(parent_seg, child_seg)  # µm
            R_ohm = r_ohm(parent_seg, child_seg)

            G.add_edge(
                parent_id,
                child_id,
                L=L_um,
                R_ohm=R_ohm,
                **_edge_geometry_attrs(parent_seg, child_seg, R_ohm),
            )

            stack.append(child_sec)

    patterns = {
        "DEND": r"dend",
        "APIC": r"apic",
        "SOMA": r"soma",
        "UNMYELIN": r"unmyelin",  # More specific pattern
        "MYELIN": r"\bmyelin\b",  # Matches 'myelin' as a whole word
        "AXON": r"axon",
        "NODE": r"node",
    }

    group_order = ["APIC", "DEND", "SOMA", "AXON", "UNMYELIN", "NODE", "MYELIN"]

    G, relabel_mapping = reorder_graph_by_patterns(G, patterns, group_order)
    id2seg = regenerate_id_map(id2seg, relabel_mapping)
    fix_graph_branchpoints(G, id2seg, data_func)
    return G, id2seg


def reorder_graph_by_patterns(
    G: nx.DiGraph,
    group_patterns: Dict[str, str],
    group_order: List[str],
    name_attribute: str = "name",
) -> Tuple[nx.DiGraph, Dict[int, int]]:
    """
    Reorders graph nodes based on regular expression patterns matched against a node attribute.

    This provides precise control to distinguish between similar names like
    'myelin' and 'unmyelin'.

    Args:
        G (nx.DiGraph): The input graph.
        group_patterns (Dict[str, str]): A dictionary where keys are group names
            and values are the string/regex patterns for that group.
            Example: {'MYELIN': r'\\bmyelin\\b', 'UNMYELIN': r'\\bunmyelin\\b'}
        group_order (List[str]): A list of the group name keys from group_patterns,
                                 defining the desired final order of the node groups.
        name_attribute (str): The node attribute to check against the patterns.

    Returns:
        Tuple[nx.DiGraph, Dict[int, int]]:
        - The new, reordered graph.
        - The mapping dictionary used for the transformation ({old_id: new_id}).
    """

    # --- 1. Define the New Order ---
    grouped_nodes: Dict[str, List[int]] = {name: [] for name in group_order}
    other_nodes: List[int] = []

    # Compile regex patterns for efficiency
    compiled_patterns = {
        group: re.compile(pattern, re.IGNORECASE)
        for group, pattern in group_patterns.items()
    }

    for node_id, attributes in G.nodes(data=True):
        node_name = attributes.get(name_attribute, "")
        matched = False

        # Iterate in the user-specified order to handle overlapping patterns correctly
        for group_name in group_order:
            pattern = compiled_patterns.get(group_name)
            if pattern and pattern.search(node_name):
                grouped_nodes[group_name].append(node_id)
                matched = True
                break  # A node belongs to the first group it matches

        if not matched:
            other_nodes.append(node_id)

    # --- 2. Create the Final Ordered List and Mapping ---
    new_order_old_ids: List[int] = []
    for group_name in group_order:
        new_order_old_ids.extend(sorted(grouped_nodes[group_name]))
    new_order_old_ids.extend(sorted(other_nodes))

    relabel_mapping: Dict[int, int] = {
        old_id: new_id for new_id, old_id in enumerate(new_order_old_ids)
    }

    # --- 3. Relabel the Graph ---
    G_reordered = nx.relabel_nodes(G, relabel_mapping, copy=True)

    return G_reordered, relabel_mapping


def regenerate_id_map(
    original_id_map: Dict[int, Any], relabel_mapping: Dict[int, int]
) -> Dict[int, Any]:
    """
    Updates an ID-to-object map to be consistent with a reordered graph.

    Args:
        original_id_map (Dict[int, Any]): The original map, e.g., {old_id: segment}.
        relabel_mapping (Dict[int, int]): The map from {old_id: new_id}.

    Returns:
        Dict[int, Any]: The new map, {new_id: segment}.
    """
    if not relabel_mapping:
        # If no relabeling was done, return a copy of the original
        return original_id_map.copy()

    new_id_map = {}
    for old_id, new_id in relabel_mapping.items():
        # For each mapping of an old ID to a new ID,
        # find the object associated with the old ID...
        obj = original_id_map[old_id]
        # ...and assign it to the new ID in the new map.
        new_id_map[new_id] = obj

    return new_id_map


def find_branch_points(G: nx.DiGraph):
    """
    Finds all branch points in a directed graph representing a tree structure.

    A branch point is defined as a node with an out-degree greater than 1.

    Args:
        G (nx.DiGraph): The directed graph to analyze. Nodes can be any hashable type
                        (e.g., integers representing compartment IDs).

    Returns:
        List[Any]: A list of the node IDs that are branch points.
    """
    branch_nodes = []
    # G.out_degree is a view that provides (node, degree) pairs.
    # We can iterate through it directly or convert it to a dict.
    # This is highly efficient.
    for node, out_degree in G.out_degree():
        if out_degree > 1:
            branch_nodes.append(node)

    return branch_nodes


def get_children_of_nodes(G: nx.DiGraph, node_ids):
    children_map = {}
    for node_id in node_ids:
        # G.successors(node_id) returns an iterator over the children of the node.
        # We convert it to a list.
        try:
            children = list(G.successors(node_id))
            children_map[node_id] = children
        except nx.NetworkXError:
            # This handles the case where a node_id in the list might not exist in the graph.
            # It's good practice to handle this gracefully.
            children_map[node_id] = []

    return children_map


def fix_graph_branchpoints(G, id2seg, data_func=None):
    children_map = get_children_of_nodes(G, find_branch_points(G))
    to_fix = {}
    for pre, post_list in children_map.items():
        for idx in post_list:
            post_seg = id2seg[idx]
            parent_seg = post_seg.sec.parentseg()
            if parent_seg is None:
                continue
            parent_x = parent_seg.x
            if parent_x == 0 or parent_x == 1:
                to_fix.setdefault(pre, {}).setdefault(parent_x, []).append(idx)
    c = 0
    for pre_idx, dct in to_fix.items():
        for x_on_pre, post_indices in dct.items():
            parent_seg_true = id2seg[pre_idx].sec(x_on_pre)
            data = data_func(parent_seg_true) if data_func else {}
            nid = len(G.nodes)
            G.add_node(
                nid,
                diam=parent_seg_true.diam,
                L=0.0,
                Ra=parent_seg_true.sec.Ra,
                cm=parent_seg_true.cm,
                name=f"branchpoint.{c}.{parent_seg_true}",
                area=parent_seg_true.area(),
                volume=0.0,
                volume_um3=0.0,
                volume_i=0.0,
                **data,
            )
            id2seg[nid] = parent_seg_true
            c += 1
            R_pre_bp = parent_seg_true.ri() * 1e6
            G.add_edge(
                pre_idx,
                nid,
                L=h.distance(id2seg[pre_idx], parent_seg_true),
                R_ohm=R_pre_bp,
                **_edge_geometry_attrs(id2seg[pre_idx], parent_seg_true, R_pre_bp),
            )
            for post_idx in post_indices:
                G.remove_edge(pre_idx, post_idx)
                R_bp_post = id2seg[post_idx].ri() * 1e6
                G.add_edge(
                    nid,
                    post_idx,
                    L=h.distance(parent_seg_true, id2seg[post_idx]),
                    R_ohm=R_bp_post,
                    **_edge_geometry_attrs(
                        parent_seg_true, id2seg[post_idx], R_bp_post
                    ),
                )
