try:
    from neuron import h, nrn  # type: ignore
except ImportError:
    pass

import re
from functools import partial
from os import PathLike
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from ..helpers import requires_packages


def _positive_finite(value, *, name: str) -> float:
    """Normalize a positive finite scalar used by morphology discretization."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number, not a boolean.")
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number.") from error
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}.")
    return value


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

        seg_x = _pt3d_x(sec, float(seg.x))

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

    freq_hz = _positive_finite(freq_hz, name="freq_hz")
    rhoa = _positive_finite(sec.Ra, name="section Ra")
    cm = _positive_finite(sec.cm, name="section cm")
    section_length = _positive_finite(sec.L, name="section L")

    # make sure diam/3‑D info are up to date
    h.define_shape()

    n3d = int(h.n3d(sec=sec))
    if n3d < 2:  # no 3‑D points → uniform cylinder shortcut
        diameter = _positive_finite(sec.diam, name="section diameter")
        return 1e5 * sqrt(diameter / (4 * pi * freq_hz * rhoa * cm))

    # --- piecewise integration along 3‑D centre line ------------------------
    x1 = h.arc3d(0, sec=sec)
    d1 = h.diam3d(0, sec=sec)
    lam = 0.0
    for i in range(1, n3d):
        x2 = h.arc3d(i, sec=sec)
        d2 = h.diam3d(i, sec=sec)
        diameter_sum = _positive_finite(
            d1 + d2, name=f"section diameter sum at pt3d interval {i - 1}:{i}"
        )
        lam += (x2 - x1) / sqrt(diameter_sum)
        x1, d1 = x2, d2

    # convert to “length‑in‑units‑of‑λ”
    lam *= sqrt(2.0) * 1e-5 * sqrt(4 * pi * freq_hz * rhoa * cm)
    lam = _positive_finite(lam, name="section electrotonic length")

    # return the actual λ (µm)
    return section_length / lam


@requires_packages("neuron")
def apply_d_lambda(all_sections: List, d_lambda: float = 0.1, freq: float = 100.0):
    """
    Apply a d_lambda value to all sections in the NEURON model.

    Parameters:
        all_sections (List[nrn.Section]): List of all sections in the NEURON model.
        d_lambda (float): The d_lambda value to apply.
        freq (float): The frequency for the d_lambda application.
    """
    d_lambda = _positive_finite(d_lambda, name="d_lambda")
    freq = _positive_finite(freq, name="freq")
    h.define_shape()

    for sec in all_sections:
        lam = lambda_f(sec, freq)  # λ(freq) in this section
        nseg = int((sec.L / (d_lambda * lam) + 0.9) / 2) * 2 + 1
        sec.nseg = max(1, nseg)  # safeguard: nseg must be ≥ 1


@requires_packages("neuron")
def read_swc(
    file_path: str | PathLike[str],
    d_lambda=0.1,
    freq=100.0,
    data_func=None,
    **kwargs,
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
    reader.input(str(file_path))
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
    file_path: str | PathLike[str],
    d_lambda=0.1,
    freq=100.0,
    data_func=None,
    **kwargs,
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
    reader.input(str(file_path))
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


def _section_orientation(sec) -> int:
    """Return the section end (0 or 1) that faces its parent.

    NEURON permits either end of a child section to be connected to its parent.
    ``Section.orientation()`` reports that end.  Root sections conventionally
    have orientation 0, which is also the direction used by NEURON's own node
    ordering.
    """
    try:
        orientation = float(sec.orientation())
    except Exception:
        orientation = float(h.section_orientation(sec=sec))
    return 0 if orientation < 0.5 else 1


def _pt3d_x(sec, section_x: float) -> float:
    """Map NEURON's logical section coordinate onto stored pt3d arclength.

    ``section_orientation()==1`` means the logical section runs opposite the
    order of its pt3d points.  NEURON already applies that reversal to segment
    diameter, area, and axial resistance.  Geometry extracted directly from
    ``arc3d`` must apply it explicitly as well.
    """
    section_x = min(max(float(section_x), 0.0), 1.0)
    return section_x if _section_orientation(sec) == 0 else 1.0 - section_x


def _segment_center(sec, idx: int):
    nseg = int(sec.nseg)
    if not 0 <= int(idx) < nseg:
        raise IndexError(f"Segment index {idx} is outside section {sec} (nseg={nseg}).")
    return sec((int(idx) + 0.5) / nseg)


def _segment_index_for_location(sec, x: float) -> int:
    """Return the computational segment containing a NEURON location.

    Interior section locations are represented electrically by the containing
    segment centre.  At an exact internal boundary NEURON selects the segment on
    the x-increasing side; ``int(x*nseg)`` reproduces that convention.
    """
    nseg = int(sec.nseg)
    x = min(max(float(x), 0.0), 1.0)
    return min(int(x * nseg), nseg - 1)


def _first_child_compartment(child_sec, parent_seg=None):
    """Return the child compartment adjacent to the parent connection.

    ``parent_seg`` is accepted for backward compatibility but is not needed:
    the connected child end is available directly from the section orientation.
    """
    endpoint = _section_orientation(child_sec)
    idx = 0 if endpoint == 0 else int(child_sec.nseg) - 1
    return _segment_center(child_sec, idx), idx


def _finite_ri_ohm(value_mohm: float, *, context: str) -> float:
    """Convert NEURON's MΩ axial resistance to Ω and reject sentinel values."""
    value_mohm = float(value_mohm)
    # NEURON uses 1e30 MΩ for a node with no parent.  Values of this order are
    # never a physical compartmental axial resistance and would silently
    # disconnect a Dendra tree.
    if not np.isfinite(value_mohm) or value_mohm >= 1e29:
        raise ValueError(
            f"NEURON returned a non-physical axial resistance ({value_mohm:g} MΩ) "
            f"while importing {context}."
        )
    if value_mohm < 0.0:
        raise ValueError(
            f"NEURON returned a negative axial resistance ({value_mohm:g} MΩ) "
            f"while importing {context}."
        )
    return value_mohm * 1e6


def _center_to_endpoint_resistance_ohm(sec, endpoint_x: int) -> float:
    """Exact NEURON resistance from an adjacent segment centre to an endpoint.

    ``Segment.ri()`` is referenced to the segment's parent node.  At the
    parent-facing endpoint, the adjacent segment centre therefore owns the
    half-segment resistance.  At the opposite endpoint, the endpoint node owns
    it.  Querying the root's parent-facing endpoint directly would return
    NEURON's 1e30-MΩ no-parent sentinel, which is the bug this helper avoids.

    For a reversed section, NEURON also reverses the mapping between logical
    ``Segment.x`` and stored pt3d arclength.  The last logical centre remains
    parent-adjacent and owns the correct (possibly tapered) half-segment
    resistance; raw pt3d geometry consumers must account for the reversal
    separately via :func:`_pt3d_x`.
    """
    endpoint_x = int(endpoint_x)
    if endpoint_x not in (0, 1):
        raise ValueError(f"endpoint_x must be 0 or 1, got {endpoint_x!r}.")

    orientation = _section_orientation(sec)
    idx = 0 if endpoint_x == 0 else int(sec.nseg) - 1
    adjacent = _segment_center(sec, idx)
    if endpoint_x == orientation:
        value = adjacent.ri()
        owner = adjacent
    else:
        owner = sec(float(endpoint_x))
        value = owner.ri()
    return _finite_ri_ohm(
        value,
        context=f"half segment {adjacent} ↔ {sec}({endpoint_x}) (ri owner {owner})",
    )


def _same_section_resistance_ohm(seg_a, seg_b) -> float:
    """Exact axial resistance between arbitrary centres in one section."""
    if seg_a.sec is not seg_b.sec:
        raise ValueError("Segments must belong to the same section.")
    sec = seg_a.sec
    ia = _segment_index(seg_a)
    ib = _segment_index(seg_b)
    if ia == ib:
        return 0.0

    orientation = _section_orientation(sec)
    total = 0.0
    lo, hi = sorted((ia, ib))
    for left_idx in range(lo, hi):
        # For orientation 0 the x-increasing centre owns the resistance to its
        # parent node.  For orientation 1, the x-decreasing centre owns it.
        owner_idx = left_idx + 1 if orientation == 0 else left_idx
        owner = _segment_center(sec, owner_idx)
        total += _finite_ri_ohm(
            owner.ri(), context=f"centre-to-centre edge in {sec} at {owner}"
        )
    return total


def r_ohm(parent_seg, child_seg):
    """Return NEURON-equivalent axial resistance between two graph centres.

    The function supports adjacent centres in one section and a direct
    parent-section/child-section connection.  Endpoint connections include the
    half segment on both sides; interior connections include only the child's
    proximal half segment because NEURON attaches them to the containing parent
    compartment node.
    """
    if parent_seg.sec is child_seg.sec:
        return _same_section_resistance_ohm(parent_seg, child_seg)

    child_sec = child_seg.sec
    parent_on_parent = child_sec.parentseg()
    if parent_on_parent is None or parent_on_parent.sec is not parent_seg.sec:
        raise ValueError(
            f"{parent_seg} and {child_seg} are not a direct NEURON "
            "parent-section/child-section pair."
        )

    parent_host_idx = _segment_index_for_location(parent_seg.sec, parent_on_parent.x)
    parent_host = _segment_center(parent_seg.sec, parent_host_idx)
    parent_path = _same_section_resistance_ohm(parent_seg, parent_host)

    parent_x = float(parent_on_parent.x)
    if parent_x == 0.0 or parent_x == 1.0:
        parent_path += _center_to_endpoint_resistance_ohm(parent_seg.sec, int(parent_x))

    child_endpoint = _section_orientation(child_sec)
    child_adjacent_idx = 0 if child_endpoint == 0 else int(child_sec.nseg) - 1
    child_adjacent = _segment_center(child_sec, child_adjacent_idx)
    child_path = _center_to_endpoint_resistance_ohm(child_sec, child_endpoint)
    child_path += _same_section_resistance_ohm(child_adjacent, child_seg)
    return parent_path + child_path


def _section_axis_arrays(sec):
    """Return pt3d/stylized arclength and diameter arrays in µm.

    NEURON sections with pt3d morphology are represented as a sequence of
    truncated cones. Repeated arclengths are retained because NEURON uses two
    coincident controls with different diameters to encode an instantaneous
    diameter step. The zero-length step contributes no volume or axial
    resistance, while its incoming and outgoing diameters define the adjacent
    positive-length frusta. For stylized sections without pt3d points we fall
    back to a uniform cylinder using ``sec.L`` and ``sec.diam``.
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

        # Ensure a nondecreasing arclength grid. Keep repeated arclengths: a
        # stable ordering preserves the incoming/outgoing sides of a diameter
        # discontinuity authored as coincident pt3d controls.
        order = np.argsort(arc, kind="stable")
        arc = arc[order]
        diam = diam[order]

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
    """Map logical section coordinate x in [0, 1] to pt3d arclength µm."""
    arc, _ = _section_axis_arrays(sec)
    total = float(arc[-1]) if arc.size else float(getattr(sec, "L", 0.0))
    return _pt3d_x(sec, x) * total


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
        span = float(a1) - float(a0)
        # A repeated arclength with a new diameter is an instantaneous step.
        # It has no volume or axial resistance. The stable controls on either
        # side still supply the correct endpoint diameters to their respective
        # positive-length frusta.
        if span <= 0.0:
            continue
        lo = max(lo_all, float(a0))
        hi = min(hi_all, float(a1))
        if hi <= lo:
            continue
        lo_fraction = (lo - float(a0)) / span
        hi_fraction = (hi - float(a0)) / span
        diameter_delta = float(d1_raw) - float(d0_raw)
        d_lo = float(d0_raw) + lo_fraction * diameter_delta
        d_hi = float(d0_raw) + hi_fraction * diameter_delta
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
    arc, _ = _section_axis_arrays(sec)
    total = float(arc[-1]) if arc.size else 0.0
    if total <= 0.0:
        return 0.0
    s0 = _section_s_um(sec, idx / nseg)
    s1 = _section_s_um(sec, (idx + 1) / nseg)
    return _integrate_section_interval(sec, s0, s1, quantity="volume")


def _section_inv_area_between_x(sec, x0: float, x1: float) -> float:
    """Return ∫ds/A between two normalized coordinates on one section."""
    s0 = _section_s_um(sec, x0)
    s1 = _section_s_um(sec, x1)
    return _integrate_section_interval(sec, s0, s1, quantity="inv_area")


def edge_inv_area_integral_um_inv(parent_seg, child_seg) -> float:
    """Return NEURON-topology-equivalent axial ∫ds/A, in µm⁻¹.

    For an endpoint connection the path contains the parent-side and child-side
    half segments.  For an interior connection, NEURON attaches the child to the
    containing parent compartment node, so only the child-side cable contributes
    at the junction (plus any requested path from a non-host parent centre).
    """
    if parent_seg.sec is child_seg.sec:
        return _section_inv_area_between_x(parent_seg.sec, parent_seg.x, child_seg.x)

    child_sec = child_seg.sec
    parent_on_parent = child_sec.parentseg()
    if parent_on_parent is None or parent_on_parent.sec is not parent_seg.sec:
        raise ValueError(
            "Cannot infer axial section path for edge; parent/child segments do not "
            "appear to be adjacent in NEURON section topology."
        )

    parent_host_idx = _segment_index_for_location(parent_seg.sec, parent_on_parent.x)
    parent_host = _segment_center(parent_seg.sec, parent_host_idx)
    parent_part = _section_inv_area_between_x(
        parent_seg.sec, parent_seg.x, parent_host.x
    )

    parent_x = float(parent_on_parent.x)
    if parent_x == 0.0 or parent_x == 1.0:
        parent_part += _section_inv_area_between_x(
            parent_seg.sec, parent_host.x, parent_x
        )

    child_end_x = float(_section_orientation(child_sec))
    child_part = _section_inv_area_between_x(child_sec, child_end_x, child_seg.x)
    return float(parent_part + child_part)


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


def _collect_neuron_sections(root_sec, exclude_set=None):
    """Collect the included NEURON section subtree in deterministic DFS order."""
    exclude_set = set() if exclude_set is None else set(exclude_set)
    sections = []
    stack = [root_sec]
    visited = set()
    while stack:
        sec = stack.pop()
        if sec in visited or sec in exclude_set:
            continue
        visited.add(sec)
        sections.append(sec)
        children = [child for child, _ in _sec_children(sec)]
        # Reverse before pushing so SectionRef child order is preserved on pop.
        stack.extend(reversed(children))
    return sections


def _safe_distance_um(seg_a, seg_b) -> float:
    """Return NEURON path distance between two computational locations."""
    value = float(h.distance(seg_a, seg_b))
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(
            f"NEURON returned invalid path distance {value!r} between "
            f"{seg_a} and {seg_b}."
        )
    return value


def _edge_attrs_from_path(
    *, R_ohm_value: float, L_um: float, inv_area_um_inv: float
) -> Dict[str, float]:
    """Create electrical and diffusion metadata for one resistor-tree edge."""
    R_ohm_value = float(R_ohm_value)
    L_um = float(L_um)
    inv_area_um_inv = float(inv_area_um_inv)
    if not np.isfinite(R_ohm_value) or R_ohm_value < 0.0:
        raise ValueError(f"Invalid axial resistance {R_ohm_value!r} Ω.")
    if not np.isfinite(L_um) or L_um < 0.0:
        raise ValueError(f"Invalid axial path length {L_um!r} µm.")
    if not np.isfinite(inv_area_um_inv) or inv_area_um_inv < 0.0:
        raise ValueError(
            f"Invalid axial inverse-area integral {inv_area_um_inv!r} µm⁻¹."
        )
    diff_geom = 1.0 / inv_area_um_inv if inv_area_um_inv > 0.0 else 0.0
    return {
        "L": L_um,
        "R_ohm": R_ohm_value,
        "diff_geom_um": float(diff_geom),
        # Kept only while zero-volume degree-two junctions are simplified.
        "_inv_area_um_inv": inv_area_um_inv,
    }


def _add_resistor_edge(graph, u, v, attrs):
    """Add one physical resistor edge, rejecting loops/parallel paths."""
    if u == v:
        raise ValueError(
            "A NEURON section connection collapsed both ends onto the same "
            "computational node; loop/self connections cannot form a Tree."
        )
    if graph.has_edge(u, v):
        raise ValueError(
            f"Multiple axial paths were found between morphology nodes {u!r} "
            f"and {v!r}; the imported morphology is not a tree."
        )
    graph.add_edge(u, v, **attrs)


def _series_edge_attrs(first: Dict[str, float], second: Dict[str, float]):
    """Combine two cable paths separated by a zero-area degree-two junction."""
    return _edge_attrs_from_path(
        R_ohm_value=float(first["R_ohm"]) + float(second["R_ohm"]),
        L_um=float(first["L"]) + float(second["L"]),
        inv_area_um_inv=float(first.get("_inv_area_um_inv", 0.0))
        + float(second.get("_inv_area_um_inv", 0.0)),
    )


def _branchpoint_node_attrs(seg, data_func=None, *, attach_objects=True):
    """Attributes for a zero-volume physical section junction."""
    if not attach_objects:
        return {"name": "branchpoint.pending"}
    data = data_func(seg) if data_func else {}
    attrs = {
        "diam": float(seg.diam),
        "L": 0.0,
        "Ra": float(seg.sec.Ra),
        "cm": float(seg.cm),
        "name": "branchpoint.pending",
        "area": 0.0,
        "volume": 0.0,
        "volume_um3": 0.0,
        "volume_i": 0.0,
        "neuron_location": str(seg),
    }
    attrs.update(data)
    # Branchpoint semantics must not be overridden by a custom data function.
    attrs.update(
        {
            "L": 0.0,
            "area": 0.0,
            "volume": 0.0,
            "volume_um3": 0.0,
            "volume_i": 0.0,
            "name": "branchpoint.pending",
        }
    )
    return attrs


def _orient_resistor_tree(graph: nx.Graph, root_node) -> nx.DiGraph:
    """Orient an undirected resistor tree away from a selected material node."""
    if root_node not in graph:
        raise ValueError("The selected root compartment is absent from the graph.")
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot orient an empty morphology graph.")
    if not nx.is_tree(graph):
        components = nx.number_connected_components(graph)
        cycles = len(nx.cycle_basis(graph))
        raise ValueError(
            "NEURON morphology did not reduce to one resistor tree "
            f"(components={components}, independent_cycles={cycles})."
        )

    directed = nx.DiGraph()
    directed.add_nodes_from(
        (node, dict(attrs)) for node, attrs in graph.nodes(data=True)
    )
    for parent, child in nx.bfs_edges(graph, root_node):
        directed.add_edge(parent, child, **dict(graph.edges[parent, child]))
    return directed


@requires_packages("neuron")
def neuron_to_dendra_graph(
    root_sec: Optional["nrn.Section"] = None,
    *,
    attach_objects: bool = True,
    data_func=None,
    extcell=None,
    exclude=None,
) -> Tuple[nx.DiGraph, Dict[int, "nrn.Segment"]]:
    """Convert a NEURON section tree into Dendra's compartment resistor tree.

    The conversion is performed in three stages:

    1. Create one material node per NEURON segment centre and one provisional
       zero-area node per distinct physical section endpoint.
    2. Merge connected endpoint nodes with each other, or with the containing
       parent compartment for interior section connections.
    3. Remove sealed degree-one endpoint nodes and series-collapse degree-two
       endpoint nodes.  Junctions of degree three or greater remain explicit
       zero-volume ``branchpoint`` nodes.

    This construction mirrors NEURON's computational topology and avoids calling
    ``ri()`` on the parentless x=0 endpoint of a root section, where NEURON
    intentionally returns its 1e30-MΩ sentinel.

    Parameters
    ----------
    root_sec : neuron.h.Section, optional
        Root of the section subtree to import.  If omitted, exactly one
        parentless section must exist in the HOC namespace.
    attach_objects : bool, optional
        Attach membrane/geometry metadata to graph nodes.  Electrical edge
        metadata and ``id2seg`` are always produced.
    data_func : callable, optional
        Additional node metadata extractor.  Defaults to :func:`xyz`.
    extcell : int, optional
        Number of extracellular layers copied by the default data extractor.
    exclude : iterable of neuron.h.Section, optional
        Sections whose complete descendant subtrees are pruned.

    Returns
    -------
    graph : networkx.DiGraph
        A rooted tree with exact NEURON axial resistances in Ω.
    id2seg : dict
        Mapping of graph node IDs to their corresponding NEURON segment or
        physical endpoint location.
    """
    if data_func is None:
        data_func = partial(xyz, extcell=extcell)

    # Discover a unique root if the caller did not supply one.
    if root_sec is None:
        roots = [s for s in h.allsec() if not h.SectionRef(sec=s).has_parent()]
        if len(roots) != 1:
            raise AssertionError(
                "There is more than one candidate root section in the hoc "
                "namespace. Please specify one."
            )
        root_sec = roots[0]

    exclude_set = set() if exclude is None else set(exclude)
    sections = _collect_neuron_sections(root_sec, exclude_set)
    if not sections:
        raise ValueError("The requested NEURON subtree is empty after exclusions.")
    section_set = set(sections)

    resistor_graph = nx.Graph()
    id2seg: Dict[int, "nrn.Segment"] = {}
    segkey2id = {}

    # ------------------------------------------------------------------
    # Material nodes: one per NEURON segment centre.
    # ------------------------------------------------------------------
    for sec in sections:
        for x, idx in _compartments(sec):
            seg = sec(x)
            nid = len(segkey2id)
            segkey2id[(sec, idx)] = nid
            id2seg[nid] = seg
            if attach_objects:
                data = data_func(seg) if data_func else {}
                attrs = {
                    "diam": float(seg.diam),
                    "L": float(sec.L) / int(sec.nseg),
                    "Ra": float(sec.Ra),
                    "cm": float(seg.cm),
                    "name": str(seg),
                    "area": float(seg.area()),
                    **_node_geometry_attrs(seg),
                }
                attrs.update(data)
                resistor_graph.add_node(nid, **attrs)
            else:
                resistor_graph.add_node(nid)

    # ------------------------------------------------------------------
    # Identify physical endpoint junctions.  A child endpoint is the same
    # electrical node as either a parent endpoint or a containing parent
    # segment centre for an interior connection.
    # ------------------------------------------------------------------
    endpoint_tokens = [
        ("endpoint", sec, endpoint) for sec in sections for endpoint in (0, 1)
    ]
    material_nodes = list(id2seg)
    union_find = nx.utils.UnionFind(endpoint_tokens + material_nodes)

    for child_sec in sections:
        if child_sec is root_sec:
            # A supplied subtree root may itself have an external parent.  That
            # connection is intentionally cut and its endpoint becomes sealed.
            continue
        parent_loc = child_sec.parentseg()
        if parent_loc is None or parent_loc.sec not in section_set:
            continue

        child_endpoint = _section_orientation(child_sec)
        child_token = ("endpoint", child_sec, child_endpoint)
        parent_x = float(parent_loc.x)
        if parent_x == 0.0 or parent_x == 1.0:
            parent_target = ("endpoint", parent_loc.sec, int(parent_x))
        else:
            parent_idx = _segment_index_for_location(parent_loc.sec, parent_x)
            parent_target = segkey2id[(parent_loc.sec, parent_idx)]
        union_find.union(child_token, parent_target)

    groups = {}
    for element in endpoint_tokens + material_nodes:
        groups.setdefault(union_find[element], []).append(element)

    endpoint_to_graph_node = {}
    provisional_junctions = []
    next_node = len(material_nodes)
    for members in groups.values():
        centers = [member for member in members if isinstance(member, int)]
        endpoints = [member for member in members if not isinstance(member, int)]
        if len(centers) > 1:
            raise ValueError(
                "A NEURON connection identified multiple membrane compartments "
                f"as one node: {centers!r}."
            )
        if centers:
            graph_node = centers[0]
        else:
            graph_node = next_node
            next_node += 1
            provisional_junctions.append(graph_node)
            _, rep_sec, rep_endpoint = endpoints[0]
            rep_seg = rep_sec(float(rep_endpoint))
            id2seg[graph_node] = rep_seg
            resistor_graph.add_node(
                graph_node,
                **_branchpoint_node_attrs(
                    rep_seg, data_func=data_func, attach_objects=attach_objects
                ),
            )
        for endpoint in endpoints:
            endpoint_to_graph_node[endpoint] = graph_node

    # ------------------------------------------------------------------
    # Centre-to-centre cable edges within each section.
    # ------------------------------------------------------------------
    for sec in sections:
        for left_idx in range(int(sec.nseg) - 1):
            right_idx = left_idx + 1
            left_seg = _segment_center(sec, left_idx)
            right_seg = _segment_center(sec, right_idx)
            inv_area = _section_inv_area_between_x(sec, left_seg.x, right_seg.x)
            attrs = _edge_attrs_from_path(
                R_ohm_value=_same_section_resistance_ohm(left_seg, right_seg),
                L_um=_safe_distance_um(left_seg, right_seg),
                inv_area_um_inv=inv_area,
            )
            _add_resistor_edge(
                resistor_graph,
                segkey2id[(sec, left_idx)],
                segkey2id[(sec, right_idx)],
                attrs,
            )

    # ------------------------------------------------------------------
    # Half-segment edges from each section centre to each physical endpoint.
    # Endpoint groups attached to an interior parent location already resolve
    # directly to that parent material node.
    # ------------------------------------------------------------------
    for sec in sections:
        for endpoint in (0, 1):
            idx = 0 if endpoint == 0 else int(sec.nseg) - 1
            center_seg = _segment_center(sec, idx)
            center_node = segkey2id[(sec, idx)]
            endpoint_node = endpoint_to_graph_node[("endpoint", sec, endpoint)]
            inv_area = _section_inv_area_between_x(sec, center_seg.x, endpoint)
            attrs = _edge_attrs_from_path(
                R_ohm_value=_center_to_endpoint_resistance_ohm(sec, endpoint),
                L_um=_safe_distance_um(center_seg, sec(float(endpoint))),
                inv_area_um_inv=inv_area,
            )
            _add_resistor_edge(resistor_graph, center_node, endpoint_node, attrs)

    # ------------------------------------------------------------------
    # A sealed endpoint is a degree-one zero-area node and has no electrical
    # effect.  A degree-two endpoint is exactly a pair of series resistors and
    # can be collapsed.  True cable junctions (degree >= 3) remain explicit.
    # ------------------------------------------------------------------
    for junction in list(provisional_junctions):
        if junction not in resistor_graph:
            continue
        degree = resistor_graph.degree(junction)
        if degree <= 1:
            resistor_graph.remove_node(junction)
            id2seg.pop(junction, None)
            continue
        if degree == 2:
            first_node, second_node = list(resistor_graph.neighbors(junction))
            first_attrs = dict(resistor_graph.edges[first_node, junction])
            second_attrs = dict(resistor_graph.edges[junction, second_node])
            combined = _series_edge_attrs(first_attrs, second_attrs)
            resistor_graph.remove_node(junction)
            id2seg.pop(junction, None)
            _add_resistor_edge(resistor_graph, first_node, second_node, combined)

    root_endpoint = _section_orientation(root_sec)
    root_idx = 0 if root_endpoint == 0 else int(root_sec.nseg) - 1
    root_node = segkey2id[(root_sec, root_idx)]
    G = _orient_resistor_tree(resistor_graph, root_node)

    # Assign stable branchpoint names only to junctions that survived the
    # degree-one/two simplification.
    retained_junctions = [node for node in provisional_junctions if node in G]
    for branch_index, node in enumerate(retained_junctions):
        G.nodes[node]["name"] = f"branchpoint.{branch_index}"

    # Internal bookkeeping is no longer needed after all series collapses.
    for _, _, attrs in G.edges(data=True):
        attrs.pop("_inv_area_um_inv", None)

    patterns = {
        "DEND": r"dend",
        "APIC": r"apic",
        "SOMA": r"soma",
        "UNMYELIN": r"unmyelin",
        "MYELIN": r"\bmyelin\b",
        "AXON": r"axon",
        "NODE": r"node",
    }
    group_order = ["APIC", "DEND", "SOMA", "AXON", "UNMYELIN", "NODE", "MYELIN"]

    G, relabel_mapping = reorder_graph_by_patterns(G, patterns, group_order)
    id2seg = regenerate_id_map(id2seg, relabel_mapping)
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
    """Deprecated compatibility hook.

    Branchpoints are now constructed from NEURON section endpoints inside
    :func:`neuron_to_dendra_graph`; no post-hoc graph rewrite is required.
    The function remains as a no-op for callers that imported it directly.
    """
    return G
