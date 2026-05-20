import colorsys
import random
import warnings
from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy.interpolate import griddata

try:
    import plotly.graph_objects as go
except ImportError:
    pass

from ..helpers import requires_packages


def _register_parula(name="parula", N=256):
    """
    Ensure a MATLAB-style *parula* colormap is registered in matplotlib.

    Returns
    -------
    matplotlib.colors.Colormap
    """
    anchor = np.array(
        [
            [0.2422, 0.1504, 0.6603],
            [0.2444, 0.1534, 0.6728],
            [0.2464, 0.1569, 0.6847],
            [0.2484, 0.1607, 0.6961],
            [0.2503, 0.1648, 0.7071],
            [0.2522, 0.1689, 0.7179],
            [0.254, 0.1732, 0.7286],
            [0.2558, 0.1773, 0.7393],
            [0.2576, 0.1814, 0.7501],
            [0.2594, 0.1854, 0.761],
            [0.2611, 0.1893, 0.7719],
            [0.2628, 0.1932, 0.7828],
            [0.2645, 0.1972, 0.7937],
            [0.2661, 0.2011, 0.8043],
            [0.2676, 0.2052, 0.8148],
            [0.2691, 0.2094, 0.8249],
            [0.2704, 0.2138, 0.8346],
            [0.2717, 0.2184, 0.8439],
            [0.2729, 0.2231, 0.8528],
            [0.274, 0.228, 0.8612],
            [0.2749, 0.233, 0.8692],
            [0.2758, 0.2382, 0.8767],
            [0.2766, 0.2435, 0.884],
            [0.2774, 0.2489, 0.8908],
            [0.2781, 0.2543, 0.8973],
            [0.2788, 0.2598, 0.9035],
            [0.2794, 0.2653, 0.9094],
            [0.2798, 0.2708, 0.915],
            [0.2802, 0.2764, 0.9204],
            [0.2806, 0.2819, 0.9255],
            [0.2809, 0.2875, 0.9305],
            [0.2811, 0.293, 0.9352],
            [0.2813, 0.2985, 0.9397],
            [0.2814, 0.304, 0.9441],
            [0.2814, 0.3095, 0.9483],
            [0.2813, 0.315, 0.9524],
            [0.2811, 0.3204, 0.9563],
            [0.2809, 0.3259, 0.96],
            [0.2807, 0.3313, 0.9636],
            [0.2803, 0.3367, 0.967],
            [0.2798, 0.3421, 0.9702],
            [0.2791, 0.3475, 0.9733],
            [0.2784, 0.3529, 0.9763],
            [0.2776, 0.3583, 0.9791],
            [0.2766, 0.3638, 0.9817],
            [0.2754, 0.3693, 0.984],
            [0.2741, 0.3748, 0.9862],
            [0.2726, 0.3804, 0.9881],
            [0.271, 0.386, 0.9898],
            [0.2691, 0.3916, 0.9912],
            [0.267, 0.3973, 0.9924],
            [0.2647, 0.403, 0.9935],
            [0.2621, 0.4088, 0.9946],
            [0.2591, 0.4145, 0.9955],
            [0.2556, 0.4203, 0.9965],
            [0.2517, 0.4261, 0.9974],
            [0.2473, 0.4319, 0.9983],
            [0.2424, 0.4378, 0.9991],
            [0.2369, 0.4437, 0.9996],
            [0.2311, 0.4497, 0.9995],
            [0.225, 0.4559, 0.9985],
            [0.2189, 0.462, 0.9968],
            [0.2128, 0.4682, 0.9948],
            [0.2066, 0.4743, 0.9926],
            [0.2006, 0.4803, 0.9906],
            [0.195, 0.4861, 0.9887],
            [0.1903, 0.4919, 0.9867],
            [0.1869, 0.4975, 0.9844],
            [0.1847, 0.503, 0.9819],
            [0.1831, 0.5084, 0.9793],
            [0.1818, 0.5138, 0.9766],
            [0.1806, 0.5191, 0.9738],
            [0.1795, 0.5244, 0.9709],
            [0.1785, 0.5296, 0.9677],
            [0.1778, 0.5349, 0.9641],
            [0.1773, 0.5401, 0.9602],
            [0.1768, 0.5452, 0.956],
            [0.1764, 0.5504, 0.9516],
            [0.1755, 0.5554, 0.9473],
            [0.174, 0.5605, 0.9432],
            [0.1716, 0.5655, 0.9393],
            [0.1686, 0.5705, 0.9357],
            [0.1649, 0.5755, 0.9323],
            [0.161, 0.5805, 0.9289],
            [0.1573, 0.5854, 0.9254],
            [0.154, 0.5902, 0.9218],
            [0.1513, 0.595, 0.9182],
            [0.1492, 0.5997, 0.9147],
            [0.1475, 0.6043, 0.9113],
            [0.1461, 0.6089, 0.908],
            [0.1446, 0.6135, 0.905],
            [0.1429, 0.618, 0.9022],
            [0.1408, 0.6226, 0.8998],
            [0.1383, 0.6272, 0.8975],
            [0.1354, 0.6317, 0.8953],
            [0.1321, 0.6363, 0.8932],
            [0.1288, 0.6408, 0.891],
            [0.1253, 0.6453, 0.8887],
            [0.1219, 0.6497, 0.8862],
            [0.1185, 0.6541, 0.8834],
            [0.1152, 0.6584, 0.8804],
            [0.1119, 0.6627, 0.877],
            [0.1085, 0.6669, 0.8734],
            [0.1048, 0.671, 0.8695],
            [0.1009, 0.675, 0.8653],
            [0.0964, 0.6789, 0.8609],
            [0.0914, 0.6828, 0.8562],
            [0.0855, 0.6865, 0.8513],
            [0.0789, 0.6902, 0.8462],
            [0.0713, 0.6938, 0.8409],
            [0.0628, 0.6972, 0.8355],
            [0.0535, 0.7006, 0.8299],
            [0.0433, 0.7039, 0.8242],
            [0.0328, 0.7071, 0.8183],
            [0.0234, 0.7103, 0.8124],
            [0.0155, 0.7133, 0.8064],
            [0.0091, 0.7163, 0.8003],
            [0.0046, 0.7192, 0.7941],
            [0.0019, 0.722, 0.7878],
            [0.0009, 0.7248, 0.7815],
            [0.0018, 0.7275, 0.7752],
            [0.0046, 0.7301, 0.7688],
            [0.0094, 0.7327, 0.7623],
            [0.0162, 0.7352, 0.7558],
            [0.0253, 0.7376, 0.7492],
            [0.0369, 0.74, 0.7426],
            [0.0504, 0.7423, 0.7359],
            [0.0638, 0.7446, 0.7292],
            [0.077, 0.7468, 0.7224],
            [0.0899, 0.7489, 0.7156],
            [0.1023, 0.751, 0.7088],
            [0.1141, 0.7531, 0.7019],
            [0.1252, 0.7552, 0.695],
            [0.1354, 0.7572, 0.6881],
            [0.1448, 0.7593, 0.6812],
            [0.1532, 0.7614, 0.6741],
            [0.1609, 0.7635, 0.6671],
            [0.1678, 0.7656, 0.6599],
            [0.1741, 0.7678, 0.6527],
            [0.1799, 0.7699, 0.6454],
            [0.1853, 0.7721, 0.6379],
            [0.1905, 0.7743, 0.6303],
            [0.1954, 0.7765, 0.6225],
            [0.2003, 0.7787, 0.6146],
            [0.2061, 0.7808, 0.6065],
            [0.2118, 0.7828, 0.5983],
            [0.2178, 0.7849, 0.5899],
            [0.2244, 0.7869, 0.5813],
            [0.2318, 0.7887, 0.5725],
            [0.2401, 0.7905, 0.5636],
            [0.2491, 0.7922, 0.5546],
            [0.2589, 0.7937, 0.5454],
            [0.2695, 0.7951, 0.536],
            [0.2809, 0.7964, 0.5266],
            [0.2929, 0.7975, 0.517],
            [0.3052, 0.7985, 0.5074],
            [0.3176, 0.7994, 0.4975],
            [0.3301, 0.8002, 0.4876],
            [0.3424, 0.8009, 0.4774],
            [0.3548, 0.8016, 0.4669],
            [0.3671, 0.8021, 0.4563],
            [0.3795, 0.8026, 0.4454],
            [0.3921, 0.8029, 0.4344],
            [0.405, 0.8031, 0.4233],
            [0.4184, 0.803, 0.4122],
            [0.4322, 0.8028, 0.4013],
            [0.4463, 0.8024, 0.3904],
            [0.4608, 0.8018, 0.3797],
            [0.4753, 0.8011, 0.3691],
            [0.4899, 0.8002, 0.3586],
            [0.5044, 0.7993, 0.348],
            [0.5187, 0.7982, 0.3374],
            [0.5329, 0.797, 0.3267],
            [0.547, 0.7957, 0.3159],
            [0.5609, 0.7943, 0.305],
            [0.5748, 0.7929, 0.2941],
            [0.5886, 0.7913, 0.2833],
            [0.6024, 0.7896, 0.2726],
            [0.6161, 0.7878, 0.2622],
            [0.6297, 0.7859, 0.2521],
            [0.6433, 0.7839, 0.2423],
            [0.6567, 0.7818, 0.2329],
            [0.6701, 0.7796, 0.2239],
            [0.6833, 0.7773, 0.2155],
            [0.6963, 0.775, 0.2075],
            [0.7091, 0.7727, 0.1998],
            [0.7218, 0.7703, 0.1924],
            [0.7344, 0.7679, 0.1852],
            [0.7468, 0.7654, 0.1782],
            [0.759, 0.7629, 0.1717],
            [0.771, 0.7604, 0.1658],
            [0.7829, 0.7579, 0.1608],
            [0.7945, 0.7554, 0.157],
            [0.806, 0.7529, 0.1546],
            [0.8172, 0.7505, 0.1535],
            [0.8281, 0.7481, 0.1536],
            [0.8389, 0.7457, 0.1546],
            [0.8495, 0.7435, 0.1564],
            [0.86, 0.7413, 0.1587],
            [0.8703, 0.7392, 0.1615],
            [0.8804, 0.7372, 0.165],
            [0.8903, 0.7353, 0.1695],
            [0.9, 0.7336, 0.1749],
            [0.9093, 0.7321, 0.1815],
            [0.9184, 0.7308, 0.189],
            [0.9272, 0.7298, 0.1973],
            [0.9357, 0.729, 0.2061],
            [0.944, 0.7285, 0.2151],
            [0.9523, 0.7284, 0.2237],
            [0.9606, 0.7285, 0.2312],
            [0.9689, 0.7292, 0.2373],
            [0.977, 0.7304, 0.2418],
            [0.9842, 0.733, 0.2446],
            [0.99, 0.7365, 0.2429],
            [0.9946, 0.7407, 0.2394],
            [0.9966, 0.7458, 0.2351],
            [0.9971, 0.7513, 0.2309],
            [0.9972, 0.7569, 0.2267],
            [0.9971, 0.7626, 0.2224],
            [0.9969, 0.7683, 0.2181],
            [0.9966, 0.774, 0.2138],
            [0.9962, 0.7798, 0.2095],
            [0.9957, 0.7856, 0.2053],
            [0.9949, 0.7915, 0.2012],
            [0.9938, 0.7974, 0.1974],
            [0.9923, 0.8034, 0.1939],
            [0.9906, 0.8095, 0.1906],
            [0.9885, 0.8156, 0.1875],
            [0.9861, 0.8218, 0.1846],
            [0.9835, 0.828, 0.1817],
            [0.9807, 0.8342, 0.1787],
            [0.9778, 0.8404, 0.1757],
            [0.9748, 0.8467, 0.1726],
            [0.972, 0.8529, 0.1695],
            [0.9694, 0.8591, 0.1665],
            [0.9671, 0.8654, 0.1636],
            [0.9651, 0.8716, 0.1608],
            [0.9634, 0.8778, 0.1582],
            [0.9619, 0.884, 0.1557],
            [0.9608, 0.8902, 0.1532],
            [0.9601, 0.8963, 0.1507],
            [0.9596, 0.9023, 0.148],
            [0.9595, 0.9084, 0.145],
            [0.9597, 0.9143, 0.1418],
            [0.9601, 0.9203, 0.1382],
            [0.9608, 0.9262, 0.1344],
            [0.9618, 0.932, 0.1304],
            [0.9629, 0.9379, 0.1261],
            [0.9642, 0.9437, 0.1216],
            [0.9657, 0.9494, 0.1168],
            [0.9674, 0.9552, 0.1116],
            [0.9692, 0.9609, 0.1061],
            [0.9711, 0.9667, 0.1001],
            [0.973, 0.9724, 0.0938],
            [0.9749, 0.9782, 0.0872],
            [0.9769, 0.9839, 0.0805],
        ]
    )

    parula = LinearSegmentedColormap.from_list(name, anchor, N=N)
    try:
        mpl.colormaps.register(name=name, cmap=parula)
    except ValueError:
        pass

    parula_r = parula.reversed(name + "_r")

    try:
        mpl.colormaps.register(name=name + "_r", cmap=parula_r)
    except ValueError:
        pass

    return parula


_register_parula()  # Register parula colormap globally


def generate_colors(n):
    return cm.get_cmap("tab10").colors[:n]


def palette_hsv(n, *, s=0.65, v=0.9, seed=0):
    """Return n colours as hex strings by uniform hue spacing in HSV space."""
    random.seed(seed)  # repeatable order if desired
    hues = [i / n for i in range(n)]
    random.shuffle(hues)  # break up any obvious gradients
    return [mpl.colors.to_hex(colorsys.hsv_to_rgb(h, s, v)) for h in hues]


def _nice_scalebar_length(span):
    """Choose a visually sensible scalebar length in data units."""
    span = float(span)
    if not np.isfinite(span) or span <= 0:
        return 1.0
    target = 0.2 * span
    exp = 10 ** np.floor(np.log10(target))
    candidates = exp * np.array([1.0, 2.0, 5.0, 10.0])
    valid = candidates[candidates <= target * (1.0 + 1e-12)]
    return float(valid[-1] if valid.size else candidates[0])


def _format_scalebar_label(length_um):
    length_um = float(length_um)
    if abs(length_um) >= 1000:
        val = length_um / 1000.0
        text = f"{val:g} mm"
    else:
        text = f"{length_um:g} µm"
    return text


def _hide_2d_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _add_2d_scalebar(
    ax,
    xlim,
    ylim,
    *,
    length=None,
    loc="lower right",
    pad_frac=0.05,
    text_pad_frac=0.01,
    linewidth=2.5,
    color="black",
    fontsize=10,
):
    """Draw a simple horizontal scalebar in data coordinates."""
    x0, x1 = map(float, xlim)
    y0, y1 = map(float, ylim)
    xspan = max(x1 - x0, 1.0)
    yspan = max(y1 - y0, 1.0)

    if length is None:
        length = _nice_scalebar_length(xspan)
    length = float(length)

    pad_x = pad_frac * xspan
    pad_y = pad_frac * yspan
    text_pad = text_pad_frac * yspan

    loc = str(loc).lower().replace("_", " ")
    if loc in {"lower right", "right", "lr"}:
        xs = x1 - pad_x - length
        xe = x1 - pad_x
    elif loc in {"lower left", "left", "ll"}:
        xs = x0 + pad_x
        xe = x0 + pad_x + length
    else:
        raise ValueError("scalebar_loc must be 'lower right' or 'lower left'")

    y = y0 + pad_y
    ax.plot(
        [xs, xe], [y, y], color=color, lw=linewidth, solid_capstyle="butt", zorder=20
    )
    # ax.plot([xs, xs], [y - tick, y + tick], color=color, lw=linewidth, zorder=20)
    # ax.plot([xe, xe], [y - tick, y + tick], color=color, lw=linewidth, zorder=20)
    ax.text(
        0.5 * (xs + xe),
        y + text_pad,
        _format_scalebar_label(length),
        ha="center",
        va="bottom",
        color=color,
        fontsize=fontsize,
        zorder=21,
    )


def _build_3d_scalebar(
    coords,
    *,
    length=None,
    pad_frac=0.06,
    axis=None,
    color="black",
    line_width=6,
    font_size=14,
):
    """Create a world-space 3D scalebar trace and matching annotation."""
    pts = np.asarray(list(coords.values()), dtype=float)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    spans = np.maximum(maxs - mins, 1.0)

    if axis is None:
        axis_idx = int(np.argmax(spans))
    else:
        axis_lookup = {"x": 0, "y": 1, "z": 2}
        axis_idx = axis_lookup[str(axis).lower()]

    if length is None:
        length = _nice_scalebar_length(spans[axis_idx])
    length = float(length)

    start = mins + pad_frac * spans
    end = start.copy()
    end[axis_idx] = start[axis_idx] + length

    tick_axis = 2 if axis_idx != 2 else 1
    tick_half = 0.02 * spans[tick_axis]

    tick1a = start.copy()
    tick1b = start.copy()
    tick1a[tick_axis] -= tick_half
    tick1b[tick_axis] += tick_half
    tick2a = end.copy()
    tick2b = end.copy()
    tick2a[tick_axis] -= tick_half
    tick2b[tick_axis] += tick_half

    trace = go.Scatter3d(
        x=[start[0], end[0], None, tick1a[0], tick1b[0], None, tick2a[0], tick2b[0]],
        y=[start[1], end[1], None, tick1a[1], tick1b[1], None, tick2a[1], tick2b[1]],
        z=[start[2], end[2], None, tick1a[2], tick1b[2], None, tick2a[2], tick2b[2]],
        mode="lines",
        line=dict(width=float(line_width), color=color),
        hoverinfo="skip",
        showlegend=False,
        name="scalebar",
    )

    label_pos = 0.5 * (start + end)
    label_pos[tick_axis] += 2.5 * tick_half
    annotation = dict(
        x=float(label_pos[0]),
        y=float(label_pos[1]),
        z=float(label_pos[2]),
        text=_format_scalebar_label(length),
        showarrow=False,
        font=dict(size=int(font_size), color=color),
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.0)",
    )
    return trace, annotation


def vis_2d(
    cell,
    idx=0,
    view="y",
    node_scale=8,
    dpi=200,
    *,
    fig=None,
    ax=None,
    show_nodes=True,
    show_edges=True,
    show_legend=True,
    edge_alpha=0.85,
    node_alpha=0.7,
    edge_scale=0.12,
    min_edge_width=0.35,
    max_edge_width=None,
    edge_color="k",
    scalebar: bool = False,
    scalebar_length=None,
    scalebar_loc: str = "lower right",
    scalebar_color: str = "black",
    scalebar_linewidth: float = 2.5,
    scalebar_fontsize: float = 10,
):
    """
    Project a 3-D morphology graph to 2-D and plot it with matplotlib.

    Parameters
    ----------
    cell : Dendra Population
        The cell object containing the graph, labels, and ``find`` method.
    idx : int, optional
        Population instance index to visualize.
    view : {'x', 'y', 'z'}, optional
        Axis to project along. Default ``'y'`` (drop y -> plot x-z).
    node_scale : float, optional
        Factor converting compartment diameter (µm) to marker area.
    dpi : int, optional
        Figure resolution in dots per inch.
    fig, ax : matplotlib objects, optional
        Existing figure/axes to draw into. If omitted, a new figure is created.
    show_nodes, show_edges, show_legend : bool, optional
        Toggle display of node markers, graph edges, and legend.
    edge_alpha, node_alpha : float, optional
        Transparency for edges and node markers.
    edge_scale : float, optional
        Scale factor converting local diameter (µm) to line width.
    min_edge_width : float, optional
        Minimum visible line width for edges.
    max_edge_width : float or None, optional
        Optional cap on edge line width.
    edge_color : matplotlib color, optional
        Fallback color for edges.
    scalebar : bool, optional
        If True, hide axes and draw a 2D scalebar instead.
    scalebar_length : float or None, optional
        Scalebar length in µm. If omitted, a visually sensible value is chosen.
    scalebar_loc : {'lower right', 'lower left'}, optional
        Placement of the scalebar.

    Returns
    -------
    (fig, ax)
        The matplotlib figure and axes.
    """
    G = cell.graph
    if G is None:
        raise ValueError("Graph is None. Please create a graph first.")

    x = cell.x[idx]
    y = cell.y[idx]
    z = cell.z[idx]

    coords = {n: (float(x[n]), float(y[n]), float(z[n])) for n in G.nodes}

    # Orthographic projection
    if view == "z":
        x_l, y_l = "x", "y"
        proj = {n: (c[0], c[1]) for n, c in coords.items()}
    elif view == "y":
        x_l, y_l = "x", "z"
        proj = {n: (c[0], c[2]) for n, c in coords.items()}
    elif view == "x":
        x_l, y_l = "y", "z"
        proj = {n: (c[1], c[2]) for n, c in coords.items()}
    else:
        raise ValueError("view must be 'x', 'y', or 'z'")

    labels = list(cell._labels)
    color_choices = palette_hsv(max(len(labels), 1))
    label_to_color = {label: color for label, color in zip(labels, color_choices)}

    created_fig = ax is None and fig is None
    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(dpi=dpi, figsize=(8, 8))
        else:
            ax = fig.gca()
    else:
        fig = ax.figure if fig is None else fig

    # Build a reverse node -> label map
    indices = torch.arange(len(G.nodes()), device=cell.device(), dtype=torch.long)
    node_to_label = {}
    for label in labels:
        nodes_for_label = indices[cell.find(label)].detach().cpu().tolist()
        for node_id in nodes_for_label:
            node_to_label[node_id] = label

    # Draw edges first with widths tied to local diameter.
    if show_edges and G.number_of_edges() > 0:
        segments = []
        widths = []
        for u, v in G.edges():
            x0, y0 = proj[u]
            x1, y1 = proj[v]
            segments.append([(x0, y0), (x1, y1)])

            diam_u = float(G.nodes[u].get("diam", 1.0) or 1.0)
            diam_v = float(G.nodes[v].get("diam", 1.0) or 1.0)
            edge_width = 0.5 * (diam_u + diam_v) * edge_scale
            edge_width = max(float(min_edge_width), float(edge_width))
            if max_edge_width is not None:
                edge_width = min(float(max_edge_width), edge_width)
            widths.append(edge_width)

        lc = LineCollection(
            segments,
            colors=edge_color,
            linewidths=widths,
            alpha=edge_alpha,
            zorder=1,
            capstyle="round",
            joinstyle="round",
        )
        ax.add_collection(lc)

    # Group nodes by label for plotting/legend entries
    data_by_label = {label: {"xs": [], "ys": [], "sizes": []} for label in labels}
    unclassified_nodes = {"xs": [], "ys": [], "sizes": []}

    for n in G.nodes():
        label = node_to_label.get(n)
        px, py = proj[n]
        diam = float(G.nodes[n].get("diam", 1.0) or 1.0)
        size = max(diam, 0.05) * node_scale

        if label is not None:
            data_by_label[label]["xs"].append(px)
            data_by_label[label]["ys"].append(py)
            data_by_label[label]["sizes"].append(size)
        else:
            unclassified_nodes["xs"].append(px)
            unclassified_nodes["ys"].append(py)
            unclassified_nodes["sizes"].append(size)

    if show_nodes:
        for label, data in data_by_label.items():
            if not data["xs"]:
                continue
            ax.scatter(
                data["xs"],
                data["ys"],
                s=data["sizes"],
                c=[label_to_color[label]],
                label=label,
                alpha=node_alpha,
                edgecolors="k",
                linewidths=0.25,
                zorder=3,
            )

        if unclassified_nodes["xs"]:
            ax.scatter(
                unclassified_nodes["xs"],
                unclassified_nodes["ys"],
                s=unclassified_nodes["sizes"],
                c="grey",
                label="unclassified",
                alpha=max(0.5, node_alpha - 0.15),
                edgecolors="k",
                linewidths=0.25,
                zorder=3,
            )

    if show_legend and show_nodes:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0.0,
            frameon=False,
        )

    # Set limits from projected coordinates, with a small padding
    proj_arr = np.asarray(list(proj.values()), dtype=float)
    if proj_arr.size:
        mins = proj_arr.min(axis=0)
        maxs = proj_arr.max(axis=0)
        spans = np.maximum(maxs - mins, 1.0)
        pad = 0.03 * spans
        xlim = (mins[0] - pad[0], maxs[0] + pad[0])
        ylim = (mins[1] - pad[1], maxs[1] + pad[1])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    else:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

    ax.set_aspect("equal")
    if scalebar:
        _hide_2d_axes(ax)
        _add_2d_scalebar(
            ax,
            xlim,
            ylim,
            length=scalebar_length,
            loc=scalebar_loc,
            linewidth=scalebar_linewidth,
            color=scalebar_color,
            fontsize=scalebar_fontsize,
        )
    else:
        ax.set_xlabel(f"{x_l} (µm)")
        ax.set_ylabel(f"{y_l} (µm)")

    fig.tight_layout()
    if created_fig:
        plt.show()
    return fig, ax


@requires_packages("plotly")
def vis_3d(
    cell,
    idx=0,
    node_scale: float = 4.0,
    height=800.0,
    width=None,
    show_nodes: bool = True,
    show_edges: bool = True,
    show_legend: bool = True,
    node_alpha: float = 0.75,
    edge_alpha: float = 0.7,
    edge_color: str = "black",
    edge_scale: float = 0.5,
    min_edge_width: float = 1.0,
    max_edge_width: float = 6.0,
    min_node_size: float = 2.0,
    max_node_size: float = 18.0,
    size_mode: str = "sqrt",
    background: str = "white",
    show_axes: bool = True,
    scalebar: bool = False,
    scalebar_length=None,
    scalebar_axis: str = None,
    scalebar_color: str = "black",
    scalebar_linewidth: float = 6.0,
    scalebar_fontsize: float = 14.0,
    show: bool = True,
):
    """
    Plot a 3D compartment graph interactively using Plotly.

    Parameters
    ----------
    scalebar : bool, optional
        If True, hide axes and add a small world-space scalebar instead.
    scalebar_length : float or None, optional
        Scalebar length in µm. If omitted, a visually sensible value is chosen.
    scalebar_axis : {'x', 'y', 'z'} or None, optional
        Axis along which to draw the 3D scalebar. Default picks the widest span.
    """
    G = cell.graph
    if not isinstance(G, nx.Graph) or G.number_of_nodes() == 0:
        raise ValueError("Graph is not a valid or non-empty NetworkX graph.")

    x = cell.x[idx]
    y = cell.y[idx]
    z = cell.z[idx]

    coords = {n: (float(x[n]), float(y[n]), float(z[n])) for n in G.nodes}
    node_list = list(G.nodes())

    labels = list(cell._labels)
    color_choices = palette_hsv(len(labels)) if labels else []
    label_to_color = {label: color for label, color in zip(labels, color_choices)}

    # Build a reverse node -> label map robustly.
    indices = torch.arange(len(G.nodes()), device=cell.device(), dtype=torch.long)
    node_to_label: Dict[int, str] = {}
    for label in labels:
        label_idx_obj = cell.find(label)
        if isinstance(label_idx_obj, slice):
            label_indices = indices[label_idx_obj]
        elif torch.is_tensor(label_idx_obj) and label_idx_obj.dtype == torch.bool:
            label_indices = indices[label_idx_obj]
        else:
            label_indices = label_idx_obj

        for node_id in label_indices.detach().cpu().tolist():
            node_to_label[int(node_id)] = label

    def _scale_node_sizes(diams):
        diams = np.clip(np.asarray(diams, dtype=float), 0.05, None)
        if size_mode == "sqrt":
            sizes = np.sqrt(diams) * float(node_scale)
        elif size_mode == "linear":
            sizes = diams * float(node_scale)
        elif size_mode == "log":
            sizes = np.log10(diams + 1.0) * float(node_scale)
        else:
            raise ValueError("size_mode must be 'sqrt', 'linear', or 'log'")
        return np.clip(sizes, float(min_node_size), float(max_node_size))

    def _node_hover_text(node_id):
        attrs = G.nodes[node_id]
        name = attrs.get("name", "N/A")
        diam = float(attrs.get("diam", 0.0) or 0.0)
        length = float(attrs.get("L", 0.0) or 0.0)
        area = float(attrs.get("area", 0.0) or 0.0)
        label = node_to_label.get(node_id, "unclassified")
        cx, cy, cz = coords[node_id]
        return (
            f"<b>Node ID: {node_id}</b><br>"
            f"Label: {label}<br>"
            f"Name: {name}<br>"
            f"Diameter: {diam:.2f} µm<br>"
            f"Length: {length:.2f} µm<br>"
            f"Area: {area:.2f} µm²<br>"
            f"x, y, z: ({cx:.2f}, {cy:.2f}, {cz:.2f}) µm"
        )

    traces = []

    if show_edges and G.number_of_edges() > 0:
        edge_records = []
        edge_widths = []
        for u, v in G.edges():
            diam_u = float(G.nodes[u].get("diam", 1.0) or 1.0)
            diam_v = float(G.nodes[v].get("diam", 1.0) or 1.0)
            edge_width = 0.5 * (diam_u + diam_v) * float(edge_scale)
            edge_width = max(float(min_edge_width), edge_width)
            if max_edge_width is not None:
                edge_width = min(float(max_edge_width), edge_width)
            edge_records.append((u, v, edge_width))
            edge_widths.append(edge_width)

        edge_widths = np.asarray(edge_widths, dtype=float)
        if len(edge_records) <= 6 or np.allclose(edge_widths.min(), edge_widths.max()):
            bin_ids = np.zeros(len(edge_records), dtype=int)
        else:
            nbins = min(6, len(edge_records))
            bins = np.linspace(edge_widths.min(), edge_widths.max(), nbins + 1)
            bin_ids = np.digitize(edge_widths, bins[1:-1], right=True)

        for bin_id in sorted(set(bin_ids.tolist())):
            edge_x, edge_y, edge_z = [], [], []
            bin_widths = []
            for (u, v, edge_width), cur_bin in zip(edge_records, bin_ids):
                if int(cur_bin) != int(bin_id):
                    continue
                edge_x.extend([coords[u][0], coords[v][0], None])
                edge_y.extend([coords[u][1], coords[v][1], None])
                edge_z.extend([coords[u][2], coords[v][2], None])
                bin_widths.append(edge_width)

            if not edge_x:
                continue

            traces.append(
                go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode="lines",
                    hoverinfo="skip",
                    showlegend=False,
                    opacity=edge_alpha,
                    line=dict(
                        width=float(np.median(bin_widths)),
                        color=edge_color,
                    ),
                )
            )

    if show_nodes:
        grouped_nodes = {label: [] for label in labels}
        unclassified = []
        for n in node_list:
            label = node_to_label.get(n)
            if label is None:
                unclassified.append(n)
            else:
                grouped_nodes[label].append(n)

        for label, nodes in grouped_nodes.items():
            if not nodes:
                continue
            traces.append(
                go.Scatter3d(
                    x=[coords[n][0] for n in nodes],
                    y=[coords[n][1] for n in nodes],
                    z=[coords[n][2] for n in nodes],
                    mode="markers",
                    name=label,
                    legendgroup=label,
                    showlegend=show_legend,
                    hoverinfo="text",
                    text=[_node_hover_text(n) for n in nodes],
                    opacity=node_alpha,
                    marker=dict(
                        showscale=False,
                        color=label_to_color[label],
                        size=_scale_node_sizes(
                            [G.nodes[n].get("diam", 1.0) for n in nodes]
                        ),
                        line=dict(width=0.3, color="black"),
                    ),
                )
            )

        if unclassified:
            traces.append(
                go.Scatter3d(
                    x=[coords[n][0] for n in unclassified],
                    y=[coords[n][1] for n in unclassified],
                    z=[coords[n][2] for n in unclassified],
                    mode="markers",
                    name="unclassified",
                    legendgroup="unclassified",
                    showlegend=show_legend,
                    hoverinfo="text",
                    text=[_node_hover_text(n) for n in unclassified],
                    opacity=max(0.5, node_alpha - 0.15),
                    marker=dict(
                        showscale=False,
                        color="grey",
                        size=_scale_node_sizes(
                            [G.nodes[n].get("diam", 1.0) for n in unclassified]
                        ),
                        line=dict(width=0.3, color="black"),
                    ),
                )
            )

    scene_show_axes = bool(show_axes and not scalebar)
    axis_template = dict(
        visible=scene_show_axes,
        showbackground=False,
        showgrid=scene_show_axes,
        zeroline=False,
    )

    fig_height = None if height is None else int(round(float(height)))
    fig_width = None if width is None else int(round(float(width)))

    scene_dict = dict(
        xaxis=dict(title="X (µm)", **axis_template),
        yaxis=dict(title="Y (µm)", **axis_template),
        zaxis=dict(title="Z (µm)", **axis_template),
        aspectmode="data",
        bgcolor=background,
    )

    if scalebar:
        scalebar_trace, scalebar_annotation = _build_3d_scalebar(
            coords,
            length=scalebar_length,
            axis=scalebar_axis,
            color=scalebar_color,
            line_width=scalebar_linewidth,
            font_size=scalebar_fontsize,
        )
        traces.append(scalebar_trace)
        scene_dict["annotations"] = [scalebar_annotation]

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            height=fig_height,
            width=fig_width,
            showlegend=show_legend and show_nodes,
            hovermode="closest",
            paper_bgcolor=background,
            plot_bgcolor=background,
            margin=dict(b=20, l=5, r=5, t=40),
            scene=scene_dict,
        ),
    )

    if show:
        fig.show()
    return fig


@requires_packages("plotly")
def vis_voltage_3d(x, y, z, voltage, height=800, width=None):
    # 1. Create the 3D scatter plot object
    #    The configuration is done inside the 'go.Scatter3d' call.
    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",  # We want to plot points, not lines
        marker=dict(
            size=5,  # Size of the markers
            color=voltage,  # Set color to our voltage data
            colorscale="Viridis",  # One of Plotly's built-in colorscales
            showscale=True,  # We want to show the color bar
            colorbar=dict(
                title="Voltage (mV)"  # Title for the color bar
            ),
        ),
    )

    # 2. Create a layout object to configure the plot's appearance
    layout = go.Layout(
        height=height,  # Set the height of the plot
        width=width,  # Set the width of the plot
        scene=dict(
            xaxis=dict(title="x (μm)"),
            yaxis=dict(title="y (μm)"),
            zaxis=dict(title="z (μm)"),
        ),
        margin=dict(l=0, r=0, b=0, t=40),  # Adjust margins
    )

    # 3. Create a figure and add the trace and layout
    fig = go.Figure(data=[trace], layout=layout)

    # 4. Show the figure
    #    This will open an interactive plot in your web browser or in your
    #    Jupyter Notebook / VS Code output cell.
    fig.show()


def vis_voltage_2d(
    x,
    y,
    z,
    voltage,
    view="y",
    node_scale=8,
    dpi=200,
    fig=None,
    cbar=True,
    cmap=None,
    norm=None,
):
    """
    Visualizes 3D voltage data in a 2D projection using matplotlib.

    Parameters
    ----------
    x, y, z : array-like
        Coordinates of the points.
    voltage : array-like
        Voltage values at each point.
    view : {'x', 'y', 'z'}, optional
        Axis to project along. Default 'z' (drop z -> use x-y).
    node_scale : float, optional
        Factor that converts diam (µm) to matplotlib marker area points².
    dpi : int, optional
        The resolution of the figure in dots per inch.
    fig : matplotlib.figure.Figure, optional
        If provided, plot into this figure; otherwise create a new one.
    cbar : bool, optional
        Whether to display a colorbar. Default is True.
    cmap : matplotlib.colors.Colormap, optional
        Colormap to use for voltage values. Default is 'viridis'.
    norm : matplotlib.colors.Normalize, optional
        Normalization for the colormap. Default is None.
    """
    if fig is None:
        fig, ax = plt.subplots(dpi=dpi, figsize=(8, 8))
    else:
        ax = fig.gca()

    if cmap is None:
        cmap = plt.get_cmap("viridis")

    # We will capture the output of ax.scatter into a variable, let's call it `sc`.
    if view == "z":
        sc = ax.scatter(
            x, y, c=voltage, s=node_scale * 10, cmap=cmap, norm=norm, alpha=0.7
        )
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
    elif view == "y":
        sc = ax.scatter(
            x, z, c=voltage, s=node_scale * 10, cmap=cmap, norm=norm, alpha=0.7
        )
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("z (µm)")
    elif view == "x":
        sc = ax.scatter(
            y, z, c=voltage, s=node_scale * 10, cmap=cmap, norm=norm, alpha=0.7
        )
        ax.set_xlabel("y (µm)")
        ax.set_ylabel("z (µm)")
    else:
        raise ValueError("view must be 'x', 'y', or 'z'")

    if cbar:
        fig.colorbar(sc, label="Voltage (mV)", ax=ax)

    fig.tight_layout()
    return fig, ax


def _relabel_longitude_ticks(
    ax, azimuth_offset, *, offset_in_degrees=False, every_deg=30, fmt="{:d}°"
):
    """
    Update the x-axis tick *labels* of a Mollweide axis so that they
    reflect the specified azimuthal rotation.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Mollweide axis returned by `plot_threshold_mollweide`.
    azimuth_offset : float
        Same value you passed earlier (positive eastward).
    offset_in_degrees : bool, optional
        Interpret `azimuth_offset` in degrees (default False → radians).
    every_deg : int, optional
        Place labels every `every_deg` degrees (default 30°).
    fmt : str or callable, optional
        How to format each degree value.  Default is "±ddd°".
        Pass a function `f(deg) -> str` for custom text.
    """
    # 1) Get current x‑tick locations (radians)
    xticks = ax.get_xticks()

    # 2) Build matching labels adjusted by the offset
    offset_deg = np.rad2deg(azimuth_offset) if not offset_in_degrees else azimuth_offset
    tick_deg = np.rad2deg(xticks)

    # shift back so labels show *unrotated* longitude
    labels = (tick_deg - offset_deg + 180) % 360 - 180  # → [‑180, 180)

    # 3) Optionally blank out labels that are not multiples of `every_deg`
    for i, d in enumerate(labels):
        if (abs(d) % every_deg) > 1e-6:  # leave small eps room
            labels[i] = np.nan  # ← NaN → empty label

    # 4) Format
    if callable(fmt):
        label_text = [fmt(d) if not np.isnan(d) else "" for d in labels]
    else:
        label_text = ["" if np.isnan(d) else fmt.format(int(d)) for d in labels]

    ax.set_xticklabels(label_text)


def vis_threshold_mollweide_2d(
    phi,
    theta,
    thr,
    *,
    # ───────── data options ─────────
    angles_in_degrees=False,
    flip_polar=True,
    azimuth_offset=-np.pi / 2,
    offset_in_degrees=False,
    normalize_to_min=False,
    mark_min=False,
    # ───────── interpolation ────────
    grid_res_deg=2.0,
    interp_method="cubic",
    # ───────── appearance ───────────
    cmap="viridis",
    vmin=None,
    vmax=None,
    ax=None,
):
    """
    Seam-free Mollweide map of thresholds on a sphere.

    Parameters
    ----------
    phi, theta, thr : 1-D array_like
        Azimuth (longitude), polar (co-latitude), threshold.
    angles_in_degrees : bool, optional
        Pass True if `phi`, `theta` are in degrees.
    flip_polar : bool, optional
        Swap North/South: θ → π = θ.
    azimuth_offset : float, optional
        Rotate longitudes eastward by this amount.
    offset_in_degrees : bool, optional
        Interpret `azimuth_offset` in degrees (otherwise radians).
    normalize_to_min : bool, optional
        Divide every threshold by its global minimum so min(thr) → 1.
    grid_res_deg : float, optional
        Lon/lat grid spacing (deg) for interpolation grid.
    interp_method : {'linear', 'nearest', 'cubic'}, optional
        Kernel for `scipy.interpolate.griddata`.
    cmap : str, optional
        Matplotlib colormap.
    ax : matplotlib.axes.Axes (projection='mollweide'), optional
        Draw into this axis; create a new one if None.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """

    # ──────────────────────────────────────────────────────────────
    # 0.  Input conversion & optional normalisation
    # ──────────────────────────────────────────────────────────────
    phi = np.asarray(phi, dtype=float)
    theta = np.asarray(theta, dtype=float)
    thr = np.asarray(thr, dtype=float)

    if angles_in_degrees:
        phi = np.deg2rad(phi)
        theta = np.deg2rad(theta)

    if normalize_to_min:
        m = np.nanmin(thr)
        thr = thr / m if m != 0 else thr

    n_pts = len(phi)

    # Optional polar flip
    if flip_polar:
        theta = np.pi - theta

    # Optional azimuthal rotation
    dλ = np.deg2rad(azimuth_offset) if offset_in_degrees else azimuth_offset
    phi = phi + dλ

    # ──────────────────────────────────────────────────────────────
    # 1.  Convert to Mollweide coords
    # ──────────────────────────────────────────────────────────────
    lon = (phi + np.pi) % (2 * np.pi) - np.pi  # wrap to [‑π, π]
    lat = 0.5 * np.pi - theta  # latitude

    # ──────────────────────────────────────────────────────────────
    # 2.  Periodic extension (duplicate at λ ± 2π)
    # ──────────────────────────────────────────────────────────────
    lon_aug = np.concatenate([lon, lon + 2 * np.pi, lon - 2 * np.pi])
    lat_aug = np.concatenate([lat, lat, lat])
    thr_aug = np.concatenate([thr, thr, thr])

    # ──────────────────────────────────────────────────────────────
    # 3.  Regular lon/lat grid
    # ──────────────────────────────────────────────────────────────
    d = np.deg2rad(grid_res_deg)
    lon_grid = np.arange(-np.pi, np.pi + 1e-12, d)
    lat_grid = np.arange(-0.5 * np.pi, 0.5 * np.pi + 1e-12, d)
    Lon, Lat = np.meshgrid(lon_grid, lat_grid)

    # ──────────────────────────────────────────────────────────────
    # 4.  Interpolate thresholds → grid
    # ──────────────────────────────────────────────────────────────
    Thr = griddata((lon_aug, lat_aug), thr_aug, (Lon, Lat), method=interp_method)
    Thr = np.ma.masked_invalid(Thr)

    # ──────────────────────────────────────────────────────────────
    # 5.  Plot
    # ──────────────────────────────────────────────────────────────
    if ax is None:
        fig = plt.figure(figsize=(8, 4.6))
        ax = fig.add_subplot(111, projection="mollweide")

    im = ax.pcolormesh(Lon, Lat, Thr, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.grid(True, alpha=0.3)

    if mark_min:
        min_idx = np.nanargmin(thr[:n_pts])  # only real points
        ax.scatter(
            lon[min_idx],
            lat[min_idx],
            marker="*",
            s=180,
            facecolor="white",
            edgecolor="black",
            linewidth=1.2,
            zorder=10,
        )

    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.07, shrink=0.8)
    cb.set_label(
        "Threshold |E| (V/m)" if not normalize_to_min else "Threshold |E| / min"
    )
    if azimuth_offset is not None:
        _relabel_longitude_ticks(
            ax,
            azimuth_offset=azimuth_offset,
            offset_in_degrees=offset_in_degrees,
            every_deg=30,
        )
    return ax


# Optional: use Matplotlib colormaps if you pass a Matplotlib name
def _mpl_to_plotly_colorscale(cmap_name_or_obj="viridis", n=256):
    try:
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        cm_obj = cm.get_cmap(cmap_name_or_obj)
        return [[i / (n - 1), mcolors.to_hex(cm_obj(i / (n - 1)))] for i in range(n)]
    except Exception:
        # Fallback to a sensible Plotly scale if matplotlib isn't available
        return "Viridis"


@requires_packages("plotly")
def _spherical_cap_patch(
    x0,
    y0,
    z0,
    *,
    R=1.0,
    alpha_deg=6.0,
    eps=1e-3,
    n=64,
    color="white",
    outline=True,
    outline_color="black",
):
    """
    Build a small spherical-cap Mesh3d centered at (x0,y0,z0) on the sphere.
    - alpha_deg: geodesic radius of the cap, in degrees
    - eps: lifts the cap slightly above the sphere (avoid z-fighting)
    - n: polygon resolution (number of segments around the rim)
    """
    # unit normal at the cap center
    nvec = np.array([x0, y0, z0], dtype=float)
    nvec /= np.linalg.norm(nvec)

    # build an orthonormal basis (u,v) in the tangent plane
    a = np.array([0.0, 0.0, 1.0]) if abs(nvec[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(a, nvec)
    u /= np.linalg.norm(u)
    v = np.cross(nvec, u)

    alpha = np.deg2rad(alpha_deg)
    Rcap = R * (1.0 + eps)

    # center vertex
    center = Rcap * nvec

    # ring vertices at geodesic radius alpha around the center
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.cos(alpha) * nvec[None, :] + np.sin(alpha) * (
        np.cos(t)[:, None] * u[None, :] + np.sin(t)[:, None] * v[None, :]
    )
    ring = Rcap * ring  # scale to lifted radius

    # assemble vertices
    verts = np.vstack([center, ring])  # shape (n+1, 3)
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]

    # triangle fan from center -> ring
    tri_i = np.zeros(n, dtype=int)
    tri_j = np.arange(1, n + 1, dtype=int)
    tri_k = np.where(tri_j < n, tri_j + 1, 1)

    traces = [
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=tri_i,
            j=tri_j,
            k=tri_k,
            color=color,
            opacity=0.9,
            flatshading=True,
            hoverinfo="skip",
            showscale=False,
            lighting=dict(ambient=0.8, specular=0.2, roughness=1.0),
            name="min patch",
        )
    ]

    if outline:
        # close the loop by repeating the first ring point
        x_ring = np.r_[ring[:, 0], ring[0, 0]]
        y_ring = np.r_[ring[:, 1], ring[0, 1]]
        z_ring = np.r_[ring[:, 2], ring[0, 2]]
        traces.append(
            go.Scatter3d(
                x=x_ring,
                y=y_ring,
                z=z_ring,
                mode="lines",
                line=dict(width=4, color=outline_color),
                hoverinfo="skip",
                name=None,
                showlegend=False,
            )
        )
    return traces


@requires_packages("plotly")
def vis_threshold_3d(
    phi,
    theta,
    thr,
    *,
    # ───────── data options ─────────
    angles_in_degrees=False,
    flip_polar=True,
    azimuth_offset=-np.pi / 2,
    offset_in_degrees=False,
    normalize_to_min=False,
    mark_min=False,
    # ───────── interpolation ────────
    grid_res_deg=2.0,
    interp_method="cubic",
    # ───────── appearance ───────────
    cmap="viridis",
    vmin=None,
    vmax=None,
    ax=None,  # ignored (Plotly), kept for signature parity
):
    """
    3D spherical visualization of thresholds using Plotly.

    Parameters
    ----------
    Same as vis_threshold_mollweide_2d, but returns a Plotly Figure.
    """
    if ax is not None:
        warnings.warn("`ax` is ignored for Plotly output; returning a Plotly Figure.")

    # ──────────────────────────────────────────────────────────────
    # 0.  Input conversion & optional normalisation
    # ──────────────────────────────────────────────────────────────
    phi = np.asarray(phi, dtype=float)
    theta = np.asarray(theta, dtype=float)
    thr = np.asarray(thr, dtype=float)

    if angles_in_degrees:
        phi = np.deg2rad(phi)
        theta = np.deg2rad(theta)

    if normalize_to_min:
        m = np.nanmin(thr)
        thr = thr / m if m != 0 else thr

    n_pts = len(phi)

    # Optional polar flip (θ → π − θ)
    if flip_polar:
        theta = np.pi - theta

    # Optional azimuthal rotation
    if azimuth_offset is not None:
        dλ = np.deg2rad(azimuth_offset) if offset_in_degrees else float(azimuth_offset)
        phi = phi + dλ

    # ──────────────────────────────────────────────────────────────
    # 1.  Convert to lon/lat (radians)
    # ──────────────────────────────────────────────────────────────
    lon = (phi + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]
    lat = 0.5 * np.pi - theta  # latitude

    # ──────────────────────────────────────────────────────────────
    # 2.  Periodic extension in longitude (seam-free interpolation)
    # ──────────────────────────────────────────────────────────────
    lon_aug = np.concatenate([lon, lon + 2 * np.pi, lon - 2 * np.pi])
    lat_aug = np.concatenate([lat, lat, lat])
    thr_aug = np.concatenate([thr, thr, thr])

    # ──────────────────────────────────────────────────────────────
    # 3.  Regular lon/lat grid
    # ──────────────────────────────────────────────────────────────
    d = np.deg2rad(grid_res_deg)
    lon_grid = np.arange(-np.pi, np.pi + 1e-12, d)
    lat_grid = np.arange(-0.5 * np.pi, 0.5 * np.pi + 1e-12, d)
    Lon, Lat = np.meshgrid(lon_grid, lat_grid)  # shapes (n_lat, n_lon)

    # ──────────────────────────────────────────────────────────────
    # 4.  Interpolate thresholds onto the lon/lat grid
    # ──────────────────────────────────────────────────────────────
    Thr = griddata((lon_aug, lat_aug), thr_aug, (Lon, Lat), method=interp_method)
    Thr = np.ma.masked_invalid(Thr)

    thr_grid = np.asarray(Thr, dtype=float)
    if isinstance(Thr, np.ma.MaskedArray):
        thr_grid = Thr.filled(np.nan)

    # ──────────────────────────────────────────────────────────────
    # 5.  Map spherical → Cartesian (unit sphere)
    # ──────────────────────────────────────────────────────────────
    R = 1.0
    X = R * np.cos(Lat) * np.cos(Lon)
    Y = R * np.cos(Lat) * np.sin(Lon)
    Z = R * np.sin(Lat)

    # ──────────────────────────────────────────────────────────────
    # 6.  Colorscale handling
    # ──────────────────────────────────────────────────────────────
    colorscale = _mpl_to_plotly_colorscale(cmap)

    # Default color limits
    cmin = np.min(Thr) if vmin is None else vmin
    cmax = np.max(Thr) if vmax is None else vmax

    # ──────────────────────────────────────────────────────────────
    # 7.  Build Plotly surface
    # ──────────────────────────────────────────────────────────────
    lon_deg = np.rad2deg(Lon)
    lat_deg = np.rad2deg(Lat)

    flat = np.column_stack((lon_deg.ravel(), lat_deg.ravel(), thr_grid.ravel()))
    hover_text = np.array(
        [
            f"lon={ld:.1f}°, lat={lt:.1f}°<br>thr={tv:.3g}" if np.isfinite(tv) else ""
            for ld, lt, tv in flat
        ],
        dtype=object,
    ).reshape(thr_grid.shape)

    surf = go.Surface(
        x=X,
        y=Y,
        z=Z,
        surfacecolor=thr_grid,
        text=hover_text,  # ← use preformatted labels
        hoverinfo="text",  # ← tell Plotly to show `text`
        cmin=cmin,
        cmax=cmax,
        colorscale=colorscale,
        colorbar=dict(
            title=(
                "Threshold |E| (V/m)" if not normalize_to_min else "Threshold |E| / min"
            ),
            len=0.75,
        ),
        showscale=True,
    )
    fig = go.Figure(data=[surf])

    # Optional: mark the global minimum (from original sample points only)
    if mark_min and np.isfinite(thr[:n_pts]).any():
        min_idx = np.nanargmin(thr[:n_pts])
        lon_min = lon[min_idx]
        lat_min = lat[min_idx]
        x0 = R * np.cos(lat_min) * np.cos(lon_min)
        y0 = R * np.cos(lat_min) * np.sin(lon_min)
        z0 = R * np.sin(lat_min)
        cap_traces = _spherical_cap_patch(
            x0,
            y0,
            z0,
            R=R,
            alpha_deg=6.0,  # size of the patch on the sphere (try 4–10°)
            eps=1.5e-3,  # lift; increase to 3e-3 if you still see z-fighting
            n=64,
            color="white",
            outline=True,
            outline_color="black",
        )
        for tr in cap_traces:
            fig.add_trace(tr)

    # ──────────────────────────────────────────────────────────────
    # 8.  Layout: equal aspect, clean axes, nice camera
    # ──────────────────────────────────────────────────────────────
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    # A gentle default camera to see the sphere well
    fig.update_layout(scene_camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)))

    return fig


def vis_morphology_by_layer(
    cell,
    *,
    threads: int = 32,
    palette_name: str = "parula",  # any qualitative palette works
    bar_fraction: float = 0.02,  # colour-bar width (fraction of plot)
    ax=None,
):
    """
    Tree layout + DHS layer colouring.
    Sequential layers cycle through a fixed qualitative palette.
    """

    # ------------------------------------------------------------------
    # 1. Build helpers
    # ------------------------------------------------------------------
    def graph_to_parent(G: nx.DiGraph):
        # ------------------------------------------------------------------
        # 0. topological order and quick look‑ups
        # ------------------------------------------------------------------
        nodes = list(nx.topological_sort(G))  # length K
        idx_of = {n: i for i, n in enumerate(nodes)}
        K = len(nodes)

        parent_idx = np.full(K, -1, dtype=np.int32)

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

        return parent_idx.tolist()

    def build_morphology(pi):
        K = len(pi)
        children = [[] for _ in range(K)]
        root = None
        for i, p in enumerate(pi):
            if p == -1:
                root = i
            else:
                children[p].append(i)
        depth = torch.zeros(K, dtype=torch.int32)
        from collections import deque

        q = deque([root])
        while q:
            u = q.popleft()
            for c in children[u]:
                depth[c] = depth[u] + 1
                q.append(c)
        return children, depth, root

    def build_dhs_layers(depth, k_threads=32):
        depth_cpu = depth.cpu().numpy()
        max_d = int(depth_cpu.max())
        order, layer_ptr = [], [0]
        bins = [[] for _ in range(max_d + 1)]
        for i, d in enumerate(depth_cpu):
            bins[d].append(i)
        for d in range(max_d, -1, -1):
            bucket = bins[d]
            for s in range(0, len(bucket), k_threads):
                chunk = bucket[s : s + k_threads]
                order.extend(chunk)
                layer_ptr.append(len(order))
        return torch.tensor(order), torch.tensor(layer_ptr)

    def compute_positions(children, root):
        pos, x_cursor = {}, 0

        def dfs(u, d):
            nonlocal x_cursor
            if not children[u]:
                pos[u] = (x_cursor, -d)
                x_cursor += 1
            else:
                for c in children[u]:
                    dfs(c, d + 1)
                xs = [pos[c][0] for c in children[u]]
                pos[u] = (sum(xs) / len(xs), -d)

        dfs(root, 0)
        return pos

    parent_idx = graph_to_parent(cell.graph)

    # ------------------------------------------------------------------
    # 2. Build the schedule information
    # ------------------------------------------------------------------
    children, depth, root = build_morphology(parent_idx)
    order, layer_ptr = build_dhs_layers(depth, k_threads=threads)
    num_layers = layer_ptr.numel() - 1

    layer_of_node = torch.full_like(depth, -1)
    for layer in range(num_layers):
        s, e = layer_ptr[layer].item(), layer_ptr[layer + 1].item()
        layer_of_node[order[s:e]] = layer

    # ------------------------------------------------------------------
    # 3. Colour map: cycle through N distinct qualitative colours
    # ------------------------------------------------------------------
    base = plt.get_cmap(palette_name)
    base_N = base.N
    if base_N > 32:
        base_N = 32
    colour_list = [base((i % base_N) / base_N) for i in range(num_layers)]
    cmap = ListedColormap(colour_list)
    # norm = BoundaryNorm(range(num_layers + 1), base_N)

    # ------------------------------------------------------------------
    # 4. Layout & drawing
    # ------------------------------------------------------------------
    pos = compute_positions(children, root)
    xs = [pos[i][0] for i in range(len(parent_idx))]
    ys = [pos[i][1] for i in range(len(parent_idx))]
    layers = layer_of_node.numpy()

    x_span = np.ptp(xs)
    y_span = np.ptp(ys)

    ratio = x_span / y_span if y_span > 0 else 1

    if ratio > 1:
        figsize = (8 * ratio, 8)
    else:
        figsize = (8, 8 / ratio)

    no_ax = False

    if ax is None:
        no_ax = True
        fig, ax = plt.subplots(figsize=figsize, dpi=200)

    # edges
    for child, p in enumerate(parent_idx):
        if p != -1:
            x1, y1 = pos[p]
            x2, y2 = pos[child]
            ax.plot([x1, x2], [y1, y2], color="lightgray", lw=0.7, zorder=0)

    scatter = ax.scatter(xs, ys, c=layers, cmap=cmap, s=15, zorder=1)

    ax.set_aspect("equal", "datalim")
    ax.axis("off")

    # narrow colour-bar
    cbar = plt.colorbar(
        scatter,
        fraction=bar_fraction,
        pad=0.02,
        ticks=range(0, num_layers, max(1, num_layers // 10)),
    )
    cbar.set_label("DHS layer (earlier → later)", rotation=90, labelpad=15)
    plt.show()

    if no_ax:
        return fig, ax
    return ax  # return the axis for further customization if needed
