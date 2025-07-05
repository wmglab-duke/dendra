import colorsys
import random

import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import networkx as nx


def generate_colors(n):
    return cm.get_cmap("tab10").colors[:n]


def palette_hsv(n, *, s=0.65, v=0.9, seed=0):
    """Return n colours as hex strings by uniform hue spacing in HSV space."""
    random.seed(seed)           # repeatable order if desired
    hues = [i / n for i in range(n)]
    random.shuffle(hues)        # break up any obvious gradients
    return [mpl.colors.to_hex(colorsys.hsv_to_rgb(h, s, v)) for h in hues]


def vis_2d(cell, view='y', node_scale=8, dpi=200):
    """
    Project a 3-D NetworkX graph to 2-D and plot with a legend.

    Parameters
    ----------
    cell : AxonML Population
        The cell object containing the graph, labels, and find method.
    view : {'x', 'y', 'z'}, optional
        Axis to project along. Default 'z' (drop z -> use x-y).
    node_scale : float, optional
        Factor that converts diam (µm) to matplotlib marker area points².
    dpi : int, optional
        The resolution of the figure in dots per inch.
    """
    # ---- 1. Collect coordinates and graph ----
    G = cell.graph
    if G is None:
        raise ValueError("Graph is None. Please create a graph first.")

    coords = {n: (float(d['x']), float(d['y']), float(d['z']))
              for n, d in G.nodes(data=True)}

    # Orthographic projection
    if view == 'z':
        x_l, y_l = 'x', 'y'
        proj = {n: (c[0], c[1]) for n, c in coords.items()}
    elif view == 'y':
        x_l, y_l = 'x', 'z'
        proj = {n: (c[0], c[2]) for n, c in coords.items()}
    elif view == 'x':
        x_l, y_l = 'y', 'z'
        proj = {n: (c[1], c[2]) for n, c in coords.items()}
    else:
        raise ValueError("view must be 'x', 'y', or 'z'")

    # ---- 2. Prepare data grouped by label for plotting ----
    labels = cell._labels
    color_choices = palette_hsv(len(labels))
    
    # Create a mapping from a label string to its color
    label_to_color = {label: color for label, color in zip(labels, color_choices)}
    
    # This dictionary will hold the plot data for each group
    # e.g., {'soma': {'xs': [...], 'ys': [...], 'sizes': [...]}, ...}
    data_by_label = {label: {'xs': [], 'ys': [], 'sizes': []} for label in labels}

    indices = torch.arange(len(G.nodes()), device=cell.device())
    
    # Create a reverse map from node_id to its label for quick lookup
    node_to_label = {}
    for label in labels:
        nodes_for_label = indices[cell.find(label)].cpu().tolist()
        for node_id in nodes_for_label:
            node_to_label[node_id] = label
            
    # Populate the data dictionary
    unclassified_nodes = {'xs': [], 'ys': [], 'sizes': []}
    for n in G.nodes():
        label = node_to_label.get(n)
        px, py = proj[n]
        size = G.nodes[n]['diam'] * node_scale
        
        if label:
            data_by_label[label]['xs'].append(px)
            data_by_label[label]['ys'].append(py)
            data_by_label[label]['sizes'].append(size)
        else:
            # Handle nodes that might not have a label
            unclassified_nodes['xs'].append(px)
            unclassified_nodes['ys'].append(py)
            unclassified_nodes['sizes'].append(size)

    # ---- 3. Plotting ----
    fig, ax = plt.subplots(dpi=dpi, figsize=(8, 8))

    # Plot edges first so they are in the background
    for u, v in G.edges():
        x0, y0 = proj[u]
        x1, y1 = proj[v]
        ax.plot([x0, x1], [y0, y1], 'k-', linewidth=0.5, alpha=0.7)
    
    # Plot each group of nodes with a separate scatter call to create legend handles
    for label, data in data_by_label.items():
        if not data['xs']: continue # Skip empty labels
        ax.scatter(
            data['xs'],
            data['ys'],
            s=data['sizes'],
            c=[label_to_color[label]], # Use a list with one color
            label=label, # This is the key for the legend!
            alpha=0.85,
            edgecolors='k',
            linewidths=0.3
        )
        
    # Plot any unclassified nodes
    if unclassified_nodes['xs']:
        ax.scatter(
            unclassified_nodes['xs'],
            unclassified_nodes['ys'],
            s=unclassified_nodes['sizes'],
            c='grey',
            label='unclassified',
            alpha=0.6,
            edgecolors='k',
            linewidths=0.3
        )
    
    # ---- 4. Create and display the legend ----
    ax.legend(
        loc='upper left',          # Specifies which corner of the legend box to anchor
        bbox_to_anchor=(1.02, 1),  # Places the anchor point outside the axes
        borderaxespad=0.,          # Removes padding between the anchor and the legend
        frameon=False
    )

    ax.set_aspect('equal')
    ax.set_xlabel(f'{x_l} (µm)')
    ax.set_ylabel(f'{y_l} (µm)')
    fig.tight_layout()
    plt.show()


def vis_3d_plotly(cell, node_scale: float = 5.0) -> None:
    import plotly.graph_objects as go
    """
    Plots a 3D NetworkX graph interactively using Plotly.
    Correctly aligns hover text with plotted nodes.

    Parameters
    ----------
    cell : Your AxonML Population-like object
        Must have .graph, .device, ._labels, and .find() attributes.
    node_scale : float, optional
        Factor that converts compartment diameter (µm) to Plotly's marker size.
    """
    G = cell.graph
    if not isinstance(G, nx.Graph) or G.number_of_nodes() == 0:
        print("Graph is not a valid or non-empty NetworkX graph. Nothing to plot.")
        return

    # ---- 1. Collect 3D coordinates for each node ----
    coords = {
        n: (
            float(d.get('x', 0)),
            float(d.get('y', 0)),
            float(d.get('z', 0))
        ) for n, d in G.nodes(data=True)
    }

    # ---- 2. Assign colors based on labels ----
    indices = torch.arange(len(G.nodes()), device=cell.device())
    labels = cell._labels
    color_choices = palette_hsv(len(labels))
    colors: Dict[int, str] = {}
    
    for i, label in enumerate(labels):
        # This part assumes cell.find() returns a slice or an integer tensor
        label_indices_obj = cell.find(label)
        if isinstance(label_indices_obj, slice):
            label_indices_tensor = indices[label_indices_obj]
        else:
            label_indices_tensor = label_indices_obj
            
        for idx in label_indices_tensor.cpu().tolist():
            colors[idx] = color_choices[i]

    # ---- 3. Prepare data for Plotly traces (the corrected part) ----
    
    # Create a fixed list of nodes to ensure all subsequent lists are in the same order.
    node_list = list(G.nodes())

    # --- For the Edges ---
    # This method is efficient for drawing all lines in one go.
    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        edge_x.extend([coords[u][0], coords[v][0], None])
        edge_y.extend([coords[u][1], coords[v][1], None])
        edge_z.extend([coords[u][2], coords[v][2], None])

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=1, color='black'),
        hoverinfo='none',
        mode='lines'
    )

    # --- For the Nodes and their Hover Text (Aligned) ---
    node_x = [coords[n][0] for n in node_list]
    node_y = [coords[n][1] for n in node_list]
    node_z = [coords[n][2] for n in node_list]
    
    node_colors = [colors.get(n, 'grey') for n in node_list]
    node_sizes = [G.nodes[n].get('diam', 1.0) * node_scale for n in node_list]
    
    hover_texts = []
    for node_id in node_list:
        attrs = G.nodes[node_id]
        name = attrs.get('name', 'N/A')
        diam = attrs.get('diam', 0.0)
        length = attrs.get('L', 0.0)
        area = attrs.get('area', 0.0)

        node_info = (
            f"<b>Node ID: {node_id}</b><br>"
            f"Name: {name}<br>"
            f"Diameter: {diam:.2f} µm<br>"
            f"Length: {length:.2f} µm<br>"
            f"Area: {area:.2f} µm²<br>"
        )
        hover_texts.append(node_info)

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        text=hover_texts,  # Assign the correctly ordered hover texts
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes,
            sizemin=4,
            line=dict(width=0.5, color='black')
        )
    )

    # ---- 4. Create the Figure and define Layout ----
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title_text='3D Neuron Morphology',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            scene=dict(
                xaxis_title='X (µm)',
                yaxis_title='Y (µm)',
                zaxis_title='Z (µm)',
                aspectmode='data'  # This ensures a 1:1:1 aspect ratio
            )
        )
    )
    
    fig.show()
