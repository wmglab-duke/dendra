import colorsys
import random

import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import networkx as nx
import numpy as np


def generate_colors(n):
    return cm.get_cmap("tab10").colors[:n]


def palette_hsv(n, *, s=0.65, v=0.9, seed=0):
    """Return n colours as hex strings by uniform hue spacing in HSV space."""
    random.seed(seed)           # repeatable order if desired
    hues = [i / n for i in range(n)]
    random.shuffle(hues)        # break up any obvious gradients
    return [mpl.colors.to_hex(colorsys.hsv_to_rgb(h, s, v)) for h in hues]


def vis_2d(cell, idx=0, view='y', node_scale=8, dpi=200):
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

    x = cell.x[idx]
    y = cell.y[idx]
    z = cell.z[idx]

    coords = {n: (float(x[n]), float(y[n]), float(z[n]))
              for n in G.nodes}

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
            label=label,
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
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.,
        frameon=False
    )

    ax.set_aspect('equal')
    ax.set_xlabel(f'{x_l} (µm)')
    ax.set_ylabel(f'{y_l} (µm)')
    fig.tight_layout()
    plt.show()


def vis_3d_plotly(cell, idx=0, node_scale: float = 10.0, height=800.0, width=None) -> None:
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

    x = cell.x[idx]
    y = cell.y[idx]
    z = cell.z[idx]

    # ---- 1. Collect 3D coordinates for each node ----
    coords = {
        n: (
            float(x[n]),
            float(y[n]),
            float(z[n])
        ) for n in G.nodes
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
    edge_text = []
    for u, v, data in G.edges(data=True):
        edge_x.extend([coords[u][0], coords[v][0], None])
        edge_y.extend([coords[u][1], coords[v][1], None])
        edge_z.extend([coords[u][2], coords[v][2], None])
        # Prepare hover text for edges
        r_ohm = data.get('R_ohm', None)
        if r_ohm is not None:
            r_ohm = float(r_ohm) * 1e-6  # Convert to MOhm
        else:
            r_ohm = 'N/A'
        edge_info = (
            f"R (MOhm): {r_ohm:.3f}<br>"
        )
        edge_text.append(edge_info)



    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=1.5, color='black'),
        hoverinfo='text',
        text=edge_text,
        mode='lines'
    )

    # --- For the Nodes and their Hover Text (Aligned) ---
    node_x = [coords[n][0] for n in node_list]
    node_y = [coords[n][1] for n in node_list]
    node_z = [coords[n][2] for n in node_list]
    
    node_colors = [colors.get(n, 'grey') for n in node_list]
    node_sizes = [G.nodes[n].get('diam', 1.0) * node_scale for n in node_list]
    node_sizes = np.log10(node_sizes)
    node_sizes = np.clip(node_sizes, 0.1, None)  # Ensure minimum size for visibility
    
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
            height=height,
            width=width,
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


def vis_voltage_3d_plotly(x, y, z, voltage, height=800, width=None):
    import plotly.graph_objects as go

    # 1. Create the 3D scatter plot object
    #    The configuration is done inside the 'go.Scatter3d' call.
    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',  # We want to plot points, not lines
        marker=dict(
            size=5,             # Size of the markers
            color=voltage,      # Set color to our voltage data
            colorscale='Viridis', # One of Plotly's built-in colorscales
            showscale=True,     # We want to show the color bar
            colorbar=dict(
                title='Voltage (mV)' # Title for the color bar
            )
        )
    )

    # 2. Create a layout object to configure the plot's appearance
    layout = go.Layout(
        height=height,  # Set the height of the plot
        width=width,    # Set the width of the plot
        scene=dict(
            xaxis=dict(title='x (μm)'),
            yaxis=dict(title='y (μm)'),
            zaxis=dict(title='z (μm)')
        ),
        margin=dict(l=0, r=0, b=0, t=40) # Adjust margins
    )

    # 3. Create a figure and add the trace and layout
    fig = go.Figure(data=[trace], layout=layout)

    # 4. Show the figure
    #    This will open an interactive plot in your web browser or in your
    #    Jupyter Notebook / VS Code output cell.
    fig.show()


def vis_voltage_2d(x, y, z, voltage, view='y', node_scale=8, dpi=200):
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
    """
    fig, ax = plt.subplots(dpi=dpi, figsize=(8, 8))

    # We will capture the output of ax.scatter into a variable, let's call it `sc`.
    if view == 'z':
        sc = ax.scatter(x, y, c=voltage, s=node_scale * 10, cmap='viridis', alpha=0.7)
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')
    elif view == 'y':
        sc = ax.scatter(x, z, c=voltage, s=node_scale * 10, cmap='viridis', alpha=0.7)
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('z (µm)')
    elif view == 'x':
        sc = ax.scatter(y, z, c=voltage, s=node_scale * 10, cmap='viridis', alpha=0.7)
        ax.set_xlabel('y (µm)')
        ax.set_ylabel('z (µm)')
    else:
        raise ValueError("view must be 'x', 'y', or 'z'")

    fig.colorbar(sc, label='Voltage (mV)', ax=ax)
    
    fig.tight_layout()
    plt.show()