import importlib

import matplotlib

# Use a non-interactive backend so figures don't pop up during CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
import torch

VIS = importlib.import_module("dendra.models.visualization")
ANALYSIS = importlib.import_module("dendra.models.analysis")


class DummyCell:
    """Mimic the Population API required by the visualization helpers."""

    def __init__(self, n=12, seed=0):
        rng = np.random.default_rng(seed)
        self._labels = ["soma", "axon", "dend"]
        self._label_map = {}
        self.graph = nx.DiGraph()
        self.n = n

        xyz = rng.normal(scale=20.0, size=(n, 3)).astype(np.float32)
        # Population coordinates have shape (population, compartments).
        self.x = torch.tensor(xyz[:, 0]).unsqueeze(0)
        self.y = torch.tensor(xyz[:, 1]).unsqueeze(0)
        self.z = torch.tensor(xyz[:, 2]).unsqueeze(0)

        diams = rng.uniform(0.5, 5.0, size=n)
        for i in range(n):
            self.graph.add_node(
                i,
                diam=float(diams[i]),
                name=f"n{i}",
                L=2.0 + i,
                area=3.0 + i,
            )
            if i:
                self.graph.add_edge(i - 1, i, R_ohm=float(rng.uniform(1.0, 10.0)))

        # Leave one node unclassified to exercise the fallback visualization.
        for idx in range(max(0, n - 1)):
            self._label_map[idx] = self._labels[(3 * idx) // max(1, n - 1)]

    def find(self, label):
        idxs = [i for i, value in self._label_map.items() if value == label]
        return torch.tensor(idxs, dtype=torch.long)

    def device(self):
        return torch.device("cpu")


@pytest.fixture
def dummy_cell():
    return DummyCell()


@pytest.fixture(autouse=True)
def _no_show(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    yield
    plt.close("all")


def test_palette_hsv_length_and_format():
    colors = VIS.palette_hsv(10)
    assert len(colors) == 10
    assert all(color.startswith("#") and len(color) == 7 for color in colors)
    assert VIS.palette_hsv(10) == colors
    assert VIS.palette_hsv(0) == []


def test_parula_registered():
    assert matplotlib.colormaps["parula"].N == 256
    assert matplotlib.colormaps["parula_r"].N == 256


@pytest.mark.parametrize("view", ["x", "y", "z"])
def test_vis2d_returns_projected_morphology(dummy_cell, view):
    fig, ax = VIS.vis_2d(dummy_cell, view=view)

    expected_labels = {
        "x": ("y (µm)", "z (µm)"),
        "y": ("x (µm)", "z (µm)"),
        "z": ("x (µm)", "y (µm)"),
    }
    assert isinstance(fig, plt.Figure)
    assert ax.figure is fig
    assert (ax.get_xlabel(), ax.get_ylabel()) == expected_labels[view]
    assert len(ax.collections) == 5  # edges and four node groups
    assert {text.get_text() for text in ax.get_legend().get_texts()} == {
        "soma",
        "axon",
        "dend",
        "unclassified",
    }
    assert ax.get_aspect() == 1.0


@pytest.mark.parametrize("loc", ["lower left", "lower right"])
def test_vis2d_scalebar_and_existing_axes(dummy_cell, loc):
    fig, ax = plt.subplots()
    returned_fig, returned_ax = VIS.vis_2d(
        dummy_cell,
        fig=fig,
        ax=ax,
        show_nodes=False,
        show_legend=False,
        scalebar=True,
        scalebar_length=10,
        scalebar_loc=loc,
    )

    assert (returned_fig, returned_ax) == (fig, ax)
    assert len(ax.collections) == 1
    assert len(ax.lines) == 1
    assert [text.get_text() for text in ax.texts] == ["10 µm"]
    assert not any(spine.get_visible() for spine in ax.spines.values())
    assert not ax.get_xticks().size
    assert not ax.get_yticks().size


def test_vis2d_rejects_invalid_inputs(dummy_cell):
    with pytest.raises(ValueError, match="view must be"):
        VIS.vis_2d(dummy_cell, view="diagonal")

    with pytest.raises(ValueError, match="scalebar_loc"):
        VIS.vis_2d(dummy_cell, scalebar=True, scalebar_loc="top")

    dummy_cell.graph = None
    with pytest.raises(ValueError, match="Graph is None"):
        VIS.vis_2d(dummy_cell)


@pytest.mark.parametrize("size_mode", ["sqrt", "linear", "log"])
def test_vis3d_returns_labeled_traces_and_scalebar(dummy_cell, size_mode):
    pytest.importorskip("plotly")
    fig = VIS.vis_3d(
        dummy_cell,
        size_mode=size_mode,
        scalebar=True,
        scalebar_length=10,
        scalebar_axis="x",
        height=401.5,
        width=602.1,
        show=False,
    )

    trace_names = [trace.name for trace in fig.data]
    assert {"soma", "axon", "dend", "unclassified", "scalebar"} <= set(trace_names)
    assert fig.layout.height == 402
    assert fig.layout.width == 602
    assert fig.layout.scene.xaxis.visible is False
    assert fig.layout.scene.annotations[0].text == "10 µm"
    soma_trace = next(trace for trace in fig.data if trace.name == "soma")
    assert all("Node ID" in text for text in soma_trace.text)


def test_vis3d_visibility_and_validation(dummy_cell):
    pytest.importorskip("plotly")
    fig = VIS.vis_3d(
        dummy_cell,
        show_nodes=False,
        show_edges=False,
        show_legend=False,
        background="linen",
        show=False,
    )
    assert len(fig.data) == 0
    assert fig.layout.showlegend is False
    assert fig.layout.scene.xaxis.visible is True

    with pytest.raises(ValueError, match="size_mode"):
        VIS.vis_3d(dummy_cell, size_mode="area", show=False)

    dummy_cell.graph.clear()
    with pytest.raises(ValueError, match="valid or non-empty"):
        VIS.vis_3d(dummy_cell, show=False)


@pytest.mark.parametrize("interp", ["linear", "nearest", "cubic"])
def test_vis_threshold_mollweide_returns_interpolation_and_minimum(interp):
    rng = np.random.default_rng(0)
    phi = rng.uniform(-np.pi, np.pi, 50)
    theta = rng.uniform(0, np.pi, 50)
    threshold = rng.uniform(0.5, 2.0, 50)

    ax = VIS.vis_threshold_mollweide_2d(
        phi,
        theta,
        threshold,
        interp_method=interp,
        mark_min=True,
        grid_res_deg=10,
    )

    assert isinstance(ax, matplotlib.axes.Axes)
    assert ax.name == "mollweide"
    assert len(ax.collections) == 2  # interpolated mesh and minimum marker
    assert ax.figure.axes[-1].get_xlabel() == "Threshold |E| (V/m)"


def test_vis_threshold_mollweide_degree_normalization_and_custom_axis():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="mollweide")
    phi = np.array([-150, -90, -30, 30, 90, 150, -120, 0, 120])
    theta = np.array([20, 50, 80, 110, 140, 160, 130, 40, 100])
    threshold = np.arange(1, 10, dtype=float)

    returned = VIS.vis_threshold_mollweide_2d(
        phi,
        theta,
        threshold,
        angles_in_degrees=True,
        flip_polar=False,
        azimuth_offset=30,
        offset_in_degrees=True,
        normalize_to_min=True,
        interp_method="nearest",
        grid_res_deg=15,
        ax=ax,
    )

    assert returned is ax
    assert ax.figure.axes[-1].get_xlabel() == "Threshold |E| / min"
    assert any(label.get_text() == "0°" for label in ax.get_xticklabels())


def test_vis_threshold_3d_surface_and_minimum_marker():
    pytest.importorskip("plotly")
    rng = np.random.default_rng(4)
    phi = rng.uniform(-180, 180, 40)
    theta = rng.uniform(5, 175, 40)
    threshold = rng.uniform(1, 3, 40)

    fig = VIS.vis_threshold_3d(
        phi,
        theta,
        threshold,
        angles_in_degrees=True,
        flip_polar=False,
        azimuth_offset=20,
        offset_in_degrees=True,
        normalize_to_min=True,
        mark_min=True,
        interp_method="nearest",
        grid_res_deg=15,
        cmap="plasma",
    )

    assert [trace.type for trace in fig.data] == ["surface", "mesh3d", "scatter3d"]
    assert fig.data[0].colorbar.title.text == "Threshold |E| / min"
    assert fig.data[0].colorscale[0][1] == matplotlib.colors.to_hex(
        matplotlib.colormaps["plasma"](0.0)
    )
    assert fig.layout.scene.aspectmode == "data"

    axis = plt.figure().add_subplot(111)
    with pytest.warns(UserWarning, match="ignored"):
        VIS.vis_threshold_3d(
            phi,
            theta,
            threshold,
            ax=axis,
            interp_method="nearest",
            grid_res_deg=30,
        )


def test_vis_voltage_3d_builds_colored_points(monkeypatch):
    pytest.importorskip("plotly")
    captured = {}

    def capture_show(fig):
        captured["fig"] = fig

    monkeypatch.setattr(VIS.go.Figure, "show", capture_show)
    result = VIS.vis_voltage_3d(
        [0.0, 1.0],
        [2.0, 3.0],
        [4.0, 5.0],
        [-70.0, 20.0],
        height=450,
        width=600,
    )

    assert result is None
    fig = captured["fig"]
    assert len(fig.data) == 1
    assert fig.data[0].marker.color == (-70.0, 20.0)
    assert fig.data[0].marker.colorbar.title.text == "Voltage (mV)"
    assert fig.layout.height == 450
    assert fig.layout.width == 600


@pytest.mark.parametrize(
    ("view", "expected_labels"),
    [
        ("x", ("y (µm)", "z (µm)")),
        ("y", ("x (µm)", "z (µm)")),
        ("z", ("x (µm)", "y (µm)")),
    ],
)
def test_vis_voltage_2d(dummy_cell, view, expected_labels):
    x = dummy_cell.x[0].numpy()
    y = dummy_cell.y[0].numpy()
    z = dummy_cell.z[0].numpy()
    voltage = np.linspace(-70, 20, dummy_cell.n)
    fig, ax = VIS.vis_voltage_2d(x, y, z, voltage, view=view)

    assert isinstance(fig, plt.Figure)
    assert (ax.get_xlabel(), ax.get_ylabel()) == expected_labels
    assert len(ax.collections) == 1
    assert ax.collections[0].get_array().tolist() == pytest.approx(voltage)
    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylabel() == "Voltage (mV)"


def test_vis_voltage_2d_custom_figure_without_colorbar(dummy_cell):
    fig = plt.figure()
    result, ax = VIS.vis_voltage_2d(
        dummy_cell.x[0],
        dummy_cell.y[0],
        dummy_cell.z[0],
        torch.linspace(-80, 30, dummy_cell.n),
        fig=fig,
        cbar=False,
        cmap="plasma",
        norm=matplotlib.colors.Normalize(-80, 30),
    )
    assert result is fig
    assert len(fig.axes) == 1
    assert ax.collections[0].cmap.name == "plasma"

    with pytest.raises(ValueError, match="view must be"):
        VIS.vis_voltage_2d([0], [0], [0], [-70], view="diagonal")


def test_vis_morphology_by_layer_handles_unbranched_tree(dummy_cell):
    fig, ax = VIS.vis_morphology_by_layer(dummy_cell, threads=2)

    assert isinstance(fig, plt.Figure)
    assert ax.figure is fig
    assert fig.get_size_inches().tolist() == pytest.approx([8.0, 8.0])
    assert len(ax.lines) == dummy_cell.n - 1
    assert len(ax.collections) == 1
    assert ax.collections[0].get_array().tolist() == list(reversed(range(dummy_cell.n)))
    assert fig.axes[-1].get_ylabel() == "DHS layer (earlier → later)"


def test_vis_morphology_by_layer_branched_tree_and_custom_axis():
    cell = DummyCell(n=7)
    cell.graph = nx.DiGraph([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    fig, supplied_ax = plt.subplots()

    returned = VIS.vis_morphology_by_layer(cell, threads=2, ax=supplied_ax)

    assert returned is supplied_ax
    assert len(supplied_ax.lines) == 6
    assert len(supplied_ax.collections) == 1
    assert len(fig.axes) == 2

    cell.graph.add_edge(2, 4)
    with pytest.raises(ValueError, match="2 parents"):
        VIS.vis_morphology_by_layer(cell)


@pytest.fixture
def chronaxie_grid():
    return {
        "p_spike_trial": torch.tensor(
            [0.1, 0.3, 0.2, 0.4, 0.4, 0.8, 0.55, 0.85, 0.8, 1.0, 0.9, 1.0]
        ),
        "pw_group_id": torch.tensor([0, 0, 1, 1] * 3),
        "pw_unique_ms": torch.tensor([0.1, 0.2]),
        "I_th": torch.tensor([1.5, 1.6]),
        "rheobase": torch.tensor(1.0),
        "chronaxie_ms": torch.tensor(0.1),
        "bracket": {
            "I_low": torch.tensor([1.0, 1.1]),
            "I_high": torch.tensor([2.0, 2.1]),
        },
    }


def test_activation_heatmap_binary_grid_and_all_overlays(chronaxie_grid):
    amplitudes = torch.tensor([1.0] * 4 + [2.0] * 4 + [3.0] * 4)
    fig, ax, mesh, grid, amp_unique, pw_unique = (
        ANALYSIS.plot_activation_heatmap_from_chronaxie_output(
            chronaxie_grid,
            amplitudes,
            mode="binary",
            agg="mean",
            threshold_kind="both",
            overlay_bracket_band=True,
            overlay_empirical_threshold=True,
            empirical_kind="all",
            title="Oracle grid",
        )
    )

    torch.testing.assert_close(grid, torch.tensor([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]))
    torch.testing.assert_close(
        amp_unique, torch.tensor([1.0, 2.0, 3.0], dtype=amp_unique.dtype)
    )
    torch.testing.assert_close(pw_unique, torch.tensor([0.1, 0.2]))
    assert mesh in ax.collections
    assert ax.get_title() == "Oracle grid"
    assert len(fig.axes) == 2
    assert {
        "I_th",
        "fit",
        "empirical lowest active",
        "empirical highest inactive",
        "empirical midpoint",
    } <= set(ax.get_legend_handles_labels()[1])
    assert len(ax.lines) == 9


@pytest.mark.parametrize(
    ("agg", "expected"),
    [
        ("max", [[0.3, 0.4], [0.8, 0.85], [1.0, 1.0]]),
        ("min", [[0.1, 0.2], [0.4, 0.55], [0.8, 0.9]]),
    ],
)
def test_activation_heatmap_probability_aggregation(chronaxie_grid, agg, expected):
    amplitudes = np.array([1.001] * 4 + [2.001] * 4 + [3.001] * 4)
    fig, supplied_ax = plt.subplots()
    result_fig, ax, _, grid, amp_unique, _ = (
        ANALYSIS.plot_activation_heatmap_from_chronaxie_output(
            chronaxie_grid,
            amplitudes,
            mode="prob",
            agg=agg,
            amp_round_decimals=1,
            ax=supplied_ax,
            show_colorbar=False,
            overlay_threshold=False,
            overlay_empirical_threshold=True,
            empirical_use_thresholded_prob_in_prob_mode=False,
            max_xticks=1,
            max_yticks=2,
        )
    )

    assert result_fig is fig
    assert ax is supplied_ax
    torch.testing.assert_close(grid, torch.tensor(expected))
    torch.testing.assert_close(
        amp_unique, torch.tensor([1.0, 2.0, 3.0], dtype=amp_unique.dtype)
    )
    assert len(fig.axes) == 1
    assert len(ax.get_xticks()) == 1
    assert len(ax.get_yticks()) == 2
    assert ax.get_legend() is None


def test_activation_heatmap_single_bin_and_point_only_threshold():
    output = {
        "p_spike_trial": torch.tensor([0.75]),
        "pw_group_id": torch.tensor([0]),
        "pw_unique_ms": torch.tensor([0.5]),
        "I_th": torch.tensor([2.0]),
    }
    _, ax, _, grid, _, _ = ANALYSIS.plot_activation_heatmap_from_chronaxie_output(
        output,
        [2.0],
        overlay_line=False,
        overlay_points=True,
        show_colorbar=False,
        overlay_empirical_threshold=True,
        empirical_kind="lowest_active",
    )
    torch.testing.assert_close(grid, torch.ones((1, 1)))
    assert {"I_th", "empirical lowest active"} == set(ax.get_legend_handles_labels()[1])


@pytest.mark.parametrize(
    ("mutate", "exception", "message"),
    [
        (lambda out: out.pop("p_spike_trial"), KeyError, "p_spike_trial"),
        (
            lambda out: out.__setitem__("p_spike_trial", [0.1] * 12),
            TypeError,
            "must be a torch.Tensor",
        ),
        (
            lambda out: out.__setitem__("pw_group_id", torch.zeros(11)),
            ValueError,
            "Mismatched P",
        ),
        (
            lambda out: out.__setitem__("pw_unique_ms", torch.tensor([])),
            ValueError,
            "is empty",
        ),
        (
            lambda out: out.__setitem__("pw_group_id", torch.full((12,), 2)),
            ValueError,
            "outside",
        ),
    ],
)
def test_activation_heatmap_validates_aligned_inputs(
    chronaxie_grid, mutate, exception, message
):
    mutate(chronaxie_grid)
    with pytest.raises(exception, match=message):
        ANALYSIS.plot_activation_heatmap_from_chronaxie_output(
            chronaxie_grid, [1.0] * 12
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"agg": "median"}, "agg must be"),
        ({"mode": "counts"}, "mode must be"),
        ({"threshold_kind": "I_th"}, "has no key 'I_th'"),
        ({"threshold_kind": "fit"}, "does not contain both"),
        ({"overlay_bracket_band": True}, "must be a dict"),
    ],
)
def test_activation_heatmap_validates_configuration(kwargs, message):
    output = {
        "p_spike_trial": torch.tensor([0.2, 0.8]),
        "pw_group_id": torch.tensor([0, 0]),
        "pw_unique_ms": torch.tensor([0.1]),
    }
    if kwargs.get("overlay_bracket_band"):
        output["I_th"] = torch.tensor([0.5])
        output["bracket"] = {"I_low": torch.tensor([0.2])}

    with pytest.raises((ValueError, KeyError), match=message):
        ANALYSIS.plot_activation_heatmap_from_chronaxie_output(
            output,
            [1.0, 2.0],
            show_colorbar=False,
            **kwargs,
        )
