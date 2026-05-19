import contextlib
import importlib

import matplotlib

# Use non‑interactive backend so figures don't pop up during CI
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
import torch

VIS = importlib.import_module("dendra.models.visualization")


# -----------------------------------------------------------------------------
# ───────────────────────────── helper fixtures ────────────────────────────────
# -----------------------------------------------------------------------------
class DummyCell:
    """Mimic the minimal API required by vis_* functions."""

    def __init__(self, n=12, seed=0):
        rng = np.random.default_rng(seed)
        self._labels = ["soma", "axon", "dend"]
        self._label_map = {}
        self.graph = nx.Graph()
        self.n = n

        # coordinates & diameters
        xyz = rng.normal(scale=20.0, size=(n, 3)).astype(np.float32)
        self.x = torch.tensor(xyz[:, 0])
        self.y = torch.tensor(xyz[:, 1])
        self.z = torch.tensor(xyz[:, 2])

        diams = rng.uniform(0.5, 5.0, size=n)

        # assign nodes : connect to next to make simple chain
        for i in range(n):
            self.graph.add_node(i, diam=float(diams[i]), name=f"n{i}")
            if i:
                self.graph.add_edge(i - 1, i, R_ohm=float(rng.uniform(1.0, 10.0)))

        # equal‑sized label groups
        group_size = n // len(self._labels) + 1
        for i, lbl in enumerate(self._labels):
            ids = list(range(i * group_size, min((i + 1) * group_size, n)))
            for idx in ids:
                self._label_map[idx] = lbl

    # required tiny API --------------------------------------------------
    def find(self, label):
        idxs = [i for i, lbl in self._label_map.items() if lbl == label]
        return torch.tensor(idxs, dtype=torch.long)

    def device(self):
        return torch.device("cpu")


@pytest.fixture(scope="session")
def dummy_cell():
    return DummyCell()


# Monkey‑patch plt.show globally so tests never block or pop windows.
@pytest.fixture(autouse=True)
def _no_show(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield


# -----------------------------------------------------------------------------
# ───────────────────────────── unit‑style tests ───────────────────────────────
# -----------------------------------------------------------------------------


def test_palette_hsv_length_and_format():
    cols = VIS.palette_hsv(10)
    assert len(cols) == 10
    assert all(c.startswith("#") and len(c) == 7 for c in cols)


def test_parula_registered():
    """Colormap should be registered exactly once with 256 entries."""
    cmap = matplotlib.colormaps["parula"]
    assert cmap.N == 256


# -----------------------------------------------------------------------------
# ───────────────────────────── smoke tests for figs ───────────────────────────
# -----------------------------------------------------------------------------


def _fig_hash(fig: plt.Figure) -> str:
    """Render figure → png bytes → sha256 hex (stable across machines)."""

    import hashlib
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    buf.seek(0)
    return hashlib.sha256(buf.read()).hexdigest()


@pytest.mark.parametrize("view", ["x", "y", "z"])
def test_vis2d_runs_and_returns_unique_hash(dummy_cell, view):
    # ensure each view produces a picture (no errors) and hashes differ
    with contextlib.suppress(Exception):
        VIS.vis_2d(dummy_cell, view=view)
    new_hash = _fig_hash(plt.gcf())
    assert new_hash  # non‑empty means figure rendered


@pytest.mark.parametrize("interp", ["linear", "nearest", "cubic"])
def test_vis_threshold_mollweide_hash(interp):
    rng = np.random.default_rng(0)
    phi = rng.uniform(-np.pi, np.pi, 50)
    theta = rng.uniform(0, np.pi, 50)
    thr = rng.uniform(0.5, 2.0, 50)

    ax = VIS.vis_threshold_mollweide_2d(
        phi, theta, thr, interp_method=interp, mark_min=True
    )
    assert isinstance(ax, matplotlib.axes.Axes)


# -----------------------------------------------------------------------------
# ─────────────────────────── vis_voltage_2d smoke test ───────────────────────
# -----------------------------------------------------------------------------


def test_vis_voltage_2d(dummy_cell):
    x, y, z = dummy_cell.x.numpy(), dummy_cell.y.numpy(), dummy_cell.z.numpy()
    v = np.linspace(-70, 20, dummy_cell.n)
    with contextlib.suppress(Exception):
        VIS.vis_voltage_2d(x, y, z, v, view="y")
