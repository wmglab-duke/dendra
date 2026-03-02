import matplotlib.pyplot as plt
import numpy as np


def render():
    plt.rcParams["figure.dpi"] = 140

    # Example 1: one RANGE variable controlled by a few region-wise sources
    n_comp = 18
    regionwise = np.full(n_comp, 35.4)
    regionwise[[0, 1, 2, 15, 16, 17]] = 60.0  # nodes / terminals
    regionwise[3:15] = 42.0  # internodes

    # Example 2: a fully compartment-specific RANGE tensor
    x = np.linspace(0, 1, n_comp)
    fully_specific = 35.4 + 10 * x + 4 * np.sin(2 * np.pi * x)

    fig, axes = plt.subplots(2, 1, figsize=(12, 3.8), constrained_layout=True)

    for ax, values, title in zip(
        axes,
        [regionwise, fully_specific],
        [
            "Region-wise sharing: a few learned sources govern groups of compartments",
            "Fully compartment-specific RANGE tensor: every compartment may differ",
        ],
    ):
        im = ax.imshow(values[None, :], aspect="auto", cmap="viridis")
        ax.set_yticks([])
        ax.set_xticks(np.arange(n_comp))
        ax.set_xticklabels(np.arange(n_comp))
        ax.set_xlabel("Compartment index")
        ax.set_title(title, fontsize=11)
        for j, v in enumerate(values):
            ax.text(
                j, 0, f"{v:.1f}", ha="center", va="center", color="white", fontsize=8
            )

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="Example RANGE value")
    plt.show()


def render2():
    # Visualize which source writes which compartment region
    source_codes = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2])
    source_values = np.array([70.0, 40.0, 55.0])
    assembled = np.where(
        source_codes == 0,
        source_values[0],
        np.where(source_codes == 1, source_values[1], source_values[2]),
    )

    fig, ax = plt.subplots(figsize=(12, 2.3), constrained_layout=True)
    ax.imshow(source_codes[None, :], aspect="auto", cmap="tab10")
    ax.set_yticks([])
    ax.set_xticks(np.arange(len(source_codes)))
    ax.set_xticklabels(np.arange(len(source_codes)))
    ax.set_xlabel("Compartment index")
    ax.set_title(
        "Illustration: different underlying sources can govern different regions of one RANGE tensor",
        fontsize=11,
    )
    for j, v in enumerate(assembled):
        ax.text(j, 0, f"{v:.0f}", ha="center", va="center", color="white", fontsize=8)
    plt.show()
