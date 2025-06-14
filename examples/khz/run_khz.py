import argparse

import torch
from tqdm import tqdm
import numpy as np

import axonml as ax
from axonml.units import nA, Hz, ms


torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser()
parser.add_argument("--chunks", type=int, default=200)

args = parser.parse_args()


def deltax(diam):
    return -8.215284e00 * diam**2 + 2.724201e02 * diam + -7.802411e02


s_idx = 0    # deliver stimulation from contact 0
nodes = 101  # nodes of Ranvier per fiber


# -- load fields --
FIELD_DATA = [np.load(f"./fields/{i}.npy") for i in range(6)]
field_x = np.load("./fields/fiber_zs.npy")


# -- generate field bases --
# machinery
interpolator = ax.precomputed_interpolate_1d(FIELD_DATA[s_idx] * 1000, field_x)


def make_ve_at_nodes(diameter, a_idx, nodes=nodes, offset=37500):
    start = (deltax(diameter) * (nodes - 1)) / 2
    interp_at = np.linspace(-start, start, nodes) + offset
    b = interpolator._interpolate(interpolator.get(a_idx), interpolator.x[0], interp_at)
    return b


diam_amp_dict = {
    5.7: np.arange(0, 10, 0.01),
    8.7: np.arange(0, 5, 0.01),
    14.0: np.arange(0, 2, 0.01),
}


a_indices = [0, 10, 15, 25, 32, 33, 45]


field_stack = []
diams = []
splits = []

# -- construct bases --
for diam, amps in diam_amp_dict.items():
    split_length = 0
    for a_idx in a_indices:
        b = make_ve_at_nodes(diam, a_idx)
        stack = np.einsum("i,j->ij", amps, b)
        field_stack.append(stack)
        diams.append([diam] * len(stack))
        split_length += len(stack)
    splits.append(split_length)

splits = np.cumsum(splits)


field_stack = np.vstack(field_stack)
diam = np.concatenate(diams)


# code to run and count APs

count = ax.callbacks.APCount(node_check=[-5], threshold=-20.0, t_start_check=50)


# run

frequencies = [1, 2, 5, 10]

input_diams = []
for _ in frequencies:
    input_diams.append(torch.tensor(diam, device="cuda").float())
input_diams = torch.cat(input_diams)

stim = ax.sin(amp=1.0, freq=np.repeat(frequencies, len(field_stack))[:, None], delay=0.5)
field_stack = torch.tensor(field_stack).repeat(len(frequencies), 1)

tstop = 100
dt = 0.001

# fiber model
mrg = ax.SMF(input_diams, nodes).cuda().load("MRG")

# intracellular stim to generate activity
intra = ax.mono_rect(amp=2*nA, pw=0.1*ms).repeat(100*Hz, delay=50*ms)
mrg[:, 5].inject(intra)

count.reset()
mrg.longrun(
    tstop=tstop, 
    dt=dt, 
    extra=(field_stack, stim), 
    chunklength=int(tstop/dt/args.chunks), 
    callbacks=[count]
)
all_n = count.numpy()


# visualize

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use("TKAgg")

data = {
    "frequency": [],
    "a_idx": [],
    "amplitude": [],
    "diameter": [],
    "# APs": [],
    "method": [],
}

all_n = np.split(all_n, len(frequencies))
all_diams = [5.7, 8.7, 14.0]

for frequency, n in zip(frequencies, all_n):
    n_by_diam = np.split(n, splits[:2])
    for d, n_d in zip(all_diams, n_by_diam):
        n_by_a_idx = np.split(n_d, len(a_indices))
        for a_idx, n_a in zip(a_indices, n_by_a_idx):
            amps = diam_amp_dict[d]
            n_nrn = np.load(
                f"./ground_truth/{frequency}khz_{d}um_2_3_0_{a_idx}.npy"
            ).flatten()
            for amp, n_ in zip(amps, n_a):
                data["frequency"].append(frequency)
                data["a_idx"].append(a_idx)
                data["amplitude"].append(amp)
                data["diameter"].append(d)
                data["# APs"].append(n_[0])
                data["method"].append("Surrogate")
            for amp, n_ in zip(amps, n_nrn):
                data["frequency"].append(frequency)
                data["a_idx"].append(a_idx)
                data["amplitude"].append(amp)
                data["diameter"].append(d)
                data["# APs"].append(n_)
                data["method"].append("NEURON")

data_df = pd.DataFrame(data)

cols = [r"5.7 $\mu m$", r"8.7 $\mu m$", r"14.0 $\mu m$"]
rows = ["1 kHz", "2 kHz", "5 kHz", "10 kHz"]

fig, axes = plt.subplots(4, 3, dpi=200, figsize=(5, 5), sharex="col", sharey=True)

for i, d_selec in enumerate(all_diams):
    for j, f_selec in enumerate(frequencies):
        data_selection = data_df[data_df["diameter"] == d_selec]
        data_selection = data_selection[data_selection["frequency"] == f_selec]
        sns.lineplot(
            data=data_selection,
            x="amplitude",
            y="# APs",
            hue="method",
            hue_order=["NEURON", "Surrogate"],
            legend=False,
            palette="colorblind",
            ax=axes[j, i],
            errorbar="sd",
        )
        axes[j, i].set_ylabel("")
        axes[j, i].set_xlabel("")

fig.add_subplot(111, frameon=False)

# hide tick and tick label of the big axis
plt.tick_params(
    labelcolor="none", which="both", top=False, bottom=False, left=False, right=False
)
plt.xlabel("Amplitude $(mA)$")
plt.ylabel("# APs", labelpad=5)

for ax, col in zip(axes[0], cols):
    ax.set_title(col, size="medium")

for ax, row in zip(axes[:, 0], rows):
    ax.set_ylabel(row, rotation=45, size="medium", labelpad=35)

from matplotlib.lines import Line2D

cmap = sns.color_palette("colorblind")
custom_lines = [
    Line2D([0], [0], color=cmap[0], lw=2),
    Line2D([0], [0], color=cmap[1], lw=2),
]
plt.gca().legend(
    custom_lines,
    ["NEURON", "S-MF"],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncols=2,
)

# plt.show()

fig.savefig("khz_stim_example.png", bbox_inches="tight")
