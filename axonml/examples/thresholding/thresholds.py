import argparse

import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt

from axonml.instruments.thresholder import Thresholder
from axonml.models import MRG

matplotlib.use("TKAgg")

parser = argparse.ArgumentParser()

parser.add_argument("-f", "--field", choices=["imthera", "livanova"], help="Cuff.")

parser.add_argument(
    "-p", "--preload", action="store_true", help="Preload bases array into memory."
)

parser.add_argument(
    "-v",
    "--visualize",
    action="store_true",
    help="Plot predicted thresholds & error histogram.",
)

split_txt = """
    Number of chunks into which to split the bases array when
    calculating thresholds.
"""

parser.add_argument("-s", "--splits", type=int, default=None, help=split_txt)

args = parser.parse_args()


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.cuda.empty_cache()

    # load field and diameter data

    field = args.field

    directory = f"./example_{field}"

    diams = np.load(f"{directory}/example_diameters_{field}.npy")
    n = len(diams)
    fp = np.memmap(
        f"{directory}/example_field_array_{field}.mmap",
        dtype="float32",
        mode="r",
        shape=(n, 1000, 101),
    )
    if args.preload:
        fp = np.array(fp)

    nrn_thresh_path = f"{directory}/example_thresholds_{field}.npy"
    thresh_nrn = np.load(nrn_thresh_path).flatten()

    mrg = MRG(handle_nan=True).cuda().load("MRG2023")
    thresholder = Thresholder(mrg, fp, diams)
    thresh, _ = thresholder.calculate_thresholds(verbose=False, splits=args.splits)

    err = 100 * (thresh - thresh_nrn) / thresh_nrn

    print(f"mean % threshold error: {err.mean()}%")
    print(f"mean abosulte % threshold error: {np.abs(err).mean()}%")
    print(f"min error: {err.min()}%, max error: {err.max()}%")

    # visualize ?
    if args.visualize:
        lim = np.linspace(0, max(thresh), 100)
        plt.plot(lim, lim, color="grey", alpha=0.6)
        plt.scatter(thresh, thresh_nrn, s=2)
        plt.xlabel("predicted threshold (mA)")
        plt.ylabel("NEURON threshold (mA)")
        plt.axis("square")
        plt.show()

        # error distrubtion
        plt.hist(err)
        plt.xlabel("% threshold error")
        plt.show()
