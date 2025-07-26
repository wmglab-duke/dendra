import torch
import axonml as ax

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cache_every", type=int, default=10000)
parser.add_argument("--chunklength", type=int, default=100)
args = parser.parse_args()


if __name__ == "__main__":
    n_ax = 10000  # number of fibers
    L = 50  # fiber length [mm]
    dx = 25.0  # fiber compartment length [um]

    diameters = torch.linspace(0.5, 2.0, n_ax)
    model = ax.Tigerholm2014(diameters, L, dx=dx, method="euler").cuda()

    # -- space --
    v_s = ax.isotropic_point(z=100.0, rhoe=500.0)(model)

    # -- time --
    dt, tstop = 0.001, 100
    freq, amp = 1, 0.5
    i_t = ax.sin(amp=amp, freq=freq).tstop(tstop)

    # -- run & record --
    indices = model.c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    states = [
        "v",
    ]
    rec = ax.callbacks.Recorder(states, node_indices=indices).set_hdf5(
        "tigerholm_voltage.h5", cache_every=args.cache_every
    )
    model.longrun(
        space=v_s, time=i_t, chunklength=args.chunklength, dt=dt, callbacks=[rec]
    )
    rec.close()
