import torch
from axonml.models import Tigerholm
from axonml.models.callbacks import Recorder

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--save_every", type=int, default=10000)
parser.add_argument("--n_chunks", type=int, default=1000)
args = parser.parse_args()

n_ax = 10000
L = 50      # mm
dx = 25.0   # um

diameters = torch.linspace(0.5, 2.0, n_ax)
model = Tigerholm(diameters, L, dx=dx, method="euler").cuda()

# -- space --
x = model.x()
z = 100.0
r = torch.sqrt(z**2 + x**2) * 1e-4
v_s = (1000 / (4 * torch.pi * 500 * r)).unsqueeze(0)

# -- time --
dt = 0.001
tstop = 100
t = torch.arange(0, tstop, dt)
amplitude = 20.0
i_t = amplitude * torch.sin(t * torch.pi).unsqueeze(0)

# -- run & record --
rec = Recorder(["v"], node_indices=model.c(0.4, 0.5)).set_hdf5(
    "tigerholm_voltage.h5", save_every=args.save_every
)
model.longrun(space=v_s, time=i_t, n_chunks=args.n_chunks, dt=dt, callbacks=[rec])
rec.close()
