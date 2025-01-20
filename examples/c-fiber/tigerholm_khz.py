import torch
from axonml.models import Tigerholm
from axonml.models.callbacks import Recorder


n_ax = 10000
L = 50  # mm
dx = 25.0

diameters = torch.linspace(0.5, 2.0, n_ax)
model = Tigerholm(diameters, L, dx=dx, method="euler").cuda()

# -- space --
x = model.x()

z = 100.0
r = torch.sqrt(z**2 + x**2) * 1e-4
ve_s = (1000 / (4 * torch.pi * 500 * r)).unsqueeze(0)

# -- time --
dt = 0.001
tstop = 100
t = torch.arange(0, tstop, dt)

amplitude = 20.0
ve_t = amplitude * torch.sin(t * 1 * torch.pi * 1).unsqueeze(0)

# -- run & record --
rec = Recorder(["v"], node_indices=model.c(0.4, 0.5)).set_hdf5("tigerholm_voltage.h5")
model.longrun(ve_s=ve_s, ve_t=ve_t, n_chunks=1000, dt=dt, reinit=True, callbacks=[rec])
rec.close()
