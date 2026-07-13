import matplotlib.pyplot as plt
import numpy as np
import torch

import dendra as dn
from dendra.models.callbacks import RecorderLambda
from dendra.models.mod import exp2syn, hh
from dendra.units import ms

dn.set_jit_enabled(False)  # True to enable jit compilation globally

N = 4
pop = dn.Population(1, N, v_init=-65.0, celsius=6.5)
pop.insert(hh)
pop.insert(exp2syn.rename("NMDA"), e=0.0, tau1=0.1, tau2=1.0)

ns = dn.NetStim(N=1, noise=0, seed=0, max_spikes=1)
net = dn.Network({"hh": pop}, netstim=ns)  # .cuda() to run on GPU

net.clear_synapses()

net.connect_one_to_one(
    net.netstim[0],
    net.hh[0, 0],
    net.hh.mech.NMDA,
)

net.connect_one_to_one(
    net.hh[0, :],
    net.hh[0, torch.roll(torch.arange(N), -1)],
    net.hh.mech.NMDA,
    weight=1.0,  # bare µS coordinate for exp2syn
    delay=5.0 * ms,
)

dt = 0.025 * ms
rec = RecorderLambda({"v": lambda net: net.hh.v})

rec.reset()
net.initialize_(dt)
net.run(tstop=50.0 * ms, callbacks=[rec], progressbar=True)

# visualize
fig = plt.figure(dpi=200, figsize=(3, 3))
v = rec.numpy("v")
plt.plot(np.arange(0, 50 + dt, dt), v[:, 0, :])
plt.show()
