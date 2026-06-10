import matplotlib.pyplot as plt
import numpy as np
import torch

import dendra as dn
from dendra.helpers import CUDA
from dendra.models.callbacks import RecorderLambda
from dendra.models.mod import exp2syn, hh
from dendra.units import ms

dn.set_jit_enabled(True)  # True to enable jit compilation globally

# HINT: keep the total number of dn.Populations in your network to a minimum
# this will yield the most optimal performance
# in the following, we have a single Population with two labeled subpopulations
# by default, Populations are single compartment, so this network consists
# of two populations of 500 single compartment neurons

# this networks exhibits alternating oscillatory behavior

N1 = 500
N2 = 500
pop = dn.Population(1, N1 + N2, v_init=-65.0, celsius=6.5)
pop.insert(hh)
pop.insert(exp2syn.rename("NMDA"), e=0.0, tau1=0.1, tau2=1.0)
pop.insert(exp2syn.rename("AMPA"), e=-100.0, tau1=0.1, tau2=2.0)

pop[0, :500].label("P1")
pop[0, 500:].label("P2")

# you can set base rate, randomness, and max # spikes as with NEURON's NetStim
ns = dn.NetStim(N=100, interval=5.0, noise=1.0, seed=0, max_spikes=10)
net = dn.Network({"all": pop}, netstim=ns)
if CUDA:
    net = net.cuda()


net.clear_synapses()


net.connect_one_to_one(  # netstim -> P1
    net.netstim,
    # 100 random cells in P1 receive input
    net.all.P1[torch.arange(N1)[torch.randperm(N1)[:100]]],
    net.all.mech.NMDA,
    weight=0.1,
)

net.connect_prob(  # inhibitory cross-connections
    net.all.P1, net.all.P2, net.all.mech.AMPA, prob=0.02, weight=1.0, delay=1.0 * ms
)

net.connect_prob(
    net.all.P2, net.all.P1, net.all.mech.AMPA, prob=0.02, weight=1.0, delay=1.0 * ms
)

net.connect_prob(  # excitatory self connections
    net.all.P1, net.all.P1, net.all.mech.NMDA, prob=0.01, weight=1.0, delay=0.2 * ms
)

net.connect_prob(
    net.all.P2, net.all.P2, net.all.mech.NMDA, prob=0.01, weight=1.0, delay=0.2 * ms
)

dt = 0.025 * ms
tstop = 100 * ms
rec = RecorderLambda({"v": lambda net: net.all.v})

rec.reset()
net.initialize(dt)
net.run(tstop=tstop, callbacks=[rec], progressbar=True)

fig = plt.figure(dpi=200, figsize=(3, 3))
v = rec.numpy("v")
plt.plot(np.arange(0, tstop + dt, dt), v[:, 0, :500].mean(-1), label="mean P1 v")
plt.plot(np.arange(0, tstop + dt, dt), v[:, 0, 500:].mean(-1), label="mean P2 v")
plt.legend()
plt.show()
