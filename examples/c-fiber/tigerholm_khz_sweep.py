import itertools

import torch
import numpy as np
import pandas as pd

import axonml as ax
from axonml.models.parametric import distributed
from axonml.units import mm, um, nA, Hz, ms


# decide if you want to retry with slower numerical methods if some 
# of the simulations fail. Default = False (won't rerun).
rerun_anomalies = False

# record voltage? This will slow down the simulation by ~15%.
# it will also generate a large datafile so run it in /work
# it will write a file model_name_voltage.h5 that you can read
# with axonml.data.H5Reader
record_v = False

# declare the parameters you're going to sweep
frequencies = [
    1, 2, 5, 10, 20, 30, 40, 50, 60, 80, 100
]
amps = np.arange(0, 26.1).tolist()
all_params = list(itertools.product(frequencies, amps)) # in AxonML, we run everything at once
total = len(all_params)

# choose model
model_type = ax.Tigerholm2014
model_name = 'tigerholm'
steady_state = False

# global parameters
tstop = 2000 * ms
dt = 0.001 * ms

pre = 200 * ms
off = 610 * ms

# fastest is 32-bit with default (Dufort-Frankel) integrator so try that first
model = model_type([1.0*um]*total, L=40.0*mm, dx=10.0*um)

# define simulation
def run(model, params, rec_suffix='', steady_state=False):

    # only necessary for Schild
    if steady_state:
        model.steady_state()
    
    t = torch.arange(0, tstop, dt)
    
    f_s, a_s = map(list, zip(*params))
    f_s = distributed(f_s, over='a', kind='stim')
    a_s = distributed(a_s, over='a', kind='stim')
    stim = ax.sin(amp=a_s, freq=f_s, delay=pre, off=off)
    
    # deliver intracellular current pulses (1 nA, 1 ms pw) @ 10 Hz after 15 ms
    # fastest is to precompute i_intra(t) and add to IntraStim object
    
    testpulse_fs = 10 * Hz
    testpulse_start = 15 * ms
    intra = ax.IntraStim(model)
    i_stim = ax.mono_rect(amp=1*nA, pw=1*ms).repeat(testpulse_fs, delay=testpulse_start)(t) 
    intra.insert(i_stim, nodes=model.c(0.1))

    # ve from point source
    ve_s = ax.isotropic_point(z=200.0, rhoe=100/1.79)(model)

    # setup callbacks
    # models will not terminate if they encounter numerical error, so use
    # AnomalyDetector to determine which results are valid
    
    anom = ax.callbacks.AnomalyDetector()
    rec = ax.callbacks.Raster(node_check=model.c(0.9), dt=dt)

    if record_v:
        indices = model.c(.1, .4, .5, .501, .502, .505, .51, .55, .6, .65, .7, .9)
        v_rec = ax.callbacks.Recorder(['v'], node_indices=indices).set_hdf5(
            f"{model_name}_voltage{rec_suffix}.h5", cache_every=10000
        )
        callbacks = [anom, rec, v_rec]
    else:
        callbacks = [anom, rec]

    model.longrun(
        space=ve_s, 
        time=stim,
        tstop=tstop,
        dt=dt,
        intra=intra,
        chunklength=1000, 
        callbacks=callbacks, 
        reinit=True, 
    )

    if record_v:
        v_rec.close()

    return anom.numpy(), rec.numpy()

# run
anomalous, raster = run(model, all_params, steady_state=steady_state)

# rerun with slower methods if any anomalous
n_anomalous = np.count_nonzero(anomalous)

if n_anomalous > 0 and rerun_anomalies:
    print('Anomalies detected. Trying with ETD1 integrator & 64-bit math.')
    robust_model = model_type(
        [1.0*um]*n_anomalous, L=40.0*mm, dx=10.0*um, integrator=ax.krylov_etd1(m=6)
    ).double()
    rerun_params = [all_params[i] for i, f in enumerate(anomalous) if f]
    anomalous_r, raster_r = run(robust_model, rerun_params, '_robust', steady_state=steady_state)


# put together data

all_times = []
n_aps_during_khz = []
all_freq = []
all_amps = []
valid = []

anomalous_i = 0

for i in range(raster.shape[1]):
    
    freq, amp = all_params[i]
    
    if anomalous[i]:
        if not rerun_anomalies:
            valid.append(False)
            continue
        if anomalous_r[anomalous_i]:
            valid.append(False)
            anomalous_i += 1
            continue
        r = raster_r
        anomalous_i += 1
    else:
        r = raster
        
    t_ap = np.where(r[:, i, 0])[0] * dt
    all_times.append(t_ap.tolist())
    n_ap = np.count_nonzero(np.logical_and(t_ap>pre, t_ap<off))
    n_aps_during_khz.append(n_ap)
    all_freq.append(freq)
    all_amps.append(amp)
    valid.append(True)

data = {
    'all spike times (ms)': all_times,
    '# APs during kHz': n_aps_during_khz,
    'freq (kHz)': all_freq,
    'amplitude (mA)': all_amps,
    'valid': valid
}

df = pd.DataFrame(data)
df.to_pickle(f'{model_name}_khz_df.pkl')
