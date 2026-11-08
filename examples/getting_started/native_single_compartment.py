"""Record the voltage response of a passive compartment to a current pulse."""

import dendra as dn
from dendra.models.mod import pas
from dendra.units import ms, nA, um


def main():
    with dn.ctx(DEVICE="cpu", JIT=0):
        model = dn.SingleCompartment(N=1, C=1, v_init=-65.0)
        model.diam.fill_(20.0 * um)
        model.dx.fill_(20.0 * um)
        model.insert(pas, g=0.001, e=-65.0)
        model[..., 0].inject(dn.mono_rect(amp=0.1 * nA, delay=0.1 * ms, pw=0.2 * ms))
        recorder = dn.Recorder(("v", "t"))
        model.initialize()
        model.run(dt=0.025 * ms, duration=1.0 * ms, callbacks=[recorder])

    time = recorder.numpy("t")
    voltage = recorder.numpy("v")[:, 0, 0]
    print(f"Recorded {time.size} samples through {time[-1]:.3f} ms")
    print(f"Peak membrane voltage {voltage.max():.3f} mV")


if __name__ == "__main__":
    main()
