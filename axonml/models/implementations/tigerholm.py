from ..core import Unmyelinated
from ..mod import (
    ks,
    kf,
    h,
    nattxs,
    nav1p8,
    nav1p9_slow_inact,
    nakpump,
    kdrTiger,
    kna,
    naoiTiger,
    koiTiger,
    leak,
)
from ..mechanisms import PARAMETER, c_context


class Tigerholm(Unmyelinated):
    """
    Computational model of unmyelinated C-fiber nociceptors based on Tigerholm et al. (2014).
    
    This model implements the biophysically detailed C-fiber axon described in 
    "Modeling activity-dependent changes of axonal spike conduction in primary 
    afferent C-nociceptors" (Tigerholm et al., 2014). The model includes multiple 
    voltage-gated ion channels, ion accumulation/diffusion mechanisms, and pump 
    dynamics that govern the excitability and conduction properties of unmyelinated 
    nociceptive axons.
    
    Parameters
    ----------
    diameters : list of float, optional
        Axon diameter(s) in μm. Default is [1.0].
    L : float, optional
        Length of axon in mm. Default is 5.0.
    dx : float, optional
        Spatial discretization step in μm. Default is 10.
    temp : float, optional
        Temperature in °C. Default is 37.0.
    v_init : float, optional
        Initial membrane potential in mV. Default is -55.0.
    method : str, optional
        Numerical integration method. Default is "dufort-frankel".
    
    Attributes
    ----------
    cm : float
        Specific membrane capacitance in μF/cm². Default is 1.0.
    rhoa : float
        Axoplasmic resistivity in Ω·cm. Default is 35.4.
    
    Notes
    -----
    The model includes the following ion channels and mechanisms:
    
    - ks: Slow potassium channel
    - kf: Fast potassium channel
    - h: Hyperpolarization-activated cyclic nucleotide-gated (HCN) channel
    - nattxs: TTX-sensitive sodium channel
    - nav1p8: Voltage-gated sodium channel Nav1.8
    - nav1p9_slow_inact: Voltage-gated sodium channel Nav1.9 with slow inactivation
    - nakpump: Na⁺/K⁺ ATPase pump
    - kdrTiger: Delayed rectifier potassium channel
    - kna: Sodium-activated potassium channel
    - naoiTiger: Sodium ion accumulation/diffusion mechanism
    - koiTiger: Potassium ion accumulation/diffusion mechanism
    - leak: Background leak conductances for Na⁺ and K⁺
    
    This model is particularly useful for studying:
    - Activity-dependent slowing (ADS) of conduction velocity
    - Effects of repetitive stimulation on nociceptor excitability
    - Pain signaling in unmyelinated C-fibers
    
    References
    ----------
    .. [1] Tigerholm J, Petersson ME, Obreja O, Lampert A, Carr R, Schmelz M,
           Fransén E (2014). Modeling activity-dependent changes of axonal spike
           conduction in primary afferent C-nociceptors. J Neurophysiol 111(9):
           1721-35. doi:10.1152/jn.00777.2012
    """
    PARAMETER(cm=1.0, rhoa=35.4)

    def __init__(
        self,
        diameters=[1.0],
        L=5.0,
        dx=10,
        temp=37.0,
        v_init=-55.0,
        method="dufort-frankel",
    ):
        super().__init__(diameters, L, dx, temp, v_init, method)

        with c_context(nai0=11.4, nao0=154.0, ki0=144.9, ko0=5.6):
            self.insert(ks, gbar=0.0069733)
            self.insert(kf, gbar=0.012756)
            self.insert(h, gbar=0.0025377)
            self.insert(nattxs, gbar=0.10664)
            self.insert(nav1p8, gbar=0.24271)
            self.insert(nav1p9_slow_inact, gbar=9.4779e-05)
            self.insert(nakpump, smalla=-0.0047891)
            self.insert(kdrTiger, gbar=0.018002)
            self.insert(kna, gbar=0.00042)
            self.insert(naoiTiger)
            self.insert(koiTiger)
            self.insert(
                leak, gkleak=1.3155237866158132e-05, gnaleak=2.1094052499393e-05
            )
