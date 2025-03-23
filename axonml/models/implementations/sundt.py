from ..core import Unmyelinated
from ..mod import kdr, pas, nahh
from ..mechanisms import PARAMETER, e_context


class Sundt(Unmyelinated):
    """
    Implementation of the unmyelinated sensory neuron model by Sundt et al. (2015).

    This model represents an unmyelinated sensory axon with a focus on spike
    propagation through dorsal root ganglia. It implements a Hodgkin-Huxley type
    membrane dynamics with specific channel distributions tailored for sensory
    neurons. The model is particularly useful for studying action potential
    propagation in pain pathways and sensory information processing.

    Parameters
    ----------
    diameters : list of float, optional
        Axon diameters in μm. Default is [1.0].
    L : float, optional
        Length of axon in mm. Default is 5.0.
    dx : float, optional
        Spatial discretization step in μm. Default is 10.
    temp : float, optional
        Temperature in °C. Default is 37.0.
    v_init : float, optional
        Initial membrane potential in mV. Default is -65.0.
    method : str, optional
        Numerical integration method. Default is "dufort-frankel".

    Attributes
    ----------
    cm : float
        Specific membrane capacitance in μF/cm². Set to 1.0 μF/cm².
    rhoa : float
        Axoplasmic resistivity in Ω·cm. Set to 100.0 Ω·cm.

    Notes
    -----
    The Sundt model includes the following ion channels:
    - Delayed rectifier potassium channel (kdr) with gkbar=0.04 S/cm²
    - Hodgkin-Huxley sodium channel (nahh) with gnabar=0.04 S/cm²
    - Passive leak conductance (pas) with g=0.0001 S/cm²

    The model uses a potassium reversal potential (ek) of -90 mV and leak reversal
    potential of -65 mV. It is specifically designed to investigate how action potentials
    navigate the complex geometry and electrical properties of dorsal root ganglia.

    This model is useful for studying:
    - Sensory neuron excitability
    - Pain signal propagation
    - Effects of geometric irregularities (like T-junctions) on signal transmission
    - Sensory neuron pathophysiology

    References
    ----------
    .. [1] Sundt D, Gamper N, Jaffe DB (2015). Spike propagation through the dorsal
           root ganglia in an unmyelinated sensory neuron: a modeling study.
           Journal of Neurophysiology, 114(6), 3140-3153. doi:10.1152/jn.00226.2015
    """

    PARAMETER(cm=1.0, rhoa=100.0)

    def __init__(
        self,
        diameters=[1.0],
        L=5.0,
        dx=10,
        temp=37.0,
        v_init=-65.0,
        method="dufort-frankel",
    ):
        super().__init__(diameters, L, dx, temp, v_init, method)

        with e_context(ek=-90.0):
            self.insert(kdr, gkbar=0.04)
            self.insert(nahh, gnabar=0.04)
            self.insert(pas, g=0.0001, e=-65.0)
