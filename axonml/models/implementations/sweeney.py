from ..core import Myelinated
from ..mod import sweeney
from ..mechanisms import PARAMETER

from axonml.models.integrators import dufort_frankel

class Sweeney1987(Myelinated):
    """
    Implementation of the mammalian myelinated nerve fiber model by Sweeney et al. (1987).

    This model represents a myelinated axon with physiologically detailed representation
    of membrane dynamics based on modified Hodgkin-Huxley equations. It is specifically
    adapted for studying functional neuromuscular stimulation and the response of
    mammalian nerve fibers to electrical stimulation.

    Parameters
    ----------
    diameters : list of float, optional
        Axon diameters in μm. Default is [10.0].
    n_node : int, optional
        Number of nodes in the model. Default is 101.
    temp : float, optional
        Temperature in °C. Default is 37.0.
    v_init : float, optional
        Initial membrane potential in mV. Default is -80.0.
    method : str, optional
        Numerical integration method. Default is "dufort-frankel".

    Attributes
    ----------
    node_l : float
        Node of Ranvier length in μm. Set to 1.5 μm.
    cm : float
        Specific membrane capacitance in μF/cm². Set to 2.5 μF/cm².
    rhoa : float
        Axoplasmic resistivity in Ω·cm. Set to 54.7 Ω·cm.

    Notes
    -----
    The Sweeney model is based on experimental data from mammalian nerve fibers and
    includes voltage-gated sodium and potassium channels. It is particularly useful for
    studying how changes in stimulus parameters affect axonal excitation and conduction.

    The model incorporates:
    - Fast sodium channels
    - Slow potassium channels
    - Nodal and internodal segments with appropriate geometric parameters
    - Temperature-dependent channel kinetics

    Key features of this model compared to other myelinated models:
    - More accurate representation of mammalian fiber properties compared to amphibian models
    - Adapted for studying functional electrical stimulation applications
    - Validated against experimental data for stimulation thresholds

    References
    ----------
    .. [1] Sweeney JD, Mortimer JT, Durand D (1987). Modeling of mammalian myelinated
           nerve for functional neuromuscular stimulation. IEEE 9th Annual Conference
           of the Engineering in Medicine and Biology Society, 1577-1578.

    .. [2] Sweeney JD, Mortimer JT, Durand D (1987). A model of the nerve fiber
           membrane under electrical stimulation. IEEE Transactions on Biomedical
           Engineering, 34(8), 630-7. doi:10.1109/TBME.1987.325975
    """

    PARAMETER(inherit=Myelinated, node_l=1.5, membrane={"cm": 2.5, "rhoa": 54.7})

    def __init__(
        self,
        diameters=[10.0],
        n_node=101,
        temp=37.0,
        v_init=-80.0,
        integrator=dufort_frankel(),
    ):
        super().__init__(diameters, n_node, temp, v_init, integrator)
        self.insert(sweeney)
