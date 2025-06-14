import torch

from ..core import Myelinated
from ..mod import axnode_myel
from ..declarations import PARAMETER

from axonml.models.integrators import eulerv1


ic = {"m": 0.0732093, "h": 0.62069505, "p": 0.20260409, "s": 0.04302994}


class SMF(Myelinated):
    """
    Surrogate Myelinated Fiber (S-MF) model from Hussain et al. (2024).

    This model implements a highly efficient GPU-based surrogate for the McIntyre-Richardson-Grill (MRG)
    myelinated axon model, designed to accurately replicate neural responses to electrical stimulation
    while achieving exceptional computational efficiency. The model uses simplified myelinated cable
    geometry with reparameterized non-linear ionic conductances to reproduce the behavior of the
    detailed NEURON implementation with orders-of-magnitude faster performance.

    Parameters
    ----------
    diameters : list of float, optional
        Axon diameters in μm. Must be ≥ 5.7 μm. Default is [8.0].
    n_node : int, optional
        Number of nodes in the model. Default is 101.
    temp : float, optional
        Temperature in °C. Default is 37.0.
    v_init : float, optional
        Initial membrane potential in mV. Default is -80.0.
    method : str, optional
        Numerical integration method. Only 'euler' or 'rk1' supported. Default is "euler".

    Attributes
    ----------
    node_l : float
        Node of Ranvier length in μm. Default is 1.0 μm.
    axon_d : dict
        Polynomial coefficients for axon diameter scaling.
    node_d : dict
        Polynomial coefficients for node diameter scaling.
    delta_x : dict
        Polynomial coefficients for internodal distance scaling.
    membrane : dict
        Membrane electrical properties including cm (10 μF/cm²) and rhoa (70 Ω·cm).

    Notes
    -----
    The S-MF model is designed to enable rapid exploration of electrical stimulation parameters
    while maintaining high accuracy compared to the gold-standard MRG model. It provides
    up to 1,000,000x speedup over single-core simulations in NEURON while accurately predicting:

    - Full spatiotemporal responses of nerve fibers to electrical stimulation
    - Action potential generation, propagation, and conduction velocity
    - Response to complex waveforms and electrode configurations
    - Nonlinear phenomena such as conduction block and subthreshold modulation
    - Dependence on prior excitation history

    Key features:
    - GPU-accelerated implementation under the AxonML framework
    - Compatible with monopolar and multipolar stimulation configurations
    - Handles various stimulus waveforms including conventional and kilohertz frequency signals
    - Supports different nerve morphologies and fiber diameters
    - Enables both gradient-free and gradient-based optimization approaches

    Applications:
    - Design of selective neural stimulation protocols
    - Optimization of electrode configurations and waveform parameters
    - Large-scale simulations of fiber populations
    - Development of neuromodulation therapies for various conditions
    - Studying complex nonlinear neural phenomena

    Limitations:
    - Minimum fiber diameter of 5.7 μm (use smolMRG for smaller fibers)
    - Limited to 'euler' or 'rk1' integration methods for computational efficiency
    - Simplified geometry compared to the full MRG model

    References
    ----------
    .. [1] M. A. Hussain, W. M. Grill, and N. A. Pelot, “Highly efficient modeling
           and optimization of neural fiber responses to electrical stimulation,”
           Nat Commun, vol. 15, no. 1, p. 7597, Aug. 2024, doi: 10.1038/s41467-024-51709-8.

    .. [2] McIntyre CC, Richardson AG, Grill WM (2002). Modeling the excitability of
           mammalian nerve fibers: influence of afterpotentials on the recovery cycle.
           Journal of Neurophysiology, 87(2), 995-1006.
    """

    PARAMETER(
        inherit=Myelinated,
        node_l=1.0,
        axon_d={
            "axond1": 0.0187623,
            "axond2": 4.787487e-01,
            "axond3": 1.203613e-01,
        },
        node_d={
            "noded1": 6.303781e-03,
            "noded2": 2.070544e-01,
            "noded3": 5.339006e-01,
        },
        delta_x={
            "deltax1": -8.215284e00,
            "deltax2": 2.724201e02,
            "deltax3": -7.802411e02,
        },
        membrane={
            "cm": 10e-3,
            "rhoa": 70.0,  # ohm-cm
        },
    )

    def __init__(
        self,
        diameters=[8.0],
        n_node=101,
        temp=37.0,
        v_init=-80.0,
        integrator=eulerv1(),
    ):
        if torch.any(torch.as_tensor(diameters) < 5.7):
            raise ValueError(
                "Fiber diameter should not be less than 5.7 um for SMF. Use smolMRG instead."
            )
        super().__init__(diameters, n_node, temp, v_init, integrator)
        self.insert(axnode_myel, ic=ic)

    def cm_(self, area):
        return self.cm * area
