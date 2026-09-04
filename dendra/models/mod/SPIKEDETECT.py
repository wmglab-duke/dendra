import torch

from ..mechanisms import Mechanism as M


class spikedetect(M):
    r"""
    Upward threshold-crossing spike detector with straight-through (STE) gradients.

    This mechanism detects a spike when the membrane potential crosses a threshold
    *from below* between two discrete simulation steps, and exposes the result as
    a buffer variable `spikes`.

    The key design goal is:

    - **Forward pass semantics are discrete and event-like** (0/1 upward crossing).
    - **Backward pass provides gradients w.r.t. `v`** using a smooth surrogate.

    Parameters
    ----------
    threshold : float
        Voltage threshold :math:`\theta` for spike detection, in mV. Default is
        0.0 mV.
    tau_gate : float
        Smoothness/steepness parameter :math:`\tau_g` used to construct a
        differentiable gate from the voltage. Smaller values make the transition
        sharper (but may produce larger gradients). This is a voltage scale in
        mV; default is 0.5 mV.
    ste_scale : float
        Dimensionless scalar :math:`\alpha` multiplying the surrogate-gradient
        contribution. Default is 1.0.

    Notes
    -----

    **Buffer variables**

    - spikes: Spike indicator per compartment for the current step.
      Forward semantics are approximately 0/1 (hard), but it is differentiable
      w.r.t. `v` due to the STE construction.
    - h_prev: Memory of the previous step's gate value (continuous in [0,1]).

    **Mathematics**

    Let :math:`v_t` be the membrane potential at the current step and :math:`v_{t-1}`
    the previous step. Define a continuous “gate”:

    .. math::

        g_t = \sigma\left(\frac{v_t - \theta}{\tau_g}\right) \in (0,1),

    where :math:`\sigma(\cdot)` is the logistic sigmoid.

    A **hard upward crossing event** is defined in terms of gate thresholding at 0.5:

    .. math::

        s_t^{\text{hard}} = \mathbb{1}[g_t > 0.5]\;\mathbb{1}[g_{t-1} \le 0.5].

    Since :math:`g_t > 0.5 \iff v_t > \theta`, this corresponds to “crossing the
    voltage threshold upward” in discrete time.

    A **soft surrogate onset** (used only for gradients) is:

    .. math::

        s_t^{\text{soft}} = \max(g_t - g_{t-1}, 0).

    Finally, the emitted variable uses a **straight-through estimator**:

    .. math::

        \text{spikes}_t =
            s_t^{\text{hard}}
            + \alpha\left(s_t^{\text{soft}} - \operatorname{stopgrad}(s_t^{\text{soft}})\right),

    where :math:`\operatorname{stopgrad}(\cdot)` denotes the detach/stop-gradient
    operator. This makes the **forward value equal** to :math:`s_t^{\text{hard}}`,
    while the **backward gradient** is shaped by :math:`s_t^{\text{soft}}`.

    **Differentiability and graph growth**

    - `spikes` is differentiable w.r.t. `v` through the surrogate term.
    - The state `h_prev` stores the previous gate. If you do not want the autograd
      graph to backpropagate through time across many steps, you can optionally
      detach the memory update in ``advance``.

    **Efficient usage in a Network**

    A common pattern is to compute threshold crossings **once per presynaptic
    compartment** and reuse them as the presynaptic driver for *all* synapses that
    originate from that presynaptic population.

    1) Insert this mechanism into all presynaptic compartments (population `pre_pop`),
       so that each compartment exposes `mech.spikedetect.spikes`.

    2) Connect synapses using `pre_var='mech.spikedetect.spikes'`, e.g.:

    .. code-block:: python

        net.connect_one_to_one(
            pre_pop,
            post_pop,
            post_pop.mech.synapse,
            threshold=None,                 # thresholding already done in spikedetect
            delay=delay,
            weight=weight,
            pre_var='mech.spikedetect.spikes'
        )

    Why this is efficient:

    - The threshold crossing computation is performed **once** for each presynaptic
      compartment in the accepted-step transition.
    - All outgoing synapses from that presynaptic population can read the already-
      computed `spikes` tensor, avoiding redundant per-synapse threshold checks.

    In other words, spike detection becomes an O(#compartments) operation per step,
    rather than O(#synapses), which can be a significant speedup in dense networks.
    """

    M.RANGE(threshold=0.0, tau_gate=0.5, ste_scale=1.0)
    M.CARRY("spikes", "h_prev")

    def initial_values(self, v, values):
        # Initialize the gate memory to the current gate so we do NOT emit a spike at t=0
        # if v starts above threshold.
        thr = torch.as_tensor(self.threshold, dtype=v.dtype, device=v.device)
        tau = torch.as_tensor(self.tau_gate, dtype=v.dtype, device=v.device).clamp_min(
            1e-3
        )

        gate0 = torch.sigmoid((v - thr) / tau)
        return {"h_prev": gate0, "spikes": torch.zeros_like(v)}

    def advance(self, v, dt, values):
        del dt
        thr = self.threshold
        tau = self.tau_gate.clamp_min(1e-3)
        ste = self.ste_scale

        # Smooth gate in [0,1], with gate==0.5 at v==threshold
        gate = torch.sigmoid((v - thr) / tau)
        old_h = values["h_prev"]

        # Hard upward crossing: from <=0.5 to >0.5 (equivalent to crossing threshold from below)
        rising = (gate > 0.5) & (old_h <= 0.5)

        # Soft "edge" term: positive only when gate increases (focuses grads on upward changes)
        rise_soft = torch.relu(gate - old_h)

        # Straight-through estimator:
        #   forward: spikes == rising (0/1)
        #   backward: gradients come from rise_soft
        spikes = rising.to(v.dtype) + ste * (rise_soft - rise_soft.detach())

        # Update memory AFTER computing rising/rise_soft
        return {"spikes": spikes, "h_prev": gate}
        # If want to prevent building a long autograd graph through time, use:
        # self.h_prev = gate.detach(), but we don't do that here.
