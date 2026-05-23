r"""Voltage-gated graded / continuous chemical synapses for Dendra.

This module implements a simple kinetic conductance synapse for graded or
non-spiking chemical transmission. The formulation follows the kinetic-synapse
modeling tradition of Destexhe, Mainen, and Sejnowski, in which synaptic
conductance is derived from receptor or release-gate dynamics rather than from
a prescribed spike-triggered waveform [1]_ [2]_.

The presynaptic release gate obeys

.. math::

    \dot{s}
    =
    \alpha T(V_\mathrm{pre})(1-s) - \beta s,

where the voltage-dependent release activation is

.. math::

    T(V_\mathrm{pre})
    =
    \frac{1}{1 + \exp[-(V_\mathrm{pre}-\theta)/\sigma]}.

The postsynaptic current is conductance-based, using Dendra's outward-current
convention,

.. math::

    I_\mathrm{syn}
    =
    g_\mathrm{scale}\,g_\mathrm{pre}(V_\mathrm{post}-E_\mathrm{syn}).

Here ``g_pre`` is the weighted, summed analog presynaptic drive delivered by
``Network.connect_continuous(...)``.

Components
----------
``graded_release_gate``
    Presynaptic mechanism that computes a dynamic voltage-dependent release
    gate ``s``. Use ``pre_var="mech.graded_release_gate.s"`` when connecting.

``graded_syn``
    Postsynaptic ``ContinuousSynapse`` that receives the analog input
    ``g_pre`` and contributes a conductance-based current.

``sigmoid_release``
    Stateless voltage-to-release transform for instantaneous graded projections.
    Use this with ``connect_continuous(..., pre_var="v", transform=...)`` when
    an explicit presynaptic gate state is not needed.

Use cases
---------
This mechanism is useful for graded chemical synapses, voltage-dependent
transmitter release, analog rate-like projections, and models in which the
presynaptic cell exposes a continuous synaptic state rather than discrete spike
events.

References
----------
.. [1] Destexhe, A., Mainen, Z. F., & Sejnowski, T. J. (1994).
   An efficient method for computing synaptic conductances based on a kinetic
   model of receptor binding. *Neural Computation*, 6(1), 14–18.

.. [2] Destexhe, A., Mainen, Z. F., & Sejnowski, T. J. (1994).
   Synthesis of models for excitable membranes, synaptic transmission and
   neuromodulation using a common kinetic formalism.
   *Journal of Computational Neuroscience*, 1, 195–230.


Examples
--------
Example 1: explicit presynaptic gate state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import dendra as dn
    from dendra.models.networks import Network
    from dendra.models.mod.graded_syn import graded_release_gate, graded_syn

    pre = dn.Population(N=10, C=1, v_init=-60.0)
    post = dn.Population(N=10, C=1, v_init=-65.0)

    # Presynaptic graded transmitter/release gate.
    pre.insert(
        graded_release_gate,
        theta=-20.0,   # mV, half-activation voltage
        sigma=2.0,     # mV, release steepness
        alpha=1.0,     # ms^-1, opening / activation rate scale
        beta=0.2,      # ms^-1, closing / deactivation rate
    )

    # Postsynaptic conductance-based continuous synapse.
    post.insert(graded_syn, e=-80.0)  # inhibitory; use e=0 for excitatory

    net = Network(dict(pre=pre, post=post))
    net.connect_continuous(
        source=pre,
        target=post,
        synapse=post.mech.graded_syn,
        conn_spec={"rule": "one_to_one"},
        pre_var="mech.graded_release_gate.s",
        input="g_pre",
        weight=0.05,
        delay=0.0,
        reduce="sum",
    )

    net.initialize(dt=0.025)
    net.run(tstop=100.0, progressbar=False)

Example 2: instantaneous voltage-to-release transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from dendra.models.mod.graded_syn import graded_syn, sigmoid_release

    post.insert(graded_syn, e=0.0)

    net.connect_continuous(
        source=pre,
        target=post,
        synapse=post.mech.graded_syn,
        conn_spec={"rule": "all_to_all"},
        pre_var="v",
        transform=sigmoid_release(theta=-20.0, sigma=2.0),
        input="g_pre",
        weight=0.001,
    )

Notes on units
--------------
``graded_syn`` is a density-style ``ContinuousSynapse`` mechanism, not a
``PointProcess``.  That means its current is interpreted in the same density
convention as ordinary Dendra mechanisms.  The connection ``weight`` should
therefore have the units/convention of conductance used by the target model.

For lumped nA/uS point-process semantics, define a separate class that combines
Dendra's ``PointProcess`` with ``ContinuousSynapse``.
"""

from __future__ import annotations

import torch

from ..mechanisms import ContinuousSynapse as CS
from ..mechanisms import Mechanism as M
from ..mechanisms import State as S


class sigmoid_release(torch.nn.Module):
    """Memoryless sigmoid transform from presynaptic voltage to release gate.

    Parameters
    ----------
    theta : float, default -20.0
        Half-activation voltage in mV.
    sigma : float, default 2.0
        Slope factor in mV. Positive values make release increase with
        depolarization.  Smaller values produce steeper activation.

    Notes
    -----
    This transform is intended for use with ``Network.connect_continuous``:

    .. code-block:: python

        net.connect_continuous(
            pre, post, post.mech.graded_syn,
            pre_var="v",
            transform=sigmoid_release(theta=-20.0, sigma=2.0),
            input="g_pre",
            weight=0.01,
        )

    For a dynamic release gate with its own state, use
    :class:`graded_release_gate` instead.
    """

    def __init__(self, theta: float = -20.0, sigma: float = 2.0):
        super().__init__()
        self.theta = float(theta)
        self.sigma = float(sigma)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        sigma = torch.as_tensor(self.sigma, dtype=v.dtype, device=v.device).clamp_min(
            1.0e-6
        )
        theta = torch.as_tensor(self.theta, dtype=v.dtype, device=v.device)
        return torch.sigmoid((v - theta) / sigma)

    def extra_repr(self) -> str:
        return f"theta={self.theta}, sigma={self.sigma}"


class graded_release_state(S):
    r"""State bundle for the presynaptic graded transmitter-release gate.

    The dynamics are

    .. math::

        \dot s = (s_\infty(V)-s)/\tau_s(V),

    with

    .. math::

        s_\infty(V) = \frac{\alpha T(V)}{\alpha T(V)+\beta},
        \quad
        \tau_s(V) = \frac{1}{\alpha T(V)+\beta}.

    This is algebraically equivalent to

    .. math::

        \dot s = \alpha T(V)(1-s)-\beta s.
    """

    S.STATE("s")
    S.ASSIGNED("T", "sinf", "tau")
    S.RANGE(theta=-20.0, sigma=2.0, alpha=1.0, beta=0.2)
    S.DERIVATIVE("s' = (sinf - s) / tau")

    def _release(self, v: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma.clamp_min(1.0e-6)
        return torch.sigmoid((v - self.theta) / sigma)

    def breakpoint(self, v, states):
        T = self._release(v)
        denom = (self.alpha * T + self.beta).clamp_min(1.0e-12)
        sinf = (self.alpha * T) / denom
        tau = 1.0 / denom
        return {"T": T, "sinf": sinf, "tau": tau}

    def inf(self, v):
        bp = self.breakpoint(v, None)
        return {"s": bp["sinf"]}


class graded_release_gate(M):
    r"""Presynaptic voltage-dependent graded transmitter-release gate.

    This mechanism exposes the state variable ``s`` for use as a continuous
    presynaptic variable in ``Network.connect_continuous``.

    Parameters
    ----------
    theta : float, default -20.0
        Presynaptic voltage at which release activation ``T(V)`` is 0.5.
    sigma : float, default 2.0
        Sigmoid slope factor in mV. Positive values make the gate activate with
        depolarization.
    alpha : float, default 1.0
        Opening/activation rate scale in ms^-1.
    beta : float, default 0.2
        Closing/deactivation rate in ms^-1.

    Notes
    -----
    ``graded_release_gate`` exposes the continuous release state ``s``. This is
    the variable that should usually be used as ``pre_var`` in
    ``Network.connect_continuous``.

    .. rubric:: Exposed variables

    .. list-table::
       :widths: 18 24 58
       :header-rows: 1

       * - Variable
         - Type
         - Description
       * - ``s``
         - ``torch.Tensor``
         - Continuous release / gating variable in ``[0, 1]``. Use this as the
           ``pre_var`` for :class:`graded_syn`.
       * - ``T``
         - ``torch.Tensor``
         - Instantaneous voltage-dependent release activation.
       * - ``sinf``
         - ``torch.Tensor``
         - Steady-state value of the release gate.
       * - ``tau``
         - ``torch.Tensor``
         - Voltage-dependent release-gate time constant.

    Examples
    --------
    Use ``s`` as the continuous presynaptic variable:

    .. code-block:: python

        net.connect_continuous(
            pre,
            post,
            post.mech.graded_syn,
            pre_var="mech.graded_release_gate.s",
            input="g_pre",
            weight=0.05,
        )
    """

    M.STATE(graded_release_state)


class graded_syn(CS):
    r"""Postsynaptic continuous conductance-based chemical synapse.

    ``graded_syn`` receives an analog presynaptic input named ``g_pre`` through
    Dendra's ``Network.connect_continuous`` path.  The default continuous receive
    behavior sums all incoming projections into ``g_pre`` once per timestep.

    The current is

    .. math::

        I_\mathrm{syn} = g_\mathrm{scale}\,g_\mathrm{pre}(V-E_\mathrm{syn}).

    Parameters
    ----------
    e : float, default 0.0
        Reversal potential in mV. Use ``e=0`` for a typical excitatory graded
        synapse, or ``e=-80`` for a typical inhibitory graded synapse.
    g_scale : float, default 1.0
        Optional multiplicative scale applied after continuous projection
        delivery. Usually leave this at 1 and use the connection ``weight`` as
        the synaptic conductance scale.

    Notes
    -----
    ``graded_syn`` declares a continuous input buffer named ``g_pre``. The
    network resets this buffer once per timestep and then sums weighted
    presynaptic analog gates into it through ``Network.connect_continuous``.

    .. rubric:: Continuous inputs

    .. list-table::
       :widths: 18 24 58
       :header-rows: 1

       * - Input
         - Type
         - Description
       * - ``g_pre``
         - ``torch.Tensor``
         - Weighted, summed presynaptic gate delivered by ``ContinuousCon``.
       * - ``g_pre_old``
         - ``torch.Tensor``
         - Previous-step value retained by
           ``ContinuousSynapse.reset_continuous_inputs``.

    Examples
    --------
    One-to-one inhibitory graded synapses:

    .. code-block:: python

        pre.insert(graded_release_gate)
        post.insert(graded_syn, e=-80.0)

        net.connect_continuous(
            pre,
            post,
            post.mech.graded_syn,
            conn_spec={"rule": "one_to_one"},
            pre_var="mech.graded_release_gate.s",
            input="g_pre",
            weight=0.05,
        )

    All-to-all excitatory graded synapses using a stateless transform:

    .. code-block:: python

        post.insert(graded_syn, e=0.0)
        net.connect_continuous(
            pre,
            post,
            post.mech.graded_syn,
            conn_spec={"rule": "all_to_all"},
            pre_var="v",
            transform=sigmoid_release(theta=-20.0, sigma=2.0),
            input="g_pre",
            weight=0.001,
        )
    """

    CS.INPUT("g_pre")
    CS.RANGE(e=0.0, g_scale=1.0)
    CS.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.g_scale * self.g_pre * (v - self.e)


__all__ = [
    "graded_release_gate",
    "graded_release_state",
    "graded_syn",
    "sigmoid_release",
]
