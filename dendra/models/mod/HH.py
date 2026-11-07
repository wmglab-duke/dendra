from ..mechanisms import Mechanism as M
from ..mechanisms import State as S
from ..mechanisms import *
from ..mechanisms.ops import exp, vtrap


class mhn(S):
    """Independent HH gates in canonical rate form ``x' = x_a - x_b*x``.

    Assigned fields contain only the temperature-adjusted source and decay
    rates. Equilibrium initialization uses the unscaled rates directly; the
    former public ``minf``/``mtau`` (and h/n equivalents) are not computed.
    """

    S.STATE("m", "h", "n")
    S.DERIVATIVE("m' = m_a - m_b * m", "h' = h_a - h_b * h", "n' = n_a - n_b * n")
    S.ASSIGNED("m_a", "m_b", "h_a", "h_b", "n_a", "n_b")
    S.DERIVED_BUFFER("q10")
    S.GLOBAL(am1=0.1, am2=4.0, ah1=0.07, ah2=1.0, an1=0.01, an2=0.125)

    def derive_buffers(self):
        return {"q10": 3.0 ** ((self.celsius - 6.3) / 10.0)}

    def assigned_values(self, v, values):
        del values
        alpha_m = self.am1 * vtrap(-(v + 40), 10)
        beta_m = self.am2 * exp(-(v + 65) / 18)
        alpha_h = self.ah1 * exp(-(v + 65) / 20)
        beta_h = self.ah2 / (exp(-(v + 35) / 10) + 1)
        alpha_n = self.an1 * vtrap(-(v + 55), 10)
        beta_n = self.an2 * exp(-(v + 65) / 80)
        q10 = self.q10
        return {
            "m_a": q10 * alpha_m,
            "m_b": q10 * (alpha_m + beta_m),
            "h_a": q10 * alpha_h,
            "h_b": q10 * (alpha_h + beta_h),
            "n_a": q10 * alpha_n,
            "n_b": q10 * (alpha_n + beta_n),
        }

    def state_defaults(self, v, values):
        del values
        alpha_m = self.am1 * vtrap(-(v + 40), 10)
        beta_m = self.am2 * exp(-(v + 65) / 18)
        alpha_h = self.ah1 * exp(-(v + 65) / 20)
        beta_h = self.ah2 / (exp(-(v + 35) / 10) + 1)
        alpha_n = self.an1 * vtrap(-(v + 55), 10)
        beta_n = self.an2 * exp(-(v + 65) / 80)
        # Temperature scales both rates equally. Avoid a q10 ratio here so
        # initial equilibrium remains exactly independent of temperature.
        return {
            "m": alpha_m / (alpha_m + beta_m),
            "h": alpha_h / (alpha_h + beta_h),
            "n": alpha_n / (alpha_n + beta_n),
        }


class hh(M):
    """
    The Hodgkin-Huxley neuron model with sodium, potassium, and leak channels.
    This model includes the dynamics of the gating variables `m`, `h`, and `n`
    which represent the activation and inactivation of sodium and potassium channels.

    Parameters
    ----------
    gnabar : float
        Maximum sodium conductance density in S/cm². Default is 0.12 S/cm²
        (equivalently 120 mS/cm²).
    gkbar : float
        Maximum potassium conductance density in S/cm². Default is 0.036 S/cm²
        (equivalently 36 mS/cm²).
    gl : float
        Leak conductance density in S/cm². Default is 0.0003 S/cm²
        (equivalently 0.3 mS/cm²).
    ena : float
        Sodium reversal potential (in mV). Default is 50.0.
    ek : float
        Potassium reversal potential (in mV). Default is -77.0.
    el : float
        Leak reversal potential (in mV). Default is -54.3.

    Notes
    -----
    The single registered nonspecific current ``i(v)`` is the sum of sodium,
    potassium, and leak currents. ``i_affine_terms(v)`` supplies conductance
    density ``G`` in S/cm² and source density ``J`` in mA/cm², with the
    outward-positive convention ``i(v) = G*v - J``. Dendra generates the
    ordinary current and conductance methods from these same coefficients.

    There are no separate ``ina``, ``ik``, or ``il`` methods. Subclasses
    changing membrane physics should override ``i_affine_terms``
    and redeclare ``AFFINE("i")``, or provide their own current declaration.

    Gates use rate coefficients ``m_a, m_b, h_a, h_b, n_a, n_b`` in
    ``x' = x_a - x_b*x``.
    """

    M.STATE_BUNDLE(mhn)
    M.GLOBAL(gnabar=0.12, gkbar=0.036, gl=0.0003, ena=50.0, ek=-77.0, el=-54.3)

    M.NONSPECIFIC_CURRENT("i")
    M.AFFINE("i")

    def i(self, v):
        G, J = self.i_affine_terms(v)
        return G * v - J

    def i_affine_terms(self, v):
        """Return total HH ``(G, J)`` with the local gates held fixed."""
        gna = self.gnabar * self.m**3 * self.h
        gk = self.gkbar * self.n**4
        return (
            self.gl + gna + gk,
            self.gl * self.el + gna * self.ena + gk * self.ek,
        )
