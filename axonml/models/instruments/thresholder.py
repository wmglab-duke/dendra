from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from axonml.models import Population
from axonml.models.callbacks import Recorder, ThresholdCallback
from axonml.models.stim import Waveform


class Thresholder:
    r"""Compute activation thresholds for a :class:`Population` model.

    This helper class wraps a :class:`~axonml.models.Population` together with
    a :class:`~axonml.models.callbacks.ThresholdCallback` and computes, for
    each element in the population, the stimulus amplitude at which an action
    potential (or some other binary state, as reported by the callback) is first
    detected.

    The stimulus is assumed to have a fixed spatiotemporal *shape* and an
    unknown scalar *amplitude* per population element. The shape can be
    specified in one of two ways:

    * By providing ``space`` and ``time`` (functional mode), where
      ``space`` encodes the spatial distribution of extracellular potential and ``time`` is a
      :class:`Waveform` describing the temporal profile; or
    * By providing precomputed spatiotemporal ``bases`` (basis mode), in
      which case each element's extracellular potential is obtained by
      scaling the corresponding basis function with that element's amplitude.

    For each element, the thresholder maintains a lower bound ``lb`` and
    an upper bound ``ub`` on the required amplitude:

    * ``lb`` is guaranteed to be subthreshold (no action potential).
    * ``ub`` is intended to be suprathreshold (action potential present).

    Threshold computation proceeds in two stages:

    1. **Bound fixing** (:meth:`_fix_bounds`):
       The initial ``ub`` values are checked, and, if necessary, iteratively
       adjusted so that they produce a suprathreshold response for each
       element. If ``block_possible=True`` is passed to
       :meth:`calculate_thresholds`, the bound-fixing stage uses both the
       activity callback and a :class:`Recorder` trace to distinguish
       between purely subthreshold responses and putative conduction block
       (large depolarisation without a successful spike as reported by the
       callback). In that case, ``ub`` may be either increased
       (clearly subthreshold) or decreased (suspected block) using the
       ``fix_bound_up`` and ``fix_bound_down`` factors. Elements whose
       bounds cannot be repaired within ``max_tries_bound_fix`` iterations
       are marked as ignored.

    2. **Bisection search** (:meth:`calculate_thresholds`):
       Once a valid bracketing interval ``[lb, ub]`` exists, a standard
       per-element bisection is performed. On each iteration:

       * Candidate amplitudes are chosen as ``stimamp = (lb + ub) / 2``.
       * The model is simulated once with these amplitudes.
       * Elements that spike at ``stimamp`` move their upper bound down
         (``ub = stimamp``); elements that do not spike move their lower
         bound up (``lb = stimamp``).

       The loop continues independently for each element until both the
       absolute and/or relative window sizes satisfy the requested
       tolerances:

       * Absolute window: ``awindow = ub - lb``
       * Relative window: ``rwindow = (ub - lb) / ub``

       Controlled by ``atol`` and ``rtol`` via :meth:`check_tolerance`.

    All heavy computations are performed in ``torch.no_grad()`` mode and
    on the same device and dtype as the underlying model. Convenience methods
    :meth:`float` and :meth:`double` convert the thresholder, model, and
    internal buffers between single- and double-precision.

    Parameters
    ----------
    model : axonml.models.Population
        The population model to compute thresholds for.
    active : ThresholdCallback
        Callback used to decide whether an action potential was generated for
        each population element. Must implement :meth:`is_active`, which returns
        a boolean tensor indicating activity for each element.
    space : Optional[Union[npt.NDArray, Tensor]], optional
        Spatial extracellular potential distribution of the stimulus. An array of
        shape ``(model.np, model.nc)`` that encodes the coupling between each population
        element and the stimulus source(s). Used together with ``time`` when
        ``bases`` is not provided.
    time : Optional[Waveform], optional
        Temporal waveform of the stimulus. When used with ``space``, the
        effective extracellular potential passed to the model is
        ``space * amp[:, None]``, where ``amp`` is the per-element amplitude
        vector.
    bases : Optional[Union[npt.NDArray, Tensor]], optional
        Precomputed spatiotemporal bases for the extracellular potential.
        Expected shape is ``(model.np, nt, model.nc)`` (or broadcastable
        to this shape) and multiplied by the per-element amplitude vector.
        When ``bases`` is provided, ``space`` and ``time`` are ignored and
        ``chunklength`` cannot be used.
    ub : optional
        Initial upper bound(s) on threshold amplitudes. If ``None``, upper
        bounds are initialised heuristically from ``model.diameters`` using
        ``0.2 / (diameter / 5)**2`` per element. Either ``ub`` must be
        provided or ``model.diameters`` must be set.
    fix_bound_up : float, optional
        Multiplicative factor used to increase the upper bound during the
        bound-fixing stage when a response is still clearly subthreshold,
        by default ``5.0``.
    fix_bound_down : float, optional
        Multiplicative factor used to decrease the upper bound during the
        bound-fixing stage in ``block_possible`` mode when a putative block
        is detected, by default ``0.1``.
    max_tries_bound_fix : int, optional
        Maximum number of attempts to repair the upper bounds before giving
        up and marking an element as ignored, by default ``10``.
    max_tries_thresh : int, optional
        Maximum number of bisection iterations per element, by default ``25``.
    atol : optional
        Absolute tolerance for the threshold interval ``ub - lb``. The search
        stops for an element once the absolute window falls below this value
        (and the relative criterion, if given, is also satisfied).
    rtol : optional
        Relative tolerance for the threshold interval, defined as
        ``(ub - lb) / ub``. The search stops for an element once this falls
        below ``rtol`` (and the absolute criterion, if given, is also
        satisfied).
    chunklength : optional
        Chunk length (in time steps) to use when calling
        :meth:`Population.longrun` instead of :meth:`Population.run` for
        long simulations, by default ``None``.

    Examples
    --------
    A typical usage pattern is to construct a :class:`Population`, define a
    spatial field and temporal waveform, and then call
    :meth:`calculate_thresholds`:

    .. code-block:: python

        import numpy as np
        import axonml as ax

        # Assume ``model`` is an existing Population with diameters defined
        model = ...  # type: axonml.models.Population

        # Build a simple temporal waveform (e.g., a Gaussian-like pulse)
        tstop = 5.0
        dt = 0.005
        times = np.arange(0.0, tstop, dt)
        values = np.exp(-0.5 * ((times - 1.0) / 0.2) ** 2)
        stim = ax.arbitrary(values=values, tpoints=times)

        # Compute the spatial extracellular potential at each compartment
        field = ax.anisotropic_point(z=200.0, rhox=100.0, rhoz=100.0)
        ve_space = field(model)  # shape (model.np, model.nc)

        # Define an activity callback that detects spikes in a subset of nodes
        active = ax.callbacks.ActiveAL(
            threshold=20.0,
            node_check=list(range(model.nc))[::20],
            at_least=5,
        )

        # Set up the thresholder and compute thresholds to 1% relative accuracy
        thresholder = Thresholder(
            model=model,
            active=active,
            space=ve_space,
            time=stim,
            ub=1.0,
            rtol=0.01,
        )
        thr_ub, thr_lb = thresholder.calculate_thresholds(tstop=tstop, dt=dt)

        # ``thr_ub`` and ``thr_lb`` now contain per-element bounds on the
        # stimulus amplitude required to elicit an action potential.

    In more elaborate workflows, this pattern can be wrapped in an outer loop
    over random waveforms and field parameters to build distributions of
    thresholds, or applied in parallel to a detailed "model" and a
    reduced "student" model to quantify distillation error, et cetera.

    Notes
    -----
    Either ``atol`` or ``rtol`` (or both) must be provided. If neither
    tolerance can be satisfied within ``max_tries_thresh`` iterations,
    the corresponding thresholds are returned as ``NaN`` and the indices
    are recorded in :attr:`ignore`.

    When threshold computation fails during the bound-fixing stage, both
    lower and upper bounds for that element are set to one and the element
    is marked as ignored. Such elements also return ``NaN`` thresholds from
    :meth:`calculate_thresholds`.
    """

    def __init__(
        self,
        model: Population,
        active: ThresholdCallback,
        space: Optional[Union[npt.NDArray, Tensor]] = None,
        time: Optional[Waveform] = None,
        bases: Optional[Union[npt.NDArray, Tensor]] = None,
        ub=None,
        fix_bound_up=5.0,
        fix_bound_down=0.1,
        max_tries_bound_fix=10,
        max_tries_thresh=25,
        atol=None,
        rtol=None,
        chunklength=None,
    ):
        self.model = model
        self.chunklength = chunklength
        self.bases = None
        self.functional = False
        self.atol = atol
        self.rtol = rtol

        self.space = None
        self.time = None

        self.model_partition = None
        self.active_partition = None

        if bases is None and (space is None and time is None):
            raise ValueError(
                "At least one of bases or space and time must be provided."
            )

        if bases is not None:
            if chunklength is not None:
                raise ValueError(
                    "Cannot use chunklength with bases. Supply space and time instead."
                )
            bases = torch.as_tensor(bases)
            self.bases = bases.to(device=model.device(), dtype=model.dtype())
            self.check_active = self._check_active_bases
            self.functional = False
        else:
            self.space = torch.as_tensor(space).to(
                device=model.device(), dtype=model.dtype()
            )
            self.time = time.to(device=model.device(), dtype=model.dtype())
            self.check_active = self._check_active_space_time
            self.functional = True

        diams = getattr(model, "diameters", None)

        if diams is not None:
            if hasattr(diams, "__iter__"):
                if self.bases is not None:
                    assert len(diams) == self.bases.shape[1]
                elif self.functional:
                    assert len(diams) == self.space.shape[0]

            elif isinstance(diams, float):
                diams = np.atleast_1d(np.full(self.bases.shape[1], diams))

            diams = torch.as_tensor(diams)
            self.diams = diams.to(device=model.device(), dtype=model.dtype())

        else:
            self.diams = None

        if fix_bound_down >= 1 or fix_bound_up <= 0:
            raise ValueError("fix_bound_down should be < 1 and > 0.")

        if fix_bound_up <= 1:
            raise ValueError("fix_bound_up should be > 1.")

        self.ignore = None

        with torch.no_grad():
            if ub is not None:
                self.ub = torch.as_tensor(
                    ub, device=model.device(), dtype=model.dtype()
                ) * torch.ones(
                    self.model.np, device=model.device(), dtype=model.dtype()
                )
            else:
                if self.diams is None:
                    raise ValueError(
                        "Either ub must be provided or model.diameters must be set."
                    )
                self.ub = 0.2 * torch.ones_like(self.diams) / (self.diams / 5) ** 2
            self.ub_initial = self.ub.clone()
            self.lb = torch.zeros_like(self.ub)

        self.fix_bound_up = fix_bound_up
        self.fix_bound_down = fix_bound_down
        self.max_tries_bound_fix = max_tries_bound_fix
        self.max_tries_thresh = max_tries_thresh

        self.active = active
        self.threshold = active.threshold
        self.rec = Recorder(["v"], max_only=True)

    def set_partition(self, model_partition, active_partition=None):
        assert sum(model_partition) == self.model.nc, (
            "Sum of partition lengths must equal number of population compartment."
        )
        if active_partition is not None:
            assert len(model_partition) == len(active_partition), (
                "Model and active partitions must have the same length."
            )
            assert all(m >= a for m, a in zip(model_partition, active_partition)), (
                "Each model partition length must be at least as large as the corresponding active partition length."
            )
        else:
            active_partition = model_partition
        with torch.no_grad():
            self.model_partition = model_partition
            self.active_partition = active_partition
            self.ub = (
                self.ub[:, None].expand(-1, len(model_partition)).contiguous().clone()
            )
            self.lb = (
                self.lb[:, None].expand(-1, len(model_partition)).contiguous().clone()
            )
            self.ub_initial = (
                self.ub_initial[:, None]
                .expand(-1, len(model_partition))
                .contiguous()
                .clone()
            )
            self.rec.set_partition(active_partition)

    def reset_bounds(self):
        """Reset upper and lower bounds to initial values."""
        with torch.no_grad():
            self.ub = self.ub_initial.clone()
            self.lb = torch.zeros_like(self.ub)
            self.ignore = None

    def check_tolerance(
        self, awindow: Tensor, rwindow: Tensor, atol=None, rtol=None
    ) -> Tensor:
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol

        if atol is not None and rtol is not None:
            return (awindow >= atol) & (rwindow >= rtol)
        elif atol is not None:
            return awindow >= atol
        elif rtol is not None:
            return rwindow >= rtol
        else:
            raise ValueError("Either atol or rtol must be provided.")

    def ve_from_s_t(self, ve_s, ve_t, device, multicontact=False):
        ve_s = torch.as_tensor(ve_s, device=device)
        ve_t = torch.as_tensor(ve_t, device=device)

        if multicontact:
            ve_s = ve_s.expand(-1, self.model.np, -1)
            ve_t = ve_t.expand(-1, self.model.np, -1)
            einsum = op_mc
        else:
            ve_s = ve_s.expand(self.model.np, -1)
            ve_t = ve_t.expand(self.model.np, -1)
            einsum = op_sc

        return einsum(ve_s, ve_t)

    def float(self):
        self.fp32 = True
        self.model = self.model.float()
        if self.bases is not None:
            self.bases = self.bases.float()
        if self.diams is not None:
            self.diams = self.diams.float()
        if self.space is not None:
            self.space = self.space.float()
        if self.time is not None:
            self.time = self.time.float()
        self.ub = self.ub.float()
        self.ub_initial = self.ub_initial.float()
        self.lb = self.lb.float()
        return self

    def double(self):
        self.fp32 = False
        self.model = self.model.double()
        if self.bases is not None:
            self.bases = self.bases.double()
        if self.diams is not None:
            self.diams = self.diams.double()
        if self.space is not None:
            self.space = self.space.double()
        if self.time is not None:
            self.time = self.time.double()
        self.ub = self.ub.double()
        self.ub_initial = self.ub_initial.double()
        self.lb = self.lb.double()
        return self

    def _check_active_bases(self, tstop, dt, bound: Tensor):
        """Check whether stimulus amplitudes generates APs.

        Parameters
        ----------
        bound : Tensor
            Amplitudes to test.

        Returns
        -------
        Tensor
            boolean
        """
        self.active.reset()
        with torch.no_grad():
            ve = self.bases * bound[None, :, None]
            self.model.initialize()
            self.model.run(
                ve=ve,
                dt=dt,
                callbacks=[self.active],
            )
        return self.active.is_active(partition=self.active_partition)

    def _check_active_space_time(self, tstop, dt, bound: Tensor):
        """Check whether stimulus amplitudes generates APs.

        Parameters
        ----------
        bound : Tensor
            Amplitudes to test.

        Returns
        -------
        Tensor
            boolean
        """
        self.active.reset()
        with torch.no_grad():
            ve = self.space * bound
            self.model.initialize()
            if self.chunklength is not None:
                self.model.longrun(
                    extra=(ve, self.time),
                    tstop=tstop,
                    dt=dt,
                    callbacks=[self.active],
                    chunklength=self.chunklength,
                )
            else:
                self.model.run(
                    extra=(ve, self.time),
                    tstop=tstop,
                    dt=dt,
                    callbacks=[self.active],
                )
        return self.active.is_active(partition=self.active_partition)

    def check_active_with_rec(self, tstop, dt, bound: Tensor):
        self.active.reset()
        self.rec.reset()
        with torch.no_grad():
            self.model.initialize()
            if not self.functional:
                ve = self.bases * bound[None, :, None]
                self.model.run(
                    ve,
                    callbacks=[self.active, self.rec],
                    dt=dt,
                )
            else:
                if self.chunklength is not None:
                    ve = self.space * bound
                    self.model.longrun(
                        extra=(ve, self.time),
                        tstop=tstop,
                        dt=dt,
                        callbacks=[self.active, self.rec],
                        chunklength=self.chunklength,
                    )
                else:
                    ve = self.space * bound
                    self.model.run(
                        extra=(ve, self.time),
                        tstop=tstop,
                        dt=dt,
                        callbacks=[self.active, self.rec],
                    )
        return self.active.is_active(partition=self.active_partition), self.rec.stack()

    def _fix_bounds(self, tstop, dt, block_possible=True):
        """Make sure upper bound generates AP."""

        with torch.no_grad():
            tries = 0
            ub = _scale_by_partition(self.ub, self.model_partition)
            if block_possible:
                mask, rec = self.check_active_with_rec(tstop, dt, ub)
            else:
                mask = self.check_active(tstop, dt, ub)
            print("Fixing bounds.", end="")
            while torch.any(~mask):
                print(".", end="")
                ub = _scale_by_partition(self.ub, self.model_partition)
                if tries >= self.max_tries_bound_fix:
                    break
                if block_possible:
                    mask, rec = self.check_active_with_rec(tstop, dt, ub)
                else:
                    mask = self.check_active(tstop, dt, ub)
                inactive = ~mask
                if block_possible:
                    self.ub[(rec.squeeze() < self.threshold) & inactive] *= (
                        self.fix_bound_up
                    )
                    self.ub[(rec.squeeze() >= self.threshold) & inactive] *= (
                        self.fix_bound_down
                    )
                else:
                    self.ub[inactive] *= self.fix_bound_up
                tries += 1
            else:
                print("Done.")
                return
            print(
                f"Unable to fix bounds within {self.max_tries_bound_fix}"
                " iterations, ignoring some."
            )
            self.ignore = ~mask
            self.ub[self.ignore] = 1
            self.lb[self.ignore] = 1

    def calculate_thresholds(
        self, tstop, dt, block_possible=False, reset_bounds=True, atol=None, rtol=None
    ) -> Tuple[Tensor, Tensor]:
        """Calculate thresholds. If bases were provided on Thresholder
        construction, they are used and tstop is ignored.

        Parameters
        ----------
        tstop : float
            Simulation stop time in milliseconds.
        dt : float
            Simulation time step in milliseconds.
        block_possible : bool, optional
            Whether to consider conduction block when fixing upper bounds,
            by default ``False``.
        reset_bounds : bool, optional
            Whether to reset upper and lower bounds to initial values before
            calculation, by default ``True``. Set to ``False`` to continue
            a previous calculation, e.g., with tighter tolerances.
        atol : optional
            Absolute tolerance for the threshold interval ``ub - lb``. If
            ``None``, the value provided at construction is used.
        rtol : optional
            Relative tolerance for the threshold interval. If ``None``, the
            value provided at construction is used.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Upper and lower bound on thresholds.
        """
        if reset_bounds:
            self.reset_bounds()
        self._fix_bounds(tstop, dt, block_possible=block_possible)
        self.rec.reset()
        self.active.reset()

        # check no active in lb
        act = self.check_active(
            tstop, dt, _scale_by_partition(self.lb, self.model_partition)
        )
        if torch.any(act):
            raise RuntimeError(
                "Some lower bounds are active. Cannot proceed with bisection."
            )

        with torch.no_grad():
            ub = _scale_by_partition(self.ub, self.model_partition)
            lb = _scale_by_partition(self.lb, self.model_partition)

            awindow = ub - lb
            rwindow = awindow / ub
            msk = self.check_tolerance(awindow, rwindow, atol=atol, rtol=rtol)
            tries = 0

            while torch.any(msk) & (tries < self.max_tries_thresh):
                ub = _scale_by_partition(self.ub, self.model_partition)
                lb = _scale_by_partition(self.lb, self.model_partition)
                stimamp = (ub + lb) / 2
                mask = self.check_active(tstop, dt, stimamp)
                mask = _agree_dims(mask, msk)
                a_thr = msk & mask
                b_thr = msk & ~mask
                self.ub[_agree_dims(a_thr, self.ub)] = stimamp[a_thr]
                self.lb[_agree_dims(b_thr, self.lb)] = stimamp[b_thr]
                awindow = ub - lb
                rwindow = awindow / ub
                msk = self.check_tolerance(awindow, rwindow, atol=atol, rtol=rtol)
                tries += 1
            if tries >= self.max_tries_thresh:
                print("hmm")
                if self.ignore is not None:
                    self.ub[self.ignore] = torch.nan
                    self.lb[self.ignore] = torch.nan
                return self.ub.cpu(), self.lb.cpu()

            if self.ignore is not None:
                self.ub[self.ignore] = torch.nan
                self.lb[self.ignore] = torch.nan

            return self.ub.cpu(), self.lb.cpu()


@torch.jit.script
def op_mc(s: Tensor, t: Tensor) -> Tensor:
    return torch.einsum("can,cat->tan", s, t).contiguous()


@torch.jit.script
def op_sc(s: Tensor, t: Tensor) -> Tensor:
    return torch.einsum("an,at->tan", s, t).contiguous()


def _scale_by_partition(B: torch.Tensor, partition=None) -> torch.Tensor:
    """
    Reshape B by repeating each column according to the corresponding
    segment length in partition.

    Parameters
    ----------
    B : torch.Tensor
        Scaling tensor of shape (m, p). Each column provides the scaling
        factor for the corresponding segment in ``partition``.
    partition : sequence of int
        Segment lengths whose sum equals ``n``.

    Returns
    -------
    torch.Tensor
        Tensor of shape (m, n) where each segment ``j`` is scaled by
        ``B[:, j]``.

    Raises
    ------
    ValueError
        If tensor shapes are incompatible or ``partition`` is invalid.
    """

    if partition is None:
        return B[:, None]

    lengths = torch.as_tensor(partition, dtype=torch.long, device="cpu")
    if lengths.dim() != 1:
        raise ValueError("partition must be a 1D sequence of integers.")
    if lengths.numel() != B.shape[1]:
        raise ValueError(
            "partition length must match the number of columns in B; "
            f"got {lengths.numel()} and {B.shape[1]}."
        )
    if lengths.numel() == 0:
        raise ValueError("partition must be non-empty.")
    if torch.any(lengths < 0):
        raise ValueError("partition values must be non-negative.")

    weights = torch.repeat_interleave(B, lengths.tolist(), dim=1)
    return weights


def _agree_dims(mask: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Expand mask to agree with the dimensions of B.

    Parameters
    ----------
    mask : torch.Tensor
        Boolean mask tensor of shape (m,).
    B : torch.Tensor
        Target tensor of shape (m, n).

    Returns
    -------
    torch.Tensor
        Expanded boolean mask of shape (m, n).
    """
    if mask.dim() == 1 and B.dim() == 2:
        return mask[:, None].expand_as(B)
    if B.dim() == 1 and mask.dim() == 2:
        return mask.squeeze(-1)
    return mask
