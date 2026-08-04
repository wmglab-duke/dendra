from typing import Optional, Tuple, Union

import numpy.typing as npt
import torch
from torch import Tensor

from dendra.models import Population
from dendra.models.callbacks import Recorder, ThresholdCallback
from dendra.models.stim import Waveform


class Thresholder:
    r"""Compute activation thresholds for a :class:`Population` model.

    This helper class wraps a :class:`~dendra.models.Population` together with
    a :class:`~dendra.models.callbacks.ThresholdCallback` and computes, for
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
    model : dendra.models.Population
        The population model to compute thresholds for.
    active : ThresholdCallback
        Callback used to decide whether an action potential was generated for
        each population element. Must implement :meth:`is_active`, which returns
        a boolean tensor indicating activity for each element.
    space : Optional[Union[npt.NDArray, Tensor]], optional
        Spatial extracellular potential distribution of the stimulus. It must
        be broadcastable to ``model.shape == (*batch, model.np, model.nc)``.
        Used together with ``time`` when ``bases`` is not provided.
    time : Optional[Waveform], optional
        Temporal waveform of the stimulus. When used with ``space``, the
        effective extracellular potential passed to the model is ``space * amp``,
        with a final singleton compartment axis added to ``amp``. Thus one
        independently searched amplitude is maintained for every entry of
        ``model.shape[:-1]`` (all batch and population lanes).
    bases : Optional[Union[npt.NDArray, Tensor]], optional
        Precomputed spatiotemporal bases for the extracellular potential. The
        canonical layout is time-first, ``(nt, *model.shape)``; payload axes
        after time may broadcast to ``model.shape``. The historical unbatched
        layout ``(model.np, nt, model.nc)`` is accepted when the canonical
        interpretation is not possible. Ambiguous layouts are rejected.
        When ``bases`` is provided, ``space`` and ``time`` are ignored and
        ``chunklength`` cannot be used.
    ub : optional
        Initial upper bound(s) on threshold amplitudes, broadcastable to
        ``model.shape[:-1]``. If ``None``, upper
        bounds are initialised heuristically from ``model.diameters`` using
        ``0.2 / (diameter / 5)**2`` per element. Either ``ub`` must be
        provided or ``model.diameters`` must be set.
    lb : optional
        Initial lower bound(s) on threshold amplitudes, broadcastable to
        ``model.shape[:-1]``. If ``None``, lower bounds are initialised to zero.
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
        import dendra as dn

        # Assume ``model`` is an existing Population with diameters defined
        model = ...  # type: dendra.models.Population

        # Build a simple temporal waveform (e.g., a Gaussian-like pulse)
        tstop = 5.0
        dt = 0.005
        times = np.arange(0.0, tstop, dt)
        values = np.exp(-0.5 * ((times - 1.0) / 0.2) ** 2)
        stim = dn.arbitrary(values=values, tpoints=times)

        # Compute the spatial extracellular potential at each compartment
        field = dn.anisotropic_point(z=200.0, rhox=100.0, rhoz=100.0)
        ve_space = field(model)  # shape (model.np, model.nc)

        # Define an activity callback that detects spikes in a subset of nodes
        active = dn.callbacks.ActiveAL(
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

    Before thresholding starts, the extracellular field is scanned for
    ``NaN`` values. Without a model partition, any population element whose
    field contains a ``NaN`` is ignored. With a partition, the check is
    performed independently for each population-element/partition pair, so
    only affected partitions are ignored. Ignored field entries are replaced
    with zero for simulation purposes so that ``NaN`` values do not propagate
    through the model, and ignored thresholds are returned as ``NaN``.

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
        lb=None,
        fix_bound_up=2.0,
        fix_bound_down=0.1,
        max_tries_bound_fix=10,
        max_tries_thresh=25,
        atol=None,
        rtol=None,
        chunklength=None,
        mode="arithmetic",
    ):
        self.model = model
        self.model_shape = tuple(
            int(size) for size in getattr(model, "shape", (model.np, model.nc))
        )
        if len(self.model_shape) < 2:
            raise ValueError(
                "model.shape must include population and compartment axes; "
                f"got {self.model_shape}."
            )
        if self.model_shape[-2:] != (int(model.np), int(model.nc)):
            raise ValueError(
                "The final axes of model.shape must be (model.np, model.nc); "
                f"got model.shape={self.model_shape}, model.np={model.np}, "
                f"and model.nc={model.nc}."
            )
        self.lane_shape = self.model_shape[:-1]
        self.chunklength = chunklength
        self.bases = None
        self.functional = False
        self.atol = atol
        self.rtol = rtol

        self.space = None
        self.time = None

        self.model_partition = None
        self.active_partition = None

        self.field_nan_ignore = None
        self._space_no_nan = None
        self._bases_no_nan = None

        valid_modes = ["arithmetic", "geometric"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Valid options are {valid_modes}.")
        self.mode = mode

        if bases is None and (space is None or time is None):
            raise ValueError(
                "At least one of bases or space and time must be provided."
            )

        if bases is not None:
            if chunklength is not None:
                raise ValueError(
                    "Cannot use chunklength with bases. Supply space and time instead."
                )
            bases = torch.as_tensor(bases, device=model.device(), dtype=model.dtype())
            self.bases = self._normalize_bases(bases)
            self.check_active = self._check_active_bases
            self.functional = False
        else:
            self.space = self._normalize_space(
                torch.as_tensor(space, device=model.device(), dtype=model.dtype())
            )
            self.time = time.to(device=model.device(), dtype=model.dtype())
            self.check_active = self._check_active_space_time
            self.functional = True

        diams = getattr(model, "diameters", None)

        if diams is not None:
            self.diams = self._normalize_lane_value(diams, "model.diameters")

        else:
            self.diams = None

        if fix_bound_down <= 0 or fix_bound_down >= 1:
            raise ValueError("fix_bound_down should be < 1 and > 0.")

        if fix_bound_up <= 1:
            raise ValueError("fix_bound_up should be > 1.")

        self.ignore = None

        with torch.no_grad():
            if ub is not None:
                self.ub = self._normalize_lane_value(ub, "ub")
            else:
                if self.diams is None:
                    raise ValueError(
                        "Either ub must be provided or model.diameters must be set."
                    )
                self.ub = 0.2 * torch.ones_like(self.diams) / (self.diams / 5) ** 2
            self.ub_initial = self.ub.clone()
            if lb is not None:
                self.lb = self._normalize_lane_value(lb, "lb")
            else:
                self.lb = torch.zeros_like(self.ub)
            self.lb_initial = self.lb.clone()

        if self.mode == "geometric":
            if torch.any(self.lb <= 0) or torch.any(self.ub <= 0):
                raise ValueError(
                    "geometric thresholding requires strictly positive lb and ub; "
                    "provide a positive lower bound instead of the default zero."
                )

        self.fix_bound_up = fix_bound_up
        self.fix_bound_down = fix_bound_down
        self.max_tries_bound_fix = max_tries_bound_fix
        self.max_tries_thresh = max_tries_thresh

        self.active = active
        self.threshold = active.threshold
        self.rec = Recorder(["v"], max_only=True)

    def _normalize_lane_value(self, value, name: str) -> Tensor:
        """Broadcast one threshold value to ``model.shape[:-1]``.

        Bounds use ordinary trailing PyTorch broadcasting. In particular, a
        value shaped ``(model.np,)`` is shared by every explicit batch, while
        per-batch values should include a final singleton population axis.
        """
        value = torch.as_tensor(
            value, device=self.model.device(), dtype=self.model.dtype()
        )
        try:
            return torch.broadcast_to(value, self.lane_shape).clone()
        except RuntimeError as exc:
            raise ValueError(
                f"{name} shape {tuple(value.shape)} is not broadcastable to "
                f"model lane shape {self.lane_shape} (= model.shape[:-1]). "
                "Use explicit singleton axes to distinguish batch and "
                "population dimensions."
            ) from exc

    def _normalize_space(self, space: Tensor) -> Tensor:
        """Broadcast a functional spatial field to the full model shape."""
        try:
            return torch.broadcast_to(space, self.model_shape)
        except RuntimeError as exc:
            raise ValueError(
                f"space shape {tuple(space.shape)} is not broadcastable to "
                f"model shape {self.model_shape}. Use explicit singleton axes "
                "to distinguish batch, population, and compartment dimensions."
            ) from exc

    def _time_first_bases_candidate(self, bases: Tensor) -> Optional[Tensor]:
        """Return the canonical time-first broadcast view, if one exists."""
        if bases.dim() < 1:
            return None
        payload_shape = tuple(bases.shape[1:])
        if len(payload_shape) > len(self.model_shape):
            return None
        shaped = bases.reshape(
            int(bases.shape[0]),
            *(1,) * (len(self.model_shape) - len(payload_shape)),
            *payload_shape,
        )
        try:
            return torch.broadcast_to(shaped, (int(bases.shape[0]), *self.model_shape))
        except RuntimeError:
            return None

    def _legacy_bases_candidate(self, bases: Tensor) -> Optional[Tensor]:
        """Normalize historical ``(model.np, nt, model.nc)`` bases."""
        if bases.dim() != 3:
            return None
        # The historical contract was explicit population-first, rather than
        # population-broadcastable. Requiring the exact population length also
        # keeps canonical one-step bases shaped ``(1, N, C)`` unambiguous.
        if int(bases.shape[0]) != int(self.model.np):
            return None
        if int(bases.shape[-1]) not in (1, int(self.model.nc)):
            return None
        return self._time_first_bases_candidate(bases.movedim(1, 0))

    def _normalize_bases(self, bases: Tensor) -> Tensor:
        """Normalize bases to canonical ``(time, *model.shape)`` layout."""
        if bases.dim() == 0:
            raise ValueError(
                "bases must have a leading time axis; got a scalar tensor."
            )

        canonical = self._time_first_bases_candidate(bases)
        legacy = self._legacy_bases_candidate(bases)
        if canonical is not None and legacy is not None:
            raise ValueError(
                f"bases shape {tuple(bases.shape)} is ambiguous: it is valid as "
                "both canonical (time, *model.shape) and historical "
                "(model.np, time, model.nc) layout. Supply canonical bases with "
                "an explicit singleton batch axis, or choose a time length that "
                "disambiguates the axes."
            )
        if canonical is not None:
            return canonical
        if legacy is not None:
            return legacy
        raise ValueError(
            f"bases shape {tuple(bases.shape)} cannot be normalized to canonical "
            f"shape (time, *model.shape) = (time, {', '.join(map(str, self.model_shape))}). "
            "Payload axes after time use ordinary trailing broadcasting. The "
            "historical unbatched (model.np, time, model.nc) layout is accepted "
            "only when unambiguous."
        )

    def set_partition(self, model_partition, active_partition=None):
        if self.model_partition is not None:
            raise RuntimeError("A model partition has already been configured.")
        model_partition = tuple(int(length) for length in model_partition)
        if not model_partition or any(length <= 0 for length in model_partition):
            raise ValueError("model_partition must contain positive lengths.")
        if sum(model_partition) != self.model.nc:
            raise ValueError(
                "Sum of partition lengths must equal number of population compartments."
            )
        if active_partition is not None:
            active_partition = tuple(int(length) for length in active_partition)
            if len(model_partition) != len(active_partition):
                raise ValueError(
                    "Model and active partitions must have the same length."
                )
            if any(length <= 0 for length in active_partition):
                raise ValueError("active_partition must contain positive lengths.")
            if not all(m >= a for m, a in zip(model_partition, active_partition)):
                raise ValueError(
                    "Each model partition length must be at least as large as "
                    "the corresponding active partition length."
                )
        else:
            active_partition = model_partition
        with torch.no_grad():
            self.model_partition = model_partition
            self.active_partition = active_partition
            n_partitions = len(model_partition)

            def expand_partition(value):
                return value.unsqueeze(-1).expand(*value.shape, n_partitions).clone()

            self.ub = expand_partition(self.ub)
            self.lb = expand_partition(self.lb)
            self.ub_initial = expand_partition(self.ub_initial)
            self.lb_initial = expand_partition(self.lb_initial)
            # Recorder observes the complete model, not the callback's selected
            # nodes, so its partition is defined in model-compartment space.
            self.rec.set_partition(model_partition)

    def reset_bounds(self):
        """Reset upper and lower bounds to initial values."""
        with torch.no_grad():
            self.ub = self.ub_initial.clone()
            self.lb = self.lb_initial.clone()
            self.ignore = None

    def check_tolerance(
        self, awindow: Tensor, rwindow: Tensor, atol=None, rtol=None
    ) -> Tensor:
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol

        if atol is not None and rtol is not None:
            # Continue while either requested accuracy has not yet been met.
            return (awindow >= atol) | (rwindow >= rtol)
        elif atol is not None:
            return awindow >= atol
        elif rtol is not None:
            return rwindow >= rtol
        else:
            raise ValueError("Either atol or rtol must be provided.")

    def _relative_window(self, awindow: Tensor) -> Tensor:
        """Return the documented relative interval width ``(ub - lb) / ub``."""
        return awindow / self.ub

    def ve_from_s_t(self, ve_s, ve_t, device, multicontact=False):
        ve_s = torch.as_tensor(ve_s, device=device, dtype=self.model.dtype())
        ve_t = torch.as_tensor(ve_t, device=device, dtype=self.model.dtype())

        if not multicontact:
            ve_s = self._normalize_space(ve_s)
            if ve_t.dim() == 0:
                raise ValueError("ve_t must have a trailing time axis.")
            time_shape = (*self.lane_shape, int(ve_t.shape[-1]))
            try:
                ve_t = torch.broadcast_to(ve_t, time_shape)
            except RuntimeError as exc:
                raise ValueError(
                    f"ve_t shape {tuple(ve_t.shape)} is not broadcastable to "
                    f"(*model.shape[:-1], time) = {time_shape}."
                ) from exc
            return op_sc(ve_s, ve_t)

        if ve_s.dim() < 1 or ve_t.dim() < 2:
            raise ValueError(
                "multicontact ve_s and ve_t need leading contact axes, and ve_t "
                "also needs a trailing time axis."
            )
        n_contacts = int(ve_s.shape[0])
        if int(ve_t.shape[0]) != n_contacts:
            raise ValueError("ve_s and ve_t must have the same number of contacts.")

        spatial_payload = tuple(ve_s.shape[1:])
        if len(spatial_payload) > len(self.model_shape):
            raise ValueError("multicontact ve_s has too many spatial axes.")
        ve_s = ve_s.reshape(
            n_contacts,
            *(1,) * (len(self.model_shape) - len(spatial_payload)),
            *spatial_payload,
        )
        temporal_payload = tuple(ve_t.shape[1:-1])
        if len(temporal_payload) > len(self.lane_shape):
            raise ValueError("multicontact ve_t has too many lane axes.")
        ve_t = ve_t.reshape(
            n_contacts,
            *(1,) * (len(self.lane_shape) - len(temporal_payload)),
            *temporal_payload,
            int(ve_t.shape[-1]),
        )
        try:
            ve_s = torch.broadcast_to(ve_s, (n_contacts, *self.model_shape))
            ve_t = torch.broadcast_to(
                ve_t, (n_contacts, *self.lane_shape, int(ve_t.shape[-1]))
            )
        except RuntimeError as exc:
            raise ValueError(
                "multicontact fields are not broadcastable to contact-first "
                "model spatial and temporal shapes."
            ) from exc
        return op_mc(ve_s, ve_t)

    def float(self):
        self.fp32 = True
        self.model = self.model.float()
        if self.bases is not None:
            self.bases = self.bases.float()
        if self.diams is not None:
            self.diams = self.diams.float()
        if self.space is not None:
            self.space = self.space.float()
        if self._space_no_nan is not None:
            self._space_no_nan = self._space_no_nan.float()
        if self._bases_no_nan is not None:
            self._bases_no_nan = self._bases_no_nan.float()
        if self.time is not None:
            self.time = self.time.float()
        self.ub = self.ub.float()
        self.ub_initial = self.ub_initial.float()
        self.lb = self.lb.float()
        self.lb_initial = self.lb_initial.float()
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
        if self._space_no_nan is not None:
            self._space_no_nan = self._space_no_nan.double()
        if self._bases_no_nan is not None:
            self._bases_no_nan = self._bases_no_nan.double()
        if self.time is not None:
            self.time = self.time.double()
        self.ub = self.ub.double()
        self.ub_initial = self.ub_initial.double()
        self.lb = self.lb.double()
        self.lb_initial = self.lb_initial.double()
        return self

    def _space_for_run(self) -> Tensor:
        """Return the spatial field used for simulations."""
        return self._space_no_nan if self._space_no_nan is not None else self.space

    def _bases_for_run(self) -> Tensor:
        """Return the basis field used for simulations."""
        return self._bases_no_nan if self._bases_no_nan is not None else self.bases

    def _scaled_bases_field(self, bound: Tensor) -> Tensor:
        """Scale canonical bases while preserving time and every model axis."""
        scale = self._bound_as_spatial(bound)
        return self._bases_for_run() * scale.unsqueeze(0)

    def _bound_as_spatial(self, bound: Tensor) -> Tensor:
        """Broadcast lane or compartment-expanded amplitudes to model shape."""
        bound = torch.as_tensor(
            bound, device=self.model.device(), dtype=self.model.dtype()
        )
        if tuple(bound.shape) == self.lane_shape:
            bound = bound.unsqueeze(-1)
        try:
            return torch.broadcast_to(bound, self.model_shape)
        except RuntimeError as exc:
            raise ValueError(
                f"bound shape {tuple(bound.shape)} is not broadcastable to model "
                f"shape {self.model_shape}. Expected lane bounds shaped "
                f"{self.lane_shape}, or compartment-expanded bounds."
            ) from exc

    def _field_nan_mask(self) -> Tensor:
        """Return a boolean ignore mask for field entries containing NaNs.

        The returned tensor has the same shape as ``self.ub``/``self.lb``:
        ``model.shape[:-1]`` for unpartitioned thresholding and
        ``(*model.shape[:-1], n_partitions)`` when a partition is active.
        """
        if self.functional:
            return self._space_nan_mask()
        return self._bases_nan_mask()

    def _space_nan_mask(self) -> Tensor:
        space_nan = torch.isnan(self.space)
        space_nan = self._spatial_nan_as_population_by_compartment(space_nan)

        if self.model_partition is None:
            return space_nan.any(dim=-1)

        return self._partition_nan_mask(space_nan, self.model_partition)

    def _bases_nan_mask(self) -> Tensor:
        bases_nan = torch.isnan(self.bases)

        if self.model_partition is None:
            return self._reduce_bases_nan(bases_nan)

        return self._reduce_bases_nan_by_partition(bases_nan, self.model_partition)

    def _spatial_nan_as_population_by_compartment(self, space_nan: Tensor) -> Tensor:
        """Broadcast a spatial NaN mask to the full model shape."""
        try:
            return torch.broadcast_to(space_nan, self.model_shape)
        except RuntimeError as exc:
            raise ValueError(
                f"space must be broadcastable to model shape {self.model_shape} "
                "for partition-aware NaN checks."
            ) from exc

    def _partition_nan_mask(self, nan_by_compartment: Tensor, partition) -> Tensor:
        cols = []
        start = 0
        for length in partition:
            stop = start + length
            cols.append(nan_by_compartment[..., start:stop].any(dim=-1))
            start = stop
        return torch.stack(cols, dim=-1)

    def _reduce_bases_nan(self, bases_nan: Tensor) -> Tensor:
        expected = (int(bases_nan.shape[0]), *self.model_shape)
        if tuple(bases_nan.shape) != expected:
            raise ValueError(
                "bases NaN reduction requires canonical time-first bases shaped "
                f"{expected}; got {tuple(bases_nan.shape)}."
            )
        return bases_nan.any(dim=0).any(dim=-1)

    def _reduce_bases_nan_by_partition(self, bases_nan: Tensor, partition) -> Tensor:
        expected = (int(bases_nan.shape[0]), *self.model_shape)
        if tuple(bases_nan.shape) != expected:
            raise ValueError(
                "partitioned bases NaN reduction requires canonical time-first "
                f"bases shaped {expected}; got {tuple(bases_nan.shape)}."
            )
        cols = []
        start = 0
        for length in partition:
            stop = start + length
            part_nan = bases_nan[..., start:stop]
            cols.append(part_nan.any(dim=0).any(dim=-1))
            start = stop
        return torch.stack(cols, dim=-1)

    def _normalize_activity(self, activity: Tensor) -> Tensor:
        """Validate the callback's one-result-per-search-lane contract."""
        if activity is None:
            raise TypeError(
                "ThresholdCallback.is_active must return a boolean tensor, not None."
            )
        activity = torch.as_tensor(activity, device=self.ub.device)
        if activity.dtype != torch.bool:
            raise TypeError(
                "ThresholdCallback.is_active must return a boolean tensor; "
                f"got dtype {activity.dtype}."
            )
        expected = tuple(self.ub.shape)
        if tuple(activity.shape) != expected:
            partition_note = (
                ""
                if self.model_partition is None
                else f" (including {len(self.model_partition)} partitions)"
            )
            raise ValueError(
                "ThresholdCallback.is_active returned shape "
                f"{tuple(activity.shape)}, but Thresholder requires {expected}"
                f"{partition_note}. Callbacks must preserve every batch and "
                "population axis; implicit broadcasting of activity is not allowed."
            )
        return activity

    def _normalize_recorder_metric(self, recorded: Tensor) -> Tensor:
        """Remove Recorder-only singleton axes without squeezing model lanes."""
        recorded = torch.as_tensor(recorded, device=self.ub.device)
        expected = tuple(self.ub.shape)
        if tuple(recorded.shape) == expected:
            return recorded
        if self.model_partition is None and tuple(recorded.shape) == (*expected, 1):
            return recorded.squeeze(-1)
        raise ValueError(
            f"Recorder block metric shape {tuple(recorded.shape)} cannot be "
            f"mapped to threshold bound shape {expected}. Expected exactly "
            f"{expected}"
            + (
                f" or {(*expected, 1)} from an unpartitioned Recorder."
                if self.model_partition is None
                else "."
            )
        )

    def _ignored_like(self, target: Tensor) -> Tensor:
        if self.ignore is None:
            return torch.zeros_like(target, dtype=torch.bool)
        if (
            self.model_partition is not None
            and tuple(self.ignore.shape)
            == (*self.lane_shape, len(self.model_partition))
            and tuple(target.shape) == self.model_shape
        ):
            return _scale_by_partition(self.ignore, self.model_partition)
        ignored = _agree_dims(self.ignore, target)
        if tuple(ignored.shape) != tuple(target.shape):
            raise ValueError(
                f"ignore shape {tuple(self.ignore.shape)} cannot be mapped to "
                f"target shape {tuple(target.shape)}."
            )
        return ignored

    def _set_ignore(self, ignore: Tensor):
        ignore = self._normalize_activity(ignore)
        if self.ignore is None:
            self.ignore = ignore.clone()
        else:
            self.ignore = self.ignore | ignore

    def _apply_field_nan_ignore(self):
        """Detect NaNs in the extracellular field and mark affected entries.

        This is called before the threshold search runs. Any detected NaNs are
        represented in ``self.ignore`` and a NaN-free copy of the field is used
        during simulations so ignored entries cannot poison the model state.
        """
        ignore = self._field_nan_mask().to(device=self.ub.device, dtype=torch.bool)
        if ignore.shape != self.ub.shape:
            raise RuntimeError(
                "Internal NaN ignore mask shape does not match threshold bounds: "
                f"got {tuple(ignore.shape)} and {tuple(self.ub.shape)}."
            )

        self.field_nan_ignore = ignore

        if not torch.any(ignore):
            self._space_no_nan = None
            self._bases_no_nan = None
            return

        if self.functional:
            self._space_no_nan = torch.nan_to_num(self.space, nan=0.0)
            self._bases_no_nan = None
        else:
            self._bases_no_nan = torch.nan_to_num(self.bases, nan=0.0)
            self._space_no_nan = None

        self._set_ignore(ignore)
        self.ub[ignore] = 0
        self.lb[ignore] = 0

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
            ve = self._scaled_bases_field(bound)
            self.model.initialize()
            self.model.run(
                ve=ve,
                dt=dt,
                callbacks=[self.active],
            )
        return self._normalize_activity(
            self.active.is_active(partition=self.active_partition)
        )

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
            ve = self._space_for_run() * self._bound_as_spatial(bound)
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
        return self._normalize_activity(
            self.active.is_active(partition=self.active_partition)
        )

    def check_active_with_rec(self, tstop, dt, bound: Tensor):
        self.active.reset()
        self.rec.reset()
        with torch.no_grad():
            self.model.initialize()
            if not self.functional:
                ve = self._scaled_bases_field(bound)
                self.model.run(
                    ve,
                    callbacks=[self.active, self.rec],
                    dt=dt,
                )
            else:
                if self.chunklength is not None:
                    ve = self._space_for_run() * self._bound_as_spatial(bound)
                    self.model.longrun(
                        extra=(ve, self.time),
                        tstop=tstop,
                        dt=dt,
                        callbacks=[self.active, self.rec],
                        chunklength=self.chunklength,
                    )
                else:
                    ve = self._space_for_run() * self._bound_as_spatial(bound)
                    self.model.run(
                        extra=(ve, self.time),
                        tstop=tstop,
                        dt=dt,
                        callbacks=[self.active, self.rec],
                    )
        activity = self._normalize_activity(
            self.active.is_active(partition=self.active_partition)
        )
        # ``stack("v")`` is time-first and avoids the additional state axis
        # introduced by ``stack()``. Recorder already reduces compartments (or
        # each compartment partition) per sample; reduce only the time axis here.
        recorded = self.rec.stack("v").amax(dim=0)
        return activity, self._normalize_recorder_metric(recorded)

    def _fix_bounds(self, tstop, dt, block_possible=True):
        """Make sure upper bound generates AP."""

        with torch.no_grad():
            tries = 0
            print("Fixing bounds.", end="")
            while True:
                ub = _scale_by_partition(self.ub, self.model_partition)
                if block_possible:
                    mask, rec = self.check_active_with_rec(tstop, dt, ub)
                    rec = self._normalize_recorder_metric(rec)
                else:
                    mask = self.check_active(tstop, dt, ub)
                mask = self._normalize_activity(mask)
                mask = mask | self._ignored_like(mask)
                if not torch.any(~mask):
                    print("Done.")
                    return
                if tries >= self.max_tries_bound_fix:
                    break

                print(".", end="")
                inactive = ~mask
                if block_possible:
                    self.ub[(rec < self.threshold) & inactive] *= self.fix_bound_up
                    self.ub[(rec >= self.threshold) & inactive] *= self.fix_bound_down
                else:
                    self.ub[inactive] *= self.fix_bound_up
                tries += 1
            print(
                f"Unable to fix bounds within {self.max_tries_bound_fix}"
                " iterations, ignoring some."
            )
            failed = ~mask
            self._set_ignore(failed)
            self.ub[failed] = 1
            self.lb[failed] = 1

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
            Upper and lower threshold bounds on CPU. Each has shape
            ``model.shape[:-1]`` without a model partition, or
            ``(*model.shape[:-1], n_partitions)`` after :meth:`set_partition`.
        """
        if reset_bounds:
            self.reset_bounds()

        # Resolve effective tolerances before running any simulations.
        effective_atol = self.atol if atol is None else atol
        effective_rtol = self.rtol if rtol is None else rtol
        if effective_atol is None and effective_rtol is None:
            raise ValueError("Either atol or rtol must be provided.")

        with torch.no_grad():
            self._apply_field_nan_ignore()

        self._fix_bounds(tstop, dt, block_possible=block_possible)
        self.rec.reset()
        self.active.reset()

        # check no active in lb
        act = self.check_active(
            tstop, dt, _scale_by_partition(self.lb, self.model_partition)
        )
        act = act & ~self._ignored_like(act)
        if torch.any(act):
            raise RuntimeError(
                "Some lower bounds are active. Cannot proceed with bisection."
            )

        with torch.no_grad():
            awindow = self.ub - self.lb
            rwindow = self._relative_window(awindow)
            msk = self.check_tolerance(awindow, rwindow, atol=atol, rtol=rtol)
            msk = msk & ~self._ignored_like(msk)
            tries = 0

            while bool(torch.any(msk)) and (tries < self.max_tries_thresh):
                stimamp = self.calc_stimamp()
                mask = self.check_active(
                    tstop, dt, _scale_by_partition(stimamp, self.model_partition)
                )
                mask = _agree_dims(mask, self.ub)
                a_thr = msk & mask
                b_thr = msk & ~mask
                self.ub[a_thr] = stimamp[a_thr]
                self.lb[b_thr] = stimamp[b_thr]
                awindow = self.ub - self.lb
                rwindow = self._relative_window(awindow)
                msk = self.check_tolerance(awindow, rwindow, atol=atol, rtol=rtol)
                msk = msk & ~self._ignored_like(msk)
                tries += 1

            # Entries still outside tolerance exhausted their independent
            # iteration budget and must not be reported as finite thresholds.
            if torch.any(msk):
                self._set_ignore(msk)

            if self.ignore is not None:
                self.ub[self.ignore] = torch.nan
                self.lb[self.ignore] = torch.nan

            return self.ub.cpu(), self.lb.cpu()

    def calc_stimamp(self):
        if self.mode == "arithmetic":
            return (self.ub + self.lb) / 2
        elif self.mode == "geometric":
            valid = ~self._ignored_like(self.lb)
            if torch.any((self.lb <= 0) & valid) or torch.any((self.ub <= 0) & valid):
                raise ValueError(
                    "geometric thresholding requires strictly positive lower "
                    "and upper bounds for every non-ignored lane."
                )
            return torch.sqrt(self.ub * self.lb)
        else:
            raise ValueError(f"Unknown mode '{self.mode}'.")


def op_mc(s: Tensor, t: Tensor) -> Tensor:
    """Combine contact-first spatial and time-last temporal fields.

    ``s`` has shape ``(contacts, *lanes, compartments)`` and ``t`` has
    shape ``(contacts, *lanes, time)``. Leading lane axes use normal PyTorch
    broadcasting and are preserved in the time-first result.
    """
    if s.dim() < 2 or t.dim() < 2:
        raise ValueError("s and t must include contact and trailing feature axes.")
    if s.shape[0] != t.shape[0]:
        raise ValueError("s and t must have the same number of contacts.")
    try:
        lane_shape = torch.broadcast_shapes(s.shape[1:-1], t.shape[1:-1])
        s = torch.broadcast_to(s, (s.shape[0], *lane_shape, s.shape[-1]))
        t = torch.broadcast_to(t, (t.shape[0], *lane_shape, t.shape[-1]))
    except RuntimeError as exc:
        raise ValueError("s and t lane axes are not broadcastable.") from exc
    combined = (s.unsqueeze(-2) * t.unsqueeze(-1)).sum(dim=0)
    return combined.movedim(-2, 0).contiguous()


def op_sc(s: Tensor, t: Tensor) -> Tensor:
    """Combine ``(*lanes, compartments)`` and ``(*lanes, time)`` fields."""
    if s.dim() < 1 or t.dim() < 1:
        raise ValueError("s and t must have trailing compartment/time axes.")
    try:
        lane_shape = torch.broadcast_shapes(s.shape[:-1], t.shape[:-1])
        s = torch.broadcast_to(s, (*lane_shape, s.shape[-1]))
        t = torch.broadcast_to(t, (*lane_shape, t.shape[-1]))
    except RuntimeError as exc:
        raise ValueError("s and t lane axes are not broadcastable.") from exc
    return (s.unsqueeze(-2) * t.unsqueeze(-1)).movedim(-2, 0).contiguous()


def _scale_by_partition(B: torch.Tensor, partition=None) -> torch.Tensor:
    """
    Reshape B by repeating each column according to the corresponding
    segment length in partition.

    Parameters
    ----------
    B : torch.Tensor
        Scaling tensor of shape ``(*lanes, p)``. The final axis provides the
        scaling factor for each segment in ``partition``.
    partition : sequence of int
        Segment lengths whose sum equals ``n``.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(*lanes, n)`` where each segment ``j`` is scaled by
        ``B[..., j]``. Without a partition, returns ``B[..., None]``.

    Raises
    ------
    ValueError
        If tensor shapes are incompatible or ``partition`` is invalid.
    """

    if partition is None:
        return B.unsqueeze(-1)

    if B.dim() < 1:
        raise ValueError("partitioned scaling requires a final partition axis.")

    lengths = torch.as_tensor(partition, dtype=torch.long, device="cpu")
    if lengths.dim() != 1:
        raise ValueError("partition must be a 1D sequence of integers.")
    if lengths.numel() != B.shape[-1]:
        raise ValueError(
            "partition length must match the final dimension of B; "
            f"got {lengths.numel()} and {B.shape[-1]}."
        )
    if lengths.numel() == 0:
        raise ValueError("partition must be non-empty.")
    if torch.any(lengths < 0):
        raise ValueError("partition values must be non-negative.")

    weights = torch.repeat_interleave(B, lengths.to(device=B.device), dim=-1)
    return weights


def _agree_dims(mask: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Expand mask to agree with the dimensions of B.

    Parameters
    ----------
    mask : torch.Tensor
        Boolean mask tensor with one fewer trailing axis than ``B``, or a
        tensor already broadcastable to ``B``.
    B : torch.Tensor
        Target tensor of shape (m, n).

    Returns
    -------
    torch.Tensor
        Expanded boolean mask of shape (m, n).
    """
    if tuple(mask.shape) == tuple(B.shape):
        return mask
    if tuple(mask.shape) == tuple(B.shape[:-1]):
        return mask.unsqueeze(-1).expand_as(B)
    if tuple(B.shape) == tuple(mask.shape[:-1]) and mask.shape[-1] == 1:
        return mask.squeeze(-1)
    try:
        return torch.broadcast_to(mask, B.shape)
    except RuntimeError:
        return mask
