from typing import Optional, Tuple, Union

import numpy as np
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

        self.field_nan_ignore = None
        self._space_no_nan = None
        self._bases_no_nan = None

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
        if self._space_no_nan is not None:
            self._space_no_nan = self._space_no_nan.float()
        if self._bases_no_nan is not None:
            self._bases_no_nan = self._bases_no_nan.float()
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
        if self._space_no_nan is not None:
            self._space_no_nan = self._space_no_nan.double()
        if self._bases_no_nan is not None:
            self._bases_no_nan = self._bases_no_nan.double()
        if self.time is not None:
            self.time = self.time.double()
        self.ub = self.ub.double()
        self.ub_initial = self.ub_initial.double()
        self.lb = self.lb.double()
        return self

    def _space_for_run(self) -> Tensor:
        """Return the spatial field used for simulations."""
        return self._space_no_nan if self._space_no_nan is not None else self.space

    def _bases_for_run(self) -> Tensor:
        """Return the basis field used for simulations."""
        return self._bases_no_nan if self._bases_no_nan is not None else self.bases

    def _scaled_bases_field(self, bound: Tensor) -> Tensor:
        """Scale basis-mode fields by unpartitioned or partition-expanded bounds."""
        bases = self._bases_for_run()
        pop_dim = self._bases_population_dim(bases)

        if pop_dim == 0:
            if bound.dim() == 1:
                scale = bound[:, None, None]
            elif bound.dim() == 2 and bound.shape[1] == 1:
                scale = bound[:, 0][:, None, None]
            elif bound.dim() == 2:
                scale = bound[:, None, :]
            else:
                raise ValueError("bound must be a 1D or 2D tensor.")
        else:
            if bound.dim() == 1:
                scale = bound[None, :, None]
            elif bound.dim() == 2 and bound.shape[1] == 1:
                scale = bound[:, 0][None, :, None]
            elif bound.dim() == 2:
                scale = bound[None, :, :]
            else:
                raise ValueError("bound must be a 1D or 2D tensor.")

        return bases * scale

    def _field_nan_mask(self) -> Tensor:
        """Return a boolean ignore mask for field entries containing NaNs.

        The returned tensor has the same shape as ``self.ub``/``self.lb``:
        ``(model.np,)`` for unpartitioned thresholding and
        ``(model.np, n_partitions)`` when a model partition is active.
        """
        if self.functional:
            return self._space_nan_mask()
        return self._bases_nan_mask()

    def _space_nan_mask(self) -> Tensor:
        space_nan = torch.isnan(self.space)
        space_nan = self._spatial_nan_as_population_by_compartment(space_nan)

        if self.model_partition is None:
            return space_nan.any(dim=1)

        return self._partition_nan_mask(space_nan, self.model_partition)

    def _bases_nan_mask(self) -> Tensor:
        bases_nan = torch.isnan(self.bases)

        if self.model_partition is None:
            return self._reduce_bases_nan(bases_nan)

        return self._reduce_bases_nan_by_partition(bases_nan, self.model_partition)

    def _spatial_nan_as_population_by_compartment(self, space_nan: Tensor) -> Tensor:
        """Broadcast a spatial NaN mask to ``(model.np, model.nc)``."""
        try:
            return torch.broadcast_to(space_nan, (self.model.np, self.model.nc))
        except RuntimeError as exc:
            raise ValueError(
                "space must be broadcastable to shape "
                f"({self.model.np}, {self.model.nc}) for partition-aware NaN checks."
            ) from exc

    def _partition_nan_mask(self, nan_by_compartment: Tensor, partition) -> Tensor:
        cols = []
        start = 0
        for length in partition:
            stop = start + length
            if length == 0:
                cols.append(
                    torch.zeros(
                        self.model.np,
                        device=nan_by_compartment.device,
                        dtype=torch.bool,
                    )
                )
            else:
                cols.append(nan_by_compartment[:, start:stop].any(dim=1))
            start = stop
        return torch.stack(cols, dim=1)

    def _bases_population_dim(self, bases: Tensor) -> Optional[int]:
        """Infer the population dimension of a basis tensor.

        ``_check_active_bases`` scales bases using ``bound[None, :, None]``, so
        dimension 1 is preferred when it has population length. Dimension 0 is
        also accepted to support the documented ``(model.np, nt, model.nc)``
        layout.
        """
        if bases.dim() >= 2 and bases.shape[1] == self.model.np:
            return 1
        if bases.dim() >= 1 and bases.shape[0] == self.model.np:
            return 0
        for dim, size in enumerate(bases.shape[:-1]):
            if size == self.model.np:
                return dim
        return None

    @staticmethod
    def _any_except(tensor: Tensor, keep_dim: int) -> Tensor:
        for dim in reversed(range(tensor.dim())):
            if dim != keep_dim:
                tensor = tensor.any(dim=dim)
        return tensor

    def _reduce_bases_nan(self, bases_nan: Tensor) -> Tensor:
        pop_dim = self._bases_population_dim(bases_nan)
        if pop_dim is None:
            return bases_nan.any().reshape(1).expand(self.model.np)
        return self._any_except(bases_nan, pop_dim)

    def _reduce_bases_nan_by_partition(self, bases_nan: Tensor, partition) -> Tensor:
        pop_dim = self._bases_population_dim(bases_nan)
        comp_dim = bases_nan.dim() - 1

        cols = []
        start = 0
        for length in partition:
            stop = start + length
            if length == 0:
                cols.append(
                    torch.zeros(
                        self.model.np, device=bases_nan.device, dtype=torch.bool
                    )
                )
                start = stop
                continue

            index = [slice(None)] * bases_nan.dim()
            index[comp_dim] = slice(start, stop)
            part_nan = bases_nan[tuple(index)]

            if pop_dim is None:
                cols.append(part_nan.any().reshape(1).expand(self.model.np))
            else:
                cols.append(self._any_except(part_nan, pop_dim))
            start = stop

        return torch.stack(cols, dim=1)

    def _ignored_like(self, target: Tensor) -> Tensor:
        if self.ignore is None:
            return torch.zeros_like(target, dtype=torch.bool)
        if (
            self.model_partition is not None
            and self.ignore.dim() == 2
            and target.dim() == 2
            and self.ignore.shape[1] == len(self.model_partition)
            and target.shape[1] == self.model.nc
        ):
            return _scale_by_partition(self.ignore, self.model_partition)
        return _agree_dims(self.ignore, target)

    def _set_ignore(self, ignore: Tensor):
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
            ve = self._space_for_run() * bound
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
                ve = self._scaled_bases_field(bound)
                self.model.run(
                    ve,
                    callbacks=[self.active, self.rec],
                    dt=dt,
                )
            else:
                if self.chunklength is not None:
                    ve = self._space_for_run() * bound
                    self.model.longrun(
                        extra=(ve, self.time),
                        tstop=tstop,
                        dt=dt,
                        callbacks=[self.active, self.rec],
                        chunklength=self.chunklength,
                    )
                else:
                    ve = self._space_for_run() * bound
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
            mask = mask | self._ignored_like(mask)
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
                mask = mask | self._ignored_like(mask)
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
            Upper and lower bound on thresholds.
        """
        if reset_bounds:
            self.reset_bounds()

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
            rwindow = awindow / self.ub
            msk = self.check_tolerance(awindow, rwindow, atol=atol, rtol=rtol)
            msk = msk & ~self._ignored_like(msk)
            tries = 0

            while bool(torch.any(msk)) and (tries < self.max_tries_thresh):
                stimamp = (self.ub + self.lb) / 2
                mask = self.check_active(
                    tstop, dt, _scale_by_partition(stimamp, self.model_partition)
                )
                mask = _agree_dims(mask, self.ub)
                a_thr = msk & mask
                b_thr = msk & ~mask
                self.ub[a_thr] = stimamp[a_thr]
                self.lb[b_thr] = stimamp[b_thr]
                awindow = self.ub - self.lb
                rwindow = awindow / self.ub
                msk = self.check_tolerance(awindow, rwindow, atol=atol, rtol=rtol)
                msk = msk & ~self._ignored_like(msk)
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

    weights = torch.repeat_interleave(B, lengths.to(device=B.device), dim=1)
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
