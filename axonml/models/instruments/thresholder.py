from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from axonml.models import Population
from axonml.models.callbacks import Recorder, ThresholdCallback
from axonml.models.stim import Waveform


class Thresholder:
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

        if atol is None and rtol is None:
            raise ValueError("Either atol or rtol must be provided.")

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
            bases = bases.permute(1, 0, 2)
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
                assert len(diams) == self.bases.shape[1]

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

        self.atol = atol
        self.rtol = rtol

        self.active = active
        self.rec = Recorder(["v"], max_only=True)

    def check_tolerance(self, awindow: Tensor, rwindow: Tensor) -> Tensor:
        if self.atol is not None and self.rtol is not None:
            return (awindow >= self.atol) & (rwindow >= self.rtol)
        elif self.atol is not None:
            return awindow >= self.atol
        elif self.rtol is not None:
            return rwindow >= self.rtol
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
        self.ub = self.ub.float()
        self.ub_initial = self.ub.clone()
        self.lb = self.lb.float()
        return self

    def double(self):
        self.fp32 = False
        self.model = self.model.double()
        if self.bases is not None:
            self.bases = self.bases.double()
        if self.diams is not None:
            self.diams = self.diams.double()
        self.ub = self.ub.double()
        self.ub_initial = self.ub.clone()
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
        return self.active.is_active()

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
            ve = self.space * bound[:, None]
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
                    space=ve,
                    time=self.time,
                    tstop=tstop,
                    dt=dt,
                    callbacks=[self.active],
                )
        return self.active.is_active()

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
                    ve = self.space * bound[:, None]
                    self.model.longrun(
                        extra=(ve, self.time),
                        tstop=tstop,
                        dt=dt,
                        callbacks=[self.active, self.rec],
                        chunklength=self.chunklength,
                    )
                else:
                    ve = self.space * bound[:, None]
                    self.model.run(
                        space=ve,
                        time=self.time,
                        tstop=tstop,
                        dt=dt,
                        callbacks=[self.active, self.rec],
                    )
        return self.active.is_active(), self.rec.stack()

    def fix_bounds(self, tstop, dt, block_possible=True):
        """Make sure upper bound generates AP."""

        with torch.no_grad():
            tries = 0
            if block_possible:
                mask, rec = self.check_active_with_rec(tstop, dt, self.ub)
            else:
                mask = self.check_active(tstop, dt, self.ub)
            print("Fixing bounds.", end="")
            while torch.any(~mask):
                print(".", end="")
                if tries >= self.max_tries_bound_fix:
                    break
                if block_possible:
                    mask, rec = self.check_active_with_rec(tstop, dt, self.ub)
                else:
                    mask = self.check_active(tstop, dt, self.ub)
                inactive = ~mask
                if block_possible:
                    self.ub[(rec[:, -1] < self.threshold) & inactive] *= (
                        self.fix_bound_up
                    )
                    self.ub[(rec[:, -1] >= self.threshold) & inactive] *= (
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
        self, tstop, dt, block_possible=False
    ) -> Tuple[Tensor, Tensor]:
        """Calculate thresholds.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Upper and lower bound on thresholds.
        """
        self.fix_bounds(tstop, dt, block_possible)
        self.rec.reset()
        self.active.reset()

        with torch.no_grad():
            ub = self.ub
            lb = self.lb

            awindow = ub - lb
            rwindow = awindow / ub
            msk = self.check_tolerance(awindow, rwindow)
            tries = 0

            while torch.any(msk) & (tries < self.max_tries_thresh):
                stimamp = (ub + lb) / 2
                mask = self.check_active(tstop, dt, stimamp)
                a_thr = msk & mask
                b_thr = msk & ~mask
                ub[a_thr] = stimamp[a_thr]
                lb[b_thr] = stimamp[b_thr]
                awindow = ub - lb
                rwindow = awindow / ub
                msk = self.check_tolerance(awindow, rwindow)
                tries += 1
            if tries >= self.max_tries_thresh:
                print("hmm")
                if self.ignore is not None:
                    ub[self.ignore] = torch.nan
                    lb[self.ignore] = torch.nan
                return ub.cpu().numpy(), lb.cpu().numpy()

            if self.ignore is not None:
                ub[self.ignore] = torch.nan
                lb[self.ignore] = torch.nan

            return ub.cpu().numpy(), lb.cpu().numpy()


@torch.jit.script
def op_mc(s: Tensor, t: Tensor) -> Tensor:
    return torch.einsum("can,cat->tan", s, t).contiguous()


@torch.jit.script
def op_sc(s: Tensor, t: Tensor) -> Tensor:
    return torch.einsum("an,at->tan", s, t).contiguous()
