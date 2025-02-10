from typing import List, Tuple, Optional, Union

import numpy as np
import numpy.typing as npt

import torch
from torch import Tensor

from axonml.models import Axon
from axonml.models.callbacks import Recorder, ThresholdCallback


class Thresholder:
    def __init__(
        self,
        model: Axon,
        active: ThresholdCallback,
        space: Optional[Union[npt.NDArray, Tensor]] = None,
        time: Optional[Union[npt.NDArray, Tensor]] = None,
        bases: Optional[Union[npt.NDArray, Tensor]] = None,
        diams: Optional[Union[npt.NDArray, Tensor, List]] = None,
        ub=None,
        fix_bound_up=5.0,
        fix_bound_down=0.1,
        max_tries_bound_fix=10,
        max_tries_thresh=25,
        resolution=0.01,
        multicontact=False,
        chunks=None,
    ):
        self.model = model
        self.chunks = chunks
        self.bases = None

        if bases is None and (space is None and time is None):
            raise ValueError(
                "At least one of bases or space and time must be provided."
            )

        if bases is not None:
            bases = torch.as_tensor(bases)
            bases = bases.permute(1, 0, 2).unsqueeze(2)
            self.bases = bases.to(model.device())
        else:
            if chunks is None:
                bases = self.ve_from_s_t(
                    space, time, self.model.device(), multicontact=multicontact
                )
                self.bases = bases
            else:
                self.space = torch.as_tensor(space).to(model.device())
                self.time = torch.as_tensor(time).to(model.device())

        if diams is not None:
            if hasattr(diams, "__iter__"):
                assert len(diams) == self.bases.shape[1]

            elif isinstance(diams, float):
                diams = np.atleast_1d(np.full(self.bases.shape[1], diams))

            diams = torch.as_tensor(diams)
            self.diams = diams.to(model.device())

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
                    self.model.n_ax, device=model.device(), dtype=model.dtype()
                )
            else:
                self.ub = 0.2 * torch.ones_like(self.diams) / (self.diams / 5) ** 2
            self.ub_initial = self.ub.clone()
            self.lb = torch.zeros_like(self.ub)

        self.fix_bound_up = fix_bound_up
        self.fix_bound_down = fix_bound_down
        self.max_tries_bound_fix = max_tries_bound_fix
        self.max_tries_thresh = max_tries_thresh
        self.resolution = resolution

        self.active = active
        self.rec = Recorder(["v"], max_only=True)

    def ve_from_s_t(self, ve_s, ve_t, device, multicontact=False):
        ve_s = torch.as_tensor(ve_s, device=device)
        ve_t = torch.as_tensor(ve_t, device=device)

        if multicontact:
            ve_s = ve_s.expand(-1, self.model.n_ax, -1)
            ve_t = ve_t.expand(-1, self.model.n_ax, -1)
            einsum = op_mc
        else:
            ve_s = ve_s.expand(self.model.n_ax, -1)
            ve_t = ve_t.expand(self.model.n_ax, -1)
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

    def check_active(self, dt, bound: Tensor):
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
        if self.chunks is None:
            ve = self.bases * bound[None, :, None, None]
            self.model.run(
                ve, callbacks=[self.active], reinit=True, dt=dt, progressbar=False
            )
        else:
            time = self.time.expand(self.model.n_ax, -1) * bound[:, None]
            self.model.longrun(
                space=self.space,
                time=time,
                reinit=True,
                progressbar=False,
                dt=dt,
                n_chunks=self.chunks,
                callbacks=[self.active],
            )
        return self.active.is_active()

    def check_active_with_rec(self, dt, bound: Tensor):
        self.active.reset()
        self.rec.reset()
        if self.chunks is None:
            ve = self.bases * bound[None, :, None, None]
            self.model.run(
                ve,
                callbacks=[self.active, self.rec],
                reinit=True,
                dt=dt,
                progressbar=False,
            )
        else:
            time = self.time.expand(self.model.n_ax, -1) * bound[:, None]
            self.model.longrun(
                space=self.space,
                time=time,
                reinit=True,
                progressbar=False,
                dt=dt,
                n_chunks=self.chunks,
                callbacks=[self.active, self.rec],
            )
        return self.active.is_active(), self.rec.stack()

    def fix_bounds(self, dt, block_possible=True):
        """Make sure upper bound generates AP."""

        with torch.no_grad():
            tries = 0
            if block_possible:
                mask, rec = self.check_active_with_rec(dt, self.ub)
            else:
                mask = self.check_active(dt, self.ub)
            print("Fixing bounds.", end="")
            while torch.any(~mask):
                print(".", end="")
                if tries >= self.max_tries_bound_fix:
                    break
                if block_possible:
                    mask, rec = self.check_active_with_rec(dt, self.ub)
                else:
                    mask = self.check_active(dt, self.ub)
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

    def calculate_thresholds(self, dt, block_possible=False) -> Tuple[Tensor, Tensor]:
        """Calculate thresholds.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Upper and lower bound on thresholds.
        """
        self.fix_bounds(dt, block_possible)
        self.rec.reset()
        self.active.reset()

        with torch.no_grad():
            ub = self.ub
            lb = self.lb

            window = (ub - lb) / ub
            msk = window >= self.resolution
            tries = 0

            while torch.any(msk) & (tries < self.max_tries_thresh):
                stimamp = (ub + lb) / 2
                mask = self.check_active(dt, stimamp)
                a_thr = msk & mask
                b_thr = msk & ~mask
                ub[a_thr] = stimamp[a_thr]
                lb[b_thr] = stimamp[b_thr]
                window = (ub - lb) / ub
                msk = window >= self.resolution
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
    return torch.einsum("can,cat->tan", s, t).unsqueeze(2)


@torch.jit.script
def op_sc(s: Tensor, t: Tensor) -> Tensor:
    return torch.einsum("an,at->tan", s, t).unsqueeze(2)
