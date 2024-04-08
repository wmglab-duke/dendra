from typing import List, Tuple

import torch
from torch import Tensor

from axonml.models import Axon
from axonml.models.callbacks import Active, Recorder


class Thresholder:
    def __init__(
        self,
        model: Axon,
        bases: Tensor,
        diams: Tensor,
        ub=None,
        fix_bound_up=5.0,
        fix_bound_down=0.1,
        max_tries_bound_fix=10,
        max_tries_thresh=25,
        resolution=0.01,
        threshold=0.0,
        node_check: List[int] = [5, -5],
        t_start_check=0.0,
    ):
        if isinstance(diams, Tensor):
            assert len(diams) == bases.shape[1]

        self.model = model.compile(bases.shape[-1], bases.shape[1])
        self.bases = bases.to(model.device())
        self.diams = diams.to(model.device())

        self.ignore = None

        with torch.no_grad():
            if ub is not None:
                self.ub = ub * torch.ones_like(diams, device=model.device())
            else:
                self.ub = 0.2 * torch.ones_like(diams) / (diams / 5) ** 2
            self.ub_initial = self.ub.clone()
            self.lb = torch.zeros_like(self.ub)

        self.fix_bound_up = fix_bound_up
        self.fix_bound_down = fix_bound_down
        self.max_tries_bound_fix = max_tries_bound_fix
        self.max_tries_thresh = max_tries_thresh
        self.resolution = resolution

        self.active = Active(threshold, t_start_check, node_check)
        self.rec = Recorder(max_only=True)

    def check_active(self, bound: Tensor):
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
        ve = self.bases * bound[None, :, None, None]
        self.model.run(ve, self.diams, callbacks=[self.active], reinit=True)
        return self.active.record

    def check_active_with_rec(self, bound: Tensor):
        self.active.reset()
        self.rec.reset()
        ve = self.bases * bound[None, :, None, None]
        self.model.run(ve, self.diams, callbacks=[self.active, self.rec], reinit=True)
        return self.active.record, self.rec.stack()

    def fix_bounds(self):
        """Make sure upper bound generates AP."""

        with torch.no_grad():
            tries = 0
            mask, rec = self.check_active_with_rec(self.ub)
            print("Fixing bounds.", end="")
            while torch.any(~mask):
                print(".", end="")
                if tries >= self.max_tries_bound_fix:
                    break
                mask, rec = self.check_active_with_rec(self.ub)
                inactive = ~mask
                self.ub[(rec[:, -1] < -20) & inactive] *= 10
                self.ub[(rec[:, -1] >= -20) & inactive] *= 0.2
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

    def calculate_thresholds(self) -> Tuple[Tensor, Tensor]:
        """Calculate thresholds.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Upper and lower bound on thresholds.
        """
        self.fix_bounds()
        self.rec.reset()

        with torch.no_grad():
            ub = self.ub
            lb = self.lb

            window = (ub - lb) / ub
            msk = window >= self.resolution
            tries = 0

            while torch.any(msk) & (tries < self.max_tries_thresh):
                stimamp = (ub + lb) / 2
                mask = self.check_active(stimamp)
                a_thr = msk & mask
                b_thr = msk & ~mask
                ub[a_thr] = stimamp[a_thr]
                lb[b_thr] = stimamp[b_thr]
                window = (ub - lb) / ub
                msk = window >= self.resolution
                tries += 1
            if tries >= self.max_tries_thresh:
                print("hmm")
                return ub, lb

            # final
            # stimamp = (ub + lb) / 2
            # mask = self.check_active(stimamp)
            # ub[mask] = stimamp[mask]

            if self.ignore is not None:
                ub[self.ignore] = torch.nan
                lb[self.ignore] = torch.nan

            return ub, lb
