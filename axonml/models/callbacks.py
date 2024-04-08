from typing import List

import torch
from torch import Tensor

from .backend import Backend as A


class Callback:
    def pre_loop_hook(self, states: Tensor):
        """Execute before entering solver loop.

        Parameters
        ----------
        states : Tensor
            System states.
        """
        pass

    def post_step_hook(self, states: Tensor):
        """Execute after each loop of solver (advance of
        single timestep.)

        Parameters
        ----------
        states : Tensor
            System states.
        """
        pass

    def post_loop_hook(self, states: Tensor):
        """Execute after solver loop completes.

        Parameters
        ----------
        states : Tensor
            System states.
        """
        pass


class CallbackList:
    def __init__(self, callbacks=None) -> None:
        self.callbacks = callbacks if callbacks is not None else []

    def __iter__(self):
        return iter(self.callbacks)

    def __len__(self):
        return len(self.callbacks)

    def __next__(self):
        return next(self.callbacks)

    def pre_loop_hook(self, states):
        for c in self:
            c.pre_loop_hook(states)

    def post_step_hook(self, states):
        for c in self:
            c.post_step_hook(states)

    def post_loop_hook(self, states):
        for c in self:
            c.post_loop_hook(states)


class Recorder(Callback):
    def __init__(self, max_only=False):
        self.rec: List[Tensor] = []
        self.max_only = max_only

    def reset(self):
        self.rec.clear()

    def pre_loop_hook(self, states):
        if self.max_only:
            self.rec.append(torch.amax(states, -1))
        else:
            self.rec.append(states)

    def post_step_hook(self, states):
        if self.max_only:
            self.rec.append(torch.amax(states, -1))
        else:
            self.rec.append(states)

    def stack(self):
        if self.max_only:
            return torch.amax(torch.stack(self.rec), 0)
        return torch.stack(self.rec)


class ThresholdCallback(Callback):
    def __init__(self, threshold=0.0, t_start_check=0.0, node_check=[5, -5], dt=None):
        self.record = None
        self.state_cache = None
        self.threshold = threshold
        self.t_start_check = t_start_check
        self.node_check = node_check
        self.i = 0
        self.dt = dt if dt is not None else A.dt

    def reset_timer(self):
        self.i = 0

    def reset_count(self):
        self.record = None

    def reset(self):
        self.reset_timer()
        self.reset_count()

    def numpy(self):
        if self.record is not None:
            return self.record.detach().cpu().numpy()
        return self.record


class APCount(ThresholdCallback):

    """Count the number of action potentials that arrived at each
    checked node.
    """

    def pre_loop_hook(self, states):
        if self.record is None:
            self.record = torch.zeros(
                states.shape[0],
                len(self.node_check),
                dtype=torch.int,
                device=states.device,
            )
        self.state_cache = states

    def post_step_hook(self, states):
        if self.i * self.dt >= self.t_start_check:
            vm_new = states[:, -1, self.node_check]
            vm = self.state_cache[:, -1, self.node_check]
            increment_count(vm, vm_new, self.record, self.threshold)
        self.state_cache = states
        self.i += 1


class Active(ThresholdCallback):

    """Record if fibers generated action potential(s)."""

    def pre_loop_hook(self, states):
        if self.record is None:
            self.record = torch.zeros(
                states.shape[0], dtype=torch.bool, device=states.device
            )
        self.state_cache = states

    def post_step_hook(self, states):
        if self.i * self.dt >= self.t_start_check:
            vm_new = states[:, -1, self.node_check]
            vm = self.state_cache[:, -1, self.node_check]
            update_active(vm, vm_new, self.record, self.threshold)
        self.state_cache = states
        self.i += 1


class Raster(ThresholdCallback):

    """Record all timepoints at which action potentials occur
    at checked nodes.
    """

    def pre_loop_hook(self, states):
        if self.record is None:
            self.record = []
        self.state_cache = states

    def post_step_hook(self, states):
        if self.i * self.dt >= self.t_start_check:
            vm_new = states[:, -1, self.node_check]
            vm = self.state_cache[:, -1, self.node_check]
            l_and = torch.logical_and(
                torch.ge(vm_new, self.threshold), torch.lt(vm, self.threshold)
            )
            self.record.append(l_and)
        self.state_cache = states
        self.i += 1

    def stack(self):
        return torch.stack(self.record)

    def numpy(self):
        return self.stack().detach().cpu().numpy()


@torch.jit.script
def increment_count(vm, vm_new, record, threshold: float) -> None:
    l_and = torch.logical_and(torch.ge(vm_new, threshold), torch.lt(vm, threshold))
    record[l_and] += 1


@torch.jit.script
def update_active(vm, vm_new, record, threshold: float) -> None:
    l_and = torch.any(
        torch.logical_and(torch.ge(vm_new, threshold), torch.lt(vm, threshold)), dim=1
    )
    record[l_and] = True
