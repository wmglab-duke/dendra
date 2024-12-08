from typing import Tuple, Dict, List

import torch

from .backend import Backend as A
from .interfaces import AxonInterface


class Callback:
    def pre_loop_hook(self, model: AxonInterface):
        """Execute before entering solver loop.

        Parameters
        ----------
        states : Tensor
            System states.
        """
        pass

    def post_step_hook(self, model: AxonInterface):
        """Execute after each loop of solver (advance of
        single timestep.)

        Parameters
        ----------
        states : Tensor
            System states.
        """
        pass

    def post_loop_hook(self, model: AxonInterface):
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

    def pre_loop_hook(self, model: AxonInterface):
        for c in self:
            c.pre_loop_hook(model)

    def post_step_hook(self, model: AxonInterface):
        for c in self:
            c.post_step_hook(model)

    def post_loop_hook(self, model: AxonInterface):
        for c in self:
            c.post_loop_hook(model)


class Recorder(Callback):

    def __init__(self, states, max_only=False, node_indices=None):
        super().__init__()
        self.states = states
        self.rec: Dict[str, List[torch.Tensor]] = {s: [] for s in states}
        self.max_only : bool = max_only
        self.node_indices = node_indices

    def reset(self):
        self.rec = {s: [] for s in self.states}

    def pre_loop_hook(self, model: AxonInterface):
        for s in self.rec:
            states = model.get_state(s)
            if self.max_only:
                self.rec[s].append(torch.amax(states, -1))
            else:
                if self.node_indices is not None:
                    self.rec[s].append(states[:, :, self.node_indices])
                else:
                    self.rec[s].append(states)

    def post_step_hook(self, model: AxonInterface):
        for s in self.rec:
            states = model.get_state(s)
            if self.max_only:
                self.rec[s].append(torch.amax(states, -1))
            else:
                if self.node_indices is not None:
                    self.rec[s].append(states[:, :, self.node_indices])
                else:
                    self.rec[s].append(states)

    def stack(self, var: str = None):
        if var is not None:
            vs = torch.stack(self.rec[var])
            if self.max_only:
                return torch.amax(vs, 0)
            return vs
        vs = torch.cat([torch.stack(self.rec[s]) for s in self.rec], dim=2)
        if self.max_only:
            return torch.amax(vs, 0)
        return vs


class ThresholdCallback(Callback):
    def __init__(self, threshold=0.0, t_start_check=0.0, node_check=[5, -5], dt=None):
        super().__init__()
        self.record : torch.Tensor = None
        self.state_cache : torch.Tensor = None
        self.threshold : float = threshold
        self.t_start_check : float = t_start_check
        self.node_check : List[int] = node_check
        self.i : int = 0
        self.dt : float= dt if dt is not None else A.dt

    def reset_timer(self):
        self.i = 0

    def reset_count(self):
        self.record = None

    def reset_state_cache(self):
        self.state_cache = None

    def reset(self):
        self.reset_timer()
        self.reset_count()
        self.reset_state_cache()

    def numpy(self):
        if self.record is not None:
            return self.record.detach().cpu().numpy()
        return self.record


class APCount(ThresholdCallback):
    """Count the number of action potentials that arrived at each
    checked node.
    """

    def pre_loop_hook(self, model: AxonInterface):
        if self.record is None:
            self.record = torch.zeros(
                model.n(),
                len(self.node_check),
                dtype=torch.int32,
                device=model.device(),
            )
        if self.state_cache is None:
            self.state_cache = torch.ones(
                model.n(),
                len(self.node_check),
                dtype=torch.bool,
                device=model.device(),
            )

    def post_step_hook(self, model: AxonInterface):
        if self.i * self.dt >= self.t_start_check:
            vm_new = model.get_state("v")[:, 0, self.node_check]
            vm = self.state_cache
            self.state_cache = increment_count_(vm, vm_new, self.record, self.threshold)
        self.i += 1


class ActiveAL(APCount):
    def __init__(
        self, threshold=0.0, t_start_check=0.0, node_check=[5, -5], dt=None, at_least=1
    ):
        super().__init__(threshold, t_start_check, node_check, dt)
        self.at_least = at_least

    def is_active(self):
        if self.record is not None:
            return is_active(self.record, self.at_least)
        return self.record

    def numpy(self):
        if self.record is not None:
            return self.is_active().detach().cpu().numpy()
        return self.record


class Active(ThresholdCallback):
    """Record if fibers generated action potential(s)."""

    def pre_loop_hook(self, model):
        if self.record is None:
            self.record = torch.zeros(
                model.n(), dtype=torch.bool, device=model.device()
            )
        if self.state_cache is None:
            self.state_cache = torch.ones(
                model.n(),
                len(self.node_check),
                dtype=torch.bool,
                device=model.device(),
            )

    def post_step_hook(self, model: AxonInterface):
        if self.i * self.dt >= self.t_start_check:
            vm_new = model.get_state("v")[:, 0, self.node_check]
            vm = self.state_cache
            self.state_cache, la = update_active(vm, vm_new, self.threshold)
            self.record[la] = True
        self.i += 1

    def is_active(self):
        return self.record

    def numpy(self):
        return self.record.detach().cpu().numpy()


class Raster(ThresholdCallback):
    """Record all timepoints at which action potentials occur
    at checked nodes.
    """

    def pre_loop_hook(self, model: AxonInterface):
        if self.record is None:
            self.record = []
        if self.state_cache is None:
            self.state_cache = torch.ones(
                model.n(),
                len(self.node_check),
                dtype=torch.bool,
                device=model.device(),
            )

    def post_step_hook(self, model: AxonInterface):
        if self.i * self.dt >= self.t_start_check:
            vm_new = model.get_state("v")[:, -1, self.node_check]
            vm = self.state_cache
            self.state_cache, la = increment_count(vm, vm_new, self.threshold)
            self.record.append(la)
        self.i += 1

    def stack(self):
        return torch.stack(self.record)

    def numpy(self):
        return self.stack().detach().cpu().numpy()


@torch.jit.script
def increment_count(vm, vm_new, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
    ge = vm_new >= threshold
    l_and = torch.logical_and(ge, vm)
    return ~ge, l_and


@torch.jit.script
def increment_count_(vm, vm_new, record, threshold: float) -> torch.Tensor:
    ge = vm_new >= threshold
    l_and = torch.logical_and(ge, vm)
    record[l_and] += 1
    return ~ge


@torch.jit.script
def update_active(vm, vm_new, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
    ge = vm_new >= threshold
    l_and = torch.any(torch.logical_and(ge, vm), dim=1)
    return ~ge, l_and


@torch.jit.script
def is_active(record, at_least: int) -> torch.Tensor:
    return torch.count_nonzero(record, dim=1) >= at_least
