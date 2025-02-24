from queue import Queue
from typing import Tuple, Dict, List
from types import MethodType
import threading
import multiprocessing as mp

from h5py import File

import torch
import torch.nn.functional as F

from .backend import Backend as A
from .interfaces import AxonInterface


TRANSFERSTREAM = torch.cuda.Stream()


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


template = """
def recorder(self, model):
  if self.save_every is not None:
    save = self.i % self.save_every == 0
  else:
    save = True
  if save:
  {implementation}
"""

impl_template = """
    states = model.mech.{mech}.{state}
    if self.max_only:
      self.rec['{full_state}'].append(torch.amax(states, -1))
    else:
      if self.node_indices is not None:
        self.rec['{full_state}'].append(atleast_3d(states[:, :, self.node_indices]))
      else:
        self.rec['{full_state}'].append(states)
"""

m_template = """
    states = model.{val}
    if self.max_only:
      self.rec['{val}'].append(torch.amax(states, -1))
    else:
      if self.node_indices is not None:
        self.rec['{val}'].append(atleast_3d(states[:, :, self.node_indices]))
      else:
        self.rec['{val}'].append(states)
"""

def is_state(s):
    return '.' in s


def parse_template(full_state):
    mech, state = full_state.split(".")
    return impl_template.format(mech=mech, state=state, full_state=full_state)


def build_recorder_func(states):
    res = []
    for s in states:
        if not is_state(s):
            res.append(m_template.format(val=s))
        else:
            res.append(parse_template(s))
    impl = "".join(res)
    forward_str = template.format(implementation=impl)
    filename = "<rec_template>"
    code = compile(forward_str, filename, "exec")
    exec(code)
    return locals()["recorder"]


def avoid_smart_indexing(node_indices):
    if node_indices is not None:
        if len(node_indices) == 1:
            return node_indices[0]
    return node_indices


def n(node_indices, model):
    if node_indices is None:
        return model.n_node
    if isinstance(node_indices, int):
        return 1
    return len(node_indices)


def atleast_2d(x: torch.Tensor) -> torch.Tensor:
    dims = x.dim()
    if dims == 1:
        return x.unsqueeze(-1)
    return x


def atleast_3d(x: torch.Tensor) -> torch.Tensor:
    dims = x.dim()
    if dims == 2:
        return x.unsqueeze(-1)
    return x


class Recorder(Callback):
    def __init__(
        self, states, max_only=False, node_indices=None, dt=None, sliding_window=None
    ):
        super().__init__()
        self.states = states
        self.rec: Dict[str, List[torch.Tensor]] = {s: [] for s in states}
        self.max_only: bool = max_only
        self.node_indices = avoid_smart_indexing(node_indices)
        self.sliding_window = sliding_window
        rfunc = build_recorder_func(states)
        setattr(self, "_post_step_hook", MethodType(rfunc, self))
        setattr(self, "_pre_loop_hook", MethodType(rfunc, self))

        self._dt = None
        self.save_dt = dt
        self.save_every = (
            int(self.save_dt / self.dt) if self.save_dt is not None else None
        )

        # HDF5 -- optional -- for large data
        self.queue = None
        self.hdf5_path = None
        self.cache_with_hdf5 = False
        self.cache_every = None
        self.save_count = 0
        self.i = 0
        self.run_number = 0
        self.manager = None
        self.writer_thread = None
        self.data_pinned = {}
        self.stream = None

    @property
    def dt(self):
        return self._dt or A.dt

    @dt.setter
    def dt(self, value):
        self._dt = value
        if self.save_dt is not None:
            self.save_every = int(self.save_dt / self.dt)

    def set_hdf5(self, hdf5: str, cache_every=10000):
        self.hdf5_path = hdf5
        mp.set_start_method("spawn", force=True)
        self.manager = mp.Manager()
        self.queue = self.manager.Queue()
        self.writer_thread = mp.Process(
            target=hdf5_write, args=(self.queue, self.hdf5_path)
        )
        self.writer_thread.start()
        self.cache_with_hdf5 = True
        self.cache_every = cache_every
        return self

    def pre_loop_hook(self, model):
        self._pre_loop_hook(model)
        if self.cache_with_hdf5:
            self.i += 1

    def cache_hdf5(self):
        with torch.cuda.stream(TRANSFERSTREAM):
            for s in self.states:
                data = self.stack(s)
                if s not in self.data_pinned:
                    self.data_pinned[s] = torch.empty(
                        data.shape, dtype=data.dtype, device="cpu", pin_memory=True
                    )
                if self.data_pinned[s].shape[0] != data.shape[0]:
                    self.data_pinned[s] = torch.empty(
                        data.shape, dtype=data.dtype, device="cpu", pin_memory=True
                    )
                self.data_pinned[s].copy_(data, non_blocking=True)
        self.queue.put("flush")
        chunks = data.shape
        chunks = (chunks[0], 1, chunks[2], chunks[3])
        for s in self.states:
            self.queue.put(
                (s, self.run_number, self.save_count, self.data_pinned[s], chunks)
            )
        self.save_count += 1

    def post_step_hook(self, model):
        self._post_step_hook(model)
        self.i += 1
        if self.cache_with_hdf5:
            if self.i % self.cache_every == 0:
                self.cache_hdf5()
                self.rec = {s: [] for s in self.states}

    def post_loop_hook(self, model):
        if self.cache_with_hdf5:
            self.cache_hdf5()
        self.save_count = 0
        self.run_number += 1

    def reset(self):
        self.rec = {s: [] for s in self.states}
        self.i = 0

    def close(self):
        if self.cache_with_hdf5:
            self.queue.put(None)
            self.writer_thread.join()

    def stack(self, var: str = None):
        if var is not None:
            vs = torch.stack(self.rec[var])
            if self.sliding_window is not None:
                vs = sliding_window_average(vs, self.sliding_window)
            if self.max_only:
                return torch.amax(vs, 0)
            return vs
        vs = torch.cat([torch.stack(self.rec[s]) for s in self.rec], dim=2)
        if self.sliding_window is not None:
            vs = sliding_window_average(vs, self.sliding_window)
        if self.max_only:
            return torch.amax(vs, 0)
        return vs

    def numpy(self, var: str = None):
        if var is not None:
            return self.stack(var).detach().cpu().numpy()
        return self.stack().detach().cpu().numpy()


def hdf5_write(queue: Queue, path: str):
    with File(path, "w", libver="latest") as f:
        while True:
            item = queue.get()
            if item == "flush":
                TRANSFERSTREAM.synchronize()
                queue.task_done()
                continue
            if item is None:
                queue.task_done()
                break
            state, run, save_count, data, chunks = item
            group = f.require_group(f"/{state}/run_{run}")
            group.create_dataset(f"{save_count}", data=data, chunks=chunks)
            queue.task_done()


class ThresholdCallback(Callback):
    def __init__(self, threshold=0.0, t_start_check=0.0, node_check=[5, -5], dt=None):
        super().__init__()
        self.record: torch.Tensor = None
        self.state_cache: torch.Tensor = None
        self.threshold: float = threshold
        self.t_start_check: float = t_start_check
        self.node_check: List[int] = avoid_smart_indexing(node_check)
        self.i: int = 0
        self.dt: float = dt if dt is not None else A.dt

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
                n(self.node_check, model),
                dtype=torch.int32,
                device=model.device(),
            )
        if self.state_cache is None:
            self.state_cache = torch.ones(
                model.n(),
                n(self.node_check, model),
                dtype=torch.bool,
                device=model.device(),
            )

    def post_step_hook(self, model: AxonInterface):
        if self.i * self.dt >= self.t_start_check:
            vm_new = atleast_2d(model.v[:, 0, self.node_check].squeeze())
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
                n(self.node_check, model),
                dtype=torch.bool,
                device=model.device(),
            )

    def post_step_hook(self, model: AxonInterface):
        if self.i * self.dt >= self.t_start_check:
            vm_new = atleast_2d(model.v[:, 0, self.node_check].squeeze())
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
                n(self.node_check, model),
                dtype=torch.bool,
                device=model.device(),
            )

    def post_step_hook(self, model: AxonInterface):
        if self.i * self.dt >= self.t_start_check:
            vm_new = atleast_2d(model.v[:, 0, self.node_check])
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


@torch.jit.script
def sliding_window_average(x, window_size: int):
    """
    Compute the sliding (moving) window average along axis 0 for a 4D array/tensor,
    with padding so that the output has the same shape as the input.

    For an input of shape (N, C, H, W) and a given window_size, the function pads the input
    along axis 0 using edge replication and then computes the average over every consecutive window.
    The output shape is (N, C, H, W).

    Parameters
    ----------
    x : np.ndarray or torch.Tensor
        A 4-dimensional array/tensor with shape (N, C, H, W).
    window_size : int
        The size of the sliding window (must be >= 1).

    Returns
    -------
    out : same type as x
        The sliding window averages computed along axis 0 with the same shape as the input.

    Raises
    ------
    ValueError
        If window_size is less than 1.
    TypeError
        If x is not a NumPy array or a PyTorch tensor.
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")

    # Compute padding sizes so that:
    #    pad_left + pad_right = window_size - 1
    if window_size % 2 == 1:
        pad_left = pad_right = window_size // 2
    else:
        pad_left = window_size // 2
        pad_right = window_size // 2 - 1

    # ---------------------------
    # PyTorch implementation
    # ---------------------------
    # x shape: (N, C, H, W)
    N, C, H, W = x.shape

    # Manually pad along axis 0 (the N dimension) using replication.
    # For pad_left, replicate the first slice; for pad_right, replicate the last slice.
    left_pad = (
        x[0:1].expand(pad_left, -1, -1, -1)
        if pad_left > 0
        else torch.empty(0, device=x.device, dtype=x.dtype)
    )
    right_pad = (
        x[-1:].expand(pad_right, -1, -1, -1)
        if pad_right > 0
        else torch.empty(0, device=x.device, dtype=x.dtype)
    )
    # Concatenate along dimension 0.
    x_padded = torch.cat([left_pad, x, right_pad], dim=0)
    N_padded = x_padded.shape[0]  # should equal N + (window_size - 1)

    # Reshape so that the padded N dimension is the "length" dimension.
    # Collapse (C, H, W) into the channel dimension and use a batch size of 1.
    # New shape: (1, C*H*W, N_padded)
    x_reshaped = x_padded.permute(1, 2, 3, 0).reshape(1, C * H * W, N_padded)

    # Create an averaging kernel for each channel.
    # For grouped conv1d with groups = C*H*W, the kernel should have shape:
    # (C*H*W, 1, window_size)
    kernel = (
        torch.ones(C * H * W, 1, window_size, dtype=x.dtype, device=x.device)
        / window_size
    )

    # Perform grouped convolution along the length dimension.
    out_conv = F.conv1d(x_reshaped, kernel, groups=C * H * W)
    # out_conv shape: (1, C*H*W, L) where L = N_padded - window_size + 1.
    L = out_conv.shape[-1]
    if L != N:
        raise RuntimeError(f"Unexpected output length: got {L}, expected {N}.")

    # Reshape back to (1, C, H, W, N) and then permute to (N, C, H, W)
    # First, view out_conv as (1, C, H, W, N)
    out_5d = out_conv.view(1, C, H, W, N)
    # Permute to bring the last dimension (N) to the front: (1, N, C, H, W)
    out_perm = out_5d.permute(0, 4, 1, 2, 3)
    # Remove the extra batch dimension (squeeze dimension 0)
    out = out_perm.squeeze(0)
    return out
