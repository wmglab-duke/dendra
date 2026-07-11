import gc
import multiprocessing as mp
from contextlib import nullcontext
from numbers import Integral
from queue import Queue
from types import MethodType
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from h5py import File

from dendra.utils.dynamic_compilation import compile_generated_function

from ..helpers import nojit
from .backend import Backend as A

if torch.cuda.is_available():
    TRANSFERSTREAM = torch.cuda.Stream()
else:
    TRANSFERSTREAM = None


class Callback(torch.nn.Module):
    """
    Base class for simulation callbacks in Dendra.

    This class defines the interface for callbacks that can be passed to
    ``run`` or ``longrun`` methods of models to monitor and interact with the
    simulation at specific points in the execution flow. Subclasses override
    hook methods to record state, detect events, or stream data during a run.

    Notes
    -----
    The default implementations are no-ops; override only the hooks you need.
    Multiple callbacks can be used simultaneously by passing them in a list to
    the model's run method or by wrapping them in :class:`CallbackList`.

    See Also
    --------
    Recorder : Records model states during simulation
    LFP : Records Local Field Potential signals
    APCount : Counts action potentials during simulation
    Active : Detects if axons fire during simulation
    Raster : Records spike events for raster plots

    Examples
    --------
    >>> class CustomCallback(Callback):
    ...     def pre_loop_hook(self, model):
    ...         print(f"Starting simulation with {model.n()} axons")
    ...     def post_loop_hook(self, model):
    ...         print(f"Finished simulation at t={model.t} ms")
    >>>
    >>> callback = CustomCallback()
    >>> model.run(ve, callbacks=[callback])
    """

    def pre_loop_hook(self, model):
        """Run once before entering the solver loop.

        Override to allocate buffers, move indices to the correct device, or
        otherwise initialize state based on the provided ``model``."""
        pass

    def pre_chunk_hook(self, model, timepoints=None):
        """Run before processing a chunk of timepoints.

        Override to prepare per-chunk scratch space or bookkeeping using the
        supplied ``timepoints``."""
        pass

    def post_chunk_hook(self, model, timepoints=None):
        """Run after finishing a chunk of timepoints.

        Override to flush results accumulated during the chunk or to update
        metrics that depend on the completed block."""
        pass

    def pre_step_hook(self, model):
        """Run before each solver step (single timestep).

        Override to inspect or modify model state just before integration."""
        pass

    def post_step_hook(self, model):
        """Run after each solver step (single timestep).

        Override to record outputs or update counters after the model advances."""
        pass

    def post_loop_hook(self, model):
        """Run once after the solver loop completes.

        Override to finalize results or free resources."""
        pass


class CallbackList(torch.nn.Module):
    """
    Lightweight container that forwards hook calls to multiple callbacks.

    Parameters
    ----------
    callbacks : list[Callback], optional
        Sequence of callback instances to run. The list is stored as a
        :class:`torch.nn.ModuleList` so callbacks are registered buffers.
    """

    def __init__(self, callbacks=None) -> None:
        super().__init__()
        self.callbacks = (
            torch.nn.ModuleList(callbacks)
            if callbacks is not None
            else torch.nn.ModuleList()
        )

    def __iter__(self):
        return iter(self.callbacks)

    def __len__(self):
        return len(self.callbacks)

    def __next__(self):
        return next(self.callbacks)

    def pre_loop_hook(self, model):
        """Call ``pre_loop_hook`` on each callback in order."""
        for c in self.callbacks:
            c.pre_loop_hook(model)

    def post_loop_hook(self, model):
        """Call ``post_loop_hook`` on each callback in order."""
        for c in self.callbacks:
            c.post_loop_hook(model)

    def pre_chunk_hook(self, model, timesteps):
        """Call ``pre_chunk_hook`` on each callback in order."""
        for c in self.callbacks:
            c.pre_chunk_hook(model, timesteps)

    def post_chunk_hook(self, model, timesteps):
        """Call ``post_chunk_hook`` on each callback in order."""
        for c in self.callbacks:
            c.post_chunk_hook(model, timesteps)

    def pre_step_hook(self, model):
        """Call ``pre_step_hook`` on each callback in order."""
        for c in self.callbacks:
            c.pre_step_hook(model)

    def post_step_hook(self, model):
        """Call ``post_step_hook`` on each callback in order."""
        for c in self.callbacks:
            c.post_step_hook(model)


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
    states = model.integrator.mech.{mech}.{state}
    {implementation}
"""

m_template = """
    states = model.{val}
    {implementation}
"""

max_only_t = "_append_tensor_max(states, self.rec['{full_state}'], self.partition)"
indexed_t = (
    "_append_tensor_indexed(states, self.rec['{full_state}'], self.node_indices)"
)
base_t = "_append_tensor(states, self.rec['{full_state}'])"


def _append_tensor(tensor: torch.Tensor, record: List[torch.Tensor]) -> None:
    """
    Append a tensor to a list of tensors.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to append.
    record : List[torch.Tensor]
        The list to which the tensor will be appended.
    """
    record.append(tensor)


def _append_tensor_max(
    tensor: torch.Tensor, record: List[torch.Tensor], partition=None
) -> None:
    """
    Append max-reduced tensor values to a list of tensors.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor from which the maximum will be taken.
    record : List[torch.Tensor]
        The list to which the maximum value will be appended.
    partition : sequence of int, optional
        Segment lengths that partition ``tensor`` along dimension 1. The sum of
        the sequence must equal ``tensor.shape[1]`` and all values must be
        positive. If provided, the maximum is computed within each segment
        along dimension 1 and stacked into a new dimension.
    """
    if partition is None:
        record.append(torch.amax(tensor, -1, keepdim=True))
        return

    if tensor.dim() < 2:
        raise ValueError("partitioned max requires tensor with at least 2 dims.")

    total = int(partition.sum().item())
    if total != tensor.shape[1]:
        raise ValueError(
            "sum(partition) must equal tensor.shape[1]; "
            f"got {total} and {tensor.shape[1]}."
        )

    segments = torch.split(tensor, partition.tolist(), dim=1)
    record.append(torch.stack([torch.amax(seg, dim=1) for seg in segments], dim=1))


def _append_tensor_indexed(
    tensor, record: List[torch.Tensor], node_indices: torch.Tensor
) -> None:
    """
    Append a tensor indexed by node_indices to a list of tensors.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to append.
    record : List[torch.Tensor]
        The list to which the tensor will be appended.
    node_indices : torch.Tensor
        Indices of nodes to select from the tensor.
    """
    record.append(tensor.index_select(-1, node_indices))


def _is_state(s):
    return "." in s


def _parse_template(full_state, max_only=False, indexed=False):
    mech, state = full_state.split(".")
    if max_only:
        impl = max_only_t
    elif indexed:
        impl = indexed_t
    else:
        impl = base_t
    implementation = impl.format(full_state=full_state)
    return impl_template.format(mech=mech, state=state, implementation=implementation)


def _parse_template_m(state, max_only=False, indexed=False):
    if max_only:
        impl = max_only_t
    elif indexed:
        impl = indexed_t
    else:
        impl = base_t
    implementation = impl.format(full_state=state)
    return m_template.format(val=state, implementation=implementation)


def _build_recorder_func(states, max_only, indexed):
    res = []
    for s in states:
        if not _is_state(s):
            res.append(_parse_template_m(s, max_only, indexed))
        else:
            res.append(_parse_template(s, max_only, indexed))
    impl = "".join(res)
    forward_str = template.format(implementation=impl)
    return compile_generated_function(
        forward_str,
        func_name="recorder",
        filename_prefix="dendra.callbacks.recorder_func",
        global_ns=globals(),
    )


def _avoid_smart_indexing(node_indices):
    if node_indices is not None:
        if len(node_indices) == 1:
            return node_indices[0]
    return node_indices


def _n(node_indices, model):
    if node_indices is None:
        return model.nc
    if node_indices.ndim == 0:
        return 1
    return len(node_indices)


def _atleast_2d(x: torch.Tensor) -> torch.Tensor:
    dims = x.dim()
    if dims == 1:
        return x.unsqueeze(-1)
    return x


def _atleast_3d(x: torch.Tensor) -> torch.Tensor:
    dims = x.dim()
    if dims == 2:
        return x.unsqueeze(-1)
    return x


def _detect_anomalies(x: torch.Tensor, prev) -> torch.Tensor:
    anomalous = (torch.isnan(x) | torch.isinf(x)).squeeze().any(-1)
    anomalous = torch.logical_or(anomalous, prev)
    return anomalous


class AnomalyDetector(Callback):
    """
    Flag NaN/Inf occurrences in membrane voltage during simulation.

    The detector keeps a boolean mask per axon indicating whether any invalid
    values have appeared so far. The mask is updated every step and can be
    retrieved as a tensor or NumPy array for inspection after a run.
    """

    def __init__(self):
        super().__init__()
        self.rec = None

    def pre_loop_hook(self, model):
        """Allocate a boolean flag per axon on the model's device."""
        self.rec = torch.zeros(model.n_ax, dtype=torch.bool, device=model.device())

    def post_step_hook(self, model):
        """Update the anomaly mask after each step using :func:`_detect_anomalies`."""
        with torch.no_grad():
            self.rec = _detect_anomalies(model.v, self.rec)

    def reset(self):
        """Clear the anomaly mask so the detector can be reused."""
        if self.rec is not None:
            self.rec = torch.zeros(
                self.rec.shape, dtype=torch.bool, device=self.rec.device
            )

    def numpy(self):
        """Return the anomaly mask as a NumPy array."""
        if self.rec is not None:
            return self.rec.detach().cpu().numpy()
        return None


class Recorder(Callback):
    """
    Record and store model states during simulation.

    This callback records specified model states (membrane potentials, ion concentrations,
    channel states, etc.) at regular intervals during simulation. It can save data directly
    in memory or cache to HDF5 files for large-scale simulations.

    Parameters
    ----------
    states : list of str
        List of state names to record. Can be model attributes (e.g., 'v') or
        mechanism states (e.g., 'hh.m').
    max_only : bool, optional
        If True, only the maximum value across nodes is recorded for each state.
        Default is False.
    node_indices : list of int, optional
        Indices of specific nodes to record. If None, all nodes are recorded.
        Default is None.
    dt : float, optional
        Time step for recording. If provided, states are recorded every
        dt/model.dt steps. Default is None (record every step).
    sliding_window : int, optional
        Size of sliding window for temporal averaging of recorded data.
        Default is None (no averaging).

    Notes
    -----
    For large-scale simulations, use the set_hdf5 method to enable caching to an
    HDF5 file, which helps manage memory usage.

    Attributes
    ----------
    states : list of str
        List of state names being recorded.
    rec : dict
        Dictionary mapping state names to lists of recorded tensors.
    save_dt : float
        Recording time step.
    save_every : int
        Number of simulation steps between recordings.
    i : int
        Current step counter.
    hdf5_path : str
        Path to HDF5 file if using HDF5 storage.
    cache_with_hdf5 : bool
        Whether to cache data to HDF5 file.
    cache_every : int
        Number of steps between HDF5 cache operations.

    """

    def __init__(
        self,
        states,
        max_only=False,
        node_indices=None,
        dt=None,
        sliding_window=None,
        partition=None,
    ):
        super().__init__()
        self.states = states
        self.rec: Dict[str, List[torch.Tensor]] = {s: [] for s in states}
        self.max_only: bool = max_only
        self.partition = None
        self.set_partition(partition)

        self.indexed = node_indices is not None
        self.node_indices = None

        if self.indexed:
            self.node_indices = torch.as_tensor(node_indices, dtype=torch.long)

        self.sliding_window = sliding_window
        rfunc = _build_recorder_func(states, self.max_only, self.indexed)
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

    def set_partition(self, partition=None):
        if partition is not None:
            lengths = torch.as_tensor(partition, dtype=torch.long, device="cpu")
            if lengths.dim() != 1:
                raise ValueError("partition must be a 1D sequence of integers.")
            if lengths.numel() == 0:
                raise ValueError("partition must be non-empty.")
            if torch.any(lengths <= 0):
                raise ValueError("partition values must be positive.")
            self.partition = lengths

    @property
    def dt(self):
        return self._dt or A.dt

    @dt.setter
    def dt(self, value):
        self._dt = value
        if self.save_dt is not None:
            self.save_every = int(self.save_dt / self.dt)

    def set_hdf5(self, hdf5: str, cache_every: int = 10000) -> "Recorder":
        """
        Enables caching of recorded data to an HDF5 file.

        This method configures the recorder to periodically save recorded data to an HDF5 file
        instead of keeping everything in memory. This is particularly useful for large-scale
        simulations where memory usage would otherwise become prohibitive.

        Parameters
        ----------
        hdf5 : str
            File path where the HDF5 data will be stored.
        cache_every : int, optional
            Number of simulation steps between cache operations. Controls how
            frequently data is written to the HDF5 file. Default is 10000 steps.

        Returns
        -------
        Recorder
            Returns the recorder instance for method chaining.

        Notes
        -----
        This method starts a separate process for writing data to the HDF5 file to avoid
        blocking the main simulation. The process will continue running until the recorder's
        close() method is called.
        """
        if isinstance(cache_every, bool) or not isinstance(cache_every, Integral):
            raise TypeError("cache_every must be a positive integer.")
        if cache_every <= 0:
            raise ValueError("cache_every must be a positive integer.")
        if self.cache_with_hdf5:
            raise RuntimeError("HDF5 caching is already enabled for this Recorder.")

        self.hdf5_path = hdf5
        mp.set_start_method("spawn", force=True)
        self.manager = mp.Manager()
        self.queue = self.manager.Queue()
        self.writer_thread = mp.Process(
            target=_hdf5_write, args=(self.queue, self.hdf5_path)
        )
        self.writer_thread.start()
        self.cache_with_hdf5 = True
        self.cache_every = int(cache_every)
        return self

    def pre_loop_hook(self, model):
        """
        Prepare recording buffers before simulation starts.

        - Moves ``node_indices`` to the model device if indexed recording is used.
        - Executes the generated recorder function once to capture the initial
          state (so ``t=0`` is included).
        - Advances the internal step counter if HDF5 caching is enabled so that
          cache flushing cadence aligns with subsequent ``post_step_hook`` calls.
        """
        if self.indexed:
            self.node_indices = self.node_indices.to(model.device())
        self._pre_loop_hook(model)
        if self.cache_with_hdf5:
            self.i += 1

    def cache_hdf5(self):
        populated = [bool(self.rec[s]) for s in self.states]
        if not self.states or not any(populated):
            return
        if not all(populated):
            missing = [
                state for state, present in zip(self.states, populated) if not present
            ]
            raise RuntimeError(
                "Cannot cache a partial Recorder frame; no samples were recorded "
                f"for state(s): {', '.join(missing)}."
            )

        transfer_context = (
            torch.cuda.stream(TRANSFERSTREAM)
            if TRANSFERSTREAM is not None
            else nullcontext()
        )
        with transfer_context:
            for s in self.states:
                data = self.stack(s)
                # Each queued write owns its buffer. Reusing a pinned tensor can
                # overwrite data that the asynchronous writer process has not
                # consumed yet.
                self.data_pinned[s] = torch.empty(
                    data.shape,
                    dtype=data.dtype,
                    device="cpu",
                    pin_memory=TRANSFERSTREAM is not None,
                )
                self.data_pinned[s].copy_(data, non_blocking=True)
        self.queue.put("flush")
        for s in self.states:
            data = self.data_pinned[s]
            chunks = list(data.shape)
            if data.ndim > 1:
                chunks[1] = 1
            self.queue.put((s, self.run_number, self.save_count, data, tuple(chunks)))
        self.save_count += 1

    @nojit
    def post_step_hook(self, model):
        """
        Record states after each solver step and optionally flush to HDF5.

        The generated recorder function pulls the requested states from the
        model and appends them to in-memory lists. The step counter is then
        incremented; if ``cache_with_hdf5`` is enabled and ``i`` hits a
        multiple of ``cache_every``, the current buffer is transferred
        (optionally via a CUDA stream) to the writer process and the in-memory
        lists are cleared.
        """
        self._post_step_hook(model)
        self.i += 1
        if self.cache_with_hdf5:
            if self.i % self.cache_every == 0:
                self.cache_hdf5()
                self.rec = {s: [] for s in self.states}

    def post_loop_hook(self, model):
        """
        Finalize recording at the end of a run.

        Flush any remaining buffered data to HDF5, then reset the per-run cache
        counters so the recorder can be reused for subsequent runs.
        """
        if self.cache_with_hdf5:
            self.cache_hdf5()
            self.rec = {s: [] for s in self.states}
            self.i = 0
        self.save_count = 0
        self.run_number += 1

    def reset(self):
        """
        Reset the recorder's state.

        This method clears all recorded data and resets the step counter.
        It's useful when you want to reuse the same recorder instance for
        multiple simulation runs without the data from previous runs.

        Parameters
        ----------
        None

        Returns
        -------
        None

        See Also
        --------
        close : Close resources used by the recorder

        Examples
        --------
        >>> recorder = Recorder(['v'])
        >>> model.run(ve, callbacks=[recorder])
        >>> # Get data from first run
        >>> first_run_data = recorder.numpy()
        >>> # Reset recorder for another run
        >>> recorder.reset()
        >>> model.run(ve2, callbacks=[recorder])
        >>> # Get data only from second run
        >>> second_run_data = recorder.numpy()
        """
        self.rec = {s: [] for s in self.states}
        self.i = 0

    def close(self) -> None:
        """
        Close resources used by the callback.

        If HDF5 caching is enabled, this method sends a termination signal
        to the writer thread and waits for it to complete. This ensures that all
        cached data is properly written to the HDF5 file before the callback is destroyed.

        Returns
        -------
        None

        See Also
        --------
        reset : Reset the recorder's state
        set_hdf5 : Enable HDF5 caching for recorded data
        """
        if self.cache_with_hdf5:
            self.queue.put(None)
            self.writer_thread.join()
            if self.manager is not None:
                self.manager.shutdown()
            self.cache_with_hdf5 = False

    def stack(self, var: str = None) -> torch.Tensor:
        """
        Stack recorded tensors into a single tensor.

        This method combines the recorded state tensors into a single tensor. If a specific
        state name is provided, only that state's data is stacked. Otherwise, all recorded
        states are stacked and concatenated.

        Parameters
        ----------
        var : str, optional
            Name of the specific state to stack. If None, all states
            are stacked and concatenated along dimension 2. Default is None.

        Returns
        -------
        torch.Tensor
            A tensor containing the stacked recorded data. If max_only is True,
            returns the maximum value along the time dimension. If sliding_window is set,
            the data is temporally averaged using the specified window size.
        """
        if var is not None:
            vs = torch.stack(self.rec[var])
            if self.sliding_window is not None:
                vs = _sliding_window_average(vs, self.sliding_window)
            return vs
        vs = torch.stack([torch.stack(self.rec[s]) for s in self.rec], dim=2)
        if self.sliding_window is not None:
            vs = _sliding_window_average(vs, self.sliding_window)
        if self.max_only:
            return torch.amax(vs, 0)
        return vs

    def numpy(self, var: str = None) -> np.ndarray:
        """
        Convert recorded tensors to NumPy arrays.

        This method converts the stacked tensor data to NumPy arrays by detaching from
        the computation graph and moving the data to CPU memory. If a specific state
        name is provided, only that state's data is converted.

        Parameters
        ----------
        var : str, optional
            Name of the specific state to convert to NumPy array.
            If None, all recorded states are stacked and converted. Default is None.

        Returns
        -------
        numpy.ndarray
            NumPy array containing the recorded data.
        """
        if var is not None:
            return self.stack(var).detach().cpu().numpy()
        return self.stack().detach().cpu().numpy()


class RecorderLambda(Callback):
    """
    Record arbitrary derived values using user-defined callables.

    Pass a mapping of ``name -> func`` where each ``func`` accepts the model and
    returns a tensor (or list of tensors) to store. Useful for recording
    composite metrics that are not direct model attributes.

    Parameters
    ----------
    funcs : Mapping[str, Callable]
        Dictionary of callables. Each is invoked every step with the model and
        its output is appended to the list under the same key in ``rec``.

    Notes
    -----
    Each function should be designed to work with the model's current state.

    Attributes
    ----------
    funcs : Mapping[str, Callable]
        User-defined functions for recording data.
    rec : dict[str, list]
        Per-key lists that collect outputs from ``funcs`` across steps.

    """

    def __init__(self, funcs):
        super().__init__()
        self.funcs = funcs
        self.rec = {}

    def pre_loop_hook(self, model):
        """Prime recording by capturing the initial outputs of each function."""
        self.post_step_hook(model)

    @nojit
    def post_step_hook(self, model):
        """Call each user function and append its output to the keyed list."""
        for name, func in self.funcs.items():
            self.rec.setdefault(name, []).append(func(model))

    def reset(self):
        self.rec = {}

    def stack(self, var: str = None) -> torch.Tensor:
        """
        Stack recorded tensors into a single tensor.

        If a specific variable name is provided, only that variable's data is stacked.
        Otherwise, all recorded variables are stacked and concatenated.

        Parameters
        ----------
        var : str, optional
            Name of the specific variable to stack. If None, all recorded variables
            are stacked and concatenated along dimension 2. Default is None.

        Returns
        -------
        torch.Tensor
            A tensor containing the stacked recorded data.
        """
        if var is not None:
            return torch.stack(self.rec.get(var, []))
        return torch.stack(
            [torch.stack(tensors) for tensors in self.rec.values()], dim=2
        )

    def numpy(self, var: str = None) -> np.ndarray:
        if var is not None:
            v = self.rec.get(var, [])
            if not v:
                return np.array([])
            return torch.stack(v).detach().cpu().numpy()
        return {
            name: torch.stack(tensors).detach().cpu().numpy()
            for name, tensors in self.rec.items()
        }


def _hdf5_write(queue: Queue, path: str):
    with File(path, "w", libver="latest") as f:
        while True:
            item = queue.get()
            if item == "flush":
                if TRANSFERSTREAM is not None:
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


class LFP(Callback):
    """
    Callback for recording Local Field Potential (LFP) signals during simulation.

    This callback records the dot product between the membrane current distribution
    and a unit vector representing the relative contribution of each compartment
    to the LFP at each time step.

    Parameters
    ----------
    v_unit : list of torch.Tensor
        List of tensors representing the unit vector for the electrode configuration.
        Each tensor should have the same shaape as model.shape == model.i_membrane.shape.
    rule : str, optional
        Einstein summation rule for computing the dot product. Default is None,
        which uses the rule '...,n...->n' which sums over all dimensions except
        the field index.

    Notes
    -----
    Requires the axon model to be compiled with fast membrane current calculation
    enabled (`IMEM=1`). Use `with dendra.ctx(IMEM=1): model = ...` when creating
    the model.

    Attributes
    ----------
    v_unit : torch.Tensor
        Tensor version of the unit vector, moved to the model's device.
    lfp : torch.Tensor
        Tensor of LFP values at each time step.
    t : torch.Tensor
        Tensor of timestamps corresponding to each LFP value.

    Examples
    --------
    >>> # Creating a point electrode 100 μm above the middle of the axon
    >>> import torch
    >>> from dendra.models import callbacks
    >>>
    >>> # Define a unit vector for a point electrode
    >>> distance = 100  # μm
    >>> r = torch.sqrt(z**2 + model.x()**2) * 1e-4
    >>> v_unit = 1000 / (4 * torch.pi * 500 * r)
    >>>
    >>> # Create the LFP callback
    >>> lfp_callback = callbacks.LFP(v_unit)
    >>>
    >>> # Run the simulation with the callback
    >>> model.run(ve, callbacks=[lfp_callback])
    >>>
    >>> # Get the LFP signal as a NumPy array
    >>> lfp_signal = lfp_callback.numpy()
    >>> # or as a PyTorch tensor
    >>> lfp_tensor = lfp_callback.lfp
    """

    def __init__(self, v_unit: list[torch.Tensor], rule=None):
        super().__init__()
        self._lfp = []
        self._t = []
        self.register_buffer("v_unit", torch.stack(v_unit))
        if rule is None:
            rule = "...,n...->n"
        self.rule = rule

    def pre_loop_hook(self, model):
        """
        Validate IMEM availability and record the baseline LFP.

        Moves ``v_unit`` to the model device, raises if the model was not
        compiled with membrane current output, and stores the t=0 dot product
        between ``model.i_membrane`` and ``v_unit``.
        """
        if not model.integrator.imem:
            raise RuntimeError(
                "Model must be compiled with IMEM=1. Use with dn.ctx(IMEM=1): model = ..."
            )
        self.v_unit = torch.as_tensor(self.v_unit, device=model.device())
        self._lfp.append(torch.einsum(self.rule, model.i_membrane, self.v_unit))
        return super().pre_loop_hook(model)

    @nojit
    def post_step_hook(self, model):
        """Append the LFP after each step via ``torch.einsum`` with ``v_unit``."""
        self._lfp.append(torch.einsum(self.rule, model.i_membrane, self.v_unit))

    @property
    def lfp(self):
        return torch.stack(self._lfp)

    @property
    def t(self):
        return torch.tensor(self._t)

    def numpy(self):
        return self.lfp.detach().cpu().numpy()

    def reset(self):
        self._lfp = []
        self._t = []
        gc.collect()


class ThresholdCallback(Callback):
    """
    Base class for callbacks that detect threshold crossings in membrane potential.

    This class provides common functionality for action potential detection by
    monitoring when membrane potential crosses a specified threshold at selected nodes.
    It serves as a base class for specialized callbacks like APCount, Active, and Raster.

    Parameters
    ----------
    threshold : float, optional
        Voltage threshold in mV to detect crossings. Default is 0.0.
    t_start_check : float, optional
        Time in ms after which to start checking for threshold crossings.
        Default is 0.0 (check from beginning).
    node_check : list of int, optional
        Indices of nodes to check for threshold crossings. Default is [5, -5]
        (check at node 5 from beginning and node 5 from end).
    dt : float, optional
        Time step in ms. If None, uses the default from backend. Default is None.

    Attributes
    ----------
    record : torch.Tensor
        Records detection results (specific format depends on subclass).
    state_cache : torch.Tensor
        Cache of state to track threshold crossings between time steps.
    threshold : float
        Voltage threshold for detection.
    t_start_check : float
        Time after which detection starts.
    node_check : list of int
        Indices of nodes being monitored.
    i : int
        Time step counter.
    dt : float
        Time step size in ms.

    """

    def __init__(
        self,
        threshold=0.0,
        t_start_check=0.0,
        t_end_check=None,
        node_check=[5, -5],
        dt=None,
    ):
        super().__init__()
        self.record: torch.Tensor = None
        self.state_cache: torch.Tensor = None
        self.threshold: float = threshold
        self.t_start_check: float = t_start_check
        self.t_end_check: float = t_end_check if t_end_check is not None else float(1e9)
        self.node_check: List[int] = node_check

        self.i: int = 0
        self._dt: float = dt if dt is not None else A.dt
        self.ind_start = int(self.t_start_check / self._dt)
        self.ind_end = int(self.t_end_check / self._dt)

    def pre_loop_hook(self, model):
        """
        Normalize and move node indices to the model device.

        Converts ``node_check`` to a tensor on the model device and wraps any
        negative indices so they reference from the end of the cable.
        """
        self.node_check = torch.as_tensor(
            self.node_check, dtype=torch.long, device=model.device()
        )

        # convert negative to positive index
        nc = model.nc
        self.node_check = self.node_check.remainder(nc)

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = value
        self.ind_start = int(self.t_start_check / self._dt)
        self.ind_end = int(self.t_end_check / self._dt)

    def reset_timer(self):
        """
        Reset the time step counter to zero.
        """
        self.i = 0

    def reset_count(self):
        """
        Reset the detection record to None.
        """
        self.record = None

    def reset_state_cache(self):
        """
        Reset the state cache used for threshold detection.
        """
        self.state_cache = None

    def reset(self):
        """
        Reset all internal state.

        This method resets the time step counter, detection record,
        and state cache all at once.
        """
        self.reset_timer()
        self.reset_count()
        self.reset_state_cache()

    def numpy(self):
        """
        Return detection results as a NumPy array.

        Returns
        -------
        numpy.ndarray or None
            Detection results as a NumPy array, or None if no results are available.
        """
        if self.record is not None:
            return self.record.detach().cpu().numpy()
        return self.record


class APCount(ThresholdCallback):
    """
    Callback for counting action potentials during axon model simulation.

    This class detects and counts the number of times membrane potential crosses above
    a specified voltage threshold at selected nodes. Each crossing is counted as an
    action potential (AP).

    Parameters
    ----------
    threshold : float, optional
        Voltage threshold in mV for AP detection. Default is 0.0.
    t_start_check : float, optional
        Time in ms after which to start checking for APs.
        Default is 0.0 (check from beginning).
    node_check : list of int, optional
        Indices of nodes to monitor for AP detection. Default is [5, -5]
        (check at node 5 from beginning and node 5 from end).
    dt : float, optional
        Time step in ms. If None, uses the default from backend. Default is None.

    See Also
    --------
    Active : Callback for detecting if axons fire at any point during simulation
    Raster : Callback for recording spike times for raster plots

    Examples
    --------
    >>> ap_counter = APCount(threshold=20.0)  # Count when v crosses +20 mV
    >>> model.run(ve, callbacks=[ap_counter])
    >>> ap_counts = ap_counter.numpy()  # Get AP counts for each axon

    Attributes
    ----------
    record : torch.Tensor
        Integer tensor of shape [n_axons, n_check_nodes] storing AP counts.
    state_cache : torch.Tensor
        Boolean tensor tracking membrane potential state relative to threshold.
    """

    def pre_loop_hook(self, model):
        """
        Allocate per-axon spike counters and state cache.

        ``record`` is created as a ``(n_axons, n_nodes_checked)`` float tensor
        and ``state_cache`` starts as ``True`` so the first upward crossing is
        counted. Negative indices in ``node_check`` have already been resolved
        by :meth:`ThresholdCallback.pre_loop_hook`.
        """
        super().pre_loop_hook(model)

        if self.record is None:
            self.record = torch.zeros(
                model.n(),
                _n(self.node_check, model),
                dtype=torch.float,
                device=model.device(),
            )
        if self.state_cache is None:
            self.state_cache = torch.ones(
                model.n(),
                _n(self.node_check, model),
                dtype=torch.bool,
                device=model.device(),
            )

    @nojit
    def post_step_hook(self, model):
        """
        Detect upward crossings and increment counters.

        After ``t_start_check`` is reached, select the monitored nodes from
        ``model.v``, update the boolean cache to reflect whether voltage is
        currently below threshold, and add one to ``record`` wherever a rising
        edge was observed this step.
        """
        if self.i >= self.ind_start and self.i < self.ind_end:
            vm_new = model.v.index_select(-1, self.node_check)
            self.state_cache, self.record = _increment_count(
                self.state_cache, vm_new, self.record, self.threshold
            )
        self.i += 1

    @property
    def n(self):
        """
        Return the number of axons being monitored.

        Returns
        -------
        int
            Number of axons in the model.
        """
        return self.record.int()


class ActiveAL(APCount):
    """
    Callback for detecting if axons fire at least a specified number of times.

    This class extends APCount to detect if the membrane potential crosses a specified
    voltage threshold at least N times during the simulation. It marks an axon as
    "active" only if the threshold is crossed at least the specified number of times.

    Parameters
    ----------
    threshold : float, optional
        Voltage threshold in mV for spike detection. Default is 0.0.
    t_start_check : float, optional
        Time in ms after which to start checking for threshold crossings.
        Default is 0.0 (check from beginning).
    node_check : list of int, optional
        Indices of nodes to monitor for spike detection. Default is [5, -5]
        (check at node 5 from beginning and node 5 from end).
    dt : float, optional
        Time step in ms. If None, uses the default from backend. Default is None.
    at_least : int, optional
        Minimum number of threshold crossings required to mark an axon as active.
        Default is 1.
    inv : bool, optional
        If True, inverts the active detection (marks axons as inactive if they
        meet the threshold). Default is False.

    See Also
    --------
    APCount : Callback for counting total spikes during simulation
    Active : Callback for detecting if axons fire at any point during simulation

    Examples
    --------
    >>> # Detect axons that fire at least 3 times
    >>> detector = ActiveAL(threshold=20.0, at_least=3)
    >>> model.run(ve, callbacks=[detector])
    >>> active_axons = detector.numpy()  # Get boolean array of active axons
    >>> active_count = active_axons.sum()  # Count how many axons fired ≥3 times

    Attributes
    ----------
    record : torch.Tensor
        Integer tensor of shape [n_axons, n_check_nodes] storing spike counts.
    state_cache : torch.Tensor
        Boolean tensor tracking membrane potential state relative to threshold.
    at_least : int
        Minimum number of threshold crossings required to mark an axon as active.
    """

    def __init__(
        self,
        threshold=0.0,
        t_start_check=0.0,
        t_end_check=None,
        node_check=[5, -5],
        dt=None,
        at_least=1,
        inv=False,
    ):
        super().__init__(threshold, t_start_check, t_end_check, node_check, dt)
        self.at_least = at_least
        self.inv = inv

    def is_active(self, partition=None):
        """
        Determine which axons are active based on spike counts.

        By default, an axon is active if the number of nonzero entries in each
        row of ``record`` is at least ``at_least``. When ``partition`` is
        provided, the check is performed independently over contiguous segments
        of each row, and a boolean mask is returned for each segment.

        Parameters
        ----------
        partition : sequence of int, optional
            Segment lengths that partition ``record`` along dimension 1. The
            sum of the sequence must equal ``record.shape[1]`` and every value
            must be at least ``at_least``. If None, the full row is used.

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape ``(n_axons,)`` when ``partition`` is None,
            otherwise ``(n_axons, len(partition))``. If ``inv`` is True, the
            result is inverted.

        Raises
        ------
        ValueError
            If ``partition`` is empty, not 1D, does not sum to
            ``record.shape[1]``, or any segment length is less than
            ``at_least``.
        """
        if self.record is None:
            return self.record
        active = _is_active(self.record, self.at_least, partition)
        if self.inv:
            return ~active
        return active

    def numpy(self, partition=None):
        if self.record is not None:
            return self.is_active(partition).detach().cpu().numpy()
        return self.record


class ActiveALCount(APCount):
    """
    Callback for determining whether an axon fired based on the total spike count.

    This class extends APCount to determine if an axon is "active" based on whether
    the total number of threshold crossings across all monitored nodes meets or
    exceeds a specified count. It marks an axon as active if the total spike count
    is at least the specified threshold.

    Parameters
    ----------
    threshold : float, optional
        Voltage threshold in mV for spike detection. Default is 0.0.
    t_start_check : float, optional
        Time in ms after which to start checking for threshold crossings.
        Default is 0.0 (check from beginning).
    node_check : list of int, optional
        Indices of nodes to monitor for spike detection. Default is [5, -5]
        (check at node 5 from beginning and node 5 from end).
    dt : float, optional
        Time step in ms. If None, uses the default from backend. Default is None.
    at_least : int, optional
        Minimum total spike count across all monitored nodes required to mark an axon as active.
        Default is 1.
    inv : bool, optional
        If True, inverts the active detection (marks axons as inactive if they
        meet the count threshold). Default is False.
    """

    def __init__(
        self,
        threshold=0.0,
        t_start_check=0.0,
        t_end_check=None,
        node_check=[5, -5],
        dt=None,
        at_least=1,
        inv=False,
    ):
        super().__init__(threshold, t_start_check, t_end_check, node_check, dt)
        self.at_least = at_least
        self.inv = inv

    def is_active(self, partition=None):
        """
        Determine which axons are active based on total spike count.

        An axon is active if the sum of counts across all monitored nodes in
        ``record`` is at least ``at_least``. When ``partition`` is provided,
        the check is performed independently over contiguous segments of each row,
        and a boolean mask is returned for each segment.

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape ``(n_axons,)`` when ``partition`` is None,
            otherwise ``(n_axons, len(partition))``. If ``inv`` is True, the
            result is inverted.
        """
        if self.record is None:
            return self.record
        active = _is_active_count(self.record, self.at_least, partition)
        if self.inv:
            return ~active
        return active

    def numpy(self, partition=None):
        if self.record is not None:
            return self.is_active(partition).detach().cpu().numpy()
        return self.record


class Active(ActiveAL):
    """
    Callback for detecting if axons fire at any point during simulation.

    This class detects if the membrane potential crosses above a specified
    voltage threshold at selected nodes at any point during the simulation.
    It subclasses ActiveAL with ``at_least=1`` to mark an axon as "active"
    (fired) as soon as the first threshold crossing is detected.

    Parameters
    ----------
    threshold : float, optional
        Voltage threshold in mV for spike detection. Default is 0.0.
    t_start_check : float, optional
        Time in ms after which to start checking for threshold crossings.
        Default is 0.0 (check from beginning).
    node_check : list of int, optional
        Indices of nodes to monitor for spike detection. Default is [5, -5]
        (check at node 5 from beginning and node 5 from end).
    dt : float, optional
        Time step in ms. If None, uses the default from backend. Default is None.
    inv : bool, optional
        If True, inverts the active detection (marks axons as inactive if they
        fired). Default is False.

    See Also
    --------
    APCount : Callback for counting total spikes during simulation
    Raster : Callback for recording spike times for raster plots

    Examples
    --------
    >>> active_detector = Active(threshold=20.0)  # Detect when v crosses +20 mV
    >>> model.run(ve, callbacks=[active_detector])
    >>> active_axons = active_detector.numpy()  # Get boolean array of active axons
    >>> active_count = active_axons.sum()  # Count how many axons fired

    Attributes
    ----------
    record : torch.Tensor
        Boolean tensor of shape [n_axons] indicating which axons fired at least once.
    state_cache : torch.Tensor
        Boolean tensor tracking membrane potential state relative to threshold.
    """

    def __init__(
        self,
        threshold=0.0,
        t_start_check=0.0,
        t_end_check=None,
        node_check=[5, -5],
        dt=None,
        inv=False,
    ):
        super().__init__(threshold, t_start_check, t_end_check, node_check, dt, inv=inv)


class _Active(ThresholdCallback):
    """
    Callback for detecting if axons fire at any point during simulation.

    This class detects if the membrane potential crosses above a specified
    voltage threshold at selected nodes at any point during the simulation.
    It marks an axon as "active" (fired) as soon as the first threshold
    crossing is detected.

    Parameters
    ----------
    threshold : float, optional
        Voltage threshold in mV for spike detection. Default is 0.0.
    t_start_check : float, optional
        Time in ms after which to start checking for threshold crossings.
        Default is 0.0 (check from beginning).
    node_check : list of int, optional
        Indices of nodes to monitor for spike detection. Default is [5, -5]
        (check at node 5 from beginning and node 5 from end).
    dt : float, optional
        Time step in ms. If None, uses the default from backend. Default is None.
    inv : bool, optional
        If True, inverts the active detection (marks axons as inactive if they
        fired). Default is False.

    See Also
    --------
    APCount : Callback for counting total spikes during simulation
    Raster : Callback for recording spike times for raster plots

    Examples
    --------
    >>> active_detector = Active(threshold=20.0)  # Detect when v crosses +20 mV
    >>> model.run(ve, callbacks=[active_detector])
    >>> active_axons = active_detector.numpy()  # Get boolean array of active axons
    >>> active_count = active_axons.sum()  # Count how many axons fired

    Attributes
    ----------
    record : torch.Tensor
        Boolean tensor of shape [n_axons] indicating which axons fired at least once.
    state_cache : torch.Tensor
        Boolean tensor tracking membrane potential state relative to threshold.
    """

    def __init__(
        self, threshold=0.0, t_start_check=0.0, node_check=[5, -5], dt=None, inv=False
    ):
        super().__init__(threshold, t_start_check, node_check, dt)
        self.inv = inv

    def pre_loop_hook(self, model):
        """
        Initialize boolean fire mask and cache for threshold detection.

        Creates a ``record`` vector (one flag per axon) and a per-node
        ``state_cache`` that tracks whether voltage was below threshold on the
        previous step.
        """
        super().pre_loop_hook(model)
        if self.record is None:
            self.record = torch.zeros(
                model.n(), dtype=torch.bool, device=model.device()
            )
        if self.state_cache is None:
            self.state_cache = torch.ones(
                model.n(),
                _n(self.node_check, model),
                dtype=torch.bool,
                device=model.device(),
            )

    @nojit
    def post_step_hook(self, model):
        """
        Mark axons as active once they cross threshold.

        After ``t_start_check`` is reached, evaluates monitored nodes for each
        axon, updates ``record`` when a rising edge occurs, and updates
        ``state_cache`` to reflect the new below-threshold mask.
        """
        if self.i >= self.ind_start:
            vm_new = model.v.index_select(-1, self.node_check)
            self.state_cache, self.record = _update_active(
                self.state_cache, vm_new, self.record, self.threshold
            )
        self.i += 1

    def is_active(self):
        if self.inv:
            return ~self.record
        return self.record

    def numpy(self):
        return self.record.detach().cpu().numpy()


class Raster(ThresholdCallback):
    """
    Callback for recording spike events for raster plot visualization.

    This class detects threshold crossings of membrane potential (spikes) at selected
    nodes and records their occurrence at each time step. The result can be used
    to create raster plots that show spiking activity across multiple axons over time.

    Parameters
    ----------
    threshold : float, optional
        Voltage threshold in mV for spike detection. Default is 0.0.
    t_start_check : float, optional
        Time in ms after which to start recording spikes.
        Default is 0.0 (record from beginning).
    node_check : list of int, optional
        Indices of nodes to monitor for spike detection. Default is [5, -5]
        (check at node 5 from beginning and node 5 from end).
    dt : float, optional
        Time step in ms. If None, uses the default from backend. Default is None.

    See Also
    --------
    APCount : Callback for counting total spikes during simulation
    Active : Callback for detecting if axons fire at any point during simulation

    Examples
    --------
    >>> raster = Raster(threshold=20.0)  # Record when v crosses +20 mV
    >>> dt = 0.005 * ms
    >>> model.run(ve, dt=dt, callbacks=[raster])
    >>> spike_data = raster.numpy()  # Get spike data for plotting
    >>>
    >>> # Plot raster
    >>> import matplotlib.pyplot as plt
    >>> fig, axis = plt.subplots(dpi=200, figsize=(5,5))
    >>> diams = model.diam.cpu().numpy()
    >>> raster.plot(diams, dt=dt, ax=axis)
    >>> plt.show()

    Attributes
    ----------
    record : list of torch.Tensor
        List of boolean tensors, one per time step, indicating which axons spiked.
    state_cache : torch.Tensor
        Boolean tensor tracking membrane potential state relative to threshold.
    """

    def pre_loop_hook(self, model):
        """
        Prepare storage for spike raster recording.

        Ensures ``node_check`` lives on the model device, initializes the list
        that will hold per-step spike masks, and seeds ``state_cache`` so the
        first upward crossings are detected.
        """
        self.node_check = torch.as_tensor(
            self.node_check, dtype=torch.long, device=model.device()
        )
        if self.record is None:
            self.record = []
        if self.state_cache is None:
            self.state_cache = torch.ones(
                model.n(),
                _n(self.node_check, model),
                dtype=torch.bool,
                device=model.device(),
            )

    @nojit
    def post_step_hook(self, model):
        """
        Append a boolean spike mask for the current step.

        After ``t_start_check`` is reached, selects the monitored nodes,
        computes rising-edge events with :func:`_increment_act`, updates
        ``state_cache``, and appends the resulting activity mask to ``record``.
        """
        if self.i >= self.ind_start:
            vm_new = _atleast_2d(model.v[:, self.node_check])
            vm = self.state_cache
            self.state_cache, la = _increment_act(vm, vm_new, self.threshold)
            self.record.append(la)
        self.i += 1

    def stack(self):
        return torch.stack(self.record)

    def numpy(self):
        return self.stack().detach().cpu().numpy()

    def plot(
        self,
        var,
        varname: str,
        dt=None,
        ax=None,
        cmap=None,
        node_idx=None,
        axon_idx=None,
    ):
        """
        Plot the raster plot of spiking activity.

        Parameters
        ----------
        var : np.ndarray
            Array of fiber diameters in μm. Must match the number of axons in the model.
        varname: str
            Name of the variable to plot.
        dt : float, optional
            Time step in ms. If None, uses the default from backend. Default is None.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure and axes. Default is None.
        cmap : str or matplotlib colormap, optional
            Colormap for the plot. Default is None (uses default colormap).
        node_idx : int, optional
            Index of the node to plot. If None, plots all nodes. Default is None.
        axon_idx : int, optional
            Index of the axon to plot. If None, plots all axons. Default is None.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the raster plot.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        if dt is None:
            dt = self.dt
        if node_idx is None:
            node_idx = 0
        if axon_idx is None:
            axon_idx = slice(None)
        if cmap is None:
            cmap = plt.cm.viridis
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        binary_array = self.numpy()[:, axon_idx, node_idx].T
        if ax is None:
            fig, ax = plt.subplots(dpi=300, figsize=(10, 6))

        num_fibers, num_timepoints = binary_array.shape
        assert len(var) == num_fibers, (
            "Length of var must match the number of axons selected to plot."
        )

        # Build a list of spike-time arrays. Each entry in data_for_eventplot
        # corresponds to a single fiber's event times.
        data_for_eventplot = []
        for row_idx in range(num_fibers):
            spike_times = np.where(binary_array[row_idx] == 1)[0] * dt
            data_for_eventplot.append(spike_times)

        norm = mpl.colors.Normalize(vmin=var.min(), vmax=var.max())

        # Convert each diameter to an RGBA color using the colormap + normalization
        line_colors = [cmap(norm(d)) for d in var]
        ax.eventplot(
            data_for_eventplot,
            lineoffsets=var,
            linelengths=0.8 * min(var[1:] - var[:-1]),
            colors=line_colors,
        )
        tstop = num_timepoints * dt
        range_var = var.max() - var.min()
        y_min = var.min() - 0.02 * range_var
        y_max = var.max() + 0.02 * range_var
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0 - 0.02 * tstop, tstop + 0.02 * tstop)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(f"{varname}")
        return ax


def _increment_count(vm, vm_new, record, threshold: float):  # pragma: no cover
    m = vm_new >= threshold
    mask = m & vm  # fused compare + and
    record = record + mask.to(record.dtype)  # one fused kernel
    next_mask = ~m  # can tag-on to same kernel
    return next_mask, record


def _increment_act(vm, vm_new, threshold: float):  # pragma: no cover
    m = vm_new >= threshold
    mask = m & vm  # fused compare + and
    next_mask = ~m  # can tag-on to same kernel
    return next_mask, mask


def _update_active(
    vm, vm_new, record, threshold: float
) -> Tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
    ge = vm_new >= threshold
    record = torch.logical_or(record, torch.any(torch.logical_and(ge, vm), dim=1))
    return ~ge, record


def _is_active(record, at_least: int, partition=None) -> torch.Tensor:
    if partition is None:
        return torch.count_nonzero(record, dim=1) >= at_least

    lengths = torch.as_tensor(partition, dtype=torch.long, device="cpu")
    if lengths.dim() != 1:
        raise ValueError("partition must be a 1D sequence of integers.")

    lengths_list = lengths.tolist()
    if not lengths_list:
        raise ValueError("partition must be non-empty.")
    if sum(lengths_list) != record.shape[1]:
        raise ValueError(
            "sum(partition) must equal record.shape[1]; "
            f"got {sum(lengths_list)} and {record.shape[1]}."
        )
    if min(lengths_list) < at_least:
        raise ValueError(
            "all partition values must be at least `at_least`; "
            f"minimum was {min(lengths_list)}."
        )

    segments = torch.split(record, lengths_list, dim=1)
    return torch.stack(
        [torch.count_nonzero(seg, dim=1) >= at_least for seg in segments], dim=1
    )


def _is_active_count(record, at_least: int, partition=None) -> torch.Tensor:
    if partition is None:
        return record.sum(dim=1) >= at_least

    lengths = torch.as_tensor(partition, dtype=torch.long, device="cpu")
    if lengths.dim() != 1:
        raise ValueError("partition must be a 1D sequence of integers.")

    lengths_list = lengths.tolist()
    if not lengths_list:
        raise ValueError("partition must be non-empty.")
    if sum(lengths_list) != record.shape[1]:
        raise ValueError(
            "sum(partition) must equal record.shape[1]; "
            f"got {sum(lengths_list)} and {record.shape[1]}."
        )
    if min(lengths_list) < at_least:
        raise ValueError(
            "all partition values must be at least `at_least`; "
            f"minimum was {min(lengths_list)}."
        )

    segments = torch.split(record, lengths_list, dim=1)
    return torch.stack([seg.sum(dim=1) >= at_least for seg in segments], dim=1)


def _sliding_window_average(x, window_size: int):
    """
    Compute the sliding (moving) window average along axis 0 for a tensor,
    with padding so that the output has the same shape as the input.

    The function pads the input along axis 0 using edge replication and then
    computes the average over every consecutive window. All remaining dimensions
    are treated as independent feature channels and the output shape matches the input.

    Parameters
    ----------
    x : torch.Tensor
        A tensor with time or samples along axis 0.
    window_size : int
        The size of the sliding window (must be >= 1).

    Returns
    -------
    out : torch.Tensor
        The sliding window averages computed along axis 0 with the same shape as the input.

    Raises
    ------
    ValueError
        If window_size is less than 1.
    TypeError
        If x is not a PyTorch tensor.
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
    if not torch.is_tensor(x):
        raise TypeError("x must be a PyTorch tensor.")
    if x.ndim < 1 or x.shape[0] == 0:
        raise ValueError("x must contain at least one sample along axis 0.")

    # Treat every dimension after time as an independent channel. Recorder
    # outputs are commonly 3-D, while other callers may use 4-D tensors.
    N = x.shape[0]
    feature_shape = x.shape[1:]
    channels = int(x[0].numel())

    # Manually pad along axis 0 (the N dimension) using replication.
    # For pad_left, replicate the first slice; for pad_right, replicate the last slice.
    left_pad = (
        x[0:1].expand((pad_left, *feature_shape))
        if pad_left > 0
        else torch.empty((0, *feature_shape), device=x.device, dtype=x.dtype)
    )
    right_pad = (
        x[-1:].expand((pad_right, *feature_shape))
        if pad_right > 0
        else torch.empty((0, *feature_shape), device=x.device, dtype=x.dtype)
    )
    # Concatenate along dimension 0.
    x_padded = torch.cat([left_pad, x, right_pad], dim=0)
    N_padded = x_padded.shape[0]  # should equal N + (window_size - 1)

    # Reshape so that the padded time dimension is the convolution length.
    x_reshaped = x_padded.movedim(0, -1).reshape(1, channels, N_padded)

    # Create an averaging kernel for each flattened feature channel.
    kernel = (
        torch.ones(channels, 1, window_size, dtype=x.dtype, device=x.device)
        / window_size
    )

    # Perform grouped convolution along the length dimension.
    out_conv = F.conv1d(x_reshaped, kernel, groups=channels)
    L = out_conv.shape[-1]
    if L != N:
        raise RuntimeError(f"Unexpected output length: got {L}, expected {N}.")

    return out_conv.squeeze(0).reshape(*feature_shape, N).movedim(-1, 0)


# Public alias retained for tests and user code; Recorder uses the private name internally.
sliding_window_average = _sliding_window_average
