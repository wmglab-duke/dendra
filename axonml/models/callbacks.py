from queue import Queue
from typing import Tuple, Dict, List
from types import MethodType
import threading
import multiprocessing as mp

from h5py import File

import numpy as np
import torch
import torch.nn.functional as F

from .backend import Backend as A
from .interfaces import AxonInterface


TRANSFERSTREAM = torch.cuda.Stream()


class Callback:
    """
    Base class for simulation callbacks in AxonML.

    This class defines the interface for callbacks that can be registered with
    axon models to monitor and interact with the simulation at specific points
    in the execution flow. Subclasses should override the hook methods to
    implement specific functionality.

    Methods
    -------
    pre_loop_hook(model)
        Called once before starting the simulation loop.
    post_step_hook(model)
        Called after each simulation time step.
    post_loop_hook(model)
        Called once after the simulation loop completes.

    Notes
    -----
    Custom callbacks should inherit from this class and override one or more
    of the hook methods. Multiple callbacks can be used simultaneously by
    passing them in a list to the model's run method.

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
      self.rec['{full_state}'].append(torch.amax(states, -1, keepdim=True))
    else:
      if self.node_indices is not None:
        self.rec['{full_state}'].append(states[:, :, self.node_indices])
      else:
        self.rec['{full_state}'].append(states)
"""

m_template = """
    states = model.{val}
    if self.max_only:
      self.rec['{val}'].append(torch.amax(states, -1, keepdim=True))
    else:
      if self.node_indices is not None:
        self.rec['{val}'].append(atleast_3d(states[:, :, self.node_indices]))
      else:
        self.rec['{val}'].append(states)
"""


def is_state(s):
    return "." in s


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
        return model.n_comp
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

    Notes
    -----
    For large-scale simulations, use the set_hdf5 method to enable caching to an
    HDF5 file, which helps manage memory usage.
    """

    def __init__(
        self, states, max_only=False, node_indices=None, dt=None, sliding_window=None
    ):
        super().__init__()
        self.states = states
        self.rec: Dict[str, List[torch.Tensor]] = {s: [] for s in states}
        self.max_only: bool = max_only
        self.node_indices = node_indices
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
        self.hdf5_path = hdf5
        mp.set_start_method("spawn", force=True)
        self.manager = mp.Manager()
        self.queue = self.manager.Queue()
        self.writer_thread = mp.Process(
            target=_hdf5_write, args=(self.queue, self.hdf5_path)
        )
        self.writer_thread.start()
        self.cache_with_hdf5 = True
        self.cache_every = cache_every
        return self

    def pre_loop_hook(self, model):
        self.dt = float(model.dt)
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


def _hdf5_write(queue: Queue, path: str):
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


class LFP(Callback):
    """
    Callback for recording Local Field Potential (LFP) signals during simulation.

    This callback records the dot product between the membrane current distribution
    and a unit vector representing the relative contribution of each compartment
    to the LFP at each time step.

    Parameters
    ----------
    v_unit : array_like
        Unit vector representing the contribution of each compartment to the LFP
        measurement. Shape should match the axon's membrane current distribution.

    Attributes
    ----------
    v_unit : torch.Tensor
        Tensor version of the unit vector, moved to the model's device.
    lfp : torch.Tensor
        Tensor of LFP values at each time step.
    t : torch.Tensor
        Tensor of timestamps corresponding to each LFP value.

    Notes
    -----
    Requires the axon model to be compiled with fast membrane current calculation
    enabled (`IMEM=1`). Use `with axonml.helpers.ctx(IMEM=1): model = ...` when creating
    the model.

    Examples
    --------
    >>> # Creating a point electrode 100 μm above the middle of the axon
    >>> import torch
    >>> from axonml.models import callbacks
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

    def __init__(self, v_unit):
        super().__init__()
        self._lfp = []
        self._t = []
        self.v_unit = v_unit

    def pre_loop_hook(self, model):
        if not model.use_fast_imem:
            raise RuntimeError(
                "Axon must be compiled with IMEM=1. Use with axonml.helpers.ctx(IMEM=1): model = ..."
            )
        self.v_unit = torch.as_tensor(self.v_unit, device=model.device())
        self._lfp.append(
            torch.einsum(
                "ij,ij->", torch.atleast_2d(model.i_membrane.squeeze()), self.v_unit
            )
        )
        self._t.append(model.t)
        return super().pre_loop_hook(model)

    def post_step_hook(self, model):
        self._lfp.append(
            torch.einsum(
                "ij,ij->", torch.atleast_2d(model.i_membrane.squeeze()), self.v_unit
            )
        )
        self._t.append(model.t)

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

    Methods
    -------
    reset_timer()
        Reset the time step counter.
    reset_count()
        Reset the detection record.
    reset_state_cache()
        Reset the state cache used for detection.
    reset()
        Reset all internal state (timer, count, and cache).
    numpy()
        Return detection results as a NumPy array.
    """

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

    Attributes
    ----------
    record : torch.Tensor
        Integer tensor of shape [n_axons, n_check_nodes] storing AP counts.
    state_cache : torch.Tensor
        Boolean tensor tracking membrane potential state relative to threshold.

    Methods
    -------
    pre_loop_hook(model)
        Initialize record and state_cache tensors.
    post_step_hook(model)
        Update AP counts by detecting threshold crossings.
    reset()
        Reset AP counters and internal state.
    numpy()
        Return AP counts as a NumPy array.

    See Also
    --------
    Active : Callback for detecting if axons fire at any point during simulation
    Raster : Callback for recording spike times for raster plots

    Examples
    --------
    >>> ap_counter = APCount(threshold=20.0)  # Count when v crosses +20 mV
    >>> model.run(ve, callbacks=[ap_counter])
    >>> ap_counts = ap_counter.numpy()  # Get AP counts for each axon
    """

    def pre_loop_hook(self, model: AxonInterface):
        """
        Initialize record and state_cache tensors before simulation.

        Parameters
        ----------
        model : AxonInterface
            The axon model being simulated.
        """
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
        """
        Update AP counts after each simulation step.

        Parameters
        ----------
        model : AxonInterface
            The axon model being simulated.
        """
        if self.i * self.dt >= self.t_start_check:
            vm_new = atleast_2d(model.v[:, 0, self.node_check].squeeze())
            vm = self.state_cache
            self.state_cache = increment_count_(vm, vm_new, self.record, self.threshold)
        self.i += 1


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

    Attributes
    ----------
    record : torch.Tensor
        Integer tensor of shape [n_axons, n_check_nodes] storing spike counts.
    state_cache : torch.Tensor
        Boolean tensor tracking membrane potential state relative to threshold.
    at_least : int
        Minimum number of threshold crossings required to mark an axon as active.

    Methods
    -------
    is_active()
        Return boolean tensor indicating which axons fired at least the required number of times.
    numpy()
        Return activity status as a NumPy array.

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
    """

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

    Attributes
    ----------
    record : torch.Tensor
        Boolean tensor of shape [n_axons] indicating which axons fired at least once.
    state_cache : torch.Tensor
        Boolean tensor tracking membrane potential state relative to threshold.

    Methods
    -------
    pre_loop_hook(model)
        Initialize record and state_cache tensors.
    post_step_hook(model)
        Update axon activity status at each simulation time step.
    is_active()
        Return boolean tensor indicating which axons are active.
    numpy()
        Return activity status as a NumPy array.

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
    """

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

    Attributes
    ----------
    record : list of torch.Tensor
        List of boolean tensors, one per time step, indicating which axons spiked.
    state_cache : torch.Tensor
        Boolean tensor tracking membrane potential state relative to threshold.

    Methods
    -------
    pre_loop_hook(model)
        Initialize record list and state_cache tensor.
    post_step_hook(model)
        Update spike record at each simulation time step.
    stack()
        Stack recorded tensors into a single tensor.
    numpy()
        Return spike events as a NumPy array.
    reset()
        Reset spike record and internal state.

    See Also
    --------
    APCount : Callback for counting total spikes during simulation
    Active : Callback for detecting if axons fire at any point during simulation

    Examples
    --------
    >>> raster = Raster(threshold=20.0)  # Record when v crosses +20 mV
    >>> model.run(ve, callbacks=[raster])
    >>> spike_data = raster.numpy()  # Get spike data for plotting
    >>>
    >>> # Plot raster
    >>> import matplotlib.pyplot as plt
    >>> fig, axis = plt.subplots(dpi=200, figsize=(5,5))
    >>> diams = model.diam.cpu().numpy()
    >>> raster.plot(diams, dt=model.dt, ax=axis)
    >>> plt.show()
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
    
    def plot(self, 
             var,
             varname: str,
             dt=None, 
             ax=None, 
             cmap=None, 
             node_idx=None, 
             axon_idx=None):
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
        import matplotlib.pyplot as plt
        import matplotlib as mpl

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
        assert len(var) == num_fibers, "Length of var must match the number of axons selected to plot."

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
            linelengths=0.8*min(var[1:]-var[:-1]), 
            colors=line_colors)
        tstop = num_timepoints * dt
        range_var = var.max() - var.min()
        y_min = var.min() - 0.02 * range_var
        y_max = var.max() + 0.02 * range_var
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0-0.02*tstop, tstop+0.02*tstop)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(f"{varname}")
        return ax


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
