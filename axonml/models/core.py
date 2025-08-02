import itertools
import math
import re
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm.auto import tqdm

from axonml.helpers import (
    BACKEND,
    COMPILE_MODE,
    DYNAMIC,
    FULLGRAPH,
    JIT,
    op_mc,
    op_sc,
    ve_from_s_t,
)
from axonml.models.backend import Backend as A
from axonml.models.callbacks import Callback, CallbackList
from axonml.models.graph import get_area_from_graph
from axonml.models.integrators import bwd_euler_sc, bwd_euler_ub
from axonml.models.mechanisms._handler import MechanismHandler
from axonml.models.mechanisms._ions import Ion, valid_ions
from axonml.models.mechanisms.validate import validate
from axonml.models.parametric import Parameterized as P
from axonml.models.stim.intrastim import IntraStim
from axonml.models.stim.waveform import Waveform
from axonml.units import mm

from .slice import Sliceable


def get_unique_keys(list_of_dicts):
    """
    Get all unique keys from a list of dictionaries.

    Parameters
    ----------
    list_of_dicts : list
        A list of dictionaries.

    Returns
    -------
    set
        A set of unique keys.
    """
    unique_keys = set()
    for dictionary in list_of_dicts:
        unique_keys.update(dictionary.keys())
    return unique_keys


def follows_pattern(base_pattern, target_string):
    regex_pattern = (
        r"\b"
        + r"\b.*?\b".join(re.escape(part) for part in base_pattern.split("."))
        + r"\b"
    )
    return re.search(regex_pattern, target_string) is not None


def matches_any_pattern(base_patterns, target_string):
    """
    Checks if a target string matches any of the provided base patterns.

    A match occurs if the parts of a base_pattern (split by '.') appear in order
    in the target_string. All parts except the last must be whole words.
    The last part can be a prefix of a word.

    For example:
    - base_pattern 'hh.gbar' will match target_string 'hh.gbar_default'.
    - base_pattern 'foo' will match 'a.foo_bar'.
    - base_pattern 'a.b' will NOT match 'a_b.c'.
    """
    for base_pattern in base_patterns:
        # Split the pattern by '.' and escape each part to treat special
        # regex characters (like '.') as literal characters.
        escaped_parts = [re.escape(part) for part in base_pattern.split(".")]

        # The separator `\b.*?\b` ensures that all intermediate parts are
        # treated as whole words.
        regex_pattern = (
            r"\b"  # The pattern must start at a word boundary.
            + r"\b.*?\b".join(escaped_parts)
            # The final r"\b" is removed from here!
        )

        if re.search(
            regex_pattern, target_string, re.IGNORECASE
        ):  # Added re.IGNORECASE for more robust matching
            return True

    return False


class SymmetricConv1D(torch.nn.Conv1d):
    def forward(self, x):
        if self.training:
            weight_ = (self.weight + torch.flip(self.weight, [-1])) / 2
        else:
            weight_ = self.weight
        return self._conv_forward(x, weight_, self.bias)


def step(integrator, model, dt, ve=None, intra=None):
    integrator.step(model, dt, ve, intra)


@torch.compile(dynamic=True)
def make_intra(intra, stims, indices):
    return intra(stims, indices)


class Population(P, Sliceable):
    """
    Base class for a population of multicompartment neurons.
    """

    P.RANGE(cm=1.0, rhoa=35.4)
    P.GLOBAL(celsius=37.0)

    def __init__(self, N: int, C: int, integrator=None, v_init=-65.0, **kwargs):
        super().__init__((N, C), **kwargs)
        self.np = N
        self.nc = C
        self.v_init = v_init

        self.is_built = False
        self.key = None

        if integrator is None:
            integrator = bwd_euler_sc()

        self.shape = integrator.shape(self.np, self.nc) if integrator else (N, C)

        self.register_buffer("_dummy", torch.zeros(1))

        self.register_buffer("v", torch.full(self.shape, self.v_init))
        self.register_buffer("diam", torch.full(self.shape, 500.0))
        self.register_buffer("dx", torch.full(self.shape, 100.0))
        self.register_buffer("t", torch.zeros(()))

        # compiler stuff
        self.backend = BACKEND.value
        self.fullgraph = bool(FULLGRAPH)
        self.dynamic = bool(DYNAMIC)
        self.jit = bool(JIT)
        self.compile_mode = COMPILE_MODE.value

        self.integrator = integrator

        self.injections = []
        self.intra = None

        self._mech_data = {}
        self._mech_everywhere = {}

        self._labels = {}

        self._m_list = []
        self._m_name = []
        self._m_keys = []
        self._m_curr = {}
        self._m_shape = {}
        self._m_unfactorable = {}
        self._m_count = {}
        self._m_has_gtot = {}
        self._m_divide_by_two = {}

        self._ion_read = {}
        self._ion_write = {}
        self._ion_write_c = {}

        self._all_read = {}
        self._all_write = {}
        self._all_write_c = {}

        self._ion_style = {}

        self.pre_initialize_hooks: List[Callable] = []
        self.post_initialize_hooks: List[Callable] = []

        torch._dynamo.reset()

        if self.jit:
            self._step = torch.compile(
                step,
                backend=self.backend,
                fullgraph=self.fullgraph,
                dynamic=self.dynamic,
                mode=self.compile_mode,
            )
        else:
            self._step = torch.compile(
                step,
                backend="eager",
            )

        self._caches = {}

        self.register_buffer("x", torch.zeros(self.shape))
        self.register_buffer("y", torch.zeros(self.shape))
        self.register_buffer("z", torch.zeros(self.shape))

        self.initialized: bool = False
        self.eval()

    @property
    def graph(self):
        """
        Returns the graph of the population.
        This is a placeholder for future graph-related functionality.
        """
        return None

    @property
    def area(self):
        """
        Returns the area of the population.
        This is a placeholder for future area-related functionality.
        """
        if self.graph is not None:
            area = get_area_from_graph(self.graph)
            if area is not None:
                return area.to(self.device(), dtype=self.dtype())
        return self.diam * 1e-4 * torch.pi * self.dx * 1e-4  # in cm²

    @property
    def mech(self):
        return self.integrator.mech

    @property
    def i_membrane(self):
        return self.integrator.i_membrane

    def collect_parameters(self, *names):
        """
        Collects parameters from the model based on the provided names.

        Parameters
        ----------
        *names : str
            Variable length argument list of parameter name patterns.
            If empty, all parameters will be collected.
            Otherwise, only parameters matching any of these patterns will be collected.

        Returns
        -------
        list
            List of parameters matching the provided names.
        """
        if not names:
            return self.parameters()
        else:
            for n, p in self.named_parameters():
                if matches_any_pattern(names, n):
                    yield p

    def collect_named_parameters(self, *names):
        """
        Collects parameters from the model based on the provided names.

        Parameters
        ----------
        *names : str
            Variable length argument list of parameter name patterns.
            If empty, all parameters will be collected.
            Otherwise, only parameters matching any of these patterns will be collected.

        Returns
        -------
        list
            List of tuples (name, parameter) matching the provided names.
        """
        if not names:
            return self.named_parameters()
        else:
            for n, p in self.named_parameters():
                if matches_any_pattern(names, n):
                    yield (n, p)

    def unfreeze(self, *names):
        """
        Unfreezes model parameters, making them trainable.

        If no names are provided, all parameters will be unfrozen.
        If names are provided, only parameters whose names match any
        of the provided patterns will be unfrozen.

        Parameters
        ----------
        *names : str
            Variable length argument list of parameter name patterns.
            If empty, all parameters will be unfrozen.
            Otherwise, only parameters matching any of these patterns will be unfrozen.

        Notes
        -----
        The matching is done using the `matches_any_pattern` function.
        When a parameter is unfrozen, a message is printed to the console.
        """
        if not names:
            for p in self.parameters():
                p.requires_grad = True
        else:
            for n, p in self.named_parameters():
                if matches_any_pattern(names, n):
                    print(f"Unfreezing {n}")
                    p.requires_grad = True
        return self

    def unfreeze_(self, *names):
        self.unfreeze(*names)

    def unfreeze_group(self, *groups):
        for g in groups:
            group = getattr(self, g)
            print(f"Unfreezing group '{g}'")
            for p in group:
                p.requires_grad = True
        return self

    def unfreeze_group_(self, *groups):
        self.unfreeze_group(*groups)

    def freeze(self, *names):
        if not names:
            for p in self.parameters():
                p.requires_grad = False
        else:
            for n, p in self.named_parameters():
                if matches_any_pattern(names, n):
                    print(f"Freezing {n}")
                    p.requires_grad = False
        return self

    def freeze_(self, *names):
        self.freeze(*names)

    def freeze_group(self, *groups):
        for g in groups:
            group = getattr(self, g)
            print(f"Freezing group '{g}'")
            for p in group:
                p.requires_grad = False
        return self

    def freeze_group_(self, *groups):
        self.freeze_group(*groups)

    def register_post_initialize_hook(self, fn: Callable):
        self.post_initialize_hooks.append(fn)

    def register_pre_initialize_hook(self, fn: Callable):
        self.pre_initialize_hooks.append(fn)

    def device(self):
        return self._dummy.device

    def dtype(self):
        return self._dummy.dtype

    def prep_intra(self, intra, n, dt):
        start = self.t
        end = start + n * dt
        t_ = torch.arange(start, end, dt, device=self.device(), dtype=self.dtype())
        t_ = t_.to(self.device(), dtype=self.dtype())
        stims, indices = intra.init(t_)
        stims = [s.unbind(0) for s in stims]
        return stims, indices

    def run(
        self,
        ve=None,
        space=None,
        time=None,
        tstop=None,
        dt=None,
        callbacks=None,
        progressbar=False,
        multicontact=False,
    ):
        """
        Run the axon model simulation.

        Parameters
        ----------
        ve : Tensor, optional
            Extracellular voltage tensor. Shape should be
            [timesteps, n_ax, 1, n_comp] or compatible.
        space : Tensor, optional
            Spatial components when ve is not directly provided.
            Used with time to construct ve.
        time : Tensor or Waveform, optional
            Temporal components when ve is not directly provided.
            Used with space to construct ve.
        tstop : float, optional
            Simulation stop time in milliseconds. If None, uses the default from backend.
            If ve is provided, this is ignored.
            If ve is provided, tstop is determined by the shape of ve.
        dt : float, optional
            Time step size in milliseconds. If None, uses the default from backend.
        callbacks : list of Callback, optional
            List of callback objects to execute during simulation steps.
        progressbar : bool or tqdm, optional
            If True, displays a progress bar during simulation. Can also be a
            tqdm instance for custom progress tracking. Default is True.
        multicontact : bool, optional
            If True, handles multiple electrode contacts for ve construction.
            Default is False.
        Raises
        ------
        ValueError
            If neither ve nor (space and time) nor intra is provided.
            If intra is provided but is not an instance of IntraStim.

        Notes
        -----
        The simulation updates the model's internal state (v, v_prev for DF method, etc.)
        and advances the model's time.
        """
        if not self.initialized:
            raise ValueError("Model must be initialized before running.")

        if self.intra is None:
            intra = self.build_intra()
            self.intra = intra
        else:
            intra = self.intra

        with_intra = intra is not None

        device = self.device()

        if ve is not None:
            ve = torch.as_tensor(ve, device=device)

        dt = dt if dt is not None else A.dt
        dt_f = float(dt)

        dt = torch.tensor(dt, device=device, dtype=self.dtype())

        local_ind = 0

        if isinstance(time, Waveform):
            time = time.to(device, dtype=self.dtype())
            time = time.assemble(self.t, self.t + tstop, dt)

        ctx = nullcontext() if self.training else torch.no_grad()

        tstart = self.t.item()

        with ctx:
            if ve is None:
                if space is not None and time is not None:
                    ve = ve_from_s_t(space, time, self.np, self.device(), multicontact)

            if ve is not None:
                n = ve.shape[0]
            else:
                n = int(tstop / dt_f)

            if with_intra:
                stims, indices = self.prep_intra(intra, n, dt_f)

            if not isinstance(callbacks, CallbackList):
                callbacks = CallbackList(callbacks)

            if callbacks:
                for c in callbacks:
                    c.dt = dt_f

            pre_loop_hook(callbacks, self)

            if (
                not self.integrator.initialized
                or self.integrator.dt != dt_f
                or self.training
            ):
                self.integrator.initialize(self, dt)

            if progressbar:
                if not isinstance(progressbar, tqdm):
                    progressbar = tqdm(
                        total=n, desc=f"{tstart + local_ind * dt_f:.1f} ms"
                    )

            for i in range(n):
                ve_c = ve[i] if ve is not None else None

                if with_intra:
                    s = [st[local_ind] for st in stims]
                    intra_c = make_intra(intra, s, indices)
                else:
                    intra_c = None

                self._step(self.integrator, self, dt, ve_c, intra_c)
                self.t = self.t + dt

                post_step_hook(callbacks, self)
                local_ind += 1

                if progressbar:
                    progressbar.update(1)
                    if local_ind % 100 == 0:
                        progressbar.set_description(
                            f"{tstart + local_ind * dt_f:.1f} ms"
                        )

            if progressbar:
                progressbar.close()

            post_loop_hook(callbacks, self)

    def longrun(
        self,
        tstop: float,
        chunklength: int,
        dt: float = None,
        extra: Optional[Tuple[Tensor, Waveform]] = None,
        callbacks: List[Callback] = None,
        progressbar=True,
        multicontact=False,
    ):
        """
        Run a long simulation by dividing it into multiple smaller chunks.

        This method splits the overall simulation into chunks of a given length,
        allowing for more efficient memory management during long simulations.
        The model's state (e.g., voltage variables, v_prev, etc.) is maintained
        between chunks, ensuring continuity across the entire simulation period.

        Parameters
        ----------
        tstop : float
            The simulation end time in milliseconds.
        chunklength : int
            The number of time steps to process in each chunk.
        dt : float, optional
            The simulation time step in milliseconds. If None, the default value
            from the backend will be used.
        extra : tuple of (Tensor, Waveform), optional
            A tuple containing extra input parameters:
            - The first element (ve_s) is a tensor representing spatial voltage components.
            - The second element (time) is either a Waveform object or a tensor representing time.
            These values are used to construct the extracellular voltage.
        callbacks : list of Callback, optional
            A list of callback objects to be executed during simulation, allowing for
            customized processing at various stages (e.g., pre-loop, post-step, post-loop).
        progressbar : bool or tqdm, optional
            If True (or if a tqdm instance is provided), displays a progress bar to
            track simulation progress across chunks.
        multicontact : bool, optional
            If True, configures the handling of multiple electrode contacts for
            constructing the extracellular voltage input.

        Returns
        -------
        None

        Notes
        -----
        - When `extra` is provided, the method uses it to assemble the extracellular
        voltage (ve) for the simulation.
        - Chunk processing helps manage memory usage during extended simulations by
        processing data in manageable segments.
        """

        if not self.initialized:
            raise ValueError("Model must be initialized before running.")

        # ve_s : [n_ax, n_comp] or [1, n_comp] or [n_contacts, *]
        # ve_t : [n_ax, n_timesteps] or [1, n_timesteps] or [n_contacts, *]

        intra = self.intra

        with_intra = intra is not None
        with_extra = extra is not None

        dt = dt if dt is not None else A.dt
        dt_f = float(dt)

        if with_extra:
            ve_s, time = extra
            ve_s = torch.as_tensor(
                ve_s, device=self.device(), dtype=self.dtype()
            ).contiguous()

            if multicontact:
                ve_s = ve_s.expand(-1, self.n_ax, -1)
            else:
                ve_s = ve_s.expand(self.n_ax, -1)

            if isinstance(time, Waveform):
                time = time.to(device=self.device(), dtype=self.dtype())
                functional = True
            else:
                time = torch.as_tensor(time, device=self.device(), dtype=self.dtype())
                functional = False

                if multicontact:
                    time = time.expand(-1, self.n_ax, -1)
                else:
                    time = time.expand(self.n_ax, -1)

        dt = torch.tensor(dt, device=self.device(), dtype=self.dtype())

        with torch.nn.utils.parametrize.cached():
            with torch.set_grad_enabled(self.training):
                t = torch.arange(
                    self.t, self.t + tstop, dt, dtype=self.dtype(), device=self.device()
                )
                n_chunks = math.ceil(len(t) / chunklength)

                t_c_f = torch.tensor_split(t, n_chunks)

                if with_extra:
                    if functional:
                        t_chunks = t_c_f
                    else:
                        t_chunks = torch.tensor_split(time, n_chunks, dim=-1)

                if multicontact:
                    einsum = op_mc
                else:
                    einsum = op_sc

                if callbacks:
                    for c in callbacks:
                        c.dt = dt_f

                callbacks = CallbackList(callbacks)

                if progressbar:
                    progressbar = tqdm(total=n_chunks, desc=f"{self.t.item():.1f} ms")

                self.integrator.initialize(self, dt)
                einsum = torch.compile(einsum)

                pre_loop_hook(callbacks, self)

                for i in range(n_chunks):
                    if with_intra:
                        stims, indices = intra.init(t_c_f[i])
                        stims = [s.unbind(0) for s in stims]

                    if with_extra:
                        if functional:
                            t = time(t_chunks[i]).to(self.dtype())
                            if multicontact:
                                t = t.unsqueeze(0).expand(-1, self.n_ax, -1)
                            else:
                                t = t.expand(self.n_ax, -1)
                        else:
                            t = t_chunks[i]
                        ve_ = einsum(ve_s, t).contiguous().unbind(dim=0)

                    for j in range(len(t_c_f[i])):
                        if with_extra:
                            ve_c = ve_[j]
                        else:
                            ve_c = None
                        if with_intra:
                            s = [st[j] for st in stims]
                            intra_c = make_intra(intra, s, indices)
                        else:
                            intra_c = None

                        self._step(self.integrator, self, dt, ve_c, intra_c)
                        post_step_hook(callbacks, self)

                        self.t = self.t + dt

                    if progressbar:
                        progressbar.update(1)
                        progressbar.set_description(f"{self.t.item():.1f} ms")

                post_loop_hook(callbacks, self)

                if progressbar:
                    progressbar.close()

    def steady_state(self, dt=0.2, tstop=200.0, with_ve=True, with_intra=True):
        """
        Run the model until it reaches a steady state and cache the result.

        Parameters
        ----------
        dt : float, optional
            Time step size in milliseconds. Default is 0.2 ms.
        t : float, optional
            Time in milliseconds to run the simulation. Default is 200 ms.

        Notes
        -----
        This method clears any previous steady state cache before creating a new one.
        The steady state can be restored later by calling model.initialize().
        """

        self.clear_steady_state()

        self.initialize()
        self.integrator.initialize(self, dt)

        maxiter = int(tstop / dt)

        with torch.no_grad():
            if with_ve:
                ve = torch.zeros_like(self.v).contiguous()
            else:
                ve = None

            if with_intra:
                intra = torch.zeros_like(self.v).contiguous()
            else:
                intra = None

            dt = torch.tensor(dt, device=self.device(), dtype=self.dtype())
            for _ in tqdm(range(maxiter), desc="Steady state "):
                self._step(self.integrator, self, dt, ve, intra)

        self.cache("_steady_state")
        self.t.detach().zero_()
        return self

    def clear_steady_state(self):
        if "_steady_state" in self._caches:
            self._caches.pop("_steady_state")

    def post_initialize(self):
        with torch.no_grad():
            for h in self.post_initialize_hooks:
                h(self)

    def pre_initialize(self):
        with torch.no_grad():
            for h in self.pre_initialize_hooks:
                h(self)

    def populate(self):
        """
        Populate the model with mechanisms and parameters.

        This method is called after the model is built to ensure that all
        mechanisms and parameters are properly initialized and ready for use.
        It should be called after the model's build() method.
        """
        self.populate_parameter_buffers()
        self.mech.populate()
        return self

    def populate_(self):
        self.populate()

    def initialize(self):
        self.build()
        self.populate_parameter_buffers()
        self.intra = self.build_intra()
        if "_steady_state" in self._caches:
            self.restore("_steady_state")
            self.post_initialize()
            self.t.detach().zero_()
            self.initialized = True
            return self
        self.integrator.init_v(self)
        self.pre_initialize()
        self.integrator.mech.initialize(self.v, self.celsius, self.diam)
        self.post_initialize()
        self.integrator.mech.initialize(self.v, self.celsius, self.diam)
        self.t.detach().zero_()
        self.initialized = True
        return self

    def initialize_(self):
        self.initialize()

    def load(self, state_dict):
        """
        Load model weights from a state dictionary.

        This method supports loading weights from:
        1. A file path as a string
        2. An actual state dictionary object

        The loaded weights are matched to the model's current state dict structure
        and only compatible weights are loaded.

        Parameters
        ----------
        state_dict : str or dict
            Can be one of:
            - A file path to a saved model state
            - A state dictionary object

        Returns
        -------
        self
            The model instance with loaded weights
        """
        if isinstance(state_dict, str):
            state_dict = torch.load(
                state_dict, map_location=self.device(), weights_only=True
            )
        matched, _ = _match_state_dict(self.state_dict(), state_dict)
        self.load_state_dict(matched, strict=False)
        return self

    def load_(self, state_dict):
        self.load(state_dict)

    def state_names(self) -> List[str]:
        out = ["v"]
        return out + self.integrator.mech.all_states()

    def cache(self, name: str = None):
        """
        Cache the current model state with an optional identifier.

        This method saves a snapshot of the model's current state dictionary
        to an internal cache. The state can later be restored using the
        restore() method with the same name.

        Parameters
        ----------
        name : str, optional
            Identifier for the cached state. If None, the state is cached
            with the name 'latest'. Default is None.

        Returns
        -------
        None

        See Also
        --------
        restore : Restore a previously cached state

        Examples
        --------
        >>> model.cache('before_training')  # Cache state before training
        >>> # ... training or simulation ...
        >>> model.restore('before_training')  # Return to cached state
        """
        if name is None:
            name = "latest"
        self._caches[name] = self.state_dict()
        return self

    def cache_(self, name: str = None):
        self.cache(name)

    def restore(self, name: str = None):
        """
        Restore a previously cached model state.

        This method loads a previously cached state dictionary from the internal
        cache and applies it to the model. It's used in conjunction with the
        cache() method, which saves states.

        Parameters
        ----------
        name : str, optional
            Identifier for the cached state to restore. If None, restores
            the state cached as 'latest'. Default is None.

        Returns
        -------
        None

        See Also
        --------
        cache : Cache the current model state

        Examples
        --------
        >>> model.cache('before_training')  # Cache state before training
        >>> # ... training or simulation ...
        >>> model.restore('before_training')  # Return to cached state
        """
        if name is None:
            name = "latest"
        self.load_state_dict(self._caches[name])
        self.initialized = True
        return self

    def restore_(self, name: str = None):
        self.restore(name)

    def delete_injections(self):
        self.injections = []
        self.intra = None

    def build_intra(self):
        if self.injections:
            return IntraStim(self, self.injections)
        return None

    def insert(self, mechanism, alias=None, index_spec=None, ic=None, **kwargs):
        """
        Insert a mechanism into the model.

        Parameters
        ----------
        mechanism : Mechanism
            The mechanism to be inserted into the model.
        alias : str, optional
            An optional alias for the mechanism. If not provided, the mechanism's name will be used.
        index_spec : IndexSpec, optional
            An optional index specification that defines where the mechanism should be inserted.
            If not provided, the mechanism will be inserted everywhere.
        **kwargs
            Additional keyword arguments to be passed to the compile_mechanism function.
        """

        if self.is_built:
            raise RuntimeError("Cannot insert mechanisms after the model is built.")

        validate(mechanism)

        key = None

        if index_spec is not None:
            key = index_spec.index

        if key is None:
            self._mech_everywhere[mechanism] = (mechanism.__name__, ic, kwargs)
            return

        if mechanism in self._mech_everywhere:
            raise ValueError(f"Mechanism {mechanism} is already inserted everywhere.")
        self._mech_data.setdefault(mechanism, []).append((alias, kwargs, key))

    def ion_style(self, ion, c_style, e_style, einit, eadvance, cinit):
        assert ion in valid_ions(), f"Invalid ion: {ion}"
        self._ion_style[ion] = (c_style, e_style, einit, eadvance, cinit)

    def get_ion_style(self, ion):
        if ion in self._ion_style:
            return self._ion_style[ion]
        return self.calc_ion_style(ion)

    def c_is_written(self, ion):
        d = self._ion_write_c.get(ion, {})
        return bool(d)

    def c_is_read(self, ion):
        d = self._ion_read.get(ion, {})
        if not d:
            return False
        check = list(itertools.chain(*d.values()))
        return f"{ion}i" in check or f"{ion}o" in check

    def e_is_read(self, ion):
        d = self._ion_read.get(ion, {})
        if not d:
            return False
        return f"e{ion}" in list(itertools.chain(*d.values()))

    def calc_ion_style(self, ion):
        c_is_written = self.c_is_written(ion)
        c_is_read = self.c_is_read(ion)
        e_is_read = self.e_is_read(ion)

        if c_is_written:
            if e_is_read:
                return (3, 2, 1, 1, 1)
            return (3, 0, 0, 0, 1)
        if c_is_read:
            if e_is_read:
                return (1, 2, 1, 0, 0)
            return (1, 0, 0, 0, 0)
        if e_is_read:
            return (0, 1, 0, 0, 0)
        return (0, 0, 0, 0, 0)

    def register_mech(self, m, shape, key):
        name = m.name
        mech = m.__class__
        self._m_name.append(name)
        self._m_list.append(m)
        self._m_keys.append(key)
        self._m_shape[name] = shape

        self._m_unfactorable[name] = None
        self._m_has_gtot[name] = None
        self._m_divide_by_two[name] = None

        for k, v in mech._currents.items():
            self._m_curr.setdefault(k, {}).update({name: v})

        for k, v in mech._read_ion.items():
            self._ion_read.setdefault(k, {}).update({name: v})

        for k, v in mech._write_ion.items():
            self._ion_write.setdefault(k, {}).update({name: v})

        for k, v in mech._write_ion_c.items():
            self._ion_write_c.setdefault(k, {}).update({name: v})

    def build(self):
        if self.is_built:
            return self

        def are_strings_unique(data: list) -> bool:
            strings_only = [item for item in data if item is not None]
            return len(strings_only) == len(set(strings_only))

        for mech, (name, ic, kwargs) in self._mech_everywhere.items():
            key = None
            shape = self.shape
            m = mech(name, self.celsius, self.diam, shape, key, ic=ic, **kwargs)
            self.register_mech(m, shape, key)

        for mech, data in self._mech_data.items():
            aliases, kwargs_list, keys = tuple(map(list, zip(*data)))
            if not are_strings_unique(aliases):
                raise ValueError(
                    f"Duplicate aliases found for mechanism {mech.__name__}."
                )
            m, shape, key = compile_mechanism(self, mech, keys, aliases, kwargs_list)
            self.register_mech(m, shape, key)

        all_ions = get_unique_keys([self._ion_read, self._ion_write, self._ion_write_c])

        _ion_write = {f"i{k}": v for k, v in self._ion_write.items()}
        self._m_curr.update(_ion_write)

        ions = {}
        for ion in all_ions:
            ion_style = self.get_ion_style(ion)
            ions[ion] = Ion(
                ion,
                self.shape,
                *ion_style,
            )
            for m in self._m_list:
                m.register_ion(ions[ion])

        mechs = {n: m for n, m in zip(self._m_name, self._m_list)}
        keys = {n: k for n, k in zip(self._m_name, self._m_keys)}
        mech = MechanismHandler(
            self.celsius,
            self.area,
            mechs,
            ions,
            self._ion_write_c,
            self._ion_read,
            self._m_curr,
        )

        for m in mech.mechanisms.values():
            m.setreference("t", lambda: self.t)

        self.integrator = self.integrator(self, mech)
        self.is_built = True
        self.eval()
        return self

    def build_(self):
        self.build()

    def detach(self):
        """
        Detach the model from the current computation graph.

        This method is used to detach the model's parameters and buffers
        from the current computation graph, which is useful for preventing
        gradients from being computed during backpropagation.
        """
        self.integrator.detach(self)
        return self

    def detach_(self):
        self.detach()

    def register_parametrization(self, name: str, parametrization: torch.nn.Module):
        torch.nn.utils.parametrize.register_parametrization(self, name, parametrization)

    def slice(
        self,
        include=None,
        exclude="branchpoint",
        fuzzy=True,
        match_case=False,
        loc=None,
    ):
        """
        Finds all indices in the tree structure based on inclusion and exclusion criteria.

        Parameters
        ----------
        include : str or list of str, optional
            Patterns to include in the search.
        exclude : str or list of str, optional
            Patterns to exclude from the search.
        fuzzy : bool, optional
            If True, performs fuzzy matching. Default is True.
        match_case : bool, optional
            If True, matches case sensitively. Default is False.
        loc : float, optional
            If provided, refines the search to the compartment whose
            ( … )-location is closest to `loc` (a float in [0, 1]).
            If None, no refinement is done.

        Returns
        -------
        Slice
            A slice object containing the indices of the matches.
        """
        return self[
            :,
            self.find(
                include=include,
                exclude=exclude,
                fuzzy=fuzzy,
                match_case=match_case,
                full_report=False,
                loc=loc,
            ),
        ]

    def find_not(self, exclude=None, fuzzy=True, match_case=False):
        indices = find_indices_smart(
            self.names,
            exclude=exclude,
            fuzzy=fuzzy,
            match_case=match_case,
            device=self.device(),
        )
        return indices.indices

    def find(
        self,
        include=None,
        exclude="branchpoint",
        fuzzy=True,
        match_case=False,
        full_report=False,
        as_list=False,
        loc=None,
    ):
        indices = find_indices_smart(
            self.names,
            include=include,
            exclude=exclude,
            fuzzy=fuzzy,
            match_case=match_case,
            device=self.device(),
            loc=loc,
        )
        if full_report:
            return indices
        else:
            # Return only the indices of the matches
            if isinstance(indices.indices, slice):
                if as_list:
                    return list(
                        range(
                            indices.indices.start,
                            indices.indices.stop,
                            indices.indices.step or 1,
                        )
                    )
                return indices.indices
            else:
                return indices.indices.tolist()
        return indices

    def terminal_indices(self):
        """
        Returns the indices of terminal nodes in the tree.

        A terminal node is defined as a node that has no children in the graph.
        """
        terminal_mask = torch.tensor(
            [
                len(list(self.graph.successors(i))) == 0
                for i in range(len(self.graph.nodes))
            ],
            device=self.device(),
        )
        return torch.nonzero(terminal_mask, as_tuple=False).squeeze(1).tolist()

    def init_v(self):
        self.integrator.init_v(self)

    def init_v_(self):
        self.init_v()

    def n(self) -> int:
        return self.v.shape[-2]

    def set_value(self, name: str, value: torch.Tensor):
        """
        Set a parameter or state variable by name.

        Parameters
        ----------
        name : str
            The name of the parameter or state variable to set.
        value : torch.Tensor
            The value to set for the specified parameter or state variable.
        """

        def _set_value(model):
            if hasattr(model, name):
                getattr(model, name).copy_(
                    value.to(device=model.device(), dtype=model.dtype())
                )
            else:
                raise AttributeError(f"Model has no attribute '{name}' to set.")

        self.register_post_initialize_hook(_set_value)


# Define the return type for clarity
class FindResult(NamedTuple):
    indices: Union[slice, torch.Tensor]
    local_indices: Dict[str, Union[slice, torch.Tensor]]
    local_sizes: Dict[str, int]
    total_size: int


# Helper function to convert numpy indices to a slice or tensor
def _indices_to_slice_or_tensor(
    numpy_indices: np.ndarray, device: Optional[torch.device] = None
) -> Union[slice, torch.Tensor]:
    """Converts a 1D numpy array of indices into a slice if possible, else a tensor."""
    num_indices = len(numpy_indices)

    if num_indices == 0:
        return slice(0, 0, None)

    if num_indices == 1:
        start = int(numpy_indices[0])
        return slice(start, start + 1, None)

    # Check if the step between all indices is constant
    diffs = np.diff(numpy_indices)
    step = int(diffs[0])

    if np.all(diffs == step):
        # The indices form an arithmetic progression. It can be a slice!
        start = int(numpy_indices[0])
        stop = int(numpy_indices[-1]) + step
        return slice(start, stop, step if step != 1 else None)
    else:
        # Indices are not contiguous, fall back to returning a tensor
        torch_indices = torch.from_numpy(numpy_indices)
        return torch_indices.to(device) if device else torch_indices


def find_indices_smart(
    data: List[str],
    include: Optional[Union[str, List[str]]] = None,
    exclude: Optional[Union[str, List[str]]] = None,
    fuzzy: bool = True,
    match_case: bool = False,
    device: Optional[torch.device] = None,
    *,
    loc: Optional[float] = None,  #
    _tol: float = 1e-9,  # tolerance for loc comparisons
) -> FindResult:
    """
    Finds indices based on criteria and returns detailed results including local indices
    for each included pattern.

    Handles complex patterns like 'axon[0]' correctly. If fuzzy=True:
    - A simple pattern like 'axon' will match 'axon', 'axon[0]', but not 'taxons'.
    - A complex pattern like 'axon[0]' will match strings containing the literal 'axon[0]'.

    Parameters
    ----------
    data (List[str]):
        The list of strings to search through.
    include (Optional[Union[str, List[str]]]):
        Patterns to include.
    exclude (Optional[Union[str, List[str]]]):
        Patterns to exclude.
    fuzzy (bool):
        If True, performs smart whole-word/substring matching. If False, an exact match.
    match_case (bool):
        If True, the matching is case-sensitive.
    device (Optional[torch.device]):
        PyTorch device for resulting tensors.
    loc (Optional[float]):
        If provided, refines the search to the compartment whose
        ( … )-location is closest to `loc` (a float in [0, 1]).
        If None, no refinement is done.
    _tol (float):
        Tolerance for comparing `loc` values, default is 1e-9.

    Returns
    -------
    FindResult:
        A named tuple with detailed matching results.
    """
    empty = FindResult(slice(0, 0), {}, {}, 0)
    if not data:
        return empty

    # ------------------------------------------------------------------ #
    # 0. basic include / exclude filtering                               #
    # ------------------------------------------------------------------ #
    s = pd.Series(data, dtype="string")
    final_mask = pd.Series(True, index=s.index)

    local_idx_map, local_sz_map, pattern_masks = {}, {}, {}

    def _make_mask(pat: str) -> pd.Series:
        if fuzzy:
            if re.search(r"[^a-zA-Z0-9_]", pat):
                rgx = re.escape(pat)
            else:
                rgx = rf"\b{re.escape(pat)}(?![a-zA-Z0-9])"
            return s.str.contains(rgx, case=match_case, regex=True, na=False)
        else:
            a = s.str.lower() if not match_case else s
            b = pat.lower() if not match_case else pat
            return a == b

    include_pats = [include] if isinstance(include, str) else (include or [])
    for pat in include_pats:
        pattern_masks[pat] = _make_mask(pat)
    if pattern_masks:
        final_mask &= pd.concat(pattern_masks.values(), axis=1).any(axis=1)

    if exclude:
        exclude_pats = [exclude] if isinstance(exclude, str) else exclude
        exc_mask = pd.Series(False, index=s.index)
        for pat in exclude_pats:
            exc_mask |= _make_mask(pat)
        final_mask &= ~exc_mask

    if not final_mask.any():
        return empty

    # ------------------------------------------------------------------ #
    # 1. optional loc‑based refinement                                   #
    # ------------------------------------------------------------------ #
    idx_arr = s.index[final_mask].to_numpy()

    if loc is not None:
        if not (0.0 <= loc <= 1.0):
            raise ValueError("loc must be within [0, 1].")
        # Parse candidate ( … ) positions
        cand = []
        for gi in idx_arr:
            m = re.search(r"\(([\d.]+)\)$", s.iloc[gi])
            if m:
                cand.append((gi, float(m.group(1))))
        if cand:  # only refine if we found any
            # Exclude terminal 0 / 1 unless requested exactly
            if abs(loc) > _tol:
                cand = [(gi, x) for gi, x in cand if abs(x) > _tol]
            if abs(loc - 1.0) > _tol:
                cand = [(gi, x) for gi, x in cand if abs(x - 1.0) > _tol]
            if not cand:  # nothing left → fall back
                pass
            else:
                gi_best, _ = min(cand, key=lambda t: abs(t[1] - loc))
                idx_arr = np.array([gi_best], dtype=np.int64)
                final_mask = pd.Series(False, index=s.index)
                final_mask[idx_arr[0]] = True

    # ------------------------------------------------------------------ #
    # 2. build return object                                             #
    # ------------------------------------------------------------------ #
    total_idx = _indices_to_slice_or_tensor(idx_arr, device)
    total_sz = len(idx_arr)

    if include_pats:
        g2l = {g: i for i, g in enumerate(idx_arr)}
        for pat in include_pats:
            pat_mask = pattern_masks[pat] & final_mask
            g_idx = s.index[pat_mask].to_numpy()
            if len(g_idx):
                l_idx = np.fromiter((g2l[g] for g in g_idx), dtype=np.int64)
                local_idx_map[pat] = _indices_to_slice_or_tensor(l_idx, device)
                local_sz_map[pat] = len(l_idx)
            else:
                local_idx_map[pat] = slice(0, 0)
                local_sz_map[pat] = 0

    return FindResult(total_idx, local_idx_map, local_sz_map, total_sz)


class Axon(Population):
    """
    Base 1D fiber class.

    This is the base class for axon models, implementing common functionality
    for simulating action potential propagation along 1D fibers.
    """

    __constants__ = [
        "n_ax",
        "n_comp",
        "temp",
        "v_init",
    ]

    def __init__(
        self, diameters, n_comp: int, celsius=37.0, v_init=-80.0, integrator=None
    ):
        if integrator is None:
            integrator = bwd_euler_ub()
        super().__init__(len(diameters), n_comp, integrator=integrator, celsius=celsius)

        self.register_buffer(
            "diameters", torch.as_tensor(diameters, dtype=self.dtype())
        )

        self.n_ax = self.np
        self.n_comp = self.nc
        self.temp = float(celsius)

        self.v_init = v_init
        self.v[:] = v_init
        self.v.detach_()

        self.x[:] = self._x()  # Initialize x positions

        self.cid = None

        if torch.is_tensor(diameters):
            diameters = diameters.to(self.dtype()).clone().detach()
        else:
            diameters = torch.tensor(diameters, dtype=self.dtype())

        if diameters.ndim == 1:
            diameters = diameters.unsqueeze(1)

        self.diam[:] = diameters
        self.diam.detach_()

        # -- biophysics --
        self.biophysics()

    def biophysics(self):
        """
        Placeholder for biophysics-related initializations.
        This method can be overridden in subclasses to add specific
        biophysics-related parameters or configurations.
        """
        pass

    def register_cid(self, cid):
        self.cid = cid
        self.names = self.cid.names.tolist()
        for name in np.unique(self.names):
            self.slice(name).label(name)

    def c(self, *args):
        """
        Convert relative positions to node indices.

        Parameters
        ----------
        *args : float
            Variable number of float values between 0 and 1, representing
            relative positions along the axon.

        Returns
        -------
        list
            List of integer node indices corresponding to the input positions.

        Examples
        --------
        >>> model.c(0.25, 0.5, 0.75)
        [25, 50, 75]  # For a model with n_comp=101
        """
        return [round((self.n_comp - 1) * i) for i in args]


def _match_state_dict(
    state_dict_a: Dict[str, torch.Tensor],
    state_dict_b: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Match tensors between two state dictionaries based on key names and shapes.
    This function filters a state dictionary to find tensors that have matching keys
    and identical shapes in another state dictionary, which is useful for selective
    parameter loading or model weight comparisons.

    Parameters
    ----------
    state_dict_a : Dict[str, torch.Tensor]
        First state dictionary used as reference for key and shape matching.
    state_dict_b : Dict[str, torch.Tensor]
        Second state dictionary to filter based on keys and shapes in state_dict_a.
    Returns
    -------
    Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        A tuple containing:
        - matched_state_dict: Dictionary with entries from state_dict_b that have
          matching keys and shapes in state_dict_a.
        - unmatched_state_dict: Dictionary with remaining entries from state_dict_b
          that don't have matching keys or shapes in state_dict_a.
    """
    matched_state_dict = {
        key: state
        for (key, state) in state_dict_b.items()
        if key in state_dict_a and state.shape == state_dict_a[key].shape
    }
    unmatched_state_dict = {
        key: state
        for (key, state) in state_dict_b.items()
        if key not in matched_state_dict
    }
    return matched_state_dict, unmatched_state_dict


class Unmyelinated(Axon):
    """
    Base unmyelinated axon model class.

    This class implements a model of unmyelinated axons (nerve fibers without myelin sheaths)
    by extending the base Axon class. It uses a uniform spatial discretization with nodes
    spaced at regular intervals.

    Parameters
    ----------
    diameters : array_like
        Diameters of the axons in μm. Can be a single value, list, or tensor.
    L : float, optional
        Length of the axon in mm (will be converted to μm internally). Default is 1.0 mm.
    dx : float, optional
        Spatial discretization step in μm. Default is 10.0 μm.
    celsius : float, optional
        Temperature in degrees Celsius. Default is 37°C.
    v_init : float, optional
        Initial membrane potential in mV. Default is -80 mV.
    integrator : Integrator, optional
        The integrator to use for the simulation. If None, a default backward Euler integrator
        will be used.

    Notes
    -----
    The number of nodes is calculated to ensure it's odd (for a centered node at position 0).

    See Also
    --------
    Axon : Base class providing common functionality for axon models.
    Myelinated : Companion class implementing myelinated axon models.
    """

    Axon.RANGE(cm=1.0, rhoa=35.4)
    Axon.GLOBAL(celsius=37.0)

    def __init__(
        self,
        diameters,
        L=1.0 * mm,
        dx=10.0,
        celsius=37.0,
        v_init=-80.0,
        integrator=None,
    ):
        # L = L * 1000  # mm -> um
        n_comp = L / dx
        n_comp = math.ceil(n_comp) // 2 * 2 + 1
        self.dx_: float = dx
        self.L: float = n_comp * dx
        super().__init__(diameters, n_comp, celsius, v_init, integrator)
        self.dx[:] = self.dx_

    def _x(self) -> torch.Tensor:  # x in um
        length = (self.n_comp - 1) * self.dx_
        x = torch.linspace(-length / 2, length / 2, self.n_comp, device=self.device())
        return torch.atleast_2d(x)


class Myelinated(Axon):
    """
    Base myelinated axon model class.

    This class implements a model of myelinated axons (nerve fibers with myelin sheaths)
    by extending the base Axon class. It models nodes of Ranvier separated by
    myelinated internodal regions, with parameters that scale with axon diameter.

    Parameters
    ----------
    diameters : array_like
        Diameters of the axons in μm. Can be a single value, list, or tensor.
    n_node : int
        Number of compartments (nodes of Ranvier) in the model.
    node_length : float, optional
        Length of the nodes of Ranvier in μm. Default is 2.0 μm.
    celsius : float, optional
        Temperature in degrees Celsius. Default is 37°C.
    v_init : float, optional
        Initial membrane potential in mV. Default is -80 mV.
    integrator : Integrator, optional
        The integrator to use for the simulation. If None, a default backward Euler integrator
        will be used.

    Notes
    -----
    The model uses quadratic equations to calculate various geometric parameters
    based on the fiber diameter, following anatomical scaling relationships.

    See Also
    --------
    Axon : Base class providing common functionality for axon models.
    Unmyelinated : Companion class implementing unmyelinated axon models.
    """

    Axon.RANGE(
        cm=1.0,
        rhoa=35.4,
    )
    Axon.GLOBAL(
        axon_d={
            "axond1": 0.0,
            "axond2": 0.7,
            "axond3": 0.0,
        },
        node_d={
            "noded1": 0.0,
            "noded2": 0.7,
            "noded3": 0.0,
        },
        delta_x={
            "deltax1": 0.0,
            "deltax2": 100.0,
            "deltax3": 0.0,
        },
        celsius=37.0,
    )

    class myelinated_rhoa(torch.nn.Module):
        def __init__(self, deltax1, deltax2, deltax3, axond1, axond2, axond3):
            super().__init__()
            self.deltax1 = deltax1
            self.deltax2 = deltax2
            self.deltax3 = deltax3
            self.axond1 = axond1
            self.axond2 = axond2
            self.axond3 = axond3

        def forward(self, rhoa, dx, diameters):
            diameters = diameters.unsqueeze(1) if diameters.ndim == 1 else diameters
            axon_d = self.axond1 * diameters**2 + self.axond2 * diameters + self.axond3
            deltax = (
                self.deltax1 * diameters**2 + self.deltax2 * diameters + self.deltax3
            )
            deltax = deltax / dx
            scale = 1 / ((axon_d / diameters) ** 2)
            rhoa = rhoa * scale * deltax
            return rhoa

    class myelinated_node_d(torch.nn.Module):
        def __init__(self, noded1, noded2, noded3):
            super().__init__()
            self.noded1 = noded1
            self.noded2 = noded2
            self.noded3 = noded3

        def forward(self, diam):
            node_d = self.noded1 * diam**2 + self.noded2 * diam + self.noded3
            return node_d

    def __init__(
        self,
        diameters,
        n_node: int,
        node_length=2.0,
        celsius=37.0,
        v_init=-80.0,
        integrator=None,
    ):
        self.node_length = node_length  # length of the nodes of Ranvier in um
        super().__init__(diameters, n_node, celsius, v_init, integrator)
        self.dx[:] = self.node_length

        self.register_parametrization(
            "diam", self.myelinated_node_d(self.noded1, self.noded2, self.noded3)
        )

        self.register_parametrization_in_graph(
            "rhoa",
            self.myelinated_rhoa(
                self.deltax1,
                self.deltax2,
                self.deltax3,
                self.axond1,
                self.axond2,
                self.axond3,
            ),
            args=("dx", "diameters"),
        )

    def deltax(self, diameters):
        deltax = self.deltax1 * diameters**2 + self.deltax2 * diameters + self.deltax3
        return deltax

    def _x(self) -> torch.Tensor:  # x in um
        length = (self.n_comp - 1) * self.deltax(self.diameters).unsqueeze(1)
        start = -length / 2
        end = length / 2
        steps = self.n_comp
        t = torch.linspace(0, 1, steps, device=length.device).unsqueeze(0)
        return (1 - t) * start + t * end


# callback helpers
def pre_loop_hook(c, m):
    c.pre_loop_hook(m)


def post_loop_hook(c, m):
    c.post_loop_hook(m)


def pre_step_hook(c, m):
    c.pre_step_hook(m)


@torch.compile
def post_step_hook(c, m):
    c.post_step_hook(m)


def pre_chunk_hook(c, m, n):
    c.pre_chunk_hook(m, n)


def post_chunk_hook(c, m, n):
    c.post_chunk_hook(m, n)


# Helper for the super-fast path: Merges overlapping/adjacent 1D intervals
def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merges a list of [start, stop) intervals."""
    if not intervals:
        return []

    # Sort intervals by their start point
    intervals.sort(key=lambda x: x[0])

    merged = [list(intervals[0])]  # Use list to allow modification

    for current_start, current_stop in intervals[1:]:
        last_start, last_stop = merged[-1]

        # If the current interval overlaps with or is adjacent to the last one
        if current_start <= last_stop:
            # Merge them by extending the last one's stop
            merged[-1][1] = max(last_stop, current_stop)
        else:
            # No overlap, start a new interval
            merged.append([current_start, current_stop])

    return [tuple(i) for i in merged]


# Custom exception to signal fallback to the slower method
class NotComposableError(Exception):
    pass


def _handle_pure_slice_union(
    indices: List[Tuple[slice, ...]], shape: Tuple[int, ...]
) -> Tuple[Tuple[slice, ...], bool, Tuple[int, ...], List[List[int]]]:
    """
    SUPER FAST PATH: Calculates the union of pure slice tuples directly
    without materializing flat indices.
    """
    ndim = len(shape)

    # 1. Normalize all slices to have concrete start, stop, step
    normalized_slices = []
    for s_tuple in indices:
        if len(s_tuple) > ndim:
            raise NotComposableError("Slice tuple has more dimensions than shape")

        # Pad with slice(None) if needed
        s_tuple_full = s_tuple + (slice(None),) * (ndim - len(s_tuple))

        current_norm = []
        for i, s in enumerate(s_tuple_full):
            start, stop, step = s.indices(shape[i])
            if step != 1:
                # This path only works for contiguous blocks (step=1)
                raise NotComposableError("Slice with step != 1 found")
            current_norm.append((start, stop))
        normalized_slices.append(tuple(current_norm))

    # 2. Merge intervals for each dimension
    final_intervals = []
    for i in range(ndim):
        dim_intervals = [s[i] for s in normalized_slices]
        merged_dim_intervals = _merge_intervals(dim_intervals)

        # If any dimension results in a non-contiguous union (e.g., [0,5) and [10,15)),
        # then the total union is not one single slice tuple.
        if len(merged_dim_intervals) != 1:
            raise NotComposableError("Union is not a single contiguous block")

        final_intervals.append(merged_dim_intervals[0])

    # 3. If we got here, the result is composable. Create final slice objects.
    final_slice_tuple = tuple(slice(s, e) for s, e in final_intervals)
    final_shape = tuple(e - s for s, e in final_intervals)

    # 4. Calculate local_indices (the most complex part)
    # We need to find where each original slice lives inside the final merged slice.
    local_indices = []
    final_slice_starts = [s.start for s in final_slice_tuple]

    for original_norm_slice in normalized_slices:
        # Create relative slices: (orig_start - final_start, orig_stop - final_start)
        relative_slices = tuple(
            slice(s - fs, e - fs)
            for (s, e), fs in zip(original_norm_slice, final_slice_starts)
        )

        # Use mgrid and ravel_multi_index on the *final_shape* to get local indices
        coords = np.mgrid[relative_slices]
        flat_local = np.ravel_multi_index(
            tuple(coords.reshape(ndim, -1)), dims=final_shape
        )
        local_indices.append(sorted(flat_local.tolist()))

    return final_slice_tuple, True, final_shape, local_indices


# The main entrypoint function
def compose_or_flatten_union(
    indices: List[Any], shape: Tuple[int, ...]
) -> Tuple[Union[Tuple[slice, ...], List[int]], bool, Tuple[int, ...], List[List[int]]]:
    """
    Calculates the union of elements selected by a list of indices. Dispatches
    to a highly optimized path for pure slice inputs or falls back to a general
    method for complex/advanced indexing.
    """
    # --- Edge-Case Handling ---
    empty_slice_tuple = tuple(slice(0, 0) for _ in shape)
    empty_shape = tuple(0 for _ in shape) or (0,)
    empty_locals = [[] for _ in indices]

    if not indices:
        return empty_slice_tuple, True, empty_shape, []

    if not shape or not all(s > 0 for s in shape):
        return empty_slice_tuple, True, empty_shape, empty_locals

    # --- SUPER-FAST-PATH DISPATCHER ---
    # Check if we can use the slice-domain optimization
    is_pure_slice_case = all(
        isinstance(idx, tuple) and all(isinstance(s, slice) for s in idx)
        for idx in indices
    )

    if is_pure_slice_case:
        try:
            # Attempt the ultra-fast path that works directly on slices
            return _handle_pure_slice_union(indices, shape)
        except NotComposableError:
            # This happens if slices have steps != 1 or their union is not a single
            # rectangle. We must fall back to the slower, general method.
            pass

    # This path is for advanced indexing (lists, bools) or non-composable slices.
    total_elements = int(np.prod(shape))
    arr = np.arange(total_elements).reshape(shape)

    contributions = []
    for idx in indices:
        try:
            selected_elements = arr[idx]
            contributions.append(sorted(list(set(selected_elements.flatten()))))
        except IndexError as e:
            raise IndexError(
                f"Indexer invalid for shape. Idx: {idx}, Shape: {shape}. Error: {e}"
            ) from e

    all_flat_indices = set()
    for contrib in contributions:
        all_flat_indices.update(contrib)

    if not all_flat_indices:
        return empty_slice_tuple, True, empty_shape, empty_locals

    sorted_union_indices = sorted(list(all_flat_indices))

    global_to_local_map = {
        g_idx: l_idx for l_idx, g_idx in enumerate(sorted_union_indices)
    }
    local_indices = [
        [global_to_local_map[g_idx] for g_idx in contrib] for contrib in contributions
    ]

    multi_dim_coords = np.unravel_index(sorted_union_indices, shape)
    min_coords = np.min(multi_dim_coords, axis=1)
    max_coords = np.max(multi_dim_coords, axis=1)

    bounding_box_dims = max_coords - min_coords + 1
    if len(sorted_union_indices) == np.prod(bounding_box_dims):
        composed_slices = tuple(
            slice(int(min_c), int(max_c) + 1)
            for min_c, max_c in zip(min_coords, max_coords)
        )
        final_shape = tuple(s.stop - s.start for s in composed_slices)
        return composed_slices, True, final_shape, local_indices
    else:
        final_shape = (len(sorted_union_indices),)
        result_indices = [int(i) for i in sorted_union_indices]
        return result_indices, False, final_shape, local_indices


def compile_mechanism(model, mechanism, indices, aliases, kwargs_list):
    total_index, is_composable, final_shape, local_indices = compose_or_flatten_union(
        indices, model.shape
    )

    # local indices is now a list of lists, where each sublist corresponds to the
    # local indices of the original indices in the final selection.
    # We can now use these local indices to compile the mechanism.

    additional_parameters = {}

    for alias, kwargs, idx in zip(aliases, kwargs_list, local_indices):
        for k, v in kwargs.items():
            additional_parameters.setdefault(k, []).append((alias, v, idx))

    m = mechanism(
        None,
        model.celsius,
        model.diam,
        final_shape,
        key=total_index,
        is_composable=is_composable,
        additional_parameters=additional_parameters,
    )

    return m, final_shape, total_index
