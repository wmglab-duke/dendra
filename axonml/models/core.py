import math
from typing import List, Tuple, Optional, Dict, Callable
import re
import itertools
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor

from tqdm.auto import tqdm

from axonml.models.stim.intrastim import IntraStim
from axonml.models.stim.waveform import Waveform

from axonml.models.callbacks import CallbackList, Callback
from axonml.models.backend import Backend as A
from axonml.models.parametric import Parameterized
from axonml.models.mechanisms.core import Mechanism, validate
from axonml.models.mechanisms import c_context, e_context
from axonml.models.declarations import PARAMETER
from axonml.models.mechanisms.handler.defaults import valid_ions
from axonml.models.mechanisms.handler.handler import build_handler
from axonml.models.mechanisms.handler.ions import build_ion
from axonml.models.mechanisms.mech_compiler import compile_mechanism
from axonml.units import mm, um
from axonml.models.mechanisms.compilers.core import MechCompiler, DF_Compiler
from axonml.models.interfaces import HandlerInterface
from axonml.models.integrators import euler, dufort_frankel

from axonml.helpers import (
    op_mc, op_sc, ve_from_s_t, 
    IMEM, CUDA, DTWARN, DEBUG, DETECT_ANOMALIES, PADE,
    ctx
)


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
    for base_pattern in base_patterns:
        regex_pattern = (
            r"\b"
            + r"\b.*?\b".join(re.escape(part) for part in base_pattern.split("."))
            + r"\b"
        )
        if re.search(regex_pattern, target_string):
            return True
    return False


class SymmetricConv1D(torch.nn.Conv1d):
    def forward(self, x):
        if self.training:
            weight_ = (self.weight + torch.flip(self.weight, [-1])) / 2
        else:
            weight_ = self.weight
        return self._conv_forward(x, weight_, self.bias)


@torch.compile(fullgraph=True)
def step(integrator, model, ve, dt, t_ind):
    integrator.step(model, ve, dt, t_ind)


@torch.compile(fullgraph=True)
def step_intra(integrator, model, ve, intra, dt, t_ind):
    integrator.step_intra(model, ve, intra, dt, t_ind)


class Axon(Parameterized):
    """
    Base 1D fiber class.

    This is the base class for axon models, implementing common functionality
    for simulating action potential propagation along 1D fibers.

    Parameters
    ----------
    diameters : array_like
        Diameters of the axons in μm.
    n_comp : int
        Number of nodes in the axon model.
    temp : float, optional
        Temperature in degrees Celsius. Default is 37.0.
    v_init : float, optional
        Initial membrane potential in mV. Default is -80.0.
    method : str, optional
        Integration method. One of 'euler', 'rk1', 'heun', 'rk2', 'rk4',
        'dufort-frankel', or 'df'. Default is 'rk1'.
    beta : float, optional
        Hyperdiffusion coefficient. Default is 0.0.

    Attributes
    ----------
    n_ax : int
        Number of axons in the model.
    n_comp : int
        Number of compartments in each axon.
    temp : float
        Temperature in degrees Celsius.
    v_init : float
        Initial membrane potential in mV.
    method : str
        Integration method.
    mech : HandlerInterface
        Handler for membrane mechanisms.
    t_ind : int
        Current time index.
    dt : float
        Time step in ms.
    """

    _dt_lim = None
    __constants__ = [
        "n_ax",
        "n_comp",
        "temp",
        "v_init",
    ]

    def __init__(
        self, diameters, n_comp: int, temp=37.0, v_init=-80.0, integrator=euler()
    ):
        super().__init__()
        
        self.n_ax = len(diameters)
        self.n_comp = n_comp
        self.temp = temp
        self.v_init = v_init
        self.shape = integrator.shape(self.n_ax, self.n_comp)

        self.register_buffer("_dummy", torch.zeros(1))

        self.cid = None

        self.compiler = integrator.compiler(DEBUG, DETECT_ANOMALIES, PADE)
        self.builder = None
        if integrator.builder is not None:
            self.builder = integrator.builder(DEBUG, IMEM)
        self.integrator = integrator

        self.t_ind: int = 0
        self.dt: float = A.dt

        self._m_list = []
        self._m_name = []
        self._m_curr = {}
        self._m_unfactorable = {}
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

        self._caches = {}

        self.register_buffer("y", torch.zeros(self.n_ax, 1))
        self.register_buffer("z", torch.zeros(self.n_ax, 1))

        if torch.is_tensor(diameters):
            diameters = diameters.to(self.dtype()).clone().detach()
        else:
            diameters = torch.tensor(diameters, dtype=self.dtype())

        self._register_buffers(diameters)

        self.initialized: bool = False

        # -- biophysics --
        self.biophysics()

        # -- constants --
        self.eval()

    def biophysics(self):
        """
        Placeholder for biophysics-related initializations.
        This method can be overridden in subclasses to add specific
        biophysics-related parameters or configurations.
        """
        pass

    @property
    def mech(self):
        return self.integrator.mech
    
    @property
    def i_membrane(self):
        return self.integrator.i_membrane

    def __init_subclass__(cls, **kwargs):
        def init_decorator(previous_init):
            def new_init(self, *args, **kwargs):
                previous_init(self, *args, **kwargs)
                if type(self) == cls:
                    Axon.__post_init__(self)

            return new_init

        cls.__init__ = init_decorator(cls.__init__)

    def __post_init__(self):
        changed = self.instantiate_parameters_lambda()
        if changed:
            self.calculate_geometric_params()
        with (
            e_context(use_last=True),
            c_context(use_last=True),
        ):
            self._build()
        if CUDA:
            self.cuda()

    def _register_buffers(self, diameters):
        self.register_buffer("diam", diameters)
        self.register_buffer("area_c", self.area_(self.diam)[:, None])
        self.register_buffer("cm_c", self.cm_(self.area_c))
        self.register_buffer("ra_c", self.ra_(self.diam)[:, None])
        self.register_buffer("v_init_c", torch.tensor(self.v_init))
        self.register_buffer("temp_c", torch.tensor(self.temp))

    def register_cid(self, cid):
        self.cid = cid

    def set_diam(self, diams):
        diams = torch.as_tensor(diams, dtype=self.dtype())
        self.diam[:] = diams
        self.instantiate_parameters_lambda()
        self.calculate_geometric_params()

    def calculate_geometric_params(self):
        self.area_c = self.area_(self.diam)[:, None]
        self.cm_c = self.cm_(self.area_c)
        self.ra_c = self.ra_(self.diam)[:, None]

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
            return [p for n, p in self.named_parameters() if matches_any_pattern(names, n)]
        
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
            return [(n, p) for n, p in self.named_parameters() if matches_any_pattern(names, n)]

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

    def unfreeze_group(self, *groups):
        for g in groups:
            group = getattr(self, g)
            print(f"Unfreezing group '{g}'")
            for p in group:
                p.requires_grad = True

    def freeze(self, *names):
        if not names:
            for p in self.parameters():
                p.requires_grad = False
        else:
            for n, p in self.named_parameters():
                if matches_any_pattern(names, n):
                    print(f"Freezing {n}")
                    p.requires_grad = False

    def freeze_group(self, *groups):
        for g in groups:
            group = getattr(self, g)
            print(f"Freezing group '{g}'")
            for p in group:
                p.requires_grad = False

    def register_post_initialize_hook(self, fn: Callable):
        self.post_initialize_hooks.append(fn)

    def register_pre_initialize_hook(self, fn: Callable):
        self.pre_initialize_hooks.append(fn)

    def set_y(self, y):
        self.y[:] = torch.as_tensor(y)
        return self

    def set_z(self, z):
        self.z[:] = torch.as_tensor(z)
        return self

    @torch.jit.export
    def n(self) -> int:
        return self.v.shape[0]

    def device(self):
        return self._dummy.device

    def dtype(self):
        return self._dummy.dtype

    def insert(self, mechanism, ic=None, mask_out=None, mask_in=None, **kwargs):
        """
        Insert a mechanism into the model.

        Parameters
        ----------
        mechanism : Mechanism
            The mechanism to be inserted into the model.
        ic : dict, optional
            Dictionary of initial conditions for the mechanism states.
            Keys are state names and values are initial values.
        mask_out : str, int, or slice, optional
            Mask specifying compartments for which the mechanism will not
            contribute to the current calculation.
        mask_in : str, int, or slice, optional
            Mask specifying compartments for which the mechanism will contribute
            to the current calculation.
        **kwargs
            Additional keyword arguments to be passed to the compile_mechanism function.
        """
        if mechanism.__name__ in self._m_name:
            raise ValueError(f"Mechanism {mechanism.__name__} already exists in the model.")

        validate(mechanism)

        m, unfactorable, has_gtot, divide_by_two = self.compiler.compile(
            mechanism, 
            self, 
            ic=ic,
            mask_in=mask_in,
            **kwargs,
        )

        self._m_list.append(m)
        self._m_name.append(mechanism.__name__)
        self._m_unfactorable[mechanism.__name__] = unfactorable
        self._m_has_gtot[mechanism.__name__] = has_gtot
        self._m_divide_by_two[mechanism.__name__] = divide_by_two

        for k, v in mechanism._currents.items():
            self._m_curr.setdefault(k, {}).update({mechanism.__name__: v})

        for k, v in mechanism._read_ion.items():
            self._ion_read.setdefault(k, {}).update({mechanism.__name__: v})

        for k, v in mechanism._write_ion.items():
            self._ion_write.setdefault(k, {}).update({mechanism.__name__: v})

        for k, v in mechanism._write_ion_c.items():
            self._ion_write_c.setdefault(k, {}).update({mechanism.__name__: v})

    def insert_at(self, index, mechanism, ic=None, **kwargs):
        if isinstance(index, int):
            index = [index]
        if isinstance(index, str):
            index = self.cid.loc(index)
        if isinstance(index, list):
            if all(isinstance(i, str) for i in index):
                index = self.cid.locs(index)
        self.insert(mechanism, ic=ic, mask_in=index, **kwargs)

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

    def _build(self):
        all_ions = get_unique_keys([self._ion_read, self._ion_write, self._ion_write_c])

        _ion_write = {f"i{k}": v for k, v in self._ion_write.items()}
        self._m_curr.update(_ion_write)

        df = self.integrator.is_df

        ions = {}
        for ion in all_ions:
            ion_write_c = self._ion_write_c.get(ion, {})
            ion_read = self._ion_read.get(ion, {})
            ion_style = self.get_ion_style(ion)
            ions[ion] = build_ion(
                ion,
                self.shape,
                self.n_ax,
                self.n_comp,
                self._m_list,
                self._m_name,
                ion_read,
                ion_write_c,
                *ion_style,
            )
            for m in self._m_list:
                m.register_ion(ions[ion])
        if self.builder is None:
            mech = build_handler(
                self._m_list,
                self._m_name,
                self._m_curr,
                self._m_unfactorable,
                self._m_has_gtot,
                self._m_divide_by_two,
                self.temp,
                ions,
                df,
            )
        else:
            mech = self.builder.build(
                self._m_list,
                self._m_name,
                self._m_curr,
                self._m_unfactorable,
                self._m_has_gtot,
                self._m_divide_by_two,
                self.temp,
                ions,
            )

        self.integrator = self.integrator(self, mech)

    def area_(self, diameters):
        raise NotImplementedError()

    def ra_(self, diameters):
        raise NotImplementedError()

    def cm_(self, area):
        return (self.cm / 1e3) * area

    def init_v(self):
        self.integrator.init_v(self)

    def detach(self):
        self.integrator.detach(self)

    @property
    def t(self):
        """
        Get the current simulation time.

        Returns
        -------
        float
            Current simulation time in milliseconds.
        """
        return self.t_ind * self.dt

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

    def run(
        self,
        ve=None,
        space=None,
        time=None,
        dt=None,
        intra=None,
        callbacks=None,
        reinit=False,
        progressbar=True,
        multicontact=False,
        first=True,
        longrunning=False,
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
        dt : float, optional
            Time step size in milliseconds. If None, uses the default from backend.
        intra : IntraStim, optional
            Intracellular stimulation object.
        callbacks : list of Callback, optional
            List of callback objects to execute during simulation steps.
        reinit : bool, optional
            If True, reinitialize the model state before running. If steady state is
            cached, it will be restored instead of initializing from scratch.
            Default is False.
        progressbar : bool or tqdm, optional
            If True, displays a progress bar during simulation. Can also be a
            tqdm instance for custom progress tracking. Default is True.
        multicontact : bool, optional
            If True, handles multiple electrode contacts for ve construction.
            Default is False.
        first : bool, optional
            If True, indicates this is the first run in a sequence, triggering
            pre-loop hooks for callbacks. Default is True.
        longrunning : bool, optional
            If True, indicates this run is part of a longer simulation sequence,
            affecting progress bar behavior. Default is False.

        Raises
        ------
        ValueError
            If neither ve nor (space and time) nor intra is provided.
            If intra is provided but is not an instance of IntraStim.

        Notes
        -----
        The simulation updates the model's internal state (v, v_prev for DF method, etc.)
        and advances the model's time index (t_ind).
        """

        with_intra = intra is not None
        if with_intra:
            if not isinstance(intra, IntraStim):
                raise ValueError("intra must be an instance of IntraStim")

        intra_only = False
        if ve is None and (space is None and time is None):
            if intra is None:
                raise ValueError(
                    "Either ve or ve_s and ve_t or intra must be provided."
                )
            intra_only = True
        ve_zero = torch.zeros_like(self.v)

        device = self.device()
        if ve is not None:
            ve = torch.as_tensor(ve, device=device)

        dt = dt if dt is not None else A.dt
        self.warn_about_dt(dt)
        self.dt = dt

        if isinstance(time, Waveform):
            if time._tstop is None:
                raise ValueError("Waveform must have a tstop value.")
            time = time.assemble(dt)

        with torch.set_grad_enabled(self.training):
            if ve is None and not intra_only:
                ve = ve_from_s_t(space, time, self.n_ax, self.device(), multicontact)

            if self.training:
                self.calculate_geometric_params()

            if (not self.initialized) or reinit:
                if "_steady_state" in self._caches:
                    self.restore("_steady_state")
                    self.t_ind = 0
                else:
                    self.integrator.init_v(self)
                    self.pre_initialize()
                    self.integrator.mech.initialize(self.v, self.v_init_c, self.temp_c)
                    self.post_initialize()
                    self.t_ind = 0
                    self.initialized = True
                if with_intra:
                    intra.init(self)
            else:
                self.detach()

            if first:
                if callbacks:
                    for c in callbacks:
                        c.dt = dt

                if not isinstance(callbacks, CallbackList):
                    callbacks = CallbackList(callbacks)
                
                pre_loop_hook(callbacks, self)

            dt = torch.as_tensor(dt, device=device)

            if first or self.training:
                self.integrator.initialize(self, dt)

            if progressbar:
                if not isinstance(progressbar, tqdm):
                    progressbar = tqdm(total=ve.shape[0], desc=f"{self.t:.3f} ms")

            for i in range(ve.shape[0]):
                ve_c = ve[i] if not intra_only else ve_zero
                if with_intra:
                    intra_c = intra(self.t_ind, self.v)
                    step_intra(self.integrator, self, ve_c, intra_c, dt, self.t_ind)
                else:
                    step(self.integrator, self, ve_c, dt, self.t_ind)
                
                post_step_hook(callbacks, self)
                self.t_ind += 1

                if progressbar:
                    progressbar.update(1)
                    if self.t_ind % 100 == 0:
                        progressbar.set_description(f"{self.t:.1f} ms")

            if not longrunning:
                if progressbar:
                    progressbar.close()
                post_loop_hook(callbacks, self)

    def longrun(
        self,
        space: Tensor,
        time: Tensor,
        chunklength: int,
        tstop: float = None,
        dt: float = None,
        reinit=False,
        callbacks: List[Callback] = None,
        intra: Optional[IntraStim] = None,
        progressbar=True,
        multicontact=False,
    ):
        """
        Run a long simulation by splitting it into multiple chunks.

        Parameters
        ----------
        space : Tensor
            Spatial components of extracellular voltage. Shape should be
            [n_ax, n_comp] or [1, n_comp] or [n_contacts, ...] for multicontact mode.
        time : Tensor or Waveform
            Temporal components of extracellular voltage. Shape should be
            [n_ax, n_timesteps] or [1, n_timesteps] or [n_contacts, ...] for multicontact mode.
        chunklength : int
            Length of chunks, in # timesteps, into which to split simulation.
        dt : float, optional
            Time step size in milliseconds. If None, uses the default from backend.
        reinit : bool, optional
            If True, reinitialize the model state before running the first chunk.
            Subsequent chunks will not reinitialize. Default is False.
        callbacks : list of Callback, optional
            List of callback objects to execute during simulation steps.
        intra : IntraStim, optional
            Intracellular stimulation object.
        progressbar : bool, optional
            If True, displays a progress bar during simulation. Default is True.
        multicontact : bool, optional
            If True, handles multiple electrode contacts for ve construction.
            Default is False.

        Notes
        -----
        This method uses the same numerical methods as the `run` method, but manages
        memory more efficiently for long simulations by processing the data in chunks.
        The state of the model (v, v_prev, etc.) is preserved between chunks.
        """

        with_intra = intra is not None
        if with_intra:
            if not isinstance(intra, IntraStim):
                raise ValueError("intra must be an instance of IntraStim")

        dt = dt if dt is not None else A.dt
        self.warn_about_dt(dt)

        ve_s = torch.as_tensor(space, device=self.device(), dtype=self.dtype())
        dt = torch.as_tensor(dt, device=self.device(), dtype=self.dtype())

        functional = False

        if tstop is not None:
            if not isinstance(time, Waveform):
                raise ValueError('`time` must be of type `Waveform`')
            time = time.to(self.dtype())
            functional = True
            t = torch.arange(0, tstop, dt, dtype=self.dtype())
            n_chunks = math.ceil(len(t) / chunklength)
            t_chunks = torch.tensor_split(t, n_chunks)

            ve_s = ve_s.expand(self.n_ax, -1)
            einsum = op_sc

            if progressbar:
                progressbar = tqdm(total=len(t), desc=f"{self.t:.1f} ms")

        else:
            if isinstance(time, Waveform):
                if time._tstop is None:
                    raise ValueError("Waveform must have a tstop value.")
                time = time.assemble(dt)

            ve_t = torch.as_tensor(time, device=self.device(), dtype=self.dtype())
            n_chunks = math.ceil(ve_t.shape[-1] / chunklength)

            # ve_s : [n_ax, n_comp] or [1, n_comp] or [n_contacts, *]
            # ve_t : [n_ax, n_timesteps] or [1, n_timesteps] or [n_contacts, *]

            if multicontact:
                ve_s = ve_s.expand(-1, self.n_ax, -1)
                ve_t = ve_t.expand(-1, self.n_ax, -1)
                einsum = op_mc
            else:
                ve_s = ve_s.expand(self.n_ax, -1)
                ve_t = ve_t.expand(self.n_ax, -1)
                einsum = op_sc

            t_chunks = torch.tensor_split(ve_t, n_chunks, dim=-1)

            if progressbar:
                progressbar = tqdm(total=ve_t.shape[-1], desc=f"{self.t:.1f} ms")

        if callbacks:
            for c in callbacks:
                c.dt = dt

        callbacks = CallbackList(callbacks)

        if (not self.initialized) or reinit:
            if "_steady_state" in self._caches:
                self.restore("_steady_state")
                self.t_ind = 0
            else:
                self.integrator.init_v(self)
                self.pre_initialize()
                self.integrator.mech.initialize(self.v, self.v_init_c, self.temp_c)
                self.post_initialize()
                self.t_ind = 0
                self.initialized = True
            if with_intra:
                intra.init(self)
        else:
            self.detach()

        self.integrator.initialize(self, dt)

        pre_loop_hook(callbacks, self)

        for i, t_chunk in enumerate(t_chunks):
            if functional:
                t = time(t_chunk).to(self.dtype())
            else:
                t = t_chunk
            ve_ = einsum(ve_s, t)
            for ve in torch.unbind(ve_, 0):
                step(self.integrator, self, ve, dt, self.t_ind)
                post_step_hook(callbacks, self)
                self.t_ind += 1

                if progressbar:
                    progressbar.update(1)
                    if self.t_ind % 100 == 0:
                        progressbar.set_description(f"{self.t:.1f} ms")

        post_loop_hook(callbacks, self)

        if progressbar:
            progressbar.close()

    def steady_state(self, dt=0.2, tstop=200.0):
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
        The steady state can be restored later by setting reinit=True when calling
        the run method.
        """

        dt = torch.as_tensor(dt, device=self.device(), dtype=self.dtype())

        if "_steady_state" in self._caches:
            self._caches.pop("_steady_state")
        ve = torch.zeros(self.n_ax, self.n_comp, device=self.device(), dtype=self.dtype())
        self.integrator.initialize(self, dt)
        maxiter = int(tstop / dt)
        with ctx(DTWARN=0):
            for i in tqdm(range(maxiter), desc=f"Steady state [dt:{dt} ms, tstop:{tstop} ms]"):
                reinit = i == 0
                step(self.integrator, self, ve, dt, self.t_ind)
        self.cache("_steady_state")
        self.t_ind = 0
        return self

    def post_initialize(self):
        with torch.no_grad():
            for h in self.post_initialize_hooks:
                h(self)

    def pre_initialize(self):
        with torch.no_grad():
            for h in self.pre_initialize_hooks:
                h(self)

    def initialize(self, v, v_init, temp):
        self.integrator.mech.initialize(v, v_init, temp)

    def load(self, state_dict):
        """
        Load model weights from a state dictionary.

        This method supports loading weights from:
        1. A key from the predefined `all_trained` dictionary
        2. A file path as a string
        3. An actual state dictionary object

        The loaded weights are matched to the model's current state dict structure
        and only compatible weights are loaded. After loading, geometric parameters
        are recalculated.

        Parameters
        ----------
        state_dict : str or dict
            Can be one of:
            - A key from the predefined `all_trained` dictionary
            - A file path to a saved model state
            - A state dictionary object

        Returns
        -------
        self
            The model instance with loaded weights
        """
        from axonml import all_trained

        if state_dict in all_trained:
            state_dict = torch.load(
                all_trained[state_dict], map_location=self.device(), weights_only=True
            )
        elif isinstance(state_dict, str):
            state_dict = torch.load(
                state_dict, map_location=self.device(), weights_only=True
            )
        matched, _ = _match_state_dict(self.state_dict(), state_dict)
        self.load_state_dict(matched, strict=False)
        self.calculate_geometric_params()
        return self

    def compile(self, callbacks: List[Callback] = None):
        ve = torch.ones(
            1, self.n_ax, 1, self.n_comp, device=self.device(), dtype=self.dtype()
        )
        for _ in range(5):
            self.run(ve, callbacks=callbacks, progressbar=False)
        self.initialized = False
        if callbacks:
            for c in callbacks:
                c.reset()
        return self

    def all_states(self) -> List[str]:
        out = ["v"]
        return out + self.integrator.mech.all_states()

    def set(self, key: str, value: float):
        self.integrator.mech.set(key, value)

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

    def cuda(self):
        super().cuda()
        self.integrator.mech.set_buffers(self.diam)
        return self

    def cpu(self):
        super().cpu()
        self.integrator.mech.set_buffers(self.diam)
        return self

    def float(self):
        super().float()
        self.integrator.mech.set_buffers(self.diam)
        return self

    def double(self):
        super().double()
        self.integrator.mech.set_buffers(self.diam)
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.integrator.mech.set_buffers(self.diam)
        return self

    def warn_about_dt(self, dt):
        if DTWARN:
            if self._dt_lim is not None:
                if dt > self._dt_lim:
                    warnings.warn(
                        f"dt ({dt}) exceeds limit ({self._dt_lim}), solution may have large oscillations."
                    )


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
    temp : float, optional
        Temperature in degrees Celsius. Default is 37°C.
    v_init : float, optional
        Initial membrane potential in mV. Default is -80 mV.
    method : str, optional
        Integration method. One of 'euler', 'rk1', 'heun', 'rk2', 'rk4',
        'dufort-frankel', or 'df'. Default is 'rk1'.

    Attributes
    ----------
    dx : float
        Spatial discretization step in μm.
    n_comp : int
        Number of compartments in the model (calculated based on L and dx).
    cm : float
        Membrane capacitance in μF/cm².
    rhoa : float
        Axial resistivity in Ω·cm.

    Methods
    -------
    x()
        Returns spatial positions of nodes in μm.
    area_(diameters)
        Calculates membrane surface area in cm² for given diameters.
    ra_(diameters)
        Calculates axial resistance in MΩ for given diameters.

    Notes
    -----
    The number of nodes is calculated to ensure it's odd (for a centered node at position 0)
    and to maintain symmetry by rounding to the next even number of segments.

    See Also
    --------
    Axon : Base class providing common functionality for axon models.
    Myelinated : Companion class implementing myelinated axon models.
    """

    PARAMETER(cm=1.0, rhoa=35.4)

    def __init__(
            self, 
            diameters, 
            L=1.0*mm, 
            dx=10.0, 
            temp=37, 
            v_init=-80,
            integrator=dufort_frankel()
        ):
        # L = L * 1000  # mm -> um
        n_comp = L / dx
        n_comp = math.ceil(n_comp) // 2 * 2 + 1
        self.dx: float = dx
        self.L: float = n_comp * dx
        super().__init__(diameters, n_comp, temp, v_init, integrator)

    def x(self) -> torch.Tensor:  # x in um
        l = (self.n_comp - 1) * self.dx
        x = torch.linspace(-l / 2, l / 2, self.n_comp, device=self.device())
        return torch.atleast_2d(x)

    def area_(self, diameters) -> torch.Tensor:
        dx = torch.full_like(diameters, self.dx / 10000)
        return torch.pi * (diameters / 10000) * dx

    def ra_(self, diameters) -> torch.Tensor:
        dx = torch.full_like(diameters, self.dx / 10000)
        radii = diameters / 20000
        return (self.rhoa * dx) / (torch.pi * (radii**2))


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
    n_comp : int
        Number of compartments (nodes) in the model.
    temp : float, optional
        Temperature in degrees Celsius. Default is 37°C.
    v_init : float, optional
        Initial membrane potential in mV. Default is -80 mV.
    method : str, optional
        Integration method. One of 'euler', 'rk1', 'heun', 'rk2', 'rk4',
        'dufort-frankel', or 'df'. Default is 'rk1'.
    beta : float, optional
        Hyperdiffusion coefficient for numerical stability. Default is 0.0.

    Attributes
    ----------
    node_l : float
        Length of the nodes of Ranvier in μm. Default is 2.0 μm.
    axond1, axond2, axond3 : float
        Coefficients for the quadratic equation calculating axon diameter.
        Default values are 0.0, 0.7, and 0.0, respectively.
    noded1, noded2, noded3 : float
        Coefficients for the quadratic equation calculating node diameter.
        Default values are 0.0, 0.7, and 0.0, respectively.
    deltax1, deltax2, deltax3 : float
        Coefficients for the quadratic equation calculating internodal distance.
        Default values are 0.0, 100.0, and 0.0, respectively.
    cm : float
        Membrane capacitance in μF/cm². Default is 1.0.
    rhoa : float
        Axial resistivity in Ω·cm. Default is 35.4.

    Methods
    -------
    x()
        Returns spatial positions of nodes in μm.
    area_(diameters)
        Calculates membrane surface area in cm² for given diameters.
    ra_(diameters)
        Calculates axial resistance in MΩ for given diameters.
    axonD(diameters)
        Calculates axon diameter based on fiber diameter.
    nodeD(diameters)
        Calculates node diameter based on fiber diameter.
    deltax(diameters)
        Calculates internodal distance based on fiber diameter.
    rhoa_scale(diameters)
        Calculates scaling factor for axial resistivity based on fiber diameter.

    Notes
    -----
    The model uses quadratic equations to calculate various geometric parameters
    based on the fiber diameter, following anatomical scaling relationships.

    See Also
    --------
    Axon : Base class providing common functionality for axon models.
    Unmyelinated : Companion class implementing unmyelinated axon models.
    """

    PARAMETER(
        node_l=2.0,
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
        membrane={
            "cm": 1.0,
            "rhoa": 35.4,  # ohm-cm
        },
    )

    def area_(self, diameters):
        lengths = self.node_l * torch.ones_like(diameters) / 10000
        return torch.pi * self.nodeD(diameters) * lengths  # cm2

    def ra_(self, diameters):
        radii = diameters / 20000  # radius in cm
        rhoa = self.rhoa * self.rhoa_scale(diameters)
        return (rhoa * self.deltax(diameters)) / (torch.pi * (radii**2))

    def rhoa_scale(self, diameters):
        return 1 / ((self.axonD(diameters) / diameters) ** 2)

    def axonD(self, diameters):
        axond = self.axond1 * diameters**2 + self.axond2 * diameters + self.axond3
        return axond

    def deltax(self, diameters):
        deltax = self.deltax1 * diameters**2 + self.deltax2 * diameters + self.deltax3
        return deltax / 10000

    def nodeD(self, diameters):
        noded = self.noded1 * diameters**2 + self.noded2 * diameters + self.noded3
        return noded / 10000

    def x(self) -> torch.Tensor:  # x in um
        l = (self.n_comp - 1) * self.deltax(self.diam) * 10000
        start = -l / 2
        end = l / 2
        steps = self.n_comp
        t = torch.linspace(0, 1, steps, device=l.device).unsqueeze(-1)
        return ((1 - t) * start + t * end).T


def pre_loop_hook(c, m):
    c.pre_loop_hook(m)

def post_loop_hook(c, m):
    c.post_loop_hook(m)

def post_step_hook(c, m):
    c.post_step_hook(m)