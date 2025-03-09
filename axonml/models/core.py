import math
from typing import List, Tuple, Optional, Dict, Callable
import re
import itertools
import warnings

import torch
from torch import Tensor

from tqdm.auto import tqdm

from axonml.models.stim.intrastim import IntraStim

from axonml.models.callbacks import CallbackList, Callback
from axonml.models.backend import Backend as A
from axonml.models.parametric import Parameterized
from axonml.models.mechanisms.core import Mechanism, validate
from axonml.models.mechanisms.declarations import PARAMETER
from axonml.models.mechanisms.handler.handler import build_handler
from axonml.models.mechanisms.handler.ions import build_ion
from axonml.models.mechanisms.mech_compiler import compile_mechanism

from axonml.helpers import op_mc, op_sc, ve_from_s_t, IMEM, CUDA, DTWARN, ctx


def get_unique_keys(list_of_dicts):
    """Gets all unique keys from a list of dictionaries.

    Args:
        list_of_dicts: A list of dictionaries.

    Returns:
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


@torch.jit.interface
class HandlerInterface:
    def initialize(self, v, v_init, temp) -> None:
        pass

    def advance(self, v, dt) -> None:
        pass

    def detach(self) -> None:
        pass

    def i_intra(self, v, intra) -> torch.Tensor:
        pass

    def i(self, v) -> torch.Tensor:
        pass

    def update(self, temp) -> None:
        pass

    def get(self, mech: str, state: str) -> torch.Tensor:
        pass

    def set(self, name: str, value: float) -> None:
        pass

    def all_states(self) -> List[str]:
        pass


class SymmetricConv1D(torch.nn.Conv1d):
    def forward(self, x):
        if self.training:
            weight_ = (self.weight + torch.flip(self.weight, [-1])) / 2
        else:
            weight_ = self.weight
        return self._conv_forward(x, weight_, self.bias)


class Axon(Parameterized, torch.jit.ScriptModule):
    """Base 1D fiber class."""
    _dt_lim = None
    __constants__ = ["method", "n_ax", "n_node", "temp", "v_init", "pade", "is_df", "use_fast_imem", "hd"]

    def __init__(
        self, diameters, n_node: int, temp=37.0, v_init=-80.0, method="rk1", pade=None, beta=0.0
    ):
        super().__init__()

        self.method_conversion = {
            "euler": "rk1",
            "rk1": "rk1",
            "heun": "rk2",
            "rk2": "rk2",
            "rk4": "rk4",
            "dufort-frankel": "df",
            "df": "df",
        }

        self.pade = pade
        self.is_df = self.method_conversion[method] == "df"

        if method not in self.method_conversion:
            raise ValueError(
                f"Invalid method: {method}. Valid methods: {list(self.method_conversion.keys())}"
            )

        # self.pi = torch.nn.Parameter(torch.tensor(math.pi), requires_grad=False)

        self.method = method
        self.n_ax = len(diameters)
        self.n_node = n_node
        self.temp = temp
        self.v_init = v_init

        self.mech: HandlerInterface = None
        self.t_ind: int = 0
        self.dt: float = A.dt

        self._m_list = []
        self._m_name = []
        self._m_curr = {}
        self._m_unfactorable = {}
        self._m_has_gtot = {}

        self._ion_read = {}
        self._ion_write = {}
        self._ion_write_c = {}

        self._all_read = {}
        self._all_write = {}
        self._all_write_c = {}

        self._ion_style = {}

        self.post_initialize_hooks: List[Callable] = []

        self._caches = {}

        self.use_fast_imem = bool(IMEM)
        if self.use_fast_imem:
            self.register_buffer("i_membrane", torch.zeros((self.n_ax, 1, n_node)))

        self.beta = beta
        self.hd = bool(beta) # hyper-diffusion
        if self.hd:
            self.filter = torch.nn.Conv1d(1, 1, 5, padding=2, bias=False, padding_mode="reflect")
            self.filter.weight.data = torch.tensor([-1, 4, -6, 4, -1], dtype=torch.float).reshape(1, 1, 5)
            for p in self.filter.parameters():
                p.requires_grad = False

        self.weight_choices = {
            "rk1": [[1.0, -2.0, 1.0], [1.0, -2.0, 1.0]],
            "rk2": [[1.0, -2.0, 1.0], [1.0, -2.0, 1.0]],
            "rk4": [[1.0, -2.0, 1.0], [1.0, -2.0, 1.0]],
            "df": [
                [1.0, +0.0, 1.0],
                [0.0, -1.0, 0.0],
                [1.0, -2.0, 1.0],
            ],
        }

        self.nc = {"rk1": 2, "rk2": 2, "rk4": 2, "df": 3}

        # solver stuff
        weight = self.weight_choices[self.method_conversion[method]]
        nc = self.nc[self.method_conversion[method]]

        self.ssd = SymmetricConv1D(
            nc, 1, 3, bias=False, padding="same", padding_mode="reflect"
        )
        self.ssd.weight.data = torch.tensor(weight).reshape(1, nc, 3)
        for p in self.ssd.parameters():
            p.requires_grad = False

        self.register_buffer("v", torch.full((self.n_ax, 1, n_node), v_init))
        if self.is_df:
            self.register_buffer("v_prev", torch.full((self.n_ax, 1, n_node), v_init))

        if torch.is_tensor(diameters):
            diameters = diameters.to(self.dtype()).clone().detach()
        else:
            diameters = torch.tensor(diameters, dtype=self.dtype())

        self._register_buffers(diameters)

        self.initialized: bool = False

        # -- constants --
        self.eval()

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
        if changed: self.calculate_geometric_params()
        self._build()
        if CUDA: self.cuda()

    def _register_buffers(self, diameters):
        self.register_buffer("diam", diameters)
        self.register_buffer("area_c", self.area_(self.diam)[:, None, None])
        self.register_buffer("cm_c", self.cm_(self.area_c))
        self.register_buffer("ra_c", self.ra_(self.diam)[:, None, None])
        self.register_buffer("v_init_c", torch.tensor(self.v_init))
        self.register_buffer("temp_c", torch.tensor(self.temp))

    def set_diam(self, diams):
        diams = torch.as_tensor(diams, dtype=self.dtype())
        self.diam[:] = diams
        self.instantiate_parameters_lambda()
        self.calculate_geometric_params()

    def calculate_geometric_params(self):
        self.area_c = self.area_(self.diam)[:, None, None]
        self.cm_c = self.cm_(self.area_c)
        self.ra_c = self.ra_(self.diam)[:, None, None]

    def unfreeze(self, *names):
        """
        Unfreezes the parameters of the model for training.
        If no parameter names are provided, all parameters of the model will be unfrozen.
        If specific parameter names are provided, only those parameters will be unfrozen.

        Args:
            *names (str): Variable length argument list of parameter names to unfreeze.

        Example:
            >>> model = SMF()
            >>> model.unfreeze('axnode_myel.gnabar', 'axnode_myel.gkbar')
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

    @torch.jit.export
    def n(self) -> int:
        return self.v.shape[0]

    def device(self):
        return self.ssd.weight.device

    def dtype(self):
        return self.ssd.weight.dtype

    def insert(
        self,
        mechanism: Mechanism,
        ic: Dict[str, float] = None,
        mask_out: str | int | slice = None,
        mask_in: str | int | slice = None,
        **kwargs,
    ):
        """Inserts a mechanism into the model.
        
        This method validates and compiles the given mechanism, then adds it to the model's
        mechanism list. It also registers the mechanism's currents and ion interactions.
        
        Args:
            mechanism: The mechanism to be inserted into the model.
            ic: Optional dictionary of initial conditions for the mechanism states.
                Keys are state names and values are initial values.
            mask_out: Optional mask specifying compartments for which the mechanism will not
                contribute to the current calculation. Can be a string, integer, or slice.
            mask_in: Optional mask specifying compartments for which the mechanism will not
                receive input. Can be a string, integer, or slice.
            **kwargs: Additional keyword arguments to be passed to the compile_mechanism function.
        """
        validate(mechanism)

        m, unfactorable, has_gtot = compile_mechanism(
            mechanism,
            self,
            ic=ic,
            mask_out=mask_out,
            mask_in=mask_in,
            **kwargs,
        )

        self._m_list.append(m)
        self._m_name.append(mechanism.__name__)
        self._m_unfactorable[mechanism.__name__] = unfactorable
        self._m_has_gtot[mechanism.__name__] = has_gtot

        for k, v in mechanism._currents.items():
            self._m_curr.setdefault(k, {}).update({mechanism.__name__: v})

        for k, v in mechanism._read_ion.items():
            self._ion_read.setdefault(k, {}).update({mechanism.__name__: v})

        for k, v in mechanism._write_ion.items():
            self._ion_write.setdefault(k, {}).update({mechanism.__name__: v})

        for k, v in mechanism._write_ion_c.items():
            self._ion_write_c.setdefault(k, {}).update({mechanism.__name__: v})

    def ion_style(self, ion, c_style, e_style, einit, eadvance, cinit):
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

        df = self.is_df

        ions = {}
        for ion in all_ions:
            ion_write_c = self._ion_write_c.get(ion, {})
            ion_read = self._ion_read.get(ion, {})
            ion_style = self.get_ion_style(ion)
            ions[ion] = build_ion(
                ion,
                self.n_ax,
                self.n_node,
                self._m_list,
                self._m_name,
                ion_read,
                ion_write_c,
                *ion_style,
            )
            for m in self._m_list:
                m.register_ion(ions[ion])

        self.mech = build_handler(
            self._m_list,
            self._m_name,
            self._m_curr,
            self._m_unfactorable,
            self._m_has_gtot,
            self.temp,
            ions,
            df,
        )

    def area_(self, diameters):
        raise NotImplementedError()

    def ra_(self, diameters):
        raise NotImplementedError()

    def cm_(self, area):
        return (self.cm / 1e3) * area

    def init_v(self):
        self.v[:] = self.v_init
        self.v.detach_()
        if self.is_df:
            self.v_prev[:] = self.v_init
            self.v_prev.detach_()

    def detach(self):
        self.v.detach_()
        if self.is_df:
            self.v_prev.detach_()
        self.mech.detach()

    @property
    def t(self):
        """Returns the current simulation time.
        
        This property calculates the current simulation time by multiplying
        the time index (t_ind) by the time step size (dt).
        
        Returns:
            float: Current simulation time in milliseconds.
        """
        return self.t_ind * self.dt

    def c(self, *args):
        """Converts relative positions to node indices.
        
        This method takes relative positions along the axon (as fractions between 0 and 1)
        and converts them to the corresponding node indices. For example, 0.5 represents
        the middle node, 0 the first node, and 1 the last node.
        
        Args:
            *args: Variable number of float values between 0 and 1, representing
                relative positions along the axon.
        
        Returns:
            list: A list of integer node indices corresponding to the input positions.
            
        Example:
            >>> model.c(0.25, 0.5, 0.75)
            [25, 50, 75]  # For a model with n_node=101
        """
        return [round((self.n_node - 1) * i) for i in args]

    def run(
        self,
        ve: Tensor = None,
        space: Tensor = None,
        time: Tensor = None,
        dt: float = None,
        intra: Optional[IntraStim] = None,
        callbacks: List[Callback] = None,
        reinit: bool = False,
        progressbar: bool = True,
        multicontact: bool = False,
        first: bool = True,
        longrunning: bool = False,
    ):
        """Runs the axon model simulation for the specified input and parameters.
        
        This method performs the numerical integration of the cable equation to simulate
        the axon's response to extracellular and/or intracellular stimulation. It can use
        different numerical methods (Euler/RK1, Heun/RK2, RK4, or Dufort-Frankel) as 
        specified during model initialization.
        
        Args:
            ve: Optional tensor of extracellular voltage. Shape should be 
                [timesteps, n_ax, 1, n_node] or compatible.
            space: Optional tensor for spatial components when ve is not directly provided.
                Used with time to construct ve.
            time: Optional tensor for temporal components when ve is not directly provided.
                Used with space to construct ve.
            dt: Time step size in milliseconds. If None, uses the default from backend.
            intra: Optional intracellular stimulation object. Must be an instance of IntraStim.
            callbacks: List of callback objects to execute during simulation steps.
            reinit: If True, reinitialize the model state before running. If steady state is
                cached, it will be restored instead of initializing from scratch.
            progressbar: If True, displays a progress bar during simulation. Can also be a
                tqdm instance for custom progress tracking.
            first: If True, indicates this is the first run in a sequence, triggering
                pre-loop hooks for callbacks.
            multicontact: If True, handles multiple electrode contacts for ve construction.
            longrunning: If True, indicates this run is part of a longer simulation sequence,
                affecting progress bar behavior.
        
        Raises:
            ValueError: If neither ve nor (space and time) nor intra is provided.
            ValueError: If intra is provided but is not an instance of IntraStim.
        
        Note:
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

        method = getattr(self, f"step_no_intra_{self.method_conversion[self.method]}")
        method_intra = getattr(
            self, f"step_intra_{self.method_conversion[self.method]}"
        )

        df = self.is_df

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
                    self.init_v()
                    self.initialize(self.v, self.v_init_c, self.temp_c)
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
                callbacks.pre_loop_hook(self)

            dt = torch.as_tensor(dt, device=device)

            if df:
                s = 2 * dt / self.cm_c
                s2 = s / self.ra_c
            else:
                cm_inv = 1 / self.cm_c
                ra_inv = 1 / self.ra_c

            if progressbar:
                if not isinstance(progressbar, tqdm):
                    progressbar = tqdm(total=ve.shape[0], desc=f"{self.t:.3f} ms")

            for i in range(ve.shape[0]):
                ve_c = ve[i] if not intra_only else ve_zero
                if df:
                    if with_intra:
                        self.v, self.v_prev = method_intra(
                            self.v,
                            self.v_prev,
                            ve_c,
                            self.area_c,
                            s,
                            s2,
                            dt,
                            self.temp_c,
                            intra(self.t_ind, self.v),
                        )
                    else:
                        self.v, self.v_prev = method(
                            self.v,
                            self.v_prev,
                            ve_c,
                            self.area_c,
                            s,
                            s2,
                            dt,
                            self.temp_c,
                        )
                else:
                    if with_intra:
                        self.v = method_intra(
                            self.v,
                            ve_c,
                            self.area_c,
                            cm_inv,
                            ra_inv,
                            dt,
                            self.temp_c,
                            intra(self.t_ind, self.v),
                        )
                    else:
                        self.v = method(
                            self.v, ve_c, self.area_c, cm_inv, ra_inv, dt, self.temp_c
                        )
                callbacks.post_step_hook(self)
                self.t_ind += 1

                if progressbar:
                    progressbar.update(1)
                    if self.t_ind % 100 == 0:
                        progressbar.set_description(f"{self.t:.1f} ms")

            if not longrunning:
                if progressbar:
                    progressbar.close()

    def longrun(
        self,
        space: Tensor,
        time: Tensor,
        n_chunks: int,
        dt: float = None,
        reinit=False,
        callbacks: List[Callback] = None,
        intra: Optional[IntraStim] = None,
        progressbar=True,
        multicontact=False,
    ):
        """Runs a long simulation by splitting it into multiple chunks.
        
        This method handles large-scale simulations by dividing the temporal component
        into smaller chunks and running them sequentially. This approach helps manage
        memory usage for very long simulations, as it avoids creating a single large
        extracellular voltage tensor.
        
        Args:
            space: Tensor for spatial components of extracellular voltage. Shape should be
                [n_ax, n_node] or [1, n_node] or [n_contacts, ...] for multicontact mode.
            time: Tensor for temporal components of extracellular voltage. Shape should be
                [n_ax, n_timesteps] or [1, n_timesteps] or [n_contacts, ...] for multicontact mode.
            n_chunks: Number of chunks to split the temporal component into.
            dt: Time step size in milliseconds. If None, uses the default from backend.
            reinit: If True, reinitialize the model state before running the first chunk.
                Subsequent chunks will not reinitialize.
            callbacks: List of callback objects to execute during simulation steps.
            progressbar: If True, displays a progress bar during simulation.
            multicontact: If True, handles multiple electrode contacts for ve construction.
        
        Note:
            This method uses the same numerical methods as the `run` method, but manages
            memory more efficiently for long simulations by processing the data in chunks.
            The state of the model (v, v_prev, etc.) is preserved between chunks.
        """
        
        ve_s = torch.as_tensor(space, device=self.device())
        ve_t = torch.as_tensor(time, device=self.device())

        # ve_s : [n_ax, n_node] or [1, n_node] or [n_contacts, *]
        # ve_t : [n_ax, n_timesteps] or [1, n_timesteps] or [n_contacts, *]

        if multicontact:
            ve_s = ve_s.expand(-1, self.n_ax, -1)
            ve_t = ve_t.expand(-1, self.n_ax, -1)
            einsum = op_mc
        else:
            ve_s = ve_s.expand(self.n_ax, -1)
            ve_t = ve_t.expand(self.n_ax, -1)
            einsum = op_sc

        dt = dt if dt is not None else A.dt
        self.warn_about_dt(dt)

        t_chunks = torch.tensor_split(ve_t, n_chunks, dim=-1)

        if progressbar:
            progressbar = tqdm(total=ve_t.shape[-1], desc=f"{self.t:.1f} ms")

        if callbacks:
            for c in callbacks:
                c.dt = dt

        callbacks = CallbackList(callbacks)

        for i, t_chunk in enumerate(t_chunks):
            if (i == 0) and reinit:
                reinit = True
            else:
                reinit = False
            ve = einsum(ve_s, t_chunk)
            with ctx(DTWARN=0):
                self.run(
                    ve,
                    dt=dt,
                    intra=intra,
                    callbacks=callbacks,
                    reinit=reinit,
                    progressbar=progressbar,
                    first=(i == 0),
                    longrunning=True,
                )

        if progressbar:
            progressbar.close()

    def steady_state(self, dt=0.2, maxiter=3000):
        """Runs the model until it reaches a steady state and caches the result.
        
        This method simulates the axon model with zero extracellular voltage for a 
        specified number of iterations to allow the model to reach a stable membrane
        potential. The resulting steady state is then cached for later use in 
        simulations that require initialization from a steady state.
        
        Args:
            dt: Time step size in milliseconds. Default is 0.2 ms.
            maxiter: Maximum number of iterations to run the simulation. Default is 3000.
            
        Note:
            This method clears any previous steady state cache before creating a new one.
            The steady state can be restored later by setting reinit=True when calling 
            the run method.
        """
        
        if "_steady_state" in self._caches:
            self._caches.pop("_steady_state")
        ve = torch.zeros(1, self.n_ax, 1, self.n_node, device=self.device())
        with ctx(DTWARN=0):
            for i in tqdm(range(maxiter), desc="Steady state..."):
                reinit = i == 0
                self.run(ve, dt, reinit=reinit, progressbar=False)
        self.cache("_steady_state")

    def post_initialize(self):
        for h in self.post_initialize_hooks:
            h(self)

    @torch.jit.script_method
    def initialize(self, v, v_init, temp):
        self.mech.initialize(v, v_init, temp)

    @torch.jit.script_method
    def FRK(self, v, ve, area, cm, ra):
        x = torch.cat([v, ve], dim=1)
        d2v = self.ssd(x)
        i_ion = self.mech.i(v, v) * area
        return cm * ((ra * d2v) - i_ion)

    @torch.jit.script_method
    def FRK_intra(self, v, ve, area, cm, ra, intra):
        x = torch.cat([v, ve], dim=1)
        d2v = self.ssd(x)
        i_ion = self.mech.i(v, v) * area - intra
        return cm * ((ra * d2v) - i_ion)

    @torch.jit.script_method
    def step_no_intra_rk1(self, v, ve, area, cm, ra, dt, temp) -> Tensor:
        K1 = self.FRK(v, ve, area, cm, ra)
        self.mech.advance(v, dt, temp)
        v_n = v + K1 * dt
        if self.use_fast_imem:
            i_cap = self.cm_c * (v_n - v) / dt
            self.i_membrane[:] = i_cap + self.mech.imem * area
        return v_n

    @torch.jit.script_method
    def step_intra_rk1(self, v, ve, area, cm, ra, dt, temp, intra) -> Tensor:
        K1 = self.FRK_intra(v, ve, area, cm, ra, intra)
        self.mech.advance(v, dt, temp)
        v_n = v + K1 * dt
        if self.use_fast_imem:
            i_cap = self.cm_c * (v_n - v) / dt
            self.i_membrane[:] = i_cap + self.mech.imem * area
        return v_n

    @torch.jit.script_method
    def step_no_intra_rk2(self, v, ve, area, cm, ra, dt, temp) -> Tensor:
        K1 = self.FRK(v, ve, area, cm, ra)
        K2 = self.FRK(v + K1 * dt, ve, area, cm, ra)
        self.mech.advance(v, dt, temp)
        v_n = v + (K1 + K2) * (dt / 2)
        if self.use_fast_imem:
            i_cap = self.cm_c * (v_n - v) / dt
            self.i_membrane[:] = i_cap + self.mech.imem * area
        return v_n

    @torch.jit.script_method
    def step_intra_rk2(self, v, ve, area, cm, ra, dt, temp, intra) -> Tensor:
        K1 = self.FRK_intra(v, ve, area, cm, ra, intra)
        K2 = self.FRK_intra(v + K1 * dt, ve, area, cm, ra, intra)
        self.mech.advance(v, dt, temp)
        v_n = v + (K1 + K2) * (dt / 2)
        if self.use_fast_imem:
            i_cap = self.cm_c * (v_n - v) / dt
            self.i_membrane[:] = i_cap + self.mech.imem * area
        return v_n

    @torch.jit.script_method
    def step_no_intra_rk4(self, v, ve, area, cm, ra, dt, temp) -> Tensor:
        # -- update vm --
        K1 = self.FRK(v, ve, area, cm, ra)
        K2 = self.FRK(v + (dt / 2) * K1, ve, area, cm, ra)
        K3 = self.FRK(v + (dt / 2) * K2, ve, area, cm, ra)
        K4 = self.FRK(v + dt * K3, ve, area, cm, ra)

        self.mech.advance(v, dt, temp)

        v_n = v + (dt / 6) * (K1 + 2 * K2 + 2 * K3 + K4)
        if self.use_fast_imem:
            i_cap = self.cm_c * (v_n - v) / dt
            self.i_membrane[:] = i_cap + self.mech.imem * area
        return v_n

    @torch.jit.script_method
    def step_intra_rk4(
        self,
        v,
        ve,
        area,
        cm,
        ra,
        dt,
        temp,
        intra,
    ) -> Tensor:
        # -- update vm --
        K1 = self.FRK_intra(v, ve, area, cm, ra, intra)
        K2 = self.FRK_intra(v + (dt / 2) * K1, ve, area, cm, ra, intra)
        K3 = self.FRK_intra(v + (dt / 2) * K2, ve, area, cm, ra, intra)
        K4 = self.FRK_intra(v + dt * K3, ve, area, cm, ra, intra)

        self.mech.advance(v, dt, temp)

        v_n = v + (dt / 6) * (K1 + 2 * K2 + 2 * K3 + K4)
        if self.use_fast_imem:
            i_cap = self.cm_c * (v_n - v) / dt
            self.i_membrane[:] = i_cap + self.mech.imem * area
        return v_n

    @torch.jit.script_method
    def step_no_intra_df(
        self, v, v_prev, ve, area, s, s2, dt, temp
    ) -> Tuple[Tensor, Tensor]:
        # -- 2nd diff --
        x = torch.cat([v, v_prev, ve], dim=1)
        d2v = self.ssd(x)

        # -- calculate ionic current --
        i_ion = self.mech.i(v_prev, v) * area

        # -- update vm --
        v_new = (v_prev + s2 * d2v - s * i_ion) / (1 + s2 + s * self.mech.gtot() * area)

        self.mech.itot(v)
        self.mech.advance(v, dt, temp)

        if self.use_fast_imem:
            i_cap = self.cm_c * (v_new - v_prev) / (2 * dt)
            self.i_membrane[:] = i_cap + self.mech.imem * area

        if self.hd:
            v_new = v_new - self.beta * self.filter(v_new)

        return v_new, v

    @torch.jit.script_method
    def step_intra_df(
        self, v, v_prev, ve, area, s, s2, dt, temp, intra
    ) -> Tuple[Tensor, Tensor]:
        # -- 2nd diff --
        x = torch.cat([v, v_prev, ve], dim=1)
        d2v = self.ssd(x)

        # -- calculate ionic current --
        i_ion = self.mech.i(v_prev, v) * area - intra

        # -- update vm --
        v_new = (v_prev + s2 * d2v - s * i_ion) / (1 + s2 + s * self.mech.gtot() * area)

        self.mech.itot(v)
        self.mech.advance(v, dt, temp)

        if self.use_fast_imem:
            i_cap = self.cm_c * (v_new - v_prev) / (2 * dt)
            self.i_membrane[:] = i_cap + self.mech.imem

        if self.hd:
            v_new = v_new - self.beta * self.filter(v_new)

        return v_new, v

    @torch.jit.script_method
    def get_state(self, s: str) -> Tensor:
        if s == "v":
            return self.v
        mech, state = s.split(".")
        return self.mech.get(mech, state)

    def load(self, state_dict):
        from axonml import all_trained

        if state_dict in all_trained:
            state_dict = torch.load(
                all_trained[state_dict], map_location=self.device(), weights_only=True
            )
        elif isinstance(state_dict, str):
            state_dict = torch.load(
                state_dict, map_location=self.device(), weights_only=True
            )
        matched, _ = match_state_dict(self.state_dict(), state_dict)
        self.load_state_dict(matched, strict=False)
        self.calculate_geometric_params()
        return self

    def compile(self, callbacks: List[Callback] = None):
        ve = torch.ones(
            1, self.n_ax, 1, self.n_node, device=self.device(), dtype=self.dtype()
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
        return out + self.mech.all_states()

    @torch.jit.export
    def set(self, key: str, value: float):
        self.mech.set(key, value)

    @torch.jit.export
    def cache(self, name: str = None):
        if name is None:
            name = "latest"
        self._caches[name] = self.state_dict()

    @torch.jit.export
    def restore(self, name: str = None):
        if name is None:
            name = "latest"
        self.load_state_dict(self._caches[name])
        self.initialized = True

    def cuda(self):
        super().cuda()
        self.mech.set_buffers(self.diam)
        return self

    def cpu(self):
        super().cpu()
        self.mech.set_buffers(self.diam)
        return self

    def float(self):
        super().float()
        self.mech.set_buffers(self.diam)
        return self

    def double(self):
        super().double()
        self.mech.set_buffers(self.diam)
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mech.set_buffers(self.diam)
        return self
    
    def warn_about_dt(self, dt):
        if DTWARN:
            if self._dt_lim is not None:
                if dt > self._dt_lim:
                    warnings.warn(f"dt ({dt}) exceeds limit ({self._dt_lim}), solution may have large oscillations.")


def match_state_dict(
    state_dict_a: Dict[str, torch.Tensor],
    state_dict_b: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Filters state_dict_b to contain only states that are present in state_dict_a.

    Matching happens according to two criteria:
        - Is the key present in state_dict_a?
        - Does the state with the same key in state_dict_a have the same shape?

    Returns
        (matched_state_dict, unmatched_state_dict)

        States in matched_state_dict contains states from state_dict_b that are also
        in state_dict_a and unmatched_state_dict contains states that have no
        corresponding state in state_dict_a.

            In addition: state_dict_b = matched_state_dict U unmatched_state_dict.
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
    PARAMETER(cm=1.0, rhoa=35.4)

    def __init__(
        self, diameters, L=1.0, dx=10.0, temp=37, v_init=-80, method="rk1", pade=None
    ):
        L = L * 1000  # mm -> um
        n_node = L / dx
        n_node = math.ceil(n_node) // 2 * 2 + 1
        self.dx: float = dx
        super().__init__(diameters, n_node, temp, v_init, method, pade)

    def x(self) -> torch.Tensor:  # x in um
        l = (self.n_node - 1) * self.dx
        return torch.linspace(-l / 2, l / 2, self.n_node, device=self.device())

    def area_(self, diameters) -> torch.Tensor:
        dx = torch.full_like(diameters, self.dx / 10000)
        return torch.pi * (diameters / 10000) * dx

    def ra_(self, diameters) -> torch.Tensor:
        dx = torch.full_like(diameters, self.dx / 10000)
        radii = diameters / 20000
        return (self.rhoa * dx) / (torch.pi * (radii**2))


class Myelinated(Axon):
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
        l = (self.n_node - 1) * self.deltax(self.diam) * 10000
        start = -l / 2
        end = l / 2
        steps = self.n_node
        t = torch.linspace(0, 1, steps, device=l.device).unsqueeze(-1)
        return ((1 - t) * start + t * end).T
