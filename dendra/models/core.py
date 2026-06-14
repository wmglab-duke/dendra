"""Core data structures and utilities for Dendra population models."""

import itertools
import math
import re
import textwrap
from contextlib import nullcontext
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import networkx as nx
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from dendra.helpers import (
    BACKEND,
    COMPILE_MODE,
    DYNAMIC,
    FULLGRAPH,
    IMEM,
    JIT,
    JIT_NETWORK_OPS,
    JIT_NETWORK_SOLVES,
    op_mc,
    op_sc,
)
from dendra.models.backend import Backend as A
from dendra.models.callbacks import Callback, CallbackList
from dendra.models.graph import get_area_from_graph
from dendra.models.integrators import bwd_euler_sc, bwd_euler_ub
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.mechanisms._ions import Ion, concentrations, equilibria, valid_ions
from dendra.models.mechanisms._material_process import MaterialProcess
from dendra.models.mechanisms._materials import (
    Material,
    MaterialFieldSpec,
    material_specs,
    valid_materials,
)
from dendra.models.mechanisms.validate import validate
from dendra.models.modular import matches_any_pattern
from dendra.models.parametric import Parameterized as P
from dendra.models.stim.intra import Intra
from dendra.models.stim.waveform import Waveform
from dendra.units import mm

from .slice import Sliceable

TensorLike = Union[torch.Tensor, "np.ndarray"]  # torch or numpy are supported

ExtraSpec = Union[
    Tuple[TensorLike, Union["Waveform", TensorLike]],
    Sequence[Tuple[TensorLike, Union["Waveform", TensorLike]]],
]


@dataclass
class _ExtraConfig:
    enabled: bool
    multicontact: bool
    functional: bool
    ve_s: Optional[torch.Tensor] = None  # [np, n_comp] or [n_contacts, np, n_comp]
    einsum: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None

    # If functional: list of Waveform (one per contact)
    waveforms: Optional[List["Waveform"]] = None

    # If non-functional, single contact: list of [np, n_t_chunk] tensors (len = n_chunks)
    time_chunks_single: Optional[List[torch.Tensor]] = None

    # If non-functional, multi-contact:
    # list over contacts, each is list over chunks -> [np, n_t_chunk]
    time_chunks_per_contact: Optional[List[List[torch.Tensor]]] = None


class NotInitializedError(AttributeError):
    """Accessed attribute before initialization."""


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
    """
    Check whether a dotted name pattern appears in a target string.

    Parameters
    ----------
    base_pattern : str
        Pattern consisting of dot-separated tokens that must appear in order.
    target_string : str
        Candidate string evaluated against the pattern.

    Returns
    -------
    bool
        True if the pattern tokens occur in order inside the target string,
        False otherwise.
    """
    regex_pattern = (
        r"\b"
        + r"\b.*?\b".join(re.escape(part) for part in base_pattern.split("."))
        + r"\b"
    )
    return re.search(regex_pattern, target_string) is not None


class SymmetricConv1D(torch.nn.Conv1d):
    """1D convolution layer with weights symmetrized during training."""

    def forward(self, x):
        """
        Apply the symmetric convolution operation.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(batch, channels, length)``.

        Returns
        -------
        Tensor
            Convolved tensor with the same shape as the input.
        """
        if self.training:
            weight_ = (self.weight + torch.flip(self.weight, [-1])) / 2
        else:
            weight_ = self.weight
        return self._conv_forward(x, weight_, self.bias)


def step(integrator, model, dt, ve=None, intra=None):
    """
    Execute a single integration step for a population model.

    Parameters
    ----------
    integrator : Integrator
        Integrator instance providing the ``step`` routine.
    model : Population
        Population model whose state is advanced.
    dt : float or Tensor
        Simulation time step in milliseconds.
    ve : Tensor, optional
        Extracellular potential applied during the step.
    intra : Any, optional
        Intra-cellular stimulation payload forwarded to the integrator.
    """
    integrator.step(model, dt, ve, intra)


def make_intra(intra, stims, indices):
    """
    Instantiate intra-cellular stimulation payload for a time step.

    Parameters
    ----------
    intra : Intra
        Intra-cellular stimulus model.
    stims : list of Tensor
        Sequence of per-channel stimulation tensors for the current step.
    indices : Any
        Index structure describing the electrodes addressed by ``stims``.

    Returns
    -------
    Any
        Instantiated stimulation payload compatible with the integrator.
    """
    return intra(stims, indices)


class Population(P, Sliceable):
    """
    Base class for a population of multicompartment neurons.
    """

    # ---------- repr knobs (safe defaults) ----------
    _REPR_MAX_MECHS: int = 18
    _REPR_MAX_IONS: int = 12
    _REPR_MAX_MATERIALS: int = 12
    _REPR_TENSOR_SAMPLES: int = 7  # sample points for big tensors
    _REPR_SHOW_KWARGS: bool = False  # kwargs can be huge; default off
    _REPR_TENSOR_STATS: str = "sample"  # "none" | "sample" | "full"

    _REPR_FULL_TENSOR_MAX_ELEMS: int = 16  # print full values if numel <= this
    _REPR_FLOAT_SIGFIGS: int = 6  # scalar + small tensor formatting
    _REPR_MAX_MECH_PARAM_ENTRIES: int = 200  # safety bound per mechanism

    P.RANGEP(cm=1.0, rhoa=35.4)
    P.GLOBAL(celsius=37.0)
    P.GLOBALP(rhoa_scale=1.0, cm_scale=1.0, area_scale=1.0)

    def __init__(self, N: int = 1, C: int = 1, integrator=None, v_init=-65.0, **kwargs):
        super().__init__((N, C), (N, C), **kwargs)
        Sliceable.__init__(self)
        self.np = N
        self.nc = C
        self.v_init = v_init

        self.is_built = False
        self._flag_rebuild = False
        self.key = None

        if integrator is None:
            integrator = bwd_euler_sc()

        self.register_buffer("_dummy", torch.zeros(1))

        # Initialize voltage from scalar v_init or from a vector of length nc.
        # The latter is useful for point-neuron populations represented as a
        # single Dendra population with one compartment per modeled neuron.
        self.register_buffer("v", self.expanded_v_init((N, C)).clone().contiguous())
        self.register_buffer("diam", torch.full(self.shape, 500.0))
        self.register_buffer("dx", torch.full(self.shape, 100.0))
        self.register_buffer("t", torch.zeros(()))

        # compiler stuff
        self.backend = BACKEND.value
        self.fullgraph = bool(FULLGRAPH)
        self.dynamic = bool(DYNAMIC)
        self.jit = bool(JIT)
        self.jit_network_solves = bool(JIT_NETWORK_SOLVES)
        self.jit_network_ops = bool(JIT_NETWORK_OPS)
        # Legacy attribute retained for external code. Internally this now means
        # "compile population solves when this population is stepped by Network".
        self.jit_in_network = self.jit_network_solves
        self.imem = bool(IMEM)
        self.compile_mode = COMPILE_MODE.value
        self.initializing_from_state_cache = False

        if self.imem:
            self.register_buffer("i_membrane", torch.zeros(self.shape))
        else:
            self.i_membrane = None  # type: ignore

        self._integrator_class = integrator
        self.integrator = None  # type: ignore

        self.injections = []
        # Parallel mechanism-level injection registry.  Standard injections are
        # still consumed by Intra/integrators; these specs are additionally
        # offered to mechanisms via Mechanism.inject(...).
        self.mechanism_injections = []
        self.mechanism_injection_accepted = []
        self.intra = None

        self._mech_data = {}
        self._mech_everywhere = {}

        self._m_list = []
        self._m_name = []
        self._m_keys = []
        self._m_curr = {}
        self._m_shape = {}

        self._ion_read = {}
        self._ion_write = {}
        self._ion_write_c = {}

        # Generic Material bookkeeping mirrors the ion maps above.  These maps
        # are populated from Mechanism.USEMATERIAL(...) declarations during
        # build() and are passed to MechanismHandler for sync/commit.
        self._material_read = {}
        self._material_write = {}
        self._material_source = {}
        self._material_process_read = {}
        self._material_process_write = {}
        self._material_process_source = {}
        self._material_configs = {}

        self._all_read = {}
        self._all_write = {}
        self._all_write_c = {}

        self._equilibria = {}
        self._concentrations = {}

        self._ion_style = {}

        self.pre_initialize_hooks: List[Callable] = []
        self.post_initialize_hooks: List[Callable] = []

        torch._dynamo.reset()

        # Keep the state-mutating Population/Integrator wrapper eager.  JIT
        # settings are propagated to the integrator, which compiles only its
        # tensor-valued numerical kernels.  This avoids Dynamo tracing
        # nn.Module.__setattr__ for model.v/model.vc/model.i_membrane commits.
        self._step = step

        self._make_intra_config = None
        self._refresh_compile_config_from_ctx()

        self._caches = {}

        self.register_buffer("x", torch.zeros(self.shape))
        self.register_buffer("y", torch.zeros(self.shape))
        self.register_buffer("z", torch.zeros(self.shape))

        self.mech: MechanismHandler = None  # type: ignore

        self.initialized: bool = False
        self.eval()

    def force_integrator_reinit(self):
        """
        Determine whether the integrator should be re-initialized.

        Returns
        -------
        bool
            True if the integrator should be re-initialized, False otherwise.
            By default, this returns True during training to ensure that any
            changes to model parameters are reflected in the integrator state.
            During evaluation, it returns False to allow the integrator to reuse
            its existing state for efficiency.
        """
        return self.training or self.initializing_from_state_cache

    def _refresh_compile_config_from_ctx(self):
        """Refresh compile flags from dendra.ctx / ContextVar state.

        Population-owned stepping stays eager; these flags are forwarded to the
        integrator with an explicit execution scope at run/initialize time.
        ``JIT_NETWORK_SOLVES`` is intentionally stored but ignored for standalone
        Population.run(), where only ``JIT`` enables integrator compilation.
        """
        self.backend = BACKEND.value
        self.fullgraph = bool(FULLGRAPH)
        self.dynamic = bool(DYNAMIC)
        self.jit = bool(JIT)
        self.jit_network_solves = bool(JIT_NETWORK_SOLVES)
        self.jit_network_ops = bool(JIT_NETWORK_OPS)
        self.jit_in_network = self.jit_network_solves
        self.compile_mode = COMPILE_MODE.value

        make_intra_config = (
            self.jit,
            self.backend,
            self.fullgraph,
            self.dynamic,
            self.compile_mode,
        )
        if getattr(self, "_make_intra_config", None) != make_intra_config:
            if self.jit:
                kwargs = dict(
                    backend=self.backend,
                    fullgraph=self.fullgraph,
                    dynamic=self.dynamic,
                )
                if self.compile_mode is not None:
                    kwargs["mode"] = self.compile_mode
                self.make_intra = torch.compile(make_intra, **kwargs)
            else:
                self.make_intra = make_intra
            self._make_intra_config = make_intra_config
        return self

    @property
    def shape(self):
        """
        Shape tuple of the membrane potential tensor.

        Returns
        -------
        tuple of int
            Dimensions of ``self.v`` including any batch axes.
        """
        return tuple(self.v.shape)

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
        """
        if self.graph is not None:
            area = get_area_from_graph(self.graph)
            if area is not None:
                return (
                    area.to(self.device(), dtype=self.dtype())
                    .reshape(-1, self.nc)
                    .expand(self.np, -1)
                )
        return self.diam * 1e-4 * torch.pi * self.dx * 1e-4  # in cm²

    def numel(self, include_batch_dimensions=True):
        """
        Count elements in the population state tensor.

        Parameters
        ----------
        include_batch_dimensions : bool, optional
            If True, include batch dimensions in the count. If False,
            only the core neuron/compartment axes are considered.

        Returns
        -------
        int
            Total number of elements in ``self.v`` according to the flag.
        """
        if not include_batch_dimensions:
            return math.prod(self.core_shape())
        return self.v.numel()

    def numelc(self):
        """
        Count elements per cell, excluding batch dimensions.

        Returns
        -------
        int
            Number of elements across the neuron and compartment axes.
        """
        return self.numel(include_batch_dimensions=False)

    def equilibria(self, **kwargs):
        """
        Register reversal potential configuration for ionic species.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to ``dendra.models.mechanisms._ions.equilibria``.
        """
        self._equilibria.update(kwargs)

    def concentrations(self, **kwargs):
        """
        Register ionic concentration configuration.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to ``dendra.models.mechanisms._ions.concentrations``.
        """
        self._concentrations.update(kwargs)

    def material(
        self,
        name: str,
        fields=None,
        *,
        initial_values: Optional[Dict[str, object]] = None,
        min_values: Optional[Dict[str, Optional[float]]] = None,
        specs: Optional[Dict[str, MaterialFieldSpec]] = None,
        domain=None,
        units=None,
        conserved=None,
        **field_initials,
    ):
        """Register or override a generic population-wide Material.

        Parameters
        ----------
        name : str
            Material name, e.g. ``"ip3"``.  For registered ions such as
            ``"ca"``, continue to use :meth:`concentrations` /
            :meth:`equilibria`; ions are already Material-like and may be read
            through ``USEION`` or ``USEMATERIAL``.
        fields : Mapping or Sequence, optional
            Field declarations passed to :class:`Material`.  A mapping specifies
            initial values, e.g. ``fields={"ip3i": 0.1}``; a sequence declares
            fields initialized to zero unless ``initial_values`` supplies values.
        initial_values : Mapping, optional
            Per-field initial value overrides.  When the material has been
            registered globally, these override the registered initial values
            while preserving other field metadata such as min values and units.
        min_values : Mapping, optional
            Per-field minimum-value guards.
        specs : Mapping[str, MaterialFieldSpec], optional
            Fully specified field specs.  This is the most explicit form and is
            forwarded directly to :class:`Material`.
        **field_initials
            Convenience initial values, e.g. ``model.material("ip3", ip3i=0.1)``.

        Returns
        -------
        Population
            The population instance for chaining.
        """
        name = str(name)
        if name in valid_ions():
            raise ValueError(
                f"{name!r} is a registered ion. Use concentrations(...) and "
                "equilibria(...) for ion defaults; mechanisms may still read ion "
                "fields through USEION or USEMATERIAL."
            )
        if self.is_built:
            self._flag_rebuild = True

        if field_initials:
            initial_values = dict(initial_values or {})
            for field, value in field_initials.items():
                initial_values[str(field)] = value

        cfg = self._material_configs.setdefault(
            name,
            {
                "fields": None,
                "initial_values": {},
                "min_values": {},
                "specs": None,
                "domain": None,
                "units": None,
                "conserved": None,
            },
        )

        if specs is not None:
            cfg["specs"] = dict(specs)
        if fields is not None:
            cfg["fields"] = fields
        if initial_values:
            cfg.setdefault("initial_values", {}).update(dict(initial_values))
        if min_values:
            cfg.setdefault("min_values", {}).update(dict(min_values))
        if domain is not None:
            cfg["domain"] = domain
        if units is not None:
            cfg["units"] = units
        if conserved is not None:
            cfg["conserved"] = conserved
        return self

    def material_(self, name: str, *args, **kwargs):
        """In-place alias of :meth:`material`."""
        self.material(name, *args, **kwargs)

    def _material_constructor_kwargs(self, name: str) -> Dict[str, object]:
        """Build Material constructor kwargs from population-local overrides."""
        cfg = self._material_configs.get(str(name), None)
        if not cfg:
            return {}

        if cfg.get("specs") is not None:
            return {"specs": cfg["specs"]}

        fields = cfg.get("fields", None)
        initial_values = dict(cfg.get("initial_values") or {})
        min_values = dict(cfg.get("min_values") or {})
        domain = cfg.get("domain", None)
        units = cfg.get("units", None)
        conserved = cfg.get("conserved", None)

        def pick(obj, key, default):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return default if obj is None else obj

        def field_initial_mapping(fields_obj):
            if fields_obj is None:
                return None
            if isinstance(fields_obj, dict):
                out = dict(fields_obj)
                out.update(initial_values)
                return out
            if isinstance(fields_obj, str):
                return {fields_obj: initial_values.get(fields_obj, 0.0)}
            return {
                str(field): initial_values.get(str(field), 0.0) for field in fields_obj
            }

        # If a registered material already has specs and the user only supplies
        # overrides, preserve registered metadata while changing requested values.
        registered = material_specs().get(str(name), None)
        if (
            fields is None
            and registered is not None
            and (
                initial_values
                or min_values
                or domain is not None
                or units is not None
                or conserved is not None
            )
        ):
            specs_out = {}
            all_fields = (
                set(registered.keys())
                | set(initial_values.keys())
                | set(min_values.keys())
            )
            for field in all_fields:
                if field in registered:
                    spec = registered[field]
                    specs_out[field] = MaterialFieldSpec(
                        name=spec.name,
                        initial=initial_values.get(field, spec.initial),
                        min_value=min_values.get(field, spec.min_value),
                        conserved=bool(pick(conserved, field, spec.conserved)),
                        domain=str(pick(domain, field, spec.domain)),
                        units=pick(units, field, spec.units),
                    )
                else:
                    specs_out[field] = MaterialFieldSpec(
                        name=field,
                        initial=initial_values.get(field, 0.0),
                        min_value=min_values.get(field, None),
                        conserved=bool(pick(conserved, field, True)),
                        domain=str(pick(domain, field, "i")),
                        units=pick(units, field, None),
                    )
            return {"specs": specs_out}

        # If the user supplied domain/units/conserved for an unregistered or
        # explicitly field-declared material, build full specs here so metadata is
        # preserved by the Material constructor.
        field_initials = field_initial_mapping(fields)
        if field_initials is not None and (
            domain is not None
            or units is not None
            or conserved is not None
            or min_values
        ):
            specs_out = {}
            for field, initial in field_initials.items():
                specs_out[field] = MaterialFieldSpec(
                    name=field,
                    initial=initial,
                    min_value=min_values.get(field, None),
                    conserved=bool(pick(conserved, field, True)),
                    domain=str(pick(domain, field, "i")),
                    units=pick(units, field, None),
                )
            return {"specs": specs_out}

        out: Dict[str, object] = {}
        if fields is not None:
            out["fields"] = fields
        if initial_values:
            out["initial_values"] = initial_values
        if min_values:
            out["min_values"] = min_values
        return out

    def unfreeze_group(self, *groups):
        """
        Unfreeze parameter groups stored as module attributes.

        Parameters
        ----------
        *groups : str
            Attribute names of iterable parameter collections to unfreeze.

        Returns
        -------
        Population
            The population instance for chaining.
        """
        for g in groups:
            group = getattr(self, g)
            print(f"Unfreezing group '{g}'")
            for p in group:
                p.requires_grad = True
        return self

    def unfreeze_group_(self, *groups):
        """
        In-place alias of :meth:`unfreeze_group`.

        Parameters
        ----------
        *groups : str
            Attribute names forwarded to :meth:`unfreeze_group`.
        """
        self.unfreeze_group(*groups)

    def freeze_group(self, *groups):
        """
        Freeze parameter groups stored as module attributes.

        Parameters
        ----------
        *groups : str
            Attribute names of iterable parameter collections to freeze.

        Returns
        -------
        Population
            The population instance for chaining.
        """
        for g in groups:
            group = getattr(self, g)
            print(f"Freezing group '{g}'")
            for p in group:
                p.requires_grad = False
        return self

    def freeze_group_(self, *groups):
        """
        In-place alias of :meth:`freeze_group`.

        Parameters
        ----------
        *groups : str
            Attribute names forwarded to :meth:`freeze_group`.
        """
        self.freeze_group(*groups)

    def retain_grad(self, *names):
        """
        Retain gradients for model buffers.

        Parameters
        ----------
        *names : str
            Variable length argument list of buffer names.
            If empty, all buffers will have their gradients retained.
            Otherwise, only buffers matching any of these names will have their gradients retained.
        """
        for n, b in self.named_buffers():
            if not names or matches_any_pattern(names, n):
                if b.requires_grad:
                    b.retain_grad()

    def register_post_initialize_hook(self, fn: Callable):
        """
        Register a hook executed after model initialization.

        Parameters
        ----------
        fn : Callable
            Callback invoked with the population instance once initialization
            completes.
        """
        self.post_initialize_hooks.append(fn)

    def register_pre_initialize_hook(self, fn: Callable):
        """
        Register a hook executed before model initialization.

        Parameters
        ----------
        fn : Callable
            Callback invoked with the population instance just prior to
            mechanism initialization.
        """
        self.pre_initialize_hooks.append(fn)

    def device(self):
        """
        Device on which population buffers reside.

        Returns
        -------
        torch.device
            Device handle inferred from the registered dummy buffer.
        """
        return self._dummy.device

    def dtype(self):
        """
        Default tensor dtype for the population.

        Returns
        -------
        torch.dtype
            Data type inferred from the registered dummy buffer.
        """
        return self._dummy.dtype

    def expanded_v_init(self, target_shape=None):
        """
        Return ``v_init`` as a tensor expanded to the model voltage shape.

        Supported ``v_init`` forms are:

        - scalar: broadcast to every element of ``model.v``;
        - one-dimensional tensor/array/list of length ``model.nc``: broadcast
          across the population axis and any batch axes;
        - tensor/array with shape ``(model.np, model.nc)`` or the current full
          voltage shape: used as explicit per-element initial voltages.

        Parameters
        ----------
        target_shape : tuple of int, optional
            Shape to expand into. Defaults to ``self.v.shape`` after ``v`` has
            been registered.

        Returns
        -------
        torch.Tensor
            A tensor view with shape ``target_shape`` on this model's device and
            dtype. The returned tensor may be expanded; callers that will mutate
            it should clone first.
        """
        if target_shape is None:
            if not hasattr(self, "v"):
                target_shape = (self.np, self.nc)
            else:
                target_shape = tuple(self.v.shape)
        else:
            target_shape = tuple(int(x) for x in target_shape)

        if len(target_shape) < 2:
            raise ValueError(
                f"target_shape must have at least population and compartment axes; "
                f"got {target_shape}."
            )

        n_pop = int(target_shape[-2])
        n_comp = int(target_shape[-1])
        v0 = torch.as_tensor(self.v_init, device=self.device(), dtype=self.dtype())

        # Scalar or scalar-like tensor/list: broadcast everywhere.
        if v0.ndim == 0 or v0.numel() == 1:
            return v0.reshape(()).expand(target_shape)

        # Length-nc vector: one initial value per compartment, broadcast across
        # cells/fibers and batch dimensions.
        if v0.ndim == 1:
            if v0.numel() != n_comp:
                raise ValueError(
                    f"v_init has length {v0.numel()}, but model.nc is {n_comp}. "
                    "Use a scalar or a vector of length model.nc."
                )
            view_shape = (1,) * (len(target_shape) - 1) + (n_comp,)
            return v0.reshape(view_shape).expand(target_shape)

        # Explicit initial voltage for the unbatched core shape.
        core_shape = (n_pop, n_comp)
        if tuple(v0.shape) == core_shape:
            view_shape = (1,) * (len(target_shape) - 2) + core_shape
            return v0.reshape(view_shape).expand(target_shape)

        # Explicit initial voltage for the full current shape.
        if tuple(v0.shape) == target_shape:
            return v0

        # Common explicit-broadcast form: [1, nc].
        if tuple(v0.shape) == (1, n_comp):
            view_shape = (1,) * (len(target_shape) - 2) + (1, n_comp)
            return v0.reshape(view_shape).expand(target_shape)

        raise ValueError(
            "Unsupported v_init shape. Expected a scalar, a 1D vector of length "
            f"model.nc ({n_comp}), shape (model.np, model.nc) = {core_shape}, "
            f"or full voltage shape {target_shape}; got shape {tuple(v0.shape)}."
        )

    def set_v_init(self, v_init):
        """
        Set and validate the model's voltage initial condition.

        ``v_init`` follows the same shape rules as :meth:`expanded_v_init`.
        This method does not immediately overwrite ``model.v``; call
        :meth:`init_v` or :meth:`initialize` to apply it.
        """
        old_v_init = self.v_init
        self.v_init = v_init
        try:
            self.expanded_v_init()
        except Exception:
            self.v_init = old_v_init
            raise
        return self

    def prep_intra(self, intra, n, dt):
        """
        Prepare intra-cellular stimulus batches for simulation.

        Parameters
        ----------
        intra : Intra
            Intra-cellular stimulation provider.
        n : int
            Number of time steps to generate stimuli for.
        dt : float
            Simulation time step in milliseconds.

        Returns
        -------
        tuple
            Pair ``(stims, indices)`` where ``stims`` is a list of sequences
            of stimuli and ``indices`` encodes electrode mapping metadata.
        """
        start = self.t
        end = start + n * dt
        t_ = torch.arange(start, end, dt, device=self.device(), dtype=self.dtype())
        t_ = t_.to(self.device(), dtype=self.dtype())
        stims, indices = intra.init(t_)
        stims = [s.unbind(-1) for s in stims]
        return stims, indices

    # extracellular helpers
    def _normalize_spatial(self, ve_s_raw: TensorLike) -> torch.Tensor:
        """
        Normalize a spatial field tensor to shape [np, n_comp].

        Accepts:
        - [n_comp]
        - [1, n_comp]
        - [np, n_comp]

        Broadcasting from leading dimension 1 to np where needed.
        """
        ve_s = torch.as_tensor(
            ve_s_raw,
            device=self.device(),
            dtype=self.dtype(),
        ).contiguous()

        if ve_s.dim() == 1:
            # [n_comp] -> [1, n_comp]
            ve_s = ve_s.unsqueeze(0)

        if ve_s.size(0) == 1:
            # [1, n_comp] -> [np, n_comp]
            ve_s = ve_s.expand(self.np, -1)
        elif ve_s.size(0) != self.np:
            raise ValueError(
                f"ve_s leading dimension ({ve_s.size(0)}) must be 1 or np ({self.np})."
            )

        return ve_s

    def _normalize_time_tensor(
        self, time_raw: TensorLike, t_global: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize a time tensor to shape [np, n_t] matching the global time grid.

        Accepts:
        - [n_t]
        - [1, n_t]
        - [np, n_t]

        Broadcasting from leading dimension 1 to np where needed.
        """
        t_tensor = torch.as_tensor(
            time_raw,
            device=self.device(),
            dtype=self.dtype(),
        )

        if t_tensor.dim() == 1:
            # [n_t] -> [1, n_t]
            t_tensor = t_tensor.unsqueeze(0)

        if t_tensor.size(0) == 1:
            # [1, n_t] -> [np, n_t]
            t_tensor = t_tensor.expand(self.np, -1)
        elif t_tensor.size(0) != self.np:
            raise ValueError(
                f"time tensor leading dimension ({t_tensor.size(0)}) "
                f"must be 1 or np ({self.np})."
            )

        if t_tensor.size(-1) != t_global.size(0):
            raise ValueError(
                f"time tensor length ({t_tensor.size(-1)}) must match "
                f"the number of simulation steps ({t_global.size(0)})."
            )

        return t_tensor

    def _prepare_extra(
        self,
        extra: Optional[ExtraSpec],
        t_global: torch.Tensor,
        n_chunks: int,
    ) -> _ExtraConfig:
        """
        Normalize and stage 'extra' into an _ExtraConfig.

        Supports:
        - extra = (ve_s, time)
        - extra = [(ve_s1, time1), (ve_s2, time2), ...]
        """
        if extra is None:
            return _ExtraConfig(enabled=False, multicontact=False, functional=False)

        # Normalize to list[(ve_s, time_spec)]
        if isinstance(extra, tuple):
            extra_pairs = [extra]
        else:
            extra_pairs = list(extra)

        if not extra_pairs:
            raise ValueError("If 'extra' is provided, it must not be empty.")

        n_contacts = len(extra_pairs)
        multicontact = n_contacts > 1

        ve_s_list: List[torch.Tensor] = []
        functional_flags: List[bool] = []
        waveforms: List["Waveform"] = []
        time_tensors: List[torch.Tensor] = []

        for ve_s_raw, time_spec_raw in extra_pairs:
            # Spatial field -> [np, n_comp]
            ve_s_i = self._normalize_spatial(ve_s_raw)
            ve_s_list.append(ve_s_i)

            # Temporal spec: Waveform vs tensor
            if isinstance(time_spec_raw, Waveform):
                functional_flags.append(True)
                waveforms.append(
                    time_spec_raw.to(device=self.device(), dtype=self.dtype())
                )
                # Placeholder for alignment (not used when functional)
                time_tensors.append(
                    torch.empty(0, device=self.device(), dtype=self.dtype())
                )
            else:
                functional_flags.append(False)
                t_tensor = self._normalize_time_tensor(time_spec_raw, t_global)
                time_tensors.append(t_tensor)
                # Placeholder for alignment (not used when non-functional)
                waveforms.append(None)  # type: ignore[arg-type]

        any_functional = any(functional_flags)
        all_functional = all(functional_flags)
        if any_functional and not all_functional:
            raise ValueError(
                "All contacts in 'extra' must use the same temporal type; "
                "mixing Waveform and tensor time specifications is not supported."
            )
        functional = any_functional

        # Stack spatial fields
        if multicontact:
            # [n_contacts, np, n_comp]
            ve_s = torch.stack(ve_s_list, dim=0)
            einsum = op_mc
        else:
            # [np, n_comp]
            ve_s = ve_s_list[0]
            einsum = op_sc

        time_chunks_single: Optional[List[torch.Tensor]] = None
        time_chunks_per_contact: Optional[List[List[torch.Tensor]]] = None

        if not functional:
            if multicontact:
                # Per-contact list of chunks
                time_chunks_per_contact = [
                    torch.tensor_split(t_tensor, n_chunks, dim=-1)
                    for t_tensor in time_tensors
                ]
            else:
                time_chunks_single = torch.tensor_split(
                    time_tensors[0], n_chunks, dim=-1
                )

        return _ExtraConfig(
            enabled=True,
            multicontact=multicontact,
            functional=functional,
            ve_s=ve_s,
            einsum=einsum,
            waveforms=waveforms if functional else None,
            time_chunks_single=time_chunks_single,
            time_chunks_per_contact=time_chunks_per_contact,
        )

    def _expand_eval_time(self, t_eval: torch.Tensor) -> torch.Tensor:
        """
        Normalize evaluated Waveform output to [np, n_t_chunk].

        Accepts:
        - [n_t_chunk]
        - [1, n_t_chunk]
        - [np, n_t_chunk]
        """
        if t_eval.dim() == 1:
            t_eval = t_eval.unsqueeze(0)

        if t_eval.size(0) == 1:
            t_eval = t_eval.expand(self.np, -1)
        elif t_eval.size(0) != self.np:
            raise ValueError(
                f"Waveform evaluation leading dimension ({t_eval.size(0)}) "
                f"must be 1 or np ({self.np})."
            )
        return t_eval

    def _compute_extra_chunk(
        self,
        cfg: _ExtraConfig,
        chunk_idx: int,
        t_chunk: torch.Tensor,
    ) -> Optional[List[torch.Tensor]]:
        """
        Build ve_ list (one entry per time step in chunk) for the given chunk,
        or return None if no 'extra' is configured.

        Output:
        - None
        - or list of length len(t_chunk), each [np, n_comp]
        """
        if not cfg.enabled:
            return None

        assert cfg.ve_s is not None
        assert cfg.einsum is not None

        # Build temporal tensor for this chunk: t_extra
        if cfg.functional:
            assert cfg.waveforms is not None

            if cfg.multicontact:
                # [n_contacts, np, n_t_chunk]
                t_per_contact: List[torch.Tensor] = []
                for wf in cfg.waveforms:
                    t_i = wf(t_chunk).to(self.dtype())
                    t_i = self._expand_eval_time(t_i)
                    t_per_contact.append(t_i.unsqueeze(0))  # [1, np, n_t_chunk]

                t_extra = torch.cat(t_per_contact, dim=0)
            else:
                wf = cfg.waveforms[0]
                t_extra = wf(t_chunk).to(self.dtype())
                t_extra = self._expand_eval_time(t_extra)
        else:
            # Non-functional: use pre-split chunks
            if cfg.multicontact:
                assert cfg.time_chunks_per_contact is not None
                t_per_contact = [
                    cfg.time_chunks_per_contact[c][chunk_idx].unsqueeze(0)
                    for c in range(len(cfg.time_chunks_per_contact))
                ]
                t_extra = torch.cat(t_per_contact, dim=0)  # [n_contacts, np, n_t_chunk]
            else:
                assert cfg.time_chunks_single is not None
                t_extra = cfg.time_chunks_single[chunk_idx]  # [np, n_t_chunk]

        # einsum:
        #  - single-contact: ve_s [np, n_comp], t_extra [np, n_t_chunk]
        #  - multi-contact : ve_s [n_contacts, np, n_comp],
        #                    t_extra [n_contacts, np, n_t_chunk]
        ve = cfg.einsum(cfg.ve_s, t_extra)  # [n_t_chunk, np, n_comp]
        return ve.unbind(dim=0)

    def run(
        self,
        ve: Optional[TensorLike] = None,
        extra: Optional[ExtraSpec] = None,
        tstop: Optional[float] = None,
        dt: Optional[float] = None,
        callbacks: Optional[Sequence[Callback]] = None,
        progressbar: bool = False,
    ):
        """
        Run the axon model simulation.

        Parameters
        ----------
        ve : Tensor, optional
            Precomputed extracellular voltage tensor. Shape should be
            ``[n_timesteps, np, n_comp]`` or broadcast-compatible with that.
            If provided, ``extra`` is ignored and the number of time steps
            is inferred from ``ve.shape[0]``.
        extra : (Tensor, Waveform or Tensor) or sequence of such tuples, optional
            Extracellular input specification(s), with the same semantics as
            :meth:`longrun`.

            Each specification is a tuple ``(ve_s, time)``:

            * ``ve_s``: spatial field tensor with shape ``[np, n_comp]`` or
              ``[1, n_comp]``. A leading dimension of ``1`` is broadcast to ``np``.

            * ``time``: either a :class:`Waveform` object (functional specification)
              or a tensor with shape ``[np, n_timesteps]`` or ``[1, n_timesteps]``.
              A leading dimension of ``1`` is broadcast to ``np``. The last
              dimension must match the number of simulation time steps.

            If a single tuple is provided, the method uses a single-contact
            formulation with :func:`op_sc`. If a sequence of tuples is provided,
            each tuple is treated as one electrode contact, and the method
            automatically switches to multi-contact mode using :func:`op_mc`:

            * Spatial fields are stacked to shape ``[n_contacts, np, n_comp]``.
            * Functional (Waveform) inputs are evaluated per time step and
              expanded/concatenated to shape ``[n_contacts, np, n_timesteps]``.
            * Non-functional (tensor) inputs are normalized once to that shape.

            Mixing :class:`Waveform` and tensor time specifications across contacts
            is not supported and will raise a :class:`ValueError`.

            If both ``ve`` and ``extra`` are provided, a :class:`ValueError` is
            raised.
        tstop : float, optional
            Simulation stop time in milliseconds when ``ve`` is not provided.
            The number of simulation time steps is derived from the time grid
            constructed from the current model time ``self.t``, ``tstop``, and
            ``dt``. If ``ve`` is provided, this parameter is ignored.
        dt : float, optional
            Time step size in milliseconds. If ``None``, the default value
            ``A.dt`` from the backend is used.
        callbacks : sequence of Callback, optional
            Callback objects to execute during simulation (pre-loop, per-step,
            post-loop, etc.). If not already a :class:`CallbackList`, it is
            wrapped into one.
        progressbar : bool or tqdm, optional
            If ``True``, displays a progress bar during simulation. Can also be
            a :class:`tqdm.tqdm` instance for custom progress tracking.

        Notes
        -----
        * ``run`` supports both precomputed ``ve`` and the higher-level
          ``extra`` specification used by :meth:`longrun`.
        * When ``extra`` is provided, the underlying construction of
          extracellular voltage matches the semantics of :meth:`longrun`,
          but the entire simulation is treated as a single chunk.
        * The simulation updates the model's internal state (e.g. ``v``,
          ``v_prev`` for DF methods) and advances the model's time ``self.t``.
        """
        if not self.initialized:
            raise ValueError("Model must be initialized before running.")
        self._refresh_compile_config_from_ctx()

        # auto-build intra if missing
        if self.intra is None:
            intra = self.build_intra()
            self.intra = intra
        else:
            intra = self.intra

        with_intra = intra is not None

        if ve is not None and extra is not None:
            raise ValueError("Provide either 've' or 'extra', not both.")

        device = self.device()
        dtype = self.dtype()

        if ve is not None:
            ve = torch.as_tensor(ve, device=device, dtype=dtype).contiguous()

        # dt as scalar and tensor
        dt = dt if dt is not None else A.dt
        dt_f = float(dt)
        dt_tensor = torch.tensor(dt, device=device, dtype=dtype)

        local_ind = 0
        tstart = self.t.item()

        ctx = nullcontext() if self.training else torch.no_grad()

        psh = post_step_hook

        with ctx:
            # --------------------------------------------------------------
            # Determine number of steps and global time grid
            # --------------------------------------------------------------
            if ve is not None:
                # Use ve length as authoritative time axis length
                n = ve.shape[0]

                # Construct a matching time grid for 'extra'-style helpers if needed
                t_global = torch.arange(
                    self.t.double(),
                    self.t.double() + n * dt_f,
                    dt_f,
                    device=device,
                    dtype=torch.double,
                ).to(dtype)
            else:
                if tstop is None:
                    raise ValueError("tstop must be provided when 've' is not given.")

                t_global = torch.arange(
                    self.t.double(),
                    self.t.double() + tstop,
                    dt_f,
                    device=device,
                    dtype=torch.double,
                ).to(dtype)
                n = t_global.size(0)

            # --------------------------------------------------------------
            # Prepare intra
            # --------------------------------------------------------------
            if with_intra:
                # Preserve existing intra preparation semantics
                stims, indices = self.prep_intra(intra, n, dt_f)

            # --------------------------------------------------------------
            # Callbacks and integrator
            # --------------------------------------------------------------
            if not isinstance(callbacks, CallbackList):
                callbacks = CallbackList(callbacks)

            if callbacks:
                for c in callbacks:
                    c.dt = dt_f

            pre_loop_hook(callbacks, self)
            self.integrator._initialize(
                self,
                dt_tensor,
                force=self.force_integrator_reinit(),
                compile_scope="population",
            )

            # Progress bar setup
            if progressbar:
                if not isinstance(progressbar, tqdm):
                    progressbar = tqdm(
                        total=n,
                        desc=f"{tstart + local_ind * dt_f:.1f} ms",
                    )

            # --------------------------------------------------------------
            # Prepare extracellular input via 'extra' if needed
            # --------------------------------------------------------------
            if ve is None and extra is not None:
                # One "chunk" for the whole run
                extra_cfg = self._prepare_extra(extra, t_global, n_chunks=1)

                if extra_cfg.einsum is not None:
                    extra_cfg.einsum = extra_cfg.einsum

                # Single chunk index 0, over full time grid
                ve_list = self._compute_extra_chunk(extra_cfg, 0, t_global)
            else:
                ve_list = None

            # --------------------------------------------------------------
            # Main time-stepping loop
            # --------------------------------------------------------------
            for i in range(n):
                # Extracellular voltage for this step
                if ve is not None:
                    ve_c = ve[i]
                elif ve_list is not None:
                    ve_c = ve_list[i]
                else:
                    ve_c = None

                # Intracellular stimulation for this step
                if with_intra:
                    s = [st[local_ind] for st in stims]
                    intra_c = self.make_intra(intra, s, indices)
                else:
                    intra_c = None

                # Integrator step
                self._step(self.integrator, self, dt_tensor, ve_c, intra_c)
                self.t = self.t + dt_tensor

                psh(callbacks, self)
                local_ind += 1

                # Progress bar update
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
        extra: Optional[ExtraSpec] = None,
        callbacks: Optional[Sequence[Callback]] = None,
        progressbar=False,
    ):
        r"""
        Run a long simulation by dividing it into multiple smaller chunks.

        This method splits the overall simulation into chunks of a given length,
        allowing for more efficient memory management during long simulations.
        The model's state (e.g., voltage variables, ``v_prev``, etc.) is maintained
        between chunks, ensuring continuity across the entire simulation period.

        Parameters
        ----------
        tstop : float
            The simulation end time in milliseconds.
        chunklength : int
            The number of time steps to process in each chunk.
        dt : float, optional
            The simulation time step in milliseconds. If ``None``, the default value
            from the backend will be used.
        extra : (Tensor, Waveform or Tensor) or sequence of such tuples, optional
            Extracellular input specification(s).

            Each specification is a tuple ``(ve_s, time)``:

            * ``ve_s``: spatial field tensor with shape ``[n_p, n_comp]`` or
              ``[1, n_comp]``. A leading dimension of ``1`` is broadcast to ``n_p``.
            * ``time``: either a :class:`Waveform` object (functional specification)
              or a tensor with shape ``[n_p, n_timesteps]`` or ``[1, n_timesteps]``.
              A leading dimension of ``1`` is broadcast to ``n_p``. The last
              dimension must match the number of simulation time steps.

            If a single tuple is provided, the method uses a single-contact
            formulation with :func:`op_sc`. If a sequence of tuples is provided,
            each tuple is treated as one electrode contact, and the method
            automatically switches to multi-contact mode with :func:`op_mc`:

            * In multi-contact mode, the spatial field tensors are stacked to shape
              ``[n_contacts, n_p, n_comp]``.
            * For a functional specification (all ``time`` are :class:`Waveform`),
              each waveform is evaluated per chunk and per contact and then
              expanded/concatenated to shape ``[n_contacts, n_p, n_t_chunk]``.
            * For a non-functional specification (all ``time`` are tensors), the
              raw time tensors are pre-split into chunks and concatenated to the
              same shape ``[n_contacts, n_p, n_t_chunk]``.

            Mixing :class:`Waveform` and tensor time specifications across
            contacts is not supported and will raise a :class:`ValueError`.
        callbacks : list of Callback, optional
            A list of callback objects to be executed during simulation, allowing for
            customized processing at various stages (e.g., pre-loop, post-step,
            post-loop).
        progressbar : bool or tqdm, optional
            If ``True`` (or if a :class:`tqdm.tqdm` instance is provided), displays
            a progress bar to track simulation progress across chunks. Default is ``False``.

        Notes
        -----
        * Multi-contact handling is inferred from the number of ``extra`` entries;
          the ``multicontact`` flag is no longer required.
        * When ``extra`` is provided, the method uses it to assemble the
          extracellular voltage ``ve`` via :func:`op_sc` (single contact) or
          :func:`op_mc` (multi-contact).
        * Chunk processing helps manage memory usage during extended simulations
          by processing data in manageable segments.
        """

        if not self.initialized:
            raise ValueError("Model must be initialized before running.")
        self._refresh_compile_config_from_ctx()

        intra = self.intra
        with_intra = intra is not None

        # dt scalars
        dt = dt if dt is not None else A.dt
        dt_f = float(dt)
        dt_tensor = torch.tensor(dt, device=self.device(), dtype=self.dtype())

        psh = post_step_hook

        with torch.nn.utils.parametrize.cached():
            with torch.set_grad_enabled(self.training):
                # --------------------------------------------------------------
                # Global time grid and chunking
                # --------------------------------------------------------------
                t = torch.arange(
                    self.t.double(),
                    self.t.double() + tstop,
                    dt_f,
                    dtype=torch.double,
                    device=self.device(),
                ).to(self.dtype())

                if t.numel() == 0:
                    return

                n_chunks = math.ceil(len(t) / chunklength)
                t_chunks = torch.tensor_split(t, n_chunks)

                # --------------------------------------------------------------
                # Extracellular configuration (single vs multi-contact, functional)
                # --------------------------------------------------------------
                extra_cfg = self._prepare_extra(extra, t, n_chunks)
                with_extra = extra_cfg.enabled

                # Compile einsum if we have one
                if extra_cfg.einsum is not None:
                    extra_cfg.einsum = extra_cfg.einsum

                # --------------------------------------------------------------
                # Callbacks, progress bar, integrator initialization
                # --------------------------------------------------------------
                if callbacks:
                    for c in callbacks:
                        c.dt = dt_f

                callbacks = CallbackList(callbacks)

                if progressbar:
                    progressbar = tqdm(total=n_chunks, desc=f"{self.t.item():.1f} ms")

                self.integrator._initialize(
                    self,
                    dt_tensor,
                    force=self.force_integrator_reinit(),
                    compile_scope="population",
                )

                pre_loop_hook(callbacks, self)

                # --------------------------------------------------------------
                # Main chunk loop
                # --------------------------------------------------------------
                for i, t_chunk in enumerate(t_chunks):
                    if with_intra:
                        stims, indices = intra.init(t_chunk)
                        stims = [s.unbind(0) for s in stims]

                    if with_extra:
                        ve_list = self._compute_extra_chunk(extra_cfg, i, t_chunk)
                    else:
                        ve_list = None

                    pre_chunk_hook(callbacks, self, t_chunk)

                    for j in range(len(t_chunk)):
                        ve_c = ve_list[j] if ve_list is not None else None

                        if with_intra:
                            s = [st[j] for st in stims]
                            intra_c = self.make_intra(intra, s, indices)
                        else:
                            intra_c = None

                        self._step(self.integrator, self, dt_tensor, ve_c, intra_c)
                        psh(callbacks, self)
                        self.t = self.t + dt_tensor

                    post_chunk_hook(callbacks, self, t_chunk)

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

        self._refresh_compile_config_from_ctx()
        self.clear_steady_state()

        self.initialize()
        self.integrator._initialize(self, dt, force=True, compile_scope="population")

        maxiter = int(tstop / dt)

        with torch.no_grad():
            if with_ve:
                ve = torch.zeros_like(self.v).reshape(self.batched_shape()).contiguous()
            else:
                ve = None

            if with_intra:
                intra = (
                    torch.zeros_like(self.v).reshape(self.batched_shape()).contiguous()
                )
            else:
                intra = None

            dt = torch.tensor(dt, device=self.device(), dtype=self.dtype())
            for _ in tqdm(range(maxiter), desc="Steady state "):
                self._step(self.integrator, self, dt, ve, intra)

        self.cache("_steady_state")
        self.t = torch.zeros_like(self.t).detach()
        return self

    def clear_steady_state(self):
        """
        Remove cached steady-state snapshot, if present.

        Returns
        -------
        None
        """
        if "_steady_state" in self._caches:
            self._caches.pop("_steady_state")

    def post_initialize(self):
        """
        Run registered post-initialization hooks with gradients disabled.
        """
        with torch.no_grad():
            for h in self.post_initialize_hooks:
                h(self)

    def pre_initialize(self):
        """
        Run registered pre-initialization hooks with gradients disabled.
        """
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
        """
        "In-place" alias of :meth:`populate`. Same as :meth:`populate`, but
        does not return self.
        """
        self.populate()

    def _restore_steady_state(self):
        if "_steady_state" in self._caches:
            self.restore("_steady_state")
            self.post_initialize()
            self.t = torch.zeros_like(self.t).detach()
            self.initialized = True
            self.initializing_from_state_cache = True
            return True
        return False

    def initialize(self, force_rebuild=False, populate_parameter_buffers=True):
        """
        Build, populate, and initialize mechanisms for simulation.

        Parameters
        ----------
        force_rebuild : bool, optional
            If True, force rebuilding of mechanisms even if a built graph
            already exists.

        Returns
        -------
        Population
            The initialized population instance.
        """
        self.build(force_rebuild)
        if populate_parameter_buffers:
            self.populate_parameter_buffers()
        self.intra = self.build_intra()
        if self._restore_steady_state():
            return self
        self.integrator.init_v(self)
        self.pre_initialize()
        self.integrator.mech.initialize(
            self.v, self.celsius, self.diam, populate=populate_parameter_buffers
        )
        self.post_initialize()
        self.integrator.mech.initialize(
            self.v, self.celsius, self.diam, populate=populate_parameter_buffers
        )
        self.t = torch.zeros_like(self.t).detach()
        self.initialized = True
        if force_rebuild:
            self.integrator.initialized = False
        return self

    def initialize_(self):
        """
        "In-place" alias of :meth:`initialize`. Same as :meth:`initialize`, but
        does not return self.
        """
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
        """
        In-place alias of :meth:`load`.

        Parameters
        ----------
        state_dict : Union[str, Mapping]
            Argument forwarded to :meth:`load`.
        """
        self.load(state_dict)

    def state_names(self) -> List[str]:
        """
        Enumerate state tensor names managed by the population.

        Returns
        -------
        list of str
            Ordered state names starting with ``'v'``.
        """
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
        """
        In-place alias of :meth:`cache`.

        Parameters
        ----------
        name : str, optional
            Cache key forwarded to :meth:`cache`.
        """
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
        """
        In-place alias of :meth:`restore`.

        Parameters
        ----------
        name : str, optional
            Cache key forwarded to :meth:`restore`.
        """
        self.restore(name)

    def register_injection(self, waveform, index_spec):
        """Register a waveform injection on both solver and mechanism paths.

        The solver path preserves the existing ``Intra`` behavior used by
        standard voltage integrators.  The mechanism path offers the same
        waveform/index information to every built mechanism through
        ``Mechanism.inject(...)``; mechanisms that do not override that method
        simply ignore it.
        """
        self.injections.append((waveform, index_spec.shape, index_spec.index))
        self.mechanism_injections.append((waveform, index_spec.shape, index_spec.index))
        self.mechanism_injection_accepted.append(False)
        # Force lazy reconstruction of the solver-level Intra object on the next
        # run/initialize after a new injection is added.
        self.intra = None

        # If mechanisms have already been built, deliver this injection
        # immediately.  If not, build() will dispatch all stored specs later.
        if getattr(self, "is_built", False) and getattr(self, "mech", None) is not None:
            self._dispatch_mechanism_injections(
                start=len(self.mechanism_injections) - 1
            )
        return self

    def _dispatch_mechanism_injections(self, mech_handler=None, *, start=0):
        """Offer stored waveform injections to built mechanisms."""
        mech_handler = self.mech if mech_handler is None else mech_handler
        if mech_handler is None:
            return
        if not self.mechanism_injections:
            return

        model_shape = tuple(self.shape)
        for inj_i, (waveform, shape, index) in enumerate(
            self.mechanism_injections[start:], start
        ):
            accepted = bool(self.mechanism_injection_accepted[inj_i])
            for mech in mech_handler.mechanisms.values():
                accepted = (
                    bool(
                        mech.inject(
                            waveform,
                            index=index,
                            shape=shape,
                            model_shape=model_shape,
                            model=self,
                        )
                    )
                    or accepted
                )
            self.mechanism_injection_accepted[inj_i] = accepted

    def delete_injections(self):
        """
        Remove all registered intra-cellular and mechanism-level injections.

        Returns
        -------
        None
        """
        self.injections = []
        self.mechanism_injections = []
        self.mechanism_injection_accepted = []
        self.intra = None
        if getattr(self, "mech", None) is not None:
            for mech in self.mech.mechanisms.values():
                clear = getattr(mech, "clear_injections", None)
                if clear is not None:
                    clear()

    def build_intra(self):
        """
        Build the intra-cellular stimulation handler.

        Returns
        -------
        Intra or None
            Intra stimulus object when injections are configured, otherwise None.
        """
        if self.injections:
            # If a mechanism accepted a waveform injection, it is responsible for
            # evaluating/padding that stimulus and exposing it internally.  Do
            # not also route the same waveform through the solver-level Intra
            # path, which would duplicate the current for standard solvers and
            # add unnecessary overhead for scnv/fused mechanisms.
            accepted = list(self.mechanism_injection_accepted)
            if len(accepted) < len(self.injections):
                accepted.extend([False] * (len(self.injections) - len(accepted)))
            solver_injections = [
                inj for inj, ok in zip(self.injections, accepted) if not ok
            ]
            if solver_injections:
                return Intra(self, solver_injections)
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
            self._flag_rebuild = True

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

    def ion_style(self, ion, einit, eadvance):
        """
        Register explicit ion handling style parameters.

        Parameters
        ----------
        ion : str
            Ion species identifier (e.g., ``'na'``).
        einit : int
            Initialization flag for equilibration.
        eadvance : int
            Advance-time flag for equilibration updates.
        """
        assert ion in valid_ions(), f"Invalid ion: {ion}"
        self._ion_style[ion] = (einit, eadvance)

    def get_ion_style(self, ion):
        """
        Retrieve the ion style tuple for a species.

        Parameters
        ----------
        ion : str
            Ion species identifier.

        Returns
        -------
        tuple
            Ion style tuple ``(c_style, e_style, einit, eadvance, cinit)``.
        """
        if ion in self._ion_style:
            return self._ion_style[ion]
        return self._calc_ion_style(ion)

    def _c_is_written(self, ion):
        """
        Determine whether concentration values are written for an ion.

        Parameters
        ----------
        ion : str
            Ion species identifier.

        Returns
        -------
        bool
            True if any mechanism writes concentrations for the ion.
        """
        d = self._ion_write_c.get(ion, {})
        if d:
            return True
        # Ion-like materials can also write concentration fields through
        # USEMATERIAL(ion, write=[...]) or source=[...].  Include them in style
        # inference so USEMATERIAL("ca", read=["eca"], write=["cai"])
        # advances eca just like USEION would.
        material_w = self._material_write.get(ion, {})
        material_s = self._material_source.get(ion, {})
        material_pw = self._material_process_write.get(ion, {})
        material_ps = self._material_process_source.get(ion, {})
        return bool(material_w or material_s or material_pw or material_ps)

    def _c_is_read(self, ion):
        """
        Determine whether concentration values are read for an ion.

        Parameters
        ----------
        ion : str
            Ion species identifier.

        Returns
        -------
        bool
            True if any mechanism reads intra- or extracellular concentration.
        """
        d = self._ion_read.get(ion, {})
        material_d = self._material_read.get(ion, {})
        material_pd = self._material_process_read.get(ion, {})
        if not d and not material_d and not material_pd:
            return False
        check = list(itertools.chain(*d.values())) if d else []
        check += list(itertools.chain(*material_d.values())) if material_d else []
        check += list(itertools.chain(*material_pd.values())) if material_pd else []
        return f"{ion}i" in check or f"{ion}o" in check

    def _e_is_read(self, ion):
        """
        Determine whether reversal potentials are read for an ion.

        Parameters
        ----------
        ion : str
            Ion species identifier.

        Returns
        -------
        bool
            True if any mechanism reads the ion's equilibrium potential.
        """
        d = self._ion_read.get(ion, {})
        material_d = self._material_read.get(ion, {})
        material_pd = self._material_process_read.get(ion, {})
        if not d and not material_d and not material_pd:
            return False
        check = list(itertools.chain(*d.values())) if d else []
        check += list(itertools.chain(*material_d.values())) if material_d else []
        check += list(itertools.chain(*material_pd.values())) if material_pd else []
        return f"e{ion}" in check

    def _calc_ion_style(self, ion):
        """
        Infer ion style flags based on current read/write registrations.

        Parameters
        ----------
        ion : str
            Ion species identifier.

        Returns
        -------
        tuple
            Tuple of style flags ``(einit, eadvance)``.
        """
        _c_is_written = self._c_is_written(ion)
        _c_is_read = self._c_is_read(ion)
        _e_is_read = self._e_is_read(ion)

        if _c_is_written:
            if _e_is_read:
                return (1, 1)
            return (0, 0)
        if _c_is_read:
            if _e_is_read:
                return (1, 0)
            return (0, 0)
        if _e_is_read:
            return (0, 0)
        return (0, 0)

    def _register_mech(self, m, shape, key):
        """
        Register a compiled mechanism with the population.

        Parameters
        ----------
        m : Mechanism
            Mechanism instance produced by ``compile_mechanism``.
        shape : tuple of int
            Shape tuple describing the mechanism's parameter layout.
        key : Any
            Indexing metadata describing where the mechanism applies.
        """
        name = m.name
        mech = m.__class__
        self._m_name.append(name)
        self._m_list.append(m)
        self._m_keys.append(key)
        self._m_shape[name] = shape

        is_material_process = isinstance(m, MaterialProcess)

        if not is_material_process:
            for k, v in mech._currents.items():
                self._m_curr.setdefault(k, {}).update({name: v})

            for k, v in mech._read_ion.items():
                self._ion_read.setdefault(k, {}).update({name: v})

            for k, v in mech._write_ion.items():
                self._ion_write.setdefault(k, {}).update({name: v})

            for k, v in mech._write_ion_c.items():
                self._ion_write_c.setdefault(k, {}).update({name: v})

            for k, v in getattr(mech, "_read_material", {}).items():
                self._material_read.setdefault(k, {}).update({name: v})

            for k, v in getattr(mech, "_write_material", {}).items():
                self._material_write.setdefault(k, {}).update({name: v})

            for k, v in getattr(mech, "_source_material", {}).items():
                self._material_source.setdefault(k, {}).update({name: v})
        else:
            for k, v in getattr(mech, "_read_material", {}).items():
                self._material_process_read.setdefault(k, {}).update({name: v})
            for k, v in getattr(mech, "_write_material", {}).items():
                self._material_process_write.setdefault(k, {}).update({name: v})
            for k, v in getattr(mech, "_source_material", {}).items():
                self._material_process_source.setdefault(k, {}).update({name: v})

    # -- Device and dtype methods --

    def cuda(self, device=None):
        """
        Move the population to a CUDA device, rebuilding mechanisms if needed.

        Parameters
        ----------
        device : int or torch.device, optional
            CUDA device identifier.

        Returns
        -------
        Population
            The population instance on the requested device.
        """
        self.build()
        return super().cuda(device=device)

    def cpu(self):
        """
        Move the population to CPU memory, rebuilding mechanisms if needed.

        Returns
        -------
        Population
            The population instance on CPU.
        """
        self.build()
        return super().cpu()

    def float(self):
        """
        Cast population parameters and buffers to ``torch.float32``.

        Returns
        -------
        Population
            The population instance with converted dtype.
        """
        self.build()
        return super().float()

    def double(self):
        """
        Cast population parameters and buffers to ``torch.float64``.

        Returns
        -------
        Population
            The population instance with converted dtype.
        """
        self.build()
        return super().double()

    def half(self):
        """
        Cast population parameters and buffers to ``torch.float16``.

        Returns
        -------
        Population
            The population instance with converted dtype.
        """
        self.build()
        return super().half()

    def bfloat16(self):
        """
        Cast population parameters and buffers to ``torch.bfloat16``.

        Returns
        -------
        Population
            The population instance with converted dtype.
        """
        self.build()
        return super().bfloat16()

    def to(self, *args, **kwargs):
        """
        Move the population to a new device or dtype, rebuilding if necessary.

        Parameters
        ----------
        *args : Any
            Positional arguments forwarded to :meth:`torch.nn.Module.to`.
        **kwargs : Any
            Keyword arguments forwarded to :meth:`torch.nn.Module.to`.

        Returns
        -------
        Population
            The population instance after conversion.
        """
        self.build()
        return super().to(*args, **kwargs)

    def build(self, force_rebuild=False):
        """
        Compile and register mechanisms, assembling ion bookkeeping.

        Parameters
        ----------
        force_rebuild : bool, optional
            If True, rebuild even when a compiled configuration already exists.

        Returns
        -------
        Population
            The population instance, ready for simulation.
        """
        self._refresh_compile_config_from_ctx()

        if self.is_built and not (force_rebuild or self._flag_rebuild):
            return self

        def are_strings_unique(data: list) -> bool:
            strings_only = [item for item in data if item is not None]
            return len(strings_only) == len(set(strings_only))

        conc = eq = nullcontext()

        if self._concentrations:
            conc = concentrations(**self._concentrations)
        if self._equilibria:
            eq = equilibria(**self._equilibria)

        with conc, eq:
            for mech, (name, ic, kwargs) in self._mech_everywhere.items():
                key = None
                shape = self._calc_shape_p()
                shape_f = self.shape
                mech.check_kwargs(kwargs)
                m = mech(
                    name, self.celsius, self.diam, shape, shape_f, key, ic=ic, **kwargs
                )
                self._register_mech(m, shape, key)

            for mech, data in self._mech_data.items():
                aliases, kwargs_list, keys = tuple(map(list, zip(*data)))
                if not are_strings_unique(aliases):
                    raise ValueError(
                        f"Duplicate aliases found for mechanism {mech.__name__}."
                    )
                m, shape, key = compile_mechanism(
                    self, mech, keys, aliases, kwargs_list
                )
                self._register_mech(m, shape, key)

            all_materials = get_unique_keys(
                [
                    self._material_read,
                    self._material_write,
                    self._material_source,
                    self._material_process_read,
                    self._material_process_write,
                    self._material_process_source,
                ]
            )
            all_materials.update(self._material_configs.keys())

            # If an ion species is used through the generic Material interface,
            # instantiate the Ion and expose the same object through both the
            # ion and material registries.  This lets USEMATERIAL("ca",
            # read=["cai", "eca"]) and USEION("ca", ...) share state.
            ion_like_materials = {m for m in all_materials if m in valid_ions()}
            all_ions = get_unique_keys(
                [self._ion_read, self._ion_write, self._ion_write_c]
            )
            all_ions.update(ion_like_materials)
            all_ions.update(self._ion_style.keys())

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
                    if not isinstance(m, MaterialProcess):
                        m.register_ion(ions[ion])

            materials = {}
            for material in sorted(all_materials):
                if material in ions:
                    continue
                try:
                    materials[material] = Material(
                        material,
                        self.shape,
                        **self._material_constructor_kwargs(material),
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Material {material!r} is used by a mechanism but has no registered fields. "
                        f"Call model.material({material!r}, fields=...) or register_material(...) before build()."
                    ) from exc

                for m in self._m_list:
                    if not isinstance(m, MaterialProcess):
                        m.register_material(materials[material])

            # Registered ions are Material subclasses.  Bind them for mechanisms
            # that requested USEMATERIAL("ca", ...), without inserting duplicate
            # references into the generic material ModuleDict.
            for ion_material in sorted(ion_like_materials):
                ion_h = ions[ion_material]
                for m in self._m_list:
                    if not isinstance(m, MaterialProcess):
                        m.register_material(ion_h)

            mechs = {n: m for n, m in zip(self._m_name, self._m_list)}
            keys = {n: k for n, k in zip(self._m_name, self._m_keys)}
            mech = MechanismHandler(
                self.celsius,
                self.area,
                mechs,
                ions=ions,
                materials=materials,
                write_ion_c=self._ion_write_c,
                read_ion=self._ion_read,
                read_material=self._material_read,
                write_material=self._material_write,
                source_material=self._material_source,
                currents=self._m_curr,
                population=self,
            )

            for m in mech.mechanisms.values():
                m.setreference("t", lambda: self.t)
            for m in getattr(mech, "material_processes", {}).values():
                m.setreference("t", lambda: self.t)

            # Give mechanisms a chance to consume waveform injections directly.
            # Mechanisms that do not implement injection support ignore these.
            self._dispatch_mechanism_injections(mech)

            self.integrator = self._integrator_class(self, mech, imem=self.imem)
            self.integrator.configure_jit(self, scope="population")
            self.mech = self.integrator.mech

        self.is_built = True
        self._flag_rebuild = False
        self.to(device=self.device(), dtype=self.dtype())
        return self

    def build_(self, force_rebuild=False):
        """
        "In-place" alias of :meth:`build`. Same as :meth:`build`,
        but does not return self.

        Parameters
        ----------
        force_rebuild : bool, optional
            Forwarded to :meth:`build`.
        """
        self.build(force_rebuild=force_rebuild)

    def detach(self):
        """
        Detach parameters and buffers from the autograd graph.

        Returns
        -------
        Population
            The population instance with detached states.
        """
        self.integrator.detach(self)
        return self

    def detach_(self):
        """
        "In-place" alias of :meth:`detach`. Same as :meth:`detach`,
        but does not return self.
        """
        self.detach()

    def register_parametrization(
        self, name: str, parametrization: torch.nn.Module, unsafe=True
    ):
        """
        Register a parametrization hook on a parameter tensor.

        Parameters
        ----------
        name : str
            Name of the parameter to parametrize.
        parametrization : torch.nn.Module
            Module providing the parametrization transform.
        unsafe : bool, optional
            Forwarded to :func:`torch.nn.utils.parametrize.register_parametrization`.
        """
        torch.nn.utils.parametrize.register_parametrization(
            self, name, parametrization, unsafe=unsafe
        )

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
        """
        Find indices that do not match provided patterns.

        Parameters
        ----------
        exclude : str or list of str, optional
            Patterns describing compartments to omit.
        fuzzy : bool, optional
            If True, enable fuzzy matching. Default is True.
        match_case : bool, optional
            If True, perform case-sensitive matching. Default is False.

        Returns
        -------
        list or slice
            Indices of entries that do not match ``exclude``.
        """
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
        """
        Locate compartment indices matching inclusion/exclusion rules.

        Parameters
        ----------
        include : str or list of str, optional
            Patterns that must be present.
        exclude : str or list of str, optional
            Patterns that must not be present. Defaults to ``'branchpoint'``.
        fuzzy : bool, optional
            If True, enable fuzzy matching. Default is True.
        match_case : bool, optional
            If True, perform case-sensitive matching. Default is False.
        full_report : bool, optional
            If True, return the full :class:`FindResult`. Default is False.
        as_list : bool, optional
            If True and the result is a slice, convert it to a list.
        loc : float, optional
            Optional location refinement parameter in ``[0, 1]``.

        Returns
        -------
        FindResult or Union[list, slice]
            Either the result object or the indices, depending on ``full_report``.
        """
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
        Indices of terminal nodes in the morphology tree.

        Returns
        -------
        list of int
            Node indices that have no outgoing edges in ``self.graph``.
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
        """
        Initialize membrane potential buffers via the integrator.
        """
        self.integrator.init_v(self)

    def init_v_(self):
        """
        In-place alias of :meth:`init_v`.
        """
        self.init_v()

    def n(self) -> int:
        """
        Number of compartments per neuron.

        Returns
        -------
        int
            Size of the penultimate dimension in ``self.v``.
        """
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

    # -- batching stuff --
    def is_batched(self):
        """
        Check whether the population has explicit batch dimensions.

        Returns
        -------
        bool
            True when ``self.v`` has more than two dimensions.
        """
        return len(self.shape) > 2

    def core_shape(self):
        """
        Shape of the neuron/compartment dimensions.

        Returns
        -------
        tuple of int
            Final two dimensions of ``self.v``.
        """
        return self.shape[-2:]

    def batched_shape(self):
        """
        Flattened shape suitable for batched integrator operations.

        Returns
        -------
        tuple of int
            Pair ``(batch_size, n_compartments)`` compatible with mechanism calls.
        """
        B = np.prod(self.shape[:-1])
        return (B, self.shape[-1])

    def n_batch_dimensions(self):
        """
        Number of leading batch dimensions in ``self.v``.

        Returns
        -------
        int
            Count of batch axes.
        """
        return len(self.shape) - 2

    def _calc_shape_p(self):
        """
        Compute per-parameter broadcast shape accounting for batching.

        Returns
        -------
        tuple of int
            Shape with singleton batch dimensions followed by the core shape.
        """
        return tuple([1] * self.n_batch_dimensions() + list(self.core_shape()))

    def batch(self, n):
        """
        Materialize explicit batch copies of state tensors.

        Parameters
        ----------
        n : int
            Number of batch replicas to create.

        Returns
        -------
        Population
            The population instance with replicated buffers.
        """
        if n <= 0:
            raise ValueError("Batch size n must be positive.")

        def _batch_tensor_attr(name: str):
            if not hasattr(self, name):
                return
            value = getattr(self, name)
            if value is None or not torch.is_tensor(value):
                return
            setattr(self, name, value.unsqueeze(0).expand(n, *value.shape).clone())

        _batch_tensor_attr("v")

        state_vars = {"v_prev", "vc"}
        integrator = getattr(self, "integrator", None)
        if integrator is not None:
            state_vars.update(getattr(integrator, "v_vars", ()))
        state_vars.discard("v")
        for name in sorted(state_vars):
            _batch_tensor_attr(name)

        if hasattr(self, "i_membrane") and self.i_membrane is not None:
            _batch_tensor_attr("i_membrane")

        self.reshape(self._calc_shape_p(), self.shape)
        for slice in self._labels.values():
            slice._batch()
        # now batch x, y, z
        self.x = self.x.unsqueeze(0).expand(n, *self.x.shape).clone()
        self.y = self.y.unsqueeze(0).expand(n, *self.y.shape).clone()
        self.z = self.z.unsqueeze(0).expand(n, *self.z.shape).clone()
        return self

    def batch_(self, n):
        """
        In-place alias of :meth:`batch`.

        Parameters
        ----------
        n : int
            Number of batch replicas forwarded to :meth:`batch`.
        """
        self.batch(n)

    # -- labeling stuff --
    def clear_labels(self):
        """
        Remove cached slice labels for compartments.
        """
        for name in self._labels.keys():
            delattr(self, name)
        self._labels.clear()

    # -- gradient checkpointing --
    def state_dict_for_checkpoint(self):
        """
        Get state dictionary for gradient checkpointing.

        Returns
        -------
        dict
            State dictionary containing model parameters and buffers.
        """
        mech_dct = self.mech.mutable_state_dict()
        integrator_dct = self.integrator.mutable_state_dict(self)
        full_dct = {
            "mech": mech_dct,
            "integrator": integrator_dct,
            "t": self.t,
        }
        return full_dct

    def restore_dict_from_checkpoint(self, state_dict):
        """
        Restore model state from a checkpoint state dictionary.

        Parameters
        ----------
        state_dict : dict
            State dictionary containing model parameters and buffers.
        """
        self.mech.restore_mutable_state_dict(state_dict["mech"])
        self.integrator.restore_mutable_state_dict(self, state_dict["integrator"])
        self.t = state_dict["t"]

    def longrun_checkpointed(
        self,
        tstop: float,
        chunklength: int,
        dt: float = None,
        extra: Optional[ExtraSpec] = None,
        callbacks: Optional[Sequence[Callback]] = None,
        progressbar=False,
        *,
        safe_checkpoint: bool = False,
        restore_state_after_backward: bool = True,
        return_final_state: bool = False,
    ):
        r"""
        Run a long simulation in chunks using activation checkpointing.

        This is a checkpointed analogue of :meth:`longrun`. Each chunk is wrapped
        in :func:`torch.utils.checkpoint.checkpoint` (with ``use_reentrant=False``),
        so intermediate activations inside the chunk are discarded and
        recomputed during the backward pass. This enables full BPTT across the
        entire ``tstop`` horizon while bounding activation memory by the chunk size.

        Parameters
        ----------
        tstop : float
            Total simulation time in milliseconds.
        chunklength : int
            Number of time steps per checkpointed chunk.
        dt : float, optional
            Time step size in milliseconds. If None, uses the global default
            ``A.dt``.
        extra : ExtraSpec, optional
            Extracellular configuration for the run. See :class:`ExtraSpec` for details.
        callbacks : Sequence[Callback] or CallbackList, optional
            Sequence of callback hooks to run at various points during the simulation.
            See :class:`Callback` for details.
        progressbar : bool, optional
            If True, display a progress bar during the run. Default is False.
        safe_checkpoint : bool, optional
            If True, clone checkpoint boundary state to avoid in-place mutation
            of checkpoint inputs. This is safer but may incur a memory overhead.
            Default is False.
        restore_state_after_backward : bool, optional
            If True, restore model state to the end of the forward pass after
            backward. This is useful when further simulation or evaluation is
            needed after backpropagation. Default is True.
        return_final_state : bool, optional
            If True, return a tuple (loss, final_state) where final_state is a
            checkpoint state dictionary suitable for restore_dict_from_checkpoint.
            Default is False.

        Returns
        -------
        torch.Tensor or None, or (torch.Tensor or None, dict)
            If return_final_state is False (default): returns the total loss contribution
            from callbacks, or None if no hook returned a non-None value.
            If return_final_state is True: returns (loss_or_none, final_state_dict).


        Notes
        -----
        **In-place update contract**

        ``torch.utils.checkpoint`` forbids in-place mutation of checkpoint *inputs*.
        If your stepping code (integrator/mechanisms) is strictly out-of-place with
        respect to the boundary state tensors, you may keep ``safe_checkpoint=False``
        (default) for best performance. If you are unsure, set
        ``safe_checkpoint=True`` to clone the boundary state at chunk entry.

        **Callback contract (restricted)**

        For correctness under checkpointing, callbacks used here must be replay-safe:

        * Hooks must be free of persistent side effects (e.g., do not append to
          Python lists intended to be consumed after the run).
        * Hooks must not mutate model state in a way that changes simulation dynamics.
        * If a hook uses internal temporary state, it must fully clear that state
          within the hook call itself.

        **Loss aggregation**

        Hooks may optionally return a scalar tensor loss contribution. Any non-None
        returns from the following hooks are summed:

        * ``post_step_hook(model)``
        * ``post_chunk_hook(model, t_chunk)``
        * ``post_loop_hook(model)``
        """
        if not self.initialized:
            raise ValueError("Model must be initialized before running.")
        self._refresh_compile_config_from_ctx()

        intra = self.intra
        with_intra = intra is not None

        # dt scalars
        dt = dt if dt is not None else A.dt
        dt_f = float(dt)
        dt_tensor = torch.tensor(dt, device=self.device(), dtype=self.dtype())

        # Normalize callback container
        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(callbacks)

        if callbacks:
            for c in callbacks:
                c.dt = dt_f

        def _as_loss_tensor(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x
            return torch.as_tensor(x, device=self.device(), dtype=self.dtype())

        def _add_loss(acc, x):
            x_t = _as_loss_tensor(x)
            if x_t is None:
                return acc, False
            if acc is None:
                return x_t, True
            return acc + x_t, True

        def _clone_checkpoint_state_dict(sd: Dict[str, Any]) -> Dict[str, Any]:
            """
            Clone all tensors in a checkpoint state dict to avoid in-place mutation
            of checkpoint inputs (a hard requirement of torch.utils.checkpoint).

            We preserve object-identity aliasing within the dict (if the same tensor
            object is referenced multiple times) by memoizing clones.
            """
            memo: Dict[int, torch.Tensor] = {}

            def _clone_any(v):
                if not torch.is_tensor(v):
                    return v
                key = id(v)
                if key in memo:
                    return memo[key]
                out = v.clone()
                memo[key] = out
                return out

            mech_in = sd.get("mech", {})
            integ_in = sd.get("integrator", {})

            mech_out = {k: _clone_any(v) for k, v in mech_in.items()}
            integ_out = {k: _clone_any(v) for k, v in integ_in.items()}

            return {
                "mech": mech_out,
                "integrator": integ_out,
                "t": _clone_any(sd["t"]),
            }

        def _copy_state_containers(sd: Dict[str, Any]) -> Dict[str, Any]:
            """
            Copy only the dict containers (not tensors). Useful to protect the
            backward-restore hook from accidental user mutation of the returned dict.
            """
            return {
                "mech": dict(sd.get("mech", {})),
                "integrator": dict(sd.get("integrator", {})),
                "t": sd["t"],
            }

        with torch.nn.utils.parametrize.cached():
            with torch.set_grad_enabled(self.training):
                # --------------------------------------------------------------
                # Global time grid and chunking
                # --------------------------------------------------------------
                t = torch.arange(
                    self.t.double(),
                    self.t.double() + tstop,
                    dt_f,
                    dtype=torch.double,
                    device=self.device(),
                ).to(self.dtype())

                if t.numel() == 0:
                    return None

                n_chunks = math.ceil(len(t) / chunklength)
                t_chunks = torch.tensor_split(t, n_chunks)

                # --------------------------------------------------------------
                # Extracellular configuration (single vs multi-contact, functional)
                # --------------------------------------------------------------
                extra_cfg = self._prepare_extra(extra, t, n_chunks)
                with_extra = extra_cfg.enabled

                # --------------------------------------------------------------
                # Integrator initialization + pre-loop hooks
                # --------------------------------------------------------------
                pre_loop_hook(callbacks, self)
                self.integrator._initialize(
                    self,
                    dt_tensor,
                    force=self.force_integrator_reinit(),
                    compile_scope="population",
                )

                if progressbar:
                    progressbar = tqdm(total=n_chunks, desc=f"{self.t.item():.1f} ms")

                # Capture boundary state after initialization
                state = self.state_dict_for_checkpoint()

                total_loss = None
                saw_any_loss = False

                # Local alias for speed
                _checkpoint = torch.utils.checkpoint.checkpoint

                # --------------------------------------------------------------
                # Main chunk loop (checkpointed)
                # --------------------------------------------------------------
                for i, t_chunk in enumerate(t_chunks):
                    chunk_idx = int(i)
                    t_chunk_local = t_chunk

                    def _run_chunk(
                        state_in, t_chunk_local=t_chunk_local, chunk_idx=chunk_idx
                    ):
                        # IMPORTANT: torch.utils.checkpoint forbids in-place mutation of
                        # checkpoint *inputs*. If out-of-place stepping is guaranteed,
                        # we can safely reuse the boundary tensors. Otherwise, clone.
                        state_local = (
                            state_in
                            if not safe_checkpoint
                            else _clone_checkpoint_state_dict(state_in)
                        )

                        # Restore model state at chunk boundary
                        self.restore_dict_from_checkpoint(state_local)

                        if with_intra:
                            stims, indices = intra.init(t_chunk_local)
                            stims = [s.unbind(0) for s in stims]
                        else:
                            stims, indices = None, None

                        if with_extra:
                            ve_list = self._compute_extra_chunk(
                                extra_cfg, chunk_idx, t_chunk_local
                            )
                        else:
                            ve_list = None

                        pre_chunk_hook(callbacks, self, t_chunk_local)

                        chunk_loss = None
                        saw_loss_local = False

                        for j in range(len(t_chunk_local)):
                            ve_c = ve_list[j] if ve_list is not None else None

                            if with_intra:
                                s = [st[j] for st in stims]
                                intra_c = self.make_intra(intra, s, indices)
                            else:
                                intra_c = None

                            self._step(self.integrator, self, dt_tensor, ve_c, intra_c)

                            # Replay-safe post-step callbacks (may contribute loss)
                            if callbacks:
                                for c in callbacks:
                                    hook = getattr(c, "post_step_hook", None)
                                    if hook is None:
                                        continue
                                    chunk_loss, saw = _add_loss(chunk_loss, hook(self))
                                    saw_loss_local = saw_loss_local or saw

                            self.t = self.t + dt_tensor

                        # Replay-safe post-chunk callbacks (may contribute loss)
                        if callbacks:
                            for c in callbacks:
                                hook = getattr(c, "post_chunk_hook", None)
                                if hook is None:
                                    continue
                                chunk_loss, saw = _add_loss(
                                    chunk_loss, hook(self, t_chunk_local)
                                )
                                saw_loss_local = saw_loss_local or saw

                        state_out = self.state_dict_for_checkpoint()

                        if chunk_loss is None:
                            chunk_loss = torch.zeros(
                                (),
                                device=self.device(),
                                dtype=self.dtype(),
                            )

                        saw_loss_flag = torch.tensor(
                            1 if saw_loss_local else 0,
                            device=self.device(),
                            dtype=torch.int32,
                        )

                        return state_out, chunk_loss, saw_loss_flag

                    state, chunk_loss, saw_loss_flag = _checkpoint(
                        _run_chunk, state, use_reentrant=False, determinism_check="none"
                    )

                    total_loss, _ = _add_loss(total_loss, chunk_loss)
                    saw_any_loss = saw_any_loss or bool(int(saw_loss_flag.item()))

                    if progressbar:
                        progressbar.update(1)
                        progressbar.set_description(f"{self.t.item():.1f} ms")

                # Replay-safe post-loop callbacks (may contribute loss)
                if callbacks:
                    for c in callbacks:
                        hook = getattr(c, "post_loop_hook", None)
                        if hook is None:
                            continue
                        total_loss, saw = _add_loss(total_loss, hook(self))
                        saw_any_loss = saw_any_loss or saw

                if progressbar:
                    progressbar.close()

        if not saw_any_loss:
            if return_final_state:
                return None, self.state_dict_for_checkpoint()
            return None

        # ------------------------------------------------------------------
        # IMPORTANT: preserve forward-final model state across backward.
        #
        # With checkpointing, backward re-runs chunk forwards and therefore
        # re-mutates self.t / buffers. Without intervention, the module state
        # after loss.backward() will typically reflect the last recomputed
        # chunk, not the true forward-final state.
        #
        # We snapshot the final state and schedule a restoration callback at
        # the *end* of backward.
        # ------------------------------------------------------------------
        final_state = None
        if restore_state_after_backward or return_final_state:
            final_state = self.state_dict_for_checkpoint()

        if restore_state_after_backward:
            # Protect hook state from accidental external mutation of the dict structure.
            final_state_for_hook = _copy_state_containers(final_state)

            def _queue_restore(grad, fs=final_state_for_hook):
                # Must be called during backward; this schedules restore after
                # the autograd engine finishes the backward pass.
                torch.autograd.Variable._execution_engine.queue_callback(
                    lambda: self.restore_dict_from_checkpoint(fs)
                )
                return grad

            # Only meaningful if backward will actually run through this tensor.
            # (register_hook requires requires_grad=True)
            if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
                total_loss.register_hook(_queue_restore)

        if return_final_state:
            return total_loss, final_state
        return total_loss

    # ---------- small formatting helpers ----------
    @staticmethod
    def _fmt_scalar(x: Any) -> str:
        """Best-effort scalar formatting; avoids dumping tensors."""
        if x is None:
            return "<?>"
        if isinstance(x, (bool, int)):
            return str(x)
        if isinstance(x, float):
            # compact but readable
            return f"{x:g}"
        if isinstance(x, str):
            return x
        if torch.is_tensor(x):
            t = x.detach()
            if t.numel() == 1:
                # NOTE: .item() can sync on CUDA; still usually acceptable for a scalar.
                try:
                    return f"{t.item():g}"
                except Exception:
                    return str(t)
            return f"Tensor(shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device})"
        return str(x)

    @staticmethod
    def _linspace_indices(n_total: int, n_samples: int) -> List[int]:
        """Deterministic sample indices without allocating big tensors."""
        if n_total <= 0:
            return []
        if n_total <= n_samples:
            return list(range(n_total))
        if n_samples <= 1:
            return [0]
        step = (n_total - 1) / (n_samples - 1)
        idx = [int(round(i * step)) for i in range(n_samples)]
        # ensure monotonic + in-bounds
        idx = [min(max(i, 0), n_total - 1) for i in idx]
        # de-dup while preserving order
        out = []
        seen = set()
        for i in idx:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out

    def _fmt_tensor_param(
        self,
        t: Any,
        *,
        units: str = "",
        stats: Optional[str] = None,  # overrides _REPR_TENSOR_STATS
    ) -> str:
        """
        Summarize a parameter tensor:
        - scalar -> value
        - big tensor -> shape (+ optional sample/full stats)
        """
        if t is None:
            return "<?>"

        stats = stats or self._REPR_TENSOR_STATS

        if isinstance(t, (float, int)):
            return f"{t:g}{units}"

        if not torch.is_tensor(t):
            return f"{t}{units}"

        x = t.detach()
        shape = tuple(x.shape)

        if x.numel() == 1:
            try:
                return f"{x.item():g}{units}"
            except Exception:
                return (
                    f"Tensor(shape={shape}, dtype={x.dtype}, device={x.device}){units}"
                )

        # stats="none" is the safest/cheapest: no reductions, no sampling.
        if stats == "none":
            return f"Tensor(shape={shape}, dtype={x.dtype}, device={x.device}){units}"

        # "sample" stats: only look at a few points (cheap, bounded)
        flat = x.reshape(-1)
        idx = self._linspace_indices(flat.numel(), self._REPR_TENSOR_SAMPLES)

        # gather a few scalar values (bounded work)
        vals: List[float] = []
        for i in idx:
            try:
                vals.append(float(flat[i].item()))
            except Exception:
                # if we can't safely scalarize, fall back
                return (
                    f"Tensor(shape={shape}, dtype={x.dtype}, device={x.device}){units}"
                )

        vmin = min(vals)
        vmax = max(vals)
        if vmin == vmax:
            # "probably constant" (sample-based)
            return f"≈{vmin:g}{units} (const?; shape={shape})"
        return f"shape={shape}, sample≈[{vmin:g}, {vmax:g}]{units}"

    @staticmethod
    def _fmt_index(idx: Any) -> str:
        """Compact placement/index description."""
        if idx is None:
            return "everywhere"
        if isinstance(idx, slice):
            return f"slice({idx.start},{idx.stop},{idx.step})"
        if isinstance(idx, tuple) and all(isinstance(x, slice) for x in idx):
            inner = ", ".join(f"{s.start}:{s.stop}:{s.step}" for s in idx)
            return f"slices({inner})"
        # list/ndarray/tensor indices can be huge: summarize length/type
        if isinstance(idx, (list, tuple)):
            return f"{type(idx).__name__}(len={len(idx)})"
        if isinstance(idx, np.ndarray):
            return f"ndarray(shape={idx.shape}, dtype={idx.dtype})"
        if torch.is_tensor(idx):
            return f"Tensor(shape={tuple(idx.shape)}, dtype={idx.dtype}, device={idx.device})"
        return type(idx).__name__

    def _fmt_kwargs(self, kwargs: Dict[str, Any], max_items: int = 4) -> str:
        """Short kwargs summary that won’t explode logs."""
        if not kwargs:
            return ""
        items = []
        for k in sorted(kwargs.keys()):
            v = kwargs[k]
            if isinstance(v, (bool, int, float, str)):
                items.append(f"{k}={v}")
            elif torch.is_tensor(v):
                if v.numel() == 1:
                    try:
                        items.append(f"{k}={v.detach().item():g}")
                    except Exception:
                        items.append(f"{k}=Tensor{tuple(v.shape)}")
                else:
                    items.append(f"{k}=Tensor{tuple(v.shape)}")
            elif isinstance(v, (list, tuple, dict)):
                items.append(f"{k}={type(v).__name__}(len={len(v)})")
            else:
                items.append(f"{k}={type(v).__name__}")
            if len(items) >= max_items:
                break
        extra = len(kwargs) - len(items)
        suffix = f", …+{extra}" if extra > 0 else ""
        return "{" + ", ".join(items) + suffix + "}"

    @staticmethod
    def _fmt_number(v, sigfigs: int = 6) -> str:
        if isinstance(v, bool):
            return "True" if v else "False"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.{sigfigs}g}"
        if isinstance(v, complex):
            return f"{v.real:.{sigfigs}g}{v.imag:+.{sigfigs}g}j"
        return str(v)

    def _fmt_tensor_value(
        self,
        t: object,
        *,
        tensor_stats: str = "sample",  # "none" | "sample" | "full"
        full_max_elems: int = None,
        sigfigs: int = None,
    ) -> str:
        """
        Scalar -> value
        Small tensor -> full values
        Large tensor -> stats summary (none/sample/full)
        """
        if t is None:
            return "<?>"

        full_max_elems = (
            self._REPR_FULL_TENSOR_MAX_ELEMS
            if full_max_elems is None
            else full_max_elems
        )
        sigfigs = self._REPR_FLOAT_SIGFIGS if sigfigs is None else sigfigs

        if isinstance(t, (float, int, bool, np.number)):
            return self._fmt_number(t, sigfigs=sigfigs)

        if not torch.is_tensor(t):
            return str(t)

        x = t.detach()
        shape = tuple(x.shape)
        dtype = x.dtype
        device = x.device

        # Scalar tensor
        if x.numel() == 1:
            try:
                return self._fmt_number(x.item(), sigfigs=sigfigs)
            except Exception:
                return f"Tensor(shape={shape}, dtype={dtype}, device={device})"

        # If "small enough", print full contents
        if x.numel() <= full_max_elems:
            try:
                y = x
                # numpy doesn't like bfloat16; cast for display only
                if y.dtype == torch.bfloat16:
                    y = y.to(torch.float32)
                if y.device.type != "cpu":
                    y = y.cpu()
                arr = y.numpy()

                s = np.array2string(
                    arr,
                    separator=", ",
                    formatter={"float_kind": lambda v: f"{float(v):.{sigfigs}g}"},
                )
                return f"{s} (shape={shape})"
            except Exception:
                # fallback
                return f"Tensor(shape={shape}, dtype={dtype}, device={device})"

        # Large tensor: stats
        if tensor_stats == "none":
            return f"Tensor(shape={shape}, dtype={dtype}, device={device})"

        if tensor_stats == "full":
            # true min/max (can be expensive; user opted in)
            try:
                y = x
                if y.dtype == torch.bfloat16:
                    y = y.to(torch.float32)
                vmin = y.amin().item()
                vmax = y.amax().item()
                if vmin == vmax:
                    return f"≈{self._fmt_number(vmin, sigfigs=sigfigs)} (const; shape={shape}, dtype={dtype}, device={device})"
                return (
                    f"shape={shape}, min={self._fmt_number(vmin, sigfigs=sigfigs)}, "
                    f"max={self._fmt_number(vmax, sigfigs=sigfigs)} (dtype={dtype}, device={device})"
                )
            except Exception:
                # fall through to sample if full fails
                pass

        # tensor_stats == "sample" (default)
        try:
            if not x.is_contiguous():
                # avoid accidental huge copies when flattening
                return f"Tensor(shape={shape}, dtype={dtype}, device={device}, noncontiguous=True)"

            flat = x.reshape(-1)
            idx = self._linspace_indices(flat.numel(), self._REPR_TENSOR_SAMPLES)

            # gather samples with one device->host transfer
            idx_t = torch.tensor(idx, device=flat.device, dtype=torch.long)
            samples = flat.index_select(0, idx_t).detach()
            if samples.dtype == torch.bfloat16:
                samples = samples.to(torch.float32)
            samples_cpu = samples.cpu()
            vals = samples_cpu.flatten().tolist()

            vmin = min(vals)
            vmax = max(vals)
            if vmin == vmax:
                return f"≈{self._fmt_number(vmin, sigfigs=sigfigs)} (const?; shape={shape})"
            return f"shape={shape}, sample≈[{self._fmt_number(vmin, sigfigs=sigfigs)}, {self._fmt_number(vmax, sigfigs=sigfigs)}]"
        except Exception:
            return f"Tensor(shape={shape}, dtype={dtype}, device={device})"

    # ---------- mechanisms / ions reporting ----------
    def _pending_mech_lines(self, *, verbose: bool, show_kwargs: bool) -> List[str]:
        lines: List[str] = []

        # Everywhere mechanisms
        ev = sorted(self._mech_everywhere.items(), key=lambda kv: kv[0].__name__)
        idx = sorted(self._mech_data.items(), key=lambda kv: kv[0].__name__)

        n_ev = len(ev)
        n_regions = sum(len(v) for _, v in idx)
        n_types = len(set([k.__name__ for k, _ in ev] + [k.__name__ for k, _ in idx]))

        if n_ev == 0 and n_regions == 0:
            lines.append("mechanisms: (none inserted)")
            return lines

        lines.append(
            f"mechanisms: {n_types} types "
            f"(everywhere={n_ev}, indexed_regions={n_regions})"
        )

        if not verbose:
            # one-line-ish names
            names = []
            for mech_cls, (name, ic, kwargs) in ev:
                names.append(name)
            for mech_cls, regions in idx:
                names.append(f"{mech_cls.__name__}×{len(regions)}")
            names = sorted(names)
            preview = names[: self._REPR_MAX_MECHS]
            more = len(names) - len(preview)
            s = ", ".join(preview) + (f", …+{more}" if more > 0 else "")
            lines.append(f"  {s}")
            return lines

        # Verbose: list each insertion
        for mech_cls, (name, ic, kwargs) in ev:
            kw = f" {self._fmt_kwargs(kwargs)}" if (show_kwargs and kwargs) else ""
            ic_s = "" if ic is None else f" ic={type(ic).__name__}"
            lines.append(f"  - {name} @ everywhere{ic_s}{kw}")

        for mech_cls, regions in idx:
            # regions: List[(alias, kwargs, key)]
            lines.append(f"  - {mech_cls.__name__} @ {len(regions)} region(s):")
            for alias, kwargs, key in regions[: self._REPR_MAX_MECHS]:
                nm = alias if alias is not None else mech_cls.__name__
                kw = f" {self._fmt_kwargs(kwargs)}" if (show_kwargs and kwargs) else ""
                lines.append(f"      • {nm} @ {self._fmt_index(key)}{kw}")
            if len(regions) > self._REPR_MAX_MECHS:
                lines.append(f"      • …+{len(regions) - self._REPR_MAX_MECHS} more")

        return lines

    def _built_mech_lines(
        self,
        *,
        verbose: bool,
        show_mechanism_parameters: bool = False,
        tensor_stats: str = "sample",
    ) -> List[str]:
        lines: List[str] = []
        mech_obj = getattr(self, "mech", None)
        if mech_obj is None:
            lines.append("mechanisms: built=True (handler missing?)")
            return lines

        try:
            mech_dict = mech_obj.mechanisms  # dict-like of name -> nn.Module
            names = sorted(list(mech_dict.keys()))
        except Exception:
            names = sorted(list(set(self._m_name)))

        key_map = {}
        if hasattr(self, "_m_name") and hasattr(self, "_m_keys"):
            for n, k in zip(self._m_name, self._m_keys):
                key_map[n] = k

        lines.append(f"mechanisms: {len(names)} built")

        if not verbose:
            preview = names[: self._REPR_MAX_MECHS]
            more = len(names) - len(preview)
            s = ", ".join(preview) + (f", …+{more}" if more > 0 else "")
            lines.append(f"  {s}")
            return lines

        for n in names:
            k = key_map.get(n, None)
            pshape = self._m_shape.get(n, None)
            pshape_s = f", pshape={pshape}" if pshape is not None else ""

            mod = None
            try:
                mod = mech_obj.mechanisms[n]
            except Exception:
                mod = None

            cls_s = mod._get_name() if isinstance(mod, torch.nn.Module) else "<?>"

            header = f"  - {n} ({cls_s}) @ {self._fmt_index(k)}{pshape_s}"

            if show_mechanism_parameters and isinstance(mod, torch.nn.Module):
                lines.append(header + " {")
                lines.extend(
                    self._mechanism_parameter_block_lines(
                        mod,
                        tensor_stats=tensor_stats,
                        indent0="    ",
                        indent1="      ",
                    )
                )
                lines.append("  }")
            else:
                lines.append(header)

        return lines

    def _infer_ion_usage_from_inserted(self) -> Dict[str, Dict[str, bool]]:
        """
        Infer ion usage without build() by reading mechanism class metadata:
        _read_ion/_write_ion/_write_ion_c.
        """
        mech_classes = set(
            list(self._mech_everywhere.keys()) + list(self._mech_data.keys())
        )
        usage: Dict[str, Dict[str, bool]] = {}

        for mech_cls in mech_classes:
            read_ion = getattr(mech_cls, "_read_ion", {}) or {}
            write_ion = getattr(mech_cls, "_write_ion", {}) or {}
            write_ion_c = getattr(mech_cls, "_write_ion_c", {}) or {}
            read_material = getattr(mech_cls, "_read_material", {}) or {}
            write_material = getattr(mech_cls, "_write_material", {}) or {}
            source_material = getattr(mech_cls, "_source_material", {}) or {}

            for ion, vars_ in read_ion.items():
                u = usage.setdefault(
                    ion,
                    {
                        "read_c": False,
                        "read_e": False,
                        "write_i": False,
                        "write_c": False,
                    },
                )
                if isinstance(vars_, (list, tuple)):
                    # match your runtime logic in _c_is_read / _e_is_read
                    if (f"{ion}i" in vars_) or (f"{ion}o" in vars_):
                        u["read_c"] = True
                    if f"e{ion}" in vars_:
                        u["read_e"] = True
                else:
                    # if unknown structure, mark as "reads something"
                    u["read_c"] = True

            for ion in write_ion.keys():
                u = usage.setdefault(
                    ion,
                    {
                        "read_c": False,
                        "read_e": False,
                        "write_i": False,
                        "write_c": False,
                    },
                )
                u["write_i"] = True

            for ion in write_ion_c.keys():
                u = usage.setdefault(
                    ion,
                    {
                        "read_c": False,
                        "read_e": False,
                        "write_i": False,
                        "write_c": False,
                    },
                )
                u["write_c"] = True

            # Ion-like materials are represented by Ion at build time.  Include
            # them in the pre-build ion summary/style inference so a mechanism
            # using USEMATERIAL("ca", read=["cai", "eca"], write=["cai"])
            # is reported consistently.
            for ion, vars_ in read_material.items():
                if ion not in valid_ions():
                    continue
                u = usage.setdefault(
                    ion,
                    {
                        "read_c": False,
                        "read_e": False,
                        "write_i": False,
                        "write_c": False,
                    },
                )
                if (f"{ion}i" in vars_) or (f"{ion}o" in vars_):
                    u["read_c"] = True
                if f"e{ion}" in vars_:
                    u["read_e"] = True

            for ion in set(write_material.keys()) | set(source_material.keys()):
                if ion not in valid_ions():
                    continue
                u = usage.setdefault(
                    ion,
                    {
                        "read_c": False,
                        "read_e": False,
                        "write_i": False,
                        "write_c": False,
                    },
                )
                u["write_c"] = True

        # include any explicitly styled ions even if no mech references them
        for ion in getattr(self, "_ion_style", {}).keys():
            usage.setdefault(
                ion,
                {"read_c": False, "read_e": False, "write_i": False, "write_c": False},
            )

        return usage

    def _built_ion_usage(self) -> Dict[str, Dict[str, bool]]:
        """Ion usage from built bookkeeping dictionaries."""
        ion_like_materials = (
            set(self._material_read.keys())
            | set(self._material_write.keys())
            | set(self._material_source.keys())
            | set(self._material_process_read.keys())
            | set(self._material_process_write.keys())
            | set(self._material_process_source.keys())
            | set(getattr(self, "_material_configs", {}).keys())
        ) & set(valid_ions())
        ions = (
            set(self._ion_read.keys())
            | set(self._ion_write.keys())
            | set(self._ion_write_c.keys())
            | set(self._ion_style.keys())
            | ion_like_materials
        )
        usage: Dict[str, Dict[str, bool]] = {}
        for ion in ions:
            usage[ion] = {
                "read_c": bool(self._c_is_read(ion)),
                "read_e": bool(self._e_is_read(ion)),
                "write_i": bool(self._ion_write.get(ion, {})),
                "write_c": bool(
                    self._ion_write_c.get(ion, {})
                    or self._material_write.get(ion, {})
                    or self._material_source.get(ion, {})
                ),
            }
        return usage

    def _ion_lines(self, *, verbose: bool) -> List[str]:
        built = bool(getattr(self, "is_built", False))
        usage = (
            self._built_ion_usage() if built else self._infer_ion_usage_from_inserted()
        )
        ions = sorted(list(usage.keys()))

        if not ions:
            return ["ions: (none)"]

        def flags(u: Dict[str, bool]) -> str:
            bits = []
            if u.get("write_i"):
                bits.append("write_i")
            if u.get("write_c"):
                bits.append("write_c")
            if u.get("read_e"):
                bits.append("read_e")
            if u.get("read_c"):
                bits.append("read_c")
            return ",".join(bits) if bits else "no-io"

        lines: List[str] = [f"ions: {len(ions)} ({'built' if built else 'inferred'})"]

        if not verbose:
            preview = ions[: self._REPR_MAX_IONS]
            more = len(ions) - len(preview)
            s = ", ".join(preview) + (f", …+{more}" if more > 0 else "")
            lines.append(f"  {s}")
            return lines

        for ion in ions:
            u = usage[ion]
            # show explicit style if present, else (if built) show computed style
            style_s = ""
            if ion in getattr(self, "_ion_style", {}):
                style_s = f", style={self._ion_style[ion]}(explicit)"
            elif built:
                try:
                    style_s = f", style={self.get_ion_style(ion)}"
                except Exception:
                    style_s = ""
            lines.append(f"  - {ion}: {flags(u)}{style_s}")

        # also show whether equilibria/concentrations configs exist
        if verbose:
            if getattr(self, "_equilibria", {}):
                lines.append(
                    f"  equilibria: keys={sorted(list(self._equilibria.keys()))}"
                )
            if getattr(self, "_concentrations", {}):
                lines.append(
                    f"  concentrations: keys={sorted(list(self._concentrations.keys()))}"
                )

        return lines

    def _infer_material_usage_from_inserted(self) -> Dict[str, Dict[str, int]]:
        """Infer generic Material usage from pending mechanism class metadata."""
        mech_classes = set(
            list(self._mech_everywhere.keys()) + list(self._mech_data.keys())
        )
        usage: Dict[str, Dict[str, int]] = {}

        for mech_cls in mech_classes:
            read_material = getattr(mech_cls, "_read_material", {}) or {}
            write_material = getattr(mech_cls, "_write_material", {}) or {}
            source_material = getattr(mech_cls, "_source_material", {}) or {}

            for material, fields in read_material.items():
                u = usage.setdefault(material, {"read": 0, "write": 0, "source": 0})
                u["read"] += len(fields)
            for material, fields in write_material.items():
                u = usage.setdefault(material, {"read": 0, "write": 0, "source": 0})
                u["write"] += len(fields)
            for material, fields in source_material.items():
                u = usage.setdefault(material, {"read": 0, "write": 0, "source": 0})
                u["source"] += len(fields)

        for material in getattr(self, "_material_configs", {}).keys():
            usage.setdefault(material, {"read": 0, "write": 0, "source": 0})
        return usage

    def _built_material_usage(self) -> Dict[str, Dict[str, int]]:
        materials = (
            set(self._material_read.keys())
            | set(self._material_write.keys())
            | set(self._material_source.keys())
            | set(self._material_process_read.keys())
            | set(self._material_process_write.keys())
            | set(self._material_process_source.keys())
            | set(getattr(self, "_material_configs", {}).keys())
        )
        mech_handler = getattr(self, "mech", None)
        if mech_handler is not None and hasattr(mech_handler, "materials"):
            materials |= set(mech_handler.materials.keys())

        usage: Dict[str, Dict[str, int]] = {}
        for material in materials:
            usage[material] = {
                "read": sum(
                    len(v) for v in self._material_read.get(material, {}).values()
                ),
                "write": sum(
                    len(v) for v in self._material_write.get(material, {}).values()
                ),
                "source": sum(
                    len(v) for v in self._material_source.get(material, {}).values()
                ),
            }
        return usage

    def _material_lines(self, *, verbose: bool) -> List[str]:
        built = bool(getattr(self, "is_built", False))
        usage = (
            self._built_material_usage()
            if built
            else self._infer_material_usage_from_inserted()
        )
        materials = sorted(list(usage.keys()))

        if not materials:
            return ["materials: (none)"]

        lines: List[str] = [
            f"materials: {len(materials)} ({'built' if built else 'inferred'})"
        ]
        if not verbose:
            preview = materials[: self._REPR_MAX_MATERIALS]
            more = len(materials) - len(preview)
            s = ", ".join(preview) + (f", …+{more}" if more > 0 else "")
            lines.append(f"  {s}")
            return lines

        for material in materials:
            u = usage[material]
            bits = []
            if u.get("read", 0):
                bits.append(f"read={u['read']}")
            if u.get("write", 0):
                bits.append(f"write={u['write']}")
            if u.get("source", 0):
                bits.append(f"source={u['source']}")
            if not bits:
                bits.append("registered")

            fields = ""
            if (
                built
                and getattr(self, "mech", None) is not None
                and hasattr(self.mech, "materials")
            ):
                if material in self.mech.materials:
                    material_h = self.mech.materials[material]
                    fields = f", fields={tuple(getattr(material_h, 'fields', ()))!r}"
            elif material in valid_materials():
                fields = (
                    f", registered_fields={tuple(material_specs()[material].keys())!r}"
                )

            lines.append(f"  - {material}: {', '.join(bits)}{fields}")
        return lines

    def _mechanism_parameter_block_lines(
        self,
        mech_module: torch.nn.Module,
        *,
        tensor_stats: str,
        indent0: str = "    ",
        indent1: str = "      ",
        max_entries: int = None,
    ) -> List[str]:
        """
        Returns lines like:
            parameters {
              a: 1
              w: [ ... ] (shape=(...))
            }
        """
        max_entries = (
            self._REPR_MAX_MECH_PARAM_ENTRIES if max_entries is None else max_entries
        )

        params = list(mech_module.named_parameters(recurse=True))
        # deterministic order
        params.sort(key=lambda kv: kv[0])

        if not params:
            return [f"{indent0}parameters {{", f"{indent1}(none)", f"{indent0}}}"]

        lines: List[str] = [f"{indent0}parameters {{"]

        shown = 0
        for name, p in params:
            if shown >= max_entries:
                break
            grad_tag = " (grad)" if getattr(p, "requires_grad", False) else ""
            val_s = self._fmt_tensor_value(p, tensor_stats=tensor_stats)
            lines.append(f"{indent1}{name}{grad_tag}: {val_s}")
            shown += 1

        remaining = len(params) - shown
        if remaining > 0:
            lines.append(f"{indent1}…+{remaining} more")

        lines.append(f"{indent0}}}")
        return lines

    # ---------- single source of truth for repr lines ----------
    def _repr_lines(
        self,
        *,
        verbose: bool = False,
        show_kwargs: Optional[bool] = None,
        tensor_stats: Optional[str] = None,
        show_mechanism_parameters: bool = False,
    ) -> List[str]:
        show_kwargs = self._REPR_SHOW_KWARGS if show_kwargs is None else show_kwargs
        if tensor_stats is not None:
            self._REPR_TENSOR_STATS = tensor_stats  # allow per-call override

        lines: List[str] = []

        # Shape / batching
        if self.is_batched():
            lines.append(
                f"batch_shape={self.shape[:-2]}, core_shape={self.core_shape()}, shape={self.shape}"
            )
        else:
            lines.append(f"core_shape={self.core_shape()} (np={self.np}, nc={self.nc})")

        lines.append(
            f"device={self.device()}, dtype={self.dtype()}, built={self.is_built}, rebuild_pending={self._flag_rebuild}"
        )

        # Parameters: show celsius scalar; cm/rhoa summarized (sample by default)
        lines.append(
            "params: "
            f"celsius={self._fmt_scalar(getattr(self, 'celsius', None))} °C, "
            f"cm={self._fmt_tensor_param(getattr(self, 'cm', None), units='')}, "
            f"rhoa={self._fmt_tensor_param(getattr(self, 'rhoa', None), units='')}"
        )

        # Mechanisms
        if self.is_built:
            lines.extend(
                self._built_mech_lines(
                    verbose=verbose,
                    show_mechanism_parameters=(show_mechanism_parameters and verbose),
                    tensor_stats=tensor_stats or "sample",
                )
            )
        else:
            lines.extend(
                self._pending_mech_lines(verbose=verbose, show_kwargs=show_kwargs)
            )

        # Ions and generic materials
        lines.extend(self._ion_lines(verbose=verbose))
        lines.extend(self._material_lines(verbose=verbose))

        return lines

    # ---------- the PyTorch hook ----------
    def extra_repr(self) -> str:
        # concise by default; keep it bounded
        return "\n".join(self._repr_lines(verbose=False))

    # optional: explicit verbose summary for debugging/logging
    def pretty(
        self,
        *,
        show_kwargs: bool = False,
        tensor_stats: str = "sample",
        show_mechanism_parameters: bool = False,
        indent: int = 2,
    ) -> str:
        """
        Human-friendly, block-formatted summary:

        Population {
          ...
        }
        """
        name = self._get_name()  # nn.Module hook (defaults to class name)

        # your existing verbose lines
        lines = self._repr_lines(
            verbose=True,
            show_kwargs=show_kwargs,
            tensor_stats=tensor_stats,
            show_mechanism_parameters=show_mechanism_parameters,
        )

        body = "\n".join(lines).rstrip()
        if body:
            body = textwrap.indent(body, " " * indent)

        # handle empty body gracefully
        if not body:
            return f"{name} {{\n}}"

        return f"{name} {{\n{body}\n}}"


class SingleCompartment(Population):
    """
    A Population subclass representing single compartment neuron(s).

    This class is a convenience wrapper around the Population class,
    pre-configured for a single compartment model.
    """

    pass


# Define the return type for clarity
class FindResult(NamedTuple):
    """Container for compartment search results."""

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
        torch_indices = torch.tensor(numpy_indices, dtype=torch.long)
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

    Notes
    -----
    - {py:class}`~dendra.models.core.Unmyelinated` and {py:class}`~dendra.models.core.Myelinated` extend this class.
    - The axon is represented as a series of compartments arranged in a line,
      with each compartment having its own diameter and biophysical properties.
      All axon classes by defalt are instantiate such that all compartments
      lie along the x-axis (y=z=0), with the central compartment at x=0 (thereby
      spanning from -axon_length/2 to axon_length/2).
    """

    __constants__ = [
        "n_ax",
        "n_comp",
        "temp",
    ]

    def __init__(
        self,
        diameters,
        n_comp: int,
        celsius=37.0,
        v_init=-80.0,
        integrator=None,
        **kwargs,
    ):
        if integrator is None:
            integrator = bwd_euler_ub()
        super().__init__(
            len(diameters),
            n_comp,
            integrator=integrator,
            celsius=celsius,
            v_init=v_init,
            **kwargs,
        )

        self.register_buffer(
            "diameters", torch.as_tensor(diameters, dtype=self.dtype())
        )

        self.n_ax = self.np
        self.n_comp = self.nc

        # ``v_init`` is normalized by Population; it may be scalar or a
        # length-n_comp vector. ``self.v`` was already created from it.

        self.x[:] = self._x()  # Initialize x positions

        self.cid = None

        if torch.is_tensor(diameters):
            diameters = diameters.to(self.dtype()).clone().detach()
        else:
            diameters = torch.tensor(diameters, dtype=self.dtype())

        if diameters.ndim == 1:
            diameters = diameters.unsqueeze(1)

        self.diam[:] = diameters
        self.diam = self.diam.detach()

    def assemble_graphs(self):
        """
        Construct directed path graphs for each axon.

        Returns
        -------
        list of networkx.DiGraph
            Morphology graphs annotated with geometry metadata.
        """
        graphs = []
        for i in range(self.n_ax):
            G = nx.path_graph(self.n_comp, create_using=nx.DiGraph)
            for node in G.nodes:
                G.nodes[node]["name"] = (
                    f"{self.__class__.__name__}[{i}]({node / (self.n_comp - 1):.2f})"
                )
                G.nodes[node]["x"] = self.x[i, node].item()
                G.nodes[node]["y"] = self.y[i, node].item()
                G.nodes[node]["z"] = self.z[i, node].item()
                G.nodes[node]["diam"] = self.diam[i, node].item()
                G.nodes[node]["L"] = self.dx[i, node].item()
                G.nodes[node]["Ra"] = self.rhoa[i, node].item()
                G.nodes[node]["Cm"] = self.cm[i, node].item()
            graphs.append(G)
        return graphs

    def register_cid(self, cid):
        """
        Register a compartment identifier table for slicing utilities.

        Parameters
        ----------
        cid : Any
            Object exposing ``names`` used for label-based slicing.
        """
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

    def csl(self, *args):
        """
        Slice the axon at specified relative positions.

        Parameters
        ----------
        *args : float
            Variable number of float values between 0 and 1, representing
            relative positions along the axon.

        Returns
        -------
        Slice
            A sliced view of the axon at the specified relative positions.
        """
        return self[:, self.c(*args)]


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
        **kwargs,
    ):
        # L = L * 1000  # mm -> um
        n_comp = L / dx
        n_comp = math.ceil(n_comp) // 2 * 2 + 1
        self.dx_: float = dx
        self.L: float = n_comp * dx
        super().__init__(diameters, n_comp, celsius, v_init, integrator, **kwargs)
        self.dx[:] = self.dx_

    def _x(self) -> torch.Tensor:  # x in um
        length = (self.n_comp - 1) * self.dx_
        x = torch.linspace(-length / 2, length / 2, self.n_comp, device=self.device())
        return torch.atleast_2d(x)

    def length(self) -> float:
        """
        Get the total length of the axon in micrometers.

        Returns
        -------
        float
            Total length of the axon in μm.
        """
        return self.dx.sum(axis=1)


class Myelinated(Axon):
    """
    Myelinated axon model with quadratic geometry scaling.

    Nodes of Ranvier are separated by myelinated internodes. Geometric quantities
    (axon diameter, node diameter, and internodal spacing) are derived from the
    fiber diameter via simple quadratic fits defined by the class-level
    ``GLOBAL(axon_d, node_d, delta_x)`` coefficients.

    Parameters
    ----------
    diameters : array_like
        Fiber diameters in µm; scalar or sequence broadcast to the population.
    n_node : int
        Number of nodes of Ranvier (compartments) in the model.
    node_length : float, optional
        Physical length of each node in µm used to set ``dx``. Default is 2.0.
    celsius : float, optional
        Temperature in degrees Celsius. Default is 37.
    v_init : float, optional
        Initial membrane potential in mV. Default is -80.
    integrator : Integrator, optional
        Integrator instance. If ``None``, a backward Euler integrator is used.
    **kwargs : Any
        Additional arguments forwarded to :class:`Axon`.

    Notes
    -----

    **Geometry**

    For a fiber diameter :math:`D` (µm), the derived quantities are

    .. math::

       d_\\text{axon}(D) &= a_1 D^2 + a_2 D + a_3 \\\\
       d_\\text{node}(D) &= n_1 D^2 + n_2 D + n_3 \\\\
       \\Delta x(D) &= \\delta_1 D^2 + \\delta_2 D + \\delta_3

    Coefficients are taken from ``GLOBAL(axon_d=..., node_d=..., delta_x=...)``.
    Defaults are:

    - ``axon_d``: :math:`a_1=0.0`, :math:`a_2=0.7`, :math:`a_3=0.0`
    - ``node_d``: :math:`n_1=0.0`, :math:`n_2=0.7`, :math:`n_3=0.0`
    - ``delta_x``: :math:`\\delta_1=0.0`, :math:`\\delta_2=100.0`,
      :math:`\\delta_3=0.0` (center-to-center internodal spacing)

    ``node_length`` controls the physical node extent used for per-node ``dx``,
    while :meth:`deltax` evaluates the center-to-center spacing polynomial above.
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
        """Parametrization module that scales axial resistivity."""

        def __init__(self, deltax1, deltax2, deltax3, axond1, axond2, axond3):
            super().__init__()
            self.deltax1 = deltax1
            self.deltax2 = deltax2
            self.deltax3 = deltax3
            self.axond1 = axond1
            self.axond2 = axond2
            self.axond3 = axond3

        def forward(self, rhoa, dx, diameters, diam):
            """
            Compute scaled axial resistivity parameters.

            Parameters
            ----------
            rhoa : Tensor
                Baseline axial resistivity.
            dx : Tensor
                Segment lengths in μm.
            diameters : Tensor
                Fiber diameters in μm.

            Returns
            -------
            Tensor
                Scaled axial resistivity values.
            """
            diameters = diameters.unsqueeze(1) if diameters.ndim == 1 else diameters
            axon_d = self.axond1 * diameters**2 + self.axond2 * diameters + self.axond3
            deltax = (
                self.deltax1 * diameters**2 + self.deltax2 * diameters + self.deltax3
            )
            deltax = deltax / dx
            scale = 1 / ((axon_d / diam) ** 2)
            rhoa = rhoa * scale * deltax
            return rhoa

    class myelinated_node_d(torch.nn.Module):
        """Parametrization module for node diameters."""

        def __init__(self, noded1, noded2, noded3):
            super().__init__()
            self.noded1 = noded1
            self.noded2 = noded2
            self.noded3 = noded3

        def forward(self, diam):
            """
            Compute node diameter from fiber diameter.

            Parameters
            ----------
            diam : Tensor
                Fiber diameters in μm.

            Returns
            -------
            Tensor
                Node diameters in μm.
            """
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
        **kwargs,
    ):
        self.node_length = node_length  # length of the nodes of Ranvier in um
        super().__init__(diameters, n_node, celsius, v_init, integrator, **kwargs)
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
            args=("dx", "diameters", "diam"),
        )

    def deltax(self, diameters):
        """
        Evaluate internodal spacing polynomial.

        Parameters
        ----------
        diameters : Tensor
            Fiber diameters in μm.

        Returns
        -------
        Tensor
            Internodal spacing in μm.
        """
        deltax = self.deltax1 * diameters**2 + self.deltax2 * diameters + self.deltax3
        return deltax

    def _x(self) -> torch.Tensor:  # x in um
        length = (self.n_comp - 1) * self.deltax(self.diameters).unsqueeze(1)
        start = -length / 2
        end = length / 2
        steps = self.n_comp
        t = torch.linspace(
            0, 1, steps, device=length.device, dtype=torch.double
        ).unsqueeze(0)
        return ((1 - t) * start + t * end).to(self.dtype())

    def length(self) -> torch.Tensor:
        """
        Get the total length of each axon in micrometers.

        Returns
        -------
        Tensor
            Total lengths of the axons in μm.
        """
        deltax = self.deltax(self.diameters)
        return (self.nc - 1) * deltax


# callback helpers
def pre_loop_hook(c, m):
    """
    Invoke the registered pre-loop hook on a callback list.

    Parameters
    ----------
    c : CallbackList
        Callback list to notify.
    m : Population
        Population instance being simulated.
    """
    c.pre_loop_hook(m)


def post_loop_hook(c, m):
    """
    Invoke the registered post-loop hook on a callback list.

    Parameters
    ----------
    c : CallbackList
        Callback list to notify.
    m : Population
        Population instance being simulated.
    """
    c.post_loop_hook(m)


def pre_step_hook(c, m):
    """
    Invoke the registered pre-step hook on a callback list.

    Parameters
    ----------
    c : CallbackList
        Callback list to notify.
    m : Population
        Population instance being simulated.
    """
    c.pre_step_hook(m)


def post_step_hook(c, m):
    """
    Invoke the registered post-step hook on a callback list.

    Parameters
    ----------
    c : CallbackList
        Callback list to notify.
    m : Population
        Population instance being simulated.
    """
    c.post_step_hook(m)


def pre_chunk_hook(c, m, t):
    """
    Invoke the registered pre-chunk hook on a callback list.

    Parameters
    ----------
    c : CallbackList
        Callback list to notify.
    m : Population
        Population instance being simulated.
    t : Sequence[float]
        Time values for the current chunk.
    """
    c.pre_chunk_hook(m, t)


def post_chunk_hook(c, m, t):
    """
    Invoke the registered post-chunk hook on a callback list.

    Parameters
    ----------
    c : CallbackList
        Callback list to notify.
    m : Population
        Population instance being simulated.
    t : Sequence[float]
        Time values for the current chunk.
    """
    c.post_chunk_hook(m, t)


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
    """Raised when slice unions cannot be composed into a single slice."""

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
    """
    Compile a mechanism over a set of indices with alias-specific parameters.

    Parameters
    ----------
    model : Population
        Population providing shape and batching information.
    mechanism : type
        Mechanism class to instantiate.
    indices : list
        Collection of index selectors describing placement of each alias.
    aliases : list of str
        Aliases assigned to each mechanism instance.
    kwargs_list : list of dict
        Additional keyword arguments for each aliased mechanism.

    Returns
    -------
    tuple
        Tuple ``(mechanism_instance, parameter_shape, total_index)`` ready for
        registration via :meth:`Population._register_mech`.
    """
    total_index, is_composable, shape, local_indices = compose_or_flatten_union(
        indices, model.core_shape()
    )

    shape_p = shape
    shape_f = shape

    if model.is_batched():
        n_batch_dimensions = len(model.shape) - 2
        shape_p = tuple(([1] * n_batch_dimensions) + (list(shape_p)))
        shape_f = []
        for i in range(n_batch_dimensions):
            shape_f.append(model.shape[i])
        shape_f += list(shape)
        shape_f = tuple(shape_f)

    # local indices is now a list of lists, where each sublist corresponds to the
    # local indices of the original indices in the final selection.
    # We can now use these local indices to compile the mechanism.

    additional_parameters = {}

    for alias, kwargs, idx in zip(aliases, kwargs_list, local_indices):
        for k, v in kwargs.items():
            additional_parameters.setdefault(k, []).append((alias, v, idx))

    mechanism.check_kwargs(additional_parameters)

    m = mechanism(
        None,
        model.celsius,
        model.diam,
        shape_p,
        shape_f,
        key=total_index,
        is_composable=is_composable,
        additional_parameters=additional_parameters,
    )

    return m, shape_p, total_index
