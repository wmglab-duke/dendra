"""Core data structures and utilities for AxonML population models."""

import itertools
import math
import re
from collections.abc import Iterable
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

from axonml.helpers import (
    BACKEND,
    COMPILE_MODE,
    DYNAMIC,
    FULLGRAPH,
    IMEM,
    JIT,
    op_mc,
    op_sc,
)
from axonml.models.backend import Backend as A
from axonml.models.callbacks import Callback, CallbackList
from axonml.models.graph import get_area_from_graph
from axonml.models.integrators import bwd_euler_sc, bwd_euler_ub
from axonml.models.mechanisms._handler import MechanismHandler
from axonml.models.mechanisms._ions import Ion, concentrations, equilibria, valid_ions
from axonml.models.mechanisms.validate import validate
from axonml.models.parametric import Parameterized as P
from axonml.models.stim.intra import Intra
from axonml.models.stim.waveform import Waveform
from axonml.units import mm

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


def matches_any_pattern(base_patterns: Iterable[str], target_string: str) -> bool:
    """
    Check whether a target string matches any dotted base pattern.

    A match occurs when each token in a pattern appears in order in the target
    string. All tokens except the final one must match entire words; the final
    token may match a word prefix.

    Additionally, the '*' character inside a pattern token is treated as a
    wildcard matching any sequence of characters (including empty).

    Parameters
    ----------
    base_patterns : Iterable[str]
        Collection of dot-separated pattern strings to test. Tokens may
        contain '*' as a wildcard.
    target_string : str
        Candidate string evaluated against each pattern.

    Returns
    -------
    bool
        True if any pattern matches the target string, False otherwise.

    Examples
    --------
    >>> matches_any_pattern(['hh.gbar'], 'hh.gbar_default')
    True
    >>> matches_any_pattern(['foo'], 'a.foo_bar')
    True
    >>> matches_any_pattern(['a.b'], 'a_b.c')
    False
    >>> matches_any_pattern(['*aug'], 'aug_default')
    True
    >>> matches_any_pattern(['*aug'], 'raug_default')
    True
    >>> matches_any_pattern(['*aug'], 'ina_aug_default')
    True
    """

    def _pattern_part_to_regex(part: str) -> str:
        # Escape everything, then turn escaped '*' (r'\*') back into '.*'
        escaped = re.escape(part)
        return escaped.replace(r"\*", ".*")

    for base_pattern in base_patterns:
        # Split the pattern by '.' and convert each part, treating '*' as wildcard.
        regex_parts = [_pattern_part_to_regex(part) for part in base_pattern.split(".")]

        # The separator `\b.*?\b` ensures that all intermediate parts are
        # treated as whole words.
        regex_pattern = (
            r"\b"  # The pattern must start at a word boundary.
            + r"\b.*?\b".join(regex_parts)
            # No trailing \b so the final token may match a word prefix.
        )

        if re.search(regex_pattern, target_string, re.IGNORECASE):
            return True

    return False


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

    P.RANGE(cm=1.0, rhoa=35.4)
    P.GLOBAL(celsius=37.0)

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

        self.register_buffer("v", torch.full((N, C), self.v_init))
        self.register_buffer("diam", torch.full(self.shape, 500.0))
        self.register_buffer("dx", torch.full(self.shape, 100.0))
        self.register_buffer("t", torch.zeros(()))

        # compiler stuff
        self.backend = BACKEND.value
        self.fullgraph = bool(FULLGRAPH)
        self.dynamic = bool(DYNAMIC)
        self.jit = bool(JIT)
        self.imem = bool(IMEM)
        self.compile_mode = COMPILE_MODE.value

        if self.imem:
            self.register_buffer("i_membrane", torch.zeros(self.shape))
        else:
            self.i_membrane = None  # type: ignore

        self._integrator_class = integrator
        self.integrator = None  # type: ignore

        self.injections = []
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

        self._all_read = {}
        self._all_write = {}
        self._all_write_c = {}

        self._equilibria = {}
        self._concentrations = {}

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
            self._step = step

        if self.jit:
            self.make_intra = torch.compile(make_intra)
        else:
            self.make_intra = make_intra

        self._caches = {}

        self.register_buffer("x", torch.zeros(self.shape))
        self.register_buffer("y", torch.zeros(self.shape))
        self.register_buffer("z", torch.zeros(self.shape))

        self.mech: MechanismHandler = None  # type: ignore

        self.initialized: bool = False
        self.eval()

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
        This is a placeholder for future area-related functionality.
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
            Keyword arguments forwarded to ``axonml.models.mechanisms._ions.equilibria``.
        """
        self._equilibria.update(kwargs)

    def concentrations(self, **kwargs):
        """
        Register ionic concentration configuration.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to ``axonml.models.mechanisms._ions.concentrations``.
        """
        self._concentrations.update(kwargs)

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
        """
        In-place alias of :meth:`unfreeze`.

        Parameters
        ----------
        *names : str
            Optional name patterns forwarded to :meth:`unfreeze`.
        """
        self.unfreeze(*names)

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

    def freeze(self, *names):
        """
        Freeze parameters to disable gradient computation.

        Parameters
        ----------
        *names : str
            Optional name patterns selecting parameters to freeze. When omitted,
            all parameters are frozen.

        Returns
        -------
        Population
            The population instance for chaining.
        """
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
        """
        In-place alias of :meth:`freeze`.

        Parameters
        ----------
        *names : str
            Optional name patterns forwarded to :meth:`freeze`.
        """
        self.freeze(*names)

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
            self.integrator._initialize(self, dt_tensor, force=self.training)

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

                post_step_hook(callbacks, self)
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

        intra = self.intra
        with_intra = intra is not None

        # dt scalars
        dt = dt if dt is not None else A.dt
        dt_f = float(dt)
        dt_tensor = torch.tensor(dt, device=self.device(), dtype=self.dtype())

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

                self.integrator._initialize(self, dt_tensor, force=self.training)

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
                        post_step_hook(callbacks, self)
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

        self.clear_steady_state()

        self.initialize()
        self.integrator._initialize(self, dt)

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
        self.t.detach().zero_()
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
        In-place alias of :meth:`populate`.
        """
        self.populate()

    def _restore_steady_state(self):
        if "_steady_state" in self._caches:
            self.restore("_steady_state")
            self.post_initialize()
            self.t.detach().zero_()
            self.initialized = True
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
        self.t = self.t.zero_().detach()
        self.initialized = True
        return self

    def initialize_(self):
        """
        In-place alias of :meth:`initialize`.
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

    def delete_injections(self):
        """
        Remove all registered intra-cellular injections.

        Returns
        -------
        None
        """
        self.injections = []
        self.intra = None

    def build_intra(self):
        """
        Build the intra-cellular stimulation handler.

        Returns
        -------
        Intra or None
            Intra stimulus object when injections are configured, otherwise None.
        """
        if self.injections:
            return Intra(self, self.injections)
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

    def ion_style(self, ion, c_style, e_style, einit, eadvance, cinit):
        """
        Register explicit ion handling style parameters.

        Parameters
        ----------
        ion : str
            Ion species identifier (e.g., ``'na'``).
        c_style : int
            Style flag for concentration handling.
        e_style : int
            Style flag for reversal potential handling.
        einit : int
            Initialization flag for equilibration.
        eadvance : int
            Advance-time flag for equilibration updates.
        cinit : int
            Initialization flag for concentration updates.
        """
        assert ion in valid_ions(), f"Invalid ion: {ion}"
        self._ion_style[ion] = (c_style, e_style, einit, eadvance, cinit)

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
        return bool(d)

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
        if not d:
            return False
        check = list(itertools.chain(*d.values()))
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
        if not d:
            return False
        return f"e{ion}" in list(itertools.chain(*d.values()))

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
            Tuple of style flags ``(c_style, e_style, einit, eadvance, cinit)``.
        """
        _c_is_written = self._c_is_written(ion)
        _c_is_read = self._c_is_read(ion)
        _e_is_read = self._e_is_read(ion)

        if _c_is_written:
            if _e_is_read:
                return (3, 2, 1, 1, 1)
            return (3, 0, 0, 0, 1)
        if _c_is_read:
            if _e_is_read:
                return (1, 2, 1, 0, 0)
            return (1, 0, 0, 0, 0)
        if _e_is_read:
            return (0, 1, 0, 0, 0)
        return (0, 0, 0, 0, 0)

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

        for k, v in mech._currents.items():
            self._m_curr.setdefault(k, {}).update({name: v})

        for k, v in mech._read_ion.items():
            self._ion_read.setdefault(k, {}).update({name: v})

        for k, v in mech._write_ion.items():
            self._ion_write.setdefault(k, {}).update({name: v})

        for k, v in mech._write_ion_c.items():
            self._ion_write_c.setdefault(k, {}).update({name: v})

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

            all_ions = get_unique_keys(
                [self._ion_read, self._ion_write, self._ion_write_c]
            )

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

            self.integrator = self._integrator_class(self, mech, imem=self.imem)
            self.mech = self.integrator.mech

        self.is_built = True
        self._flag_rebuild = False
        self.to(device=self.device(), dtype=self.dtype())
        self.eval()
        return self

    def build_(self, force_rebuild=False):
        """
        In-place alias of :meth:`build`.

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
        In-place alias of :meth:`detach`.
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
        self.v = self.v.unsqueeze(0).expand(n, *self.v.shape).clone()
        if hasattr(self, "v_prev"):
            self.v_prev = self.v_prev.unsqueeze(0).expand(n, *self.v_prev.shape).clone()
        if hasattr(self, "i_membrane"):
            self.i_membrane = (
                self.i_membrane.unsqueeze(0).expand(n, *self.i_membrane.shape).clone()
            )
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
            G = nx.path_graph(self.n_comp).to_directed()
            for node in G.nodes:
                G.nodes[node]["name"] = f"axon[{i}]({node / (self.n_comp - 1):.2f})"
                G.nodes[node]["x"] = self.x[i, node].item()
                G.nodes[node]["y"] = 0.0
                G.nodes[node]["z"] = 0.0
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
        """Parametrization module that scales axial resistivity."""

        def __init__(self, deltax1, deltax2, deltax3, axond1, axond2, axond3):
            super().__init__()
            self.deltax1 = deltax1
            self.deltax2 = deltax2
            self.deltax3 = deltax3
            self.axond1 = axond1
            self.axond2 = axond2
            self.axond3 = axond3

        def forward(self, rhoa, dx, diameters):
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
            scale = 1 / ((axon_d / diameters) ** 2)
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
        t = torch.linspace(0, 1, steps, device=length.device).unsqueeze(0)
        return (1 - t) * start + t * end


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


@torch.compile
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
