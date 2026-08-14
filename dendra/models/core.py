"""Core data structures and utilities for Dendra population models."""

import copy
import hashlib
import itertools
import math
import os
import re
import sys
import textwrap
from contextlib import nullcontext
from dataclasses import dataclass
from fractions import Fraction
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
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

from dendra._bootstrap import (
    reset_torch_compiler,
    torch_compiler_warning_context,
)
from dendra.helpers import (
    BACKEND,
    COMPILE_MODE,
    DYNAMIC,
    FULLGRAPH,
    IMEM,
    JIT,
    JIT_NETWORK_OPS,
    JIT_NETWORK_SOLVES,
    _normalize_dtype_value,
    compile_options_key,
    current_compile_options,
    current_device,
    current_dtype,
    current_runtime_contract_validation,
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
    _canonical_material_name,
    material_specs,
    valid_materials,
)
from dendra.models.mechanisms.validate import validate
from dendra.models.modular import matches_any_pattern
from dendra.models.parametric import Parameterized as P
from dendra.models.parametric import create_param_expander
from dendra.models.rng import _validate_rng_checkpoint_payload
from dendra.models.stim.intra import Intra
from dendra.models.stim.waveform import Waveform
from dendra.units import mm

from .slice import IndexSpec, Slice, Sliceable, parse_key

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
    # [*batch, np, n_comp] or [n_contacts, *batch, np, n_comp]
    ve_s: Optional[torch.Tensor] = None
    einsum: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None

    # If functional: list of Waveform (one per contact)
    waveforms: Optional[List["Waveform"]] = None

    # Single contact: list of [*batch, np, n_t_chunk] tensors.
    time_chunks_single: Optional[List[torch.Tensor]] = None

    # If non-functional, multi-contact:
    # list over contacts, each is list over chunks -> [*batch, np, n_t_chunk]
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


def _index_to_cpu(index):
    """Clone structural indexing metadata onto CPU-owned storage."""
    if isinstance(index, torch.Tensor):
        return index.detach().to(device="cpu").clone()
    if isinstance(index, np.ndarray):
        return index.copy()
    if isinstance(index, tuple):
        return tuple(_index_to_cpu(item) for item in index)
    if isinstance(index, list):
        return [_index_to_cpu(item) for item in index]
    return index


def _core_flat_indices(index, shape):
    """Materialize a structural selector as physical core-flat indices."""
    if isinstance(index, IndexSpec):
        index = index.index
    index = _index_to_cpu(index)
    grid = torch.arange(math.prod(shape), dtype=torch.long).reshape(shape)
    return grid[index].reshape(-1)


def _stable_unique_long(values):
    """Return first-occurrence unique values as a CPU LongTensor."""
    values = torch.as_tensor(values, dtype=torch.long, device="cpu").reshape(-1)
    if values.numel() < 2:
        return values.clone()
    seen = set()
    positions = []
    for position, value in enumerate(values.tolist()):
        if value not in seen:
            seen.add(value)
            positions.append(position)
    return values.index_select(0, torch.as_tensor(positions, dtype=torch.long))


def _sorted_unique_long(values):
    """Return sorted unique values as a CPU LongTensor."""
    values = torch.as_tensor(values, dtype=torch.long, device="cpu").reshape(-1)
    if values.numel() == 0:
        return values.clone()
    return torch.unique(values, sorted=True)


def _core_key_from_flat(values, shape):
    """Encode physical flat indices as a dimension-correct advanced key."""
    values = torch.as_tensor(values, dtype=torch.long, device="cpu").reshape(-1)
    return tuple(coord.clone() for coord in torch.unravel_index(values, shape))


def _unpack_mechanism_insertion_record(record):
    """Normalize legacy and current sparse mechanism insertion records."""
    if len(record) == 3:
        alias, kwargs, key = record
        preserve = False
        copies = 1
    elif len(record) == 4:
        alias, kwargs, key, preserve = record
        copies = 1
    elif len(record) == 5:
        alias, kwargs, key, preserve, copies = record
    else:
        raise ValueError(
            "Mechanism insertion records must contain either "
            "(alias, kwargs, key), "
            "(alias, kwargs, key, preserve_duplicate_indices), "
            "or (alias, kwargs, key, preserve_duplicate_indices, copies)."
        )
    copies = int(copies)
    return alias, kwargs, key, bool(preserve or copies != 1), copies


def _configuration_values_equal(left, right):
    """Compare nested initial-condition configuration without tensor ambiguity."""
    if left is right:
        return True
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            _configuration_values_equal(left[key], right[key]) for key in left
        )
    if torch.is_tensor(left) or torch.is_tensor(right):
        try:
            return torch.equal(torch.as_tensor(left), torch.as_tensor(right))
        except (TypeError, ValueError, RuntimeError):
            return False
    try:
        result = left == right
    except Exception:
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _global_configuration_values_equal(left, right):
    """Compare GLOBAL values after scalar numeric representation differences."""
    if _configuration_values_equal(left, right):
        return True
    try:
        left_tensor = torch.as_tensor(left).detach()
        right_tensor = torch.as_tensor(right).detach()
    except (TypeError, ValueError, RuntimeError):
        return False
    if left_tensor.numel() != 1 or right_tensor.numel() != 1:
        return False
    if left_tensor.dtype == torch.bool or right_tensor.dtype == torch.bool:
        return False

    numeric_kinds = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    }
    if (
        left_tensor.dtype not in numeric_kinds
        or right_tensor.dtype not in numeric_kinds
    ):
        return False
    if not (
        left_tensor.is_floating_point() or right_tensor.is_floating_point()
    ) and not (left_tensor.is_complex() or right_tensor.is_complex()):
        return bool(left_tensor.item() == right_tensor.item())

    def explicitly_typed(value):
        return torch.is_tensor(value) or isinstance(value, (np.ndarray, np.generic))

    typed_float_dtypes = [
        tensor.dtype
        for value, tensor in ((left, left_tensor), (right, right_tensor))
        if explicitly_typed(value)
        and (tensor.is_floating_point() or tensor.is_complex())
    ]
    if typed_float_dtypes:
        for dtype in set(typed_float_dtypes):
            if dtype not in (torch.complex64, torch.complex128) and (
                left_tensor.is_complex() or right_tensor.is_complex()
            ):
                return False
            try:
                left_comparable = torch.as_tensor(
                    left, device="cpu", dtype=dtype
                ).detach()
                right_comparable = torch.as_tensor(
                    right, device="cpu", dtype=dtype
                ).detach()
            except (TypeError, ValueError, RuntimeError):
                return False
            if not torch.equal(left_comparable, right_comparable):
                return False
        return True
    return bool(left_tensor.item() == right_tensor.item())


def _mechanism_spatial_parameter_names(mechanism):
    """Collect spatial RANGE names declared by a mechanism and its states."""
    names = set()
    for owner in (mechanism, *tuple(getattr(mechanism, "_state", ()))):
        for declaration in ("_range", "_range_p", "_range_n"):
            names.update(getattr(owner, declaration, {}).keys())
    return names


def _mechanism_batch_parameter_names(mechanism):
    """Collect BATCH names declared by a mechanism and its states."""
    names = set()
    for owner in (mechanism, *tuple(getattr(mechanism, "_state", ()))):
        for declaration in ("_batch", "_batch_p", "_batch_n"):
            names.update(getattr(owner, declaration, {}).keys())
    return names


def _mechanism_global_parameter_names(mechanism):
    """Collect GLOBAL parameter names declared by a mechanism and its states."""
    names = set()
    for owner in (mechanism, *tuple(getattr(mechanism, "_state", ()))):
        for declaration in ("_global", "_global_p", "_global_n"):
            names.update(getattr(owner, declaration, {}).keys())
    return names


def _mechanism_initial_defaults(mechanism):
    """Return class-declared mechanism state initial values."""
    return dict(getattr(mechanism, "_init", {}))


def _project_indexed_override(
    value,
    old_core_indices,
    keep_mask,
    core_shape,
    *,
    context,
    logical_shape=None,
):
    """Project an indexed override onto its surviving physical support."""
    old_core_indices = torch.as_tensor(
        old_core_indices, dtype=torch.long, device="cpu"
    ).reshape(-1)
    keep_mask = torch.as_tensor(keep_mask, dtype=torch.bool, device="cpu").reshape(-1)
    if keep_mask.numel() != old_core_indices.numel():
        raise RuntimeError(
            f"{context}: internal override support and retention mask disagree."
        )
    if bool(torch.all(keep_mask)):
        return value

    if isinstance(value, torch.nn.Module):
        raise ValueError(
            f"{context} uses a module-valued override whose support cannot be "
            "projected safely. Remove the whole override region or replace it "
            "with a scalar/Tensor override before deleting compartments."
        )

    if isinstance(value, (bool, int, float, complex, np.number)):
        return value

    try:
        tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} has unsupported value type {type(value).__name__}; "
            "Dendra cannot safely project it after a partial deletion."
        ) from exc

    if tensor.ndim == 0:
        return value

    key = old_core_indices.to(device=tensor.device)
    try:
        expand = create_param_expander(
            tensor,
            key,
            tuple(core_shape),
            logical_shape=logical_shape,
            context=context,
        )
        expanded = expand(tensor).reshape(-1)
    except (IndexError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"{context} with shape {tuple(tensor.shape)} cannot be projected "
            "safely onto the surviving mechanism support."
        ) from exc

    projected = expanded.index_select(
        0,
        torch.nonzero(keep_mask, as_tuple=False).reshape(-1).to(device=expanded.device),
    ).clone()
    if isinstance(value, torch.nn.Parameter):
        return torch.nn.Parameter(
            projected.detach(), requires_grad=bool(value.requires_grad)
        )
    return projected


def _validate_deletion_stable_batch_override(value, *, context):
    """Reject BATCH layouts whose row association cannot be preserved safely."""
    if isinstance(value, torch.nn.Module):
        raise ValueError(
            f"{context} uses a module-valued BATCH override. Dendra cannot "
            "prove that its row association survives a structural deletion."
        )
    if isinstance(value, (bool, int, float, complex, np.number)):
        return
    try:
        tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} has unsupported BATCH override type {type(value).__name__}."
        ) from exc
    if tensor.numel() == 1:
        return
    raise ValueError(
        f"{context} is a non-scalar BATCH override. A deletion can change the "
        "compiled mechanism's row groups, so Dendra cannot preserve this value "
        "unambiguously. Use a scalar BATCH value, delete the whole mechanism, "
        "or rebuild the final placement explicitly."
    )


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


def _clone_nested_state(value, *, detach_tensors=False, memo=None):
    """Clone tensors and mutable containers in a nested runtime-state payload.

    Tensor identity aliases are preserved.  ``detach_tensors`` is intended for
    rollback snapshots; checkpoint boundary clones keep their autograd history.
    """
    if memo is None:
        memo = {}
    if torch.is_tensor(value):
        key = id(value)
        if key not in memo:
            tensor = value.detach() if detach_tensors else value
            memo[key] = tensor.clone()
        return memo[key]
    if isinstance(value, Mapping):
        items = [
            (key, _clone_nested_state(item, detach_tensors=detach_tensors, memo=memo))
            for key, item in value.items()
        ]
        try:
            cloned = value.__class__(items)
        except TypeError:
            cloned = dict(items)
        if hasattr(value, "_metadata"):
            cloned._metadata = copy.deepcopy(value._metadata)
        return cloned
    if isinstance(value, list):
        return [
            _clone_nested_state(item, detach_tensors=detach_tensors, memo=memo)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _clone_nested_state(item, detach_tensors=detach_tensors, memo=memo)
            for item in value
        )
    return copy.deepcopy(value)


def _validate_time_scalar(value, *, name: str, positive: bool | None) -> float:
    """Normalize one finite scalar simulation-time argument."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar, not a boolean.")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real scalar.") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    if positive is True and normalized <= 0:
        raise ValueError(f"{name} must be positive.")
    if positive is False and normalized < 0:
        raise ValueError(f"{name} must be non-negative.")
    return normalized


def _duration_step_budget(
    duration: float, dt: float, pending: float = 0.0
) -> tuple[int, float]:
    """Resolve complete fixed steps and retained physical time exactly.

    ``duration`` and ``pending`` are physical milliseconds requested by the
    caller but not necessarily representable as a complete fixed step.  Decimal
    control arithmetic avoids both binary floating-point boundary promotion and
    the loss of nominal decimal identities such as ``0.15 + 0.15 == 0.3``.

    Parameters
    ----------
    duration : float
        New non-negative duration requested by this call.
    dt : float
        Positive fixed timestep.
    pending : float, optional
        Non-negative physical duration retained from earlier calls.

    Returns
    -------
    tuple[int, float]
        Number of complete steps and the remaining unsimulated milliseconds.
    """
    duration = float(duration)
    dt = float(dt)
    pending = float(pending)
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("duration must be finite and non-negative.")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("timestep must be finite and positive.")
    if not math.isfinite(pending) or pending < 0.0:
        raise ValueError("pending duration must be finite and non-negative.")

    # Programmatically constructed exact multiples can carry a harmless decimal
    # tail (for example ``157 * 0.025 == 3.9250000000000003``) even though binary
    # division recovers the intended integer exactly. Normalize that case before
    # decimal control arithmetic, while requiring reconstruction to preserve
    # genuinely tiny nonzero durations whose quotient underflows to zero.
    try:
        binary_total = math.fsum((pending, duration))
    except OverflowError:
        binary_total = math.inf
    binary_quotient = binary_total / dt if math.isfinite(binary_total) else math.inf
    normalized_steps = None
    if math.isfinite(binary_quotient):
        if binary_quotient.is_integer() and binary_quotient * dt == binary_total:
            normalized_steps = int(binary_quotient)
        elif Fraction.from_float(dt) != Fraction(str(dt)):
            # For decimal timesteps such as 0.1, one-ULP differences around an
            # integer cannot reliably distinguish input roundoff from a genuine
            # boundary. Preserve the established decimal-grid interpretation.
            nearest = round(binary_quotient)
            if nearest >= 1 and abs(binary_quotient - nearest) <= 8.0 * math.ulp(
                binary_quotient
            ):
                normalized_steps = int(nearest)
    if normalized_steps is not None:
        n_steps = normalized_steps
        if n_steps > sys.maxsize:
            raise ValueError("duration and dt imply too many simulation steps.")
        return n_steps, 0.0

    total = Fraction(str(pending)) + Fraction(str(duration))
    dt_exact = Fraction(str(dt))
    n_steps = total // dt_exact
    if n_steps > sys.maxsize:
        raise ValueError("duration and dt imply too many simulation steps.")
    remainder = total - n_steps * dt_exact
    return int(n_steps), float(remainder)


def _duration_step_count(duration: float, dt: float) -> int:
    """Return complete fixed timesteps in one duration without prior carry."""
    return _duration_step_budget(duration, dt, 0.0)[0]


def _time_grid_from_step_count(
    start: torch.Tensor,
    n_steps: int,
    dt: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Construct an exact-length float64 time grid from integer offsets."""
    offsets = torch.arange(n_steps, device=device, dtype=torch.double)
    return start.to(device=device, dtype=torch.double) + offsets * dt


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

    def __init__(
        self,
        N: int = 1,
        C: int = 1,
        integrator=None,
        v_init=-65.0,
        *,
        device=None,
        dtype=None,
        **kwargs,
    ):
        init_device = (
            current_device(torch.device("cpu"))
            if device is None
            else torch.device(device)
        )
        init_dtype = (
            current_dtype(torch.float32)
            if dtype is None
            else _normalize_dtype_value(dtype)
        )
        super().__init__((N, C), (N, C), device=init_device, dtype=init_dtype, **kwargs)
        Sliceable.__init__(self)
        self.np = N
        self.nc = C
        self.v_init = v_init

        self.is_built = False
        self._flag_rebuild = False
        self.key = None

        if integrator is None:
            integrator = bwd_euler_sc()

        self.register_buffer(
            "_dummy", torch.zeros(1, device=init_device, dtype=init_dtype)
        )

        # Initialize voltage from scalar v_init or from a vector of length nc.
        # The latter is useful for point-neuron populations represented as a
        # single Dendra population with one compartment per modeled neuron.
        self.register_buffer("v", self.expanded_v_init((N, C)).clone().contiguous())
        self.register_buffer(
            "diam", torch.full(self.shape, 500.0, device=init_device, dtype=init_dtype)
        )
        self.register_buffer(
            "dx", torch.full(self.shape, 100.0, device=init_device, dtype=init_dtype)
        )
        self.register_buffer("t", torch.zeros((), device=init_device, dtype=init_dtype))
        # Physical milliseconds requested by duration-based execution calls but
        # not yet sufficient to form a complete fixed step.  Keep this separate
        # from model time: explicit step() and array-driven run(ve=...) advance
        # state without consuming the duration budget.
        self.register_buffer(
            "_duration_remainder",
            torch.zeros((), device=init_device, dtype=torch.float64),
        )

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
        self.compile_options = current_compile_options()
        self.compile_options_key = compile_options_key(self.compile_options)
        self.initializing_from_state_cache = False

        if self.imem:
            self.register_buffer(
                "i_membrane",
                torch.zeros(self.shape, device=init_device, dtype=init_dtype),
            )
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
        # Everywhere insertions can be cropped without losing their class-wide
        # constructor kwargs or initial conditions.  Values are physical
        # population-core flat indices excluded from the dense insertion.
        self._mech_exclusions = {}
        # Initial conditions are class-wide for a compiled sparse mechanism:
        # one mechanism instance is built from the union of all region records.
        self._mech_data_ic = {}
        # GLOBAL parameters are likewise class-wide. Region records hold only
        # indexed RANGE/BATCH overrides; conflicting GLOBAL values cannot be
        # represented by one compiled mechanism instance.
        self._mech_data_base_kwargs = {}
        # Slice-scoped overrides applied to a compiled mechanism must remain
        # structural configuration rather than state owned only by that
        # disposable compiled instance.  Records use population-core indices
        # so they can be replayed after batching and forced rebuilds.
        self._slice_mechanism_parametrizations = []

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
        # Named finite-volume geometries for chemical/material transport.  The
        # registry stores buffer names rather than tensor objects so ordinary
        # Module dtype/device moves cannot leave stale Python references behind.
        self._material_geometries = {}

        self._all_read = {}
        self._all_write = {}
        self._all_write_c = {}

        self._equilibria = {}
        self._concentrations = {}

        self._ion_style = {}

        self.pre_initialize_hooks: List[Callable] = []
        self.post_initialize_hooks: List[Callable] = []

        reset_torch_compiler(prefer_public=False)

        # Keep the state-mutating Population/Integrator wrapper eager.  JIT
        # settings are propagated to the integrator, which compiles only its
        # tensor-valued numerical kernels.  This avoids Dynamo tracing
        # nn.Module.__setattr__ for model.v/model.vc/model.i_membrane commits.
        self._step = step

        self._make_intra_config = None
        self._refresh_compile_config_from_ctx()

        self._caches = {}

        self.register_buffer(
            "x", torch.zeros(self.shape, device=init_device, dtype=init_dtype)
        )
        self.register_buffer(
            "y", torch.zeros(self.shape, device=init_device, dtype=init_dtype)
        )
        self.register_buffer(
            "z", torch.zeros(self.shape, device=init_device, dtype=init_dtype)
        )

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

    def _runtime_workspace_rebuild_pending(self):
        """Whether direct execution will refresh all tracked workspace inputs."""
        return self.force_integrator_reinit()

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
        self.compile_options = current_compile_options()
        self.compile_options_key = compile_options_key(self.compile_options)

        make_intra_config = (
            self.jit,
            self.backend,
            self.fullgraph,
            self.dynamic,
            self.compile_mode,
            self.compile_options_key,
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
                if self.compile_options is not None:
                    kwargs["options"] = dict(self.compile_options)
                with torch_compiler_warning_context():
                    self.make_intra = torch.compile(make_intra, **kwargs)
            else:
                self.make_intra = make_intra
            self._make_intra_config = make_intra_config
        return self

    def _call_make_intra(self, intra, stims, indices):
        with torch_compiler_warning_context():
            return self.make_intra(intra, stims, indices)

    def clear_jit_cache(self):
        """Drop lazily compiled functions attached to this population.

        ``torch.compile`` callables are process-local and may capture
        TorchDynamo/Inductor configuration objects that cannot be pickled.  The
        compiled helpers are regenerated lazily by ``run``/``initialize`` after
        unpickling, so clearing them does not discard model state.
        """
        # ``make_intra`` may be a torch.compile wrapper.  Reset to the top-level
        # function and invalidate the config sentinel so the next context refresh
        # can recompile it when JIT is enabled.
        self.make_intra = make_intra
        self._make_intra_config = None

        integrator = getattr(self, "integrator", None)
        if integrator is not None:
            clear = getattr(integrator, "clear_jit_cache", None)
            if clear is not None:
                clear()
            elif hasattr(integrator, "_compiled_kernels"):
                integrator._compiled_kernels.clear()
        return self

    def pickleable(
        self,
        *,
        inplace: bool = False,
        clone: bool = False,
        reset_global_compiler: bool = False,
    ):
        """Return a pickle-friendly population handle.

        By default this method is non-mutating and simply returns ``self``.
        Pickling then uses :meth:`__getstate__`, which strips process-local JIT
        callables from the serialized state without clearing the live object's
        compiled caches.  This lets users checkpoint a running model and keep
        using the already-compiled kernels afterward.

        Parameters
        ----------
        inplace : bool, default False
            If True, also clear this live population's local JIT caches.  The
            next JIT-enabled run may need to recompile.
        clone : bool, default False
            If True, return a sanitized deep copy.  This avoids mutating the
            live population but duplicates tensor storage, so it is usually not
            appropriate for very large models.
        reset_global_compiler : bool, default False
            Also clear global Torch compiler caches.  This can force recompiles
            and should usually be False when saving mid-simulation.
        """
        if inplace and clone:
            raise ValueError(
                "pickleable(...): choose at most one of inplace=True or clone=True."
            )
        if clone:
            import copy as _copy

            obj = _copy.deepcopy(self)
            obj.clear_jit_cache()
        elif inplace:
            obj = self.clear_jit_cache()
        else:
            obj = self
        if reset_global_compiler:
            reset_torch_compiler()
        return obj

    def __deepcopy__(self, memo):
        """Copy runtime state while deliberately severing live autograd graphs.

        PyTorch supports deepcopy only for leaf tensors. Training-mode material
        state and differentiable spatial-operator caches are valid non-leaf
        buffers, so seed the copy memo with detached value copies. Parameters
        remain independently copied leaves. Any spatial process whose cached
        coefficients were detached is marked for a lazy rebuild from the
        cloned Parameters before its next transport step.
        """
        import copy as _copy

        existing = memo.get(id(self))
        if existing is not None:
            return existing

        detached_runtime = False
        for module in self.modules():
            for value in module._buffers.values():
                if torch.is_tensor(value) and not value.is_leaf:
                    if id(value) not in memo:
                        memo[id(value)] = value.detach().clone()
                    detached_runtime = True

        result = self.__class__.__new__(self.__class__)
        memo[id(self)] = result
        state = _copy.deepcopy(self.__getstate__(), memo)
        result.__setstate__(state)

        if detached_runtime and getattr(result, "is_built", False):
            for process in getattr(
                getattr(result, "mech", None), "material_processes", {}
            ).values():
                if hasattr(process, "_spatial_configured"):
                    process._spatial_configured = False
        return result

    def __getstate__(self):
        """Serialize without process-local compiled helpers.

        This makes ``pickle.dump(population, f)`` work even after a JIT-enabled
        run.  The live object is not modified; only the serialized state is
        sanitized.
        """
        state = self.__dict__.copy()
        state["make_intra"] = make_intra
        state["_make_intra_config"] = None
        return state

    def __setstate__(self, state):
        """Restore populations serialized before mechanism deletion support."""
        state.setdefault("_mech_exclusions", {})
        state.setdefault("_mech_data_ic", {})
        state.setdefault("_mech_data_base_kwargs", {})
        state.setdefault("_material_geometries", {})

        # Sparse insertion records historically stored GLOBAL and indexed
        # parameters together. GLOBAL values now belong to the one compiled
        # mechanism instance, so migrate old records before either direct build
        # or MultiPopulation reinsertion can reinterpret them as spatial data.
        migrated_data = {}
        base_kwargs_by_mechanism = {
            mechanism: dict(values)
            for mechanism, values in state["_mech_data_base_kwargs"].items()
        }
        for mechanism, records in state.get("_mech_data", {}).items():
            global_names = _mechanism_global_parameter_names(mechanism)
            base_kwargs = base_kwargs_by_mechanism.setdefault(mechanism, {})
            migrated_records = []
            for record in records:
                alias, kwargs, key, preserve, copies = (
                    _unpack_mechanism_insertion_record(record)
                )
                indexed_kwargs = dict(kwargs)
                for name in indexed_kwargs.keys() & global_names:
                    value = indexed_kwargs.pop(name)
                    if name in base_kwargs and not _global_configuration_values_equal(
                        base_kwargs[name], value
                    ):
                        raise ValueError(
                            "Cannot restore serialized Population: sparse "
                            f"mechanism {mechanism.__name__!r} contains "
                            f"conflicting class-wide GLOBAL values for {name!r}."
                        )
                    base_kwargs[name] = value
                migrated_records.append(
                    (
                        alias,
                        indexed_kwargs,
                        _index_to_cpu(key),
                        preserve,
                        copies,
                    )
                )
            migrated_data[mechanism] = migrated_records
            if not base_kwargs:
                base_kwargs_by_mechanism.pop(mechanism, None)
        state["_mech_data"] = migrated_data
        state["_mech_data_base_kwargs"] = base_kwargs_by_mechanism
        super().__setstate__(state)

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

    def _validate_static_runtime_contracts(self, mode):
        """Validate subclass-specific static contracts before public execution."""
        return None

    def _integrator_workspace_contract_signature(self):
        """Return metadata for inputs captured by this model's solver workspace."""
        return None

    def _record_integrator_workspace_contracts(self):
        """Record solver inputs only after a workspace was built successfully."""
        signature = self._integrator_workspace_contract_signature()
        if signature is not None:
            self._validated_integrator_workspace_signature = signature
            self._record_unversioned_integrator_workspace_values(signature)

    def _record_unversioned_integrator_workspace_values(self, signature):
        """Optionally snapshot dependencies whose tensors expose no versions."""
        return None

    def _unversioned_integrator_workspace_values_match(self, signature=None):
        """Return whether any required unversioned dependency snapshots match."""
        return True

    def _validate_integrator_workspace_contracts(self, *, rebuild_pending=False):
        """Reject execution with a stale, already-built solver workspace."""
        signature = self._integrator_workspace_contract_signature()
        if signature is None:
            return
        integrator = getattr(self, "integrator", None)
        if integrator is None or not integrator.initialized:
            # Population stepping will build the workspace before using it.
            return
        expected = getattr(self, "_validated_integrator_workspace_signature", None)
        if (
            expected != signature
            or not self._unversioned_integrator_workspace_values_match(signature)
        ):
            if rebuild_pending:
                # Direct Population training rebuilds its integrator before the
                # next numerical step. Network execution passes False because
                # it cannot refresh one population in isolation.
                return
            raise RuntimeError(
                f"{type(self).__name__} solver-affecting inputs changed after "
                "its integrator workspace was built. Reinitialize the model "
                "before continuing (or reinitialize its owning Network or "
                "MultiPopulation). Use initialize(populate_parameter_buffers=False) "
                "when an intentional direct buffer override must be retained."
            )

    def _validate_runtime_contracts(self, *, workspace_rebuild_pending=False):
        """Validate static and cached-workspace contracts before execution."""
        mode = current_runtime_contract_validation()
        if mode == "initialize":
            return
        self._validate_static_runtime_contracts(mode)
        self._validate_integrator_workspace_contracts(
            rebuild_pending=workspace_rebuild_pending
        )

    def _validate_integrator_rebuild_contracts(self):
        """Validate static contracts immediately before solver workspace rebuilds."""
        return None

    def _apply(self, fn, recurse=True):
        """Apply dtype/device transforms and invalidate derived solver workspaces."""
        voltage = getattr(self, "v", None)
        before = (voltage.device, voltage.dtype) if torch.is_tensor(voltage) else None
        result = super()._apply(fn, recurse=recurse)
        voltage = getattr(self, "v", None)
        after = (voltage.device, voltage.dtype) if torch.is_tensor(voltage) else None
        if before != after:
            integrator = getattr(self, "integrator", None)
            if integrator is not None:
                # Coefficients must be recomputed at the destination precision
                # and for the selected device backend. MechanismHandler._apply
                # separately moves its non-state scratch and rebuilds scaling
                # closures, so direct Population execution can rebuild lazily.
                integrator.initialized = False
        return result

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
        domain : str or Mapping[str, str], optional
            Spatial-domain metadata applied to every field or selected per field.
            Common aliases are ``"i"`` / ``"intracellular"``, ``"o"`` /
            ``"extracellular"``, and ``"membrane"``.  Geometry-aware material
            processes use this metadata to choose an appropriate volume or area.
        units : str or Mapping[str, str], optional
            Descriptive unit label applied to every field or selected per field,
            for example ``{"ip3i": "mM"}``.  Dendra preserves this label in
            :class:`MaterialFieldSpec`, but does not convert values or perform
            dimensional analysis; mechanism and process equations must use a
            consistent numeric convention.
        conserved : bool or Mapping[str, bool], optional
            Descriptive conservation intent applied to every field or selected
            per field.  This flag is retained as field metadata; setting it does
            not itself enforce conservation. Conservation follows from the
            selected operation (for example finite-volume diffusion or an
            ``ExchangeProcess``) and compatible geometry/units.
        **field_initials
            Convenience initial values, e.g. ``model.material("ip3", ip3i=0.1)``.

        Notes
        -----
        Generic materials have no universal physical unit.  Declare ``units``
        when a field has one, and keep all initial values, mechanism writes,
        additive sources, and material-process parameters consistent with it.
        Ion concentration fields are the important built-in special case: their
        intracellular and extracellular values use mM.

        Examples
        --------
        A concentration-like intracellular field can make all three metadata
        declarations explicit::

            model.material(
                "ip3",
                fields={"ip3i": 0.1},
                min_values={"ip3i": 0.0},
                domain={"ip3i": "intracellular"},
                units={"ip3i": "mM"},
                conserved={"ip3i": True},
            )

        Returns
        -------
        Population
            The population instance for chaining.
        """
        name = _canonical_material_name(name)
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

    def register_material_geometry(
        self,
        name: str,
        *,
        domain: str,
        volume,
        edge_area=None,
        edge_distance=None,
        edge_factor=None,
    ):
        """Register named finite-volume geometry for material transport.

        A named geometry describes a chemical storage/transport domain.  It is
        deliberately independent of electrical extracellular parameters such as
        ``xraxial``, ``xc``, and ``xg``.  Reference it from
        :meth:`DiffusionProcess.DIFFUSE` with ``geometry=name``.

        Parameters
        ----------
        name
            Geometry identifier referenced by
            ``DiffusionProcess.DIFFUSE(..., geometry=name)``.
        domain
            Chemical/material domain represented by this geometry, for example
            ``"extracellular"`` or ``"intracellular"``.
        volume
            Scalar or node/control-volume tensor in ``um^3``, or the name of a
            registered Population buffer/parameter containing it.  It must
            broadcast to the material field; a non-scalar tensor normally has
            final dimension ``C`` (the compartment count).
        edge_area, edge_distance
            Scalar or interface-aligned area in ``um^2`` and center-to-center
            transport distance in ``um``.  Both must broadcast over the
            population's interfaces. Non-scalar 1-D tensors normally have final
            dimension ``C - 1``. A Tree uses compact edge dimension ``E`` in
            the stable order exposed by
            :attr:`~dendra.models.tree.Tree.material_edge_index`. Dendra forms
            ``edge_factor = edge_area / edge_distance``.
        edge_factor
            Scalar or interface-aligned precomputed geometry factor in ``um``.
            Specify this instead of ``edge_area`` and ``edge_distance``.
            Diffusive edge conductance is ``D_edge * edge_factor`` in
            ``um^3 / ms``.

        Returns
        -------
        Population
            ``self``, so registration may be chained.

        Notes
        -----
        String-valued components must name registered buffers or parameters so
        they participate in dtype/device moves and ``state_dict`` handling.
        Direct ordinary tensor-like values are registered as Population
        buffers by this method. A direct ``nn.Parameter`` is instead registered
        as a Population parameter and is discoverable through
        ``model.parameters()``. Differentiable sources preserve autograd.

        A bound DiffusionProcess samples current source values whenever its
        spatial operator is configured (normally on the first step after
        initialization or after explicit timestep reconfiguration), rather
        than reading them on every timestep.  Reconfigure after changing a
        geometry source.

        On the active process region, unbranched control volumes must be finite
        and positive. Implicit Tree diffusion also permits finite zero-volume
        algebraic junctions, provided every conductive component contains at
        least one positive-volume node; explicit Tree diffusion requires every
        active volume to be positive. Interface area/factor must be finite and
        non-negative, and explicit edge distances finite and positive. A zero
        edge factor seals that interface. Validation that depends on the
        eventual process region is deferred until process binding/operator
        configuration. Every field using this geometry must declare the same
        effective chemical domain as ``domain``.

        Named Tree geometry reuses the morphology's immutable parent-child
        topology while replacing chemical storage volumes and edge weights. It
        may seal existing edges but cannot add edges or define an independent
        chemical transport graph.
        """
        name = str(name)
        if not name or not name.replace("_", "").isalnum():
            raise ValueError(
                "Material geometry names must contain only letters, digits, "
                "and underscores."
            )
        if name in self._material_geometries:
            raise ValueError(f"Material geometry {name!r} is already registered.")
        if edge_factor is None:
            if edge_area is None or edge_distance is None:
                raise ValueError(
                    "Material geometry requires edge_factor or both edge_area "
                    "and edge_distance."
                )
        elif edge_area is not None or edge_distance is not None:
            raise ValueError(
                "Specify edge_factor or edge_area/edge_distance, not both."
            )

        aliases = {
            "i": "intracellular",
            "inside": "intracellular",
            "cytosol": "intracellular",
            "cytoplasm": "intracellular",
            "o": "extracellular",
            "outside": "extracellular",
            "extra": "extracellular",
        }
        domain = aliases.get(str(domain).lower(), str(domain).lower())

        def register_component(component: str, value):
            if isinstance(value, str):
                registered = value in self._buffers or value in self._parameters
                if not registered:
                    raise AttributeError(
                        f"Material geometry {name!r} requires {value!r} to be a "
                        "registered Population buffer or parameter."
                    )
                if not torch.is_tensor(getattr(self, value)):
                    raise TypeError(
                        f"Material geometry component {value!r} must be a tensor."
                    )
                return value
            registered_name = f"_material_geometry_{name}_{component}"
            if isinstance(value, torch.nn.Parameter):
                # Reuse an already registered source by identity so passing
                # ``volume=model.volume_parameter`` does not create a second
                # alias in the state dict. Direct Parameters otherwise become
                # true Population parameters and are optimizer-discoverable.
                for parameter_name, parameter in self._parameters.items():
                    if parameter is value:
                        return parameter_name
                parameter = value
                if (
                    parameter.device != self.diam.device
                    or parameter.dtype != self.diam.dtype
                ):
                    parameter = torch.nn.Parameter(
                        parameter.detach().to(
                            device=self.diam.device, dtype=self.diam.dtype
                        ),
                        requires_grad=bool(parameter.requires_grad),
                    )
                self.register_parameter(registered_name, parameter)
                return registered_name

            tensor = torch.as_tensor(
                value, device=self.diam.device, dtype=self.diam.dtype
            )
            self.register_buffer(registered_name, tensor)
            return registered_name

        config = {
            "name": name,
            "domain": domain,
            "volume": register_component("volume", volume),
        }
        if edge_factor is not None:
            config["edge_factor"] = register_component("edge_factor", edge_factor)
        else:
            config["edge_area"] = register_component("edge_area", edge_area)
            config["edge_distance"] = register_component("edge_distance", edge_distance)
        self._material_geometries[name] = config
        if self.is_built:
            self._flag_rebuild = True
        return self

    def material_geometry(self, name: str):
        """Return a copy of a named material-transport geometry configuration.

        The returned mapping contains registered Population source names
        (buffers or parameters) rather than detached tensor snapshots.
        """
        try:
            return dict(self._material_geometries[str(name)])
        except KeyError as exc:
            raise KeyError(
                f"Unknown material geometry {name!r}. Available geometries: "
                f"{sorted(self._material_geometries)}."
            ) from exc

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
        t_ = _time_grid_from_step_count(self.t, n, dt, device=self.device()).to(
            self.dtype()
        )
        stims, indices = intra.init(t_)
        stims = [s.unbind(-1) for s in stims]
        return stims, indices

    # extracellular helpers
    def _broadcast_step_value(
        self, value: TensorLike, *, name: str, dtype=None
    ) -> torch.Tensor:
        """Normalize one spatial/per-step value to the full Population shape.

        Ordinary trailing PyTorch broadcasting is authoritative.  If that
        fails, a value that broadcasts to the explicit batch shape is treated
        as batch-only and padded with singleton population/compartment axes.
        This fallback makes ``[B]`` useful for a ``[B, N, C]`` model without
        changing the established meaning of ``[C]`` when ``B == C``.
        """
        value_t = torch.as_tensor(
            value,
            device=self.device(),
            dtype=self.dtype() if dtype is None else dtype,
        )
        target_shape = tuple(self.v.shape)
        try:
            return torch.broadcast_to(value_t, target_shape)
        except RuntimeError as exc:
            trailing_error = exc

        batch_shape = tuple(self.v.shape[:-2])
        if batch_shape and 0 < value_t.ndim <= len(batch_shape):
            try:
                batch_value = torch.broadcast_to(value_t, batch_shape)
                candidate = batch_value.reshape(*batch_shape, 1, 1)
                return torch.broadcast_to(candidate, target_shape)
            except RuntimeError:
                pass

        raise ValueError(
            f"{name} shape {tuple(value_t.shape)} is not broadcastable to "
            f"model shape {target_shape}. Ordinary trailing PyTorch "
            "broadcasting is tried first; a batch-only fallback accepts values "
            f"broadcastable to batch shape {batch_shape}. Use explicit singleton "
            "axes to disambiguate batch and spatial intent."
        ) from trailing_error

    def _normalize_precomputed_ve(self, value: TensorLike) -> torch.Tensor:
        """Normalize time-first extracellular voltages to ``[T, *model]``."""
        value_t = torch.as_tensor(
            value,
            device=self.device(),
            dtype=self.dtype(),
        )
        if value_t.dim() == 0:
            raise ValueError(
                "Precomputed ve needs a leading time axis; got a scalar tensor."
            )

        n_steps = int(value_t.shape[0])
        payload_shape = tuple(value_t.shape[1:])
        model_shape = tuple(self.v.shape)
        target_shape = (n_steps, *model_shape)
        trailing_error = None
        if len(payload_shape) <= len(model_shape):
            trailing_shape = (
                n_steps,
                *(1,) * (len(model_shape) - len(payload_shape)),
                *payload_shape,
            )
            try:
                candidate = value_t.reshape(trailing_shape)
                return torch.broadcast_to(candidate, target_shape)
            except RuntimeError as exc:
                trailing_error = exc

        batch_shape = tuple(self.v.shape[:-2])
        if batch_shape and 0 < len(payload_shape) <= len(batch_shape):
            batch_aligned_shape = (
                n_steps,
                *(1,) * (len(batch_shape) - len(payload_shape)),
                *payload_shape,
            )
            try:
                batch_value = torch.broadcast_to(
                    value_t.reshape(batch_aligned_shape), (n_steps, *batch_shape)
                )
                candidate = batch_value.reshape(n_steps, *batch_shape, 1, 1)
                return torch.broadcast_to(candidate, target_shape)
            except RuntimeError:
                pass

        raise ValueError(
            f"Precomputed ve per-step shape {payload_shape} is not "
            f"broadcastable to model shape {model_shape}. Ordinary trailing "
            "PyTorch broadcasting is tried first; a batch-only fallback accepts "
            f"payloads broadcastable to batch shape {batch_shape}. Use explicit "
            "singleton axes to disambiguate batch and spatial intent."
        ) from trailing_error

    def _normalize_spatial(self, ve_s_raw: TensorLike) -> torch.Tensor:
        """
        Normalize a spatial field tensor to the full model shape.

        For an unbatched Population this accepts ``[n_comp]``,
        ``[1, n_comp]``, or ``[np, n_comp]``.  After ``batch()``, an
        unbatched field remains valid and is shared by every batch replica;
        ``[*batch, np, n_comp]`` provides replica-specific fields.  A tensor
        uses ordinary trailing broadcasting first.  Only if that fails may a
        low-rank tensor broadcast right-aligned within the explicit batch shape
        and then be padded with singleton neuron/compartment axes.
        """
        ve_s = torch.as_tensor(
            ve_s_raw,
            device=self.device(),
            dtype=self.dtype(),
        ).contiguous()

        if ve_s.dim() == 0:
            raise ValueError(
                "ve_s needs at least one spatial or batch dimension; got a scalar."
            )
        try:
            ve_s = self._broadcast_step_value(ve_s, name="ve_s")
        except ValueError as exc:
            axis_name = (
                "compartment"
                if int(ve_s.shape[-1]) not in (1, int(self.nc))
                else "leading"
            )
            raise ValueError(
                f"ve_s has an incompatible {axis_name} dimension: {exc}"
            ) from exc

        return ve_s

    def _normalize_time_tensor(
        self, time_raw: TensorLike, t_global: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize a time tensor to ``[*batch, np, n_t]``.

        The last axis is always time.  Unbatched ``[np, n_t]`` inputs remain
        valid after batching and are shared across replicas.  Ordinary trailing
        broadcasting is tried first; only on failure may low-rank leading axes
        broadcast right-aligned within the explicit batch shape and be shared
        over neurons.  ``[*batch, 1, n_t]`` makes per-batch intent explicit.
        """
        t_tensor = torch.as_tensor(
            time_raw,
            device=self.device(),
            dtype=self.dtype(),
        )

        if t_tensor.dim() == 0:
            raise ValueError(
                "time tensor needs a trailing time axis; "
                f"got shape {tuple(t_tensor.shape)}."
            )
        if t_tensor.size(-1) != t_global.size(0):
            raise ValueError(
                f"time tensor length ({t_tensor.size(-1)}) must match "
                f"the number of simulation steps ({t_global.size(0)})."
            )
        return self._normalize_temporal_values(
            t_tensor,
            n_steps=int(t_global.size(0)),
            name="time tensor",
        )

    def _normalize_temporal_values(
        self,
        value: torch.Tensor,
        *,
        n_steps: int,
        name: str,
    ) -> torch.Tensor:
        """Broadcast trailing-time data to ``[*batch, np, n_steps]``."""
        if value.dim() == 0:
            value = value.expand(n_steps)
        if value.size(-1) != n_steps:
            raise ValueError(
                f"{name} chunk length ({value.size(-1)}) must match the "
                f"requested chunk length ({n_steps})."
            )

        target_leading = tuple(self.v.shape[:-1])
        leading = tuple(value.shape[:-1])
        batch_shape = tuple(self.v.shape[:-2])
        target_shape = (*target_leading, n_steps)
        try:
            return torch.broadcast_to(value, target_shape)
        except RuntimeError as exc:
            trailing_error = exc

        if batch_shape and 0 < len(leading) <= len(batch_shape):
            try:
                batch_value = torch.broadcast_to(
                    value,
                    (*batch_shape, n_steps),
                )
                candidate = batch_value.reshape(*batch_shape, 1, n_steps)
                return torch.broadcast_to(candidate, target_shape)
            except RuntimeError:
                pass

        raise ValueError(
            f"{name} leading dimension shape {leading} is not broadcastable "
            f"to model batch/population shape {target_leading}. Ordinary trailing "
            "PyTorch broadcasting is tried first; a batch-only fallback accepts "
            f"leading values broadcastable to batch shape {batch_shape}. Use "
            "explicit singleton axes to disambiguate batch and population intent."
        ) from trailing_error

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
            # A tuple is usually one ``(space, time)`` contact, but users may
            # naturally supply a tuple of contact pairs as the documented
            # sequence form.  Recognize that nested structure explicitly.
            tuple_of_pairs = bool(extra) and all(
                isinstance(item, tuple) and len(item) == 2 for item in extra
            )
            extra_pairs = list(extra) if tuple_of_pairs else [extra]
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

    def _expand_eval_time(
        self, t_eval: torch.Tensor, *, n_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Normalize evaluated Waveform output to ``[*batch, np, n_t_chunk]``.

        Accepts:
        - scalar (broadcast across ``n_steps`` when provided)
        - [n_t_chunk]
        - [1, n_t_chunk]
        - [np, n_t_chunk]

        Batched values follow the same trailing-first, batch-only-fallback
        convention as :meth:`_normalize_time_tensor`.
        """
        if n_steps is None:
            if t_eval.dim() == 0:
                raise ValueError(
                    "Scalar Waveform output requires the expected number of steps."
                )
            n_steps = int(t_eval.size(-1))
        return self._normalize_temporal_values(
            t_eval,
            n_steps=int(n_steps),
            name="Waveform output",
        )

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
                    t_i = self._expand_eval_time(t_i, n_steps=t_chunk.numel())
                    t_per_contact.append(t_i.unsqueeze(0))  # [1, np, n_t_chunk]

                t_extra = torch.cat(t_per_contact, dim=0)
            else:
                wf = cfg.waveforms[0]
                t_extra = wf(t_chunk).to(self.dtype())
                t_extra = self._expand_eval_time(t_extra, n_steps=t_chunk.numel())
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

    def _duration_budget(self, duration: float, dt: float) -> tuple[int, float]:
        """Resolve a duration call against this Population's retained time."""
        pending = float(self._duration_remainder.detach().cpu().item())
        return _duration_step_budget(duration, dt, pending)

    def _set_duration_remainder(self, value: float) -> None:
        """Commit retained physical time without mutating checkpoint aliases."""
        self._duration_remainder = torch.tensor(
            value,
            device=self._duration_remainder.device,
            dtype=torch.float64,
        )

    def _clear_duration_remainder(self) -> None:
        """Start a fresh duration budget for a new simulation episode."""
        self._duration_remainder = torch.zeros(
            (), device=self._duration_remainder.device, dtype=torch.float64
        )

    def step(
        self,
        dt: Optional[float] = None,
        *,
        ve: Optional[TensorLike] = None,
        extra: Optional[ExtraSpec] = None,
        callbacks: Optional[Sequence[Callback]] = None,
        loop_hooks: bool = False,
    ):
        """Advance this population by exactly one timestep.

        Parameters
        ----------
        dt : float, optional
            Timestep in milliseconds. If ``None``, uses the backend default
            ``A.dt``, matching :meth:`run`.
        ve : Tensor, optional
            Extracellular voltage for this single step, normalized to
            ``[*batch, np, n_comp]``. Ordinary trailing PyTorch broadcasting is
            tried first. If it fails, low-rank data may broadcast right-aligned
            within the explicit batch shape and be shared spatially. A leading
            singleton time dimension ``[1, ...]`` is also accepted.
        extra : extracellular specification, optional
            Higher-level extracellular stimulation specification with the same
            shape and broadcasting semantics as :meth:`run`. It is evaluated
            at the model's current time and converted to this step's ``ve``.
        callbacks : sequence of Callback, optional
            Callbacks to execute around this single step. By default ``step``
            calls ``pre_step_hook`` and ``post_step_hook`` only.
        loop_hooks : bool, optional
            If ``True``, also call ``pre_loop_hook`` before the step and
            ``post_loop_hook`` after the step. Leave this ``False`` when
            manually stepping inside an outer user-controlled loop.

        Returns
        -------
        Population
            ``self``, for chaining.
        """
        if not self.initialized:
            raise ValueError("Model must be initialized before stepping.")
        self._validate_runtime_contracts(
            workspace_rebuild_pending=self._runtime_workspace_rebuild_pending()
        )
        self._refresh_compile_config_from_ctx()

        dt_f = _validate_time_scalar(
            A.dt if dt is None else dt, name="dt", positive=True
        )

        if ve is not None and extra is not None:
            raise ValueError("Provide either 've' or 'extra', not both.")

        # Match Population.run: lazily rebuild solver-level intracellular
        # stimulation when new injections invalidated ``self.intra``.
        if self.intra is None:
            self.intra = self.build_intra()
        intra = self.intra
        with_intra = intra is not None

        device = self.device()
        dtype = self.dtype()
        dt_tensor = torch.tensor(dt_f, device=device, dtype=dtype)

        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(callbacks)
        if callbacks:
            for c in callbacks:
                c.dt = dt_f

        ctx = nullcontext() if self.training else torch.no_grad()
        with ctx:
            self.integrator._initialize(
                self,
                dt_tensor,
                force=self.force_integrator_reinit(),
                compile_scope="population",
            )

            if loop_hooks:
                pre_loop_hook(callbacks, self)
            pre_step_hook(callbacks, self)

            if ve is not None:
                ve_c = torch.as_tensor(ve, device=device, dtype=dtype).contiguous()
                if ve_c.dim() == self.v.dim() + 1 and int(ve_c.shape[0]) == 1:
                    ve_c = ve_c[0]
                ve_c = self._broadcast_step_value(ve_c, name="ve").contiguous()
            elif extra is not None:
                t_step = self.t.reshape(1).to(device=device, dtype=dtype)
                extra_cfg = self._prepare_extra(extra, t_step, n_chunks=1)
                ve_list = self._compute_extra_chunk(extra_cfg, 0, t_step)
                ve_c = None if ve_list is None else ve_list[0]
            else:
                ve_c = None

            if with_intra:
                stims, indices = self.prep_intra(intra, 1, dt_f)
                s = [st[0] for st in stims]
                intra_c = self._call_make_intra(intra, s, indices)
            else:
                intra_c = None

            self._step(self.integrator, self, dt_tensor, ve_c, intra_c)
            self.t = self.t + dt_tensor

            post_step_hook(callbacks, self)
            if loop_hooks:
                post_loop_hook(callbacks, self)

        return self

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
            Precomputed time-first extracellular voltage, normalized to
            ``[n_timesteps, *batch, np, n_comp]``. Payload axes after time use
            ordinary trailing broadcasting first; low-rank payloads may use a
            right-aligned batch-only fallback. Use
            ``[n_timesteps, *batch, 1, 1]`` for explicit per-batch, spatially
            shared input. The number of time steps is inferred from
            ``ve.shape[0]``.
        extra : tuple or sequence of tuples, optional
            Extracellular input specification(s), with the same semantics as
            :meth:`longrun`.

            Each specification is a tuple ``(ve_s, time)``:

            * ``ve_s`` is normalized to ``[*batch, np, n_comp]``.

            * ``time`` is either a :class:`Waveform` or a time-last tensor
              normalized to ``[*batch, np, n_timesteps]``.

            Both values use ordinary trailing broadcasting first. Only if that
            fails may low-rank value axes broadcast right-aligned within the
            explicit batch shape and then be shared over spatial axes. Explicit
            singleton axes disambiguate equal-size batch and spatial dimensions.

            If a single tuple is provided, the method uses a single-contact
            formulation with :func:`op_sc`. If a sequence of tuples is provided,
            each tuple is treated as one electrode contact, and the method
            automatically switches to multi-contact mode using :func:`op_mc`:

            * Spatial fields are stacked to shape
              ``[n_contacts, *batch, np, n_comp]``.
            * Functional (Waveform) inputs are evaluated per time step and
              expanded/concatenated to shape
              ``[n_contacts, *batch, np, n_timesteps]``.
            * Non-functional (tensor) inputs are normalized once to that shape.

            Mixing :class:`Waveform` and tensor time specifications across contacts
            is not supported and will raise a :class:`ValueError`.

            If both ``ve`` and ``extra`` are provided, a :class:`ValueError` is
            raised.
        tstop : float, optional
            Simulation duration in milliseconds to advance from the current
            model time ``self.t`` when ``ve`` is not provided. Only complete
            fixed timesteps are executed. Any fractional physical duration is
            retained and combined with the next duration-based ``run``,
            :meth:`longrun`, or :meth:`longrun_checkpointed` call. If ``ve`` is
            provided, its leading length is authoritative, ``tstop`` is ignored,
            and the retained duration is left unchanged.
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
        self._validate_runtime_contracts(
            workspace_rebuild_pending=self._runtime_workspace_rebuild_pending()
        )
        self._refresh_compile_config_from_ctx()

        dt_f = _validate_time_scalar(
            A.dt if dt is None else dt, name="dt", positive=True
        )
        if ve is not None and extra is not None:
            raise ValueError("Provide either 've' or 'extra', not both.")
        tstop_f = None
        duration_steps = None
        duration_remainder = None
        if ve is None:
            if tstop is None:
                raise ValueError("tstop must be provided when 've' is not given.")
            tstop_f = _validate_time_scalar(tstop, name="tstop", positive=False)
            duration_steps, duration_remainder = self._duration_budget(tstop_f, dt_f)

        # auto-build intra if missing
        if self.intra is None:
            intra = self.build_intra()
            self.intra = intra
        else:
            intra = self.intra

        with_intra = intra is not None

        device = self.device()
        dtype = self.dtype()

        if ve is not None:
            ve = self._normalize_precomputed_ve(ve).contiguous()

        # dt as scalar and tensor
        dt_tensor = torch.tensor(dt_f, device=device, dtype=dtype)

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
                t_global = _time_grid_from_step_count(
                    self.t, n, dt_f, device=device
                ).to(dtype)
            else:
                n = duration_steps
                t_global = _time_grid_from_step_count(
                    self.t, n, dt_f, device=device
                ).to(dtype)

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

            if duration_remainder is not None:
                self._set_duration_remainder(duration_remainder)

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
                    intra_c = self._call_make_intra(intra, s, indices)
                else:
                    intra_c = None

                # Integrator step
                pre_step_hook(callbacks, self)
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
            Simulation duration in milliseconds to advance from the current
            model time ``self.t``. Only complete fixed timesteps are executed;
            fractional physical duration is retained and shared with subsequent
            duration-based Population execution calls.
        chunklength : int
            The number of time steps to process in each chunk.
        dt : float, optional
            The simulation time step in milliseconds. If ``None``, the default value
            from the backend will be used.
        extra : tuple or sequence of tuples, optional
            Extracellular input specification(s).

            Each specification is a tuple ``(ve_s, time)``:

            * ``ve_s`` is normalized to ``[*batch, n_p, n_comp]``.
            * ``time`` is either a :class:`Waveform` or a time-last tensor
              normalized to ``[*batch, n_p, n_timesteps]``.

            Both use ordinary trailing broadcasting first, followed only on
            failure by a low-rank, right-aligned batch-only fallback. Explicit
            singleton axes disambiguate equal-size batch and spatial dimensions.

            If a single tuple is provided, the method uses a single-contact
            formulation with :func:`op_sc`. If a sequence of tuples is provided,
            each tuple is treated as one electrode contact, and the method
            automatically switches to multi-contact mode with :func:`op_mc`:

            * In multi-contact mode, spatial fields are stacked to shape
              ``[n_contacts, *batch, n_p, n_comp]``.
            * For a functional specification (all ``time`` are :class:`Waveform`),
              each waveform is evaluated per chunk and per contact and then
              expanded/concatenated to shape
              ``[n_contacts, *batch, n_p, n_t_chunk]``.
            * For a non-functional specification (all ``time`` are tensors), the
              raw time tensors are pre-split into chunks and concatenated to the
              same shape ``[n_contacts, *batch, n_p, n_t_chunk]``.

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
        self._validate_runtime_contracts(
            workspace_rebuild_pending=self._runtime_workspace_rebuild_pending()
        )
        if not isinstance(chunklength, int) or isinstance(chunklength, bool):
            raise ValueError("chunklength must be a positive integer.")
        if chunklength <= 0:
            raise ValueError("chunklength must be a positive integer.")
        dt_f = _validate_time_scalar(
            A.dt if dt is None else dt, name="dt", positive=True
        )
        tstop_f = _validate_time_scalar(tstop, name="tstop", positive=False)
        n_steps, duration_remainder = self._duration_budget(tstop_f, dt_f)
        self._refresh_compile_config_from_ctx()

        # Match step()/run(): injections registered after initialization
        # invalidate ``self.intra`` and are rebuilt lazily on first use.
        if self.intra is None:
            self.intra = self.build_intra()
        intra = self.intra
        with_intra = intra is not None

        # dt scalars
        dt_tensor = torch.tensor(dt_f, device=self.device(), dtype=self.dtype())

        psh = post_step_hook

        with torch.nn.utils.parametrize.cached():
            with torch.set_grad_enabled(self.training):
                # --------------------------------------------------------------
                # Global time grid and chunking
                # --------------------------------------------------------------
                t = _time_grid_from_step_count(
                    self.t, n_steps, dt_f, device=self.device()
                ).to(self.dtype())

                if t.numel() == 0:
                    self._set_duration_remainder(duration_remainder)
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

                self._set_duration_remainder(duration_remainder)
                pre_loop_hook(callbacks, self)

                # --------------------------------------------------------------
                # Main chunk loop
                # --------------------------------------------------------------
                for i, t_chunk in enumerate(t_chunks):
                    if with_intra:
                        stims, indices = intra.init(t_chunk)
                        # Waveform output always carries time on its last axis;
                        # leading axes are population batch/sweep axes.
                        stims = [s.unbind(-1) for s in stims]

                    if with_extra:
                        ve_list = self._compute_extra_chunk(extra_cfg, i, t_chunk)
                    else:
                        ve_list = None

                    pre_chunk_hook(callbacks, self, t_chunk)

                    for j in range(len(t_chunk)):
                        ve_c = ve_list[j] if ve_list is not None else None

                        if with_intra:
                            s = [st[j] for st in stims]
                            intra_c = self._call_make_intra(intra, s, indices)
                        else:
                            intra_c = None

                        pre_step_hook(callbacks, self)
                        self._step(self.integrator, self, dt_tensor, ve_c, intra_c)
                        self.t = self.t + dt_tensor
                        psh(callbacks, self)

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

        dt_f = _validate_time_scalar(dt, name="dt", positive=True)
        tstop_f = _validate_time_scalar(tstop, name="tstop", positive=False)
        self._refresh_compile_config_from_ctx()
        self.clear_steady_state()

        self.initialize()
        self.integrator._initialize(self, dt_f, force=True, compile_scope="population")

        maxiter = int(tstop_f / dt_f)

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

            dt_tensor = torch.tensor(dt_f, device=self.device(), dtype=self.dtype())
            for _ in tqdm(range(maxiter), desc="Steady state "):
                self._step(self.integrator, self, dt_tensor, ve, intra)

        self.cache("_steady_state")
        self.t = torch.zeros_like(self.t).detach()
        self._clear_duration_remainder()
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

    def populate(self, random_generation=None):
        """
        Populate the model with mechanisms and parameters.

        This method is called after the model is built to ensure that all
        mechanisms and parameters are properly initialized and ready for use.
        It should be called after the model's build() method.
        """
        self.populate_parameter_buffers(random_generation=random_generation)
        self.mech.populate(random_generation=random_generation)
        return self

    def populate_(self):
        """
        "In-place" alias of :meth:`populate`. Same as :meth:`populate`, but
        does not return self.
        """
        self.populate()

    def _has_random_parameters_resampled_on_initialize(self):
        """Return True if this population or any mechanism redraws on initialize."""

        if any(spec.resample_on_initialize for spec in self.random_parameters.values()):
            return True
        mech_handler = getattr(self, "mech", None)
        if mech_handler is None:
            return False
        for mech in getattr(mech_handler, "mechanisms", {}).values():
            if any(
                spec.resample_on_initialize for spec in mech.random_parameters.values()
            ):
                return True
            for state_module in getattr(mech, "DE", {}).values():
                if any(
                    spec.resample_on_initialize
                    for spec in state_module.random_parameters.values()
                ):
                    return True
        return False

    def resample_random_parameters(self, *names, force: bool = True):
        """Resample random parameters on the population and inserted mechanisms."""

        if names:
            local = tuple(n for n in names if n in self.random_parameters)
            if local:
                super().resample_random_parameters(*local, force=force)
        else:
            super().resample_random_parameters(force=force)
        if hasattr(self, "mech"):
            self.mech.resample_random_parameters(*names, force=force)
        self.clear_steady_state()
        return self

    def resample_runtime_noise(self, *names, phase: str | None = None, dt=None):
        """Manually refresh detached runtime NOISE buffers.

        Runtime NOISE samples are detached and updated in-place. This method is
        useful for debugging or for explicitly refreshing ``GLOBALNOISE``,
        ``RANGENOISE``, or ``BATCHNOISE`` variables outside the normal simulation
        phases. It does not clear the steady-state cache because runtime noise is
        interpreted as an exogenous simulation drive rather than quenched model
        heterogeneity.
        """

        if names:
            local = tuple(n for n in names if n in self.runtime_noises)
            if local:
                super().resample_runtime_noise(*local, phase=phase, dt=dt)
        else:
            super().resample_runtime_noise(phase=phase, dt=dt)
        if hasattr(self, "mech"):
            self.mech.resample_runtime_noise(*names, phase=phase, dt=dt)
        return self

    def _restore_steady_state(self):
        if "_steady_state" in self._caches:
            self.restore("_steady_state")
            # ``restore`` marks ordinary named-cache restores usable. A
            # steady-state initialization still has a post-initialize phase,
            # so keep this transition fail-closed until every hook succeeds.
            self.initialized = False
            self.post_initialize()
            self.t = torch.zeros_like(self.t).detach()
            self._clear_duration_remainder()
            self.initialized = True
            self.initializing_from_state_cache = True
            return True
        return False

    def _refresh_parameter_views_for_initialization(self):
        """Refresh subclass-owned derived parameter views before state setup."""
        return None

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
        # Initialization is a state transition, not a best-effort refresh.  If
        # any build/population/hook phase fails, callers must repair and retry
        # instead of stepping a partially reset model advertised as initialized.
        self.initialized = False
        self.initializing_from_state_cache = False
        self._clear_duration_remainder()
        existing_integrator = getattr(self, "integrator", None)
        if existing_integrator is not None:
            existing_integrator.initialized = False

        self.build(force_rebuild)
        random_generation = object() if populate_parameter_buffers else None
        if (
            populate_parameter_buffers
            and "_steady_state" in self._caches
            and self._has_random_parameters_resampled_on_initialize()
        ):
            self.clear_steady_state()
        if populate_parameter_buffers:
            self.populate_parameter_buffers(random_generation=random_generation)
        self._refresh_parameter_views_for_initialization()
        self.intra = self.build_intra()
        if self._restore_steady_state():
            return self
        self.integrator.init_v(self)
        self.pre_initialize()
        self.integrator.mech.initialize(
            self.v,
            self.celsius,
            self.diam,
            populate=populate_parameter_buffers,
            random_generation=random_generation,
        )
        self.post_initialize()
        self.integrator.mech.initialize(
            self.v,
            self.celsius,
            self.diam,
            populate=populate_parameter_buffers,
            random_generation=random_generation,
        )
        self.t = torch.zeros_like(self.t).detach()
        self._clear_duration_remainder()
        self.initialized = True
        # Voltage/mechanism state was reconstructed even when topology and dt
        # were unchanged. Rebuild timestep-dependent solver workspaces before
        # the next step in every case.
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
        if isinstance(state_dict, (str, os.PathLike)):
            state_dict = torch.load(
                state_dict, map_location=self.device(), weights_only=True
            )
        has_duration_remainder = (
            isinstance(state_dict, Mapping) and "_duration_remainder" in state_dict
        )
        _load_compatible_state_dict_transactionally(self, state_dict)
        if not has_duration_remainder:
            # Normal state dictionaries created before duration carry support
            # must not retain unrelated pending time from the receiver.
            self._clear_duration_remainder()
        else:
            self._set_duration_remainder(
                float(self._duration_remainder.detach().cpu().item())
            )
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
        self._caches[name] = _clone_nested_state(self.state_dict(), detach_tensors=True)
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
        _load_population_cache_transactionally(self, self._caches[name])
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

    def _dispatch_mechanism_injections(
        self, mech_handler=None, *, start=0, reset_acceptance=False
    ):
        """Offer each stored waveform injection to at most one mechanism.

        Returning ``True`` from ``Mechanism.inject`` claims complete ownership
        of that injection.  The first accepting mechanism wins; otherwise the
        solver-level :class:`Intra` path remains responsible for it.
        """
        mech_handler = self.mech if mech_handler is None else mech_handler
        if mech_handler is None:
            return
        if not self.mechanism_injections:
            return

        if reset_acceptance:
            self.mechanism_injection_accepted = [False] * len(self.mechanism_injections)

        model_shape = tuple(self.shape)
        for inj_i, (waveform, shape, index) in enumerate(
            self.mechanism_injections[start:], start
        ):
            accepted = bool(self.mechanism_injection_accepted[inj_i])
            if accepted:
                continue
            for mech in mech_handler.mechanisms.values():
                accepted = bool(
                    mech.inject(
                        waveform,
                        index=index,
                        shape=shape,
                        model_shape=model_shape,
                        model=self,
                    )
                )
                if accepted:
                    break
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

    def _invalidate_mechanism_structure(self):
        """Fail closed after a successful mechanism-layout mutation."""
        if self.is_built:
            self._flag_rebuild = True
        self.initialized = False
        self.initializing_from_state_cache = False
        integrator = getattr(self, "integrator", None)
        if integrator is not None:
            integrator.initialized = False
        self.intra = None
        self.mechanism_injection_accepted = [
            False for _ in self.mechanism_injection_accepted
        ]
        self.clear_steady_state()

    def _configured_mechanism_names(self, mechanism):
        """Return runtime names associated with one configured mechanism class."""
        names = {
            str(getattr(mechanism, "_name", None) or mechanism.__name__),
            str(mechanism.__name__),
        }
        if mechanism in self._mech_everywhere:
            names.add(str(self._mech_everywhere[mechanism][0]))
        return names

    def _configured_mechanism_class_for_instance(self, name, instance):
        """Resolve a parametrized runtime proxy back to its configured class."""
        candidates = set(self._mech_everywhere) | set(self._mech_data)
        named = [
            mechanism
            for mechanism in candidates
            if str(name) in self._configured_mechanism_names(mechanism)
            and isinstance(instance, mechanism)
        ]
        if len(named) == 1:
            return named[0]
        return type(instance)

    def _mechanism_support_flat(self, mechanism):
        """Return sorted physical core support for a configured mechanism class."""
        core_shape = tuple(self.core_shape())
        support = []
        if mechanism in self._mech_everywhere:
            all_indices = torch.arange(math.prod(core_shape), dtype=torch.long)
            excluded = torch.as_tensor(
                self._mech_exclusions.get(mechanism, []), dtype=torch.long
            ).reshape(-1)
            if excluded.numel():
                all_indices = all_indices[~torch.isin(all_indices, excluded)]
            support.append(all_indices)
        for record in self._mech_data.get(mechanism, ()):
            _, _, key, _, _ = _unpack_mechanism_insertion_record(record)
            support.append(_core_flat_indices(key, core_shape))
        if not support:
            return torch.empty(0, dtype=torch.long)
        return _sorted_unique_long(torch.cat(support))

    def _sparse_mechanism_slot_domain(self, mechanism):
        """Return physical core indices in compiled sparse-slot order."""
        records = self._mech_data.get(mechanism, ())
        if not records:
            return torch.empty(0, dtype=torch.long)
        keys = []
        preserve = []
        copies = []
        for record in records:
            _, _, key, keep_duplicates, n_copies = _unpack_mechanism_insertion_record(
                record
            )
            keys.append(key)
            preserve.append(keep_duplicates)
            copies.append(n_copies)
        if any(preserve):
            total_index, _, _, _ = compose_or_flatten_multiset(
                keys, tuple(self.core_shape()), preserve, copies
            )
            return torch.as_tensor(total_index, dtype=torch.long)
        total_index, is_composable, _, _ = compose_or_flatten_union(
            keys, tuple(self.core_shape())
        )
        if is_composable:
            return _core_flat_indices(total_index, tuple(self.core_shape()))
        return torch.as_tensor(total_index, dtype=torch.long).reshape(-1)

    def insert(
        self,
        mechanism,
        alias=None,
        index_spec=None,
        ic=None,
        preserve_duplicate_indices: bool = False,
        preserve_multiplicity: bool | None = None,
        copies: int = 1,
        **kwargs,
    ):
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
        ic : dict, optional
            Initial values for mechanism state fields. For region-restricted
            insertions these values are class-wide because all regions of one
            mechanism class compile into one mechanism instance.
        preserve_duplicate_indices : bool, optional
            If True, duplicate compartment indices selected by this insertion are
            retained as independent mechanism slots instead of being collapsed
            into the usual unique union. This is useful for colocated point
            processes or synapse banks that should maintain independent state
            while scattering their currents back to the same compartment.
        preserve_multiplicity : bool, optional
            User-facing synonym for ``preserve_duplicate_indices``.
        copies : int, optional
            Number of independent copies of this insertion region to allocate.
            ``copies > 1`` implies ``preserve_duplicate_indices=True`` and is
            intended for compact banks of colocated point processes.
        **kwargs
            Mechanism parameters. RANGE/BATCH values are restricted to the
            insertion region. GLOBAL values are class-wide; later insertion
            records for the same class must use compatible GLOBAL values.
        """
        validate(mechanism)
        if hasattr(mechanism, "normalize_mech_kwargs"):
            kwargs = mechanism.normalize_mech_kwargs(kwargs)
        elif hasattr(mechanism, "normalize_random_kwargs"):
            kwargs = mechanism.normalize_random_kwargs(kwargs)
        if preserve_multiplicity is not None:
            preserve_duplicate_indices = bool(
                preserve_duplicate_indices or preserve_multiplicity
            )
        copies = int(copies)
        if copies < 1:
            raise ValueError(f"copies must be a positive integer; got {copies!r}.")

        key = None

        if index_spec is not None:
            key = _index_to_cpu(index_spec.index)

        if key is None:
            if copies != 1 or preserve_duplicate_indices:
                raise ValueError(
                    "copies and preserve_duplicate_indices are only supported "
                    "for region-restricted mechanism insertions."
                )
            if alias is None:
                if mechanism in self._mech_everywhere:
                    raise ValueError(
                        f"Mechanism {mechanism} is already inserted everywhere."
                    )
                if self._mech_data.get(mechanism):
                    raise ValueError(
                        f"Mechanism {mechanism} already has region-restricted "
                        "insertions and cannot also be inserted everywhere."
                    )
                self._mech_everywhere[mechanism] = (mechanism.__name__, ic, kwargs)
                self._mech_exclusions.pop(mechanism, None)
                self._invalidate_mechanism_structure()
                return
            key = _core_key_from_flat(
                torch.arange(math.prod(self.core_shape()), dtype=torch.long),
                tuple(self.core_shape()),
            )

        if mechanism in self._mech_everywhere:
            raise ValueError(f"Mechanism {mechanism} is already inserted everywhere.")
        existing_ic = self._mech_data_ic.get(mechanism)
        if ic is not None:
            if existing_ic is not None and not _configuration_values_equal(
                existing_ic, ic
            ):
                raise ValueError(
                    f"Mechanism {mechanism} already has different class-wide "
                    "initial conditions for another insertion region."
                )

        global_names = _mechanism_global_parameter_names(mechanism)
        global_kwargs = {name: kwargs[name] for name in kwargs.keys() & global_names}
        indexed_kwargs = {
            name: value for name, value in kwargs.items() if name not in global_names
        }
        existing_base_kwargs = self._mech_data_base_kwargs.get(mechanism, {})
        planned_base_kwargs = dict(existing_base_kwargs)
        for name, value in global_kwargs.items():
            if name in planned_base_kwargs and not _global_configuration_values_equal(
                planned_base_kwargs[name], value
            ):
                raise ValueError(
                    f"Mechanism {mechanism} already has a different class-wide "
                    f"GLOBAL value for {name!r}."
                )
            planned_base_kwargs[name] = value

        if ic is not None:
            self._mech_data_ic[mechanism] = ic
        if planned_base_kwargs:
            self._mech_data_base_kwargs[mechanism] = planned_base_kwargs
        self._mech_data.setdefault(mechanism, []).append(
            (
                alias,
                indexed_kwargs,
                _index_to_cpu(key),
                bool(preserve_duplicate_indices or copies != 1),
                copies,
            )
        )
        self._invalidate_mechanism_structure()

    def delete(self, mechanism, index=None, *, strict=False):
        """Delete a mechanism class everywhere or where it intersects a region.

        Parameters
        ----------
        mechanism : type
            Exact mechanism class to remove.
        index : IndexSpec or object, optional
            Physical population-core selection to remove. ``None`` (the
            default) removes every configured placement of ``mechanism``.
        strict : bool, default False
            If True, require the mechanism to exist at every selected physical
            compartment and fail atomically otherwise. If False, delete only
            the intersection between the selection and current support; a
            missing class or empty intersection is a no-op.

        Notes
        -----
        Deletion subtracts selected support from every overlapping insertion
        record, including all duplicate/copy slots at those compartments.
        Reinitialize the population after a successful deletion.
        """
        changed = self._delete_mechanism_configuration(
            mechanism,
            index=index,
            strict=strict,
        )
        if changed:
            self._invalidate_mechanism_structure()

    def delete_all(self, index=None):
        """Delete every mechanism class present in a population region.

        Parameters
        ----------
        index : IndexSpec or object, optional
            Physical population-core selection. ``None`` (the default) removes
            every configured mechanism placement from the Population.

        Notes
        -----
        This operation uses intersection semantics for every exact configured
        mechanism class. Planning and projection are transactional across
        classes: if any affected configuration cannot be projected safely, no
        mechanism configuration or lifecycle state is changed.
        """
        normalized_index = index
        if index is not None:
            core_shape = tuple(self.core_shape())
            requested = _stable_unique_long(_core_flat_indices(index, core_shape))
            if requested.numel() == 0:
                return
            normalized_index = _core_key_from_flat(requested, core_shape)

        mechanisms = list(self._mech_everywhere)
        mechanisms.extend(
            mechanism
            for mechanism, records in self._mech_data.items()
            if records and mechanism not in self._mech_everywhere
        )
        if not mechanisms:
            return

        registry_objects = {
            "_mech_everywhere": self._mech_everywhere,
            "_mech_exclusions": self._mech_exclusions,
            "_mech_data": self._mech_data,
            "_mech_data_ic": self._mech_data_ic,
            "_mech_data_base_kwargs": self._mech_data_base_kwargs,
            "_slice_mechanism_parametrizations": (
                self._slice_mechanism_parametrizations
            ),
        }
        snapshots = {
            "_mech_everywhere": dict(registry_objects["_mech_everywhere"]),
            "_mech_exclusions": dict(registry_objects["_mech_exclusions"]),
            "_mech_data": dict(registry_objects["_mech_data"]),
            "_mech_data_ic": dict(registry_objects["_mech_data_ic"]),
            "_mech_data_base_kwargs": dict(registry_objects["_mech_data_base_kwargs"]),
            "_slice_mechanism_parametrizations": list(
                registry_objects["_slice_mechanism_parametrizations"]
            ),
        }
        changed = False
        try:
            for mechanism in mechanisms:
                changed = (
                    self._delete_mechanism_configuration(
                        mechanism,
                        index=normalized_index,
                        strict=False,
                    )
                    or changed
                )
        except Exception:
            for name, snapshot in snapshots.items():
                registry = registry_objects[name]
                registry.clear()
                if isinstance(registry, dict):
                    registry.update(snapshot)
                else:
                    registry.extend(snapshot)
                setattr(self, name, registry)
            raise

        if changed:
            self._invalidate_mechanism_structure()

    def _delete_mechanism_configuration(
        self,
        mechanism,
        index=None,
        *,
        strict=False,
    ):
        """Plan and commit one class deletion without invalidating lifecycle."""
        validate(mechanism)
        requested = None
        if index is not None:
            core_shape = tuple(self.core_shape())
            requested = _stable_unique_long(_core_flat_indices(index, core_shape))
        configured = mechanism in self._mech_everywhere or bool(
            self._mech_data.get(mechanism)
        )
        if not configured:
            if strict:
                raise ValueError(
                    f"Mechanism {getattr(mechanism, '__name__', mechanism)!r} "
                    "is not inserted in this Population."
                )
            return False

        mechanism_names = self._configured_mechanism_names(mechanism)

        def record_matches(record):
            recorded_class = record.get("mechanism_class")
            if recorded_class is not None:
                if recorded_class is mechanism:
                    return True
                configured_classes = set(self._mech_everywhere) | set(self._mech_data)
                try:
                    is_runtime_proxy = (
                        recorded_class not in configured_classes
                        and issubclass(recorded_class, mechanism)
                    )
                except TypeError:
                    is_runtime_proxy = False
                return bool(
                    is_runtime_proxy and record.get("mechanism_name") in mechanism_names
                )
            return record.get("mechanism_name") in mechanism_names

        if index is None:
            new_parametrizations = [
                record
                for record in self._slice_mechanism_parametrizations
                if not record_matches(record)
            ]
            self._mech_everywhere.pop(mechanism, None)
            self._mech_exclusions.pop(mechanism, None)
            self._mech_data.pop(mechanism, None)
            self._mech_data_ic.pop(mechanism, None)
            self._mech_data_base_kwargs.pop(mechanism, None)
            self._slice_mechanism_parametrizations = new_parametrizations
            return True

        if requested.numel() == 0:
            return False

        old_support = self._mechanism_support_flat(mechanism)
        supported = torch.isin(requested, old_support)
        if strict and not bool(torch.all(supported)):
            missing = requested[~supported][:10].tolist()
            raise ValueError(
                f"Cannot delete mechanism {mechanism.__name__!r}: selected "
                f"physical core indices {missing} do not currently host that "
                "exact mechanism class. No changes were made."
            )
        requested = requested[supported]
        if requested.numel() == 0:
            return False

        remaining_support = old_support[~torch.isin(old_support, requested)]
        planned_everywhere = self._mech_everywhere.get(mechanism)
        planned_exclusions = self._mech_exclusions.get(mechanism)
        batch_names = _mechanism_batch_parameter_names(mechanism)

        if planned_everywhere is not None:
            configured_name, configured_ic, configured_kwargs = planned_everywhere
            excluded = torch.as_tensor(
                [] if planned_exclusions is None else planned_exclusions,
                dtype=torch.long,
            ).reshape(-1)
            old_everywhere_support = torch.arange(
                math.prod(core_shape), dtype=torch.long
            )
            if excluded.numel():
                old_everywhere_support = old_everywhere_support[
                    ~torch.isin(old_everywhere_support, excluded)
                ]
            everywhere_keep = ~torch.isin(old_everywhere_support, requested)
            planned_exclusions = _sorted_unique_long(torch.cat((excluded, requested)))
            if not bool(torch.any(everywhere_keep)):
                planned_everywhere = None
                planned_exclusions = None
            elif not bool(torch.all(everywhere_keep)):
                for parameter_name in configured_kwargs.keys() & batch_names:
                    _validate_deletion_stable_batch_override(
                        configured_kwargs[parameter_name],
                        context=(
                            f"Mechanism {mechanism.__name__!r} everywhere "
                            f"parameter {parameter_name!r}"
                        ),
                    )
                indexed_names = _mechanism_spatial_parameter_names(mechanism)
                configured_kwargs = {
                    name: (
                        _project_indexed_override(
                            value,
                            old_everywhere_support,
                            everywhere_keep,
                            core_shape,
                            context=(
                                f"Mechanism {mechanism.__name__!r} everywhere "
                                f"parameter {name!r}"
                            ),
                        )
                        if name in indexed_names
                        else value
                    )
                    for name, value in configured_kwargs.items()
                }
                if configured_ic is not None:
                    configured_ic = {
                        name: _project_indexed_override(
                            value,
                            old_everywhere_support,
                            everywhere_keep,
                            core_shape,
                            context=(
                                f"Mechanism {mechanism.__name__!r} everywhere "
                                f"initial condition {name!r}"
                            ),
                        )
                        for name, value in configured_ic.items()
                    }
                planned_everywhere = (
                    configured_name,
                    configured_ic,
                    configured_kwargs,
                )

        planned_records = []
        planned_sparse_ic = self._mech_data_ic.get(mechanism)
        old_sparse_domain = self._sparse_mechanism_slot_domain(mechanism)
        indexed_names = _mechanism_spatial_parameter_names(mechanism)
        sparse_records = tuple(self._mech_data.get(mechanism, ()))
        for record_number, record in enumerate(sparse_records):
            alias, kwargs, key, preserve, copies = _unpack_mechanism_insertion_record(
                record
            )
            selected = _core_flat_indices(key, core_shape)
            if preserve:
                value_domain = selected.repeat(copies)
                value_keep = ~torch.isin(value_domain, requested)
                key_keep = ~torch.isin(selected, requested)
                retained = selected[key_keep]
            else:
                value_domain = _sorted_unique_long(selected)
                value_keep = ~torch.isin(value_domain, requested)
                retained = value_domain[value_keep]

            logical_shape = None
            if preserve:
                if value_domain.numel() % copies:
                    raise RuntimeError(
                        "Copied mechanism insertion has an inconsistent "
                        f"stored layout: {value_domain.numel()} slots for "
                        f"{copies} copies."
                    )
                logical_shape = (copies, value_domain.numel() // copies)

            if retained.numel() == 0:
                continue

            for parameter_name in kwargs.keys() & batch_names:
                _validate_deletion_stable_batch_override(
                    kwargs[parameter_name],
                    context=(
                        f"Mechanism {mechanism.__name__!r} insertion record "
                        f"{record_number} parameter {parameter_name!r}"
                    ),
                )

            projected_kwargs = kwargs
            if not bool(torch.all(value_keep)):
                projected_kwargs = {
                    name: (
                        _project_indexed_override(
                            value,
                            value_domain,
                            value_keep,
                            core_shape,
                            context=(
                                f"Mechanism {mechanism.__name__!r} insertion "
                                f"record {record_number} parameter {name!r}"
                            ),
                            logical_shape=logical_shape,
                        )
                        if name in indexed_names
                        else value
                    )
                    for name, value in kwargs.items()
                }
            planned_records.append(
                (
                    alias,
                    projected_kwargs,
                    _core_key_from_flat(retained, core_shape),
                    preserve,
                    copies,
                )
            )

        if (
            planned_records
            and planned_sparse_ic is not None
            and old_sparse_domain.numel()
        ):
            sparse_keep = ~torch.isin(old_sparse_domain, requested)
            if not bool(torch.all(sparse_keep)):
                planned_sparse_ic = {
                    name: _project_indexed_override(
                        value,
                        old_sparse_domain,
                        sparse_keep,
                        core_shape,
                        context=(
                            f"Mechanism {mechanism.__name__!r} sparse initial "
                            f"condition {name!r}"
                        ),
                    )
                    for name, value in planned_sparse_ic.items()
                }

        planned_parametrizations = []
        for record_number, record in enumerate(self._slice_mechanism_parametrizations):
            if not record_matches(record):
                planned_parametrizations.append(record)
                continue

            old_indices = torch.as_tensor(
                record["core_indices"], dtype=torch.long, device="cpu"
            ).reshape(-1)
            keep = torch.isin(old_indices, remaining_support)
            if not bool(torch.any(keep)):
                continue
            if record["name"] in batch_names:
                _validate_deletion_stable_batch_override(
                    record["value"],
                    context=(
                        f"Slice parametrization {record_number} for mechanism "
                        f"{record['mechanism_name']!r}, parameter "
                        f"{record['name']!r}"
                    ),
                )
            if bool(torch.all(keep)):
                planned_parametrizations.append(record)
                continue

            projected = dict(record)
            if record["name"] in indexed_names:
                projected["value"] = _project_indexed_override(
                    record["value"],
                    old_indices,
                    keep,
                    core_shape,
                    context=(
                        f"Slice parametrization {record_number} for mechanism "
                        f"{record['mechanism_name']!r}, parameter {record['name']!r}"
                    ),
                )
            projected["core_indices"] = old_indices[keep].clone()
            projected["mechanism_class"] = mechanism
            planned_parametrizations.append(projected)

        # Commit only after support and every affected value have been validated.
        if planned_everywhere is None:
            self._mech_everywhere.pop(mechanism, None)
            self._mech_exclusions.pop(mechanism, None)
        else:
            self._mech_everywhere[mechanism] = planned_everywhere
            if planned_exclusions is None or planned_exclusions.numel() == 0:
                self._mech_exclusions.pop(mechanism, None)
            else:
                self._mech_exclusions[mechanism] = planned_exclusions.clone()

        if planned_records:
            self._mech_data[mechanism] = planned_records
            if planned_sparse_ic is not None:
                self._mech_data_ic[mechanism] = planned_sparse_ic
        else:
            self._mech_data.pop(mechanism, None)
            if planned_everywhere is None:
                self._mech_data_ic.pop(mechanism, None)
                self._mech_data_base_kwargs.pop(mechanism, None)
        self._slice_mechanism_parametrizations = planned_parametrizations
        return True

    def _register_slice_mechanism_parametrization(
        self,
        mechanism_name,
        mechanism_class,
        name,
        value,
        core_indices,
        core_shape,
        alias,
    ):
        """Persist a compiled-mechanism Slice override across rebuilds."""
        self._slice_mechanism_parametrizations.append(
            {
                "mechanism_name": str(mechanism_name),
                "mechanism_class": mechanism_class,
                "name": str(name),
                "value": value,
                "core_indices": torch.as_tensor(
                    core_indices, dtype=torch.long, device="cpu"
                )
                .reshape(-1)
                .clone(),
                "core_shape": tuple(int(size) for size in core_shape),
                "alias": alias,
            }
        )

    def _apply_slice_mechanism_parametrizations(self):
        """Replay persistent Slice overrides on the current mechanism tree."""
        if not self._slice_mechanism_parametrizations:
            return

        core_shape = tuple(self.core_shape())
        mechanisms = self.mech.mechanisms
        for record in self._slice_mechanism_parametrizations:
            recorded_shape = tuple(record["core_shape"])
            if recorded_shape != core_shape:
                raise RuntimeError(
                    "A Slice-scoped mechanism parametrization was created for "
                    f"population core shape {recorded_shape}, but the current "
                    f"core shape is {core_shape}. Recreate the parametrization "
                    "after changing model topology."
                )

            mechanism_name = record["mechanism_name"]
            if mechanism_name not in mechanisms:
                raise RuntimeError(
                    "Cannot restore Slice-scoped parametrization for missing "
                    f"mechanism {mechanism_name!r}."
                )

            core_indices = record["core_indices"].to(device=self.device())
            core_key = torch.unravel_index(core_indices, core_shape)
            spec = parse_key(core_key, core_shape, device=self.device())
            mechanism = mechanisms[mechanism_name]
            mechanism_slice = Slice(
                mechanism,
                spec,
                base_shape=core_shape,
                root_model=self,
                module_path=("mech", "mechanisms", mechanism_name),
            )
            key = mechanism_slice._parameter_key(mechanism, record["name"])
            mechanism.parametrize(
                record["name"],
                record["value"],
                key=key,
                alias=record["alias"],
            )

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

        def register_material_fields(target, declarations):
            for declared_name, fields in declarations.items():
                material_name = _canonical_material_name(declared_name)
                current = target.setdefault(material_name, {}).setdefault(name, [])
                current.extend(field for field in fields if field not in current)

        def register_material_sources(target, declarations):
            for declared_name, field_map in declarations.items():
                material_name = _canonical_material_name(declared_name)
                target.setdefault(material_name, {}).setdefault(name, {}).update(
                    field_map
                )

        if not is_material_process:
            for k, v in mech._currents.items():
                self._m_curr.setdefault(k, {}).update({name: v})

            for k, v in mech._read_ion.items():
                self._ion_read.setdefault(k, {}).update({name: v})

            for k, v in mech._write_ion.items():
                self._ion_write.setdefault(k, {}).update({name: v})

            for k, v in mech._write_ion_c.items():
                self._ion_write_c.setdefault(k, {}).update({name: v})

            register_material_fields(
                self._material_read, getattr(mech, "_read_material", {})
            )
            register_material_fields(
                self._material_write, getattr(mech, "_write_material", {})
            )
            register_material_sources(
                self._material_source, getattr(mech, "_source_material", {})
            )
        else:
            register_material_fields(
                self._material_process_read, getattr(mech, "_read_material", {})
            )
            register_material_fields(
                self._material_process_write, getattr(mech, "_write_material", {})
            )
            register_material_sources(
                self._material_process_source, getattr(mech, "_source_material", {})
            )

    def _reset_compiled_mechanism_registries(self):
        """Clear metadata derived exclusively from the pending insertion config."""
        self._m_list = []
        self._m_name = []
        self._m_keys = []
        self._m_curr = {}
        self._m_shape = {}

        self._ion_read = {}
        self._ion_write = {}
        self._ion_write_c = {}

        self._material_read = {}
        self._material_write = {}
        self._material_source = {}
        self._material_process_read = {}
        self._material_process_write = {}
        self._material_process_source = {}

        self._all_read = {}
        self._all_write = {}
        self._all_write_c = {}

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
        Move the population to a new device or dtype.

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

        Notes
        -----
        Changing device or dtype after initialization moves mechanism scratch,
        rebuilds point-process scaling closures, and invalidates derived solver
        coefficients. Direct Population execution rebuilds the solver workspace
        lazily. If this population belongs to an already-built Network,
        reinitialize the Network before execution so all owner state is current.
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

        # Every entry below is derived from _mech_everywhere/_mech_data. A
        # rebuild must start from a blank registry; appending to the previous
        # graph leaves removed mechanisms, currents, ions, and materials live.
        self._reset_compiled_mechanism_registries()

        canonical_configs = {}
        configured_names = {}
        for configured_name, config in self._material_configs.items():
            canonical = _canonical_material_name(configured_name)
            if canonical in canonical_configs:
                previous = configured_names[canonical]
                raise ValueError(
                    f"Material configurations {previous!r} and {configured_name!r} "
                    f"both resolve to canonical material {canonical!r}. Configure "
                    "that material through only one name."
                )
            canonical_configs[canonical] = config
            configured_names[canonical] = configured_name
        self._material_configs = canonical_configs

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
                excluded = torch.as_tensor(
                    self._mech_exclusions.get(mech, []), dtype=torch.long
                ).reshape(-1)
                if excluded.numel() == 0:
                    key = None
                    shape = self._calc_shape_p()
                    shape_f = self.shape
                    if hasattr(mech, "normalize_random_kwargs"):
                        kwargs = mech.normalize_random_kwargs(kwargs)
                    mech.check_kwargs(kwargs)
                    m = mech(
                        name,
                        self.celsius,
                        self.diam,
                        shape,
                        shape_f,
                        key,
                        ic=ic,
                        **kwargs,
                    )
                else:
                    all_indices = torch.arange(
                        math.prod(self.core_shape()), dtype=torch.long
                    )
                    remaining = all_indices[~torch.isin(all_indices, excluded)]
                    if remaining.numel() == 0:
                        raise RuntimeError(
                            f"Mechanism {mech.__name__!r} has no remaining "
                            "support but is still configured for insertion."
                        )
                    remaining_key = _core_key_from_flat(
                        remaining, tuple(self.core_shape())
                    )
                    m, shape, key = compile_mechanism(
                        self,
                        mech,
                        [remaining_key],
                        [None],
                        [{}],
                        ic=ic,
                        base_kwargs=kwargs,
                        name=name,
                        force_flat=True,
                    )
                self._register_mech(m, shape, key)

            for mech, data in self._mech_data.items():
                aliases = []
                kwargs_list = []
                keys = []
                preserve_duplicate_indices = []
                copies_list = []
                for record in data:
                    alias, kwargs, key, preserve, copies = (
                        _unpack_mechanism_insertion_record(record)
                    )
                    aliases.append(alias)
                    kwargs_list.append(kwargs)
                    keys.append(key)
                    preserve_duplicate_indices.append(
                        bool(preserve or int(copies) != 1)
                    )
                    copies_list.append(int(copies))
                if not are_strings_unique(aliases):
                    raise ValueError(
                        f"Duplicate aliases found for mechanism {mech.__name__}."
                    )
                m, shape, key = compile_mechanism(
                    self,
                    mech,
                    keys,
                    aliases,
                    kwargs_list,
                    preserve_duplicate_indices=preserve_duplicate_indices,
                    copies=copies_list,
                    ic=self._mech_data_ic.get(mech),
                    base_kwargs=self._mech_data_base_kwargs.get(mech),
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
            self._dispatch_mechanism_injections(mech, reset_acceptance=True)

            self.integrator = self._integrator_class(self, mech, imem=self.imem)
            # The population may have entered train/eval mode before this lazy
            # child hierarchy existed. Newly attached modules otherwise keep
            # torch.nn.Module's default training=True state.
            self.integrator.train(self.training)
            self.integrator.configure_jit(self, scope="population")
            self.mech = self.integrator.mech
            self._apply_slice_mechanism_parametrizations()

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
        workspace_signature = self._integrator_workspace_contract_signature()
        workspace_was_valid = bool(
            workspace_signature is not None
            and self.integrator.initialized
            and workspace_signature
            == getattr(self, "_validated_integrator_workspace_signature", None)
            and self._unversioned_integrator_workspace_values_match(workspace_signature)
        )
        self.integrator.detach(self)
        if workspace_was_valid:
            # Detach is value preserving but rebinds tensors, so rebase only a
            # workspace that was coherent before the operation. Never bless a
            # pre-existing stale dependency accidentally.
            self._record_integrator_workspace_contracts()
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
            If True, use token-aware glob matching. Punctuation, including
            underscores, dots, brackets, and parentheses, separates name tokens;
            ``*`` explicitly matches zero or more characters. Default is True.
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
            ...,
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
            If True, use token-aware glob matching. Punctuation, including
            underscores, dots, brackets, and parentheses, separates name tokens;
            ``*`` explicitly matches zero or more characters. Default is True.
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
            If True, use token-aware glob matching. For example, ``"soma"``
            matches ``"MelnickSG_soma(0.5)"`` and ``"Cell.soma[0](0.5)"``,
            while ``"soma[0]"`` matches only that literal indexed section.
            Adjacent letters or digits are not delimiters, so ``"myelin"`` does
            not match ``"unmyelin"`` or ``"myelinated"``. Use ``"*myelin"``
            to request the broader suffix match explicitly. Default is True.
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

        def _batch_injection_specs(specs):
            """Rebase stored slice indices onto the new leading batch axis."""
            batched = []
            for waveform, _, index in specs:
                current_index = index if isinstance(index, tuple) else (index,)
                new_index = (
                    current_index
                    if current_index and current_index[0] is Ellipsis
                    else (slice(None),) + current_index
                )
                selected_shape = tuple(self.v[new_index].shape)
                batched.append((waveform, selected_shape, new_index))
            return batched

        # Injection specs capture their index tuple when inject() is called;
        # unlike named Slice objects, they therefore cannot update themselves.
        # Promote both solver and mechanism registries so inject-then-batch and
        # batch-then-inject address the same replicated compartments.
        self.injections = _batch_injection_specs(self.injections)
        self.mechanism_injections = _batch_injection_specs(self.mechanism_injections)
        self.intra = None

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
        for population_slice in self._labels.values():
            population_slice._batch()
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
    def _stochastic_state_dict_for_checkpoint(self):
        """Snapshot Population-level RAND/NOISE buffers and RNG streams."""
        buffer_names = []
        rng_names = []

        for name in self.__class__._rng:
            rng_names.append(name)
        for specs in (self.random_parameters, self.runtime_noises):
            for name, spec in specs.items():
                buffer_names.append(name)
                rng_names.append(spec.effective_rng_name)

        buffers = {}
        for name in dict.fromkeys(buffer_names):
            value = getattr(self, name)
            if not torch.is_tensor(value):
                raise TypeError(
                    f"Population stochastic buffer {name!r} must be a tensor."
                )
            buffers[name] = value.clone()

        rng_states = {}
        for name in dict.fromkeys(rng_names):
            rng = getattr(self, name)
            if not hasattr(rng, "rng_state"):
                raise TypeError(
                    f"Population stochastic generator {name!r} does not expose "
                    "rng_state()."
                )
            rng_states[name] = {
                "base_seed": int(rng._base_seed),
                "rng_state": _clone_nested_state(rng.rng_state(), detach_tensors=True),
            }

        return {"buffers": buffers, "rng_states": rng_states}

    def _prepare_stochastic_state_from_checkpoint(self, state_dict):
        """Validate and normalize Population stochastic checkpoint state."""

        if not isinstance(state_dict, Mapping):
            raise TypeError("Population stochastic checkpoint state must be a mapping.")
        if "buffers" not in state_dict or "rng_states" not in state_dict:
            raise KeyError(
                "Population stochastic checkpoint requires 'buffers' and "
                "'rng_states' entries."
            )

        expected = self._stochastic_state_dict_for_checkpoint()
        buffers = state_dict["buffers"]
        rng_states = state_dict["rng_states"]
        if not isinstance(buffers, Mapping) or not isinstance(rng_states, Mapping):
            raise TypeError(
                "Population stochastic checkpoint buffers and RNG states must be mappings."
            )
        if set(buffers) != set(expected["buffers"]):
            raise ValueError(
                "Population stochastic checkpoint buffer names do not match this "
                f"model: expected {sorted(expected['buffers'])}, got {sorted(buffers)}."
            )
        if set(rng_states) != set(expected["rng_states"]):
            raise ValueError(
                "Population stochastic checkpoint RNG names do not match this "
                f"model: expected {sorted(expected['rng_states'])}, "
                f"got {sorted(rng_states)}."
            )

        prepared_buffers = {}
        for name, candidate in buffers.items():
            reference = getattr(self, name)
            if not torch.is_tensor(candidate):
                raise TypeError(
                    f"Population stochastic checkpoint buffer {name!r} must be a tensor."
                )
            if tuple(candidate.shape) != tuple(reference.shape):
                raise ValueError(
                    f"Population stochastic checkpoint buffer {name!r} has shape "
                    f"{tuple(candidate.shape)}, expected {tuple(reference.shape)}."
                )
            prepared_buffers[name] = candidate.to(
                device=reference.device, dtype=reference.dtype
            ).clone()

        prepared_rng_states = {}
        for name, rng_payload in rng_states.items():
            try:
                prepared_rng_states[name] = _validate_rng_checkpoint_payload(
                    rng_payload, allow_legacy=True
                )
            except (KeyError, RuntimeError, TypeError, ValueError) as error:
                raise type(error)(
                    f"Invalid Population stochastic RNG payload {name!r}: {error}"
                ) from error

        return prepared_buffers, prepared_rng_states

    def _apply_prepared_stochastic_state(self, prepared_state):
        """Apply state returned by
        :meth:`_prepare_stochastic_state_from_checkpoint`.
        """

        buffers, rng_states = prepared_state
        for name, restored in buffers.items():
            setattr(self, name, restored)
        for name, (base_seed, rng_state) in rng_states.items():
            rng = getattr(self, name)
            if base_seed is None:
                # Early checkpoint payloads stored only the raw device-state
                # mapping. Preserve the target's constructor seed for reset_rng().
                rng.set_rng_state(rng_state)
            else:
                rng.set_extra_state({"base_seed": base_seed, "rng_state": rng_state})

    def _restore_stochastic_state_from_checkpoint(self, state_dict):
        """Restore a Population stochastic checkpoint payload."""

        prepared = self._prepare_stochastic_state_from_checkpoint(state_dict)
        self._apply_prepared_stochastic_state(prepared)

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
            "duration_remainder": self._duration_remainder,
            "stochastic": self._stochastic_state_dict_for_checkpoint(),
        }
        return full_dct

    def _restore_dict_from_checkpoint_unchecked(self, state_dict):
        if not isinstance(state_dict, Mapping):
            raise TypeError("Population checkpoint state must be a mapping.")
        missing = [
            name for name in ("mech", "integrator", "t") if name not in state_dict
        ]
        if missing:
            raise KeyError(f"Population checkpoint is missing entries: {missing}.")
        if not isinstance(state_dict["integrator"], Mapping):
            raise TypeError("Population checkpoint integrator state must be a mapping.")

        expected_integrator = self.integrator.mutable_state_dict(self)
        restored_integrator = {}
        for name, reference in expected_integrator.items():
            if name not in state_dict["integrator"]:
                raise KeyError(
                    f"Population checkpoint integrator state is missing {name!r}."
                )
            candidate = state_dict["integrator"][name]
            if torch.is_tensor(reference):
                if not torch.is_tensor(candidate):
                    raise TypeError(
                        f"Population checkpoint integrator state {name!r} must be a tensor."
                    )
                if tuple(candidate.shape) != tuple(reference.shape):
                    raise ValueError(
                        f"Population checkpoint integrator state {name!r} has shape "
                        f"{tuple(candidate.shape)}, expected {tuple(reference.shape)}."
                    )
                candidate = candidate.to(
                    device=reference.device, dtype=reference.dtype
                ).clone()
            else:
                candidate = copy.deepcopy(candidate)
            restored_integrator[name] = candidate

        t = state_dict["t"]
        if not torch.is_tensor(t):
            raise TypeError("Population checkpoint time must be a tensor.")
        if tuple(t.shape) != tuple(self.t.shape):
            raise ValueError(
                f"Population checkpoint time has shape {tuple(t.shape)}, "
                f"expected {tuple(self.t.shape)}."
            )

        # Runtime checkpoints created before fractional-duration carry support
        # have no entry; restoring them starts with no pending physical time.
        if "duration_remainder" not in state_dict:
            duration_remainder = torch.zeros(
                (), device=self._duration_remainder.device, dtype=torch.float64
            )
        else:
            duration_remainder = state_dict["duration_remainder"]
            if not torch.is_tensor(duration_remainder):
                raise TypeError(
                    "Population checkpoint duration_remainder must be a tensor."
                )
            if tuple(duration_remainder.shape) != ():
                raise ValueError(
                    "Population checkpoint duration_remainder must be a scalar tensor."
                )
            if not torch.is_floating_point(duration_remainder):
                raise TypeError(
                    "Population checkpoint duration_remainder must have floating dtype."
                )
            duration_remainder = duration_remainder.to(
                device=self._duration_remainder.device, dtype=torch.float64
            ).clone()
            duration_value = float(duration_remainder.detach().cpu().item())
            if not math.isfinite(duration_value) or duration_value < 0.0:
                raise ValueError(
                    "Population checkpoint duration_remainder must be finite and "
                    "non-negative."
                )

        # Preflight stochastic metadata before any mechanism or integrator
        # field is changed. Older checkpoints remain loadable only when the
        # target Population has no Population-level stochastic declarations.
        prepared_stochastic = None
        if "stochastic" in state_dict:
            prepared_stochastic = self._prepare_stochastic_state_from_checkpoint(
                state_dict["stochastic"]
            )
        elif any(
            (
                self.__class__._rng,
                self.random_parameters,
                self.runtime_noises,
            )
        ):
            raise KeyError(
                "Population checkpoint predates stochastic-state support but this "
                "model declares Population-level RNG/RAND/NOISE state."
            )

        self.mech.restore_mutable_state_dict(state_dict["mech"])
        self.integrator.restore_mutable_state_dict(self, restored_integrator)
        if prepared_stochastic is not None:
            self._apply_prepared_stochastic_state(prepared_stochastic)
        self.t = t.to(device=self.t.device, dtype=self.t.dtype).clone()
        self._duration_remainder = duration_remainder

    def restore_dict_from_checkpoint(self, state_dict):
        """
        Restore model state from a checkpoint state dictionary.

        Parameters
        ----------
        state_dict : dict
            State dictionary containing model parameters and buffers.
        """
        previous = self.state_dict_for_checkpoint()
        try:
            self._restore_dict_from_checkpoint_unchecked(state_dict)
        except Exception:
            try:
                self._restore_dict_from_checkpoint_unchecked(previous)
            except Exception as rollback_error:
                raise RuntimeError(
                    "Population checkpoint restore failed and rollback could not "
                    "recover the previous runtime state."
                ) from rollback_error
            raise
        return self

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
            Simulation duration in milliseconds to advance from the current
            model time ``self.t``. This uses the same cumulative complete-step
            budget and retained physical-duration semantics as :meth:`longrun`.
        chunklength : int
            Number of time steps per checkpointed chunk.
        dt : float, optional
            Time step size in milliseconds. If None, uses the global default
            ``A.dt``.
        extra : ExtraSpec, optional
            Extracellular configuration with the same ``(ve_s, time)`` shape
            and trailing-first broadcasting contract as :meth:`longrun`.
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
        self._validate_runtime_contracts(
            workspace_rebuild_pending=self._runtime_workspace_rebuild_pending()
        )
        if not isinstance(chunklength, int) or isinstance(chunklength, bool):
            raise ValueError("chunklength must be a positive integer.")
        if chunklength <= 0:
            raise ValueError("chunklength must be a positive integer.")
        dt_f = _validate_time_scalar(
            A.dt if dt is None else dt, name="dt", positive=True
        )
        tstop_f = _validate_time_scalar(tstop, name="tstop", positive=False)
        n_steps, duration_remainder = self._duration_budget(tstop_f, dt_f)
        self._refresh_compile_config_from_ctx()

        # Match step()/run(): injections registered after initialization
        # invalidate ``self.intra`` and are rebuilt lazily on first use.
        if self.intra is None:
            self.intra = self.build_intra()
        intra = self.intra
        with_intra = intra is not None

        # dt scalars
        dt_tensor = torch.tensor(dt_f, device=self.device(), dtype=self.dtype())

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
            return _clone_nested_state(sd)

        def _copy_state_containers(sd: Dict[str, Any]) -> Dict[str, Any]:
            """
            Copy only nested containers (not tensors). Useful to protect the
            backward-restore hook from accidental user mutation of the returned dict.
            """

            def _copy(value):
                if isinstance(value, Mapping):
                    return {key: _copy(item) for key, item in value.items()}
                if isinstance(value, list):
                    return [_copy(item) for item in value]
                if isinstance(value, tuple):
                    return tuple(_copy(item) for item in value)
                return value

            return _copy(sd)

        with torch.nn.utils.parametrize.cached():
            with torch.set_grad_enabled(self.training):
                # --------------------------------------------------------------
                # Global time grid and chunking
                # --------------------------------------------------------------
                t = _time_grid_from_step_count(
                    self.t, n_steps, dt_f, device=self.device()
                ).to(self.dtype())

                if t.numel() == 0:
                    self._set_duration_remainder(duration_remainder)
                    if return_final_state:
                        return None, self.state_dict_for_checkpoint()
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

                # Commit before the first checkpoint boundary is captured so
                # forward replay and backward restoration observe one coherent
                # duration budget.
                self._set_duration_remainder(duration_remainder)
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
                            # Waveform output always carries time on its last
                            # axis; leading axes are population batch/sweep axes.
                            stims = [s.unbind(-1) for s in stims]
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
                                intra_c = self._call_make_intra(intra, s, indices)
                            else:
                                intra_c = None

                            self._step(self.integrator, self, dt_tensor, ve_c, intra_c)
                            self.t = self.t + dt_tensor

                            # Replay-safe post-step callbacks (may contribute loss)
                            if callbacks:
                                for c in callbacks:
                                    hook = getattr(c, "post_step_hook", None)
                                    if hook is None:
                                        continue
                                    chunk_loss, saw = _add_loss(chunk_loss, hook(self))
                                    saw_loss_local = saw_loss_local or saw

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
            # regions: List[(alias, kwargs, key[, preserve_duplicate_indices[, copies]])]
            lines.append(f"  - {mech_cls.__name__} @ {len(regions)} region(s):")
            for record in regions[: self._REPR_MAX_MECHS]:
                alias, kwargs, key = record[:3]
                preserve = bool(record[3]) if len(record) >= 4 else False
                copies = int(record[4]) if len(record) >= 5 else 1
                nm = alias if alias is not None else mech_cls.__name__
                kw = f" {self._fmt_kwargs(kwargs)}" if (show_kwargs and kwargs) else ""
                extra = []
                if preserve:
                    extra.append("preserve_multiplicity=True")
                if copies != 1:
                    extra.append(f"copies={copies}")
                extra_s = (" [" + ", ".join(extra) + "]") if extra else ""
                lines.append(f"      • {nm} @ {self._fmt_index(key)}{extra_s}{kw}")
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
        """Return a human-friendly, block-formatted summary.

        For example::

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


_MORPHOLOGY_ALNUM = r"A-Za-z0-9"


def _is_ascii_alnum(char: str) -> bool:
    """Return whether *char* participates in a morphology identifier token."""
    return len(char) == 1 and char.isascii() and char.isalnum()


def _glob_body_regex(pattern: str) -> str:
    """Escape a find pattern while retaining ``*`` as a glob wildcard."""
    return ".*".join(re.escape(part) for part in pattern.split("*"))


def _fuzzy_name_regex(pattern: str) -> str:
    """Build a token-aware glob regex for morphology/name searches.

    Dendra names commonly combine an owning object, a section name, an optional
    NEURON section-array index, and a segment location, for example::

        MelnickSG_soma(0.5)
        Cell[0].soma[0](0.5)

    Python's ``\b`` treats an underscore as a word character. Consequently, a
    query such as ``"soma"`` used not to match ``"MelnickSG_soma(0.5)"`` even
    though an underscore is a structural delimiter in these names. Use explicit
    alphanumeric boundaries instead: punctuation (including ``_``, ``.``, ``[``,
    ``]``, ``(``, and ``)``) separates tokens, while adjacent letters or digits do
    not.

    All characters except ``*`` are literal. An asterisk is an explicit glob
    wildcard matching zero or more characters. Thus ``"myelin"`` remains
    distinct from ``"unmyelin"``, while ``"*myelin"`` intentionally matches
    both suffixes. Selectors such as ``"soma[0]"`` keep their brackets literal.
    """
    if not isinstance(pattern, str):
        raise TypeError(
            "Compartment search patterns must be strings; "
            f"received {type(pattern).__name__}."
        )
    if not pattern:
        return ""

    body = _glob_body_regex(pattern)

    # A leading wildcard deliberately permits an alphanumeric prefix (for
    # example, ``*myelin`` matching ``unmyelin``). Otherwise, protect selectors
    # that begin like identifiers from matching inside a larger identifier.
    left = (
        rf"(?<![{_MORPHOLOGY_ALNUM}])"
        if pattern[0] != "*" and _is_ascii_alnum(pattern[0])
        else ""
    )

    # Likewise, a trailing wildcard deliberately permits an alphanumeric suffix.
    # Without one, require an identifier boundary after names and indexed
    # selectors so ``myelin`` does not match ``myelinated`` and ``soma[0]`` does
    # not match ``soma[0]extra``.
    right = (
        rf"(?![{_MORPHOLOGY_ALNUM}])"
        if pattern[-1] != "*" and (_is_ascii_alnum(pattern[-1]) or pattern[-1] in "])")
        else ""
    )
    return f"{left}{body}{right}"


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

    When ``fuzzy=True``, matching is morphology-token aware and supports
    ``*`` as an explicit glob wildcard:

    - ``"soma"`` matches ``"soma"``, ``"MelnickSG_soma(0.5)"``, and
      ``"Cell[0].soma[0](0.5)"``, but not ``"presoma"`` or ``"somatic"``.
    - ``"soma[0]"`` matches the literal indexed section and not ``"soma[1]"``.
    - ``"myelin"`` does not match ``"unmyelin"``, ``"demyelin"``, or
      ``"myelinated"`` because adjacent letters and digits remain part of the
      same identifier token.
    - ``"*myelin"`` deliberately matches suffixes such as ``"myelin"`` and
      ``"unmyelin"``; ``"myelin*"`` deliberately permits a suffix.
    - Apart from ``*``, selector characters are literal, and punctuation
      delimits tokens.

    Parameters
    ----------
    data (List[str]):
        The list of strings to search through.
    include (Optional[Union[str, List[str]]]):
        Patterns to include.
    exclude (Optional[Union[str, List[str]]]):
        Patterns to exclude.
    fuzzy (bool):
        If True, performs token-aware glob matching with ``*`` as the only
        wildcard. If False, requires an exact full-string match.
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
            rgx = _fuzzy_name_regex(pat)
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


class Cable(Population):
    """Generic unbranched cable using Dendra's tridiagonal fast path.

    ``Cable`` is the geometry-general counterpart to :class:`Axon`.  It can
    represent any material-only native Section morphology whose compiled
    electrical graph is one simple path, including tapered pt3d Sections and
    heterogeneous Section membrane properties.  Exact compiled membrane areas
    and axial edge resistances remain the numerical source of truth.

    Construct native cables with :meth:`from_morphology` or
    :meth:`from_compartment_graph`.  The ordinary constructor is primarily the
    shared implementation base for specialized path models such as
    :class:`Axon`.
    """

    _supports_native_cable_factory = True

    _CANONICAL_RUNTIME_CONTRACT_TENSORS = (
        "diam",
        "dx",
        "rhoa",
        "volume",
        "volume_um3",
        "volume_i",
        "volume_o",
        "diff_geom_um",
        "_canonical_area_cm2",
        "_canonical_edge_resistance_ohm",
        "diff_parent_index",
        "_canonical_morphology_fingerprint",
        "_canonical_geometry_reference",
        "rhoa_scale",
    )

    _SOLVER_WORKSPACE_CONTRACT_TENSORS = (
        "cm",
        "cm_scale",
        "area_scale",
        "rhoa_scale",
        "diam",
        "dx",
        "rhoa",
    )

    _UNVERSIONED_WORKSPACE_VALUE_TENSORS = (
        "cm",
        "cm_scale",
        "area_scale",
        "rhoa_scale",
    )

    def __init__(self, N, C, graph=None, integrator=None, **kwargs):
        if graph is not None:
            raise TypeError(
                "Cable graph topology cannot be supplied to the low-level "
                "constructor. Use Cable.from_morphology(...) or "
                "Cable.from_compartment_graph(...) so path order and exact "
                "geometry are validated together."
            )
        if integrator is None:
            integrator = bwd_euler_ub()
        if self._supports_native_cable_factory and not getattr(
            integrator, "supports_unbranched_cable", False
        ):
            raise TypeError(
                "Generic Cable requires an integrator that declares "
                "supports_unbranched_cable=True. Use bwd_euler_ub(...) for "
                "the tridiagonal fast path or dhs(...) for the graph solver."
            )
        super().__init__(N, C, integrator=integrator, **kwargs)
        self._graph = None
        self._compartment_graph = None
        self.register_buffer("_canonical_area_cm2", None)
        self.register_buffer("_canonical_edge_resistance_ohm", None)
        self.register_buffer("_canonical_morphology_fingerprint", None)
        self.register_buffer("_canonical_geometry_reference", None)
        self._canonical_morphology_fingerprint_reference = None
        self._validated_runtime_contract_signature = None
        self._validated_integrator_workspace_signature = None
        self._validated_unversioned_workspace_values = None

    def __getstate__(self):
        """Serialize without process-local tensor identity/version metadata."""
        state = super().__getstate__()
        current_workspace = self._integrator_workspace_contract_signature()
        integrator = getattr(self, "integrator", None)
        state["_serialized_integrator_workspace_contract_valid"] = bool(
            current_workspace is not None
            and integrator is not None
            and integrator.initialized
            and current_workspace
            == getattr(self, "_validated_integrator_workspace_signature", None)
            and self._unversioned_integrator_workspace_values_match(current_workspace)
        )
        state["_validated_runtime_contract_signature"] = None
        state["_validated_integrator_workspace_signature"] = None
        state["_validated_unversioned_workspace_values"] = None
        return state

    def __setstate__(self, state):
        """Rebase valid process-local workspace metadata after deserialization."""
        workspace_was_valid = bool(
            state.pop("_serialized_integrator_workspace_contract_valid", False)
        )
        super().__setstate__(state)
        self._validated_runtime_contract_signature = None
        self._validated_integrator_workspace_signature = None
        self._validated_unversioned_workspace_values = None
        if workspace_was_valid:
            self._record_integrator_workspace_contracts()

    @staticmethod
    def _fingerprint_compartment_graph(graph) -> bytes:
        """Return a stable digest of canonical topology, geometry, and metadata."""
        geometry_fields = tuple(
            (
                name,
                tuple(float(value).hex() for value in getattr(graph.geometry, name)),
            )
            for name in graph.geometry.__dataclass_fields__
        )
        metadata_fields = (
            ("name", tuple(graph.metadata.name)),
            ("kind", tuple(graph.metadata.kind)),
            ("section_name", tuple(graph.metadata.section_name)),
            ("segment_index", tuple(graph.metadata.segment_index)),
            (
                "section_x",
                tuple(
                    None if value is None else float(value).hex()
                    for value in graph.metadata.section_x
                ),
            ),
            (
                "labels",
                tuple(tuple(sorted(labels)) for labels in graph.metadata.labels),
            ),
        )
        payload = (
            int(graph.schema_version),
            tuple(graph.topology.parent_index),
            geometry_fields,
            metadata_fields,
        )
        return hashlib.sha256(repr(payload).encode("utf-8")).digest()

    def _tensor_contract_signature(self, names, *, require_versions):
        """Return identity/version/storage metadata for the named tensors."""
        signature = []
        for name in names:
            value = getattr(self, name, None)
            if not torch.is_tensor(value):
                return None
            try:
                version = value._version
            except RuntimeError:
                # Tensors created under torch.inference_mode() deliberately do
                # not expose version counters. Static versioned validation must
                # fall back to a full scan. Workspace metadata can still prove
                # identity/device/dtype stability. Exact value snapshots cover
                # ordinary inference-tensor mutations; only writes through
                # raw aliases remain outside the supported contract.
                if require_versions:
                    return None
                version = None
            metadata = (
                tuple(value.shape),
                tuple(value.stride()),
                value.storage_offset(),
                value.dtype,
                value.device,
                value.layout,
                value.data_ptr(),
            )
            signature.append((name, id(value), version, metadata))
        return tuple(signature)

    def _runtime_contract_signature(self):
        """Return a cheap change signature, or ``None`` without version counters."""
        return self._tensor_contract_signature(
            self._CANONICAL_RUNTIME_CONTRACT_TENSORS,
            require_versions=True,
        )

    def _integrator_workspace_contract_signature(self):
        """Track native Cable inputs captured in exact-edge solver coefficients."""
        if self._compartment_graph is None:
            # Legacy Axon subclasses expose parametrized geometry whose public
            # tensors may be reconstructed out of place on access. Their
            # established training/reinitialization lifecycle remains separate
            # from the immutable native-Cable contract implemented here.
            return None
        return self._tensor_contract_signature(
            self._SOLVER_WORKSPACE_CONTRACT_TENSORS,
            require_versions=False,
        )

    def _record_unversioned_integrator_workspace_values(self, signature):
        """Snapshot mutable workspace inputs only when versions are unavailable."""
        if any(entry[2] is None for entry in signature):
            self._validated_unversioned_workspace_values = tuple(
                (name, getattr(self, name).detach().clone())
                for name in self._UNVERSIONED_WORKSPACE_VALUE_TENSORS
            )
        else:
            self._validated_unversioned_workspace_values = None

    def _unversioned_integrator_workspace_values_match(self, signature=None):
        """Compare inference-tensor dependencies against their built values."""
        if signature is None:
            signature = self._integrator_workspace_contract_signature()
        if signature is None or not any(entry[2] is None for entry in signature):
            return True
        snapshots = self._validated_unversioned_workspace_values
        if snapshots is None:
            return False
        return all(
            torch.equal(getattr(self, name), expected) for name, expected in snapshots
        )

    def _validate_canonical_geometry(self):
        """Fail closed if immutable compiled geometry has been edited in place."""
        if self._compartment_graph is None:
            return
        reference = self._canonical_geometry_reference
        if reference is None:
            raise RuntimeError("Native Cable canonical geometry reference is missing.")

        checks = (
            ("diam", self.diam),
            ("dx", self.dx),
            ("rhoa", self.rhoa),
            ("volume", self.volume),
            ("volume_um3", self.volume_um3),
            ("volume_i", self.volume_i),
            ("volume_o", self.volume_o),
            ("diff_geom_um", self.diff_geom_um),
            ("_canonical_area_cm2", self._canonical_area_cm2),
            ("_canonical_edge_resistance_ohm", self._canonical_edge_resistance_ohm),
        )
        if reference.shape[0] != len(checks):
            raise RuntimeError("Native Cable canonical geometry reference is invalid.")
        for index, (name, actual) in enumerate(checks):
            try:
                actual_view = torch.broadcast_to(actual, self.shape)
                reference_view = torch.broadcast_to(reference[index], self.shape)
            except RuntimeError as error:
                raise RuntimeError(
                    f"Native Cable compiled geometry buffer {name!r} no longer "
                    f"broadcasts to model shape {self.shape}. Recompile the "
                    "Morphology instead of editing compiled geometry."
                ) from error
            # rhoa is a bounded RANGE parameter. Cross-dtype state restoration
            # can round its internal unconstrained coordinate once before the
            # public value is reconstructed, so permit only representation-
            # scale roundoff there; all direct geometry buffers remain exact.
            equal = torch.equal(actual_view, reference_view)
            if name == "rhoa" and not equal:
                eps = torch.finfo(actual_view.dtype).eps
                equal = torch.allclose(
                    actual_view,
                    reference_view,
                    rtol=4.0 * eps,
                    atol=0.0,
                )
            if not equal:
                raise RuntimeError(
                    f"Native Cable compiled geometry buffer {name!r} was modified. "
                    "Compiled diam/dx/rhoa, area, volume, and edge geometry are "
                    "immutable; edit the Morphology and construct a new Cable."
                )

        expected_parent = torch.as_tensor(
            self._compartment_graph.topology.parent_index,
            device=self.diff_parent_index.device,
            dtype=self.diff_parent_index.dtype,
        )
        if not torch.equal(self.diff_parent_index, expected_parent):
            raise RuntimeError(
                "Native Cable compiled geometry buffer 'diff_parent_index' was "
                "modified. Edit the Morphology and construct a new Cable."
            )

        fingerprint_reference = self._canonical_morphology_fingerprint_reference
        fingerprint = self._canonical_morphology_fingerprint
        if fingerprint_reference is None:
            raise RuntimeError(
                "Native Cable canonical morphology fingerprint reference is missing."
            )
        expected_fingerprint = torch.as_tensor(
            list(fingerprint_reference),
            device=fingerprint.device,
            dtype=fingerprint.dtype,
        )
        if not torch.equal(fingerprint, expected_fingerprint):
            raise RuntimeError(
                "Native Cable canonical morphology fingerprint was modified. "
                "Edit the Morphology and construct a new Cable."
            )

        scale = torch.broadcast_to(self.rhoa_scale, self.shape).reshape(
            -1, self.shape[-1]
        )
        if scale.shape[-1] > 1 and not torch.equal(
            scale, scale[:, :1].expand_as(scale)
        ):
            raise ValueError(
                "A canonical Cable supports only spatially uniform rhoa_scale "
                "within each cable. Exact edge totals cannot recover distinct "
                "left/right half-path scaling."
            )

        # Cache only a state which has passed every value-level contract above.
        # ``None`` is intentional for inference tensors: their missing version
        # counters force a full scan at every versioned validation boundary.
        self._validated_runtime_contract_signature = self._runtime_contract_signature()

    def initialize(self, *args, **kwargs):
        """Validate frozen native geometry before entering the normal lifecycle."""
        self._validate_canonical_geometry()
        try:
            result = super().initialize(*args, **kwargs)
            self._validate_canonical_geometry()
        except Exception:
            self.initialized = False
            if self.integrator is not None:
                self.integrator.initialized = False
            raise
        return result

    def _validate_static_runtime_contracts(self, mode):
        """Validate native compiled geometry at public execution boundaries."""
        if self._compartment_graph is None:
            return

        if mode == "strict":
            self._validate_canonical_geometry()
            return

        signature = self._runtime_contract_signature()
        if signature is None or signature != self._validated_runtime_contract_signature:
            self._validate_canonical_geometry()

    def _validate_integrator_rebuild_contracts(self):
        """Fully validate geometry before rebuilding cached solver coefficients."""
        self._validate_canonical_geometry()

    def _preflight_incoming_canonical_geometry(self, state_dict, prefix):
        """Validate frozen checkpoint geometry before any tensor is copied."""
        self._validate_canonical_geometry()
        local_reference = self._canonical_geometry_reference
        reference_key = f"{prefix}_canonical_geometry_reference"
        incoming_reference = state_dict.get(reference_key)
        if (
            not torch.is_tensor(incoming_reference)
            or not torch.is_floating_point(incoming_reference)
            or tuple(incoming_reference.shape) != tuple(local_reference.shape)
        ):
            raise RuntimeError(
                "Cannot load a native Cable checkpoint with missing or corrupt "
                "canonical geometry."
            )

        # The same binary64 graph may have been deliberately materialized in a
        # different model dtype. Compare both references in the less precise of
        # the two dtypes so float32 <-> float64 restoration remains exact at the
        # intentional cast boundary, without weakening same-dtype validation.
        incoming_eps = torch.finfo(incoming_reference.dtype).eps
        local_eps = torch.finfo(local_reference.dtype).eps
        comparison_dtype = (
            incoming_reference.dtype
            if incoming_eps >= local_eps
            else local_reference.dtype
        )
        incoming_reference_native_cpu = incoming_reference.detach().cpu()
        incoming_reference_cpu = incoming_reference_native_cpu.to(
            device="cpu", dtype=comparison_dtype
        )
        local_reference_cpu = local_reference.detach().to(
            device="cpu", dtype=comparison_dtype
        )
        if not torch.equal(incoming_reference_cpu, local_reference_cpu):
            raise RuntimeError(
                "Cannot load a native Cable checkpoint with corrupt canonical "
                "geometry inconsistent with its morphology fingerprint."
            )

        checks = (
            ("diam", 0),
            ("dx", 1),
            ("rhoa", 2),
            ("volume", 3),
            ("volume_um3", 4),
            ("volume_i", 5),
            ("volume_o", 6),
            ("diff_geom_um", 7),
            ("_canonical_area_cm2", 8),
            ("_canonical_edge_resistance_ohm", 9),
        )
        for name, index in checks:
            candidate = state_dict.get(f"{prefix}{name}")
            current = getattr(self, name)
            if (
                not torch.is_tensor(candidate)
                or not torch.is_floating_point(candidate)
                or tuple(candidate.shape) != tuple(current.shape)
            ):
                raise RuntimeError(
                    "Cannot load a native Cable checkpoint with missing or "
                    f"corrupt canonical geometry buffer {name!r}."
                )
            candidate_cpu = candidate.detach().cpu()
            try:
                expected = torch.broadcast_to(
                    incoming_reference_native_cpu[index].to(dtype=candidate_cpu.dtype),
                    candidate_cpu.shape,
                )
            except RuntimeError as error:
                raise RuntimeError(
                    "Cannot load a native Cable checkpoint with corrupt canonical "
                    f"geometry buffer {name!r}."
                ) from error
            equal = torch.equal(candidate_cpu, expected)
            if name == "rhoa" and not equal:
                eps = torch.finfo(candidate_cpu.dtype).eps
                equal = torch.allclose(
                    candidate_cpu,
                    expected,
                    rtol=4.0 * eps,
                    atol=0.0,
                )
            if not equal:
                raise RuntimeError(
                    "Cannot load a native Cable checkpoint with corrupt canonical "
                    f"geometry buffer {name!r}."
                )

        parent_key = f"{prefix}diff_parent_index"
        incoming_parent = state_dict.get(parent_key)
        if (
            not torch.is_tensor(incoming_parent)
            or incoming_parent.dtype != self.diff_parent_index.dtype
            or not torch.equal(
                incoming_parent.detach().cpu(),
                self.diff_parent_index.detach().cpu(),
            )
        ):
            raise RuntimeError(
                "Cannot load a native Cable checkpoint with corrupt canonical "
                "geometry buffer 'diff_parent_index'."
            )

    def _preserve_local_canonical_geometry_during_load(self, state_dict, prefix):
        """Replace validated frozen payload entries with target-local values."""
        frozen = {
            "diam": self.diam,
            "dx": self.dx,
            "rhoa": self.rhoa,
            "volume": self.volume,
            "volume_um3": self.volume_um3,
            "volume_i": self.volume_i,
            "volume_o": self.volume_o,
            "diff_geom_um": self.diff_geom_um,
            "diff_parent_index": self.diff_parent_index,
            "_canonical_area_cm2": self._canonical_area_cm2,
            "_canonical_edge_resistance_ohm": (self._canonical_edge_resistance_ohm),
            "_canonical_morphology_fingerprint": (
                self._canonical_morphology_fingerprint
            ),
            "_canonical_geometry_reference": self._canonical_geometry_reference,
        }
        for name, value in frozen.items():
            key = f"{prefix}{name}"
            if key in state_dict:
                state_dict[key] = value.detach().clone()

        # ``rhoa`` is a bounded RANGE parameter. Its public buffer and the
        # unconstrained source coordinate must remain one target-local pair;
        # retaining only the public value would let the next parameter
        # population silently recreate lower-precision checkpoint geometry.
        for name, value in self.rhoa_param.state_dict().items():
            key = f"{prefix}rhoa_param.{name}"
            if key not in state_dict:
                continue
            state_dict[key] = (
                value.detach().clone()
                if torch.is_tensor(value)
                else copy.deepcopy(value)
            )

        # A built model contains aliases/snapshots of the same morphology in
        # its handler, mechanisms, material processes, and material spatial
        # operators. Preserve those target-local copies too. Mutable mechanism
        # and material state remains loadable; only geometry-derived entries
        # are replaced.
        for name, value in self.state_dict().items():
            leaf = name.rsplit(".", 1)[-1]
            handler_area = name in {"mech.area", "integrator.mech.area"}
            nested_geometry = "." in name and (
                handler_area
                or leaf == "diam"
                or leaf.startswith("_mp_")
                or "._spatial_operators." in name
            )
            if not nested_geometry:
                continue
            key = f"{prefix}{name}"
            if key not in state_dict:
                continue
            state_dict[key] = (
                value.detach().clone()
                if torch.is_tensor(value)
                else copy.deepcopy(value)
            )

        # Direct state_dict loading can occur after the timestep workspace was
        # initialized. Child buffers are copied after this hook returns, so
        # force the next public step to rebuild every geometry-derived solver
        # workspace from the preserved target sources.
        if self.integrator is not None:
            self.integrator.initialized = False

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Reject checkpoints authored for a different canonical morphology."""
        key = f"{prefix}_canonical_morphology_fingerprint"
        current = self._canonical_morphology_fingerprint
        incoming = state_dict.get(key)
        if (current is None) != (incoming is None):
            raise RuntimeError(
                "Cannot load state between a canonical native Cable and a "
                "non-canonical Cable/Axon model."
            )
        if current is not None:
            if not torch.is_tensor(incoming) or not torch.equal(
                current.detach().cpu(), incoming.detach().cpu()
            ):
                raise RuntimeError(
                    "Cannot load a native Cable checkpoint into a different "
                    "canonical morphology. Topology, geometry, or provenance differs."
                )
            self._preflight_incoming_canonical_geometry(state_dict, prefix)
            self._preserve_local_canonical_geometry_during_load(state_dict, prefix)
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def state_dict_for_checkpoint(self):
        """Include canonical identity in runtime/activation checkpoints."""
        self._validate_canonical_geometry()
        state = super().state_dict_for_checkpoint()
        fingerprint = self._canonical_morphology_fingerprint
        if fingerprint is not None:
            state["canonical_morphology_fingerprint"] = fingerprint.clone()
        return state

    def _restore_dict_from_checkpoint_unchecked(self, state_dict):
        """Reject runtime state from a different canonical morphology first."""
        self._validate_canonical_geometry()
        fingerprint = self._canonical_morphology_fingerprint
        incoming = state_dict.get("canonical_morphology_fingerprint")
        if fingerprint is not None:
            if not torch.is_tensor(incoming) or not torch.equal(
                fingerprint.detach().cpu(), incoming.detach().cpu()
            ):
                raise RuntimeError(
                    "Cannot restore a native Cable runtime checkpoint into a "
                    "different canonical morphology."
                )
        elif incoming is not None:
            raise RuntimeError(
                "Cannot restore canonical native Cable state into a "
                "non-canonical Cable/Axon model."
            )
        return super()._restore_dict_from_checkpoint_unchecked(state_dict)

    @property
    def graph(self):
        """Fresh NetworkX path view aligned to model compartment storage."""
        self._validate_canonical_geometry()
        return self._canonical_graph_view()

    def _canonical_graph_view(self):
        """Return a defensive graph whose edge numerics match model buffers.

        ``_compartment_graph`` and ``_graph`` retain the binary64 morphology
        snapshot used for provenance and checkpoint identity.  A dtype move is
        intentionally different: for example, float32 -> float64 widens the
        already-rounded model buffers and cannot recover the original binary64
        geometry.  Runtime graph consumers must therefore read resistance and
        diffusion geometry from the same model-dtype frozen buffers as the UB
        voltage solver and material process, rather than silently switching
        back to the provenance precision through ``_graph``.
        """
        if self._graph is None:
            return None

        graph = self._graph.copy()
        resistance = torch.broadcast_to(
            self._canonical_edge_resistance_ohm, self.shape
        ).reshape(-1, self.nc)[0]
        diffusion = torch.broadcast_to(self.diff_geom_um, self.shape).reshape(
            -1, self.nc
        )[0]
        for child, parent in enumerate(self._compartment_graph.topology.parent_index):
            if parent == -1:
                continue
            edge = graph.edges[parent, child]
            edge["R_ohm"] = resistance[child].detach().item()
            edge["diff_geom_um"] = diffusion[child].detach().item()
        return graph

    @property
    def compartment_graph(self):
        """Immutable path-ordered canonical morphology snapshot, if present."""
        return self._compartment_graph

    @property
    def area(self):
        """Exact compiled membrane area for native cables, in cm²."""
        self._validate_canonical_geometry()
        area = self._canonical_area_cm2
        if area is not None:
            # Mechanism construction may retain this tensor as registered
            # state. Materialize broadcasted batch axes so state restoration
            # never attempts to copy into a zero-stride expanded view.
            return torch.broadcast_to(area, self.shape).clone()
        return super().area

    @property
    def edge_resistance_ohm(self):
        """Child-indexed exact axial resistance, including a zero root entry."""
        self._validate_canonical_geometry()
        resistance = self._canonical_edge_resistance_ohm
        if resistance is None:
            return None
        return torch.broadcast_to(resistance, self.shape).clone()

    def material_volume(self, domain="intracellular"):
        """Return an exact native-cable material-domain volume buffer."""
        self._validate_canonical_geometry()
        domain = str(domain or "intracellular").lower()
        if domain in {"i", "inside", "cytosol", "cytoplasm", "intracellular"}:
            name = "volume_i"
        elif domain in {"o", "outside", "extracellular"}:
            name = "volume_o"
        elif domain in {"total", "volume", "all"}:
            name = "volume"
        elif domain in {"membrane", "surface", "area"}:
            return self.area
        else:
            raise ValueError(f"Unsupported material domain {domain!r} for Cable.")
        if not hasattr(self, name):
            raise RuntimeError(
                "Exact material volumes are available only on a Cable built "
                "from a canonical compartment graph."
            )
        return getattr(self, name)

    def assemble_graphs(self):
        """Return one exact path graph view per core population member."""
        self._validate_canonical_geometry()
        graph = self._canonical_graph_view()
        if graph is None:
            raise RuntimeError(
                "Cable graph assembly requires construction from a canonical "
                "compartment graph."
            )
        return [graph.copy() for _ in range(self.np)]

    @classmethod
    def from_compartment_graph(cls, graph, N=1, integrator=None, **kwargs):
        """Construct a fast generic cable from a canonical material path.

        The source graph is reordered deterministically from one physical end
        to the other.  Branches and retained zero-area junctions are rejected.
        Exact edge resistance and membrane area are retained rather than being
        reconstructed from one representative compartment diameter and length.
        """
        from .morphology import CompartmentGraph
        from .tree import (
            _register_canonical_internal_nodes,
            _register_compartment_graph_labels,
        )

        if not getattr(cls, "_supports_native_cable_factory", False):
            raise NotImplementedError(
                f"{cls.__name__}.from_compartment_graph requires an explicit "
                "adapter because that class defines specialized path geometry. "
                "Use Cable.from_compartment_graph for a generic native path."
            )
        if not isinstance(graph, CompartmentGraph):
            raise TypeError("graph must be a CompartmentGraph.")
        if "rhoa" in kwargs:
            raise ValueError(
                "A canonical Cable's exact edge resistance already incorporates "
                "Section rhoa. Set rhoa on the source Morphology instead of "
                "overriding it during construction."
            )

        path = graph.path_ordered()
        geometry = path.geometry
        C = path.n_compartments
        cm = kwargs.pop("cm", geometry.cm_uF_cm2)
        cable = cls(
            N,
            C,
            integrator=integrator,
            cm=cm,
            rhoa=geometry.rhoa_ohm_cm,
            **kwargs,
        )
        cable._graph = path.to_networkx()

        def expanded(values, *, scale=1.0):
            # Canonical morphology calculations and unit conversions happen in
            # binary64; cast only the final value to the configured model dtype.
            value = torch.as_tensor(values, dtype=torch.float64)
            if scale != 1.0:
                value = value * scale
            value = value.to(device=cable.device(), dtype=cable.dtype()).reshape(1, C)
            return value.expand(N, -1).clone()

        cable.dx.copy_(expanded(geometry.length_um))
        cable.diam.copy_(expanded(geometry.diameter_um))
        cable.x.copy_(expanded(geometry.x_um))
        cable.y.copy_(expanded(geometry.y_um))
        cable.z.copy_(expanded(geometry.z_um))
        cable._canonical_area_cm2 = expanded(geometry.area_um2, scale=1e-8)
        cable._canonical_edge_resistance_ohm = expanded(geometry.edge_resistance_ohm)
        fingerprint = cls._fingerprint_compartment_graph(path)
        cable._canonical_morphology_fingerprint_reference = fingerprint
        cable._canonical_morphology_fingerprint = torch.tensor(
            list(fingerprint),
            device=cable.device(),
            dtype=torch.uint8,
        )

        for name, values in (
            ("volume", geometry.volume_um3),
            ("volume_um3", geometry.volume_um3),
            ("volume_i", geometry.volume_i_um3),
            ("volume_o", geometry.volume_o_um3),
            ("diff_geom_um", geometry.edge_diff_geom_um),
        ):
            cable.register_buffer(name, expanded(values))
        cable.register_buffer(
            "diff_parent_index",
            torch.as_tensor(
                path.topology.parent_index,
                device=cable.device(),
                dtype=torch.long,
            ),
        )

        cable.names = list(path.metadata.name)
        cable._compartment_graph = path
        cable._canonical_geometry_reference = (
            torch.stack(
                (
                    cable.diam[:1],
                    cable.dx[:1],
                    cable.rhoa[:1],
                    cable.volume[:1],
                    cable.volume_um3[:1],
                    cable.volume_i[:1],
                    cable.volume_o[:1],
                    cable.diff_geom_um[:1],
                    cable._canonical_area_cm2[:1],
                    cable._canonical_edge_resistance_ohm[:1],
                ),
                dim=0,
            )
            .detach()
            .clone()
        )
        _register_canonical_internal_nodes(cable, path)
        _register_compartment_graph_labels(cable, path)
        return cable

    @classmethod
    def from_morphology(cls, morphology, N=1, integrator=None, **kwargs):
        """Compile a native Section path and construct a generic Cable.

        Construction takes an immutable snapshot. Later Section updates on the
        source Morphology do not affect this Cable; call ``from_morphology``
        again to build a Cable from the revised declaration.
        """
        from .morphology import Morphology
        from .tree import _register_compartment_graph_labels

        if not isinstance(morphology, Morphology):
            raise TypeError("morphology must be a Morphology.")
        cable = cls.from_compartment_graph(
            morphology.compile(), N=N, integrator=integrator, **kwargs
        )
        graph = cable.compartment_graph
        ordered_labels = {}
        for section in morphology.sections:
            section_nodes = sorted(
                (
                    node
                    for node, section_name in enumerate(graph.metadata.section_name)
                    if section_name == section.name
                ),
                key=lambda node: graph.metadata.segment_index[node],
            )
            for label in section.labels:
                ordered_labels.setdefault(label, []).extend(section_nodes)
        _register_compartment_graph_labels(cable, graph, ordered=ordered_labels)
        return cable


class Axon(Cable):
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

    _supports_native_cable_factory = False

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

        diameter_values = (
            torch.as_tensor(diameters, device=self.device(), dtype=self.dtype())
            .clone()
            .detach()
        )
        self.register_buffer("diameters", diameter_values)

        self.n_ax = self.np
        self.n_comp = self.nc

        # ``v_init`` is normalized by Population; it may be scalar or a
        # length-n_comp vector. ``self.v`` was already created from it.

        self.x[:] = self._x()  # Initialize x positions

        self.cid = None
        self._cid_label_names = set()

        compartment_diameters = self.diameters
        if compartment_diameters.ndim == 1:
            compartment_diameters = compartment_diameters.unsqueeze(1)

        self.diam.copy_(compartment_diameters)
        self.diam = self.diam.detach()

    def clear_labels(self):
        """Clear slice labels and forget any CID label ownership."""
        super().clear_labels()
        self._cid_label_names.clear()

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
                location = 0.5 if self.n_comp == 1 else node / (self.n_comp - 1)
                G.nodes[node]["name"] = (
                    f"{self.__class__.__name__}[{i}]({location:.2f})"
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
        """Register compartment names and expose their labelled slices.

        This connects a model-definition
        :class:`~dendra.models.heterogeneous.CompartmentID` to an axon. Each
        unique name becomes both an entry in ``_labels`` and an attribute
        containing the corresponding :class:`~dendra.models.slice.Slice`.

        Parameters
        ----------
        cid : dendra.models.heterogeneous.CompartmentID
            Identifier table whose expanded ``names`` array has length
            ``n_comp``.

        Examples
        --------
        .. code-block:: python

            from dendra.models.core import Unmyelinated
            from dendra.models.heterogeneous import CompartmentID
            from dendra.units import um

            axon = Unmyelinated(diameters=[8.0], L=60.0 * um, dx=10.0 * um)
            cid = CompartmentID(["node", "internode * 2"], n_repeats=2)
            axon.register_cid(cid)

            axon.node       # compartments 0, 3, and 6
            axon.internode  # compartments 1, 2, 4, and 5
        """
        if not hasattr(cid, "names"):
            raise TypeError("cid must expose a one-dimensional 'names' sequence.")
        raw_names = cid.names
        names_ndim = getattr(raw_names, "ndim", None)
        if isinstance(raw_names, (str, np.str_)) or (
            names_ndim is not None and int(names_ndim) != 1
        ):
            raise TypeError("cid.names must be a one-dimensional sequence.")
        raw_names = raw_names.tolist() if hasattr(raw_names, "tolist") else raw_names
        if isinstance(raw_names, (str, np.str_)):
            raise TypeError("cid.names must be a one-dimensional sequence.")
        try:
            names = list(raw_names)
        except TypeError as error:
            raise TypeError("cid.names must be a one-dimensional sequence.") from error
        if any(not isinstance(name, (str, np.str_)) for name in names):
            raise TypeError("Every compartment name in cid.names must be a string.")
        names = [str(name) for name in names]
        if len(names) != self.n_comp:
            raise ValueError(
                f"Axon requires {self.n_comp} compartment names, got {len(names)}."
            )

        unique_names = list(dict.fromkeys(names))
        old_owned = set(self._cid_label_names)
        always_reserved = {"cid", "names", "_cid_label_names", "_labels"}
        for name in unique_names:
            replaceable_old_label = (
                name in old_owned
                and name in self.__dict__
                and self.__dict__[name] is self._labels.get(name)
            )
            conflicts = name in always_reserved or (
                not replaceable_old_label
                and (hasattr(self, name) or name in self._labels)
            )
            if conflicts:
                raise ValueError(
                    f"Compartment label {name!r} conflicts with an existing attribute."
                )

        names_array = np.asarray(names, dtype=object)
        prepared_labels = {
            name: self[..., np.flatnonzero(names_array == name).tolist()]
            for name in unique_names
        }

        old_cid = self.cid
        old_names_present = "names" in self.__dict__
        old_names = self.__dict__.get("names")
        old_cid_label_names = set(self._cid_label_names)
        old_labels = dict(self._labels)
        touched_names = old_owned | set(unique_names)
        old_attributes = {
            name: self.__dict__[name] for name in touched_names if name in self.__dict__
        }

        try:
            for name in old_owned:
                old_label = self._labels.pop(name, None)
                if name in self.__dict__ and self.__dict__[name] is old_label:
                    delattr(self, name)

            self.cid = cid
            self.names = names
            for name, label in prepared_labels.items():
                setattr(self, name, label)
                self._labels[name] = label
            self._cid_label_names = set(unique_names)
        except Exception:
            for name in touched_names:
                if name in self.__dict__:
                    delattr(self, name)
            for name, value in old_attributes.items():
                setattr(self, name, value)
            self._labels.clear()
            self._labels.update(old_labels)
            self.cid = old_cid
            self._cid_label_names = old_cid_label_names
            if old_names_present:
                self.names = old_names
            elif "names" in self.__dict__:
                delattr(self, "names")
            raise

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
        return self[..., self.c(*args)]


def _match_state_dict(
    state_dict_a: Dict[str, Any],
    state_dict_b: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Match entries between two state dictionaries based on key names and compatibility.

    Tensor entries must have identical shapes. Non-tensor entries, such as the
    dictionaries produced by ``nn.Module.get_extra_state()``, are compatible when
    the corresponding key exists and both entries are non-tensors.

    Parameters
    ----------
    state_dict_a : Dict[str, Any]
        First state dictionary used as reference for key and shape matching.
    state_dict_b : Dict[str, Any]
        Second state dictionary to filter based on keys and shapes in state_dict_a.
    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        A tuple containing:
        - matched_state_dict: Dictionary with entries from state_dict_b that have
          matching keys and shapes in state_dict_a.
        - unmatched_state_dict: Dictionary with remaining entries from state_dict_b
          that don't have matching keys or shapes in state_dict_a.
    """

    def compatible(reference, candidate):
        reference_is_tensor = torch.is_tensor(reference)
        candidate_is_tensor = torch.is_tensor(candidate)
        if reference_is_tensor != candidate_is_tensor:
            return False
        if reference_is_tensor:
            return reference.shape == candidate.shape
        return True

    matched_state_dict = {
        key: state
        for (key, state) in state_dict_b.items()
        if key in state_dict_a and compatible(state_dict_a[key], state)
    }
    unmatched_state_dict = {
        key: state
        for (key, state) in state_dict_b.items()
        if key not in matched_state_dict
    }
    return matched_state_dict, unmatched_state_dict


def _load_state_dict_transactionally(module, state_dict, *, strict: bool):
    """Load module state without exposing mutations from a failed load."""

    if not isinstance(state_dict, Mapping):
        raise TypeError("state_dict must be a mapping.")
    rollback_state = _clone_nested_state(module.state_dict(), detach_tensors=True)
    try:
        return module.load_state_dict(state_dict, strict=strict)
    except Exception:
        try:
            module.load_state_dict(rollback_state, strict=True)
        except Exception as rollback_error:
            raise RuntimeError(
                "State-dictionary load failed and rollback could not recover the "
                "previous module state."
            ) from rollback_error
        raise


def _load_population_cache_transactionally(population, state_dict):
    """Restore a named Population cache while rebuilding solver workspaces.

    Direct buffers owned by an integrator are derived from morphology, parameters,
    and the next step's ``dt``.  They are therefore not runtime state, and their
    shapes may legitimately change when a lazily initialized solver first runs.
    Keep the live workspace buffers during the strict state-dictionary load and
    invalidate the integrator so the next step reconstructs them.  Every other
    key retains strict, transactional loading semantics.
    """

    if not isinstance(state_dict, Mapping):
        raise TypeError("Cached population state must be a mapping.")

    live_state = population.state_dict()
    prepared = dict(state_dict)
    integrator = getattr(population, "integrator", None)
    if integrator is not None:
        prefix = "integrator."
        # Inject current direct workspace buffers. Nested mechanism state
        # (``integrator.mech.*``), including unexpected keys, remains part of
        # the strict snapshot validation.
        for name in integrator._buffers:
            key = f"{prefix}{name}"
            if key in live_state:
                prepared[key] = live_state[key]

    result = _load_state_dict_transactionally(population, prepared, strict=True)
    if integrator is not None:
        integrator.initialized = False
    return result


def _load_compatible_state_dict_transactionally(module, state_dict):
    """Apply compatible state entries without exposing partial failed loads.

    Shape-incompatible and unknown entries retain the historical ``load()``
    behavior and are ignored. If loading a compatible entry raises (for example,
    because an RNG ``_extra_state`` payload is corrupt), every earlier mutation is
    rolled back before the original exception is re-raised.
    """
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            "state_dict must be a mapping or a path containing a state dictionary."
        )
    live_state = module.state_dict()
    matched, _ = _match_state_dict(live_state, state_dict)
    _load_state_dict_transactionally(module, matched, strict=False)


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
        Requested axon length in μm. Bare values are interpreted as μm; use
        :data:`dendra.units.mm` when specifying millimetres. Default is
        ``1.0 * mm`` (1000 μm).
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


def compose_or_flatten_multiset(
    indices: List[Any],
    shape: Tuple[int, ...],
    preserve_duplicate_indices: List[bool],
    copies: List[int] | None = None,
) -> Tuple[List[int], bool, Tuple[int, ...], List[List[int]]]:
    """Compose mechanism placement while preserving selected duplicates.

    ``compose_or_flatten_union`` intentionally collapses overlapping insertion
    regions into one mechanism state per compartment.  That is the correct
    default for distributed mechanisms, but it prevents efficient banks of
    colocated point processes: several independent synapses may live at one
    compartment and should scatter-add their currents back to that compartment.

    This helper is an opt-in multiset variant.  Records marked with
    ``preserve_duplicate_indices=True`` allocate a fresh local mechanism slot
    for every selected compartment occurrence.  Unmarked records retain the
    ordinary unique-sharing behavior among themselves.  The result is always a
    flat advanced-index placement so repeated compartment keys are preserved.
    """

    empty_shape = (0,)
    empty_locals = [[] for _ in indices]
    if not indices:
        return [], False, empty_shape, []
    if not shape or not all(s > 0 for s in shape):
        return [], False, empty_shape, empty_locals
    if len(preserve_duplicate_indices) != len(indices):
        raise ValueError(
            "preserve_duplicate_indices must have one entry per mechanism insertion."
        )
    if copies is None:
        copies = [1] * len(indices)
    if len(copies) != len(indices):
        raise ValueError("copies must have one entry per mechanism insertion.")
    copies = [int(c) for c in copies]
    if any(c < 1 for c in copies):
        raise ValueError("all copy counts must be positive integers.")

    total_elements = int(np.prod(shape))
    arr = np.arange(total_elements).reshape(shape)

    total_indices: List[int] = []
    local_indices: List[List[int]] = []
    shared_local_by_global: Dict[int, int] = {}

    for idx, preserve, n_copies in zip(indices, preserve_duplicate_indices, copies):
        try:
            selected = np.asarray(arr[idx]).reshape(-1)
        except IndexError as e:
            raise IndexError(
                f"Indexer invalid for shape. Idx: {idx}, Shape: {shape}. Error: {e}"
            ) from e

        if bool(preserve):
            locs = []
            selected_list = [int(g_idx) for g_idx in selected.tolist()]
            for _copy in range(int(n_copies)):
                for g_idx in selected_list:
                    locs.append(len(total_indices))
                    total_indices.append(int(g_idx))
            local_indices.append(locs)
            continue

        unique_selected = sorted(set(int(g) for g in selected.tolist()))
        locs = []
        for g_idx in unique_selected:
            local = shared_local_by_global.get(g_idx)
            if local is None:
                local = len(total_indices)
                shared_local_by_global[g_idx] = local
                total_indices.append(int(g_idx))
            locs.append(local)
        local_indices.append(locs)

    if not total_indices:
        return [], False, empty_shape, empty_locals

    final_shape = (len(total_indices),)
    return total_indices, False, final_shape, local_indices


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


def compile_mechanism(
    model,
    mechanism,
    indices,
    aliases,
    kwargs_list,
    preserve_duplicate_indices=None,
    copies=None,
    *,
    ic=None,
    base_kwargs=None,
    name=None,
    force_flat=False,
):
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
    preserve_duplicate_indices : list of bool, optional
        Per-insertion flags.  When any entry is true, selected duplicate
        compartment indices for those insertions are retained as independent
        local mechanism slots rather than collapsed into the default union.
    copies : list of int, optional
        Per-insertion copy counts. Values greater than one allocate repeated
        independent local mechanism slots for the selected region.
    ic : dict, optional
        Class-wide initial-condition overrides for the compiled mechanism.
    base_kwargs : dict, optional
        Constructor-level parameters applied uniformly across the compiled
        support. Alias-specific values remain in ``kwargs_list``.
    name : str, optional
        Explicit runtime mechanism name.
    force_flat : bool, optional
        Store an otherwise composable placement as a flat sparse slot axis.

    Returns
    -------
    tuple
        Tuple ``(mechanism_instance, parameter_shape, total_index)`` ready for
        registration via :meth:`Population._register_mech`.
    """
    if preserve_duplicate_indices is None:
        preserve_duplicate_indices = [False] * len(indices)
    if copies is None:
        copies = [1] * len(indices)
    preserve_duplicate_indices = [bool(v) for v in preserve_duplicate_indices]
    copies = [int(c) for c in copies]
    preserve_duplicate_indices = [
        bool(p or c != 1) for p, c in zip(preserve_duplicate_indices, copies)
    ]
    uses_multiset_layout = any(preserve_duplicate_indices)
    if uses_multiset_layout:
        total_index, is_composable, shape, local_indices = compose_or_flatten_multiset(
            indices,
            model.core_shape(),
            preserve_duplicate_indices,
            copies,
        )
    else:
        total_index, is_composable, shape, local_indices = compose_or_flatten_union(
            indices, model.core_shape()
        )

    base_kwargs = dict(base_kwargs or {})

    def cannot_target_composed_shape(value):
        if isinstance(value, torch.nn.Module):
            return False
        try:
            value = torch.as_tensor(value)
        except (TypeError, ValueError):
            return False
        if value.ndim == 0:
            return False
        try:
            torch.broadcast_to(value, shape)
        except RuntimeError:
            return value.numel() == math.prod(shape)
        return False

    indexed_base_names = _mechanism_spatial_parameter_names(mechanism)
    shape_sensitive_values = [
        value for key, value in base_kwargs.items() if key in indexed_base_names
    ]
    if ic is not None:
        shape_sensitive_values.extend(ic.values())
    force_flat = bool(
        force_flat
        or (
            is_composable
            and any(cannot_target_composed_shape(v) for v in shape_sensitive_values)
        )
    )
    if force_flat and is_composable:
        core_grid = np.arange(math.prod(model.core_shape())).reshape(model.core_shape())
        total_index = np.asarray(core_grid[total_index]).reshape(-1).tolist()
        shape = (len(total_index),)
        is_composable = False

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
    spatial_parameter_names = _mechanism_spatial_parameter_names(mechanism)

    for alias, kwargs, idx, n_copies, preserves_multiplicity in zip(
        aliases,
        kwargs_list,
        local_indices,
        copies,
        preserve_duplicate_indices,
    ):
        if hasattr(mechanism, "normalize_random_kwargs"):
            kwargs = mechanism.normalize_random_kwargs(kwargs)
        logical_shape = None
        if preserves_multiplicity:
            n_record_slots = len(idx)
            if n_record_slots % n_copies:
                raise RuntimeError(
                    "Copied mechanism insertion produced an inconsistent "
                    f"record layout: {n_record_slots} slots for {n_copies} copies."
                )
            logical_shape = (n_copies, n_record_slots // n_copies)
        for k, v in kwargs.items():
            parameter_logical_shape = (
                logical_shape if k in spatial_parameter_names else None
            )
            additional_parameters.setdefault(k, []).append(
                (alias, v, idx, parameter_logical_shape)
            )

    mechanism.check_kwargs(additional_parameters)
    if hasattr(mechanism, "normalize_random_kwargs"):
        base_kwargs = mechanism.normalize_random_kwargs(base_kwargs)
    mechanism.check_kwargs(base_kwargs)

    m = mechanism(
        name,
        model.celsius,
        model.diam,
        shape_p,
        shape_f,
        key=total_index,
        is_composable=is_composable,
        additional_parameters=additional_parameters,
        ic=ic,
        **base_kwargs,
    )

    return m, shape_p, total_index
