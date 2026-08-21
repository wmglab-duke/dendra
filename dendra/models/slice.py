"""Helpers for slicing neural populations and mechanisms.

This module provides the :class:`Slice` abstraction and related utilities for
working with indexed views of neural populations in Dendra.

A :class:`~dendra.models.core.Population` typically represents a collection of
neuronal compartments arranged in a regular tensor shape
(e.g. ``(n_cells, n_compartments)``). In Dendra, indexing a population with
standard Python / NumPy / PyTorch semantics does **not** return a raw tensor;
instead it returns a :class:`Slice` object:

.. code-block:: python

    pop = Population(...)
    soma = pop[:, 0]       # Slice
    first10 = pop[:10]     # Slice
    subset = pop[mask]     # Slice

A :class:`Slice` is a lightweight, logical view that:

* Remembers the selection in a canonical :class:`IndexSpec` structure.
* Provides convenience methods for reading and writing state restricted to
  that subset of compartments.
* Supports targeted intracellular current injections.
* Restricts mechanism insertion and configuration to the selected region.
* Can itself be sliced again (slices-of-slices) without materialising
  intermediate tensors.

The actual numerical state (voltages, gating variables, parameters, etc.)
remains owned by the underlying :class:`~dendra.models.core.Population` or its
mechanisms. :class:`Slice` simply routes reads and writes through the correct
indices in a safe and convenient way.
"""

from __future__ import annotations

import keyword
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch

# Common NumPy/PyTorch indexing elements.  PyTorch accepts a few additional
# array-like objects at runtime; ``parse_key`` remains the source of truth.
IndexElement = Union[
    int,
    bool,
    slice,
    np.ndarray,
    torch.Tensor,
    list,
    tuple,
    type(Ellipsis),
    None,
]


@dataclass(slots=True)
class IndexSpec:
    """Canonical description of a single indexing request.

    Attributes
    ----------
    index : tuple of IndexElement
        Stable, owned indexing key.  The outer container is always a tuple;
        ``Ellipsis`` and ``None`` are retained because they are meaningful
        parts of PyTorch indexing semantics.
    is_scalar : bool
        ``True`` when the result is scalar valued.
    shape : torch.Size
        Shape produced by applying ``index`` to a tensor.
    source_shape : tuple of int
        Logical model shape against which the index was last resolved.  Slice
        uses this to preserve a core-region selection when leading batch axes
        are added.
    """

    index: Tuple[IndexElement, ...]
    is_scalar: bool
    shape: torch.Size
    source_shape: Tuple[int, ...] = ()

    def to_key(self, model) -> torch.LongTensor:
        """Return selected flat parameter-storage keys for ``model``."""
        storage_shape = tuple(model.shape_p)
        logical_shape = tuple(getattr(model, "shape", storage_shape))
        base = torch.arange(
            math.prod(storage_shape), device=model.device(), dtype=torch.long
        ).view(storage_shape)
        try:
            base = torch.broadcast_to(base, logical_shape)
        except RuntimeError:
            pass
        selected = base[self.index].reshape(-1)
        if selected.numel() < 2:
            return selected
        seen = set()
        positions = []
        for position, value in enumerate(selected.detach().cpu().tolist()):
            if value not in seen:
                seen.add(value)
                positions.append(position)
        return selected.index_select(
            0,
            torch.as_tensor(positions, device=selected.device, dtype=torch.long),
        )


def _population_flat_indices(model, index=None) -> torch.LongTensor:
    """Return population-flat indices selected by ``index``."""
    grid = torch.arange(
        math.prod(tuple(model.shape)),
        device=model.device(),
        dtype=torch.long,
    ).view(tuple(model.shape))
    if index is None:
        return grid.reshape(-1)
    return grid[index].reshape(-1)


def _mechanism_slot_to_flat_index(model, synapse) -> torch.LongTensor:
    """Return the population-flat compartment index for each local mechanism slot."""
    if getattr(synapse, "key", None) is None:
        return torch.arange(
            math.prod(tuple(model.shape)),
            device=model.device(),
            dtype=torch.long,
        )
    grid = torch.arange(
        math.prod(tuple(model.shape)),
        device=model.device(),
        dtype=torch.long,
    ).view(tuple(model.shape))
    return synapse.get(grid).reshape(-1).to(dtype=torch.long)


@dataclass(slots=True)
class SynapseSlots:
    """Explicit selection of local slots inside a point-process mechanism.

    A regular :class:`Slice` identifies physical compartments.  That is enough
    for distributed mechanisms, but a banked point-process mechanism can have
    several independent local slots attached to the same compartment.  A
    ``SynapseSlots`` object addresses those local slots directly while retaining
    their mapping back to parent-population compartments.
    """

    model: Any
    synapse: Any
    local_index: torch.LongTensor
    slot_to_flat_index: torch.LongTensor
    name: Optional[str] = None

    def __post_init__(self):
        device = self.model.device()
        local_index = torch.as_tensor(
            self.local_index, device=device, dtype=torch.long
        ).reshape(-1)
        slot_to_flat_index = torch.as_tensor(
            self.slot_to_flat_index, device=device, dtype=torch.long
        ).reshape(-1)

        n_slots = int(slot_to_flat_index.numel())
        if torch.any(local_index < 0) or torch.any(local_index >= n_slots):
            bad = local_index[(local_index < 0) | (local_index >= n_slots)][:10]
            raise IndexError(
                f"Synapse slot index out of range for mechanism "
                f"{getattr(self.synapse, 'name', type(self.synapse).__name__)!r}: "
                f"valid range is [0, {max(n_slots - 1, 0)}], got "
                f"{bad.detach().cpu().tolist()}."
            )

        self.local_index = local_index
        self.slot_to_flat_index = slot_to_flat_index
        if self.name is None:
            model_name = getattr(self.model, "name", "<population>")
            syn_name = getattr(self.synapse, "name", type(self.synapse).__name__)
            self.name = f"{model_name}.{syn_name}.slots"

    @classmethod
    def from_region(
        cls,
        model,
        synapse,
        *,
        region_index=None,
        slot_index=None,
        name: Optional[str] = None,
    ) -> "SynapseSlots":
        """Create a slot selection from a population region and/or slot index.

        ``slot_index`` addresses the mechanism's absolute local slot axis.  If
        ``region_index`` is also provided, the selected absolute slots are
        validated against that population region.  Relative selection within an
        already-created slot view is available through ``SynapseSlots.__getitem__``.
        """
        slot_to_flat = _mechanism_slot_to_flat_index(model, synapse)
        all_local = torch.arange(
            slot_to_flat.numel(), device=model.device(), dtype=torch.long
        )

        if slot_index is None:
            local = all_local
        else:
            spec = parse_key(
                slot_index,
                (int(slot_to_flat.numel()),),
                device=model.device(),
            )
            local = all_local.reshape(-1)[spec.index].reshape(-1)

        if region_index is not None:
            region_flat = _population_flat_indices(model, region_index)
            if region_flat.numel() == 0:
                if slot_index is not None and local.numel() > 0:
                    raise ValueError(
                        "Selected synapse slot(s) are outside the requested "
                        "population region."
                    )
                local = local[:0]
            else:
                selected_flat = slot_to_flat.index_select(0, local)
                mask = torch.isin(selected_flat, region_flat)
                if slot_index is not None and not bool(torch.all(mask)):
                    bad = local[~mask][:10].detach().cpu().tolist()
                    raise ValueError(
                        "Selected synapse slot(s) are outside the requested "
                        f"population region: {bad}."
                    )
                local = local[mask]

        return cls(
            model=model,
            synapse=synapse,
            local_index=local,
            slot_to_flat_index=slot_to_flat,
            name=name,
        )

    @property
    def shape(self) -> Tuple[int, ...]:
        return (int(self.local_index.numel()),)

    @property
    def index(self):
        """Slot-local indices, provided for endpoint-like introspection."""
        return self.local_index

    @property
    def is_empty(self) -> bool:
        return int(self.local_index.numel()) == 0

    @property
    def flat_index(self) -> torch.LongTensor:
        """Population-flat compartment index for each selected local slot."""
        return self.slot_to_flat_index.index_select(0, self.local_index)

    @property
    def numel(self) -> int:
        return int(self.local_index.numel())

    def __len__(self) -> int:
        return int(self.local_index.numel())

    def __getitem__(self, key) -> "SynapseSlots":
        spec = parse_key(key, self.shape, device=self.model.device())
        selected = self.local_index.reshape(self.shape)[spec.index].reshape(-1)
        return SynapseSlots(
            model=self.model,
            synapse=self.synapse,
            local_index=selected,
            slot_to_flat_index=self.slot_to_flat_index,
            name=self.name,
        )

    def to(self, device=None) -> "SynapseSlots":
        """Return a copy of the selection tensors on ``device``."""
        if device is None:
            device = self.model.device()
        return SynapseSlots(
            model=self.model,
            synapse=self.synapse,
            local_index=self.local_index.to(device=device),
            slot_to_flat_index=self.slot_to_flat_index.to(device=device),
            name=self.name,
        )

    def __repr__(self) -> str:
        syn_name = getattr(self.synapse, "name", type(self.synapse).__name__)
        return (
            f"SynapseSlots(name={self.name!r}, synapse={syn_name!r}, "
            f"n_slots={int(self.local_index.numel())})"
        )


class Slice:
    """
    Indexed view onto a subset of a :class:`dendra.models.core.Population`.

    Overview
    --------
    A :class:`~dendra.models.core.Population` represents a collection of
    neuronal compartments arranged in a tensor ``shape``
    (e.g. ``(n_cells, n_compartments)``). Indexing a population with any
    valid NumPy / PyTorch key (integers, slices, ellipses, boolean masks,
    integer arrays, or tuples thereof) returns a :class:`Slice` rather than
    a raw tensor:

    .. code-block:: python

        pop = Population(...)

        # Basic indexing
        soma = pop[:, 0]          # first compartment of every cell
        first_ten = pop[:10]      # first 10 cells
        distal = pop[:, 5:]       # distal compartments

        # Advanced / fancy indexing
        subset = pop[[0, 3, 7]]   # pick specific cells
        masked = pop[cell_mask]   # boolean-mask selection

    The :class:`Slice` object is a lightweight, logical view that:

    * Stores an owned :class:`IndexSpec` describing the selection.
    * Provides methods for reading and writing model and mechanism state
      restricted to that selection.
    * Supports targeted intracellular current injections via :meth:`inject`.
    * Restricts mechanism insertion and removal to that subset via
      :meth:`insert`, :meth:`delete`, and :meth:`delete_all`.
    * Can itself be sliced again; see *Nested slices* below.

    Importantly, a :class:`Slice` does **not** own any numerical state.
    All tensors and parameters remain stored on the underlying population
    or mechanisms. :class:`Slice` merely routes reads and writes through the
    correct indices.

    Creating slices
    ---------------
    Slices are created by indexing any :class:`Sliceable` object, typically
    an :class:`dendra.models.core.Population`:

    .. code-block:: python

        # Standard Python / NumPy semantics
        pop = Population(...)
        soma = pop[:, 0]              # select soma compartment
        dend = pop[:, 1:]             # all dendritic compartments
        middle = pop[:, 2:4]          # compartments 2 and 3

        # Explicit tuples of indices are also supported
        band = pop[(slice(None), slice(2, 6))]

    Advanced indexing mixes are handled by delegating to PyTorch's indexing rules.

    Nested slices (slices of slices)
    --------------------------------
    Applying further indexing to an existing :class:`Slice` returns another
    :class:`Slice` corresponding to the composition of the two selections:

    .. code-block:: python

        distal = pop[:, 5:]
        distal_mid = distal[:, 2:4]

        # Equivalent direct selection on the population
        direct = pop[:, 7:9]
        assert torch.allclose(distal_mid.v, direct.v)

    Internally, :func:`compose_indices` is used to compute an equivalent index
    into the original population. It materialises an integer coordinate map,
    but never materialises or copies the population state itself.

    Reading state: :meth:`inspect` and :meth:`get`
    ----------------------------------------------
    Use :meth:`inspect` (or its alias :meth:`get`) to read variables restricted
    to the slice:

    .. code-block:: python

        # Read a model-level buffer/parameter
        v_soma = pop[:, 0].get("v")              # membrane voltage in the soma

        # Read a mechanism field
        m_gate = pop[:, 0].get("m", mechanism="NaTs2t")

    When the target is stored sparsely (e.g. keyed mechanisms), the slice will
    internally materialise a dense tensor, index it, and return the selected
    subset.

    Attribute-style access
    ----------------------
    For common cases, you can also rely on attribute access instead of calling
    :meth:`get` explicitly. The :class:`Slice` intercepts attribute access and
    returns a sliced snapshot of buffers and parameters, or a retained Slice
    wrapper for submodules:

    .. code-block:: python

        # Sliced buffer access
        v_soma = pop[:, 0].v
        # Equivalent to:
        # v_soma = pop[:, 0].get("v")

        # Sliced mechanism access through the mechanism submodule
        m_gate = pop[:, 0].mech.NaTs2t.m
        # Equivalent to:
        # m_gate = pop[:, 0].get("m", mechanism="NaTs2t")

    This attribute-style syntax is often the most concise way to work with
    subsets of state.

    Writing state: :meth:`set` and attribute assignment
    ---------------------------------------------------
    To modify state only on the selected compartments, use :meth:`set`:

    .. code-block:: python

        soma = pop[:, 0]
        soma.set("v", torch.full(soma.shape, -65.0, device=pop.device()))

    If the target is registered as a buffer on the underlying PyTorch module,
    you can also write via attribute assignment on the :class:`Slice`:

    .. code-block:: python

        # Assuming ``v`` is a registered buffer on ``pop``
        soma.v = torch.full(soma.shape, -65.0, device=pop.device())

    Both pathways validate the complete update first, then copy it under
    ``torch.no_grad()``. Registered tensor identity, dtype, device and unrelated
    entries are preserved.

    Current injection: :meth:`inject`
    ---------------------------------
    :meth:`inject` registers an intracellular current waveform that should be
    applied only to the compartments in this slice during simulation:

    .. code-block:: python

        soma = pop[:, 0]
        soma.inject(step_current)

    Here ``step_current`` is any :py:class:`~dendra.models.stim.waveform.core.Waveform`
    object understood by the population's injection machinery. The slice records
    both the waveform and the :class:`IndexSpec` so that the solver can apply the
    current to the correct subset at run time.

    Mechanism insertion: :meth:`insert`
    -----------------------------------
    Mechanisms (ion channels, synapses, etc.) can be inserted only on a
    selected subset of compartments by calling :meth:`insert` on a slice:

    .. code-block:: python

        soma = pop[:, 0]
        soma.insert("NaTs2t", alias="Na_soma")

    Only the compartments included in the slice will host the mechanism. All
    other compartments in the population are unaffected. Any additional keyword
    arguments are forwarded to the underlying ``Population.insert`` call.

    Labelling slices: :meth:`label`
    -------------------------------
    Often it is convenient to name a particular subset once and reuse it
    throughout a model or experiment. :meth:`label` attaches a name to a
    slice and exposes it as an attribute on the owning population (or on the
    parent slice, if labelling a nested view):

    .. code-block:: python

        # Define labels
        pop[:, 0].label("soma")
        pop[:, 1:].label("dendrites")

        # Later, possibly in a different module:
        pop.soma.inject(step_current)
        pop.dendrites.set(
            "g",
            torch.full(pop.dendrites.shape, 1e-4),  # S/cm²
            mechanism="pas",
        )

    Labels are also stored in ``population._labels`` (a simple ``dict``),
    allowing programmatic access via ``population._labels["soma"]``.

    Attributes
    ----------
    index_spec : IndexSpec
        Canonicalised description of the selection (index, shape, scalar flag).
    parent_slice : Slice or None
        Parent slice if this slice was created from another slice.

    Notes
    -----
    A :class:`Slice` owns no simulation state. Read operations deliberately
    return non-aliasing tensor snapshots, while writes mutate the registered
    state on the owning model through the explicit Slice mutation contract.
    """

    _RESERVED = (
        "_model",
        "index_spec",
        "_relative_index_spec",
        "_base_shape",
        "parent_slice",
        "root_model",
        "module_path",
        "_labels",
    )

    def __init__(
        self,
        model,
        index_spec: IndexSpec,
        base_shape=None,
        parent_slice=None,
        *,
        root_model=None,
        module_path=(),
        relative_index_spec: IndexSpec | None = None,
    ):
        # Bypass interception for internal fields
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "index_spec", index_spec)
        object.__setattr__(
            self,
            "_relative_index_spec",
            index_spec if relative_index_spec is None else relative_index_spec,
        )
        object.__setattr__(self, "parent_slice", parent_slice)
        object.__setattr__(
            self, "root_model", model if root_model is None else root_model
        )
        object.__setattr__(self, "module_path", tuple(module_path))
        object.__setattr__(self, "_labels", {})
        if base_shape is None:
            base_shape = index_spec.source_shape or tuple(
                (model if root_model is None else root_model).shape
            )
        if not index_spec.source_shape:
            index_spec.source_shape = tuple(base_shape)
        object.__setattr__(self, "_base_shape", tuple(base_shape))

    def _sync_model(self):
        """Resolve a wrapped submodule against the population's live module tree."""
        root = object.__getattribute__(self, "root_model")
        current = root
        path = object.__getattribute__(self, "module_path")
        for part in path:
            modules = getattr(current, "_modules", {})
            if part not in modules or modules[part] is None:
                dotted = ".".join(path)
                raise RuntimeError(
                    f"Slice submodule {dotted!r} is no longer present on the model. "
                    "Create a new Slice after changing the model structure."
                )
            current = modules[part]
        object.__setattr__(self, "_model", current)
        return current

    @staticmethod
    def _index_on_device(index, device):
        """Move owned tensor indices with the model while preserving key syntax."""
        moved = []
        for item in index:
            if torch.is_tensor(item):
                moved.append(item.to(device=device))
            else:
                moved.append(item)
        return tuple(moved)

    def _sync_index(self):
        """Rebase a logical core selection across newly added batch axes."""
        root = object.__getattribute__(self, "root_model")
        root_shape = tuple(root.shape)
        spec = object.__getattribute__(self, "index_spec")
        source_shape = tuple(spec.source_shape)
        device = root.device()

        index_moved = any(
            torch.is_tensor(item) and item.device != device for item in spec.index
        )
        index = self._index_on_device(spec.index, device)
        if root_shape != source_shape:
            adds_leading_axes = (
                len(root_shape) >= len(source_shape)
                and tuple(root_shape[-len(source_shape) :]) == source_shape
            )
            if not adds_leading_axes:
                raise RuntimeError(
                    "This Slice was created for logical model shape "
                    f"{source_shape}, but the model now has incompatible shape "
                    f"{root_shape}. Create a new Slice after changing model topology."
                )

            n_new = len(root_shape) - len(source_shape)
            if n_new and not (index and index[0] is Ellipsis):
                index = (slice(None),) * n_new + index

        if root_shape != source_shape or index_moved:
            out = torch.empty(root_shape, device=device, dtype=torch.bool)[index]
            spec.index = index
            spec.is_scalar = out.ndim == 0
            spec.shape = out.shape
            spec.source_shape = root_shape

        object.__setattr__(self, "_base_shape", root_shape)
        return spec

    def _sync_relative_index(self):
        """Rebase this Slice's parent-relative key across new batch axes.

        ``index_spec`` describes the composed selection in root-population
        coordinates.  Label propagation additionally needs the key that was
        applied to the immediate parent Slice.  Keeping and synchronizing that
        key avoids trying to infer indexing history from a composed advanced
        index, which is not generally possible.
        """
        spec = object.__getattribute__(self, "_relative_index_spec")
        parent = object.__getattribute__(self, "parent_slice")

        # A top-level Slice's parent-relative and root-relative selections are
        # the same owned IndexSpec.
        if parent is None and spec is object.__getattribute__(self, "index_spec"):
            return self._sync_index()

        root = object.__getattribute__(self, "root_model")
        if parent is None:
            parent_shape = tuple(root.shape)
        else:
            parent._sync()
            parent_shape = tuple(parent.shape)

        source_shape = tuple(spec.source_shape)
        device = root.device()
        index_moved = any(
            torch.is_tensor(item) and item.device != device for item in spec.index
        )
        index = self._index_on_device(spec.index, device)

        if parent_shape != source_shape:
            suffix_matches = (
                not source_shape
                or tuple(parent_shape[-len(source_shape) :]) == source_shape
            )
            adds_leading_axes = (
                len(parent_shape) >= len(source_shape) and suffix_matches
            )
            if not adds_leading_axes:
                raise RuntimeError(
                    "This Slice's relative index was created for parent shape "
                    f"{source_shape}, but its parent now has incompatible shape "
                    f"{parent_shape}. Create a new Slice after changing model topology."
                )

            n_new = len(parent_shape) - len(source_shape)
            if n_new and not (index and index[0] is Ellipsis):
                index = (slice(None),) * n_new + index

        if parent_shape != source_shape or index_moved:
            out = torch.empty(parent_shape, device=device, dtype=torch.bool)[index]
            spec.index = index
            spec.is_scalar = out.ndim == 0
            spec.shape = out.shape
            spec.source_shape = parent_shape

        return spec

    def _sync(self):
        self._sync_model()
        return self._sync_index()

    @property
    def model(self):
        """Live population or submodule wrapped by this Slice."""
        return self._sync_model()

    @property
    def base_shape(self):
        """Current logical shape of the owning population."""
        self._sync_index()
        return object.__getattribute__(self, "_base_shape")

    # -------------------------
    # Simple, safe properties
    # -------------------------
    @property
    def index(self) -> Tuple[IndexElement, ...]:
        """Stable tuple index describing this slice in the current model layout."""
        return self._sync().index

    @property
    def flat_index(self) -> torch.LongTensor:
        """Return selected locations as population-flat integer indices.

        The result is always a one-dimensional ``torch.long`` tensor on the
        population's device.  Its order and multiplicity match the flattened
        Slice result, so scalar selections contain one index, empty selections
        contain none, and repeated selections retain their repeats.  A fresh
        tensor is materialized on every access and may be sampled, permuted, or
        otherwise modified without changing this Slice.  Access materializes
        an integer grid with one entry per location in the logical population.
        """
        spec = self._sync()
        root = object.__getattribute__(self, "root_model")
        return _population_flat_indices(root, spec.index)

    @property
    def shape(self):
        """Shape produced by applying :attr:`index` to the underlying population."""
        return self._sync().shape

    @property
    def is_scalar(self) -> bool:
        """Whether the selection is scalar-valued (no remaining dimensions)."""
        return self._sync().is_scalar

    def numel(self) -> int:
        """Return the number of selected elements."""
        return math.prod(self.shape)

    def __len__(self) -> int:
        """Return the size of the first selected dimension, like ``len(tensor)``."""
        if self.is_scalar:
            raise TypeError("len() of a scalar Slice")
        return int(self.shape[0])

    @property
    def name(self) -> str:
        """Name of the underlying population (delegated from ``model.name``)."""
        return self.model.name

    @property
    def is_empty(self) -> bool:
        """Return ``True`` if this slice selects no elements."""
        return self.numel() == 0

    # -------------------------
    # Spatial tensor routing
    # -------------------------
    def _root_shape(self) -> tuple[int, ...]:
        return tuple(object.__getattribute__(self, "root_model").shape)

    def _core_ndim(self) -> int:
        root = object.__getattribute__(self, "root_model")
        core_shape = getattr(root, "core_shape", None)
        if callable(core_shape):
            return len(tuple(core_shape()))
        return len(self._root_shape())

    @staticmethod
    def _is_mapper(model) -> bool:
        return (
            getattr(model, "key", None) is not None
            and callable(getattr(model, "get", None))
            and callable(getattr(model, "put", None))
        )

    def _find_mapper(self, owner):
        if self._is_mapper(owner):
            return owner

        root = object.__getattribute__(self, "root_model")
        current = root
        mapper = current if self._is_mapper(current) else None
        for part in object.__getattribute__(self, "module_path"):
            current = current._modules[part]
            if self._is_mapper(current):
                mapper = current
            if current is owner:
                break
        return mapper

    @staticmethod
    def _missing_value(dtype: torch.dtype):
        if dtype.is_floating_point or dtype.is_complex:
            return torch.nan
        return 0

    @staticmethod
    def _values_equal(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        equal = left == right
        if left.is_floating_point() or left.is_complex():
            equal = equal | (torch.isnan(left) & torch.isnan(right))
        return equal

    def _mapper_is_unique(self, mapper) -> bool:
        support_map = getattr(mapper, "support_map", None)
        if support_map is not None:
            return support_map.is_injective(getattr(mapper, "key", None))

        cached = getattr(mapper, "_dendra_slice_mapping_is_unique", None)
        if cached is not None:
            return bool(cached)

        root = object.__getattribute__(self, "root_model")
        logical_shape = tuple(root.shape)
        grid = torch.arange(
            math.prod(logical_shape), device=root.device(), dtype=torch.long
        ).reshape(logical_shape)
        mapped = mapper.get(grid).reshape(-1)
        unique = int(torch.unique(mapped).numel()) == int(mapped.numel())
        setattr(mapper, "_dendra_slice_mapping_is_unique", unique)
        return unique

    def _logical_tensor(self, owner, name: str, tensor: torch.Tensor, mapper=None):
        """Return ``tensor`` expanded/scattered into population coordinates."""
        root_shape = self._root_shape()
        if tensor.ndim == 0:
            raise ValueError(
                f"{name!r} is global scalar state and has no Slice coordinates; "
                f"access it on {type(owner).__name__} directly."
            )

        if mapper is None:
            mapper = self._find_mapper(owner)

        if mapper is not None:
            if name == "key":
                raise ValueError(
                    "Mechanism key metadata is not a spatial field; access it on "
                    "the mechanism directly."
                )
            if not self._mapper_is_unique(mapper):
                raise ValueError(
                    "This mechanism has multiple independent slots at the same "
                    "physical compartment. A physical Slice cannot represent "
                    "slot-local state; access the mechanism storage directly or "
                    "select explicit SynapseSlots."
                )
            template = torch.empty(root_shape, device=tensor.device, dtype=tensor.dtype)
            local_shape = tuple(mapper.get(template).shape)
            try:
                broadcast_shape = torch.broadcast_shapes(
                    tuple(tensor.shape), local_shape
                )
            except RuntimeError as error:
                raise ValueError(
                    f"{name!r} with storage shape {tuple(tensor.shape)} is not "
                    f"spatially compatible with mechanism shape {local_shape}."
                ) from error
            if tuple(broadcast_shape) != local_shape:
                raise ValueError(
                    f"{name!r} with storage shape {tuple(tensor.shape)} is not "
                    f"spatially compatible with mechanism shape {local_shape}."
                )
            local = torch.broadcast_to(tensor, local_shape)
            dense = torch.full(
                root_shape,
                self._missing_value(tensor.dtype),
                device=tensor.device,
                dtype=tensor.dtype,
            )
            dense = mapper.put(local, dense, template)
            return dense, local_shape, mapper

        if tensor.ndim < self._core_ndim():
            raise ValueError(
                f"{name!r} with shape {tuple(tensor.shape)} is not spatially "
                f"indexed over model shape {root_shape}; access it on the owning "
                "model directly."
            )
        try:
            broadcast_shape = torch.broadcast_shapes(tuple(tensor.shape), root_shape)
        except RuntimeError as error:
            raise ValueError(
                f"{name!r} with shape {tuple(tensor.shape)} is not spatially "
                f"compatible with model shape {root_shape}."
            ) from error
        if tuple(broadcast_shape) != root_shape:
            raise ValueError(
                f"{name!r} with shape {tuple(tensor.shape)} is not spatially "
                f"compatible with model shape {root_shape}."
            )
        return torch.broadcast_to(tensor, root_shape), root_shape, None

    def _support_mask(self, mapper, local_shape, *, device) -> torch.Tensor:
        root_shape = self._root_shape()
        local = torch.ones(local_shape, device=device, dtype=torch.bool)
        dense = torch.zeros(root_shape, device=device, dtype=torch.bool)
        template = torch.empty(root_shape, device=device, dtype=torch.bool)
        return mapper.put(local, dense, template)

    def _read_tensor(self, owner, name: str, tensor: torch.Tensor, mapper=None):
        spec = self._sync()
        logical, local_shape, mapper = self._logical_tensor(
            owner, name, tensor, mapper=mapper
        )
        if mapper is not None and not (
            tensor.dtype.is_floating_point or tensor.dtype.is_complex
        ):
            support = self._support_mask(mapper, local_shape, device=tensor.device)
            if not bool(torch.all(support[spec.index])):
                raise ValueError(
                    f"Slice includes locations where mechanism field {name!r} is "
                    "not present, and its dtype has no NaN missing-value marker."
                )
        return logical[spec.index].clone()

    def _registered_tensor(self, owner, name: str) -> torch.Tensor:
        buffers = getattr(owner, "_buffers", {})
        parameters = getattr(owner, "_parameters", {})
        if name in buffers:
            tensor = buffers[name]
        elif name in parameters:
            tensor = parameters[name]
        else:
            raise AttributeError(
                f"{type(owner).__name__}.{name} is not a registered writable "
                "buffer or Parameter."
            )
        if tensor is None or not torch.is_tensor(tensor):
            raise AttributeError(f"{name!r} is not a writable tensor field.")
        return tensor

    def _assignment_values(self, index, value, tensor: torch.Tensor):
        root_shape = self._root_shape()
        coordinates = torch.arange(
            math.prod(root_shape), device=tensor.device, dtype=torch.long
        ).reshape(root_shape)[index]
        assigned = torch.empty(
            coordinates.shape, device=tensor.device, dtype=tensor.dtype
        )
        try:
            assigned[...] = value
        except (RuntimeError, TypeError, ValueError) as error:
            value_shape = tuple(value.shape) if torch.is_tensor(value) else "scalar"
            raise ValueError(
                f"Cannot assign value with shape {value_shape} to Slice shape "
                f"{tuple(coordinates.shape)}."
            ) from error

        flat_coordinates = coordinates.reshape(-1)
        if flat_coordinates.numel() > 1:
            order = torch.argsort(flat_coordinates)
            sorted_coordinates = flat_coordinates[order]
            sorted_values = assigned.reshape(-1)[order]
            repeated = sorted_coordinates[1:] == sorted_coordinates[:-1]
            conflicts = repeated & ~self._values_equal(
                sorted_values[1:], sorted_values[:-1]
            )
            if bool(torch.any(conflicts)):
                raise ValueError(
                    "Slice contains repeated physical locations with conflicting "
                    "assigned values. Use one value per physical location."
                )
        return assigned

    def _collapse_to_storage(
        self, logical: torch.Tensor, storage_shape: tuple[int, ...], *, name: str
    ) -> torch.Tensor:
        logical_shape = tuple(logical.shape)
        if len(storage_shape) > len(logical_shape):
            raise ValueError(
                f"Cannot represent logical field {name!r} in storage shape "
                f"{storage_shape}."
            )
        aligned = (1,) * (len(logical_shape) - len(storage_shape)) + storage_shape
        if any(s not in (1, d) for s, d in zip(aligned, logical_shape)):
            raise ValueError(
                f"Cannot represent logical field {name!r} in storage shape "
                f"{storage_shape}."
            )

        collapsed = logical
        for dim, (stored, expanded) in enumerate(zip(aligned, logical_shape)):
            if stored == 1 and expanded != 1:
                representative = collapsed.narrow(dim, 0, 1)
                if not bool(
                    torch.all(
                        self._values_equal(
                            collapsed, representative.expand_as(collapsed)
                        )
                    )
                ):
                    raise ValueError(
                        f"Slice write to {name!r} requests different values along "
                        "a dimension shared by its broadcast storage. The field "
                        "cannot represent batch-dependent values."
                    )
                collapsed = representative
        return collapsed.reshape(storage_shape)

    def _write_tensor(self, owner, name: str, value, mapper=None):
        spec = self._sync()
        tensor = self._registered_tensor(owner, name)
        logical, local_shape, mapper = self._logical_tensor(
            owner, name, tensor, mapper=mapper
        )

        with torch.no_grad():
            assigned = self._assignment_values(spec.index, value, tensor)
            if mapper is None and tuple(tensor.shape) == self._root_shape():
                tensor[spec.index] = assigned
                return self

            if mapper is not None:
                support = self._support_mask(mapper, local_shape, device=tensor.device)
                if not bool(torch.all(support[spec.index])):
                    raise ValueError(
                        f"Slice write to mechanism field {name!r} includes locations "
                        "outside that mechanism's insertion region."
                    )

            updated = logical.clone()
            updated[spec.index] = assigned
            stored = mapper.get(updated) if mapper is not None else updated
            collapsed = self._collapse_to_storage(
                stored, tuple(tensor.shape), name=name
            )
            tensor.copy_(collapsed)
        return self

    @staticmethod
    def _stable_unique_indices(indices: torch.Tensor) -> torch.LongTensor:
        """Return first-occurrence unique indices without changing their order."""
        flat = indices.reshape(-1).to(dtype=torch.long)
        if flat.numel() < 2:
            return flat
        seen = set()
        positions = []
        for position, value in enumerate(flat.detach().cpu().tolist()):
            if value not in seen:
                seen.add(value)
                positions.append(position)
        return flat.index_select(
            0, torch.as_tensor(positions, device=flat.device, dtype=torch.long)
        )

    def _parameter_key(self, model, name: str) -> torch.LongTensor:
        """Map a logical Slice selection into a parameter field's storage."""
        spec = self._sync()
        target = getattr(model, name)
        if not torch.is_tensor(target):
            raise ValueError(f"Parameter field {name!r} is not tensor-valued.")
        storage_indices = torch.arange(
            target.numel(), device=target.device, dtype=torch.long
        ).reshape(target.shape)
        logical, local_shape, mapper = self._logical_tensor(
            model, name, storage_indices
        )
        if mapper is not None:
            support = self._support_mask(mapper, local_shape, device=target.device)
            if not bool(torch.all(support[spec.index])):
                raise ValueError(
                    f"Slice parameterization of {name!r} includes locations "
                    "outside that mechanism's insertion region."
                )
        key = self._stable_unique_indices(logical[spec.index])
        is_batch = (
            name in getattr(model, "batch_t", {})
            or name in getattr(model, "batch_p", {})
            or name in getattr(model, "batch_n", {})
        )
        if is_batch:
            # ``Parameterized.parametrize`` accepts coordinates in the RANGE
            # grid and collapses its final compartment axis for BATCH fields.
            # The logical projection above necessarily starts with already-
            # collapsed storage indices, so lift them back into one canonical
            # RANGE coordinate before handing them off. Without this step an
            # (N, 1) BATCH buffer would be divided by K a second time.
            key = key * int(model.shape_p[-1])
        return key

    def _core_index_spec(self, *, preserve_multiplicity: bool = False) -> IndexSpec:
        """Project logical batch coordinates onto structural core coordinates."""
        spec = self._sync()
        root = object.__getattribute__(self, "root_model")
        root_shape = tuple(root.shape)
        core_shape = tuple(root.core_shape())
        core_numel = math.prod(core_shape)
        full = torch.arange(
            math.prod(root_shape), device=root.device(), dtype=torch.long
        ).reshape(root_shape)
        selected = full[spec.index].reshape(-1)
        core_indices = torch.remainder(selected, core_numel)

        if preserve_multiplicity and core_indices.numel() > 0:
            batch_indices = torch.div(selected, core_numel, rounding_mode="floor")
            counts: dict[tuple[int, int], int] = {}
            order = []
            for batch_index, core_index in zip(
                batch_indices.detach().cpu().tolist(),
                core_indices.detach().cpu().tolist(),
            ):
                pair = (batch_index, core_index)
                counts[pair] = counts.get(pair, 0) + 1
                if core_index not in order:
                    order.append(core_index)
            max_counts = {
                core_index: max(
                    count
                    for (batch_index, candidate), count in counts.items()
                    if candidate == core_index
                )
                for core_index in order
            }
            projected = [
                core_index
                for core_index in order
                for _ in range(max_counts[core_index])
            ]
            core_indices = torch.as_tensor(
                projected, device=root.device(), dtype=torch.long
            )
        else:
            core_indices = self._stable_unique_indices(core_indices)

        core_key = torch.unravel_index(core_indices, core_shape)
        return parse_key(core_key, core_shape, device=root.device())

    # -------------------------
    # Public API
    # -------------------------
    def inspect(self, var: str, mechanism: Optional[str] = None) -> Any:
        """
        Return a non-aliasing snapshot of ``var`` constrained to the slice.

        This is the main low-level accessor for reading model or mechanism
        state. It honours any sparse/keyed storage used by mechanisms and
        returns a value whose leading dimensions correspond to this slice.

        Parameters
        ----------
        var : str
            Attribute name to inspect. This may refer to a model-level buffer
            or parameter (e.g. ``"v"``), or to a mechanism field such as a
            gating variable.
        mechanism : str or None, optional
            Mechanism identifier when querying mechanism state. If ``None``,
            ``var`` is looked up directly on the wrapped ``model``.

        Returns
        -------
        Any
            Snapshot of the requested spatial tensor with shape
            :attr:`shape`. Global and non-spatial tensors raise ``ValueError``.

        Examples
        --------
        Read membrane voltage in the soma:

        .. code-block:: python

            soma = pop[:, 0]
            v_soma = soma.inspect("v")       # shape == soma.shape

        Read a mechanism state variable:

        .. code-block:: python

            m_gate = soma.inspect("m", mechanism="NaTs2t")
        """
        model = self.model

        if mechanism is not None:
            mech = model.mech.mechanisms[mechanism]
            value = getattr(mech, var)
            if not torch.is_tensor(value):
                raise TypeError(
                    f"Mechanism field {mechanism}.{var} is not tensor-valued."
                )
            mapper = mech if self._is_mapper(mech) else None
            return self._read_tensor(mech, var, value, mapper=mapper)

        value = getattr(model, var)
        if not torch.is_tensor(value):
            raise TypeError(f"{type(model).__name__}.{var} is not tensor-valued.")
        return self._read_tensor(model, var, value)

    def _inspect(self, var: str):
        """Inspect ``var`` on the wrapped model without mechanism handling.

        This is an internal helper used to implement attribute access for
        buffers, parameters and submodules. It respects sparse/keyed storage
        when the underlying model exposes a ``key`` attribute.
        """
        model = self.model
        value = getattr(model, var)
        if not torch.is_tensor(value):
            raise TypeError(f"{type(model).__name__}.{var} is not tensor-valued.")
        return self._read_tensor(model, var, value)

    def get(self, var: str, mechanism: Optional[str] = None) -> torch.Tensor:
        """
        Convenience alias for :meth:`inspect` returning a tensor snapshot.

        This method simply forwards to :meth:`inspect` and is provided for
        readability in user code that predominantly deals with tensor-valued
        variables.

        Parameters
        ----------
        var : str
            Attribute name to read.
        mechanism : str or None, optional
            Mechanism identifier when querying mechanism state.

        Returns
        -------
        torch.Tensor
            Non-aliasing tensor snapshot restricted to the slice.

        Examples
        --------
        .. code-block:: python

            v_soma = pop[:, 0].get("v")
            m_gate = pop[:, 0].get("m", mechanism="NaTs2t")
        """
        return self.inspect(var, mechanism)

    def set(self, var: str, value: torch.Tensor, mechanism: Optional[str] = None):
        """
        Write ``value`` into ``var`` constrained to the slice.

        This is the main low-level mutator for updating state on a subset of
        compartments. The complete update is validated before an in-place copy
        under ``torch.no_grad()``. Registered buffer or Parameter identity,
        dtype, device and unrelated entries are preserved.

        Parameters
        ----------
        var : str
            Attribute name to mutate. May refer to a model-level buffer or
            parameter, or to a mechanism field.
        value : torch.Tensor
            Tensor data to assign. Its shape must be broadcastable to the
            slice's :attr:`shape`.
        mechanism : str or None, optional
            Mechanism identifier when writing mechanism state. If provided,
            the appropriate mechanism's storage is updated; otherwise the
            attribute is looked up on the wrapped ``model``.

        Examples
        --------
        Set the soma voltage to a constant:

        .. code-block:: python

            soma = pop[:, 0]
            target = torch.full(soma.shape, -65.0, device=pop.device())
            soma.set("v", target)

        Modify a mechanism gating variable:

        .. code-block:: python

            soma.set("m", new_m_values, mechanism="NaTs2t")
        """
        model = self.model

        if mechanism is not None:
            mech = model.mech.mechanisms[mechanism]
            mapper = mech if self._is_mapper(mech) else None
            return self._write_tensor(mech, var, value, mapper=mapper)

        return self._write_tensor(model, var, value)

    def inject(self, waveform):
        """
        Register an intracellular current waveform targeting this slice.

        The waveform is not applied immediately. Instead, this method records
        the tuple ``(waveform, index_spec.shape, index_spec.index)`` on the
        underlying population's ``injections`` list and, when available, on the
        population's mechanism-level injection registry. The solver path applies
        the current to the selected compartments in the usual way, while
        mechanisms that implement ``inject(...)`` can consume the waveform
        directly.

        Parameters
        ----------
        waveform : Any
            :py:class:`~dendra.models.stim.waveform.core.Waveform` object
            understood by the population's injection infrastructure.

        Notes
        -----
        Calling :meth:`inject` on an empty slice is a no-op.

        Examples
        --------
        .. code-block:: python

            soma = pop[:, 0]
            soma.inject(step_current)        # step_current defined elsewhere
        """
        if self.is_empty:
            return  # no-op for empty slices
        model = self.model
        index_spec = self._sync()
        if hasattr(model, "register_injection"):
            model.register_injection(waveform, index_spec)
        else:
            model.injections.append((waveform, index_spec.shape, index_spec.index))

    def slots(self, synapse, slot_index=None, *, local_index=None):
        """Return synapse-local slots for ``synapse`` that lie in this slice.

        ``Slice`` selects physical compartments.  For banked point-process
        mechanisms, multiple independent slots may live on each selected
        compartment; this method exposes those slots explicitly for network
        connectivity.
        """
        if local_index is not None:
            if slot_index is not None:
                raise ValueError("Provide either slot_index or local_index, not both.")
            slot_index = local_index
        return SynapseSlots.from_region(
            object.__getattribute__(self, "root_model"),
            synapse,
            region_index=self.index,
            slot_index=slot_index,
        )

    def _visible_label(self, name: str):
        """Return the nearest owned label and the relative path to this Slice."""
        if object.__getattribute__(self, "module_path"):
            return None

        path = []
        current = self
        while True:
            registry = object.__getattribute__(current, "_labels")
            if name in registry:
                return registry[name], tuple(reversed(path))

            path.append(current)
            parent = object.__getattribute__(current, "parent_slice")
            if parent is None:
                break
            current = parent

        root = object.__getattribute__(self, "root_model")
        registry = getattr(root, "_labels", None)
        if isinstance(registry, dict) and name in registry:
            return registry[name], tuple(reversed(path))
        return None

    @staticmethod
    def _is_full_slice(item) -> bool:
        return (
            isinstance(item, slice)
            and item.start is None
            and item.stop is None
            and item.step is None
        )

    @classmethod
    def _label_key_variants(cls, index):
        """Yield a key plus variants with redundant full slices removed.

        A label may reduce an axis that was explicitly retained as ``:`` in
        the descendant key.  Removing only full slices is semantics-neutral
        with respect to that axis; the subsequent root-coordinate validation
        decides whether a variant truly commutes with the label.  ``None``,
        advanced indices, non-full slices, and their relative ordering are
        never rewritten.
        """
        index = tuple(index)
        yield 0, index
        full_positions = [
            position for position, item in enumerate(index) if cls._is_full_slice(item)
        ]
        for count in range(1, len(full_positions) + 1):
            for removed in combinations(full_positions, count):
                removed = set(removed)
                yield (
                    count,
                    tuple(
                        item
                        for position, item in enumerate(index)
                        if position not in removed
                    ),
                )

    def _label_propagation_error(self, name: str, detail: str | None = None):
        message = (
            f"Slice label {name!r} cannot be propagated through this Slice "
            "because its relative indexing path does not commute exactly with "
            "the labelled selection. Access the label from its owner and index "
            "it explicitly, or use slice.intersect(owner_label) for a canonical "
            "one-dimensional physical intersection."
        )
        if detail:
            message = f"{message} {detail}"
        raise AttributeError(message)

    @staticmethod
    def _same_label_candidate(root, left, right) -> bool:
        """Return whether two replay candidates have identical Slice semantics."""
        if left.shape != right.shape or left.is_scalar is not right.is_scalar:
            return False
        left_flat = _population_flat_indices(root, left.index)
        right_flat = _population_flat_indices(root, right.index)
        return bool(torch.equal(left_flat, right_flat))

    def _retain_label_candidate(self, name, candidates, candidate, removed):
        """Deduplicate replay states, keeping their least-rewritten path."""
        root = object.__getattribute__(self, "root_model")
        for position, (existing, existing_removed) in enumerate(candidates):
            if self._same_label_candidate(root, candidate, existing):
                if removed < existing_removed:
                    candidates[position] = (candidate, removed)
                return
        if len(candidates) >= 256:
            self._label_propagation_error(
                name,
                "The indexing path admits too many distinct full-slice rewrites "
                "to validate safely.",
            )
        candidates.append((candidate, removed))

    def _propagate_label(self, name: str, label, path):
        """Replay and validate a visible label through this Slice's ancestry."""
        if label is self:
            return self
        if not isinstance(label, Slice):
            self._label_propagation_error(
                name, "The owner's label registry does not contain a Slice."
            )

        root = object.__getattribute__(self, "root_model")
        if object.__getattribute__(label, "root_model") is not root:
            self._label_propagation_error(
                name, "The labelled Slice belongs to a different root population."
            )

        self._sync()
        label._sync()
        label_flat = _population_flat_indices(root, label.index)
        if int(torch.unique(label_flat).numel()) != int(label_flat.numel()):
            self._label_propagation_error(
                name,
                "The label contains duplicate physical compartments, so implicit "
                "propagation would make multiplicity ambiguous.",
            )

        current_flat = _population_flat_indices(root, self.index)
        expected = current_flat[torch.isin(current_flat, label_flat)]

        candidates = [(label, 0)]
        replay_attempts = 0
        for descendant in path:
            relative = descendant._sync_relative_index()
            replayed = []
            for candidate, total_removed in candidates:
                for removed, key in self._label_key_variants(relative.index):
                    replay_attempts += 1
                    if replay_attempts > 4096:
                        self._label_propagation_error(
                            name,
                            "The indexing path admits too many full-slice rewrites "
                            "to validate safely.",
                        )
                    try:
                        replayed_candidate = candidate[key]
                    except IndexError:
                        continue
                    self._retain_label_candidate(
                        name,
                        replayed,
                        replayed_candidate,
                        total_removed + removed,
                    )
            if not replayed:
                self._label_propagation_error(name)
            candidates = replayed

        expected_sorted = torch.sort(expected).values
        valid = []
        for candidate, removed in candidates:
            candidate_flat = _population_flat_indices(root, candidate.index)
            if int(expected.numel()) != int(candidate_flat.numel()):
                continue
            if not expected.numel() or torch.equal(
                expected_sorted, torch.sort(candidate_flat).values
            ):
                valid.append((candidate, candidate_flat, removed))

        if valid:
            fewest_removed = min(item[2] for item in valid)
            minimal = [item for item in valid if item[2] == fewest_removed]
            reference, reference_flat, _ = minimal[0]
            for candidate, candidate_flat, _ in minimal[1:]:
                same_semantics = (
                    candidate.shape == reference.shape
                    and candidate.is_scalar is reference.is_scalar
                    and torch.equal(candidate_flat, reference_flat)
                )
                if not same_semantics:
                    self._label_propagation_error(
                        name,
                        "Removing redundant full slices admits more than one "
                        "shape or traversal order, so propagation is ambiguous.",
                    )
            return reference

        self._label_propagation_error(name)

    def insert(
        self,
        mechanism,
        alias=None,
        ic=None,
        preserve_duplicate_indices: bool = False,
        preserve_multiplicity: bool | None = None,
        copies: int = 1,
        **kwargs,
    ):
        """
        Insert a mechanism restricted to this slice.

        Only the selected compartments will host the inserted mechanism. This
        method forwards to ``model.insert`` while passing along the slice's
        :class:`IndexSpec`, so the underlying population can allocate and wire
        the mechanism appropriately.

        Parameters
        ----------
        mechanism : Any
            Mechanism class.
        alias : str or None, optional
            Optional alias with which the mechanism should be registered on
            the model. If ``None``, the default aliasing behaviour of
            ``model.insert`` is used.
        ic : Any, optional
            Optional initial-condition mapping. It is class-wide because all
            regions of one mechanism class compile into one mechanism instance.
        preserve_duplicate_indices : bool, optional
            If True, duplicate selected compartments are retained as independent
            mechanism slots.  This is useful for colocated point-process banks
            that should keep separate state while scattering current to the same
            compartment.
        preserve_multiplicity : bool, optional
            User-facing synonym for ``preserve_duplicate_indices``.
        copies : int, optional
            Number of independent copies of this insertion region to allocate.
            ``copies > 1`` implies duplicate preservation.
        **kwargs
            Mechanism parameters forwarded to ``model.insert``. RANGE/BATCH
            values apply to this Slice; GLOBAL values are class-wide and must
            agree across insertion records for the same mechanism class.

        Notes
        -----
        Calling :meth:`insert` on an empty slice is a no-op.

        Examples
        --------
        .. code-block:: python

            soma = pop[:, 0]
            soma.insert(NaTs2t, alias="Na_soma")

            # Insert a passive mechanism only in dendrites
            pop[:, 1:].insert(pas, alias="pas_dend", g=1e-4)  # S/cm²
        """
        if self.is_empty:
            return  # no-op for empty slices
        if preserve_multiplicity is not None:
            preserve_duplicate_indices = bool(
                preserve_duplicate_indices or preserve_multiplicity
            )
        structural_index = self._core_index_spec(
            preserve_multiplicity=preserve_duplicate_indices and int(copies) == 1
        )
        self.model.insert(
            mechanism,
            alias=alias,
            index_spec=structural_index,
            ic=ic,
            preserve_duplicate_indices=preserve_duplicate_indices,
            copies=copies,
            **kwargs,
        )

    def delete(self, mechanism, *, strict=False):
        """Delete a mechanism class where it intersects this Slice.

        Parameters
        ----------
        mechanism : type
            Exact mechanism class to remove.
        strict : bool, default False
            If True, require every selected physical compartment to host the
            mechanism and fail atomically otherwise. If False, remove the
            mechanism only where it is present.

        Notes
        -----
        Overlapping insertion aliases are all cropped, and every
        copied/duplicate mechanism slot at a selected physical compartment is
        removed. On a batched population, the Slice is projected onto the
        shared structural core, so the deletion applies to every batch replica.
        Calling this method on an empty Slice is a no-op. Reinitialize the
        population after a successful deletion.

        Examples
        --------
        .. code-block:: python

            # Remove pas from dendrites while retaining it in the soma.
            pop.dendrites.delete(pas)

            # Remove the class from all remaining compartments.
            pop.delete(pas)
        """
        if self.is_empty:
            return
        root = object.__getattribute__(self, "root_model")
        structural_index = self._core_index_spec()
        root.delete(mechanism, index=structural_index, strict=strict)

    def delete_all(self):
        """Delete every mechanism class present anywhere in this Slice.

        The operation uses physical support-intersection semantics for each
        exact configured class and is transactional across classes. On a
        batched Population the structural selection applies to every replica.
        An empty Slice is a no-op. Reinitialize after a successful deletion.
        """
        if self.is_empty:
            return
        root = object.__getattribute__(self, "root_model")
        structural_index = self._core_index_spec()
        root.delete_all(index=structural_index)

    def parametrize(self, name, value, alias=None):
        """
        Add or update an alias-specific parameter override on this slice.

        This method forwards to ``model.parametrize`` while passing along the
        slice's :class:`IndexSpec`, so that only the selected compartments
        receive the parameter override.

        Parameters
        ----------
        name : str
            Parameter name to override.
        value : float, torch.Tensor, torch.nn.Parameter, or torch.nn.Module
            New parameter value.
        alias : str or None, optional
            Optional mechanism alias to which the parameter applies. If
            ``None``, the parameter is assumed to be a model-level parameter.

        Examples
        --------
        .. code-block:: python

            # Override a model-level parameter in the soma
            pop[:, 0].parametrize("rhoa", 150.0)

            # Override a mechanism parameter in dendrites
            # Passive conductance density is expressed in S/cm².
            pop[:, 1:].mech.pas.parametrize("g", 1e-4, alias="dend")
        """
        if self.is_empty:
            return  # no-op for empty slices
        model = self.model
        root = object.__getattribute__(self, "root_model")
        mechanism_name = None
        handler = getattr(root, "mech", None)
        mechanisms = getattr(handler, "mechanisms", {})
        for candidate_name, candidate in mechanisms.items():
            if candidate is model:
                mechanism_name = candidate_name
                break

        persistent_region = None
        if mechanism_name is not None and hasattr(
            root, "_register_slice_mechanism_parametrization"
        ):
            core_shape = tuple(root.core_shape())
            core_spec = self._core_index_spec()
            core_grid = torch.arange(
                math.prod(core_shape), device=root.device(), dtype=torch.long
            ).reshape(core_shape)
            persistent_region = (
                core_grid[core_spec.index].reshape(-1),
                core_shape,
            )

        resolved_alias = model.parametrize(
            name,
            value,
            key=self._parameter_key(model, name),
            alias=alias,
        )
        if persistent_region is not None:
            core_indices, core_shape = persistent_region
            mechanism_class = type(model)
            resolve_class = getattr(
                root, "_configured_mechanism_class_for_instance", None
            )
            if resolve_class is not None:
                mechanism_class = resolve_class(mechanism_name, model)
            root._register_slice_mechanism_parametrization(
                mechanism_name,
                mechanism_class,
                name,
                value,
                core_indices,
                core_shape,
                resolved_alias,
            )

    def label(self, name: str, *, replace: bool = False):
        """
        Attach a label to the slice for convenient, reusable access.

        This method exposes the slice under the given ``name`` as an attribute
        on the owning population (for top-level slices) or on the parent slice
        (for nested slices). For populations, it also adds an entry to the
        ``_labels`` dictionary.

        Parameters
        ----------
        name : str
            Non-private Python identifier to use as the label. Existing model
            attributes and Slice API names are reserved.
        replace : bool, optional
            Replace an existing Slice label with the same owner and name. This
            never permits replacement of a non-label attribute. Defaults to
            ``False``.

        Returns
        -------
        Slice
            This Slice, allowing ``region = pop[key].label("region")``.

        Examples
        --------
        Label soma and dendrites on a population:

        .. code-block:: python

            pop[:, 0].label("soma")
            pop[:, 1:].label("dendrites")

            # Later, reuse the labels
            pop.soma.inject(step_current)
            pop.dendrites.set(
                "g",
                torch.full(pop.dendrites.shape, 1e-4),  # S/cm²
                mechanism="pas",
            )

        Labels on nested slices attach to the outer slice:

        .. code-block:: python

            distal = pop[:, 5:]
            distal[:, :2].label("distal_proximal")   # stored on ``distal``

            distal.distal_proximal.set("gNa", 0.0)
        """
        if not isinstance(name, str):
            raise TypeError("Slice label names must be strings.")
        if not name.isidentifier() or keyword.iskeyword(name) or name.startswith("_"):
            raise ValueError(
                f"Slice label {name!r} must be a non-private Python identifier."
            )
        if object.__getattribute__(self, "module_path"):
            raise ValueError("Only population-backed Slices can be labelled.")

        slice_api_names = set(type(self)._RESERVED)
        for cls in type(self).__mro__:
            slice_api_names.update(vars(cls))
        if name in slice_api_names:
            raise ValueError(
                f"Slice label {name!r} conflicts with an existing Slice API name."
            )

        parent_slice = object.__getattribute__(self, "parent_slice")
        if parent_slice is not None:
            owner = parent_slice
            registry = object.__getattribute__(owner, "_labels")
            owner_description = "parent Slice"
        else:
            owner = object.__getattribute__(self, "root_model")
            registry = getattr(owner, "_labels", None)
            if not isinstance(registry, dict):
                raise ValueError("The owning model does not support Slice labels.")
            owner_description = type(owner).__name__

        existing = registry.get(name)
        if existing is self:
            return self
        if existing is not None and not replace:
            raise ValueError(
                f"Slice label {name!r} already exists on {owner_description}."
            )

        if parent_slice is not None and existing is None:
            visible = owner._visible_label(name)
            if visible is not None:
                raise ValueError(
                    f"Nested Slice label {name!r} would shadow a label visible "
                    "from an enclosing Slice or population. Choose a unique name."
                )

        replaceable = existing is not None and (
            (name in vars(owner) and vars(owner)[name] is existing)
            or getattr(owner, name, None) is existing
        )
        if hasattr(owner, name) and not (replace and replaceable):
            raise ValueError(
                f"Slice label {name!r} conflicts with an existing attribute on "
                f"{owner_description}."
            )

        # Validation above completes before either namespace is changed.
        if parent_slice is not None:
            object.__setattr__(owner, name, self)
        else:
            setattr(owner, name, self)
        registry[name] = self
        return self

    def __getitem__(self, key):
        """
        Return a nested slice produced by applying ``key``.

        This enables composition of selections: ``pop[idx1][idx2]`` is
        equivalent to ``pop[idx3]`` where ``idx3`` is a single index into the
        original population computed by :func:`compose_indices`.

        Parameters
        ----------
        key : Any
            Index compatible with PyTorch/NumPy semantics, applied to the
            current slice.

        Returns
        -------
        Slice
            New :class:`Slice` that selects a subset of the original population
            corresponding to the composition of the two indices.

        Examples
        --------
        .. code-block:: python

            # Two-stage selection
            distal = pop[:, 5:]
            mid_distal = distal[:, 2:4]

            # Direct equivalent
            direct = pop[:, 7:9]
            assert torch.allclose(mid_distal.v, direct.v)
        """
        self._sync()
        root = object.__getattribute__(self, "root_model")
        relative = parse_key(key, self.shape, device=root.device())
        idx = compose_indices(
            root.shape,
            self.index,
            relative.index,
            device=root.device(),
        )
        idx = parse_key(idx, root.shape, device=root.device())
        return type(self)(
            self.model,
            idx,
            parent_slice=self,
            base_shape=self.base_shape,
            root_model=root,
            module_path=object.__getattribute__(self, "module_path"),
            relative_index_spec=relative,
        )

    def intersect(self, other: Slice) -> Slice:
        """Return the physical intersection with another population Slice.

        Intersection is deliberately an explicit operation because its shape
        contract differs from ordinary Slice indexing.  At creation, the
        result is a canonical one-dimensional Slice.  It preserves the
        traversal order and duplicate occurrences of ``self`` while treating
        ``other`` as a set of physical compartments.  Consequently it never
        widens ``self`` and duplicates in ``other`` do not multiply the result.
        As with any retained Slice, later batching prepends batch axes.

        Both operands must be population-backed Slices of the same root model.
        Mechanism-local and cross-population intersections are rejected rather
        than guessing how their coordinate systems relate.
        """
        if not isinstance(other, Slice):
            raise TypeError("Slice.intersect(other) requires another Slice.")

        self_path = object.__getattribute__(self, "module_path")
        other_path = object.__getattribute__(other, "module_path")
        if self_path or other_path:
            raise ValueError(
                "Slice.intersect() is defined only for population-backed Slices."
            )

        root = object.__getattribute__(self, "root_model")
        other_root = object.__getattribute__(other, "root_model")
        if root is not other_root:
            raise ValueError(
                "Slice.intersect() operands must belong to the same root population."
            )

        self._sync()
        other._sync()
        grid = torch.arange(
            math.prod(tuple(root.shape)), device=root.device(), dtype=torch.long
        ).reshape(tuple(root.shape))
        left = grid[self.index]
        right = grid[other.index].reshape(-1)
        membership = torch.isin(left, right)

        # Boolean indexing supplies the public canonical 1-D contract and also
        # retains a genuine parent-relative key for lifecycle synchronization.
        return self[membership]

    # -------------------------
    # Interceptors
    # -------------------------
    def __setattr__(self, name, value):
        """
        Intercept assignments and route writes into model buffers when possible.

        Behaviour is as follows:

        * Writing an attribute whose name matches a registered spatial buffer
          or Parameter uses the same validated writer as :meth:`set`.
        * Internal attributes used by Slice itself bypass interception.
        * Unknown, global, and non-spatial assignments raise rather than
          creating transient wrapper metadata.

        This allows natural syntax such as ``pop[:, 0].v = value`` while making
        misspelled field names deterministic failures.
        """
        # Always allow internal fields used by construction and synchronization.
        if name in type(self)._RESERVED:
            object.__setattr__(self, name, value)
            return

        model = self.model
        if name in getattr(model, "_buffers", {}) or name in getattr(
            model, "_parameters", {}
        ):
            self._write_tensor(model, name, value)
            return

        raise AttributeError(
            f"Cannot assign unknown or non-writable Slice attribute {name!r}. "
            "Slice assignment is limited to registered spatial buffers and "
            "Parameters; use slice.set(...) for an explicit field write."
        )

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the underlying model when appropriate.

        Resolution order:

        1. If ``name`` is a label owned by this Slice or a visible ancestor,
           return it directly or propagate it through an exactly commuting
           relative indexing path.
        2. If ``name`` matches a buffer on the underlying PyTorch module,
           return a sliced snapshot via :meth:`_inspect`.
        3. If ``name`` matches a submodule, return a new :class:`Slice` that
           wraps the submodule but shares this slice's :class:`IndexSpec`.
        4. If ``name`` matches a parameter, return a sliced snapshot of that
           parameter via :meth:`_inspect`.
        5. Otherwise, delegate attribute access directly to the wrapped model.

        This allows convenient access patterns such as:

        .. code-block:: python

            # Sliced buffer
            v_soma = pop[:, 0].v
            # same as: v_soma = pop[:, 0].get("v")

            # Sliced submodule (e.g. mechanism collection) and mechanism field
            m_gate = pop[:, 0].mech.NaTs2t.m
            # same as: m_gate = pop[:, 0].get("m", mechanism="NaTs2t")

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        Any
            Either a sliced tensor, a new :class:`Slice` on a submodule, or
            the underlying model's attribute.
        """
        # Only runs if normal lookup failed
        try:
            object.__getattribute__(self, "root_model")
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name!r}"
            ) from None
        model = object.__getattribute__(self, "_sync_model")()

        self._sync_index()

        nested_labels = object.__getattribute__(self, "_labels")
        if name in nested_labels:
            return nested_labels[name]

        visible_label = self._visible_label(name)
        if visible_label is not None:
            label, path = visible_label
            return self._propagate_label(name, label, path)

        # Buffers: return sliced/inspected view
        if name in model._buffers:
            return self._inspect(name)

        # Submodules: return a wrapped Slice
        if name in model._modules:
            sub = model._modules[name]
            return type(self)(
                sub,
                object.__getattribute__(self, "index_spec"),
                base_shape=self.base_shape,
                root_model=object.__getattribute__(self, "root_model"),
                module_path=object.__getattribute__(self, "module_path") + (name,),
            )

        # Parameters (optional): often handy to read through
        if name in model._parameters:
            return self._inspect(name)

        # Derived tensor properties (for example ``Population.area``) retain
        # Slice semantics even though they are not registered buffers.
        value = getattr(model, name)
        if torch.is_tensor(value):
            return self._read_tensor(model, name, value)

        kind = "method" if callable(value) else "attribute"
        raise AttributeError(
            f"{name!r} is a model-wide {kind}, not Slice-scoped state. "
            f"Access slice.model.{name} explicitly if whole-model behavior is "
            "intended."
        )

    # -------------------------
    # Misc
    # -------------------------
    def __repr__(self):
        """Return a developer-friendly representation summarising the selection."""
        spec = self._sync()
        return (
            f"Slice(index={spec.index}, shape={spec.shape}, is_scalar={spec.is_scalar})"
        )

    def _batch(self):
        """
        Promote the slice to include a leading batch dimension.

        This internal helper synchronizes the index with every newly added
        leading batch axis, recomputes the result shape, and updates the shared
        :class:`IndexSpec` in place.
        """
        self._sync()


class Sliceable:
    """Mixin enabling convenient slicing of population-like objects.

    Classes that mix in :class:`Sliceable` gain NumPy-style indexing semantics
    that return :class:`Slice` instances instead of raw tensors. The primary
    intended user is :class:`dendra.models.core.Population`, but any object
    that:

    * Exposes a ``shape`` attribute describing its logical layout, and
    * Exposes the state and methods expected by :class:`Slice`

    can be made sliceable by inheriting from this mixin.

    Examples
    --------
    .. code-block:: python

        class Population(Sliceable, nn.Module):
            def __init__(self, shape, ...):
                super().__init__()
                self.shape = shape
                ...

        pop = Population((10, 3), ...)
        soma = pop[:, 0]          # returns a Slice
        soma.set("v", torch.zeros_like(soma.get("v")))
    """

    def __init__(self):
        # Mapping from string labels to Slice objects (populated by Slice.label)
        self._labels = {}

    def slots(self, synapse, slot_index=None, *, local_index=None):
        """Return explicit local slots inside ``synapse``.

        With no ``slot_index`` this returns every local slot in the mechanism.
        Passing ``local_index`` or ``slot_index`` selects those local slots.
        """
        if local_index is not None:
            if slot_index is not None:
                raise ValueError("Provide either slot_index or local_index, not both.")
            slot_index = local_index
        return SynapseSlots.from_region(
            self,
            synapse,
            slot_index=slot_index,
        )

    def __getitem__(self, key):
        """
        Return a :class:`Slice` corresponding to ``key``.

        Parameters
        ----------
        key : Any
            Index compatible with PyTorch/NumPy semantics.

        Returns
        -------
        Slice
            A new :class:`Slice` that wraps ``self`` and stores the canonical
            index and resulting shape.

        Examples
        --------
        .. code-block:: python

            pop = Population(...)
            soma = pop[:, 0]          # Slice selecting soma compartment
            dend = pop[:, 1:]         # Slice selecting dendrites

            # Label and reuse
            soma.label("soma")
            pop.soma.inject(step_current)
        """
        index = parse_key(key, self.shape, device=self.device())
        return Slice(self, index)


def expand_into_shape(src, index, shape, fill_value=torch.nan):
    """Scatter a source tensor into a larger target shape.

    Parameters
    ----------
    src : torch.Tensor
        Source tensor to insert into ``out``.
    index : tuple
        Index tuple compatible with ``shape``.
    shape : Sequence[int]
        Target tensor shape.
    fill_value : float, optional
        Value used to initialise the output tensor. Defaults to ``torch.nan``.

    Returns
    -------
    torch.Tensor
        Tensor with ``src`` written at ``index``.

    Examples
    --------
    >>> src = torch.tensor([1., 2.])
    >>> out = expand_into_shape(src, (torch.tensor([0, 2]),), (4,))
    >>> out
    tensor([1., nan, 2., nan])
    """
    out = torch.full(shape, fill_value, dtype=src.dtype, device=src.device)
    out[index] = src
    return out


def compose_indices(shape, idx1, idx2, *, device="cpu"):
    """Compose two successive indexing operations.

    This helper computes an index ``idx3`` such that, for any tensor ``t`` of
    the given ``shape``,

    .. code-block:: python

        torch.allclose(t[idx1][idx2], t[idx3])

    holds. It is used to implement slicing of slices without materialising
    intermediate *state* tensors; it does allocate an integer coordinate map
    with ``prod(shape)`` entries.

    Parameters
    ----------
    shape : Sequence[int]
        Shape of the original tensor.
    idx1 : tuple
        First index applied to the tensor.
    idx2 : tuple
        Second index applied to the intermediate result ``tensor[idx1]``.
    device : str or torch.device, optional
        Device for intermediate tensors.

    Returns
    -------
    tuple of torch.Tensor
        Tuple ``idx3`` satisfying ``tensor[idx1][idx2] == tensor[idx3]``.

    Examples
    --------
    .. code-block:: python

        shape = (4, 5)
        base = torch.arange(20).reshape(shape)

        idx1 = (slice(None), slice(2, 5))
        idx2 = (slice(None), slice(1, 3))

        idx3 = compose_indices(shape, idx1, idx2)
        assert torch.allclose(base[idx1][idx2], base[idx3])
    """
    # 1) Build a flat index map shaped like `shape`
    numel = math.prod(shape)
    base = torch.arange(numel, device=device).reshape(shape)

    # 2) Apply the two-stage indexing to the map
    flat = base[idx1][idx2]  # same shape as t[idx1][idx2]

    # 3) Convert selected flat positions back to per-dimension indices
    idx3 = torch.unravel_index(flat, shape)  # tuple of tensors

    return idx3  # use as t[idx3]


def _own_index_item(item, *, device=None):
    """Copy mutable array indices so a Slice cannot change behind its metadata."""
    if torch.is_tensor(item):
        return item.detach().clone().to(device=device)
    if isinstance(item, np.ndarray):
        return torch.as_tensor(item.copy(), device=device)
    if isinstance(item, list):
        if not item:
            return torch.empty(0, device=device, dtype=torch.long)
        return torch.as_tensor(item, device=device).clone()
    return item


def parse_key(key: Any, shape: Sequence[int], device=None) -> IndexSpec:
    """Normalise an indexing key for a tensor with ``shape``.

    This function applies the given ``key`` to a dummy tensor with the
    specified ``shape`` in order to infer the resulting shape and scalar-ness
    using PyTorch's own indexing semantics. The resulting information is
    wrapped in an :class:`IndexSpec`.

    Parameters
    ----------
    key : Any
        Index compatible with PyTorch/NumPy semantics (ints, slices, ellipses,
        boolean masks, integer arrays, tuples thereof, ...).
    shape : Sequence[int]
        Shape of the array to index.
    device : torch.device or None, optional
        Device used for temporary tensor allocation.

    Returns
    -------
    IndexSpec
        Structured description of the indexing request.

    Examples
    --------
    >>> spec = parse_key((slice(None), 0), (10, 3))
    >>> spec.index
    (slice(None, None, None), 0)
    >>> spec.shape
    torch.Size([10])
    >>> spec.is_scalar
    False
    """
    shape = tuple(int(dim) for dim in shape)
    raw_index = key if isinstance(key, tuple) else (key,)
    index = tuple(_own_index_item(item, device=device) for item in raw_index)
    out = torch.empty(shape, device=device, dtype=torch.bool)[index]

    return IndexSpec(
        index=index,
        is_scalar=out.ndim == 0,
        shape=out.shape,
        source_shape=shape,
    )


def concat_slices(slices: Sequence[Slice], dim: int = -1) -> Slice:
    """Concatenate multiple :class:`Slice` objects into a single slice.

    Parameters
    ----------
    slices : Sequence[Slice]
        Slices to concatenate. All slices must wrap the same underlying model
        and be compatible for concatenation along ``dim``.
    dim : int, optional
        Dimension along which to concatenate. If None, concatenates
        flattened slices. Defaults to ``-1``.

    Returns
    -------
    Slice
        New :class:`Slice` representing the concatenation of the inputs.

    Raises
    ------
    ValueError
        If the input slices wrap different models or are incompatible for
        concatenation.
    """

    if not slices:
        raise ValueError("At least one slice must be provided for concatenation.")

    base_model = slices[0].model
    base_shape = tuple(base_model.shape)

    indices = []
    shapes = []
    for slc in slices:
        if slc.model is not base_model:
            raise ValueError("All slices must wrap the same underlying model.")
        indices.append(slc.index)
        shapes.append(slc.shape)

    final_shape = _assess_shape_compatibility(shapes, dim)

    flat_index_map = torch.arange(
        math.prod(base_shape), device=base_model.device(), dtype=torch.long
    ).reshape(base_shape)
    selected_indices = [flat_index_map[idx] for idx in indices]
    if dim is None:
        selected_indices = [selected.reshape(-1) for selected in selected_indices]
        concatenated_idx = torch.cat(selected_indices)
    else:
        normalized_dim = dim if dim >= 0 else dim + len(shapes[0])
        concatenated_idx = torch.cat(selected_indices, dim=normalized_dim)
    new_index = torch.unravel_index(concatenated_idx, base_shape)

    return Slice(
        base_model,
        IndexSpec(
            index=new_index,
            shape=torch.Size(final_shape),
            is_scalar=False,
            source_shape=base_shape,
        ),
        base_shape=base_shape,
        root_model=object.__getattribute__(slices[0], "root_model"),
        module_path=object.__getattribute__(slices[0], "module_path"),
    )


def _assess_shape_compatibility(shapes: Sequence[Sequence[int]], dim: Optional[int]):
    """Check that shapes are compatible for concatenation along ``dim``.

    Parameters
    ----------
    shapes : Sequence[Sequence[int]]
        Shapes to assess.
    dim : int or None
        Dimension along which concatenation is intended. If None, each slice
        is flattened independently and the resulting vectors are concatenated.

    Raises
    ------
    ValueError
        If the shapes are incompatible for concatenation.
    """

    # first convert negative dim to positive
    if dim is not None:
        ndim = len(shapes[0])
        if dim < 0:
            dim += ndim
        if dim < 0 or dim >= ndim:
            raise ValueError(
                f"Concatenation dimension {dim} is out of range for {ndim}D slices."
            )

    if dim is not None:
        # All shapes must match in all dimensions except ``dim``
        ref_shape = list(shapes[0])
        for shape in shapes[1:]:
            if len(shape) != len(ref_shape):
                raise ValueError(
                    "All slices must have the same number of dimensions for "
                    "concatenation."
                )
            for d in range(len(shape)):
                if d != dim and shape[d] != ref_shape[d]:
                    raise ValueError(
                        f"Shapes differ at dimension {d}, cannot concatenate."
                    )

    # If we reach here, shapes are compatible
    # Compute final shape after concatenation
    final_shape = list(shapes[0])
    if dim is None:
        final_shape = (sum(int(np.prod(shape)) for shape in shapes),)
    else:
        final_shape[dim] = sum(shape[dim] for shape in shapes)
    return tuple(final_shape)


def _merge_indices(indices: Sequence[torch.Tensor]) -> torch.Tensor:
    """Merge multiple index tensors into a single concatenated index.
    If can be a slice, return a slice.

    Parameters
    ----------
    indices : Sequence[torch.Tensor]
        Index tensors to merge. Each tensor should be 1D.

    Returns
    -------
    torch.Tensor or slice
        Concatenated index tensor.
    """
    concatenated = torch.cat(indices)

    # A singleton/empty selection has no meaningful stride. Repeated indices
    # likewise cannot be represented by a valid Python slice (step zero).
    if concatenated.numel() < 2:
        return concatenated

    # Check if the concatenated indices form a contiguous range
    # with a uniform step (may not be 1)

    diffs = concatenated[1:] - concatenated[:-1]
    if diffs[0] > 0 and torch.all(diffs == diffs[0]):
        start = concatenated[0].item()
        stop = concatenated[-1].item() + diffs[0].item()
        step = diffs[0].item()
        return slice(start, stop, step)

    return concatenated
