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

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch

# Anything NumPy or PyTorch accepts in __getitem__
IndexElement = Union[int, slice, np.ndarray, list, tuple]


@dataclass(slots=True)
class IndexSpec:
    """Canonical description of a single indexing request.

    Attributes
    ----------
    index : tuple of IndexElement
        Expanded index without ellipsis or ``None`` axes.
    is_scalar : bool
        ``True`` when the result is scalar valued.
    shape : tuple of int
        Shape produced by applying ``index`` to a tensor.
    """

    index: Tuple[IndexElement, ...]
    is_scalar: bool
    shape: Tuple[int, ...]

    def to_key(self, model) -> torch.LongTensor:
        """Given a model with shape `shape_p`, return a flat key tensor."""
        # Build a flat index map once and apply the slice directly.
        base = torch.arange(
            math.prod(model.shape_p), device=model.device(), dtype=torch.long
        ).view(model.shape_p)
        return base[self.index].reshape(-1)


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

    * Stores a canonicalised :class:`IndexSpec` describing the selection.
    * Provides methods for reading and writing model and mechanism state
      restricted to that selection.
    * Supports targeted intracellular current injections via :meth:`inject`.
    * Restricts mechanism insertion to that subset via :meth:`insert`.
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
    into the original population without ever materialising intermediate
    tensors. This keeps nested slicing both expressive and efficient.

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
    returns a sliced view of buffers, parameters, and submodules:

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

    Both pathways perform writes under ``torch.no_grad()`` and call
    ``detach_()`` afterwards to preserve the tensor identity but drop autograd
    history, which is typically what you want inside a simulation loop.

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
        pop.dendrites.set("g_pas", torch.full(pop.dendrites.shape, 1e-4))

    Labels are also stored in ``population._labels`` (a simple ``dict``),
    allowing programmatic access via ``population._labels["soma"]``.

    Attributes
    ----------
    model : Population
        Underlying population (or submodule) providing the state and methods.
    index_spec : IndexSpec
        Canonicalised description of the selection (index, shape, scalar flag).
    base_shape : tuple of int
        Shape of the original population before any slicing was applied.
    parent_slice : Slice or None
        Parent slice if this slice was created from another slice.

    Notes
    -----
    :class:`Slice` is a pure view; creating or discarding slices does not copy
    simulation state. The main cost is that of the underlying tensor indexing
    when :meth:`inspect`, :meth:`set`, attribute access, or other operations
    are performed.
    """

    _RESERVED = ("model", "index_spec", "base_shape", "parent_slice")

    def __init__(
        self, model, index_spec: IndexSpec, base_shape=None, parent_slice=None
    ):
        # Bypass interception for internal fields
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "index_spec", index_spec)
        object.__setattr__(self, "parent_slice", parent_slice)
        if base_shape is None:
            base_shape = model.shape
        object.__setattr__(self, "base_shape", base_shape)

    # -------------------------
    # Simple, safe properties
    # -------------------------
    @property
    def index(self) -> Tuple[IndexElement, ...]:
        """Canonical index (tuple) describing this slice."""
        return object.__getattribute__(self, "index_spec").index

    @property
    def shape(self):
        """Shape produced by applying :attr:`index` to the underlying population."""
        return object.__getattribute__(self, "index_spec").shape

    @property
    def is_scalar(self) -> bool:
        """Whether the selection is scalar-valued (no remaining dimensions)."""
        return object.__getattribute__(self, "index_spec").is_scalar

    def numel(self) -> int:
        """Return the number of selected elements."""
        return int(np.prod(object.__getattribute__(self, "index_spec").shape))

    @property
    def name(self) -> str:
        """Name of the underlying population (delegated from ``model.name``)."""
        return object.__getattribute__(self, "model").name

    @property
    def is_empty(self) -> bool:
        """Return ``True`` if this slice selects no elements."""
        return self.numel() == 0

    # -------------------------
    # Public API
    # -------------------------
    def inspect(self, var: str, mechanism: Optional[str] = None) -> Any:
        """
        Return a read-only view of ``var`` constrained to the slice.

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
            Sliced value of the requested attribute. For tensors, the leading
            dimensions match :attr:`shape` of the slice. The exact return type
            depends on how the underlying model stores ``var``.

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
        model = object.__getattribute__(self, "model")
        idx = object.__getattribute__(self, "index_spec").index

        if mechanism is not None:
            mech = model.mech.mechanisms[mechanism]
            if mech.key is None:
                return getattr(mech, var)[idx]
            dummy = torch.tensor(torch.nan, device=model.device(), dtype=model.dtype())
            dummy = mech.put(getattr(mech, var), dummy, model.v)
            return dummy[idx]

        return getattr(model, var)[idx]

    def _inspect(self, var: str):
        """Inspect ``var`` on the wrapped model without mechanism handling.

        This is an internal helper used to implement attribute access for
        buffers, parameters and submodules. It respects sparse/keyed storage
        when the underlying model exposes a ``key`` attribute.
        """
        model = object.__getattribute__(self, "model")
        base_shape = object.__getattribute__(self, "base_shape")
        idx = object.__getattribute__(self, "index_spec").index

        if getattr(model, "key", None) is not None:
            v = getattr(model, var)
            dummy = torch.tensor(torch.nan, device=v.device, dtype=v.dtype)
            dummy = model.put(
                v, dummy, torch.empty(base_shape, device=v.device, dtype=v.dtype)
            )
            return dummy[idx]
        return getattr(model, var)[idx]

    def get(self, var: str, mechanism: Optional[str] = None) -> torch.Tensor:
        """
        Convenience alias for :meth:`inspect` returning a tensor.

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
            Tensor view of the requested variable restricted to the slice.

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
        compartments. The write is performed in-place under ``torch.no_grad()``
        and followed by ``detach_()`` on the underlying tensor to drop autograd
        history while preserving identity (important when the tensor is a
        registered buffer).

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
        model = object.__getattribute__(self, "model")
        idx = object.__getattribute__(self, "index_spec").index

        if mechanism is not None:
            mech = model.mech.mechanisms[mechanism]
            if mech.key is None:
                with torch.no_grad():
                    getattr(mech, var)[idx] = value
                    getattr(mech, var).detach_()
                return

            dummy = torch.tensor(torch.nan, device=model.device(), dtype=model.dtype())
            dummy = mech.put(getattr(mech, var), dummy, model.v)
            with torch.no_grad():
                dummy[idx] = value
                getattr(mech, var).copy_(mech.get(dummy))
                getattr(mech, var).detach_()
            return

        with torch.no_grad():
            getattr(model, var)[idx] = value
            getattr(model, var).detach_()  # keep identity, drop history

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
        model = object.__getattribute__(self, "model")
        index_spec = object.__getattribute__(self, "index_spec")
        if hasattr(model, "register_injection"):
            model.register_injection(waveform, index_spec)
        else:
            model.injections.append((waveform, index_spec.shape, index_spec.index))

    def insert(self, mechanism, alias=None, ic=None, **kwargs):
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
            Optional initial-conditions object or configuration passed through
            to the underlying ``insert`` call. (Exact semantics depend on the
            population implementation.)
        **kwargs
            Additional keyword arguments forwarded to ``model.insert``.

        Notes
        -----
        Calling :meth:`insert` on an empty slice is a no-op.

        Examples
        --------
        .. code-block:: python

            soma = pop[:, 0]
            soma.insert(NaTs2t, alias="Na_soma")

            # Insert a passive mechanism only in dendrites
            pop[:, 1:].insert(pas, alias="pas_dend", g=1e-4)
        """
        if self.is_empty:
            return  # no-op for empty slices
        object.__getattribute__(self, "model").insert(
            mechanism,
            alias=alias,
            index_spec=object.__getattribute__(self, "index_spec"),
            **kwargs,
        )

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
            pop[:, 1:].mech.pas.parametrize("g", 1e-4, alias="dend")
        """
        if self.is_empty:
            return  # no-op for empty slices
        model = object.__getattribute__(self, "model")
        index_spec = object.__getattribute__(self, "index_spec")
        model.parametrize(
            name,
            value,
            key=index_spec.to_key(model),
            alias=alias,
        )

    def label(self, name: str):
        """
        Attach a label to the slice for convenient, reusable access.

        This method exposes the slice under the given ``name`` as an attribute
        on the owning population (for top-level slices) or on the parent slice
        (for nested slices). For populations, it also adds an entry to the
        ``_labels`` dictionary.

        Parameters
        ----------
        name : str
            Attribute name to use as the label.

        Examples
        --------
        Label soma and dendrites on a population:

        .. code-block:: python

            pop[:, 0].label("soma")
            pop[:, 1:].label("dendrites")

            # Later, reuse the labels
            pop.soma.inject(step_current)
            pop.dendrites.set("g_pas", torch.full(pop.dendrites.shape, 1e-4))

        Labels on nested slices attach to the outer slice:

        .. code-block:: python

            distal = pop[:, 5:]
            distal[:, :2].label("distal_proximal")   # stored on ``distal``

            distal.distal_proximal.set("gNa", 0.0)
        """
        parent_slice = object.__getattribute__(self, "parent_slice")
        if parent_slice is not None:
            # Attach label to the *wrapper* safely (avoid buffer interception)
            object.__setattr__(parent_slice, name, self)
            return
        model = object.__getattribute__(self, "model")
        setattr(model, name, self)
        model._labels[name] = self

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
        model = object.__getattribute__(self, "model")
        idx = compose_indices(
            model.shape,
            object.__getattribute__(self, "index"),
            key,
            device=model.device(),
        )
        idx = parse_key(idx, model.shape, device=model.device())
        return type(self)(
            model,
            idx,
            parent_slice=self,
            base_shape=object.__getattribute__(self, "base_shape"),
        )

    # -------------------------
    # Interceptors
    # -------------------------
    def __setattr__(self, name, value):
        """
        Intercept assignments and route writes into model buffers when possible.

        Behaviour is as follows:

        * Writing an attribute whose name matches a registered buffer on the
          underlying PyTorch module writes only to the slice portion of that
          buffer (and then detaches it).
        * Assignments to reserved/internal attributes (``model``,
          ``index_spec``, ``base_shape``, ``parent_slice``) bypass interception.
        * All other assignments set attributes directly on the :class:`Slice`
          instance itself.

        This allows natural syntax such as ``pop[:, 0].v = value`` for buffer
        updates, while still permitting arbitrary user-defined attributes on a
        slice object.
        """
        # Always allow internal fields
        if name in Slice._RESERVED:
            object.__setattr__(self, name, value)
            return

        # Safely fetch model without triggering our __getattr__
        model = object.__getattribute__(self, "model")

        # Intercept writes to model buffers
        buffers = model._buffers  # nn.Module guarantee
        if name in buffers:
            buf = buffers[name]
            idx = object.__getattribute__(self, "index_spec").index
            with torch.no_grad():
                buf[idx] = value
                buf.detach_()  # drop history but keep identity
            return

        # Otherwise set on this wrapper
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the underlying model when appropriate.

        Resolution order:

        1. If ``name`` matches a buffer on the underlying PyTorch module,
           return a sliced view via :meth:`_inspect`.
        2. If ``name`` matches a submodule, return a new :class:`Slice` that
           wraps the submodule but shares this slice's :class:`IndexSpec`.
        3. If ``name`` matches a parameter, return a sliced view of that
           parameter via :meth:`_inspect`.
        4. Otherwise, delegate attribute access directly to the wrapped model.

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
            model = object.__getattribute__(self, "model")
        except AttributeError:
            raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

        # Buffers: return sliced/inspected view
        if name in model._buffers:
            return self._inspect(name)

        # Submodules: return a wrapped Slice
        if name in model._modules:
            sub = model._modules[name]
            return type(self)(
                sub,
                object.__getattribute__(self, "index_spec"),
                base_shape=object.__getattribute__(self, "base_shape"),
            )

        if name in model._labels:
            model = object.__getattribute__(self, "model")
            idx = compose_indices(
                model.shape,
                model._labels[name].index,
                object.__getattribute__(self, "index"),
                device=model.device(),
            )
            idx = parse_key(idx, model.shape, device=model.device())
            return type(self)(
                model,
                idx,
                base_shape=object.__getattribute__(self, "base_shape"),
            )

        # Parameters (optional): often handy to read through
        if name in model._parameters:
            return self._inspect(name)

        # Fallback: delegate to the wrapped model (methods, attrs, etc.)
        return getattr(model, name)

    # -------------------------
    # Misc
    # -------------------------
    def __repr__(self):
        """Return a developer-friendly representation summarising the selection."""
        spec = object.__getattribute__(self, "index_spec")
        return (
            f"Slice(index={spec.index}, shape={spec.shape}, is_scalar={spec.is_scalar})"
        )

    def _batch(self):
        """
        Promote the slice to include a leading batch dimension.

        This internal helper prepends a leading ``slice(None)`` to the current
        index (unless the first element is already an ellipsis), recomputes the
        resulting shape, and updates the :class:`IndexSpec` in-place. It is
        typically used when switching from unbatched to batched simulation
        layouts.
        """
        model = object.__getattribute__(self, "model")
        index_spec = object.__getattribute__(self, "index_spec")

        current_index = index_spec.index
        new_index = (
            current_index
            if (current_index and current_index[0] is Ellipsis)
            else (slice(None),) + current_index
        )
        index_spec.index = new_index

        test = torch.empty(model.shape, device=model.device(), dtype=model.dtype())[
            new_index
        ]
        index_spec.is_scalar = test.ndim == 0
        index_spec.shape = test.shape


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
        index = parse_key(key, self.shape)
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
    intermediate tensors.

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
    (10,)
    >>> spec.is_scalar
    False
    """
    out = torch.empty(shape, device=device)[key]  # type: ignore

    return IndexSpec(
        index=key,
        is_scalar=out.ndim == 0,
        shape=out.shape,
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
    base_shape = list(slices[0].base_shape)

    indices = []
    shapes = []
    for slc in slices:
        if slc.model is not base_model:
            raise ValueError("All slices must wrap the same underlying model.")
        indices.append(slc.index)
        shapes.append(slc.shape)

    final_shape = _assess_shape_compatibility(shapes, dim)

    if dim is None:
        # Flatten all slices before concatenation
        flat_indices = []
        for idx, shape in zip(indices, shapes):
            flat_idx = torch.arange(np.prod(shape), device=base_model.device()).reshape(
                shape
            )[idx]
            flat_indices.append(flat_idx.flatten())
        concatenated_idx = torch.cat(flat_indices)
        new_index = (concatenated_idx,)
    else:
        # Concatenate along the specified dimension
        dim_indices = []
        for idx in indices:
            dim_indices.append(
                torch.arange(base_shape[dim], device=base_model.device())[idx[dim]]
            )
        concatenated_idx = _merge_indices(dim_indices)
        new_index = list(indices[0])
        new_index[dim] = concatenated_idx
        new_index = tuple(new_index)

    return Slice(
        base_model,
        IndexSpec(index=new_index, shape=final_shape, is_scalar=False),
        base_shape=tuple(base_shape),
    )


def _assess_shape_compatibility(shapes: Sequence[Sequence[int]], dim: Optional[int]):
    """Check that shapes are compatible for concatenation along ``dim``.

    Parameters
    ----------
    shapes : Sequence[Sequence[int]]
        Shapes to assess.
    dim : int or None
        Dimension along which concatenation is intended. If None, all shapes
        must be identical when flattened.

    Raises
    ------
    ValueError
        If the shapes are incompatible for concatenation.
    """

    # first convert negative dim to positive
    if dim is not None and dim < 0:
        dim += len(shapes[0])

    if dim is None:
        # All shapes must have the same number of elements when flattened
        numel_set = {int(np.prod(shape)) for shape in shapes}
        if len(numel_set) > 1:
            raise ValueError(
                "All slices must have the same number of elements when flattened for concatenation."
            )
    else:
        # All shapes must match in all dimensions except ``dim``
        ref_shape = list(shapes[0])
        for shape in shapes[1:]:
            if len(shape) != len(ref_shape):
                raise ValueError(
                    "All slices must have the same number of dimensions for concatenation."
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

    # Check if the concatenated indices form a contiguous range
    # with a uniform step (may not be 1)

    diffs = concatenated[1:] - concatenated[:-1]
    if torch.all(diffs == diffs[0]):
        start = concatenated[0].item()
        stop = concatenated[-1].item() + diffs[0].item()
        step = diffs[0].item()
        return slice(start, stop, step)

    return concatenated
