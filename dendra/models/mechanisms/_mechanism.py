import inspect
import textwrap
from types import MethodType
from typing import Dict

import torch

from dendra.helpers import classproperty
from dendra.models.parametric import Parameterized

from ._ions import VALENCES
from ._state import State
from ._symbolic import build_current_eq


class Mechanism(Parameterized):
    """
    Base class for Dendra mechanisms.

    Mechanisms encapsulate state variables, ionic currents, and parameter
    declarations that can be attached to neuronal morphologies. They extend
    :class:`dendra.models.parametric.Parameterized` to leverage the shared
    parameter declaration and population infrastructure.

    Use uppercase classmethods (``STATE``, ``GLOBAL``, ``RANGE``, ``ASSIGNED``,
    ``USEION``, ``NONSPECIFIC_CURRENT``, etc.) at class definition time to
    declare structure. Override lowercase hooks (``initial``, ``breakpoint``,
    current methods) to implement behavior.

    - ``STATE(StateSubclass, ...)``: register one or more State bundles. Each
      State subclass manages its own state variables and derivatives. These are
      accessible via the ``mechanism.DE`` ModuleDict.
    - ``GLOBAL/RANGE``: shared vs per-compartment parameters.
    - ``ASSIGNED``: mechanism-level buffers (analogous to State ``BUFFER``),
      typically set in ``initial``/``breakpoint``.
    - ``USEION``: ionic read/write dependencies.
    - ``NONSPECIFIC_CURRENT`` / current methods: contribute to membrane balance.
    - ``initial(self, v)``: one-time setup; set ASSIGNED buffers, etc.
    - ``breakpoint(self, v)``: per-step computation of currents/ASSIGNED values.

    Notes
    -----
    Subclasses declare state, assigned, and ionic variables using the
    :meth:`STATE`, :meth:`ASSIGNED`, :meth:`SAVE`, :meth:`USEION`, and
    :meth:`NONSPECIFIC_CURRENT` helpers during class definition. Override
    :meth:`initial` and :meth:`breakpoint` to populate buffers and assemble
    currents each step.
    """

    _state = set()
    _ion = set()
    _save = set()
    _assigned = set()
    _explicit = set()
    _numerical = set()

    _state_declarations = []
    _ion_declarations = []
    _save_declarations = []
    _assigned_declarations = []
    _explicit_declarations = []
    _numerical_declarations = []

    _conductances = {}
    _currents = {}
    _init = {}

    _read_ion = {}
    _write_ion = {}
    _write_ion_c = {}

    _conductances_declarations = []
    _currents_declarations = []
    _init_declarations = []

    _read_ion_declarations = []
    _write_ion_declarations = []
    _write_ion_c_declarations = []

    _renamed_aliases = {}

    def __init_subclass__(cls, **kwargs):
        """
        This special method is called automatically whenever a class
        inherits from Parameterized.
        """
        # Call the parent's __init_subclass__ WITHOUT our custom kwargs,
        # as the base 'object' class does not accept them.
        super().__init_subclass__(**kwargs)

        # Start with a fresh dictionary for the new class's parameters.
        new_state = set()
        new_ion = set()
        new_save = set()
        new_assigned = set()
        new_explicit = set()
        new_numerical = set()

        new_read_ion = {}
        new_write_ion = {}
        new_write_ion_c = {}

        new_currents = {}
        new_init = {}

        # Walk MRO in reverse to build up params from parent to child
        for base in reversed(cls.__mro__):
            # We look for a _params attribute defined directly on the base
            if "_state" in base.__dict__:
                new_state.update(base._state)
            if "_ion" in base.__dict__:
                new_ion.update(base._ion)
            if "_save" in base.__dict__:
                new_save.update(base._save)
            if "_assigned" in base.__dict__:
                new_assigned.update(base._assigned)
            if "_read_ion" in base.__dict__:
                new_read_ion.update(base._read_ion)
            if "_write_ion" in base.__dict__:
                new_write_ion.update(base._write_ion)
            if "_write_ion_c" in base.__dict__:
                new_write_ion_c.update(base._write_ion_c)
            if "_currents" in base.__dict__:
                new_currents.update(base._currents)
            if "_init" in base.__dict__:
                new_init.update(base._init)
            if "_explicit" in base.__dict__:
                new_explicit.update(base._explicit)
            if "_numerical" in base.__dict__:
                new_numerical.update(base._numerical)

        if Mechanism._state_declarations:
            for s_list in Mechanism._state_declarations:
                new_state.update(s_list)
            Mechanism._state_declarations = []  # Clear for next class
        if Mechanism._ion_declarations:
            for i_list in Mechanism._ion_declarations:
                new_ion.update(i_list)
            Mechanism._ion_declarations = []
        if Mechanism._save_declarations:
            for s_list in Mechanism._save_declarations:
                new_save.update(s_list)
            Mechanism._save_declarations = []
        if Mechanism._assigned_declarations:
            for a_list in Mechanism._assigned_declarations:
                new_assigned.update(a_list)
            Mechanism._assigned_declarations = []
        if Mechanism._read_ion_declarations:
            for r_dict in Mechanism._read_ion_declarations:
                new_read_ion.update(r_dict)
            Mechanism._read_ion_declarations = []
        if Mechanism._write_ion_declarations:
            for w_dict in Mechanism._write_ion_declarations:
                new_write_ion.update(w_dict)
            Mechanism._write_ion_declarations = []
        if Mechanism._write_ion_c_declarations:
            for w_dict in Mechanism._write_ion_c_declarations:
                new_write_ion_c.update(w_dict)
            Mechanism._write_ion_c_declarations = []
        if Mechanism._currents_declarations:
            for c_list in Mechanism._currents_declarations:
                new_currents.setdefault("nonspecific", []).extend(c_list)
            Mechanism._currents_declarations = []
        if Mechanism._init_declarations:
            for i_dict in Mechanism._init_declarations:
                new_init.update(i_dict)
            Mechanism._init_declarations = []
        if Mechanism._explicit_declarations:
            for v_list in Mechanism._explicit_declarations:
                new_explicit.update(v_list)
            Mechanism._explicit_declarations = []
        if Mechanism._numerical_declarations:
            for v_list in Mechanism._numerical_declarations:
                new_numerical.update(v_list)
            Mechanism._numerical_declarations = []

        cls.state_classes = {s.__name__: s for s in new_state}

        cls._state = new_state
        cls._ion = new_ion
        cls._save = new_save
        cls._currents = new_currents
        cls._assigned = new_assigned
        cls._read_ion = new_read_ion
        cls._write_ion = new_write_ion
        cls._write_ion_c = new_write_ion_c
        cls._init = new_init
        cls._explicit = new_explicit
        cls._numerical = new_numerical
        cls._name = None

    def __init__(
        self,
        name: str,
        celsius,
        diameters,
        shape,
        shape_f,
        key=None,
        is_composable=False,
        additional_parameters=None,
        ic: dict = None,
        **kwargs,
    ):
        """
        Initialize a mechanism instance and register declared buffers.

        Parameters
        ----------
        name : str
            Mechanism alias. If ``None``, falls back to the class name.
        celsius : Tensor or float
            Temperature values broadcastable to the mechanism shape.
        diameters : Tensor
            Compartment diameters shared with sub-components.
        shape : tuple of int
            Base shape for parameter tensors (without batch dimensions).
        shape_f : tuple of int
            Full shape including batch dimensions.
        key : Any, optional
            Index selector identifying the attached compartments.
        is_composable : bool, optional
            Whether ``key`` represents a tuple of slices instead of flat indices.
        additional_parameters : dict, optional
            Additional parameter declarations injected by the parent population.
        ic : dict, optional
            Initial condition overrides for state buffers.
        **kwargs
            Extra keyword arguments forwarded to
            :class:`dendra.models.parametric.Parameterized`.
        """
        super().__init__(
            shape, shape_f, additional_parameters=additional_parameters, **kwargs
        )
        if name is None:
            name = self.__class__._name or self.__class__.__name__

        self.name = name

        self.register_buffer("celsius", celsius)
        self.base_ndim = 2

        self.register_buffer("dt", torch.tensor(0.0))

        # Mechanism-level waveform injection support.  The base "inject"
        # method is intentionally a no-op; mechanisms that want to consume
        # waveform stimuli can override it and call register_waveform_injection().
        # Waveforms are stored as submodules so their parameters move with the
        # mechanism and remain differentiable.
        self.injected_waveforms = torch.nn.ModuleList()
        self._injection_specs = []

        # Mechanism-level delayed-state support.  This is intended for fused
        # mechanisms that bypass Network/NetCon but still need fixed axonal or
        # state delays.  Each entry in _delayed_state_specs maps a user-facing
        # delay name to registered buffer/pointer names and update policy.
        self._delayed_state_specs = {}

        if key is not None:
            if is_composable:
                self.key = key
            else:
                self.register_buffer("key", torch.tensor(key, dtype=torch.long))
        else:
            self.key = None

        self.is_composable = is_composable

        def get_fancy(tensor):
            if tensor.ndim == 0:
                # If tensor is scalar, return it directly
                return tensor
            # Preserves batch dimensions by only flattening the base dimensions
            batch_shape = tensor.shape[: -self.base_ndim]
            flat_tensor = tensor.reshape(*batch_shape, -1)
            # Select along the last dimension (the flattened base dimension)
            return flat_tensor.index_select(-1, self.key)

        def add_fancy_(tensor, what):
            # Use scatter_add_ for batched index_add_
            batch_shape = tensor.shape[: -self.base_ndim]
            flat_tensor = tensor.reshape(*batch_shape, -1)

            # Expand key to match batch dimensions for scatter
            # e.g., key shape [N] -> [B1, B2, ..., N]
            expanded_key = self.key.expand(*batch_shape, -1)

            what = what.expand_as(expanded_key)

            # what should have shape [B1, B2, ..., N]
            flat_tensor.scatter_add_(-1, expanded_key, what)
            return tensor  # Return original tensor for chaining

        def add_fancy(tensor, what):
            # Use scatter_add for batched index_add
            batch_shape = tensor.shape[: -self.base_ndim]
            flat_tensor = tensor.reshape(*batch_shape, -1)

            what = what.expand_as(self.key)

            # Expand key to match batch dimensions for scatter
            expanded_key = self.key.expand(*batch_shape, -1)

            # what should have shape [B1, B2, ..., N]
            return flat_tensor.scatter_add(-1, expanded_key, what).reshape_as(tensor)

        if self.key is None:
            self.get = lambda tensor: tensor
            self.add_ = lambda add_to, add_what: add_to.add_(add_what)
            self.add = lambda add_to, add_what: add_to.add(add_what)
            self.put = self.put_no_op
        elif self.is_composable:
            self.get = (
                lambda tensor: tensor[..., *self.key] if tensor.ndim > 0 else tensor
            )
            self.add_ = lambda add_to, add_what: add_to[..., *self.key].add_(add_what)
            self.add = lambda add_to, add_what: add_to[..., *self.key].add(add_what)
            self.put = self.put_slice
        else:
            self.get = get_fancy
            self.add_ = add_fancy_
            self.add = add_fancy
            self.put = self.put_fancy

        self.read_ion = self._read_ion
        self.write_ion_c = self._write_ion_c

        states = [
            state(
                self.get(celsius),
                self.get(diameters),
                key,
                shape,
                shape_f,
                additional_parameters=additional_parameters,
                **kwargs,
            )
            for state in self._state
        ]

        self.DE: dict[str, State] = torch.nn.ModuleDict(
            {state._name: state for state in states}
        )

        self._init_params: Dict[str, float] = {k: v for k, v in self._init.items()}
        if ic is not None:
            self._init_params.update(ic)

        self.register_buffer("diam", self.get(diameters))
        for state in self.DE.values():
            for state_name in state._state:
                self.register_buffer(state_name, torch.zeros(shape))
                # getattr(self, state_name).requires_grad_(True)

        for r in self._save:
            self.register_buffer(f"{r}_", torch.zeros(shape))

        for a in self._assigned:
            self.register_buffer(a, torch.zeros(shape))

        # factorize current equations
        current_eqs = []
        for _, v in self._currents.items():
            current_eqs.extend(v)
        for _, v in self._write_ion.items():
            current_eqs.extend(v)

        for k in current_eqs:
            assign = k in self._save
            eq, factorable = build_current_eq(self, k, assign=assign)
            setattr(self, f"{k}_with_g", MethodType(eq, self))
            setattr(getattr(self.__class__, k), "factorable", factorable)
            setattr(self, "factorable", factorable)

        self.populate()
        self.instantiate_tables()
        for state in self.DE.values():
            state.instantiate_tables()

    def set_dt(self, dt):
        """
        Update the per-mechanism time-step buffer.

        Parameters
        ----------
        dt : float or Tensor
            New integration step size in milliseconds.
        """
        self.dt = self.dt.fill_(dt).detach()

    def put_no_op(self, ion_conc_u, ion_conc_o, v, clone=True):
        """
        Return ionic concentrations unchanged.

        Parameters
        ----------
        ion_conc_u : Tensor
            Updated ionic concentrations (ignored).
        ion_conc_o : Tensor
            Original ionic concentrations.
        v : Tensor
            Voltage reference tensor (ignored).
        clone : bool, optional
            Unused for the no-op path.

        Returns
        -------
        Tensor
            ``ion_conc_o`` unchanged.
        """
        return ion_conc_u

    def put_slice_(self, ion_conc_u, ion_conc_o, v, clone=True):
        """
        Write ionic concentrations into a slice in place.

        Parameters
        ----------
        ion_conc_u : Tensor
            Updated ionic concentrations matching the slice length.
        ion_conc_o : Tensor
            Original ionic concentrations to be updated.
        v : Tensor
            Voltage tensor used for broadcasting shape.
        clone : bool, optional
            If True, operate on a cloned copy of ``ion_conc_o``.

        Returns
        -------
        Tensor
            Tensor with the slice replaced by ``ion_conc_u``.
        """
        ion_conc_o = ion_conc_o.expand_as(v)
        if clone:
            ion_conc_o = ion_conc_o.clone()
        ion_conc_o[..., *self.key] = ion_conc_u
        return ion_conc_o

    def put_slice(self, ion_conc_u, ion_conc_o, v, clone=True):
        """
        Write ionic concentrations into a slice and return a new tensor.

        Parameters
        ----------
        ion_conc_u : Tensor
            Updated ionic concentrations matching the slice length.
        ion_conc_o : Tensor
            Original ionic concentrations to be updated.
        v : Tensor
            Voltage tensor used for broadcasting shape.
        clone : bool, optional
            If True, operate on a cloned copy of ``ion_conc_o``.

        Returns
        -------
        Tensor
            Tensor with the slice replaced by ``ion_conc_u``.
        """
        # ion_conc_o is the full tensor, v is a reference for shape, ion_conc_u is the update
        ion_conc_o = ion_conc_o.expand_as(v)
        if clone:
            ion_conc_o = ion_conc_o.clone()

        # Apply the update using Ellipsis
        ion_conc_o[..., *self.key] = ion_conc_u
        return ion_conc_o

    def put_fancy_(self, ion_conc_u, ion_conc_o, v, clone=True):
        """
        Write ionic concentrations using flattened fancy indexing in place.

        Parameters
        ----------
        ion_conc_u : Tensor
            Updated ionic concentrations with length ``len(self.key)``.
        ion_conc_o : Tensor
            Original ionic concentrations to be updated.
        v : Tensor
            Voltage tensor used for broadcasting shape.
        clone : bool, optional
            If True, operate on a cloned copy of ``ion_conc_o``.

        Returns
        -------
        Tensor
            Tensor with entries replaced at ``self.key`` indices.
        """
        ion_conc_o = ion_conc_o.expand_as(v)
        if clone:
            ion_conc_o = ion_conc_o.clone()
        ion_conc_o.view(-1).index_put_((self.key,), ion_conc_u)
        return ion_conc_o

    def put_fancy(self, ion_conc_u, ion_conc_o, v, clone=True):
        """
        Write ionic concentrations using flattened fancy indexing.

        Parameters
        ----------
        ion_conc_u : Tensor
            Updated ionic concentrations with length ``len(self.key)``.
        ion_conc_o : Tensor
            Original ionic concentrations to be updated.
        v : Tensor
            Voltage tensor used for broadcasting shape.
        clone : bool, optional
            If True, operate on a cloned copy of ``ion_conc_o`` before writing.

        Returns
        -------
        Tensor
            Tensor with entries replaced at ``self.key`` indices.
        """
        # ion_conc_u: The new values to put, shape [..., len(key)]
        # ion_conc_o: The destination tensor, shape [..., *base_shape]
        # v: Reference tensor for shape

        ion_conc_o = ion_conc_o.expand_as(v)
        if clone:
            ion_conc_o = ion_conc_o.clone()

        # Get batch shape from the destination tensor
        batch_shape = ion_conc_o.shape[: -self.base_ndim]

        # Reshape destination to [B, S]
        flat_dest = ion_conc_o.view(*batch_shape, -1)

        # Expand key to match batch dimensions for scatter
        expanded_key = self.key.expand(*batch_shape, -1)

        # Use scatter to place the values from ion_conc_u into flat_dest
        # scatter_(dim, index, src)
        flat_dest.scatter_(-1, expanded_key, ion_conc_u)

        # The original ion_conc_o tensor is modified in place, so we can just return it
        return ion_conc_o

    def register_ion(self, ion):
        """
        Attach shared ion buffers used by the mechanism.

        Parameters
        ----------
        ion : Ion
            Ion descriptor exposing concentration and reversal potential tensors.
        """
        name = ion.name
        if name in self.read_ion:
            for v in self.read_ion[name]:
                q = getattr(ion, v)
                if self.key is not None and q.ndim > 0:
                    self.register_buffer(v, self.get(q))
                    for _, s in self.DE.items():
                        s.register_buffer(v, self.get(q))
                else:
                    self.register_buffer(v, q)
                    for _, s in self.DE.items():
                        s.register_buffer(v, q)

        if name in self.write_ion_c:
            for v in self.write_ion_c[name]:
                q = getattr(ion, v)
                if self.key is not None and q.ndim > 0:
                    qk = self.get(q)
                    self.register_buffer(
                        v, torch.empty(qk.shape, device=qk.device, dtype=qk.dtype)
                    )
                    getattr(self, v).copy_(qk)
                else:
                    self.register_buffer(v, q)

    def _init_buffers_s(self, v_init):
        for state_module in self.DE.values():
            state_names = state_module._state
            for state_name in state_names:
                if state_name in self._init_params:
                    buffer_tensor = (
                        torch.tensor(
                            self._init_params[state_name],
                            device=v_init.device,
                            dtype=v_init.dtype,
                        )
                        .expand_as(v_init)
                        .clone()
                    )
                    setattr(self, state_name, buffer_tensor.detach())
                else:
                    if inf := state_module.inf(v_init):
                        buffer_tensor = inf[state_name]
                        setattr(self, state_name, buffer_tensor.detach())

        self.initial(v_init)

        for _, s in self.DE.items():
            s.initialize(v_init)

        return

    # Classmethod declarations
    @staticmethod
    def STATE(*args):
        """
        Declare state variables for the mechanism class body.

        Parameters
        ----------
        *args : type
            State module classes registered to ``Mechanism._state``.
        """
        Mechanism._state_declarations.append(args)

    @staticmethod
    def ASSIGNED(*args):
        """
        Declare mechanism-level assigned buffers (analogous to State BUFFER).

        Parameters
        ----------
        *args : str
            Names of assigned buffers to allocate per instance. Populate these
            in :meth:`initial` or :meth:`breakpoint`.
        """
        Mechanism._assigned_declarations.append(args)

    @staticmethod
    def SAVE(*args):
        """
        Declare state variables that must be saved each time step.

        Parameters
        ----------
        *args : str
            Names of buffers mirrored with a trailing underscore.
        """
        Mechanism._save_declarations.append(args)

    @staticmethod
    def USEION(ion, read=None, write=None):
        """
        Declare ionic read/write dependencies for the mechanism class body.

        Parameters
        ----------
        ion : str
            Ion species identifier (e.g., ``'na'``).
        read : Sequence[str], optional
            Ion variables to be read; must be among ``{ion}i``, ``{ion}o``,
            ``e{ion}``, or ``i{ion}`` (e.g., ``['nai', 'nao', 'ena']``).
        write : Sequence[str], optional
            Ion variables to be written; same allowed set (e.g., ``['ina']`` for
            current contribution, or ``['nai']`` to update concentration).

        Raises
        ------
        AssertionError
            If the ion name is unknown or a read/write symbol is invalid.
        ValueError
            If reversal potentials are written or a variable is both read and written.
        """
        read = read or []
        write = write or []

        if not read and not write:
            return

        assert ion in VALENCES, (
            f"Unknown ion {ion}. Valid ions are {list(VALENCES.keys())}."
        )

        if f"e{ion}" in write:
            raise ValueError(f"e{ion} cannot be written")

        if common := set(read).intersection(write):
            raise ValueError(f"{common} is/are both read and written")

        valid = {f"{ion}i", f"{ion}o", f"e{ion}", f"i{ion}"}

        for r in read or []:
            assert r in valid, f"read {r} is not valid"
        for w in write or []:
            assert w in valid, f"write {w} is not valid"

        if read:
            Mechanism._read_ion_declarations.append({ion: read})

        if write:
            c_write = []
            other = []
            for w in write:
                if w in {f"{ion}i", f"{ion}o"}:
                    c_write.append(w)
                else:
                    other.append(w)

            if c_write:
                Mechanism._write_ion_c_declarations.append({ion: c_write})
            if other:
                Mechanism._write_ion_declarations.append({ion: other})

    @staticmethod
    def NONSPECIFIC_CURRENT(*args):
        """
        Declare non-specific (leak) currents produced by the mechanism.

        Parameters
        ----------
        *args : str
            Current names to register as non-specific.
        """
        Mechanism._currents_declarations.append(args)

    @staticmethod
    def INIT(**kwargs):
        """
        Declare initial buffer values for state variables.

        Parameters
        ----------
        **kwargs
            Mapping from state names to scalar initial conditions.
        """
        Mechanism._init_declarations.append(kwargs)

    @staticmethod
    def EXPLICIT(*args):
        """
        Mark currents as voltage independent when assembling the RHS.

        Parameters
        ----------
        *args : str
            Current names that should not contribute to conductance terms.
        """
        Mechanism._explicit_declarations.append(args)

    @staticmethod
    def NUMERICAL(*args):
        """
        Mark currents as requiring numerical differentiation.

        Parameters
        ----------
        *args : str
            Current names that should be numerically differentiated.
        """
        Mechanism._numerical_declarations.append(args)

    def breakpoint(self, v):
        """
        Evaluate mechanism currents at the breakpoint stage.

        Parameters
        ----------
        v : Tensor
            Membrane potential values for the local compartments.

        Notes
        -----
        Override to compute mechanism-level :meth:`ASSIGNED` buffers and
        assemble currents (e.g., ``ina``, ``ik``, ``il``). Called each step
        before current accumulation.
        """
        return

    def detach(self):
        """
        Detach mechanism buffers and nested state modules from autograd.
        """
        super().detach()
        for state_module in self.DE.values():
            state_module.detach()

    def _advance(self, v, dt):
        """
        Advance nested state modules by one time step.

        Parameters
        ----------
        v : Tensor
            Membrane potentials for the local compartments.
        dt : Tensor
            Time-step tensor propagated from the integrator.
        """
        for state_module in self.DE.values():
            states = {
                state_name: self._buffers[state_name]
                for state_name in state_module._state
            }
            local = state_module.advance(v, dt, states)
            self._buffers.update(local)

    def populate(self):
        """
        Populate mechanism and nested state parameter buffers.
        """
        self.populate_parameter_buffers()
        for state_module in self.DE.values():
            state_module.populate_parameter_buffers()

    def initial(self, v):
        """
        Hook for subclasses to initialize buffers from membrane potential.

        Parameters
        ----------
        v : Tensor
            Membrane potential values used for initialization.

        Notes
        -----
        Use this to populate mechanism-level :meth:`ASSIGNED` buffers (e.g.,
        cached conductances) or perform any one-time setup before stepping.
        """
        return

    @classmethod
    def rename(cls, new_name=None, suffix=None):
        """
        Create a renamed clone of the current mechanism class.

        Parameters
        ----------
        new_name : str, optional
            Explicit name for the cloned class. Required when ``suffix`` is ``None``.
        suffix : str, optional
            Suffix appended to the original class name when ``new_name`` is omitted.

        Returns
        -------
        type
            Mechanism subclass identical to ``cls`` but with a new ``__name__``.

        Raises
        ------
        ValueError
            If both ``new_name`` and ``suffix`` are ``None``.
        TypeError
            If ``suffix`` is provided but not a string.
        """
        if new_name is None and suffix is None:
            raise ValueError("Either new_name or suffix must be provided.")

        if new_name is None:
            new_name = cls.__name__

        if suffix is not None:
            if not isinstance(suffix, str):
                raise TypeError("Suffix must be a string.")
            new_name += f"_{suffix}"

        if new_name in cls._renamed_aliases:
            return cls._renamed_aliases[new_name]

        cls._renamed_aliases[new_name] = rename(cls, new_name=new_name)
        return cls._renamed_aliases[new_name]

    def batch(self, batch_size: int):
        """
        Broadcast mechanism buffers across an explicit batch dimension.

        Parameters
        ----------
        batch_size : int
            Size of the leading batch dimension to materialize.

        Returns
        -------
        Mechanism
            The mechanism instance with batched buffers.
        """
        super().batch(batch_size)
        for state_module in self.DE.values():
            state_module.batch(batch_size)
        return self

    def states(self):
        """
        Collect fully qualified state names.

        Returns
        -------
        list of str
            Names in the form ``\"{mechanism}.{state}\"``.
        """
        states = []
        for state_module in self.DE.values():
            states.extend(state_module._state)
        return [f"{self.name}.{state}" for state in states]

    @classmethod
    def state_names(cls):
        """
        Collect state variable names declared by the mechanism class.

        Returns
        -------
        list of str
            Names of state variables registered in ``cls._state``.
        """
        states = []
        for state_module in cls._state:
            states.extend(state_module._state)
        return states

    @classproperty
    def code(cls):
        """
        Source code of the mechanism class.

        Returns
        -------
        str
            Dedented string containing the class definition.
        """
        source_code = inspect.getsource(cls)
        return textwrap.dedent(source_code)

    @classproperty
    def file_code(cls) -> str:
        """
        Source text of the Python module defining the class.

        Returns
        -------
        str
            Entire module contents containing ``cls``.

        Raises
        ------
        RuntimeError
            If the module where ``cls`` is defined cannot be located.
        """
        mod = inspect.getmodule(cls)
        if mod is None:
            raise RuntimeError(f"Cannot locate module for {cls.__qualname__}")
        return inspect.getsource(mod)  # whole file

    def states_dict(self):
        """
        Map state names to their underlying buffers.

        Returns
        -------
        dict
            Dictionary from raw state names to tensors.
        """
        dct = {}
        for state_module in self.DE.values():
            for state_name in state_module._state:
                dct[state_name] = self._buffers[state_name]
        return dct

    # -- rng --
    def init_rng(self):
        for state_module in self.DE.values():
            state_module.init_rng()
        super().init_rng()

    def reset_rng(self):
        for state_module in self.DE.values():
            state_module.reset_rng()
        super().reset_rng()

    # -- tables --
    def usetables(self, value: bool):
        """
        Enable or disable table usage for the mechanism and nested states.

        Parameters
        ----------
        value : bool
            Whether to use tables for function approximations.
        """
        for state_module in self.DE.values():
            state_module.usetables(value)
        super().usetables(value)

    @classmethod
    def all_mech_parameter_names(cls):
        """
        Collect all parameter names declared by the mechanism class.

        Returns
        -------
        list of str
            Names of parameters registered in ``cls._parameters``.
        """
        param_names = {}
        param_names["mechanism"] = cls.all_parameter_names()
        for state_module in cls._state:
            param_names[state_module.__name__] = state_module.all_parameter_names()
        return param_names

    @classmethod
    def check_kwargs(cls, kwargs):
        """
        Check for unexpected keyword arguments.

        Parameters
        ----------
        **kwargs
            Keyword arguments to validate.

        Raises
        ------
        ValueError
            If any unexpected keyword arguments are found.
        """
        valid_keys = set()
        for state_module in cls._state:
            valid_keys.update(state_module.all_parameter_names())
        valid_keys.update(cls.all_parameter_names())

        for key in kwargs.keys():
            if key not in valid_keys:
                raise ValueError(f"Unexpected keyword argument: {key}")

    # -- mechanism-level delayed states ---------------------------------------
    def register_delayed_state(
        self,
        name,
        like,
        delay_steps: int,
        *,
        mode="auto",
        buffer_name=None,
        pointer_name=None,
        insert_axis: int = -1,
        clear: bool = True,
    ):
        """Register a fixed-step delayed state buffer on this mechanism.

        This helper is for fused mechanisms that need NetCon-like fixed delays
        while bypassing :class:`~dendra.models.networks.NetCon`.  The delayed
        state has two update backends:

        ``mode="shift"``
            Graph-safe functional update. Each call constructs a new queue with
            ``torch.cat``. This is appropriate for training / BPTT because the
            delayed value remains connected to the computation graph.

        ``mode="circular"``
            Fast in-place circular buffer. This avoids shifting/copying the
            whole queue each step, but it intentionally writes under
            ``torch.no_grad()`` and is therefore intended for evaluation only.

        ``mode="auto"``
            Use ``circular`` when the mechanism is in eval mode and gradients are
            disabled; otherwise use ``shift``.

        Parameters
        ----------
        name : str
            User-facing delay name used with :meth:`delayed_state`.
        like : torch.Tensor
            Tensor whose shape/device/dtype define the payload shape.
        delay_steps : int
            Integer delay in simulation steps. A value emitted at call ``k``
            appears after ``delay_steps`` subsequent calls, matching the queue
            convention used by Dendra's event delay buffers. The current delayed-state
            helper assumes one uniform delay per registered delayed state. For heterogeneous
            per-connection delays, use ``NetCon`` / ``ContinuousCon`` or register
            separate delayed states for each distinct delay.
        mode : {"auto", "shift", "circular"}, default "auto"
            Backend selection policy.
        buffer_name : str, optional
            Name of the registered buffer. Defaults to
            ``f"{name}_delay_buffer"``. Existing ``ASSIGNED`` buffers can be
            reused by passing their name here.
        pointer_name : str, optional
            Name of the circular-buffer write pointer.
        insert_axis : int, default -1
            Axis before which the delay dimension is inserted. The default
            inserts the delay axis before the final payload dimension, so a
            payload of shape ``(..., n)`` becomes ``(..., depth, n)``.
        clear : bool, default True
            If true, reset the delay buffer and pointer.

        Returns
        -------
        torch.Tensor
            The registered delay buffer.
        """
        if mode is None:
            mode = "auto"
        mode = str(mode).lower()
        aliases = {
            "functional": "shift",
            "queue": "shift",
            "shift_queue": "shift",
            "eval_circular": "auto",
            "fast_eval": "auto",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"auto", "shift", "circular"}:
            raise ValueError(
                "delayed-state mode must be 'auto', 'shift', or 'circular'; "
                f"got {mode!r}."
            )

        delay_steps = int(delay_steps)
        depth = max(1, delay_steps + 1)
        like = torch.as_tensor(like)

        if insert_axis < 0:
            insert_axis = like.ndim + 1 + insert_axis
        if insert_axis < 0 or insert_axis > like.ndim:
            raise ValueError(
                f"insert_axis={insert_axis} is invalid for payload ndim={like.ndim}."
            )

        buffer_name = buffer_name or f"{name}_delay_buffer"
        pointer_name = pointer_name or f"{name}_delay_ptr"

        buffer_shape = (
            tuple(like.shape[:insert_axis]) + (depth,) + tuple(like.shape[insert_axis:])
        )
        buffer = torch.zeros(buffer_shape, device=like.device, dtype=like.dtype)
        pointer = torch.zeros((), device=like.device, dtype=torch.long)

        if buffer_name in self._buffers:
            if clear or tuple(getattr(self, buffer_name).shape) != buffer_shape:
                setattr(self, buffer_name, buffer)
        else:
            self.register_buffer(buffer_name, buffer)

        if pointer_name in self._buffers:
            if clear:
                setattr(self, pointer_name, pointer)
        else:
            self.register_buffer(pointer_name, pointer)

        self._delayed_state_specs[name] = {
            "buffer": buffer_name,
            "pointer": pointer_name,
            "steps": delay_steps,
            "depth": depth,
            "axis": int(insert_axis),
            "mode": mode,
        }
        return getattr(self, buffer_name)

    def register_delayed_states(
        self,
        name,
        like,
        delay_steps,
        *,
        mode="auto",
        buffer_name=None,
        pointer_name=None,
        steps_name=None,
        stream_axis: int = -2,
        clear: bool = True,
    ):
        """Register a batched fixed-step delay line for several streams.

        This is the multi-stream counterpart of :meth:`register_delayed_state`.
        It is intended for fused mechanisms that need many pathway-level delays
        over tensors with the same payload shape.  For example, a value tensor
        with shape ``(batch, n_streams, n)`` and ``stream_axis=-2`` is stored in
        one delay buffer with shape ``(batch, depth, n_streams, n)``.

        Parameters
        ----------
        name : str
            User-facing delay group name used with :meth:`delayed_states`.
        like : torch.Tensor
            Example value tensor. One dimension is interpreted as the stream /
            pathway axis; all other dimensions are payload dimensions.
        delay_steps : int or sequence[int] or torch.Tensor
            Integer delays in simulation steps. A scalar applies the same delay
            to every stream. A vector must have length ``like.shape[stream_axis]``.
        mode : {"auto", "shift", "circular"}, default "auto"
            Backend selection policy. ``auto`` uses circular buffers in eval /
            no-grad mode and graph-safe shifted queues during training/grad mode.
        buffer_name : str, optional
            Name of the registered delay buffer. Defaults to
            ``f"{name}_delay_buffer"``.
        pointer_name : str, optional
            Name of the circular-buffer write pointer. Defaults to
            ``f"{name}_delay_ptr"``.
        steps_name : str, optional
            Name of the registered integer delay vector. Defaults to
            ``f"{name}_delay_steps"``.
        stream_axis : int, default -2
            Axis of ``like`` containing independent delay streams. The delay
            dimension is inserted immediately before this axis.
        clear : bool, default True
            If true, reset the delay buffer and circular pointer.

        Returns
        -------
        torch.Tensor
            The registered delay buffer.

        Notes
        -----
        ``delayed_states`` supports heterogeneous per-stream integer delays,
        but all streams in a group share one circular pointer and one buffer
        depth equal to ``max(delay_steps) + 1``. This is useful for fused
        models with several fixed pathway delays and avoids one delayed-state
        helper call per pathway.
        """
        if mode is None:
            mode = "auto"
        mode = str(mode).lower()
        aliases = {
            "functional": "shift",
            "queue": "shift",
            "shift_queue": "shift",
            "eval_circular": "auto",
            "fast_eval": "auto",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"auto", "shift", "circular"}:
            raise ValueError(
                "batched delayed-state mode must be 'auto', 'shift', or 'circular'; "
                f"got {mode!r}."
            )

        like = torch.as_tensor(like)
        if like.ndim == 0:
            raise ValueError("register_delayed_states requires a non-scalar payload.")

        if stream_axis < 0:
            stream_axis = like.ndim + stream_axis
        if stream_axis < 0 or stream_axis >= like.ndim:
            raise ValueError(
                f"stream_axis={stream_axis} is invalid for payload ndim={like.ndim}."
            )
        n_streams = int(like.shape[stream_axis])
        if n_streams <= 0:
            raise ValueError("delayed-state stream axis must be non-empty.")

        steps = torch.as_tensor(delay_steps, device=like.device, dtype=torch.long)
        if steps.ndim == 0 or steps.numel() == 1:
            steps = steps.reshape(1).expand(n_streams).clone()
        else:
            steps = steps.reshape(-1).clone()
            if steps.numel() != n_streams:
                raise ValueError(
                    f"delay_steps has length {steps.numel()}, but the stream axis "
                    f"has length {n_streams}."
                )
        if torch.any(steps < 0):
            raise ValueError("delay_steps must be non-negative integers.")

        max_steps = int(steps.max().item()) if steps.numel() else 0
        depth = max(1, max_steps + 1)

        # Insert delay axis immediately before the stream axis.  If like has
        # shape (..., streams, n), the buffer has shape (..., depth, streams, n).
        delay_axis = int(stream_axis)
        buffer_stream_axis = delay_axis + 1
        buffer_shape = (
            tuple(like.shape[:delay_axis]) + (depth,) + tuple(like.shape[delay_axis:])
        )

        buffer_name = buffer_name or f"{name}_delay_buffer"
        pointer_name = pointer_name or f"{name}_delay_ptr"
        steps_name = steps_name or f"{name}_delay_steps"

        buffer = torch.zeros(buffer_shape, device=like.device, dtype=like.dtype)
        pointer = torch.zeros((), device=like.device, dtype=torch.long)

        if buffer_name in self._buffers:
            if clear or tuple(getattr(self, buffer_name).shape) != buffer_shape:
                setattr(self, buffer_name, buffer)
        else:
            self.register_buffer(buffer_name, buffer)

        if pointer_name in self._buffers:
            if clear:
                setattr(self, pointer_name, pointer)
        else:
            self.register_buffer(pointer_name, pointer)

        if steps_name in self._buffers:
            if clear or tuple(getattr(self, steps_name).shape) != tuple(steps.shape):
                setattr(self, steps_name, steps)
        else:
            self.register_buffer(steps_name, steps)

        # Static delay metadata used to avoid scalar tensor reductions in the
        # compiled per-step path.  In particular, do not call Tensor.item() from
        # delayed_states(...): TorchDynamo treats that as a graph break unless
        # capture_scalar_outputs is enabled.  These booleans are valid for the
        # registered delay vector; if callers override delay_steps dynamically,
        # delayed_states falls back to the generic update path without relying on
        # these flags.
        has_zero_delay = bool(torch.any(steps == 0).item())
        all_zero_delay = bool(torch.all(steps <= 0).item())

        self._delayed_state_specs[name] = {
            "buffer": buffer_name,
            "pointer": pointer_name,
            "steps_buffer": steps_name,
            "steps": max_steps,
            "depth": depth,
            "axis": delay_axis,
            "stream_axis": buffer_stream_axis,
            "value_stream_axis": int(stream_axis),
            "n_streams": n_streams,
            "mode": mode,
            "batched": True,
            "has_zero_delay": has_zero_delay,
            "all_zero_delay": all_zero_delay,
        }
        return getattr(self, buffer_name)

    def reset_delayed_states(self, *names):
        """Zero registered delayed-state buffers and reset circular pointers."""
        if not names:
            names = tuple(self._delayed_state_specs.keys())
        with torch.no_grad():
            for name in names:
                spec = self._delayed_state_specs[name]
                getattr(self, spec["buffer"]).zero_()
                getattr(self, spec["pointer"]).zero_()
        return self

    def delayed_state(self, name, value, *, delay_steps=None, mode=None):
        """Return a delayed copy of ``value`` and update the named delay line.

        The delay line must first be created with
        :meth:`register_delayed_state`. ``mode="auto"`` uses the in-place
        circular buffer only in eval/no-grad mode; otherwise it uses a graph-safe
        functional shift update.
        """
        if name not in self._delayed_state_specs:
            raise KeyError(
                f"No delayed state named {name!r} has been registered. "
                "Call register_delayed_state(...) during initial(...)."
            )
        spec = self._delayed_state_specs[name]
        steps = int(spec["steps"] if delay_steps is None else delay_steps)
        if steps <= 0:
            return value

        selected_mode = str(spec["mode"] if mode is None else mode).lower()
        aliases = {"functional": "shift", "queue": "shift", "shift_queue": "shift"}
        selected_mode = aliases.get(selected_mode, selected_mode)
        if selected_mode == "auto":
            selected_mode = (
                "shift" if (self.training or torch.is_grad_enabled()) else "circular"
            )

        if selected_mode == "circular":
            return self._delayed_state_circular(spec, value, steps)
        if selected_mode == "shift":
            return self._delayed_state_shift(spec, value, steps)
        raise ValueError(
            "delayed-state mode must be 'auto', 'shift', or 'circular'; "
            f"got {selected_mode!r}."
        )

    def _delayed_state_shift(self, spec, value, steps: int):
        """Graph-safe delayed-state update using an out-of-place shifted queue."""
        buf_name = spec["buffer"]
        axis = int(spec["axis"])
        buf = getattr(self, buf_name)
        depth = int(buf.shape[axis])
        if steps + 1 != depth:
            # Re-register if a caller overrides delay_steps with a different
            # length. This is uncommon but keeps the helper predictable.
            state_name = next(
                k for k, v in self._delayed_state_specs.items() if v is spec
            )
            self.register_delayed_state(
                state_name,
                value,
                steps,
                mode=spec["mode"],
                buffer_name=buf_name,
                pointer_name=spec["pointer"],
                insert_axis=axis,
                clear=True,
            )
            buf = getattr(self, buf_name)
            depth = int(buf.shape[axis])

        sl = [slice(None)] * buf.ndim
        sl[axis] = slice(0, depth - 1)
        buf_new = torch.cat((value.unsqueeze(axis), buf[tuple(sl)]), dim=axis)
        setattr(self, buf_name, buf_new)
        return buf_new.select(axis, depth - 1)

    def _delayed_state_circular(self, spec, value, steps: int):
        """Fast eval delayed-state update using an in-place circular buffer."""
        buf = getattr(self, spec["buffer"])
        ptr = getattr(self, spec["pointer"])
        axis = int(spec["axis"])
        depth = int(buf.shape[axis])
        if steps + 1 != depth:
            raise ValueError(
                "Circular delayed_state cannot change delay_steps without "
                "re-registering the delayed state."
            )

        read_idx = torch.remainder(ptr - int(steps), depth).reshape(1)
        delayed = buf.index_select(axis, read_idx).squeeze(axis)

        # Evaluation-only fast path: mutate the circular buffer without building
        # autograd history. In training/grad mode, delayed_state(..., mode='auto')
        # selects the graph-safe shift backend instead.
        with torch.no_grad():
            write_idx = ptr.reshape(1)
            buf.index_copy_(axis, write_idx, value.detach().unsqueeze(axis))
            ptr.add_(1).remainder_(depth)
        return delayed

    def delayed_states(self, name, values, *, delay_steps=None, mode=None):
        """Return delayed copies of a multi-stream value tensor.

        The delay group must first be created with
        :meth:`register_delayed_states`.  ``values`` must have the same shape as
        the ``like`` tensor used at registration time.  The stream axis can have
        distinct integer delays supplied during registration.
        """
        if name not in self._delayed_state_specs:
            raise KeyError(
                f"No delayed state group named {name!r} has been registered. "
                "Call register_delayed_states(...) during initial(...)."
            )
        spec = self._delayed_state_specs[name]
        if not spec.get("batched", False):
            raise ValueError(
                f"Delayed state {name!r} was registered with register_delayed_state; "
                "use delayed_state(...) instead."
            )

        # Fast all-zero shortcut for the registered delay vector.  This is a
        # Python bool stored at registration time, so it is safe under
        # torch.compile.  Avoid Tensor.item() here: the per-step path may be
        # captured by TorchDynamo, and scalar extraction causes a graph break.
        if delay_steps is None and bool(spec.get("all_zero_delay", False)):
            return values

        steps = self._delayed_states_steps(spec, values, delay_steps)

        selected_mode = str(spec["mode"] if mode is None else mode).lower()
        aliases = {"functional": "shift", "queue": "shift", "shift_queue": "shift"}
        selected_mode = aliases.get(selected_mode, selected_mode)
        if selected_mode == "auto":
            selected_mode = (
                "shift" if (self.training or torch.is_grad_enabled()) else "circular"
            )

        if selected_mode == "circular":
            return self._delayed_states_circular(spec, values, steps)
        if selected_mode == "shift":
            return self._delayed_states_shift(spec, values, steps)
        raise ValueError(
            "batched delayed-state mode must be 'auto', 'shift', or 'circular'; "
            f"got {selected_mode!r}."
        )

    def _delayed_states_steps(self, spec, values, delay_steps=None):
        if delay_steps is None:
            steps = getattr(self, spec["steps_buffer"])
        else:
            steps = torch.as_tensor(delay_steps, device=values.device, dtype=torch.long)
        steps = steps.to(device=values.device, dtype=torch.long)
        if steps.ndim == 0 or steps.numel() == 1:
            steps = steps.reshape(1).expand(int(spec["n_streams"]))
        else:
            steps = steps.reshape(-1)
        if steps.numel() != int(spec["n_streams"]):
            raise ValueError(
                f"delay_steps has length {steps.numel()}, expected {spec['n_streams']}."
            )
        return steps

    def _delayed_states_gather(self, buf, spec, read_idx):
        axis = int(spec["axis"])
        stream_axis = int(spec["stream_axis"])
        n_streams = int(spec["n_streams"])

        index_shape = [1] * buf.ndim
        index_shape[axis] = 1
        index_shape[stream_axis] = n_streams

        expand_shape = list(buf.shape)
        expand_shape[axis] = 1

        idx = read_idx.reshape(index_shape).expand(expand_shape)
        return torch.gather(buf, dim=axis, index=idx).squeeze(axis)

    def _delayed_states_zero_mask(self, spec, values, steps):
        value_stream_axis = int(spec["value_stream_axis"])
        mask_shape = [1] * values.ndim
        mask_shape[value_stream_axis] = int(spec["n_streams"])
        return (steps == 0).reshape(mask_shape)

    def _delayed_states_shift(self, spec, values, steps):
        """Graph-safe multi-stream delay update using a shifted queue."""
        buf_name = spec["buffer"]
        axis = int(spec["axis"])
        buf = getattr(self, buf_name)
        depth = int(buf.shape[axis])
        required_depth = int(spec.get("steps", 0)) + 1
        if required_depth != depth:
            state_name = next(
                k for k, v in self._delayed_state_specs.items() if v is spec
            )
            self.register_delayed_states(
                state_name,
                values,
                steps,
                mode=spec["mode"],
                buffer_name=buf_name,
                pointer_name=spec["pointer"],
                steps_name=spec["steps_buffer"],
                stream_axis=int(spec["value_stream_axis"]),
                clear=True,
            )
            spec = self._delayed_state_specs[state_name]
            buf = getattr(self, buf_name)
            depth = int(buf.shape[axis])

        sl = [slice(None)] * buf.ndim
        sl[axis] = slice(0, depth - 1)
        buf_new = torch.cat((values.unsqueeze(axis), buf[tuple(sl)]), dim=axis)
        setattr(self, buf_name, buf_new)
        return self._delayed_states_gather(buf_new, spec, steps)

    def _delayed_states_circular(self, spec, values, steps):
        """Fast eval multi-stream delay update using one in-place ring buffer."""
        buf = getattr(self, spec["buffer"])
        ptr = getattr(self, spec["pointer"])
        axis = int(spec["axis"])
        depth = int(buf.shape[axis])
        required_depth = int(spec.get("steps", 0)) + 1
        if required_depth > depth:
            raise ValueError(
                "Circular delayed_states cannot increase max delay_steps without "
                "re-registering the delayed state group."
            )

        read_idx = torch.remainder(ptr - steps.to(device=buf.device), depth)
        delayed = self._delayed_states_gather(buf, spec, read_idx)

        # A stream with zero delay should deliver the current value, not the
        # previous content of the circular slot that will be overwritten below.
        if bool(spec.get("has_zero_delay", False)):
            delayed = torch.where(
                self._delayed_states_zero_mask(spec, values, steps), values, delayed
            )

        with torch.no_grad():
            write_idx = ptr.reshape(1)
            buf.index_copy_(axis, write_idx, values.detach().unsqueeze(axis))
            ptr.add_(1).remainder_(depth)
        return delayed

    # -- mechanism-level waveform injections ---------------------------------
    def inject(
        self,
        waveform,
        *,
        index=None,
        shape=None,
        model_shape=None,
        model=None,
        **kwargs,
    ):
        """Optionally attach a waveform stimulus to this mechanism.

        The default implementation is deliberately a no-op and returns ``False``.
        Mechanisms that own their own voltage/current dynamics can override this
        method and either consume the arguments directly or call
        :meth:`register_waveform_injection` to get padded ``I(t)`` tensors.

        Parameters
        ----------
        waveform : Waveform
            Waveform object supplied through ``model[idx].inject(waveform)``.
        index : tuple, optional
            Population-level index tuple identifying the targeted compartments.
        shape : tuple, optional
            Shape produced by applying ``index`` to the population.
        model_shape : tuple, optional
            Full population voltage shape at registration time.
        model : Population, optional
            Owning population. Used for device/dtype/shape information.
        **kwargs
            Reserved for future extension.

        Returns
        -------
        bool
            ``True`` if the mechanism accepted the injection, otherwise ``False``.
        """
        return False

    def clear_injections(self):
        """Remove all waveform injections registered on this mechanism."""
        self.injected_waveforms = torch.nn.ModuleList()
        self._injection_specs = []
        for name in list(self._buffers.keys()):
            if name.startswith("_injection_mask_") or name.startswith(
                "_injection_scale_"
            ):
                delattr(self, name)
        return self

    def register_waveform_injection(
        self,
        waveform,
        *,
        index=None,
        model_shape=None,
        current_name="i_inj",
        scale=1.0,
        model=None,
    ):
        """Register a waveform and build a local padding mask for this mechanism.

        This helper is intended for mechanism subclasses that override
        :meth:`inject`.  It computes the overlap between a population-level
        injection index and the compartments occupied by this mechanism, stores
        the waveform as a submodule, and records a boolean local mask.  Later,
        :meth:`evaluate_injections` evaluates all registered waveforms at the
        current mechanism time and returns a tensor shaped like the local voltage
        argument, with zeros outside the targeted compartments.
        """
        device = self.diam.device
        dtype = self.diam.dtype
        if model is not None:
            device = model.device()
            dtype = model.dtype()

        if hasattr(waveform, "to"):
            waveform = waveform.to(device=device, dtype=dtype)

        if model_shape is None:
            # No population frame was supplied: treat this as targeting every
            # compartment where the mechanism resides.
            local_mask = torch.ones_like(self.diam, dtype=torch.bool, device=device)
        else:
            full_mask = torch.zeros(tuple(model_shape), dtype=torch.bool, device=device)
            if index is None:
                full_mask.fill_(True)
            else:
                full_mask[index] = True
            local_mask = self.get(full_mask)

        if local_mask.numel() == 0 or not bool(torch.any(local_mask).item()):
            return False

        k = len(self.injected_waveforms)
        mask_name = f"_injection_mask_{k}"
        scale_name = f"_injection_scale_{k}"
        self.register_buffer(mask_name, local_mask.detach().clone())
        self.register_buffer(
            scale_name,
            torch.as_tensor(scale, device=device, dtype=dtype).detach().clone(),
        )
        self.injected_waveforms.append(waveform)
        self._injection_specs.append(
            {"mask": mask_name, "scale": scale_name, "current_name": current_name}
        )

        # Expose the current variable immediately for introspection, even before
        # the first timestep.  Subclasses may also declare it with ASSIGNED.
        if not hasattr(self, current_name):
            self.register_buffer(
                current_name, torch.zeros_like(local_mask, dtype=dtype)
            )
        return True

    def _expand_injection_value(self, value, mask, out):
        """Return ``value`` padded/broadcast into ``out`` at ``mask`` locations."""
        value = torch.as_tensor(value, device=out.device, dtype=out.dtype)

        # Bring an unbatched mask up to the current local state shape.
        mask = mask.to(device=out.device, dtype=torch.bool)
        while mask.ndim < out.ndim:
            mask = mask.unsqueeze(0)
        mask = mask.expand_as(out)

        if value.ndim == 0 or value.numel() == 1:
            return value.reshape(()) * mask.to(out.dtype)

        if tuple(value.shape) == tuple(out.shape):
            return value * mask.to(out.dtype)

        # Common case: waveform returns the unbatched local mechanism shape.
        if value.ndim <= out.ndim:
            v = value
            while v.ndim < out.ndim:
                v = v.unsqueeze(0)
            if tuple(v.shape) == tuple(out.shape) or all(
                a == b or a == 1 for a, b in zip(v.shape, out.shape)
            ):
                return v.expand_as(out) * mask.to(out.dtype)

        # Vector over selected compartments.  This supports either a single
        # unbatched vector of length n_selected or a batched tensor whose last
        # dimension is n_selected.
        n_selected = int(mask.reshape(-1).sum().item()) if out.ndim == mask.ndim else 0
        if value.numel() == n_selected:
            padded = torch.zeros_like(out)
            padded.reshape(-1)[mask.reshape(-1)] = value.reshape(-1)
            return padded

        # Last-resort attempt: rely on PyTorch broadcasting, then mask.
        try:
            return value.expand_as(out) * mask.to(out.dtype)
        except RuntimeError as exc:
            raise ValueError(
                f"Waveform output shape {tuple(value.shape)} cannot be broadcast "
                f"or padded into mechanism-local shape {tuple(out.shape)}."
            ) from exc

    def evaluate_injections(self, v=None, *, t=None, current_name="i_inj"):
        """Evaluate registered waveform injections and expose the result.

        Parameters
        ----------
        v : torch.Tensor, optional
            Local voltage/state tensor that defines the desired output shape.  If
            omitted, the first injection mask shape is used.
        t : torch.Tensor or float, optional
            Evaluation time in ms.  Defaults to the mechanism's ``t`` reference,
            which :class:`Population` sets during build.
        current_name : str, optional
            Name of the exposed current variable.  Defaults to ``i_inj``.

        Returns
        -------
        torch.Tensor
            Sum of all registered waveform currents, padded to ``v``'s shape.
        """
        if v is None:
            if self._injection_specs:
                v = getattr(self, self._injection_specs[0]["mask"]).to(self.diam.dtype)
            elif hasattr(self, current_name):
                v = getattr(self, current_name)
            else:
                v = self.diam

        out = torch.zeros_like(v, dtype=v.dtype, device=v.device)
        if not self._injection_specs:
            setattr(self, current_name, out)
            return out

        if t is None:
            t = (
                self.t
                if hasattr(self, "t")
                else torch.zeros((), device=v.device, dtype=v.dtype)
            )
        t = torch.as_tensor(t, device=v.device, dtype=v.dtype)

        for k, spec in enumerate(self._injection_specs):
            mask = getattr(self, spec["mask"])
            scale = getattr(self, spec["scale"])
            value = self.injected_waveforms[k](t) * scale
            out = out + self._expand_injection_value(value, mask, out)

        setattr(self, current_name, out)
        return out


class VoltageProcess(Mechanism):
    """
    Mechanism subtype that updates the membrane potential ``v``.

    Subclasses must implement :meth:`update_v` to return a new membrane potential
    tensor each time step.
    """

    def update_v(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the updated membrane potential. Avoid in-place modification
        to preserve autograd compatibility.

        Parameters
        ----------
        v : torch.Tensor
            Membrane potential tensor to be advanced.

        Returns
        -------
        torch.Tensor
            New membrane potential values.
        """
        raise NotImplementedError(
            "VoltageProcess.update_v() must be implemented in subclasses."
        )


class PointProcess(Mechanism):
    """
    A PointProcess is a Mechanism that delivers a lumped current (units nA)
    to a single point in space. Channel conductances must be in units uS.

    Implementing a Mechanism as PointProcess simply instructs Dendra to scale
    the currents and conductances by the area of the relevant compartments to translate
    them to densities. As such, they cannot be inserted at branchpoints
    (which have 0 area), and doing so will produce a numerical error.
    """

    pass


class Synapse(Mechanism):
    """
    A Synapse is a Mechanism that receives spikes and delivers a synaptic current.
    Synapses implement the `net_receive` method, which is called every timestep
    and accepts `weights` (the sum of all incoming weighted spike events at that
    timestep) and `netcon` (the `NetCon` instance for which the synapse is the target)
    as arguments.

    Note that a given synapse may be the target for multiple `NetCon`s, and as such its
    `net_receive` method may be executed multiple times per timestep.

    The `net_receive` method must be overridden in subclasses to implement
    specific behavior for how the synapse responds to incoming spikes.
    """

    def net_receive(self, weights, netcon):
        """
        Handle weighted spike arrivals. Avoid in-place modifications of
        state buffers to preserve autograd compatibility.

        Example: increment an internal synaptic conductance ``g_syn`` by the
        incoming weight without in-place ops:

        .. code-block:: python

            def net_receive(self, weights, netcon):
                self.g_syn = self.g_syn + weights

        Parameters
        ----------
        weights : torch.Tensor
            Aggregate weights of incoming spike events at the current step.
        netcon : NetCon
            Connectivity handle delivering the spikes.
        """
        raise NotImplementedError(
            "Synapse.net_receive() must be implemented in subclasses."
        )


class ContinuousSynapse(Mechanism):
    """
    A mechanism that receives continuously valued presynaptic variables.

    ``ContinuousSynapse`` is the analog counterpart to :class:`Synapse`.  It is
    intended for graded transmitter gates, rate-coded projections, neuromodulatory
    drives, and other connections where the presynaptic mechanism emits a
    continuous variable rather than discrete events.

    Subclasses declare continuous input buffers with ``ContinuousSynapse.INPUT``.
    The network's continuous-connection machinery resets those buffers once per
    timestep, then delivers weighted presynaptic values by calling
    :meth:`continuous_receive`.  By default, deliveries are **summed** into the
    named input, which is the natural behavior for convergent synaptic currents.
    The previous timestep's input is also available as ``<input>_old`` when the
    input was declared with the default ``keep_old=True``.

    Example
    -------

    .. code-block:: python

        class graded_gaba(ContinuousSynapse):
            ContinuousSynapse.INPUT("g_pre")
            ContinuousSynapse.RANGE(e=-80.0)
            ContinuousSynapse.NONSPECIFIC_CURRENT("i")

            def i(self, v):
                return self.g_pre * (v - self.e)
    """

    _continuous_inputs = tuple()
    _continuous_input_old = {}
    _continuous_input_declarations = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        inputs = []
        old_map = {}
        for base in reversed(cls.__mro__):
            if "_continuous_inputs" in base.__dict__:
                inputs.extend(list(base._continuous_inputs))
            if "_continuous_input_old" in base.__dict__:
                old_map.update(dict(base._continuous_input_old))

        for names, keep_old in ContinuousSynapse._continuous_input_declarations:
            for name in names:
                if name not in inputs:
                    inputs.append(name)
                if keep_old:
                    old_map[name] = f"{name}_old"

        ContinuousSynapse._continuous_input_declarations = []
        cls._continuous_inputs = tuple(inputs)
        cls._continuous_input_old = old_map

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Used by ContinuousCon for optional introspection/debugging.  The
        # current implementation normally resets through a designated first
        # ContinuousCon per target synapse, but this buffer also makes explicit
        # step-aware reset policies possible later.
        self.register_buffer(
            "_continuous_reset_count", torch.zeros((), dtype=torch.long)
        )

    @staticmethod
    def INPUT(*names, keep_old=True):
        """Declare one or more per-step continuous input buffers.

        Parameters
        ----------
        *names : str
            Input buffer names to declare.
        keep_old : bool, optional
            If True (default), also declare ``<name>_old`` buffers and populate
            them with the previous timestep's input during reset.
        """
        if not names:
            raise ValueError("ContinuousSynapse.INPUT requires at least one name.")
        names = tuple(str(n) for n in names)
        assigned = list(names)
        if keep_old:
            assigned.extend(f"{name}_old" for name in names)
        Mechanism.ASSIGNED(*assigned)
        ContinuousSynapse._continuous_input_declarations.append((names, bool(keep_old)))

    def reset_continuous_inputs(self):
        """Reset continuous input buffers before new analog deliveries.

        For each declared input ``x``, ``x_old`` is first updated to the current
        value when available, and ``x`` is then reset to zeros.  Rebinding rather
        than in-place mutation keeps the operation compatible with autograd.
        """
        for name in self._continuous_inputs:
            current = getattr(self, name)
            old_name = self._continuous_input_old.get(name, None)
            if old_name is not None and hasattr(self, old_name):
                setattr(self, old_name, current)
            setattr(self, name, torch.zeros_like(current))
        self._continuous_reset_count = self._continuous_reset_count + 1

    def continuous_receive(self, value, con=None, input=None, reduce=None):
        """Receive a continuously valued presynaptic projection.

        Parameters
        ----------
        value : torch.Tensor
            Delivered value in the synapse-local shape.
        con : ContinuousCon, optional
            Connectivity object delivering the value.
        input : str, optional
            Name of the input buffer to update.  If omitted, the first declared
            input is used.
        reduce : {"sum", "set", "max", "min"}, optional
            Reduction used when multiple continuous projections target the same
            input.  Defaults to the connection's ``reduce`` attribute if present,
            otherwise ``"sum"``.
        """
        if input is None:
            if len(self._continuous_inputs) != 1:
                raise ValueError(
                    "continuous_receive requires `input=` when the synapse has "
                    f"{len(self._continuous_inputs)} declared inputs."
                )
            input = self._continuous_inputs[0]

        if input not in self._continuous_inputs:
            raise ValueError(
                f"{self.name!r} has no continuous input {input!r}. "
                f"Declared inputs are {self._continuous_inputs!r}."
            )

        if reduce is None:
            reduce = getattr(con, "reduce", "sum")

        current = getattr(self, input)
        value = value.to(device=current.device, dtype=current.dtype)

        if reduce in ("sum", "add"):
            setattr(self, input, current + value)
        elif reduce in ("set", "replace", "last"):
            setattr(self, input, value)
        elif reduce == "max":
            setattr(self, input, torch.maximum(current, value))
        elif reduce == "min":
            setattr(self, input, torch.minimum(current, value))
        else:
            raise ValueError(f"Unsupported continuous reduction mode: {reduce!r}.")


def rename(mechanism, new_name=None):
    """
    Clone a mechanism class under a new name.

    Parameters
    ----------
    mechanism : type
        Mechanism subclass to be cloned.
    new_name : str, optional
        Name assigned to the cloned class. Defaults to the original name.

    Returns
    -------
    type
        Mechanism subclass with identical behavior but a different ``__name__``.
    """
    # The three-argument form of type(): type(name, bases, dict)
    # 1. name: The new class name (a string).
    # 2. bases: A tuple of the original class's base classes.
    # 3. dict: A dictionary containing the attributes and methods of the
    #          original class. We create a copy to avoid side effects.

    # Copy the original class's namespace dictionary.
    class_dict = dict(mechanism.__dict__)

    # The __dict__ of a class doesn't always include '__module__',
    # so we copy it over explicitly to make the new class look authentic.
    if "__module__" not in class_dict:
        class_dict["__module__"] = mechanism.__module__

    new_class = type(new_name, mechanism.__bases__, class_dict)
    new_class._name = new_name  # Set the new name attribute

    return new_class
