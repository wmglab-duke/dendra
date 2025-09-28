import inspect
import textwrap
from types import MethodType
from typing import Dict

import torch

from axonml.helpers import classproperty
from axonml.models.parametric import Parameterized

from ._ions import VALENCES
from ._symbolic import build_current_eq


class Mechanism(Parameterized):
    """
    `Mechanism` is the base class for all mechanisms in AxonML.
    It provides a framework for defining state variables, ion channels,
    and currents, and for managing the parameters of these mechanisms.
    `Mechanism` is a subclass of `Parameterized`, which provides the
    functionality for handling parameters and their declarations.

    Mechanisms are responsible for keeping track of where in the model
    they are inserted, which facilitates AxonML's sparse internal representations
    of states and parameters.
    """

    _state = set()
    _ion = set()
    _save = set()
    _assigned = set()
    _explicit = set()

    _state_declarations = []
    _ion_declarations = []
    _save_declarations = []
    _assigned_declarations = []
    _explicit_declarations = []

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
        Initialize the Mechanism with parameters and declarations.
        """
        super().__init__(
            shape, shape_f, additional_parameters=additional_parameters, **kwargs
        )
        if name is None:
            name = self.__class__._name or self.__class__.__name__

        self.name = name

        self.register_buffer("celsius", celsius)
        self.base_ndim = 2

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

        self.DE = torch.nn.ModuleDict({state._name: state for state in states})

        self._init_params: Dict[str, float] = {k: v for k, v in self._init.items()}
        if ic is not None:
            self._init_params.update(ic)

        self.register_buffer("diam", self.get(diameters))
        for state in self.DE.values():
            for state_name in state._state:
                self.register_buffer(state_name, torch.zeros(shape))
                # getattr(self, state_name).requires_grad_(True)

        for r in self._save:
            self.register_buffer(r, torch.zeros(shape))

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

        self.populate()

    def put_no_op(self, ion_conc_u, ion_conc_o, v, clone=True):
        return ion_conc_u

    def put_slice_(self, ion_conc_u, ion_conc_o, v, clone=True):
        ion_conc_o = ion_conc_o.expand_as(v)
        if clone:
            ion_conc_o = ion_conc_o.clone()
        ion_conc_o[..., *self.key] = ion_conc_u
        return ion_conc_o

    def put_slice(self, ion_conc_u, ion_conc_o, v, clone=True):
        # ion_conc_o is the full tensor, v is a reference for shape, ion_conc_u is the update
        ion_conc_o = ion_conc_o.expand_as(v)
        if clone:
            ion_conc_o = ion_conc_o.clone()

        # Apply the update using Ellipsis
        ion_conc_o[..., *self.key] = ion_conc_u
        return ion_conc_o

    def put_fancy_(self, ion_conc_u, ion_conc_o, v, clone=True):
        ion_conc_o = ion_conc_o.expand_as(v)
        if clone:
            ion_conc_o = ion_conc_o.clone()
        ion_conc_o.view(-1).index_put_((self.key,), ion_conc_u)
        return ion_conc_o

    def put_fancy(self, ion_conc_u, ion_conc_o, v, clone=True):
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
                    setattr(self, state_name, buffer_tensor)
                    buffer_tensor.detach_()
                    # buffer_tensor.requires_grad_(True)
                else:
                    if hasattr(state_module, "inf"):
                        inf = state_module.inf(v_init)
                        buffer_tensor = inf[state_name]
                        setattr(self, state_name, buffer_tensor)
                        buffer_tensor.detach_()
                        # buffer_tensor.requires_grad_(True)

        self.initial(v_init)

        for _, s in self.DE.items():
            s.initialize(v_init)

        return

    @staticmethod
    def STATE(*args):
        Mechanism._state_declarations.append(args)

    @staticmethod
    def ASSIGNED(*args):
        Mechanism._assigned_declarations.append(args)

    @staticmethod
    def SAVE(*args):
        Mechanism._save_declarations.append(args)

    @staticmethod
    def USEION(ion, read=None, write=None):
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
        Mechanism._currents_declarations.append(args)

    @staticmethod
    def INIT(**kwargs):
        Mechanism._init_declarations.append(kwargs)

    @staticmethod
    def EXPLICIT(*args):
        """
        Marks the mechanism as independent of voltage for the given currents.
        This means that the current does not depend on the membrane potential `v`.
        Instead of contributing a conductance to the LHS, it is handled entirely
        on the RHS.
        """
        Mechanism._explicit_declarations.append(args)

    def breakpoint(self, v, states):
        return

    def detach(self):
        super().detach()
        for state_module in self.DE.values():
            state_module.detach()

    def _advance(self, v, dt):
        for state_module in self.DE.values():
            states = {
                state_name: self._buffers[state_name]
                for state_name in state_module._state
            }
            local = state_module.advance(v, dt, states)
            self._buffers.update(local)

    def populate(self):
        self.populate_parameter_buffers()
        for state_module in self.DE.values():
            state_module.populate_parameter_buffers()

    def initial(self, v):
        return

    @classmethod
    def rename(cls, new_name=None, suffix=None):
        """
        Returns a new mechanism class that is an exact copy of `cls` but with a
        different name.

        Args:
            new_name (str): The name of the new class. If None, the original class's
                            name will be used with an optional suffix.
            suffix (str): An optional suffix to append to the new class name.

        Returns:
            type: A new class, identical in behavior to the original but with a new name.
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
        Returns a new instance of the mechanism with the parameters
        distributed over the specified batch size.
        """
        super().batch(batch_size)
        for state_module in self.DE.values():
            state_module.batch(batch_size)
        return self

    def states(self):
        """
        Returns a list of all state names in the Mechanism.
        """
        states = []
        for state_module in self.DE.values():
            states.extend(state_module._state)
        return [f"{self.name}.{state}" for state in states]

    @classproperty
    def code(cls):
        """
        Returns the source code of the mechanism.
        This is useful for debugging and introspection.
        """
        source_code = inspect.getsource(cls)
        return textwrap.dedent(source_code)

    @classproperty
    def file_code(cls) -> str:
        """Full source of the .py file that defines this class."""
        mod = inspect.getmodule(cls)
        if mod is None:
            raise RuntimeError(f"Cannot locate module for {cls.__qualname__}")
        return inspect.getsource(mod)  # whole file

    def states_dict(self):
        """
        Returns a dictionary of all state names in the Mechanism
        mapped to their corresponding tensors.
        """
        return {state: self._buffers[state] for state in self.DE.keys()}


class VoltageProcess(Mechanism):
    """
    A VoltageProcess is a Mechanism that updates the membrane potential `v`.
    VoltageProcesses implement the `update_v` method, which is called during the
    simulation to update the membrane potential based on the mechanism's dynamics.
    It must be overridden in subclasses to implement specific behavior.
    """

    def update_v(self, v, dt):
        """
        This method must be overridden in subclasses to implement specific behavior.
        Any modifications to the membrane potential should not be in-place,
        but rather return a new tensor.

        Parameters
        ----------
            v (torch.Tensor): The membrane potential tensor.
            dt (torch.Tensor): The time step for the update.

        Returns
        -------
            v (torch.Tensor): The updated membrane potential tensor.
        """
        raise NotImplementedError(
            "VoltageProcess.update_v() must be implemented in subclasses."
        )


class PointProcess(Mechanism):
    """
    A PointProcess is a Mechanism that delivers a lumped current (units nA)
    to a single point in space. Channel conductances must be in units uS.

    Implementing a Mechanism as PointProcess simply instructs AxonML to scale
    the currents and conductances by the area of the relevant compartments to translate
    them to densities. As such, unlike in NEURON, they cannot be inserted at branchpoints
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
        This method is called when the synapse receives a spike.
        It should be overridden in subclasses to implement specific behavior.

        Args:
            weight (torch.Tensor): The weight of the synapse.
            netcon (NetCon): The `NetCon` instance that delivered the spikes.

        Returns:
            None
        """
        raise NotImplementedError(
            "Synapse.net_receive() must be implemented in subclasses."
        )


def rename(mechanism, new_name=None):
    """
    Returns a new mechanism that is an exact copy of `mechanism` but with a
    different name.

    Args:
        mechanism (Mechanism): The mechanism to be copied.
        new_name (str): The name of the new class.

    Returns:
        type: A new class, identical in behavior to the original but with a new name.
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
