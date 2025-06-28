from types import MethodType
import torch

from axonml.models.parametric import _Parameterized
from ._ions import VALENCES
from ._symbolic import build_current_eq


class Mechanism(_Parameterized):
    _state = set()
    _ion = set()
    _range = set()
    _assigned = set()

    _state_declarations = []
    _ion_declarations = []
    _range_declarations = []
    _assigned_declarations = []

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
        new_range = set()
        new_assigned = set()

        new_read_ion = {}
        new_write_ion = {}
        new_write_ion_c = {}

        new_currents = {}
        new_init = {}
        
        # Walk MRO in reverse to build up params from parent to child
        for base in reversed(cls.__mro__):
            # We look for a _params attribute defined directly on the base
            if '_state' in base.__dict__:
                new_state.update(base._state)
            if '_ion' in base.__dict__:
                new_ion.update(base._ion)
            if '_range' in base.__dict__:
                new_range.update(base._range)
            if '_assigned' in base.__dict__:
                new_assigned.update(base._assigned)
            if '_read_ion' in base.__dict__:
                new_read_ion.update(base._read_ion)
            if '_write_ion' in base.__dict__:
                new_write_ion.update(base._write_ion)
            if '_write_ion_c' in base.__dict__:
                new_write_ion_c.update(base._write_ion_c)
            if '_currents' in base.__dict__:
                new_currents.update(base._currents)
            if '_init' in base.__dict__:
                new_init.update(base._init)
        
        if Mechanism._state_declarations:
            for s_list in Mechanism._state_declarations:
                new_state.update(s_list)
            Mechanism._state_declarations = [] # Clear for next class
        if Mechanism._ion_declarations:
            for i_list in Mechanism._ion_declarations:
                new_ion.update(i_list)
            Mechanism._ion_declarations = []
        if Mechanism._range_declarations:
            for r_list in Mechanism._range_declarations:
                new_range.update(r_list)
            Mechanism._range_declarations = []
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
                new_currents.setdefault('nonspecific', []).extend(c_list)
            Mechanism._currents_declarations = []
        if Mechanism._init_declarations:
            for i_dict in Mechanism._init_declarations:
                new_init.update(i_dict)
            Mechanism._init_declarations = []

        cls._state = new_state
        cls._ion = new_ion
        cls._range = new_range
        cls._currents = new_currents
        cls._assigned = new_assigned
        cls._read_ion = new_read_ion
        cls._write_ion = new_write_ion
        cls._write_ion_c = new_write_ion_c

    def __init__(
        self,
        name:str, 
        celsius, 
        diameters, 
        shape, 
        key=None,
        is_composable=False,
        additional_parameters=None,
        ic: dict = None,
        **kwargs
    ):
        """
        Initialize the Mechanism with parameters and declarations.
        """
        super().__init__(shape, additional_parameters=additional_parameters, **kwargs)
        self._name = name
        self.celsius = celsius

        if key is not None:
            if is_composable:
                self.key = key
            else:
                self.register_buffer('key', torch.tensor(key, dtype=torch.long))
        else:
            self.key = None

        self.is_composable = is_composable

        if self.key is None:
            self.get  = lambda tensor: tensor
            self.add_ = lambda add_to, add_what: add_to.add_(add_what)
            self.add  = lambda add_to, add_what: add_to.add(add_what)
            self.put  = self.put_no_op
        elif self.is_composable:
            self.get  = lambda tensor: tensor[self.key] if tensor.ndim > 0 else tensor
            self.add_ = lambda add_to, add_what: add_to[self.key].add_(add_what)
            self.add  = lambda add_to, add_what: add_to[self.key].add(add_what)
            self.put  = self.put_slice
        else:
            self.get  = lambda tensor: tensor.view(-1).index_select(0, self.key) if tensor.ndim > 0 else tensor
            self.add_ = lambda add_to, add_what: add_to.view(-1).index_add_(0, self.key, add_what)
            self.add  = lambda add_to, add_what: add_to.view(-1).index_add(0, self.key, add_what)
            self.put  = self.put_fancy

        self.read_ion    = self._read_ion
        self.write_ion_c = self._write_ion_c

        states = [
            state(
                celsius, 
                diameters, 
                key, 
                shape, 
                additional_parameters=additional_parameters, 
                **kwargs
            )
            for state in self._state
        ]

        self.DE = torch.nn.ModuleDict(
            {state._name: state for state in states}
        )

        self._init_params: Dict[str, float] = {k: v for k, v in self._init.items()}
        if ic is not None:
            self._init_params.update(ic)

        self.register_buffer('diam', diameters)
        for state_name in self.DE:
            self.register_buffer(state_name, torch.zeros(shape))

        for r in self._range:
            self.register_buffer(r, torch.zeros(shape))

        # factorize current equations
        current_eqs = []
        for _, v in self._currents.items():
            current_eqs.extend(v)
        for _, v in self._write_ion.items():
            current_eqs.extend(v)

        for k in current_eqs:
            assign = k in self._range
            eq = build_current_eq(self, k, assign=assign)
            setattr(self, k, MethodType(eq, self))

        self.populate()

    @property
    def name(self):
        return self._name

    def put_no_op(self, ion_conc_u, ion_conc_o, v):
        return ion_conc_u

    def put_slice(self, ion_conc_u, ion_conc_o, v):
        ion_conc_o = ion_conc_o.expand_as(v).clone()
        ion_conc_o[self.key] = ion_conc_u
        return ion_conc_o

    def put_fancy(self, ion_conc_u, ion_conc_o, v):
        ion_conc_o = ion_conc_o.expand_as(v).clone()
        ion_conc_o.view(-1).index_put_((self.key,), ion_conc_u)
        return ion_conc_o

    def register_ion(self, ion):
        name = ion.name
        if name in self.read_ion:
            for v in self.read_ion[name]:
                q = getattr(ion, v)
                if (k:=self.key) is not None and q.ndim > 0:
                    self.register_buffer(v, q[k])
                    for _, s in self.DE.items():
                        s.register_buffer(v, q[k])
                else:
                    self.register_buffer(v, q)
                    for _, s in self.DE.items():
                        s.register_buffer(v, q)

        if name in self.write_ion_c:
            for v in self.write_ion_c[name]:
                q = getattr(ion, v)
                if (k:=self.key) is not None and q.ndim > 0:
                    self.register_buffer(v, torch.empty(q[k].shape, device=q[k].device, dtype=q[k].dtype))
                    getattr(self, v).copy_(q[k])
                else:
                    self.register_buffer(v, q)

    def _init_buffers_s(self, v_init):
        for state_module in self.DE.values():
            state_names = state_module._state
            for state_name in state_names:
                if state_name in self._init_params:
                    buffer_tensor = torch.tensor(
                        self._init_params[state_name], device=v_init.device, dtype=v_init.dtype
                    )
                    setattr(self, state_name, buffer_tensor)
                    buffer_tensor.detach_()
                else:
                    if hasattr(state_module, 'inf'):
                        inf = state_module.inf(v_init)
                        buffer_tensor = inf[state_name]
                        setattr(self, state_name, buffer_tensor)
                        buffer_tensor.detach_()

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
    def RANGE(*args):
        Mechanism._range_declarations.append(args)

    @staticmethod
    def USEION(ion, read=None, write=None):
        read = read or []
        write = write or []

        if not read and not write:
            return
        
        assert ion in VALENCES, f"Unknown ion {ion}. Valid ions are {list(VALENCES.keys())}."

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

    def detach(self):
        for n, b in self.named_buffers():
            b.detach_()

    def _advance(self, v, dt):
        local = {}
        for state_module in self.DE.values():
            states = {state_name: getattr(self, state_name) for state_name in state_module._state}
            local.update(state_module.advance(v, dt, states))
        for k, v in local.items():
            setattr(self, k, v)

    def populate(self):
        self.populate_parameter_buffers()
        for state_module in self.DE.values():
            state_module.populate_parameter_buffers()

    def initial(self, v):
        pass
