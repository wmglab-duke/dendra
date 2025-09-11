import torch

from ._mechanism import Mechanism, PointProcess, VoltageProcess


def make_scaler(mech, area):
    """
    A factory that returns a scaler function.

    The behavior of the scaler depends on the type of 'mech'.
    - For Mechanism: It's an identity function (returns inputs unchanged).
    - For PointProcess: It scales inputs by a factor derived from the area.

    The returned scaler intelligently returns a single value for a single
    input, or a tuple for multiple inputs.
    """
    if isinstance(mech, PointProcess):
        # Calculate the scaling factor once. This is a closure.
        # The 'arr' variable will be remembered by the _scaler function.
        arr = 1e6 * mech.get(area)

        # Add a check to prevent division by zero.
        if torch.any(arr == 0):
            raise ValueError(
                "Calculated area factor is zero, perhaps you inserted a PointProcess at a branchpoint?"
            )

        def _scaler(*args):
            # Scale all incoming arguments
            scaled_values = tuple(a / arr for a in args)

            # If only one argument was passed, return the single scaled value.
            if len(scaled_values) == 1:
                return scaled_values[0]
            # Otherwise, return the tuple of scaled values.
            return scaled_values

        return _scaler

    def _scaler(*args):
        # If only one argument was passed, return it directly.
        if len(args) == 1:
            return args[0]
        # Otherwise, return the tuple of arguments.
        return args

    return _scaler


class MechanismHandler(torch.nn.Module):
    """
    Base class for handling mechanisms in a model.
    This class provides a structure for managing mechanisms, including
    initialization, parameter handling, and method dispatching.
    """

    def __init__(
        self,
        celsius,
        area,
        mechs,
        ions=None,
        write_ion_c=None,
        read_ion=None,
        currents=None,
    ):
        super().__init__()
        self.mechanisms = torch.nn.ModuleDict()
        self.voltage_processes = torch.nn.ModuleDict()
        self.ions = torch.nn.ModuleDict()

        self.register_buffer("celsius", celsius)
        self.register_buffer("area", area)

        self.write_ion_c = write_ion_c if write_ion_c is not None else {}
        self.read_ion = read_ion if read_ion is not None else {}
        self.currents = currents if currents is not None else {}

        for mech_name, mech in mechs.items():
            if isinstance(mech, VoltageProcess):
                self.voltage_processes[mech_name] = mech
                setattr(self, mech_name, mech)
            else:
                if not isinstance(mech, Mechanism):
                    raise TypeError(
                        f"Mechanism {mech_name} must be an instance of Mechanism or VoltageProcess."
                    )
            self.mechanisms[mech_name] = mech
            setattr(self, mech_name, mech)

        if ions is not None:
            for ion_name, ion in ions.items():
                self.ions[ion_name] = ion
                setattr(self, f"{ion_name}_ion", ion)

        # --- flattened mapping (current-index, mechanism-obj, fn) ------------
        self._map = []
        self._map_exp = []

        self.ion_to_buff_idx = {}
        self.i_g_buffers_initialized = False

    def make_maps(self):
        """
        Create the mapping of current indices to mechanisms and their functions.
        This is used to efficiently compute currents and conductances.
        """
        self._map = []
        self._map_exp = []
        for c_idx, mech_dict in enumerate(self.currents.values()):
            for mech_name, ions in mech_dict.items():
                mech = self.mechanisms[mech_name]
                scale_f = make_scaler(mech, self.area)
                for ion in ions:
                    self._map.append(
                        (c_idx, mech, f"{ion}_with_g", scale_f)
                    )
                    self._map_exp.append(
                        (c_idx, mech, f"{ion}", scale_f)
                    )

    def initialize(self, v, celsius, diameters):
        self.make_maps()
        self.populate()
        self.ion_init(celsius)
        self.set_buffers(diameters)
        self.init_i_g_bufs(v)
        self.read_from_ions()
        self.compute_initial_conditions(v)
        self.i(v)
        self.write_to_ions(v)
        for ion in self.ions.values():
            ion.advance(celsius)
        self.read_from_ions()

    def update_v(self, v, dt):
        for vp in self.voltage_processes.values():
            v = vp.update_v(v, dt)
        return v

    def init_i_g_bufs(self, v):
        current_names = list(self.currents.keys())
        self._buf_i = [torch.zeros_like(v) for _ in current_names]
        self._buf_g = [torch.zeros_like(v) for _ in current_names]
        for ion in self.ions.keys():
            idx = current_names.index(f"i{ion}")
            self.ion_to_buff_idx[ion] = idx
        self.i_g_buffers_initialized = True

    def populate(self, mech=None) -> None:
        if mech is not None:
            self.mechanisms[mech].populate()
        else:
            for mech in self.mechanisms.values():
                mech.populate()

    def ion_init(self, temp) -> None:
        for ion in self.ions.values():
            ion.initialize(temp)

    def write_to_ions(self, v):
        for ion, ion_c_write in self.write_ion_c.items():
            for k, conc_list in ion_c_write.items():
                mech = self.mechanisms[k]
                for conc in conc_list:
                    ion_conc_u = mech._buffers[conc]
                    ion_conc_u = mech.put(ion_conc_u, self.ions[ion]._buffers[conc], v)
                    self.ions[ion]._buffers[conc] = ion_conc_u

    def read_from_ions(self):
        for ion, ion_read in self.read_ion.items():
            for k, conc_list in ion_read.items():
                mech = self.mechanisms[k]
                for conc in conc_list:
                    ion_conc = mech.get(self.ions[ion]._buffers[conc])
                    mech._buffers[conc] = ion_conc
                    for s in mech.DE.values():
                        s._buffers[conc] = ion_conc

    def advance(self, v, dt, temp):
        for mech_name, mech in self.mechanisms.items():
            mech._advance(mech.get(v), dt)

        # write ion concentrations
        self.write_to_ions(v)

        # update equilibrium potentials
        for ion in self.ions.values():
            ion.advance(temp)

        # read ion concentrations & equilibria
        self.read_from_ions()

    def detach_i_g_bufs(self):
        if not self.i_g_buffers_initialized:
            return
        for buf in self._buf_i:
            buf.detach_()
        for buf in self._buf_g:
            buf.detach_()

    def detach(self):
        for mech in self.mechanisms.values():
            mech.detach()
        for ion in self.ions.values():
            ion.detach()
        self.detach_i_g_bufs()

    def i(self, v):
        if not self.currents:
            return torch.tensor(0.0, dtype=v.dtype, device=v.device), torch.tensor(
                0.0, dtype=v.dtype, device=v.device
            )

        for mech in self.mechanisms.values():
            mech.breakpoint(mech.get(v))

        # reset buffers in-place (no realloc)
        for buf in self._buf_i:
            buf.detach().zero_()
        for buf in self._buf_g:
            buf.detach().zero_()

        # core loop: minimal Python, pure aten ops inside
        for c_idx, mech, fn, scale_f in self._map:
            i, g = scale_f(*getattr(mech, fn)(mech.get(v)))
            mech.add_(self._buf_i[c_idx], i)
            mech.add_(self._buf_g[c_idx], g)

        # sum up currents and conductances
        tot_i = sum(self._buf_i)
        tot_g = sum(self._buf_g)

        # expose per-ion currents
        for ion, ion_h in self.ions.items():
            ion_h._buffers[f"i{ion}"] = self._buf_i[self.ion_to_buff_idx[ion]]

        return tot_i, tot_g

    def iexp(self, v):
        if not self.currents:
            return torch.tensor(0.0, dtype=v.dtype, device=v.device)

        for mech in self.mechanisms.values():
            mech.breakpoint(mech.get(v))

        for buf in self._buf_i:
            buf.detach().zero_()

        # core loop: minimal Python, pure aten ops inside
        for c_idx, mech, fn, scale_f in self._map_exp:
            i = scale_f(getattr(mech, fn)(mech.get(v)))
            mech.add_(self._buf_i[c_idx], i)

        # sum up currents and conductances
        tot_i = sum(self._buf_i)

        # expose per-ion currents
        for ion, ion_h in self.ions.items():
            ion_h._buffers[f"i{ion}"] = self._buf_i[self.ion_to_buff_idx[ion]]

        return tot_i

    def idf(self, v, v_prev):
        if not self.currents:
            return torch.tensor(0.0, dtype=v.dtype, device=v.device), torch.tensor(
                0.0, dtype=v.dtype, device=v.device
            )

        v_half = 0.5 * v_prev

        for mech in self.mechanisms.values():
            mech.breakpoint(mech.get(v))

        for buf in self._buf_i:
            buf.detach().zero_()
        for buf in self._buf_g:
            buf.detach().zero_()

        for c_idx, mech, fn, scale_f in self._map:
            if mech.factorable:
                v_in = v_half
            else:
                v_in = v
            i, g = scale_f(getattr(mech, fn)(mech.get(v_in)))
            mech.add_(self._buf_i[c_idx], i)
            mech.add_(self._buf_g[c_idx], g)

        # sum up currents and conductances
        tot_i = sum(self._buf_i)
        tot_g = sum(self._buf_g)

        return tot_i, tot_g

    def itot(self, v):
        if not self.currents:
            return

        for buf in self._buf_i:
            buf.detach().zero_()

        for c_idx, mech, fn, scale_f in self._map_exp:
            i = scale_f(fn(mech.get(v)))
            mech.add_(self._buf_i[c_idx], i)

        for ion, ion_h in self.ions.items():
            ion_h._buffers[f"i{ion}"] = self._buf_i[self.ion_to_buff_idx[ion]]

    def set_buffers(self, diameters):
        for m in self.mechanisms.values():
            m.diam.set_(diameters)

        for ion, dict_of_mech_and_quantities in self.write_ion_c.items():
            for mech, quantities in dict_of_mech_and_quantities.items():
                m = self.mechanisms[mech]
                for quantity in quantities:
                    q = m.get(getattr(self.ions[ion], quantity))
                    setattr(
                        m,
                        quantity,
                        torch.empty(q.shape, device=q.device, dtype=q.dtype),
                    )
                    getattr(m, quantity).copy_(q)
                    for _, s in self.mechanisms[mech].DE.items():
                        setattr(s, quantity, getattr(m, quantity))

    def compute_initial_conditions(self, v):
        for m, mech in self.mechanisms.items():
            mech._init_buffers_s(mech.get(v))

    def all_states(self):
        """
        Return a dictionary of all states in the MechanismHandler.
        """
        states = []
        for mech in self.mechanisms.values():
            states.extend(mech.states())
        return states
