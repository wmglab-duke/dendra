import torch
import torch._dynamo as dynamo
import torch._inductor.config as inductor_config

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

        self.update_ion_buf = {}

        self.shape = None

        if self.write_ion_c:
            self.write_to_ions = dynamo.disable(self.write_to_ions)
        if self.read_ion:
            self.read_from_ions = dynamo.disable(self.read_from_ions)

        if self.write_ion_c or self.read_ion:
            inductor_config.cpp_wrapper = False
        else:
            inductor_config.cpp_wrapper = True

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
                    self._map.append((c_idx, mech, f"{ion}_with_g", scale_f))
                    self._map_exp.append((c_idx, mech, f"{ion}", scale_f))

    def initialize(self, v, celsius, diameters, populate=True):
        self.make_maps()
        self.init_rng()
        if populate:
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
        self.write_to_ions(v)

    def update_v(self, v):
        for vp in self.voltage_processes.values():
            v = vp.update_v(v)
        return v

    def init_i_g_bufs(self, v):
        current_names = list(self.currents.keys())
        self._buf_i = [torch.zeros_like(v) for _ in current_names]
        self._buf_g = [torch.zeros_like(v) for _ in current_names]
        for ion in self.ions.keys():
            try:
                idx = current_names.index(f"i{ion}")
                self.ion_to_buff_idx[ion] = idx
                self.update_ion_buf[ion] = True
            except ValueError:
                self.update_ion_buf[ion] = False

        self.i_g_buffers_initialized = True

    def init_rng(self):
        for mech in self.mechanisms.values():
            mech.init_rng()

    def reset_rng(self):
        for mech in self.mechanisms.values():
            mech.reset_rng()

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
        for i, buf in enumerate(self._buf_i):
            self._buf_i[i] = buf.detach()
        for i, buf in enumerate(self._buf_g):
            self._buf_g[i] = buf.detach()

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

        # reset buffers
        for i, buf in enumerate(self._buf_i):
            self._buf_i[i] = torch.zeros_like(buf)
        for i, buf in enumerate(self._buf_g):
            self._buf_g[i] = torch.zeros_like(buf)

        # core loop: minimal Python, pure aten ops inside
        for c_idx, mech, fn, scale_f in self._map:
            i, g = scale_f(*getattr(mech, fn)(mech.get(v)))
            mech.add_(self._buf_i[c_idx], i)
            mech.add_(self._buf_g[c_idx], g)

        # sum up currents and conductances
        tot_i = torch.stack(self._buf_i).sum(dim=0)
        tot_g = torch.stack(self._buf_g).sum(dim=0)

        # expose per-ion currents
        for ion, ion_h in self.ions.items():
            if self.update_ion_buf.get(ion, False):
                ion_h._buffers[f"i{ion}"] = self._buf_i[self.ion_to_buff_idx[ion]]

        return tot_i, tot_g

    def iexp(self, v):
        if not self.currents:
            return torch.tensor(0.0, dtype=v.dtype, device=v.device)

        for mech in self.mechanisms.values():
            mech.breakpoint(mech.get(v))

        for i, buf in enumerate(self._buf_i):
            self._buf_i[i] = torch.zeros_like(buf)

        # core loop: minimal Python, pure aten ops inside
        for c_idx, mech, fn, scale_f in self._map_exp:
            i = scale_f(getattr(mech, fn)(mech.get(v)))
            mech.add_(self._buf_i[c_idx], i)

        # sum up currents and conductances
        tot_i = torch.stack(self._buf_i).sum(dim=0)

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

        # reset buffers
        for i, buf in enumerate(self._buf_i):
            self._buf_i[i] = torch.zeros_like(buf)
        for i, buf in enumerate(self._buf_g):
            self._buf_g[i] = torch.zeros_like(buf)

        for c_idx, mech, fn, scale_f in self._map:
            if mech.factorable:
                v_in = v_half
            else:
                v_in = v
            i, g = scale_f(getattr(mech, fn)(mech.get(v_in)))
            mech.add_(self._buf_i[c_idx], i)
            mech.add_(self._buf_g[c_idx], g)

        # sum up currents and conductances
        tot_i = torch.stack(self._buf_i).sum(dim=0)
        tot_g = torch.stack(self._buf_g).sum(dim=0)

        return tot_i, tot_g

    def itot(self, v):
        if not self.currents:
            return

        for i, buf in enumerate(self._buf_i):
            self._buf_i[i] = torch.zeros_like(buf)

        for c_idx, mech, fn, scale_f in self._map_exp:
            i = scale_f(getattr(mech, fn)(mech.get(v)))
            mech.add_(self._buf_i[c_idx], i)

        for ion, ion_h in self.ions.items():
            ion_h._buffers[f"i{ion}"] = self._buf_i[self.ion_to_buff_idx[ion]]

        return torch.stack(self._buf_i).sum(dim=0)

    def set_buffers(self, diameters):
        for m in self.mechanisms.values():
            m.diam = m.diam.set_(diameters).detach().clone()

        for ion, dict_of_mech_and_quantities in self.write_ion_c.items():
            for mech, quantities in dict_of_mech_and_quantities.items():
                m = self.mechanisms[mech]
                for quantity in quantities:
                    q = m.get(getattr(self.ions[ion], quantity))
                    setattr(
                        m,
                        quantity,
                        q.clone(),
                    )
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

    def mutable_state_dict(self):
        """
        Return a dictionary of all mutable / rebound states in the MechanismHandler.
        """
        states = {}
        for mech_name, mech in self.mechanisms.items():
            for rng_name in mech._rng:
                states[f"{mech_name}.{rng_name}"] = getattr(mech, rng_name).rng_state()
            for _, state in mech.DE.items():
                for state_name in state._state:
                    states[f"{mech_name}.{state_name}"] = getattr(mech, state_name)
                for rng_name in state._rng:
                    states[f"{mech_name}.{state_name}.{rng_name}"] = getattr(
                        state, rng_name
                    ).rng_state()
            for buffer_name in mech._assigned:
                states[f"{mech_name}.{buffer_name}"] = mech._buffers[buffer_name]
        for ion, ion_read in self.read_ion.items():
            for k, conc_list in ion_read.items():
                mech = self.mechanisms[k]
                for conc in conc_list:
                    states[f"{k}.{conc}"] = mech._buffers[conc]
        for ion_name, ion in self.ions.items():
            for buffer_name, buffer in ion.named_buffers():
                states[f"{ion_name}_ion.{buffer_name}"] = buffer
        return states

    def restore_mutable_state_dict(self, state_dict):
        """
        Restore mutable states from a given state dictionary.
        Assumes that the state_dict was created by mutable_state_dict().
        """
        for mech_name, mech in self.mechanisms.items():
            for rng_name in mech._rng:
                key = f"{mech_name}.{rng_name}"
                getattr(mech, rng_name).set_rng_state(state_dict[key])
            for _, state in mech.DE.items():
                for state_name in state._state:
                    key = f"{mech_name}.{state_name}"
                    setattr(mech, state_name, state_dict[key])
                for rng_name in state._rng:
                    key = f"{mech_name}.{state_name}.{rng_name}"
                    getattr(state, rng_name).set_rng_state(state_dict[key])
            for buffer_name in mech._assigned:
                key = f"{mech_name}.{buffer_name}"
                setattr(mech, buffer_name, state_dict[key])
        for ion, ion_read in self.read_ion.items():
            for k, conc_list in ion_read.items():
                mech = self.mechanisms[k]
                for conc in conc_list:
                    key = f"{k}.{conc}"
                    setattr(mech, conc, state_dict[key])
                    for s in mech.DE.values():
                        setattr(s, conc, state_dict[key])
        for ion_name, ion in self.ions.items():
            for buffer_name, _ in ion.named_buffers():
                key = f"{ion_name}_ion.{buffer_name}"
                setattr(ion, buffer_name, state_dict[key])
