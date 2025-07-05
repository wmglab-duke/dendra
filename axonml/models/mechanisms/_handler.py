import torch


class MechanismHandler(torch.nn.Module):
    """
    Base class for handling mechanisms in a model.
    This class provides a structure for managing mechanisms, including
    initialization, parameter handling, and method dispatching.
    """

    def __init__(
        self, 
        celsius, 
        mechs,
        ions=None, 
        write_ion_c=None, 
        read_ion=None, 
        currents=None
    ):
        super().__init__()
        self.mechanisms = torch.nn.ModuleDict()
        self.ions = torch.nn.ModuleDict()
        self.celsius = celsius

        self.write_ion_c = write_ion_c if write_ion_c is not None else {}
        self.read_ion = read_ion if read_ion is not None else {}
        self.currents = currents if currents is not None else {}

        for mech_name, mech in mechs.items():
            self.mechanisms[mech_name] = mech
            setattr(self, mech_name, mech)

        if ions is not None:
            for ion_name, ion in ions.items():
                self.ions[ion_name] = ion
                setattr(self, f"{ion_name}_ion", ion)

        # --- flattened mapping (current-index, mechanism-obj, fn) ------------
        self._map = []
        self._map_df = []
        for c_idx, mech_dict in enumerate(currents.values()):
            for mech_name, ions in mech_dict.items():
                mech = self.mechanisms[mech_name]
                for ion in ions:
                    self._map.append( (c_idx, mech, getattr(mech, f"{ion}_with_g")) )
                    self._map_df.append( (c_idx, mech, getattr(mech, f"{ion}")) )

        self.ion_to_buff_idx = {}
        self.i_g_buffers_initialized = False

    def initialize(self, v, celsius, diameters):
        self.ion_init(celsius)
        self.set_buffers(diameters)
        self.init_i_g_bufs(v)
        self.compute_initial_conditions(v)
        self.read_from_ions()
        self.i(v)
        self.write_to_ions(v)
        for ion in self.ions.values():
            ion.advance(celsius)
        self.read_from_ions()

    def init_i_g_bufs(self, v):
        if not self.i_g_buffers_initialized:
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
                    ion_conc_u = getattr(mech, conc)
                    ion_conc_u = mech.put(ion_conc_u, getattr(self.ions[ion], conc), v)
                    setattr(self.ions[ion], conc, ion_conc_u)

    def read_from_ions(self):
        for ion, ion_read in self.read_ion.items():
            for k, conc_list in ion_read.items():
                mech = self.mechanisms[k]
                for conc in conc_list:
                    ion_conc = mech.get(getattr(self.ions[ion], conc))
                    setattr(mech, conc, ion_conc)
                    for s in mech.DE.values():
                        setattr(s, conc, ion_conc)

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

    def detach(self):
        for mech in self.mechanisms.values():
            mech.detach()
        for ion in self.ions.values():
            ion.detach()


    def i(self, v):
        if not self.currents:
            return 0.0, 0.0

        # reset buffers in-place (no realloc)
        for t in self._buf_i:
            t.zero_()
        for t in self._buf_g:
            t.zero_()

        for mech in self.mechanisms.values():
            mech.breakpoint(v)

        # core loop: minimal Python, pure aten ops inside
        for c_idx, mech, fn in self._map:
            i, g = fn(mech.get(v))
            mech.add_(self._buf_i[c_idx], i)
            mech.add_(self._buf_g[c_idx], g)
        
        # sum up currents and conductances
        tot_i = sum(self._buf_i)
        tot_g = sum(self._buf_g)

        # expose per-ion currents
        for (ion, ion_h) in self.ions.items():
            setattr(ion_h, f"i{ion}", self._buf_i[self.ion_to_buff_idx[ion]])

        return tot_i, tot_g


    def idf(self, v, v_prev):
        if not self.currents:
            return 0.0, 0.0

        for t in self._buf_i:
            t.zero_()
        for t in self._buf_g:
            t.zero_()

        v_half = 0.5 * v_prev

        for mech in self.mechanisms.values():
            mech.breakpoint(v)

        for c_idx, mech, fn in self._map:
            if mech.factorable:
                v_in = v_half
            else:
                v_in = v
            i, g = fn(mech.get(v_in))
            mech.add_(self._buf_i[c_idx], i)
            mech.add_(self._buf_g[c_idx], g)
        
        # sum up currents and conductances
        tot_i = sum(self._buf_i)
        tot_g = sum(self._buf_g)

        return tot_i, tot_g

    def itot(self, v):
        if not self.currents:
            return

        for t in self._buf_i:
            t.zero_()

        for c_idx, mech, fn in self._map_df:
            i = fn(mech.get(v))
            mech.add_(self._buf_i[c_idx], i)

        for ion, ion_h in self.ions.items():
            setattr(ion_h, f"i{ion}", self._buf_i[self.ion_to_buff_idx[ion]])


    def set_buffers(self, diameters):
        for m in self.mechanisms.values():
            m.diam.set_(diameters)

        for ion, dict_of_mech_and_quantities in self.write_ion_c.items():
            for mech, quantities in dict_of_mech_and_quantities.items():
                m = self.mechanisms[mech]
                for quantity in quantities:
                    q = m.get(getattr(self.ions[ion], quantity))
                    nd = q.ndim
                    setattr(m, quantity, torch.empty(q.shape, device=q.device, dtype=q.dtype))
                    getattr(m, quantity).copy_(q)
                    for _, s in self.mechanisms[mech].DE.items():
                        setattr(s, quantity, getattr(m, quantity))
        
    def compute_initial_conditions(self, v):
        for m, mech in self.mechanisms.items():
            mech._init_buffers_s(mech.get(v))
