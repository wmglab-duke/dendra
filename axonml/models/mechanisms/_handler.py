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
        keys,
        has_gtot, 
        ions=None, 
        write_ion_c=None, 
        read_ion=None, 
        currents=None
    ):
        super().__init__()
        self.mechanisms = torch.nn.ModuleDict()
        self.mech_with_gtot = torch.nn.ModuleList()
        self.ions = torch.nn.ModuleDict()
        self.celsius = celsius

        self.write_ion_c = write_ion_c if write_ion_c is not None else {}
        self.read_ion = read_ion if read_ion is not None else {}
        self.currents = currents if currents is not None else {}

        self.keys = keys

        for mech_name, mech in mechs.items():
            self.mechanisms[mech_name] = mech
            setattr(self, mech_name, mech)
            if has_gtot[mech_name]:
                self.mech_with_gtot.append(mech)

        if ions is not None:
            for ion_name, ion in ions.items():
                self.ions[ion_name] = ion
                setattr(self, f"{ion_name}_ion", ion)

    def initialize(self, v, v_init, temp):
        self.ion_init(temp)
        self.init_buffers(v)
        self.i(v)

    def ion_init(self, temp) -> None:
        for ion in self.ions.values():
            ion.initialize(temp)

    def advance(self, v, dt, temp):
        for mech, key in zip(self.mechanisms.values(), self.keys.values()):
            if key is not None:
                mech._advance(v[key], dt)
            else:
                mech._advance(v, dt)

        # write ion concentrations
        for ion, ion_c_write in self.write_ion_c.items():
            for k, conc_list in ion_c_write.items():
                for conc in conc_list:
                    ion_conc_u = getattr(self.mechanisms[k], conc)
                    if (key := self.keys[k]) is not None:
                        ion_conc_o = getattr(self.ions[ion], conc).copy().expand_as(v)
                        ion_conc_o[key] = ion_conc_u
                        setattr(self.ions[ion], conc, ion_conc_o)
                    else:
                        setattr(self.ions[ion], conc, ion_conc_u)
        
        # update equilibrium potentials
        for ion in self.ions.values():
            ion.advance(temp)

        # read ion concentrations & equilibria
        for ion, ion_read in self.read_ion.items():
            for k, v in ion_read.items():
                for conc in v:
                    q = getattr(self.ions[ion], conc)
                    if (key := self.keys[k]) is not None and q.ndim > 0:
                        setattr(self.mechanisms[k], conc, q[key])
                    else:
                        setattr(self.mechanisms[k], conc, q)


    def detach(self):
        for mech in self.mechanisms.values():
            mech.detach()
        for ion in self.ions.values():
            ion.detach()

    def i(self, v):
        currents = {}
        gtot = {}
        for current, cdict in self.currents.items():
            currents[current] = torch.zeros_like(v)
            gtot[current] = torch.zeros_like(v)
            for mech, i_ion_list in cdict.items():
                for i_ion in i_ion_list:
                    if (key := self.keys[mech]) is not None:
                        i, g = getattr(self.mechanisms[mech], i_ion)(v[key])
                        currents[current][key] += i
                        gtot[current][key] += g
                    else:
                        i, g = getattr(self.mechanisms[mech], i_ion)(v)
                        currents[current] += i
                        gtot[current] += g
        for ion, ion_h in self.ions.items():
            setattr(ion_h, f"i{ion}", currents[f"i{ion}"])
        total_i = torch.stack(list(currents.values()), dim=0).sum(dim=0)
        total_g = torch.stack(list(gtot.values()), dim=0).sum(dim=0)
        return total_i, total_g


    def set_buffers(self, diameters):
        for m in self.mechanisms.values():
            m.diam.set_(diameters)

        for ion, dict_of_mech_and_quantities in self.read_ion.items():
            for mech, quantities in dict_of_mech_and_quantities.items():
                for quantity in quantities:
                    q = getattr(self.ions[ion], quantity)
                    nd = q.ndim
                    if self.keys[mech] is not None and nd > 0:
                        q = q[self.keys[mech]]
                    setattr(self.mechanisms[mech], quantity, q)
                    for _, s in self.mechanisms[mech].DE.items():
                        setattr(s, quantity, q)

        for ion, dict_of_mech_and_quantities in self.write_ion_c.items():
            for mech, quantities in dict_of_mech_and_quantities.items():
                for quantity in quantities:
                    q = getattr(self.ions[ion], quantity)
                    nd = q.ndim
                    if self.keys[mech] is not None and nd > 0:
                        q = q[self.keys[mech]]
                    setattr(self.mechanisms[mech], quantity, q)
        return

    def init_buffers(self, v):
        for m, mech in self.mechanisms.items():
            key = self.keys[m]
            if key is not None:
                mech._init_buffers_s(v[key])
            else:
                mech._init_buffers_s(v)

    def gtot(self, v):
        """
        Calculate the total conductance for all mechanisms.
        This method sums the conductances of all mechanisms and returns the total.
        """
        gtot = torch.zeros_like(v)
        for mech in self.mech_with_gtot:
            if self.keys[mech._name] is not None:
                gtot[self.keys[mech._name]] += mech.gtot(v[self.keys[mech._name]])
            else:
                gtot += mech.gtot(v)
        return gtot

