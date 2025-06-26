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

    def initialize(self, v, celsius, diameters):
        self.ion_init(celsius)
        self.set_buffers(diameters)
        self.init_buffers(v)
        self.i(v)

    def ion_init(self, temp) -> None:
        for ion in self.ions.values():
            ion.initialize(temp)
    
    def write_to_ions(self, v):
        for ion, ion_c_write in self.write_ion_c.items():
            for k, conc_list in ion_c_write.items():
                meck = self.mechanisms[k]
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
        currents = {}
        gtot = {}
        if not self.currents:
            return torch.zeros_like(v), torch.zeros_like(v)
        for current, cdict in self.currents.items():
            currents[current] = torch.zeros_like(v)
            gtot[current] = torch.zeros_like(v)
            for mech, i_ion_list in cdict.items():
                mechanism = self.mechanisms[mech]
                for i_ion in i_ion_list:
                    i, g = getattr(mechanism, i_ion)(mechanism.get(v))
                    mechanism.add_(currents[current], i)
                    mechanism.add_(gtot[current], g)
        for ion, ion_h in self.ions.items():
            setattr(ion_h, f"i{ion}", currents[f"i{ion}"])
        total_i = torch.stack(list(currents.values()), dim=0).sum(dim=0)
        total_g = torch.stack(list(gtot.values()), dim=0).sum(dim=0)
        return total_i, total_g


    def set_buffers(self, diameters):
        for m in self.mechanisms.values():
            m.diam.set_(diameters)

        self.read_from_ions()

        for ion, dict_of_mech_and_quantities in self.write_ion_c.items():
            for mech, quantities in dict_of_mech_and_quantities.items():
                for quantity in quantities:
                    q = getattr(self.ions[ion], quantity)
                    nd = q.ndim
                    if self.keys[mech] is not None and nd > 0:
                        q = q[self.keys[mech]]
                    setattr(self.mechanisms[mech], quantity, torch.empty(q.shape, device=q.device, dtype=q.dtype))
                    getattr(self.mechanisms[mech], quantity).copy_(q)
                    for _, s in self.mechanisms[mech].DE.items():
                        setattr(s, quantity, getattr(self.mechanisms[mech], quantity))
        return

    def init_buffers(self, v):
        for m, mech in self.mechanisms.items():
            mech._init_buffers_s(mech.get(v))

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

