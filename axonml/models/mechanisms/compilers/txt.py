template = """
class {mech}(torch.nn.Module):
    _init_params: Dict[str, float]
    def __init__(
            self, 
            temp, 
            diameters, 
            shape,
            n_ax: int,
            n_comps: int,
            name: str, 
            params, 
            distributions, 
            read_ion, 
            write_ion_c, 
            states, 
            init, 
            model,
            ic: dict = None
        ):
        super().__init__()
        
        self._name = name

        self.instantiate_parameters(params, model)
        self.instantiate_distributions(distributions)
        self.temp = temp

        if len(diameters.shape) == 1:
            final_axis = 1
        else:
            final_axis = diameters.shape[-1]
        
        try:
            self.register_buffer("diam", diameters.view(diameters.shape[0], 1, final_axis))
        except:
            self.register_buffer("diam", diameters)

        self.DE = torch.nn.ModuleDict(
            {{state._name: state for state in states}}
        )
                
        self._init_params: Dict[str, float] = {{k: v for k, v in init.items()}}

        if ic is not None:
            self._init_params.update(ic)

        self.n_ax = n_ax
        self.n_comps = n_comps
        self.read_ion = read_ion
        self.write_ion_c = write_ion_c

{mask}

{state_buffers}

{distribution_buffers}

{current_buffers}

{implicit_buffers}

{assigned}

    def register_ion(self, ion):
        name = ion.name
        if name in self.read_ion:
            for v in self.read_ion[name]:
                self.register_buffer(v, getattr(ion, v))
                for _, s in self.DE.items():
                    s.register_buffer(v, getattr(ion, v))

        if name in self.write_ion_c:
            for v in self.write_ion_c[name]:
                self.register_buffer(v, getattr(ion, v))

    def instantiate_parameters(self, params, model):
        if params is not None:
            for name, value in params.items():
                if isinstance(value, dict):
                    setattr(self, name, [])
                    for pname, pval in value.items():
                        setattr(self, pname, to_param(pval, model))
                        getattr(self, name).append(getattr(self, pname))
                else:
                    setattr(self, name, to_param(value, model))

    def detach(self):
{detach}
        return

    def instantiate_distributions(self, distributions):
        if distributions is not None:
            for name, dist in distributions.items():
                setattr(self, name+"_d", dist)

    def _init_buffers_s(self, v_init):
{init_state_buffers}
{init_distribution_buffers}
        self.initial(v_init)
        for _, s in self.DE.items():
            s.initialize(v_init)
        return

    @torch.jit.ignore
    def set(self, key: str, value):
        p = getattr(self, key)
        if isinstance(p, torch.Tensor):
            p.data = torch.as_tensor(value, dtype=p.data.dtype, device=p.device)

    @torch.jit.ignore
    def get(self, key: str):
        return getattr(self, key)

    def _advance(self, v, dt):
{advance}
        return

{initial_f}

{breakpoint_f}

{generic_f}

{net_receive_f}

{coupled_infs}

{current_equations}

{gtot}

{irev}
"""


init_state_buffer_template = """
if '{state}' in self._init_params:
    buffer_tensor = torch.tensor(self._init_params['{state}'], device=v_init.device, dtype=v_init.dtype)
else:
    buffer_tensor = self.DE['{state}'].inf(v_init)
self.{state}[:] = buffer_tensor
self.{state}.detach_()
"""


init_state_buffers_coupled_template = """
if '{state}' in self._init_params:
    buffer_tensor = torch.tensor(self._init_params['{state}'], device=v_init.device, dtype=v_init.dtype)
else:
    buffer_tensor = self.{state}_inf(v_init)
self.{state}[:] = buffer_tensor
self.{state}.detach_()
"""


init_distribution_buffers_template = """
self.{name} = self.{name}_d._sample(self.{name})
"""