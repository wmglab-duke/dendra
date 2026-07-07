import torch
import torch._dynamo as dynamo
import torch._inductor.config as inductor_config

from ._material_process import MaterialProcess
from ._mechanism import Mechanism, PointProcess, VoltageProcess

_MATERIAL_PHASE_ALIASES = {
    "": "post_local",
    "none": "post_local",
    "default": "post_local",
    "postlocal": "post_local",
    "post_local": "post_local",
    "after_local": "post_local",
    "after_reactions": "post_local",
    "exchange": "post_local",
    "exchanges": "post_local",
    "pool_exchange": "post_local",
    "local_exchange": "post_local",
    "reaction": "post_local",
    "reactions": "post_local",
    "transport": "transport",
    "spatial": "transport",
    "diffusion": "transport",
    "posttransport": "post_transport",
    "post_transport": "post_transport",
    "after_transport": "post_transport",
    "post_diffusion": "post_transport",
    "after_diffusion": "post_transport",
    "clamp": "post_transport",
    "clamps": "post_transport",
    "post_clamp": "post_transport",
    "bath": "post_transport",
    "boundary": "post_transport",
}

_MATERIAL_PHASE_ORDER = ("post_local", "transport", "post_transport")


def _canonical_material_phase(phase) -> str:
    p = str(phase or "post_local").lower().replace("-", "_")
    return _MATERIAL_PHASE_ALIASES.get(p, p)


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
        materials=None,
        write_ion_c=None,
        read_ion=None,
        read_material=None,
        write_material=None,
        source_material=None,
        currents=None,
        population=None,
    ):
        super().__init__()
        self.mechanisms = torch.nn.ModuleDict()
        self.voltage_processes = torch.nn.ModuleDict()
        self.material_processes = torch.nn.ModuleDict()
        self.ions = torch.nn.ModuleDict()
        self.materials = torch.nn.ModuleDict()

        self.register_buffer("celsius", celsius)
        self.register_buffer("area", area)

        self.write_ion_c = write_ion_c if write_ion_c is not None else {}
        self.read_ion = read_ion if read_ion is not None else {}
        self.read_material = read_material if read_material is not None else {}
        self.write_material = write_material if write_material is not None else {}
        self.source_material = source_material if source_material is not None else {}
        self.currents = currents if currents is not None else {}

        for mech_name, mech in mechs.items():
            if isinstance(mech, MaterialProcess):
                self.material_processes[mech_name] = mech
                setattr(self, mech_name, mech)
                continue
            if isinstance(mech, VoltageProcess):
                self.voltage_processes[mech_name] = mech
                setattr(self, mech_name, mech)
            else:
                if not isinstance(mech, Mechanism):
                    raise TypeError(
                        f"Mechanism {mech_name} must be an instance of Mechanism, "
                        "VoltageProcess, or MaterialProcess."
                    )
            self.mechanisms[mech_name] = mech
            setattr(self, mech_name, mech)

        if ions is not None:
            for ion_name, ion in ions.items():
                self.ions[ion_name] = ion
                setattr(self, f"{ion_name}_ion", ion)

        if materials is not None:
            for material_name, material in materials.items():
                self.materials[material_name] = material
                setattr(self, f"{material_name}_material", material)

        for process in self.material_processes.values():
            process.bind_materials(self._get_material, population=population)

        self._material_process_order = self._ordered_material_process_names()

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
        if self.write_material or self.source_material:
            self.write_to_materials = dynamo.disable(self.write_to_materials)
        if self.read_material:
            self.read_from_materials = dynamo.disable(self.read_from_materials)

        if (
            self.write_ion_c
            or self.read_ion
            or self.write_material
            or self.source_material
            or self.read_material
            or self.material_processes
        ):
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

    def initialize(self, v, celsius, diameters, populate=True, random_generation=None):
        self.make_maps()
        self.init_rng()
        if populate:
            self.populate(random_generation=random_generation)
        self.ion_init(celsius)
        self.material_init(celsius)
        self.set_buffers(diameters)
        self.init_i_g_bufs(v)
        self.read_from_ions()
        self.read_from_materials()
        self.compute_initial_conditions(v)
        self.i(v)
        self.write_to_ions(v)
        self.write_to_materials(v)
        for ion in self.ions.values():
            ion.advance(celsius)
        for material in self.materials.values():
            material.advance(celsius)
        self.read_from_ions()
        self.read_from_materials()
        self.write_to_ions(v)
        self.write_to_materials(v)

    def set_dt(self, dt):
        """Propagate timestep changes to local mechanisms and material processes.

        Integrators call this from their initialization path so process-level
        solvers can precompute timestep-dependent quantities once per run/dt,
        rather than inside every MaterialProcess.advance_materials(...) call.
        """
        for mech in self.mechanisms.values():
            mech.set_dt(dt)
        for process in self.material_processes.values():
            process.set_dt(dt)

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
        for process in self.material_processes.values():
            process.init_rng()

    def reset_rng(self):
        for mech in self.mechanisms.values():
            mech.reset_rng()
        for process in self.material_processes.values():
            process.reset_rng()

    def populate(self, mech=None, random_generation=None) -> None:
        if mech is not None:
            if mech in self.mechanisms:
                self.mechanisms[mech].populate(random_generation=random_generation)
            elif mech in self.material_processes:
                self.material_processes[mech].populate(
                    random_generation=random_generation
                )
            else:
                raise KeyError(mech)
        else:
            for mech in self.mechanisms.values():
                mech.populate(random_generation=random_generation)
            for process in self.material_processes.values():
                process.populate(random_generation=random_generation)

    def resample_random_parameters(self, *names, force: bool = True):
        """Resample random parameters on all mechanisms that define them."""

        for mech in self.mechanisms.values():
            if hasattr(mech, "resample_random_parameters"):
                mech.resample_random_parameters(*names, force=force)
        for process in self.material_processes.values():
            if hasattr(process, "resample_random_parameters"):
                process.resample_random_parameters(*names, force=force)
        return self

    def sample_runtime_noises_(
        self,
        *names,
        dt=None,
        phase: str | None = "pre_state",
        step_index: int | None = None,
        force: bool = False,
    ):
        """Refresh detached runtime ``NOISE`` buffers on mechanisms/processes.

        The handler owns the population-level collection of inserted mechanisms,
        so the integrator calls this single method before the compiled step
        kernel.  Individual mechanisms and nested States still own the actual
        runtime-noise declarations and in-place sampling logic.
        """

        any_sampled = False
        for mech in self.mechanisms.values():
            if hasattr(mech, "sample_runtime_noises_"):
                any_sampled = (
                    bool(
                        mech.sample_runtime_noises_(
                            *names,
                            dt=dt,
                            phase=phase,
                            step_index=step_index,
                            force=force,
                        )
                    )
                    or any_sampled
                )
        for process in self.material_processes.values():
            if hasattr(process, "sample_runtime_noises_"):
                any_sampled = (
                    bool(
                        process.sample_runtime_noises_(
                            *names,
                            dt=dt,
                            phase=phase,
                            step_index=step_index,
                            force=force,
                        )
                    )
                    or any_sampled
                )
        return any_sampled

    def resample_runtime_noise(self, *names, dt=None, phase=None):
        """Explicitly resample detached runtime ``NOISE`` buffers."""

        self.sample_runtime_noises_(*names, dt=dt, phase=phase, force=True)
        return self

    def ion_init(self, temp) -> None:
        for ion in self.ions.values():
            ion.initialize(temp)

    def material_init(self, temp=None) -> None:
        for material in self.materials.values():
            material.initialize(temp)

    def _get_material(self, name):
        """Return a generic material or an ion used through USEMATERIAL."""
        if name in self.materials:
            return self.materials[name]
        if name in self.ions:
            return self.ions[name]
        raise KeyError(
            f"Unknown material {name!r}. Available materials: "
            f"{list(self.materials.keys())}; ions usable as materials: {list(self.ions.keys())}."
        )

    def write_to_materials(self, v):
        for material, material_write in self.write_material.items():
            material_h = self._get_material(material)
            for k, field_list in material_write.items():
                mech = self.mechanisms[k]
                for field in field_list:
                    field_u = mech._buffers[field]
                    field_u = mech.put(field_u, material_h._buffers[field], v)
                    material_h._buffers[field] = field_u

        for material, material_source in self.source_material.items():
            material_h = self._get_material(material)
            for k, field_map in material_source.items():
                mech = self.mechanisms[k]
                for field, local_name in field_map.items():
                    source_u = mech._buffers[local_name]
                    zeros = torch.zeros_like(material_h._buffers[field])
                    source_u = mech.put(source_u, zeros, v)
                    material_h._buffers[field] = material_h._buffers[field] + source_u

    def read_from_materials(self):
        for material, material_read in self.read_material.items():
            material_h = self._get_material(material)
            for k, field_list in material_read.items():
                mech = self.mechanisms[k]
                for field in field_list:
                    field_value = mech.get(material_h._buffers[field])
                    mech._buffers[field] = field_value
                    for s in mech.DE.values():
                        s._buffers[field] = field_value

    def _ordered_material_process_names(self):
        items = tuple(self.material_processes.items())
        ordered = []
        seen = set()
        for phase in _MATERIAL_PHASE_ORDER:
            for name, process in items:
                process_phase = _canonical_material_phase(
                    getattr(type(process), "_material_process_phase", "post_local")
                )
                if process_phase == phase:
                    ordered.append(name)
                    seen.add(name)
        # Preserve insertion order for custom/unrecognized phases.  This keeps
        # PHASE extensible without silently dropping a process from the scheduler.
        for name, _process in items:
            if name not in seen:
                ordered.append(name)
        return tuple(ordered)

    def advance_material_processes(self, dt):
        for process_name in self._material_process_order:
            self.material_processes[process_name].advance_materials(dt)

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

        # write ion concentrations and generic material fields
        self.write_to_ions(v)
        self.write_to_materials(v)

        # run full-field material processes such as diffusion
        self.advance_material_processes(dt)

        # update equilibrium potentials and generic material guards/derived fields
        for ion in self.ions.values():
            ion.advance(temp)
        for material in self.materials.values():
            material.advance(temp)

        # read ion concentrations/equilibria and generic material fields
        self.read_from_ions()
        self.read_from_materials()

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
        for process in self.material_processes.values():
            process.detach()
        for ion in self.ions.values():
            ion.detach()
        for material in self.materials.values():
            material.detach()
        self.detach_i_g_bufs()

    def i(self, v):
        for mech in self.mechanisms.values():
            mech.breakpoint(mech.get(v))

        if not self.currents:
            z = torch.zeros_like(v)
            return z, z

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
        for mech in self.mechanisms.values():
            mech.breakpoint(mech.get(v))

        if not self.currents:
            return torch.zeros_like(v)

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
        for mech in self.mechanisms.values():
            mech.breakpoint(mech.get(v))

        if not self.currents:
            z = torch.zeros_like(v)
            return z, z

        v_half = 0.5 * v_prev

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
            return torch.zeros_like(v)

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

        for material, dict_of_mech_and_fields in self.write_material.items():
            for mech, fields in dict_of_mech_and_fields.items():
                m = self.mechanisms[mech]
                material_h = self._get_material(material)
                for field in fields:
                    q = m.get(material_h._buffers[field])
                    setattr(m, field, q.clone())
                    for _, s in self.mechanisms[mech].DE.items():
                        setattr(s, field, getattr(m, field))

        for material, dict_of_mech_and_sources in self.source_material.items():
            for mech, field_map in dict_of_mech_and_sources.items():
                m = self.mechanisms[mech]
                material_h = self._get_material(material)
                for field, local_name in field_map.items():
                    q = m.get(material_h._buffers[field])
                    setattr(m, local_name, torch.zeros_like(q))
                    for _, s in self.mechanisms[mech].DE.items():
                        setattr(s, local_name, getattr(m, local_name))

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
        for process in self.material_processes.values():
            states.extend(process.states())
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
        for process_name, process in self.material_processes.items():
            for rng_name in process._rng:
                states[f"{process_name}.{rng_name}"] = getattr(
                    process, rng_name
                ).rng_state()
            for _, state in process.DE.items():
                for state_name in state._state:
                    states[f"{process_name}.{state_name}"] = getattr(
                        process, state_name
                    )
                for rng_name in state._rng:
                    states[f"{process_name}.{state_name}.{rng_name}"] = getattr(
                        state, rng_name
                    ).rng_state()
            for buffer_name in process._assigned:
                states[f"{process_name}.{buffer_name}"] = process._buffers[buffer_name]
        for ion, ion_read in self.read_ion.items():
            for k, conc_list in ion_read.items():
                mech = self.mechanisms[k]
                for conc in conc_list:
                    states[f"{k}.{conc}"] = mech._buffers[conc]
        for ion_name, ion in self.ions.items():
            for buffer_name in getattr(
                ion, "fields", tuple(dict(ion.named_buffers()).keys())
            ):
                states[f"{ion_name}_ion.{buffer_name}"] = ion._buffers[buffer_name]
        for material_name, material in self.materials.items():
            for buffer_name in getattr(
                material, "fields", tuple(dict(material.named_buffers()).keys())
            ):
                states[f"{material_name}_material.{buffer_name}"] = material._buffers[
                    buffer_name
                ]
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
        for process_name, process in self.material_processes.items():
            for rng_name in process._rng:
                key = f"{process_name}.{rng_name}"
                getattr(process, rng_name).set_rng_state(state_dict[key])
            for _, state in process.DE.items():
                for state_name in state._state:
                    key = f"{process_name}.{state_name}"
                    setattr(process, state_name, state_dict[key])
                for rng_name in state._rng:
                    key = f"{process_name}.{state_name}.{rng_name}"
                    getattr(state, rng_name).set_rng_state(state_dict[key])
            for buffer_name in process._assigned:
                key = f"{process_name}.{buffer_name}"
                setattr(process, buffer_name, state_dict[key])
        for ion, ion_read in self.read_ion.items():
            for k, conc_list in ion_read.items():
                mech = self.mechanisms[k]
                for conc in conc_list:
                    key = f"{k}.{conc}"
                    setattr(mech, conc, state_dict[key])
                    for s in mech.DE.values():
                        setattr(s, conc, state_dict[key])
        for ion_name, ion in self.ions.items():
            for buffer_name in getattr(
                ion, "fields", tuple(dict(ion.named_buffers()).keys())
            ):
                key = f"{ion_name}_ion.{buffer_name}"
                setattr(ion, buffer_name, state_dict[key])
        for material_name, material in self.materials.items():
            for buffer_name in getattr(
                material, "fields", tuple(dict(material.named_buffers()).keys())
            ):
                key = f"{material_name}_material.{buffer_name}"
                setattr(material, buffer_name, state_dict[key])
