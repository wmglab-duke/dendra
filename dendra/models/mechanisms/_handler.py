import copy
from collections.abc import Mapping

import torch
import torch._dynamo as dynamo
import torch._inductor.config as inductor_config

from ..rng import _validate_rng_checkpoint_payload
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
_DELAY_SPEC_STATE_SUFFIX = ".__dendra_delayed_state_specs__"
_STOCHASTIC_STATE_SEGMENT = ".__dendra_stochastic_state__."


def _stochastic_rng_names(module):
    """Return every RNG stream whose position affects future simulation state."""

    names = set(getattr(module.__class__, "_rng", {}))
    for collection_name in ("random_parameters", "runtime_noises"):
        for spec in getattr(module, collection_name, {}).values():
            names.add(spec.effective_rng_name)
    names.update(getattr(module, "_sde_rng_names", ()))
    return tuple(sorted(names))


def _capture_stochastic_state(states, prefix, module):
    """Add RNG positions and live stochastic samples to ``states``."""

    for rng_name in _stochastic_rng_names(module):
        rng = getattr(module, rng_name, None)
        if rng is not None and hasattr(rng, "rng_state"):
            key = f"{prefix}{_STOCHASTIC_STATE_SEGMENT}rng.{rng_name}"
            states[key] = {
                "base_seed": getattr(rng, "_base_seed", None),
                "rng_state": rng.rng_state(),
            }
    for noise_name in sorted(getattr(module, "runtime_noises", {})):
        key = f"{prefix}{_STOCHASTIC_STATE_SEGMENT}runtime_noise.{noise_name}"
        states[key] = getattr(module, noise_name)
    for parameter_name in sorted(getattr(module, "random_parameters", {})):
        key = f"{prefix}{_STOCHASTIC_STATE_SEGMENT}random_parameter.{parameter_name}"
        states[key] = getattr(module, parameter_name)


def _integer_metadata(value, *, label, minimum=0):
    """Validate an integer-valued delayed-state metadata field."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer.")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return int(value)


def _delay_buffer_shape(spec):
    value_shape = tuple(spec["value_shape"])
    axis = int(spec["axis"])
    return value_shape[:axis] + (int(spec["depth"]),) + value_shape[axis:]


def _validate_delayed_state_specs(module, payload, *, label):
    """Validate saved delay metadata without changing the live registration."""

    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} delayed-state metadata must be a mapping.")

    live_specs = module._delayed_state_specs
    saved_names = set(payload)
    live_names = set(live_specs)
    if saved_names != live_names:
        raise ValueError(
            f"{label} delayed-state names do not match the live registration: "
            f"saved={sorted(saved_names)}, live={sorted(live_names)}."
        )

    validated = {}
    for name in sorted(live_specs):
        saved = payload[name]
        live = live_specs[name]
        item_label = f"{label} delayed state {name!r}"
        if not isinstance(saved, Mapping):
            raise TypeError(f"{item_label} metadata must be a mapping.")
        if set(saved) != set(live):
            raise ValueError(
                f"{item_label} metadata fields do not match the live "
                f"registration: saved={sorted(saved)}, live={sorted(live)}."
            )

        batched = bool(live.get("batched", False))
        if "batched" in saved and not isinstance(saved["batched"], bool):
            raise TypeError(f"{item_label} field 'batched' must be a bool.")
        if bool(saved.get("batched", False)) != batched:
            raise ValueError(f"{item_label} changed its batched registration kind.")

        name_fields = ["buffer", "pointer"]
        if batched:
            name_fields.append("steps_buffer")
        for field in name_fields:
            if not isinstance(saved[field], str):
                raise TypeError(f"{item_label} field {field!r} must be a string.")
            if saved[field] != live[field]:
                raise ValueError(
                    f"{item_label} field {field!r} must remain {live[field]!r}; "
                    f"got {saved[field]!r}."
                )

        steps = _integer_metadata(saved["steps"], label=f"{item_label} steps")
        depth = _integer_metadata(
            saved["depth"], label=f"{item_label} depth", minimum=1
        )
        if depth != max(1, steps + 1):
            raise ValueError(
                f"{item_label} depth {depth} is inconsistent with {steps} steps."
            )

        value_shape = saved["value_shape"]
        if not isinstance(value_shape, (tuple, list)):
            raise TypeError(f"{item_label} value_shape must be a tuple or list.")
        value_shape = tuple(
            _integer_metadata(size, label=f"{item_label} value_shape[{index}]")
            for index, size in enumerate(value_shape)
        )
        if value_shape != tuple(live["value_shape"]):
            raise ValueError(
                f"{item_label} value_shape {value_shape} does not match the live "
                f"payload shape {tuple(live['value_shape'])}."
            )

        axis = _integer_metadata(saved["axis"], label=f"{item_label} axis")
        if axis > len(value_shape):
            raise ValueError(
                f"{item_label} axis {axis} is invalid for payload shape {value_shape}."
            )
        if axis != int(live["axis"]):
            raise ValueError(
                f"{item_label} axis {axis} does not match the live axis "
                f"{int(live['axis'])}."
            )

        mode = saved["mode"]
        if not isinstance(mode, str):
            raise TypeError(f"{item_label} mode must be a string.")
        if mode not in {"auto", "shift", "circular"}:
            raise ValueError(f"{item_label} has invalid mode {mode!r}.")
        if mode != live["mode"]:
            raise ValueError(
                f"{item_label} mode {mode!r} does not match the live mode "
                f"{live['mode']!r}."
            )

        if batched:
            n_streams = _integer_metadata(
                saved["n_streams"], label=f"{item_label} n_streams", minimum=1
            )
            if n_streams != int(live["n_streams"]):
                raise ValueError(
                    f"{item_label} n_streams {n_streams} does not match the live "
                    f"stream count {int(live['n_streams'])}."
                )
            for field in ("stream_axis", "value_stream_axis"):
                restored_axis = _integer_metadata(
                    saved[field], label=f"{item_label} {field}"
                )
                if restored_axis != int(live[field]):
                    raise ValueError(
                        f"{item_label} {field} {restored_axis} does not match "
                        f"the live value {int(live[field])}."
                    )
            for field in ("has_zero_delay", "all_zero_delay"):
                if not isinstance(saved[field], bool):
                    raise TypeError(f"{item_label} field {field!r} must be a bool.")

        restored = copy.deepcopy(dict(saved))
        restored["steps"] = steps
        restored["depth"] = depth
        restored["axis"] = axis
        restored["value_shape"] = value_shape
        validated[name] = restored
    return validated


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
                    factorable = mech._current_factorable.get(
                        ion, bool(getattr(mech, "factorable", False))
                    )
                    self._map.append(
                        (c_idx, mech, f"{ion}_with_g", scale_f, factorable)
                    )
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
                    if field in self.write_material.get(material, {}).get(k, ()):
                        field_value = field_value.clone()
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
        for c_idx, mech, fn, scale_f, _factorable in self._map:
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
            if self.update_ion_buf.get(ion, False):
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

        for c_idx, mech, fn, scale_f, factorable in self._map:
            if factorable:
                v_in = v_half
            else:
                v_in = v
            i, g = scale_f(*getattr(mech, fn)(mech.get(v_in)))
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
            if self.update_ion_buf.get(ion, False):
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

    def _material_local_buffer_names(self):
        """Return mechanism-local material buffers that cross step boundaries."""
        names = {}
        for material_map in (self.read_material, self.write_material):
            for mechanism_map in material_map.values():
                for mech_name, fields in mechanism_map.items():
                    names.setdefault(mech_name, set()).update(fields)
        for mechanism_map in self.source_material.values():
            for mech_name, field_map in mechanism_map.items():
                names.setdefault(mech_name, set()).update(field_map.values())
        return {
            mech_name: tuple(sorted(buffer_names))
            for mech_name, buffer_names in names.items()
        }

    def mutable_state_dict(self):
        """
        Return a dictionary of all mutable / rebound states in the MechanismHandler.
        """
        states = {}
        for mech_name, mech in self.mechanisms.items():
            _capture_stochastic_state(states, mech_name, mech)
            for state_key, state in mech.DE.items():
                for state_name in sorted(state._state):
                    states[f"{mech_name}.{state_name}"] = getattr(mech, state_name)
                _capture_stochastic_state(states, f"{mech_name}.DE.{state_key}", state)
            for buffer_name in sorted(mech._assigned):
                states[f"{mech_name}.{buffer_name}"] = mech._buffers[buffer_name]
            for saved_name in sorted(mech._save):
                buffer_name = f"{saved_name}_"
                states[f"{mech_name}.{buffer_name}"] = mech._buffers[buffer_name]
            # Delayed-state queues and ring pointers are registered dynamically,
            # so they are not necessarily present in Mechanism._assigned.
            for name in sorted(mech._delayed_state_specs):
                spec = mech._delayed_state_specs[name]
                for spec_key in ("buffer", "pointer", "steps_buffer"):
                    buffer_name = spec.get(spec_key)
                    if buffer_name is not None and buffer_name in mech._buffers:
                        states[f"{mech_name}.{buffer_name}"] = mech._buffers[
                            buffer_name
                        ]
            if mech._delayed_state_specs:
                states[f"{mech_name}{_DELAY_SPEC_STATE_SUFFIX}"] = copy.deepcopy(
                    mech._delayed_state_specs
                )
        for process_name, process in self.material_processes.items():
            _capture_stochastic_state(states, process_name, process)
            for state_key, state in process.DE.items():
                for state_name in sorted(state._state):
                    states[f"{process_name}.{state_name}"] = getattr(
                        process, state_name
                    )
                _capture_stochastic_state(
                    states, f"{process_name}.DE.{state_key}", state
                )
            for buffer_name in sorted(process._assigned):
                states[f"{process_name}.{buffer_name}"] = process._buffers[buffer_name]
            for saved_name in sorted(process._save):
                buffer_name = f"{saved_name}_"
                states[f"{process_name}.{buffer_name}"] = process._buffers[buffer_name]
            for name in sorted(process._delayed_state_specs):
                spec = process._delayed_state_specs[name]
                for spec_key in ("buffer", "pointer", "steps_buffer"):
                    buffer_name = spec.get(spec_key)
                    if buffer_name is not None and buffer_name in process._buffers:
                        states[f"{process_name}.{buffer_name}"] = process._buffers[
                            buffer_name
                        ]
            if process._delayed_state_specs:
                states[f"{process_name}{_DELAY_SPEC_STATE_SUFFIX}"] = copy.deepcopy(
                    process._delayed_state_specs
                )
        for ion, ion_read in self.read_ion.items():
            for k, conc_list in ion_read.items():
                mech = self.mechanisms[k]
                for conc in conc_list:
                    states[f"{k}.{conc}"] = mech._buffers[conc]
        for ion, ion_write in self.write_ion_c.items():
            for k, conc_list in ion_write.items():
                mech = self.mechanisms[k]
                for conc in conc_list:
                    states[f"{k}.{conc}"] = mech._buffers[conc]
        for mech_name, buffer_names in self._material_local_buffer_names().items():
            mech = self.mechanisms[mech_name]
            for buffer_name in buffer_names:
                states[f"{mech_name}.{buffer_name}"] = mech._buffers[buffer_name]
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
        # Capture an actual boundary snapshot: later in-place simulation updates
        # must not mutate the saved tensors. Preserve graph connectivity and
        # aliases by cloning once per distinct tensor object.
        memo = {}
        snapshot = {}
        for name, value in states.items():
            if not torch.is_tensor(value):
                snapshot[name] = value
                continue
            value_id = id(value)
            if value_id not in memo:
                memo[value_id] = value.clone()
            snapshot[name] = memo[value_id]
        return snapshot

    def restore_mutable_state_dict(self, state_dict):
        """Restore a mutable-state snapshot as a two-phase transaction.

        Every tensor and delayed-state registration is validated and converted
        for the live module before the first mutation.  A late backend/RNG
        failure restores the original tensor objects, not merely equal clones.
        """

        plan = self._preflight_mutable_state_dict(state_dict)
        assignments = plan["assignments"]
        rng_assignments = plan["rng_assignments"]
        delay_assignments = plan["delay_assignments"]

        missing = object()
        original_tensors = [
            (owner, name, getattr(owner, name, missing))
            for owner, name, _ in assignments
        ]
        original_delays = [
            (module, module._delayed_state_specs) for module, _ in delay_assignments
        ]
        original_rngs = {}
        for rng, _, _ in rng_assignments:
            if id(rng) in original_rngs:
                continue
            original_rngs[id(rng)] = (
                rng,
                rng._base_seed,
                rng.rng_state(),
                dict(rng._device_gens),
                getattr(rng, "rng", missing),
            )

        try:
            for rng, base_seed, rng_state in rng_assignments:
                if base_seed is not None:
                    rng.reseed(base_seed)
                rng.set_rng_state(rng_state)
            for owner, name, value in assignments:
                setattr(owner, name, value)
            for module, specs in delay_assignments:
                module._delayed_state_specs = specs
        except Exception:
            for owner, name, original in reversed(original_tensors):
                if original is missing:
                    if hasattr(owner, name):
                        delattr(owner, name)
                else:
                    setattr(owner, name, original)
            for module, specs in original_delays:
                module._delayed_state_specs = specs
            for (
                rng,
                base_seed,
                rng_state,
                device_gens,
                active_rng,
            ) in original_rngs.values():
                rng._device_gens = device_gens
                rng._base_seed = base_seed
                rng.set_rng_state(rng_state)
                if active_rng is not missing:
                    rng.rng = active_rng
            raise

    def _preflight_mutable_state_dict(self, state_dict):
        """Build a fully validated, non-mutating mutable-state restore plan."""

        if not isinstance(state_dict, Mapping):
            raise TypeError("MechanismHandler mutable state must be a mapping.")

        assignment_map = {}
        tensor_memo = {}
        rng_assignments = []
        delay_assignments = []
        missing = object()

        def add_tensor(
            owner,
            name,
            key,
            *,
            expected_shape=None,
            optional=False,
            reference=None,
        ):
            if key not in state_dict:
                if optional:
                    return
                raise KeyError(f"MechanismHandler mutable state is missing {key!r}.")
            candidate = state_dict[key]
            if not torch.is_tensor(candidate):
                raise TypeError(
                    f"MechanismHandler mutable state {key!r} must be a tensor."
                )

            live = getattr(owner, name, missing) if reference is None else reference
            if live is missing or not torch.is_tensor(live):
                raise TypeError(
                    f"Live mutable field {key!r} must be a tensor before restore."
                )
            shape = (
                tuple(live.shape) if expected_shape is None else tuple(expected_shape)
            )
            if tuple(candidate.shape) != shape:
                raise ValueError(
                    f"MechanismHandler mutable state {key!r} has shape "
                    f"{tuple(candidate.shape)}, expected {shape}."
                )
            if candidate.layout != live.layout:
                raise ValueError(
                    f"MechanismHandler mutable state {key!r} has layout "
                    f"{candidate.layout}, expected {live.layout}."
                )

            memo_key = (id(candidate), live.device, live.dtype, live.layout)
            if memo_key not in tensor_memo:
                tensor_memo[memo_key] = candidate.to(
                    device=live.device, dtype=live.dtype
                ).clone()
            prepared = tensor_memo[memo_key]
            target = (id(owner), name)
            existing = assignment_map.get(target)
            if existing is not None and existing[2] is not prepared:
                raise ValueError(
                    f"Mutable field {name!r} received conflicting snapshot entries."
                )
            assignment_map[target] = (owner, name, prepared)

        def add_stochastic(module, prefix, legacy_prefixes):
            for rng_name in _stochastic_rng_names(module):
                canonical_key = f"{prefix}{_STOCHASTIC_STATE_SEGMENT}rng.{rng_name}"
                payload_key = canonical_key if canonical_key in state_dict else None
                if payload_key is None:
                    for legacy_prefix in legacy_prefixes:
                        legacy_key = f"{legacy_prefix}.{rng_name}"
                        if legacy_key in state_dict:
                            payload_key = legacy_key
                            break
                if payload_key is None:
                    raise KeyError(
                        "MechanismHandler mutable state is missing RNG stream "
                        f"{rng_name!r} for {prefix!r}."
                    )
                base_seed, rng_state = _validate_rng_checkpoint_payload(
                    state_dict[payload_key], allow_legacy=True
                )
                rng_assignments.append(
                    (getattr(module, rng_name), base_seed, rng_state)
                )

            for noise_name in sorted(getattr(module, "runtime_noises", {})):
                key = f"{prefix}{_STOCHASTIC_STATE_SEGMENT}runtime_noise.{noise_name}"
                add_tensor(module, noise_name, key)
            for parameter_name in sorted(getattr(module, "random_parameters", {})):
                key = (
                    f"{prefix}{_STOCHASTIC_STATE_SEGMENT}"
                    f"random_parameter.{parameter_name}"
                )
                add_tensor(module, parameter_name, key)

        def require_integer_tensor(key, expected_shape):
            if key not in state_dict:
                raise KeyError(f"MechanismHandler mutable state is missing {key!r}.")
            value = state_dict[key]
            if not torch.is_tensor(value):
                raise TypeError(
                    f"MechanismHandler mutable state {key!r} must be a tensor."
                )
            if tuple(value.shape) != tuple(expected_shape):
                raise ValueError(
                    f"MechanismHandler mutable state {key!r} has shape "
                    f"{tuple(value.shape)}, expected {tuple(expected_shape)}."
                )
            if (
                value.dtype == torch.bool
                or value.is_floating_point()
                or value.is_complex()
            ):
                raise TypeError(
                    f"MechanismHandler mutable state {key!r} must have integer dtype."
                )
            return value.detach().cpu().to(dtype=torch.long)

        def add_delayed_state(module, prefix):
            specs_key = f"{prefix}{_DELAY_SPEC_STATE_SUFFIX}"
            if specs_key not in state_dict:
                if module._delayed_state_specs:
                    raise KeyError(
                        "MechanismHandler mutable state is missing delayed-state "
                        f"metadata {specs_key!r}."
                    )
                return
            specs = _validate_delayed_state_specs(
                module, state_dict[specs_key], label=prefix
            )
            delay_assignments.append((module, specs))
            for name in sorted(specs):
                spec = specs[name]
                buffer_name = spec["buffer"]
                buffer_key = f"{prefix}.{buffer_name}"
                add_tensor(
                    module,
                    buffer_name,
                    buffer_key,
                    expected_shape=_delay_buffer_shape(spec),
                )

                pointer_name = spec["pointer"]
                pointer_key = f"{prefix}.{pointer_name}"
                pointer = require_integer_tensor(pointer_key, ())
                pointer_value = int(pointer.item())
                if pointer_value < 0 or pointer_value >= int(spec["depth"]):
                    raise ValueError(
                        f"MechanismHandler mutable state {pointer_key!r} must be "
                        f"between 0 and {int(spec['depth']) - 1}."
                    )
                add_tensor(module, pointer_name, pointer_key, expected_shape=())

                if spec.get("batched", False):
                    steps_name = spec["steps_buffer"]
                    steps_key = f"{prefix}.{steps_name}"
                    steps = require_integer_tensor(steps_key, (int(spec["n_streams"]),))
                    if bool(torch.any(steps < 0).item()):
                        raise ValueError(
                            f"MechanismHandler mutable state {steps_key!r} must "
                            "contain non-negative delays."
                        )
                    max_steps = int(steps.max().item()) if steps.numel() else 0
                    has_zero = bool(torch.any(steps == 0).item())
                    all_zero = bool(torch.all(steps == 0).item())
                    if max_steps != int(spec["steps"]):
                        raise ValueError(
                            f"{prefix} delayed state {name!r} metadata says "
                            f"steps={int(spec['steps'])}, but {steps_key!r} has "
                            f"maximum {max_steps}."
                        )
                    if int(spec["depth"]) != max(1, max_steps + 1):
                        raise ValueError(
                            f"{prefix} delayed state {name!r} depth is inconsistent "
                            f"with {steps_key!r}."
                        )
                    if bool(spec["has_zero_delay"]) != has_zero:
                        raise ValueError(
                            f"{prefix} delayed state {name!r} has_zero_delay is "
                            f"inconsistent with {steps_key!r}."
                        )
                    if bool(spec["all_zero_delay"]) != all_zero:
                        raise ValueError(
                            f"{prefix} delayed state {name!r} all_zero_delay is "
                            f"inconsistent with {steps_key!r}."
                        )
                    add_tensor(
                        module,
                        steps_name,
                        steps_key,
                        expected_shape=(int(spec["n_streams"]),),
                    )

        def add_module(module, prefix):
            add_stochastic(module, prefix, (prefix,))
            for state_key, state in module.DE.items():
                state_names = tuple(sorted(state._state))
                for state_name in state_names:
                    add_tensor(module, state_name, f"{prefix}.{state_name}")
                add_stochastic(
                    state,
                    f"{prefix}.DE.{state_key}",
                    tuple(f"{prefix}.{name}" for name in state_names),
                )
            for buffer_name in sorted(module._assigned):
                add_tensor(module, buffer_name, f"{prefix}.{buffer_name}")
            for saved_name in sorted(module._save):
                buffer_name = f"{saved_name}_"
                add_tensor(module, buffer_name, f"{prefix}.{buffer_name}")
            add_delayed_state(module, prefix)

        for mech_name, mech in self.mechanisms.items():
            add_module(mech, mech_name)
        for process_name, process in self.material_processes.items():
            add_module(process, process_name)

        for ion_read in self.read_ion.values():
            for mech_name, concentrations in ion_read.items():
                mech = self.mechanisms[mech_name]
                for concentration in concentrations:
                    key = f"{mech_name}.{concentration}"
                    add_tensor(mech, concentration, key)
                    for state in mech.DE.values():
                        add_tensor(state, concentration, key)
        for ion_write in self.write_ion_c.values():
            for mech_name, concentrations in ion_write.items():
                mech = self.mechanisms[mech_name]
                for concentration in concentrations:
                    key = f"{mech_name}.{concentration}"
                    add_tensor(mech, concentration, key)
                    for state in mech.DE.values():
                        state_value = getattr(state, concentration, missing)
                        add_tensor(
                            state,
                            concentration,
                            key,
                            reference=(
                                mech._buffers[concentration]
                                if state_value is missing
                                else state_value
                            ),
                        )
        for mech_name, buffer_names in self._material_local_buffer_names().items():
            mech = self.mechanisms[mech_name]
            for buffer_name in buffer_names:
                key = f"{mech_name}.{buffer_name}"
                add_tensor(mech, buffer_name, key)
                for state in mech.DE.values():
                    add_tensor(state, buffer_name, key)
        for ion_name, ion in self.ions.items():
            for buffer_name in getattr(
                ion, "fields", tuple(dict(ion.named_buffers()).keys())
            ):
                add_tensor(ion, buffer_name, f"{ion_name}_ion.{buffer_name}")
        for material_name, material in self.materials.items():
            for buffer_name in getattr(
                material, "fields", tuple(dict(material.named_buffers()).keys())
            ):
                add_tensor(
                    material,
                    buffer_name,
                    f"{material_name}_material.{buffer_name}",
                )

        return {
            "assignments": list(assignment_map.values()),
            "rng_assignments": rng_assignments,
            "delay_assignments": delay_assignments,
        }
