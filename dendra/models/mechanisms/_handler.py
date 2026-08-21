import copy
import itertools
from collections.abc import Mapping

import torch
import torch._dynamo as dynamo
import torch._inductor.config as inductor_config

from ..rng import _validate_rng_checkpoint_payload
from ._material_process import DiffusionProcess, MaterialProcess
from ._materials import _canonical_material_name
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
_SUPPORT_IDENTITY_STATE_KEY = "__dendra_mechanism_support_identity__"
_SUPPORT_IDENTITY_VERSION = 1


def _support_spec(module):
    """Return the structural support description owned by ``module``."""

    support_map = getattr(module, "support_map", None)
    spec = getattr(support_map, "spec", None)
    if spec is not None:
        return spec
    return getattr(module, "support_spec", None)


def _capture_support_identity_entry(module):
    """Capture immutable support metadata plus an exact packed-key fallback."""

    spec = _support_spec(module)
    if spec is None:
        # Directly constructed legacy Mechanisms can lack enough source-shape
        # information for a SupportSpec. Preserve their checkpoint behavior
        # without claiming an identity that Dendra cannot prove.
        return None

    flat_key = None
    if not spec.has_compact_identity:
        runtime_key = getattr(module, "key", None)
        if not torch.is_tensor(runtime_key):
            raise TypeError(
                f"Packed support for {module.name!r} must have a tensor key."
            )
        flat_key = runtime_key.detach().reshape(-1).clone()
    elif getattr(module, "key", None) is not None:
        spec.validate_runtime_key(
            module.key,
            context=f"Mechanism {module.name!r} support",
        )

    return {
        "metadata": copy.deepcopy(spec.checkpoint_identity()),
        "flat_key": flat_key,
    }


def _plain_metadata_equal(candidate, expected):
    """Compare checkpoint metadata without invoking tensor truth conversion."""

    if isinstance(expected, tuple):
        return (
            isinstance(candidate, tuple)
            and len(candidate) == len(expected)
            and all(
                _plain_metadata_equal(left, right)
                for left, right in zip(candidate, expected)
            )
        )
    if isinstance(expected, Mapping):
        return (
            isinstance(candidate, Mapping)
            and set(candidate) == set(expected)
            and all(
                _plain_metadata_equal(candidate[key], expected[key]) for key in expected
            )
        )
    if expected is None:
        return candidate is None
    if type(candidate) is not type(expected):
        return False
    return candidate == expected


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
    """Return the mechanism-to-density conversion function.

    Ordinary ``Mechanism`` values already use current density in mA/cm² and
    conductance density in S/cm², so their scaler is the identity. A
    ``PointProcess`` instead supplies lumped current in nA and conductance in
    µS. Dividing either by ``1e6 * area_cm2`` converts it to the corresponding
    mA/cm² or S/cm² density. Zero-area locations are invalid for point
    processes.

    The returned function yields one value for one input and a tuple for
    multiple inputs.
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


def _same_current_support(left, right):
    """Return whether two mechanisms have the same ordered physical support.

    Current aggregation may share a gathered voltage only when the two local
    tensors have exactly the same layout.  In particular, fancy indices are
    compared in order and with duplicates intact; set-equivalent selectors are
    not interchangeable because mechanism state and parameters follow local
    selector order.
    """

    if (
        left.base_ndim != right.base_ndim
        or left.shape_f != right.shape_f
        or left.is_composable != right.is_composable
    ):
        return False

    left_map = getattr(left, "support_map", None)
    right_map = getattr(right, "support_map", None)
    if left_map is not None and right_map is not None:
        if not left_map.spec.has_compact_identity and (
            not getattr(left, "_support_key_values_valid", True)
            or not getattr(right, "_support_key_values_valid", True)
        ):
            # Packed support identity depends on selector values. After a meta
            # move followed by ``to_empty``, those buffers are uninitialized;
            # keep mechanisms separate until state loading restores the keys.
            return left is right
        return left_map.same_ordered_support(
            right_map,
            left.key,
            right.key,
        )

    left_key = left.key
    right_key = right.key
    if left_key is None or right_key is None:
        return left_key is None and right_key is None

    if left.is_composable:
        if len(left_key) != len(right_key):
            return False
        for left_part, right_part in zip(left_key, right_key):
            if isinstance(left_part, slice) and isinstance(right_part, slice):
                if (
                    left_part.start,
                    left_part.stop,
                    left_part.step,
                ) != (
                    right_part.start,
                    right_part.stop,
                    right_part.step,
                ):
                    return False
            elif left_part != right_part:
                return False
        return True

    # Meta tensors intentionally carry shape/device metadata but no index
    # values, so ordered equality cannot be established.  Conservatively keep
    # distinct fancy mechanisms in separate groups until they reach a device
    # with materialized keys.
    if left_key.device.type == "meta" or right_key.device.type == "meta":
        return left is right

    return (
        left_key.shape == right_key.shape
        and left_key.dtype == right_key.dtype
        and left_key.device == right_key.device
        and torch.equal(left_key, right_key)
    )


def _support_scatter_is_unambiguous(mech):
    """Return whether local summation preserves this support's scatter order.

    A fancy selector may deliberately contain duplicate physical indices (for
    example, colocated point-process copies).  Sharing its voltage gather is
    safe, but combining multiple mechanisms before ``scatter_add_`` changes the
    reduction association across duplicate slots.  Keep those scatters
    separate and aggregate only unique-index, slice, or global supports.
    """

    support_map = getattr(mech, "support_map", None)
    if support_map is not None:
        return support_map.is_injective(mech.key)
    if mech.key is None or mech.is_composable:
        return True
    if mech.key.device.type == "meta":
        return False
    return torch.unique(mech.key).numel() == mech.key.numel()


def _current_value_for_buffer(value, buffer):
    """Match destination casting without touching already-compatible tensors."""

    if (
        torch.is_tensor(value)
        and value.device == buffer.device
        and value.dtype == buffer.dtype
    ):
        return value
    return torch.as_tensor(value, device=buffer.device, dtype=buffer.dtype)


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
        self._ion_current_reads = tuple(
            (ion, f"i{ion}", mech_name)
            for ion, mechanism_reads in self.read_ion.items()
            for mech_name, fields in mechanism_reads.items()
            if f"i{ion}" in fields
        )
        self.current_read_ions = tuple(
            sorted({ion for ion, _, _ in self._ion_current_reads})
        )
        self._ion_current_reader_names = tuple(
            dict.fromkeys(mech_name for _, _, mech_name in self._ion_current_reads)
        )
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

        # Safe defaults for directly constructed handlers. ``make_maps``
        # replaces these with dependency-specialized tuples during normal
        # Population initialization.
        self._pre_current_mechanism_names = tuple(self.mechanisms)
        self._current_evaluation_mechanisms = tuple(self.mechanisms.values())
        self._ion_current_frame_ions = tuple(sorted(self.ions))
        current_names = tuple(self.currents)
        self._ion_current_frame_indices = tuple(
            current_names.index(f"i{ion}") if f"i{ion}" in current_names else -1
            for ion in self._ion_current_frame_ions
        )

        for process in self.material_processes.values():
            process.bind_materials(self._get_material, population=population)

        self._validate_diffusion_process_overlaps()

        self._material_process_order = self._ordered_material_process_names()

        # --- flattened mapping (current-index, mechanism-obj, fn) ------------
        self._map = []
        self._map_exp = []
        self._current_support_representatives = ()
        self._current_breakpoint_plan = ()
        self._map_grouped = ()
        self._map_exp_grouped = ()
        self._ion_support_representatives = ()
        self._ion_source_breakpoint_plan = ()
        self._map_exp_ion_reads_grouped = ()
        self._total_support_representatives = ()
        self._map_exp_total_grouped = ()

        # Preserve breakpoint behavior for directly constructed handlers even
        # before ``make_maps()`` or full Population initialization.
        (
            self._current_support_representatives,
            self._current_breakpoint_plan,
            _support_by_mechanism,
        ) = self._partition_current_supports(self._current_evaluation_mechanisms)
        self._make_voltage_support_plans()

        self.ion_to_buff_idx = {}
        self.i_g_buffers_initialized = False

        self.update_ion_buf = {}

        self.shape = None

        # Selector keys are persistent for backward compatibility, but they are
        # structural configuration rather than transferable weights. Validate
        # them before any child state is loaded, then rebuild derived maps after
        # a compatible load completes.
        self.register_load_state_dict_pre_hook(self._validate_support_keys_before_load)
        self.register_load_state_dict_post_hook(self._refresh_maps_after_load)

        self._install_sync_wrappers()

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

    _SYNC_METHODS = (
        "write_to_ions",
        "read_from_ions",
        "write_to_materials",
        "read_from_materials",
    )

    def _install_sync_wrappers(self):
        """Bind conditional Dynamo barriers to this exact handler instance."""

        # ``dynamo.disable(bound_method)`` returns an instance function that
        # closes over that bound method. Never preserve such wrappers across a
        # deepcopy/pickle boundary: they would continue mutating the source.
        for method_name in self._SYNC_METHODS:
            self.__dict__.pop(method_name, None)
        if self.write_ion_c:
            self.write_to_ions = dynamo.disable(self.write_to_ions)
        if self.read_ion:
            self.read_from_ions = dynamo.disable(self.read_from_ions)
        if self.write_material or self.source_material:
            self.write_to_materials = dynamo.disable(self.write_to_materials)
        if self.read_material:
            self.read_from_materials = dynamo.disable(self.read_from_materials)

    def __getstate__(self):
        """Serialize configuration without source-bound Dynamo wrappers."""

        state = super().__getstate__()
        for method_name in self._SYNC_METHODS:
            state.pop(method_name, None)
        return state

    def __setstate__(self, state):
        """Restore synchronization wrappers bound to the restored handler."""

        for method_name in self._SYNC_METHODS:
            state.pop(method_name, None)
        super().__setstate__(state)
        self._install_sync_wrappers()

    @staticmethod
    def _refresh_maps_after_load(module, incompatible_keys):
        del incompatible_keys
        module.make_maps()

    @staticmethod
    def _validate_support_keys_before_load(
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Reject state dictionaries that try to move mechanism placement."""

        del local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        collections = (
            ("mechanisms", module.mechanisms),
            ("material_processes", module.material_processes),
        )
        for collection_name, mechanisms in collections:
            for name, mechanism in mechanisms.items():
                spec = _support_spec(mechanism)
                if spec is None or getattr(mechanism, "key", None) is None:
                    continue
                key_names = (
                    f"{prefix}{collection_name}.{name}.key",
                    f"{prefix}{name}.key",
                )
                for key_name in key_names:
                    if key_name not in state_dict:
                        continue
                    try:
                        spec.validate_runtime_key(
                            state_dict[key_name],
                            context=(f"State-dictionary mechanism {name!r} support"),
                        )
                    except (TypeError, ValueError, RuntimeError) as exc:
                        raise ValueError(
                            f"Cannot load {key_name!r}: selector keys encode "
                            "mechanism placement and must match the target "
                            "Population exactly."
                        ) from exc

    def _apply(self, fn, recurse=True):
        """Move registered state plus current scratch and scaling closures."""
        result = super()._apply(fn, recurse=recurse)

        # Current aggregation scratch is intentionally kept out of state_dict,
        # but Module._apply therefore cannot discover it. Preserve an
        # initialized handler across Population.float()/double()/to() by moving
        # these tensors explicitly.
        if getattr(self, "i_g_buffers_initialized", False):
            self._buf_i = [fn(buffer) for buffer in self._buf_i]
            self._buf_g = [fn(buffer) for buffer in self._buf_g]

        # Point-process density scalers close over an area-derived tensor.
        # Rebuild the maps after area/mechanism conversion so no callable keeps
        # a source-device or source-dtype tensor alive.
        if hasattr(self, "_map"):
            self.make_maps()
        return result

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

        current_names = tuple(self.currents)
        current_name_to_index = {
            current_name: current_index
            for current_index, current_name in enumerate(current_names)
        }
        self._ion_current_frame_ions = tuple(sorted(self.ions))
        self._ion_current_frame_indices = tuple(
            current_name_to_index.get(f"i{ion}", -1)
            for ion in self._ion_current_frame_ions
        )
        self._ion_current_indices = tuple(
            current_name_to_index[f"i{ion}"]
            for ion in self.current_read_ions
            if f"i{ion}" in current_name_to_index
        )
        current_index_set = set(self._ion_current_indices)
        self._map_exp_ion_reads = tuple(
            entry for entry in self._map_exp if entry[0] in current_index_set
        )
        source_mechanisms = []
        for _, mech, _, _ in self._map_exp_ion_reads:
            if all(mech is not source for source in source_mechanisms):
                source_mechanisms.append(mech)
        self._ion_current_sources = tuple(source_mechanisms)

        reader_names = set(self._ion_current_reader_names)
        current_source_names = {
            mech_name
            for current_map in self.currents.values()
            for mech_name in current_map
        }
        cyclic = sorted(reader_names.intersection(current_source_names))
        if cyclic:
            raise ValueError(
                "Mechanisms that READ iion cannot also contribute membrane "
                f"current in the same scheduler phase: {cyclic}. Split current "
                "generation and current-driven state into separate mechanisms."
            )
        current_reading_voltage_processes = sorted(
            name for name in reader_names if name in self.voltage_processes
        )
        if current_reading_voltage_processes:
            raise ValueError(
                "VoltageProcess mechanisms cannot READ iion because voltage "
                "processes run before current evaluation: "
                f"{current_reading_voltage_processes}."
            )

        self._pre_current_mechanism_names = tuple(
            name for name in self.mechanisms if name not in reader_names
        )
        self._current_evaluation_mechanisms = tuple(
            self.mechanisms[name] for name in self._pre_current_mechanism_names
        )
        self._make_voltage_support_plans()
        self._make_grouped_current_plans()

    @staticmethod
    def _partition_current_supports(mechanisms):
        """Build exact ordered-support groups for a mechanism sequence."""

        representatives = []
        mechanism_plan = []
        support_by_mechanism = {}
        for mech in mechanisms:
            support_index = next(
                (
                    index
                    for index, representative in enumerate(representatives)
                    if _same_current_support(mech, representative)
                ),
                None,
            )
            if support_index is None:
                support_index = len(representatives)
                representatives.append(mech)
            mechanism_plan.append((mech, support_index))
            support_by_mechanism[id(mech)] = support_index
        return (
            tuple(representatives),
            tuple(mechanism_plan),
            support_by_mechanism,
        )

    def _make_voltage_support_plans(self):
        """Plan gather-once voltage views for state and initialization phases."""

        all_mechanisms = tuple(self.mechanisms.values())
        (
            self._state_support_representatives,
            self._state_advance_plan,
            state_support_by_mechanism,
        ) = self._partition_current_supports(all_mechanisms)
        self._state_support_by_mechanism = state_support_by_mechanism

        pre_current = tuple(
            self.mechanisms[name]
            for name in self._pre_current_mechanism_names
            if name in self.mechanisms
        )
        (
            self._pre_state_support_representatives,
            self._pre_state_advance_plan,
            _pre_support_by_mechanism,
        ) = self._partition_current_supports(pre_current)

        current_readers = tuple(
            self.mechanisms[name]
            for name in self._ion_current_reader_names
            if name in self.mechanisms
        )
        (
            self._post_state_support_representatives,
            self._post_state_advance_plan,
            _post_support_by_mechanism,
        ) = self._partition_current_supports(current_readers)
        self._state_reader_breakpoint_plan = tuple(
            (mech, state_support_by_mechanism[id(mech)]) for mech in current_readers
        )
        self._make_field_read_plans()

    @staticmethod
    def _gather_support_fields(field, representatives):
        """Gather one local tensor for each exact ordered support group."""

        return tuple(representative.get(field) for representative in representatives)

    def _make_field_read_plans(self):
        """Group shared Ion/Material field reads by exact mechanism support."""

        def append_group(groups, key, payload):
            if key not in groups:
                groups[key] = []
            groups[key].append(payload)

        ion_groups = {}
        for ion, mechanism_reads in self.read_ion.items():
            for mechanism_name, fields in mechanism_reads.items():
                mechanism = self.mechanisms[mechanism_name]
                support_index = self._state_support_by_mechanism[id(mechanism)]
                for field in fields:
                    append_group(
                        ion_groups,
                        (ion, field, support_index),
                        mechanism,
                    )
        self._ion_read_support_plan = tuple(
            (
                ion,
                field,
                self._state_support_representatives[support_index],
                tuple(mechanisms),
            )
            for (ion, field, support_index), mechanisms in ion_groups.items()
        )

        material_groups = {}
        for material, mechanism_reads in self.read_material.items():
            for mechanism_name, fields in mechanism_reads.items():
                mechanism = self.mechanisms[mechanism_name]
                support_index = self._state_support_by_mechanism[id(mechanism)]
                written_fields = self.write_material.get(material, {}).get(
                    mechanism_name, ()
                )
                for field in fields:
                    append_group(
                        material_groups,
                        (material, field, support_index),
                        (mechanism, field in written_fields),
                    )
        self._material_read_support_plan = tuple(
            (
                material,
                field,
                self._state_support_representatives[support_index],
                tuple(bindings),
            )
            for (material, field, support_index), bindings in material_groups.items()
        )

        ion_current_groups = {}
        for ion, field, mechanism_name in self._ion_current_reads:
            mechanism = self.mechanisms[mechanism_name]
            support_index = self._state_support_by_mechanism[id(mechanism)]
            append_group(
                ion_current_groups,
                (ion, field, support_index),
                mechanism,
            )
        self._ion_current_read_support_plan = tuple(
            (
                ion,
                field,
                self._state_support_representatives[support_index],
                tuple(mechanisms),
            )
            for (ion, field, support_index), mechanisms in ion_current_groups.items()
        )

    @staticmethod
    def _bind_mechanism_field(mechanism, field, value):
        """Bind one local shared field to a mechanism and all nested States."""

        mechanism._buffers[field] = value
        for state in mechanism.DE.values():
            state._buffers[field] = value

    @staticmethod
    def _group_current_map(entries, representatives, support_by_mechanism):
        """Fuse adjacent exact-support entries into local reduction runs.

        Runs remain in the original current-map order.  Only entries with the
        same destination current and exact support are combined, and duplicate
        fancy selectors retain one scatter per entry.
        """

        runs = []
        for entry in entries:
            current_index, mech, *payload = entry
            support_index = support_by_mechanism[id(mech)]
            can_extend = (
                bool(runs)
                and runs[-1][0] == current_index
                and runs[-1][1] == support_index
                and _support_scatter_is_unambiguous(representatives[support_index])
            )
            current_entry = (mech, *payload)
            if can_extend:
                runs[-1][3].append(current_entry)
            else:
                runs.append(
                    [
                        current_index,
                        support_index,
                        representatives[support_index],
                        [current_entry],
                    ]
                )
        return tuple(
            (current_index, support_index, representative, tuple(run_entries))
            for current_index, support_index, representative, run_entries in runs
        )

    def _make_grouped_current_plans(self):
        """Compile gather-once/scatter-once plans for every current API."""

        current_mechanisms = list(self._current_evaluation_mechanisms)
        current_ids = {id(mech) for mech in current_mechanisms}
        for _, mech, *_ in self._map:
            if id(mech) not in current_ids:
                current_mechanisms.append(mech)
                current_ids.add(id(mech))
        (
            self._current_support_representatives,
            current_plan,
            current_support_by_mechanism,
        ) = self._partition_current_supports(current_mechanisms)
        breakpoint_ids = {id(mech) for mech in self._current_evaluation_mechanisms}
        self._current_breakpoint_plan = tuple(
            (mech, support_index)
            for mech, support_index in current_plan
            if id(mech) in breakpoint_ids
        )
        self._map_grouped = self._group_current_map(
            self._map,
            self._current_support_representatives,
            current_support_by_mechanism,
        )
        self._map_exp_grouped = self._group_current_map(
            self._map_exp,
            self._current_support_representatives,
            current_support_by_mechanism,
        )

        ion_mechanisms = list(self._ion_current_sources)
        ion_ids = {id(mech) for mech in ion_mechanisms}
        for _, mech, *_ in self._map_exp_ion_reads:
            if id(mech) not in ion_ids:
                ion_mechanisms.append(mech)
                ion_ids.add(id(mech))
        (
            self._ion_support_representatives,
            ion_plan,
            ion_support_by_mechanism,
        ) = self._partition_current_supports(ion_mechanisms)
        source_ids = {id(mech) for mech in self._ion_current_sources}
        self._ion_source_breakpoint_plan = tuple(
            (mech, support_index)
            for mech, support_index in ion_plan
            if id(mech) in source_ids
        )
        self._map_exp_ion_reads_grouped = self._group_current_map(
            self._map_exp_ion_reads,
            self._ion_support_representatives,
            ion_support_by_mechanism,
        )

        total_mechanisms = []
        total_ids = set()
        for _, mech, *_ in self._map_exp:
            if id(mech) not in total_ids:
                total_mechanisms.append(mech)
                total_ids.add(id(mech))
        (
            self._total_support_representatives,
            _total_plan,
            total_support_by_mechanism,
        ) = self._partition_current_supports(total_mechanisms)
        self._map_exp_total_grouped = self._group_current_map(
            self._map_exp,
            self._total_support_representatives,
            total_support_by_mechanism,
        )

    def initialize(self, v, celsius, diameters, populate=True, random_generation=None):
        """Initialize shared fields and mechanisms as one ordered transaction.

        Ionic-current readers require a provisional current frame before their
        ``INITIAL`` blocks can run.  That frame is only a dependency input: the
        accepted frame is recomputed after replacement writes, concentration
        guards, reversal-potential updates, and shared-field reads are coherent.

        Material ``source`` declarations and MaterialProcess instances are
        step operators.  No simulated time has elapsed here, so neither is
        applied during initialization.
        """
        self._sync_celsius(celsius)
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
        if self._ion_current_reads:
            local_voltages = self._gather_support_fields(
                v, self._pre_state_support_representatives
            )
            for mech, support_index in self._pre_state_advance_plan:
                mech._init_buffers_s(local_voltages[support_index])
            self.i(v)
            initial_frame = self.capture_ion_current_frame()
            self._publish_ion_current_frame(initial_frame)
            local_voltages = self._gather_support_fields(
                v, self._post_state_support_representatives
            )
            for mech, support_index in self._post_state_advance_plan:
                local_v = local_voltages[support_index]
                mech._init_buffers_s(local_v)
                mech.breakpoint(local_v)
        else:
            self.compute_initial_conditions(v)
            if self.write_ion_c or self.write_material:
                # A BREAKPOINT block may prepare an absolute concentration or
                # material write.  Preserve that initialization contract before
                # committing replacements; the final evaluation below then
                # refreshes currents from the post-commit shared state.
                self.i(v)

        # Commit absolute mechanism-owned values exactly once.  Additive
        # sources are per-step increments and must not alter the t=0 state.
        self.write_to_ions(v)
        self.write_material_replacements(v)

        # Enforce guards and update derived fields (for example Nernst
        # potentials) after the only replacement commit.  There are no writes
        # after this phase that could undo those constraints.
        for ion in self.ions.values():
            ion.advance(celsius)
        for material in self.materials.values():
            material.advance(celsius)

        self.read_from_ions()
        self.read_from_materials()

        # The provisional frame, when one was needed, may have depended on
        # pre-commit concentrations or reversal potentials.  Re-evaluate once
        # from the coherent shared state and publish only this final frame as
        # the accepted t=0 ionic current.
        self.i(v)
        self._publish_ion_current_frame(self.capture_ion_current_frame())

        # BREAKPOINT is allowed to prepare local write buffers.  Rebind shared
        # fields after the final evaluation so those provisional next-step
        # values cannot leave read/write mechanisms out of sync at t=0.
        self.read_from_ions()
        self.read_from_materials()

    def _sync_celsius(self, celsius):
        """Rebind every local temperature view before parameter population.

        Population ``GLOBAL`` values are refreshed out of place during
        ``initialize()``.  Keeping the temperature object captured when a
        mechanism was first built can therefore retain an obsolete autograd
        graph even when shared storage makes its numerical value appear current.
        Q10 caches are rebuilt by ``populate()`` below, so refresh the handler,
        mechanisms, material processes, and nested State views first.
        """
        self._buffers["celsius"] = celsius
        for module in itertools.chain(
            self.mechanisms.values(), self.material_processes.values()
        ):
            local_celsius = module.get(celsius)
            module._buffers["celsius"] = local_celsius
            for state in module.DE.values():
                state._buffers["celsius"] = local_celsius

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
        requested = str(name)
        canonical = _canonical_material_name(requested)
        for candidate in dict.fromkeys((requested, canonical)):
            if candidate in self.materials:
                return self.materials[candidate]
            if candidate in self.ions:
                return self.ions[candidate]
        raise KeyError(
            f"Unknown material {requested!r}. Available materials: "
            f"{list(self.materials.keys())}; ions usable as materials: {list(self.ions.keys())}."
        )

    def write_material_replacements(self, v):
        """Commit absolute USEMATERIAL ``write`` values to shared fields."""

        for material, material_write in self.write_material.items():
            material_h = self._get_material(material)
            for k, field_list in material_write.items():
                mech = self.mechanisms[k]
                for field in field_list:
                    field_u = mech._buffers[field]
                    field_u = mech.put(field_u, material_h._buffers[field], v)
                    material_h._buffers[field] = field_u

    def add_material_sources(self, v):
        """Apply additive USEMATERIAL ``source`` increments for one step."""

        for material, material_source in self.source_material.items():
            material_h = self._get_material(material)
            for k, field_map in material_source.items():
                mech = self.mechanisms[k]
                for field, local_name in field_map.items():
                    source_u = mech._buffers[local_name]
                    zeros = torch.zeros_like(material_h._buffers[field])
                    source_u = mech.put(source_u, zeros, v)
                    material_h._buffers[field] = material_h._buffers[field] + source_u

    def write_to_materials(self, v):
        """Commit material replacements followed by additive step sources."""

        self.write_material_replacements(v)
        self.add_material_sources(v)

    def read_from_materials(self):
        for (
            material,
            field,
            representative,
            bindings,
        ) in self._material_read_support_plan:
            material_h = self._get_material(material)
            field_value = representative.get(material_h._buffers[field])
            for mechanism, clone in bindings:
                local_value = field_value.clone() if clone else field_value
                self._bind_mechanism_field(mechanism, field, local_value)

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

    def _validate_diffusion_process_overlaps(self):
        """Reject order-dependent overlapping spatial writes to one field."""
        writers = {}
        for process_name, process in self.material_processes.items():
            if not isinstance(process, DiffusionProcess):
                continue
            phase = _canonical_material_phase(
                getattr(type(process), "_material_process_phase", "transport")
            )
            support = process._buffers["_mp_node_mask"]
            targets = {
                (
                    phase,
                    _canonical_material_name(spec.material),
                    str(spec.field),
                )
                for spec in type(process)._diffusion_specs
            }
            for target in targets:
                for previous_name, previous_support in writers.get(target, ()):
                    if bool(torch.any(support & previous_support).item()):
                        _, material, field = target
                        raise ValueError(
                            "Overlapping DiffusionProcess instances "
                            f"{previous_name!r} and {process_name!r} both write "
                            f"{material}.{field}. Use one fused process or make "
                            "their compartment supports disjoint."
                        )
                writers.setdefault(target, []).append((process_name, support))
        return None

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
        for ion, field, representative, mechanisms in self._ion_read_support_plan:
            local_value = representative.get(self.ions[ion]._buffers[field])
            for mechanism in mechanisms:
                self._bind_mechanism_field(mechanism, field, local_value)

    def capture_ion_current_frame(self):
        """Return the latest per-ion currents as an ephemeral solver frame.

        Frame slots follow ``_ion_current_frame_ions`` and retain their
        autograd graphs. Current evaluation rebinds its scratch tensors, so a
        captured Runge--Kutta stage remains stable across later stages.
        """
        frame = []
        for ion, current_index in zip(
            self._ion_current_frame_ions, self._ion_current_frame_indices
        ):
            if current_index >= 0:
                frame.append(self._buf_i[current_index])
            else:
                frame.append(torch.zeros_like(self.ions[ion]._buffers[f"i{ion}"]))
        return tuple(frame)

    def capture_ion_conductance_frame(self):
        """Return per-ion conductances aligned with the latest current frame."""
        frame = []
        for ion, current_index in zip(
            self._ion_current_frame_ions, self._ion_current_frame_indices
        ):
            if current_index >= 0:
                frame.append(self._buf_g[current_index])
            else:
                frame.append(torch.zeros_like(self.ions[ion]._buffers[f"i{ion}"]))
        return tuple(frame)

    def _publish_ion_current_frame(self, current_frame):
        """Commit a solver-selected frame to shared ions and local readers."""
        if len(current_frame) != len(self._ion_current_frame_ions):
            raise ValueError(
                "Ion-current frame has "
                f"{len(current_frame)} slots; expected "
                f"{len(self._ion_current_frame_ions)}."
            )

        frame_by_ion = {}
        for ion, current in zip(self._ion_current_frame_ions, current_frame):
            current_field = f"i{ion}"
            self.ions[ion]._buffers[current_field] = current
            frame_by_ion[ion] = current

        self._bind_ion_current_fields(frame_by_ion)

    def _bind_ion_current_fields(self, frame_by_ion=None):
        """Bind each current frame once per exact reader support."""

        for (
            ion,
            field,
            representative,
            mechanisms,
        ) in self._ion_current_read_support_plan:
            source = (
                self.ions[ion]._buffers[field]
                if frame_by_ion is None
                else frame_by_ion[ion]
            )
            local_value = representative.get(source)
            for mechanism in mechanisms:
                self._bind_mechanism_field(mechanism, field, local_value)

    def _read_current_from_ions(self, v):
        """Refresh declared ionic-current reads before state advancement.

        Re-evaluate currents at the voltage and channel state presented to
        this advance. The shared ``i{ion}`` buffer may otherwise contain an
        intermediate solver-stage or diagnostic evaluation. Pull only the
        current fields here so existing concentration phase semantics remain
        unchanged.
        """
        if not self._ion_current_reads:
            return

        local_voltages = tuple(
            representative.get(v)
            for representative in self._ion_support_representatives
        )
        for source, support_index in self._ion_source_breakpoint_plan:
            source.breakpoint(local_voltages[support_index])

        for current_index in self._ion_current_indices:
            self._buf_i[current_index] = torch.zeros_like(self._buf_i[current_index])

        for (
            c_idx,
            support_index,
            representative,
            entries,
        ) in self._map_exp_ion_reads_grouped:
            local_current = None
            local_voltage = local_voltages[support_index]
            for mech, fn, scale_f in entries:
                current = scale_f(getattr(mech, fn)(local_voltage))
                current = _current_value_for_buffer(current, self._buf_i[c_idx])
                local_current = (
                    current if local_current is None else local_current + current
                )
            representative.add_(self._buf_i[c_idx], local_current)

        for ion, current_field, _ in self._ion_current_reads:
            if self.update_ion_buf.get(ion, False):
                buffer_index = self.ion_to_buff_idx[ion]
                self.ions[ion]._buffers[current_field] = self._buf_i[buffer_index]
            else:
                self.ions[ion]._buffers[current_field] = torch.zeros_like(
                    self.ions[ion]._buffers[current_field]
                )

        self._bind_ion_current_fields()

    def _finish_advance(self, v, dt, temp):
        """Commit mechanism writes and refresh all shared derived state."""
        self.write_to_ions(v)
        self.write_to_materials(v)

        self.advance_material_processes(dt)

        for ion in self.ions.values():
            ion.advance(temp)
        for material in self.materials.values():
            material.advance(temp)

        self.read_from_ions()
        self.read_from_materials()

    def advance_pre_current(self, v, dt, temp):
        """Advance state without a declared dependency on this step's iion."""
        if not self._ion_current_reads:
            self.advance(v, dt, temp)
            return

        local_voltages = self._gather_support_fields(
            v, self._pre_state_support_representatives
        )
        for mech, support_index in self._pre_state_advance_plan:
            mech._advance(local_voltages[support_index], dt)

    def advance_post_current(self, v, dt, temp, current_frame):
        """Publish the accepted current, advance its readers, and commit state.

        Current readers receive the same state-update voltage used by the
        pre-current phase. The accepted frame supplies the solver's current
        quadrature (midpoint/RK-weighted/linearized endpoint). This split is
        exact for flux-driven states such as concentration dynamics; a state
        coupled independently to solver-stage voltage requires a dedicated
        coupled integrator.
        """
        if not self._ion_current_reads:
            self._publish_ion_current_frame(current_frame)
            return

        self._publish_ion_current_frame(current_frame)
        local_voltages = self._gather_support_fields(
            v, self._post_state_support_representatives
        )
        for mech, support_index in self._post_state_advance_plan:
            local_v = local_voltages[support_index]
            mech.breakpoint(local_v)
            mech._advance(local_v, dt)

        self._finish_advance(v, dt, temp)

    def advance(self, v, dt, temp):
        """Compatibility one-shot advance that refreshes declared iion reads.

        Stable integrators use ``advance_pre_current`` and
        ``advance_post_current`` so the solver-selected current frame is the
        only frame published to ions and current-reading mechanisms.
        """
        self._read_current_from_ions(v)

        local_voltages = self._gather_support_fields(
            v, self._state_support_representatives
        )
        for mech, support_index in self._state_reader_breakpoint_plan:
            mech.breakpoint(local_voltages[support_index])

        for mech, support_index in self._state_advance_plan:
            mech._advance(local_voltages[support_index], dt)

        self._finish_advance(v, dt, temp)

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

    # Current evaluators fill ephemeral scratch and preserve the normal
    # breakpoint/SAVE side effects, but do not overwrite committed shared Ion
    # or current-reader fields. ``advance_post_current`` publishes the frame
    # selected by the solver.
    def i(self, v):
        local_voltages = tuple(
            representative.get(v)
            for representative in self._current_support_representatives
        )
        for mech, support_index in self._current_breakpoint_plan:
            mech.breakpoint(local_voltages[support_index])

        if not self.currents:
            z = torch.zeros_like(v)
            return z, z

        # reset buffers
        for i, buf in enumerate(self._buf_i):
            self._buf_i[i] = torch.zeros_like(buf)
        for i, buf in enumerate(self._buf_g):
            self._buf_g[i] = torch.zeros_like(buf)

        # Evaluate every authored current in map order, but reduce adjacent
        # exact-support contributions locally before touching full-sized
        # population buffers.
        for c_idx, support_index, representative, entries in self._map_grouped:
            local_i = None
            local_g = None
            local_voltage = local_voltages[support_index]
            for mech, fn, scale_f, _factorable in entries:
                i, g = scale_f(*getattr(mech, fn)(local_voltage))
                i = _current_value_for_buffer(i, self._buf_i[c_idx])
                g = _current_value_for_buffer(g, self._buf_g[c_idx])
                local_i = i if local_i is None else local_i + i
                local_g = g if local_g is None else local_g + g
            representative.add_(self._buf_i[c_idx], local_i)
            representative.add_(self._buf_g[c_idx], local_g)

        # sum up currents and conductances
        tot_i = torch.stack(self._buf_i).sum(dim=0)
        tot_g = torch.stack(self._buf_g).sum(dim=0)

        return tot_i, tot_g

    def iexp(self, v):
        local_voltages = tuple(
            representative.get(v)
            for representative in self._current_support_representatives
        )
        for mech, support_index in self._current_breakpoint_plan:
            mech.breakpoint(local_voltages[support_index])

        if not self.currents:
            return torch.zeros_like(v)

        for i, buf in enumerate(self._buf_i):
            self._buf_i[i] = torch.zeros_like(buf)

        for c_idx, support_index, representative, entries in self._map_exp_grouped:
            local_i = None
            local_voltage = local_voltages[support_index]
            for mech, fn, scale_f in entries:
                i = scale_f(getattr(mech, fn)(local_voltage))
                i = _current_value_for_buffer(i, self._buf_i[c_idx])
                local_i = i if local_i is None else local_i + i
            representative.add_(self._buf_i[c_idx], local_i)

        # sum up currents and conductances
        tot_i = torch.stack(self._buf_i).sum(dim=0)

        return tot_i

    def idf(self, v, v_prev):
        local_voltages = tuple(
            representative.get(v)
            for representative in self._current_support_representatives
        )
        for mech, support_index in self._current_breakpoint_plan:
            mech.breakpoint(local_voltages[support_index])

        if not self.currents:
            z = torch.zeros_like(v)
            return z, z

        v_half = 0.5 * v_prev
        local_half_voltages = tuple(
            representative.get(v_half)
            for representative in self._current_support_representatives
        )

        # reset buffers
        for i, buf in enumerate(self._buf_i):
            self._buf_i[i] = torch.zeros_like(buf)
        for i, buf in enumerate(self._buf_g):
            self._buf_g[i] = torch.zeros_like(buf)

        for c_idx, support_index, representative, entries in self._map_grouped:
            local_i = None
            local_g = None
            for mech, fn, scale_f, factorable in entries:
                if factorable:
                    i, g = scale_f(
                        *getattr(mech, fn)(local_half_voltages[support_index])
                    )
                else:
                    current_fn = fn.removesuffix("_with_g")
                    raw_i = getattr(mech, current_fn)(local_voltages[support_index])
                    if current_fn in mech._save:
                        setattr(mech, f"{current_fn}_", raw_i)
                    i = scale_f(raw_i)
                    g = torch.zeros_like(i) if torch.is_tensor(i) else 0.0
                i = _current_value_for_buffer(i, self._buf_i[c_idx])
                g = _current_value_for_buffer(g, self._buf_g[c_idx])
                local_i = i if local_i is None else local_i + i
                local_g = g if local_g is None else local_g + g
            representative.add_(self._buf_i[c_idx], local_i)
            representative.add_(self._buf_g[c_idx], local_g)

        # sum up currents and conductances
        tot_i = torch.stack(self._buf_i).sum(dim=0)
        tot_g = torch.stack(self._buf_g).sum(dim=0)

        return tot_i, tot_g

    def itot(self, v):
        if not self.currents:
            return torch.zeros_like(v)

        for i, buf in enumerate(self._buf_i):
            self._buf_i[i] = torch.zeros_like(buf)

        local_voltages = tuple(
            representative.get(v)
            for representative in self._total_support_representatives
        )
        for (
            c_idx,
            support_index,
            representative,
            entries,
        ) in self._map_exp_total_grouped:
            local_i = None
            local_voltage = local_voltages[support_index]
            for mech, fn, scale_f in entries:
                i = scale_f(getattr(mech, fn)(local_voltage))
                i = _current_value_for_buffer(i, self._buf_i[c_idx])
                local_i = i if local_i is None else local_i + i
            representative.add_(self._buf_i[c_idx], local_i)

        return torch.stack(self._buf_i).sum(dim=0)

    def set_buffers(self, diameters):
        local_geometry = self._gather_support_fields(
            diameters, self._state_support_representatives
        )
        for m, support_index in self._state_advance_plan:
            # ``diameters`` is population-wide, while a restricted mechanism
            # owns only the compartments selected by its support.  Re-gather
            # the local geometry on every bind (including batched binds), and
            # keep each nested State on the exact same tensor as its parent.
            local_diameters = local_geometry[support_index].detach().clone()
            m.diam = local_diameters
            for state in m.DE.values():
                state.diam = local_diameters

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
        local_voltages = self._gather_support_fields(
            v, self._state_support_representatives
        )
        for mech, support_index in self._state_advance_plan:
            mech._init_buffers_s(local_voltages[support_index])

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

        Structural support is configuration rather than mutable state. It is
        nevertheless recorded so restore can reject a same-shaped checkpoint
        whose local slots refer to different physical compartments. Checkpoints
        created before support identity was added remain loadable.
        """
        states = {
            _SUPPORT_IDENTITY_STATE_KEY: {
                "version": _SUPPORT_IDENTITY_VERSION,
                "mechanisms": {
                    name: _capture_support_identity_entry(mechanism)
                    for name, mechanism in self.mechanisms.items()
                },
                "material_processes": {
                    name: _capture_support_identity_entry(process)
                    for name, process in self.material_processes.items()
                },
            }
        }
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

        self._validate_support_identity_payload(state_dict)

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

    def _validate_support_identity_payload(self, state_dict):
        """Reject checkpoints whose local slots address another support.

        This validation intentionally runs before any tensor, delay, or RNG
        assignment is prepared or applied. Absence of the reserved entry means
        the checkpoint predates support identity and retains the legacy
        shape-only compatibility rule.
        """

        if _SUPPORT_IDENTITY_STATE_KEY not in state_dict:
            return

        payload = state_dict[_SUPPORT_IDENTITY_STATE_KEY]
        if not isinstance(payload, Mapping):
            raise TypeError("Mechanism support identity must be a mapping.")

        required_fields = {"version", "mechanisms", "material_processes"}
        if set(payload) != required_fields:
            raise ValueError(
                "Mechanism support identity fields do not match the checkpoint "
                f"schema: expected {sorted(required_fields)}, got "
                f"{sorted(map(str, payload))}."
            )

        version = payload["version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise TypeError("Mechanism support identity version must be an integer.")
        if version != _SUPPORT_IDENTITY_VERSION:
            raise ValueError(
                "Unsupported Mechanism support identity version "
                f"{version}; expected {_SUPPORT_IDENTITY_VERSION}."
            )

        collections = (
            ("mechanisms", self.mechanisms),
            ("material_processes", self.material_processes),
        )
        for collection_name, live_modules in collections:
            saved_modules = payload[collection_name]
            if not isinstance(saved_modules, Mapping):
                raise TypeError(
                    f"Mechanism support identity {collection_name!r} must be a mapping."
                )
            if set(saved_modules) != set(live_modules):
                raise ValueError(
                    f"Mechanism support identity {collection_name!r} names do "
                    "not match the live model: "
                    f"saved={sorted(map(str, saved_modules))}, "
                    f"live={sorted(map(str, live_modules))}."
                )

            for name, module in live_modules.items():
                label = f"{collection_name} entry {name!r}"
                candidate = saved_modules[name]
                spec = _support_spec(module)
                if spec is None:
                    if candidate is not None:
                        raise ValueError(
                            f"Mechanism support identity {label} is unavailable "
                            "for the live legacy module."
                        )
                    continue

                if not isinstance(candidate, Mapping):
                    raise TypeError(
                        f"Mechanism support identity {label} must be a mapping."
                    )
                if set(candidate) != {"metadata", "flat_key"}:
                    raise ValueError(
                        f"Mechanism support identity {label} must contain exactly "
                        "'metadata' and 'flat_key'."
                    )
                if not _plain_metadata_equal(
                    candidate["metadata"], spec.checkpoint_identity()
                ):
                    raise ValueError(
                        f"Mechanism support identity mismatch for {label}; the "
                        "checkpoint addresses different physical slots or uses "
                        "a different local support layout."
                    )

                candidate_key = candidate["flat_key"]
                if spec.has_compact_identity:
                    if candidate_key is not None:
                        raise ValueError(
                            f"Mechanism support identity {label} must not contain "
                            "a packed flat key."
                        )
                    continue

                if not torch.is_tensor(candidate_key):
                    raise TypeError(
                        f"Mechanism support identity {label} flat_key must be a tensor."
                    )
                if (
                    candidate_key.dtype == torch.bool
                    or candidate_key.is_floating_point()
                    or candidate_key.is_complex()
                ):
                    raise TypeError(
                        f"Mechanism support identity {label} flat_key must have "
                        "integer dtype."
                    )
                if candidate_key.ndim != 1:
                    raise ValueError(
                        f"Mechanism support identity {label} flat_key must be "
                        "one-dimensional."
                    )

                live_key = getattr(module, "key", None)
                if not torch.is_tensor(live_key):
                    raise TypeError(
                        f"Live packed support for {label} must have a tensor key."
                    )
                live_key = live_key.detach().reshape(-1)
                if tuple(candidate_key.shape) != tuple(live_key.shape):
                    raise ValueError(
                        f"Mechanism support identity mismatch for {label}: saved "
                        f"flat-key shape {tuple(candidate_key.shape)}, live "
                        f"{tuple(live_key.shape)}."
                    )
                if (
                    candidate_key.device.type == "meta"
                    or live_key.device.type == "meta"
                ):
                    raise ValueError(
                        f"Mechanism support identity for packed {label} cannot be "
                        "validated on the meta device."
                    )
                prepared_key = candidate_key.to(
                    device=live_key.device, dtype=torch.long
                )
                if not torch.equal(prepared_key, live_key.to(dtype=torch.long)):
                    raise ValueError(
                        f"Mechanism support identity mismatch for {label}; the "
                        "checkpoint flat key addresses different physical slots."
                    )
