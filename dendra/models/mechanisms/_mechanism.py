import inspect
import textwrap
import warnings
from types import MethodType
from typing import Dict

import torch

from dendra.helpers import classproperty
from dendra.models._class_declarations import (
    _DECLARATIONS_KEY,
    consume_class_values,
    declare_class_value,
)
from dendra.models.parametric import Parameterized

from ._ions import VALENCES
from ._materials import _canonical_material_name
from ._state import State
from ._symbolic import build_current_eq


def _merge_list_dict(dst, src):
    for key, values in src.items():
        dst.setdefault(key, [])
        for value in values:
            if value not in dst[key]:
                dst[key].append(value)


def _merge_nested_dict(dst, src):
    for key, values in src.items():
        dst.setdefault(key, {})
        dst[key].update(values)


def _as_name_list(values):
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    return [str(v) for v in values]


def _normalize_material_source(source):
    if not source:
        return {}
    if isinstance(source, dict):
        return {str(field): str(local_name) for field, local_name in source.items()}
    if isinstance(source, str):
        return {source: f"{source}_source"}
    return {str(field): f"{field}_source" for field in source}


def _mro_attribute_owner(cls, name):
    """Return the class whose namespace supplies ``name`` under ``cls``'s MRO."""
    for base in cls.__mro__:
        if name in base.__dict__:
            return base
    return None


# -- TorchDynamo-friendly mechanism advance generation -----------------------


def _safe_generated_identifier(value: object) -> str:
    """Return a Python-identifier fragment suitable for generated helper names."""
    text = str(value)
    chars = [ch if (ch.isalnum() or ch == "_") else "_" for ch in text]
    out = "".join(chars).strip("_") or "mechanism"
    if out[0].isdigit():
        out = f"_{out}"
    return out


def _nonnegative_integer_steps(value, *, device=None, label="delay_steps"):
    """Normalize delay metadata without silently truncating fractional values."""
    try:
        raw = torch.as_tensor(value, device=device)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must contain non-negative integers.") from exc

    if raw.dtype == torch.bool or raw.is_complex():
        raise TypeError(f"{label} must contain non-negative integers.")
    if raw.is_floating_point():
        integral = torch.isfinite(raw) & (raw == torch.round(raw))
        if not bool(torch.all(integral).item()):
            raise ValueError(f"{label} must contain integer values.")

    steps = raw.to(dtype=torch.long)
    if bool(torch.any(steps < 0).item()):
        raise ValueError(f"{label} must contain non-negative integers.")
    return steps


def _mechanism_advance_signature(mech) -> tuple:
    """Static layout signature for a generated mechanism advance fast path."""
    state_layout = []
    for state_name, state_module in mech.DE.items():
        state_layout.append(
            (
                state_name,
                tuple(state_module._state),
                getattr(type(state_module), "advance", None) is State.advance,
                getattr(type(state_module), "breakpoint", None) is State.breakpoint,
                getattr(state_module, "method", None),
            )
        )
    return (tuple(state_layout), tuple(mech._all_states))


def _compile_monomorphic_mechanism_advance(mech, signature: tuple):
    """Compile a per-mechanism/per-proxy `_advance` implementation.

    The generated function deliberately avoids the shared base-class loop over
    ``self.DE.values()`` and uses literal state-module names / buffer keys.  This
    gives TorchDynamo a distinct code object for each generated mechanism proxy,
    avoiding cache churn from one polymorphic ``Mechanism._advance`` frame being
    called with many unrelated ``self`` types.

    State-module updates are evaluated from a single snapshot of mechanism
    buffers taken at the start of the timestep.  The returned locals are committed
    only after every state module has evaluated its breakpoint/solve call.  This
    avoids order-dependent Gauss-Seidel semantics across ``mech.DE`` entries and
    matches the usual ODE interpretation that all state bundles advance from
    time-n values to time-(n+1) values together.
    """
    cls = mech.__class__
    cls_name = _safe_generated_identifier(cls.__name__)
    mech_name = _safe_generated_identifier(getattr(mech, "name", cls.__name__))
    func_name = f"_advance_{cls_name}_{mech_name}_{id(cls):x}"

    lines = [
        f"def {func_name}(self, v, dt):",
        "    __buffers = self._buffers",
    ]

    if not mech.DE:
        lines.append("    return None")
    else:
        lines.append("    __DE = self.DE")
        all_state_names = tuple(mech._all_states)

        if all_state_names:
            lines.append(
                "    # Snapshot all mechanism states/assigned buffers before solving any bundle."
            )
            lines.append("    __states = {")
            for state_name in all_state_names:
                lines.append(f"        {state_name!r}: __buffers[{state_name!r}],")
            lines.append("    }")
        else:
            lines.append("    __states = {}")

        commit_lines = []
        for idx, (state_key, state_module) in enumerate(mech.DE.items()):
            state_var_names = tuple(state_module._state)
            uses_default_advance = (
                getattr(type(state_module), "advance", None) is State.advance
                and getattr(state_module, "method", None) != "euler_heun"
            )
            uses_default_breakpoint = (
                getattr(type(state_module), "breakpoint", None) is State.breakpoint
            )

            lines.extend(
                [
                    f"    # State bundle {idx}: {state_key}",
                    f"    __state_module_{idx} = __DE[{state_key!r}]",
                ]
            )

            if uses_default_advance:
                if uses_default_breakpoint:
                    lines.append(f"    __breakpoint_{idx} = {{}}")
                else:
                    lines.append(
                        f"    __breakpoint_{idx} = __state_module_{idx}.breakpoint(v, __states)"
                    )
                lines.append(
                    f"    __local_{idx} = __state_module_{idx}.solve(dt, **__breakpoint_{idx}, **__states)"
                )
                for state_var_name in state_var_names:
                    commit_lines.append(
                        f"    __buffers[{state_var_name!r}] = __local_{idx}[{state_var_name!r}]"
                    )
            else:
                # Preserve custom State.advance call semantics, but delay
                # committing its returned updates until all bundles have read the
                # start-of-step snapshot.  Any internal side effects performed by
                # a custom advance method remain that method's responsibility.
                lines.append(
                    f"    __local_{idx} = __state_module_{idx}.advance(v, dt, __states)"
                )
                commit_lines.append(f"    __buffers.update(__local_{idx})")

        if commit_lines:
            lines.append(
                "    # Commit all returned updates after every bundle has evaluated."
            )
            lines.extend(commit_lines)
        lines.append("    return None")

    source = "\n".join(lines) + "\n"
    filename = (
        f"<dendra.mechanism.advance.{cls.__module__}.{cls.__qualname__}.{id(cls):x}>"
    )
    namespace = {}
    exec(compile(source, filename, "exec"), {}, namespace)
    fn = namespace[func_name]
    fn.__name__ = "_advance"
    fn.__qualname__ = f"{cls.__qualname__}._advance"
    fn.__module__ = cls.__module__
    fn.__doc__ = (
        "Generated monomorphic mechanism state-advance fast path.  The source "
        "is stored on the owning class as `_dendra_monomorphic_advance_source`."
    )
    fn._dendra_monomorphic_advance = True
    fn._dendra_monomorphic_advance_signature = signature
    fn._dendra_monomorphic_advance_source = source
    return fn, source


class Mechanism(Parameterized):
    """
    Base class for Dendra mechanisms.

    Mechanisms encapsulate state variables, ionic currents, and parameter
    declarations that can be attached to neuronal morphologies. They extend
    :class:`dendra.models.parametric.Parameterized` to leverage the shared
    parameter declaration and population infrastructure.

    Use uppercase classmethods (``STATE``, ``GLOBAL``, ``RANGE``, ``BUFFER``,
    ``USEION``, ``NONSPECIFIC_CURRENT``, etc.) at class definition time to
    declare structure. Override lowercase hooks (``initial``, ``breakpoint``,
    current methods) to implement behavior.

    - ``STATE(StateSubclass, ...)``: register one or more State bundles. Each
      State subclass manages its own state variables and derivatives. These are
      accessible via the ``mechanism.DE`` ModuleDict.
    - ``GLOBAL/RANGE``: shared vs per-compartment parameters.
    - ``BUFFER``: mechanism-level buffers (analogous to State ``BUFFER``),
      typically set in ``initial``/``breakpoint``.
      ``ASSIGNED`` remains as a legacy alias for this declaration.
    - ``USEION``: ionic read/write dependencies.
    - ``NONSPECIFIC_CURRENT`` / current methods: contribute to membrane balance.
    - ``initial(self, v)``: one-time setup; set mechanism buffers, etc.
    - ``breakpoint(self, v)``: per-step computation of currents/buffer values.

    Notes
    -----
    An ordinary ``Mechanism`` is a distributed membrane mechanism. Its voltage
    and reversal potentials are in mV, each declared current method returns
    outward-positive current density in mA/cm², and the corresponding voltage
    derivative/conductance is in S/cm². Thus a density-style Ohmic current is
    written directly as ``g * (v - e)`` because
    ``S/cm² * mV = mA/cm²``. Dendra keeps these values as densities during
    mechanism assembly and applies compartment area in the voltage solver.

    Inherit from :class:`PointProcess` when a mechanism instead returns a
    lumped current in nA and uses conductance in µS. The conversion factors in
    :mod:`dendra.units` are plain scalars for documented public base units; do
    not form density units with expressions such as ``S / cm**2``.

    Subclasses declare state, mechanism-buffer, and ionic variables using the
    :meth:`STATE`, :meth:`BUFFER`, :meth:`SAVE`, :meth:`USEION`, and
    :meth:`NONSPECIFIC_CURRENT` helpers during class definition. Override
    :meth:`initial` and :meth:`breakpoint` to populate buffers and assemble
    currents each step. ``ASSIGNED`` is retained as a deprecated alias for
    :meth:`BUFFER` for compatibility with older mechanism definitions.
    """

    _state = set()
    _ion = set()
    _material = set()
    _save = set()
    _assigned = set()
    _explicit = set()
    _numerical = set()
    _affine = set()
    _affine_method_owners = {}

    _state_declarations = []
    _ion_declarations = []
    _material_declarations = []
    _save_declarations = []
    _assigned_declarations = []
    _explicit_declarations = []
    _numerical_declarations = []
    _affine_declarations = []

    _conductances = {}
    _currents = {}
    _init = {}

    _read_ion = {}
    _write_ion = {}
    _write_ion_c = {}

    _read_material = {}
    _write_material = {}
    _source_material = {}

    _conductances_declarations = []
    _currents_declarations = []
    _init_declarations = []

    _read_ion_declarations = []
    _write_ion_declarations = []
    _write_ion_c_declarations = []

    _read_material_declarations = []
    _write_material_declarations = []
    _source_material_declarations = []

    _renamed_aliases = {}

    def __init_subclass__(cls, **kwargs):
        """
        This special method is called automatically whenever a class
        inherits from Parameterized.
        """
        # Call the parent's __init_subclass__ WITHOUT our custom kwargs,
        # as the base 'object' class does not accept them.
        super().__init_subclass__(**kwargs)

        # Renamed aliases are cached per source mechanism class.  Sharing the
        # inherited dictionary would let an alias requested from one mechanism
        # satisfy a same-named request from an unrelated mechanism.
        cls._renamed_aliases = {}

        # Start with a fresh dictionary for the new class's parameters.
        new_state = set()
        new_ion = set()
        new_material = set()
        new_save = set()
        new_assigned = set()
        new_explicit = set()
        new_numerical = set()
        new_affine = set()
        inherited_affine_method_owners = {}

        new_read_ion = {}
        new_write_ion = {}
        new_write_ion_c = {}

        new_read_material = {}
        new_write_material = {}
        new_source_material = {}

        new_currents = {}
        new_init = {}

        # Walk MRO in reverse to build up params from parent to child
        for base in reversed(cls.__mro__):
            # We look for a _params attribute defined directly on the base
            if "_state" in base.__dict__:
                new_state.update(base._state)
            if "_ion" in base.__dict__:
                new_ion.update(base._ion)
            if "_material" in base.__dict__:
                new_material.update(base._material)
            if "_save" in base.__dict__:
                new_save.update(base._save)
            if "_assigned" in base.__dict__:
                new_assigned.update(base._assigned)
            if "_read_ion" in base.__dict__:
                new_read_ion.update(base._read_ion)
            if "_write_ion" in base.__dict__:
                new_write_ion.update(base._write_ion)
            if "_write_ion_c" in base.__dict__:
                new_write_ion_c.update(base._write_ion_c)
            if "_read_material" in base.__dict__:
                _merge_list_dict(new_read_material, base._read_material)
            if "_write_material" in base.__dict__:
                _merge_list_dict(new_write_material, base._write_material)
            if "_source_material" in base.__dict__:
                _merge_nested_dict(new_source_material, base._source_material)
            if "_currents" in base.__dict__:
                # Current declarations are list-valued.  Copy those lists so a
                # subclass declaration cannot append into its parent's class
                # metadata (notably when constructing a renamed mechanism).
                new_currents.update(
                    {name: list(currents) for name, currents in base._currents.items()}
                )
            if "_init" in base.__dict__:
                new_init.update(base._init)
            if "_explicit" in base.__dict__:
                new_explicit.update(base._explicit)
            if "_numerical" in base.__dict__:
                new_numerical.update(base._numerical)

        # Keep every inherited affine contract until the effective current
        # method has been selected by Python's MRO. A simple union of ``_affine``
        # is unsafe for multiple inheritance: a lower-priority base's assertion
        # must not apply to a different method supplied by a higher-priority
        # base.
        for base in cls.__mro__[1:]:
            if "_affine_method_owners" not in base.__dict__:
                continue
            for current, owner in base._affine_method_owners.items():
                inherited_affine_method_owners.setdefault(current, set()).add(owner)

        for s_list in consume_class_values(
            cls, "mechanism.state", Mechanism._state_declarations
        ):
            new_state.update(s_list)
        for i_list in consume_class_values(
            cls, "mechanism.ion", Mechanism._ion_declarations
        ):
            new_ion.update(i_list)
        for m_list in consume_class_values(
            cls, "mechanism.material", Mechanism._material_declarations
        ):
            new_material.update(m_list)
        for s_list in consume_class_values(
            cls, "mechanism.save", Mechanism._save_declarations
        ):
            new_save.update(s_list)
        for a_list in consume_class_values(
            cls, "mechanism.assigned", Mechanism._assigned_declarations
        ):
            new_assigned.update(a_list)
        for r_dict in consume_class_values(
            cls, "mechanism.read_ion", Mechanism._read_ion_declarations
        ):
            new_read_ion.update(r_dict)
        for w_dict in consume_class_values(
            cls, "mechanism.write_ion", Mechanism._write_ion_declarations
        ):
            new_write_ion.update(w_dict)
        for w_dict in consume_class_values(
            cls, "mechanism.write_ion_c", Mechanism._write_ion_c_declarations
        ):
            new_write_ion_c.update(w_dict)
        for r_dict in consume_class_values(
            cls,
            "mechanism.read_material",
            Mechanism._read_material_declarations,
        ):
            _merge_list_dict(new_read_material, r_dict)
        for w_dict in consume_class_values(
            cls,
            "mechanism.write_material",
            Mechanism._write_material_declarations,
        ):
            _merge_list_dict(new_write_material, w_dict)
        for s_dict in consume_class_values(
            cls,
            "mechanism.source_material",
            Mechanism._source_material_declarations,
        ):
            _merge_nested_dict(new_source_material, s_dict)
        for c_list in consume_class_values(
            cls, "mechanism.currents", Mechanism._currents_declarations
        ):
            new_currents.setdefault("nonspecific", []).extend(c_list)
        for i_dict in consume_class_values(
            cls, "mechanism.init", Mechanism._init_declarations
        ):
            new_init.update(i_dict)
        for v_list in consume_class_values(
            cls, "mechanism.explicit", Mechanism._explicit_declarations
        ):
            new_explicit.update(v_list)
        for v_list in consume_class_values(
            cls, "mechanism.numerical", Mechanism._numerical_declarations
        ):
            new_numerical.update(v_list)
        declared_affine = set()
        for v_list in consume_class_values(
            cls, "mechanism.affine", Mechanism._affine_declarations
        ):
            declared_affine.update(v_list)

        # An affine assertion belongs to the method selected when it was made.
        # Retain an inherited assertion only when that same method owner still
        # wins for the subclass. This handles both ordinary overrides and
        # competing multiple-inheritance branches without relying on function
        # object identity (an assertionless class may directly alias a base
        # function and still constitutes a new ownership boundary).
        new_affine_method_owners = {}
        if "_affine" in cls.__dict__:
            # Dynamic rename aliases copy the complete computed metadata into
            # their namespace. Rebind those exact metadata/method clones to the
            # owner selected in the alias's new MRO.
            for current in cls.__dict__["_affine"]:
                owner = _mro_attribute_owner(cls, current)
                new_affine.add(current)
                new_affine_method_owners[current] = owner
        else:
            for current, asserted_owners in inherited_affine_method_owners.items():
                owner = _mro_attribute_owner(cls, current)
                if owner in asserted_owners:
                    new_affine.add(current)
                    new_affine_method_owners[current] = owner

        for current in declared_affine:
            owner = _mro_attribute_owner(cls, current)
            new_affine.add(current)
            new_affine_method_owners[current] = owner

        cls.state_classes = {s.__name__: s for s in new_state}

        cls._state = new_state
        cls._ion = new_ion
        cls._material = new_material
        cls._save = new_save
        cls._currents = new_currents
        cls._assigned = new_assigned
        cls._read_ion = new_read_ion
        cls._write_ion = new_write_ion
        cls._write_ion_c = new_write_ion_c
        cls._read_material = new_read_material
        cls._write_material = new_write_material
        cls._source_material = new_source_material
        cls._init = new_init
        cls._explicit = new_explicit
        cls._numerical = new_numerical
        cls._affine = new_affine
        cls._affine_method_owners = new_affine_method_owners
        cls._name = None

    def __init__(
        self,
        name: str,
        celsius,
        diameters,
        shape,
        shape_f,
        key=None,
        is_composable=False,
        additional_parameters=None,
        ic: dict = None,
        **kwargs,
    ):
        """
        Initialize a mechanism instance and register declared buffers.

        Parameters
        ----------
        name : str
            Mechanism alias. If ``None``, falls back to the class name.
        celsius : Tensor or float
            Temperature in °C, broadcastable to the mechanism shape.
        diameters : Tensor
            Compartment diameters in µm, shared with sub-components.
        shape : tuple of int
            Base shape for parameter tensors (without batch dimensions).
        shape_f : tuple of int
            Full shape including batch dimensions.
        key : Any, optional
            Index selector identifying the attached compartments.
        is_composable : bool, optional
            Whether ``key`` represents a tuple of slices instead of flat indices.
        additional_parameters : dict, optional
            Additional parameter declarations injected by the parent population.
        ic : dict, optional
            Initial condition overrides for state buffers.
        **kwargs
            Extra keyword arguments forwarded to
            :class:`dendra.models.parametric.Parameterized`.
        """
        super().__init__(
            shape, shape_f, additional_parameters=additional_parameters, **kwargs
        )
        if name is None:
            name = self.__class__._name or self.__class__.__name__

        self.name = name

        self.register_buffer("celsius", celsius)
        self.base_ndim = 2

        self.register_buffer("dt", torch.tensor(0.0))

        # Mechanism-level waveform injection support.  The base "inject"
        # method is intentionally a no-op; mechanisms that want to consume
        # waveform stimuli can override it and call register_waveform_injection().
        # Waveforms are stored as submodules so their parameters move with the
        # mechanism and remain differentiable.
        self.injected_waveforms = torch.nn.ModuleList()
        self._injection_specs = []

        # Mechanism-level delayed-state support.  This is intended for fused
        # mechanisms that bypass Network/NetCon but still need fixed axonal or
        # state delays.  Each entry in _delayed_state_specs maps a user-facing
        # delay name to registered buffer/pointer names and update policy.
        self._delayed_state_specs = {}

        if key is not None:
            if is_composable:
                self.key = key
            else:
                self.register_buffer(
                    "key", torch.as_tensor(key, dtype=torch.long).detach().clone()
                )
        else:
            self.key = None

        self.is_composable = is_composable

        def get_fancy(tensor):
            if tensor.ndim == 0:
                # If tensor is scalar, return it directly
                return tensor
            # Preserves batch dimensions by only flattening the base dimensions
            batch_shape = tensor.shape[: -self.base_ndim]
            flat_tensor = tensor.reshape(*batch_shape, -1)
            # Select along the last dimension (the flattened base dimension)
            return flat_tensor.index_select(-1, self.key)

        def add_fancy_(tensor, what):
            # Use scatter_add_ for batched index_add_
            batch_shape = tensor.shape[: -self.base_ndim]
            flat_tensor = tensor.reshape(*batch_shape, -1)

            # Expand key to match batch dimensions for scatter
            # e.g., key shape [N] -> [B1, B2, ..., N]
            expanded_key = self.key.expand(*batch_shape, -1)

            what = torch.as_tensor(
                what,
                device=tensor.device,
                dtype=tensor.dtype,
            )
            what = what.expand_as(expanded_key)

            # what should have shape [B1, B2, ..., N]
            flat_tensor.scatter_add_(-1, expanded_key, what)
            return tensor  # Return original tensor for chaining

        def add_fancy(tensor, what):
            # Use scatter_add for batched index_add
            batch_shape = tensor.shape[: -self.base_ndim]
            flat_tensor = tensor.reshape(*batch_shape, -1)

            # Expand key to match batch dimensions for scatter
            expanded_key = self.key.expand(*batch_shape, -1)

            what = torch.as_tensor(
                what,
                device=tensor.device,
                dtype=tensor.dtype,
            )
            what = what.expand_as(expanded_key)

            # what should have shape [B1, B2, ..., N]
            return flat_tensor.scatter_add(-1, expanded_key, what).reshape_as(tensor)

        if self.key is None:
            self.get = lambda tensor: tensor
            self.add_ = lambda add_to, add_what: add_to.add_(add_what)
            self.add = lambda add_to, add_what: add_to.add(add_what)
            self.put = self.put_no_op
        elif self.is_composable:
            self.get = lambda tensor: (
                tensor[..., *self.key] if tensor.ndim > 0 else tensor
            )
            self.add_ = lambda add_to, add_what: add_to[..., *self.key].add_(add_what)
            self.add = lambda add_to, add_what: add_to[..., *self.key].add(add_what)
            self.put = self.put_slice
        else:
            self.get = get_fancy
            self.add_ = add_fancy_
            self.add = add_fancy
            self.put = self.put_fancy

        self.read_ion = self._read_ion
        self.write_ion_c = self._write_ion_c
        self.read_material = self._read_material
        self.write_material = self._write_material
        self.source_material = self._source_material

        states = [
            state(
                self.get(celsius),
                self.get(diameters),
                key,
                shape,
                shape_f,
                additional_parameters=additional_parameters,
                **kwargs,
            )
            for state in self._state
        ]

        self.DE: dict[str, State] = torch.nn.ModuleDict(
            {state._name: state for state in states}
        )

        self._init_params: Dict[str, float] = {k: v for k, v in self._init.items()}
        if ic is not None:
            self._init_params.update(ic)

        self.register_buffer("diam", self.get(diameters))
        for state in self.DE.values():
            for state_name in state._state:
                self.register_buffer(state_name, torch.zeros(shape))
                # getattr(self, state_name).requires_grad_(True)

        for r in self._save:
            self.register_buffer(f"{r}_", torch.zeros(shape))

        for a in self._assigned:
            self.register_buffer(a, torch.zeros(shape))

        self._all_states = []
        for state_module in self.DE.values():
            for state_name in state_module._state:
                self._all_states.append(state_name)

        self._all_states += [a for a in self._assigned]

        # factorize current equations
        self._current_factorable = {}
        self._current_conductance_mode = {}
        self._current_conductance_fallback_reason = {}
        current_eqs = []
        for _, v in self._currents.items():
            current_eqs.extend(v)
        for _, v in self._write_ion.items():
            current_eqs.extend(v)

        for k in current_eqs:
            assign = k in self._save
            eq, factorable = build_current_eq(self, k, assign=assign)
            setattr(self, f"{k}_with_g", MethodType(eq, self))
            setattr(getattr(self.__class__, k), "factorable", factorable)
            self._current_factorable[k] = bool(factorable)
            self._current_conductance_mode[k] = eq._dendra_conductance_mode
            self._current_conductance_fallback_reason[k] = (
                eq._dendra_conductance_fallback_reason
            )

        # Retain the legacy aggregate attribute for downstream callers while
        # keeping the per-current truth needed by mixed mechanisms.
        self.factorable = all(self._current_factorable.values())

        self.populate()
        self.instantiate_tables()
        for state in self.DE.values():
            state.instantiate_tables()
        self._install_monomorphic_advance()

    def set_dt(self, dt):
        """
        Update the per-mechanism time-step buffer.

        Parameters
        ----------
        dt : float or Tensor
            New integration step size in milliseconds.
        """
        self.dt = self.dt.fill_(dt).detach()

    def put_no_op(self, ion_conc_u, ion_conc_o, v, clone=True):
        """
        Return ionic concentrations unchanged.

        Parameters
        ----------
        ion_conc_u : Tensor
            Updated ionic concentrations (ignored).
        ion_conc_o : Tensor
            Original ionic concentrations.
        v : Tensor
            Voltage reference tensor (ignored).
        clone : bool, optional
            Unused for the no-op path.

        Returns
        -------
        Tensor
            ``ion_conc_o`` unchanged.
        """
        return ion_conc_u

    def put_slice_(self, ion_conc_u, ion_conc_o, v, clone=True):
        """
        Write ionic concentrations into a slice in place.

        Parameters
        ----------
        ion_conc_u : Tensor
            Updated ionic concentrations matching the slice length.
        ion_conc_o : Tensor
            Original ionic concentrations to be updated.
        v : Tensor
            Voltage tensor used for broadcasting shape.
        clone : bool, optional
            If True, operate on a cloned copy of ``ion_conc_o``.

        Returns
        -------
        Tensor
            Tensor with the slice replaced by ``ion_conc_u``.
        """
        ion_conc_o = ion_conc_o.expand_as(v)
        if clone:
            ion_conc_o = ion_conc_o.clone()
        ion_conc_o[..., *self.key] = ion_conc_u
        return ion_conc_o

    def put_slice(self, ion_conc_u, ion_conc_o, v, clone=True):
        """
        Write ionic concentrations into a slice and return a new tensor.

        Parameters
        ----------
        ion_conc_u : Tensor
            Updated ionic concentrations matching the slice length.
        ion_conc_o : Tensor
            Original ionic concentrations to be updated.
        v : Tensor
            Voltage tensor used for broadcasting shape.
        clone : bool, optional
            If True, operate on a cloned copy of ``ion_conc_o``.

        Returns
        -------
        Tensor
            Tensor with the slice replaced by ``ion_conc_u``.
        """
        # ion_conc_o is the full tensor, v is a reference for shape, ion_conc_u is the update
        ion_conc_o = ion_conc_o.expand_as(v)
        if clone:
            ion_conc_o = ion_conc_o.clone()

        # Apply the update using Ellipsis
        ion_conc_o[..., *self.key] = ion_conc_u
        return ion_conc_o

    def put_fancy_(self, ion_conc_u, ion_conc_o, v, clone=True):
        """
        Write ionic concentrations using flattened fancy indexing in place.

        Parameters
        ----------
        ion_conc_u : Tensor
            Updated ionic concentrations with length ``len(self.key)``.
        ion_conc_o : Tensor
            Original ionic concentrations to be updated.
        v : Tensor
            Voltage tensor used for broadcasting shape.
        clone : bool, optional
            If True, operate on a cloned copy of ``ion_conc_o``.

        Returns
        -------
        Tensor
            Tensor with entries replaced at ``self.key`` indices.
        """
        ion_conc_o = ion_conc_o.expand_as(v)
        if clone:
            ion_conc_o = ion_conc_o.clone()
        ion_conc_o.view(-1).index_put_((self.key,), ion_conc_u)
        return ion_conc_o

    def put_fancy(self, ion_conc_u, ion_conc_o, v, clone=True):
        """
        Write ionic concentrations using flattened fancy indexing.

        Parameters
        ----------
        ion_conc_u : Tensor
            Updated ionic concentrations with length ``len(self.key)``.
        ion_conc_o : Tensor
            Original ionic concentrations to be updated.
        v : Tensor
            Voltage tensor used for broadcasting shape.
        clone : bool, optional
            If True, operate on a cloned copy of ``ion_conc_o`` before writing.

        Returns
        -------
        Tensor
            Tensor with entries replaced at ``self.key`` indices.
        """
        # ion_conc_u: The new values to put, shape [..., len(key)]
        # ion_conc_o: The destination tensor, shape [..., *base_shape]
        # v: Reference tensor for shape

        ion_conc_o = ion_conc_o.expand_as(v)
        if clone:
            ion_conc_o = ion_conc_o.clone()

        # Get batch shape from the destination tensor
        batch_shape = ion_conc_o.shape[: -self.base_ndim]

        # Reshape destination to [B, S]
        flat_dest = ion_conc_o.view(*batch_shape, -1)

        # Expand key to match batch dimensions for scatter
        expanded_key = self.key.expand(*batch_shape, -1)

        # Use scatter to place the values from ion_conc_u into flat_dest
        # scatter_(dim, index, src)
        flat_dest.scatter_(-1, expanded_key, ion_conc_u)

        # The original ion_conc_o tensor is modified in place, so we can just return it
        return ion_conc_o

    def register_ion(self, ion):
        """
        Attach shared ion buffers used by the mechanism.

        Parameters
        ----------
        ion : Ion
            Ion descriptor exposing concentration and reversal potential tensors.
        """
        name = ion.name
        if name in self.read_ion:
            for v in self.read_ion[name]:
                q = getattr(ion, v)
                if self.key is not None and q.ndim > 0:
                    self.register_buffer(v, self.get(q))
                    for _, s in self.DE.items():
                        s.register_buffer(v, self.get(q))
                else:
                    self.register_buffer(v, q)
                    for _, s in self.DE.items():
                        s.register_buffer(v, q)

        if name in self.write_ion_c:
            for v in self.write_ion_c[name]:
                q = getattr(ion, v)
                if self.key is not None and q.ndim > 0:
                    qk = self.get(q)
                    self.register_buffer(
                        v, torch.empty(qk.shape, device=qk.device, dtype=qk.dtype)
                    )
                    getattr(self, v).copy_(qk)
                else:
                    self.register_buffer(v, q)

    def _set_local_material_buffer(self, name, value, *, expose_to_states=True):
        """Register or rebind a local material buffer on mechanism and states."""
        if name in self._buffers:
            self._buffers[name] = value
        else:
            self.register_buffer(name, value)
        if expose_to_states:
            for _, state_module in self.DE.items():
                if name in state_module._buffers:
                    state_module._buffers[name] = value
                else:
                    state_module.register_buffer(name, value)

    def _material_local_view(self, material, field):
        if not hasattr(material, "has_field"):
            raise TypeError(
                "register_material expects a Material-like object with has_field(field)."
            )
        if not material.has_field(field):
            raise ValueError(
                f"Material {material.name!r} has no field {field!r}. "
                f"Available fields: {getattr(material, 'fields', tuple(material._buffers.keys()))!r}."
            )
        q = material._buffers[field]
        if self.key is not None and q.ndim > 0:
            return self.get(q)
        return q

    def register_material(self, material):
        """Attach shared material buffers used by the mechanism.

        Material fields are population-wide.  Read bindings expose local views of
        the full material field to the mechanism and nested State modules.  Write
        bindings allocate local buffers initialized from the material field; the
        owning handler commits those buffers back to the full Material after the
        local mechanism phase.  Source bindings allocate additive increment
        buffers named ``<field>_source`` by default, or by the explicit local name
        supplied to ``USEMATERIAL(..., source={field: local_name})``.
        """
        name = _canonical_material_name(material.name)

        read_fields = []
        for declared_name, fields in self.read_material.items():
            if _canonical_material_name(declared_name) == name:
                read_fields.extend(
                    field for field in fields if field not in read_fields
                )
        for field in read_fields:
            self._set_local_material_buffer(
                field, self._material_local_view(material, field)
            )

        write_fields = []
        for declared_name, fields in self.write_material.items():
            if _canonical_material_name(declared_name) == name:
                write_fields.extend(
                    field for field in fields if field not in write_fields
                )
        for field in write_fields:
            local = self._material_local_view(material, field)
            # Writable material fields should be local tensors, not aliases,
            # so mechanism state updates do not mutate the population field
            # before the handler's commit phase.
            local = local.clone()
            self._set_local_material_buffer(field, local)

        source_fields = {}
        for declared_name, field_map in self.source_material.items():
            if _canonical_material_name(declared_name) == name:
                source_fields.update(field_map)
        for field, local_name in source_fields.items():
            local = self._material_local_view(material, field)
            self._set_local_material_buffer(local_name, torch.zeros_like(local))

    def _init_buffers_s(self, v_init):
        for state_module in self.DE.values():
            state_names = state_module._state
            for state_name in state_names:
                if state_name in self._init_params:
                    buffer_tensor = (
                        torch.as_tensor(
                            self._init_params[state_name],
                            device=v_init.device,
                            dtype=v_init.dtype,
                        )
                        .expand_as(v_init)
                        .clone()
                    )
                    setattr(self, state_name, buffer_tensor.detach())
                else:
                    if inf := state_module.inf(v_init):
                        buffer_tensor = inf[state_name]
                        setattr(self, state_name, buffer_tensor.detach())

        self.initial(v_init)

        for _, s in self.DE.items():
            s.initialize(v_init)

        return

    # Classmethod declarations
    @staticmethod
    def STATE(*args):
        """
        Declare state variables for the mechanism class body.

        Parameters
        ----------
        *args : type
            State module classes registered to ``Mechanism._state``.
        """
        declare_class_value("mechanism.state", args, Mechanism._state_declarations)

    @staticmethod
    def BUFFER(*args):
        """
        Declare mechanism-level buffers.

        These buffers are allocated per mechanism instance and are typically
        populated in :meth:`initial` or :meth:`breakpoint`. They are analogous
        to :meth:`State.BUFFER` rather than :meth:`State.ASSIGNED`: unlike a
        State ASSIGNED variable, a mechanism buffer does not have to be
        computed and returned from a state ``breakpoint`` function.

        Parameters
        ----------
        *args : str
            Names of mechanism buffers to allocate per instance.
        """
        declare_class_value(
            "mechanism.assigned", args, Mechanism._assigned_declarations
        )

    @staticmethod
    def ASSIGNED(*args):
        """
        Deprecated alias for :meth:`BUFFER`.

        Mechanism-level ``ASSIGNED`` historically declared mutable buffers.
        New code should use ``Mechanism.BUFFER(...)`` to avoid confusion with
        ``State.ASSIGNED(...)``, whose names must be computed by a State
        ``breakpoint`` function.
        """
        warnings.warn(
            "Mechanism.ASSIGNED(...) is deprecated; use Mechanism.BUFFER(...) "
            "for mechanism-level buffers. State.ASSIGNED(...) is unchanged.",
            DeprecationWarning,
            stacklevel=2,
        )
        Mechanism.BUFFER(*args)

    @staticmethod
    def SAVE(*args):
        """
        Declare state variables that must be saved each time step.

        Parameters
        ----------
        *args : str
            Names of buffers mirrored with a trailing underscore.
        """
        declare_class_value("mechanism.save", args, Mechanism._save_declarations)

    @staticmethod
    def USEION(ion, read=None, write=None):
        """
        Declare ionic read/write dependencies for the mechanism class body.

        Parameters
        ----------
        ion : str
            Ion species identifier (e.g., ``'na'``).
        read : Sequence[str], optional
            Ion variables to be read; must be among ``{ion}i``, ``{ion}o``,
            ``e{ion}``, or ``i{ion}`` (e.g., ``['nai', 'nao', 'ena']``).
        write : Sequence[str], optional
            Ion variables to be written; same allowed set (e.g., ``['ina']`` for
            current contribution, or ``['nai']`` to update concentration).

        Raises
        ------
        AssertionError
            If the ion name is unknown or a read/write symbol is invalid.
        ValueError
            If reversal potentials are written or a variable is both read and written.
        """
        read = read or []
        write = write or []

        if not read and not write:
            return

        assert ion in VALENCES, (
            f"Unknown ion {ion}. Valid ions are {list(VALENCES.keys())}."
        )

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
            declare_class_value(
                "mechanism.read_ion", {ion: read}, Mechanism._read_ion_declarations
            )

        if write:
            c_write = []
            other = []
            for w in write:
                if w in {f"{ion}i", f"{ion}o"}:
                    c_write.append(w)
                else:
                    other.append(w)

            if c_write:
                declare_class_value(
                    "mechanism.write_ion_c",
                    {ion: c_write},
                    Mechanism._write_ion_c_declarations,
                )
            if other:
                declare_class_value(
                    "mechanism.write_ion",
                    {ion: other},
                    Mechanism._write_ion_declarations,
                )

    @staticmethod
    def USEMATERIAL(material, read=None, write=None, source=None):
        """Declare generic material read/write dependencies.

        Parameters
        ----------
        material : str
            Material name, e.g. ``"ip3"`` or ``"ca"``.
        read : Sequence[str], optional
            Material fields to expose as local read buffers on this mechanism and
            its nested State modules.
        write : Sequence[str], optional
            Material fields that this mechanism locally replaces.  The handler
            commits these local buffers back to the full population Material
            after the local mechanism phase.
        source : Sequence[str] or Mapping[str, str], optional
            Additive material increments.  A sequence such as ``["ip3i"]``
            creates local source buffers named ``"ip3i_source"``.  A mapping
            such as ``{"ip3i": "j_ip3"}`` uses explicit local buffer names.

        Notes
        -----
        Unlike USEION, USEMATERIAL allows a field to appear in both ``read`` and
        ``write`` because local reaction mechanisms commonly need to read a
        material and then write its updated value.
        """
        read = _as_name_list(read)
        write = _as_name_list(write)
        source_map = _normalize_material_source(source)

        if not read and not write and not source_map:
            return

        material = str(material)
        declare_class_value(
            "mechanism.material", (material,), Mechanism._material_declarations
        )

        if read:
            declare_class_value(
                "mechanism.read_material",
                {material: read},
                Mechanism._read_material_declarations,
            )
        if write:
            declare_class_value(
                "mechanism.write_material",
                {material: write},
                Mechanism._write_material_declarations,
            )
        if source_map:
            declare_class_value(
                "mechanism.source_material",
                {material: source_map},
                Mechanism._source_material_declarations,
            )

    @staticmethod
    def NONSPECIFIC_CURRENT(*args):
        """
        Declare non-specific (leak) currents produced by the mechanism.

        Current methods and any properties they read must be deterministic
        during one solver evaluation.  Symbolic conductance assembly may
        evaluate a coefficient separately from the authored current method;
        stateful or stochastic property access can therefore make an
        ``(i, g)`` pair internally inconsistent.  Such currents should provide
        an exact ``<current>_with_conductance`` method instead.

        Parameters
        ----------
        *args : str
            Current names to register as non-specific.
        """
        declare_class_value(
            "mechanism.currents", args, Mechanism._currents_declarations
        )

    @staticmethod
    def INIT(**kwargs):
        """
        Declare initial buffer values for state variables.

        Parameters
        ----------
        **kwargs
            Mapping from state names to scalar initial conditions.
        """
        declare_class_value("mechanism.init", kwargs, Mechanism._init_declarations)

    @staticmethod
    def EXPLICIT(*args):
        """
        Mark currents as voltage independent when assembling the RHS.

        Parameters
        ----------
        *args : str
            Current names that should not contribute to conductance terms.
        """
        declare_class_value(
            "mechanism.explicit", args, Mechanism._explicit_declarations
        )

    @staticmethod
    def AFFINE(*args):
        """Assert that currents are exactly affine functions of voltage.

        Parameters
        ----------
        *args : str
            Current names satisfying ``I(v) = g * v + b``, where ``g`` and
            ``b`` do not depend on ``v`` during a solver evaluation.

        Notes
        -----
        Dufort--Frankel can center an affine current exactly between its two
        stored voltage levels. Analytic and numerically differentiated current
        pairs do not by themselves prove this stronger property: a nonlinear
        current may have a perfectly valid local derivative. Use ``AFFINE``
        when source analysis cannot establish the affine form. An incorrect
        declaration can produce an incorrect Dufort--Frankel trajectory.
        """
        declare_class_value("mechanism.affine", args, Mechanism._affine_declarations)

    @staticmethod
    def NUMERICAL(*args):
        """
        Assert that currents are safe for numerical differentiation.

        Parameters
        ----------
        *args : str
            Current names that should be numerically differentiated. Each
            current must be deterministic, side-effect free, and pointwise in
            voltage.

        Notes
        -----
        Numerical currents use centered finite differences and support float32
        and float64 voltage tensors. The declaration is an explicit assertion
        of the pointwise/purity contract: Dendra perturbs every voltage element
        simultaneously, which is correct for local currents but is a directional
        derivative for currents that couple compartments or batch elements.
        Stateful, nondeterministic, or voltage-mutating current methods are also
        invalid because centered differences evaluate them repeatedly. Like any
        same-precision finite difference, this path can lose accuracy when the
        current is dominated by a very large voltage-independent offset. Prefer
        a symbolically factorable current or an exact
        ``<current>_with_conductance`` method when available.
        """
        declare_class_value(
            "mechanism.numerical", args, Mechanism._numerical_declarations
        )

    def breakpoint(self, v):
        """
        Evaluate mechanism currents at the breakpoint stage.

        Parameters
        ----------
        v : Tensor
            Membrane potential values for the local compartments.

        Notes
        -----
        Override to compute mechanism-level :meth:`BUFFER` values and assemble
        currents (e.g., ``ina``, ``ik``, ``il``). Called each step before
        current accumulation.
        """
        return

    def detach(self):
        """
        Detach mechanism buffers and nested state modules from autograd.
        """
        super().detach()
        for state_module in self.DE.values():
            state_module.detach()

    def _install_monomorphic_advance(self):
        """Install a generated per-proxy `_advance` fast path when safe.

        The base `_advance` method is intentionally generic and polymorphic.  It
        is convenient for eager execution, but TorchDynamo specializes it on
        ``type(self)``.  In models with many generated mechanism proxy classes,
        that shared code object can recompile once per mechanism.  This installer
        replaces the inherited generic method on the concrete mechanism/proxy
        class with a generated method whose code object is unique to that class
        and whose state-module names / buffer keys are static literals.

        Subclasses that define their own `_advance` are left untouched.
        """
        cls = self.__class__
        if cls is Mechanism:
            return

        existing = cls.__dict__.get("_advance", None)
        base_advance = Mechanism.__dict__.get("_advance")

        if (
            existing is not None
            and existing is not base_advance
            and not getattr(existing, "_dendra_monomorphic_advance", False)
        ):
            return

        signature = _mechanism_advance_signature(self)
        if (
            getattr(existing, "_dendra_monomorphic_advance_signature", None)
            == signature
        ):
            return

        advance_fn, source = _compile_monomorphic_mechanism_advance(self, signature)
        setattr(cls, "_advance", advance_fn)
        cls._dendra_monomorphic_advance_signature = signature
        cls._dendra_monomorphic_advance_source = source

    def _advance(self, v, dt):
        """
        Advance nested state modules by one time step.

        Parameters
        ----------
        v : Tensor
            Membrane potentials for the local compartments.
        dt : Tensor
            Time-step tensor propagated from the integrator.

        Notes
        -----
        This generic fallback is normally replaced at instance construction by
        :meth:`_install_monomorphic_advance`, which installs a generated method
        on the concrete mechanism/proxy class.
        """
        states = self._gather_states()
        updates = {}
        for state_module in self.DE.values():
            local = state_module.advance(v, dt, states)
            updates.update(local)
        self._buffers.update(updates)

    def _gather_states(self):
        states = {
            state_name: self._buffers[state_name] for state_name in self._all_states
        }
        return states

    def populate(self, random_generation=None):
        """
        Populate mechanism and nested state parameter buffers.
        """
        self.populate_parameter_buffers(random_generation=random_generation)
        for state_module in self.DE.values():
            state_module.populate_parameter_buffers(random_generation=random_generation)

    def resample_random_parameters(self, *names, force: bool = True):
        """Resample mechanism-level and nested-State random parameters."""

        if names:
            local = tuple(n for n in names if n in self.random_parameters)
            if local:
                super().resample_random_parameters(*local, force=force)
        else:
            super().resample_random_parameters(force=force)
        for state_module in self.DE.values():
            if names:
                local = tuple(n for n in names if n in state_module.random_parameters)
                if local:
                    state_module.resample_random_parameters(*local, force=force)
            else:
                state_module.resample_random_parameters(force=force)
        return self

    def sample_runtime_noises_(
        self,
        *names,
        dt=None,
        phase: str | None = "pre_state",
        step_index: int | None = None,
        force: bool = False,
    ):
        """Sample mechanism-level and nested-State runtime noise in-place."""

        if names:
            local = tuple(n for n in names if n in self.runtime_noises)
            if local:
                super().sample_runtime_noises_(
                    *local, dt=dt, phase=phase, step_index=step_index, force=force
                )
        else:
            super().sample_runtime_noises_(
                dt=dt, phase=phase, step_index=step_index, force=force
            )
        for state_module in self.DE.values():
            if names:
                local = tuple(n for n in names if n in state_module.runtime_noises)
                if local:
                    state_module.sample_runtime_noises_(
                        *local, dt=dt, phase=phase, step_index=step_index, force=force
                    )
            else:
                state_module.sample_runtime_noises_(
                    dt=dt, phase=phase, step_index=step_index, force=force
                )
        return self

    def resample_runtime_noise(self, *names, dt=None, phase=None):
        self.sample_runtime_noises_(*names, dt=dt, phase=phase, force=True)
        return self

    def initial(self, v):
        """
        Hook for subclasses to initialize buffers from membrane potential.

        Parameters
        ----------
        v : Tensor
            Membrane potential values used for initialization.

        Notes
        -----
        Use this to populate mechanism-level :meth:`BUFFER` values (e.g.,
        cached conductances) or perform any one-time setup before stepping.
        """
        return

    @classmethod
    def rename(cls, new_name=None, suffix=None):
        """
        Create a renamed clone of the current mechanism class.

        Parameters
        ----------
        new_name : str, optional
            Explicit name for the cloned class. Required when ``suffix`` is ``None``.
        suffix : str, optional
            Suffix appended to the original class name when ``new_name`` is omitted.

        Returns
        -------
        type
            Mechanism subclass identical to ``cls`` but with a new ``__name__``.

        Raises
        ------
        ValueError
            If both ``new_name`` and ``suffix`` are ``None``.
        TypeError
            If ``suffix`` is provided but not a string.
        """
        if new_name is None and suffix is None:
            raise ValueError("Either new_name or suffix must be provided.")

        if new_name is None:
            new_name = cls.__name__

        if suffix is not None:
            if not isinstance(suffix, str):
                raise TypeError("Suffix must be a string.")
            new_name += f"_{suffix}"

        if new_name in cls._renamed_aliases:
            return cls._renamed_aliases[new_name]

        cls._renamed_aliases[new_name] = rename(cls, new_name=new_name)
        return cls._renamed_aliases[new_name]

    def batch(self, batch_size: int):
        """
        Broadcast mechanism buffers across an explicit batch dimension.

        Parameters
        ----------
        batch_size : int
            Size of the leading batch dimension to materialize.

        Returns
        -------
        Mechanism
            The mechanism instance with batched buffers.
        """
        super().batch(batch_size)
        for state_module in self.DE.values():
            state_module.batch(batch_size)
        return self

    def states(self):
        """
        Collect fully qualified state names.

        Returns
        -------
        list of str
            Names in the form ``\"{mechanism}.{state}\"``.
        """
        states = []
        for state_module in self.DE.values():
            states.extend(state_module._state)
        return [f"{self.name}.{state}" for state in states]

    @classmethod
    def state_names(cls):
        """
        Collect state variable names declared by the mechanism class.

        Returns
        -------
        list of str
            Names of state variables registered in ``cls._state``.
        """
        states = []
        for state_module in cls._state:
            states.extend(state_module._state)
        return states

    @staticmethod
    def _usage_values(values):
        """Return a stable, duplicate-free list of usage variable names."""
        if values is None:
            return []
        if isinstance(values, str):
            values = [values]
        out = []
        seen = set()
        for value in values:
            value = str(value)
            if value not in seen:
                seen.add(value)
                out.append(value)
        return sorted(out)

    @staticmethod
    def _format_usage_values(values):
        values = Mechanism._usage_values(values)
        return ", ".join(values) if values else "—"

    @staticmethod
    def _format_source_map(source_map):
        if not source_map:
            return "—"
        parts = []
        for field in sorted(source_map):
            local_name = source_map[field]
            if str(local_name) == f"{field}_source":
                parts.append(str(field))
            else:
                parts.append(f"{field}→{local_name}")
        return ", ".join(parts) if parts else "—"

    @classmethod
    def material_usage(cls, *, include_ions=True):
        """Return structured material/ion dependency metadata for this class.

        Parameters
        ----------
        include_ions : bool, default True
            If True, include ``USEION`` declarations alongside generic
            ``USEMATERIAL`` declarations.  Ion entries use ``read`` for
            concentration/reversal/current reads, ``write_current`` for ionic
            current writes such as ``ica``, and ``write_concentration`` for
            concentration writes such as ``cai``.

        Returns
        -------
        dict
            A dictionary with ``"materials"`` and, when requested, ``"ions"``
            entries.  The result is intended for debugging, documentation, and
            population-build diagnostics; it does not require mechanism
            instantiation.
        """
        materials = {}
        material_names = set(getattr(cls, "_material", set()) or set())
        material_names.update((getattr(cls, "_read_material", {}) or {}).keys())
        material_names.update((getattr(cls, "_write_material", {}) or {}).keys())
        material_names.update((getattr(cls, "_source_material", {}) or {}).keys())

        for material in sorted(material_names):
            source_map = dict(
                (getattr(cls, "_source_material", {}) or {}).get(material, {}) or {}
            )
            materials[material] = {
                "read": Mechanism._usage_values(
                    (getattr(cls, "_read_material", {}) or {}).get(material, [])
                ),
                "write": Mechanism._usage_values(
                    (getattr(cls, "_write_material", {}) or {}).get(material, [])
                ),
                "source": {str(k): str(v) for k, v in sorted(source_map.items())},
            }

        usage = {"materials": materials}

        if include_ions:
            ions = {}
            ion_names = set(getattr(cls, "_ion", set()) or set())
            ion_names.update((getattr(cls, "_read_ion", {}) or {}).keys())
            ion_names.update((getattr(cls, "_write_ion", {}) or {}).keys())
            ion_names.update((getattr(cls, "_write_ion_c", {}) or {}).keys())

            for ion in sorted(ion_names):
                ions[ion] = {
                    "read": Mechanism._usage_values(
                        (getattr(cls, "_read_ion", {}) or {}).get(ion, [])
                    ),
                    "write_current": Mechanism._usage_values(
                        (getattr(cls, "_write_ion", {}) or {}).get(ion, [])
                    ),
                    "write_concentration": Mechanism._usage_values(
                        (getattr(cls, "_write_ion_c", {}) or {}).get(ion, [])
                    ),
                }
            usage["ions"] = ions

        return usage

    @classmethod
    def material_summary(cls, *, include_ions=True, include_empty=False):
        """Return a printable summary of Material and Ion usage.

        Parameters
        ----------
        include_ions : bool, default True
            Include ``USEION`` declarations in the summary.  Ions are treated as
            specialized materials for the purpose of this report.
        include_empty : bool, default False
            If True, include empty ``materials`` / ``ions`` blocks even when the
            mechanism declares none.

        Returns
        -------
        str
            Human-readable multi-line summary.

        Examples
        --------
        >>> print(MyMechanism.material_summary())
        MyMechanism material usage:
          ions:
            ca: read=ica, write_concentration=cai
          materials:
            ip3: read=ip3i, source=ip3i→j_ip3
        """
        usage = cls.material_usage(include_ions=include_ions)
        label = getattr(cls, "_name", None) or cls.__name__
        lines = [f"{label} material usage:"]

        ions = usage.get("ions", {}) if include_ions else {}
        materials = usage.get("materials", {})

        if ions or include_empty:
            lines.append("  ions:" if ions else "  ions: —")
            for ion, data in ions.items():
                parts = []
                if data.get("read"):
                    parts.append(f"read={Mechanism._format_usage_values(data['read'])}")
                if data.get("write_current"):
                    parts.append(
                        f"write_current={Mechanism._format_usage_values(data['write_current'])}"
                    )
                if data.get("write_concentration"):
                    parts.append(
                        "write_concentration="
                        f"{Mechanism._format_usage_values(data['write_concentration'])}"
                    )
                lines.append(f"    {ion}: {', '.join(parts) if parts else '—'}")

        if materials or include_empty:
            lines.append("  materials:" if materials else "  materials: —")
            for material, data in materials.items():
                parts = []
                if data.get("read"):
                    parts.append(f"read={Mechanism._format_usage_values(data['read'])}")
                if data.get("write"):
                    parts.append(
                        f"write={Mechanism._format_usage_values(data['write'])}"
                    )
                if data.get("source"):
                    parts.append(
                        f"source={Mechanism._format_source_map(data['source'])}"
                    )
                lines.append(f"    {material}: {', '.join(parts) if parts else '—'}")

        if not ions and not materials and not include_empty:
            lines.append("  —")

        return "\n".join(lines)

    @classmethod
    def materials_summary(cls, **kwargs):
        """Alias for :meth:`material_summary`."""
        return cls.material_summary(**kwargs)

    @classproperty
    def code(cls):
        """
        Source code of the mechanism class.

        Returns
        -------
        str
            Dedented string containing the class definition.
        """
        source_code = inspect.getsource(cls)
        return textwrap.dedent(source_code)

    @classproperty
    def file_code(cls) -> str:
        """
        Source text of the Python module defining the class.

        Returns
        -------
        str
            Entire module contents containing ``cls``.

        Raises
        ------
        RuntimeError
            If the module where ``cls`` is defined cannot be located.
        """
        mod = inspect.getmodule(cls)
        if mod is None:
            raise RuntimeError(f"Cannot locate module for {cls.__qualname__}")
        return inspect.getsource(mod)  # whole file

    def states_dict(self):
        """
        Map state names to their underlying buffers.

        Returns
        -------
        dict
            Dictionary from raw state names to tensors.
        """
        dct = {}
        for state_module in self.DE.values():
            for state_name in state_module._state:
                dct[state_name] = self._buffers[state_name]
        return dct

    # -- rng --
    def init_rng(self):
        for state_module in self.DE.values():
            state_module.init_rng()
        super().init_rng()

    def reset_rng(self):
        for state_module in self.DE.values():
            state_module.reset_rng()
        super().reset_rng()

    # -- tables --
    def usetables(self, value: bool):
        """
        Enable or disable table usage for the mechanism and nested states.

        Parameters
        ----------
        value : bool
            Whether to use tables for function approximations.
        """
        for state_module in self.DE.values():
            state_module.usetables(value)
        super().usetables(value)

    @classmethod
    def all_mech_parameter_names(cls):
        """
        Collect all parameter names declared by the mechanism class.

        Returns
        -------
        list of str
            Names of parameters registered in ``cls._parameters``.
        """
        param_names = {}
        param_names["mechanism"] = cls.all_parameter_names()
        for state_module in cls._state:
            param_names[state_module.__name__] = state_module.all_parameter_names()
        return param_names

    @classmethod
    def check_kwargs(cls, kwargs):
        """
        Check for unexpected keyword arguments.

        Parameters
        ----------
        **kwargs
            Keyword arguments to validate.

        Raises
        ------
        ValueError
            If any unexpected keyword arguments are found.
        """
        valid_keys = set()
        for state_module in cls._state:
            valid_keys.update(state_module.all_parameter_names())
        valid_keys.update(cls.all_parameter_names())

        for key in kwargs.keys():
            if key not in valid_keys:
                raise ValueError(f"Unexpected keyword argument: {key}")

    # -- mechanism-level delayed states ---------------------------------------
    def register_delayed_state(
        self,
        name,
        like,
        delay_steps: int,
        *,
        mode="auto",
        buffer_name=None,
        pointer_name=None,
        insert_axis: int = -1,
        clear: bool = True,
    ):
        """Register a fixed-step delayed state buffer on this mechanism.

        This helper is for fused mechanisms that need NetCon-like fixed delays
        while bypassing :class:`~dendra.models.networks.NetCon`.  The delayed
        state has two update backends:

        ``mode="shift"``
            Graph-safe functional update. Each call constructs a new queue with
            ``torch.cat``. This is appropriate for training / BPTT because the
            delayed value remains connected to the computation graph.

        ``mode="circular"``
            Fast in-place circular buffer. This avoids shifting/copying the
            whole queue each step, but it intentionally writes under
            ``torch.no_grad()`` and is therefore intended for evaluation only.

        ``mode="auto"``
            Use ``circular`` when the mechanism is in eval mode and gradients are
            disabled; otherwise use ``shift``.

        Parameters
        ----------
        name : str
            User-facing delay name used with :meth:`delayed_state`.
        like : torch.Tensor
            Tensor whose shape/device/dtype define the payload shape.
        delay_steps : int
            Integer delay in simulation steps. A value emitted at call ``k``
            appears after ``delay_steps`` subsequent calls, matching the queue
            convention used by Dendra's event delay buffers. The current delayed-state
            helper assumes one uniform delay per registered delayed state. For heterogeneous
            per-connection delays, use ``NetCon`` / ``ContinuousCon`` or register
            separate delayed states for each distinct delay.
        mode : {"auto", "shift", "circular"}, default "auto"
            Backend selection policy.
        buffer_name : str, optional
            Name of the registered buffer. Defaults to
            ``f"{name}_delay_buffer"``. Existing mechanism ``BUFFER`` variables can be
            reused by passing their name here.
        pointer_name : str, optional
            Name of the circular-buffer write pointer.
        insert_axis : int, default -1
            Axis before which the delay dimension is inserted. The default
            inserts the delay axis before the final payload dimension, so a
            payload of shape ``(..., n)`` becomes ``(..., depth, n)``.
        clear : bool, default True
            If true, reset the delay buffer and pointer.

        Returns
        -------
        torch.Tensor
            The registered delay buffer.
        """
        if mode is None:
            mode = "auto"
        mode = str(mode).lower()
        aliases = {
            "functional": "shift",
            "queue": "shift",
            "shift_queue": "shift",
            "eval_circular": "auto",
            "fast_eval": "auto",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"auto", "shift", "circular"}:
            raise ValueError(
                "delayed-state mode must be 'auto', 'shift', or 'circular'; "
                f"got {mode!r}."
            )

        delay_steps_t = _nonnegative_integer_steps(delay_steps)
        if delay_steps_t.numel() != 1:
            raise ValueError("delay_steps must be one non-negative integer.")
        delay_steps = int(delay_steps_t.reshape(()).item())
        depth = max(1, delay_steps + 1)
        like = torch.as_tensor(like)

        if insert_axis < 0:
            insert_axis = like.ndim + 1 + insert_axis
        if insert_axis < 0 or insert_axis > like.ndim:
            raise ValueError(
                f"insert_axis={insert_axis} is invalid for payload ndim={like.ndim}."
            )

        buffer_name = buffer_name or f"{name}_delay_buffer"
        pointer_name = pointer_name or f"{name}_delay_ptr"

        buffer_shape = (
            tuple(like.shape[:insert_axis]) + (depth,) + tuple(like.shape[insert_axis:])
        )
        buffer = torch.zeros(buffer_shape, device=like.device, dtype=like.dtype)
        pointer = torch.zeros((), device=like.device, dtype=torch.long)

        buffer_replaced = False
        if buffer_name in self._buffers:
            current_buffer = getattr(self, buffer_name)
            buffer_replaced = (
                tuple(current_buffer.shape) != buffer_shape
                or current_buffer.device != buffer.device
                or current_buffer.dtype != buffer.dtype
            )
            if clear or buffer_replaced:
                setattr(self, buffer_name, buffer)
        else:
            self.register_buffer(buffer_name, buffer)
            buffer_replaced = True

        if pointer_name in self._buffers:
            # A pointer into a newly allocated buffer has no meaningful history
            # to preserve and may be outside the new buffer's bounds.
            if clear or buffer_replaced:
                setattr(self, pointer_name, pointer)
        else:
            self.register_buffer(pointer_name, pointer)

        self._delayed_state_specs[name] = {
            "buffer": buffer_name,
            "pointer": pointer_name,
            "steps": delay_steps,
            "depth": depth,
            "axis": int(insert_axis),
            "value_shape": tuple(like.shape),
            "mode": mode,
        }
        return getattr(self, buffer_name)

    def register_delayed_states(
        self,
        name,
        like,
        delay_steps,
        *,
        mode="auto",
        buffer_name=None,
        pointer_name=None,
        steps_name=None,
        stream_axis: int = -2,
        delay_axis: int | None = None,
        clear: bool = True,
    ):
        """Register a batched fixed-step delay line for several streams.

        This is the multi-stream counterpart of :meth:`register_delayed_state`.
        It is intended for fused mechanisms that need many pathway-level delays
        over tensors with the same payload shape.  For example, a value tensor
        with shape ``(batch, n_streams, n)`` and ``stream_axis=-2`` is stored in
        one delay buffer with shape ``(batch, depth, n_streams, n)``.

        Parameters
        ----------
        name : str
            User-facing delay group name used with :meth:`delayed_states`.
        like : torch.Tensor
            Example value tensor. One dimension is interpreted as the stream /
            pathway axis; all other dimensions are payload dimensions.
        delay_steps : int or sequence[int] or torch.Tensor
            Integer delays in simulation steps. A scalar applies the same delay
            to every stream. A vector must have length ``like.shape[stream_axis]``.
        mode : {"auto", "shift", "circular"}, default "auto"
            Backend selection policy. ``auto`` uses circular buffers in eval /
            no-grad mode and graph-safe shifted queues during training/grad mode.
        buffer_name : str, optional
            Name of the registered delay buffer. Defaults to
            ``f"{name}_delay_buffer"``.
        pointer_name : str, optional
            Name of the circular-buffer write pointer. Defaults to
            ``f"{name}_delay_ptr"``.
        steps_name : str, optional
            Name of the registered integer delay vector. Defaults to
            ``f"{name}_delay_steps"``.
        stream_axis : int, default -2
            Axis of ``like`` containing independent delay streams.
        delay_axis : int or None, default None
            Axis of the delay buffer where the delay dimension is inserted. If
            ``None``, the delay dimension is inserted immediately before
            ``stream_axis`` for backwards compatibility. Use ``delay_axis=0``
            for a time-major buffer layout ``(depth, *like.shape)``, which is
            often preferable for GPU execution.
        clear : bool, default True
            If true, reset the delay buffer and circular pointer.

        Returns
        -------
        torch.Tensor
            The registered delay buffer.

        Notes
        -----
        ``delayed_states`` supports heterogeneous per-stream integer delays,
        but all streams in a group share one circular pointer and one buffer
        depth equal to ``max(delay_steps) + 1``. This is useful for fused
        models with several fixed pathway delays and avoids one delayed-state
        helper call per pathway.
        """
        if mode is None:
            mode = "auto"
        mode = str(mode).lower()
        aliases = {
            "functional": "shift",
            "queue": "shift",
            "shift_queue": "shift",
            "eval_circular": "auto",
            "fast_eval": "auto",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"auto", "shift", "circular"}:
            raise ValueError(
                "batched delayed-state mode must be 'auto', 'shift', or 'circular'; "
                f"got {mode!r}."
            )

        like = torch.as_tensor(like)
        if like.ndim == 0:
            raise ValueError("register_delayed_states requires a non-scalar payload.")

        if stream_axis < 0:
            stream_axis = like.ndim + stream_axis
        if stream_axis < 0 or stream_axis >= like.ndim:
            raise ValueError(
                f"stream_axis={stream_axis} is invalid for payload ndim={like.ndim}."
            )
        n_streams = int(like.shape[stream_axis])
        if n_streams <= 0:
            raise ValueError("delayed-state stream axis must be non-empty.")

        steps = _nonnegative_integer_steps(delay_steps, device=like.device)
        if steps.ndim == 0 or steps.numel() == 1:
            steps = steps.reshape(1).expand(n_streams).clone()
        else:
            steps = steps.reshape(-1).clone()
            if steps.numel() != n_streams:
                raise ValueError(
                    f"delay_steps has length {steps.numel()}, but the stream axis "
                    f"has length {n_streams}."
                )
        if torch.any(steps < 0):
            raise ValueError("delay_steps must be non-negative integers.")

        max_steps = int(steps.max().item()) if steps.numel() else 0
        depth = max(1, max_steps + 1)

        # Insert the delay axis.  By default this preserves the original layout
        # (..., depth, streams, n).  Passing delay_axis=0 gives a time-major
        # layout (depth, ...), which tends to be more compiler/GPU friendly.
        if delay_axis is None:
            delay_axis = int(stream_axis)
        else:
            if delay_axis < 0:
                delay_axis = like.ndim + 1 + delay_axis
            if delay_axis < 0 or delay_axis > like.ndim:
                raise ValueError(
                    f"delay_axis={delay_axis} is invalid for payload ndim={like.ndim}."
                )

        buffer_stream_axis = (
            int(stream_axis) + 1 if delay_axis <= int(stream_axis) else int(stream_axis)
        )
        buffer_shape = (
            tuple(like.shape[:delay_axis]) + (depth,) + tuple(like.shape[delay_axis:])
        )

        buffer_name = buffer_name or f"{name}_delay_buffer"
        pointer_name = pointer_name or f"{name}_delay_ptr"
        steps_name = steps_name or f"{name}_delay_steps"

        buffer = torch.zeros(buffer_shape, device=like.device, dtype=like.dtype)
        pointer = torch.zeros((), device=like.device, dtype=torch.long)

        buffer_replaced = False
        if buffer_name in self._buffers:
            current_buffer = getattr(self, buffer_name)
            buffer_replaced = (
                tuple(current_buffer.shape) != buffer_shape
                or current_buffer.device != buffer.device
                or current_buffer.dtype != buffer.dtype
            )
            if clear or buffer_replaced:
                setattr(self, buffer_name, buffer)
        else:
            self.register_buffer(buffer_name, buffer)
            buffer_replaced = True

        if pointer_name in self._buffers:
            if clear or buffer_replaced:
                setattr(self, pointer_name, pointer)
        else:
            self.register_buffer(pointer_name, pointer)

        if steps_name in self._buffers:
            # Delay metadata is configuration, not queue history. Re-registering
            # with clear=False preserves compatible payload history but must still
            # install the newly requested delay vector.
            setattr(self, steps_name, steps)
        else:
            self.register_buffer(steps_name, steps)

        # Static delay metadata used to avoid scalar tensor reductions in the
        # compiled per-step path.  In particular, do not call Tensor.item() from
        # delayed_states(...): TorchDynamo treats that as a graph break unless
        # capture_scalar_outputs is enabled.  These booleans are valid for the
        # registered delay vector; if callers override delay_steps dynamically,
        # delayed_states falls back to the generic update path without relying on
        # these flags.
        has_zero_delay = bool(torch.any(steps == 0).item())
        all_zero_delay = bool(torch.all(steps <= 0).item())

        self._delayed_state_specs[name] = {
            "buffer": buffer_name,
            "pointer": pointer_name,
            "steps_buffer": steps_name,
            "steps": max_steps,
            "depth": depth,
            "axis": delay_axis,
            "stream_axis": buffer_stream_axis,
            "value_stream_axis": int(stream_axis),
            "value_shape": tuple(like.shape),
            "n_streams": n_streams,
            "mode": mode,
            "batched": True,
            "has_zero_delay": has_zero_delay,
            "all_zero_delay": all_zero_delay,
        }
        return getattr(self, buffer_name)

    def reset_delayed_states(self, *names):
        """Zero registered delayed-state buffers and reset circular pointers."""
        if not names:
            names = tuple(self._delayed_state_specs.keys())
        for name in names:
            spec = self._delayed_state_specs[name]
            # Shift queues may be graph-connected and their returned delayed
            # values may be views/copies whose backward pass still references the
            # queue. Rebinding fresh detached tensors both severs the old history
            # and avoids invalidating an outstanding backward pass in place.
            buffer = getattr(self, spec["buffer"])
            pointer = getattr(self, spec["pointer"])
            setattr(self, spec["buffer"], torch.zeros_like(buffer).detach())
            setattr(self, spec["pointer"], torch.zeros_like(pointer).detach())
        return self

    def delayed_state(self, name, value, *, delay_steps=None, mode=None):
        """Return a delayed copy of ``value`` and update the named delay line.

        The delay line must first be created with
        :meth:`register_delayed_state`. ``mode="auto"`` uses the in-place
        circular buffer only in eval/no-grad mode; otherwise it uses a graph-safe
        functional shift update.
        """
        if name not in self._delayed_state_specs:
            raise KeyError(
                f"No delayed state named {name!r} has been registered. "
                "Call register_delayed_state(...) during initial(...)."
            )
        spec = self._delayed_state_specs[name]
        if spec.get("batched", False):
            raise ValueError(
                f"Delayed state {name!r} was registered with "
                "register_delayed_states; use delayed_states(...) instead."
            )
        if delay_steps is None:
            steps = int(spec["steps"])
        else:
            steps_t = _nonnegative_integer_steps(delay_steps)
            if steps_t.numel() != 1:
                raise ValueError("delay_steps must be one non-negative integer.")
            steps = int(steps_t.reshape(()).item())

        selected_mode = str(spec["mode"] if mode is None else mode).lower()
        aliases = {"functional": "shift", "queue": "shift", "shift_queue": "shift"}
        selected_mode = aliases.get(selected_mode, selected_mode)
        if selected_mode == "auto":
            selected_mode = (
                "shift" if (self.training or torch.is_grad_enabled()) else "circular"
            )
        if selected_mode not in {"shift", "circular"}:
            raise ValueError(
                "delayed-state mode must be 'auto', 'shift', or 'circular'; "
                f"got {selected_mode!r}."
            )

        if not torch.is_tensor(value):
            raise TypeError("delayed_state value must be a torch.Tensor.")
        expected_shape = spec.get("value_shape")
        if expected_shape is None:
            expected_shape = list(getattr(self, spec["buffer"]).shape)
            del expected_shape[int(spec["axis"])]
        expected_shape = tuple(expected_shape)
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"Delayed state {name!r} expected value shape {expected_shape}, "
                f"but received {tuple(value.shape)}."
            )
        if steps <= 0:
            return value

        if selected_mode == "circular":
            return self._delayed_state_circular(spec, value, steps)
        if selected_mode == "shift":
            return self._delayed_state_shift(spec, value, steps)

    def _delayed_state_shift(self, spec, value, steps: int):
        """Graph-safe delayed-state update using an out-of-place shifted queue."""
        buf_name = spec["buffer"]
        axis = int(spec["axis"])
        buf = getattr(self, buf_name)
        depth = int(buf.shape[axis])
        if steps + 1 != depth:
            # Re-register if a caller overrides delay_steps with a different
            # length. This is uncommon but keeps the helper predictable.
            state_name = next(
                k for k, v in self._delayed_state_specs.items() if v is spec
            )
            self.register_delayed_state(
                state_name,
                value,
                steps,
                mode=spec["mode"],
                buffer_name=buf_name,
                pointer_name=spec["pointer"],
                insert_axis=axis,
                clear=True,
            )
            buf = getattr(self, buf_name)
            depth = int(buf.shape[axis])

        sl = [slice(None)] * buf.ndim
        sl[axis] = slice(0, depth - 1)
        buf_new = torch.cat((value.unsqueeze(axis), buf[tuple(sl)]), dim=axis)
        setattr(self, buf_name, buf_new)
        return buf_new.select(axis, depth - 1)

    def _delayed_state_circular(self, spec, value, steps: int):
        """Fast eval delayed-state update using an in-place circular buffer."""
        buf = getattr(self, spec["buffer"])
        ptr = getattr(self, spec["pointer"])
        axis = int(spec["axis"])
        depth = int(buf.shape[axis])
        if steps + 1 != depth:
            raise ValueError(
                "Circular delayed_state cannot change delay_steps without "
                "re-registering the delayed state."
            )

        read_idx = torch.remainder(ptr - int(steps), depth).reshape(1)
        delayed = buf.index_select(axis, read_idx).squeeze(axis)

        # Evaluation-only fast path: mutate the circular buffer without building
        # autograd history. In training/grad mode, delayed_state(..., mode='auto')
        # selects the graph-safe shift backend instead.
        with torch.no_grad():
            write_idx = ptr.reshape(1)
            buf.index_copy_(axis, write_idx, value.detach().unsqueeze(axis))
            ptr.add_(1).remainder_(depth)
        return delayed

    def delayed_states(self, name, values, *, delay_steps=None, mode=None):
        """Return delayed copies of a multi-stream value tensor.

        The delay group must first be created with
        :meth:`register_delayed_states`.  ``values`` must have the same shape as
        the ``like`` tensor used at registration time.  The stream axis can have
        distinct integer delays supplied during registration.
        """
        if name not in self._delayed_state_specs:
            raise KeyError(
                f"No delayed state group named {name!r} has been registered. "
                "Call register_delayed_states(...) during initial(...)."
            )
        spec = self._delayed_state_specs[name]
        if not spec.get("batched", False):
            raise ValueError(
                f"Delayed state {name!r} was registered with register_delayed_state; "
                "use delayed_state(...) instead."
            )

        selected_mode = str(spec["mode"] if mode is None else mode).lower()
        aliases = {"functional": "shift", "queue": "shift", "shift_queue": "shift"}
        selected_mode = aliases.get(selected_mode, selected_mode)
        if selected_mode == "auto":
            selected_mode = (
                "shift" if (self.training or torch.is_grad_enabled()) else "circular"
            )
        if selected_mode not in {"shift", "circular"}:
            raise ValueError(
                "batched delayed-state mode must be 'auto', 'shift', or 'circular'; "
                f"got {selected_mode!r}."
            )

        if not torch.is_tensor(values):
            raise TypeError("delayed_states values must be a torch.Tensor.")
        expected_shape = spec.get("value_shape")
        if expected_shape is None:
            expected_shape = list(getattr(self, spec["buffer"]).shape)
            del expected_shape[int(spec["axis"])]
        expected_shape = tuple(expected_shape)
        if tuple(values.shape) != expected_shape:
            raise ValueError(
                f"Delayed state group {name!r} expected values shape "
                f"{expected_shape}, but received {tuple(values.shape)}."
            )

        # Fast all-zero shortcut for the registered delay vector.  This is a
        # Python bool stored at registration time, so it is safe under
        # torch.compile.  Avoid Tensor.item() here: the per-step path may be
        # captured by TorchDynamo, and scalar extraction causes a graph break.
        if delay_steps is None and bool(spec.get("all_zero_delay", False)):
            return values

        steps = self._delayed_states_steps(spec, values, delay_steps)

        runtime_override = delay_steps is not None
        if selected_mode == "circular":
            return self._delayed_states_circular(
                spec, values, steps, validate_capacity=runtime_override
            )
        if selected_mode == "shift":
            return self._delayed_states_shift(
                spec, values, steps, resize=runtime_override
            )

    def _delayed_states_steps(self, spec, values, delay_steps=None):
        # Registered delay vectors are normalized at initialization and kept as
        # long buffers on the mechanism device.  Avoid per-step .to(...),
        # torch.as_tensor(...), and shape checks in the common path.
        if delay_steps is None:
            return getattr(self, spec["steps_buffer"])

        steps = _nonnegative_integer_steps(delay_steps, device=values.device)
        if steps.ndim == 0 or steps.numel() == 1:
            return steps.reshape(1).expand(int(spec["n_streams"]))
        steps = steps.reshape(-1)
        if steps.numel() != int(spec["n_streams"]):
            raise ValueError(
                f"delay_steps has length {steps.numel()}, but the stream axis "
                f"has length {int(spec['n_streams'])}."
            )
        return steps

    def _delayed_states_gather(self, buf, spec, read_idx):
        axis = int(spec["axis"])
        stream_axis = int(spec["stream_axis"])
        n_streams = int(spec["n_streams"])

        index_shape = [1] * buf.ndim
        index_shape[axis] = 1
        index_shape[stream_axis] = n_streams

        expand_shape = list(buf.shape)
        expand_shape[axis] = 1

        idx = read_idx.reshape(index_shape).expand(expand_shape)
        return torch.gather(buf, dim=axis, index=idx).squeeze(axis)

    def _delayed_states_zero_mask(self, spec, values, steps):
        value_stream_axis = int(spec["value_stream_axis"])
        mask_shape = [1] * values.ndim
        mask_shape[value_stream_axis] = int(spec["n_streams"])
        return (steps == 0).reshape(mask_shape)

    def _delayed_states_shift(self, spec, values, steps, *, resize=False):
        """Graph-safe multi-stream delay update using a shifted queue."""
        buf_name = spec["buffer"]
        axis = int(spec["axis"])
        buf = getattr(self, buf_name)
        depth = int(buf.shape[axis])
        required_depth = int(spec.get("steps", 0)) + 1
        if resize:
            required_depth = int(steps.max().item()) + 1 if steps.numel() else 1
        if required_depth != depth:
            state_name = next(
                k for k, v in self._delayed_state_specs.items() if v is spec
            )
            self.register_delayed_states(
                state_name,
                values,
                steps,
                mode=spec["mode"],
                buffer_name=buf_name,
                pointer_name=spec["pointer"],
                steps_name=spec["steps_buffer"],
                stream_axis=int(spec["value_stream_axis"]),
                delay_axis=int(spec["axis"]),
                clear=True,
            )
            spec = self._delayed_state_specs[state_name]
            buf = getattr(self, buf_name)
            depth = int(buf.shape[axis])

        sl = [slice(None)] * buf.ndim
        sl[axis] = slice(0, depth - 1)
        buf_new = torch.cat((values.unsqueeze(axis), buf[tuple(sl)]), dim=axis)
        setattr(self, buf_name, buf_new)
        return self._delayed_states_gather(buf_new, spec, steps)

    def _delayed_states_circular(self, spec, values, steps, *, validate_capacity=False):
        """Fast eval multi-stream delay update using one in-place ring buffer."""
        buf = getattr(self, spec["buffer"])
        ptr = getattr(self, spec["pointer"])
        axis = int(spec["axis"])
        depth = int(buf.shape[axis])

        if validate_capacity and bool(torch.any(steps >= depth).item()):
            raise ValueError(
                "Runtime delay_steps exceed the registered circular-buffer "
                f"capacity of {depth - 1} steps. Re-register the delayed state."
            )

        read_idx = torch.remainder(ptr - steps, depth)
        delayed = self._delayed_states_gather(buf, spec, read_idx)

        # A stream with zero delay should deliver the current value, not the
        # previous content of the circular slot that will be overwritten below.
        delayed = torch.where(
            self._delayed_states_zero_mask(spec, values, steps), values, delayed
        )

        with torch.no_grad():
            write_idx = ptr.reshape(1)
            buf.index_copy_(axis, write_idx, values.detach().unsqueeze(axis))
            ptr.add_(1).remainder_(depth)
        return delayed

    # -- mechanism-level waveform injections ---------------------------------
    def inject(
        self,
        waveform,
        *,
        index=None,
        shape=None,
        model_shape=None,
        model=None,
        **kwargs,
    ):
        """Optionally attach a waveform stimulus to this mechanism.

        The default implementation is deliberately a no-op and returns ``False``.
        Mechanisms that own their own voltage/current dynamics can override this
        method and either consume the arguments directly or call
        :meth:`register_waveform_injection` to get padded ``I(t)`` tensors.

        Parameters
        ----------
        waveform : Waveform
            Waveform object supplied through ``model[idx].inject(waveform)``.
        index : tuple, optional
            Population-level index tuple identifying the targeted compartments.
        shape : tuple, optional
            Shape produced by applying ``index`` to the population.
        model_shape : tuple, optional
            Full population voltage shape at registration time.
        model : Population, optional
            Owning population. Used for device/dtype/shape information.
        **kwargs
            Reserved for future extension.

        Returns
        -------
        bool
            ``True`` only if the mechanism accepts complete ownership of the
            injection.  Returning ``False`` leaves delivery to the solver path.
        """
        return False

    def clear_injections(self):
        """Remove all waveform injections registered on this mechanism."""
        self.injected_waveforms = torch.nn.ModuleList()
        self._injection_specs = []
        for name in list(self._buffers.keys()):
            if (
                name.startswith("_injection_mask_")
                or name.startswith("_injection_scale_")
                or name.startswith("_injection_index_")
            ):
                delattr(self, name)
        return self

    def register_waveform_injection(
        self,
        waveform,
        *,
        index=None,
        model_shape=None,
        current_name="i_inj",
        scale=1.0,
        model=None,
    ):
        """Register a waveform and build a local padding mask for this mechanism.

        This helper is intended for mechanism subclasses that override
        :meth:`inject`.  It computes the overlap between a population-level
        injection index and the compartments occupied by this mechanism, stores
        the waveform as a submodule, and records a boolean local mask.  Later,
        :meth:`evaluate_injections` evaluates all registered waveforms at the
        current mechanism time and returns a tensor shaped like the local voltage
        argument, with zeros outside the targeted compartments.

        Registration succeeds only when this mechanism covers every targeted
        population location.  Partial overlap returns ``False`` without storing
        state so the solver can deliver the full injection exactly once.
        """
        device = self.diam.device
        dtype = self.diam.dtype
        if model is not None:
            device = model.device()
            dtype = model.dtype()

        if hasattr(waveform, "to"):
            waveform = waveform.to(device=device, dtype=dtype)

        if model_shape is None:
            # No population frame was supplied: treat this as targeting every
            # compartment where the mechanism resides.
            local_mask = torch.ones_like(self.diam, dtype=torch.bool, device=device)
        else:
            full_mask = torch.zeros(tuple(model_shape), dtype=torch.bool, device=device)
            if index is None:
                full_mask.fill_(True)
            else:
                full_mask[index] = True
            local_mask = self.get(full_mask)

        if local_mask.numel() == 0 or not bool(torch.any(local_mask).item()):
            return False

        # ``True`` from Mechanism.inject means that this mechanism owns the
        # complete population-level injection.  Reject partial overlap before
        # registering anything so the solver fallback can safely deliver the
        # whole stimulus without duplicating covered locations.
        local_coverage = local_mask.to(dtype=torch.bool)
        full_coverage = self.put(
            local_coverage,
            torch.zeros_like(full_mask),
            full_mask,
        ).to(dtype=torch.bool)
        if not bool(torch.all(full_coverage[full_mask]).item()):
            return False

        k = len(self.injected_waveforms)
        mask_name = f"_injection_mask_{k}"
        scale_name = f"_injection_scale_{k}"
        index_name = f"_injection_index_{k}"
        selected_index = torch.nonzero(local_mask.reshape(-1), as_tuple=False).reshape(
            -1
        )
        self.register_buffer(mask_name, local_mask.detach().clone())
        self.register_buffer(
            scale_name,
            torch.as_tensor(scale, device=device, dtype=dtype).detach().clone(),
        )
        self.register_buffer(index_name, selected_index.detach().clone())
        self.injected_waveforms.append(waveform)
        self._injection_specs.append(
            {
                "mask": mask_name,
                "scale": scale_name,
                "index": index_name,
                "n_selected": int(selected_index.numel()),
                "current_name": current_name,
            }
        )

        # Expose the current variable immediately for introspection, even before
        # the first timestep.  Subclasses may also declare it with BUFFER.
        if not hasattr(self, current_name):
            self.register_buffer(
                current_name, torch.zeros_like(local_mask, dtype=dtype)
            )
        return True

    def _expand_injection_value(
        self,
        value,
        mask,
        out,
        *,
        selected_index=None,
        n_local_selected=None,
    ):
        """Return ``value`` padded/broadcast into ``out`` at ``mask`` locations."""
        value = torch.as_tensor(value, device=out.device, dtype=out.dtype)

        # Bring an unbatched mask up to the current local state shape.
        local_mask = mask.to(device=out.device, dtype=torch.bool)
        mask = local_mask
        while mask.ndim < out.ndim:
            mask = mask.unsqueeze(0)
        mask = mask.expand_as(out)

        if value.ndim == 0 or value.numel() == 1:
            return value.reshape(()) * mask.to(out.dtype)

        if tuple(value.shape) == tuple(out.shape):
            return value * mask.to(out.dtype)

        # Common case: waveform returns the unbatched local mechanism shape.
        if value.ndim <= out.ndim:
            v = value
            while v.ndim < out.ndim:
                v = v.unsqueeze(0)
            if tuple(v.shape) == tuple(out.shape) or all(
                a == b or a == 1 for a, b in zip(v.shape, out.shape)
            ):
                return v.expand_as(out) * mask.to(out.dtype)

        # Vector over selected compartments.  This supports either a single
        # unbatched vector of length n_selected or a batched tensor whose last
        # dimension is n_selected.
        # If the mask describes the local (unbatched) suffix, accept either one
        # selected vector shared across batches or one vector per batch.
        if local_mask.ndim <= out.ndim and tuple(
            out.shape[out.ndim - local_mask.ndim :]
        ) == tuple(local_mask.shape):
            batch_shape = tuple(out.shape[: out.ndim - local_mask.ndim])
            if n_local_selected is None:
                n_local_selected = int(local_mask.reshape(-1).sum().item())
            selected_shape = batch_shape + (n_local_selected,)
            selected = None
            if value.numel() == n_local_selected:
                selected = value.reshape(
                    (1,) * len(batch_shape) + (n_local_selected,)
                ).expand(selected_shape)
            else:
                try:
                    selected = value.expand(selected_shape)
                except RuntimeError:
                    pass
            if selected is not None:
                if selected_index is None:
                    selected_index = torch.nonzero(
                        local_mask.reshape(-1), as_tuple=False
                    ).reshape(-1)
                padded = torch.zeros_like(out).reshape(-1, local_mask.numel())
                padded = padded.index_copy(
                    1,
                    selected_index.to(device=out.device),
                    selected.reshape(-1, n_local_selected),
                )
                return padded.reshape_as(out)

        n_selected = int(mask.reshape(-1).sum().item())
        if value.numel() == n_selected:
            padded = torch.zeros_like(out)
            padded.reshape(-1)[mask.reshape(-1)] = value.reshape(-1)
            return padded

        # Last-resort attempt: rely on PyTorch broadcasting, then mask.
        try:
            return value.expand_as(out) * mask.to(out.dtype)
        except RuntimeError as exc:
            raise ValueError(
                f"Waveform output shape {tuple(value.shape)} cannot be broadcast "
                f"or padded into mechanism-local shape {tuple(out.shape)}."
            ) from exc

    def evaluate_injections(self, v=None, *, t=None, current_name="i_inj"):
        """Evaluate registered waveform injections and expose the result.

        Parameters
        ----------
        v : torch.Tensor, optional
            Local voltage/state tensor that defines the desired output shape.  If
            omitted, the first injection mask shape is used.
        t : torch.Tensor or float, optional
            Evaluation time in ms.  Defaults to the mechanism's ``t`` reference,
            which :class:`Population` sets during build.
        current_name : str, optional
            Name of the exposed current variable.  Defaults to ``i_inj``.

        Returns
        -------
        torch.Tensor
            Sum of all registered waveform currents, padded to ``v``'s shape.
        """
        if v is None:
            if self._injection_specs:
                v = getattr(self, self._injection_specs[0]["mask"]).to(self.diam.dtype)
            elif hasattr(self, current_name):
                v = getattr(self, current_name)
            else:
                v = self.diam

        out = torch.zeros_like(v, dtype=v.dtype, device=v.device)
        if not self._injection_specs:
            setattr(self, current_name, out)
            return out

        if t is None:
            t = (
                self.t
                if hasattr(self, "t")
                else torch.zeros((), device=v.device, dtype=v.dtype)
            )
        t = torch.as_tensor(t, device=v.device, dtype=v.dtype)
        t = torch.atleast_1d(t)
        if t.numel() != 1:
            raise ValueError(
                "Mechanism waveform injections are evaluated one timestep at a "
                f"time; got {t.numel()} time values."
            )

        for k, spec in enumerate(self._injection_specs):
            if spec["current_name"] != current_name:
                continue
            mask = getattr(self, spec["mask"])
            scale = getattr(self, spec["scale"])
            selected_index = getattr(self, spec["index"])
            value = self.injected_waveforms[k](t) * scale
            # Dendra Waveforms always return time on the last axis.  This
            # method evaluates one timestep, so remove that singleton before
            # applying spatial/batch broadcasting.  Custom modules that return
            # a spatial value directly remain supported.
            if value.ndim > 0 and value.shape[-1] == 1:
                value = value.squeeze(-1)
            out = out + self._expand_injection_value(
                value,
                mask,
                out,
                selected_index=selected_index,
                n_local_selected=int(spec["n_selected"]),
            )

        setattr(self, current_name, out)
        return out


class VoltageProcess(Mechanism):
    """
    Mechanism subtype that updates the membrane potential ``v``.

    Subclasses must implement :meth:`update_v` to return a new membrane potential
    tensor each time step. Both its input and return value are in mV.
    """

    def update_v(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the updated membrane potential. Avoid in-place modification
        to preserve autograd compatibility.

        Parameters
        ----------
        v : torch.Tensor
            Membrane potential tensor in mV to be advanced.

        Returns
        -------
        torch.Tensor
            New membrane potential values in mV.
        """
        raise NotImplementedError(
            "VoltageProcess.update_v() must be implemented in subclasses."
        )


class PointProcess(Mechanism):
    """
    Base class for a mechanism that delivers lumped current at one location.

    A point-process current method returns a numerical value in nA and its
    conductance or voltage derivative is in µS. Voltage remains in mV, so an
    Ohmic method can use ``g * (v - e)`` directly because ``µS * mV = nA``.
    Built-in ``expsyn`` and ``exp2syn`` event weights are consequently numerical
    values in µS (for example, ``weight=0.05`` means 0.05 µS).

    Dendra divides point-process currents and conductances by
    ``1e6 * area_cm2`` before adding them to distributed membrane densities.
    This conversion makes nA into mA/cm² and µS into S/cm². It also means a
    point process cannot be inserted at a zero-area branchpoint.

    Notes
    -----
    Point-process nA/µS values are a local mechanism coordinate convention.
    Do not multiply a built-in point-process weight or conductance parameter by
    :data:`dendra.units.uS`, which converts a value to the absolute-S convention
    used elsewhere. To convert, for example, 50 nS into the required µS
    coordinate, use ``50 * dendra.units.nS / dendra.units.uS``.
    """

    pass


class Synapse(Mechanism):
    """
    A Synapse is a Mechanism that receives spikes and delivers a synaptic current.
    Synapses implement the `net_receive` method, which is called every timestep
    and accepts `weights` (the sum of all incoming weighted spike events at that
    timestep) and `netcon` (the `NetCon` instance for which the synapse is the target)
    as arguments.

    Note that a given synapse may be the target for multiple `NetCon`s, and as such its
    `net_receive` method may be executed multiple times per timestep.

    The `net_receive` method must be overridden in subclasses to implement
    specific behavior for how the synapse responds to incoming spikes.

    ``NetCon`` weights inherit their units from the target synapse. For the
    built-in point-process ``expsyn`` and ``exp2syn`` mechanisms, weights are
    numerical values in µS. A custom density-style ``Synapse`` may define a
    different weight contract and should document it explicitly.
    """

    def net_receive(self, weights, netcon):
        """
        Handle weighted spike arrivals. Avoid in-place modifications of
        state buffers to preserve autograd compatibility.

        Example: increment an internal synaptic conductance ``g_syn`` by the
        incoming weight without in-place ops:

        .. code-block:: python

            def net_receive(self, weights, netcon):
                self.g_syn = self.g_syn + weights

        Parameters
        ----------
        weights : torch.Tensor
            Aggregate weights of incoming spike events at the current step.
        netcon : NetCon
            Connectivity handle delivering the spikes.
        """
        raise NotImplementedError(
            "Synapse.net_receive() must be implemented in subclasses."
        )


class ContinuousSynapse(Mechanism):
    """
    A mechanism that receives continuously valued presynaptic variables.

    ``ContinuousSynapse`` is the analog counterpart to :class:`Synapse`.  It is
    intended for graded transmitter gates, rate-coded projections, neuromodulatory
    drives, and other connections where the presynaptic mechanism emits a
    continuous variable rather than discrete events.

    Subclasses declare continuous input buffers with ``ContinuousSynapse.INPUT``.
    The network's continuous-connection machinery resets those buffers once per
    timestep, then delivers weighted presynaptic values by calling
    :meth:`continuous_receive`.  By default, deliveries are **summed** into the
    named input, which is the natural behavior for convergent synaptic currents.
    The previous timestep's input is also available as ``<input>_old`` when the
    input was declared with the default ``keep_old=True``.

    A ``ContinuousSynapse`` follows the ordinary distributed
    :class:`Mechanism` unit contract unless it also inherits
    :class:`PointProcess`: current in mA/cm² and conductance in S/cm². The
    product of a connection weight and its transformed presynaptic value must
    have the units declared for the target input buffer. For a dimensionless
    presynaptic gate delivered to a conductance-density input, the weight is in
    S/cm².

    Example
    -------

    .. code-block:: python

        class graded_gaba(ContinuousSynapse):
            ContinuousSynapse.INPUT("g_pre")
            ContinuousSynapse.RANGE(e=-80.0)
            ContinuousSynapse.NONSPECIFIC_CURRENT("i")

            def i(self, v):
                return self.g_pre * (v - self.e)
    """

    _continuous_inputs = tuple()
    _continuous_input_old = {}
    _continuous_input_declarations = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        inputs = []
        old_map = {}
        for base in reversed(cls.__mro__):
            if "_continuous_inputs" in base.__dict__:
                inputs.extend(list(base._continuous_inputs))
            if "_continuous_input_old" in base.__dict__:
                old_map.update(dict(base._continuous_input_old))

        declarations = consume_class_values(
            cls,
            "continuous_synapse.inputs",
            ContinuousSynapse._continuous_input_declarations,
        )
        for names, keep_old in declarations:
            for name in names:
                if name not in inputs:
                    inputs.append(name)
                if keep_old:
                    old_map[name] = f"{name}_old"

        cls._continuous_inputs = tuple(inputs)
        cls._continuous_input_old = old_map

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Used by ContinuousCon for optional introspection/debugging.  The
        # current implementation normally resets through a designated first
        # ContinuousCon per target synapse, but this buffer also makes explicit
        # step-aware reset policies possible later.
        self.register_buffer(
            "_continuous_reset_count", torch.zeros((), dtype=torch.long)
        )
        # Per-element delivery history for the current network step.  ContinuousCon
        # scatters into a full synapse-shaped tensor, so a single input-level flag
        # cannot distinguish a genuine zero delivery from an untargeted slot.
        # These masks are ephemeral: Network resets them before every group of
        # continuous deliveries, and they are recreated on the live input device.
        self._continuous_received_masks = {}

    @staticmethod
    def INPUT(*names, keep_old=True):
        """Declare one or more per-step continuous input buffers.

        Parameters
        ----------
        *names : str
            Input buffer names to declare.
        keep_old : bool, optional
            If True (default), also declare ``<name>_old`` buffers and populate
            them with the previous timestep's input during reset.
        """
        if not names:
            raise ValueError("ContinuousSynapse.INPUT requires at least one name.")
        names = tuple(str(n) for n in names)
        assigned = list(names)
        if keep_old:
            assigned.extend(f"{name}_old" for name in names)
        Mechanism.BUFFER(*assigned)
        declare_class_value(
            "continuous_synapse.inputs",
            (names, bool(keep_old)),
            ContinuousSynapse._continuous_input_declarations,
        )

    def reset_continuous_inputs(self):
        """Reset continuous input buffers before new analog deliveries.

        For each declared input ``x``, ``x_old`` is first updated to the current
        value when available, and ``x`` is then reset to zeros.  Rebinding rather
        than in-place mutation keeps the operation compatible with autograd.
        """
        self._continuous_received_masks.clear()
        for name in self._continuous_inputs:
            current = getattr(self, name)
            old_name = self._continuous_input_old.get(name, None)
            if old_name is not None and hasattr(self, old_name):
                setattr(self, old_name, current)
            setattr(self, name, torch.zeros_like(current))
        self._continuous_reset_count = self._continuous_reset_count + 1

    def continuous_receive(self, value, con=None, input=None, reduce=None, mask=None):
        """Receive a continuously valued presynaptic projection.

        Parameters
        ----------
        value : torch.Tensor
            Delivered value in the synapse-local shape.
        con : ContinuousCon, optional
            Connectivity object delivering the value.
        input : str, optional
            Name of the input buffer to update.  If omitted, the first declared
            input is used.
        reduce : {"sum", "set", "max", "min"}, optional
            Reduction used when multiple continuous projections target the same
            input.  Defaults to the connection's ``reduce`` attribute if present,
            otherwise ``"sum"``.
        mask : torch.Tensor, optional
            Boolean tensor identifying elements actually targeted by this
            delivery. If omitted, every element is treated as delivered. This is
            supplied by :class:`ContinuousCon` so its zero-filled scatter slots do
            not participate in non-additive reductions.
        """
        if input is None:
            if len(self._continuous_inputs) != 1:
                raise ValueError(
                    "continuous_receive requires `input=` when the synapse has "
                    f"{len(self._continuous_inputs)} declared inputs."
                )
            input = self._continuous_inputs[0]

        if input not in self._continuous_inputs:
            raise ValueError(
                f"{self.name!r} has no continuous input {input!r}. "
                f"Declared inputs are {self._continuous_inputs!r}."
            )

        if reduce is None:
            reduce = getattr(con, "reduce", "sum")

        current = getattr(self, input)
        value = value.to(device=current.device, dtype=current.dtype)
        if mask is None:
            delivery_mask = torch.ones_like(current, dtype=torch.bool)
        else:
            delivery_mask = torch.as_tensor(
                mask, device=current.device, dtype=torch.bool
            )
            try:
                delivery_mask = torch.broadcast_to(delivery_mask, current.shape)
            except RuntimeError as error:
                raise ValueError(
                    f"Continuous delivery mask with shape {tuple(delivery_mask.shape)} "
                    f"cannot be broadcast to input {input!r} with shape "
                    f"{tuple(current.shape)}."
                ) from error

        received = self._continuous_received_masks.get(input)
        if received is None or tuple(received.shape) != tuple(current.shape):
            received = torch.zeros_like(current, dtype=torch.bool)
        elif received.device != current.device:
            received = received.to(device=current.device)

        if reduce in ("sum", "add"):
            update = torch.where(delivery_mask, value, torch.zeros_like(value))
            setattr(self, input, current + update)
        elif reduce in ("set", "replace", "last"):
            setattr(self, input, torch.where(delivery_mask, value, current))
        elif reduce == "max":
            reduced = torch.maximum(current, value)
            update = torch.where(received, reduced, value)
            setattr(self, input, torch.where(delivery_mask, update, current))
        elif reduce == "min":
            reduced = torch.minimum(current, value)
            update = torch.where(received, reduced, value)
            setattr(self, input, torch.where(delivery_mask, update, current))
        else:
            raise ValueError(f"Unsupported continuous reduction mode: {reduce!r}.")
        self._continuous_received_masks[input] = received | delivery_mask


def rename(mechanism, new_name=None):
    """
    Clone a mechanism class under a new name.

    Parameters
    ----------
    mechanism : type
        Mechanism subclass to be cloned.
    new_name : str, optional
        Name assigned to the cloned class. Defaults to the original name.

    Returns
    -------
    type
        Mechanism subclass with identical behavior but a different ``__name__``.
    """
    # The three-argument form of type(): type(name, bases, dict)
    # 1. name: The new class name (a string).
    # 2. bases: A tuple of the original class's base classes.
    # 3. dict: A dictionary containing the attributes and methods of the
    #          original class. We create a copy to avoid side effects.

    # ``Parameterized.__init_subclass__`` rebuilds and clears declaration-
    # ownership registries on every new class. Preserve the source ownership
    # metadata separately so future subclasses of the alias resolve parameter
    # category precedence exactly as subclasses of the source class do.
    declaration_ownership = {
        name: value.copy()
        for name, value in mechanism.__dict__.items()
        if name.endswith("_defined_here")
    }

    # Copy the original class's namespace dictionary.
    class_dict = dict(mechanism.__dict__)

    # The computed declaration attributes copied above already describe the
    # complete class.  Replaying its original class-body declaration journal
    # through ``__init_subclass__`` would apply non-idempotent declarations
    # (especially current lists) a second time.
    class_dict.pop(_DECLARATIONS_KEY, None)

    # Do not clone generated monomorphic advance functions.  A renamed class may
    # be used alongside the source class; sharing the same generated `_advance`
    # code object would reintroduce cross-class Dynamo guard churn.  The first
    # instance of the renamed class will generate its own fast path.
    if getattr(class_dict.get("_advance"), "_dendra_monomorphic_advance", False):
        class_dict.pop("_advance", None)
    class_dict.pop("_dendra_monomorphic_advance_signature", None)
    class_dict.pop("_dendra_monomorphic_advance_source", None)

    # The __dict__ of a class doesn't always include '__module__',
    # so we copy it over explicitly to make the new class look authentic.
    if "__module__" not in class_dict:
        class_dict["__module__"] = mechanism.__module__

    new_class = type(new_name, mechanism.__bases__, class_dict)
    for name, value in declaration_ownership.items():
        setattr(new_class, name, value)
    # Preserve stable source provenance for symbolic current analysis.  A
    # ``type(...)`` alias has no class statement of its own, but its current
    # methods are exact clones of the source mechanism.  Following this link
    # avoids making symbolic differentiation depend on whether a platform's
    # inspection stack happens to recover the dynamic alias body.
    new_class._dendra_symbolic_source_class = mechanism.__dict__.get(
        "_dendra_symbolic_source_class", mechanism
    )
    new_class._name = new_name  # Set the new name attribute

    return new_class
