"""Experimental tensor-only single-step boundary for canonical HH + CPU DHS.

This is deliberately private and opt-in. The ordinary imperative runners own
stimulation, hooks, the clock, and publication. Only an observer-stable runner
segment may create a binding; unsupported models keep the established kernel.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import MethodType

import torch
from torch.func import functional_call
from torch.fx.experimental.proxy_tensor import make_fx

from dendra._bootstrap import torch_compiler_warning_context
from dendra.helpers import MEMBRANE_ASSEMBLY


def _layout(value):
    return (tuple(value.shape), tuple(value.stride()), value.dtype, value.device)


def _function_key(value):
    value = getattr(value, "__func__", value)
    return getattr(value, "__code__", None)


def _identity_alias_pattern(operands):
    """Describe input identity sharing without pinning process-local IDs."""
    first_position = {}
    return tuple(
        first_position.setdefault(id(value), index)
        for index, value in enumerate(operands)
    )


def _stock_instance(value, owner):
    """Permit only an unchanged Referency clock proxy above the stock owner.

    Creating the proxy runs Mechanism/Parameterized.__init_subclass__, which
    copies declaration dictionaries onto it. Those are metadata, not authored
    overrides; compare their effective declarations instead of rejecting them.
    """
    from ..parametric import staticproperty

    declarations = {
        "_params",
        "_params_p",
        "_params_n",
        "_flags",
        "_global",
        "_global_p",
        "_global_n",
        "_range",
        "_range_p",
        "_range_n",
        "_batch",
        "_batch_p",
        "_batch_n",
        "_rng",
        "_random_parameters",
        "_runtime_noises",
        "_table",
        "state_classes",
        "_state",
        "_ion",
        "_material",
        "_save",
        "_currents",
        "_carry",
        "_carry_specs",
        "_assigned",
        "_derived_buffers",
        "_derived_buffer_specs",
        "_timestep_buffers",
        "_timestep_buffer_shapes",
        "_read_ion",
        "_write_ion",
        "_write_ion_c",
        "_read_material",
        "_write_material",
        "_source_material",
        "_explicit",
        "_numerical",
        "_affine",
        "_affine_method_owners",
        "_name",
    }
    for cls in type(value).__mro__:
        if cls is owner:
            return True
        base = cls.__bases__[0]
        if (
            cls.__module__ != "dendra.models.parametric"
            or len(cls.__bases__) != 1
            or cls.__name__ != "_" + base.__name__ + "Proxy"
        ):
            return False
        for name, item in vars(cls).items():
            if name in {"__module__", "__doc__"}:
                continue
            if name == "t":
                if type(item) is not staticproperty:
                    return False
            elif name == "_renamed_aliases":
                # Mechanism.__init_subclass__ gives each clock proxy a fresh
                # rename lookup cache. Warming the stock owner's cache does
                # not change this instance's declarations or executable hooks.
                if type(item) is not dict:
                    return False
            elif name.endswith("_defined_here"):
                if type(item) not in (dict, tuple, list, set) or item:
                    return False
            elif name not in declarations or item != getattr(owner, name, None):
                return False
    return False


def _source_method(value, name, owner):
    if name in vars(value):
        return False
    resolved = getattr(type(value), name, None)
    expected = getattr(owner, name, None)
    return (
        resolved is expected
        and inspect.isfunction(expected)
        and inspect.getfile(expected) == inspect.getfile(owner)
    )


def _instance_dispatch_plain(module, *, allowed=()):
    """Do not execute instance-authored Python once during FX capture."""
    return all(
        name in allowed or not callable(value) or isinstance(value, torch.nn.Module)
        for name, value in vars(module).items()
    )


def _make_tensor_step_binding(integrator):
    """Fail closed unless the enclosing segment proved canonical execution."""
    from ..mechanisms._handler import MechanismHandler
    from ..mod.HH import hh, mhn
    from .tree import _dhs

    if (
        not integrator._tensor_step_enabled
        or not integrator.jit
        or integrator.compile_scope != "population"
        or integrator.compile_device_type != "cpu"
        or integrator.imem
        or MEMBRANE_ASSEMBLY.value != "auto"
        or not integrator._direct_membrane_structure_prevalidated
        or not isinstance(integrator, _dhs)
        or not _source_method(integrator, "step", _dhs)
        or not _source_method(integrator, "_step", _dhs)
        or integrator._cpu_topology_plan is None
        or integrator._solve is not torch.ops.dendra_solvers.dhs_solve
        or not _instance_dispatch_plain(integrator, allowed=("_solve",))
    ):
        return None
    handler = integrator.mech
    if (
        type(handler) is not MechanismHandler
        or len(handler.mechanisms) != 1
        or not _instance_dispatch_plain(handler)
    ):
        return None
    name, mechanism = next(iter(handler.mechanisms.items()))
    if (
        not _stock_instance(mechanism, hh)
        or not _source_method(mechanism, "i_affine_terms", hh)
        or mechanism._save
        or mechanism._injection_specs
        or mechanism._delayed_state_specs
        or len(mechanism.DE) != 1
        or not _instance_dispatch_plain(mechanism, allowed=("i_with_g",))
        or mechanism._affine_evaluators != {"i": "i_affine_terms"}
        or mechanism._has_authored_advance
        or mechanism._has_authored_assigned_values
        or tuple(mechanism._transition_input_names) != ("m", "h", "n")
    ):
        return None
    state = next(iter(mechanism.DE.values()))
    if (
        not _stock_instance(state, mhn)
        or not _source_method(state, "assigned_values", mhn)
        or state.method != "cnexp"
        or state.method_kwargs not in ({}, {"pade": False})
        or not _instance_dispatch_plain(state, allowed=("_solve",))
        or type(state._solve) is not MethodType
        or state._solve.__self__ is not state
        or not getattr(state._solve.__func__, "__generated_filename__", "").startswith(
            "<dendra.cnexp.solve:"
        )
        or state._solve.__func__.__code__.co_filename
        != state._solve.__func__.__generated_filename__
        or not isinstance(getattr(state._solve.__func__, "__source__", None), str)
        or state._has_authored_advance
        or not state._has_authored_assigned_values
        or any(getattr(module, "runtime_noises", None) for module in handler.modules())
        or any(
            getattr(module, "parametrizations", None) for module in handler.modules()
        )
        or any(
            getattr(module, "random_parameters", None) for module in handler.modules()
        )
    ):
        return None
    for module, stock in ((mechanism, hh), (state, mhn)):
        if any(
            getattr(module, field) != getattr(stock, field)
            for field in (
                "_state",
                "_carry",
                "_assigned",
                "_derived_buffers",
                "_timestep_buffers",
            )
        ):
            return None
    values = tuple(integrator.buffers()) + tuple(integrator.parameters())
    if any(
        type(value) not in (torch.Tensor, torch.nn.Parameter)
        or value.device.type != "cpu"
        or (
            value.is_floating_point()
            and value.dtype not in (torch.float32, torch.float64)
        )
        for value in values
    ):
        return None
    return _TensorStepBinding(integrator, name)


class _CaptureStep(torch.nn.Module):
    def __init__(self, integrator, mechanism_name):
        super().__init__()
        self.integrator = integrator
        self.mechanism_name = mechanism_name

    def forward(self, v, dt, temp, token, intra):
        # Supply the lifetime token explicitly instead of capturing a mutable
        # plan owner. Only ordinary reverse AD is promised by this prototype.
        def solve(diagonal, axial, rhs, _parent, _order, _layers):
            return torch.ops.dendra_solvers.dhs_solve_plan(diagonal, axial, rhs, token)

        voltage, _current = self.integrator._step(
            v, dt, temp, None, intra, solver=solve
        )
        mechanism = self.integrator.mech.mechanisms[self.mechanism_name]
        return voltage, mechanism.m, mechanism.h, mechanism.n


@dataclass
class _CapturedTensorStep:
    graph: object
    indices: tuple[int, ...]
    signature: tuple


class _TensorStepBinding:
    """One segment's eager operand readers and tensor-result publication."""

    def __init__(self, integrator, mechanism_name):
        self.integrator = integrator
        self.mechanism_name = mechanism_name
        modules = dict(integrator.named_modules(remove_duplicate=False))
        self.slots = tuple(
            (name, modules[path], local)
            for name, _value in (
                *integrator.named_parameters(remove_duplicate=False),
                *integrator.named_buffers(remove_duplicate=False),
            )
            for path, _separator, local in (name.rpartition("."),)
        )
        self.successful_calls = 0
        self.capture_count = 0
        self.last_graph = None
        self.last_operand_count = 0
        self._dispatch = {}
        self._active = None
        self._active_source = None
        self._active_indices = None
        self._active_index_values = ()
        self._active_mode = None
        self._compiled_call = None
        self._alias_rejected = False
        self._selected_heads = ()
        self._selected_slots = ()
        self._selected_tail = ()
        self._slot_readers = tuple(
            (owner._parameters if local in owner._parameters else owner._buffers, local)
            for _name, owner, local in self.slots
        )

    def _read_slots(self):
        return tuple(registry[name] for registry, name in self._slot_readers)

    def _state_operands_unaliased(self, model, operands):
        """Reject evolving-state aliases that split after eager publication.

        make_fx can specialize identical inputs to one placeholder. Ordinary
        HH publication produces separate voltage/gate tensors, so such an
        input cannot be treated as a persistent alias through the recurrence.
        Multiple registered paths to one owner/field are the same logical
        state and remain valid aliases.
        """
        mechanism = self.integrator.mech.mechanisms[self.mechanism_name]
        evolving = {id(model.v): ("voltage",)}
        for gate in ("m", "h", "n"):
            value = mechanism._buffers[gate]
            if id(value) in evolving:
                return False
            evolving[id(value)] = ("slot", id(mechanism), gate)
        logical = (
            ("voltage",),
            ("dt",),
            ("temperature",),
            ("token",),
            *(("slot", id(owner), name) for _path, owner, name in self.slots),
            *(("drive", index) for index in range(len(operands) - 4 - len(self.slots))),
        )
        return all(
            id(value) not in evolving or evolving[id(value)] == identity
            for value, identity in zip(operands, logical, strict=True)
        )

    def _execute(self, captured, key, model, dt, tail):
        head = (
            model.v,
            dt,
            model.celsius,
            self.integrator._cpu_topology_plan._token,
        )
        # Placeholder order is the original flat-input order: explicit head,
        # registered slots, then drive leaves. Selection was lowered once; all
        # used numerical tensors are nevertheless reread live on every step.
        operands = (
            *(head[index] for index in self._selected_heads),
            *(registry[name] for registry, name in self._selected_slots),
            *(tail[index] for index in self._selected_tail),
        )
        with torch_compiler_warning_context():
            compiled = self._compiled_call
            if compiled is None:
                compiled = self.integrator._compiled_kernels.get(key)
                if compiled is None:
                    compiled = torch.compile(
                        captured.graph.forward, **self.integrator._compile_kwargs()
                    )
                    self.integrator._compiled_kernels[key] = compiled
                self._compiled_call = compiled
            voltage, m, h, n = compiled(*operands)
        # Rebind, never detach/copy into old storage: prior timesteps and
        # autograd graphs retain their exact ownership and history.
        mechanism = self.integrator.mech.mechanisms[self.mechanism_name]
        mechanism._buffers.update(m=m, h=h, n=n)
        model.v = voltage
        self.successful_calls += 1
        self.last_graph = captured.graph
        self.last_operand_count = len(captured.indices)
        return True

    def _capture(self, operands, source, counts, signature):
        from dendra.func._population import _deepcopy_execution_module

        # No source state is executed during lowering. The disposable facade's
        # complete registered tensor mapping is explicit to make_fx.
        with torch.inference_mode(False), torch.no_grad():
            cloned = _deepcopy_execution_module(self.integrator)
            cloned._direct_membrane_structure_prevalidated = True
            facade = _CaptureStep(cloned, self.mechanism_name)
            names = tuple("integrator." + name for name, _owner, _local in self.slots)
            slot_count = len(names)

            def authored(*flat):
                mapping = dict(zip(names, flat[4 : 4 + slot_count], strict=True))
                tail = flat[4 + slot_count :]
                if source is None:
                    intra = tail[0] if counts == "tensor" else None
                else:
                    stimulus_count, index_counts = counts
                    stimuli = list(tail[:stimulus_count])
                    cursor = stimulus_count
                    indices = []
                    for count in index_counts:
                        indices.append(tuple(tail[cursor : cursor + count]))
                        cursor += count
                    intra = source(stimuli, indices)
                return functional_call(
                    facade,
                    mapping,
                    (*flat[:4], intra),
                    tie_weights=False,
                )

            graph = make_fx(authored)(*operands)
        # Prune unused module operands before compiling. No module tree enters
        # the repeated numerical call, including its guard inventory.
        indices = []
        position = 0
        for node in tuple(graph.graph.nodes):
            if node.op == "placeholder":
                if node.users:
                    indices.append(position)
                else:
                    graph.graph.erase_node(node)
                position += 1
            elif node.op == "get_attr":
                raise RuntimeError(
                    "tensor-step lowering retained an implicit graph constant: "
                    + str(node.target)
                )
        graph.graph.lint()
        graph.recompile()
        self.capture_count += 1
        return _CapturedTensorStep(graph, tuple(indices), signature)

    def try_step(self, model, dt, ve, intra):
        from ..stim.intra import Intra
        from .core import _DeferredIntra, _NoiseFreeStep

        if (
            ve is not None
            or not self.integrator._tensor_step_enabled
            or self._alias_rejected
        ):
            return False
        if isinstance(intra, _NoiseFreeStep):
            intra = intra.intra
        if self._active is not None:
            # Eligibility is segment-local and forbids authored mutations.
            # Dynamo still checks every actual operand. Avoid rebuilding the
            # source schema or visiting unused buffers in the repeated step.
            if self._active_mode == "deferred":
                if (
                    not isinstance(intra, _DeferredIntra)
                    or intra.source is not self._active_source
                    or intra.indices is not self._active_indices
                ):
                    return False
                tail = tuple(intra.stimuli) + self._active_index_values
            elif self._active_mode == "tensor":
                if not torch.is_tensor(intra):
                    return False
                tail = (intra,)
            elif intra is None:
                tail = ()
            else:
                return False
            return self._execute(*self._active, model, dt, tail)
        source = None
        tail = ()
        counts = None
        drive_signature = None
        if isinstance(intra, _DeferredIntra):
            source = intra.source
            if type(source) is not Intra or not _source_method(
                source, "__call__", Intra
            ):
                return False
            if type(intra.stimuli) is not list or type(intra.indices) is not list:
                return False
            counts = (len(intra.stimuli), tuple(len(item) for item in intra.indices))
            tail = tuple(intra.stimuli) + tuple(
                value for item in intra.indices for value in item
            )
            drive_signature = (
                "deferred",
                tuple(source.shape),
                tuple(source.batch_shape),
                source.dtype,
                source.device,
                tuple(tuple(item) for item in source.selected_batch_shapes),
                counts,
            )
        elif torch.is_tensor(intra):
            tail = (intra,)
            counts = "tensor"
            drive_signature = "tensor"
        elif intra is not None:
            return False
        if any(type(value) not in (torch.Tensor, torch.nn.Parameter) for value in tail):
            return False
        token = self.integrator._cpu_topology_plan._token
        if not self.integrator.mech._direct_membrane_dtype_compatible(model.v):
            return False
        operands = (model.v, dt, model.celsius, token, *self._read_slots(), *tail)
        if any(
            type(value) not in (torch.Tensor, torch.nn.Parameter) for value in operands
        ):
            return False
        if any(
            value.is_floating_point() and value.dtype != model.v.dtype
            for value in operands
        ):
            # A mixed G/J result could select legacy current assembly, whose
            # scratch publication and constructed current method are not part
            # of this four-output tensor-only contract.
            return False
        if not self._state_operands_unaliased(model, operands):
            # Keep the whole segment on its general path: the first fallback
            # step itself can split the initial aliases and must not authorize
            # a different capture implicitly in the middle of the recurrence.
            self._alias_rejected = True
            return False
        # All numerical coefficients, indices and plan lifetime tokens remain
        # live inputs. Only Python-specialized structure keys graph reuse.
        mechanism = self.integrator.mech.mechanisms[self.mechanism_name]
        state = next(iter(mechanism.DE.values()))
        signature = (
            self.mechanism_name,
            self.integrator.B,
            self.integrator.K,
            tuple(self.integrator.base_shape),
            mechanism.support_spec,
            _function_key(mechanism._advance_states),
            _function_key(state._solve),
            _function_key(mechanism.i_affine_terms),
            tuple(state._state),
            tuple(state._assigned),
            tuple(state._carry),
            tuple(name for name, _owner, _local in self.slots),
            drive_signature,
            tuple(_layout(value) for value in operands),
            _identity_alias_pattern(operands),
        )
        captured = self._dispatch.get(signature)
        if captured is None:
            graphs = self.integrator.__dict__.setdefault("_tensor_step_graphs", {})
            errors = self.integrator.__dict__.setdefault(
                "_tensor_step_capture_errors", {}
            )
            if signature in errors:
                return False
            captured = graphs.get(signature)
            if captured is None:
                try:
                    captured = self._capture(operands, source, counts, signature)
                except Exception as exc:
                    errors[signature] = f"{type(exc).__name__}: {exc}"
                    return False
                graphs[signature] = captured
            self._dispatch[signature] = captured
        key = (
            "forward",
            "tensor_step",
            signature,
            self.integrator.compile_scope,
            self.integrator.backend,
            self.integrator.fullgraph,
            self.integrator.dynamic,
            self.integrator.compile_mode,
            self.integrator.compile_options_key,
        )
        self._active = (captured, key)
        slot_end = 4 + len(self._slot_readers)
        self._selected_heads = tuple(index for index in captured.indices if index < 4)
        self._selected_slots = tuple(
            self._slot_readers[index - 4]
            for index in captured.indices
            if 4 <= index < slot_end
        )
        self._selected_tail = tuple(
            index - slot_end for index in captured.indices if index >= slot_end
        )
        if source is not None:
            self._active_mode = "deferred"
            self._active_source = source
            self._active_indices = intra.indices
            self._active_index_values = tuple(
                value for item in intra.indices for value in item
            )
        else:
            self._active_mode = counts
        return self._execute(captured, key, model, dt, tail)
