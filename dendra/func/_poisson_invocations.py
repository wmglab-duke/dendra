"""Cold capability checks for fixed Poisson waveform invocation profiles."""

from __future__ import annotations

from types import MethodType

import torch
from torch.func import functional_call
from torch.utils import _pytree

from dendra.helpers import current_check_purity

from ._types import FunctionalizationError


def _instrument(module, bindings):
    counts, names, order = {}, {}, []
    for name, path in bindings:
        node = module.get_submodule(path) if path else module
        identity = id(node)
        names.setdefault(identity, []).append(name)
        if identity in counts:
            continue
        counts[identity] = 0
        original = node.fn

        def counted(self, *args, _original=original, _identity=identity, **kwargs):
            del self
            counts[_identity] += 1
            order.append(_identity)
            return _original(*args, **kwargs)

        node.fn = MethodType(counted, node)
    return counts, names, order


def _check(counts, names, order, phase):
    ambiguous = [
        aliases
        for identity, aliases in names.items()
        if counts[identity] and len(aliases) > 1
    ]
    if ambiguous:
        raise FunctionalizationError(
            "Multiple registrations of one physical randomized Poisson waveform "
            f"during {phase} are currently unsupported; registrations={ambiguous}."
        )
    return (
        {
            name: counts[identity]
            for identity, aliases in names.items()
            for name in aliases
        },
        tuple(names[identity][0] for identity in order),
    )


def profile_waveform_invocations(waveform, node_paths, times, *, check_purity=None):
    """Observe registered child calls on a disposable fixed-schedule recipe.

    ``waveform`` is an isolated execution root whose randomized Poisson nodes
    already consume explicit schedules. The returned paths retain the authored
    call order and repetitions, rather than the module registration order.
    Registered operands, including shared tensors, are owned by this probe.
    Class-state and global-RNG diagnostics follow the frozen purity policy;
    the trusted path does not inventory authored Python state.
    """
    from ._population import (
        _authored_workspace_classes,
        _class_state_snapshot,
        _deepcopy_execution_module,
        _global_rng_snapshot,
        _restore_class_state,
        _restore_global_rng,
    )

    device = times.device
    if check_purity is None:
        check_purity = current_check_purity()
    snapshot = _global_rng_snapshot(device) if check_purity else None
    classes = (
        _class_state_snapshot(_authored_workspace_classes(waveform))
        if check_purity
        else None
    )
    try:
        with torch.inference_mode(False), torch.no_grad():
            execution = _deepcopy_execution_module(waveform)
            aliases, operands = {}, {}
            for name, value in (
                *execution.named_parameters(remove_duplicate=False),
                *execution.named_buffers(remove_duplicate=False),
            ):
                if id(value) not in aliases:
                    aliases[id(value)] = value.detach().clone(
                        memory_format=torch.preserve_format
                    )
                operands[name] = aliases[id(value)]
            counts, names, order = _instrument(
                execution, ((path, path) for path in node_paths)
            )
            functional_call(
                execution,
                operands,
                (times.detach().clone(),),
                tie_weights=False,
                strict=True,
            )
            _counts, paths = _check(counts, names, order, "waveform evaluation")
            return paths
    finally:
        try:
            if classes is not None:
                _restore_class_state(classes)
        finally:
            if snapshot is not None:
                _restore_global_rng(snapshot, device)


def validate_poisson_invocations(functional, tensors):
    """Capture a fixed, ordered invocation recipe on disposable operands.

    This is a capability check, independent of optional authored-purity
    diagnostics. Disposable fixed-schedule programs preserve their normal
    numerical call paths while counters observe the number of evaluations.
    Source tensors and random streams are never used as mutable operands.
    """
    plan = functional._poisson_plan
    if plan is None:
        return
    entries = tuple(entry for entry in plan.entries if entry.kind == "mechanism")
    if not entries:
        return
    bindings = tuple((entry.key, entry.mapping.rsplit(".", 1)[0]) for entry in entries)
    from ._population import (
        _authored_workspace_classes,
        _class_state_snapshot,
        _deepcopy_execution_module,
        _functional_probe_restore_scope,
        _global_rng_snapshot,
        _initialization_voltage_kwargs,
        _rebind_population_time_references,
    )

    snapshot = (
        _global_rng_snapshot(functional.device) if functional._check_purity else None
    )
    classes = (
        _class_state_snapshot(_authored_workspace_classes(functional._transition))
        if functional._check_purity
        else None
    )
    with _functional_probe_restore_scope(functional, classes, snapshot):
        with torch.inference_mode(False), torch.no_grad():
            parameters, constants, state = _pytree.tree_map_only(
                torch.Tensor,
                lambda value: value.detach().clone(),
                (tensors.parameters, tensors.constants, tensors.state),
            )
            prepared = functional._prepare_values(parameters, constants)
            mapping, dt, celsius, voltage, time, remainder = functional._mapping(
                parameters, prepared, state, isolate_read_only=True
            )
            transition = _deepcopy_execution_module(functional._transition)
            _rebind_population_time_references(transition.population)
            counts, names, order = _instrument(transition, bindings)
            functional_call(
                transition,
                mapping,
                (
                    voltage,
                    time,
                    remainder,
                    dt,
                    celsius,
                    None,
                    None,
                    1,
                    None,
                    prepared.get("noise_workspaces"),
                ),
                tie_weights=False,
            )
            runtime, runtime_order = _check(counts, names, order, "an accepted step")
            initialization_profile = {entry.key: 0 for entry in entries}
            initialization_order = ()
            if functional._initialization is not None:
                initialization = _deepcopy_execution_module(functional._initialization)
                _rebind_population_time_references(initialization.population)
                counts, names, order = _instrument(initialization, bindings)
                # Rebuild every mapping tensor after the step so even a trusted
                # authored in-place write cannot become initializer input.
                parameters, constants, state = _pytree.tree_map_only(
                    torch.Tensor,
                    lambda value: value.detach().clone(),
                    (tensors.parameters, tensors.constants, tensors.state),
                )
                state["clock"] = {
                    **state["clock"],
                    "t": torch.zeros_like(state["clock"]["t"]),
                }
                prepared = functional._prepare_values(parameters, constants)
                mapping, *_rest = functional._mapping(
                    parameters, prepared, state, isolate_read_only=True
                )
                functional_call(
                    initialization,
                    mapping,
                    (
                        tensors.initialization.v_init.detach().clone(),
                        *(
                            tensors.initialization.states[entry.name].detach().clone()
                            for entry in initialization.explicit_layout
                        ),
                    ),
                    _initialization_voltage_kwargs(state),
                    tie_weights=False,
                )
                initialization_profile, initialization_order = _check(
                    counts, names, order, "fresh initialization"
                )
            plan.set_invocation_profile(
                runtime=runtime,
                initialization=initialization_profile,
                runtime_order=runtime_order,
                initialization_order=initialization_order,
            )
