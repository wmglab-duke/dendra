"""Static registrations and guards for explicit Mechanism delay histories."""

import copy
from types import MethodType

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._mechanism_delays import (
    MechanismDelayPlan,
    mechanism_delay_constant_slots,
    mechanism_delay_failures,
    mechanism_delay_helper,
    mechanism_delay_owners,
    mechanism_delay_state_slots,
)
from dendra.models.mechanisms import Mechanism, _delays

pytestmark = pytest.mark.cpu


class _DelayedCurrent(Mechanism):
    Mechanism.GLOBAL(g=0.001)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")
    supports_population_axis_multistream_delays = True

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * 0

    def advance(self, v, dt, values):
        del dt
        _delayed, single = self.delayed_state_values("signal", v, values)
        _delayed, multi = self.delayed_states_values(
            "streams", torch.stack((v, v)), values
        )
        return {**single, **multi}


def _population():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        population = dn.SingleCompartment(C=2, dtype=torch.float64)
        population.insert(_DelayedCurrent.rename("delayed"))
        population.initialize()
        owner = population.mech.delayed
        owner.register_delayed_state("signal", population.v, 2, mode="shift")
        owner.register_delayed_states(
            "streams",
            torch.stack((population.v, population.v)),
            (0, 2),
            stream_axis=0,
            mode="shift",
        )
    return population


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch_compiler_warning_context(), dn.ctx(JIT=0, REQUIRE_GRAD=1):
            yield
    finally:
        torch.set_num_threads(previous)


def test_delay_plan_classifies_registered_history_and_integer_constants():
    population = _population()
    plan = MechanismDelayPlan(population)
    owners = mechanism_delay_owners(population)
    assert len(owners) == 1
    path, owner = owners[0]
    assert path == "integrator.mech.mechanisms.delayed"
    assert not mechanism_delay_failures(owner)
    assert tuple(dict(mechanism_delay_state_slots(owner))) == (
        "signal_delay_buffer",
        "signal_delay_ptr",
        "streams_delay_buffer",
        "streams_delay_ptr",
    )
    assert tuple(dict(mechanism_delay_constant_slots(owner))) == (
        "streams_delay_steps",
    )
    assert plan.constant_entries == ((path, "streams_delay_steps", (0, 2)),)
    assert mechanism_delay_owners(population, ()) == ()
    plan.validate_source()
    plan.validate_source_tensor_values()
    for name in (
        "delayed_state_values",
        "delayed_states_values",
        "initial_delay_values",
    ):
        assert mechanism_delay_helper(owner, getattr(owner, name)) == name
    assert mechanism_delay_helper(owner, owner.delayed_state) is None


def test_delay_plan_accepts_compatible_clones_and_advanced_history_values():
    population = _population()
    plan = MechanismDelayPlan(population)
    clone = copy.deepcopy(population)
    owner = clone.mech.delayed
    owner.signal_delay_buffer = owner.signal_delay_buffer + 2
    owner.streams_delay_buffer = owner.streams_delay_buffer - 3
    owner.signal_delay_ptr = torch.ones_like(owner.signal_delay_ptr)
    plan.validate_population(clone)
    plan.validate_population(_population())


@pytest.mark.parametrize(
    "change", ("delay", "mode", "axis", "name", "capacity", "stream", "frame")
)
def test_delay_plan_rejects_changed_metadata_or_capacity(change):
    population = _population()
    plan = MechanismDelayPlan(population)
    owner = population.mech.delayed
    if change == "delay":
        owner.register_delayed_state("signal", population.v, 1, mode="shift")
    elif change == "mode":
        owner._delayed_state_specs["signal"]["mode"] = "circular"
    elif change == "axis":
        owner._delayed_state_specs["signal"]["axis"] = 0
    elif change == "name":
        owner._delayed_state_specs["signal"]["buffer"] = "replacement"
    elif change == "capacity":
        owner.signal_delay_buffer = owner.signal_delay_buffer[..., :1, :]
    elif change == "stream":
        owner._delayed_state_specs["streams"]["value_stream_axis"] = 1
    else:
        owner._delay_history_buffer_names = ()
    with pytest.raises(dn.func.FunctionalizationError, match="[Dd]elay|history|stream"):
        plan.validate_source()


@pytest.mark.parametrize(
    "change",
    (
        "pointer_shape",
        "pointer_dtype",
        "pointer_range",
        "steps_dtype",
        "steps_shape",
        "steps_negative",
        "steps_flags",
    ),
)
def test_delay_plan_rejects_malformed_registered_tensors(change):
    population = _population()
    owner = population.mech.delayed
    if change == "pointer_shape":
        owner.signal_delay_ptr = torch.ones(1, dtype=torch.long)
    elif change == "pointer_dtype":
        owner.signal_delay_ptr = torch.zeros((), dtype=torch.float64)
    elif change == "pointer_range":
        owner.signal_delay_ptr.fill_(3)
    elif change == "steps_dtype":
        owner.streams_delay_steps = owner.streams_delay_steps.double()
    elif change == "steps_shape":
        owner.streams_delay_steps = owner.streams_delay_steps.reshape(1, 2)
    elif change == "steps_negative":
        owner.streams_delay_steps[0] = -1
    else:
        owner.streams_delay_steps[0] = 1
    with pytest.raises(dn.func.FunctionalizationError, match="[Dd]elay|pointer"):
        MechanismDelayPlan(population)


@pytest.mark.parametrize("change", ("merge", "split"))
def test_delay_plan_preserves_history_alias_routing(change):
    population = _population()
    owner = population.mech.delayed
    owner.register_buffer("extra_history", owner.signal_delay_buffer.clone())
    if change == "split":
        owner.extra_history = owner.signal_delay_buffer
    plan = MechanismDelayPlan(population)
    owner.extra_history = (
        owner.signal_delay_buffer if change == "merge" else owner.extra_history.clone()
    )
    with pytest.raises(dn.func.FunctionalizationError, match="aliases"):
        plan.validate_source()


@pytest.mark.parametrize(
    "name",
    (
        "delayed_state_values",
        "delayed_states_values",
        "initial_delay_values",
        "_delay_values_spec",
        "register_delayed_state",
    ),
)
def test_delay_plan_rejects_custom_helper_adapters(name):
    population = _population()
    owner = population.mech.delayed
    setattr(owner, name, MethodType(lambda _self, *args, **kwargs: args, owner))
    with pytest.raises(dn.func.FunctionalizationError, match="canonical"):
        MechanismDelayPlan(population)


@pytest.mark.parametrize("location", ("shared_binding", "native_binding", "code"))
def test_delay_plan_guards_shared_numerical_implementation(location, monkeypatch):
    population = _population()
    plan = MechanismDelayPlan(population)
    helper = _delays.shifted_history_values

    def replacement(buffer, value, *, axis):
        return buffer

    if location == "shared_binding":
        monkeypatch.setattr(_delays, "shifted_history_values", replacement)
    elif location == "native_binding":
        monkeypatch.setitem(
            Mechanism.delayed_state_values.__globals__,
            "shifted_history_values",
            replacement,
        )
    else:
        monkeypatch.setattr(helper, "__code__", replacement.__code__)
    with pytest.raises(dn.func.FunctionalizationError, match="canonical"):
        plan.validate_source()


def test_delay_plan_rejects_effectful_metadata_without_calling_it():
    class Metadata:
        calls = 0

        def __bool__(self):
            self.calls += 1
            return True

        def __eq__(self, _other):
            self.calls += 1
            return True

    population = _population()
    plan = MechanismDelayPlan(population)
    before = plan.source_binding_signature()
    metadata = Metadata()
    population.mech.delayed._delayed_state_specs["streams"]["all_zero_delay"] = metadata
    assert plan.source_binding_signature() != before
    with pytest.raises(dn.func.FunctionalizationError, match="literal zero-delay"):
        plan.validate_source()
    assert metadata.calls == 0


@pytest.mark.parametrize("change", ("values", "metadata", "helper", "shared"))
def test_warmed_compiled_delay_contract_guards_reject_changes(change, monkeypatch):
    population = _population()
    plan = MechanismDelayPlan(population)
    owner = population.mech.delayed
    expected = plan.source_binding_signature()

    def evaluate(value):
        if plan.source_binding_signature() != expected:
            raise dn.func.FunctionalizationError(
                "Mechanism delayed-state binding changed"
            )
        plan.validate_source_tensor_values()
        return value + 1

    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(torch.zeros(2)), torch.ones(2))
    if change == "values":
        owner.streams_delay_steps[0] = 1
    elif change == "metadata":
        owner._delayed_state_specs["signal"]["mode"] = "circular"
    elif change == "helper":
        owner.delayed_state_values = MethodType(lambda self, *args: args, owner)
    else:
        monkeypatch.setattr(
            _delays, "shifted_history_values", lambda *args, **kwargs: args
        )
    with pytest.raises(
        (RuntimeError, dn.func.FunctionalizationError, torch._dynamo.exc.Unsupported)
    ):
        compiled(torch.zeros(2))


def test_warmed_explicit_integer_constants_reject_changed_values():
    population = _population()
    plan = MechanismDelayPlan(population)
    path, name, _expected = plan.constant_entries[0]
    slot = f"{path}.{name}"

    def evaluate(steps, value):
        plan.validate_constant_values({slot: steps})
        return value + 1

    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    torch.testing.assert_close(
        compiled(torch.tensor((0, 2)), torch.zeros(2)), torch.ones(2)
    )
    with pytest.raises(RuntimeError):
        compiled(torch.tensor((1, 2)), torch.zeros(2))


def test_explicit_delay_values_and_compatible_models_preserve_native_schema():
    population = _population()
    plan = MechanismDelayPlan(population)
    path, name, _expected = plan.constant_entries[0]
    slot = f"{path}.{name}"
    alternate = torch.tensor((2, 0), dtype=torch.long)
    plan.validate_constant_values({slot: alternate})
    clone = copy.deepcopy(population)
    clone.mech.delayed.streams_delay_steps = alternate
    plan.validate_population(clone)

    def evaluate(value):
        plan.validate_constant_values({slot: value})
        return value + 1

    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(alternate), alternate + 1)
    population.mech.delayed.streams_delay_steps = alternate
    with pytest.raises(dn.func.FunctionalizationError, match="source delay vector"):
        plan.validate_source()


@pytest.mark.parametrize("delays", ((-1, 2), (0, 1), (1, 2), (0, 0)))
def test_explicit_delay_values_reject_invalid_capacity_and_zero_schema(delays):
    plan = MechanismDelayPlan(_population())
    path, name, _expected = plan.constant_entries[0]
    slot = f"{path}.{name}"
    with pytest.raises(RuntimeError, match="capacity and zero-delay"):
        plan.validate_constant_values({slot: torch.tensor(delays)})


@pytest.mark.parametrize("operation", ("step", "rollout"))
@pytest.mark.parametrize(
    "change", ("steps", "mode", "helper", "core", "code", "alias", "training", "frame")
)
def test_compiled_population_delay_guards_reject_source_changes(
    operation, change, monkeypatch
):
    population = _population()
    functional, tensors = dn.func.make_functional(population, dt=0.125)

    def evaluate(parameters, constants, state):
        if operation == "step":
            return functional.prepare_and_step(parameters, constants, state)
        return functional.prepare_and_rollout(parameters, constants, state, steps=2)

    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    compiled(tensors.parameters, tensors.constants, tensors.state)
    before = copy.deepcopy(tensors)
    owner = population.mech.delayed
    if change == "steps":
        owner.streams_delay_steps[0] = 1
    elif change == "mode":
        owner._delayed_state_specs["signal"]["mode"] = "circular"
    elif change == "helper":
        owner.delayed_state_values = MethodType(lambda self, *args: args, owner)
    elif change == "core":
        monkeypatch.setattr(
            _delays, "shifted_history_values", lambda *args, **kwargs: args
        )
    elif change == "code":

        def replacement(buffer, value, *, axis):
            return buffer

        monkeypatch.setattr(
            _delays.shifted_history_values, "__code__", replacement.__code__
        )
    elif change == "alias":
        owner.streams_delay_ptr = owner.signal_delay_ptr
    elif change == "training":
        population.train(not owner.training)
    else:
        owner._transition_input_names = owner._transition_input_names[:-1]
    with pytest.raises(
        (RuntimeError, dn.func.FunctionalizationError, torch._dynamo.exc.Unsupported)
    ):
        compiled(tensors.parameters, tensors.constants, tensors.state)
    torch.testing.assert_close(tensors, before)


@pytest.mark.parametrize(
    "change",
    ("class", "forward", "backward", "jvp", "setup_context", "vmap_rule", "wrapper"),
)
def test_delay_plan_guards_circular_gradient_contract(change, monkeypatch):
    population = _population()
    plan = MechanismDelayPlan(population)
    if change == "class":
        monkeypatch.setattr(_delays, "_OverwriteHistory", type("Replacement", (), {}))
    elif change == "vmap_rule":
        monkeypatch.setattr(_delays._OverwriteHistory, "generate_vmap_rule", False)
    elif change == "wrapper":
        monkeypatch.setattr(
            _delays, "_overwrite_history", lambda current, updated: updated
        )
    else:
        monkeypatch.setattr(
            _delays._OverwriteHistory, change, staticmethod(lambda *args: args)
        )
    with pytest.raises(dn.func.FunctionalizationError, match="canonical"):
        plan.validate_source()


@pytest.mark.parametrize("change", ("vmap_rule", "backward"))
def test_warmed_circular_gradient_contract_guards(change, monkeypatch):
    population = _population()
    plan = MechanismDelayPlan(population)
    expected = plan.source_binding_signature()

    def evaluate(value):
        if plan.source_binding_signature() != expected:
            raise dn.func.FunctionalizationError(
                "Mechanism delayed-state gradient binding changed"
            )
        return value + 1

    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    compiled(torch.zeros(2))
    if change == "vmap_rule":
        monkeypatch.setattr(_delays._OverwriteHistory, "generate_vmap_rule", False)
    else:
        monkeypatch.setattr(
            _delays._OverwriteHistory,
            "backward",
            staticmethod(lambda ctx, gradient: (gradient, None)),
        )
    with pytest.raises((dn.func.FunctionalizationError, torch._dynamo.exc.Unsupported)):
        compiled(torch.zeros(2))


@pytest.mark.parametrize("training", (False, True))
@pytest.mark.parametrize("grad_enabled", (False, True))
def test_scan_preserves_native_auto_delay_training_and_grad_policy(
    training, grad_enabled
):
    population, native = _population(), _population()
    for model in (population, native):
        model.train(training)
        for spec in model.mech.delayed._delayed_state_specs.values():
            spec["mode"] = "auto"
        model.integrator._initialize(model, model.t.new_tensor(0.125), force=True)
        model.detach()
    runner = dn.compile(
        population,
        dt=0.125,
        steps=4,
        execution="scan",
        backend="aot_eager",
        inference=not grad_enabled,
    )
    with torch.set_grad_enabled(grad_enabled):
        owner = population.mech.delayed
        _delayed, updates = owner.delayed_state_values(
            "signal", population.v, owner._runtime_value_frame()
        )
        circular = not (training or grad_enabled)
        torch.testing.assert_close(
            updates["signal_delay_ptr"],
            owner.signal_delay_ptr.new_tensor(int(circular)),
        )
        actual, _auxiliary = runner()
        assert torch.is_grad_enabled() is grad_enabled
        for _ in range(4):
            # Functional execution owns the caller's grad context. Native
            # Population.step additionally disables grads in eval mode, so
            # compare the shared integrator in the same execution context.
            dt = native.t.new_tensor(0.125)
            native.integrator.step(native, dt)
            native.t = native.t + dt
        expected = runner.functional.extract(native).state
        torch.testing.assert_close(actual, expected)
        assert torch.is_grad_enabled() is grad_enabled


@pytest.mark.parametrize("grad_enabled", (False, True))
def test_tensor_step_capture_restores_enclosing_grad_mode(grad_enabled):
    from dendra.func._scan_tree import _capture_tensor_step

    def body(value):
        return value + (1 if torch.is_grad_enabled() else 2)

    with torch.set_grad_enabled(not grad_enabled):
        graph = _capture_tensor_step(body, (torch.ones(2),), grad_enabled=grad_enabled)
        assert torch.is_grad_enabled() is not grad_enabled
    torch.testing.assert_close(
        graph(torch.ones(2)), torch.full((2,), 2.0 if grad_enabled else 3.0)
    )


@pytest.mark.parametrize("fail", (False, True))
def test_tensor_step_capture_isolates_outer_save_hooks_and_restores_them(fail):
    from dendra.func._scan_tree import _capture_tensor_step

    events = []
    value = torch.tensor([0.2, 0.4], requires_grad=True)

    def body(current):
        output = current.sin()
        if fail:
            raise RuntimeError("injected body capture error")
        return output

    def pack(tensor):
        events.append("pack")
        return tensor.detach()

    def unpack(tensor):
        events.append("unpack")
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        with torch.no_grad():
            if fail:
                with pytest.raises(RuntimeError, match="injected body capture error"):
                    _capture_tensor_step(body, (value,), grad_enabled=True)
            else:
                _capture_tensor_step(body, (value,), grad_enabled=True)
            assert not torch.is_grad_enabled()
        assert events == []
        torch.autograd.grad(value.sin().sum(), value)
        assert events == ["pack", "unpack"]


@pytest.mark.parametrize(
    "field",
    (
        "_init_fn",
        "_abstract_fn",
        "_setup_context_fn",
        "_backward_fn",
        "_vmap_fn",
        "_schema",
        "code",
    ),
)
def test_delay_plan_guards_opaque_history_callbacks(field, monkeypatch):
    population = _population()
    plan = MechanismDelayPlan(population)
    raw = _delays._raw_overwrite_history
    if field == "code":

        def replacement(current, updated):
            return current

        monkeypatch.setattr(raw._init_fn, "__code__", replacement.__code__)
    elif field == "_schema":
        monkeypatch.setattr(raw, field, "changed")
    else:
        monkeypatch.setattr(raw, field, lambda *args, **kwargs: args)
    with pytest.raises(dn.func.FunctionalizationError, match="canonical"):
        plan.validate_source()


@pytest.mark.parametrize(
    "field", ("operator", "_abstract_fn", "_backward_fn", "_vmap_fn")
)
def test_warmed_compiled_opaque_history_callbacks_are_guarded(field, monkeypatch):
    plan = MechanismDelayPlan(_population())
    expected = plan.source_binding_signature()

    def evaluate(value):
        if plan.source_binding_signature() != expected:
            raise dn.func.FunctionalizationError(
                "Mechanism delayed-state opaque callback changed"
            )
        return value + 1

    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    compiled(torch.zeros(2))
    if field == "operator":
        monkeypatch.setattr(
            _delays, "_raw_overwrite_history", lambda current, updated: updated
        )
    else:
        monkeypatch.setattr(_delays._raw_overwrite_history, field, lambda *args: args)
    with pytest.raises((dn.func.FunctionalizationError, torch._dynamo.exc.Unsupported)):
        compiled(torch.zeros(2))


@pytest.mark.parametrize("lanes", (0, 2))
def test_functional_delay_counts_support_vmap_and_empty_vmap(lanes):
    functional, tensors = dn.func.make_functional(_population(), dt=0.125)
    key = next(
        name for name in tensors.constants if name.startswith("mechanism_delays.")
    )
    counts = torch.tensor(((0, 2), (2, 0)), dtype=torch.long)[:lanes]

    def evaluate(delays):
        return functional.prepare_and_step(
            tensors.parameters, {**tensors.constants, key: delays}, tensors.state
        )[0]

    actual = torch.vmap(evaluate)(counts)
    assert actual["integrator"]["v"].shape == (lanes, *functional.shape)
    if lanes:
        expected = torch.utils._pytree.tree_map(
            lambda *values: torch.stack(values), *(evaluate(value) for value in counts)
        )
        torch.testing.assert_close(actual, expected)
        invalid = counts.clone()
        invalid[1, 0] = -1
        with pytest.raises(RuntimeError):
            torch.vmap(evaluate)(invalid)
