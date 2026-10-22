"""Frozen ownership and waveform contracts for mechanism current injections."""

import copy
from types import MethodType

import numpy as np
import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._mechanism_injections import (
    MechanismInjectionPlan,
    configure_mechanism_injection_execution,
    mechanism_injection_constant_slots,
    mechanism_injection_ephemeral_slots,
    mechanism_injection_failures,
    mechanism_injection_helper,
    mechanism_injection_owners,
    network_mechanism_injection_alias_signature,
)
from dendra.models.mechanisms import Mechanism, _injections
from dendra.models.stim.waveform import Waveform

pytestmark = pytest.mark.cpu


class _InjectedCurrent(Mechanism):
    Mechanism.GLOBAL(g=0.001)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def inject(self, waveform, *, index=None, model_shape=None, model=None, **kwargs):
        return self.register_waveform_injection(
            waveform, index=index, model_shape=model_shape, model=model
        )

    def i_affine_terms(self, v):
        return self.g, self.injection_values(v, t=self.t)


class _MutableWaveform(Waveform):
    FUNCTIONAL_PURE = True

    def __init__(self):
        super().__init__()
        self.counter = 0

    def fn(self, t):
        self.counter += 1
        return torch.ones_like(t)


class _UnmarkedWaveform(Waveform):
    def fn(self, t):
        return torch.ones_like(t)


class _RandomWaveform(Waveform):
    FUNCTIONAL_PURE = True

    def fn(self, t):
        return torch.rand_like(t)


class _BufferWaveform(Waveform):
    FUNCTIONAL_PURE = True

    def __init__(self):
        super().__init__()
        self.register_buffer("offset", torch.tensor(2.0))

    def fn(self, t):
        return torch.ones_like(t) * self.offset


class _ParameterizedWaveform(Waveform):
    FUNCTIONAL_PURE = True
    Waveform.PARAMETERP(value=0.2)

    def fn(self, t):
        return torch.ones_like(t) * self.value()


_HIDDEN_TENSOR = torch.tensor(0.2)
_HIDDEN_VALUES = [_HIDDEN_TENSOR]


class _HiddenTensorWaveform(Waveform):
    FUNCTIONAL_PURE = True

    def fn(self, t):
        return torch.ones_like(t) * _HIDDEN_TENSOR


class _ScaleModule(torch.nn.Module):
    def forward(self, t):
        return torch.ones_like(t) * 0.2


class _ChildModuleWaveform(Waveform):
    FUNCTIONAL_PURE = True

    def __init__(self):
        super().__init__()
        self.scale = _ScaleModule()

    def fn(self, t):
        return self.scale(t)


class _ClassConfigWaveform(Waveform):
    FUNCTIONAL_PURE = True
    config = {"gain": 0.2}

    def fn(self, t):
        return torch.ones_like(t) * self.config["gain"]


class _ClassTensorWaveform(Waveform):
    FUNCTIONAL_PURE = True
    gain = torch.tensor(0.2)

    def fn(self, t):
        return torch.ones_like(t) * self.gain


class _ClassMutatingWaveform(Waveform):
    FUNCTIONAL_PURE = True
    calls = 0
    counters = {"calls": 0}
    raise_on_call = False

    def fn(self, t):
        cls = self.__class__
        cls.calls += 1
        cls.counters["calls"] += 1
        if cls.raise_on_call:
            raise RuntimeError("injected waveform audit failure")
        return torch.ones_like(t) * 0.2


class _ClassMutatingChildModule(torch.nn.Module):
    calls = 0

    def forward(self, t):
        self.__class__.calls += 1
        return torch.ones_like(t) * 0.2


class _ClassMutatingChildWaveform(Waveform):
    FUNCTIONAL_PURE = True

    def __init__(self):
        super().__init__()
        self.scale = _ClassMutatingChildModule()

    def fn(self, t):
        return self.scale(t)


def _population(waveform=None, *, inject=True):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64):
        population = dn.SingleCompartment(C=2, dtype=torch.float64)
        population.insert(_InjectedCurrent.rename("injected"))
        if inject:
            population[:, 0].inject(
                dn.constant(value=0.2) if waveform is None else waveform
            )
        population.initialize()
    return population


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, CHECK_PURITY=True),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def test_plan_classifies_routing_waveform_buffers_and_ephemeral_current():
    population = _population(_BufferWaveform())
    plan = MechanismInjectionPlan(population)
    ((path, owner),) = mechanism_injection_owners(population)
    assert path == "integrator.mech.mechanisms.injected"
    assert not mechanism_injection_failures(owner)
    assert tuple(dict(mechanism_injection_constant_slots(owner))) == (
        "_injection_mask_0",
        "_injection_scale_0",
        "_injection_index_0",
        "injected_waveforms.0.offset",
    )
    assert tuple(dict(mechanism_injection_ephemeral_slots(owner))) == ("i_inj",)
    assert [entry[1] for entry in plan.constant_entries] == [
        "_injection_mask_0",
        "_injection_index_0",
    ]
    assert (
        mechanism_injection_helper(owner, owner.injection_values) == "injection_values"
    )
    assert mechanism_injection_helper(owner, owner.evaluate_injections) is None
    assert not tuple(mechanism_injection_constant_slots(population))
    assert mechanism_injection_owners(population, ()) == ()
    plan.validate_source()
    plan.validate_source_tensor_values()


def test_plan_accepts_compatible_population_and_numerical_leaf_replacements():
    population = _population()
    plan = MechanismInjectionPlan(population)
    clone = copy.deepcopy(population)
    clone.mech.injected._injection_scale_0.add_(3)
    clone.mech.injected.injected_waveforms[0].value.data.add_(4)
    plan.validate_population(clone)
    plan.validate_population(_population())
    population.mech.injected._injection_scale_0.add_(5)
    plan.validate_source()


@pytest.mark.parametrize(
    "change", ("current", "count", "name", "field", "list", "waveforms")
)
def test_plan_rejects_registration_schema_changes(change):
    population = _population()
    plan = MechanismInjectionPlan(population)
    owner = population.mech.injected
    if change == "current":
        owner._injection_specs[0]["current_name"] = "other"
    elif change == "count":
        owner._injection_specs[0]["n_selected"] += 1
    elif change == "name":
        owner._injection_specs[0]["mask"] = "missing"
    elif change == "field":
        owner._injection_specs[0]["extra"] = True
    elif change == "list":
        owner._injection_specs = tuple(owner._injection_specs)
    else:
        owner.injected_waveforms.append(dn.constant(value=1))
    with pytest.raises(dn.func.FunctionalizationError, match="[Ii]njection|waveform"):
        plan.validate_source()


@pytest.mark.parametrize(
    "change",
    (
        "mask_dtype",
        "mask_shape",
        "mask_value",
        "index_dtype",
        "index_shape",
        "index_value",
        "scale_dtype",
    ),
)
def test_plan_rejects_malformed_or_changed_routing(change):
    population = _population()
    plan = MechanismInjectionPlan(population)
    owner = population.mech.injected
    if change == "mask_dtype":
        owner._injection_mask_0 = owner._injection_mask_0.long()
    elif change == "mask_shape":
        owner._injection_mask_0 = owner._injection_mask_0.reshape(-1)
    elif change == "mask_value":
        owner._injection_mask_0.logical_not_()
    elif change == "index_dtype":
        owner._injection_index_0 = owner._injection_index_0.float()
    elif change == "index_shape":
        owner._injection_index_0 = owner._injection_index_0.unsqueeze(0)
    elif change == "index_value":
        owner._injection_index_0.add_(1)
    else:
        owner._injection_scale_0 = owner._injection_scale_0.long()
    with pytest.raises(
        dn.func.FunctionalizationError, match="[Ii]njection|mask|index|scale"
    ):
        plan.validate_source()


@pytest.mark.parametrize(
    "change",
    ("false", "true_without_owner", "missing_flag", "missing_spec", "solver_mismatch"),
)
def test_plan_rejects_stale_or_false_ownership_claims(change):
    population = _population(inject=change == "false")
    if change == "false":
        population.mechanism_injection_accepted[0] = False
    else:
        population[:, 0].inject(dn.constant(value=0.3))
        if change != "missing_flag":
            population.mech.injected.clear_injections()
        if change == "missing_flag":
            population.mechanism_injection_accepted.clear()
        elif change == "missing_spec":
            population.mechanism_injections.clear()
        elif change == "solver_mismatch":
            waveform, shape, index = population.injections[0]
            population.injections[0] = (dn.constant(value=0.4), shape, index)
    with pytest.raises(dn.func.FunctionalizationError, match="[Ii]njection|owner"):
        MechanismInjectionPlan(population)


@pytest.mark.parametrize(
    "waveform", (_MutableWaveform, _UnmarkedWaveform, _RandomWaveform)
)
def test_plan_audits_waveforms_without_mutating_source_or_rng(waveform):
    population = _population(waveform())
    owner = population.mech.injected
    source = owner.injected_waveforms[0]
    counter = getattr(source, "counter", None)
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(
        dn.func.FunctionalizationError, match="purity|mutat|determin|RNG"
    ):
        MechanismInjectionPlan(population)
    assert torch.equal(rng, torch.random.get_rng_state())
    if hasattr(source, "counter"):
        assert source.counter == counter


@pytest.mark.parametrize(
    "change", ("helper", "code", "core", "class_code", "defaults", "globals", "math")
)
def test_plan_guards_canonical_helpers_and_waveform_behavior(change, monkeypatch):
    population = _population()
    plan = MechanismInjectionPlan(population)
    owner = population.mech.injected
    original = plan.source_binding_signature()
    if change == "helper":
        owner.injection_values = MethodType(lambda self, v, **kwargs: v, owner)
    elif change == "code":
        monkeypatch.setattr(
            Mechanism.injection_values,
            "__code__",
            (lambda self, v, **kwargs: v).__code__,
        )
    elif change == "core":
        monkeypatch.setattr(
            _injections, "expand_injection_value", lambda *args, **kwargs: args[0]
        )
    elif change == "defaults":
        monkeypatch.setattr(
            Mechanism.injection_values, "__kwdefaults__", {"current_name": "other"}
        )
    elif change == "globals":
        monkeypatch.setitem(Mechanism.injection_values.__globals__, "torch", object())
    elif change == "math":
        monkeypatch.setattr(_injections.math, "prod", lambda values: 0)
    else:
        waveform = owner.injected_waveforms[0]
        monkeypatch.setattr(
            type(waveform), "fn", lambda self, t: t * 0 + self.value * 2
        )
    assert plan.source_binding_signature() != original
    with pytest.raises(
        dn.func.FunctionalizationError, match="[Ii]njection|helper|waveform|canonical"
    ):
        plan.validate_source()


def test_plan_fails_explicit_routing_mutation_and_permits_numerical_scale():
    population = _population()
    plan = MechanismInjectionPlan(population)
    values = {
        f"{path}.{name}": expected.clone()
        for path, name, expected in plan.constant_entries
    }
    plan.validate_constant_values(values)
    key = next(key for key in values if "mask" in key)
    values[key].logical_not_()
    with pytest.raises(RuntimeError, match="routing"):
        plan.validate_constant_values(values)


def test_plan_alias_changes_require_lowering_again():
    population = _population(_BufferWaveform())
    owner = population.mech.injected
    owner.register_buffer("external", owner.injected_waveforms[0].offset)
    plan = MechanismInjectionPlan(population)
    owner.external = owner.external.clone()
    with pytest.raises(dn.func.FunctionalizationError, match="aliases"):
        plan.validate_source()


def test_execution_configuration_retains_waveform_training_flags():
    population = _population(dn.constant(value=0.2))
    population.eval()
    clone = copy.deepcopy(population)
    configure_mechanism_injection_execution(clone)
    assert not clone.mech.injected.injected_waveforms[0].training
    assert not population.mech.injected.injected_waveforms[0].training


def test_execution_configuration_clears_child_parameter_caches_only_on_clone():
    from dendra.models.parametric import cacheable

    population = _population(_ParameterizedWaveform())
    population.eval()
    waveform = population.mech.injected.injected_waveforms[0]
    waveform(torch.tensor(0.0))
    child = next(
        module for module in waveform.modules() if isinstance(module, cacheable)
    )
    assert child._cache is not None
    plan = MechanismInjectionPlan(population)
    clone = copy.deepcopy(population)
    plan.configure_execution(clone)
    cloned_waveform = clone.mech.injected.injected_waveforms[0]
    cloned_child = next(
        module for module in cloned_waveform.modules() if isinstance(module, cacheable)
    )
    assert not cloned_waveform.training
    assert cloned_child.training and cloned_child._cache is None
    assert not child.training and child._cache is not None


@pytest.mark.parametrize(
    "kind", ("sin", "rect", "composite", "repeat", "poisson", "buffer", "parameterized")
)
def test_plan_accepts_shared_waveform_recipes(kind):
    if kind == "sin":
        waveform = dn.sin(amp=0.2, freq=0.3)
    elif kind == "rect":
        waveform = dn.mono_rect(amp=0.2, delay=0.1, pw=0.2)
    elif kind == "composite":
        waveform = dn.constant(value=0.1) + dn.sin(amp=0.2, freq=0.3)
    elif kind == "repeat":
        waveform = dn.constant(value=0.1).repeat(freq=0.2)
    elif kind == "poisson":
        waveform = dn.constant(value=0.1).poisson(interval=0.2, n=2)
    elif kind == "buffer":
        waveform = _BufferWaveform()
    else:
        waveform = _ParameterizedWaveform()
    plan = MechanismInjectionPlan(_population(waveform))
    plan.validate_source()


def test_plan_rejects_hidden_external_tensor_dependencies():
    with pytest.raises(
        dn.func.FunctionalizationError, match="hidden mutable numerical"
    ):
        MechanismInjectionPlan(_population(_HiddenTensorWaveform()))


@pytest.mark.parametrize(
    "change", ("current", "owner", "helper", "waveform", "mask", "index")
)
def test_warmed_compiled_plan_guards_routing_and_behavior(change, monkeypatch):
    population = _population()
    plan = MechanismInjectionPlan(population)
    expected = plan.source_binding_signature()

    def evaluate(value):
        if plan.source_binding_signature() != expected:
            raise dn.func.FunctionalizationError("Mechanism injection bindings changed")
        plan.validate_source_tensor_values()
        return value + 1

    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(torch.zeros(2)), torch.ones(2))
    owner = population.mech.injected
    if change == "current":
        owner._injection_specs[0]["current_name"] = "other"
    elif change == "owner":
        population.mechanism_injection_accepted[0] = False
    elif change == "helper":
        owner.injection_values = MethodType(lambda self, v, **kwargs: v, owner)
    elif change == "waveform":
        monkeypatch.setattr(type(owner.injected_waveforms[0]), "fn", lambda self, t: t)
    elif change == "mask":
        owner._injection_mask_0.logical_not_()
    else:
        owner._injection_index_0.add_(1)
    with pytest.raises(
        (RuntimeError, dn.func.FunctionalizationError, torch._dynamo.exc.Unsupported)
    ):
        compiled(torch.zeros(2))


@pytest.mark.parametrize("change", ("metadata", "routing", "waveform", "alias"))
def test_compiled_population_guards_source_injection_changes(change, monkeypatch):
    population = _population()
    functional, tensors = dn.func.make_functional(population, dt=0.125)

    def evaluate(parameters, constants, state):
        return functional.prepare_and_step(parameters, constants, state)

    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    compiled(tensors.parameters, tensors.constants, tensors.state)
    owner = population.mech.injected
    if change == "metadata":
        owner._injection_specs[0]["current_name"] = "other"
    elif change == "routing":
        owner._injection_index_0.add_(1)
    elif change == "waveform":
        monkeypatch.setattr(type(owner.injected_waveforms[0]), "FUNCTIONAL_PURE", False)
    else:
        owner.register_buffer("alias", owner._injection_scale_0)
    with pytest.raises(
        (RuntimeError, dn.func.FunctionalizationError, torch._dynamo.exc.Unsupported)
    ):
        compiled(tensors.parameters, tensors.constants, tensors.state)


def test_network_alias_signature_preserves_shared_waveform_parameters():
    waveform = dn.constant(value=0.2)
    network = dn.Network(
        {"left": _population(waveform), "right": _population(waveform)}
    )
    signature = network_mechanism_injection_alias_signature(network)
    right = network.populations["right"].mech.injected
    right.injected_waveforms[0] = copy.deepcopy(right.injected_waveforms[0])
    assert network_mechanism_injection_alias_signature(network) != signature


def test_warmed_compiled_plan_guards_authored_tensor_index_values():
    population = _population()
    waveform, shape, index = population.injections[0]
    selected = torch.zeros_like(population.v, dtype=torch.bool)
    selected[index] = True
    index = torch.nonzero(selected, as_tuple=True)
    population.injections[0] = (waveform, shape, index)
    population.mechanism_injections[0] = (waveform, shape, index)
    plan = MechanismInjectionPlan(population)

    def evaluate(value):
        plan.validate_source_tensor_values()
        return value + 1

    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    compiled(torch.zeros(2))
    index[-1].add_(1)
    with pytest.raises(RuntimeError):
        compiled(torch.zeros(2))


@pytest.mark.parametrize(
    "change",
    (
        "forward_hook",
        "pre_hook",
        "backward_hook",
        "child_hook",
        "child_forward",
        "class_config",
        "class_callback",
    ),
)
def test_compiled_population_guards_post_lowering_waveform_hooks_and_configuration(
    change, monkeypatch
):
    waveform = (
        _ChildModuleWaveform()
        if change.startswith("child")
        else _ClassConfigWaveform()
        if change.startswith("class")
        else dn.constant(value=0.2)
    )
    population = _population(waveform)
    functional, tensors = dn.func.make_functional(population, dt=0.125)

    def evaluate(parameters, constants, state):
        return functional.prepare_and_step(parameters, constants, state)

    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    compiled(tensors.parameters, tensors.constants, tensors.state)
    if change == "forward_hook":
        waveform.register_forward_hook(lambda _module, _args, output: output * 2)
    elif change == "pre_hook":
        waveform.register_forward_pre_hook(lambda _module, args: (args[0] + 1,))
    elif change == "backward_hook":
        waveform.register_full_backward_hook(lambda _module, inputs, _outputs: inputs)
    elif change == "child_hook":
        waveform.scale.register_forward_hook(lambda _module, _args, output: output * 2)
    elif change == "child_forward":
        waveform.scale.forward = MethodType(
            lambda self, t: torch.ones_like(t) * 0.4, waveform.scale
        )
    elif change == "class_config":
        monkeypatch.setitem(type(waveform).config, "gain", 0.4)
    else:
        monkeypatch.setitem(type(waveform).config, "gain", lambda t: t)
    with pytest.raises(
        (RuntimeError, dn.func.FunctionalizationError, torch._dynamo.exc.Unsupported)
    ):
        compiled(tensors.parameters, tensors.constants, tensors.state)
    with pytest.raises(dn.func.FunctionalizationError):
        functional._mechanism_injection_plan.validate_source()


def test_plan_rejects_hidden_class_tensor_configuration():
    with pytest.raises(dn.func.FunctionalizationError, match="unauditable.*Tensor"):
        MechanismInjectionPlan(_population(_ClassTensorWaveform()))


@pytest.mark.parametrize("kind", ("numpy", "generator"))
@pytest.mark.parametrize("container", ("direct", "dict", "list", "tuple"))
def test_plan_rejects_hidden_class_numerical_configuration_without_mutation(
    kind, container
):
    value = np.array([0.2]) if kind == "numpy" else torch.Generator().manual_seed(2)
    config = {
        "direct": value,
        "dict": {"nested": value},
        "list": [value],
        "tuple": (value,),
    }[container]
    waveform_type = type(
        "HiddenClassNumericalConfiguration",
        (Waveform,),
        {
            "FUNCTIONAL_PURE": True,
            "config": config,
            "fn": lambda self, t: torch.ones_like(t) * 0.2,
        },
    )
    expected = value.copy() if kind == "numpy" else value.get_state().clone()
    population = _population(waveform_type())
    with pytest.raises(
        dn.func.FunctionalizationError, match="hidden mutable numerical state"
    ):
        MechanismInjectionPlan(population)
    assert waveform_type.config is config
    if kind == "numpy":
        np.testing.assert_array_equal(value, expected)
    else:
        torch.testing.assert_close(value.get_state(), expected)


@pytest.mark.parametrize("kind", ("numpy", "generator"))
def test_lowered_plan_rejects_introduced_hidden_class_numerical_configuration(
    kind, monkeypatch
):
    population = _population(_ClassConfigWaveform())
    functional, tensors = dn.func.make_functional(population, dt=0.125)

    def evaluate(parameters, constants, state):
        return functional.prepare_and_step(parameters, constants, state)[0][
            "integrator"
        ]["v"]

    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    compiled(tensors.parameters, tensors.constants, tensors.state)
    value = np.array([0.8]) if kind == "numpy" else torch.Generator().manual_seed(8)
    monkeypatch.setitem(_ClassConfigWaveform.config, "gain", value)
    with pytest.raises(
        dn.func.FunctionalizationError, match="hidden mutable numerical state"
    ):
        functional._mechanism_injection_plan.validate_source()
    with pytest.raises(
        (RuntimeError, dn.func.FunctionalizationError, torch._dynamo.exc.Unsupported)
    ):
        compiled(tensors.parameters, tensors.constants, tensors.state)
    assert _ClassConfigWaveform.config["gain"] is value


def test_plan_retains_immutable_numpy_scalar_class_configuration(monkeypatch):
    monkeypatch.setitem(_ClassConfigWaveform.config, "gain", np.float64(0.2))
    population = _population(_ClassConfigWaveform())
    plan = MechanismInjectionPlan(population)
    plan.validate_source()
    monkeypatch.setitem(_ClassConfigWaveform.config, "gain", np.float64(0.8))
    with pytest.raises(dn.func.FunctionalizationError, match="changed; lower again"):
        plan.validate_source()


@pytest.mark.parametrize("container", ("direct", "dict", "list", "tuple"))
def test_plan_rejects_mutable_class_set_configuration_without_mutation(container):
    values = {0.2}
    config = {
        "direct": values,
        "dict": {"nested": values},
        "list": [values],
        "tuple": (values,),
    }[container]
    waveform_type = type(
        "MutableClassSetConfiguration",
        (Waveform,),
        {
            "FUNCTIONAL_PURE": True,
            "config": config,
            "fn": lambda self, t: torch.ones_like(t) * 0.2,
        },
    )
    with pytest.raises(dn.func.FunctionalizationError, match="mutable set"):
        MechanismInjectionPlan(_population(waveform_type()))
    assert waveform_type.config is config and values == {0.2}


def _dispatch_waveform(kind):
    def fn(self, t):
        return torch.ones_like(t) * 0.2

    def call(self, t):
        return torch.ones_like(t) * _HIDDEN_TENSOR

    def getattribute(self, name):
        return object.__getattribute__(self, name)

    def getattr_(self, name):
        return torch.nn.Module.__getattr__(self, name)

    override = {
        "__call__": call,
        "_call_impl": call,
        "_wrapped_call_impl": call,
        "__getattribute__": getattribute,
        "__getattr__": getattr_,
    }[kind]
    return type(
        f"CustomDispatch_{kind}",
        (Waveform,),
        {"FUNCTIONAL_PURE": True, "fn": fn, kind: override},
    )()


@pytest.mark.parametrize(
    "kind",
    ("__call__", "_call_impl", "_wrapped_call_impl", "__getattribute__", "__getattr__"),
)
def test_plan_rejects_custom_waveform_call_and_attribute_dispatch(kind):
    with pytest.raises(dn.func.FunctionalizationError, match="canonical.*dispatch"):
        MechanismInjectionPlan(_population(_dispatch_waveform(kind)))


@pytest.mark.parametrize(
    "kind",
    ("__call__", "_call_impl", "_wrapped_call_impl", "__getattribute__", "__getattr__"),
)
@pytest.mark.parametrize("placement", ("waveform", "child", "instance"))
def test_warmed_compiled_population_guards_waveform_dispatch(
    kind, placement, monkeypatch
):
    waveform = _ChildModuleWaveform()
    population = _population(waveform)
    functional, tensors = dn.func.make_functional(population, dt=0.125)

    def evaluate(parameters, constants, state):
        return functional.prepare_and_step(parameters, constants, state)[0][
            "integrator"
        ]["v"]

    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    compiled(tensors.parameters, tensors.constants, tensors.state)
    override = getattr(type(_dispatch_waveform(kind)), kind)
    if placement == "instance":
        monkeypatch.setattr(waveform, kind, MethodType(override, waveform))
    else:
        target = waveform if placement == "waveform" else waveform.scale
        monkeypatch.setattr(type(target), kind, override)
    with pytest.raises(
        (RuntimeError, dn.func.FunctionalizationError, torch._dynamo.exc.Unsupported)
    ):
        compiled(tensors.parameters, tensors.constants, tensors.state)
    with pytest.raises(dn.func.FunctionalizationError, match="canonical.*dispatch"):
        functional._mechanism_injection_plan.validate_source()


@pytest.mark.parametrize("fail", (False, True))
def test_plan_restores_shared_waveform_class_state_on_purity_failure(fail, monkeypatch):
    waveform = _ClassMutatingWaveform()
    population = _population(waveform)
    cls = type(waveform)
    monkeypatch.setattr(cls, "raise_on_call", fail)
    calls = cls.calls
    counters = cls.counters
    values = counters.copy()
    with pytest.raises(
        dn.func.FunctionalizationError, match="mutated shared class state"
    ):
        MechanismInjectionPlan(population)
    assert cls.calls == calls
    assert cls.counters is counters
    assert cls.counters == values
    assert cls.raise_on_call is fail


def test_plan_restores_shared_child_module_class_state_on_purity_failure():
    waveform = _ClassMutatingChildWaveform()
    population = _population(waveform)
    cls = type(waveform.scale)
    calls = cls.calls
    with pytest.raises(
        dn.func.FunctionalizationError, match="mutated shared class state"
    ):
        MechanismInjectionPlan(population)
    assert cls.calls == calls


def test_fullgraph_explicit_parameter_hook_retains_autograd_edge_and_eager_admission_check():
    population = _population()
    waveform = population.mech.injected.injected_waveforms[0]
    parameter = next(waveform.parameters())
    functional, tensors = dn.func.make_functional(population, dt=0.125)

    def evaluate(parameters, constants, state):
        return functional.prepare_and_step(parameters, constants, state)[0][
            "integrator"
        ]["v"].sum()

    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    before = torch.autograd.grad(
        compiled(tensors.parameters, tensors.constants, tensors.state), parameter
    )[0]
    assert before.abs().sum() > 0
    handle = parameter.register_hook(lambda gradient: gradient * 2)
    try:
        after = torch.autograd.grad(
            compiled(tensors.parameters, tensors.constants, tensors.state), parameter
        )[0]
        torch.testing.assert_close(after, before * 2)
        with pytest.raises(dn.func.FunctionalizationError, match="gradient hooks"):
            functional._mechanism_injection_plan.validate_source()
    finally:
        handle.remove()


def test_plan_rejects_nonliteral_metadata_keys_without_calling_their_hash_again():
    population = _population()
    plan = MechanismInjectionPlan(population)

    class Key:
        reject = False

        def __hash__(self):
            if self.reject:
                raise AssertionError("metadata hash must not execute during validation")
            return 987654321

    key = Key()
    population.mech.injected._injection_specs[0][key] = True
    key.reject = True
    with pytest.raises(dn.func.FunctionalizationError, match="metadata"):
        plan.validate_source()


def _indirect_waveform(kind):
    def hidden_recipe(self, t):
        return torch.ones_like(t) * _HIDDEN_TENSOR

    def local_recipe(self, t):
        return torch.ones_like(t) * self.factor

    if kind == "closure":
        hidden = torch.tensor(0.2)

        def recipe(self, t):
            return torch.ones_like(t) * hidden
    elif kind == "default":

        def recipe(self, t, value=_HIDDEN_TENSOR):
            return torch.ones_like(t) * value
    elif kind == "kwdefault":

        def recipe(self, t, *, value=_HIDDEN_TENSOR):
            return torch.ones_like(t) * value
    elif kind == "mutable_list":

        def recipe(self, t):
            return torch.ones_like(t) * _HIDDEN_VALUES[0]
    elif kind in ("registered", "registered_property", "unreachable"):
        recipe = local_recipe
    else:
        recipe = hidden_recipe

    def fn(self, t):
        return self.recipe(t)

    namespace = {"FUNCTIONAL_PURE": True, "fn": fn, "recipe": recipe}
    if kind == "staticmethod":

        def static_recipe(t):
            return torch.ones_like(t) * _HIDDEN_TENSOR

        namespace["recipe"] = staticmethod(static_recipe)
    elif kind == "classmethod":
        namespace["recipe"] = classmethod(hidden_recipe)
    elif kind == "property":

        def factor(self):
            return _HIDDEN_TENSOR

        namespace["factor"] = property(factor)
        namespace["recipe"] = local_recipe
    elif kind == "registered_property":

        def factor(self):
            return self.registered_factor

        namespace["factor"] = property(factor)
    elif kind.startswith("child"):

        class Child(torch.nn.Module):
            recipe = hidden_recipe

            def forward(self, t):
                return self.recipe(t)

        def child_fn(self, t):
            return self.child.recipe(t)

        def child_forward_fn(self, t):
            return self.child(t)

        namespace["fn"] = child_fn if kind == "child_method" else child_forward_fn
        namespace["child_type"] = Child
    elif kind == "unreachable":
        namespace["unused"] = hidden_recipe
    waveform_type = type(f"Indirect_{kind}", (Waveform,), namespace)
    waveform = waveform_type()
    if kind == "registered_property":
        waveform.register_buffer("registered_factor", torch.tensor(0.2))
    elif kind != "property":
        waveform.register_buffer("factor", torch.tensor(0.2))
    if kind.startswith("child"):
        waveform.child = waveform.child_type()
    return waveform


@pytest.mark.parametrize(
    "kind",
    (
        "helper",
        "property",
        "staticmethod",
        "classmethod",
        "child_forward",
        "child_method",
        "closure",
        "default",
        "kwdefault",
        "mutable_list",
    ),
)
def test_plan_rejects_hidden_numerical_dependencies_in_reachable_helpers(kind):
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="hidden mutable numerical|mutable external Python",
    ):
        MechanismInjectionPlan(_population(_indirect_waveform(kind)))


@pytest.mark.parametrize("kind", ("registered", "registered_property", "unreachable"))
def test_plan_accepts_helpers_using_registered_values_and_ignores_unused_helpers(kind):
    plan = MechanismInjectionPlan(_population(_indirect_waveform(kind)))
    plan.validate_source()


@pytest.mark.parametrize("kind", ("alias", "index"))
def test_plan_rejects_opaque_child_module_helper_dispatch(kind):
    class Child(torch.nn.Module):
        def forward(self, t):
            return torch.ones_like(t) * 0.2

        def recipe(self, t):
            return torch.ones_like(t) * _HIDDEN_TENSOR

    def alias_fn(self, t):
        child = self.child
        return child.recipe(t)

    def index_fn(self, t):
        return self.layers[0].recipe(t)

    waveform_type = type(
        f"Escaping_{kind}",
        (Waveform,),
        {"FUNCTIONAL_PURE": True, "fn": alias_fn if kind == "alias" else index_fn},
    )
    waveform = waveform_type()
    if kind == "alias":
        waveform.child = Child()
    else:
        waveform.layers = torch.nn.ModuleList([Child()])
    with pytest.raises(dn.func.FunctionalizationError, match="purity cannot be proven"):
        MechanismInjectionPlan(_population(waveform))


def test_plan_can_skip_optional_purity_diagnostics_and_freezes_the_policy():
    population = _population(_MutableWaveform())
    waveform = population.mech.mechanisms["injected"].injected_waveforms[0]
    counter = waveform.counter
    with dn.ctx(CHECK_PURITY=False):
        plan = MechanismInjectionPlan(population)
    assert not plan.check_purity
    assert waveform.counter == counter
    # The outer fixture enables diagnostics; validating a trusted plan must
    # retain its original policy and must not execute a numerical probe.
    plan.validate_source()
    assert waveform.counter == counter
    with pytest.raises(dn.func.FunctionalizationError, match="purity|mutat|state"):
        MechanismInjectionPlan(population)
