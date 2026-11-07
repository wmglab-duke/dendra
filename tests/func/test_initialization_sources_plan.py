"""Frozen zero-argument recipe dependencies and lazy initialization admission."""

from __future__ import annotations

from types import MethodType

import numpy as np
import pytest
import torch

import dendra as dn
from dendra.func._initialization_sources import (
    InitializationSourcePlan,
    configure_initialization_source_execution,
    initialization_source_constant_slots,
    initialization_source_roots,
)
from dendra.func._population import _deepcopy_execution_module
from dendra.func._types import FunctionalizationError
from dendra.models.mechanisms._ions import Ion
from dendra.models.mechanisms._materials import Material
from dendra.models.multi import _PackedMaterialInitial
from dendra.models.parametric import Bounded, NegativeParam, PositiveParam

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _diagnostic_purity():
    with dn.ctx(CHECK_PURITY=True):
        yield


class _AffineSource(torch.nn.Module):
    def __init__(self, value=1.5):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(value, dtype=torch.float64))
        self.register_buffer("offset", torch.tensor(0.25, dtype=torch.float64))
        self.gain = 2.0

    def forward(self):
        return self.scale() + self.offset

    def scale(self):
        return self.value * self.gain


class _PropertySource(_AffineSource):
    @property
    def scaled(self):
        return self.value * self.gain

    def forward(self):
        return self.scaled + self.offset


class _RandomSource(_AffineSource):
    def forward(self):
        return self.value + torch.rand_like(self.value)


class _MutationSource(_AffineSource):
    def forward(self):
        self.offset.add_(1.0)
        return self.value + self.offset


class _PythonMutationSource(_AffineSource):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def forward(self):
        self.counter += 1
        return self.value


class _ClassMutationSource(_AffineSource):
    counter = 0

    def forward(self):
        self.__class__.counter += 1
        return self.value


class _HiddenTensorSource(_AffineSource):
    hidden = torch.tensor(1.0)

    def forward(self):
        return self.value + self.hidden


class _HiddenArraySource(_AffineSource):
    hidden = np.array([1.0])

    def forward(self):
        return self.value


class _CallOverrideSource(_AffineSource):
    def __call__(self):
        return self.value


class _CopyOverrideSource(_AffineSource):
    def __deepcopy__(self, memo):
        return self


class _TrainingSource(_AffineSource):
    def forward(self):
        return self.value + (self.offset if self.training else -self.offset)


_HIDDEN_UPPER_OFFSET = torch.tensor(0.5)


class _HiddenUpperBounded(Bounded):
    def _apply_upper(self, value):
        return value + _HIDDEN_UPPER_OFFSET


class _GradRandomUpperBounded(Bounded):
    def _apply_upper(self, value):
        if torch.is_grad_enabled():
            return value + torch.rand_like(value)
        return value


class _PureUpperBounded(Bounded):
    def _apply_upper(self, value):
        return value + 0.5


def _population(source=None, *, ions=False, shape=(1, 2)):
    if source is None:
        source = _AffineSource()
    population = torch.nn.Module()
    population.mech = torch.nn.Module()
    population.mech.ions = torch.nn.ModuleDict()
    population.mech.materials = torch.nn.ModuleDict(
        {"pool": Material("pool", shape, fields={"amount": source, "reserve": source})}
    )
    if ions:
        ion = Ion("na", (1, 2), einit=1, eadvance=1)
        del ion.o_init
        ion.o_init = source
        population.mech.ions["na"] = ion
    return population


def _failures(source):
    population = _population(source)
    before = {
        name: value.detach().clone() for name, value in population.named_parameters()
    }
    buffers = {
        name: value.detach().clone() for name, value in population.named_buffers()
    }
    plan = InitializationSourcePlan(population)
    assert plan.failures
    with pytest.raises(FunctionalizationError, match="Initial-value source"):
        plan.validate_initialization()
    for name, expected in before.items():
        torch.testing.assert_close(dict(population.named_parameters())[name], expected)
    for name, expected in buffers.items():
        torch.testing.assert_close(dict(population.named_buffers())[name], expected)
    return plan


@pytest.mark.parametrize("kind", (_AffineSource, _PropertySource, _TrainingSource))
@pytest.mark.parametrize("training", (False, True))
def test_pure_zero_argument_sources_are_audited_without_source_mutation(kind, training):
    source = kind().train(training)
    population = _population(source, ions=True)
    plan = InitializationSourcePlan(population)
    assert not plan.failures, plan.failures
    plan.validate_initialization()
    assert source.training is training
    assert source.value.grad is None
    plan.validate_population(_deepcopy_execution_module(population))


def test_registration_paths_and_tensor_slots_keep_all_source_aliases():
    source = _AffineSource()
    population = _population(source, ions=True)
    roots = initialization_source_roots(population)
    assert tuple(path for path, _module in roots) == (
        "mech.ions.na.o_init",
        "mech.materials.pool._initial_sources.amount",
        "mech.materials.pool._initial_sources.reserve",
    )
    assert all(module is source for _path, module in roots)
    assert dict(initialization_source_constant_slots(source))["offset"] is source.offset


@pytest.mark.parametrize("value", (1.5, [1.5, 2.5]))
@pytest.mark.parametrize("training", (False, True))
def test_positive_parameter_uses_shared_pure_compute_and_preserves_mode(
    value, training
):
    source = PositiveParam(torch.tensor(value, dtype=torch.float64), requires_grad=True)
    source.train(training)
    population = _population(source, ions=True)
    expected_cache = source._cache
    plan = InitializationSourcePlan(population)
    assert not plan.failures, plan.failures
    plan.validate_initialization()
    assert source.training is training
    assert source._cache is expected_cache
    private = _deepcopy_execution_module(population)
    configure_initialization_source_execution(private)
    cloned = dict(initialization_source_roots(private))["mech.ions.na.o_init"]
    assert cloned is not source
    assert cloned.training is training
    assert cloned._cache is None


def test_packed_sources_reuse_original_recipes_and_parameters():
    shared = _AffineSource()
    inner = _PackedMaterialInitial(
        [shared, PositiveParam(torch.tensor(2.0))], [(1, 1), (1, 1)]
    )
    outer = _PackedMaterialInitial([inner, shared], [(1, 2), (1, 1)])
    population = _population(outer, shape=(1, 3))
    plan = InitializationSourcePlan(population)
    assert not plan.failures, plan.failures
    assert outer.source_0.source_0 is outer.source_1 is shared
    plan.validate_initialization()


@pytest.mark.parametrize(
    "kind",
    (
        _RandomSource,
        _MutationSource,
        _PythonMutationSource,
        _ClassMutationSource,
        _HiddenTensorSource,
        _HiddenArraySource,
        _CallOverrideSource,
        _CopyOverrideSource,
    ),
)
def test_unsupported_sources_have_lazy_fresh_boundary_failures(kind):
    _failures(kind())


def test_implicit_rng_audit_restores_global_state():
    population = _population(_RandomSource())
    before = torch.random.get_rng_state().clone()
    plan = InitializationSourcePlan(population)
    assert plan.failures
    torch.testing.assert_close(torch.random.get_rng_state(), before)


def test_class_mutation_does_not_escape_a_failed_probe():
    source = _ClassMutationSource()
    population = _population(source)
    before = type(source).counter
    plan = InitializationSourcePlan(population)
    assert plan.failures
    assert type(source).counter == before


@pytest.mark.parametrize("hook", ("forward", "backward", "parameter"))
def test_hooks_are_rejected_without_invocation(hook):
    source = _AffineSource()
    population = _population(source)
    called = []
    if hook == "forward":
        handle = source.register_forward_hook(lambda *args: called.append(args))
    elif hook == "backward":
        handle = source.register_full_backward_hook(lambda *args: called.append(args))
    else:
        handle = source.value.register_hook(lambda value: value)
    try:
        plan = InitializationSourcePlan(population)
        assert plan.failures
        assert not called
    finally:
        handle.remove()


@pytest.mark.parametrize(
    "change",
    (
        "config",
        "helper",
        "parameter_alias",
        "buffer_shape",
        "mode",
        "hook",
        "instance_forward",
    ),
)
def test_eager_validation_rejects_changed_recipe_contract(change, monkeypatch):
    source = _AffineSource()
    population = _population(source)
    plan = InitializationSourcePlan(population)
    assert not plan.failures, plan.failures
    if change == "config":
        source.gain = 3.0
    elif change == "helper":
        monkeypatch.setattr(_AffineSource, "scale", lambda self: self.value * 3.0)
    elif change == "parameter_alias":
        population.mech.materials.pool._initial_sources.reserve = _AffineSource()
    elif change == "buffer_shape":
        source.offset = source.offset.expand(2).clone()
    elif change == "mode":
        source.eval()
    elif change == "hook":
        source.register_forward_hook(lambda *_args: None)
    else:
        source.forward = MethodType(lambda self: self.value, source)
    with pytest.raises(FunctionalizationError):
        plan.validate_source()


def test_raw_parameter_value_updates_remain_allowed():
    source = _AffineSource()
    population = _population(source)
    plan = InitializationSourcePlan(population)
    with torch.no_grad():
        source.value.add_(0.25)
        source.offset.add_(0.5)
    plan.validate_source()
    assert plan.source_binding_signature() == plan._bindings


def test_no_module_sources_keep_existing_tensor_initialization_admission():
    population = _population(torch.nn.Parameter(torch.tensor(1.0)))
    plan = InitializationSourcePlan(population)
    assert not initialization_source_roots(population)
    assert not plan.failures
    plan.validate_initialization()


def test_configured_private_recipes_keep_compatible_public_source_contracts():
    source = PositiveParam(torch.tensor(2.0), requires_grad=True).eval()
    source_population = _population(source)
    plan = InitializationSourcePlan(source_population)
    private = _deepcopy_execution_module(source_population)
    configure_initialization_source_execution(private)
    plan.validate_population(private)
    private_plan = InitializationSourcePlan(private)
    assert not private_plan.failures, private_plan.failures
    private_plan.validate_population(source_population)
    private_plan.validate_source()


@pytest.mark.parametrize(
    "change",
    (
        "config",
        "helper",
        "class_method",
        "class_method_none",
        "buffer_shape",
        "mode",
        "hook",
    ),
)
def test_fullgraph_warm_source_guards_track_recipe_dependencies(change, monkeypatch):
    from dendra._bootstrap import torch_compiler_warning_context

    source = _AffineSource()
    plan = InitializationSourcePlan(_population(source))
    expected = plan.source_binding_signature()

    def evaluate(value):
        if plan.source_binding_signature() != expected:
            raise FunctionalizationError(
                "Initial-value source dependencies changed; lower again."
            )
        return value + 1

    with torch_compiler_warning_context():
        compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(torch.tensor(1.0)), torch.tensor(2.0))
        if change == "config":
            source.gain = 3.0
        elif change == "helper":
            monkeypatch.setattr(_AffineSource, "scale", lambda self: self.value * 3.0)
        elif change == "class_method":
            monkeypatch.setattr(_AffineSource, "__call__", lambda self: self.value)
        elif change == "class_method_none":
            monkeypatch.setattr(_AffineSource, "__call__", None)
        elif change == "buffer_shape":
            source.offset = source.offset.expand(2).clone()
        elif change == "mode":
            source.eval()
        else:
            source.register_forward_hook(lambda *_args: None)
        with pytest.raises(
            (FunctionalizationError, torch._dynamo.exc.Unsupported),
            match="changed|[Ee]xception|raise",
        ):
            compiled(torch.tensor(1.0))


class _SetterSource(_AffineSource):
    writes = 0

    def __setattr__(self, name, value):
        type(self).writes += 1
        return super().__setattr__(name, value)


def test_cold_source_qualification_does_not_call_authored_attribute_setters():
    source = _SetterSource()
    population = _population(source)
    before = type(source).writes
    plan = InitializationSourcePlan(population)
    assert plan.failures
    assert type(source).writes == before


def test_authored_marker_cannot_bypass_canonical_cache_dispatch():
    source = PositiveParam(torch.tensor(2.0)).eval()
    source._initialization_source_cache_free = True
    plan = InitializationSourcePlan(_population(source))
    assert plan.failures
    assert "reserved functional execution metadata" in " ".join(plan.failures)


def test_source_config_buffer_cannot_alias_mutable_material_state():
    from dendra.func._initialization_sources import (
        initialization_source_runtime_buffer_ids,
    )

    source = _AffineSource()
    population = _population(source)
    source.offset = population.mech.materials.pool.amount
    assert id(source.offset) in initialization_source_runtime_buffer_ids(population)
    plan = InitializationSourcePlan(population)
    assert "alias runtime state" in " ".join(plan.failures)
    plan.validate_source()  # Snapshot transition admission remains separate.


def test_source_config_buffer_can_alias_registered_raw_parameter():
    from dendra.func._initialization_sources import (
        initialization_source_runtime_buffer_ids,
    )

    source = _AffineSource()
    source._buffers["offset"] = source.value
    population = _population(source)
    assert id(source.value) not in initialization_source_runtime_buffer_ids(population)
    plan = InitializationSourcePlan(population)
    assert not plan.failures, plan.failures
    plan.validate_initialization()


def test_source_config_buffer_can_share_static_registered_config():
    source = _AffineSource()
    population = _population(source)
    population.config = torch.nn.Module()
    population.config.register_buffer("offset", source.offset)
    plan = InitializationSourcePlan(population)
    assert not plan.failures, plan.failures


def test_network_source_config_cannot_capture_another_populations_runtime_state():
    from dendra.func._initialization_sources import (
        initialization_source_ownership_failures,
    )

    left, right = _population(), _population()
    left.mech.materials.pool._initial_sources.amount.offset = (
        right.mech.materials.pool.amount
    )
    network = torch.nn.Module()
    network.populations = torch.nn.ModuleDict({"left": left, "right": right})
    failures = initialization_source_ownership_failures(network)
    assert failures
    assert "Population 'left'" in failures[0]
    assert "alias runtime state" in failures[0]


def test_hidden_numerical_closure_is_not_an_explicit_initializer_leaf():
    hidden = torch.tensor(0.25)

    class _CapturedSource(_AffineSource):
        def forward(self):
            return self.value + hidden

    _failures(_CapturedSource())


def test_immutable_closure_and_its_changes_are_visible_to_fullgraph_guards():
    from dendra._bootstrap import torch_compiler_warning_context

    gain = 2.0

    class _CapturedSource(_AffineSource):
        def forward(self):
            return self.value * gain

    source = _CapturedSource()
    plan = InitializationSourcePlan(_population(source))
    assert not plan.failures, plan.failures
    expected = plan.source_binding_signature()

    def evaluate(value):
        if plan.source_binding_signature() != expected:
            raise FunctionalizationError(
                "Initial-value source dependencies changed; lower again."
            )
        return value + 1

    with torch_compiler_warning_context():
        compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(torch.tensor(1.0)), torch.tensor(2.0))
        gain = 3.0
        with pytest.raises(
            (FunctionalizationError, torch._dynamo.exc.Unsupported),
            match="changed|[Ee]xception|raise",
        ):
            compiled(torch.tensor(1.0))


@pytest.mark.parametrize(
    "method",
    (
        "named_modules",
        "modules",
        "parameters",
        "named_parameters",
        "buffers",
        "named_buffers",
    ),
)
def test_rejected_metadata_dispatch_is_not_executed_by_plan_or_snapshot_guards(method):
    class _MetadataSource(_AffineSource):
        calls = 0

    canonical = getattr(torch.nn.Module, method)

    def authored(self, *args, **kwargs):
        type(self).calls += 1
        return canonical(self, *args, **kwargs)

    setattr(_MetadataSource, method, authored)
    source = _MetadataSource()
    population = _population(source)
    before = type(source).calls
    plan = InitializationSourcePlan(population)
    assert plan.failures
    plan.validate_source()
    tuple(initialization_source_constant_slots(source))
    assert type(source).calls == before


@pytest.mark.parametrize("training", (False, True))
def test_negative_parameter_uses_shared_canonical_compute(training):
    source = NegativeParam(torch.tensor(-65.0, dtype=torch.float64), requires_grad=True)
    source.train(training)
    population = _population(source)
    plan = InitializationSourcePlan(population)
    assert not plan.failures, plan.failures
    plan.validate_initialization()
    assert source.training is training
    assert source._cache is None


def test_reserved_class_marker_is_rejected_without_running_its_descriptor():
    class _MarkerSource(_AffineSource):
        @property
        def _initialization_source_cache_free(self):
            raise AssertionError("reserved marker descriptor was executed")

    source = _MarkerSource()
    # Construct the ordinary field first so Native initial resolution never
    # sees the deliberately unsupported descriptor during fixture setup.
    population = _population()
    population.mech.materials.pool._initial_sources.amount = source
    population.mech.materials.pool._initial_sources.reserve = source
    plan = InitializationSourcePlan(population)
    assert plan.failures
    assert "reserved functional execution metadata" in " ".join(plan.failures)


def test_tensor_only_initialization_does_not_collect_module_recipe_dependencies(
    monkeypatch,
):
    from dendra.func import _initialization_sources as sources

    population = _population(torch.nn.Parameter(torch.tensor(1.0)))

    def fail(*_args, **_kwargs):
        raise AssertionError("module-source dependency discovery was executed")

    monkeypatch.setattr(sources, "_collect_dependencies", fail)
    monkeypatch.setattr(sources, "_alias_signature", fail)
    plan = InitializationSourcePlan(population)
    assert plan.source_binding_signature() == ()
    assert not plan.failures


@pytest.mark.parametrize("kind", (_HiddenUpperBounded, _GradRandomUpperBounded))
def test_canonical_bounded_compute_does_not_trust_authored_helper_dependencies(kind):
    _failures(kind(torch.tensor(2.0), max_val=3.0))


def test_canonical_bounded_compute_accepts_a_pure_authored_helper():
    source = _PureUpperBounded(torch.tensor(2.0), max_val=3.0)
    plan = InitializationSourcePlan(_population(source))
    assert not plan.failures, plan.failures
    plan.validate_initialization()


@pytest.mark.parametrize("method", ("forward", "_compute", "_apply_upper"))
def test_source_qualification_does_not_evaluate_authored_method_descriptors(method):
    base = _AffineSource if method == "forward" else Bounded

    class _DescriptorSource(base):
        reads = 0

        def evaluate(self, *args):
            return args[0] if args else self.value

    def getter(self):
        type(self).reads += 1
        return self.evaluate

    setattr(_DescriptorSource, method, property(getter))
    source = (
        _DescriptorSource()
        if method == "forward"
        else _DescriptorSource(torch.tensor(2.0), max_val=3.0)
    )
    population = _population()
    population.mech.materials.pool._initial_sources.amount = source
    population.mech.materials.pool._initial_sources.reserve = source
    before = _DescriptorSource.reads
    plan = InitializationSourcePlan(population)
    assert plan.failures
    plan.validate_source()  # Unsupported recipes still allow snapshot execution.
    assert _DescriptorSource.reads == before


def test_canonical_bounded_configuration_descriptors_are_not_evaluated_by_audit():
    class _DescriptorSource(Bounded):
        reads = 0

        @property
        def max_val(self):
            type(self).reads += 1
            return self.upper

        @max_val.setter
        def max_val(self, value):
            self.upper = value

    source = _DescriptorSource(torch.tensor(2.0), max_val=3.0)
    population = _population()
    population.mech.materials.pool._initial_sources.amount = source
    population.mech.materials.pool._initial_sources.reserve = source
    before = _DescriptorSource.reads
    plan = InitializationSourcePlan(population)
    assert plan.failures
    plan.validate_source()
    assert _DescriptorSource.reads == before


def test_canonical_bounded_nondata_configuration_descriptors_are_not_evaluated():
    class _Descriptor:
        def __get__(self, owner, kind=None):
            kind.reads += 1
            return 3.0

    class _DescriptorSource(Bounded):
        reads = 0
        max_val = _Descriptor()

    source = _DescriptorSource(torch.tensor(2.0), max_val=3.0)
    del source.max_val  # Expose the nondata descriptor after ordinary construction.
    population = _population()
    population.mech.materials.pool._initial_sources.amount = source
    population.mech.materials.pool._initial_sources.reserve = source
    before = _DescriptorSource.reads
    plan = InitializationSourcePlan(population)
    assert plan.failures
    plan.validate_source()
    assert _DescriptorSource.reads == before
