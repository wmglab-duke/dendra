"""Custom full-field operator admission, dependency, and read-only contracts."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.func._custom_material_processes import (
    CustomMaterialProcessPlan,
    custom_material_constant_slots,
    custom_material_process_failures,
)
from dendra.func._population import _deepcopy_execution_module
from dendra.func._types import FunctionalizationError
from dendra.models.mechanisms import Material, MaterialProcess

pytestmark = pytest.mark.cpu
DTYPE = torch.float64
KEY = ("pool", "amount")
OTHER = ("pool", "reserve")
_HIDDEN = torch.tensor(0.25, dtype=DTYPE)


@pytest.fixture(autouse=True)
def _diagnostic_purity():
    with dn.ctx(CHECK_PURITY=True):
        yield


class _Transfer(MaterialProcess):
    MaterialProcess.GLOBAL(rate=0.25)

    @property
    def material_fields(self):
        return (KEY, OTHER)

    def prepare_materials(self, fields, dt):
        return dt * self._resolve_quantity("rate", fields[KEY])

    def material_updates(self, fields, prepared):
        transfer = prepared * fields[KEY]
        return {KEY: fields[KEY] - transfer, OTHER: fields[OTHER] + transfer}


def _process(cls=_Transfer):
    c = torch.tensor([[0.1, 0.7, 1.3]], dtype=DTYPE)
    material = Material("pool", c.shape, fields={"amount": c, "reserve": c * 0.25})
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        process = cls(
            name="process",
            celsius=c.new_tensor(34.0),
            diameters=torch.ones_like(c),
            shape=c.shape,
            shape_f=c.shape,
        ).to(dtype=DTYPE)
        process._bind_materials({"pool": material}.__getitem__)
        process.train()
        process.populate()
    fields = {KEY: material._buffers["amount"], OTHER: material._buffers["reserve"]}
    return process, fields


def test_custom_plan_uses_shared_hooks_without_changing_source_or_rng():
    process, fields = _process()
    before = {name: value.clone() for name, value in process.named_buffers()}
    rng = torch.random.get_rng_state().clone()
    plan = CustomMaterialProcessPlan(process, fields, 0.125)
    assert plan.field_keys == (KEY, OTHER)
    assert plan.output_keys == (KEY, OTHER)
    assert plan.prepared_schema[1] == (((), DTYPE, torch.device("cpu")),)
    assert custom_material_process_failures(process) == ()
    plan.validate_source()
    plan.validate_compatible(_deepcopy_execution_module(process))
    torch.testing.assert_close(rng, torch.random.get_rng_state())
    for name, value in process.named_buffers():
        torch.testing.assert_close(value, before[name])


def test_class_tuple_field_declaration_and_context_independent_recipe():
    class Fixed(_Transfer):
        material_fields = (KEY, OTHER)

    process, fields = _process(Fixed)
    with dn.ctx(CHECK_PURITY=False):
        plan = CustomMaterialProcessPlan(process, fields, 0.125)
    assert not plan.diagnostic
    assert plan.source_binding_signature() == ()
    plan.validate_source()
    plan.validate_compatible(_deepcopy_execution_module(process))


@pytest.mark.parametrize("empty", ["tuple", "dict"])
def test_empty_preparation_is_supported(empty):
    class Empty(_Transfer):
        def prepare_materials(self, fields, dt):
            return () if empty == "tuple" else {}

        def material_updates(self, fields, prepared):
            return {KEY: fields[KEY] + 1.0}

    process, fields = _process(Empty)
    assert CustomMaterialProcessPlan(process, fields, 0.125).prepared_schema[1] == ()


@pytest.mark.parametrize("style", ["direct", "view", "sum", "arithmetic", "grad_only"])
def test_preparation_rejects_evolving_field_reads(style):
    class ReadsFields(_Transfer):
        def prepare_materials(self, fields, dt):
            if style == "direct":
                return fields[KEY]
            if style == "view":
                return fields[KEY].view(-1)
            if style == "sum":
                return fields[KEY].sum()
            if style == "arithmetic":
                return fields[KEY] * dt
            if torch.is_grad_enabled():
                return fields[KEY] * dt
            return fields[KEY].new_tensor(0.5)

    process, fields = _process(ReadsFields)
    with pytest.raises(FunctionalizationError, match="field|effectful"):
        CustomMaterialProcessPlan(process, fields, 0.125)


def test_preparation_allows_tensor_factories_using_field_metadata():
    class Metadata(_Transfer):
        def prepare_materials(self, fields, dt):
            like = fields[KEY]
            return like.new_full(like.shape, dt), torch.ones_like(like)

        def material_updates(self, fields, prepared):
            first, second = prepared
            return {KEY: fields[KEY] + first * second}

    process, fields = _process(Metadata)
    plan = CustomMaterialProcessPlan(process, fields, 0.125)
    assert len(plan.prepared_schema[1]) == 2


@pytest.mark.parametrize("where", ["prepare", "update", "helper", "property"])
def test_hidden_global_tensors_are_rejected_before_evaluation(where):
    class Hidden(_Transfer):
        @property
        def material_fields(self):
            return (KEY, OTHER) if where != "property" or _HIDDEN is not None else ()

        def _extra(self):
            return _HIDDEN

        def prepare_materials(self, fields, dt):
            if where == "prepare":
                return _HIDDEN * dt
            if where == "helper":
                return self._extra() * dt
            return fields[KEY].new_tensor(dt)

        def material_updates(self, fields, prepared):
            return {
                KEY: fields[KEY] + prepared + (_HIDDEN if where == "update" else 0.0)
            }

    process, fields = _process(Hidden)
    with pytest.raises(FunctionalizationError, match="hidden mutable numerical"):
        CustomMaterialProcessPlan(process, fields, 0.125)


@pytest.mark.parametrize("kind", ["input", "parameter", "python", "prepared", "random"])
def test_effectful_custom_operators_are_rejected(kind):
    class Effectful(_Transfer):
        def material_updates(self, fields, prepared):
            if kind == "input":
                fields[KEY].add_(1)
            elif kind == "parameter":
                self.rate.add_(1)
            elif kind == "python":
                self.calls = 1
            elif kind == "prepared":
                prepared.add_(1)
            else:
                return {KEY: fields[KEY] + torch.rand_like(fields[KEY])}
            return {KEY: fields[KEY]}

    process, fields = _process(Effectful)
    before = torch.random.get_rng_state().clone()
    with pytest.raises(FunctionalizationError, match="pure|read-only|effectful|RNG"):
        CustomMaterialProcessPlan(process, fields, 0.125)
    torch.testing.assert_close(before, torch.random.get_rng_state())


@pytest.mark.parametrize("check_purity", (False, True))
@pytest.mark.parametrize(
    "name",
    [
        "advance_materials",
        "configure_process",
        "set_dt",
        "_reset_initialization_timestep",
    ],
)
def test_custom_runtime_lifecycle_overrides_are_rejected(name, check_purity):
    cls = type("Legacy", (_Transfer,), {name: lambda self, *args, **kwargs: None})
    process, _ = _process(cls)
    with dn.ctx(CHECK_PURITY=check_purity):
        assert any(
            "canonical lifecycle" in item
            for item in custom_material_process_failures(process)
        )


def test_hook_descriptors_fail_without_calling_the_getter():
    calls = []

    class Descriptor(_Transfer):
        @property
        def prepare_materials(self):
            calls.append("getter")
            return lambda fields, dt: dt

    process, _ = _process(Descriptor)
    assert custom_material_process_failures(process)
    assert calls == []


def test_helper_override_cannot_capture_hidden_tensor():
    class Helper(_Transfer):
        def _expand_material_operand(self, value, like, *, local=False):
            return value * _HIDDEN

    process, _ = _process(Helper)
    assert any(
        "hidden mutable numerical" in item
        for item in custom_material_process_failures(process)
    )


@pytest.mark.parametrize("change", ["hook", "default", "helper", "config", "global"])
def test_source_guards_reject_changed_authored_recipe(change, monkeypatch):
    class Configured(_Transfer):
        factor = 2.0

        def material_updates(self, fields, prepared, scale=1.0):
            return {KEY: fields[KEY] + prepared * scale * self.factor}

    process, fields = _process(Configured)
    plan = CustomMaterialProcessPlan(process, fields, 0.125)
    if change == "hook":
        monkeypatch.setattr(
            Configured,
            "material_updates",
            lambda self, fields, prepared: {KEY: fields[KEY]},
        )
    elif change == "default":
        monkeypatch.setattr(Configured.material_updates, "__defaults__", (2.0,))
    elif change == "helper":
        monkeypatch.setattr(
            Configured,
            "_resolve_quantity",
            lambda self, value, like, **kwargs: like.new_tensor(1.0),
        )
    elif change == "config":
        monkeypatch.setattr(Configured, "factor", 3.0)
    else:
        monkeypatch.setitem(Configured.material_updates.__globals__, "KEY", OTHER)
    with pytest.raises(FunctionalizationError, match="changed"):
        plan.validate_source()


def test_registered_child_module_parameters_and_buffers_remain_explicit():
    class Child(_Transfer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.transform = torch.nn.Linear(1, 1, dtype=DTYPE)
            self.transform.register_buffer("offset", torch.tensor(0.5, dtype=DTYPE))
            self.register_buffer("gain", torch.tensor(0.25, dtype=DTYPE))

        def prepare_materials(self, fields, dt):
            return dt * self.gain

        def material_updates(self, fields, prepared):
            return {
                KEY: self.transform(fields[KEY].unsqueeze(-1)).squeeze(-1) * prepared
            }

    process, fields = _process(Child)
    plan = CustomMaterialProcessPlan(process, fields, 0.125)
    assert {name for name, _ in custom_material_constant_slots(process)} == {
        "gain",
        "transform.offset",
    }
    plan.validate_source()


@pytest.mark.parametrize("bad", ["none", "scalar", "key", "shape", "dtype"])
def test_invalid_prepared_and_output_schemas_fail_closed(bad):
    class Invalid(_Transfer):
        def prepare_materials(self, fields, dt):
            if bad == "none":
                return None
            if bad == "scalar":
                return dt
            return fields[KEY].new_tensor(dt)

        def material_updates(self, fields, prepared):
            if bad == "key":
                return {("missing", "c"): fields[KEY]}
            if bad == "shape":
                return {KEY: fields[KEY].reshape(-1)}
            return {KEY: fields[KEY].to(torch.float32)}

    process, fields = _process(Invalid)
    with pytest.raises(
        FunctionalizationError, match="preparation|declared|shape, dtype"
    ):
        CustomMaterialProcessPlan(process, fields, 0.125)


def test_private_execution_guards_output_schema_before_publication():
    class Variable(_Transfer):
        def material_updates(self, fields, prepared):
            return {KEY: fields[KEY][: prepared.shape[0]]}

        def prepare_materials(self, fields, dt):
            return fields[KEY].new_ones((fields[KEY].shape[0],))

    process, fields = _process(Variable)
    plan = CustomMaterialProcessPlan(process, fields, 0.125)
    private = _deepcopy_execution_module(process)
    plan.configure_execution(private)
    with pytest.raises(FunctionalizationError, match="shape, dtype"):
        private.material_updates(fields, torch.ones(0, dtype=DTYPE))
