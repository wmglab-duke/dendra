"""Native LFP configuration separates authored sources from recorded history."""

from __future__ import annotations

import pickle
import weakref
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from dendra.models._lfp_callbacks import (
    _materialize_lfp_fields,
    _project_lfp,
    _validate_lfp_fields,
)
from dendra.models.callbacks import LFP

pytestmark = pytest.mark.cpu


def _model(shape=(2, 3), *, dtype=torch.float64):
    current = torch.arange(1, torch.Size(shape).numel() + 1, dtype=dtype).reshape(shape)
    return SimpleNamespace(
        integrator=SimpleNamespace(imem=True),
        i_membrane=current,
        device=lambda: current.device,
    )


@pytest.mark.parametrize("batch", ((), (2,), (2, 1)))
@pytest.mark.parametrize("field_shape", ((), (3,), (2, 3)))
def test_native_default_projection_preserves_batch_axes_and_broadcasts_fields(
    batch, field_shape
):
    model = _model((*batch, 2, 3))
    field = torch.full(field_shape, 1.5, dtype=torch.float32)
    callback = LFP([field, field])
    callback.pre_loop_hook(model)
    expected = (model.i_membrane * field.to(torch.float64)).sum((-1, -2))
    torch.testing.assert_close(
        callback.lfp, torch.stack((expected, expected), -1)[None]
    )
    assert callback.lfp.shape == (1, *batch, 2)
    assert callback.v_unit.shape == (2, *field_shape)
    assert callback.v_unit.dtype == torch.float64
    callback.post_step_hook(model)
    assert callback.lfp.shape == (2, *batch, 2)
    assert callback.t.numel() == 0


def test_default_and_explicit_default_looking_rules_remain_distinct():
    model = _model((2, 2, 3))
    field = torch.ones(2, 3, dtype=torch.float64)
    automatic = LFP([field])
    explicit = LFP([field], rule="...,n...->n")
    assert automatic.rule == explicit.rule == "...,n...->n"
    assert automatic._functional_lfp_configuration()["rule"] is None
    assert explicit._functional_lfp_configuration()["rule"] == "...,n...->n"
    automatic.pre_loop_hook(model)
    explicit.pre_loop_hook(model)
    assert automatic.lfp.shape == (1, 2, 1)
    assert explicit.lfp.shape == (1, 1)
    torch.testing.assert_close(automatic.lfp.sum(1), explicit.lfp)
    automatic.rule = "...,n...->n"
    assert automatic._functional_lfp_configuration()["rule"] == "...,n...->n"
    automatic.reset()
    automatic.pre_loop_hook(model)
    torch.testing.assert_close(automatic.lfp, explicit.lfp)
    automatic.rule = None
    assert automatic._functional_lfp_configuration()["rule"] is None


def test_explicit_projection_rule_uses_converted_fields_and_exact_einsum():
    model = _model((2, 2, 3))
    field = torch.ones(2, 3, dtype=torch.float32)
    callback = LFP([field, field * 2], rule="bnc,knc->bk")
    callback.pre_loop_hook(model)
    expected = torch.einsum("bnc,knc->bk", model.i_membrane, callback.v_unit)
    torch.testing.assert_close(callback.lfp, expected[None])
    assert callback.v_unit.dtype == model.i_membrane.dtype


def test_native_subclass_rule_setter_remains_authoritative():
    class RewrittenLFP(LFP):
        @property
        def rule(self):
            return self._rule

        @rule.setter
        def rule(self, value):
            self._rule = "bnc,knc->k"

    model = _model((2, 2, 3))
    callback = RewrittenLFP([torch.ones(2, 3)])
    assert callback._functional_lfp_configuration()["rule"] == "bnc,knc->k"
    callback.pre_loop_hook(model)
    expected = torch.einsum("bnc,knc->k", model.i_membrane, callback.v_unit)
    torch.testing.assert_close(callback.lfp, expected[None])
    assert callback.lfp.shape == (1, 1)


def test_dynamic_native_public_rule_getter_remains_authoritative():
    class DynamicLFP(LFP):
        @property
        def rule(self):
            return self.__dict__.get("_dynamic_rule", self._rule)

        @rule.setter
        def rule(self, value):
            self._rule = value

    model = _model((2, 2, 3))
    callback = DynamicLFP([torch.ones(2, 3)])
    assert callback._functional_lfp_configuration()["rule"] is None
    callback._dynamic_rule = "bnc,knc->k"
    assert callback._functional_lfp_configuration()["rule"] == "bnc,knc->k"
    callback.pre_loop_hook(model)
    expected = torch.einsum("bnc,knc->k", model.i_membrane, callback.v_unit)
    torch.testing.assert_close(callback.lfp, expected[None])


def test_configuration_retains_raw_aliases_without_extra_registered_sources():
    field = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.float32))
    supplied = [field, field]
    callback = LFP(supplied)
    configuration = callback._functional_lfp_configuration()
    assert configuration["v_unit"] is not supplied
    assert configuration["v_unit"][0] is configuration["v_unit"][1] is field
    supplied.append(torch.zeros_like(field))
    assert len(callback._functional_lfp_configuration()["v_unit"]) == 2
    assert list(callback.named_parameters()) == []
    assert tuple(dict(callback.named_buffers())) == ("v_unit",)
    assert tuple(callback.state_dict()) == ("v_unit",)
    assert callback._published_lfp_fields_snapshot.requires_grad is False
    published = callback.v_unit
    with torch.no_grad():
        field.add_(2)
    callback.pre_loop_hook(_model())
    assert callback.v_unit is not published
    torch.testing.assert_close(
        callback.v_unit, torch.full((2, 2, 3), 3.0, dtype=torch.float64)
    )
    assert callback._functional_lfp_configuration()["v_unit"][0] is field


def test_tensor_subclass_fields_preserve_source_identity_and_dispatch():
    class Field(torch.Tensor):
        pass

    field = torch.ones(2, 3, dtype=torch.float32).as_subclass(Field)
    callback = LFP([field, field])
    assert type(callback.v_unit) is Field
    assert callback._functional_lfp_configuration()["v_unit"][0] is field
    callback.pre_loop_hook(_model())
    assert type(callback.v_unit) is Field and type(callback.lfp) is Field
    assert callback._functional_lfp_configuration()["v_unit"][0] is field
    callback.v_unit.add_(2)
    configured = callback._functional_lfp_configuration()["v_unit"]
    assert type(configured[0]) is Field and configured[0] is not field
    torch.testing.assert_close(configured[0], torch.full_like(configured[0], 3))


@pytest.mark.parametrize("inference", (False, True))
def test_public_buffer_edits_override_raw_sources(inference):
    field = torch.ones(2, 3, dtype=torch.float64)
    with torch.inference_mode(inference):
        callback = LFP([field])
        public = callback.v_unit
        public.add_(4)
    selected = callback._functional_lfp_configuration()["v_unit"]
    assert selected[0] is not field
    torch.testing.assert_close(selected[0], torch.full_like(field, 5))
    field.fill_(17)
    callback.pre_loop_hook(_model())
    torch.testing.assert_close(
        callback.v_unit, torch.full((1, 2, 3), 5.0, dtype=torch.float64)
    )
    replacement = torch.full_like(callback.v_unit, 7)
    callback.v_unit = replacement
    selected = callback._functional_lfp_configuration()["v_unit"]
    selected[0].add_(2)
    torch.testing.assert_close(replacement, torch.full_like(replacement, 9))
    callback.pre_loop_hook(_model())
    torch.testing.assert_close(callback.v_unit, torch.full_like(replacement, 9))


def test_module_dtype_conversion_makes_public_buffer_authoritative():
    field = torch.ones(2, 3, dtype=torch.float32)
    callback = LFP([field])
    original = callback.v_unit
    callback.to(dtype=torch.float64)
    assert callback.v_unit is not original and callback.v_unit.dtype == torch.float64
    selected = callback._functional_lfp_configuration()["v_unit"]
    assert selected[0].dtype == torch.float64 and selected[0] is not field
    field.fill_(23)
    callback.pre_loop_hook(_model())
    torch.testing.assert_close(
        callback.v_unit, torch.ones(1, 2, 3, dtype=torch.float64)
    )


@pytest.mark.parametrize("operation", ("assign", "to", "load"))
def test_public_source_replacement_releases_retired_publication_storage(operation):
    callback = LFP([torch.ones(2, 3, dtype=torch.float32)])
    published = weakref.ref(callback.v_unit)
    snapshot = weakref.ref(callback._published_lfp_fields_snapshot)
    raw = weakref.ref(callback._configured_lfp_fields[0])
    if operation == "assign":
        callback.v_unit = torch.zeros(1, 2, 3, dtype=torch.float32)
    elif operation == "to":
        callback.to(dtype=torch.float64)
    else:
        callback.load_state_dict({"v_unit": torch.zeros_like(callback.v_unit)})
    assert snapshot() is None and raw() is None
    if operation != "load":
        assert published() is None
    else:
        assert published() is callback.v_unit
    assert callback._configured_lfp_fields is None
    assert callback._configured_lfp_stacked_source is None
    assert callback._published_lfp_fields is None
    assert callback._published_lfp_fields_version is None
    assert callback._published_lfp_fields_snapshot is None


def test_noop_module_conversion_keeps_live_original_field_sources():
    field = torch.ones(2, 3, dtype=torch.float32)
    callback = LFP([field])
    publication = callback.v_unit
    snapshot = callback._published_lfp_fields_snapshot
    callback.to(dtype=torch.float32)
    assert callback.v_unit is publication
    assert callback._published_lfp_fields_snapshot is snapshot
    assert callback._functional_lfp_configuration()["v_unit"][0] is field


@pytest.mark.parametrize("copy_kind", (None, "deepcopy", "pickle"))
def test_same_value_public_write_overrides_mutated_raw_source(copy_kind):
    field = torch.ones(2, 3, dtype=torch.float64)
    callback = LFP([field])
    field.fill_(17)
    callback.v_unit.copy_(torch.ones_like(callback.v_unit))
    if copy_kind == "deepcopy":
        callback = deepcopy(callback)
    elif copy_kind == "pickle":
        callback = pickle.loads(pickle.dumps(callback))
    callback.pre_loop_hook(_model())
    torch.testing.assert_close(
        callback.v_unit, torch.ones(1, 2, 3, dtype=torch.float64)
    )


@pytest.mark.parametrize("capture", ("grad", "no_grad", "inference"))
def test_public_trainable_override_keeps_gradient_connection_during_cold_capture(
    capture,
):
    callback = LFP([torch.ones(2, 3)])
    public = torch.nn.Parameter(torch.full((2, 2, 3), 1.5, dtype=torch.float32))
    callback.v_unit = public
    mode = (
        torch.inference_mode()
        if capture == "inference"
        else torch.no_grad()
        if capture == "no_grad"
        else nullcontext()
    )
    with mode:
        sources = callback._functional_lfp_configuration()["v_unit"]
    assert all(field.requires_grad and not field.is_inference() for field in sources)
    model = _model()
    for iteration in range(2):
        if iteration:
            with torch.no_grad():
                public.add_(1)
        fields = _materialize_lfp_fields(
            sources, device=model.i_membrane.device, dtype=model.i_membrane.dtype
        )
        projection = _project_lfp(model.i_membrane, fields)
        gradient = torch.autograd.grad(projection.sum(), public)[0]
        torch.testing.assert_close(
            gradient, model.i_membrane.expand_as(public).to(public.dtype)
        )


def test_native_parameter_override_keeps_public_registration_identity_and_gradients():
    callback = LFP([torch.ones(2, 3)])
    public = torch.nn.Parameter(torch.full((2, 2, 3), 1.5, dtype=torch.float32))
    callback.v_unit = public
    model = _model()
    for iteration in range(2):
        callback.reset()
        if iteration:
            with torch.no_grad():
                public.add_(1)
        callback.pre_loop_hook(model)
        callback.post_step_hook(model)
        assert callback.v_unit is public
        assert dict(callback.named_parameters())["v_unit"] is public
        assert "v_unit" not in dict(callback.named_buffers())
        assert tuple(callback.state_dict()) == ("v_unit",)
        expected = torch.einsum("nc,knc->k", model.i_membrane, public.to(torch.float64))
        torch.testing.assert_close(callback.lfp, expected.expand(2, -1))
        gradient = torch.autograd.grad(callback.lfp.sum(), public)[0]
        torch.testing.assert_close(
            gradient, (2 * model.i_membrane).expand_as(public).to(public.dtype)
        )


def test_native_public_leaf_view_owner_retains_gradients_after_optimizer_updates():
    callback = LFP([torch.ones(2, 3)])
    public = (
        torch.linspace(0.25, 1.25, 12, dtype=torch.float64)
        .reshape(2, 2, 3)
        .requires_grad_()
    )
    assert (
        public.is_leaf and public._base is not None and not public._base.requires_grad
    )
    callback.v_unit = public
    model = _model()
    model.i_membrane.requires_grad_()
    for iteration in range(2):
        callback.reset()
        if iteration:
            with torch.no_grad():
                public.add_(0.25)
        callback.pre_loop_hook(model)
        assert callback._lfp_stacked_source() is public
        observed = callback.lfp
        expected = torch.einsum("nc,knc->k", model.i_membrane, public)[None]
        torch.testing.assert_close(observed, expected)
        gradients = torch.autograd.grad(
            observed.square().sum(), (public, model.i_membrane)
        )
        reference = torch.autograd.grad(
            expected.square().sum(), (public, model.i_membrane)
        )
        torch.testing.assert_close(gradients, reference)


def test_retiring_native_public_source_releases_retained_owner_and_serialized_choice():
    callback = LFP([torch.ones(2, 3)])
    public = torch.ones(1, 2, 3, dtype=torch.float64)
    callback.v_unit = public
    callback.pre_loop_hook(_model())
    assert callback._lfp_stacked_source() is public
    owner = weakref.ref(public)
    callback.v_unit.add_(1)
    state = callback.__getstate__()
    assert state["_configured_lfp_stacked_source"] is None
    del state, public
    callback.v_unit = torch.zeros(1, 2, 3, dtype=torch.float64)
    assert callback._configured_lfp_stacked_source is None
    assert owner() is None


def test_unchanged_inference_nan_publication_retains_repaired_raw_sources():
    field = torch.full((2, 3), float("nan"), dtype=torch.float64)
    with torch.inference_mode():
        callback = LFP([field])
    field.fill_(2)
    assert callback._functional_lfp_configuration()["v_unit"][0] is field
    callback.pre_loop_hook(_model())
    assert torch.isfinite(callback.lfp).all()
    torch.testing.assert_close(callback.v_unit, field[None])


def test_inference_content_fallback_preserves_integer_exactness():
    field = torch.full((2, 3), 2**60, dtype=torch.int64)
    with torch.inference_mode():
        callback = LFP([field])
        callback.v_unit[0, 0, 0] += 1
    selected = callback._functional_lfp_configuration()["v_unit"]
    assert selected[0] is not field
    assert selected[0][0, 0].item() == 2**60 + 1


@pytest.mark.parametrize("assign", (False, True))
@pytest.mark.parametrize("value", (1.0, 5.0))
def test_state_dict_loading_is_authoritative_even_for_equal_public_values(
    assign, value
):
    field = torch.ones(2, 3, dtype=torch.float64)
    callback = LFP([field])
    payload = {"v_unit": torch.full_like(callback.v_unit, value)}
    field.fill_(17)
    callback.load_state_dict(payload, assign=assign)
    fields = callback._functional_lfp_configuration()["v_unit"]
    assert fields[0] is not field
    callback.pre_loop_hook(_model())
    torch.testing.assert_close(callback.v_unit, payload["v_unit"])
    assert tuple(callback.state_dict()) == ("v_unit",)


@pytest.mark.parametrize("copy_kind", ("deepcopy", "pickle"))
@pytest.mark.parametrize("edited", (False, True))
def test_copy_preserves_source_choice_and_authored_default_rule(copy_kind, edited):
    field = torch.ones(2, 3, dtype=torch.float64)
    callback = LFP([field, field])
    callback.pre_loop_hook(_model())
    if edited:
        callback.v_unit.add_(4)
    copied = (
        deepcopy(callback)
        if copy_kind == "deepcopy"
        else pickle.loads(pickle.dumps(callback))
    )
    configuration = copied._functional_lfp_configuration()
    assert configuration["rule"] is None
    assert len(configuration["v_unit"]) == 2
    assert copied.v_unit is not callback.v_unit
    if edited:
        torch.testing.assert_close(
            configuration["v_unit"][0], torch.full_like(field, 5)
        )
    else:
        assert configuration["v_unit"][0] is configuration["v_unit"][1]
    copied.pre_loop_hook(_model())
    expected = torch.full((2, 2, 3), 5.0 if edited else 1.0, dtype=torch.float64)
    torch.testing.assert_close(copied.v_unit, expected)
    assert len(copied._lfp) == 2 and len(callback._lfp) == 1


def test_legacy_full_object_state_uses_public_effective_fields_and_rule():
    callback = LFP([torch.ones(2, 3, dtype=torch.float64)])
    for name in (
        "_configured_lfp_fields",
        "_configured_lfp_stacked_source",
        "_configured_lfp_rule",
        "_published_lfp_fields",
        "_published_lfp_fields_version",
        "_published_lfp_fields_snapshot",
    ):
        callback.__dict__.pop(name, None)
    configuration = callback._functional_lfp_configuration()
    assert configuration["rule"] == "...,n...->n"
    torch.testing.assert_close(configuration["v_unit"][0], callback.v_unit[0])
    callback.pre_loop_hook(_model((2, 2, 3)))
    assert callback.lfp.shape == (1, 1)


@pytest.mark.parametrize("inference_constructor", (False, True))
def test_fresh_native_runs_rebuild_field_graphs_and_retain_raw_gradient_aliases(
    inference_constructor,
):
    field = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.float32))
    with torch.inference_mode(inference_constructor):
        callback = LFP([field, field])
    model = _model()
    model.i_membrane.requires_grad_()
    for iteration in range(3):
        callback.reset()
        if iteration == 2:
            with torch.no_grad():
                field.add_(0.5)
        callback.pre_loop_hook(model)
        projection = (model.i_membrane * field.to(torch.float64)).sum()
        gradients = torch.autograd.grad(
            callback.lfp.square().sum(), (field, model.i_membrane)
        )
        torch.testing.assert_close(
            gradients[0], (4 * projection * model.i_membrane).to(torch.float32)
        )
        torch.testing.assert_close(
            gradients[1], 4 * projection * field.to(torch.float64)
        )
        assert callback._functional_lfp_configuration()["v_unit"][0] is field
        assert callback.v_unit.requires_grad and not callback.v_unit.is_inference()


def test_native_run_reuse_keeps_baseline_and_reset_history_behavior():
    model = _model()
    callback = LFP([torch.ones(2, 3)])
    callback.pre_loop_hook(model)
    callback.post_step_hook(model)
    callback.pre_loop_hook(model)
    assert callback.lfp.shape == (3, 1) and callback.t.numel() == 0
    configuration = callback._functional_lfp_configuration()
    callback.reset()
    assert callback._lfp == [] and callback._t == []
    assert (
        callback._functional_lfp_configuration()["v_unit"][0]
        is configuration["v_unit"][0]
    )


def test_native_imem_validation_precedes_materialization_and_recording():
    callback = LFP([torch.ones(2, 3)])
    original = callback.v_unit
    model = _model()
    model.integrator.imem = False
    with pytest.raises(RuntimeError, match="IMEM=1"):
        callback.pre_loop_hook(model)
    assert callback.v_unit is original and callback._lfp == []


@pytest.mark.parametrize("field_shape", ((4,), (2, 2, 3)))
def test_native_default_rejects_fields_that_expand_the_population_shape(field_shape):
    callback = LFP([torch.ones(field_shape)])
    with pytest.raises((RuntimeError, ValueError), match="shape|broadcast"):
        callback.pre_loop_hook(_model())
    assert callback._lfp == []


def test_shared_helpers_retain_fields_and_preserve_explicit_carry_layout():
    field = torch.ones(2, 3, dtype=torch.float32, requires_grad=True)
    sources = _validate_lfp_fields([field, field])
    assert sources[0] is sources[1] is field
    current = _model((2, 2, 3)).i_membrane
    plain = _materialize_lfp_fields(sources, device=current.device, dtype=current.dtype)
    padded = _materialize_lfp_fields(
        sources, device=current.device, dtype=current.dtype, padding=1
    )
    assert plain.shape == (2, 2, 3) and padded.shape == (2, 1, 2, 3)
    torch.testing.assert_close(
        _project_lfp(current, plain, padding=1), _project_lfp(current, padded)
    )
    explicit = _project_lfp(current, plain, rule="bnc,knc->bk")
    torch.testing.assert_close(explicit, torch.einsum("bnc,knc->bk", current, plain))
    gradient = torch.autograd.grad(explicit.sum(), field)[0]
    torch.testing.assert_close(gradient, (2 * current.sum(0)).to(torch.float32))
