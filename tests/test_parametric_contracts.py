"""Correctness contracts for bounded parameters, caches, and serialization."""

from __future__ import annotations

import copy
import math
import pickle

import pytest
import torch
import torch.nn.functional as F

import dendra.models.parametric as M
from dendra import ctx

_DECLARATION_NAMES = (
    "_params_declarations",
    "_params_p_declarations",
    "_params_n_declarations",
    "_flags_declarations",
    "_global_declarations",
    "_global_p_declarations",
    "_global_n_declarations",
    "_range_declarations",
    "_range_p_declarations",
    "_range_n_declarations",
    "_batch_declarations",
    "_batch_p_declarations",
    "_batch_n_declarations",
    "_rng_declarations",
    "_global_rand_declarations",
    "_range_rand_declarations",
    "_batch_rand_declarations",
    "_global_noise_declarations",
    "_range_noise_declarations",
    "_batch_noise_declarations",
    "_table_declarations",
)


@pytest.fixture(autouse=True)
def isolated_parameter_declarations():
    snapshots = []
    for owner in (M.SimpleParameterized, M.Parameterized):
        for name in _DECLARATION_NAMES:
            if hasattr(owner, name):
                snapshots.append((owner, name, list(getattr(owner, name))))
                setattr(owner, name, [])
    try:
        yield
    finally:
        for owner, name, value in snapshots:
            setattr(owner, name, value)


def test_softplus_inverse_round_trips_small_and_large_values():
    values = torch.tensor([1e-6, 0.1, 2.0, 50.0], dtype=torch.float64)
    inverse = M.softplus_inv(values, beta=1.7, threshold=20.0)
    torch.testing.assert_close(
        F.softplus(inverse, beta=1.7, threshold=20.0),
        values,
        rtol=1e-10,
        atol=1e-12,
    )


def test_softplus_inverse_has_finite_gradients_across_numeric_branches():
    values = torch.tensor([1e-8, 0.1, 100.0], requires_grad=True)

    inverse = M.softplus_inv(values)
    inverse.sum().backward()

    assert torch.isfinite(inverse).all()
    assert torch.isfinite(values.grad).all()
    torch.testing.assert_close(F.softplus(inverse), values, rtol=1e-5, atol=1e-10)


def test_softplus_inverse_honors_low_linear_threshold():
    values = torch.tensor([0.2, 2.0], dtype=torch.float64)
    inverse = M.softplus_inv(values, beta=1.0, threshold=1.0)
    torch.testing.assert_close(F.softplus(inverse, beta=1.0, threshold=1.0), values)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: M.PositiveParam(1.0),
        lambda: M.Bounded(1.0, max_val=2.0, cap_mode="softcap"),
        lambda: M.Bounded(
            1.0,
            min_val=0.0,
            max_val=2.0,
            cap_mode="hard-ste",
        ),
    ],
    ids=("lower-softplus", "upper-softcap", "bounded-hard-cap"),
)
def test_scalar_softplus_parameters_accept_empty_functional_vmap(factory):
    parameter = factory()
    raw = parameter.rho.detach().clone()

    def resolve_lane(rho):
        return torch.func.functional_call(parameter, {"rho": rho}, ())

    lanes = raw + raw.new_tensor((-0.2, 0.0, 0.3))
    actual = torch.vmap(resolve_lane)(lanes)
    expected = torch.stack(tuple(resolve_lane(lane) for lane in lanes))
    torch.testing.assert_close(actual, expected)

    empty = torch.vmap(resolve_lane)(raw.new_empty((0,)))
    assert empty.shape == (0,)


@pytest.mark.parametrize(
    "value,constraint,expected_type",
    [
        (1.0, "positive", M.PositiveParam),
        (-1.0, "negative", M.NegativeParam),
    ],
)
@pytest.mark.parametrize("requires_grad", [False, True])
def test_constrained_to_param_preserves_requested_dtype_and_trainability(
    value, constraint, expected_type, requires_grad
):
    parameter = M.to_param(
        value,
        **{constraint: True},
        requires_grad=requires_grad,
        dtype=torch.float64,
    )

    assert isinstance(parameter, expected_type)
    assert parameter.rho.dtype == torch.float64
    assert parameter.rho.requires_grad is requires_grad
    torch.testing.assert_close(parameter(), torch.tensor(value, dtype=torch.float64))


def test_explicit_dtype_wins_over_context_default_and_survives_shape_mutation():
    with ctx(DTYPE="float32"):
        parameter = M.to_param(
            2.0, positive=True, dtype=torch.float64, requires_grad=True
        )
        parameter.batch(2)

    assert parameter.rho.dtype == torch.float64
    assert parameter.rho.requires_grad


def test_to_param_preserves_existing_floating_tensor_precision_by_default():
    source = torch.tensor(
        [0.12345678901234568, 1.9876543210987654], dtype=torch.float64
    )

    with ctx(DTYPE="float32"):
        preserved = M.to_param(source)
        converted = M.to_param(source, dtype=torch.float32)
        integer = M.to_param(torch.tensor([1, 2], dtype=torch.int64))

    assert preserved.dtype == torch.float64
    torch.testing.assert_close(preserved, source, rtol=0.0, atol=0.0)
    assert converted.dtype == torch.float32
    torch.testing.assert_close(converted, source.float(), rtol=0.0, atol=0.0)
    # Integer inputs still use the active floating construction default.
    assert integer.dtype == torch.float32


@pytest.mark.parametrize("beta", [0.0, -1.0, math.inf, math.nan])
def test_softplus_inverse_and_bounded_reject_invalid_beta(beta):
    with pytest.raises(ValueError, match="beta.*finite and positive"):
        M.softplus_inv(torch.tensor(1.0), beta=beta)
    with pytest.raises(ValueError, match="beta.*finite and positive"):
        M.Bounded(1.0, min_val=0.0, beta=beta)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"min_val": 0.0, "lower_mode": "unknown"}, "lower_mode"),
        ({"max_val": 10.0, "cap_mode": "unknown"}, "cap_mode"),
        ({"max_val": 10.0, "cap_mode": "sigmoid"}, "requires both"),
        (
            {"min_val": 0.0, "max_val": 10.0, "cap_mode": "softcap"},
            "upper-only",
        ),
    ],
)
def test_bounded_rejects_unsupported_transform_configurations(kwargs, message):
    with pytest.raises(ValueError, match=message):
        M.Bounded(1.0, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_val": float("nan")},
        {"min_val": -float("inf")},
        {"max_val": float("inf")},
    ],
)
def test_bounded_rejects_nonfinite_one_sided_bounds(kwargs):
    with pytest.raises(ValueError, match="finite"):
        M.Bounded(1.0, **kwargs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bounded_endpoint_initialization_is_finite_in_reduced_precision(dtype):
    parameter = M.Bounded(
        torch.tensor([0.0, 10.0], dtype=dtype), min_val=0.0, max_val=10.0
    )

    assert torch.isfinite(parameter.rho).all()
    assert torch.isfinite(parameter()).all()
    assert torch.all(parameter() >= 0.0)
    assert torch.all(parameter() <= 10.0)


@pytest.mark.parametrize("initial", [2.0, 8.0, 9.0])
def test_upper_only_softcap_initialization_and_set_are_true_inverses(initial):
    parameter = M.Bounded(initial, max_val=10.0, requires_grad=True)
    torch.testing.assert_close(parameter(), torch.tensor(initial))

    parameter.eval()
    parameter.set(initial - 0.75)
    torch.testing.assert_close(parameter(), torch.tensor(initial - 0.75))
    assert parameter().item() <= 10.0


@pytest.mark.parametrize(
    "kwargs,initial,updated",
    [
        ({}, 2.0, -3.0),
        ({"min_val": 0.0}, 2.0, 3.0),
        ({"max_val": 10.0}, 8.0, 4.0),
        ({"min_val": 0.0, "max_val": 10.0}, 6.0, 3.0),
        (
            {"min_val": 0.0, "max_val": 10.0, "cap_mode": "hard-ste"},
            6.0,
            12.0,
        ),
    ],
)
def test_bounded_set_round_trip_and_forward_bounds(kwargs, initial, updated):
    parameter = M.Bounded(initial, requires_grad=True, **kwargs)
    torch.testing.assert_close(parameter(), torch.tensor(initial))
    parameter.set(updated)
    value = parameter()

    expected = updated
    if kwargs.get("min_val") is not None:
        expected = max(expected, kwargs["min_val"])
    if kwargs.get("max_val") is not None:
        expected = min(expected, kwargs["max_val"])
    torch.testing.assert_close(value, torch.tensor(float(expected)))


def test_bounded_ste_clamps_forward_and_preserves_configured_gradients():
    raw = torch.tensor([-2.0, 0.5, 4.0], requires_grad=True)
    value = M.ste_clamp(raw, lo=0.0, hi=2.0, alpha_lo=0.25, alpha_hi=0.5)
    torch.testing.assert_close(value, torch.tensor([0.0, 0.5, 2.0]))
    value.sum().backward()
    torch.testing.assert_close(raw.grad, torch.tensor([0.25, 1.0, 0.5]))


def test_capped_inclusive_positive_parameter_honors_lower_surrogate_slope():
    parameter = M.PositiveParam(
        -1.0, include_zero=True, max_val=10.0, lower_alpha=0.25, requires_grad=True
    )

    value = parameter()
    torch.testing.assert_close(value, torch.tensor(0.0))
    value.backward()
    torch.testing.assert_close(parameter.rho.grad, torch.tensor(0.25))


def test_eval_cache_is_invalidated_by_set_and_mode_changes():
    parameter = M.PositiveParam(1.0, requires_grad=True).eval()
    first = parameter()
    assert parameter() is first

    parameter.set(2.0)
    second = parameter()
    assert second is not first
    torch.testing.assert_close(second, torch.tensor(2.0))

    assert parameter.train() is parameter
    assert parameter._cache is None
    assert parameter.eval() is parameter
    assert parameter._cache is None


def test_eval_cache_is_invalidated_when_state_dict_is_loaded():
    parameter = M.PositiveParam(1.0, requires_grad=True).eval()
    cached = parameter()
    replacement = M.PositiveParam(4.0, requires_grad=True)

    parameter.load_state_dict(replacement.state_dict())

    assert parameter._cache is None
    restored = parameter()
    assert restored is not cached
    torch.testing.assert_close(restored, torch.tensor(4.0))


def test_legacy_bounded_state_dict_coordinates_are_migrated():
    hard_capped = M.Bounded(1.0, min_val=0.0, max_val=10.0, cap_mode="hard-ste")
    legacy_hard_state = hard_capped.state_dict()
    legacy_hard_state["rho"] = torch.tensor(6.0)
    legacy_hard_state._metadata[""]["version"] = 1
    hard_capped.load_state_dict(legacy_hard_state)
    torch.testing.assert_close(hard_capped(), torch.tensor(6.0))

    inclusive = M.PositiveParam(1.0, include_zero=True, max_val=10.0)
    legacy_inclusive_state = inclusive.state_dict()
    legacy_inclusive_state["rho"] = torch.tensor(0.0)
    legacy_inclusive_state._metadata[""]["version"] = 1
    inclusive.load_state_dict(legacy_inclusive_state)
    torch.testing.assert_close(inclusive(), torch.tensor(5.0))

    negative = M.NegativeParam(-1.0, include_zero=True, min_val=-10.0)
    legacy_negative_state = negative.state_dict()
    legacy_negative_state["rho"] = torch.tensor(0.0)
    legacy_negative_state._metadata[""]["version"] = 1
    negative.load_state_dict(legacy_negative_state)
    torch.testing.assert_close(negative(), torch.tensor(-5.0))

    assert inclusive.state_dict()._metadata[""]["version"] == 2


def test_eval_cache_is_invalidated_by_dtype_conversion():
    parameter = M.PositiveParam(2.0, requires_grad=True).eval()
    cached = parameter()

    parameter.to(dtype=torch.float64)

    assert parameter._cache is None
    converted = parameter()
    assert converted is not cached
    assert converted.dtype == torch.float64
    torch.testing.assert_close(converted, torch.tensor(2.0, dtype=torch.float64))


def test_eval_cache_is_invalidated_when_trainability_changes():
    parameter = M.PositiveParam(2.0, requires_grad=False).eval()
    cached = parameter()

    assert parameter.requires_grad_(True) is parameter

    assert parameter._cache is None
    refreshed = parameter()
    assert refreshed is not cached
    refreshed.backward()
    assert torch.isfinite(parameter.rho.grad)


def test_eval_cache_is_excluded_from_deepcopy_and_pickle_state():
    parameter = M.PositiveParam(2.0, requires_grad=True).eval()
    parameter()

    cloned = copy.deepcopy(parameter)
    restored = pickle.loads(pickle.dumps(parameter))

    for replica in (cloned, restored):
        assert replica._cache is None
        torch.testing.assert_close(replica(), torch.tensor(2.0))


@pytest.mark.parametrize("method", ["batch", "repeat_and_reinit"])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_shape_mutations_preserve_parameter_trainability(method, requires_grad):
    parameter = M.PositiveParam(2.0, requires_grad=requires_grad)

    assert getattr(parameter, method)(3) is parameter

    assert parameter.rho.requires_grad is requires_grad
    expected_shape = (3, 1) if method == "batch" else (3,)
    torch.testing.assert_close(parameter(), torch.full(expected_shape, 2.0))


@pytest.mark.parametrize(
    "initial",
    [
        torch.tensor([1.0, 2.0]),
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    ],
)
def test_repeat_and_reinit_adds_a_leading_axis_for_nonscalar_parameters(initial):
    parameter = M.Bounded(initial, requires_grad=True)

    parameter.repeat_and_reinit(3)

    assert parameter().shape == (3,) + tuple(initial.shape)
    torch.testing.assert_close(
        parameter(), initial.unsqueeze(0).expand((3,) + tuple(initial.shape))
    )


@pytest.mark.parametrize("method", ["batch", "repeat_and_reinit"])
@pytest.mark.parametrize("n", [-1, 1.5, True])
def test_shape_mutations_reject_invalid_repeat_counts(method, n):
    parameter = M.PositiveParam(2.0)
    with pytest.raises((TypeError, ValueError), match="non-negative integer"):
        getattr(parameter, method)(n)


def test_negative_parameter_accepts_scalars_and_preserves_external_sign_on_set_batch():
    parameter = M.NegativeParam(-2.0, requires_grad=True)
    torch.testing.assert_close(parameter(), torch.tensor(-2.0))
    parameter.eval()
    parameter.set(-3.0)
    torch.testing.assert_close(parameter(), torch.tensor(-3.0))

    assert parameter.batch(2) is parameter
    torch.testing.assert_close(parameter(), torch.full((2, 1), -3.0))
    with pytest.raises(ValueError, match="min_val must be negative"):
        M.NegativeParam(-1.0, min_val=1.0)


def test_simple_parameter_names_and_in_place_updates_cover_constraints_and_flags():
    M.SimpleParameterized.PARAMETER(alpha=1.0)
    M.SimpleParameterized.PARAMETERP(rate=2.0)
    M.SimpleParameterized.FLAG(enabled=True)

    class Model(M.SimpleParameterized):
        pass

    model = Model()
    assert model.all_parameter_names() == ["alpha", "rate", "enabled"]
    assert model.check_kwargs({"alpha": 3.0, "rate": 4.0, "enabled": False})

    model.parameter_set_(alpha=3.0, rate=4.0)
    torch.testing.assert_close(model.alpha, torch.tensor(3.0))
    torch.testing.assert_close(model.rate(), torch.tensor(4.0))

    model.eval()
    cached = model.rate()
    model.parameter_set_(**{"rate.rho": torch.tensor(10.0)})
    assert model.rate._cache is None
    assert model.rate() is not cached
    torch.testing.assert_close(model.rate(), F.softplus(torch.tensor(10.0)))

    with pytest.raises(ValueError, match="Unknown parameter"):
        model.parameter_set_(missing=5.0)


def test_pattern_parameter_set_preflights_all_matches_before_mutation():
    M.SimpleParameterized.PARAMETERP(vec=[1.0, 2.0], scalar=1.0)

    class Model(M.SimpleParameterized):
        pass

    model = Model()
    model.vec.requires_grad_(True)
    model.scalar.requires_grad_(True)
    model.eval()
    cached = (model.vec(), model.scalar())
    before = {name: value.detach().clone() for name, value in model.named_parameters()}

    with pytest.raises(ValueError, match="cannot broadcast"):
        model.parameter_set_(rho=torch.tensor([3.0, 4.0]))

    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name])
    (cached[0].sum() + cached[1]).backward()
    assert torch.isfinite(model.vec.rho.grad).all()
    assert torch.isfinite(model.scalar.rho.grad).all()


def _parameterized_model():
    M.Parameterized.GLOBAL(global_value=1.0)
    M.Parameterized.RANGE(range_value=2.0)
    M.Parameterized.BATCH(batch_value=3.0)

    class Model(M.Parameterized):
        pass

    return Model(shape=(2, 3), shape_f=(2, 3))


def test_parameter_snapshots_are_independent_and_strict_load_checks_both_directions():
    model = _parameterized_model()
    snapshot = model.parameters_dict()
    expected = {name: value.clone() for name, value in snapshot.items()}

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)
    assert all(torch.equal(snapshot[name], expected[name]) for name in expected)
    model.load_parameters_dict(snapshot)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, expected[name])

    missing = dict(snapshot)
    missing.pop(next(iter(missing)))
    with pytest.raises(KeyError, match="not found"):
        model.load_parameters_dict(missing, strict=True)
    unexpected = dict(snapshot, not_a_parameter=torch.tensor(1.0))
    with pytest.raises(KeyError, match="Unexpected parameter"):
        model.load_parameters_dict(unexpected, strict=True)
    model.load_parameters_dict(unexpected, strict=False)


def test_parameter_snapshot_load_is_atomic_on_key_errors_and_invalidates_caches():
    M.Parameterized.GLOBALP(gain=1.0, scale=2.0)

    class Model(M.Parameterized):
        pass

    target = Model(shape=(1, 1), shape_f=(1, 1)).eval()
    source = Model(shape=(1, 1), shape_f=(1, 1), gain=4.0)
    cached = target.gain_param()

    target.load_parameters_dict(source.parameters_dict())

    assert target.gain_param._cache is None
    restored = target.gain_param()
    assert restored is not cached
    torch.testing.assert_close(restored, torch.tensor(4.0))

    before = target.parameters_dict()
    incomplete = {name: value + 1.0 for name, value in before.items()}
    incomplete.pop(next(reversed(incomplete)))
    with pytest.raises(KeyError, match="not found"):
        target.load_parameters_dict(incomplete, strict=True)
    for name, parameter in target.named_parameters():
        torch.testing.assert_close(parameter, before[name])

    wrong_shape = {name: value + 2.0 for name, value in before.items()}
    wrong_shape[next(reversed(wrong_shape))] = torch.ones(2)
    with pytest.raises(ValueError, match="shape"):
        target.load_parameters_dict(wrong_shape, strict=True)
    for name, parameter in target.named_parameters():
        torch.testing.assert_close(parameter, before[name])

    live = dict(target.named_parameters())
    first_name, second_name = tuple(live)
    first_before = live[first_name].detach().clone()
    second_before = live[second_name].detach().clone()
    target.load_parameters_dict(
        {first_name: live[second_name], second_name: live[first_name]}, strict=True
    )
    torch.testing.assert_close(live[first_name], second_before)
    torch.testing.assert_close(live[second_name], first_before)


def test_failed_parameter_snapshot_load_rolls_back_and_invalidates_eval_caches():
    M.Parameterized.GLOBALP(first=1.0, second=2.0)

    class Model(M.Parameterized):
        pass

    model = Model(shape=(1, 1), shape_f=(1, 1)).eval()
    model.first_param.rho.requires_grad_(True)
    model.second_param.rho.requires_grad_(True)
    model.first_param()
    model.second_param()
    before = model.parameters_dict()
    updates = {name: value + 1.0 for name, value in before.items()}
    updates[next(reversed(updates))] = torch.empty((), device="meta")

    with pytest.raises((RuntimeError, NotImplementedError)):
        model.load_parameters_dict(updates)

    assert model.first_param._cache is None
    assert model.second_param._cache is None
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name])
    (model.first_param() + model.second_param()).backward()
    assert torch.isfinite(model.first_param.rho.grad)
    assert torch.isfinite(model.second_param.rho.grad)


def test_trainable_only_parameter_snapshot_filters_frozen_parameters():
    model = _parameterized_model()
    parameters = dict(model.named_parameters())
    first_name = next(iter(parameters))
    parameters[first_name].requires_grad_(True)
    snapshot = model.parameters_dict(trainable_only=True)
    assert set(snapshot) == {first_name}


def test_batch_key_collapse_is_ordered_unique_and_empty_safe():
    model = _parameterized_model()
    key = torch.tensor([5, 0, 4, 3, 5, 1])
    assert model._collapse_batch_key(key).tolist() == [1, 0]
    assert model._collapse_batch_key(torch.empty(0, dtype=torch.long)).numel() == 0


def test_batch_and_range_parameter_overrides_have_known_scatter_semantics():
    model = _parameterized_model()
    model.parametrize("range_value", 7.0, key=torch.tensor([0, 5]), alias="ends")
    model.parametrize("batch_value", torch.tensor([8.0, 9.0]), key=torch.tensor([0, 5]))
    model.populate_parameter_buffers()

    assert model.range_value.tolist() == [[7.0, 2.0, 2.0], [2.0, 2.0, 7.0]]
    assert model.batch_value.tolist() == [[8.0], [9.0]]
    with pytest.raises(ValueError, match="already exists"):
        model.parametrize("range_value", 1.0, key=torch.tensor([1]), alias="ends")
    with pytest.raises(KeyError, match="Unknown or non-indexable"):
        model.parametrize("missing", 1.0)
    with pytest.raises(KeyError, match="Unknown or non-indexable"):
        model.parametrize("global_value", 1.0)


def test_constructor_additional_parameters_remain_functional_tensor_slots():
    M.Parameterized.RANGE(rate=1.0)

    class Model(M.Parameterized):
        def forward(self):
            return self._derive_parameter_buffers()["rate"]

    model = Model(
        shape=(2, 3),
        shape_f=(2, 3),
        dtype=torch.float64,
        additional_parameters={
            "rate": [
                ("left", 2.0, torch.tensor([0, 3])),
                (
                    "rows",
                    torch.tensor([[3.0], [5.0]], dtype=torch.float64),
                    torch.tensor([1, 2, 4, 5]),
                ),
            ]
        },
    )

    assert tuple(
        parameter_name for _fill, parameter_name in model.additional_parameters["rate"]
    ) == ("rate_left", "rate_rows")
    assert {"rate_left", "rate_rows"} <= set(dict(model.named_parameters()))

    left = torch.tensor(7.0, dtype=torch.float64, requires_grad=True)
    rows = torch.tensor([[11.0], [13.0]], dtype=torch.float64, requires_grad=True)
    actual = torch.func.functional_call(
        model,
        {"rate_left": left, "rate_rows": rows},
        (),
        tie_weights=False,
    )

    torch.testing.assert_close(
        actual,
        torch.tensor([[7.0, 11.0, 11.0], [7.0, 13.0, 13.0]], dtype=torch.float64),
    )
    left_gradient, rows_gradient = torch.autograd.grad(actual.sum(), (left, rows))
    torch.testing.assert_close(left_gradient, torch.tensor(2.0, dtype=torch.float64))
    torch.testing.assert_close(
        rows_gradient,
        torch.tensor([[2.0], [2.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(model.rate_left, torch.tensor(2.0, dtype=torch.float64))
    torch.testing.assert_close(
        model.rate_rows,
        torch.tensor([[3.0], [5.0]], dtype=torch.float64),
    )


def test_dynamic_additional_parameter_remains_a_functional_tensor_slot():
    M.Parameterized.RANGE(rate=1.0)

    class Model(M.Parameterized):
        def forward(self):
            return self._derive_parameter_buffers()["rate"]

    model = Model(shape=(1, 3), shape_f=(1, 3), dtype=torch.float64)
    model.parametrize("rate", 2.0, key=torch.tensor([0, 2]), alias="ends")

    assert model.additional_parameters["rate"][0][1] == "rate_ends"
    replacement = torch.tensor(7.0, dtype=torch.float64, requires_grad=True)
    actual = torch.func.functional_call(
        model,
        {"rate_ends": replacement},
        (),
        tie_weights=False,
    )

    torch.testing.assert_close(
        actual,
        torch.tensor([[7.0, 1.0, 7.0]], dtype=torch.float64),
    )
    (gradient,) = torch.autograd.grad(actual.sum(), (replacement,))
    torch.testing.assert_close(gradient, torch.tensor(2.0, dtype=torch.float64))


def test_additional_parameter_slots_survive_copy_conversion_and_state_dict():
    M.Parameterized.RANGE(rate=1.0)

    class Model(M.Parameterized):
        pass

    def make(left, rows):
        return Model(
            shape=(2, 3),
            shape_f=(2, 3),
            dtype=torch.float32,
            additional_parameters={
                "rate": [
                    ("left", left, torch.tensor([0, 3])),
                    ("rows", rows, torch.tensor([1, 2, 4, 5])),
                ]
            },
        )

    source = make(2.0, torch.tensor([[3.0], [5.0]]))
    source.populate_parameter_buffers()
    clone = copy.deepcopy(source).double()

    assert tuple(
        source_name for _fill, source_name in clone.additional_parameters["rate"]
    ) == ("rate_left", "rate_rows")
    assert clone.rate_left.dtype == torch.float64
    assert clone.rate_rows.dtype == torch.float64
    with torch.no_grad():
        clone.rate_left.fill_(7.0)
        clone.rate_rows.copy_(torch.tensor([[11.0], [13.0]], dtype=torch.float64))
    torch.testing.assert_close(
        clone._derive_parameter_buffers()["rate"],
        torch.tensor([[7.0, 11.0, 11.0], [7.0, 13.0, 13.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        source._derive_parameter_buffers()["rate"],
        torch.tensor([[2.0, 3.0, 3.0], [2.0, 5.0, 5.0]]),
    )

    meta = copy.deepcopy(source).to(device="meta")
    meta_rate = meta._derive_parameter_buffers()["rate"]
    assert meta_rate.device.type == "meta"
    assert meta_rate.shape == source.rate.shape

    target = make(17.0, torch.tensor([[19.0], [23.0]]))
    result = target.load_state_dict(source.state_dict(), strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    torch.testing.assert_close(
        target._derive_parameter_buffers()["rate"],
        source._derive_parameter_buffers()["rate"],
    )


def test_constrained_parametrize_wraps_raw_parameters_and_preserves_trainability():
    M.Parameterized.RANGEP(rate=2.0)

    class Model(M.Parameterized):
        pass

    model = Model(shape=(1, 2), shape_f=(1, 2))
    raw = torch.nn.Parameter(torch.tensor(3.0), requires_grad=True)
    model.parametrize("rate", raw, key=torch.tensor([0]))
    override = model.rate_param_0

    assert isinstance(override, M.PositiveParam)
    assert override.rho.requires_grad
    model.populate_parameter_buffers()
    torch.testing.assert_close(model.rate, torch.tensor([[3.0, 2.0]]))


def test_dynamic_parametrize_uses_live_target_dtype_after_module_conversion():
    M.Parameterized.RANGE(rate=2.0)

    class Model(M.Parameterized):
        pass

    with ctx(DTYPE="float32"):
        model = Model(shape=(1, 2), shape_f=(1, 2), dtype=torch.float64)
    assert model.rate.dtype == model.rate_param.dtype == torch.float64

    model.parametrize("rate", 3.0, key=torch.tensor([0]))
    model.populate_parameter_buffers()

    assert model.rate_param_0.dtype == torch.float64
    torch.testing.assert_close(
        model.rate, torch.tensor([[3.0, 2.0]], dtype=torch.float64)
    )


def test_constructor_module_only_additional_parameters_do_not_require_scatter_keys():
    M.Parameterized.RANGE(rate=2.0)

    class Constant(torch.nn.Module):
        def forward(self, buffer):
            return buffer.new_tensor(3.0)

    class Model(M.Parameterized):
        pass

    model = Model(
        shape=(1, 3),
        shape_f=(1, 3),
        additional_parameters={"rate": [("module", Constant(), torch.tensor([0, 1]))]},
    )
    model.populate_parameter_buffers()

    torch.testing.assert_close(model.rate, torch.tensor([[3.0, 3.0, 2.0]]))


def test_module_override_keeps_public_parameter_name_and_strict_round_trip():
    M.Parameterized.RANGE(rate=2.0)

    class Affine(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.theta = torch.nn.Parameter(torch.tensor(value))

        def forward(self, _buffer):
            return self.theta

    class Model(M.Parameterized):
        pass

    def make(value):
        model = Model(shape=(1, 2), shape_f=(1, 2))
        model.parametrize(
            "rate",
            Affine(value),
            key=torch.tensor([0]),
            alias="left",
        )
        return model

    source = make(3.0)
    parameter_names = tuple(dict(source.named_parameters()))
    assert "rate_left.theta" in parameter_names
    assert not any(
        name.startswith("_in_graph_parametrization") for name in parameter_names
    )

    aliases = dict(source.named_parameters(remove_duplicate=False))
    private_name = "_in_graph_parametrization_rate_0.func.theta"
    assert aliases["rate_left.theta"] is aliases[private_name]

    target = make(9.0)
    result = target.load_state_dict(source.state_dict(), strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    target.populate_parameter_buffers()
    torch.testing.assert_close(target.rate, torch.tensor([[3.0, 2.0]]))


def test_module_override_strict_load_accepts_pre_registration_checkpoint():
    M.Parameterized.RANGE(rate=2.0)

    class Affine(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.theta = torch.nn.Parameter(torch.tensor(value))

        def forward(self, _buffer):
            return self.theta

    class Model(M.Parameterized):
        pass

    def make(value):
        model = Model(shape=(1, 2), shape_f=(1, 2))
        model.parametrize("rate", Affine(value), key=torch.tensor([0]), alias="left")
        return model

    source = make(3.0)
    old_layout = {
        name: value
        for name, value in source.state_dict().items()
        if not name.startswith("_in_graph_parametrization_rate_0.")
    }
    target = make(9.0)

    result = target.load_state_dict(old_layout, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    target.populate_parameter_buffers()
    torch.testing.assert_close(target.rate, torch.tensor([[3.0, 2.0]]))


def test_module_override_consumes_and_validates_legacy_persistent_structural_key():
    M.Parameterized.RANGE(rate=2.0)

    class Affine(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.theta = torch.nn.Parameter(torch.tensor(value))

        def forward(self, _buffer):
            return self.theta

    class Model(M.Parameterized):
        pass

    def make():
        model = Model(shape=(1, 2), shape_f=(1, 2))
        model.parametrize("rate", Affine(3.0), key=torch.tensor([0]), alias="left")
        return model

    source = make()
    key_name = "_in_graph_parametrization_rate_0.key"
    legacy = dict(source.state_dict())
    legacy[key_name] = getattr(source, "_in_graph_parametrization_rate_0").key.clone()

    result = make().load_state_dict(legacy, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []

    legacy[key_name] = torch.tensor([1])
    with pytest.raises(RuntimeError, match="Structural parametrization key mismatch"):
        make().load_state_dict(legacy, strict=True)


def test_parameter_materializer_adopts_effective_buffer_dtype_from_module_source():
    M.Parameterized.RANGE(rate=2.0)

    class Source(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = torch.nn.Parameter(torch.tensor(3.0, dtype=torch.float32))

        def forward(self):
            return self.theta

    class Model(M.Parameterized):
        pass

    model = Model(shape=(2, 3), shape_f=(2, 3), dtype=torch.float64, rate=Source())
    model.populate_parameter_buffers()

    assert model.rate.dtype == torch.float64
    torch.testing.assert_close(model.rate, torch.full((2, 3), 3.0, dtype=torch.float64))
    model.rate.sum().backward()
    torch.testing.assert_close(model.rate_param.theta.grad, torch.tensor(6.0))


def test_in_place_transform_receives_writable_copy_without_mutating_raw_source():
    M.Parameterized.RANGE(rate=2.0)

    class Twice(torch.nn.Module):
        def forward(self, value):
            return value.mul_(2.0)

    class Model(M.Parameterized):
        pass

    model = Model(shape=(2, 3), shape_f=(2, 3))
    model.register_parametrization_in_graph("rate", Twice())
    raw_before = model.rate_param.detach().clone()

    derived = model._derive_parameter_buffers()
    torch.testing.assert_close(derived["rate"], torch.full((2, 3), 4.0))
    torch.testing.assert_close(model.rate_param, raw_before)

    model.populate_parameter_buffers()
    torch.testing.assert_close(model.rate, torch.full((2, 3), 4.0))
    torch.testing.assert_close(model.rate_param, raw_before)


def test_dynamic_parametrize_moves_stale_keys_before_committing_new_override():
    M.Parameterized.RANGE(rate=2.0)

    class Model(M.Parameterized):
        pass

    model = Model(shape=(1, 3), shape_f=(1, 3))
    model.parametrize("rate", 3.0, key=torch.tensor([0]), alias="first")
    model.to(device="meta")

    model.parametrize("rate", 4.0, key=torch.tensor([1]), alias="second")

    assert model.keys["rate"].device.type == "meta"
    assert hasattr(model, "rate_second")
    assert len(model.additional_parameters["rate"]) == 2


def test_failed_parametrize_does_not_leave_an_orphan_alias():
    M.Parameterized.RANGE(rate=2.0)

    class Model(M.Parameterized):
        pass

    model = Model(shape=(2, 3), shape_f=(2, 3))
    with pytest.raises(ValueError):
        model.parametrize(
            "rate", torch.ones(2, 2), key=torch.tensor([0, 5]), alias="bad"
        )
    assert not hasattr(model, "rate_bad")

    model.parametrize("rate", 3.0, key=torch.tensor([0, 5]), alias="bad")
    model.populate_parameter_buffers()
    assert model.rate[0, 0].item() == pytest.approx(3.0)
    assert model.rate[1, 2].item() == pytest.approx(3.0)


def test_make_contiguous_repairs_expanded_range_and_batch_buffers():
    model = _parameterized_model()
    model.range_value = torch.ones(1, 3).expand(2, 3)
    model.batch_value = torch.ones(1, 1).expand(2, 1)
    assert not model.range_value.is_contiguous()
    assert not model.batch_value.is_contiguous()
    model.make_contiguous()
    assert model.range_value.is_contiguous()
    assert model.batch_value.is_contiguous()
