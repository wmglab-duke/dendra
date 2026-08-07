import types

import pytest
import torch

from dendra.models.instruments.thresholder import (
    Thresholder,
    _agree_dims,
    _scale_by_partition,
    op_mc,
    op_sc,
)


class FakeWaveform:
    def __init__(self, dtype=torch.float64):
        self.dtype = dtype

    def to(self, *, device, dtype):
        self.dtype = dtype
        return self

    def float(self):
        self.dtype = torch.float32
        return self

    def double(self):
        self.dtype = torch.float64
        return self


class FakeActive:
    threshold = 0.0

    def __init__(self, thresholds):
        self.thresholds = torch.as_tensor(thresholds, dtype=torch.float64)
        self.last_field = None
        self.last_field_time_first = False
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def is_active(self, partition=None):
        field = self.last_field
        if self.last_field_time_first:
            amp_by_comp = field.abs().amax(dim=0)
        else:
            amp_by_comp = field.abs()

        if partition is None:
            lane_shape = amp_by_comp.shape[:-1]
            thresholds = torch.broadcast_to(self.thresholds, lane_shape)
            return amp_by_comp.amax(dim=-1) >= thresholds.to(amp_by_comp)

        cols = []
        start = 0
        for length in partition:
            stop = start + length
            cols.append(amp_by_comp[..., start:stop].amax(dim=-1))
            start = stop
        amplitudes = torch.stack(cols, dim=-1)
        thresholds = torch.broadcast_to(self.thresholds, amplitudes.shape)
        return amplitudes >= thresholds.to(amplitudes)


class FakeModel:
    def __init__(self, np=2, nc=4, dtype=torch.float64, diameters=None, batch_shape=()):
        self.np = np
        self.nc = nc
        self.shape = (*batch_shape, np, nc)
        self._dtype = dtype
        self.diameters = diameters
        self.initialize_calls = 0
        self.run_calls = 0
        self.longrun_calls = 0

    def device(self):
        return torch.device("cpu")

    def dtype(self):
        return self._dtype

    def float(self):
        self._dtype = torch.float32
        return self

    def double(self):
        self._dtype = torch.float64
        return self

    def initialize(self):
        self.initialize_calls += 1
        return self

    @staticmethod
    def _field_from_args(args, ve, extra):
        if ve is not None:
            return ve
        if args:
            return args[0]
        if extra is not None:
            return extra[0]
        raise AssertionError("A field must be supplied to the fake model.")

    def run(self, *args, ve=None, extra=None, callbacks=None, **kwargs):
        self.run_calls += 1
        field = self._field_from_args(args, ve, extra)
        for callback in callbacks or []:
            if isinstance(callback, FakeActive):
                callback.last_field = field
                callback.last_field_time_first = extra is None

    def longrun(self, *args, extra=None, callbacks=None, **kwargs):
        self.longrun_calls += 1
        field = self._field_from_args(args, None, extra)
        for callback in callbacks or []:
            if isinstance(callback, FakeActive):
                callback.last_field = field
                callback.last_field_time_first = False


def _basis_thresholder(
    *, thresholds=(2.0, 4.0), ub=8.0, batch_shape=(), np=2, nc=4, **kwargs
):
    model = FakeModel(np=np, nc=nc, batch_shape=batch_shape)
    active = FakeActive(thresholds)
    bases = torch.ones((3, *model.shape), dtype=torch.float64)
    thresholder = Thresholder(
        model,
        active,
        bases=bases,
        ub=ub,
        atol=0.01,
        **kwargs,
    )
    return thresholder, model, active


def _functional_thresholder(
    *, thresholds=(2.0, 4.0), ub=8.0, batch_shape=(), np=2, nc=4, **kwargs
):
    model = FakeModel(np=np, nc=nc, batch_shape=batch_shape)
    active = FakeActive(thresholds)
    space = torch.ones(model.shape, dtype=torch.float64)
    thresholder = Thresholder(
        model,
        active,
        space=space,
        time=FakeWaveform(),
        ub=ub,
        atol=0.01,
        **kwargs,
    )
    return thresholder, model, active


def test_thresholder_requires_a_stimulus_and_valid_bounds_configuration():
    model = FakeModel()
    active = FakeActive([1.0, 1.0])

    with pytest.raises(ValueError, match="At least one"):
        Thresholder(model, active, ub=1.0)
    with pytest.raises(ValueError, match="At least one"):
        Thresholder(model, active, space=torch.ones(2, 4), ub=1.0)
    with pytest.raises(ValueError, match="At least one"):
        Thresholder(model, active, time=FakeWaveform(), ub=1.0)
    with pytest.raises(ValueError, match="chunklength"):
        Thresholder(model, active, bases=torch.ones(1, 2, 4), ub=1.0, chunklength=2)
    with pytest.raises(ValueError, match="fix_bound_down"):
        Thresholder(
            model,
            active,
            bases=torch.ones(1, 2, 4),
            ub=1.0,
            fix_bound_down=1.0,
        )
    with pytest.raises(ValueError, match="fix_bound_down"):
        Thresholder(
            model,
            active,
            bases=torch.ones(1, 2, 4),
            ub=1.0,
            fix_bound_down=0.0,
        )
    with pytest.raises(ValueError, match="fix_bound_up"):
        Thresholder(
            model,
            active,
            bases=torch.ones(1, 2, 4),
            ub=1.0,
            fix_bound_up=1.0,
        )


def test_thresholder_uses_diameters_for_default_upper_bounds():
    model = FakeModel(diameters=torch.tensor([5.0, 10.0]))
    active = FakeActive([1.0, 1.0])
    th = Thresholder(model, active, bases=torch.ones(3, 2, 4), atol=0.1)
    assert torch.allclose(th.ub, torch.tensor([0.2, 0.05], dtype=th.ub.dtype))

    model_no_diam = FakeModel()
    with pytest.raises(ValueError, match="Either ub"):
        Thresholder(model_no_diam, active, bases=torch.ones(3, 2, 4), atol=0.1)


def test_check_tolerance_supports_absolute_relative_and_combined_rules():
    th, _, _ = _basis_thresholder()
    absolute = torch.tensor([0.2, 0.05])
    relative = torch.tensor([0.2, 0.02])

    assert torch.equal(
        th.check_tolerance(absolute, relative, atol=0.1, rtol=None),
        torch.tensor([True, False]),
    )
    th.atol = None
    assert torch.equal(
        th.check_tolerance(absolute, relative, atol=None, rtol=0.1),
        torch.tensor([True, False]),
    )
    th.atol = 0.1
    assert torch.equal(
        th.check_tolerance(
            torch.tensor([0.2, 0.05]),
            torch.tensor([0.02, 0.2]),
            atol=0.1,
            rtol=0.1,
        ),
        torch.tensor([True, True]),
    )
    th.atol = None
    with pytest.raises(ValueError, match="Either atol or rtol"):
        th.check_tolerance(absolute, relative, atol=None, rtol=None)


def test_relative_window_uses_the_documented_upper_bound_denominator():
    th, _, _ = _basis_thresholder(ub=torch.tensor([4.0, 8.0]), lb=[2.0, 4.0])
    awindow = th.ub - th.lb
    assert torch.equal(th._relative_window(awindow), torch.tensor([0.5, 0.5]))


def test_basis_threshold_search_finds_independent_thresholds():
    th, model, active = _basis_thresholder()
    upper, lower = th.calculate_thresholds(tstop=1.0, dt=0.1)

    assert torch.all(upper >= torch.tensor([2.0, 4.0]))
    assert torch.all(lower < torch.tensor([2.0, 4.0]))
    assert torch.all((upper - lower) < 0.01)
    assert model.run_calls > 1
    assert active.reset_calls > 1


def test_functional_threshold_search_supports_run_and_longrun():
    th, model, _ = _functional_thresholder()
    upper, lower = th.calculate_thresholds(tstop=1.0, dt=0.1)
    assert torch.all((upper - lower) < 0.01)
    assert model.run_calls > 0

    chunked, chunked_model, _ = _functional_thresholder(chunklength=4)
    chunked.calculate_thresholds(tstop=1.0, dt=0.1)
    assert chunked_model.longrun_calls > 0
    assert chunked_model.run_calls == 0


@pytest.mark.parametrize(
    ("batch_shape", "thresholds"),
    [
        ((3,), torch.tensor([[2.0], [3.0], [4.0]])),
        ((2, 3), torch.tensor([[[1.0], [2.0], [3.0]], [[2.0], [3.0], [4.0]]])),
    ],
)
def test_functional_threshold_search_preserves_all_batched_lanes(
    batch_shape, thresholds
):
    th, model, _ = _functional_thresholder(
        batch_shape=batch_shape, np=1, thresholds=thresholds
    )
    upper, lower = th.calculate_thresholds(tstop=1.0, dt=0.1)

    assert upper.shape == model.shape[:-1]
    assert lower.shape == model.shape[:-1]
    assert torch.all(upper >= thresholds)
    assert torch.all(lower < thresholds)
    assert torch.all((upper - lower) < 0.01)


@pytest.mark.parametrize("batch_shape", [(3,), (2, 3)])
def test_basis_threshold_search_preserves_all_batched_lanes(batch_shape):
    lane_shape = (*batch_shape, 1)
    thresholds = torch.arange(
        1, torch.tensor(lane_shape).prod().item() + 1, dtype=torch.float64
    ).reshape(lane_shape)
    th, model, _ = _basis_thresholder(
        batch_shape=batch_shape, np=1, thresholds=thresholds, ub=8.0
    )
    upper, lower = th.calculate_thresholds(tstop=999.0, dt=0.1)

    assert th.bases.shape == (3, *model.shape)
    assert upper.shape == model.shape[:-1]
    assert torch.all(upper >= thresholds)
    assert torch.all(lower < thresholds)


@pytest.mark.parametrize(
    "factory",
    [_functional_thresholder, _basis_thresholder],
    ids=["functional", "basis"],
)
def test_threshold_search_preserves_repeated_batches_and_population_axis(factory):
    thresholds = torch.arange(1.0, 13.0, dtype=torch.float64).reshape(2, 3, 2)
    th, model, _ = factory(batch_shape=(2, 3), np=2, thresholds=thresholds, ub=16.0)

    upper, lower = th.calculate_thresholds(tstop=1.0, dt=0.1)

    assert upper.shape == model.shape[:-1] == (2, 3, 2)
    assert lower.shape == model.shape[:-1]
    assert torch.all(upper >= thresholds)
    assert torch.all(lower < thresholds)
    assert torch.all((upper - lower) < 0.01)


def test_fix_bounds_expands_small_upper_bounds_and_marks_failures():
    th, _, _ = _basis_thresholder(ub=0.5, max_tries_bound_fix=8)
    th._fix_bounds(tstop=1.0, dt=0.1, block_possible=False)
    assert torch.all(th.ub >= torch.tensor([2.0, 4.0]))
    assert th.ignore is None

    failed, _, _ = _basis_thresholder(
        thresholds=(100.0, 100.0), ub=1.0, max_tries_bound_fix=1
    )
    failed._fix_bounds(tstop=1.0, dt=0.1, block_possible=False)
    assert torch.equal(failed.ignore, torch.tensor([True, True]))
    assert torch.equal(failed.ub, torch.ones(2))
    assert torch.equal(failed.lb, torch.ones(2))


def test_fix_bounds_evaluates_the_final_allowed_adjustment():
    th, _, _ = _basis_thresholder(thresholds=(2.0, 2.0), ub=1.0, max_tries_bound_fix=1)
    th._fix_bounds(tstop=1.0, dt=0.1, block_possible=False)
    assert torch.equal(th.ub, torch.tensor([2.0, 2.0], dtype=th.ub.dtype))
    assert th.ignore is None


def test_fix_bounds_block_mode_moves_bounds_up_or_down():
    th, _, _ = _basis_thresholder(ub=2.0, max_tries_bound_fix=2)
    calls = iter(
        [
            (torch.tensor([False, False]), torch.tensor([-1.0, 1.0])),
            (torch.tensor([True, True]), torch.zeros(2)),
        ]
    )
    th.check_active_with_rec = types.MethodType(
        lambda self, tstop, dt, bound: next(calls), th
    )
    th.threshold = 0.0
    th._fix_bounds(tstop=1.0, dt=0.1, block_possible=True)
    assert torch.allclose(th.ub, th.ub.new_tensor([4.0, 0.2]))


def test_fix_bounds_block_mode_preserves_batched_singleton_population_axis():
    th, _, _ = _basis_thresholder(
        batch_shape=(3,),
        np=1,
        thresholds=torch.ones((3, 1)),
        ub=2.0,
        max_tries_bound_fix=1,
    )
    calls = iter(
        [
            (
                torch.zeros((3, 1), dtype=torch.bool),
                torch.tensor([[-1.0], [1.0], [-1.0]]),
            ),
            (
                torch.ones((3, 1), dtype=torch.bool),
                torch.zeros((3, 1)),
            ),
        ]
    )
    th.check_active_with_rec = types.MethodType(
        lambda self, tstop, dt, bound: next(calls), th
    )
    th.threshold = 0.0
    th._fix_bounds(tstop=1.0, dt=0.1, block_possible=True)
    assert th.ub.shape == (3, 1)
    assert torch.allclose(th.ub[:, 0], th.ub.new_tensor([4.0, 0.2, 4.0]))


def test_recorder_metric_normalization_rejects_axis_reinterpretation():
    th, _, _ = _basis_thresholder(batch_shape=(2,), np=3, thresholds=torch.ones((2, 3)))

    normalized = th._normalize_recorder_metric(torch.zeros((2, 3, 1)))
    assert normalized.shape == (2, 3)
    with pytest.raises(ValueError, match="Recorder block metric shape"):
        th._normalize_recorder_metric(torch.zeros((3, 2)))


def test_calculate_thresholds_rejects_active_lower_bound():
    th, _, _ = _basis_thresholder(thresholds=(-1.0, -1.0))
    with pytest.raises(RuntimeError, match="lower bounds are active"):
        th.calculate_thresholds(tstop=1.0, dt=0.1)


def test_partitioned_threshold_search_and_reset_bounds():
    thresholds = torch.tensor([[2.0, 4.0], [1.0, 3.0]])
    th, _, _ = _basis_thresholder(thresholds=thresholds)
    th.set_partition([2, 2], active_partition=[1, 2])
    assert torch.equal(th.rec.partition, torch.tensor([2, 2]))

    upper, lower = th.calculate_thresholds(tstop=1.0, dt=0.1)
    assert upper.shape == (2, 2)
    assert torch.all(upper >= thresholds)
    assert torch.all(lower < thresholds)

    th.ub.zero_()
    th.lb.fill_(2.0)
    th.ignore = torch.ones_like(th.ub, dtype=torch.bool)
    th.reset_bounds()
    assert torch.equal(th.ub, th.ub_initial)
    assert torch.count_nonzero(th.lb) == 0
    assert th.ignore is None


@pytest.mark.parametrize("batch_shape", [(3,), (2, 2)])
def test_partitioned_batched_threshold_search_uses_a_final_partition_axis(
    batch_shape,
):
    n_lanes = int(torch.tensor(batch_shape).prod().item())
    thresholds = (
        torch.arange(n_lanes * 2, dtype=torch.float64)
        .remainder(5)
        .add(1)
        .reshape(*batch_shape, 1, 2)
    )
    th, model, _ = _basis_thresholder(
        batch_shape=batch_shape, np=1, thresholds=thresholds
    )
    th.set_partition([2, 2], active_partition=[2, 2])

    upper, lower = th.calculate_thresholds(tstop=1.0, dt=0.1)
    assert upper.shape == (*model.shape[:-1], 2)
    assert lower.shape == (*model.shape[:-1], 2)
    assert torch.all(upper >= thresholds)
    assert torch.all(lower < thresholds)


def test_reset_bounds_preserves_nonzero_lower_bounds_partition_and_dtype():
    th, _, _ = _basis_thresholder(lb=torch.tensor([0.5, 1.0]))
    th.set_partition([2, 2])
    expected_lb = torch.tensor([[0.5, 0.5], [1.0, 1.0]], dtype=th.lb.dtype)

    th.lb.zero_()
    th.ub.zero_()
    th.reset_bounds()
    assert torch.equal(th.lb, expected_lb)
    assert torch.equal(th.ub, th.ub_initial)

    th.float()
    assert th.lb_initial.dtype == torch.float32
    th.reset_bounds()
    assert th.lb.dtype == torch.float32
    assert torch.equal(th.lb, expected_lb.float())


def test_geometric_mode_rejects_zero_lower_bounds():
    with pytest.raises(ValueError, match="strictly positive"):
        _basis_thresholder(mode="geometric")

    th, _, _ = _basis_thresholder(mode="geometric", lb=0.5)
    assert torch.all(th.calc_stimamp() > 0)


def test_unconverged_search_lanes_are_returned_as_nan():
    th, _, _ = _basis_thresholder(max_tries_thresh=0)
    upper, lower = th.calculate_thresholds(tstop=1.0, dt=0.1)
    assert torch.isnan(upper).all()
    assert torch.isnan(lower).all()
    assert torch.equal(th.ignore, torch.ones_like(th.ignore))


def test_partition_validation_rejects_inconsistent_layouts():
    th, _, _ = _basis_thresholder()
    with pytest.raises(ValueError, match="Sum of partition"):
        th.set_partition([1, 1])
    with pytest.raises(ValueError, match="same length"):
        th.set_partition([2, 2], [1])
    with pytest.raises(ValueError, match="at least as large"):
        th.set_partition([2, 2], [3, 1])


def test_nan_fields_are_ignored_without_poisoning_simulation_inputs():
    th, _, _ = _functional_thresholder()
    th.space[0, 1] = torch.nan
    th._apply_field_nan_ignore()
    assert torch.equal(th.field_nan_ignore, torch.tensor([True, False]))
    assert not torch.isnan(th._space_for_run()).any()
    assert th.ub[0] == 0

    clean, _, _ = _functional_thresholder()
    clean._apply_field_nan_ignore()
    assert clean.ignore is None
    assert clean._space_no_nan is None


def test_partitioned_nan_masks_only_ignore_affected_regions():
    th, _, _ = _basis_thresholder()
    th.set_partition([2, 2])
    th.bases[0, 0, 3] = torch.nan
    th._apply_field_nan_ignore()

    assert torch.equal(
        th.field_nan_ignore, torch.tensor([[False, True], [False, False]])
    )
    assert not torch.isnan(th._bases_for_run()).any()
    expanded = th._ignored_like(torch.zeros((2, 4), dtype=torch.bool))
    assert torch.equal(
        expanded,
        torch.tensor([[False, False, True, True], [False, False, False, False]]),
    )


def test_basis_nan_reduction_uses_canonical_time_and_final_compartment_axes():
    th, _, _ = _basis_thresholder()
    canonical = torch.zeros((3, 2, 4), dtype=torch.bool)
    canonical[1, 0, 2] = True
    assert torch.equal(th._reduce_bases_nan(canonical), torch.tensor([True, False]))

    with pytest.raises(ValueError, match="canonical time-first"):
        th._reduce_bases_nan(torch.zeros((2, 3, 4), dtype=torch.bool))


def test_basis_layout_normalization_accepts_canonical_broadcast_and_legacy():
    model = FakeModel(np=2, nc=4, batch_shape=(3,))
    active = FakeActive(torch.ones((3, 2)))

    canonical = torch.arange(5 * 2 * 4, dtype=torch.float64).reshape(5, 2, 4)
    th = Thresholder(model, active, bases=canonical, ub=8.0, atol=0.1)
    assert th.bases.shape == (5, 3, 2, 4)
    assert torch.equal(th.bases[:, 0], canonical)

    legacy = torch.arange(2 * 5 * 4, dtype=torch.float64).reshape(2, 5, 4)
    legacy_th = Thresholder(model, active, bases=legacy, ub=8.0, atol=0.1)
    assert legacy_th.bases.shape == (5, 3, 2, 4)
    assert torch.equal(legacy_th.bases[:, 0], legacy.movedim(1, 0))

    unbatched = FakeModel(np=2, nc=4)
    with pytest.raises(ValueError, match="ambiguous"):
        Thresholder(
            unbatched,
            FakeActive([1.0, 1.0]),
            bases=torch.ones((2, 2, 4)),
            ub=8.0,
            atol=0.1,
        )


def test_batched_nan_masks_preserve_lanes_and_partition_final_axis():
    model = FakeModel(np=1, nc=4, batch_shape=(3,))
    space = torch.zeros(model.shape, dtype=torch.float64)
    space[1, 0, 3] = torch.nan
    th = Thresholder(
        model,
        FakeActive(torch.ones((3, 1, 2))),
        space=space,
        time=FakeWaveform(),
        ub=8.0,
        atol=0.1,
    )
    th.set_partition([2, 2])
    th._apply_field_nan_ignore()

    expected = torch.zeros((3, 1, 2), dtype=torch.bool)
    expected[1, 0, 1] = True
    assert torch.equal(th.field_nan_ignore, expected)
    assert th.ignore.shape == (3, 1, 2)
    assert not torch.isnan(th._space_for_run()).any()


def test_space_and_bound_broadcast_contract_is_explicit_for_batches():
    model = FakeModel(np=2, nc=4, batch_shape=(3,))
    active = FakeActive(torch.ones((3, 2)))
    th = Thresholder(
        model,
        active,
        space=torch.ones((2, 4)),
        time=FakeWaveform(),
        ub=torch.tensor([8.0, 6.0]),
        lb=0.5,
        atol=0.1,
    )
    assert th.space.shape == model.shape
    assert th.ub.shape == model.shape[:-1]
    assert torch.equal(th.ub[0], torch.tensor([8.0, 6.0], dtype=th.ub.dtype))

    with pytest.raises(ValueError, match="lane shape"):
        Thresholder(
            model,
            active,
            space=torch.ones((2, 4)),
            time=FakeWaveform(),
            ub=torch.ones(3),
            atol=0.1,
        )


def test_thresholder_rejects_callback_output_that_drops_population_axis():
    th, _, active = _functional_thresholder(batch_shape=(3,), np=1)
    active.is_active = types.MethodType(
        lambda self, partition=None: torch.zeros(3, dtype=torch.bool), active
    )
    with pytest.raises(ValueError, match="must preserve every batch"):
        th.check_active(1.0, 0.1, _scale_by_partition(th.ub, th.model_partition))


@pytest.mark.parametrize(
    "result", [torch.full((3, 1), torch.nan), None], ids=["float-nan", "none"]
)
def test_thresholder_rejects_non_boolean_callback_output(result):
    th, _, active = _functional_thresholder(batch_shape=(3,), np=1)
    active.is_active = types.MethodType(
        lambda self, partition=None: result,
        active,
    )
    with pytest.raises(TypeError, match="must return a boolean tensor"):
        th.check_active(1.0, 0.1, _scale_by_partition(th.ub, th.model_partition))


def test_dtype_conversion_updates_model_fields_and_waveform():
    th, model, _ = _functional_thresholder()
    th._space_no_nan = th.space.clone()
    th.double()
    assert model.dtype() == torch.float64
    assert th.space.dtype == torch.float64
    assert th.time.dtype == torch.float64

    th.float()
    assert model.dtype() == torch.float32
    assert th.space.dtype == torch.float32
    assert th.ub.dtype == torch.float32
    assert th.time.dtype == torch.float32


def test_field_scaling_and_einsum_helpers_cover_supported_layouts():
    th, _, _ = _basis_thresholder()
    one_dimensional = th._scaled_bases_field(torch.tensor([2.0, 3.0]))
    assert one_dimensional.shape == th.bases.shape
    assert torch.all(one_dimensional[:, 0] == 2.0)
    assert torch.all(one_dimensional[:, 1] == 3.0)

    th.set_partition([2, 2])
    bound = _scale_by_partition(torch.tensor([[2.0, 3.0], [4.0, 5.0]]), [2, 2])
    partitioned = th._scaled_bases_field(bound)
    assert torch.equal(partitioned[0, 0], torch.tensor([2.0, 2.0, 3.0, 3.0]))

    s = torch.arange(8.0).reshape(2, 4)
    t = torch.ones((2, 3))
    assert op_sc(s, t).shape == (3, 2, 4)

    smc = torch.arange(16.0).reshape(2, 2, 4)
    tmc = torch.ones((2, 2, 3))
    assert op_mc(smc, tmc).shape == (3, 2, 4)

    functional, model, _ = _functional_thresholder()
    assert functional.ve_from_s_t(s, t, model.device()).shape == (3, 2, 4)
    assert functional.ve_from_s_t(
        smc, tmc, model.device(), multicontact=True
    ).shape == (
        3,
        2,
        4,
    )

    batched, batched_model, _ = _functional_thresholder(
        batch_shape=(2, 3), np=1, thresholds=torch.ones((2, 3, 1))
    )
    sb = torch.ones(batched_model.shape)
    tb = torch.ones((*batched_model.shape[:-1], 5))
    assert batched.ve_from_s_t(sb, tb, batched_model.device()).shape == (
        5,
        *batched_model.shape,
    )

    smcb = torch.ones((2, *batched_model.shape))
    tmcb = torch.ones((2, *batched_model.shape[:-1], 5))
    assert batched.ve_from_s_t(
        smcb, tmcb, batched_model.device(), multicontact=True
    ).shape == (5, *batched_model.shape)


def test_partition_and_dimension_helpers_validate_and_expand():
    B = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.equal(
        _scale_by_partition(B, [1, 2]),
        torch.tensor([[1.0, 2.0, 2.0], [3.0, 4.0, 4.0]]),
    )
    assert _scale_by_partition(torch.tensor([1.0, 2.0])).shape == (2, 1)
    with pytest.raises(ValueError, match="1D"):
        _scale_by_partition(B, [[1, 1]])
    with pytest.raises(ValueError, match="final dimension"):
        _scale_by_partition(B, [1])
    with pytest.raises(ValueError, match="non-empty"):
        _scale_by_partition(torch.empty((2, 0)), [])
    with pytest.raises(ValueError, match="non-negative"):
        _scale_by_partition(B, [1, -1])

    mask = torch.tensor([True, False])
    target = torch.zeros((2, 3))
    assert _agree_dims(mask, target).shape == target.shape
    assert _agree_dims(mask[:, None], torch.zeros(2)).shape == (2,)
    assert _agree_dims(mask, torch.zeros(2)).shape == (2,)

    batched = torch.arange(12.0).reshape(2, 3, 2)
    expanded = _scale_by_partition(batched, [1, 2])
    assert expanded.shape == (2, 3, 3)
    assert torch.equal(expanded[..., 1:], batched[..., 1:].expand(2, 3, 2))
