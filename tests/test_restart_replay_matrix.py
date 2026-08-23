from __future__ import annotations

import copy

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism
from dendra.models.mod import pas

DTYPE = torch.float64
DT = 0.1


class _ReplayPopulation(dn.SingleCompartment):
    dn.SingleCompartment.RANGENOISE(
        "ambient",
        distribution="normal",
        mu=0.0,
        sigma=1.0,
        seed=808,
    )


class _NoisyDrive(Mechanism):
    Mechanism.RANGENOISE(
        "eta",
        distribution="normal",
        mu=0.0,
        sigma=1.0,
        seed=1729,
    )

    def i(self, v):
        # The stochastic stream participates in the numerical trajectory while
        # remaining small enough to keep this replay fixture well conditioned.
        return self.eta * 1.0e-3


class _SuffixTrace(dn.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.samples = []

    def post_step_hook(self, model):
        self.samples.append(
            (
                model.t.detach().clone(),
                model.v.detach().clone(),
                model.mech._NoisyDrive.eta.detach().clone(),
            )
        )


def _model(*, compartments=2):
    model = _ReplayPopulation(
        N=1,
        C=compartments,
        v_init=-65.0,
        dtype=DTYPE,
    )
    model.insert(pas, g=0.001, e=-70.0)
    model.insert(_NoisyDrive)
    model.initialize()
    mechanism = model.mech._NoisyDrive
    mechanism.register_delayed_state(
        "eta_history",
        torch.zeros_like(mechanism.eta),
        delay_steps=2,
        mode="circular",
    )
    return model


def _clone_nested(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return value.__class__(
            (key, _clone_nested(item)) for key, item in value.items()
        )
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_nested(item) for item in value)
    return copy.deepcopy(value)


def _assert_nested_equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
        return
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
        return
    assert actual == expected


def _advance(model, callback=None):
    # Population-level NOISE is an explicitly user-driven stream; mechanism
    # NOISE is sampled automatically by Integrator._call_kernel.
    model.resample_runtime_noise("ambient", dt=DT, phase="pre_state")
    model.step(dt=DT, callbacks=[] if callback is None else [callback])
    mechanism = model.mech._NoisyDrive
    delayed = mechanism.delayed_state("eta_history", mechanism.eta, mode="circular")
    return {
        "t": model.t.detach().clone(),
        "v": model.v.detach().clone(),
        "ambient": model.ambient.detach().clone(),
        "eta": mechanism.eta.detach().clone(),
        "delayed": delayed.detach().clone(),
        "delay_buffer": mechanism.eta_history_delay_buffer.detach().clone(),
        "delay_pointer": mechanism.eta_history_delay_ptr.detach().clone(),
    }


def test_fresh_population_checkpoint_replays_exact_stochastic_delayed_suffix():
    uninterrupted = _model()
    uninterrupted.ambient_rng.reseed(9917)
    for _ in range(4):
        _advance(uninterrupted)
    checkpoint = _clone_nested(uninterrupted.state_dict_for_checkpoint())

    expected_callback = _SuffixTrace()
    expected = [_advance(uninterrupted, expected_callback) for _ in range(6)]

    resumed = _model()
    assert resumed.restore_dict_from_checkpoint(checkpoint) is resumed
    actual_callback = _SuffixTrace()
    actual = [_advance(resumed, actual_callback) for _ in range(6)]

    _assert_nested_equal(actual, expected)
    _assert_nested_equal(actual_callback.samples, expected_callback.samples)

    # Restoring only the generator suffix would make the replay above pass but
    # leave reset_rng() tied to the fresh object's constructor seed.
    uninterrupted.reset_rng()
    resumed.reset_rng()
    uninterrupted.resample_runtime_noise("ambient", dt=DT, phase="pre_state")
    resumed.resample_runtime_noise("ambient", dt=DT, phase="pre_state")
    assert torch.equal(resumed.ambient, uninterrupted.ambient)


def test_checkpointed_longrun_threads_rng_state_and_restores_final_suffix_state():
    class VoltageLoss(dn.callbacks.Callback):
        def post_step_hook(self, model):
            return model.v.square().mean()

    uninterrupted = _model()
    uninterrupted.train()
    uninterrupted.run(tstop=0.4, dt=DT)

    with dn.ctx(REQUIRE_GRAD=1):
        checkpointed = _model()
    checkpointed.train()
    loss, final_state = checkpointed.longrun_checkpointed(
        tstop=0.4,
        chunklength=2,
        dt=DT,
        callbacks=[VoltageLoss()],
        safe_checkpoint=True,
        restore_state_after_backward=True,
        return_final_state=True,
    )

    assert torch.equal(checkpointed.v, uninterrupted.v)
    assert torch.equal(
        checkpointed.mech._NoisyDrive.eta,
        uninterrupted.mech._NoisyDrive.eta,
    )
    before_backward = _clone_nested(checkpointed.state_dict_for_checkpoint())
    assert loss is not None and loss.requires_grad
    loss.backward()

    _assert_nested_equal(checkpointed.state_dict_for_checkpoint(), before_backward)
    _assert_nested_equal(final_state, before_backward)


def test_population_checkpoint_accepts_legacy_raw_rng_state_payload():
    source = _model()
    source.ambient_rng.reseed(6061)
    for _ in range(3):
        _advance(source)
    checkpoint = _clone_nested(source.state_dict_for_checkpoint())
    for name, payload in checkpoint["stochastic"]["rng_states"].items():
        checkpoint["stochastic"]["rng_states"][name] = payload["rng_state"]

    expected = _advance(source)
    resumed = _model()
    resumed.restore_dict_from_checkpoint(checkpoint)

    _assert_nested_equal(_advance(resumed), expected)


def test_checkpoint_restore_failure_rolls_back_every_runtime_state():
    model = _model()
    for _ in range(3):
        _advance(model)
    checkpoint = _clone_nested(model.state_dict_for_checkpoint())
    _advance(model)
    live_before_failure = _clone_nested(model.state_dict_for_checkpoint())

    del checkpoint["integrator"]["v"]
    with pytest.raises(KeyError, match="integrator state is missing 'v'"):
        model.restore_dict_from_checkpoint(checkpoint)

    _assert_nested_equal(model.state_dict_for_checkpoint(), live_before_failure)


def test_checkpoint_shape_mismatch_is_rejected_without_mutation():
    source = _model(compartments=2)
    for _ in range(2):
        _advance(source)
    checkpoint = _clone_nested(source.state_dict_for_checkpoint())

    target = _model(compartments=3)
    _advance(target)
    before = _clone_nested(target.state_dict_for_checkpoint())

    with pytest.raises(ValueError, match="integrator state 'v'.*shape"):
        target.restore_dict_from_checkpoint(checkpoint)

    _assert_nested_equal(target.state_dict_for_checkpoint(), before)


def test_public_load_keeps_partial_compatibility_and_ignores_mismatches():
    model = _model()
    original_diam = model.diam.detach().clone()

    returned = model.load(
        {
            "v": torch.full_like(model.v, -42.0),
            "diam": torch.zeros(99, dtype=DTYPE),
            "not_a_model_key": torch.tensor(1.0, dtype=DTYPE),
        }
    )

    assert returned is model
    assert torch.equal(model.v, torch.full_like(model.v, -42.0))
    assert torch.equal(model.diam, original_diam)


def test_public_load_corrupt_rng_payload_rolls_back_earlier_tensor_updates():
    model = _model()
    before = _clone_nested(model.state_dict())
    corrupt = _clone_nested(model.state_dict())
    corrupt["v"] = torch.full_like(model.v, -12.0)
    rng_key = next(key for key in corrupt if key.endswith("eta_rng._extra_state"))
    corrupt[rng_key] = {
        "base_seed": 1729,
        "rng_state": {"cpu": torch.zeros(3, dtype=torch.uint8)},
    }

    with pytest.raises(RuntimeError, match="not a valid generator state"):
        model.load(corrupt)

    _assert_nested_equal(model.state_dict(), before)


def test_public_load_rejects_non_mapping_without_mutation():
    model = _model()
    before = _clone_nested(model.state_dict())

    with pytest.raises(TypeError, match="state_dict must be a mapping"):
        model.load([("v", torch.zeros_like(model.v))])

    _assert_nested_equal(model.state_dict(), before)


@pytest.mark.parametrize(
    "malformation",
    (
        "empty",
        "missing_rng_state",
        "missing_base_seed",
        "boolean_base_seed",
        "empty_rng_state",
        "legacy_raw_state",
    ),
)
def test_public_population_load_rejects_malformed_rng_metadata_atomically(
    malformation,
):
    model = _model()
    before = _clone_nested(model.state_dict())
    corrupt = _clone_nested(before)
    corrupt["v"].fill_(-17.0)
    rng_key = next(key for key in corrupt if key.endswith("eta_rng._extra_state"))
    valid = before[rng_key]

    payloads = {
        "empty": {},
        "missing_rng_state": {"base_seed": valid["base_seed"]},
        "missing_base_seed": {"rng_state": _clone_nested(valid["rng_state"])},
        "boolean_base_seed": {
            "base_seed": True,
            "rng_state": _clone_nested(valid["rng_state"]),
        },
        "empty_rng_state": {"base_seed": valid["base_seed"], "rng_state": {}},
        "legacy_raw_state": _clone_nested(valid["rng_state"]),
    }
    corrupt[rng_key] = payloads[malformation]

    with pytest.raises((KeyError, TypeError, ValueError)):
        model.load(corrupt)

    _assert_nested_equal(model.state_dict(), before)


@pytest.mark.parametrize(
    "malformation",
    ("empty", "missing_cpu", "invalid_cpu", "invalid_device", "partial_modern"),
)
def test_population_checkpoint_rejects_malformed_legacy_rng_state_atomically(
    malformation,
):
    model = _model()
    for _ in range(2):
        _advance(model)
    before = _clone_nested(model.state_dict_for_checkpoint())
    corrupt = _clone_nested(before)
    rng_name = next(iter(corrupt["stochastic"]["rng_states"]))
    valid = before["stochastic"]["rng_states"][rng_name]
    valid_state = valid["rng_state"]

    payloads = {
        "empty": {},
        "missing_cpu": {"cuda:7": valid_state["cpu"].clone()},
        "invalid_cpu": {"cpu": torch.zeros(3, dtype=torch.uint8)},
        "invalid_device": {
            "cpu": valid_state["cpu"].clone(),
            "cuda:7": torch.ones(8, dtype=torch.int64),
        },
        "partial_modern": {"base_seed": valid["base_seed"]},
    }
    corrupt["stochastic"]["rng_states"][rng_name] = payloads[malformation]
    corrupt["integrator"]["v"].fill_(-11.0)

    with pytest.raises((KeyError, RuntimeError, TypeError, ValueError)):
        model.restore_dict_from_checkpoint(corrupt)

    _assert_nested_equal(model.state_dict_for_checkpoint(), before)


def test_public_network_load_rejects_empty_rng_metadata_atomically():
    network = dn.Network({}, seed=2718)
    before = _clone_nested(network.state_dict())
    corrupt = _clone_nested(before)
    corrupt["t"].fill_(9.0)
    corrupt["_extra_state"] = {}

    with pytest.raises(KeyError, match="base_seed.*rng_state"):
        network.load(corrupt)

    _assert_nested_equal(network.state_dict(), before)


def test_network_checkpoint_rejects_population_rng_corruption_atomically():
    population = _model()
    network = dn.Network({"population": population}, seed=31415)
    before = _clone_nested(network.state_dict_for_checkpoint())
    corrupt = _clone_nested(before)
    rng_states = corrupt["populations"]["population"]["stochastic"]["rng_states"]
    rng_states[next(iter(rng_states))] = {}
    corrupt["t"].fill_(12.0)

    with pytest.raises(KeyError, match="cpu"):
        network.restore_dict_from_checkpoint(corrupt)

    _assert_nested_equal(network.state_dict_for_checkpoint(), before)
