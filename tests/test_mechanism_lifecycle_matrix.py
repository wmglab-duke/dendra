from __future__ import annotations

import copy
import math
import warnings

import pytest
import torch

import dendra  # noqa: F401 - initialize Dendra before mechanism imports
from dendra.models.mechanisms import Mechanism, PointProcess, State
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.mechanisms._ions import Ion
from dendra.models.mechanisms._materials import Material
from dendra.models.mod.PAS import pas


class _PointLeak(PointProcess):
    PointProcess.RANGE(g=2.0, e=-5.0)
    PointProcess.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.g * (v - self.e)


class _SodiumLeak(Mechanism):
    Mechanism.RANGE(g=0.01, e=50.0)
    Mechanism.USEION("na", write=["ina"])

    def ina(self, v):
        return self.g * (v - self.e)


class _SodiumWriter(Mechanism):
    Mechanism.USEION("na", write=["nai"])


class _MaterialUser(Mechanism):
    Mechanism.USEMATERIAL(
        "pool", read=["amount"], write=["amount"], source={"amount": "delta"}
    )


class _DecayState(State):
    State.STATE("x")
    State.DERIVATIVE("x' = -x")


class _StatefulBuffer(Mechanism):
    Mechanism.STATE(_DecayState)
    Mechanism.BUFFER("scratch")


class _RecoveryState(State):
    State.STATE("y")
    State.DERIVATIVE("y' = -y")


class _MultiStatefulBuffer(Mechanism):
    Mechanism.STATE(_DecayState, _RecoveryState)


class _DerivedLifecycleState(State):
    State.STATE("x")
    State.DERIVATIVE("x' = -x")
    State.DERIVED_BUFFER("state_scale")
    State.BUFFER("initial_seen")

    def derive_buffers(self):
        return {"state_scale": self.celsius - self.diam}

    def initial(self, v):
        self.initial_seen = self.state_scale + v


class _DerivedLifecycleMechanism(Mechanism):
    Mechanism.STATE(_DerivedLifecycleState)
    Mechanism.DERIVED_BUFFER("gain")
    Mechanism.BUFFER("initial_seen")

    def derive_buffers(self):
        return {"gain": 2.0 * self.celsius + self.diam}

    def initial(self, v):
        self.initial_seen = self.gain + v


class _DerivedInfState(State):
    State.STATE("x")
    State.DERIVATIVE("x' = -x")
    State.DERIVED_BUFFER("drive")
    State.BUFFER("initial_seen")

    def derive_buffers(self):
        return {"drive": 3.0 * self.diam}

    def inf(self, v):
        return {"x": self.drive}

    def initial(self, v):
        self.initial_seen = self.drive + v


class _DerivedInfMechanism(Mechanism):
    Mechanism.STATE(_DerivedInfState)
    Mechanism.BUFFER("state_drive_seen")

    def initial(self, v):
        self.state_drive_seen = self.DE._DerivedInfState.drive


class _ConstantWaveform(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.register_buffer("value", torch.as_tensor(value, dtype=torch.float64))

    def forward(self, _t):
        return self.value


class _InjectionProbe:
    def __init__(self, accepts):
        self.accepts = accepts
        self.calls = []

    def inject(self, waveform, **kwargs):
        self.calls.append((waveform, kwargs))
        return self.accepts


class _InjectionHandler:
    def __init__(self, **mechanisms):
        self.mechanisms = mechanisms


def _base_mechanism(shape=(2, 3), *, key=None):
    return Mechanism(
        "base",
        torch.full(shape, 34.0, dtype=torch.float64),
        torch.ones(shape, dtype=torch.float64),
        shape,
        shape,
        key=key,
    )


def _derived_lifecycle_mechanism():
    shape = (2, 3)
    celsius = torch.full(
        shape,
        34.0,
        dtype=torch.float64,
        requires_grad=True,
    )
    diam = torch.linspace(
        1.0,
        2.0,
        math.prod(shape),
        dtype=torch.float64,
        requires_grad=True,
    ).reshape(shape)
    mechanism = _DerivedLifecycleMechanism(
        "derived",
        celsius,
        diam,
        shape,
        shape,
    )
    return mechanism, celsius, diam


def _population_with_registered_injection(*, accepted=False):
    population = dendra.Population(N=1, C=1, dtype=torch.float64)
    waveform = dendra.constant(value=1.0)
    index = (slice(None), slice(None))
    spec = (waveform, tuple(population.v[index].shape), index)
    population.injections = [spec]
    population.mechanism_injections = [spec]
    population.mechanism_injection_accepted = [accepted]
    return population


def _registered_single(delay, mode):
    mech = _base_mechanism(shape=(2,))
    mech.register_delayed_state(
        "signal", torch.zeros(2, dtype=torch.float64), delay, mode=mode
    )
    return mech


def _run_single(mech, values, *, mode=None, delay_steps=None):
    outputs = []
    context = torch.no_grad() if mode == "circular" else torch.enable_grad()
    with context:
        for value in values:
            outputs.append(
                mech.delayed_state(
                    "signal", value, mode=mode, delay_steps=delay_steps
                ).clone()
            )
    return outputs


def test_derived_buffers_refresh_before_authored_initial_and_retain_gradients():
    mechanism, celsius, diam = _derived_lifecycle_mechanism()
    voltage = torch.linspace(
        -70.0,
        -60.0,
        mechanism.diam.numel(),
        dtype=mechanism.diam.dtype,
    ).reshape_as(mechanism.diam)

    mechanism._init_buffers_s(voltage)
    state = mechanism.DE["_DerivedLifecycleState"]

    expected_gain = 2.0 * celsius + diam
    expected_state_scale = celsius - diam
    torch.testing.assert_close(mechanism.gain, expected_gain)
    torch.testing.assert_close(mechanism.initial_seen, expected_gain + voltage)
    torch.testing.assert_close(state.state_scale, expected_state_scale)
    torch.testing.assert_close(state.initial_seen, expected_state_scale + voltage)

    celsius_gradient, diameter_gradient = torch.autograd.grad(
        mechanism.gain.sum() + 2.0 * state.state_scale.sum(),
        (celsius, diam),
    )
    torch.testing.assert_close(
        celsius_gradient,
        torch.full_like(celsius_gradient, 4.0),
    )
    torch.testing.assert_close(
        diameter_gradient,
        torch.full_like(diameter_gradient, -1.0),
    )

    assert "gain" in mechanism._assigned
    assert "gain" in mechanism.state_dict()
    assert "state_scale" in state._state_buffers
    assert "DE._DerivedLifecycleState.state_scale" in mechanism.state_dict()

    handler = MechanismHandler(
        celsius,
        torch.ones_like(diam),
        {"derived": mechanism},
    )
    checkpoint = handler.mutable_state_dict()
    torch.testing.assert_close(checkpoint["derived.gain"], mechanism.gain)
    torch.testing.assert_close(
        checkpoint["derived.DE._DerivedLifecycleState.state_scale"],
        state.state_scale,
    )


def test_state_derived_buffers_are_ready_for_inf_and_mechanism_initial():
    shape = (2, 3)
    diam = torch.arange(1, 7, dtype=torch.float64).reshape(shape)
    voltage = torch.full(shape, -65.0, dtype=torch.float64)
    mechanism = _DerivedInfMechanism(
        "derived_inf",
        torch.tensor(34.0, dtype=torch.float64),
        diam,
        shape,
        shape,
    )

    mechanism._init_buffers_s(voltage)
    state = mechanism.DE._DerivedInfState
    expected_drive = 3.0 * diam
    torch.testing.assert_close(mechanism.x, expected_drive, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        mechanism.state_drive_seen,
        expected_drive,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        state.initial_seen,
        expected_drive + voltage,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("builder", "error", "message"),
    [
        (lambda self: None, TypeError, "must return a mapping"),
        (lambda self: {}, ValueError, "keys do not match"),
        (
            lambda self: {
                "workspace": self.diam,
                "unexpected": self.diam,
            },
            ValueError,
            "unexpected=.*unexpected",
        ),
        (
            lambda self: {"workspace": 1.0},
            TypeError,
            "must be a Tensor",
        ),
        (
            lambda self: {"workspace": self.diam.float()},
            ValueError,
            "must use.*float64",
        ),
        (
            lambda self: {"workspace": self.diam.new_zeros(2, 2)},
            ValueError,
            "not broadcastable",
        ),
    ],
)
def test_derived_buffer_builder_contract_fails_before_authored_initial(
    builder,
    error,
    message,
):
    class InvalidDerivedBuffer(Mechanism):
        Mechanism.DERIVED_BUFFER("workspace")
        Mechanism.BUFFER("initial_seen")

        def derive_buffers(self):
            return builder(self)

        def initial(self, v):
            self.initial_seen = torch.ones_like(v)

    shape = (2, 3)
    mechanism = InvalidDerivedBuffer(
        "invalid",
        torch.full(shape, 34.0, dtype=torch.float64),
        torch.ones(shape, dtype=torch.float64),
        shape,
        shape,
    )
    with pytest.raises(error, match=message):
        mechanism._init_buffers_s(torch.zeros(shape, dtype=torch.float64))
    assert torch.count_nonzero(mechanism.initial_seen) == 0


def test_derived_buffer_builder_rejects_registered_tensor_mutation():
    class MutatingDerivedBuffer(Mechanism):
        Mechanism.DERIVED_BUFFER("workspace")

        def derive_buffers(self):
            self.workspace.add_(1.0)
            return {"workspace": self.diam}

    shape = (2, 3)
    mechanism = MutatingDerivedBuffer(
        "mutating",
        torch.full(shape, 34.0, dtype=torch.float64),
        torch.ones(shape, dtype=torch.float64),
        shape,
        shape,
    )
    with pytest.raises(RuntimeError, match="must not mutate registered"):
        mechanism._init_buffers_s(torch.zeros(shape, dtype=torch.float64))


def test_derived_buffer_builder_supports_and_guards_inference_tensors():
    class InferenceDerivedBuffer(Mechanism):
        Mechanism.DERIVED_BUFFER("workspace")

        def derive_buffers(self):
            return {"workspace": 2.0 * self.diam}

    class MutatingInferenceDerivedBuffer(Mechanism):
        Mechanism.DERIVED_BUFFER("workspace")

        def derive_buffers(self):
            self.workspace.add_(1.0)
            return {"workspace": 2.0 * self.diam}

    class NaNInferenceDerivedBuffer(Mechanism):
        Mechanism.DERIVED_BUFFER("workspace")

        def derive_buffers(self):
            return {"workspace": torch.full_like(self.diam, torch.nan)}

    shape = (2, 3)
    voltage = torch.zeros(shape, dtype=torch.float64)
    with torch.inference_mode():
        mechanism = InferenceDerivedBuffer(
            "inference",
            torch.tensor(34.0, dtype=torch.float64),
            torch.ones(shape, dtype=torch.float64),
            shape,
            shape,
        )
        mechanism._init_buffers_s(voltage)
        mechanism._init_buffers_s(voltage)
        torch.testing.assert_close(
            mechanism.workspace,
            torch.full(shape, 2.0, dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )

        mutating = MutatingInferenceDerivedBuffer(
            "mutating_inference",
            torch.tensor(34.0, dtype=torch.float64),
            torch.ones(shape, dtype=torch.float64),
            shape,
            shape,
        )
        with pytest.raises(RuntimeError, match="must not mutate registered"):
            mutating._init_buffers_s(voltage)

        nan_workspace = NaNInferenceDerivedBuffer(
            "nan_inference",
            torch.tensor(34.0, dtype=torch.float64),
            torch.ones(shape, dtype=torch.float64),
            shape,
            shape,
        )
        nan_workspace._init_buffers_s(voltage)
        nan_workspace._init_buffers_s(voltage)
        assert torch.isnan(nan_workspace.workspace).all()


def test_derived_buffer_expanded_views_are_independent_checkpoint_destinations():
    class ExpandedDerivedBuffer(Mechanism):
        Mechanism.DERIVED_BUFFER("workspace")

        def derive_buffers(self):
            return {"workspace": self.celsius.expand_as(self.diam)}

    shape = (2, 3)

    def make_mechanism():
        mechanism = ExpandedDerivedBuffer(
            "expanded",
            torch.tensor(34.0, dtype=torch.float64),
            torch.ones(shape, dtype=torch.float64),
            shape,
            shape,
        )
        mechanism._init_buffers_s(torch.zeros(shape, dtype=torch.float64))
        return mechanism

    source = make_mechanism()
    target = make_mechanism()
    assert source.workspace.stride() != (0, 0)
    assert (
        source.workspace.untyped_storage().data_ptr()
        != source.celsius.untyped_storage().data_ptr()
    )

    target.load_state_dict(source.state_dict())
    torch.testing.assert_close(target.workspace, source.workspace, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("delay", [0, 1, 3])
@pytest.mark.parametrize("mode", ["shift", "circular"])
def test_single_delayed_state_matches_exact_queue(delay, mode):
    mech = _registered_single(delay, mode)
    values = [
        torch.tensor([float(k), float(k + 10)], dtype=torch.float64) for k in range(5)
    ]

    outputs = _run_single(mech, values, mode=mode)

    for k, output in enumerate(outputs):
        expected = values[k - delay] if k >= delay else torch.zeros_like(values[k])
        torch.testing.assert_close(output, expected)


def test_single_delayed_state_auto_selects_graph_safe_and_eval_backends():
    train_mech = _registered_single(1, "auto")
    train_mech.train()
    train_buffer = train_mech.signal_delay_buffer
    train_mech.delayed_state(
        "signal", torch.ones(2, dtype=torch.float64, requires_grad=True)
    )
    assert train_mech.signal_delay_buffer is not train_buffer

    eval_mech = _registered_single(1, "auto")
    eval_mech.eval()
    eval_buffer = eval_mech.signal_delay_buffer
    with torch.no_grad():
        eval_mech.delayed_state("signal", torch.ones(2, dtype=torch.float64))
    assert eval_mech.signal_delay_buffer is eval_buffer
    assert eval_mech.signal_delay_ptr.item() == 1


def test_single_delayed_state_reset_override_and_gradient():
    mech = _registered_single(1, "shift")
    first = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
    second = torch.tensor([5.0, 7.0], dtype=torch.float64, requires_grad=True)

    assert torch.equal(mech.delayed_state("signal", first), torch.zeros(2))
    delayed = mech.delayed_state("signal", second)
    torch.testing.assert_close(delayed, first)
    delayed.sum().backward()
    torch.testing.assert_close(first.grad, torch.ones_like(first))
    torch.testing.assert_close(second.grad, torch.zeros_like(second))

    mech.reset_delayed_states("signal")
    assert torch.count_nonzero(mech.signal_delay_buffer) == 0
    assert mech.signal_delay_ptr.item() == 0

    delayed = mech.delayed_state("signal", torch.ones(2), delay_steps=3, mode="shift")
    assert torch.equal(delayed, torch.zeros(2))
    assert mech.signal_delay_buffer.shape == (2, 4)
    assert mech._delayed_state_specs["signal"]["steps"] == 3


def test_delayed_state_reset_preserves_outstanding_backward_and_severs_history():
    mech = _registered_single(1, "shift")
    first = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
    second = torch.tensor([5.0, 7.0], dtype=torch.float64, requires_grad=True)
    mech.delayed_state("signal", first)
    delayed = mech.delayed_state("signal", second)
    loss = delayed.square().sum()

    mech.reset_delayed_states("signal")

    # Resetting the queue must not mutate a value already returned to the graph.
    torch.testing.assert_close(delayed, first)
    loss.backward()
    torch.testing.assert_close(first.grad, 2 * first.detach())

    first_grad = first.grad.clone()
    new_first = torch.tensor([11.0, 13.0], dtype=torch.float64, requires_grad=True)
    new_second = torch.tensor([17.0, 19.0], dtype=torch.float64, requires_grad=True)
    mech.delayed_state("signal", new_first)
    mech.delayed_state("signal", new_second).sum().backward()
    torch.testing.assert_close(first.grad, first_grad)
    torch.testing.assert_close(new_first.grad, torch.ones_like(new_first))


def test_multi_delayed_state_reset_severs_old_autograd_history():
    mech = _base_mechanism(shape=(2, 2))
    mech.register_delayed_states(
        "streams", torch.zeros(2, 2), [1, 1], mode="shift", stream_axis=0
    )
    old = torch.ones(2, 2, requires_grad=True)
    mech.delayed_states("streams", old)
    mech.reset_delayed_states("streams")
    new = torch.full((2, 2), 2.0, requires_grad=True)
    mech.delayed_states("streams", new)
    output = mech.delayed_states("streams", new + 1)

    output.sum().backward()

    assert old.grad is None
    torch.testing.assert_close(new.grad, torch.ones_like(new))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"delay_steps": -1}, "non-negative"),
        ({"delay_steps": 1.5}, "integer"),
        ({"delay_steps": 1, "mode": "mystery"}, "mode"),
        ({"delay_steps": 1, "insert_axis": 4}, "insert_axis"),
    ],
)
def test_single_delayed_state_registration_validation(kwargs, message):
    mech = _base_mechanism(shape=(2,))
    with pytest.raises((TypeError, ValueError), match=message):
        mech.register_delayed_state("signal", torch.zeros(2), **kwargs)


def test_single_delayed_state_call_validation_and_wrong_api():
    mech = _registered_single(1, "shift")
    with pytest.raises(ValueError, match="mode"):
        mech.delayed_state("signal", torch.ones(2), mode="mystery")
    for delay, message in [(-1, "non-negative"), (1.5, "integer")]:
        with pytest.raises(ValueError, match=message):
            mech.delayed_state("signal", torch.ones(2), delay_steps=delay)

    with pytest.raises(KeyError, match="No delayed state"):
        mech.delayed_state("missing", torch.ones(2))

    circular = _registered_single(1, "circular")
    with pytest.raises(ValueError, match="cannot change"):
        circular.delayed_state("signal", torch.ones(2), delay_steps=2, mode="circular")

    multi = _base_mechanism(shape=(3, 2))
    multi.register_delayed_states("group", torch.zeros(3, 2), [0, 1, 2])
    with pytest.raises(ValueError, match="registered with register_delayed_states"):
        multi.delayed_state("group", torch.ones(3, 2))

    with pytest.raises(ValueError, match="use delayed_state"):
        mech.delayed_states("signal", torch.ones(2))

    zero = _registered_single(0, "shift")
    with pytest.raises(ValueError, match="mode"):
        zero.delayed_state("signal", torch.ones(2), mode="mystery")
    with pytest.raises(ValueError, match="shape"):
        zero.delayed_state("signal", torch.ones(3))


def test_delayed_state_buffers_follow_module_dtype_changes():
    mech = _base_mechanism(shape=(2,)).to(dtype=torch.float32)
    mech.register_delayed_state("signal", torch.zeros(2), 1, mode="shift")
    mech = mech.to(dtype=torch.float64)
    value = torch.ones(2, dtype=torch.float64)

    output = mech.delayed_state("signal", value)

    assert output.dtype == torch.float64
    assert mech.signal_delay_buffer.dtype == torch.float64


def _run_multi(mech, values, *, mode, delay_steps=None):
    outputs = []
    context = torch.no_grad() if mode == "circular" else torch.enable_grad()
    with context:
        for value in values:
            outputs.append(
                mech.delayed_states(
                    "streams", value, mode=mode, delay_steps=delay_steps
                ).clone()
            )
    return outputs


@pytest.mark.parametrize("mode", ["shift", "circular"])
@pytest.mark.parametrize("delay_axis", [None, 0])
def test_multi_delayed_states_match_heterogeneous_reference(mode, delay_axis):
    delays = torch.tensor([0, 1, 3])
    mech = _base_mechanism(shape=(2, 3, 2))
    like = torch.zeros(2, 3, 2, dtype=torch.float64)
    mech.register_delayed_states(
        "streams",
        like,
        delays,
        mode=mode,
        stream_axis=1,
        delay_axis=delay_axis,
    )
    values = [
        torch.arange(12, dtype=torch.float64).reshape(2, 3, 2) + 100 * k
        for k in range(5)
    ]

    outputs = _run_multi(mech, values, mode=mode)

    for k, output in enumerate(outputs):
        expected = torch.zeros_like(output)
        for stream, delay in enumerate(delays.tolist()):
            if k >= delay:
                expected[:, stream] = values[k - delay][:, stream]
        torch.testing.assert_close(output, expected)


def test_multi_delayed_states_runtime_override_resizes_and_honors_zero_delay():
    values = torch.arange(6, dtype=torch.float64).reshape(3, 2) + 1
    shift = _base_mechanism(shape=(3, 2))
    shift.register_delayed_states(
        "streams", torch.zeros_like(values), [1, 1, 1], mode="shift", stream_axis=0
    )

    output = shift.delayed_states(
        "streams", values, delay_steps=[0, 2, 1], mode="shift"
    )
    torch.testing.assert_close(output[0], values[0])
    assert torch.count_nonzero(output[1:]) == 0
    assert shift.streams_delay_buffer.shape[0] == 3
    assert shift._delayed_state_specs["streams"]["steps"] == 2

    circular = _base_mechanism(shape=(3, 2))
    circular.register_delayed_states(
        "streams",
        torch.zeros_like(values),
        [1, 1, 1],
        mode="circular",
        stream_axis=0,
    )
    with torch.no_grad():
        output = circular.delayed_states(
            "streams", values, delay_steps=[0, 1, 1], mode="circular"
        )
    torch.testing.assert_close(output[0], values[0])
    assert torch.count_nonzero(output[1:]) == 0

    with pytest.raises(ValueError, match="capacity"):
        with torch.no_grad():
            circular.delayed_states(
                "streams", values, delay_steps=[0, 1, 3], mode="circular"
            )


def test_multi_delayed_states_gradient_routes_to_the_right_timestep_and_stream():
    mech = _base_mechanism(shape=(2, 2))
    mech.register_delayed_states(
        "streams", torch.zeros(2, 2), [0, 1], mode="shift", stream_axis=0
    )
    first = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    second = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    mech.delayed_states("streams", first)
    output = mech.delayed_states("streams", second)

    output.sum().backward()

    torch.testing.assert_close(first.grad, torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
    torch.testing.assert_close(second.grad, torch.tensor([[1.0, 1.0], [0.0, 0.0]]))


@pytest.mark.parametrize("mode", ["shift", "circular"])
def test_multi_delayed_states_common_path_is_torch_compile_fullgraph_safe(mode):
    shape = (3, 2)
    mech = _base_mechanism(shape=shape).to(dtype=torch.float32)
    mech.register_delayed_states(
        "streams", torch.zeros(shape), [0, 1, 2], mode=mode, stream_axis=0
    )

    def step(values):
        return mech.delayed_states("streams", values, mode=mode)

    compiled = torch.compile(step, backend="eager", fullgraph=True)
    values = torch.arange(6, dtype=torch.float32).reshape(shape)
    context = torch.no_grad() if mode == "circular" else torch.enable_grad()
    with context:
        output = compiled(values)

    expected = torch.zeros_like(values)
    expected[0] = values[0]
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    "like, delays, kwargs, message",
    [
        (torch.tensor(0.0), 1, {}, "non-scalar"),
        (torch.empty(0, 2), [], {"stream_axis": 0}, "non-empty"),
        (torch.zeros(3, 2), [1, 2], {"stream_axis": 0}, "length"),
        (torch.zeros(3, 2), [0, -1, 2], {"stream_axis": 0}, "non-negative"),
        (torch.zeros(3, 2), [0, 1.5, 2], {"stream_axis": 0}, "integer"),
        (torch.zeros(3, 2), [0, 1, 2], {"stream_axis": 4}, "stream_axis"),
        (
            torch.zeros(3, 2),
            [0, 1, 2],
            {"stream_axis": 0, "delay_axis": 4},
            "delay_axis",
        ),
        (
            torch.zeros(3, 2),
            [0, 1, 2],
            {"stream_axis": 0, "mode": "mystery"},
            "mode",
        ),
    ],
)
def test_multi_delayed_state_registration_validation(like, delays, kwargs, message):
    mech = _base_mechanism(shape=(3, 2))
    with pytest.raises((TypeError, ValueError), match=message):
        mech.register_delayed_states("streams", like, delays, **kwargs)


def test_multi_delayed_state_runtime_override_validation():
    mech = _base_mechanism(shape=(3, 2))
    values = torch.ones(3, 2)
    mech.register_delayed_states(
        "streams", values, [0, 1, 2], mode="shift", stream_axis=0
    )

    for delays, message in [
        ([0, 1], "length"),
        ([0, -1, 2], "non-negative"),
        ([0, 1.5, 2], "integer"),
    ]:
        with pytest.raises(ValueError, match=message):
            mech.delayed_states("streams", values, delay_steps=delays)
    with pytest.raises(ValueError, match="mode"):
        mech.delayed_states("streams", values, mode="mystery")
    with pytest.raises(KeyError, match="No delayed state group"):
        mech.delayed_states("missing", values)

    all_zero = _base_mechanism(shape=(3, 2))
    all_zero.register_delayed_states(
        "streams", values, [0, 0, 0], mode="shift", stream_axis=0
    )
    with pytest.raises(ValueError, match="mode"):
        all_zero.delayed_states("streams", values, mode="mystery")
    with pytest.raises(ValueError, match="shape"):
        all_zero.delayed_states("streams", torch.ones(4, 2))


def test_delayed_state_reregistration_without_clear_updates_metadata_safely():
    single = _registered_single(3, "circular")
    with torch.no_grad():
        for value in range(3):
            single.delayed_state(
                "signal",
                torch.full((2,), float(value), dtype=torch.float64),
                mode="circular",
            )
    single.register_delayed_state(
        "signal",
        torch.zeros(2, dtype=torch.float64),
        1,
        mode="circular",
        clear=False,
    )
    assert single.signal_delay_ptr.item() == 0
    with torch.no_grad():
        torch.testing.assert_close(
            single.delayed_state(
                "signal", torch.ones(2, dtype=torch.float64), mode="circular"
            ),
            torch.zeros(2, dtype=torch.float64),
        )

    multi = _base_mechanism(shape=(3, 2))
    like = torch.zeros(3, 2)
    multi.register_delayed_states(
        "streams", like, [0, 1, 2], mode="shift", stream_axis=0
    )
    original_buffer = multi.streams_delay_buffer
    multi.register_delayed_states(
        "streams", like, [2, 1, 0], mode="shift", stream_axis=0, clear=False
    )
    assert multi.streams_delay_buffer is original_buffer
    assert multi.streams_delay_steps.tolist() == [2, 1, 0]

    circular = _base_mechanism(shape=(3, 2))
    circular.register_delayed_states(
        "streams", like, [3, 3, 3], mode="circular", stream_axis=0
    )
    with torch.no_grad():
        for value in range(3):
            circular.delayed_states(
                "streams", torch.full_like(like, float(value)), mode="circular"
            )
    circular.register_delayed_states(
        "streams", like, [1, 1, 1], mode="circular", stream_axis=0, clear=False
    )
    assert circular.streams_delay_ptr.item() == 0
    assert circular.streams_delay_steps.tolist() == [1, 1, 1]
    with torch.no_grad():
        circular.delayed_states("streams", torch.ones_like(like), mode="circular")


def test_tensor_key_copy_preserves_device_dtype_and_independence_without_warning():
    key = torch.tensor([1, 4], dtype=torch.long)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        mech = _base_mechanism(key=key)

    assert mech.key.dtype == torch.long
    assert mech.key.device == key.device
    assert mech.key.data_ptr() != key.data_ptr()
    key[0] = 0
    assert mech.key.tolist() == [1, 4]


@pytest.mark.parametrize("support_kind", ["packed", "slice"])
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
def test_handler_set_buffers_keeps_restricted_state_diameters_local(
    support_kind, batch_shape
):
    source_shape = (2, 4)
    celsius = torch.full(source_shape, 34.0, dtype=torch.float64)
    initial_diameters = torch.arange(1, 9, dtype=torch.float64).reshape(source_shape)

    if support_kind == "packed":
        key = torch.tensor([1, 3, 4, 6], dtype=torch.long)
        is_composable = False
        local_shape = (4,)

        def gather(values):
            return values.reshape(*values.shape[:-2], -1).index_select(-1, key)

    else:
        key = (slice(None), slice(1, 3))
        is_composable = True
        local_shape = (2, 2)

        def gather(values):
            return values[..., *key]

    mech = _MultiStatefulBuffer(
        "restricted",
        celsius,
        initial_diameters,
        local_shape,
        local_shape,
        key=key,
        is_composable=is_composable,
    )
    handler = MechanismHandler(
        celsius,
        torch.ones(source_shape, dtype=torch.float64),
        {"restricted": mech},
    )

    assert len(mech.DE) == 2
    rebound_shape = (*batch_shape, *source_shape)
    value_count = math.prod(rebound_shape)

    for offset in (10.0, 100.0):
        population_diameters = (
            torch.arange(value_count, dtype=torch.float64).reshape(rebound_shape)
            + offset
        )
        expected = gather(population_diameters).clone()
        previous_diameters = mech.diam

        handler.set_buffers(population_diameters)

        assert mech.diam is not previous_diameters
        assert mech._buffers["diam"] is mech.diam
        assert tuple(mech.diam.shape) == (*batch_shape, *local_shape)
        torch.testing.assert_close(mech.diam, expected)
        for state in mech.DE.values():
            assert state.diam is mech.diam
            assert state._buffers["diam"] is mech.diam
            torch.testing.assert_close(state.diam, expected)

        # Rebinding owns its local geometry rather than aliasing the caller's
        # population-wide tensor.
        population_diameters.add_(1000.0)
        torch.testing.assert_close(mech.diam, expected)


def test_waveform_injections_respect_current_names_masks_scales_and_clear():
    mech = _base_mechanism()
    column_one = (slice(None), 1)
    column_two = (slice(None), 2)
    assert mech.register_waveform_injection(
        _ConstantWaveform(2.0),
        index=column_one,
        model_shape=(2, 3),
        current_name="i_a",
        scale=3.0,
    )
    assert mech.register_waveform_injection(
        _ConstantWaveform(5.0),
        index=column_two,
        model_shape=(2, 3),
        current_name="i_b",
    )

    i_a = mech.evaluate_injections(t=0.0, current_name="i_a")
    i_b = mech.evaluate_injections(t=0.0, current_name="i_b")
    expected_a = torch.zeros(2, 3, dtype=torch.float64)
    expected_b = torch.zeros_like(expected_a)
    expected_a[:, 1] = 6.0
    expected_b[:, 2] = 5.0
    torch.testing.assert_close(i_a, expected_a)
    torch.testing.assert_close(i_b, expected_b)

    assert mech.clear_injections() is mech
    assert len(mech.injected_waveforms) == 0
    assert not mech._injection_specs
    assert not any(name.startswith("_injection_") for name in mech._buffers)
    torch.testing.assert_close(
        mech.evaluate_injections(current_name="i_a"), torch.zeros_like(expected_a)
    )


def test_waveform_injection_rejects_an_empty_population_overlap():
    mech = _base_mechanism()
    accepted = mech.register_waveform_injection(
        _ConstantWaveform(2.0),
        index=(slice(None), slice(0, 0)),
        model_shape=(2, 3),
    )

    assert not accepted
    assert len(mech.injected_waveforms) == 0
    assert not mech._injection_specs


def test_waveform_injection_evaluates_builtin_time_last_values_at_scalar_time():
    mech = _base_mechanism()
    waveform = dendra.constant(value=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    assert waveform(torch.tensor([0.0], dtype=torch.float64)).shape == (3, 1)
    assert mech.register_waveform_injection(
        waveform,
        index=(slice(None), slice(None)),
        model_shape=(2, 3),
    )

    actual = mech.evaluate_injections(t=0.0)
    expected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64).expand(2, 3)
    torch.testing.assert_close(actual, expected)


def test_waveform_injection_rejects_partial_model_coverage_without_state():
    mech = _base_mechanism(key=torch.tensor([0, 1], dtype=torch.long))
    buffers_before = set(mech._buffers)

    accepted = mech.register_waveform_injection(
        dendra.constant(value=2.0),
        index=(slice(None), slice(None)),
        model_shape=(2, 3),
    )

    assert not accepted
    assert len(mech.injected_waveforms) == 0
    assert not mech._injection_specs
    assert set(mech._buffers) == buffers_before
    assert not hasattr(mech, "i_inj")


def test_mechanism_injection_dispatch_stops_after_first_full_acceptor():
    population = _population_with_registered_injection()
    first = _InjectionProbe(accepts=True)
    second = _InjectionProbe(accepts=True)
    handler = _InjectionHandler(first=first, second=second)

    population._dispatch_mechanism_injections(handler)

    assert len(first.calls) == 1
    assert second.calls == []
    assert population.mechanism_injection_accepted == [True]
    assert population.build_intra() is None


def test_fresh_mechanism_handler_recomputes_stale_injection_acceptance():
    population = _population_with_registered_injection(accepted=True)
    rejecting = _InjectionProbe(accepts=False)

    population._dispatch_mechanism_injections(
        _InjectionHandler(rejecting=rejecting),
        reset_acceptance=True,
    )

    assert len(rejecting.calls) == 1
    assert population.mechanism_injection_accepted == [False]
    assert population.build_intra() is not None


def test_waveform_selected_vectors_expand_across_batch_dimensions():
    mech = _base_mechanism()
    mask = torch.zeros(2, 3, dtype=torch.bool)
    mask[:, 1] = True
    out = torch.zeros(2, 2, 3, dtype=torch.float64)

    unbatched = mech._expand_injection_value(torch.tensor([4.0, 5.0]), mask, out)
    expected = torch.zeros_like(out)
    expected[:, 0, 1] = 4.0
    expected[:, 1, 1] = 5.0
    torch.testing.assert_close(unbatched, expected)

    batched = mech._expand_injection_value(
        torch.tensor([[4.0, 5.0], [6.0, 7.0]]), mask, out
    )
    expected[1, 0, 1] = 6.0
    expected[1, 1, 1] = 7.0
    torch.testing.assert_close(batched, expected)

    with pytest.raises(ValueError, match="cannot be broadcast"):
        mech._expand_injection_value(torch.ones(5), mask, out)


def test_waveform_selected_vectors_are_torch_compile_fullgraph_safe():
    mech = _base_mechanism(shape=(2, 3)).to(dtype=torch.float32)
    assert mech.register_waveform_injection(
        _ConstantWaveform([4.0, 5.0]),
        index=(slice(None), 1),
        model_shape=(2, 3),
        current_name="i_selected",
    )

    def evaluate(v, t):
        return mech.evaluate_injections(v, t=t, current_name="i_selected")

    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    output = compiled(torch.zeros(2, 2, 3), torch.tensor(0.0))
    expected = torch.zeros_like(output)
    expected[:, 0, 1] = 4.0
    expected[:, 1, 1] = 5.0
    torch.testing.assert_close(output, expected)


def _make_current_handler(*, with_ion=False):
    shape = (1, 3)
    celsius = torch.full(shape, 34.0, dtype=torch.float64)
    diameters = torch.ones(shape, dtype=torch.float64)
    density = pas("density", celsius, diameters, shape, shape).to(torch.float64)
    point = _PointLeak("point", celsius, diameters, shape, shape).to(torch.float64)
    area = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float64)
    ions = {"na": Ion("na", shape, einit=0, eadvance=0)} if with_ion else None
    handler = MechanismHandler(
        celsius,
        area,
        {"density": density, "point": point},
        ions=ions,
        currents={"nonspecific": {"density": ["i"], "point": ["i"]}},
    ).to(torch.float64)
    handler.make_maps()
    handler.init_i_g_bufs(torch.zeros(shape, dtype=torch.float64))
    return handler


def test_handler_current_apis_match_exact_density_and_point_aggregation():
    handler = _make_current_handler()
    v = torch.tensor([[-70.0, -60.0, -50.0]], dtype=torch.float64)
    v_prev = torch.tensor([[-72.0, -62.0, -52.0]], dtype=torch.float64)
    area = handler.area

    def exact(at_v):
        current = 0.001 * (at_v + 70.0) + 2.0 * (at_v + 5.0) / (1e6 * area)
        conductance = torch.full_like(at_v, 0.001) + 2.0 / (1e6 * area)
        return current, conductance

    expected_i, expected_g = exact(v)
    current, conductance = handler.i(v)
    torch.testing.assert_close(current, expected_i)
    torch.testing.assert_close(conductance, expected_g)
    torch.testing.assert_close(handler.iexp(v), expected_i)
    torch.testing.assert_close(handler.itot(v), expected_i)

    expected_df_i, expected_df_g = exact(0.5 * v_prev)
    df_i, df_g = handler.idf(v, v_prev)
    torch.testing.assert_close(df_i, expected_df_i)
    torch.testing.assert_close(df_g, expected_df_g)


def test_handler_noncurrent_ion_is_safe_across_current_apis():
    handler = _make_current_handler(with_ion=True)
    v = torch.tensor([[-70.0, -60.0, -50.0]], dtype=torch.float64)
    expected_i, _ = handler.i(v)

    torch.testing.assert_close(handler.iexp(v), expected_i)
    torch.testing.assert_close(handler.itot(v), expected_i)
    assert handler.update_ion_buf == {"na": False}
    assert torch.count_nonzero(handler.ions["na"].ina) == 0


def test_handler_current_diagnostics_require_explicit_ion_frame_commit():
    shape = (1, 2)
    celsius = torch.full(shape, 34.0, dtype=torch.float64)
    sodium = _SodiumLeak("sodium", celsius, torch.ones(shape), shape, shape).to(
        torch.float64
    )
    ion = Ion("na", shape, einit=0, eadvance=0).to(torch.float64)
    handler = MechanismHandler(
        celsius,
        torch.ones(shape, dtype=torch.float64),
        {"sodium": sodium},
        ions={"na": ion},
        currents={"ina": {"sodium": ["ina"]}},
    ).to(torch.float64)
    handler.make_maps()
    handler.init_i_g_bufs(torch.zeros(shape, dtype=torch.float64))
    v = torch.tensor([[-70.0, -50.0]], dtype=torch.float64)
    expected = 0.01 * (v - 50.0)
    committed = ion.ina.clone()

    current, _ = handler.i(v)
    torch.testing.assert_close(current, expected)
    torch.testing.assert_close(ion.ina, committed)
    torch.testing.assert_close(handler.iexp(v), expected)
    torch.testing.assert_close(ion.ina, committed)
    torch.testing.assert_close(handler.itot(v), expected)
    torch.testing.assert_close(ion.ina, committed)

    handler._publish_ion_current_frame(handler.capture_ion_current_frame())
    torch.testing.assert_close(ion.ina, expected)


def test_handler_empty_current_and_material_lookup_guards():
    shape = (1, 2)
    celsius = torch.full(shape, 34.0)
    area = torch.ones(shape)
    ion = Ion("na", shape, einit=0, eadvance=0)
    material = Material("pool", shape, fields={"amount": 1.0})
    handler = MechanismHandler(
        celsius,
        area,
        {},
        ions={"na": ion},
        materials={"pool": material},
    )
    v = torch.tensor([[-70.0, -60.0]])

    for result in (handler.i(v), handler.idf(v, v)):
        assert isinstance(result, tuple)
        assert all(torch.equal(item, torch.zeros_like(v)) for item in result)
    assert torch.equal(handler.iexp(v), torch.zeros_like(v))
    assert torch.equal(handler.itot(v), torch.zeros_like(v))
    assert handler._get_material("pool") is material
    assert handler._get_material("na") is ion
    with pytest.raises(KeyError, match="Unknown material"):
        handler._get_material("missing")

    with pytest.raises(TypeError, match="must be an instance"):
        MechanismHandler(celsius, area, {"bad": torch.nn.Identity()})


def test_handler_material_write_and_source_use_local_commit_buffers():
    shape = (1, 2)
    celsius = torch.full(shape, 34.0)
    material = Material("pool", shape, fields={"amount": 1.0})
    mech = _MaterialUser("user", celsius, torch.ones(shape), shape, shape)
    mech.register_material(material)

    assert mech.amount.data_ptr() != material.amount.data_ptr()
    mech.amount.add_(10.0)
    torch.testing.assert_close(material.amount, torch.ones(shape))

    handler = MechanismHandler(
        celsius,
        torch.ones(shape),
        {"user": mech},
        materials={"pool": material},
        read_material={"pool": {"user": ["amount"]}},
        write_material={"pool": {"user": ["amount"]}},
        source_material={"pool": {"user": {"amount": "delta"}}},
    )
    mech.amount = torch.tensor([[10.0, 20.0]])
    mech.delta = torch.tensor([[0.5, 1.0]])
    handler.write_to_materials(torch.zeros(shape))
    torch.testing.assert_close(material.amount, torch.tensor([[10.5, 21.0]]))

    material.amount = torch.tensor([[3.0, 4.0]])
    handler.read_from_materials()
    torch.testing.assert_close(mech.amount, material.amount)
    assert mech.amount.data_ptr() != material.amount.data_ptr()
    mech.amount.add_(100.0)
    torch.testing.assert_close(material.amount, torch.tensor([[3.0, 4.0]]))


def test_handler_rejects_zero_area_for_point_process_scaling():
    shape = (1, 2)
    celsius = torch.full(shape, 34.0)
    point = _PointLeak("point", celsius, torch.ones(shape), shape, shape)
    handler = MechanismHandler(
        celsius,
        torch.tensor([[1.0, 0.0]]),
        {"point": point},
        currents={"nonspecific": {"point": ["i"]}},
    )
    with pytest.raises(ValueError, match="area factor is zero"):
        handler.make_maps()


def test_handler_mutable_state_snapshot_is_independent_and_restorable():
    shape = (1, 2)
    celsius = torch.full(shape, 34.0)
    mech = _StatefulBuffer("stateful", celsius, torch.ones(shape), shape, shape)
    mech.x = torch.tensor([[2.0, 3.0]])
    mech.scratch = torch.tensor([[5.0, 7.0]])
    ion = Ion("na", shape, einit=0, eadvance=0)
    ion.nai = torch.tensor([[11.0, 13.0]])
    handler = MechanismHandler(
        celsius, torch.ones(shape), {"stateful": mech}, ions={"na": ion}
    )

    snapshot = handler.mutable_state_dict()
    expected = {
        name: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for name, value in snapshot.items()
    }
    mech.x.add_(100.0)
    mech.scratch.mul_(0.0)
    ion.nai.add_(200.0)

    for name, value in expected.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(snapshot[name], value)
        else:
            assert snapshot[name] == value

    handler.restore_mutable_state_dict(snapshot)
    torch.testing.assert_close(mech.x, expected["stateful.x"])
    torch.testing.assert_close(mech.scratch, expected["stateful.scratch"])
    torch.testing.assert_close(ion.nai, expected["na_ion.nai"])

    mech.x.add_(500.0)
    mech.scratch.add_(600.0)
    ion.nai.add_(700.0)
    for name, value in expected.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(snapshot[name], value)
        else:
            assert snapshot[name] == value


def test_handler_mutable_state_snapshot_preserves_tensor_aliases():
    shape = (1, 2)
    celsius = torch.full(shape, 34.0)
    mech = _StatefulBuffer("stateful", celsius, torch.ones(shape), shape, shape)
    shared = torch.tensor([[2.0, 3.0]], requires_grad=True)
    mech.x = shared
    mech.scratch = shared
    handler = MechanismHandler(celsius, torch.ones(shape), {"stateful": mech})

    snapshot = handler.mutable_state_dict()

    assert snapshot["stateful.x"] is snapshot["stateful.scratch"]
    assert snapshot["stateful.x"] is not shared
    assert snapshot["stateful.x"].grad_fn is not None

    handler.restore_mutable_state_dict(snapshot)
    assert mech.x is mech.scratch
    assert mech.x is not snapshot["stateful.x"]
    assert mech.x.grad_fn is not None


@pytest.mark.parametrize("mode", ["shift", "circular"])
def test_handler_checkpoint_round_trip_includes_delayed_state(mode):
    mech = _registered_single(2, mode)
    handler = MechanismHandler(
        torch.full((2,), 34.0, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        {"base": mech},
    )
    context = torch.no_grad() if mode == "circular" else torch.enable_grad()
    with context:
        mech.delayed_state(
            "signal", torch.tensor([1.0, 2.0], dtype=torch.float64), mode=mode
        )
    snapshot = handler.mutable_state_dict()
    expected_buffer = snapshot["base.signal_delay_buffer"].clone()
    expected_pointer = snapshot["base.signal_delay_ptr"].clone()
    with context:
        mech.delayed_state(
            "signal", torch.tensor([3.0, 4.0], dtype=torch.float64), mode=mode
        )

    handler.restore_mutable_state_dict(snapshot)

    torch.testing.assert_close(mech.signal_delay_buffer, expected_buffer)
    torch.testing.assert_close(mech.signal_delay_ptr, expected_pointer)
    assert mech.signal_delay_buffer is not snapshot["base.signal_delay_buffer"]
    mech.signal_delay_buffer.add_(100.0)
    torch.testing.assert_close(snapshot["base.signal_delay_buffer"], expected_buffer)


def test_handler_checkpoint_restores_delay_metadata_after_runtime_resize():
    mech = _registered_single(1, "shift")
    handler = MechanismHandler(
        torch.full((2,), 34.0, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        {"base": mech},
    )
    first = torch.tensor([1.0, 2.0], dtype=torch.float64)
    second = torch.tensor([3.0, 4.0], dtype=torch.float64)
    mech.delayed_state("signal", first, mode="shift")
    snapshot = handler.mutable_state_dict()

    mech.delayed_state("signal", second, delay_steps=3, mode="shift")
    assert mech._delayed_state_specs["signal"]["steps"] == 3
    assert mech.signal_delay_buffer.shape == (2, 4)

    handler.restore_mutable_state_dict(snapshot)

    spec = mech._delayed_state_specs["signal"]
    assert spec["steps"] == 1
    assert spec["depth"] == 2
    assert mech.signal_delay_buffer.shape == (2, 2)
    # The restored one-step queue must deliver its saved history rather than
    # being cleared by a stale three-step specification on the next call.
    torch.testing.assert_close(
        mech.delayed_state("signal", second, mode="shift"), first
    )


def test_handler_checkpoint_round_trip_includes_local_material_buffers():
    shape = (1, 2)
    celsius = torch.full(shape, 34.0)
    material = Material("pool", shape, fields={"amount": 1.0})
    mech = _MaterialUser("user", celsius, torch.ones(shape), shape, shape)
    mech.register_material(material)
    handler = MechanismHandler(
        celsius,
        torch.ones(shape),
        {"user": mech},
        materials={"pool": material},
        read_material={"pool": {"user": ["amount"]}},
        write_material={"pool": {"user": ["amount"]}},
        source_material={"pool": {"user": {"amount": "delta"}}},
    )
    mech.amount = torch.tensor([[2.0, 3.0]])
    mech.delta = torch.tensor([[0.25, 0.5]])
    material.amount = torch.tensor([[2.0, 3.0]])
    snapshot = handler.mutable_state_dict()
    mech.amount = torch.tensor([[20.0, 30.0]])
    mech.delta = torch.tensor([[5.0, 6.0]])
    material.amount = torch.tensor([[40.0, 50.0]])

    handler.restore_mutable_state_dict(snapshot)

    torch.testing.assert_close(mech.amount, torch.tensor([[2.0, 3.0]]))
    torch.testing.assert_close(mech.delta, torch.tensor([[0.25, 0.5]]))
    torch.testing.assert_close(material.amount, torch.tensor([[2.0, 3.0]]))
    handler.write_to_materials(torch.zeros(shape))
    torch.testing.assert_close(material.amount, torch.tensor([[2.25, 3.5]]))


def test_handler_checkpoint_round_trip_includes_written_ion_concentrations():
    shape = (1, 2)
    celsius = torch.full(shape, 34.0)
    ion = Ion("na", shape, einit=0, eadvance=0)
    mech = _SodiumWriter("writer", celsius, torch.ones(shape), shape, shape)
    mech.register_ion(ion)
    handler = MechanismHandler(
        celsius,
        torch.ones(shape),
        {"writer": mech},
        ions={"na": ion},
        write_ion_c={"na": {"writer": ["nai"]}},
    )
    mech.nai = torch.tensor([[2.0, 3.0]])
    ion.nai = torch.tensor([[2.0, 3.0]])
    snapshot = handler.mutable_state_dict()
    mech.nai = torch.tensor([[20.0, 30.0]])
    ion.nai = torch.tensor([[40.0, 50.0]])

    handler.restore_mutable_state_dict(snapshot)

    torch.testing.assert_close(mech.nai, torch.tensor([[2.0, 3.0]]))
    torch.testing.assert_close(ion.nai, torch.tensor([[2.0, 3.0]]))
    handler.write_to_ions(torch.zeros(shape))
    torch.testing.assert_close(ion.nai, torch.tensor([[2.0, 3.0]]))
