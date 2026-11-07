"""One initialization draw packet preserves the existing ordered lifecycle."""

from __future__ import annotations

import os

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_continuation import _assign, _close, _plain, _snapshot
from tests.func.test_explicit_initialization import _case, _defaults, _frames
from tests.func.test_lowering_api import DT, G, _network, _population, _target, _voltage
from tests.func.test_runner_initialization import _inputs, _program, _reference
from tests.func.test_stimulation_rng_sharing import DT as SHARED_DT
from tests.func.test_stimulation_rng_sharing import _network as _random_network
from tests.func.test_stimulation_rng_sharing import _population as _random_population

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0,
                REQUIRE_GRAD=1,
                CHECK_PURITY=False,
                DTYPE=torch.float64,
                DEVICE="cpu",
                IMEM=False,
            ),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


class _CountedScale(torch.nn.Module):
    calls = []

    def forward(self, value, factor):
        # Invocation instrumentation is intentional with CHECK_PURITY disabled.
        _CountedScale.calls.append(torch.is_grad_enabled())
        return (value * factor,)


def _counted_case(network):
    model = _network() if network else _population()
    owner = model.left if network else model
    owner.register_pre_initialize_transform(
        "counted",
        _CountedScale(),
        reads=(f"parameters.{G}",),
        writes=(f"parameters.{G}",),
        inputs={"factor": owner.v.new_tensor(1.25)},
    )
    functional, tensors = dn.func.make_functional(model, dt=DT)
    _CountedScale.calls.clear()
    return functional, tensors


def _pure(functional, tensors, inputs, draws, network):
    if network:
        return functional.initialize(
            tensors.parameters, tensors.constants, tensors.state, inputs, draws=draws
        )
    return functional.initialize(
        tensors.parameters, tensors.constants, inputs, state=tensors.state, draws=draws
    )


def test_initialization_packet_is_separate_from_runtime_draws_and_old_layouts():
    assert dn.func.InitializationDraws._fields == (
        "random",
        "stimulation",
        "netstim",
        "connections",
    )
    assert dn.func.InitializationDraws() == (None, None, None, None)
    assert dn.func.RuntimeDraws._fields == ("draws", "rng")
    assert dn.func.StepInput._fields == ("ve", "intra", "noise", "stimulation")
    assert dn.func.RolloutInput._fields == ("ve", "intra", "noise", "stimulation")


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("runner", (False, True), ids=("raw", "runner"))
def test_sample_and_replay_match_automatic_initialization_without_advancing_sources(
    network,
    runner,
):
    if network:
        source, pulses = _random_network()
    else:
        source, _, pulse = _random_population()
        pulses = [pulse]
    if runner:
        target = dn.compile(
            source,
            dt=SHARED_DT,
            steps=2,
            backend="eager",
            disable=True,
        )
        functional, arguments = target.functional, ()
        originals = (
            target.parameters,
            target.constants,
            target.initial_state,
            target.initialization_inputs,
        )
        report = target.execution_report()
        prepared = None if network else target._prepared
    else:
        target, tensors = dn.func.make_functional(source, dt=SHARED_DT)
        functional, arguments = target, (tensors,)
        originals = tensors
    before = _snapshot(originals)
    native = _snapshot(functional.extract())
    source_rng = pulses[0].generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    with torch.inference_mode():
        packet = dn.func.sample_initialization_draws(target, *arguments)
    assert type(packet) is dn.func.InitializationDraws
    assert packet.random is not None and packet.stimulation is not None
    for value in pytree.tree_leaves(packet):
        if torch.is_tensor(value):
            assert not value.requires_grad and value.grad_fn is None
            assert not value.is_inference()
    frozen = _snapshot(packet)
    _close(dn.func.sample_initialization_draws(target, *arguments), packet)
    automatic = dn.func.initialize(target, *arguments)
    replayed = dn.func.initialize(target, *arguments, draws=packet)
    assert type(replayed) is dn.func.Continuation and replayed.callback_state is None
    _close(_frames(replayed), _frames(automatic))
    if not runner:
        direct = _pure(
            target, tensors, _defaults(target, tensors, network), packet, network
        )
        _close(_frames(direct), _frames(replayed))
    else:
        assert target.execution_report() is report
        if not network:
            assert target._prepared is prepared
    _close(packet, frozen)
    _close(originals, before)
    _close(functional.extract(), native)
    _close(pulses[0].generator.get_state(), source_rng)
    _close(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_sampling_and_replay_are_distinct_single_initialization_executions(network):
    functional, tensors = _counted_case(network)
    packet = dn.func.sample_initialization_draws(functional, tensors)
    assert _CountedScale.calls == [False]
    _CountedScale.calls.clear()
    replayed = dn.func.initialize(functional, tensors, draws=packet)
    assert _CountedScale.calls == [True]
    _CountedScale.calls.clear()
    automatic = dn.func.initialize(functional, tensors)
    assert _CountedScale.calls == [True]
    _close(_frames(replayed), _frames(automatic))


def test_capture_retains_supplied_legacy_draw_tensor_identity_and_gradient_flags():
    model, _, _ = _random_population()
    functional, tensors = dn.func.make_functional(model, dt=SHARED_DT)
    supplied = dn.func.sample_initialization_draws(functional, tensors)
    base = next(
        value
        for value in pytree.tree_leaves(supplied)
        if torch.is_tensor(value) and torch.is_floating_point(value) and value.numel()
    )
    base.requires_grad_()
    initialization = tensors.initialization._replace(
        random_draw=supplied.random,
        stimulation=supplied.stimulation,
    )
    captured = dn.func.sample_initialization_draws(
        functional,
        tensors,
        initialization=initialization,
    )
    expected, spec = pytree.tree_flatten(supplied)
    actual, actual_spec = pytree.tree_flatten(captured)
    assert actual_spec == spec
    for value, previous in zip(actual, expected, strict=True):
        if torch.is_tensor(previous):
            assert value is previous and value.requires_grad == previous.requires_grad
    assert any(value is base for value in actual)
    assert base.requires_grad and base.grad is None


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_runner_overrides_preserve_aliases_gradients_and_checkpoint_continuation(
    network,
    tmp_path,
):
    program = _program(
        network,
        scaled=True,
        inference=False,
        checkpoint_every=2,
        callbacks={
            "trace": dn.Recorder(("left.v", "right.v", "t") if network else ("v", "t"))
        },
        backend="aot_eager",
        disable=False,
    )
    parameters, constants = _plain(program.parameters), _plain(program.constants)
    upstream = _target(parameters, network).detach().clone().requires_grad_()
    _assign(parameters, network, upstream * 1.2)
    factor = upstream.new_tensor(1.7, requires_grad=True)
    initialization = _inputs(program, network, factor=factor)
    before = _snapshot((parameters, constants, initialization, program.initial_state))
    packet = dn.func.sample_initialization_draws(
        program,
        parameters=parameters,
        constants=constants,
        initialization=initialization,
    )
    actual = dn.func.initialize(
        program,
        parameters=parameters,
        constants=constants,
        initialization=initialization,
        draws=packet,
    )
    expected = _reference(
        program,
        parameters=parameters,
        constants=constants,
        initialization=initialization,
    )
    _close(_frames(actual), _frames(expected))
    effective = _target(actual.parameters, network)
    assert effective.grad_fn is not None
    torch.testing.assert_close(effective, 1.2 * upstream * factor)
    if network:
        assert effective is actual.parameters["populations"]["right"][G]
    _, first = program(actual, steps=2)
    final, last = program(first["continuation"], steps=3)
    reference, whole = program(expected, steps=5)
    observed = torch.autograd.grad(
        _voltage(final, network).square().sum(),
        (upstream, factor),
        retain_graph=True,
    )
    wanted = torch.autograd.grad(
        _voltage(reference, network).square().sum(),
        (upstream, factor),
    )
    _close(observed, wanted)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed
    )
    _close(final, reference)
    _close((parameters, constants, initialization, program.initial_state), before)
    path = tmp_path / "continuation.pt"
    torch.save(program.export_continuation(first["continuation"]), path)
    restored = program.restore_continuation(torch.load(path, weights_only=True))
    with torch.no_grad():
        resumed, suffix = program(restored, steps=3)
    _close(resumed, final)
    _close(suffix["callbacks"].outputs, last["callbacks"].outputs)
    key = "left.v" if network else "v"
    _close(
        torch.cat((first["callbacks"]["trace"][key], last["callbacks"]["trace"][key])),
        whole["callbacks"]["trace"][key],
    )


@pytest.mark.parametrize(
    "network,legacy",
    (
        (False, "random"),
        (False, "stimulation"),
        (True, "random"),
        (True, "stimulation"),
        (True, "netstim"),
        (True, "connections"),
    ),
)
def test_new_and_legacy_draws_conflict_before_transforms_or_sampling(network, legacy):
    functional, tensors = _counted_case(network)
    inputs = _defaults(functional, tensors, network)
    field = {"random": "random_draw", "connections": "connection_draw"}.get(
        legacy, legacy
    )
    if legacy == "netstim":
        field = "netstim_draw"
    if network and legacy in ("random", "stimulation"):
        inputs = inputs._replace(
            populations={
                **inputs.populations,
                "left": inputs.populations["left"]._replace(**{field: object()}),
            }
        )
    else:
        inputs = inputs._replace(**{field: object()})
    global_rng = torch.get_rng_state().clone()
    for execute in (
        lambda: dn.func.initialize(
            functional,
            tensors,
            initialization=inputs,
            draws=dn.func.InitializationDraws(),
        ),
        lambda: _pure(
            functional, tensors, inputs, dn.func.InitializationDraws(), network
        ),
    ):
        with pytest.raises(ValueError, match="draw|legacy|once|combined"):
            execute()
        assert _CountedScale.calls == []
        _close(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("kind", ("mapping", "runtime", "family"))
def test_malformed_packets_fail_before_initialization_hooks(network, kind):
    functional, tensors = _counted_case(network)
    packet = {
        "mapping": {},
        "runtime": dn.func.RuntimeDraws({}, {}),
        "family": dn.func.InitializationDraws(random=object()),
    }[kind]
    with pytest.raises((TypeError, ValueError), match="draw|Initialization|random"):
        dn.func.initialize(functional, tensors, draws=packet)
    assert _CountedScale.calls == []


def test_sampler_target_and_override_validation_matches_initializer_surface():
    functional, tensors = _case(False)
    program = _program(False)
    calls = (
        lambda: dn.func.sample_initialization_draws(functional),
        lambda: dn.func.sample_initialization_draws(program, tensors),
        lambda: dn.func.sample_initialization_draws(functional, tensors, parameters={}),
        lambda: dn.func.sample_initialization_draws(functional, tensors, constants={}),
        lambda: dn.func.sample_initialization_draws(
            functional, tensors, draws=dn.func.InitializationDraws()
        ),
    )
    for call in calls:
        with pytest.raises(TypeError):
            call()
    with pytest.raises((TypeError, dn.func.FunctionalizationError)):
        dn.func.sample_initialization_draws(functional.bind(tensors), tensors)
    initial = dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants)
    with pytest.raises(TypeError, match="RuntimeDraws"):
        dn.func.run(functional, initial, steps=0, draws=dn.func.InitializationDraws())


@pytest.mark.mps
def test_mps_initialization_draw_replay_preserves_gradients_and_while_continuation():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DTYPE=torch.float32, DEVICE="mps"):
        program = dn.compile(
            _population(),
            dt=DT,
            steps=4,
            chunk_steps=4,
            execution="while",
            callbacks={"trace": dn.Recorder(("v", "t"))},
            backend="aot_eager",
        )
        voltage = program.initialization_inputs.v_init.detach().clone().requires_grad_()
        inputs = _inputs(program, False, voltage=voltage)
        draws = dn.func.sample_initialization_draws(program, initialization=inputs)
        actual = dn.func.initialize(program, initialization=inputs, draws=draws)
        expected = _reference(program, initialization=inputs)
        _close(_frames(actual), _frames(expected))
        assert actual.state["integrator"]["v"].device.type == "mps"
        observed = torch.autograd.grad(actual.state["integrator"]["v"].sum(), voltage)[
            0
        ]
        wanted = torch.autograd.grad(expected.state["integrator"]["v"].sum(), voltage)[
            0
        ]
        torch.testing.assert_close(observed, wanted)
        _, first = program(actual)
        final, last = program(first["continuation"])
        reference, whole = program(expected, steps=8)
        _close(final, reference)
        _close(
            torch.cat(
                (first["callbacks"]["trace"]["v"], last["callbacks"]["trace"]["v"])
            ),
            whole["callbacks"]["trace"]["v"],
        )
