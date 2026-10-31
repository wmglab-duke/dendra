"""One callback factory preserves native and explicit-plan ownership."""

from __future__ import annotations

import os

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func import _lowering
from dendra.models._recorder_storage import _RecorderHDF5Session
from dendra.models.mod import pas
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_continuation_checkpoints import _disk
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _module_snapshot,
)
from tests.func.test_native_callback_storage import _execute, _join
from tests.func.test_native_network_checkpoint_callbacks import _NetworkEnergy
from tests.func.test_native_population_functional_callbacks import (
    _PortableEnergyReducer,
)
from tests.func.test_network_population_callbacks import CONDUCTANCE, DT, _network
from tests.func.test_stimulation_rng_sharing import DT as STOCHASTIC_DT
from tests.func.test_stimulation_rng_sharing import _network as _stochastic_network
from tests.func.test_stimulation_rng_sharing import (
    _population as _stochastic_population,
)
from tests.test_recorder_h5_integration import local_hdf5_writer as local_hdf5_writer

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64, IMEM=True),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _population(kind=dn.SingleCompartment, *, initialize=True):
    model = kind(N=1, C=3, v_init=-63, integrator=dn.bwd_euler_sc(imem=True))
    model.insert(pas, g=0.001, e=-60)
    model.train()
    if initialize:
        model.initialize()
    else:
        model.build()
    return model


def _case(network):
    return _network() if network else _population()


def _trace(network):
    return ("left.v", "right.v", "t") if network else ("v", "t")


def _specifications(model, field):
    network = isinstance(model, dn.Network)
    lfp = dn.callbacks.LFP([field, field])
    return {
        "trace": dn.Recorder(_trace(network), dt=2 * DT),
        "lfp": dn.callbacks.PopulationCallback("left", lfp) if network else lfp,
        "energy": _NetworkEnergy() if network else _PortableEnergyReducer(),
    }


def _initial(tensors):
    return dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants)


def _objective(outputs, network):
    return (
        outputs["energy"]["energy"].sum()
        + outputs["trace"][_trace(network)[0]].square().mean()
        + (outputs["lfp"] * 1e6).square().mean()
    )


def _raw(model):
    return (
        model.left.mech.pas.g_param
        if isinstance(model, dn.Network)
        else model.mech.pas.g_param
    )


def _failure(call):
    try:
        call()
    except Exception as error:
        return type(error), str(error)
    raise AssertionError("invalid callback preparation must fail")


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_factory_callbacks_match_legacy_and_pure_execution_across_native_methods(
    network, monkeypatch
):
    model = _case(network)
    population = model.left if network else model
    field = torch.ones_like(population.v, requires_grad=True)
    specifications = _specifications(model, field)
    calls = []
    name = "_lower_network" if network else "make_functional"
    lower = getattr(_lowering, name)

    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return lower(*args, **kwargs)

    monkeypatch.setattr(_lowering, name, counted)
    callbacks = dn.func.make_callbacks(
        model, specifications, **({} if network else {"dt": DT})
    )
    assert len(calls) == 1
    functional = callbacks._functional
    source = functional._source if network else functional._population
    assert source is model
    tensors = functional.extract()
    prepared = dn.func.make_callbacks(functional, specifications, dt=DT)
    legacy = functional.make_callbacks(specifications)
    assert prepared._functional is legacy._functional is functional
    assert len(calls) == 1
    _, expected = dn.func.run(
        functional, _initial(tensors), steps=5, callbacks=prepared
    )
    _, compatible = dn.func.run(
        functional, _initial(tensors), steps=5, callbacks=legacy
    )
    _close(expected["callbacks"].outputs, compatible["callbacks"].outputs)
    parts = []
    for method, steps in (
        ("step", 1),
        ("run", 1),
        ("longrun_checkpointed", 2),
        ("longrun", 1),
    ):
        parts.append(
            _execute(
                model,
                method,
                callbacks,
                state=None if not parts else parts[-1].state,
                steps=steps,
            )
        )
    outputs = {
        "trace": _join(parts, "trace"),
        "lfp": torch.cat(tuple(part["lfp"] for part in parts)),
        "energy": parts[-1]["energy"],
    }
    _close(outputs, expected["callbacks"].outputs)
    raw = (
        tensors.parameters["populations"]["left"][CONDUCTANCE]
        if network
        else tensors.parameters[CONDUCTANCE]
    )
    if network:
        assert raw is tensors.parameters["populations"]["right"][CONDUCTANCE]
        assert raw is not _raw(model)
    else:
        assert raw is _raw(model)
    observed = torch.autograd.grad(_objective(outputs, network), (field, _raw(model)))
    wanted = torch.autograd.grad(
        _objective(expected["callbacks"], network), (field, raw)
    )
    _close(observed, wanted)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed
    )


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("authority", ("instance", "class"))
def test_existing_plan_method_authority_is_preserved_without_relowering(
    network, authority, monkeypatch
):
    functional, _ = dn.func.make_functional(_case(network), dt=DT)
    configurations, sentinel, calls = object(), object(), []

    def forbidden(*args, **kwargs):
        raise AssertionError("an existing plan must not be lowered again")

    def authored(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(_lowering, "make_functional", forbidden)
    monkeypatch.setattr(_lowering, "_lower_network", forbidden)
    if authority == "instance":
        monkeypatch.setattr(functional, "make_callbacks", authored)
    else:
        monkeypatch.setattr(type(functional), "make_callbacks", authored)
    assert dn.func.make_callbacks(functional, configurations) is sentinel
    assert dn.func.make_callbacks(functional, configurations, dt=DT) is sentinel
    expected = (
        (configurations,) if authority == "instance" else (functional, configurations)
    )
    assert calls == [expected, expected]


def test_unsupported_targets_are_rejected_without_attribute_or_timestep_coercion():
    class Unsupported:
        @property
        def __class__(self):
            raise AssertionError(
                "unsupported target class descriptors must not be read"
            )

        @property
        def dt(self):
            raise AssertionError("unsupported target dt must not be read")

        @property
        def functional(self):
            raise AssertionError("unsupported target functional must not be read")

        @property
        def populations(self):
            raise AssertionError("unsupported target populations must not be read")

    class InvalidTime:
        def __float__(self):
            raise AssertionError("unsupported target timestep must not be coerced")

    with pytest.raises(TypeError, match="target|Population|Network"):
        dn.func.make_callbacks(Unsupported(), object(), dt=InvalidTime())


def test_factory_does_not_implicitly_accept_runners_bindings_or_tensor_packets():
    model = _population()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    program = dn.compile(model, dt=DT, steps=4, backend="eager")
    targets = (program, functional.bind(tensors), tensors, _initial(tensors))
    for target in targets:
        with pytest.raises(TypeError, match="target|Population|Network"):
            dn.func.make_callbacks(target, object(), dt=DT)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_timestep_validation_precedes_callback_preparation(network, monkeypatch):
    model = _case(network)
    functional, _ = dn.func.make_functional(model, dt=DT)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("invalid dt must precede callback preparation")

    monkeypatch.setattr(functional, "make_callbacks", forbidden)
    for value, error in (
        (True, TypeError),
        (0, ValueError),
        (-DT, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
    ):
        with pytest.raises(error, match="dt"):
            dn.func.make_callbacks(functional, object(), dt=value)
        if network:
            with pytest.raises(error, match="dt"):
                dn.func.make_callbacks(model, object(), dt=value)
        else:
            # Population preparation retains the existing lowering diagnostic,
            # including its capability-slice error for invalid numeric dt.
            assert _failure(
                lambda: dn.func.make_callbacks(model, object(), dt=value)
            ) == _failure(lambda: dn.func.make_functional(model, dt=value))
    if network:
        with pytest.raises(ValueError, match="dt|timestep"):
            dn.func.make_callbacks(model, object(), dt=2 * DT)
    else:
        with pytest.raises(TypeError, match="dt"):
            dn.func.make_callbacks(model, object())
    with pytest.raises(ValueError, match="dt|timestep"):
        dn.func.make_callbacks(functional, object(), dt=2 * DT)
    assert calls == []


def test_factory_requires_cold_construction_before_reading_timestep_or_callbacks(
    monkeypatch,
):
    functional, _ = dn.func.make_functional(_population(), dt=DT)

    class InvalidTime:
        def __float__(self):
            raise AssertionError(
                "cold-construction validation must precede dt coercion"
            )

    def prepare():
        return dn.func.make_callbacks(functional, object(), dt=InvalidTime())

    with monkeypatch.context() as patch:
        patch.setattr(torch.compiler, "is_compiling", lambda: True)
        with pytest.raises(dn.func.FunctionalizationError, match="outside"):
            prepare()
    with torch.autograd.forward_ad.dual_level():
        with pytest.raises(dn.func.FunctionalizationError, match="outside"):
            prepare()
    with pytest.raises(dn.func.FunctionalizationError, match="outside"):
        torch.func.grad(lambda value: prepare())(torch.ones(()))


def test_factory_keeps_native_initialization_and_subclass_eligibility_contracts():
    class Configured(dn.SingleCompartment):
        pass

    class CustomStep(Configured):
        def step(self, *args, **kwargs):
            return super().step(*args, **kwargs)

    specifications = {"trace": dn.Recorder(("v", "t"))}
    compatible = _population(Configured)
    callbacks = dn.func.make_callbacks(compatible, specifications, dt=DT)
    assert callbacks._functional._population is compatible
    for model in (_population(initialize=False), _population(CustomStep)):
        expected = _failure(lambda: dn.func.make_functional(model, dt=DT))
        actual = _failure(lambda: dn.func.make_callbacks(model, specifications, dt=DT))
        assert actual == expected


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_native_factory_preserves_tensor_identity_versions_history_and_rng(network):
    if network:
        model, pulses = _stochastic_network()
        populations = tuple(model.populations.values())
    else:
        model, _, pulse = _stochastic_population()
        populations, pulses = (model,), (pulse,)
    snapshots = tuple(_module_snapshot(population) for population in populations)
    checkpoint = _snapshot(model.state_dict_for_checkpoint())
    global_rng = torch.get_rng_state().clone()
    streams = tuple(pulse.generator.get_state().clone() for pulse in pulses)
    recorder = dn.Recorder(_trace(network), dt=2 * STOCHASTIC_DT)
    marker = torch.tensor(29.0)
    recorder.rec[_trace(network)[0]].append(marker)
    recorder.i = 37
    history = recorder.rec
    callbacks = dn.func.make_callbacks(model, {"trace": recorder}, dt=STOCHASTIC_DT)
    assert type(callbacks) is dn.func.FunctionalCallbacks
    assert recorder.rec is history and recorder.rec[_trace(network)[0]][0] is marker
    assert recorder.i == 37 and recorder.run_number == recorder.save_count == 0
    for population, snapshot in zip(populations, snapshots, strict=True):
        _assert_module_snapshot(population, snapshot)
    _assert_packet_equal(model.state_dict_for_checkpoint(), checkpoint)
    _close(torch.get_rng_state(), global_rng)
    for pulse, state in zip(pulses, streams, strict=True):
        _close(pulse.generator.get_state(), state)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_new_bindings_keep_live_tokens_independent_and_portable_carry_compatible(
    tmp_path, network
):
    model = _case(network)
    specifications = {"trace": dn.Recorder(_trace(network), dt=2 * DT)}
    first = dn.func.make_callbacks(model, specifications, dt=DT)
    second = dn.func.make_callbacks(model, specifications, dt=DT)
    assert first._functional is not second._functional
    prefix = _execute(model, "run", first, steps=3)
    frozen = _snapshot(model.state_dict_for_checkpoint())
    with pytest.raises(dn.func.FunctionalizationError, match="different callback plan"):
        _execute(model, "longrun", second, state=prefix.state, steps=0)
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
    other = _case(network)
    with pytest.raises(dn.func.FunctionalizationError, match="exact source"):
        _execute(other, "run", first, steps=0)
    packet = _disk(first.export_state(prefix.state), tmp_path / "callbacks.pt")
    restored = second.restore_state(packet)
    suffix = _execute(model, "longrun", second, state=restored, steps=2)
    reference = _case(network)
    functional, _ = dn.func.make_functional(reference, dt=DT)
    expected = _execute(
        reference, "run", functional.make_callbacks(specifications), steps=5
    )
    _close(_join((prefix, suffix), "trace"), expected["trace"])
    legacy = functional.make_callbacks(specifications)
    resumed = legacy.restore_state(
        _disk(second.export_state(suffix.state), tmp_path / "legacy.pt")
    )
    assert resumed["trace"].item() == 5


def test_native_factory_defers_hdf5_storage_sessions_and_preserves_storage_identity(
    tmp_path,
    local_hdf5_writer,  # noqa: F811 - pytest injects the imported fixture
    monkeypatch,
):
    model = _population()
    recorder = dn.Recorder(("v", "t"), dt=2 * DT).set_hdf5(
        str(tmp_path / "deferred.h5"), cache_every=3
    )
    queue, writer, history = recorder.queue, recorder.writer_thread, recorder.rec
    frozen = _snapshot(model.state_dict_for_checkpoint())

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "callback preparation must not start storage or write frames"
        )

    try:
        with monkeypatch.context() as patch:
            patch.setattr(_RecorderHDF5Session, "__init__", forbidden)
            patch.setattr(writer, "start", forbidden)
            patch.setattr(recorder, "_write_hdf5_frame", forbidden)
            callbacks = dn.func.make_callbacks(model, {"disk": recorder}, dt=DT)
        assert callbacks._storage["disk"][0] is recorder
        assert recorder.queue is queue and recorder.writer_thread is writer
        assert (
            recorder._hdf5_session is None
            and recorder.run_number == recorder.save_count == 0
        )
        assert recorder.rec is history and all(
            not values for values in history.values()
        )
        assert local_hdf5_writer.queue.items == []
        _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
    finally:
        recorder.close()


def test_factory_reuses_packed_regions_scoped_lfp_and_shared_source_gradients():
    model = _network(packed=True)
    native_g = model.packed.mech.pas.g_left
    assert native_g is model.packed.mech.pas.g_right
    with torch.no_grad():
        native_g.fill_(0.003)
    model.initialize(DT)
    field = torch.ones_like(model.packed.v, requires_grad=True)
    specifications = {
        "region": dn.callbacks.PopulationCallback(
            "packed",
            dn.Recorder(("v", "i_membrane", "t"), region=model.packed.left.soma),
        ),
        "lfp": dn.callbacks.PopulationCallback(
            "packed", dn.callbacks.LFP([field, field])
        ),
    }
    callbacks = dn.func.make_callbacks(model, specifications)
    functional = callbacks._functional
    tensors = functional.extract()
    legacy = functional.make_callbacks(specifications)
    _, expected = dn.func.run(functional, _initial(tensors), steps=3, callbacks=legacy)
    actual = _execute(model, "run", callbacks, steps=3)
    _close(actual.outputs, expected["callbacks"].outputs)
    assert actual["region"]["v"].shape == (
        4,
        *model.packed.v[model.packed.left.soma.index].shape,
    )
    # The packed mechanism retains an unused declaration-default g_param;
    # the live component bindings preserve the authored shared conductance.
    assert native_g is model.packed.mech.pas.g_left is model.packed.mech.pas.g_right
    parameters = tensors.parameters["populations"]["packed"]
    sources = functional.populations["packed"]._parameter_sources(model.packed)
    aliases = tuple(name for name, value in sources.items() if value is native_g)
    assert aliases
    packet_g = parameters[aliases[0]]
    assert all(parameters[name] is packet_g for name in aliases)
    assert packet_g is not native_g
    observed = torch.autograd.grad(actual["lfp"].square().sum(), (field, native_g))
    wanted = torch.autograd.grad(
        expected["callbacks"]["lfp"].square().sum(), (field, packet_g)
    )
    _close(observed, wanted)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed
    )


@pytest.mark.mps
def test_mps_prepared_callbacks_preserve_native_and_functional_recording_gradients():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DEVICE="mps", DTYPE=torch.float32):
        model = _population()
        specifications = {"trace": dn.Recorder(("v", "t"), dt=2 * DT)}
        callbacks = dn.func.make_callbacks(model, specifications, dt=DT)
        functional = callbacks._functional
        tensors = functional.extract()
        _, expected = dn.func.run(
            functional,
            _initial(tensors),
            steps=3,
            callbacks=dn.func.make_callbacks(functional, specifications),
        )
        actual = _execute(model, "run", callbacks, steps=3)
        _close(actual.outputs, expected["callbacks"].outputs)
        observed = torch.autograd.grad(
            actual["trace"]["v"].square().mean(), _raw(model)
        )[0]
        wanted = torch.autograd.grad(
            expected["callbacks"]["trace"]["v"].square().mean(),
            tensors.parameters[CONDUCTANCE],
        )[0]
        _close(observed, wanted)
        assert observed.device.type == "mps" and observed.abs().sum() > 0
