"""Packet shorthand preserves cold source selection and legacy binding ingress."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Mapping

import pytest
import torch

import dendra as dn
from tests.func.test_bound_continuation import _case
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_lowering_api import _target
from tests.func.test_network_mapping_boundaries import _OnItems
from tests.func.test_packet_binding import _KINDS
from tests.func.test_packet_binding import _environment as _environment
from tests.func.test_packet_binding import _factory

pytestmark = pytest.mark.cpu


class _Unused(Mapping):
    def __getitem__(self, key):
        raise AssertionError("binding must not read unused packet fields")

    def __iter__(self):
        raise AssertionError("binding must not traverse unused packet fields")

    def __len__(self):
        raise AssertionError("binding must not inspect unused packet fields")

    def items(self):
        raise AssertionError("binding must not normalize unused packet fields")


class _Marker:
    pass


def _failure(call):
    try:
        call()
    except Exception as error:
        return type(error), str(error)
    raise AssertionError("invalid binding request must fail")


@pytest.mark.parametrize("network", (False, True))
def test_binding_ignores_cyclic_and_opaque_carry_and_does_not_retain_source_packet(
    network,
):
    functional, tensors, parameters, constants, _, _ = _case(network)
    factories = [_factory(functional, network, kind) for kind in _KINDS]
    bindings = []
    for continuation in (False, True):
        marker, unused = _Marker(), _Unused()
        carry = {"marker": marker}
        carry["cycle"] = carry
        if continuation:
            packet = dn.func.Continuation(tensors.state, parameters, constants)
            object.__setattr__(packet, "state", carry)
            object.__setattr__(packet, "callback_state", unused)
            packet_reference = weakref.ref(packet)
        else:
            packet = tensors._replace(
                parameters=parameters, constants=constants, state=carry
            )
            if not network:
                packet = packet._replace(initialization=unused)
            packet_reference = None  # NamedTuple bundles do not support weakrefs.
        marker_reference, unused_reference = weakref.ref(marker), weakref.ref(unused)
        for bind in factories:
            bindings.append(bind(packet))
        del packet, carry, marker, unused
        gc.collect()
        assert marker_reference() is None
        assert unused_reference() is None
        if packet_reference is not None:
            assert packet_reference() is None
    assert all(
        _target(bound.parameters, network) is _target(parameters, network)
        for bound in bindings
    )


@pytest.mark.parametrize("network", (False, True))
def test_packet_binding_performs_no_initialization_preparation_sampling_or_native_extraction(
    network, monkeypatch
):
    functional, tensors, parameters, constants, _, _ = _case(network)
    packet = tensors._replace(parameters=parameters, constants=constants)
    factories = [_factory(functional, network, kind) for kind in _KINDS]
    before = _snapshot((tensors.state, parameters, constants))
    rng = torch.get_rng_state().clone()

    def reject(*args, **kwargs):
        raise AssertionError("binding source packets must not run model machinery")

    with monkeypatch.context() as guarded:
        plans = (
            (functional, *functional.populations.values()) if network else (functional,)
        )
        for plan in plans:
            for name in (
                "extract",
                "_extract",
                "_extract_native_state",
                "prepare",
                "_prepare_values",
                "_prepare_source_validated",
                "initialize",
                "initialize_host",
                "_initialize_host",
                "sample_runtime_draws",
                "sample_initialization_draw",
            ):
                if hasattr(plan, name):
                    guarded.setattr(plan, name, reject)
        bindings = [bind(packet) for bind in factories]
    assert all(
        _target(bound.parameters, network) is _target(parameters, network)
        for bound in bindings
    )
    _close((tensors.state, parameters, constants), before)
    torch.testing.assert_close(torch.get_rng_state(), rng)


@pytest.mark.parametrize("network", (False, True))
def test_packet_conflicts_reject_before_source_fields_are_read(network, monkeypatch):
    functional, tensors, parameters, constants, _, _ = _case(network)
    factories = [_factory(functional, network, kind) for kind in _KINDS]
    packets = (
        tensors._replace(parameters=parameters, constants=constants),
        dn.func.Continuation(tensors.state, parameters, constants),
    )

    def reject(*args):
        raise AssertionError("conflicting packet operands must reject before fields")

    for packet in packets:
        with monkeypatch.context() as guarded:
            guarded.setattr(type(packet), "parameters", property(reject), raising=False)
            guarded.setattr(type(packet), "constants", property(reject), raising=False)
            for bind in factories:
                for value in (None, {}):
                    with pytest.raises(
                        TypeError,
                        match="packet binding cannot be combined with prepared or constants",
                    ):
                        bind(packet, constants=value)
                    if not network:
                        with pytest.raises(
                            TypeError,
                            match="packet binding cannot be combined with prepared or constants",
                        ):
                            bind(packet, value)
                        with pytest.raises(
                            TypeError,
                            match="packet binding cannot be combined with prepared or constants",
                        ):
                            bind(packet, prepared=value)


@pytest.mark.parametrize("network", (False, True))
def test_packet_family_and_subclasses_reject_before_authored_properties(network):
    functional, tensors, parameters, constants, _, _ = _case(network)
    _, other, _, _, _, _ = _case(not network)
    factories = [_factory(functional, network, kind) for kind in _KINDS]

    class AuthoredBundle(type(tensors)):
        @property
        def parameters(self):
            raise AssertionError("packet subclass source fields must not be read")

        @property
        def constants(self):
            raise AssertionError("packet subclass source fields must not be read")

    class AuthoredContinuation(dn.func.Continuation):
        @property
        def parameters(self):
            raise AssertionError("Continuation subclass source fields must not be read")

        @property
        def constants(self):
            raise AssertionError("Continuation subclass source fields must not be read")

    message = (
        "Network binding requires an exact NetworkTensors or Continuation"
        if network
        else "Population binding requires an exact PopulationTensors or Continuation"
    )
    for bind in factories:
        for invalid in (
            other,
            AuthoredBundle(*tensors),
            object.__new__(AuthoredContinuation),
        ):
            with pytest.raises(TypeError, match=message):
                bind(invalid)
        malformed = dn.func.Continuation(tensors.state, {}, constants)
        assert _failure(lambda: bind(malformed)) == _failure(
            lambda: bind({}, constants=constants)
        )


class _AuthoredClass(Mapping):
    def __init__(self, values):
        self.values = values

    @property
    def __class__(self):
        raise AssertionError("legacy Mapping traversal reached authored __class__")

    def __getitem__(self, key):
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


@pytest.mark.parametrize("network", (False, True))
def test_packet_detection_preserves_legacy_errors_before_authored_mapping_traversal(
    network,
):
    functional, _, parameters, constants, _, _ = _case(network)
    for kind in _KINDS:
        bind = _factory(functional, network, kind)
        authored = _AuthoredClass(parameters)
        message = (
            "missing 1 required positional argument: 'constants'"
            if network
            else "bind requires prepared or constants"
        )
        with pytest.raises(TypeError, match=message):
            bind(authored)
        if not network:
            with pytest.raises(
                TypeError, match="prepared and constants are mutually exclusive"
            ):
                bind(authored, None, constants=None)
        # Valid raw ingress still uses the existing Mapping normalization,
        # whose isinstance checks can invoke an authored __class__ getter.
        with pytest.raises(
            AssertionError, match="legacy Mapping traversal reached authored __class__"
        ):
            bind(authored, constants=_AuthoredClass(constants))


def test_network_compiler_boundary_rejects_valid_packets_before_source_fields(
    monkeypatch,
):
    functional, tensors, _, _, _, _ = _case(True)
    factories = [_factory(functional, True, kind) for kind in _KINDS]

    def reject(*args):
        raise AssertionError("Network cold rejection must precede packet fields")

    monkeypatch.setattr(type(tensors), "parameters", property(reject))
    monkeypatch.setattr(type(tensors), "constants", property(reject))
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    for bind in factories:
        with pytest.raises(
            dn.func.FunctionalizationError, match="BoundNetwork is an eager binding"
        ):
            bind(tensors)


@pytest.mark.parametrize("network", (False, True))
def test_existing_positional_bindings_keep_prepared_and_raw_meaning(network):
    functional, tensors, parameters, constants, _, _ = _case(network)
    prepared = None if network else functional.prepare(parameters, constants)
    for kind in _KINDS:
        bind = _factory(functional, network, kind)
        bound = bind(parameters, constants if network else prepared)
        if not network:
            assert bound.prepared is prepared and bound.constants is None
            with pytest.raises(TypeError, match="mutually exclusive"):
                bind(parameters, prepared, constants=constants)
            with pytest.raises(TypeError, match="bind requires prepared or constants"):
                bind(parameters)
        actual, _ = dn.func.run(bound, tensors.state, steps=bound.steps)
        expected, _ = functional.prepare_and_rollout(
            parameters, constants, tensors.state, steps=bound.steps
        )
        _close(actual, expected)


def test_network_raw_mapping_missing_constants_keeps_entry_specific_diagnostics():
    functional, _, parameters, _, _, _ = _case(True)
    names = {
        "plan": "FunctionalNetwork.bind",
        "kernel": "CompiledNetworkChunk.bind",
        "eager_constructor": "BoundNetwork.__init__",
        "compiled_constructor": "BoundNetwork.__init__",
    }
    for kind, name in names.items():
        bind = _factory(functional, True, kind)
        error, message = _failure(lambda: bind(parameters))
        assert error is TypeError
        assert (
            message == f"{name}() missing 1 required positional argument: 'constants'"
        )


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("field", ("parameters", "constants"))
def test_authored_packet_source_traversal_finishes_before_source_validation(
    network, field
):
    functional, tensors, parameters, constants, _, _ = _case(network)
    source = functional._source if network else functional._population
    packet = tensors._replace(parameters=parameters, constants=constants)
    wrapped = _OnItems(
        getattr(packet, field), lambda: setattr(source, "_step", object())
    )
    packet = packet._replace(**{field: wrapped})
    with pytest.raises(dn.func.FunctionalizationError):
        functional.bind(packet)
    assert wrapped.accessed
